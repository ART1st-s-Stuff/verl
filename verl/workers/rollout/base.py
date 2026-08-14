# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib
from abc import ABC, abstractmethod
from typing import Generator

import torch
from torch.distributed.device_mesh import DeviceMesh

from verl import DataProto
from verl.utils.config import omega_conf_to_dataclass
from verl.workers.config import HFModelConfig, RolloutConfig

__all__ = ["BaseRollout", "get_rollout_class", "register_rollout"]


class BaseRollout(ABC):
    """Base class for rollout."""

    def __init__(
        self,
        config: RolloutConfig,
        model_config: HFModelConfig,
        device_mesh: DeviceMesh,
    ):
        self.config = omega_conf_to_dataclass(config)
        self.model_config: HFModelConfig = omega_conf_to_dataclass(model_config, dataclass_type=HFModelConfig)
        self.device_mesh = device_mesh

    @abstractmethod
    async def resume(self, tags: list[str]):
        """Resume rollout weights or kv cache in GPU memory.

        Args:
            tags: weights or kv_cache.
        """
        pass

    @abstractmethod
    async def update_weights(
        self,
        weights: Generator[tuple[str, torch.Tensor], None, None],
        **kwargs,
    ):
        """Update the weights of the rollout model.

        Args:
            weights: A generator that yields the name of the weight tensor and the tensor itself.
        """
        pass

    @abstractmethod
    async def release(self):
        """Release weights and kv cache in GPU memory."""
        pass

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """Batch generate sequences in sync mode.

        Args:
            prompts: The input prompts.

        Returns:
            The output sequences.
        """
        raise NotImplementedError


_ROLLOUT_REGISTRY = {
    ("vllm", "sync"): "verl.workers.rollout.vllm_rollout.vLLMRollout",
    ("vllm", "async"): "verl.workers.rollout.vllm_rollout.vLLMAsyncRollout",
    ("sglang", "sync"): "verl.workers.rollout.sglang_rollout.sglang_rollout.SGLangRollout",
    ("sglang", "async"): "verl.workers.rollout.sglang_rollout.sglang_rollout.ServerAdapter",
}


def register_rollout(rollout_name: str, mode: str, fqdn: str) -> None:
    """Register an external rollout class before worker rollout construction."""

    if not isinstance(rollout_name, str) or not rollout_name:
        raise ValueError("rollout_name must be a non-empty string")
    if mode not in {"sync", "async"}:
        raise ValueError("rollout mode must be sync or async")
    if (
        not isinstance(fqdn, str)
        or "." not in fqdn
        or fqdn.startswith(".")
        or fqdn.endswith(".")
    ):
        raise ValueError("rollout fqdn must contain a module and class name")
    key = (rollout_name, mode)
    existing = _ROLLOUT_REGISTRY.get(key)
    if existing is None:
        _ROLLOUT_REGISTRY[key] = fqdn
        return
    if existing != fqdn:
        raise ValueError(
            f"rollout {rollout_name}/{mode} is already registered as {existing}"
        )


def get_rollout_class(rollout_name: str, mode: str) -> type[BaseRollout]:
    """Get the rollout class by name.

    Args:
        rollout_name: The name of the rollout.
        mode: The mode of the rollout, sync: spmd mode, async: server mode.

    Returns:
        The rollout class.
    """
    assert (rollout_name, mode) in _ROLLOUT_REGISTRY, f"Rollout {rollout_name} with mode {mode} not found"
    fqdn = _ROLLOUT_REGISTRY[(rollout_name, mode)]
    module_name, class_name = fqdn.rsplit(".", 1)
    rollout_module = importlib.import_module(module_name)
    return getattr(rollout_module, class_name)
