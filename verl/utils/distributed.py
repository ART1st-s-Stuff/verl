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
"""Utilities for distributed training."""

import ctypes
import os
import sys
from datetime import timedelta

import ray
import torch.distributed

from verl.utils.device import get_device_name, get_nccl_backend, get_torch_device, is_npu_available


def _read_parent_cmdline() -> str:
    try:
        with open(f"/proc/{os.getppid()}/cmdline", "rb") as f:
            return f.read().replace(b"\x00", b" ").decode("utf-8", errors="ignore").strip()
    except Exception:
        return ""


def _should_fail_fast_interactive_launch() -> tuple[bool, dict]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    parent_cmdline = _read_parent_cmdline()
    stdin_is_tty = bool(getattr(sys.stdin, "isatty", lambda: False)())
    has_dist_init_method = bool(os.environ.get("DIST_INIT_METHOD"))
    has_torchrun_marker = bool(os.environ.get("TORCHELASTIC_RUN_ID"))
    has_master_addr = bool(os.environ.get("MASTER_ADDR"))
    has_master_port = bool(os.environ.get("MASTER_PORT"))
    interactive_shell_parent = any(shell in parent_cmdline for shell in ("bash", "zsh", "fish", "sh "))

    should_fail = (
        world_size > 1
        and interactive_shell_parent
        and stdin_is_tty
        and not has_dist_init_method
        and not has_torchrun_marker
    )
    return should_fail, {
        "world_size": world_size,
        "parent_cmdline": parent_cmdline,
        "stdin_is_tty": stdin_is_tty,
        "has_dist_init_method": has_dist_init_method,
        "has_torchrun_marker": has_torchrun_marker,
        "has_master_addr": has_master_addr,
        "has_master_port": has_master_port,
    }


def set_numa_affinity():
    if is_npu_available:
        # TODO (FightingZhen) libnuma.so is not available in e2e_ascend CI image, remove this code after image update.
        return

    initialized = False
    try:
        libnuma = ctypes.CDLL("libnuma.so")
        if libnuma.numa_available() < 0:
            return

        import pynvml

        pynvml.nvmlInit()
        initialized = True
        device_name = "NPU" if is_npu_available else "GPU"
        local_rank = int(ray.get_runtime_context().get_accelerator_ids()[device_name][0])
        handle = pynvml.nvmlDeviceGetHandleByIndex(local_rank)
        pynvml.nvmlDeviceSetCpuAffinity(handle)
    except ImportError:
        print("Warning: pynvml not available, skipping NUMA affinity setup")
    except Exception as e:
        print(f"Warning: Failed to set NUMA affinity: {e}")
    finally:
        if initialized:
            pynvml.nvmlShutdown()


def initialize_global_process_group(timeout_second=36000):
    should_fail_fast, fail_fast_data = _should_fail_fast_interactive_launch()
    if should_fail_fast:
        raise RuntimeError(
            "Detected an interactive shell launch with WORLD_SIZE > 1. "
            "This entrypoint expects a real distributed launcher and would otherwise hang in "
            "torch.distributed.init_process_group(). "
            "Use torchrun/srun to launch one Python process per rank, or override "
            "WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 for single-process smoke tests."
        )
    torch.distributed.init_process_group(
        get_nccl_backend(),
        timeout=timedelta(seconds=timeout_second),
        init_method=os.environ.get("DIST_INIT_METHOD", None),
    )
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if torch.distributed.is_initialized():
        get_torch_device().set_device(local_rank)
    return local_rank, rank, world_size


def destroy_global_process_group():
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def initialize_global_process_group_ray(timeout_second=None):
    # in current ray environment, LOCAL_RANK is always zero.

    import torch.distributed

    timeout = timedelta(seconds=timeout_second) if timeout_second is not None else None

    if not torch.distributed.is_initialized():
        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        torch.distributed.init_process_group(
            backend=f"cpu:gloo,{get_device_name()}:{get_nccl_backend()}",
            rank=rank,
            world_size=world_size,
            timeout=timeout,
            init_method=os.environ.get("DIST_INIT_METHOD", None),
        )
