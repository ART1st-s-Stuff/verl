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

import logging
import os
from functools import partial

import psutil
import torch
from codetiming import Timer

from verl import DataProto
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, make_nd_compute_dataproto_dispatch_fn, register
from verl.utils.device import (
    get_device_id,
    get_device_name,
    get_torch_device,
)
from verl.utils.distributed import initialize_global_process_group_ray
from verl.utils.flops_counter import FlopsCounter
from verl.utils.profiler import DistProfiler, DistProfilerExtension
from verl.utils.py_functional import append_to_dict
from verl.workers.config import ActorConfig
from verl.workers.roles.utils.losses import ppo_loss
from verl.workers.roles.utils.mcts_planner import MCTSPlanner, MCTSPlannerConfig
from verl.workers.roles.utils.padding import left_right_2_no_padding, no_padding_2_padding
from verl.workers.roles.utils.world_model import LatentStateEncoder, TransitionRewardNet

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

device_name = get_device_name()
ACTION_TOKENS = [
    "<|act_moveahead|>",
    "<|act_moveback|>",
    "<|act_moveright|>",
    "<|act_moveleft|>",
    "<|act_rotateright|>",
    "<|act_rotateleft|>",
    "<|act_lookup|>",
    "<|act_lookdown|>",
]


def _can_enable_world_model(config: ActorConfig) -> bool:
    return config.num_actions > 0 and config.world_state_dim > 0


def _can_enable_latent_mcts(config: ActorConfig) -> bool:
    return _can_enable_world_model(config) and config.enable_latent_mcts


class ActorWorker(Worker, DistProfilerExtension):
    """
    This worker can be instantiated as a standalone actor or a standalone rollout or a standalone reference policy
    or a hybrid engine based on the config.rollout
    """

    def __init__(self, config: ActorConfig):
        self.config = config
        Worker.__init__(self)
        self.profiler_config = self.config.profiler
        tool_config = self.profiler_config.tool_config
        DistProfilerExtension.__init__(
            self, DistProfiler(rank=self.rank, config=self.profiler_config, tool_config=tool_config)
        )

        initialize_global_process_group_ray(timeout_second=None)

        self.loss_fn = partial(ppo_loss, config=self.config)

    def _build_engine(self):
        self.model_config = self.config.model_config
        self.engine_config = self.config.engine
        self.optimizer_config = self.config.optim
        self.checkpoint_config = self.config.checkpoint

        from verl.workers.engine import BaseEngine, EngineRegistry

        self.engine: BaseEngine = EngineRegistry.new(
            model_type="language_model",
            backend=self.config.strategy,
            model_config=self.model_config,
            engine_config=self.engine_config,
            optimizer_config=self.optimizer_config,
            checkpoint_config=self.checkpoint_config,
        )

        # build dispatch info
        self._register_dispatch_collect_info(
            mesh_name="actor",
            dp_rank=self.engine.get_data_parallel_rank(),
            is_collect=self.engine.is_mp_src_rank_with_outputs(),
        )

        # aggregate with bon sampling
        self.ppo_mini_batch_size = self.config.ppo_mini_batch_size * self.config.rollout_n
        assert self.ppo_mini_batch_size % self.engine.get_data_parallel_size() == 0, (
            f"{self.ppo_mini_batch_size=} is not divisible by {self.engine.get_data_parallel_size()=}"
        )
        self.ppo_mini_batch_size_per_dp = self.ppo_mini_batch_size // self.engine.get_data_parallel_size()

        # setup flops counter
        self.flops_counter = FlopsCounter(self.model_config.hf_config)

        self.state_encoder = None
        self.transition_reward_net = None
        self.world_model_optimizer = None
        self.mcts_planner = None
        if _can_enable_world_model(self.config):
            hidden_size = self.model_config.hf_config.hidden_size
            transition_hidden_dim = self.config.transition_hidden_dim or hidden_size
            self.state_encoder = LatentStateEncoder(hidden_size, self.config.world_state_dim).to(get_device_id())
            self.transition_reward_net = TransitionRewardNet(
                world_state_dim=self.config.world_state_dim,
                num_actions=self.config.num_actions,
                hidden_dim=transition_hidden_dim,
            ).to(get_device_id())
            world_model_lr = self.config.action_head_lr or self.optimizer_config.lr
            self.world_model_optimizer = torch.optim.AdamW(
                list(self.state_encoder.parameters()) + list(self.transition_reward_net.parameters()),
                lr=world_model_lr,
            )
            self.loss_fn = partial(
                ppo_loss,
                config=self.config,
                state_encoder=self.state_encoder,
                transition_reward_net=self.transition_reward_net,
            )
            self.mcts_planner = MCTSPlanner(
                transition_model=self.transition_reward_net,
                num_actions=self.config.num_actions,
                config=MCTSPlannerConfig(
                    depth=self.config.mcts.depth,
                    branching=self.config.mcts.branching,
                    c_puct=self.config.mcts.c_puct,
                    rollout_steps=self.config.mcts.rollout_steps,
                    discount=self.config.mcts.discount,
                ),
            )

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        self._build_engine()
        self.engine.initialize()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def set_loss_fn(self, loss_fn):
        self.loss_fn = loss_fn

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @DistProfiler.annotate(color="blue", role="actor_compute_log_prob")
    def compute_log_prob(self, data: DataProto):
        extract_latent = data.meta_info.get("extract_latent", False)
        enable_gradient_for_latent = data.meta_info.get("enable_gradient_for_latent", False)
        data.meta_info["use_dynamic_bsz"] = self.config.use_dynamic_bsz
        data.meta_info["use_fused_kernels"] = self.config.use_fused_kernels
        data.meta_info["calculate_entropy"] = True
        if self.config.use_dynamic_bsz:
            data.meta_info["max_token_len_per_gpu"] = self.config.ppo_infer_max_token_len_per_gpu
        else:
            data.meta_info["micro_batch_size_per_gpu"] = self.config.ppo_infer_micro_batch_size_per_gpu

        with self.engine.eval_mode():
            if self.state_encoder is not None:
                self.state_encoder.eval()
            if self.transition_reward_net is not None:
                self.transition_reward_net.eval()
            # TODO: make worker API to accept TensorDict as well
            data = data.to_tensordict()
            data = left_right_2_no_padding(data)
            if enable_gradient_for_latent:
                # infer_batch wraps the forward pass with torch.no_grad().
                # Use forward_backward_batch(forward_only=True) to preserve autograd graph.
                output = self.engine.forward_backward_batch(data, loss_function=None, forward_only=True)
            else:
                output = self.engine.infer_batch(data)

        if self.engine.is_mp_src_rank_with_outputs():
            output = output["model_output"]
            log_probs = output["log_probs"]
            log_probs = no_padding_2_padding(log_probs, data)  # (bsz, response_length)

            entropy = output["entropy"]
            if entropy is not None:
                entropy = no_padding_2_padding(entropy, data)  # (bsz, response_length)

            # in megatron, only last pp contains valid data and returned to the single controller
            output_tensors = {"old_log_probs": log_probs.float()}
            if entropy is not None:
                output_tensors["entropys"] = entropy.float()
            output_non_tensors = {}
            if extract_latent and "latent" in output:
                # Keep only the last token latent for each sample.
                latent = output["latent"]
                latent = latent.values()[latent.offsets()[1:] - 1]
                output_tensors["latent"] = latent
                if self.state_encoder is not None:
                    latent_input = latent.detach() if self.config.action_head_detach_latent else latent
                    world_state = self.state_encoder(latent_input)
                    output_tensors["world_state"] = world_state
                    if _can_enable_latent_mcts(self.config) and self.mcts_planner is not None:
                        action_ids = self.mcts_planner.plan(world_state)
                        output_tensors["planned_action_ids"] = action_ids
                        if self.config.num_actions <= len(ACTION_TOKENS):
                            output_non_tensors["planned_action_tokens"] = [
                                ACTION_TOKENS[int(x)] for x in action_ids.tolist()
                            ]
            output = DataProto.from_dict(
                tensors=output_tensors,
                non_tensors=output_non_tensors,
            )
            if not enable_gradient_for_latent:
                output = output.to("cpu")
        return output

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @DistProfiler.annotate(color="red", role="actor_update")
    def update_actor(self, data: DataProto):
        data.meta_info["use_dynamic_bsz"] = self.config.use_dynamic_bsz
        data.meta_info["use_fused_kernels"] = self.config.use_fused_kernels
        data.meta_info["calculate_entropy"] = self.config.entropy_coeff != 0.0
        data.meta_info["extract_latent"] = self.state_encoder is not None
        data.meta_info["enable_gradient_for_latent"] = self.state_encoder is not None and not self.config.action_head_detach_latent
        if self.config.use_dynamic_bsz:
            data.meta_info["max_token_len_per_gpu"] = self.config.ppo_max_token_len_per_gpu
        else:
            data.meta_info["micro_batch_size_per_gpu"] = self.config.ppo_micro_batch_size_per_gpu

        metrics = {}
        # Support all hardwares
        data = data.to(get_device_id())
        # perform forward computation
        with self.engine.train_mode():
            if self.state_encoder is not None:
                self.state_encoder.train()
            if self.transition_reward_net is not None:
                self.transition_reward_net.train()
            dataloader = data.make_iterator(
                mini_batch_size=self.ppo_mini_batch_size_per_dp,
                epochs=self.config.ppo_epochs,
                seed=self.config.data_loader_seed + self.engine.get_data_parallel_rank(),
                dataloader_kwargs={"shuffle": self.config.shuffle},
            )
            with Timer(name="update_policy", logger=None) as timer:
                for batch_idx, mini_batch in enumerate(dataloader):
                    mini_batch.meta_info["global_batch_size"] = self.config.ppo_mini_batch_size
                    # TODO: make worker API to accept TensorDict as well
                    mini_batch = mini_batch.to_tensordict()
                    mini_batch = left_right_2_no_padding(mini_batch)
                    if self.world_model_optimizer is not None:
                        self.world_model_optimizer.zero_grad(set_to_none=True)
                    output = self.engine.train_batch(mini_batch, self.loss_fn)
                    if self.world_model_optimizer is not None:
                        self.world_model_optimizer.step()
                    mini_batch_metrics = output.get("metrics", {})
                    append_to_dict(metrics, mini_batch_metrics, prefix="actor/")

            delta_time = timer.last

            global_num_tokens = data.meta_info["global_token_num"]
            estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
            metrics["perf/mfu/actor"] = estimated_flops * self.config.ppo_epochs / promised_flops / self.world_size
            metrics["perf/max_memory_allocated_gb"] = get_torch_device().max_memory_allocated() / (1024**3)
            metrics["perf/max_memory_reserved_gb"] = get_torch_device().max_memory_reserved() / (1024**3)
            metrics["perf/cpu_memory_used_gb"] = psutil.virtual_memory().used / (1024**3)

            lr = self.engine.lr_scheduler_step()
            metrics["actor/lr"] = lr

            output = DataProto(batch=None, meta_info={"metrics": metrics})

        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        return self.engine.save_checkpoint(local_path, hdfs_path, global_step, max_ckpt_to_keep)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=False):
        return self.engine.load_checkpoint(local_path, hdfs_path, del_local_after_load)
