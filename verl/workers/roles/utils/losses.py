# Copyright 2025 Bytedance Ltd. and/or its affiliates
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


import torch
import torch.nn.functional as F
from tensordict import TensorDict

from verl.trainer.ppo.core_algos import agg_loss, compute_value_loss, get_policy_loss_fn, kl_penalty
from verl.utils import tensordict_utils as tu
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.utils.torch_functional import masked_mean, masked_sum
from verl.workers.config import ActorConfig, CriticConfig
from verl.workers.roles.utils.padding import no_padding_2_padding
from verl.workers.roles.utils.world_model import cast_tensor_to_module_dtype


def sft_loss(config: ActorConfig, model_output, data: TensorDict, dp_group=None):
    pad_mode = tu.get_non_tensor_data(data=data, key="pad_mode", default=DatasetPadMode.NO_PADDING)
    dp_size = data["dp_size"]
    batch_num_tokens = data["batch_num_tokens"]

    log_prob = model_output["log_probs"]

    if pad_mode == DatasetPadMode.NO_PADDING:
        # log_prob and loss mask are nested tensors of shape [bsz, j1]
        # for each sample, loss mask shape is [1, prompt_length + response_length]
        loss_mask = data["loss_mask"]

        log_prob_flatten = log_prob.values()
        loss_mask_flatten = loss_mask.values()

        # left-shift the loss mask by one token to align with log_prob
        loss_mask_flatten = torch.roll(loss_mask_flatten, shifts=-1, dims=0)

        # NOTE: loss is averaged over all tokens in the batch across all data parallel groups,
        # For FSDP backend, the loss is directly used for backward; while for Megatron backend,
        # the loss should be scaled by `num_microbatches` and `cp_size` for pp schedule.
        loss = -masked_sum(log_prob_flatten, loss_mask_flatten) / batch_num_tokens * dp_size
    else:
        response_mask = data["response_mask"].to(bool)
        loss = -masked_sum(log_prob, response_mask) / batch_num_tokens * dp_size

    return loss, {"loss": loss.detach().item()}


def ppo_loss(config: ActorConfig, model_output, data: TensorDict, dp_group=None, state_encoder=None, transition_reward_net=None):
    log_prob = model_output["log_probs"]
    entropy = model_output.get("entropy", None)

    log_prob = no_padding_2_padding(log_prob, data)  # (bsz, response_length)
    if entropy is not None:
        entropy = no_padding_2_padding(entropy, data)  # (bsz, response_length)

    metrics = {}

    response_mask = data["response_mask"].to(bool)
    # compute policy loss
    old_log_prob = data["old_log_probs"]
    advantages = data["advantages"]

    loss_agg_mode = config.loss_agg_mode

    loss_mode = config.policy_loss.get("loss_mode", "vanilla")

    policy_loss_fn = get_policy_loss_fn(loss_mode)
    pg_loss, pg_metrics = policy_loss_fn(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
        config=config,
    )
    metrics.update(pg_metrics)
    metrics["actor/pg_loss"] = pg_loss.detach().item()
    policy_loss = pg_loss

    # add entropy loss
    if entropy is not None:
        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
        entropy_coeff = config.entropy_coeff
        policy_loss -= entropy_coeff * entropy_loss

    # add kl loss
    if config.use_kl_loss:
        ref_log_prob = data["ref_log_prob"]
        # compute kl loss
        kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=config.kl_loss_type)
        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=config.loss_agg_mode)

        policy_loss += kl_loss * config.kl_loss_coef
        metrics["kl_loss"] = kl_loss.detach().item()
        metrics["kl_coef"] = config.kl_loss_coef

    world_model_enabled = state_encoder is not None and transition_reward_net is not None
    metrics["actor/world_model_enabled"] = 1.0 if world_model_enabled else 0.0

    # optional latent world-model loss branch
    if world_model_enabled and "action_labels" in data and "step_rewards" in data and "latent" in model_output:
        latent = model_output["latent"]
        latent_last = latent.values()[latent.offsets()[1:] - 1]
        if config.action_head_detach_latent:
            latent_last = latent_last.detach()
        action_labels = data["action_labels"].long().reshape(-1)
        step_rewards = data["step_rewards"].float().reshape(-1)
        supervision_mask = torch.ones_like(action_labels, dtype=torch.bool)
        if "action_label_mask" in data:
            supervision_mask = supervision_mask & data["action_label_mask"].to(torch.bool).reshape(-1)
        if not supervision_mask.any():
            metrics["actor/world_model_supervision_missing"] = 1.0
            return policy_loss, metrics

        latent_last = cast_tensor_to_module_dtype(latent_last, state_encoder)
        world_state = state_encoder(latent_last[supervision_mask])
        pred_next_state, pred_reward = transition_reward_net(world_state, action_labels[supervision_mask])

        reward_target = step_rewards[supervision_mask].reshape_as(pred_reward)
        reward_loss = F.mse_loss(pred_reward, reward_target)
        policy_loss = policy_loss + config.reward_loss_coef * reward_loss
        metrics["actor/reward_loss"] = reward_loss.detach().item()
        metrics["actor/world_model_loss"] = (config.reward_loss_coef * reward_loss.detach()).item()

        if "next_latent" in data:
            next_latent = data["next_latent"]
            if next_latent.dim() == 3:
                next_latent = next_latent[:, -1, :]
            next_mask = supervision_mask.clone()
            if "next_latent_mask" in data:
                next_mask = next_mask & data["next_latent_mask"].to(torch.bool).reshape(-1)
            if next_mask.any():
                with torch.no_grad():
                    target_next_state = state_encoder(
                        cast_tensor_to_module_dtype(next_latent[next_mask], state_encoder)
                    )
                pred_next_state_for_state = transition_reward_net(
                    state_encoder(latent_last[next_mask]), action_labels[next_mask]
                )[0]
                state_loss = F.mse_loss(pred_next_state_for_state, target_next_state)
                policy_loss = policy_loss + config.state_loss_coef * state_loss
                metrics["actor/state_loss"] = state_loss.detach().item()
                metrics["actor/world_model_loss"] += (config.state_loss_coef * state_loss.detach()).item()
    elif world_model_enabled:
        metrics["actor/world_model_supervision_missing"] = 1.0

    return policy_loss, metrics


def value_loss(config: CriticConfig, model_output, data: TensorDict, dp_group=None):
    vpreds = model_output["values"]
    vpreds = no_padding_2_padding(vpreds, data)  # (bsz, response_length)

    values = data["values"]
    returns = data["returns"]
    response_mask = data["response_mask"].to(bool)

    vf_loss, vf_clipfrac = compute_value_loss(
        vpreds=vpreds,
        values=values,
        returns=returns,
        response_mask=response_mask,
        cliprange_value=config.cliprange_value,
        loss_agg_mode=config.loss_agg_mode,
    )

    metrics = {}

    metrics.update(
        {
            "critic/vf_loss": vf_loss.detach().item(),
            "critic/vf_clipfrac": vf_clipfrac.detach().item(),
            "critic/vpred_mean": masked_mean(vpreds, response_mask).detach().item(),
        }
    )

    return vf_loss, metrics
