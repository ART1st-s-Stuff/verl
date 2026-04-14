import torch
import torch.nn as nn
import torch.nn.functional as F


def cast_tensor_to_module_dtype(tensor: torch.Tensor, module: nn.Module) -> torch.Tensor:
    """Cast floating-point inputs to match a module's parameter dtype."""
    if not tensor.is_floating_point():
        return tensor
    param = next(module.parameters(), None)
    if param is None or tensor.dtype == param.dtype:
        return tensor
    return tensor.to(dtype=param.dtype)


def canonicalize_latent_z(latent: torch.Tensor) -> torch.Tensor:
    """Return the canonical DreamAgent latent z_t as a [bs, hidden] tensor.

    z_t is defined as the final valid pre-output hidden state from the VLM.
    `world_state` must always be derived from this tensor through `LatentStateEncoder`
    instead of being mixed with it semantically.
    """
    if latent.dim() == 3:
        latent = latent[:, -1, :]
    if latent.dim() != 2:
        raise ValueError(f"Expected latent z_t with shape [bs, hidden], got {tuple(latent.shape)}")
    return latent


def extract_latent_z(
    latent: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    anchor_token_id: int | None = None,
) -> torch.Tensor:
    """Extract canonical z_t, preferring the token right before `anchor_token_id`.

    Fallback order:
    1. token immediately before the first valid anchor token
    2. last valid token in the sequence
    """
    if latent.dim() == 2:
        return canonicalize_latent_z(latent)

    if latent.dim() == 3 and input_ids.dim() == 2:
        batch_size = latent.shape[0]
        device = latent.device
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device)

        valid_lengths = attention_mask.long().sum(dim=1)
        selected_positions = torch.clamp(valid_lengths - 1, min=0)
        if anchor_token_id is not None:
            for i in range(batch_size):
                valid_len = int(valid_lengths[i].item())
                if valid_len <= 0:
                    continue
                valid_tokens = input_ids[i, :valid_len]
                matches = (valid_tokens == anchor_token_id).nonzero(as_tuple=False)
                if matches.numel() > 0:
                    anchor_pos = int(matches[0].item())
                    selected_positions[i] = max(anchor_pos - 1, 0)
        batch_indices = torch.arange(batch_size, device=device)
        return canonicalize_latent_z(latent[batch_indices, selected_positions])

    if hasattr(latent, "values") and hasattr(latent, "offsets") and hasattr(input_ids, "values") and hasattr(input_ids, "offsets"):
        latent_values = latent.values()
        latent_offsets = latent.offsets()
        input_values = input_ids.values()
        input_offsets = input_ids.offsets()
        rows = []
        for start, end, token_start, token_end in zip(
            latent_offsets[:-1].tolist(),
            latent_offsets[1:].tolist(),
            input_offsets[:-1].tolist(),
            input_offsets[1:].tolist(),
        ):
            if end <= start:
                raise ValueError("Cannot extract latent z_t from empty packed sequence.")
            select_idx = end - 1
            if anchor_token_id is not None:
                tokens = input_values[token_start:token_end]
                matches = (tokens == anchor_token_id).nonzero(as_tuple=False)
                if matches.numel() > 0:
                    anchor_rel = int(matches[0].item())
                    select_idx = start + max(anchor_rel - 1, 0)
            rows.append(latent_values[select_idx])
        return canonicalize_latent_z(torch.stack(rows, dim=0))

    raise ValueError(
        f"Unsupported latent/input_ids shapes for z_t extraction: latent={tuple(latent.shape)}, input_ids={tuple(input_ids.shape)}"
    )


class LatentStateEncoder(nn.Module):
    """Map LLM latent vector to compact world state."""

    def __init__(self, latent_dim: int, world_state_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, world_state_dim),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.net(latent)


class TransitionRewardNet(nn.Module):
    """Predict next world state and expected reward from (state, action)."""

    def __init__(self, world_state_dim: int, num_actions: int, hidden_dim: int):
        super().__init__()
        self.action_embed = nn.Embedding(num_actions, hidden_dim)
        self.state_proj = nn.Linear(world_state_dim, hidden_dim)
        self.trunk = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.next_state_head = nn.Linear(hidden_dim, world_state_dim)
        self.reward_head = nn.Linear(hidden_dim, 1)

    def forward(self, world_state: torch.Tensor, action_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        state_h = self.state_proj(world_state)
        action_h = self.action_embed(action_ids)
        hidden = self.trunk(state_h + action_h)
        next_state = self.next_state_head(hidden)
        expected_reward = self.reward_head(hidden).squeeze(-1)
        return next_state, expected_reward


class BaseLatentPredictor(nn.Module):
    """Unified interface for predictor variants used in Phase2/Phase3."""

    def forward_step(self, latent: torch.Tensor, action_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    def forward_multi_step(
        self,
        latent_0: torch.Tensor,
        action_seq: torch.Tensor,
        horizon: int | None = None,
    ) -> dict[str, torch.Tensor]:
        if action_seq.dim() != 2:
            raise ValueError(f"Expected action_seq with shape [bs, T], got {tuple(action_seq.shape)}")
        steps = action_seq.shape[1] if horizon is None else min(int(horizon), action_seq.shape[1])
        rollout_latents = []
        rollout_rewards = []
        cur_latent = latent_0
        for t in range(steps):
            step = self.forward_step(cur_latent, action_seq[:, t])
            cur_latent = step["pred_next_latent"]
            rollout_latents.append(cur_latent)
            if "pred_reward" in step and step["pred_reward"] is not None:
                rollout_rewards.append(step["pred_reward"])
        outputs: dict[str, torch.Tensor] = {"pred_next_latent_seq": torch.stack(rollout_latents, dim=1)}
        if len(rollout_rewards) > 0:
            outputs["pred_reward_seq"] = torch.stack(rollout_rewards, dim=1)
        return outputs

    def compute_auxiliary_losses(
        self,
        latent: torch.Tensor,
        action_ids: torch.Tensor,
        step_rewards: torch.Tensor,
        config,
        next_latent: torch.Tensor | None = None,
        next_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Return (weighted_aux_loss, metric_dict)."""
        raise NotImplementedError


class WorldStatePredictor(BaseLatentPredictor):
    """State-space predictor with optional state->latent decoder."""

    def __init__(
        self,
        latent_dim: int,
        world_state_dim: int,
        num_actions: int,
        hidden_dim: int,
        use_decoder: bool,
    ):
        super().__init__()
        self.use_decoder = use_decoder
        self.state_encoder = LatentStateEncoder(latent_dim, world_state_dim)
        self.transition_reward_net = TransitionRewardNet(
            world_state_dim=world_state_dim,
            num_actions=num_actions,
            hidden_dim=hidden_dim,
        )
        self.state_decoder = (
            nn.Sequential(
                nn.Linear(world_state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, latent_dim),
            )
            if use_decoder
            else None
        )

    def encode_world_state(self, latent: torch.Tensor) -> torch.Tensor:
        latent = cast_tensor_to_module_dtype(latent, self.state_encoder)
        return self.state_encoder(latent)

    def forward_step(self, latent: torch.Tensor, action_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        world_state = self.encode_world_state(latent)
        pred_next_state, pred_reward = self.transition_reward_net(world_state, action_ids.long())
        if self.state_decoder is not None:
            pred_next_latent = self.state_decoder(pred_next_state)
        else:
            # Legacy fallback keeps shape contract while preserving old behavior.
            pred_next_latent = latent
        return {
            "pred_next_world_state": pred_next_state,
            "pred_next_latent": pred_next_latent,
            "pred_reward": pred_reward,
        }

    def compute_auxiliary_losses(
        self,
        latent: torch.Tensor,
        action_ids: torch.Tensor,
        step_rewards: torch.Tensor,
        config,
        next_latent: torch.Tensor | None = None,
        next_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        step_out = self.forward_step(latent=latent, action_ids=action_ids)
        pred_reward = step_out["pred_reward"]
        reward_loss = F.mse_loss(pred_reward, step_rewards.reshape_as(pred_reward))
        total = config.reward_loss_coef * reward_loss
        metrics = {
            "actor/reward_loss": float(reward_loss.detach().item()),
            "actor/world_model_loss": float((config.reward_loss_coef * reward_loss.detach()).item()),
        }

        if next_latent is None:
            return total, metrics

        if next_mask is None:
            next_mask = torch.ones(next_latent.shape[0], device=next_latent.device, dtype=torch.bool)
        if not next_mask.any():
            return total, metrics

        if self.use_decoder:
            pred_next_latent = step_out["pred_next_latent"][next_mask]
            target_next_latent = next_latent[next_mask].to(pred_next_latent.dtype)
            latent_loss = F.mse_loss(pred_next_latent, target_next_latent)
            total = total + config.state_loss_coef * latent_loss
            metrics["actor/state_loss"] = float(latent_loss.detach().item())
            metrics["actor/world_model_loss"] += float((config.state_loss_coef * latent_loss.detach()).item())
            return total, metrics

        with torch.no_grad():
            target_next_state = self.state_encoder(cast_tensor_to_module_dtype(next_latent[next_mask], self.state_encoder))
        pred_next_state = self.transition_reward_net(
            self.state_encoder(cast_tensor_to_module_dtype(latent[next_mask], self.state_encoder)),
            action_ids[next_mask],
        )[0]
        state_loss = F.mse_loss(pred_next_state, target_next_state)
        total = total + config.state_loss_coef * state_loss
        metrics["actor/state_loss"] = float(state_loss.detach().item())
        metrics["actor/world_model_loss"] += float((config.state_loss_coef * state_loss.detach()).item())
        return total, metrics


class DirectLatentMLPPredictor(BaseLatentPredictor):
    """Direct latent dynamics predictor: (latent, action) -> next latent."""

    def __init__(self, latent_dim: int, num_actions: int, hidden_dim: int):
        super().__init__()
        self.action_embed = nn.Embedding(num_actions, hidden_dim)
        self.latent_proj = nn.Linear(latent_dim, hidden_dim)
        self.trunk = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.next_latent_head = nn.Linear(hidden_dim, latent_dim)
        self.reward_head = nn.Linear(hidden_dim, 1)

    def forward_step(self, latent: torch.Tensor, action_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.trunk(self.latent_proj(latent) + self.action_embed(action_ids.long()))
        return {
            "pred_next_latent": self.next_latent_head(hidden),
            "pred_reward": self.reward_head(hidden).squeeze(-1),
        }

    def compute_auxiliary_losses(
        self,
        latent: torch.Tensor,
        action_ids: torch.Tensor,
        step_rewards: torch.Tensor,
        config,
        next_latent: torch.Tensor | None = None,
        next_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        step_out = self.forward_step(latent=latent, action_ids=action_ids)
        pred_reward = step_out["pred_reward"]
        reward_loss = F.mse_loss(pred_reward, step_rewards.reshape_as(pred_reward))
        total = config.reward_loss_coef * reward_loss
        metrics = {
            "actor/reward_loss": float(reward_loss.detach().item()),
            "actor/world_model_loss": float((config.reward_loss_coef * reward_loss.detach()).item()),
        }

        if next_latent is not None:
            if next_mask is None:
                next_mask = torch.ones(next_latent.shape[0], device=next_latent.device, dtype=torch.bool)
            if next_mask.any():
                latent_loss = F.mse_loss(
                    step_out["pred_next_latent"][next_mask],
                    next_latent[next_mask].to(step_out["pred_next_latent"].dtype),
                )
                total = total + config.state_loss_coef * latent_loss
                metrics["actor/state_loss"] = float(latent_loss.detach().item())
                metrics["actor/world_model_loss"] += float((config.state_loss_coef * latent_loss.detach()).item())
        return total, metrics


class LeWMLatentPredictor(BaseLatentPredictor):
    """LeWM-like latent dynamics wrapper backed by ARPredictor."""

    def __init__(
        self,
        latent_dim: int,
        num_actions: int,
        hidden_dim: int,
        num_frames: int,
    ):
        super().__init__()
        from lewm.module import ARPredictor

        self.num_frames = max(int(num_frames), 1)
        self.action_embed = nn.Embedding(num_actions, latent_dim)
        self.predictor = ARPredictor(
            num_frames=self.num_frames,
            depth=2,
            heads=4,
            mlp_dim=max(hidden_dim, latent_dim),
            input_dim=latent_dim,
            hidden_dim=max(hidden_dim, latent_dim),
            output_dim=latent_dim,
        )
        self.reward_head = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward_step(self, latent: torch.Tensor, action_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        x = latent.unsqueeze(1)
        c = self.action_embed(action_ids.long()).unsqueeze(1)
        pred_next_latent = self.predictor(x, c)[:, -1, :]
        pred_reward = self.reward_head(torch.cat([latent, c.squeeze(1)], dim=-1)).squeeze(-1)
        return {"pred_next_latent": pred_next_latent, "pred_reward": pred_reward}

    def forward_multi_step(
        self,
        latent_0: torch.Tensor,
        action_seq: torch.Tensor,
        horizon: int | None = None,
    ) -> dict[str, torch.Tensor]:
        if action_seq.dim() != 2:
            raise ValueError(f"Expected action_seq with shape [bs, T], got {tuple(action_seq.shape)}")
        steps = action_seq.shape[1] if horizon is None else min(int(horizon), action_seq.shape[1])
        emb_hist = latent_0.unsqueeze(1)
        pred_latents = []
        pred_rewards = []
        for t in range(steps):
            act_emb_hist = self.action_embed(action_seq[:, : t + 1].long())
            x = emb_hist[:, -self.num_frames :]
            c = act_emb_hist[:, -self.num_frames :]
            pred_seq = self.predictor(x, c)
            pred_next_latent = pred_seq[:, -1, :]
            pred_latents.append(pred_next_latent)
            prev_latent = emb_hist[:, -1, :]
            reward = self.reward_head(torch.cat([prev_latent, c[:, -1, :]], dim=-1)).squeeze(-1)
            pred_rewards.append(reward)
            emb_hist = torch.cat([emb_hist, pred_next_latent.unsqueeze(1)], dim=1)
        return {
            "pred_next_latent_seq": torch.stack(pred_latents, dim=1),
            "pred_reward_seq": torch.stack(pred_rewards, dim=1),
        }

    def compute_auxiliary_losses(
        self,
        latent: torch.Tensor,
        action_ids: torch.Tensor,
        step_rewards: torch.Tensor,
        config,
        next_latent: torch.Tensor | None = None,
        next_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        step_out = self.forward_step(latent=latent, action_ids=action_ids)
        reward_loss = F.mse_loss(step_out["pred_reward"], step_rewards.reshape_as(step_out["pred_reward"]))
        total = config.reward_loss_coef * reward_loss
        metrics = {
            "actor/reward_loss": float(reward_loss.detach().item()),
            "actor/world_model_loss": float((config.reward_loss_coef * reward_loss.detach()).item()),
        }
        if next_latent is not None:
            if next_mask is None:
                next_mask = torch.ones(next_latent.shape[0], device=next_latent.device, dtype=torch.bool)
            if next_mask.any():
                latent_loss = F.mse_loss(
                    step_out["pred_next_latent"][next_mask],
                    next_latent[next_mask].to(step_out["pred_next_latent"].dtype),
                )
                total = total + config.state_loss_coef * latent_loss
                metrics["actor/state_loss"] = float(latent_loss.detach().item())
                metrics["actor/world_model_loss"] += float((config.state_loss_coef * latent_loss.detach()).item())
        return total, metrics


def build_latent_predictor(
    predictor_mode: str,
    latent_dim: int,
    num_actions: int,
    world_state_dim: int,
    hidden_dim: int,
    multi_step_horizon: int = 1,
) -> BaseLatentPredictor:
    if predictor_mode == "world_state_with_decoder":
        return WorldStatePredictor(
            latent_dim=latent_dim,
            world_state_dim=world_state_dim,
            num_actions=num_actions,
            hidden_dim=hidden_dim,
            use_decoder=True,
        )
    if predictor_mode == "direct_latent_mlp":
        return DirectLatentMLPPredictor(
            latent_dim=latent_dim,
            num_actions=num_actions,
            hidden_dim=hidden_dim,
        )
    if predictor_mode == "lewm_latent_dynamics":
        return LeWMLatentPredictor(
            latent_dim=latent_dim,
            num_actions=num_actions,
            hidden_dim=hidden_dim,
            num_frames=multi_step_horizon,
        )
    if predictor_mode == "world_state_mlp":
        return WorldStatePredictor(
            latent_dim=latent_dim,
            world_state_dim=world_state_dim,
            num_actions=num_actions,
            hidden_dim=hidden_dim,
            use_decoder=False,
        )
    raise ValueError(f"Unsupported predictor_mode: {predictor_mode!r}")


def compute_world_model_aux_loss(
    predictor: BaseLatentPredictor | None,
    latent: torch.Tensor | None,
    action_labels: torch.Tensor | None,
    step_rewards: torch.Tensor | None,
    config,
    action_label_mask: torch.Tensor | None = None,
    next_latent: torch.Tensor | None = None,
    next_latent_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor | None, dict[str, float], bool]:
    if predictor is None or latent is None or action_labels is None or step_rewards is None:
        return None, {}, False

    action_labels = action_labels.long().reshape(-1)
    step_rewards = step_rewards.float().reshape(-1)
    supervision_mask = torch.ones_like(action_labels, dtype=torch.bool)
    if action_label_mask is not None:
        supervision_mask = supervision_mask & action_label_mask.to(torch.bool).reshape(-1)
    if not supervision_mask.any():
        return None, {"actor/world_model_supervision_missing": 1.0}, True

    latent = latent[supervision_mask]
    action_labels = action_labels[supervision_mask]
    step_rewards = step_rewards[supervision_mask]
    next_latent_filtered = None
    next_mask = None
    if next_latent is not None:
        if next_latent.dim() == 3:
            next_latent = next_latent[:, -1, :]
        next_latent_filtered = next_latent[supervision_mask]
        next_mask = torch.ones(next_latent_filtered.shape[0], device=next_latent_filtered.device, dtype=torch.bool)
        if next_latent_mask is not None:
            next_mask = next_mask & next_latent_mask.to(torch.bool).reshape(-1)[supervision_mask]

    aux_loss, metrics = predictor.compute_auxiliary_losses(
        latent=latent,
        action_ids=action_labels,
        step_rewards=step_rewards,
        config=config,
        next_latent=next_latent_filtered,
        next_mask=next_mask,
    )
    return aux_loss, metrics, True
