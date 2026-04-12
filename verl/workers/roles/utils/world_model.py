import torch
import torch.nn as nn


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
