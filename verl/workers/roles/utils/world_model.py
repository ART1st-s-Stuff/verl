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
