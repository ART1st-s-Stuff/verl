import torch
import torch.nn as nn


class FeatureEncoderHead(nn.Module):
    def __init__(self, latent_dim: int, out_dim: int, hidden_dim: int | None = None):
        super().__init__()
        hidden_dim = hidden_dim or latent_dim
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.net(latent)


def extract_action_start_latent(
    hidden_states: torch.Tensor,
    input_ids: torch.Tensor,
    action_start_token_id: int | None = None,
) -> torch.Tensor:
    """
    Pick latent at action-start token position for each sample.
    Fallback to the last valid token when action-start token is not found.
    """
    if hidden_states.dim() != 3 or input_ids.dim() != 2:
        raise ValueError(f"Unexpected shapes: hidden_states={hidden_states.shape}, input_ids={input_ids.shape}")
    bs, seq_len = input_ids.shape
    if hidden_states.shape[0] != bs or hidden_states.shape[1] != seq_len:
        raise ValueError(f"Mismatched shapes: hidden_states={hidden_states.shape}, input_ids={input_ids.shape}")

    if action_start_token_id is None:
        return hidden_states[:, -1, :]

    idx = torch.full((bs,), seq_len - 1, device=input_ids.device, dtype=torch.long)
    mask = input_ids.eq(action_start_token_id)
    for b in range(bs):
        pos = torch.nonzero(mask[b], as_tuple=False)
        if pos.numel() > 0:
            idx[b] = pos[0, 0]
    return hidden_states[torch.arange(bs, device=input_ids.device), idx]
