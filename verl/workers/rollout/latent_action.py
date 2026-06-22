from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor


DEFAULT_LATENT_STATE_TOKEN = "<|latent_state|>"
DEFAULT_ACTION_START_TOKEN = "<|action_start|>"
DEFAULT_ACTION_TOKENS = tuple(f"<|action_({idx})|>" for idx in range(8))


@dataclass(frozen=True)
class LatentActionTokenIds:
    latent_state_token_id: int
    action_start_token_id: int
    action_token_ids: tuple[int, ...]


def _single_token_id(tokenizer, token: str) -> int:
    token_id = tokenizer.convert_tokens_to_ids(token)
    unk_id = getattr(tokenizer, "unk_token_id", None)
    if token_id is None or token_id == unk_id:
        raise ValueError(f"{token!r} is not in the tokenizer vocabulary.")

    encoded = tokenizer.encode(token, add_special_tokens=False)
    if len(encoded) != 1 or int(encoded[0]) != int(token_id):
        raise ValueError(f"{token!r} must be encoded as one token, got ids={encoded}.")
    return int(token_id)


def get_latent_action_token_ids(
    tokenizer,
    latent_state_token: str = DEFAULT_LATENT_STATE_TOKEN,
    action_start_token: str = DEFAULT_ACTION_START_TOKEN,
    action_tokens: Sequence[str] = DEFAULT_ACTION_TOKENS,
) -> LatentActionTokenIds:
    return LatentActionTokenIds(
        latent_state_token_id=_single_token_id(tokenizer, latent_state_token),
        action_start_token_id=_single_token_id(tokenizer, action_start_token),
        action_token_ids=tuple(_single_token_id(tokenizer, token) for token in action_tokens),
    )


def _last_hidden_state(output) -> Tensor:
    hidden_states = getattr(output, "hidden_states", None)
    if hidden_states is not None:
        return hidden_states[-1]
    last_hidden_state = getattr(output, "last_hidden_state", None)
    if last_hidden_state is not None:
        return last_hidden_state
    raise ValueError("Model output must include hidden_states or last_hidden_state.")


def _allocate_positions(batch_size: int, count: int, device: torch.device) -> Tensor:
    return torch.full((batch_size, count), -1, dtype=torch.long, device=device)


def _extract_from_outputs(
    input_ids: Tensor,
    attention_mask: Tensor,
    hidden_states: Tensor,
    logits: Tensor,
    token_ids: LatentActionTokenIds,
) -> dict[str, Tensor]:
    batch_size, _ = input_ids.shape
    device = input_ids.device
    valid_mask = attention_mask.to(torch.bool)
    action_token_ids = torch.tensor(token_ids.action_token_ids, dtype=torch.long, device=logits.device)

    latent_matches = (input_ids == token_ids.latent_state_token_id) & valid_mask
    action_start_matches = (input_ids == token_ids.action_start_token_id) & valid_mask

    max_latent_count = int(latent_matches.sum(dim=1).max().item()) if batch_size > 0 else 0
    max_action_count = int(action_start_matches.sum(dim=1).max().item()) if batch_size > 0 else 0
    hidden_size = hidden_states.shape[-1]
    num_actions = len(token_ids.action_token_ids)

    latent_embeddings = hidden_states.new_zeros((batch_size, max_latent_count, hidden_size))
    latent_positions = _allocate_positions(batch_size, max_latent_count, device)
    latent_mask = torch.zeros((batch_size, max_latent_count), dtype=torch.bool, device=device)

    action_prior_logits = logits.new_zeros((batch_size, max_action_count, num_actions))
    action_prior_positions = _allocate_positions(batch_size, max_action_count, device)
    action_prior_mask = torch.zeros((batch_size, max_action_count), dtype=torch.bool, device=device)

    for batch_idx in range(batch_size):
        latent_pos = torch.nonzero(latent_matches[batch_idx], as_tuple=False).flatten()
        if latent_pos.numel() > 0:
            count = latent_pos.numel()
            latent_embeddings[batch_idx, :count] = hidden_states[batch_idx, latent_pos]
            latent_positions[batch_idx, :count] = latent_pos
            latent_mask[batch_idx, :count] = True

        action_start_pos = torch.nonzero(action_start_matches[batch_idx], as_tuple=False).flatten()
        valid_action_starts = []
        for pos in action_start_pos.tolist():
            next_pos = pos + 1
            if next_pos < input_ids.shape[1] and bool(valid_mask[batch_idx, next_pos]):
                valid_action_starts.append(pos)
        if valid_action_starts:
            starts = torch.tensor(valid_action_starts, dtype=torch.long, device=device)
            count = starts.numel()
            action_prior_logits[batch_idx, :count] = logits[batch_idx, starts][:, action_token_ids]
            action_prior_positions[batch_idx, :count] = starts + 1
            action_prior_mask[batch_idx, :count] = True

    return {
        "latent_state_attention_embeddings": latent_embeddings,
        "latent_state_positions": latent_positions,
        "latent_state_mask": latent_mask,
        "action_prior_logits": action_prior_logits,
        "action_prior_positions": action_prior_positions,
        "action_prior_mask": action_prior_mask,
    }


@torch.no_grad()
def extract_latent_action_from_model(
    model,
    input_ids: Tensor,
    attention_mask: Tensor,
    position_ids: Tensor,
    token_ids: LatentActionTokenIds,
    multi_modal_inputs: dict | None = None,
) -> dict[str, Tensor]:
    model_was_training = bool(getattr(model, "training", False))
    model.eval()
    kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=False,
        output_hidden_states=True,
        return_dict=True,
    )
    if multi_modal_inputs:
        kwargs.update(multi_modal_inputs)
    output = model(**kwargs)
    if model_was_training:
        model.train()

    hidden_states = _last_hidden_state(output)
    logits = getattr(output, "logits", None)
    if logits is None:
        raise ValueError("Model output must include logits to extract action prior logits.")
    return _extract_from_outputs(
        input_ids=input_ids,
        attention_mask=attention_mask,
        hidden_states=hidden_states,
        logits=logits,
        token_ids=token_ids,
    )
