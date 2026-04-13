"""Shared latent-planner action schema.

This module defines a single source of truth for:
- planner boundary tokens
- latent planner action tokens
- token/name/id mappings
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

ACTION_START_TOKEN = "<|action_start|>"
ACTION_END_TOKEN = "<|action_end|>"
LATENT_TOKEN = "<|latent|>"

ACTION_TOKEN_TO_NAME: dict[str, str] = {
    "<|act_moveahead|>": "move_forward",
    "<|act_moveback|>": "move_backward",
    "<|act_moveright|>": "move_right",
    "<|act_moveleft|>": "move_left",
    "<|act_rotateright|>": "turn_right",
    "<|act_rotateleft|>": "turn_left",
    "<|act_lookup|>": "look_up",
    "<|act_lookdown|>": "look_down",
}

ACTION_TOKENS: tuple[str, ...] = tuple(ACTION_TOKEN_TO_NAME.keys())
ACTION_NAMES: tuple[str, ...] = tuple(ACTION_TOKEN_TO_NAME.values())
ACTION_NAME_ALIASES: dict[str, str] = {
    "moveahead": "move_forward",
    "moveback": "move_backward",
    "moveright": "move_right",
    "moveleft": "move_left",
    "rotateright": "turn_right",
    "rotateleft": "turn_left",
    "lookup": "look_up",
    "lookdown": "look_down",
}

PLANNER_SPECIAL_TOKENS: tuple[str, ...] = (
    LATENT_TOKEN,
    ACTION_START_TOKEN,
    ACTION_END_TOKEN,
)


def normalize_action_name(name: str) -> str:
    canonical_name = str(name).lower().strip()
    return ACTION_NAME_ALIASES.get(canonical_name, canonical_name)


def get_special_tokens(num_actions: int | None = None) -> tuple[str, ...]:
    ordered_tokens = [*PLANNER_SPECIAL_TOKENS, *get_action_tokens(num_actions)]
    return tuple(dict.fromkeys(ordered_tokens))


def get_action_tokens(num_actions: int | None = None) -> tuple[str, ...]:
    if num_actions is None:
        return ACTION_TOKENS
    return ACTION_TOKENS[:num_actions]


def get_action_names(num_actions: int | None = None) -> tuple[str, ...]:
    if num_actions is None:
        return ACTION_NAMES
    return ACTION_NAMES[:num_actions]


def _get_tokenizer_from_processing_class(processing_class: Any):
    return getattr(processing_class, "tokenizer", processing_class)


def get_action_token_ids(processing_class: Any, num_actions: int | None = None) -> list[int]:
    tokenizer = _get_tokenizer_from_processing_class(processing_class)
    token_ids: list[int] = []
    for token in get_action_tokens(num_actions):
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is None:
            raise ValueError(f"Failed to resolve token id for action token: {token}")
        token_ids.append(int(token_id))
    return token_ids


def get_special_token_id(processing_class: Any, token: str) -> int:
    tokenizer = _get_tokenizer_from_processing_class(processing_class)
    token_id = tokenizer.convert_tokens_to_ids(token)
    if token_id is None:
        raise ValueError(f"Failed to resolve token id for special token: {token}")
    return int(token_id)


def get_action_start_token_id(processing_class: Any) -> int:
    return get_special_token_id(processing_class, ACTION_START_TOKEN)


def get_latent_token_id(processing_class: Any) -> int:
    return get_special_token_id(processing_class, LATENT_TOKEN)


def register_special_tokens(processing_class: Any, num_actions: int | None = None) -> int:
    tokenizer = _get_tokenizer_from_processing_class(processing_class)
    if tokenizer is None or not hasattr(tokenizer, "add_special_tokens"):
        return 0

    existing_tokens = set(getattr(tokenizer, "get_vocab", lambda: {})().keys())
    missing_tokens = [token for token in get_special_tokens(num_actions) if token not in existing_tokens]
    if not missing_tokens:
        return 0

    return int(tokenizer.add_special_tokens({"additional_special_tokens": missing_tokens}))


def maybe_resize_token_embeddings(model: Any, processing_class: Any) -> bool:
    tokenizer = _get_tokenizer_from_processing_class(processing_class)
    if tokenizer is None or not hasattr(model, "resize_token_embeddings"):
        return False

    input_embeddings = model.get_input_embeddings()
    if input_embeddings is None or not hasattr(input_embeddings, "num_embeddings"):
        return False

    target_vocab_size = len(tokenizer)
    if int(input_embeddings.num_embeddings) == int(target_vocab_size):
        return False

    model.resize_token_embeddings(target_vocab_size)
    return True


def get_action_token_id_tensor(
    processing_class: Any,
    num_actions: int | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    return torch.tensor(get_action_token_ids(processing_class, num_actions), dtype=torch.long, device=device)


def _unwrap_output_head_module(module: Any):
    current = module
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        if hasattr(current, "lm_head"):
            return current.lm_head
        if hasattr(current, "language_model") and hasattr(current.language_model, "lm_head"):
            return current.language_model.lm_head
        if hasattr(current, "output_layer"):
            return current.output_layer
        if hasattr(current, "language_model") and hasattr(current.language_model, "output_layer"):
            return current.language_model.output_layer
        if hasattr(current, "_fsdp_wrapped_module"):
            current = current._fsdp_wrapped_module
            continue
        if hasattr(current, "module"):
            current = current.module
            continue
        current = None
    return None


def compute_action_prior_from_latent(
    latent_z: torch.Tensor,
    output_head_owner: Any,
    processing_class: Any,
    num_actions: int,
    temperature: float = 1.0,
) -> dict[str, torch.Tensor]:
    if num_actions <= 0:
        return {}
    output_head = _unwrap_output_head_module(output_head_owner)
    if output_head is None or not hasattr(output_head, "weight"):
        return {}

    latent_z = latent_z if latent_z.dim() == 2 else latent_z[:, -1, :]
    action_token_ids = get_action_token_id_tensor(processing_class, num_actions=num_actions, device=latent_z.device)
    if action_token_ids.numel() == 0:
        return {}
    if int(action_token_ids.max().item()) >= output_head.weight.shape[0]:
        return {}
    action_weight = output_head.weight[action_token_ids]
    action_bias = getattr(output_head, "bias", None)
    if action_bias is not None:
        action_bias = action_bias[action_token_ids]

    action_logits = F.linear(latent_z, action_weight, action_bias)
    if temperature != 1.0:
        action_logits = action_logits / temperature
    action_log_probs = torch.log_softmax(action_logits, dim=-1)
    action_probs = action_log_probs.exp()
    action_ids = action_probs.argmax(dim=-1)
    return {
        "action_prior_logits": action_logits,
        "action_prior_log_probs": action_log_probs,
        "action_prior_probs": action_probs,
        "action_prior_action_ids": action_ids,
    }


ACTION_NAME_TO_TOKEN: dict[str, str] = {name: token for token, name in ACTION_TOKEN_TO_NAME.items()}
ACTION_NAME_TO_TOKEN.update(
    {alias: ACTION_NAME_TO_TOKEN[canonical] for alias, canonical in ACTION_NAME_ALIASES.items()}
)
ACTION_NAME_TO_ID: dict[str, int] = {name: idx for idx, name in enumerate(ACTION_NAMES)}
ACTION_ID_TO_NAME: dict[int, str] = {idx: name for name, idx in ACTION_NAME_TO_ID.items()}
