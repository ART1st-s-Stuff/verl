"""Shared latent-planner action schema.

This module defines a single source of truth for:
- planner boundary tokens
- latent planner action tokens
- token/name/id mappings
"""

from __future__ import annotations

ACTION_START_TOKEN = "<|action_start|>"
ACTION_END_TOKEN = "<|action_end|>"

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


def normalize_action_name(name: str) -> str:
    canonical_name = str(name).lower().strip()
    return ACTION_NAME_ALIASES.get(canonical_name, canonical_name)


ACTION_NAME_TO_TOKEN: dict[str, str] = {name: token for token, name in ACTION_TOKEN_TO_NAME.items()}
ACTION_NAME_TO_TOKEN.update(
    {alias: ACTION_NAME_TO_TOKEN[canonical] for alias, canonical in ACTION_NAME_ALIASES.items()}
)
ACTION_NAME_TO_ID: dict[str, int] = {name: idx for idx, name in enumerate(ACTION_NAMES)}
ACTION_ID_TO_NAME: dict[int, str] = {idx: name for name, idx in ACTION_NAME_TO_ID.items()}
