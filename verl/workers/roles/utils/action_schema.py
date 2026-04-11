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
    "<|act_moveahead|>": "moveahead",
    "<|act_moveback|>": "moveback",
    "<|act_moveright|>": "moveright",
    "<|act_moveleft|>": "moveleft",
    "<|act_rotateright|>": "rotateright",
    "<|act_rotateleft|>": "rotateleft",
    "<|act_lookup|>": "lookup",
    "<|act_lookdown|>": "lookdown",
}

ACTION_TOKENS: tuple[str, ...] = tuple(ACTION_TOKEN_TO_NAME.keys())
ACTION_NAMES: tuple[str, ...] = tuple(ACTION_TOKEN_TO_NAME.values())
ACTION_NAME_TO_TOKEN: dict[str, str] = {name: token for token, name in ACTION_TOKEN_TO_NAME.items()}
ACTION_NAME_TO_ID: dict[str, int] = {name: idx for idx, name in enumerate(ACTION_NAMES)}
ACTION_ID_TO_NAME: dict[int, str] = {idx: name for name, idx in ACTION_NAME_TO_ID.items()}
