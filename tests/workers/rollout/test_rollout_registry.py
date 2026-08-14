import pytest

from verl.workers.rollout.base import get_rollout_class, register_rollout


def test_register_rollout_is_idempotent_and_rejects_conflicts() -> None:
    name = "unit_test_external_rollout"
    register_rollout(name, "async", "builtins.str")
    register_rollout(name, "async", "builtins.str")
    assert get_rollout_class(name, "async") is str

    with pytest.raises(ValueError, match="already registered"):
        register_rollout(name, "async", "builtins.int")


def test_register_rollout_rejects_invalid_identity() -> None:
    for args in (
        ("", "async", "builtins.str"),
        ("valid", "bad-mode", "builtins.str"),
        ("valid", "async", "str"),
    ):
        with pytest.raises(ValueError):
            register_rollout(*args)
