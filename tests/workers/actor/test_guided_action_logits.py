from types import SimpleNamespace

import torch
from torch import nn

from verl.workers.actor.dp_actor import DataParallelPPOActor


class _TinyCausalLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.arange(7, dtype=torch.float32))
        self.forward_count = 0

    def forward(self, *, input_ids, **_kwargs):
        self.forward_count += 1
        batch, sequence = input_ids.shape
        logits = self.bias.view(1, 1, -1).expand(batch, sequence, -1)
        return SimpleNamespace(logits=logits)


def _actor() -> tuple[DataParallelPPOActor, _TinyCausalLM]:
    actor = object.__new__(DataParallelPPOActor)
    model = _TinyCausalLM()
    actor.actor_module = model
    actor.use_remove_padding = False
    actor.use_fused_kernels = False
    actor.device_name = "cpu"
    actor.param_dtype = torch.float32
    actor.config = SimpleNamespace(entropy_checkpointing=False)
    return actor, model


def test_same_forward_returns_raw_action_boundary_logits_before_temperature() -> None:
    actor, model = _actor()
    micro_batch = {
        "input_ids": torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]]),
        "attention_mask": torch.ones(2, 4, dtype=torch.long),
        "position_ids": torch.arange(4).expand(2, -1),
        "responses": torch.tensor([[3, 4], [2, 1]]),
    }
    entropy, token_log_probs, action_logits = actor._forward_micro_batch(
        micro_batch,
        temperature=2.0,
        action_token_ids=torch.tensor([1, 5]),
        action_response_indices=torch.tensor([0, 1]),
    )
    assert entropy is None
    assert token_log_probs.shape == (2, 2)
    assert torch.equal(action_logits, torch.tensor([[1.0, 5.0], [1.0, 5.0]]))
    assert model.forward_count == 1
    action_logits.sum().backward()
    assert model.bias.grad is not None
    assert float(model.bias.grad[1]) == 2.0
    assert float(model.bias.grad[5]) == 2.0


def test_action_logit_request_rejects_fused_or_misaligned_inputs() -> None:
    actor, _model = _actor()
    micro_batch = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.ones(1, 3, dtype=torch.long),
        "position_ids": torch.arange(3).expand(1, -1),
        "responses": torch.tensor([[2, 3]]),
    }
    actor.use_fused_kernels = True
    try:
        actor._forward_micro_batch(
            micro_batch,
            temperature=1.0,
            action_token_ids=[1, 2],
            action_response_indices=[0],
        )
    except ValueError as exc:
        assert "fused" in str(exc)
    else:
        raise AssertionError("expected fused-kernel rejection")
