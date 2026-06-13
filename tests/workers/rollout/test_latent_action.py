from __future__ import annotations

from types import SimpleNamespace

import torch

from verl.workers.rollout.latent_action import LatentActionTokenIds, extract_latent_action_from_model


class FakeModel:
    def __init__(self, vocab_size: int = 32) -> None:
        self.training = True
        self.vocab_size = vocab_size
        self.eval_called = False
        self.train_called = False

    def eval(self):
        self.eval_called = True
        self.training = False

    def train(self):
        self.train_called = True
        self.training = True

    def __call__(self, input_ids, attention_mask, position_ids, use_cache, output_hidden_states, return_dict):
        assert use_cache is False
        assert output_hidden_states is True
        assert return_dict is True
        batch, seq_len = input_ids.shape
        hidden = torch.arange(batch * seq_len * 4, dtype=torch.float32, device=input_ids.device).reshape(batch, seq_len, 4)
        logits = torch.zeros(batch, seq_len, self.vocab_size, dtype=torch.float32, device=input_ids.device)
        logits[0, 3, 20] = 5.0
        logits[0, 3, 21] = -1.0
        logits[0, 6, 21] = 7.0
        return SimpleNamespace(hidden_states=(hidden - 100.0, hidden), logits=logits)


def test_extract_latent_action_from_model() -> None:
    token_ids = LatentActionTokenIds(
        latent_state_token_id=10,
        action_start_token_id=11,
        action_token_ids=(20, 21),
    )
    input_ids = torch.tensor([[1, 10, 2, 11, 20, 3, 11, 21, 0]])
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0]])
    position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0)
    model = FakeModel()

    out = extract_latent_action_from_model(
        model=model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        token_ids=token_ids,
    )

    assert model.eval_called
    assert model.train_called
    assert out["latent_state_attention_embeddings"].shape == (1, 1, 4)
    torch.testing.assert_close(out["latent_state_attention_embeddings"][0, 0], torch.tensor([4.0, 5.0, 6.0, 7.0]))
    torch.testing.assert_close(out["latent_state_positions"], torch.tensor([[1]]))
    torch.testing.assert_close(out["latent_state_mask"], torch.tensor([[True]]))

    assert out["action_prior_logits"].shape == (1, 2, 2)
    torch.testing.assert_close(out["action_prior_positions"], torch.tensor([[4, 7]]))
    torch.testing.assert_close(out["action_prior_mask"], torch.tensor([[True, True]]))
    torch.testing.assert_close(out["action_prior_logits"][0, 0], torch.tensor([5.0, -1.0]))
    torch.testing.assert_close(out["action_prior_logits"][0, 1], torch.tensor([0.0, 7.0]))
