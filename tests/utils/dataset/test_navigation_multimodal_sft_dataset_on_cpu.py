import types

import torch

from verl.utils.dataset.dataset_utils import DatasetPadMode
from vagen.datasets.navigation_multimodal_sft_dataset import NavigationMultimodalSFTDataset


class DummyTokenizer:
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [ord(ch) for ch in text]


class DummyImageProcessor:
    pass


class DummyProcessor:
    def __init__(self):
        self.tokenizer = DummyTokenizer()
        self.image_processor = DummyImageProcessor()

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, enable_thinking=None, **kwargs):
        del tokenize, enable_thinking, kwargs
        parts = []
        for message in messages:
            content = message["content"]
            if isinstance(content, list):
                rendered = "".join(item.get("text", "") if item.get("type") == "text" else "<image>" for item in content)
            else:
                rendered = content
            parts.append(f"<{message['role']}>{rendered}</{message['role']}>")
        if add_generation_prompt:
            parts.append("<assistant>")
        return "".join(parts)

    def __call__(self, text, images=None, return_tensors="pt"):
        del return_tensors
        prompt = text[0]
        token_ids = []
        index = 0
        while index < len(prompt):
            if prompt.startswith("<image>", index):
                token_ids.extend([500, 501, 502])
                index += len("<image>")
                continue
            token_ids.append(ord(prompt[index]))
            index += 1

        if images:
            assert 500 in token_ids, "Image-bearing samples should include image expansion tokens"

        input_ids = torch.tensor([token_ids], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        return {"input_ids": input_ids, "attention_mask": attention_mask}


def test_navigation_multimodal_dataset_uses_processor_tokenization_for_alignment():
    dataset = NavigationMultimodalSFTDataset.__new__(NavigationMultimodalSFTDataset)
    dataset.pad_mode = DatasetPadMode.NO_PADDING
    dataset.truncation = "error"
    dataset.max_length = 512
    dataset.shuffle = False
    dataset.seed = None
    dataset.max_samples = -1
    dataset.apply_chat_template_kwargs = {}
    dataset.tokenizer = DummyTokenizer()
    dataset.processor = DummyProcessor()
    dataset.messages = [
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "look"},
                    {"type": "image", "image": "dummy-image"},
                ],
                "loss_mask": 0,
            },
            {
                "role": "assistant",
                "content": "<think>go</think><|latent|><|action_start|><|move_forward|>",
                "loss_mask": 1,
            },
        ]
    ]
    dataset.enable_thinking = [False]
    dataset._collect_images = types.MethodType(
        lambda self, messages: [object()] if any(isinstance(msg.get("content"), list) for msg in messages) else [],
        dataset,
    )

    sample = dataset[0]

    full_text = dataset._apply_chat_template(dataset.messages[0], enable_thinking=False, add_generation_prompt=False)
    text_only_token_count = len(dataset.tokenizer.encode(full_text, add_special_tokens=False))

    assert sample["input_ids"].shape[0] > text_only_token_count
    assert sample["input_ids"].shape[0] == sample["loss_mask"].shape[0]
    assert sample["input_ids"].shape[0] == sample["position_ids"].shape[-1]
    assert int(sample["loss_mask"].sum().item()) > 0
    assert "attention_mask" not in sample
