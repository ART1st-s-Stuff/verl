# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
SFT dataset
- We assume user pass a single parquet file.
- We load all the data into the memory.
Each parquet file contains
"""

import numpy as np
import pandas as pd
import torch
from PIL import Image
from omegaconf.listconfig import ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.utils.model import compute_position_id_with_mask


class SFTDataset(Dataset):
    """
    This is an in-memory SFTDataset

    Arguments:
        config (OmegaConf): the data config
    """

    def __init__(self, parquet_files: str | ListConfig, tokenizer, config, max_samples: int = -1):
        prompt_key = config.get("prompt_key", "prompt")
        prompt_dict_keys = config.get("prompt_dict_keys", None)
        response_key = config.get("response_key", "response")
        response_dict_keys = config.get("response_dict_keys", None)
        max_length = config.get("max_length", 1024)
        truncation = config.get("truncation", "error")
        use_shm = config.get("use_shm", False)
        self.shuffle = config.get("shuffle", False)
        self.seed = config.get("seed")
        self.apply_chat_template_kwargs = config.get("apply_chat_template_kwargs", {})
        self.latent_sft_cfg = config.get("latent_sft", {})
        self.latent_sft_enable = bool(self.latent_sft_cfg.get("enable", False))

        assert truncation in ["error", "left", "right"]
        self.truncation = truncation
        self.use_shm = use_shm

        if not isinstance(parquet_files, ListConfig):
            parquet_files = [parquet_files]

        self.parquet_files = parquet_files
        self.max_samples = max_samples
        if isinstance(tokenizer, str):
            tokenizer = hf_tokenizer(tokenizer)
        self.tokenizer: PreTrainedTokenizer = tokenizer

        self.prompt_key = prompt_key if isinstance(prompt_key, tuple | list) else [prompt_key]
        self.response_key = response_key if isinstance(response_key, tuple | list) else [response_key]
        self.prompt_dict_keys = prompt_dict_keys if prompt_dict_keys else []
        self.response_dict_keys = response_dict_keys if response_dict_keys else []

        self.max_length = max_length

        self._download()
        self._read_files_and_tokenize()

        self._clip_processor = None
        self._clip_model = None
        self._mae_processor = None
        self._mae_model = None

    def _download(self):
        for i, parquet_file in enumerate(self.parquet_files):
            self.parquet_files[i] = copy_to_local(parquet_file, verbose=True, use_shm=self.use_shm)

    def _read_files_and_tokenize(self):
        def series_to_item(ls):
            import numpy
            import pandas

            while isinstance(ls, pandas.core.series.Series | numpy.ndarray) and len(ls) == 1:
                ls = ls[0]
            return ls

        dataframes = []
        for parquet_file in self.parquet_files:
            # read parquet files and cache
            dataframe = pd.read_parquet(parquet_file)
            dataframes.append(dataframe)
        self.dataframe = pd.concat(dataframes)

        total = len(self.dataframe)
        print(f"dataset len: {len(self.dataframe)}")

        if self.max_samples > 0 and self.max_samples < total:
            if self.shuffle:
                rngs_args = (self.seed,) if self.seed is not None else ()
                rng = np.random.default_rng(*rngs_args)
                indices = rng.choice(total, size=self.max_samples, replace=False)
            else:
                indices = np.arange(self.max_samples)
            self.dataframe = self.dataframe.iloc[indices.tolist()]
            print(f"selected {self.max_samples} random samples out of {total}")

        self.prompts = self.dataframe[self.prompt_key]
        for key in self.prompt_dict_keys:
            # type(x): pandas.core.series.Series
            # type(x[0]): numpy.ndarray
            # type(x[0][0]): dict
            try:
                self.prompts = self.prompts.apply(lambda x: series_to_item(x)[key], axis=1)  # noqa: B023
            except Exception:
                print(f"self.prompts={self.prompts}")
                raise
        if isinstance(self.prompts, pd.DataFrame):
            self.prompts = self.prompts.squeeze()
        self.prompts = self.prompts.tolist()
        self.responses = self.dataframe[self.response_key]
        for key in self.response_dict_keys:
            try:
                self.responses = self.responses.apply(lambda x: series_to_item(x)[key], axis=1)  # noqa: B023
            except Exception:
                print(f"self.responses={self.responses}")
                raise
        if isinstance(self.responses, pd.DataFrame):
            self.responses = self.responses.squeeze()
        self.responses = self.responses.tolist()

        if self.latent_sft_enable:
            self.clip_gt_col = self.dataframe["clip_gt"] if "clip_gt" in self.dataframe.columns else None
            self.mae_gt_col = self.dataframe["mae_gt"] if "mae_gt" in self.dataframe.columns else None
            self.image_path_col = self.dataframe["image_path"] if "image_path" in self.dataframe.columns else None
            self.image_bytes_col = self.dataframe["image_bytes"] if "image_bytes" in self.dataframe.columns else None

    def _lazy_init_feature_models(self):
        if self._clip_model is not None and self._mae_model is not None:
            return
        clip_name = self.latent_sft_cfg.get("clip_model_name", "openai/clip-vit-base-patch32")
        mae_name = self.latent_sft_cfg.get("mae_model_name", "facebook/vit-mae-base")
        from transformers import CLIPModel, CLIPProcessor, ViTMAEModel, ViTImageProcessor

        self._clip_processor = CLIPProcessor.from_pretrained(clip_name)
        self._clip_model = CLIPModel.from_pretrained(clip_name).eval()
        self._mae_processor = ViTImageProcessor.from_pretrained(mae_name)
        self._mae_model = ViTMAEModel.from_pretrained(mae_name).eval()

    def _compute_gt_features_from_image(self, image: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        self._lazy_init_feature_models()
        with torch.no_grad():
            clip_inputs = self._clip_processor(images=image, return_tensors="pt")
            clip_feat = self._clip_model.get_image_features(**clip_inputs).squeeze(0).to(torch.float32)
            clip_feat = torch.nn.functional.normalize(clip_feat, dim=-1)

            mae_inputs = self._mae_processor(images=image, return_tensors="pt")
            mae_out = self._mae_model(**mae_inputs)
            mae_feat = mae_out.last_hidden_state.mean(dim=1).squeeze(0).to(torch.float32)
        return clip_feat, mae_feat

    def _try_load_image(self, item: int) -> Image.Image | None:
        if self.image_path_col is not None:
            path = self.image_path_col.iloc[item]
            if isinstance(path, str) and path:
                return Image.open(path).convert("RGB")
        if self.image_bytes_col is not None:
            img_bytes = self.image_bytes_col.iloc[item]
            if img_bytes is not None:
                import io

                return Image.open(io.BytesIO(img_bytes)).convert("RGB")
        return None

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, item):
        tokenizer = self.tokenizer

        prompt = self.prompts[item]
        response = self.responses[item]

        # apply chat template
        prompt_chat = [{"role": "user", "content": prompt}]

        # string
        prompt_chat_str = tokenizer.apply_chat_template(
            prompt_chat, add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs
        )
        response_chat_str = response + tokenizer.eos_token

        # tokenize
        prompt_ids_output = tokenizer(prompt_chat_str, return_tensors="pt", add_special_tokens=False)
        prompt_ids = prompt_ids_output["input_ids"][0]
        prompt_attention_mask = prompt_ids_output["attention_mask"][0]

        response_ids_output = tokenizer(response_chat_str, return_tensors="pt", add_special_tokens=False)
        response_ids = response_ids_output["input_ids"][0]
        response_attention_mask = response_ids_output["attention_mask"][0]

        prompt_length = prompt_ids.shape[0]
        response_length = response_ids.shape[0]

        input_ids = torch.cat((prompt_ids, response_ids), dim=-1)
        attention_mask = torch.cat((prompt_attention_mask, response_attention_mask), dim=-1)

        # padding to max length
        sequence_length = input_ids.shape[0]
        if sequence_length < self.max_length:
            padded_input_ids = (
                torch.ones(size=(self.max_length - sequence_length,), dtype=input_ids.dtype)
                * self.tokenizer.pad_token_id
            )
            padded_attention_mask = torch.zeros(size=(self.max_length - sequence_length,), dtype=attention_mask.dtype)

            input_ids = torch.cat((input_ids, padded_input_ids))
            attention_mask = torch.cat((attention_mask, padded_attention_mask))
        elif sequence_length > self.max_length:
            if self.truncation == "left":
                # actually, left truncation may not be reasonable
                input_ids = input_ids[-self.max_length :]
                attention_mask = attention_mask[-self.max_length :]
            elif self.truncation == "right":
                input_ids = input_ids[: self.max_length]
                attention_mask = attention_mask[: self.max_length]
            elif self.truncation == "error":
                raise NotImplementedError(f"{sequence_length=} is larger than {self.max_length=}")
            else:
                raise NotImplementedError(f"Unknown truncation method {self.truncation}")

        position_ids = compute_position_id_with_mask(attention_mask)

        loss_mask = attention_mask.clone()
        if prompt_length > 1:
            # mask out prompt for SFT.
            loss_mask[: min(prompt_length, loss_mask.size(0)) - 1] = 0
        # mask out the last token in response
        loss_mask[min(prompt_length + response_length, loss_mask.size(0)) - 1] = 0

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "loss_mask": loss_mask,
        } | self._get_latent_supervision_fields(item)

    def _get_latent_supervision_fields(self, item: int) -> dict[str, torch.Tensor]:
        if not self.latent_sft_enable:
            return {}
        clip_gt = None
        mae_gt = None
        if self.clip_gt_col is not None:
            raw = self.clip_gt_col.iloc[item]
            if raw is not None and not (isinstance(raw, float) and np.isnan(raw)):
                clip_gt = torch.tensor(raw, dtype=torch.float32)
        if self.mae_gt_col is not None:
            raw = self.mae_gt_col.iloc[item]
            if raw is not None and not (isinstance(raw, float) and np.isnan(raw)):
                mae_gt = torch.tensor(raw, dtype=torch.float32)

        hybrid_fill = bool(self.latent_sft_cfg.get("hybrid_fill", {}).get("enable", False))
        if (clip_gt is None or mae_gt is None) and hybrid_fill:
            image = self._try_load_image(item)
            if image is not None:
                clip_gt, mae_gt = self._compute_gt_features_from_image(image)

        if clip_gt is None or mae_gt is None:
            # Keep batch shape valid; mask out in loss.
            clip_gt = torch.zeros(self.latent_sft_cfg.get("clip_feature_dim", 512), dtype=torch.float32)
            mae_gt = torch.zeros(self.latent_sft_cfg.get("mae_feature_dim", 768), dtype=torch.float32)
            valid = torch.tensor(0.0, dtype=torch.float32)
        else:
            valid = torch.tensor(1.0, dtype=torch.float32)
        return {"clip_gt": clip_gt, "mae_gt": mae_gt, "latent_gt_valid": valid}
