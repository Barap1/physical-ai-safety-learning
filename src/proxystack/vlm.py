"""SmolVLM2 high-level controller. Greedy decoding, frozen revision."""

from __future__ import annotations

import os

import torch
from PIL import Image

from proxystack.constants import (
    ATTN_IMPLEMENTATION,
    DO_SAMPLE,
    MAX_NEW_TOKENS,
    MODEL_ID,
    MODEL_REVISION,
    TORCH_DTYPE,
)

os.environ.setdefault("MUJOCO_GL", "egl")


class FrozenVLM:
    def __init__(self):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        dtype = getattr(torch, TORCH_DTYPE)
        self.processor = AutoProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
        self.model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            torch_dtype=dtype,
            attn_implementation=ATTN_IMPLEMENTATION,
        ).to("cuda")
        self.model.eval()
        self.dtype = dtype

    def generate(self, image: Image.Image | str, prompt: str) -> str:
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {key: value.to(self.model.device) if hasattr(value, "to") else value for key, value in inputs.items()}
        for key, value in list(inputs.items()):
            if hasattr(value, "dtype") and value.dtype.is_floating_point:
                inputs[key] = value.to(dtype=self.dtype)
        with torch.no_grad():
            generated = self.model.generate(
                **inputs,
                do_sample=DO_SAMPLE,
                max_new_tokens=MAX_NEW_TOKENS,
            )
        prompt_len = inputs["input_ids"].shape[-1]
        new_tokens = generated[:, prompt_len:]
        text = self.processor.batch_decode(new_tokens, skip_special_tokens=True)[0]
        return text.strip()
