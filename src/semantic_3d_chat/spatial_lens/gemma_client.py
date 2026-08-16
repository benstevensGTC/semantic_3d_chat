"""A thin, local, zero-training interface to Gemma-4 for images and text.

The rest of this package needs exactly two things from the model: "look at this
picture and answer", and "read this text and answer".  Neither trains anything;
both run entirely from the pinned local snapshot.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

MODEL_ID = "google/gemma-4-E2B-it"
MODEL_REVISION = "3e22461f65e89153144f8adb70e3b8c2cc9845a7"


@dataclass
class GemmaChat:
    """Local Gemma-4, used as an ordinary multimodal chat model."""

    model: Any
    processor: Any
    device: Any

    @classmethod
    def load(cls, *, device: str | None = None) -> GemmaChat:
        import torch
        from transformers import AutoProcessor, Gemma4ForConditionalGeneration

        if device is not None:
            selected = torch.device(device)
        elif torch.backends.mps.is_available():
            selected = torch.device("mps")
        else:
            selected = torch.device("cpu")
        processor = AutoProcessor.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION, local_files_only=True
        )
        model = Gemma4ForConditionalGeneration.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            dtype=torch.bfloat16,
            local_files_only=True,
            low_cpu_mem_usage=True,
            device_map={"": selected},
        )
        model.requires_grad_(False)
        model.eval()
        return cls(model=model, processor=processor, device=selected)

    def _generate(self, conversation: list[dict[str, Any]], max_new_tokens: int) -> str:
        import torch

        inputs = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            enable_thinking=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        moved = {
            key: (
                value.to(device=self.device, dtype=torch.bfloat16)
                if torch.is_floating_point(value)
                else value.to(self.device)
            )
            for key, value in inputs.items()
        }
        prompt_length = moved["input_ids"].shape[1]
        with torch.inference_mode():
            output = self.model.generate(
                **moved,
                do_sample=False,
                max_new_tokens=int(max_new_tokens),
                pad_token_id=self.processor.tokenizer.pad_token_id,
            )
        return self.processor.decode(
            output[0, prompt_length:].detach().cpu(), skip_special_tokens=True
        ).strip()

    def ask_image(self, image: Any, question: str, *, max_new_tokens: int = 24) -> str:
        return self._generate(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": question},
                    ],
                }
            ],
            max_new_tokens,
        )

    def ask_text(
        self, prompt: str, *, system: str | None = None, max_new_tokens: int = 256
    ) -> str:
        conversation: list[dict[str, Any]] = []
        if system:
            conversation.append(
                {"role": "system", "content": [{"type": "text", "text": system}]}
            )
        conversation.append(
            {"role": "user", "content": [{"type": "text", "text": prompt}]}
        )
        return self._generate(conversation, max_new_tokens)


__all__ = ["MODEL_ID", "MODEL_REVISION", "GemmaChat"]
