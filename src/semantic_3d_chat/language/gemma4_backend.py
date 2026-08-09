"""Gemma 4 continuous-prefix preparation and cached local decoding.

Gemma 4 E2B uses Per-Layer Embeddings (PLE) in addition to its ordinary token
embedding stream. Arbitrary scene vectors therefore cannot be passed through
``inputs_embeds`` alone. This backend gives each continuous scene vector the
same non-semantic PAD-token PLE identity used by Gemma's native visual soft
tokens, retains real PLE identities for prompt/answer tokens, and performs a
tested custom cached decoding loop.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from semantic_3d_chat.language.prefix_injection import (
    SCENE_BOUNDARY_MODE_GEMMA4_NATIVE_IMAGE,
    SCENE_BOUNDARY_MODE_LEARNED,
    validate_scene_boundary_mode,
)


@dataclass(frozen=True)
class Gemma4PrefixInputs:
    inputs_embeds: torch.Tensor
    attention_mask: torch.Tensor
    per_layer_inputs: torch.Tensor
    mm_token_type_ids: torch.Tensor
    labels: torch.Tensor | None
    scene_prefix_length: int


class Gemma4PrefixBackend:
    """Prepare and decode arbitrary continuous prefixes with explicit PLE."""

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        tokenizer: Any | None = None,
        model_revision: str | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.model_revision = model_revision
        base = getattr(model, "model", None)
        text_model = getattr(base, "language_model", None)
        if text_model is None:
            raise TypeError("Gemma4PrefixBackend requires model.model.language_model")
        self.text_model = text_model
        config = getattr(model, "config", None)
        self.text_config = getattr(config, "text_config", None)
        if self.text_config is None:
            raise TypeError("Gemma 4 model config must expose text_config")
        if not int(getattr(self.text_config, "hidden_size_per_layer_input", 0)):
            raise ValueError("Gemma 4 E2B backend requires per-layer embeddings")
        if getattr(self.text_config, "pad_token_id", None) is None:
            raise ValueError("Gemma 4 E2B backend requires a PAD token ID")

    @property
    def hidden_size(self) -> int:
        return int(self.text_config.hidden_size)

    def _special_token_id(
        self,
        name: str,
        *,
        config_source: Any,
    ) -> int:
        config_value = getattr(config_source, name, None)
        if isinstance(config_value, bool) or not isinstance(config_value, int):
            raise TypeError(f"Loaded Gemma 4 config has no valid {name}")
        if self.tokenizer is not None:
            tokenizer_value = getattr(self.tokenizer, name, None)
            if isinstance(tokenizer_value, bool) or not isinstance(tokenizer_value, int):
                raise ValueError(f"Loaded Gemma 4 tokenizer has no valid {name}")
            if tokenizer_value != config_value:
                raise ValueError(
                    f"Gemma 4 tokenizer/config {name} mismatch: "
                    f"{tokenizer_value} != {config_value}"
                )
        return int(config_value)

    def native_image_contract(self) -> dict[str, object]:
        """Describe model-derived identities for the native visual-token protocol."""

        if not isinstance(self.model_revision, str) or not self.model_revision:
            raise ValueError("Native Gemma boundary mode requires a pinned model revision")
        model_config = self.model.config
        contract: dict[str, object] = {
            "schema_version": 1,
            "model_revision": self.model_revision,
            "bos_token_id": self._special_token_id(
                "bos_token_id", config_source=self.text_config
            ),
            "pad_token_id": self._special_token_id(
                "pad_token_id", config_source=self.text_config
            ),
            "boi_token_id": self._special_token_id(
                "boi_token_id", config_source=model_config
            ),
            "image_token_id": self._special_token_id(
                "image_token_id", config_source=model_config
            ),
            "eoi_token_id": self._special_token_id(
                "eoi_token_id", config_source=model_config
            ),
            "use_bidirectional_attention": getattr(
                self.text_config, "use_bidirectional_attention", None
            ),
        }
        vocabulary_size = int(getattr(self.text_config, "vocab_size", 0))
        for name in (
            "bos_token_id",
            "pad_token_id",
            "boi_token_id",
            "image_token_id",
            "eoi_token_id",
        ):
            token_id = int(contract[name])
            if token_id < 0 or token_id >= vocabulary_size:
                raise ValueError(
                    f"Native Gemma boundary {name}={token_id} is outside vocab_size={vocabulary_size}"
                )
        return contract

    def native_boundary_embeddings(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return exact scaled BOI/EOI embeddings from the loaded checkpoint."""

        contract = self.native_image_contract()
        device = self.model.get_input_embeddings().weight.device
        ids = torch.tensor(
            [[int(contract["boi_token_id"]), int(contract["eoi_token_id"])]],
            dtype=torch.long,
            device=device,
        )
        with torch.no_grad():
            embeddings = self.model.get_input_embeddings()(ids).detach()
        return embeddings[:, :1].clone(), embeddings[:, 1:].clone()

    def _token_embeddings_and_ple(
        self,
        token_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape [B, T]")
        embeddings = self.model.get_input_embeddings()(token_ids)
        ple = self.text_model.get_per_layer_inputs(token_ids, embeddings)
        return embeddings, ple

    def padding_metadata(
        self,
        batch_size: int,
        length: int,
        *,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return native PAD PLE and text-modality IDs for right-padding slots."""

        _, pad_ple, pad_types = self.padding_values(
            batch_size, length, device=device
        )
        return pad_ple, pad_types

    def padding_values(
        self,
        batch_size: int,
        length: int,
        *,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return exact main embedding, PLE, and modality values for PAD slots."""

        if batch_size < 1 or length < 0:
            raise ValueError("batch_size must be positive and padding length non-negative")
        pad_ids = torch.full(
            (batch_size, length),
            int(self.text_config.pad_token_id),
            dtype=torch.long,
            device=device,
        )
        pad_embeddings, pad_ple = self._token_embeddings_and_ple(pad_ids)
        return pad_embeddings, pad_ple, torch.zeros_like(pad_ids)

    def prepare(
        self,
        scene_prefix: torch.Tensor,
        prompt_ids: torch.Tensor,
        answer_ids: torch.Tensor | None = None,
        *,
        scene_prefix_after_bos: bool = False,
        scene_boundary_mode: str = SCENE_BOUNDARY_MODE_LEARNED,
    ) -> Gemma4PrefixInputs:
        """Combine scene and text while preserving Gemma 4's auxiliary PLE stream."""

        if not isinstance(scene_prefix_after_bos, bool):
            raise TypeError("scene_prefix_after_bos must be a boolean")
        scene_boundary_mode = validate_scene_boundary_mode(scene_boundary_mode)
        native_boundaries = scene_boundary_mode == SCENE_BOUNDARY_MODE_GEMMA4_NATIVE_IMAGE
        if native_boundaries and not scene_prefix_after_bos:
            raise ValueError(
                "gemma4_native_image boundary mode requires scene_prefix_after_bos=True"
            )
        if scene_prefix.ndim != 3 or scene_prefix.shape[-1] != self.hidden_size:
            raise ValueError(
                f"scene_prefix must have shape [B, S, {self.hidden_size}], "
                f"got {tuple(scene_prefix.shape)}"
            )
        if prompt_ids.ndim != 2 or prompt_ids.shape[0] != scene_prefix.shape[0]:
            raise ValueError("Prompt and scene batch sizes must match")
        prompt_ids = prompt_ids.to(scene_prefix.device)
        if scene_prefix_after_bos:
            bos_token_id = getattr(self.text_config, "bos_token_id", None)
            if bos_token_id is None:
                raise ValueError("BOS-first scene-prefix layout requires a BOS token ID")
            if prompt_ids.shape[1] < 1 or not torch.all(prompt_ids[:, 0] == int(bos_token_id)):
                observed = (
                    []
                    if prompt_ids.shape[1] == 0
                    else sorted({int(value) for value in prompt_ids[:, 0].detach().cpu()})
                )
                raise ValueError(
                    "BOS-first scene-prefix layout requires every prompt to start with "
                    f"bos_token_id={int(bos_token_id)}; observed={observed}"
                )
        prompt_embeddings, prompt_ple = self._token_embeddings_and_ple(prompt_ids)
        scene_prefix = scene_prefix.to(dtype=prompt_embeddings.dtype)
        if native_boundaries:
            if scene_prefix.shape[1] < 3:
                raise ValueError(
                    "Native Gemma scene prefix must contain BOI, at least one latent, and EOI"
                )
            contract = self.native_image_contract()
            native_start, native_end = self.native_boundary_embeddings()
            native_start = native_start.to(scene_prefix)
            native_end = native_end.to(scene_prefix)
            expected_start = native_start.expand(scene_prefix.shape[0], -1, -1)
            expected_end = native_end.expand(scene_prefix.shape[0], -1, -1)
            if not torch.equal(scene_prefix[:, :1], expected_start):
                raise ValueError("Scene-prefix BOI embedding does not match loaded Gemma")
            if not torch.equal(scene_prefix[:, -1:], expected_end):
                raise ValueError("Scene-prefix EOI embedding does not match loaded Gemma")
            latent_count = scene_prefix.shape[1] - 2
            boundary_ids = torch.tensor(
                [[int(contract["boi_token_id"]), int(contract["eoi_token_id"])]],
                dtype=torch.long,
                device=scene_prefix.device,
            ).expand(scene_prefix.shape[0], -1)
            _, boundary_ple = self._token_embeddings_and_ple(boundary_ids)
            scene_placeholder_ids = torch.full(
                (scene_prefix.shape[0], latent_count),
                int(contract["pad_token_id"]),
                dtype=torch.long,
                device=scene_prefix.device,
            )
            _, latent_ple = self._token_embeddings_and_ple(scene_placeholder_ids)
            scene_ple = torch.cat(
                (boundary_ple[:, :1], latent_ple, boundary_ple[:, 1:]), dim=1
            )
            # Gemma's processor marks only visual soft-token slots as image
            # modality. BOI/EOI remain type 0. E2B currently uses causal masks
            # regardless, but retaining the native IDs makes that assumption
            # explicit and checkpoint-verifiable.
            scene_mm_token_type_ids = torch.cat(
                (
                    torch.zeros(
                        (scene_prefix.shape[0], 1),
                        dtype=torch.long,
                        device=scene_prefix.device,
                    ),
                    torch.ones(
                        (scene_prefix.shape[0], latent_count),
                        dtype=torch.long,
                        device=scene_prefix.device,
                    ),
                    torch.zeros(
                        (scene_prefix.shape[0], 1),
                        dtype=torch.long,
                        device=scene_prefix.device,
                    ),
                ),
                dim=1,
            )
        else:
            scene_placeholder_ids = torch.full(
                scene_prefix.shape[:2],
                int(self.text_config.pad_token_id),
                dtype=torch.long,
                device=scene_prefix.device,
            )
            _, scene_ple = self._token_embeddings_and_ple(scene_placeholder_ids)
            scene_mm_token_type_ids = torch.zeros(
                scene_prefix.shape[:2], dtype=torch.long, device=scene_prefix.device
            )
        if scene_prefix_after_bos:
            embeddings = [prompt_embeddings[:, :1], scene_prefix, prompt_embeddings[:, 1:]]
            ple_parts = [prompt_ple[:, :1], scene_ple, prompt_ple[:, 1:]]
            mm_parts = [
                torch.zeros_like(prompt_ids[:, :1]),
                scene_mm_token_type_ids,
                torch.zeros_like(prompt_ids[:, 1:]),
            ]
        else:
            embeddings = [scene_prefix, prompt_embeddings]
            ple_parts = [scene_ple, prompt_ple]
            mm_parts = [scene_mm_token_type_ids, torch.zeros_like(prompt_ids)]
        labels = None
        if answer_ids is not None:
            if answer_ids.ndim != 2 or answer_ids.shape[0] != scene_prefix.shape[0]:
                raise ValueError("Answer and scene batch sizes must match")
            answer_ids = answer_ids.to(scene_prefix.device)
            answer_embeddings, answer_ple = self._token_embeddings_and_ple(answer_ids)
            embeddings.append(answer_embeddings)
            ple_parts.append(answer_ple)
            mm_parts.append(torch.zeros_like(answer_ids))
            ignored = torch.full(
                (scene_prefix.shape[0], scene_prefix.shape[1] + prompt_ids.shape[1]),
                -100,
                dtype=torch.long,
                device=scene_prefix.device,
            )
            labels = torch.cat((ignored, answer_ids), dim=1)

        inputs_embeds = torch.cat(embeddings, dim=1)
        per_layer_inputs = torch.cat(ple_parts, dim=1)
        mm_token_type_ids = torch.cat(mm_parts, dim=1)
        expected_ple_shape = (
            *inputs_embeds.shape[:2],
            int(self.text_config.num_hidden_layers),
            int(self.text_config.hidden_size_per_layer_input),
        )
        if tuple(per_layer_inputs.shape) != expected_ple_shape:
            raise RuntimeError(
                f"Gemma 4 PLE shape mismatch: {tuple(per_layer_inputs.shape)} "
                f"!= {expected_ple_shape}"
            )
        attention_mask = torch.ones(
            inputs_embeds.shape[:2], dtype=torch.long, device=inputs_embeds.device
        )
        if mm_token_type_ids.shape != attention_mask.shape:
            raise RuntimeError(
                "Gemma 4 modality metadata shape mismatch: "
                f"{tuple(mm_token_type_ids.shape)} != {tuple(attention_mask.shape)}"
            )
        return Gemma4PrefixInputs(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            per_layer_inputs=per_layer_inputs,
            mm_token_type_ids=mm_token_type_ids,
            labels=labels,
            scene_prefix_length=int(scene_prefix.shape[1]),
        )

    def prefill(
        self,
        prepared: Gemma4PrefixInputs,
        *,
        use_cache: bool = True,
    ) -> Any:
        kwargs: dict[str, Any] = {
            "inputs_embeds": prepared.inputs_embeds,
            "per_layer_inputs": prepared.per_layer_inputs,
            "attention_mask": prepared.attention_mask,
            "mm_token_type_ids": prepared.mm_token_type_ids,
            "use_cache": use_cache,
            "return_dict": True,
        }
        if prepared.labels is None:
            kwargs["logits_to_keep"] = 1
        else:
            kwargs["labels"] = prepared.labels
        return self.model(**kwargs)

    def decode_step(
        self,
        token_id: torch.Tensor,
        *,
        past_key_values: Any,
        attention_mask: torch.Tensor,
    ) -> Any:
        """Decode one token while extending an already cached scene prefix."""

        if token_id.shape != (1, 1):
            raise ValueError("Cached interactive decoding requires token_id shape [1, 1]")
        token_id = token_id.to(attention_mask.device)
        embeddings, ple = self._token_embeddings_and_ple(token_id)
        return self.model(
            inputs_embeds=embeddings,
            per_layer_inputs=ple,
            attention_mask=attention_mask,
            mm_token_type_ids=torch.zeros_like(token_id),
            past_key_values=past_key_values,
            use_cache=True,
            logits_to_keep=1,
            return_dict=True,
        )

    @torch.inference_mode()
    def generate(
        self,
        prepared: Gemma4PrefixInputs,
        *,
        max_new_tokens: int,
        eos_token_ids: int | Sequence[int] | None,
    ) -> torch.Tensor:
        """Greedy generation with one prefix prefill followed by cached steps."""

        if prepared.inputs_embeds.shape[0] != 1:
            raise ValueError("Interactive Gemma 4 generation currently supports batch size one")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if prepared.labels is not None:
            raise ValueError("Generation inputs cannot contain teacher-forced answer labels")
        stop_ids: set[int] = set()
        if eos_token_ids is not None:
            stop_ids = (
                {int(eos_token_ids)}
                if isinstance(eos_token_ids, int)
                else {int(value) for value in eos_token_ids}
            )

        output = self.prefill(prepared, use_cache=True)
        next_id = output.logits[:, -1].float().argmax(dim=-1, keepdim=True)
        generated = [next_id]
        past = output.past_key_values
        attention_mask = prepared.attention_mask
        if int(next_id.item()) in stop_ids:
            return next_id
        for _ in range(max_new_tokens - 1):
            attention_mask = torch.cat(
                (
                    attention_mask,
                    torch.ones((1, 1), dtype=attention_mask.dtype, device=attention_mask.device),
                ),
                dim=1,
            )
            output = self.decode_step(
                next_id,
                past_key_values=past,
                attention_mask=attention_mask,
            )
            past = output.past_key_values
            next_id = output.logits[:, -1].float().argmax(dim=-1, keepdim=True)
            generated.append(next_id)
            if int(next_id.item()) in stop_ids:
                break
        return torch.cat(generated, dim=1)


__all__ = ["Gemma4PrefixBackend", "Gemma4PrefixInputs"]
