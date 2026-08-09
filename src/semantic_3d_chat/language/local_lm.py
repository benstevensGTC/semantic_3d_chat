from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from semantic_3d_chat.device import safe_dtype, select_device
from semantic_3d_chat.language.prefix_injection import (
    SCENE_BOUNDARY_MODE_GEMMA4_NATIVE_IMAGE,
    SCENE_BOUNDARY_MODE_LEARNED,
    validate_scene_boundary_mode,
)


@dataclass
class LocalLanguageModel:
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    device: torch.device
    prefix_backend: Any | None = None
    backend_name: str = "causal_lm"
    decoder_gradient_checkpointing_enabled: bool = False

    @property
    def hidden_size(self) -> int:
        config = self.model.config
        text_config = getattr(config, "text_config", None)
        return int((text_config or config).hidden_size)

    @property
    def bos_token_id(self) -> int | None:
        """Resolve the native text BOS identity without inventing a marker."""

        tokenizer_value = getattr(self.tokenizer, "bos_token_id", None)
        if tokenizer_value is not None:
            return int(tokenizer_value)
        config = self.model.config
        text_config = getattr(config, "text_config", None)
        config_value = getattr(text_config or config, "bos_token_id", None)
        return None if config_value is None else int(config_value)

    @property
    def decoder_module(self) -> torch.nn.Module:
        """Return the text decoder without including Gemma's sensor towers."""

        return (
            self.prefix_backend.text_model
            if self.prefix_backend is not None
            else self.model
        )

    def scene_boundary_contract(self, mode: str) -> dict[str, object] | None:
        """Resolve native identities from the loaded model, never from unchecked constants."""

        mode = validate_scene_boundary_mode(mode)
        if mode == SCENE_BOUNDARY_MODE_LEARNED:
            return None
        if mode == SCENE_BOUNDARY_MODE_GEMMA4_NATIVE_IMAGE:
            if self.prefix_backend is None or self.backend_name != "gemma4":
                raise ValueError(
                    "gemma4_native_image boundary mode requires the Gemma4 prefix backend"
                )
            return self.prefix_backend.native_image_contract()
        raise AssertionError(f"Unhandled scene boundary mode: {mode}")

    def scene_boundary_embeddings(
        self, mode: str
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Return frozen native delimiter embeddings for composer construction."""

        mode = validate_scene_boundary_mode(mode)
        if mode == SCENE_BOUNDARY_MODE_LEARNED:
            return None
        self.scene_boundary_contract(mode)
        return self.prefix_backend.native_boundary_embeddings()

    def generate_from_scene_prefix(
        self,
        scene_prefix: torch.Tensor,
        prompt_ids: torch.Tensor,
        *,
        max_new_tokens: int,
        eos_token_ids: int | Sequence[int] | None,
        scene_prefix_after_bos: bool = False,
        scene_boundary_mode: str = SCENE_BOUNDARY_MODE_LEARNED,
        fallback: Callable[
            [torch.nn.Module, torch.Tensor, torch.Tensor, int, int | Sequence[int] | None],
            torch.Tensor,
        ],
    ) -> torch.Tensor:
        """Dispatch continuous-prefix generation without changing caller semantics."""

        if not isinstance(scene_prefix_after_bos, bool):
            raise TypeError("scene_prefix_after_bos must be a boolean")
        scene_boundary_mode = validate_scene_boundary_mode(scene_boundary_mode)
        if self.prefix_backend is not None:
            prepared = self.prefix_backend.prepare(
                scene_prefix,
                prompt_ids,
                scene_prefix_after_bos=scene_prefix_after_bos,
                scene_boundary_mode=scene_boundary_mode,
            )
            return self.prefix_backend.generate(
                prepared,
                max_new_tokens=max_new_tokens,
                eos_token_ids=eos_token_ids,
            )
        if scene_boundary_mode == SCENE_BOUNDARY_MODE_GEMMA4_NATIVE_IMAGE:
            raise ValueError(
                "gemma4_native_image boundary mode requires the Gemma4 prefix backend"
            )
        prompt_embeddings = self.model.get_input_embeddings()(prompt_ids).to(scene_prefix.dtype)
        if scene_prefix_after_bos:
            bos_token_id = self.bos_token_id
            if bos_token_id is None:
                raise ValueError("BOS-first scene-prefix layout requires a BOS token ID")
            if prompt_ids.shape[1] < 1 or not torch.all(prompt_ids[:, 0] == bos_token_id):
                observed = (
                    []
                    if prompt_ids.shape[1] == 0
                    else sorted({int(value) for value in prompt_ids[:, 0].detach().cpu()})
                )
                raise ValueError(
                    "BOS-first scene-prefix layout requires every prompt to start with "
                    f"bos_token_id={bos_token_id}; observed={observed}"
                )
            inputs_embeds = torch.cat(
                (prompt_embeddings[:, :1], scene_prefix, prompt_embeddings[:, 1:]), dim=1
            )
        else:
            inputs_embeds = torch.cat((scene_prefix, prompt_embeddings), dim=1)
        attention_mask = torch.ones(
            inputs_embeds.shape[:2], dtype=torch.long, device=inputs_embeds.device
        )
        return fallback(
            self.model,
            inputs_embeds,
            attention_mask,
            max_new_tokens,
            eos_token_ids,
        )

    def forward_prefix_batch(self, batch: Any, *, use_cache: bool = False) -> Any:
        """Teacher-force a generic or Gemma 4 variable-length prefix batch."""

        kwargs: dict[str, Any] = {
            "inputs_embeds": batch.inputs_embeds,
            "attention_mask": batch.attention_mask,
            "labels": batch.labels,
            "use_cache": use_cache,
        }
        if self.prefix_backend is not None:
            if batch.per_layer_inputs is None or batch.mm_token_type_ids is None:
                raise ValueError("Gemma 4 prefix batch is missing PLE or modality metadata")
            kwargs["per_layer_inputs"] = batch.per_layer_inputs
            kwargs["mm_token_type_ids"] = batch.mm_token_type_ids
        # Checkpointing is useful only for the backward pass. Pair-gate and
        # validation forwards run under inference_mode; temporarily switching
        # the decoder to eval avoids needless checkpoint wrappers and keeps
        # those measurements deterministic if a future decoder has dropout.
        decoder = self.decoder_module
        bypass_checkpointing = (
            self.decoder_gradient_checkpointing_enabled and not torch.is_grad_enabled()
        )
        was_training = decoder.training
        if bypass_checkpointing:
            decoder.eval()
        try:
            return self.model(**kwargs)
        finally:
            if bypass_checkpointing:
                decoder.train(was_training)

    def enable_decoder_gradient_checkpointing(self) -> None:
        """Checkpoint only the language decoder during frozen-prefix training.

        Gemma 4's conditional-generation wrapper also owns vision and audio
        towers.  Calling the wrapper's generic checkpointing method would mark
        those unused towers too, so the continuous-prefix path deliberately
        targets ``model.language_model`` exposed by the Gemma backend.  Generic
        causal LMs use the whole model as their decoder.

        Transformer checkpointing runs only while decoder layers are in train
        mode.  The parameters stay frozen; train mode merely activates the
        recomputation path.  Non-reentrant checkpointing is required because
        the trainable scene vectors arrive through ``inputs_embeds``.
        """

        decoder = self.decoder_module
        enable = getattr(decoder, "gradient_checkpointing_enable", None)
        if not callable(enable):
            raise TypeError(
                f"{type(decoder).__name__} does not support decoder gradient checkpointing"
            )
        enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        # The continuous scene prefix already requires gradients.  Avoid the
        # extra embedding-output hook installed by Transformers for input-ID
        # and PEFT training; the frozen text embeddings do not need gradients.
        disable_input_grads = getattr(decoder, "disable_input_require_grads", None)
        if callable(disable_input_grads):
            disable_input_grads()
        decoder.train()
        for model_config in (
            getattr(self.model, "config", None),
            getattr(decoder, "config", None),
        ):
            if model_config is not None and hasattr(model_config, "use_cache"):
                model_config.use_cache = False
        self.decoder_gradient_checkpointing_enabled = bool(
            getattr(decoder, "is_gradient_checkpointing", True)
        )
        if not self.decoder_gradient_checkpointing_enabled:
            raise RuntimeError("Decoder did not activate gradient checkpointing")


def load_local_language_model(
    model_id: str,
    revision: str = "main",
    requested_dtype: str = "float16",
    freeze: bool = True,
    local_files_only: bool = False,
    backend: str = "auto",
    decoder_gradient_checkpointing: bool = False,
) -> LocalLanguageModel:
    device = select_device()
    dtype = safe_dtype(device, requested_dtype)
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, revision=revision, local_files_only=local_files_only
    )
    selected_backend = backend.casefold()
    if selected_backend == "auto":
        selected_backend = "gemma4" if "gemma-4" in model_id.casefold() else "causal_lm"
    if selected_backend == "gemma4":
        try:
            from transformers import Gemma4ForConditionalGeneration
        except ImportError as exc:  # pragma: no cover - main baseline uses Transformers 4
            raise RuntimeError(
                "Gemma 4 requires the isolated Transformers 5 environment; "
                "run `make setup-gemma4-probe`"
            ) from exc
        gemma_load_kwargs: dict[str, Any] = {
            "revision": revision,
            "dtype": dtype,
            "local_files_only": local_files_only,
            "low_cpu_mem_usage": True,
        }
        if device.type == "mps":
            # Transformers 5 streams safetensor slices directly to MPS via its
            # pread loader. Loading on CPU and then calling `.to("mps")` can
            # transiently hold two ~10 GiB copies on a 24 GiB unified-memory Mac.
            gemma_load_kwargs["device_map"] = {"": device}
        model = Gemma4ForConditionalGeneration.from_pretrained(
            model_id, **gemma_load_kwargs
        )
        if device.type != "mps":
            model = model.to(device)
        from semantic_3d_chat.language.gemma4_backend import Gemma4PrefixBackend

        prefix_backend: Any | None = Gemma4PrefixBackend(
            model,
            tokenizer=tokenizer,
            model_revision=revision,
        )
    elif selected_backend == "causal_lm":
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            torch_dtype=dtype,
            local_files_only=local_files_only,
        ).to(device)
        prefix_backend = None
    else:
        raise ValueError("language backend must be auto, causal_lm, or gemma4")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if freeze:
        model.requires_grad_(False)
        model.eval()
    language = LocalLanguageModel(
        model=model,
        tokenizer=tokenizer,
        device=device,
        prefix_backend=prefix_backend,
        backend_name=selected_backend,
    )
    if decoder_gradient_checkpointing:
        language.enable_decoder_gradient_checkpointing()
    return language


def prompt_token_ids(
    tokenizer: PreTrainedTokenizerBase,
    system_prompt: str,
    question: str,
    device: torch.device,
) -> torch.Tensor:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        encoded = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        )
        # Transformers 4 returns a tensor for several tokenizers, while
        # Transformers 5 may return a BatchEncoding from the same call.
        ids = (
            encoded.input_ids
            if hasattr(encoded, "input_ids")
            else encoded["input_ids"]
            if isinstance(encoded, dict)
            else encoded
        )
    else:
        ids = tokenizer(
            f"System: {system_prompt}\nUser: {question}\nAssistant:", return_tensors="pt"
        ).input_ids
    if ids.ndim != 2 or ids.shape[1] == 0:
        raise ValueError("Prompt tokenizer returned no tokens")
    return ids.to(device)


def question_token_ids(
    tokenizer: PreTrainedTokenizerBase,
    question: str,
    device: torch.device,
) -> torch.Tensor:
    """Tokenize only user text for the auxiliary grounding query.

    The causal-LM prompt still contains the stable system instruction.  Keeping
    that shared text out of this auxiliary representation prevents it from
    overwhelming the object/relation words that must select a spatial latent.
    """

    encoded = tokenizer(question, add_special_tokens=False, return_tensors="pt")
    ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    if ids.ndim != 2 or ids.shape[1] == 0:
        raise ValueError("Question tokenizer returned no tokens")
    return ids.to(device)
