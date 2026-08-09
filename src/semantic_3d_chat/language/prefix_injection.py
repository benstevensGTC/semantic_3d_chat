from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import nn

SCENE_BOUNDARY_MODE_LEARNED = "learned"
SCENE_BOUNDARY_MODE_GEMMA4_NATIVE_IMAGE = "gemma4_native_image"
SCENE_BOUNDARY_MODES = frozenset(
    {SCENE_BOUNDARY_MODE_LEARNED, SCENE_BOUNDARY_MODE_GEMMA4_NATIVE_IMAGE}
)
_NATIVE_GEMMA4_CONTRACT_FIELDS = frozenset(
    {
        "schema_version",
        "model_revision",
        "bos_token_id",
        "pad_token_id",
        "boi_token_id",
        "image_token_id",
        "eoi_token_id",
        "use_bidirectional_attention",
    }
)


def validate_scene_boundary_mode(value: object) -> str:
    """Validate the checkpointed scene-boundary protocol enum."""

    if not isinstance(value, str) or value not in SCENE_BOUNDARY_MODES:
        raise ValueError(
            "language.scene_boundary_mode must be one of "
            f"{sorted(SCENE_BOUNDARY_MODES)}; got {value!r}"
        )
    return value


def scene_boundary_mode_setting(config: dict) -> str:
    """Return the strict boundary protocol, defaulting old configs to learned markers."""

    value = config.get("language", {}).get(
        "scene_boundary_mode", SCENE_BOUNDARY_MODE_LEARNED
    )
    return validate_scene_boundary_mode(value)


def native_gemma4_image_contract_setting(config: dict) -> dict[str, object] | None:
    """Validate the explicit native-Gemma protocol expected by an opt-in config.

    The values are experimental protocol assertions, not runtime defaults.  The
    loaded model and tokenizer must independently reproduce every value before
    training or inference can use the native boundary mode.
    """

    mode = scene_boundary_mode_setting(config)
    language = config.get("language", {})
    raw = language.get("gemma4_native_image_contract")
    if mode == SCENE_BOUNDARY_MODE_LEARNED:
        if raw is not None:
            raise ValueError(
                "language.gemma4_native_image_contract is only valid when "
                "scene_boundary_mode is gemma4_native_image"
            )
        return None
    if str(language.get("backend", "auto")).casefold() != "gemma4":
        raise ValueError("gemma4_native_image boundary mode requires language.backend=gemma4")
    if not scene_prefix_after_bos_setting(config):
        raise ValueError("gemma4_native_image boundary mode requires scene_prefix_after_bos=true")
    if not isinstance(raw, dict):
        raise TypeError(
            "gemma4_native_image boundary mode requires a strict "
            "language.gemma4_native_image_contract mapping"
        )
    missing = sorted(_NATIVE_GEMMA4_CONTRACT_FIELDS - raw.keys())
    unexpected = sorted(raw.keys() - _NATIVE_GEMMA4_CONTRACT_FIELDS)
    if missing or unexpected:
        raise ValueError(
            "Invalid gemma4_native_image_contract fields: "
            f"missing={missing} unexpected={unexpected}"
        )
    contract = dict(raw)
    if contract["schema_version"] != 1:
        raise ValueError("gemma4_native_image_contract.schema_version must be 1")
    revision = language.get("revision")
    if not isinstance(revision, str) or not revision:
        raise ValueError("Native Gemma boundary mode requires a pinned language revision")
    if contract["model_revision"] != revision:
        raise ValueError(
            "gemma4_native_image_contract.model_revision must exactly match "
            "language.revision"
        )
    for key in (
        "bos_token_id",
        "pad_token_id",
        "boi_token_id",
        "image_token_id",
        "eoi_token_id",
    ):
        value = contract[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"gemma4_native_image_contract.{key} must be a nonnegative int")
    attention_mode = contract["use_bidirectional_attention"]
    if attention_mode is not None and not isinstance(attention_mode, str):
        raise ValueError(
            "gemma4_native_image_contract.use_bidirectional_attention must be null or a string"
        )
    return contract


def scene_boundary_contract_mismatch(
    metadata: dict,
    runtime_mode: str,
    runtime_native_contract: dict[str, object] | None,
) -> dict[str, object] | None:
    """Compare checkpoint and runtime boundary identities with legacy compatibility."""

    runtime_mode = validate_scene_boundary_mode(runtime_mode)
    checkpoint_mode = metadata.get("scene_boundary_mode", SCENE_BOUNDARY_MODE_LEARNED)
    if checkpoint_mode != runtime_mode:
        return {
            "checkpoint": (
                "<missing; legacy learned>"
                if "scene_boundary_mode" not in metadata
                else checkpoint_mode
            ),
            "runtime": runtime_mode,
        }
    checkpoint_contract = metadata.get("gemma4_native_image_contract")
    if runtime_mode == SCENE_BOUNDARY_MODE_GEMMA4_NATIVE_IMAGE:
        if checkpoint_contract != runtime_native_contract:
            return {"checkpoint": checkpoint_contract, "runtime": runtime_native_contract}
    elif checkpoint_contract is not None:
        return {"checkpoint": checkpoint_contract, "runtime": None}
    return None


def scene_prefix_after_bos_setting(config: dict) -> bool:
    """Return the strict, legacy-compatible continuous-prefix layout setting."""

    enabled = config.get("language", {}).get("scene_prefix_after_bos", False)
    if not isinstance(enabled, bool):
        raise TypeError("language.scene_prefix_after_bos must be a boolean")
    return enabled


def scene_prefix_after_bos_contract_mismatch(
    metadata: dict,
    runtime_enabled: bool,
) -> dict[str, object] | None:
    """Compare a checkpoint layout with runtime while accepting legacy false.

    Checkpoints written before this field existed used the scene-before-BOS
    layout exclusively, so a missing field is equivalent to ``False``. Missing
    metadata can never opt a checkpoint into the new BOS-first layout.
    """

    if not isinstance(runtime_enabled, bool):
        raise TypeError("runtime scene_prefix_after_bos must be a boolean")
    if "scene_prefix_after_bos" not in metadata:
        if not runtime_enabled:
            return None
        return {"checkpoint": "<missing; legacy false>", "runtime": True}
    checkpoint_enabled = metadata["scene_prefix_after_bos"]
    if not isinstance(checkpoint_enabled, bool) or checkpoint_enabled != runtime_enabled:
        return {"checkpoint": checkpoint_enabled, "runtime": runtime_enabled}
    return None


def _prepare_control_tokens(
    control_tokens: torch.Tensor | None,
    *,
    batch_size: int,
    hidden_size: int,
    reference: torch.Tensor,
) -> torch.Tensor | None:
    """Validate and align optional decoder-side continuous control tokens."""

    if control_tokens is None:
        return None
    if not isinstance(control_tokens, torch.Tensor):
        raise TypeError("control_tokens must be a tensor with shape [B,C,H]")
    if control_tokens.ndim != 3:
        raise ValueError("control_tokens must have shape [B,C,H]")
    if control_tokens.shape[0] != batch_size:
        raise ValueError("Control-token and scene batch sizes must match")
    if control_tokens.shape[-1] != hidden_size:
        raise ValueError(
            f"control_tokens hidden size must be {hidden_size}; "
            f"got {control_tokens.shape[-1]}"
        )
    prepared = control_tokens.to(device=reference.device, dtype=reference.dtype)
    if not bool(torch.isfinite(prepared).all().item()):
        raise ValueError("control_tokens must contain only finite values")
    return prepared


@dataclass
class PrefixBatch:
    inputs_embeds: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor | None
    scene_prefix_length: int
    per_layer_inputs: torch.Tensor | None = None
    mm_token_type_ids: torch.Tensor | None = None


class ContinuousPrefixComposer(nn.Module):
    """Compose a question-independent scene prefix with optional BOS anchoring.

    The legacy/default layout places the complete continuous prefix before the
    text prompt.  ``scene_prefix_after_bos`` instead preserves the prompt's
    native BOS embedding as position zero and inserts the scene memory
    immediately after it. Learned boundaries remain the default. The opt-in
    Gemma-native mode stores exact, frozen BOI/EOI embeddings as buffers, never
    as trainable parameters; its backend supplies the corresponding PLE stream.
    """

    def __init__(
        self,
        hidden_size: int,
        *,
        scene_prefix_after_bos: bool = False,
        bos_token_id: int | None = None,
        scene_boundary_mode: str = SCENE_BOUNDARY_MODE_LEARNED,
        native_boundary_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(scene_prefix_after_bos, bool):
            raise TypeError("scene_prefix_after_bos must be a boolean")
        self.scene_boundary_mode = validate_scene_boundary_mode(scene_boundary_mode)
        if (
            self.scene_boundary_mode == SCENE_BOUNDARY_MODE_GEMMA4_NATIVE_IMAGE
            and not scene_prefix_after_bos
        ):
            raise ValueError(
                "gemma4_native_image boundary mode requires scene_prefix_after_bos=True"
            )
        if self.scene_boundary_mode == SCENE_BOUNDARY_MODE_LEARNED:
            if native_boundary_embeddings is not None:
                raise ValueError("Native boundary embeddings require gemma4_native_image mode")
            self.scene_start = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)
            self.scene_end = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)
        else:
            if native_boundary_embeddings is None:
                start = torch.zeros(1, 1, hidden_size)
                end = torch.zeros(1, 1, hidden_size)
                initialized = False
            else:
                if len(native_boundary_embeddings) != 2:
                    raise ValueError("native_boundary_embeddings must contain BOI and EOI")
                start, end = native_boundary_embeddings
                expected = (1, 1, hidden_size)
                if tuple(start.shape) != expected or tuple(end.shape) != expected:
                    raise ValueError(
                        "Native boundary embeddings must each have shape "
                        f"{expected}; got {tuple(start.shape)} and {tuple(end.shape)}"
                    )
                start = start.detach().clone()
                end = end.detach().clone()
                initialized = True
            self.register_buffer("scene_start", start, persistent=True)
            self.register_buffer("scene_end", end, persistent=True)
            self.register_buffer(
                "_native_boundaries_initialized",
                torch.tensor(initialized, dtype=torch.bool),
                persistent=True,
            )
        self.scene_prefix_after_bos = scene_prefix_after_bos
        self.bos_token_id = None if bos_token_id is None else int(bos_token_id)

    def _require_native_boundaries(self) -> None:
        if self.scene_boundary_mode != SCENE_BOUNDARY_MODE_GEMMA4_NATIVE_IMAGE:
            return
        if not bool(self._native_boundaries_initialized.item()):
            raise RuntimeError(
                "Native Gemma boundary buffers are uninitialized; load a v7 checkpoint "
                "or construct them from the validated local model"
            )

    def validate_native_boundary_embeddings(
        self, expected: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Require checkpoint buffers to equal the loaded model's scaled embeddings."""

        if self.scene_boundary_mode != SCENE_BOUNDARY_MODE_GEMMA4_NATIVE_IMAGE:
            raise ValueError("Native boundary validation requires gemma4_native_image mode")
        self._require_native_boundaries()
        actual = (self.scene_start, self.scene_end)
        for name, value, target in zip(("BOI", "EOI"), actual, expected, strict=True):
            target = target.to(device=value.device, dtype=value.dtype)
            if value.shape != target.shape or not torch.equal(value, target):
                raise ValueError(
                    f"Checkpoint {name} boundary embedding does not exactly match loaded Gemma"
                )

    def _validate_bos_first_prompt(self, prompt_ids: torch.Tensor) -> None:
        if not self.scene_prefix_after_bos:
            return
        if self.bos_token_id is None:
            raise ValueError("BOS-first scene-prefix layout requires a BOS token ID")
        if prompt_ids.shape[1] < 1:
            raise ValueError("BOS-first scene-prefix layout requires a nonempty prompt")
        if not torch.all(prompt_ids[:, 0] == self.bos_token_id):
            observed = sorted({int(value) for value in prompt_ids[:, 0].detach().cpu()})
            raise ValueError(
                "BOS-first scene-prefix layout requires every prompt to start with "
                f"bos_token_id={self.bos_token_id}; observed={observed}"
            )

    def scene_prefix(self, scene_tokens: torch.Tensor) -> torch.Tensor:
        if scene_tokens.ndim != 3:
            raise ValueError("scene_tokens must have shape [B,L,H]")
        self._require_native_boundaries()
        batch = scene_tokens.shape[0]
        return torch.cat(
            (
                self.scene_start.expand(batch, -1, -1).to(scene_tokens),
                scene_tokens,
                self.scene_end.expand(batch, -1, -1).to(scene_tokens),
            ),
            dim=1,
        )

    def compose(
        self,
        scene_tokens: torch.Tensor,
        prompt_ids: torch.Tensor,
        embedding_layer: nn.Module,
        answer_ids: torch.Tensor | None = None,
        prefix_backend: object | None = None,
        control_tokens: torch.Tensor | None = None,
    ) -> PrefixBatch:
        if prompt_ids.ndim != 2 or prompt_ids.shape[0] != scene_tokens.shape[0]:
            raise ValueError("Prompt and scene batch sizes must match")
        self._validate_bos_first_prompt(prompt_ids)
        prefix = self.scene_prefix(scene_tokens)
        control_tokens = _prepare_control_tokens(
            control_tokens,
            batch_size=scene_tokens.shape[0],
            hidden_size=prefix.shape[-1],
            reference=prefix,
        )
        if prefix_backend is not None:
            backend_options = {
                "scene_prefix_after_bos": self.scene_prefix_after_bos,
                "scene_boundary_mode": self.scene_boundary_mode,
            }
            if control_tokens is not None:
                backend_options["control_tokens"] = control_tokens
            prepared = prefix_backend.prepare(prefix, prompt_ids, answer_ids, **backend_options)
            return PrefixBatch(
                inputs_embeds=prepared.inputs_embeds,
                attention_mask=prepared.attention_mask,
                labels=prepared.labels,
                scene_prefix_length=prepared.scene_prefix_length,
                per_layer_inputs=prepared.per_layer_inputs,
                mm_token_type_ids=prepared.mm_token_type_ids,
            )
        if self.scene_boundary_mode == SCENE_BOUNDARY_MODE_GEMMA4_NATIVE_IMAGE:
            raise ValueError(
                "gemma4_native_image boundary mode requires the Gemma4 prefix backend"
            )
        prompt_embeddings = embedding_layer(prompt_ids).to(prefix.dtype)
        parts = (
            [prompt_embeddings[:, :1], prefix, prompt_embeddings[:, 1:]]
            if self.scene_prefix_after_bos
            else [prefix, prompt_embeddings]
        )
        if control_tokens is not None:
            parts.append(control_tokens)
        labels = None
        if answer_ids is not None:
            if answer_ids.ndim != 2 or answer_ids.shape[0] != scene_tokens.shape[0]:
                raise ValueError("Answer and scene batch sizes must match")
            answer_embeddings = embedding_layer(answer_ids).to(prefix.dtype)
            parts.append(answer_embeddings)
            ignored = torch.full(
                (
                    scene_tokens.shape[0],
                    prefix.shape[1]
                    + prompt_ids.shape[1]
                    + (0 if control_tokens is None else control_tokens.shape[1]),
                ),
                -100,
                dtype=torch.long,
                device=scene_tokens.device,
            )
            labels = torch.cat((ignored, answer_ids), dim=1)
        inputs = torch.cat(parts, dim=1)
        attention_mask = torch.ones(inputs.shape[:2], dtype=torch.long, device=inputs.device)
        return PrefixBatch(inputs, attention_mask, labels, prefix.shape[1])


def stack_prefix_batches(
    batches: Sequence[PrefixBatch],
    device: torch.device,
    *,
    prefix_backend: object | None = None,
) -> PrefixBatch:
    """Right-pad independently tokenized prefixes without losing model metadata.

    Gemma 4 E2B consumes a token-identity Per-Layer Embedding (PLE) stream in
    addition to ``inputs_embeds``. Padding either stream with arbitrary zeros is
    not equivalent to the model's PAD token because the learned PAD row need
    not be zero. Its backend therefore supplies exact native PAD main/PLE
    values for masked positions. Generic causal models keep both auxiliary
    fields absent and follow their original path.
    """

    if not batches:
        raise ValueError("Cannot stack an empty prefix batch")
    if any(batch.inputs_embeds.shape[0] != 1 for batch in batches):
        raise ValueError("Each independently composed prefix must have batch size one")
    has_ple = [batch.per_layer_inputs is not None for batch in batches]
    has_mm_ids = [batch.mm_token_type_ids is not None for batch in batches]
    if len(set(has_ple)) != 1 or len(set(has_mm_ids)) != 1 or has_ple != has_mm_ids:
        raise ValueError("Prefix batches must consistently provide both PLE and modality IDs")
    has_backend_metadata = has_ple[0]
    if has_backend_metadata and prefix_backend is None:
        raise ValueError("A backend is required to create native PLE padding metadata")
    if not has_backend_metadata and prefix_backend is not None:
        raise ValueError("Backend selected but composed prefix has no backend metadata")
    scene_prefix_lengths = {int(batch.scene_prefix_length) for batch in batches}
    if len(scene_prefix_lengths) != 1:
        raise ValueError("All examples in a batch must use the same scene-prefix length")
    if any(batch.labels is None for batch in batches):
        raise ValueError("Teacher-forced prefix batches require answer labels")

    max_length = max(batch.inputs_embeds.shape[1] for batch in batches)
    hidden = batches[0].inputs_embeds.shape[-1]
    dtype = batches[0].inputs_embeds.dtype
    inputs = torch.zeros(len(batches), max_length, hidden, dtype=dtype, device=device)
    masks = torch.zeros(len(batches), max_length, dtype=torch.long, device=device)
    labels = torch.full((len(batches), max_length), -100, dtype=torch.long, device=device)
    per_layer_inputs = None
    mm_token_type_ids = None
    if has_backend_metadata:
        padding_inputs, per_layer_inputs, mm_token_type_ids = prefix_backend.padding_values(
            len(batches), max_length, device=device
        )
        inputs.copy_(padding_inputs.to(dtype=dtype))
    for index, batch in enumerate(batches):
        length = batch.inputs_embeds.shape[1]
        inputs[index, :length] = batch.inputs_embeds[0]
        masks[index, :length] = batch.attention_mask[0]
        labels[index, :length] = batch.labels[0]
        if has_backend_metadata:
            per_layer_inputs[index, :length] = batch.per_layer_inputs[0]
            mm_token_type_ids[index, :length] = batch.mm_token_type_ids[0]
    return PrefixBatch(
        inputs_embeds=inputs,
        attention_mask=masks,
        labels=labels,
        scene_prefix_length=scene_prefix_lengths.pop(),
        per_layer_inputs=per_layer_inputs,
        mm_token_type_ids=mm_token_type_ids,
    )


def prefix_sha256(prefix: torch.Tensor) -> str:
    import hashlib

    canonical = prefix.detach().float().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(canonical).hexdigest()
