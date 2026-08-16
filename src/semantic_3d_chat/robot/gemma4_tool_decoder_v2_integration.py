"""Production installation seam for a promoted Gemma tool-decoder V2.

The static V54 runtime owns the large local Gemma model and its six frozen QA
LoRA banks.  This module installs only the disjoint final-MLP tool adapter into
that already resident model, strictly loads the promoted two-file checkpoint,
and keeps the adapter inactive outside action generation.  Ordinary room QA
therefore retains the exact V54 decoder function.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

import torch
from torch import nn

from semantic_3d_chat.chat.file_audit import FileAccessAudit
from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.language.gemma4_tool_decoder_v2 import (
    CLEARANCE_TOKEN_COUNT,
    HIDDEN_SIZE,
    LORA_ALPHA,
    LORA_PARAMETER_COUNT,
    LORA_RANK,
    MODEL_ID,
    MODEL_REVISION,
    PROJECTOR_PARAMETER_COUNT,
    TARGET_PROJECTION,
    TARGET_TOKEN_COUNT,
    TOTAL_TRAINABLE_PARAMETER_COUNT,
    NumericToolContextProjectorV2,
    tool_decoder_lora_settings_v2,
    validate_decoder_surface_v2,
)
from semantic_3d_chat.language.gemma4_tool_decoder_v2_checkpoint import (
    ARCHITECTURE,
    METADATA_FILENAME,
    TRAINING_STATUS,
    WEIGHTS_FILENAME,
    load_runtime_checkpoint_v2,
)
from semantic_3d_chat.language.lora import LoRAInstallation, install_lora_adapters
from semantic_3d_chat.robot.gemma4_tool_decoder_v2_backend import (
    ContinuousGemmaToolDecoderBackendV2,
)
from semantic_3d_chat.robot.llm_tool_policy import GeneratedToolProposal
from semantic_3d_chat.robot.navigation_policy import ACTION_NAMES
from semantic_3d_chat.robot.semantic_agent import ContinuousTextEncoder

_FILES: Final[frozenset[str]] = frozenset({WEIGHTS_FILENAME, METADATA_FILENAME})
_BLOCKED: Final[frozenset[str]] = frozenset({"oracle", "qa", "training", "scorer_only"})
_SHA256 = re.compile(r"[0-9a-f]{64}")
_PROVENANCE_FIELDS: Final[tuple[str, ...]] = (
    "base_checkpoint_sha256",
    "preregistration_sha256",
    "cpu_preflight_sha256",
    "training_authorization_sha256",
    "clearance_cache_sha256",
    "prefix_inventory_sha256",
)
_METADATA_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "architecture",
        "training_status",
        "status",
        "model_id",
        "model_revision",
        *_PROVENANCE_FIELDS,
        "target_module",
        "lora_rank",
        "lora_alpha",
        "lora_parameter_count",
        "numeric_projector_parameter_count",
        "total_trainable_parameter_count",
        "lora_state_sha256",
        "numeric_projector_state_sha256",
        "weights_sha256",
        "scene_prefix_tokens",
        "robot_tokens",
        "target_tokens",
        "clearance_tokens",
        "hidden_size",
        "max_new_tokens",
        "tool_vocabulary",
        "task_trained",
        "promotion_gates_passed",
        "saved_runtime_execution_gate_passed",
        "complete_scene_prefix_required",
        "question_independent_static_scene_prefix_required",
        "numeric_robot_tokens_required",
        "continuous_target_tokens_required",
        "numeric_clearance_tokens_required",
        "all_map_voxels_scored_for_target_grounding",
        "environmental_text_inputs",
        "oracle_inputs_at_runtime",
        "collision_interlock_required",
        "runtime_required_files",
    }
)


def _rooted(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    rooted = candidate if candidate.is_absolute() else PROJECT_ROOT / candidate
    return Path(os.path.abspath(rooted))


def _reject_runtime_path(path: Path) -> None:
    if _BLOCKED & {part.casefold() for part in path.parts}:
        raise ValueError("Gemma tool-decoder runtime paths cannot enter blocked data trees")
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current /= component
        if current.is_symlink():
            raise ValueError("Gemma tool-decoder runtime paths cannot contain symbolic links")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate Gemma tool-decoder metadata field: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"Nonfinite Gemma tool-decoder metadata constant: {value}")


def _base_runtime(runtime: Any) -> Any:
    refresher = getattr(runtime, "prefix_refresher", None)
    wrapped = getattr(refresher, "runtime", None)
    base = getattr(wrapped, "base", wrapped)
    if base is None:
        raise TypeError("Gemma tool-decoder integration requires a refreshing chat runtime")
    return base


def inspect_promoted_gemma_tool_decoder_v2(
    checkpoint: str | Path,
    *,
    expected_model_id: str = MODEL_ID,
    expected_model_revision: str = MODEL_REVISION,
    base_checkpoint: str | Path | None = None,
    audit: FileAccessAudit | None = None,
) -> dict[str, Any]:
    """Authenticate promotion metadata and file hashes without loading a model."""

    root = _rooted(checkpoint)
    _reject_runtime_path(root)
    if not root.is_dir() or {item.name for item in root.iterdir()} != set(_FILES):
        raise ValueError("Promoted Gemma tool decoder must contain exactly two files")
    entries = {item.name: item for item in root.iterdir()}
    if any(item.is_symlink() or not item.is_file() for item in entries.values()):
        raise ValueError("Gemma tool-decoder checkpoint entries must be regular files")
    metadata_path = entries[METADATA_FILENAME]
    weights_path = entries[WEIGHTS_FILENAME]
    if audit is not None:
        audit.record(metadata_path)
        audit.record(weights_path)
    try:
        metadata = json.loads(
            metadata_path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Gemma tool-decoder runtime metadata is invalid JSON") from error
    if not isinstance(metadata, dict):
        raise TypeError("Gemma tool-decoder runtime metadata must be a mapping")
    provenance = {field: metadata.get(field) for field in _PROVENANCE_FIELDS}
    if (
        set(metadata) != set(_METADATA_FIELDS)
        or metadata.get("schema_version") != 2
        or metadata.get("architecture") != ARCHITECTURE
        or metadata.get("training_status") != TRAINING_STATUS
        or metadata.get("status") != "promoted_runtime"
        or metadata.get("promotion_gates_passed") is not True
        or metadata.get("saved_runtime_execution_gate_passed") is not True
        or metadata.get("task_trained") is not True
        or metadata.get("model_id") != expected_model_id
        or metadata.get("model_revision") != expected_model_revision
        or metadata.get("target_module") != TARGET_PROJECTION
        or metadata.get("lora_rank") != LORA_RANK
        or metadata.get("lora_alpha") != LORA_ALPHA
        or metadata.get("lora_parameter_count") != LORA_PARAMETER_COUNT
        or metadata.get("numeric_projector_parameter_count")
        != PROJECTOR_PARAMETER_COUNT
        or metadata.get("total_trainable_parameter_count")
        != TOTAL_TRAINABLE_PARAMETER_COUNT
        or any(
            not isinstance(metadata.get(field), str)
            or _SHA256.fullmatch(metadata[field]) is None
            for field in (
                "lora_state_sha256",
                "numeric_projector_state_sha256",
                "weights_sha256",
            )
        )
        or metadata.get("scene_prefix_tokens") != 258
        or metadata.get("robot_tokens") != 4
        or metadata.get("target_tokens") != TARGET_TOKEN_COUNT
        or metadata.get("clearance_tokens") != CLEARANCE_TOKEN_COUNT
        or metadata.get("hidden_size") != HIDDEN_SIZE
        or metadata.get("max_new_tokens") != 24
        or metadata.get("tool_vocabulary") != list(ACTION_NAMES)
        or metadata.get("complete_scene_prefix_required") is not True
        or metadata.get("question_independent_static_scene_prefix_required") is not True
        or metadata.get("numeric_robot_tokens_required") is not True
        or metadata.get("continuous_target_tokens_required") is not True
        or metadata.get("numeric_clearance_tokens_required") is not True
        or metadata.get("all_map_voxels_scored_for_target_grounding") is not True
        or metadata.get("collision_interlock_required") is not True
        or metadata.get("environmental_text_inputs") != []
        or metadata.get("oracle_inputs_at_runtime") is not False
        or metadata.get("runtime_required_files")
        != [WEIGHTS_FILENAME, METADATA_FILENAME]
        or any(
            not isinstance(value, str) or _SHA256.fullmatch(value) is None
            for value in provenance.values()
        )
    ):
        raise ValueError("Gemma tool-decoder checkpoint is unpromoted or inference unsafe")
    observed_weights = _sha256(weights_path)
    if observed_weights != metadata.get("weights_sha256"):
        raise ValueError("Gemma tool-decoder weights hash changed")
    if base_checkpoint is not None:
        base_root = _rooted(base_checkpoint)
        _reject_runtime_path(base_root)
        base_adapter = base_root / "adapter.safetensors"
        if not base_adapter.is_file() or base_adapter.is_symlink():
            raise FileNotFoundError("Gemma tool decoder requires the exact V54 base adapter")
        if audit is not None:
            audit.record(base_adapter)
        if _sha256(base_adapter) != metadata.get("base_checkpoint_sha256"):
            raise ValueError("Selected static runtime is not the tool decoder's V54 base")
    return dict(metadata)


class ScopedGemmaToolDecoderBackendV2:
    """Install the tool LoRA only for one serialized action-generation call."""

    def __init__(
        self,
        backend: ContinuousGemmaToolDecoderBackendV2,
        installation: LoRAInstallation,
        model: nn.Module,
    ) -> None:
        if not isinstance(installation, LoRAInstallation):
            raise TypeError("Scoped Gemma tool decoder requires a LoRA installation")
        if installation.target_names != (TARGET_PROJECTION,) or len(installation.adapters) != 1:
            raise ValueError("Scoped Gemma tool decoder received the wrong LoRA surface")
        if not isinstance(model, nn.Module):
            raise TypeError("Scoped Gemma tool decoder requires the resident language model")
        installation.validate_state()
        self.backend = backend
        self.installation = installation
        self.model = model
        parent_path, _, self._attribute = TARGET_PROJECTION.rpartition(".")
        self._parent = model.get_submodule(parent_path)
        self._adapter = installation.adapters[0]
        self._original = self._adapter.base
        if getattr(self._parent, self._attribute, None) is not self._adapter:
            raise RuntimeError("Gemma tool adapter is not installed on its exact target")
        self._lock = threading.RLock()
        self._deactivate()

    def _deactivate(self) -> None:
        # Module removal, rather than scale=0, preserves the literal frozen V54
        # decoder function and avoids all tool-adapter arithmetic during QA.
        setattr(self._parent, self._attribute, self._original)

    @property
    def last_context(self) -> dict[str, Any] | None:
        return self.backend.last_context

    @property
    def last_grounding(self) -> dict[str, Any] | None:
        """Expose the backend's numeric grounding attestation to the CLI loop."""

        return self.backend.last_context

    @torch.inference_mode()
    def generate(
        self,
        instruction: str,
        *,
        correction_code: str | None,
    ) -> GeneratedToolProposal:
        with self._lock:
            if getattr(self._parent, self._attribute, None) is not self._original:
                raise RuntimeError("Gemma tool adapter was active outside action generation")
            try:
                setattr(self._parent, self._attribute, self._adapter)
                if getattr(self._parent, self._attribute, None) is not self._adapter:
                    raise RuntimeError("Gemma tool adapter activation did not take effect")
                return self.backend.generate(
                    instruction,
                    correction_code=correction_code,
                )
            finally:
                self._deactivate()


def load_promoted_gemma_tool_decoder_v2(
    runtime: Any,
    config: Mapping[str, Any],
    checkpoint: str | Path,
    *,
    audit: FileAccessAudit | None = None,
    text_encoder: ContinuousTextEncoder | None = None,
) -> tuple[ScopedGemmaToolDecoderBackendV2, dict[str, Any]]:
    """Install and strict-load V2 into the resident Gemma, rolling back on failure."""

    base = _base_runtime(runtime)
    language = getattr(base, "language", None)
    base_config = getattr(base, "config", None)
    base_checkpoint = getattr(base, "checkpoint_path", None)
    if language is None or not isinstance(base_config, Mapping) or base_checkpoint is None:
        raise TypeError("Gemma tool decoder requires a loaded static chat base")
    language_config = base_config.get("language")
    if not isinstance(language_config, Mapping):
        raise TypeError("Static chat base lacks a pinned language configuration")
    if (
        getattr(language, "backend_name", None) != "gemma4"
        or getattr(language, "prefix_backend", None) is None
        or getattr(language, "hidden_size", None) != HIDDEN_SIZE
        or language_config.get("model_id") != MODEL_ID
        or language_config.get("revision") != MODEL_REVISION
    ):
        raise ValueError("Gemma tool decoder requires the exact local Gemma-4 E2B runtime")
    metadata = inspect_promoted_gemma_tool_decoder_v2(
        checkpoint,
        expected_model_id=MODEL_ID,
        expected_model_revision=MODEL_REVISION,
        base_checkpoint=base_checkpoint,
        audit=audit,
    )
    provenance = {field: str(metadata[field]) for field in _PROVENANCE_FIELDS}
    model = language.model
    original = validate_decoder_surface_v2(model)
    parent_path, _, attribute = TARGET_PROJECTION.rpartition(".")
    parent = model.get_submodule(parent_path)
    if getattr(parent, attribute) is not original:
        raise RuntimeError("Gemma tool-decoder target changed during validation")
    installation: LoRAInstallation | None = None
    try:
        installation = install_lora_adapters(model, tool_decoder_lora_settings_v2())
        if installation is None:
            raise RuntimeError("Gemma tool-decoder LoRA installation was unexpectedly disabled")
        projector = NumericToolContextProjectorV2().to(language.device)
        loaded = load_runtime_checkpoint_v2(
            checkpoint,
            installation,
            projector,
            expected_provenance=provenance,
            require_promoted=True,
        )
        if loaded != metadata:
            raise RuntimeError("Gemma tool-decoder metadata changed during strict loading")
        for parameter in installation.parameters():
            parameter.requires_grad_(False)
        installation.eval()
        projector.requires_grad_(False).eval()
        model.requires_grad_(False)
        backend = ContinuousGemmaToolDecoderBackendV2(
            runtime,
            projector,
            loaded,
            config,
            text_encoder=text_encoder,
            max_new_tokens=24,
        )
        return ScopedGemmaToolDecoderBackendV2(backend, installation, model), dict(loaded)
    except BaseException:
        # Restore the exact pre-install module so a failed/unpromoted load cannot
        # alter ordinary V54 QA or poison a later valid startup attempt.
        setattr(parent, attribute, original)
        raise


__all__ = [
    "ScopedGemmaToolDecoderBackendV2",
    "inspect_promoted_gemma_tool_decoder_v2",
    "load_promoted_gemma_tool_decoder_v2",
]
