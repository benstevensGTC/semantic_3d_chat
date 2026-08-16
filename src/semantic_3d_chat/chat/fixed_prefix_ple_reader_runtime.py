"""Strict Gemma-4 reader over the immutable V54 continuous scene prefix.

The V54 runtime constructs its complete 258-token environment prefix before
this module installs the small per-layer-projection (PLE) reader adapter.  The
adapter changes only how the frozen local Gemma model reads that prefix; it
does not compile, retrieve, caption, label, or otherwise alter the scene.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import torch
from safetensors.torch import load_file
from torch import nn

from semantic_3d_chat.chat.file_audit import FileAccessAudit
from semantic_3d_chat.chat.runtime import ChatAnswer, StaticChatRuntime
from semantic_3d_chat.chat.runtime_config import effective_runtime_config_sha256
from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.language.lora import (
    LoRAInstallation,
    LoRASettings,
    install_lora_adapters,
)

PLE_READER_ARTIFACT: Final[str] = "gemma4_v54_fixed_prefix_ple_reader_v4"
PLE_READER_MODEL_ID: Final[str] = "google/gemma-4-E2B-it"
PLE_READER_MODEL_REVISION: Final[str] = (
    "3e22461f65e89153144f8adb70e3b8c2cc9845a7"
)
PLE_READER_TARGET_MODULE: Final[str] = (
    "model.language_model.per_layer_model_projection"
)
PLE_READER_RANK: Final[int] = 4
PLE_READER_ALPHA: Final[float] = 8.0
PLE_READER_DROPOUT: Final[float] = 0.0
PLE_READER_PROJECTION_IN_FEATURES: Final[int] = 1536
PLE_READER_PROJECTION_OUT_FEATURES: Final[int] = 35 * 256
PLE_READER_PARAMETER_COUNT: Final[int] = 41_984
PLE_READER_PREFIX_TOKENS: Final[int] = 258
PLE_READER_SCENE_LATENTS: Final[int] = 256
PLE_READER_HIDDEN_DIMENSION: Final[int] = 1536

_V54_BASE_CHECKPOINT_SHA256: Final[str] = (
    "3e128b40c1b73bb32750285679cda6b1bea364e67465e986a94a81dfc95e81e8"
)
_V54_RUNTIME_CONFIG_EFFECTIVE_SHA256: Final[str] = (
    "714c60ce9ccb1dff69c72f6618f8afb6f31bc60a830b5ee0fb794fedaa8a321e"
)
_V54_ADAPTER_SHA256: Final[str] = (
    "6c627f0a0d9efb7100489c24cbd4acafe10456b79280a64a85399a21cb541daf"
)
_V54_RUNTIME_METADATA_SHA256: Final[str] = (
    "807515461c71b08c08dfbd08a184a653e791413748530fa69402512eca6f6fdd"
)
_V4_PREREGISTRATION_SHA256: Final[str] = (
    "34b4576a6ced7003c916c5dc3deabecf8e6e70a0e39bcc8329d039fd00ef3d59"
)
_SHA256 = re.compile(r"[0-9a-f]{64}")
_FORBIDDEN_COMPONENTS = frozenset(
    {
        "oracle",
        "qa",
        "rendered",
        "features",
        "training",
        "scorer_only",
        "scorer-only",
    }
)
_READER_CHECKPOINT_FILES = frozenset(
    {"adapter.safetensors", "runtime_metadata.json"}
)
_V54_RUNTIME_FILES = frozenset({"adapter.safetensors", "runtime_metadata.json"})
_V54_CHECKPOINT_FILES = _V54_RUNTIME_FILES | {"metadata.json"}
_READER_METADATA_FIELDS = frozenset(
    {
        "schema_version",
        "artifact",
        "base_checkpoint_sha256",
        "base_runtime_config_effective_sha256",
        "model_id",
        "model_revision",
        "fixed_prefix_tokens",
        "scene_latents",
        "scene_hidden_dimension",
        "prefix_computed_before_question",
        "question_dependent_scene_retrieval",
        "environmental_text_inputs",
        "oracle_runtime_access",
        "target_module",
        "rank",
        "alpha",
        "dropout",
        "trainable_parameter_count",
        "adapter_state_sha256",
        "adapter_file_sha256",
        "selection_summary_sha256",
        "preregistration_sha256",
    }
)


@dataclass(frozen=True)
class ValidatedPLEReaderCheckpoint:
    root: Path
    weights_path: Path
    metadata_path: Path
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ValidatedV54Checkpoint:
    root: Path
    adapter_sha256: str
    runtime_metadata_sha256: str


def _rooted(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    rooted = candidate if candidate.is_absolute() else PROJECT_ROOT / candidate
    unresolved = Path(os.path.abspath(rooted))
    current = Path(unresolved.anchor)
    for component in unresolved.parts[1:]:
        current /= component
        if current.is_symlink():
            raise ValueError(f"PLE reader runtime paths must not contain symlinks: {current}")
    return unresolved.resolve()


def _reject_forbidden_location(path: Path, purpose: str) -> None:
    components = {part.casefold() for part in path.parts}
    if components & _FORBIDDEN_COMPONENTS:
        raise ValueError(
            f"{purpose} must be physically separate from environmental supervision"
        )


def _audited_sha256(path: Path, audit: FileAccessAudit | None) -> str:
    if audit is not None:
        audit.record(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_json_object(path: Path, audit: FileAccessAudit | None) -> dict[str, Any]:
    if audit is not None:
        audit.record(path)

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"PLE reader metadata has a duplicate field: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicates,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("PLE reader runtime metadata is invalid JSON") from error
    if not isinstance(value, dict):
        raise TypeError("PLE reader runtime metadata must be a JSON object")
    return value


def _require_digest(metadata: Mapping[str, Any], field: str) -> str:
    value = metadata.get(field)
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"PLE reader {field} must be a lowercase SHA-256 digest")
    return value


def _validate_reader_metadata(metadata: dict[str, Any]) -> None:
    if set(metadata) != _READER_METADATA_FIELDS:
        raise ValueError(
            "PLE reader runtime metadata fields changed: "
            f"expected={sorted(_READER_METADATA_FIELDS)} observed={sorted(metadata)}"
        )
    exact = {
        "schema_version": 1,
        "artifact": PLE_READER_ARTIFACT,
        "base_checkpoint_sha256": _V54_BASE_CHECKPOINT_SHA256,
        "base_runtime_config_effective_sha256": _V54_RUNTIME_CONFIG_EFFECTIVE_SHA256,
        "model_id": PLE_READER_MODEL_ID,
        "model_revision": PLE_READER_MODEL_REVISION,
        "fixed_prefix_tokens": PLE_READER_PREFIX_TOKENS,
        "scene_latents": PLE_READER_SCENE_LATENTS,
        "scene_hidden_dimension": PLE_READER_HIDDEN_DIMENSION,
        "prefix_computed_before_question": True,
        "question_dependent_scene_retrieval": False,
        "environmental_text_inputs": [],
        "oracle_runtime_access": False,
        "target_module": PLE_READER_TARGET_MODULE,
        "rank": PLE_READER_RANK,
        "alpha": PLE_READER_ALPHA,
        "dropout": PLE_READER_DROPOUT,
        "trainable_parameter_count": PLE_READER_PARAMETER_COUNT,
        "preregistration_sha256": _V4_PREREGISTRATION_SHA256,
    }
    for field, expected in exact.items():
        observed = metadata.get(field)
        if isinstance(expected, bool):
            matches = observed is expected
        elif isinstance(expected, int):
            matches = type(observed) is int and observed == expected
        elif isinstance(expected, float):
            matches = (
                type(observed) in {int, float}
                and not isinstance(observed, bool)
                and math.isclose(float(observed), expected, rel_tol=0.0, abs_tol=0.0)
            )
        else:
            matches = observed == expected
        if not matches:
            raise ValueError(
                f"PLE reader runtime contract mismatch for {field}: "
                f"{observed!r} != {expected!r}"
            )
    for field in (
        "adapter_state_sha256",
        "adapter_file_sha256",
        "selection_summary_sha256",
        "preregistration_sha256",
        "base_checkpoint_sha256",
        "base_runtime_config_effective_sha256",
    ):
        _require_digest(metadata, field)


def validate_ple_reader_checkpoint(
    checkpoint: str | Path,
    *,
    audit: FileAccessAudit | None = None,
) -> ValidatedPLEReaderCheckpoint:
    """Authenticate the two-file, numeric-only V4 reader checkpoint."""

    root = _rooted(checkpoint)
    _reject_forbidden_location(root, "PLE reader checkpoint")
    if not root.is_dir() or root.is_symlink():
        raise FileNotFoundError(f"PLE reader checkpoint is unavailable or unsafe: {root}")
    inventory = {item.name for item in root.iterdir()}
    if inventory != _READER_CHECKPOINT_FILES:
        raise ValueError(
            "PLE reader checkpoint must contain only sanitized runtime files: "
            f"expected={sorted(_READER_CHECKPOINT_FILES)} observed={sorted(inventory)}"
        )
    weights_path = root / "adapter.safetensors"
    metadata_path = root / "runtime_metadata.json"
    for path in (weights_path, metadata_path):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"PLE reader checkpoint member is unsafe: {path}")
    metadata = _strict_json_object(metadata_path, audit)
    _validate_reader_metadata(metadata)
    observed_weights_sha256 = _audited_sha256(weights_path, audit)
    if observed_weights_sha256 != metadata["adapter_file_sha256"]:
        raise ValueError("PLE reader adapter file digest changed")
    return ValidatedPLEReaderCheckpoint(
        root=root,
        weights_path=weights_path,
        metadata_path=metadata_path,
        metadata=metadata,
    )


def validate_v54_checkpoint(
    checkpoint: str | Path,
    *,
    audit: FileAccessAudit | None = None,
) -> ValidatedV54Checkpoint:
    """Bind the exact V54 runtime files without opening training metadata."""

    root = _rooted(checkpoint)
    _reject_forbidden_location(root, "V54 base checkpoint")
    if not root.is_dir() or root.is_symlink():
        raise FileNotFoundError(f"V54 base checkpoint is unavailable or unsafe: {root}")
    inventory = {item.name for item in root.iterdir()}
    if inventory not in {_V54_RUNTIME_FILES, _V54_CHECKPOINT_FILES}:
        raise ValueError(f"V54 base checkpoint inventory changed: {sorted(inventory)}")
    for name in _V54_RUNTIME_FILES:
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"V54 base checkpoint member is unsafe: {path}")
    # metadata.json is offline training/evaluation provenance.  Deliberately
    # authenticate only the two sanctioned inference inputs and deny any later
    # attempt to open the adjacent training metadata while the audit is active.
    training_metadata = (root / "metadata.json").resolve()
    if (
        audit is not None
        and training_metadata.exists()
        and training_metadata not in audit.forbidden_roots
    ):
        audit.forbidden_roots.append(training_metadata)
    adapter_sha256 = _audited_sha256(root / "adapter.safetensors", audit)
    runtime_metadata_sha256 = _audited_sha256(
        root / "runtime_metadata.json", audit
    )
    if adapter_sha256 != _V54_ADAPTER_SHA256:
        raise ValueError("V54 base adapter digest changed")
    if runtime_metadata_sha256 != _V54_RUNTIME_METADATA_SHA256:
        raise ValueError("V54 sanitized runtime metadata digest changed")
    return ValidatedV54Checkpoint(root, adapter_sha256, runtime_metadata_sha256)


def validate_v54_runtime_config(config: Mapping[str, Any]) -> str:
    """Require the exact standalone, supervision-free V54 runtime config."""

    if config.get("_runtime_safe_config") is not True:
        raise ValueError("PLE reader requires a standalone validated runtime config")
    digest = effective_runtime_config_sha256(config)
    if digest != _V54_RUNTIME_CONFIG_EFFECTIVE_SHA256:
        raise ValueError("PLE reader requires the exact V54 effective runtime config")
    language = config.get("language")
    scene_encoder = config.get("scene_encoder")
    if (
        not isinstance(language, Mapping)
        or language.get("backend") != "gemma4"
        or language.get("model_id") != PLE_READER_MODEL_ID
        or language.get("revision") != PLE_READER_MODEL_REVISION
        or not isinstance(scene_encoder, Mapping)
        or scene_encoder.get("global_latents") != PLE_READER_SCENE_LATENTS
    ):
        raise ValueError("PLE reader V54 model or full-scene prefix contract changed")
    return digest


def _reader_lora_settings() -> LoRASettings:
    return LoRASettings(
        enabled=True,
        rank=PLE_READER_RANK,
        alpha=PLE_READER_ALPHA,
        dropout=PLE_READER_DROPOUT,
        target_modules=(PLE_READER_TARGET_MODULE,),
    )


def _validate_projection_surface(model: nn.Module) -> nn.Linear:
    try:
        projection = model.get_submodule(PLE_READER_TARGET_MODULE)
    except AttributeError as error:
        raise ValueError(
            f"Loaded Gemma model lacks the PLE reader target: {PLE_READER_TARGET_MODULE}"
        ) from error
    if not isinstance(projection, nn.Linear):
        raise TypeError("Gemma PLE reader target must be an unwrapped torch.nn.Linear")
    observed = (
        projection.in_features,
        projection.out_features,
        projection.bias is None,
    )
    expected = (
        PLE_READER_PROJECTION_IN_FEATURES,
        PLE_READER_PROJECTION_OUT_FEATURES,
        True,
    )
    if observed != expected:
        raise ValueError(f"Gemma PLE reader projection changed: {observed} != {expected}")
    return projection


def _install_validated_reader(
    model: nn.Module,
    checkpoint: ValidatedPLEReaderCheckpoint,
) -> LoRAInstallation:
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("PLE reader requires a completely frozen V54 language model")
    _validate_projection_surface(model)
    installation = install_lora_adapters(model, _reader_lora_settings())
    if installation is None:
        raise RuntimeError("PLE reader LoRA installation unexpectedly returned no adapter")
    expected_state = installation.state_module.state_dict()
    state = load_file(str(checkpoint.weights_path), device="cpu")
    if set(state) != set(expected_state):
        raise ValueError(
            "PLE reader adapter tensor names changed: "
            f"expected={sorted(expected_state)} observed={sorted(state)}"
        )
    for name, expected in expected_state.items():
        observed = state[name]
        if (
            observed.dtype != torch.float32
            or observed.shape != expected.shape
            or not torch.isfinite(observed).all()
        ):
            raise ValueError(f"PLE reader adapter tensor contract changed: {name}")
    installation.state_module.load_state_dict(state, strict=True)
    installation.validate_state()
    if installation.parameter_count != PLE_READER_PARAMETER_COUNT:
        raise ValueError("PLE reader installed parameter count changed")
    observed_state_sha256 = installation.state_sha256()
    if observed_state_sha256 != checkpoint.metadata["adapter_state_sha256"]:
        raise ValueError("PLE reader adapter state digest changed")
    installation.eval()
    model.requires_grad_(False)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("PLE reader inference model retained trainable parameters")
    return installation


def load_ple_reader_adapter(
    model: nn.Module,
    checkpoint: str | Path,
    *,
    audit: FileAccessAudit | None = None,
) -> tuple[LoRAInstallation, dict[str, Any], Path]:
    """Validate and install the frozen V4 reader onto an already loaded V54 model."""

    validated = validate_ple_reader_checkpoint(checkpoint, audit=audit)
    installation = _install_validated_reader(model, validated)
    return installation, dict(validated.metadata), validated.root


class FixedPrefixPLEReaderChatRuntime:
    """Local chat using one byte-invariant V54 prefix and one frozen PLE reader."""

    def __init__(
        self,
        base: StaticChatRuntime,
        *,
        reader_installation: LoRAInstallation,
        reader_metadata: Mapping[str, Any],
        reader_checkpoint_path: Path,
    ) -> None:
        if base.questions_answered != 0:
            raise RuntimeError("PLE reader must be installed before the first user question")
        base.assert_prefix_unchanged()
        if tuple(base.scene_prefix.shape) != (
            1,
            PLE_READER_PREFIX_TOKENS,
            PLE_READER_HIDDEN_DIMENSION,
        ):
            raise ValueError("V54 full-scene prefix shape changed")
        if reader_installation.state_sha256() != reader_metadata.get(
            "adapter_state_sha256"
        ):
            raise ValueError("Installed PLE reader state does not match runtime metadata")
        if reader_installation.training:
            raise ValueError("PLE reader adapter must be in evaluation mode")
        self.base = base
        self.config = base.config
        self.scene_id = base.scene_id
        self.scene_prefix = base.scene_prefix
        self.scene_prefix_hash = base.scene_prefix_hash
        self.reader_installation = reader_installation
        self.reader_metadata = dict(reader_metadata)
        self.reader_checkpoint_path = reader_checkpoint_path.resolve()
        self._startup_prefix_hash = base.current_prefix_hash()
        if self._startup_prefix_hash != self.scene_prefix_hash:
            raise RuntimeError("V54 scene prefix changed while installing the PLE reader")

    @classmethod
    def load(
        cls,
        config: dict[str, Any],
        scene_id: str,
        *,
        base_checkpoint: str | Path,
        reader_checkpoint: str | Path,
        audit: FileAccessAudit | None = None,
        local_files_only: bool = True,
    ) -> FixedPrefixPLEReaderChatRuntime:
        runtime_config_sha256 = validate_v54_runtime_config(config)
        reader = validate_ple_reader_checkpoint(reader_checkpoint, audit=audit)
        base_contract = validate_v54_checkpoint(base_checkpoint, audit=audit)
        if (
            reader.metadata["base_checkpoint_sha256"] != _V54_BASE_CHECKPOINT_SHA256
            or reader.metadata["base_runtime_config_effective_sha256"]
            != runtime_config_sha256
        ):
            raise ValueError("PLE reader is not bound to the selected V54 base runtime")
        base = StaticChatRuntime.load(
            config,
            scene_id,
            checkpoint=base_contract.root,
            audit=audit,
            local_files_only=local_files_only,
        )
        if base.checkpoint_path.resolve() != base_contract.root:
            raise RuntimeError("Loaded V54 runtime checkpoint differs from validated input")
        base.assert_prefix_unchanged()
        # Model construction can take minutes on this Mac. Re-authenticate the
        # tiny reader payload after that interval so an external replacement
        # cannot win the startup validation/load gap.
        reader_after_model_load = validate_ple_reader_checkpoint(
            reader.root, audit=audit
        )
        if reader_after_model_load.metadata != reader.metadata:
            raise RuntimeError("PLE reader metadata changed while V54 was loading")
        installation = _install_validated_reader(
            base.language.model, reader_after_model_load
        )
        return cls(
            base,
            reader_installation=installation,
            reader_metadata=reader_after_model_load.metadata,
            reader_checkpoint_path=reader_after_model_load.root,
        )

    @property
    def questions_answered(self) -> int:
        return self.base.questions_answered

    def current_prefix_hash(self) -> str:
        return self.base.current_prefix_hash()

    def assert_prefix_unchanged(self) -> None:
        self.base.assert_prefix_unchanged()
        current = self.current_prefix_hash()
        if current != self._startup_prefix_hash:
            raise RuntimeError(
                "PLE reader environment prefix changed after startup: "
                f"{self._startup_prefix_hash} != {current}"
            )

    def startup_summary(self) -> dict[str, Any]:
        self.assert_prefix_unchanged()
        base = self.base.startup_summary()
        return {
            **base,
            "phase": "fixed_prefix_ple_reader_ready",
            "prefix_hash": self.scene_prefix_hash,
            "prefix_shape": list(self.scene_prefix.shape),
            "scene_prefix_computed_before_question": self.questions_answered == 0,
            "scene_prefix_computed_before_reader_installation": True,
            "reader_adapter_loaded_before_first_question": self.questions_answered == 0,
            "strict_fixed_environment_embedding_input": True,
            "environment_conditioned_input_sha256": self.scene_prefix_hash,
            "question_conditioned_scene_readout_tokens": False,
            "question_dependent_scene_retrieval": False,
            "environmental_text_inputs": [],
            "oracle_runtime_access": False,
            "reader_checkpoint": str(self.reader_checkpoint_path),
            "reader_artifact": self.reader_metadata["artifact"],
            "reader_target_module": self.reader_metadata["target_module"],
            "reader_rank": self.reader_metadata["rank"],
            "reader_parameter_count": self.reader_installation.parameter_count,
            "reader_adapter_state_sha256": self.reader_installation.state_sha256(),
            "reader_checkpoint_files": sorted(_READER_CHECKPOINT_FILES),
            "base_training_metadata_opened": False,
        }

    def answer(self, question: str) -> ChatAnswer:
        # The environment hash is checked before the base runtime tokenizes the
        # question; no scene method or question-dependent selector is invoked.
        self.assert_prefix_unchanged()
        result = self.base.answer(question)
        self.assert_prefix_unchanged()
        if result.prefix_hash != self.scene_prefix_hash:
            raise RuntimeError("PLE reader answer used a different scene prefix")
        return result


__all__ = [
    "PLE_READER_ARTIFACT",
    "PLE_READER_PARAMETER_COUNT",
    "PLE_READER_TARGET_MODULE",
    "FixedPrefixPLEReaderChatRuntime",
    "ValidatedPLEReaderCheckpoint",
    "ValidatedV54Checkpoint",
    "load_ple_reader_adapter",
    "validate_ple_reader_checkpoint",
    "validate_v54_checkpoint",
    "validate_v54_runtime_config",
]
