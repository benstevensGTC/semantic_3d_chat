"""Read-only authentication for a sealed V65 runtime checkpoint.

This module deliberately validates only the two-file inference artifact.  It
does not import the trainer, open QA/oracle data, or inspect any evaluation
references.  The terminal gate separately binds the checkpoint's saved-runtime
attestation to the create-once V65 training report.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from semantic_3d_chat.evaluation.predict_question_control import (
    _control_checkpoint_sha256,
)

V65_ARCHITECTURE: Final[str] = "magnitude_gated_teacher_basis_full_scene_control_v6"
V65_CHECKPOINT_SCHEMA_VERSION: Final[int] = 6

_SHA256 = re.compile(r"[0-9a-f]{64}")
_REQUIRED_METADATA_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "architecture",
        "hidden_size",
        "control_tokens",
        "expected_environment_latents",
        "moment_count",
        "interaction_dim",
        "trunk_dim",
        "output_basis_rank",
        "maximum_control_rms",
        "initial_control_rms",
        "activation_rms_threshold",
        "activation_rms_aggregation",
        "weights_sha256",
        "base_checkpoint_sha256",
        "base_runtime_config_sha256",
        "source_v65_training_fit_state_sha256",
        "source_v65_value_state_sha256",
        "magnitude_gated_continuous_control",
        "exact_no_control_below_threshold",
        "question_dependent_scene_retrieval",
        "complete_scene_prefix_required",
        "environmental_text_inputs",
        "fixed_global_scene_moments",
        "boundary_tokens_excluded_from_scene_signature",
        "softmax_scene_attention_used",
        "control_values_scene_question_bilinear",
        "gate_scene_question_conditioned",
        "route_is_environmental_retrieval",
        "saved_runtime_training_gate_required",
        "saved_runtime_training_gate_passed",
        "saved_runtime_training_gate_attestation_sha256",
    }
)


@dataclass(frozen=True)
class SealedV65Checkpoint:
    """Authenticated identities for a public, training-gated V65 checkpoint."""

    root: Path
    fingerprint_sha256: str
    files: dict[str, dict[str, Any]]
    metadata: dict[str, Any]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _hash(value: object) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _positive_integer(value: object) -> bool:
    return type(value) is int and value > 0


def _finite_positive(value: object) -> bool:
    return type(value) in {int, float} and math.isfinite(float(value)) and float(value) > 0.0


def _load_metadata(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("V65 runtime metadata is invalid JSON") from exc
    if not isinstance(value, dict):
        raise TypeError("V65 runtime metadata must be a JSON object")
    return value


def validate_sealed_v65_checkpoint(
    checkpoint: str | Path,
    *,
    base_checkpoint_sha256: str | None = None,
    runtime_config_sha256: str | None = None,
) -> SealedV65Checkpoint:
    """Authenticate a strict, public schema-6 V65 runtime artifact.

    An ungated staging checkpoint is always rejected.  Optional base/runtime
    digests bind the artifact to the exact inference stack before an output
    journal can be created.
    """

    root = Path(os.path.abspath(Path(checkpoint).expanduser()))
    fingerprint = _control_checkpoint_sha256(root)
    metadata_path = root / "runtime_metadata.json"
    files = {
        name: {
            "sha256": _sha256_file(root / name),
            "size_bytes": (root / name).stat().st_size,
        }
        for name in ("control.safetensors", "runtime_metadata.json")
    }
    metadata = _load_metadata(metadata_path)
    if set(metadata) != _REQUIRED_METADATA_FIELDS:
        raise ValueError("V65 runtime metadata fields differ from sealed schema 6")

    digest_fields = (
        "weights_sha256",
        "base_checkpoint_sha256",
        "base_runtime_config_sha256",
        "source_v65_training_fit_state_sha256",
        "source_v65_value_state_sha256",
        "saved_runtime_training_gate_attestation_sha256",
    )
    if any(not _hash(metadata.get(field)) for field in digest_fields):
        raise ValueError("V65 runtime metadata contains an invalid digest")
    integer_fields = (
        "hidden_size",
        "control_tokens",
        "expected_environment_latents",
        "moment_count",
        "interaction_dim",
        "trunk_dim",
        "output_basis_rank",
    )
    if any(not _positive_integer(metadata.get(field)) for field in integer_fields):
        raise ValueError("V65 runtime metadata contains an invalid dimension")
    numeric_fields = (
        "maximum_control_rms",
        "initial_control_rms",
        "activation_rms_threshold",
    )
    if any(not _finite_positive(metadata.get(field)) for field in numeric_fields):
        raise ValueError("V65 runtime metadata contains an invalid RMS value")

    if (
        metadata["schema_version"] != V65_CHECKPOINT_SCHEMA_VERSION
        or metadata["architecture"] != V65_ARCHITECTURE
        or metadata["hidden_size"] != 1536
        or metadata["control_tokens"] != 4
        or metadata["expected_environment_latents"] != 256
        or float(metadata["activation_rms_threshold"]) >= float(metadata["maximum_control_rms"])
        or metadata["activation_rms_aggregation"] != "maximum_over_control_tokens"
        or metadata["weights_sha256"] != files["control.safetensors"]["sha256"]
        or metadata["source_v65_training_fit_state_sha256"]
        != metadata["source_v65_value_state_sha256"]
        or metadata["magnitude_gated_continuous_control"] is not True
        or metadata["exact_no_control_below_threshold"] is not True
        or metadata["question_dependent_scene_retrieval"] is not False
        or metadata["complete_scene_prefix_required"] is not True
        or metadata["environmental_text_inputs"] != []
        or metadata["fixed_global_scene_moments"] is not True
        or metadata["boundary_tokens_excluded_from_scene_signature"] is not True
        or metadata["softmax_scene_attention_used"] is not False
        or metadata["control_values_scene_question_bilinear"] is not True
        or metadata["gate_scene_question_conditioned"] is not True
        or metadata["route_is_environmental_retrieval"] is not False
        or metadata["saved_runtime_training_gate_required"] is not True
        or metadata["saved_runtime_training_gate_passed"] is not True
    ):
        raise ValueError("V65 checkpoint is not a sealed magnitude-gated runtime")
    if (
        base_checkpoint_sha256 is not None
        and metadata["base_checkpoint_sha256"] != base_checkpoint_sha256
    ):
        raise ValueError("V65 checkpoint is bound to a different base checkpoint")
    if (
        runtime_config_sha256 is not None
        and metadata["base_runtime_config_sha256"] != runtime_config_sha256
    ):
        raise ValueError("V65 checkpoint is bound to a different runtime configuration")
    return SealedV65Checkpoint(
        root=root,
        fingerprint_sha256=fingerprint,
        files=files,
        metadata=metadata,
    )


__all__ = [
    "V65_ARCHITECTURE",
    "V65_CHECKPOINT_SCHEMA_VERSION",
    "SealedV65Checkpoint",
    "validate_sealed_v65_checkpoint",
]
