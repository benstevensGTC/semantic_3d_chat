"""Pure-data contract for the V75 official validation boundary.

This module deliberately imports no language-model, training-data, or scorer
code.  Prediction and scoring use the same content identities without sharing
answer-bearing objects or opening each other's private inputs.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal

from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.evaluation.question_manifest import (
    QuestionManifest,
    load_question_manifest,
)

ARTIFACT: Final[str] = "v75_official_validation_v1"
EXPECTED_SCENE_IDS: Final[tuple[str, ...]] = tuple(
    f"scene_{index:06d}" for index in range(57, 63)
)
EXPECTED_QUESTION_COUNT: Final[int] = 216
EXPECTED_QUESTIONS_PER_SCENE: Final[int] = 36
EXPECTED_QUESTION_MANIFEST_SHA256: Final[str] = (
    "74fcfb181bbf809dd6dc3b07800de728558298149e9d76325870c6b4d665b0a2"
)
EXPECTED_QUESTIONS_SHA256: Final[str] = (
    "e468d851e46ad606c9599ac1a8016ed10fa974f9985dfc3add6250f3403f8b25"
)
EXPECTED_REFERENCE_SHA256: Final[str] = (
    "30ed9006ed442198b3e2444e0c3cdda73cb77c01e7285f31000709b94bb8acad"
)
EXPECTED_SOURCE_V75_CANDIDATE_SHA256: Final[str] = (
    "d01275538489b3493a8e1ff080109d1db46832be6ca2a26f6d89d161c597188a"
)
EXPECTED_BASE_CHECKPOINT_SHA256: Final[str] = (
    "7c3e679702ccd204fa4d7ae4077b065f3d7a7fe36df7dbc45492d67566e97f59"
)
EXPECTED_RUNTIME_CONFIG_SHA256: Final[str] = (
    "714c60ce9ccb1dff69c72f6618f8afb6f31bc60a830b5ee0fb794fedaa8a321e"
)
V75_RUNTIME_ARCHITECTURE: Final[str] = "dense_full_scene_continuous_control_v75"

DEFAULT_RUNTIME_CONFIG: Final[Path] = Path(
    "configs/runtime/gemma4_v56_question_control.yaml"
)
DEFAULT_BASE_CHECKPOINT: Final[Path] = Path(
    "data_gemma4/runtime/checkpoints/gemma4_v54_release_v1"
)
DEFAULT_CONTROL_CHECKPOINT: Final[Path] = Path(
    "data_gemma4/runtime/checkpoints/gemma4_v75_nll_control_release_v1"
)
DEFAULT_QUESTIONS_MANIFEST: Final[Path] = Path(
    "reports/gemma4/questions/v56_fresh_development_validation.json"
)
DEFAULT_REFERENCES: Final[Path] = Path("data_diverse52/qa/validation.jsonl")
DEFAULT_PREDICTIONS: Final[Path] = Path(
    "reports/gemma4/predictions/v75_official_validation.jsonl"
)
DEFAULT_SCORE: Final[Path] = Path(
    "reports/gemma4/metrics/v75_official_validation_score.json"
)

_SHA256 = re.compile(r"[0-9a-f]{64}")
_PATH_TOKEN = re.compile(r"[a-z0-9]+")
_FORBIDDEN_PREDICTION_PATH_TOKENS: Final[frozenset[str]] = frozenset(
    {"oracle", "qa", "answer", "answers", "reference", "references", "scorer"}
)
_CONTROL_FILES: Final[tuple[str, str]] = (
    "control.safetensors",
    "runtime_metadata.json",
)
_V75_METADATA_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "architecture",
        "hidden_size",
        "control_tokens",
        "environment_latents",
        "query_count",
        "model_dimension",
        "coefficient_decoder_hidden_dimension",
        "output_basis_rank",
        "uniform_floor_mass",
        "maximum_control_rms",
        "weights_sha256",
        "base_checkpoint_sha256",
        "base_runtime_config_sha256",
        "source_v75_candidate_sha256",
        "source_v75_training_fit_state_sha256",
        "saved_runtime_training_gate_required",
        "saved_runtime_training_gate_passed",
        "saved_runtime_training_gate_attestation_sha256",
        "complete_scene_prefix_required",
        "full_unchanged_prefix_retained_separately",
        "prequestion_scene_key_value_cache",
        "all_environment_latents_attended",
        "positive_attention_floor",
        "bilinear_question_scene_value_interaction",
        "bias_free_nonlinear_coefficient_decoder",
        "zero_preserving_coefficient_activation",
        "question_only_output_path_exists",
        "question_dependent_scene_retrieval",
        "latent_selection_or_top_k_used",
        "environmental_text_inputs",
        "training_answers_runtime_loaded",
        "answer_text_runtime_loaded",
        "answer_class_codebook_runtime_loaded",
        "teacher_cache_runtime_loaded",
        "oracle_runtime_loaded",
        "question_or_answer_text_serialized",
    }
)
_V75_AUDIT_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "architecture",
        "bias_free_nonlinear_coefficient_decoder",
        "scene_token_count",
        "environment_latent_count",
        "control_token_count",
        "coefficient_decoder_hidden_dimension",
        "every_scene_token_influenced_output",
        "minimum_attention_weight",
        "positive_attention_floor",
        "softmax_scene_attention_used",
        "bilinear_question_scene_value_interaction",
        "question_only_output_path_exists",
        "question_dependent_scene_retrieval",
        "latent_selection_or_top_k_used",
        "immutable_full_prefix_retained_separately",
        "prequestion_scene_key_value_cache",
        "zero_scene_produces_exact_zero_controls",
        "control_used",
        "maximum_control_rms",
        "saved_runtime_training_gate_required",
        "training_answers_runtime_loaded",
        "answer_text_runtime_loaded",
        "answer_class_codebook_runtime_loaded",
        "zero_preserving_coefficient_activation",
    }
)


@dataclass(frozen=True)
class V75ControlIdentity:
    path: Path
    sha256: str
    weights_sha256: str
    runtime_metadata_sha256: str
    metadata: dict[str, Any]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_path(path: str | Path) -> Path:
    value = Path(path).expanduser()
    rooted = value if value.is_absolute() else PROJECT_ROOT / value
    return Path(os.path.abspath(rooted))


def reject_symlink_components(path: str | Path, purpose: str) -> Path:
    source = resolve_path(path)
    current = Path(source.anchor)
    for component in source.parts[1:]:
        current /= component
        if current.is_symlink():
            raise ValueError(f"V75 {purpose} path contains a symbolic link: {current}")
    return source


def _path_tokens(path: Path) -> set[str]:
    return {
        token
        for component in path.parts
        for token in _PATH_TOKEN.findall(component.casefold())
    }


def safe_prediction_input(
    path: str | Path,
    purpose: str,
    *,
    kind: Literal["file", "directory"],
) -> Path:
    """Resolve an inference input and reject answer/oracle path vocabulary."""

    source = reject_symlink_components(path, purpose)
    forbidden = sorted(_path_tokens(source) & _FORBIDDEN_PREDICTION_PATH_TOKENS)
    if forbidden:
        raise ValueError(
            f"V75 prediction refuses {purpose} path tokens: {forbidden}"
        )
    valid = source.is_file() if kind == "file" else source.is_dir()
    if not valid:
        raise FileNotFoundError(f"V75 {purpose} is unavailable: {source}")
    return source


def safe_prediction_output(path: str | Path) -> Path:
    destination = reject_symlink_components(path, "prediction output")
    forbidden = sorted(
        _path_tokens(destination) & _FORBIDDEN_PREDICTION_PATH_TOKENS
    )
    if forbidden:
        raise ValueError(
            f"V75 prediction refuses output path tokens: {forbidden}"
        )
    if destination.exists() and not destination.is_file():
        raise ValueError(f"V75 prediction output is not a regular file: {destination}")
    return destination


def validate_official_question_manifest(path: str | Path) -> QuestionManifest:
    source = safe_prediction_input(path, "questions manifest", kind="file")
    manifest = load_question_manifest(source)
    scene_counts = Counter(record.scene_id for record in manifest.questions)
    if (
        manifest.manifest_sha256 != EXPECTED_QUESTION_MANIFEST_SHA256
        or manifest.questions_sha256 != EXPECTED_QUESTIONS_SHA256
        or manifest.source_qa_sha256 != EXPECTED_REFERENCE_SHA256
        or manifest.question_count != EXPECTED_QUESTION_COUNT
        or manifest.scene_count != len(EXPECTED_SCENE_IDS)
        or scene_counts
        != Counter(
            {
                scene_id: EXPECTED_QUESTIONS_PER_SCENE
                for scene_id in EXPECTED_SCENE_IDS
            }
        )
    ):
        raise ValueError("V75 official questions differ from the frozen sanitized manifest")
    return manifest


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"V75 {field} must be a mapping")
    return value


def validate_v75_runtime_metadata(
    value: object,
    *,
    observed_weights_sha256: str,
) -> dict[str, Any]:
    metadata = dict(_mapping(value, "runtime metadata"))
    if set(metadata) != _V75_METADATA_FIELDS:
        raise ValueError("V75 runtime metadata fields changed")
    required_true = (
        "saved_runtime_training_gate_required",
        "bias_free_nonlinear_coefficient_decoder",
        "zero_preserving_coefficient_activation",
        "saved_runtime_training_gate_passed",
        "complete_scene_prefix_required",
        "full_unchanged_prefix_retained_separately",
        "prequestion_scene_key_value_cache",
        "all_environment_latents_attended",
        "positive_attention_floor",
        "bilinear_question_scene_value_interaction",
        "bias_free_nonlinear_coefficient_decoder",
        "zero_preserving_coefficient_activation",
    )
    required_false = (
        "question_only_output_path_exists",
        "question_dependent_scene_retrieval",
        "latent_selection_or_top_k_used",
        "training_answers_runtime_loaded",
        "answer_text_runtime_loaded",
        "answer_class_codebook_runtime_loaded",
        "teacher_cache_runtime_loaded",
        "oracle_runtime_loaded",
        "question_or_answer_text_serialized",
    )
    if (
        metadata.get("schema_version") != 75
        or metadata.get("architecture") != V75_RUNTIME_ARCHITECTURE
        or metadata.get("hidden_size") != 1536
        or metadata.get("control_tokens") != 4
        or metadata.get("environment_latents") != 256
        or metadata.get("query_count") != 4
        or metadata.get("model_dimension") != 128
        or metadata.get("coefficient_decoder_hidden_dimension") != 768
        or metadata.get("output_basis_rank") != 112
        or metadata.get("uniform_floor_mass") != 0.05
        or metadata.get("maximum_control_rms") != 0.25
        or metadata.get("weights_sha256") != observed_weights_sha256
        or metadata.get("base_checkpoint_sha256")
        != EXPECTED_BASE_CHECKPOINT_SHA256
        or metadata.get("base_runtime_config_sha256")
        != EXPECTED_RUNTIME_CONFIG_SHA256
        or metadata.get("source_v75_candidate_sha256")
        != EXPECTED_SOURCE_V75_CANDIDATE_SHA256
        or metadata.get("environmental_text_inputs") != []
        or any(metadata.get(field) is not True for field in required_true)
        or any(metadata.get(field) is not False for field in required_false)
    ):
        raise ValueError("V75 runtime checkpoint is outside the official contract")
    for field in (
        "weights_sha256",
        "base_checkpoint_sha256",
        "base_runtime_config_sha256",
        "source_v75_candidate_sha256",
        "source_v75_training_fit_state_sha256",
        "saved_runtime_training_gate_attestation_sha256",
    ):
        if not isinstance(metadata.get(field), str) or _SHA256.fullmatch(
            str(metadata[field])
        ) is None:
            raise ValueError(f"V75 runtime metadata has an invalid {field}")
    return metadata


def validate_v75_control_audit(value: object) -> dict[str, Any]:
    """Validate the per-answer proof that V75 consumed every scene latent."""

    audit = dict(_mapping(value, "control audit"))
    if set(audit) != _V75_AUDIT_FIELDS:
        raise ValueError("V75 control-audit fields changed")
    minimum = audit.get("minimum_attention_weight")
    maximum_rms = audit.get("maximum_control_rms")
    required_true = (
        "every_scene_token_influenced_output",
        "positive_attention_floor",
        "softmax_scene_attention_used",
        "bilinear_question_scene_value_interaction",
        "immutable_full_prefix_retained_separately",
        "prequestion_scene_key_value_cache",
        "zero_scene_produces_exact_zero_controls",
        "control_used",
        "saved_runtime_training_gate_required",
    )
    required_false = (
        "question_only_output_path_exists",
        "question_dependent_scene_retrieval",
        "latent_selection_or_top_k_used",
        "training_answers_runtime_loaded",
        "answer_text_runtime_loaded",
        "answer_class_codebook_runtime_loaded",
    )
    floor = 0.05 / 256
    if (
        audit.get("architecture") != V75_RUNTIME_ARCHITECTURE
        or audit.get("scene_token_count") != 258
        or audit.get("environment_latent_count") != 256
        or audit.get("control_token_count") != 4
        or audit.get("coefficient_decoder_hidden_dimension") != 768
        or isinstance(minimum, bool)
        or not isinstance(minimum, (int, float))
        or not math.isfinite(float(minimum))
        or float(minimum) < floor - 1e-8
        or isinstance(maximum_rms, bool)
        or not isinstance(maximum_rms, (int, float))
        or not math.isfinite(float(maximum_rms))
        or not 0.0 <= float(maximum_rms) <= 0.25 + 1e-6
        or any(audit.get(field) is not True for field in required_true)
        or any(audit.get(field) is not False for field in required_false)
    ):
        raise ValueError("V75 control audit violates full-scene inference")
    return audit


def authenticate_v75_control_checkpoint(path: str | Path) -> V75ControlIdentity:
    source = safe_prediction_input(path, "control checkpoint", kind="directory")
    if {item.name for item in source.iterdir()} != set(_CONTROL_FILES):
        raise ValueError("V75 control checkpoint must contain exactly two runtime files")
    entries: list[dict[str, Any]] = []
    digests: dict[str, str] = {}
    for name in _CONTROL_FILES:
        item = reject_symlink_components(source / name, f"control {name}")
        if not item.is_file() or item.is_symlink():
            raise ValueError(f"V75 control checkpoint entry is unsafe: {item}")
        digest = sha256_file(item)
        digests[name] = digest
        entries.append(
            {"name": name, "sha256": digest, "size_bytes": item.stat().st_size}
        )
    try:
        metadata_value = json.loads(
            (source / "runtime_metadata.json").read_text(encoding="utf-8")
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("V75 runtime metadata is invalid JSON") from error
    metadata = validate_v75_runtime_metadata(
        metadata_value,
        observed_weights_sha256=digests["control.safetensors"],
    )
    payload = json.dumps(
        entries, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return V75ControlIdentity(
        path=source,
        sha256=hashlib.sha256(payload).hexdigest(),
        weights_sha256=digests["control.safetensors"],
        runtime_metadata_sha256=digests["runtime_metadata.json"],
        metadata=metadata,
    )


__all__ = [
    "ARTIFACT",
    "DEFAULT_BASE_CHECKPOINT",
    "DEFAULT_CONTROL_CHECKPOINT",
    "DEFAULT_PREDICTIONS",
    "DEFAULT_QUESTIONS_MANIFEST",
    "DEFAULT_REFERENCES",
    "DEFAULT_RUNTIME_CONFIG",
    "DEFAULT_SCORE",
    "EXPECTED_BASE_CHECKPOINT_SHA256",
    "EXPECTED_QUESTION_COUNT",
    "EXPECTED_REFERENCE_SHA256",
    "EXPECTED_RUNTIME_CONFIG_SHA256",
    "EXPECTED_SCENE_IDS",
    "EXPECTED_SOURCE_V75_CANDIDATE_SHA256",
    "V75_RUNTIME_ARCHITECTURE",
    "V75ControlIdentity",
    "authenticate_v75_control_checkpoint",
    "reject_symlink_components",
    "resolve_path",
    "safe_prediction_input",
    "safe_prediction_output",
    "sha256_file",
    "validate_official_question_manifest",
    "validate_v75_control_audit",
    "validate_v75_runtime_metadata",
]
