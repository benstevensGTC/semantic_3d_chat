"""Fail-closed one-shot terminal for V66 always-on continuous control.

All public and candidate inputs are authenticated before an immutable launch
claim is created.  Only then is the scorer-only reference file opened, exactly
once.  The report stores aggregate measurements and hashes, never answers.

This module is terminal-only.  Training and chat inference do not import it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import torch

from semantic_3d_chat.chat.question_control_runtime import _load_control_head
from semantic_3d_chat.chat.runtime_config import (
    effective_runtime_config_sha256,
    load_runtime_config,
    runtime_config_file_sha256,
)
from semantic_3d_chat.evaluation import v62_internal_validation_terminal_gate as common
from semantic_3d_chat.evaluation import v62_pair_disjoint_preregistration as boundary
from semantic_3d_chat.evaluation import v66_internal_validation_preregistration as contract
from semantic_3d_chat.evaluation.prediction_artifacts import (
    PROVENANCE_SCHEMA_VERSION,
    checkpoint_fingerprint,
    scene_map_manifest_sha256,
    validate_scene_map_manifest,
)
from semantic_3d_chat.evaluation.question_manifest import QuestionManifest
from semantic_3d_chat.evaluation.v62_pair_disjoint_preregistration import (
    validate_baseline_lock,
)
from semantic_3d_chat.evaluation.v66_internal_validation_preregistration import (
    validate_v66_internal_validation_preregistration,
)
from semantic_3d_chat.scene_encoder.question_control_v7 import (
    AlwaysOnTeacherBasisFullSceneQuestionControlV7,
)
from semantic_3d_chat.training.question_control_v7_checkpoint import (
    v7_value_state_sha256,
)

ARTIFACT: Final[str] = "v66_always_on_internal_validation_terminal"
SCHEMA: Final[str] = "semantic_3d_chat.v66.always_on_internal_validation_terminal.v1"
CLAIM_ARTIFACT: Final[str] = "v66_always_on_internal_validation_launch_claim"
CLAIM_SCHEMA: Final[str] = "semantic_3d_chat.v66.always_on_internal_validation_claim.v1"
ARCHITECTURE: Final[str] = "always_on_teacher_basis_full_scene_control_v7"
TRAINING_REPORT_ARTIFACT: Final[str] = "v66b_allrow_paired_opposite_pair_disjoint_training"
NATURAL_RUN_KIND: Final[str] = "continuous_scene_question_control_v1"
SCENE_SWAP_RUN_KIND: Final[str] = "continuous_scene_question_control_scene_swap_v1"
SCENE_SWAP_CONDITION_PREFIX: Final[str] = "all_questions_bidirectional_scene_swap"
PINNED_TRAINING_BASELINE_LOCK_SHA256: Final[str] = (
    "b1f20e64889116cceb0904ecb3842a6e43fcd6fa3cb0675c32a24f4d278e55e6"
)

_SHA256 = re.compile(r"[0-9a-f]{64}")
_SCENE_ID = re.compile(r"scene_[0-9]{6}")
_QUESTION_ID = re.compile(r"q_[0-9]{6}")
_PROTECTED_FRESH_SCENES: Final[tuple[str, ...]] = tuple(
    f"scene_{number:06d}" for number in range(57, 63)
)
_PROTECTED_FINAL_SCENES: Final[tuple[str, ...]] = tuple(
    f"scene_{number:06d}" for number in range(25, 31)
)
_PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[3]
_V7_AUDIT_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "architecture",
        "scene_token_count",
        "environment_latent_count",
        "control_token_count",
        "scene_moment_count",
        "every_scene_token_influenced_output",
        "question_dependent_scene_retrieval",
        "softmax_scene_attention_used",
        "control_values_scene_question_bilinear",
        "gate_scene_question_conditioned",
        "inherited_v60_state_frozen",
        "separate_question_scene_route_projections",
        "normalized_route_factors",
        "all_scene_moments_consumed_by_route",
        "low_rank_bilinear_route",
        "route_uses_inherited_value_trunk",
        "route_factor_rank",
        "gate_probability",
        "control_used",
        "maximum_control_rms",
        "exact_no_control_route",
        "activation_rms",
        "activation_rms_threshold",
        "exact_no_control_below_threshold",
        "always_on_continuous_control",
        "legacy_route_parameters_ignored",
        "saved_runtime_training_gate_required",
    }
)


@dataclass(frozen=True)
class CandidateInputs:
    """All authenticated non-reference inputs frozen by the launch claim."""

    internal_preregistration: dict[str, Any]
    internal_preregistration_sha256: str
    parent_preregistration: dict[str, Any]
    parent_preregistration_sha256: str
    training_preregistration_sha256: str
    questions: QuestionManifest
    questions_manifest_sha256: str
    baseline: dict[str, Any]
    baseline_sha256: str
    natural_rows: tuple[dict[str, Any], ...]
    natural_sha256: str
    natural_provenance_sha256: str
    swap_rows: tuple[dict[str, Any], ...]
    swap_sha256: str
    swap_provenance_sha256: str
    base_checkpoint_sha256: str
    base_checkpoint_files: tuple[dict[str, Any], ...]
    control_checkpoint_sha256: str
    control_files: dict[str, dict[str, Any]]
    control_metadata: dict[str, Any]
    runtime_config_file_sha256: str
    runtime_config_effective_sha256: str
    training_report_sha256: str
    training_report_artifact: str
    natural_prefixes: dict[str, str]
    natural_signatures: dict[str, str]


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(value: object, *, pretty: bool = False) -> bytes:
    options: dict[str, Any] = {
        "sort_keys": True,
        "ensure_ascii": False,
        "allow_nan": False,
    }
    options["indent" if pretty else "separators"] = 2 if pretty else (",", ":")
    return (json.dumps(value, **options) + "\n").encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return _sha256_bytes(_canonical_json_bytes(value))


def _resolve(path: str | Path) -> Path:
    return Path(os.path.abspath(Path(path).expanduser()))


def _reject_symlink_components(path: Path, label: str) -> None:
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current /= component
        if current.is_symlink():
            raise ValueError(f"V66 {label} path contains a symbolic link: {current}")


def _safe_existing_file(path: str | Path, label: str) -> Path:
    source = _resolve(path)
    _reject_symlink_components(source, label)
    if not source.is_file():
        raise FileNotFoundError(f"V66 {label} is unavailable: {source}")
    return source


def _safe_existing_directory(path: str | Path, label: str) -> Path:
    source = _resolve(path)
    _reject_symlink_components(source, label)
    if not source.is_dir():
        raise FileNotFoundError(f"V66 {label} is unavailable: {source}")
    return source


def _safe_destination(path: str | Path, label: str) -> Path:
    destination = _resolve(path)
    _reject_symlink_components(destination, label)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"V66 immutable {label} already exists: {destination}")
    return destination


def _load_json_object(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"V66 {label} is invalid JSON") from exc
    if not isinstance(value, dict):
        raise TypeError(f"V66 {label} must be a JSON object")
    return value, raw


def _load_jsonl(path: Path, label: str) -> tuple[tuple[dict[str, Any], ...], bytes]:
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"V66 {label} is not UTF-8") from exc
    if not text.endswith("\n"):
        raise ValueError(f"V66 {label} must end with a newline")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line:
            raise ValueError(f"V66 {label} has a blank line at {line_number}")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"V66 {label} line {line_number} is invalid JSON") from exc
        if not isinstance(value, dict):
            raise TypeError(f"V66 {label} line {line_number} must be an object")
        rows.append(value)
    return tuple(rows), raw


def _atomic_create_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_canonical_json_bytes(dict(payload), pretty=True))
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            raise FileExistsError(f"V66 immutable artifact already exists: {path}") from None
    finally:
        temporary.unlink(missing_ok=True)


def _expected_validation_scenes() -> tuple[str, ...]:
    specs = {spec.pair_id: spec for spec in boundary.PAIR_INVENTORY}
    return tuple(
        scene_id
        for pair_id in boundary.INTERNAL_VALIDATION_PAIR_IDS
        for scene_id in specs[pair_id].scene_ids
    )


def _load_and_validate_parent_preregistration(path: Path) -> tuple[dict[str, Any], str]:
    value, raw = _load_json_object(path, "parent V62 preregistration")
    common._validate_preregistration(value, raw)
    return value, _sha256_bytes(raw)


def _load_training_preregistration(path: Path) -> tuple[dict[str, Any], str]:
    value, raw = _load_json_object(path, "V66 training preregistration")
    digest = _sha256_bytes(raw)
    if (
        digest != contract.PINNED_V66_TRAINING_PREREGISTRATION_SHA256
        or value.get("schema_version") != 1
        or value.get("artifact") != "v66b_allrow_paired_opposite_training_preregistration"
        or value.get("status") != "locked_before_v66b_controller_training_or_generation"
        or value.get("thresholds") != contract.TRAINING_THRESHOLDS
        or value.get("controls", {}).get("unverified_native_answer_embedding_fallback_permitted")
        is not False
        or value.get("controls", {}).get("exact_paired_opposite_scene_prefix_and_signature")
        is not True
        or value.get("controls", {}).get("same_question_byte_identity_required") is not True
        or value.get("controls", {}).get(
            "answer_follows_injected_scene_scored_against_opposite_reference"
        )
        is not True
        or value.get("scope", {}).get("training_only") is not True
        or any(
            value.get("scope", {}).get(field) is not False
            for field in (
                "validation_inputs_used",
                "scorer_inputs_used",
                "oracle_loaded",
                "fresh_development_loaded",
                "deferred_final_loaded",
            )
        )
    ):
        raise ValueError("V66 training preregistration differs from its public pin")
    return value, digest


def _control_checkpoint_fingerprint(path: Path) -> tuple[str, dict[str, dict[str, Any]]]:
    expected = ("control.safetensors", "runtime_metadata.json")
    if sorted(item.name for item in path.iterdir()) != sorted(expected):
        raise ValueError("V66 control checkpoint inventory is not runtime-minimal")
    entries: list[dict[str, Any]] = []
    files: dict[str, dict[str, Any]] = {}
    for name in expected:
        item = path / name
        if item.is_symlink() or not item.is_file():
            raise ValueError(f"V66 control checkpoint entry is not regular: {item}")
        entry = {
            "name": name,
            "sha256": _sha256_file(item),
            "size_bytes": item.stat().st_size,
        }
        entries.append(entry)
        files[name] = {"sha256": entry["sha256"], "size_bytes": entry["size_bytes"]}
    payload = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    return _sha256_bytes(payload), files


def _validate_control_checkpoint(
    path: Path,
    *,
    base_checkpoint_sha256: str,
    runtime_config_sha256: str,
) -> tuple[str, dict[str, dict[str, Any]], dict[str, Any]]:
    fingerprint, files = _control_checkpoint_fingerprint(path)
    module, metadata = _load_control_head(
        path,
        hidden_size=1536,
        device=torch.device("cpu"),
    )
    if (
        type(module) is not AlwaysOnTeacherBasisFullSceneQuestionControlV7
        or metadata.get("schema_version") != 7
        or metadata.get("architecture") != ARCHITECTURE
        or metadata.get("base_checkpoint_sha256") != base_checkpoint_sha256
        or metadata.get("base_runtime_config_sha256") != runtime_config_sha256
        or metadata.get("expected_environment_latents") != 256
        or metadata.get("control_tokens") != 4
        or metadata.get("always_on_continuous_control") is not True
        or metadata.get("complete_scene_prefix_required") is not True
        or metadata.get("question_dependent_scene_retrieval") is not False
        or metadata.get("training_answers_runtime_loaded") is not False
        or metadata.get("answer_class_codebook_runtime_loaded") is not False
        or metadata.get("environmental_text_inputs") != []
        or metadata.get("saved_runtime_training_gate_required") is not True
        or metadata.get("saved_runtime_training_gate_passed") is not True
        or metadata.get("weights_sha256") != files["control.safetensors"]["sha256"]
        or v7_value_state_sha256(module) != metadata.get("source_v66_training_fit_state_sha256")
    ):
        raise ValueError("V66 sealed schema-7 checkpoint contract is invalid")
    return fingerprint, files, metadata


def _provenance_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "schema_version",
        "config_sha256",
        "config_file_sha256",
        "checkpoint_sha256",
        "references_sha256",
        "scene_map_manifest_sha256",
        "split",
        "run_kind",
        "condition",
    )
    return {field: value.get(field) for field in fields}


def _validate_provenance(
    value: Mapping[str, Any],
    *,
    raw: bytes,
    label: str,
    run_kind: str,
    condition: str,
    questions_sha256: str,
    base_checkpoint_sha256: str,
    base_checkpoint_files: Sequence[Mapping[str, Any]],
    config_file_sha256: str,
    config_effective_sha256: str,
) -> str:
    required = {
        "schema_version",
        "config_sha256",
        "config_file_sha256",
        "checkpoint_sha256",
        "references_sha256",
        "scene_map_manifest_sha256",
        "split",
        "run_kind",
        "condition",
        "provenance_sha256",
        "config_path",
        "checkpoint_path",
        "checkpoint_files",
        "references_path",
        "scene_map_manifest",
    }
    maps = validate_scene_map_manifest(value.get("scene_map_manifest"))
    identity = _provenance_identity(value)
    if (
        set(value) != required
        or value.get("schema_version") != PROVENANCE_SCHEMA_VERSION
        or value.get("provenance_sha256") != common._canonical_identity_sha256(identity)
        or value.get("config_sha256") != config_effective_sha256
        or value.get("config_file_sha256") != config_file_sha256
        or value.get("checkpoint_sha256") != base_checkpoint_sha256
        or value.get("checkpoint_files") != list(base_checkpoint_files)
        or value.get("references_sha256") != questions_sha256
        or value.get("scene_map_manifest_sha256") != scene_map_manifest_sha256(maps)
        or set(maps) != set(_expected_validation_scenes())
        or value.get("run_kind") != run_kind
        or value.get("condition") != condition
        or value.get("split") not in {"train", "validation"}
    ):
        raise ValueError(f"V66 {label} provenance does not bind the exact run")
    return _sha256_bytes(raw)


def _valid_v7_audit(value: object, metadata: Mapping[str, Any]) -> bool:
    if not isinstance(value, Mapping) or set(value) != _V7_AUDIT_FIELDS:
        return False
    probability = value.get("gate_probability")
    maximum = value.get("maximum_control_rms")
    return bool(
        value.get("architecture") == ARCHITECTURE
        and value.get("scene_token_count") == 258
        and value.get("environment_latent_count") == 256
        and value.get("control_token_count") == 4
        and value.get("scene_moment_count") == metadata.get("moment_count")
        and value.get("every_scene_token_influenced_output") is True
        and value.get("question_dependent_scene_retrieval") is False
        and value.get("softmax_scene_attention_used") is False
        and value.get("control_values_scene_question_bilinear") is True
        and value.get("gate_scene_question_conditioned") is False
        and value.get("control_used") is True
        and value.get("exact_no_control_route") is False
        and value.get("always_on_continuous_control") is True
        and value.get("legacy_route_parameters_ignored") is True
        and value.get("saved_runtime_training_gate_required") is True
        and value.get("activation_rms") is None
        and value.get("activation_rms_threshold") is None
        and value.get("exact_no_control_below_threshold") is False
        and type(probability) in {int, float}
        and math.isfinite(float(probability))
        and float(probability) > 0.999999
        and type(maximum) in {int, float}
        and math.isfinite(float(maximum))
        and 0.0 <= float(maximum) <= float(metadata.get("maximum_control_rms", -1.0))
    )


def _validate_natural_predictions(
    rows: Sequence[Mapping[str, Any]],
    *,
    manifest: QuestionManifest,
    provenance_sha256: str,
    control_checkpoint_sha256: str,
    control_metadata: Mapping[str, Any],
    baseline: Mapping[str, Any],
) -> tuple[dict[str, str], dict[str, str]]:
    if len(rows) != 384:
        raise ValueError("V66 natural candidate needs exactly 384 predictions")
    expected = [(row.scene_id, row.question_id) for row in manifest.questions]
    actual: list[tuple[str, str]] = []
    prefixes: defaultdict[str, set[str]] = defaultdict(set)
    signatures: defaultdict[str, set[str]] = defaultdict(set)
    required = {
        "scene_id",
        "question_id",
        "predicted_answer",
        "prefix_hash",
        "scene_control_signature_sha256",
        "control_checkpoint_sha256",
        "control_audit",
        "provenance_sha256",
    }
    for index, row in enumerate(rows, start=1):
        scene_id = row.get("scene_id")
        question_id = row.get("question_id")
        prefix = row.get("prefix_hash")
        signature = row.get("scene_control_signature_sha256")
        if (
            not required <= set(row)
            or not isinstance(scene_id, str)
            or _SCENE_ID.fullmatch(scene_id) is None
            or not isinstance(question_id, str)
            or _QUESTION_ID.fullmatch(question_id) is None
            or not isinstance(row.get("predicted_answer"), str)
            or not isinstance(prefix, str)
            or _SHA256.fullmatch(prefix) is None
            or not isinstance(signature, str)
            or _SHA256.fullmatch(signature) is None
            or row.get("control_checkpoint_sha256") != control_checkpoint_sha256
            or row.get("provenance_sha256") != provenance_sha256
            or not _valid_v7_audit(row.get("control_audit"), control_metadata)
        ):
            raise ValueError(f"V66 natural prediction {index} has invalid content")
        actual.append((scene_id, question_id))
        prefixes[scene_id].add(prefix)
        signatures[scene_id].add(signature)
    if actual != expected or len(set(actual)) != 384:
        raise ValueError("V66 natural prediction ordering/inventory changed")
    scenes = set(_expected_validation_scenes())
    if (
        set(prefixes) != scenes
        or set(signatures) != scenes
        or any(len(values) != 1 for values in prefixes.values())
        or any(len(values) != 1 for values in signatures.values())
    ):
        raise ValueError("V66 natural prefix/signature is not immutable per scene")
    fixed_prefixes = {scene: next(iter(values)) for scene, values in prefixes.items()}
    fixed_signatures = {scene: next(iter(values)) for scene, values in signatures.items()}
    if fixed_prefixes != baseline.get("scene_prefix_hashes"):
        raise ValueError("V66 natural scene prefixes differ from the public V54 lock")
    if len(set(fixed_signatures.values())) != len(fixed_signatures):
        raise ValueError("V66 scene signatures are not distinct across scene maps")
    return fixed_prefixes, fixed_signatures


def _validate_swap_predictions(
    rows: Sequence[Mapping[str, Any]],
    *,
    manifest: QuestionManifest,
    provenance_sha256: str,
    control_checkpoint_sha256: str,
    control_metadata: Mapping[str, Any],
    natural_prefixes: Mapping[str, str],
    natural_signatures: Mapping[str, str],
) -> None:
    if len(rows) != 384:
        raise ValueError("V66 scene-swap requires all 384 blind predictions")
    expected = [(row.scene_id, row.question_id) for row in manifest.questions]
    questions = {(row.scene_id, row.question_id): row.question for row in manifest.questions}
    paired = {
        scene_id: opposite
        for spec in common._expected_validation_specs()
        for scene_id, opposite in (
            (spec.reference_scene_id, spec.counterfactual_scene_id),
            (spec.counterfactual_scene_id, spec.reference_scene_id),
        )
    }
    required = {
        "scene_id",
        "question_id",
        "injected_scene_id",
        "predicted_answer",
        "prefix_hash",
        "scene_control_signature_sha256",
        "question_sha256",
        "control_checkpoint_sha256",
        "control_audit",
        "provenance_sha256",
    }
    actual: list[tuple[str, str]] = []
    for index, row in enumerate(rows, start=1):
        key = row.get("scene_id"), row.get("question_id")
        injected = row.get("injected_scene_id")
        if (
            not required <= set(row)
            or key not in set(expected)
            or injected != paired.get(str(key[0]))
            or not isinstance(row.get("predicted_answer"), str)
            or row.get("prefix_hash") != natural_prefixes.get(str(injected))
            or row.get("scene_control_signature_sha256") != natural_signatures.get(str(injected))
            or row.get("question_sha256")
            != _sha256_bytes(questions[(str(key[0]), str(key[1]))].encode("utf-8"))
            or row.get("control_checkpoint_sha256") != control_checkpoint_sha256
            or row.get("provenance_sha256") != provenance_sha256
            or not _valid_v7_audit(row.get("control_audit"), control_metadata)
        ):
            raise ValueError(f"V66 scene-swap prediction {index} has invalid content")
        actual.append((str(key[0]), str(key[1])))
    if actual != expected or len(set(actual)) != 384:
        raise ValueError("V66 scene-swap ordering/inventory changed")


def _all_true(value: object, expected: Mapping[str, bool]) -> bool:
    return bool(
        isinstance(value, Mapping)
        and dict(value) == dict(expected)
        and expected
        and all(expected.values())
    )


def _cv_training_checks(
    metrics: Mapping[str, Any], thresholds: Mapping[str, Any]
) -> dict[str, bool]:
    per_type = metrics.get("per_type_by_sha256")
    if not isinstance(per_type, Mapping):
        raise TypeError("V66b CV report lacks per-type measurements")
    per_type_checks = {
        f"per_type_{answer_type}": int(
            per_type.get(hashlib.sha256(answer_type.encode()).hexdigest(), {}).get("exact", -1)
        )
        >= int(minimum)
        for answer_type, minimum in thresholds["per_type_minimum_exact"]
    }
    return {
        "held_supported_exact": int(metrics.get("supported_exact", -1))
        >= int(thresholds["held_supported_exact_minimum"]),
        "held_supported_total": int(metrics.get("supported_total", -1))
        == int(thresholds["held_supported_total"]),
        "held_unsupported_total": int(metrics.get("unsupported_total", -1))
        == int(thresholds["held_unsupported_total"]),
        "complete_inventory": int(metrics.get("inventory_total", -1)) == 576,
        "eligible_fold_total": int(metrics.get("eligible_fold_total", -1))
        == int(thresholds["eligible_fold_total"]),
        "eligible_folds_with_exact_hit": int(metrics.get("eligible_folds_with_exact_hit", -1))
        >= int(thresholds["eligible_folds_with_exact_hit_minimum"]),
        "held_changed_side_exact": int(metrics.get("changed_side_exact", -1))
        >= int(thresholds["held_changed_side_exact_minimum"]),
        "held_changed_side_total": int(metrics.get("changed_side_total", -1))
        == int(thresholds["held_changed_side_total"]),
        "held_complete_units": int(metrics.get("complete_changed_units", -1))
        >= int(thresholds["held_complete_unit_minimum"]),
        "held_complete_unit_total": int(metrics.get("changed_unit_total", -1))
        == int(thresholds["held_complete_unit_total"]),
        "held_prediction_change_units": int(metrics.get("prediction_change_units", -1))
        >= int(thresholds["held_prediction_change_unit_minimum"]),
        "held_prediction_change_unit_total": int(metrics.get("changed_unit_total", -1))
        == int(thresholds["held_prediction_change_unit_total"]),
        **per_type_checks,
    }


def _final_training_checks(
    metrics: Mapping[str, Any], thresholds: Mapping[str, Any]
) -> dict[str, bool]:
    return {
        "final_exact": int(metrics.get("supported_exact", -1))
        >= int(thresholds["final_exact_minimum"]),
        "final_total": int(metrics.get("supported_total", -1)) == int(thresholds["final_total"]),
        "no_unsupported_final_rows": int(metrics.get("unsupported_total", -1)) == 0,
        "complete_inventory": int(metrics.get("inventory_total", -1))
        == int(thresholds["final_total"]),
        "final_complete_units": int(metrics.get("complete_changed_units", -1))
        >= int(thresholds["final_complete_unit_minimum"]),
        "final_complete_unit_total": int(metrics.get("changed_unit_total", -1))
        == int(thresholds["final_complete_unit_total"]),
    }


def _paired_opposite_training_checks(
    metrics: Mapping[str, Any], thresholds: Mapping[str, Any]
) -> dict[str, bool]:
    sides = int(thresholds["paired_opposite_side_total"])
    units = int(thresholds["paired_opposite_unit_total"])
    return {
        "follows_injected_side_minimum": int(metrics.get("answer_follows_injected_scene", -1))
        >= int(thresholds["paired_opposite_follows_side_minimum"]),
        "paired_opposite_side_total": int(metrics.get("paired_opposite_side_total", -1)) == sides,
        "follows_injected_complete_unit_minimum": int(
            metrics.get("answer_follows_injected_scene_complete_units", -1)
        )
        >= int(thresholds["paired_opposite_follows_complete_minimum"]),
        "paired_opposite_unit_total": int(metrics.get("paired_opposite_unit_total", -1)) == units,
        "original_reference_exact_ceiling": int(
            metrics.get("answer_matches_original_reference", sides + 1)
        )
        <= int(thresholds["paired_opposite_original_exact_maximum"]),
        "original_reference_complete_ceiling": int(
            metrics.get("answer_matches_original_reference_complete_units", units + 1)
        )
        <= int(thresholds["paired_opposite_original_complete_maximum"]),
        "question_identity_complete": int(metrics.get("question_identity_count", -1)) == sides,
        "exact_paired_scene_complete": int(metrics.get("exact_paired_scene_count", -1)) == sides,
        "exact_paired_scene_prefix_complete": int(
            metrics.get("exact_paired_scene_prefix_count", -1)
        )
        == sides,
        "exact_paired_scene_signature_complete": int(
            metrics.get("exact_paired_scene_signature_count", -1)
        )
        == sides,
        "differing_reference_complete": int(metrics.get("differing_reference_count", -1)) == sides,
        "cross_swap_complete": int(metrics.get("cross_swap_complete_units", -1)) == units,
        "no_question_or_answer_text_stored": metrics.get("answer_or_question_text_stored") is False,
    }


def _validate_saved_runtime_reload(
    value: object,
    *,
    metadata: Mapping[str, Any],
    thresholds: Mapping[str, Any],
) -> None:
    if not isinstance(value, Mapping):
        raise TypeError("V66b report lacks a saved-runtime reload gate")
    metrics = value.get("metrics")
    if not isinstance(metrics, Mapping):
        raise TypeError("V66b saved-runtime gate lacks measurements")
    checks = _final_training_checks(metrics, thresholds)
    fit_state = metadata.get("source_v66_training_fit_state_sha256")
    attestation = metadata.get("saved_runtime_training_gate_attestation_sha256")
    production_device = value.get("production_device")
    if (
        value.get("strict_loader_passed") is not True
        or value.get("architecture") != ARCHITECTURE
        or value.get("training_fit_state_sha256") != fit_state
        or value.get("gate_attestation_sha256") != attestation
        or value.get("reloaded_state_exact") is not True
        or value.get("raw_question_token_embeddings_used") is not True
        or not isinstance(production_device, str)
        or not production_device
        or value.get("passed_before_publication") is not True
        or not _all_true(value.get("checks"), checks)
    ):
        raise ValueError("V66b saved-runtime gate did not pass exactly")
    expected_attestation = _canonical_sha256(
        {
            "schema_version": 1,
            "artifact": "v66_saved_runtime_training_gate_attestation",
            "training_fit_state_sha256": fit_state,
            "production_device": production_device,
            "raw_question_token_embeddings_used": True,
            "behavior": dict(metrics),
            "checks": checks,
            "answer_or_question_text_stored": False,
        }
    )
    if expected_attestation != attestation:
        raise ValueError("V66b saved-runtime gate attestation digest is invalid")


def _validate_training_report(
    path: Path,
    *,
    training_preregistration: Mapping[str, Any],
    training_preregistration_sha256: str,
    baseline_sha256: str,
    parent_preregistration: Mapping[str, Any],
    base_checkpoint_sha256: str,
    runtime_config_sha256: str,
    control_files: Mapping[str, Mapping[str, Any]],
    control_metadata: Mapping[str, Any],
) -> tuple[str, str]:
    report, raw = _load_json_object(path, "V66b training report")
    authorization = report.get("authorization")
    architecture = report.get("architecture")
    scope = report.get("scope")
    cv = report.get("cv")
    final_fit = report.get("final_fit")
    dependence = report.get("paired_opposite_scene_dependence")
    checkpoint = report.get("checkpoint")
    thresholds = training_preregistration["thresholds"]
    if (
        report.get("schema_version") != 1
        or report.get("artifact") != TRAINING_REPORT_ARTIFACT
        or report.get("promotion_eligible") is not False
        or report.get("terminal_reason")
        != "training_and_paired_dependence_gates_passed_checkpoint_saved"
        or report.get("thresholds") != thresholds
        or report.get("thresholds") != contract.TRAINING_THRESHOLDS
        or not isinstance(authorization, Mapping)
        or authorization.get("baseline_lock_sha256") != baseline_sha256
        or authorization.get("preregistration_sha256") != training_preregistration_sha256
        or authorization.get("filtered_training_qa_sha256")
        != parent_preregistration["artifacts"]["filtered_training"]["sha256"]
        or authorization.get("training_baseline_lock_sha256")
        != PINNED_TRAINING_BASELINE_LOCK_SHA256
        or not isinstance(architecture, Mapping)
        or architecture
        != {
            "name": ARCHITECTURE,
            "complete_scene_prefix": True,
            "scene_latents": 256,
            "question_dependent_scene_retrieval": False,
            "environmental_text_inputs": [],
            "runtime_answer_codebook": False,
        }
        or not isinstance(scope, Mapping)
        or scope
        != {
            "training_only": True,
            "gemma_frozen": True,
            "gemma_backward_used": False,
            "validation_inputs_used": False,
            "scorer_inputs_used": False,
            "oracle_loaded": False,
            "fresh_development_loaded": False,
            "deferred_final_loaded": False,
            "question_or_answer_text_stored": False,
        }
        or not isinstance(report.get("teacher_audit"), Mapping)
        or not isinstance(report.get("work_manifest_sha256"), str)
        or _SHA256.fullmatch(str(report.get("work_manifest_sha256"))) is None
    ):
        raise ValueError("V66b training report provenance is invalid")

    if not isinstance(cv, Mapping) or not isinstance(cv.get("metrics"), Mapping):
        raise TypeError("V66b report lacks pair-disjoint CV")
    cv_checks = _cv_training_checks(cv["metrics"], thresholds)
    folds = cv.get("folds")
    if (
        cv.get("protocol") != "leave_one_counterfactual_pair_out_all_576_rows"
        or cv.get("passed") is not True
        or not _all_true(cv.get("checks"), cv_checks)
        or not isinstance(folds, list)
        or len(folds) != 12
        or [fold.get("held_pair_id") for fold in folds if isinstance(fold, Mapping)]
        != list(boundary.TRAIN_PAIR_IDS)
        or any(
            not isinstance(fold, Mapping)
            or fold.get("held_rows_used_for_optimization") is not False
            or fold.get("held_teacher_sources_used") is not False
            or not isinstance(fold.get("behavior"), Mapping)
            or not isinstance(fold.get("fit"), Mapping)
            for fold in folds
        )
    ):
        raise ValueError("V66b pair-disjoint training gates did not pass exactly")

    if not isinstance(final_fit, Mapping) or not isinstance(final_fit.get("metrics"), Mapping):
        raise TypeError("V66b report lacks its all-training fit gate")
    final_checks = _final_training_checks(final_fit["metrics"], thresholds)
    if (
        final_fit.get("passed") is not True
        or not _all_true(final_fit.get("checks"), final_checks)
        or not isinstance(final_fit.get("fit_state_sha256"), str)
        or _SHA256.fullmatch(str(final_fit.get("fit_state_sha256"))) is None
    ):
        raise ValueError("V66b all-training behavior gate did not pass exactly")

    if not isinstance(dependence, Mapping) or not isinstance(dependence.get("metrics"), Mapping):
        raise TypeError("V66b report lacks the paired-opposite dependence gate")
    dependence_checks = _paired_opposite_training_checks(dependence["metrics"], thresholds)
    if (
        dependence.get("protocol")
        != (
            "exact_counterfactual_paired_opposite_scene_prefix_and_signature_"
            "same_byte_identical_question"
        )
        or dependence.get("passed") is not True
        or not _all_true(dependence.get("checks"), dependence_checks)
    ):
        raise ValueError("V66b paired-opposite scene-dependence gate did not pass")

    expected_checkpoint = {
        "weights_sha256": control_files["control.safetensors"]["sha256"],
        "runtime_metadata_sha256": control_files["runtime_metadata.json"]["sha256"],
        "source_v66_training_fit_state_sha256": control_metadata[
            "source_v66_training_fit_state_sha256"
        ],
    }
    if (
        checkpoint != expected_checkpoint
        or control_metadata.get("base_checkpoint_sha256") != base_checkpoint_sha256
        or control_metadata.get("base_runtime_config_sha256") != runtime_config_sha256
    ):
        raise ValueError("V66b report/checkpoint file binding is invalid")
    _validate_saved_runtime_reload(
        report.get("saved_runtime_reload"),
        metadata=control_metadata,
        thresholds=thresholds,
    )
    return _sha256_bytes(raw), TRAINING_REPORT_ARTIFACT


def _minimum(value: Mapping[str, Any]) -> int:
    minimum = value.get("minimum")
    if isinstance(minimum, bool) or not isinstance(minimum, int):
        raise TypeError("V66 preregistered minimum must be an integer")
    return minimum


def score_populations(
    candidate: CandidateInputs,
    references: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Score sealed populations without retaining answer text in the result."""

    questions = {(row.scene_id, row.question_id): row for row in candidate.questions.questions}
    refs = {(str(row["scene_id"]), str(row["question_id"])): row for row in references}
    natural = {
        (str(row["scene_id"]), str(row["question_id"])): row for row in candidate.natural_rows
    }
    swaps = {(str(row["scene_id"]), str(row["question_id"])): row for row in candidate.swap_rows}
    if (
        set(refs) != set(questions)
        or set(natural) != set(questions)
        or set(swaps) != set(questions)
    ):
        raise ValueError("V66 scorer, natural, and swap populations differ")

    canonical_exact = 0
    changed_side_exact = 0
    per_route: dict[str, Counter[str]] = defaultdict(Counter)
    per_type: dict[str, Counter[str]] = defaultdict(Counter)
    per_family: dict[str, Counter[str]] = defaultdict(Counter)
    per_change: dict[str, Counter[str]] = defaultdict(Counter)
    for key, reference in refs.items():
        exact = common._answer_matches(str(natural[key]["predicted_answer"]), reference)
        changed = reference["route_label"] is True
        canonical_exact += int(exact)
        changed_side_exact += int(changed and exact)
        route = "changed" if changed else "unchanged"
        answer_type = str(reference["answer_type"])
        family = boundary._question_family(questions[key].question)
        change_type = str(reference["counterfactual_change_type"])
        for target, name in (
            (per_route, route),
            (per_type, answer_type),
            (per_family, family),
            (per_change, change_type),
        ):
            target[name]["total"] += 1
            target[name]["exact"] += int(exact)

    groups: defaultdict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for reference in references:
        groups[
            (
                str(reference["counterfactual_pair_id"]),
                str(reference["counterfactual_question_key"]),
            )
        ].append(reference)
    changed_groups = {
        key: values for key, values in groups.items() if values[0]["route_label"] is True
    }
    if len(changed_groups) != 26:
        raise ValueError("V66 internal validation expected 26 changed paired units")

    complete_units = 0
    correct_direction = 0
    question_identity = 0
    distinct_prefixes = 0
    distinct_signatures = 0
    for members in changed_groups.values():
        if len(members) != 2:
            raise ValueError("V66 changed scorer unit does not contain two sides")
        first, second = members
        first_key = str(first["scene_id"]), str(first["question_id"])
        second_key = str(second["scene_id"]), str(second["question_id"])
        first_prediction = str(natural[first_key]["predicted_answer"])
        second_prediction = str(natural[second_key]["predicted_answer"])
        complete_units += int(
            common._answer_matches(first_prediction, first)
            and common._answer_matches(second_prediction, second)
        )
        correct_direction += int(
            common._correct_direction(
                first_prediction,
                second_prediction,
                first,
                second,
            )
        )
        question_identity += int(
            questions[first_key].question.encode("utf-8")
            == questions[second_key].question.encode("utf-8")
        )
        distinct_prefixes += int(
            natural[first_key]["prefix_hash"] != natural[second_key]["prefix_hash"]
        )
        distinct_signatures += int(
            natural[first_key]["scene_control_signature_sha256"]
            != natural[second_key]["scene_control_signature_sha256"]
        )

    changed_keys = {
        (str(row["scene_id"]), str(row["question_id"]))
        for row in references
        if row["route_label"] is True
    }
    question_bytes_unchanged = 0
    opposite_prefix_exact = 0
    opposite_signature_exact = 0
    follows_injected = 0
    follows_by_key: dict[tuple[str, str], bool] = {}
    for key in changed_keys:
        prediction = swaps[key]
        reference = refs[key]
        paired_scene = str(reference["counterfactual_paired_scene_id"])
        question_bytes_unchanged += int(
            prediction["question_sha256"] == _sha256_bytes(questions[key].question.encode("utf-8"))
        )
        opposite_prefix_exact += int(
            prediction["injected_scene_id"] == paired_scene
            and prediction["prefix_hash"] == candidate.natural_prefixes[paired_scene]
            and prediction["prefix_hash"] != candidate.natural_prefixes[key[0]]
        )
        opposite_signature_exact += int(
            prediction["scene_control_signature_sha256"]
            == candidate.natural_signatures[paired_scene]
            and prediction["scene_control_signature_sha256"] != candidate.natural_signatures[key[0]]
        )
        injected_reference = next(
            item
            for item in changed_groups[
                (
                    str(reference["counterfactual_pair_id"]),
                    str(reference["counterfactual_question_key"]),
                )
            ]
            if str(item["scene_id"]) == paired_scene
        )
        follows = common._answer_matches(str(prediction["predicted_answer"]), injected_reference)
        follows_injected += int(follows)
        follows_by_key[key] = follows
    swap_complete = sum(
        all(follows_by_key[(str(side["scene_id"]), str(side["question_id"]))] for side in members)
        for members in changed_groups.values()
    )

    thresholds = candidate.internal_preregistration["thresholds"]
    primary = thresholds["internal_validation"]
    same = thresholds["same_question_different_scene"]
    swap_threshold = thresholds["scene_swap"]
    checks = {
        "natural_population_complete": len(natural) == 384,
        "canonical_exact": canonical_exact >= _minimum(primary["canonical_exact"]),
        "changed_side_exact": changed_side_exact >= _minimum(primary["changed_side_exact"]),
        "changed_paired_unit_complete": complete_units
        >= _minimum(primary["changed_paired_unit_complete"]),
        "changed_paired_unit_correct_direction": correct_direction
        >= _minimum(primary["changed_paired_unit_correct_direction"]),
        "same_question_complete_unit_coverage": len(changed_groups)
        >= _minimum(same["complete_unit_coverage"]),
        "same_question_text_identity": question_identity
        >= _minimum(same["question_text_identity"]),
        "same_question_distinct_scene_prefixes": distinct_prefixes
        >= _minimum(same["distinct_scene_prefix_hashes"]),
        "same_question_distinct_scene_signatures": distinct_signatures
        >= _minimum(same["distinct_scene_signature_hashes"]),
        "scene_swap_side_coverage": len(changed_keys)
        >= _minimum(swap_threshold["swapped_side_coverage"]),
        "scene_swap_question_bytes_unchanged": question_bytes_unchanged
        >= _minimum(swap_threshold["question_bytes_unchanged"]),
        "scene_swap_opposite_prefix_exact": opposite_prefix_exact
        >= _minimum(swap_threshold["opposite_prefix_hash_exact"]),
        "scene_swap_opposite_signature_exact": opposite_signature_exact
        >= _minimum(swap_threshold["opposite_signature_hash_exact"]),
        "scene_swap_answer_follows_injected_scene": follows_injected
        >= _minimum(swap_threshold["answer_follows_injected_scene"]),
        "scene_swap_bidirectional_unit_complete": swap_complete
        >= _minimum(swap_threshold["bidirectional_unit_complete"]),
    }

    def breakdown(values: Mapping[str, Counter[str]]) -> dict[str, dict[str, Any]]:
        return {
            key: {
                "exact": counts["exact"],
                "total": counts["total"],
                "accuracy": counts["exact"] / counts["total"],
            }
            for key, counts in sorted(values.items())
        }

    return {
        "passed": all(checks.values()),
        "checks": checks,
        "metrics": {
            "natural": {
                "canonical_exact": canonical_exact,
                "total": 384,
                "canonical_accuracy": canonical_exact / 384,
                "changed_side_exact": changed_side_exact,
                "changed_side_total": 52,
                "changed_paired_unit_complete": complete_units,
                "changed_paired_unit_total": 26,
                "changed_paired_unit_correct_direction": correct_direction,
                "always_on_control_rows": 384,
                "by_route": breakdown(per_route),
                "by_answer_type": breakdown(per_type),
                "by_question_family": breakdown(per_family),
                "by_change_type": breakdown(per_change),
            },
            "same_question_different_scene": {
                "complete_unit_coverage": len(changed_groups),
                "question_text_identity": question_identity,
                "distinct_scene_prefix_hashes": distinct_prefixes,
                "distinct_scene_signature_hashes": distinct_signatures,
            },
            "scene_swap": {
                "swapped_side_coverage": len(changed_keys),
                "blind_supplied_side_count": len(swaps),
                "question_bytes_unchanged": question_bytes_unchanged,
                "opposite_prefix_hash_exact": opposite_prefix_exact,
                "opposite_signature_hash_exact": opposite_signature_exact,
                "answer_follows_injected_scene": follows_injected,
                "bidirectional_unit_complete": swap_complete,
                "always_on_control_rows": 384,
            },
        },
        "thresholds": thresholds,
    }


def authenticate_candidate_inputs(
    *,
    candidate_predictions: str | Path,
    scene_swap_predictions: str | Path,
    internal_preregistration: str | Path,
    parent_preregistration: str | Path,
    training_preregistration: str | Path,
    baseline_lock: str | Path,
    questions_manifest: str | Path,
    base_checkpoint: str | Path,
    control_checkpoint: str | Path,
    runtime_config: str | Path,
    training_report: str | Path,
) -> CandidateInputs:
    """Authenticate every non-reference byte before the launch claim."""

    internal_path = _safe_existing_file(
        internal_preregistration, "internal-validation preregistration"
    )

    internal = validate_v66_internal_validation_preregistration(internal_path)
    internal_digest = _sha256_file(internal_path)

    parent_path = _safe_existing_file(parent_preregistration, "parent preregistration")
    parent, parent_digest = _load_and_validate_parent_preregistration(parent_path)
    if parent_digest != internal["source_boundary"]["parent_v62_preregistration_sha256"]:
        raise ValueError("V66 internal contract binds a different parent boundary")

    training_prereg_path = _safe_existing_file(training_preregistration, "training preregistration")
    training_prereg, training_prereg_digest = _load_training_preregistration(training_prereg_path)
    if training_prereg_digest != internal["candidate_contract"]["training_preregistration_sha256"]:
        raise ValueError("V66 internal contract binds a different training run")

    questions_path = _safe_existing_file(questions_manifest, "questions manifest")
    questions, questions_digest = common._validate_questions(questions_path, parent)
    if questions_digest != internal["source_boundary"]["questions_manifest_sha256"]:
        raise ValueError("V66 internal contract binds a different question manifest")

    baseline_path = _safe_existing_file(baseline_lock, "baseline lock")
    baseline_raw = baseline_path.read_bytes()
    baseline = validate_baseline_lock(baseline_path)
    baseline_digest = _sha256_bytes(baseline_raw)
    if (
        baseline_digest != internal["source_boundary"]["baseline_lock_sha256"]
        or baseline.get("questions_manifest_sha256") != questions_digest
        or baseline.get("questions_sha256") != questions.questions_sha256
    ):
        raise ValueError("V66 baseline lock does not bind the supplied questions")

    config_path = _safe_existing_file(runtime_config, "runtime configuration")
    config = load_runtime_config(config_path)
    config_file_digest = runtime_config_file_sha256(config_path)
    config_effective_digest = effective_runtime_config_sha256(config)

    base_path = _resolve(base_checkpoint)
    _reject_symlink_components(base_path, "base checkpoint")
    base_digest, base_files = checkpoint_fingerprint(base_path)
    if base_digest != baseline.get("v54_checkpoint_sha256") or base_files != baseline.get(
        "v54_checkpoint_files"
    ):
        raise ValueError("V66 base checkpoint differs from the public V54 lock")

    control_path = _safe_existing_directory(control_checkpoint, "control checkpoint")
    expected_control_path = (
        _PROJECT_ROOT / internal["candidate_contract"]["expected_checkpoint_path"]
    )
    if control_path != expected_control_path.resolve():
        raise ValueError("V66 candidate checkpoint path differs from its preregistration")
    control_digest, control_files, control_metadata = _validate_control_checkpoint(
        control_path,
        base_checkpoint_sha256=base_digest,
        runtime_config_sha256=config_effective_digest,
    )
    condition = f"all_questions;control_checkpoint_sha256={control_digest}"

    natural_path = _safe_existing_file(candidate_predictions, "candidate predictions")
    natural_rows, natural_raw = _load_jsonl(natural_path, "candidate predictions")
    natural_provenance_path = _safe_existing_file(
        natural_path.with_name(f"{natural_path.name}.provenance.json"),
        "candidate prediction provenance",
    )
    natural_provenance, natural_provenance_raw = _load_json_object(
        natural_provenance_path, "candidate prediction provenance"
    )
    natural_provenance_digest = _validate_provenance(
        natural_provenance,
        raw=natural_provenance_raw,
        label="candidate prediction",
        run_kind=NATURAL_RUN_KIND,
        condition=condition,
        questions_sha256=questions_digest,
        base_checkpoint_sha256=base_digest,
        base_checkpoint_files=base_files,
        config_file_sha256=config_file_digest,
        config_effective_sha256=config_effective_digest,
    )
    natural_prefixes, natural_signatures = _validate_natural_predictions(
        natural_rows,
        manifest=questions,
        provenance_sha256=str(natural_provenance["provenance_sha256"]),
        control_checkpoint_sha256=control_digest,
        control_metadata=control_metadata,
        baseline=baseline,
    )

    swap_path = _safe_existing_file(scene_swap_predictions, "scene-swap predictions")
    swap_rows, swap_raw = _load_jsonl(swap_path, "scene-swap predictions")
    swap_provenance_path = _safe_existing_file(
        swap_path.with_name(f"{swap_path.name}.provenance.json"),
        "scene-swap prediction provenance",
    )
    swap_provenance, swap_provenance_raw = _load_json_object(
        swap_provenance_path, "scene-swap prediction provenance"
    )
    swap_condition = f"{SCENE_SWAP_CONDITION_PREFIX};control_checkpoint_sha256={control_digest}"
    swap_provenance_digest = _validate_provenance(
        swap_provenance,
        raw=swap_provenance_raw,
        label="scene-swap prediction",
        run_kind=SCENE_SWAP_RUN_KIND,
        condition=swap_condition,
        questions_sha256=questions_digest,
        base_checkpoint_sha256=base_digest,
        base_checkpoint_files=base_files,
        config_file_sha256=config_file_digest,
        config_effective_sha256=config_effective_digest,
    )
    if swap_provenance.get("scene_map_manifest") != natural_provenance.get(
        "scene_map_manifest"
    ) or swap_provenance.get("split") != natural_provenance.get("split"):
        raise ValueError("V66 natural and swap populations used different maps or splits")
    _validate_swap_predictions(
        swap_rows,
        manifest=questions,
        provenance_sha256=str(swap_provenance["provenance_sha256"]),
        control_checkpoint_sha256=control_digest,
        control_metadata=control_metadata,
        natural_prefixes=natural_prefixes,
        natural_signatures=natural_signatures,
    )

    report_path = _safe_existing_file(training_report, "V66b training report")
    expected_report_path = (
        _PROJECT_ROOT / internal["candidate_contract"]["expected_training_report_path"]
    )
    if report_path != expected_report_path.resolve():
        raise ValueError("V66 training report path differs from its preregistration")
    report_digest, report_artifact = _validate_training_report(
        report_path,
        training_preregistration=training_prereg,
        training_preregistration_sha256=training_prereg_digest,
        baseline_sha256=baseline_digest,
        parent_preregistration=parent,
        base_checkpoint_sha256=base_digest,
        runtime_config_sha256=config_effective_digest,
        control_files=control_files,
        control_metadata=control_metadata,
    )

    return CandidateInputs(
        internal_preregistration=internal,
        internal_preregistration_sha256=internal_digest,
        parent_preregistration=parent,
        parent_preregistration_sha256=parent_digest,
        training_preregistration_sha256=training_prereg_digest,
        questions=questions,
        questions_manifest_sha256=questions_digest,
        baseline=baseline,
        baseline_sha256=baseline_digest,
        natural_rows=tuple(dict(row) for row in natural_rows),
        natural_sha256=_sha256_bytes(natural_raw),
        natural_provenance_sha256=natural_provenance_digest,
        swap_rows=tuple(dict(row) for row in swap_rows),
        swap_sha256=_sha256_bytes(swap_raw),
        swap_provenance_sha256=swap_provenance_digest,
        base_checkpoint_sha256=base_digest,
        base_checkpoint_files=tuple(dict(item) for item in base_files),
        control_checkpoint_sha256=control_digest,
        control_files={name: dict(value) for name, value in control_files.items()},
        control_metadata=dict(control_metadata),
        runtime_config_file_sha256=config_file_digest,
        runtime_config_effective_sha256=config_effective_digest,
        training_report_sha256=report_digest,
        training_report_artifact=report_artifact,
        natural_prefixes=natural_prefixes,
        natural_signatures=natural_signatures,
    )


def _launch_claim(
    candidate: CandidateInputs,
    *,
    scorer_path: Path,
    terminal_output: Path,
) -> dict[str, Any]:
    return {
        "schema": CLAIM_SCHEMA,
        "schema_version": 1,
        "artifact": CLAIM_ARTIFACT,
        "status": "sealed_before_scorer_reference_open",
        "attempt_number": 1,
        "retry_permitted": False,
        "candidate_selection_after_reference_open_permitted": False,
        "terminal_output": str(terminal_output),
        "inputs": {
            "internal_preregistration_sha256": (candidate.internal_preregistration_sha256),
            "parent_preregistration_sha256": candidate.parent_preregistration_sha256,
            "training_preregistration_sha256": (candidate.training_preregistration_sha256),
            "questions_manifest_sha256": candidate.questions_manifest_sha256,
            "baseline_lock_sha256": candidate.baseline_sha256,
            "candidate_predictions_sha256": candidate.natural_sha256,
            "candidate_prediction_provenance_sha256": (candidate.natural_provenance_sha256),
            "scene_swap_predictions_sha256": candidate.swap_sha256,
            "scene_swap_prediction_provenance_sha256": (candidate.swap_provenance_sha256),
            "base_checkpoint_sha256": candidate.base_checkpoint_sha256,
            "control_checkpoint_sha256": candidate.control_checkpoint_sha256,
            "control_runtime_schema_version": candidate.control_metadata["schema_version"],
            "saved_runtime_training_gate_attestation_sha256": (
                candidate.control_metadata["saved_runtime_training_gate_attestation_sha256"]
            ),
            "runtime_config_file_sha256": candidate.runtime_config_file_sha256,
            "runtime_config_effective_sha256": (candidate.runtime_config_effective_sha256),
            "training_report_sha256": candidate.training_report_sha256,
            "expected_scorer_references_sha256": (contract.PINNED_SCORER_REFERENCES_SHA256),
            "expected_scorer_records_sha256": contract.PINNED_SCORER_RECORDS_SHA256,
        },
        "scorer_reference_path": str(scorer_path),
        "scorer_reference_bytes_opened_before_claim": False,
        "fresh_development_57_62_qa_or_oracle_opened": False,
        "deferred_final_25_30_qa_or_oracle_opened": False,
    }


def seal_terminal(
    *,
    candidate_predictions: str | Path,
    scene_swap_predictions: str | Path,
    scorer_references: str | Path,
    internal_preregistration: str | Path,
    parent_preregistration: str | Path,
    training_preregistration: str | Path,
    baseline_lock: str | Path,
    questions_manifest: str | Path,
    base_checkpoint: str | Path,
    control_checkpoint: str | Path,
    runtime_config: str | Path,
    training_report: str | Path,
    launch_claim: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    """Seal the sole attempt, open references last, score, and publish."""

    # Existing terminal state closes the attempt before any candidate input is
    # opened.  A crash after claim creation also permanently consumes it.
    claim_path = _safe_destination(launch_claim, "launch claim")
    output_path = _safe_destination(output, "terminal output")
    if claim_path == output_path:
        raise ValueError("V66 launch claim and terminal output must be distinct")

    # Resolve and constrain the scorer path without reading its bytes.  Its
    # existence/content is deliberately checked only after the claim exists.
    scorer_path = _resolve(scorer_references)
    _reject_symlink_components(scorer_path, "scorer-only references")
    scorer_parts = {part.casefold() for part in scorer_path.parts}
    if not scorer_parts & {"scorer_only", "scorer-only"} or scorer_parts & {
        "runtime",
        "chat",
        "training",
        "questions",
        "predictions",
    }:
        raise ValueError("V66 scorer references must remain in scorer_only")

    candidate = authenticate_candidate_inputs(
        candidate_predictions=candidate_predictions,
        scene_swap_predictions=scene_swap_predictions,
        internal_preregistration=internal_preregistration,
        parent_preregistration=parent_preregistration,
        training_preregistration=training_preregistration,
        baseline_lock=baseline_lock,
        questions_manifest=questions_manifest,
        base_checkpoint=base_checkpoint,
        control_checkpoint=control_checkpoint,
        runtime_config=runtime_config,
        training_report=training_report,
    )
    claim = _launch_claim(candidate, scorer_path=scorer_path, terminal_output=output_path)
    _atomic_create_json(claim_path, claim)
    claim_sha256 = _sha256_file(claim_path)

    # This is the first and only scorer-reference byte open in this process.
    scorer_file = _safe_existing_file(scorer_path, "scorer-only references")
    references, scorer_sha256 = common._validate_scorer_records(
        scorer_file,
        preregistration=candidate.parent_preregistration,
        questions=candidate.questions,
    )
    if scorer_sha256 != contract.PINNED_SCORER_REFERENCES_SHA256:
        raise ValueError("V66 scorer digest differs from its public pin")
    measurement = score_populations(candidate, references)
    passed = measurement["passed"] is True
    terminal = {
        "schema": SCHEMA,
        "schema_version": 1,
        "artifact": ARTIFACT,
        "status": "terminal_pass" if passed else "terminal_fail",
        "passed": passed,
        "attempt_number": 1,
        "retry_permitted": False,
        "candidate_tuning_after_terminal_permitted": False,
        "launch_claim": {
            "sha256": claim_sha256,
            "artifact": CLAIM_ARTIFACT,
            "sealed_before_scorer_reference_open": True,
        },
        "inputs": {
            "internal_preregistration_sha256": (candidate.internal_preregistration_sha256),
            "parent_preregistration_sha256": candidate.parent_preregistration_sha256,
            "training_preregistration_sha256": (candidate.training_preregistration_sha256),
            "questions_manifest_sha256": candidate.questions_manifest_sha256,
            "questions_sha256": candidate.questions.questions_sha256,
            "baseline_lock_sha256": candidate.baseline_sha256,
            "candidate_predictions_sha256": candidate.natural_sha256,
            "candidate_prediction_provenance_sha256": (candidate.natural_provenance_sha256),
            "scene_swap_predictions_sha256": candidate.swap_sha256,
            "scene_swap_prediction_provenance_sha256": (candidate.swap_provenance_sha256),
            "scorer_references_sha256": scorer_sha256,
            "base_checkpoint_sha256": candidate.base_checkpoint_sha256,
            "base_checkpoint_files": list(candidate.base_checkpoint_files),
            "control_checkpoint_sha256": candidate.control_checkpoint_sha256,
            "control_checkpoint_files": candidate.control_files,
            "control_architecture": candidate.control_metadata["architecture"],
            "control_runtime_schema_version": candidate.control_metadata["schema_version"],
            "saved_runtime_training_gate_attestation_sha256": (
                candidate.control_metadata["saved_runtime_training_gate_attestation_sha256"]
            ),
            "runtime_config_file_sha256": candidate.runtime_config_file_sha256,
            "runtime_config_effective_sha256": (candidate.runtime_config_effective_sha256),
            "training_report_sha256": candidate.training_report_sha256,
            "training_report_artifact": candidate.training_report_artifact,
        },
        "checks": measurement["checks"],
        "metrics": measurement["metrics"],
        "thresholds": measurement["thresholds"],
        "training_gate": {
            "pair_disjoint_cv_authenticated": True,
            "saved_runtime_generation_authenticated": True,
            "paired_opposite_scene_dependence_authenticated": True,
            "all_256_scene_latents_required": True,
            "question_dependent_scene_retrieval": False,
            "always_on_continuous_control": True,
        },
        "protected_evaluation_state": {
            "fresh_development_scene_ids": list(_PROTECTED_FRESH_SCENES),
            "deferred_final_scene_ids": list(_PROTECTED_FINAL_SCENES),
            "fresh_development_57_62_qa_or_oracle_opened": False,
            "deferred_final_25_30_qa_or_oracle_opened": False,
            "candidate_training_attested_no_protected_data": True,
            "candidate_prediction_scene_inventory_excludes_protected_data": True,
            "terminal_process_opened_protected_data": False,
        },
        "authorization": {
            "fresh_development_57_62_one_shot_authorized": passed,
            "fresh_development_authorization_scope": (
                list(_PROTECTED_FRESH_SCENES) if passed else []
            ),
            "deferred_final_25_30_authorized": False,
            "internal_validation_retry_authorized": False,
            "new_sealed_gate_required_after_fresh_development": True,
        },
        "access_boundary": {
            "scorer_references_opened_only_after_launch_claim": True,
            "scorer_references_loaded_by_runtime": False,
            "scorer_references_loaded_by_trainer": False,
            "scorer_reference_open_count_in_terminal_process": 1,
            "environmental_answer_text_stored_in_terminal": False,
            "prediction_answer_text_stored_in_terminal": False,
        },
    }
    _atomic_create_json(output_path, terminal)
    return terminal


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-predictions", required=True)
    parser.add_argument("--scene-swap-predictions", required=True)
    parser.add_argument("--scorer-references", required=True)
    parser.add_argument("--internal-preregistration", required=True)
    parser.add_argument("--parent-preregistration", required=True)
    parser.add_argument("--training-preregistration", required=True)
    parser.add_argument("--baseline-lock", required=True)
    parser.add_argument("--questions-manifest", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--control-checkpoint", required=True)
    parser.add_argument("--runtime-config", required=True)
    parser.add_argument("--training-report", required=True)
    parser.add_argument("--launch-claim", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = seal_terminal(
        candidate_predictions=args.candidate_predictions,
        scene_swap_predictions=args.scene_swap_predictions,
        scorer_references=args.scorer_references,
        internal_preregistration=args.internal_preregistration,
        parent_preregistration=args.parent_preregistration,
        training_preregistration=args.training_preregistration,
        baseline_lock=args.baseline_lock,
        questions_manifest=args.questions_manifest,
        base_checkpoint=args.base_checkpoint,
        control_checkpoint=args.control_checkpoint,
        runtime_config=args.runtime_config,
        training_report=args.training_report,
        launch_claim=args.launch_claim,
        output=args.output,
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
