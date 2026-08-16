"""Seal and score the single V62 pair-disjoint internal-validation attempt.

The scorer-only reference file is the last input opened.  Before that open,
this module authenticates the preregistration, questions-only manifest, V54
hash-only baseline, candidate predictions and provenance, bidirectional scene
swap predictions and provenance, runtime configuration, frozen base adapter,
continuous-control checkpoint, and train-only report.  It then creates an
immutable launch claim.  A claim survives a crash and permanently forbids a
second attempt.

Neither an inference process nor a trainer imports this module.  The terminal
report contains aggregate measurements and content hashes, never reference or
predicted answer text.
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

from semantic_3d_chat.chat.runtime_config import (
    effective_runtime_config_sha256,
    load_runtime_config,
    runtime_config_file_sha256,
)
from semantic_3d_chat.evaluation import v62_pair_disjoint_preregistration as boundary
from semantic_3d_chat.evaluation.metrics import (
    exact_normalized_match,
    list_order_insensitive_match,
)
from semantic_3d_chat.evaluation.prediction_artifacts import (
    PROVENANCE_SCHEMA_VERSION,
    checkpoint_fingerprint,
    scene_map_manifest_sha256,
    validate_scene_map_manifest,
)
from semantic_3d_chat.evaluation.question_manifest import (
    QuestionManifest,
    load_question_manifest,
)
from semantic_3d_chat.evaluation.v65_candidate_contract import (
    V65_ARCHITECTURE,
    validate_sealed_v65_checkpoint,
)

ARTIFACT: Final[str] = "v62_pair_disjoint_internal_validation_terminal"
SCHEMA: Final[str] = "semantic_3d_chat.v62.internal_validation_terminal.v1"
CLAIM_ARTIFACT: Final[str] = "v62_pair_disjoint_internal_validation_launch_claim"
CLAIM_SCHEMA: Final[str] = "semantic_3d_chat.v62.internal_validation_claim.v1"
NATURAL_RUN_KIND: Final[str] = "continuous_scene_question_control_v1"
SCENE_SWAP_RUN_KIND: Final[str] = "continuous_scene_question_control_scene_swap_v1"
SCENE_SWAP_CONDITION_PREFIX: Final[str] = "all_questions_bidirectional_scene_swap"
SCENE_SWAP_CHANGED_CONDITION_PREFIX: Final[str] = "all_changed_bidirectional_scene_swap"

_SHA256 = re.compile(r"[0-9a-f]{64}")
_SCENE_ID = re.compile(r"scene_[0-9]{6}")
_QUESTION_ID = re.compile(r"q_[0-9]{6}")
_PAIR_ID = re.compile(r"pair_[0-9]{6}")
_QUESTION_KEY = re.compile(r"cfq_[0-9a-f]{16}")
_PROTECTED_FRESH_SCENES: Final[tuple[str, ...]] = tuple(
    f"scene_{number:06d}" for number in range(57, 63)
)
_PROTECTED_FINAL_SCENES: Final[tuple[str, ...]] = tuple(
    f"scene_{number:06d}" for number in range(25, 31)
)
_EXPECTED_INTERNAL_VALIDATION_THRESHOLDS: Final[dict[str, Any]] = {
    "changed_side_exact": {"minimum": 42, "total": 52},
    "changed_paired_unit_complete": {"minimum": 19, "total": 26},
    "changed_paired_unit_correct_direction": {"minimum": 23, "total": 26},
    "retention_exact_no_control_output_identity": {
        "minimum": 332,
        "total": 332,
        "comparison": "exact_utf8_output_bytes_sha256",
    },
    "minimum_complete_changed_units_by_change_type": {
        "book_support": 2,
        "chair_orientation": 1,
        "color_swap": 2,
        "mirror_lr": 2,
        "object_count": 1,
        "object_relocation": 2,
        "object_removal": 2,
        "picture_support": 2,
    },
}
_EXPECTED_SAME_PREFIX_THRESHOLDS: Final[dict[str, Any]] = {
    "complete_unit_coverage": {"minimum": 26, "total": 26},
    "distinct_scene_prefix_hashes": {"minimum": 26, "total": 26},
    "question_text_identity": {"minimum": 26, "total": 26},
    "changed_side_exact": {"minimum": 42, "total": 52},
    "changed_paired_unit_complete": {"minimum": 19, "total": 26},
    "correct_changed_direction": {"minimum": 23, "total": 26},
}
_EXPECTED_SWAP_THRESHOLDS: Final[dict[str, Any]] = {
    "swapped_side_coverage": {"minimum": 52, "total": 52},
    "question_bytes_unchanged": {"minimum": 52, "total": 52},
    "opposite_prefix_hash_exact": {"minimum": 52, "total": 52},
    "answer_follows_injected_scene": {"minimum": 42, "total": 52},
    "bidirectional_unit_complete": {"minimum": 19, "total": 26},
}
_V65_TRAINING_REPORT_ARTIFACT: Final[str] = "v65_magnitude_gated_canonical_answer_distillation"
_PINNED_V65_TRAINING_BASELINE_LOCK_SHA256: Final[str] = (
    "b1f20e64889116cceb0904ecb3842a6e43fcd6fa3cb0675c32a24f4d278e55e6"
)
_V65_GENERATION_SEMANTICS: Final[str] = (
    "runtime_v6_magnitude_gate_checked_then_exact_control_or_no_token_path"
)
_EXPECTED_V65_BEHAVIOR_THRESHOLDS: Final[dict[str, int]] = {
    "held_supported_side_exact_minimum": 45,
    "held_supported_side_total": 60,
    "held_fully_supported_complete_unit_minimum": 19,
    "held_fully_supported_unit_total": 28,
    "eligible_folds_with_exact_hit_minimum": 7,
    "eligible_fold_total": 8,
    "inventory_fold_total": 12,
    "held_inventory_side_total": 80,
    "held_inventory_unit_total": 40,
    "held_unsupported_side_total": 20,
    "held_retention_exact_no_control_total": 496,
    "final_train_side_exact_minimum": 76,
    "final_train_side_total": 80,
    "final_train_complete_unit_minimum": 36,
    "final_train_complete_unit_total": 40,
    "final_retention_exact_no_control_total": 496,
}


@dataclass(frozen=True)
class CandidateInputs:
    """Authenticated non-reference inputs frozen by the launch claim."""

    preregistration: dict[str, Any]
    preregistration_sha256: str
    questions: QuestionManifest
    questions_manifest_sha256: str
    baseline: dict[str, Any]
    baseline_sha256: str
    natural_rows: tuple[dict[str, Any], ...]
    natural_sha256: str
    natural_provenance: dict[str, Any]
    natural_provenance_sha256: str
    swap_rows: tuple[dict[str, Any], ...]
    swap_sha256: str
    swap_provenance: dict[str, Any]
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
    if pretty:
        options["indent"] = 2
    else:
        options["separators"] = (",", ":")
    return (json.dumps(value, **options) + "\n").encode("utf-8")


def _canonical_identity_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            dict(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _resolve(path: str | Path) -> Path:
    source = Path(path).expanduser()
    return Path(os.path.abspath(source))


def _reject_symlink_components(path: Path, label: str) -> None:
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current /= component
        if current.is_symlink():
            raise ValueError(f"V62 {label} path contains a symbolic link: {current}")


def _safe_existing_file(path: str | Path, label: str) -> Path:
    source = _resolve(path)
    _reject_symlink_components(source, label)
    if not source.is_file():
        raise FileNotFoundError(f"V62 {label} is unavailable: {source}")
    return source


def _safe_destination(path: str | Path, label: str) -> Path:
    destination = _resolve(path)
    _reject_symlink_components(destination, label)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"V62 immutable {label} already exists: {destination}")
    return destination


def _load_json_object(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"V62 {label} is invalid JSON") from exc
    if not isinstance(value, dict):
        raise TypeError(f"V62 {label} must be a JSON object")
    return value, raw


def _load_jsonl(path: Path, label: str) -> tuple[tuple[dict[str, Any], ...], bytes]:
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"V62 {label} is not UTF-8") from exc
    if not text.endswith("\n"):
        raise ValueError(f"V62 {label} must end with a newline")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line:
            raise ValueError(f"V62 {label} has a blank line at {line_number}")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"V62 {label} line {line_number} is invalid JSON") from exc
        if not isinstance(value, dict):
            raise TypeError(f"V62 {label} line {line_number} must be an object")
        rows.append(value)
    return tuple(rows), raw


def _atomic_create_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Create a durable JSON artifact without an overwrite race."""

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
            raise FileExistsError(f"V62 immutable artifact already exists: {path}") from None
    finally:
        temporary.unlink(missing_ok=True)


def _expected_validation_specs() -> tuple[boundary.PairSpec, ...]:
    by_id = {spec.pair_id: spec for spec in boundary.PAIR_INVENTORY}
    return tuple(by_id[pair_id] for pair_id in boundary.INTERNAL_VALIDATION_PAIR_IDS)


def _expected_validation_scenes() -> tuple[str, ...]:
    return tuple(scene_id for spec in _expected_validation_specs() for scene_id in spec.scene_ids)


def _expected_training_scenes() -> tuple[str, ...]:
    by_id = {spec.pair_id: spec for spec in boundary.PAIR_INVENTORY}
    return tuple(
        scene_id for pair_id in boundary.TRAIN_PAIR_IDS for scene_id in by_id[pair_id].scene_ids
    )


def _validate_preregistration(value: Mapping[str, Any], raw: bytes) -> None:
    if _sha256_bytes(raw) != boundary.PINNED_V62_PREREGISTRATION_SHA256:
        raise ValueError("V62 preregistration differs from its immutable public pin")
    boundary._validate_preregistration_payload(value)
    artifacts = value.get("artifacts")
    split = value.get("split")
    thresholds = value.get("thresholds")
    if not isinstance(artifacts, Mapping) or not isinstance(split, Mapping):
        raise TypeError("V62 preregistration artifact/split sections are invalid")
    if not isinstance(thresholds, Mapping):
        raise TypeError("V62 preregistration thresholds are invalid")
    if (
        thresholds.get("internal_validation") != _EXPECTED_INTERNAL_VALIDATION_THRESHOLDS
        or thresholds.get("same_question_different_prefix_control")
        != _EXPECTED_SAME_PREFIX_THRESHOLDS
        or thresholds.get("scene_swap_control") != _EXPECTED_SWAP_THRESHOLDS
    ):
        raise ValueError("V62 terminal thresholds changed after preregistration")
    if (
        split.get("internal_validation_pair_ids") != list(boundary.INTERNAL_VALIDATION_PAIR_IDS)
        or split.get("internal_validation_scene_ids") != list(_expected_validation_scenes())
        or split.get("training_pair_ids") != list(boundary.TRAIN_PAIR_IDS)
        or split.get("training_scene_ids") != list(_expected_training_scenes())
        or split.get("pair_disjoint") is not True
        or split.get("scene_disjoint") is not True
    ):
        raise ValueError("V62 pair-disjoint split differs from the immutable inventory")
    protected = set(_PROTECTED_FRESH_SCENES) | set(_PROTECTED_FINAL_SCENES)
    if protected & set(split["internal_validation_scene_ids"]):
        raise ValueError("V62 internal validation overlaps protected held-out scenes")


def _question_key_rows(manifest: QuestionManifest) -> list[dict[str, str]]:
    return [
        {"scene_id": row.scene_id, "question_id": row.question_id} for row in manifest.questions
    ]


def _canonical_jsonl_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    return _sha256_bytes(b"".join(_canonical_json_bytes(dict(row)) for row in rows))


def _validate_questions(
    path: Path,
    preregistration: Mapping[str, Any],
) -> tuple[QuestionManifest, str]:
    manifest = load_question_manifest(path)
    digest = _sha256_file(path)
    artifact = preregistration["artifacts"]["internal_validation_questions"]
    if (
        digest != artifact["sha256"]
        or digest != boundary.PINNED_V62_QUESTIONS_MANIFEST_SHA256
        or manifest.questions_sha256 != artifact["questions_sha256"]
        or manifest.questions_sha256 != boundary.PINNED_V62_QUESTIONS_SHA256
        or manifest.question_count != 384
        or manifest.scene_count != 16
        or tuple(sorted(manifest.by_scene())) != tuple(sorted(_expected_validation_scenes()))
        or any(len(rows) != 24 for rows in manifest.by_scene().values())
        or _canonical_jsonl_sha256(_question_key_rows(manifest))
        != artifact["question_key_inventory_sha256"]
    ):
        raise ValueError("V62 questions-only manifest differs from the sealed inventory")
    return manifest, digest


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
    if set(value) != required:
        raise ValueError(f"V62 {label} provenance fields differ from schema v2")
    identity = _provenance_identity(value)
    expected_identity = _canonical_identity_sha256(identity)
    maps = validate_scene_map_manifest(value["scene_map_manifest"])
    if (
        value["schema_version"] != PROVENANCE_SCHEMA_VERSION
        or value["provenance_sha256"] != expected_identity
        or value["config_sha256"] != config_effective_sha256
        or value["config_file_sha256"] != config_file_sha256
        or value["checkpoint_sha256"] != base_checkpoint_sha256
        or value["checkpoint_files"] != list(base_checkpoint_files)
        or value["references_sha256"] != questions_sha256
        or value["scene_map_manifest_sha256"] != scene_map_manifest_sha256(maps)
        or set(maps) != set(_expected_validation_scenes())
        or value["run_kind"] != run_kind
        or value["condition"] != condition
        or value["split"] not in {"train", "validation"}
    ):
        raise ValueError(f"V62 {label} provenance does not bind the exact candidate run")
    return _sha256_bytes(raw)


def _valid_prediction_audit(
    value: object,
    *,
    expected_activation_rms_threshold: float | None = None,
) -> bool:
    if not isinstance(value, Mapping):
        return False
    activation = value.get("activation_rms")
    threshold = value.get("activation_rms_threshold")
    maximum = value.get("maximum_control_rms")
    probability = value.get("gate_probability")
    if any(
        type(item) not in {int, float} or not float("-inf") < float(item) < float("inf")
        for item in (activation, threshold, maximum, probability)
    ):
        return False
    if float(threshold) <= 0.0 or float(activation) < 0.0 or float(maximum) < float(activation):
        return False
    control_used = value.get("control_used")
    diagnostic_probability = 1.0 / (
        1.0 + math.exp(-((float(activation) - float(threshold)) / max(float(threshold), 1e-8)))
    )
    return bool(
        value.get("architecture") == V65_ARCHITECTURE
        and value.get("environment_latent_count") == 256
        and value.get("every_scene_token_influenced_output") is True
        and value.get("question_dependent_scene_retrieval") is False
        and value.get("softmax_scene_attention_used") is False
        and value.get("control_values_scene_question_bilinear") is True
        and value.get("gate_scene_question_conditioned") is True
        and value.get("exact_no_control_below_threshold") is True
        and value.get("saved_runtime_training_gate_required") is True
        and type(control_used) is bool
        and value.get("exact_no_control_route") is (not control_used)
        and (
            expected_activation_rms_threshold is None
            or float(threshold) == expected_activation_rms_threshold
        )
        and 0.0 <= float(probability) <= 1.0
        and abs(float(probability) - diagnostic_probability) <= 1e-6
        and control_used is (float(activation) >= float(threshold))
    )


def _validate_natural_predictions(
    rows: Sequence[Mapping[str, Any]],
    *,
    manifest: QuestionManifest,
    provenance_sha256: str,
    control_checkpoint_sha256: str,
    activation_rms_threshold: float,
    baseline: Mapping[str, Any],
) -> dict[str, str]:
    if len(rows) != 384:
        raise ValueError("V62 natural candidate needs exactly 384 predictions")
    expected_keys = [(question.scene_id, question.question_id) for question in manifest.questions]
    actual_keys: list[tuple[str, str]] = []
    prefixes: defaultdict[str, set[str]] = defaultdict(set)
    for index, row in enumerate(rows, start=1):
        required = {
            "scene_id",
            "question_id",
            "predicted_answer",
            "prefix_hash",
            "control_checkpoint_sha256",
            "control_audit",
            "provenance_sha256",
        }
        if not required <= set(row):
            raise ValueError(f"V62 natural prediction {index} misses required fields")
        scene_id = row["scene_id"]
        question_id = row["question_id"]
        answer = row["predicted_answer"]
        prefix = row["prefix_hash"]
        if (
            not isinstance(scene_id, str)
            or _SCENE_ID.fullmatch(scene_id) is None
            or not isinstance(question_id, str)
            or _QUESTION_ID.fullmatch(question_id) is None
            or not isinstance(answer, str)
            or not isinstance(prefix, str)
            or _SHA256.fullmatch(prefix) is None
            or row["control_checkpoint_sha256"] != control_checkpoint_sha256
            or row["provenance_sha256"] != provenance_sha256
            or not _valid_prediction_audit(
                row["control_audit"],
                expected_activation_rms_threshold=activation_rms_threshold,
            )
        ):
            raise ValueError(f"V62 natural prediction {index} has invalid content")
        actual_keys.append((scene_id, question_id))
        prefixes[scene_id].add(prefix)
    if actual_keys != expected_keys or len(set(actual_keys)) != 384:
        raise ValueError("V62 natural prediction ordering/inventory differs from questions")
    baseline_prefixes = baseline["scene_prefix_hashes"]
    if set(prefixes) != set(_expected_validation_scenes()) or any(
        len(values) != 1 for values in prefixes.values()
    ):
        raise ValueError("V62 natural predictions do not prove one prefix per scene")
    result = {scene_id: next(iter(values)) for scene_id, values in prefixes.items()}
    if result != baseline_prefixes:
        raise ValueError("V62 candidate prefixes differ from the pre-training V54 lock")
    return result


def _validate_swap_predictions(
    rows: Sequence[Mapping[str, Any]],
    *,
    manifest: QuestionManifest,
    provenance_sha256: str,
    control_checkpoint_sha256: str,
    activation_rms_threshold: float | None = None,
) -> None:
    # The bundled inference process is intentionally blind to route labels and
    # emits all 384 sides.  The terminal also accepts a separately sealed
    # 52-side artifact; whether that subset is exactly the changed inventory is
    # checked only after the one-shot claim and scorer open.
    if len(rows) not in {52, 384}:
        raise ValueError("V62 scene-swap artifact needs 52 changed or 384 blind sides")
    known = [(row.scene_id, row.question_id) for row in manifest.questions]
    known_set = set(known)
    paired_scene_by_scene = {
        scene_id: paired_scene_id
        for spec in _expected_validation_specs()
        for scene_id, paired_scene_id in (
            (spec.reference_scene_id, spec.counterfactual_scene_id),
            (spec.counterfactual_scene_id, spec.reference_scene_id),
        )
    }
    seen: set[tuple[str, str]] = set()
    actual: list[tuple[str, str]] = []
    for index, row in enumerate(rows, start=1):
        required = {
            "scene_id",
            "question_id",
            "injected_scene_id",
            "predicted_answer",
            "prefix_hash",
            "question_sha256",
            "control_checkpoint_sha256",
            "control_audit",
            "provenance_sha256",
        }
        if not required <= set(row):
            raise ValueError(f"V62 swap prediction {index} misses required fields")
        key = row["scene_id"], row["question_id"]
        injected = row["injected_scene_id"]
        if (
            key not in known_set
            or key in seen
            or not isinstance(injected, str)
            or injected != paired_scene_by_scene.get(str(key[0]))
            or not isinstance(row["predicted_answer"], str)
            or not isinstance(row["prefix_hash"], str)
            or _SHA256.fullmatch(row["prefix_hash"]) is None
            or not isinstance(row["question_sha256"], str)
            or _SHA256.fullmatch(row["question_sha256"]) is None
            or row["control_checkpoint_sha256"] != control_checkpoint_sha256
            or row["provenance_sha256"] != provenance_sha256
            or not _valid_prediction_audit(
                row["control_audit"],
                expected_activation_rms_threshold=activation_rms_threshold,
            )
        ):
            raise ValueError(f"V62 swap prediction {index} has invalid content")
        seen.add(key)
        actual.append((str(key[0]), str(key[1])))
    if len(rows) == 384 and actual != known:
        raise ValueError("V62 scene-swap predictions differ from blind question ordering")


def _validate_control_checkpoint(
    path: Path,
    *,
    base_checkpoint_sha256: str,
    runtime_config_sha256: str,
) -> tuple[str, dict[str, dict[str, Any]], dict[str, Any]]:
    sealed = validate_sealed_v65_checkpoint(
        path,
        base_checkpoint_sha256=base_checkpoint_sha256,
        runtime_config_sha256=runtime_config_sha256,
    )
    return sealed.fingerprint_sha256, sealed.files, sealed.metadata


def _mapping_all_true(value: object, *, exact_keys: set[str] | None = None) -> bool:
    return bool(
        isinstance(value, Mapping)
        and (exact_keys is None or set(value) == exact_keys)
        and value
        and all(item is True for item in value.values())
    )


def _v65_cv_checks(behavior: Mapping[str, Any]) -> dict[str, bool]:
    threshold = _EXPECTED_V65_BEHAVIOR_THRESHOLDS
    fields = {
        "supported_side_exact",
        "supported_side_total",
        "unsupported_side_total",
        "fully_supported_complete_units",
        "fully_supported_unit_total",
        "eligible_folds_with_exact_hit",
        "eligible_fold_count",
        "side_total",
        "unit_total",
        "pair_count",
    }
    if any(type(behavior.get(field)) is not int for field in fields):
        raise ValueError("V65 cross-validation metrics are incomplete")
    return {
        "held_supported_side_exact": behavior["supported_side_exact"]
        >= threshold["held_supported_side_exact_minimum"],
        "held_supported_side_total": behavior["supported_side_total"]
        == threshold["held_supported_side_total"],
        "held_unsupported_side_total": behavior["unsupported_side_total"]
        == threshold["held_unsupported_side_total"],
        "held_fully_supported_complete_units": behavior["fully_supported_complete_units"]
        >= threshold["held_fully_supported_complete_unit_minimum"],
        "held_fully_supported_unit_total": behavior["fully_supported_unit_total"]
        == threshold["held_fully_supported_unit_total"],
        "eligible_folds_with_exact_hit": behavior["eligible_folds_with_exact_hit"]
        >= threshold["eligible_folds_with_exact_hit_minimum"],
        "eligible_fold_total": behavior["eligible_fold_count"] == threshold["eligible_fold_total"],
        "held_inventory_side_total": behavior["side_total"]
        == threshold["held_inventory_side_total"],
        "held_inventory_unit_total": behavior["unit_total"]
        == threshold["held_inventory_unit_total"],
        "held_inventory_fold_total": behavior["pair_count"] == threshold["inventory_fold_total"],
    }


def _v65_final_checks(behavior: Mapping[str, Any]) -> dict[str, bool]:
    threshold = _EXPECTED_V65_BEHAVIOR_THRESHOLDS
    fields = {"side_exact", "side_total", "complete_units", "unit_total"}
    if any(type(behavior.get(field)) is not int for field in fields):
        raise ValueError("V65 final behavior metrics are incomplete")
    return {
        "train_side_exact": behavior["side_exact"] >= threshold["final_train_side_exact_minimum"],
        "train_side_total": behavior["side_total"] == threshold["final_train_side_total"],
        "train_complete_units": behavior["complete_units"]
        >= threshold["final_train_complete_unit_minimum"],
        "train_complete_unit_total": behavior["unit_total"]
        == threshold["final_train_complete_unit_total"],
    }


def _validate_v65_saved_runtime_reload(
    value: object,
    *,
    control_metadata: Mapping[str, Any],
) -> None:
    if not isinstance(value, Mapping):
        raise TypeError("V65 report lacks a saved-runtime training gate")
    behavior = value.get("changed_behavior")
    checks = value.get("checks")
    retention = value.get("retention")
    expected_checks = {
        **_v65_final_checks(behavior if isinstance(behavior, Mapping) else {}),
        "retention_inventory_exact": True,
        "every_retention_row_exact_no_control": True,
        "base_output_identity_by_construction": True,
    }
    gate_attestation = control_metadata["saved_runtime_training_gate_attestation_sha256"]
    fit_state = control_metadata["source_v65_training_fit_state_sha256"]
    production_device = value.get("production_device")
    if (
        value.get("strict_loader_passed") is not True
        or value.get("architecture") != V65_ARCHITECTURE
        or value.get("training_fit_state_sha256") != fit_state
        or value.get("gate_attestation_sha256") != gate_attestation
        or value.get("reloaded_state_exact") is not True
        or value.get("raw_question_token_embeddings_used") is not True
        or not isinstance(production_device, str)
        or not production_device
        or value.get("passed_before_publication") is not True
        or checks != expected_checks
        or not _mapping_all_true(
            retention, exact_keys=set(expected_checks) - set(_v65_final_checks(behavior))
        )
    ):
        raise ValueError("V65 saved-runtime training gate did not pass exactly")
    attestation_payload = {
        "schema_version": 1,
        "artifact": "v65_saved_runtime_training_gate_attestation",
        "training_fit_state_sha256": fit_state,
        "production_device": production_device,
        "raw_question_token_embeddings_used": True,
        "changed_behavior": behavior,
        "retention": retention,
        "checks": checks,
        "answer_or_question_text_stored": False,
    }
    if _sha256_bytes(_canonical_json_bytes(attestation_payload)) != gate_attestation:
        raise ValueError("V65 saved-runtime gate attestation does not authenticate its metrics")


def _validate_training_report(
    path: Path,
    *,
    baseline_sha256: str,
    preregistration: Mapping[str, Any],
    base_checkpoint_sha256: str,
    runtime_config_sha256: str,
    control_files: Mapping[str, Mapping[str, Any]],
    control_metadata: Mapping[str, Any],
) -> tuple[str, str]:
    report, raw = _load_json_object(path, "candidate training report")
    artifact = report.get("artifact")
    authorization = report.get("authorization")
    inputs = report.get("inputs")
    base = report.get("base")
    checkpoint = report.get("checkpoint")
    scope = report.get("scope")
    architecture = report.get("architecture")
    codebook = report.get("codebook")
    cross_validation = report.get("cross_validation")
    final_fit = report.get("final_fit")
    saved_reload = report.get("saved_runtime_reload")
    filtered_sha256 = preregistration["artifacts"]["filtered_training"]["sha256"]
    expected_retention = {
        "retention_inventory_exact",
        "every_retention_row_exact_no_control",
        "base_output_identity_by_construction",
    }

    if (
        report.get("schema_version") != 2
        or artifact != _V65_TRAINING_REPORT_ARTIFACT
        or report.get("offline_checks_passed") is not True
        or report.get("promotion_eligible") is not False
        or report.get("successor_factorized_route_required") is not False
        or report.get("terminal_reason") != "training_behavior_gates_passed_checkpoint_saved"
        or not isinstance(authorization, Mapping)
        or authorization.get("baseline_lock_sha256") != baseline_sha256
        or authorization.get("training_baseline_lock_sha256")
        != _PINNED_V65_TRAINING_BASELINE_LOCK_SHA256
        or authorization.get("filtered_training_qa_sha256") != filtered_sha256
        or authorization.get("baseline_validated_before_training_data") is not True
        or authorization.get("training_v54_hash_inventory_count") != 576
        or not isinstance(inputs, Mapping)
        or inputs.get("training_record_count") != 576
        or inputs.get("training_scene_count") != 24
        or inputs.get("training_pair_count") != 12
        or inputs.get("changed_teacher_side_count") != 80
        or inputs.get("changed_paired_unit_count") != 40
        or not isinstance(base, Mapping)
        or base.get("checkpoint_sha256") != base_checkpoint_sha256
        or base.get("runtime_config_effective_sha256") != runtime_config_sha256
        or not isinstance(scope, Mapping)
        or scope.get("training_answers_used_only_to_build_numeric_codebook_and_score_training")
        is not True
        or scope.get("runtime_answer_strings") is not False
        or scope.get("training_v54_output_hashes_only") is not True
        or scope.get("gemma_backward_used") is not False
        or any(
            scope.get(field) is not False
            for field in (
                "validation_inputs_used",
                "scorer_inputs_used",
                "prediction_inputs_used",
                "oracle_loaded",
                "fresh_development_loaded",
                "deferred_final_loaded",
            )
        )
        or not isinstance(architecture, Mapping)
        or architecture.get("name") != V65_ARCHITECTURE
        or architecture.get("runtime_schema_version") != 6
        or architecture.get("hidden_size") != 1536
        or architecture.get("control_tokens") != 4
        or architecture.get("global_scene_latents") != 256
        or architecture.get("activation_rms_aggregation") != "maximum_over_control_tokens"
        or architecture.get("activation_rms_threshold")
        != control_metadata.get("activation_rms_threshold")
        or architecture.get("exact_no_control_below_threshold") is not True
        or architecture.get("unified_scene_question_value_and_route") is not True
        or architecture.get("question_dependent_scene_retrieval") is not False
        or architecture.get("complete_scene_prefix_retained") is not True
        or not isinstance(codebook, Mapping)
        or codebook.get("final_all_training_only") is not True
        or codebook.get("folds_use_separate_training_only_codebooks") is not True
        or codebook.get("held_fold_label_codebook_visible") is not False
        or codebook.get("held_teacher_used_in_fold_codebook_or_basis") is not False
        or codebook.get("answer_strings_serialized_in_report_or_runtime") is not False
        or not isinstance(checkpoint, Mapping)
        or set(checkpoint)
        != {
            "weights_sha256",
            "runtime_metadata_sha256",
            "source_v65_training_fit_state_sha256",
            "source_v65_value_state_sha256",
        }
        or checkpoint.get("weights_sha256") != control_files["control.safetensors"]["sha256"]
        or checkpoint.get("runtime_metadata_sha256")
        != control_files["runtime_metadata.json"]["sha256"]
        or checkpoint.get("source_v65_training_fit_state_sha256")
        != control_metadata.get("source_v65_training_fit_state_sha256")
        or checkpoint.get("source_v65_value_state_sha256")
        != control_metadata.get("source_v65_value_state_sha256")
    ):
        raise ValueError("V65 candidate training report provenance is invalid")

    if not isinstance(cross_validation, Mapping):
        raise TypeError("V65 training report lacks pair-disjoint cross-validation")
    cv_behavior = cross_validation.get("behavior")
    expected_cv_checks = _v65_cv_checks(cv_behavior if isinstance(cv_behavior, Mapping) else {})
    folds = cross_validation.get("folds")
    if (
        cross_validation.get("protocol") != "deterministic_leave_one_counterfactual_pair_out"
        or cross_validation.get("pair_count") != 12
        or cross_validation.get("each_changed_training_side_generated_exactly_once") is not True
        or cross_validation.get("fold_specific_training_only_codebook_and_basis") is not True
        or cross_validation.get("held_teacher_used_in_fold_codebook_or_basis") is not False
        or cross_validation.get("unsupported_closed_vocabulary_sides_excluded_from_primary_cv_gate")
        is not True
        or cross_validation.get("held_scene_question_examples_used_for_fold_optimization")
        is not False
        or cross_validation.get("thresholds") != _EXPECTED_V65_BEHAVIOR_THRESHOLDS
        or cross_validation.get("checks") != expected_cv_checks
        or not all(expected_cv_checks.values())
        or not _mapping_all_true(cross_validation.get("retention"), exact_keys=expected_retention)
        or cross_validation.get("passed") is not True
        or not isinstance(folds, list)
        or len(folds) != 12
        or [fold.get("held_pair_id") for fold in folds if isinstance(fold, Mapping)]
        != list(boundary.TRAIN_PAIR_IDS)
        or any(
            not isinstance(fold, Mapping)
            or fold.get("training_pair_count") != 11
            or fold.get("held_scene_question_examples_used_for_optimization") is not False
            or fold.get("held_teacher_used_in_codebook_or_basis") is not False
            or fold.get("generation_semantics") != _V65_GENERATION_SEMANTICS
            or not _mapping_all_true(fold.get("retention"), exact_keys=expected_retention)
            for fold in folds
        )
    ):
        raise ValueError("V65 pair-disjoint training gate did not pass exactly")

    if not isinstance(final_fit, Mapping):
        raise TypeError("V65 training report lacks its final fit gate")
    final_behavior = final_fit.get("behavior")
    expected_final_checks = _v65_final_checks(
        final_behavior if isinstance(final_behavior, Mapping) else {}
    )
    reported_final_checks = final_fit.get("checks")
    if (
        not isinstance(reported_final_checks, Mapping)
        or any(
            reported_final_checks.get(key) is not value
            for key, value in expected_final_checks.items()
        )
        or not _mapping_all_true(reported_final_checks)
        or reported_final_checks.get("source_v60_question_norm_exact") is not True
        or reported_final_checks.get("source_v60_question_norm_frozen") is not True
        or any(reported_final_checks.get(key) is not True for key in expected_retention)
        or final_fit.get("passed") is not True
        or not _mapping_all_true(final_fit.get("retention"), exact_keys=expected_retention)
    ):
        raise ValueError("V65 all-training behavior gate did not pass exactly")

    _validate_v65_saved_runtime_reload(
        saved_reload,
        control_metadata=control_metadata,
    )
    return _sha256_bytes(raw), str(artifact)


def authenticate_candidate_inputs(
    *,
    candidate_predictions: str | Path,
    scene_swap_predictions: str | Path,
    preregistration: str | Path,
    baseline_lock: str | Path,
    questions_manifest: str | Path,
    base_checkpoint: str | Path,
    control_checkpoint: str | Path,
    runtime_config: str | Path,
    training_report: str | Path,
) -> CandidateInputs:
    """Authenticate all non-reference inputs without opening scorer answers."""

    prereg_path = _safe_existing_file(preregistration, "preregistration")
    prereg, prereg_raw = _load_json_object(prereg_path, "preregistration")
    _validate_preregistration(prereg, prereg_raw)

    questions_path = _safe_existing_file(questions_manifest, "questions manifest")
    questions, questions_digest = _validate_questions(questions_path, prereg)

    baseline_path = _safe_existing_file(baseline_lock, "baseline lock")
    baseline_raw = baseline_path.read_bytes()
    baseline = boundary.validate_baseline_lock(
        baseline_path,
        preregistration=prereg_path,
    )
    baseline_digest = _sha256_bytes(baseline_raw)
    if (
        baseline["preregistration_sha256"] != _sha256_bytes(prereg_raw)
        or baseline["questions_manifest_sha256"] != questions_digest
        or baseline["questions_sha256"] != questions.questions_sha256
    ):
        raise ValueError("V62 baseline lock does not bind the supplied sealed questions")

    config_path = _safe_existing_file(runtime_config, "runtime configuration")
    config = load_runtime_config(config_path)
    config_file_digest = runtime_config_file_sha256(config_path)
    config_effective_digest = effective_runtime_config_sha256(config)

    base_path = _resolve(base_checkpoint)
    _reject_symlink_components(base_path, "base checkpoint")
    base_digest, base_files = checkpoint_fingerprint(base_path)
    if (
        base_digest != baseline["v54_checkpoint_sha256"]
        or base_files != baseline["v54_checkpoint_files"]
    ):
        raise ValueError("V62 candidate base checkpoint differs from the V54 baseline lock")

    control_path = _resolve(control_checkpoint)
    _reject_symlink_components(control_path, "continuous control checkpoint")
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
        natural_provenance_path,
        "candidate prediction provenance",
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
    if _sha256_bytes(natural_raw) != _sha256_file(natural_path):
        raise AssertionError("V62 natural prediction bytes changed while authenticating")
    natural_prefixes = _validate_natural_predictions(
        natural_rows,
        manifest=questions,
        provenance_sha256=natural_provenance["provenance_sha256"],
        control_checkpoint_sha256=control_digest,
        activation_rms_threshold=float(control_metadata["activation_rms_threshold"]),
        baseline=baseline,
    )

    swap_path = _safe_existing_file(scene_swap_predictions, "scene-swap predictions")
    swap_rows, swap_raw = _load_jsonl(swap_path, "scene-swap predictions")
    swap_condition_prefix = (
        SCENE_SWAP_CONDITION_PREFIX
        if len(swap_rows) == 384
        else SCENE_SWAP_CHANGED_CONDITION_PREFIX
    )
    swap_condition = f"{swap_condition_prefix};control_checkpoint_sha256={control_digest}"
    swap_provenance_path = _safe_existing_file(
        swap_path.with_name(f"{swap_path.name}.provenance.json"),
        "scene-swap prediction provenance",
    )
    swap_provenance, swap_provenance_raw = _load_json_object(
        swap_provenance_path,
        "scene-swap prediction provenance",
    )
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
    if _sha256_bytes(swap_raw) != _sha256_file(swap_path):
        raise AssertionError("V62 scene-swap bytes changed while authenticating")
    if (
        swap_provenance["scene_map_manifest"] != natural_provenance["scene_map_manifest"]
        or swap_provenance["split"] != natural_provenance["split"]
    ):
        raise ValueError("V62 natural and scene-swap runs used different maps or splits")
    _validate_swap_predictions(
        swap_rows,
        manifest=questions,
        provenance_sha256=swap_provenance["provenance_sha256"],
        control_checkpoint_sha256=control_digest,
        activation_rms_threshold=float(control_metadata["activation_rms_threshold"]),
    )

    # Recheck the natural prefixes only against authenticated public data.  No
    # scorer reference has been opened at this point.
    if natural_prefixes != baseline["scene_prefix_hashes"]:
        raise AssertionError("V62 natural prefix authentication was not stable")

    training_path = _safe_existing_file(training_report, "candidate training report")
    training_digest, training_artifact = _validate_training_report(
        training_path,
        baseline_sha256=baseline_digest,
        preregistration=prereg,
        base_checkpoint_sha256=base_digest,
        runtime_config_sha256=config_effective_digest,
        control_files=control_files,
        control_metadata=control_metadata,
    )

    return CandidateInputs(
        preregistration=prereg,
        preregistration_sha256=_sha256_bytes(prereg_raw),
        questions=questions,
        questions_manifest_sha256=questions_digest,
        baseline=baseline,
        baseline_sha256=baseline_digest,
        natural_rows=tuple(dict(row) for row in natural_rows),
        natural_sha256=_sha256_bytes(natural_raw),
        natural_provenance=dict(natural_provenance),
        natural_provenance_sha256=natural_provenance_digest,
        swap_rows=tuple(dict(row) for row in swap_rows),
        swap_sha256=_sha256_bytes(swap_raw),
        swap_provenance=dict(swap_provenance),
        swap_provenance_sha256=swap_provenance_digest,
        base_checkpoint_sha256=base_digest,
        base_checkpoint_files=tuple(dict(item) for item in base_files),
        control_checkpoint_sha256=control_digest,
        control_files={name: dict(value) for name, value in control_files.items()},
        control_metadata=dict(control_metadata),
        runtime_config_file_sha256=config_file_digest,
        runtime_config_effective_sha256=config_effective_digest,
        training_report_sha256=training_digest,
        training_report_artifact=training_artifact,
    )


def _validate_scorer_records(
    scorer_path: Path,
    *,
    preregistration: Mapping[str, Any],
    questions: QuestionManifest,
) -> tuple[tuple[dict[str, Any], ...], str]:
    """Open the scorer-only file exactly once after the launch claim exists."""

    scorer, raw = _load_json_object(scorer_path, "scorer-only references")
    expected_sha256 = preregistration["artifacts"]["scorer_references"]["sha256"]
    if _sha256_bytes(raw) != expected_sha256:
        raise ValueError("V62 scorer-only references differ from the preregistered hash")
    expected_fields = {
        "schema",
        "schema_version",
        "source_qa_sha256",
        "question_count",
        "pair_count",
        "paired_unit_count",
        "records_sha256",
        "contains_question_text",
        "runtime_access_permitted",
        "records",
    }
    records = scorer.get("records")
    if (
        set(scorer) != expected_fields
        or scorer.get("schema") != "semantic_3d_chat.v62.scorer_references.v1"
        or scorer.get("schema_version") != 1
        or scorer.get("question_count") != 384
        or scorer.get("pair_count") != 8
        or scorer.get("paired_unit_count") != 192
        or scorer.get("contains_question_text") is not False
        or scorer.get("runtime_access_permitted") is not False
        or not isinstance(records, list)
        or len(records) != 384
    ):
        raise ValueError("V62 scorer-only reference schema changed")
    scorer_artifact = preregistration["artifacts"]["scorer_references"]
    canonical_records_digest = _canonical_jsonl_sha256(records)
    if (
        canonical_records_digest != scorer["records_sha256"]
        or canonical_records_digest != scorer_artifact["records_sha256"]
    ):
        raise ValueError("V62 scorer-only record bytes changed")

    record_fields = {
        "scene_id",
        "question_id",
        "answer",
        "answer_items",
        "answer_type",
        "route_label",
        "counterfactual_pair_id",
        "counterfactual_paired_scene_id",
        "counterfactual_question_key",
        "counterfactual_change_type",
        "counterfactual_role",
    }
    expected_keys = [(question.scene_id, question.question_id) for question in questions.questions]
    actual_keys: list[tuple[str, str]] = []
    for index, record in enumerate(records, start=1):
        if not isinstance(record, dict) or set(record) != record_fields:
            raise ValueError(f"V62 scorer reference {index} fields changed")
        key = record["scene_id"], record["question_id"]
        answer_items = record["answer_items"]
        if (
            not isinstance(key[0], str)
            or _SCENE_ID.fullmatch(key[0]) is None
            or not isinstance(key[1], str)
            or _QUESTION_ID.fullmatch(key[1]) is None
            or not isinstance(record["answer"], str)
            or not record["answer"].strip()
            or (
                answer_items is not None
                and (
                    not isinstance(answer_items, list)
                    or not all(isinstance(item, str) for item in answer_items)
                )
            )
            or not isinstance(record["answer_type"], str)
            or type(record["route_label"]) is not bool
            or not isinstance(record["counterfactual_pair_id"], str)
            or _PAIR_ID.fullmatch(record["counterfactual_pair_id"]) is None
            or not isinstance(record["counterfactual_paired_scene_id"], str)
            or _SCENE_ID.fullmatch(record["counterfactual_paired_scene_id"]) is None
            or not isinstance(record["counterfactual_question_key"], str)
            or _QUESTION_KEY.fullmatch(record["counterfactual_question_key"]) is None
            or record["counterfactual_role"] not in {"reference", "counterfactual"}
        ):
            raise ValueError(f"V62 scorer reference {index} has invalid content")
        actual_keys.append(key)
    if actual_keys != expected_keys or len(set(actual_keys)) != 384:
        raise ValueError("V62 scorer inventory differs from the questions manifest")

    grouped: defaultdict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    pair_counts: Counter[str] = Counter()
    for record in records:
        pair_id = str(record["counterfactual_pair_id"])
        grouped[(pair_id, str(record["counterfactual_question_key"]))].append(record)
        pair_counts[pair_id] += 1
    specs = {spec.pair_id: spec for spec in _expected_validation_specs()}
    if set(pair_counts) != set(specs) or set(pair_counts.values()) != {48} or len(grouped) != 192:
        raise ValueError("V62 scorer pair/unit inventory changed")
    changed_units = 0
    for (pair_id, _question_key), members in grouped.items():
        spec = specs[pair_id]
        if len(members) != 2 or {str(item["scene_id"]) for item in members} != set(spec.scene_ids):
            raise ValueError("V62 scorer contains a malformed paired unit")
        by_scene = {str(item["scene_id"]): item for item in members}
        first = by_scene[spec.reference_scene_id]
        second = by_scene[spec.counterfactual_scene_id]
        if (
            first["counterfactual_paired_scene_id"] != spec.counterfactual_scene_id
            or second["counterfactual_paired_scene_id"] != spec.reference_scene_id
            or first["route_label"] is not second["route_label"]
            or first["counterfactual_change_type"] != spec.change_type
            or second["counterfactual_change_type"] != spec.change_type
            or first["counterfactual_role"] != "reference"
            or second["counterfactual_role"] != "counterfactual"
        ):
            raise ValueError("V62 scorer paired-unit semantics changed")
        changed_units += int(first["route_label"])
    if changed_units != 26:
        raise ValueError("V62 scorer changed-unit inventory differs from the pin")
    return tuple(dict(record) for record in records), _sha256_bytes(raw)


def _answer_matches(prediction: str, reference: Mapping[str, Any]) -> bool:
    items = reference.get("answer_items")
    if items is not None:
        return list_order_insensitive_match(prediction, items)
    return exact_normalized_match(prediction, reference["answer"])


def _correct_direction(
    first_prediction: str,
    second_prediction: str,
    first_reference: Mapping[str, Any],
    second_reference: Mapping[str, Any],
) -> bool:
    """Require more own-side than cross-side matches for a changed pair."""

    own = int(_answer_matches(first_prediction, first_reference)) + int(
        _answer_matches(second_prediction, second_reference)
    )
    crossed = int(_answer_matches(first_prediction, second_reference)) + int(
        _answer_matches(second_prediction, first_reference)
    )
    return own > crossed


def _minimum(threshold: Mapping[str, Any]) -> int:
    value = threshold.get("minimum")
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("V62 preregistered minimum must be an integer")
    return value


def score_populations(
    candidate: CandidateInputs,
    references: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Score sealed natural and scene-swap populations without returning answers."""

    questions_by_key = {
        (row.scene_id, row.question_id): row for row in candidate.questions.questions
    }
    references_by_key = {(str(row["scene_id"]), str(row["question_id"])): row for row in references}
    natural_by_key = {
        (str(row["scene_id"]), str(row["question_id"])): row for row in candidate.natural_rows
    }
    swap_by_key = {
        (str(row["scene_id"]), str(row["question_id"])): row for row in candidate.swap_rows
    }
    if set(references_by_key) != set(questions_by_key) or set(natural_by_key) != set(
        questions_by_key
    ):
        raise ValueError("V62 scorer/natural populations do not cover identical keys")

    baseline_hashes = {
        (str(row["scene_id"]), str(row["question_id"])): str(row["raw_output_sha256"])
        for row in candidate.baseline["required_output_hashes"]
    }
    prefixes = candidate.baseline["scene_prefix_hashes"]
    exact_total = 0
    changed_side_exact = 0
    retention_side_exact = 0
    retention_identity = 0
    retention_exact_no_control = 0
    changed_route_used = 0
    per_route: dict[str, Counter[str]] = defaultdict(Counter)
    per_answer_type: dict[str, Counter[str]] = defaultdict(Counter)
    per_question_family: dict[str, Counter[str]] = defaultdict(Counter)
    per_change_type: dict[str, Counter[str]] = defaultdict(Counter)
    for key, reference in references_by_key.items():
        prediction = natural_by_key[key]
        answer = str(prediction["predicted_answer"])
        exact = _answer_matches(answer, reference)
        route = "changed" if reference["route_label"] else "retention"
        family = boundary._question_family(questions_by_key[key].question)
        exact_total += int(exact)
        changed_side_exact += int(bool(reference["route_label"]) and exact)
        retention_side_exact += int(not reference["route_label"] and exact)
        per_route[route]["total"] += 1
        per_route[route]["exact"] += int(exact)
        per_answer_type[str(reference["answer_type"])]["total"] += 1
        per_answer_type[str(reference["answer_type"])]["exact"] += int(exact)
        per_question_family[family]["total"] += 1
        per_question_family[family]["exact"] += int(exact)
        change_type = str(reference["counterfactual_change_type"])
        per_change_type[change_type]["total"] += 1
        per_change_type[change_type]["exact"] += int(exact)
        audit = prediction["control_audit"]
        if reference["route_label"]:
            changed_route_used += int(audit["control_used"] is True)
        else:
            raw_identity = _sha256_bytes(answer.encode("utf-8")) == baseline_hashes[key]
            exact_route = audit["control_used"] is False and audit["exact_no_control_route"] is True
            retention_identity += int(raw_identity)
            retention_exact_no_control += int(raw_identity and exact_route)

    groups: defaultdict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for reference in references:
        groups[
            (
                str(reference["counterfactual_pair_id"]),
                str(reference["counterfactual_question_key"]),
            )
        ].append(reference)
    changed_groups = {
        key: members for key, members in groups.items() if members[0]["route_label"] is True
    }
    if len(changed_groups) != 26:
        raise ValueError("V62 score expected exactly 26 changed paired units")
    complete_units = 0
    correct_direction = 0
    same_question_coverage = 0
    same_question_identity = 0
    distinct_prefixes = 0
    complete_by_type: Counter[str] = Counter()
    for members in changed_groups.values():
        first, second = members
        first_key = str(first["scene_id"]), str(first["question_id"])
        second_key = str(second["scene_id"]), str(second["question_id"])
        first_prediction = str(natural_by_key[first_key]["predicted_answer"])
        second_prediction = str(natural_by_key[second_key]["predicted_answer"])
        first_exact = _answer_matches(first_prediction, first)
        second_exact = _answer_matches(second_prediction, second)
        complete = first_exact and second_exact
        complete_units += int(complete)
        if complete:
            complete_by_type[str(first["counterfactual_change_type"])] += 1
        correct_direction += int(
            _correct_direction(first_prediction, second_prediction, first, second)
        )
        same_question_coverage += 1
        first_question = questions_by_key[first_key].question.encode("utf-8")
        second_question = questions_by_key[second_key].question.encode("utf-8")
        same_question_identity += int(first_question == second_question)
        distinct_prefixes += int(
            natural_by_key[first_key]["prefix_hash"] != natural_by_key[second_key]["prefix_hash"]
        )

    changed_keys = {
        (str(row["scene_id"]), str(row["question_id"]))
        for row in references
        if row["route_label"] is True
    }
    if not changed_keys <= set(swap_by_key) or len(swap_by_key) not in {52, 384}:
        raise ValueError("V62 scene-swap inventory does not cover all 52 changed sides")
    if len(swap_by_key) == 52 and set(swap_by_key) != changed_keys:
        raise ValueError("V62 52-side scene-swap artifact is not the exact changed inventory")
    if len(swap_by_key) == 384 and set(swap_by_key) != set(questions_by_key):
        raise ValueError("V62 blind scene-swap artifact is not the complete question inventory")
    changed_swap_by_key = {key: swap_by_key[key] for key in changed_keys}
    question_bytes_unchanged = 0
    opposite_prefix_exact = 0
    swap_answer_exact = 0
    swap_route_used = 0
    swap_exact_by_key: dict[tuple[str, str], bool] = {}
    for key, prediction in changed_swap_by_key.items():
        reference = references_by_key[key]
        paired_scene = str(reference["counterfactual_paired_scene_id"])
        injected = str(prediction["injected_scene_id"])
        question_digest = _sha256_bytes(questions_by_key[key].question.encode("utf-8"))
        question_bytes_unchanged += int(prediction["question_sha256"] == question_digest)
        opposite_prefix_exact += int(
            injected == paired_scene
            and prediction["prefix_hash"] == prefixes[paired_scene]
            and prediction["prefix_hash"] != prefixes[key[0]]
        )
        injected_reference = next(
            member
            for member in changed_groups[
                (
                    str(reference["counterfactual_pair_id"]),
                    str(reference["counterfactual_question_key"]),
                )
            ]
            if member["scene_id"] == paired_scene
        )
        follows = injected == paired_scene and _answer_matches(
            str(prediction["predicted_answer"]), injected_reference
        )
        swap_answer_exact += int(follows)
        swap_exact_by_key[key] = follows
        swap_route_used += int(prediction["control_audit"]["control_used"] is True)
    swap_complete_units = sum(
        all(
            swap_exact_by_key[(str(member["scene_id"]), str(member["question_id"]))]
            for member in members
        )
        for members in changed_groups.values()
    )

    thresholds = candidate.preregistration["thresholds"]
    primary_thresholds = thresholds["internal_validation"]
    same_thresholds = thresholds["same_question_different_prefix_control"]
    swap_thresholds = thresholds["scene_swap_control"]
    type_minima = primary_thresholds["minimum_complete_changed_units_by_change_type"]
    per_type_checks = {
        change_type: complete_by_type[change_type] >= minimum
        for change_type, minimum in sorted(type_minima.items())
    }
    checks = {
        "natural_population_complete": len(natural_by_key) == 384,
        "changed_side_exact": changed_side_exact
        >= _minimum(primary_thresholds["changed_side_exact"]),
        "changed_paired_unit_complete": complete_units
        >= _minimum(primary_thresholds["changed_paired_unit_complete"]),
        "changed_paired_unit_correct_direction": correct_direction
        >= _minimum(primary_thresholds["changed_paired_unit_correct_direction"]),
        "retention_exact_raw_identity": retention_identity
        >= _minimum(primary_thresholds["retention_exact_no_control_output_identity"]),
        "retention_exact_no_control_route": retention_exact_no_control == 332,
        "minimum_complete_changed_units_by_change_type": all(per_type_checks.values()),
        "same_question_complete_unit_coverage": same_question_coverage
        >= _minimum(same_thresholds["complete_unit_coverage"]),
        "same_question_distinct_prefixes": distinct_prefixes
        >= _minimum(same_thresholds["distinct_scene_prefix_hashes"]),
        "same_question_text_identity": same_question_identity
        >= _minimum(same_thresholds["question_text_identity"]),
        "same_question_changed_side_exact": changed_side_exact
        >= _minimum(same_thresholds["changed_side_exact"]),
        "same_question_changed_unit_complete": complete_units
        >= _minimum(same_thresholds["changed_paired_unit_complete"]),
        "same_question_correct_direction": correct_direction
        >= _minimum(same_thresholds["correct_changed_direction"]),
        "scene_swap_side_coverage": len(changed_swap_by_key)
        >= _minimum(swap_thresholds["swapped_side_coverage"]),
        "scene_swap_question_bytes_unchanged": question_bytes_unchanged
        >= _minimum(swap_thresholds["question_bytes_unchanged"]),
        "scene_swap_opposite_prefix_exact": opposite_prefix_exact
        >= _minimum(swap_thresholds["opposite_prefix_hash_exact"]),
        "scene_swap_answer_follows_injected_scene": swap_answer_exact
        >= _minimum(swap_thresholds["answer_follows_injected_scene"]),
        "scene_swap_bidirectional_unit_complete": swap_complete_units
        >= _minimum(swap_thresholds["bidirectional_unit_complete"]),
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
                "exact": exact_total,
                "total": 384,
                "exact_accuracy": exact_total / 384,
                "changed_side_exact": changed_side_exact,
                "changed_side_total": 52,
                "retention_side_exact": retention_side_exact,
                "retention_side_total": 332,
                "retention_exact_raw_identity": retention_identity,
                "retention_exact_no_control_route": retention_exact_no_control,
                "changed_route_used": changed_route_used,
                "changed_paired_unit_complete": complete_units,
                "changed_paired_unit_total": 26,
                "changed_paired_unit_correct_direction": correct_direction,
                "complete_changed_units_by_change_type": dict(sorted(complete_by_type.items())),
                "minimum_checks_by_change_type": per_type_checks,
                "by_route": breakdown(per_route),
                "by_answer_type": breakdown(per_answer_type),
                "by_question_family": breakdown(per_question_family),
                "by_change_type": breakdown(per_change_type),
            },
            "same_question_different_prefix": {
                "complete_unit_coverage": same_question_coverage,
                "question_text_identity": same_question_identity,
                "distinct_scene_prefix_hashes": distinct_prefixes,
                "changed_side_exact": changed_side_exact,
                "changed_paired_unit_complete": complete_units,
                "correct_changed_direction": correct_direction,
            },
            "scene_swap": {
                "swapped_side_coverage": len(changed_swap_by_key),
                "blind_supplied_side_count": len(swap_by_key),
                "question_bytes_unchanged": question_bytes_unchanged,
                "opposite_prefix_hash_exact": opposite_prefix_exact,
                "answer_follows_injected_scene": swap_answer_exact,
                "bidirectional_unit_complete": swap_complete_units,
                "control_route_used": swap_route_used,
            },
        },
        "thresholds": thresholds,
    }


def _launch_claim(
    candidate: CandidateInputs,
    *,
    scorer_path: Path,
    terminal_output: Path,
) -> dict[str, Any]:
    scorer_artifact = candidate.preregistration["artifacts"]["scorer_references"]
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
            "preregistration_sha256": candidate.preregistration_sha256,
            "questions_manifest_sha256": candidate.questions_manifest_sha256,
            "baseline_lock_sha256": candidate.baseline_sha256,
            "candidate_predictions_sha256": candidate.natural_sha256,
            "candidate_prediction_provenance_sha256": (candidate.natural_provenance_sha256),
            "scene_swap_predictions_sha256": candidate.swap_sha256,
            "scene_swap_prediction_provenance_sha256": (candidate.swap_provenance_sha256),
            "base_checkpoint_sha256": candidate.base_checkpoint_sha256,
            "control_checkpoint_sha256": candidate.control_checkpoint_sha256,
            "control_runtime_schema_version": candidate.control_metadata["schema_version"],
            "saved_runtime_training_gate_attestation_sha256": candidate.control_metadata[
                "saved_runtime_training_gate_attestation_sha256"
            ],
            "runtime_config_file_sha256": candidate.runtime_config_file_sha256,
            "runtime_config_effective_sha256": (candidate.runtime_config_effective_sha256),
            "training_report_sha256": candidate.training_report_sha256,
            "expected_scorer_references_sha256": scorer_artifact["sha256"],
            "expected_scorer_records_sha256": scorer_artifact["records_sha256"],
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
    preregistration: str | Path,
    baseline_lock: str | Path,
    questions_manifest: str | Path,
    base_checkpoint: str | Path,
    control_checkpoint: str | Path,
    runtime_config: str | Path,
    training_report: str | Path,
    launch_claim: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    """Create the one-shot claim, open references once, score, and seal."""

    # Existing terminal state closes the attempt before any evaluation input is
    # opened.  This ordering is part of the no-retry contract.
    claim_path = _safe_destination(launch_claim, "launch claim")
    output_path = _safe_destination(output, "terminal output")
    if claim_path == output_path:
        raise ValueError("V62 launch claim and terminal output must be distinct")

    scorer_path = _safe_existing_file(scorer_references, "scorer-only references")
    scorer_parts = {part.casefold() for part in scorer_path.parts}
    if not scorer_parts & {"scorer_only", "scorer-only"} or scorer_parts & {
        "runtime",
        "chat",
        "training",
        "questions",
        "predictions",
    }:
        raise ValueError("V62 scorer references must remain in a scorer_only directory")

    candidate = authenticate_candidate_inputs(
        candidate_predictions=candidate_predictions,
        scene_swap_predictions=scene_swap_predictions,
        preregistration=preregistration,
        baseline_lock=baseline_lock,
        questions_manifest=questions_manifest,
        base_checkpoint=base_checkpoint,
        control_checkpoint=control_checkpoint,
        runtime_config=runtime_config,
        training_report=training_report,
    )
    claim = _launch_claim(
        candidate,
        scorer_path=scorer_path,
        terminal_output=output_path,
    )
    _atomic_create_json(claim_path, claim)
    claim_sha256 = _sha256_file(claim_path)

    # This is intentionally the first and only scorer-reference byte open.
    references, scorer_sha256 = _validate_scorer_records(
        scorer_path,
        preregistration=candidate.preregistration,
        questions=candidate.questions,
    )
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
            "preregistration_sha256": candidate.preregistration_sha256,
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
            "saved_runtime_training_gate_attestation_sha256": candidate.control_metadata[
                "saved_runtime_training_gate_attestation_sha256"
            ],
            "runtime_config_file_sha256": candidate.runtime_config_file_sha256,
            "runtime_config_effective_sha256": (candidate.runtime_config_effective_sha256),
            "training_report_sha256": candidate.training_report_sha256,
            "training_report_artifact": candidate.training_report_artifact,
        },
        "checks": measurement["checks"],
        "metrics": measurement["metrics"],
        "thresholds": measurement["thresholds"],
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
            "robot_semantic_navigation_authorized": False,
            "new_sealed_gate_required_after_fresh_development": True,
            "internal_validation_retry_authorized": False,
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
    parser.add_argument("--preregistration", required=True)
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
        preregistration=args.preregistration,
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


__all__ = [
    "ARTIFACT",
    "CLAIM_ARTIFACT",
    "CandidateInputs",
    "authenticate_candidate_inputs",
    "main",
    "score_populations",
    "seal_terminal",
]
