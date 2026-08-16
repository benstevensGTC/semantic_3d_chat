"""Fail-closed scoring gate for strict fixed-prefix atlas predictions.

The prediction process sees only a sanitized question manifest.  This separate
evaluation process authenticates that manifest, the exact runtime inputs, every
prediction row, and the byte-invariant per-scene prefix before it opens answer
references.  A launch claim is created atomically before that final reference
open, so a result cannot be tuned and silently rerun under the same attempt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from semantic_3d_chat.chat.runtime_config import load_runtime_config
from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.evaluation.baseline_io import sha256_file
from semantic_3d_chat.evaluation.metrics import (
    exact_normalized_match,
    list_order_insensitive_match,
    score_predictions,
)
from semantic_3d_chat.evaluation.predict_fixed_prefix_atlas import _provenance
from semantic_3d_chat.evaluation.prediction_artifacts import (
    PROVENANCE_SCHEMA_VERSION,
    PredictionProvenance,
    checkpoint_fingerprint,
    provenance_path_for,
)
from semantic_3d_chat.evaluation.question_manifest import (
    QuestionManifest,
    load_question_manifest,
)
from semantic_3d_chat.evaluation.v67_strict_atlas_preregistration import (
    validate_v67_strict_atlas_preregistration,
)
from semantic_3d_chat.training.fixed_prefix_atlas_checkpoint import (
    two_file_checkpoint_fingerprint,
    validate_fixed_prefix_atlas_metadata,
)

SCHEMA: Final[str] = "semantic_3d_chat.fixed_prefix_atlas_gate.v1"
CLAIM_SCHEMA: Final[str] = "semantic_3d_chat.fixed_prefix_atlas_gate_claim.v1"
RUN_KIND: Final[str] = "strict_fixed_continuous_scene_atlas"
CONDITION: Final[str] = "same_complete_prefix_every_question"
V67_TRAINING_REPORT_ARTIFACT: Final[str] = "v67_pair_objective_behavioral_training_v1"

_SHA256 = re.compile(r"[0-9a-f]{64}")
_SCENE_ID = re.compile(r"scene_[0-9]{6}")
_QUESTION_ID = re.compile(r"q_[0-9]{6}")
_PREDICTION_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "scene_id",
        "question_id",
        "predicted_answer",
        "grounding_xyz",
        "grounding_confidence",
        "prefix_hash",
        "generated_tokens",
        "elapsed_seconds",
        "question_dependent_scene_processing",
        "language_model_environment_conditioning_question_dependent",
        "auxiliary_grounding_question_conditioned",
        "auxiliary_grounding_affects_language_model",
        "provenance_sha256",
    }
)
_PROVENANCE_FIELDS: Final[frozenset[str]] = frozenset(
    {
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
)
_SCORER_CONTAINER_BASE_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema",
        "schema_version",
        "source_qa_sha256",
        "question_count",
        "records_sha256",
        "contains_question_text",
        "runtime_access_permitted",
        "records",
    }
)
_V67_REFERENCE_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "route_label",
        "counterfactual_pair_id",
        "counterfactual_paired_scene_id",
        "counterfactual_question_key",
        "counterfactual_change_type",
        "counterfactual_role",
    }
)


def _rooted(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    rooted = candidate if candidate.is_absolute() else PROJECT_ROOT / candidate
    return Path(os.path.abspath(rooted))


def _reject_symlink_components(path: Path, label: str) -> None:
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current /= component
        if current.is_symlink():
            raise ValueError(f"{label} path contains a symbolic-link component: {current}")


def _regular_file(path: str | Path, label: str) -> Path:
    source = _rooted(path)
    _reject_symlink_components(source, label)
    if not source.is_file():
        raise FileNotFoundError(f"{label} is unavailable: {source}")
    return source


def _new_destination(path: str | Path, label: str) -> Path:
    destination = _rooted(path)
    _reject_symlink_components(destination, label)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Immutable {label} already exists: {destination}")
    return destination


def _canonical_json_bytes(value: object, *, pretty: bool = False) -> bytes:
    options: dict[str, Any] = {
        "sort_keys": True,
        "ensure_ascii": False,
        "allow_nan": False,
    }
    options["indent" if pretty else "separators"] = 2 if pretty else (",", ":")
    return (json.dumps(value, **options) + "\n").encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_jsonl_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    return _sha256_bytes(b"".join(_canonical_json_bytes(dict(row)) for row in rows))


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
            raise FileExistsError(f"Immutable artifact already exists: {path}") from None
    finally:
        temporary.unlink(missing_ok=True)


def _read_json_object(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is invalid JSON") from exc
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a JSON object")
    return value, raw


def _read_jsonl_bytes(raw: bytes, label: str) -> tuple[dict[str, Any], ...]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} is not UTF-8") from exc
    if not text.endswith("\n"):
        raise ValueError(f"{label} must end with a newline")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line:
            raise ValueError(f"{label} contains a blank line at {line_number}")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{label} line {line_number} is invalid JSON") from exc
        if not isinstance(value, dict):
            raise TypeError(f"{label} line {line_number} must be an object")
        rows.append(value)
    if not rows:
        raise ValueError(f"{label} contains no records")
    return tuple(rows)


def _expected_provenance(
    *,
    config_path: Path,
    base_checkpoint: Path,
    atlas_checkpoint: Path,
    questions_path: Path,
    scene_ids: list[str],
    split: str,
) -> PredictionProvenance:
    config = load_runtime_config(config_path)
    return _provenance(
        config=config,
        config_path=config_path,
        base_checkpoint=base_checkpoint,
        atlas_checkpoint=atlas_checkpoint,
        questions_path=questions_path,
        scene_ids=scene_ids,
        split=split,
    )


def _load_predictions(path: Path) -> tuple[dict[str, Any], ...]:
    return _read_jsonl_bytes(path.read_bytes(), "fixed-atlas predictions")


def _finite_number(value: object, label: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise ValueError(f"{label} is outside its valid range")
    return result


def validate_prediction_bundle(
    *,
    questions: QuestionManifest,
    predictions: Sequence[Mapping[str, Any]],
    provenance: Mapping[str, Any],
    expected_provenance: PredictionProvenance,
) -> dict[str, Any]:
    """Validate exact coverage, row contracts, and one fixed prefix per scene."""

    expected_sidecar = expected_provenance.as_dict()
    if set(provenance) != _PROVENANCE_FIELDS or dict(provenance) != expected_sidecar:
        raise ValueError("Fixed-atlas provenance does not bind the exact runtime inputs")
    if provenance.get("schema_version") != PROVENANCE_SCHEMA_VERSION:
        raise ValueError("Fixed-atlas provenance schema is unsupported")
    if provenance.get("run_kind") != RUN_KIND or provenance.get("condition") != CONDITION:
        raise ValueError("Predictions are not from the strict fixed-prefix atlas runtime")

    expected_keys = {(row.scene_id, row.question_id) for row in questions.questions}
    observed: dict[tuple[str, str], Mapping[str, Any]] = {}
    prefixes: dict[str, set[str]] = {}
    for index, row in enumerate(predictions, start=1):
        if set(row) != _PREDICTION_FIELDS:
            raise ValueError(f"Prediction {index} has an invalid field inventory")
        scene_id = row.get("scene_id")
        question_id = row.get("question_id")
        key = (str(scene_id), str(question_id))
        if (
            not isinstance(scene_id, str)
            or _SCENE_ID.fullmatch(scene_id) is None
            or not isinstance(question_id, str)
            or _QUESTION_ID.fullmatch(question_id) is None
        ):
            raise ValueError(f"Prediction {index} has a non-opaque key")
        if key in observed:
            raise ValueError(f"Duplicate prediction key: {key}")
        answer = row.get("predicted_answer")
        prefix = row.get("prefix_hash")
        xyz = row.get("grounding_xyz")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError(f"Prediction {index} has no answer")
        if not isinstance(prefix, str) or _SHA256.fullmatch(prefix) is None:
            raise ValueError(f"Prediction {index} has an invalid prefix hash")
        if (
            not isinstance(xyz, list)
            or len(xyz) != 3
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in xyz
            )
        ):
            raise ValueError(f"Prediction {index} has invalid grounding coordinates")
        _finite_number(row.get("grounding_confidence"), "grounding_confidence", minimum=0.0)
        generated = row.get("generated_tokens")
        if isinstance(generated, bool) or not isinstance(generated, int) or generated < 1:
            raise ValueError(f"Prediction {index} has an invalid generated-token count")
        _finite_number(row.get("elapsed_seconds"), "elapsed_seconds", minimum=0.0)
        if (
            row.get("question_dependent_scene_processing") is not False
            or row.get("language_model_environment_conditioning_question_dependent") is not False
            or row.get("auxiliary_grounding_question_conditioned") is not True
            or row.get("auxiliary_grounding_affects_language_model") is not False
            or row.get("provenance_sha256") != expected_provenance.sha256
        ):
            raise ValueError(f"Prediction {index} violates the fixed-prefix runtime contract")
        observed[key] = row
        prefixes.setdefault(scene_id, set()).add(prefix)

    observed_keys = set(observed)
    if observed_keys != expected_keys:
        missing = sorted(expected_keys - observed_keys)
        extra = sorted(observed_keys - expected_keys)
        raise ValueError(
            f"Prediction coverage is not exact; missing={missing[:3]}, extra={extra[:3]}"
        )
    if any(len(values) != 1 for values in prefixes.values()):
        raise ValueError("A user question changed a scene's fixed environmental prefix")
    expected_scenes = {row.scene_id for row in questions.questions}
    if set(prefixes) != expected_scenes:
        raise ValueError("Fixed-prefix evidence does not cover every question scene")
    prefix_by_scene = {
        scene_id: next(iter(values)) for scene_id, values in sorted(prefixes.items())
    }
    return {
        "question_count": len(expected_keys),
        "scene_count": len(expected_scenes),
        "prefix_by_scene": prefix_by_scene,
        "prefix_invariant_within_every_scene": True,
        "question_dependent_scene_processing": False,
        "question_dependent_retrieval": False,
        "strict_fixed_environment_embedding_input": True,
    }


def _reference_key(row: Mapping[str, Any], index: int) -> tuple[str, str]:
    scene_id = row.get("scene_id")
    question_id = row.get("question_id")
    if (
        not isinstance(scene_id, str)
        or _SCENE_ID.fullmatch(scene_id) is None
        or not isinstance(question_id, str)
        or _QUESTION_ID.fullmatch(question_id) is None
    ):
        raise ValueError(f"Reference {index} has a non-opaque key")
    if not isinstance(row.get("answer"), str) or not str(row["answer"]).strip():
        raise ValueError(f"Reference {index} has no answer")
    if not isinstance(row.get("answer_type"), str) or not str(row["answer_type"]).strip():
        raise ValueError(f"Reference {index} has no answer type")
    return scene_id, question_id


def _load_references(
    path: Path,
    *,
    questions: QuestionManifest,
    expected_sha256: str,
) -> tuple[tuple[dict[str, Any], ...], str, str]:
    """Open references once, after the launch claim, and validate exact coverage."""

    raw = path.read_bytes()
    digest = _sha256_bytes(raw)
    if digest != expected_sha256:
        raise ValueError("Fixed-atlas reference bytes differ from the predeclared digest")
    reference_format = "qa_jsonl"
    try:
        decoded = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError):
        decoded = None
    if isinstance(decoded, dict) and "records" in decoded:
        reference_format = "scorer_container"
        allowed_fields = _SCORER_CONTAINER_BASE_FIELDS | frozenset(
            {"pair_count", "paired_unit_count"}
        )
        if frozenset(decoded) not in {
            _SCORER_CONTAINER_BASE_FIELDS,
            allowed_fields,
        }:
            raise ValueError("Fixed-atlas scorer container field inventory is invalid")
        schema = decoded.get("schema")
        records = decoded.get("records")
        if (
            schema
            not in {
                "semantic_3d_chat.v62.scorer_references.v1",
                "semantic_3d_chat.fixed_prefix_atlas.scorer_references.v1",
            }
            or decoded.get("schema_version") != 1
            or decoded.get("source_qa_sha256") != questions.source_qa_sha256
            or decoded.get("question_count") != questions.question_count
            or decoded.get("contains_question_text") is not False
            or decoded.get("runtime_access_permitted") is not False
            or not isinstance(records, list)
            or len(records) != questions.question_count
            or decoded.get("records_sha256") != _canonical_jsonl_sha256(records)
        ):
            raise ValueError("Fixed-atlas scorer container contract is invalid")
        reference_rows = tuple(records)
    else:
        if digest != questions.source_qa_sha256:
            raise ValueError("QA JSONL references differ from the manifest's source hash")
        reference_rows = _read_jsonl_bytes(raw, "fixed-atlas QA references")

    question_rows = list(questions.questions)
    if len(reference_rows) != len(question_rows):
        raise ValueError("Reference count differs from the question manifest")
    seen: set[tuple[str, str]] = set()
    for index, (reference, question) in enumerate(
        zip(reference_rows, question_rows, strict=True), start=1
    ):
        if not isinstance(reference, dict):
            raise TypeError(f"Reference {index} must be an object")
        key = _reference_key(reference, index)
        expected_key = (question.scene_id, question.question_id)
        if key != expected_key or key in seen:
            raise ValueError("References do not exactly preserve question-manifest order")
        if "question" in reference and reference["question"] != question.question:
            raise ValueError(f"Reference {index} question text differs from the manifest")
        seen.add(key)
    return reference_rows, digest, reference_format


def _project_relative(path: Path, label: str) -> str:
    try:
        return path.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"V67 {label} must be inside the project at its locked path") from exc


def _v67_launch_requested(
    *,
    atlas_checkpoint: Path,
    predictions_path: Path,
    launch_claim_path: Path,
    output_path: Path,
    preregistration_path: str | Path | None,
) -> bool:
    if preregistration_path is not None:
        return True
    candidates = (
        atlas_checkpoint.name,
        predictions_path.name,
        launch_claim_path.name,
        output_path.name,
    )
    return any("v67" in value.casefold() for value in candidates)


def _read_v67_training_source(
    *,
    source_checkpoint: Path,
    training_report: Path,
    training_preregistration_sha256: str,
) -> dict[str, Any]:
    """Authenticate the successful V67 controller before atlas scoring."""

    source_fingerprint, source_files = two_file_checkpoint_fingerprint(source_checkpoint)
    metadata, metadata_raw = _read_json_object(
        source_checkpoint / "runtime_metadata.json",
        "V67 source-controller metadata",
    )
    report, report_raw = _read_json_object(training_report, "V67 source training report")
    checkpoint = report.get("checkpoint")
    saved_reload = report.get("saved_runtime_reload")
    scope = report.get("scope")
    if (
        report.get("schema_version") != 1
        or report.get("artifact") != V67_TRAINING_REPORT_ARTIFACT
        or report.get("terminal_reason") != "all_v67_training_gates_passed_checkpoint_saved"
        or report.get("preregistration_sha256") != training_preregistration_sha256
        or not isinstance(report.get("cv"), Mapping)
        or report["cv"].get("passed") is not True
        or not isinstance(report.get("final_fit"), Mapping)
        or report["final_fit"].get("passed") is not True
        or not isinstance(report.get("paired_opposite_scene_dependence"), Mapping)
        or report["paired_opposite_scene_dependence"].get("passed") is not True
        or not isinstance(saved_reload, Mapping)
        or saved_reload.get("strict_loader_passed") is not True
        or saved_reload.get("passed_before_publication") is not True
        or saved_reload.get("raw_question_token_embeddings_used") is not True
        or not isinstance(scope, Mapping)
        or scope.get("internal_validation_loaded") is not False
        or scope.get("scorer_inputs_used") is not False
        or scope.get("oracle_loaded") is not False
        or scope.get("deferred_final_loaded") is not False
    ):
        raise ValueError("V67 source training report did not pass its locked gates")
    expected_checkpoint = {
        "weights_sha256": source_files["control.safetensors"],
        "runtime_metadata_sha256": source_files["runtime_metadata.json"],
        "source_v66_training_fit_state_sha256": metadata.get(
            "source_v66_training_fit_state_sha256"
        ),
    }
    if (
        checkpoint != expected_checkpoint
        or metadata.get("schema_version") != 7
        or metadata.get("architecture") != "always_on_teacher_basis_full_scene_control_v7"
        or metadata.get("saved_runtime_training_gate_required") is not True
        or metadata.get("saved_runtime_training_gate_passed") is not True
        or metadata.get("saved_runtime_training_gate_attestation_sha256")
        != saved_reload.get("gate_attestation_sha256")
        or metadata.get("question_dependent_scene_retrieval") is not False
        or metadata.get("environmental_text_inputs") != []
    ):
        raise ValueError("V67 source checkpoint differs from its successful training report")
    return {
        "source_controller_checkpoint_sha256": source_fingerprint,
        "source_controller_metadata_sha256": _sha256_bytes(metadata_raw),
        "source_training_report_sha256": _sha256_bytes(report_raw),
        "source_training_preregistration_sha256": training_preregistration_sha256,
        "saved_runtime_training_gate_passed": True,
    }


def _validate_v67_pre_reference_boundary(
    *,
    preregistration_path: Path,
    questions_path: Path,
    questions: QuestionManifest,
    base_checkpoint: Path,
    atlas_checkpoint: Path,
    predictions_path: Path,
    expected_references_sha256: str,
    expected_provenance: PredictionProvenance,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Bind V67 preregistration, code, checkpoints, and sealed prediction inputs."""

    preregistration = validate_v67_strict_atlas_preregistration(preregistration_path)
    source = preregistration["source_boundary"]
    candidate = preregistration["candidate_contract"]
    expected_paths = {
        "atlas checkpoint": str(candidate["expected_atlas_checkpoint"]),
        "atlas predictions": str(candidate["expected_atlas_predictions"]),
    }
    observed_paths = {
        "atlas checkpoint": _project_relative(atlas_checkpoint, "atlas checkpoint"),
        "atlas predictions": _project_relative(predictions_path, "atlas predictions"),
    }
    if observed_paths != expected_paths:
        raise ValueError("V67 atlas candidate paths differ from preregistration")

    static_sources = {
        "atlas_compiler_config_sha256": PROJECT_ROOT
        / "configs/experiments/gemma4_strict_fixed_prefix_atlas_v1.yaml",
        "atlas_compiler_source_sha256": PROJECT_ROOT
        / "src/semantic_3d_chat/training/fixed_prefix_atlas_checkpoint.py",
        "atlas_runtime_source_sha256": PROJECT_ROOT
        / "src/semantic_3d_chat/scene_encoder/fixed_prefix_atlas.py",
        "atlas_prediction_source_sha256": PROJECT_ROOT
        / "src/semantic_3d_chat/evaluation/predict_fixed_prefix_atlas.py",
        "terminal_gate_source_sha256": Path(__file__).resolve(),
    }
    static_hashes: dict[str, str] = {}
    for field, path in static_sources.items():
        regular = _regular_file(path, f"V67 {field}")
        digest = sha256_file(regular)
        if source.get(field) != digest:
            raise ValueError(f"V67 source boundary changed: {field}")
        static_hashes[field] = digest

    question_key_rows = [
        {"scene_id": row.scene_id, "question_id": row.question_id} for row in questions.questions
    ]
    if (
        source.get("questions_manifest_sha256") != sha256_file(questions_path)
        or source.get("questions_sha256") != questions.questions_sha256
        or source.get("question_key_inventory_sha256") != _canonical_jsonl_sha256(question_key_rows)
        or source.get("scorer_references_sha256") != expected_references_sha256
    ):
        raise ValueError("V67 question/reference source boundary differs from preregistration")

    base_sha256, _base_files = checkpoint_fingerprint(base_checkpoint)
    atlas_metadata_path = _regular_file(
        atlas_checkpoint / "runtime_metadata.json", "V67 atlas runtime metadata"
    )
    atlas_weights_path = _regular_file(atlas_checkpoint / "atlas.safetensors", "V67 atlas weights")
    atlas_metadata_raw = atlas_metadata_path.read_bytes()
    try:
        atlas_metadata_value = json.loads(atlas_metadata_raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("V67 atlas runtime metadata is invalid JSON") from exc
    atlas_metadata = validate_fixed_prefix_atlas_metadata(atlas_metadata_value)
    if (
        atlas_metadata.get("weights_sha256") != sha256_file(atlas_weights_path)
        or atlas_metadata.get("architecture") != candidate.get("atlas_architecture")
        or atlas_metadata.get("environment_latents") != candidate.get("global_environment_latents")
        or atlas_metadata.get("probe_count") != candidate.get("probe_count")
        or atlas_metadata.get("fixed_prefix_tokens")
        != candidate.get("compiled_fixed_prefix_tokens")
        or atlas_metadata.get("base_checkpoint_sha256") != base_sha256
        or atlas_metadata.get("base_runtime_config_sha256") != expected_provenance.config_sha256
        or atlas_metadata.get("question_dependent_scene_processing") is not False
        or atlas_metadata.get("question_dependent_retrieval") is not False
        or atlas_metadata.get("environmental_text_inputs") != []
    ):
        raise ValueError("V67 atlas metadata differs from the preregistered candidate")

    source_checkpoint = _regular_file(
        PROJECT_ROOT / str(candidate["expected_source_checkpoint"]) / "runtime_metadata.json",
        "V67 source-controller metadata",
    ).parent
    training_report = _regular_file(
        PROJECT_ROOT / str(candidate["expected_source_training_report"]),
        "V67 source training report",
    )
    training_preregistration = _regular_file(
        PROJECT_ROOT / "reports/gemma4/metrics/v67_pair_objective_preregistration.json",
        "V67 source training preregistration",
    )
    training_preregistration_sha256 = sha256_file(training_preregistration)
    if training_preregistration_sha256 != candidate.get("source_training_preregistration_sha256"):
        raise ValueError("V67 source training preregistration differs from its pin")
    training = _read_v67_training_source(
        source_checkpoint=source_checkpoint,
        training_report=training_report,
        training_preregistration_sha256=training_preregistration_sha256,
    )
    if (
        atlas_metadata.get("source_controller_checkpoint_sha256")
        != training["source_controller_checkpoint_sha256"]
        or atlas_metadata.get("source_controller_metadata_sha256")
        != training["source_controller_metadata_sha256"]
    ):
        raise ValueError("V67 atlas was not compiled from the authenticated source controller")
    boundary = {
        "preregistration_sha256": sha256_file(preregistration_path),
        "static_source_sha256": static_hashes,
        "questions_manifest_sha256": sha256_file(questions_path),
        "questions_sha256": questions.questions_sha256,
        "question_key_inventory_sha256": _canonical_jsonl_sha256(question_key_rows),
        "expected_references_sha256": expected_references_sha256,
        "base_checkpoint_sha256": base_sha256,
        "atlas_runtime_metadata_sha256": _sha256_bytes(atlas_metadata_raw),
        **training,
    }
    return boundary, preregistration


def _v67_answer_matches(prediction: str, reference: Mapping[str, Any]) -> bool:
    answer_items = reference.get("answer_items")
    if answer_items is not None:
        return list_order_insensitive_match(prediction, answer_items)
    return exact_normalized_match(prediction, reference["answer"])


def _v67_reference_answers_match(first: Mapping[str, Any], second: Mapping[str, Any]) -> bool:
    first_items = first.get("answer_items")
    second_items = second.get("answer_items")
    if first_items is not None or second_items is not None:
        first_value = first_items if first_items is not None else first.get("answer")
        second_value = second_items if second_items is not None else second.get("answer")
        return list_order_insensitive_match(first_value, second_value)
    return exact_normalized_match(first.get("answer"), second.get("answer"))


def _v67_correct_direction(
    first_prediction: str,
    second_prediction: str,
    first_reference: Mapping[str, Any],
    second_reference: Mapping[str, Any],
) -> bool:
    own = int(_v67_answer_matches(first_prediction, first_reference)) + int(
        _v67_answer_matches(second_prediction, second_reference)
    )
    crossed = int(_v67_answer_matches(first_prediction, second_reference)) + int(
        _v67_answer_matches(second_prediction, first_reference)
    )
    return own > crossed


def _v67_minimum(value: object, label: str) -> int:
    if not isinstance(value, Mapping):
        raise TypeError(f"V67 threshold {label} must be an object")
    minimum = value.get("minimum")
    if isinstance(minimum, bool) or not isinstance(minimum, int):
        raise TypeError(f"V67 threshold {label} minimum must be an integer")
    return minimum


def _score_v67_terminal_metrics(
    *,
    questions: QuestionManifest,
    references: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    preregistration: Mapping[str, Any],
    normalized_exact_accuracy: float,
) -> dict[str, Any]:
    """Score the four locked V67 natural/counterfactual thresholds."""

    population = preregistration["population"]
    thresholds = preregistration["thresholds"]
    if (
        _canonical_jsonl_sha256(references)
        != preregistration["source_boundary"]["scorer_records_sha256"]
    ):
        raise ValueError("V67 scorer record bytes differ from preregistration")
    question_by_key = {(row.scene_id, row.question_id): row.question for row in questions.questions}
    reference_by_key: dict[tuple[str, str], Mapping[str, Any]] = {}
    prediction_by_key = {
        (str(row["scene_id"]), str(row["question_id"])): row for row in predictions
    }
    groups: defaultdict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    canonical_exact = 0
    changed_side_exact = 0
    for index, reference in enumerate(references, start=1):
        missing = _V67_REFERENCE_FIELDS - set(reference)
        if missing:
            raise ValueError(f"V67 reference {index} lacks paired fields: {sorted(missing)}")
        key = _reference_key(reference, index)
        if key in reference_by_key:
            raise ValueError("V67 references contain a duplicate key")
        if type(reference.get("route_label")) is not bool:
            raise TypeError("V67 reference route_label must be boolean")
        reference_by_key[key] = reference
        prediction = prediction_by_key.get(key)
        if prediction is None:
            raise ValueError("V67 prediction population differs from references")
        exact = _v67_answer_matches(str(prediction["predicted_answer"]), reference)
        canonical_exact += int(exact)
        changed_side_exact += int(reference["route_label"] is True and exact)
        groups[
            (
                str(reference["counterfactual_pair_id"]),
                str(reference["counterfactual_question_key"]),
            )
        ].append(reference)
    if (
        set(reference_by_key) != set(question_by_key)
        or set(prediction_by_key) != set(question_by_key)
        or len(references) != int(population["natural_question_count"])
        or len(groups) != int(population["paired_units"])
    ):
        raise ValueError("V67 scorer population differs from its preregistration")

    changed_groups: list[list[Mapping[str, Any]]] = []
    for members in groups.values():
        if len(members) != 2:
            raise ValueError("V67 paired unit must contain exactly two sides")
        first, second = members
        first_key = str(first["scene_id"]), str(first["question_id"])
        second_key = str(second["scene_id"]), str(second["question_id"])
        changed = first["route_label"] is True
        if (
            second["route_label"] is not changed
            or first["counterfactual_paired_scene_id"] != second["scene_id"]
            or second["counterfactual_paired_scene_id"] != first["scene_id"]
            or {first["counterfactual_role"], second["counterfactual_role"]}
            != {"reference", "counterfactual"}
            or first["counterfactual_change_type"] != second["counterfactual_change_type"]
            or question_by_key[first_key].encode("utf-8")
            != question_by_key[second_key].encode("utf-8")
        ):
            raise ValueError("V67 paired-unit semantics differ from preregistration")
        if changed:
            if _v67_reference_answers_match(first, second):
                raise ValueError("V67 changed paired unit does not encode an answer change")
            changed_groups.append(members)
    if len(changed_groups) != int(population["changed_paired_units"]) or sum(
        len(group) for group in changed_groups
    ) != int(population["changed_sides"]):
        raise ValueError("V67 changed paired-unit inventory differs from preregistration")

    complete_units = 0
    correct_direction = 0
    for first, second in changed_groups:
        first_key = str(first["scene_id"]), str(first["question_id"])
        second_key = str(second["scene_id"]), str(second["question_id"])
        first_prediction = str(prediction_by_key[first_key]["predicted_answer"])
        second_prediction = str(prediction_by_key[second_key]["predicted_answer"])
        complete_units += int(
            _v67_answer_matches(first_prediction, first)
            and _v67_answer_matches(second_prediction, second)
        )
        correct_direction += int(
            _v67_correct_direction(first_prediction, second_prediction, first, second)
        )
    checks = {
        "natural_canonical_exact": canonical_exact
        >= _v67_minimum(thresholds["natural_canonical_exact"], "natural_canonical_exact"),
        "changed_side_exact": changed_side_exact
        >= _v67_minimum(thresholds["changed_side_exact"], "changed_side_exact"),
        "changed_paired_unit_complete": complete_units
        >= _v67_minimum(
            thresholds["changed_paired_unit_complete"],
            "changed_paired_unit_complete",
        ),
        "changed_paired_unit_correct_direction": correct_direction
        >= _v67_minimum(
            thresholds["changed_paired_unit_correct_direction"],
            "changed_paired_unit_correct_direction",
        ),
        "normalized_exact_accuracy": normalized_exact_accuracy
        >= float(thresholds["normalized_exact_accuracy"]["minimum"]),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "metrics": {
            "natural_canonical_exact": canonical_exact,
            "natural_question_total": len(references),
            "changed_side_exact": changed_side_exact,
            "changed_side_total": sum(len(group) for group in changed_groups),
            "changed_paired_unit_complete": complete_units,
            "changed_paired_unit_correct_direction": correct_direction,
            "changed_paired_unit_total": len(changed_groups),
            "normalized_exact_accuracy": normalized_exact_accuracy,
        },
        "thresholds": {
            key: thresholds[key]
            for key in (
                "natural_canonical_exact",
                "changed_side_exact",
                "changed_paired_unit_complete",
                "changed_paired_unit_correct_direction",
                "normalized_exact_accuracy",
            )
        },
    }


def score_fixed_prefix_atlas(
    *,
    config_path: str | Path,
    questions_manifest: str | Path,
    base_checkpoint: str | Path,
    atlas_checkpoint: str | Path,
    predictions_path: str | Path,
    references_path: str | Path,
    expected_references_sha256: str,
    split: str,
    minimum_normalized_exact_accuracy: float,
    launch_claim_path: str | Path,
    output_path: str | Path,
    preregistration_path: str | Path | None = None,
) -> dict[str, Any]:
    """Authenticate a complete run, consume one attempt, then score references."""

    if _SHA256.fullmatch(expected_references_sha256) is None:
        raise ValueError("expected_references_sha256 must be a lowercase SHA-256")
    threshold = _finite_number(
        minimum_normalized_exact_accuracy,
        "minimum_normalized_exact_accuracy",
        minimum=0.0,
    )
    if threshold > 1.0:
        raise ValueError("minimum_normalized_exact_accuracy cannot exceed one")
    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation, or test")

    base = _rooted(base_checkpoint)
    atlas = _rooted(atlas_checkpoint)
    prediction_candidate = _rooted(predictions_path)
    claim_candidate = _rooted(launch_claim_path)
    destination_candidate = _rooted(output_path)
    v67_requested = _v67_launch_requested(
        atlas_checkpoint=atlas,
        predictions_path=prediction_candidate,
        launch_claim_path=claim_candidate,
        output_path=destination_candidate,
        preregistration_path=preregistration_path,
    )
    if v67_requested and preregistration_path is None:
        raise ValueError("V67 launch requires an explicit --preregistration contract")

    claim_path = _new_destination(claim_candidate, "fixed-atlas launch claim")
    destination = _new_destination(destination_candidate, "fixed-atlas terminal report")
    if claim_path == destination:
        raise ValueError("Launch claim and terminal report paths must differ")
    config = _regular_file(config_path, "runtime config")
    question_path = _regular_file(questions_manifest, "question manifest")
    predictions = _regular_file(prediction_candidate, "fixed-atlas predictions")
    provenance_path = _regular_file(
        provenance_path_for(predictions), "fixed-atlas prediction provenance"
    )
    _reject_symlink_components(base, "base checkpoint")
    _reject_symlink_components(atlas, "atlas checkpoint")
    if not base.is_dir() or not atlas.is_dir():
        raise FileNotFoundError("Fixed-atlas base and atlas checkpoints must exist")
    reference_path = _rooted(references_path)
    _reject_symlink_components(reference_path, "scorer references")

    questions = load_question_manifest(question_path)
    scene_ids = sorted(questions.by_scene())
    expected = _expected_provenance(
        config_path=config,
        base_checkpoint=base,
        atlas_checkpoint=atlas,
        questions_path=question_path,
        scene_ids=scene_ids,
        split=split,
    )
    provenance, provenance_raw = _read_json_object(
        provenance_path, "fixed-atlas prediction provenance"
    )
    prediction_rows = _load_predictions(predictions)
    integrity = validate_prediction_bundle(
        questions=questions,
        predictions=prediction_rows,
        provenance=provenance,
        expected_provenance=expected,
    )

    v67_boundary: dict[str, Any] | None = None
    v67_preregistration: dict[str, Any] | None = None
    if v67_requested:
        if preregistration_path is None:  # Narrowed by the fail-closed check above.
            raise AssertionError("V67 preregistration path was not preserved")
        preregistration = _regular_file(
            preregistration_path,
            "V67 strict-atlas preregistration",
        )
        v67_boundary, v67_preregistration = _validate_v67_pre_reference_boundary(
            preregistration_path=preregistration,
            questions_path=question_path,
            questions=questions,
            base_checkpoint=base,
            atlas_checkpoint=atlas,
            predictions_path=predictions,
            expected_references_sha256=expected_references_sha256,
            expected_provenance=expected,
        )
        preregistered_threshold = float(
            v67_preregistration["thresholds"]["normalized_exact_accuracy"]["minimum"]
        )
        if split != "validation" or threshold != preregistered_threshold:
            raise ValueError("V67 split or normalized-exact threshold differs from preregistration")

    # Cache every non-reference digest before consuming the immutable attempt.
    # After the reference bytes are opened, scoring uses memory only and writes
    # the terminal report without reopening an input artifact.
    input_hashes = {
        "config_file_sha256": sha256_file(config),
        "questions_manifest_sha256": sha256_file(question_path),
        "predictions_sha256": sha256_file(predictions),
        "prediction_provenance_sha256": _sha256_bytes(provenance_raw),
    }

    claim = {
        "schema": CLAIM_SCHEMA,
        "schema_version": 1,
        "status": "sealed_before_reference_open",
        "retry_under_same_paths_permitted": False,
        "references_opened_before_claim": False,
        "inputs": {
            "config_file_sha256": input_hashes["config_file_sha256"],
            "questions_manifest_sha256": input_hashes["questions_manifest_sha256"],
            "questions_sha256": questions.questions_sha256,
            "predictions_sha256": input_hashes["predictions_sha256"],
            "prediction_provenance_sha256": input_hashes["prediction_provenance_sha256"],
            "runtime_input_provenance_sha256": expected.sha256,
            "expected_references_sha256": expected_references_sha256,
            "split": split,
            "minimum_normalized_exact_accuracy": threshold,
            "prefix_by_scene": integrity["prefix_by_scene"],
        },
        "terminal_output": str(destination),
    }
    if v67_boundary is not None:
        claim["v67_strict_atlas"] = {
            "preregistration_required": True,
            "source_boundary_validated_before_reference_open": True,
            **v67_boundary,
        }
    claim_sha256 = _sha256_bytes(_canonical_json_bytes(claim, pretty=True))
    _atomic_create_json(claim_path, claim)

    # This is intentionally the first reference-file byte open in this process.
    if not reference_path.is_file():
        raise FileNotFoundError(f"Scorer references are unavailable: {reference_path}")
    references, references_sha256, reference_format = _load_references(
        reference_path,
        questions=questions,
        expected_sha256=expected_references_sha256,
    )
    metrics = score_predictions(references, prediction_rows)
    exact = metrics.get("normalized_exact_accuracy")
    normalized_exact_passed = isinstance(exact, (int, float)) and float(exact) >= threshold
    v67_terminal: dict[str, Any] | None = None
    if v67_preregistration is not None:
        if not isinstance(exact, (int, float)):
            raise ValueError("V67 normalized exact accuracy is unavailable")
        v67_terminal = _score_v67_terminal_metrics(
            questions=questions,
            references=references,
            predictions=prediction_rows,
            preregistration=v67_preregistration,
            normalized_exact_accuracy=float(exact),
        )
    behavioral_passed = normalized_exact_passed and (
        v67_terminal is None or v67_terminal["passed"] is True
    )
    report = {
        "schema": SCHEMA,
        "schema_version": 1,
        "status": "terminal_pass" if behavioral_passed else "terminal_fail",
        "passed": behavioral_passed,
        "integrity_passed": True,
        "behavioral_threshold_passed": behavioral_passed,
        "retry_under_same_paths_permitted": False,
        "references_opened_only_after_launch_claim": True,
        "runtime_loaded_references": False,
        "strict_fixed_prefix": integrity,
        "thresholds": {"minimum_normalized_exact_accuracy": threshold},
        "metrics": metrics,
        "inputs": {
            "launch_claim_sha256": claim_sha256,
            "questions_manifest_sha256": input_hashes["questions_manifest_sha256"],
            "predictions_sha256": input_hashes["predictions_sha256"],
            "prediction_provenance_sha256": input_hashes["prediction_provenance_sha256"],
            "references_sha256": references_sha256,
            "reference_format": reference_format,
            "runtime_input_provenance_sha256": expected.sha256,
            "scene_map_manifest_sha256": expected.scene_map_manifest_sha256,
            "split": split,
        },
    }
    if v67_terminal is not None:
        report["v67_strict_atlas"] = {
            "source_boundary": v67_boundary,
            "terminal_gate": v67_terminal,
        }
    _atomic_create_json(destination, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--questions-manifest", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--atlas-checkpoint", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--references", required=True)
    parser.add_argument("--expected-references-sha256", required=True)
    parser.add_argument("--split", required=True, choices=("train", "validation", "test"))
    parser.add_argument("--minimum-normalized-exact-accuracy", required=True, type=float)
    parser.add_argument(
        "--preregistration",
        help="Required for any V67 atlas terminal launch",
    )
    parser.add_argument("--launch-claim", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = score_fixed_prefix_atlas(
        config_path=args.config,
        questions_manifest=args.questions_manifest,
        base_checkpoint=args.base_checkpoint,
        atlas_checkpoint=args.atlas_checkpoint,
        predictions_path=args.predictions,
        references_path=args.references,
        expected_references_sha256=args.expected_references_sha256,
        split=args.split,
        minimum_normalized_exact_accuracy=args.minimum_normalized_exact_accuracy,
        launch_claim_path=args.launch_claim,
        output_path=args.output,
        preregistration_path=args.preregistration,
    )
    summary = {
        "passed": report["passed"],
        "status": report["status"],
        "reference_count": report["metrics"]["reference_count"],
        "normalized_exact_accuracy": report["metrics"]["normalized_exact_accuracy"],
        "prefix_invariant_within_every_scene": report["strict_fixed_prefix"][
            "prefix_invariant_within_every_scene"
        ],
    }
    if "v67_strict_atlas" in report:
        summary["v67_terminal_gate"] = report["v67_strict_atlas"]["terminal_gate"]
    print(json.dumps(summary, sort_keys=True, allow_nan=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "score_fixed_prefix_atlas",
    "validate_prediction_bundle",
]
