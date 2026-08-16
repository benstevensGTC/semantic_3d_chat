"""Isolated scorer for the V75 216-question official validation run.

Only this process accepts the answer-bearing reference file.  It authenticates
the sanitized manifest, prediction provenance, sealed V75 controller, complete
question coverage, and one invariant prefix per scene before calculating
structured metrics.  The emitted report contains no questions or answers.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from semantic_3d_chat.evaluation.metrics import score_predictions
from semantic_3d_chat.evaluation.prediction_artifacts import (
    PROVENANCE_SCHEMA_VERSION,
    PredictionProvenance,
    provenance_path_for,
    scene_map_manifest_sha256,
    validate_scene_map_manifest,
)
from semantic_3d_chat.evaluation.v56_fresh_development_score import (
    EXPECTED_GROUNDING_TARGET_COUNT,
    EXPECTED_TYPE_COUNTS,
    _changed_metrics,
    _prediction_index,
    canonical_type_specific_match,
    threshold_contract,
)
from semantic_3d_chat.evaluation.v75_official_validation_contract import (
    ARTIFACT,
    DEFAULT_CONTROL_CHECKPOINT,
    DEFAULT_PREDICTIONS,
    DEFAULT_QUESTIONS_MANIFEST,
    DEFAULT_REFERENCES,
    DEFAULT_SCORE,
    EXPECTED_BASE_CHECKPOINT_SHA256,
    EXPECTED_QUESTION_COUNT,
    EXPECTED_REFERENCE_SHA256,
    EXPECTED_RUNTIME_CONFIG_SHA256,
    EXPECTED_SCENE_IDS,
    V75ControlIdentity,
    authenticate_v75_control_checkpoint,
    reject_symlink_components,
    resolve_path,
    safe_prediction_input,
    sha256_file,
    validate_official_question_manifest,
    validate_v75_control_audit,
)

SCORE_ARTIFACT: Final[str] = "v75_official_validation_score_v1"
RUN_KIND: Final[str] = "continuous_scene_question_control_v1"
_HEX = frozenset("0123456789abcdef")
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
        "control_checkpoint_sha256",
        "control_audit",
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


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value).issubset(_HEX)
    )


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"V75 official {field} must be a mapping")
    return value


def _load_jsonl(path: Path, field: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError(f"V75 {field} are not UTF-8") from error
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid V75 {field} JSON at line {line_number}") from error
        if not isinstance(value, dict):
            raise TypeError(f"V75 {field} line {line_number} must be an object")
        records.append(value)
    return records


def _safe_references(path: str | Path) -> Path:
    """Authenticate bytes before parsing the scorer-only answer file."""

    source = reject_symlink_components(path, "scorer references")
    if source.is_symlink() or not source.is_file():
        raise FileNotFoundError(f"V75 scorer references are unavailable: {source}")
    if sha256_file(source) != EXPECTED_REFERENCE_SHA256:
        raise ValueError("V75 scorer references differ from the frozen official split")
    return source


def _load_provenance(
    predictions: Path,
    *,
    manifest_path: Path,
    control: V75ControlIdentity,
) -> PredictionProvenance:
    source = safe_prediction_input(
        provenance_path_for(predictions), "prediction provenance", kind="file"
    )
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("V75 prediction provenance is invalid JSON") from error
    value = dict(_mapping(raw, "prediction provenance"))
    if set(value) != _PROVENANCE_FIELDS:
        raise ValueError("V75 prediction provenance fields changed")
    checkpoint_files = value.get("checkpoint_files")
    scene_maps = validate_scene_map_manifest(value.get("scene_map_manifest"))
    if not isinstance(checkpoint_files, list) or not all(
        isinstance(item, dict) for item in checkpoint_files
    ):
        raise TypeError("V75 provenance checkpoint_files must be a list of mappings")
    for field in (
        "config_sha256",
        "config_file_sha256",
        "checkpoint_sha256",
        "references_sha256",
        "scene_map_manifest_sha256",
        "provenance_sha256",
    ):
        if not _is_sha256(value.get(field)):
            raise ValueError(f"V75 provenance has an invalid {field}")
    reconstructed = PredictionProvenance(
        config_path=str(value.get("config_path")),
        config_sha256=str(value["config_sha256"]),
        config_file_sha256=str(value["config_file_sha256"]),
        checkpoint_path=str(value.get("checkpoint_path")),
        checkpoint_sha256=str(value["checkpoint_sha256"]),
        checkpoint_files=tuple(dict(item) for item in checkpoint_files),
        references_path=str(value.get("references_path")),
        references_sha256=str(value["references_sha256"]),
        scene_map_manifest_sha256=str(value["scene_map_manifest_sha256"]),
        scene_map_manifest=scene_maps,
        split=str(value.get("split")),
        run_kind=str(value.get("run_kind")),
        condition=(
            None if value.get("condition") is None else str(value.get("condition"))
        ),
    )
    expected_condition = (
        f"all_questions;control_checkpoint_sha256={control.sha256}"
    )
    if (
        reconstructed.as_dict() != value
        or value.get("schema_version") != PROVENANCE_SCHEMA_VERSION
        or reconstructed.config_sha256 != EXPECTED_RUNTIME_CONFIG_SHA256
        or reconstructed.checkpoint_sha256 != EXPECTED_BASE_CHECKPOINT_SHA256
        or reconstructed.references_sha256 != sha256_file(manifest_path)
        or resolve_path(reconstructed.references_path) != manifest_path
        or reconstructed.split != "validation"
        or reconstructed.run_kind != RUN_KIND
        or reconstructed.condition != expected_condition
        or set(reconstructed.scene_map_manifest) != set(EXPECTED_SCENE_IDS)
        or reconstructed.scene_map_manifest_sha256
        != scene_map_manifest_sha256(reconstructed.scene_map_manifest)
    ):
        raise ValueError("V75 prediction provenance differs from the official run")
    return reconstructed


def _reference_key(record: Mapping[str, Any]) -> tuple[str, str]:
    scene_id = record.get("scene_id")
    question_id = record.get("question_id")
    if not isinstance(scene_id, str) or not isinstance(question_id, str):
        raise TypeError("V75 official records require opaque string identifiers")
    return scene_id, question_id


def _validate_references(
    references: Sequence[Mapping[str, Any]],
    manifest_questions: Mapping[tuple[str, str], str],
) -> None:
    if len(references) != EXPECTED_QUESTION_COUNT:
        raise ValueError("V75 scorer requires exactly 216 references")
    seen: set[tuple[str, str]] = set()
    type_counts: Counter[str] = Counter()
    grounding_targets = 0
    for record in references:
        key = _reference_key(record)
        if key in seen:
            raise ValueError(f"Duplicate V75 reference key: {key}")
        seen.add(key)
        if record.get("question") != manifest_questions.get(key):
            raise ValueError("V75 references do not match the sanitized question projection")
        if "answer" not in record or not isinstance(record.get("answer_type"), str):
            raise ValueError(f"V75 reference lacks scoring fields: {key}")
        type_counts[str(record["answer_type"])] += 1
        target = record.get("target_xyz")
        if target is not None:
            if (
                not isinstance(target, list)
                or len(target) != 3
                or any(
                    isinstance(item, bool)
                    or not isinstance(item, (int, float))
                    or not math.isfinite(float(item))
                    for item in target
                )
            ):
                raise ValueError(f"V75 reference has an invalid grounding target: {key}")
            grounding_targets += 1
    if seen != set(manifest_questions):
        raise ValueError("V75 references do not exactly cover the sanitized manifest")
    if dict(sorted(type_counts.items())) != EXPECTED_TYPE_COUNTS:
        raise ValueError("V75 official answer-type inventory changed")
    if grounding_targets != EXPECTED_GROUNDING_TARGET_COUNT:
        raise ValueError("V75 official grounding-target inventory changed")


def _validate_number(value: object, field: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"V75 prediction {field} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number < minimum:
        raise ValueError(f"V75 prediction {field} is invalid")
    return number


def _validate_predictions(
    predictions: Sequence[Mapping[str, Any]],
    manifest_questions: Mapping[tuple[str, str], str],
    *,
    control_sha256: str,
    provenance_sha256: str,
) -> dict[str, str]:
    if len(predictions) != EXPECTED_QUESTION_COUNT:
        raise ValueError("V75 scorer requires exactly 216 predictions")
    seen: set[tuple[str, str]] = set()
    prefixes: defaultdict[str, set[str]] = defaultdict(set)
    for record in predictions:
        if set(record) != _PREDICTION_FIELDS:
            raise ValueError("V75 official prediction fields changed")
        key = _reference_key(record)
        if key in seen:
            raise ValueError(f"Duplicate V75 prediction key: {key}")
        seen.add(key)
        answer = record.get("predicted_answer")
        prefix = record.get("prefix_hash")
        xyz = record.get("grounding_xyz")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError(f"V75 prediction answer is empty: {key}")
        if not _is_sha256(prefix):
            raise ValueError(f"V75 prediction prefix hash is invalid: {key}")
        if (
            not isinstance(xyz, list)
            or len(xyz) != 3
            or any(
                isinstance(item, bool)
                or not isinstance(item, (int, float))
                or not math.isfinite(float(item))
                for item in xyz
            )
        ):
            raise ValueError(f"V75 prediction grounding coordinate is invalid: {key}")
        confidence = _validate_number(record.get("grounding_confidence"), "confidence")
        if confidence > 1.0:
            raise ValueError(f"V75 prediction confidence exceeds one: {key}")
        generated = record.get("generated_tokens")
        if isinstance(generated, bool) or not isinstance(generated, int) or generated < 0:
            raise ValueError(f"V75 generated-token count is invalid: {key}")
        _validate_number(record.get("elapsed_seconds"), "elapsed seconds")
        if record.get("control_checkpoint_sha256") != control_sha256:
            raise ValueError("V75 prediction used a different sealed controller")
        if record.get("provenance_sha256") != provenance_sha256:
            raise ValueError("V75 prediction row has stale provenance")
        validate_v75_control_audit(record.get("control_audit"))
        prefixes[key[0]].add(str(prefix))
    if seen != set(manifest_questions):
        raise ValueError("V75 predictions do not exactly cover the sanitized manifest")
    if set(prefixes) != set(EXPECTED_SCENE_IDS) or any(
        len(values) != 1 for values in prefixes.values()
    ):
        raise ValueError("V75 scene prefix was not invariant for every question")
    return {
        scene_id: next(iter(prefixes[scene_id])) for scene_id in EXPECTED_SCENE_IDS
    }


def score_official_records(
    references: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Return text-free structured metrics after boundary validation."""

    indexed = _prediction_index(predictions)
    standard = score_predictions(references, predictions)
    per_type_counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for reference in references:
        answer_type = str(reference["answer_type"])
        prediction = indexed[_reference_key(reference)]
        correct = canonical_type_specific_match(
            answer_type,
            prediction.get("predicted_answer"),
            reference.get("answer"),
        )
        per_type_counts[answer_type]["total"] += 1
        per_type_counts[answer_type]["correct"] += int(correct)
    canonical_per_type = {
        answer_type: {
            "correct": counts["correct"],
            "total": counts["total"],
            "accuracy": counts["correct"] / counts["total"],
        }
        for answer_type, counts in sorted(per_type_counts.items())
    }
    canonical_correct = sum(value["correct"] for value in canonical_per_type.values())
    changed = _changed_metrics(references, indexed)
    compact_standard = {
        key: standard[key]
        for key in (
            "reference_count",
            "prediction_count",
            "matched_prediction_count",
            "missing_prediction_count",
            "extra_prediction_count",
            "normalized_exact_accuracy",
            "list_order_insensitive_accuracy",
            "count",
            "spatial_relation_accuracy",
            "presence",
            "grounding",
            "per_type",
            "counterfactual",
        )
    }
    return {
        "standard": compact_standard,
        "canonical": {
            "correct": canonical_correct,
            "total": len(references),
            "accuracy": canonical_correct / len(references),
            "per_type": canonical_per_type,
        },
        "changed_counterfactual": changed,
    }


def _gate(metrics: Mapping[str, Any]) -> dict[str, bool]:
    standard = _mapping(metrics.get("standard"), "standard metrics")
    canonical = _mapping(metrics.get("canonical"), "canonical metrics")
    changed = _mapping(
        metrics.get("changed_counterfactual"), "changed counterfactual metrics"
    )
    count = _mapping(standard.get("count"), "count metrics")
    presence = _mapping(standard.get("presence"), "presence metrics")
    thresholds = threshold_contract()
    return {
        "complete_216_question_coverage": (
            standard.get("reference_count") == EXPECTED_QUESTION_COUNT
            and standard.get("prediction_count") == EXPECTED_QUESTION_COUNT
            and standard.get("missing_prediction_count") == 0
            and standard.get("extra_prediction_count") == 0
        ),
        "normalized_exact_accuracy": float(standard["normalized_exact_accuracy"])
        >= float(thresholds["normalized_exact_accuracy_minimum"]),
        "canonical_correct": int(canonical["correct"])
        >= int(thresholds["canonical_correct_minimum"]),
        "spatial_relation_accuracy": float(standard["spatial_relation_accuracy"])
        >= float(thresholds["spatial_relation_accuracy_minimum"]),
        "count_accuracy": float(count["accuracy"])
        >= float(thresholds["count_accuracy_minimum"]),
        "presence_f1": float(presence["f1"])
        >= float(thresholds["presence_f1_minimum"]),
        "changed_complete_units": int(changed["canonical_complete_units"])
        >= int(thresholds["canonical_complete_units_minimum"]),
        "changed_correct_sides": int(changed["canonical_correct_sides"])
        >= int(thresholds["canonical_correct_sides_minimum"]),
        "prediction_changed_units": int(
            changed["canonical_prediction_changed_units"]
        )
        >= int(thresholds["canonical_prediction_changed_units_minimum"]),
        "successful_change_families": int(
            changed["physical_change_families_with_complete_unit"]
        )
        >= int(thresholds["successful_physical_change_families_minimum"]),
    }


def score_v75_official_validation(
    *,
    questions_manifest_path: str | Path,
    references_path: str | Path,
    predictions_path: str | Path,
    control_checkpoint_path: str | Path,
) -> dict[str, Any]:
    """Authenticate prediction/reference isolation and calculate official metrics."""

    manifest = validate_official_question_manifest(questions_manifest_path)
    if manifest.manifest_path is None or manifest.manifest_sha256 is None:
        raise RuntimeError("V75 official manifest lacks file provenance")
    manifest_questions = {
        (record.scene_id, record.question_id): record.question
        for record in manifest.questions
    }
    control = authenticate_v75_control_checkpoint(control_checkpoint_path)
    predictions_source = safe_prediction_input(
        predictions_path, "predictions", kind="file"
    )
    provenance = _load_provenance(
        predictions_source,
        manifest_path=manifest.manifest_path,
        control=control,
    )
    predictions = _load_jsonl(predictions_source, "predictions")
    prefix_hashes = _validate_predictions(
        predictions,
        manifest_questions,
        control_sha256=control.sha256,
        provenance_sha256=provenance.sha256,
    )

    # This is the first and only answer-bearing file open in the scorer path.
    references_source = _safe_references(references_path)
    references = _load_jsonl(references_source, "references")
    _validate_references(references, manifest_questions)
    if manifest.source_qa_sha256 != sha256_file(references_source):
        raise ValueError("V75 sanitized manifest is not bound to scorer references")
    metrics = score_official_records(references, predictions)
    gates = _gate(metrics)
    report = {
        "schema_version": 1,
        "artifact": SCORE_ARTIFACT,
        "passed": all(gates.values()),
        "scope": {
            "split": "validation",
            "scene_ids": list(EXPECTED_SCENE_IDS),
            "question_count": EXPECTED_QUESTION_COUNT,
            "candidate_count": 1,
            "model_loaded": False,
            "scene_map_loaded": False,
            "simulator_oracle_loaded": False,
            "answer_references_loaded_only_by_isolated_scorer": True,
            "prediction_process_accepts_answer_references": False,
            "question_or_answer_text_serialized": False,
            "question_dependent_scene_retrieval": False,
            "all_256_environment_latents_attended": True,
            "scene_prefix_built_before_questions": True,
            "scene_prefix_invariant_within_scene": True,
        },
        "inputs": {
            "questions_manifest_sha256": manifest.manifest_sha256,
            "references_sha256": sha256_file(references_source),
            "predictions_sha256": sha256_file(predictions_source),
            "prediction_provenance_sha256": provenance.sha256,
            "control_checkpoint_sha256": control.sha256,
            "control_weights_sha256": control.weights_sha256,
            "source_v75_candidate_sha256": control.metadata[
                "source_v75_candidate_sha256"
            ],
            "base_checkpoint_sha256": provenance.checkpoint_sha256,
            "runtime_config_sha256": provenance.config_sha256,
            "scene_map_manifest_sha256": provenance.scene_map_manifest_sha256,
        },
        "prefix_sha256_by_scene": prefix_hashes,
        "metrics": metrics,
        "thresholds": threshold_contract(),
        "gates": gates,
    }
    serialized = json.dumps(report, sort_keys=True, allow_nan=False)
    if any(
        token in serialized
        for token in ('"question":', '"answer":', '"predicted_answer":')
    ):
        raise AssertionError("V75 score report contains forbidden example text")
    return report


def _atomic_write_json(path: Path, value: Mapping[str, Any], *, force: bool) -> None:
    destination = reject_symlink_components(path, "score output")
    if destination.exists() and not force:
        raise FileExistsError(f"V75 score already exists: {destination}")
    if destination.exists() and not destination.is_file():
        raise ValueError(f"V75 score output is not a regular file: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--questions-manifest", type=Path, default=DEFAULT_QUESTIONS_MANIFEST
    )
    parser.add_argument("--references", type=Path, default=DEFAULT_REFERENCES)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument(
        "--control-checkpoint", type=Path, default=DEFAULT_CONTROL_CHECKPOINT
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_SCORE)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = score_v75_official_validation(
        questions_manifest_path=args.questions_manifest,
        references_path=args.references,
        predictions_path=args.predictions,
        control_checkpoint_path=args.control_checkpoint,
    )
    _atomic_write_json(resolve_path(args.output), report, force=args.force)
    print(
        json.dumps(
            {
                "artifact": ARTIFACT,
                "passed": report["passed"],
                "canonical_accuracy": report["metrics"]["canonical"]["accuracy"],
                "normalized_exact_accuracy": report["metrics"]["standard"][
                    "normalized_exact_accuracy"
                ],
                "counterfactual_complete_units": report["metrics"][
                    "changed_counterfactual"
                ]["canonical_complete_units"],
                "output": str(resolve_path(args.output)),
            },
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "SCORE_ARTIFACT",
    "main",
    "score_official_records",
    "score_v75_official_validation",
]
