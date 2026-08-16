"""Verify and score a complete continuous-scene control suite.

This is an evaluation-side command: it is allowed to open reference QA after
inference has finished, but it never imports a chat runtime or loads a model.
Before scoring, it authenticates every prediction file and verifies that the
question-independent prefix receipt is internally consistent for every scene.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from semantic_3d_chat.evaluation.ablations import file_sha256
from semantic_3d_chat.evaluation.control_predict import CONTROL_CONDITIONS
from semantic_3d_chat.evaluation.run import load_jsonl, score_jsonl_files

REQUIRED_CAUSAL_CONTROLS: Final[tuple[str, ...]] = CONTROL_CONDITIONS


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _prediction_path(
    manifest_path: Path,
    condition: str,
    report: Mapping[str, Any],
) -> Path:
    recorded = report.get("path")
    if not isinstance(recorded, str) or not recorded:
        raise ValueError(f"Control {condition} has no prediction path")
    path = Path(recorded).expanduser()
    if not path.is_absolute():
        path = manifest_path.parent / path
    path = path.resolve()
    if path.parent != manifest_path.parent or path.name != f"{condition}.jsonl":
        raise ValueError(
            f"Control {condition} prediction must be {manifest_path.parent / f'{condition}.jsonl'}"
        )
    if not path.is_file():
        raise FileNotFoundError(f"Control prediction does not exist: {path}")
    expected_hash = report.get("sha256")
    actual_hash = file_sha256(path)
    if expected_hash != actual_hash:
        raise ValueError(
            f"Control {condition} prediction hash mismatch: "
            f"manifest={expected_hash!r} actual={actual_hash}"
        )
    return path


def _validate_prefix_receipts(
    condition: str,
    report: Mapping[str, Any],
    predictions: Sequence[Mapping[str, Any]],
) -> None:
    scene_reports = report.get("scenes")
    if not isinstance(scene_reports, dict) or not scene_reports:
        raise ValueError(f"Control {condition} has no per-scene prefix receipts")
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for record in predictions:
        if record.get("condition") != condition:
            raise ValueError(f"Prediction is mislabeled in {condition}.jsonl")
        scene_id = record.get("scene_id")
        if not isinstance(scene_id, str) or not scene_id:
            raise ValueError(f"Control {condition} prediction has invalid scene_id")
        grouped.setdefault(scene_id, []).append(record)
    if set(grouped) != set(scene_reports):
        raise ValueError(f"Control {condition} prediction scenes do not match its prefix manifest")
    for scene_id, records in grouped.items():
        scene_report = scene_reports[scene_id]
        if not isinstance(scene_report, dict):
            raise TypeError(f"Control {condition}/{scene_id} receipt must be an object")
        if scene_report.get("prefix_built_before_questions") is not True:
            raise ValueError(f"Control {condition}/{scene_id} prefix timing is not attested")
        prefix_hash = scene_report.get("prefix_hash")
        hashes = {record.get("prefix_hash") for record in records}
        if hashes != {prefix_hash} or not isinstance(prefix_hash, str):
            raise ValueError(f"Control {condition}/{scene_id} prefix is not invariant")
        source_scene = scene_report.get("prefix_source_scene_id")
        record_sources = {record.get("prefix_source_scene_id") for record in records}
        if record_sources != {source_scene}:
            raise ValueError(f"Control {condition}/{scene_id} source receipt is inconsistent")
        if condition == "wrong_scene_prefix":
            if source_scene == scene_id:
                raise ValueError(f"Wrong-scene control reused {scene_id}'s own prefix")
        elif source_scene != scene_id:
            raise ValueError(f"Control {condition}/{scene_id} used another scene's prefix")


def _validate_map_control_receipt(
    condition: str,
    report: Mapping[str, Any],
) -> None:
    expected_fields = {
        "semantic_shuffle": ["semantic"],
        "position_shuffle": ["xyz"],
        "geometry_only": ["semantic"],
        "semantics_without_xyz": ["xyz"],
        "remove_rgb": ["rgb"],
        "remove_normals": ["normal"],
    }
    if condition not in expected_fields:
        return
    scenes = report["scenes"]
    for scene_id, scene_report in scenes.items():
        metadata = scene_report.get("metadata")
        if not isinstance(metadata, dict):
            raise TypeError(f"Control {condition}/{scene_id} has no transform receipt")
        if metadata.get("affected_fields") != expected_fields[condition]:
            raise ValueError(f"Control {condition}/{scene_id} changed unexpected fields")
        if metadata.get("question_dependent_selection") is not False:
            raise ValueError(f"Control {condition}/{scene_id} used question selection")
        if condition in {"semantic_shuffle", "position_shuffle"}:
            permutation_hash = metadata.get("permutation_sha256")
            if not isinstance(permutation_hash, str) or len(permutation_hash) != 64:
                raise ValueError(f"Control {condition}/{scene_id} lacks permutation receipt")


def score_control_suite(
    manifest_path: str | Path,
    references_path: str | Path,
    *,
    output_directory: str | Path,
    summary_path: str | Path | None = None,
    required_conditions: Sequence[str] = REQUIRED_CAUSAL_CONTROLS,
    require_complete_predictions: bool = True,
) -> dict[str, Any]:
    """Authenticate, score, and compare continuous-scene controls."""

    source = Path(manifest_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Control manifest does not exist: {source}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("Unsupported control manifest")
    if payload.get("question_dependent_retrieval") is not False:
        raise ValueError("Control manifest does not prohibit question-dependent retrieval")
    if payload.get("one_prefix_per_scene_condition") is not True:
        raise ValueError("Control manifest does not attest one prefix per scene/condition")
    reports = payload.get("conditions")
    if not isinstance(reports, dict):
        raise TypeError("Control manifest conditions must be an object")
    required = tuple(required_conditions)
    if not required or len(set(required)) != len(required):
        raise ValueError("Required controls must be a non-empty unique sequence")
    unknown = sorted(set(required) - set(CONTROL_CONDITIONS))
    if unknown:
        raise ValueError(f"Unknown required controls: {unknown}")
    missing = sorted(set(required) - set(reports))
    if missing:
        raise ValueError(f"Control manifest is missing required conditions: {missing}")

    references = Path(references_path).expanduser().resolve()
    reference_records = load_jsonl(references)
    reference_keys = {
        (record.get("scene_id"), record.get("question_id")) for record in reference_records
    }
    output_root = Path(output_directory).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    for condition in required:
        report = reports[condition]
        if not isinstance(report, dict):
            raise TypeError(f"Control {condition} report must be an object")
        predictions_path = _prediction_path(source, condition, report)
        prediction_records = load_jsonl(predictions_path)
        _validate_prefix_receipts(condition, report, prediction_records)
        _validate_map_control_receipt(condition, report)
        prediction_keys = {
            (record.get("scene_id"), record.get("question_id")) for record in prediction_records
        }
        if require_complete_predictions and prediction_keys != reference_keys:
            raise ValueError(f"Control {condition} prediction keys do not exactly match references")
        metrics_path = output_root / f"{condition}.json"
        metrics = score_jsonl_files(
            references,
            predictions_path,
            output_path=metrics_path,
        )
        results[condition] = {
            "normalized_exact_accuracy": metrics["normalized_exact_accuracy"],
            "spatial_relation_accuracy": metrics["spatial_relation_accuracy"],
            "count_accuracy": metrics["count"]["accuracy"],
            "grounding_mean_error_m": metrics["grounding"]["mean_coordinate_error_m"],
            "counterfactual_changed_rate": metrics["counterfactual"]["changed_when_expected_rate"],
            "prediction_count": metrics["prediction_count"],
            "prediction_sha256": metrics["predictions_sha256"],
            "metrics_path": str(metrics_path),
            "metrics_sha256": file_sha256(metrics_path),
        }

    primary_accuracy = results["primary"]["normalized_exact_accuracy"]
    for result in results.values():
        result["exact_accuracy_delta_vs_primary"] = (
            result["normalized_exact_accuracy"] - primary_accuracy
        )
    summary = {
        "schema_version": 1,
        "artifact": "continuous_scene_control_scores",
        "control_manifest_path": str(source),
        "control_manifest_sha256": file_sha256(source),
        "references_path": str(references),
        "references_sha256": file_sha256(references),
        "complete_prediction_coverage_required": bool(require_complete_predictions),
        "question_dependent_retrieval": False,
        "one_prefix_per_scene_condition": True,
        "results": results,
    }
    destination = (
        Path(summary_path).expanduser().resolve()
        if summary_path is not None
        else output_root / "summary.json"
    )
    _atomic_json(destination, summary)
    return {**summary, "summary_path": str(destination)}


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--condition", action="append", choices=CONTROL_CONDITIONS)
    parser.add_argument("--allow-partial", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = score_control_suite(
        args.manifest,
        args.references,
        output_directory=args.output_dir,
        summary_path=args.summary,
        required_conditions=tuple(args.condition or REQUIRED_CAUSAL_CONTROLS),
        require_complete_predictions=not args.allow_partial,
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
