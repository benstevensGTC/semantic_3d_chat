"""Model-free scorer for the evaluation-only oracle-text upper bound.

Only this process opens answer-bearing QA references.  It authenticates the
completed inference artifacts, requires exact key coverage, and emits aggregate
metrics without serializing questions, reference answers, or model answers.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.evaluation.baseline_io import (
    read_jsonl,
    sha256_file,
    text_fingerprint,
)
from semantic_3d_chat.evaluation.metrics import score_predictions
from semantic_3d_chat.evaluation.oracle_text_artifacts import (
    PREDICTION_BASELINE,
    atomic_write_json,
    authenticate_completed_prediction_report,
    canonical_json_sha256,
    default_prediction_report_path,
    default_provenance_path,
    load_prediction_provenance,
    load_prediction_records,
    validate_v55_development_scope,
)

DEFAULT_REFERENCES = Path("data_diverse28/qa/validation.jsonl")
DEFAULT_PREDICTIONS = Path(
    "reports/gemma4/evaluation_only/oracle_text_upper_bound/v55_predictions.jsonl"
)
DEFAULT_OUTPUT = Path("reports/gemma4/metrics/oracle_text_upper_bound_v55_development.json")


def _key(record: Mapping[str, Any]) -> tuple[str, str]:
    scene_id = record.get("scene_id")
    question_id = record.get("question_id")
    if not isinstance(scene_id, str) or not isinstance(question_id, str):
        raise TypeError("Reference and prediction records require opaque string keys")
    return scene_id, question_id


def score_oracle_text_predictions(
    references_path: str | Path,
    predictions_path: str | Path,
    output_path: str | Path,
    *,
    provenance_path: str | Path | None = None,
    inference_report_path: str | Path | None = None,
    require_v55_development: bool = True,
    require_local_gemma: bool = True,
) -> dict[str, Any]:
    """Authenticate and score a completed answer-blind inference run."""

    references_source = Path(references_path).expanduser().resolve()
    predictions_source = Path(predictions_path).expanduser().resolve()
    provenance_source = (
        Path(provenance_path).expanduser().resolve()
        if provenance_path is not None
        else default_provenance_path(predictions_source)
    )
    inference_report_source = (
        Path(inference_report_path).expanduser().resolve()
        if inference_report_path is not None
        else default_prediction_report_path(predictions_source)
    )
    if not references_source.is_file() or references_source.is_symlink():
        raise FileNotFoundError(f"QA references are unavailable or unsafe: {references_source}")

    provenance = load_prediction_provenance(provenance_source)
    provenance_sha256 = str(provenance["inference_provenance_sha256"])
    identity = provenance["identity"]
    if not isinstance(identity, Mapping):
        raise TypeError("Prediction provenance identity must be an object")
    inference_contract = identity.get("inference_contract")
    if not isinstance(inference_contract, Mapping):
        raise TypeError("Prediction provenance lacks an inference contract")
    if require_local_gemma and (
        inference_contract.get("generation_backend") != "local_gemma"
        or inference_contract.get("scientific_measurement_eligible") is not True
    ):
        raise ValueError("Scientific oracle-text scoring requires actual local Gemma inference")
    completed = authenticate_completed_prediction_report(
        inference_report_source,
        predictions_path=predictions_source,
        provenance_path=provenance_source,
    )
    if completed.get("inference_provenance_sha256") != provenance_sha256:
        raise ValueError("Completed report and prediction provenance disagree")
    predictions = load_prediction_records(
        predictions_source,
        provenance_sha256=provenance_sha256,
    )
    references_sha256 = sha256_file(references_source)
    if references_sha256 != identity.get("source_qa_sha256"):
        raise ValueError("Answer-bearing references differ from the prepared question source")
    references = read_jsonl(references_source)
    validate_v55_development_scope(
        [str(record.get("scene_id")) for record in references],
        len(references),
        required=require_v55_development,
    )
    reference_keys = {_key(record) for record in references}
    prediction_keys = {_key(record) for record in predictions}
    if len(reference_keys) != len(references):
        raise ValueError("QA references contain duplicate opaque keys")
    if prediction_keys != reference_keys or len(predictions) != len(references):
        raise ValueError(
            "Oracle-text scoring requires exact prediction coverage; "
            f"missing={len(reference_keys - prediction_keys)} "
            f"extra={len(prediction_keys - reference_keys)}"
        )
    prediction_index = {_key(record): record for record in predictions}
    for reference in references:
        key = _key(reference)
        question = reference.get("question")
        if not isinstance(question, str) or not question:
            raise ValueError(f"QA reference {key} lacks question text")
        expected_question_sha256 = text_fingerprint(key[0], key[1], question)
        if prediction_index[key]["question_sha256"] != expected_question_sha256:
            raise ValueError(f"Prediction {key} was generated for a different question")

    metrics = score_predictions(references, predictions)
    compact_metrics = {
        key: metrics[key]
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
    scoring_identity = {
        "baseline": PREDICTION_BASELINE,
        "evaluation_only": True,
        "primary_path_eligible": False,
        "prohibited_primary_input": True,
        "inference_provenance_sha256": provenance_sha256,
        "predictions_sha256": sha256_file(predictions_source),
        "references_sha256": references_sha256,
        "scoring_implementation_sha256": sha256_file(Path(__file__).resolve()),
    }
    report = {
        "schema": "semantic_3d_chat.oracle_text_score.v1",
        "schema_version": 1,
        "artifact": (
            "oracle_text_upper_bound_v55_development"
            if require_v55_development
            else "oracle_text_upper_bound_custom"
        ),
        "baseline": PREDICTION_BASELINE,
        "evaluation_only": True,
        "primary_path_eligible": False,
        "prohibited_primary_input": True,
        "claim": (
            "Textual oracle facts were deliberately exposed to local Gemma; this is an "
            "evaluation upper bound and cannot satisfy the primary continuous-memory goal."
        ),
        "scope": {
            "split": "development_validation" if require_v55_development else "custom",
            "scene_ids": sorted({str(record["scene_id"]) for record in references}),
            "question_count": len(references),
            "model_loaded_by_scorer": False,
            "oracle_loaded_by_scorer": False,
            "answer_bearing_references_loaded_by_scorer": True,
            "question_or_answer_text_serialized_in_report": False,
            "local_gemma_inference_authenticated": (
                inference_contract.get("generation_backend") == "local_gemma"
                and inference_contract.get("scientific_measurement_eligible") is True
            ),
        },
        "inputs": {
            "references_path": str(references_source),
            "references_sha256": references_sha256,
            "predictions_path": str(predictions_source),
            "predictions_sha256": scoring_identity["predictions_sha256"],
            "prediction_provenance_path": str(provenance_source),
            "prediction_provenance_file_sha256": sha256_file(provenance_source),
            "inference_report_path": str(inference_report_source),
            "inference_report_sha256": sha256_file(inference_report_source),
            "inference_provenance_sha256": provenance_sha256,
        },
        "scoring_identity": scoring_identity,
        "scoring_sha256": canonical_json_sha256(scoring_identity),
        "metrics": compact_metrics,
    }
    destination = Path(output_path).expanduser().resolve()
    atomic_write_json(destination, report)
    return {**report, "metrics_path": str(destination)}


def _project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path, default=DEFAULT_REFERENCES)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--provenance", type=Path)
    parser.add_argument("--inference-report", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--allow-non-v55-scope", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = score_oracle_text_predictions(
        _project_path(args.references),
        _project_path(args.predictions),
        _project_path(args.output),
        provenance_path=_project_path(args.provenance) if args.provenance else None,
        inference_report_path=(
            _project_path(args.inference_report) if args.inference_report else None
        ),
        require_v55_development=not args.allow_non_v55_scope,
    )
    print(
        json.dumps(
            {
                "metrics_path": report["metrics_path"],
                "reference_count": report["metrics"]["reference_count"],
                "normalized_exact_accuracy": report["metrics"]["normalized_exact_accuracy"],
                "spatial_relation_accuracy": report["metrics"]["spatial_relation_accuracy"],
                "count_accuracy": report["metrics"]["count"]["accuracy"],
                "counterfactual": report["metrics"]["counterfactual"],
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
