"""Model-free, label-isolated structured scorer for V95 known development.

The complete prediction/provenance/access/completion bundle is authenticated
before this process opens the pinned answer-bearing file.  Only aggregate
counts and accuracies are serialized; questions, answers, predictions, opaque
row identifiers, and per-row outcomes remain process-local.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from semantic_3d_chat.chat.file_audit import FileAccessAudit
from semantic_3d_chat.evaluation.baseline_io import read_jsonl
from semantic_3d_chat.evaluation.v56_fresh_development_score import (
    EXPECTED_CHANGED_SIDE_COUNT,
    EXPECTED_CHANGED_UNIT_COUNT,
    EXPECTED_TYPE_COUNTS,
    _changed_metrics,
    canonical_answer_key,
    canonical_type_specific_match,
)
from semantic_3d_chat.evaluation.v85_strict_multiscene_preflight import sha256_file_v85
from semantic_3d_chat.evaluation.v95_known_development_common import (
    ARMS,
    PAIRED_WRONG_SCENE,
    PRIMARY,
    QUESTION_COUNT,
    REFERENCE_SHA256,
    SCENE_IDS,
    SCHEMA_VERSION,
    STRUCTURED_SCORE_ARTIFACT,
    assert_aggregate_only_v95,
    authenticate_prediction_bundle_v95,
    canonical_sha256_v95,
    resolve_v95,
    structured_score_forbidden_roots_v95,
    validate_structured_metrics_v95,
    write_json_create_once_v95,
)
from semantic_3d_chat.evaluation.v95_known_development_implementation import (
    hardened_evaluation_stage_v95,
)
from semantic_3d_chat.evaluation.v95_strict_causal_successor_preflight import CONFIG


def load_references_v95(config: Mapping[str, Any], questions: Any) -> list[dict[str, Any]]:
    """Load the pinned references; callers must authenticate inference first."""

    path = resolve_v95(config["known_development_gate"]["labels_path"])
    if (
        path.is_symlink()
        or not path.is_file()
        or sha256_file_v85(path) != REFERENCE_SHA256
        or questions.source_qa_sha256 != REFERENCE_SHA256
    ):
        raise ValueError("V95 known-development reference bytes changed")
    rows = read_jsonl(path)
    question_by_key = {(row.scene_id, row.question_id): row.question for row in questions.questions}
    seen: set[tuple[str, str]] = set()
    types: Counter[str] = Counter()
    changed_rows = 0
    for row in rows:
        key = (str(row.get("scene_id")), str(row.get("question_id")))
        if (
            key in seen
            or key not in question_by_key
            or row.get("question") != question_by_key[key]
            or not isinstance(row.get("answer"), str)
            or not isinstance(row.get("answer_type"), str)
        ):
            raise ValueError(f"V95 reference projection changed: {key}")
        seen.add(key)
        types[str(row["answer_type"])] += 1
        changed_rows += row.get("counterfactual_expected_change") is True
    if (
        len(rows) != QUESTION_COUNT
        or seen != set(question_by_key)
        or dict(sorted(types.items())) != EXPECTED_TYPE_COUNTS
        or changed_rows != EXPECTED_CHANGED_SIDE_COUNT
    ):
        raise ValueError("V95 known-development reference coverage changed")
    return rows


def _accuracy_v95(
    references: Sequence[Mapping[str, Any]],
    predictions: Mapping[tuple[str, str], Mapping[str, Any]],
    field: str,
) -> dict[str, Any]:
    values: list[bool] = []
    by_type: defaultdict[str, list[bool]] = defaultdict(list)
    for reference in references:
        key = (str(reference["scene_id"]), str(reference["question_id"]))
        correct = canonical_type_specific_match(
            str(reference["answer_type"]), predictions[key][field], reference["answer"]
        )
        values.append(correct)
        by_type[str(reference["answer_type"])].append(correct)
    return {
        "correct": sum(values),
        "total": len(values),
        "accuracy": sum(values) / len(values),
        "by_answer_type": {
            answer_type: {
                "correct": sum(items),
                "total": len(items),
                "accuracy": sum(items) / len(items),
            }
            for answer_type, items in sorted(by_type.items())
        },
    }


def structured_metrics_v95(
    references: Sequence[Mapping[str, Any]],
    prediction_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Return aggregate structured metrics from in-memory reference/prediction rows."""

    predictions = {(str(row["scene_id"]), str(row["question_id"])): row for row in prediction_rows}
    reference_keys = {(str(row["scene_id"]), str(row["question_id"])) for row in references}
    if (
        len(references) != QUESTION_COUNT
        or len(predictions) != QUESTION_COUNT
        or set(predictions) != reference_keys
    ):
        raise ValueError("V95 structured scorer requires exact matching 216-row inputs")
    arms = {arm: _accuracy_v95(references, predictions, f"{arm}_prediction") for arm in ARMS}
    changed = _changed_metrics(
        references,
        {
            key: {"predicted_answer": row[f"{PRIMARY}_prediction"]}
            for key, row in predictions.items()
        },
    )
    if (
        changed.get("unit_count") != EXPECTED_CHANGED_UNIT_COUNT
        or changed.get("side_count") != EXPECTED_CHANGED_SIDE_COUNT
    ):
        raise ValueError("V95 counterfactual aggregate inventory changed")
    comparisons: dict[str, dict[str, Any]] = {}
    for arm in ARMS[1:]:
        changes = sum(
            canonical_answer_key(
                str(reference["answer_type"]),
                predictions[(str(reference["scene_id"]), str(reference["question_id"]))][
                    f"{PRIMARY}_prediction"
                ],
            )
            != canonical_answer_key(
                str(reference["answer_type"]),
                predictions[(str(reference["scene_id"]), str(reference["question_id"]))][
                    f"{arm}_prediction"
                ],
            )
            for reference in references
        )
        comparisons[arm] = {
            "accuracy_drop_from_primary": arms[PRIMARY]["accuracy"] - arms[arm]["accuracy"],
            "prediction_change_count": changes,
            "prediction_change_rate": changes / QUESTION_COUNT,
        }
    metrics = {
        "arms": arms,
        "counterfactual": changed,
        "comparisons": comparisons,
        "paired_wrong_scene_included": PAIRED_WRONG_SCENE in arms,
    }
    # The boolean above is useful context but the common strict validator keeps
    # the aggregate schema minimal.  Remove it after asserting the condition.
    if metrics.pop("paired_wrong_scene_included") is not True:
        raise RuntimeError("V95 structured scorer omitted paired-wrong-scene")
    validate_structured_metrics_v95(metrics)
    return metrics


@hardened_evaluation_stage_v95
def score_known_development_v95(
    config_path: str | Path = CONFIG,
) -> dict[str, Any]:
    """Authenticate predictions, then open labels and publish one aggregate."""

    # This call is intentionally first: it has its own label-blocking audit and
    # authenticates all 216 rows plus fixed-final immutability before label I/O.
    bundle = authenticate_prediction_bundle_v95(config_path)
    path = bundle["paths"].structured_score
    if path.exists() or path.is_symlink():
        raise FileExistsError("V95 structured score is create-once")
    audit = FileAccessAudit(
        forbidden_roots=structured_score_forbidden_roots_v95(bundle["config"]),
        forbidden_component_names={"oracle"},
        block_forbidden=True,
    )
    with audit:
        references = load_references_v95(bundle["config"], bundle["questions"])
        metrics = structured_metrics_v95(references, bundle["rows"])
    audit.assert_clean()
    report = {
        "artifact": STRUCTURED_SCORE_ARTIFACT,
        "schema_version": SCHEMA_VERSION,
        "status": "measured_aggregate_only_not_yet_gated",
        "candidate_fingerprint_sha256": bundle["fixed"].candidate["fingerprint_sha256"],
        "memory_manifest_sha256": bundle["fixed"].memory_manifest_sha256,
        "bound_memory_inventory_sha256": canonical_sha256_v95(bundle["fixed"].memory_hashes),
        "question_manifest_sha256": bundle["questions"].manifest_sha256,
        "questions_sha256": bundle["questions"].questions_sha256,
        "reference_sha256": REFERENCE_SHA256,
        "prediction_sha256": bundle["prediction_sha256"],
        "prediction_provenance_sha256": bundle["provenance"]["provenance_sha256"],
        "prediction_access_sha256": bundle["access_sha256"],
        "prediction_completion_sha256": bundle["completion_sha256"],
        "row_count": QUESTION_COUNT,
        "scene_count": len(SCENE_IDS),
        "prediction_bundle_authenticated_before_labels_opened": True,
        "labels_opened_only_by_separate_scorer": True,
        "scorer_loaded_model": False,
        "row_level_content_serialized": False,
        "metrics": metrics,
        "runtime_promotion_authorized": False,
    }
    assert_aggregate_only_v95(report)
    write_json_create_once_v95(path, report)
    return {**report, "structured_score_sha256": sha256_file_v85(path)}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(CONFIG))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    print(json.dumps(score_known_development_v95(args.config), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "load_references_v95",
    "main",
    "score_known_development_v95",
    "structured_metrics_v95",
]
