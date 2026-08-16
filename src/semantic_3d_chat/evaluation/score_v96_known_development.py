"""Model-free, label-isolated structured scorer for V96 development."""

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
from semantic_3d_chat.evaluation.v96_atomic_pair_repair_preflight import CONFIG
from semantic_3d_chat.evaluation.v96_known_development_common import (
    ARMS,
    INVARIANT_SIDE_COUNT,
    INVARIANT_UNIT_COUNT,
    PAIR_SCENE,
    PAIRED_WRONG_SCENE,
    PRIMARY,
    QUESTION_COUNT,
    REFERENCE_SHA256,
    SCENE_IDS,
    SCHEMA_VERSION,
    STRUCTURED_SCORE_ARTIFACT,
    assert_aggregate_only_v96,
    authenticate_prediction_bundle_v96,
    canonical_sha256_v96,
    resolve_v96,
    structured_score_forbidden_roots_v96,
    validate_structured_metrics_v96,
    write_json_create_once_v96,
)
from semantic_3d_chat.evaluation.v96_known_development_implementation import (
    hardened_evaluation_stage_v96,
)


def load_references_v96(config: Mapping[str, Any], questions: Any) -> list[dict[str, Any]]:
    """Open the pinned labels only after the prediction bundle is authenticated."""

    path = resolve_v96(config["known_development_gate"]["labels_path"])
    if (
        path.is_symlink()
        or not path.is_file()
        or sha256_file_v85(path) != REFERENCE_SHA256
        or questions.source_qa_sha256 != REFERENCE_SHA256
    ):
        raise ValueError("V96 known-development reference bytes changed")
    rows = read_jsonl(path)
    question_by_key = {
        (row.scene_id, row.question_id): row.question for row in questions.questions
    }
    seen: set[tuple[str, str]] = set()
    types: Counter[str] = Counter()
    changed_rows = 0
    invariant_rows = 0
    pair_fields = (
        "counterfactual_pair_id",
        "counterfactual_paired_scene_id",
        "counterfactual_question_key",
        "counterfactual_change_type",
    )
    for row in rows:
        key = (str(row.get("scene_id")), str(row.get("question_id")))
        expected_change = row.get("counterfactual_expected_change")
        if (
            key in seen
            or key not in question_by_key
            or row.get("question") != question_by_key[key]
            or not isinstance(row.get("answer"), str)
            or not isinstance(row.get("answer_type"), str)
            or type(expected_change) is not bool
            or any(not isinstance(row.get(field), str) or not row[field] for field in pair_fields)
            or row.get("counterfactual_paired_scene_id") != PAIR_SCENE.get(key[0])
        ):
            raise ValueError(f"V96 reference projection changed: {key}")
        seen.add(key)
        types[str(row["answer_type"])] += 1
        changed_rows += expected_change is True
        invariant_rows += expected_change is False
    if (
        len(rows) != QUESTION_COUNT
        or seen != set(question_by_key)
        or dict(sorted(types.items())) != EXPECTED_TYPE_COUNTS
        or changed_rows != EXPECTED_CHANGED_SIDE_COUNT
        or invariant_rows != INVARIANT_SIDE_COUNT
    ):
        raise ValueError("V96 known-development reference coverage changed")
    return rows


def _accuracy_v96(
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


def stable_invariant_metrics_v96(
    references: Sequence[Mapping[str, Any]],
    predictions: Mapping[tuple[str, str], Mapping[str, Any]],
) -> dict[str, Any]:
    """Count sides whose prediction changes across a fact-invariant scene pair."""

    grouped: defaultdict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for reference in references:
        if reference.get("counterfactual_expected_change") is False:
            grouped[
                (
                    str(reference["counterfactual_pair_id"]),
                    str(reference["counterfactual_question_key"]),
                )
            ].append(reference)
    if len(grouped) != INVARIANT_UNIT_COUNT or sum(map(len, grouped.values())) != INVARIANT_SIDE_COUNT:
        raise ValueError("V96 invariant pair inventory changed")
    false_change_sides = 0
    for unit, sides in grouped.items():
        if len(sides) != 2:
            raise ValueError(f"V96 invariant unit does not have two sides: {unit}")
        left, right = sides
        left_scene = str(left["scene_id"])
        right_scene = str(right["scene_id"])
        answer_type = str(left["answer_type"])
        if (
            PAIR_SCENE.get(left_scene) != right_scene
            or PAIR_SCENE.get(right_scene) != left_scene
            or right.get("answer_type") != answer_type
            or canonical_answer_key(answer_type, left["answer"])
            != canonical_answer_key(answer_type, right["answer"])
        ):
            raise ValueError(f"V96 invariant unit semantics changed: {unit}")
        left_key = (left_scene, str(left["question_id"]))
        right_key = (right_scene, str(right["question_id"]))
        changed = canonical_answer_key(
            answer_type, predictions[left_key][f"{PRIMARY}_prediction"]
        ) != canonical_answer_key(
            answer_type, predictions[right_key][f"{PRIMARY}_prediction"]
        )
        false_change_sides += 2 * int(changed)
    return {
        "side_count": INVARIANT_SIDE_COUNT,
        "unit_count": INVARIANT_UNIT_COUNT,
        "invariant_false_change_count": false_change_sides,
        "invariant_false_change_rate": false_change_sides / INVARIANT_SIDE_COUNT,
    }


def structured_metrics_v96(
    references: Sequence[Mapping[str, Any]],
    prediction_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    predictions = {
        (str(row["scene_id"]), str(row["question_id"])): row
        for row in prediction_rows
    }
    reference_keys = {
        (str(row["scene_id"]), str(row["question_id"])) for row in references
    }
    if (
        len(references) != QUESTION_COUNT
        or len(predictions) != QUESTION_COUNT
        or set(predictions) != reference_keys
    ):
        raise ValueError("V96 structured scorer requires exact matching 216-row inputs")
    arms = {
        arm: _accuracy_v96(references, predictions, f"{arm}_prediction")
        for arm in ARMS
    }
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
        raise ValueError("V96 counterfactual aggregate inventory changed")
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
            "accuracy_drop_from_primary": arms[PRIMARY]["accuracy"]
            - arms[arm]["accuracy"],
            "prediction_change_count": changes,
            "prediction_change_rate": changes / QUESTION_COUNT,
        }
    if PAIRED_WRONG_SCENE not in arms:
        raise RuntimeError("V96 scorer omitted paired-wrong-scene")
    metrics = {
        "arms": arms,
        "counterfactual": changed,
        "comparisons": comparisons,
        "stable_invariant": stable_invariant_metrics_v96(references, predictions),
    }
    validate_structured_metrics_v96(metrics)
    return metrics


@hardened_evaluation_stage_v96
def score_known_development_v96(
    config_path: str | Path = CONFIG,
) -> dict[str, Any]:
    """Authenticate predictions first, then publish aggregate-only metrics."""

    bundle = authenticate_prediction_bundle_v96(config_path)
    path = bundle["paths"].structured_score
    if path.exists() or path.is_symlink():
        raise FileExistsError("V96 structured score is create-once")
    audit = FileAccessAudit(
        forbidden_roots=structured_score_forbidden_roots_v96(bundle["config"]),
        forbidden_component_names={"oracle"},
        block_forbidden=True,
    )
    with audit:
        references = load_references_v96(bundle["config"], bundle["questions"])
        metrics = structured_metrics_v96(references, bundle["rows"])
    audit.assert_clean()
    report = {
        "artifact": STRUCTURED_SCORE_ARTIFACT,
        "schema_version": SCHEMA_VERSION,
        "status": "measured_aggregate_only_not_yet_gated",
        "candidate_fingerprint_sha256": bundle["fixed"].candidate[
            "fingerprint_sha256"
        ],
        "frozen_v95_state_sha256": bundle["fixed"].candidate[
            "frozen_v95_state_sha256"
        ],
        "memory_manifest_sha256": bundle["fixed"].memory_manifest_sha256,
        "bound_memory_inventory_sha256": canonical_sha256_v96(
            bundle["fixed"].memory_hashes
        ),
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
    assert_aggregate_only_v96(report)
    write_json_create_once_v96(path, report)
    return {**report, "structured_score_sha256": sha256_file_v85(path)}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(CONFIG))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    print(json.dumps(score_known_development_v96(args.config), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "load_references_v96",
    "main",
    "score_known_development_v96",
    "stable_invariant_metrics_v96",
    "structured_metrics_v96",
]
