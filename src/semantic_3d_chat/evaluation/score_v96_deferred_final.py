"""Model-free, label-isolated V96 deferred-final structured scorer."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from semantic_3d_chat.chat.file_audit import FileAccessAudit
from semantic_3d_chat.evaluation.baseline_io import read_jsonl
from semantic_3d_chat.evaluation.v56_fresh_development_score import (
    canonical_answer_key,
    canonical_type_specific_match,
)
from semantic_3d_chat.evaluation.v85_strict_multiscene_preflight import (
    sha256_file_v85,
)
from semantic_3d_chat.evaluation.v95_deferred_final_qa import FINAL_QA, PAIR_SCENES
from semantic_3d_chat.evaluation.v96_deferred_final_common import (
    ARMS,
    PRIMARY,
    SCHEMA_VERSION,
    STRUCTURED_SCORE_ARTIFACT,
    assert_aggregate_only_v96_final,
    audit_report_v96_final,
    authenticate_prediction_bundle_v96_final,
    score_forbidden_roots_v96_final,
    write_json_create_once_v96,
)
from semantic_3d_chat.evaluation.v96_deferred_final_evaluation import (
    ANSWER_TYPE_TOTALS,
    CHANGED_SIDE_COUNT,
    CHANGED_UNIT_COUNT,
    PAIR_SCENE,
    QUESTION_COUNT,
    SCENE_IDS,
    hardened_deferred_evaluation_stage_v96,
    output_paths_v96_final,
)
from semantic_3d_chat.evaluation.v96_known_development_common_v2 import (
    read_json_strict_v96,
)


def load_references_v96_final(
    source_qa_sha256: str,
    questions: Any,
) -> list[dict[str, Any]]:
    """Open exactly one label file after both prediction bundles authenticate."""

    path = FINAL_QA.resolve()
    if (
        path.is_symlink()
        or not path.is_file()
        or sha256_file_v85(path) != source_qa_sha256
        or questions.source_qa_sha256 != source_qa_sha256
    ):
        raise ValueError("V96 deferred-final label bytes changed")
    rows = read_jsonl(path)
    question_by_key = {(row.scene_id, row.question_id): row.question for row in questions.questions}
    seen: set[tuple[str, str]] = set()
    types: Counter[str] = Counter()
    changed_rows = 0
    grouped: defaultdict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    required_pair_fields = (
        "counterfactual_pair_id",
        "counterfactual_paired_scene_id",
        "counterfactual_question_key",
        "counterfactual_change_type",
    )
    for row in rows:
        key = (str(row.get("scene_id")), str(row.get("question_id")))
        changed = row.get("counterfactual_expected_change")
        if (
            key in seen
            or key not in question_by_key
            or row.get("question") != question_by_key[key]
            or not isinstance(row.get("answer"), str)
            or not isinstance(row.get("answer_type"), str)
            or type(changed) is not bool
            or any(
                not isinstance(row.get(field), str) or not row[field]
                for field in required_pair_fields
            )
            or row.get("counterfactual_paired_scene_id") != PAIR_SCENE.get(key[0])
        ):
            raise ValueError(f"V96 deferred-final reference projection changed: {key}")
        seen.add(key)
        types[str(row["answer_type"])] += 1
        changed_rows += int(changed)
        grouped[
            (
                str(row["counterfactual_pair_id"]),
                str(row["counterfactual_question_key"]),
            )
        ].append(row)
    if (
        len(rows) != QUESTION_COUNT
        or seen != set(question_by_key)
        or dict(sorted(types.items())) != ANSWER_TYPE_TOTALS
        or changed_rows != CHANGED_SIDE_COUNT
        or len(grouped) != QUESTION_COUNT // 2
        or any(len(sides) != 2 for sides in grouped.values())
    ):
        raise ValueError("V96 deferred-final reference coverage changed")
    return rows


def _accuracy(
    references: Sequence[Mapping[str, Any]],
    predictions: Mapping[tuple[str, str], Mapping[str, Any]],
    field: str,
) -> dict[str, Any]:
    values: list[bool] = []
    by_type: defaultdict[str, list[bool]] = defaultdict(list)
    for reference in references:
        key = (str(reference["scene_id"]), str(reference["question_id"]))
        correct = canonical_type_specific_match(
            str(reference["answer_type"]),
            predictions[key][field],
            reference["answer"],
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


def _stable_invariant(
    references: Sequence[Mapping[str, Any]],
    predictions: Mapping[tuple[str, str], Mapping[str, Any]],
) -> dict[str, Any]:
    grouped: defaultdict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in references:
        if row["counterfactual_expected_change"] is False:
            grouped[
                (
                    str(row["counterfactual_pair_id"]),
                    str(row["counterfactual_question_key"]),
                )
            ].append(row)
    expected_units = QUESTION_COUNT // 2 - CHANGED_UNIT_COUNT
    if len(grouped) != expected_units or any(len(sides) != 2 for sides in grouped.values()):
        raise ValueError("V96 deferred-final invariant-unit inventory changed")
    false_change_sides = 0
    for unit, sides in grouped.items():
        left, right = sides
        answer_type = str(left["answer_type"])
        if PAIR_SCENE[str(left["scene_id"])] != str(right["scene_id"]) or canonical_answer_key(
            answer_type, left["answer"]
        ) != canonical_answer_key(answer_type, right["answer"]):
            raise ValueError(f"V96 deferred invariant unit changed: {unit}")
        left_key = (str(left["scene_id"]), str(left["question_id"]))
        right_key = (str(right["scene_id"]), str(right["question_id"]))
        changed = canonical_answer_key(
            answer_type, predictions[left_key][f"{PRIMARY}_prediction"]
        ) != canonical_answer_key(answer_type, predictions[right_key][f"{PRIMARY}_prediction"])
        false_change_sides += 2 * int(changed)
    side_count = QUESTION_COUNT - CHANGED_SIDE_COUNT
    return {
        "side_count": side_count,
        "unit_count": expected_units,
        "invariant_false_change_count": false_change_sides,
        "invariant_false_change_rate": false_change_sides / side_count,
    }


def _changed_metrics_v96_final(
    references: Sequence[Mapping[str, Any]],
    predictions: Mapping[tuple[str, str], Mapping[str, Any]],
) -> dict[str, Any]:
    """Score the exact deferred scene pairs without V56's hard-coded IDs."""

    grouped: defaultdict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for reference in references:
        if reference["counterfactual_expected_change"] is True:
            grouped[
                (
                    str(reference["counterfactual_pair_id"]),
                    str(reference["counterfactual_question_key"]),
                )
            ].append(reference)
    if len(grouped) != CHANGED_UNIT_COUNT:
        raise ValueError("V96 deferred final requires exactly twelve changed units")

    by_pair = {pair_id: Counter() for pair_id in PAIR_SCENES}
    correct_sides = 0
    complete_units = 0
    prediction_changed_units = 0
    for (pair_id, question_key), sides in sorted(grouped.items()):
        if pair_id not in PAIR_SCENES or len(sides) != 2:
            raise ValueError(f"V96 deferred changed unit is invalid: {pair_id}/{question_key}")
        ordered = sorted(sides, key=lambda item: str(item["scene_id"]))
        if tuple(str(item["scene_id"]) for item in ordered) != PAIR_SCENES[pair_id]:
            raise ValueError(f"V96 deferred changed unit uses wrong scenes: {pair_id}")
        answer_types = {str(item["answer_type"]) for item in ordered}
        if len(answer_types) != 1:
            raise ValueError("V96 deferred changed-unit answer types disagree")
        answer_type = next(iter(answer_types))
        expected = [canonical_answer_key(answer_type, item["answer"]) for item in ordered]
        if any(value is None for value in expected) or expected[0] == expected[1]:
            raise ValueError("V96 deferred changed-unit labels did not change")
        observed: list[object | None] = []
        correctness: list[bool] = []
        for item in ordered:
            key = (str(item["scene_id"]), str(item["question_id"]))
            prediction = predictions.get(key)
            predicted = None if prediction is None else prediction.get("predicted_answer")
            observed.append(canonical_answer_key(answer_type, predicted))
            correctness.append(
                prediction is not None
                and canonical_type_specific_match(answer_type, predicted, item["answer"])
            )
        side_correct = sum(correctness)
        complete = int(all(correctness))
        changed = int(all(value is not None for value in observed) and observed[0] != observed[1])
        correct_sides += side_correct
        complete_units += complete
        prediction_changed_units += changed
        by_pair[pair_id]["unit_count"] += 1
        by_pair[pair_id]["correct_sides"] += side_correct
        by_pair[pair_id]["complete_units"] += complete
        by_pair[pair_id]["prediction_changed_units"] += changed
    if any(values["unit_count"] != 4 for values in by_pair.values()):
        raise ValueError("V96 deferred final requires four changed units per pair")
    return {
        "atomic_scene_pair_count": len(PAIR_SCENES),
        "unit_count": CHANGED_UNIT_COUNT,
        "side_count": CHANGED_SIDE_COUNT,
        "canonical_correct_sides": correct_sides,
        "canonical_complete_units": complete_units,
        "canonical_prediction_changed_units": prediction_changed_units,
        "physical_change_family_count": len(PAIR_SCENES),
        "physical_change_families_with_complete_unit": sum(
            values["complete_units"] >= 1 for values in by_pair.values()
        ),
        "by_family": {pair_id: dict(values) for pair_id, values in sorted(by_pair.items())},
    }


def structured_metrics_v96_final(
    references: Sequence[Mapping[str, Any]],
    v96_rows: Sequence[Mapping[str, Any]],
    v94_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    v96 = {(str(row["scene_id"]), str(row["question_id"])): row for row in v96_rows}
    v94 = {(str(row["scene_id"]), str(row["question_id"])): row for row in v94_rows}
    keys = {(str(row["scene_id"]), str(row["question_id"])) for row in references}
    if len(keys) != QUESTION_COUNT or set(v96) != keys or set(v94) != keys:
        raise ValueError("V96 and V94 final rows are not the exact same 216 keys")
    arms = {arm: _accuracy(references, v96, f"{arm}_prediction") for arm in ARMS}
    v94_same_rows = _accuracy(references, v94, "prediction")
    changed = _changed_metrics_v96_final(
        references,
        {key: {"predicted_answer": row[f"{PRIMARY}_prediction"]} for key, row in v96.items()},
    )
    if (
        changed.get("unit_count") != CHANGED_UNIT_COUNT
        or changed.get("side_count") != CHANGED_SIDE_COUNT
    ):
        raise ValueError("V96 deferred counterfactual inventory changed")
    comparisons: dict[str, dict[str, Any]] = {}
    for arm in ARMS[1:]:
        changes = sum(
            canonical_answer_key(str(reference["answer_type"]), v96[key][f"{PRIMARY}_prediction"])
            != canonical_answer_key(str(reference["answer_type"]), v96[key][f"{arm}_prediction"])
            for reference in references
            for key in [(str(reference["scene_id"]), str(reference["question_id"]))]
        )
        comparisons[arm] = {
            "accuracy_drop_from_primary": arms[PRIMARY]["accuracy"] - arms[arm]["accuracy"],
            "prediction_change_count": changes,
            "prediction_change_rate": changes / QUESTION_COUNT,
        }
    metrics = {
        "arms": arms,
        "v94_same_rows": v94_same_rows,
        "v96_accuracy_margin_over_v94_same_rows": (
            arms[PRIMARY]["accuracy"] - v94_same_rows["accuracy"]
        ),
        "counterfactual": changed,
        "comparisons": comparisons,
        "stable_invariant": _stable_invariant(references, v96),
    }
    assert_aggregate_only_v96_final(metrics)
    return metrics


@hardened_deferred_evaluation_stage_v96(label_process=True)
def score_deferred_final_v96() -> dict[str, Any]:
    """Authenticate both label-blind bundles, then open labels once."""

    v96 = authenticate_prediction_bundle_v96_final("v96")
    v94 = authenticate_prediction_bundle_v96_final("v94")
    if (
        v96["questions"].manifest_sha256 != v94["questions"].manifest_sha256
        or v96["questions"].questions_sha256 != v94["questions"].questions_sha256
        or v96["fixed"].memory_inventory_sha256 != v94["fixed"].memory_inventory_sha256
        or v96["provenance"]["source_qa_sha256"] != v94["provenance"]["source_qa_sha256"]
    ):
        raise ValueError("V94 comparator did not use V96's exact rows and memories")
    paths = output_paths_v96_final()
    outputs = (paths["structured_score"], paths["structured_access"])
    if any(path.exists() or path.is_symlink() for path in outputs):
        raise FileExistsError("V96 deferred structured score is create-once")
    audit = FileAccessAudit(
        score_forbidden_roots_v96_final(),
        forbidden_component_names={"oracle"},
        block_forbidden=True,
    )
    with audit:
        references = load_references_v96_final(v96["questions"].source_qa_sha256, v96["questions"])
        metrics = structured_metrics_v96_final(references, v96["rows"], v94["rows"])
    audit.assert_clean()
    access = audit_report_v96_final(audit)
    # The generic ordering flag applies only to predictor audits.
    access["all_six_memory_tensors_opened_before_question_manifest"] = True
    access["passed"] = not access["forbidden_accesses"]
    report = {
        "artifact": STRUCTURED_SCORE_ARTIFACT,
        "schema_version": SCHEMA_VERSION,
        "status": "measured_aggregate_only_not_yet_gated",
        "candidate_fingerprint_sha256": v96["fixed"].candidate["fingerprint_sha256"],
        "candidate_attestation_file_sha256": v96["fixed"].candidate[
            "attestation_file_sha256"
        ],
        "candidate_attestation_identity_sha256": v96["fixed"].candidate[
            "attestation_identity_sha256"
        ],
        "memory_inventory_sha256": v96["fixed"].memory_inventory_sha256,
        "question_manifest_sha256": v96["questions"].manifest_sha256,
        "questions_sha256": v96["questions"].questions_sha256,
        "reference_sha256": v96["questions"].source_qa_sha256,
        "v96_prediction_sha256": v96["prediction_sha256"],
        "v94_prediction_sha256": v94["prediction_sha256"],
        "row_count": QUESTION_COUNT,
        "scene_count": len(SCENE_IDS),
        "both_prediction_bundles_authenticated_before_labels_opened": True,
        "labels_opened_only_by_separate_scorer": True,
        "scorer_loaded_model": False,
        "row_level_content_serialized": False,
        "metrics": metrics,
        "runtime_packaging_requires_separate_leakage_gate": True,
        "runtime_promotion_authorized": False,
    }
    assert_aggregate_only_v96_final(report)
    write_json_create_once_v96(paths["structured_score"], report)
    write_json_create_once_v96(paths["structured_access"], access)
    return authenticate_structured_score_v96_final()


def authenticate_structured_score_v96_final() -> dict[str, Any]:
    v96 = authenticate_prediction_bundle_v96_final("v96")
    v94 = authenticate_prediction_bundle_v96_final("v94")
    paths = output_paths_v96_final()
    report = read_json_strict_v96(paths["structured_score"])
    access = read_json_strict_v96(paths["structured_access"])
    metrics = report.get("metrics")
    if not isinstance(metrics, Mapping):
        raise TypeError("V96 deferred structured metrics are missing")
    assert_aggregate_only_v96_final(metrics)
    if (
        report.get("artifact") != STRUCTURED_SCORE_ARTIFACT
        or report.get("schema_version") != SCHEMA_VERSION
        or report.get("candidate_fingerprint_sha256")
        != v96["fixed"].candidate["fingerprint_sha256"]
        or report.get("candidate_attestation_file_sha256")
        != v96["fixed"].candidate["attestation_file_sha256"]
        or report.get("candidate_attestation_identity_sha256")
        != v96["fixed"].candidate["attestation_identity_sha256"]
        or report.get("memory_inventory_sha256") != v96["fixed"].memory_inventory_sha256
        or report.get("question_manifest_sha256") != v96["questions"].manifest_sha256
        or report.get("v96_prediction_sha256") != v96["prediction_sha256"]
        or report.get("v94_prediction_sha256") != v94["prediction_sha256"]
        or report.get("row_count") != QUESTION_COUNT
        or report.get("both_prediction_bundles_authenticated_before_labels_opened") is not True
        or report.get("row_level_content_serialized") is not False
        or report.get("scorer_loaded_model") is not False
        or report.get("runtime_promotion_authorized") is not False
        or access.get("passed") is not True
        or access.get("protected_read_count") != 0
        or str(FINAL_QA.resolve()) not in set(access.get("loaded_files", []))
    ):
        raise ValueError("V96 deferred structured score authentication failed")
    assert_aggregate_only_v96_final(report)
    return {
        "report": report,
        "access": access,
        "sha256": sha256_file_v85(paths["structured_score"]),
        "access_sha256": sha256_file_v85(paths["structured_access"]),
        "v96": v96,
        "v94": v94,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("score", "authenticate"), nargs="?", default="score")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = (
        score_deferred_final_v96()
        if args.command == "score"
        else authenticate_structured_score_v96_final()
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "authenticate_structured_score_v96_final",
    "load_references_v96_final",
    "main",
    "score_deferred_final_v96",
    "structured_metrics_v96_final",
]
