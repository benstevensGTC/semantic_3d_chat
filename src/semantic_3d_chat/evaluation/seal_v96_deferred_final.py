"""Apply V95's sealed thresholds to V96 deferred-final aggregate evidence.

This module is model-free and label-free.  It authenticates the two prediction
bundles and the already-produced aggregate structured/NLL reports, recomputes
every pre-registered gate, and writes create-once final evidence.  Passing this
gate never promotes a runtime; the independent leakage/runtime gate remains a
separate required step.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

from semantic_3d_chat.evaluation.nll_v96_deferred_final import (
    authenticate_nll_v96_final,
    validate_nll_metrics_v96_final,
)
from semantic_3d_chat.evaluation.score_v96_deferred_final import (
    authenticate_structured_score_v96_final,
)
from semantic_3d_chat.evaluation.v85_strict_multiscene_preflight import (
    sha256_file_v85,
)
from semantic_3d_chat.evaluation.v96_deferred_final_common import (
    EVIDENCE_ARTIFACT,
    FINAL_SCORE_ARTIFACT,
    PRIMARY,
    SCHEMA_VERSION,
    assert_aggregate_only_v96_final,
    finite_number,
    write_json_create_once_v96,
)
from semantic_3d_chat.evaluation.v96_deferred_final_evaluation import (
    ANSWER_TYPE_TOTALS,
    CHANGED_SIDE_COUNT,
    CHANGED_UNIT_COUNT,
    FINAL_GATE_CONTRACT,
    QUESTION_COUNT,
    authenticate_preregistration_v96_final,
    output_paths_v96_final,
)
from semantic_3d_chat.evaluation.v96_known_development_common_v2 import (
    assert_output_bundle_state_v96,
    canonical_sha256_v96,
    read_json_strict_v96,
)


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"V96 deferred final {label} must be a mapping")
    return value


def _count_record(value: object, label: str, *, expected_total: int) -> Mapping[str, Any]:
    record = _mapping(value, label)
    correct = record.get("correct")
    total = record.get("total")
    accuracy = finite_number(record.get("accuracy"), f"{label} accuracy")
    if (
        isinstance(correct, bool)
        or not isinstance(correct, int)
        or total != expected_total
        or correct < 0
        or correct > expected_total
        or not math.isclose(accuracy, correct / expected_total, abs_tol=1e-12)
    ):
        raise ValueError(f"V96 deferred final {label} count/accuracy changed")
    return record


def final_gate_results_v96_final(
    structured_metrics: Mapping[str, Any],
    nll_metrics: Mapping[str, Any],
    contract: Mapping[str, Any],
    *,
    fixed_candidate_immutable: bool,
    prefix_invariant: bool,
    label_isolation_proven: bool,
    protected_read_count: int,
    separate_leakage_gate_required: bool,
) -> dict[str, bool]:
    """Recompute every inclusive V95 final threshold without label rows."""

    if dict(contract) != FINAL_GATE_CONTRACT:
        raise ValueError("V96 deferred final gate contract differs from sealed V95")
    validate_nll_metrics_v96_final(nll_metrics)
    arms = _mapping(structured_metrics.get("arms"), "arms")
    primary = _count_record(arms.get(PRIMARY), "primary arm", expected_total=QUESTION_COUNT)
    v94 = _count_record(
        structured_metrics.get("v94_same_rows"),
        "V94 same-row comparator",
        expected_total=QUESTION_COUNT,
    )
    by_type = _mapping(primary.get("by_answer_type"), "answer types")
    if set(by_type) != set(ANSWER_TYPE_TOTALS):
        raise ValueError("V96 deferred final answer-type inventory changed")
    type_records = {
        answer_type: _count_record(
            by_type[answer_type],
            answer_type,
            expected_total=expected_total,
        )
        for answer_type, expected_total in ANSWER_TYPE_TOTALS.items()
    }
    changed = _mapping(structured_metrics.get("counterfactual"), "counterfactual")
    margin = finite_number(
        structured_metrics.get("v96_accuracy_margin_over_v94_same_rows"),
        "V94 margin",
    )
    if not math.isclose(
        margin,
        float(primary["accuracy"]) - float(v94["accuracy"]),
        abs_tol=1e-12,
    ):
        raise ValueError("V96 deferred final V94 margin is inconsistent")
    if (
        changed.get("unit_count") != CHANGED_UNIT_COUNT
        or changed.get("side_count") != CHANGED_SIDE_COUNT
    ):
        raise ValueError("V96 deferred final counterfactual scope changed")

    gates: dict[str, bool] = {
        "exact_216_row_same_row_scope": primary["total"] == v94["total"] == QUESTION_COUNT,
        "canonical_accuracy_minimum": float(primary["accuracy"])
        >= float(contract["canonical_accuracy_minimum"]),
        "canonical_accuracy_margin_over_fixed_v94_same_rows": margin
        >= float(contract["canonical_accuracy_margin_over_fixed_v94_same_rows"]),
    }
    for answer_type in ANSWER_TYPE_TOTALS:
        gates[f"{answer_type}_correct_minimum"] = int(type_records[answer_type]["correct"]) >= int(
            contract[f"{answer_type}_correct_minimum"]
        )
    gates.update(
        {
            "changed_side_correct_minimum": int(changed.get("canonical_correct_sides", -1))
            >= int(contract["changed_side_correct_minimum"]),
            "complete_changed_units_minimum": int(changed.get("canonical_complete_units", -1))
            >= int(contract["complete_changed_units_minimum"]),
            "canonical_prediction_changing_units_minimum": int(
                changed.get("canonical_prediction_changed_units", -1)
            )
            >= int(contract["canonical_prediction_changing_units_minimum"]),
            "mean_changed_side_wrong_minus_correct_nll_minimum": finite_number(
                nll_metrics.get("mean_changed_wrong_minus_primary_nll"),
                "changed wrong-primary NLL gap",
            )
            >= float(contract["mean_changed_side_wrong_minus_correct_nll_minimum"]),
            "zero_payload_mean_nll_gap_minimum": finite_number(
                nll_metrics.get("zero_payload_mean_nll_gap"),
                "zero-payload NLL gap",
            )
            >= float(contract["zero_payload_mean_nll_gap_minimum"]),
            "permutation_mean_nll_gap_minimum": finite_number(
                nll_metrics.get("full_interior_permutation_mean_nll_gap"),
                "permutation NLL gap",
            )
            >= float(contract["permutation_mean_nll_gap_minimum"]),
            "correct_scene_nll_below_zero_payload_required": (
                not bool(contract["correct_scene_nll_below_zero_payload_required"])
                or finite_number(nll_metrics.get("primary_mean_nll"), "primary mean NLL")
                < finite_number(
                    nll_metrics.get("zero_payload_mean_nll"),
                    "zero-payload mean NLL",
                )
            ),
            "correct_scene_nll_below_permuted_payload_required": (
                not bool(contract["correct_scene_nll_below_permuted_payload_required"])
                or finite_number(nll_metrics.get("primary_mean_nll"), "primary mean NLL")
                < finite_number(
                    nll_metrics.get("full_interior_permutation_mean_nll"),
                    "permutation mean NLL",
                )
            ),
            "exact_prefix_hash_invariance_required": (
                not bool(contract["exact_prefix_hash_invariance_required"]) or prefix_invariant
            ),
            "question_label_isolation_required": (
                not bool(contract["question_label_isolation_required"]) or label_isolation_proven
            ),
            "protected_read_count_maximum": protected_read_count
            <= int(contract["protected_read_count_maximum"]),
            "fixed_candidate_immutable": fixed_candidate_immutable,
            "runtime_packaging_requires_separate_leakage_gate": (
                bool(contract["runtime_packaging_requires_separate_leakage_gate"])
                and separate_leakage_gate_required
            ),
            "automatic_runtime_promotion_disabled": contract["automatic_runtime_promotion"]
            is False,
        }
    )
    if not gates or not all(type(value) is bool for value in gates.values()):
        raise AssertionError("V96 deferred final gate results must be booleans")
    return gates


def _build_final_report() -> tuple[dict[str, Any], dict[str, Any]]:
    preregistration = authenticate_preregistration_v96_final()
    structured = authenticate_structured_score_v96_final()
    nll = authenticate_nll_v96_final()
    structured_report = structured["report"]
    nll_report = nll["report"]
    structured_metrics = _mapping(structured_report.get("metrics"), "structured metrics")
    nll_metrics = _mapping(nll_report.get("metrics"), "NLL metrics")
    v96 = structured["v96"]
    v94 = structured["v94"]
    candidate_fingerprint = preregistration["candidate"]["fingerprint_sha256"]
    fixed_candidate_immutable = all(
        value == candidate_fingerprint
        for value in (
            v96["fixed"].candidate["fingerprint_sha256"],
            v94["fixed"].candidate["fingerprint_sha256"],
            structured_report["candidate_fingerprint_sha256"],
            nll_report["candidate_fingerprint_sha256"],
            nll["completion"]["candidate_fingerprint_before"],
            nll["completion"]["candidate_fingerprint_after"],
        )
    )
    prefix_invariant = all(
        item is True
        for item in (
            v96["access"]["all_six_memory_tensors_opened_before_question_manifest"],
            v94["access"]["all_six_memory_tensors_opened_before_question_manifest"],
            v96["completion"]["all_memory_hashes_invariant"],
            v94["completion"]["all_memory_hashes_invariant"],
            v96["completion"]["all_memories_bound_before_questions"],
            v94["completion"]["all_memories_bound_before_questions"],
        )
    )
    accesses = (
        v96["access"],
        v94["access"],
        structured["access"],
        nll["access"],
    )
    protected_read_count = sum(int(access["protected_read_count"]) for access in accesses)
    label_isolation = (
        all(access.get("passed") is True for access in accesses)
        and v96["provenance"].get("labels_loaded") is False
        and v94["provenance"].get("labels_loaded") is False
        and structured_report.get("both_prediction_bundles_authenticated_before_labels_opened")
        is True
        and nll_report.get("both_prediction_bundles_authenticated_before_labels_opened") is True
        and structured_report.get("labels_opened_only_by_separate_scorer") is True
        and structured_report.get("scorer_loaded_model") is False
        and nll_report.get("labels_opened_only_by_separate_nll_evaluator") is True
    )
    separate_leakage = (
        structured_report.get("runtime_packaging_requires_separate_leakage_gate") is True
        and nll_report.get("runtime_packaging_requires_separate_leakage_gate") is True
    )
    gates = final_gate_results_v96_final(
        structured_metrics,
        nll_metrics,
        preregistration["v95_gate_source"]["contract"],
        fixed_candidate_immutable=fixed_candidate_immutable,
        prefix_invariant=prefix_invariant,
        label_isolation_proven=label_isolation,
        protected_read_count=protected_read_count,
        separate_leakage_gate_required=separate_leakage,
    )
    report = {
        "artifact": FINAL_SCORE_ARTIFACT,
        "schema_version": SCHEMA_VERSION,
        "status": (
            "passed_deferred_final_not_runtime_promoted"
            if all(gates.values())
            else "failed_deferred_final_not_runtime_promoted"
        ),
        "passed": all(gates.values()),
        "candidate_fingerprint_sha256": candidate_fingerprint,
        "candidate_attestation_file_sha256": preregistration["candidate"][
            "attestation_file_sha256"
        ],
        "candidate_attestation_identity_sha256": preregistration["candidate"][
            "attestation_identity_sha256"
        ],
        "v2_implementation_seal_sha256": preregistration["candidate"][
            "v2_implementation_seal_sha256"
        ],
        "v1_implementation_seal_sha256": preregistration["known_development"][
            "v1_implementation_seal_sha256"
        ],
        "preregistration_file_sha256": preregistration["preregistration_file_sha256"],
        "preregistration_identity_sha256": preregistration["preregistration_identity_sha256"],
        "gate_contract_sha256": preregistration["v95_gate_source"]["contract_sha256"],
        "structured_score_sha256": structured["sha256"],
        "structured_access_sha256": structured["access_sha256"],
        "nll_sha256": nll["sha256"],
        "nll_access_sha256": nll["access_sha256"],
        "nll_completion_sha256": nll["completion_sha256"],
        "v96_prediction_sha256": v96["prediction_sha256"],
        "v94_same_row_prediction_sha256": v94["prediction_sha256"],
        "row_count": QUESTION_COUNT,
        "protected_read_count": protected_read_count,
        "fixed_candidate_immutable": fixed_candidate_immutable,
        "prefix_hash_invariant": prefix_invariant,
        "question_label_isolation_proven": label_isolation,
        "same_row_v94_comparator_authenticated": True,
        "gate_results": gates,
        "gate_results_sha256": canonical_sha256_v96(gates),
        "metrics": {
            "primary": structured_metrics["arms"][PRIMARY],
            "v94_same_rows": structured_metrics["v94_same_rows"],
            "v96_accuracy_margin_over_v94_same_rows": structured_metrics[
                "v96_accuracy_margin_over_v94_same_rows"
            ],
            "counterfactual": structured_metrics["counterfactual"],
            "nll": dict(nll_metrics),
        },
        "runtime_packaging_requires_separate_leakage_gate": True,
        "eligible_for_separate_runtime_leakage_evaluation": all(gates.values()),
        "runtime_promotion_authorized": False,
        "automatic_runtime_promotion": False,
    }
    assert_aggregate_only_v96_final(report)
    support = {
        "preregistration": preregistration,
        "structured": structured,
        "nll": nll,
    }
    return report, support


def _build_evidence(report: Mapping[str, Any]) -> dict[str, Any]:
    paths = output_paths_v96_final()
    evidence = {
        "artifact": EVIDENCE_ARTIFACT,
        "schema_version": SCHEMA_VERSION,
        "status": report["status"],
        "deferred_final_gate_passed": report["passed"],
        "candidate_fingerprint_sha256": report["candidate_fingerprint_sha256"],
        "candidate_attestation_file_sha256": report[
            "candidate_attestation_file_sha256"
        ],
        "candidate_attestation_identity_sha256": report[
            "candidate_attestation_identity_sha256"
        ],
        "v2_implementation_seal_sha256": report[
            "v2_implementation_seal_sha256"
        ],
        "v1_implementation_seal_sha256": report[
            "v1_implementation_seal_sha256"
        ],
        "final_score_sha256": sha256_file_v85(paths["final_score"]),
        "preregistration_file_sha256": report["preregistration_file_sha256"],
        "preregistration_identity_sha256": report["preregistration_identity_sha256"],
        "implementation_source_inventory_sha256": authenticate_preregistration_v96_final()[
            "implementation_source_inventory_sha256"
        ],
        "gate_contract_sha256": report["gate_contract_sha256"],
        "gate_results_sha256": report["gate_results_sha256"],
        "structured_score_sha256": report["structured_score_sha256"],
        "nll_sha256": report["nll_sha256"],
        "v96_prediction_sha256": report["v96_prediction_sha256"],
        "v94_same_row_prediction_sha256": report["v94_same_row_prediction_sha256"],
        "question_label_isolation_proven": report["question_label_isolation_proven"],
        "prefix_hash_invariant": report["prefix_hash_invariant"],
        "protected_read_count": report["protected_read_count"],
        "row_level_content_serialized": False,
        "runtime_packaging_requires_separate_leakage_gate": True,
        "runtime_promotion_authorized": False,
        "automatic_runtime_promotion": False,
    }
    evidence["evidence_identity_sha256"] = canonical_sha256_v96(evidence)
    assert_aggregate_only_v96_final(evidence)
    return evidence


def seal_deferred_final_v96() -> dict[str, Any]:
    paths = output_paths_v96_final()
    outputs = (paths["final_score"], paths["evidence"])
    states = [path.exists() or path.is_symlink() for path in outputs]
    if all(states):
        return authenticate_deferred_final_evidence_v96()
    assert_output_bundle_state_v96(outputs, complete=False)
    report, _support = _build_final_report()
    write_json_create_once_v96(paths["final_score"], report)
    write_json_create_once_v96(paths["evidence"], _build_evidence(report))
    return authenticate_deferred_final_evidence_v96()


def authenticate_deferred_final_evidence_v96() -> dict[str, Any]:
    paths = output_paths_v96_final()
    assert_output_bundle_state_v96((paths["final_score"], paths["evidence"]), complete=True)
    report = read_json_strict_v96(paths["final_score"])
    evidence = read_json_strict_v96(paths["evidence"])
    expected_report, _support = _build_final_report()
    if report != expected_report:
        raise ValueError("V96 deferred final score changed after sealing")
    expected_evidence = _build_evidence(report)
    if evidence != expected_evidence:
        raise ValueError("V96 deferred final evidence changed after sealing")
    return {
        **evidence,
        "authenticated": True,
        "evidence_file_sha256": sha256_file_v85(paths["evidence"]),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("seal", "authenticate"), nargs="?", default="seal")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = (
        seal_deferred_final_v96()
        if args.command == "seal"
        else authenticate_deferred_final_evidence_v96()
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "authenticate_deferred_final_evidence_v96",
    "final_gate_results_v96_final",
    "main",
    "seal_deferred_final_v96",
]
