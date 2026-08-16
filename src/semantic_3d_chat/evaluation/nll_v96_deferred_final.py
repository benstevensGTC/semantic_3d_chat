"""Aggregate-only four-arm NLL evaluator for V96's deferred final."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from types import SimpleNamespace
from typing import Any

import torch

from semantic_3d_chat.chat.file_audit import FileAccessAudit
from semantic_3d_chat.evaluation.predict_v96_known_development_v2 import (
    load_predictor_stack_v96,
)
from semantic_3d_chat.evaluation.score_v96_deferred_final import (
    load_references_v96_final,
)
from semantic_3d_chat.evaluation.v85_strict_multiscene_preflight import (
    sha256_file_v85,
)
from semantic_3d_chat.evaluation.v95_deferred_final_qa import FINAL_QA
from semantic_3d_chat.evaluation.v96_atomic_pair_repair_preflight import (
    CONFIG,
    load_config_v96,
)
from semantic_3d_chat.evaluation.v96_deferred_final_common import (
    ARMS,
    FULL_INTERIOR_PERMUTATION,
    NLL_ARTIFACT,
    NLL_COMPLETION_ARTIFACT,
    PAIRED_WRONG_SCENE,
    PRIMARY,
    SCHEMA_VERSION,
    ZERO_PAYLOAD,
    assert_aggregate_only_v96_final,
    audit_report_v96_final,
    authenticate_prediction_bundle_v96_final,
    finite_number,
    score_forbidden_roots_v96_final,
    write_json_create_once_v96,
)
from semantic_3d_chat.evaluation.v96_deferred_final_evaluation import (
    CHANGED_SIDE_COUNT,
    QUESTION_COUNT,
    hardened_deferred_evaluation_stage_v96,
    output_paths_v96_final,
)
from semantic_3d_chat.evaluation.v96_known_development_common_v2 import (
    assert_output_bundle_state_v96,
    authenticate_fixed_final_candidate_v96,
    read_json_strict_v96,
)
from semantic_3d_chat.language.prefix_injection import prefix_sha256
from semantic_3d_chat.training.train_v84_strict_bridge import _measure_nll_v84


def _mean(values: Sequence[float], label: str) -> float:
    if not values:
        raise ValueError(f"V96 deferred NLL aggregate is empty: {label}")
    return sum(values) / len(values)


def validate_nll_metrics_v96_final(metrics: Mapping[str, Any]) -> None:
    expected = {
        "primary_mean_nll",
        "paired_wrong_scene_mean_nll",
        "zero_payload_mean_nll",
        "full_interior_permutation_mean_nll",
        "mean_wrong_minus_primary_nll",
        "mean_changed_wrong_minus_primary_nll",
        "zero_payload_mean_nll_gap",
        "full_interior_permutation_mean_nll_gap",
        "row_count_per_arm",
        "changed_row_count",
    }
    if set(metrics) != expected:
        raise ValueError("V96 deferred NLL aggregate fields changed")
    if (
        metrics.get("row_count_per_arm") != QUESTION_COUNT
        or metrics.get("changed_row_count") != CHANGED_SIDE_COUNT
    ):
        raise ValueError("V96 deferred NLL row counts changed")
    for key in expected - {"row_count_per_arm", "changed_row_count"}:
        finite_number(metrics.get(key), key)
    assert_aggregate_only_v96_final(metrics)


@hardened_deferred_evaluation_stage_v96(label_process=True)
@torch.inference_mode()
def measure_nll_v96_final() -> dict[str, Any]:
    """Authenticate both prediction bundles, then open labels and load Gemma."""

    v96 = authenticate_prediction_bundle_v96_final("v96")
    v94 = authenticate_prediction_bundle_v96_final("v94")
    if (
        v96["questions"].manifest_sha256 != v94["questions"].manifest_sha256
        or v96["fixed"].memory_inventory_sha256 != v94["fixed"].memory_inventory_sha256
    ):
        raise ValueError("V96 NLL same-row comparator bindings disagree")
    paths = output_paths_v96_final()
    outputs = (paths["nll"], paths["nll_access"], paths["nll_completion"])
    states = [path.exists() or path.is_symlink() for path in outputs]
    if all(states):
        return authenticate_nll_v96_final()
    assert_output_bundle_state_v96(outputs, complete=False)

    audit = FileAccessAudit(
        score_forbidden_roots_v96_final(),
        forbidden_component_names={"oracle"},
        block_forbidden=True,
    )
    with audit:
        config = load_config_v96(CONFIG, allow_draft=False)
        fixed = v96["fixed"]
        stack = load_predictor_stack_v96(
            config,
            expected_candidate_state_sha256=str(fixed.candidate["state_sha256"]),
        )
        references = load_references_v96_final(v96["questions"].source_qa_sha256, v96["questions"])
        totals: defaultdict[str, list[float]] = defaultdict(list)
        for ordinal, reference in enumerate(references, 1):
            scene = str(reference["scene_id"])
            row = SimpleNamespace(
                question=str(reference["question"]), answer=str(reference["answer"])
            )
            arm_nll: dict[str, float] = {}
            for arm in ARMS:
                measured, _layout = _measure_nll_v84(
                    stack.language,
                    stack.system_prompt,
                    fixed.memories[arm][scene],
                    row,
                )
                arm_nll[arm] = float(measured["mean_nll"])
            primary = arm_nll[PRIMARY]
            for arm in ARMS:
                totals[arm].append(arm_nll[arm])
            totals["wrong_gap"].append(arm_nll[PAIRED_WRONG_SCENE] - primary)
            totals["zero_gap"].append(arm_nll[ZERO_PAYLOAD] - primary)
            totals["permutation_gap"].append(arm_nll[FULL_INTERIOR_PERMUTATION] - primary)
            if reference["counterfactual_expected_change"] is True:
                totals["changed_wrong_gap"].append(arm_nll[PAIRED_WRONG_SCENE] - primary)
            if ordinal == 1 or ordinal % 12 == 0 or ordinal == QUESTION_COUNT:
                print(
                    json.dumps(
                        {
                            "event": "v96_deferred_final_nll",
                            "ordinal": ordinal,
                            "total": QUESTION_COUNT,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        after_hashes = {
            arm: {scene: prefix_sha256(fixed.memories[arm][scene]) for scene in fixed.memories[arm]}
            for arm in ARMS
        }
        candidate_after = authenticate_fixed_final_candidate_v96(
            config, config_path=CONFIG, audit=audit
        )
    audit.assert_clean()
    if candidate_after != fixed.candidate or after_hashes != fixed.memory_hashes:
        raise RuntimeError("V96 deferred NLL mutated candidate or memory")
    metrics = {
        "primary_mean_nll": _mean(totals[PRIMARY], PRIMARY),
        "paired_wrong_scene_mean_nll": _mean(totals[PAIRED_WRONG_SCENE], PAIRED_WRONG_SCENE),
        "zero_payload_mean_nll": _mean(totals[ZERO_PAYLOAD], ZERO_PAYLOAD),
        "full_interior_permutation_mean_nll": _mean(
            totals[FULL_INTERIOR_PERMUTATION], FULL_INTERIOR_PERMUTATION
        ),
        "mean_wrong_minus_primary_nll": _mean(totals["wrong_gap"], "wrong gap"),
        "mean_changed_wrong_minus_primary_nll": _mean(
            totals["changed_wrong_gap"], "changed wrong gap"
        ),
        "zero_payload_mean_nll_gap": _mean(totals["zero_gap"], "zero gap"),
        "full_interior_permutation_mean_nll_gap": _mean(
            totals["permutation_gap"], "permutation gap"
        ),
        "row_count_per_arm": len(totals[PRIMARY]),
        "changed_row_count": len(totals["changed_wrong_gap"]),
    }
    validate_nll_metrics_v96_final(metrics)
    report = {
        "artifact": NLL_ARTIFACT,
        "schema_version": SCHEMA_VERSION,
        "status": "measured_aggregate_only_not_yet_gated",
        "candidate_fingerprint_sha256": fixed.candidate["fingerprint_sha256"],
        "candidate_attestation_file_sha256": fixed.candidate[
            "attestation_file_sha256"
        ],
        "candidate_attestation_identity_sha256": fixed.candidate[
            "attestation_identity_sha256"
        ],
        "memory_inventory_sha256": fixed.memory_inventory_sha256,
        "question_manifest_sha256": v96["questions"].manifest_sha256,
        "questions_sha256": v96["questions"].questions_sha256,
        "reference_sha256": v96["questions"].source_qa_sha256,
        "v96_prediction_sha256": v96["prediction_sha256"],
        "v94_prediction_sha256": v94["prediction_sha256"],
        "row_count": QUESTION_COUNT,
        "scene_count": 6,
        "arms": list(ARMS),
        "both_prediction_bundles_authenticated_before_labels_opened": True,
        "labels_opened_only_by_separate_nll_evaluator": True,
        "row_level_content_serialized": False,
        "metrics": metrics,
        "runtime_packaging_requires_separate_leakage_gate": True,
        "runtime_promotion_authorized": False,
    }
    assert_aggregate_only_v96_final(report)
    access = audit_report_v96_final(audit)
    access["all_six_memory_tensors_opened_before_question_manifest"] = True
    access["passed"] = not access["forbidden_accesses"]
    if str(FINAL_QA.resolve()) not in access["loaded_files"]:
        raise RuntimeError("V96 deferred NLL did not audit its sole label read")
    write_json_create_once_v96(paths["nll"], report)
    write_json_create_once_v96(paths["nll_access"], access)
    completion = {
        "artifact": NLL_COMPLETION_ARTIFACT,
        "schema_version": SCHEMA_VERSION,
        "candidate_fingerprint_before": fixed.candidate["fingerprint_sha256"],
        "candidate_fingerprint_after": candidate_after["fingerprint_sha256"],
        "candidate_attestation_file_sha256": fixed.candidate[
            "attestation_file_sha256"
        ],
        "candidate_attestation_identity_sha256": fixed.candidate[
            "attestation_identity_sha256"
        ],
        "v2_implementation_seal_sha256": fixed.candidate[
            "v2_implementation_seal_sha256"
        ],
        "candidate_immutable": True,
        "memory_hashes_invariant": True,
        "nll_sha256": sha256_file_v85(paths["nll"]),
        "nll_access_sha256": sha256_file_v85(paths["nll_access"]),
        "row_count_per_arm": QUESTION_COUNT,
        "changed_row_count": CHANGED_SIDE_COUNT,
        "row_level_content_serialized": False,
        "runtime_promotion_authorized": False,
    }
    assert_aggregate_only_v96_final(completion)
    write_json_create_once_v96(paths["nll_completion"], completion)
    return authenticate_nll_v96_final()


def authenticate_nll_v96_final() -> dict[str, Any]:
    v96 = authenticate_prediction_bundle_v96_final("v96")
    v94 = authenticate_prediction_bundle_v96_final("v94")
    paths = output_paths_v96_final()
    assert_output_bundle_state_v96(
        (paths["nll"], paths["nll_access"], paths["nll_completion"]), complete=True
    )
    report = read_json_strict_v96(paths["nll"])
    access = read_json_strict_v96(paths["nll_access"])
    completion = read_json_strict_v96(paths["nll_completion"])
    metrics = report.get("metrics")
    if not isinstance(metrics, Mapping):
        raise TypeError("V96 deferred NLL metrics are missing")
    validate_nll_metrics_v96_final(metrics)
    if (
        report.get("artifact") != NLL_ARTIFACT
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
        or report.get("runtime_promotion_authorized") is not False
        or access.get("passed") is not True
        or access.get("protected_read_count") != 0
        or str(FINAL_QA.resolve()) not in set(access.get("loaded_files", []))
        or completion.get("artifact") != NLL_COMPLETION_ARTIFACT
        or completion.get("candidate_fingerprint_before")
        != v96["fixed"].candidate["fingerprint_sha256"]
        or completion.get("candidate_fingerprint_after")
        != v96["fixed"].candidate["fingerprint_sha256"]
        or completion.get("candidate_attestation_file_sha256")
        != v96["fixed"].candidate["attestation_file_sha256"]
        or completion.get("candidate_attestation_identity_sha256")
        != v96["fixed"].candidate["attestation_identity_sha256"]
        or completion.get("v2_implementation_seal_sha256")
        != v96["fixed"].candidate["v2_implementation_seal_sha256"]
        or completion.get("memory_hashes_invariant") is not True
        or completion.get("nll_sha256") != sha256_file_v85(paths["nll"])
        or completion.get("nll_access_sha256") != sha256_file_v85(paths["nll_access"])
    ):
        raise ValueError("V96 deferred NLL authentication failed")
    assert_aggregate_only_v96_final(report)
    return {
        "report": report,
        "access": access,
        "completion": completion,
        "sha256": sha256_file_v85(paths["nll"]),
        "access_sha256": sha256_file_v85(paths["nll_access"]),
        "completion_sha256": sha256_file_v85(paths["nll_completion"]),
        "v96": v96,
        "v94": v94,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("measure", "authenticate"), nargs="?", default="measure"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = measure_nll_v96_final() if args.command == "measure" else authenticate_nll_v96_final()
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "authenticate_nll_v96_final",
    "main",
    "measure_nll_v96_final",
    "validate_nll_metrics_v96_final",
]
