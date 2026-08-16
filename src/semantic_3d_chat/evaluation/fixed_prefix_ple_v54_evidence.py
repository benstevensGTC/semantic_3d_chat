"""Authenticate the terminal V1--V5 fixed-prefix PLE-reader evidence chain.

This module is deliberately model-free and source-snapshot independent.  It reads
only immutable, explicitly allowlisted reports, verifies their byte digests and
cross-links, and confirms that no rejected reader checkpoint exists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Final

from semantic_3d_chat.config import PROJECT_ROOT

EVIDENCE_SHA256: Final[dict[Path, str]] = {
    Path("reports/gemma4/metrics/gemma4_v54_fixed_prefix_ple_reader_v1_preregistration.json"): (
        "07c28a95badf87c08692532ed1b8f9064af37763f11bc8c469581dae147bff52"
    ),
    Path("reports/gemma4/metrics/gemma4_v54_fixed_prefix_ple_reader_v1_smoke.json"): (
        "c1c8b6efe101fd1ce78d02fea9bafb1a090f3356171122a397bbd42cae7dcfa5"
    ),
    Path("reports/gemma4/metrics/gemma4_v54_fixed_prefix_ple_reader_v2_preregistration.json"): (
        "b82163a0e3fe030f84403e10822944a52a8c3c99ef215d7a67a90dcb88d6d8fd"
    ),
    Path("reports/gemma4/metrics/gemma4_v54_fixed_prefix_ple_reader_v2_smoke.json"): (
        "f7daaf2df2f052d0dad45fdf5cacff3c21652c99b953adc12f0361072ba189f0"
    ),
    Path("reports/gemma4/metrics/gemma4_v54_fixed_prefix_ple_reader_v2_abort.json"): (
        "97ec2de77484fcf5b014478c8701ed20b5ab8b0e7f394cdd2158a452c2005210"
    ),
    Path("reports/gemma4/metrics/gemma4_v54_fixed_prefix_ple_reader_v3_preregistration.json"): (
        "eff55d288be9bb6337e2a9d9a086359aba7c9c181b105d7188dfa6dbefcea614"
    ),
    Path("reports/gemma4/metrics/gemma4_v54_fixed_prefix_ple_reader_v3_smoke.json"): (
        "3f2fbb71e7fa69491d606b19984d5207ba5642945bb26e4876b94ead350d12e9"
    ),
    Path("reports/gemma4/metrics/gemma4_v54_fixed_prefix_ple_reader_v3_abort.json"): (
        "12461df8dffa9304646a97c33dcad1855496fa65cf9d9d0050663362fc01dcdd"
    ),
    Path("reports/gemma4/metrics/gemma4_v54_fixed_prefix_ple_reader_v4_preregistration.json"): (
        "34b4576a6ced7003c916c5dc3deabecf8e6e70a0e39bcc8329d039fd00ef3d59"
    ),
    Path("reports/gemma4/metrics/gemma4_v54_fixed_prefix_ple_reader_v4_smoke.json"): (
        "4d76f2f6de14fd5d1e5130d50fded5627418cec83fb5f505d57db49f9244d345"
    ),
    Path("reports/gemma4/metrics/gemma4_v54_fixed_prefix_ple_reader_v4_result.json"): (
        "ea16351a39ba1e0eb7441a4c8f371466b2f413ed6d352bdfb745e1f047766139"
    ),
    Path("reports/gemma4/metrics/gemma4_v54_fixed_prefix_ple_reader_v5_preregistration.json"): (
        "7503de97af2d39282ccac3b91566f18bdebd718d81ace56f0cf065bff28db3e6"
    ),
    Path("reports/gemma4/metrics/gemma4_v54_fixed_prefix_ple_reader_v5_smoke.json"): (
        "445c58339b4787d6c30c21b92a976da8bf7bcc1958f2aac4f3b9f8db67371523"
    ),
    Path("reports/gemma4/metrics/gemma4_v54_fixed_prefix_ple_reader_v5_result.json"): (
        "a39e3d9720ce595dfbf275cce51cf3e6bdd6c0ac312b6c1c916c82e69f716aa0"
    ),
}

CHECKPOINT_PATHS: Final[tuple[Path, ...]] = tuple(
    Path(f"data_gemma4/checkpoints/gemma4_v54_fixed_prefix_ple_reader_v{version}")
    for version in range(1, 6)
)
ABSENT_RESULT_PATHS: Final[tuple[Path, ...]] = tuple(
    Path(f"reports/gemma4/metrics/gemma4_v54_fixed_prefix_ple_reader_v{version}_result.json")
    for version in range(1, 4)
)
FAILED_CAUSAL_CHECKS: Final[tuple[str, ...]] = (
    "changed_pair_complete_unit_delta",
    "changed_wrong_prefix_positive_margin_rate",
    "changed_wrong_prefix_positive_margin_rate_delta",
    "greedy_exact_accuracy_delta",
)
PASSED_RETENTION_AND_NLL_CHECKS: Final[tuple[str, ...]] = (
    "retention_mean_ce_increase",
    "retention_mean_kl",
    "retention_next_token_top1_agreement",
    "validation_answer_nll_improvement",
)


def _resolve(path: Path, root: Path) -> Path:
    return path if path.is_absolute() else root / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read(path: Path, root: Path) -> dict[str, Any]:
    value = json.loads(_resolve(path, root).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def _selection_summary(result: dict[str, Any]) -> dict[str, Any]:
    selection = result["selection"]
    baseline = selection["baseline_teacher"]
    candidate = selection["candidate_teacher"]
    retention = selection["candidate_retention"]
    checks = selection["checks"]
    return {
        "validation_rows": int(candidate["answer_nll_count"]),
        "changed_sides": int(candidate["changed_side_count"]),
        "changed_units": int(candidate["changed_unit_count"]),
        "answer_nll_before": float(baseline["answer_nll_mean"]),
        "answer_nll_after": float(candidate["answer_nll_mean"]),
        "answer_nll_improvement": float(baseline["answer_nll_mean"])
        - float(candidate["answer_nll_mean"]),
        "positive_wrong_prefix_sides_before": int(
            baseline["changed_positive_margin_sides"]
        ),
        "positive_wrong_prefix_sides_after": int(
            candidate["changed_positive_margin_sides"]
        ),
        "positive_wrong_prefix_rate_before": float(
            baseline["changed_positive_margin_rate"]
        ),
        "positive_wrong_prefix_rate_after": float(
            candidate["changed_positive_margin_rate"]
        ),
        "complete_changed_units_before": int(baseline["changed_complete_units"]),
        "complete_changed_units_after": int(candidate["changed_complete_units"]),
        "retention_mean_ce_increase_nats": float(retention["mean_ce_increase_nats"]),
        "retention_mean_kl_nats": float(retention["mean_kl_nats"]),
        "retention_next_token_top1_agreement": float(
            retention["next_token_top1_agreement"]
        ),
        "failed_checks": sorted(name for name, passed in checks.items() if passed is False),
        "passed_checks": sorted(name for name, passed in checks.items() if passed is True),
        "greedy_executed": selection["greedy"] is not None,
    }


def authenticate_v1_v5_negative_results(
    root: str | Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Return a fail-closed, model-free authentication of the terminal chain."""

    resolved_root = Path(root).expanduser().resolve()
    observed_hashes: dict[str, str] = {}
    reports: dict[str, dict[str, Any]] = {}
    for path, expected in EVIDENCE_SHA256.items():
        resolved = _resolve(path, resolved_root)
        if not resolved.is_file() or resolved.is_symlink():
            raise FileNotFoundError(f"PLE evidence is absent or linked: {path}")
        observed = _sha256(resolved)
        if observed != expected:
            raise ValueError(f"PLE evidence digest changed: {path}: {observed} != {expected}")
        observed_hashes[path.as_posix()] = observed
        reports[path.stem] = _read(path, resolved_root)

    checkpoint_absence = {
        path.as_posix(): not _resolve(path, resolved_root).exists()
        for path in CHECKPOINT_PATHS
    }
    early_result_absence = {
        path.as_posix(): not _resolve(path, resolved_root).exists()
        for path in ABSENT_RESULT_PATHS
    }

    v1_pre = reports["gemma4_v54_fixed_prefix_ple_reader_v1_preregistration"]
    v1_smoke = reports["gemma4_v54_fixed_prefix_ple_reader_v1_smoke"]
    v2_pre = reports["gemma4_v54_fixed_prefix_ple_reader_v2_preregistration"]
    v2_smoke = reports["gemma4_v54_fixed_prefix_ple_reader_v2_smoke"]
    v2_abort = reports["gemma4_v54_fixed_prefix_ple_reader_v2_abort"]
    v3_pre = reports["gemma4_v54_fixed_prefix_ple_reader_v3_preregistration"]
    v3_smoke = reports["gemma4_v54_fixed_prefix_ple_reader_v3_smoke"]
    v3_abort = reports["gemma4_v54_fixed_prefix_ple_reader_v3_abort"]
    v4_pre = reports["gemma4_v54_fixed_prefix_ple_reader_v4_preregistration"]
    v4_smoke = reports["gemma4_v54_fixed_prefix_ple_reader_v4_smoke"]
    v4_result = reports["gemma4_v54_fixed_prefix_ple_reader_v4_result"]
    v5_pre = reports["gemma4_v54_fixed_prefix_ple_reader_v5_preregistration"]
    v5_smoke = reports["gemma4_v54_fixed_prefix_ple_reader_v5_smoke"]
    v5_result = reports["gemma4_v54_fixed_prefix_ple_reader_v5_result"]

    v4_metrics = _selection_summary(v4_result)
    v5_metrics = _selection_summary(v5_result)
    expected_failed = sorted(FAILED_CAUSAL_CHECKS)
    expected_passed = sorted(PASSED_RETENTION_AND_NLL_CHECKS)

    checks = {
        "all_evidence_hashes_match": len(observed_hashes) == len(EVIDENCE_SHA256),
        "all_reader_checkpoints_absent": all(checkpoint_absence.values()),
        "v1_v3_terminal_result_files_absent": all(early_result_absence.values()),
        "shared_fixed_prefix_contract": all(
            result.get("fixed_prefix")
            == {
                "all_scene_latents_present": True,
                "computed_before_question": True,
                "question_dependent_retrieval": False,
                "same_prefix_for_unchanged_scene": True,
                "shape": [1, 258, 1536],
            }
            for result in (v4_result, v5_result)
        ),
        "v1_failed_only_numerical_smoke_tolerance": (
            v1_pre.get("artifact") == "gemma4_v54_fixed_prefix_ple_reader_v1"
            and v1_smoke.get("artifact")
            == "gemma4_v54_fixed_prefix_ple_reader_v1_gradient_smoke"
            and v1_smoke.get("passed") is False
            and math.isfinite(float(v1_smoke["loss"]))
            and float(v1_smoke["gradient_l2"]) > 0.0
            and 1e-6 < abs(float(v1_smoke["initial_retention_kl"])) <= 1e-5
            and v1_smoke.get("preregistration_sha256")
            == EVIDENCE_SHA256[
                Path(
                    "reports/gemma4/metrics/"
                    "gemma4_v54_fixed_prefix_ple_reader_v1_preregistration.json"
                )
            ]
            and v2_pre.get("v1_failure", {}).get("only_failed_condition")
            == "retention_self_kl_above_1e-6"
        ),
        "v2_smoke_passed_then_serialization_abort_before_training": (
            v2_smoke.get("passed") is True
            and v2_smoke.get("v1_training_objective_or_gate_changed") is False
            and v2_abort.get("status")
            == "aborted_before_optimizer_construction_no_checkpoint"
            and v2_abort.get("error", {}).get("type") == "TypeError"
            and v2_abort.get("failure_scope", {}).get("adapter_update_count") == 0
            and v2_abort.get("failure_scope", {}).get("optimizer_constructed") is False
            and v2_abort.get("failure_scope", {}).get("training_started") is False
            and v2_abort.get("checkpoint_absent") is True
        ),
        "v3_serialization_fix_then_resource_abort_before_training": (
            v3_pre.get("only_change", {}).get("field")
            == "diagnostic_hash_serialization.for_tuple_keyed_mappings"
            and v3_smoke.get("passed") is True
            and v3_smoke.get("status") == "passed_by_exact_v2_smoke_inheritance"
            and v3_abort.get("status")
            == "aborted_before_optimizer_construction_mps_oom_no_checkpoint"
            and v3_abort.get("error", {}).get("type") == "RuntimeError"
            and v3_abort.get("failure_scope", {}).get("adapter_update_count") == 0
            and v3_abort.get("failure_scope", {}).get("optimizer_constructed") is False
            and v3_abort.get("failure_scope", {}).get("training_started") is False
            and v3_abort.get("checkpoint_absent") is True
        ),
        "v4_resource_change_equivalence_passed": (
            v4_pre.get("resource_only_changes", {}).get("same_correct_wrong_prefix_objective")
            is True
            and v4_smoke.get("passed") is True
            and v4_smoke.get("synthetic_equivalence", {}).get("exact") is True
            and float(v4_smoke["real_one_row_absolute_difference"]) <= 1e-6
            and float(v4_smoke["tail_gradient_l2"]) > 0.0
        ),
        "v4_terminal_negative_exact": (
            v4_result.get("status") == "failed_no_checkpoint"
            and v4_result.get("passed") is False
            and v4_result.get("checkpoint_published") is False
            and v4_result.get("promotion_eligible") is False
            and v4_result.get("checkpoint") is None
            and v4_result.get("training", {}).get("updates") == 40
            and v4_metrics["failed_checks"] == expected_failed
            and v4_metrics["passed_checks"] == expected_passed
            and v4_metrics["greedy_executed"] is False
        ),
        "v5_schedule_and_objective_smoke_passed": (
            v5_pre.get("single_v5_arm", {}).get("updates") == 80
            and v5_pre.get("single_v5_arm", {}).get("pair_cycles") == 2
            and v5_pre.get("single_v5_arm", {}).get("all_496_broad_rows_exactly_once")
            is True
            and v5_smoke.get("passed") is True
            and float(v5_smoke["gradient_l2"]) > 0.0
            and v5_smoke.get("deferred_holdout_accessed") is False
            and v5_smoke.get("final_scenes_accessed") is False
        ),
        "v5_terminal_negative_exact": (
            v5_result.get("status") == "failed_no_checkpoint"
            and v5_result.get("passed") is False
            and v5_result.get("checkpoint_published") is False
            and v5_result.get("promotion_eligible") is False
            and v5_result.get("checkpoint") is None
            and v5_result.get("training", {}).get("updates") == 80
            and v5_result.get("training", {}).get("pair_cycles") == 2
            and v5_result.get("training", {}).get("broad_rows_consumed_exactly_once")
            == 496
            and v5_metrics["failed_checks"] == expected_failed
            and v5_metrics["passed_checks"] == expected_passed
            and v5_metrics["greedy_executed"] is False
        ),
        "v4_v5_baselines_identical": (
            v4_result["selection"]["baseline_teacher"]
            == v5_result["selection"]["baseline_teacher"]
            and v4_result["selection"]["baseline_retention"]
            == v5_result["selection"]["baseline_retention"]
        ),
        "v5_deferred_and_final_splits_untouched": (
            v5_result.get("deferred_holdout", {}).get("accessed") is False
            and v5_result.get("final_scenes_000025_through_000030_accessed") is False
        ),
        "environmental_text_and_oracle_runtime_absent": all(
            result.get("runtime_leakage", {}).get("environmental_text_inputs") == []
            and result.get("runtime_leakage", {}).get("oracle_runtime_access") is False
            for result in (v4_result, v5_result)
        ),
    }
    if not all(checks.values()):
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise RuntimeError(f"PLE V1--V5 negative evidence failed authentication: {failed}")

    return {
        "schema_version": 1,
        "artifact": "gemma4_v54_fixed_prefix_ple_reader_v1_v5_terminal_evidence",
        "status": "authenticated_terminal_negative_no_checkpoint",
        "evidence_authenticated": True,
        "passed": False,
        "promotion_eligible": False,
        "checkpoint_published": False,
        "model_loaded": False,
        "mps_used": False,
        "target_module": "model.language_model.per_layer_model_projection",
        "trainable_parameter_count": 41_984,
        "fixed_prefix_shape": [1, 258, 1536],
        "question_dependent_retrieval": False,
        "versions": {
            "v1": {
                "status": "gradient_smoke_failed_no_training",
                "updates": 0,
                "retention_self_kl": float(v1_smoke["initial_retention_kl"]),
                "smoke_tolerance": 1e-6,
                "diagnosis": "finite repeat-forward noise exceeded the original tolerance",
            },
            "v2": {
                "status": v2_abort["status"],
                "updates": 0,
                "diagnosis": "diagnostic tuple-key JSON serialization error",
            },
            "v3": {
                "status": v3_abort["status"],
                "updates": 0,
                "diagnosis": "full-sequence vocabulary loss exceeded MPS memory",
                "attempted_allocation_mib": float(
                    v3_abort["error"]["memory"]["attempted_allocation_mib"]
                ),
            },
            "v4": {
                "status": v4_result["status"],
                "updates": 40,
                "elapsed_seconds": float(v4_result["elapsed_seconds"]),
                "peak_process_rss_bytes": int(v4_result["memory"]["peak_process_rss_bytes"]),
                "mps_driver_allocated_bytes": int(
                    v4_result["memory"]["mps_driver_allocated_bytes"]
                ),
                **v4_metrics,
            },
            "v5": {
                "status": v5_result["status"],
                "updates": 80,
                "pair_cycles": 2,
                "broad_rows_consumed_exactly_once": 496,
                "elapsed_seconds": float(v5_result["elapsed_seconds"]),
                "peak_process_rss_bytes": int(v5_result["memory"]["peak_process_rss_bytes"]),
                "mps_driver_allocated_bytes": int(
                    v5_result["memory"]["mps_driver_allocated_bytes"]
                ),
                "deferred_holdout_accessed": False,
                "final_split_accessed": False,
                **v5_metrics,
            },
        },
        "scientific_conclusion": (
            "The PLE surface improved generic answer likelihood while failing to improve "
            "and in both completed runs reducing scene-selective wrong-prefix and complete-"
            "pair metrics; the longer, stronger V5 objective did not reverse that result."
        ),
        "checks": checks,
        "checkpoint_absence": checkpoint_absence,
        "early_result_absence": early_result_absence,
        "evidence_sha256": observed_hashes,
        "measurement_evidence_paths": sorted(observed_hashes),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()
    print(
        json.dumps(
            authenticate_v1_v5_negative_results(args.root),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
