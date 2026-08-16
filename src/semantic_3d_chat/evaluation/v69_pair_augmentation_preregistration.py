"""Create the immutable V69 transition-balanced augmentation preregistration."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any, Final

from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.evaluation.v67_pair_objective_preregistration import (
    V67_NUMERIC_SCREEN_THRESHOLDS,
)
from semantic_3d_chat.training.train_question_control_v66 import (
    V66B_BEHAVIOR_THRESHOLDS,
)

_FAILED_V68_REPORT: Final[str] = (
    "reports/gemma4/metrics/v68_regularized_pair_numeric_grid.json"
)
_FAILED_V68_REPORT_SHA256: Final[str] = (
    "0584383d4b3806b6a4205cc516e2957ea05b5c76066134335485d5f638c9e609"
)
_FAILED_V68_PREREGISTRATION_SHA256: Final[str] = (
    "6642b16b38e169df0059b2ccfb6aba0f8b1315052f3aad0e2871b30eeda6811f"
)

V69_FOUNDATION_ARM: Final[dict[str, str | int | float]] = {
    "arm_id": "strong_all_value_anchor",
    "optimizer_scope": "all_value",
    "pair_refinement_epochs": 70,
    "pair_refinement_repeats": 4,
    "pair_learning_rate": 0.00003,
    "pair_value_weight": 1.0,
    "pair_delta_weight": 8.0,
    "pair_opposite_weight": 10.0,
    "pair_opposite_margin": 0.25,
    "pair_classification_weight": 2.0,
    "hard_negative_weight": 10.0,
    "hard_negative_margin": 0.12,
    "retention_batches_per_epoch": 4,
    "retention_weight": 1.5,
    "anchor_weight": 1.5,
}

V69_COMMON_HYPERPARAMETERS: Final[dict[str, int | float]] = {
    # Preserve the exact V68-strong foundation seed; the augmentation phase
    # uses a disjoint deterministic offset below.
    "seed": 680068,
    "augmentation_seed_offset": 9_000_069,
    "augmentation_batch_size": 10,
    "retention_batch_size": 48,
    "retention_batches_per_epoch": 3,
    "gradient_clip_norm": 1.0,
    "prototype_temperature": 0.07,
    "anchor_scale_floor": 0.0001,
    "augmentation_value_weight": 1.0,
    "augmentation_delta_weight": 8.0,
    "augmentation_opposite_weight": 12.0,
    "augmentation_opposite_margin": 0.28,
    "augmentation_classification_weight": 2.0,
    "augmentation_hard_negative_weight": 10.0,
    "augmentation_hard_negative_margin": 0.14,
    "retention_weight": 1.5,
    "anchor_weight": 2.0,
}

# Fixed-priority screen.  Every arm begins with the exact strongest V68
# training protocol, then applies only the stated training-fold augmentation.
# The first all-gate pass is selected; later arms are not executed.
V69_ARM_GRID: Final[tuple[dict[str, str | int | float], ...]] = (
    {
        "arm_id": "balanced_extrapolation_010",
        "augmentation_epochs": 24,
        "augmentation_repeats": 2,
        "augmentation_learning_rate": 0.00002,
        "signature_expansion": 0.10,
        "question_mix_weight": 0.0,
    },
    {
        "arm_id": "balanced_extrapolation_010_question_mix_010",
        "augmentation_epochs": 28,
        "augmentation_repeats": 2,
        "augmentation_learning_rate": 0.00002,
        "signature_expansion": 0.10,
        "question_mix_weight": 0.10,
    },
    {
        "arm_id": "balanced_extrapolation_020_question_mix_015",
        "augmentation_epochs": 32,
        "augmentation_repeats": 2,
        "augmentation_learning_rate": 0.000015,
        "signature_expansion": 0.20,
        "question_mix_weight": 0.15,
    },
)

_V68_PRESERVED_PATH_HASHES: Final[dict[str, str]] = {
    "reports/gemma4/metrics/v68_regularized_pair_preregistration.json": (
        _FAILED_V68_PREREGISTRATION_SHA256
    ),
    "reports/gemma4/metrics/v68_regularized_pair_numeric_grid.json": (
        _FAILED_V68_REPORT_SHA256
    ),
    "scripts/run_gemma4_v68_regularized_pair.sh": (
        "b5d43262dc7debdd18267add9abb1c6a6e51849234c7ba69966927f25c0f80f3"
    ),
    "src/semantic_3d_chat/training/question_control_v68_objective.py": (
        "36df4669ab46053ff850734bae715e2ea7799b3a1570239285f1512c26165c01"
    ),
    "src/semantic_3d_chat/training/train_question_control_v68.py": (
        "23f11bb4e595d0b84a8c3be49a7f83b2c69fc337d134524f17772da48bdfc0a2"
    ),
}

_IMPLEMENTATION_PATHS: Final[tuple[str, ...]] = (
    "src/semantic_3d_chat/evaluation/v69_pair_augmentation_preregistration.py",
    "src/semantic_3d_chat/training/question_control_v69_objective.py",
    "src/semantic_3d_chat/training/train_question_control_v69.py",
    "scripts/run_gemma4_v69_pair_augmentation.sh",
    "src/semantic_3d_chat/training/train_question_control_v68.py",
    "src/semantic_3d_chat/training/question_control_v68_objective.py",
    "src/semantic_3d_chat/training/train_question_control_v67.py",
    "src/semantic_3d_chat/training/question_control_v67_objective.py",
    "src/semantic_3d_chat/training/train_question_control_v66.py",
    "src/semantic_3d_chat/training/train_question_control_v63.py",
    "src/semantic_3d_chat/training/train_question_control_v65.py",
    "src/semantic_3d_chat/scene_encoder/question_control_v7.py",
)


def _resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return (value if value.is_absolute() else PROJECT_ROOT / value).resolve()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def implementation_source_hashes_v69() -> dict[str, str]:
    """Hash V69 and its inherited executable implementation graph."""

    result: dict[str, str] = {}
    for relative in _IMPLEMENTATION_PATHS:
        source = _resolve(relative)
        if not source.is_file() or source.is_symlink():
            raise FileNotFoundError(f"V69 implementation source is unavailable: {relative}")
        result[relative] = _sha256_file(source)
    for relative, expected in _V68_PRESERVED_PATH_HASHES.items():
        if _sha256_file(_resolve(relative)) != expected:
            raise ValueError(f"V69 refuses modified V68 predecessor bytes: {relative}")
    return result


def _authenticated_v68_failure() -> dict[str, Any]:
    source = _resolve(_FAILED_V68_REPORT)
    if not source.is_file() or source.is_symlink():
        raise FileNotFoundError("V69 requires the create-once V68 numeric grid report")
    if _sha256_file(source) != _FAILED_V68_REPORT_SHA256:
        raise ValueError("V68 numeric grid differs from the V69 failure diagnosis")
    payload = json.loads(source.read_text(encoding="utf-8"))
    results = payload.get("arm_results") if isinstance(payload, dict) else None
    expected = {
        "supported_class_exact": 489,
        "supported_total": 571,
        "changed_class_exact": 53,
        "changed_total": 75,
        "complete_class_units": 14,
        "complete_unit_total": 35,
        "prediction_change_units": 17,
        "positive_own_over_opposite_sides": 50,
        "fully_supported_pair_sides": 70,
        "positive_pair_delta_units": 28,
    }
    strongest = (
        next(
            (
                result
                for result in results
                if isinstance(result, dict)
                and result.get("arm_id") == "strong_all_value_anchor"
            ),
            None,
        )
        if isinstance(results, list)
        else None
    )
    metrics = strongest.get("metrics") if isinstance(strongest, dict) else None
    checks = strongest.get("checks") if isinstance(strongest, dict) else None
    if (
        payload.get("artifact") != "v68_regularized_pair_numeric_grid_v1"
        or payload.get("passed") is not False
        or payload.get("checkpoint_published") is not False
        or payload.get("gemma_generation_used") is not False
        or payload.get("preregistration_sha256") != _FAILED_V68_PREREGISTRATION_SHA256
        or not isinstance(results, list)
        or len(results) != 3
        or any(result.get("passed") is not False for result in results)
        or not isinstance(metrics, dict)
        or any(metrics.get(key) != value for key, value in expected.items())
        or not isinstance(checks, dict)
        or checks.get("held_complete_class_units") is not False
        or checks.get("held_prediction_change_units") is not False
        or checks.get("positive_own_over_opposite_sides") is not False
    ):
        raise ValueError("V68 numeric grid no longer matches the V69 diagnosis")
    return payload


def build_v69_preregistration() -> dict[str, Any]:
    """Return the V69 protocol frozen before any V69 numeric screen."""

    predecessor = _authenticated_v68_failure()
    strongest = next(
        result
        for result in predecessor["arm_results"]
        if result["arm_id"] == "strong_all_value_anchor"
    )
    metrics = strongest["metrics"]
    thresholds = asdict(V67_NUMERIC_SCREEN_THRESHOLDS)
    return {
        "schema_version": 1,
        "artifact": "v69_transition_balanced_pair_augmentation_preregistration",
        "status": "locked_before_v69_training_screen_or_generation",
        "research_change": (
            "strong_v68_foundation_plus_transition_balanced_continuous_"
            "question_and_pair_signature_augmentation"
        ),
        "failed_predecessor": {
            "artifact": predecessor["artifact"],
            "path": _FAILED_V68_REPORT,
            "sha256": _FAILED_V68_REPORT_SHA256,
            "preregistration_sha256": _FAILED_V68_PREREGISTRATION_SHA256,
            "selected_diagnostic_arm": "strong_all_value_anchor",
            "selected_for_new_research_reason": (
                "highest_held_changed_complete_and_prediction_change_counts"
            ),
            "held_supported_exact": metrics["supported_class_exact"],
            "held_supported_total": metrics["supported_total"],
            "held_changed_side_exact": metrics["changed_class_exact"],
            "held_changed_side_total": metrics["changed_total"],
            "held_complete_units_observed": metrics["complete_class_units"],
            "held_complete_units_required": thresholds[
                "held_complete_class_units_minimum"
            ],
            "held_prediction_changes_observed": metrics["prediction_change_units"],
            "held_prediction_changes_required": thresholds[
                "held_prediction_change_units_minimum"
            ],
            "held_positive_margins_observed": metrics[
                "positive_own_over_opposite_sides"
            ],
            "held_positive_margins_required": thresholds[
                "positive_own_over_opposite_sides_minimum"
            ],
            "mean_own_over_opposite_margin": metrics[
                "mean_own_over_opposite_margin"
            ],
            "mean_pair_delta_cosine": metrics["mean_pair_delta_cosine"],
            "predecessor_artifact_bytes_modified": False,
        },
        "diagnosed_mechanism": {
            "ordinary_supported_class_gate_has_large_headroom": True,
            "mean_pair_delta_and_positive_delta_gates_pass": True,
            "remaining_failures_are_discrete_pair_sensitivity_gates": True,
            "collapsed_opposite_scene_predictions_remain": True,
            "transition_imbalance_and_local_boundary_width_are_training_only_hypotheses": True,
        },
        "foundation_arm": dict(V69_FOUNDATION_ARM),
        "common_hyperparameters": dict(V69_COMMON_HYPERPARAMETERS),
        "ordered_arm_grid": [dict(arm) for arm in V69_ARM_GRID],
        "augmentation_contract": {
            "changed_units_bucketed_by_ordered_opaque_numeric_class_transition": True,
            "every_transition_oversampled_to_fold_local_maximum_bucket_size": True,
            "pair_midpoint_preserved_by_symmetric_signature_extrapolation": True,
            "question_mix_partner_has_identical_ordered_numeric_transition": True,
            "singleton_transition_questions_unchanged": True,
            "held_pair_rows_used_for_augmentation": False,
            "held_pair_questions_used_for_augmentation": False,
            "held_pair_teacher_sources_used": False,
            "natural_language_serialized": False,
        },
        "arm_selection": {
            "rule": "run_in_declared_order_and_select_first_all_gate_pass",
            "later_arms_skipped_after_first_pass": True,
            "best_metric_cherry_picking": False,
            "greedy_generation_during_selection": False,
            "full_mode_requires_exact_selected_arm_authorization": True,
        },
        "numeric_screen": {
            "protocol": "leave_one_counterfactual_pair_out_numeric_only",
            "all_12_pairs_held_out_once_per_executed_arm": True,
            "fold_codebook_and_basis_built_after_exclusion": True,
            "question_and_answer_text_serialized": False,
            "thresholds_unchanged_from_v68_v67": True,
            "thresholds": thresholds,
        },
        "behavioral_gates": {
            "unchanged_from_v68_v67_and_v66b": True,
            "thresholds": json.loads(
                json.dumps(asdict(V66B_BEHAVIOR_THRESHOLDS), allow_nan=False)
            ),
            "actual_greedy_local_gemma_required_after_numeric_pass": True,
            "exact_paired_opposite_scene_dependence_required": True,
            "saved_runtime_raw_question_token_generation_required": True,
            "sealed_checkpoint_public_reload_required": True,
        },
        "implementation_source_hashes": implementation_source_hashes_v69(),
        "preserved_v68_path_hashes": dict(_V68_PRESERVED_PATH_HASHES),
        "publication": {
            "screen_report_create_once": True,
            "screen_pass_authorization_required_for_full_run": True,
            "checkpoint_absent_on_all_arm_screen_failure": True,
            "checkpoint_absent_on_behavior_failure": True,
            "checkpoint_absent_on_paired_opposite_failure": True,
            "training_codebook_or_answer_text_in_checkpoint": False,
        },
        "scope": {
            "training_only": True,
            "validation_inputs_used": False,
            "scorer_inputs_used": False,
            "oracle_loaded": False,
            "fresh_development_loaded": False,
            "internal_validation_loaded": False,
            "deferred_final_loaded": False,
        },
    }


def write_v69_preregistration(path: str | Path) -> tuple[Path, str]:
    destination = _resolve(path)
    if destination.exists():
        raise FileExistsError(f"V69 preregistration already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            build_v69_preregistration(),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode()
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination, _sha256_file(destination)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    path, digest = write_v69_preregistration(args.output)
    print(json.dumps({"path": str(path), "sha256": digest}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "V69_ARM_GRID",
    "V69_COMMON_HYPERPARAMETERS",
    "V69_FOUNDATION_ARM",
    "build_v69_preregistration",
    "implementation_source_hashes_v69",
    "write_v69_preregistration",
]
