"""Create the immutable V68 regularized pair-screen preregistration."""

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

_FAILED_V67_REPORT: Final[str] = "reports/gemma4/metrics/v67_pair_objective_numeric_screen.json"
_FAILED_V67_REPORT_SHA256: Final[str] = (
    "d2e0085857a4647f518e3906d2a1a4c02826d314d021da80f7cf380065328304"
)
_FAILED_V67_PREREGISTRATION_SHA256: Final[str] = (
    "a87ad59102c48da95390659839b76707c3d32af726034ab930fae5e01ba7ab8f"
)


V68_COMMON_HYPERPARAMETERS: Final[dict[str, int | float]] = {
    "seed": 680068,
    "basis_rank": 112,
    "moment_count": 8,
    "interaction_dim": 32,
    "trunk_dim": 192,
    "maximum_control_rms": 0.25,
    "initial_control_rms": 0.075,
    "base_epochs": 160,
    "base_batch_size": 48,
    "base_learning_rate": 0.0003,
    "base_weight_decay": 0.0001,
    "prototype_classification_epochs": 40,
    "prototype_classification_weight": 1.0,
    "prototype_classification_temperature": 0.07,
    "prototype_value_preservation_weight": 1.0,
    "pair_unit_batch_size": 10,
    "retention_batch_size": 48,
    "gradient_clip_norm": 1.0,
    "anchor_scale_floor": 0.0001,
}


# This is a deliberately small, ordered training-only screen.  The first arm
# passing every unchanged V67 numeric gate is selected; later arms are not run.
# Thus there is no result-driven hyperparameter editing or best-of-grid cherry
# picking, and no Gemma generation occurs during arm selection.
V68_ARM_GRID: Final[tuple[dict[str, str | int | float], ...]] = (
    {
        "arm_id": "balanced_all_value_anchor",
        "optimizer_scope": "all_value",
        "pair_refinement_epochs": 45,
        "pair_refinement_repeats": 3,
        "pair_learning_rate": 0.00005,
        "pair_value_weight": 1.0,
        "pair_delta_weight": 6.0,
        "pair_opposite_weight": 8.0,
        "pair_opposite_margin": 0.20,
        "pair_classification_weight": 2.0,
        "hard_negative_weight": 6.0,
        "hard_negative_margin": 0.08,
        "retention_batches_per_epoch": 3,
        "retention_weight": 1.0,
        "anchor_weight": 0.5,
    },
    {
        "arm_id": "interaction_only_anchor",
        "optimizer_scope": "interaction_only",
        "pair_refinement_epochs": 60,
        "pair_refinement_repeats": 4,
        "pair_learning_rate": 0.000075,
        "pair_value_weight": 1.0,
        "pair_delta_weight": 8.0,
        "pair_opposite_weight": 8.0,
        "pair_opposite_margin": 0.20,
        "pair_classification_weight": 2.0,
        "hard_negative_weight": 8.0,
        "hard_negative_margin": 0.10,
        "retention_batches_per_epoch": 3,
        "retention_weight": 1.0,
        "anchor_weight": 0.75,
    },
    {
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
    },
)


_IMPLEMENTATION_PATHS: Final[tuple[str, ...]] = (
    "src/semantic_3d_chat/evaluation/v68_regularized_pair_preregistration.py",
    "src/semantic_3d_chat/training/question_control_v68_objective.py",
    "src/semantic_3d_chat/training/train_question_control_v68.py",
    "scripts/run_gemma4_v68_regularized_pair.sh",
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


def implementation_source_hashes_v68() -> dict[str, str]:
    """Hash every V68-owned entry point plus its inherited executable core."""

    result: dict[str, str] = {}
    for relative in _IMPLEMENTATION_PATHS:
        source = _resolve(relative)
        if not source.is_file() or source.is_symlink():
            raise FileNotFoundError(f"V68 implementation source is unavailable: {relative}")
        result[relative] = _sha256_file(source)
    return result


def _authenticated_v67_failure() -> dict[str, Any]:
    source = _resolve(_FAILED_V67_REPORT)
    if not source.is_file() or source.is_symlink():
        raise FileNotFoundError("V68 requires the create-once V67 numeric screen")
    if _sha256_file(source) != _FAILED_V67_REPORT_SHA256:
        raise ValueError("V67 numeric screen differs from the V68 failure diagnosis")
    payload = json.loads(source.read_text(encoding="utf-8"))
    metrics = payload.get("metrics") if isinstance(payload, dict) else None
    checks = payload.get("checks") if isinstance(payload, dict) else None
    expected = {
        "supported_class_exact": 482,
        "supported_total": 571,
        "changed_class_exact": 52,
        "changed_total": 75,
        "complete_class_units": 13,
        "complete_unit_total": 35,
        "prediction_change_units": 14,
        "positive_own_over_opposite_sides": 47,
        "fully_supported_pair_sides": 70,
    }
    if (
        payload.get("artifact") != "v67_pair_objective_numeric_screen_v1"
        or payload.get("passed") is not False
        or payload.get("checkpoint_published") is not False
        or payload.get("gemma_generation_used") is not False
        or payload.get("preregistration_sha256") != _FAILED_V67_PREREGISTRATION_SHA256
        or not isinstance(metrics, dict)
        or any(metrics.get(key) != value for key, value in expected.items())
        or not isinstance(checks, dict)
        or checks.get("held_complete_class_units") is not False
        or checks.get("held_prediction_change_units") is not False
        or checks.get("positive_own_over_opposite_sides") is not False
    ):
        raise ValueError("V67 numeric screen no longer matches the V68 diagnosis")
    return payload


def build_v68_preregistration() -> dict[str, Any]:
    """Return the V68 protocol frozen before its numeric screen or generation."""

    predecessor = _authenticated_v67_failure()
    metrics = predecessor["metrics"]
    return {
        "schema_version": 1,
        "artifact": "v68_regularized_pair_grid_preregistration",
        "status": "locked_before_v68_training_screen_or_generation",
        "research_change": (
            "ordered_regularized_pair_grid_with_fold_local_hard_negative_"
            "prototype_margin_and_base_parameter_anchor"
        ),
        "failed_predecessor": {
            "artifact": predecessor["artifact"],
            "path": _FAILED_V67_REPORT,
            "sha256": _FAILED_V67_REPORT_SHA256,
            "preregistration_sha256": _FAILED_V67_PREREGISTRATION_SHA256,
            "promotion_eligible": False,
            "terminal_reason": predecessor["terminal_reason"],
            "held_supported_exact": metrics["supported_class_exact"],
            "held_supported_total": metrics["supported_total"],
            "held_changed_side_exact": metrics["changed_class_exact"],
            "held_changed_side_total": metrics["changed_total"],
            "held_complete_units": metrics["complete_class_units"],
            "held_complete_unit_total": metrics["complete_unit_total"],
            "held_prediction_change_units": metrics["prediction_change_units"],
            "positive_own_over_opposite_sides": metrics["positive_own_over_opposite_sides"],
            "fully_supported_pair_sides": metrics["fully_supported_pair_sides"],
            "held_complete_units_observed": 13,
            "held_complete_units_required": 15,
            "held_prediction_changes_observed": 14,
            "held_prediction_changes_required": 20,
            "held_positive_margins_observed": 47,
            "held_positive_margins_required": 53,
            "predecessor_artifact_bytes_modified": False,
        },
        "diagnosed_mechanism": {
            "seen_pair_delta_cosine_near_one": True,
            "seen_pair_own_over_opposite_fraction_one": True,
            "held_pair_three_gate_shortfall": True,
            "overfitting_risk_from_unanchored_all_value_refinement": True,
            "fold_local_hardest_wrong_class_not_explicitly_margined": True,
        },
        "common_hyperparameters": dict(V68_COMMON_HYPERPARAMETERS),
        "ordered_arm_grid": [dict(arm) for arm in V68_ARM_GRID],
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
            "held_pair_rows_used_for_optimization": False,
            "held_pair_teacher_sources_used": False,
            "fold_codebook_and_basis_built_after_exclusion": True,
            "question_and_answer_text_serialized": False,
            "thresholds_unchanged_from_v67": True,
            "thresholds": asdict(V67_NUMERIC_SCREEN_THRESHOLDS),
        },
        "behavioral_gates": {
            "unchanged_from_v67_and_v66b": True,
            "thresholds": json.loads(json.dumps(asdict(V66B_BEHAVIOR_THRESHOLDS), allow_nan=False)),
            "actual_greedy_local_gemma_required_after_numeric_pass": True,
            "exact_paired_opposite_scene_dependence_required": True,
            "saved_runtime_raw_question_token_generation_required": True,
            "sealed_checkpoint_public_reload_required": True,
        },
        "implementation_source_hashes": implementation_source_hashes_v68(),
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


def write_v68_preregistration(path: str | Path) -> tuple[Path, str]:
    destination = _resolve(path)
    if destination.exists():
        raise FileExistsError(f"V68 preregistration already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            build_v68_preregistration(),
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
    path, digest = write_v68_preregistration(args.output)
    print(json.dumps({"path": str(path), "sha256": digest}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "V68_ARM_GRID",
    "V68_COMMON_HYPERPARAMETERS",
    "build_v68_preregistration",
    "implementation_source_hashes_v68",
    "write_v68_preregistration",
]
