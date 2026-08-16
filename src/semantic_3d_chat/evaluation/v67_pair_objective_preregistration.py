"""Create the immutable V67 pair-objective training preregistration."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.training.train_question_control_v66 import (
    V66B_BEHAVIOR_THRESHOLDS,
)

_FILTERED_TRAIN_SHA256: Final[str] = (
    "84b99385fadc5d06e44465ada5902f56131192298ca1539373dc3b334608cbf1"
)
_TRAINING_BASELINE_LOCK_SHA256: Final[str] = (
    "b1f20e64889116cceb0904ecb3842a6e43fcd6fa3cb0675c32a24f4d278e55e6"
)
_FAILED_V66B_REPORT_SHA256: Final[str] = (
    "253bd0f299148b6e7fa0bba79be3a3be2fc0d4bdfce87c7ef3795420d72fadf3"
)
_V66B_PREREGISTRATION_SHA256: Final[str] = (
    "9c47e43e85b66bcf07794ccc206783db6a40b18af8ad29407475f081e60930bf"
)


@dataclass(frozen=True)
class V67NumericScreenThresholds:
    """Pair-held-out numeric screen required before any V67 generation."""

    held_supported_class_exact_minimum: int = 300
    held_supported_total: int = 571
    held_unsupported_total: int = 5
    held_changed_class_exact_minimum: int = 45
    held_changed_total: int = 75
    held_complete_class_units_minimum: int = 15
    held_complete_unit_total: int = 35
    held_prediction_change_units_minimum: int = 20
    mean_pair_delta_cosine_minimum: float = 0.35
    positive_pair_delta_units_minimum: int = 27
    mean_own_over_opposite_margin_minimum: float = 0.02
    positive_own_over_opposite_sides_minimum: int = 53
    fully_supported_pair_sides: int = 70


V67_NUMERIC_SCREEN_THRESHOLDS: Final[V67NumericScreenThresholds] = (
    V67NumericScreenThresholds()
)


V67_HYPERPARAMETERS: Final[dict[str, int | float]] = {
    "seed": 670067,
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
    "pair_refinement_epochs": 60,
    "pair_refinement_repeats": 4,
    "pair_unit_batch_size": 10,
    "pair_learning_rate": 0.0001,
    "pair_value_weight": 1.0,
    "pair_delta_weight": 4.0,
    "pair_opposite_weight": 4.0,
    "pair_opposite_margin": 0.15,
    "pair_classification_weight": 1.0,
    "retention_batches_per_epoch": 2,
    "retention_batch_size": 48,
    "retention_weight": 0.5,
    "gradient_clip_norm": 1.0,
}


def _resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return (value if value.is_absolute() else PROJECT_ROOT / value).resolve()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_v67_preregistration() -> dict[str, Any]:
    """Return a protocol frozen before V67 training or Gemma generation."""

    return {
        "schema_version": 1,
        "artifact": "v67_pair_objective_training_preregistration",
        "status": "locked_before_v67_training_screen_or_generation",
        "research_change": (
            "changed_fact_oversampling_plus_native_pair_delta_and_exact_"
            "paired_opposite_margin_refinement"
        ),
        "failed_predecessor": {
            "artifact": "v66b_allrow_paired_opposite_pair_disjoint_training",
            "path": "reports/gemma4/metrics/v66b_allrow_always_on_distillation.json",
            "sha256": _FAILED_V66B_REPORT_SHA256,
            "preregistration_sha256": _V66B_PREREGISTRATION_SHA256,
            "promotion_eligible": False,
            "terminal_reason": "pair_disjoint_allrow_behavior_gate_failed",
            "held_supported_exact": 409,
            "held_supported_total": 571,
            "held_changed_side_exact": 37,
            "held_changed_side_total": 75,
            "held_complete_units": 5,
            "held_complete_unit_total": 35,
            "held_prediction_change_units": 16,
            "spatial_relation_exact": 49,
            "spatial_relation_total": 120,
            "predecessor_artifact_bytes_modified": False,
        },
        "diagnosed_mechanism": {
            "v66_changed_repeats": 1,
            "v66_pair_delta_weight": 0.0,
            "v66_pair_refinement_steps": 0,
            "v66_all_rows_shared_equal_sampling": True,
            "v66_train_numeric_prototype_top1_near_one_but_pair_behavior_failed": True,
        },
        "authorization": {
            "filtered_training_qa_sha256": _FILTERED_TRAIN_SHA256,
            "training_baseline_lock_sha256": _TRAINING_BASELINE_LOCK_SHA256,
            "training_rows": 576,
            "training_scenes": 24,
            "counterfactual_pairs": 12,
            "changed_sides": 80,
            "changed_units": 40,
            "answer_classes": 28,
        },
        "fixed_hyperparameters": dict(V67_HYPERPARAMETERS),
        "numeric_screen": {
            "required_before_any_greedy_generation": True,
            "protocol": "leave_one_counterfactual_pair_out_numeric_only",
            "all_12_pairs_held_out_once": True,
            "held_pair_rows_used_for_optimization": False,
            "held_pair_teacher_sources_used": False,
            "fold_codebook_and_basis_built_after_exclusion": True,
            "question_and_answer_text_serialized": False,
            "thresholds": asdict(V67_NUMERIC_SCREEN_THRESHOLDS),
        },
        "pair_objective": {
            "identical_question_both_sides_required": True,
            "different_canonical_target_both_sides_required": True,
            "native_width_value_preservation": True,
            "predicted_pair_delta_aligned_to_numeric_teacher_delta": True,
            "own_teacher_closer_than_exact_paired_opposite_teacher": True,
            "changed_units_oversampled_atomically": True,
            "stable_row_retention_batches": True,
            "question_dependent_scene_retrieval": False,
            "runtime_answer_codebook": False,
            "runtime_training_teacher_cache": False,
        },
        "behavioral_gates": {
            "unchanged_from_v66b": True,
            "thresholds": json.loads(
                json.dumps(asdict(V66B_BEHAVIOR_THRESHOLDS), allow_nan=False)
            ),
            "actual_greedy_local_gemma_required": True,
            "exact_paired_opposite_scene_dependence_required": True,
            "saved_runtime_raw_question_token_generation_required": True,
            "sealed_checkpoint_public_reload_required": True,
        },
        "publication": {
            "screen_report_create_once": True,
            "screen_pass_authorization_required_for_full_run": True,
            "checkpoint_absent_on_screen_failure": True,
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


def write_v67_preregistration(path: str | Path) -> tuple[Path, str]:
    destination = _resolve(path)
    if destination.exists():
        raise FileExistsError(f"V67 preregistration already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            build_v67_preregistration(),
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
    path, digest = write_v67_preregistration(args.output)
    print(json.dumps({"path": str(path), "sha256": digest}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "V67_HYPERPARAMETERS",
    "V67_NUMERIC_SCREEN_THRESHOLDS",
    "V67NumericScreenThresholds",
    "build_v67_preregistration",
    "write_v67_preregistration",
]
