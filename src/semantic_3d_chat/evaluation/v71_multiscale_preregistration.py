"""Create the immutable V71 two-branch multiscale preregistration."""

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
from semantic_3d_chat.evaluation.v69_pair_augmentation_preregistration import (
    V69_ARM_GRID,
    V69_COMMON_HYPERPARAMETERS,
    V69_FOUNDATION_ARM,
)
from semantic_3d_chat.training.train_question_control_v66 import (
    V66B_BEHAVIOR_THRESHOLDS,
)

_FAILED_V70_REPORT: Final[str] = (
    "reports/gemma4/metrics/v70_low_frequency_moments_numeric_screen.json"
)
_FAILED_V70_REPORT_SHA256: Final[str] = (
    "711255b33f8c93133af7c0647234b3296532e574358b3a96a0e053eb894bfe53"
)
_FAILED_V70_PREREGISTRATION: Final[str] = (
    "reports/gemma4/metrics/v70_low_frequency_moments_preregistration.json"
)
_FAILED_V70_PREREGISTRATION_SHA256: Final[str] = (
    "395cfabf1d82cb552d0f4a828367061c3d3066f914b34c950bc2ccfbedfeb64a"
)
_FAILED_V70_TRAINING_IDENTITY_SHA256: Final[str] = (
    "a3e53d21ea894ebe15cbf7e18a39b4c52f23cf45e30db9b790c4de6cf4248a30"
)

V71_AUGMENTATION_ARM: Final[dict[str, str | int | float]] = dict(V69_ARM_GRID[0])
V71_BRANCH_SEED_OFFSET: Final[int] = 7_100_032

_IMPLEMENTATION_PATHS: Final[tuple[str, ...]] = (
    "src/semantic_3d_chat/evaluation/v71_multiscale_preregistration.py",
    "src/semantic_3d_chat/scene_encoder/question_control_v71.py",
    "src/semantic_3d_chat/training/train_question_control_v71.py",
    "scripts/run_gemma4_v71_multiscale.sh",
    "src/semantic_3d_chat/training/question_control_v69_objective.py",
    "src/semantic_3d_chat/training/train_question_control_v69.py",
    "src/semantic_3d_chat/training/question_control_v68_objective.py",
    "src/semantic_3d_chat/training/train_question_control_v68.py",
    "src/semantic_3d_chat/training/question_control_v67_objective.py",
    "src/semantic_3d_chat/training/train_question_control_v67.py",
    "src/semantic_3d_chat/training/train_question_control_v66.py",
    "src/semantic_3d_chat/training/train_question_control_v63.py",
    "src/semantic_3d_chat/training/train_question_control_v65.py",
    "src/semantic_3d_chat/scene_encoder/question_control_v7.py",
)

_PRESERVED_PREDECESSOR_HASHES: Final[dict[str, str]] = {
    "reports/gemma4/metrics/v68_regularized_pair_preregistration.json": (
        "6642b16b38e169df0059b2ccfb6aba0f8b1315052f3aad0e2871b30eeda6811f"
    ),
    "reports/gemma4/metrics/v68_regularized_pair_numeric_grid.json": (
        "0584383d4b3806b6a4205cc516e2957ea05b5c76066134335485d5f638c9e609"
    ),
    "reports/gemma4/metrics/v69_pair_augmentation_preregistration.json": (
        "5cd567a129e083600b8913aa0438c0a8115aba83bd70c24c40ce5475a4bcfb3e"
    ),
    "reports/gemma4/metrics/v69_pair_augmentation_numeric_grid.json": (
        "6f6a6af8ab0c254bd8ea1704593770c8445aebaad02bbb55478b94f61103e2a8"
    ),
    _FAILED_V70_PREREGISTRATION: _FAILED_V70_PREREGISTRATION_SHA256,
    _FAILED_V70_REPORT: _FAILED_V70_REPORT_SHA256,
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


def implementation_source_hashes_v71() -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in _IMPLEMENTATION_PATHS:
        source = _resolve(relative)
        if not source.is_file() or source.is_symlink():
            raise FileNotFoundError(f"V71 implementation source missing: {relative}")
        result[relative] = _sha256_file(source)
    return result


def _authenticated_v70_failure() -> dict[str, Any]:
    for relative, expected in _PRESERVED_PREDECESSOR_HASHES.items():
        source = _resolve(relative)
        if (
            not source.is_file()
            or source.is_symlink()
            or _sha256_file(source) != expected
        ):
            raise ValueError(f"V71 predecessor bytes changed: {relative}")
    payload = json.loads(_resolve(_FAILED_V70_REPORT).read_text(encoding="utf-8"))
    result = payload.get("result") if isinstance(payload, dict) else None
    metrics = result.get("metrics") if isinstance(result, dict) else None
    checks = result.get("checks") if isinstance(result, dict) else None
    expected_metrics = {
        "supported_class_exact": 484,
        "supported_total": 571,
        "changed_class_exact": 55,
        "changed_total": 75,
        "complete_class_units": 15,
        "complete_unit_total": 35,
        "prediction_change_units": 16,
        "positive_own_over_opposite_sides": 51,
        "fully_supported_pair_sides": 70,
        "positive_pair_delta_units": 30,
    }
    if (
        payload.get("artifact") != "v70_low_frequency_moments_numeric_screen_v1"
        or payload.get("passed") is not False
        or payload.get("promotion_eligible") is not False
        or payload.get("checkpoint_published") is not False
        or payload.get("gemma_generation_used") is not False
        or payload.get("full_behavioral_run_executed") is not False
        or payload.get("atlas_compilation_executed") is not False
        or payload.get("training_identity_sha256")
        != _FAILED_V70_TRAINING_IDENTITY_SHA256
        or not isinstance(metrics, dict)
        or any(metrics.get(key) != value for key, value in expected_metrics.items())
        or not isinstance(checks, dict)
        or checks.get("held_prediction_change_units") is not False
        or checks.get("positive_own_over_opposite_sides") is not False
    ):
        raise ValueError("V70 no longer matches the diagnosed V71 predecessor")
    return payload


def build_v71_preregistration() -> dict[str, Any]:
    predecessor = _authenticated_v70_failure()
    metrics = predecessor["result"]["metrics"]
    return {
        "schema_version": 1,
        "artifact": "v71_multiscale_two_branch_preregistration",
        "status": "locked_before_v71_numeric_screen",
        "research_change": (
            "independent_first_8_and_first_32_all_latent_dct_value_branches_"
            "with_one_training_fold_learned_bounded_global_fusion_scalar"
        ),
        "failed_predecessor": {
            "artifact": predecessor["artifact"],
            "path": _FAILED_V70_REPORT,
            "sha256": _FAILED_V70_REPORT_SHA256,
            "preregistration_path": _FAILED_V70_PREREGISTRATION,
            "preregistration_sha256": _FAILED_V70_PREREGISTRATION_SHA256,
            "training_identity_sha256": _FAILED_V70_TRAINING_IDENTITY_SHA256,
            "complete_class_units": metrics["complete_class_units"],
            "prediction_change_units": metrics["prediction_change_units"],
            "positive_own_over_opposite_sides": metrics[
                "positive_own_over_opposite_sides"
            ],
            "positive_pair_delta_units": metrics["positive_pair_delta_units"],
            "predecessor_artifact_bytes_modified": False,
        },
        "diagnosed_mechanism": {
            "v69_first_8_prediction_change_units": 18,
            "v70_first_32_prediction_change_units": 16,
            "v69_first_8_positive_own_over_opposite_sides": 50,
            "v70_first_32_positive_own_over_opposite_sides": 51,
            "independent_predictor_errors_may_be_complementary": True,
            "spectral_band_32_to_63_is_not_the_tested_complement": True,
        },
        "architecture": {
            "environment_latent_count": 256,
            "branch_moment_counts": [8, 32],
            "branch_8_moment_indices": [0, 8],
            "branch_32_moment_indices": [0, 32],
            "both_branches_process_every_environment_latent": True,
            "branches_have_independent_scene_projections": True,
            "branches_have_independent_question_projections": True,
            "branches_have_independent_trunks": True,
            "branches_have_independent_coefficient_and_magnitude_heads": True,
            "shared_fold_local_output_basis": True,
            "fusion_formula": "w*branch_8+(1-w)*branch_32",
            "fusion_weight_formula": "0.10+0.80*sigmoid(global_trainable_logit)",
            "fusion_initial_weight_branch_8": 0.5,
            "fusion_scalar_trained_only_on_training_fold_rows": True,
            "held_rows_never_tune_or_select_fusion": True,
            "question_dependent_scene_retrieval": False,
        },
        "training_protocol": {
            "foundation_arm": dict(V69_FOUNDATION_ARM),
            "augmentation_arm": dict(V71_AUGMENTATION_ARM),
            "augmentation_arm_is_exact_v69_balanced_extrapolation_010": (
                V71_AUGMENTATION_ARM == dict(V69_ARM_GRID[0])
            ),
            "v69_common_hyperparameters": dict(V69_COMMON_HYPERPARAMETERS),
            "branch_seed_offset": V71_BRANCH_SEED_OFFSET,
            "one_arm_only": True,
            "no_hyperparameter_search": True,
        },
        "numeric_screen": {
            "protocol": "leave_one_counterfactual_pair_out_numeric_only",
            "all_12_pairs_held_out_once": True,
            "held_pair_rows_used_for_optimization": False,
            "held_pair_teacher_sources_used": False,
            "fold_codebook_and_basis_built_after_exclusion": True,
            "question_and_answer_text_serialized": False,
            "thresholds_unchanged_from_v70_v69_v68_v67": True,
            "thresholds": asdict(V67_NUMERIC_SCREEN_THRESHOLDS),
            "wall_time_budget_seconds": 1200,
        },
        "behavioral_gates": {
            "unchanged_from_v66b": True,
            "thresholds": json.loads(
                json.dumps(asdict(V66B_BEHAVIOR_THRESHOLDS), allow_nan=False)
            ),
            "not_executed_by_v71_numeric_screen": True,
        },
        "implementation_source_hashes": implementation_source_hashes_v71(),
        "preserved_predecessor_hashes": dict(_PRESERVED_PREDECESSOR_HASHES),
        "publication": {
            "screen_report_create_once": True,
            "checkpoint_never_written_by_numeric_screen": True,
            "gemma_generation_never_used_by_numeric_screen": True,
            "atlas_compilation_never_used_by_numeric_screen": True,
            "passing_screen_requires_parent_authorization": True,
            "failing_screen_publishes_no_checkpoint": True,
        },
        "scope": {
            "training_only": True,
            "numeric_teacher_and_prefix_cache_only": True,
            "validation_inputs_used": False,
            "scorer_inputs_used": False,
            "oracle_loaded": False,
            "fresh_development_loaded": False,
            "internal_validation_loaded": False,
            "deferred_final_loaded": False,
        },
    }


def write_v71_preregistration(path: str | Path) -> tuple[Path, str]:
    destination = _resolve(path)
    if destination.exists():
        raise FileExistsError(f"V71 preregistration already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            build_v71_preregistration(),
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
    path, digest = write_v71_preregistration(args.output)
    print(json.dumps({"path": str(path), "sha256": digest}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "V71_AUGMENTATION_ARM",
    "V71_BRANCH_SEED_OFFSET",
    "build_v71_preregistration",
    "implementation_source_hashes_v71",
    "write_v71_preregistration",
]
