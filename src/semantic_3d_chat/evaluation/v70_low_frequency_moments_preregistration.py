"""Create the immutable V70 32-moment numeric-screen preregistration."""

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
from semantic_3d_chat.evaluation.v68_regularized_pair_preregistration import (
    V68_ARM_GRID,
    V68_COMMON_HYPERPARAMETERS,
)
from semantic_3d_chat.training.train_question_control_v66 import (
    V66B_BEHAVIOR_THRESHOLDS,
)

_FAILED_V69_REPORT: Final[str] = (
    "reports/gemma4/metrics/v69_pair_augmentation_numeric_grid.json"
)
_FAILED_V69_REPORT_SHA256: Final[str] = (
    "6f6a6af8ab0c254bd8ea1704593770c8445aebaad02bbb55478b94f61103e2a8"
)
_FAILED_V69_PREREGISTRATION: Final[str] = (
    "reports/gemma4/metrics/v69_pair_augmentation_preregistration.json"
)
_FAILED_V69_PREREGISTRATION_SHA256: Final[str] = (
    "5cd567a129e083600b8913aa0438c0a8115aba83bd70c24c40ce5475a4bcfb3e"
)
_FAILED_V69_TRAINING_IDENTITY_SHA256: Final[str] = (
    "0d16b6cae0ad5984860f4e94aa0cd0dc029d17b187d582dcc8c2cec8e3094a9e"
)

_V70_COMMON = dict(V68_COMMON_HYPERPARAMETERS)
_V70_COMMON["moment_count"] = 32
V70_COMMON_HYPERPARAMETERS: Final[dict[str, int | float]] = _V70_COMMON
V70_ARM: Final[dict[str, str | int | float]] = dict(V68_ARM_GRID[2])

_IMPLEMENTATION_PATHS: Final[tuple[str, ...]] = (
    "src/semantic_3d_chat/evaluation/v70_low_frequency_moments_preregistration.py",
    "src/semantic_3d_chat/training/train_question_control_v70.py",
    "scripts/run_gemma4_v70_low_frequency_moments.sh",
    "src/semantic_3d_chat/evaluation/v68_regularized_pair_preregistration.py",
    "src/semantic_3d_chat/training/question_control_v68_objective.py",
    "src/semantic_3d_chat/training/train_question_control_v68.py",
    "src/semantic_3d_chat/training/question_control_v67_objective.py",
    "src/semantic_3d_chat/training/train_question_control_v67.py",
    "src/semantic_3d_chat/training/train_question_control_v66.py",
    "src/semantic_3d_chat/training/train_question_control_v63.py",
    "src/semantic_3d_chat/training/train_question_control_v65.py",
    "src/semantic_3d_chat/scene_encoder/question_control_v7.py",
)

_PRESERVED_V68_PATH_HASHES: Final[dict[str, str]] = {
    "reports/gemma4/metrics/v68_regularized_pair_numeric_grid.json": (
        "0584383d4b3806b6a4205cc516e2957ea05b5c76066134335485d5f638c9e609"
    ),
    "reports/gemma4/metrics/v68_regularized_pair_preregistration.json": (
        "6642b16b38e169df0059b2ccfb6aba0f8b1315052f3aad0e2871b30eeda6811f"
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


def _resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return (value if value.is_absolute() else PROJECT_ROOT / value).resolve()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def implementation_source_hashes_v70() -> dict[str, str]:
    """Hash the V70 entry points and every inherited executable component."""

    result: dict[str, str] = {}
    for relative in _IMPLEMENTATION_PATHS:
        source = _resolve(relative)
        if not source.is_file() or source.is_symlink():
            raise FileNotFoundError(f"V70 implementation source is unavailable: {relative}")
        result[relative] = _sha256_file(source)
    return result


def _authenticated_v69_failure() -> dict[str, Any]:
    preregistration = _resolve(_FAILED_V69_PREREGISTRATION)
    report = _resolve(_FAILED_V69_REPORT)
    if (
        not preregistration.is_file()
        or preregistration.is_symlink()
        or _sha256_file(preregistration) != _FAILED_V69_PREREGISTRATION_SHA256
    ):
        raise ValueError("V70 requires the exact sealed V69 preregistration")
    if (
        not report.is_file()
        or report.is_symlink()
        or _sha256_file(report) != _FAILED_V69_REPORT_SHA256
    ):
        raise ValueError("V70 requires the exact sealed V69 numeric screen")
    payload = json.loads(report.read_text(encoding="utf-8"))
    arms = payload.get("arm_results") if isinstance(payload, dict) else None
    expected = (
        ("balanced_extrapolation_010", 15, 18, 50),
        ("balanced_extrapolation_010_question_mix_010", 15, 17, 49),
        ("balanced_extrapolation_020_question_mix_015", 15, 16, 49),
    )
    if (
        payload.get("artifact") != "v69_pair_augmentation_numeric_grid_v1"
        or payload.get("passed") is not False
        or payload.get("promotion_eligible") is not False
        or payload.get("checkpoint_published") is not False
        or payload.get("gemma_generation_used") is not False
        or payload.get("training_identity_sha256")
        != _FAILED_V69_TRAINING_IDENTITY_SHA256
        or payload.get("preregistration_sha256")
        != _FAILED_V69_PREREGISTRATION_SHA256
        or not isinstance(arms, list)
        or len(arms) != len(expected)
    ):
        raise ValueError("V69 numeric screen no longer matches the V70 diagnosis")
    for arm, (arm_id, complete, changes, margins) in zip(arms, expected, strict=True):
        metrics = arm.get("metrics") if isinstance(arm, dict) else None
        checks = arm.get("checks") if isinstance(arm, dict) else None
        if (
            arm.get("arm_id") != arm_id
            or arm.get("status") != "failed"
            or arm.get("passed") is not False
            or not isinstance(metrics, dict)
            or metrics.get("complete_class_units") != complete
            or metrics.get("prediction_change_units") != changes
            or metrics.get("positive_own_over_opposite_sides") != margins
            or not isinstance(checks, dict)
            or checks.get("held_prediction_change_units") is not False
            or checks.get("positive_own_over_opposite_sides") is not False
        ):
            raise ValueError("A V69 arm no longer matches the V70 failure diagnosis")
    for relative, expected_sha256 in _PRESERVED_V68_PATH_HASHES.items():
        if _sha256_file(_resolve(relative)) != expected_sha256:
            raise ValueError(f"V70 inherited V68 bytes changed: {relative}")
    return payload


def build_v70_preregistration() -> dict[str, Any]:
    """Return the one-variable protocol frozen before the V70 screen."""

    predecessor = _authenticated_v69_failure()
    best = predecessor["arm_results"][0]["metrics"]
    return {
        "schema_version": 1,
        "artifact": "v70_low_frequency_moments_preregistration",
        "status": "locked_before_v70_numeric_screen",
        "research_change": "first_8_to_first_32_fixed_dct_scene_moments_only",
        "failed_predecessor": {
            "artifact": predecessor["artifact"],
            "path": _FAILED_V69_REPORT,
            "sha256": _FAILED_V69_REPORT_SHA256,
            "preregistration_path": _FAILED_V69_PREREGISTRATION,
            "preregistration_sha256": _FAILED_V69_PREREGISTRATION_SHA256,
            "training_identity_sha256": _FAILED_V69_TRAINING_IDENTITY_SHA256,
            "held_supported_exact": best["supported_class_exact"],
            "held_supported_total": best["supported_total"],
            "held_changed_side_exact": best["changed_class_exact"],
            "held_changed_side_total": best["changed_total"],
            "held_complete_units": best["complete_class_units"],
            "held_complete_unit_total": best["complete_unit_total"],
            "held_prediction_change_units": best["prediction_change_units"],
            "positive_own_over_opposite_sides": best[
                "positive_own_over_opposite_sides"
            ],
            "predecessor_artifact_bytes_modified": False,
        },
        "diagnosed_mechanism": {
            "supported_class_learning_has_large_headroom": True,
            "v67_v68_v69_prediction_changes": [14, 17, 18],
            "v67_v68_v69_positive_margins": [47, 50, 50],
            "three_loss_or_augmentation_successors_plateaued": True,
            "current_signature_uses_only_first_8_of_256_fixed_dct_moments": True,
            "cached_prefix_pair_delta_energy_fraction_rounded": {
                "pair_000006": {"first_8": 0.0221, "first_32": 0.1132},
                "pair_000016": {"first_8": 0.0168, "first_32": 0.1030},
                "pair_000026": {"first_8": 0.1529, "first_32": 0.2140},
            },
        },
        "controlled_ablation": {
            "environment_latent_count": 256,
            "source_moment_count": 8,
            "candidate_moment_count": 32,
            "moment_family": "fixed_low_frequency_dct",
            "every_environment_latent_influences_dc_moment": True,
            "complete_scene_prefix_unchanged": True,
            "question_independent_scene_prefix_unchanged": True,
            "v68_strong_foundation_exact": True,
            "v69_augmentation_used": False,
            "question_mixing_used": False,
            "only_preregistered_variable_changed": "moment_count",
        },
        "common_hyperparameters": dict(V70_COMMON_HYPERPARAMETERS),
        "single_arm": dict(V70_ARM),
        "numeric_screen": {
            "protocol": "leave_one_counterfactual_pair_out_numeric_only",
            "all_12_pairs_held_out_once": True,
            "held_pair_rows_used_for_optimization": False,
            "held_pair_teacher_sources_used": False,
            "fold_codebook_and_basis_built_after_exclusion": True,
            "question_and_answer_text_serialized": False,
            "thresholds_unchanged_from_v69_v68_v67": True,
            "thresholds": asdict(V67_NUMERIC_SCREEN_THRESHOLDS),
            "wall_time_budget_seconds": 1200,
        },
        "behavioral_gates": {
            "unchanged_from_v69_v68_v67_and_v66b": True,
            "thresholds": json.loads(
                json.dumps(asdict(V66B_BEHAVIOR_THRESHOLDS), allow_nan=False)
            ),
            "not_executed_by_v70_numeric_screen": True,
        },
        "implementation_source_hashes": implementation_source_hashes_v70(),
        "preserved_v68_path_hashes": dict(_PRESERVED_V68_PATH_HASHES),
        "publication": {
            "screen_report_create_once": True,
            "checkpoint_never_written_by_numeric_screen": True,
            "gemma_generation_never_used_by_numeric_screen": True,
            "passing_screen_requires_parent_authorization_before_atlas_or_full_run": True,
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


def write_v70_preregistration(path: str | Path) -> tuple[Path, str]:
    destination = _resolve(path)
    if destination.exists():
        raise FileExistsError(f"V70 preregistration already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            build_v70_preregistration(),
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
    path, digest = write_v70_preregistration(args.output)
    print(json.dumps({"path": str(path), "sha256": digest}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "V70_ARM",
    "V70_COMMON_HYPERPARAMETERS",
    "build_v70_preregistration",
    "implementation_source_hashes_v70",
    "write_v70_preregistration",
]
