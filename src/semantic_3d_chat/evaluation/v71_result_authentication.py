"""Read-only authentication for the sealed V71 numeric-screen result."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.evaluation.v62_pair_disjoint_preregistration import (
    TRAIN_PAIR_IDS,
)
from semantic_3d_chat.training import train_question_control_v67 as v67

V71_PREREGISTRATION: Final[Path] = Path(
    "reports/gemma4/metrics/v71_multiscale_preregistration.json"
)
V71_NUMERIC_SCREEN: Final[Path] = Path(
    "reports/gemma4/metrics/v71_multiscale_numeric_screen.json"
)
V71_CHECKPOINT: Final[Path] = Path(
    "data_gemma4/checkpoints/gemma4_v71_multiscale_control"
)
V71_PREREGISTRATION_SHA256: Final[str] = (
    "0b4be59db7b21991039f3f7bf32d4e268c7da7f8bdfbc9d2a679d403e265ced4"
)
V71_NUMERIC_SCREEN_SHA256: Final[str] = (
    "f04482423dbd49b36116303c629bc5bf8286a206a81ad9ab294760a47c9b0a8f"
)
V71_TRAINING_IDENTITY_SHA256: Final[str] = (
    "52966ac3f0febf7db82c15f6e509f5573d33be5a8b9d18eeb007fd2530428b9f"
)
V71_EXPECTED_METRICS: Final[dict[str, int | float | bool]] = {
    "answer_or_question_text_stored": False,
    "changed_class_exact": 57,
    "changed_total": 75,
    "complete_class_units": 17,
    "complete_unit_total": 35,
    "fully_supported_pair_sides": 70,
    "gemma_generation_used": False,
    "inventory_total": 576,
    "mean_own_over_opposite_margin": 0.10917966876711165,
    "mean_pair_delta_cosine": 0.49870162941515445,
    "own_over_opposite_margin_sum": 7.642576813697815,
    "pair_delta_cosine_sum": 17.454557029530406,
    "positive_own_over_opposite_sides": 52,
    "positive_pair_delta_units": 28,
    "prediction_change_units": 17,
    "supported_class_exact": 489,
    "supported_total": 571,
    "unsupported_total": 5,
}
V71_EXPECTED_FAILED_CHECKS: Final[tuple[str, ...]] = (
    "held_prediction_change_units",
    "positive_own_over_opposite_sides",
)


def _absolute(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return (value if value.is_absolute() else PROJECT_ROOT / value).resolve()


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with _absolute(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_object(path: str | Path) -> dict[str, Any]:
    value = json.loads(_absolute(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _authenticate(
    preregistration_path: str | Path,
    screen_path: str | Path,
    checkpoint_path: str | Path,
) -> dict[str, Any]:
    preregistration = _read_object(preregistration_path)
    report = _read_object(screen_path)
    if _sha256_file(preregistration_path) != V71_PREREGISTRATION_SHA256:
        raise ValueError("V71 preregistration digest differs from its immutable pin")
    if _sha256_file(screen_path) != V71_NUMERIC_SCREEN_SHA256:
        raise ValueError("V71 numeric-screen digest differs from its immutable pin")
    if _absolute(checkpoint_path).exists():
        raise ValueError("V71 checkpoint exists despite the failed numeric screen")

    source_hashes = preregistration.get("implementation_source_hashes")
    preserved = preregistration.get("preserved_predecessor_hashes")
    architecture_lock = preregistration.get("architecture")
    numeric_lock = preregistration.get("numeric_screen")
    publication = preregistration.get("publication")
    scope_lock = preregistration.get("scope")
    if not all(
        isinstance(value, Mapping)
        for value in (
            source_hashes,
            preserved,
            architecture_lock,
            numeric_lock,
            publication,
            scope_lock,
        )
    ):
        raise TypeError("V71 preregistration lacks its evidence contract")
    for path_text, digest in {**source_hashes, **preserved}.items():
        if not _is_sha256(digest) or _sha256_file(path_text) != digest:
            raise ValueError(f"V71 authenticated source/evidence changed: {path_text}")
    if (
        preregistration.get("schema_version") != 1
        or preregistration.get("artifact")
        != "v71_multiscale_two_branch_preregistration"
        or preregistration.get("status") != "locked_before_v71_numeric_screen"
        or architecture_lock.get("environment_latent_count") != 256
        or architecture_lock.get("branch_moment_counts") != [8, 32]
        or architecture_lock.get("both_branches_process_every_environment_latent")
        is not True
        or architecture_lock.get("held_rows_never_tune_or_select_fusion") is not True
        or architecture_lock.get("question_dependent_scene_retrieval") is not False
        or numeric_lock.get("all_12_pairs_held_out_once") is not True
        or numeric_lock.get("held_pair_rows_used_for_optimization") is not False
        or numeric_lock.get("held_pair_teacher_sources_used") is not False
        or numeric_lock.get("wall_time_budget_seconds") != 1200
        or publication.get("checkpoint_never_written_by_numeric_screen") is not True
        or publication.get("gemma_generation_never_used_by_numeric_screen") is not True
        or publication.get("atlas_compilation_never_used_by_numeric_screen") is not True
        or any(
            scope_lock.get(field) is not False
            for field in (
                "validation_inputs_used",
                "scorer_inputs_used",
                "oracle_loaded",
                "fresh_development_loaded",
                "internal_validation_loaded",
                "deferred_final_loaded",
            )
        )
    ):
        raise ValueError("V71 preregistration semantic contract changed")

    result = report.get("result")
    report_scope = report.get("scope")
    architecture = report.get("architecture")
    if not all(
        isinstance(value, Mapping)
        for value in (result, report_scope, architecture)
    ):
        raise TypeError("V71 result lacks architecture, scope, or numeric evidence")
    folds = result.get("folds")
    metrics = result.get("metrics")
    checks = result.get("checks")
    if (
        not isinstance(folds, list)
        or not isinstance(metrics, Mapping)
        or not isinstance(checks, Mapping)
    ):
        raise TypeError("V71 result lacks folds, metrics, or checks")
    if [fold.get("held_pair_id") for fold in folds] != list(TRAIN_PAIR_IDS):
        raise ValueError("V71 held-pair inventory or order changed")

    fusion_weights: list[float] = []
    numeric_folds: list[Mapping[str, Any]] = []
    expected_optimizer_names: list[str] | None = None
    for fold in folds:
        fit = fold.get("fit")
        numeric = fold.get("numeric_screen")
        if not isinstance(fit, Mapping) or not isinstance(numeric, Mapping):
            raise TypeError("V71 fold lacks fit or numeric evidence")
        optimizer_names = fit.get("optimizer_parameter_names")
        if (
            fold.get("held_rows_used_for_optimization") is not False
            or fold.get("held_teacher_sources_used") is not False
            or not _is_sha256(fold.get("fold_codebook_sha256"))
            or not _is_sha256(fold.get("fold_basis_sha256"))
            or fit.get("question_norm_frozen") is not True
            or fit.get("v71_branch_moment_counts") != [8, 32]
            or fit.get("v71_fusion_trained_on_held_rows") is not False
            or not isinstance(optimizer_names, list)
            or "coefficient_output.fusion_logit" not in optimizer_names
            or not any(
                name.startswith("scene_projection.branch_8")
                for name in optimizer_names
            )
            or not any(
                name.startswith("scene_projection.branch_32")
                for name in optimizer_names
            )
            or numeric.get("gemma_generation_used") is not False
            or numeric.get("question_or_answer_text_stored") is not False
        ):
            raise ValueError("V71 fold isolation or architecture evidence changed")
        if expected_optimizer_names is None:
            expected_optimizer_names = list(optimizer_names)
        elif optimizer_names != expected_optimizer_names:
            raise ValueError("V71 optimizer scope changed across folds")
        weight = fit.get("v71_fusion_weight_branch_8")
        if (
            isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not math.isfinite(float(weight))
            or not 0.10 <= float(weight) <= 0.90
        ):
            raise ValueError("V71 fold fusion weight is invalid")
        fusion_weights.append(float(weight))
        numeric_folds.append(numeric)

    aggregate = v67.aggregate_numeric_screens_v67(numeric_folds)
    recomputed_checks = v67.assess_numeric_screen_v67(aggregate)
    failed_checks = tuple(
        sorted(name for name, passed in recomputed_checks.items() if not passed)
    )
    if (
        dict(metrics) != aggregate
        or dict(metrics) != V71_EXPECTED_METRICS
        or dict(checks) != recomputed_checks
        or failed_checks != V71_EXPECTED_FAILED_CHECKS
    ):
        raise ValueError("V71 aggregate metrics or gates differ from fold evidence")
    numeric_fit_seconds = report.get("numeric_fit_seconds")
    total_seconds = report.get("total_wall_time_seconds")
    if (
        report.get("schema_version") != 1
        or report.get("artifact") != "v71_multiscale_numeric_screen_v1"
        or report.get("preregistration_sha256") != V71_PREREGISTRATION_SHA256
        or report.get("training_identity_sha256")
        != V71_TRAINING_IDENTITY_SHA256
        or report.get("implementation_source_hashes") != source_hashes
        or report.get("passed") is not False
        or report.get("promotion_eligible") is not False
        or report.get("checkpoint_published") is not False
        or report.get("gemma_generation_used") is not False
        or report.get("full_behavioral_run_executed") is not False
        or report.get("atlas_compilation_executed") is not False
        or report.get("terminal_reason")
        != "numeric_screen_failed_no_generation_or_checkpoint_authorized"
        or result.get("status") != "failed"
        or result.get("passed") is not False
        or architecture.get("branch_moment_counts") != [8, 32]
        or architecture.get("both_branches_process_all_environment_latents") is not True
        or report_scope.get("question_or_answer_text_stored") is not False
        or any(
            report_scope.get(field) is not False
            for field in (
                "validation_inputs_used",
                "scorer_inputs_used",
                "oracle_loaded",
                "fresh_development_loaded",
                "internal_validation_loaded",
                "deferred_final_loaded",
            )
        )
        or not isinstance(numeric_fit_seconds, (int, float))
        or not isinstance(total_seconds, (int, float))
        or isinstance(numeric_fit_seconds, bool)
        or isinstance(total_seconds, bool)
        or not 0.0 <= float(numeric_fit_seconds) <= float(total_seconds) <= 1200.0
    ):
        raise ValueError("V71 terminal, timing, publication, or scope evidence changed")

    return {
        "status": "authenticated_numeric_screen_failed_no_publication",
        "measurement_status": "authenticated_screen_failed",
        "measurement_authenticated": True,
        "passed": False,
        "promotion_eligible": False,
        "checkpoint_published": False,
        "checkpoint_absent": True,
        "gemma_generation_used": False,
        "full_behavioral_run_executed": False,
        "atlas_compilation_executed": False,
        "fold_count": len(folds),
        "metrics": dict(metrics),
        "gate_checks": dict(checks),
        "failed_checks": list(failed_checks),
        "fail_gaps": {
            "held_prediction_change_units": 3,
            "positive_own_over_opposite_sides": 1,
        },
        "fusion_weight_branch_8_minimum": min(fusion_weights),
        "fusion_weight_branch_8_maximum": max(fusion_weights),
        "numeric_fit_seconds": float(numeric_fit_seconds),
        "total_wall_time_seconds": float(total_seconds),
        "wall_time_budget_seconds": 1200,
        "preregistration_sha256": V71_PREREGISTRATION_SHA256,
        "numeric_screen_sha256": V71_NUMERIC_SCREEN_SHA256,
        "training_identity_sha256": V71_TRAINING_IDENTITY_SHA256,
        "implementation_source_sha256": dict(source_hashes),
        "authentication_checks": {
            "preregistration_digest_matches": True,
            "numeric_screen_digest_matches": True,
            "implementation_source_digests_match": True,
            "predecessor_evidence_digests_match": True,
            "fold_inventory_matches": True,
            "fold_aggregates_recomputed": True,
            "unchanged_gate_recomputed": True,
            "held_rows_never_tuned_fusion": True,
            "training_only_scope_attested": True,
            "no_generation_full_or_atlas": True,
            "checkpoint_absent": True,
            "wall_time_budget_respected": True,
        },
        "measurement_evidence_paths": [
            Path(preregistration_path).as_posix(),
            Path(screen_path).as_posix(),
            *source_hashes.keys(),
            *preserved.keys(),
        ],
    }


def authenticate_v71_result(
    preregistration_path: str | Path = V71_PREREGISTRATION,
    screen_path: str | Path = V71_NUMERIC_SCREEN,
    checkpoint_path: str | Path = V71_CHECKPOINT,
) -> dict[str, Any]:
    """Return a fail-closed, machine-readable authentication summary."""

    evidence = [Path(preregistration_path).as_posix(), Path(screen_path).as_posix()]
    try:
        return _authenticate(preregistration_path, screen_path, checkpoint_path)
    except (KeyError, OSError, TypeError, ValueError, ZeroDivisionError) as error:
        return {
            "status": "v71_numeric_screen_evidence_authentication_failed",
            "measurement_status": "artifact_present_authentication_failed",
            "measurement_authenticated": False,
            "measurement_evidence_error": f"{type(error).__name__}: {error}",
            "measurement_evidence_paths": evidence,
        }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", default=str(V71_PREREGISTRATION))
    parser.add_argument("--screen", default=str(V71_NUMERIC_SCREEN))
    parser.add_argument("--checkpoint", default=str(V71_CHECKPOINT))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = authenticate_v71_result(
        args.preregistration, args.screen, args.checkpoint
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("measurement_authenticated") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "V71_NUMERIC_SCREEN_SHA256",
    "V71_PREREGISTRATION_SHA256",
    "V71_TRAINING_IDENTITY_SHA256",
    "authenticate_v71_result",
]
