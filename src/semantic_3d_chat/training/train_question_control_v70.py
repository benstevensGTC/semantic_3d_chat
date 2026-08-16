"""V70 one-arm, numeric-only screen with 32 fixed scene moments.

This executable deliberately has no full or generation mode.  It reuses the
exact V68 strong fitting objective and changes only the number of fixed DCT
moments extracted from every complete 256-latent scene prefix.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import signal
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any, Final

import torch

from semantic_3d_chat.evaluation.v62_pair_disjoint_preregistration import (
    TRAIN_PAIR_IDS,
)
from semantic_3d_chat.evaluation.v67_pair_objective_preregistration import (
    V67_NUMERIC_SCREEN_THRESHOLDS,
)
from semantic_3d_chat.evaluation.v70_low_frequency_moments_preregistration import (
    V70_ARM,
    V70_COMMON_HYPERPARAMETERS,
    build_v70_preregistration,
    implementation_source_hashes_v70,
)
from semantic_3d_chat.training import train_question_control_v66 as v66
from semantic_3d_chat.training import train_question_control_v67 as v67
from semantic_3d_chat.training import train_question_control_v68 as v68
from semantic_3d_chat.training.question_control_v66_prototypes import (
    build_hybrid_answer_prototype_codebook_v66,
)
from semantic_3d_chat.training.train_question_control_v56 import (
    _resolve,
    _sha256_file,
)
from semantic_3d_chat.training.train_question_control_v63 import (
    V63Preflight,
    build_v63_preflight,
)
from semantic_3d_chat.training.train_question_control_v65 import (
    V65RuntimeBundle,
    _load_runtime,
    _tensor_sha256,
    validate_training_baseline_lock,
)

_EXPECTED_ROWS: Final[int] = 576
_WORK_ARTIFACT: Final[str] = "v70_low_frequency_moments_work_v1"
_FOLD_ARTIFACT: Final[str] = "v70_low_frequency_moments_fold_v1"
_SCREEN_ARTIFACT: Final[str] = "v70_low_frequency_moments_numeric_screen_v1"
_SOURCE_MOMENT_COUNT: Final[int] = 8
_CANDIDATE_MOMENT_COUNT: Final[int] = 32
_WALL_TIME_BUDGET_SECONDS: Final[int] = 1200


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _arm_sha256() -> str:
    return _canonical_sha256(V70_ARM)


def validate_v70_preregistration(path: str | Path) -> dict[str, Any]:
    source = _resolve(path)
    if not source.is_file() or source.is_symlink():
        raise FileNotFoundError("V70 preregistration is unavailable")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload != build_v70_preregistration():
        raise ValueError("V70 preregistration or implementation source differs from its lock")
    if payload.get("implementation_source_hashes") != implementation_source_hashes_v70():
        raise ValueError("V70 implementation source lock changed")
    return payload


def _preflight_args(args: argparse.Namespace) -> argparse.Namespace:
    """Authenticate V60 at its native 8 moments; inherit only question norm."""

    values = vars(v68._locked_fit_args(args, V70_ARM)).copy()
    values["moment_count"] = _SOURCE_MOMENT_COUNT
    return argparse.Namespace(**values)


def _fit_args(args: argparse.Namespace) -> argparse.Namespace:
    """Return exact V68-strong settings with only moment_count changed."""

    values = vars(v68._locked_fit_args(args, V70_ARM)).copy()
    values["moment_count"] = _CANDIDATE_MOMENT_COUNT
    return argparse.Namespace(**values)


def _training_identity(
    preflight: V63Preflight,
    teacher_audit: Mapping[str, Any],
    args: argparse.Namespace,
    preregistration: Mapping[str, Any],
) -> dict[str, Any]:
    identity = {
        "schema_version": 1,
        "artifact": "v70_low_frequency_moments_training_identity_v1",
        "preregistration_sha256": _sha256_file(_resolve(args.preregistration)),
        "filtered_training_qa_sha256": preflight.filtered_train_sha256,
        "training_baseline_lock_sha256": _sha256_file(
            _resolve(args.training_baseline_lock)
        ),
        "base_checkpoint_sha256": preflight.base_checkpoint_sha256,
        "runtime_config_sha256": preflight.runtime_config_sha256,
        "prefix_cache_manifest_sha256": preflight.prefix_manifest_sha256,
        "combined_teacher_audit_sha256": _canonical_sha256(teacher_audit),
        "source_v60_moment_count": _SOURCE_MOMENT_COUNT,
        "candidate_moment_count": _CANDIDATE_MOMENT_COUNT,
        "source_v60_inheritance": "frozen_question_norm_only",
        "common_hyperparameters": dict(V70_COMMON_HYPERPARAMETERS),
        "single_arm": dict(V70_ARM),
        "implementation_source_hashes": dict(
            preregistration["implementation_source_hashes"]
        ),
        "pair_ids": list(TRAIN_PAIR_IDS),
        "wall_time_budget_seconds": _WALL_TIME_BUDGET_SECONDS,
        "validation_inputs_used": False,
        "scorer_inputs_used": False,
        "oracle_loaded": False,
        "fresh_development_loaded": False,
        "internal_validation_loaded": False,
        "deferred_final_loaded": False,
    }
    return {**identity, "sha256": _canonical_sha256(identity)}


def _validate_cached_fold_v70(
    payload: object,
    *,
    held_pair_id: str,
    run_signature_sha256: str,
    held_rows: Sequence[Any],
    codebook: Any,
    basis: torch.Tensor,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise TypeError("V70 cached fold must be an object")
    if (
        payload.get("artifact") != _FOLD_ARTIFACT
        or payload.get("arm_id") != V70_ARM["arm_id"]
        or payload.get("arm_sha256") != _arm_sha256()
    ):
        raise ValueError("V70 cached fold provenance changed")
    translated = dict(payload)
    translated["artifact"] = v68._FOLD_ARTIFACT
    v68._validate_cached_fold_v68(
        translated,
        mode="screen",
        arm=V70_ARM,
        held_pair_id=held_pair_id,
        run_signature_sha256=run_signature_sha256,
        held_rows=held_rows,
        codebook=codebook,
        basis=basis,
    )
    return payload


def _run_folds(
    *,
    preflight: V63Preflight,
    teachers: Mapping[tuple[str, str], torch.Tensor],
    bundle: V65RuntimeBundle,
    args: argparse.Namespace,
    training_identity: Mapping[str, Any],
) -> tuple[dict[str, Any], float]:
    fit_args = _fit_args(args)
    work = _resolve(args.work_directory)
    run_manifest = {
        "schema_version": 1,
        "artifact": _WORK_ARTIFACT,
        "mode": "numeric_screen_only",
        "training_identity_sha256": training_identity["sha256"],
        "arm_id": V70_ARM["arm_id"],
        "arm_sha256": _arm_sha256(),
        "source_moment_count": _SOURCE_MOMENT_COUNT,
        "candidate_moment_count": _CANDIDATE_MOMENT_COUNT,
    }
    run_manifest["run_signature_sha256"] = _canonical_sha256(run_manifest)
    v66._prepare_work_directory(work, run_manifest)

    started = time.perf_counter()
    fold_payloads: list[dict[str, Any]] = []
    for fold_index, held_pair in enumerate(TRAIN_PAIR_IDS):
        elapsed = time.perf_counter() - started
        if elapsed >= _WALL_TIME_BUDGET_SECONDS:
            raise TimeoutError(
                f"V70 exceeded {_WALL_TIME_BUDGET_SECONDS}s before fold {held_pair}"
            )
        train_rows = tuple(row for row in preflight.rows if row.pair_id != held_pair)
        held_rows = tuple(row for row in preflight.rows if row.pair_id == held_pair)
        train_keys = {row.key for row in train_rows}
        fold_teachers = {
            key: value for key, value in teachers.items() if key in train_keys
        }
        codebook = build_hybrid_answer_prototype_codebook_v66(
            train_rows,
            fold_teachers,
            expected_class_count=None,
            scope=f"v70_32_moments_{held_pair}",
            forbidden_pair_id=held_pair,
        )
        basis = v66._codebook_basis(codebook, int(fit_args.basis_rank))
        fold_path = work / f"fold_{held_pair}.json"
        if fold_path.exists():
            cached = _validate_cached_fold_v70(
                json.loads(fold_path.read_text(encoding="utf-8")),
                held_pair_id=held_pair,
                run_signature_sha256=str(run_manifest["run_signature_sha256"]),
                held_rows=held_rows,
                codebook=codebook,
                basis=basis,
            )
            fold_payloads.append(cached)
            continue
        fit = v68._fit_regularized_pair(
            rows=train_rows,
            codebook=codebook,
            preflight=preflight,
            questions=bundle.question_embeddings,
            basis=basis,
            args=fit_args,
            arm=V70_ARM,
            seed=int(fit_args.seed) + (fold_index + 1) * 100_003,
            phase=f"v70_32_moments_{held_pair}",
        )
        held_fit = v68._held_fit(fit, held_rows, preflight)
        numeric_metrics, numeric_evidence = v67.numeric_screen_fold_v67(
            held_fit,
            held_rows,
            codebook=codebook,
            questions=bundle.question_embeddings,
        )
        payload: dict[str, Any] = {
            "schema_version": 1,
            "artifact": _FOLD_ARTIFACT,
            "run_signature_sha256": run_manifest["run_signature_sha256"],
            "arm_id": V70_ARM["arm_id"],
            "arm_sha256": _arm_sha256(),
            "held_pair_id": held_pair,
            "held_rows_used_for_optimization": False,
            "held_teacher_sources_used": False,
            "fold_codebook_sha256": codebook.sha256,
            "fold_basis_sha256": _tensor_sha256(basis),
            "fit": v68._fit_audit(held_fit),
            "numeric_screen": numeric_metrics,
            "numeric_evidence": list(numeric_evidence),
        }
        v66._atomic_new_json(fold_path, payload)
        fold_payloads.append(payload)

    elapsed = time.perf_counter() - started
    if elapsed > _WALL_TIME_BUDGET_SECONDS:
        raise TimeoutError(f"V70 exceeded its {_WALL_TIME_BUDGET_SECONDS}s budget")
    metrics = v67.aggregate_numeric_screens_v67(
        [fold["numeric_screen"] for fold in fold_payloads]
    )
    checks = v67.assess_numeric_screen_v67(metrics)
    result = {
        "arm_id": V70_ARM["arm_id"],
        "arm_sha256": _arm_sha256(),
        "status": "passed" if all(checks.values()) else "failed",
        "passed": all(checks.values()),
        "metrics": metrics,
        "checks": checks,
        "gemma_generation_used": False,
        "folds": [
            {
                key: fold[key]
                for key in (
                    "held_pair_id",
                    "held_rows_used_for_optimization",
                    "held_teacher_sources_used",
                    "fold_codebook_sha256",
                    "fold_basis_sha256",
                    "fit",
                    "numeric_screen",
                )
            }
            for fold in fold_payloads
        ],
    }
    return result, elapsed


def train_v70(
    args: argparse.Namespace,
    *,
    runtime_provider: Callable[..., V65RuntimeBundle] | None = None,
    supplemental_loader: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    total_started = time.perf_counter()
    preregistration = validate_v70_preregistration(args.preregistration)
    preregistration_sha256 = _sha256_file(_resolve(args.preregistration))
    preflight = build_v63_preflight(_preflight_args(args))
    if len(preflight.rows) != _EXPECTED_ROWS:
        raise ValueError("V70 authenticated training inventory changed")
    validate_training_baseline_lock(
        args.training_baseline_lock, expected_rows=preflight.rows
    )
    teachers, teacher_audit = v66.load_combined_verified_teachers_v66(
        preflight,
        args.supplemental_teacher_cache,
        supplemental_loader=supplemental_loader,
    )
    training_identity = _training_identity(
        preflight, teacher_audit, args, preregistration
    )
    output_checkpoint = _resolve(args.output_checkpoint)
    if output_checkpoint.exists():
        raise FileExistsError("V70 numeric screen requires an absent checkpoint target")
    bundle = (runtime_provider or _load_runtime)(
        preflight, requested_device=args.device
    )
    result, elapsed = _run_folds(
        preflight=preflight,
        teachers=teachers,
        bundle=bundle,
        args=args,
        training_identity=training_identity,
    )
    if output_checkpoint.exists():
        raise RuntimeError("V70 numeric screen unexpectedly published a checkpoint")
    passed = result["passed"] is True
    report = {
        "schema_version": 1,
        "artifact": _SCREEN_ARTIFACT,
        "passed": passed,
        "promotion_eligible": False,
        "checkpoint_published": False,
        "gemma_generation_used": False,
        "full_behavioral_run_executed": False,
        "atlas_compilation_executed": False,
        "terminal_reason": (
            "numeric_screen_passed_parent_authorization_required_before_atlas_or_full_run"
            if passed
            else "numeric_screen_failed_no_generation_or_checkpoint_authorized"
        ),
        "preregistration_sha256": preregistration_sha256,
        "training_identity_sha256": training_identity["sha256"],
        "implementation_source_hashes": dict(
            training_identity["implementation_source_hashes"]
        ),
        "authorization": {
            "baseline_lock_sha256": preflight.baseline_lock_sha256,
            "filtered_training_qa_sha256": preflight.filtered_train_sha256,
            "teacher_audit_sha256": _canonical_sha256(teacher_audit),
        },
        "architecture": {
            "complete_scene_prefix": True,
            "environment_latent_count": 256,
            "source_moment_count": _SOURCE_MOMENT_COUNT,
            "candidate_moment_count": _CANDIDATE_MOMENT_COUNT,
            "moment_family": "fixed_low_frequency_dct",
            "question_independent_scene_prefix": True,
            "question_dependent_scene_retrieval": False,
            "environmental_text_inputs": [],
        },
        "single_arm": dict(V70_ARM),
        "thresholds": asdict(V67_NUMERIC_SCREEN_THRESHOLDS),
        "result": result,
        "numeric_fit_seconds": elapsed,
        "total_wall_time_seconds": time.perf_counter() - total_started,
        "wall_time_budget_seconds": _WALL_TIME_BUDGET_SECONDS,
        "scope": {
            "training_only": True,
            "numeric_teacher_and_prefix_cache_only": True,
            "question_or_answer_text_stored": False,
            "validation_inputs_used": False,
            "scorer_inputs_used": False,
            "oracle_loaded": False,
            "fresh_development_loaded": False,
            "internal_validation_loaded": False,
            "deferred_final_loaded": False,
        },
    }
    v66._atomic_new_json(_resolve(args.training_report), report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-lock", required=True)
    parser.add_argument("--preregistration", required=True)
    parser.add_argument("--training-baseline-lock", required=True)
    parser.add_argument("--filtered-train-qa", required=True)
    parser.add_argument("--teacher-cache", required=True)
    parser.add_argument("--supplemental-teacher-cache", required=True)
    parser.add_argument("--prefix-cache", required=True)
    parser.add_argument("--base-runtime-config", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--source-v60-checkpoint", required=True)
    parser.add_argument("--work-directory", required=True)
    parser.add_argument("--output-checkpoint", required=True)
    parser.add_argument("--training-report", required=True)
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    random.seed(int(V70_COMMON_HYPERPARAMETERS["seed"]))
    torch.manual_seed(int(V70_COMMON_HYPERPARAMETERS["seed"]))
    previous_handler = signal.getsignal(signal.SIGALRM)

    def _budget_expired(_signum: int, _frame: object) -> None:
        raise TimeoutError(f"V70 exceeded its {_WALL_TIME_BUDGET_SECONDS}s hard budget")

    signal.signal(signal.SIGALRM, _budget_expired)
    signal.alarm(_WALL_TIME_BUDGET_SECONDS)
    try:
        report = train_v70(args)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)
    print(json.dumps(report, sort_keys=True))
    return 0 if report.get("passed") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "train_v70",
    "validate_v70_preregistration",
]
