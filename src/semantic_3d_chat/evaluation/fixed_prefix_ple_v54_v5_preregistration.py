"""Immutable V5 protocol for a scene-selective fixed-prefix Gemma-4 reader.

V4 reduced held-out answer NLL but failed every scene-discrimination gate. V5
keeps the exact V54 prefix, model, rank-4 PLE surface, data split, and promotion
gates. Its single change is a preregistered pair-symmetric training protocol
that gives the same-question wrong-prefix objective substantially more weight
and exposes every broad training row exactly once.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.evaluation.fixed_prefix_ple_v54_v4_preregistration import (
    authenticate_preregistration as authenticate_v4_preregistration,
)

ARTIFACT: Final[str] = "gemma4_v54_fixed_prefix_ple_reader_v5"
CONFIG: Final[str] = "configs/experiments/gemma4_v54_fixed_prefix_ple_reader_v5.yaml"
PREREGISTRATION: Final[str] = (
    "reports/gemma4/metrics/gemma4_v54_fixed_prefix_ple_reader_v5_preregistration.json"
)
SMOKE_REPORT: Final[str] = "reports/gemma4/metrics/gemma4_v54_fixed_prefix_ple_reader_v5_smoke.json"
RESULT_REPORT: Final[str] = (
    "reports/gemma4/metrics/gemma4_v54_fixed_prefix_ple_reader_v5_result.json"
)
OUTPUT_CHECKPOINT: Final[str] = "data_gemma4/checkpoints/gemma4_v54_fixed_prefix_ple_reader_v5"

V4_PREREGISTRATION: Final[str] = (
    "reports/gemma4/metrics/gemma4_v54_fixed_prefix_ple_reader_v4_preregistration.json"
)
V4_SMOKE: Final[str] = "reports/gemma4/metrics/gemma4_v54_fixed_prefix_ple_reader_v4_smoke.json"
V4_RESULT: Final[str] = "reports/gemma4/metrics/gemma4_v54_fixed_prefix_ple_reader_v4_result.json"
V4_PREREGISTRATION_SHA256: Final[str] = (
    "34b4576a6ced7003c916c5dc3deabecf8e6e70a0e39bcc8329d039fd00ef3d59"
)
V4_SMOKE_SHA256: Final[str] = "4d76f2f6de14fd5d1e5130d50fded5627418cec83fb5f505d57db49f9244d345"
V4_RESULT_SHA256: Final[str] = "ea16351a39ba1e0eb7441a4c8f371466b2f413ed6d352bdfb745e1f047766139"
V4_SOURCE_HASHES: Final[dict[str, str]] = {
    "src/semantic_3d_chat/evaluation/fixed_prefix_ple_v54_v4_preregistration.py": (
        "cbcb629593aeb1608a68575e1bf67df12534180f8c9ea207c6f86085c7270595"
    ),
    "src/semantic_3d_chat/training/train_fixed_prefix_ple_v54_v4.py": (
        "6e023ac794b73f063cc89c55955c7b45ecf774a0b0d93fa0750d70487961a9ed"
    ),
    "scripts/run_gemma4_v54_fixed_prefix_ple_reader_v4.sh": (
        "1271b5f4aa72becb4ae84c2cc6eb845867d1fa9c3fe28ce7349c0e213a793297"
    ),
    "tests/test_fixed_prefix_ple_v54_v4.py": (
        "31073b89abdac623c0c85fe964c959e37ff4f942216c1a643089b109db7e11f2"
    ),
}

_IMPLEMENTATION_PATHS: Final[tuple[str, ...]] = (
    "src/semantic_3d_chat/evaluation/fixed_prefix_ple_v54_v5_preregistration.py",
    "src/semantic_3d_chat/training/train_fixed_prefix_ple_v54_v5.py",
    "scripts/run_gemma4_v54_fixed_prefix_ple_reader_v5.sh",
    CONFIG,
    "tests/test_fixed_prefix_ple_v54_v5.py",
)


def _resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return (value if value.is_absolute() else PROJECT_ROOT / value).resolve()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with _resolve(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def v5_implementation_hashes() -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in _IMPLEMENTATION_PATHS:
        source = _resolve(relative)
        if not source.is_file() or source.is_symlink():
            raise FileNotFoundError(f"PLE-V54 V5 implementation missing: {relative}")
        result[relative] = sha256_file(relative)
    return result


def authenticate_v4_terminal() -> dict[str, Any]:
    prereg = authenticate_v4_preregistration(V4_PREREGISTRATION)
    expected_evidence = {
        V4_PREREGISTRATION: V4_PREREGISTRATION_SHA256,
        V4_SMOKE: V4_SMOKE_SHA256,
        V4_RESULT: V4_RESULT_SHA256,
    }
    for relative, expected in expected_evidence.items():
        if sha256_file(relative) != expected:
            raise ValueError(f"PLE-V54 V4 evidence changed: {relative}")
    for relative, expected in V4_SOURCE_HASHES.items():
        if sha256_file(relative) != expected:
            raise ValueError(f"PLE-V54 V4 source changed: {relative}")
    result = json.loads(_resolve(V4_RESULT).read_text(encoding="utf-8"))
    baseline = result.get("selection", {}).get("baseline_teacher", {})
    candidate = result.get("selection", {}).get("candidate_teacher", {})
    if (
        prereg["sha256"] != V4_PREREGISTRATION_SHA256
        or result.get("status") != "failed_no_checkpoint"
        or result.get("passed") is not False
        or result.get("checkpoint_published") is not False
        or result.get("checkpoint") is not None
        or baseline.get("answer_nll_mean") != 3.2358317994512618
        or candidate.get("answer_nll_mean") != 2.036423228079608
        or baseline.get("changed_positive_margin_sides") != 30
        or candidate.get("changed_positive_margin_sides") != 28
        or baseline.get("changed_complete_units") != 12
        or candidate.get("changed_complete_units") != 10
        or _resolve("data_gemma4/checkpoints/gemma4_v54_fixed_prefix_ple_reader_v4").exists()
    ):
        raise ValueError("PLE-V54 V4 terminal failure contract changed")
    return result


def build_preregistration() -> dict[str, Any]:
    v4 = authenticate_v4_terminal()
    return {
        "schema_version": 1,
        "artifact": ARTIFACT,
        "status": "locked_before_v5_gradient_smoke_and_single_training_run",
        "v4_diagnostic": {
            "preregistration_sha256": V4_PREREGISTRATION_SHA256,
            "smoke_sha256": V4_SMOKE_SHA256,
            "result_sha256": V4_RESULT_SHA256,
            "status": v4["status"],
            "checkpoint_absent": True,
            "answer_nll_before": 3.2358317994512618,
            "answer_nll_after": 2.036423228079608,
            "positive_margin_sides_before": 30,
            "positive_margin_sides_after": 28,
            "complete_pair_units_before": 12,
            "complete_pair_units_after": 10,
            "interpretation": "generic answer reading improved while scene discrimination regressed",
        },
        "unchanged_from_v4": {
            "base": "structurally_authenticated_v54_complete_question_independent_continuous_prefix",
            "fixed_prefix_shape": [1, 258, 1536],
            "scene_latents": 256,
            "model_id": "google/gemma-4-E2B-it",
            "model_revision": "3e22461f65e89153144f8adb70e3b8c2cc9845a7",
            "local_files_only": True,
            "target_module": "model.language_model.per_layer_model_projection",
            "rank": 4,
            "alpha": 8.0,
            "dropout": 0.0,
            "trainable_parameter_count": 41_984,
            "base_model_frozen": True,
            "v54_scene_stack_frozen": True,
            "prefix_cache_frozen": True,
            "training_scenes": 24,
            "training_rows": 576,
            "validation_scenes": 16,
            "validation_rows": 384,
            "validation_changed_sides": 52,
            "question_dependent_retrieval": False,
            "environmental_text_inputs": [],
            "oracle_runtime_access": False,
        },
        "single_v5_arm": {
            "seed": 720054,
            "optimizer": "adamw",
            "weight_decay": 0.0,
            "updates": 80,
            "pair_cycles": 2,
            "all_40_changed_units_once_per_cycle": True,
            "all_496_broad_rows_exactly_once": True,
            "broad_rows_by_update": {"updates_1_through_64": 6, "updates_65_through_80": 7},
            "pair_loss": {
                "correct_answer_ce_weight": 0.5,
                "symmetric_wrong_prefix_hinge_weight": 4.0,
                "margin_nats_per_answer_token": 0.5,
                "both_counterfactual_sides_in_every_update": True,
            },
            "broad_answer_ce_weight": 0.5,
            "retention_next_token_kl_weight": 0.5,
            "linear_warmup_updates": 8,
            "peak_learning_rate": 0.0001,
            "cosine_decay_minimum_learning_rate": 0.00001,
            "gradient_clip_l2": 1.0,
            "decoder_gradient_checkpointing": True,
            "answer_logit_positions_only": True,
            "evaluation_microbatch_size": 1,
            "intermediate_selection": False,
            "intermediate_checkpoint": False,
            "candidate_state_for_all_gates": "single_final_state_after_update_80_only",
            "best_loss_selection": False,
            "post_hoc_state_selection": False,
            "one_arm_only": True,
        },
        "unchanged_v4_promotion_gates": {
            "validation_answer_nll_improvement_minimum": 0.03,
            "changed_wrong_prefix_positive_margin_rate_minimum": 0.65,
            "changed_wrong_prefix_positive_margin_rate_delta_minimum": 0.10,
            "changed_pair_complete_unit_delta_minimum": 3,
            "greedy_exact_accuracy_delta_minimum": 0.02,
            "retention_mean_ce_increase_nats_maximum": 0.03,
            "retention_mean_kl_nats_maximum": 0.02,
            "retention_next_token_top1_agreement_minimum": 0.98,
            "all_required": True,
            "failed_run_publishes_no_checkpoint": True,
        },
        "deferred_holdout": {
            "scene_ids": [f"scene_{index:06d}" for index in range(57, 63)],
            "question_count": 216,
            "prefixes_not_compiled_before_internal_selection_passes": True,
            "qa_not_accessed_for_v5_design_training_or_internal_selection": True,
            "evaluate_only_if_all_internal_v4_gates_pass": True,
        },
        "final_split": {
            "scene_ids": [f"scene_{index:06d}" for index in range(25, 31)],
            "accessed": False,
            "forbidden_during_v5_internal_selection": True,
        },
        "v4_source_hashes": V4_SOURCE_HASHES,
        "v5_implementation_source_hashes": v5_implementation_hashes(),
        "paths": {
            "config": CONFIG,
            "smoke": SMOKE_REPORT,
            "result": RESULT_REPORT,
            "checkpoint": OUTPUT_CHECKPOINT,
        },
    }


def _serialized(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    ).encode()


def write_preregistration(path: str | Path = PREREGISTRATION) -> tuple[Path, str]:
    destination = _resolve(path)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"PLE-V54 V5 preregistration exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = _serialized(build_preregistration())
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination, hashlib.sha256(payload).hexdigest()


def authenticate_preregistration(path: str | Path = PREREGISTRATION) -> dict[str, Any]:
    source = _resolve(path)
    if not source.is_file() or source.is_symlink():
        raise FileNotFoundError("PLE-V54 V5 preregistration is missing or unsafe")
    observed = json.loads(source.read_text(encoding="utf-8"))
    if observed != build_preregistration():
        raise ValueError("PLE-V54 V5 preregistration differs from pinned sources")
    return {
        "artifact": ARTIFACT,
        "path": str(source.relative_to(PROJECT_ROOT)),
        "sha256": sha256_file(source),
        "status": observed["status"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=PREREGISTRATION)
    parser.add_argument("--authenticate", action="store_true")
    args = parser.parse_args(argv)
    if args.authenticate:
        result = authenticate_preregistration(args.output)
    else:
        path, digest = write_preregistration(args.output)
        result = {"path": str(path), "sha256": digest}
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
