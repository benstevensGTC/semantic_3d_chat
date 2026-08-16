"""Immutable protocol for the V54 fixed-prefix Gemma-4 PLE reader.

This experiment is intentionally independent of the failed Atlas and numeric
question-controller lines.  It keeps the accepted V54 258-token continuous
prefix byte-for-byte fixed and trains one rank-4, unmerged LoRA on Gemma-4's
per-layer input projection.  The user question never participates in scene
encoding or selection.
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

import torch

from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.evaluation.ple_reader_preregistration import (
    LORA_PARAMETER_COUNT,
    MODEL_ID,
    MODEL_REVISION,
    TARGET_MODULE,
    answer_only_wrong_prefix_objective,
    reader_lora_settings,
)

ARTIFACT: Final[str] = "gemma4_v54_fixed_prefix_ple_reader_v1"
CONFIG: Final[str] = "configs/experiments/gemma4_v54_fixed_prefix_ple_reader_v1.yaml"
RETENTION: Final[str] = (
    "configs/experiments/gemma4_v54_fixed_prefix_ple_reader_v1_retention.json"
)
RUNTIME_CONFIG: Final[str] = "configs/runtime/gemma4_v54.yaml"
BASE_CHECKPOINT: Final[str] = (
    "data_gemma4/checkpoints/gemma4_v54_semantic_greedy_gate/update_000"
)
PREFIX_CACHE: Final[str] = "data_gemma4/scene_tokens/v56_question_control_full_prefixes"
TRAIN_QA: Final[str] = "data_gemma4/training/v62_pair_disjoint/train.jsonl"
VALIDATION_QUESTIONS: Final[str] = "reports/gemma4/questions/v62_internal_validation.json"
VALIDATION_REFERENCES: Final[str] = (
    "reports/gemma4/scorer_only/v62_internal_validation_references.json"
)
BASELINE_PREDICTIONS: Final[str] = (
    "reports/gemma4/predictions/v62_v54_no_control_internal_validation.jsonl"
)
PREREGISTRATION: Final[str] = (
    "reports/gemma4/metrics/gemma4_v54_fixed_prefix_ple_reader_v1_preregistration.json"
)
SMOKE_REPORT: Final[str] = (
    "reports/gemma4/metrics/gemma4_v54_fixed_prefix_ple_reader_v1_smoke.json"
)
RESULT_REPORT: Final[str] = (
    "reports/gemma4/metrics/gemma4_v54_fixed_prefix_ple_reader_v1_result.json"
)
OUTPUT_CHECKPOINT: Final[str] = (
    "data_gemma4/checkpoints/gemma4_v54_fixed_prefix_ple_reader_v1"
)

TRAIN_SCENES: Final[tuple[str, ...]] = tuple(
    f"scene_{index:06d}"
    for index in (*range(11, 25), *range(31, 39), 53, 54)
)
VALIDATION_SCENES: Final[tuple[str, ...]] = tuple(
    f"scene_{index:06d}"
    for index in (*range(39, 53), 55, 56)
)

_EXPECTED_INPUT_HASHES: Final[dict[str, str]] = {
    RUNTIME_CONFIG: "891c58faaaa5fcd2ed76c7e3871f14c5d8c5ae2e05d9fa4ddd5193773d40e56b",
    f"{BASE_CHECKPOINT}/adapter.safetensors": (
        "6c627f0a0d9efb7100489c24cbd4acafe10456b79280a64a85399a21cb541daf"
    ),
    f"{BASE_CHECKPOINT}/metadata.json": (
        "db1435f8d38ca587e34dcd55dc4d37532efc0504bfb62bc115838dc0ab7a7ece"
    ),
    f"{BASE_CHECKPOINT}/runtime_metadata.json": (
        "807515461c71b08c08dfbd08a184a653e791413748530fa69402512eca6f6fdd"
    ),
    f"{PREFIX_CACHE}/manifest.json": (
        "5a288a7fef65a957ba7b20132c63380cfadc7edbc37b32c1885037f939b9db61"
    ),
    TRAIN_QA: "84b99385fadc5d06e44465ada5902f56131192298ca1539373dc3b334608cbf1",
    VALIDATION_QUESTIONS: (
        "078f65e1402e6e382a7bfdb2ad4b8a65d58e3164705a8a46cd222503aa201052"
    ),
    VALIDATION_REFERENCES: (
        "4202e777ee57ab3f7da329f15589e56b8b0464b782fb4d856dd1a3281ff3115c"
    ),
    BASELINE_PREDICTIONS: (
        "df66de37e918ba068fbcd91308803746122c938ccccadf063d1b1343f1a4c902"
    ),
    RETENTION: "0b2c48236e085960811ac6c9be94440814a141fdc05ed92c1e8f498a2c04f3cb",
}

_IMPLEMENTATION_PATHS: Final[tuple[str, ...]] = (
    "src/semantic_3d_chat/evaluation/fixed_prefix_ple_v54_preregistration.py",
    "src/semantic_3d_chat/training/train_fixed_prefix_ple_v54.py",
    "scripts/run_gemma4_v54_fixed_prefix_ple_reader.sh",
    CONFIG,
    RETENTION,
    "tests/test_fixed_prefix_ple_v54.py",
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


def implementation_source_hashes() -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative in _IMPLEMENTATION_PATHS:
        source = _resolve(relative)
        if not source.is_file() or source.is_symlink():
            raise FileNotFoundError(f"PLE-V54 implementation source missing: {relative}")
        hashes[relative] = sha256_file(source)
    return hashes


def authenticate_inputs() -> dict[str, str]:
    observed: dict[str, str] = {}
    for relative, expected in _EXPECTED_INPUT_HASHES.items():
        source = _resolve(relative)
        if not source.is_file() or source.is_symlink():
            raise FileNotFoundError(f"PLE-V54 pinned input missing or unsafe: {relative}")
        digest = sha256_file(source)
        if digest != expected:
            raise ValueError(
                f"PLE-V54 pinned input changed: {relative}: {digest} != {expected}"
            )
        observed[relative] = digest
    return observed


def validate_objective(
    correct: torch.Tensor,
    wrong: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Apply the one locked PLE-V54 answer and prefix-selectivity objective."""

    return answer_only_wrong_prefix_objective(
        correct,
        wrong,
        margin=0.25,
        answer_ce_weight=1.0,
        wrong_prefix_weight=1.0,
    )


def _validate_prefix_manifest() -> dict[str, Any]:
    path = _resolve(f"{PREFIX_CACHE}/manifest.json")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError("PLE-V54 prefix manifest must be an object")
    scenes = value.get("scenes")
    all_scenes = set(TRAIN_SCENES) | set(VALIDATION_SCENES)
    if (
        value.get("artifact") != "question_independent_scene_prefix_cache_v1"
        or value.get("question_inputs_used") is not False
        or value.get("question_dependent_scene_retrieval") is not False
        or value.get("complete_scene_prefixes") is not True
        or value.get("environmental_text_inputs") != []
        or value.get("base_checkpoint_sha256")
        != "3e128b40c1b73bb32750285679cda6b1bea364e67465e986a94a81dfc95e81e8"
        or value.get("base_runtime_config_sha256")
        != "714c60ce9ccb1dff69c72f6618f8afb6f31bc60a830b5ee0fb794fedaa8a321e"
        or not isinstance(scenes, Mapping)
        or set(scenes) != all_scenes
    ):
        raise ValueError("PLE-V54 fixed-prefix manifest contract changed")
    for scene_id, raw in scenes.items():
        if not isinstance(raw, Mapping):
            raise TypeError(f"PLE-V54 prefix entry is invalid: {scene_id}")
        if raw.get("shape") != [1, 258, 1536] or raw.get("dtype") != "bfloat16":
            raise ValueError(f"PLE-V54 prefix shape/dtype changed: {scene_id}")
    return dict(value)


def build_preregistration() -> dict[str, Any]:
    """Build the exact one-arm contract; no model or answer-bearing row is loaded."""

    input_hashes = authenticate_inputs()
    prefix_manifest = _validate_prefix_manifest()
    return {
        "schema_version": 1,
        "artifact": ARTIFACT,
        "status": "locked_before_gradient_smoke_and_single_training_run",
        "research_question": (
            "Can a rank-4 adapter on Gemma-4's per-layer model projection improve "
            "scene-specific reading of the fixed complete V54 continuous prefix?"
        ),
        "independence": {
            "depends_on_failed_atlas": False,
            "depends_on_failed_question_controllers": False,
            "base": "accepted_v54_258_token_question_independent_prefix",
            "base_prefix_tokens": 258,
            "base_scene_latents": 256,
            "base_hidden_dimension": 1536,
            "all_scene_tokens_fit_inside_sliding_window_with_locked_prompts": True,
        },
        "model": {
            "model_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "local_files_only": True,
            "target_module": TARGET_MODULE,
            "target_projection_shape": [8960, 1536],
        },
        "trainable_surface": {
            "type": "unmerged_fp32_lora",
            "rank": 4,
            "alpha": 8.0,
            "dropout": 0.0,
            "parameter_count": LORA_PARAMETER_COUNT,
            "exact_target_modules": [TARGET_MODULE],
            "base_model_frozen": True,
            "v54_scene_stack_frozen": True,
            "prefix_cache_frozen": True,
            "no_merge": True,
        },
        "data": {
            "training_scenes": list(TRAIN_SCENES),
            "validation_scenes": list(VALIDATION_SCENES),
            "scene_disjoint": set(TRAIN_SCENES).isdisjoint(VALIDATION_SCENES),
            "training_rows": 576,
            "validation_rows": 384,
            "training_changed_sides": 80,
            "validation_changed_sides": 52,
            "prefix_manifest_scene_count": prefix_manifest["scene_count"],
            "test_split_accessed": False,
            "oracle_runtime_access": False,
        },
        "objective": {
            "answer_token_normalized_ce_weight": 1.0,
            "same_question_wrong_prefix_hinge_weight": 1.0,
            "same_question_wrong_prefix_margin_nats_per_token": 0.25,
            "hinge_rows": "changed_counterfactual_sides_only",
            "wrong_prefix": "paired_scene_complete_258_token_prefix",
            "labels_before_answer_suffix": -100,
            "retention_next_token_kl_weight_per_update": 0.2,
        },
        "optimization": {
            "seed": 720054,
            "optimizer": "adamw",
            "learning_rate": 0.0003,
            "weight_decay": 0.0,
            "gradient_accumulation": 4,
            "maximum_updates": 40,
            "gradient_clip_l2": 1.0,
            "one_arm_only": True,
            "intermediate_selection": False,
            "decoder_gradient_checkpointing": True,
            "adapter_dtype": "float32",
            "base_dtype": "bfloat16",
        },
        "selection": {
            "split": "scene_disjoint_internal_validation",
            "teacher_forced_rows": 384,
            "greedy_subset_rule": "first_6_question_manifest_rows_per_validation_scene",
            "greedy_subset_rows": 96,
            "validation_answer_nll_improvement_minimum": 0.03,
            "changed_wrong_prefix_positive_margin_rate_minimum": 0.65,
            "changed_wrong_prefix_positive_margin_rate_delta_minimum": 0.10,
            "changed_pair_complete_unit_delta_minimum": 3,
            "greedy_exact_accuracy_delta_minimum": 0.02,
            "retention_mean_ce_increase_nats_maximum": 0.03,
            "retention_mean_kl_nats_maximum": 0.02,
            "retention_next_token_top1_agreement_minimum": 0.98,
            "all_gates_required": True,
            "candidate_count": 1,
        },
        "runtime_contract": {
            "prefix_computed_before_question": True,
            "identical_prefix_for_unchanged_scene": True,
            "question_dependent_retrieval": False,
            "environmental_text_inputs": [],
            "oracle_forbidden": True,
            "runtime_files": [
                "sanitized_continuous_v54_prefix_or_map",
                "frozen_v54_checkpoint",
                "ple_reader_adapter",
                "local_gemma4_snapshot",
            ],
        },
        "publication": {
            "one_terminal_report_create_once": True,
            "checkpoint_published_only_if_every_gate_passes": True,
            "failed_run_publishes_no_checkpoint": True,
            "no_intermediate_runtime_checkpoints": True,
        },
        "pinned_input_hashes": input_hashes,
        "implementation_source_hashes": implementation_source_hashes(),
        "reader_lora_contract": reader_lora_settings().contract(),
    }


def _serialized(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n"
    ).encode()


def write_preregistration(path: str | Path = PREREGISTRATION) -> tuple[Path, str]:
    destination = _resolve(path)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"PLE-V54 preregistration already exists: {destination}")
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
        raise FileNotFoundError("PLE-V54 preregistration is missing or unsafe")
    observed = json.loads(source.read_text(encoding="utf-8"))
    expected = build_preregistration()
    if observed != expected:
        raise ValueError("PLE-V54 preregistration differs from current pinned sources")
    return {
        "path": str(source.relative_to(PROJECT_ROOT)),
        "sha256": sha256_file(source),
        "artifact": observed["artifact"],
        "status": observed["status"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=PREREGISTRATION)
    parser.add_argument("--authenticate", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.authenticate:
        result = authenticate_preregistration(args.output)
    else:
        path, digest = write_preregistration(args.output)
        result = {"path": str(path), "sha256": digest}
    print(json.dumps(result, sort_keys=True))
    return 0


__all__ = [
    "ARTIFACT",
    "BASE_CHECKPOINT",
    "OUTPUT_CHECKPOINT",
    "PREFIX_CACHE",
    "PREREGISTRATION",
    "RESULT_REPORT",
    "SMOKE_REPORT",
    "TRAIN_SCENES",
    "VALIDATION_SCENES",
    "authenticate_inputs",
    "authenticate_preregistration",
    "build_preregistration",
    "implementation_source_hashes",
    "sha256_file",
    "validate_objective",
    "write_preregistration",
]


if __name__ == "__main__":
    raise SystemExit(main())
