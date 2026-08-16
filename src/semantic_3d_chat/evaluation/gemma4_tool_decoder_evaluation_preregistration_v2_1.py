"""Finite resource-only V2.1 evaluation contract sealed before heavy execution."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Final

from semantic_3d_chat.config import PROJECT_ROOT, load_config
from semantic_3d_chat.training.gemma4_tool_decoder_v2_data import (
    CAUSAL_VALIDATION_SAMPLE_IDS_SHA256,
    GREEDY_CONTROL_SAMPLE_IDS_SHA256_V2_1,
    causal_validation_indices_v2,
    greedy_control_validation_indices_v2_1,
    load_tool_decoder_dataset_v2,
)

DEFAULT_OUTPUT: Final[Path] = Path(
    "reports/gemma4/metrics/"
    "gemma4_embodied_tool_decoder_evaluation_preregistration_v2_1.json"
)
PARENT_EVALUATION_PATH: Final[Path] = Path(
    "reports/gemma4/metrics/"
    "gemma4_embodied_tool_decoder_evaluation_preregistration_v2.json"
)
PARENT_EVALUATION_SHA256: Final[str] = (
    "081bb382264ae4e052473f31c928ef376a690833c3b7a831f37c0bcc648b1f6e"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ids_sha256(values: list[str]) -> str:
    return hashlib.sha256(
        json.dumps(values, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _distribution(dataset: Any, indices: tuple[int, ...]) -> dict[str, Any]:
    def count(attribute: str) -> dict[str, int]:
        return dict(
            sorted(
                Counter(
                    getattr(dataset.samples[index], attribute) for index in indices
                ).items()
            )
        )

    return {
        "scenes": count("scene_id"),
        "families": count("family"),
        "actions": count("action_name"),
    }


def build_evaluation_preregistration_v2_1() -> dict[str, Any]:
    """Bind exact rows, costs, metrics, gates, and execution non-occurrence."""

    parent = PROJECT_ROOT / PARENT_EVALUATION_PATH
    if _sha256(parent) != PARENT_EVALUATION_SHA256:
        raise ValueError("V2.1 parent evaluation preregistration changed")
    dataset = load_tool_decoder_dataset_v2(
        load_config("configs/experiments/gemma4_embodied_tool_decoder_v2.yaml")
    )
    teacher_causal = causal_validation_indices_v2(dataset)
    greedy_controls = greedy_control_validation_indices_v2_1(dataset)
    all_heldout_ids = [
        dataset.samples[index].sample_id for index in dataset.validation_indices
    ]
    teacher_causal_ids = [dataset.samples[index].sample_id for index in teacher_causal]
    greedy_control_ids = [dataset.samples[index].sample_id for index in greedy_controls]
    if (
        _ids_sha256(teacher_causal_ids) != CAUSAL_VALIDATION_SAMPLE_IDS_SHA256
        or _ids_sha256(greedy_control_ids)
        != GREEDY_CONTROL_SAMPLE_IDS_SHA256_V2_1
    ):
        raise ValueError("V2.1 fixed evaluation sample identity changed")
    return {
        "schema_version": "2.1",
        "artifact": "gemma4_embodied_tool_decoder_evaluation_preregistration_v2_1",
        "status": "resource_bounded_sealed_before_full_model_load_or_training",
        "supersedes_resource_plan_only": {
            "path": str(PARENT_EVALUATION_PATH),
            "sha256": PARENT_EVALUATION_SHA256,
            "reason": (
                "The V2 plan required 4,416 greedy sequences and was finite but too "
                "expensive for the local Mac; no V2 sequence was generated."
            ),
        },
        "unchanged_architecture_preregistration_sha256": (
            "0e1e41a6af2830f9b36a8711fb0649246e96254a88cdcc76b97dcb06ee3f82f4"
        ),
        "cpu_preflight_sha256": (
            "412f1d8bb9804b2d38b0335c985225c9cf1e4226758858cee18d906dc5f742e7"
        ),
        "clearance_cache_sha256": dataset.clearance_cache_sha256,
        "trace_rows_sha256": dataset.trace_rows_sha256,
        "prefix_inventory_sha256": dataset.prefix_inventory_sha256,
        "all_heldout_primary_teacher_forced": {
            "sample_count": len(all_heldout_ids),
            "scene_count": 8,
            "sample_ids_sha256": _ids_sha256(all_heldout_ids),
            "sample_ids": all_heldout_ids,
            "metrics": [
                "answer_token_nll",
                "answer_token_accuracy",
                "exact_sequence_accuracy",
                "teacher_forced_argmax_valid_schema_rate",
                "teacher_forced_argmax_canonical_rate",
                "teacher_forced_argmax_tool_accuracy",
            ],
            "early_gate_before_any_greedy_generation": {
                "answer_token_nll_maximum": 2.0,
                "answer_token_accuracy_minimum": 0.80,
                "exact_sequence_accuracy_minimum": 0.30,
                "teacher_forced_argmax_valid_schema_rate_minimum": 0.80,
                "teacher_forced_argmax_tool_accuracy_minimum": 0.70,
            },
        },
        "teacher_forced_causal_controls": {
            "sample_count_per_condition": len(teacher_causal),
            "condition_count": 9,
            "sample_ids_sha256": CAUSAL_VALIDATION_SAMPLE_IDS_SHA256,
            "sample_ids": teacher_causal_ids,
            "distribution": _distribution(dataset, teacher_causal),
            "conditions": [
                "primary",
                "wrong_scene",
                "zero_scene",
                "wrong_robot",
                "zero_robot",
                "wrong_target",
                "zero_target",
                "wrong_clearance",
                "zero_clearance",
            ],
            "forward_count": len(teacher_causal) * 9,
            "primary_rows_reused_from_all_heldout": True,
            "additional_unique_forward_count": len(teacher_causal) * 8,
            "causal_gates": {
                "wrong_and_zero_scene_nll_increase_minimum": 0.01,
                "wrong_and_zero_robot_targeted_nll_increase_minimum": 0.01,
                "wrong_and_zero_target_targeted_nll_increase_minimum": 0.02,
                "wrong_and_zero_clearance_targeted_nll_increase_minimum": 0.01,
            },
        },
        "bounded_greedy_generation": {
            "decoder": "deterministic_argmax",
            "maximum_new_tokens": 24,
            "primary": {
                "sample_count": len(teacher_causal),
                "rows_per_scene_family": 8,
                "sample_ids_sha256": CAUSAL_VALIDATION_SAMPLE_IDS_SHA256,
                "sample_ids": teacher_causal_ids,
                "distribution": _distribution(dataset, teacher_causal),
            },
            "altered_controls": {
                "sample_count_per_condition": len(greedy_controls),
                "rows_per_scene_family": 1,
                "condition_count": 8,
                "sample_ids_sha256": GREEDY_CONTROL_SAMPLE_IDS_SHA256_V2_1,
                "sample_ids": greedy_control_ids,
                "distribution": _distribution(dataset, greedy_controls),
            },
            "total_unique_sequences": len(teacher_causal) + len(greedy_controls) * 8,
            "hard_maximum_total_unique_sequences": 1024,
            "metrics": [
                "exact_json_accuracy",
                "valid_schema_rate",
                "canonical_json_rate",
                "tool_accuracy",
                "argument_mae_normalized",
                "turn_sign_accuracy",
                "collision_risk_rate",
                "greedy_output_change_rate_from_primary",
            ],
        },
        "resource_accounting": {
            "training_microbatches": 512,
            "optimizer_updates": 64,
            "teacher_forced_unique_evaluation_forwards": (
                len(all_heldout_ids) + len(teacher_causal) * 8
            ),
            "greedy_unique_sequences": len(teacher_causal)
            + len(greedy_controls) * 8,
            "greedy_maximum_decode_tokens": (
                len(teacher_causal) + len(greedy_controls) * 8
            )
            * 24,
        },
        "answer_tail_memory_contract": {
            "training_and_teacher_forcing_labels_passed_to_model": False,
            "labels_used_only_to_locate_contiguous_answer_suffix": True,
            "model_logits_to_keep": "answer_label_positions_minus_one",
            "selected_logits_shape": "[1,answer_token_count,vocabulary_size]",
            "full_sequence_vocabulary_logits_materialized_during_training": False,
            "cross_entropy_dtype": "float32",
            "token_normalized_objective_unchanged": True,
            "real_one_row_full_vs_tail_nll_equivalence_tolerance": 1e-6,
            "real_one_row_equivalence_required_before_optimizer_construction": True,
            "tail_gradient_required_before_optimizer_construction": True,
        },
        "execution": {
            "full_model_loaded": False,
            "mps_used": False,
            "optimizer_steps": 0,
            "teacher_forced_forwards": 0,
            "greedy_generations": 0,
            "checkpoint_published": False,
        },
        "environmental_text_inputs": [],
        "oracle_inputs_at_runtime": False,
    }


def write_evaluation_preregistration_v2_1(
    output: str | Path = DEFAULT_OUTPUT,
) -> tuple[Path, str]:
    destination = Path(output)
    if not destination.is_absolute():
        destination = PROJECT_ROOT / destination
    if destination.exists():
        raise FileExistsError("V2.1 evaluation preregistration is create-once")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                build_evaluation_preregistration_v2_1(),
                handle,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    finally:
        Path(temporary_name).unlink(missing_ok=True)
    return destination, _sha256(destination)


__all__ = [
    "DEFAULT_OUTPUT",
    "build_evaluation_preregistration_v2_1",
    "write_evaluation_preregistration_v2_1",
]
