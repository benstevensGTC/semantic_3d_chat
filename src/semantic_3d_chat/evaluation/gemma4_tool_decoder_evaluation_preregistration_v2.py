"""Supplemental immutable evaluation contract for tool-decoder V2.

The original architecture preregistration remains byte-exact.  This additive
contract seals bounded evaluation rows and the all-heldout early gate before a
full model is loaded or any V2 optimizer update occurs.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from semantic_3d_chat.config import PROJECT_ROOT, load_config
from semantic_3d_chat.training.gemma4_tool_decoder_v2_data import (
    CAUSAL_VALIDATION_SAMPLE_IDS_SHA256,
    PRIMARY_VALIDATION_SAMPLE_IDS_SHA256,
    causal_validation_indices_v2,
    load_tool_decoder_dataset_v2,
    primary_validation_indices_v2,
)

DEFAULT_OUTPUT = Path(
    "reports/gemma4/metrics/"
    "gemma4_embodied_tool_decoder_evaluation_preregistration_v2.json"
)
_CONFIG = "configs/experiments/gemma4_embodied_tool_decoder_v2.yaml"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sample_id_sha256(values: list[str]) -> str:
    return hashlib.sha256(
        json.dumps(values, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _distribution(dataset: Any, indices: tuple[int, ...]) -> dict[str, Any]:
    def counts(attribute: str) -> dict[str, int]:
        return dict(
            sorted(Counter(getattr(dataset.samples[index], attribute) for index in indices).items())
        )

    return {
        "scenes": counts("scene_id"),
        "families": counts("family"),
        "actions": counts("action_name"),
    }


def build_evaluation_preregistration_v2() -> dict[str, Any]:
    config = load_config(_CONFIG)
    dataset = load_tool_decoder_dataset_v2(config)
    causal = causal_validation_indices_v2(dataset)
    primary = primary_validation_indices_v2(dataset)
    causal_ids = [dataset.samples[index].sample_id for index in causal]
    primary_ids = [dataset.samples[index].sample_id for index in primary]
    all_ids = [dataset.samples[index].sample_id for index in dataset.validation_indices]
    if (
        _sample_id_sha256(causal_ids) != CAUSAL_VALIDATION_SAMPLE_IDS_SHA256
        or _sample_id_sha256(primary_ids) != PRIMARY_VALIDATION_SAMPLE_IDS_SHA256
    ):
        raise ValueError("V2 evaluation sample digest changed during preregistration")
    return {
        "schema_version": 2,
        "artifact": "gemma4_embodied_tool_decoder_evaluation_preregistration_v2",
        "status": "sealed_before_full_model_load_or_training",
        "parent_architecture_preregistration_sha256": (
            "0e1e41a6af2830f9b36a8711fb0649246e96254a88cdcc76b97dcb06ee3f82f4"
        ),
        "cpu_preflight_sha256": (
            "412f1d8bb9804b2d38b0335c985225c9cf1e4226758858cee18d906dc5f742e7"
        ),
        "clearance_cache_sha256": (
            "658822707389e67481fa59b035a7e7f19c360487b19d3157b80bc23ede1db048"
        ),
        "trace_rows_sha256": dataset.trace_rows_sha256,
        "prefix_inventory_sha256": dataset.prefix_inventory_sha256,
        "all_heldout_teacher_forced": {
            "sample_count": len(all_ids),
            "scene_count": 8,
            "sample_ids_sha256": _sample_id_sha256(all_ids),
            "sample_ids": all_ids,
            "metrics": [
                "answer_token_nll",
                "answer_token_accuracy",
                "exact_sequence_accuracy",
            ],
            "early_before_any_greedy_generation": True,
            "hard_gates": {
                "answer_token_nll_maximum": 2.0,
                "answer_token_accuracy_minimum": 0.80,
                "exact_sequence_accuracy_minimum": 0.30,
            },
        },
        "greedy_generation": {
            "decoder": "deterministic_argmax",
            "max_new_tokens": 24,
            "schema_validation": "strict_existing_robot_tool_validator",
            "primary": {
                "maximum_rows_per_scene_family": 16,
                "sample_count": len(primary),
                "sample_ids_sha256": PRIMARY_VALIDATION_SAMPLE_IDS_SHA256,
                "sample_ids": primary_ids,
                "distribution": _distribution(dataset, primary),
            },
            "causal_controls": {
                "rows_per_scene_family": 8,
                "sample_count_per_condition": len(causal),
                "sample_ids_sha256": CAUSAL_VALIDATION_SAMPLE_IDS_SHA256,
                "sample_ids": causal_ids,
                "distribution": _distribution(dataset, causal),
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
            },
            "selection_algorithm": (
                "sort validation rows by sample_id within each scene/family/action; "
                "round-robin lexicographically sorted actions; then globally sort indices"
            ),
            "total_unique_sequences": 4416,
        },
        "quality_gates": {
            "primary_exact_json_accuracy_minimum": 0.60,
            "primary_valid_schema_rate_minimum": 0.95,
            "primary_tool_accuracy_minimum": 0.80,
            "primary_turn_sign_accuracy_minimum": 0.80,
            "primary_argument_mae_normalized_maximum": 0.25,
            "unsafe_motion_count": 0,
            "wrong_and_zero_scene_tool_accuracy_drop_minimum": 0.05,
            "wrong_and_zero_robot_tool_accuracy_drop_minimum": 0.05,
            "wrong_and_zero_target_tool_accuracy_drop_minimum": 0.10,
            "wrong_and_zero_clearance_safe_proposal_drop_minimum": 0.05,
        },
        "execution": {
            "full_model_loaded": False,
            "mps_used": False,
            "optimizer_steps": 0,
            "greedy_generations": 0,
            "checkpoint_published": False,
        },
        "environmental_text_inputs": [],
        "oracle_inputs_at_runtime": False,
    }


def write_evaluation_preregistration_v2(
    output: str | Path = DEFAULT_OUTPUT,
) -> tuple[Path, str]:
    destination = Path(output)
    if not destination.is_absolute():
        destination = PROJECT_ROOT / destination
    if destination.exists():
        raise FileExistsError("V2 evaluation preregistration is create-once")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                build_evaluation_preregistration_v2(),
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
    "build_evaluation_preregistration_v2",
    "write_evaluation_preregistration_v2",
]
