"""Exact eight-update V44-u8 retention-repair pilot for V45.

V45 is deliberately train-only.  It constructs the original authenticated
V41 bundle, strictly overlays the exact stopped V44 update-8 adapter, and then
allows only ``block_cross_residual.w_o`` plus layer-14 query LoRA A/B to move.
Every update uses the same complete, pre-question scene caches for all sixteen
training scenes.  Validation, oracle, final-scene, selector, and promotion
access remain forbidden.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from semantic_3d_chat.chat.file_audit import FileAccessAudit
from semantic_3d_chat.config import PROJECT_ROOT, artifact_root, config_hash, load_config
from semantic_3d_chat.data.dataset import QARecord
from semantic_3d_chat.language.lora import tensor_state_sha256
from semantic_3d_chat.scene_encoder.block_cross_residual import BlockCrossResidual
from semantic_3d_chat.training.checkpointing import (
    RUNTIME_METADATA_FILENAME,
    TRAINING_METADATA_FILENAME,
    load_adapter_checkpoint,
    module_collection_state_sha256,
    runtime_checkpoint_metadata,
    save_adapter_checkpoint,
    save_optimizer_checkpoint,
    validate_runtime_checkpoint_metadata,
)
from semantic_3d_chat.training.pair_curriculum import (
    CounterfactualPairUnit,
    build_exact_question_pair_units,
)
from semantic_3d_chat.training.train_block_cross_v35 import (
    broad_answer_nll,
    build_v35_schedule,
    current_scene_tokens,
    load_v35_train_qa_records,
    paired_cross_prefix_objective,
    v35_settings,
)
from semantic_3d_chat.training.train_environmental_sidecar_v33 import (
    assert_deferred_final_scenes_absent,
)
from semantic_3d_chat.training.train_joint_block_cross_v36 import (
    training_broad_nll,
    training_greedy_metrics,
    v36_broad_calibration_records,
)
from semantic_3d_chat.training.train_joint_pair_v30 import require_approved_v29_source
from semantic_3d_chat.training.train_joint_pair_v31 import v31_contract
from semantic_3d_chat.training.train_joint_scene_readout_v44 import (
    _PARAMETER_COUNTS,
    _PARAMETER_NAMES,
    _PARAMETER_SHAPES,
    assert_v44_trainable_surface,
    block_source_stack_state_sha256,
    freeze_for_v44,
    frozen_v44_state_sha256,
    source_prefix_trust_penalty,
    v44_contract,
)
from semantic_3d_chat.training.train_joint_scene_readout_v44 import (
    _SOURCE_AUTHORIZED_SHA256 as _V41_AUTHORIZED_SHA256,
)
from semantic_3d_chat.training.train_joint_scene_readout_v44 import (
    _SOURCE_CHECKPOINT as _V41_CONSTRUCTION_SOURCE,
)
from semantic_3d_chat.training.train_joint_scene_readout_v44 import (
    _SOURCE_FILES as _V41_SOURCE_FILES,
)
from semantic_3d_chat.training.train_joint_scene_readout_v44 import (
    _SOURCE_FULL_SHA256 as _V41_FULL_SHA256,
)
from semantic_3d_chat.training.train_joint_scene_readout_v44 import (
    _source_tensors as _v41_source_tensors,
)
from semantic_3d_chat.training.train_projected_gradient_v41 import (
    _prefix_replay_attestation,
    cache_v41_train_scenes,
    load_v41_bundle,
    priority_side_deficit,
    training_pair_gate_diagnostics,
    v41_loader_config,
    validate_per_unit_nll_diagnostics,
)
from semantic_3d_chat.training.train_scene_ingress_kv_v37 import (
    validate_v37_training_cache_boundary,
)

DEFAULT_CONFIG = Path(
    "configs/experiments/gemma4_diverse28_retention_repair_v45.yaml"
)
DEFAULT_OUTPUT = Path(
    "data_gemma4/checkpoints/gemma4_v45_retention_repair_l14_query"
)
_CONFIG_FILE_SHA256 = "9a4b77c43d30d258be9e4e6d60c477ef6af593fe440d61d28082f2f343519436"
_V44_TERMINAL_PATH = Path(
    "reports/gemma4/metrics/v44_joint_scene_readout_terminal_gate.json"
)
_V44_TERMINAL_SHA256 = "b968c46c686051e864417b7539db7e90160a1f0b4639af031d02aab005643b67"
_SOURCE_CHECKPOINT = Path(
    "data_gemma4/checkpoints/gemma4_v44_joint_scene_readout_l14_query/update_008"
)
_SOURCE_FILES = {
    "adapter.safetensors": "22f7e0276a91d45e31893843345e98e310fbffd14147852c05c5c3bec4dc6589",
    TRAINING_METADATA_FILENAME: "797fcdb87da3391c5196fda15fca4d352846dda2d5dcc49263ca3f7854fcd1b3",
    RUNTIME_METADATA_FILENAME: "59542b55239d64a9c28b9b99ec0a39b47c1dd93839753f61d145722ea7c50acf",
}
_SOURCE_FULL_SHA256 = "ad9b2227e68020ae785084666c9dca58c3d479e5b1e3e4c13461539fcb19c6fb"
_SOURCE_AUTHORIZED_SHA256 = (
    "f56fdc4ce31a3e97c80e9a214948b6855fd87ae3b7f96ebf1f152229cf833e02"
)
_FROZEN_SHA256 = "31cb215cf0d1623886d4a79203c501912e66537021da53dd711124abdc2e36fa"
_PROTECTED_REPORT = Path(
    "reports/gemma4/metrics/training_selection_gemma4_color_mirror_full_vocab_v11_resume36.json"
)
_PROTECTED_REPORT_SHA256 = (
    "c0086f66edbb8854a7938e09c57535bfd47100adbaf3b3c95eeb4b08014ce2f8"
)
_SAVED_STEPS = (0, 2, 4, 6, 8)
_HEX = re.compile(r"[0-9a-f]{64}")

_TARGET_QUESTION_KEYS = (
    "cfq_5c84a2c27d2be251",
    "cfq_699675ceeaf65406",
    "cfq_5c84a2c27d2be251",
    "cfq_699675ceeaf65406",
    "cfq_163eb92339ad35a5",
    "cfq_163eb92339ad35a5",
    "cfq_163eb92339ad35a5",
    "cfq_163eb92339ad35a5",
)
_TARGET_PAIR_IDS = (
    "pair_000006",
    "pair_000016",
    "pair_000006",
    "pair_000016",
    "pair_000015",
    "pair_000015",
    "pair_000015",
    "pair_000015",
)
_BROAD_ROW_NUMBERS = tuple(range(9, 17))
_BROAD_QUESTION_IDS = (
    "q_000111",
    "q_000079",
    "q_000092",
    "q_000100",
    "q_000099",
    "q_000138",
    "q_000053",
    "q_000089",
)
_FRAGILE_SIDE_SPECS = (
    ("cfq_a578dc166be9a217", 0),
    ("cfq_0a79d507273195ef", 0),
    ("cfq_5c84a2c27d2be251", 0),
    ("cfq_736067b51ce93c49", 0),
    ("cfq_997610c185204121", 0),
    ("cfq_699675ceeaf65406", 0),
    ("cfq_699675ceeaf65406", 1),
    ("cfq_90b3d9852a93ce2a", 1),
)
_FRAGILE_SIDE_SOURCE_MARGINS = (
    0.4687870144844055,
    0.125,
    0.125,
    0.0625,
    0.25,
    0.5,
    0.25,
    0.125,
)
_FRAGILE_SIDE_PAIR_IDS = {
    "cfq_a578dc166be9a217": "pair_000005",
    "cfq_0a79d507273195ef": "pair_000006",
    "cfq_5c84a2c27d2be251": "pair_000006",
    "cfq_736067b51ce93c49": "pair_000007",
    "cfq_997610c185204121": "pair_000007",
    "cfq_699675ceeaf65406": "pair_000016",
    "cfq_90b3d9852a93ce2a": "pair_000018",
}
_BOOK_CROSS_SPECS = (
    ("cfq_13b1138d14c52a7c", 0),
    ("cfq_13b1138d14c52a7c", 1),
    ("cfq_a1c673a1197a0961", 0),
    ("cfq_a1c673a1197a0961", 1),
)
_BOOK_CROSS_SOURCE_MARGINS = (
    0.1605992317199707,
    0.026900842785835266,
    0.030324697494506836,
    0.03217540681362152,
)
_BOOK_CROSS_PAIR_ID = "pair_000015"
_LOST_SIDE_SPECS = (
    ("pair_000006", "cfq_5c84a2c27d2be251", 0),
    ("pair_000016", "cfq_699675ceeaf65406", 1),
)
_SOURCE_PRIORITY_DEFICIT = 31.113729119300842
_SOURCE_BROAD_NLL = 2.9013306349515915
_BROAD_NLL_MAXIMUM = 2.9213306349515915
_SOURCE_OPTIMIZER_SHA256_PROVENANCE = (
    "cdf9eb0c3560be1bc1542963354444eddb7a89ed0d063ffaa769c45231b9d61a"
)


@dataclass(frozen=True)
class V45Settings:
    optimizer_steps: int
    checkpoint_steps: tuple[int, ...]
    broad_nll_weight: float
    pair_correct_nll_weight: float
    target_side_hinge_weight: float
    target_cross_prefix_weight: float
    target_side_hinge_margin: float
    target_cross_prefix_margin: float
    retention_weight: float
    retention_side_floor: float
    retention_book_cross_floor: float
    source_prefix_trust_weight: float
    source_prefix_trust_scale: float
    scene_readout_learning_rate: float
    query_learning_rate: float
    weight_decay: float
    gradient_clip_norm: float

    @property
    def side_hinge_margin(self) -> float:
        """Compatibility name used by full 25-unit training diagnostics."""

        return self.target_side_hinge_margin

    @property
    def cross_prefix_flip_margin(self) -> float:
        """Compatibility name used by full 25-unit training diagnostics."""

        return self.target_cross_prefix_margin


@dataclass(frozen=True)
class V45Contract:
    terminal_report: Path
    configured_terminal_sha256: str
    source_checkpoint: Path
    construction_source_checkpoint: Path
    source_file_sha256: Mapping[str, str]
    source_full_state_sha256: str
    source_authorized_state_sha256: str
    frozen_state_sha256: str
    authorized_parameter_names: tuple[str, ...]
    authorized_parameter_shapes: tuple[tuple[int, ...], ...]
    total_parameter_count: int


@dataclass(frozen=True)
class V45ScheduleRow:
    optimizer_step: int
    target_unit: CounterfactualPairUnit
    broad_record: QARecord


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (PROJECT_ROOT / value).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be a mapping")
    return value


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def v45_settings(config: Mapping[str, Any]) -> V45Settings:
    training = _mapping(config.get("training"), "training")
    raw = _mapping(training.get("v45_retention_repair"), "V45 training")
    expected = {
        "enabled": True,
        "optimizer_steps": 8,
        "checkpoint_steps": list(_SAVED_STEPS),
        "broad_nll_weight": 0.25,
        "pair_correct_nll_weight": 0.5,
        "target_side_hinge_weight": 8.0,
        "target_cross_prefix_weight": 8.0,
        "target_side_hinge_margin": 0.5,
        "target_cross_prefix_margin": 0.1,
        "retention_weight": 8.0,
        "retention_side_floor": 0.125,
        "retention_book_cross_floor": 0.025,
        "source_prefix_trust_weight": 0.001,
        "source_prefix_trust_scale": 0.05,
        "scene_readout_learning_rate": 1.0e-5,
        "query_learning_rate": 8.0e-6,
        "weight_decay": 0.0,
        "gradient_clip_norm": 1.0,
    }
    if set(raw) != set(expected) or any(
        raw.get(key) != value for key, value in expected.items()
    ):
        raise ValueError("V45 exact optimizer/objective settings changed")
    return V45Settings(
        optimizer_steps=8,
        checkpoint_steps=_SAVED_STEPS,
        broad_nll_weight=0.25,
        pair_correct_nll_weight=0.5,
        target_side_hinge_weight=8.0,
        target_cross_prefix_weight=8.0,
        target_side_hinge_margin=0.5,
        target_cross_prefix_margin=0.1,
        retention_weight=8.0,
        retention_side_floor=0.125,
        retention_book_cross_floor=0.025,
        source_prefix_trust_weight=0.001,
        source_prefix_trust_scale=0.05,
        scene_readout_learning_rate=1.0e-5,
        query_learning_rate=8.0e-6,
        weight_decay=0.0,
        gradient_clip_norm=1.0,
    )


def _fragile_constraint_records() -> list[dict[str, Any]]:
    return [
        {
            "pair_id": _FRAGILE_SIDE_PAIR_IDS[question_key],
            "question_key": question_key,
            "side_index": side_index,
            "source_margin": source_margin,
            "retention_floor": 0.125,
        }
        for (question_key, side_index), source_margin in zip(
            _FRAGILE_SIDE_SPECS, _FRAGILE_SIDE_SOURCE_MARGINS, strict=True
        )
    ]


def _book_cross_constraint_records() -> list[dict[str, Any]]:
    return [
        {
            "pair_id": _BOOK_CROSS_PAIR_ID,
            "question_key": question_key,
            "side_index": side_index,
            "source_margin": source_margin,
            "retention_floor": 0.025,
        }
        for (question_key, side_index), source_margin in zip(
            _BOOK_CROSS_SPECS, _BOOK_CROSS_SOURCE_MARGINS, strict=True
        )
    ]


def _update4_contract() -> dict[str, Any]:
    return {
        "complete_units_minimum": 9,
        "positive_sides_minimum": 34,
        "cross_prefix_complete_units_minimum": 17,
        "complete_physical_pair_id_coverage_minimum": 4,
        "priority_side_deficit_minimum_improvement_vs_original_v41_u0": 0.5,
        "broad_nll_maximum": _BROAD_NLL_MAXIMUM,
        "lost_side_margins_must_both_be_strictly_positive": True,
        "both_authorized_parameter_groups_must_change": True,
        "frozen_state_must_remain_exact": True,
    }


def _update8_contract() -> dict[str, Any]:
    return {
        "require_recorded_update4_gate_passed": True,
        "complete_units_minimum": 10,
        "positive_sides_minimum": 35,
        "cross_prefix_complete_units_minimum": 17,
        "complete_physical_pair_id_coverage_minimum": 5,
        "mirror_complete_units_minimum": 2,
        "book_complete_units_minimum": 1,
        "book_cross_prefix_complete_units_minimum": 1,
        "priority_side_deficit_minimum_improvement_vs_original_v41_u0": 0.5,
        "broad_nll_maximum": _BROAD_NLL_MAXIMUM,
        "train_greedy_complete_units_minimum": 5,
        "broad_greedy_exact_correct_minimum": 23,
        "broad_greedy_row_count_exact": 48,
        "lost_side_margins_must_remain_strictly_positive": True,
        "both_authorized_parameter_groups_must_change": True,
        "frozen_state_must_remain_exact": True,
        "u8_prefix_trust_rms_maximum": 0.002,
    }


def v45_contract(config: Mapping[str, Any]) -> V45Contract:
    raw = _mapping(config.get("v45_retention_repair"), "v45_retention_repair")
    if raw.get("schema_version") != 1 or raw.get("role") != (
        "exact_v44_u8_retention_repair_layer14_query_train_only_pilot"
    ):
        raise ValueError("V45 contract identity changed")
    names = tuple(str(value) for value in raw.get("authorized_parameter_names", ()))
    shapes = tuple(
        tuple(int(item) for item in value)
        for value in raw.get("authorized_parameter_shapes", ())
    )
    checks = {
        "names": names == _PARAMETER_NAMES,
        "shapes": shapes == _PARAMETER_SHAPES,
        "scene_count": raw.get("scene_readout_parameter_count") == 393_216,
        "query_count": raw.get("query_parameter_count") == 22_528,
        "total": raw.get("total_trainable_parameter_count") == 415_744,
        "optimizer": raw.get("optimizer") == "fresh_adamw_two_groups",
        "validation": raw.get("validation_access_authorized") is False,
        "oracle": raw.get("oracle_access_authorized") is False,
        "final": raw.get("final_test_access_authorized") is False,
        "selector": raw.get("selector_execution_authorized") is False,
        "promotion": raw.get("runtime_promotion_authorized") is False,
        "optimizer_provenance": raw.get("source_optimizer_file_sha256_provenance")
        == _SOURCE_OPTIMIZER_SHA256_PROVENANCE,
        "optimizer_open": raw.get("source_optimizer_file_open_authorized") is False,
        "target_schedule": tuple(raw.get("target_question_key_schedule", ()))
        == _TARGET_QUESTION_KEYS,
        "broad_rows": tuple(raw.get("fixed_broad_row_numbers", ()))
        == _BROAD_ROW_NUMBERS,
        "broad_ids": tuple(raw.get("fixed_broad_question_ids", ()))
        == _BROAD_QUESTION_IDS,
        "fragile": list(raw.get("fragile_side_constraints", ()))
        == _fragile_constraint_records(),
        "book_cross": list(raw.get("book_cross_constraints", ()))
        == _book_cross_constraint_records(),
        "gate4": dict(_mapping(raw.get("update4_gate"), "V45 update4 gate"))
        == _update4_contract(),
        "gate8": dict(_mapping(raw.get("update8_gate"), "V45 update8 gate"))
        == _update8_contract(),
        "source_full": raw.get("source_full_tensor_state_sha256")
        == _SOURCE_FULL_SHA256,
        "source_authorized": raw.get("source_authorized_surface_state_sha256")
        == _SOURCE_AUTHORIZED_SHA256,
        "frozen": raw.get("frozen_excluding_authorized_state_sha256")
        == _FROZEN_SHA256,
    }
    if not all(checks.values()):
        raise ValueError(f"V45 exact contract changed: {checks}")
    source_files = dict(_mapping(raw.get("source_file_sha256"), "V45 source hashes"))
    if source_files != _SOURCE_FILES:
        raise ValueError("V45 source file hashes changed")
    terminal_report = _resolve(str(raw["v44_terminal_report"]))
    source_checkpoint = _resolve(str(raw["source_checkpoint"]))
    construction = _resolve(str(raw["construction_source_checkpoint"]))
    if terminal_report != _resolve(_V44_TERMINAL_PATH):
        raise ValueError("V45 terminal path differs from its authorization")
    if source_checkpoint != _resolve(_SOURCE_CHECKPOINT):
        raise ValueError("V45 source checkpoint path differs from its authorization")
    if construction != _resolve(_V41_CONSTRUCTION_SOURCE):
        raise ValueError("V45 construction must start from exact original V41")
    return V45Contract(
        terminal_report=terminal_report,
        configured_terminal_sha256=str(raw["v44_terminal_report_sha256"]),
        source_checkpoint=source_checkpoint,
        construction_source_checkpoint=construction,
        source_file_sha256=source_files,
        source_full_state_sha256=_SOURCE_FULL_SHA256,
        source_authorized_state_sha256=_SOURCE_AUTHORIZED_SHA256,
        frozen_state_sha256=_FROZEN_SHA256,
        authorized_parameter_names=names,
        authorized_parameter_shapes=shapes,
        total_parameter_count=415_744,
    )


def _expected_v45_authorization() -> dict[str, Any]:
    """Return the exact successor dictionary from the one-shot V44 seal."""

    fragile = _fragile_constraint_records()
    book_cross = _book_cross_constraint_records()
    lost = [
        {
            "pair_id": pair_id,
            "question_key": question_key,
            "side_index": side_index,
            "update8_margin": 0.0,
            "gate4_required_relation": "strictly_greater_than_zero",
        }
        for pair_id, question_key, side_index in _LOST_SIDE_SPECS
    ]
    lost_gate_specs = [
        {
            "pair_id": pair_id,
            "question_key": question_key,
            "side_index": side_index,
        }
        for pair_id, question_key, side_index in _LOST_SIDE_SPECS
    ]
    return {
        "schema_version": 1,
        "authorization_id": "v45_train_only_retention_repair_pilot",
        "authorized": True,
        "only_exact_action": "one_bounded_v45_train_only_retention_repair_pilot",
        "authorized_config": str(DEFAULT_CONFIG),
        "authorized_output_root": str(DEFAULT_OUTPUT),
        "source_checkpoint": str(_SOURCE_CHECKPOINT),
        "source_checkpoint_file_sha256": {
            "adapter.safetensors": _SOURCE_FILES["adapter.safetensors"],
            TRAINING_METADATA_FILENAME: _SOURCE_FILES[TRAINING_METADATA_FILENAME],
            "optimizer.pt": _SOURCE_OPTIMIZER_SHA256_PROVENANCE,
            RUNTIME_METADATA_FILENAME: _SOURCE_FILES[RUNTIME_METADATA_FILENAME],
        },
        "source_adapter_tensor_count": 179,
        "source_full_tensor_state_sha256": _SOURCE_FULL_SHA256,
        "source_authorized_surface_state_sha256": _SOURCE_AUTHORIZED_SHA256,
        "source_frozen_excluding_authorized_state_sha256": _FROZEN_SHA256,
        "source_optimizer_policy": {
            "source_optimizer_file_present_and_authenticated": True,
            "source_optimizer_file_open_authorized_by_v45": False,
            "source_optimizer_deserialization_authorized": False,
            "source_optimizer_state_loading_authorized": False,
            "fresh_optimizer_required": True,
        },
        "trainable_surface": {
            "parameter_names": list(_PARAMETER_NAMES),
            "parameter_shapes": [list(value) for value in _PARAMETER_SHAPES],
            "parameter_counts": list(_PARAMETER_COUNTS),
            "scene_readout_parameter_count": 393_216,
            "query_parameter_count": 22_528,
            "total_parameter_count": 415_744,
            "block_qkv_frozen": True,
            "gemma_base_and_all_other_lora_banks_frozen": True,
            "every_other_tensor_and_buffer_frozen": True,
        },
        "optimizer": {
            "implementation": "fresh_torch_adamw_two_groups",
            "source_optimizer_loaded": False,
            "scene_readout_learning_rate": 1.0e-5,
            "query_learning_rate": 8.0e-6,
            "weight_decay": 0.0,
            "foreach": False,
            "fused": False,
            "per_group_gradient_clip_norm": 1.0,
        },
        "objective": {
            "broad_nll_weight": 0.25,
            "pair_correct_nll_weight": 0.5,
            "target_side_hinge_weight": 8.0,
            "target_cross_prefix_hinge_weight": 8.0,
            "target_side_hinge_margin": 0.5,
            "target_cross_prefix_hinge_margin": 0.1,
            "retention_weight": 8.0,
            "retention_formula": (
                "mean(relu(0.125-fragile_side_margin),8)"
                "+mean(relu(0.025-book_cross_prefix_margin),4)"
            ),
            "u8_prefix_trust_weight": 0.001,
            "u8_prefix_trust_scale": 0.05,
            "u8_prefix_reference_computed_before_optimizer_step_one": True,
        },
        "retention_control": {
            "fragile_side_constraints": fragile,
            "fragile_side_constraint_count": 8,
            "fragile_side_floor": 0.125,
            "book_cross_constraints": book_cross,
            "book_cross_constraint_count": 4,
            "book_cross_floor": 0.025,
            "lost_side_gate4_constraints": lost,
            "source_is_exact_original_v41_update_zero_pair_metrics": True,
            "applied_at_every_optimizer_update": True,
            "fragile_side_hinges_mean_normalized_separately": True,
            "book_cross_hinges_mean_normalized_separately": True,
            "two_normalized_means_summed_before_single_weight": True,
        },
        "target_schedule": [
            {"optimizer_update": index, "question_key": question_key}
            for index, question_key in enumerate(_TARGET_QUESTION_KEYS, start=1)
        ],
        "schedule": {
            "maximum_optimizer_updates": 8,
            "checkpoint_steps": list(_SAVED_STEPS),
            "gate4_must_pass_before_updates_five_through_eight": True,
            "true_optimizer_step_per_target_schedule_row": True,
            "target_schedule_is_fixed_and_nonadaptive": True,
        },
        "reference_baselines": {
            "original_v41_update_zero_priority_side_deficit": _SOURCE_PRIORITY_DEFICIT,
            "original_v41_update_zero_broad_nll": _SOURCE_BROAD_NLL,
            "broad_nll_maximum": _BROAD_NLL_MAXIMUM,
            "prefix_trust_reference": "exact_v44_update_008_full_scene_prefixes",
        },
        "update4_gate": {
            "complete_units_minimum": 9,
            "positive_sides_minimum": 34,
            "cross_prefix_complete_units_minimum": 17,
            "complete_physical_pair_id_coverage_minimum": 4,
            "priority_side_deficit_minimum_improvement_vs_original_v41_u0": 0.5,
            "broad_nll_maximum": _BROAD_NLL_MAXIMUM,
            "lost_side_margins_must_both_be_strictly_positive": lost_gate_specs,
            "both_authorized_parameter_groups_must_change": True,
            "frozen_state_must_remain_exact": True,
        },
        "update8_gate": {
            "require_recorded_update4_gate_passed": True,
            "complete_units_minimum": 10,
            "positive_sides_minimum": 35,
            "cross_prefix_complete_units_minimum": 17,
            "complete_physical_pair_id_coverage_minimum": 5,
            "mirror_complete_units_minimum": 2,
            "book_complete_units_minimum": 1,
            "book_cross_prefix_complete_units_minimum": 1,
            "priority_side_deficit_minimum_improvement_vs_original_v41_u0": 0.5,
            "broad_nll_maximum": _BROAD_NLL_MAXIMUM,
            "train_greedy_complete_units_minimum": 5,
            "broad_greedy_exact_correct_minimum": 23,
            "broad_greedy_row_count_exact": 48,
            "lost_side_margins_must_remain_strictly_positive": lost_gate_specs,
            "both_authorized_parameter_groups_must_change": True,
            "frozen_state_must_remain_exact": True,
            "u8_prefix_trust_rms_maximum": 0.002,
        },
        "scope": {
            "training_qa_and_maps_only": True,
            "all_occupied_blocks_processed": True,
            "question_dependent_scene_processing": False,
            "question_dependent_retrieval": False,
            "validation_access_authorized": False,
            "oracle_access_authorized": False,
            "final_test_access_authorized": False,
            "selector_execution_authorized": False,
            "runtime_promotion_authorized": False,
            "chat_promotion_authorized": False,
            "new_terminal_seal_required_after_training": True,
        },
    }


def require_v44_terminal_gate(
    config: Mapping[str, Any], *, expected_sha256: str
) -> dict[str, Any]:
    contract = v45_contract(config)
    if _HEX.fullmatch(expected_sha256) is None or expected_sha256 != _V44_TERMINAL_SHA256:
        raise ValueError("V45 requires the exact pinned V44 terminal SHA-256")
    if contract.configured_terminal_sha256 != _V44_TERMINAL_SHA256:
        raise ValueError("V45 configured V44 terminal hash changed")
    path = contract.terminal_report
    if path.is_symlink() or not path.is_file() or _sha256(path) != expected_sha256:
        raise ValueError("V45 requires the exact real V44 terminal seal")
    report = json.loads(path.read_text(encoding="utf-8"))
    authorization = _mapping(
        _mapping(report, "V44 terminal").get("conditional_successor_authorization"),
        "V45 successor authorization",
    )
    checks = {
        "schema_version": report.get("schema_version") == 1,
        "artifact": report.get("artifact")
        == "v44_joint_scene_readout_terminal_gate",
        "passed": report.get("passed") is True,
        "successor": report.get("only_exact_successor_authorized")
        == "v45_train_only_retention_repair_pilot",
        "v45_authorized": report.get(
            "v45_train_only_retention_repair_pilot_authorized"
        )
        is True,
        "validation": report.get("validation_access_authorized") is False,
        "selector": report.get("selector_execution_authorized") is False,
        "promotion": report.get("chat_or_runtime_promotion_authorized") is False,
        "authorization_exact": dict(authorization) == _expected_v45_authorization(),
    }
    if not all(checks.values()):
        raise ValueError(f"V44 does not authorize exact V45: {checks}")
    return {
        "path": str(path),
        "sha256": expected_sha256,
        "authorization": dict(authorization),
        "exact_authorization_fields_verified": True,
    }


def _v44_u8_source(
    contract: V45Contract,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Authenticate only the three V45-readable V44-u8 checkpoint files."""

    if contract.source_checkpoint.is_symlink() or not contract.source_checkpoint.is_dir():
        raise FileNotFoundError("V45 source must be a real V44 update-008 directory")
    for name, expected in contract.source_file_sha256.items():
        path = contract.source_checkpoint / name
        if path.is_symlink() or not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"V45 exact source file changed: {name}")
    metadata = json.loads(
        (contract.source_checkpoint / TRAINING_METADATA_FILENAME).read_text(
            encoding="utf-8"
        )
    )
    runtime = json.loads(
        (contract.source_checkpoint / RUNTIME_METADATA_FILENAME).read_text(
            encoding="utf-8"
        )
    )
    validate_runtime_checkpoint_metadata(runtime)
    if runtime != runtime_checkpoint_metadata(metadata):
        raise ValueError("V45 source runtime metadata is not freshly sanitized")
    stage = _mapping(metadata.get("v44_joint_scene_readout"), "V44-u8 source stage")
    if (
        metadata.get("optimizer_step") != 8
        or stage.get("optimizer_step") != 8
        or _mapping(stage.get("update8_train_only_gate"), "V44 update8 gate").get(
            "passed"
        )
        is not False
        or stage.get("validation_qa_loaded") is not False
        or stage.get("oracle_environment_files_loaded") is not False
        or stage.get("deferred_final_scene_ids_loaded") != []
    ):
        raise ValueError("V45 source is not the exact stopped train-only V44 update eight")
    tensors = load_file(
        contract.source_checkpoint / "adapter.safetensors", device="cpu"
    )
    if len(tensors) != 179 or tensor_state_sha256(tensors) != _SOURCE_FULL_SHA256:
        raise ValueError("V45 V44-u8 source tensor state changed")
    observed = tuple(name for name in _PARAMETER_NAMES if name in tensors)
    if observed != _PARAMETER_NAMES:
        raise ValueError("V45 authorized source tensor inventory changed")
    if tuple(tuple(tensors[name].shape) for name in observed) != _PARAMETER_SHAPES:
        raise ValueError("V45 authorized source tensor shapes changed")
    if tuple(int(tensors[name].numel()) for name in observed) != _PARAMETER_COUNTS:
        raise ValueError("V45 authorized source parameter counts changed")
    authorized = {name: tensors[name] for name in _PARAMETER_NAMES}
    frozen = {name: value for name, value in tensors.items() if name not in authorized}
    if tensor_state_sha256(authorized) != _SOURCE_AUTHORIZED_SHA256:
        raise ValueError("V45 authorized source surface changed")
    if tensor_state_sha256(frozen) != _FROZEN_SHA256:
        raise ValueError("V45 frozen source surface changed")
    return tensors, metadata


def _unit_index(
    units: Sequence[CounterfactualPairUnit],
) -> dict[str, CounterfactualPairUnit]:
    by_key: dict[str, CounterfactualPairUnit] = {}
    for unit in units:
        if unit.question_key in by_key:
            raise ValueError(f"V45 duplicate question key: {unit.question_key}")
        by_key[unit.question_key] = unit
    if len(by_key) != 25:
        raise RuntimeError("V45 exact 25-unit training inventory changed")
    return by_key


def validate_v45_unit_inventory(
    units: Sequence[CounterfactualPairUnit],
) -> dict[str, Any]:
    by_key = _unit_index(units)
    target_pairs = []
    for question_key, pair_id in zip(
        _TARGET_QUESTION_KEYS, _TARGET_PAIR_IDS, strict=True
    ):
        unit = by_key.get(question_key)
        if unit is None or unit.pair_id != pair_id:
            raise ValueError(f"V45 target unit changed: {question_key}")
        target_pairs.append(pair_id)
    for question_key, _side_index in _FRAGILE_SIDE_SPECS:
        unit = by_key.get(question_key)
        if unit is None or unit.pair_id != _FRAGILE_SIDE_PAIR_IDS[question_key]:
            raise ValueError(f"V45 fragile-side unit changed: {question_key}")
    for question_key, _side_index in _BOOK_CROSS_SPECS:
        unit = by_key.get(question_key)
        if unit is None or unit.pair_id != _BOOK_CROSS_PAIR_ID:
            raise ValueError(f"V45 book-cross unit changed: {question_key}")
    return {
        "unit_count": len(by_key),
        "target_question_keys": list(_TARGET_QUESTION_KEYS),
        "target_pair_ids": target_pairs,
        "fragile_side_constraints": _fragile_constraint_records(),
        "book_cross_constraints": _book_cross_constraint_records(),
        "question_keys_are_opaque": True,
    }


def build_v45_schedule(
    records: Sequence[QARecord],
    units: Sequence[CounterfactualPairUnit],
    *,
    config: Mapping[str, Any],
) -> tuple[list[V45ScheduleRow], dict[str, Any], list[QARecord]]:
    """Build the exact fixed target schedule and broad rows 9 through 16."""

    by_key = _unit_index(units)
    inherited, _ = build_v35_schedule(
        records,
        units,
        settings=v35_settings(v41_loader_config(config)),
        seed=int(config["seed"]),
    )
    broad_all = v36_broad_calibration_records(inherited)
    if len(broad_all) != 48:
        raise RuntimeError("V45 fixed broad-48 calibration inventory changed")
    broad_rows = [broad_all[index - 1] for index in _BROAD_ROW_NUMBERS]
    if tuple(record.question_id for record in broad_rows) != _BROAD_QUESTION_IDS:
        raise RuntimeError("V45 fixed broad training rows 9-16 changed")
    schedule = [
        V45ScheduleRow(
            optimizer_step=step,
            target_unit=by_key[question_key],
            broad_record=broad_record,
        )
        for step, (question_key, broad_record) in enumerate(
            zip(_TARGET_QUESTION_KEYS, broad_rows, strict=True), start=1
        )
    ]
    contract_rows = [
        {
            "optimizer_update": row.optimizer_step,
            "target_pair_id": row.target_unit.pair_id,
            "target_question_key": row.target_unit.question_key,
            "broad_row_number": broad_row,
            "broad_question_id": row.broad_record.question_id,
        }
        for row, broad_row in zip(schedule, _BROAD_ROW_NUMBERS, strict=True)
    ]
    return schedule, {
        "schema_version": 1,
        "rows": contract_rows,
        "schedule_sha256": _canonical_sha256(contract_rows),
        "target_schedule_is_fixed_and_nonadaptive": True,
        "one_true_optimizer_step_per_row": True,
        "fixed_broad_rows": list(_BROAD_ROW_NUMBERS),
    }, broad_all


def retention_hinge_from_selected_margins(
    side_margins: torch.Tensor,
    book_cross_margins: torch.Tensor,
    *,
    side_floor: float = 0.125,
    book_cross_floor: float = 0.025,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the exact separately normalized V45 retention objective."""

    if side_margins.shape != (8,) or book_cross_margins.shape != (4,):
        raise ValueError("V45 retention margin vectors must have shapes [8] and [4]")
    if not torch.isfinite(side_margins).all() or not torch.isfinite(
        book_cross_margins
    ).all():
        raise ValueError("V45 retention margins must be finite")
    side_mean = torch.relu(side_floor - side_margins).mean()
    book_cross_mean = torch.relu(book_cross_floor - book_cross_margins).mean()
    return side_mean + book_cross_mean, side_mean, book_cross_mean


def _margin_rows(pair_metrics: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    raw = pair_metrics.get("units")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise TypeError("V45 pair metrics units must be a sequence")
    rows: dict[str, Mapping[str, Any]] = {}
    for value in raw:
        row = _mapping(value, "V45 pair metric row")
        key = str(row.get("question_key"))
        if key in rows:
            raise ValueError(f"V45 duplicate metric question key: {key}")
        rows[key] = row
    if len(rows) != 25:
        raise ValueError("V45 gate metrics must contain all 25 training units")
    return rows


def v45_retention_diagnostics(pair_metrics: Mapping[str, Any]) -> dict[str, Any]:
    rows = _margin_rows(pair_metrics)
    fragile = []
    for question_key, side_index in _FRAGILE_SIDE_SPECS:
        row = rows[question_key]
        margins = row.get("side_margins")
        if row.get("pair_id") != _FRAGILE_SIDE_PAIR_IDS[question_key] or not isinstance(
            margins, Sequence
        ):
            raise ValueError(f"V45 fragile metric changed: {question_key}")
        value = float(margins[side_index])
        fragile.append(
            {
                "pair_id": row["pair_id"],
                "question_key": question_key,
                "side_index": side_index,
                "margin": value,
                "retention_floor": 0.125,
                "at_or_above_floor": value >= 0.125,
            }
        )
    book = []
    for question_key, side_index in _BOOK_CROSS_SPECS:
        row = rows[question_key]
        margins = row.get("cross_prefix_margins")
        if row.get("pair_id") != _BOOK_CROSS_PAIR_ID or not isinstance(
            margins, Sequence
        ):
            raise ValueError(f"V45 book-cross metric changed: {question_key}")
        value = float(margins[side_index])
        book.append(
            {
                "pair_id": row["pair_id"],
                "question_key": question_key,
                "side_index": side_index,
                "margin": value,
                "retention_floor": 0.025,
                "at_or_above_floor": value >= 0.025,
            }
        )
    lost = []
    for pair_id, question_key, side_index in _LOST_SIDE_SPECS:
        row = rows[question_key]
        margins = row.get("side_margins")
        if row.get("pair_id") != pair_id or not isinstance(margins, Sequence):
            raise ValueError(f"V45 lost-side metric changed: {question_key}")
        value = float(margins[side_index])
        lost.append(
            {
                "pair_id": pair_id,
                "question_key": question_key,
                "side_index": side_index,
                "margin": value,
                "strictly_positive": value > 0.0,
            }
        )
    return {
        "fragile_sides": fragile,
        "book_cross_sides": book,
        "lost_sides": lost,
        "all_fragile_sides_at_or_above_floor": all(
            row["at_or_above_floor"] for row in fragile
        ),
        "all_book_cross_sides_at_or_above_floor": all(
            row["at_or_above_floor"] for row in book
        ),
        "both_lost_sides_strictly_positive": all(
            row["strictly_positive"] for row in lost
        ),
    }


def _family_counts(pair_metrics: Mapping[str, Any]) -> Mapping[str, Any]:
    return _mapping(
        pair_metrics.get("complete_units_by_family"), "V45 complete family counts"
    )


def _cross_family_counts(pair_metrics: Mapping[str, Any]) -> Mapping[str, Any]:
    return _mapping(
        pair_metrics.get("cross_prefix_complete_units_by_family"),
        "V45 cross family counts",
    )


def validate_v45_update_zero_baseline(
    *, pair_metrics: Mapping[str, Any], broad_nll: float
) -> dict[str, Any]:
    """Attest live V44-u8 decoding behavior before optimizer construction."""

    families = _family_counts(pair_metrics)
    deficit = float(priority_side_deficit(pair_metrics)["combined"])
    retention = v45_retention_diagnostics(pair_metrics)
    expected_families = {
        "book_support": 0,
        "mirror_lr": 1,
        "picture_support": 0,
    }
    checks = {
        "full_pair_unit_count_exact_25": int(pair_metrics["unit_count"]) == 25,
        "complete_units_exact_7": int(pair_metrics["complete_units"]) == 7,
        "positive_sides_exact_32": int(pair_metrics["positive_sides"]) == 32,
        "cross_prefix_complete_units_exact_18": int(
            pair_metrics["cross_prefix_complete_units"]
        )
        == 18,
        "complete_physical_pair_id_coverage_exact_3": int(
            pair_metrics["complete_physical_pair_coverage"]
        )
        == 3,
        "complete_family_counts_exact": dict(families) == expected_families,
        "priority_side_deficit_exact_within_1e_6": abs(
            deficit - 29.92463493347168
        )
        <= 1.0e-6,
        "broad_nll_exact_within_1e_6": abs(broad_nll - 2.898227721452713)
        <= 1.0e-6,
        "both_lost_side_margins_exact_zero": all(
            float(row["margin"]) == 0.0 for row in retention["lost_sides"]
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"V45 live update-zero baseline changed: {checks}")
    return {
        "passed": True,
        "checks": checks,
        **checks,
        "complete_units": 7,
        "positive_sides": 32,
        "cross_prefix_complete_units": 18,
        "complete_physical_pair_id_coverage": 3,
        "complete_units_by_family": expected_families,
        "priority_side_deficit": deficit,
        "broad_nll": broad_nll,
        "retention_diagnostics": retention,
        "computed_before_optimizer_construction": True,
        "validation_qa_loaded": False,
    }


def v45_update4_gate(
    *,
    pair_metrics: Mapping[str, Any],
    broad_nll: float,
    scene_readout_state_changed: bool,
    query_state_changed: bool,
    frozen_state_exact: bool,
    trust_rms: float,
) -> dict[str, Any]:
    retention = v45_retention_diagnostics(pair_metrics)
    deficit = float(priority_side_deficit(pair_metrics)["combined"])
    improvement = _SOURCE_PRIORITY_DEFICIT - deficit
    checks = {
        "teacher_complete_units_at_least_9": int(pair_metrics["complete_units"])
        >= 9,
        "teacher_positive_sides_at_least_34": int(pair_metrics["positive_sides"])
        >= 34,
        "teacher_cross_complete_units_at_least_17": int(
            pair_metrics["cross_prefix_complete_units"]
        )
        >= 17,
        "complete_physical_pair_id_coverage_at_least_4": int(
            pair_metrics["complete_physical_pair_coverage"]
        )
        >= 4,
        "priority_teacher_deficit_improved_at_least_0_5_vs_original_v41_u0": (
            improvement >= 0.5
        ),
        "broad_nll_at_most_authorized_maximum": broad_nll
        <= _BROAD_NLL_MAXIMUM,
        "both_lost_side_margins_strictly_positive": retention[
            "both_lost_sides_strictly_positive"
        ],
        "scene_readout_state_changed": scene_readout_state_changed,
        "query_state_changed": query_state_changed,
        "both_authorized_parameter_groups_changed": scene_readout_state_changed
        and query_state_changed,
        "frozen_state_exact": frozen_state_exact,
    }
    return {
        "checks": checks,
        **checks,
        "passed": all(checks.values()),
        "priority_teacher_side_deficit": deficit,
        "priority_teacher_side_deficit_improvement_vs_original_v41_u0": improvement,
        "broad_nll": broad_nll,
        "broad_nll_maximum": _BROAD_NLL_MAXIMUM,
        "u8_prefix_trust_rms": trust_rms,
        "retention_diagnostics": retention,
        "full_train_pair_unit_count": int(pair_metrics["unit_count"]),
        "full_broad_nll_row_count": 48,
        "training_scenes_only": True,
        "validation_qa_loaded": False,
    }


def v45_update8_gate(
    *,
    update4_gate: Mapping[str, Any],
    pair_metrics: Mapping[str, Any],
    broad_nll: float,
    greedy_metrics: Mapping[str, Any],
    scene_readout_state_changed: bool,
    query_state_changed: bool,
    frozen_state_exact: bool,
    trust_rms: float,
) -> dict[str, Any]:
    retention = v45_retention_diagnostics(pair_metrics)
    deficit = float(priority_side_deficit(pair_metrics)["combined"])
    improvement = _SOURCE_PRIORITY_DEFICIT - deficit
    families = _family_counts(pair_metrics)
    cross_families = _cross_family_counts(pair_metrics)
    checks = {
        "recorded_update4_gate_passed": update4_gate.get("passed") is True,
        "teacher_complete_units_at_least_10": int(pair_metrics["complete_units"])
        >= 10,
        "teacher_positive_sides_at_least_35": int(pair_metrics["positive_sides"])
        >= 35,
        "teacher_cross_complete_units_at_least_17": int(
            pair_metrics["cross_prefix_complete_units"]
        )
        >= 17,
        "complete_physical_pair_id_coverage_at_least_5": int(
            pair_metrics["complete_physical_pair_coverage"]
        )
        >= 5,
        "mirror_complete_units_at_least_2": int(families.get("mirror_lr", 0))
        >= 2,
        "book_complete_units_at_least_1": int(families.get("book_support", 0))
        >= 1,
        "book_cross_prefix_complete_units_at_least_1": int(
            cross_families.get("book_support", 0)
        )
        >= 1,
        "priority_teacher_deficit_improved_at_least_0_5_vs_original_v41_u0": (
            improvement >= 0.5
        ),
        "broad_nll_at_most_authorized_maximum": broad_nll
        <= _BROAD_NLL_MAXIMUM,
        "train_greedy_complete_units_at_least_5": int(greedy_metrics["complete_units"])
        >= 5,
        "broad_greedy_exact_correct_at_least_23_of_48": int(
            greedy_metrics["broad_exact_correct"]
        )
        >= 23
        and int(greedy_metrics["broad_row_count"]) == 48,
        "both_lost_side_margins_remain_strictly_positive": retention[
            "both_lost_sides_strictly_positive"
        ],
        "scene_readout_state_changed": scene_readout_state_changed,
        "query_state_changed": query_state_changed,
        "both_authorized_parameter_groups_changed": scene_readout_state_changed
        and query_state_changed,
        "frozen_state_exact": frozen_state_exact,
        "u8_prefix_trust_rms_at_most_0_002": trust_rms <= 0.002,
    }
    return {
        "checks": checks,
        **checks,
        "passed": all(checks.values()),
        "priority_teacher_side_deficit": deficit,
        "priority_teacher_side_deficit_improvement_vs_original_v41_u0": improvement,
        "broad_nll": broad_nll,
        "broad_nll_maximum": _BROAD_NLL_MAXIMUM,
        "u8_prefix_trust_rms": trust_rms,
        "retention_diagnostics": retention,
        "training_greedy_metrics": dict(greedy_metrics),
        "full_train_pair_unit_count": int(pair_metrics["unit_count"]),
        "full_broad_nll_row_count": 48,
        "training_scenes_only": True,
        "validation_qa_loaded": False,
        "selector_execution_authorized": False,
    }


def v45_stop_reason(
    optimizer_step: int,
    *,
    update4_gate: Mapping[str, Any] | None,
    update8_gate: Mapping[str, Any] | None,
) -> str | None:
    if optimizer_step == 4:
        if update4_gate is None:
            raise RuntimeError("V45 update-4 stop decision lacks its gate")
        return (
            None
            if update4_gate.get("passed") is True
            else "update4_train_only_gate_failed"
        )
    if optimizer_step == 8:
        if update8_gate is None:
            raise RuntimeError("V45 update-8 stop decision lacks its gate")
        return (
            None
            if update8_gate.get("passed") is True
            else "update8_train_only_gate_failed"
        )
    return None


def v45_saved_optimizer_steps(history: Sequence[Mapping[str, Any]]) -> list[int]:
    return [
        int(row["optimizer_update"])
        for row in history
        if row.get("saved_checkpoint") is True
    ]


def v45_optimizer(
    scene_readout: Sequence[torch.nn.Parameter],
    query: Sequence[torch.nn.Parameter],
    settings: V45Settings,
) -> torch.optim.AdamW:
    if len(scene_readout) != 1 or len(query) != 2:
        raise ValueError("V45 optimizer requires one scene and two query tensors")
    optimizer = torch.optim.AdamW(
        [
            {
                "name": "scene_readout",
                "params": list(scene_readout),
                "lr": settings.scene_readout_learning_rate,
            },
            {
                "name": "layer14_query",
                "params": list(query),
                "lr": settings.query_learning_rate,
            },
        ],
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        foreach=False,
        fused=False,
    )
    if optimizer.state:
        raise RuntimeError("V45 AdamW must start with empty state")
    return optimizer


def v45_optimizer_audit(optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    groups = optimizer.param_groups
    if len(groups) != 2 or [group.get("name") for group in groups] != [
        "scene_readout",
        "layer14_query",
    ]:
        raise ValueError("V45 optimizer group inventory changed")
    expected = ((1.0e-5, 1), (8.0e-6, 2))
    if any(
        float(group["lr"]) != lr
        or float(group["weight_decay"]) != 0.0
        or len(group["params"]) != count
        or group.get("foreach") is not False
        or group.get("fused") is not False
        for group, (lr, count) in zip(groups, expected, strict=True)
    ):
        raise ValueError("V45 optimizer group settings changed")
    return {
        "implementation": "torch.optim.AdamW",
        "group_names": ["scene_readout", "layer14_query"],
        "learning_rates": [1.0e-5, 8.0e-6],
        "parameter_counts": [393_216, 22_528],
        "weight_decay": 0.0,
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "foreach": False,
        "fused": False,
        "gradient_clip_method": "independent_per_optimizer_group",
        "per_group_gradient_clip_norm": 1.0,
        "source_optimizer_loaded": False,
    }


def _preflight_forbidden_roots(config: Mapping[str, Any]) -> list[Path]:
    loader = v41_loader_config(config)
    qa_root = artifact_root(loader, "qa").resolve()
    return [
        artifact_root(loader, "oracle").resolve(),
        artifact_root(loader, "maps").resolve(),
        artifact_root(loader, "rendered").resolve(),
        artifact_root(loader, "features").resolve(),
        qa_root / "validation.jsonl",
        qa_root / "test.jsonl",
        *[path.resolve() for path in PROJECT_ROOT.rglob("optimizer.pt")],
    ]


def _training_forbidden_roots(config: Mapping[str, Any]) -> list[Path]:
    loader = v41_loader_config(config)
    split = v31_contract(loader)
    qa_root = artifact_root(loader, "qa").resolve()
    maps_root = artifact_root(loader, "maps").resolve()
    roots = [
        artifact_root(loader, "oracle").resolve(),
        artifact_root(loader, "rendered").resolve(),
        artifact_root(loader, "features").resolve(),
        qa_root / "validation.jsonl",
        qa_root / "test.jsonl",
    ]
    allowed = set(split.train_scene_ids)
    if maps_root.is_dir():
        roots.extend(
            path.resolve() for path in maps_root.iterdir() if path.name not in allowed
        )
    roots.extend(path.resolve() for path in PROJECT_ROOT.rglob("optimizer.pt"))
    return roots


def preflight_v45(
    config_path: str | Path = DEFAULT_CONFIG,
    *,
    v44_terminal_sha256: str,
) -> dict[str, Any]:
    config_path = _resolve(config_path)
    if config_path != _resolve(DEFAULT_CONFIG) or _sha256(config_path) != _CONFIG_FILE_SHA256:
        raise ValueError("V45 config path or bytes differ from the exact authorization")
    config = load_config(config_path)
    settings = v45_settings(config)
    contract = v45_contract(config)
    terminal = require_v44_terminal_gate(
        config, expected_sha256=v44_terminal_sha256
    )
    loader = v41_loader_config(config)
    assert_deferred_final_scenes_absent(loader)
    audit = FileAccessAudit(
        _preflight_forbidden_roots(config),
        forbidden_component_names={"oracle"},
        block_forbidden=True,
    )
    with audit:
        tensors, metadata = _v44_u8_source(contract)
        construction_tensors, construction_metadata = _v41_source_tensors(
            v44_contract(config)
        )
        records, qa_audit = load_v35_train_qa_records(loader)
        units = build_exact_question_pair_units(records)
        unit_audit = validate_v45_unit_inventory(units)
        schedule, schedule_audit, broad_all = build_v45_schedule(
            records, units, config=config
        )
    audit.assert_clean()
    if (
        len(records) != 384
        or len(units) != 25
        or len(schedule) != 8
        or len(broad_all) != 48
    ):
        raise RuntimeError("V45 exact train-only inventory changed")
    return {
        "schema_version": 1,
        "artifact": "v45_retention_repair_preflight",
        "passed": True,
        "config_path": str(config_path),
        "config_hash": config_hash(dict(config)),
        "terminal": terminal,
        "source_checkpoint": str(contract.source_checkpoint),
        "construction_source_checkpoint": str(contract.construction_source_checkpoint),
        "construction_source_tensor_count": len(construction_tensors),
        "construction_source_full_tensor_state_sha256": tensor_state_sha256(
            construction_tensors
        ),
        "construction_source_metadata_optimizer_step": construction_metadata.get(
            "optimizer_step"
        ),
        "source_tensor_count": len(tensors),
        "source_full_tensor_state_sha256": tensor_state_sha256(tensors),
        "source_metadata_optimizer_step": metadata.get("optimizer_step"),
        "source_optimizer_file_sha256_provenance": (
            _SOURCE_OPTIMIZER_SHA256_PROVENANCE
        ),
        "source_optimizer_file_opened": False,
        "source_optimizer_state_loaded": False,
        "trainable_parameter_names": list(contract.authorized_parameter_names),
        "trainable_parameter_shapes": [
            list(value) for value in contract.authorized_parameter_shapes
        ],
        "trainable_parameter_count": contract.total_parameter_count,
        "settings": settings.__dict__,
        "train_question_count": len(records),
        "train_pair_unit_count": len(units),
        "broad_gate_row_count": len(broad_all),
        "bounded_schedule_steps": [row.optimizer_step for row in schedule],
        "target_question_keys": [row.target_unit.question_key for row in schedule],
        "fixed_broad_question_ids": [
            row.broad_record.question_id for row in schedule
        ],
        "schedule_audit": schedule_audit,
        "unit_audit": unit_audit,
        "qa_audit": qa_audit,
        "gemma_loaded": False,
        "scene_maps_loaded": False,
        "optimizer_file_opened": False,
        "validation_qa_loaded": False,
        "oracle_loaded": False,
        "final_test_scenes_touched": False,
        "selector_execution_authorized": False,
        "runtime_promotion_authorized": False,
        "loaded_files": audit.unique_paths,
        "forbidden_file_accesses": audit.forbidden_accesses(),
    }


def _unit_tokens(
    unit: CounterfactualPairUnit,
    *,
    caches: Mapping[str, Any],
    block_core: BlockCrossResidual,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        scene_id: current_scene_tokens(
            caches[scene_id], block_core, device=device
        )
        for scene_id in unit.scene_ids
    }


def _mps_empty_cache(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.empty_cache()


def _backward_v45_retention(
    *,
    units_by_key: Mapping[str, CounterfactualPairUnit],
    caches: Mapping[str, Any],
    block_core: BlockCrossResidual,
    bundle: Any,
    settings: V45Settings,
) -> dict[str, Any]:
    """Backpropagate each unit sequentially while preserving exact two means."""

    side_by_key: defaultdict[str, list[int]] = defaultdict(list)
    for question_key, side_index in _FRAGILE_SIDE_SPECS:
        side_by_key[question_key].append(side_index)
    cross_by_key: defaultdict[str, list[int]] = defaultdict(list)
    for question_key, side_index in _BOOK_CROSS_SPECS:
        cross_by_key[question_key].append(side_index)

    selected_side_margins: list[float] = []
    selected_side_hinges: list[float] = []
    for question_key in dict.fromkeys(key for key, _index in _FRAGILE_SIDE_SPECS):
        unit = units_by_key[question_key]
        tokens = _unit_tokens(
            unit,
            caches=caches,
            block_core=block_core,
            device=bundle.language.device,
        )
        pair_nll, side_hinge, cross_hinge, diagnostics = (
            paired_cross_prefix_objective(
                unit=unit,
                scene_tokens=tokens,
                bundle=bundle,
                side_margin=settings.target_side_hinge_margin,
                cross_prefix_margin=settings.target_cross_prefix_margin,
            )
        )
        margins = diagnostics["side_margins"].reshape(2)
        selected = torch.stack([margins[index] for index in side_by_key[question_key]])
        hinges = torch.relu(settings.retention_side_floor - selected)
        contribution = hinges.sum() / len(_FRAGILE_SIDE_SPECS)
        (settings.retention_weight * contribution).backward()
        selected_side_margins.extend(float(value) for value in selected.detach().cpu())
        selected_side_hinges.extend(float(value) for value in hinges.detach().cpu())
        del (
            pair_nll,
            side_hinge,
            cross_hinge,
            diagnostics,
            margins,
            selected,
            hinges,
            contribution,
            tokens,
        )
        _mps_empty_cache(bundle.language.device)

    selected_cross_margins: list[float] = []
    selected_cross_hinges: list[float] = []
    for question_key in dict.fromkeys(key for key, _index in _BOOK_CROSS_SPECS):
        unit = units_by_key[question_key]
        tokens = _unit_tokens(
            unit,
            caches=caches,
            block_core=block_core,
            device=bundle.language.device,
        )
        pair_nll, side_hinge, cross_hinge, diagnostics = (
            paired_cross_prefix_objective(
                unit=unit,
                scene_tokens=tokens,
                bundle=bundle,
                side_margin=settings.target_side_hinge_margin,
                cross_prefix_margin=settings.target_cross_prefix_margin,
            )
        )
        margins = diagnostics["cross_prefix_margins"].reshape(2)
        selected = torch.stack([margins[index] for index in cross_by_key[question_key]])
        hinges = torch.relu(settings.retention_book_cross_floor - selected)
        contribution = hinges.sum() / len(_BOOK_CROSS_SPECS)
        (settings.retention_weight * contribution).backward()
        selected_cross_margins.extend(float(value) for value in selected.detach().cpu())
        selected_cross_hinges.extend(float(value) for value in hinges.detach().cpu())
        del (
            pair_nll,
            side_hinge,
            cross_hinge,
            diagnostics,
            margins,
            selected,
            hinges,
            contribution,
            tokens,
        )
        _mps_empty_cache(bundle.language.device)

    side_mean = sum(selected_side_hinges) / len(_FRAGILE_SIDE_SPECS)
    cross_mean = sum(selected_cross_hinges) / len(_BOOK_CROSS_SPECS)
    return {
        "retention_hinge": side_mean + cross_mean,
        "fragile_side_hinge_mean": side_mean,
        "book_cross_hinge_mean": cross_mean,
        "weighted_retention_loss": settings.retention_weight
        * (side_mean + cross_mean),
        "selected_fragile_side_margins": selected_side_margins,
        "selected_fragile_side_hinges": selected_side_hinges,
        "selected_book_cross_margins": selected_cross_margins,
        "selected_book_cross_hinges": selected_cross_hinges,
        "fragile_side_denominator": 8,
        "book_cross_denominator": 4,
        "sequential_unit_backward": True,
        "two_normalized_means_summed_before_single_weight": True,
    }


def _metadata(
    *,
    source_metadata: Mapping[str, Any],
    config: Mapping[str, Any],
    terminal: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
    optimizer_step: int,
    bundle: Any,
    block_core: BlockCrossResidual,
    source_prefix_hashes: Mapping[str, str],
    source_frozen_hash: str,
    source_audit: Mapping[str, Any],
    schedule_audit: Mapping[str, Any],
    gate4: Mapping[str, Any] | None,
    gate8: Mapping[str, Any] | None,
) -> dict[str, Any]:
    result = copy.deepcopy(dict(source_metadata))
    result.update(
        {
            "schema_version": 1,
            "config_hash": config_hash(dict(config)),
            "optimizer_step": optimizer_step,
            "epoch": optimizer_step,
            "history": [dict(row) for row in history],
            "question_dependent_scene_processing": False,
            **bundle.lora_installation.checkpoint_metadata(),
            "block_cross_residual_state_sha256": block_core.state_sha256(),
            "frozen_block_cross_source_stack_state_sha256": (
                block_source_stack_state_sha256(bundle, block_core)
            ),
        }
    )
    result["v45_retention_repair"] = {
        "schema_version": 1,
        "optimizer_step": optimizer_step,
        "conditional_v44_terminal_gate": {
            "path": terminal["path"],
            "sha256": terminal["sha256"],
        },
        "conditional_authorization": dict(terminal["authorization"]),
        "source_checkpoint": str(v45_contract(config).source_checkpoint),
        "construction_source_checkpoint": str(
            v45_contract(config).construction_source_checkpoint
        ),
        "source_full_tensor_state_sha256": _SOURCE_FULL_SHA256,
        "source_authorized_surface_state_sha256": _SOURCE_AUTHORIZED_SHA256,
        "frozen_excluding_authorized_source_state_sha256": source_frozen_hash,
        "frozen_excluding_authorized_state_sha256": frozen_v44_state_sha256(bundle),
        "trainable_surface": assert_v44_trainable_surface(bundle, block_core),
        "source_prefix_sha256_by_train_scene": dict(source_prefix_hashes),
        "source_prefix_reference": "exact_v44_update_008_full_scene_prefixes",
        "source_audit": dict(source_audit),
        "schedule_audit": dict(schedule_audit),
        "update4_train_only_gate": None if gate4 is None else dict(gate4),
        "update8_train_only_gate": None if gate8 is None else dict(gate8),
        "validation_qa_loaded": False,
        "oracle_environment_files_loaded": False,
        "deferred_final_scene_ids_loaded": [],
        "question_dependent_scene_processing": False,
        "question_dependent_retrieval": False,
        "selector_execution_authorized": False,
        "runtime_promotion_authorized": False,
        "independent_terminal_seal_required": True,
    }
    return result


def _save(
    path: Path,
    *,
    bundle: Any,
    metadata: dict[str, Any],
    optimizer: torch.optim.Optimizer | None,
) -> None:
    if path.is_symlink() or path.exists():
        raise FileExistsError(f"V45 checkpoint destination already exists: {path}")
    save_adapter_checkpoint(path, bundle.checkpoint_modules, metadata)
    if optimizer is not None:
        save_optimizer_checkpoint(path, optimizer)


def _run_v45_impl(
    *,
    config_path: str | Path = DEFAULT_CONFIG,
    output: str | Path = DEFAULT_OUTPUT,
    v44_terminal_sha256: str,
) -> dict[str, Any]:
    config_path = _resolve(config_path)
    output = _resolve(output)
    if config_path != _resolve(DEFAULT_CONFIG) or _sha256(config_path) != _CONFIG_FILE_SHA256:
        raise ValueError("V45 config path or bytes differ from the exact authorization")
    config = load_config(config_path)
    settings = v45_settings(config)
    contract = v45_contract(config)
    terminal = require_v44_terminal_gate(
        config, expected_sha256=v44_terminal_sha256
    )
    if output != _resolve(DEFAULT_OUTPUT):
        raise ValueError("V45 output differs from its exact bounded namespace")
    if output.exists() or output.is_symlink():
        raise FileExistsError("V45 is one-shot and refuses any existing output root")

    source_tensors, source_checkpoint_metadata = _v44_u8_source(contract)
    loader = v41_loader_config(config)
    assert_deferred_final_scenes_absent(loader)
    records, qa_audit = load_v35_train_qa_records(loader)
    units = build_exact_question_pair_units(records)
    unit_audit = validate_v45_unit_inventory(units)
    units_by_key = _unit_index(units)
    schedule, schedule_audit, broad_records = build_v45_schedule(
        records, units, config=config
    )
    if (
        len(records) != 384
        or len(units) != 25
        or len(schedule) != 8
        or len(broad_records) != 48
    ):
        raise RuntimeError("V45 training inventory changed")

    # Construct from the original authenticated V41 checkpoint first.  Only
    # after that shape/provenance check passes is the exact V44-u8 state
    # strictly overlaid.  No optimizer file is opened in either operation.
    construction_contract = v44_contract(config)
    v41_tensors, v41_metadata = _v41_source_tensors(construction_contract)
    if (
        construction_contract.source_checkpoint != contract.construction_source_checkpoint
        or tensor_state_sha256(v41_tensors) != _V41_FULL_SHA256
        or tensor_state_sha256(
            {name: v41_tensors[name] for name in _PARAMETER_NAMES}
        )
        != _V41_AUTHORIZED_SHA256
        or construction_contract.source_file_sha256 != _V41_SOURCE_FILES
    ):
        raise RuntimeError("V45 original V41 construction source changed")
    approved = require_approved_v29_source(loader)
    bundle, block_core, loaded_v41_metadata, loader_transition = load_v41_bundle(
        config,
        approved,
        construction_contract.source_checkpoint,
        v41_tensors,
    )
    if loaded_v41_metadata != v41_metadata:
        raise RuntimeError("V45 original V41 construction metadata changed")
    if module_collection_state_sha256(bundle.checkpoint_modules) != _V41_FULL_SHA256:
        raise RuntimeError("V45 did not construct exact original V41 state")
    overlaid_metadata = load_adapter_checkpoint(
        contract.source_checkpoint,
        bundle.checkpoint_modules,
        device="cpu",
        metadata_filename=TRAINING_METADATA_FILENAME,
    )
    if overlaid_metadata != source_checkpoint_metadata:
        raise RuntimeError("V45 strict V44-u8 overlay metadata changed")
    if module_collection_state_sha256(bundle.checkpoint_modules) != _SOURCE_FULL_SHA256:
        raise RuntimeError("V45 strict V44-u8 overlay tensor state changed")
    named = freeze_for_v44(bundle, block_core)
    if frozen_v44_state_sha256(bundle) != _FROZEN_SHA256:
        raise RuntimeError("V45 live frozen state differs after strict overlay")
    surface = assert_v44_trainable_surface(bundle, block_core)
    source_authorized_hash = tensor_state_sha256(
        {name: value.detach().cpu() for name, value in named.items()}
    )
    if source_authorized_hash != _SOURCE_AUTHORIZED_SHA256:
        raise RuntimeError("V45 live authorized surface differs from V44-u8")

    split = v31_contract(loader)
    manifest_ids = (*split.train_scene_ids, *split.validation_scene_ids)
    caches, cache_audit = cache_v41_train_scenes(
        config=loader,
        bundle=bundle,
        source_metadata=source_checkpoint_metadata,
        scene_ids=split.train_scene_ids,
        manifest_scene_ids=manifest_ids,
    )
    cache_audit.update(
        {
            "scene_scope": "training_only",
            "authenticated_manifest_scene_count": len(manifest_ids),
            "authenticated_manifest_train_subset_count": len(split.train_scene_ids),
            "validation_scene_ids_loaded": [],
            "validation_environment_maps_loaded": False,
            "deferred_final_scene_ids_loaded": [],
        }
    )
    cache_boundary = validate_v37_training_cache_boundary(
        cache_audit=cache_audit,
        caches=caches,
        config=loader,
        train_scene_ids=split.train_scene_ids,
        validation_scene_ids=split.validation_scene_ids,
    )
    _prefix_replay_attestation(
        caches=caches,
        block_cross_residual=block_core,
        bundle=bundle,
        expected_scene_ids=split.train_scene_ids,
    )
    if tuple(sorted(caches)) != tuple(sorted(split.train_scene_ids)) or len(caches) != 16:
        raise RuntimeError("V45 must cache every one of the 16 training scenes")
    with torch.inference_mode():
        source_scene_tokens = {
            scene_id: current_scene_tokens(
                caches[scene_id], block_core, device=bundle.language.device
            )
            .detach()
            .cpu()
            .clone()
            for scene_id in sorted(caches)
        }
    source_prefix_hashes = {
        scene_id: tensor_state_sha256({"scene_tokens": value})
        for scene_id, value in source_scene_tokens.items()
    }
    source_pair, source_nll = training_pair_gate_diagnostics(
        units=units,
        caches=caches,
        block_cross_residual=block_core,
        bundle=bundle,
        settings=settings,
    )
    source_broad = training_broad_nll(
        records=broad_records,
        caches=caches,
        block_cross_residual=block_core,
        bundle=bundle,
    )
    validate_per_unit_nll_diagnostics(source_nll, source_pair)
    update_zero_attestation = validate_v45_update_zero_baseline(
        pair_metrics=source_pair, broad_nll=source_broad
    )
    source_scene_hash = tensor_state_sha256(
        {_PARAMETER_NAMES[0]: named[_PARAMETER_NAMES[0]].detach().cpu()}
    )
    source_query_hash = tensor_state_sha256(
        {
            name: named[name].detach().cpu()
            for name in (_PARAMETER_NAMES[1], _PARAMETER_NAMES[2])
        }
    )
    source_audit = {
        "construction_route": "exact_original_v41_then_strict_v44_u8_overlay",
        "construction_source_checkpoint": str(
            contract.construction_source_checkpoint
        ),
        "construction_source_full_tensor_state_sha256": _V41_FULL_SHA256,
        "source_checkpoint": str(contract.source_checkpoint),
        "source_file_sha256": dict(contract.source_file_sha256),
        "source_optimizer_file_sha256_provenance": (
            _SOURCE_OPTIMIZER_SHA256_PROVENANCE
        ),
        "source_full_tensor_state_sha256": tensor_state_sha256(source_tensors),
        "source_authorized_surface_state_sha256": source_authorized_hash,
        "source_frozen_state_sha256": _FROZEN_SHA256,
        "source_optimizer_file_opened": False,
        "source_optimizer_deserialized": False,
        "source_optimizer_state_loaded": False,
        "strict_overlay": True,
        "live_update_zero_diagnostic_attestation": update_zero_attestation,
    }
    optimizer = v45_optimizer(
        [named[_PARAMETER_NAMES[0]]],
        [named[_PARAMETER_NAMES[1]], named[_PARAMETER_NAMES[2]]],
        settings,
    )
    optimizer_audit = v45_optimizer_audit(optimizer)
    assert_v44_trainable_surface(bundle, block_core, optimizer=optimizer)
    history: list[dict[str, Any]] = [
        {
            "optimizer_update": 0,
            "source_pair_metrics": source_pair,
            "source_per_unit_nll_diagnostics": source_nll,
            "source_broad_train_nll": source_broad,
            "source_retention_diagnostics": v45_retention_diagnostics(source_pair),
            "source_update_zero_diagnostic_attestation": update_zero_attestation,
            "source_prefix_trust_rms": 0.0,
            "authorized_state_sha256": source_authorized_hash,
            "scene_readout_state_sha256": source_scene_hash,
            "query_state_sha256": source_query_hash,
            "frozen_state_sha256": _FROZEN_SHA256,
            "validation_qa_loaded": False,
            "oracle_environment_files_loaded": False,
            "saved_checkpoint": True,
        }
    ]
    gate4: Mapping[str, Any] | None = None
    gate8: Mapping[str, Any] | None = None
    completed_steps = 0
    stop_reason: str | None = None
    output.mkdir(parents=True, exist_ok=False)
    metadata0 = _metadata(
        source_metadata=source_checkpoint_metadata,
        config=config,
        terminal=terminal,
        history=history,
        optimizer_step=0,
        bundle=bundle,
        block_core=block_core,
        source_prefix_hashes=source_prefix_hashes,
        source_frozen_hash=_FROZEN_SHA256,
        source_audit=source_audit,
        schedule_audit=schedule_audit,
        gate4=None,
        gate8=None,
    )
    _save(output / "update_000", bundle=bundle, metadata=metadata0, optimizer=None)

    for item in schedule:
        step = item.optimizer_step
        freeze_for_v44(bundle, block_core)
        assert_v44_trainable_surface(bundle, block_core, optimizer=optimizer)
        optimizer.zero_grad(set_to_none=True)

        broad_tokens = current_scene_tokens(
            caches[item.broad_record.scene_id],
            block_core,
            device=bundle.language.device,
        )
        broad = broad_answer_nll(
            scene_tokens=broad_tokens, record=item.broad_record, bundle=bundle
        )
        broad_value = float(broad.detach().cpu())
        (settings.broad_nll_weight * broad).backward()
        del broad, broad_tokens

        target_tokens = _unit_tokens(
            item.target_unit,
            caches=caches,
            block_core=block_core,
            device=bundle.language.device,
        )
        pair_nll, side_hinge, cross_hinge, target_diagnostics = (
            paired_cross_prefix_objective(
                unit=item.target_unit,
                scene_tokens=target_tokens,
                bundle=bundle,
                side_margin=settings.target_side_hinge_margin,
                cross_prefix_margin=settings.target_cross_prefix_margin,
            )
        )
        pair_nll_value = float(pair_nll.detach().cpu())
        side_hinge_value = float(side_hinge.detach().cpu())
        cross_hinge_value = float(cross_hinge.detach().cpu())
        target_loss = (
            settings.pair_correct_nll_weight * pair_nll
            + settings.target_side_hinge_weight * side_hinge
            + settings.target_cross_prefix_weight * cross_hinge
        )
        target_loss.backward()
        del (
            target_loss,
            pair_nll,
            side_hinge,
            cross_hinge,
            target_diagnostics,
            target_tokens,
        )
        _mps_empty_cache(bundle.language.device)

        retention_values = _backward_v45_retention(
            units_by_key=units_by_key,
            caches=caches,
            block_core=block_core,
            bundle=bundle,
            settings=settings,
        )

        trust, _trust_rms = source_prefix_trust_penalty(
            caches=caches,
            references=source_scene_tokens,
            block_core=block_core,
            device=bundle.language.device,
            scale=settings.source_prefix_trust_scale,
        )
        trust_value = float(trust.detach().cpu())
        trust_loss = settings.source_prefix_trust_weight * trust
        trust_loss.backward()
        del trust_loss, trust
        loss_value = (
            settings.broad_nll_weight * broad_value
            + settings.pair_correct_nll_weight * pair_nll_value
            + settings.target_side_hinge_weight * side_hinge_value
            + settings.target_cross_prefix_weight * cross_hinge_value
            + float(retention_values["weighted_retention_loss"])
            + settings.source_prefix_trust_weight * trust_value
        )
        if any(
            parameter.grad is None
            or not torch.isfinite(parameter.grad).all()
            or torch.count_nonzero(parameter.grad).item() == 0
            for parameter in named.values()
        ):
            raise RuntimeError("V45 active gradient is absent, zero, or nonfinite")
        scene_preclip = float(
            torch.nn.utils.clip_grad_norm_(
                (named[_PARAMETER_NAMES[0]],), settings.gradient_clip_norm
            )
        )
        query_preclip = float(
            torch.nn.utils.clip_grad_norm_(
                (named[_PARAMETER_NAMES[1]], named[_PARAMETER_NAMES[2]]),
                settings.gradient_clip_norm,
            )
        )
        if not math.isfinite(scene_preclip) or not math.isfinite(query_preclip):
            raise RuntimeError("V45 per-group gradient norm is nonfinite")
        optimizer.step()
        completed_steps = step
        if frozen_v44_state_sha256(bundle) != _FROZEN_SHA256:
            raise RuntimeError("V45 changed a frozen tensor or buffer")

        should_save = step in settings.checkpoint_steps
        pair_metrics: Mapping[str, Any] | None = None
        per_unit_nll: list[dict[str, Any]] | None = None
        broad_diagnostic: float | None = None
        greedy_diagnostic: Mapping[str, Any] | None = None
        with torch.inference_mode():
            _trust_diagnostic, trust_rms_diagnostic = source_prefix_trust_penalty(
                caches=caches,
                references=source_scene_tokens,
                block_core=block_core,
                device=bundle.language.device,
                scale=settings.source_prefix_trust_scale,
            )
        trust_rms_value = float(trust_rms_diagnostic.detach().cpu())
        if step in (4, 8):
            pair_metrics, per_unit_nll = training_pair_gate_diagnostics(
                units=units,
                caches=caches,
                block_cross_residual=block_core,
                bundle=bundle,
                settings=settings,
            )
            broad_diagnostic = training_broad_nll(
                records=broad_records,
                caches=caches,
                block_cross_residual=block_core,
                bundle=bundle,
            )
            validate_per_unit_nll_diagnostics(per_unit_nll, pair_metrics)
        if step == 8:
            greedy_diagnostic = training_greedy_metrics(
                units=units,
                broad_records=broad_records,
                caches=caches,
                block_cross_residual=block_core,
                bundle=bundle,
                config=loader,
            )
        authorized_hash = tensor_state_sha256(
            {name: value.detach().cpu() for name, value in named.items()}
        )
        scene_hash = tensor_state_sha256(
            {_PARAMETER_NAMES[0]: named[_PARAMETER_NAMES[0]].detach().cpu()}
        )
        query_hash = tensor_state_sha256(
            {
                name: named[name].detach().cpu()
                for name in (_PARAMETER_NAMES[1], _PARAMETER_NAMES[2])
            }
        )
        scene_changed = scene_hash != source_scene_hash
        query_changed = query_hash != source_query_hash
        if step == 4:
            assert pair_metrics is not None and broad_diagnostic is not None
            gate4 = v45_update4_gate(
                pair_metrics=pair_metrics,
                broad_nll=broad_diagnostic,
                scene_readout_state_changed=scene_changed,
                query_state_changed=query_changed,
                frozen_state_exact=True,
                trust_rms=trust_rms_value,
            )
        if step == 8:
            assert (
                gate4 is not None
                and pair_metrics is not None
                and broad_diagnostic is not None
                and greedy_diagnostic is not None
            )
            gate8 = v45_update8_gate(
                update4_gate=gate4,
                pair_metrics=pair_metrics,
                broad_nll=broad_diagnostic,
                greedy_metrics=greedy_diagnostic,
                scene_readout_state_changed=scene_changed,
                query_state_changed=query_changed,
                frozen_state_exact=True,
                trust_rms=trust_rms_value,
            )
        history.append(
            {
                "optimizer_update": step,
                "true_optimizer_step": True,
                "target_pair_id": item.target_unit.pair_id,
                "target_question_key": item.target_unit.question_key,
                "broad_question_id": item.broad_record.question_id,
                "train_loss": loss_value,
                "train_broad_nll": broad_value,
                "train_pair_correct_nll": pair_nll_value,
                "train_target_side_hinge": side_hinge_value,
                "train_target_cross_prefix_hinge": cross_hinge_value,
                "train_retention": retention_values,
                "train_u8_prefix_trust_penalty": trust_value,
                "u8_prefix_trust_rms": trust_rms_value,
                "scene_readout_preclip_gradient_norm": scene_preclip,
                "query_preclip_gradient_norm": query_preclip,
                "per_group_gradient_clip_norm": settings.gradient_clip_norm,
                "pair_metrics": pair_metrics,
                "per_unit_nll_diagnostics": per_unit_nll,
                "broad_diagnostic_nll": broad_diagnostic,
                "training_greedy_metrics": greedy_diagnostic,
                "update4_train_only_gate": None if gate4 is None else dict(gate4),
                "update8_train_only_gate": None if gate8 is None else dict(gate8),
                "authorized_state_sha256": authorized_hash,
                "scene_readout_state_sha256": scene_hash,
                "query_state_sha256": query_hash,
                "scene_readout_state_changed": scene_changed,
                "query_state_changed": query_changed,
                "frozen_state_sha256": _FROZEN_SHA256,
                "validation_qa_loaded": False,
                "oracle_environment_files_loaded": False,
                "saved_checkpoint": should_save,
            }
        )
        if not should_save:
            continue
        metadata = _metadata(
            source_metadata=source_checkpoint_metadata,
            config=config,
            terminal=terminal,
            history=history,
            optimizer_step=step,
            bundle=bundle,
            block_core=block_core,
            source_prefix_hashes=source_prefix_hashes,
            source_frozen_hash=_FROZEN_SHA256,
            source_audit=source_audit,
            schedule_audit=schedule_audit,
            gate4=gate4,
            gate8=gate8,
        )
        _save(
            output / f"update_{step:03d}",
            bundle=bundle,
            metadata=metadata,
            optimizer=optimizer,
        )
        print(
            json.dumps(
                {
                    "phase": "v45_retention_repair_checkpoint",
                    "optimizer_step": step,
                    "update4_gate_passed": None
                    if gate4 is None
                    else gate4.get("passed"),
                    "update8_gate_passed": None
                    if gate8 is None
                    else gate8.get("passed"),
                    "u8_prefix_trust_rms": trust_rms_value,
                    "validation_qa_loaded": False,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        stop_reason = v45_stop_reason(
            step, update4_gate=gate4, update8_gate=gate8
        )
        if stop_reason is not None:
            break

    return {
        "schema_version": 1,
        "artifact": "v45_retention_repair_train_only_pilot",
        "passed": gate8 is not None and gate8.get("passed") is True,
        "bounded_training_completed": gate8 is not None,
        "stopped_at_train_only_gate": stop_reason,
        "output": str(output),
        "optimizer_updates": completed_steps,
        "saved_optimizer_steps": v45_saved_optimizer_steps(history),
        "terminal": terminal,
        "optimizer": optimizer_audit,
        "trainable_surface": surface,
        "source_audit": source_audit,
        "loader_transition": loader_transition,
        "cache_audit": cache_audit,
        "cache_boundary": cache_boundary,
        "qa_audit": qa_audit,
        "unit_audit": unit_audit,
        "schedule_audit": schedule_audit,
        "update4_train_only_gate": gate4,
        "update8_train_only_gate": gate8,
        "validation_qa_loaded": False,
        "oracle_loaded": False,
        "final_test_scenes_touched": False,
        "selector_execution_authorized": False,
        "runtime_promotion_authorized": False,
    }


def run_v45(
    *,
    config_path: str | Path = DEFAULT_CONFIG,
    output: str | Path = DEFAULT_OUTPUT,
    v44_terminal_sha256: str,
) -> dict[str, Any]:
    """Run V45 under a strict process-wide train-only file-read audit."""

    resolved_config = _resolve(config_path)
    config = load_config(resolved_config)
    protected = _resolve(_PROTECTED_REPORT)
    if protected.is_symlink() or not protected.is_file():
        raise ValueError("V45 protected selection report is unavailable or unsafe")
    protected_before = _sha256(protected)
    if protected_before != _PROTECTED_REPORT_SHA256:
        raise ValueError("V45 protected selection report changed before training")
    audit = FileAccessAudit(
        _training_forbidden_roots(config),
        forbidden_component_names={"oracle"},
        block_forbidden=True,
    )
    with audit:
        result = _run_v45_impl(
            config_path=resolved_config,
            output=output,
            v44_terminal_sha256=v44_terminal_sha256,
        )
    audit.assert_clean()
    if _sha256(protected) != protected_before:
        raise RuntimeError("V45 changed the protected selection report")
    loader = v41_loader_config(config)
    split = v31_contract(loader)
    expected_maps = {
        str((artifact_root(loader, "maps") / scene_id / "voxel_map.npz").resolve())
        for scene_id in split.train_scene_ids
    }
    observed_maps = {
        path for path in audit.unique_paths if path.endswith("/voxel_map.npz")
    }
    if observed_maps != expected_maps:
        raise RuntimeError("V45 file audit did not observe exactly all 16 training maps")
    optimizer_reads = [
        path for path in audit.unique_paths if path.endswith("/optimizer.pt")
    ]
    if optimizer_reads:
        raise RuntimeError(f"V45 opened a forbidden optimizer file: {optimizer_reads}")
    result.update(
        {
            "file_access_audit": {
                "passed": True,
                "loaded_files": audit.unique_paths,
                "loaded_training_maps": sorted(observed_maps),
                "forbidden_file_accesses": [],
                "validation_qa_loaded": False,
                "oracle_environment_files_loaded": False,
                "source_optimizer_file_opened": False,
                "source_optimizer_state_loaded": False,
                "protected_report_sha256_before_and_after": protected_before,
            },
            "all_16_training_maps_observed_by_process_audit": True,
        }
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--v44-terminal-sha256", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    result = (
        preflight_v45(
            args.config, v44_terminal_sha256=args.v44_terminal_sha256
        )
        if args.preflight_only
        else run_v45(
            config_path=args.config,
            output=args.output,
            v44_terminal_sha256=args.v44_terminal_sha256,
        )
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0 if result.get("passed") is True else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "V45Contract",
    "V45ScheduleRow",
    "V45Settings",
    "build_v45_schedule",
    "preflight_v45",
    "require_v44_terminal_gate",
    "retention_hinge_from_selected_margins",
    "run_v45",
    "v45_contract",
    "v45_optimizer",
    "v45_optimizer_audit",
    "v45_retention_diagnostics",
    "v45_saved_optimizer_steps",
    "v45_settings",
    "v45_stop_reason",
    "v45_update4_gate",
    "v45_update8_gate",
    "validate_v45_unit_inventory",
    "validate_v45_update_zero_baseline",
]
