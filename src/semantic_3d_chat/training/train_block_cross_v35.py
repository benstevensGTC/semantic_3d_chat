"""Bounded V35 all-block cross-residual training from exact V33 update 64.

V35 is deliberately narrow.  It never inherits V34's failed base-route
weights.  Gemma, every LoRA bank, and the complete V33 scene stack are loaded
from the exact numbered V33 update-64 checkpoint and frozen.  The only new
surface is a zero-output 983,040-parameter cross-attention residual from every
occupied spatial-block token into all 256 persistent scene slots.

Training reads only the persisted training QA.  Validation scene IDs are used
solely to prove question-free update-zero scene-prefix identity; validation QA
is deferred to the independent selector and is never opened here.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import re
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from semantic_3d_chat.config import PROJECT_ROOT, artifact_root, config_hash, load_config
from semantic_3d_chat.data.dataset import QARecord
from semantic_3d_chat.evaluation.v34_terminal_gate import audit_v34_update32
from semantic_3d_chat.language.local_lm import prompt_token_ids
from semantic_3d_chat.language.lora import tensor_state_sha256
from semantic_3d_chat.language.prefix_injection import prefix_sha256, stack_prefix_batches
from semantic_3d_chat.scene_encoder.block_cross_residual import (
    BlockCrossResidual,
    block_cross_residual_settings,
    construct_block_cross_residual,
    validate_block_cross_residual_state,
)
from semantic_3d_chat.scene_encoder.map_io import load_map_tensors
from semantic_3d_chat.training.checkpointing import (
    RUNTIME_METADATA_FILENAME,
    TRAINING_METADATA_FILENAME,
    load_adapter_checkpoint,
    load_optimizer_checkpoint,
    module_collection_state_sha256,
    runtime_checkpoint_metadata,
    save_adapter_checkpoint,
    save_optimizer_checkpoint,
    validate_runtime_checkpoint_metadata,
)
from semantic_3d_chat.training.pair_curriculum import (
    CounterfactualPairUnit,
    build_exact_question_pair_units,
    differing_answer_token_masks,
    pair_ranking_hinge,
    restrict_labels_to_answer_mask,
    token_normalized_nll,
)
from semantic_3d_chat.training.train_adapter import (
    forward_prefix_batch,
    load_qa_split_dataset,
    map_forward,
    tokenize_answer,
)
from semantic_3d_chat.training.train_environmental_sidecar_v33 import (
    assert_deferred_final_scenes_absent,
)
from semantic_3d_chat.training.train_joint_pair_v30 import (
    V30Bundle,
    load_v30_bundle,
    require_approved_v29_source,
    select_balanced_broad_records,
)
from semantic_3d_chat.training.train_joint_pair_v31 import V31Contract, v31_contract

DEFAULT_CONFIG = Path("configs/experiments/gemma4_diverse28_block_cross_v35.yaml")
DEFAULT_OUTPUT = Path("data_gemma4/checkpoints/gemma4_v35_diverse28_block_cross")
_UPDATE_DIRECTORY = re.compile(r"update_([0-9]{3})")
_TERMINAL_REPORT_SHA256 = "b0833a72ba5bc507178fa07cacc8cbef798fce4de94a5f85f2e402aafb46679f"
_SOURCE_TENSOR_STATE_SHA256 = "cb7bb3b48ace60212ee5c7f326839bf2ddd993810417de45c9a9cbc666313fe6"
_SOURCE_FILE_SHA256 = {
    "adapter.safetensors": "32c071d7acca0e52f8ae4c3dee8cba83319d67b184bbb3ab9957a6f6c4fcf987",
    TRAINING_METADATA_FILENAME: "ef97dfc3415eb4cfbdf30fe952e85db5ea4c54e4dec896a40725fb41fd787c91",
    RUNTIME_METADATA_FILENAME: "fe8df1c8c052ac50899eb19952f96b74ac691780e20b604ba4e11072db32e168",
    "optimizer.pt": "845aa42380b5c8c575162cb003fcadc7761fd615071b9dab71d9da4a85ba3d09",
}
_PARAMETER_NAMES = ("w_q", "w_k", "w_v", "w_o")
_QKV_NAMES = ("w_q", "w_k", "w_v")
_OUTPUT_NAMES = ("w_o",)
_PAIR_FAMILIES = {
    "book_support": "pair_000015",
    "mirror_lr": "pair_000016",
    "picture_support": "pair_000017",
}


@dataclass(frozen=True)
class V35Settings:
    enabled: bool
    optimizer_steps: int
    checkpoint_interval_steps: int
    broad_batch_size: int
    pair_units_per_step: int
    broad_exclude_expected_change: bool
    broad_nll_weight: float
    pair_correct_nll_weight: float
    side_hinge_weight: float
    side_hinge_margin: float
    cross_prefix_flip_weight: float
    cross_prefix_flip_margin: float
    residual_penalty_weight: float
    residual_penalty_scale: float
    qkv_learning_rate: float
    output_learning_rate: float
    weight_decay: float
    qkv_gradient_clip_norm: float
    output_gradient_clip_norm: float
    minimum_answer_types: int

    @property
    def saved_optimizer_steps(self) -> tuple[int, ...]:
        regular = tuple(range(0, self.optimizer_steps, self.checkpoint_interval_steps))
        return (*regular, self.optimizer_steps)


@dataclass(frozen=True)
class V35Contract:
    v31: V31Contract
    terminal_report: Path
    terminal_report_sha256: str
    source_checkpoint: Path
    source_file_sha256: Mapping[str, str]
    source_tensor_state_sha256: str
    core_initial_state_sha256: str
    saved_optimizer_steps: tuple[int, ...]
    update32_gate: Mapping[str, Any]
    update64_gate: Mapping[str, Any]


@dataclass(frozen=True)
class V35Microstep:
    optimizer_step: int
    broad_record: QARecord
    pair_unit: CounterfactualPairUnit


@dataclass(frozen=True)
class V35SceneCache:
    scene_id: str
    source_scene_tokens: torch.Tensor
    block_tokens: torch.Tensor
    block_positions_normalized: torch.Tensor
    source_prefix_sha256: str
    voxel_count: int
    processed_voxels: int
    occupied_block_count: int
    tokens_per_block: int


@dataclass(frozen=True)
class V35SeparationReference:
    source_prefixes: Mapping[str, torch.Tensor]
    changed_pairs: Mapping[str, tuple[str, str]]
    unrelated_pairs: tuple[tuple[str, str], ...]
    changed_rms: Mapping[str, float]
    unrelated_rms: Mapping[tuple[str, str], float]
    audit_sha256: str


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


def _positive_int(field: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _finite(field: str, value: object, *, positive: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed) or (parsed <= 0 if positive else parsed < 0):
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{field} must be finite and {qualifier}")
    return parsed


def v35_settings(config: Mapping[str, Any]) -> V35Settings:
    training = _mapping(config.get("training"), "training")
    raw = _mapping(training.get("v35_block_cross"), "training.v35_block_cross")
    fields = set(V35Settings.__dataclass_fields__)
    if set(raw) != fields:
        raise ValueError(
            "training.v35_block_cross fields differ from the locked schema: "
            f"missing={sorted(fields - set(raw))} unknown={sorted(set(raw) - fields)}"
        )
    for field in ("enabled", "broad_exclude_expected_change"):
        if not isinstance(raw[field], bool):
            raise TypeError(f"training.v35_block_cross.{field} must be boolean")
    settings = V35Settings(
        enabled=raw["enabled"],
        optimizer_steps=_positive_int("optimizer_steps", raw["optimizer_steps"]),
        checkpoint_interval_steps=_positive_int(
            "checkpoint_interval_steps", raw["checkpoint_interval_steps"]
        ),
        broad_batch_size=_positive_int("broad_batch_size", raw["broad_batch_size"]),
        pair_units_per_step=_positive_int(
            "pair_units_per_step", raw["pair_units_per_step"]
        ),
        broad_exclude_expected_change=raw["broad_exclude_expected_change"],
        broad_nll_weight=_finite("broad_nll_weight", raw["broad_nll_weight"], positive=True),
        pair_correct_nll_weight=_finite(
            "pair_correct_nll_weight", raw["pair_correct_nll_weight"], positive=True
        ),
        side_hinge_weight=_finite(
            "side_hinge_weight", raw["side_hinge_weight"], positive=True
        ),
        side_hinge_margin=_finite(
            "side_hinge_margin", raw["side_hinge_margin"], positive=True
        ),
        cross_prefix_flip_weight=_finite(
            "cross_prefix_flip_weight", raw["cross_prefix_flip_weight"], positive=True
        ),
        cross_prefix_flip_margin=_finite(
            "cross_prefix_flip_margin", raw["cross_prefix_flip_margin"], positive=True
        ),
        residual_penalty_weight=_finite(
            "residual_penalty_weight", raw["residual_penalty_weight"], positive=True
        ),
        residual_penalty_scale=_finite(
            "residual_penalty_scale", raw["residual_penalty_scale"], positive=True
        ),
        qkv_learning_rate=_finite(
            "qkv_learning_rate", raw["qkv_learning_rate"], positive=True
        ),
        output_learning_rate=_finite(
            "output_learning_rate", raw["output_learning_rate"], positive=True
        ),
        weight_decay=_finite("weight_decay", raw["weight_decay"], positive=False),
        qkv_gradient_clip_norm=_finite(
            "qkv_gradient_clip_norm", raw["qkv_gradient_clip_norm"], positive=True
        ),
        output_gradient_clip_norm=_finite(
            "output_gradient_clip_norm", raw["output_gradient_clip_norm"], positive=True
        ),
        minimum_answer_types=_positive_int(
            "minimum_answer_types", raw["minimum_answer_types"]
        ),
    )
    expected: Mapping[str, object] = {
        "enabled": True,
        "optimizer_steps": 100,
        "checkpoint_interval_steps": 8,
        "broad_batch_size": 1,
        "pair_units_per_step": 1,
        "broad_exclude_expected_change": True,
        "broad_nll_weight": 0.25,
        "pair_correct_nll_weight": 0.5,
        "side_hinge_weight": 4.0,
        "side_hinge_margin": 0.5,
        "cross_prefix_flip_weight": 8.0,
        "cross_prefix_flip_margin": 0.25,
        "residual_penalty_weight": 0.001,
        "residual_penalty_scale": 0.05,
        "qkv_learning_rate": 1e-4,
        "output_learning_rate": 2.5e-5,
        "weight_decay": 0.0,
        "qkv_gradient_clip_norm": 0.5,
        "output_gradient_clip_norm": 1.0,
        "minimum_answer_types": 4,
    }
    mismatches = {
        field: {"observed": getattr(settings, field), "expected": value}
        for field, value in expected.items()
        if getattr(settings, field) != value
    }
    if mismatches:
        raise ValueError(f"V35 locked optimizer/objective settings changed: {mismatches}")
    if settings.saved_optimizer_steps != (*range(0, 97, 8), 100):
        raise RuntimeError("V35 saved-step calculation changed")
    return settings


def _validate_core_config(config: Mapping[str, Any]) -> str:
    scene_encoder = _mapping(config.get("scene_encoder"), "scene_encoder")
    raw = _mapping(
        scene_encoder.get("block_cross_residual"),
        "scene_encoder.block_cross_residual",
    )
    required = {
        "enabled",
        "attention_dim",
        "heads",
        "spatial_temperature",
        "residual_scale",
        "uniform_floor",
        "initialization_seed",
        "expected_initial_state_sha256",
    }
    if set(raw) != required:
        raise ValueError(
            "block_cross_residual fields differ from the locked schema: "
            f"missing={sorted(required - set(raw))} unknown={sorted(set(raw) - required)}"
        )
    exact = {
        "enabled": True,
        "attention_dim": 256,
        "heads": 4,
        "spatial_temperature": 0.20,
        "residual_scale": 0.25,
        "uniform_floor": 0.01,
        "initialization_seed": 35035,
    }
    mismatches = {
        key: {"observed": raw.get(key), "expected": value}
        for key, value in exact.items()
        if raw.get(key) != value
    }
    if mismatches:
        raise ValueError(f"V35 locked core architecture changed: {mismatches}")
    initial_hash = str(raw["expected_initial_state_sha256"])
    if re.fullmatch(r"[0-9a-f]{64}", initial_hash) is None:
        raise ValueError("V35 core initial-state hash is not a pinned SHA-256")
    if initial_hash != "72ae7f492f5953e58d809b6782d559dc64669637d5d6a79ae98f3a31296a12cd":
        raise ValueError("V35 exact deterministic core initial-state hash changed")
    return initial_hash


def v35_contract(config: Mapping[str, Any]) -> V35Contract:
    v31 = v31_contract(config)
    settings = v35_settings(config)
    core_hash = _validate_core_config(config)
    raw = _mapping(config.get("v35_block_cross"), "v35_block_cross")
    required = {
        "schema_version",
        "role",
        "engine",
        "v34_terminal_gate_report",
        "v34_terminal_gate_report_sha256",
        "source_checkpoint",
        "source_optimizer_step",
        "source_file_sha256",
        "source_v33_config_sha256",
        "source_v33_schedule_sha256",
        "source_v33_tensor_state_sha256",
        "train_scene_ids",
        "validation_scene_ids",
        "deferred_final_scene_ids",
        "train_question_count",
        "train_changed_pair_unit_count",
        "optimizer_steps",
        "checkpoint_interval_steps",
        "saved_optimizer_steps",
        "exact_pair_unit_recurrence",
        "exact_unique_physical_changed_pair_count",
        "exact_nonchanged_train_scene_pair_count",
        "train_pair_family_ids",
        "exact_trainable_parameter_names",
        "exact_trainable_parameter_count",
        "step_1_trainable_parameter_names",
        "step_2_plus_trainable_parameter_names",
        "cache_scene_count",
        "cache_source_scene_tokens_dtype",
        "cache_block_tokens_dtype",
        "cache_block_positions_dtype",
        "cache_requires_processed_voxels_equal_voxel_count",
        "cache_requires_all_block_tokens",
        "cache_requires_repeated_normalized_block_positions",
        "cache_uses_question_or_answer_text",
        "cache_uses_oracle_environment_inputs",
        "cache_uses_question_dependent_retrieval",
        "exact_update0_source_tensors",
        "exact_update0_source_prefixes_all_22_scenes",
        "exact_update0_zero_residual_identity",
        "gemma_decoder_frozen",
        "all_lora_banks_frozen",
        "complete_v33_stack_frozen",
        "validation_qa_loaded_during_training",
        "continuation_gates_use_training_only",
        "update32_gate",
        "update64_gate",
        "selector_uses_validation_only_after_complete_training",
        "final_test_deferred",
    }
    if set(raw) != required:
        raise ValueError(
            "v35_block_cross fields differ from the locked schema: "
            f"missing={sorted(required - set(raw))} unknown={sorted(set(raw) - required)}"
        )
    exact: Mapping[str, object] = {
        "schema_version": 1,
        "role": "exact_v33_u64_block_token_cross_residual_v35",
        "engine": "exact_zero_all_block_cross_residual_true_microsteps",
        "v34_terminal_gate_report_sha256": _TERMINAL_REPORT_SHA256,
        "source_optimizer_step": 64,
        "source_v33_config_sha256": "e920d28da8ab0abc3c0ab2c4ad812a2743d1894b769c6302097ac41c31da3905",
        "source_v33_schedule_sha256": "90b7c3b337f573b47a75ed3faefc915eacd98c9ef11b572ff3c45c4166fc9590",
        "source_v33_tensor_state_sha256": _SOURCE_TENSOR_STATE_SHA256,
        "train_question_count": 384,
        "train_changed_pair_unit_count": 25,
        "optimizer_steps": 100,
        "checkpoint_interval_steps": 8,
        "exact_pair_unit_recurrence": 4,
        "exact_unique_physical_changed_pair_count": 8,
        "exact_nonchanged_train_scene_pair_count": 112,
        "exact_trainable_parameter_count": 983_040,
        "cache_scene_count": 22,
        "cache_source_scene_tokens_dtype": "float32_cpu",
        "cache_block_tokens_dtype": "float16_cpu",
        "cache_block_positions_dtype": "float16_cpu",
    }
    mismatches = {
        key: {"observed": raw.get(key), "expected": value}
        for key, value in exact.items()
        if raw.get(key) != value
    }
    if mismatches:
        raise ValueError(f"V35 locked contract changed: {mismatches}")
    true_fields = (
        "cache_requires_processed_voxels_equal_voxel_count",
        "cache_requires_all_block_tokens",
        "cache_requires_repeated_normalized_block_positions",
        "exact_update0_source_tensors",
        "exact_update0_source_prefixes_all_22_scenes",
        "exact_update0_zero_residual_identity",
        "gemma_decoder_frozen",
        "all_lora_banks_frozen",
        "complete_v33_stack_frozen",
        "continuation_gates_use_training_only",
        "selector_uses_validation_only_after_complete_training",
        "final_test_deferred",
    )
    if any(raw.get(field) is not True for field in true_fields):
        raise ValueError("V35 required true-valued safety field changed")
    false_fields = (
        "cache_uses_question_or_answer_text",
        "cache_uses_oracle_environment_inputs",
        "cache_uses_question_dependent_retrieval",
        "validation_qa_loaded_during_training",
    )
    if any(raw.get(field) is not False for field in false_fields):
        raise ValueError("V35 forbidden-input field changed")
    if tuple(raw["train_scene_ids"]) != v31.train_scene_ids:
        raise ValueError("V35 training scenes differ from the diverse28 lock")
    if tuple(raw["validation_scene_ids"]) != v31.validation_scene_ids:
        raise ValueError("V35 validation scene IDs differ from the diverse28 lock")
    if tuple(raw["deferred_final_scene_ids"]) != v31.deferred_final_scene_ids:
        raise ValueError("V35 deferred final scenes differ from the diverse28 lock")
    if tuple(raw["saved_optimizer_steps"]) != settings.saved_optimizer_steps:
        raise ValueError("V35 saved arms must be exactly 0,8,...,96,100")
    if tuple(raw["exact_trainable_parameter_names"]) != _PARAMETER_NAMES:
        raise ValueError("V35 trainable core tensor names changed")
    if tuple(raw["step_1_trainable_parameter_names"]) != _OUTPUT_NAMES:
        raise ValueError("V35 step-one output-only surface changed")
    if tuple(raw["step_2_plus_trainable_parameter_names"]) != _PARAMETER_NAMES:
        raise ValueError("V35 step-two-plus QKV/output surface changed")
    if dict(_mapping(raw["train_pair_family_ids"], "train_pair_family_ids")) != _PAIR_FAMILIES:
        raise ValueError("V35 train changed-pair family IDs changed")
    source_hashes = _mapping(raw["source_file_sha256"], "source_file_sha256")
    if dict(source_hashes) != _SOURCE_FILE_SHA256:
        raise ValueError("V35 exact V33 update-64 source hashes changed")
    update32 = _mapping(raw["update32_gate"], "update32_gate")
    expected32 = {
        "optimizer_step": 32,
        "changed_selectivity_geometric_mean_minimum": 1.02,
        "changed_pair_coverage_minimum": 6,
        "changed_pair_selectivity_minimum": 0.98,
        "unrelated_median_ratio_minimum": 0.98,
        "unrelated_median_ratio_maximum": 1.02,
        "unrelated_p90_abs_log_ratio_maximum": math.log(1.02),
        "require_mean_margin_strict_improvement": True,
        "require_complete_count_strict_improvement": True,
        "residual_rms_maximum": 0.10,
    }
    update64 = _mapping(raw["update64_gate"], "update64_gate")
    expected64 = {
        "optimizer_step": 64,
        "require_update32_passed": True,
        "complete_changed_qa_units_minimum": 8,
        "require_one_complete_per_family": True,
        "require_mean_margin_strict_improvement": True,
        "residual_rms_maximum": 0.10,
    }
    if dict(update32) != expected32 or dict(update64) != expected64:
        raise ValueError("V35 train-only continuation gates changed")
    return V35Contract(
        v31=v31,
        terminal_report=_resolve(str(raw["v34_terminal_gate_report"])),
        terminal_report_sha256=str(raw["v34_terminal_gate_report_sha256"]),
        source_checkpoint=_resolve(str(raw["source_checkpoint"])),
        source_file_sha256=dict(source_hashes),
        source_tensor_state_sha256=str(raw["source_v33_tensor_state_sha256"]),
        core_initial_state_sha256=core_hash,
        saved_optimizer_steps=settings.saved_optimizer_steps,
        update32_gate=dict(update32),
        update64_gate=dict(update64),
    )


def require_v34_terminal_gate(config: Mapping[str, Any]) -> dict[str, Any]:
    contract = v35_contract(config)
    path = contract.terminal_report
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"V35 requires a real V34 terminal report: {path}")
    observed_sha = _sha256(path)
    if observed_sha != contract.terminal_report_sha256:
        raise ValueError("V34 terminal report hash differs from V35's immutable pin")
    report = json.loads(path.read_text(encoding="utf-8"))
    audited = audit_v34_update32()
    if report != audited:
        raise ValueError("V34 terminal report does not replay from pinned metadata/tensors")
    authorization = _mapping(report.get("conditional_authorization"), "conditional_authorization")
    required = {
        "artifact": "v34_update32_terminal_gate",
        "passed": True,
        "stopped_at_optimizer_step": 32,
        "no_update_040_or_later": True,
        "final_test_scenes_touched": False,
        "oracle_loaded": False,
        "v34_development_selection_passed": False,
        "v34_chat_promotion_eligible": False,
        "conditional_v35_block_cross_residual_authorized": True,
    }
    mismatch = {
        key: {"observed": report.get(key), "expected": value}
        for key, value in required.items()
        if report.get(key) != value
    }
    if mismatch:
        raise ValueError(f"V34 terminal report does not authorize V35: {mismatch}")
    if dict(authorization) != {
        "all_other_followup_architectures_authorized": False,
        "authorized": True,
        "chat_promotion_authorized": False,
        "final_test_access_authorized": False,
        "scope": "exact_zero_block_token_cross_residual_only",
        "stage": "v35_block_cross_residual",
    }:
        raise ValueError("V34 terminal report authorizes a different follow-up scope")
    return {"path": str(path), "sha256": observed_sha, "report": report}


def require_exact_v33_source(config: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    contract = v35_contract(config)
    require_v34_terminal_gate(config)
    source = contract.source_checkpoint
    if source.is_symlink() or not source.is_dir() or source.name != "update_064":
        raise FileNotFoundError(f"V35 source must be real numbered V33 update 64: {source}")
    for filename, expected in contract.source_file_sha256.items():
        candidate = source / filename
        if candidate.is_symlink() or not candidate.is_file() or _sha256(candidate) != expected:
            raise ValueError(f"V35 source file differs from its exact pin: {candidate}")
    source_state = load_file(source / "adapter.safetensors", device="cpu")
    if tensor_state_sha256(source_state) != contract.source_tensor_state_sha256:
        raise ValueError("V35 source tensor-state hash differs from exact V33 update 64")
    metadata = json.loads((source / TRAINING_METADATA_FILENAME).read_text(encoding="utf-8"))
    if not isinstance(metadata, dict) or metadata.get("optimizer_step") != 64:
        raise ValueError("V35 source metadata is not V33 update 64")
    runtime = json.loads((source / RUNTIME_METADATA_FILENAME).read_text(encoding="utf-8"))
    validate_runtime_checkpoint_metadata(runtime)
    if runtime != runtime_checkpoint_metadata(metadata):
        raise ValueError("V35 source runtime metadata is not exact sanitized V33 source")
    return source, metadata


def load_v35_train_qa_records(
    config: Mapping[str, Any],
) -> tuple[list[QARecord], dict[str, Any]]:
    """Load only train.jsonl after validating the persisted scene split."""

    contract = v35_contract(config)
    qa_root = artifact_root(dict(config), "qa").resolve()
    manifest_path = qa_root / "splits.json"
    train_path = qa_root / "train.jsonl"
    validation_path = qa_root / "validation.jsonl"
    if not manifest_path.is_file() or not train_path.is_file():
        raise FileNotFoundError("V35 requires persisted splits.json and train.jsonl")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    splits = _mapping(manifest.get("splits"), "QA splits")
    observed_splits = {
        key: tuple(str(value) for value in splits.get(key, ()))
        for key in ("train", "validation", "test")
    }
    expected_splits = {
        "train": contract.v31.train_scene_ids,
        "validation": contract.v31.validation_scene_ids,
        "test": (),
    }
    if observed_splits != expected_splits:
        raise ValueError(
            "V35 persisted QA split differs from locked development scenes: "
            f"observed={observed_splits} expected={expected_splits}"
        )
    train = list(load_qa_split_dataset(qa_root, "train").records)
    if len(train) != 384:
        raise ValueError(f"V35 requires exactly 384 train QA rows, got {len(train)}")
    if tuple(sorted({record.scene_id for record in train})) != contract.v31.train_scene_ids:
        raise ValueError("V35 train QA does not cover exactly its 16 locked scenes")
    if {record.scene_id for record in train} & set(contract.v31.deferred_final_scene_ids):
        raise ValueError("V35 train QA touched deferred final scenes")
    units = build_exact_question_pair_units(train)
    observed_types = Counter(str(unit.reference.counterfactual_change_type) for unit in units)
    if len(units) != 25 or dict(observed_types) != dict(
        contract.v31.train_changed_pair_units_by_type
    ):
        raise ValueError("V35 train changed-pair distribution differs from its lock")
    return train, {
        "schema_version": 1,
        "qa_root": str(qa_root),
        "loaded_files": [str(manifest_path), str(train_path)],
        "train_question_count": len(train),
        "train_scene_ids": list(contract.v31.train_scene_ids),
        "train_changed_pair_unit_count": len(units),
        "validation_scene_ids_from_pinned_contract": list(contract.v31.validation_scene_ids),
        "validation_qa_path": str(validation_path),
        "validation_qa_loaded": False,
        "deferred_final_qa_loaded": False,
        "oracle_environment_files_loaded": False,
    }


def build_v35_schedule(
    records: Sequence[QARecord],
    pair_units: Sequence[CounterfactualPairUnit],
    *,
    settings: V35Settings,
    seed: int,
) -> tuple[list[V35Microstep], dict[str, Any]]:
    if len(pair_units) != 25:
        raise ValueError("V35 requires exactly 25 changed QA units")
    broad = select_balanced_broad_records(
        records,
        count=settings.optimizer_steps,
        seed=seed,
        exclude_expected_change=settings.broad_exclude_expected_change,
    )
    canonical = sorted(pair_units, key=lambda unit: (unit.pair_id, unit.question_key))
    scheduled: list[CounterfactualPairUnit] = []
    for recurrence in range(4):
        cycle = list(canonical)
        random.Random(seed + 35_000 + recurrence).shuffle(cycle)
        scheduled.extend(cycle)
    if len(scheduled) != 100 or len(broad) != 100:
        raise RuntimeError("V35 schedule must contain exactly 100 true microsteps")
    appearances = Counter((unit.pair_id, unit.question_key) for unit in scheduled)
    if set(appearances.values()) != {4} or len(appearances) != 25:
        raise RuntimeError("Every V35 changed QA unit must recur exactly four times")
    if any(record.counterfactual_expected_change is True for record in broad):
        raise RuntimeError("V35 broad schedule contains changed-pair supervision")
    if len({record.answer_type for record in broad}) < settings.minimum_answer_types:
        raise RuntimeError("V35 balanced broad schedule lacks answer-type coverage")
    steps = [
        V35Microstep(index + 1, broad[index], scheduled[index])
        for index in range(settings.optimizer_steps)
    ]
    payload = [
        {
            "optimizer_step": row.optimizer_step,
            "broad": (row.broad_record.scene_id, row.broad_record.question_id),
            "pair": (row.pair_unit.pair_id, row.pair_unit.question_key),
        }
        for row in steps
    ]
    return steps, {
        "schema_version": 1,
        "optimizer_step_count": 100,
        "true_optimizer_step_per_schedule_row": True,
        "broad_records_per_step": 1,
        "broad_expected_change_excluded": True,
        "broad_answer_type_counts": dict(
            sorted(Counter(record.answer_type for record in broad).items())
        ),
        "pair_units_per_step": 1,
        "pair_unit_count": 25,
        "exact_pair_unit_recurrence": 4,
        "pair_units_atomic": True,
        "schedule_sha256": hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "questions_or_answers_serialized_to_runtime": False,
    }


def physical_pair_sets(
    units: Sequence[CounterfactualPairUnit],
) -> tuple[dict[str, tuple[str, str]], tuple[tuple[str, str], ...]]:
    changed: dict[str, tuple[str, str]] = {}
    for unit in units:
        scenes = tuple(sorted(unit.scene_ids))
        prior = changed.setdefault(unit.pair_id, scenes)
        if prior != scenes:
            raise ValueError(f"Physical pair {unit.pair_id} changed scene membership")
    if len(changed) != 8:
        raise ValueError(f"V35 requires eight unique physical changed pairs, got {len(changed)}")
    scene_ids = sorted({scene for pair in changed.values() for scene in pair})
    if len(scene_ids) != 16:
        raise ValueError("V35 physical pairs must partition all 16 training scenes")
    changed_sets = {frozenset(pair) for pair in changed.values()}
    unrelated = tuple(
        pair
        for pair in combinations(scene_ids, 2)
        if frozenset(pair) not in changed_sets
    )
    if len(unrelated) != 112:
        raise RuntimeError("V35 nonchanged training-scene pair count must be 112")
    return dict(sorted(changed.items())), unrelated


def v35_update32_gate(
    *,
    separation: Mapping[str, Any],
    pair_metrics: Mapping[str, Any],
    baseline_pair_metrics: Mapping[str, Any],
    residual_rms: float,
    contract: V35Contract,
) -> dict[str, Any]:
    gate = contract.update32_gate
    checks = {
        "changed_selectivity_geometric_mean_at_least_1_02": (
            float(separation["changed_selectivity_ratio_geometric_mean"])
            >= float(gate["changed_selectivity_geometric_mean_minimum"])
        ),
        "at_least_6_of_8_changed_pairs_over_1_02": (
            int(separation["changed_selectivity_over_1_02_count"])
            >= int(gate["changed_pair_coverage_minimum"])
        ),
        "no_physical_pair_selectivity_below_0_98": (
            float(separation["changed_selectivity_ratio_minimum"])
            >= float(gate["changed_pair_selectivity_minimum"])
        ),
        "unrelated_median_two_sided_within_1_02": (
            float(gate["unrelated_median_ratio_minimum"])
            <= float(separation["unrelated_ratio_median"])
            <= float(gate["unrelated_median_ratio_maximum"])
        ),
        "unrelated_p90_abs_log_within_log_1_02": (
            float(separation["unrelated_abs_log_ratio_p90"])
            <= float(gate["unrelated_p90_abs_log_ratio_maximum"])
        ),
        "train_mean_margin_strictly_improved": (
            float(pair_metrics["mean_margin"]) > float(baseline_pair_metrics["mean_margin"])
        ),
        "train_complete_count_strictly_improved": (
            int(pair_metrics["complete_units"])
            > int(baseline_pair_metrics["complete_units"])
        ),
        "residual_rms_at_most_0_10": residual_rms <= float(gate["residual_rms_maximum"]),
    }
    return {**checks, "passed": all(checks.values()), "training_scenes_only": True}


def v35_update64_gate(
    *,
    update32_gate: Mapping[str, Any],
    pair_metrics: Mapping[str, Any],
    baseline_pair_metrics: Mapping[str, Any],
    residual_rms: float,
    contract: V35Contract,
) -> dict[str, Any]:
    gate = contract.update64_gate
    family = _mapping(pair_metrics.get("complete_units_by_family"), "complete_units_by_family")
    checks = {
        "update32_train_only_gate_remains_passed": update32_gate.get("passed") is True,
        "at_least_8_of_25_train_changed_qa_units_complete": (
            int(pair_metrics["complete_units"])
            >= int(gate["complete_changed_qa_units_minimum"])
        ),
        "book_support_has_complete_unit": int(family.get("book_support", 0)) >= 1,
        "mirror_lr_has_complete_unit": int(family.get("mirror_lr", 0)) >= 1,
        "picture_support_has_complete_unit": int(family.get("picture_support", 0)) >= 1,
        "train_mean_margin_strictly_improved": (
            float(pair_metrics["mean_margin"]) > float(baseline_pair_metrics["mean_margin"])
        ),
        "residual_rms_at_most_0_10": residual_rms <= float(gate["residual_rms_maximum"]),
    }
    return {**checks, "passed": all(checks.values()), "training_scenes_only": True}


def _audit_scalar(audit: Mapping[str, torch.Tensor], field: str) -> float:
    value = audit.get(field)
    if not isinstance(value, torch.Tensor) or value.numel() != 1:
        raise TypeError(f"Scene audit field {field!r} must be a scalar tensor")
    return float(value.detach().cpu())


def _inherited_pre_v33_prefixes(source_metadata: Mapping[str, Any]) -> dict[str, str]:
    """Return V33's inherited update-zero prefix provenance.

    These hashes describe the scene stack *before* V33 trained the dense
    sidecar.  V33 intentionally carries the original V30 cache audit forward
    in its metadata, so these values prove scene inventory and ancestry but
    must never be treated as post-V33 output hashes.
    """

    v30 = _mapping(source_metadata.get("v30_joint_pair"), "source v30_joint_pair")
    cache = _mapping(v30.get("scene_cache"), "source scene_cache")
    raw = _mapping(cache.get("source_prefix_sha256_by_scene"), "source prefix hashes")
    parsed = {str(key): str(value) for key, value in raw.items()}
    if len(parsed) != 22 or any(
        re.fullmatch(r"[0-9a-f]{64}", value) is None for value in parsed.values()
    ):
        raise ValueError("V33 source does not pin exactly 22 inherited prefix hashes")
    return parsed


def pinned_post_v33_prefix_manifest(
    *,
    source_metadata: Mapping[str, Any],
    terminal: Mapping[str, Any],
    expected_scene_ids: Sequence[str],
) -> dict[str, Any]:
    """Load the exact post-V33 hashes from V34's pinned update-zero audit.

    V34 update 32's metadata is byte-pinned by the audited terminal report.
    Its ``update_zero_equivalence`` was measured after exact V33 update 64 was
    loaded and before any V34 optimizer step.  That is the correct immutable
    reference boundary for V35; the V33 ``scene_cache`` hashes are older,
    pre-V33-sidecar provenance.
    """

    requested = tuple(sorted(set(expected_scene_ids)))
    if len(requested) != 22:
        raise ValueError("V35 post-V33 prefix manifest requires exactly 22 scenes")
    inherited = _inherited_pre_v33_prefixes(source_metadata)
    if tuple(sorted(inherited)) != requested:
        raise ValueError("V33 inherited prefix inventory differs from V35 scene IDs")

    report = _mapping(terminal.get("report"), "V34 terminal report")
    if (
        report.get("artifact") != "v34_update32_terminal_gate"
        or report.get("passed") is not True
        or report.get("stopped_at_optimizer_step") != 32
    ):
        raise ValueError("V35 post-V33 prefix manifest lacks an audited V34 terminal")
    update32 = _resolve(str(report.get("update32_checkpoint")))
    metadata_path = update32 / TRAINING_METADATA_FILENAME
    file_hashes = _mapping(report.get("update32_file_sha256"), "V34 update32 hashes")
    expected_metadata_sha = file_hashes.get(TRAINING_METADATA_FILENAME)
    if (
        metadata_path.is_symlink()
        or not metadata_path.is_file()
        or not isinstance(expected_metadata_sha, str)
        or re.fullmatch(r"[0-9a-f]{64}", expected_metadata_sha) is None
        or _sha256(metadata_path) != expected_metadata_sha
    ):
        raise ValueError("V34 post-V33 prefix metadata differs from its terminal pin")

    metadata_raw = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata = _mapping(metadata_raw, "V34 update32 metadata")
    if metadata.get("optimizer_step") != 32:
        raise ValueError("Pinned V34 post-V33 prefix metadata is not update 32")
    v30 = _mapping(metadata.get("v30_joint_pair"), "V34 update32 v30_joint_pair")
    cache = _mapping(v30.get("scene_cache"), "V34 inherited scene cache")
    v34_inherited_raw = _mapping(
        cache.get("source_prefix_sha256_by_scene"),
        "V34 inherited pre-V33 prefix hashes",
    )
    v34_inherited = {str(key): str(value) for key, value in v34_inherited_raw.items()}
    if v34_inherited != inherited:
        raise ValueError("V34 inherited pre-V33 prefix provenance changed")

    equivalence = _mapping(
        v30.get("update_zero_equivalence"), "V34 update-zero equivalence"
    )
    required_equivalence = {
        "exact_v33_update64_source_prefixes": True,
        "exact_v33_update64_source_tensors": True,
        "oracle_environment_files_loaded": False,
        "question_dependent_retrieval": False,
        "question_dependent_scene_processing": False,
        "source_prefix_scene_count": 22,
        "source_tensor_state_sha256": _SOURCE_TENSOR_STATE_SHA256,
    }
    mismatch = {
        key: {"observed": equivalence.get(key), "expected": value}
        for key, value in required_equivalence.items()
        if equivalence.get(key) != value
    }
    if mismatch:
        raise ValueError(
            f"V34 post-V33 update-zero prefix attestation changed: {mismatch}"
        )
    raw_post = _mapping(
        equivalence.get("source_prefix_sha256_by_scene"),
        "V34 post-V33 prefix hashes",
    )
    post = {str(key): str(value) for key, value in raw_post.items()}
    if tuple(sorted(post)) != requested or any(
        re.fullmatch(r"[0-9a-f]{64}", value) is None for value in post.values()
    ):
        raise ValueError("V34 does not pin 22 valid post-V33 source-prefix hashes")
    if any(post[scene_id] == inherited[scene_id] for scene_id in requested):
        raise ValueError("Post-V33 prefix manifest aliases inherited pre-V33 provenance")
    return {
        "post_v33_prefix_sha256_by_scene": post,
        "inherited_pre_v33_prefix_sha256_by_scene": inherited,
        "attesting_metadata_path": str(metadata_path),
        "attesting_metadata_sha256": expected_metadata_sha,
        "attesting_optimizer_step": 0,
        "carrier_checkpoint_optimizer_step": 32,
    }


def validate_v35_scene_cache(cache: V35SceneCache) -> None:
    if cache.source_scene_tokens.device.type != "cpu" or cache.source_scene_tokens.dtype != torch.float32:
        raise ValueError("V35 source scene tokens must be cached as float32 CPU")
    if cache.source_scene_tokens.shape != (1, 256, 1536):
        raise ValueError("V35 source scene tokens must have shape [1,256,1536]")
    if cache.block_tokens.device.type != "cpu" or cache.block_tokens.dtype != torch.float16:
        raise ValueError("V35 block tokens must be cached as float16 CPU")
    if cache.block_tokens.ndim != 2 or cache.block_tokens.shape[1] != 384:
        raise ValueError("V35 block tokens must have shape [T,384]")
    positions = cache.block_positions_normalized
    if positions.device.type != "cpu" or positions.dtype != torch.float16:
        raise ValueError("V35 normalized block positions must be cached as float16 CPU")
    if positions.shape != (cache.block_tokens.shape[0], 3):
        raise ValueError("V35 normalized block positions must align with every block token")
    if cache.processed_voxels != cache.voxel_count or cache.voxel_count <= 0:
        raise ValueError("V35 cache omitted one or more occupied voxels")
    expected_tokens = cache.tokens_per_block * cache.occupied_block_count
    if cache.block_tokens.shape[0] != expected_tokens:
        raise ValueError("V35 cache omitted an occupied-block token")
    if cache.tokens_per_block < 2 or cache.occupied_block_count <= 0:
        raise ValueError("V35 cache has invalid block coverage cardinality")
    repeated = positions.reshape(cache.occupied_block_count, cache.tokens_per_block, 3)
    if not torch.equal(repeated, repeated[:, :1].expand_as(repeated)):
        raise ValueError("V35 cache does not retain every repeated normalized block position")
    if not torch.isfinite(cache.source_scene_tokens).all():
        raise ValueError("V35 source scene tokens contain NaN or infinity")
    if not torch.isfinite(cache.block_tokens).all() or not torch.isfinite(positions).all():
        raise ValueError("V35 block cache contains NaN or infinity")
    if re.fullmatch(r"[0-9a-f]{64}", cache.source_prefix_sha256) is None:
        raise ValueError("V35 cache contains an invalid source-prefix hash")


def validate_v35_cache_audit(
    audit: Mapping[str, Any], *, expected_scene_ids: Sequence[str]
) -> None:
    expected = tuple(sorted(expected_scene_ids))
    expected_count = len(expected)
    if expected_count <= 0:
        raise ValueError("V35 cache audit requires at least one expected scene")
    required = {
        "cache_boundary": "exact_post_v33_scene_tokens_plus_all_frozen_block_tokens",
        "scene_count": expected_count,
        "scene_ids": list(expected),
        "source_scene_tokens_dtype": "torch.float32_cpu",
        "block_tokens_dtype": "torch.float16_cpu",
        "block_positions_dtype": "torch.float16_cpu",
        "all_voxels_covered": True,
        "all_occupied_blocks_processed": True,
        "all_block_tokens_cached": True,
        "all_repeated_normalized_block_positions_cached": True,
        "source_prefixes_match_exact_v33_update64": True,
        "source_prefixes_match_terminal_pinned_post_v33_manifest": True,
        "inherited_prefixes_treated_as_pre_v33_provenance_only": True,
        "question_inputs_to_scene_cache": False,
        "answer_inputs_to_scene_cache": False,
        "question_dependent_scene_processing": False,
        "question_dependent_retrieval": False,
        "oracle_environment_files_loaded": False,
        "validation_qa_loaded": False,
    }
    mismatch = {
        key: {"observed": audit.get(key), "expected": value}
        for key, value in required.items()
        if audit.get(key) != value
    }
    if mismatch:
        raise ValueError(f"V35 scene-cache audit failed closed: {mismatch}")
    counts = _mapping(audit.get("coverage_by_scene"), "coverage_by_scene")
    hashes = _mapping(audit.get("source_prefix_sha256_by_scene"), "source hashes")
    inherited_hashes = _mapping(
        audit.get("inherited_pre_v33_prefix_sha256_by_scene"),
        "inherited pre-V33 hashes",
    )
    if (
        tuple(sorted(counts)) != expected
        or tuple(sorted(hashes)) != expected
        or tuple(sorted(inherited_hashes)) != expected
    ):
        raise ValueError("V35 cache audit does not cover exactly the requested pinned scenes")
    if any(hashes[scene_id] == inherited_hashes[scene_id] for scene_id in expected):
        raise ValueError("V35 cache confused pre-V33 provenance with post-V33 output")
    attestation = _mapping(
        audit.get("post_v33_prefix_manifest_attestation"),
        "post-V33 prefix manifest attestation",
    )
    if (
        attestation.get("attesting_optimizer_step") != 0
        or attestation.get("carrier_checkpoint_optimizer_step") != 32
        or not isinstance(attestation.get("attesting_metadata_path"), str)
        or re.fullmatch(
            r"[0-9a-f]{64}", str(attestation.get("attesting_metadata_sha256"))
        )
        is None
    ):
        raise ValueError("V35 post-V33 prefix manifest attestation is incomplete")
    for scene_id, raw in counts.items():
        item = _mapping(raw, f"coverage_by_scene.{scene_id}")
        if int(item["processed_voxels"]) != int(item["voxel_count"]):
            raise ValueError(f"V35 cache omitted voxels for {scene_id}")
        if int(item["token_count"]) != int(item["tokens_per_block"]) * int(
            item["occupied_block_count"]
        ):
            raise ValueError(f"V35 cache omitted block tokens for {scene_id}")
    loaded = audit.get("loaded_environment_files")
    if not isinstance(loaded, list) or len(loaded) != expected_count:
        raise ValueError("V35 cache must record exactly one numeric map file per requested scene")
    if any("oracle" in {part.casefold() for part in Path(path).parts} for path in loaded):
        raise ValueError("V35 cache audit contains an oracle path")


def cache_v35_scenes(
    *,
    config: Mapping[str, Any],
    bundle: V30Bundle,
    source_metadata: Mapping[str, Any],
    terminal: Mapping[str, Any],
    scene_ids: Sequence[str],
    manifest_scene_ids: Sequence[str] | None = None,
) -> tuple[dict[str, V35SceneCache], dict[str, Any]]:
    """Cache exact post-V33 tokens and all question-free occupied-block tokens."""

    requested = tuple(sorted(set(scene_ids)))
    manifest_requested = tuple(
        sorted(set(requested if manifest_scene_ids is None else manifest_scene_ids))
    )
    if not requested or not set(requested).issubset(manifest_requested):
        raise ValueError("V35 cache scenes must be a nonempty subset of the pinned manifest")
    prefix_manifest = pinned_post_v33_prefix_manifest(
        source_metadata=source_metadata,
        terminal=terminal,
        expected_scene_ids=manifest_requested,
    )
    expected_hashes = _mapping(
        prefix_manifest["post_v33_prefix_sha256_by_scene"],
        "pinned post-V33 prefix hashes",
    )
    model_dtype = next(bundle.language.model.parameters()).dtype
    tokens_per_block = int(bundle.scene_model.block_encoder.tokens_per_block)
    caches: dict[str, V35SceneCache] = {}
    loaded_files: list[str] = []
    coverage: dict[str, dict[str, int]] = {}
    started = time.perf_counter()
    for scene_id in requested:
        map_path = (artifact_root(dict(config), "maps") / scene_id / "voxel_map.npz").resolve()
        if "oracle" in {part.casefold() for part in map_path.parts}:
            raise RuntimeError("V35 refuses oracle environmental input")
        data = load_map_tensors(
            map_path,
            config["scene"]["room_size_m"],
            bundle.language.device,
            input_voxel_size_m=config["scene_encoder"].get("input_voxel_size_m"),
        )
        with torch.inference_mode():
            output = map_forward(
                bundle.scene_model,
                data,
                bundle.global_scene_residual,
                bundle.signed_x_scene_residual,
                bundle.dense_aligner,
                None,
            )
            if output.aligned_sidecar_tokens is None:
                raise RuntimeError(f"V35 source lacks all-voxel sidecar tokens: {scene_id}")
            source_tokens = bundle.dense_sidecar_adapter(
                output.scene_tokens, output.aligned_sidecar_tokens
            )
            source_prefix = bundle.composer.scene_prefix(source_tokens.to(model_dtype))
            observed_hash = prefix_sha256(source_prefix)
            if observed_hash != expected_hashes[scene_id]:
                raise RuntimeError(
                    f"V35 post-V33 source prefix mismatch for {scene_id}: "
                    f"expected={expected_hashes[scene_id]} observed={observed_hash}"
                )
            processed = int(_audit_scalar(output.audit, "processed_voxels"))
            occupied_blocks = int(output.audit["block_indices"].shape[0])
            cache = V35SceneCache(
                scene_id=scene_id,
                source_scene_tokens=source_tokens.detach().float().cpu().contiguous(),
                block_tokens=output.block_tokens.detach().to(device="cpu", dtype=torch.float16).contiguous(),
                block_positions_normalized=output.audit[
                    "block_token_positions_normalized"
                ].detach().to(device="cpu", dtype=torch.float16).contiguous(),
                source_prefix_sha256=observed_hash,
                voxel_count=int(data.voxel_count),
                processed_voxels=processed,
                occupied_block_count=occupied_blocks,
                tokens_per_block=tokens_per_block,
            )
            validate_v35_scene_cache(cache)
            reconstructed = bundle.composer.scene_prefix(
                cache.source_scene_tokens.to(bundle.language.device).to(model_dtype)
            )
            if prefix_sha256(reconstructed) != observed_hash:
                raise RuntimeError(f"V35 fp32 CPU source-token cache changed prefix: {scene_id}")
            caches[scene_id] = cache
            coverage[scene_id] = {
                "voxel_count": cache.voxel_count,
                "processed_voxels": cache.processed_voxels,
                "occupied_block_count": cache.occupied_block_count,
                "tokens_per_block": cache.tokens_per_block,
                "token_count": int(cache.block_tokens.shape[0]),
            }
        loaded_files.append(str(map_path))
        del data, output
        if bundle.language.device.type == "mps":
            torch.mps.empty_cache()
    audit: dict[str, Any] = {
        "schema_version": 1,
        "cache_boundary": "exact_post_v33_scene_tokens_plus_all_frozen_block_tokens",
        "scene_count": len(caches),
        "scene_ids": list(requested),
        "cache_build_seconds": time.perf_counter() - started,
        "source_scene_tokens_dtype": "torch.float32_cpu",
        "block_tokens_dtype": "torch.float16_cpu",
        "block_positions_dtype": "torch.float16_cpu",
        "all_voxels_covered": True,
        "all_occupied_blocks_processed": True,
        "all_block_tokens_cached": True,
        "all_repeated_normalized_block_positions_cached": True,
        "source_prefixes_match_exact_v33_update64": True,
        "source_prefixes_match_terminal_pinned_post_v33_manifest": True,
        "inherited_prefixes_treated_as_pre_v33_provenance_only": True,
        "question_inputs_to_scene_cache": False,
        "answer_inputs_to_scene_cache": False,
        "question_dependent_scene_processing": False,
        "question_dependent_retrieval": False,
        "oracle_environment_files_loaded": False,
        "validation_qa_loaded": False,
        "loaded_environment_files": loaded_files,
        "source_prefix_sha256_by_scene": {
            scene_id: caches[scene_id].source_prefix_sha256 for scene_id in requested
        },
        "inherited_pre_v33_prefix_sha256_by_scene": {
            scene_id: prefix_manifest["inherited_pre_v33_prefix_sha256_by_scene"][scene_id]
            for scene_id in requested
        },
        "post_v33_prefix_manifest_attestation": {
            key: prefix_manifest[key]
            for key in (
                "attesting_metadata_path",
                "attesting_metadata_sha256",
                "attesting_optimizer_step",
                "carrier_checkpoint_optimizer_step",
            )
        },
        "coverage_by_scene": coverage,
    }
    validate_v35_cache_audit(audit, expected_scene_ids=requested)
    return caches, audit


def current_scene_tokens(
    cache: V35SceneCache,
    block_cross_residual: torch.nn.Module,
    *,
    device: torch.device,
) -> torch.Tensor:
    source = cache.source_scene_tokens.to(device=device, dtype=torch.float32)
    return block_cross_residual(
        source,
        cache.block_tokens.to(device=device, dtype=source.dtype),
        cache.block_positions_normalized.to(device=device, dtype=source.dtype),
    )


def _compose_answer_batch(
    *,
    scene_tokens: torch.Tensor,
    question: str,
    answer: str,
    bundle: V30Bundle,
):
    prompt_ids = prompt_token_ids(
        bundle.language.tokenizer,
        str(bundle.config["language"]["system_prompt"]),
        question,
        bundle.language.device,
    )
    answer_ids = tokenize_answer(bundle.language.tokenizer, answer, bundle.language.device)
    return bundle.composer.compose(
        scene_tokens.to(next(bundle.language.model.parameters()).dtype),
        prompt_ids,
        bundle.language.model.get_input_embeddings(),
        answer_ids,
        prefix_backend=getattr(bundle.language, "prefix_backend", None),
    )


def broad_answer_nll(
    *, scene_tokens: torch.Tensor, record: QARecord, bundle: V30Bundle
) -> torch.Tensor:
    batch = stack_prefix_batches(
        [
            _compose_answer_batch(
                scene_tokens=scene_tokens,
                question=record.question,
                answer=record.answer,
                bundle=bundle,
            )
        ],
        bundle.language.device,
        prefix_backend=getattr(bundle.language, "prefix_backend", None),
    )
    output = forward_prefix_batch(bundle.language, batch)
    if batch.labels is None:
        raise RuntimeError("V35 broad batch lacks answer labels")
    nll = token_normalized_nll(output.logits, batch.labels).mean()
    if nll.ndim != 0 or not torch.isfinite(nll):
        raise RuntimeError("V35 broad answer NLL is invalid")
    return nll


def pair_and_cross_prefix_hinges(
    *,
    correct_rank_nll: torch.Tensor,
    swapped_rank_nll: torch.Tensor,
    side_margin: float,
    cross_prefix_margin: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return within-prefix and same-answer-across-prefix margins/hinges."""

    if correct_rank_nll.shape != (2,) or swapped_rank_nll.shape != (2,):
        raise ValueError("V35 exact pair score vectors must each have shape [2]")
    side_hinge, side_margins = pair_ranking_hinge(
        correct_rank_nll.reshape(1, 2),
        swapped_rank_nll.reshape(1, 2),
        margin=side_margin,
    )
    cross_prefix_margins = torch.stack(
        (
            swapped_rank_nll[1] - correct_rank_nll[0],
            swapped_rank_nll[0] - correct_rank_nll[1],
        )
    )
    cross_prefix_hinge = torch.relu(cross_prefix_margin - cross_prefix_margins).mean()
    return side_hinge, side_margins.reshape(2), cross_prefix_hinge, cross_prefix_margins


def paired_cross_prefix_objective(
    *,
    unit: CounterfactualPairUnit,
    scene_tokens: Mapping[str, torch.Tensor],
    bundle: V30Bundle,
    side_margin: float,
    cross_prefix_margin: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Score the 2x2 answer/prefix matrix at exact differing-token positions.

    Side margins compare A-vs-B within each scene.  Cross-prefix margins compare
    the same answer token under its own physical scene and the paired scene.
    They are therefore genuine cross-prefix scores, not a scene-vector distance.
    """

    first, second = unit.records
    if first.question != second.question:
        raise ValueError("V35 atomic pair must use an identical question on both scenes")
    first_ids = tokenize_answer(bundle.language.tokenizer, first.answer, bundle.language.device)
    second_ids = tokenize_answer(bundle.language.tokenizer, second.answer, bundle.language.device)
    first_mask, second_mask = differing_answer_token_masks(first_ids, second_ids)
    correct_specs = ((first, first.answer), (second, second.answer))
    swapped_specs = ((first, second.answer), (second, first.answer))
    correct_batches = [
        _compose_answer_batch(
            scene_tokens=scene_tokens[record.scene_id],
            question=record.question,
            answer=answer,
            bundle=bundle,
        )
        for record, answer in correct_specs
    ]
    swapped_batches = [
        _compose_answer_batch(
            scene_tokens=scene_tokens[record.scene_id],
            question=record.question,
            answer=answer,
            bundle=bundle,
        )
        for record, answer in swapped_specs
    ]
    correct = stack_prefix_batches(
        correct_batches,
        bundle.language.device,
        prefix_backend=getattr(bundle.language, "prefix_backend", None),
    )
    correct_output = forward_prefix_batch(bundle.language, correct)
    if correct.labels is None:
        raise RuntimeError("V35 correct pair batch lacks labels")
    correct_answer_nll = token_normalized_nll(correct_output.logits, correct.labels)
    correct_rank_labels = correct.labels.clone()
    restrict_labels_to_answer_mask(correct_rank_labels, 0, first_mask)
    restrict_labels_to_answer_mask(correct_rank_labels, 1, second_mask)
    correct_rank_nll = token_normalized_nll(correct_output.logits, correct_rank_labels)
    del correct_output, correct

    swapped = stack_prefix_batches(
        swapped_batches,
        bundle.language.device,
        prefix_backend=getattr(bundle.language, "prefix_backend", None),
    )
    swapped_output = forward_prefix_batch(bundle.language, swapped)
    if swapped.labels is None:
        raise RuntimeError("V35 swapped pair batch lacks labels")
    swapped_rank_labels = swapped.labels.clone()
    restrict_labels_to_answer_mask(swapped_rank_labels, 0, second_mask)
    restrict_labels_to_answer_mask(swapped_rank_labels, 1, first_mask)
    swapped_rank_nll = token_normalized_nll(swapped_output.logits, swapped_rank_labels)

    side_hinge, side_margins, cross_prefix_hinge, cross_prefix_margins = (
        pair_and_cross_prefix_hinges(
            correct_rank_nll=correct_rank_nll,
            swapped_rank_nll=swapped_rank_nll,
            side_margin=side_margin,
            cross_prefix_margin=cross_prefix_margin,
        )
    )
    correct_nll = correct_answer_nll.mean()
    if not all(torch.isfinite(value) for value in (correct_nll, side_hinge, cross_prefix_hinge)):
        raise RuntimeError("V35 pair/cross-prefix objective is nonfinite")
    return correct_nll, side_hinge, cross_prefix_hinge, {
        "side_margins": side_margins.reshape(2),
        "cross_prefix_margins": cross_prefix_margins,
        "correct_answer_nll": correct_answer_nll,
        "correct_ranking_nll": correct_rank_nll,
        "swapped_ranking_nll": swapped_rank_nll,
        "true_cross_prefix_differing_token_scores": torch.tensor(
            True, device=correct_nll.device
        ),
    }


def residual_penalty(
    *,
    caches: Mapping[str, V35SceneCache],
    block_cross_residual: torch.nn.Module,
    device: torch.device,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    squared: list[torch.Tensor] = []
    for scene_id in sorted(caches):
        cache = caches[scene_id]
        source = cache.source_scene_tokens.to(device)
        current = current_scene_tokens(cache, block_cross_residual, device=device)
        squared.append((current - source).square().mean())
    mean_square = torch.stack(squared).mean()
    return mean_square / (scale**2), mean_square.sqrt()


def v35_weighted_objective(
    *,
    broad_nll: torch.Tensor,
    pair_correct_nll: torch.Tensor,
    side_hinge: torch.Tensor,
    cross_prefix_flip_hinge: torch.Tensor,
    normalized_residual_penalty: torch.Tensor,
    settings: V35Settings,
) -> torch.Tensor:
    return (
        settings.broad_nll_weight * broad_nll
        + settings.pair_correct_nll_weight * pair_correct_nll
        + settings.side_hinge_weight * side_hinge
        + settings.cross_prefix_flip_weight * cross_prefix_flip_hinge
        + settings.residual_penalty_weight * normalized_residual_penalty
    )


def _rms_difference(left: torch.Tensor, right: torch.Tensor, floor: float = 1e-6) -> torch.Tensor:
    return ((left - right).square().mean() + floor**2).sqrt()


def _scene_prefixes(
    *,
    caches: Mapping[str, V35SceneCache],
    block_cross_residual: torch.nn.Module,
    bundle: V30Bundle,
    scene_ids: Sequence[str],
) -> dict[str, torch.Tensor]:
    model_dtype = next(bundle.language.model.parameters()).dtype
    return {
        scene_id: bundle.composer.scene_prefix(
            current_scene_tokens(
                caches[scene_id], block_cross_residual, device=bundle.language.device
            ).to(model_dtype)
        ).float()
        for scene_id in scene_ids
    }


def build_v35_separation_reference(
    *,
    units: Sequence[CounterfactualPairUnit],
    caches: Mapping[str, V35SceneCache],
    block_cross_residual: torch.nn.Module,
    bundle: V30Bundle,
) -> V35SeparationReference:
    changed, unrelated = physical_pair_sets(units)
    scene_ids = sorted({scene_id for pair in changed.values() for scene_id in pair})
    with torch.inference_mode():
        prefixes = _scene_prefixes(
            caches=caches,
            block_cross_residual=block_cross_residual,
            bundle=bundle,
            scene_ids=scene_ids,
        )
    changed_rms = {
        pair_id: float(_rms_difference(prefixes[left], prefixes[right]))
        for pair_id, (left, right) in changed.items()
    }
    unrelated_rms = {
        pair: float(_rms_difference(prefixes[pair[0]], prefixes[pair[1]]))
        for pair in unrelated
    }
    if min(*changed_rms.values(), *unrelated_rms.values()) <= 1e-6:
        raise ValueError("V35 update-zero prefix separation reached the RMS floor")
    payload = {
        "changed_pairs": changed,
        "unrelated_pairs": unrelated,
        "changed_rms": changed_rms,
        "unrelated_rms": {"|".join(key): value for key, value in unrelated_rms.items()},
    }
    audit_hash = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return V35SeparationReference(
        source_prefixes={key: value.detach().cpu().clone() for key, value in prefixes.items()},
        changed_pairs=changed,
        unrelated_pairs=unrelated,
        changed_rms=changed_rms,
        unrelated_rms=unrelated_rms,
        audit_sha256=audit_hash,
    )


def v35_separation_diagnostics(
    *,
    reference: V35SeparationReference,
    caches: Mapping[str, V35SceneCache],
    block_cross_residual: torch.nn.Module,
    bundle: V30Bundle,
) -> dict[str, Any]:
    with torch.inference_mode():
        prefixes = _scene_prefixes(
            caches=caches,
            block_cross_residual=block_cross_residual,
            bundle=bundle,
            scene_ids=sorted(reference.source_prefixes),
        )
        changed_ratios = torch.stack(
            [
                _rms_difference(prefixes[left], prefixes[right])
                / reference.changed_rms[pair_id]
                for pair_id, (left, right) in reference.changed_pairs.items()
            ]
        )
        unrelated_ratios = torch.stack(
            [
                _rms_difference(prefixes[left], prefixes[right])
                / reference.unrelated_rms[(left, right)]
                for left, right in reference.unrelated_pairs
            ]
        )
        selectivity = (
            changed_ratios.log() - unrelated_ratios.log().mean()
        ).exp()
        abs_log = unrelated_ratios.log().abs()
    return {
        "schema_version": 1,
        "unique_changed_physical_pair_count": int(changed_ratios.numel()),
        "all_nonchanged_train_scene_pair_count": int(unrelated_ratios.numel()),
        "changed_ratio_mean": float(changed_ratios.mean()),
        "changed_ratio_minimum": float(changed_ratios.min()),
        "changed_ratio_maximum": float(changed_ratios.max()),
        "unrelated_ratio_mean": float(unrelated_ratios.mean()),
        "unrelated_ratio_median": float(torch.quantile(unrelated_ratios, 0.5)),
        "unrelated_abs_log_ratio_p90": float(torch.quantile(abs_log, 0.9)),
        "unrelated_abs_log_ratio_maximum": float(abs_log.max()),
        "changed_selectivity_ratio_geometric_mean": float(selectivity.log().mean().exp()),
        "changed_selectivity_ratio_minimum": float(selectivity.min()),
        "changed_selectivity_over_1_02_count": int((selectivity >= 1.02).sum()),
        "changed_selectivity_ratios_by_pair": {
            pair_id: float(value)
            for pair_id, value in zip(reference.changed_pairs, selectivity, strict=True)
        },
        "question_or_answer_text_used": False,
        "oracle_environment_inputs_used": False,
        "validation_scenes_used": False,
    }


def training_pair_metrics(
    *,
    units: Sequence[CounterfactualPairUnit],
    caches: Mapping[str, V35SceneCache],
    block_cross_residual: torch.nn.Module,
    bundle: V30Bundle,
    settings: V35Settings,
) -> dict[str, Any]:
    margins: list[torch.Tensor] = []
    cross_margins: list[torch.Tensor] = []
    rows: list[dict[str, Any]] = []
    complete_by_family = {name: 0 for name in _PAIR_FAMILIES}
    block_cross_residual.eval()
    for unit in sorted(units, key=lambda value: (value.pair_id, value.question_key)):
        with torch.inference_mode():
            tokens = {
                scene_id: current_scene_tokens(
                    caches[scene_id],
                    block_cross_residual,
                    device=bundle.language.device,
                )
                for scene_id in unit.scene_ids
            }
            _, _, _, diagnostics = paired_cross_prefix_objective(
                unit=unit,
                scene_tokens=tokens,
                bundle=bundle,
                side_margin=settings.side_hinge_margin,
                cross_prefix_margin=settings.cross_prefix_flip_margin,
            )
        side = diagnostics["side_margins"].detach().float().cpu()
        cross = diagnostics["cross_prefix_margins"].detach().float().cpu()
        complete = bool(side.gt(0).all())
        family = next(
            (name for name, pair_id in _PAIR_FAMILIES.items() if unit.pair_id == pair_id),
            "other",
        )
        if complete and family in complete_by_family:
            complete_by_family[family] += 1
        rows.append(
            {
                "pair_id": unit.pair_id,
                "question_key": unit.question_key,
                "scene_ids": list(unit.scene_ids),
                "family": family,
                "side_margins": [float(value) for value in side],
                "cross_prefix_margins": [float(value) for value in cross],
                "complete": complete,
                "cross_prefix_complete": bool(cross.gt(0).all()),
            }
        )
        margins.append(side)
        cross_margins.append(cross)
    stacked = torch.stack(margins)
    stacked_cross = torch.stack(cross_margins)
    return {
        "schema_version": 1,
        "unit_count": len(rows),
        "side_count": int(stacked.numel()),
        "mean_margin": float(stacked.mean()),
        "minimum_margin": float(stacked.min()),
        "complete_units": sum(int(row["complete"]) for row in rows),
        "positive_sides": int(stacked.gt(0).sum()),
        "mean_cross_prefix_margin": float(stacked_cross.mean()),
        "cross_prefix_complete_units": sum(
            int(row["cross_prefix_complete"]) for row in rows
        ),
        "complete_units_by_family": complete_by_family,
        "units": rows,
        "training_scenes_only": True,
        "validation_qa_loaded": False,
        "true_cross_prefix_differing_token_scores": True,
    }


def residual_rms_diagnostics(
    *,
    caches: Mapping[str, V35SceneCache],
    block_cross_residual: torch.nn.Module,
    device: torch.device,
) -> dict[str, Any]:
    values: dict[str, float] = {}
    with torch.inference_mode():
        for scene_id in sorted(caches):
            cache = caches[scene_id]
            source = cache.source_scene_tokens.to(device)
            current = current_scene_tokens(cache, block_cross_residual, device=device)
            values[scene_id] = float((current - source).float().square().mean().sqrt().cpu())
    tensor = torch.tensor(list(values.values()), dtype=torch.float32)
    return {
        "scene_count": len(values),
        "aggregate_rms": float(tensor.square().mean().sqrt()),
        "mean_scene_rms": float(tensor.mean()),
        "maximum_scene_rms": float(tensor.max()),
        "rms_by_scene": values,
    }


def _named_core_parameters(
    block_cross_residual: torch.nn.Module,
) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    named = dict(block_cross_residual.named_parameters())
    if tuple(named) != _PARAMETER_NAMES:
        raise RuntimeError(
            f"V35 core parameter names/order changed: observed={tuple(named)}"
        )
    return [named[name] for name in _QKV_NAMES], [named[name] for name in _OUTPUT_NAMES]


def freeze_for_v35(
    bundle: V30Bundle, block_cross_residual: torch.nn.Module, *, optimizer_step: int
) -> list[torch.nn.Parameter]:
    bundle.language.model.requires_grad_(False)
    for module in bundle.checkpoint_modules.values():
        module.requires_grad_(False)
        module.eval()
    block_cross_residual.requires_grad_(False)
    qkv, output = _named_core_parameters(block_cross_residual)
    for parameter in output:
        parameter.requires_grad_(True)
    if optimizer_step >= 1:
        for parameter in qkv:
            parameter.requires_grad_(True)
    block_cross_residual.train()
    return [*qkv, *output]


def assert_v35_trainable_surface(
    bundle: V30Bundle,
    block_cross_residual: torch.nn.Module,
    *,
    optimizer_step: int,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, Any]:
    qkv, output = _named_core_parameters(block_cross_residual)
    all_core = [*qkv, *output]
    named = dict(block_cross_residual.named_parameters())
    expected_active = {id(named["w_o"])} if optimizer_step == 0 else {
        id(parameter) for parameter in all_core
    }
    observed_active = {
        id(parameter)
        for module in bundle.checkpoint_modules.values()
        for parameter in module.parameters()
        if parameter.requires_grad
    }
    if observed_active != expected_active:
        raise RuntimeError("V35 active trainable surface differs from its staged lock")
    if any(parameter.requires_grad for parameter in bundle.language.model.parameters()):
        raise RuntimeError("V35 Gemma decoder must remain frozen")
    if optimizer is not None:
        optimizer_ids = {
            id(parameter) for group in optimizer.param_groups for parameter in group["params"]
        }
        if optimizer_ids != {id(parameter) for parameter in all_core}:
            raise RuntimeError("V35 optimizer contains an unauthorized tensor")
    counts = {
        "qkv": sum(parameter.numel() for parameter in qkv),
        "output": sum(parameter.numel() for parameter in output),
    }
    if counts != {"qkv": 589_824, "output": 393_216}:
        raise RuntimeError(f"V35 core parameter counts changed: {counts}")
    return {
        "parameter_names": list(_PARAMETER_NAMES),
        "group_parameter_counts": counts,
        "total_parameter_count": sum(counts.values()),
        "active_parameter_names": [
            name for name, parameter in named.items() if parameter.requires_grad
        ],
        "step_1_output_only": optimizer_step == 0,
        "gemma_decoder_frozen": True,
        "all_lora_banks_frozen": True,
        "complete_v33_stack_frozen": True,
        "every_other_parameter_frozen": True,
    }


def frozen_v35_source_state_sha256(
    bundle: V30Bundle, block_cross_residual: torch.nn.Module
) -> str:
    inherited = {
        name: module
        for name, module in bundle.checkpoint_modules.items()
        if module is not block_cross_residual and name != "block_cross_residual"
    }
    return module_collection_state_sha256(inherited)


def _optimizer(
    bundle: V30Bundle,
    block_cross_residual: torch.nn.Module,
    settings: V35Settings,
) -> torch.optim.AdamW:
    qkv, output = _named_core_parameters(block_cross_residual)
    optimizer = torch.optim.AdamW(
        [
            {
                "name": "block_cross_residual.qkv",
                "params": qkv,
                "lr": 0.0,
                "weight_decay": settings.weight_decay,
            },
            {
                "name": "block_cross_residual.output",
                "params": output,
                "lr": settings.output_learning_rate,
                "weight_decay": settings.weight_decay,
            },
        ]
    )
    assert_v35_trainable_surface(
        bundle,
        block_cross_residual,
        optimizer_step=0,
        optimizer=optimizer,
    )
    return optimizer


def set_v35_optimizer_stage(
    *,
    bundle: V30Bundle,
    block_cross_residual: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    optimizer_step_to_run: int,
    settings: V35Settings,
) -> None:
    if optimizer_step_to_run < 1:
        raise ValueError("V35 optimizer step to run must be positive")
    # Step 1 is output-only. Q/K/V become active immediately after it.
    completed_steps = optimizer_step_to_run - 1
    freeze_for_v35(bundle, block_cross_residual, optimizer_step=completed_steps)
    groups = {str(group.get("name")): group for group in optimizer.param_groups}
    groups["block_cross_residual.qkv"]["lr"] = (
        0.0 if optimizer_step_to_run == 1 else settings.qkv_learning_rate
    )
    groups["block_cross_residual.output"]["lr"] = settings.output_learning_rate
    assert_v35_trainable_surface(
        bundle,
        block_cross_residual,
        optimizer_step=completed_steps,
        optimizer=optimizer,
    )


def _zero_equivalence() -> dict[str, Any]:
    return {
        "verified": True,
        "base": "exact_v33_update64_post_sidecar_scene_tokens",
        "application_order": "after_v33_dense_sidecar_before_prefix_composer",
        "all_scene_slots_accounted": True,
        "all_occupied_block_tokens_accounted": True,
        "normalized_block_positions_used": True,
        "all_voxels_covered": True,
        "question_dependent_scene_processing": False,
    }


def construct_v35_core(config: Mapping[str, Any], *, device: torch.device) -> BlockCrossResidual:
    module = construct_block_cross_residual(
        config,
        scene_dim=1536,
        block_dim=384,
        latent_count=256,
    )
    if module is None:
        raise RuntimeError("V35 block-cross residual is disabled")
    module = module.to(device)
    expected = v35_contract(config).core_initial_state_sha256
    validate_block_cross_residual_state(
        module,
        expected_parameter_count=983_040,
        expected_state_sha256=expected,
        context="V35 deterministic construction",
    )
    return module


def preflight_v35(
    config: Mapping[str, Any], *, require_train_qa: bool = True
) -> dict[str, Any]:
    contract = v35_contract(config)
    settings = v35_settings(config)
    terminal = require_v34_terminal_gate(config)
    source, _ = require_exact_v33_source(config)
    assert_deferred_final_scenes_absent(config)
    qa_audit = None
    if require_train_qa:
        _, qa_audit = load_v35_train_qa_records(config)
    module = construct_v35_core(config, device=torch.device("cpu"))
    with torch.inference_mode():
        base = torch.randn(1, 256, 1536)
        blocks = torch.randn(13, 384)
        positions = torch.rand(13, 3).mul(2).sub(1)
        output = module(base, blocks, positions)
    if not torch.equal(output, base):
        raise RuntimeError("V35 exact-zero core is not a bit-identical identity")
    return {
        "schema_version": 1,
        "artifact": "v35_block_cross_preflight",
        "passed": True,
        "source_checkpoint": str(source),
        "source_optimizer_step": 64,
        "source_v33_tensor_state_sha256": contract.source_tensor_state_sha256,
        "terminal_report_sha256": terminal["sha256"],
        "core_initial_state_sha256": module.state_sha256(),
        "exact_trainable_parameter_count": module.parameter_count,
        "saved_optimizer_steps": list(settings.saved_optimizer_steps),
        "train_qa_loaded": require_train_qa,
        "validation_qa_loaded": False,
        "gemma_loaded": False,
        "scene_maps_loaded": False,
        "oracle_loaded": False,
        "final_test_scenes_touched": False,
        "qa_audit": qa_audit,
    }


def _deterministic_cache_audit(audit: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: copy.deepcopy(value)
        for key, value in audit.items()
        if key != "cache_build_seconds"
    }


def _metadata(
    *,
    source_metadata: Mapping[str, Any],
    config: Mapping[str, Any],
    terminal: Mapping[str, Any],
    schedule_audit: Mapping[str, Any],
    cache_audit: Mapping[str, Any],
    qa_audit: Mapping[str, Any],
    separation_reference: V35SeparationReference,
    update_zero: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
    optimizer_step: int,
    block_cross_residual: BlockCrossResidual,
    frozen_source_hash: str,
    surface: Mapping[str, Any],
    update32_gate: Mapping[str, Any] | None,
    update64_gate: Mapping[str, Any] | None,
) -> dict[str, Any]:
    contract = v35_contract(config)
    metadata = copy.deepcopy(dict(source_metadata))
    metadata.update(
        {
            "config_hash": config_hash(dict(config)),
            "epoch": optimizer_step,
            "optimizer_step": optimizer_step,
            "best_epoch": 0,
            "best_monitor_loss": None,
            "monitor_name": "independent_v35_selector_required",
            "history": list(history),
            "block_cross_residual": block_cross_residual_settings(config).contract(),
            "block_cross_residual_parameter_count": block_cross_residual.parameter_count,
            "block_cross_residual_initial_state_sha256": contract.core_initial_state_sha256,
            "block_cross_residual_state_sha256": block_cross_residual.state_sha256(),
            "block_cross_residual_zero_output_equivalence": _zero_equivalence(),
            "frozen_block_cross_source_stack_state_sha256": frozen_source_hash,
            "question_dependent_scene_processing": False,
        }
    )
    metadata["v35_block_cross"] = {
        "schema_version": 1,
        "artifact": "v35_diverse28_block_cross_training",
        "optimizer_step": optimizer_step,
        "conditional_v34_terminal_gate": {
            "path": terminal["path"],
            "sha256": terminal["sha256"],
        },
        "source_checkpoint": str(contract.source_checkpoint),
        "source_file_sha256": dict(contract.source_file_sha256),
        "source_optimizer_step": 64,
        "source_v33_tensor_state_sha256": contract.source_tensor_state_sha256,
        "schedule": dict(schedule_audit),
        "scene_cache": _deterministic_cache_audit(cache_audit),
        "train_qa_dataset": dict(qa_audit),
        "validation_qa_loaded": False,
        "exact_trainable_parameter_count": 983_040,
        "trainable_surface": dict(surface),
        "frozen_block_cross_source_stack_state_sha256": frozen_source_hash,
        "gemma_decoder_frozen": True,
        "all_lora_banks_frozen": True,
        "complete_v33_stack_frozen": True,
        "update_zero_equivalence": dict(update_zero),
        "separation_reference_sha256": separation_reference.audit_sha256,
        "separation_unique_changed_pair_count": 8,
        "separation_unrelated_pair_count": 112,
        "update32_train_only_gate": None if update32_gate is None else dict(update32_gate),
        "update64_train_only_gate": None if update64_gate is None else dict(update64_gate),
        "deferred_final_scene_ids_loaded": [],
        "oracle_environment_files_loaded": False,
        "question_dependent_scene_processing": False,
        "question_dependent_retrieval": False,
        "development_progress_is_not_chat_promotion": True,
        "independent_selector_required": True,
    }
    return metadata


def _save(
    path: Path,
    *,
    bundle: V30Bundle,
    metadata: dict[str, Any],
    optimizer: torch.optim.Optimizer | None,
) -> None:
    save_adapter_checkpoint(path, bundle.checkpoint_modules, metadata)
    if optimizer is not None:
        save_optimizer_checkpoint(path, optimizer)


def latest_v35_resume_checkpoint(output: Path, contract: V35Contract) -> Path | None:
    if not output.exists():
        return None
    if output.is_symlink() or not output.is_dir():
        raise ValueError(f"V35 output root must be a real directory: {output}")
    parsed: dict[int, Path] = {}
    for path in output.glob("update_*"):
        if path.is_symlink() or not path.is_dir():
            raise ValueError(f"V35 update path must be a real directory: {path}")
        match = _UPDATE_DIRECTORY.fullmatch(path.name)
        if match is None or int(match.group(1)) not in contract.saved_optimizer_steps:
            raise ValueError(f"V35 output contains an unauthorized arm: {path.name}")
        parsed[int(match.group(1))] = path
    complete = [
        step
        for step in contract.saved_optimizer_steps
        if step in parsed
        and all(
            (parsed[step] / name).is_file()
            for name in (
                "adapter.safetensors",
                TRAINING_METADATA_FILENAME,
                RUNTIME_METADATA_FILENAME,
                *(("optimizer.pt",) if step else ()),
            )
        )
    ]
    if complete != list(contract.saved_optimizer_steps[: len(complete)]):
        raise ValueError("V35 complete arms are not a contiguous saved-step prefix")
    return None if not complete else parsed[complete[-1]]


def _optimizer_step_audit(path: Path, *, expected_step: int) -> None:
    state = torch.load(path / "optimizer.pt", map_location="cpu", weights_only=True)
    if not isinstance(state, Mapping):
        raise TypeError("V35 optimizer checkpoint must be a mapping")
    groups = state.get("param_groups")
    values = state.get("state")
    if not isinstance(groups, list) or not isinstance(values, Mapping) or len(groups) != 2:
        raise ValueError("V35 optimizer must retain exactly two AdamW groups")
    by_name = {str(group.get("name")): group for group in groups}
    if set(by_name) != {"block_cross_residual.qkv", "block_cross_residual.output"}:
        raise ValueError("V35 optimizer group names changed")
    if float(by_name["block_cross_residual.qkv"]["lr"]) != 1e-4:
        raise ValueError("V35 resumed QKV learning rate changed")
    if float(by_name["block_cross_residual.output"]["lr"]) != 2.5e-5:
        raise ValueError("V35 resumed output learning rate changed")
    if any(float(group["weight_decay"]) != 0.0 for group in groups):
        raise ValueError("V35 optimizer weight decay must remain zero")
    if len(values) != 4:
        raise ValueError("V35 saved Adam state must cover exactly four core tensors")
    for entry in values.values():
        if not isinstance(entry, Mapping):
            raise TypeError("V35 Adam parameter state must be a mapping")
        step = entry.get("step")
        if isinstance(step, torch.Tensor) and step.numel() == 1:
            step = step.item()
        # Q/K/V start on optimizer step 2 and therefore have one fewer Adam
        # update than W_o.  Both values are exact and accepted explicitly.
        if int(step) not in {expected_step, expected_step - 1}:
            raise ValueError("V35 Adam state does not prove the staged optimizer step")
        for name in ("exp_avg", "exp_avg_sq"):
            value = entry.get(name)
            if not isinstance(value, torch.Tensor) or not torch.isfinite(value).all():
                raise ValueError("V35 Adam moments are invalid")


def validate_v35_resume_checkpoint(
    *,
    config: Mapping[str, Any],
    output: Path,
    resume: Path,
    contract: V35Contract,
    terminal: Mapping[str, Any],
    schedule_audit: Mapping[str, Any],
    cache_audit: Mapping[str, Any],
    separation_reference: V35SeparationReference,
    frozen_source_hash: str,
) -> dict[str, Any]:
    if resume.parent != output or resume.is_symlink() or not resume.is_dir():
        raise ValueError("V35 resume must be a real numbered arm inside its output root")
    match = _UPDATE_DIRECTORY.fullmatch(resume.name)
    if match is None:
        raise ValueError("V35 resume path is not a numbered update arm")
    step = int(match.group(1))
    if step not in contract.saved_optimizer_steps:
        raise ValueError("V35 resume update is not an authorized saved arm")
    for filename in (
        "adapter.safetensors",
        TRAINING_METADATA_FILENAME,
        RUNTIME_METADATA_FILENAME,
        *(("optimizer.pt",) if step else ()),
    ):
        if not (resume / filename).is_file():
            raise FileNotFoundError(f"V35 resume checkpoint is incomplete: {filename}")
    metadata = json.loads((resume / TRAINING_METADATA_FILENAME).read_text(encoding="utf-8"))
    stage = _mapping(metadata.get("v35_block_cross"), "resume v35_block_cross")
    if metadata.get("config_hash") != config_hash(dict(config)):
        raise ValueError("V35 resume config hash changed")
    if metadata.get("optimizer_step") != step or stage.get("optimizer_step") != step:
        raise ValueError("V35 resume optimizer step mismatch")
    if stage.get("source_v33_tensor_state_sha256") != contract.source_tensor_state_sha256:
        raise ValueError("V35 resume source is not exact V33 update 64")
    if stage.get("conditional_v34_terminal_gate") != {
        "path": terminal["path"],
        "sha256": terminal["sha256"],
    }:
        raise ValueError("V35 resume terminal authorization changed")
    saved_schedule = _mapping(stage.get("schedule"), "resume schedule")
    if saved_schedule.get("schedule_sha256") != schedule_audit["schedule_sha256"]:
        raise ValueError("V35 resume schedule changed")
    if stage.get("scene_cache") != _deterministic_cache_audit(cache_audit):
        raise ValueError("V35 resume question-free scene cache changed")
    qa = _mapping(stage.get("train_qa_dataset"), "resume train QA audit")
    if qa.get("validation_qa_loaded") is not False:
        raise ValueError("V35 resume metadata says validation QA was loaded")
    if stage.get("separation_reference_sha256") != separation_reference.audit_sha256:
        raise ValueError("V35 resume separation reference changed")
    if stage.get("frozen_block_cross_source_stack_state_sha256") != frozen_source_hash:
        raise ValueError("V35 resume frozen V33 stack hash changed")
    history = metadata.get("history")
    if not isinstance(history, list) or len(history) != step + 1:
        raise ValueError("V35 resume history is incomplete")
    if [row.get("optimizer_update") for row in history] != list(range(step + 1)):
        raise ValueError("V35 resume history does not prove every true microstep")
    update32 = stage.get("update32_train_only_gate")
    update64 = stage.get("update64_train_only_gate")
    if step >= 32 and not isinstance(update32, Mapping):
        raise ValueError("V35 resume at/after 32 lacks the update-32 train-only gate")
    if step >= 32 and update32.get("passed") is not True:
        raise ValueError("V35 cannot resume past a failed update-32 train-only gate")
    if step >= 64 and not isinstance(update64, Mapping):
        raise ValueError("V35 resume at/after 64 lacks the update-64 train-only gate")
    if step >= 64 and update64.get("passed") is not True:
        raise ValueError("V35 cannot resume past a failed update-64 train-only gate")
    runtime = json.loads((resume / RUNTIME_METADATA_FILENAME).read_text(encoding="utf-8"))
    validate_runtime_checkpoint_metadata(runtime)
    if runtime != runtime_checkpoint_metadata(metadata):
        raise ValueError("V35 resume runtime metadata is not exact sanitized training metadata")
    if step:
        _optimizer_step_audit(resume, expected_step=step)
    return metadata


def _assert_update0_identity(
    *,
    caches: Mapping[str, V35SceneCache],
    block_cross_residual: BlockCrossResidual,
    bundle: V30Bundle,
) -> dict[str, Any]:
    model_dtype = next(bundle.language.model.parameters()).dtype
    observed: dict[str, str] = {}
    repeated: dict[str, str] = {}
    with torch.inference_mode():
        for scene_id, cache in sorted(caches.items()):
            source = cache.source_scene_tokens.to(bundle.language.device)
            current = current_scene_tokens(
                cache, block_cross_residual, device=bundle.language.device
            )
            if not torch.equal(current, source):
                raise RuntimeError(f"V35 update-zero core changed source tokens: {scene_id}")
            first = bundle.composer.scene_prefix(current.to(model_dtype))
            second_tokens = current_scene_tokens(
                cache, block_cross_residual, device=bundle.language.device
            )
            second = bundle.composer.scene_prefix(second_tokens.to(model_dtype))
            observed[scene_id] = prefix_sha256(first)
            repeated[scene_id] = prefix_sha256(second)
            if observed[scene_id] != cache.source_prefix_sha256:
                raise RuntimeError(f"V35 update-zero prefix differs from V33: {scene_id}")
    if repeated != observed:
        raise RuntimeError("V35 update-zero prefix replay is nondeterministic")
    return {
        "exact_v33_update64_source_tensors": True,
        "source_tensor_state_sha256": _SOURCE_TENSOR_STATE_SHA256,
        "exact_v33_update64_post_sidecar_scene_tokens": True,
        "exact_zero_residual_identity": True,
        "exact_v33_update64_source_prefixes_all_22_scenes": True,
        "source_prefix_sha256_by_scene": observed,
        "source_prefix_scene_count": len(observed),
        "source_prefixes_replayed_bit_exact": True,
        "fresh_adam_state": True,
        "question_dependent_scene_processing": False,
        "question_dependent_retrieval": False,
        "validation_qa_loaded": False,
        "oracle_environment_files_loaded": False,
    }


def _gradient_norm(parameters: Sequence[torch.nn.Parameter]) -> float:
    if not parameters:
        return 0.0
    squared = sum(
        (
            torch.zeros((), device=parameter.device)
            if parameter.grad is None
            else parameter.grad.detach().float().square().sum()
        )
        for parameter in parameters
    )
    return float(squared.sqrt().cpu())


def run_v35(
    *, config: dict[str, Any], output: Path, resume: Path | None = None
) -> dict[str, Any]:
    if resume is None and output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty V35 output: {output}")
    contract = v35_contract(config)
    settings = v35_settings(config)
    terminal = require_v34_terminal_gate(config)
    source_checkpoint, pinned_source_metadata = require_exact_v33_source(config)
    assert_deferred_final_scenes_absent(config)
    seed = int(config["seed"])
    torch.manual_seed(seed)
    random.seed(seed)
    train_records, qa_audit = load_v35_train_qa_records(config)
    train_pairs = build_exact_question_pair_units(train_records)
    schedule, schedule_audit = build_v35_schedule(
        train_records, train_pairs, settings=settings, seed=seed
    )

    approved_v29 = require_approved_v29_source(config)
    bundle = load_v30_bundle(config, approved_v29)
    source_metadata = load_adapter_checkpoint(
        source_checkpoint,
        bundle.checkpoint_modules,
        device="cpu",
        metadata_filename=TRAINING_METADATA_FILENAME,
    )
    if source_metadata != pinned_source_metadata:
        raise RuntimeError("V35 source metadata changed during exact adapter load")
    inherited_hash = module_collection_state_sha256(bundle.checkpoint_modules)
    if inherited_hash != contract.source_tensor_state_sha256:
        raise RuntimeError(
            "V35 loaded checkpoint modules are not bit-exact V33 update 64: "
            f"expected={contract.source_tensor_state_sha256} observed={inherited_hash}"
        )
    block_cross_residual = construct_v35_core(
        config, device=bundle.language.device
    )
    bundle.checkpoint_modules["block_cross_residual"] = block_cross_residual
    all_scene_ids = (*contract.v31.train_scene_ids, *contract.v31.validation_scene_ids)
    caches, cache_audit = cache_v35_scenes(
        config=config,
        bundle=bundle,
        source_metadata=pinned_source_metadata,
        terminal=terminal,
        scene_ids=all_scene_ids,
    )
    # Validation caches exist only for the update-zero question-free identity
    # proof and future independent selector reuse. No optimization, residual
    # penalty, separation diagnostic, or continuation gate may consume them.
    train_caches = {
        scene_id: caches[scene_id] for scene_id in contract.v31.train_scene_ids
    }
    freeze_for_v35(bundle, block_cross_residual, optimizer_step=0)
    surface0 = assert_v35_trainable_surface(
        bundle, block_cross_residual, optimizer_step=0
    )
    frozen_source_hash = frozen_v35_source_state_sha256(
        bundle, block_cross_residual
    )
    if frozen_source_hash != contract.source_tensor_state_sha256:
        raise RuntimeError("V35 frozen inherited stack differs from exact V33 update 64")
    update_zero = _assert_update0_identity(
        caches=caches,
        block_cross_residual=block_cross_residual,
        bundle=bundle,
    )
    separation_reference = build_v35_separation_reference(
        units=train_pairs,
        caches=train_caches,
        block_cross_residual=block_cross_residual,
        bundle=bundle,
    )
    baseline_separation = v35_separation_diagnostics(
        reference=separation_reference,
        caches=train_caches,
        block_cross_residual=block_cross_residual,
        bundle=bundle,
    )
    baseline_pairs = training_pair_metrics(
        units=train_pairs,
        caches=train_caches,
        block_cross_residual=block_cross_residual,
        bundle=bundle,
        settings=settings,
    )
    baseline_residual = residual_rms_diagnostics(
        caches=train_caches,
        block_cross_residual=block_cross_residual,
        device=bundle.language.device,
    )
    if baseline_residual["aggregate_rms"] != 0.0:
        raise RuntimeError("V35 update-zero residual RMS is not exact zero")
    history: list[dict[str, Any]] = [
        {
            "optimizer_update": 0,
            "train_pair_metrics": baseline_pairs,
            "training_prefix_separation": baseline_separation,
            "training_residual_rms": baseline_residual,
            "validation_qa_loaded": False,
            "update_0_equivalence_verified": True,
            "saved_checkpoint": True,
        }
    ]
    optimizer = _optimizer(bundle, block_cross_residual, settings)
    if optimizer.state:
        raise RuntimeError("V35 optimizer is not fresh at update zero")
    start_step = 0
    accepted_update32: Mapping[str, Any] | None = None
    accepted_update64: Mapping[str, Any] | None = None
    output.mkdir(parents=True, exist_ok=True)
    if resume is not None:
        resume_metadata = validate_v35_resume_checkpoint(
            config=config,
            output=output,
            resume=resume,
            contract=contract,
            terminal=terminal,
            schedule_audit=schedule_audit,
            cache_audit=cache_audit,
            separation_reference=separation_reference,
            frozen_source_hash=frozen_source_hash,
        )
        loaded = load_adapter_checkpoint(
            resume,
            bundle.checkpoint_modules,
            device="cpu",
            metadata_filename=TRAINING_METADATA_FILENAME,
        )
        if loaded != resume_metadata:
            raise RuntimeError("V35 resume metadata changed during adapter load")
        start_step = int(resume_metadata["optimizer_step"])
        if start_step:
            load_optimizer_checkpoint(resume, optimizer, bundle.language.device)
        history = list(resume_metadata["history"])
        resume_stage = _mapping(
            resume_metadata.get("v35_block_cross"), "resume v35 stage"
        )
        accepted_update32 = resume_stage.get("update32_train_only_gate")
        accepted_update64 = resume_stage.get("update64_train_only_gate")
        if frozen_v35_source_state_sha256(bundle, block_cross_residual) != frozen_source_hash:
            raise RuntimeError("V35 resume changed the frozen V33 stack")
        validate_block_cross_residual_state(
            block_cross_residual,
            expected_parameter_count=983_040,
            expected_state_sha256=str(resume_metadata["block_cross_residual_state_sha256"]),
            context="V35 resumed core",
        )
    else:
        metadata0 = _metadata(
            source_metadata=pinned_source_metadata,
            config=config,
            terminal=terminal,
            schedule_audit=schedule_audit,
            cache_audit=cache_audit,
            qa_audit=qa_audit,
            separation_reference=separation_reference,
            update_zero=update_zero,
            history=history,
            optimizer_step=0,
            block_cross_residual=block_cross_residual,
            frozen_source_hash=frozen_source_hash,
            surface=surface0,
            update32_gate=None,
            update64_gate=None,
        )
        _save(
            output / "update_000",
            bundle=bundle,
            metadata=metadata0,
            optimizer=None,
        )
        saved0 = load_file(output / "update_000" / "adapter.safetensors", device="cpu")
        saved_inherited = {
            key: value
            for key, value in saved0.items()
            if not key.startswith("block_cross_residual.")
        }
        if tensor_state_sha256(saved_inherited) != contract.source_tensor_state_sha256:
            raise RuntimeError("V35 saved update zero changed an inherited V33 tensor")
        saved_core = {
            key.removeprefix("block_cross_residual."): value
            for key, value in saved0.items()
            if key.startswith("block_cross_residual.")
        }
        if tensor_state_sha256(saved_core) != contract.core_initial_state_sha256:
            raise RuntimeError("V35 saved update zero changed the deterministic core")

    qkv_parameters, output_parameters = _named_core_parameters(block_cross_residual)
    for item in schedule[start_step:]:
        step = item.optimizer_step
        set_v35_optimizer_stage(
            bundle=bundle,
            block_cross_residual=block_cross_residual,
            optimizer=optimizer,
            optimizer_step_to_run=step,
            settings=settings,
        )
        optimizer.zero_grad(set_to_none=True)

        broad_tokens = current_scene_tokens(
            train_caches[item.broad_record.scene_id],
            block_cross_residual,
            device=bundle.language.device,
        )
        broad = broad_answer_nll(
            scene_tokens=broad_tokens,
            record=item.broad_record,
            bundle=bundle,
        )
        (settings.broad_nll_weight * broad).backward()
        broad_value = float(broad.detach().cpu())
        del broad, broad_tokens

        pair_tokens = {
            scene_id: current_scene_tokens(
                train_caches[scene_id],
                block_cross_residual,
                device=bundle.language.device,
            )
            for scene_id in item.pair_unit.scene_ids
        }
        pair_nll, side_hinge, cross_hinge, pair_diagnostics = (
            paired_cross_prefix_objective(
                unit=item.pair_unit,
                scene_tokens=pair_tokens,
                bundle=bundle,
                side_margin=settings.side_hinge_margin,
                cross_prefix_margin=settings.cross_prefix_flip_margin,
            )
        )
        pair_objective = (
            settings.pair_correct_nll_weight * pair_nll
            + settings.side_hinge_weight * side_hinge
            + settings.cross_prefix_flip_weight * cross_hinge
        )
        pair_objective.backward()
        pair_nll_value = float(pair_nll.detach().cpu())
        side_hinge_value = float(side_hinge.detach().cpu())
        cross_hinge_value = float(cross_hinge.detach().cpu())
        side_margin_mean = float(
            pair_diagnostics["side_margins"].detach().float().mean().cpu()
        )
        cross_margin_mean = float(
            pair_diagnostics["cross_prefix_margins"].detach().float().mean().cpu()
        )
        del pair_nll, side_hinge, cross_hinge, pair_diagnostics, pair_objective, pair_tokens

        normalized_residual, residual_rms = residual_penalty(
            caches=train_caches,
            block_cross_residual=block_cross_residual,
            device=bundle.language.device,
            scale=settings.residual_penalty_scale,
        )
        (settings.residual_penalty_weight * normalized_residual).backward()
        normalized_residual_value = float(normalized_residual.detach().cpu())
        residual_rms_value = float(residual_rms.detach().cpu())
        del normalized_residual, residual_rms

        active_parameters = [
            parameter for parameter in (*qkv_parameters, *output_parameters) if parameter.requires_grad
        ]
        if any(parameter.grad is None for parameter in active_parameters):
            raise RuntimeError("V35 one or more active core tensors lacks a gradient")
        if any(not torch.isfinite(parameter.grad).all() for parameter in active_parameters):
            raise RuntimeError("V35 core gradient is nonfinite")
        qkv_preclip = _gradient_norm(qkv_parameters) if step > 1 else 0.0
        output_preclip = _gradient_norm(output_parameters)
        if step > 1:
            torch.nn.utils.clip_grad_norm_(
                qkv_parameters, settings.qkv_gradient_clip_norm
            )
        torch.nn.utils.clip_grad_norm_(
            output_parameters, settings.output_gradient_clip_norm
        )
        optimizer.step()
        if frozen_v35_source_state_sha256(bundle, block_cross_residual) != frozen_source_hash:
            raise RuntimeError("V35 changed the frozen V33 source stack")
        validate_block_cross_residual_state(
            block_cross_residual,
            expected_parameter_count=983_040,
            context=f"V35 update {step}",
        )

        should_save = step in contract.saved_optimizer_steps
        separation = None
        residual_diagnostics = None
        pair_metrics = None
        if should_save:
            separation = v35_separation_diagnostics(
                reference=separation_reference,
                caches=train_caches,
                block_cross_residual=block_cross_residual,
                bundle=bundle,
            )
            residual_diagnostics = residual_rms_diagnostics(
                caches=train_caches,
                block_cross_residual=block_cross_residual,
                device=bundle.language.device,
            )
        if step in {32, 64, 100}:
            pair_metrics = training_pair_metrics(
                units=train_pairs,
                caches=train_caches,
                block_cross_residual=block_cross_residual,
                bundle=bundle,
                settings=settings,
            )
        if step == 32:
            if separation is None or residual_diagnostics is None or pair_metrics is None:
                raise RuntimeError("V35 update-32 gate lacks train-only diagnostics")
            accepted_update32 = v35_update32_gate(
                separation=separation,
                pair_metrics=pair_metrics,
                baseline_pair_metrics=baseline_pairs,
                residual_rms=float(residual_diagnostics["aggregate_rms"]),
                contract=contract,
            )
        if step == 64:
            if (
                not isinstance(accepted_update32, Mapping)
                or separation is None
                or residual_diagnostics is None
                or pair_metrics is None
            ):
                raise RuntimeError("V35 update-64 gate lacks prior train-only evidence")
            accepted_update64 = v35_update64_gate(
                update32_gate=accepted_update32,
                pair_metrics=pair_metrics,
                baseline_pair_metrics=baseline_pairs,
                residual_rms=float(residual_diagnostics["aggregate_rms"]),
                contract=contract,
            )

        objective_value = (
            settings.broad_nll_weight * broad_value
            + settings.pair_correct_nll_weight * pair_nll_value
            + settings.side_hinge_weight * side_hinge_value
            + settings.cross_prefix_flip_weight * cross_hinge_value
            + settings.residual_penalty_weight * normalized_residual_value
        )
        history.append(
            {
                "optimizer_update": step,
                "true_optimizer_step": True,
                "train_broad_answer_token_nll": broad_value,
                "train_pair_correct_answer_token_nll": pair_nll_value,
                "train_side_hinge": side_hinge_value,
                "train_cross_prefix_flip_hinge": cross_hinge_value,
                "train_side_margin_mean": side_margin_mean,
                "train_cross_prefix_margin_mean": cross_margin_mean,
                "train_normalized_residual_penalty": normalized_residual_value,
                "train_residual_rms": residual_rms_value,
                "train_objective": objective_value,
                "preclip_gradient_norm_by_group": {
                    "qkv": qkv_preclip,
                    "output": output_preclip,
                },
                "qkv_active": step > 1,
                "output_active": True,
                "separate_group_clipping": True,
                "training_prefix_separation": separation,
                "training_pair_metrics": pair_metrics,
                "training_residual_diagnostics": residual_diagnostics,
                "update32_train_only_gate": accepted_update32,
                "update64_train_only_gate": accepted_update64,
                "validation_qa_loaded": False,
                "saved_checkpoint": should_save,
            }
        )
        if not should_save:
            continue
        completed_surface = assert_v35_trainable_surface(
            bundle,
            block_cross_residual,
            optimizer_step=step,
            optimizer=optimizer,
        )
        metadata = _metadata(
            source_metadata=pinned_source_metadata,
            config=config,
            terminal=terminal,
            schedule_audit=schedule_audit,
            cache_audit=cache_audit,
            qa_audit=qa_audit,
            separation_reference=separation_reference,
            update_zero=update_zero,
            history=history,
            optimizer_step=step,
            block_cross_residual=block_cross_residual,
            frozen_source_hash=frozen_source_hash,
            surface=completed_surface,
            update32_gate=accepted_update32,
            update64_gate=accepted_update64,
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
                    "phase": "v35_true_block_cross_checkpoint",
                    "optimizer_step": step,
                    "training_complete_changed_units": (
                        None if pair_metrics is None else pair_metrics["complete_units"]
                    ),
                    "training_mean_margin": (
                        None if pair_metrics is None else pair_metrics["mean_margin"]
                    ),
                    "training_changed_selectivity_ratio": separation[
                        "changed_selectivity_ratio_geometric_mean"
                    ],
                    "training_changed_selectivity_coverage": separation[
                        "changed_selectivity_over_1_02_count"
                    ],
                    "training_residual_rms": residual_diagnostics["aggregate_rms"],
                    "update32_gate_passed": (
                        None
                        if accepted_update32 is None
                        else accepted_update32.get("passed")
                    ),
                    "update64_gate_passed": (
                        None
                        if accepted_update64 is None
                        else accepted_update64.get("passed")
                    ),
                    "validation_qa_loaded": False,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if step == 32 and accepted_update32.get("passed") is not True:
            raise RuntimeError(
                "V35 update-32 train-only causal gate failed; stop bounded block-cross arm"
            )
        if step == 64 and accepted_update64.get("passed") is not True:
            raise RuntimeError(
                "V35 update-64 train-only causal gate failed; stop bounded block-cross arm"
            )

    return {
        "schema_version": 1,
        "artifact": "v35_diverse28_block_cross_training",
        "output": str(output),
        "optimizer_updates": 100,
        "resumed_from_optimizer_step": start_step,
        "saved_optimizer_steps": list(contract.saved_optimizer_steps),
        "exact_trainable_parameter_count": 983_040,
        "source_v33_tensor_state_sha256": frozen_source_hash,
        "v34_terminal_report_sha256": terminal["sha256"],
        "update32_train_only_gate": accepted_update32,
        "update64_train_only_gate": accepted_update64,
        "validation_qa_loaded": False,
        "final_test_scene_ids_loaded": [],
        "question_dependent_scene_processing": False,
        "question_dependent_retrieval": False,
        "oracle_environment_files_loaded": False,
        "independent_selector_required_before_promotion": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--resume-latest", action="store_true")
    args = parser.parse_args()
    if args.resume is not None and args.resume_latest:
        parser.error("--resume and --resume-latest are mutually exclusive")
    config = load_config(args.config)
    if args.preflight_only:
        report = preflight_v35(config)
    else:
        output = _resolve(args.output)
        resume = _resolve(args.resume) if args.resume is not None else None
        if args.resume_latest:
            resume = latest_v35_resume_checkpoint(output, v35_contract(config))
            if resume is None:
                raise FileNotFoundError("V35 has no complete checkpoint to resume")
        report = run_v35(config=config, output=output, resume=resume)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "V35Contract",
    "V35Microstep",
    "V35SceneCache",
    "V35SeparationReference",
    "V35Settings",
    "assert_v35_trainable_surface",
    "build_v35_schedule",
    "cache_v35_scenes",
    "construct_v35_core",
    "freeze_for_v35",
    "latest_v35_resume_checkpoint",
    "load_v35_train_qa_records",
    "paired_cross_prefix_objective",
    "pinned_post_v33_prefix_manifest",
    "preflight_v35",
    "residual_penalty",
    "run_v35",
    "set_v35_optimizer_stage",
    "v35_contract",
    "v35_separation_diagnostics",
    "v35_settings",
    "v35_update32_gate",
    "v35_update64_gate",
    "v35_weighted_objective",
    "validate_v35_cache_audit",
    "validate_v35_resume_checkpoint",
    "validate_v35_scene_cache",
]
