"""Bounded V36 joint all-block scene bridge and Gemma query-LoRA training.

V36 is a causal decoder-readout experiment.  It loads the exact stopped V35
update-32 block core because that arm improved train-only answer margins and
failed only its Euclidean scene-distance surrogate.  It preserves V35's exact
V33 stack, verifies that ``extension_v30_joint_pair_query`` is still
exact-zero output, discards V35 optimizer momentum, and trains only that bank
plus the learned block core.  Validation QA is reserved for an independent
selector.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from semantic_3d_chat.config import PROJECT_ROOT, config_hash, load_config
from semantic_3d_chat.data.dataset import QARecord
from semantic_3d_chat.evaluation.metrics import exact_normalized_match
from semantic_3d_chat.evaluation.v30_joint_pair_selector import _question_logits_and_answer
from semantic_3d_chat.evaluation.v35_terminal_gate import audit_v35_update32
from semantic_3d_chat.language.lora import (
    LoRALinear,
    lora_banks_checkpoint_contract,
    lora_banks_optimizer_settings,
    lora_banks_settings,
    tensor_state_sha256,
)
from semantic_3d_chat.language.prefix_injection import prefix_sha256, stack_prefix_batches
from semantic_3d_chat.scene_encoder.block_cross_residual import (
    BlockCrossResidual,
    block_cross_residual_settings,
    validate_block_cross_residual_state,
)
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
    token_normalized_nll,
)
from semantic_3d_chat.training.train_adapter import forward_prefix_batch
from semantic_3d_chat.training.train_block_cross_v35 import (
    V35SceneCache,
    V35SeparationReference,
    _compose_answer_batch,
    _deterministic_cache_audit,
    broad_answer_nll,
    build_v35_schedule,
    build_v35_separation_reference,
    cache_v35_scenes,
    construct_v35_core,
    current_scene_tokens,
    load_v35_train_qa_records,
    paired_cross_prefix_objective,
    require_v34_terminal_gate,
    residual_penalty,
    residual_rms_diagnostics,
    training_pair_metrics,
    v35_contract,
    v35_separation_diagnostics,
    v35_settings,
)
from semantic_3d_chat.training.train_environmental_sidecar_v33 import (
    assert_deferred_final_scenes_absent,
)
from semantic_3d_chat.training.train_joint_pair_v30 import (
    V30Bundle,
    load_v30_bundle,
    require_approved_v29_source,
)
from semantic_3d_chat.training.train_joint_pair_v31 import v31_contract

DEFAULT_CONFIG = Path("configs/experiments/gemma4_diverse28_joint_block_cross_v36.yaml")
DEFAULT_OUTPUT = Path("data_gemma4/checkpoints/gemma4_v36_diverse28_joint_block_cross")
_UPDATE_DIRECTORY = re.compile(r"update_([0-9]{3})")
_BANK_NAME = "extension_v30_joint_pair_query"
_BANK_INITIAL_STATE_SHA256 = "2b1d89fbb9189ac551bf12905cf94036ebaa84696449b31c2b37b69d478fb70d"
_SOURCE_CHECKPOINT = Path("data_gemma4/checkpoints/gemma4_v35_diverse28_block_cross/update_032")
_SOURCE_FILE_SHA256 = {
    "adapter.safetensors": "4ecd5d9a38f4610387f96d36fca6111d2e248d206fd029e471cce0b1114afda0",
    TRAINING_METADATA_FILENAME: "0f106ecf5dccbe49ae1a15977d45610b560042d7f21a7b0b7ea0bf4ebea6af77",
    RUNTIME_METADATA_FILENAME: "fc06bd605b101ef9a64bf5e38cc83e91cdb8a9a37f8825e6e89c0e6a2ebfd7f1",
    "optimizer.pt": "add72932ce8cd8b58260068472ba0b2486d7011c283b4ce6785ae0f99b12b497",
}
_SOURCE_TENSOR_STATE_SHA256 = "cb7bb3b48ace60212ee5c7f326839bf2ddd993810417de45c9a9cbc666313fe6"
_V35_SOURCE_TENSOR_STATE_SHA256 = "1fe8f278460faeb1e13d9da09051a497965a566565c79a4f6ea28c56a9120326"
_CORE_INITIAL_STATE_SHA256 = "72ae7f492f5953e58d809b6782d559dc64669637d5d6a79ae98f3a31296a12cd"
_CORE_SOURCE_STATE_SHA256 = "75af995833d9387e3eb01fb022eaade7327e44960466671123a51aa43afa4cf3"
_FROZEN_NONAUTHORIZED_STATE_SHA256 = (
    "b394d502f0c32a694c2d1a448cdf3849c47efc4058cb1f1331fe4a97d381b1dc"
)
_CORE_NAMES = ("w_q", "w_k", "w_v", "w_o")
_CORE_QKV_NAMES = ("w_q", "w_k", "w_v")
_CORE_OUTPUT_NAMES = ("w_o",)
_CORE_PREFIX = "block_cross_residual."
_CORE_PARAMETER_NAMES = tuple(f"{_CORE_PREFIX}{name}" for name in _CORE_NAMES)
_BANK_PREFIX = f"lora_banks.{_BANK_NAME}."
_BANK_PARAMETER_NAMES = tuple(
    f"{_BANK_PREFIX}adapters.{index}.{side}" for index in range(4) for side in ("lora_a", "lora_b")
)
_BANK_PARAMETER_NAME_SET = frozenset(_BANK_PARAMETER_NAMES)
_BANK_OPTIMIZER_PARAMETER_NAMES = tuple(
    f"{_BANK_PREFIX}adapters.{index}.{side}" for side in ("lora_a", "lora_b") for index in range(4)
)
_V35_TERMINAL_REPORT_SHA256 = "88205d018de14fc0518fe695bf7420c44ac832a1ee95eea0e2ae1f41deff4a27"


@dataclass(frozen=True)
class V36Settings:
    enabled: bool
    optimizer_steps: int
    checkpoint_interval_steps: int
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
    decoder_learning_rate: float
    weight_decay: float
    qkv_gradient_clip_norm: float
    output_gradient_clip_norm: float
    decoder_gradient_clip_norm: float

    @property
    def saved_optimizer_steps(self) -> tuple[int, ...]:
        regular = tuple(range(0, self.optimizer_steps, self.checkpoint_interval_steps))
        return (*regular, self.optimizer_steps)


@dataclass(frozen=True)
class V36Contract:
    source_checkpoint: Path
    source_file_sha256: Mapping[str, str]
    source_tensor_state_sha256: str
    inherited_v33_tensor_state_sha256: str
    core_source_state_sha256: str
    decoder_bank_initial_state_sha256: str
    frozen_nonauthorized_state_sha256: str
    v35_terminal_report: Path
    v35_terminal_report_sha256: str
    saved_optimizer_steps: tuple[int, ...]
    update16_gate: Mapping[str, Any]
    update32_gate: Mapping[str, Any]
    update64_gate: Mapping[str, Any]


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


def v36_settings(config: Mapping[str, Any]) -> V36Settings:
    training = _mapping(config.get("training"), "training")
    raw = _mapping(training.get("v36_joint_block_cross"), "training.v36_joint_block_cross")
    fields = set(V36Settings.__dataclass_fields__)
    if set(raw) != fields:
        raise ValueError(
            "training.v36_joint_block_cross fields differ from the locked schema: "
            f"missing={sorted(fields - set(raw))} unknown={sorted(set(raw) - fields)}"
        )
    if not isinstance(raw["enabled"], bool):
        raise TypeError("training.v36_joint_block_cross.enabled must be boolean")
    result = V36Settings(
        enabled=raw["enabled"],
        optimizer_steps=_positive_int("optimizer_steps", raw["optimizer_steps"]),
        checkpoint_interval_steps=_positive_int(
            "checkpoint_interval_steps", raw["checkpoint_interval_steps"]
        ),
        broad_nll_weight=_finite("broad_nll_weight", raw["broad_nll_weight"], positive=True),
        pair_correct_nll_weight=_finite(
            "pair_correct_nll_weight", raw["pair_correct_nll_weight"], positive=True
        ),
        side_hinge_weight=_finite("side_hinge_weight", raw["side_hinge_weight"], positive=True),
        side_hinge_margin=_finite("side_hinge_margin", raw["side_hinge_margin"], positive=True),
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
        qkv_learning_rate=_finite("qkv_learning_rate", raw["qkv_learning_rate"], positive=True),
        output_learning_rate=_finite(
            "output_learning_rate", raw["output_learning_rate"], positive=True
        ),
        decoder_learning_rate=_finite(
            "decoder_learning_rate", raw["decoder_learning_rate"], positive=True
        ),
        weight_decay=_finite("weight_decay", raw["weight_decay"], positive=False),
        qkv_gradient_clip_norm=_finite(
            "qkv_gradient_clip_norm", raw["qkv_gradient_clip_norm"], positive=True
        ),
        output_gradient_clip_norm=_finite(
            "output_gradient_clip_norm", raw["output_gradient_clip_norm"], positive=True
        ),
        decoder_gradient_clip_norm=_finite(
            "decoder_gradient_clip_norm", raw["decoder_gradient_clip_norm"], positive=True
        ),
    )
    expected: Mapping[str, object] = {
        "enabled": True,
        "optimizer_steps": 100,
        "checkpoint_interval_steps": 8,
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
        "decoder_learning_rate": 2e-5,
        "weight_decay": 0.0,
        "qkv_gradient_clip_norm": 0.5,
        "output_gradient_clip_norm": 1.0,
        "decoder_gradient_clip_norm": 1.0,
    }
    mismatches = {
        field: {"observed": getattr(result, field), "expected": value}
        for field, value in expected.items()
        if getattr(result, field) != value
    }
    if mismatches:
        raise ValueError(f"V36 locked optimizer/objective settings changed: {mismatches}")
    if result.saved_optimizer_steps != (*range(0, 97, 8), 100):
        raise RuntimeError("V36 saved-step calculation changed")
    return result


def v36_contract(config: Mapping[str, Any]) -> V36Contract:
    # Replaying V35's contract here pins the clean V33 source, block geometry,
    # scene splits, and exact question-free cache rather than copying them.
    v35_contract(config)
    settings = v36_settings(config)
    v31 = v31_contract(config)
    raw = _mapping(config.get("v36_joint_block_cross"), "v36_joint_block_cross")
    required = {
        "schema_version",
        "role",
        "engine",
        "v35_terminal_gate_report",
        "v35_terminal_gate_report_sha256",
        "source_checkpoint",
        "source_optimizer_step",
        "source_file_sha256",
        "source_v35_tensor_state_sha256",
        "inherited_v33_tensor_state_sha256",
        "source_block_core_state_sha256",
        "decoder_bank_name",
        "decoder_bank_parameter_count",
        "decoder_bank_initial_state_sha256",
        "block_core_parameter_count",
        "joint_trainable_parameter_count",
        "frozen_nonauthorized_state_sha256",
        "train_scene_ids",
        "validation_scene_ids",
        "deferred_final_scene_ids",
        "validation_qa_loaded_during_training",
        "continuation_gates_use_training_only",
        "source_v35_learned_core_loaded",
        "source_v35_optimizer_state_loaded",
        "question_dependent_scene_processing",
        "saved_optimizer_steps",
        "update16_gate",
        "update32_gate",
        "update64_gate",
        "selector_uses_validation_only_after_complete_training",
        "final_test_deferred",
    }
    if set(raw) != required:
        raise ValueError(
            "v36_joint_block_cross fields differ from the locked schema: "
            f"missing={sorted(required - set(raw))} unknown={sorted(set(raw) - required)}"
        )
    exact = {
        "schema_version": 1,
        "role": "exact_v35_u32_block_cross_plus_zero_query_lora_v36",
        "engine": "fresh_adam_joint_learned_block_cross_and_zero_query_lora_true_microsteps",
        "source_optimizer_step": 32,
        "source_v35_tensor_state_sha256": _V35_SOURCE_TENSOR_STATE_SHA256,
        "inherited_v33_tensor_state_sha256": _SOURCE_TENSOR_STATE_SHA256,
        "source_block_core_state_sha256": _CORE_SOURCE_STATE_SHA256,
        "decoder_bank_name": _BANK_NAME,
        "decoder_bank_parameter_count": 131_072,
        "decoder_bank_initial_state_sha256": _BANK_INITIAL_STATE_SHA256,
        "block_core_parameter_count": 983_040,
        "joint_trainable_parameter_count": 1_114_112,
        "frozen_nonauthorized_state_sha256": _FROZEN_NONAUTHORIZED_STATE_SHA256,
    }
    mismatches = {
        key: {"observed": raw.get(key), "expected": value}
        for key, value in exact.items()
        if raw.get(key) != value
    }
    if mismatches:
        raise ValueError(f"V36 locked contract changed: {mismatches}")
    if tuple(raw["train_scene_ids"]) != v31.train_scene_ids:
        raise ValueError("V36 training scenes differ from the diverse28 lock")
    if tuple(raw["validation_scene_ids"]) != v31.validation_scene_ids:
        raise ValueError("V36 validation scenes differ from the diverse28 lock")
    if tuple(raw["deferred_final_scene_ids"]) != v31.deferred_final_scene_ids:
        raise ValueError("V36 deferred final scenes differ from the diverse28 lock")
    true_fields = (
        "source_v35_learned_core_loaded",
        "continuation_gates_use_training_only",
        "selector_uses_validation_only_after_complete_training",
        "final_test_deferred",
    )
    false_fields = (
        "validation_qa_loaded_during_training",
        "source_v35_optimizer_state_loaded",
        "question_dependent_scene_processing",
    )
    if any(raw.get(field) is not True for field in true_fields):
        raise ValueError("V36 required true-valued safety field changed")
    if any(raw.get(field) is not False for field in false_fields):
        raise ValueError("V36 forbidden-input/source field changed")
    if tuple(raw["saved_optimizer_steps"]) != settings.saved_optimizer_steps:
        raise ValueError("V36 saved optimizer arms differ from its schedule")
    if _resolve(str(raw["source_checkpoint"])) != _resolve(_SOURCE_CHECKPOINT):
        raise ValueError("V36 source must remain exact stopped V35 update 32")
    source_hashes = dict(_mapping(raw["source_file_sha256"], "source_file_sha256"))
    if source_hashes != _SOURCE_FILE_SHA256:
        raise ValueError("V36 V35 update-32 source file pins changed")
    terminal_hash = str(raw["v35_terminal_gate_report_sha256"])
    if re.fullmatch(r"[0-9a-f]{64}", terminal_hash) is None:
        raise ValueError("V36 V35 terminal report SHA-256 is not pinned")
    if terminal_hash != _V35_TERMINAL_REPORT_SHA256:
        raise ValueError("V36 exact V35 terminal report pin changed")
    update16 = dict(_mapping(raw["update16_gate"], "update16_gate"))
    update32 = dict(_mapping(raw["update32_gate"], "update32_gate"))
    update64 = dict(_mapping(raw["update64_gate"], "update64_gate"))
    expected16 = {
        "optimizer_step": 16,
        "complete_units_minimum": 10,
        "cross_prefix_complete_units_minimum": 16,
        "positive_sides_minimum": 34,
        "mean_cross_prefix_margin_strict_minimum": 1.32265043258667,
        "complete_physical_pair_coverage_minimum": 5,
        "unchanged_broad_nll_ratio_maximum": 1.02,
        "residual_rms_maximum": 0.075,
        "decoder_bank_state_must_change": True,
        "frozen_nonauthorized_state_must_remain_exact": True,
    }
    expected32 = {
        "optimizer_step": 32,
        "require_update16_passed": True,
        "complete_units_minimum": 12,
        "cross_prefix_complete_units_minimum": 18,
        "positive_sides_minimum": 37,
        "mean_cross_prefix_margin_minimum": 1.37265043258667,
        "complete_physical_pair_coverage_minimum": 6,
        "require_one_complete_per_priority_family": True,
        "unchanged_broad_nll_ratio_maximum": 1.03,
        "residual_rms_maximum": 0.075,
    }
    expected64 = {
        "optimizer_step": 64,
        "require_update32_passed": True,
        "complete_units_minimum": 15,
        "cross_prefix_complete_units_minimum": 20,
        "positive_sides_minimum": 40,
        "complete_physical_pair_coverage_minimum": 7,
        "require_one_complete_per_priority_family": True,
        "greedy_complete_units_minimum": 6,
        "require_one_greedy_complete_per_priority_family": True,
        "broad_greedy_exact_accuracy_maximum_drop": 0.02,
        "residual_rms_maximum": 0.10,
    }
    if update16 != expected16 or update32 != expected32 or update64 != expected64:
        raise ValueError("V36 train-only continuation gates changed")
    return V36Contract(
        source_checkpoint=_resolve(_SOURCE_CHECKPOINT),
        source_file_sha256=source_hashes,
        source_tensor_state_sha256=_V35_SOURCE_TENSOR_STATE_SHA256,
        inherited_v33_tensor_state_sha256=_SOURCE_TENSOR_STATE_SHA256,
        core_source_state_sha256=_CORE_SOURCE_STATE_SHA256,
        decoder_bank_initial_state_sha256=_BANK_INITIAL_STATE_SHA256,
        frozen_nonauthorized_state_sha256=_FROZEN_NONAUTHORIZED_STATE_SHA256,
        v35_terminal_report=_resolve(str(raw["v35_terminal_gate_report"])),
        v35_terminal_report_sha256=terminal_hash,
        saved_optimizer_steps=settings.saved_optimizer_steps,
        update16_gate=update16,
        update32_gate=update32,
        update64_gate=update64,
    )


def require_v35_terminal_gate(config: Mapping[str, Any]) -> dict[str, Any]:
    """Replay and pin the tensor-only seal that authorizes exactly V36."""

    contract = v36_contract(config)
    path = contract.v35_terminal_report
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"V36 requires a real V35 terminal report: {path}")
    observed_sha = _sha256(path)
    if observed_sha != contract.v35_terminal_report_sha256:
        raise ValueError("V35 terminal report bytes differ from V36's immutable pin")
    report = json.loads(path.read_text(encoding="utf-8"))
    audited = audit_v35_update32()
    if report != audited:
        raise ValueError("V35 terminal report does not replay from pinned tensors")
    expected = {
        "artifact": "v35_update32_terminal_gate",
        "passed": True,
        "stopped_at_optimizer_step": 32,
        "no_update_040_or_later": True,
        "final_test_scenes_touched": False,
        "oracle_loaded": False,
        "qa_loaded": False,
        "conditional_v36_joint_upper_lora_authorized": True,
        "v35_chat_promotion_eligible": False,
    }
    mismatches = {
        key: {"observed": report.get(key), "expected": value}
        for key, value in expected.items()
        if report.get(key) != value
    }
    if mismatches:
        raise ValueError(f"V35 terminal report does not authorize V36: {mismatches}")
    authorization = _mapping(
        report.get("conditional_authorization"), "V35 conditional_authorization"
    )
    exact_authorization = {
        "authorized": True,
        "stage": "v36_joint_block_cross_upper_lora",
        "scope": (
            "exact_v35_update32_block_cross_plus_existing_exact_zero_"
            "extension_v30_joint_pair_query_joint_only"
        ),
        "source_checkpoint": str(_SOURCE_CHECKPOINT),
        "v35_block_cross_matrices_may_continue_training": True,
        "authorized_existing_lora_bank": _BANK_NAME,
        "authorized_existing_lora_state_sha256": _BANK_INITIAL_STATE_SHA256,
        "authorized_existing_lora_parameter_count": 131_072,
        "authorized_existing_lora_rank": 8,
        "authorized_existing_lora_alpha": 16.0,
        "authorized_existing_lora_dropout": 0.0,
        "authorized_existing_lora_target_language_layers": [18, 19, 20, 21],
        "authorized_existing_lora_target_module_suffixes": ["self_attn.q_proj"],
        "authorized_existing_lora_output_matrices_are_exact_zero": True,
        "fresh_adam_state_required": True,
        "optimizer_updates_1_through_8": "authorized_existing_lora_bank_only",
        "optimizer_updates_9_through_100": (
            "authorized_existing_lora_bank_plus_v35_block_cross_matrices"
        ),
        "new_lora_bank_authorized": False,
        "all_non_authorized_inherited_v33_tensors_frozen": True,
        "all_other_preexisting_lora_banks_frozen": True,
        "all_other_followup_architectures_authorized": False,
        "chat_promotion_authorized": False,
        "final_test_access_authorized": False,
    }
    if dict(authorization) != exact_authorization:
        raise ValueError("V35 terminal report authorizes a different V36 surface")
    return {"path": str(path), "sha256": observed_sha, "report": report}


def require_exact_v35_source(config: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    contract = v36_contract(config)
    require_v35_terminal_gate(config)
    source = contract.source_checkpoint
    if source.is_symlink() or not source.is_dir() or source.name != "update_032":
        raise FileNotFoundError(f"V36 source must be real numbered V35 update 32: {source}")
    for filename, expected in contract.source_file_sha256.items():
        candidate = source / filename
        if candidate.is_symlink() or not candidate.is_file() or _sha256(candidate) != expected:
            raise ValueError(f"V36 source file differs from its exact pin: {candidate}")
    state = load_file(source / "adapter.safetensors", device="cpu")
    if tensor_state_sha256(state) != contract.source_tensor_state_sha256:
        raise ValueError("V36 source tensor state differs from exact V35 update 32")
    inherited = {
        key: value for key, value in state.items() if not key.startswith("block_cross_residual.")
    }
    if tensor_state_sha256(inherited) != contract.inherited_v33_tensor_state_sha256:
        raise ValueError("V36 source no longer contains the exact frozen V33 stack")
    core = {
        key.removeprefix("block_cross_residual."): value
        for key, value in state.items()
        if key.startswith("block_cross_residual.")
    }
    if tensor_state_sha256(core) != contract.core_source_state_sha256:
        raise ValueError("V36 source block core differs from exact V35 update 32")
    bank_prefix = f"lora_banks.{_BANK_NAME}."
    bank = {
        key.removeprefix(bank_prefix): value
        for key, value in state.items()
        if key.startswith(bank_prefix)
    }
    if tensor_state_sha256(bank) != contract.decoder_bank_initial_state_sha256:
        raise ValueError("V36 source query bank is not its exact zero-output state")
    metadata = json.loads((source / TRAINING_METADATA_FILENAME).read_text(encoding="utf-8"))
    if not isinstance(metadata, dict) or metadata.get("optimizer_step") != 32:
        raise ValueError("V36 source metadata is not V35 update 32")
    stage = _mapping(metadata.get("v35_block_cross"), "source v35_block_cross")
    if stage.get("update32_train_only_gate", {}).get("passed") is not False:
        raise ValueError("V36 source does not preserve V35's stopped update-32 evidence")
    if metadata.get("frozen_block_cross_source_stack_state_sha256") != (
        contract.inherited_v33_tensor_state_sha256
    ):
        raise ValueError("V36 source metadata does not pin its frozen V33 stack")
    runtime = json.loads((source / RUNTIME_METADATA_FILENAME).read_text(encoding="utf-8"))
    validate_runtime_checkpoint_metadata(runtime)
    if runtime != runtime_checkpoint_metadata(metadata):
        raise ValueError("V36 source runtime metadata is not exact sanitized V35 metadata")
    return source, metadata


def construct_v36_source_core(
    config: Mapping[str, Any], *, device: torch.device
) -> BlockCrossResidual:
    """Construct the architecture, then load only the learned V35 core state."""

    contract = v36_contract(config)
    core = construct_v35_core(config, device=torch.device("cpu"))
    state = load_file(contract.source_checkpoint / "adapter.safetensors", device="cpu")
    prefix = "block_cross_residual."
    core_state = {
        key.removeprefix(prefix): value for key, value in state.items() if key.startswith(prefix)
    }
    core.load_state_dict(core_state, strict=True)
    core = core.to(device)
    validate_block_cross_residual_state(
        core,
        expected_parameter_count=983_040,
        expected_state_sha256=contract.core_source_state_sha256,
        context="V36 learned V35 source core",
    )
    return core


def _target_bank(bundle: V30Bundle):
    if bundle.trainable_bank_name != _BANK_NAME:
        raise ValueError("V36 bundle does not expose the locked query bank")
    bank = bundle.lora_installation.bank(_BANK_NAME).installation
    if bank.parameter_count != 131_072:
        raise RuntimeError("V36 query-bank parameter count changed")
    if tuple(bank.target_names) != tuple(
        f"model.language_model.layers.{index}.self_attn.q_proj" for index in range(18, 22)
    ):
        raise RuntimeError("V36 query-bank target modules changed")
    return bank


def decoder_parameter_groups(
    bundle: V30Bundle,
) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    bank = _target_bank(bundle)
    a_parameters: list[torch.nn.Parameter] = []
    b_parameters: list[torch.nn.Parameter] = []
    for adapter in bank.adapters:
        if not isinstance(adapter, LoRALinear):
            raise TypeError("V36 target bank contains a non-LoRA adapter")
        a_parameters.append(adapter.lora_a)
        b_parameters.append(adapter.lora_b)
    counts = (
        sum(parameter.numel() for parameter in a_parameters),
        sum(parameter.numel() for parameter in b_parameters),
    )
    if counts != (49_152, 81_920):
        raise RuntimeError(f"V36 query-bank A/B counts changed: {counts}")
    return a_parameters, b_parameters


def core_parameter_groups(
    block_cross_residual: torch.nn.Module,
) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    named = dict(block_cross_residual.named_parameters())
    if tuple(named) != _CORE_NAMES:
        raise RuntimeError(f"V36 core parameter names/order changed: {tuple(named)}")
    return (
        [named[name] for name in _CORE_QKV_NAMES],
        [named[name] for name in _CORE_OUTPUT_NAMES],
    )


def verify_v36_update_zero_surfaces(
    bundle: V30Bundle, block_cross_residual: BlockCrossResidual
) -> dict[str, Any]:
    bank = _target_bank(bundle)
    if bank.state_sha256() != _BANK_INITIAL_STATE_SHA256:
        raise ValueError("V36 query bank is not the exact V33 zero-output state")
    target_outputs: dict[str, bool] = {}
    for name, adapter in zip(bank.target_names, bank.adapters, strict=True):
        if torch.count_nonzero(adapter.lora_b).item() != 0:
            raise ValueError(f"V36 query-bank B is not exact zero: {name}")
        values = (
            torch.linspace(-0.25, 0.25, steps=2 * adapter.in_features, dtype=torch.float32)
            .reshape(2, adapter.in_features)
            .to(adapter.base.weight.device, adapter.base.weight.dtype)
        )
        was_training = adapter.training
        adapter.eval()
        with torch.inference_mode():
            target_outputs[name] = bool(torch.equal(adapter.base(values), adapter(values)))
        adapter.train(was_training)
        if not target_outputs[name]:
            raise RuntimeError(f"V36 query bank changes update-zero output: {name}")
    validate_block_cross_residual_state(
        block_cross_residual,
        expected_parameter_count=983_040,
        expected_state_sha256=_CORE_SOURCE_STATE_SHA256,
        context="V36 learned V35 block core",
    )
    with torch.inference_mode():
        base = torch.randn(1, 256, 1536, device=next(block_cross_residual.parameters()).device)
        blocks = torch.randn(9, 384, device=base.device)
        positions = torch.rand(9, 3, device=base.device).mul(2).sub(1)
        core_output = block_cross_residual(base, blocks, positions)
    if torch.equal(core_output, base) or not torch.isfinite(core_output).all():
        raise RuntimeError("V36 learned source core is inactive or nonfinite")
    return {
        "source": "exact_stopped_v35_update32",
        "exact_stopped_v35_update32_loaded": True,
        "fresh_v35_optimizer_state_loaded": False,
        "decoder_bank": _BANK_NAME,
        "decoder_bank_initial_state_sha256": bank.state_sha256(),
        "decoder_bank_exact_zero_output": True,
        "decoder_target_outputs_bit_exact": target_outputs,
        "block_core_source_state_sha256": block_cross_residual.state_sha256(),
        "learned_block_core_active": True,
        "joint_update_zero_equivalent_to_v35_update32": True,
    }


def freeze_for_v36(
    bundle: V30Bundle,
    block_cross_residual: BlockCrossResidual,
    *,
    optimizer_step: int,
) -> list[torch.nn.Parameter]:
    """Freeze everything except the exact stage-authorized V36 surfaces."""

    bundle.language.model.requires_grad_(False)
    for module in bundle.checkpoint_modules.values():
        module.requires_grad_(False)
        module.eval()
    block_cross_residual.requires_grad_(False)
    qkv, core_output = core_parameter_groups(block_cross_residual)
    decoder_a, decoder_b = decoder_parameter_groups(bundle)
    for parameter in (*decoder_a, *decoder_b):
        parameter.requires_grad_(True)
    if optimizer_step >= 8:
        for parameter in (*qkv, *core_output):
            parameter.requires_grad_(True)
    block_cross_residual.train()
    _target_bank(bundle).train(True)
    return [*qkv, *core_output, *decoder_a, *decoder_b]


def assert_v36_trainable_surface(
    bundle: V30Bundle,
    block_cross_residual: BlockCrossResidual,
    *,
    optimizer_step: int,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, Any]:
    qkv, core_output = core_parameter_groups(block_cross_residual)
    decoder_a, decoder_b = decoder_parameter_groups(bundle)
    all_parameters = [*qkv, *core_output, *decoder_a, *decoder_b]
    decoder = [*decoder_a, *decoder_b]
    expected_active = {
        id(parameter) for parameter in (decoder if optimizer_step < 8 else all_parameters)
    }
    observed_active = {
        id(parameter)
        for module in bundle.checkpoint_modules.values()
        for parameter in module.parameters()
        if parameter.requires_grad
    }
    if observed_active != expected_active:
        raise RuntimeError("V36 active trainable surface differs from its exact lock")
    if optimizer is not None:
        optimizer_ids = {
            id(parameter) for group in optimizer.param_groups for parameter in group["params"]
        }
        if optimizer_ids != {id(parameter) for parameter in all_parameters}:
            raise RuntimeError("V36 optimizer contains an unauthorized tensor")
    base_trainable = [
        name
        for name, parameter in bundle.language.model.named_parameters()
        if parameter.requires_grad
        and id(parameter) not in {id(value) for value in (*decoder_a, *decoder_b)}
    ]
    if base_trainable:
        raise RuntimeError(f"V36 unfroze Gemma base parameters: {base_trainable}")
    counts = {
        "block_qkv": sum(parameter.numel() for parameter in qkv),
        "block_output": sum(parameter.numel() for parameter in core_output),
        "decoder_a": sum(parameter.numel() for parameter in decoder_a),
        "decoder_b": sum(parameter.numel() for parameter in decoder_b),
    }
    if counts != {
        "block_qkv": 589_824,
        "block_output": 393_216,
        "decoder_a": 49_152,
        "decoder_b": 81_920,
    }:
        raise RuntimeError(f"V36 parameter counts changed: {counts}")
    return {
        "block_core_parameter_names": list(_CORE_NAMES),
        "decoder_bank": _BANK_NAME,
        "decoder_bank_parameter_names": list(_target_bank(bundle).state_module.state_dict()),
        "group_parameter_counts": counts,
        "block_core_parameter_count": 983_040,
        "decoder_bank_parameter_count": 131_072,
        "total_parameter_count": 1_114_112,
        "active_stage": "lora_only" if optimizer_step < 8 else "joint_full",
        "gemma_base_frozen": True,
        "all_other_lora_banks_frozen": True,
        "complete_v33_scene_stack_frozen": True,
        "every_other_parameter_frozen": True,
    }


def v36_optimizer(
    bundle: V30Bundle,
    block_cross_residual: BlockCrossResidual,
    settings: V36Settings,
) -> torch.optim.AdamW:
    qkv, core_output = core_parameter_groups(block_cross_residual)
    decoder_a, decoder_b = decoder_parameter_groups(bundle)
    optimizer = torch.optim.AdamW(
        [
            {
                "name": "block_cross_residual.qkv",
                "params": qkv,
                "parameter_names": [f"{_CORE_PREFIX}{name}" for name in _CORE_QKV_NAMES],
                "lr": 0.0,
                "weight_decay": settings.weight_decay,
            },
            {
                "name": "block_cross_residual.output",
                "params": core_output,
                "parameter_names": [f"{_CORE_PREFIX}{name}" for name in _CORE_OUTPUT_NAMES],
                "lr": 0.0,
                "weight_decay": settings.weight_decay,
            },
            {
                "name": f"lora_banks.{_BANK_NAME}",
                "params": [*decoder_a, *decoder_b],
                "parameter_names": list(_BANK_OPTIMIZER_PARAMETER_NAMES),
                "lr": settings.decoder_learning_rate,
                "weight_decay": settings.weight_decay,
            },
        ]
    )
    assert_v36_trainable_surface(
        bundle,
        block_cross_residual,
        optimizer_step=0,
        optimizer=optimizer,
    )
    return optimizer


def set_v36_optimizer_stage(
    *,
    bundle: V30Bundle,
    block_cross_residual: BlockCrossResidual,
    optimizer: torch.optim.Optimizer,
    optimizer_step_to_run: int,
    settings: V36Settings,
) -> None:
    if optimizer_step_to_run < 1:
        raise ValueError("V36 optimizer step to run must be positive")
    completed_steps = optimizer_step_to_run - 1
    freeze_for_v36(bundle, block_cross_residual, optimizer_step=completed_steps)
    groups = {str(group.get("name")): group for group in optimizer.param_groups}
    groups["block_cross_residual.qkv"]["lr"] = (
        0.0 if optimizer_step_to_run <= 8 else settings.qkv_learning_rate
    )
    groups["block_cross_residual.output"]["lr"] = (
        0.0 if optimizer_step_to_run <= 8 else settings.output_learning_rate
    )
    groups[f"lora_banks.{_BANK_NAME}"]["lr"] = settings.decoder_learning_rate
    assert_v36_trainable_surface(
        bundle,
        block_cross_residual,
        optimizer_step=completed_steps,
        optimizer=optimizer,
    )


def frozen_v36_state_sha256(bundle: V30Bundle, block_cross_residual: BlockCrossResidual) -> str:
    """Hash every persisted tensor except the two authorized V36 surfaces."""

    excluded_module = f"lora_banks.{_BANK_NAME}"
    state: dict[str, torch.Tensor] = {}
    for module_name, module in bundle.checkpoint_modules.items():
        if module_name == excluded_module:
            continue
        for name, value in module.state_dict().items():
            if (
                module is block_cross_residual or module_name == "block_cross_residual"
            ) and name in _CORE_NAMES:
                continue
            state[f"{module_name}.{name}"] = value
    return tensor_state_sha256(state)


def v36_weighted_objective(
    *,
    broad_nll: torch.Tensor,
    pair_correct_nll: torch.Tensor,
    side_hinge: torch.Tensor,
    cross_prefix_flip_hinge: torch.Tensor,
    normalized_residual_penalty: torch.Tensor,
    settings: V36Settings,
) -> torch.Tensor:
    return (
        settings.broad_nll_weight * broad_nll
        + settings.pair_correct_nll_weight * pair_correct_nll
        + settings.side_hinge_weight * side_hinge
        + settings.cross_prefix_flip_weight * cross_prefix_flip_hinge
        + settings.residual_penalty_weight * normalized_residual_penalty
    )


def v36_broad_calibration_records(
    schedule: Sequence[Any], *, count: int = 48
) -> tuple[QARecord, ...]:
    """Select one deterministic, train-only broad calibration suite."""

    selected: list[QARecord] = []
    seen: set[tuple[str, str]] = set()
    for item in schedule:
        record = item.broad_record
        key = (record.scene_id, record.question_id)
        if key in seen:
            continue
        if record.counterfactual_expected_change is True:
            raise ValueError("V36 broad calibration contains changed supervision")
        seen.add(key)
        selected.append(record)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError(f"V36 needs {count} unique unchanged broad calibration rows")
    return tuple(selected)


def training_broad_nll(
    *,
    records: Sequence[QARecord],
    caches: Mapping[str, V35SceneCache],
    block_cross_residual: BlockCrossResidual,
    bundle: V30Bundle,
) -> float:
    total = 0.0
    count = 0
    block_cross_residual.eval()
    with torch.inference_mode():
        scene_tokens = {
            scene_id: current_scene_tokens(
                caches[scene_id], block_cross_residual, device=bundle.language.device
            )
            for scene_id in sorted({record.scene_id for record in records})
        }
        for offset in range(0, len(records), 2):
            rows = records[offset : offset + 2]
            batches = [
                _compose_answer_batch(
                    scene_tokens=scene_tokens[record.scene_id],
                    question=record.question,
                    answer=record.answer,
                    bundle=bundle,
                )
                for record in rows
            ]
            batch = stack_prefix_batches(
                batches,
                bundle.language.device,
                prefix_backend=getattr(bundle.language, "prefix_backend", None),
            )
            if batch.labels is None:
                raise RuntimeError("V36 broad calibration batch lacks answer labels")
            values = token_normalized_nll(
                forward_prefix_batch(bundle.language, batch).logits, batch.labels
            ).reshape(-1)
            total += float(values.sum().cpu())
            count += len(rows)
    if count != len(records) or count <= 0:
        raise RuntimeError("V36 broad calibration omitted training rows")
    result = total / count
    if not math.isfinite(result):
        raise RuntimeError("V36 broad calibration NLL is nonfinite")
    return result


def _family_for_pair(pair_id: str) -> str:
    return {
        "pair_000015": "book_support",
        "pair_000016": "mirror_lr",
        "pair_000017": "picture_support",
    }.get(pair_id, "other")


def complete_physical_pair_coverage(pair_metrics: Mapping[str, Any]) -> int:
    rows = pair_metrics.get("units")
    if not isinstance(rows, list):
        raise TypeError("V36 pair metrics lack unit rows")
    complete = {
        str(row["pair_id"])
        for row in rows
        if isinstance(row, Mapping) and row.get("complete") is True
    }
    return len(complete)


def _training_greedy_metrics_impl(
    *,
    units: Sequence[CounterfactualPairUnit],
    broad_records: Sequence[QARecord],
    caches: Mapping[str, V35SceneCache],
    block_cross_residual: BlockCrossResidual,
    bundle: V30Bundle,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Greedily score only locked training rows; validation is unreachable."""

    scene_ids = sorted(
        {record.scene_id for unit in units for record in unit.records}
        | {record.scene_id for record in broad_records}
    )
    model_dtype = next(bundle.language.model.parameters()).dtype
    block_cross_residual.eval()
    with torch.inference_mode():
        prefixes = {
            scene_id: bundle.composer.scene_prefix(
                current_scene_tokens(
                    caches[scene_id], block_cross_residual, device=bundle.language.device
                ).to(model_dtype)
            )
            for scene_id in scene_ids
        }
    complete = 0
    complete_pairs: set[str] = set()
    complete_by_family = {
        "book_support": 0,
        "mirror_lr": 0,
        "picture_support": 0,
    }
    changed_rows = 0
    broad_correct = 0
    with torch.inference_mode():
        for unit in sorted(units, key=lambda value: (value.pair_id, value.question_key)):
            correct = []
            for record in unit.records:
                _, prediction, _ = _question_logits_and_answer(
                    bundle.language, prefixes[record.scene_id], dict(config), record.question
                )
                correct.append(exact_normalized_match(prediction, record.answer))
            is_complete = all(correct)
            complete += int(is_complete)
            changed_rows += sum(int(value) for value in correct)
            if is_complete:
                complete_pairs.add(unit.pair_id)
                family = _family_for_pair(unit.pair_id)
                if family in complete_by_family:
                    complete_by_family[family] += 1
        for record in broad_records:
            _, prediction, _ = _question_logits_and_answer(
                bundle.language, prefixes[record.scene_id], dict(config), record.question
            )
            broad_correct += int(exact_normalized_match(prediction, record.answer))
    return {
        "schema_version": 1,
        "changed_unit_count": len(units),
        "changed_row_count": 2 * len(units),
        "changed_rows_exact_correct": changed_rows,
        "complete_units": complete,
        "complete_physical_pair_coverage": len(complete_pairs),
        "complete_units_by_family": complete_by_family,
        "broad_row_count": len(broad_records),
        "broad_exact_correct": broad_correct,
        "broad_exact_accuracy": broad_correct / len(broad_records),
        "training_scenes_only": True,
        "validation_qa_loaded": False,
    }


def training_greedy_metrics(
    *,
    units: Sequence[CounterfactualPairUnit],
    broad_records: Sequence[QARecord],
    caches: Mapping[str, V35SceneCache],
    block_cross_residual: BlockCrossResidual,
    bundle: V30Bundle,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    decoder = bundle.language.decoder_module
    was_training = bool(decoder.training)
    decoder.eval()
    try:
        return _training_greedy_metrics_impl(
            units=units,
            broad_records=broad_records,
            caches=caches,
            block_cross_residual=block_cross_residual,
            bundle=bundle,
            config=config,
        )
    finally:
        decoder.train(was_training)


def _priority_families_complete(pair_metrics: Mapping[str, Any]) -> bool:
    families = _mapping(pair_metrics.get("complete_units_by_family"), "complete families")
    return all(
        int(families.get(name, 0)) >= 1
        for name in (
            "book_support",
            "mirror_lr",
            "picture_support",
        )
    )


def v36_update16_gate(
    *,
    pair_metrics: Mapping[str, Any],
    broad_nll: float,
    source_broad_nll: float,
    residual_rms: float,
    decoder_bank_state_sha256: str,
    frozen_nonauthorized_state_sha256: str,
    contract: V36Contract,
) -> dict[str, Any]:
    rules = contract.update16_gate
    checks = {
        "teacher_complete_units_at_least_10": int(pair_metrics["complete_units"]) >= 10,
        "teacher_cross_complete_units_at_least_16": int(pair_metrics["cross_prefix_complete_units"])
        >= 16,
        "teacher_positive_sides_at_least_34": int(pair_metrics["positive_sides"]) >= 34,
        "mean_cross_prefix_margin_strictly_above_v35_source": float(
            pair_metrics["mean_cross_prefix_margin"]
        )
        > float(rules["mean_cross_prefix_margin_strict_minimum"]),
        "complete_physical_pair_coverage_at_least_5": complete_physical_pair_coverage(pair_metrics)
        >= 5,
        "unchanged_broad_nll_within_1_02x_source": broad_nll
        <= source_broad_nll * float(rules["unchanged_broad_nll_ratio_maximum"]),
        "residual_rms_at_most_0_075": residual_rms <= 0.075,
        "decoder_bank_state_changed": decoder_bank_state_sha256
        != contract.decoder_bank_initial_state_sha256,
        "frozen_nonauthorized_state_exact": frozen_nonauthorized_state_sha256
        == contract.frozen_nonauthorized_state_sha256,
    }
    return {
        **checks,
        "passed": all(checks.values()),
        "training_scenes_only": True,
        "validation_qa_loaded": False,
    }


def v36_update32_gate(
    *,
    update16_gate: Mapping[str, Any],
    pair_metrics: Mapping[str, Any],
    broad_nll: float,
    source_broad_nll: float,
    residual_rms: float,
    contract: V36Contract,
) -> dict[str, Any]:
    rules = contract.update32_gate
    checks = {
        "update16_gate_passed": update16_gate.get("passed") is True,
        "teacher_complete_units_at_least_12": int(pair_metrics["complete_units"]) >= 12,
        "teacher_cross_complete_units_at_least_18": int(pair_metrics["cross_prefix_complete_units"])
        >= 18,
        "teacher_positive_sides_at_least_37": int(pair_metrics["positive_sides"]) >= 37,
        "mean_cross_prefix_margin_at_least_1_37265043258667": float(
            pair_metrics["mean_cross_prefix_margin"]
        )
        >= float(rules["mean_cross_prefix_margin_minimum"]),
        "complete_physical_pair_coverage_at_least_6": complete_physical_pair_coverage(pair_metrics)
        >= 6,
        "each_priority_family_has_a_complete_unit": _priority_families_complete(pair_metrics),
        "unchanged_broad_nll_within_1_03x_source": broad_nll
        <= source_broad_nll * float(rules["unchanged_broad_nll_ratio_maximum"]),
        "residual_rms_at_most_0_075": residual_rms <= 0.075,
    }
    return {
        **checks,
        "passed": all(checks.values()),
        "training_scenes_only": True,
        "validation_qa_loaded": False,
    }


def v36_update64_gate(
    *,
    update32_gate: Mapping[str, Any],
    pair_metrics: Mapping[str, Any],
    greedy_metrics: Mapping[str, Any],
    source_greedy_metrics: Mapping[str, Any],
    residual_rms: float,
    contract: V36Contract,
) -> dict[str, Any]:
    rules = contract.update64_gate
    greedy_families = _mapping(
        greedy_metrics.get("complete_units_by_family"), "greedy complete families"
    )
    checks = {
        "update32_gate_passed": update32_gate.get("passed") is True,
        "teacher_complete_units_at_least_15": int(pair_metrics["complete_units"]) >= 15,
        "teacher_cross_complete_units_at_least_20": int(pair_metrics["cross_prefix_complete_units"])
        >= 20,
        "teacher_positive_sides_at_least_40": int(pair_metrics["positive_sides"]) >= 40,
        "complete_physical_pair_coverage_at_least_7": complete_physical_pair_coverage(pair_metrics)
        >= 7,
        "each_priority_family_teacher_complete": _priority_families_complete(pair_metrics),
        "greedy_complete_units_at_least_6": int(greedy_metrics["complete_units"]) >= 6,
        "each_priority_family_greedy_complete": all(
            int(greedy_families.get(name, 0)) >= 1
            for name in ("book_support", "mirror_lr", "picture_support")
        ),
        "broad_greedy_exact_accuracy_within_0_02_of_source": float(
            greedy_metrics["broad_exact_accuracy"]
        )
        >= float(source_greedy_metrics["broad_exact_accuracy"])
        - float(rules["broad_greedy_exact_accuracy_maximum_drop"]),
        "residual_rms_at_most_0_10": residual_rms <= 0.10,
    }
    return {
        **checks,
        "passed": all(checks.values()),
        "training_scenes_only": True,
        "validation_qa_loaded": False,
    }


def preflight_v36(config: Mapping[str, Any], *, require_train_qa: bool = True) -> dict[str, Any]:
    contract = v36_contract(config)
    settings = v36_settings(config)
    terminal = require_v35_terminal_gate(config)
    source, _ = require_exact_v35_source(config)
    assert_deferred_final_scenes_absent(config)
    qa_audit = None
    if require_train_qa:
        records, qa_audit = load_v35_train_qa_records(config)
        if len(records) != 384:
            raise RuntimeError("V36 train-only QA count changed")
    fresh_architecture = construct_v35_core(config, device=torch.device("cpu"))
    validate_block_cross_residual_state(
        fresh_architecture,
        expected_parameter_count=983_040,
        expected_state_sha256=_CORE_INITIAL_STATE_SHA256,
        context="V36 preflight architecture reconstruction",
    )
    core = construct_v36_source_core(config, device=torch.device("cpu"))
    return {
        "schema_version": 1,
        "artifact": "v36_joint_block_cross_preflight",
        "passed": True,
        "source_checkpoint": str(source),
        "source_optimizer_step": 32,
        "source_v35_tensor_state_sha256": contract.source_tensor_state_sha256,
        "inherited_v33_tensor_state_sha256": contract.inherited_v33_tensor_state_sha256,
        "source_block_core_state_sha256": core.state_sha256(),
        "fresh_architecture_initial_state_sha256": fresh_architecture.state_sha256(),
        "decoder_bank_initial_state_sha256": contract.decoder_bank_initial_state_sha256,
        "v35_terminal_report_sha256": terminal["sha256"],
        "exact_trainable_parameter_count": 1_114_112,
        "saved_optimizer_steps": list(settings.saved_optimizer_steps),
        "train_qa_loaded": require_train_qa,
        "validation_qa_loaded": False,
        "gemma_loaded": False,
        "scene_maps_loaded": False,
        "exact_stopped_v35_update32_selected_as_source": True,
        "v35_optimizer_state_loaded": False,
        "oracle_loaded": False,
        "final_test_scenes_touched": False,
        "qa_audit": qa_audit,
    }


def _block_source_stack_state_sha256(
    bundle: V30Bundle, block_cross_residual: BlockCrossResidual
) -> str:
    modules = {
        name: module
        for name, module in bundle.checkpoint_modules.items()
        if module is not block_cross_residual and name != "block_cross_residual"
    }
    return module_collection_state_sha256(modules)


def _block_zero_equivalence() -> dict[str, Any]:
    # This is the architecture's update-zero equivalence contract. V36 loads
    # learned V35 weights, while the runtime still verifies the same route and
    # application order.
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


def _gradient_norm(parameters: Sequence[torch.nn.Parameter]) -> float:
    if not parameters:
        return 0.0
    squared = sum(
        torch.zeros((), device=parameter.device)
        if parameter.grad is None
        else parameter.grad.detach().float().square().sum()
        for parameter in parameters
    )
    return float(squared.sqrt().cpu())


def _source_replay_attestation(
    *,
    source_metadata: Mapping[str, Any],
    pair_metrics: Mapping[str, Any],
    residual: Mapping[str, Any],
) -> dict[str, Any]:
    history = source_metadata.get("history")
    if not isinstance(history, list) or len(history) != 33:
        raise ValueError("V36 source lacks V35's complete update-32 history")
    row = _mapping(history[-1], "V35 source history[-1]")
    expected_pair = _mapping(row.get("training_pair_metrics"), "V35 source pair metrics")
    expected_residual = _mapping(
        row.get("training_residual_diagnostics"), "V35 source residual diagnostics"
    )
    if dict(pair_metrics) != dict(expected_pair):
        raise RuntimeError("V36 update zero does not bit-replay V35 pair metrics")
    if dict(residual) != dict(expected_residual):
        raise RuntimeError("V36 update zero does not bit-replay V35 residual diagnostics")
    return {
        "exact_stopped_v35_update32_loaded": True,
        "source_optimizer_step": 32,
        "source_pair_metrics_bit_exact": True,
        "source_residual_diagnostics_bit_exact": True,
        "source_complete_units": int(pair_metrics["complete_units"]),
        "source_cross_prefix_complete_units": int(pair_metrics["cross_prefix_complete_units"]),
        "source_positive_sides": int(pair_metrics["positive_sides"]),
        "source_mean_cross_prefix_margin": float(pair_metrics["mean_cross_prefix_margin"]),
        "source_complete_units_by_family": dict(
            _mapping(pair_metrics["complete_units_by_family"], "source families")
        ),
        "source_residual_rms": float(residual["aggregate_rms"]),
        "v35_optimizer_state_loaded": False,
        "fresh_adam_state": True,
        "validation_qa_loaded": False,
    }


def _prefix_replay_attestation(
    *,
    caches: Mapping[str, V35SceneCache],
    block_cross_residual: BlockCrossResidual,
    bundle: V30Bundle,
) -> dict[str, Any]:
    model_dtype = next(bundle.language.model.parameters()).dtype
    first: dict[str, str] = {}
    repeated: dict[str, str] = {}
    with torch.inference_mode():
        for scene_id, cache in sorted(caches.items()):
            current = current_scene_tokens(
                cache, block_cross_residual, device=bundle.language.device
            )
            first[scene_id] = prefix_sha256(bundle.composer.scene_prefix(current.to(model_dtype)))
            replay = current_scene_tokens(
                cache, block_cross_residual, device=bundle.language.device
            )
            repeated[scene_id] = prefix_sha256(bundle.composer.scene_prefix(replay.to(model_dtype)))
    if first != repeated or len(first) != 22:
        raise RuntimeError("V36 source prefix replay is incomplete or nondeterministic")
    return {
        "source_prefix_scene_count": 22,
        "source_prefix_sha256_by_scene": first,
        "source_prefixes_replayed_bit_exact": True,
        "scene_prefixes_built_before_questions": True,
        "validation_scene_prefixes_question_free": True,
        "validation_qa_loaded": False,
        "question_dependent_scene_processing": False,
        "question_dependent_retrieval": False,
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
    update_zero_surfaces: Mapping[str, Any],
    source_replay: Mapping[str, Any],
    prefix_replay: Mapping[str, Any],
    source_pair_metrics: Mapping[str, Any],
    source_broad_nll: float,
    source_greedy_metrics: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
    optimizer_step: int,
    bundle: V30Bundle,
    block_cross_residual: BlockCrossResidual,
    surface: Mapping[str, Any],
    gate16: Mapping[str, Any] | None,
    gate32: Mapping[str, Any] | None,
    gate64: Mapping[str, Any] | None,
) -> dict[str, Any]:
    contract = v36_contract(config)
    lora_settings = lora_banks_settings(config)
    lora_optimizer = lora_banks_optimizer_settings(config, lora_settings)
    metadata = copy.deepcopy(dict(source_metadata))
    metadata.update(
        {
            "config_hash": config_hash(dict(config)),
            "epoch": optimizer_step,
            "optimizer_step": optimizer_step,
            "best_epoch": 0,
            "best_monitor_loss": None,
            "monitor_name": "independent_v36_selector_required",
            "history": list(history),
            "lora": lora_banks_checkpoint_contract(
                lora_settings,
                lora_optimizer,
                bundle.lora_installation.parameter_counts,
            ),
            **bundle.lora_installation.checkpoint_metadata(),
            "block_cross_residual": block_cross_residual_settings(config).contract(),
            "block_cross_residual_parameter_count": block_cross_residual.parameter_count,
            "block_cross_residual_initial_state_sha256": _CORE_INITIAL_STATE_SHA256,
            "block_cross_residual_state_sha256": block_cross_residual.state_sha256(),
            "block_cross_residual_zero_output_equivalence": _block_zero_equivalence(),
            # Runtime checks the complete current non-core stack (including the
            # authorized V36 LoRA bank) against this content hash.
            "frozen_block_cross_source_stack_state_sha256": (
                _block_source_stack_state_sha256(bundle, block_cross_residual)
            ),
            "question_dependent_scene_processing": False,
        }
    )
    metadata["v36_joint_block_cross"] = {
        "schema_version": 1,
        "artifact": "v36_diverse28_joint_block_cross_training",
        "optimizer_step": optimizer_step,
        "conditional_v35_terminal_gate": {
            "path": terminal["path"],
            "sha256": terminal["sha256"],
        },
        "source_checkpoint": str(contract.source_checkpoint),
        "source_file_sha256": dict(contract.source_file_sha256),
        "source_optimizer_step": 32,
        "source_v35_tensor_state_sha256": contract.source_tensor_state_sha256,
        "inherited_v33_tensor_state_sha256": (contract.inherited_v33_tensor_state_sha256),
        "source_block_core_state_sha256": contract.core_source_state_sha256,
        "decoder_bank_initial_state_sha256": contract.decoder_bank_initial_state_sha256,
        "schedule": dict(schedule_audit),
        "scene_cache": _deterministic_cache_audit(cache_audit),
        "train_qa_dataset": dict(qa_audit),
        "validation_qa_loaded": False,
        "update_zero_equivalence": dict(update_zero_surfaces),
        "source_replay_attestation": dict(source_replay),
        "prefix_replay_attestation": dict(prefix_replay),
        "source_pair_metrics": dict(source_pair_metrics),
        "source_broad_train_nll": source_broad_nll,
        "source_train_greedy_metrics": dict(source_greedy_metrics),
        "exact_trainable_parameter_count": 1_114_112,
        "trainable_surface": dict(surface),
        "frozen_nonauthorized_state_sha256": frozen_v36_state_sha256(bundle, block_cross_residual),
        "current_block_source_stack_state_sha256": (
            _block_source_stack_state_sha256(bundle, block_cross_residual)
        ),
        "fresh_adam": True,
        "source_v35_optimizer_state_loaded": False,
        "optimizer_stage_updates_1_through_8": "lora_only",
        "optimizer_stage_updates_9_through_100": "joint_core_and_lora",
        "separation_reference_sha256": separation_reference.audit_sha256,
        "euclidean_separation_is_descriptive_only": True,
        "update16_train_only_gate": None if gate16 is None else dict(gate16),
        "update32_train_only_gate": None if gate32 is None else dict(gate32),
        "update64_train_only_gate": None if gate64 is None else dict(gate64),
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


def latest_v36_resume_checkpoint(output: Path, contract: V36Contract) -> Path | None:
    if not output.exists():
        return None
    if output.is_symlink() or not output.is_dir():
        raise ValueError(f"V36 output root must be a real directory: {output}")
    parsed: dict[int, Path] = {}
    for path in output.glob("update_*"):
        if path.is_symlink() or not path.is_dir():
            raise ValueError(f"V36 update path must be a real directory: {path}")
        match = _UPDATE_DIRECTORY.fullmatch(path.name)
        if match is None or int(match.group(1)) not in contract.saved_optimizer_steps:
            raise ValueError(f"V36 output contains an unauthorized arm: {path.name}")
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
        raise ValueError("V36 complete arms are not a contiguous saved-step prefix")
    incomplete = sorted(set(parsed) - set(complete))
    if incomplete:
        raise ValueError(
            f"V36 output contains an incomplete arm; refusing an ambiguous resume: {incomplete}"
        )
    return None if not complete else parsed[complete[-1]]


def _adam_step(value: object, field: str) -> int:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"{field} is not scalar")
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be numeric")
    result = int(value)
    if float(value) != result or result < 1:
        raise ValueError(f"{field} must be a positive integer")
    return result


def _optimizer_step_audit(
    path: Path,
    *,
    expected_step: int,
    tensors: Mapping[str, torch.Tensor] | None = None,
) -> dict[str, Any]:
    """Prove exact parameter-to-moment identity before loading Adam state."""

    if tensors is None:
        tensors = load_file(path / "adapter.safetensors", device="cpu")
    state = torch.load(path / "optimizer.pt", map_location="cpu", weights_only=True)
    if not isinstance(state, Mapping):
        raise TypeError("V36 optimizer checkpoint must be a mapping")
    groups = state.get("param_groups")
    values = state.get("state")
    if not isinstance(groups, list) or not isinstance(values, Mapping) or len(groups) != 3:
        raise ValueError("V36 optimizer must retain exactly three AdamW groups")
    names = {
        "block_cross_residual.qkv",
        "block_cross_residual.output",
        f"lora_banks.{_BANK_NAME}",
    }
    by_name = {str(group.get("name")): group for group in groups}
    if set(by_name) != names or len(by_name) != len(groups):
        raise ValueError("V36 optimizer group names changed")
    expected_lrs = {
        "block_cross_residual.qkv": 0.0 if expected_step <= 8 else 1e-4,
        "block_cross_residual.output": 0.0 if expected_step <= 8 else 2.5e-5,
        f"lora_banks.{_BANK_NAME}": 2e-5,
    }
    if any(float(by_name[name]["lr"]) != lr for name, lr in expected_lrs.items()):
        raise ValueError("V36 optimizer learning-rate stage changed")
    if any(float(group["weight_decay"]) != 0.0 for group in groups):
        raise ValueError("V36 optimizer weight decay must remain zero")
    ordered_names = {
        "block_cross_residual.qkv": list(_CORE_PARAMETER_NAMES[:3]),
        "block_cross_residual.output": list(_CORE_PARAMETER_NAMES[3:]),
        f"lora_banks.{_BANK_NAME}": list(_BANK_OPTIMIZER_PARAMETER_NAMES),
    }
    expected_state_names = set(_BANK_PARAMETER_NAMES)
    if expected_step > 8:
        expected_state_names.update(_CORE_PARAMETER_NAMES)
    observed_parameter_ids: set[int] = set()
    expected_state_ids: set[int] = set()
    inspected: list[str] = []
    next_parameter_id = 0
    for group_name, parameter_names in ordered_names.items():
        group = by_name[group_name]
        raw_ids = group.get("params")
        if not isinstance(raw_ids, list) or len(raw_ids) != len(parameter_names):
            raise ValueError(f"V36 optimizer parameter inventory changed for {group_name}")
        canonical_ids = list(range(next_parameter_id, next_parameter_id + len(parameter_names)))
        next_parameter_id += len(parameter_names)
        if raw_ids != canonical_ids:
            raise ValueError(f"V36 optimizer parameter IDs changed for {group_name}")
        if group.get("parameter_names") != parameter_names:
            raise ValueError(f"V36 optimizer parameter order changed for {group_name}")
        for parameter_id, tensor_name in zip(raw_ids, parameter_names, strict=True):
            if isinstance(parameter_id, bool) or not isinstance(parameter_id, int):
                raise TypeError("V36 optimizer parameter ID must be an integer")
            if parameter_id in observed_parameter_ids:
                raise ValueError("V36 optimizer aliases a parameter across groups")
            observed_parameter_ids.add(parameter_id)
            if tensor_name not in tensors:
                raise ValueError(f"V36 adapter tensor is absent for {tensor_name}")
            tensor = tensors[tensor_name]
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"V36 adapter value is not a tensor for {tensor_name}")
            entry = values.get(parameter_id)
            if tensor_name not in expected_state_names:
                if entry is not None:
                    raise ValueError("V36 frozen-stage core unexpectedly has Adam state")
                continue
            expected_state_ids.add(parameter_id)
            if not isinstance(entry, Mapping) or set(entry) != {
                "step",
                "exp_avg",
                "exp_avg_sq",
            }:
                raise ValueError(f"V36 Adam state is incomplete for {tensor_name}")
            expected_tensor_step = (
                expected_step if tensor_name in _BANK_PARAMETER_NAME_SET else expected_step - 8
            )
            if _adam_step(entry["step"], f"Adam step for {tensor_name}") != (expected_tensor_step):
                raise ValueError(f"V36 Adam step changed for {tensor_name}")
            for moment_name in ("exp_avg", "exp_avg_sq"):
                moment = entry[moment_name]
                if (
                    not isinstance(moment, torch.Tensor)
                    or tuple(moment.shape) != tuple(tensor.shape)
                    or not torch.isfinite(moment).all()
                ):
                    raise ValueError(f"V36 Adam {moment_name} is invalid for {tensor_name}")
            inspected.append(tensor_name)
    if set(values) != expected_state_ids:
        raise ValueError("V36 Adam state contains an unauthorized parameter")
    return {
        "group_count": 3,
        "moment_tensor_count": 2 * len(inspected),
        "parameter_states_inspected": sorted(inspected),
        "lora_optimizer_step": expected_step,
        "block_core_optimizer_step": None if expected_step <= 8 else expected_step - 8,
        "exact_parameter_order_verified": True,
        "fresh_v36_adam_staging_verified": True,
    }


def validate_v36_resume_checkpoint(
    *,
    config: Mapping[str, Any],
    output: Path,
    resume: Path,
    contract: V36Contract,
    terminal: Mapping[str, Any],
    schedule_audit: Mapping[str, Any],
    cache_audit: Mapping[str, Any],
    separation_reference: V35SeparationReference,
    update_zero_surfaces: Mapping[str, Any],
    source_replay: Mapping[str, Any],
    source_broad_nll: float,
    source_greedy_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    if resume.parent != output or resume.is_symlink() or not resume.is_dir():
        raise ValueError("V36 resume must be a real numbered arm inside its output root")
    latest = latest_v36_resume_checkpoint(output, contract)
    if latest is None or latest != resume:
        raise ValueError("V36 resume must be the latest contiguous complete arm")
    match = _UPDATE_DIRECTORY.fullmatch(resume.name)
    if match is None:
        raise ValueError("V36 resume path is not a numbered update arm")
    step = int(match.group(1))
    if step not in contract.saved_optimizer_steps:
        raise ValueError("V36 resume update is not an authorized saved arm")
    required_files = (
        "adapter.safetensors",
        TRAINING_METADATA_FILENAME,
        RUNTIME_METADATA_FILENAME,
        *(("optimizer.pt",) if step else ()),
    )
    for filename in required_files:
        candidate = resume / filename
        if candidate.is_symlink() or not candidate.is_file():
            raise FileNotFoundError(f"V36 resume checkpoint is incomplete: {filename}")
    metadata = json.loads((resume / TRAINING_METADATA_FILENAME).read_text(encoding="utf-8"))
    stage = _mapping(metadata.get("v36_joint_block_cross"), "resume V36 stage")
    if metadata.get("config_hash") != config_hash(dict(config)):
        raise ValueError("V36 resume config hash changed")
    if metadata.get("optimizer_step") != step or stage.get("optimizer_step") != step:
        raise ValueError("V36 resume optimizer step mismatch")
    if stage.get("conditional_v35_terminal_gate") != {
        "path": terminal["path"],
        "sha256": terminal["sha256"],
    }:
        raise ValueError("V36 resume terminal authorization changed")
    if stage.get("schedule", {}).get("schedule_sha256") != schedule_audit["schedule_sha256"]:
        raise ValueError("V36 resume schedule changed")
    if stage.get("scene_cache") != _deterministic_cache_audit(cache_audit):
        raise ValueError("V36 resume question-free scene cache changed")
    if stage.get("update_zero_equivalence") != dict(update_zero_surfaces):
        raise ValueError("V36 resume update-zero surface equivalence changed")
    if stage.get("source_replay_attestation") != dict(source_replay):
        raise ValueError("V36 resume source replay changed")
    if float(stage.get("source_broad_train_nll")) != source_broad_nll:
        raise ValueError("V36 resume broad source baseline changed")
    if stage.get("source_train_greedy_metrics") != dict(source_greedy_metrics):
        raise ValueError("V36 resume greedy source baseline changed")
    if stage.get("separation_reference_sha256") != separation_reference.audit_sha256:
        raise ValueError("V36 resume separation reference changed")
    if stage.get("validation_qa_loaded") is not False:
        raise ValueError("V36 resume metadata says validation QA was loaded")
    history = metadata.get("history")
    if not isinstance(history, list) or len(history) != step + 1:
        raise ValueError("V36 resume history is incomplete")
    if [row.get("optimizer_update") for row in history] != list(range(step + 1)):
        raise ValueError("V36 resume history does not prove every true microstep")
    gate16 = stage.get("update16_train_only_gate")
    gate32 = stage.get("update32_train_only_gate")
    gate64 = stage.get("update64_train_only_gate")
    if step >= 16 and (not isinstance(gate16, Mapping) or gate16.get("passed") is not True):
        raise ValueError("V36 cannot resume past a failed/missing update-16 gate")
    if step >= 32 and (not isinstance(gate32, Mapping) or gate32.get("passed") is not True):
        raise ValueError("V36 cannot resume past a failed/missing update-32 gate")
    if step >= 64 and (not isinstance(gate64, Mapping) or gate64.get("passed") is not True):
        raise ValueError("V36 cannot resume past a failed/missing update-64 gate")
    runtime = json.loads((resume / RUNTIME_METADATA_FILENAME).read_text(encoding="utf-8"))
    validate_runtime_checkpoint_metadata(runtime)
    if runtime != runtime_checkpoint_metadata(metadata):
        raise ValueError("V36 resume runtime metadata is not sanitized exactly")
    if step:
        _optimizer_step_audit(resume, expected_step=step)
    return metadata


def run_v36(*, config: dict[str, Any], output: Path, resume: Path | None = None) -> dict[str, Any]:
    if resume is None and output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty V36 output: {output}")
    contract = v36_contract(config)
    settings = v36_settings(config)
    terminal = require_v35_terminal_gate(config)
    source_checkpoint, pinned_source_metadata = require_exact_v35_source(config)
    v34_terminal = require_v34_terminal_gate(config)
    assert_deferred_final_scenes_absent(config)
    seed = int(config["seed"])
    torch.manual_seed(seed)
    random.seed(seed)
    train_records, qa_audit = load_v35_train_qa_records(config)
    train_pairs = build_exact_question_pair_units(train_records)
    schedule, schedule_audit = build_v35_schedule(
        train_records,
        train_pairs,
        settings=v35_settings(config),
        seed=seed,
    )
    if schedule_audit["schedule_sha256"] != (terminal["report"]["schedule_sha256"]):
        raise RuntimeError("V36 schedule is not the exact V35 isolation schedule")
    broad_calibration = v36_broad_calibration_records(schedule)

    approved_v29 = require_approved_v29_source(config)
    bundle = load_v30_bundle(config, approved_v29)
    block_cross_residual = construct_v36_source_core(config, device=bundle.language.device)
    bundle.checkpoint_modules["block_cross_residual"] = block_cross_residual
    source_metadata = load_adapter_checkpoint(
        source_checkpoint,
        bundle.checkpoint_modules,
        device="cpu",
        metadata_filename=TRAINING_METADATA_FILENAME,
    )
    if source_metadata != pinned_source_metadata:
        raise RuntimeError("V36 source metadata changed during exact adapter load")
    if module_collection_state_sha256(bundle.checkpoint_modules) != (
        contract.source_tensor_state_sha256
    ):
        raise RuntimeError("V36 loaded modules are not exact V35 update 32")
    update_zero_surfaces = verify_v36_update_zero_surfaces(bundle, block_cross_residual)
    if frozen_v36_state_sha256(bundle, block_cross_residual) != (
        contract.frozen_nonauthorized_state_sha256
    ):
        raise RuntimeError("V36 source frozen-nonauthorized state changed")

    all_scene_ids = (
        *v31_contract(config).train_scene_ids,
        *v31_contract(config).validation_scene_ids,
    )
    caches, cache_audit = cache_v35_scenes(
        config=config,
        bundle=bundle,
        source_metadata=pinned_source_metadata,
        terminal=v34_terminal,
        scene_ids=all_scene_ids,
    )
    train_caches = {scene_id: caches[scene_id] for scene_id in v31_contract(config).train_scene_ids}
    prefix_replay = _prefix_replay_attestation(
        caches=caches,
        block_cross_residual=block_cross_residual,
        bundle=bundle,
    )
    source_pair_metrics = training_pair_metrics(
        units=train_pairs,
        caches=train_caches,
        block_cross_residual=block_cross_residual,
        bundle=bundle,
        settings=v35_settings(config),
    )
    source_residual = residual_rms_diagnostics(
        caches=train_caches,
        block_cross_residual=block_cross_residual,
        device=bundle.language.device,
    )
    source_replay = _source_replay_attestation(
        source_metadata=pinned_source_metadata,
        pair_metrics=source_pair_metrics,
        residual=source_residual,
    )
    source_broad_nll = training_broad_nll(
        records=broad_calibration,
        caches=train_caches,
        block_cross_residual=block_cross_residual,
        bundle=bundle,
    )
    source_greedy_metrics = training_greedy_metrics(
        units=train_pairs,
        broad_records=broad_calibration,
        caches=train_caches,
        block_cross_residual=block_cross_residual,
        bundle=bundle,
        config=config,
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

    freeze_for_v36(bundle, block_cross_residual, optimizer_step=0)
    surface0 = assert_v36_trainable_surface(bundle, block_cross_residual, optimizer_step=0)
    history: list[dict[str, Any]] = [
        {
            "optimizer_update": 0,
            "source_pair_metrics": source_pair_metrics,
            "source_broad_train_nll": source_broad_nll,
            "source_train_greedy_metrics": source_greedy_metrics,
            "training_prefix_separation": baseline_separation,
            "training_residual_diagnostics": source_residual,
            "update_zero_surfaces": update_zero_surfaces,
            "validation_qa_loaded": False,
            "oracle_environment_files_loaded": False,
            "saved_checkpoint": True,
        }
    ]
    optimizer = v36_optimizer(bundle, block_cross_residual, settings)
    if optimizer.state:
        raise RuntimeError("V36 Adam state is not fresh at update zero")
    start_step = 0
    accepted16: Mapping[str, Any] | None = None
    accepted32: Mapping[str, Any] | None = None
    accepted64: Mapping[str, Any] | None = None
    output.mkdir(parents=True, exist_ok=True)
    if resume is not None:
        resume_metadata = validate_v36_resume_checkpoint(
            config=config,
            output=output,
            resume=resume,
            contract=contract,
            terminal=terminal,
            schedule_audit=schedule_audit,
            cache_audit=cache_audit,
            separation_reference=separation_reference,
            update_zero_surfaces=update_zero_surfaces,
            source_replay=source_replay,
            source_broad_nll=source_broad_nll,
            source_greedy_metrics=source_greedy_metrics,
        )
        loaded = load_adapter_checkpoint(
            resume,
            bundle.checkpoint_modules,
            device="cpu",
            metadata_filename=TRAINING_METADATA_FILENAME,
        )
        if loaded != resume_metadata:
            raise RuntimeError("V36 resume metadata changed during adapter load")
        validate_block_cross_residual_state(
            block_cross_residual,
            expected_parameter_count=983_040,
            expected_state_sha256=str(resume_metadata["block_cross_residual_state_sha256"]),
            context="V36 resumed block core",
        )
        resumed_bank_hashes = _mapping(
            resume_metadata.get("lora_bank_state_sha256"),
            "V36 resume LoRA-bank hashes",
        )
        if _target_bank(bundle).state_sha256() != resumed_bank_hashes.get(_BANK_NAME):
            raise RuntimeError("V36 resumed query-bank state differs from metadata")
        if frozen_v36_state_sha256(bundle, block_cross_residual) != (
            contract.frozen_nonauthorized_state_sha256
        ):
            raise RuntimeError("V36 resume changed a nonauthorized inherited tensor")
        if _block_source_stack_state_sha256(bundle, block_cross_residual) != (
            resume_metadata.get("frozen_block_cross_source_stack_state_sha256")
        ):
            raise RuntimeError("V36 resumed non-core runtime stack hash changed")
        start_step = int(resume_metadata["optimizer_step"])
        if start_step:
            load_optimizer_checkpoint(resume, optimizer, bundle.language.device)
        history = list(resume_metadata["history"])
        stage = _mapping(resume_metadata["v36_joint_block_cross"], "resume V36 stage")
        accepted16 = stage.get("update16_train_only_gate")
        accepted32 = stage.get("update32_train_only_gate")
        accepted64 = stage.get("update64_train_only_gate")
    else:
        metadata0 = _metadata(
            source_metadata=pinned_source_metadata,
            config=config,
            terminal=terminal,
            schedule_audit=schedule_audit,
            cache_audit=cache_audit,
            qa_audit=qa_audit,
            separation_reference=separation_reference,
            update_zero_surfaces=update_zero_surfaces,
            source_replay=source_replay,
            prefix_replay=prefix_replay,
            source_pair_metrics=source_pair_metrics,
            source_broad_nll=source_broad_nll,
            source_greedy_metrics=source_greedy_metrics,
            history=history,
            optimizer_step=0,
            bundle=bundle,
            block_cross_residual=block_cross_residual,
            surface=surface0,
            gate16=None,
            gate32=None,
            gate64=None,
        )
        _save(output / "update_000", bundle=bundle, metadata=metadata0, optimizer=None)
        saved0 = load_file(output / "update_000" / "adapter.safetensors", device="cpu")
        if tensor_state_sha256(saved0) != contract.source_tensor_state_sha256:
            raise RuntimeError("V36 saved update zero differs from exact V35 update 32")

    qkv, core_output = core_parameter_groups(block_cross_residual)
    decoder_a, decoder_b = decoder_parameter_groups(bundle)
    decoder_parameters = [*decoder_a, *decoder_b]
    for item in schedule[start_step:]:
        step = item.optimizer_step
        set_v36_optimizer_stage(
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
        pair_nll, side_hinge, cross_hinge, pair_diagnostics = paired_cross_prefix_objective(
            unit=item.pair_unit,
            scene_tokens=pair_tokens,
            bundle=bundle,
            side_margin=settings.side_hinge_margin,
            cross_prefix_margin=settings.cross_prefix_flip_margin,
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
        side_margin_mean = float(pair_diagnostics["side_margins"].float().mean().cpu())
        cross_margin_mean = float(pair_diagnostics["cross_prefix_margins"].float().mean().cpu())
        del pair_nll, side_hinge, cross_hinge, pair_diagnostics, pair_objective, pair_tokens

        normalized_residual, residual_rms = residual_penalty(
            caches=train_caches,
            block_cross_residual=block_cross_residual,
            device=bundle.language.device,
            scale=settings.residual_penalty_scale,
        )
        if step >= 9:
            (settings.residual_penalty_weight * normalized_residual).backward()
        normalized_residual_value = float(normalized_residual.detach().cpu())
        residual_rms_value = float(residual_rms.detach().cpu())
        del normalized_residual, residual_rms

        active_parameters = [
            parameter
            for parameter in (*qkv, *core_output, *decoder_parameters)
            if parameter.requires_grad
        ]
        if any(parameter.grad is None for parameter in active_parameters):
            raise RuntimeError("V36 one or more active tensors lacks a gradient")
        if any(not torch.isfinite(parameter.grad).all() for parameter in active_parameters):
            raise RuntimeError("V36 active gradient is nonfinite")
        qkv_preclip = _gradient_norm(qkv) if step >= 9 else 0.0
        output_preclip = _gradient_norm(core_output) if step >= 9 else 0.0
        decoder_preclip = _gradient_norm(decoder_parameters)
        if step >= 9:
            torch.nn.utils.clip_grad_norm_(qkv, settings.qkv_gradient_clip_norm)
            torch.nn.utils.clip_grad_norm_(core_output, settings.output_gradient_clip_norm)
        torch.nn.utils.clip_grad_norm_(decoder_parameters, settings.decoder_gradient_clip_norm)
        optimizer.step()
        if frozen_v36_state_sha256(bundle, block_cross_residual) != (
            contract.frozen_nonauthorized_state_sha256
        ):
            raise RuntimeError("V36 changed a nonauthorized inherited tensor")
        validate_block_cross_residual_state(
            block_cross_residual,
            expected_parameter_count=983_040,
            context=f"V36 update {step}",
        )
        _target_bank(bundle).validate_state()

        should_save = step in contract.saved_optimizer_steps
        separation = None
        residual_diagnostics = None
        pair_metrics = None
        broad_nll_diagnostic = None
        greedy_metrics = None
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
        if step in {8, 16, 32, 64, 100}:
            pair_metrics = training_pair_metrics(
                units=train_pairs,
                caches=train_caches,
                block_cross_residual=block_cross_residual,
                bundle=bundle,
                settings=v35_settings(config),
            )
        if step in {16, 32, 64}:
            broad_nll_diagnostic = training_broad_nll(
                records=broad_calibration,
                caches=train_caches,
                block_cross_residual=block_cross_residual,
                bundle=bundle,
            )
        if step == 64:
            greedy_metrics = training_greedy_metrics(
                units=train_pairs,
                broad_records=broad_calibration,
                caches=train_caches,
                block_cross_residual=block_cross_residual,
                bundle=bundle,
                config=config,
            )
        if step == 16:
            assert pair_metrics is not None and residual_diagnostics is not None
            assert broad_nll_diagnostic is not None
            accepted16 = v36_update16_gate(
                pair_metrics=pair_metrics,
                broad_nll=broad_nll_diagnostic,
                source_broad_nll=source_broad_nll,
                residual_rms=float(residual_diagnostics["aggregate_rms"]),
                decoder_bank_state_sha256=_target_bank(bundle).state_sha256(),
                frozen_nonauthorized_state_sha256=frozen_v36_state_sha256(
                    bundle, block_cross_residual
                ),
                contract=contract,
            )
        if step == 32:
            if not isinstance(accepted16, Mapping):
                raise RuntimeError("V36 update-32 gate lacks update-16 evidence")
            assert pair_metrics is not None and residual_diagnostics is not None
            assert broad_nll_diagnostic is not None
            accepted32 = v36_update32_gate(
                update16_gate=accepted16,
                pair_metrics=pair_metrics,
                broad_nll=broad_nll_diagnostic,
                source_broad_nll=source_broad_nll,
                residual_rms=float(residual_diagnostics["aggregate_rms"]),
                contract=contract,
            )
        if step == 64:
            if not isinstance(accepted32, Mapping):
                raise RuntimeError("V36 update-64 gate lacks update-32 evidence")
            assert pair_metrics is not None and residual_diagnostics is not None
            assert greedy_metrics is not None
            accepted64 = v36_update64_gate(
                update32_gate=accepted32,
                pair_metrics=pair_metrics,
                greedy_metrics=greedy_metrics,
                source_greedy_metrics=source_greedy_metrics,
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
                "optimizer_stage": "lora_only" if step <= 8 else "joint_core_and_lora",
                "preclip_gradient_norm_by_group": {
                    "qkv": qkv_preclip,
                    "output": output_preclip,
                    "decoder_lora": decoder_preclip,
                },
                "separate_group_clipping": True,
                "training_prefix_separation_descriptive_only": separation,
                "training_pair_metrics": pair_metrics,
                "training_broad_nll": broad_nll_diagnostic,
                "training_greedy_metrics": greedy_metrics,
                "training_residual_diagnostics": residual_diagnostics,
                "update16_train_only_gate": accepted16,
                "update32_train_only_gate": accepted32,
                "update64_train_only_gate": accepted64,
                "validation_qa_loaded": False,
                "oracle_environment_files_loaded": False,
                "saved_checkpoint": should_save,
            }
        )
        if not should_save:
            continue
        surface = assert_v36_trainable_surface(
            bundle,
            block_cross_residual,
            optimizer_step=step - 1,
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
            update_zero_surfaces=update_zero_surfaces,
            source_replay=source_replay,
            prefix_replay=prefix_replay,
            source_pair_metrics=source_pair_metrics,
            source_broad_nll=source_broad_nll,
            source_greedy_metrics=source_greedy_metrics,
            history=history,
            optimizer_step=step,
            bundle=bundle,
            block_cross_residual=block_cross_residual,
            surface=surface,
            gate16=accepted16,
            gate32=accepted32,
            gate64=accepted64,
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
                    "phase": "v36_joint_block_cross_checkpoint",
                    "optimizer_step": step,
                    "optimizer_stage": "lora_only" if step <= 8 else "joint_core_and_lora",
                    "training_complete_units": None
                    if pair_metrics is None
                    else pair_metrics["complete_units"],
                    "training_cross_complete_units": None
                    if pair_metrics is None
                    else pair_metrics["cross_prefix_complete_units"],
                    "training_residual_rms": residual_diagnostics["aggregate_rms"],
                    "update16_gate_passed": None
                    if accepted16 is None
                    else accepted16.get("passed"),
                    "update32_gate_passed": None
                    if accepted32 is None
                    else accepted32.get("passed"),
                    "update64_gate_passed": None
                    if accepted64 is None
                    else accepted64.get("passed"),
                    "validation_qa_loaded": False,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if step == 16 and accepted16.get("passed") is not True:
            raise RuntimeError("V36 update-16 train-only causal gate failed")
        if step == 32 and accepted32.get("passed") is not True:
            raise RuntimeError("V36 update-32 train-only causal gate failed")
        if step == 64 and accepted64.get("passed") is not True:
            raise RuntimeError("V36 update-64 train-only causal gate failed")

    return {
        "schema_version": 1,
        "artifact": "v36_diverse28_joint_block_cross_training",
        "output": str(output),
        "optimizer_updates": 100,
        "resumed_from_optimizer_step": start_step,
        "saved_optimizer_steps": list(contract.saved_optimizer_steps),
        "exact_trainable_parameter_count": 1_114_112,
        "source_v35_tensor_state_sha256": contract.source_tensor_state_sha256,
        "inherited_v33_tensor_state_sha256": contract.inherited_v33_tensor_state_sha256,
        "source_block_core_state_sha256": contract.core_source_state_sha256,
        "v35_terminal_report_sha256": terminal["sha256"],
        "update16_train_only_gate": accepted16,
        "update32_train_only_gate": accepted32,
        "update64_train_only_gate": accepted64,
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
        report = preflight_v36(config)
    else:
        output = _resolve(args.output)
        resume = _resolve(args.resume) if args.resume is not None else None
        if args.resume_latest:
            resume = latest_v36_resume_checkpoint(output, v36_contract(config))
            if resume is None:
                raise FileNotFoundError("V36 has no complete checkpoint to resume")
        report = run_v36(config=config, output=output, resume=resume)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "V36Contract",
    "V36Settings",
    "assert_v36_trainable_surface",
    "core_parameter_groups",
    "decoder_parameter_groups",
    "freeze_for_v36",
    "frozen_v36_state_sha256",
    "latest_v36_resume_checkpoint",
    "preflight_v36",
    "require_exact_v35_source",
    "require_v35_terminal_gate",
    "run_v36",
    "set_v36_optimizer_stage",
    "v36_contract",
    "v36_optimizer",
    "v36_settings",
    "v36_update16_gate",
    "v36_update32_gate",
    "v36_update64_gate",
    "v36_weighted_objective",
    "validate_v36_resume_checkpoint",
    "verify_v36_update_zero_surfaces",
]
