"""Bounded V37 training of the existing learned scene-ingress K/V LoRA bank.

V37 starts from the exact conditionally authorized V36 update 16.  It never
loads V36 Adam state and trains only ``extension_v23_shared_kv`` on the exact
Gemma layer-13/14 K/V projections.  The learned V36 block core and query bank,
the complete scene stack, Gemma base, and every other persisted tensor remain
bit exact.  Validation QA is reserved for the independent V37 selector.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from semantic_3d_chat.config import PROJECT_ROOT, artifact_root, config_hash, load_config
from semantic_3d_chat.data.dataset import QARecord
from semantic_3d_chat.language.lora import (
    InstalledLoRABank,
    LoRAInstallation,
    lora_banks_checkpoint_contract,
    lora_banks_optimizer_settings,
    lora_banks_settings,
    tensor_state_sha256,
    validate_lora_banks_checkpoint_state,
)
from semantic_3d_chat.language.prefix_injection import prefix_sha256
from semantic_3d_chat.scene_encoder.block_cross_residual import (
    BlockCrossResidual,
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
)
from semantic_3d_chat.training.train_block_cross_v35 import (
    V35Microstep,
    V35SceneCache,
    _deterministic_cache_audit,
    broad_answer_nll,
    build_v35_schedule,
    cache_v35_scenes,
    current_scene_tokens,
    load_v35_train_qa_records,
    paired_cross_prefix_objective,
    residual_rms_diagnostics,
    training_pair_metrics,
    v35_settings,
)
from semantic_3d_chat.training.train_environmental_sidecar_v33 import (
    assert_deferred_final_scenes_absent,
)
from semantic_3d_chat.training.train_joint_block_cross_v36 import (
    _gradient_norm,
    complete_physical_pair_coverage,
    construct_v36_source_core,
    require_v34_terminal_gate,
    training_broad_nll,
    training_greedy_metrics,
    v36_broad_calibration_records,
)
from semantic_3d_chat.training.train_joint_pair_v30 import (
    ApprovedV29Source,
    V30Bundle,
    load_v30_bundle,
    require_approved_v29_source,
    select_balanced_broad_records,
)
from semantic_3d_chat.training.train_joint_pair_v31 import v31_contract

DEFAULT_CONFIG = Path("configs/experiments/gemma4_diverse28_scene_ingress_kv_v37.yaml")
DEFAULT_OUTPUT = Path("data_gemma4/checkpoints/gemma4_v37_diverse28_scene_ingress_kv")
_UPDATE_DIRECTORY = re.compile(r"update_([0-9]{3})")
OPTIMIZER_AUDIT_FILENAME = "optimizer_audit.json"
_TARGET_BANK = "extension_v23_shared_kv"
_QUERY_BANK = "extension_v30_joint_pair_query"
_TARGET_PREFIX = f"lora_banks.{_TARGET_BANK}."
_QUERY_PREFIX = f"lora_banks.{_QUERY_BANK}."
_CORE_PREFIX = "block_cross_residual."
_CORE_PARAMETER_BASENAMES = ("w_q", "w_k", "w_v", "w_o")
_CORE_PARAMETER_NAMES = frozenset(f"{_CORE_PREFIX}{name}" for name in _CORE_PARAMETER_BASENAMES)
_TARGET_PARAMETER_NAMES = tuple(
    f"{_TARGET_PREFIX}adapters.{index}.{side}"
    for index in range(4)
    for side in ("lora_a", "lora_b")
)
_TARGET_PARAMETER_NAME_SET = frozenset(_TARGET_PARAMETER_NAMES)
_TARGET_MODULES = (
    "model.language_model.layers.13.self_attn.k_proj",
    "model.language_model.layers.13.self_attn.v_proj",
    "model.language_model.layers.14.self_attn.k_proj",
    "model.language_model.layers.14.self_attn.v_proj",
)
_QUERY_MODULES = (
    "model.language_model.layers.18.self_attn.q_proj",
    "model.language_model.layers.19.self_attn.q_proj",
    "model.language_model.layers.20.self_attn.q_proj",
    "model.language_model.layers.21.self_attn.q_proj",
)
_QUERY_CONSTRUCTION_STATE_SHA256 = (
    "2b1d89fbb9189ac551bf12905cf94036ebaa84696449b31c2b37b69d478fb70d"
)
_PAIR_FAMILIES = {
    "book_support": "pair_000015",
    "mirror_lr": "pair_000016",
    "picture_support": "pair_000017",
}
_WARMUP_KEYS = (
    "cfq_13b1138d14c52a7c",
    "cfq_1c8b8cd72fcde904",
    "cfq_163eb92339ad35a5",
    "cfq_66aab89cee5bef49",
    "cfq_a1c673a1197a0961",
    "cfq_d469c4ac156ac42d",
    "cfq_ac7ac024c40aaddc",
    "cfq_fa3601dfffa80a0e",
)
_TAIL_KEYS = _WARMUP_KEYS[:6]
_SOURCE_CHECKPOINT = Path(
    "data_gemma4/checkpoints/gemma4_v36_diverse28_joint_block_cross/update_016"
)
_SOURCE_FILE_SHA256 = {
    "adapter.safetensors": "6ed86fb51502f7330c75cc48b9be970eb0a933eb19da971a7e04726c419c3be5",
    TRAINING_METADATA_FILENAME: "7e7c257a1e42d20b7f2270a0257969ae006c3c27859e707c18d21b5537a89342",
    RUNTIME_METADATA_FILENAME: "63a27773e5d127c063b762cf110c1ed1d4022908bd9e4b843509dc399fe7f6dc",
    "optimizer.pt": "51a76712d87f24af793a28848d743034b9229d5e1df63d02c81e13efb5f12569",
}
_SOURCE_TENSOR_STATE_SHA256 = "e9b6d1362d58f34aede04817b0c8d81320c616dcd4b64e9c0d3bbe56b5835dd7"
_SOURCE_CORE_STATE_SHA256 = "92652fd2dbde2406227503f50717b2031baa1bcbc050902a379ddb9ddb52764f"
_SOURCE_QUERY_STATE_SHA256 = "050706c300e6fb0ac8e4cc02e26c565b54a9a89505104302d4ffcedc02124c64"
_TARGET_INITIAL_STATE_SHA256 = "91a9eea577cab5a37e840cdf4007722a398415846af91280713bcb2cda0f045c"
_SOURCE_V36_FROZEN_STATE_SHA256 = "b394d502f0c32a694c2d1a448cdf3849c47efc4058cb1f1331fe4a97d381b1dc"
_V37_FROZEN_COMPLEMENT_SHA256 = "c82b8715aebcb775a6e23cb5cd477520922682b5f41929017f4f91917eafe061"
_V36_TERMINAL_SHA256 = "cb5b1248a4904dc58a685b64e052f980c02771b59eed5578bdbf2865ddbf5877"
_SCHEDULE_SHA256 = "76a123412d4bd3aeee012515b37095c22d9cbf9eb56934b622d715daca45fa2b"
_SOURCE_PAIR_MEAN = 1.4565558433532715
_SOURCE_BROAD_NLL = 2.915099874138832


@dataclass(frozen=True)
class V37Settings:
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
    learning_rate: float
    weight_decay: float
    gradient_clip_norm: float

    @property
    def saved_optimizer_steps(self) -> tuple[int, ...]:
        return tuple(range(0, self.optimizer_steps + 1, self.checkpoint_interval_steps))


@dataclass(frozen=True)
class V37Contract:
    source_checkpoint: Path
    source_file_sha256: Mapping[str, str]
    source_tensor_state_sha256: str
    source_core_state_sha256: str
    source_query_state_sha256: str
    target_source_state_sha256: str
    source_v36_frozen_state_sha256: str
    frozen_complement_state_sha256: str
    terminal_report: Path
    terminal_report_sha256: str
    schedule_sha256: str
    saved_optimizer_steps: tuple[int, ...]
    update16_gate: Mapping[str, Any]
    update32_gate: Mapping[str, Any]
    update64_gate: Mapping[str, Any]


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (PROJECT_ROOT / value).resolve()


def _unresolved_project_path(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return Path(os.path.abspath(value if value.is_absolute() else PROJECT_ROOT / value))


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


def _finite(value: object, field: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0):
        raise ValueError(f"{field} must be finite" + (" and positive" if positive else ""))
    return result


def _positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def v37_settings(config: Mapping[str, Any]) -> V37Settings:
    training = _mapping(config.get("training"), "training")
    raw = _mapping(
        training.get("v37_scene_ingress_kv"),
        "training.v37_scene_ingress_kv",
    )
    settings = V37Settings(
        enabled=raw.get("enabled") is True,
        optimizer_steps=_positive_int(raw.get("optimizer_steps"), "optimizer_steps"),
        checkpoint_interval_steps=_positive_int(
            raw.get("checkpoint_interval_steps"), "checkpoint_interval_steps"
        ),
        broad_nll_weight=_finite(raw.get("broad_nll_weight"), "broad_nll_weight"),
        pair_correct_nll_weight=_finite(
            raw.get("pair_correct_nll_weight"), "pair_correct_nll_weight"
        ),
        side_hinge_weight=_finite(raw.get("side_hinge_weight"), "side_hinge_weight"),
        side_hinge_margin=_finite(raw.get("side_hinge_margin"), "side_hinge_margin"),
        cross_prefix_flip_weight=_finite(
            raw.get("cross_prefix_flip_weight"), "cross_prefix_flip_weight"
        ),
        cross_prefix_flip_margin=_finite(
            raw.get("cross_prefix_flip_margin"), "cross_prefix_flip_margin"
        ),
        residual_penalty_weight=_finite(
            raw.get("residual_penalty_weight"), "residual_penalty_weight"
        ),
        residual_penalty_scale=_finite(
            raw.get("residual_penalty_scale"), "residual_penalty_scale", positive=True
        ),
        learning_rate=_finite(raw.get("learning_rate"), "learning_rate", positive=True),
        weight_decay=_finite(raw.get("weight_decay"), "weight_decay"),
        gradient_clip_norm=_finite(
            raw.get("gradient_clip_norm"), "gradient_clip_norm", positive=True
        ),
    )
    expected = {
        "enabled": True,
        "optimizer_steps": 64,
        "checkpoint_interval_steps": 8,
        "broad_nll_weight": 0.25,
        "pair_correct_nll_weight": 0.5,
        "side_hinge_weight": 4.0,
        "side_hinge_margin": 0.5,
        "cross_prefix_flip_weight": 8.0,
        "cross_prefix_flip_margin": 0.25,
        "residual_penalty_weight": 0.001,
        "residual_penalty_scale": 0.05,
        "learning_rate": 2e-5,
        "weight_decay": 0.0,
        "gradient_clip_norm": 1.0,
    }
    if set(raw) != set(expected) or settings.__dict__ != expected:
        raise ValueError("V37 optimizer/objective settings changed from the terminal lock")
    if (
        _finite(training.get("lora_learning_rate"), "training.lora_learning_rate")
        != settings.learning_rate
        or _finite(training.get("lora_weight_decay"), "training.lora_weight_decay")
        != settings.weight_decay
    ):
        raise ValueError("V37 global LoRA optimizer contract disagrees with its actual AdamW")
    return settings


def v37_contract(config: Mapping[str, Any]) -> V37Contract:
    settings = v37_settings(config)
    raw = _mapping(config.get("v37_scene_ingress_kv"), "v37_scene_ingress_kv")
    expected_contract_keys = {
        "schema_version",
        "role",
        "engine",
        "v36_terminal_gate_report",
        "v36_terminal_gate_report_sha256",
        "source_checkpoint",
        "source_optimizer_step",
        "source_file_sha256",
        "source_v36_tensor_state_sha256",
        "source_block_core_state_sha256",
        "source_query_bank_name",
        "source_query_bank_state_sha256",
        "target_bank_name",
        "target_bank_parameter_count",
        "target_bank_source_state_sha256",
        "target_bank_rank",
        "target_bank_alpha",
        "target_bank_dropout",
        "target_modules",
        "target_tensor_count",
        "source_v36_frozen_nonauthorized_state_sha256",
        "v37_frozen_nonauthorized_state_sha256",
        "source_optimizer_state_loaded",
        "source_optimizer_file_opened",
        "validation_qa_loaded_during_training",
        "continuation_gates_use_training_only",
        "question_dependent_scene_processing",
        "question_dependent_retrieval",
        "train_scene_ids",
        "validation_scene_ids",
        "deferred_final_scene_ids",
        "priority_pair_ids",
        "warmup_question_keys",
        "tail_question_keys",
        "schedule_sha256",
        "saved_optimizer_steps",
        "update16_gate",
        "update32_gate",
        "update64_gate",
        "selector_uses_validation_only_after_complete_training",
        "promotion_greedy_complete_units_minimum",
        "promotion_validation_unit_count",
        "final_test_deferred",
    }
    if set(raw) != expected_contract_keys:
        raise ValueError("V37 contract contains missing or unknown fields")
    if raw.get("schema_version") != 1:
        raise ValueError("V37 contract schema changed")
    source_files = {
        str(key): str(value)
        for key, value in _mapping(raw.get("source_file_sha256"), "source files").items()
    }
    exact_fields = {
        "role": "exact_v36_u16_learned_shared_kv_lora_v37",
        "engine": "fresh_adam_existing_learned_scene_ingress_kv_true_microsteps",
        "source_optimizer_step": 16,
        "source_v36_tensor_state_sha256": _SOURCE_TENSOR_STATE_SHA256,
        "source_block_core_state_sha256": _SOURCE_CORE_STATE_SHA256,
        "source_query_bank_name": _QUERY_BANK,
        "source_query_bank_state_sha256": _SOURCE_QUERY_STATE_SHA256,
        "target_bank_name": _TARGET_BANK,
        "target_bank_parameter_count": 30_720,
        "target_bank_source_state_sha256": _TARGET_INITIAL_STATE_SHA256,
        "target_bank_rank": 4,
        "target_bank_alpha": 8.0,
        "target_bank_dropout": 0.0,
        "target_tensor_count": 8,
        "source_v36_frozen_nonauthorized_state_sha256": _SOURCE_V36_FROZEN_STATE_SHA256,
        "v37_frozen_nonauthorized_state_sha256": _V37_FROZEN_COMPLEMENT_SHA256,
        "source_optimizer_state_loaded": False,
        "source_optimizer_file_opened": False,
        "validation_qa_loaded_during_training": False,
        "continuation_gates_use_training_only": True,
        "question_dependent_scene_processing": False,
        "question_dependent_retrieval": False,
        "selector_uses_validation_only_after_complete_training": True,
        "promotion_greedy_complete_units_minimum": 6,
        "promotion_validation_unit_count": 12,
        "final_test_deferred": True,
    }
    if any(raw.get(key) != value for key, value in exact_fields.items()):
        raise ValueError("V37 exact source or surface contract changed")
    if _resolve(str(raw.get("source_checkpoint"))) != _resolve(_SOURCE_CHECKPOINT):
        raise ValueError("V37 source checkpoint changed")
    if source_files != _SOURCE_FILE_SHA256:
        raise ValueError("V37 source file hashes changed")
    if tuple(raw.get("target_modules", ())) != _TARGET_MODULES:
        raise ValueError("V37 exact Gemma K/V target paths changed")
    if dict(_mapping(raw.get("priority_pair_ids"), "priority_pair_ids")) != _PAIR_FAMILIES:
        raise ValueError("V37 priority family IDs changed")
    if tuple(raw.get("warmup_question_keys", ())) != _WARMUP_KEYS:
        raise ValueError("V37 warmup schedule changed")
    if tuple(raw.get("tail_question_keys", ())) != _TAIL_KEYS:
        raise ValueError("V37 tail schedule changed")
    if raw.get("v36_terminal_gate_report_sha256") != _V36_TERMINAL_SHA256:
        raise ValueError("V37 V36 terminal report hash is absent or changed")
    schedule_hash = str(raw.get("schedule_sha256"))
    if schedule_hash != _SCHEDULE_SHA256:
        raise ValueError("V37 schedule hash is absent or changed")
    if tuple(raw.get("saved_optimizer_steps", ())) != settings.saved_optimizer_steps:
        raise ValueError("V37 saved-arm envelope changed")
    split = v31_contract(v37_loader_config(config))
    if tuple(raw.get("train_scene_ids", ())) != split.train_scene_ids:
        raise ValueError("V37 train scenes changed")
    if tuple(raw.get("validation_scene_ids", ())) != split.validation_scene_ids:
        raise ValueError("V37 validation scenes changed")
    if tuple(raw.get("deferred_final_scene_ids", ())) != split.deferred_final_scene_ids:
        raise ValueError("V37 deferred final scenes changed")
    lora = lora_banks_settings(config)
    bank = next((value for value in lora.banks if value.name == _TARGET_BANK), None)
    query_bank = next((value for value in lora.banks if value.name == _QUERY_BANK), None)
    if bank is None or not (
        bank.adapter.rank == 4
        and bank.adapter.alpha == 8.0
        and bank.adapter.dropout == 0.0
        and bank.adapter.target_modules == _TARGET_MODULES
        and bank.initialization_algorithm == "checkpoint_overwrite"
        and bank.initialization_seed is None
        and bank.expected_initial_state_sha256 == _TARGET_INITIAL_STATE_SHA256
        and bank.trainable is True
    ):
        raise ValueError("V37 inherited learned K/V bank architecture changed")
    if query_bank is None or not (
        query_bank.trainable is False
        and query_bank.adapter.rank == 8
        and query_bank.adapter.alpha == 16.0
        and query_bank.adapter.dropout == 0.0
        and query_bank.adapter.target_modules == _QUERY_MODULES
        and query_bank.initialization_algorithm == "checkpoint_overwrite"
        and query_bank.initialization_seed is None
        and query_bank.expected_initial_state_sha256 == _SOURCE_QUERY_STATE_SHA256
    ):
        raise ValueError("V37 learned V36 query bank must be configured frozen")
    expected_gates = {
        "update16_gate": {
            "optimizer_step": 16,
            "complete_units_minimum": 10,
            "complete_physical_pair_coverage_minimum": 5,
            "cross_prefix_complete_units_minimum": 16,
            "positive_sides_minimum": 35,
            "mean_cross_prefix_margin_minimum": _SOURCE_PAIR_MEAN,
            "book_and_picture_complete_units_minimum": 1,
            "book_cross_prefix_complete_units_minimum": 1,
            "picture_cross_prefix_complete_units_minimum": 2,
            "mirror_complete_units_minimum": 2,
            "broad_nll_maximum": 2.973401871621609,
            "target_bank_state_must_change": True,
            "frozen_nonauthorized_state_must_remain_exact": True,
            "residual_diagnostics_must_remain_exact": True,
        },
        "update32_gate": {
            "optimizer_step": 32,
            "require_update16_passed": True,
            "complete_units_minimum": 12,
            "complete_physical_pair_coverage_minimum": 6,
            "cross_prefix_complete_units_minimum": 18,
            "positive_sides_minimum": 37,
            "mean_cross_prefix_margin_minimum": _SOURCE_PAIR_MEAN,
            "book_complete_units_minimum": 1,
            "picture_complete_units_minimum": 1,
            "mirror_complete_units_minimum": 2,
            "broad_nll_maximum": 3.002552870362997,
            "frozen_nonauthorized_state_must_remain_exact": True,
            "residual_diagnostics_must_remain_exact": True,
        },
        "update64_gate": {
            "optimizer_step": 64,
            "require_update32_passed": True,
            "complete_units_minimum": 15,
            "complete_physical_pair_coverage_minimum": 7,
            "cross_prefix_complete_units_minimum": 20,
            "positive_sides_minimum": 40,
            "book_complete_units_minimum": 1,
            "picture_complete_units_minimum": 1,
            "mirror_complete_units_minimum": 2,
            "greedy_complete_units_minimum": 6,
            "require_one_greedy_complete_per_priority_family": True,
            "broad_greedy_exact_accuracy_maximum_drop": 0.02,
            "broad_nll_maximum": 3.060854867845774,
            "frozen_nonauthorized_state_must_remain_exact": True,
            "residual_diagnostics_must_remain_exact": True,
        },
    }
    if any(dict(_mapping(raw.get(key), key)) != value for key, value in expected_gates.items()):
        raise ValueError("V37 train-only continuation gates changed")
    return V37Contract(
        source_checkpoint=_resolve(_SOURCE_CHECKPOINT),
        source_file_sha256=source_files,
        source_tensor_state_sha256=_SOURCE_TENSOR_STATE_SHA256,
        source_core_state_sha256=_SOURCE_CORE_STATE_SHA256,
        source_query_state_sha256=_SOURCE_QUERY_STATE_SHA256,
        target_source_state_sha256=_TARGET_INITIAL_STATE_SHA256,
        source_v36_frozen_state_sha256=_SOURCE_V36_FROZEN_STATE_SHA256,
        frozen_complement_state_sha256=_V37_FROZEN_COMPLEMENT_SHA256,
        terminal_report=_resolve(str(raw.get("v36_terminal_gate_report"))),
        terminal_report_sha256=_V36_TERMINAL_SHA256,
        schedule_sha256=schedule_hash,
        saved_optimizer_steps=settings.saved_optimizer_steps,
        update16_gate=dict(_mapping(raw.get("update16_gate"), "update16_gate")),
        update32_gate=dict(_mapping(raw.get("update32_gate"), "update32_gate")),
        update64_gate=dict(_mapping(raw.get("update64_gate"), "update64_gate")),
    )


def require_v36_terminal_gate(config: Mapping[str, Any]) -> dict[str, Any]:
    contract = v37_contract(config)
    path = contract.terminal_report
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"V37 requires a real V36 terminal report: {path}")
    if _sha256(path) != contract.terminal_report_sha256:
        raise ValueError("V37 V36 terminal report hash changed")
    report = json.loads(path.read_text(encoding="utf-8"))
    authorization = _mapping(report.get("conditional_authorization"), "V36 authorization")
    expected = {
        "authorized": True,
        "stage": "v37_scene_ingress_kv",
        "scope": "continue_existing_extension_v23_shared_kv_only",
        "authorized_existing_lora_bank": _TARGET_BANK,
        "authorized_existing_lora_parameter_count": 30_720,
        "authorized_existing_lora_tensor_count": 8,
        "authorized_existing_lora_rank": 4,
        "authorized_existing_lora_alpha": 8.0,
        "authorized_existing_lora_dropout": 0.0,
        "authorized_existing_lora_target_module_paths": list(_TARGET_MODULES),
        "source_checkpoint": str(_SOURCE_CHECKPOINT),
        "source_full_tensor_state_sha256": _SOURCE_TENSOR_STATE_SHA256,
        "source_learned_block_core_state_sha256": _SOURCE_CORE_STATE_SHA256,
        "source_learned_v30_query_bank_state_sha256": _SOURCE_QUERY_STATE_SHA256,
        "source_existing_shared_kv_bank_state_sha256": _TARGET_INITIAL_STATE_SHA256,
        "source_frozen_nonauthorized_state_sha256": _SOURCE_V36_FROZEN_STATE_SHA256,
        "v37_frozen_complement_state_sha256": _V37_FROZEN_COMPLEMENT_SHA256,
        "fresh_adam_required": True,
        "v36_optimizer_state_may_be_loaded": False,
        "maximum_true_optimizer_steps": 64,
        "learning_rate": 2e-5,
        "weight_decay": 0.0,
        "gradient_clip_norm": 1.0,
        "validation_qa_or_model_selection_before_complete_update64": False,
        "final_test_access_authorized": False,
        "oracle_access_authorized": False,
    }
    if report.get("artifact") != "v36_update16_terminal_gate" or report.get("passed") is not True:
        raise ValueError("V37 V36 terminal report did not pass")
    if report.get("conditional_v37_scene_ingress_kv_authorized") is not True:
        raise ValueError("V37 continuation is not conditionally authorized")
    if any(authorization.get(key) != value for key, value in expected.items()):
        raise ValueError("V37 terminal authorization surface changed")
    if dict(_mapping(authorization.get("source_file_sha256"), "terminal source files")) != (
        _SOURCE_FILE_SHA256
    ):
        raise ValueError("V37 terminal source hashes changed")
    terminal_warmup = tuple(
        str(row["question_key"]) for row in authorization.get("updates_1_through_8", ())
    )
    terminal_tail = tuple(
        str(row["question_key"]) for row in authorization.get("updates_59_through_64", ())
    )
    if terminal_warmup != _WARMUP_KEYS or terminal_tail != _TAIL_KEYS:
        raise ValueError("V37 terminal schedule authorization changed")
    return {
        "path": str(path),
        "sha256": contract.terminal_report_sha256,
        "report": report,
        "authorization": dict(authorization),
    }


def _bank_state(tensors: Mapping[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    return {
        name.removeprefix(prefix): value
        for name, value in tensors.items()
        if name.startswith(prefix)
    }


def _v36_frozen_source_state(
    tensors: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {
        name: value
        for name, value in tensors.items()
        if not name.startswith(_QUERY_PREFIX)
        and not (name.startswith(_CORE_PREFIX) and name in _CORE_PARAMETER_NAMES)
    }


def _v37_frozen_complement(
    tensors: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {name: value for name, value in tensors.items() if not name.startswith(_TARGET_PREFIX)}


def require_exact_v36_source(
    config: Mapping[str, Any],
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    """Audit V36 update 16 without opening its optimizer file."""

    contract = v37_contract(config)
    terminal = require_v36_terminal_gate(config)
    source = contract.source_checkpoint
    if source.is_symlink() or not source.is_dir():
        raise FileNotFoundError(f"V37 source must be a real directory: {source}")
    for filename in contract.source_file_sha256:
        candidate = source / filename
        if candidate.is_symlink() or not candidate.is_file():
            raise FileNotFoundError(f"V37 source file is missing or aliased: {candidate}")
    observed = {
        filename: _sha256(source / filename)
        for filename in (
            "adapter.safetensors",
            TRAINING_METADATA_FILENAME,
            RUNTIME_METADATA_FILENAME,
        )
    }
    if observed != {
        key: value for key, value in contract.source_file_sha256.items() if key != "optimizer.pt"
    }:
        raise ValueError("V37 exact V36 source files changed")
    metadata = json.loads((source / TRAINING_METADATA_FILENAME).read_text(encoding="utf-8"))
    runtime = json.loads((source / RUNTIME_METADATA_FILENAME).read_text(encoding="utf-8"))
    validate_runtime_checkpoint_metadata(runtime)
    if runtime != runtime_checkpoint_metadata(metadata):
        raise ValueError("V37 source runtime metadata is not exactly sanitized")
    tensors = load_file(source / "adapter.safetensors", device="cpu")
    if tensor_state_sha256(tensors) != contract.source_tensor_state_sha256:
        raise ValueError("V37 source tensor state changed")
    core = _bank_state(tensors, _CORE_PREFIX)
    target = _bank_state(tensors, _TARGET_PREFIX)
    query = _bank_state(tensors, _QUERY_PREFIX)
    if tensor_state_sha256(core) != contract.source_core_state_sha256:
        raise ValueError("V37 source learned block core changed")
    if tensor_state_sha256(query) != contract.source_query_state_sha256:
        raise ValueError("V37 source learned query bank changed")
    if {f"{_TARGET_PREFIX}{name}" for name in target} != _TARGET_PARAMETER_NAME_SET:
        raise ValueError("V37 source target-bank tensor inventory changed")
    if sum(value.numel() for value in target.values()) != 30_720:
        raise ValueError("V37 source target-bank parameter count changed")
    if tensor_state_sha256(target) != contract.target_source_state_sha256:
        raise ValueError("V37 source learned target bank changed")
    if tensor_state_sha256(_v36_frozen_source_state(tensors)) != (
        contract.source_v36_frozen_state_sha256
    ):
        raise ValueError("V37 V36 inherited frozen-state provenance changed")
    if tensor_state_sha256(_v37_frozen_complement(tensors)) != (
        contract.frozen_complement_state_sha256
    ):
        raise ValueError("V37 frozen complement source changed")
    bank_hashes = _mapping(metadata.get("lora_bank_state_sha256"), "source bank hashes")
    if (
        metadata.get("optimizer_step") != 16
        or metadata.get("block_cross_residual_state_sha256") != contract.source_core_state_sha256
        or bank_hashes.get(_QUERY_BANK) != contract.source_query_state_sha256
        or bank_hashes.get(_TARGET_BANK) != contract.target_source_state_sha256
    ):
        raise ValueError("V37 source metadata state hashes changed")
    history = metadata.get("history")
    if not isinstance(history, list) or len(history) != 17:
        raise ValueError("V37 source does not contain the complete V36 update-16 history")
    source_row = _mapping(history[-1], "V36 update-16 row")
    if (
        source_row.get("optimizer_update") != 16
        or source_row.get("saved_checkpoint") is not True
        or source_row.get("validation_qa_loaded") is not False
        or float(source_row.get("training_broad_nll")) != _SOURCE_BROAD_NLL
    ):
        raise ValueError("V37 source train-only replay row changed")
    audit = {
        "source_checkpoint": str(source),
        "source_file_sha256_verified_without_optimizer": observed,
        "source_optimizer_expected_sha256_from_terminal": contract.source_file_sha256[
            "optimizer.pt"
        ],
        "source_optimizer_file_present": True,
        "source_optimizer_file_opened": False,
        "source_optimizer_state_loaded": False,
        "source_tensor_state_sha256": tensor_state_sha256(tensors),
        "source_core_state_sha256": tensor_state_sha256(core),
        "source_query_bank_state_sha256": tensor_state_sha256(query),
        "source_target_bank_state_sha256": tensor_state_sha256(target),
        "frozen_complement_state_sha256": tensor_state_sha256(_v37_frozen_complement(tensors)),
        "conditional_authorization_sha256": terminal["sha256"],
    }
    return source, metadata, audit


def _pair_family(unit: CounterfactualPairUnit) -> str:
    return next(
        (family for family, pair_id in _PAIR_FAMILIES.items() if pair_id == unit.pair_id),
        "other",
    )


def build_v37_schedule(
    records: Sequence[QARecord],
    pair_units: Sequence[CounterfactualPairUnit],
    *,
    seed: int,
) -> tuple[list[V35Microstep], dict[str, Any]]:
    if len(pair_units) != 25:
        raise ValueError("V37 requires the exact 25 changed training units")
    canonical = sorted(pair_units, key=lambda unit: (unit.pair_id, unit.question_key))
    by_key = {unit.question_key: unit for unit in canonical}
    if len(by_key) != 25 or any(key not in by_key for key in _WARMUP_KEYS):
        raise ValueError("V37 priority unit inventory changed")
    warmup = [by_key[key] for key in _WARMUP_KEYS]
    tail = [by_key[key] for key in _TAIL_KEYS]
    if [_pair_family(unit) for unit in warmup] != [
        "book_support",
        "picture_support",
    ] * 4:
        raise ValueError("V37 warmup no longer alternates sorted book/picture units")
    if [_pair_family(unit) for unit in tail] != [
        "book_support",
        "picture_support",
    ] * 3:
        raise ValueError("V37 tail no longer alternates sorted book/picture units")
    scheduled = [*warmup, *canonical, *canonical, *tail]
    if len(scheduled) != 64:
        raise RuntimeError("V37 schedule is not exactly 64 true updates")
    broad = select_balanced_broad_records(
        records,
        count=64,
        seed=seed,
        exclude_expected_change=True,
    )
    steps = [V35Microstep(index + 1, broad[index], scheduled[index]) for index in range(64)]
    payload = [
        {
            "optimizer_step": row.optimizer_step,
            "broad": (row.broad_record.scene_id, row.broad_record.question_id),
            "pair": (row.pair_unit.pair_id, row.pair_unit.question_key),
        }
        for row in steps
    ]
    schedule_hash = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    appearances = Counter((unit.pair_id, unit.question_key) for unit in scheduled)
    family_counts = Counter(_pair_family(unit) for unit in scheduled)
    return steps, {
        "schema_version": 1,
        "optimizer_step_count": 64,
        "true_optimizer_step_per_schedule_row": True,
        "one_unchanged_broad_row_per_update": True,
        "broad_expected_change_excluded": True,
        "broad_answer_type_counts": dict(
            sorted(Counter(record.answer_type for record in broad).items())
        ),
        "pair_units_atomic": True,
        "pair_unit_count": 25,
        "warmup_updates": [1, 8],
        "warmup_question_keys": list(_WARMUP_KEYS),
        "complete_cycle_updates": [9, 58],
        "complete_deterministic_cycle_count": 2,
        "tail_updates": [59, 64],
        "tail_question_keys": list(_TAIL_KEYS),
        "appearance_counts_by_unit": {
            f"{pair_id}:{question_key}": count
            for (pair_id, question_key), count in sorted(appearances.items())
        },
        "appearance_counts_by_family": dict(sorted(family_counts.items())),
        "schedule_sha256": schedule_hash,
        "questions_or_answers_serialized_to_runtime": False,
    }


def augment_pair_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(metrics))
    cross = {family: 0 for family in _PAIR_FAMILIES}
    rows = result.get("units")
    if not isinstance(rows, list) or len(rows) != 25:
        raise ValueError("V37 pair metrics do not contain all 25 units")
    for row in rows:
        family = str(_mapping(row, "pair metric row").get("family"))
        if family in cross and row.get("cross_prefix_complete") is True:
            cross[family] += 1
    result["cross_prefix_complete_units_by_family"] = cross
    result["complete_physical_pair_coverage"] = complete_physical_pair_coverage(result)
    return result


def _family_counts(metrics: Mapping[str, Any]) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    complete = _mapping(metrics.get("complete_units_by_family"), "complete family counts")
    cross = _mapping(metrics.get("cross_prefix_complete_units_by_family"), "cross family counts")
    return complete, cross


def v37_update16_gate(
    *,
    pair_metrics: Mapping[str, Any],
    broad_nll: float,
    target_bank_state_sha256: str,
    frozen_complement_state_sha256: str,
    residual_exact: bool,
    contract: V37Contract,
) -> dict[str, Any]:
    rule = contract.update16_gate
    complete, cross = _family_counts(pair_metrics)
    checks = {
        "teacher_complete_units_at_least_10": int(pair_metrics["complete_units"])
        >= int(rule["complete_units_minimum"]),
        "complete_physical_pair_coverage_at_least_5": int(
            pair_metrics["complete_physical_pair_coverage"]
        )
        >= int(rule["complete_physical_pair_coverage_minimum"]),
        "teacher_cross_complete_units_at_least_16": int(pair_metrics["cross_prefix_complete_units"])
        >= int(rule["cross_prefix_complete_units_minimum"]),
        "teacher_positive_sides_at_least_35": int(pair_metrics["positive_sides"])
        >= int(rule["positive_sides_minimum"]),
        "mean_cross_prefix_margin_at_least_source": float(pair_metrics["mean_cross_prefix_margin"])
        >= float(rule["mean_cross_prefix_margin_minimum"]),
        "book_or_picture_teacher_complete": int(complete.get("book_support", 0))
        + int(complete.get("picture_support", 0))
        >= int(rule["book_and_picture_complete_units_minimum"]),
        "book_cross_prefix_complete": int(cross.get("book_support", 0))
        >= int(rule["book_cross_prefix_complete_units_minimum"]),
        "picture_cross_prefix_complete_at_least_2": int(cross.get("picture_support", 0))
        >= int(rule["picture_cross_prefix_complete_units_minimum"]),
        "mirror_teacher_complete_at_least_2": int(complete.get("mirror_lr", 0))
        >= int(rule["mirror_complete_units_minimum"]),
        "broad_nll_within_absolute_lock": broad_nll <= float(rule["broad_nll_maximum"]),
        "target_bank_state_changed": target_bank_state_sha256
        != contract.target_source_state_sha256,
        "frozen_complement_state_exact": frozen_complement_state_sha256
        == contract.frozen_complement_state_sha256,
        "scene_prefix_and_block_residual_exact": residual_exact,
    }
    return {
        **checks,
        "passed": all(checks.values()),
        "training_scenes_only": True,
        "validation_qa_loaded": False,
    }


def v37_update32_gate(
    *,
    update16_gate: Mapping[str, Any],
    pair_metrics: Mapping[str, Any],
    broad_nll: float,
    frozen_complement_state_sha256: str,
    residual_exact: bool,
    contract: V37Contract,
) -> dict[str, Any]:
    rule = contract.update32_gate
    complete, _cross = _family_counts(pair_metrics)
    checks = {
        "update16_train_only_gate_remains_passed": update16_gate.get("passed") is True,
        "teacher_complete_units_at_least_12": int(pair_metrics["complete_units"])
        >= int(rule["complete_units_minimum"]),
        "complete_physical_pair_coverage_at_least_6": int(
            pair_metrics["complete_physical_pair_coverage"]
        )
        >= int(rule["complete_physical_pair_coverage_minimum"]),
        "teacher_cross_complete_units_at_least_18": int(pair_metrics["cross_prefix_complete_units"])
        >= int(rule["cross_prefix_complete_units_minimum"]),
        "teacher_positive_sides_at_least_37": int(pair_metrics["positive_sides"])
        >= int(rule["positive_sides_minimum"]),
        "mean_cross_prefix_margin_at_least_source": float(pair_metrics["mean_cross_prefix_margin"])
        >= float(rule["mean_cross_prefix_margin_minimum"]),
        "book_teacher_complete": int(complete.get("book_support", 0))
        >= int(rule["book_complete_units_minimum"]),
        "picture_teacher_complete": int(complete.get("picture_support", 0))
        >= int(rule["picture_complete_units_minimum"]),
        "mirror_teacher_complete_at_least_2": int(complete.get("mirror_lr", 0))
        >= int(rule["mirror_complete_units_minimum"]),
        "broad_nll_within_absolute_lock": broad_nll <= float(rule["broad_nll_maximum"]),
        "frozen_complement_state_exact": frozen_complement_state_sha256
        == contract.frozen_complement_state_sha256,
        "scene_prefix_and_block_residual_exact": residual_exact,
    }
    return {
        **checks,
        "passed": all(checks.values()),
        "training_scenes_only": True,
        "validation_qa_loaded": False,
    }


def v37_update64_gate(
    *,
    update32_gate: Mapping[str, Any],
    pair_metrics: Mapping[str, Any],
    greedy_metrics: Mapping[str, Any],
    source_greedy_metrics: Mapping[str, Any],
    broad_nll: float,
    frozen_complement_state_sha256: str,
    residual_exact: bool,
    contract: V37Contract,
) -> dict[str, Any]:
    rule = contract.update64_gate
    complete, _cross = _family_counts(pair_metrics)
    greedy_family = _mapping(greedy_metrics.get("complete_units_by_family"), "greedy family counts")
    checks = {
        "update32_train_only_gate_remains_passed": update32_gate.get("passed") is True,
        "teacher_complete_units_at_least_15": int(pair_metrics["complete_units"])
        >= int(rule["complete_units_minimum"]),
        "complete_physical_pair_coverage_at_least_7": int(
            pair_metrics["complete_physical_pair_coverage"]
        )
        >= int(rule["complete_physical_pair_coverage_minimum"]),
        "teacher_cross_complete_units_at_least_20": int(pair_metrics["cross_prefix_complete_units"])
        >= int(rule["cross_prefix_complete_units_minimum"]),
        "teacher_positive_sides_at_least_40": int(pair_metrics["positive_sides"])
        >= int(rule["positive_sides_minimum"]),
        "teacher_book_complete": int(complete.get("book_support", 0))
        >= int(rule["book_complete_units_minimum"]),
        "teacher_picture_complete": int(complete.get("picture_support", 0))
        >= int(rule["picture_complete_units_minimum"]),
        "teacher_mirror_complete_at_least_2": int(complete.get("mirror_lr", 0))
        >= int(rule["mirror_complete_units_minimum"]),
        "train_greedy_complete_units_at_least_6": int(greedy_metrics["complete_units"])
        >= int(rule["greedy_complete_units_minimum"]),
        "train_greedy_each_priority_family": all(
            int(greedy_family.get(family, 0)) >= 1 for family in _PAIR_FAMILIES
        ),
        "broad_greedy_accuracy_within_source_minus_0_02": float(
            greedy_metrics["broad_exact_accuracy"]
        )
        >= float(source_greedy_metrics["broad_exact_accuracy"])
        - float(rule["broad_greedy_exact_accuracy_maximum_drop"]),
        "broad_nll_within_absolute_lock": broad_nll <= float(rule["broad_nll_maximum"]),
        "frozen_complement_state_exact": frozen_complement_state_sha256
        == contract.frozen_complement_state_sha256,
        "scene_prefix_and_block_residual_exact": residual_exact,
    }
    return {
        **checks,
        "passed": all(checks.values()),
        "training_scenes_only": True,
        "validation_qa_loaded": False,
    }


def _target_bank(bundle: V30Bundle) -> LoRAInstallation:
    collection = bundle.lora_installation
    if collection is None:
        raise RuntimeError("V37 requires installed LoRA banks")
    bank = collection.bank(_TARGET_BANK).installation
    if bank.parameter_count != 30_720:
        raise RuntimeError("V37 target-bank parameter count changed")
    if bank.target_names != _TARGET_MODULES:
        raise RuntimeError("V37 target-bank Gemma paths changed")
    if not (
        bank.settings.rank == 4 and bank.settings.alpha == 8.0 and bank.settings.dropout == 0.0
    ):
        raise RuntimeError("V37 target-bank LoRA architecture changed")
    state = bank.state_module.state_dict()
    if tuple(f"{_TARGET_PREFIX}{name}" for name in state) != _TARGET_PARAMETER_NAMES:
        raise RuntimeError("V37 target-bank tensor order changed")
    expected_shapes = (
        (4, 1536),
        (256, 4),
        (4, 1536),
        (256, 4),
        (4, 1536),
        (512, 4),
        (4, 1536),
        (512, 4),
    )
    if tuple(tuple(value.shape) for value in state.values()) != expected_shapes:
        raise RuntimeError("V37 target-bank tensor shapes changed")
    return bank


def _query_bank(bundle: V30Bundle) -> LoRAInstallation:
    collection = bundle.lora_installation
    if collection is None:
        raise RuntimeError("V37 requires installed LoRA banks")
    return collection.bank(_QUERY_BANK).installation


def frozen_v37_state_sha256(bundle: V30Bundle) -> str:
    state = {
        f"{module_name}.{name}": value
        for module_name, module in bundle.checkpoint_modules.items()
        if module_name != f"lora_banks.{_TARGET_BANK}"
        for name, value in module.state_dict().items()
    }
    return tensor_state_sha256(state)


def freeze_for_v37(bundle: V30Bundle) -> list[torch.nn.Parameter]:
    bundle.language.model.requires_grad_(False).eval()
    for module in bundle.checkpoint_modules.values():
        module.requires_grad_(False).eval()
    target = _target_bank(bundle)
    target.state_module.requires_grad_(True)
    target.train(True)
    return target.parameters()


def assert_v37_trainable_surface(
    bundle: V30Bundle,
    *,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, Any]:
    target = _target_bank(bundle)
    parameters = target.parameters()
    expected_ids = {id(parameter) for parameter in parameters}
    observed_ids = {
        id(parameter)
        for module in bundle.checkpoint_modules.values()
        for parameter in module.parameters()
        if parameter.requires_grad
    }
    if observed_ids != expected_ids:
        raise RuntimeError("V37 active trainable surface differs from its exact lock")
    language_trainable = {
        id(parameter) for parameter in bundle.language.model.parameters() if parameter.requires_grad
    }
    if language_trainable != expected_ids:
        raise RuntimeError("V37 Gemma trainability differs from the target LoRA tensors")
    if optimizer is not None:
        optimizer_ids = {
            id(parameter) for group in optimizer.param_groups for parameter in group["params"]
        }
        if optimizer_ids != expected_ids:
            raise RuntimeError("V37 optimizer contains an unauthorized tensor")
    if len(parameters) != 8 or sum(parameter.numel() for parameter in parameters) != 30_720:
        raise RuntimeError("V37 trainable tensor count changed")
    return {
        "target_bank": _TARGET_BANK,
        "target_module_paths": list(_TARGET_MODULES),
        "target_parameter_names": list(_TARGET_PARAMETER_NAMES),
        "trainable_tensor_count": 8,
        "trainable_parameter_count": 30_720,
        "rank": 4,
        "alpha": 8.0,
        "dropout": 0.0,
        "existing_learned_bank_continued_without_reinitialization": True,
        "gemma_base_frozen": True,
        "v36_learned_block_core_frozen": True,
        "v36_learned_query_bank_frozen": True,
        "complete_scene_stack_frozen": True,
        "all_other_lora_banks_frozen": True,
        "every_other_tensor_and_buffer_frozen": True,
    }


def v37_optimizer(bundle: V30Bundle, settings: V37Settings) -> torch.optim.AdamW:
    parameters = freeze_for_v37(bundle)
    optimizer = torch.optim.AdamW(
        [
            {
                "name": f"lora_banks.{_TARGET_BANK}",
                "params": parameters,
                "parameter_names": list(_TARGET_PARAMETER_NAMES),
                "lr": settings.learning_rate,
                "weight_decay": settings.weight_decay,
            }
        ]
    )
    assert_v37_trainable_surface(bundle, optimizer=optimizer)
    if optimizer.state:
        raise RuntimeError("V37 Adam state is not fresh")
    return optimizer


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


def _fresh_adamw_group_defaults() -> dict[str, Any]:
    probe = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.AdamW(
        [
            {
                "name": f"lora_banks.{_TARGET_BANK}",
                "params": [probe],
                "parameter_names": ["probe"],
                "lr": 2e-5,
                "weight_decay": 0.0,
            }
        ]
    )
    group = dict(optimizer.state_dict()["param_groups"][0])
    for key in ("name", "params", "parameter_names"):
        group.pop(key)
    return group


def _optimizer_payload_audit(
    payload: Mapping[str, Any],
    *,
    expected_step: int,
    tensors: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("V37 optimizer checkpoint must be a mapping")
    groups = payload.get("param_groups")
    state = payload.get("state")
    if not isinstance(groups, list) or len(groups) != 1 or not isinstance(state, Mapping):
        raise ValueError("V37 optimizer must contain exactly one AdamW group")
    group = _mapping(groups[0], "V37 optimizer group")
    expected_defaults = _fresh_adamw_group_defaults()
    observed_defaults = {
        key: value
        for key, value in group.items()
        if key not in {"name", "params", "parameter_names"}
    }
    if (
        group.get("name") != f"lora_banks.{_TARGET_BANK}"
        or group.get("params") != list(range(8))
        or group.get("parameter_names") != list(_TARGET_PARAMETER_NAMES)
        or set(group) != {"name", "params", "parameter_names", *expected_defaults}
        or observed_defaults != expected_defaults
    ):
        raise ValueError("V37 optimizer group identity/order/settings changed")
    if set(state) != set(range(8)):
        raise ValueError("V37 Adam state contains a missing or unauthorized parameter")
    for parameter_id, tensor_name in enumerate(_TARGET_PARAMETER_NAMES):
        tensor = tensors.get(tensor_name)
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"V37 adapter tensor is absent or invalid: {tensor_name}")
        entry = state.get(parameter_id)
        if not isinstance(entry, Mapping) or set(entry) != {
            "step",
            "exp_avg",
            "exp_avg_sq",
        }:
            raise ValueError(f"V37 Adam state is incomplete for {tensor_name}")
        if _adam_step(entry["step"], f"Adam step for {tensor_name}") != expected_step:
            raise ValueError(f"V37 Adam step changed for {tensor_name}")
        for moment_name in ("exp_avg", "exp_avg_sq"):
            moment = entry[moment_name]
            if (
                not isinstance(moment, torch.Tensor)
                or tuple(moment.shape) != tuple(tensor.shape)
                or not torch.isfinite(moment).all()
            ):
                raise ValueError(f"V37 Adam {moment_name} is invalid for {tensor_name}")
    return {
        "group_count": 1,
        "parameter_states_inspected": list(_TARGET_PARAMETER_NAMES),
        "moment_tensor_count": 16,
        "optimizer_step": expected_step,
        "exact_parameter_order_verified": True,
        "exact_adamw_group_schema_verified": True,
        "adamw_group_defaults": expected_defaults,
        "fresh_v37_adam_verified": True,
    }


def optimizer_step_audit(
    checkpoint: Path,
    *,
    expected_step: int,
    tensors: Mapping[str, torch.Tensor] | None = None,
) -> dict[str, Any]:
    if tensors is None:
        tensors = load_file(checkpoint / "adapter.safetensors", device="cpu")
    optimizer_path = checkpoint / "optimizer.pt"
    manifest_path = checkpoint / OPTIMIZER_AUDIT_FILENAME
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise FileNotFoundError("V37 optimizer integrity manifest is missing or aliased")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_manifest = {
        "schema_version": 1,
        "artifact": "v37_optimizer_integrity_manifest",
        "optimizer_step": expected_step,
        "optimizer_filename": "optimizer.pt",
        "optimizer_sha256": _sha256(optimizer_path),
    }
    if manifest != expected_manifest:
        raise ValueError("V37 optimizer integrity manifest or file hash changed")
    payload = torch.load(optimizer_path, map_location="cpu", weights_only=True)
    audit = _optimizer_payload_audit(
        payload,
        expected_step=expected_step,
        tensors=tensors,
    )
    return {
        **audit,
        "optimizer_sha256": expected_manifest["optimizer_sha256"],
        "self_hash_linkage_verified": True,
    }


def preflight_v37(config: Mapping[str, Any]) -> dict[str, Any]:
    contract = v37_contract(config)
    terminal = require_v36_terminal_gate(config)
    _source, _metadata, source_audit = require_exact_v36_source(config)
    loader_config = v37_loader_config(config)
    assert_deferred_final_scenes_absent(loader_config)
    records, qa_audit = load_v35_train_qa_records(loader_config)
    units = build_exact_question_pair_units(records)
    _schedule, schedule_audit = build_v37_schedule(records, units, seed=int(config["seed"]))
    if schedule_audit["schedule_sha256"] != contract.schedule_sha256:
        raise ValueError("V37 generated schedule differs from its pinned hash")
    return {
        "schema_version": 1,
        "artifact": "v37_scene_ingress_kv_preflight",
        "passed": True,
        "source_checkpoint": str(contract.source_checkpoint),
        "source_optimizer_step": 16,
        "source_v36_tensor_state_sha256": contract.source_tensor_state_sha256,
        "source_block_core_state_sha256": contract.source_core_state_sha256,
        "source_query_bank_state_sha256": contract.source_query_state_sha256,
        "source_target_bank_state_sha256": contract.target_source_state_sha256,
        "frozen_complement_state_sha256": contract.frozen_complement_state_sha256,
        "exact_trainable_tensor_count": 8,
        "exact_trainable_parameter_count": 30_720,
        "target_modules": list(_TARGET_MODULES),
        "schedule": schedule_audit,
        "saved_optimizer_steps": list(contract.saved_optimizer_steps),
        "qa_audit": qa_audit,
        "terminal_gate": {"path": terminal["path"], "sha256": terminal["sha256"]},
        "source_audit": source_audit,
        "source_optimizer_file_opened": False,
        "source_optimizer_state_loaded": False,
        "gemma_loaded": False,
        "scene_maps_loaded": False,
        "validation_qa_loaded": False,
        "oracle_loaded": False,
        "final_test_scenes_touched": False,
    }


def v37_loader_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return a V30-compatible construction copy; never use it for metadata."""

    loader = copy.deepcopy(dict(config))
    banks = _mapping(_mapping(loader["language"], "language")["lora_banks"], "lora banks")
    _mapping(banks[_TARGET_BANK], _TARGET_BANK)["trainable"] = False
    query = _mapping(banks[_QUERY_BANK], _QUERY_BANK)
    query.update(
        {
            "trainable": True,
            "initialization_algorithm": "cpu_kaiming_uniform_a_exact_zero_b",
            "initialization_seed": 30030,
            "expected_initial_state_sha256": _QUERY_CONSTRUCTION_STATE_SHA256,
        }
    )
    training = _mapping(loader["training"], "loader training")
    training["lora_learning_rate"] = 0.0002
    training["lora_weight_decay"] = 0.0
    query_bank = lora_banks_settings(loader).bank(_QUERY_BANK)
    if not (
        query_bank.trainable is True
        and query_bank.adapter.rank == 8
        and query_bank.adapter.alpha == 16.0
        and query_bank.adapter.dropout == 0.0
        and query_bank.adapter.target_modules == _QUERY_MODULES
        and query_bank.initialization_algorithm == "cpu_kaiming_uniform_a_exact_zero_b"
        and query_bank.initialization_seed == 30030
        and query_bank.expected_initial_state_sha256 == _QUERY_CONSTRUCTION_STATE_SHA256
    ):
        raise RuntimeError("V37 construction-copy query-bank contract changed")
    return loader


def retag_bundle_for_v37(bundle: V30Bundle, config: Mapping[str, Any]) -> dict[str, Any]:
    """Retag the loaded collection without changing a single parameter value."""

    collection = bundle.lora_installation
    if collection is None:
        raise RuntimeError("V37 requires a multi-bank LoRA collection")
    before_names = collection.bank_names
    before_targets = collection.wrapped_modules
    before_hashes = collection.state_sha256()
    actual = lora_banks_settings(config)
    old_by_name = {bank.settings.name: bank for bank in collection.banks}
    if tuple(bank.name for bank in actual.banks) != before_names:
        raise RuntimeError("V37 loader-to-runtime bank inventory changed")
    collection.settings = actual
    collection.banks = tuple(
        InstalledLoRABank(
            settings=setting,
            installation=old_by_name[setting.name].installation,
        )
        for setting in actual.banks
    )
    after_hashes = collection.state_sha256()
    if (
        collection.bank_names != before_names
        or collection.wrapped_modules != before_targets
        or after_hashes != before_hashes
    ):
        raise RuntimeError("V37 loader retag changed bank identity, paths, or tensors")
    if collection.trainable_parameter_count != 30_720:
        raise RuntimeError("V37 retag did not authorize exactly 30,720 parameters")
    bundle.config = copy.deepcopy(dict(config))
    bundle.trainable_bank_name = _TARGET_BANK
    return {
        "construction_used_v36_compatible_trainability_copy": True,
        "construction_copy_serialized_to_metadata": False,
        "bank_names_bit_exact": True,
        "target_paths_bit_exact": True,
        "state_hashes_bit_exact": True,
        "v37_trainable_bank": _TARGET_BANK,
        "v37_frozen_query_bank": _QUERY_BANK,
        "v37_trainable_parameter_count": collection.trainable_parameter_count,
    }


def load_v37_bundle(
    config: dict[str, Any],
    approved_v29: ApprovedV29Source,
    source_checkpoint: Path,
) -> tuple[V30Bundle, BlockCrossResidual, dict[str, Any], dict[str, Any]]:
    """Construct through V36's audited path, load exact source, then retag."""

    loader_config = v37_loader_config(config)
    bundle = load_v30_bundle(loader_config, approved_v29)
    block_core = construct_v36_source_core(loader_config, device=bundle.language.device)
    bundle.checkpoint_modules["block_cross_residual"] = block_core
    source_metadata = load_adapter_checkpoint(
        source_checkpoint,
        bundle.checkpoint_modules,
        device="cpu",
        metadata_filename=TRAINING_METADATA_FILENAME,
    )
    if module_collection_state_sha256(bundle.checkpoint_modules) != (_SOURCE_TENSOR_STATE_SHA256):
        raise RuntimeError("V37 constructed bundle did not load exact V36 update 16")
    transition = retag_bundle_for_v37(bundle, config)
    freeze_for_v37(bundle)
    assert_v37_trainable_surface(bundle)
    return bundle, block_core, source_metadata, transition


def construction_preflight_v37(config: dict[str, Any]) -> dict[str, Any]:
    """Load real local Gemma and the exact V36 adapter without maps or Adam."""

    contract = v37_contract(config)
    source, pinned_metadata, source_audit = require_exact_v36_source(config)
    loader_config = v37_loader_config(config)
    approved_v29 = require_approved_v29_source(loader_config)
    bundle, block_core, loaded_metadata, transition = load_v37_bundle(
        config, approved_v29, source
    )
    if loaded_metadata != pinned_metadata:
        raise RuntimeError("V37 construction preflight loaded changed source metadata")
    v37_runtime_metadata = copy.deepcopy(dict(loaded_metadata))
    v37_runtime_metadata.update(bundle.lora_installation.checkpoint_metadata())
    validate_lora_banks_checkpoint_state(v37_runtime_metadata, bundle.lora_installation)
    surface = assert_v37_trainable_surface(bundle)
    if (
        block_core.state_sha256() != contract.source_core_state_sha256
        or _query_bank(bundle).state_sha256() != contract.source_query_state_sha256
        or _target_bank(bundle).state_sha256() != contract.target_source_state_sha256
        or frozen_v37_state_sha256(bundle) != contract.frozen_complement_state_sha256
    ):
        raise RuntimeError("V37 construction preflight changed an exact source tensor")
    return {
        "schema_version": 1,
        "artifact": "v37_real_gemma_construction_preflight",
        "passed": True,
        "model_id": str(config["language"]["model_id"]),
        "device": str(bundle.language.device),
        "source_checkpoint": str(source),
        "source_tensor_state_sha256": contract.source_tensor_state_sha256,
        "source_core_state_sha256": contract.source_core_state_sha256,
        "source_query_bank_state_sha256": contract.source_query_state_sha256,
        "source_target_bank_state_sha256": contract.target_source_state_sha256,
        "frozen_complement_state_sha256": contract.frozen_complement_state_sha256,
        "v36_source_metadata_retagged_for_v37_runtime": True,
        "runtime_checkpoint_state_validation_passed": True,
        "trainable_surface": surface,
        "loader_transition": transition,
        "source_audit": source_audit,
        "source_optimizer_file_opened": False,
        "source_optimizer_state_loaded": False,
        "scene_maps_loaded": False,
        "validation_qa_loaded": False,
        "oracle_loaded": False,
        "final_test_scenes_touched": False,
    }


def _dynamic_block_source_stack_sha256(bundle: V30Bundle) -> str:
    state = {
        f"{module_name}.{name}": value
        for module_name, module in bundle.checkpoint_modules.items()
        if module_name != "block_cross_residual"
        for name, value in module.state_dict().items()
    }
    return tensor_state_sha256(state)


def _v37_prefix_replay_attestation(
    *,
    caches: Mapping[str, V35SceneCache],
    block_cross_residual: BlockCrossResidual,
    bundle: V30Bundle,
    expected_scene_ids: Sequence[str],
) -> dict[str, Any]:
    """Replay exactly the 16 training prefixes without opening validation maps."""

    model_dtype = next(bundle.language.model.parameters()).dtype
    first: dict[str, str] = {}
    repeated: dict[str, str] = {}
    with torch.inference_mode():
        for scene_id, cache in sorted(caches.items()):
            current = current_scene_tokens(
                cache, block_cross_residual, device=bundle.language.device
            )
            first[scene_id] = prefix_sha256(
                bundle.composer.scene_prefix(current.to(model_dtype))
            )
            replay = current_scene_tokens(
                cache, block_cross_residual, device=bundle.language.device
            )
            repeated[scene_id] = prefix_sha256(
                bundle.composer.scene_prefix(replay.to(model_dtype))
            )
    expected = tuple(sorted(set(expected_scene_ids)))
    if len(expected) != 16 or tuple(sorted(first)) != expected or first != repeated:
        raise RuntimeError("V37 train-only prefix replay is incomplete or nondeterministic")
    return {
        "source_prefix_scene_count": 16,
        "source_prefix_scene_ids": list(expected),
        "source_prefix_sha256_by_scene": first,
        "replayed_prefix_sha256_by_scene": repeated,
        "source_prefixes_replayed_bit_exact": True,
        "current_v36_u16_prefixes_recomputed_deterministically_from_exact_tensors": True,
        "external_prefix_manifest_used": False,
        "scene_prefixes_built_before_questions": True,
        "training_scene_prefixes_question_free": True,
        "validation_environment_maps_loaded": False,
        "validation_qa_loaded": False,
        "question_dependent_scene_processing": False,
        "question_dependent_retrieval": False,
    }


def validate_v37_training_cache_boundary(
    *,
    cache_audit: Mapping[str, Any],
    caches: Mapping[str, V35SceneCache],
    config: Mapping[str, Any],
    train_scene_ids: Sequence[str],
    validation_scene_ids: Sequence[str],
) -> dict[str, Any]:
    train = tuple(sorted(set(train_scene_ids)))
    validation = tuple(sorted(set(validation_scene_ids)))
    expected_files = [
        str((artifact_root(dict(config), "maps") / scene_id / "voxel_map.npz").resolve())
        for scene_id in train
    ]
    loaded = cache_audit.get("loaded_environment_files")
    if (
        len(train) != 16
        or len(validation) != 6
        or set(train).intersection(validation)
        or cache_audit.get("scene_count") != len(train)
        or tuple(cache_audit.get("scene_ids", ())) != train
        or cache_audit.get("scene_scope") != "training_only"
        or cache_audit.get("authenticated_manifest_scene_count") != 22
        or cache_audit.get("authenticated_manifest_train_subset_count") != 16
        or cache_audit.get("validation_scene_ids_loaded") != []
        or cache_audit.get("validation_environment_maps_loaded") is not False
        or cache_audit.get("deferred_final_scene_ids_loaded") != []
        or tuple(sorted(caches)) != train
        or loaded != expected_files
        or any(scene_id in str(path) for path in (loaded or ()) for scene_id in validation)
    ):
        raise RuntimeError("V37 training cache crossed the exact train-map boundary")
    return {
        "exact_train_scene_count": len(train),
        "exact_train_scene_ids": list(train),
        "loaded_environment_files": expected_files,
        "validation_environment_maps_loaded": False,
        "oracle_environment_files_loaded": False,
    }


def v37_loss_values(
    *,
    settings: V37Settings,
    broad_nll: float,
    pair_correct_nll: float,
    side_hinge: float,
    cross_prefix_hinge: float,
    frozen_normalized_residual: float,
) -> tuple[float, float]:
    """Return the optimized loss and separately reported frozen composite."""

    optimized = (
        settings.broad_nll_weight * broad_nll
        + settings.pair_correct_nll_weight * pair_correct_nll
        + settings.side_hinge_weight * side_hinge
        + settings.cross_prefix_flip_weight * cross_prefix_hinge
    )
    reported = optimized + settings.residual_penalty_weight * frozen_normalized_residual
    return optimized, reported


def _source_replay_attestation(
    *,
    source_metadata: Mapping[str, Any],
    pair_metrics: Mapping[str, Any],
    broad_nll: float,
    residual: Mapping[str, Any],
    prefix_replay: Mapping[str, Any],
    expected_scene_ids: Sequence[str],
) -> dict[str, Any]:
    history = source_metadata.get("history")
    if not isinstance(history, list) or len(history) != 17:
        raise ValueError("V37 source history changed")
    row = _mapping(history[-1], "V36 update-16 source row")
    expected_pair = _mapping(row.get("training_pair_metrics"), "V36 source pair metrics")
    comparable_pair = {
        key: value
        for key, value in pair_metrics.items()
        if key
        not in {
            "cross_prefix_complete_units_by_family",
            "complete_physical_pair_coverage",
        }
    }
    if comparable_pair != dict(expected_pair):
        raise ValueError("V37 update zero did not bit-replay V36 pair metrics")
    if broad_nll != float(row.get("training_broad_nll")) or broad_nll != _SOURCE_BROAD_NLL:
        raise ValueError("V37 update zero did not bit-replay V36 broad NLL")
    expected_residual = _mapping(row.get("training_residual_diagnostics"), "V36 source residual")
    if dict(residual) != dict(expected_residual):
        raise ValueError("V37 update zero did not bit-replay V36 residual diagnostics")
    expected = tuple(sorted(set(expected_scene_ids)))
    first_hashes = _mapping(
        prefix_replay.get("source_prefix_sha256_by_scene"), "V37 first prefix hashes"
    )
    repeated_hashes = _mapping(
        prefix_replay.get("replayed_prefix_sha256_by_scene"),
        "V37 repeated prefix hashes",
    )
    if (
        len(expected) != 16
        or tuple(prefix_replay.get("source_prefix_scene_ids", ())) != expected
        or prefix_replay.get("source_prefix_scene_count") != 16
        or tuple(sorted(first_hashes)) != expected
        or tuple(sorted(repeated_hashes)) != expected
        or dict(first_hashes) != dict(repeated_hashes)
        or any(
            re.fullmatch(r"[0-9a-f]{64}", str(value)) is None
            for value in first_hashes.values()
        )
        or prefix_replay.get("source_prefixes_replayed_bit_exact") is not True
        or prefix_replay.get(
            "current_v36_u16_prefixes_recomputed_deterministically_from_exact_tensors"
        )
        is not True
        or prefix_replay.get("external_prefix_manifest_used") is not False
    ):
        raise ValueError("V37 update-zero prefixes did not replay bit exactly")
    return {
        "exact_stopped_v36_update16_loaded": True,
        "source_optimizer_step": 16,
        "v36_optimizer_file_opened": False,
        "v36_optimizer_state_loaded": False,
        "fresh_adam_state": True,
        "source_pair_metrics_bit_exact": True,
        "source_broad_nll_bit_exact": True,
        "source_residual_diagnostics_bit_exact": True,
        "source_prefixes_replayed_bit_exact": True,
        "current_v36_u16_prefixes_recomputed_deterministically_from_exact_tensors": True,
        "external_prefix_manifest_used": False,
        "source_prefix_scene_ids": list(expected),
        "source_complete_units": int(pair_metrics["complete_units"]),
        "source_cross_prefix_complete_units": int(pair_metrics["cross_prefix_complete_units"]),
        "source_positive_sides": int(pair_metrics["positive_sides"]),
        "source_mean_cross_prefix_margin": float(pair_metrics["mean_cross_prefix_margin"]),
        "source_broad_train_nll": broad_nll,
        "source_residual_rms": float(residual["aggregate_rms"]),
        "validation_qa_loaded": False,
    }


def _metadata(
    *,
    source_metadata: Mapping[str, Any],
    config: Mapping[str, Any],
    terminal: Mapping[str, Any],
    source_audit: Mapping[str, Any],
    schedule_audit: Mapping[str, Any],
    cache_audit: Mapping[str, Any],
    qa_audit: Mapping[str, Any],
    source_replay: Mapping[str, Any],
    prefix_replay: Mapping[str, Any],
    source_pair_metrics: Mapping[str, Any],
    source_broad_nll: float,
    source_greedy_metrics: Mapping[str, Any],
    source_residual: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
    optimizer_step: int,
    bundle: V30Bundle,
    surface: Mapping[str, Any],
    gate16: Mapping[str, Any] | None,
    gate32: Mapping[str, Any] | None,
    gate64: Mapping[str, Any] | None,
) -> dict[str, Any]:
    contract = v37_contract(config)
    metadata = copy.deepcopy(dict(source_metadata))
    metadata.update(
        {
            "schema_version": 1,
            "config_hash": config_hash(dict(config)),
            "optimizer_step": optimizer_step,
            "epoch": optimizer_step,
            "history": [dict(row) for row in history],
            "block_cross_residual_state_sha256": contract.source_core_state_sha256,
            "frozen_block_cross_source_stack_state_sha256": (
                _dynamic_block_source_stack_sha256(bundle)
            ),
            "question_dependent_scene_processing": False,
            "lora": lora_banks_checkpoint_contract(
                lora_banks_settings(config),
                lora_banks_optimizer_settings(config, lora_banks_settings(config)),
                bundle.lora_installation.parameter_counts,
            ),
            **bundle.lora_installation.checkpoint_metadata(),
        }
    )
    metadata["v37_scene_ingress_kv"] = {
        "schema_version": 1,
        "optimizer_step": optimizer_step,
        "conditional_v36_terminal_gate": {
            "path": terminal["path"],
            "sha256": terminal["sha256"],
        },
        "conditional_authorization": dict(terminal["authorization"]),
        "source_checkpoint": str(contract.source_checkpoint),
        "source_file_sha256": dict(contract.source_file_sha256),
        "source_v36_tensor_state_sha256": contract.source_tensor_state_sha256,
        "source_block_core_state_sha256": contract.source_core_state_sha256,
        "source_query_bank_state_sha256": contract.source_query_state_sha256,
        "target_bank_source_state_sha256": contract.target_source_state_sha256,
        "source_v36_frozen_nonauthorized_state_sha256": (contract.source_v36_frozen_state_sha256),
        "frozen_complement_state_sha256": frozen_v37_state_sha256(bundle),
        "source_audit": dict(source_audit),
        "source_optimizer_state_loaded": False,
        "source_optimizer_file_opened": False,
        "fresh_adam": True,
        "trainable_surface": dict(surface),
        "target_bank_state_sha256": _target_bank(bundle).state_sha256(),
        "learned_query_bank_state_sha256": _query_bank(bundle).state_sha256(),
        "learned_block_core_state_sha256": contract.source_core_state_sha256,
        "dynamic_block_source_stack_state_sha256": (_dynamic_block_source_stack_sha256(bundle)),
        "schedule": dict(schedule_audit),
        "scene_cache": _deterministic_cache_audit(cache_audit),
        "train_qa_dataset": dict(qa_audit),
        "source_replay_attestation": dict(source_replay),
        "prefix_replay_attestation": dict(prefix_replay),
        "source_pair_metrics": dict(source_pair_metrics),
        "source_broad_train_nll": source_broad_nll,
        "source_train_greedy_metrics": dict(source_greedy_metrics),
        "source_residual_diagnostics": dict(source_residual),
        "update16_train_only_gate": None if gate16 is None else dict(gate16),
        "update32_train_only_gate": None if gate32 is None else dict(gate32),
        "update64_train_only_gate": None if gate64 is None else dict(gate64),
        "validation_qa_loaded": False,
        "oracle_environment_files_loaded": False,
        "deferred_final_scene_ids_loaded": [],
        "question_dependent_scene_processing": False,
        "question_dependent_retrieval": False,
        "independent_selector_required": True,
    }
    lora_contract = _mapping(metadata.get("lora"), "V37 lora contract")
    banks = lora_contract.get("banks")
    if not isinstance(banks, list):
        raise TypeError("V37 lora contract banks must be a list")
    trainable_by_name = {str(bank["name"]): bank.get("trainable") for bank in banks}
    if (
        trainable_by_name.get(_TARGET_BANK) is not True
        or trainable_by_name.get(_QUERY_BANK) is not False
        or metadata.get("lora_trainable_parameter_count") != 30_720
        or bundle.trainable_bank_name != _TARGET_BANK
    ):
        raise RuntimeError("V37 runtime metadata advertises the wrong LoRA surface")
    return metadata


def _save(
    path: Path,
    *,
    bundle: V30Bundle,
    metadata: dict[str, Any],
    optimizer: torch.optim.Optimizer | None,
) -> None:
    if path.is_symlink() or (path.exists() and not path.is_dir()):
        raise ValueError(f"V37 checkpoint destination is unsafe: {path}")
    save_adapter_checkpoint(path, bundle.checkpoint_modules, metadata)
    if optimizer is not None:
        save_optimizer_checkpoint(path, optimizer)
        optimizer_path = path / "optimizer.pt"
        manifest_path = path / OPTIMIZER_AUDIT_FILENAME
        payload = {
            "schema_version": 1,
            "artifact": "v37_optimizer_integrity_manifest",
            "optimizer_step": int(metadata["optimizer_step"]),
            "optimizer_filename": "optimizer.pt",
            "optimizer_sha256": _sha256(optimizer_path),
        }
        temporary = manifest_path.with_name(f".{manifest_path.name}.{os.getpid()}.tmp")
        try:
            temporary.write_text(
                json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, manifest_path)
        finally:
            temporary.unlink(missing_ok=True)
    runtime = json.loads((path / RUNTIME_METADATA_FILENAME).read_text(encoding="utf-8"))
    validate_runtime_checkpoint_metadata(runtime)
    if runtime != runtime_checkpoint_metadata(metadata):
        raise RuntimeError("V37 runtime metadata sanitizer changed during save")


def replay_v37_gates(
    metadata: Mapping[str, Any], contract: V37Contract
) -> tuple[Mapping[str, Any] | None, Mapping[str, Any] | None, Mapping[str, Any] | None]:
    history = metadata.get("history")
    if not isinstance(history, list) or not history:
        raise ValueError("V37 checkpoint history is absent")
    stage = _mapping(metadata.get("v37_scene_ingress_kv"), "V37 stage")
    source_greedy = _mapping(stage.get("source_train_greedy_metrics"), "V37 source greedy metrics")
    gate16: Mapping[str, Any] | None = None
    gate32: Mapping[str, Any] | None = None
    gate64: Mapping[str, Any] | None = None
    if len(history) > 16:
        row = _mapping(history[16], "V37 history[16]")
        gate16 = v37_update16_gate(
            pair_metrics=_mapping(row.get("training_pair_metrics"), "V37 u16 pairs"),
            broad_nll=float(row.get("training_broad_nll")),
            target_bank_state_sha256=str(row.get("target_bank_state_sha256")),
            frozen_complement_state_sha256=str(row.get("frozen_complement_state_sha256")),
            residual_exact=row.get("scene_prefix_and_block_residual_exact") is True,
            contract=contract,
        )
        if gate16 != row.get("update16_train_only_gate") or gate16 != stage.get(
            "update16_train_only_gate"
        ):
            raise ValueError("V37 independently replayed update-16 gate differs")
    if len(history) > 32:
        if gate16 is None:
            raise ValueError("V37 update-32 gate lacks update-16 evidence")
        row = _mapping(history[32], "V37 history[32]")
        gate32 = v37_update32_gate(
            update16_gate=gate16,
            pair_metrics=_mapping(row.get("training_pair_metrics"), "V37 u32 pairs"),
            broad_nll=float(row.get("training_broad_nll")),
            frozen_complement_state_sha256=str(row.get("frozen_complement_state_sha256")),
            residual_exact=row.get("scene_prefix_and_block_residual_exact") is True,
            contract=contract,
        )
        if gate32 != row.get("update32_train_only_gate") or gate32 != stage.get(
            "update32_train_only_gate"
        ):
            raise ValueError("V37 independently replayed update-32 gate differs")
    if len(history) > 64:
        if gate32 is None:
            raise ValueError("V37 update-64 gate lacks update-32 evidence")
        row = _mapping(history[64], "V37 history[64]")
        gate64 = v37_update64_gate(
            update32_gate=gate32,
            pair_metrics=_mapping(row.get("training_pair_metrics"), "V37 u64 pairs"),
            greedy_metrics=_mapping(row.get("training_greedy_metrics"), "V37 u64 greedy"),
            source_greedy_metrics=source_greedy,
            broad_nll=float(row.get("training_broad_nll")),
            frozen_complement_state_sha256=str(row.get("frozen_complement_state_sha256")),
            residual_exact=row.get("scene_prefix_and_block_residual_exact") is True,
            contract=contract,
        )
        if gate64 != row.get("update64_train_only_gate") or gate64 != stage.get(
            "update64_train_only_gate"
        ):
            raise ValueError("V37 independently replayed update-64 gate differs")
    return gate16, gate32, gate64


def latest_v37_resume_checkpoint(output: Path, contract: V37Contract) -> Path | None:
    if output.is_symlink() or (output.exists() and not output.is_dir()):
        raise ValueError("V37 output root must be a real directory")
    if not output.exists():
        return None
    parsed: dict[int, Path] = {}
    for path in output.glob("update_*"):
        match = _UPDATE_DIRECTORY.fullmatch(path.name)
        if match is None:
            raise ValueError(f"V37 output contains an unexpected arm: {path.name}")
        if path.is_symlink() or not path.is_dir():
            raise ValueError(f"V37 arm must be a real directory: {path}")
        parsed[int(match.group(1))] = path
    complete = [
        step
        for step in contract.saved_optimizer_steps
        if step in parsed
        and all(
            (parsed[step] / name).is_file() and not (parsed[step] / name).is_symlink()
            for name in (
                "adapter.safetensors",
                TRAINING_METADATA_FILENAME,
                RUNTIME_METADATA_FILENAME,
                *(("optimizer.pt", OPTIMIZER_AUDIT_FILENAME) if step else ()),
            )
        )
    ]
    if complete != list(contract.saved_optimizer_steps[: len(complete)]):
        raise ValueError("V37 complete arms are not a contiguous saved-step prefix")
    incomplete = sorted(set(parsed) - set(complete))
    if incomplete:
        raise ValueError(f"V37 output contains incomplete arms: {incomplete}")
    return None if not complete else parsed[complete[-1]]


def validate_v37_resume_checkpoint(
    *,
    config: Mapping[str, Any],
    output: Path,
    resume: Path,
    contract: V37Contract,
    terminal: Mapping[str, Any],
    schedule_audit: Mapping[str, Any],
    cache_audit: Mapping[str, Any],
    source_replay: Mapping[str, Any],
    prefix_replay: Mapping[str, Any],
    source_pair_metrics: Mapping[str, Any],
    source_broad_nll: float,
    source_greedy_metrics: Mapping[str, Any],
    source_residual: Mapping[str, Any],
) -> dict[str, Any]:
    if resume.parent != output or resume.is_symlink() or not resume.is_dir():
        raise ValueError("V37 resume must be a real numbered arm in its output root")
    if latest_v37_resume_checkpoint(output, contract) != resume:
        raise ValueError("V37 resume must be the latest contiguous complete arm")
    match = _UPDATE_DIRECTORY.fullmatch(resume.name)
    if match is None:
        raise ValueError("V37 resume path is not a numbered update arm")
    step = int(match.group(1))
    if step not in contract.saved_optimizer_steps:
        raise ValueError("V37 resume arm is outside the bounded envelope")
    metadata = json.loads((resume / TRAINING_METADATA_FILENAME).read_text(encoding="utf-8"))
    stage = _mapping(metadata.get("v37_scene_ingress_kv"), "V37 resume stage")
    if (
        metadata.get("config_hash") != config_hash(dict(config))
        or metadata.get("optimizer_step") != step
        or stage.get("optimizer_step") != step
    ):
        raise ValueError("V37 resume config or optimizer step changed")
    if (
        stage.get("conditional_v36_terminal_gate")
        != {
            "path": terminal["path"],
            "sha256": terminal["sha256"],
        }
        or stage.get("conditional_authorization") != terminal["authorization"]
    ):
        raise ValueError("V37 resume terminal authorization changed")
    static_expected = {
        "source_checkpoint": str(contract.source_checkpoint),
        "source_v36_tensor_state_sha256": contract.source_tensor_state_sha256,
        "source_block_core_state_sha256": contract.source_core_state_sha256,
        "source_query_bank_state_sha256": contract.source_query_state_sha256,
        "target_bank_source_state_sha256": contract.target_source_state_sha256,
        "source_optimizer_state_loaded": False,
        "source_optimizer_file_opened": False,
        "validation_qa_loaded": False,
        "oracle_environment_files_loaded": False,
        "deferred_final_scene_ids_loaded": [],
        "question_dependent_scene_processing": False,
        "question_dependent_retrieval": False,
    }
    if any(stage.get(key) != value for key, value in static_expected.items()):
        raise ValueError("V37 resume source/data boundary changed")
    if (
        stage.get("schedule") != dict(schedule_audit)
        or stage.get("scene_cache") != _deterministic_cache_audit(cache_audit)
        or stage.get("source_replay_attestation") != dict(source_replay)
        or stage.get("prefix_replay_attestation") != dict(prefix_replay)
        or stage.get("source_pair_metrics") != dict(source_pair_metrics)
        or float(stage.get("source_broad_train_nll")) != source_broad_nll
        or stage.get("source_train_greedy_metrics") != dict(source_greedy_metrics)
        or stage.get("source_residual_diagnostics") != dict(source_residual)
    ):
        raise ValueError("V37 resume deterministic source evidence changed")
    history = metadata.get("history")
    if not isinstance(history, list) or len(history) != step + 1:
        raise ValueError("V37 resume history is incomplete")
    if [row.get("optimizer_update") for row in history] != list(range(step + 1)):
        raise ValueError("V37 resume history is not one row per true optimizer step")
    if any(
        row.get("validation_qa_loaded") is not False
        or row.get("oracle_environment_files_loaded") is not False
        for row in history
    ):
        raise ValueError("V37 resume history crossed its train-only boundary")
    runtime = json.loads((resume / RUNTIME_METADATA_FILENAME).read_text(encoding="utf-8"))
    validate_runtime_checkpoint_metadata(runtime)
    if runtime != runtime_checkpoint_metadata(metadata):
        raise ValueError("V37 resume runtime metadata is not freshly sanitized")
    tensors = load_file(resume / "adapter.safetensors", device="cpu")
    if tensor_state_sha256(_v37_frozen_complement(tensors)) != (
        contract.frozen_complement_state_sha256
    ):
        raise ValueError("V37 resume changed a frozen tensor or buffer")
    if tensor_state_sha256(_bank_state(tensors, _CORE_PREFIX)) != (
        contract.source_core_state_sha256
    ) or tensor_state_sha256(_bank_state(tensors, _QUERY_PREFIX)) != (
        contract.source_query_state_sha256
    ):
        raise ValueError("V37 resume changed the learned core or query bank")
    target_hash = tensor_state_sha256(_bank_state(tensors, _TARGET_PREFIX))
    bank_hashes = _mapping(metadata.get("lora_bank_state_sha256"), "resume bank hashes")
    if target_hash != bank_hashes.get(_TARGET_BANK) or target_hash != stage.get(
        "target_bank_state_sha256"
    ):
        raise ValueError("V37 resume target-bank hash differs from metadata")
    source_tensors = load_file(contract.source_checkpoint / "adapter.safetensors", device="cpu")
    if set(tensors) != set(source_tensors):
        raise ValueError("V37 resume tensor inventory changed")
    changed = {name for name in tensors if not torch.equal(tensors[name], source_tensors[name])}
    if step == 0:
        if changed or tensor_state_sha256(tensors) != contract.source_tensor_state_sha256:
            raise ValueError("V37 update zero is not exact V36 update 16")
    elif not changed or not changed.issubset(_TARGET_PARAMETER_NAME_SET):
        raise ValueError("V37 resume changed an unauthorized tensor")
    if step:
        optimizer_step_audit(resume, expected_step=step, tensors=tensors)
    gate16, gate32, gate64 = replay_v37_gates(metadata, contract)
    if step >= 16 and (gate16 is None or gate16.get("passed") is not True):
        raise ValueError("V37 cannot resume past a failed update-16 gate")
    if step >= 32 and (gate32 is None or gate32.get("passed") is not True):
        raise ValueError("V37 cannot resume past a failed update-32 gate")
    if step >= 64 and (gate64 is None or gate64.get("passed") is not True):
        raise ValueError("V37 completed arm lacks a passed update-64 gate")
    return metadata


def run_v37(*, config: dict[str, Any], output: Path, resume: Path | None = None) -> dict[str, Any]:
    if output.is_symlink() or (output.exists() and not output.is_dir()):
        raise ValueError("V37 output root must be a real directory")
    if resume is None and output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty V37 output: {output}")
    contract = v37_contract(config)
    settings = v37_settings(config)
    terminal = require_v36_terminal_gate(config)
    source_checkpoint, pinned_source_metadata, source_audit = require_exact_v36_source(config)
    loader_config = v37_loader_config(config)
    assert_deferred_final_scenes_absent(loader_config)
    records, qa_audit = load_v35_train_qa_records(loader_config)
    train_pairs = build_exact_question_pair_units(records)
    schedule, schedule_audit = build_v37_schedule(records, train_pairs, seed=int(config["seed"]))
    if schedule_audit["schedule_sha256"] != contract.schedule_sha256:
        raise RuntimeError("V37 generated schedule differs from its exact lock")
    v36_schedule, _v36_schedule_audit = build_v35_schedule(
        records,
        train_pairs,
        settings=v35_settings(loader_config),
        seed=int(config["seed"]),
    )
    broad_calibration = v36_broad_calibration_records(v36_schedule)

    approved_v29 = require_approved_v29_source(loader_config)
    bundle, block_core, source_metadata, loader_transition = load_v37_bundle(
        config, approved_v29, source_checkpoint
    )
    if source_metadata != pinned_source_metadata:
        raise RuntimeError("V37 source metadata changed during exact adapter load")
    validate_block_cross_residual_state(
        block_core,
        expected_parameter_count=983_040,
        expected_state_sha256=contract.source_core_state_sha256,
        context="V37 frozen learned V36 block core",
    )
    if _query_bank(bundle).state_sha256() != contract.source_query_state_sha256:
        raise RuntimeError("V37 learned query bank differs after source load")
    if _target_bank(bundle).state_sha256() != contract.target_source_state_sha256:
        raise RuntimeError("V37 learned target bank was reinitialized")
    if frozen_v37_state_sha256(bundle) != contract.frozen_complement_state_sha256:
        raise RuntimeError("V37 loaded frozen complement differs from its exact source")

    split = v31_contract(loader_config)
    all_development_scene_ids = (*split.train_scene_ids, *split.validation_scene_ids)
    caches, cache_audit = cache_v35_scenes(
        config=loader_config,
        bundle=bundle,
        source_metadata=pinned_source_metadata,
        terminal=require_v34_terminal_gate(loader_config),
        scene_ids=split.train_scene_ids,
        manifest_scene_ids=all_development_scene_ids,
    )
    cache_audit.update(
        {
            "scene_scope": "training_only",
            "authenticated_manifest_scene_count": len(all_development_scene_ids),
            "authenticated_manifest_train_subset_count": len(split.train_scene_ids),
            "validation_scene_ids_loaded": [],
            "validation_environment_maps_loaded": False,
            "deferred_final_scene_ids_loaded": [],
        }
    )
    train_caches: dict[str, V35SceneCache] = dict(caches)
    cache_boundary = validate_v37_training_cache_boundary(
        cache_audit=cache_audit,
        caches=train_caches,
        config=loader_config,
        train_scene_ids=split.train_scene_ids,
        validation_scene_ids=split.validation_scene_ids,
    )
    freeze_for_v37(bundle)
    prefix_replay = _v37_prefix_replay_attestation(
        caches=caches,
        block_cross_residual=block_core,
        bundle=bundle,
        expected_scene_ids=split.train_scene_ids,
    )
    source_pair_metrics = augment_pair_metrics(
        training_pair_metrics(
            units=train_pairs,
            caches=train_caches,
            block_cross_residual=block_core,
            bundle=bundle,
            settings=v35_settings(loader_config),
        )
    )
    source_residual = residual_rms_diagnostics(
        caches=train_caches,
        block_cross_residual=block_core,
        device=bundle.language.device,
    )
    source_broad_nll = training_broad_nll(
        records=broad_calibration,
        caches=train_caches,
        block_cross_residual=block_core,
        bundle=bundle,
    )
    source_greedy_metrics = training_greedy_metrics(
        units=train_pairs,
        broad_records=broad_calibration,
        caches=train_caches,
        block_cross_residual=block_core,
        bundle=bundle,
        config=loader_config,
    )
    source_replay = _source_replay_attestation(
        source_metadata=pinned_source_metadata,
        pair_metrics=source_pair_metrics,
        broad_nll=source_broad_nll,
        residual=source_residual,
        prefix_replay=prefix_replay,
        expected_scene_ids=split.train_scene_ids,
    )
    source_replay["training_cache_boundary"] = cache_boundary
    target = _target_bank(bundle)
    learned_b = [adapter.lora_b for adapter in target.adapters]
    if any(torch.count_nonzero(value).item() == 0 for value in learned_b):
        raise RuntimeError("V37 target bank is not the learned V23 source")
    update_zero = {
        "exact_v36_update16_adapter_loaded": True,
        "source_tensor_state_sha256": module_collection_state_sha256(bundle.checkpoint_modules),
        "existing_learned_target_bank_loaded_without_reinitialization": True,
        "target_bank_source_state_sha256": target.state_sha256(),
        "target_bank_all_b_tensors_nonzero": True,
        "learned_block_core_state_sha256": block_core.state_sha256(),
        "learned_query_bank_state_sha256": _query_bank(bundle).state_sha256(),
        "frozen_complement_state_sha256": frozen_v37_state_sha256(bundle),
        "source_optimizer_file_opened": False,
        "source_optimizer_state_loaded": False,
        "fresh_adam_state": True,
        "source_pair_broad_residual_and_prefix_replay_exact": True,
        "loader_transition": loader_transition,
        "validation_qa_loaded": False,
        "oracle_environment_files_loaded": False,
    }
    surface = assert_v37_trainable_surface(bundle)
    optimizer = v37_optimizer(bundle, settings)
    history: list[dict[str, Any]] = [
        {
            "optimizer_update": 0,
            "source_pair_metrics": source_pair_metrics,
            "source_broad_train_nll": source_broad_nll,
            "source_train_greedy_metrics": source_greedy_metrics,
            "training_residual_diagnostics": source_residual,
            "scene_prefix_and_block_residual_exact": True,
            "target_bank_state_sha256": target.state_sha256(),
            "frozen_complement_state_sha256": frozen_v37_state_sha256(bundle),
            "update_zero_equivalence": update_zero,
            "validation_qa_loaded": False,
            "oracle_environment_files_loaded": False,
            "saved_checkpoint": True,
        }
    ]
    start_step = 0
    accepted16: Mapping[str, Any] | None = None
    accepted32: Mapping[str, Any] | None = None
    accepted64: Mapping[str, Any] | None = None
    output.mkdir(parents=True, exist_ok=True)
    if resume is not None:
        resume_metadata = validate_v37_resume_checkpoint(
            config=config,
            output=output,
            resume=resume,
            contract=contract,
            terminal=terminal,
            schedule_audit=schedule_audit,
            cache_audit=cache_audit,
            source_replay=source_replay,
            prefix_replay=prefix_replay,
            source_pair_metrics=source_pair_metrics,
            source_broad_nll=source_broad_nll,
            source_greedy_metrics=source_greedy_metrics,
            source_residual=source_residual,
        )
        loaded = load_adapter_checkpoint(
            resume,
            bundle.checkpoint_modules,
            device="cpu",
            metadata_filename=TRAINING_METADATA_FILENAME,
        )
        if loaded != resume_metadata:
            raise RuntimeError("V37 resume metadata changed during adapter load")
        freeze_for_v37(bundle)
        assert_v37_trainable_surface(bundle, optimizer=optimizer)
        if frozen_v37_state_sha256(bundle) != contract.frozen_complement_state_sha256:
            raise RuntimeError("V37 resumed a changed frozen complement")
        start_step = int(resume_metadata["optimizer_step"])
        if start_step:
            load_optimizer_checkpoint(resume, optimizer, bundle.language.device)
            _optimizer_payload_audit(
                optimizer.state_dict(),
                expected_step=start_step,
                tensors=load_file(resume / "adapter.safetensors", device="cpu"),
            )
        history = list(resume_metadata["history"])
        stage = _mapping(resume_metadata["v37_scene_ingress_kv"], "V37 resume stage")
        accepted16 = stage.get("update16_train_only_gate")
        accepted32 = stage.get("update32_train_only_gate")
        accepted64 = stage.get("update64_train_only_gate")
    else:
        metadata0 = _metadata(
            source_metadata=pinned_source_metadata,
            config=config,
            terminal=terminal,
            source_audit={**source_audit, "loader_transition": loader_transition},
            schedule_audit=schedule_audit,
            cache_audit=cache_audit,
            qa_audit=qa_audit,
            source_replay=source_replay,
            prefix_replay=prefix_replay,
            source_pair_metrics=source_pair_metrics,
            source_broad_nll=source_broad_nll,
            source_greedy_metrics=source_greedy_metrics,
            source_residual=source_residual,
            history=history,
            optimizer_step=0,
            bundle=bundle,
            surface=surface,
            gate16=None,
            gate32=None,
            gate64=None,
        )
        _save(output / "update_000", bundle=bundle, metadata=metadata0, optimizer=None)
        saved0 = load_file(output / "update_000" / "adapter.safetensors", device="cpu")
        if tensor_state_sha256(saved0) != contract.source_tensor_state_sha256:
            raise RuntimeError("V37 saved update zero differs from exact V36 update 16")

    target_parameters = tuple(_target_bank(bundle).parameters())
    frozen_residual_rms_value = float(source_residual["aggregate_rms"])
    normalized_residual_value = (
        frozen_residual_rms_value / settings.residual_penalty_scale
    ) ** 2
    for item in schedule[start_step:]:
        step = item.optimizer_step
        freeze_for_v37(bundle)
        assert_v37_trainable_surface(bundle, optimizer=optimizer)
        optimizer.zero_grad(set_to_none=True)
        broad_tokens = current_scene_tokens(
            train_caches[item.broad_record.scene_id],
            block_core,
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
                train_caches[scene_id], block_core, device=bundle.language.device
            )
            for scene_id in item.pair_unit.scene_ids
        }
        pair_nll, side_hinge, cross_hinge, diagnostics = paired_cross_prefix_objective(
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
        side_margin_mean = float(diagnostics["side_margins"].float().mean().cpu())
        cross_margin_mean = float(diagnostics["cross_prefix_margins"].float().mean().cpu())
        del pair_nll, side_hinge, cross_hinge, diagnostics, pair_objective, pair_tokens

        # The only trainable tensors are decoder-side K/V LoRA parameters.
        # The frozen scene-stack residual is constant, so it is reused rather
        # than recomputed over every training map at every optimizer update.
        residual_rms_value = frozen_residual_rms_value
        if any(parameter.grad is None for parameter in target_parameters):
            raise RuntimeError("V37 target-bank tensor lacks a gradient")
        if any(not torch.isfinite(parameter.grad).all() for parameter in target_parameters):
            raise RuntimeError("V37 target-bank gradient is nonfinite")
        preclip_norm = _gradient_norm(target_parameters)
        torch.nn.utils.clip_grad_norm_(target_parameters, settings.gradient_clip_norm)
        optimizer.step()
        _target_bank(bundle).validate_state()
        if frozen_v37_state_sha256(bundle) != contract.frozen_complement_state_sha256:
            raise RuntimeError("V37 changed a frozen tensor or buffer")
        if (
            block_core.state_sha256() != contract.source_core_state_sha256
            or _query_bank(bundle).state_sha256() != contract.source_query_state_sha256
        ):
            raise RuntimeError("V37 changed the learned V36 core or query bank")

        should_save = step in contract.saved_optimizer_steps
        pair_metrics: Mapping[str, Any] | None = None
        broad_diagnostic: float | None = None
        greedy_metrics: Mapping[str, Any] | None = None
        residual_diagnostics: Mapping[str, Any] | None = None
        prefix_exact: bool | None = None
        if should_save:
            residual_diagnostics = residual_rms_diagnostics(
                caches=train_caches,
                block_cross_residual=block_core,
                device=bundle.language.device,
            )
        if step in {8, 16, 32, 64}:
            pair_metrics = augment_pair_metrics(
                training_pair_metrics(
                    units=train_pairs,
                    caches=train_caches,
                    block_cross_residual=block_core,
                    bundle=bundle,
                    settings=v35_settings(loader_config),
                )
            )
        if step in {16, 32, 64}:
            broad_diagnostic = training_broad_nll(
                records=broad_calibration,
                caches=train_caches,
                block_cross_residual=block_core,
                bundle=bundle,
            )
            current_prefix = _v37_prefix_replay_attestation(
                caches=caches,
                block_cross_residual=block_core,
                bundle=bundle,
                expected_scene_ids=split.train_scene_ids,
            )
            prefix_exact = current_prefix == prefix_replay
        if step == 64:
            greedy_metrics = training_greedy_metrics(
                units=train_pairs,
                broad_records=broad_calibration,
                caches=train_caches,
                block_cross_residual=block_core,
                bundle=bundle,
                config=loader_config,
            )
        residual_exact = residual_diagnostics == source_residual
        scene_exact = bool(prefix_exact and residual_exact) if prefix_exact is not None else None
        target_hash = _target_bank(bundle).state_sha256()
        frozen_hash = frozen_v37_state_sha256(bundle)
        if step == 16:
            assert pair_metrics is not None and broad_diagnostic is not None
            accepted16 = v37_update16_gate(
                pair_metrics=pair_metrics,
                broad_nll=broad_diagnostic,
                target_bank_state_sha256=target_hash,
                frozen_complement_state_sha256=frozen_hash,
                residual_exact=scene_exact is True,
                contract=contract,
            )
        if step == 32:
            if not isinstance(accepted16, Mapping):
                raise RuntimeError("V37 update-32 gate lacks update-16 evidence")
            assert pair_metrics is not None and broad_diagnostic is not None
            accepted32 = v37_update32_gate(
                update16_gate=accepted16,
                pair_metrics=pair_metrics,
                broad_nll=broad_diagnostic,
                frozen_complement_state_sha256=frozen_hash,
                residual_exact=scene_exact is True,
                contract=contract,
            )
        if step == 64:
            if not isinstance(accepted32, Mapping):
                raise RuntimeError("V37 update-64 gate lacks update-32 evidence")
            assert pair_metrics is not None and broad_diagnostic is not None
            assert greedy_metrics is not None
            accepted64 = v37_update64_gate(
                update32_gate=accepted32,
                pair_metrics=pair_metrics,
                greedy_metrics=greedy_metrics,
                source_greedy_metrics=source_greedy_metrics,
                broad_nll=broad_diagnostic,
                frozen_complement_state_sha256=frozen_hash,
                residual_exact=scene_exact is True,
                contract=contract,
            )
        optimized_loss_value, reported_composite_value = v37_loss_values(
            settings=settings,
            broad_nll=broad_value,
            pair_correct_nll=pair_nll_value,
            side_hinge=side_hinge_value,
            cross_prefix_hinge=cross_hinge_value,
            frozen_normalized_residual=normalized_residual_value,
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
                "train_optimized_loss": optimized_loss_value,
                "train_reported_composite_including_frozen_residual": (
                    reported_composite_value
                ),
                "train_objective": optimized_loss_value,
                "frozen_residual_descriptive_only": True,
                "residual_penalty_contributes_gradient": False,
                "optimizer_stage": "existing_scene_ingress_kv_lora_only",
                "preclip_gradient_norm": preclip_norm,
                "gradient_clip_norm": settings.gradient_clip_norm,
                "training_pair_metrics": pair_metrics,
                "training_broad_nll": broad_diagnostic,
                "training_greedy_metrics": greedy_metrics,
                "training_residual_diagnostics": residual_diagnostics,
                "scene_prefix_and_block_residual_exact": scene_exact,
                "target_bank_state_sha256": target_hash,
                "frozen_complement_state_sha256": frozen_hash,
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
        metadata = _metadata(
            source_metadata=pinned_source_metadata,
            config=config,
            terminal=terminal,
            source_audit={**source_audit, "loader_transition": loader_transition},
            schedule_audit=schedule_audit,
            cache_audit=cache_audit,
            qa_audit=qa_audit,
            source_replay=source_replay,
            prefix_replay=prefix_replay,
            source_pair_metrics=source_pair_metrics,
            source_broad_nll=source_broad_nll,
            source_greedy_metrics=source_greedy_metrics,
            source_residual=source_residual,
            history=history,
            optimizer_step=step,
            bundle=bundle,
            surface=assert_v37_trainable_surface(bundle, optimizer=optimizer),
            gate16=accepted16,
            gate32=accepted32,
            gate64=accepted64,
        )
        checkpoint = output / f"update_{step:03d}"
        _save(checkpoint, bundle=bundle, metadata=metadata, optimizer=optimizer)
        optimizer_step_audit(
            checkpoint,
            expected_step=step,
            tensors=load_file(checkpoint / "adapter.safetensors", device="cpu"),
        )
        print(
            json.dumps(
                {
                    "phase": "v37_scene_ingress_kv_checkpoint",
                    "optimizer_step": step,
                    "training_complete_units": None
                    if pair_metrics is None
                    else pair_metrics["complete_units"],
                    "training_cross_complete_units": None
                    if pair_metrics is None
                    else pair_metrics["cross_prefix_complete_units"],
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
            raise RuntimeError("V37 update-16 train-only gate failed")
        if step == 32 and accepted32.get("passed") is not True:
            raise RuntimeError("V37 update-32 train-only gate failed")
        if step == 64 and accepted64.get("passed") is not True:
            raise RuntimeError("V37 update-64 train-only gate failed")

    return {
        "schema_version": 1,
        "artifact": "v37_diverse28_scene_ingress_kv_training",
        "output": str(output),
        "optimizer_updates": 64,
        "resumed_from_optimizer_step": start_step,
        "saved_optimizer_steps": list(contract.saved_optimizer_steps),
        "exact_trainable_parameter_count": 30_720,
        "source_v36_tensor_state_sha256": contract.source_tensor_state_sha256,
        "source_block_core_state_sha256": contract.source_core_state_sha256,
        "source_query_bank_state_sha256": contract.source_query_state_sha256,
        "source_target_bank_state_sha256": contract.target_source_state_sha256,
        "v36_terminal_report_sha256": terminal["sha256"],
        "update16_train_only_gate": accepted16,
        "update32_train_only_gate": accepted32,
        "update64_train_only_gate": accepted64,
        "source_optimizer_file_opened": False,
        "source_optimizer_state_loaded": False,
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
    parser.add_argument("--construction-preflight-only", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--resume-latest", action="store_true")
    args = parser.parse_args()
    if sum(
        bool(value)
        for value in (
            args.resume is not None,
            args.resume_latest,
            args.preflight_only,
            args.construction_preflight_only,
        )
    ) > 1:
        parser.error("preflight and resume modes are mutually exclusive")
    config = load_config(args.config)
    if args.preflight_only:
        print(json.dumps(preflight_v37(config), indent=2, sort_keys=True))
        return 0
    if args.construction_preflight_only:
        print(json.dumps(construction_preflight_v37(config), indent=2, sort_keys=True))
        return 0
    output = _unresolved_project_path(args.output)
    if output.is_symlink() or (output.exists() and not output.is_dir()):
        raise ValueError("V37 output path is a symlink or non-directory")
    contract = v37_contract(config)
    resume = None
    if args.resume is not None:
        unresolved_resume = _unresolved_project_path(args.resume)
        if unresolved_resume.is_symlink():
            raise ValueError("V37 resume path may not be a symlink")
        resume = unresolved_resume
    elif args.resume_latest:
        resume = latest_v37_resume_checkpoint(output, contract)
        if resume is None:
            raise FileNotFoundError("V37 output contains no complete resume arm")
    result = run_v37(config=config, output=output, resume=resume)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
