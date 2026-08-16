"""Train-only V41 projected-gradient continuation at Gemma layer 14.

V41 starts from the authenticated V40 update-zero checkpoint and continues
only adapter index 1 of the existing ``extension_v28_stage_b_query`` bank: its
layer-14 ``q_proj`` LoRA-B tensor.  The objective and exact surface are
authorized by the immutable V40 update-three terminal seal.  No source
optimizer, validation record, final scene, or oracle artifact is reachable
from this module.
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

from semantic_3d_chat.config import PROJECT_ROOT, config_hash, load_config
from semantic_3d_chat.data.dataset import QARecord
from semantic_3d_chat.evaluation.v39_layer14_query_gradient_screen import (
    cache_v39_train_scenes as cache_v41_train_scenes,
)
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
    current_scene_tokens,
    load_v35_train_qa_records,
    paired_cross_prefix_objective,
    residual_rms_diagnostics,
    v35_settings,
)
from semantic_3d_chat.training.train_environmental_sidecar_v33 import (
    assert_deferred_final_scenes_absent,
)
from semantic_3d_chat.training.train_joint_block_cross_v36 import (
    construct_v36_source_core,
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
from semantic_3d_chat.training.train_query_recovery_v38 import (
    v38_loader_config as _v41_loader_base_config,
)
from semantic_3d_chat.training.train_scene_ingress_kv_v37 import (
    augment_pair_metrics,
    validate_v37_training_cache_boundary,
)

DEFAULT_CONFIG = Path("configs/experiments/gemma4_diverse28_projected_gradient_v41.yaml")
DEFAULT_OUTPUT = Path(
    "data_gemma4/checkpoints/gemma4_v41_diverse28_projected_gradient_l14_query"
)
_AUTHORIZED_OUTPUT = DEFAULT_OUTPUT
_RETRY1_OUTPUT = Path(
    "data_gemma4/checkpoints/"
    "gemma4_v41_retry1_diverse28_projected_gradient_l14_query"
)
_FAILED_V41_ROOT = DEFAULT_OUTPUT
_RETRY1_TERMINAL = Path(
    "reports/gemma4/metrics/v41_update1_conversion_terminal_gate.json"
)
_FAILED_V41_FILES = {
    "update_000/adapter.safetensors": (
        "b0bdceb7699e9d97467915c69186f433d3b0fac2b09144d38c3078afe1f70cb0"
    ),
    "update_000/metadata.json": (
        "b1e66cec1aba693a3ffa6d5fd91dea78da4eb925db14c37b79171fd5bf94e4d8"
    ),
    "update_000/runtime_metadata.json": (
        "0037ac30ce329dc7041f0b18369ee56b947ff67780421cbaa4d72cbf01a4f1e2"
    ),
    "guard_failure_update_001.json": (
        "b416bd598832c7fcd07a1d098e03e29a50648489b486591158867e4dc586c53d"
    ),
}
OPTIMIZER_AUDIT_FILENAME = "optimizer_audit.json"
_UPDATE_DIRECTORY = re.compile(r"update_([0-9]{3})")

_V23_BANK = "extension_v23_shared_kv"
_QUERY_BANK = "extension_v28_stage_b_query"
_V23_PREFIX = f"lora_banks.{_V23_BANK}."
_QUERY_PREFIX = f"lora_banks.{_QUERY_BANK}."
_CORE_PREFIX = "block_cross_residual."
_TARGET_ADAPTER_INDEX = 1
_QUERY_PARAMETER_NAMES = (
    f"{_QUERY_PREFIX}adapters.{_TARGET_ADAPTER_INDEX}.lora_b",
)
_QUERY_PARAMETER_NAME_SET = frozenset(_QUERY_PARAMETER_NAMES)
_V23_PARAMETER_NAMES = tuple(
    f"{_V23_PREFIX}adapters.{index}.{side}"
    for index in range(4)
    for side in ("lora_a", "lora_b")
)
_V23_PARAMETER_NAME_SET = frozenset(_V23_PARAMETER_NAMES)
_V23_MODULES = (
    "model.language_model.layers.13.self_attn.k_proj",
    "model.language_model.layers.13.self_attn.v_proj",
    "model.language_model.layers.14.self_attn.k_proj",
    "model.language_model.layers.14.self_attn.v_proj",
)
_QUERY_MODULES = ("model.language_model.layers.14.self_attn.q_proj",)
_QUERY_SHAPES = ((4096, 4),)
_V23_SHAPES = (
    (4, 1536),
    (256, 4),
    (4, 1536),
    (256, 4),
    (4, 1536),
    (512, 4),
    (4, 1536),
    (512, 4),
)
_PAIR_FAMILIES = {
    "book_support": "pair_000015",
    "mirror_lr": "pair_000016",
    "picture_support": "pair_000017",
}
_PRIORITY_KEYS = (
    "cfq_13b1138d14c52a7c",
    "cfq_1c8b8cd72fcde904",
    "cfq_163eb92339ad35a5",
    "cfq_66aab89cee5bef49",
    "cfq_a1c673a1197a0961",
    "cfq_d469c4ac156ac42d",
    "cfq_ac7ac024c40aaddc",
    "cfq_fa3601dfffa80a0e",
)
_V40_SOURCE = Path(
    "data_gemma4/checkpoints/"
    "gemma4_v40_diverse28_cross_preserving_l14_query/update_000"
)
_V40_FILES = {
    "adapter.safetensors": "b0bdceb7699e9d97467915c69186f433d3b0fac2b09144d38c3078afe1f70cb0",
    TRAINING_METADATA_FILENAME: "ee74b1572061cae09d20bdf2b07e5f94ce9ef5c3ebfb6908131448bf8e5b484d",
    RUNTIME_METADATA_FILENAME: "209858f923ffa0916484209aeefad6f56a2cb4902bbd0dacd29decc222245c49",
}
_SOURCE_FULL_STATE_SHA256 = "7b951c6d7ae4f7b50603159f0bc4dfb4d50b5b40f9325134d78d1de1dae87fc0"
_V40_TERMINAL = Path("reports/gemma4/metrics/v40_update3_terminal_gate.json")
_V40_TERMINAL_SHA256 = (
    "d4c30be9e4f685697478b6e5a37f4f55d6e99962484b1cbae5c3c3214c24b35e"
)
_HYBRID_V23_STATE_SHA256 = "5c9233bd96b381e2f63443f8a739a868a21997b28061d90fb407a46d9de2d4cb"
_V28_BANK_STATE_SHA256 = "cc9dfa838bb87f32e2922d675658af4a1085d53a84ccdca6d5bacc6f7097217b"
_QUERY_SOURCE_STATE_SHA256 = "d0834cc588ee2a9edf08aabedfd01e0a6d2b01c6b6ae7e3a3d764eaddf58cc3e"
_FROZEN_STATE_SHA256 = "cec01bc088bb87c6bb44e0659eb03aa766f951ddeee706ca9a70edaa080dea5e"
_CORE_STATE_SHA256 = "92652fd2dbde2406227503f50717b2031baa1bcbc050902a379ddb9ddb52764f"
_PAIR_SCHEDULE_SHA256 = "4e26160c9a36e20ac894ff5e26604e6fad985ff5c84211282fd9807c9e738b3f"
_FULL_SCHEDULE_SHA256 = "2e7a2a8136c968739a8aaaa1138be87bec322fefa0fb340e6fb9bc6e07278bea"
_TRANSIENT_PRE_UPDATE3_TARGET_SHA256 = (
    "8c50aa3a5975f450c3c95fb00dbf077a33285bf22ac3208f5d745cd617bd8d48"
)
_SAVED_STEPS = (0, 8, 16, 24, 32, 40, 41)
_DIAGNOSTIC_STEPS = (0, 8, 16, 41)


@dataclass(frozen=True)
class V41Settings:
    enabled: bool
    optimizer_steps: int
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
        return _SAVED_STEPS


@dataclass(frozen=True)
class V41Contract:
    terminal_report: Path
    terminal_report_sha256: str
    source_checkpoint: Path
    source_file_sha256: Mapping[str, str]
    source_tensor_state_sha256: str
    hybrid_tensor_state_sha256: str
    hybrid_v23_state_sha256: str
    frozen_state_sha256: str
    query_source_state_sha256: str
    core_state_sha256: str
    pair_schedule_sha256: str
    schedule_sha256: str
    saved_optimizer_steps: tuple[int, ...]
    diagnostic_steps: tuple[int, ...]
    update_zero_expected: Mapping[str, Any]
    update8_gate: Mapping[str, Any]
    update16_gate: Mapping[str, Any]
    update41_gate: Mapping[str, Any]


@dataclass(frozen=True)
class V41Retry1Contract:
    terminal_report: Path
    terminal_report_sha256: str
    predecessor_root: Path
    predecessor_file_sha256: Mapping[str, str]
    authorized_output_root: Path


class V41GradientGuardFailure(RuntimeError):
    """A pre-mutation V41 direction guard failed with persistable evidence."""

    def __init__(self, audit: Mapping[str, Any]) -> None:
        self.audit = dict(audit)
        super().__init__(f"V41 component-gradient guard failed: {self.audit}")


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


def _bank_state(
    tensors: Mapping[str, torch.Tensor], prefix: str
) -> dict[str, torch.Tensor]:
    return {
        name.removeprefix(prefix): value
        for name, value in tensors.items()
        if name.startswith(prefix)
    }


def _frozen_excluding_query(
    tensors: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {name: value for name, value in tensors.items() if name not in _QUERY_PARAMETER_NAME_SET}


def v41_settings(config: Mapping[str, Any]) -> V41Settings:
    training = _mapping(config.get("training"), "training")
    raw = _mapping(training.get("v41_projected_gradient"), "training.v41_projected_gradient")
    expected = {
        "enabled": True,
        "optimizer_steps": 41,
        "broad_nll_weight": 1.0,
        "pair_correct_nll_weight": 0.5,
        "side_hinge_weight": 8.0,
        "side_hinge_margin": 0.5,
        "cross_prefix_flip_weight": 56.0,
        "cross_prefix_flip_margin": 0.10,
        "residual_penalty_weight": 0.0,
        "residual_penalty_scale": 0.05,
        "learning_rate": 0.003,
        "weight_decay": 0.0,
        "gradient_clip_norm": 1.0,
    }
    if set(raw) != set(expected):
        raise ValueError("V41 optimizer/objective contract contains missing or unknown fields")
    settings = V41Settings(
        enabled=raw.get("enabled") is True,
        optimizer_steps=_positive_int(raw.get("optimizer_steps"), "optimizer_steps"),
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
    if settings.__dict__ != expected:
        raise ValueError("V41 optimizer/objective settings changed from the terminal lock")
    if (
        _finite(training.get("lora_learning_rate"), "training.lora_learning_rate")
        != settings.learning_rate
        or _finite(training.get("lora_weight_decay"), "training.lora_weight_decay")
        != settings.weight_decay
    ):
        raise ValueError("V41 global LoRA optimizer settings disagree with SGD")
    return settings


def v41_loader_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return a shape-only V30 construction copy; never serialize it."""

    loader = _v41_loader_base_config(config)
    banks = _mapping(
        _mapping(loader.get("language"), "V41 loader language").get("lora_banks"),
        "V41 loader LoRA banks",
    )
    v28 = _mapping(banks.get(_QUERY_BANK), "V41 loader V28 bank")
    v28.update(
        {
            "trainable": False,
            "initialization_algorithm": "checkpoint_overwrite",
            "initialization_seed": None,
            "expected_initial_state_sha256": _V28_BANK_STATE_SHA256,
        }
    )
    parsed = lora_banks_settings(loader).bank(_QUERY_BANK)
    if not (
        parsed.trainable is False
        and parsed.initialization_algorithm == "checkpoint_overwrite"
        and parsed.initialization_seed is None
        and parsed.expected_initial_state_sha256 == _V28_BANK_STATE_SHA256
    ):
        raise RuntimeError("V41 construction copy failed to restore exact V28 source pin")
    return loader


def v41_contract(config: Mapping[str, Any]) -> V41Contract:
    v41_settings(config)
    raw = _mapping(config.get("v41_projected_gradient"), "v41_projected_gradient")
    expected_keys = {
        "schema_version",
        "role",
        "engine",
        "v40_terminal_report",
        "v40_terminal_report_sha256",
        "source_checkpoint",
        "source_optimizer_step",
        "source_file_sha256",
        "source_full_tensor_state_sha256",
        "complete_v28_bank_state_sha256",
        "frozen_excluding_target_state_sha256",
        "target_bank_name",
        "target_adapter_index",
        "target_module",
        "target_parameter_names",
        "target_parameter_shapes",
        "target_parameter_count",
        "target_tensor_count",
        "target_rank",
        "target_alpha",
        "target_dropout",
        "target_source_state_sha256",
        "source_v23_state_sha256",
        "source_block_core_state_sha256",
        "source_optimizer_state_loaded",
        "source_optimizer_file_opened",
        "validation_qa_loaded_during_training",
        "continuation_gates_use_training_only",
        "question_dependent_scene_processing",
        "question_dependent_retrieval",
        "priority_pair_ids",
        "priority_question_keys",
        "pair_schedule_sha256",
        "schedule_sha256",
        "saved_optimizer_steps",
        "per_unit_nll_diagnostics_required_at_steps",
        "priority_side_deficit_margin",
        "update_zero_expected",
        "update8_gate",
        "update16_gate",
        "update41_gate",
        "final_test_deferred",
    }
    if set(raw) != expected_keys:
        raise ValueError("V41 contract contains missing or unknown fields")
    exact = {
        "schema_version": 1,
        "role": "exact_v40_u0_existing_v28_layer14_projected_gradient_v41",
        "engine": "fresh_cpu_float64_projected_gradient_sgd_existing_layer14_lora_b_true_microsteps",
        "source_checkpoint": str(_V40_SOURCE),
        "source_optimizer_step": 0,
        "source_file_sha256": _V40_FILES,
        "source_full_tensor_state_sha256": _SOURCE_FULL_STATE_SHA256,
        "complete_v28_bank_state_sha256": _V28_BANK_STATE_SHA256,
        "frozen_excluding_target_state_sha256": _FROZEN_STATE_SHA256,
        "target_bank_name": _QUERY_BANK,
        "target_adapter_index": _TARGET_ADAPTER_INDEX,
        "target_module": _QUERY_MODULES[0],
        "target_parameter_names": list(_QUERY_PARAMETER_NAMES),
        "target_parameter_shapes": [list(shape) for shape in _QUERY_SHAPES],
        "target_parameter_count": 16_384,
        "target_tensor_count": 1,
        "target_rank": 4,
        "target_alpha": 8.0,
        "target_dropout": 0.0,
        "target_source_state_sha256": _QUERY_SOURCE_STATE_SHA256,
        "source_v23_state_sha256": _HYBRID_V23_STATE_SHA256,
        "source_block_core_state_sha256": _CORE_STATE_SHA256,
        "source_optimizer_state_loaded": False,
        "source_optimizer_file_opened": False,
        "validation_qa_loaded_during_training": False,
        "continuation_gates_use_training_only": True,
        "question_dependent_scene_processing": False,
        "question_dependent_retrieval": False,
        "priority_pair_ids": _PAIR_FAMILIES,
        "priority_question_keys": list(_PRIORITY_KEYS),
        "pair_schedule_sha256": _PAIR_SCHEDULE_SHA256,
        "schedule_sha256": _FULL_SCHEDULE_SHA256,
        "saved_optimizer_steps": list(_SAVED_STEPS),
        "per_unit_nll_diagnostics_required_at_steps": list(_DIAGNOSTIC_STEPS),
        "priority_side_deficit_margin": 0.5,
        "final_test_deferred": True,
    }
    mismatch = {
        key: {"observed": raw.get(key), "expected": value}
        for key, value in exact.items()
        if raw.get(key) != value
    }
    if mismatch:
        raise ValueError(f"V41 exact source/surface/schedule contract changed: {mismatch}")
    expected_zero = {
        "floating_absolute_tolerance": 1e-6,
        "broad_train_nll": 2.9013306349515915,
        "broad_greedy_exact_correct": 23,
        "broad_greedy_exact_total": 48,
        "priority_book_side_deficit": 16.926229,
        "priority_picture_side_deficit": 14.1875001,
        "priority_combined_side_deficit": 31.1137291,
        "complete_units": 9,
        "complete_physical_pair_coverage": 4,
        "cross_prefix_complete_units": 17,
        "positive_sides": 34,
        "mean_cross_prefix_margin": 1.4390720129,
        "book_complete_units": 0,
        "picture_complete_units": 0,
        "mirror_complete_units": 2,
        "book_cross_prefix_complete_units": 2,
        "picture_cross_prefix_complete_units": 2,
    }
    expected_gates = {
        "update8_gate": {
            "optimizer_step": 8,
            "priority_side_deficit_minimum_improvement": 0.5,
            "complete_units_minimum": 9,
            "positive_sides_minimum": 34,
            "cross_prefix_complete_units_minimum": 17,
            "broad_nll_maximum_increase": 0.02,
        },
        "update16_gate": {
            "optimizer_step": 16,
            "require_update8_passed": True,
            "priority_side_deficit_minimum_improvement": 3.12,
            "complete_units_minimum": 10,
            "positive_sides_minimum": 35,
            "complete_physical_pair_coverage_minimum": 5,
            "book_or_picture_complete_units_minimum": 1,
            "cross_prefix_complete_units_minimum": 17,
            "broad_nll_maximum_increase": 0.02,
        },
        "update41_gate": {
            "optimizer_step": 41,
            "require_update16_passed": True,
            "priority_side_deficit_minimum_improvement": 6.24,
            "complete_units_minimum": 12,
            "positive_sides_minimum": 37,
            "complete_physical_pair_coverage_minimum": 6,
            "book_complete_units_minimum": 1,
            "picture_complete_units_minimum": 1,
            "mirror_complete_units_minimum": 2,
            "cross_prefix_complete_units_minimum": 18,
            "greedy_complete_units_minimum": 6,
            "require_one_greedy_complete_per_priority_family": True,
            "broad_greedy_accuracy_must_meet_source": True,
            "broad_nll_maximum_increase": 0.02,
        },
    }
    if dict(_mapping(raw.get("update_zero_expected"), "update_zero_expected")) != expected_zero:
        raise ValueError("V41 update-zero behavioral lock changed")
    if any(
        dict(_mapping(raw.get(key), key)) != value
        for key, value in expected_gates.items()
    ):
        raise ValueError("V41 hard train-only gates changed")
    banks = lora_banks_settings(config)
    v28 = banks.bank(_QUERY_BANK)
    v30 = banks.bank("extension_v30_joint_pair_query")
    v23 = banks.bank(_V23_BANK)
    if not (
        v28.trainable is False
        and v28.adapter.target_modules
        == (
            "model.language_model.layers.13.self_attn.q_proj",
            "model.language_model.layers.14.self_attn.q_proj",
        )
        and v28.adapter.rank == 4
        and v28.adapter.alpha == 8.0
        and v28.adapter.dropout == 0.0
        and v28.expected_initial_state_sha256 is None
        and v30.trainable is False
        and v23.trainable is False
    ):
        raise ValueError("V41 runtime LoRA bank contract changed")
    split = v31_contract(v41_loader_config(config))
    if len(split.train_scene_ids) != 16 or len(split.validation_scene_ids) != 6:
        raise ValueError("V41 inherited split changed")
    terminal_sha = str(raw.get("v40_terminal_report_sha256"))
    if len(terminal_sha) != 64:
        raise ValueError("V41 terminal report SHA-256 is not pinned")
    return V41Contract(
        terminal_report=_resolve(str(raw.get("v40_terminal_report"))),
        terminal_report_sha256=terminal_sha,
        source_checkpoint=_resolve(_V40_SOURCE),
        source_file_sha256=_V40_FILES,
        source_tensor_state_sha256=_SOURCE_FULL_STATE_SHA256,
        hybrid_tensor_state_sha256=_SOURCE_FULL_STATE_SHA256,
        hybrid_v23_state_sha256=_HYBRID_V23_STATE_SHA256,
        frozen_state_sha256=_FROZEN_STATE_SHA256,
        query_source_state_sha256=_QUERY_SOURCE_STATE_SHA256,
        core_state_sha256=_CORE_STATE_SHA256,
        pair_schedule_sha256=_PAIR_SCHEDULE_SHA256,
        schedule_sha256=_FULL_SCHEDULE_SHA256,
        saved_optimizer_steps=_SAVED_STEPS,
        diagnostic_steps=_DIAGNOSTIC_STEPS,
        update_zero_expected=expected_zero,
        update8_gate=expected_gates["update8_gate"],
        update16_gate=expected_gates["update16_gate"],
        update41_gate=expected_gates["update41_gate"],
    )


def require_v40_terminal_gate(config: Mapping[str, Any]) -> dict[str, Any]:
    """Authenticate the immutable V40 failure seal and exact V41 grant."""

    contract = v41_contract(config)
    path = contract.terminal_report
    if path != _resolve(_V40_TERMINAL) or path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"V41 requires the real immutable V40 seal: {path}")
    observed = _sha256(path)
    if observed != _V40_TERMINAL_SHA256 or observed != contract.terminal_report_sha256:
        raise ValueError("V41 V40 terminal seal hash changed")
    report = json.loads(path.read_text(encoding="utf-8"))
    authorization = _mapping(
        _mapping(report, "V40 terminal").get("conditional_successor_authorization"),
        "V40 conditional successor authorization",
    )
    source = _mapping(authorization.get("source"), "V40 authorized source")
    target = _mapping(authorization.get("target_surface"), "V40 target surface")
    objective = _mapping(
        authorization.get("weighted_objective_components"),
        "V40 weighted objective",
    )
    solver = _mapping(
        authorization.get("projected_gradient_solver"), "V40 projection solver"
    )
    enumeration = _mapping(
        solver.get("active_set_enumeration"), "V40 active-set enumeration"
    )
    determinism = _mapping(solver.get("determinism"), "V40 determinism")
    cpu_safety = _mapping(solver.get("cpu_solution_safety"), "V40 CPU safety")
    cast_safety = _mapping(solver.get("device_cast_safety"), "V40 cast safety")
    clip_safety = _mapping(solver.get("scalar_clip_safety"), "V40 clip safety")
    transient = _mapping(
        authorization.get("transient_replay_gate"), "V40 transient replay gate"
    )
    optimizer = _mapping(authorization.get("optimizer"), "V40 optimizer")
    schedule = _mapping(authorization.get("schedule"), "V40 schedule")
    stop = _mapping(authorization.get("stop_and_isolation"), "V40 stop envelope")
    required = {
        "report.artifact": (report.get("artifact"), "v40_update3_terminal_gate"),
        "report.passed": (report.get("passed"), True),
        "report.successor": (
            report.get("only_exact_successor_authorized"),
            "v41_train_only_projected_gradient_continuation",
        ),
        "report.v41_authorized": (
            report.get("v41_train_only_projected_gradient_continuation_authorized"),
            True,
        ),
        "report.selector_denied": (report.get("selector_execution_authorized"), False),
        "report.validation_denied": (report.get("validation_access_authorized"), False),
        "report.oracle_denied": (report.get("oracle_access_authorized"), False),
        "report.final_denied": (report.get("final_test_access_authorized"), False),
        "authorization.id": (
            authorization.get("authorization_id"),
            "v41_cpu_f64_halfspace_projected_l14_lora_b",
        ),
        "authorization.authorized": (authorization.get("authorized"), True),
        "source.checkpoint": (source.get("checkpoint"), str(_V40_SOURCE)),
        "source.full": (
            source.get("full_tensor_state_sha256"),
            _SOURCE_FULL_STATE_SHA256,
        ),
        "source.target": (
            source.get("target_lora_b_state_sha256"),
            _QUERY_SOURCE_STATE_SHA256,
        ),
        "source.frozen": (
            source.get("frozen_excluding_b_state_sha256"),
            _FROZEN_STATE_SHA256,
        ),
        "target.names": (target.get("parameter_names"), list(_QUERY_PARAMETER_NAMES)),
        "target.shapes": (target.get("parameter_shapes"), [list(_QUERY_SHAPES[0])]),
        "target.count": (target.get("parameter_count"), 16_384),
        "objective.broad": (objective.get("broad"), 1.0),
        "objective.answer": (objective.get("answer"), 0.5),
        "objective.side": (objective.get("side"), 8.0),
        "objective.cross": (objective.get("cross"), 56.0),
        "solver.device": (solver.get("solver_device"), "cpu"),
        "solver.dtype": (solver.get("solver_dtype"), "torch.float64"),
        "solver.authorization_revision": (solver.get("authorization_revision"), 3),
        "solver.beta_floor": (solver.get("beta_absolute_floor"), 1e-12),
        "solver.beta_scale": (solver.get("beta_raw_norm_multiplier"), 1e-4),
        "solver.component_order": (
            solver.get("component_order"),
            ["broad", "answer", "side", "cross"],
        ),
        "solver.constraint_order": (
            solver.get("constraint_direction_order"),
            list(_PROJECTION_DIRECTION_NAMES),
        ),
        "solver.nonfinite_action": (
            solver.get("nonfinite_component_action"),
            "fail_stop_before_mutation",
        ),
        "solver.require_all_finite": (
            solver.get("require_all_raw_components_finite"),
            True,
        ),
        "solver.active_rule": (
            solver.get("active_constraint_rule"),
            "finite_l2_norm_strictly_greater_than_zero",
        ),
        "solver.all_may_zero": (
            solver.get("all_constraint_directions_may_be_zero_and_inactive"),
            True,
        ),
        "solver.side_not_constraint": (
            solver.get("standalone_side_is_not_a_constraint"),
            True,
        ),
        "solver.side_may_zero": (solver.get("standalone_side_may_be_zero"), True),
        "solver.zero_policy": (
            solver.get("zero_constraint_policy"),
            "record_inactive_and_first_order_satisfied_without_normalization",
        ),
        "solver.minimum_active": (solver.get("minimum_active_constraint_count"), 1),
        "solver.raw_norm_minimum": (
            solver.get("raw_total_norm_minimum_exclusive"),
            1e-12,
        ),
        "enumeration.mask_order": (
            enumeration.get("mask_order"),
            "ascending_integer_over_canonical_active_direction_order",
        ),
        "enumeration.mask_counts": (
            enumeration.get("mask_count_allowed"),
            [2, 4, 8, 16],
        ),
        "enumeration.active_counts": (
            enumeration.get("active_constraint_count_allowed"),
            [1, 2, 3, 4],
        ),
        "enumeration.rank_atol": (
            enumeration.get("rank_absolute_tolerance"),
            _PROJECTION_RANK_ABSOLUTE_TOLERANCE,
        ),
        "enumeration.rank_rtol": (
            enumeration.get("rank_relative_tolerance"),
            _PROJECTION_RANK_RELATIVE_TOLERANCE,
        ),
        "enumeration.dual": (
            enumeration.get("dual_lambda_lower_tolerance"),
            _PROJECTION_DUAL_LAMBDA_LOWER_TOLERANCE,
        ),
        "enumeration.kkt_atol": (
            enumeration.get("kkt_absolute_tolerance"),
            _PROJECTION_KKT_ABSOLUTE_TOLERANCE,
        ),
        "enumeration.kkt_rtol": (
            enumeration.get("kkt_relative_tolerance"),
            _PROJECTION_KKT_RELATIVE_TOLERANCE,
        ),
        "enumeration.tie_rtol": (
            enumeration.get("objective_tie_relative_tolerance"),
            _PROJECTION_OBJECTIVE_TIE_RELATIVE_TOLERANCE,
        ),
        "determinism.double": (
            determinism.get("solve_twice_from_independent_cpu_float64_clones"),
            True,
        ),
        "cpu_safety.cosine": (
            cpu_safety.get("projected_to_raw_cosine_minimum"),
            _PROJECTION_MINIMUM_RAW_COSINE,
        ),
        "cpu_safety.correction": (
            cpu_safety.get("correction_ratio_maximum"),
            _PROJECTION_MAXIMUM_CORRECTION_RATIO,
        ),
        "cast_safety.margin": (
            cast_safety.get("normalized_constraint_margin_minimum"),
            "beta/2",
        ),
        "clip_safety.norm": (clip_safety.get("clip_norm"), 1.0),
        "clip_safety.cosine": (
            clip_safety.get("projected_to_clipped_cosine_minimum"),
            0.9999999,
        ),
        "transient.hash": (
            transient.get("exact_target_hash_after_replayed_steps_one_and_two"),
            _TRANSIENT_PRE_UPDATE3_TARGET_SHA256,
        ),
        "optimizer.lr": (optimizer.get("learning_rate"), 0.003),
        "optimizer.foreach": (optimizer.get("foreach"), False),
        "optimizer.fused": (optimizer.get("fused"), False),
        "schedule.hash": (schedule.get("full_schedule_sha256"), _FULL_SCHEDULE_SHA256),
        "schedule.saved": (schedule.get("saved_optimizer_steps"), list(_SAVED_STEPS)),
        "stop.output": (stop.get("authorized_output_root"), str(_AUTHORIZED_OUTPUT)),
        "stop.new_terminal": (stop.get("new_terminal_seal_required_after_training"), True),
    }
    mismatch = {
        name: {"observed": value, "expected": expected}
        for name, (value, expected) in required.items()
        if value != expected
    }
    if mismatch:
        raise ValueError(f"V40 terminal does not authorize exact V41: {mismatch}")
    return {
        "path": str(path),
        "sha256": observed,
        "report": report,
        "authorization": dict(authorization),
    }


def v41_retry1_contract(config: Mapping[str, Any]) -> V41Retry1Contract | None:
    raw_value = config.get("v41_retry1")
    if raw_value is None:
        return None
    raw = _mapping(raw_value, "v41_retry1")
    expected_keys = {
        "schema_version",
        "role",
        "terminal_report",
        "terminal_report_sha256",
        "predecessor_root",
        "predecessor_file_sha256",
        "authorized_output_root",
        "cpu_first_mps_conversion_required",
        "validation_access_authorized",
        "oracle_access_authorized",
        "final_test_access_authorized",
        "selector_execution_authorized",
    }
    if set(raw) != expected_keys:
        raise ValueError("V41 retry1 contract contains missing or unknown fields")
    expected = {
        "schema_version": 1,
        "role": "v41_retry1_cpu_first_mps_conversion",
        "terminal_report": str(_RETRY1_TERMINAL),
        "predecessor_root": str(_FAILED_V41_ROOT),
        "predecessor_file_sha256": _FAILED_V41_FILES,
        "authorized_output_root": str(_RETRY1_OUTPUT),
        "cpu_first_mps_conversion_required": True,
        "validation_access_authorized": False,
        "oracle_access_authorized": False,
        "final_test_access_authorized": False,
        "selector_execution_authorized": False,
    }
    mismatch = {
        key: {"observed": raw.get(key), "expected": value}
        for key, value in expected.items()
        if raw.get(key) != value
    }
    terminal_sha = str(raw.get("terminal_report_sha256"))
    if mismatch or len(terminal_sha) != 64:
        raise ValueError(f"V41 retry1 exact contract changed: {mismatch}")
    return V41Retry1Contract(
        terminal_report=_resolve(str(raw["terminal_report"])),
        terminal_report_sha256=terminal_sha,
        predecessor_root=_resolve(str(raw["predecessor_root"])),
        predecessor_file_sha256=dict(_FAILED_V41_FILES),
        authorized_output_root=_resolve(str(raw["authorized_output_root"])),
    )


def authenticate_v41_retry1_predecessor(
    retry: V41Retry1Contract,
) -> dict[str, Any]:
    root = retry.predecessor_root
    if root != _resolve(_FAILED_V41_ROOT) or root.is_symlink() or not root.is_dir():
        raise ValueError("V41 retry1 predecessor root changed or is aliased")
    entries = sorted(path.name for path in root.iterdir())
    if entries != ["guard_failure_update_001.json", "update_000"]:
        raise ValueError(f"V41 retry1 predecessor envelope changed: {entries}")
    update0 = root / "update_000"
    if update0.is_symlink() or not update0.is_dir():
        raise ValueError("V41 retry1 predecessor update zero is unsafe")
    if sorted(path.name for path in update0.iterdir()) != [
        "adapter.safetensors",
        "metadata.json",
        "runtime_metadata.json",
    ]:
        raise ValueError("V41 retry1 predecessor update-zero inventory changed")
    for relative, expected in retry.predecessor_file_sha256.items():
        path = root / relative
        if path.is_symlink() or not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"V41 retry1 predecessor file changed: {relative}")
    failure_path = root / "guard_failure_update_001.json"
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    audit = _mapping(failure.get("audit"), "V41 predecessor failure audit")
    projection = _mapping(
        audit.get("projected_gradient_attestation"),
        "V41 predecessor failed projection",
    )
    raw = _mapping(
        audit.get("raw_component_gradient_diagnostic"),
        "V41 predecessor raw diagnostic",
    )
    projected_component_finite = _mapping(
        projection.get("component_finite"),
        "V41 predecessor failed projection finite flags",
    )
    raw_component_finite = _mapping(
        raw.get("component_finite"),
        "V41 predecessor raw diagnostic finite flags",
    )
    if (
        failure.get("artifact") != "v41_pre_step_gradient_guard_failure"
        or failure.get("optimizer_step_not_executed") != 1
        or failure.get("optimizer_step_executed") is not False
        or failure.get("checkpoint_written") is not False
        or failure.get("validation_qa_loaded") is not False
        or failure.get("oracle_environment_files_loaded") is not False
        or audit.get("failed_guard_stage") != "projection_input"
        or audit.get("clip_direction_attestation") is not None
        or audit.get("target_hash_before") != _QUERY_SOURCE_STATE_SHA256
        or audit.get("target_hash_after") != _QUERY_SOURCE_STATE_SHA256
        or audit.get("frozen_excluding_b_hash_before") != _FROZEN_STATE_SHA256
        or audit.get("frozen_excluding_b_hash_after") != _FROZEN_STATE_SHA256
        or projection.get("failure_reason")
        != "nonfinite_component_or_too_small_raw_total"
        or set(projected_component_finite)
        != {"broad", "answer", "side", "cross", "scene", "raw_total"}
        or any(value is not False for value in projected_component_finite.values())
        or set(raw_component_finite)
        != {"broad", "answer", "side", "cross", "scene", "total"}
        or any(value is not True for value in raw_component_finite.values())
    ):
        raise ValueError("V41 retry1 predecessor failure evidence changed")
    return {
        "schema_version": 1,
        "predecessor_root": str(root),
        "root_entries": entries,
        "predecessor_file_sha256": dict(retry.predecessor_file_sha256),
        "failed_before_optimizer_step": 1,
        "target_and_frozen_state_unchanged": True,
        "raw_cpu_first_diagnostic_finite": True,
        "combined_mps_to_cpu_float64_projection_nonfinite": True,
        "validation_qa_loaded": False,
        "oracle_environment_files_loaded": False,
        "final_test_scenes_loaded": False,
    }


def require_v41_training_authorization(config: Mapping[str, Any]) -> dict[str, Any]:
    base = require_v40_terminal_gate(config)
    retry = v41_retry1_contract(config)
    if retry is None:
        return {
            **base,
            "authorized_output_root": str(_resolve(_AUTHORIZED_OUTPUT)),
            "retry1": None,
        }
    predecessor = authenticate_v41_retry1_predecessor(retry)
    path = retry.terminal_report
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"V41 retry1 requires a real terminal seal: {path}")
    observed = _sha256(path)
    if observed != retry.terminal_report_sha256:
        raise ValueError("V41 retry1 terminal seal hash changed")
    report = json.loads(path.read_text(encoding="utf-8"))
    authorization = _mapping(
        _mapping(report, "V41 retry1 terminal").get(
            "conditional_successor_authorization"
        ),
        "V41 retry1 successor authorization",
    )
    required = {
        "artifact": (report.get("artifact"), "v41_update1_conversion_terminal_gate"),
        "passed": (report.get("passed"), True),
        "successor": (
            report.get("only_exact_successor_authorized"),
            "v41_retry1_train_only_projected_gradient_continuation",
        ),
        "retry_authorized": (
            report.get("v41_retry1_train_only_projected_gradient_continuation_authorized"),
            True,
        ),
        "validation_denied": (report.get("validation_access_authorized"), False),
        "oracle_denied": (report.get("oracle_access_authorized"), False),
        "final_denied": (report.get("final_test_access_authorized"), False),
        "selector_denied": (report.get("selector_execution_authorized"), False),
        "authorization.id": (
            authorization.get("authorization_id"),
            "v41_retry1_cpu_first_projected_gradient_l14_lora_b",
        ),
        "authorization.output": (
            authorization.get("authorized_output_root"),
            str(_RETRY1_OUTPUT),
        ),
        "authorization.cpu_first": (
            authorization.get("cpu_first_mps_conversion_required"),
            True,
        ),
        "authorization.predecessor_failure": (
            authorization.get("predecessor_guard_failure_sha256"),
            _FAILED_V41_FILES["guard_failure_update_001.json"],
        ),
    }
    mismatch = {
        key: {"observed": value, "expected": expected}
        for key, (value, expected) in required.items()
        if value != expected
    }
    if mismatch:
        raise ValueError(f"V41 retry1 terminal authorization changed: {mismatch}")
    return {
        **base,
        "authorized_output_root": str(retry.authorized_output_root),
        "retry1": {
            "path": str(path),
            "sha256": observed,
            "report": report,
            "authorization": dict(authorization),
            "predecessor_attestation": predecessor,
        },
    }


def require_exact_v41_sources(
    config: Mapping[str, Any],
) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
    """Authenticate exact V40 update zero without touching any optimizer."""

    contract = v41_contract(config)
    terminal = require_v40_terminal_gate(config)
    source = contract.source_checkpoint
    for filename, expected in contract.source_file_sha256.items():
        path = source / filename
        if path.is_symlink() or not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"V41 exact V40-u0 source file changed: {path}")
    metadata = json.loads((source / TRAINING_METADATA_FILENAME).read_text(encoding="utf-8"))
    runtime = json.loads((source / RUNTIME_METADATA_FILENAME).read_text(encoding="utf-8"))
    validate_runtime_checkpoint_metadata(runtime)
    if runtime != runtime_checkpoint_metadata(metadata):
        raise ValueError("V41 source runtime metadata is not exactly sanitized")
    tensors = load_file(source / "adapter.safetensors", device="cpu")
    hashes = {
        "full": tensor_state_sha256(tensors),
        "target": tensor_state_sha256(
            {
                name.removeprefix(f"{_QUERY_PREFIX}adapters.1."): value
                for name, value in tensors.items()
                if name in _QUERY_PARAMETER_NAME_SET
            }
        ),
        "v28_bank": tensor_state_sha256(_bank_state(tensors, _QUERY_PREFIX)),
        "v23_bank": tensor_state_sha256(_bank_state(tensors, _V23_PREFIX)),
        "block_core": tensor_state_sha256(_bank_state(tensors, _CORE_PREFIX)),
        "frozen_excluding_target": tensor_state_sha256(_frozen_excluding_query(tensors)),
    }
    expected = {
        "full": contract.source_tensor_state_sha256,
        "target": contract.query_source_state_sha256,
        "v28_bank": _V28_BANK_STATE_SHA256,
        "v23_bank": contract.hybrid_v23_state_sha256,
        "block_core": contract.core_state_sha256,
        "frozen_excluding_target": contract.frozen_state_sha256,
    }
    if hashes != expected or len(tensors) != 179:
        raise ValueError(f"V41 exact V40-u0 tensor audit failed: {hashes}")
    if any(not torch.isfinite(value).all() for value in tensors.values()):
        raise ValueError("V41 source contains nonfinite tensors")
    return tensors, metadata, {
        "schema_version": 1,
        "scope": "exact_authenticated_v40_update_zero",
        "source_checkpoint": str(_V40_SOURCE),
        "source_file_sha256_verified_without_optimizer": dict(_V40_FILES),
        "source_tensor_hashes": hashes,
        "v40_terminal_report_sha256": terminal["sha256"],
        "source_optimizer_file_opened": False,
        "source_optimizer_state_loaded": False,
        "validation_qa_loaded": False,
        "oracle_environment_files_loaded": False,
        "deferred_final_scene_ids_loaded": [],
    }


def _pair_family(unit: CounterfactualPairUnit) -> str:
    return next(
        (family for family, pair_id in _PAIR_FAMILIES.items() if pair_id == unit.pair_id),
        "other",
    )


def build_v41_schedule(
    records: Sequence[QARecord],
    pair_units: Sequence[CounterfactualPairUnit],
    *,
    seed: int,
) -> tuple[list[V35Microstep], dict[str, Any]]:
    if len(pair_units) != 25:
        raise ValueError("V41 requires the exact 25 changed training units")
    canonical = sorted(pair_units, key=lambda unit: (unit.pair_id, unit.question_key))
    by_key = {unit.question_key: unit for unit in canonical}
    if len(by_key) != 25 or any(key not in by_key for key in _PRIORITY_KEYS):
        raise ValueError("V41 priority unit inventory changed")
    priority = [by_key[key] for key in _PRIORITY_KEYS]
    if [_pair_family(unit) for unit in priority] != [
        "book_support",
        "picture_support",
    ] * 4:
        raise ValueError("V41 priority schedule no longer alternates book/picture")
    pairs = [*priority, *priority, *canonical]
    broad = select_balanced_broad_records(
        records,
        count=41,
        seed=seed,
        exclude_expected_change=True,
    )
    if len(pairs) != 41 or len(broad) != 41:
        raise RuntimeError("V41 schedule is not exactly 41 true updates")
    steps = [V35Microstep(index + 1, broad[index], pairs[index]) for index in range(41)]
    pair_payload = [
        {
            "optimizer_step": item.optimizer_step,
            "pair_id": item.pair_unit.pair_id,
            "question_key": item.pair_unit.question_key,
        }
        for item in steps
    ]
    full_payload = [
        {
            "optimizer_step": item.optimizer_step,
            "broad": (item.broad_record.scene_id, item.broad_record.question_id),
            "pair": (item.pair_unit.pair_id, item.pair_unit.question_key),
        }
        for item in steps
    ]
    pair_hash = hashlib.sha256(
        json.dumps(pair_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    full_hash = hashlib.sha256(
        json.dumps(full_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return steps, {
        "schema_version": 1,
        "optimizer_step_count": 41,
        "true_optimizer_step_per_schedule_row": True,
        "one_unchanged_broad_row_per_update": True,
        "broad_expected_change_excluded": True,
        "broad_answer_type_counts": dict(
            sorted(Counter(record.answer_type for record in broad).items())
        ),
        "pair_units_atomic": True,
        "pair_unit_count": 25,
        "priority_updates": [1, 16],
        "priority_question_keys": list(_PRIORITY_KEYS),
        "steps_9_through_16_repeat_steps_1_through_8_exactly": True,
        "canonical_cycle_updates": [17, 41],
        "pair_schedule_sha256": pair_hash,
        "schedule_sha256": full_hash,
        "saved_optimizer_steps": list(_SAVED_STEPS),
        "per_unit_nll_diagnostic_steps": list(_DIAGNOSTIC_STEPS),
        "questions_or_answers_serialized_to_runtime": False,
    }


def preflight_v41(config: Mapping[str, Any]) -> dict[str, Any]:
    """Read only terminal/source tensors plus train QA; load no model or maps."""

    contract = v41_contract(config)
    terminal = require_v41_training_authorization(config)
    _hybrid, _metadata, source_audit = require_exact_v41_sources(config)
    loader = v41_loader_config(config)
    assert_deferred_final_scenes_absent(loader)
    records, qa_audit = load_v35_train_qa_records(loader)
    units = build_exact_question_pair_units(records)
    _schedule, schedule_audit = build_v41_schedule(records, units, seed=int(config["seed"]))
    if (
        schedule_audit["pair_schedule_sha256"] != contract.pair_schedule_sha256
        or schedule_audit["schedule_sha256"] != contract.schedule_sha256
    ):
        raise ValueError("V41 generated schedule differs from its immutable pins")
    return {
        "schema_version": 1,
        "artifact": "v41_projected_gradient_preflight",
        "passed": True,
        "terminal_gate": {"path": terminal["path"], "sha256": terminal["sha256"]},
        "authorized_output_root": terminal["authorized_output_root"],
        "retry1_terminal_gate": None
        if terminal["retry1"] is None
        else {
            "path": terminal["retry1"]["path"],
            "sha256": terminal["retry1"]["sha256"],
        },
        "retry1_predecessor_attestation": None
        if terminal["retry1"] is None
        else dict(terminal["retry1"]["predecessor_attestation"]),
        "source_checkpoint": str(contract.source_checkpoint),
        "hybrid_tensor_state_sha256": contract.hybrid_tensor_state_sha256,
        "hybrid_v23_state_sha256": contract.hybrid_v23_state_sha256,
        "target_query_source_state_sha256": contract.query_source_state_sha256,
        "frozen_excluding_target_state_sha256": contract.frozen_state_sha256,
        "exact_trainable_tensor_count": 1,
        "exact_trainable_parameter_count": 16_384,
        "target_modules": list(_QUERY_MODULES),
        "schedule": schedule_audit,
        "saved_optimizer_steps": list(contract.saved_optimizer_steps),
        "per_unit_diagnostic_steps": list(contract.diagnostic_steps),
        "qa_audit": qa_audit,
        "source_audit": source_audit,
        "source_optimizer_files_opened": False,
        "source_optimizer_states_loaded": False,
        "gemma_loaded": False,
        "scene_maps_loaded": False,
        "validation_qa_loaded": False,
        "oracle_loaded": False,
        "final_test_scenes_touched": False,
    }


def _v23_bank(bundle: V30Bundle) -> LoRAInstallation:
    collection = bundle.lora_installation
    if collection is None:
        raise RuntimeError("V41 requires installed named LoRA banks")
    bank = collection.bank(_V23_BANK).installation
    if (
        bank.parameter_count != 30_720
        or bank.target_names != _V23_MODULES
        or tuple(tuple(value.shape) for value in bank.state_module.state_dict().values())
        != _V23_SHAPES
    ):
        raise RuntimeError("V41 V23 bank architecture changed")
    return bank


def _query_bank(bundle: V30Bundle) -> LoRAInstallation:
    collection = bundle.lora_installation
    if collection is None:
        raise RuntimeError("V41 requires installed named LoRA banks")
    bank = collection.bank(_QUERY_BANK).installation
    state = bank.state_module.state_dict()
    if (
        bank.parameter_count != 36_864
        or bank.target_names
        != (
            "model.language_model.layers.13.self_attn.q_proj",
            "model.language_model.layers.14.self_attn.q_proj",
        )
        or tuple(tuple(value.shape) for value in state.values())
        != ((4, 1536), (2048, 4), (4, 1536), (4096, 4))
        or not (
            bank.settings.rank == 4
            and bank.settings.alpha == 8.0
            and bank.settings.dropout == 0.0
        )
    ):
        raise RuntimeError("V41 query-bank architecture changed")
    return bank


def _target_parameters(bundle: V30Bundle) -> list[torch.nn.Parameter]:
    adapter = _query_bank(bundle).adapters[_TARGET_ADAPTER_INDEX]
    return [adapter.lora_b]


def target_v41_state_sha256(bundle: V30Bundle) -> str:
    adapter = _query_bank(bundle).adapters[_TARGET_ADAPTER_INDEX]
    return tensor_state_sha256({"lora_b": adapter.lora_b})


def frozen_v41_state_sha256(bundle: V30Bundle) -> str:
    state = {
        f"{module_name}.{name}": value
        for module_name, module in bundle.checkpoint_modules.items()
        for name, value in module.state_dict().items()
        if f"{module_name}.{name}" not in _QUERY_PARAMETER_NAME_SET
    }
    return tensor_state_sha256(state)


def retag_bundle_for_v41(bundle: V30Bundle, config: Mapping[str, Any]) -> dict[str, Any]:
    """Retag the V30-constructed collection without changing any tensor."""

    collection = bundle.lora_installation
    if collection is None:
        raise RuntimeError("V41 requires a multi-bank LoRA collection")
    before_names = collection.bank_names
    before_targets = collection.wrapped_modules
    before_hashes = collection.state_sha256()
    actual = lora_banks_settings(config)
    old = {bank.settings.name: bank for bank in collection.banks}
    if tuple(bank.name for bank in actual.banks) != before_names:
        raise RuntimeError("V41 loader-to-runtime bank inventory changed")
    collection.settings = actual
    collection.banks = tuple(
        InstalledLoRABank(settings=setting, installation=old[setting.name].installation)
        for setting in actual.banks
    )
    if (
        collection.bank_names != before_names
        or collection.wrapped_modules != before_targets
        or collection.state_sha256() != before_hashes
        or collection.trainable_parameter_count != 0
    ):
        raise RuntimeError("V41 retag changed bank identity, paths, tensors, or surface")
    bundle.config = copy.deepcopy(dict(config))
    bundle.trainable_bank_name = None
    return {
        "construction_used_v30_compatible_copy": True,
        "construction_copy_serialized_to_metadata": False,
        "bank_names_bit_exact": True,
        "target_paths_bit_exact": True,
        "state_hashes_bit_exact": True,
        "v41_trainable_bank": None,
        "v41_manually_trainable_adapter": f"{_QUERY_BANK}.adapters.1.lora_b",
        "v41_frozen_v23_bank": _V23_BANK,
        "v41_trainable_parameter_count": 16_384,
    }


def freeze_for_v41(bundle: V30Bundle) -> list[torch.nn.Parameter]:
    bundle.language.model.requires_grad_(False).eval()
    for module in bundle.checkpoint_modules.values():
        module.requires_grad_(False).eval()
    parameters = _target_parameters(bundle)
    for parameter in parameters:
        parameter.requires_grad_(True)
        parameter.grad = None
    _query_bank(bundle).eval()
    return parameters


def assert_v41_trainable_surface(
    bundle: V30Bundle,
    *,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, Any]:
    _query_bank(bundle)
    parameters = _target_parameters(bundle)
    expected_ids = {id(parameter) for parameter in parameters}
    observed_ids = {
        id(parameter)
        for module in bundle.checkpoint_modules.values()
        for parameter in module.parameters()
        if parameter.requires_grad
    }
    language_ids = {
        id(parameter) for parameter in bundle.language.model.parameters() if parameter.requires_grad
    }
    if observed_ids != expected_ids or language_ids != expected_ids:
        raise RuntimeError("V41 active trainable surface differs from its exact lock")
    if optimizer is not None:
        optimizer_ids = {
            id(parameter) for group in optimizer.param_groups for parameter in group["params"]
        }
        if optimizer_ids != expected_ids:
            raise RuntimeError("V41 optimizer contains an unauthorized tensor")
    if len(parameters) != 1 or sum(value.numel() for value in parameters) != 16_384:
        raise RuntimeError("V41 trainable tensor count changed")
    return {
        "target_bank": _QUERY_BANK,
        "target_adapter_index": _TARGET_ADAPTER_INDEX,
        "target_adapter_tensor": "lora_b",
        "target_module_paths": list(_QUERY_MODULES),
        "target_parameter_names": list(_QUERY_PARAMETER_NAMES),
        "trainable_tensor_count": 1,
        "trainable_parameter_count": 16_384,
        "rank": 4,
        "alpha": 8.0,
        "dropout": 0.0,
        "existing_learned_bank_continued_without_reinitialization": True,
        "gemma_base_frozen": True,
        "v23_k_only_hybrid_frozen": True,
        "v36_learned_block_core_frozen": True,
        "complete_scene_stack_frozen": True,
        "all_other_lora_banks_frozen": True,
        "every_other_tensor_and_buffer_frozen": True,
    }


def load_v41_bundle(
    config: dict[str, Any],
    approved_v29: ApprovedV29Source,
    source_checkpoint: Path,
    hybrid_tensors: Mapping[str, torch.Tensor],
) -> tuple[V30Bundle, BlockCrossResidual, dict[str, Any], dict[str, Any]]:
    """Construct fresh shapes and load exact authenticated V40 update zero."""

    contract = v41_contract(config)
    loader = v41_loader_config(config)
    bundle = load_v30_bundle(loader, approved_v29)
    block_core = construct_v36_source_core(loader, device=bundle.language.device)
    bundle.checkpoint_modules["block_cross_residual"] = block_core
    source_metadata = load_adapter_checkpoint(
        source_checkpoint,
        bundle.checkpoint_modules,
        device="cpu",
        metadata_filename=TRAINING_METADATA_FILENAME,
    )
    if module_collection_state_sha256(bundle.checkpoint_modules) != (
        contract.source_tensor_state_sha256
    ):
        raise RuntimeError("V41 constructed bundle did not load exact V40 update zero")
    if tensor_state_sha256(hybrid_tensors) != contract.hybrid_tensor_state_sha256:
        raise RuntimeError("V41 caller supplied a changed V40 update-zero tensor state")
    if module_collection_state_sha256(bundle.checkpoint_modules) != (
        contract.hybrid_tensor_state_sha256
    ):
        raise RuntimeError("V41 exact V40 source changed during retagging")
    transition = retag_bundle_for_v41(bundle, config)
    freeze_for_v41(bundle)
    assert_v41_trainable_surface(bundle)
    if (
        _v23_bank(bundle).state_sha256() != contract.hybrid_v23_state_sha256
        or _query_bank(bundle).state_sha256() != _V28_BANK_STATE_SHA256
        or target_v41_state_sha256(bundle) != contract.query_source_state_sha256
        or block_core.state_sha256() != contract.core_state_sha256
        or frozen_v41_state_sha256(bundle) != contract.frozen_state_sha256
    ):
        raise RuntimeError("V41 construction changed an authenticated hybrid tensor")
    return bundle, block_core, source_metadata, transition


def construction_preflight_v41(config: dict[str, Any]) -> dict[str, Any]:
    """Load real local Gemma and exact hybrid without maps or optimizer state."""

    contract = v41_contract(config)
    terminal = require_v41_training_authorization(config)
    hybrid, pinned_metadata, source_audit = require_exact_v41_sources(config)
    loader = v41_loader_config(config)
    approved = require_approved_v29_source(loader)
    bundle, block_core, loaded_metadata, transition = load_v41_bundle(
        config, approved, contract.source_checkpoint, hybrid
    )
    if loaded_metadata != pinned_metadata:
        raise RuntimeError("V41 construction loaded changed V40 update-zero metadata")
    runtime_metadata = copy.deepcopy(dict(loaded_metadata))
    runtime_metadata.update(bundle.lora_installation.checkpoint_metadata())
    validate_lora_banks_checkpoint_state(runtime_metadata, bundle.lora_installation)
    validate_block_cross_residual_state(
        block_core,
        expected_parameter_count=983_040,
        expected_state_sha256=contract.core_state_sha256,
        context="V41 frozen learned block core",
    )
    return {
        "schema_version": 1,
        "artifact": "v41_real_gemma_construction_preflight",
        "passed": True,
        "model_id": str(config["language"]["model_id"]),
        "device": str(bundle.language.device),
        "terminal_gate": {"path": terminal["path"], "sha256": terminal["sha256"]},
        "authorized_output_root": terminal["authorized_output_root"],
        "retry1_terminal_gate": None
        if terminal["retry1"] is None
        else {
            "path": terminal["retry1"]["path"],
            "sha256": terminal["retry1"]["sha256"],
        },
        "retry1_predecessor_attestation": None
        if terminal["retry1"] is None
        else dict(terminal["retry1"]["predecessor_attestation"]),
        "source_checkpoint": str(contract.source_checkpoint),
        "hybrid_tensor_state_sha256": contract.hybrid_tensor_state_sha256,
        "hybrid_v23_state_sha256": contract.hybrid_v23_state_sha256,
        "source_target_lora_b_state_sha256": contract.query_source_state_sha256,
        "frozen_excluding_query_state_sha256": contract.frozen_state_sha256,
        "runtime_checkpoint_state_validation_passed": True,
        "trainable_surface": assert_v41_trainable_surface(bundle),
        "loader_transition": transition,
        "source_audit": source_audit,
        "source_optimizer_files_opened": False,
        "source_optimizer_states_loaded": False,
        "scene_maps_loaded": False,
        "validation_qa_loaded": False,
        "oracle_loaded": False,
        "final_test_scenes_touched": False,
    }


def _prefix_replay_attestation(
    *,
    caches: Mapping[str, V35SceneCache],
    block_cross_residual: BlockCrossResidual,
    bundle: V30Bundle,
    expected_scene_ids: Sequence[str],
) -> dict[str, Any]:
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
        raise RuntimeError("V41 train-only prefix replay is incomplete or nondeterministic")
    return {
        "scene_count": 16,
        "scene_ids": list(expected),
        "prefix_sha256_by_scene": first,
        "replayed_prefix_sha256_by_scene": repeated,
        "prefixes_replayed_bit_exact": True,
        "scene_prefixes_built_before_questions": True,
        "all_occupied_blocks_processed": True,
        "training_scene_prefixes_question_free": True,
        "validation_environment_maps_loaded": False,
        "validation_qa_loaded": False,
        "question_dependent_scene_processing": False,
        "question_dependent_retrieval": False,
    }


def training_pair_gate_diagnostics(
    *,
    units: Sequence[CounterfactualPairUnit],
    caches: Mapping[str, V35SceneCache],
    block_cross_residual: BlockCrossResidual,
    bundle: V30Bundle,
    settings: V41Settings,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Compute all 25 metrics and persist numeric per-unit 2x2 NLL evidence."""

    side_values: list[torch.Tensor] = []
    cross_values: list[torch.Tensor] = []
    metric_rows: list[dict[str, Any]] = []
    nll_rows: list[dict[str, Any]] = []
    complete_by_family = {name: 0 for name in _PAIR_FAMILIES}
    block_cross_residual.eval()
    for unit in sorted(units, key=lambda value: (value.pair_id, value.question_key)):
        with torch.inference_mode():
            tokens = {
                scene_id: current_scene_tokens(
                    caches[scene_id], block_cross_residual, device=bundle.language.device
                )
                for scene_id in unit.scene_ids
            }
            correct_nll, _side_hinge, _cross_hinge, diagnostics = (
                paired_cross_prefix_objective(
                    unit=unit,
                    scene_tokens=tokens,
                    bundle=bundle,
                    side_margin=settings.side_hinge_margin,
                    cross_prefix_margin=settings.cross_prefix_flip_margin,
                )
            )
        side = diagnostics["side_margins"].detach().float().cpu().reshape(2)
        cross = diagnostics["cross_prefix_margins"].detach().float().cpu().reshape(2)
        correct_rank = diagnostics["correct_ranking_nll"].detach().float().cpu().reshape(2)
        swapped_rank = diagnostics["swapped_ranking_nll"].detach().float().cpu().reshape(2)
        correct_answer = diagnostics["correct_answer_nll"].detach().float().cpu().reshape(2)
        family = _pair_family(unit)
        complete = bool(side.gt(0).all())
        cross_complete = bool(cross.gt(0).all())
        if complete and family in complete_by_family:
            complete_by_family[family] += 1
        metric_rows.append(
            {
                "pair_id": unit.pair_id,
                "question_key": unit.question_key,
                "scene_ids": list(unit.scene_ids),
                "family": family,
                "side_margins": [float(value) for value in side],
                "cross_prefix_margins": [float(value) for value in cross],
                "complete": complete,
                "cross_prefix_complete": cross_complete,
            }
        )
        nll_rows.append(
            {
                "pair_id": unit.pair_id,
                "question_key": unit.question_key,
                "family": family,
                "scene_ids": list(unit.scene_ids),
                "correct_answer_nll_mean": float(correct_nll.detach().float().cpu()),
                "correct_answer_nll": [float(value) for value in correct_answer],
                "correct_ranking_nll": [float(value) for value in correct_rank],
                "swapped_ranking_nll": [float(value) for value in swapped_rank],
                "side_margins": [float(value) for value in side],
                "cross_prefix_margins": [float(value) for value in cross],
                "side_correct": [bool(value) for value in side.gt(0)],
                "cross_prefix_correct": [bool(value) for value in cross.gt(0)],
                "side_complete": complete,
                "cross_prefix_complete": cross_complete,
            }
        )
        side_values.append(side)
        cross_values.append(cross)
    if len(metric_rows) != 25 or len(nll_rows) != 25:
        raise RuntimeError("V41 gate diagnostics omitted a changed training unit")
    side_tensor = torch.stack(side_values)
    cross_tensor = torch.stack(cross_values)
    metrics = augment_pair_metrics(
        {
            "schema_version": 1,
            "unit_count": 25,
            "side_count": 50,
            "mean_margin": float(side_tensor.mean()),
            "minimum_margin": float(side_tensor.min()),
            "complete_units": sum(int(row["complete"]) for row in metric_rows),
            "positive_sides": int(side_tensor.gt(0).sum()),
            "mean_cross_prefix_margin": float(cross_tensor.mean()),
            "cross_prefix_complete_units": sum(
                int(row["cross_prefix_complete"]) for row in metric_rows
            ),
            "complete_units_by_family": complete_by_family,
            "units": metric_rows,
            "training_scenes_only": True,
            "validation_qa_loaded": False,
            "true_cross_prefix_differing_token_scores": True,
        }
    )
    return metrics, nll_rows


def priority_side_deficit(
    pair_metrics: Mapping[str, Any], *, margin: float = 0.5
) -> dict[str, float]:
    if not math.isfinite(margin) or margin < 0:
        raise ValueError("V41 priority deficit margin must be finite and nonnegative")
    rows = pair_metrics.get("units")
    if not isinstance(rows, list) or len(rows) != 25:
        raise ValueError("V41 priority deficit requires all 25 unit metrics")
    result = {"book_support": 0.0, "picture_support": 0.0}
    counts = {"book_support": 0, "picture_support": 0}
    for row in rows:
        item = _mapping(row, "V41 pair metric row")
        family = str(item.get("family"))
        if family not in result:
            continue
        margins = item.get("side_margins")
        if not isinstance(margins, list) or len(margins) != 2:
            raise ValueError("V41 pair metric row lacks two side margins")
        result[family] += sum(max(0.0, margin - float(value)) for value in margins)
        counts[family] += 1
    if counts != {"book_support": 4, "picture_support": 4}:
        raise ValueError("V41 priority deficit inventory changed")
    return {
        **result,
        "combined": result["book_support"] + result["picture_support"],
        "margin": margin,
    }


def validate_update_zero_baseline(
    *,
    pair_metrics: Mapping[str, Any],
    broad_nll: float,
    greedy_metrics: Mapping[str, Any],
    contract: V41Contract,
) -> dict[str, Any]:
    expected = contract.update_zero_expected
    tolerance = float(expected["floating_absolute_tolerance"])
    complete = _mapping(pair_metrics.get("complete_units_by_family"), "complete families")
    cross = _mapping(
        pair_metrics.get("cross_prefix_complete_units_by_family"), "cross families"
    )
    deficit = priority_side_deficit(pair_metrics, margin=0.5)
    observed = {
        "broad_train_nll": broad_nll,
        "broad_greedy_exact_correct": int(greedy_metrics["broad_exact_correct"]),
        "broad_greedy_exact_total": int(greedy_metrics["broad_row_count"]),
        "priority_book_side_deficit": float(deficit["book_support"]),
        "priority_picture_side_deficit": float(deficit["picture_support"]),
        "priority_combined_side_deficit": float(deficit["combined"]),
        "complete_units": int(pair_metrics["complete_units"]),
        "complete_physical_pair_coverage": int(
            pair_metrics["complete_physical_pair_coverage"]
        ),
        "cross_prefix_complete_units": int(pair_metrics["cross_prefix_complete_units"]),
        "positive_sides": int(pair_metrics["positive_sides"]),
        "mean_cross_prefix_margin": float(pair_metrics["mean_cross_prefix_margin"]),
        "book_complete_units": int(complete.get("book_support", 0)),
        "picture_complete_units": int(complete.get("picture_support", 0)),
        "mirror_complete_units": int(complete.get("mirror_lr", 0)),
        "book_cross_prefix_complete_units": int(cross.get("book_support", 0)),
        "picture_cross_prefix_complete_units": int(cross.get("picture_support", 0)),
    }
    floating = {
        "broad_train_nll",
        "priority_book_side_deficit",
        "priority_picture_side_deficit",
        "priority_combined_side_deficit",
        "mean_cross_prefix_margin",
    }
    mismatch = {
        key: {"observed": observed[key], "expected": expected[key]}
        for key in observed
        if (
            abs(float(observed[key]) - float(expected[key])) > tolerance
            if key in floating
            else observed[key] != expected[key]
        )
    }
    if mismatch:
        raise ValueError(f"V41 exact hybrid update-zero baseline changed: {mismatch}")
    return {
        "schema_version": 1,
        "passed": True,
        "floating_absolute_tolerance": tolerance,
        "observed": observed,
        "expected": dict(expected),
        "training_scenes_only": True,
        "validation_qa_loaded": False,
        "inherited_v37_metrics_used_as_hybrid_baseline": False,
        "recomputed_before_optimizer_step_1": True,
    }


def _family_counts(metrics: Mapping[str, Any]) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    return (
        _mapping(metrics.get("complete_units_by_family"), "complete family counts"),
        _mapping(
            metrics.get("cross_prefix_complete_units_by_family"), "cross family counts"
        ),
    )


def validate_per_unit_nll_diagnostics(
    diagnostics: Sequence[Mapping[str, Any]], pair_metrics: Mapping[str, Any]
) -> dict[str, Any]:
    metric_rows = pair_metrics.get("units")
    if not isinstance(metric_rows, list) or len(metric_rows) != 25:
        raise ValueError("V41 diagnostic validation requires all 25 pair metric rows")
    expected_rows = {
        (str(row["pair_id"]), str(row["question_key"])): _mapping(
            row, "V41 pair metric row"
        )
        for row in metric_rows
    }
    expected = set(expected_rows)
    observed: set[tuple[str, str]] = set()
    for raw in diagnostics:
        row = _mapping(raw, "V41 per-unit NLL diagnostic")
        identity = (str(row.get("pair_id")), str(row.get("question_key")))
        if identity in observed:
            raise ValueError("V41 per-unit NLL diagnostics contain a duplicate identity")
        observed.add(identity)
        scene_ids = row.get("scene_ids")
        correct_answer = row.get("correct_answer_nll")
        correct_rank = row.get("correct_ranking_nll")
        swapped_rank = row.get("swapped_ranking_nll")
        side_margins = row.get("side_margins")
        cross_margins = row.get("cross_prefix_margins")
        side_correct = row.get("side_correct")
        cross_correct = row.get("cross_prefix_correct")
        vectors = (correct_answer, correct_rank, swapped_rank, side_margins, cross_margins)
        if (
            not isinstance(scene_ids, list)
            or len(scene_ids) != 2
            or len({str(value) for value in scene_ids}) != 2
            or any(not isinstance(vector, list) or len(vector) != 2 for vector in vectors)
            or any(
                not math.isfinite(float(value))
                for vector in vectors
                for value in vector
            )
            or not isinstance(side_correct, list)
            or len(side_correct) != 2
            or any(not isinstance(value, bool) for value in side_correct)
            or not isinstance(cross_correct, list)
            or len(cross_correct) != 2
            or any(not isinstance(value, bool) for value in cross_correct)
            or not math.isfinite(float(row.get("correct_answer_nll_mean")))
        ):
            raise ValueError("V41 per-unit NLL diagnostic row is incomplete or nonfinite")
        metric = expected_rows.get(identity)
        if metric is None:
            raise ValueError("V41 per-unit NLL diagnostic has an unknown identity")
        correct_answer_values = [float(value) for value in correct_answer]
        correct_rank_values = [float(value) for value in correct_rank]
        swapped_rank_values = [float(value) for value in swapped_rank]
        side_values = [float(value) for value in side_margins]
        cross_values = [float(value) for value in cross_margins]
        expected_side = [
            swapped_rank_values[index] - correct_rank_values[index] for index in range(2)
        ]
        expected_cross = [
            swapped_rank_values[1] - correct_rank_values[0],
            swapped_rank_values[0] - correct_rank_values[1],
        ]
        def close(left: object, right: object) -> bool:
            return abs(float(left) - float(right)) <= 1e-6
        if (
            [str(value) for value in scene_ids]
            != [str(value) for value in metric.get("scene_ids", ())]
            or row.get("family") != metric.get("family")
            or any(
                not close(value, expected_value)
                for value, expected_value in zip(
                    side_values, metric.get("side_margins", ()), strict=True
                )
            )
            or any(
                not close(value, expected_value)
                for value, expected_value in zip(
                    cross_values, metric.get("cross_prefix_margins", ()), strict=True
                )
            )
            or any(
                not close(value, expected_value)
                for value, expected_value in zip(side_values, expected_side, strict=True)
            )
            or any(
                not close(value, expected_value)
                for value, expected_value in zip(cross_values, expected_cross, strict=True)
            )
            or not close(
                float(row["correct_answer_nll_mean"]), sum(correct_answer_values) / 2.0
            )
            or side_correct != [value > 0.0 for value in side_values]
            or cross_correct != [value > 0.0 for value in cross_values]
            or row.get("side_complete") is not all(side_correct)
            or row.get("cross_prefix_complete") is not all(cross_correct)
            or row.get("side_complete") is not metric.get("complete")
            or row.get("cross_prefix_complete") is not metric.get("cross_prefix_complete")
        ):
            raise ValueError("V41 per-unit NLL diagnostic disagrees with its 2x2 scores")
    if len(diagnostics) != 25 or observed != expected:
        raise ValueError("V41 per-unit NLL diagnostic inventory changed")
    return {
        "unit_count": 25,
        "unique_pair_question_key_count": 25,
        "per_side_correct_answer_nll_finite": True,
        "per_side_rank_nll_finite": True,
        "per_side_correctness_persisted": True,
    }


def v41_update8_gate(
    *,
    pair_metrics: Mapping[str, Any],
    broad_nll: float,
    source_broad_nll: float,
    source_priority_deficit: float,
    query_state_sha256: str,
    frozen_state_sha256: str,
    scene_state_exact: bool,
    per_unit_nll_diagnostics: Sequence[Mapping[str, Any]],
    contract: V41Contract,
) -> dict[str, Any]:
    rule = contract.update8_gate
    diagnostic_audit = validate_per_unit_nll_diagnostics(
        per_unit_nll_diagnostics, pair_metrics
    )
    deficit = float(priority_side_deficit(pair_metrics)["combined"])
    improvement = source_priority_deficit - deficit
    checks = {
        "priority_teacher_deficit_improved_at_least_0_5": improvement
        >= float(rule["priority_side_deficit_minimum_improvement"]),
        "teacher_complete_units_at_least_9": int(pair_metrics["complete_units"])
        >= int(rule["complete_units_minimum"]),
        "teacher_positive_sides_at_least_34": int(pair_metrics["positive_sides"])
        >= int(rule["positive_sides_minimum"]),
        "teacher_cross_complete_units_at_least_17": int(
            pair_metrics["cross_prefix_complete_units"]
        )
        >= int(rule["cross_prefix_complete_units_minimum"]),
        "broad_nll_within_hybrid_update_zero_plus_0_02": broad_nll
        <= source_broad_nll + float(rule["broad_nll_maximum_increase"]),
        "query_bank_state_changed": query_state_sha256 != contract.query_source_state_sha256,
        "frozen_state_exact": frozen_state_sha256 == contract.frozen_state_sha256,
        "scene_prefix_and_residual_exact": scene_state_exact,
        "all_25_per_unit_nll_diagnostics_persisted": diagnostic_audit["unit_count"] == 25,
    }
    return {
        "checks": dict(checks),
        **checks,
        "passed": all(checks.values()),
        "priority_teacher_side_deficit": deficit,
        "source_priority_teacher_side_deficit": source_priority_deficit,
        "priority_teacher_side_deficit_improvement": improvement,
        "broad_nll_delta_from_update_zero": broad_nll - source_broad_nll,
        "training_scenes_only": True,
        "validation_qa_loaded": False,
    }


def v41_update16_gate(
    *,
    update8_gate: Mapping[str, Any],
    pair_metrics: Mapping[str, Any],
    broad_nll: float,
    source_broad_nll: float,
    source_priority_deficit: float,
    query_state_sha256: str,
    frozen_state_sha256: str,
    scene_state_exact: bool,
    per_unit_nll_diagnostics: Sequence[Mapping[str, Any]],
    contract: V41Contract,
) -> dict[str, Any]:
    rule = contract.update16_gate
    diagnostic_audit = validate_per_unit_nll_diagnostics(
        per_unit_nll_diagnostics, pair_metrics
    )
    complete, _cross = _family_counts(pair_metrics)
    deficit = float(priority_side_deficit(pair_metrics)["combined"])
    improvement = source_priority_deficit - deficit
    checks = {
        "update8_train_only_gate_remains_passed": update8_gate.get("passed") is True,
        "priority_teacher_deficit_improved_at_least_3_12": improvement
        >= float(rule["priority_side_deficit_minimum_improvement"]),
        "teacher_complete_units_at_least_10": int(pair_metrics["complete_units"])
        >= int(rule["complete_units_minimum"]),
        "teacher_positive_sides_at_least_35": int(pair_metrics["positive_sides"])
        >= int(rule["positive_sides_minimum"]),
        "complete_physical_pair_coverage_at_least_5": int(
            pair_metrics["complete_physical_pair_coverage"]
        )
        >= int(rule["complete_physical_pair_coverage_minimum"]),
        "book_or_picture_teacher_complete": int(complete.get("book_support", 0))
        + int(complete.get("picture_support", 0))
        >= int(rule["book_or_picture_complete_units_minimum"]),
        "teacher_cross_complete_units_at_least_17": int(
            pair_metrics["cross_prefix_complete_units"]
        )
        >= int(rule["cross_prefix_complete_units_minimum"]),
        "broad_nll_within_hybrid_update_zero_plus_0_02": broad_nll
        <= source_broad_nll + float(rule["broad_nll_maximum_increase"]),
        "query_bank_state_changed": query_state_sha256 != contract.query_source_state_sha256,
        "frozen_state_exact": frozen_state_sha256 == contract.frozen_state_sha256,
        "scene_prefix_and_residual_exact": scene_state_exact,
        "all_25_per_unit_nll_diagnostics_persisted": diagnostic_audit["unit_count"] == 25,
    }
    return {
        "checks": dict(checks),
        **checks,
        "passed": all(checks.values()),
        "priority_teacher_side_deficit": deficit,
        "source_priority_teacher_side_deficit": source_priority_deficit,
        "priority_teacher_side_deficit_improvement": improvement,
        "broad_nll_delta_from_update_zero": broad_nll - source_broad_nll,
        "training_scenes_only": True,
        "validation_qa_loaded": False,
    }


def v41_update41_gate(
    *,
    update16_gate: Mapping[str, Any],
    pair_metrics: Mapping[str, Any],
    greedy_metrics: Mapping[str, Any],
    broad_nll: float,
    source_broad_nll: float,
    source_priority_deficit: float,
    query_state_sha256: str,
    frozen_state_sha256: str,
    scene_state_exact: bool,
    per_unit_nll_diagnostics: Sequence[Mapping[str, Any]],
    contract: V41Contract,
) -> dict[str, Any]:
    rule = contract.update41_gate
    diagnostic_audit = validate_per_unit_nll_diagnostics(
        per_unit_nll_diagnostics, pair_metrics
    )
    complete, _cross = _family_counts(pair_metrics)
    greedy_family = _mapping(
        greedy_metrics.get("complete_units_by_family"), "greedy complete families"
    )
    deficit = float(priority_side_deficit(pair_metrics)["combined"])
    improvement = source_priority_deficit - deficit
    checks = {
        "update16_train_only_gate_remains_passed": update16_gate.get("passed") is True,
        "priority_teacher_deficit_improved_at_least_6_24": improvement
        >= float(rule["priority_side_deficit_minimum_improvement"]),
        "teacher_complete_units_at_least_12": int(pair_metrics["complete_units"])
        >= int(rule["complete_units_minimum"]),
        "teacher_positive_sides_at_least_37": int(pair_metrics["positive_sides"])
        >= int(rule["positive_sides_minimum"]),
        "complete_physical_pair_coverage_at_least_6": int(
            pair_metrics["complete_physical_pair_coverage"]
        )
        >= int(rule["complete_physical_pair_coverage_minimum"]),
        "teacher_book_complete": int(complete.get("book_support", 0))
        >= int(rule["book_complete_units_minimum"]),
        "teacher_picture_complete": int(complete.get("picture_support", 0))
        >= int(rule["picture_complete_units_minimum"]),
        "teacher_mirror_complete_at_least_2": int(complete.get("mirror_lr", 0))
        >= int(rule["mirror_complete_units_minimum"]),
        "teacher_cross_complete_units_at_least_18": int(
            pair_metrics["cross_prefix_complete_units"]
        )
        >= int(rule["cross_prefix_complete_units_minimum"]),
        "train_greedy_complete_units_at_least_6": int(greedy_metrics["complete_units"])
        >= int(rule["greedy_complete_units_minimum"]),
        "train_greedy_each_priority_family": all(
            int(greedy_family.get(family, 0)) >= 1 for family in _PAIR_FAMILIES
        ),
        "broad_greedy_exact_correct_at_least_23_of_48": int(
            greedy_metrics["broad_exact_correct"]
        )
        >= 23
        and int(greedy_metrics["broad_row_count"]) == 48,
        "broad_nll_within_hybrid_update_zero_plus_0_02": broad_nll
        <= source_broad_nll + float(rule["broad_nll_maximum_increase"]),
        "query_bank_state_changed": query_state_sha256 != contract.query_source_state_sha256,
        "frozen_state_exact": frozen_state_sha256 == contract.frozen_state_sha256,
        "scene_prefix_and_residual_exact": scene_state_exact,
        "all_25_per_unit_nll_diagnostics_persisted": diagnostic_audit["unit_count"] == 25,
    }
    return {
        "checks": dict(checks),
        **checks,
        "passed": all(checks.values()),
        "priority_teacher_side_deficit": deficit,
        "source_priority_teacher_side_deficit": source_priority_deficit,
        "priority_teacher_side_deficit_improvement": improvement,
        "broad_nll_delta_from_update_zero": broad_nll - source_broad_nll,
        "training_scenes_only": True,
        "validation_qa_loaded": False,
        "chat_promotion_authorized": False,
        "independent_validation_selector_required": True,
    }


def v41_loss_values(
    *,
    settings: V41Settings,
    broad_nll: float,
    pair_correct_nll: float,
    side_hinge: float,
    cross_prefix_hinge: float,
    frozen_normalized_residual: float,
) -> tuple[float, float]:
    optimized = (
        settings.broad_nll_weight * broad_nll
        + settings.pair_correct_nll_weight * pair_correct_nll
        + settings.side_hinge_weight * side_hinge
        + settings.cross_prefix_flip_weight * cross_prefix_hinge
    )
    reported = optimized + settings.residual_penalty_weight * frozen_normalized_residual
    return optimized, reported


def _component_gradients(
    loss: torch.Tensor,
    parameters: Sequence[torch.nn.Parameter],
    *,
    retain_graph: bool,
) -> tuple[torch.Tensor, ...]:
    try:
        gradients = torch.autograd.grad(
            loss,
            tuple(parameters),
            retain_graph=retain_graph,
            create_graph=False,
            allow_unused=False,
        )
    except (RuntimeError, ValueError) as exc:
        raise V41GradientGuardFailure(
            {
                "schema_version": 1,
                "guard_stage": "component_autograd",
                "gradient_method": "torch.autograd.grad",
                "failure_reason": type(exc).__name__,
                "failure_message": str(exc),
            }
        ) from exc
    if len(gradients) != 1 or tuple(gradients[0].shape) != _QUERY_SHAPES[0]:
        raise V41GradientGuardFailure(
            {
                "schema_version": 1,
                "guard_stage": "component_autograd",
                "gradient_method": "torch.autograd.grad",
                "failure_reason": "escaped_exact_lora_b_surface",
                "observed_tensor_count": len(gradients),
                "observed_shapes": [list(value.shape) for value in gradients],
            }
        )
    return tuple(value.detach() for value in gradients)


def _gradient_vector(values: Sequence[torch.Tensor]) -> torch.Tensor:
    # MPS does not implement float64 tensors.  Move each detached gradient to
    # CPU first, then promote it for the high-precision dot/cosine audit.
    return torch.cat([value.detach().reshape(-1).cpu().double() for value in values])


def _gradient_state_sha256(values: Sequence[torch.Tensor]) -> str:
    """Hash an ordered gradient tuple without depending on its source device."""

    return tensor_state_sha256(
        {
            f"gradient_{index:03d}": value.detach().cpu().contiguous()
            for index, value in enumerate(values)
        }
    )


def raw_component_gradient_diagnostic(
    components: Mapping[str, Sequence[torch.Tensor]],
) -> tuple[tuple[torch.Tensor, ...], dict[str, Any]]:
    """Audit the unprojected objective gradient without rejecting conflicts.

    A negative directional dot product is precisely the condition V41 is
    designed to repair, so it is evidence rather than a failure here.  Only an
    invalid inventory, non-finite tensor, or zero total can stop execution at
    this stage.
    """

    required = ("broad", "answer", "side", "cross")
    if tuple(components) != required:
        raise ValueError("V41 component-gradient inventory or order changed")
    if any(len(components[name]) != 1 for name in required):
        raise ValueError("V41 component gradients differ from the one-tensor surface")
    total = tuple(sum(components[name][index] for name in required) for index in range(1))
    scene = tuple(components["side"][index] + components["cross"][index] for index in range(1))
    vectors = {
        **{name: _gradient_vector(components[name]) for name in required},
        "scene": _gradient_vector(scene),
        "total": _gradient_vector(total),
    }
    finite = {name: bool(torch.isfinite(value).all()) for name, value in vectors.items()}
    raw_norms = {name: float(value.norm()) for name, value in vectors.items()}
    norms: dict[str, float | None] = {
        name: value if math.isfinite(value) else None for name, value in raw_norms.items()
    }
    total_norm = norms["total"]
    if not all(finite.values()) or total_norm is None or total_norm <= 0.0:
        raise V41GradientGuardFailure(
            {
                "schema_version": 1,
                "guard_stage": "raw_component_direction",
                "gradient_method": "torch.autograd.grad_separate_components",
                "component_order": list(required),
                "component_norms": norms,
                "component_finite": finite,
                "directional_checks": {},
                "raw_guard_passed": False,
                "failure_reason": "nonfinite_component_or_zero_total",
                "guard_evaluated_before_clip_and_optimizer_step": True,
            }
        )
    directional: dict[str, dict[str, Any]] = {}
    raw_feasible = True
    for name in ("broad", "answer", "scene", "cross"):
        vector = vectors[name]
        norm = norms[name]
        assert norm is not None
        nonzero = norm > 0.0
        raw_dot = float(torch.dot(vectors["total"], vector)) if nonzero else None
        raw_cosine = (
            raw_dot / (total_norm * norm) if raw_dot is not None else None
        )
        dot_finite = raw_dot is None or math.isfinite(raw_dot)
        cosine_finite = raw_cosine is None or math.isfinite(raw_cosine)
        dot = raw_dot if dot_finite else None
        cosine = raw_cosine if cosine_finite else None
        check = not nonzero or (
            dot_finite
            and cosine_finite
            and dot is not None
            and cosine is not None
            and dot > 0.0
            and cosine > 0.0
        )
        directional[name] = {
            "nonzero": nonzero,
            "dot_with_total": dot,
            "cosine_with_total": cosine,
            "dot_finite": dot_finite,
            "cosine_finite": cosine_finite,
            "strictly_positive_if_nonzero": check,
        }
        raw_feasible = raw_feasible and check
    audit = {
        "schema_version": 1,
        "diagnostic_stage": "raw_component_direction_before_projection",
        "gradient_method": "torch.autograd.grad_separate_components",
        "component_order": list(required),
        "component_norms": norms,
        "component_finite": finite,
        "directional_checks": directional,
        "raw_total_state_sha256": _gradient_state_sha256(total),
        "raw_direction_already_feasible": raw_feasible,
        "conflicting_directions": [
            name
            for name, check in directional.items()
            if check["nonzero"] and not check["strictly_positive_if_nonzero"]
        ],
        "directional_conflicts_are_informational": True,
        "diagnostic_evaluated_before_projection_clip_and_optimizer_step": True,
    }
    return total, audit


_PROJECTION_MARGIN_SCALE = 1e-4
_PROJECTION_MINIMUM_BETA = 1e-12
_PROJECTION_RANK_ABSOLUTE_TOLERANCE = 1e-12
_PROJECTION_RANK_RELATIVE_TOLERANCE = 1e-10
_PROJECTION_KKT_ABSOLUTE_TOLERANCE = 1e-10
_PROJECTION_KKT_RELATIVE_TOLERANCE = 1e-8
_PROJECTION_DUAL_LAMBDA_LOWER_TOLERANCE = -1e-10
_PROJECTION_OBJECTIVE_TIE_RELATIVE_TOLERANCE = 1e-12
_PROJECTION_POSTCAST_BETA_FRACTION = 0.5
_PROJECTION_MAXIMUM_CORRECTION_RATIO = 0.25
_PROJECTION_MINIMUM_RAW_COSINE = 0.95
_PROJECTION_DIRECTION_NAMES = ("broad", "answer", "scene", "cross")


def _projection_tolerance(*values: float) -> float:
    scale = max(1.0, *(abs(value) for value in values if math.isfinite(value)))
    return (
        _PROJECTION_KKT_ABSOLUTE_TOLERANCE
        + _PROJECTION_KKT_RELATIVE_TOLERANCE * scale
    )


def _json_finite(value: float) -> float | None:
    return value if math.isfinite(value) else None


def _solve_projection_active_sets(
    matrix: torch.Tensor,
    bounds: torch.Tensor,
    solve_bounds: torch.Tensor,
    raw: torch.Tensor,
) -> dict[str, Any]:
    """Run one complete deterministic CPU-float64 active-set enumeration."""

    matrix = matrix.detach().clone().to(device="cpu", dtype=torch.float64)
    bounds = bounds.detach().clone().to(device="cpu", dtype=torch.float64)
    solve_bounds = solve_bounds.detach().clone().to(
        device="cpu", dtype=torch.float64
    )
    raw = raw.detach().clone().to(device="cpu", dtype=torch.float64)
    constraint_count = int(matrix.shape[0]) if matrix.ndim == 2 else 0
    if (
        matrix.ndim != 2
        or constraint_count not in {1, 2, 3, 4}
        or bounds.shape != (constraint_count,)
        or solve_bounds.shape != (constraint_count,)
    ):
        raise ValueError("V41 projection solver requires one to four constraints")
    gram = matrix @ matrix.T
    raw_constraints = matrix @ raw
    candidates: list[dict[str, Any]] = []
    feasible: list[tuple[float, int, torch.Tensor, torch.Tensor]] = []
    mask_count = 1 << constraint_count
    for mask in range(mask_count):
        indices = tuple(
            index for index in range(constraint_count) if mask & (1 << index)
        )
        multipliers = torch.zeros(constraint_count, dtype=torch.float64)
        solve_rank = 0
        solve_method = "empty_active_set"
        rejection: str | None = None
        if indices:
            selected = torch.tensor(indices, dtype=torch.long)
            selected_gram = gram.index_select(0, selected).index_select(1, selected)
            rhs = solve_bounds.index_select(
                0, selected
            ) - raw_constraints.index_select(0, selected)
            try:
                solve_rank = int(
                    torch.linalg.matrix_rank(
                        selected_gram,
                        atol=_PROJECTION_RANK_ABSOLUTE_TOLERANCE,
                        rtol=_PROJECTION_RANK_RELATIVE_TOLERANCE,
                    )
                )
            except RuntimeError as exc:
                rejection = f"matrix_rank_failed:{type(exc).__name__}"
            if rejection is None and solve_rank != len(indices):
                rejection = "linearly_dependent_active_subset"
            if rejection is None:
                try:
                    solved = torch.linalg.solve(selected_gram, rhs)
                    multipliers.index_copy_(0, selected, solved)
                    solve_method = "torch.linalg.solve_cpu_float64"
                except RuntimeError as exc:
                    rejection = f"active_gram_solve_failed:{type(exc).__name__}"
        if rejection is not None:
            candidates.append(
                {
                    "mask": mask,
                    "active_indices": list(indices),
                    "solve_method": solve_method,
                    "solve_rank": solve_rank,
                    "rejection_reason": rejection,
                    "multipliers": None,
                    "constraint_values": None,
                    "constraint_slacks": None,
                    "objective_half_squared_adjustment": None,
                    "minimum_primal_slack": None,
                    "minimum_dual_lambda": None,
                    "active_equality_residual_max": None,
                    "stationarity_residual_l2": None,
                    "complementarity_residual_max": None,
                    "kkt_checks": {
                        "independent_active_subset": False,
                        "finite": False,
                        "primal_feasible": False,
                        "dual_feasible": False,
                        "active_equality_feasible": False,
                        "stationarity": False,
                        "complementarity": False,
                    },
                    "feasible": False,
                }
            )
            continue
        candidate = raw + matrix.T @ multipliers
        values = matrix @ candidate
        slacks = values - bounds
        stationarity = candidate - raw - matrix.T @ multipliers
        adjustment = candidate - raw
        objective = 0.5 * float(torch.dot(adjustment, adjustment))
        active_residual = (
            max(abs(float(slacks[index])) for index in indices) if indices else 0.0
        )
        stationarity_residual = float(stationarity.norm())
        complementarity_residual = float((multipliers * slacks).abs().max())
        primal_tolerance = _projection_tolerance(
            float(values.abs().max()), float(bounds.abs().max())
        )
        equality_tolerance = _projection_tolerance(float(bounds.abs().max()))
        stationarity_tolerance = _projection_tolerance(
            float(candidate.norm()), float(raw.norm())
        )
        complementarity_tolerance = _projection_tolerance(
            float(multipliers.abs().max()), float(slacks.abs().max())
        )
        finite = bool(
            torch.isfinite(candidate).all()
            and torch.isfinite(multipliers).all()
            and torch.isfinite(values).all()
            and math.isfinite(objective)
        )
        kkt = {
            "independent_active_subset": True,
            "finite": finite,
            "primal_feasible": bool(torch.all(values >= bounds)),
            "dual_feasible": bool(
                float(multipliers.min())
                >= _PROJECTION_DUAL_LAMBDA_LOWER_TOLERANCE
            ),
            "active_equality_feasible": bool(
                active_residual <= equality_tolerance
            ),
            "stationarity": bool(
                stationarity_residual <= stationarity_tolerance
            ),
            "complementarity": bool(
                complementarity_residual <= complementarity_tolerance
            ),
        }
        is_feasible = all(kkt.values())
        candidates.append(
            {
                "mask": mask,
                "active_indices": list(indices),
                "solve_method": solve_method,
                "solve_rank": solve_rank,
                "rejection_reason": None,
                "multipliers": [_json_finite(float(value)) for value in multipliers],
                "constraint_values": [_json_finite(float(value)) for value in values],
                "constraint_slacks": [_json_finite(float(value)) for value in slacks],
                "objective_half_squared_adjustment": _json_finite(objective),
                "minimum_primal_slack": _json_finite(float(slacks.min())),
                "minimum_dual_lambda": _json_finite(float(multipliers.min())),
                "active_equality_residual_max": _json_finite(active_residual),
                "stationarity_residual_l2": _json_finite(stationarity_residual),
                "complementarity_residual_max": _json_finite(
                    complementarity_residual
                ),
                "kkt_tolerances": {
                    "primal": _json_finite(primal_tolerance),
                    "dual_lambda_lower": _PROJECTION_DUAL_LAMBDA_LOWER_TOLERANCE,
                    "active_equality": _json_finite(equality_tolerance),
                    "stationarity": _json_finite(stationarity_tolerance),
                    "complementarity": _json_finite(complementarity_tolerance),
                },
                "kkt_checks": kkt,
                "feasible": is_feasible,
            }
        )
        if is_feasible:
            feasible.append((objective, mask, candidate, multipliers))
    if not feasible:
        return {
            "feasible": False,
            "candidate_audits": candidates,
            "feasible_candidate_count": 0,
            "gram": gram,
            "raw_constraints": raw_constraints,
        }
    minimum = min(value[0] for value in feasible)
    tied = [
        value
        for value in feasible
        if math.isclose(
            value[0],
            minimum,
            rel_tol=_PROJECTION_OBJECTIVE_TIE_RELATIVE_TOLERANCE,
            abs_tol=0.0,
        )
    ]
    objective, mask, direction, multipliers = min(tied, key=lambda value: value[1])
    return {
        "feasible": True,
        "candidate_audits": candidates,
        "feasible_candidate_count": len(feasible),
        "gram": gram,
        "raw_constraints": raw_constraints,
        "selected_objective": objective,
        "selected_mask": mask,
        "selected_direction": direction,
        "selected_multipliers": multipliers,
    }


def project_gradient_to_feasible_descent(
    components: Mapping[str, Sequence[torch.Tensor]],
) -> tuple[tuple[torch.Tensor, ...], dict[str, Any]]:
    """Return the terminal-authorized, doubly solved projected gradient."""

    required = ("broad", "answer", "side", "cross")
    if tuple(components) != required or any(
        len(components[name]) != 1 for name in required
    ):
        raise V41GradientGuardFailure(
            {
                "schema_version": 1,
                "guard_stage": "projection_input",
                "failure_reason": "component_inventory_changed",
            }
        )
    source = components["broad"][0]
    if any(
        value[0].device != source.device
        or value[0].dtype != source.dtype
        or value[0].shape != source.shape
        for value in components.values()
    ):
        raise V41GradientGuardFailure(
            {
                "schema_version": 1,
                "guard_stage": "projection_input",
                "failure_reason": "component_device_dtype_or_shape_changed",
            }
        )
    cpu = {
        name: components[name][0]
        .detach()
        # MPS cannot represent float64.  Complete the device transfer before
        # promoting on CPU; a combined MPS->CPU+float64 conversion can yield
        # nonfinite values without raising on some PyTorch/macOS builds.
        .cpu()
        .double()
        .reshape(-1)
        .clone()
        for name in required
    }
    cpu["scene"] = cpu["side"] + cpu["cross"]
    raw = cpu["broad"] + cpu["answer"] + cpu["side"] + cpu["cross"]
    raw_source = tuple(
        sum(components[name][index] for name in required) for index in range(1)
    )
    vectors = {**cpu, "raw_total": raw}
    finite = {name: bool(torch.isfinite(value).all()) for name, value in vectors.items()}
    norms_raw = {name: float(value.norm()) for name, value in vectors.items()}
    norms = {name: _json_finite(value) for name, value in norms_raw.items()}
    raw_norm_value = norms["raw_total"]
    if (
        not all(finite.values())
        or raw_norm_value is None
        or float(raw_norm_value) <= 1e-12
    ):
        raise V41GradientGuardFailure(
            {
                "schema_version": 1,
                "guard_stage": "projection_input",
                "failure_reason": "nonfinite_component_or_too_small_raw_total",
                "component_finite": finite,
                "component_norms": norms,
                "raw_total_norm_minimum_exclusive": 1e-12,
            }
        )
    raw_norm = float(raw_norm_value)
    active_names = tuple(
        name
        for name in _PROJECTION_DIRECTION_NAMES
        if float(norms[name]) > 0.0
    )
    inactive_names = tuple(
        name
        for name in _PROJECTION_DIRECTION_NAMES
        if float(norms[name]) == 0.0
    )
    if not active_names:
        raise V41GradientGuardFailure(
            {
                "schema_version": 1,
                "guard_stage": "projection_input",
                "failure_reason": "no_active_nonzero_constraint_direction",
                "component_finite": finite,
                "component_norms": norms,
                "inactive_constraint_direction_names": list(inactive_names),
            }
        )
    matrix = torch.stack(
        [cpu[name] / float(norms[name]) for name in active_names]
    )
    beta = max(_PROJECTION_MINIMUM_BETA, _PROJECTION_MARGIN_SCALE * raw_norm)
    active_count = len(active_names)
    mask_count = 1 << active_count
    bounds = torch.full((active_count,), beta, dtype=torch.float64)
    gram = matrix @ matrix.T
    raw_constraints = matrix @ raw
    component_hashes = {
        name: tensor_state_sha256({name: cpu[name]})
        for name in (*required, "scene")
    }
    raw_hash = tensor_state_sha256({"raw_total": raw})
    raw_solver_feasible = bool(torch.all(raw_constraints >= bounds))
    solve_safety_delta = (
        0.0 if raw_solver_feasible else _PROJECTION_KKT_ABSOLUTE_TOLERANCE
    )
    solve_bounds = bounds + solve_safety_delta
    input_hash = tensor_state_sha256(
        {
            **{f"component_{name}": cpu[name] for name in (*required, "scene")},
            "raw_total": raw,
            "normalized_constraint_matrix": matrix,
            "bounds": bounds,
            "active_solve_bounds": solve_bounds,
            "gram": gram,
        }
    )
    try:
        first = _solve_projection_active_sets(
            matrix.clone(), bounds.clone(), solve_bounds.clone(), raw.clone()
        )
        second = _solve_projection_active_sets(
            matrix.clone(), bounds.clone(), solve_bounds.clone(), raw.clone()
        )
    except (RuntimeError, TypeError, ValueError) as exc:
        raise V41GradientGuardFailure(
            {
                "schema_version": 1,
                "guard_stage": "cpu_float64_active_set_projection",
                "failure_reason": "cpu_projection_solver_operation_failed",
                "failure_type": type(exc).__name__,
                "failure_message": str(exc),
                "solver_input_state_sha256": input_hash,
                "raw_total_norm": raw_norm,
                "beta": beta,
                "active_solve_safety_delta": solve_safety_delta,
                "normalized_constraint_gram_matrix": [
                    [_json_finite(float(value)) for value in row] for row in gram
                ],
                "raw_constraint_values": [
                    _json_finite(float(value)) for value in raw_constraints
                ],
                "projection_feasible": False,
            }
        ) from exc
    base_audit: dict[str, Any] = {
        "schema_version": 1,
        "guard_stage": "cpu_float64_active_set_projection",
        "solver_device": "cpu",
        "solver_dtype": "torch.float64",
        "authorization_revision": 3,
        "source_device": str(source.device),
        "source_dtype": str(source.dtype),
        "source_shape": list(source.shape),
        "component_order": list(required),
        "constraint_direction_order": list(_PROJECTION_DIRECTION_NAMES),
        "active_constraint_direction_names": list(active_names),
        "inactive_constraint_direction_names": list(inactive_names),
        "active_constraint_count": active_count,
        "inactive_constraint_count": len(inactive_names),
        "inactive_constraint_policy": (
            "exact_zero_norm_recorded_first_order_satisfied_without_normalization"
        ),
        "active_constraint_rule": "finite_l2_norm_strictly_greater_than_zero",
        "raw_total_norm_minimum_exclusive": 1e-12,
        "standalone_side_is_not_a_constraint": True,
        "component_finite": finite,
        "weighted_component_norms": norms,
        "weighted_component_state_sha256": component_hashes,
        "raw_total_norm": raw_norm,
        "raw_total_state_sha256": raw_hash,
        "solver_input_state_sha256": input_hash,
        "beta": beta,
        "active_solve_safety_delta": solve_safety_delta,
        "active_solve_beta": beta + solve_safety_delta,
        "active_solve_safety_delta_within_kkt_absolute_tolerance": (
            solve_safety_delta <= _PROJECTION_KKT_ABSOLUTE_TOLERANCE
        ),
        "candidate_kkt_residuals_computed_against_original_beta": True,
        "beta_formula": "max(1e-12, 1e-4 * l2(g_raw))",
        "normalized_constraint_gram_matrix": [
            [_json_finite(float(value)) for value in row] for row in gram
        ],
        "raw_constraint_values": [
            _json_finite(float(value)) for value in raw_constraints
        ],
        "rank_absolute_tolerance": _PROJECTION_RANK_ABSOLUTE_TOLERANCE,
        "rank_relative_tolerance": _PROJECTION_RANK_RELATIVE_TOLERANCE,
        "dual_lambda_lower_tolerance": _PROJECTION_DUAL_LAMBDA_LOWER_TOLERANCE,
        "kkt_absolute_tolerance": _PROJECTION_KKT_ABSOLUTE_TOLERANCE,
        "kkt_relative_tolerance": _PROJECTION_KKT_RELATIVE_TOLERANCE,
        "objective_tie_relative_tolerance": (
            _PROJECTION_OBJECTIVE_TIE_RELATIVE_TOLERANCE
        ),
        "enumerated_mask_order": list(range(mask_count)),
        "enumerated_subset_count": mask_count,
        "enumerated_mask_count_formula": "2 ** active_constraint_count",
        "candidate_audits": first["candidate_audits"],
        "feasible_candidate_count": first["feasible_candidate_count"],
    }
    if not first["feasible"] or not second["feasible"]:
        raise V41GradientGuardFailure(
            {
                **base_audit,
                "failure_reason": "closed_halfspace_qp_has_no_feasible_candidate",
                "projection_feasible": False,
                "second_solve_feasible": bool(second["feasible"]),
            }
        )
    first_direction = first["selected_direction"]
    second_direction = second["selected_direction"]
    first_lambdas = first["selected_multipliers"]
    second_lambdas = second["selected_multipliers"]
    first_hash = tensor_state_sha256({"projected": first_direction})
    second_hash = tensor_state_sha256({"projected": second_direction})
    replay = {
        "solve_twice_from_independent_cpu_float64_clones": True,
        "selected_mask_exact": first["selected_mask"] == second["selected_mask"],
        "selected_lambdas_bit_exact": torch.equal(first_lambdas, second_lambdas),
        "projected_direction_bit_exact": torch.equal(
            first_direction, second_direction
        ),
        "projected_direction_sha256_exact": first_hash == second_hash,
        "first_projected_direction_sha256": first_hash,
        "second_projected_direction_sha256": second_hash,
    }
    replay_passed = all(
        value is True for key, value in replay.items() if key.endswith(("exact", "clones"))
    )
    if not replay_passed:
        raise V41GradientGuardFailure(
            {
                **base_audit,
                "failure_reason": "independent_double_solve_replay_mismatch",
                "double_solve_replay": replay,
                "projection_feasible": False,
            }
        )
    projected_cpu = first_direction
    projected_cpu_norm = float(projected_cpu.norm())
    cpu_margins = matrix @ projected_cpu
    cpu_directional: dict[str, Any] = {}
    cpu_directional_passed = math.isfinite(projected_cpu_norm) and projected_cpu_norm > 0
    for index, name in enumerate(active_names):
        margin = float(cpu_margins[index])
        dot = margin * float(norms[name])
        cosine = margin / projected_cpu_norm
        passed = (
            math.isfinite(margin)
            and math.isfinite(dot)
            and math.isfinite(cosine)
            and margin >= beta
            and dot > 0.0
            and cosine > 0.0
        )
        cpu_directional[name] = {
            "active": True,
            "inactive_reason": None,
            "normalized_constraint_margin": _json_finite(margin),
            "required_minimum": beta,
            "dot": _json_finite(dot),
            "cosine": _json_finite(cosine),
            "passed": passed,
        }
        cpu_directional_passed = cpu_directional_passed and passed
    for name in inactive_names:
        cpu_directional[name] = {
            "normalized_constraint_margin": None,
            "required_minimum": None,
            "dot": 0.0,
            "cosine": None,
            "active": False,
            "inactive_reason": "exact_zero_norm_first_order_satisfied",
            "passed": True,
        }
    adjustment = projected_cpu - raw
    correction_ratio = float(adjustment.norm()) / raw_norm
    projected_raw_cosine = float(
        torch.dot(projected_cpu, raw) / (projected_cpu_norm * raw_norm)
    )
    cpu_global_passed = (
        cpu_directional_passed
        and math.isfinite(correction_ratio)
        and correction_ratio <= _PROJECTION_MAXIMUM_CORRECTION_RATIO
        and math.isfinite(projected_raw_cosine)
        and projected_raw_cosine >= _PROJECTION_MINIMUM_RAW_COSINE
    )
    selected_mask = int(first["selected_mask"])
    projected = (
        raw_source[0]
        if selected_mask == 0 and raw_solver_feasible
        else projected_cpu.reshape(source.shape).to(
            device=source.device, dtype=source.dtype
        )
    )
    postcast = projected.detach().cpu().double().reshape(-1)
    postcast_norm = float(postcast.norm())
    postcast_margins = matrix @ postcast
    postcast_directional: dict[str, Any] = {}
    postcast_passed = math.isfinite(postcast_norm) and postcast_norm > 0.0
    for index, name in enumerate(active_names):
        margin = float(postcast_margins[index])
        dot = margin * float(norms[name])
        cosine = margin / postcast_norm
        passed = (
            math.isfinite(margin)
            and math.isfinite(dot)
            and math.isfinite(cosine)
            and margin >= beta / 2.0
            and dot > 0.0
            and cosine > 0.0
        )
        postcast_directional[name] = {
            "active": True,
            "inactive_reason": None,
            "normalized_constraint_margin": _json_finite(margin),
            "required_minimum": beta / 2.0,
            "dot": _json_finite(dot),
            "cosine": _json_finite(cosine),
            "passed": passed,
        }
        postcast_passed = postcast_passed and passed
    for name in inactive_names:
        postcast_directional[name] = {
            "normalized_constraint_margin": None,
            "required_minimum": None,
            "dot": 0.0,
            "cosine": None,
            "active": False,
            "inactive_reason": "exact_zero_norm_first_order_satisfied",
            "passed": True,
        }
    postcast_adjustment = postcast - raw
    postcast_correction_ratio = float(postcast_adjustment.norm()) / raw_norm
    postcast_raw_cosine = float(
        torch.dot(postcast, raw) / (postcast_norm * raw_norm)
    )
    postcast_global_passed = (
        postcast_passed
        and math.isfinite(postcast_correction_ratio)
        and postcast_correction_ratio <= _PROJECTION_MAXIMUM_CORRECTION_RATIO
        and math.isfinite(postcast_raw_cosine)
        and postcast_raw_cosine >= _PROJECTION_MINIMUM_RAW_COSINE
    )
    raw_source_replay = raw_source[0].detach().cpu().double().reshape(-1)
    raw_source_replay_norm = float(raw_source_replay.norm())
    if not math.isfinite(raw_source_replay_norm) or raw_source_replay_norm <= 0.0:
        raise V41GradientGuardFailure(
            {
                **base_audit,
                "failure_reason": "source_dtype_raw_total_is_zero_or_nonfinite",
                "projection_feasible": False,
            }
        )
    applied_correction_ratio = float(
        (postcast - raw_source_replay).norm()
    ) / raw_source_replay_norm
    applied_raw_cosine = float(
        torch.dot(postcast, raw_source_replay)
        / (postcast_norm * raw_source_replay_norm)
    )
    applied_source_safety_passed = (
        math.isfinite(applied_correction_ratio)
        and applied_correction_ratio <= _PROJECTION_MAXIMUM_CORRECTION_RATIO
        and math.isfinite(applied_raw_cosine)
        and applied_raw_cosine >= _PROJECTION_MINIMUM_RAW_COSINE
    )
    raw_source_hash = _gradient_state_sha256(raw_source)
    postcast_hash = _gradient_state_sha256((projected,))
    raw_bit_exact = (
        not raw_solver_feasible
        or (selected_mask == 0 and postcast_hash == raw_source_hash)
    )
    audit = {
        **base_audit,
        "selected_mask": selected_mask,
        "selected_active_constraints": [
            name
            for index, name in enumerate(active_names)
            if selected_mask & (1 << index)
        ],
        "selected_lambdas": [
            _json_finite(float(value)) for value in first_lambdas
        ],
        "projection_objective": _json_finite(float(first["selected_objective"])),
        "double_solve_replay": replay,
        "cpu_projected_direction_sha256": first_hash,
        "cpu_projected_direction_norm": _json_finite(projected_cpu_norm),
        "cpu_directional_attestation": cpu_directional,
        "cpu_correction_ratio": _json_finite(correction_ratio),
        "cpu_projected_raw_cosine": _json_finite(projected_raw_cosine),
        "cpu_solution_safety_passed": cpu_global_passed,
        "post_device_cast_state_sha256": postcast_hash,
        "post_device_cast_norm": _json_finite(postcast_norm),
        "post_device_cast_directional_attestation": postcast_directional,
        "post_device_cast_correction_ratio": _json_finite(
            postcast_correction_ratio
        ),
        "post_device_cast_projected_raw_cosine": _json_finite(
            postcast_raw_cosine
        ),
        "post_device_cast_safety_passed": postcast_global_passed,
        "applied_vs_source_dtype_raw_correction_ratio": _json_finite(
            applied_correction_ratio
        ),
        "applied_vs_source_dtype_raw_cosine": _json_finite(applied_raw_cosine),
        "applied_vs_source_dtype_raw_safety_passed": applied_source_safety_passed,
        "maximum_correction_ratio": _PROJECTION_MAXIMUM_CORRECTION_RATIO,
        "minimum_projected_raw_cosine": _PROJECTION_MINIMUM_RAW_COSINE,
        "raw_solver_feasible": raw_solver_feasible,
        "raw_source_state_sha256": raw_source_hash,
        "raw_feasible_returned_bit_exact": raw_bit_exact,
        "projection_applied": selected_mask != 0,
        "projection_feasible": (
            replay_passed
            and cpu_global_passed
            and postcast_global_passed
            and applied_source_safety_passed
            and raw_bit_exact
        ),
    }
    if not audit["projection_feasible"]:
        audit["failure_reason"] = "projection_or_postcast_safety_attestation_failed"
        raise V41GradientGuardFailure(audit)
    return (projected,), audit


def clip_direction_attestation(
    *,
    parameters: Sequence[torch.nn.Parameter],
    projected_total: Sequence[torch.Tensor],
    components: Mapping[str, Sequence[torch.Tensor]],
    projection_attestation: Mapping[str, Any],
    clip_norm: float,
) -> dict[str, Any]:
    """Apply the authorized scalar global clip and prove direction preservation."""

    try:
        for parameter, gradient in zip(parameters, projected_total, strict=True):
            parameter.grad = gradient.clone()
        preclip = float(torch.nn.utils.clip_grad_norm_(parameters, clip_norm))
        clipped = tuple(
            parameter.grad.detach()
            for parameter in parameters
            if parameter.grad is not None
        )
        projected_vector = _gradient_vector(projected_total)
        clipped_vector = _gradient_vector(clipped)
        projected_norm = float(projected_vector.norm())
        clipped_norm = float(clipped_vector.norm())
        projected_hash = _gradient_state_sha256(projected_total)
        clipped_hash = _gradient_state_sha256(clipped)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise V41GradientGuardFailure(
            {
                "schema_version": 1,
                "guard_stage": "scalar_global_clip",
                "failure_reason": "clip_operation_or_attestation_failed",
                "failure_type": type(exc).__name__,
                "failure_message": str(exc),
                "optimizer_step_executed": False,
            }
        ) from exc
    expected_projected_hash = projection_attestation.get(
        "post_device_cast_state_sha256"
    )
    expected_projected_norm = projection_attestation.get("post_device_cast_norm")
    projection_input_linked = (
        projected_hash == expected_projected_hash
        and isinstance(expected_projected_norm, (int, float))
        and float(expected_projected_norm) == projected_norm
    )
    if (
        not math.isfinite(preclip)
        or not math.isfinite(projected_norm)
        or not math.isfinite(clipped_norm)
        or projected_norm <= 0.0
        or clipped_norm <= 0.0
        or not projection_input_linked
    ):
        raise V41GradientGuardFailure(
            {
                "schema_version": 1,
                "guard_stage": "scalar_global_clip",
                "clip_kind": "single_global_scalar_l2_clip",
                "clip_norm": clip_norm,
                "torch_reported_preclip_norm": preclip if math.isfinite(preclip) else None,
                "projected_total_norm": (
                    projected_norm if math.isfinite(projected_norm) else None
                ),
                "clipped_total_norm": clipped_norm if math.isfinite(clipped_norm) else None,
                "projected_input_state_sha256": projected_hash,
                "expected_projected_input_state_sha256": expected_projected_hash,
                "projection_input_linked": projection_input_linked,
                "scalar_clip_direction_preserved": False,
                "failure_reason": "invalid_or_unlinked_projected_clip_input",
            }
        )
    raw_scalar = clipped_norm / projected_norm
    raw_direction_cosine = float(
        torch.dot(projected_vector, clipped_vector)
        / (projected_norm * clipped_norm)
    )
    scalar = raw_scalar if math.isfinite(raw_scalar) else None
    direction_cosine = (
        raw_direction_cosine if math.isfinite(raw_direction_cosine) else None
    )
    scalar_residual = float(
        (clipped_vector - projected_vector * float(raw_scalar)).abs().max()
    )
    scalar_residual_tolerance = 1e-6 * max(1.0, clipped_norm)
    guarded_vectors = {
        name: _gradient_vector(components[name])
        for name in ("broad", "answer", "cross")
    }
    guarded_vectors["scene"] = _gradient_vector(
        components["side"]
    ) + _gradient_vector(components["cross"])
    directional: dict[str, dict[str, Any]] = {}
    expected_active_names = tuple(
        projection_attestation.get("active_constraint_direction_names", ())
    )
    expected_inactive_names = tuple(
        projection_attestation.get("inactive_constraint_direction_names", ())
    )
    observed_active_names = tuple(
        name for name in _PROJECTION_DIRECTION_NAMES if guarded_vectors[name].norm() > 0
    )
    activity_linked = (
        expected_active_names == observed_active_names
        and expected_inactive_names
        == tuple(
            name
            for name in _PROJECTION_DIRECTION_NAMES
            if name not in observed_active_names
        )
    )
    passed = (
        scalar is not None
        and scalar > 0.0
        and direction_cosine is not None
        and direction_cosine >= 0.9999999
        and math.isfinite(scalar_residual)
        and scalar_residual <= scalar_residual_tolerance
        and activity_linked
    )
    for name in ("broad", "answer", "scene", "cross"):
        vector = guarded_vectors[name]
        norm = float(vector.norm())
        raw_dot = float(torch.dot(clipped_vector, vector)) if norm > 0.0 else None
        raw_cosine = raw_dot / (clipped_norm * norm) if raw_dot is not None else None
        dot_finite = raw_dot is None or math.isfinite(raw_dot)
        cosine_finite = raw_cosine is None or math.isfinite(raw_cosine)
        dot = raw_dot if dot_finite else None
        cosine = raw_cosine if cosine_finite else None
        check = norm == 0.0 or (
            math.isfinite(norm)
            and dot_finite
            and cosine_finite
            and dot is not None
            and cosine is not None
            and dot > 0.0
            and cosine > 0.0
        )
        directional[name] = {
            "active": norm > 0.0,
            "inactive_reason": (
                None if norm > 0.0 else "exact_zero_norm_first_order_satisfied"
            ),
            "dot_with_clipped_total": dot,
            "cosine_with_clipped_total": cosine,
            "dot_finite": dot_finite,
            "cosine_finite": cosine_finite,
            "strictly_positive_if_nonzero": check,
        }
        passed = passed and check
    audit = {
        "schema_version": 1,
        "guard_stage": "scalar_global_clip",
        "clip_kind": "single_global_scalar_l2_clip",
        "clip_norm": clip_norm,
        "torch_reported_preclip_norm": preclip,
        "projected_total_norm": projected_norm,
        "clipped_total_norm": clipped_norm,
        "projected_input_state_sha256": projected_hash,
        "clipped_output_state_sha256": clipped_hash,
        "expected_projected_input_state_sha256": expected_projected_hash,
        "projection_input_linked": projection_input_linked,
        "active_constraint_direction_names": list(expected_active_names),
        "inactive_constraint_direction_names": list(expected_inactive_names),
        "constraint_activity_linked": activity_linked,
        "observed_scalar": scalar,
        "projected_to_clipped_cosine": direction_cosine,
        "minimum_projected_to_clipped_cosine": 0.9999999,
        "scalar_multiplication_residual_max": _json_finite(scalar_residual),
        "scalar_multiplication_residual_tolerance": scalar_residual_tolerance,
        "directional_checks": directional,
        "scalar_clip_direction_preserved": passed,
    }
    if not passed:
        raise V41GradientGuardFailure(audit)
    return audit


def persist_gradient_guard_failure(
    output: Path,
    *,
    optimizer_step: int,
    audit: Mapping[str, Any],
) -> Path:
    """Persist fail-stop evidence atomically without writing a checkpoint."""

    path = output / f"guard_failure_update_{optimizer_step:03d}.json"
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"V41 guard failure evidence already exists: {path}")
    payload = {
        "schema_version": 1,
        "artifact": "v41_pre_step_gradient_guard_failure",
        "optimizer_step_not_executed": optimizer_step,
        "optimizer_step_executed": False,
        "checkpoint_written": False,
        "validation_qa_loaded": False,
        "oracle_environment_files_loaded": False,
        "audit": dict(audit),
    }
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def v41_optimizer(bundle: V30Bundle, settings: V41Settings) -> torch.optim.SGD:
    parameters = freeze_for_v41(bundle)
    optimizer = torch.optim.SGD(
        [
            {
                "name": f"lora_banks.{_QUERY_BANK}.adapters.1",
                "params": parameters,
                "parameter_names": list(_QUERY_PARAMETER_NAMES),
                "lr": settings.learning_rate,
                "weight_decay": settings.weight_decay,
                "momentum": 0.0,
                "dampening": 0.0,
                "nesterov": False,
            }
        ],
        foreach=False,
        fused=False,
    )
    assert_v41_trainable_surface(bundle, optimizer=optimizer)
    if optimizer.state:
        raise RuntimeError("V41 momentum-free SGD state is not fresh")
    return optimizer


def _fresh_sgd_group_defaults() -> dict[str, Any]:
    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD(
        [
            {
                "name": f"lora_banks.{_QUERY_BANK}.adapters.1",
                "params": [parameter],
                "parameter_names": ["probe"],
                "lr": 0.003,
                "weight_decay": 0.0,
                "momentum": 0.0,
                "dampening": 0.0,
                "nesterov": False,
            }
        ],
        foreach=False,
        fused=False,
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
    groups = payload.get("param_groups") if isinstance(payload, Mapping) else None
    state = payload.get("state") if isinstance(payload, Mapping) else None
    if not isinstance(groups, list) or len(groups) != 1 or not isinstance(state, Mapping):
        raise ValueError("V41 optimizer must contain exactly one SGD group")
    group = _mapping(groups[0], "V41 optimizer group")
    defaults = _fresh_sgd_group_defaults()
    observed_defaults = {
        key: value
        for key, value in group.items()
        if key not in {"name", "params", "parameter_names"}
    }
    if (
        group.get("name") != f"lora_banks.{_QUERY_BANK}.adapters.1"
        or group.get("params") != [0]
        or group.get("parameter_names") != list(_QUERY_PARAMETER_NAMES)
        or set(group) != {"name", "params", "parameter_names", *defaults}
        or observed_defaults != defaults
    ):
        raise ValueError("V41 optimizer group identity/order/settings changed")
    if state:
        raise ValueError("V41 momentum-free SGD must have no per-parameter state")
    for tensor_name in _QUERY_PARAMETER_NAMES:
        tensor = tensors.get(tensor_name)
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"V41 adapter tensor is absent or invalid: {tensor_name}")
        if tuple(tensor.shape) != _QUERY_SHAPES[_QUERY_PARAMETER_NAMES.index(tensor_name)]:
            raise ValueError(f"V41 target tensor shape changed: {tensor_name}")
    return {
        "group_count": 1,
        "parameter_states_inspected": list(_QUERY_PARAMETER_NAMES),
        "moment_tensor_count": 0,
        "optimizer_step": expected_step,
        "exact_parameter_order_verified": True,
        "exact_sgd_group_schema_verified": True,
        "sgd_group_defaults": defaults,
        "momentum_free_stateless_sgd_verified": True,
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
    if (
        optimizer_path.is_symlink()
        or not optimizer_path.is_file()
        or manifest_path.is_symlink()
        or not manifest_path.is_file()
    ):
        raise FileNotFoundError("V41 optimizer or integrity manifest is missing or aliased")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_manifest = {
        "schema_version": 1,
        "artifact": "v41_optimizer_integrity_manifest",
        "optimizer_step": expected_step,
        "optimizer_filename": "optimizer.pt",
        "optimizer_sha256": _sha256(optimizer_path),
    }
    if manifest != expected_manifest:
        raise ValueError("V41 optimizer integrity manifest or file hash changed")
    payload = torch.load(optimizer_path, map_location="cpu", weights_only=True)
    return {
        **_optimizer_payload_audit(payload, expected_step=expected_step, tensors=tensors),
        "optimizer_sha256": expected_manifest["optimizer_sha256"],
        "self_hash_linkage_verified": True,
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
    prefix_replay: Mapping[str, Any],
    update_zero_attestation: Mapping[str, Any],
    source_pair_metrics: Mapping[str, Any],
    source_per_unit_nll: Sequence[Mapping[str, Any]],
    source_broad_nll: float,
    source_greedy_metrics: Mapping[str, Any],
    source_residual: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
    optimizer_step: int,
    bundle: V30Bundle,
    surface: Mapping[str, Any],
    gate8: Mapping[str, Any] | None,
    gate16: Mapping[str, Any] | None,
    gate41: Mapping[str, Any] | None,
) -> dict[str, Any]:
    contract = v41_contract(config)
    projection_history_attestation = validate_v41_projection_history(
        history, contract
    )
    metadata = copy.deepcopy(dict(source_metadata))
    metadata.update(
        {
            "schema_version": 1,
            "config_hash": config_hash(dict(config)),
            "optimizer_step": optimizer_step,
            "epoch": optimizer_step,
            "history": [dict(row) for row in history],
            "block_cross_residual_state_sha256": contract.core_state_sha256,
            "question_dependent_scene_processing": False,
            "lora": lora_banks_checkpoint_contract(
                lora_banks_settings(config),
                lora_banks_optimizer_settings(config, lora_banks_settings(config)),
                bundle.lora_installation.parameter_counts,
            ),
            **bundle.lora_installation.checkpoint_metadata(),
        }
    )
    metadata["v41_projected_gradient"] = {
        "schema_version": 1,
        "optimizer_step": optimizer_step,
        "conditional_v40_terminal_gate": {
            "path": terminal["path"],
            "sha256": terminal["sha256"],
        },
        "conditional_authorization": dict(terminal["authorization"]),
        "authorized_output_root": terminal["authorized_output_root"],
        "conditional_v41_retry1_terminal_gate": None
        if terminal["retry1"] is None
        else {
            "path": terminal["retry1"]["path"],
            "sha256": terminal["retry1"]["sha256"],
        },
        "retry1_conditional_authorization": None
        if terminal["retry1"] is None
        else dict(terminal["retry1"]["authorization"]),
        "retry1_predecessor_attestation": None
        if terminal["retry1"] is None
        else dict(terminal["retry1"]["predecessor_attestation"]),
        "source_checkpoint": str(contract.source_checkpoint),
        "source_v40_u0_tensor_state_sha256": contract.source_tensor_state_sha256,
        "update_zero_tensor_state_sha256": contract.hybrid_tensor_state_sha256,
        "hybrid_v23_state_sha256": contract.hybrid_v23_state_sha256,
        "source_target_lora_b_state_sha256": contract.query_source_state_sha256,
        "source_block_core_state_sha256": contract.core_state_sha256,
        "frozen_excluding_query_state_sha256": frozen_v41_state_sha256(bundle),
        "complete_v28_bank_state_sha256": _query_bank(bundle).state_sha256(),
        "target_lora_b_state_sha256": target_v41_state_sha256(bundle),
        "source_audit": dict(source_audit),
        "source_optimizer_states_loaded": False,
        "source_optimizer_files_opened": False,
        "fresh_projected_gradient_sgd": True,
        "trainable_surface": dict(surface),
        "schedule": dict(schedule_audit),
        "scene_cache": _deterministic_cache_audit(cache_audit),
        "train_qa_dataset": dict(qa_audit),
        "prefix_replay_attestation": dict(prefix_replay),
        "update_zero_attestation": dict(update_zero_attestation),
        "source_pair_metrics": dict(source_pair_metrics),
        "source_per_unit_nll_diagnostics": [dict(row) for row in source_per_unit_nll],
        "source_broad_train_nll": source_broad_nll,
        "source_train_greedy_metrics": dict(source_greedy_metrics),
        "source_residual_diagnostics": dict(source_residual),
        "projection_history_attestation": projection_history_attestation,
        "update8_train_only_gate": None if gate8 is None else dict(gate8),
        "update16_train_only_gate": None if gate16 is None else dict(gate16),
        "update41_train_only_gate": None if gate41 is None else dict(gate41),
        "validation_qa_loaded": False,
        "oracle_environment_files_loaded": False,
        "deferred_final_scene_ids_loaded": [],
        "question_dependent_scene_processing": False,
        "question_dependent_retrieval": False,
        "independent_selector_required": True,
    }
    lora_contract = _mapping(metadata.get("lora"), "V41 lora contract")
    banks = lora_contract.get("banks")
    if not isinstance(banks, list):
        raise TypeError("V41 lora contract banks must be a list")
    trainable = {str(bank["name"]): bank.get("trainable") for bank in banks}
    if (
        trainable.get(_QUERY_BANK) is not False
        or trainable.get(_V23_BANK) is not False
        or metadata.get("lora_trainable_parameter_count") != 0
        or bundle.trainable_bank_name is not None
    ):
        raise RuntimeError("V41 runtime metadata advertises the wrong trainable surface")
    return metadata


def _save(
    path: Path,
    *,
    bundle: V30Bundle,
    metadata: dict[str, Any],
    optimizer: torch.optim.Optimizer | None,
) -> None:
    if path.is_symlink() or (path.exists() and not path.is_dir()):
        raise ValueError(f"V41 checkpoint destination is unsafe: {path}")
    save_adapter_checkpoint(path, bundle.checkpoint_modules, metadata)
    if optimizer is not None:
        save_optimizer_checkpoint(path, optimizer)
        optimizer_path = path / "optimizer.pt"
        manifest_path = path / OPTIMIZER_AUDIT_FILENAME
        payload = {
            "schema_version": 1,
            "artifact": "v41_optimizer_integrity_manifest",
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
        raise RuntimeError("V41 runtime metadata sanitizer changed during save")


def replay_v41_gates(
    metadata: Mapping[str, Any], contract: V41Contract
) -> tuple[Mapping[str, Any] | None, Mapping[str, Any] | None, Mapping[str, Any] | None]:
    history = metadata.get("history")
    if not isinstance(history, list) or not history:
        raise ValueError("V41 checkpoint history is absent")
    stage = _mapping(metadata.get("v41_projected_gradient"), "V41 stage")
    source_broad = float(stage["source_broad_train_nll"])
    update_zero = _mapping(
        stage["update_zero_attestation"], "V41 update-zero attestation"
    )
    behavioral_baseline = _mapping(
        update_zero["behavioral_baseline"], "V41 update-zero behavioral baseline"
    )
    observed_baseline = _mapping(
        behavioral_baseline["observed"], "V41 update-zero observed baseline"
    )
    source_deficit = float(
        observed_baseline["priority_combined_side_deficit"]
    )
    gate8: Mapping[str, Any] | None = None
    gate16: Mapping[str, Any] | None = None
    gate41: Mapping[str, Any] | None = None
    if len(history) > 8:
        row = _mapping(history[8], "V41 history[8]")
        diagnostics = row.get("per_unit_nll_diagnostics")
        if not isinstance(diagnostics, list):
            raise ValueError("V41 update-8 per-unit diagnostics are absent")
        gate8 = v41_update8_gate(
            pair_metrics=_mapping(row.get("training_pair_metrics"), "V41 u8 pairs"),
            broad_nll=float(row.get("training_broad_nll")),
            source_broad_nll=source_broad,
            source_priority_deficit=source_deficit,
            query_state_sha256=str(row.get("query_bank_state_sha256")),
            frozen_state_sha256=str(row.get("frozen_excluding_query_state_sha256")),
            scene_state_exact=row.get("scene_prefix_and_residual_exact") is True,
            per_unit_nll_diagnostics=diagnostics,
            contract=contract,
        )
        if gate8 != row.get("update8_train_only_gate") or gate8 != stage.get(
            "update8_train_only_gate"
        ):
            raise ValueError("V41 independently replayed update-8 gate differs")
    if len(history) > 16:
        if gate8 is None:
            raise ValueError("V41 update-16 gate lacks update-8 evidence")
        row = _mapping(history[16], "V41 history[16]")
        diagnostics = row.get("per_unit_nll_diagnostics")
        if not isinstance(diagnostics, list):
            raise ValueError("V41 update-16 per-unit diagnostics are absent")
        gate16 = v41_update16_gate(
            update8_gate=gate8,
            pair_metrics=_mapping(row.get("training_pair_metrics"), "V41 u16 pairs"),
            broad_nll=float(row.get("training_broad_nll")),
            source_broad_nll=source_broad,
            source_priority_deficit=source_deficit,
            query_state_sha256=str(row.get("query_bank_state_sha256")),
            frozen_state_sha256=str(row.get("frozen_excluding_query_state_sha256")),
            scene_state_exact=row.get("scene_prefix_and_residual_exact") is True,
            per_unit_nll_diagnostics=diagnostics,
            contract=contract,
        )
        if gate16 != row.get("update16_train_only_gate") or gate16 != stage.get(
            "update16_train_only_gate"
        ):
            raise ValueError("V41 independently replayed update-16 gate differs")
    if len(history) > 41:
        if gate16 is None:
            raise ValueError("V41 update-41 gate lacks update-16 evidence")
        row = _mapping(history[41], "V41 history[41]")
        diagnostics = row.get("per_unit_nll_diagnostics")
        if not isinstance(diagnostics, list):
            raise ValueError("V41 update-41 per-unit diagnostics are absent")
        gate41 = v41_update41_gate(
            update16_gate=gate16,
            pair_metrics=_mapping(row.get("training_pair_metrics"), "V41 u41 pairs"),
            greedy_metrics=_mapping(row.get("training_greedy_metrics"), "V41 u41 greedy"),
            broad_nll=float(row.get("training_broad_nll")),
            source_broad_nll=source_broad,
            source_priority_deficit=source_deficit,
            query_state_sha256=str(row.get("query_bank_state_sha256")),
            frozen_state_sha256=str(row.get("frozen_excluding_query_state_sha256")),
            scene_state_exact=row.get("scene_prefix_and_residual_exact") is True,
            per_unit_nll_diagnostics=diagnostics,
            contract=contract,
        )
        if gate41 != row.get("update41_train_only_gate") or gate41 != stage.get(
            "update41_train_only_gate"
        ):
            raise ValueError("V41 independently replayed update-41 gate differs")
    return gate8, gate16, gate41


def latest_v41_resume_checkpoint(output: Path, contract: V41Contract) -> Path | None:
    if output.is_symlink() or (output.exists() and not output.is_dir()):
        raise ValueError("V41 output root must be a real directory")
    if not output.exists():
        return None
    guard_failures = sorted(output.glob("guard_failure_update_*.json"))
    if guard_failures:
        raise RuntimeError(
            "V41 fail-stop evidence exists; immutable authorization forbids resume: "
            f"{[path.name for path in guard_failures]}"
        )
    parsed: dict[int, Path] = {}
    for path in output.glob("update_*"):
        match = _UPDATE_DIRECTORY.fullmatch(path.name)
        if match is None or path.is_symlink() or not path.is_dir():
            raise ValueError(f"V41 output contains an unsafe or unexpected arm: {path.name}")
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
        raise ValueError("V41 complete arms are not a contiguous saved-step prefix")
    if sorted(set(parsed) - set(complete)):
        raise ValueError("V41 output contains incomplete or unauthorized arms")
    return None if not complete else parsed[complete[-1]]


def validate_v41_projection_history(
    history: Sequence[Mapping[str, Any]], contract: V41Contract
) -> dict[str, Any]:
    """Authenticate every projected microstep before save or resume."""

    if not history or history[0].get("optimizer_update") != 0:
        raise ValueError("V41 history must begin at update zero")
    previous_target = history[0].get("query_bank_state_sha256")
    if previous_target != contract.query_source_state_sha256:
        raise ValueError("V41 update-zero target hash changed")
    for step, raw_row in enumerate(history[1:], start=1):
        row = _mapping(raw_row, f"V41 history update {step}")
        raw = _mapping(row.get("raw_component_gradient_diagnostic"), "raw diagnostic")
        projection = _mapping(row.get("projected_gradient_attestation"), "projection")
        clip = _mapping(row.get("clip_direction_attestation"), "clip")
        replay = _mapping(projection.get("double_solve_replay"), "double solve")
        cpu = _mapping(projection.get("cpu_directional_attestation"), "CPU directions")
        cast = _mapping(
            projection.get("post_device_cast_directional_attestation"),
            "cast directions",
        )
        clipped = _mapping(clip.get("directional_checks"), "clip directions")
        candidates = projection.get("candidate_audits")
        norms = _mapping(projection.get("weighted_component_norms"), "component norms")
        component_finite = _mapping(
            projection.get("component_finite"), "component finite flags"
        )
        active_names = tuple(projection.get("active_constraint_direction_names", ()))
        inactive_names = tuple(
            projection.get("inactive_constraint_direction_names", ())
        )
        expected_active = tuple(
            name
            for name in _PROJECTION_DIRECTION_NAMES
            if isinstance(norms.get(name), (int, float)) and float(norms[name]) > 0.0
        )
        expected_inactive = tuple(
            name
            for name in _PROJECTION_DIRECTION_NAMES
            if norms.get(name) == 0.0
        )
        active_count = len(active_names)
        mask_count = 1 << active_count
        hashes = (
            raw.get("raw_total_state_sha256"),
            projection.get("raw_source_state_sha256"),
            projection.get("raw_total_state_sha256"),
            projection.get("cpu_projected_direction_sha256"),
            projection.get("post_device_cast_state_sha256"),
            clip.get("projected_input_state_sha256"),
            clip.get("clipped_output_state_sha256"),
        )
        delta = projection.get("active_solve_safety_delta")
        expected_delta = (
            0.0
            if projection.get("raw_solver_feasible") is True
            else _PROJECTION_KKT_ABSOLUTE_TOLERANCE
        )
        delta_numeric = float(delta) if isinstance(delta, (int, float)) else math.nan
        candidate_inventory_valid = (
            1 <= active_count <= 4
            and isinstance(candidates, list)
            and len(candidates) == mask_count
        )
        if candidate_inventory_valid:
            for mask, raw_candidate in enumerate(candidates):
                candidate = _mapping(raw_candidate, "V41 projection candidate")
                kkt = _mapping(candidate.get("kkt_checks"), "V41 candidate KKT")
                rejected = candidate.get("rejection_reason") is not None
                candidate_inventory_valid = candidate_inventory_valid and (
                    candidate.get("mask") == mask
                    and set(kkt)
                    == {
                        "independent_active_subset",
                        "finite",
                        "primal_feasible",
                        "dual_feasible",
                        "active_equality_feasible",
                        "stationarity",
                        "complementarity",
                    }
                    and isinstance(candidate.get("feasible"), bool)
                    and (
                        (candidate.get("feasible") is False)
                        if rejected
                        else (
                            isinstance(candidate.get("multipliers"), list)
                            and len(candidate["multipliers"]) == active_count
                            and isinstance(candidate.get("constraint_values"), list)
                            and len(candidate["constraint_values"]) == active_count
                            and isinstance(candidate.get("constraint_slacks"), list)
                            and len(candidate["constraint_slacks"]) == active_count
                            and candidate.get("feasible") is all(kkt.values())
                        )
                    )
                )
        selected_mask = projection.get("selected_mask")
        selected_candidate_valid = (
            isinstance(selected_mask, int)
            and 0 <= selected_mask < mask_count
            and candidate_inventory_valid
            and candidates[selected_mask].get("feasible") is True
            and projection.get("selected_active_constraints")
            == [
                name
                for index, name in enumerate(active_names)
                if selected_mask & (1 << index)
            ]
            and isinstance(projection.get("selected_lambdas"), list)
            and len(projection["selected_lambdas"]) == active_count
        )
        replay_hash_linked = (
            replay.get("first_projected_direction_sha256")
            == replay.get("second_projected_direction_sha256")
            == projection.get("cpu_projected_direction_sha256")
        )
        expected_beta = max(
            _PROJECTION_MINIMUM_BETA,
            _PROJECTION_MARGIN_SCALE * float(norms.get("raw_total", 0.0)),
        )
        raw_constraint_values = projection.get("raw_constraint_values")
        recomputed_raw_feasible = (
            isinstance(raw_constraint_values, list)
            and len(raw_constraint_values) == active_count
            and all(float(value) >= expected_beta for value in raw_constraint_values)
        )
        selected_candidate = (
            candidates[selected_mask] if selected_candidate_valid else {}
        )
        feasible_objectives = [
            (float(candidate["objective_half_squared_adjustment"]), candidate["mask"])
            for candidate in candidates
            if candidate.get("feasible") is True
        ] if candidate_inventory_valid else []
        selected_objective_minimal = False
        if feasible_objectives and selected_candidate_valid:
            minimum_objective = min(value[0] for value in feasible_objectives)
            tied_masks = [
                mask
                for objective, mask in feasible_objectives
                if math.isclose(
                    objective,
                    minimum_objective,
                    rel_tol=_PROJECTION_OBJECTIVE_TIE_RELATIVE_TOLERANCE,
                    abs_tol=0.0,
                )
            ]
            selected_objective_minimal = (
                selected_mask == min(tied_masks)
                and projection.get("projection_objective")
                == selected_candidate.get("objective_half_squared_adjustment")
                and projection.get("selected_lambdas")
                == selected_candidate.get("multipliers")
            )
        if (
            row.get("optimizer_update") != step
            or row.get("true_optimizer_step") is not True
            or raw.get("diagnostic_stage")
            != "raw_component_direction_before_projection"
            or raw.get("directional_conflicts_are_informational") is not True
            or projection.get("guard_stage")
            != "cpu_float64_active_set_projection"
            or projection.get("solver_device") != "cpu"
            or projection.get("solver_dtype") != "torch.float64"
            or projection.get("authorization_revision") != 3
            or projection.get("active_constraint_rule")
            != "finite_l2_norm_strictly_greater_than_zero"
            or projection.get("raw_total_norm_minimum_exclusive") != 1e-12
            or projection.get("standalone_side_is_not_a_constraint") is not True
            or projection.get("constraint_direction_order")
            != list(_PROJECTION_DIRECTION_NAMES)
            or active_names != expected_active
            or inactive_names != expected_inactive
            or projection.get("active_constraint_count") != active_count
            or projection.get("inactive_constraint_count") != len(inactive_names)
            or projection.get("rank_absolute_tolerance")
            != _PROJECTION_RANK_ABSOLUTE_TOLERANCE
            or projection.get("rank_relative_tolerance")
            != _PROJECTION_RANK_RELATIVE_TOLERANCE
            or projection.get("dual_lambda_lower_tolerance")
            != _PROJECTION_DUAL_LAMBDA_LOWER_TOLERANCE
            or projection.get("kkt_absolute_tolerance")
            != _PROJECTION_KKT_ABSOLUTE_TOLERANCE
            or projection.get("kkt_relative_tolerance")
            != _PROJECTION_KKT_RELATIVE_TOLERANCE
            or projection.get("objective_tie_relative_tolerance")
            != _PROJECTION_OBJECTIVE_TIE_RELATIVE_TOLERANCE
            or projection.get("enumerated_mask_order") != list(range(mask_count))
            or projection.get("enumerated_subset_count") != mask_count
            or projection.get("enumerated_mask_count_formula")
            != "2 ** active_constraint_count"
            or not isinstance(candidates, list)
            or len(candidates) != mask_count
            or [candidate.get("mask") for candidate in candidates]
            != list(range(mask_count))
            or not candidate_inventory_valid
            or not selected_candidate_valid
            or not replay_hash_linked
            or projection.get("beta") != expected_beta
            or projection.get("raw_solver_feasible") is not recomputed_raw_feasible
            or projection.get("active_solve_beta") != expected_beta + delta_numeric
            or not selected_objective_minimal
            or projection.get("candidate_kkt_residuals_computed_against_original_beta")
            is not True
            or projection.get(
                "active_solve_safety_delta_within_kkt_absolute_tolerance"
            )
            is not True
            or delta != expected_delta
            or any(
                replay.get(key) is not True
                for key in (
                    "solve_twice_from_independent_cpu_float64_clones",
                    "selected_mask_exact",
                    "selected_lambdas_bit_exact",
                    "projected_direction_bit_exact",
                    "projected_direction_sha256_exact",
                )
            )
            or projection.get("projection_feasible") is not True
            or projection.get("cpu_solution_safety_passed") is not True
            or projection.get("post_device_cast_safety_passed") is not True
            or projection.get("applied_vs_source_dtype_raw_safety_passed") is not True
            or float(projection.get("cpu_correction_ratio", math.inf)) > 0.25
            or float(projection.get("post_device_cast_correction_ratio", math.inf))
            > 0.25
            or float(projection.get("cpu_projected_raw_cosine", -math.inf)) < 0.95
            or float(
                projection.get("post_device_cast_projected_raw_cosine", -math.inf)
            )
            < 0.95
            or float(
                projection.get(
                    "applied_vs_source_dtype_raw_correction_ratio", math.inf
                )
            )
            > 0.25
            or float(
                projection.get("applied_vs_source_dtype_raw_cosine", -math.inf)
            )
            < 0.95
            or set(cpu) != set(_PROJECTION_DIRECTION_NAMES)
            or set(cast) != set(_PROJECTION_DIRECTION_NAMES)
            or any(cpu[name].get("passed") is not True for name in cpu)
            or any(cast[name].get("passed") is not True for name in cast)
            or any(
                cpu[name].get("active") is not False
                or cast[name].get("active") is not False
                or clipped[name].get("active") is not False
                or cpu[name].get("inactive_reason")
                != "exact_zero_norm_first_order_satisfied"
                or cast[name].get("inactive_reason")
                != "exact_zero_norm_first_order_satisfied"
                or clipped[name].get("inactive_reason")
                != "exact_zero_norm_first_order_satisfied"
                for name in inactive_names
            )
            or any(
                cpu[name].get("active") is not True
                or cast[name].get("active") is not True
                or clipped[name].get("active") is not True
                for name in active_names
            )
            or set(component_finite)
            != {"broad", "answer", "side", "cross", "scene", "raw_total"}
            or not all(value is True for value in component_finite.values())
            or any(
                not isinstance(norms.get(name), (int, float))
                or float(norms[name]) < 0.0
                for name in (
                    "broad",
                    "answer",
                    "side",
                    "cross",
                    "scene",
                    "raw_total",
                )
            )
            or float(norms.get("raw_total", 0.0)) <= 1e-12
            or any(not isinstance(value, str) or len(value) != 64 for value in hashes)
            or hashes[0] != hashes[1]
            or hashes[4] != hashes[5]
            or clip.get("guard_stage") != "scalar_global_clip"
            or clip.get("projection_input_linked") is not True
            or clip.get("constraint_activity_linked") is not True
            or tuple(clip.get("active_constraint_direction_names", ()))
            != active_names
            or tuple(clip.get("inactive_constraint_direction_names", ()))
            != inactive_names
            or float(clip.get("projected_to_clipped_cosine", -math.inf)) < 0.9999999
            or clip.get("scalar_clip_direction_preserved") is not True
            or set(clipped) != set(_PROJECTION_DIRECTION_NAMES)
            or any(
                clipped[name].get("strictly_positive_if_nonzero") is not True
                for name in clipped
            )
            or row.get("target_hash_before") != previous_target
            or row.get("target_hash_after") != row.get("query_bank_state_sha256")
            or row.get("frozen_excluding_b_hash_before") != contract.frozen_state_sha256
            or row.get("frozen_excluding_b_hash_after") != contract.frozen_state_sha256
            or row.get("frozen_excluding_query_state_sha256")
            != contract.frozen_state_sha256
            or row.get("validation_qa_loaded") is not False
            or row.get("oracle_environment_files_loaded") is not False
        ):
            raise ValueError(f"V41 projection history changed at update {step}")
        transient = row.get("transient_pre_update3_replay_attestation")
        if step == 3:
            evidence = _mapping(transient, "V41 update-three transient replay")
            if (
                evidence.get("observed_target_state_sha256")
                != _TRANSIENT_PRE_UPDATE3_TARGET_SHA256
                or evidence.get("observed_target_state_sha256")
                != row.get("target_hash_before")
                or evidence.get("expected_target_state_sha256")
                != _TRANSIENT_PRE_UPDATE3_TARGET_SHA256
                or evidence.get("exact_replay_of_v40_transient_steps_one_and_two")
                is not True
                or evidence.get(
                    "checked_before_component_gradients_clip_and_optimizer_step"
                )
                is not True
            ):
                raise ValueError("V41 update-three transient replay changed")
        elif transient is not None:
            raise ValueError("V41 transient replay appeared outside update three")
        previous_target = row.get("target_hash_after")
    return {
        "schema_version": 1,
        "validated_optimizer_steps": len(history) - 1,
        "all_projected_microsteps_authenticated": True,
    }


def validate_v41_resume_checkpoint(
    *,
    config: Mapping[str, Any],
    output: Path,
    resume: Path,
    contract: V41Contract,
    terminal: Mapping[str, Any],
    source_audit: Mapping[str, Any],
    schedule_audit: Mapping[str, Any],
    cache_audit: Mapping[str, Any],
    prefix_replay: Mapping[str, Any],
    update_zero_attestation: Mapping[str, Any],
    source_pair_metrics: Mapping[str, Any],
    source_per_unit_nll: Sequence[Mapping[str, Any]],
    source_broad_nll: float,
    source_greedy_metrics: Mapping[str, Any],
    source_residual: Mapping[str, Any],
) -> dict[str, Any]:
    if resume.parent != output or resume.is_symlink() or not resume.is_dir():
        raise ValueError("V41 resume must be a real numbered arm in its output root")
    if latest_v41_resume_checkpoint(output, contract) != resume:
        raise ValueError("V41 resume must be the latest contiguous complete arm")
    match = _UPDATE_DIRECTORY.fullmatch(resume.name)
    if match is None or (step := int(match.group(1))) not in contract.saved_optimizer_steps:
        raise ValueError("V41 resume arm is outside the bounded envelope")
    metadata = json.loads((resume / TRAINING_METADATA_FILENAME).read_text(encoding="utf-8"))
    stage = _mapping(metadata.get("v41_projected_gradient"), "V41 resume stage")
    expected_retry_terminal = (
        None
        if terminal["retry1"] is None
        else {
            "path": terminal["retry1"]["path"],
            "sha256": terminal["retry1"]["sha256"],
        }
    )
    expected_retry_authorization = (
        None
        if terminal["retry1"] is None
        else terminal["retry1"]["authorization"]
    )
    expected_predecessor = (
        None
        if terminal["retry1"] is None
        else terminal["retry1"]["predecessor_attestation"]
    )
    if (
        metadata.get("config_hash") != config_hash(dict(config))
        or metadata.get("optimizer_step") != step
        or stage.get("optimizer_step") != step
        or stage.get("conditional_v40_terminal_gate")
        != {"path": terminal["path"], "sha256": terminal["sha256"]}
        or stage.get("conditional_authorization") != terminal["authorization"]
        or stage.get("authorized_output_root") != terminal["authorized_output_root"]
        or stage.get("conditional_v41_retry1_terminal_gate")
        != expected_retry_terminal
        or stage.get("retry1_conditional_authorization")
        != expected_retry_authorization
        or stage.get("retry1_predecessor_attestation") != expected_predecessor
    ):
        raise ValueError("V41 resume config, step, or authorization changed")
    static = {
        "source_checkpoint": str(contract.source_checkpoint),
        "source_v40_u0_tensor_state_sha256": contract.source_tensor_state_sha256,
        "update_zero_tensor_state_sha256": contract.hybrid_tensor_state_sha256,
        "hybrid_v23_state_sha256": contract.hybrid_v23_state_sha256,
        "source_target_lora_b_state_sha256": contract.query_source_state_sha256,
        "source_block_core_state_sha256": contract.core_state_sha256,
        "source_optimizer_states_loaded": False,
        "source_optimizer_files_opened": False,
        "validation_qa_loaded": False,
        "oracle_environment_files_loaded": False,
        "deferred_final_scene_ids_loaded": [],
        "question_dependent_scene_processing": False,
        "question_dependent_retrieval": False,
    }
    if any(stage.get(key) != value for key, value in static.items()):
        raise ValueError("V41 resume source/data boundary changed")
    if (
        stage.get("source_audit") != dict(source_audit)
        or stage.get("schedule") != dict(schedule_audit)
        or stage.get("scene_cache") != _deterministic_cache_audit(cache_audit)
        or stage.get("prefix_replay_attestation") != dict(prefix_replay)
        or stage.get("update_zero_attestation") != dict(update_zero_attestation)
        or stage.get("source_pair_metrics") != dict(source_pair_metrics)
        or stage.get("source_per_unit_nll_diagnostics")
        != [dict(row) for row in source_per_unit_nll]
        or float(stage.get("source_broad_train_nll")) != source_broad_nll
        or stage.get("source_train_greedy_metrics") != dict(source_greedy_metrics)
        or stage.get("source_residual_diagnostics") != dict(source_residual)
    ):
        raise ValueError("V41 resume deterministic update-zero evidence changed")
    history = metadata.get("history")
    if not isinstance(history, list) or len(history) != step + 1:
        raise ValueError("V41 resume history is incomplete")
    if [row.get("optimizer_update") for row in history] != list(range(step + 1)):
        raise ValueError("V41 resume history is not one row per true optimizer step")
    projection_history_attestation = validate_v41_projection_history(
        history, contract
    )
    if stage.get("projection_history_attestation") != projection_history_attestation:
        raise ValueError("V41 resume projection-history attestation changed")
    if any(
        row.get("validation_qa_loaded") is not False
        or row.get("oracle_environment_files_loaded") is not False
        for row in history
    ):
        raise ValueError("V41 resume history crossed its train-only boundary")
    for gate_step in contract.diagnostic_steps:
        if gate_step <= step:
            rows = history[gate_step].get("per_unit_nll_diagnostics")
            if not isinstance(rows, list) or len(rows) != 25:
                raise ValueError("V41 resume lacks required per-unit gate diagnostics")
    runtime = json.loads((resume / RUNTIME_METADATA_FILENAME).read_text(encoding="utf-8"))
    validate_runtime_checkpoint_metadata(runtime)
    if runtime != runtime_checkpoint_metadata(metadata):
        raise ValueError("V41 resume runtime metadata is not freshly sanitized")
    tensors = load_file(resume / "adapter.safetensors", device="cpu")
    if tensor_state_sha256(_frozen_excluding_query(tensors)) != contract.frozen_state_sha256:
        raise ValueError("V41 resume changed a frozen tensor or buffer")
    query_hash = tensor_state_sha256(
        {
            "lora_b": tensors[_QUERY_PARAMETER_NAMES[0]],
        }
    )
    if query_hash != stage.get("target_lora_b_state_sha256"):
        raise ValueError("V41 resume target LoRA-B hash differs from metadata")
    if history[-1].get("target_hash_after", query_hash) != query_hash:
        raise ValueError("V41 resume final history target differs from adapter bytes")
    hybrid, _metadata, _audit = require_exact_v41_sources(config)
    if set(tensors) != set(hybrid):
        raise ValueError("V41 resume tensor inventory changed")
    changed = {name for name in tensors if not torch.equal(tensors[name], hybrid[name])}
    if step == 0:
        if changed or tensor_state_sha256(tensors) != contract.hybrid_tensor_state_sha256:
            raise ValueError("V41 update zero is not exact V40 update zero")
    elif not changed or not changed.issubset(_QUERY_PARAMETER_NAME_SET):
        raise ValueError("V41 resume changed an unauthorized tensor")
    if step:
        optimizer_step_audit(resume, expected_step=step, tensors=tensors)
    gate8, gate16, gate41 = replay_v41_gates(metadata, contract)
    if step >= 8 and (gate8 is None or gate8.get("passed") is not True):
        raise ValueError("V41 cannot resume at or past a failed update-8 gate")
    if step >= 16 and (gate16 is None or gate16.get("passed") is not True):
        raise ValueError("V41 cannot resume at or past a failed update-16 gate")
    if step >= 41 and (gate41 is None or gate41.get("passed") is not True):
        raise ValueError("V41 completed arm lacks a passed update-41 gate")
    return metadata


def run_v41(
    *, config: dict[str, Any], output: Path, resume: Path | None = None
) -> dict[str, Any]:
    terminal = require_v41_training_authorization(config)
    if _resolve(output) != Path(terminal["authorized_output_root"]):
        raise ValueError("V41 output root differs from the exact training authorization")
    if output.is_symlink() or (output.exists() and not output.is_dir()):
        raise ValueError("V41 output root must be a real directory")
    if resume is None and output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty V41 output: {output}")
    contract = v41_contract(config)
    settings = v41_settings(config)
    hybrid, pinned_source_metadata, source_audit = require_exact_v41_sources(config)
    loader = v41_loader_config(config)
    assert_deferred_final_scenes_absent(loader)
    records, qa_audit = load_v35_train_qa_records(loader)
    train_pairs = build_exact_question_pair_units(records)
    schedule, schedule_audit = build_v41_schedule(
        records, train_pairs, seed=int(config["seed"])
    )
    if (
        schedule_audit["pair_schedule_sha256"] != contract.pair_schedule_sha256
        or schedule_audit["schedule_sha256"] != contract.schedule_sha256
    ):
        raise RuntimeError("V41 generated schedule differs from its exact lock")
    inherited_schedule, _inherited_audit = build_v35_schedule(
        records,
        train_pairs,
        settings=v35_settings(loader),
        seed=int(config["seed"]),
    )
    broad_calibration = v36_broad_calibration_records(inherited_schedule)

    approved = require_approved_v29_source(loader)
    bundle, block_core, source_metadata, loader_transition = load_v41_bundle(
        config, approved, contract.source_checkpoint, hybrid
    )
    if source_metadata != pinned_source_metadata:
        raise RuntimeError("V41 source metadata changed during exact adapter load")
    source_audit = {**source_audit, "loader_transition": loader_transition}
    validate_block_cross_residual_state(
        block_core,
        expected_parameter_count=983_040,
        expected_state_sha256=contract.core_state_sha256,
        context="V41 frozen learned block core",
    )
    if (
        module_collection_state_sha256(bundle.checkpoint_modules)
        != contract.hybrid_tensor_state_sha256
        or _query_bank(bundle).state_sha256() != _V28_BANK_STATE_SHA256
        or target_v41_state_sha256(bundle) != contract.query_source_state_sha256
        or _v23_bank(bundle).state_sha256() != contract.hybrid_v23_state_sha256
        or frozen_v41_state_sha256(bundle) != contract.frozen_state_sha256
    ):
        raise RuntimeError("V41 exact hybrid changed before map caching")

    split = v31_contract(loader)
    all_development_scene_ids = (*split.train_scene_ids, *split.validation_scene_ids)
    caches, cache_audit = cache_v41_train_scenes(
        config=loader,
        bundle=bundle,
        source_metadata=pinned_source_metadata,
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
        config=loader,
        train_scene_ids=split.train_scene_ids,
        validation_scene_ids=split.validation_scene_ids,
    )
    prefix_replay = _prefix_replay_attestation(
        caches=train_caches,
        block_cross_residual=block_core,
        bundle=bundle,
        expected_scene_ids=split.train_scene_ids,
    )
    source_pair_metrics, source_per_unit_nll = training_pair_gate_diagnostics(
        units=train_pairs,
        caches=train_caches,
        block_cross_residual=block_core,
        bundle=bundle,
        settings=settings,
    )
    validate_per_unit_nll_diagnostics(source_per_unit_nll, source_pair_metrics)
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
        config=loader,
    )
    update_zero_baseline = validate_update_zero_baseline(
        pair_metrics=source_pair_metrics,
        broad_nll=source_broad_nll,
        greedy_metrics=source_greedy_metrics,
        contract=contract,
    )
    update_zero_attestation = {
        "exact_v40_update_zero_source_loaded": True,
        "hybrid_tensor_state_sha256": module_collection_state_sha256(
            bundle.checkpoint_modules
        ),
        "hybrid_v23_state_sha256": _v23_bank(bundle).state_sha256(),
        "existing_learned_query_loaded_without_reinitialization": True,
        "complete_v28_bank_source_state_sha256": _query_bank(bundle).state_sha256(),
        "target_lora_b_source_state_sha256": target_v41_state_sha256(bundle),
        "target_lora_b_nonzero": bool(
            torch.count_nonzero(_target_parameters(bundle)[0]).item() > 0
        ),
        "learned_block_core_state_sha256": block_core.state_sha256(),
        "frozen_excluding_query_state_sha256": frozen_v41_state_sha256(bundle),
        "source_optimizer_files_opened": False,
        "source_optimizer_states_loaded": False,
        "fresh_momentum_free_sgd_state": True,
        "hybrid_behavior_recomputed_before_optimizer": True,
        "behavioral_baseline": update_zero_baseline,
        "training_cache_boundary": cache_boundary,
        "validation_qa_loaded": False,
        "oracle_environment_files_loaded": False,
    }
    if update_zero_attestation["target_lora_b_nonzero"] is not True:
        raise RuntimeError("V41 layer-14 LoRA-B is not the learned V40-u0 source")
    source_priority_deficit = float(
        update_zero_baseline["observed"]["priority_combined_side_deficit"]
    )
    surface = assert_v41_trainable_surface(bundle)
    optimizer = v41_optimizer(bundle, settings)
    source_audit["source_optimizer_files_opened"] = False
    source_audit["source_optimizer_states_loaded"] = False
    history: list[dict[str, Any]] = [
        {
            "optimizer_update": 0,
            "source_pair_metrics": source_pair_metrics,
            "per_unit_nll_diagnostics": source_per_unit_nll,
            "source_broad_train_nll": source_broad_nll,
            "source_train_greedy_metrics": source_greedy_metrics,
            "training_residual_diagnostics": source_residual,
            "scene_prefix_and_residual_exact": True,
            "query_bank_state_sha256": target_v41_state_sha256(bundle),
            "frozen_excluding_query_state_sha256": frozen_v41_state_sha256(bundle),
            "update_zero_attestation": update_zero_attestation,
            "update8_train_only_gate": None,
            "update16_train_only_gate": None,
            "update41_train_only_gate": None,
            "validation_qa_loaded": False,
            "oracle_environment_files_loaded": False,
            "saved_checkpoint": True,
        }
    ]
    start_step = 0
    accepted8: Mapping[str, Any] | None = None
    accepted16: Mapping[str, Any] | None = None
    accepted41: Mapping[str, Any] | None = None
    output.mkdir(parents=True, exist_ok=True)
    if resume is not None:
        resume_metadata = validate_v41_resume_checkpoint(
            config=config,
            output=output,
            resume=resume,
            contract=contract,
            terminal=terminal,
            source_audit=source_audit,
            schedule_audit=schedule_audit,
            cache_audit=cache_audit,
            prefix_replay=prefix_replay,
            update_zero_attestation=update_zero_attestation,
            source_pair_metrics=source_pair_metrics,
            source_per_unit_nll=source_per_unit_nll,
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
            raise RuntimeError("V41 resume metadata changed during adapter load")
        freeze_for_v41(bundle)
        assert_v41_trainable_surface(bundle, optimizer=optimizer)
        resume_stage = _mapping(
            resume_metadata["v41_projected_gradient"], "V41 loaded resume stage"
        )
        live_target_hash = target_v41_state_sha256(bundle)
        file_tensors = load_file(resume / "adapter.safetensors", device="cpu")
        file_target_hash = tensor_state_sha256(
            {"lora_b": file_tensors[_QUERY_PARAMETER_NAMES[0]]}
        )
        if (
            live_target_hash != resume_stage.get("target_lora_b_state_sha256")
            or file_target_hash != live_target_hash
        ):
            raise RuntimeError("V41 resumed live LoRA-B differs from authenticated file/metadata")
        if frozen_v41_state_sha256(bundle) != contract.frozen_state_sha256:
            raise RuntimeError("V41 resumed a changed frozen surface")
        start_step = int(resume_metadata["optimizer_step"])
        if start_step:
            load_optimizer_checkpoint(resume, optimizer, bundle.language.device)
            _optimizer_payload_audit(
                optimizer.state_dict(),
                expected_step=start_step,
                tensors=load_file(resume / "adapter.safetensors", device="cpu"),
            )
        history = list(resume_metadata["history"])
        accepted8 = resume_stage.get("update8_train_only_gate")
        accepted16 = resume_stage.get("update16_train_only_gate")
        accepted41 = resume_stage.get("update41_train_only_gate")
    else:
        metadata0 = _metadata(
            source_metadata=pinned_source_metadata,
            config=config,
            terminal=terminal,
            source_audit=source_audit,
            schedule_audit=schedule_audit,
            cache_audit=cache_audit,
            qa_audit=qa_audit,
            prefix_replay=prefix_replay,
            update_zero_attestation=update_zero_attestation,
            source_pair_metrics=source_pair_metrics,
            source_per_unit_nll=source_per_unit_nll,
            source_broad_nll=source_broad_nll,
            source_greedy_metrics=source_greedy_metrics,
            source_residual=source_residual,
            history=history,
            optimizer_step=0,
            bundle=bundle,
            surface=surface,
            gate8=None,
            gate16=None,
            gate41=None,
        )
        _save(output / "update_000", bundle=bundle, metadata=metadata0, optimizer=None)
        saved0 = load_file(output / "update_000" / "adapter.safetensors", device="cpu")
        if tensor_state_sha256(saved0) != contract.hybrid_tensor_state_sha256:
            raise RuntimeError("V41 saved update zero differs from the exact hybrid")

    target_parameters = tuple(_target_parameters(bundle))
    frozen_residual_rms = float(source_residual["aggregate_rms"])
    normalized_residual = (frozen_residual_rms / settings.residual_penalty_scale) ** 2
    for item in schedule[start_step:]:
        step = item.optimizer_step
        freeze_for_v41(bundle)
        assert_v41_trainable_surface(bundle, optimizer=optimizer)
        optimizer.zero_grad(set_to_none=True)
        target_hash_before = target_v41_state_sha256(bundle)
        frozen_hash_before = frozen_v41_state_sha256(bundle)
        if frozen_hash_before != contract.frozen_state_sha256:
            raise RuntimeError("V41 frozen surface changed before optimizer step")
        transient_replay_attestation: dict[str, Any] | None = None
        if step == 3:
            transient_replay_attestation = {
                "required_before_optimizer_step": 3,
                "observed_target_state_sha256": target_hash_before,
                "expected_target_state_sha256": (
                    _TRANSIENT_PRE_UPDATE3_TARGET_SHA256
                ),
                "exact_replay_of_v40_transient_steps_one_and_two": (
                    target_hash_before == _TRANSIENT_PRE_UPDATE3_TARGET_SHA256
                ),
                "checked_before_component_gradients_clip_and_optimizer_step": True,
            }
            if not transient_replay_attestation[
                "exact_replay_of_v40_transient_steps_one_and_two"
            ]:
                failure = V41GradientGuardFailure(
                    {
                        "schema_version": 1,
                        "guard_stage": "transient_pre_update3_replay",
                        "failure_reason": "v40_transient_target_hash_mismatch",
                        "transient_replay_attestation": transient_replay_attestation,
                    }
                )
                persist_gradient_guard_failure(
                    output,
                    optimizer_step=step,
                    audit={
                        **failure.audit,
                        "target_hash_before": target_hash_before,
                        "target_hash_after": target_v41_state_sha256(bundle),
                        "frozen_excluding_b_hash_before": frozen_hash_before,
                        "frozen_excluding_b_hash_after": frozen_v41_state_sha256(bundle),
                    },
                )
                raise failure
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
        try:
            broad_gradients = _component_gradients(
                settings.broad_nll_weight * broad,
                target_parameters,
                retain_graph=False,
            )
        except V41GradientGuardFailure as exc:
            persist_gradient_guard_failure(
                output,
                optimizer_step=step,
                audit={
                    "failed_guard_stage": exc.audit.get("guard_stage"),
                    "failed_component": "broad",
                    "component_autograd_failure": exc.audit,
                    "clip_direction_attestation": None,
                    "target_hash_before": target_hash_before,
                    "target_hash_after": target_v41_state_sha256(bundle),
                    "frozen_excluding_b_hash_before": frozen_hash_before,
                    "frozen_excluding_b_hash_after": frozen_v41_state_sha256(bundle),
                },
            )
            raise
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
        try:
            answer_gradients = _component_gradients(
                settings.pair_correct_nll_weight * pair_nll,
                target_parameters,
                retain_graph=True,
            )
            side_gradients = _component_gradients(
                settings.side_hinge_weight * side_hinge,
                target_parameters,
                retain_graph=True,
            )
            cross_gradients = _component_gradients(
                settings.cross_prefix_flip_weight * cross_hinge,
                target_parameters,
                retain_graph=False,
            )
        except V41GradientGuardFailure as exc:
            persist_gradient_guard_failure(
                output,
                optimizer_step=step,
                audit={
                    "failed_guard_stage": exc.audit.get("guard_stage"),
                    "failed_component": "answer_side_or_cross",
                    "component_autograd_failure": exc.audit,
                    "clip_direction_attestation": None,
                    "target_hash_before": target_hash_before,
                    "target_hash_after": target_v41_state_sha256(bundle),
                    "frozen_excluding_b_hash_before": frozen_hash_before,
                    "frozen_excluding_b_hash_after": frozen_v41_state_sha256(bundle),
                },
            )
            raise
        pair_nll_value = float(pair_nll.detach().cpu())
        side_hinge_value = float(side_hinge.detach().cpu())
        cross_hinge_value = float(cross_hinge.detach().cpu())
        side_margin_mean = float(
            diagnostics["side_margins"].detach().float().mean().cpu()
        )
        cross_margin_mean = float(
            diagnostics["cross_prefix_margins"].detach().float().mean().cpu()
        )
        del pair_nll, side_hinge, cross_hinge, diagnostics, pair_tokens

        component_gradients = {
            "broad": broad_gradients,
            "answer": answer_gradients,
            "side": side_gradients,
            "cross": cross_gradients,
        }
        raw_diagnostic: dict[str, Any] | None = None
        projection_attestation: dict[str, Any] | None = None
        try:
            _raw_total, raw_diagnostic = raw_component_gradient_diagnostic(
                component_gradients
            )
            projected_total, projection_attestation = (
                project_gradient_to_feasible_descent(component_gradients)
            )
            if (
                raw_diagnostic["raw_total_state_sha256"]
                != projection_attestation["raw_source_state_sha256"]
            ):
                raise V41GradientGuardFailure(
                    {
                        "schema_version": 1,
                        "guard_stage": "raw_to_projection_hash_link",
                        "failure_reason": "raw_direction_hash_mismatch",
                        "raw_diagnostic_state_sha256": raw_diagnostic[
                            "raw_total_state_sha256"
                        ],
                        "projection_raw_state_sha256": projection_attestation[
                            "raw_source_state_sha256"
                        ],
                    }
                )
            clip_attestation = clip_direction_attestation(
                parameters=target_parameters,
                projected_total=projected_total,
                components=component_gradients,
                projection_attestation=projection_attestation,
                clip_norm=settings.gradient_clip_norm,
            )
        except V41GradientGuardFailure as exc:
            persist_gradient_guard_failure(
                output,
                optimizer_step=step,
                audit={
                    "failed_guard_stage": exc.audit.get("guard_stage"),
                    "raw_component_gradient_diagnostic": raw_diagnostic,
                    "projected_gradient_attestation": (
                        exc.audit
                        if exc.audit.get("guard_stage")
                        in {
                            "projection_input",
                            "cpu_float64_active_set_projection",
                            "raw_to_projection_hash_link",
                        }
                        else projection_attestation
                    ),
                    "clip_direction_attestation": (
                        exc.audit
                        if exc.audit.get("guard_stage") == "scalar_global_clip"
                        else None
                    ),
                    "target_hash_before": target_hash_before,
                    "target_hash_after": target_v41_state_sha256(bundle),
                    "frozen_excluding_b_hash_before": frozen_hash_before,
                    "frozen_excluding_b_hash_after": frozen_v41_state_sha256(bundle),
                },
            )
            raise
        preclip_norm = float(clip_attestation["projected_total_norm"])
        optimizer.step()
        _query_bank(bundle).validate_state()
        query_hash = target_v41_state_sha256(bundle)
        frozen_hash = frozen_v41_state_sha256(bundle)
        if frozen_hash != contract.frozen_state_sha256:
            raise RuntimeError("V41 changed a frozen tensor or buffer")
        if (
            _v23_bank(bundle).state_sha256() != contract.hybrid_v23_state_sha256
            or block_core.state_sha256() != contract.core_state_sha256
        ):
            raise RuntimeError("V41 changed the hybrid V23 bank or learned block core")

        should_save = step in contract.saved_optimizer_steps
        pair_metrics: Mapping[str, Any] | None = None
        per_unit_nll: list[dict[str, Any]] | None = None
        broad_diagnostic: float | None = None
        greedy_metrics: Mapping[str, Any] | None = None
        residual_diagnostics: Mapping[str, Any] | None = None
        scene_exact: bool | None = None
        if should_save:
            residual_diagnostics = residual_rms_diagnostics(
                caches=train_caches,
                block_cross_residual=block_core,
                device=bundle.language.device,
            )
        if step in contract.diagnostic_steps:
            pair_metrics, per_unit_nll = training_pair_gate_diagnostics(
                units=train_pairs,
                caches=train_caches,
                block_cross_residual=block_core,
                bundle=bundle,
                settings=settings,
            )
            validate_per_unit_nll_diagnostics(per_unit_nll, pair_metrics)
            broad_diagnostic = training_broad_nll(
                records=broad_calibration,
                caches=train_caches,
                block_cross_residual=block_core,
                bundle=bundle,
            )
            current_prefix = _prefix_replay_attestation(
                caches=train_caches,
                block_cross_residual=block_core,
                bundle=bundle,
                expected_scene_ids=split.train_scene_ids,
            )
            scene_exact = current_prefix == prefix_replay and residual_diagnostics == source_residual
        if step == 41:
            greedy_metrics = training_greedy_metrics(
                units=train_pairs,
                broad_records=broad_calibration,
                caches=train_caches,
                block_cross_residual=block_core,
                bundle=bundle,
                config=loader,
            )
        if step == 8:
            assert pair_metrics is not None and per_unit_nll is not None
            assert broad_diagnostic is not None
            accepted8 = v41_update8_gate(
                pair_metrics=pair_metrics,
                broad_nll=broad_diagnostic,
                source_broad_nll=source_broad_nll,
                source_priority_deficit=source_priority_deficit,
                query_state_sha256=query_hash,
                frozen_state_sha256=frozen_hash,
                scene_state_exact=scene_exact is True,
                per_unit_nll_diagnostics=per_unit_nll,
                contract=contract,
            )
        if step == 16:
            if not isinstance(accepted8, Mapping):
                raise RuntimeError("V41 update-16 gate lacks update-8 evidence")
            assert pair_metrics is not None and per_unit_nll is not None
            assert broad_diagnostic is not None
            accepted16 = v41_update16_gate(
                update8_gate=accepted8,
                pair_metrics=pair_metrics,
                broad_nll=broad_diagnostic,
                source_broad_nll=source_broad_nll,
                source_priority_deficit=source_priority_deficit,
                query_state_sha256=query_hash,
                frozen_state_sha256=frozen_hash,
                scene_state_exact=scene_exact is True,
                per_unit_nll_diagnostics=per_unit_nll,
                contract=contract,
            )
        if step == 41:
            if not isinstance(accepted16, Mapping):
                raise RuntimeError("V41 update-41 gate lacks update-16 evidence")
            assert pair_metrics is not None and per_unit_nll is not None
            assert broad_diagnostic is not None and greedy_metrics is not None
            accepted41 = v41_update41_gate(
                update16_gate=accepted16,
                pair_metrics=pair_metrics,
                greedy_metrics=greedy_metrics,
                broad_nll=broad_diagnostic,
                source_broad_nll=source_broad_nll,
                source_priority_deficit=source_priority_deficit,
                query_state_sha256=query_hash,
                frozen_state_sha256=frozen_hash,
                scene_state_exact=scene_exact is True,
                per_unit_nll_diagnostics=per_unit_nll,
                contract=contract,
            )
        optimized_loss, reported_loss = v41_loss_values(
            settings=settings,
            broad_nll=broad_value,
            pair_correct_nll=pair_nll_value,
            side_hinge=side_hinge_value,
            cross_prefix_hinge=cross_hinge_value,
            frozen_normalized_residual=normalized_residual,
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
                "train_normalized_residual_penalty": normalized_residual,
                "train_residual_rms": frozen_residual_rms,
                "train_optimized_loss": optimized_loss,
                "train_reported_composite_including_frozen_residual": reported_loss,
                "train_objective": optimized_loss,
                "frozen_residual_descriptive_only": True,
                "residual_penalty_contributes_gradient": False,
                "optimizer_stage": "existing_v28_layer14_lora_b_only_projected_gradient_sgd",
                "preclip_gradient_norm": preclip_norm,
                "gradient_clip_norm": settings.gradient_clip_norm,
                "raw_component_gradient_diagnostic": raw_diagnostic,
                "projected_gradient_attestation": projection_attestation,
                "clip_direction_attestation": clip_attestation,
                "transient_pre_update3_replay_attestation": (
                    transient_replay_attestation
                ),
                "target_hash_before": target_hash_before,
                "target_hash_after": query_hash,
                "frozen_excluding_b_hash_before": frozen_hash_before,
                "frozen_excluding_b_hash_after": frozen_hash,
                "training_pair_metrics": pair_metrics,
                "per_unit_nll_diagnostics": per_unit_nll,
                "training_broad_nll": broad_diagnostic,
                "training_greedy_metrics": greedy_metrics,
                "training_residual_diagnostics": residual_diagnostics,
                "scene_prefix_and_residual_exact": scene_exact,
                "query_bank_state_sha256": query_hash,
                "frozen_excluding_query_state_sha256": frozen_hash,
                "update8_train_only_gate": accepted8,
                "update16_train_only_gate": accepted16,
                "update41_train_only_gate": accepted41,
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
            source_audit=source_audit,
            schedule_audit=schedule_audit,
            cache_audit=cache_audit,
            qa_audit=qa_audit,
            prefix_replay=prefix_replay,
            update_zero_attestation=update_zero_attestation,
            source_pair_metrics=source_pair_metrics,
            source_per_unit_nll=source_per_unit_nll,
            source_broad_nll=source_broad_nll,
            source_greedy_metrics=source_greedy_metrics,
            source_residual=source_residual,
            history=history,
            optimizer_step=step,
            bundle=bundle,
            surface=assert_v41_trainable_surface(bundle, optimizer=optimizer),
            gate8=accepted8,
            gate16=accepted16,
            gate41=accepted41,
        )
        checkpoint = output / f"update_{step:03d}"
        _save(checkpoint, bundle=bundle, metadata=metadata, optimizer=optimizer)
        saved_tensors = load_file(checkpoint / "adapter.safetensors", device="cpu")
        saved_target_hash = tensor_state_sha256(
            {"lora_b": saved_tensors[_QUERY_PARAMETER_NAMES[0]]}
        )
        if (
            len(saved_tensors) != 179
            or saved_target_hash != query_hash
            or saved_target_hash != target_v41_state_sha256(bundle)
            or tensor_state_sha256(_frozen_excluding_query(saved_tensors))
            != contract.frozen_state_sha256
            or tensor_state_sha256(saved_tensors)
            != module_collection_state_sha256(bundle.checkpoint_modules)
        ):
            raise RuntimeError("V41 saved checkpoint differs from live/frozen exact state")
        optimizer_step_audit(
            checkpoint,
            expected_step=step,
            tensors=saved_tensors,
        )
        print(
            json.dumps(
                {
                    "phase": "v41_projected_gradient_checkpoint",
                    "optimizer_step": step,
                    "training_complete_units": None
                    if pair_metrics is None
                    else pair_metrics["complete_units"],
                    "training_cross_complete_units": None
                    if pair_metrics is None
                    else pair_metrics["cross_prefix_complete_units"],
                    "update8_gate_passed": None
                    if accepted8 is None
                    else accepted8.get("passed"),
                    "update16_gate_passed": None
                    if accepted16 is None
                    else accepted16.get("passed"),
                    "update41_gate_passed": None
                    if accepted41 is None
                    else accepted41.get("passed"),
                    "validation_qa_loaded": False,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if step == 8 and accepted8.get("passed") is not True:
            raise RuntimeError("V41 update-8 train-only gate failed")
        if step == 16 and accepted16.get("passed") is not True:
            raise RuntimeError("V41 update-16 train-only gate failed")
        if step == 41 and accepted41.get("passed") is not True:
            raise RuntimeError("V41 update-41 train-only gate failed")

    return {
        "schema_version": 1,
        "artifact": "v41_diverse28_projected_gradient_training",
        "output": str(output),
        "optimizer_updates": 41,
        "resumed_from_optimizer_step": start_step,
        "saved_optimizer_steps": list(contract.saved_optimizer_steps),
        "exact_trainable_parameter_count": 16_384,
        "hybrid_tensor_state_sha256": contract.hybrid_tensor_state_sha256,
        "hybrid_v23_state_sha256": contract.hybrid_v23_state_sha256,
        "source_query_state_sha256": contract.query_source_state_sha256,
        "source_block_core_state_sha256": contract.core_state_sha256,
        "v40_terminal_report_sha256": terminal["sha256"],
        "v41_retry1_terminal_report_sha256": None
        if terminal["retry1"] is None
        else terminal["retry1"]["sha256"],
        "retry1_predecessor_attestation": None
        if terminal["retry1"] is None
        else terminal["retry1"]["predecessor_attestation"],
        "update8_train_only_gate": accepted8,
        "update16_train_only_gate": accepted16,
        "update41_train_only_gate": accepted41,
        "source_optimizer_files_opened": False,
        "source_optimizer_states_loaded": False,
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
    parser.add_argument("--output", type=Path)
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
        print(json.dumps(preflight_v41(config), indent=2, sort_keys=True))
        return 0
    if args.construction_preflight_only:
        print(json.dumps(construction_preflight_v41(config), indent=2, sort_keys=True))
        return 0
    training_authorization = require_v41_training_authorization(config)
    selected_output = (
        args.output
        if args.output is not None
        else Path(training_authorization["authorized_output_root"])
    )
    output = _unresolved_project_path(selected_output)
    if output.is_symlink() or (output.exists() and not output.is_dir()):
        raise ValueError("V41 output path is a symlink or non-directory")
    contract = v41_contract(config)
    resume: Path | None = None
    if args.resume is not None:
        unresolved = _unresolved_project_path(args.resume)
        if unresolved.is_symlink():
            raise ValueError("V41 resume path may not be a symlink")
        resume = unresolved
    elif args.resume_latest:
        resume = latest_v41_resume_checkpoint(output, contract)
        if resume is None:
            raise FileNotFoundError("V41 output contains no complete resume arm")
    result = run_v41(config=config, output=output, resume=resume)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
