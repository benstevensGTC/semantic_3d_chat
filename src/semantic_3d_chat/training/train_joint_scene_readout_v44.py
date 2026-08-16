"""Bounded train-only joint scene-readout/layer-14 pilot for V44.

The only trainable tensors are ``block_cross_residual.w_o`` and LoRA A/B for
the existing layer-14 query adapter.  The run starts from exact V41 retry1
update zero, uses a fresh two-group AdamW, stops at the first failed train-only
gate, and never makes validation, oracle, final-scene, selector, or runtime
promotion access legal.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from semantic_3d_chat.chat.file_audit import FileAccessAudit
from semantic_3d_chat.config import PROJECT_ROOT, artifact_root, config_hash, load_config
from semantic_3d_chat.language.lora import tensor_state_sha256
from semantic_3d_chat.scene_encoder.block_cross_residual import BlockCrossResidual
from semantic_3d_chat.training.checkpointing import (
    RUNTIME_METADATA_FILENAME,
    TRAINING_METADATA_FILENAME,
    module_collection_state_sha256,
    runtime_checkpoint_metadata,
    save_adapter_checkpoint,
    save_optimizer_checkpoint,
    validate_runtime_checkpoint_metadata,
)
from semantic_3d_chat.training.pair_curriculum import build_exact_question_pair_units
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
from semantic_3d_chat.training.train_projected_gradient_v41 import (
    _prefix_replay_attestation,
    _query_bank,
    build_v41_schedule,
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
    "configs/experiments/gemma4_diverse28_joint_scene_readout_v44.yaml"
)
DEFAULT_OUTPUT = Path(
    "data_gemma4/checkpoints/gemma4_v44_joint_scene_readout_l14_query"
)
_CONFIG_FILE_SHA256 = "a3f3b65dc3a32612060a679cbcc40c115e5b0c4d014670c9a8f7f752e4a7abb7"
_V43_TERMINAL_PATH = Path(
    "reports/gemma4/metrics/v43_aggregate_projected_screen_terminal_gate.json"
)
_SOURCE_CHECKPOINT = Path(
    "data_gemma4/checkpoints/"
    "gemma4_v41_retry1_diverse28_projected_gradient_l14_query/update_000"
)
_V43_TERMINAL_SHA256 = "013fbe79ac42e842e83989e33f132b9ff3529746a8045feb212ded32e50a2cc2"
_SOURCE_FULL_SHA256 = "7b951c6d7ae4f7b50603159f0bc4dfb4d50b5b40f9325134d78d1de1dae87fc0"
_SOURCE_AUTHORIZED_SHA256 = "b935c7e6ccceb1068f80e679b4159c6ca756f9f81868b954b93ac683e014f5a0"
_FROZEN_SHA256 = "31cb215cf0d1623886d4a79203c501912e66537021da53dd711124abdc2e36fa"
_PROTECTED_REPORT = Path(
    "reports/gemma4/metrics/training_selection_gemma4_color_mirror_full_vocab_v11_resume36.json"
)
_PROTECTED_REPORT_SHA256 = (
    "c0086f66edbb8854a7938e09c57535bfd47100adbaf3b3c95eeb4b08014ce2f8"
)
_SOURCE_FILES = {
    "adapter.safetensors": "b0bdceb7699e9d97467915c69186f433d3b0fac2b09144d38c3078afe1f70cb0",
    TRAINING_METADATA_FILENAME: "331cda3f2ebc1539e8ee27ebbae398be5e19f3fd77d0aa20dde635d569e29d6d",
    RUNTIME_METADATA_FILENAME: "690e790b612e0b75323c1f27f7e9afe87243ccc1564c8cc690e86a442cffbfcd",
}
_PARAMETER_NAMES = (
    "block_cross_residual.w_o",
    "lora_banks.extension_v28_stage_b_query.adapters.1.lora_a",
    "lora_banks.extension_v28_stage_b_query.adapters.1.lora_b",
)
_PARAMETER_SHAPES = ((256, 1536), (4, 1536), (4096, 4))
_PARAMETER_COUNTS = (393_216, 6_144, 16_384)
_SAVED_STEPS = (0, 4, 8, 16)
_HEX = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class V44Settings:
    optimizer_steps: int
    checkpoint_steps: tuple[int, ...]
    broad_nll_weight: float
    pair_correct_nll_weight: float
    side_hinge_weight: float
    cross_prefix_flip_weight: float
    side_hinge_margin: float
    cross_prefix_flip_margin: float
    source_prefix_trust_weight: float
    source_prefix_trust_scale: float
    scene_readout_learning_rate: float
    query_learning_rate: float
    weight_decay: float
    gradient_clip_norm: float


@dataclass(frozen=True)
class V44Contract:
    terminal_report: Path
    configured_terminal_sha256: str
    source_checkpoint: Path
    source_file_sha256: Mapping[str, str]
    source_full_state_sha256: str
    frozen_state_sha256: str
    authorized_parameter_names: tuple[str, ...]
    authorized_parameter_shapes: tuple[tuple[int, ...], ...]
    total_parameter_count: int


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


def v44_settings(config: Mapping[str, Any]) -> V44Settings:
    training = _mapping(config.get("training"), "training")
    raw = _mapping(training.get("v44_joint_scene_readout"), "V44 training")
    expected = {
        "enabled": True,
        "optimizer_steps": 16,
        "checkpoint_steps": list(_SAVED_STEPS),
        "broad_nll_weight": 0.25,
        "pair_correct_nll_weight": 0.5,
        "side_hinge_weight": 8.0,
        "cross_prefix_flip_weight": 8.0,
        "side_hinge_margin": 0.5,
        "cross_prefix_flip_margin": 0.1,
        "source_prefix_trust_weight": 0.001,
        "source_prefix_trust_scale": 0.05,
        "scene_readout_learning_rate": 2.5e-5,
        "query_learning_rate": 2.0e-5,
        "weight_decay": 0.0,
        "gradient_clip_norm": 1.0,
    }
    if set(raw) != set(expected) or any(raw.get(key) != value for key, value in expected.items()):
        raise ValueError("V44 exact optimizer/objective settings changed")
    return V44Settings(
        optimizer_steps=16,
        checkpoint_steps=_SAVED_STEPS,
        broad_nll_weight=0.25,
        pair_correct_nll_weight=0.5,
        side_hinge_weight=8.0,
        cross_prefix_flip_weight=8.0,
        side_hinge_margin=0.5,
        cross_prefix_flip_margin=0.1,
        source_prefix_trust_weight=0.001,
        source_prefix_trust_scale=0.05,
        scene_readout_learning_rate=2.5e-5,
        query_learning_rate=2.0e-5,
        weight_decay=0.0,
        gradient_clip_norm=1.0,
    )


def v44_contract(config: Mapping[str, Any]) -> V44Contract:
    raw = _mapping(config.get("v44_joint_scene_readout"), "v44_joint_scene_readout")
    if raw.get("schema_version") != 1 or raw.get("role") != (
        "exact_v41_u0_joint_scene_readout_layer14_query_train_only_pilot"
    ):
        raise ValueError("V44 contract identity changed")
    names = tuple(str(value) for value in raw.get("authorized_parameter_names", ()))
    shapes = tuple(tuple(int(item) for item in value) for value in raw.get("authorized_parameter_shapes", ()))
    required = {
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
        "source_full": raw.get("source_full_tensor_state_sha256") == _SOURCE_FULL_SHA256,
        "source_authorized": raw.get("source_authorized_surface_state_sha256")
        == _SOURCE_AUTHORIZED_SHA256,
        "frozen": raw.get("frozen_excluding_authorized_state_sha256") == _FROZEN_SHA256,
    }
    if not all(required.values()):
        raise ValueError(f"V44 exact contract changed: {required}")
    source_files = dict(_mapping(raw.get("source_file_sha256"), "V44 source hashes"))
    if source_files != _SOURCE_FILES:
        raise ValueError("V44 source file hashes changed")
    terminal_report = _resolve(str(raw["v43_terminal_report"]))
    source_checkpoint = _resolve(str(raw["source_checkpoint"]))
    if terminal_report != _resolve(_V43_TERMINAL_PATH):
        raise ValueError("V44 terminal path differs from its exact authorization")
    if source_checkpoint != _resolve(_SOURCE_CHECKPOINT):
        raise ValueError("V44 source checkpoint path differs from its exact authorization")
    return V44Contract(
        terminal_report=terminal_report,
        configured_terminal_sha256=str(raw["v43_terminal_report_sha256"]),
        source_checkpoint=source_checkpoint,
        source_file_sha256=source_files,
        source_full_state_sha256=_SOURCE_FULL_SHA256,
        frozen_state_sha256=_FROZEN_SHA256,
        authorized_parameter_names=names,
        authorized_parameter_shapes=shapes,
        total_parameter_count=415_744,
    )


def require_v43_terminal_gate(
    config: Mapping[str, Any], *, expected_sha256: str
) -> dict[str, Any]:
    contract = v44_contract(config)
    if _HEX.fullmatch(expected_sha256) is None or expected_sha256 != _V43_TERMINAL_SHA256:
        raise ValueError("V44 requires the exact pinned V43 terminal SHA-256")
    if contract.configured_terminal_sha256 != _V43_TERMINAL_SHA256:
        raise ValueError("V44 configured V43 terminal hash changed")
    path = contract.terminal_report
    if path.is_symlink() or not path.is_file() or _sha256(path) != expected_sha256:
        raise ValueError("V44 requires the exact real V43 terminal seal")
    report = json.loads(path.read_text(encoding="utf-8"))
    authorization = _mapping(
        _mapping(report, "V43 terminal").get("conditional_successor_authorization"),
        "V44 successor authorization",
    )
    expected_authorization = {
        "authorization_id": "v44_joint_scene_readout_train_only_pilot",
        "authorized": True,
        "authorized_config": str(DEFAULT_CONFIG),
        "authorized_output_root": str(DEFAULT_OUTPUT),
        "frozen_excluding_authorized_state_sha256": _FROZEN_SHA256,
        "objective": {
            "broad_nll_weight": 0.25,
            "cross_prefix_flip_margin": 0.1,
            "cross_prefix_flip_weight": 8.0,
            "pair_correct_nll_weight": 0.5,
            "side_hinge_margin": 0.5,
            "side_hinge_weight": 8.0,
            "source_prefix_trust_scale": 0.05,
            "source_prefix_trust_weight": 0.001,
        },
        "only_exact_action": "one_bounded_v44_joint_scene_readout_training_pilot",
        "optimizer": {
            "foreach": False,
            "fused": False,
            "implementation": "fresh_torch_adamw_two_groups",
            "per_group_gradient_clip_norm": 1.0,
            "query_learning_rate": 2e-5,
            "scene_readout_learning_rate": 2.5e-5,
            "source_optimizer_loaded": False,
            "weight_decay": 0.0,
        },
        "schedule": {
            "checkpoint_steps": list(_SAVED_STEPS),
            "maximum_optimizer_updates": 16,
            "true_optimizer_step_per_schedule_row": True,
            "update4_is_diagnostic_only": True,
            "update8_must_pass_before_updates_9_through_16": True,
        },
        "schema_version": 1,
        "scope": {
            "all_occupied_blocks_processed": True,
            "final_test_access_authorized": False,
            "new_terminal_seal_required_after_training": True,
            "oracle_access_authorized": False,
            "question_dependent_retrieval": False,
            "question_dependent_scene_processing": False,
            "runtime_promotion_authorized": False,
            "selector_execution_authorized": False,
            "training_qa_and_maps_only": True,
            "validation_access_authorized": False,
        },
        "source_authorized_surface_state_sha256": _SOURCE_AUTHORIZED_SHA256,
        "source_checkpoint": (
            "data_gemma4/checkpoints/"
            "gemma4_v41_retry1_diverse28_projected_gradient_l14_query/update_000"
        ),
        "source_file_sha256": dict(_SOURCE_FILES),
        "source_full_tensor_state_sha256": _SOURCE_FULL_SHA256,
        "trainable_surface": {
            "block_qkv_frozen": True,
            "gemma_base_and_all_other_lora_banks_frozen": True,
            "parameter_names": list(_PARAMETER_NAMES),
            "parameter_shapes": [list(value) for value in _PARAMETER_SHAPES],
            "query_parameter_count": 22_528,
            "scene_readout_parameter_count": 393_216,
            "total_parameter_count": 415_744,
        },
        "update16_gate": {
            "book_or_picture_complete_units_minimum": 1,
            "broad_greedy_exact_correct_minimum": 23,
            "broad_nll_maximum_increase": 0.02,
            "complete_physical_pair_coverage_minimum": 5,
            "complete_units_minimum": 10,
            "cross_prefix_complete_units_minimum": 17,
            "greedy_complete_units_minimum": 5,
            "positive_sides_minimum": 35,
            "priority_side_deficit_minimum_improvement": 0.5,
            "require_update8_passed": True,
        },
        "update8_gate": {
            "both_authorized_parameter_groups_must_change": True,
            "broad_nll_maximum_increase": 0.02,
            "complete_units_minimum": 9,
            "cross_prefix_complete_units_minimum": 17,
            "frozen_state_must_remain_exact": True,
            "positive_sides_minimum": 34,
            "priority_side_deficit_minimum_improvement": 0.5,
        },
    }
    top_level_checks = {
        "schema_version": report.get("schema_version") == 1,
        "artifact": report.get("artifact")
        == "v43_aggregate_projected_screen_terminal_gate",
        "passed": report.get("passed") is True,
        "successor": report.get("only_exact_successor_authorized")
        == "v44_joint_scene_readout_train_only_pilot",
        "v44_authorized": report.get("v44_train_only_pilot_authorized") is True,
        "validation": report.get("validation_access_authorized") is False,
        "selector": report.get("selector_execution_authorized") is False,
        "promotion": report.get("runtime_promotion_authorized") is False,
        "authorization_exact": dict(authorization) == expected_authorization,
    }
    if not all(top_level_checks.values()):
        raise ValueError(f"V43 does not authorize exact V44: {top_level_checks}")
    return {
        "path": str(path),
        "sha256": expected_sha256,
        "authorization": dict(authorization),
        "exact_authorization_fields_verified": True,
    }


def _source_tensors(contract: V44Contract) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    for name, expected in contract.source_file_sha256.items():
        path = contract.source_checkpoint / name
        if path.is_symlink() or not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"V44 exact source file changed: {name}")
    metadata = json.loads(
        (contract.source_checkpoint / TRAINING_METADATA_FILENAME).read_text(encoding="utf-8")
    )
    runtime = json.loads(
        (contract.source_checkpoint / RUNTIME_METADATA_FILENAME).read_text(encoding="utf-8")
    )
    validate_runtime_checkpoint_metadata(runtime)
    if runtime != runtime_checkpoint_metadata(metadata):
        raise ValueError("V44 source runtime metadata is not freshly sanitized")
    tensors = load_file(contract.source_checkpoint / "adapter.safetensors", device="cpu")
    if tensor_state_sha256(tensors) != contract.source_full_state_sha256:
        raise ValueError("V44 source tensor state changed")
    observed = tuple(name for name in _PARAMETER_NAMES if name in tensors)
    if observed != _PARAMETER_NAMES:
        raise ValueError("V44 authorized source tensor inventory changed")
    if tuple(tuple(tensors[name].shape) for name in observed) != _PARAMETER_SHAPES:
        raise ValueError("V44 authorized source tensor shapes changed")
    if tuple(int(tensors[name].numel()) for name in observed) != _PARAMETER_COUNTS:
        raise ValueError("V44 authorized parameter counts changed")
    if (
        tensor_state_sha256({name: tensors[name] for name in _PARAMETER_NAMES})
        != _SOURCE_AUTHORIZED_SHA256
    ):
        raise ValueError("V44 authorized source surface changed")
    frozen = {name: value for name, value in tensors.items() if name not in _PARAMETER_NAMES}
    if tensor_state_sha256(frozen) != contract.frozen_state_sha256:
        raise ValueError("V44 frozen source tensor state changed")
    return tensors, metadata


def _query_adapter(bundle: Any) -> torch.nn.Module:
    return _query_bank(bundle).adapters[1]


def v44_named_parameters(bundle: Any, block_core: BlockCrossResidual) -> dict[str, torch.nn.Parameter]:
    adapter = _query_adapter(bundle)
    values = {
        _PARAMETER_NAMES[0]: block_core.w_o,
        _PARAMETER_NAMES[1]: adapter.lora_a,
        _PARAMETER_NAMES[2]: adapter.lora_b,
    }
    if tuple(values) != _PARAMETER_NAMES:
        raise RuntimeError("V44 trainable parameter order changed")
    if tuple(tuple(value.shape) for value in values.values()) != _PARAMETER_SHAPES:
        raise RuntimeError("V44 live trainable parameter shapes changed")
    return values


def configure_v44_decoder_training_mode(bundle: Any) -> dict[str, Any]:
    """Keep frozen Gemma in train mode when recomputation is configured."""

    decoder = bundle.language.decoder_module
    checkpointing = bool(bundle.language.decoder_gradient_checkpointing_enabled)
    decoder.train(checkpointing)
    if checkpointing and (
        not decoder.training
        or not bool(getattr(decoder, "is_gradient_checkpointing", True))
    ):
        raise RuntimeError("V44 decoder gradient checkpointing is inactive")
    return {
        "decoder_training": bool(decoder.training),
        "decoder_gradient_checkpointing_enabled": checkpointing,
        "recomputation_active": checkpointing and bool(decoder.training),
    }


def freeze_for_v44(bundle: Any, block_core: BlockCrossResidual) -> dict[str, torch.nn.Parameter]:
    bundle.language.model.requires_grad_(False).eval()
    for module in bundle.checkpoint_modules.values():
        module.requires_grad_(False).eval()
    configure_v44_decoder_training_mode(bundle)
    _query_bank(bundle).train(True)
    named = v44_named_parameters(bundle, block_core)
    for parameter in named.values():
        parameter.requires_grad_(True)
        parameter.grad = None
    return named


def frozen_v44_state_sha256(bundle: Any) -> str:
    state = {
        f"{module_name}.{name}": value
        for module_name, module in bundle.checkpoint_modules.items()
        for name, value in module.state_dict().items()
        if f"{module_name}.{name}" not in _PARAMETER_NAMES
    }
    return tensor_state_sha256(state)


def block_source_stack_state_sha256(
    bundle: Any, block_core: BlockCrossResidual
) -> str:
    """Hash the current non-core stack exactly as chat runtime validates it."""

    modules = {
        name: module
        for name, module in bundle.checkpoint_modules.items()
        if module is not block_core and name != "block_cross_residual"
    }
    if not modules:
        raise ValueError("V44 block-cross source stack cannot be empty")
    return module_collection_state_sha256(modules)


def assert_v44_trainable_surface(
    bundle: Any,
    block_core: BlockCrossResidual,
    *,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, Any]:
    named = v44_named_parameters(bundle, block_core)
    expected_ids = {id(value) for value in named.values()}
    observed = {
        id(parameter)
        for module in bundle.checkpoint_modules.values()
        for parameter in module.parameters()
        if parameter.requires_grad
    }
    language_observed = {
        id(parameter)
        for parameter in bundle.language.model.parameters()
        if parameter.requires_grad
    }
    if observed != expected_ids or sum(value.numel() for value in named.values()) != 415_744:
        raise RuntimeError("V44 trainable surface escaped its exact three tensors")
    if language_observed != {id(named[name]) for name in _PARAMETER_NAMES[1:]}:
        raise RuntimeError("V44 Gemma trainable surface escaped the exact query tensors")
    if optimizer is not None:
        optimizer_ids = {
            id(parameter) for group in optimizer.param_groups for parameter in group["params"]
        }
        if optimizer_ids != expected_ids:
            raise RuntimeError("V44 optimizer contains an unauthorized parameter")
    decoder_mode = configure_v44_decoder_training_mode(bundle)
    return {
        "parameter_names": list(named),
        "parameter_shapes": [list(value.shape) for value in named.values()],
        "scene_readout_parameter_count": int(block_core.w_o.numel()),
        "query_parameter_count": int(
            _query_adapter(bundle).lora_a.numel() + _query_adapter(bundle).lora_b.numel()
        ),
        "trainable_parameter_count": sum(int(value.numel()) for value in named.values()),
        "trainable_tensor_count": 3,
        "everything_else_frozen": True,
        **decoder_mode,
    }


def v44_optimizer(
    scene_readout: Sequence[torch.nn.Parameter],
    query: Sequence[torch.nn.Parameter],
    settings: V44Settings,
) -> torch.optim.AdamW:
    if len(scene_readout) != 1 or len(query) != 2:
        raise ValueError("V44 optimizer requires one scene and two query tensors")
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
        raise RuntimeError("V44 AdamW must start with empty state")
    return optimizer


def v44_optimizer_audit(optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    groups = optimizer.param_groups
    if len(groups) != 2 or [group.get("name") for group in groups] != [
        "scene_readout",
        "layer14_query",
    ]:
        raise ValueError("V44 optimizer group inventory changed")
    expected = ((2.5e-5, 1), (2.0e-5, 2))
    if any(
        float(group["lr"]) != lr
        or float(group["weight_decay"]) != 0.0
        or len(group["params"]) != count
        or group.get("foreach") is not False
        or group.get("fused") is not False
        for group, (lr, count) in zip(groups, expected, strict=True)
    ):
        raise ValueError("V44 optimizer group settings changed")
    return {
        "implementation": "torch.optim.AdamW",
        "group_names": ["scene_readout", "layer14_query"],
        "learning_rates": [2.5e-5, 2.0e-5],
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


def source_prefix_trust_penalty(
    *,
    caches: Mapping[str, Any],
    references: Mapping[str, torch.Tensor],
    block_core: BlockCrossResidual,
    device: torch.device,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if tuple(sorted(caches)) != tuple(sorted(references)) or len(caches) != 16:
        raise ValueError("V44 source-prefix trust inventory changed")
    squared = []
    for scene_id in sorted(caches):
        current = current_scene_tokens(caches[scene_id], block_core, device=device)
        reference = references[scene_id].to(device=device, dtype=current.dtype)
        squared.append((current - reference).square().mean())
    mean_square = torch.stack(squared).mean()
    return mean_square / (scale**2), mean_square.sqrt()


def _family_counts(pair_metrics: Mapping[str, Any]) -> Mapping[str, Any]:
    return _mapping(pair_metrics.get("complete_units_by_family"), "V44 family counts")


def v44_update8_gate(
    *,
    pair_metrics: Mapping[str, Any],
    broad_nll: float,
    source_broad_nll: float,
    source_priority_deficit: float,
    scene_readout_state_changed: bool,
    query_state_changed: bool,
    frozen_state_exact: bool,
    trust_rms: float,
) -> dict[str, Any]:
    deficit = float(priority_side_deficit(pair_metrics)["combined"])
    checks = {
        "priority_teacher_deficit_improved_at_least_0_5": source_priority_deficit
        - deficit
        >= 0.5,
        "teacher_complete_units_at_least_9": int(pair_metrics["complete_units"]) >= 9,
        "teacher_positive_sides_at_least_34": int(pair_metrics["positive_sides"]) >= 34,
        "teacher_cross_complete_units_at_least_17": int(
            pair_metrics["cross_prefix_complete_units"]
        )
        >= 17,
        "broad_nll_within_source_plus_0_02": broad_nll <= source_broad_nll + 0.02,
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
        "priority_teacher_side_deficit_improvement": source_priority_deficit - deficit,
        "broad_nll_delta_from_update_zero": broad_nll - source_broad_nll,
        "source_prefix_trust_rms": trust_rms,
        "training_scenes_only": True,
        "validation_qa_loaded": False,
    }


def v44_update16_gate(
    *,
    update8_gate: Mapping[str, Any],
    pair_metrics: Mapping[str, Any],
    broad_nll: float,
    source_broad_nll: float,
    source_priority_deficit: float,
    greedy_metrics: Mapping[str, Any],
    scene_readout_state_changed: bool,
    query_state_changed: bool,
    frozen_state_exact: bool,
    trust_rms: float,
) -> dict[str, Any]:
    deficit = float(priority_side_deficit(pair_metrics)["combined"])
    families = _family_counts(pair_metrics)
    checks = {
        "update8_gate_remains_passed": update8_gate.get("passed") is True,
        "priority_teacher_deficit_improved_at_least_0_5": source_priority_deficit - deficit
        >= 0.5,
        "teacher_complete_units_at_least_10": int(pair_metrics["complete_units"]) >= 10,
        "teacher_positive_sides_at_least_35": int(pair_metrics["positive_sides"]) >= 35,
        "complete_physical_pair_coverage_at_least_5": int(
            pair_metrics["complete_physical_pair_coverage"]
        )
        >= 5,
        "book_or_picture_teacher_complete": int(families.get("book_support", 0))
        + int(families.get("picture_support", 0))
        >= 1,
        "teacher_cross_complete_units_at_least_17": int(
            pair_metrics["cross_prefix_complete_units"]
        )
        >= 17,
        "train_greedy_complete_units_at_least_5": int(greedy_metrics["complete_units"])
        >= 5,
        "broad_greedy_exact_correct_at_least_23_of_48": int(
            greedy_metrics["broad_exact_correct"]
        )
        >= 23
        and int(greedy_metrics["broad_row_count"]) == 48,
        "broad_nll_within_source_plus_0_02": broad_nll <= source_broad_nll + 0.02,
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
        "priority_teacher_side_deficit_improvement": source_priority_deficit - deficit,
        "broad_nll_delta_from_update_zero": broad_nll - source_broad_nll,
        "source_prefix_trust_rms": trust_rms,
        "training_greedy_metrics": dict(greedy_metrics),
        "training_scenes_only": True,
        "validation_qa_loaded": False,
        "selector_execution_authorized": False,
    }


def v44_stop_reason(
    optimizer_step: int,
    *,
    update8_gate: Mapping[str, Any] | None,
    update16_gate: Mapping[str, Any] | None,
) -> str | None:
    """Return the exact bounded-stop reason after a checkpointed gate."""

    if optimizer_step == 8:
        if update8_gate is None:
            raise RuntimeError("V44 update-8 stop decision lacks its gate")
        return (
            None
            if update8_gate.get("passed") is True
            else "update8_train_only_gate_failed"
        )
    if optimizer_step == 16:
        if update16_gate is None:
            raise RuntimeError("V44 update-16 stop decision lacks its gate")
        return (
            None
            if update16_gate.get("passed") is True
            else "update16_train_only_gate_failed"
        )
    return None


def v44_saved_optimizer_steps(history: Sequence[Mapping[str, Any]]) -> list[int]:
    """Report only checkpoints that the bounded run actually materialized."""

    return [
        int(row["optimizer_update"])
        for row in history
        if row.get("saved_checkpoint") is True
    ]


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
        roots.extend(path.resolve() for path in maps_root.iterdir() if path.name not in allowed)
    roots.extend(path.resolve() for path in PROJECT_ROOT.rglob("optimizer.pt"))
    return roots


def preflight_v44(
    config_path: str | Path = DEFAULT_CONFIG,
    *,
    v43_terminal_sha256: str,
) -> dict[str, Any]:
    config_path = _resolve(config_path)
    if config_path != _resolve(DEFAULT_CONFIG) or _sha256(config_path) != _CONFIG_FILE_SHA256:
        raise ValueError("V44 config path or bytes differ from the exact authorization")
    config = load_config(config_path)
    settings = v44_settings(config)
    contract = v44_contract(config)
    terminal = require_v43_terminal_gate(config, expected_sha256=v43_terminal_sha256)
    loader = v41_loader_config(config)
    assert_deferred_final_scenes_absent(loader)
    audit = FileAccessAudit(
        _preflight_forbidden_roots(config),
        forbidden_component_names={"oracle"},
        block_forbidden=True,
    )
    with audit:
        tensors, metadata = _source_tensors(contract)
        records, qa_audit = load_v35_train_qa_records(loader)
        units = build_exact_question_pair_units(records)
        schedule, schedule_audit = build_v41_schedule(records, units, seed=int(config["seed"]))
    audit.assert_clean()
    if len(records) != 384 or len(units) != 25 or len(schedule[:16]) != 16:
        raise RuntimeError("V44 exact train-only data/schedule inventory changed")
    return {
        "schema_version": 1,
        "artifact": "v44_joint_scene_readout_preflight",
        "passed": True,
        "config_path": str(config_path),
        "config_hash": config_hash(dict(config)),
        "terminal": terminal,
        "source_checkpoint": str(contract.source_checkpoint),
        "source_tensor_count": len(tensors),
        "source_full_tensor_state_sha256": tensor_state_sha256(tensors),
        "source_metadata_optimizer_step": metadata.get("optimizer_step"),
        "trainable_parameter_names": list(contract.authorized_parameter_names),
        "trainable_parameter_shapes": [list(value) for value in contract.authorized_parameter_shapes],
        "trainable_parameter_count": contract.total_parameter_count,
        "settings": settings.__dict__,
        "train_question_count": len(records),
        "train_pair_unit_count": len(units),
        "bounded_schedule_steps": [row.optimizer_step for row in schedule[:16]],
        "full_schedule_sha256": schedule_audit["schedule_sha256"],
        "qa_audit": qa_audit,
        "gemma_loaded": False,
        "scene_maps_loaded": False,
        "validation_qa_loaded": False,
        "oracle_loaded": False,
        "final_test_scenes_touched": False,
        "selector_execution_authorized": False,
        "loaded_files": audit.unique_paths,
        "forbidden_file_accesses": audit.forbidden_accesses(),
    }


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temporary_path = Path(temporary)
    try:
        temporary_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


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
    gate8: Mapping[str, Any] | None,
    gate16: Mapping[str, Any] | None,
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
            # Both authorized V44 groups are part of hashes enforced by the
            # production chat loader.  Refresh them from the live checkpoint
            # state rather than inheriting stale V41 update-zero metadata.
            "block_cross_residual_state_sha256": block_core.state_sha256(),
            "frozen_block_cross_source_stack_state_sha256": (
                block_source_stack_state_sha256(bundle, block_core)
            ),
        }
    )
    result["v44_joint_scene_readout"] = {
        "schema_version": 1,
        "optimizer_step": optimizer_step,
        "conditional_v43_terminal_gate": {
            "path": terminal["path"],
            "sha256": terminal["sha256"],
        },
        "conditional_authorization": dict(terminal["authorization"]),
        "source_checkpoint": str(v44_contract(config).source_checkpoint),
        "source_full_tensor_state_sha256": _SOURCE_FULL_SHA256,
        "frozen_excluding_authorized_source_state_sha256": source_frozen_hash,
        "frozen_excluding_authorized_state_sha256": frozen_v44_state_sha256(bundle),
        "trainable_surface": assert_v44_trainable_surface(bundle, block_core),
        "source_prefix_sha256_by_train_scene": dict(source_prefix_hashes),
        "update8_train_only_gate": None if gate8 is None else dict(gate8),
        "update16_train_only_gate": None if gate16 is None else dict(gate16),
        "validation_qa_loaded": False,
        "oracle_environment_files_loaded": False,
        "deferred_final_scene_ids_loaded": [],
        "question_dependent_scene_processing": False,
        "question_dependent_retrieval": False,
        "independent_selector_required": True,
    }
    return result


def _save(
    path: Path,
    *,
    bundle: Any,
    metadata: dict[str, Any],
    optimizer: torch.optim.Optimizer | None,
) -> None:
    if path.is_symlink() or (path.exists() and not path.is_dir()):
        raise ValueError(f"V44 checkpoint destination is unsafe: {path}")
    save_adapter_checkpoint(path, bundle.checkpoint_modules, metadata)
    if optimizer is not None:
        save_optimizer_checkpoint(path, optimizer)


def _run_v44_impl(
    *,
    config_path: str | Path = DEFAULT_CONFIG,
    output: str | Path = DEFAULT_OUTPUT,
    v43_terminal_sha256: str,
) -> dict[str, Any]:
    config_path = _resolve(config_path)
    output = _resolve(output)
    if config_path != _resolve(DEFAULT_CONFIG) or _sha256(config_path) != _CONFIG_FILE_SHA256:
        raise ValueError("V44 config path or bytes differ from the exact authorization")
    config = load_config(config_path)
    settings = v44_settings(config)
    contract = v44_contract(config)
    terminal = require_v43_terminal_gate(config, expected_sha256=v43_terminal_sha256)
    if output != _resolve(DEFAULT_OUTPUT):
        raise ValueError("V44 output differs from its exact bounded namespace")
    if output.is_symlink() or (output.exists() and not output.is_dir()):
        raise FileExistsError("V44 output is not a real directory")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("V44 refuses to overwrite a nonempty output")
    source_tensors, source_checkpoint_metadata = _source_tensors(contract)
    loader = v41_loader_config(config)
    assert_deferred_final_scenes_absent(loader)
    records, qa_audit = load_v35_train_qa_records(loader)
    units = build_exact_question_pair_units(records)
    schedule, schedule_audit = build_v41_schedule(records, units, seed=int(config["seed"]))
    schedule = schedule[: settings.optimizer_steps]
    inherited_schedule, _ = build_v35_schedule(
        records, units, settings=v35_settings(loader), seed=int(config["seed"])
    )
    broad_records = v36_broad_calibration_records(inherited_schedule)
    if len(records) != 384 or len(units) != 25 or len(schedule) != 16 or len(broad_records) != 48:
        raise RuntimeError("V44 training inventory changed")

    source_audit = {
        "source_checkpoint": str(contract.source_checkpoint),
        "source_file_sha256": dict(contract.source_file_sha256),
        "source_full_tensor_state_sha256": tensor_state_sha256(source_tensors),
        "source_authorized_surface_state_sha256": tensor_state_sha256(
            {name: source_tensors[name] for name in _PARAMETER_NAMES}
        ),
        "source_optimizer_file_opened": False,
        "source_optimizer_state_loaded": False,
        "exact_v41_retry1_update_000_loaded": True,
    }
    approved = require_approved_v29_source(loader)
    bundle, block_core, loaded_metadata, loader_transition = load_v41_bundle(
        config, approved, contract.source_checkpoint, source_tensors
    )
    if loaded_metadata != source_checkpoint_metadata:
        raise RuntimeError("V44 loaded metadata differs from authenticated V41 retry1 u0")
    if module_collection_state_sha256(bundle.checkpoint_modules) != _SOURCE_FULL_SHA256:
        raise RuntimeError("V44 live update zero differs from exact source")
    named = freeze_for_v44(bundle, block_core)
    if frozen_v44_state_sha256(bundle) != _FROZEN_SHA256:
        raise RuntimeError("V44 live frozen state differs at construction")
    surface = assert_v44_trainable_surface(bundle, block_core)

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
    with torch.inference_mode():
        source_scene_tokens = {
            scene_id: current_scene_tokens(caches[scene_id], block_core, device=bundle.language.device)
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
        records=broad_records, caches=caches, block_cross_residual=block_core, bundle=bundle
    )
    source_deficit = float(priority_side_deficit(source_pair)["combined"])
    source_authorized_hash = tensor_state_sha256(
        {name: value.detach().cpu() for name, value in named.items()}
    )
    source_scene_readout_hash = tensor_state_sha256(
        {_PARAMETER_NAMES[0]: named[_PARAMETER_NAMES[0]].detach().cpu()}
    )
    source_query_hash = tensor_state_sha256(
        {
            name: named[name].detach().cpu()
            for name in (_PARAMETER_NAMES[1], _PARAMETER_NAMES[2])
        }
    )
    if source_authorized_hash != _SOURCE_AUTHORIZED_SHA256:
        raise RuntimeError("V44 live authorized surface differs from exact V41 retry1 u0")
    optimizer = v44_optimizer(
        [named[_PARAMETER_NAMES[0]]],
        [named[_PARAMETER_NAMES[1]], named[_PARAMETER_NAMES[2]]],
        settings,
    )
    optimizer_audit = v44_optimizer_audit(optimizer)
    assert_v44_trainable_surface(bundle, block_core, optimizer=optimizer)
    history: list[dict[str, Any]] = [
        {
            "optimizer_update": 0,
            "source_pair_metrics": source_pair,
            "source_per_unit_nll_diagnostics": source_nll,
            "source_broad_train_nll": source_broad,
            "source_prefix_trust_rms": 0.0,
            "authorized_state_sha256": source_authorized_hash,
            "scene_readout_state_sha256": source_scene_readout_hash,
            "query_state_sha256": source_query_hash,
            "frozen_state_sha256": _FROZEN_SHA256,
            "validation_qa_loaded": False,
            "oracle_environment_files_loaded": False,
            "saved_checkpoint": True,
        }
    ]
    gate8: Mapping[str, Any] | None = None
    gate16: Mapping[str, Any] | None = None
    completed_steps = 0
    stop_reason: str | None = None
    output.mkdir(parents=True, exist_ok=True)
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
        gate8=None,
        gate16=None,
    )
    _save(output / "update_000", bundle=bundle, metadata=metadata0, optimizer=None)

    for item in schedule:
        step = item.optimizer_step
        freeze_for_v44(bundle, block_core)
        assert_v44_trainable_surface(bundle, block_core, optimizer=optimizer)
        optimizer.zero_grad(set_to_none=True)
        broad_tokens = current_scene_tokens(
            caches[item.broad_record.scene_id], block_core, device=bundle.language.device
        )
        broad = broad_answer_nll(
            scene_tokens=broad_tokens, record=item.broad_record, bundle=bundle
        )
        broad_value = float(broad.detach().cpu())
        (settings.broad_nll_weight * broad).backward()
        del broad, broad_tokens

        pair_tokens = {
            scene_id: current_scene_tokens(caches[scene_id], block_core, device=bundle.language.device)
            for scene_id in item.pair_unit.scene_ids
        }
        pair_nll, side_hinge, cross_hinge, _diagnostics = paired_cross_prefix_objective(
            unit=item.pair_unit,
            scene_tokens=pair_tokens,
            bundle=bundle,
            side_margin=settings.side_hinge_margin,
            cross_prefix_margin=settings.cross_prefix_flip_margin,
        )
        pair_nll_value = float(pair_nll.detach().cpu())
        side_hinge_value = float(side_hinge.detach().cpu())
        cross_hinge_value = float(cross_hinge.detach().cpu())
        pair_loss = (
            settings.pair_correct_nll_weight * pair_nll
            + settings.side_hinge_weight * side_hinge
            + settings.cross_prefix_flip_weight * cross_hinge
        )
        pair_loss.backward()
        del pair_loss, pair_nll, side_hinge, cross_hinge, pair_tokens, _diagnostics
        if bundle.language.device.type == "mps":
            torch.mps.empty_cache()

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
            + settings.side_hinge_weight * side_hinge_value
            + settings.cross_prefix_flip_weight * cross_hinge_value
            + settings.source_prefix_trust_weight * trust_value
        )
        if any(
            parameter.grad is None
            or not torch.isfinite(parameter.grad).all()
            or torch.count_nonzero(parameter.grad).item() == 0
            for parameter in named.values()
        ):
            raise RuntimeError("V44 active gradient is absent, zero, or nonfinite")
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
            raise RuntimeError("V44 per-group gradient norm is nonfinite")
        optimizer.step()
        completed_steps = step
        if frozen_v44_state_sha256(bundle) != _FROZEN_SHA256:
            raise RuntimeError("V44 changed a frozen tensor or buffer")

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
        if step in (4, 8, 16):
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
        if step == 16:
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
        scene_readout_hash = tensor_state_sha256(
            {_PARAMETER_NAMES[0]: named[_PARAMETER_NAMES[0]].detach().cpu()}
        )
        query_hash = tensor_state_sha256(
            {
                name: named[name].detach().cpu()
                for name in (_PARAMETER_NAMES[1], _PARAMETER_NAMES[2])
            }
        )
        scene_readout_changed = scene_readout_hash != source_scene_readout_hash
        query_changed = query_hash != source_query_hash
        if step == 8:
            assert pair_metrics is not None and broad_diagnostic is not None
            gate8 = v44_update8_gate(
                pair_metrics=pair_metrics,
                broad_nll=broad_diagnostic,
                source_broad_nll=source_broad,
                source_priority_deficit=source_deficit,
                scene_readout_state_changed=scene_readout_changed,
                query_state_changed=query_changed,
                frozen_state_exact=True,
                trust_rms=trust_rms_value,
            )
        if step == 16:
            assert (
                gate8 is not None
                and pair_metrics is not None
                and broad_diagnostic is not None
                and greedy_diagnostic is not None
            )
            gate16 = v44_update16_gate(
                update8_gate=gate8,
                pair_metrics=pair_metrics,
                broad_nll=broad_diagnostic,
                source_broad_nll=source_broad,
                source_priority_deficit=source_deficit,
                greedy_metrics=greedy_diagnostic,
                scene_readout_state_changed=scene_readout_changed,
                query_state_changed=query_changed,
                frozen_state_exact=True,
                trust_rms=trust_rms_value,
            )
        history.append(
            {
                "optimizer_update": step,
                "true_optimizer_step": True,
                "train_loss": loss_value,
                "train_broad_nll": broad_value,
                "train_pair_correct_nll": pair_nll_value,
                "train_side_hinge": side_hinge_value,
                "train_cross_prefix_hinge": cross_hinge_value,
                "train_source_prefix_trust_penalty": trust_value,
                "source_prefix_trust_rms": trust_rms_value,
                "scene_readout_preclip_gradient_norm": scene_preclip,
                "query_preclip_gradient_norm": query_preclip,
                "per_group_gradient_clip_norm": settings.gradient_clip_norm,
                "pair_metrics": pair_metrics,
                "per_unit_nll_diagnostics": per_unit_nll,
                "broad_diagnostic_nll": broad_diagnostic,
                "training_greedy_metrics": greedy_diagnostic,
                "update8_train_only_gate": None if gate8 is None else dict(gate8),
                "update16_train_only_gate": None if gate16 is None else dict(gate16),
                "authorized_state_sha256": authorized_hash,
                "scene_readout_state_sha256": scene_readout_hash,
                "query_state_sha256": query_hash,
                "scene_readout_state_changed": scene_readout_changed,
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
            gate8=gate8,
            gate16=gate16,
        )
        _save(output / f"update_{step:03d}", bundle=bundle, metadata=metadata, optimizer=optimizer)
        print(
            json.dumps(
                {
                    "phase": "v44_joint_scene_readout_checkpoint",
                    "optimizer_step": step,
                    "update8_gate_passed": None if gate8 is None else gate8.get("passed"),
                    "update16_gate_passed": None if gate16 is None else gate16.get("passed"),
                    "source_prefix_trust_rms": trust_rms_value,
                    "validation_qa_loaded": False,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        stop_reason = v44_stop_reason(
            step, update8_gate=gate8, update16_gate=gate16
        )
        if stop_reason is not None:
            break

    return {
        "schema_version": 1,
        "artifact": "v44_joint_scene_readout_train_only_pilot",
        "passed": gate16 is not None and gate16.get("passed") is True,
        "bounded_training_completed": gate16 is not None,
        "stopped_at_train_only_gate": stop_reason,
        "output": str(output),
        "optimizer_updates": completed_steps,
        "saved_optimizer_steps": v44_saved_optimizer_steps(history),
        "terminal": terminal,
        "optimizer": optimizer_audit,
        "trainable_surface": surface,
        "source_audit": source_audit,
        "loader_transition": loader_transition,
        "cache_audit": cache_audit,
        "cache_boundary": cache_boundary,
        "qa_audit": qa_audit,
        "schedule_sha256": schedule_audit["schedule_sha256"],
        "update8_train_only_gate": gate8,
        "update16_train_only_gate": gate16,
        "validation_qa_loaded": False,
        "oracle_loaded": False,
        "final_test_scenes_touched": False,
        "selector_execution_authorized": False,
        "runtime_promotion_authorized": False,
    }


def run_v44(
    *,
    config_path: str | Path = DEFAULT_CONFIG,
    output: str | Path = DEFAULT_OUTPUT,
    v43_terminal_sha256: str,
) -> dict[str, Any]:
    """Run V44 under a process-wide train-only read audit."""

    resolved_config = _resolve(config_path)
    config = load_config(resolved_config)
    protected = _resolve(_PROTECTED_REPORT)
    if protected.is_symlink() or not protected.is_file():
        raise ValueError("V44 protected selection report is unavailable or unsafe")
    protected_before = _sha256(protected)
    if protected_before != _PROTECTED_REPORT_SHA256:
        raise ValueError("V44 protected selection report changed before training")
    audit = FileAccessAudit(
        _training_forbidden_roots(config),
        forbidden_component_names={"oracle"},
        block_forbidden=True,
    )
    with audit:
        result = _run_v44_impl(
            config_path=resolved_config,
            output=output,
            v43_terminal_sha256=v43_terminal_sha256,
        )
    audit.assert_clean()
    if _sha256(protected) != protected_before:
        raise RuntimeError("V44 changed the protected selection report")
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
        raise RuntimeError("V44 file audit did not observe exactly all 16 training maps")
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
    parser.add_argument("--v43-terminal-sha256", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    result = (
        preflight_v44(args.config, v43_terminal_sha256=args.v43_terminal_sha256)
        if args.preflight_only
        else run_v44(
            config_path=args.config,
            output=args.output,
            v43_terminal_sha256=args.v43_terminal_sha256,
        )
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0 if result.get("passed") is True else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "V44Contract",
    "V44Settings",
    "assert_v44_trainable_surface",
    "block_source_stack_state_sha256",
    "configure_v44_decoder_training_mode",
    "freeze_for_v44",
    "frozen_v44_state_sha256",
    "preflight_v44",
    "require_v43_terminal_gate",
    "run_v44",
    "source_prefix_trust_penalty",
    "v44_contract",
    "v44_optimizer",
    "v44_optimizer_audit",
    "v44_saved_optimizer_steps",
    "v44_settings",
    "v44_stop_reason",
    "v44_update8_gate",
    "v44_update16_gate",
]
