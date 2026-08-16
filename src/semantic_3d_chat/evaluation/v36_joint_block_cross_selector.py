"""Select a completed V36 joint block-cross/query-LoRA development arm.

This module is deliberately a post-training process.  It validates the entire
bounded checkpoint envelope, including tensor transitions, sanitized runtime
metadata, fresh staged Adam state, and all training-only continuation gates,
before constructing an evaluator or opening validation QA.  Deferred final
scenes remain inaccessible.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

import torch
from safetensors.torch import load_file

from semantic_3d_chat.config import config_hash, load_config
from semantic_3d_chat.evaluation.v27_sidecar_screen import _atomic_json
from semantic_3d_chat.evaluation.v28_stage_b_selector import _retention_control_config
from semantic_3d_chat.evaluation.v30_joint_pair_selector import (
    SelectionRequirements,
    _metadata,
    _selection_requirements,
    _source_v29_evidence,
    _validate_source_against_config,
)
from semantic_3d_chat.evaluation.v35_block_cross_selector import (
    V35GreedyEvidence,
    V35TeacherEvidence,
    _promotion,
    _V35RuntimeEvaluator,
)
from semantic_3d_chat.language.lora import tensor_state_sha256
from semantic_3d_chat.scene_encoder.block_cross_residual import (
    block_cross_residual_settings,
)
from semantic_3d_chat.training.checkpointing import (
    RUNTIME_METADATA_FILENAME,
    TRAINING_METADATA_FILENAME,
    runtime_checkpoint_metadata,
    validate_runtime_checkpoint_metadata,
)
from semantic_3d_chat.training.train_block_cross_v35 import (
    construct_v35_core,
    validate_v35_cache_audit,
)
from semantic_3d_chat.training.train_environmental_sidecar_v33 import (
    prefix_separation_ratios,
)
from semantic_3d_chat.training.train_joint_block_cross_v36 import (
    V36Contract,
    require_exact_v35_source,
    require_v35_terminal_gate,
    v36_contract,
    v36_update16_gate,
    v36_update32_gate,
    v36_update64_gate,
)
from semantic_3d_chat.training.train_joint_pair_v31 import v31_contract

DEFAULT_CONFIG = Path("configs/experiments/gemma4_diverse28_joint_block_cross_v36.yaml")
DEFAULT_CHECKPOINT_ROOT = Path("data_gemma4/checkpoints/gemma4_v36_diverse28_joint_block_cross")
DEFAULT_OUTPUT = Path("reports/gemma4/metrics/v36_joint_block_cross_selection.json")

_CORE_PREFIX = "block_cross_residual."
_CORE_PARAMETER_BASENAMES = ("w_q", "w_k", "w_v", "w_o")
_CORE_PARAMETER_NAMES = frozenset(f"{_CORE_PREFIX}{name}" for name in _CORE_PARAMETER_BASENAMES)
_CORE_BUFFER_BASENAMES = (
    "architecture_marker",
    "architecture_dimensions",
    "initialization_seed_state",
    "latent_anchors",
    "spatial_temperature",
    "uniform_floor",
    "residual_scale",
)
_CORE_BUFFER_NAMES = frozenset(f"{_CORE_PREFIX}{name}" for name in _CORE_BUFFER_BASENAMES)
_CORE_STATE_NAMES = _CORE_PARAMETER_NAMES | _CORE_BUFFER_NAMES
_BANK_NAME = "extension_v30_joint_pair_query"
_BANK_PREFIX = f"lora_banks.{_BANK_NAME}."
_BANK_PARAMETER_NAMES = tuple(
    f"{_BANK_PREFIX}adapters.{index}.{side}" for index in range(4) for side in ("lora_a", "lora_b")
)
_BANK_PARAMETER_NAME_SET = frozenset(_BANK_PARAMETER_NAMES)
_BANK_OPTIMIZER_PARAMETER_NAMES = tuple(
    f"{_BANK_PREFIX}adapters.{index}.{side}" for side in ("lora_a", "lora_b") for index in range(4)
)
_AUTHORIZED_PARAMETER_NAMES = _CORE_PARAMETER_NAMES | _BANK_PARAMETER_NAME_SET
_GREEDY_STEPS = frozenset({32, 64, 100})


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be a mapping")
    return value


def _finite(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _core_state(tensors: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    state = {
        name.removeprefix(_CORE_PREFIX): value
        for name, value in tensors.items()
        if name.startswith(_CORE_PREFIX)
    }
    if {f"{_CORE_PREFIX}{name}" for name in state} != _CORE_STATE_NAMES:
        raise ValueError("V36 checkpoint block-core state inventory changed")
    return state


def _bank_state(tensors: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    state = {
        name.removeprefix(_BANK_PREFIX): value
        for name, value in tensors.items()
        if name.startswith(_BANK_PREFIX)
    }
    if {f"{_BANK_PREFIX}{name}" for name in state} != _BANK_PARAMETER_NAME_SET:
        raise ValueError("V36 checkpoint query-LoRA state inventory changed")
    return state


def _frozen_state(tensors: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Return every tensor that V36 is not authorized to optimize.

    Persistent block-core buffers are intentionally included.  Only the four
    core matrices and eight target-bank tensors are excluded.
    """

    return {
        name: value for name, value in tensors.items() if name not in _AUTHORIZED_PARAMETER_NAMES
    }


def _dynamic_source_stack_state(
    tensors: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Return the runtime stack below the block core, including trained LoRA."""

    return {name: value for name, value in tensors.items() if not name.startswith(_CORE_PREFIX)}


def _approved_v29_runtime_tensor_envelope(
    update0: Mapping[str, torch.Tensor],
    approved_v29: Mapping[str, torch.Tensor],
    *,
    config: Mapping[str, Any],
    expected_bank_initial_state_sha256: str,
) -> dict[str, torch.Tensor]:
    """Build an inference-equivalent V29 envelope in V36's module inventory.

    V36 update zero contains V35's *learned* block core, so it cannot directly
    carry the V29 baseline.  We replace that core with the deterministic
    exact-zero initial core, retain the exact-zero query bank, and overwrite
    every shared tensor with approved V29.  No environmental text is involved.
    """

    if not set(approved_v29).issubset(update0):
        raise ValueError("Approved V29 contains tensors absent from V36 update zero")
    for name, value in approved_v29.items():
        if tuple(value.shape) != tuple(update0[name].shape):
            raise ValueError(f"Approved V29 tensor shape changed: {name}")
    extra = set(update0) - set(approved_v29)
    bank = {name for name in extra if name.startswith(_BANK_PREFIX)}
    core = {name for name in extra if name.startswith(_CORE_PREFIX)}
    if bank != _BANK_PARAMETER_NAME_SET or core != _CORE_STATE_NAMES or extra != bank | core:
        raise ValueError("V36 V29 envelope contains an unauthorized compatibility tensor")
    if tensor_state_sha256(_bank_state(update0)) != expected_bank_initial_state_sha256:
        raise ValueError("V36 V29 envelope query bank differs from exact initialization")
    if any(
        torch.count_nonzero(update0[name]).item()
        for name in _BANK_PARAMETER_NAMES
        if name.endswith(".lora_b")
    ):
        raise ValueError("V36 V29 envelope query bank is not exact-zero output")

    settings = block_cross_residual_settings(config)
    fresh_core = construct_v35_core(config, device=torch.device("cpu"))
    if fresh_core.state_sha256() != settings.expected_initial_state_sha256:
        raise ValueError("V36 V29 envelope could not reproduce the exact-zero block core")
    merged = dict(update0)
    merged.update(
        {f"{_CORE_PREFIX}{name}": value for name, value in fresh_core.state_dict().items()}
    )
    merged.update(approved_v29)
    if torch.count_nonzero(merged[f"{_CORE_PREFIX}w_o"]).item():
        raise ValueError("V36 V29 envelope block route is not exact-zero output")
    return merged


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
    checkpoint: Path,
    *,
    expected_step: int,
    tensors: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    """Independently inspect all three staged AdamW groups and moments."""

    state = torch.load(checkpoint / "optimizer.pt", map_location="cpu", weights_only=True)
    if not isinstance(state, Mapping):
        raise TypeError("V36 optimizer checkpoint must be a mapping")
    groups = state.get("param_groups")
    values = state.get("state")
    if not isinstance(groups, list) or len(groups) != 3 or not isinstance(values, Mapping):
        raise ValueError("V36 optimizer must retain exactly three AdamW groups")
    by_name = {str(group.get("name")): group for group in groups}
    expected_names = {
        "block_cross_residual.qkv",
        "block_cross_residual.output",
        f"lora_banks.{_BANK_NAME}",
    }
    if set(by_name) != expected_names:
        raise ValueError("V36 optimizer group names changed")
    if any(float(group.get("weight_decay", math.nan)) != 0.0 for group in groups):
        raise ValueError("V36 optimizer weight decay must remain zero")
    expected_lrs = {
        "block_cross_residual.qkv": 0.0 if expected_step <= 8 else 1e-4,
        "block_cross_residual.output": 0.0 if expected_step <= 8 else 2.5e-5,
        f"lora_banks.{_BANK_NAME}": 2e-5,
    }
    for name, expected in expected_lrs.items():
        if float(by_name[name].get("lr", math.nan)) != expected:
            raise ValueError(f"V36 optimizer learning rate changed for {name}")

    ordered_names = {
        "block_cross_residual.qkv": [
            f"{_CORE_PREFIX}w_q",
            f"{_CORE_PREFIX}w_k",
            f"{_CORE_PREFIX}w_v",
        ],
        "block_cross_residual.output": [f"{_CORE_PREFIX}w_o"],
        f"lora_banks.{_BANK_NAME}": list(_BANK_OPTIMIZER_PARAMETER_NAMES),
    }
    expected_state_names = set(_BANK_PARAMETER_NAMES)
    if expected_step > 8:
        expected_state_names.update(_CORE_PARAMETER_NAMES)
    observed_parameter_ids: set[object] = set()
    inspected: list[str] = []
    next_expected_id = 0
    for group_name, names in ordered_names.items():
        raw_ids = by_name[group_name].get("params")
        if not isinstance(raw_ids, list) or len(raw_ids) != len(names):
            raise ValueError(f"V36 optimizer parameter inventory changed for {group_name}")
        if by_name[group_name].get("parameter_names") != names:
            raise ValueError(f"V36 optimizer ordered names changed for {group_name}")
        expected_ids = list(range(next_expected_id, next_expected_id + len(names)))
        next_expected_id += len(names)
        if raw_ids != expected_ids:
            raise ValueError(f"V36 optimizer parameter identifiers changed for {group_name}")
        for parameter_id, tensor_name in zip(raw_ids, names, strict=True):
            if parameter_id in observed_parameter_ids:
                raise ValueError("V36 optimizer aliases a parameter across groups")
            observed_parameter_ids.add(parameter_id)
            entry = values.get(parameter_id)
            if tensor_name not in expected_state_names:
                if entry is not None:
                    raise ValueError("V36 frozen-stage core unexpectedly has Adam state")
                continue
            if not isinstance(entry, Mapping) or set(entry) != {"step", "exp_avg", "exp_avg_sq"}:
                raise ValueError(f"V36 Adam state is incomplete for {tensor_name}")
            expected_tensor_step = (
                expected_step if tensor_name in _BANK_PARAMETER_NAME_SET else expected_step - 8
            )
            if _adam_step(entry["step"], f"Adam step for {tensor_name}") != expected_tensor_step:
                raise ValueError(f"V36 Adam step changed for {tensor_name}")
            for moment_name in ("exp_avg", "exp_avg_sq"):
                moment = entry[moment_name]
                if (
                    not isinstance(moment, torch.Tensor)
                    or tuple(moment.shape) != tuple(tensors[tensor_name].shape)
                    or not torch.isfinite(moment).all()
                ):
                    raise ValueError(f"V36 Adam {moment_name} is invalid for {tensor_name}")
            inspected.append(tensor_name)
    if set(values) != {
        parameter_id
        for group_name, names in ordered_names.items()
        for parameter_id, tensor_name in zip(by_name[group_name]["params"], names, strict=True)
        if tensor_name in expected_state_names
    }:
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


def _validate_surface(
    metadata: Mapping[str, Any],
    tensors: Mapping[str, torch.Tensor],
    contract: V36Contract,
) -> dict[str, Any]:
    stage = _mapping(metadata.get("v36_joint_block_cross"), "metadata.v36_joint_block_cross")
    surface = _mapping(stage.get("trainable_surface"), "v36.trainable_surface")
    counts = _mapping(surface.get("group_parameter_counts"), "v36 group counts")
    expected_counts = {
        "block_qkv": 589_824,
        "block_output": 393_216,
        "decoder_a": 49_152,
        "decoder_b": 81_920,
    }
    if dict(counts) != expected_counts:
        raise ValueError("V36 trainable group sizes changed")
    if (
        surface.get("block_core_parameter_count") != 983_040
        or surface.get("decoder_bank_parameter_count") != 131_072
        or surface.get("total_parameter_count") != 1_114_112
    ):
        raise ValueError("V36 trainable surface parameter count changed")
    for field in (
        "gemma_base_frozen",
        "all_other_lora_banks_frozen",
        "complete_v33_scene_stack_frozen",
        "every_other_parameter_frozen",
    ):
        if surface.get(field) is not True:
            raise ValueError(f"V36 checkpoint does not prove {field}")

    core = _core_state(tensors)
    bank = _bank_state(tensors)
    frozen = _frozen_state(tensors)
    core_hash = tensor_state_sha256(core)
    bank_hash = tensor_state_sha256(bank)
    frozen_hash = tensor_state_sha256(frozen)
    dynamic_source_stack_hash = tensor_state_sha256(_dynamic_source_stack_state(tensors))
    if sum(int(tensors[name].numel()) for name in _CORE_PARAMETER_NAMES) != 983_040:
        raise ValueError("V36 block-core tensor count changed")
    if sum(int(tensors[name].numel()) for name in _BANK_PARAMETER_NAMES) != 131_072:
        raise ValueError("V36 query-LoRA tensor count changed")
    if any(not torch.isfinite(tensors[name]).all() for name in _AUTHORIZED_PARAMETER_NAMES):
        raise ValueError("V36 authorized tensor contains NaN or infinity")
    if metadata.get("block_cross_residual_state_sha256") != core_hash:
        raise ValueError("V36 block-core hash differs from metadata")
    bank_hashes = _mapping(metadata.get("lora_bank_state_sha256"), "LoRA bank hashes")
    if bank_hashes.get(_BANK_NAME) != bank_hash:
        raise ValueError("V36 query-LoRA hash differs from metadata")
    if frozen_hash != contract.frozen_nonauthorized_state_sha256:
        raise ValueError("V36 changed a non-authorized inherited tensor")
    if stage.get("frozen_nonauthorized_state_sha256") != frozen_hash:
        raise ValueError("V36 frozen-state metadata hash changed")
    if metadata.get("frozen_block_cross_source_stack_state_sha256") != (dynamic_source_stack_hash):
        raise ValueError("V36 dynamic runtime source-stack hash is stale")
    if stage.get("current_block_source_stack_state_sha256") != dynamic_source_stack_hash:
        raise ValueError("V36 training-stage source-stack hash is stale")
    return {
        "core_state_sha256": core_hash,
        "decoder_bank_state_sha256": bank_hash,
        "frozen_nonauthorized_state_sha256": frozen_hash,
        "dynamic_block_cross_source_stack_state_sha256": dynamic_source_stack_hash,
        "authorized_parameter_count": 1_114_112,
    }


def _residual_rms(row: Mapping[str, Any], label: str) -> float:
    residual = _mapping(row.get("training_residual_diagnostics"), f"{label} residual")
    return _finite(residual.get("aggregate_rms"), f"{label} residual RMS")


def _replay_train_gates(
    metadata: Mapping[str, Any],
    contract: V36Contract,
    *,
    state_hashes_by_step: Mapping[int, Mapping[str, str]],
) -> tuple[
    Mapping[str, Any] | None,
    Mapping[str, Any] | None,
    Mapping[str, Any] | None,
]:
    """Recompute every continuation decision from recorded train-only inputs."""

    history = metadata.get("history")
    if not isinstance(history, list) or not history:
        raise ValueError("V36 checkpoint history is absent")
    stage = _mapping(metadata.get("v36_joint_block_cross"), "v36 stage")
    source_broad_nll = _finite(stage.get("source_broad_train_nll"), "V36 source broad NLL")
    source_greedy = _mapping(stage.get("source_train_greedy_metrics"), "V36 source greedy metrics")
    gate16: Mapping[str, Any] | None = None
    gate32: Mapping[str, Any] | None = None
    gate64: Mapping[str, Any] | None = None
    if len(history) > 16:
        state16 = _mapping(state_hashes_by_step.get(16), "V36 update-16 tensor hashes")
        row16 = _mapping(history[16], "history[16]")
        gate16 = v36_update16_gate(
            pair_metrics=_mapping(row16.get("training_pair_metrics"), "update16 pairs"),
            broad_nll=_finite(row16.get("training_broad_nll"), "update16 broad NLL"),
            source_broad_nll=source_broad_nll,
            residual_rms=_residual_rms(row16, "update16"),
            decoder_bank_state_sha256=str(state16.get("decoder_bank_state_sha256")),
            frozen_nonauthorized_state_sha256=str(state16.get("frozen_nonauthorized_state_sha256")),
            contract=contract,
        )
        if gate16 != row16.get("update16_train_only_gate") or gate16 != stage.get(
            "update16_train_only_gate"
        ):
            raise ValueError("V36 independently replayed update-16 gate differs")
    if len(history) > 32:
        if gate16 is None:
            raise ValueError("V36 update-32 gate lacks update-16 evidence")
        row32 = _mapping(history[32], "history[32]")
        gate32 = v36_update32_gate(
            update16_gate=gate16,
            pair_metrics=_mapping(row32.get("training_pair_metrics"), "update32 pairs"),
            broad_nll=_finite(row32.get("training_broad_nll"), "update32 broad NLL"),
            source_broad_nll=source_broad_nll,
            residual_rms=_residual_rms(row32, "update32"),
            contract=contract,
        )
        if gate32 != row32.get("update32_train_only_gate") or gate32 != stage.get(
            "update32_train_only_gate"
        ):
            raise ValueError("V36 independently replayed update-32 gate differs")
    if len(history) > 64:
        if gate32 is None:
            raise ValueError("V36 update-64 gate lacks update-32 evidence")
        row64 = _mapping(history[64], "history[64]")
        gate64 = v36_update64_gate(
            update32_gate=gate32,
            pair_metrics=_mapping(row64.get("training_pair_metrics"), "update64 pairs"),
            greedy_metrics=_mapping(row64.get("training_greedy_metrics"), "update64 greedy"),
            source_greedy_metrics=source_greedy,
            residual_rms=_residual_rms(row64, "update64"),
            contract=contract,
        )
        if gate64 != row64.get("update64_train_only_gate") or gate64 != stage.get(
            "update64_train_only_gate"
        ):
            raise ValueError("V36 independently replayed update-64 gate differs")
    return gate16, gate32, gate64


def _validate_update_zero_envelope(
    *,
    tensors: Mapping[str, torch.Tensor],
    source_tensors: Mapping[str, torch.Tensor],
    surface: Mapping[str, Any],
    stage: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
    contract: V36Contract,
) -> None:
    if set(tensors) != set(source_tensors) or any(
        not torch.equal(tensors[name], source_tensors[name]) for name in tensors
    ):
        raise ValueError("V36 update zero is not tensor-bit-exact V35 update 32")
    if surface["core_state_sha256"] != contract.core_source_state_sha256:
        raise ValueError("V36 update-zero block core differs from V35 update 32")
    if surface["decoder_bank_state_sha256"] != contract.decoder_bank_initial_state_sha256:
        raise ValueError("V36 update-zero query bank differs from exact-zero source")
    source_replay = _mapping(stage.get("source_replay_attestation"), "update-zero source replay")
    stage_surfaces = _mapping(stage.get("update_zero_equivalence"), "stage update-zero surfaces")
    history_surfaces = _mapping(
        _mapping(history[0], "history[0]").get("update_zero_surfaces"),
        "history update-zero surfaces",
    )
    if dict(stage_surfaces) != dict(history_surfaces):
        raise ValueError("V36 update-zero surface proof differs between metadata locations")
    required_source_replay = {
        "exact_stopped_v35_update32_loaded": True,
        "v35_optimizer_state_loaded": False,
        "fresh_adam_state": True,
        "validation_qa_loaded": False,
    }
    required_surfaces = {
        "exact_stopped_v35_update32_loaded": True,
        "fresh_v35_optimizer_state_loaded": False,
        "decoder_bank_exact_zero_output": True,
        "learned_block_core_active": True,
        "joint_update_zero_equivalent_to_v35_update32": True,
    }
    if any(source_replay.get(key) != value for key, value in required_source_replay.items()) or any(
        stage_surfaces.get(key) != value for key, value in required_surfaces.items()
    ):
        raise ValueError("V36 update-zero equivalence proof is incomplete")


def _validate_source_baseline_provenance(
    *,
    stage: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
    source_metadata: Mapping[str, Any],
) -> None:
    """Bind every V36 source baseline to update zero and exact V35 evidence."""

    if not history:
        raise ValueError("V36 source-baseline proof lacks update-zero history")
    update0 = _mapping(history[0], "V36 history[0]")
    source_history = source_metadata.get("history")
    if not isinstance(source_history, list) or len(source_history) != 33:
        raise ValueError("Exact V35 source lacks its complete update-32 history")
    source_row = _mapping(source_history[-1], "V35 source history[-1]")

    stage_pair = _mapping(stage.get("source_pair_metrics"), "V36 source pair metrics")
    update0_pair = _mapping(
        update0.get("source_pair_metrics"), "V36 update-zero source pair metrics"
    )
    exact_pair = _mapping(source_row.get("training_pair_metrics"), "exact V35 source pair metrics")
    if dict(stage_pair) != dict(update0_pair) or dict(stage_pair) != dict(exact_pair):
        raise ValueError("V36 source pair baseline is not exact V35 update-32 evidence")

    stage_broad = _finite(stage.get("source_broad_train_nll"), "V36 source broad NLL")
    update0_broad = _finite(
        update0.get("source_broad_train_nll"), "V36 update-zero source broad NLL"
    )
    if stage_broad != update0_broad:
        raise ValueError("V36 source broad-NLL baseline differs from update zero")
    stage_greedy = _mapping(stage.get("source_train_greedy_metrics"), "V36 source greedy metrics")
    update0_greedy = _mapping(
        update0.get("source_train_greedy_metrics"),
        "V36 update-zero source greedy metrics",
    )
    if dict(stage_greedy) != dict(update0_greedy):
        raise ValueError("V36 source greedy baseline differs from update zero")

    source_replay = _mapping(
        stage.get("source_replay_attestation"), "V36 source replay attestation"
    )
    family = _mapping(
        stage_pair.get("complete_units_by_family"),
        "V36 source complete units by family",
    )
    expected_replay = {
        "source_complete_units": int(stage_pair["complete_units"]),
        "source_cross_prefix_complete_units": int(stage_pair["cross_prefix_complete_units"]),
        "source_positive_sides": int(stage_pair["positive_sides"]),
        "source_mean_cross_prefix_margin": float(stage_pair["mean_cross_prefix_margin"]),
        "source_complete_units_by_family": dict(family),
    }
    if any(source_replay.get(key) != value for key, value in expected_replay.items()):
        raise ValueError("V36 source replay numerics differ from its exact pair baseline")

    update0_residual = _mapping(
        update0.get("training_residual_diagnostics"),
        "V36 update-zero residual diagnostics",
    )
    exact_residual = _mapping(
        source_row.get("training_residual_diagnostics"),
        "exact V35 source residual diagnostics",
    )
    if dict(update0_residual) != dict(exact_residual):
        raise ValueError("V36 source residual baseline is not exact V35 update-32 evidence")
    replay_rms = _finite(source_replay.get("source_residual_rms"), "V36 source replay residual RMS")
    residual_rms = _finite(update0_residual.get("aggregate_rms"), "V36 update-zero residual RMS")
    if replay_rms != residual_rms:
        raise ValueError("V36 source replay residual differs from exact update zero")


def validate_v36_checkpoint_envelope(
    config: Mapping[str, Any],
    checkpoint_root: Path,
    contract: V36Contract,
) -> tuple[tuple[Path, ...], list[dict[str, Any]]]:
    """Validate all 14 arms before any validation QA or Gemma load is legal."""

    if checkpoint_root.is_symlink() or not checkpoint_root.is_dir():
        raise ValueError("V36 checkpoint root must be a real directory")
    checkpoints = tuple(
        checkpoint_root / f"update_{step:03d}" for step in contract.saved_optimizer_steps
    )
    observed = sorted(path.name for path in checkpoint_root.glob("update_*"))
    expected = [path.name for path in checkpoints]
    if observed != expected or contract.saved_optimizer_steps != (*range(0, 97, 8), 100):
        raise FileNotFoundError(
            f"V36 requires the complete saved-arm envelope: observed={observed} expected={expected}"
        )

    terminal = require_v35_terminal_gate(config)
    source, source_metadata = require_exact_v35_source(config)
    source_tensors = load_file(source / "adapter.safetensors", device="cpu")
    expected_config_hash = config_hash(dict(config))
    prior_history: list[Mapping[str, Any]] = []
    common_schedule: Mapping[str, Any] | None = None
    common_cache: Mapping[str, Any] | None = None
    common_source_metrics: Mapping[str, Any] | None = None
    update0_tensors: Mapping[str, torch.Tensor] | None = None
    accepted16: Mapping[str, Any] | None = None
    accepted32: Mapping[str, Any] | None = None
    accepted64: Mapping[str, Any] | None = None
    state_hashes_by_step: dict[int, Mapping[str, str]] = {}
    audits: list[dict[str, Any]] = []

    for step, checkpoint in zip(contract.saved_optimizer_steps, checkpoints, strict=True):
        if checkpoint.is_symlink() or not checkpoint.is_dir():
            raise ValueError(f"V36 arm must be a real directory: {checkpoint}")
        required = [
            "adapter.safetensors",
            TRAINING_METADATA_FILENAME,
            RUNTIME_METADATA_FILENAME,
        ]
        if step:
            required.append("optimizer.pt")
        if any(
            not (checkpoint / name).is_file() or (checkpoint / name).is_symlink()
            for name in required
        ):
            raise FileNotFoundError(f"V36 arm is incomplete or aliased: {checkpoint.name}")
        if step == 0 and (checkpoint / "optimizer.pt").exists():
            raise ValueError("V36 update zero must not carry inherited Adam state")

        metadata = _metadata(checkpoint)
        stage = _mapping(metadata.get("v36_joint_block_cross"), "metadata.v36_joint_block_cross")
        if metadata.get("optimizer_step") != step or stage.get("optimizer_step") != step:
            raise ValueError(f"V36 optimizer-step mismatch: {checkpoint.name}")
        if metadata.get("config_hash") != expected_config_hash:
            raise ValueError(f"V36 config hash changed: {checkpoint.name}")
        terminal_pin = {
            "path": terminal["path"],
            "sha256": terminal["sha256"],
        }
        if stage.get("conditional_v35_terminal_gate") != terminal_pin:
            raise ValueError(f"V36 terminal authorization changed: {checkpoint.name}")
        if (
            Path(str(stage.get("source_checkpoint"))).resolve() != source
            or stage.get("source_file_sha256") != dict(contract.source_file_sha256)
            or stage.get("source_v35_tensor_state_sha256") != contract.source_tensor_state_sha256
            or stage.get("inherited_v33_tensor_state_sha256")
            != contract.inherited_v33_tensor_state_sha256
            or stage.get("source_block_core_state_sha256") != contract.core_source_state_sha256
            or stage.get("decoder_bank_initial_state_sha256")
            != contract.decoder_bank_initial_state_sha256
        ):
            raise ValueError(f"V36 source provenance changed: {checkpoint.name}")
        forbidden = {
            "validation_qa_loaded": False,
            "oracle_environment_files_loaded": False,
            "question_dependent_scene_processing": False,
            "question_dependent_retrieval": False,
            "source_v35_optimizer_state_loaded": False,
        }
        if any(stage.get(key) != value for key, value in forbidden.items()):
            raise ValueError(f"V36 training crossed its data/source boundary: {checkpoint.name}")
        if "v35_optimizer_state_loaded" in stage:
            raise ValueError("V36 metadata contains a deprecated optimizer-source alias")
        if not (
            stage.get("fresh_adam") is True
            and stage.get("optimizer_stage_updates_1_through_8") == "lora_only"
            and stage.get("optimizer_stage_updates_9_through_100") == "joint_core_and_lora"
        ):
            raise ValueError(f"V36 fresh optimizer staging changed: {checkpoint.name}")
        if stage.get("deferred_final_scene_ids_loaded") != []:
            raise ValueError(f"V36 training touched deferred final scenes: {checkpoint.name}")

        qa = _mapping(stage.get("train_qa_dataset"), "V36 train QA audit")
        loaded_qa = qa.get("loaded_files")
        if (
            qa.get("validation_qa_loaded") is not False
            or qa.get("deferred_final_qa_loaded") is not False
            or not isinstance(loaded_qa, list)
            or any(Path(str(path)).name == "validation.jsonl" for path in loaded_qa)
        ):
            raise ValueError(f"V36 training loaded validation/final QA: {checkpoint.name}")
        cache = _mapping(stage.get("scene_cache"), "V36 scene cache")
        split = v31_contract(config)
        validate_v35_cache_audit(
            cache,
            expected_scene_ids=(*split.train_scene_ids, *split.validation_scene_ids),
        )
        schedule = _mapping(stage.get("schedule"), "V36 schedule")
        if not (
            schedule.get("optimizer_step_count") == 100
            and schedule.get("pair_unit_count") == 25
            and schedule.get("exact_pair_unit_recurrence") == 4
            and schedule.get("pair_units_atomic") is True
            and schedule.get("true_optimizer_step_per_schedule_row") is True
            and schedule.get("questions_or_answers_serialized_to_runtime") is False
        ):
            raise ValueError(f"V36 schedule proof failed: {checkpoint.name}")
        source_metrics = {
            "source_broad_train_nll": stage.get("source_broad_train_nll"),
            "source_train_greedy_metrics": stage.get("source_train_greedy_metrics"),
            "source_pair_metrics": stage.get("source_pair_metrics"),
        }
        if common_schedule is None:
            common_schedule = dict(schedule)
            common_cache = dict(cache)
            common_source_metrics = source_metrics
        elif (
            schedule != common_schedule
            or cache != common_cache
            or source_metrics != common_source_metrics
        ):
            raise ValueError(f"V36 schedule/cache/source metrics changed: {checkpoint.name}")

        history = metadata.get("history")
        if not isinstance(history, list) or len(history) != step + 1:
            raise ValueError(f"V36 history is incomplete: {checkpoint.name}")
        if [row.get("optimizer_update") for row in history] != list(range(step + 1)):
            raise ValueError(f"V36 history is not one row per true update: {checkpoint.name}")
        _validate_source_baseline_provenance(
            stage=stage,
            history=history,
            source_metadata=source_metadata,
        )
        if any(
            row.get("validation_qa_loaded") is not False
            or row.get("oracle_environment_files_loaded") is not False
            for row in history
        ):
            raise ValueError(f"V36 history crossed the training-only boundary: {checkpoint.name}")
        if prior_history and history[: len(prior_history)] != prior_history:
            raise ValueError(f"V36 history was rewritten across arms: {checkpoint.name}")
        prior_history = list(history)
        row = _mapping(history[-1], "history[-1]")
        if row.get("saved_checkpoint") is not True or row.get("validation_qa_loaded") is not False:
            raise ValueError(f"V36 saved-row audit failed: {checkpoint.name}")
        if step and (
            row.get("true_optimizer_step") is not True
            or row.get("separate_group_clipping") is not True
            or row.get("training_residual_diagnostics") is None
            or row.get("training_prefix_separation_descriptive_only") is None
            or not isinstance(row.get("preclip_gradient_norm_by_group"), Mapping)
        ):
            raise ValueError(f"V36 saved diagnostics are incomplete: {checkpoint.name}")
        if step in {8, 16, 32, 64, 100} and row.get("training_pair_metrics") is None:
            raise ValueError(f"V36 designated arm lacks train pair metrics: {checkpoint.name}")
        if step in {16, 32, 64} and row.get("training_broad_nll") is None:
            raise ValueError(f"V36 gate arm lacks broad train NLL: {checkpoint.name}")
        if step == 64 and row.get("training_greedy_metrics") is None:
            raise ValueError("V36 update-64 gate lacks train-only greedy evidence")

        runtime = json.loads((checkpoint / RUNTIME_METADATA_FILENAME).read_text(encoding="utf-8"))
        validate_runtime_checkpoint_metadata(runtime)
        if runtime != runtime_checkpoint_metadata(metadata):
            raise ValueError(f"V36 runtime metadata was not freshly sanitized: {checkpoint.name}")
        tensors = load_file(checkpoint / "adapter.safetensors", device="cpu")
        surface = _validate_surface(metadata, tensors, contract)
        expected_active_stage = "lora_only" if step <= 8 else "joint_full"
        if (
            _mapping(stage.get("trainable_surface"), "V36 surface").get("active_stage")
            != expected_active_stage
        ):
            raise ValueError(f"V36 active optimizer stage changed: {checkpoint.name}")
        state_hashes_by_step[step] = {
            "decoder_bank_state_sha256": str(surface["decoder_bank_state_sha256"]),
            "frozen_nonauthorized_state_sha256": str(surface["frozen_nonauthorized_state_sha256"]),
        }
        if update0_tensors is None:
            update0_tensors = tensors
            _validate_update_zero_envelope(
                tensors=tensors,
                source_tensors=source_tensors,
                surface=surface,
                stage=stage,
                history=history,
                contract=contract,
            )
        else:
            if set(tensors) != set(update0_tensors):
                raise ValueError(f"V36 tensor inventory changed: {checkpoint.name}")
            if any(
                tuple(tensors[name].shape) != tuple(update0_tensors[name].shape)
                or tensors[name].dtype != update0_tensors[name].dtype
                for name in tensors
            ):
                raise ValueError(f"V36 tensor shape/dtype changed: {checkpoint.name}")
            changed = {
                name for name in tensors if not torch.equal(tensors[name], update0_tensors[name])
            }
            if not changed or not changed.issubset(_AUTHORIZED_PARAMETER_NAMES):
                raise ValueError(f"V36 arm changed an unauthorized tensor: {checkpoint.name}")
            if not changed.intersection(_BANK_PARAMETER_NAME_SET):
                raise ValueError(f"V36 arm did not train its query bank: {checkpoint.name}")
            if step <= 8 and changed.intersection(_CORE_PARAMETER_NAMES):
                raise ValueError("V36 changed its block core during the LoRA-only stage")
            if step > 8 and not changed.intersection(_CORE_PARAMETER_NAMES):
                raise ValueError("V36 joint stage did not change its block core")
            if any(
                not torch.equal(tensors[name], update0_tensors[name]) for name in _CORE_BUFFER_NAMES
            ):
                raise ValueError("V36 changed a persistent block-core buffer")

        optimizer_audit = None
        if step:
            optimizer_audit = _optimizer_step_audit(checkpoint, expected_step=step, tensors=tensors)
        replay16, replay32, replay64 = _replay_train_gates(
            metadata,
            contract,
            state_hashes_by_step=state_hashes_by_step,
        )
        if step >= 16:
            if replay16 is None or replay16.get("passed") is not True:
                raise ValueError(f"V36 arm lacks a passed update-16 train gate: {checkpoint.name}")
            if accepted16 is None:
                accepted16 = replay16
            elif replay16 != accepted16:
                raise ValueError("V36 update-16 gate changed across later arms")
        if step >= 32:
            if replay32 is None or replay32.get("passed") is not True:
                raise ValueError(f"V36 arm lacks a passed update-32 train gate: {checkpoint.name}")
            if accepted32 is None:
                accepted32 = replay32
            elif replay32 != accepted32:
                raise ValueError("V36 update-32 gate changed across later arms")
        if step >= 64:
            if replay64 is None or replay64.get("passed") is not True:
                raise ValueError(f"V36 arm lacks a passed update-64 train gate: {checkpoint.name}")
            if accepted64 is None:
                accepted64 = replay64
            elif replay64 != accepted64:
                raise ValueError("V36 update-64 gate changed across later arms")
        audits.append(
            {
                "checkpoint": str(checkpoint),
                "optimizer_step": step,
                "tensor_and_buffer_inventory_inspected": True,
                "runtime_metadata_inspected": True,
                "optimizer_state": optimizer_audit,
                **surface,
            }
        )

    if accepted16 is None or accepted32 is None or accepted64 is None:
        raise ValueError("V36 complete run lacks all three accepted train-only gates")
    return checkpoints, audits


class V36ArmEvaluator(Protocol):
    validation_scene_ids: tuple[str, ...]
    cache_audit: Mapping[str, Any]

    def install(
        self, tensors: Mapping[str, torch.Tensor], *, approved_v29: bool = False
    ) -> None: ...

    def evaluate_teacher(self) -> V35TeacherEvidence: ...

    def evaluate_greedy(self) -> V35GreedyEvidence: ...

    def evaluate_aggregate_exact(self) -> tuple[int, int]: ...

    def attest_prefix_invariance(self) -> Mapping[str, Any]: ...


class _V36RuntimeEvaluator(_V35RuntimeEvaluator):
    """One-Gemma V36 evaluator using the complete question-free scene cache."""

    def evaluate_teacher(self) -> V35TeacherEvidence:
        evidence = super().evaluate_teacher()
        prefix = dict(evidence.prefix_diagnostics)
        prefix["tensor"] = "composed_v36_joint_continuous_scene_prefix"
        return V35TeacherEvidence(
            validation_answer_token_nll=evidence.validation_answer_token_nll,
            pair_margins=evidence.pair_margins,
            family_teacher=evidence.family_teacher,
            prefix_diagnostics=prefix,
            color_full_vocab_sides=evidence.color_full_vocab_sides,
            mirror_full_vocab_sides=evidence.mirror_full_vocab_sides,
            negative_sides=evidence.negative_sides,
            prefix_sha256_by_scene=evidence.prefix_sha256_by_scene,
        )


def _development_checks_v36(
    *,
    teacher: V35TeacherEvidence,
    greedy: V35GreedyEvidence,
    source: V35TeacherEvidence,
    approved: V35TeacherEvidence,
    approved_greedy: V35GreedyEvidence,
) -> tuple[dict[str, bool], dict[str, float], int, list[tuple[str, str]]]:
    ratios = prefix_separation_ratios(teacher.prefix_diagnostics, source.prefix_diagnostics)
    nonmirror = sum(
        int(_mapping(teacher.family_teacher[family], family)["complete_units"])
        for family in ("book_support", "picture_support")
    )
    new_negatives = sorted(teacher.negative_sides - approved.negative_sides)
    checks = {
        "validation_answer_token_nll_improved_from_v35_u32": (
            teacher.validation_answer_token_nll < source.validation_answer_token_nll
        ),
        "validation_pair_mean_margin_improved_from_v35_u32": (
            teacher.pair_margins.mean_margin > source.pair_margins.mean_margin
        ),
        "validation_pair_passed_units_not_below_v35_u32": (
            teacher.pair_margins.passed_units >= source.pair_margins.passed_units
        ),
        "nonmirror_teacher_complete": nonmirror >= 1,
        "greedy_development_unit_demonstrated": (
            greedy.generation.exact_complete_units_correct >= 1
        ),
        "approved_v29_color_12_sides_retained": teacher.color_full_vocab_sides >= 12,
        "approved_v29_mirror_10_sides_retained": teacher.mirror_full_vocab_sides >= 10,
        "approved_v29_controls_no_new_negatives": not new_negatives,
        "broad_retention_vs_approved_v29": (
            greedy.generation.broad_exact_accuracy
            >= approved_greedy.generation.broad_exact_accuracy
        ),
    }
    return checks, ratios, nonmirror, new_negatives


def select_v36(
    config_path: Path,
    checkpoint_root: Path,
    *,
    evaluator_factory: Callable[
        [dict[str, Any], dict[str, Any], Path, SelectionRequirements], V36ArmEvaluator
    ] = _V36RuntimeEvaluator,
) -> dict[str, Any]:
    config = load_config(config_path)
    contract = v36_contract(config)
    split = v31_contract(config)
    checkpoints, envelope_audits = validate_v36_checkpoint_envelope(
        config, checkpoint_root, contract
    )

    # No code above this line may construct Gemma or open validation QA.  The
    # complete envelope and all train-only gates have now been independently
    # accepted, so a single local evaluator may cross the development boundary.
    requirements = _selection_requirements(config)
    control = _retention_control_config(config)
    source_v29 = _source_v29_evidence(_metadata(checkpoints[0]))
    _validate_source_against_config(source_v29, config)
    evaluator = evaluator_factory(config, control, checkpoints[0], requirements)
    if tuple(evaluator.validation_scene_ids) != tuple(
        f"scene_{index:06d}" for index in range(19, 25)
    ):
        raise ValueError("V36 evaluator must remain exactly on scenes 19--24")

    update0 = load_file(checkpoints[0] / "adapter.safetensors", device="cpu")
    approved_tensors = load_file(
        Path(str(source_v29["checkpoint"])) / "adapter.safetensors", device="cpu"
    )
    evaluator.install(
        _approved_v29_runtime_tensor_envelope(
            update0,
            approved_tensors,
            config=config,
            expected_bank_initial_state_sha256=contract.decoder_bank_initial_state_sha256,
        ),
        approved_v29=True,
    )
    approved_teacher = evaluator.evaluate_teacher()
    approved_greedy = evaluator.evaluate_greedy()
    approved_aggregate = evaluator.evaluate_aggregate_exact()

    # V36 update zero is exact stopped V35 update 32 and is the causal
    # improvement baseline.  It is evaluated after the V29 retention baseline
    # with the same already-loaded Gemma instance.
    evaluator.install(update0)
    source_teacher = evaluator.evaluate_teacher()
    arms: list[dict[str, Any]] = []
    for step, checkpoint in zip(contract.saved_optimizer_steps, checkpoints, strict=True):
        tensors = load_file(checkpoint / "adapter.safetensors", device="cpu")
        evaluator.install(tensors)
        teacher = evaluator.evaluate_teacher()
        greedy: V35GreedyEvidence | None = None
        if step in _GREEDY_STEPS:
            greedy = evaluator.evaluate_greedy()
        ratios = prefix_separation_ratios(
            teacher.prefix_diagnostics, source_teacher.prefix_diagnostics
        )
        arm: dict[str, Any] = {
            "checkpoint": str(checkpoint),
            "optimizer_step": step,
            "update": step,
            "validation_answer_token_nll": teacher.validation_answer_token_nll,
            "validation_pair_passed_units": teacher.pair_margins.passed_units,
            "validation_pair_mean_margin": teacher.pair_margins.mean_margin,
            "validation_pair_minimum_margin": teacher.pair_margins.minimum_margin,
            "validation_family_teacher": dict(teacher.family_teacher),
            "validation_prefix_separation": dict(teacher.prefix_diagnostics),
            "validation_prefix_ratios_from_v35_u32": ratios,
            "color_full_vocab_sides": teacher.color_full_vocab_sides,
            "mirror_full_vocab_sides": teacher.mirror_full_vocab_sides,
            "new_negative_sides_vs_approved_v29": sorted(
                teacher.negative_sides - approved_teacher.negative_sides
            ),
            "prefix_sha256_by_validation_scene": dict(
                sorted(teacher.prefix_sha256_by_scene.items())
            ),
            "greedy_screen_designated": step in _GREEDY_STEPS,
            "greedy_exact_complete_units_correct": None,
            "greedy_prediction_changed_units": None,
            "greedy_complete_units_by_family": None,
            "greedy_prediction_changed_by_family": None,
            "broad_retention_exact_accuracy": None,
            "checks": {},
            "eligible": False,
        }
        if greedy is not None:
            checks, ratios, nonmirror, new_negatives = _development_checks_v36(
                teacher=teacher,
                greedy=greedy,
                source=source_teacher,
                approved=approved_teacher,
                approved_greedy=approved_greedy,
            )
            arm.update(
                {
                    "nonmirror_teacher_complete_units": nonmirror,
                    "new_negative_sides_vs_approved_v29": new_negatives,
                    "greedy_exact_complete_units_correct": (
                        greedy.generation.exact_complete_units_correct
                    ),
                    "greedy_prediction_changed_units": (greedy.generation.prediction_changed_units),
                    "greedy_complete_units_by_family": dict(greedy.complete_by_family),
                    "greedy_prediction_changed_by_family": dict(
                        greedy.prediction_changed_by_family
                    ),
                    "broad_retention_exact_accuracy": (greedy.generation.broad_exact_accuracy),
                    "checks": checks,
                    "eligible": all(checks.values()),
                }
            )
        arms.append(arm)

    candidates = [arm for arm in arms if arm["eligible"]]
    selected = min(
        candidates,
        key=lambda arm: (
            -int(arm["greedy_exact_complete_units_correct"]),
            -sum(
                int(value) > 0
                for value in _mapping(
                    arm["greedy_complete_units_by_family"], "greedy families"
                ).values()
            ),
            -int(arm["nonmirror_teacher_complete_units"]),
            -int(arm["validation_pair_passed_units"]),
            float(arm["validation_answer_token_nll"]),
            int(arm["optimizer_step"]),
        ),
        default=None,
    )
    selected_aggregate: tuple[int, int] | None = None
    prefix_attestation: Mapping[str, Any] | None = None
    if selected is not None:
        selected_path = checkpoint_root / f"update_{int(selected['optimizer_step']):03d}"
        evaluator.install(load_file(selected_path / "adapter.safetensors", device="cpu"))
        initial_attestation = evaluator.attest_prefix_invariance()
        if initial_attestation.get("passed") is not True:
            raise ValueError("V36 selected prefix failed its pre-question invariance replay")
        selected_aggregate = evaluator.evaluate_aggregate_exact()
        prefix_attestation = evaluator.attest_prefix_invariance()

    promotion = _promotion(
        selected,
        approved_v29_aggregate=approved_aggregate,
        selected_aggregate=selected_aggregate,
        prefix_attestation=prefix_attestation,
    )
    terminal = require_v35_terminal_gate(config)
    return {
        "schema_version": 1,
        "artifact": "v36_joint_block_cross_development_selection",
        "development_validation_model_selection_only": True,
        "training_completed_before_validation_loaded": True,
        "validation_used_for_training_continuation": False,
        "final_test_scenes_touched": False,
        "deferred_final_scene_ids": list(split.deferred_final_scene_ids),
        "oracle_loaded": False,
        "question_text_serialized": False,
        "answer_text_serialized": False,
        "question_dependent_scene_processing": False,
        "question_dependent_retrieval": False,
        "all_scene_prefixes_built_before_questions": True,
        "gemma_base_frozen": True,
        "only_existing_query_lora_and_block_core_trained": True,
        "complete_v33_scene_stack_frozen": True,
        "exact_trainable_parameter_count": 1_114_112,
        "model_load_count": 1,
        "source_v29": source_v29,
        "retention_and_aggregate_baseline": "approved_v29",
        "improvement_baseline": "exact_stopped_v35_update_032",
        "v35_terminal_gate": {"path": terminal["path"], "sha256": terminal["sha256"]},
        "train_scene_ids": list(split.train_scene_ids),
        "validation_scene_ids": list(evaluator.validation_scene_ids),
        "saved_optimizer_steps": list(contract.saved_optimizer_steps),
        "all_saved_arms_inspected": True,
        "checkpoint_envelope_audits": envelope_audits,
        "optimizer_state_steps_verified": list(contract.saved_optimizer_steps[1:]),
        "train_only_update16_gate_passed": True,
        "train_only_update32_gate_passed": True,
        "train_only_update64_gate_passed": True,
        "complete_question_independent_block_cache": dict(evaluator.cache_audit),
        "teacher_scored_steps": list(contract.saved_optimizer_steps),
        "greedy_screen_steps": sorted(_GREEDY_STEPS),
        "development_requirements": {
            "validation_nll_strictly_improves_v35_u32": True,
            "validation_pair_mean_strictly_improves_v35_u32": True,
            "validation_pair_passed_units_no_regression_from_v35_u32": True,
            "nonmirror_teacher_complete_minimum": 1,
            "approved_v29_color_sides_minimum": 12,
            "approved_v29_mirror_sides_minimum": 10,
            "approved_v29_no_new_control_negatives": True,
            "approved_v29_broad_accuracy_no_regression": True,
        },
        "approved_v29_teacher_baseline": {
            "validation_answer_token_nll": approved_teacher.validation_answer_token_nll,
            "color_full_vocab_sides": approved_teacher.color_full_vocab_sides,
            "mirror_full_vocab_sides": approved_teacher.mirror_full_vocab_sides,
            "broad_retention_exact_accuracy": approved_greedy.generation.broad_exact_accuracy,
        },
        "v35_u32_teacher_baseline": {
            "validation_answer_token_nll": source_teacher.validation_answer_token_nll,
            "validation_pair_passed_units": source_teacher.pair_margins.passed_units,
            "validation_pair_mean_margin": source_teacher.pair_margins.mean_margin,
            "validation_prefix_separation": dict(source_teacher.prefix_diagnostics),
        },
        "arms": arms,
        "selected_checkpoint": None if selected is None else selected["checkpoint"],
        "selected_update": None if selected is None else selected["optimizer_step"],
        "selected_optimizer_step": None if selected is None else selected["optimizer_step"],
        "development_selection_passed": selected is not None,
        "chat_promotion": promotion,
        "chat_promotion_eligible": promotion["eligible"],
        "development_progress_is_not_chat_promotion": True,
        "passed": selected is not None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = select_v36(args.config, args.checkpoint_root)
    _atomic_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "V36ArmEvaluator",
    "select_v36",
    "validate_v36_checkpoint_envelope",
]
