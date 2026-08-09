"""Strict no-live-step structural preflight for the V20 signed-X local field.

This supervised, offline diagnostic reconstructs the exact first V20 epoch
from the pinned V18 epoch-four checkpoint.  It accumulates the real twelve
microstep gradient on a fresh signed-X output projection, but it never creates
or steps an optimizer over that live module.  The first AdamW update is instead
executed on an isolated clone so a later stage-one verifier can compare exact
parameter and optimizer-state hashes.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from semantic_3d_chat.config import config_hash
from semantic_3d_chat.evaluation.v18_structural_preflight import (
    capture_rng_states,
    fp64_delta_metrics,
    fp64_pair_delta_metrics,
    restore_rng_states,
    rng_state_evidence,
)
from semantic_3d_chat.evaluation.v19_optimizer_state import (
    V19_SIGNED_X_OPTIMIZER_GROUP_NAME,
    canonical_v19_adamw_state,
)
from semantic_3d_chat.language.lora import tensor_state_sha256
from semantic_3d_chat.scene_encoder.global_residual import (
    ZERO_SPATIAL_MEAN_CONTENT_GATE_V1,
)
from semantic_3d_chat.scene_encoder.signed_x_local_field import SIGNED_X_LOCAL_FIELD_V2

V20_PREFLIGHT_ROLE = "v20_exact_ordered_signed_x_local_field_structural_preflight"
V20_SCREEN_ROLE = "signed_x_local_field_architecture_screen"
V20_EXPERIMENT_ROLE = "exploratory_reflection_odd_local_field_screen_v20"
SIGNED_X_OPTIMIZER_GROUP_NAME = V19_SIGNED_X_OPTIMIZER_GROUP_NAME
COLOR_PAIR_ID = "pair_000001"
MIRROR_PAIR_ID = "pair_000003"

# The V20 curriculum intentionally inherits the already-audited V18 selection
# and epoch-one ordering.  These constants fail closed if either the QA source
# or any sampling/scheduling implementation changes.
EXPECTED_SELECTION_SHA256 = "7f0714e3151c9ddb57c1da95a457820a833e490c070881a88a9fee4a9168f933"
EXPECTED_ORDERED_UNIT_SHA256 = "1d77157b18636abc6a5dd4a2d63bc62861d7c8147832105d40b87f1470fa3359"
EXPECTED_PAIR_MEMBERSHIP_SHA256 = "99ee448c23fb71b7269a353a54b2156ac55701847af170597dcc351af15cbcbe"
EXPECTED_PAIR_UNIT_SELECTION_SHA256 = (
    "d5928cb783339ef62fff5c14a8c7f85f90d3a7a6cb8edad0a784998082740d3e"
)
EXPECTED_SCENE_IDS = (
    "scene_000003",
    "scene_000004",
    "scene_000007",
    "scene_000008",
)
EXPECTED_SIGNED_X_INITIAL_STATE_SHA256 = (
    "3f249307901df75ba07a758a7dc5b02c7c6ff9bbb969987741a106b8d8977ce1"
)
EXPECTED_SOURCE_GLOBAL_RESIDUAL_SHA256 = (
    "ce3bf864eed6dd4a50f1b67296981e144d6a79e9cf192ad9a9230f2ae18208dc"
)
EXPECTED_SOURCE_SCENE_STATE_SHA256 = (
    "690bd890bfda024dbb5c7d3c68087b8113bc3b8ee81dd6143c7eb2a884e7245b"
)
EXPECTED_SOURCE_LORA_SHA256 = {
    "extension_v13": "4eb90fb9b0bea579d14cfcb0f61ebd5b6d566fd600bd3d5e1bfe5177a39e1b34",
    "inherited_v12": "dec768bed654c8c4e16da0318857543ad54d8f5f68f4d24a9a87cd19ec706594",
}
EXPECTED_SOURCE_ADAPTER_SHA256 = "1a7946d2e40aaf4bf66dc570bff19fa8d6ba4425e4e0d59bd52b809bd23dae7a"
EXPECTED_SOURCE_METADATA_SHA256 = "4853355ef4810f284d9b36eca1f0f1ade71319f4f6f579a5b079ce6178eb2344"
EXPECTED_RESOLVED_CONFIG_HASH = "a40636303078"


class V20StructuralPreflightViolation(ValueError):
    """A fail-closed V20 configuration or evidence violation."""


def _fail(message: str) -> None:
    raise V20StructuralPreflightViolation(message)


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{field} must be an object")
    return value


def _sequence(value: Any, field: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        _fail(f"{field} must be a sequence")
    return value


def canonical_sha256(value: Any) -> str:
    """Hash JSON-compatible evidence with one canonical representation."""

    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: str | Path, value: Mapping[str, Any]) -> Path:
    """Atomically persist finite, sorted evidence with an fsync boundary."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return destination


def _validate_optimizer_contract(raw: Mapping[str, Any]) -> dict[str, Any]:
    expected_keys = {
        "name",
        "learning_rate",
        "betas",
        "epsilon",
        "weight_decay",
        "foreach",
        "fused",
        "capturable",
        "maximize",
        "amsgrad",
        "gradient_clip_norm",
        "accumulation_divisor",
        "step_index",
    }
    if set(raw) != expected_keys:
        _fail(
            "training.optimizer keys mismatch: "
            f"missing={sorted(expected_keys - set(raw))} "
            f"unknown={sorted(set(raw) - expected_keys)}"
        )
    normalized = {
        "name": raw["name"],
        "learning_rate": float(raw["learning_rate"]),
        "betas": [float(value) for value in _sequence(raw["betas"], "optimizer.betas")],
        "epsilon": float(raw["epsilon"]),
        "weight_decay": float(raw["weight_decay"]),
        "foreach": raw["foreach"],
        "fused": raw["fused"],
        "capturable": raw["capturable"],
        "maximize": raw["maximize"],
        "amsgrad": raw["amsgrad"],
        "gradient_clip_norm": float(raw["gradient_clip_norm"]),
        "accumulation_divisor": raw["accumulation_divisor"],
        "step_index": raw["step_index"],
    }
    pinned = {
        "name": "AdamW",
        "learning_rate": 1.0e-4,
        "betas": [0.9, 0.999],
        "epsilon": 1.0e-8,
        "weight_decay": 0.0,
        "foreach": False,
        "fused": False,
        "capturable": False,
        "maximize": False,
        "amsgrad": False,
        "gradient_clip_norm": 1.0,
        "accumulation_divisor": 12,
        "step_index": 1,
    }
    for field in ("foreach", "fused", "capturable", "maximize", "amsgrad"):
        if type(normalized[field]) is not bool:
            _fail(f"training.optimizer.{field} must be a boolean")
    for field in ("accumulation_divisor", "step_index"):
        if type(normalized[field]) is not int or normalized[field] < 1:
            _fail(f"training.optimizer.{field} must be a positive integer")
    if len(normalized["betas"]) != 2 or any(
        not math.isfinite(float(value)) for value in normalized.values() if isinstance(value, float)
    ):
        _fail("training.optimizer contains an invalid finite scalar or beta contract")
    if normalized != pinned:
        _fail(f"V20 AdamW contract mismatch: expected={pinned} observed={normalized}")
    return normalized


def validate_v20_config_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the complete gradient-defining V20 launch surface."""

    from semantic_3d_chat.scene_encoder.global_residual import (
        global_scene_residual_settings,
    )
    from semantic_3d_chat.scene_encoder.signed_x_dispatch import (
        signed_x_scene_residual_settings,
    )
    from semantic_3d_chat.training.pair_curriculum import (
        pair_curriculum_settings,
        pair_objective_policy_contract,
        pair_objective_policy_settings,
    )

    observed_config_hash = config_hash(dict(config))
    if observed_config_hash != EXPECTED_RESOLVED_CONFIG_HASH:
        _fail(
            "V20 resolved config hash mismatch: "
            f"expected={EXPECTED_RESOLVED_CONFIG_HASH} observed={observed_config_hash}"
        )
    if config.get("structural_preflight") is not None or config.get("v18_screen") is not None:
        _fail("V20 must not inherit a completed V18 controller or preflight contract")
    scene_encoder = _mapping(config.get("scene_encoder"), "scene_encoder")
    if scene_encoder.get("global_latents") != 256:
        _fail("V20 requires exactly 256 globally complete scene slots")
    global_settings = global_scene_residual_settings(config)
    signed_settings = signed_x_scene_residual_settings(config)
    if (
        not global_settings.enabled
        or global_settings.architecture_version != ZERO_SPATIAL_MEAN_CONTENT_GATE_V1
        or global_settings.width != 128
    ):
        _fail("V20 requires the enabled 128D centered-content V18 residual base")
    if (
        not signed_settings.enabled
        or signed_settings.architecture_version != SIGNED_X_LOCAL_FIELD_V2
        or signed_settings.expected_initial_state_sha256 != EXPECTED_SIGNED_X_INITIAL_STATE_SHA256
    ):
        _fail("V20 signed-X local-field architecture or initial-state pin mismatch")

    training = _mapping(config.get("training"), "training")
    optimizer = _validate_optimizer_contract(_mapping(training.get("optimizer"), "optimizer"))
    training_checks = {
        "output_namespace": training.get("output_namespace")
        == "gemma4_color_mirror_signed_x_local_field_v20",
        "initialize_from": training.get("initialize_from")
        == "data_gemma4/checkpoints/gemma4_color_mirror_centered_content_gate_v18/epoch_004",
        "initialize_expected_adapter_sha256": training.get("initialize_expected_adapter_sha256")
        == EXPECTED_SOURCE_ADAPTER_SHA256,
        "initialize_expected_metadata_sha256": training.get("initialize_expected_metadata_sha256")
        == EXPECTED_SOURCE_METADATA_SHA256,
        "initialize_expected_global_scene_residual_state_sha256": training.get(
            "initialize_expected_global_scene_residual_state_sha256"
        )
        == EXPECTED_SOURCE_GLOBAL_RESIDUAL_SHA256,
        "initialize_source_residual_into_frozen_base": training.get(
            "initialize_source_residual_into_frozen_base"
        )
        is True,
        "initialize_named_lora_freeze_transition": training.get(
            "initialize_named_lora_freeze_transition"
        )
        is False,
        "freeze_scene_adapter": training.get("freeze_scene_adapter") is True,
        "train_global_scene_residual_only": training.get("train_global_scene_residual_only")
        is False,
        "train_signed_x_scene_residual_only": training.get("train_signed_x_scene_residual_only")
        is True,
        "epochs": training.get("epochs") == 4,
        "batch_size": training.get("batch_size") == 2,
        "max_questions_per_scene": training.get("max_questions_per_scene") == 6,
        "gradient_accumulation": training.get("gradient_accumulation") == 12,
        "gradient_clip_norm": float(training.get("gradient_clip_norm", math.nan)) == 1.0,
        "learning_rate": float(training.get("learning_rate", math.nan)) == 1.0e-4,
        "weight_decay": float(training.get("weight_decay", math.nan)) == 0.0,
        "pair_only_mode": training.get("pair_only_mode") is True,
        "pair_only_scene_ids": training.get("pair_only_scene_ids") == list(EXPECTED_SCENE_IDS),
        "pair_steps_per_epoch": training.get("pair_steps_per_epoch") == 12,
        "pair_units_per_batch": training.get("pair_units_per_batch") == 1,
        "pair_batch_fraction": float(training.get("pair_batch_fraction", math.nan)) == 1.0,
        "pair_max_units_per_pair": training.get("pair_max_units_per_pair") == 6,
        "pair_ranking_mode": training.get("pair_ranking_mode") == "candidate_logit",
        "grounding_weight": float(training.get("grounding_weight", math.nan)) == 0.0,
        "grounding_anchor_weight": float(training.get("grounding_anchor_weight", math.nan)) == 0.0,
        "latent_diversity_weight": float(training.get("latent_diversity_weight", math.nan)) == 0.0,
        "paired_scene_separation_weight": float(
            training.get("paired_scene_separation_weight", math.nan)
        )
        == 0.0,
        "spatial_answer_contrastive_weight": float(
            training.get("spatial_answer_contrastive_weight", math.nan)
        )
        == 0.0,
        "spatial_relation_contrastive_weight": float(
            training.get("spatial_relation_contrastive_weight", math.nan)
        )
        == 0.0,
        "optimizer": training.get("optimizer") == optimizer,
    }
    failed_training = sorted(name for name, passed in training_checks.items() if not passed)
    if failed_training:
        _fail(f"V20 training contract mismatch: {failed_training}")

    curriculum = pair_curriculum_settings(config)
    if (
        not curriculum.enabled
        or not curriculum.pair_only
        or curriculum.steps_per_epoch != 12
        or curriculum.units_per_batch != 1
        or curriculum.ranking_mode != "candidate_logit"
    ):
        _fail("V20 pair curriculum no longer defines the exact twelve-step candidate schedule")
    policies = pair_objective_policy_settings(config)
    expected_policy_contracts = {
        COLOR_PAIR_ID: {
            "role": "retention_control",
            "language_nll_weight": 0.0,
            "candidate_hinge_weight": 8.0,
            "candidate_margin": 0.25,
            "full_vocab_hinge_weight": 2.0,
            "full_vocab_margin": 0.25,
        },
        MIRROR_PAIR_ID: {
            "role": "signed_target",
            "language_nll_weight": 0.0,
            "candidate_hinge_weight": 8.0,
            "candidate_margin": 1.0,
            "full_vocab_hinge_weight": 2.0,
            "full_vocab_margin": 1.0,
        },
    }
    observed_policy_contracts = {
        pair_id: policies.resolve(pair_id).contract() for pair_id in policies.pair_ids
    }
    if (
        not policies.configured
        or policies.allow_unlisted_pair_ids
        or observed_policy_contracts != expected_policy_contracts
    ):
        _fail("V20 opaque per-pair objective policy mismatch")

    experiment = _mapping(config.get("experiment"), "experiment")
    experiment_checks = {
        "schema_version": experiment.get("schema_version") == 1,
        "role": experiment.get("role") == V20_EXPERIMENT_ROLE,
        "question_dependent_scene_processing": experiment.get("question_dependent_scene_processing")
        is False,
        "source_checkpoint_epoch": experiment.get("source_checkpoint_epoch") == 4,
        "residual_parameter_count": experiment.get("residual_parameter_count") == 400_128,
        "signed_x_residual_parameter_count": experiment.get("signed_x_residual_parameter_count")
        == 196_608,
        "screen_optimizer_updates": experiment.get("screen_optimizer_updates") == 4,
        "conditional_max_optimizer_updates": experiment.get("conditional_max_optimizer_updates")
        == 8,
        "source_scene_state_sha256": experiment.get("source_scene_state_sha256")
        == EXPECTED_SOURCE_SCENE_STATE_SHA256,
        "source_global_scene_residual_state_sha256": experiment.get(
            "source_global_scene_residual_state_sha256"
        )
        == EXPECTED_SOURCE_GLOBAL_RESIDUAL_SHA256,
        "source_inherited_bank_sha256": experiment.get("source_inherited_bank_sha256")
        == EXPECTED_SOURCE_LORA_SHA256["inherited_v12"],
        "source_extension_bank_sha256": experiment.get("source_extension_bank_sha256")
        == EXPECTED_SOURCE_LORA_SHA256["extension_v13"],
    }
    failed_experiment = sorted(name for name, passed in experiment_checks.items() if not passed)
    if failed_experiment:
        _fail(f"V20 experiment contract mismatch: {failed_experiment}")

    screen = _mapping(config.get("v20_screen"), "v20_screen")
    expected_screen = {
        "schema_version": 1,
        "role": V20_SCREEN_ROLE,
        "source_checkpoint_epoch": 4,
        "screen_optimizer_updates": 4,
        "conditional_max_optimizer_updates": 8,
        "stage_1_optimizer_updates": 1,
        "stage_1_stop_required": True,
        "structural_preflight_requires": {
            "maximum_per_scene_raw_and_effective_delta_to_core_rms_ratio": 0.01,
            "minimum_mirror_effective_residual_to_core_rms_ratio": 0.01,
            "minimum_mirror_to_color_normalized_effective_selectivity": 1.5,
            "minimum_local_hidden_spatial_rank": 2,
        },
        "eligibility_requires": {
            "color_full_vocab_sides": 12,
            "color_full_vocab_units": 6,
            "color_positive_minimum_candidate_margin": True,
            "color_positive_minimum_full_vocab_margin": True,
        },
        "continuation_requires": {
            "mirror_minimum_full_vocab_sides": 8,
            "mirror_minimum_full_vocab_units": 2,
        },
        "full_teacher_gate_requires": {
            "color_full_vocab_sides": 12,
            "color_full_vocab_units": 6,
            "mirror_full_vocab_sides": 12,
            "mirror_full_vocab_units": 6,
            "all_candidate_and_full_vocab_minimum_margins_positive": True,
        },
        "greedy_audit_only_after_full_teacher_gate": True,
    }
    if dict(screen) != expected_screen:
        _fail("V20 staged screen contract mismatch")

    language = _mapping(config.get("language"), "language")
    language_checks = {
        "model_id": language.get("model_id") == "google/gemma-4-E2B-it",
        "revision": language.get("revision") == "3e22461f65e89153144f8adb70e3b8c2cc9845a7",
        "backend": language.get("backend") == "gemma4",
        "dtype": language.get("dtype") == "bfloat16",
        "scene_prefix_after_bos": language.get("scene_prefix_after_bos") is True,
        "scene_boundary_mode": language.get("scene_boundary_mode") == "gemma4_native_image",
    }
    failed_language = sorted(name for name, passed in language_checks.items() if not passed)
    if failed_language:
        _fail(f"V20 language contract mismatch: {failed_language}")

    normalized = {
        "schema_version": 1,
        "role": V20_PREFLIGHT_ROLE,
        "source_checkpoint_epoch": 4,
        "latent_count": 256,
        "scene_dim": 1536,
        "content_dim": 128,
        "signed_x_parameter_count": 196_608,
        "exact_epoch": 1,
        "microsteps": 12,
        "resolved_config_hash": observed_config_hash,
        "structural_preflight_requires": expected_screen["structural_preflight_requires"],
        "optimizer": optimizer,
        "pair_objective_policy": pair_objective_policy_contract(policies),
        "expected_hashes": {
            "source_adapter_sha256": EXPECTED_SOURCE_ADAPTER_SHA256,
            "source_metadata_sha256": EXPECTED_SOURCE_METADATA_SHA256,
            "source_scene_state_sha256": EXPECTED_SOURCE_SCENE_STATE_SHA256,
            "source_global_scene_residual_state_sha256": (EXPECTED_SOURCE_GLOBAL_RESIDUAL_SHA256),
            "source_lora_bank_state_sha256": dict(EXPECTED_SOURCE_LORA_SHA256),
            "initial_signed_x_state_sha256": EXPECTED_SIGNED_X_INITIAL_STATE_SHA256,
            "selection_sha256": EXPECTED_SELECTION_SHA256,
            "pair_unit_selection_sha256": EXPECTED_PAIR_UNIT_SELECTION_SHA256,
            "ordered_unit_sha256": EXPECTED_ORDERED_UNIT_SHA256,
            "pair_membership_sha256": EXPECTED_PAIR_MEMBERSHIP_SHA256,
        },
        "v20_screen": expected_screen,
    }
    normalized["contract_sha256"] = canonical_sha256(normalized)
    return normalized


def ordered_curriculum_evidence(curriculum: Sequence[Any]) -> tuple[list[dict[str, Any]], str]:
    """Return the exact opaque identity and hash of the V20 epoch-one order."""

    entries: list[dict[str, Any]] = []
    for microstep, batch in enumerate(curriculum, start=1):
        if getattr(batch, "kind", None) != "pair":
            _fail(f"V20 microstep {microstep} is not a pair batch")
        units = tuple(getattr(batch, "pair_units", ()))
        if len(units) != 1:
            _fail(f"V20 microstep {microstep} must contain exactly one pair unit")
        unit = units[0]
        entries.append(
            {
                "microstep": microstep,
                "pair_id": str(unit.pair_id),
                "question_key": str(unit.question_key),
                "reference_scene_id": str(unit.reference.scene_id),
                "reference_question_id": str(unit.reference.question_id),
                "counterfactual_scene_id": str(unit.counterfactual.scene_id),
                "counterfactual_question_id": str(unit.counterfactual.question_id),
            }
        )
    return entries, canonical_sha256(entries)


def pair_unit_selection_evidence(units: Sequence[Any]) -> tuple[list[dict[str, Any]], str]:
    """Hash pair units using the trainer's persisted selection-report schema."""

    entries = sorted(
        (
            {
                "pair_id": str(unit.pair_id),
                "question_key": str(unit.question_key),
                "scene_ids": [str(scene_id) for scene_id in unit.scene_ids],
                "question_ids": [str(record.question_id) for record in unit.records],
            }
            for unit in units
        ),
        key=lambda value: (
            value["pair_id"],
            value["question_key"],
            value["question_ids"][0],
            value["question_ids"][1],
        ),
    )
    return entries, canonical_sha256(entries)


def signed_x_residual_state_sha256(module: nn.Module, output_weight: torch.Tensor) -> str:
    """Hash a predicted full signed-X module in checkpoint namespace."""

    state = {
        f"signed_x_scene_residual.{name}": value.detach()
        for name, value in module.state_dict().items()
    }
    state["signed_x_scene_residual.output_projection.weight"] = output_weight.detach().to(
        device=module.output_projection.weight.device,
        dtype=module.output_projection.weight.dtype,
    )
    return tensor_state_sha256(state)


def exact_clone_adamw_evidence(
    module: nn.Module,
    optimizer_contract: Mapping[str, Any],
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Apply the configured update to a clone, leaving the live module untouched."""

    live_named = list(module.named_parameters())
    if [name for name, _parameter in live_named] != ["output_projection.weight"]:
        raise ValueError("V20 signed-X trainable surface is not exactly the output weight")
    if live_named[0][1].grad is None:
        raise ValueError("V20 signed-X clone simulation requires an accumulated gradient")
    clone = copy.deepcopy(module)
    clone_named = list(clone.named_parameters())
    clone_parameter = clone_named[0][1]
    live_parameter = live_named[0][1]
    clone_parameter.grad = live_parameter.grad.detach().clone()
    unclipped_gradient = clone_parameter.grad.detach().float().cpu().contiguous()
    pre_clip_norm = float(clone_parameter.grad.detach().float().norm().cpu())
    returned_norm = torch.nn.utils.clip_grad_norm_(
        [clone_parameter], float(optimizer_contract["gradient_clip_norm"])
    )
    clipped_gradient = clone_parameter.grad.detach().float().cpu().contiguous()
    optimizer = torch.optim.AdamW(
        [
            {
                "name": SIGNED_X_OPTIMIZER_GROUP_NAME,
                "params": [clone_parameter],
                "lr": float(optimizer_contract["learning_rate"]),
                "weight_decay": float(optimizer_contract["weight_decay"]),
            }
        ],
        betas=tuple(float(value) for value in optimizer_contract["betas"]),
        eps=float(optimizer_contract["epsilon"]),
        foreach=bool(optimizer_contract["foreach"]),
        fused=bool(optimizer_contract["fused"]),
        capturable=bool(optimizer_contract["capturable"]),
        maximize=bool(optimizer_contract["maximize"]),
        amsgrad=bool(optimizer_contract["amsgrad"]),
    )
    if optimizer.state:
        raise RuntimeError("Fresh signed-X clone AdamW unexpectedly has state")
    optimizer.step()
    simulated_weight = clone.output_projection.weight.detach().clone()
    if torch.equal(simulated_weight, live_parameter.detach()):
        raise RuntimeError("Signed-X clone AdamW produced no parameter update")
    optimizer_manifest, optimizer_sha256 = canonical_v19_adamw_state(
        optimizer.state_dict(), optimizer_contract
    )
    state_tensors = {
        f"output_projection.weight.{state_name}": value.detach().float().cpu().contiguous()
        for state_name, value in optimizer.state[clone_parameter].items()
        if isinstance(value, torch.Tensor)
    }
    report = {
        "implementation": "isolated_signed_x_torch_adamw_clone",
        "gradient_parameter_keys": ["output_projection.weight"],
        "changed_parameter_keys": ["output_projection.weight"],
        "parameter_count": int(simulated_weight.numel()),
        "pre_clip_gradient_l2_norm": pre_clip_norm,
        "clip_returned_pre_clip_gradient_l2_norm": float(returned_norm.detach().cpu()),
        "post_clip_gradient_l2_norm": float(clone_parameter.grad.detach().float().norm().cpu()),
        "gradient_sha256": tensor_state_sha256({"output_projection.weight": unclipped_gradient}),
        "clipped_gradient_sha256": tensor_state_sha256(
            {"output_projection.weight": clipped_gradient}
        ),
        "predicted_output_weight_sha256": tensor_state_sha256(
            {"signed_x_scene_residual.output_projection.weight": simulated_weight}
        ),
        "predicted_signed_x_scene_residual_state_sha256": (
            signed_x_residual_state_sha256(module, simulated_weight)
        ),
        "update_l2_norm": float(simulated_weight.float().norm().cpu()),
        "update_rms": float(simulated_weight.float().square().mean().sqrt().cpu()),
        "update_absolute_maximum": float(simulated_weight.float().abs().max().cpu()),
        "nonzero_update_count": int(torch.count_nonzero(simulated_weight).cpu()),
        "finite_update": bool(torch.isfinite(simulated_weight).all()),
        "canonical_adamw_state_manifest": optimizer_manifest,
        "canonical_adamw_state_sha256": optimizer_sha256,
        "optimizer_state_tensor_sha256": tensor_state_sha256(state_tensors),
    }
    return simulated_weight, report


def functional_local_field_delta(
    module: nn.Module,
    centered_content: torch.Tensor,
    simulated_output_weight: torch.Tensor,
    *,
    base_tokens: torch.Tensor,
    model_dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the FP32 delta and exact decoder-visible BF16 cast delta.

    The scene tokenizer and both residual branches produce FP32 tokens.  The
    trainer and chat runtime apply V20 in FP32 and only then cast the completed
    scene tokens at the Gemma boundary.  The exact arithmetic is therefore
    ``BF16(base_fp32 + delta_fp32) - BF16(base_fp32)``.  Casting either operand
    before the addition is not equivalent.
    """

    if simulated_output_weight.shape != module.output_projection.weight.shape:
        raise ValueError("Simulated signed-X output weight shape mismatch")
    if model_dtype is not torch.bfloat16:
        raise ValueError("V20 structural preflight requires Gemma bfloat16")
    if base_tokens.dtype is not torch.float32:
        raise ValueError("V20 structural preflight requires FP32 pre-Gemma scene tokens")
    hidden = module.hidden_values(centered_content)
    raw_delta = F.linear(hidden, simulated_output_weight.float())
    raw_delta = raw_delta - raw_delta.mean(dim=1, keepdim=True)
    base_model = base_tokens.to(dtype=model_dtype)
    adapted_model = (base_tokens + raw_delta).to(dtype=model_dtype)
    effective_delta = adapted_model.float() - base_model.float()
    return raw_delta, effective_delta


def bf16_cast_audit(
    base_tokens: torch.Tensor,
    raw_delta: torch.Tensor,
    effective_delta: torch.Tensor,
) -> dict[str, Any]:
    """Quantify exact BF16 survival and round-trip error for one scene."""

    if raw_delta.shape != base_tokens.shape or effective_delta.shape != base_tokens.shape:
        raise ValueError("Base, raw delta, and effective delta shapes must match")
    raw = raw_delta.detach().float().double()
    effective = effective_delta.detach().float().double()
    error = effective - raw
    raw_rms = float(raw.square().mean().sqrt())
    effective_rms = float(effective.square().mean().sqrt())
    error_rms = float(error.square().mean().sqrt())
    raw_norm = float(raw.norm())
    effective_norm = float(effective.norm())
    cosine = None
    if raw_norm > 0.0 and effective_norm > 0.0:
        cosine = float(torch.dot(raw.reshape(-1), effective.reshape(-1))) / (
            raw_norm * effective_norm
        )
    changed = effective != 0
    return {
        "schema_version": 1,
        "algorithm": "bfloat16_cast_of_fp32_base_plus_fp32_delta",
        "base_source_dtype": str(base_tokens.dtype).removeprefix("torch."),
        "model_dtype": "bfloat16",
        "comparison_dtype": "float64",
        "element_count": int(effective.numel()),
        "changed_element_count": int(changed.sum()),
        "changed_element_fraction": float(changed.double().mean()),
        "raw_delta_rms": raw_rms,
        "effective_delta_rms": effective_rms,
        "effective_to_raw_rms_ratio": (None if raw_rms == 0.0 else effective_rms / raw_rms),
        "quantization_error_rms": error_rms,
        "quantization_error_to_raw_rms_ratio": (None if raw_rms == 0.0 else error_rms / raw_rms),
        "raw_effective_cosine": cosine,
        "raw_delta_sha256": tensor_state_sha256({"raw_fp32_delta": raw_delta}),
        "effective_delta_sha256": tensor_state_sha256(
            {"bf16_effective_delta_float32": effective_delta}
        ),
    }


def spatial_rank_evidence(
    values: torch.Tensor, *, relative_tolerance: float = 1.0e-5
) -> dict[str, Any]:
    """Return deterministic per-batch spatial matrix-rank evidence."""

    if values.ndim != 3 or values.shape[1] < 2 or values.shape[2] < 1:
        raise ValueError("Spatial rank values must have shape [B,L,C] with L >= 2")
    if not math.isfinite(relative_tolerance) or relative_tolerance <= 0.0:
        raise ValueError("relative_tolerance must be positive and finite")
    batches: list[dict[str, Any]] = []
    for batch_index, matrix in enumerate(values.detach().float().cpu()):
        singular = torch.linalg.svdvals(matrix)
        maximum = float(singular.max()) if singular.numel() else 0.0
        threshold = maximum * relative_tolerance
        rank = int(torch.count_nonzero(singular > threshold)) if maximum > 0.0 else 0
        energy = singular.square()
        stable_rank = 0.0 if maximum == 0.0 else float(energy.sum() / (maximum * maximum))
        batches.append(
            {
                "batch_index": batch_index,
                "spatial_rank": rank,
                "stable_rank": stable_rank,
                "maximum_singular_value": maximum,
                "rank_threshold": threshold,
                "top_singular_values": [float(value) for value in singular[:8]],
            }
        )
    return {
        "schema_version": 1,
        "shape": list(values.shape),
        "relative_tolerance": relative_tolerance,
        "minimum_spatial_rank": min(item["spatial_rank"] for item in batches),
        "batches": batches,
    }


def local_dependence_evidence(module: nn.Module) -> dict[str, Any]:
    """Prove every input slot changes only its corresponding local hidden slot.

    The probes are paired ``+1/-1`` perturbations, so each input remains
    exactly centered across slots and exercises the module's public validation
    path.  V19's global moment changes all output slots for these probes; V20's
    unreduced field changes exactly the two perturbed slots.
    """

    latent_count = int(module.latent_count)
    content_dim = int(module.content_dim)
    if latent_count % 2:
        raise ValueError("Local-dependence proof requires an even latent count")
    probe_count = latent_count // 2
    probes = torch.zeros(
        probe_count,
        latent_count,
        content_dim,
        device=module.output_projection.weight.device,
        dtype=torch.float32,
    )
    rows = torch.arange(probe_count, device=probes.device)
    first_slots = rows * 2
    second_slots = first_slots + 1
    probes[rows, first_slots, 0] = 1.0
    probes[rows, second_slots, 0] = -1.0
    with torch.no_grad():
        hidden = module.hidden_values(probes)
    changed = hidden.detach().float().abs().amax(dim=-1) > 0.0
    expected = torch.zeros_like(changed)
    expected[rows, first_slots] = True
    expected[rows, second_slots] = True
    changed_per_probe = changed.sum(dim=1)
    changed_slot_union = changed.any(dim=0)
    exact_local_support = torch.equal(changed, expected)
    return {
        "schema_version": 1,
        "probe_shape": list(probes.shape),
        "hidden_shape": list(hidden.shape),
        "probe_count": probe_count,
        "paired_centered_perturbations": True,
        "maximum_probe_spatial_mean_absolute": float(probes.mean(dim=1).abs().max().cpu()),
        "minimum_changed_slots_per_probe": int(changed_per_probe.min().cpu()),
        "maximum_changed_slots_per_probe": int(changed_per_probe.max().cpu()),
        "changed_slot_union_count": int(changed_slot_union.sum().cpu()),
        "all_input_slots_exercised": bool(changed_slot_union.all()),
        "unperturbed_output_slots_exactly_unchanged": exact_local_support,
        "exact_two_slot_local_support": bool(
            exact_local_support and torch.all(changed_per_probe == 2)
        ),
        "no_global_moment_broadcast": exact_local_support,
        "hidden_sha256": tensor_state_sha256({"local_probe_hidden": hidden}),
    }


def normalized_pair_selectivity(
    pair_metrics: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Compare pair residuals after normalizing each by its own core signal."""

    if COLOR_PAIR_ID not in pair_metrics or MIRROR_PAIR_ID not in pair_metrics:
        raise ValueError("Color and mirror pair metrics are both required")
    color = float(pair_metrics[COLOR_PAIR_ID]["residual_to_core_pair_difference_ratio"])
    mirror = float(pair_metrics[MIRROR_PAIR_ID]["residual_to_core_pair_difference_ratio"])
    if not math.isfinite(color) or color < 0.0 or not math.isfinite(mirror) or mirror < 0.0:
        raise ValueError("Pair residual/core ratios must be finite and non-negative")
    ratio = None if color == 0.0 else mirror / color
    return {
        "schema_version": 1,
        "color_residual_to_core_rms_ratio": color,
        "mirror_residual_to_core_rms_ratio": mirror,
        "color_ratio_exact_zero": color == 0.0,
        "mirror_to_color_normalized_selectivity": ratio,
    }


def evaluate_v20_structural_gate(
    raw_scene_metrics: Mapping[str, Mapping[str, Any]],
    effective_scene_metrics: Mapping[str, Mapping[str, Any]],
    *,
    raw_pair_metrics: Mapping[str, Mapping[str, Any]],
    effective_pair_metrics: Mapping[str, Mapping[str, Any]],
    bf16_audits: Mapping[str, Mapping[str, Any]],
    structural_state: Mapping[str, Any],
    local_dependence: Mapping[str, Any],
    local_hidden_ranks: Mapping[str, Mapping[str, Any]],
    requirements: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply V20's predeclared local-field, BF16, and selectivity gate."""

    if not raw_scene_metrics or set(raw_scene_metrics) != set(effective_scene_metrics):
        raise ValueError("Raw/effective V20 scene metric sets must be equal and nonempty")
    if set(raw_scene_metrics) != set(bf16_audits) or set(raw_scene_metrics) != set(
        local_hidden_ranks
    ):
        raise ValueError("V20 scene, BF16-audit, and local-rank sets must be equal")
    maximum_ratio = float(
        requirements["maximum_per_scene_raw_and_effective_delta_to_core_rms_ratio"]
    )
    minimum_mirror_ratio = float(
        requirements["minimum_mirror_effective_residual_to_core_rms_ratio"]
    )
    minimum_selectivity = float(
        requirements["minimum_mirror_to_color_normalized_effective_selectivity"]
    )
    minimum_rank = int(requirements["minimum_local_hidden_spatial_rank"])
    per_scene: dict[str, dict[str, bool]] = {}
    for scene_id in sorted(raw_scene_metrics):
        raw = raw_scene_metrics[scene_id]
        effective = effective_scene_metrics[scene_id]
        cast = bf16_audits[scene_id]
        rank = local_hidden_ranks[scene_id]
        per_scene[scene_id] = {
            "raw_positive_finite_total_energy": bool(raw["positive_finite_total_energy"]),
            "raw_fp32_centered": float(raw["across_slot_mean_energy_fraction"]) <= 1.0e-6,
            "raw_slot_varying": float(raw["slot_varying_energy_fraction"]) >= 0.999999,
            "raw_delta_ratio_bounded": float(raw["delta_to_core_rms_ratio"]) <= maximum_ratio,
            "effective_finite": bool(effective["positive_finite_total_energy"]),
            "effective_delta_ratio_bounded": float(effective["delta_to_core_rms_ratio"])
            <= maximum_ratio,
            "bf16_changed_nonzero": int(cast["changed_element_count"]) > 0,
            "bf16_quantization_error_finite": math.isfinite(float(cast["quantization_error_rms"])),
            "local_hidden_spatial_rank": int(rank["minimum_spatial_rank"]) >= minimum_rank,
        }
    all_slots = (
        structural_state.get("all_slots_accounted") is True
        and structural_state.get("accounted_slot_count") == structural_state.get("latent_count")
        and structural_state.get("latent_count") == 256
    )
    local_structure = (
        structural_state.get("architecture_version") == SIGNED_X_LOCAL_FIELD_V2
        and structural_state.get("architecture_marker") == 2
        and structural_state.get("spatial_reduction") == "none"
        and local_dependence.get("all_input_slots_exercised") is True
        and local_dependence.get("exact_two_slot_local_support") is True
        and local_dependence.get("no_global_moment_broadcast") is True
    )
    raw_selectivity = normalized_pair_selectivity(raw_pair_metrics)
    effective_selectivity = normalized_pair_selectivity(effective_pair_metrics)
    observed_selectivity = effective_selectivity["mirror_to_color_normalized_selectivity"]
    selectivity_checks = {
        "raw_mirror_residual_positive_finite": float(
            raw_selectivity["mirror_residual_to_core_rms_ratio"]
        )
        > 0.0,
        "effective_mirror_residual_at_least_minimum": float(
            effective_selectivity["mirror_residual_to_core_rms_ratio"]
        )
        >= minimum_mirror_ratio,
        "effective_normalized_selectivity_at_least_minimum": (
            effective_selectivity["color_ratio_exact_zero"] is True
            and float(effective_selectivity["mirror_residual_to_core_rms_ratio"]) > 0.0
        )
        or (
            observed_selectivity is not None and float(observed_selectivity) >= minimum_selectivity
        ),
    }
    passed = (
        all_slots
        and local_structure
        and all(all(checks.values()) for checks in per_scene.values())
        and all(selectivity_checks.values())
    )
    return {
        "schema_version": 1,
        "requirements": dict(requirements),
        "all_slots_accounted": all_slots,
        "local_field_structure_verified": local_structure,
        "scene_checks": per_scene,
        "selectivity_checks": selectivity_checks,
        "raw_pair_selectivity": raw_selectivity,
        "bf16_effective_pair_selectivity": effective_selectivity,
        "maximum_observed_raw_delta_to_core_rms_ratio": max(
            float(value["delta_to_core_rms_ratio"]) for value in raw_scene_metrics.values()
        ),
        "maximum_observed_bf16_effective_delta_to_core_rms_ratio": max(
            float(value["delta_to_core_rms_ratio"]) for value in effective_scene_metrics.values()
        ),
        "passed": passed,
    }


def _module_gradient_manifest(modules: Mapping[str, nn.Module]) -> dict[str, Any]:
    entries: dict[str, Any] = {}
    for module_name, module in modules.items():
        for name, parameter in module.named_parameters():
            if parameter.grad is None:
                continue
            gradient = parameter.grad.detach().float().cpu().contiguous()
            entries[f"{module_name}.{name}"] = {
                "l2_norm": float(gradient.norm()),
                "nonzero_count": int(torch.count_nonzero(gradient)),
                "finite": bool(torch.isfinite(gradient).all()),
                "sha256": tensor_state_sha256({f"{module_name}.{name}": gradient}),
            }
    return {
        "parameter_keys_with_gradient_tensor": sorted(entries),
        "entries": entries,
    }


def _zero_output_equivalence(
    *,
    core_outputs: Mapping[str, Any],
    base_outputs: Mapping[str, Any],
    centered_content: Mapping[str, torch.Tensor],
    signed_x_scene_residual: nn.Module,
    composer: nn.Module,
    model_dtype: torch.dtype,
) -> dict[str, Any]:
    from semantic_3d_chat.scene_encoder.signed_x_dispatch import apply_signed_x_scene_residual
    from semantic_3d_chat.training.train_adapter import prefix_sha256

    scenes: dict[str, Any] = {}
    with torch.no_grad():
        for scene_id in sorted(base_outputs):
            base = base_outputs[scene_id]
            adapted = apply_signed_x_scene_residual(
                base, signed_x_scene_residual, centered_content[scene_id]
            )
            tokens_equal = torch.equal(base.scene_tokens, adapted.scene_tokens)
            base_prefix = composer.scene_prefix(base.scene_tokens.to(dtype=model_dtype))
            adapted_prefix = composer.scene_prefix(adapted.scene_tokens.to(dtype=model_dtype))
            prefixes_equal = torch.equal(base_prefix, adapted_prefix)
            base_hash = prefix_sha256(base_prefix)
            adapted_hash = prefix_sha256(adapted_prefix)
            scenes[scene_id] = {
                "core_scene_token_sha256": tensor_state_sha256(
                    {"scene_tokens": core_outputs[scene_id].scene_tokens}
                ),
                "v18_base_scene_token_sha256": tensor_state_sha256(
                    {"scene_tokens": base.scene_tokens}
                ),
                "v18_base_prefix_sha256": base_hash,
                "signed_x_adapted_prefix_sha256": adapted_hash,
                "scene_tokens_exactly_equal": tokens_equal,
                "prefixes_exactly_equal": prefixes_equal,
                "prefix_hashes_equal": base_hash == adapted_hash,
            }
    verified = len(scenes) == len(EXPECTED_SCENE_IDS) and all(
        evidence["scene_tokens_exactly_equal"]
        and evidence["prefixes_exactly_equal"]
        and evidence["prefix_hashes_equal"]
        for evidence in scenes.values()
    )
    return {
        "verified": verified,
        "question_dependent_scene_processing": False,
        "base": "loaded_frozen_global_scene_residual",
        "all_scene_slots_accounted": True,
        "scene_count": len(scenes),
        "scene_prefixes": scenes,
    }


def run_preflight(config_path: str | Path, report_path: str | Path) -> dict[str, Any]:
    """Execute the real V20 epoch-one diagnostic without a live optimizer step."""

    # Heavy model and supervised-data dependencies stay out of unit-test imports.
    from semantic_3d_chat.config import (
        PROJECT_ROOT,
        artifact_root,
        config_hash,
        load_config,
        project_path,
    )
    from semantic_3d_chat.data.dataset import SceneQADataset
    from semantic_3d_chat.language.local_lm import load_local_language_model
    from semantic_3d_chat.language.lora import install_lora_banks, lora_banks_settings
    from semantic_3d_chat.language.prefix_injection import (
        ContinuousPrefixComposer,
        scene_boundary_mode_setting,
        scene_prefix_after_bos_setting,
    )
    from semantic_3d_chat.scene_encoder.global_residual import (
        apply_global_scene_residual,
        construct_global_scene_residual,
        global_scene_residual_settings,
    )
    from semantic_3d_chat.scene_encoder.map_io import load_map_tensors
    from semantic_3d_chat.scene_encoder.signed_x_dispatch import (
        apply_signed_x_scene_residual,
        construct_signed_x_scene_residual,
        frozen_v18_centered_content_values,
    )
    from semantic_3d_chat.training.checkpointing import (
        load_adapter_checkpoint,
        module_collection_state_sha256,
    )
    from semantic_3d_chat.training.losses import QuestionGroundingHead
    from semantic_3d_chat.training.pair_curriculum import (
        build_epoch_curriculum,
        build_exact_question_pair_units,
        cap_pair_units_per_pair,
        pair_curriculum_settings,
        pair_objective_policy_contract,
        pair_objective_policy_settings,
        ranking_margin_hinge,
        select_pair_only_records,
        validate_pair_objective_policy_coverage,
    )
    from semantic_3d_chat.training.source_provenance import (
        capture_git_source_provenance,
        require_clean_committed_source,
    )
    from semantic_3d_chat.training.train_adapter import (
        combine_pair_training_losses,
        construct_scene_tokenizer,
        pair_batch_objective,
        select_training_records,
        set_seed,
        training_counterfactual_scene_pairs,
        training_selection_summary,
        validate_global_scene_residual_state,
        validate_lora_banks_checkpoint_state,
        validate_signed_x_scene_residual_state,
    )

    config = load_config(config_path)
    contract = validate_v20_config_contract(config)
    source_provenance = capture_git_source_provenance(PROJECT_ROOT)
    try:
        require_clean_committed_source(source_provenance)
    except RuntimeError as error:
        _fail(f"V20 preflight requires clean committed source: {error}")
    set_seed(int(config["seed"]))
    training = config["training"]
    pair_settings = pair_curriculum_settings(config)
    pair_policies = pair_objective_policy_settings(config)

    dataset = SceneQADataset(artifact_root(config, "qa") / "train.jsonl")
    selected_available = select_pair_only_records(
        dataset.records, pair_settings.pair_only_scene_ids
    )
    selected_available = cap_pair_units_per_pair(
        selected_available,
        pair_settings.max_units_per_pair,
        seed=int(config["seed"]),
    )
    records = select_training_records(
        selected_available,
        max_questions_per_scene=training.get("max_questions_per_scene"),
    )
    selection = training_selection_summary(selected_available, records)
    if selection["selected_ids_sha256"] != EXPECTED_SELECTION_SHA256:
        _fail("V20 selected training-record hash mismatch")
    pair_units = build_exact_question_pair_units(records)
    selected_pair_units, pair_unit_selection_hash = pair_unit_selection_evidence(pair_units)
    if pair_unit_selection_hash != EXPECTED_PAIR_UNIT_SELECTION_SHA256:
        _fail(
            "V20 selected pair-unit hash mismatch: "
            f"expected={EXPECTED_PAIR_UNIT_SELECTION_SHA256} "
            f"observed={pair_unit_selection_hash}"
        )
    policy_coverage = validate_pair_objective_policy_coverage(
        pair_policies, sorted({unit.pair_id for unit in pair_units})
    )
    by_scene: dict[str, list[Any]] = defaultdict(list)
    for record in records:
        by_scene[record.scene_id].append(record)
    if tuple(sorted(by_scene)) != EXPECTED_SCENE_IDS:
        _fail(f"V20 selected scene mismatch: {sorted(by_scene)}")
    curriculum = build_epoch_curriculum(
        by_scene,
        pair_units,
        standard_batch_size=int(training["batch_size"]),
        pair_units_per_batch=pair_settings.units_per_batch,
        pair_batch_fraction=pair_settings.batch_fraction,
        pair_only=pair_settings.pair_only,
        seed=int(config["seed"]) + 1,
        steps_per_epoch=pair_settings.steps_per_epoch,
    )
    ordered_units, ordered_hash = ordered_curriculum_evidence(curriculum)
    if len(curriculum) != 12 or ordered_hash != EXPECTED_ORDERED_UNIT_SHA256:
        _fail(
            "V20 ordered epoch-one unit mismatch: "
            f"expected={EXPECTED_ORDERED_UNIT_SHA256} observed={ordered_hash}"
        )
    training_pairs = training_counterfactual_scene_pairs(records)
    pair_membership_text = "\n".join(
        f"{pair_id}:{first_scene}:{second_scene}"
        for pair_id, first_scene, second_scene in training_pairs
    )
    pair_membership_hash = hashlib.sha256(pair_membership_text.encode("utf-8")).hexdigest()
    if pair_membership_hash != EXPECTED_PAIR_MEMBERSHIP_SHA256:
        _fail("V20 pair-membership hash mismatch")

    language = load_local_language_model(
        config["language"]["model_id"],
        config["language"]["revision"],
        config["language"]["dtype"],
        freeze=True,
        local_files_only=True,
        backend=str(config["language"].get("backend", "auto")),
        decoder_gradient_checkpointing=bool(
            training.get("language_decoder_gradient_checkpointing", False)
        ),
    )
    language.model.config.use_cache = False
    lora = install_lora_banks(language.model, lora_banks_settings(config))
    if lora is None or lora.trainable_parameter_count != 0:
        _fail("V20 requires both named LoRA banks installed and entirely frozen")
    lora.eval()

    maps = {
        scene_id: load_map_tensors(
            project_path(config, "maps", scene_id, "voxel_map.npz"),
            config["scene"]["room_size_m"],
            language.device,
            input_voxel_size_m=config["scene_encoder"].get("input_voxel_size_m"),
        )
        for scene_id in EXPECTED_SCENE_IDS
    }
    feature_dims = {data.feature_dim for data in maps.values()}
    if len(feature_dims) != 1:
        _fail(f"V20 semantic feature dimensions differ: {sorted(feature_dims)}")
    scene_model = construct_scene_tokenizer(config, feature_dims.pop(), language.hidden_size).to(
        language.device
    )
    global_residual = construct_global_scene_residual(
        config,
        scene_dim=language.hidden_size,
        latent_count=int(config["scene_encoder"]["global_latents"]),
    )
    if global_residual is None:
        _fail("V20 global residual construction returned None")
    global_residual = global_residual.to(language.device)
    signed_residual = construct_signed_x_scene_residual(
        config,
        scene_dim=language.hidden_size,
        latent_count=int(config["scene_encoder"]["global_latents"]),
        content_dim=global_scene_residual_settings(config).width,
    )
    if signed_residual is None:
        _fail("V20 signed-X residual construction returned None")
    signed_residual = signed_residual.to(language.device)
    composer = ContinuousPrefixComposer(
        language.hidden_size,
        scene_prefix_after_bos=scene_prefix_after_bos_setting(config),
        bos_token_id=language.bos_token_id,
        scene_boundary_mode=scene_boundary_mode_setting(config),
        native_boundary_embeddings=language.scene_boundary_embeddings(
            scene_boundary_mode_setting(config)
        ),
    ).to(language.device)
    grounding = QuestionGroundingHead(
        int(config["scene_encoder"]["model_dim"]),
        language.hidden_size,
        int(config["scene_encoder"]["global_latents"]),
        int(config["scene_encoder"]["model_dim"]),
    ).to(language.device)

    initial_global_hash = module_collection_state_sha256({"global_scene_residual": global_residual})
    if initial_global_hash != global_scene_residual_settings(config).expected_initial_state_sha256:
        _fail("V20 deterministic global residual initial-state mismatch before source load")
    initial_signed_hash = module_collection_state_sha256(
        {"signed_x_scene_residual": signed_residual}
    )
    if initial_signed_hash != EXPECTED_SIGNED_X_INITIAL_STATE_SHA256:
        _fail("V20 deterministic signed-X initial-state mismatch")
    if torch.count_nonzero(signed_residual.output_projection.weight).item() != 0:
        _fail("V20 signed-X output projection is not exact zero at construction")

    source = Path(str(training["initialize_from"]))
    if not source.is_absolute():
        source = PROJECT_ROOT / source
    source = source.resolve()
    source_artifact_hashes = {
        "adapter_sha256": file_sha256(source / "adapter.safetensors"),
        "metadata_sha256": file_sha256(source / "metadata.json"),
    }
    if source_artifact_hashes != {
        "adapter_sha256": EXPECTED_SOURCE_ADAPTER_SHA256,
        "metadata_sha256": EXPECTED_SOURCE_METADATA_SHA256,
    }:
        _fail(f"V20 source artifact hash mismatch: {source_artifact_hashes}")
    scene_state_modules = {
        "scene_model": scene_model,
        "composer": composer,
        "grounding": grounding,
    }
    source_modules = {
        **scene_state_modules,
        "global_scene_residual": global_residual,
        **lora.state_modules(),
    }
    source_metadata = load_adapter_checkpoint(source, source_modules, device=str(language.device))
    if source_metadata.get("epoch") != 4:
        _fail("V20 must initialize from exact V18 epoch four")
    if (
        source_metadata.get("global_scene_residual")
        != global_scene_residual_settings(config).contract()
    ):
        _fail("V20 source global residual metadata contract mismatch")
    validate_lora_banks_checkpoint_state(source_metadata, lora)
    composer.validate_native_boundary_embeddings(
        language.scene_boundary_embeddings(scene_boundary_mode_setting(config))
    )
    validate_global_scene_residual_state(
        global_residual,
        expected_parameter_count=400_128,
        context="V20 preflight frozen V18 source",
    )
    validate_signed_x_scene_residual_state(
        signed_residual,
        expected_parameter_count=196_608,
        context="V20 preflight fresh signed-X branch",
    )
    observed_source_hashes = {
        **source_artifact_hashes,
        "scene_state_sha256": module_collection_state_sha256(scene_state_modules),
        "global_scene_residual_state_sha256": module_collection_state_sha256(
            {"global_scene_residual": global_residual}
        ),
        "lora_bank_state_sha256": lora.state_sha256(),
    }
    expected_source_hashes = {
        "adapter_sha256": EXPECTED_SOURCE_ADAPTER_SHA256,
        "metadata_sha256": EXPECTED_SOURCE_METADATA_SHA256,
        "scene_state_sha256": EXPECTED_SOURCE_SCENE_STATE_SHA256,
        "global_scene_residual_state_sha256": EXPECTED_SOURCE_GLOBAL_RESIDUAL_SHA256,
        "lora_bank_state_sha256": EXPECTED_SOURCE_LORA_SHA256,
    }
    if observed_source_hashes != expected_source_hashes:
        _fail(
            "V20 loaded source state hash mismatch: "
            f"expected={expected_source_hashes} observed={observed_source_hashes}"
        )

    for module in source_modules.values():
        module.requires_grad_(False).eval()
    signed_residual.requires_grad_(True).train()
    if [
        name for name, parameter in signed_residual.named_parameters() if parameter.requires_grad
    ] != ["output_projection.weight"]:
        _fail("V20 fresh branch trainable surface is not exactly its output projection")
    structural_state = signed_residual.validate_structural_state()
    if structural_state.get("parameter_count") != 196_608:
        _fail("V20 signed-X structural parameter count mismatch")

    live_source_state_before = module_collection_state_sha256(source_modules)
    live_signed_state_before = module_collection_state_sha256(
        {"signed_x_scene_residual": signed_residual}
    )
    with torch.no_grad():
        core_outputs = {
            scene_id: scene_model(
                data.semantic,
                data.xyz,
                data.rgb,
                data.normal,
                data.confidence,
                data.observation_count,
                data.room_min,
                data.room_max,
            )
            for scene_id, data in maps.items()
        }
        centered_content = {
            scene_id: frozen_v18_centered_content_values(global_residual, output.scene_tokens)
            for scene_id, output in core_outputs.items()
        }
        base_outputs = {
            scene_id: apply_global_scene_residual(output, global_residual)
            for scene_id, output in core_outputs.items()
        }
    zero_equivalence = _zero_output_equivalence(
        core_outputs=core_outputs,
        base_outputs=base_outputs,
        centered_content=centered_content,
        signed_x_scene_residual=signed_residual,
        composer=composer,
        model_dtype=language.model.get_input_embeddings().weight.dtype,
    )
    if not zero_equivalence["verified"]:
        _fail("V20 fresh signed-X branch failed exact step-zero identity")

    local_dependence = local_dependence_evidence(signed_residual)
    centered_content_evidence: dict[str, Any] = {}
    local_hidden_ranks: dict[str, dict[str, Any]] = {}
    for scene_id, values in centered_content.items():
        values32 = values.detach().float()
        with torch.no_grad():
            local_hidden = signed_residual.hidden_values(values)
        rank = spatial_rank_evidence(local_hidden)
        local_hidden_ranks[scene_id] = rank
        centered_content_evidence[scene_id] = {
            "shape": list(values.shape),
            "finite": bool(torch.isfinite(values32).all()),
            "across_slot_mean_absolute_maximum": float(values32.mean(dim=1).abs().max().cpu()),
            "local_hidden_rms": float(local_hidden.detach().square().mean().sqrt().cpu()),
            "local_hidden_sha256": tensor_state_sha256({"local_hidden": local_hidden}),
            "local_hidden_spatial_rank": rank,
            "sha256": tensor_state_sha256({"centered_content": values}),
        }

    rng_before = capture_rng_states(require_mps=language.device.type == "mps")
    signed_residual.zero_grad(set_to_none=True)
    zero = torch.zeros((), device=language.device)
    microstep_losses: list[dict[str, Any]] = []
    color_gradient_contributions: list[dict[str, Any]] = []
    mirror_gradient_contributions: list[dict[str, Any]] = []
    output_parameter = signed_residual.output_projection.weight
    for microstep, batch in enumerate(curriculum, start=1):
        unit = batch.pair_units[0]
        policy = pair_policies.resolve(unit.pair_id)
        outputs = {
            scene_id: apply_signed_x_scene_residual(
                base_outputs[scene_id], signed_residual, centered_content[scene_id]
            )
            for scene_id in unit.scene_ids
        }
        (
            base_loss,
            language_loss,
            grounding_loss,
            _legacy_ranking_loss,
            diagnostics,
        ) = pair_batch_objective(
            outputs,
            [unit],
            maps,
            language,
            composer,
            grounding,
            config,
            ranking_margin=pair_settings.ranking_margin,
            ranking_mode=pair_settings.ranking_mode,
            collect_full_vocab_first_answer_token=policy.full_vocab_hinge_weight > 0.0,
            full_vocab_ranking_margin=policy.full_vocab_margin,
        )
        margins = diagnostics["margins"]
        if not isinstance(margins, torch.Tensor):
            _fail("V20 pair objective did not return differentiable candidate margins")
        candidate_loss, _ = ranking_margin_hinge(margins, margin=policy.candidate_margin)
        full_vocab_loss = diagnostics["first_answer_token_full_vocab_ranking_loss"]
        full_vocab_margins = diagnostics["first_answer_token_full_vocab_margins"]
        if not isinstance(full_vocab_loss, torch.Tensor) or not isinstance(
            full_vocab_margins, torch.Tensor
        ):
            _fail("V20 pair objective did not return full-vocabulary ranking tensors")
        loss = combine_pair_training_losses(
            base_loss,
            candidate_loss,
            full_vocab_loss,
            zero,
            zero,
            language_loss=language_loss,
            language_nll_weight=policy.language_nll_weight,
            pair_ranking_weight=policy.candidate_hinge_weight,
            full_vocab_ranking_weight=policy.full_vocab_hinge_weight,
            diversity_weight=0.0,
            scene_separation_weight=0.0,
        )
        gradient_before = (
            torch.zeros_like(output_parameter)
            if output_parameter.grad is None
            else output_parameter.grad.detach().clone()
        )
        (loss / contract["optimizer"]["accumulation_divisor"]).backward()
        if output_parameter.grad is None:
            _fail(f"V20 microstep {microstep} produced no signed-X gradient tensor")
        gradient_after = output_parameter.grad.detach().clone()
        contribution = gradient_after - gradient_before
        contribution_evidence = {
            "microstep": microstep,
            "pair_id": unit.pair_id,
            "exact_zero": bool(torch.count_nonzero(contribution).item() == 0),
            "finite": bool(torch.isfinite(contribution).all()),
            "l2_norm": float(contribution.float().norm().cpu()),
            "nonzero_count": int(torch.count_nonzero(contribution).cpu()),
            "sha256": tensor_state_sha256({"gradient_contribution": contribution}),
        }
        if unit.pair_id == COLOR_PAIR_ID:
            color_gradient_contributions.append(contribution_evidence)
        elif unit.pair_id == MIRROR_PAIR_ID:
            mirror_gradient_contributions.append(contribution_evidence)
        else:
            _fail(f"Unexpected V20 opaque pair ID: {unit.pair_id}")
        microstep_losses.append(
            {
                "microstep": microstep,
                "pair_id": unit.pair_id,
                "question_key": unit.question_key,
                "pair_objective_policy": policy.contract(),
                "total_loss": float(loss.detach().cpu()),
                "raw_language_nll": float(language_loss.detach().cpu()),
                "weighted_language_nll": float(
                    policy.language_nll_weight * language_loss.detach().cpu()
                ),
                "grounding_loss": float(grounding_loss.detach().cpu()),
                "candidate_hinge_loss": float(candidate_loss.detach().cpu()),
                "weighted_candidate_hinge_loss": float(
                    policy.candidate_hinge_weight * candidate_loss.detach().cpu()
                ),
                "candidate_margins": margins.detach().float().cpu().tolist(),
                "full_vocab_hinge_loss": float(full_vocab_loss.detach().cpu()),
                "weighted_full_vocab_hinge_loss": float(
                    policy.full_vocab_hinge_weight * full_vocab_loss.detach().cpu()
                ),
                "full_vocab_margins": full_vocab_margins.detach().float().cpu().tolist(),
                "gradient_contribution": contribution_evidence,
            }
        )
        del outputs, diagnostics, loss, base_loss, language_loss, grounding_loss
        del candidate_loss, full_vocab_loss, margins, full_vocab_margins

    gradient_manifest = _module_gradient_manifest(
        {**source_modules, "signed_x_scene_residual": signed_residual}
    )
    gradient_keys = gradient_manifest["parameter_keys_with_gradient_tensor"]
    only_signed_output_has_gradient = gradient_keys == [
        "signed_x_scene_residual.output_projection.weight"
    ]
    final_gradient = output_parameter.grad
    if final_gradient is None:
        _fail("V20 accumulated gradient disappeared")
    gradient_finite_nonzero = bool(
        torch.isfinite(final_gradient).all() and torch.count_nonzero(final_gradient).item() > 0
    )
    color_losses_exact_zero = all(
        item["total_loss"] == 0.0
        and item["candidate_hinge_loss"] == 0.0
        and item["full_vocab_hinge_loss"] == 0.0
        and item["weighted_language_nll"] == 0.0
        for item in microstep_losses
        if item["pair_id"] == COLOR_PAIR_ID
    )
    color_gradient_exact_zero = bool(color_gradient_contributions) and all(
        item["exact_zero"] for item in color_gradient_contributions
    )
    mirror_gradient_nonzero = bool(mirror_gradient_contributions) and any(
        not item["exact_zero"] and item["finite"] for item in mirror_gradient_contributions
    )

    simulated_weight, clone_evidence = exact_clone_adamw_evidence(
        signed_residual, contract["optimizer"]
    )
    raw_scene_metrics: dict[str, dict[str, Any]] = {}
    effective_scene_metrics: dict[str, dict[str, Any]] = {}
    bf16_audits: dict[str, dict[str, Any]] = {}
    raw_deltas: dict[str, torch.Tensor] = {}
    effective_deltas: dict[str, torch.Tensor] = {}
    bf16_base_tokens: dict[str, torch.Tensor] = {}
    model_dtype = language.model.get_input_embeddings().weight.dtype
    if model_dtype is not torch.bfloat16:
        _fail(f"V20 expected Gemma bfloat16, observed {model_dtype}")
    with torch.no_grad():
        for scene_id in EXPECTED_SCENE_IDS:
            base_tokens = base_outputs[scene_id].scene_tokens
            raw_delta, effective_delta = functional_local_field_delta(
                signed_residual,
                centered_content[scene_id],
                simulated_weight,
                base_tokens=base_tokens,
                model_dtype=model_dtype,
            )
            bf16_base = base_tokens.float().to(dtype=model_dtype).float()
            raw_deltas[scene_id] = raw_delta
            effective_deltas[scene_id] = effective_delta
            bf16_base_tokens[scene_id] = bf16_base
            bf16_audits[scene_id] = bf16_cast_audit(base_tokens, raw_delta, effective_delta)
            raw_scene_metrics[scene_id] = {
                **fp64_delta_metrics(base_tokens, raw_delta),
                "delta_sha256": tensor_state_sha256({"raw_fp32_delta": raw_delta}),
                "dtype": str(raw_delta.dtype).removeprefix("torch."),
            }
            effective_scene_metrics[scene_id] = {
                **fp64_delta_metrics(bf16_base, effective_delta),
                "delta_sha256": tensor_state_sha256(
                    {"bf16_effective_delta_float32": effective_delta}
                ),
                "dtype": "bfloat16_round_trip_float32_delta",
            }
    scene_pair_by_id = {
        pair_id: (first_scene, second_scene)
        for pair_id, first_scene, second_scene in training_pairs
    }
    raw_pair_metrics = {
        pair_id: {
            "first_scene_id": scene_pair_by_id[pair_id][0],
            "second_scene_id": scene_pair_by_id[pair_id][1],
            **fp64_pair_delta_metrics(
                base_outputs[scene_pair_by_id[pair_id][0]].scene_tokens,
                base_outputs[scene_pair_by_id[pair_id][1]].scene_tokens,
                raw_deltas[scene_pair_by_id[pair_id][0]],
                raw_deltas[scene_pair_by_id[pair_id][1]],
            ),
        }
        for pair_id in (COLOR_PAIR_ID, MIRROR_PAIR_ID)
    }
    effective_pair_metrics = {
        pair_id: {
            "first_scene_id": scene_pair_by_id[pair_id][0],
            "second_scene_id": scene_pair_by_id[pair_id][1],
            **fp64_pair_delta_metrics(
                bf16_base_tokens[scene_pair_by_id[pair_id][0]],
                bf16_base_tokens[scene_pair_by_id[pair_id][1]],
                effective_deltas[scene_pair_by_id[pair_id][0]],
                effective_deltas[scene_pair_by_id[pair_id][1]],
            ),
        }
        for pair_id in (COLOR_PAIR_ID, MIRROR_PAIR_ID)
    }
    structural_gate = evaluate_v20_structural_gate(
        raw_scene_metrics,
        effective_scene_metrics,
        raw_pair_metrics=raw_pair_metrics,
        effective_pair_metrics=effective_pair_metrics,
        bf16_audits=bf16_audits,
        structural_state=structural_state,
        local_dependence=local_dependence,
        local_hidden_ranks=local_hidden_ranks,
        requirements=contract["structural_preflight_requires"],
    )

    rng_after = capture_rng_states(require_mps=language.device.type == "mps")
    rng_evidence = rng_state_evidence(rng_before, rng_after)
    if not rng_evidence["all_available_domains_unchanged"]:
        restore_rng_states(rng_before)
        rng_evidence["restored_after_mismatch"] = True
    else:
        rng_evidence["restored_after_mismatch"] = False
    signed_residual.zero_grad(set_to_none=True)
    live_source_state_after = module_collection_state_sha256(source_modules)
    live_signed_state_after = module_collection_state_sha256(
        {"signed_x_scene_residual": signed_residual}
    )
    live_source_state_unchanged = live_source_state_before == live_source_state_after
    live_signed_state_unchanged = live_signed_state_before == live_signed_state_after
    predicted_update_valid = bool(
        clone_evidence["finite_update"]
        and clone_evidence["nonzero_update_count"] > 0
        and clone_evidence["update_l2_norm"] > 0.0
    )
    authorization_checks = {
        "source_and_config_contracts_passed": True,
        "exact_selection_and_order_passed": True,
        "step_zero_identity_all_scenes": bool(zero_equivalence["verified"]),
        "color_losses_exactly_zero": color_losses_exact_zero,
        "color_isolated_signed_x_gradient_exactly_zero": color_gradient_exact_zero,
        "mirror_signed_x_gradient_finite_nonzero": mirror_gradient_nonzero,
        "accumulated_signed_x_gradient_finite_nonzero": gradient_finite_nonzero,
        "only_signed_x_output_weight_has_gradient": only_signed_output_has_gradient,
        "predicted_adamw_update_finite_nonzero": predicted_update_valid,
        "local_field_rank_bf16_selectivity_gate": bool(structural_gate["passed"]),
        "live_source_state_unchanged": live_source_state_unchanged,
        "live_signed_x_state_unchanged": live_signed_state_unchanged,
        "rng_state_unchanged": bool(rng_evidence["all_available_domains_unchanged"]),
    }
    authorization = all(authorization_checks.values())

    implementation_path = Path(__file__).resolve()
    signed_implementation_path = Path(
        __import__(
            "semantic_3d_chat.scene_encoder.signed_x_local_field",
            fromlist=["__file__"],
        ).__file__
    ).resolve()
    dispatch_implementation_path = Path(
        __import__(
            "semantic_3d_chat.scene_encoder.signed_x_dispatch",
            fromlist=["__file__"],
        ).__file__
    ).resolve()
    frozen_state_hashes = {
        "scene_state_sha256": observed_source_hashes["scene_state_sha256"],
        "global_scene_residual_state_sha256": observed_source_hashes[
            "global_scene_residual_state_sha256"
        ],
        "lora_bank_state_sha256": observed_source_hashes["lora_bank_state_sha256"],
        "combined_source_state_sha256": live_source_state_before,
    }
    pair_gradient_audit = {
        "color_pair_id": COLOR_PAIR_ID,
        "mirror_pair_id": MIRROR_PAIR_ID,
        "color_total_loss_exact_zero": color_losses_exact_zero,
        "color_gradient_exact_zero": color_gradient_exact_zero,
        "mirror_gradient_positive_finite": mirror_gradient_nonzero,
        "color_losses_exact_zero": color_losses_exact_zero,
        "color_isolated_signed_x_gradient_exact_zero": color_gradient_exact_zero,
        "mirror_signed_x_gradient_finite_nonzero": mirror_gradient_nonzero,
        "only_signed_x_output_weight_has_gradient": only_signed_output_has_gradient,
        "color_contributions": color_gradient_contributions,
        "mirror_contributions": mirror_gradient_contributions,
    }
    gradient_evidence = {
        **gradient_manifest,
        "ordered_microstep_count": len(curriculum),
        "accumulated_finite_nonzero": gradient_finite_nonzero,
        "accumulated_gradient_l2_norm": float(final_gradient.float().norm().cpu()),
        "accumulated_gradient_sha256": tensor_state_sha256(
            {"signed_x_scene_residual.output_projection.weight": final_gradient}
        ),
        "unclipped_gradient_sha256": clone_evidence["gradient_sha256"],
        "clipped_gradient_sha256": clone_evidence["clipped_gradient_sha256"],
        "pre_clip_gradient_l2_norm": clone_evidence["pre_clip_gradient_l2_norm"],
        "post_clip_gradient_l2_norm": clone_evidence["post_clip_gradient_l2_norm"],
        "predicted_update_l2_norm": clone_evidence["update_l2_norm"],
        "predicted_update_rms": clone_evidence["update_rms"],
        "predicted_update_nonzero_count": clone_evidence["nonzero_update_count"],
        "predicted_signed_x_state_sha256": clone_evidence[
            "predicted_signed_x_scene_residual_state_sha256"
        ],
        "predicted_output_projection_sha256": clone_evidence["predicted_output_weight_sha256"],
        "optimizer_state_manifest": clone_evidence["canonical_adamw_state_manifest"],
        "optimizer_state_sha256": clone_evidence["canonical_adamw_state_sha256"],
        "optimizer_state_tensor_sha256": clone_evidence["optimizer_state_tensor_sha256"],
        "changed_parameter_keys": clone_evidence["changed_parameter_keys"],
    }
    report = {
        "schema_version": 1,
        "audit_type": V20_PREFLIGHT_ROLE,
        "runtime_eligible": False,
        "uses_supervised_qa_metadata": True,
        "question_dependent_scene_processing": False,
        "live_optimizer_constructed": False,
        "live_optimizer_step_executed": False,
        "optimizer_steps": 0,
        "isolated_clone_optimizer_constructed": True,
        "isolated_clone_optimizer_steps": 1,
        "authorized": authorization,
        "structural_authorization": authorization,
        "authorization_checks": authorization_checks,
        "config_path": str(Path(config["_config_path"]).resolve().relative_to(PROJECT_ROOT)),
        "config_hash": config_hash(config, length=64),
        "contract": contract,
        "adamw_contract": contract["optimizer"],
        "source_provenance": source_provenance,
        "implementation_source": str(implementation_path.relative_to(PROJECT_ROOT)),
        "implementation_source_sha256": file_sha256(implementation_path),
        "signed_x_implementation_source": str(signed_implementation_path.relative_to(PROJECT_ROOT)),
        "signed_x_implementation_source_sha256": file_sha256(signed_implementation_path),
        "signed_x_dispatch_implementation_source": str(
            dispatch_implementation_path.relative_to(PROJECT_ROOT)
        ),
        "signed_x_dispatch_implementation_source_sha256": file_sha256(dispatch_implementation_path),
        "source_checkpoint": str(source.relative_to(PROJECT_ROOT)),
        "source_checkpoint_epoch": source_metadata.get("epoch"),
        "source_artifact_hashes": source_artifact_hashes,
        "frozen_state_hashes": frozen_state_hashes,
        "source_hashes": observed_source_hashes,
        "source_metadata_global_residual_state_sha256": source_metadata.get(
            "global_scene_residual_state_sha256"
        ),
        "source_metadata_lora_bank_state_sha256": source_metadata.get("lora_bank_state_sha256"),
        "initial_signed_x_state_sha256": initial_signed_hash,
        "live_source_state_sha256_before": live_source_state_before,
        "live_source_state_sha256_after": live_source_state_after,
        "live_source_state_unchanged": live_source_state_unchanged,
        "live_signed_x_state_sha256_before": live_signed_state_before,
        "live_signed_x_state_sha256_after": live_signed_state_after,
        "live_signed_x_state_unchanged": live_signed_state_unchanged,
        "live_parameter_state_unchanged": (
            live_source_state_unchanged and live_signed_state_unchanged
        ),
        "selection_sha256": selection["selected_ids_sha256"],
        "pair_membership_sha256": pair_membership_hash,
        "pair_unit_selection_sha256": pair_unit_selection_hash,
        "selected_pair_units": selected_pair_units,
        "ordered_unit_sha256": ordered_hash,
        "ordered_units": ordered_units,
        "pair_objective_policy": pair_objective_policy_contract(pair_policies),
        "pair_objective_policy_coverage": policy_coverage,
        "zero_output_prefix_equivalence": zero_equivalence,
        "signed_x_structural_state": structural_state,
        "local_field_structural_state": structural_state,
        "local_dependence": local_dependence,
        "local_hidden_spatial_rank": local_hidden_ranks,
        "centered_content": centered_content_evidence,
        "microsteps": microstep_losses,
        "microstep_losses": microstep_losses,
        "pair_gradient_audit": pair_gradient_audit,
        "gradient": gradient_evidence,
        "predicted_first_update": clone_evidence,
        # Repeated at top level to make a later exact stage-one verifier simple
        # and independent of report-layout traversal.
        "predicted_output_weight_sha256": clone_evidence["predicted_output_weight_sha256"],
        "predicted_signed_x_scene_residual_state_sha256": clone_evidence[
            "predicted_signed_x_scene_residual_state_sha256"
        ],
        "predicted_canonical_adamw_state_sha256": clone_evidence["canonical_adamw_state_sha256"],
        "predicted_canonical_adamw_state_manifest": clone_evidence[
            "canonical_adamw_state_manifest"
        ],
        "raw_fp32_centered_scene_delta": raw_scene_metrics,
        "bf16_cast_audit": bf16_audits,
        "bf16_effective_scene_delta": effective_scene_metrics,
        "effective_cast_scene_delta": effective_scene_metrics,
        "raw_fp32_centered_pair_delta": raw_pair_metrics,
        "bf16_effective_pair_delta": effective_pair_metrics,
        "effective_cast_pair_delta": effective_pair_metrics,
        "structural_gate": structural_gate,
        "rng_state": rng_evidence,
    }
    destination = Path(report_path)
    if not destination.is_absolute():
        destination = PROJECT_ROOT / destination
    atomic_write_json(destination, report)
    if not authorization:
        raise V20StructuralPreflightViolation(
            f"V20 structural preflight failed; evidence written to {destination}"
        )
    print(json.dumps({"phase": "v20_structural_preflight_passed", "report": str(destination)}))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--report", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_preflight(args.config, args.report)


if __name__ == "__main__":  # pragma: no cover - local model command
    main()


__all__ = [
    "COLOR_PAIR_ID",
    "EXPECTED_PAIR_UNIT_SELECTION_SHA256",
    "EXPECTED_RESOLVED_CONFIG_HASH",
    "MIRROR_PAIR_ID",
    "SIGNED_X_OPTIMIZER_GROUP_NAME",
    "V20_PREFLIGHT_ROLE",
    "V20StructuralPreflightViolation",
    "atomic_write_json",
    "bf16_cast_audit",
    "canonical_sha256",
    "evaluate_v20_structural_gate",
    "exact_clone_adamw_evidence",
    "functional_local_field_delta",
    "local_dependence_evidence",
    "normalized_pair_selectivity",
    "ordered_curriculum_evidence",
    "pair_unit_selection_evidence",
    "run_preflight",
    "signed_x_residual_state_sha256",
    "spatial_rank_evidence",
    "validate_v20_config_contract",
]
