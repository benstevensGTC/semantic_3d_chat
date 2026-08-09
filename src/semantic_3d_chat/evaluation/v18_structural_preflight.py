"""Strict, no-step structural preflight for the V18 scene residual.

This is a supervised training diagnostic, not an inference component.  It
replays the exact ordered epoch-one pair curriculum, accumulates the same first
gradient as training, and predicts the first AdamW output-weight update by
stepping an isolated full-module clone.  It never constructs or steps an
optimizer over the live module and fails closed before authorizing the
four-update V18 screen.
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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.func import functional_call

from semantic_3d_chat.evaluation.v18_optimizer_state import (
    V18_RESIDUAL_OPTIMIZER_GROUP_NAME,
    canonical_v18_adamw_state,
)
from semantic_3d_chat.scene_encoder.global_residual import (
    ZERO_SPATIAL_MEAN_CONTENT_GATE_V1,
)

STRUCTURAL_PREFLIGHT_ROLE = "v18_exact_ordered_epoch1_structural_preflight"
V18_SCREEN_ROLE = "v18_slot_centered_residual_screen"
COLOR_PAIR_ID = "pair_000001"
MIRROR_PAIR_ID = "pair_000003"
EXPECTED_RANKING_FIELDS = (
    "mirror_full_vocab_units",
    "mirror_full_vocab_sides",
    "mirror_candidate_units",
    "mirror_candidate_sides",
    "mirror_mean_full_vocab_margin",
    "mirror_minimum_full_vocab_margin",
)
EXPECTED_ELIGIBILITY = {
    "color_full_vocab_sides": 12,
    "color_full_vocab_units": 6,
    "color_positive_minimum_candidate_margin": True,
    "color_positive_minimum_full_vocab_margin": True,
}
EXPECTED_CONTINUATION = {
    **EXPECTED_ELIGIBILITY,
    "mirror_minimum_full_vocab_sides": 8,
    "mirror_minimum_full_vocab_units": 2,
}
EXPECTED_FULL_TEACHER_GATE = {
    "color_full_vocab_sides": 12,
    "color_full_vocab_units": 6,
    "mirror_full_vocab_sides": 12,
    "mirror_full_vocab_units": 6,
    "all_candidate_and_full_vocab_minimum_margins_positive": True,
}
_HEX_SHA256 = re.compile(r"[0-9a-f]{64}")


class V18StructuralPreflightViolation(ValueError):
    """A fail-closed V18 configuration or evidence violation."""


def _fail(message: str) -> None:
    raise V18StructuralPreflightViolation(message)


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{field} must be an object")
    return value


def _sequence(value: Any, field: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        _fail(f"{field} must be a sequence")
    return value


def _strict_keys(value: Mapping[str, Any], expected: set[str], field: str) -> None:
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing or unknown:
        _fail(f"{field} keys mismatch: missing={missing} unknown={unknown}")


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        _fail(f"{field} must be a positive integer")
    return value


def _finite(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        _fail(f"{field} must be finite")
    return result


def _sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or _HEX_SHA256.fullmatch(value) is None:
        _fail(f"{field} must be a lowercase SHA-256 digest")
    return value


def canonical_sha256(value: Any) -> str:
    """Hash JSON-compatible evidence with one canonical encoding."""

    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class StructuralThresholds:
    """Predeclared FP64 structural launch bounds."""

    maximum_raw_mean_energy_fraction: float = 1.0e-6
    minimum_raw_slot_varying_energy_fraction: float = 0.999999
    maximum_effective_mean_energy_fraction: float = 1.0e-3
    minimum_effective_slot_varying_energy_fraction: float = 0.999
    maximum_effective_delta_to_core_rms_ratio: float = 0.05
    require_positive_finite_raw_total_energy: bool = True
    require_positive_finite_effective_total_energy: bool = True
    require_positive_finite_pair_delta: bool = True

    def __post_init__(self) -> None:
        for name in (
            "maximum_raw_mean_energy_fraction",
            "minimum_raw_slot_varying_energy_fraction",
            "maximum_effective_mean_energy_fraction",
            "minimum_effective_slot_varying_energy_fraction",
            "maximum_effective_delta_to_core_rms_ratio",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in (
            "maximum_raw_mean_energy_fraction",
            "minimum_raw_slot_varying_energy_fraction",
            "maximum_effective_mean_energy_fraction",
            "minimum_effective_slot_varying_energy_fraction",
        ):
            if getattr(self, name) > 1.0:
                raise ValueError(f"{name} must be at most one")
        for name in (
            "require_positive_finite_raw_total_energy",
            "require_positive_finite_effective_total_energy",
            "require_positive_finite_pair_delta",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a boolean")


def _validate_thresholds(raw: Mapping[str, Any]) -> StructuralThresholds:
    expected = set(StructuralThresholds.__dataclass_fields__)
    _strict_keys(raw, expected, "structural_preflight.thresholds")
    thresholds = StructuralThresholds(**raw)
    pinned = StructuralThresholds()
    if thresholds != pinned:
        _fail(
            "V18 structural thresholds differ from the pinned raw/effective BF16 contract: "
            f"expected={asdict(pinned)} observed={asdict(thresholds)}"
        )
    return thresholds


def _validate_optimizer_contract(raw: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
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
    _strict_keys(raw, expected, "structural_preflight.optimizer")
    normalized = {
        "name": raw["name"],
        "learning_rate": _finite(raw["learning_rate"], "optimizer.learning_rate"),
        "betas": list(_sequence(raw["betas"], "optimizer.betas")),
        "epsilon": _finite(raw["epsilon"], "optimizer.epsilon"),
        "weight_decay": _finite(raw["weight_decay"], "optimizer.weight_decay"),
        "foreach": raw["foreach"],
        "fused": raw["fused"],
        "capturable": raw["capturable"],
        "maximize": raw["maximize"],
        "amsgrad": raw["amsgrad"],
        "gradient_clip_norm": _finite(raw["gradient_clip_norm"], "optimizer.gradient_clip_norm"),
        "accumulation_divisor": _positive_int(
            raw["accumulation_divisor"], "optimizer.accumulation_divisor"
        ),
        "step_index": _positive_int(raw["step_index"], "optimizer.step_index"),
    }
    pinned = {
        "name": "AdamW",
        "learning_rate": 1.0e-3,
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
    if normalized != pinned:
        _fail(f"V18 AdamW contract mismatch: expected={pinned} observed={normalized}")
    return normalized


def _validate_expected_hashes(raw: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "ordered_unit_sha256",
        "source_adapter_sha256",
        "source_metadata_sha256",
        "frozen_scene_state_sha256",
        "frozen_lora_bank_state_sha256",
        "initial_residual_state_sha256",
        "position_features_sha256",
        "selection_sha256",
        "pair_membership_sha256",
        "core_prefix_sha256",
        "v16_gradient_audit_sha256",
        "v17_lr_response_sha256",
    }
    _strict_keys(raw, expected, "structural_preflight.expected_hashes")
    normalized: dict[str, Any] = {}
    scalar_fields = expected - {"frozen_lora_bank_state_sha256", "core_prefix_sha256"}
    for field in scalar_fields:
        normalized[field] = _sha256(raw[field], f"expected_hashes.{field}")
    banks = _mapping(
        raw["frozen_lora_bank_state_sha256"],
        "expected_hashes.frozen_lora_bank_state_sha256",
    )
    _strict_keys(banks, {"inherited_v12", "extension_v13"}, "expected frozen banks")
    normalized["frozen_lora_bank_state_sha256"] = {
        name: _sha256(value, f"expected frozen bank {name}")
        for name, value in sorted(banks.items())
    }
    prefixes = _mapping(raw["core_prefix_sha256"], "expected_hashes.core_prefix_sha256")
    if len(prefixes) != 4:
        _fail("expected_hashes.core_prefix_sha256 must contain exactly four scenes")
    normalized["core_prefix_sha256"] = {
        str(scene_id): _sha256(value, f"core prefix {scene_id}")
        for scene_id, value in sorted(prefixes.items())
    }
    return normalized


def _validate_screen_contract(raw: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "schema_version",
        "role",
        "learning_rate",
        "screen_optimizer_updates",
        "conditional_max_optimizer_updates",
        "epoch_tiebreaker",
        "execution_stages",
        "eligibility_requires",
        "ranking_descending",
        "continuation_requires",
        "full_teacher_gate_requires",
        "greedy_audit_only_after_full_teacher_gate",
    }
    _strict_keys(raw, expected, "v18_screen")
    if raw["schema_version"] != 1 or raw["role"] != V18_SCREEN_ROLE:
        _fail("v18_screen schema_version or role mismatch")
    observed_ranking = tuple(_sequence(raw["ranking_descending"], "v18_screen.ranking"))
    stages = _mapping(raw["execution_stages"], "v18_screen.execution_stages")
    _strict_keys(
        stages,
        {
            "stage_1_exact_v14_restart_updates",
            "stage_1_stop_required",
            "predicted_preflight_state_must_match_epoch_001",
            "stage_2_resume_from_epoch",
            "stage_2_load_optimizer_state",
            "stage_2_load_history",
            "stage_2_target_total_optimizer_updates",
        },
        "v18_screen.execution_stages",
    )
    normalized = {
        **dict(raw),
        "learning_rate": _finite(raw["learning_rate"], "v18_screen.learning_rate"),
        "screen_optimizer_updates": _positive_int(
            raw["screen_optimizer_updates"], "v18_screen.screen_optimizer_updates"
        ),
        "conditional_max_optimizer_updates": _positive_int(
            raw["conditional_max_optimizer_updates"],
            "v18_screen.conditional_max_optimizer_updates",
        ),
        "ranking_descending": list(observed_ranking),
        "execution_stages": dict(stages),
    }
    checks = {
        "learning_rate": normalized["learning_rate"] == 1.0e-3,
        "screen_optimizer_updates": normalized["screen_optimizer_updates"] == 4,
        "conditional_max_optimizer_updates": (
            normalized["conditional_max_optimizer_updates"] == 12
        ),
        "epoch_tiebreaker": raw["epoch_tiebreaker"] == "lower_epoch",
        "execution_stages": stages
        == {
            "stage_1_exact_v14_restart_updates": 1,
            "stage_1_stop_required": True,
            "predicted_preflight_state_must_match_epoch_001": True,
            "stage_2_resume_from_epoch": 1,
            "stage_2_load_optimizer_state": True,
            "stage_2_load_history": True,
            "stage_2_target_total_optimizer_updates": 4,
        },
        "eligibility_requires": raw["eligibility_requires"] == EXPECTED_ELIGIBILITY,
        "ranking_descending": observed_ranking == EXPECTED_RANKING_FIELDS,
        "continuation_requires": raw["continuation_requires"] == EXPECTED_CONTINUATION,
        "full_teacher_gate_requires": (
            raw["full_teacher_gate_requires"] == EXPECTED_FULL_TEACHER_GATE
        ),
        "greedy_audit_only_after_full_teacher_gate": (
            raw["greedy_audit_only_after_full_teacher_gate"] is True
        ),
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        _fail(f"v18_screen differs from the pinned V17-derived policy: {failed}")
    return normalized


def validate_v18_config_contract(
    config: Mapping[str, Any], *, implementation_source_sha256: str | None = None
) -> dict[str, Any]:
    """Validate and normalize the complete V18 launch contract."""

    raw = _mapping(config.get("structural_preflight"), "structural_preflight")
    expected = {
        "schema_version",
        "required",
        "role",
        "architecture_version",
        "spatial_centering",
        "content_gate",
        "implementation_source_sha256",
        "source_must_be_clean",
        "latent_count",
        "scene_dim",
        "residual_parameter_count",
        "exact_epoch",
        "microsteps",
        "optimizer",
        "thresholds",
        "expected_hashes",
        "evidence_paths",
    }
    _strict_keys(raw, expected, "structural_preflight")
    if raw["schema_version"] != 1:
        _fail("structural_preflight.schema_version must equal 1")
    if raw["required"] is not True or raw["role"] != STRUCTURAL_PREFLIGHT_ROLE:
        _fail("structural_preflight must be required and use the pinned V18 role")
    architecture_checks = {
        "architecture_version": raw["architecture_version"] == ZERO_SPATIAL_MEAN_CONTENT_GATE_V1,
        "spatial_centering": raw["spatial_centering"] == "all_slots_fp32",
        "content_gate": raw["content_gate"] == "bias_free_scalar_sigmoid_centered_content",
    }
    failed_architecture = sorted(name for name, passed in architecture_checks.items() if not passed)
    if failed_architecture:
        _fail(f"V18 structural architecture mismatch: {failed_architecture}")
    implementation_hash = _sha256(
        raw["implementation_source_sha256"], "implementation_source_sha256"
    )
    if (
        implementation_source_sha256 is not None
        and implementation_hash != implementation_source_sha256
    ):
        _fail(
            "V18 implementation source hash mismatch: "
            f"expected={implementation_hash} observed={implementation_source_sha256}"
        )
    if raw["source_must_be_clean"] is not True:
        _fail("V18 requires a clean tracked source tree")
    for field, expected_value in {
        "latent_count": 256,
        "scene_dim": 1536,
        "residual_parameter_count": 400_128,
    }.items():
        if raw[field] != expected_value:
            _fail(f"structural_preflight.{field} must equal {expected_value}")
    if raw["exact_epoch"] != 1 or raw["microsteps"] != 12:
        _fail("V18 preflight must reproduce exactly epoch 1 and 12 microsteps")
    optimizer = _validate_optimizer_contract(
        _mapping(raw["optimizer"], "structural_preflight.optimizer")
    )
    thresholds = _validate_thresholds(
        _mapping(raw["thresholds"], "structural_preflight.thresholds")
    )
    hashes = _validate_expected_hashes(
        _mapping(raw["expected_hashes"], "structural_preflight.expected_hashes")
    )
    paths = _mapping(raw["evidence_paths"], "structural_preflight.evidence_paths")
    _strict_keys(paths, {"v16_gradient_audit", "v17_lr_response"}, "evidence_paths")
    for name, value in paths.items():
        if not isinstance(value, str) or not value:
            _fail(f"evidence_paths.{name} must be a nonempty path string")

    residual = _mapping(
        _mapping(config.get("scene_encoder"), "scene_encoder").get("global_scene_residual"),
        "scene_encoder.global_scene_residual",
    )
    if residual.get("enabled") is not True:
        _fail("V18 requires an enabled global scene residual")
    for field, expected_value in {
        "architecture_version": ZERO_SPATIAL_MEAN_CONTENT_GATE_V1,
        "spatial_centering": None,
    }.items():
        if field == "spatial_centering":
            continue
        if residual.get(field) != expected_value:
            _fail(f"scene_encoder.global_scene_residual.{field} mismatch")
    if residual.get("expected_initial_state_sha256") != hashes["initial_residual_state_sha256"]:
        _fail("Residual initial-state hash disagrees with the preflight contract")
    scene_encoder = _mapping(config.get("scene_encoder"), "scene_encoder")
    if scene_encoder.get("global_latents") != 256:
        _fail("scene_encoder.global_latents must equal the pinned 256 V18 slots")

    training = _mapping(config.get("training"), "training")
    language = _mapping(config.get("language"), "language")
    system_prompt = language.get("system_prompt")
    if not isinstance(system_prompt, str):
        _fail("language.system_prompt must be a string")
    training_objective = {
        "batch_size": training.get("batch_size"),
        "max_questions_per_scene": training.get("max_questions_per_scene"),
        "language_decoder_gradient_checkpointing": training.get(
            "language_decoder_gradient_checkpointing"
        ),
        "initialize_legacy_lora_into_bank": training.get(
            "initialize_legacy_lora_into_bank"
        ),
        "initialize_named_lora_freeze_transition": training.get(
            "initialize_named_lora_freeze_transition"
        ),
        "pair_only_scene_ids": training.get("pair_only_scene_ids"),
        "pair_ranking_weight": training.get("pair_ranking_weight"),
        "pair_ranking_margin": training.get("pair_ranking_margin"),
        "pair_ranking_mode": training.get("pair_ranking_mode"),
        "pair_full_vocab_ranking_weight": training.get(
            "pair_full_vocab_ranking_weight"
        ),
        "pair_full_vocab_ranking_margin": training.get(
            "pair_full_vocab_ranking_margin"
        ),
        "grounding_weight": training.get("grounding_weight"),
        "grounding_anchor_weight": training.get("grounding_anchor_weight"),
        "latent_diversity_weight": training.get("latent_diversity_weight"),
        "paired_scene_separation_weight": training.get(
            "paired_scene_separation_weight"
        ),
        "spatial_answer_contrastive_weight": training.get(
            "spatial_answer_contrastive_weight"
        ),
        "spatial_answer_warmup_steps": training.get("spatial_answer_warmup_steps"),
        "spatial_relation_contrastive_weight": training.get(
            "spatial_relation_contrastive_weight"
        ),
        "spatial_relation_warmup_steps": training.get(
            "spatial_relation_warmup_steps"
        ),
        "pair_gate_enabled": training.get("pair_gate_enabled"),
        "pair_gate_every_epochs": training.get("pair_gate_every_epochs"),
        "pair_gate_changed_unit_accuracy": training.get(
            "pair_gate_changed_unit_accuracy"
        ),
        "pair_gate_prediction_flip_rate": training.get(
            "pair_gate_prediction_flip_rate"
        ),
        "pair_gate_wrong_prefix_flip_rate": training.get(
            "pair_gate_wrong_prefix_flip_rate"
        ),
        "pair_gate_first_answer_token_top1_accuracy": training.get(
            "pair_gate_first_answer_token_top1_accuracy"
        ),
        "language_model_id": language.get("model_id"),
        "language_revision": language.get("revision"),
        "language_backend": language.get("backend"),
        "language_dtype": language.get("dtype"),
        "scene_prefix_after_bos": language.get("scene_prefix_after_bos"),
        "scene_boundary_mode": language.get("scene_boundary_mode"),
        "system_prompt_sha256": hashlib.sha256(system_prompt.encode("utf-8")).hexdigest(),
    }
    expected_training_objective = {
        "batch_size": 2,
        "max_questions_per_scene": 6,
        "language_decoder_gradient_checkpointing": True,
        "initialize_legacy_lora_into_bank": None,
        "initialize_named_lora_freeze_transition": True,
        "pair_only_scene_ids": [
            "scene_000003",
            "scene_000004",
            "scene_000007",
            "scene_000008",
        ],
        "pair_ranking_weight": 8.0,
        "pair_ranking_margin": 1.0,
        "pair_ranking_mode": "candidate_logit",
        "pair_full_vocab_ranking_weight": 2.0,
        "pair_full_vocab_ranking_margin": 1.0,
        "grounding_weight": 0.0,
        "grounding_anchor_weight": 0.0,
        "latent_diversity_weight": 0.0,
        "paired_scene_separation_weight": 0.0,
        "spatial_answer_contrastive_weight": 0.0,
        "spatial_answer_warmup_steps": 0,
        "spatial_relation_contrastive_weight": 0.0,
        "spatial_relation_warmup_steps": 0,
        "pair_gate_enabled": True,
        "pair_gate_every_epochs": 1,
        "pair_gate_changed_unit_accuracy": 0.95,
        "pair_gate_prediction_flip_rate": 1.0,
        "pair_gate_wrong_prefix_flip_rate": 1.0,
        "pair_gate_first_answer_token_top1_accuracy": 1.0,
        "language_model_id": "google/gemma-4-E2B-it",
        "language_revision": "3e22461f65e89153144f8adb70e3b8c2cc9845a7",
        "language_backend": "gemma4",
        "language_dtype": "bfloat16",
        "scene_prefix_after_bos": True,
        "scene_boundary_mode": "gemma4_native_image",
        "system_prompt_sha256": (
            "c961a411aa0626c0a8d8d8e103d80c8d790b21cfe33a887ea5cae9f4a4e10afe"
        ),
    }
    if training_objective != expected_training_objective:
        failed_objective = sorted(
            key
            for key, expected_value in expected_training_objective.items()
            if training_objective.get(key) != expected_value
        )
        _fail(f"V18 gradient-defining objective contract mismatch: {failed_objective}")
    training_checks = {
        "train_global_scene_residual_only": training.get("train_global_scene_residual_only")
        is True,
        "freeze_scene_adapter": training.get("freeze_scene_adapter") is True,
        "learning_rate": float(training.get("learning_rate", math.nan)) == 1.0e-3,
        "weight_decay": float(training.get("weight_decay", math.nan)) == 0.0,
        "gradient_clip_norm": float(training.get("gradient_clip_norm", math.nan)) == 1.0,
        "gradient_accumulation": training.get("gradient_accumulation") == 12,
        "pair_steps_per_epoch": training.get("pair_steps_per_epoch") == 12,
        "epochs": training.get("epochs") == 4,
        "pair_only_mode": training.get("pair_only_mode") is True,
        "pair_batch_fraction": float(training.get("pair_batch_fraction", math.nan)) == 1.0,
        "pair_units_per_batch": training.get("pair_units_per_batch") == 1,
        "pair_max_units_per_pair": training.get("pair_max_units_per_pair") == 6,
        "pair_gate_stop_when_passed": training.get("pair_gate_stop_when_passed") is False,
        "early_stopping_patience": training.get("early_stopping_patience") == 0,
        "initialize_expected_adapter_sha256": training.get("initialize_expected_adapter_sha256")
        == hashes["source_adapter_sha256"],
        "initialize_expected_metadata_sha256": training.get("initialize_expected_metadata_sha256")
        == hashes["source_metadata_sha256"],
        "optimizer": training.get("optimizer") == optimizer,
    }
    failed_training = sorted(name for name, passed in training_checks.items() if not passed)
    if failed_training:
        _fail(f"V18 training contract mismatch: {failed_training}")
    experiment = _mapping(config.get("experiment"), "experiment")
    if experiment.get("residual_parameter_count") != 400_128:
        _fail("experiment.residual_parameter_count must equal 400128")

    screen = _validate_screen_contract(_mapping(config.get("v18_screen"), "v18_screen"))
    if screen["learning_rate"] != optimizer["learning_rate"]:
        _fail("V18 screen and AdamW learning rates disagree")
    normalized = {
        "schema_version": 1,
        "required": True,
        "role": STRUCTURAL_PREFLIGHT_ROLE,
        "architecture_version": ZERO_SPATIAL_MEAN_CONTENT_GATE_V1,
        "spatial_centering": "all_slots_fp32",
        "content_gate": "bias_free_scalar_sigmoid_centered_content",
        "implementation_source_sha256": implementation_hash,
        "source_must_be_clean": True,
        "latent_count": 256,
        "scene_dim": 1536,
        "residual_parameter_count": 400_128,
        "exact_epoch": 1,
        "microsteps": 12,
        "optimizer": optimizer,
        "thresholds": asdict(thresholds),
        "expected_hashes": hashes,
        "evidence_paths": dict(paths),
        "training_objective": training_objective,
        "v18_screen": screen,
    }
    normalized["contract_sha256"] = canonical_sha256(normalized)
    return normalized


def ordered_curriculum_evidence(curriculum: Sequence[Any]) -> tuple[list[dict[str, Any]], str]:
    """Return opaque epoch-one microstep identities and their canonical hash."""

    entries: list[dict[str, Any]] = []
    for microstep, batch in enumerate(curriculum, start=1):
        if getattr(batch, "kind", None) != "pair":
            _fail(f"V18 microstep {microstep} is not a pair batch")
        units = tuple(getattr(batch, "pair_units", ()))
        if len(units) != 1:
            _fail(f"V18 microstep {microstep} must contain exactly one pair unit")
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


def fp64_delta_metrics(core: torch.Tensor, delta: torch.Tensor) -> dict[str, Any]:
    """Measure one scene's raw or effective residual with an FP64 decomposition."""

    if core.shape != delta.shape or core.ndim != 3 or core.shape[0] != 1 or core.shape[1] < 2:
        raise ValueError("Core and delta must have matching [1,L,H] shapes with L > 1")
    core64 = core.detach().to(device="cpu", dtype=torch.float64)
    delta64 = delta.detach().to(device="cpu", dtype=torch.float64)
    if not torch.isfinite(core64).all() or not torch.isfinite(delta64).all():
        raise ValueError("Core and delta must be finite")
    slots = int(delta64.shape[1])
    slot_mean = delta64.mean(dim=1, keepdim=True)
    slot_varying = delta64 - slot_mean
    total_energy = delta64.square().sum()
    mean_energy = float(slots) * slot_mean.square().sum()
    varying_energy = slot_varying.square().sum()
    core_rms = core64.square().mean().sqrt()
    delta_rms = delta64.square().mean().sqrt()
    positive_finite_total = bool(torch.isfinite(total_energy) and total_energy > 0.0)
    positive_finite_core = bool(torch.isfinite(core_rms) and core_rms > 0.0)
    mean_fraction = mean_energy / total_energy if positive_finite_total else torch.zeros(())
    varying_fraction = varying_energy / total_energy if positive_finite_total else torch.zeros(())
    ratio = delta_rms / core_rms if positive_finite_core else torch.full((), float("inf"))
    return {
        "shape": list(delta.shape),
        "core_rms": float(core_rms),
        "delta_rms": float(delta_rms),
        "delta_to_core_rms_ratio": float(ratio),
        "total_energy": float(total_energy),
        "across_slot_mean_energy": float(mean_energy),
        "slot_varying_energy": float(varying_energy),
        "across_slot_mean_energy_fraction": float(mean_fraction),
        "slot_varying_energy_fraction": float(varying_fraction),
        "slot_mean_absolute_maximum": float(slot_mean.abs().max()),
        "delta_absolute_maximum": float(delta64.abs().max()),
        "energy_closure_absolute_error": float((total_energy - mean_energy - varying_energy).abs()),
        "positive_finite_total_energy": positive_finite_total,
        "positive_finite_core_rms": positive_finite_core,
    }


def fp64_pair_delta_metrics(
    first_core: torch.Tensor,
    second_core: torch.Tensor,
    first_delta: torch.Tensor,
    second_delta: torch.Tensor,
) -> dict[str, Any]:
    """Measure whether a simulated residual differs across one scene pair."""

    shapes = {tuple(value.shape) for value in (first_core, second_core, first_delta, second_delta)}
    if len(shapes) != 1:
        raise ValueError("Pair tensors must all have the same shape")
    core_difference = (
        first_core.detach().to(torch.float64).cpu() - second_core.detach().to(torch.float64).cpu()
    )
    delta_difference = (
        first_delta.detach().to(torch.float64).cpu() - second_delta.detach().to(torch.float64).cpu()
    )
    if not torch.isfinite(core_difference).all() or not torch.isfinite(delta_difference).all():
        raise ValueError("Pair differences must be finite")
    core_rms = core_difference.square().mean().sqrt()
    delta_rms = delta_difference.square().mean().sqrt()
    core_norm = core_difference.norm()
    delta_norm = delta_difference.norm()
    valid_core = bool(core_norm > 0.0 and torch.isfinite(core_norm))
    positive_finite_delta = bool(delta_norm > 0.0 and torch.isfinite(delta_norm))
    cosine = (
        torch.dot(core_difference.reshape(-1), delta_difference.reshape(-1))
        / (core_norm * delta_norm)
        if valid_core and positive_finite_delta
        else torch.zeros(())
    )
    return {
        "core_pair_difference_rms": float(core_rms),
        "residual_pair_difference_rms": float(delta_rms),
        "residual_to_core_pair_difference_ratio": (
            float(delta_rms / core_rms) if valid_core else float("inf")
        ),
        "residual_core_difference_cosine": float(cosine),
        "positive_finite_pair_delta": positive_finite_delta,
        "positive_finite_core_difference": valid_core,
    }


def evaluate_structural_gate(
    raw_scene_metrics: Mapping[str, Mapping[str, Any]],
    effective_scene_metrics: Mapping[str, Mapping[str, Any]],
    raw_pair_metrics: Mapping[str, Mapping[str, Any]],
    effective_pair_metrics: Mapping[str, Mapping[str, Any]],
    thresholds: StructuralThresholds,
) -> dict[str, Any]:
    """Apply every predeclared bound per scene and fail closed on no-op deltas."""

    if not raw_scene_metrics or set(raw_scene_metrics) != set(effective_scene_metrics):
        _fail("Raw/effective scene metric sets must be equal and nonempty")
    required_pairs = {COLOR_PAIR_ID, MIRROR_PAIR_ID}
    if set(raw_pair_metrics) != required_pairs or set(effective_pair_metrics) != required_pairs:
        _fail("Raw/effective pair metrics must contain exactly color and mirror pairs")

    scene_checks: dict[str, dict[str, bool]] = {}
    for scene_id in sorted(raw_scene_metrics):
        raw = raw_scene_metrics[scene_id]
        effective = effective_scene_metrics[scene_id]
        scene_checks[scene_id] = {
            "raw_positive_finite_total_energy": (
                bool(raw.get("positive_finite_total_energy"))
                if thresholds.require_positive_finite_raw_total_energy
                else True
            ),
            "effective_positive_finite_total_energy": (
                bool(effective.get("positive_finite_total_energy"))
                if thresholds.require_positive_finite_effective_total_energy
                else True
            ),
            "raw_mean_energy_fraction": float(raw.get("across_slot_mean_energy_fraction", math.inf))
            <= thresholds.maximum_raw_mean_energy_fraction,
            "raw_slot_varying_energy_fraction": float(
                raw.get("slot_varying_energy_fraction", -math.inf)
            )
            >= thresholds.minimum_raw_slot_varying_energy_fraction,
            "effective_mean_energy_fraction": float(
                effective.get("across_slot_mean_energy_fraction", math.inf)
            )
            <= thresholds.maximum_effective_mean_energy_fraction,
            "effective_slot_varying_energy_fraction": float(
                effective.get("slot_varying_energy_fraction", -math.inf)
            )
            >= thresholds.minimum_effective_slot_varying_energy_fraction,
            "effective_delta_to_core_rms_ratio": float(
                effective.get("delta_to_core_rms_ratio", math.inf)
            )
            <= thresholds.maximum_effective_delta_to_core_rms_ratio,
        }
    pair_checks = {
        pair_id: {
            "raw_positive_finite_pair_delta": (
                bool(raw_pair_metrics[pair_id].get("positive_finite_pair_delta"))
                if thresholds.require_positive_finite_pair_delta
                else True
            ),
            "effective_positive_finite_pair_delta": (
                bool(effective_pair_metrics[pair_id].get("positive_finite_pair_delta"))
                if thresholds.require_positive_finite_pair_delta
                else True
            ),
        }
        for pair_id in sorted(required_pairs)
    }
    scene_passed = all(all(checks.values()) for checks in scene_checks.values())
    pairs_passed = all(all(checks.values()) for checks in pair_checks.values())
    return {
        "schema_version": 1,
        "thresholds": asdict(thresholds),
        "scene_checks": scene_checks,
        "pair_checks": pair_checks,
        "maximum_observed_raw_mean_energy_fraction": max(
            float(value["across_slot_mean_energy_fraction"]) for value in raw_scene_metrics.values()
        ),
        "minimum_observed_raw_slot_varying_energy_fraction": min(
            float(value["slot_varying_energy_fraction"]) for value in raw_scene_metrics.values()
        ),
        "maximum_observed_effective_mean_energy_fraction": max(
            float(value["across_slot_mean_energy_fraction"])
            for value in effective_scene_metrics.values()
        ),
        "minimum_observed_effective_slot_varying_energy_fraction": min(
            float(value["slot_varying_energy_fraction"])
            for value in effective_scene_metrics.values()
        ),
        "maximum_observed_effective_delta_to_core_rms_ratio": max(
            float(value["delta_to_core_rms_ratio"]) for value in effective_scene_metrics.values()
        ),
        "all_scenes_passed": scene_passed,
        "all_pairs_nonzero_finite": pairs_passed,
        "passed": scene_passed and pairs_passed,
    }


class _RawCenteredDeltaView(nn.Module):
    def __init__(self, residual: nn.Module) -> None:
        super().__init__()
        self.residual = residual

    def forward(self, scene_tokens: torch.Tensor) -> torch.Tensor:
        return self.residual.centered_delta_values(scene_tokens)


def functional_simulated_deltas(
    residual: nn.Module,
    scene_tokens: torch.Tensor,
    simulated_output_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate raw centered and effective deltas without mutating live state."""

    expected_weight = residual.output_projection.weight
    if simulated_output_weight.shape != expected_weight.shape:
        raise ValueError("Simulated output weight shape mismatch")
    state = {**dict(residual.named_parameters()), **dict(residual.named_buffers())}
    state["output_projection.weight"] = simulated_output_weight.to(
        device=expected_weight.device, dtype=expected_weight.dtype
    )
    effective_output = functional_call(residual, state, (scene_tokens,), strict=True)

    view = _RawCenteredDeltaView(residual)
    view_state = {f"residual.{name}": value for name, value in state.items()}
    raw_delta = functional_call(view, view_state, (scene_tokens,), strict=True)
    return raw_delta, effective_output - scene_tokens


def _tensor_state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    from semantic_3d_chat.language.lora import tensor_state_sha256

    return tensor_state_sha256(state)


def simulated_residual_state_sha256(
    residual: nn.Module, simulated_output_weight: torch.Tensor
) -> str:
    state = {
        f"global_scene_residual.{name}": value.detach()
        for name, value in residual.state_dict().items()
    }
    state["global_scene_residual.output_projection.weight"] = simulated_output_weight.detach().to(
        device=residual.output_projection.weight.device,
        dtype=residual.output_projection.weight.dtype,
    )
    return _tensor_state_sha256(state)


def capture_rng_states(*, require_mps: bool) -> dict[str, torch.Tensor | None]:
    """Capture CPU and, when available, MPS RNG state without advancing either."""

    cpu = torch.random.get_rng_state().detach().cpu().clone()
    mps_available = bool(torch.backends.mps.is_available())
    if require_mps and not mps_available:
        _fail("V18 preflight requires MPS RNG evidence but MPS is unavailable")
    mps = torch.mps.get_rng_state().detach().cpu().clone() if mps_available else None
    return {"cpu": cpu, "mps": mps}


def rng_state_evidence(
    before: Mapping[str, torch.Tensor | None],
    after: Mapping[str, torch.Tensor | None],
) -> dict[str, Any]:
    """Hash and compare the two RNG domains used by local Gemma execution."""

    if set(before) != {"cpu", "mps"} or set(after) != {"cpu", "mps"}:
        raise ValueError("RNG evidence must contain exactly CPU and MPS states")
    domains: dict[str, dict[str, Any]] = {}
    for name in ("cpu", "mps"):
        first = before[name]
        second = after[name]
        available = first is not None and second is not None
        if (first is None) != (second is None):
            unchanged = False
        elif not available:
            unchanged = True
        else:
            assert first is not None and second is not None
            unchanged = torch.equal(first, second)
        domains[name] = {
            "available": available,
            "before_sha256": (
                None if first is None else _tensor_state_sha256({f"{name}_rng": first})
            ),
            "after_sha256": (
                None if second is None else _tensor_state_sha256({f"{name}_rng": second})
            ),
            "unchanged": unchanged,
        }
    return {
        "domains": domains,
        "all_available_domains_unchanged": all(value["unchanged"] for value in domains.values()),
    }


def restore_rng_states(states: Mapping[str, torch.Tensor | None]) -> None:
    """Restore a captured RNG snapshot after a failed no-mutation check."""

    cpu = states.get("cpu")
    if not isinstance(cpu, torch.Tensor):
        raise TypeError("Captured CPU RNG state is missing")
    torch.random.set_rng_state(cpu)
    mps = states.get("mps")
    if mps is not None:
        torch.mps.set_rng_state(mps)


def _exact_clone_adamw_evidence(
    residual: nn.Module,
    optimizer_contract: Mapping[str, Any],
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Step an isolated full-module clone through the exact PyTorch AdamW path.

    The earlier analytic first-step formula is mathematically equivalent but
    not bit-equivalent to PyTorch's ordered FP32 moment operations. V18 requires
    a state-hash prediction, so this routine clones every residual parameter
    and gradient, applies the same clipping call and explicit optimizer switches
    as training, and never mutates the live module or its empty optimizer state.
    """

    live_named = list(residual.named_parameters())
    if not live_named:
        raise ValueError("Residual clone simulation requires parameters")
    clone = copy.deepcopy(residual)
    clone_named = list(clone.named_parameters())
    if [name for name, _ in clone_named] != [name for name, _ in live_named]:
        raise RuntimeError("Residual clone parameter order changed")
    gradient_manifest: dict[str, torch.Tensor] = {}
    for (name, live), (clone_name, cloned) in zip(live_named, clone_named, strict=True):
        if name != clone_name:
            raise RuntimeError("Residual clone parameter names changed")
        if live.grad is None:
            cloned.grad = None
            continue
        cloned.grad = live.grad.detach().clone()
        gradient_manifest[name] = cloned.grad.detach().float().cpu().contiguous()
    output_name = "output_projection.weight"
    if output_name not in gradient_manifest:
        raise ValueError("Residual clone simulation lacks the output-projection gradient")

    cloned_parameters = [parameter for _name, parameter in clone_named]
    pre_clip_norm_tensor = torch.linalg.vector_norm(
        torch.stack(
            [
                parameter.grad.detach().float().norm()
                for parameter in cloned_parameters
                if parameter.grad is not None
            ]
        )
    )
    returned_pre_clip_norm = torch.nn.utils.clip_grad_norm_(
        cloned_parameters,
        float(optimizer_contract["gradient_clip_norm"]),
    )
    clipped_manifest = {
        name: parameter.grad.detach().float().cpu().contiguous()
        for name, parameter in clone_named
        if parameter.grad is not None
    }
    optimizer = torch.optim.AdamW(
        [
            {
                "name": V18_RESIDUAL_OPTIMIZER_GROUP_NAME,
                "params": cloned_parameters,
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
        raise RuntimeError("Fresh clone AdamW unexpectedly has optimizer moments")
    optimizer.step()
    optimizer_state_manifest, optimizer_state_sha256 = canonical_v18_adamw_state(
        optimizer.state_dict(),
        optimizer_contract,
        parameter_specs=tuple((name, tuple(parameter.shape)) for name, parameter in clone_named),
    )
    clone_by_name = dict(clone.named_parameters())
    simulated_weight = clone_by_name[output_name].detach().clone()
    changed_parameters = [
        name
        for (name, live), (_clone_name, cloned) in zip(
            live_named, clone.named_parameters(), strict=True
        )
        if not torch.equal(live.detach(), cloned.detach())
    ]
    if changed_parameters != [output_name]:
        raise RuntimeError(
            f"Exact clone AdamW changed an unexpected residual surface: {changed_parameters}"
        )
    optimizer_state_tensors = {
        f"{parameter_name}.{state_name}": value.detach().float().cpu().contiguous()
        for parameter_name, parameter in clone_named
        for state_name, value in optimizer.state[parameter].items()
        if isinstance(value, torch.Tensor)
    }
    pre_clip_norm = float(pre_clip_norm_tensor.detach().cpu())
    post_clip_norm = float(
        torch.linalg.vector_norm(
            torch.stack(
                [
                    parameter.grad.detach().float().norm()
                    for parameter in cloned_parameters
                    if parameter.grad is not None
                ]
            )
        )
        .detach()
        .cpu()
    )
    evidence = {
        "implementation": "isolated_full_residual_torch_adamw_clone",
        "parameter_count": int(simulated_weight.numel()),
        "pre_clip_gradient_l2_norm": pre_clip_norm,
        "clip_returned_pre_clip_gradient_l2_norm": float(returned_pre_clip_norm.detach().cpu()),
        "clip_scale": min(
            1.0,
            float(optimizer_contract["gradient_clip_norm"]) / (pre_clip_norm + 1.0e-6),
        ),
        "post_clip_gradient_l2_norm": post_clip_norm,
        "update_l2_norm": float(simulated_weight.detach().float().norm().cpu()),
        "update_rms": float(simulated_weight.detach().float().square().mean().sqrt().cpu()),
        "update_absolute_maximum": float(simulated_weight.detach().float().abs().max().cpu()),
        "nonzero_update_count": int(torch.count_nonzero(simulated_weight).cpu()),
        "gradient_parameter_keys": sorted(gradient_manifest),
        "changed_parameter_keys": changed_parameters,
        "gradient_sha256": _tensor_state_sha256(gradient_manifest),
        "clipped_gradient_sha256": _tensor_state_sha256(clipped_manifest),
        "simulated_update_sha256": _tensor_state_sha256(
            {"output_projection.update": simulated_weight}
        ),
        "clone_optimizer_state_tensor_sha256": _tensor_state_sha256(optimizer_state_tensors),
        "clone_optimizer_state_manifest": optimizer_state_manifest,
        "clone_optimizer_state_sha256": optimizer_state_sha256,
        "clone_optimizer_state_parameter_count": len(optimizer.state),
        "clone_residual_state_sha256": simulated_residual_state_sha256(residual, simulated_weight),
    }
    return simulated_weight, evidence


def run_preflight(config_path: str | Path, report_path: str | Path) -> dict[str, Any]:
    """Run the pinned V18 preflight without mutating live model/optimizer state."""

    # Heavy supervised/model imports remain local to this offline entry point.
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
    from semantic_3d_chat.scene_encoder import global_residual as residual_source
    from semantic_3d_chat.scene_encoder.global_residual import (
        apply_global_scene_residual,
        construct_global_scene_residual,
        global_scene_residual_settings,
    )
    from semantic_3d_chat.scene_encoder.map_io import load_map_tensors
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
        select_pair_only_records,
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
        verify_zero_output_scene_residual_equivalence,
    )

    config = load_config(config_path)
    implementation_path = Path(residual_source.__file__).resolve()
    contract = validate_v18_config_contract(
        config, implementation_source_sha256=file_sha256(implementation_path)
    )
    source_provenance = capture_git_source_provenance(PROJECT_ROOT)
    try:
        require_clean_committed_source(source_provenance)
    except RuntimeError as error:
        raise V18StructuralPreflightViolation(str(error)) from error
    set_seed(int(config["seed"]))

    expected_hashes = contract["expected_hashes"]
    for name, path_value in contract["evidence_paths"].items():
        path = Path(path_value)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        expected = expected_hashes[f"{name}_sha256"]
        observed = file_sha256(path)
        if observed != expected:
            _fail(f"Pinned {name} evidence hash mismatch: expected={expected} observed={observed}")

    training = config["training"]
    pair_settings = pair_curriculum_settings(config)
    qa_root = artifact_root(config, "qa")
    dataset = SceneQADataset(qa_root / "train.jsonl")
    selected_available = select_pair_only_records(
        dataset.records, pair_settings.pair_only_scene_ids
    )
    selected_available = cap_pair_units_per_pair(
        selected_available, pair_settings.max_units_per_pair, seed=int(config["seed"])
    )
    records = select_training_records(
        selected_available,
        max_questions_per_scene=training.get("max_questions_per_scene"),
    )
    selection = training_selection_summary(selected_available, records)
    if selection["selected_ids_sha256"] != expected_hashes["selection_sha256"]:
        _fail("V18 selected training-record hash mismatch")
    pair_units = build_exact_question_pair_units(records)
    by_scene: dict[str, list[Any]] = defaultdict(list)
    for record in records:
        by_scene[record.scene_id].append(record)
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
    if ordered_hash != expected_hashes["ordered_unit_sha256"]:
        _fail(
            "V18 ordered epoch-one unit hash mismatch: "
            f"expected={expected_hashes['ordered_unit_sha256']} observed={ordered_hash}"
        )
    training_pairs = training_counterfactual_scene_pairs(records)
    pair_membership_text = "\n".join(
        f"{pair_id}:{first_scene}:{second_scene}"
        for pair_id, first_scene, second_scene in training_pairs
    )
    pair_membership_hash = hashlib.sha256(pair_membership_text.encode("utf-8")).hexdigest()
    if pair_membership_hash != expected_hashes["pair_membership_sha256"]:
        _fail("V18 pair-membership hash mismatch")

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
        _fail("V18 requires installed and entirely frozen named LoRA banks")
    lora.eval()

    scene_ids = sorted(by_scene)
    if scene_ids != sorted(expected_hashes["core_prefix_sha256"]):
        _fail("V18 selected scene IDs disagree with expected core-prefix hashes")
    maps = {
        scene_id: load_map_tensors(
            project_path(config, "maps", scene_id, "voxel_map.npz"),
            config["scene"]["room_size_m"],
            language.device,
            input_voxel_size_m=config["scene_encoder"].get("input_voxel_size_m"),
        )
        for scene_id in scene_ids
    }
    feature_dims = {data.feature_dim for data in maps.values()}
    if len(feature_dims) != 1:
        _fail(f"Inconsistent V18 semantic dimensions: {sorted(feature_dims)}")
    scene_model = construct_scene_tokenizer(config, feature_dims.pop(), language.hidden_size).to(
        language.device
    )
    residual = construct_global_scene_residual(
        config,
        scene_dim=language.hidden_size,
        latent_count=int(config["scene_encoder"]["global_latents"]),
    )
    if residual is None:
        _fail("V18 residual construction returned None")
    residual = residual.to(language.device)
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

    source = Path(str(training["initialize_from"]))
    if not source.is_absolute():
        source = PROJECT_ROOT / source
    source = source.resolve()
    scene_state_modules = {
        "scene_model": scene_model,
        "composer": composer,
        "grounding": grounding,
    }
    source_modules = {**scene_state_modules, **lora.state_modules()}
    source_metadata = load_adapter_checkpoint(source, source_modules, device=str(language.device))
    if source_metadata.get("epoch") != 7:
        _fail("V18 must restart from the exact V14 epoch-7 checkpoint")
    observed_source_hashes = {
        "source_adapter_sha256": file_sha256(source / "adapter.safetensors"),
        "source_metadata_sha256": file_sha256(source / "metadata.json"),
        "frozen_scene_state_sha256": module_collection_state_sha256(scene_state_modules),
        "frozen_lora_bank_state_sha256": lora.state_sha256(),
    }
    for name in ("source_adapter_sha256", "source_metadata_sha256"):
        if observed_source_hashes[name] != expected_hashes[name]:
            _fail(f"V18 pinned source hash mismatch for {name}")
    if (
        observed_source_hashes["frozen_scene_state_sha256"]
        != expected_hashes["frozen_scene_state_sha256"]
    ):
        _fail("V18 frozen scene-state hash mismatch")
    if (
        observed_source_hashes["frozen_lora_bank_state_sha256"]
        != expected_hashes["frozen_lora_bank_state_sha256"]
    ):
        _fail("V18 frozen LoRA-bank hash mismatch")

    scene_model.requires_grad_(False).eval()
    composer.requires_grad_(False).eval()
    grounding.requires_grad_(False).eval()
    residual.requires_grad_(True).train()
    structural_state = residual.validate_structural_state()
    residual_settings = global_scene_residual_settings(config)
    if residual_settings.architecture_version != ZERO_SPATIAL_MEAN_CONTENT_GATE_V1:
        _fail("Legacy residual architecture cannot run the V18 preflight")
    initial_residual_hash = module_collection_state_sha256({"global_scene_residual": residual})
    if initial_residual_hash != expected_hashes["initial_residual_state_sha256"]:
        _fail("V18 deterministic residual initial-state hash mismatch")
    expected_structural_state = {
        "architecture_version": ZERO_SPATIAL_MEAN_CONTENT_GATE_V1,
        "parameter_count": 400_128,
        "latent_count": 256,
        "scene_dim": 1536,
        "gate_temperature": float(
            config["scene_encoder"]["global_scene_residual"]["gate_temperature"]
        ),
        "spatial_centering": "all_slots_fp32",
        "content_gate": "bias_free_scalar_sigmoid_centered_content",
    }
    if structural_state != expected_structural_state:
        _fail(
            "V18 residual structural state mismatch: "
            f"expected={expected_structural_state} observed={structural_state}"
        )
    position_features_hash = _tensor_state_sha256({"position_features": residual.position_features})
    if position_features_hash != expected_hashes["position_features_sha256"]:
        _fail("V18 deterministic position-feature hash mismatch")
    if torch.count_nonzero(residual.output_projection.weight).item() != 0:
        _fail("V18 output projection must begin at exact zero")
    state_hash_before = initial_residual_hash
    zero_equivalence = verify_zero_output_scene_residual_equivalence(
        scene_model,
        residual,
        composer,
        maps,
        model_dtype=next(language.model.parameters()).dtype,
    )
    observed_prefixes = {
        scene_id: values["core_prefix_sha256"]
        for scene_id, values in zero_equivalence["scene_prefixes"].items()
    }
    if observed_prefixes != expected_hashes["core_prefix_sha256"]:
        _fail("V18 update-zero core-prefix hash mismatch")

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

    zero = torch.zeros((), device=language.device)
    rng_before = capture_rng_states(require_mps=language.device.type == "mps")
    residual.zero_grad(set_to_none=True)
    microstep_losses: list[dict[str, Any]] = []
    for index, batch in enumerate(curriculum, start=1):
        unit = batch.pair_units[0]
        outputs = {
            scene_id: apply_global_scene_residual(core_outputs[scene_id], residual)
            for scene_id in unit.scene_ids
        }
        base, language_loss, grounding_loss, ranking_loss, diagnostics = pair_batch_objective(
            outputs,
            [unit],
            maps,
            language,
            composer,
            grounding,
            config,
            ranking_margin=pair_settings.ranking_margin,
            ranking_mode=pair_settings.ranking_mode,
            collect_full_vocab_first_answer_token=True,
            full_vocab_ranking_margin=pair_settings.full_vocab_ranking_margin,
        )
        full_vocab_loss = diagnostics["first_answer_token_full_vocab_ranking_loss"]
        if not isinstance(full_vocab_loss, torch.Tensor):
            _fail("V18 microstep did not produce a full-vocabulary ranking loss")
        loss = combine_pair_training_losses(
            base,
            ranking_loss,
            full_vocab_loss,
            zero,
            zero,
            pair_ranking_weight=pair_settings.ranking_weight,
            full_vocab_ranking_weight=pair_settings.full_vocab_ranking_weight,
            diversity_weight=0.0,
            scene_separation_weight=0.0,
        )
        (loss / contract["optimizer"]["accumulation_divisor"]).backward()
        microstep_losses.append(
            {
                "microstep": index,
                "pair_id": unit.pair_id,
                "question_key": unit.question_key,
                "total_loss": float(loss.detach()),
                "language_loss": float(language_loss.detach()),
                "grounding_loss": float(grounding_loss.detach()),
                "candidate_ranking_loss": float(ranking_loss.detach()),
                "full_vocab_ranking_loss": float(full_vocab_loss.detach()),
            }
        )
        del outputs, base, loss, language_loss, grounding_loss, ranking_loss, diagnostics

    output_parameter = residual.output_projection.weight
    if output_parameter.grad is None:
        _fail("V18 ordered curriculum produced no output-projection gradient")
    non_output_gradient_norms = {
        name: (
            0.0 if parameter.grad is None else float(parameter.grad.detach().float().norm().cpu())
        )
        for name, parameter in residual.named_parameters()
        if name != "output_projection.weight"
    }
    if any(value != 0.0 for value in non_output_gradient_norms.values()):
        _fail("V18 zero-output start unexpectedly opened non-output gradients")
    simulated_weight, gradient_report = _exact_clone_adamw_evidence(
        residual,
        contract["optimizer"],
    )
    simulated_weight = simulated_weight.to(language.device)
    simulated_state_hash = simulated_residual_state_sha256(residual, simulated_weight)

    raw_deltas: dict[str, torch.Tensor] = {}
    effective_deltas: dict[str, torch.Tensor] = {}
    raw_metrics: dict[str, dict[str, Any]] = {}
    effective_metrics: dict[str, dict[str, Any]] = {}
    gate_values: dict[str, dict[str, Any]] = {}
    with torch.no_grad():
        for scene_id, output in core_outputs.items():
            raw_delta, effective_delta = functional_simulated_deltas(
                residual, output.scene_tokens, simulated_weight
            )
            raw_deltas[scene_id] = raw_delta
            effective_deltas[scene_id] = effective_delta
            raw_metrics[scene_id] = {
                **fp64_delta_metrics(output.scene_tokens, raw_delta),
                "delta_sha256": _tensor_state_sha256({"delta": raw_delta}),
            }
            effective_metrics[scene_id] = {
                **fp64_delta_metrics(output.scene_tokens, effective_delta),
                "delta_sha256": _tensor_state_sha256({"delta": effective_delta}),
            }
            values = residual.content_gate_values(output.scene_tokens).detach().float().cpu()
            gate_values[scene_id] = {
                "shape": list(values.shape),
                "minimum": float(values.min()),
                "maximum": float(values.max()),
                "mean": float(values.mean()),
                "sha256": _tensor_state_sha256({"content_gate": values}),
            }

    raw_pairs: dict[str, dict[str, Any]] = {}
    effective_pairs: dict[str, dict[str, Any]] = {}
    scene_pair_by_id = {
        pair_id: (first_scene, second_scene)
        for pair_id, first_scene, second_scene in training_pairs
    }
    for pair_id in (COLOR_PAIR_ID, MIRROR_PAIR_ID):
        first_id, second_id = scene_pair_by_id[pair_id]
        raw_pairs[pair_id] = {
            "first_scene_id": first_id,
            "second_scene_id": second_id,
            **fp64_pair_delta_metrics(
                core_outputs[first_id].scene_tokens,
                core_outputs[second_id].scene_tokens,
                raw_deltas[first_id],
                raw_deltas[second_id],
            ),
        }
        effective_pairs[pair_id] = {
            "first_scene_id": first_id,
            "second_scene_id": second_id,
            **fp64_pair_delta_metrics(
                core_outputs[first_id].scene_tokens,
                core_outputs[second_id].scene_tokens,
                effective_deltas[first_id],
                effective_deltas[second_id],
            ),
        }

    thresholds = StructuralThresholds(**contract["thresholds"])
    gate = evaluate_structural_gate(
        raw_metrics, effective_metrics, raw_pairs, effective_pairs, thresholds
    )
    rng_after = capture_rng_states(require_mps=language.device.type == "mps")
    rng_evidence = rng_state_evidence(rng_before, rng_after)
    if not rng_evidence["all_available_domains_unchanged"]:
        restore_rng_states(rng_before)
        rng_evidence["restored_after_mismatch"] = True
    else:
        rng_evidence["restored_after_mismatch"] = False
    residual.zero_grad(set_to_none=True)
    state_hash_after = module_collection_state_sha256({"global_scene_residual": residual})
    live_state_unchanged = state_hash_after == state_hash_before
    authorization = (
        gate["passed"] and live_state_unchanged and rng_evidence["all_available_domains_unchanged"]
    )
    report = {
        "schema_version": 1,
        "audit_type": "v18_exact_ordered_structural_preflight",
        "runtime_eligible": False,
        "uses_supervised_qa_metadata": True,
        "question_dependent_scene_processing": False,
        "live_optimizer_constructed": False,
        "live_optimizer_step_executed": False,
        "optimizer_steps": 0,
        "isolated_clone_optimizer_constructed": True,
        "isolated_clone_optimizer_steps": 1,
        "structural_authorization": authorization,
        "config_path": str(Path(config["_config_path"]).relative_to(PROJECT_ROOT)),
        "config_sha256": config_hash(config, length=64),
        "contract": contract,
        "source_provenance": source_provenance,
        "implementation_source": str(implementation_path.relative_to(PROJECT_ROOT)),
        "implementation_source_sha256": file_sha256(implementation_path),
        "source_checkpoint": str(source.relative_to(PROJECT_ROOT)),
        "source_checkpoint_epoch": source_metadata.get("epoch"),
        "source_hashes": observed_source_hashes,
        "initial_residual_state_sha256": initial_residual_hash,
        "live_residual_state_sha256_before": state_hash_before,
        "live_residual_state_sha256_after": state_hash_after,
        "live_parameter_state_unchanged": live_state_unchanged,
        "simulated_first_output_projection_state_sha256": simulated_state_hash,
        "structural_state": structural_state,
        "position_features_sha256": position_features_hash,
        "rng_state": rng_evidence,
        "zero_output_prefix_equivalence": zero_equivalence,
        "selection_sha256": selection["selected_ids_sha256"],
        "pair_membership_sha256": pair_membership_hash,
        "ordered_unit_sha256": ordered_hash,
        "ordered_units": ordered_units,
        "microstep_losses": microstep_losses,
        "gradient": {
            **gradient_report,
            "non_output_parameter_gradient_l2_norms": non_output_gradient_norms,
            "ordered_microstep_count": len(curriculum),
        },
        "adamw_contract": contract["optimizer"],
        "content_gate": gate_values,
        "raw_centered_scene_delta": raw_metrics,
        "effective_scene_delta": effective_metrics,
        "raw_centered_pair_delta": raw_pairs,
        "effective_pair_delta": effective_pairs,
        "structural_gate": gate,
    }
    destination = Path(report_path)
    if not destination.is_absolute():
        destination = PROJECT_ROOT / destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not authorization:
        raise V18StructuralPreflightViolation(
            f"V18 structural preflight failed; evidence written to {destination}"
        )
    print(json.dumps({"phase": "v18_structural_preflight_passed", "report": str(destination)}))
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
    "MIRROR_PAIR_ID",
    "STRUCTURAL_PREFLIGHT_ROLE",
    "V18_SCREEN_ROLE",
    "StructuralThresholds",
    "V18StructuralPreflightViolation",
    "canonical_sha256",
    "capture_rng_states",
    "evaluate_structural_gate",
    "fp64_delta_metrics",
    "fp64_pair_delta_metrics",
    "functional_simulated_deltas",
    "ordered_curriculum_evidence",
    "restore_rng_states",
    "rng_state_evidence",
    "run_preflight",
    "simulated_residual_state_sha256",
    "validate_v18_config_contract",
]
