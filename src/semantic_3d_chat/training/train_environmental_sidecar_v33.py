"""V33 environmental-only sidecar recovery from approved V29 update 004.

V32 proved that decoder adaptation can lower NLL without making the model use
weak scene differences.  V33 therefore freezes Gemma and every LoRA bank and
optimizes only 404,608 parameters on the all-voxel dense-sidecar path.  It is
strictly conditional on the pinned, terminal V32 rejection report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from semantic_3d_chat.config import PROJECT_ROOT, config_hash, load_config
from semantic_3d_chat.data.dataset import QARecord
from semantic_3d_chat.language.lora import tensor_state_sha256
from semantic_3d_chat.scene_encoder.dense_sidecar_adapter import (
    DenseSidecarAdapter,
    validate_dense_sidecar_adapter_state,
)
from semantic_3d_chat.training.checkpointing import (
    RUNTIME_METADATA_FILENAME,
    TRAINING_METADATA_FILENAME,
    load_adapter_checkpoint,
    load_optimizer_checkpoint,
    runtime_checkpoint_metadata,
    save_adapter_checkpoint,
    save_optimizer_checkpoint,
    validate_runtime_checkpoint_metadata,
)
from semantic_3d_chat.training.pair_curriculum import (
    CounterfactualPairUnit,
    build_exact_question_pair_units,
)
from semantic_3d_chat.training.train_joint_pair_v30 import (
    ApprovedV29Source,
    CachedPreSidecarScene,
    V30Bundle,
    V30Settings,
    adapted_scene_tokens,
    cache_pre_sidecar_scenes,
    cached_broad_answer_nll,
    load_v30_bundle,
    paired_canonical_answer_objective,
    require_approved_v29_source,
    select_balanced_broad_records,
    v30_contract,
    validation_answer_nll,
    validation_pair_metrics,
    verify_fresh_bank_update_zero,
)
from semantic_3d_chat.training.train_joint_pair_v30 import (
    _metadata as _v30_metadata,
)
from semantic_3d_chat.training.train_joint_pair_v31 import (
    V31Contract,
    load_v31_qa_records,
    v31_contract,
)
from semantic_3d_chat.training.train_post_stack_decoder import _source_validation_nll

DEFAULT_CONFIG = Path("configs/experiments/gemma4_diverse28_environmental_sidecar_v33.yaml")
DEFAULT_OUTPUT = Path("data_gemma4/checkpoints/gemma4_v33_diverse28_environmental_sidecar")
_NEW_TRAIN_SCENES = tuple(f"scene_{index:06d}" for index in range(31, 39))
_UPDATE_DIRECTORY = re.compile(r"update_([0-9]{3})")
_V32_REPORT_SHA256 = "2ffeb2655cd6a8627ea9e06c8f261113b0b225a1b39de4eb32126693063c13b7"
_OUTPUT_NAMES = ("output_projection.weight", "channel_gain")
_HIDDEN_NAMES = (
    "sidecar_norm.weight",
    "sidecar_norm.bias",
    "sidecar_projection.weight",
    "sidecar_projection.bias",
)
_POSITION_NAMES = ("position_projection.weight", "position_projection.bias")
_TRAINABLE_NAMES = (*_OUTPUT_NAMES, *_HIDDEN_NAMES, *_POSITION_NAMES)
_FROZEN_BASE_NAMES = (
    "base_norm.weight",
    "base_norm.bias",
    "base_projection.weight",
    "base_projection.bias",
)
_VALIDATION_FAMILY_PAIR_IDS = {
    "book_support": "pair_000009",
    "mirror_lr": "pair_000010",
    "picture_support": "pair_000011",
}


@dataclass(frozen=True)
class V33Settings:
    enabled: bool
    optimizer_steps: int
    checkpoint_interval_steps: int
    broad_batch_size: int
    pair_units_per_step: int
    broad_exclude_expected_change: bool
    broad_nll_weight: float
    pair_language_nll_weight: float
    pair_margin_weight: float
    pair_margin: float
    output_learning_rate: float
    hidden_learning_rate: float
    position_learning_rate: float
    weight_decay: float
    output_gradient_clip_norm: float
    hidden_gradient_clip_norm: float
    position_gradient_clip_norm: float
    minimum_answer_types: int

    @property
    def saved_optimizer_steps(self) -> tuple[int, ...]:
        regular = tuple(range(0, self.optimizer_steps, self.checkpoint_interval_steps))
        return (*regular, self.optimizer_steps)


@dataclass(frozen=True)
class V33Contract:
    v31: V31Contract
    v32_selection_report: Path
    v32_selection_report_sha256: str
    saved_optimizer_steps: tuple[int, ...]
    optimizer_steps: int
    checkpoint_interval_steps: int
    minimum_pair_unit_recurrence: int
    development_changed_complete_pairs_minimum: int
    chat_promotion_changed_complete_pairs_minimum: int


@dataclass(frozen=True)
class V33Environmental:
    optimizer_step: int
    broad_records: tuple[QARecord, ...]
    pair_units: tuple[CounterfactualPairUnit, ...]


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (PROJECT_ROOT / value).resolve()


def _positive_int(field: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _finite(field: str, value: object, *, positive: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed) or (parsed <= 0.0 if positive else parsed < 0.0):
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{field} must be finite and {qualifier}")
    return parsed


def _true(raw: Mapping[str, Any], field: str) -> None:
    if raw.get(field) is not True:
        raise ValueError(f"v33_environmental.{field} must remain true")


def v33_settings(config: Mapping[str, Any]) -> V33Settings:
    training = config.get("training")
    if not isinstance(training, Mapping):
        raise TypeError("training must be a mapping")
    raw = training.get("v33_environmental")
    if not isinstance(raw, Mapping):
        raise TypeError("training.v33_environmental must be a mapping")
    required = {
        "enabled",
        "optimizer_steps",
        "checkpoint_interval_steps",
        "broad_batch_size",
        "pair_units_per_step",
        "broad_exclude_expected_change",
        "broad_nll_weight",
        "pair_language_nll_weight",
        "pair_margin_weight",
        "pair_margin",
        "output_learning_rate",
        "hidden_learning_rate",
        "position_learning_rate",
        "weight_decay",
        "output_gradient_clip_norm",
        "hidden_gradient_clip_norm",
        "position_gradient_clip_norm",
        "minimum_answer_types",
    }
    if set(raw) != required:
        raise ValueError(
            "training.v33_environmental fields differ from the locked schema: "
            f"missing={sorted(required - set(raw))} unknown={sorted(set(raw) - required)}"
        )
    if not isinstance(raw["enabled"], bool) or not isinstance(
        raw["broad_exclude_expected_change"], bool
    ):
        raise TypeError("V33 boolean settings must be boolean")
    settings = V33Settings(
        enabled=raw["enabled"],
        optimizer_steps=_positive_int("optimizer_steps", raw["optimizer_steps"]),
        checkpoint_interval_steps=_positive_int(
            "checkpoint_interval_steps", raw["checkpoint_interval_steps"]
        ),
        broad_batch_size=_positive_int("broad_batch_size", raw["broad_batch_size"]),
        pair_units_per_step=_positive_int("pair_units_per_step", raw["pair_units_per_step"]),
        broad_exclude_expected_change=raw["broad_exclude_expected_change"],
        broad_nll_weight=_finite("broad_nll_weight", raw["broad_nll_weight"], positive=True),
        pair_language_nll_weight=_finite(
            "pair_language_nll_weight", raw["pair_language_nll_weight"], positive=True
        ),
        pair_margin_weight=_finite("pair_margin_weight", raw["pair_margin_weight"], positive=True),
        pair_margin=_finite("pair_margin", raw["pair_margin"], positive=True),
        output_learning_rate=_finite(
            "output_learning_rate", raw["output_learning_rate"], positive=True
        ),
        hidden_learning_rate=_finite(
            "hidden_learning_rate", raw["hidden_learning_rate"], positive=True
        ),
        position_learning_rate=_finite(
            "position_learning_rate", raw["position_learning_rate"], positive=True
        ),
        weight_decay=_finite("weight_decay", raw["weight_decay"], positive=False),
        output_gradient_clip_norm=_finite(
            "output_gradient_clip_norm", raw["output_gradient_clip_norm"], positive=True
        ),
        hidden_gradient_clip_norm=_finite(
            "hidden_gradient_clip_norm", raw["hidden_gradient_clip_norm"], positive=True
        ),
        position_gradient_clip_norm=_finite(
            "position_gradient_clip_norm", raw["position_gradient_clip_norm"], positive=True
        ),
        minimum_answer_types=_positive_int("minimum_answer_types", raw["minimum_answer_types"]),
    )
    return settings


def v33_contract(config: Mapping[str, Any]) -> V33Contract:
    """Validate V33 without loading QA, Gemma, or any scene artifact."""

    v31 = v31_contract(config)
    raw = config.get("v33_environmental")
    if not isinstance(raw, Mapping):
        raise TypeError("V33 requires a v33_environmental mapping")
    required = {
        "schema_version",
        "role",
        "engine",
        "source_selected_update",
        "v32_selection_report",
        "v32_selection_report_sha256",
        "train_scene_ids",
        "validation_scene_ids",
        "deferred_final_scene_ids",
        "train_question_count",
        "validation_question_count",
        "train_changed_pair_unit_count",
        "validation_changed_pair_unit_count",
        "optimizer_steps",
        "checkpoint_interval_steps",
        "saved_optimizer_steps",
        "exact_pair_unit_recurrence",
        "trainable_parameter_names",
        "output_parameter_count",
        "hidden_parameter_count",
        "position_parameter_count",
        "exact_trainable_parameter_count",
        "gemma_decoder_frozen",
        "all_lora_banks_frozen",
        "base_norm_and_projection_frozen",
        "exact_zero_source_cache",
        "allow_only_new_training_scenes_to_derive_prefixes",
        "inspect_every_saved_arm",
        "greedy_screen_steps",
        "weak_pair_prefix_rms_minimum_ratio",
        "unrelated_prefix_rms_maximum_ratio",
        "update64_nonmirror_teacher_complete_minimum",
        "update64_book_picture_advantage_positive",
        "development_changed_complete_pairs_minimum",
        "chat_promotion_changed_complete_pairs_minimum",
        "chat_promotion_aggregate_exact_no_regression",
        "chat_promotion_requires_each_validation_family",
        "old_color_full_vocab_sides_minimum",
        "old_mirror_full_vocab_sides_minimum",
        "old_controls_no_new_negatives",
        "conditional_next_surface",
    }
    if set(raw) != required:
        raise ValueError(
            "v33_environmental fields differ from the locked schema: "
            f"missing={sorted(required - set(raw))} unknown={sorted(set(raw) - required)}"
        )
    if raw["schema_version"] != 1:
        raise ValueError("v33_environmental.schema_version must be 1")
    if raw["role"] != "approved_v29_diverse28_environmental_sidecar_v33":
        raise ValueError("v33_environmental.role does not authorize this experiment")
    if raw["engine"] != "v29_exact_zero_environmental_only_true_microsteps":
        raise ValueError("V33 must use the audited exact-zero environmental engine")
    settings = v33_settings(config)
    if not settings.enabled:
        raise ValueError("V33 environmental training is disabled")
    expected_settings = {
        "optimizer_steps": 100,
        "checkpoint_interval_steps": 8,
        "broad_batch_size": 1,
        "pair_units_per_step": 1,
        "broad_exclude_expected_change": True,
        "broad_nll_weight": 1.0,
        "pair_language_nll_weight": 1.0,
        "pair_margin_weight": 4.0,
        "pair_margin": 0.5,
        "output_learning_rate": 2.5e-5,
        "hidden_learning_rate": 1.0e-4,
        "position_learning_rate": 1.0e-4,
        "weight_decay": 0.0,
        "output_gradient_clip_norm": 1.0,
        "hidden_gradient_clip_norm": 1.0,
        "position_gradient_clip_norm": 1.0,
        "minimum_answer_types": 4,
    }
    mismatches = {
        name: {"observed": getattr(settings, name), "expected": expected}
        for name, expected in expected_settings.items()
        if getattr(settings, name) != expected
    }
    if mismatches:
        raise ValueError(f"V33 locked optimizer settings changed: {mismatches}")
    expected_saved = (*range(0, 100, 8), 100)
    raw_saved = raw["saved_optimizer_steps"]
    if not isinstance(raw_saved, Sequence) or isinstance(raw_saved, (str, bytes)):
        raise TypeError("v33_environmental.saved_optimizer_steps must be a sequence")
    saved = tuple(raw_saved)
    if saved != expected_saved or saved != settings.saved_optimizer_steps:
        raise ValueError("V33 saved optimizer-step arms must be 0,8,...,96,100")
    if (
        raw["source_selected_update"] != 4
        or raw["optimizer_steps"] != settings.optimizer_steps
        or raw["checkpoint_interval_steps"] != settings.checkpoint_interval_steps
        or raw["exact_pair_unit_recurrence"] != 4
    ):
        raise ValueError("V33 source/optimizer schedule differs from its locked repair")
    if tuple(raw["train_scene_ids"]) != v31.train_scene_ids:
        raise ValueError("V33 training scenes differ from locked diverse28 V31")
    if tuple(raw["validation_scene_ids"]) != v31.validation_scene_ids:
        raise ValueError("V33 validation scenes must remain scenes 19--24")
    if tuple(raw["deferred_final_scene_ids"]) != v31.deferred_final_scene_ids:
        raise ValueError("V33 deferred final scenes must remain scenes 25--30")
    if (
        raw["train_question_count"] != v31.train_question_count
        or raw["validation_question_count"] != v31.validation_question_count
        or raw["train_changed_pair_unit_count"] != v31.train_changed_pair_unit_count
        or raw["validation_changed_pair_unit_count"] != 12
    ):
        raise ValueError("V33 QA or pair counts differ from locked diverse28 development data")
    for field in (
        "exact_zero_source_cache",
        "allow_only_new_training_scenes_to_derive_prefixes",
        "inspect_every_saved_arm",
        "gemma_decoder_frozen",
        "all_lora_banks_frozen",
        "base_norm_and_projection_frozen",
        "update64_book_picture_advantage_positive",
        "chat_promotion_requires_each_validation_family",
        "chat_promotion_aggregate_exact_no_regression",
        "old_controls_no_new_negatives",
    ):
        _true(raw, field)
    development_min = _positive_int(
        "development_changed_complete_pairs_minimum",
        raw["development_changed_complete_pairs_minimum"],
    )
    promotion_min = _positive_int(
        "chat_promotion_changed_complete_pairs_minimum",
        raw["chat_promotion_changed_complete_pairs_minimum"],
    )
    if development_min != 1 or promotion_min != 6:
        raise ValueError("V33 development/chat gates must remain 1/12 and 6/12")
    inherited = v30_contract(config)
    configured_names = raw["trainable_parameter_names"]
    if (
        not isinstance(configured_names, Sequence)
        or isinstance(configured_names, (str, bytes))
        or tuple(configured_names) != _TRAINABLE_NAMES
    ):
        raise ValueError("V33 environmental trainable parameter names changed")
    if (
        inherited["source_selected_update"] != 4
        or raw["output_parameter_count"] != 198_144
        or raw["hidden_parameter_count"] != 199_808
        or raw["position_parameter_count"] != 6_656
        or raw["exact_trainable_parameter_count"] != 404_608
    ):
        raise ValueError("V33 changed the approved V29 source or 404,608-parameter surface")
    selection = inherited["selection_requires"]
    promotion = inherited["promotion_requires"]
    if not (
        selection.get("color_full_vocab_sides") == 12
        and selection.get("mirror_full_vocab_sides") == 10
        and selection.get("no_new_negative_sides") is True
        and selection.get("source_v29_validation_nll_must_improve") is True
        and selection.get("minimum_greedy_complete_units_correct") == 1
        and promotion.get("validation_changed_complete_pairs_minimum") == 6
        and promotion.get("aggregate_validation_exact_accuracy_no_regression") is True
    ):
        raise ValueError("V33 inherited selection or chat-promotion gates were weakened")
    if raw["old_color_full_vocab_sides_minimum"] != 12:
        raise ValueError("V33 old-color retention threshold changed")
    if raw["old_mirror_full_vocab_sides_minimum"] != 10:
        raise ValueError("V33 old-mirror retention threshold changed")
    if tuple(raw["greedy_screen_steps"]) != (32, 64, 100):
        raise ValueError("V33 greedy screens must remain updates 32, 64, and 100")
    if float(raw["weak_pair_prefix_rms_minimum_ratio"]) != 1.25:
        raise ValueError("V33 weak-pair prefix threshold changed")
    if float(raw["unrelated_prefix_rms_maximum_ratio"]) != 1.25:
        raise ValueError("V33 unrelated-prefix inflation threshold changed")
    if raw["update64_nonmirror_teacher_complete_minimum"] != 1:
        raise ValueError("V33 update-64 nonmirror teacher gate changed")
    conditional = raw["conditional_next_surface"]
    if not isinstance(conditional, Mapping) or conditional != {
        "enabled": False,
        "parameter_names": list(_FROZEN_BASE_NAMES),
        "parameter_count": 199_808,
        "trigger": "no_nonmirror_teacher_complete_by_update_64",
    }:
        raise ValueError("V33 conditional next surface must remain defined and disabled")
    report_sha = str(raw["v32_selection_report_sha256"])
    if report_sha != _V32_REPORT_SHA256:
        raise ValueError("V33 V32 rejection hash differs from the approved terminal result")
    selection_report = _resolve(str(raw["v32_selection_report"]))
    return V33Contract(
        v31=v31,
        v32_selection_report=selection_report,
        v32_selection_report_sha256=report_sha,
        saved_optimizer_steps=saved,
        optimizer_steps=settings.optimizer_steps,
        checkpoint_interval_steps=settings.checkpoint_interval_steps,
        minimum_pair_unit_recurrence=4,
        development_changed_complete_pairs_minimum=development_min,
        chat_promotion_changed_complete_pairs_minimum=promotion_min,
    )


def v32_rejection_status(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return a fail-closed status for V33's pinned terminal V32 dependency."""

    contract = v33_contract(config)
    path = contract.v32_selection_report
    if path.is_symlink() or not path.is_file():
        return {
            "status": "pending",
            "report": str(path),
            "report_sha256": None,
            "training_authorized": False,
        }
    report_bytes = path.read_bytes()
    observed_sha = hashlib.sha256(report_bytes).hexdigest()
    if observed_sha != contract.v32_selection_report_sha256:
        raise ValueError("V32 selection report hash differs from V33's immutable pin")
    report = json.loads(report_bytes)
    if not isinstance(report, Mapping):
        raise TypeError("V32 selection report must be a JSON object")
    required = {
        "schema_version": 1,
        "artifact": "v32_true_microstep_development_selection",
        "all_saved_arms_inspected": True,
        "final_test_scenes_touched": False,
        "passed": False,
        "development_selection_passed": False,
        "chat_promotion_eligible": False,
        "selected_checkpoint": None,
        "selected_update": None,
        "selected_optimizer_step": None,
    }
    mismatch = {
        field: {"observed": report.get(field), "expected": expected}
        for field, expected in required.items()
        if report.get(field) != expected
    }
    if mismatch:
        raise ValueError(f"V32 selection report is not an audited terminal rejection: {mismatch}")
    if tuple(report.get("validation_scene_ids", ())) != contract.v31.validation_scene_ids:
        raise ValueError("V32 selector validation scenes differ from V33's locked development set")
    if tuple(report.get("train_scene_ids", ())) != contract.v31.train_scene_ids:
        raise ValueError("V32 selector training scenes differ from V33's locked development set")
    if tuple(report.get("deferred_final_scene_ids", ())) != contract.v31.deferred_final_scene_ids:
        raise ValueError("V32 selector deferred-final scenes differ from V33's lock")
    terminal_contract = {
        "development_validation_model_selection_only": True,
        "training_evaluation_only": True,
        "oracle_loaded": False,
        "question_dependent_scene_processing": False,
        "question_dependent_retrieval": False,
        "development_progress_is_not_chat_promotion": True,
    }
    terminal_mismatch = {
        field: {"observed": report.get(field), "expected": expected}
        for field, expected in terminal_contract.items()
        if report.get(field) != expected
    }
    if terminal_mismatch:
        raise ValueError(
            "V32 selection report lacks its terminal leakage/development contract: "
            f"{terminal_mismatch}"
        )
    arms = report.get("arms")
    if not isinstance(arms, list) or len(arms) != 11:
        raise ValueError("V32 rejection report must contain all eleven independently scored arms")
    observed_updates: list[int] = []
    eligible_updates: list[int] = []
    for index, value in enumerate(arms):
        if not isinstance(value, Mapping):
            raise TypeError(f"V32 rejection arm {index} must be a mapping")
        update = value.get("optimizer_step")
        eligible = value.get("eligible")
        if isinstance(update, bool) or not isinstance(update, int):
            raise TypeError(f"V32 rejection arm {index} lacks an integer optimizer step")
        if not isinstance(eligible, bool):
            raise TypeError(f"V32 rejection arm {index} lacks a boolean eligible result")
        observed_updates.append(update)
        if eligible:
            eligible_updates.append(update)
    if observed_updates != list(range(0, 81, 8)):
        raise ValueError("V32 rejection arms must be exactly optimizer steps 0,8,...,80")
    passed = report.get("passed")
    if not isinstance(passed, bool):
        raise TypeError("V32 selector report lacks a boolean passed field")
    development_passed = report.get("development_selection_passed")
    if not isinstance(development_passed, bool) or development_passed != passed:
        raise ValueError("V32 passed fields do not agree")
    selected_update = report.get("selected_update")
    selected_checkpoint = report.get("selected_checkpoint")
    if passed:
        if len(eligible_updates) == 0 or selected_update not in eligible_updates:
            raise ValueError("Passing V32 report does not select an eligible arm")
        if not isinstance(selected_checkpoint, str) or not selected_checkpoint:
            raise ValueError("Passing V32 report lacks its selected checkpoint")
        if Path(selected_checkpoint).name != f"update_{selected_update:03d}":
            raise ValueError("V32 selected checkpoint/update fields disagree")
    elif eligible_updates or selected_update is not None or selected_checkpoint is not None:
        raise ValueError("Rejected V32 report still contains an eligible or selected arm")
    promotion = report.get("chat_promotion")
    if not isinstance(promotion, Mapping) or not isinstance(promotion.get("eligible"), bool):
        raise TypeError("V32 report lacks a complete chat-promotion result")
    if report.get("chat_promotion_eligible") is not promotion.get("eligible"):
        raise ValueError("V32 chat-promotion fields do not agree")
    if not passed and promotion.get("eligible") is not False:
        raise ValueError("Rejected V32 report cannot be chat-promotion eligible")
    return {
        "status": "passed" if passed else "rejected",
        "report": str(path),
        "report_sha256": observed_sha,
        "training_authorized": not passed,
    }


def require_v32_rejection(config: Mapping[str, Any]) -> dict[str, Any]:
    status = v32_rejection_status(config)
    if status["training_authorized"] is not True:
        raise RuntimeError(
            "V33 is conditional on the pinned audited V32 rejection; "
            f"current V32 status is {status['status']}"
        )
    return status


def assert_deferred_final_scenes_absent(config: Mapping[str, Any]) -> None:
    """Prove V33 cannot inherit an accidental scenes-25--30 footprint."""

    contract = v33_contract(config)
    paths = config.get("paths")
    if not isinstance(paths, Mapping):
        raise TypeError("V33 config paths must be a mapping")
    data_root = _resolve(str(paths["data_root"]))
    roots = (
        data_root / "oracle",
        data_root / "rendered",
        _resolve(str(paths["features_root"])),
        _resolve(str(paths["maps_root"])),
    )
    footprints = [
        str(candidate)
        for root in roots
        for scene_id in contract.v31.deferred_final_scene_ids
        for candidate in (root / scene_id,)
        if candidate.exists() or candidate.is_symlink()
    ]
    if footprints:
        raise RuntimeError(f"V33 refuses an existing deferred-final footprint: {footprints}")


def _saved_checkpoint_files(step: int) -> tuple[str, ...]:
    common = (
        "adapter.safetensors",
        TRAINING_METADATA_FILENAME,
        RUNTIME_METADATA_FILENAME,
    )
    return common if step == 0 else (*common, "optimizer.pt")


def latest_v33_resume_checkpoint(output: Path, contract: V33Contract) -> Path | None:
    """Return the latest contiguous, complete V33 arm in an interrupted root."""

    if not output.exists():
        return None
    if output.is_symlink() or not output.is_dir():
        raise ValueError(f"V33 output root must be a real directory: {output}")
    observed_directories = sorted(path for path in output.glob("update_*") if path.is_dir())
    parsed: dict[int, Path] = {}
    for path in observed_directories:
        match = _UPDATE_DIRECTORY.fullmatch(path.name)
        if match is None:
            raise ValueError(f"V33 output contains an unrecognized update directory: {path.name}")
        step = int(match.group(1))
        if step not in contract.saved_optimizer_steps:
            raise ValueError(f"V33 output contains an unauthorized saved step: {path.name}")
        if path.is_symlink():
            raise ValueError(f"V33 update checkpoint must not be a symlink: {path}")
        for filename in _saved_checkpoint_files(step):
            candidate = path / filename
            if candidate.is_symlink():
                raise ValueError(f"V33 checkpoint file must not be a symlink: {candidate}")
        parsed[step] = path

    complete = [
        step
        for step in contract.saved_optimizer_steps
        if step in parsed
        and all((parsed[step] / filename).is_file() for filename in _saved_checkpoint_files(step))
    ]
    expected_prefix = list(contract.saved_optimizer_steps[: len(complete)])
    if complete != expected_prefix:
        raise ValueError(
            "V33 complete checkpoints are not a contiguous saved-step prefix: "
            f"observed={complete} expected={expected_prefix}"
        )
    return None if not complete else parsed[complete[-1]]


def _optimizer_checkpoint_step(path: Path, expected_step: int, settings: V33Settings) -> None:
    optimizer_path = path / "optimizer.pt"
    payload = torch.load(optimizer_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise TypeError("V33 resume optimizer state must be a mapping")
    groups = payload.get("param_groups")
    state = payload.get("state")
    if not isinstance(groups, list) or len(groups) != 3 or not isinstance(state, Mapping):
        raise ValueError("V33 resume optimizer must contain exactly three parameter groups")
    expected_groups = (
        ("dense_sidecar_adapter.output", settings.output_learning_rate, 2),
        ("dense_sidecar_adapter.sidecar_hidden", settings.hidden_learning_rate, 4),
        ("dense_sidecar_adapter.position", settings.position_learning_rate, 2),
    )
    parameter_ids: list[Any] = []
    for index, (group, (expected_name, expected_lr, expected_parameter_count)) in enumerate(
        zip(groups, expected_groups, strict=True)
    ):
        if not isinstance(group, Mapping):
            raise TypeError(f"V33 optimizer group {index} must be a mapping")
        if (
            group.get("name") != expected_name
            or float(group.get("lr", math.nan)) != expected_lr
            or float(group.get("weight_decay", math.nan)) != settings.weight_decay
        ):
            raise ValueError(f"V33 optimizer group {index} changed its locked hyperparameters")
        raw_parameters = group.get("params")
        if not isinstance(raw_parameters, list) or len(raw_parameters) != expected_parameter_count:
            raise ValueError(f"V33 optimizer group {index} parameter count changed")
        parameter_ids.extend(raw_parameters)
    if len(parameter_ids) != len(set(parameter_ids)) or set(state) != set(parameter_ids):
        raise ValueError("V33 optimizer state does not cover each trainable tensor exactly once")
    observed_steps: set[int] = set()
    for parameter_id in parameter_ids:
        entry = state[parameter_id]
        if not isinstance(entry, Mapping):
            raise TypeError("V33 optimizer parameter state must be a mapping")
        if set(entry) != {"step", "exp_avg", "exp_avg_sq"}:
            raise ValueError("V33 Adam resume state fields changed")
        for moment_name in ("exp_avg", "exp_avg_sq"):
            moment = entry[moment_name]
            if not isinstance(moment, torch.Tensor) or not torch.isfinite(moment).all():
                raise ValueError(f"V33 Adam resume {moment_name} is invalid")
        raw_step = entry.get("step")
        if isinstance(raw_step, torch.Tensor):
            if raw_step.numel() != 1:
                raise ValueError("V33 optimizer Adam step must be scalar")
            raw_step = raw_step.item()
        if isinstance(raw_step, bool) or not isinstance(raw_step, (int, float)):
            raise TypeError("V33 optimizer Adam step must be numeric")
        parsed_step = int(raw_step)
        if float(raw_step) != parsed_step:
            raise ValueError("V33 optimizer Adam step must be integral")
        observed_steps.add(parsed_step)
    if observed_steps != {expected_step}:
        raise ValueError(
            f"V33 resume optimizer does not prove step {expected_step}: "
            f"observed={sorted(observed_steps)}"
        )


def validate_v33_resume_checkpoint(
    *,
    config: Mapping[str, Any],
    output: Path,
    resume: Path,
    contract: V33Contract,
    settings: V33Settings,
    condition: Mapping[str, Any],
    schedule_audit: Mapping[str, Any],
    cache_audit: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate an interrupted-run checkpoint before loading any of its tensors."""

    latest = latest_v33_resume_checkpoint(output, contract)
    if latest is None or latest.resolve() != resume.resolve():
        raise ValueError(f"V33 resume must use the latest complete saved arm: latest={latest}")
    match = _UPDATE_DIRECTORY.fullmatch(resume.name)
    if match is None:
        raise ValueError("V33 resume path is not an update checkpoint")
    step = int(match.group(1))
    metadata = json.loads((resume / TRAINING_METADATA_FILENAME).read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise TypeError("V33 resume metadata must be a JSON object")
    runtime_metadata = json.loads((resume / RUNTIME_METADATA_FILENAME).read_text(encoding="utf-8"))
    if not isinstance(runtime_metadata, dict):
        raise TypeError("V33 resume runtime metadata must be a JSON object")
    validate_runtime_checkpoint_metadata(runtime_metadata)
    if runtime_metadata != runtime_checkpoint_metadata(metadata):
        raise ValueError("V33 resume runtime/training metadata mismatch")
    if metadata.get("config_hash") != config_hash(dict(config)):
        raise ValueError("V33 resume config hash changed")
    if metadata.get("optimizer_step") != step:
        raise ValueError("V33 resume metadata/checkpoint step mismatch")
    v30 = metadata.get("v30_joint_pair")
    v33 = metadata.get("v33_environmental")
    if not isinstance(v30, Mapping) or not isinstance(v33, Mapping):
        raise TypeError("V33 resume metadata lacks its training contracts")
    if (
        tuple(v30.get("train_scene_ids", ())) != contract.v31.train_scene_ids
        or tuple(v30.get("validation_scene_ids", ())) != contract.v31.validation_scene_ids
        or v30.get("final_test_scene_ids_loaded") != []
        or v30.get("oracle_environment_files_loaded") is not False
    ):
        raise ValueError("V33 resume metadata changed its data/leakage boundary")
    saved_cache = v30.get("scene_cache")
    if not isinstance(saved_cache, Mapping):
        raise TypeError("V33 resume metadata lacks source-prefix cache provenance")
    cache_provenance_fields = (
        "scene_count",
        "exact_source_scene_prefixes",
        "derived_source_prefixes_recomputed_bit_exact",
        "deterministically_derived_source_scene_ids",
        "historically_pinned_source_scene_ids",
        "source_prefix_sha256_by_scene",
        "loaded_environment_files",
    )
    if any(saved_cache.get(field) != cache_audit.get(field) for field in cache_provenance_fields):
        raise ValueError("V33 resume source-prefix cache provenance changed")
    saved_condition = v33.get("conditional_v32_rejection")
    saved_schedule = v33.get("schedule")
    if not isinstance(saved_condition, Mapping) or not isinstance(saved_schedule, Mapping):
        raise TypeError("V33 resume metadata lacks condition or schedule provenance")
    if (
        saved_condition.get("status") != "rejected"
        or saved_condition.get("training_authorized") is not True
        or saved_condition.get("report_sha256") != condition.get("report_sha256")
    ):
        raise ValueError("V33 resume V32-rejection provenance changed")
    if (
        saved_schedule.get("schedule_sha256") != schedule_audit.get("schedule_sha256")
        or saved_schedule.get("optimizer_step_count") != contract.optimizer_steps
    ):
        raise ValueError("V33 resume schedule changed")
    if (
        v33.get("optimizer_step") != step
        or v33.get("exact_trainable_parameter_count") != 404_608
        or tuple(v33.get("train_scene_ids", ())) != contract.v31.train_scene_ids
        or tuple(v33.get("validation_scene_ids", ())) != contract.v31.validation_scene_ids
        or v33.get("deferred_final_scene_ids_loaded") != []
    ):
        raise ValueError("V33 nested resume contract changed")
    history = metadata.get("history")
    if not isinstance(history, list) or len(history) != step + 1:
        raise ValueError("V33 resume history must contain one row per true optimizer step")
    for expected_step, row in enumerate(history):
        if not isinstance(row, Mapping) or row.get("optimizer_update") != expected_step:
            raise ValueError("V33 resume history is not contiguous")
        if expected_step > 0 and row.get("true_optimizer_step") is not True:
            raise ValueError("V33 resume history contains an unproven optimizer step")
        should_have_validation = expected_step in contract.saved_optimizer_steps
        if (row.get("validation_answer_token_nll") is not None) != should_have_validation:
            raise ValueError("V33 resume validation history does not match saved-step intervals")
        if should_have_validation and row.get("validation_pair_metrics") is None:
            raise ValueError("V33 resume saved arm lacks validation pair metrics")
        if should_have_validation and (
            row.get("adapted_prefix_separation") is None
            or row.get("adapted_prefix_separation_ratios_from_update0") is None
            or row.get("validation_family_teacher_metrics") is None
        ):
            raise ValueError("V33 resume saved arm lacks environmental diagnostics")
        if expected_step > 0 and row.get("separate_group_clipping") is not True:
            raise ValueError("V33 resume history lacks separate group clipping proof")
    if step >= 64:
        update64_metrics = history[64]["validation_family_teacher_metrics"]
        update0_metrics = history[0]["validation_family_teacher_metrics"]
        if not isinstance(update64_metrics, Mapping) or not isinstance(update0_metrics, Mapping):
            raise TypeError("V33 resume update-64 family metrics must be mappings")
        gate = update64_environmental_gate(
            update64_metrics,
            update0_metrics,
        )
        if gate["passed"] is not True:
            raise RuntimeError(
                "V33 cannot resume beyond its failed update-64 environmental gate; "
                "the saved update_064 remains valid evidence for a manual conditional follow-up"
            )
    best_update = metadata.get("best_epoch")
    best_validation = metadata.get("best_monitor_loss")
    if (
        isinstance(best_update, bool)
        or not isinstance(best_update, int)
        or best_update not in contract.saved_optimizer_steps
        or best_update > step
    ):
        raise ValueError("V33 resume best-update metadata is invalid")
    if isinstance(best_validation, bool) or not isinstance(best_validation, (int, float)):
        raise TypeError("V33 resume best validation must be numeric")
    observed_validation = history[best_update].get("validation_answer_token_nll")
    if (
        not math.isfinite(float(best_validation))
        or observed_validation is None
        or float(observed_validation) != float(best_validation)
    ):
        raise ValueError("V33 resume best checkpoint/validation metadata disagrees")
    saved_validation = [
        (index, float(row["validation_answer_token_nll"]))
        for index, row in enumerate(history)
        if row.get("validation_answer_token_nll") is not None
    ]
    expected_best = min(saved_validation, key=lambda item: (item[1], item[0]))
    if expected_best != (best_update, float(best_validation)):
        raise ValueError("V33 resume best checkpoint is not the best saved validation arm")
    if step > 0:
        _optimizer_checkpoint_step(resume, step, settings)
    return metadata


def build_v33_environmental_schedule(
    records: Sequence[QARecord],
    pair_units: Sequence[CounterfactualPairUnit],
    *,
    settings: V33Settings,
    seed: int,
) -> tuple[list[V33Environmental], dict[str, Any]]:
    """Build deterministic balanced broad+atomic-pair true optimizer steps."""

    if not pair_units:
        raise ValueError("V33 schedule requires changed-answer pair units")
    broad_count = settings.optimizer_steps * settings.broad_batch_size
    broad = select_balanced_broad_records(
        records,
        count=broad_count,
        seed=seed,
        exclude_expected_change=settings.broad_exclude_expected_change,
    )
    broad_batches = [
        tuple(broad[offset : offset + settings.broad_batch_size])
        for offset in range(0, len(broad), settings.broad_batch_size)
    ]
    if len(broad_batches) != settings.optimizer_steps:
        raise RuntimeError("V33 broad schedule does not have exactly one batch per update")
    if any(len({record.scene_id for record in batch}) != 1 for batch in broad_batches):
        # The locked batch size is one. Keep this guard for future controlled
        # extensions because cached_broad_answer_nll is single-scene.
        raise RuntimeError("V33 broad microbatch crossed scene boundaries")

    canonical_units = sorted(pair_units, key=lambda unit: (unit.pair_id, unit.question_key))
    scheduled_units: list[CounterfactualPairUnit] = []
    round_index = 0
    while len(scheduled_units) < settings.optimizer_steps * settings.pair_units_per_step:
        current = list(canonical_units)
        random.Random(seed + 10_000 + round_index).shuffle(current)
        scheduled_units.extend(current)
        round_index += 1
    scheduled_units = scheduled_units[: settings.optimizer_steps * settings.pair_units_per_step]
    pair_batches = [
        tuple(scheduled_units[offset : offset + settings.pair_units_per_step])
        for offset in range(0, len(scheduled_units), settings.pair_units_per_step)
    ]
    appearances = Counter((unit.pair_id, unit.question_key) for unit in scheduled_units)
    if set(appearances) != {(unit.pair_id, unit.question_key) for unit in pair_units}:
        raise RuntimeError("V33 pair schedule omitted one or more atomic units")
    if set(appearances.values()) != {4}:
        raise RuntimeError("V33 must recur each of the 25 pair units exactly four times")
    steps = [
        V33Environmental(
            optimizer_step=index + 1,
            broad_records=broad_batches[index],
            pair_units=pair_batches[index],
        )
        for index in range(settings.optimizer_steps)
    ]
    schedule_payload = [
        {
            "optimizer_step": step.optimizer_step,
            "broad": [(record.scene_id, record.question_id) for record in step.broad_records],
            "pairs": [(unit.pair_id, unit.question_key) for unit in step.pair_units],
        }
        for step in steps
    ]
    audit = {
        "schema_version": 1,
        "optimizer_step_count": len(steps),
        "true_optimizer_step_per_schedule_row": True,
        "broad_records_per_step": settings.broad_batch_size,
        "pair_units_per_step": settings.pair_units_per_step,
        "broad_answer_type_counts": dict(sorted(Counter(r.answer_type for r in broad).items())),
        "broad_expected_change_excluded": settings.broad_exclude_expected_change,
        "pair_unit_count": len(pair_units),
        "pair_unit_minimum_recurrence": min(appearances.values()),
        "pair_unit_maximum_recurrence": max(appearances.values()),
        "every_pair_unit_recurred": True,
        "pair_units_atomic": True,
        "schedule_sha256": hashlib.sha256(
            json.dumps(schedule_payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "questions_or_answers_serialized_to_runtime": False,
    }
    return steps, audit


def _compat_v30_settings(settings: V33Settings) -> V30Settings:
    """Populate legacy provenance fields; V33 metadata overrides its surface."""

    return V30Settings(
        enabled=settings.enabled,
        max_optimizer_steps=settings.optimizer_steps,
        evaluation_interval_steps=settings.checkpoint_interval_steps,
        broad_questions_per_cycle=settings.optimizer_steps * settings.broad_batch_size,
        broad_batch_size=settings.broad_batch_size,
        broad_exclude_expected_change=settings.broad_exclude_expected_change,
        pair_repeats_per_cycle=4,
        pair_units_per_batch=settings.pair_units_per_step,
        broad_nll_weight=settings.broad_nll_weight,
        pair_language_nll_weight=settings.pair_language_nll_weight,
        pair_margin_weight=settings.pair_margin_weight,
        pair_margin=settings.pair_margin,
        sidecar_learning_rate=settings.output_learning_rate,
        decoder_learning_rate=0.0,
        weight_decay=settings.weight_decay,
        gradient_clip_norm=max(
            settings.output_gradient_clip_norm,
            settings.hidden_gradient_clip_norm,
            settings.position_gradient_clip_norm,
        ),
        minimum_answer_types=settings.minimum_answer_types,
        trainable_bank="extension_v30_joint_pair_query",
    )


def _named_sidecar_groups(
    module: DenseSidecarAdapter,
) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    named = dict(module.named_parameters())
    if not {*_TRAINABLE_NAMES, *_FROZEN_BASE_NAMES}.issubset(named):
        raise RuntimeError("Dense sidecar lacks V33's audited environmental surfaces")
    return (
        [named[name] for name in _OUTPUT_NAMES],
        [named[name] for name in _HIDDEN_NAMES],
        [named[name] for name in _POSITION_NAMES],
    )


def freeze_for_v33(bundle: V30Bundle) -> list[torch.nn.Parameter]:
    """Freeze Gemma, LoRA, and all non-environmental parameters."""

    bundle.language.model.requires_grad_(False)
    for module in bundle.checkpoint_modules.values():
        module.requires_grad_(False)
        module.eval()
    groups = _named_sidecar_groups(bundle.dense_sidecar_adapter)
    for parameter in (parameter for group in groups for parameter in group):
        parameter.requires_grad_(True)
    bundle.dense_sidecar_adapter.train()
    return [parameter for group in groups for parameter in group]


def assert_v33_trainable_surface(
    bundle: V30Bundle, optimizer: torch.optim.Optimizer | None = None
) -> dict[str, Any]:
    named = dict(bundle.dense_sidecar_adapter.named_parameters())
    authorized_ids = {id(named[name]) for name in _TRAINABLE_NAMES}
    observed_ids = {
        id(parameter)
        for module in bundle.checkpoint_modules.values()
        for parameter in module.parameters()
        if parameter.requires_grad
    }
    if observed_ids != authorized_ids:
        raise RuntimeError("V33 trainable surface includes a frozen or omits an authorized tensor")
    if any(parameter.requires_grad for parameter in bundle.language.model.parameters()):
        raise RuntimeError("V33 Gemma decoder must remain fully frozen")
    if any(
        parameter.requires_grad
        for bank in bundle.lora_installation.banks
        for parameter in bank.installation.parameters()
    ):
        raise RuntimeError("V33 every LoRA bank must remain fully frozen")
    if optimizer is not None:
        optimizer_ids = {
            id(parameter) for group in optimizer.param_groups for parameter in group["params"]
        }
        if optimizer_ids != authorized_ids:
            raise RuntimeError("V33 optimizer contains an unauthorized parameter")
    counts = {
        "output": sum(named[name].numel() for name in _OUTPUT_NAMES),
        "sidecar_hidden": sum(named[name].numel() for name in _HIDDEN_NAMES),
        "position": sum(named[name].numel() for name in _POSITION_NAMES),
    }
    if counts != {"output": 198_144, "sidecar_hidden": 199_808, "position": 6_656}:
        raise RuntimeError(f"V33 environmental parameter counts changed: {counts}")
    return {
        "parameter_names": [f"dense_sidecar_adapter.{name}" for name in _TRAINABLE_NAMES],
        "group_parameter_counts": counts,
        "total_parameter_count": sum(counts.values()),
        "gemma_decoder_frozen": True,
        "all_lora_banks_frozen": True,
        "base_norm_and_projection_frozen": all(
            not named[name].requires_grad for name in _FROZEN_BASE_NAMES
        ),
        "every_other_parameter_frozen": True,
    }


def frozen_v33_state_sha256(bundle: V30Bundle) -> str:
    state = {
        f"{module_name}.{name}": value
        for module_name, module in bundle.checkpoint_modules.items()
        for name, value in module.state_dict().items()
        if not (module_name == "dense_sidecar_adapter" and name in _TRAINABLE_NAMES)
    }
    return tensor_state_sha256(state)


def assert_frozen_v33_state(bundle: V30Bundle, expected: str) -> None:
    observed = frozen_v33_state_sha256(bundle)
    if observed != expected:
        raise RuntimeError(f"V33 frozen inherited state changed: {expected} != {observed}")


def prefix_separation_diagnostics(
    *,
    units: Sequence[CounterfactualPairUnit],
    caches: Mapping[str, CachedPreSidecarScene],
    bundle: V30Bundle,
) -> dict[str, Any]:
    """Measure actual adapted continuous-prefix distances, never QA text."""

    pair_scenes: dict[str, tuple[str, str]] = {}
    for unit in units:
        current = tuple(unit.scene_ids)
        previous = pair_scenes.setdefault(unit.pair_id, current)
        if previous != current:
            raise ValueError(f"V33 pair {unit.pair_id} changed scene membership")
    if set(pair_scenes) != set(_VALIDATION_FAMILY_PAIR_IDS.values()):
        raise ValueError("V33 validation prefix diagnostics require the three locked families")
    scene_ids = sorted({scene_id for scenes in pair_scenes.values() for scene_id in scenes})
    model_dtype = next(bundle.language.model.parameters()).dtype
    prefixes: dict[str, torch.Tensor] = {}
    with torch.inference_mode():
        for scene_id in scene_ids:
            tokens = adapted_scene_tokens(caches[scene_id], bundle)
            prefixes[scene_id] = (
                bundle.composer.scene_prefix(tokens.to(model_dtype)).detach().float().cpu()
            )

    def rms(left: str, right: str) -> float:
        return float((prefixes[left] - prefixes[right]).square().mean().sqrt())

    by_family = {
        family: rms(*pair_scenes[pair_id])
        for family, pair_id in _VALIDATION_FAMILY_PAIR_IDS.items()
    }
    paired = {frozenset(scenes) for scenes in pair_scenes.values()}
    unrelated = [
        rms(left, right)
        for left_index, left in enumerate(scene_ids)
        for right in scene_ids[left_index + 1 :]
        if frozenset((left, right)) not in paired
    ]
    if len(unrelated) != 12 or any(not math.isfinite(value) for value in unrelated):
        raise RuntimeError("V33 unrelated adapted-prefix separation audit is incomplete")
    return {
        "schema_version": 1,
        "tensor": "composed_adapted_continuous_scene_prefix",
        "rms_by_validation_family": by_family,
        "weak_pair_mean_rms": (by_family["book_support"] + by_family["picture_support"]) / 2.0,
        "unrelated_pair_count": len(unrelated),
        "unrelated_mean_rms": sum(unrelated) / len(unrelated),
        "question_inputs_used": False,
        "all_validation_scenes_processed": True,
    }


def prefix_separation_ratios(
    current: Mapping[str, Any], baseline: Mapping[str, Any]
) -> dict[str, float]:
    current_families = current.get("rms_by_validation_family")
    baseline_families = baseline.get("rms_by_validation_family")
    if not isinstance(current_families, Mapping) or not isinstance(baseline_families, Mapping):
        raise TypeError("V33 prefix diagnostics lack family RMS values")

    def ratio(numerator: object, denominator: object, field: str) -> float:
        top = float(numerator)
        bottom = float(denominator)
        if not math.isfinite(top) or not math.isfinite(bottom) or bottom <= 0.0:
            raise ValueError(f"V33 invalid prefix separation ratio for {field}")
        return top / bottom

    return {
        "book_support": ratio(
            current_families["book_support"],
            baseline_families["book_support"],
            "book_support",
        ),
        "mirror_lr": ratio(
            current_families["mirror_lr"], baseline_families["mirror_lr"], "mirror_lr"
        ),
        "picture_support": ratio(
            current_families["picture_support"],
            baseline_families["picture_support"],
            "picture_support",
        ),
        "weak_pair_mean": ratio(
            current["weak_pair_mean_rms"], baseline["weak_pair_mean_rms"], "weak_pair_mean"
        ),
        "unrelated_mean": ratio(
            current["unrelated_mean_rms"],
            baseline["unrelated_mean_rms"],
            "unrelated_mean",
        ),
    }


def validation_family_teacher_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    rows = metrics.get("margins_by_unit")
    if not isinstance(rows, list):
        raise TypeError("V33 validation pair metrics lack margins_by_unit")
    grouped: dict[str, list[list[float]]] = {family: [] for family in _VALIDATION_FAMILY_PAIR_IDS}
    reverse = {pair_id: family for family, pair_id in _VALIDATION_FAMILY_PAIR_IDS.items()}
    for row in rows:
        if not isinstance(row, Mapping) or row.get("pair_id") not in reverse:
            raise ValueError("V33 validation pair metrics contain an unknown family")
        margins = row.get("margins")
        if not isinstance(margins, list) or len(margins) != 2:
            raise ValueError("V33 validation family margins must contain two sides")
        grouped[reverse[str(row["pair_id"])]].append([float(value) for value in margins])
    if any(len(values) != 4 for values in grouped.values()):
        raise ValueError("V33 validation families must each contain exactly four units")
    return {
        family: {
            "unit_count": len(values),
            "complete_units": sum(all(side > 0.0 for side in pair) for pair in values),
            "mean_margin": sum(side for pair in values for side in pair) / (2 * len(values)),
        }
        for family, values in grouped.items()
    }


def update64_environmental_gate(
    current: Mapping[str, Any], baseline: Mapping[str, Any]
) -> dict[str, bool]:
    if not all(
        isinstance(current.get(family), Mapping) and isinstance(baseline.get(family), Mapping)
        for family in _VALIDATION_FAMILY_PAIR_IDS
    ):
        raise TypeError("V33 update-64 family evidence is incomplete")
    nonmirror_complete = sum(
        int(current[family]["complete_units"]) for family in ("book_support", "picture_support")
    )
    result = {
        "nonmirror_teacher_complete": nonmirror_complete >= 1,
        "book_advantage_positive": (
            float(current["book_support"]["mean_margin"])
            > float(baseline["book_support"]["mean_margin"])
        ),
        "picture_advantage_positive": (
            float(current["picture_support"]["mean_margin"])
            > float(baseline["picture_support"]["mean_margin"])
        ),
    }
    return {**result, "passed": all(result.values())}


def _optimizer(bundle: V30Bundle, settings: V33Settings) -> torch.optim.AdamW:
    output, hidden, position = _named_sidecar_groups(bundle.dense_sidecar_adapter)
    optimizer = torch.optim.AdamW(
        [
            {
                "name": "dense_sidecar_adapter.output",
                "params": output,
                "lr": settings.output_learning_rate,
                "weight_decay": settings.weight_decay,
            },
            {
                "name": "dense_sidecar_adapter.sidecar_hidden",
                "params": hidden,
                "lr": settings.hidden_learning_rate,
                "weight_decay": settings.weight_decay,
            },
            {
                "name": "dense_sidecar_adapter.position",
                "params": position,
                "lr": settings.position_learning_rate,
                "weight_decay": settings.weight_decay,
            },
        ]
    )
    assert_v33_trainable_surface(bundle, optimizer)
    return optimizer


def _metadata(
    *,
    bundle: V30Bundle,
    settings: V33Settings,
    cache_audit: Mapping[str, Any],
    qa_audit: Mapping[str, Any],
    schedule_audit: Mapping[str, Any],
    condition: Mapping[str, Any],
    frozen_hash: str,
    update_zero: Mapping[str, Any],
    train_records: Sequence[QARecord],
    validation_records: Sequence[QARecord],
    pair_units: Sequence[CounterfactualPairUnit],
    history: Sequence[Mapping[str, Any]],
    optimizer_step: int,
    best_update: int,
    best_validation: float,
    trainable_surface: Mapping[str, Any],
    prefix_diagnostics: Mapping[str, Any],
    prefix_ratios: Mapping[str, Any],
    family_teacher_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    metadata = _v30_metadata(
        bundle=bundle,
        settings=_compat_v30_settings(settings),
        cache_audit=cache_audit,
        qa_audit=qa_audit,
        frozen_hash=frozen_hash,
        update_zero=update_zero,
        train_records=train_records,
        validation_records=validation_records,
        pair_units=pair_units,
        history=history,
        optimizer_step=optimizer_step,
        best_update=best_update,
        best_validation=best_validation,
        trainable_surface=trainable_surface,
    )
    metadata["v33_environmental"] = {
        "schema_version": 1,
        "artifact": "v33_diverse28_true_environmental_training",
        "optimizer_step": optimizer_step,
        "settings": settings.__dict__,
        "schedule": dict(schedule_audit),
        "conditional_v32_rejection": dict(condition),
        "source_is_approved_v29_update_004": True,
        "exact_trainable_parameter_count": trainable_surface["total_parameter_count"],
        "train_scene_ids": sorted({record.scene_id for record in train_records}),
        "validation_scene_ids": sorted({record.scene_id for record in validation_records}),
        "deferred_final_scene_ids_loaded": [],
        "gemma_decoder_frozen": True,
        "all_lora_banks_frozen": True,
        "base_norm_and_projection_frozen": True,
        "adapted_prefix_separation": dict(prefix_diagnostics),
        "adapted_prefix_separation_ratios_from_update0": dict(prefix_ratios),
        "validation_family_teacher_metrics": dict(family_teacher_metrics),
        "greedy_screen_required": optimizer_step in {32, 64, 100},
        "new_train_prefix_scene_ids": list(_NEW_TRAIN_SCENES),
        "every_saved_arm_requires_independent_selection": True,
        "development_progress_is_not_chat_promotion": True,
        "question_dependent_scene_processing": False,
        "question_dependent_retrieval": False,
        "oracle_environment_files_loaded": False,
    }
    legacy = metadata["v30_joint_pair"]
    legacy["objective"] = "environmental_sidecar_only_broad_plus_atomic_pair_margin"
    legacy["trainable_surface"] = dict(trainable_surface)
    legacy["fresh_bank"] = None
    legacy["fresh_bank_parameter_count"] = 0
    legacy["all_inherited_lora_banks_frozen"] = True
    legacy["all_sidecar_hidden_parameters_frozen"] = False
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


def preflight_v33(
    config: Mapping[str, Any],
    *,
    require_qa: bool = True,
    require_rejection: bool = False,
) -> ApprovedV29Source:
    v33_contract(config)
    source = require_approved_v29_source(config)
    if require_qa:
        load_v31_qa_records(config)
    if require_rejection:
        require_v32_rejection(config)
        assert_deferred_final_scenes_absent(config)
    return source


def run_v33(*, config: dict[str, Any], output: Path, resume: Path | None = None) -> dict[str, Any]:
    """Run 100 true micro-updates; always restart from approved V29 update 004."""

    if resume is None and output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty V33 output: {output}")
    contract = v33_contract(config)
    settings = v33_settings(config)
    source = preflight_v33(config, require_qa=True, require_rejection=True)
    condition = require_v32_rejection(config)
    assert_deferred_final_scenes_absent(config)
    seed = int(config["seed"])
    torch.manual_seed(seed)
    random.seed(seed)
    train_records, validation_records, qa_audit = load_v31_qa_records(config)
    if {record.scene_id for record in train_records} & {
        record.scene_id for record in validation_records
    }:
        raise ValueError("V33 train/validation scenes overlap")
    if len({record.answer_type for record in train_records}) < settings.minimum_answer_types:
        raise ValueError("V33 broad training lacks required answer-type coverage")
    train_pairs = build_exact_question_pair_units(train_records)
    validation_pairs = build_exact_question_pair_units(validation_records)
    if len(train_pairs) != contract.v31.train_changed_pair_unit_count:
        raise ValueError("V33 training pair count differs from its diverse28 contract")
    if len(validation_pairs) != 12:
        raise ValueError("V33 validation pair count differs from its locked contract")
    schedule, schedule_audit = build_v33_environmental_schedule(
        train_records,
        train_pairs,
        settings=settings,
        seed=seed,
    )
    if schedule_audit["pair_unit_minimum_recurrence"] < contract.minimum_pair_unit_recurrence:
        raise RuntimeError("V33 schedule did not recur every pair unit enough times")

    bundle = load_v30_bundle(config, source)
    scene_ids = sorted(
        {record.scene_id for record in train_records}
        | {record.scene_id for record in validation_records}
    )
    caches, cache_audit = cache_pre_sidecar_scenes(
        bundle,
        scene_ids,
        allow_unpinned_source_scene_ids=_NEW_TRAIN_SCENES,
    )
    if tuple(cache_audit["deterministically_derived_source_scene_ids"]) != _NEW_TRAIN_SCENES:
        raise RuntimeError("V33 derived source-prefix set differs from new training scenes 31--38")
    all_trainable = freeze_for_v33(bundle)
    frozen_hash = frozen_v33_state_sha256(bundle)
    trainable_surface = assert_v33_trainable_surface(bundle)
    if trainable_surface["total_parameter_count"] != 404_608:
        raise RuntimeError("V33 changed the exact audited trainable surface")
    fresh_zero = verify_fresh_bank_update_zero(bundle)
    baseline = validation_answer_nll(
        records=validation_records,
        caches=caches,
        bundle=bundle,
        batch_size=settings.broad_batch_size,
    )
    observed_nll = float(baseline["answer_token_nll"])
    source_nll = _source_validation_nll(bundle.source_training_metadata)
    tolerance = float(v30_contract(config)["update_zero_validation_nll_absolute_tolerance"])
    if abs(observed_nll - source_nll) > tolerance:
        raise RuntimeError(
            "V33 update zero differs from approved V29 validation NLL: "
            f"source={source_nll} observed={observed_nll} tolerance={tolerance}"
        )
    baseline_pairs = validation_pair_metrics(
        units=validation_pairs,
        caches=caches,
        bundle=bundle,
        margin=settings.pair_margin,
    )
    baseline_prefix_diagnostics = prefix_separation_diagnostics(
        units=validation_pairs, caches=caches, bundle=bundle
    )
    baseline_prefix_ratios = prefix_separation_ratios(
        baseline_prefix_diagnostics, baseline_prefix_diagnostics
    )
    baseline_family_metrics = validation_family_teacher_metrics(baseline_pairs)
    update_zero = {
        "approved_v29_source": True,
        **fresh_zero,
        "exact_source_scene_prefixes": True,
        "exact_source_validation_nll": True,
        "source_validation_answer_token_nll": source_nll,
        "observed_validation_answer_token_nll": observed_nll,
        "validation_nll_absolute_tolerance": tolerance,
        "question_dependent_scene_processing": False,
        "question_dependent_retrieval": False,
        "oracle_environment_files_loaded": False,
    }
    optimizer = _optimizer(bundle, settings)
    history: list[dict[str, Any]] = [
        {
            "optimizer_update": 0,
            "validation_answer_token_nll": observed_nll,
            "validation_pair_metrics": baseline_pairs,
            "adapted_prefix_separation": baseline_prefix_diagnostics,
            "adapted_prefix_separation_ratios_from_update0": baseline_prefix_ratios,
            "validation_family_teacher_metrics": baseline_family_metrics,
            "update_0_equivalence_verified": True,
        }
    ]
    best_update = 0
    best_validation = observed_nll
    start_step = 0
    output.mkdir(parents=True, exist_ok=True)
    if resume is not None:
        resume = resume.resolve()
        resume_metadata = validate_v33_resume_checkpoint(
            config=config,
            output=output,
            resume=resume,
            contract=contract,
            settings=settings,
            condition=condition,
            schedule_audit=schedule_audit,
            cache_audit=cache_audit,
        )
        loaded_metadata = load_adapter_checkpoint(
            resume,
            bundle.checkpoint_modules,
            device="cpu",
            metadata_filename=TRAINING_METADATA_FILENAME,
        )
        if loaded_metadata != resume_metadata:
            raise RuntimeError("V33 resume metadata changed during adapter load")
        start_step = int(resume_metadata["optimizer_step"])
        if start_step > 0:
            load_optimizer_checkpoint(resume, optimizer, bundle.language.device)
        history = list(resume_metadata["history"])
        best_update = int(resume_metadata["best_epoch"])
        best_validation = float(resume_metadata["best_monitor_loss"])
        saved_equivalence = resume_metadata["v30_joint_pair"]["update_zero_equivalence"]
        if not isinstance(saved_equivalence, Mapping):
            raise TypeError("V33 resume update-zero equivalence must be a mapping")
        exact_equivalence_fields = set(update_zero) - {"observed_validation_answer_token_nll"}
        if (
            any(
                saved_equivalence.get(field) != update_zero[field]
                for field in exact_equivalence_fields
            )
            or abs(
                float(saved_equivalence.get("observed_validation_answer_token_nll", math.inf))
                - observed_nll
            )
            > tolerance
        ):
            raise ValueError("V33 resume update-zero equivalence changed")
        assert_frozen_v33_state(bundle, frozen_hash)
        bundle.lora_installation.validate_state()
        validate_dense_sidecar_adapter_state(
            bundle.dense_sidecar_adapter,
            expected_parameter_count=int(
                bundle.source_runtime_metadata["dense_sidecar_adapter_parameter_count"]
            ),
            expected_state_sha256=str(resume_metadata["dense_sidecar_adapter_state_sha256"]),
            context="V33 resume sidecar",
        )
        bank_hashes = resume_metadata.get("lora_bank_state_sha256")
        if not isinstance(bank_hashes, Mapping) or any(
            bank.installation.state_sha256() != bank_hashes.get(bank.settings.name)
            for bank in bundle.lora_installation.banks
        ):
            raise ValueError("V33 resume frozen LoRA-bank hash mismatch")
        # ``best`` is a derived convenience pointer, whereas the numbered
        # checkpoints are the causal record. Rebuild it on every resume so a
        # crash between saving an update and refreshing ``best`` cannot leave
        # a silently stale or partial runtime checkpoint.
        best_source = output / f"update_{best_update:03d}"
        if best_source.resolve() == resume:
            best_metadata = resume_metadata
        else:
            best_metadata = load_adapter_checkpoint(
                best_source,
                bundle.checkpoint_modules,
                device="cpu",
                metadata_filename=TRAINING_METADATA_FILENAME,
            )
            if best_metadata.get("optimizer_step") != best_update:
                raise ValueError("V33 numbered best checkpoint metadata disagrees")
        _save(output / "best", bundle=bundle, metadata=best_metadata, optimizer=None)
        if best_source.resolve() != resume:
            reloaded = load_adapter_checkpoint(
                resume,
                bundle.checkpoint_modules,
                device="cpu",
                metadata_filename=TRAINING_METADATA_FILENAME,
            )
            if reloaded != resume_metadata:
                raise RuntimeError("V33 resume metadata changed while rebuilding best")
            assert_frozen_v33_state(bundle, frozen_hash)
            validate_dense_sidecar_adapter_state(
                bundle.dense_sidecar_adapter,
                expected_parameter_count=int(
                    bundle.source_runtime_metadata["dense_sidecar_adapter_parameter_count"]
                ),
                expected_state_sha256=str(resume_metadata["dense_sidecar_adapter_state_sha256"]),
                context="V33 post-best-repair resume sidecar",
            )
        print(
            json.dumps(
                {
                    "phase": "v33_true_environmental_resume",
                    "optimizer_step": start_step,
                    "checkpoint": str(resume),
                }
            ),
            flush=True,
        )
    else:
        initial_metadata = _metadata(
            bundle=bundle,
            settings=settings,
            cache_audit=cache_audit,
            qa_audit=qa_audit,
            schedule_audit=schedule_audit,
            condition=condition,
            frozen_hash=frozen_hash,
            update_zero=update_zero,
            train_records=train_records,
            validation_records=validation_records,
            pair_units=train_pairs,
            history=history,
            optimizer_step=0,
            best_update=0,
            best_validation=best_validation,
            trainable_surface=trainable_surface,
            prefix_diagnostics=baseline_prefix_diagnostics,
            prefix_ratios=baseline_prefix_ratios,
            family_teacher_metrics=baseline_family_metrics,
        )
        _save(output / "update_000", bundle=bundle, metadata=initial_metadata, optimizer=None)
        _save(output / "best", bundle=bundle, metadata=initial_metadata, optimizer=None)

    all_trainable = freeze_for_v33(bundle)
    output_parameters, hidden_parameters, position_parameters = _named_sidecar_groups(
        bundle.dense_sidecar_adapter
    )
    for item in schedule[start_step:]:
        step = item.optimizer_step
        broad_record = item.broad_records[0]
        bundle.dense_sidecar_adapter.train()
        bundle.lora_installation.eval()
        optimizer.zero_grad(set_to_none=True)
        broad_loss = cached_broad_answer_nll(
            cache=caches[broad_record.scene_id],
            records=item.broad_records,
            bundle=bundle,
        )
        # Backpropagate the broad term before constructing the four-sequence
        # correct/swapped pair graph. Both gradients participate in this one
        # optimizer step, while peak MPS memory stays at the established V30
        # pair-forward footprint instead of retaining a fifth sequence graph.
        (settings.broad_nll_weight * broad_loss).backward()
        pair_language, pair_hinge, pair_diagnostics = paired_canonical_answer_objective(
            units=item.pair_units,
            caches=caches,
            bundle=bundle,
            margin=settings.pair_margin,
        )
        pair_objective = (
            settings.pair_language_nll_weight * pair_language
            + settings.pair_margin_weight * pair_hinge
        )
        pair_objective.backward()
        broad_value = float(broad_loss.detach().cpu())
        pair_language_value = float(pair_language.detach().cpu())
        pair_hinge_value = float(pair_hinge.detach().cpu())
        pair_side_accuracy = float(pair_diagnostics["side_accuracy"].detach().cpu())
        objective_value = (
            settings.broad_nll_weight * broad_value
            + settings.pair_language_nll_weight * pair_language_value
            + settings.pair_margin_weight * pair_hinge_value
        )
        del broad_loss, pair_language, pair_hinge, pair_diagnostics, pair_objective
        assert_v33_trainable_surface(bundle, optimizer)
        missing_gradients = [
            index for index, parameter in enumerate(all_trainable) if parameter.grad is None
        ]
        if missing_gradients:
            raise RuntimeError(f"V33 trainable tensors lack gradients: {missing_gradients}")
        if any(not torch.isfinite(parameter.grad).all() for parameter in all_trainable):
            raise RuntimeError("V33 trainable gradient is nonfinite")
        group_gradient_norms = {
            "output": float(
                torch.nn.utils.clip_grad_norm_(
                    output_parameters, settings.output_gradient_clip_norm
                )
                .detach()
                .cpu()
            ),
            "sidecar_hidden": float(
                torch.nn.utils.clip_grad_norm_(
                    hidden_parameters, settings.hidden_gradient_clip_norm
                )
                .detach()
                .cpu()
            ),
            "position": float(
                torch.nn.utils.clip_grad_norm_(
                    position_parameters, settings.position_gradient_clip_norm
                )
                .detach()
                .cpu()
            ),
        }
        optimizer.step()
        assert_frozen_v33_state(bundle, frozen_hash)
        bundle.lora_installation.validate_state()
        validate_dense_sidecar_adapter_state(
            bundle.dense_sidecar_adapter,
            expected_parameter_count=int(
                bundle.source_runtime_metadata["dense_sidecar_adapter_parameter_count"]
            ),
            context="V33 post-environmental sidecar",
        )
        should_save = step in contract.saved_optimizer_steps
        validation = (
            validation_answer_nll(
                records=validation_records,
                caches=caches,
                bundle=bundle,
                batch_size=settings.broad_batch_size,
            )
            if should_save
            else None
        )
        pair_validation = (
            validation_pair_metrics(
                units=validation_pairs,
                caches=caches,
                bundle=bundle,
                margin=settings.pair_margin,
            )
            if should_save
            else None
        )
        prefix_diagnostics = (
            prefix_separation_diagnostics(units=validation_pairs, caches=caches, bundle=bundle)
            if should_save
            else None
        )
        prefix_ratios = (
            prefix_separation_ratios(prefix_diagnostics, baseline_prefix_diagnostics)
            if prefix_diagnostics is not None
            else None
        )
        family_metrics = (
            validation_family_teacher_metrics(pair_validation)
            if pair_validation is not None
            else None
        )
        validation_value = None if validation is None else float(validation["answer_token_nll"])
        if validation_value is not None and validation_value < best_validation:
            best_update = step
            best_validation = validation_value
        history.append(
            {
                "optimizer_update": step,
                "true_optimizer_step": True,
                "train_broad_answer_token_nll": broad_value,
                "train_pair_answer_token_nll": pair_language_value,
                "train_pair_margin_hinge": pair_hinge_value,
                "train_pair_side_accuracy": pair_side_accuracy,
                "train_objective": objective_value,
                "validation_answer_token_nll": validation_value,
                "validation_pair_metrics": pair_validation,
                "adapted_prefix_separation": prefix_diagnostics,
                "adapted_prefix_separation_ratios_from_update0": prefix_ratios,
                "validation_family_teacher_metrics": family_metrics,
                "preclip_gradient_norm_by_group": group_gradient_norms,
                "separate_group_clipping": True,
                "saved_checkpoint": should_save,
            }
        )
        if not should_save:
            continue
        metadata = _metadata(
            bundle=bundle,
            settings=settings,
            cache_audit=cache_audit,
            qa_audit=qa_audit,
            schedule_audit=schedule_audit,
            condition=condition,
            frozen_hash=frozen_hash,
            update_zero=update_zero,
            train_records=train_records,
            validation_records=validation_records,
            pair_units=train_pairs,
            history=history,
            optimizer_step=step,
            best_update=best_update,
            best_validation=best_validation,
            trainable_surface=trainable_surface,
            prefix_diagnostics=prefix_diagnostics,
            prefix_ratios=prefix_ratios,
            family_teacher_metrics=family_metrics,
        )
        _save(output / f"update_{step:03d}", bundle=bundle, metadata=metadata, optimizer=optimizer)
        if best_update == step:
            _save(output / "best", bundle=bundle, metadata=metadata, optimizer=None)
        print(
            json.dumps(
                {
                    "phase": "v33_true_environmental_checkpoint",
                    "optimizer_step": step,
                    "validation_answer_token_nll": validation_value,
                    "validation_pair_passed_units": pair_validation["passed_units"],
                    "weak_pair_prefix_rms_ratio": prefix_ratios["weak_pair_mean"],
                    "unrelated_prefix_rms_ratio": prefix_ratios["unrelated_mean"],
                    "best_update": best_update,
                }
            ),
            flush=True,
        )
        if step == 64:
            assert family_metrics is not None
            gate = update64_environmental_gate(family_metrics, baseline_family_metrics)
            if gate["passed"] is not True:
                raise RuntimeError(
                    "V33 update-64 environmental gate failed; conditional base surface "
                    "is defined but intentionally not auto-enabled"
                )
    assert_frozen_v33_state(bundle, frozen_hash)
    return {
        "schema_version": 1,
        "artifact": "v33_diverse28_true_environmental_training",
        "output": str(output),
        "best_checkpoint": str(output / "best"),
        "best_update": best_update,
        "baseline_validation_answer_token_nll": observed_nll,
        "best_validation_answer_token_nll": best_validation,
        "optimizer_updates": settings.optimizer_steps,
        "resumed_from_optimizer_step": start_step,
        "saved_optimizer_steps": list(contract.saved_optimizer_steps),
        "trainable_surface": trainable_surface,
        "schedule": schedule_audit,
        "source_v29_selection_sha256": source.selection_sha256,
        "v32_condition": condition,
        "final_test_scene_ids_loaded": [],
        "question_dependent_scene_processing": False,
        "question_dependent_retrieval": False,
        "oracle_environment_files_loaded": False,
        "causal_selection_required_before_promotion": True,
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
    if args.preflight_only and (args.resume is not None or args.resume_latest):
        parser.error("resume options are not valid with --preflight-only")
    config = load_config(args.config)
    if args.preflight_only:
        source = preflight_v33(config, require_qa=True, require_rejection=True)
        report = {
            "artifact": "v33_diverse28_true_environmental_preflight",
            "passed": True,
            "source_v29_checkpoint": str(source.checkpoint),
            "v32_condition": v32_rejection_status(config),
            "training_starts_only_after_v32_rejection": True,
            "gemma_decoder_frozen": True,
            "all_lora_banks_frozen": True,
            "exact_trainable_parameter_count": 404_608,
            "final_test_scenes_touched": False,
        }
    else:
        output = _resolve(args.output)
        resume = None if args.resume is None else _resolve(args.resume)
        if args.resume_latest:
            resume = latest_v33_resume_checkpoint(output, v33_contract(config))
            if resume is None:
                raise FileNotFoundError(
                    f"V33 output has no complete checkpoint to resume: {output}"
                )
        report = run_v33(config=config, output=output, resume=resume)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "V33Contract",
    "V33Environmental",
    "V33Settings",
    "assert_deferred_final_scenes_absent",
    "assert_v33_trainable_surface",
    "build_v33_environmental_schedule",
    "freeze_for_v33",
    "latest_v33_resume_checkpoint",
    "prefix_separation_diagnostics",
    "prefix_separation_ratios",
    "preflight_v33",
    "require_v32_rejection",
    "run_v33",
    "v32_rejection_status",
    "v33_contract",
    "v33_settings",
    "validate_v33_resume_checkpoint",
    "validation_family_teacher_metrics",
]
