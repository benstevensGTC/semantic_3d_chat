"""Conditional V32 true-microstep repair from the approved V29 checkpoint.

V31 averages an entire broad/pair cycle into one clipped optimizer update.
V32 keeps the identical audited 329,216-parameter surface but performs one
real optimizer update for each balanced broad-record plus atomic pair unit.
It is authorized to train only after the independent V31 selector rejects all
V31 arms.  Update zero is still bit-exact to approved V29, every full scene is
cached before any question, and deferred final scenes are forbidden.
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
from semantic_3d_chat.scene_encoder.dense_sidecar_adapter import (
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
    V30Bundle,
    V30Settings,
    _sidecar_trainable_parameters,
    assert_frozen_inherited_state,
    assert_v30_trainable_surface,
    cache_pre_sidecar_scenes,
    cached_broad_answer_nll,
    freeze_for_v30,
    frozen_inherited_state_sha256,
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

DEFAULT_CONFIG = Path("configs/experiments/gemma4_diverse28_microstep_v32.yaml")
DEFAULT_OUTPUT = Path("data_gemma4/checkpoints/gemma4_v32_diverse28_microstep")
_NEW_TRAIN_SCENES = tuple(f"scene_{index:06d}" for index in range(31, 39))
_UPDATE_DIRECTORY = re.compile(r"update_([0-9]{3})")


@dataclass(frozen=True)
class V32Settings:
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
    sidecar_learning_rate: float
    decoder_learning_rate: float
    weight_decay: float
    gradient_clip_norm: float
    minimum_answer_types: int
    trainable_bank: str

    @property
    def saved_optimizer_steps(self) -> tuple[int, ...]:
        return tuple(range(0, self.optimizer_steps + 1, self.checkpoint_interval_steps))


@dataclass(frozen=True)
class V32Contract:
    v31: V31Contract
    v31_selection_report: Path
    saved_optimizer_steps: tuple[int, ...]
    optimizer_steps: int
    checkpoint_interval_steps: int
    minimum_pair_unit_recurrence: int
    development_changed_complete_pairs_minimum: int
    chat_promotion_changed_complete_pairs_minimum: int


@dataclass(frozen=True)
class V32Microstep:
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
        raise ValueError(f"v32_microstep.{field} must remain true")


def v32_settings(config: Mapping[str, Any]) -> V32Settings:
    training = config.get("training")
    if not isinstance(training, Mapping):
        raise TypeError("training must be a mapping")
    raw = training.get("v32_microstep")
    if not isinstance(raw, Mapping):
        raise TypeError("training.v32_microstep must be a mapping")
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
        "sidecar_learning_rate",
        "decoder_learning_rate",
        "weight_decay",
        "gradient_clip_norm",
        "minimum_answer_types",
        "trainable_bank",
    }
    if set(raw) != required:
        raise ValueError(
            "training.v32_microstep fields differ from the locked schema: "
            f"missing={sorted(required - set(raw))} unknown={sorted(set(raw) - required)}"
        )
    if not isinstance(raw["enabled"], bool) or not isinstance(
        raw["broad_exclude_expected_change"], bool
    ):
        raise TypeError("V32 boolean settings must be boolean")
    settings = V32Settings(
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
        sidecar_learning_rate=_finite(
            "sidecar_learning_rate", raw["sidecar_learning_rate"], positive=True
        ),
        decoder_learning_rate=_finite(
            "decoder_learning_rate", raw["decoder_learning_rate"], positive=True
        ),
        weight_decay=_finite("weight_decay", raw["weight_decay"], positive=False),
        gradient_clip_norm=_finite("gradient_clip_norm", raw["gradient_clip_norm"], positive=True),
        minimum_answer_types=_positive_int("minimum_answer_types", raw["minimum_answer_types"]),
        trainable_bank=str(raw["trainable_bank"]),
    )
    if settings.optimizer_steps % settings.checkpoint_interval_steps:
        raise ValueError("V32 optimizer steps must divide exactly into checkpoint intervals")
    return settings


def v32_contract(config: Mapping[str, Any]) -> V32Contract:
    """Validate V32 without loading QA, Gemma, or any scene artifact."""

    v31 = v31_contract(config)
    raw = config.get("v32_microstep")
    if not isinstance(raw, Mapping):
        raise TypeError("V32 requires a v32_microstep mapping")
    required = {
        "schema_version",
        "role",
        "engine",
        "source_selected_update",
        "v31_selection_report",
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
        "minimum_pair_unit_recurrence",
        "exact_trainable_parameter_count",
        "exact_zero_source_cache",
        "allow_only_new_training_scenes_to_derive_prefixes",
        "inspect_every_saved_arm",
        "strict_validation_nll_improvement",
        "old_color_mirror_no_new_negative_retention",
        "development_changed_complete_pairs_minimum",
        "chat_promotion_changed_complete_pairs_minimum",
        "chat_promotion_aggregate_exact_no_regression",
        "conditional_on_v31_rejection",
        "training_requires_v31_rejection",
    }
    if set(raw) != required:
        raise ValueError(
            "v32_microstep fields differ from the locked schema: "
            f"missing={sorted(required - set(raw))} unknown={sorted(set(raw) - required)}"
        )
    if raw["schema_version"] != 1:
        raise ValueError("v32_microstep.schema_version must be 1")
    if raw["role"] != "approved_v29_diverse28_microstep_optimizer_repair_v32":
        raise ValueError("v32_microstep.role does not authorize this experiment")
    if raw["engine"] != "v30_exact_zero_frozen_state_true_microsteps":
        raise ValueError("V32 must use the audited exact-zero microstep engine")
    settings = v32_settings(config)
    if not settings.enabled:
        raise ValueError("V32 microstep training is disabled")
    expected_settings = {
        "optimizer_steps": 80,
        "checkpoint_interval_steps": 8,
        "broad_batch_size": 1,
        "pair_units_per_step": 1,
        "broad_exclude_expected_change": True,
        "broad_nll_weight": 1.0,
        "pair_language_nll_weight": 1.0,
        "pair_margin_weight": 4.0,
        "pair_margin": 0.5,
        "sidecar_learning_rate": 2.5e-5,
        "decoder_learning_rate": 2.0e-5,
        "weight_decay": 0.0,
        "gradient_clip_norm": 1.0,
        "minimum_answer_types": 4,
        "trainable_bank": "extension_v30_joint_pair_query",
    }
    mismatches = {
        name: {"observed": getattr(settings, name), "expected": expected}
        for name, expected in expected_settings.items()
        if getattr(settings, name) != expected
    }
    if mismatches:
        raise ValueError(f"V32 locked optimizer settings changed: {mismatches}")
    expected_saved = tuple(range(0, 81, 8))
    raw_saved = raw["saved_optimizer_steps"]
    if not isinstance(raw_saved, Sequence) or isinstance(raw_saved, (str, bytes)):
        raise TypeError("v32_microstep.saved_optimizer_steps must be a sequence")
    saved = tuple(raw_saved)
    if saved != expected_saved or saved != settings.saved_optimizer_steps:
        raise ValueError("V32 saved optimizer-step arms must be 0,8,...,80")
    if (
        raw["source_selected_update"] != 4
        or raw["optimizer_steps"] != settings.optimizer_steps
        or raw["checkpoint_interval_steps"] != settings.checkpoint_interval_steps
        or raw["minimum_pair_unit_recurrence"] != 3
    ):
        raise ValueError("V32 source/optimizer schedule differs from its locked repair")
    if tuple(raw["train_scene_ids"]) != v31.train_scene_ids:
        raise ValueError("V32 training scenes differ from locked diverse28 V31")
    if tuple(raw["validation_scene_ids"]) != v31.validation_scene_ids:
        raise ValueError("V32 validation scenes must remain scenes 19--24")
    if tuple(raw["deferred_final_scene_ids"]) != v31.deferred_final_scene_ids:
        raise ValueError("V32 deferred final scenes must remain scenes 25--30")
    if (
        raw["train_question_count"] != v31.train_question_count
        or raw["validation_question_count"] != v31.validation_question_count
        or raw["train_changed_pair_unit_count"] != v31.train_changed_pair_unit_count
        or raw["validation_changed_pair_unit_count"] != 12
    ):
        raise ValueError("V32 QA or pair counts differ from locked diverse28 development data")
    for field in (
        "exact_zero_source_cache",
        "allow_only_new_training_scenes_to_derive_prefixes",
        "inspect_every_saved_arm",
        "strict_validation_nll_improvement",
        "old_color_mirror_no_new_negative_retention",
        "chat_promotion_aggregate_exact_no_regression",
        "conditional_on_v31_rejection",
        "training_requires_v31_rejection",
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
        raise ValueError("V32 development/chat gates must remain 1/12 and 6/12")
    inherited = v30_contract(config)
    if (
        inherited["source_selected_update"] != 4
        or inherited["joint_trainable_parameter_count"] != 329_216
        or raw["exact_trainable_parameter_count"] != 329_216
        or inherited["fresh_bank"] != settings.trainable_bank
    ):
        raise ValueError("V32 changed the approved V29 source or 329,216-parameter surface")
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
        raise ValueError("V32 inherited selection or chat-promotion gates were weakened")
    selection_report = _resolve(str(raw["v31_selection_report"]))
    return V32Contract(
        v31=v31,
        v31_selection_report=selection_report,
        saved_optimizer_steps=saved,
        optimizer_steps=settings.optimizer_steps,
        checkpoint_interval_steps=settings.checkpoint_interval_steps,
        minimum_pair_unit_recurrence=3,
        development_changed_complete_pairs_minimum=development_min,
        chat_promotion_changed_complete_pairs_minimum=promotion_min,
    )


def v31_rejection_status(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return a fail-closed status for V32's conditional V31 dependency."""

    contract = v32_contract(config)
    path = contract.v31_selection_report
    if path.is_symlink() or not path.is_file():
        return {
            "status": "pending",
            "report": str(path),
            "report_sha256": None,
            "training_authorized": False,
        }
    report_bytes = path.read_bytes()
    report = json.loads(report_bytes)
    if not isinstance(report, Mapping):
        raise TypeError("V31 selection report must be a JSON object")
    required = {
        "artifact": "v31_diverse28_joint_pair_development_selection",
        "all_intermediate_checkpoints_inspected": True,
        "final_test_scenes_touched": False,
    }
    mismatch = {
        field: {"observed": report.get(field), "expected": expected}
        for field, expected in required.items()
        if report.get(field) != expected
    }
    if mismatch:
        raise ValueError(f"V31 selection report is not an audited terminal result: {mismatch}")
    if tuple(report.get("validation_scene_ids", ())) != contract.v31.validation_scene_ids:
        raise ValueError("V31 selector validation scenes differ from V32's locked development set")
    if tuple(report.get("train_scene_ids", ())) != contract.v31.train_scene_ids:
        raise ValueError("V31 selector training scenes differ from V32's locked development set")
    if tuple(report.get("deferred_final_scene_ids", ())) != contract.v31.deferred_final_scene_ids:
        raise ValueError("V31 selector deferred-final scenes differ from V32's lock")
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
            "V31 selection report lacks its terminal leakage/development contract: "
            f"{terminal_mismatch}"
        )
    arms = report.get("arms")
    if not isinstance(arms, list) or len(arms) != 9:
        raise ValueError("V31 rejection report must contain all nine independently scored arms")
    observed_updates: list[int] = []
    eligible_updates: list[int] = []
    for index, value in enumerate(arms):
        if not isinstance(value, Mapping):
            raise TypeError(f"V31 rejection arm {index} must be a mapping")
        update = value.get("update")
        eligible = value.get("eligible")
        if isinstance(update, bool) or not isinstance(update, int):
            raise TypeError(f"V31 rejection arm {index} lacks an integer update")
        if not isinstance(eligible, bool):
            raise TypeError(f"V31 rejection arm {index} lacks a boolean eligible result")
        observed_updates.append(update)
        if eligible:
            eligible_updates.append(update)
    if observed_updates != list(range(9)):
        raise ValueError("V31 rejection report arms must be exactly updates 0 through 8")
    passed = report.get("passed")
    if not isinstance(passed, bool):
        raise TypeError("V31 selector report lacks a boolean passed field")
    development_passed = report.get("development_selection_passed")
    if not isinstance(development_passed, bool) or development_passed != passed:
        raise ValueError("V31 passed fields do not agree")
    selected_update = report.get("selected_update")
    selected_checkpoint = report.get("selected_checkpoint")
    if passed:
        if len(eligible_updates) == 0 or selected_update not in eligible_updates:
            raise ValueError("Passing V31 report does not select an eligible arm")
        if not isinstance(selected_checkpoint, str) or not selected_checkpoint:
            raise ValueError("Passing V31 report lacks its selected checkpoint")
        if Path(selected_checkpoint).name != f"update_{selected_update:03d}":
            raise ValueError("V31 selected checkpoint/update fields disagree")
    elif eligible_updates or selected_update is not None or selected_checkpoint is not None:
        raise ValueError("Rejected V31 report still contains an eligible or selected arm")
    promotion = report.get("chat_promotion")
    if not isinstance(promotion, Mapping) or not isinstance(promotion.get("eligible"), bool):
        raise TypeError("V31 report lacks a complete chat-promotion result")
    if report.get("chat_promotion_eligible") is not promotion.get("eligible"):
        raise ValueError("V31 chat-promotion fields do not agree")
    if not passed and promotion.get("eligible") is not False:
        raise ValueError("Rejected V31 report cannot be chat-promotion eligible")
    return {
        "status": "passed" if passed else "rejected",
        "report": str(path),
        "report_sha256": hashlib.sha256(report_bytes).hexdigest(),
        "training_authorized": not passed,
    }


def require_v31_rejection(config: Mapping[str, Any]) -> dict[str, Any]:
    status = v31_rejection_status(config)
    if status["training_authorized"] is not True:
        raise RuntimeError(
            "V32 is conditional on an audited V31 rejection; "
            f"current V31 status is {status['status']}"
        )
    return status


def _saved_checkpoint_files(step: int) -> tuple[str, ...]:
    common = (
        "adapter.safetensors",
        TRAINING_METADATA_FILENAME,
        RUNTIME_METADATA_FILENAME,
    )
    return common if step == 0 else (*common, "optimizer.pt")


def latest_v32_resume_checkpoint(output: Path, contract: V32Contract) -> Path | None:
    """Return the latest contiguous, complete V32 arm in an interrupted root."""

    if not output.exists():
        return None
    if output.is_symlink() or not output.is_dir():
        raise ValueError(f"V32 output root must be a real directory: {output}")
    observed_directories = sorted(path for path in output.glob("update_*") if path.is_dir())
    parsed: dict[int, Path] = {}
    for path in observed_directories:
        match = _UPDATE_DIRECTORY.fullmatch(path.name)
        if match is None:
            raise ValueError(f"V32 output contains an unrecognized update directory: {path.name}")
        step = int(match.group(1))
        if step not in contract.saved_optimizer_steps:
            raise ValueError(f"V32 output contains an unauthorized saved step: {path.name}")
        if path.is_symlink():
            raise ValueError(f"V32 update checkpoint must not be a symlink: {path}")
        for filename in _saved_checkpoint_files(step):
            candidate = path / filename
            if candidate.is_symlink():
                raise ValueError(f"V32 checkpoint file must not be a symlink: {candidate}")
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
            "V32 complete checkpoints are not a contiguous saved-step prefix: "
            f"observed={complete} expected={expected_prefix}"
        )
    return None if not complete else parsed[complete[-1]]


def _optimizer_checkpoint_step(path: Path, expected_step: int, settings: V32Settings) -> None:
    optimizer_path = path / "optimizer.pt"
    payload = torch.load(optimizer_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise TypeError("V32 resume optimizer state must be a mapping")
    groups = payload.get("param_groups")
    state = payload.get("state")
    if not isinstance(groups, list) or len(groups) != 2 or not isinstance(state, Mapping):
        raise ValueError("V32 resume optimizer must contain exactly two parameter groups")
    expected_groups = (
        ("dense_sidecar_adapter.output_surfaces", settings.sidecar_learning_rate, 2),
        (settings.trainable_bank, settings.decoder_learning_rate, 8),
    )
    parameter_ids: list[Any] = []
    for index, (group, (expected_name, expected_lr, expected_parameter_count)) in enumerate(
        zip(groups, expected_groups, strict=True)
    ):
        if not isinstance(group, Mapping):
            raise TypeError(f"V32 optimizer group {index} must be a mapping")
        if (
            group.get("name") != expected_name
            or float(group.get("lr", math.nan)) != expected_lr
            or float(group.get("weight_decay", math.nan)) != settings.weight_decay
        ):
            raise ValueError(f"V32 optimizer group {index} changed its locked hyperparameters")
        raw_parameters = group.get("params")
        if (
            not isinstance(raw_parameters, list)
            or len(raw_parameters) != expected_parameter_count
        ):
            raise ValueError(f"V32 optimizer group {index} parameter count changed")
        parameter_ids.extend(raw_parameters)
    if len(parameter_ids) != len(set(parameter_ids)) or set(state) != set(parameter_ids):
        raise ValueError("V32 optimizer state does not cover each trainable tensor exactly once")
    observed_steps: set[int] = set()
    for parameter_id in parameter_ids:
        entry = state[parameter_id]
        if not isinstance(entry, Mapping):
            raise TypeError("V32 optimizer parameter state must be a mapping")
        if set(entry) != {"step", "exp_avg", "exp_avg_sq"}:
            raise ValueError("V32 Adam resume state fields changed")
        for moment_name in ("exp_avg", "exp_avg_sq"):
            moment = entry[moment_name]
            if not isinstance(moment, torch.Tensor) or not torch.isfinite(moment).all():
                raise ValueError(f"V32 Adam resume {moment_name} is invalid")
        raw_step = entry.get("step")
        if isinstance(raw_step, torch.Tensor):
            if raw_step.numel() != 1:
                raise ValueError("V32 optimizer Adam step must be scalar")
            raw_step = raw_step.item()
        if isinstance(raw_step, bool) or not isinstance(raw_step, (int, float)):
            raise TypeError("V32 optimizer Adam step must be numeric")
        parsed_step = int(raw_step)
        if float(raw_step) != parsed_step:
            raise ValueError("V32 optimizer Adam step must be integral")
        observed_steps.add(parsed_step)
    if observed_steps != {expected_step}:
        raise ValueError(
            f"V32 resume optimizer does not prove step {expected_step}: "
            f"observed={sorted(observed_steps)}"
        )


def validate_v32_resume_checkpoint(
    *,
    config: Mapping[str, Any],
    output: Path,
    resume: Path,
    contract: V32Contract,
    settings: V32Settings,
    condition: Mapping[str, Any],
    schedule_audit: Mapping[str, Any],
    cache_audit: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate an interrupted-run checkpoint before loading any of its tensors."""

    latest = latest_v32_resume_checkpoint(output, contract)
    if latest is None or latest.resolve() != resume.resolve():
        raise ValueError(f"V32 resume must use the latest complete saved arm: latest={latest}")
    match = _UPDATE_DIRECTORY.fullmatch(resume.name)
    if match is None:
        raise ValueError("V32 resume path is not an update checkpoint")
    step = int(match.group(1))
    metadata = json.loads((resume / TRAINING_METADATA_FILENAME).read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise TypeError("V32 resume metadata must be a JSON object")
    runtime_metadata = json.loads(
        (resume / RUNTIME_METADATA_FILENAME).read_text(encoding="utf-8")
    )
    if not isinstance(runtime_metadata, dict):
        raise TypeError("V32 resume runtime metadata must be a JSON object")
    validate_runtime_checkpoint_metadata(runtime_metadata)
    if runtime_metadata != runtime_checkpoint_metadata(metadata):
        raise ValueError("V32 resume runtime/training metadata mismatch")
    if metadata.get("config_hash") != config_hash(dict(config)):
        raise ValueError("V32 resume config hash changed")
    if metadata.get("optimizer_step") != step:
        raise ValueError("V32 resume metadata/checkpoint step mismatch")
    v30 = metadata.get("v30_joint_pair")
    v32 = metadata.get("v32_microstep")
    if not isinstance(v30, Mapping) or not isinstance(v32, Mapping):
        raise TypeError("V32 resume metadata lacks its training contracts")
    if (
        tuple(v30.get("train_scene_ids", ())) != contract.v31.train_scene_ids
        or tuple(v30.get("validation_scene_ids", ())) != contract.v31.validation_scene_ids
        or v30.get("final_test_scene_ids_loaded") != []
        or v30.get("oracle_environment_files_loaded") is not False
    ):
        raise ValueError("V32 resume metadata changed its data/leakage boundary")
    saved_cache = v30.get("scene_cache")
    if not isinstance(saved_cache, Mapping):
        raise TypeError("V32 resume metadata lacks source-prefix cache provenance")
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
        raise ValueError("V32 resume source-prefix cache provenance changed")
    saved_condition = v32.get("conditional_v31_rejection")
    saved_schedule = v32.get("schedule")
    if not isinstance(saved_condition, Mapping) or not isinstance(saved_schedule, Mapping):
        raise TypeError("V32 resume metadata lacks condition or schedule provenance")
    if (
        saved_condition.get("status") != "rejected"
        or saved_condition.get("training_authorized") is not True
        or saved_condition.get("report_sha256") != condition.get("report_sha256")
    ):
        raise ValueError("V32 resume V31-rejection provenance changed")
    if (
        saved_schedule.get("schedule_sha256") != schedule_audit.get("schedule_sha256")
        or saved_schedule.get("optimizer_step_count") != contract.optimizer_steps
    ):
        raise ValueError("V32 resume schedule changed")
    if (
        v32.get("optimizer_step") != step
        or v32.get("exact_trainable_parameter_count") != 329_216
        or tuple(v32.get("train_scene_ids", ())) != contract.v31.train_scene_ids
        or tuple(v32.get("validation_scene_ids", ())) != contract.v31.validation_scene_ids
        or v32.get("deferred_final_scene_ids_loaded") != []
    ):
        raise ValueError("V32 nested resume contract changed")
    history = metadata.get("history")
    if not isinstance(history, list) or len(history) != step + 1:
        raise ValueError("V32 resume history must contain one row per true optimizer step")
    for expected_step, row in enumerate(history):
        if not isinstance(row, Mapping) or row.get("optimizer_update") != expected_step:
            raise ValueError("V32 resume history is not contiguous")
        if expected_step > 0 and row.get("true_optimizer_step") is not True:
            raise ValueError("V32 resume history contains an unproven optimizer step")
        should_have_validation = expected_step in contract.saved_optimizer_steps
        if (row.get("validation_answer_token_nll") is not None) != should_have_validation:
            raise ValueError("V32 resume validation history does not match saved-step intervals")
        if should_have_validation and row.get("validation_pair_metrics") is None:
            raise ValueError("V32 resume saved arm lacks validation pair metrics")
    best_update = metadata.get("best_epoch")
    best_validation = metadata.get("best_monitor_loss")
    if (
        isinstance(best_update, bool)
        or not isinstance(best_update, int)
        or best_update not in contract.saved_optimizer_steps
        or best_update > step
    ):
        raise ValueError("V32 resume best-update metadata is invalid")
    if isinstance(best_validation, bool) or not isinstance(best_validation, (int, float)):
        raise TypeError("V32 resume best validation must be numeric")
    observed_validation = history[best_update].get("validation_answer_token_nll")
    if (
        not math.isfinite(float(best_validation))
        or observed_validation is None
        or float(observed_validation) != float(best_validation)
    ):
        raise ValueError("V32 resume best checkpoint/validation metadata disagrees")
    saved_validation = [
        (index, float(row["validation_answer_token_nll"]))
        for index, row in enumerate(history)
        if row.get("validation_answer_token_nll") is not None
    ]
    expected_best = min(saved_validation, key=lambda item: (item[1], item[0]))
    if expected_best != (best_update, float(best_validation)):
        raise ValueError("V32 resume best checkpoint is not the best saved validation arm")
    if step > 0:
        _optimizer_checkpoint_step(resume, step, settings)
    return metadata


def build_v32_microstep_schedule(
    records: Sequence[QARecord],
    pair_units: Sequence[CounterfactualPairUnit],
    *,
    settings: V32Settings,
    seed: int,
) -> tuple[list[V32Microstep], dict[str, Any]]:
    """Build deterministic balanced broad+atomic-pair true optimizer steps."""

    if not pair_units:
        raise ValueError("V32 schedule requires changed-answer pair units")
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
        raise RuntimeError("V32 broad schedule does not have exactly one batch per update")
    if any(len({record.scene_id for record in batch}) != 1 for batch in broad_batches):
        # The locked batch size is one. Keep this guard for future controlled
        # extensions because cached_broad_answer_nll is single-scene.
        raise RuntimeError("V32 broad microbatch crossed scene boundaries")

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
        raise RuntimeError("V32 pair schedule omitted one or more atomic units")
    if min(appearances.values()) < 3 or max(appearances.values()) - min(appearances.values()) > 1:
        raise RuntimeError("V32 pair recurrence is incomplete or imbalanced")
    steps = [
        V32Microstep(
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


def _compat_v30_settings(settings: V32Settings) -> V30Settings:
    """Describe the real V32 optimizer in V30-compatible training metadata."""

    return V30Settings(
        enabled=settings.enabled,
        max_optimizer_steps=settings.optimizer_steps,
        evaluation_interval_steps=settings.checkpoint_interval_steps,
        broad_questions_per_cycle=settings.optimizer_steps * settings.broad_batch_size,
        broad_batch_size=settings.broad_batch_size,
        broad_exclude_expected_change=settings.broad_exclude_expected_change,
        pair_repeats_per_cycle=3,
        pair_units_per_batch=settings.pair_units_per_step,
        broad_nll_weight=settings.broad_nll_weight,
        pair_language_nll_weight=settings.pair_language_nll_weight,
        pair_margin_weight=settings.pair_margin_weight,
        pair_margin=settings.pair_margin,
        sidecar_learning_rate=settings.sidecar_learning_rate,
        decoder_learning_rate=settings.decoder_learning_rate,
        weight_decay=settings.weight_decay,
        gradient_clip_norm=settings.gradient_clip_norm,
        minimum_answer_types=settings.minimum_answer_types,
        trainable_bank=settings.trainable_bank,
    )


def _optimizer(bundle: V30Bundle, settings: V32Settings) -> torch.optim.AdamW:
    sidecar = _sidecar_trainable_parameters(bundle.dense_sidecar_adapter)
    decoder = bundle.lora_installation.bank(settings.trainable_bank).installation.parameters()
    optimizer = torch.optim.AdamW(
        [
            {
                "name": "dense_sidecar_adapter.output_surfaces",
                "params": sidecar,
                "lr": settings.sidecar_learning_rate,
                "weight_decay": settings.weight_decay,
            },
            {
                "name": settings.trainable_bank,
                "params": decoder,
                "lr": settings.decoder_learning_rate,
                "weight_decay": settings.weight_decay,
            },
        ]
    )
    assert_v30_trainable_surface(bundle, optimizer)
    return optimizer


def _metadata(
    *,
    bundle: V30Bundle,
    settings: V32Settings,
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
    metadata["v32_microstep"] = {
        "schema_version": 1,
        "artifact": "v32_diverse28_true_microstep_training",
        "optimizer_step": optimizer_step,
        "settings": settings.__dict__,
        "schedule": dict(schedule_audit),
        "conditional_v31_rejection": dict(condition),
        "source_is_approved_v29_update_004": True,
        "exact_trainable_parameter_count": trainable_surface["total_parameter_count"],
        "train_scene_ids": sorted({record.scene_id for record in train_records}),
        "validation_scene_ids": sorted({record.scene_id for record in validation_records}),
        "deferred_final_scene_ids_loaded": [],
        "new_train_prefix_scene_ids": list(_NEW_TRAIN_SCENES),
        "every_saved_arm_requires_independent_selection": True,
        "development_progress_is_not_chat_promotion": True,
        "question_dependent_scene_processing": False,
        "question_dependent_retrieval": False,
        "oracle_environment_files_loaded": False,
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


def preflight_v32(
    config: Mapping[str, Any],
    *,
    require_qa: bool = True,
    require_rejection: bool = False,
) -> ApprovedV29Source:
    v32_contract(config)
    source = require_approved_v29_source(config)
    if require_qa:
        load_v31_qa_records(config)
    if require_rejection:
        require_v31_rejection(config)
    return source


def run_v32(
    *, config: dict[str, Any], output: Path, resume: Path | None = None
) -> dict[str, Any]:
    """Run 80 true micro-updates; never continue V30 or V31 weights."""

    if resume is None and output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty V32 output: {output}")
    contract = v32_contract(config)
    settings = v32_settings(config)
    source = preflight_v32(config, require_qa=True, require_rejection=True)
    condition = require_v31_rejection(config)
    seed = int(config["seed"])
    torch.manual_seed(seed)
    random.seed(seed)
    train_records, validation_records, qa_audit = load_v31_qa_records(config)
    if {record.scene_id for record in train_records} & {
        record.scene_id for record in validation_records
    }:
        raise ValueError("V32 train/validation scenes overlap")
    if len({record.answer_type for record in train_records}) < settings.minimum_answer_types:
        raise ValueError("V32 broad training lacks required answer-type coverage")
    train_pairs = build_exact_question_pair_units(train_records)
    validation_pairs = build_exact_question_pair_units(validation_records)
    if len(train_pairs) != contract.v31.train_changed_pair_unit_count:
        raise ValueError("V32 training pair count differs from its diverse28 contract")
    if len(validation_pairs) != 12:
        raise ValueError("V32 validation pair count differs from its locked contract")
    schedule, schedule_audit = build_v32_microstep_schedule(
        train_records,
        train_pairs,
        settings=settings,
        seed=seed,
    )
    if schedule_audit["pair_unit_minimum_recurrence"] < contract.minimum_pair_unit_recurrence:
        raise RuntimeError("V32 schedule did not recur every pair unit enough times")

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
        raise RuntimeError("V32 derived source-prefix set differs from new training scenes 31--38")
    frozen_hash = frozen_inherited_state_sha256(bundle)
    trainable_surface = assert_v30_trainable_surface(bundle)
    if trainable_surface["total_parameter_count"] != 329_216:
        raise RuntimeError("V32 changed the exact audited trainable surface")
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
            "V32 update zero differs from approved V29 validation NLL: "
            f"source={source_nll} observed={observed_nll} tolerance={tolerance}"
        )
    baseline_pairs = validation_pair_metrics(
        units=validation_pairs,
        caches=caches,
        bundle=bundle,
        margin=settings.pair_margin,
    )
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
            "update_0_equivalence_verified": True,
        }
    ]
    best_update = 0
    best_validation = observed_nll
    start_step = 0
    output.mkdir(parents=True, exist_ok=True)
    if resume is not None:
        resume = resume.resolve()
        resume_metadata = validate_v32_resume_checkpoint(
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
            raise RuntimeError("V32 resume metadata changed during adapter load")
        start_step = int(resume_metadata["optimizer_step"])
        if start_step > 0:
            load_optimizer_checkpoint(resume, optimizer, bundle.language.device)
        history = list(resume_metadata["history"])
        best_update = int(resume_metadata["best_epoch"])
        best_validation = float(resume_metadata["best_monitor_loss"])
        saved_equivalence = resume_metadata["v30_joint_pair"]["update_zero_equivalence"]
        if not isinstance(saved_equivalence, Mapping):
            raise TypeError("V32 resume update-zero equivalence must be a mapping")
        exact_equivalence_fields = set(update_zero) - {
            "observed_validation_answer_token_nll"
        }
        if any(
            saved_equivalence.get(field) != update_zero[field]
            for field in exact_equivalence_fields
        ) or abs(
            float(saved_equivalence.get("observed_validation_answer_token_nll", math.inf))
            - observed_nll
        ) > tolerance:
            raise ValueError("V32 resume update-zero equivalence changed")
        assert_frozen_inherited_state(bundle, frozen_hash)
        bundle.lora_installation.validate_state()
        validate_dense_sidecar_adapter_state(
            bundle.dense_sidecar_adapter,
            expected_parameter_count=int(
                bundle.source_runtime_metadata["dense_sidecar_adapter_parameter_count"]
            ),
            expected_state_sha256=str(resume_metadata["dense_sidecar_adapter_state_sha256"]),
            context="V32 resume sidecar",
        )
        bank_hashes = resume_metadata.get("lora_bank_state_sha256")
        if (
            not isinstance(bank_hashes, Mapping)
            or bundle.lora_installation.bank(settings.trainable_bank).installation.state_sha256()
            != bank_hashes.get(settings.trainable_bank)
        ):
            raise ValueError("V32 resume decoder-bank hash mismatch")
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
                raise ValueError("V32 numbered best checkpoint metadata disagrees")
        _save(output / "best", bundle=bundle, metadata=best_metadata, optimizer=None)
        if best_source.resolve() != resume:
            reloaded = load_adapter_checkpoint(
                resume,
                bundle.checkpoint_modules,
                device="cpu",
                metadata_filename=TRAINING_METADATA_FILENAME,
            )
            if reloaded != resume_metadata:
                raise RuntimeError("V32 resume metadata changed while rebuilding best")
            assert_frozen_inherited_state(bundle, frozen_hash)
            validate_dense_sidecar_adapter_state(
                bundle.dense_sidecar_adapter,
                expected_parameter_count=int(
                    bundle.source_runtime_metadata["dense_sidecar_adapter_parameter_count"]
                ),
                expected_state_sha256=str(
                    resume_metadata["dense_sidecar_adapter_state_sha256"]
                ),
                context="V32 post-best-repair resume sidecar",
            )
        print(
            json.dumps(
                {
                    "phase": "v32_true_microstep_resume",
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
        )
        _save(output / "update_000", bundle=bundle, metadata=initial_metadata, optimizer=None)
        _save(output / "best", bundle=bundle, metadata=initial_metadata, optimizer=None)

    all_trainable = freeze_for_v30(bundle)
    for item in schedule[start_step:]:
        step = item.optimizer_step
        broad_record = item.broad_records[0]
        bundle.dense_sidecar_adapter.train()
        bundle.lora_installation.train()
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
        assert_v30_trainable_surface(bundle, optimizer)
        missing_gradients = [
            index for index, parameter in enumerate(all_trainable) if parameter.grad is None
        ]
        if missing_gradients:
            raise RuntimeError(f"V32 trainable tensors lack gradients: {missing_gradients}")
        if any(not torch.isfinite(parameter.grad).all() for parameter in all_trainable):
            raise RuntimeError("V32 trainable gradient is nonfinite")
        gradient_norm = torch.nn.utils.clip_grad_norm_(all_trainable, settings.gradient_clip_norm)
        optimizer.step()
        assert_frozen_inherited_state(bundle, frozen_hash)
        bundle.lora_installation.validate_state()
        validate_dense_sidecar_adapter_state(
            bundle.dense_sidecar_adapter,
            expected_parameter_count=int(
                bundle.source_runtime_metadata["dense_sidecar_adapter_parameter_count"]
            ),
            context="V32 post-microstep sidecar",
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
                "preclip_gradient_norm": float(gradient_norm.detach().cpu()),
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
        )
        _save(output / f"update_{step:03d}", bundle=bundle, metadata=metadata, optimizer=optimizer)
        if best_update == step:
            _save(output / "best", bundle=bundle, metadata=metadata, optimizer=None)
        print(
            json.dumps(
                {
                    "phase": "v32_true_microstep_checkpoint",
                    "optimizer_step": step,
                    "validation_answer_token_nll": validation_value,
                    "validation_pair_passed_units": pair_validation["passed_units"],
                    "best_update": best_update,
                }
            ),
            flush=True,
        )
    assert_frozen_inherited_state(bundle, frozen_hash)
    return {
        "schema_version": 1,
        "artifact": "v32_diverse28_true_microstep_training",
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
        "v31_condition": condition,
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
        source = preflight_v32(config, require_qa=True, require_rejection=False)
        report = {
            "artifact": "v32_diverse28_true_microstep_preflight",
            "passed": True,
            "source_v29_checkpoint": str(source.checkpoint),
            "v31_condition": v31_rejection_status(config),
            "training_starts_only_after_v31_rejection": True,
            "final_test_scenes_touched": False,
        }
    else:
        output = _resolve(args.output)
        resume = None if args.resume is None else _resolve(args.resume)
        if args.resume_latest:
            resume = latest_v32_resume_checkpoint(output, v32_contract(config))
            if resume is None:
                raise FileNotFoundError(
                    f"V32 output has no complete checkpoint to resume: {output}"
                )
        report = run_v32(config=config, output=output, resume=resume)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "V32Contract",
    "V32Microstep",
    "V32Settings",
    "build_v32_microstep_schedule",
    "latest_v32_resume_checkpoint",
    "preflight_v32",
    "require_v31_rejection",
    "run_v32",
    "v31_rejection_status",
    "v32_contract",
    "v32_settings",
    "validate_v32_resume_checkpoint",
]
