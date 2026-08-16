"""V34 bounded base-route repair from V33's exact stopped update 64.

Gemma, all LoRA banks, and all 404,608 parameters learned by V33 are frozen.
Only the predeclared 199,808-parameter base normalization/projection route is
optimized.  In addition to broad and atomic-pair language losses, V34 uses a
question-free scene separation loss over the eight unique changed training
scene pairs and all 112 other training-scene pairs.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from semantic_3d_chat.config import PROJECT_ROOT, config_hash, load_config
from semantic_3d_chat.data.dataset import QARecord
from semantic_3d_chat.evaluation.v33_terminal_gate import audit_v33_update64
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
from semantic_3d_chat.training.train_environmental_sidecar_v33 import (
    _TRAINABLE_NAMES as _V33_TRAINABLE_NAMES,
)
from semantic_3d_chat.training.train_environmental_sidecar_v33 import (
    assert_deferred_final_scenes_absent,
    prefix_separation_diagnostics,
    prefix_separation_ratios,
    validation_family_teacher_metrics,
)
from semantic_3d_chat.training.train_joint_pair_v30 import (
    CachedPreSidecarScene,
    V30Bundle,
    adapted_scene_tokens,
    cache_pre_sidecar_scenes,
    cached_broad_answer_nll,
    load_v30_bundle,
    paired_canonical_answer_objective,
    prefix_sha256,
    require_approved_v29_source,
    select_balanced_broad_records,
    validation_answer_nll,
    validation_pair_metrics,
)
from semantic_3d_chat.training.train_joint_pair_v31 import (
    V31Contract,
    load_v31_qa_records,
    v31_contract,
)

DEFAULT_CONFIG = Path("configs/experiments/gemma4_diverse28_base_surface_v34.yaml")
DEFAULT_OUTPUT = Path("data_gemma4/checkpoints/gemma4_v34_diverse28_base_surface")
_UPDATE_DIRECTORY = re.compile(r"update_([0-9]{3})")
_NEW_TRAIN_SCENES = tuple(f"scene_{index:06d}" for index in range(31, 39))
_BASE_NORM_NAMES = ("base_norm.weight", "base_norm.bias")
_BASE_PROJECTION_NAMES = ("base_projection.weight", "base_projection.bias")
_TRAINABLE_NAMES = (*_BASE_NORM_NAMES, *_BASE_PROJECTION_NAMES)
_V33_FROZEN_NAMES = tuple(_V33_TRAINABLE_NAMES)
_TERMINAL_REPORT_SHA256 = "703525975c7a03a9b995c6f950dda92ed2945bd1857008196a1086e2a6c19a49"
_SOURCE_FILE_SHA256 = {
    "adapter.safetensors": "32c071d7acca0e52f8ae4c3dee8cba83319d67b184bbb3ab9957a6f6c4fcf987",
    TRAINING_METADATA_FILENAME: "ef97dfc3415eb4cfbdf30fe952e85db5ea4c54e4dec896a40725fb41fd787c91",
    RUNTIME_METADATA_FILENAME: "fe8df1c8c052ac50899eb19952f96b74ac691780e20b604ba4e11072db32e168",
    "optimizer.pt": "845aa42380b5c8c575162cb003fcadc7761fd615071b9dab71d9da4a85ba3d09",
}


@dataclass(frozen=True)
class V34Settings:
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
    separation_rank_weight: float
    separation_rank_margin: float
    unrelated_stability_weight: float
    pair_ratio_cap: float
    pair_ratio_cap_weight: float
    source_trust_region_weight: float
    separation_rms_floor: float
    base_norm_learning_rate: float
    base_projection_learning_rate: float
    weight_decay: float
    base_norm_gradient_clip_norm: float
    base_projection_gradient_clip_norm: float
    minimum_answer_types: int

    @property
    def saved_optimizer_steps(self) -> tuple[int, ...]:
        return tuple(range(0, self.optimizer_steps + 1, self.checkpoint_interval_steps))


@dataclass(frozen=True)
class V34Contract:
    v31: V31Contract
    terminal_report: Path
    terminal_report_sha256: str
    source_checkpoint: Path
    source_file_sha256: Mapping[str, str]
    saved_optimizer_steps: tuple[int, ...]
    early_gate_optimizer_step: int
    early_gate_changed_pair_selectivity_ratio_minimum: float
    early_gate_changed_pair_coverage_minimum: int
    early_gate_minimum_physical_pair_selectivity_ratio: float
    early_gate_unrelated_median_ratio_minimum: float
    early_gate_unrelated_median_ratio_maximum: float
    early_gate_unrelated_p90_abs_log_ratio_maximum: float
    development_training_changed_pair_selectivity_ratio_minimum: float
    development_training_changed_pair_coverage_minimum: int
    development_validation_weak_minus_unrelated_ratio_minimum: float
    development_unrelated_ratio_minimum: float
    development_unrelated_ratio_maximum: float


@dataclass(frozen=True)
class V34Microstep:
    optimizer_step: int
    broad_records: tuple[QARecord, ...]
    pair_units: tuple[CounterfactualPairUnit, ...]


@dataclass(frozen=True)
class PrefixSeparationReference:
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
        raise ValueError(f"{field} must be finite and {'positive' if positive else 'nonnegative'}")
    return parsed


def v34_settings(config: Mapping[str, Any]) -> V34Settings:
    training = _mapping(config.get("training"), "training")
    raw = _mapping(training.get("v34_base_surface"), "training.v34_base_surface")
    fields = set(V34Settings.__dataclass_fields__)
    if set(raw) != fields:
        raise ValueError(
            "training.v34_base_surface fields differ from the locked schema: "
            f"missing={sorted(fields - set(raw))} unknown={sorted(set(raw) - fields)}"
        )
    if not isinstance(raw["enabled"], bool) or not isinstance(
        raw["broad_exclude_expected_change"], bool
    ):
        raise TypeError("V34 boolean settings must be boolean")
    settings = V34Settings(
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
        pair_margin_weight=_finite(
            "pair_margin_weight", raw["pair_margin_weight"], positive=True
        ),
        pair_margin=_finite("pair_margin", raw["pair_margin"], positive=True),
        separation_rank_weight=_finite(
            "separation_rank_weight", raw["separation_rank_weight"], positive=True
        ),
        separation_rank_margin=_finite(
            "separation_rank_margin", raw["separation_rank_margin"], positive=True
        ),
        unrelated_stability_weight=_finite(
            "unrelated_stability_weight", raw["unrelated_stability_weight"], positive=True
        ),
        pair_ratio_cap=_finite("pair_ratio_cap", raw["pair_ratio_cap"], positive=True),
        pair_ratio_cap_weight=_finite(
            "pair_ratio_cap_weight", raw["pair_ratio_cap_weight"], positive=True
        ),
        source_trust_region_weight=_finite(
            "source_trust_region_weight", raw["source_trust_region_weight"], positive=True
        ),
        separation_rms_floor=_finite(
            "separation_rms_floor", raw["separation_rms_floor"], positive=True
        ),
        base_norm_learning_rate=_finite(
            "base_norm_learning_rate", raw["base_norm_learning_rate"], positive=True
        ),
        base_projection_learning_rate=_finite(
            "base_projection_learning_rate",
            raw["base_projection_learning_rate"],
            positive=True,
        ),
        weight_decay=_finite("weight_decay", raw["weight_decay"], positive=False),
        base_norm_gradient_clip_norm=_finite(
            "base_norm_gradient_clip_norm", raw["base_norm_gradient_clip_norm"], positive=True
        ),
        base_projection_gradient_clip_norm=_finite(
            "base_projection_gradient_clip_norm",
            raw["base_projection_gradient_clip_norm"],
            positive=True,
        ),
        minimum_answer_types=_positive_int("minimum_answer_types", raw["minimum_answer_types"]),
    )
    expected = {
        "enabled": True,
        "optimizer_steps": 64,
        "checkpoint_interval_steps": 8,
        "broad_batch_size": 1,
        "pair_units_per_step": 1,
        "broad_exclude_expected_change": True,
        "broad_nll_weight": 1.0,
        "pair_language_nll_weight": 1.0,
        "pair_margin_weight": 4.0,
        "pair_margin": 0.5,
        "separation_rank_weight": 2.0,
        "separation_rank_margin": math.log(1.02),
        "unrelated_stability_weight": 1.0,
        "pair_ratio_cap": 1.25,
        "pair_ratio_cap_weight": 1.0,
        "source_trust_region_weight": 0.25,
        "separation_rms_floor": 1e-6,
        "base_norm_learning_rate": 2.5e-5,
        "base_projection_learning_rate": 1e-4,
        "weight_decay": 0.0,
        "base_norm_gradient_clip_norm": 1.0,
        "base_projection_gradient_clip_norm": 1.0,
        "minimum_answer_types": 4,
    }
    mismatches = {
        field: {"observed": getattr(settings, field), "expected": value}
        for field, value in expected.items()
        if not math.isclose(float(getattr(settings, field)), float(value), rel_tol=0, abs_tol=1e-15)
    }
    if mismatches:
        raise ValueError(f"V34 locked optimizer/objective settings changed: {mismatches}")
    return settings


def v34_contract(config: Mapping[str, Any]) -> V34Contract:
    v31 = v31_contract(config)
    settings = v34_settings(config)
    raw = _mapping(config.get("v34_base_surface"), "v34_base_surface")
    required = {
        "schema_version", "role", "engine", "v33_terminal_gate_report",
        "v33_terminal_gate_report_sha256", "source_checkpoint", "source_optimizer_step",
        "source_file_sha256", "source_v33_config_sha256", "source_v33_schedule_sha256",
        "train_scene_ids", "validation_scene_ids", "deferred_final_scene_ids",
        "train_question_count", "validation_question_count", "train_changed_pair_unit_count",
        "validation_changed_pair_unit_count", "optimizer_steps", "checkpoint_interval_steps",
        "saved_optimizer_steps", "exact_balanced_pair_recurrence_minimum",
        "exact_balanced_pair_recurrence_maximum", "exact_pair_units_with_third_recurrence",
        "trainable_parameter_names", "exact_trainable_parameter_count",
        "frozen_v33_parameter_names", "frozen_v33_parameter_count", "gemma_decoder_frozen",
        "all_lora_banks_frozen", "every_v33_learned_tensor_frozen",
        "exact_update0_source_tensors", "exact_update0_source_prefixes",
        "exact_update0_source_validation_nll", "separation_uses_training_scenes_only",
        "separation_uses_question_or_answer_text", "separation_uses_oracle_environment_inputs",
        "separation_baselines_fixed_at_update0", "separation_unique_physical_changed_pair_count",
        "separation_all_nonchanged_train_scene_pair_count", "separation_all_pair_sets_fixed_before_training",
        "separation_reduction", "separation_objective_bounded", "inspect_every_saved_arm",
        "early_gate_optimizer_step", "early_gate_changed_pair_selectivity_ratio_minimum",
        "early_gate_changed_pair_coverage_minimum", "early_gate_minimum_physical_pair_selectivity_ratio",
        "early_gate_unrelated_median_ratio_minimum", "early_gate_unrelated_median_ratio_maximum",
        "early_gate_unrelated_p90_abs_log_ratio_maximum", "early_gate_uses_training_scenes_only",
        "development_training_changed_pair_selectivity_ratio_minimum",
        "development_training_changed_pair_coverage_minimum",
        "development_validation_weak_minus_unrelated_ratio_minimum",
        "development_each_weak_family_exceeds_unrelated", "development_unrelated_ratio_minimum",
        "development_unrelated_ratio_maximum", "development_nonmirror_teacher_complete_minimum",
        "development_changed_complete_pairs_minimum", "chat_promotion_changed_complete_pairs_minimum",
        "chat_promotion_requires_each_validation_family", "chat_promotion_aggregate_exact_no_regression",
        "old_color_full_vocab_sides_minimum", "old_mirror_full_vocab_sides_minimum",
        "old_controls_no_new_negatives",
    }
    if set(raw) != required:
        raise ValueError(
            "v34_base_surface fields differ from the locked schema: "
            f"missing={sorted(required - set(raw))} unknown={sorted(set(raw) - required)}"
        )
    exact = {
        "schema_version": 1,
        "role": "conditional_v33_u64_base_scene_surface_v34",
        "engine": "exact_v33_u64_base_only_true_microsteps_with_bounded_separation_rank",
        "v33_terminal_gate_report_sha256": _TERMINAL_REPORT_SHA256,
        "source_optimizer_step": 64,
        "source_v33_config_sha256": "e920d28da8ab0abc3c0ab2c4ad812a2743d1894b769c6302097ac41c31da3905",
        "source_v33_schedule_sha256": "90b7c3b337f573b47a75ed3faefc915eacd98c9ef11b572ff3c45c4166fc9590",
        "optimizer_steps": 64,
        "checkpoint_interval_steps": 8,
        "exact_balanced_pair_recurrence_minimum": 2,
        "exact_balanced_pair_recurrence_maximum": 3,
        "exact_pair_units_with_third_recurrence": 14,
        "exact_trainable_parameter_count": 199_808,
        "frozen_v33_parameter_count": 404_608,
        "separation_unique_physical_changed_pair_count": 8,
        "separation_all_nonchanged_train_scene_pair_count": 112,
        "separation_reduction": "separate_log_ratio_means_over_8_changed_and_112_nonchanged_pairs",
        "early_gate_optimizer_step": 32,
        "early_gate_changed_pair_selectivity_ratio_minimum": 1.02,
        "early_gate_changed_pair_coverage_minimum": 6,
        "early_gate_minimum_physical_pair_selectivity_ratio": 0.98,
        "early_gate_unrelated_median_ratio_minimum": 1 / 1.02,
        "early_gate_unrelated_median_ratio_maximum": 1.02,
        "early_gate_unrelated_p90_abs_log_ratio_maximum": math.log(1.02),
        "development_training_changed_pair_selectivity_ratio_minimum": 1.02,
        "development_training_changed_pair_coverage_minimum": 6,
        "development_validation_weak_minus_unrelated_ratio_minimum": 0.005,
        "development_unrelated_ratio_minimum": 1 / 1.02,
        "development_unrelated_ratio_maximum": 1.02,
        "development_nonmirror_teacher_complete_minimum": 1,
        "development_changed_complete_pairs_minimum": 1,
        "chat_promotion_changed_complete_pairs_minimum": 6,
        "old_color_full_vocab_sides_minimum": 12,
        "old_mirror_full_vocab_sides_minimum": 10,
    }
    mismatches = {key: {"observed": raw.get(key), "expected": value} for key, value in exact.items()
                  if raw.get(key) != value}
    if mismatches:
        raise ValueError(f"V34 locked contract changed: {mismatches}")
    true_fields = (
        "gemma_decoder_frozen", "all_lora_banks_frozen", "every_v33_learned_tensor_frozen",
        "exact_update0_source_tensors", "exact_update0_source_prefixes",
        "exact_update0_source_validation_nll", "separation_uses_training_scenes_only",
        "separation_baselines_fixed_at_update0", "separation_all_pair_sets_fixed_before_training",
        "separation_objective_bounded", "inspect_every_saved_arm", "early_gate_uses_training_scenes_only",
        "development_each_weak_family_exceeds_unrelated", "chat_promotion_requires_each_validation_family",
        "chat_promotion_aggregate_exact_no_regression", "old_controls_no_new_negatives",
    )
    if any(raw.get(field) is not True for field in true_fields):
        raise ValueError("V34 required true-valued safety field changed")
    if raw.get("separation_uses_question_or_answer_text") is not False or raw.get(
        "separation_uses_oracle_environment_inputs"
    ) is not False:
        raise ValueError("V34 separation objective acquired a forbidden input")
    if tuple(raw["train_scene_ids"]) != v31.train_scene_ids or tuple(
        raw["validation_scene_ids"]
    ) != v31.validation_scene_ids or tuple(raw["deferred_final_scene_ids"]) != v31.deferred_final_scene_ids:
        raise ValueError("V34 scene split differs from locked diverse28 development data")
    if (raw["train_question_count"], raw["validation_question_count"],
        raw["train_changed_pair_unit_count"], raw["validation_changed_pair_unit_count"]) != (
        384, 216, 25, 12
    ):
        raise ValueError("V34 QA counts differ from locked diverse28 development data")
    if tuple(raw["saved_optimizer_steps"]) != settings.saved_optimizer_steps:
        raise ValueError("V34 saved arms must be exactly 0,8,...,64")
    if tuple(raw["trainable_parameter_names"]) != _TRAINABLE_NAMES:
        raise ValueError("V34 base-only trainable tensor names changed")
    if tuple(raw["frozen_v33_parameter_names"]) != _V33_FROZEN_NAMES:
        raise ValueError("V34 frozen V33 tensor names changed")
    source_hashes = _mapping(raw["source_file_sha256"], "source_file_sha256")
    if dict(source_hashes) != _SOURCE_FILE_SHA256:
        raise ValueError("V34 exact V33 update-64 source hashes changed")
    return V34Contract(
        v31=v31,
        terminal_report=_resolve(str(raw["v33_terminal_gate_report"])),
        terminal_report_sha256=str(raw["v33_terminal_gate_report_sha256"]),
        source_checkpoint=_resolve(str(raw["source_checkpoint"])),
        source_file_sha256=dict(source_hashes),
        saved_optimizer_steps=settings.saved_optimizer_steps,
        early_gate_optimizer_step=32,
        early_gate_changed_pair_selectivity_ratio_minimum=1.02,
        early_gate_changed_pair_coverage_minimum=6,
        early_gate_minimum_physical_pair_selectivity_ratio=0.98,
        early_gate_unrelated_median_ratio_minimum=1 / 1.02,
        early_gate_unrelated_median_ratio_maximum=1.02,
        early_gate_unrelated_p90_abs_log_ratio_maximum=math.log(1.02),
        development_training_changed_pair_selectivity_ratio_minimum=1.02,
        development_training_changed_pair_coverage_minimum=6,
        development_validation_weak_minus_unrelated_ratio_minimum=0.005,
        development_unrelated_ratio_minimum=1 / 1.02,
        development_unrelated_ratio_maximum=1.02,
    )


def require_v33_terminal_gate(config: Mapping[str, Any]) -> dict[str, Any]:
    contract = v34_contract(config)
    path = contract.terminal_report
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"V34 requires a real V33 terminal report: {path}")
    observed_sha = _sha256(path)
    if observed_sha != contract.terminal_report_sha256:
        raise ValueError("V33 terminal report hash differs from V34's immutable pin")
    report = json.loads(path.read_text(encoding="utf-8"))
    audited = audit_v33_update64()
    if report != audited:
        raise ValueError("V33 terminal report does not replay from its pinned metadata/tensors")
    required = {
        "artifact": "v33_update64_terminal_gate", "passed": True,
        "stopped_at_optimizer_step": 64, "no_update_072_or_later": True,
        "final_test_scenes_touched": False, "oracle_loaded": False,
        "v33_development_selection_passed": False, "v33_chat_promotion_eligible": False,
        "conditional_v34_base_surface_authorized": True,
    }
    mismatch = {key: {"observed": report.get(key), "expected": value}
                for key, value in required.items() if report.get(key) != value}
    if mismatch:
        raise ValueError(f"V33 terminal report does not authorize V34: {mismatch}")
    return {"path": str(path), "sha256": observed_sha, "report": report}


def require_exact_v33_source(config: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    contract = v34_contract(config)
    require_v33_terminal_gate(config)
    source = contract.source_checkpoint
    if source.is_symlink() or not source.is_dir() or source.name != "update_064":
        raise FileNotFoundError(f"V34 source must be the real numbered V33 update 64: {source}")
    for filename, expected in contract.source_file_sha256.items():
        candidate = source / filename
        if candidate.is_symlink() or not candidate.is_file() or _sha256(candidate) != expected:
            raise ValueError(f"V34 source file differs from its exact pin: {candidate}")
    metadata = json.loads((source / TRAINING_METADATA_FILENAME).read_text(encoding="utf-8"))
    if not isinstance(metadata, dict) or metadata.get("optimizer_step") != 64:
        raise ValueError("V34 source metadata is not V33 update 64")
    runtime = json.loads((source / RUNTIME_METADATA_FILENAME).read_text(encoding="utf-8"))
    validate_runtime_checkpoint_metadata(runtime)
    if runtime != runtime_checkpoint_metadata(metadata):
        raise ValueError("V34 source runtime metadata is not the exact sanitized source")
    return source, metadata


def build_v34_schedule(
    records: Sequence[QARecord], pair_units: Sequence[CounterfactualPairUnit], *,
    settings: V34Settings, seed: int,
) -> tuple[list[V34Microstep], dict[str, Any]]:
    if len(pair_units) != 25:
        raise ValueError("V34 requires exactly 25 changed QA units")
    broad = select_balanced_broad_records(
        records, count=settings.optimizer_steps * settings.broad_batch_size, seed=seed,
        exclude_expected_change=settings.broad_exclude_expected_change,
    )
    canonical = sorted(pair_units, key=lambda unit: (unit.pair_id, unit.question_key))
    third = list(canonical)
    random.Random(seed + 34_000).shuffle(third)
    scheduled = [*canonical, *canonical, *third[:14]]
    if len(scheduled) != settings.optimizer_steps:
        raise RuntimeError("V34 balanced pair schedule does not contain exactly 64 rows")
    appearances = Counter((unit.pair_id, unit.question_key) for unit in scheduled)
    if Counter(appearances.values()) != Counter({2: 11, 3: 14}):
        raise RuntimeError("V34 pair schedule must recur 11 units twice and 14 units three times")
    steps = [
        V34Microstep(index + 1, (broad[index],), (scheduled[index],))
        for index in range(settings.optimizer_steps)
    ]
    payload = [
        {"optimizer_step": row.optimizer_step,
         "broad": [(record.scene_id, record.question_id) for record in row.broad_records],
         "pairs": [(unit.pair_id, unit.question_key) for unit in row.pair_units]}
        for row in steps
    ]
    return steps, {
        "schema_version": 1, "optimizer_step_count": 64,
        "true_optimizer_step_per_schedule_row": True, "broad_records_per_step": 1,
        "pair_units_per_step": 1, "pair_unit_count": 25,
        "pair_unit_minimum_recurrence": 2, "pair_unit_maximum_recurrence": 3,
        "pair_units_with_third_recurrence": 14, "pair_units_atomic": True,
        "broad_expected_change_excluded": True,
        "broad_answer_type_counts": dict(sorted(Counter(r.answer_type for r in broad).items())),
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
        raise ValueError(f"V34 requires eight unique physical changed pairs, got {len(changed)}")
    scene_ids = sorted({scene for pair in changed.values() for scene in pair})
    if len(scene_ids) != 16:
        raise ValueError("V34 changed physical pairs must partition all 16 training scenes")
    changed_sets = {frozenset(pair) for pair in changed.values()}
    unrelated = tuple(
        pair for pair in combinations(scene_ids, 2) if frozenset(pair) not in changed_sets
    )
    if len(unrelated) != 112:
        raise RuntimeError(f"V34 all-nonchanged train pair set must contain 112 pairs: {len(unrelated)}")
    return dict(sorted(changed.items())), unrelated


def _prefixes(
    scene_ids: Sequence[str], caches: Mapping[str, CachedPreSidecarScene], bundle: V30Bundle,
    *, inference: bool,
) -> dict[str, torch.Tensor]:
    dtype = next(bundle.language.model.parameters()).dtype
    context = torch.inference_mode() if inference else torch.enable_grad()
    with context:
        return {
            scene_id: bundle.composer.scene_prefix(
                adapted_scene_tokens(caches[scene_id], bundle).to(dtype)
            ).float()
            for scene_id in scene_ids
        }


def _rms_difference(left: torch.Tensor, right: torch.Tensor, floor: float) -> torch.Tensor:
    # Add the squared floor before sqrt. A post-sqrt clamp has an undefined
    # 0 * inf backward at exact-zero inputs on some backends.
    return ((left - right).square().mean() + floor**2).sqrt()


def build_prefix_separation_reference(
    units: Sequence[CounterfactualPairUnit], caches: Mapping[str, CachedPreSidecarScene],
    bundle: V30Bundle, *, rms_floor: float,
) -> PrefixSeparationReference:
    changed, unrelated = physical_pair_sets(units)
    scene_ids = sorted({scene for pair in changed.values() for scene in pair})
    prefixes = _prefixes(scene_ids, caches, bundle, inference=True)
    changed_rms = {
        pair_id: float(_rms_difference(prefixes[left], prefixes[right], rms_floor))
        for pair_id, (left, right) in changed.items()
    }
    unrelated_rms = {
        pair: float(_rms_difference(prefixes[pair[0]], prefixes[pair[1]], rms_floor))
        for pair in unrelated
    }
    if min(*changed_rms.values(), *unrelated_rms.values()) <= rms_floor:
        raise ValueError("V34 source pair separation reached the configured RMS floor")
    payload = {
        "changed_pairs": changed, "unrelated_pairs": unrelated,
        "changed_rms": changed_rms,
        "unrelated_rms": {"|".join(pair): value for pair, value in unrelated_rms.items()},
        "rms_floor": rms_floor,
    }
    audit_sha = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return PrefixSeparationReference(
        source_prefixes={key: value.detach().clone() for key, value in prefixes.items()},
        changed_pairs=changed, unrelated_pairs=unrelated, changed_rms=changed_rms,
        unrelated_rms=unrelated_rms, audit_sha256=audit_sha,
    )


def separation_loss_and_diagnostics(
    *, prefixes: Mapping[str, torch.Tensor], reference: PrefixSeparationReference,
    settings: V34Settings,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    changed_ratios = torch.stack([
        _rms_difference(prefixes[left], prefixes[right], settings.separation_rms_floor)
        / reference.changed_rms[pair_id]
        for pair_id, (left, right) in reference.changed_pairs.items()
    ])
    unrelated_ratios = torch.stack([
        _rms_difference(prefixes[left], prefixes[right], settings.separation_rms_floor)
        / reference.unrelated_rms[(left, right)]
        for left, right in reference.unrelated_pairs
    ])
    log_changed = changed_ratios.log()
    log_unrelated = unrelated_ratios.log()
    unrelated_log_mean = log_unrelated.mean()
    selectivity_log = log_changed - unrelated_log_mean
    rank_hinge = torch.relu(settings.separation_rank_margin - selectivity_log).mean()
    unrelated_stability = log_unrelated.square().mean()
    pair_cap = torch.relu(log_changed - math.log(settings.pair_ratio_cap)).square().mean()
    trust_terms: list[torch.Tensor] = []
    mean_shift_terms: list[torch.Tensor] = []
    norm_shift_terms: list[torch.Tensor] = []
    for scene_id, current in prefixes.items():
        source = reference.source_prefixes[scene_id].to(current)
        source_rms = (
            source.square().mean() + settings.separation_rms_floor**2
        ).sqrt()
        normalized_mean_shift = (current.mean() - source.mean()) / source_rms
        current_rms = (
            current.square().mean() + settings.separation_rms_floor**2
        ).sqrt()
        log_norm_shift = (current_rms / source_rms).log()
        trust_terms.append(normalized_mean_shift.square() + log_norm_shift.square())
        mean_shift_terms.append(normalized_mean_shift.abs())
        norm_shift_terms.append(log_norm_shift.abs())
    trust_region = torch.stack(trust_terms).mean()
    total = (
        settings.separation_rank_weight * rank_hinge
        + settings.unrelated_stability_weight * unrelated_stability
        + settings.pair_ratio_cap_weight * pair_cap
        + settings.source_trust_region_weight * trust_region
    )
    return total, {
        "rank_hinge": rank_hinge, "unrelated_log_stability": unrelated_stability,
        "pair_cap_penalty": pair_cap, "source_trust_region": trust_region,
        "changed_ratios": changed_ratios, "unrelated_ratios": unrelated_ratios,
        "selectivity_ratios": selectivity_log.exp(),
        "prefix_mean_shift_abs_mean": torch.stack(mean_shift_terms).mean(),
        "prefix_norm_log_shift_abs_mean": torch.stack(norm_shift_terms).mean(),
    }


def _quantile(values: torch.Tensor, q: float) -> float:
    return float(torch.quantile(values.detach().float().cpu(), q))


def summarize_separation(diagnostics: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    changed = diagnostics["changed_ratios"].detach().float().cpu()
    unrelated = diagnostics["unrelated_ratios"].detach().float().cpu()
    selectivity = diagnostics["selectivity_ratios"].detach().float().cpu()
    unrelated_abs_log = unrelated.log().abs()
    return {
        "schema_version": 1,
        "unique_changed_physical_pair_count": int(changed.numel()),
        "all_nonchanged_train_scene_pair_count": int(unrelated.numel()),
        "changed_ratio_mean": float(changed.mean()), "changed_ratio_median": _quantile(changed, 0.5),
        "changed_ratio_minimum": float(changed.min()), "changed_ratio_maximum": float(changed.max()),
        "unrelated_ratio_mean": float(unrelated.mean()), "unrelated_ratio_median": _quantile(unrelated, 0.5),
        "unrelated_ratio_p90": _quantile(unrelated, 0.9), "unrelated_ratio_maximum": float(unrelated.max()),
        "unrelated_abs_log_ratio_p90": _quantile(unrelated_abs_log, 0.9),
        "unrelated_abs_log_ratio_maximum": float(unrelated_abs_log.max()),
        "changed_selectivity_ratio_geometric_mean": float(selectivity.log().mean().exp()),
        "changed_selectivity_ratio_minimum": float(selectivity.min()),
        "changed_selectivity_over_1_02_count": int((selectivity >= 1.02).sum()),
        "changed_selectivity_ratios": [float(value) for value in selectivity],
        "rank_hinge": float(diagnostics["rank_hinge"].detach().cpu()),
        "unrelated_log_stability": float(diagnostics["unrelated_log_stability"].detach().cpu()),
        "pair_cap_penalty": float(diagnostics["pair_cap_penalty"].detach().cpu()),
        "source_trust_region": float(diagnostics["source_trust_region"].detach().cpu()),
        "prefix_mean_shift_abs_mean": float(diagnostics["prefix_mean_shift_abs_mean"].detach().cpu()),
        "prefix_norm_log_shift_abs_mean": float(
            diagnostics["prefix_norm_log_shift_abs_mean"].detach().cpu()
        ),
        "question_or_answer_text_used": False,
        "oracle_environment_inputs_used": False,
        "validation_scenes_used": False,
    }


def training_separation_diagnostics(
    *, reference: PrefixSeparationReference, caches: Mapping[str, CachedPreSidecarScene],
    bundle: V30Bundle, settings: V34Settings,
) -> dict[str, Any]:
    prefixes = _prefixes(sorted(reference.source_prefixes), caches, bundle, inference=True)
    with torch.inference_mode():
        _loss, diagnostics = separation_loss_and_diagnostics(
            prefixes=prefixes, reference=reference, settings=settings
        )
    return summarize_separation(diagnostics)


def v34_early_training_gate(
    diagnostics: Mapping[str, Any], contract: V34Contract,
) -> dict[str, bool]:
    checks = {
        "changed_selectivity_geometric_mean_at_least_1_02": (
            float(diagnostics["changed_selectivity_ratio_geometric_mean"])
            >= contract.early_gate_changed_pair_selectivity_ratio_minimum
        ),
        "at_least_6_of_8_changed_pairs_over_1_02": (
            int(diagnostics["changed_selectivity_over_1_02_count"])
            >= contract.early_gate_changed_pair_coverage_minimum
        ),
        "no_physical_pair_selectivity_below_0_98": (
            float(diagnostics["changed_selectivity_ratio_minimum"])
            >= contract.early_gate_minimum_physical_pair_selectivity_ratio
        ),
        "unrelated_median_two_sided_within_1_02": (
            contract.early_gate_unrelated_median_ratio_minimum
            <= float(diagnostics["unrelated_ratio_median"])
            <= contract.early_gate_unrelated_median_ratio_maximum
        ),
        "unrelated_p90_abs_log_within_log_1_02": (
            float(diagnostics["unrelated_abs_log_ratio_p90"])
            <= contract.early_gate_unrelated_p90_abs_log_ratio_maximum
        ),
    }
    return {**checks, "passed": all(checks.values()), "training_scenes_only": True}


def _named_base_groups(module: DenseSidecarAdapter) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    named = dict(module.named_parameters())
    required = {*_TRAINABLE_NAMES, *_V33_FROZEN_NAMES}
    if not required.issubset(named):
        raise RuntimeError("Dense sidecar lacks V34's audited base/V33 surfaces")
    return ([named[name] for name in _BASE_NORM_NAMES],
            [named[name] for name in _BASE_PROJECTION_NAMES])


def freeze_for_v34(bundle: V30Bundle) -> list[torch.nn.Parameter]:
    bundle.language.model.requires_grad_(False)
    for module in bundle.checkpoint_modules.values():
        module.requires_grad_(False)
        module.eval()
    groups = _named_base_groups(bundle.dense_sidecar_adapter)
    for parameter in (parameter for group in groups for parameter in group):
        parameter.requires_grad_(True)
    bundle.dense_sidecar_adapter.train()
    return [parameter for group in groups for parameter in group]


def assert_v34_trainable_surface(
    bundle: V30Bundle, optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, Any]:
    named = dict(bundle.dense_sidecar_adapter.named_parameters())
    authorized = {id(named[name]) for name in _TRAINABLE_NAMES}
    observed = {id(parameter) for module in bundle.checkpoint_modules.values()
                for parameter in module.parameters() if parameter.requires_grad}
    if observed != authorized:
        raise RuntimeError("V34 trainable surface is not exactly the four base tensors")
    if any(parameter.requires_grad for parameter in bundle.language.model.parameters()):
        raise RuntimeError("V34 Gemma decoder must remain frozen")
    if optimizer is not None:
        optimizer_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
        if optimizer_ids != authorized:
            raise RuntimeError("V34 optimizer contains an unauthorized tensor")
    counts = {
        "base_norm": sum(named[name].numel() for name in _BASE_NORM_NAMES),
        "base_projection": sum(named[name].numel() for name in _BASE_PROJECTION_NAMES),
    }
    if counts != {"base_norm": 3_072, "base_projection": 196_736}:
        raise RuntimeError(f"V34 base parameter counts changed: {counts}")
    return {
        "parameter_names": [f"dense_sidecar_adapter.{name}" for name in _TRAINABLE_NAMES],
        "group_parameter_counts": counts, "total_parameter_count": sum(counts.values()),
        "gemma_decoder_frozen": True, "all_lora_banks_frozen": True,
        "all_v33_learned_tensors_frozen": all(not named[name].requires_grad for name in _V33_FROZEN_NAMES),
        "every_other_parameter_frozen": True,
    }


def frozen_v34_state_sha256(bundle: V30Bundle) -> str:
    excluded = {f"dense_sidecar_adapter.{name}" for name in _TRAINABLE_NAMES}
    state = {f"{module_name}.{name}": value for module_name, module in bundle.checkpoint_modules.items()
             for name, value in module.state_dict().items()
             if f"{module_name}.{name}" not in excluded}
    return tensor_state_sha256(state)


def _optimizer(bundle: V30Bundle, settings: V34Settings) -> torch.optim.AdamW:
    norm, projection = _named_base_groups(bundle.dense_sidecar_adapter)
    optimizer = torch.optim.AdamW([
        {"name": "dense_sidecar_adapter.base_norm", "params": norm,
         "lr": settings.base_norm_learning_rate, "weight_decay": settings.weight_decay},
        {"name": "dense_sidecar_adapter.base_projection", "params": projection,
         "lr": settings.base_projection_learning_rate, "weight_decay": settings.weight_decay},
    ])
    assert_v34_trainable_surface(bundle, optimizer)
    return optimizer


def _gradient_snapshot(parameters: Sequence[torch.nn.Parameter]) -> list[torch.Tensor]:
    return [
        torch.zeros_like(parameter) if parameter.grad is None else parameter.grad.detach().clone()
        for parameter in parameters
    ]


def _gradient_norm(values: Sequence[torch.Tensor]) -> float:
    if not values:
        raise ValueError("V34 gradient norm requires at least one tensor")
    squared = sum(value.detach().float().square().sum() for value in values)
    return float(squared.sqrt().cpu())


def _incremental_gradient_norms(
    current: Sequence[torch.Tensor], previous: Sequence[torch.Tensor]
) -> dict[str, float]:
    if len(current) != 4 or len(previous) != 4:
        raise ValueError("V34 incremental gradient audit requires exactly four tensors")
    delta = [now - before for now, before in zip(current, previous, strict=True)]
    return {
        "base_norm": _gradient_norm(delta[:2]),
        "base_projection": _gradient_norm(delta[2:]),
    }


def _optimizer_step_audit(path: Path, expected_step: int, settings: V34Settings) -> None:
    payload = torch.load(path / "optimizer.pt", map_location="cpu", weights_only=True)
    groups = payload.get("param_groups") if isinstance(payload, Mapping) else None
    state = payload.get("state") if isinstance(payload, Mapping) else None
    if not isinstance(groups, list) or len(groups) != 2 or not isinstance(state, Mapping):
        raise ValueError("V34 optimizer must contain two fresh base-route groups")
    expected = (("dense_sidecar_adapter.base_norm", settings.base_norm_learning_rate, 2),
                ("dense_sidecar_adapter.base_projection", settings.base_projection_learning_rate, 2))
    ids: list[Any] = []
    for index, (group, (name, lr, count)) in enumerate(zip(groups, expected, strict=True)):
        parsed = _mapping(group, f"optimizer group {index}")
        params = parsed.get("params")
        if parsed.get("name") != name or float(parsed.get("lr", math.nan)) != lr or float(
            parsed.get("weight_decay", math.nan)
        ) != 0.0 or not isinstance(params, list) or len(params) != count:
            raise ValueError(f"V34 optimizer group {index} differs from its lock")
        ids.extend(params)
    if len(ids) != 4 or len(set(ids)) != 4 or set(state) != set(ids):
        raise ValueError("V34 fresh Adam state must cover exactly four tensors")
    for parameter_id in ids:
        entry = _mapping(state[parameter_id], f"optimizer state {parameter_id}")
        if set(entry) != {"step", "exp_avg", "exp_avg_sq"}:
            raise ValueError("V34 Adam state fields changed")
        step = entry["step"]
        step = step.item() if isinstance(step, torch.Tensor) and step.numel() == 1 else step
        if float(step) != expected_step:
            raise ValueError(f"V34 Adam state does not prove update {expected_step}")
        if any(not isinstance(entry[name], torch.Tensor) or not torch.isfinite(entry[name]).all()
               for name in ("exp_avg", "exp_avg_sq")):
            raise ValueError("V34 Adam moments are invalid")


def _metadata(
    *, source_metadata: Mapping[str, Any], config: Mapping[str, Any], bundle: V30Bundle,
    terminal: Mapping[str, Any], schedule_audit: Mapping[str, Any], cache_audit: Mapping[str, Any],
    qa_audit: Mapping[str, Any], separation_reference: PrefixSeparationReference,
    update_zero: Mapping[str, Any], train_records: Sequence[QARecord],
    validation_records: Sequence[QARecord], history: Sequence[Mapping[str, Any]],
    optimizer_step: int, best_update: int, best_validation: float,
    frozen_hash: str, surface: Mapping[str, Any], training_separation: Mapping[str, Any],
    validation_prefix: Mapping[str, Any], validation_ratios: Mapping[str, Any],
    family_teacher: Mapping[str, Any], early_gate: Mapping[str, Any] | None,
) -> dict[str, Any]:
    metadata = copy.deepcopy(dict(source_metadata))
    metadata.update({
        "config_hash": config_hash(dict(config)), "epoch": optimizer_step,
        "optimizer_step": optimizer_step, "best_epoch": best_update,
        "best_monitor_loss": best_validation, "monitor_name": "validation_answer_token_nll",
        "history": list(history),
        "dense_sidecar_adapter_state_sha256": bundle.dense_sidecar_adapter.state_sha256(),
    })
    legacy = copy.deepcopy(_mapping(metadata.get("v30_joint_pair"), "source v30 metadata"))
    legacy.update({
        "objective": "v34_base_only_broad_atomic_pair_plus_bounded_log_selectivity",
        "trainable_surface": dict(surface), "frozen_inherited_state_sha256": frozen_hash,
        "scene_cache": dict(cache_audit), "qa_dataset": dict(qa_audit),
        "update_zero_equivalence": dict(update_zero),
        "train_scene_ids": sorted({record.scene_id for record in train_records}),
        "validation_scene_ids": sorted({record.scene_id for record in validation_records}),
        "final_test_scene_ids_loaded": [], "oracle_environment_files_loaded": False,
        "question_dependent_scene_processing": False, "question_dependent_retrieval": False,
    })
    metadata["v30_joint_pair"] = legacy
    metadata["v34_base_surface"] = {
        "schema_version": 1, "artifact": "v34_diverse28_true_base_surface_training",
        "optimizer_step": optimizer_step,
        "conditional_v33_terminal_gate": {"path": terminal["path"], "sha256": terminal["sha256"]},
        "source_checkpoint": str(v34_contract(config).source_checkpoint),
        "source_file_sha256": dict(v34_contract(config).source_file_sha256),
        "source_optimizer_step": 64, "schedule": dict(schedule_audit),
        "exact_trainable_parameter_count": 199_808, "trainable_surface": dict(surface),
        "frozen_state_sha256": frozen_hash, "gemma_decoder_frozen": True,
        "all_lora_banks_frozen": True, "all_v33_learned_tensors_frozen": True,
        "train_scene_ids": sorted({record.scene_id for record in train_records}),
        "validation_scene_ids": sorted({record.scene_id for record in validation_records}),
        "deferred_final_scene_ids_loaded": [], "oracle_environment_files_loaded": False,
        "question_dependent_scene_processing": False, "question_dependent_retrieval": False,
        "separation_reference_sha256": separation_reference.audit_sha256,
        "separation_unique_changed_pair_count": 8, "separation_unrelated_pair_count": 112,
        "separation_uses_training_scenes_only": True,
        "separation_uses_question_or_answer_text": False,
        "separation_uses_oracle_environment_inputs": False,
        "training_separation": dict(training_separation),
        "validation_adapted_prefix_separation": dict(validation_prefix),
        "validation_adapted_prefix_ratios_from_update0": dict(validation_ratios),
        "validation_family_teacher_metrics": dict(family_teacher),
        "early_training_gate": None if early_gate is None else dict(early_gate),
        "development_progress_is_not_chat_promotion": True,
        "every_saved_arm_requires_independent_selection": True,
    }
    return metadata


def _save(path: Path, *, bundle: V30Bundle, metadata: dict[str, Any],
          optimizer: torch.optim.Optimizer | None) -> None:
    save_adapter_checkpoint(path, bundle.checkpoint_modules, metadata)
    if optimizer is not None:
        save_optimizer_checkpoint(path, optimizer)


def latest_v34_resume_checkpoint(output: Path, contract: V34Contract) -> Path | None:
    if not output.exists():
        return None
    if output.is_symlink() or not output.is_dir():
        raise ValueError(f"V34 output root must be a real directory: {output}")
    parsed: dict[int, Path] = {}
    for path in output.glob("update_*"):
        if not path.is_dir() or path.is_symlink():
            raise ValueError(f"V34 update path must be a real directory: {path}")
        match = _UPDATE_DIRECTORY.fullmatch(path.name)
        if match is None or int(match.group(1)) not in contract.saved_optimizer_steps:
            raise ValueError(f"V34 output contains an unauthorized arm: {path.name}")
        parsed[int(match.group(1))] = path
    complete = [step for step in contract.saved_optimizer_steps if step in parsed and all(
        (parsed[step] / name).is_file() for name in (
            "adapter.safetensors", TRAINING_METADATA_FILENAME, RUNTIME_METADATA_FILENAME,
            *(("optimizer.pt",) if step else ()),
        )
    )]
    if complete != list(contract.saved_optimizer_steps[:len(complete)]):
        raise ValueError("V34 complete arms are not a contiguous saved-step prefix")
    return None if not complete else parsed[complete[-1]]


def validate_v34_resume_checkpoint(
    *, config: Mapping[str, Any], output: Path, resume: Path, contract: V34Contract,
    settings: V34Settings, terminal: Mapping[str, Any], schedule_audit: Mapping[str, Any],
    cache_audit: Mapping[str, Any], separation_reference: PrefixSeparationReference,
) -> dict[str, Any]:
    latest = latest_v34_resume_checkpoint(output, contract)
    if latest is None or latest.resolve() != resume.resolve():
        raise ValueError("V34 resume must use the latest contiguous complete arm")
    metadata = json.loads((resume / TRAINING_METADATA_FILENAME).read_text(encoding="utf-8"))
    step = metadata.get("optimizer_step") if isinstance(metadata, Mapping) else None
    if not isinstance(step, int) or step not in contract.saved_optimizer_steps:
        raise ValueError("V34 resume metadata has an invalid optimizer step")
    stage = _mapping(metadata.get("v34_base_surface"), "resume v34 metadata")
    if metadata.get("config_hash") != config_hash(dict(config)) or stage.get(
        "conditional_v33_terminal_gate"
    ) != {"path": terminal["path"], "sha256": terminal["sha256"]}:
        raise ValueError("V34 resume config/terminal provenance changed")
    if _mapping(stage.get("schedule"), "resume schedule").get("schedule_sha256") != schedule_audit.get(
        "schedule_sha256"
    ) or stage.get("separation_reference_sha256") != separation_reference.audit_sha256:
        raise ValueError("V34 resume schedule/separation reference changed")
    if _mapping(metadata["v30_joint_pair"], "resume v30").get("scene_cache") != cache_audit:
        raise ValueError("V34 resume all-voxel cache audit changed")
    history = metadata.get("history")
    if not isinstance(history, list) or len(history) != step + 1:
        raise ValueError("V34 resume history is not one row per true update")
    if step >= contract.early_gate_optimizer_step:
        gate = _mapping(stage.get("early_training_gate"), "resume early gate")
        if gate.get("passed") is not True:
            raise RuntimeError("V34 cannot resume beyond its failed train-only update-32 gate")
    runtime = json.loads((resume / RUNTIME_METADATA_FILENAME).read_text(encoding="utf-8"))
    validate_runtime_checkpoint_metadata(runtime)
    if runtime != runtime_checkpoint_metadata(metadata):
        raise ValueError("V34 resume runtime metadata is not freshly sanitized")
    if step:
        _optimizer_step_audit(resume, step, settings)
    return dict(metadata)


def preflight_v34(config: Mapping[str, Any], *, require_qa: bool = True) -> dict[str, Any]:
    contract = v34_contract(config)
    terminal = require_v33_terminal_gate(config)
    source, _metadata_source = require_exact_v33_source(config)
    assert_deferred_final_scenes_absent(config)
    if require_qa:
        load_v31_qa_records(config)
    return {
        "artifact": "v34_diverse28_base_surface_preflight", "passed": True,
        "source_checkpoint": str(source), "source_optimizer_step": 64,
        "terminal_report_sha256": terminal["sha256"],
        "saved_optimizer_steps": list(contract.saved_optimizer_steps),
        "exact_trainable_parameter_count": 199_808, "gemma_decoder_frozen": True,
        "all_lora_banks_frozen": True, "all_v33_learned_tensors_frozen": True,
        "early_gate_training_scenes_only": True, "final_test_scenes_touched": False,
    }


def run_v34(*, config: dict[str, Any], output: Path, resume: Path | None = None) -> dict[str, Any]:
    if resume is None and output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty V34 output: {output}")
    contract = v34_contract(config)
    settings = v34_settings(config)
    terminal = require_v33_terminal_gate(config)
    source_checkpoint, pinned_source_metadata = require_exact_v33_source(config)
    assert_deferred_final_scenes_absent(config)
    seed = int(config["seed"])
    torch.manual_seed(seed)
    random.seed(seed)
    train_records, validation_records, qa_audit = load_v31_qa_records(config)
    train_pairs = build_exact_question_pair_units(train_records)
    validation_pairs = build_exact_question_pair_units(validation_records)
    schedule, schedule_audit = build_v34_schedule(
        train_records, train_pairs, settings=settings, seed=seed
    )
    approved_v29 = require_approved_v29_source(config)
    bundle = load_v30_bundle(config, approved_v29)
    scene_ids = sorted({record.scene_id for record in train_records} |
                       {record.scene_id for record in validation_records})
    caches, cache_audit = cache_pre_sidecar_scenes(
        bundle, scene_ids, allow_unpinned_source_scene_ids=_NEW_TRAIN_SCENES
    )
    if not (
        cache_audit.get("cache_boundary") == "complete_frozen_pre_sidecar_scene_stack"
        and cache_audit.get("all_voxels_covered") is True
        and cache_audit.get("question_inputs_to_scene_cache") is False
        and cache_audit.get("question_dependent_scene_processing") is False
    ):
        raise RuntimeError("V34 cache is not the proven complete pre-sidecar boundary")
    source_metadata = load_adapter_checkpoint(
        source_checkpoint, bundle.checkpoint_modules, device="cpu",
        metadata_filename=TRAINING_METADATA_FILENAME,
    )
    if source_metadata != pinned_source_metadata:
        raise RuntimeError("V34 source metadata changed during exact adapter load")
    source_tensors = load_file(source_checkpoint / "adapter.safetensors", device="cpu")
    source_state_hash = tensor_state_sha256(source_tensors)
    if module_collection_state_sha256(bundle.checkpoint_modules) != source_state_hash:
        raise RuntimeError("V34 loaded module collection is not bit-exact V33 update 64")
    all_trainable = freeze_for_v34(bundle)
    surface = assert_v34_trainable_surface(bundle)
    frozen_hash = frozen_v34_state_sha256(bundle)
    if surface["total_parameter_count"] != 199_808:
        raise RuntimeError("V34 changed its exact base-route parameter count")

    train_caches = {scene_id: caches[scene_id] for scene_id in contract.v31.train_scene_ids}
    separation_reference = build_prefix_separation_reference(
        train_pairs, train_caches, bundle, rms_floor=settings.separation_rms_floor
    )
    all_source_prefixes = _prefixes(scene_ids, caches, bundle, inference=True)
    source_prefix_hashes = {
        scene_id: prefix_sha256(prefix)
        for scene_id, prefix in all_source_prefixes.items()
    }
    repeated_prefix_hashes = {
        scene_id: prefix_sha256(prefix)
        for scene_id, prefix in _prefixes(
            scene_ids, caches, bundle, inference=True
        ).items()
    }
    if repeated_prefix_hashes != source_prefix_hashes:
        raise RuntimeError("V34 update-zero V33 source prefixes are not bit-exact on replay")
    baseline = validation_answer_nll(
        records=validation_records, caches=caches, bundle=bundle,
        batch_size=settings.broad_batch_size,
    )
    observed_nll = float(baseline["answer_token_nll"])
    source_nll = float(pinned_source_metadata["history"][-1]["validation_answer_token_nll"])
    if observed_nll != source_nll:
        raise RuntimeError(
            "V34 update-zero validation NLL is not bit-exact V33 update 64: "
            f"source={source_nll} observed={observed_nll}"
        )
    baseline_pairs = validation_pair_metrics(
        units=validation_pairs, caches=caches, bundle=bundle, margin=settings.pair_margin
    )
    baseline_family = validation_family_teacher_metrics(baseline_pairs)
    source_family = _mapping(
        pinned_source_metadata["history"][-1]["validation_family_teacher_metrics"],
        "V33 source family metrics",
    )
    if baseline_family != source_family:
        raise RuntimeError("V34 update-zero family margins are not bit-exact V33 update 64")
    baseline_validation_prefix = prefix_separation_diagnostics(
        units=validation_pairs, caches=caches, bundle=bundle
    )
    baseline_validation_ratios = prefix_separation_ratios(
        baseline_validation_prefix, baseline_validation_prefix
    )
    baseline_training_separation = training_separation_diagnostics(
        reference=separation_reference, caches=train_caches, bundle=bundle, settings=settings
    )
    update_zero = {
        "exact_v33_update64_source_tensors": True,
        "source_tensor_state_sha256": source_state_hash,
        "exact_v33_update64_source_prefixes": True,
        "source_prefix_sha256_by_scene": source_prefix_hashes,
        "source_prefix_scene_count": len(source_prefix_hashes),
        "exact_v33_update64_validation_nll": True,
        "source_validation_answer_token_nll": source_nll,
        "observed_validation_answer_token_nll": observed_nll,
        "fresh_adam_state": True, "question_dependent_scene_processing": False,
        "question_dependent_retrieval": False, "oracle_environment_files_loaded": False,
    }
    optimizer = _optimizer(bundle, settings)
    history: list[dict[str, Any]] = [{
        "optimizer_update": 0, "validation_answer_token_nll": observed_nll,
        "validation_pair_metrics": baseline_pairs,
        "validation_family_teacher_metrics": baseline_family,
        "validation_adapted_prefix_separation": baseline_validation_prefix,
        "validation_adapted_prefix_ratios_from_update0": baseline_validation_ratios,
        "training_separation": baseline_training_separation,
        "update_0_equivalence_verified": True, "saved_checkpoint": True,
    }]
    best_update = 0
    best_validation = observed_nll
    start_step = 0
    accepted_early_gate: Mapping[str, Any] | None = None
    output.mkdir(parents=True, exist_ok=True)
    if resume is not None:
        resume_metadata = validate_v34_resume_checkpoint(
            config=config, output=output, resume=resume, contract=contract,
            settings=settings, terminal=terminal, schedule_audit=schedule_audit,
            cache_audit=cache_audit, separation_reference=separation_reference,
        )
        loaded = load_adapter_checkpoint(
            resume, bundle.checkpoint_modules, device="cpu", metadata_filename=TRAINING_METADATA_FILENAME
        )
        if loaded != resume_metadata:
            raise RuntimeError("V34 resume metadata changed during adapter load")
        start_step = int(resume_metadata["optimizer_step"])
        if start_step:
            load_optimizer_checkpoint(resume, optimizer, bundle.language.device)
        history = list(resume_metadata["history"])
        best_update = int(resume_metadata["best_epoch"])
        best_validation = float(resume_metadata["best_monitor_loss"])
        resume_stage = _mapping(
            resume_metadata.get("v34_base_surface"), "resume v34 metadata"
        )
        if start_step >= contract.early_gate_optimizer_step:
            accepted_early_gate = _mapping(
                resume_stage.get("early_training_gate"), "resume accepted early gate"
            )
        if frozen_v34_state_sha256(bundle) != frozen_hash:
            raise RuntimeError("V34 resume changed a frozen V33/Gemma/scene tensor")
    else:
        metadata0 = _metadata(
            source_metadata=pinned_source_metadata, config=config, bundle=bundle,
            terminal=terminal, schedule_audit=schedule_audit, cache_audit=cache_audit,
            qa_audit=qa_audit, separation_reference=separation_reference,
            update_zero=update_zero, train_records=train_records,
            validation_records=validation_records, history=history, optimizer_step=0,
            best_update=0, best_validation=best_validation, frozen_hash=frozen_hash,
            surface=surface, training_separation=baseline_training_separation,
            validation_prefix=baseline_validation_prefix,
            validation_ratios=baseline_validation_ratios, family_teacher=baseline_family,
            early_gate=None,
        )
        _save(output / "update_000", bundle=bundle, metadata=metadata0, optimizer=None)
        _save(output / "best", bundle=bundle, metadata=metadata0, optimizer=None)
        saved0 = load_file(output / "update_000" / "adapter.safetensors", device="cpu")
        if tensor_state_sha256(saved0) != source_state_hash:
            raise RuntimeError("V34 saved update zero is not bit-exact V33 update 64")

    norm_parameters, projection_parameters = _named_base_groups(bundle.dense_sidecar_adapter)
    for item in schedule[start_step:]:
        step = item.optimizer_step
        bundle.dense_sidecar_adapter.train()
        bundle.lora_installation.eval()
        optimizer.zero_grad(set_to_none=True)
        broad = cached_broad_answer_nll(
            cache=caches[item.broad_records[0].scene_id], records=item.broad_records, bundle=bundle
        )
        (settings.broad_nll_weight * broad).backward()
        broad_gradient = _gradient_snapshot(all_trainable)
        zero_gradient = [torch.zeros_like(value) for value in broad_gradient]
        gradient_by_loss = {
            "broad": _incremental_gradient_norms(broad_gradient, zero_gradient)
        }
        broad_value = float(broad.detach().cpu())
        del broad
        separation_prefixes = _prefixes(
            sorted(separation_reference.source_prefixes), train_caches, bundle, inference=False
        )
        separation_loss, separation_raw = separation_loss_and_diagnostics(
            prefixes=separation_prefixes, reference=separation_reference, settings=settings
        )
        separation_loss.backward()
        after_separation_gradient = _gradient_snapshot(all_trainable)
        gradient_by_loss["separation"] = _incremental_gradient_norms(
            after_separation_gradient, broad_gradient
        )
        separation_value = float(separation_loss.detach().cpu())
        separation_step_summary = summarize_separation(separation_raw)
        del separation_loss, separation_raw, separation_prefixes
        pair_language, pair_hinge, pair_diagnostics = paired_canonical_answer_objective(
            units=item.pair_units, caches=caches, bundle=bundle, margin=settings.pair_margin
        )
        pair_objective = settings.pair_language_nll_weight * pair_language + settings.pair_margin_weight * pair_hinge
        pair_objective.backward()
        after_pair_gradient = _gradient_snapshot(all_trainable)
        gradient_by_loss["pair"] = _incremental_gradient_norms(
            after_pair_gradient, after_separation_gradient
        )
        pair_language_value = float(pair_language.detach().cpu())
        pair_hinge_value = float(pair_hinge.detach().cpu())
        pair_side_accuracy = float(pair_diagnostics["side_accuracy"].detach().cpu())
        del pair_language, pair_hinge, pair_diagnostics, pair_objective
        assert_v34_trainable_surface(bundle, optimizer)
        if any(parameter.grad is None for parameter in all_trainable):
            raise RuntimeError("V34 one or more base tensors lacks a gradient")
        if any(not torch.isfinite(parameter.grad).all() for parameter in all_trainable):
            raise RuntimeError("V34 base-route gradient is nonfinite")
        preclip = {
            "base_norm": float(torch.nn.utils.clip_grad_norm_(
                norm_parameters, settings.base_norm_gradient_clip_norm
            ).detach().cpu()),
            "base_projection": float(torch.nn.utils.clip_grad_norm_(
                projection_parameters, settings.base_projection_gradient_clip_norm
            ).detach().cpu()),
        }
        optimizer.step()
        if frozen_v34_state_sha256(bundle) != frozen_hash:
            raise RuntimeError("V34 changed a frozen V33/Gemma/scene tensor")
        validate_dense_sidecar_adapter_state(bundle.dense_sidecar_adapter)
        should_save = step in contract.saved_optimizer_steps
        validation = validation_answer_nll(
            records=validation_records, caches=caches, bundle=bundle,
            batch_size=settings.broad_batch_size,
        ) if should_save else None
        pair_validation = validation_pair_metrics(
            units=validation_pairs, caches=caches, bundle=bundle, margin=settings.pair_margin
        ) if should_save else None
        family = validation_family_teacher_metrics(pair_validation) if pair_validation else None
        validation_prefix = prefix_separation_diagnostics(
            units=validation_pairs, caches=caches, bundle=bundle
        ) if should_save else None
        validation_ratios = prefix_separation_ratios(
            validation_prefix, baseline_validation_prefix
        ) if validation_prefix else None
        train_separation = training_separation_diagnostics(
            reference=separation_reference, caches=train_caches, bundle=bundle, settings=settings
        ) if should_save else None
        if step == contract.early_gate_optimizer_step:
            accepted_early_gate = v34_early_training_gate(train_separation, contract)
        early_gate = accepted_early_gate
        validation_value = float(validation["answer_token_nll"]) if validation else None
        if validation_value is not None and validation_value < best_validation:
            best_update, best_validation = step, validation_value
        objective_value = broad_value + pair_language_value + 4.0 * pair_hinge_value + separation_value
        history.append({
            "optimizer_update": step, "true_optimizer_step": True,
            "train_broad_answer_token_nll": broad_value,
            "train_pair_answer_token_nll": pair_language_value,
            "train_pair_margin_hinge": pair_hinge_value,
            "train_pair_side_accuracy": pair_side_accuracy,
            "train_separation_objective": separation_value,
            "train_separation_step_summary": separation_step_summary,
            "train_objective": objective_value, "preclip_gradient_norm_by_group": preclip,
            "preclip_gradient_norm_by_loss_and_group": gradient_by_loss,
            "separate_group_clipping": True, "validation_answer_token_nll": validation_value,
            "validation_pair_metrics": pair_validation,
            "validation_family_teacher_metrics": family,
            "validation_adapted_prefix_separation": validation_prefix,
            "validation_adapted_prefix_ratios_from_update0": validation_ratios,
            "training_separation": train_separation, "early_training_gate": early_gate,
            "saved_checkpoint": should_save,
        })
        if not should_save:
            continue
        metadata = _metadata(
            source_metadata=pinned_source_metadata, config=config, bundle=bundle,
            terminal=terminal, schedule_audit=schedule_audit, cache_audit=cache_audit,
            qa_audit=qa_audit, separation_reference=separation_reference,
            update_zero=update_zero, train_records=train_records,
            validation_records=validation_records, history=history, optimizer_step=step,
            best_update=best_update, best_validation=best_validation, frozen_hash=frozen_hash,
            surface=surface, training_separation=train_separation,
            validation_prefix=validation_prefix, validation_ratios=validation_ratios,
            family_teacher=family, early_gate=early_gate,
        )
        _save(output / f"update_{step:03d}", bundle=bundle, metadata=metadata, optimizer=optimizer)
        if best_update == step:
            _save(output / "best", bundle=bundle, metadata=metadata, optimizer=None)
        print(json.dumps({
            "phase": "v34_true_base_surface_checkpoint", "optimizer_step": step,
            "validation_answer_token_nll": validation_value,
            "training_changed_selectivity_ratio": train_separation[
                "changed_selectivity_ratio_geometric_mean"
            ],
            "training_changed_coverage": train_separation["changed_selectivity_over_1_02_count"],
            "training_unrelated_median_ratio": train_separation["unrelated_ratio_median"],
            "training_unrelated_p90_abs_log_ratio": train_separation["unrelated_abs_log_ratio_p90"],
            "best_update": best_update,
        }, sort_keys=True), flush=True)
        if (
            step == contract.early_gate_optimizer_step
            and accepted_early_gate["passed"] is not True
        ):
            raise RuntimeError(
                "V34 update-32 train-only selectivity gate failed; stop this bounded isolation arm"
            )
    return {
        "schema_version": 1, "artifact": "v34_diverse28_true_base_surface_training",
        "output": str(output), "best_checkpoint": str(output / "best"),
        "best_update": best_update, "baseline_validation_answer_token_nll": observed_nll,
        "best_validation_answer_token_nll": best_validation, "optimizer_updates": 64,
        "resumed_from_optimizer_step": start_step,
        "saved_optimizer_steps": list(contract.saved_optimizer_steps),
        "trainable_surface": surface, "schedule": schedule_audit,
        "v33_terminal_report_sha256": terminal["sha256"],
        "final_test_scene_ids_loaded": [], "question_dependent_scene_processing": False,
        "question_dependent_retrieval": False, "oracle_environment_files_loaded": False,
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
    config = load_config(args.config)
    if args.preflight_only:
        report = preflight_v34(config)
    else:
        output = _resolve(args.output)
        resume = _resolve(args.resume) if args.resume is not None else None
        if args.resume_latest:
            resume = latest_v34_resume_checkpoint(output, v34_contract(config))
            if resume is None:
                raise FileNotFoundError("V34 has no complete checkpoint to resume")
        report = run_v34(config=config, output=output, resume=resume)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "PrefixSeparationReference", "V34Contract", "V34Settings", "assert_v34_trainable_surface",
    "build_prefix_separation_reference", "build_v34_schedule", "freeze_for_v34",
    "latest_v34_resume_checkpoint", "physical_pair_sets", "preflight_v34",
    "require_v33_terminal_gate", "separation_loss_and_diagnostics",
    "training_separation_diagnostics", "v34_contract", "v34_early_training_gate", "v34_settings",
]
