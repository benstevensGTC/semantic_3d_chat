"""Model-free contract for V96's exact atomic-pair repair successor.

V96 freezes V95's exact nine-bank, non-promoted fixed-final parent and adds
one disjoint rank-8 query-projection bank.  Training uses only the forty
existing training scenes.  Exact canonical answer strings are preserved for
teacher forcing; normalization is used only to derive class IDs and weights.

This module may authenticate V95's row-free aggregate development result, but
it never opens development questions, labels, predictions, deferred-final
artifacts, a full Gemma model, or an optimizer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import torch
import yaml
from safetensors import safe_open
from torch import nn

from semantic_3d_chat.chat.file_audit import FileAccessAudit
from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.evaluation.metrics import normalize_answer
from semantic_3d_chat.evaluation.v85_strict_multiscene_preflight import (
    canonical_sha256_v85,
    resolve_v85,
    sha256_file_v85,
)
from semantic_3d_chat.evaluation.v95_known_development_common import (
    authenticate_fixed_final_candidate_v95,
)
from semantic_3d_chat.evaluation.v95_strict_causal_successor_preflight import (
    DEFERRED_FINAL_SCENES,
    PRIOR_EVALUATION_SCENES,
    TRAINING_SCENES,
    authenticate_training_sources_v95,
    load_config_v95,
)
from semantic_3d_chat.language.lora import (
    LoRASettings,
    initialize_lora_adapter_state,
    install_lora_adapters,
)
from semantic_3d_chat.language.prefix_injection import prefix_sha256
from semantic_3d_chat.training.v82_reader_artifacts import load_v82_cache

CONFIG: Final[Path] = Path("configs/experiments/gemma4_v96_atomic_pair_repair.yaml")
FRESH_BANK_NAME: Final[str] = "v96_atomic_pair_repair_bridge"
TARGET_MODULES: Final[tuple[str, ...]] = ("model.language_model.layers.9.self_attn.q_proj",)
PINNED_TENSORS: Final[dict[str, list[int]]] = {
    TARGET_MODULES[0] + ".weight": [4096, 1536],
}
FRESH_PARAMETER_COUNT: Final[int] = 45_056
EXPECTED_INITIAL_STATE_SHA256: Final[str] = (
    "a52c4aa9915006c7f04721899723cb684ee65b773a32aed8d03c9083e0075b2b"
)
EXPECTED_FROZEN_BANK_COUNT: Final[int] = 9
EXPECTED_FROZEN_PARAMETER_COUNT: Final[int] = 819_200
EXPECTED_TOTAL_ADAPTER_PARAMETER_COUNT: Final[int] = 864_256
EXPECTED_RETENTION_STEPS: Final[int] = 1_920
EXPECTED_CHANGED_PAIR_STEPS: Final[int] = 264
EXPECTED_INVARIANT_PAIR_STEPS: Final[int] = 96
EXPECTED_MICRO_STEPS: Final[int] = 2_280
EXPECTED_OPTIMIZER_UPDATES: Final[int] = 285
EXPECTED_TOTAL_NLL_FORWARDS: Final[int] = 3_168
PREREG_ARTIFACT: Final[str] = "gemma4_v96_atomic_pair_repair_preregistration_v1"
PREFLIGHT_ARTIFACT: Final[str] = "gemma4_v96_atomic_pair_repair_cpu_preflight_v1"
_DRAFT: Final[str] = "draft_contract_unsealed_training_not_authorized"
_SEALED: Final[str] = "sealed_before_v96_full_model_load"
_HEX64: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{64}")
_ANSWER_TYPE_QUOTAS: Final[dict[str, int]] = {
    "attribute": 14,
    "count": 14,
    "metric": 13,
    "orientation": 13,
    "presence": 14,
    "spatial_relation": 14,
    "support": 14,
}


@dataclass(frozen=True)
class RowV96:
    scene_id: str
    question_id: str
    question: str
    answer: str
    answer_class: str
    answer_type: str
    pair_id: str
    paired_scene_id: str
    question_key: str
    change_type: str
    expected_change: bool

    @property
    def key(self) -> tuple[str, str]:
        return self.scene_id, self.question_id


@dataclass(frozen=True)
class PairUnitV96:
    pair_id: str
    question_key: str
    change_type: str
    answer_type: str
    left: RowV96
    right: RowV96

    @property
    def key(self) -> tuple[str, str]:
        return self.pair_id, self.question_key


@dataclass(frozen=True)
class TrainingStepV96:
    kind: str
    round_index: int
    row: RowV96 | None = None
    unit: PairUnitV96 | None = None

    def identity(self) -> list[Any]:
        if self.kind == "retention" and self.row is not None and self.unit is None:
            return [self.kind, self.round_index, self.row.scene_id, self.row.question_id]
        if self.kind in {"changed_pair", "invariant_pair"} and self.unit is not None:
            return [self.kind, self.round_index, self.unit.pair_id, self.unit.question_key]
        raise ValueError("V96 malformed training step")


def _require(value: object, expected: object, label: str) -> None:
    if value != expected:
        raise ValueError(f"V96 {label} changed")


def _require_hash(value: object, label: str, *, draft: bool) -> None:
    if isinstance(value, str) and _HEX64.fullmatch(value):
        return
    if draft and value == "TO_FILL":
        return
    raise ValueError(f"V96 {label} is not sealed")


def _leaf_path(path: str | Path) -> Path:
    """Return an absolute path without resolving the final symlink."""

    value = Path(path).expanduser()
    if not value.is_absolute():
        value = PROJECT_ROOT / value
    return Path(os.path.abspath(value))


def _strict_json(path: str | Path) -> dict[str, Any]:
    source = _leaf_path(path)
    if source.is_symlink() or not source.is_file():
        raise FileNotFoundError(source)
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"V96 JSON must contain one object: {source}")
    return value


def load_config_v96(path: str | Path = CONFIG, *, allow_draft: bool = True) -> dict[str, Any]:
    source = _leaf_path(path)
    if source.is_symlink() or not source.is_file():
        raise FileNotFoundError(source)
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or set(payload) != {"v96"}:
        raise ValueError("V96 config must contain exactly one v96 mapping")
    config = payload["v96"]
    if not isinstance(config, Mapping):
        raise TypeError("V96 config payload must be a mapping")
    _require(config.get("schema_version"), 96, "schema version")
    _require(
        config.get("artifact"),
        "gemma4_v96_atomic_pair_repair_direct_memory_lora_v1",
        "artifact",
    )
    status = config.get("status")
    if status not in ({_DRAFT, _SEALED} if allow_draft else {_SEALED}):
        raise ValueError("V96 config status is not authorized")
    _require(config.get("seed"), 960096, "seed")
    draft = status == _DRAFT

    sources = config.get("sources")
    if not isinstance(sources, Mapping):
        raise TypeError("V96 sources must be a mapping")
    for key in (
        "runtime_config_sha256",
        "training_qa_sha256",
        "train_memory_tensor_sha256",
        "train_memory_metadata_sha256",
        "development_memory_tensor_sha256",
        "development_memory_metadata_sha256",
        "frozen_v95_config_sha256",
        "frozen_v95_bridge_sha256",
        "frozen_v95_bridge_metadata_sha256",
        "v95_training_report_sha256",
        "v95_known_development_structured_sha256",
        "v95_known_development_final_score_sha256",
        "v95_known_development_evidence_sha256",
        "preflight_source_sha256",
        "trainer_source_sha256",
        "model_blob_sha256_identity",
    ):
        _require_hash(sources.get(key), key, draft=draft)
    _require(sources.get("model_id"), "google/gemma-4-E2B-it", "model ID")
    _require(
        sources.get("model_revision"),
        "3e22461f65e89153144f8adb70e3b8c2cc9845a7",
        "model revision",
    )

    _require(
        config.get("strict_input_contract"),
        {
            "shape_per_scene": [1, 738, 1536],
            "native_boi_retained": True,
            "native_eoi_retained": True,
            "continuous_environment_payload_tokens": 736,
            "compiled_before_question": True,
            "reused_byte_identically_across_questions": True,
            "supplied_directly_to_native_gemma_image_prefix": True,
            "all_memory_slots_retained": True,
            "question_derived_environmental_tokens": 0,
            "question_conditioned_environmental_readout": False,
            "question_dependent_scene_processing": False,
            "question_dependent_retrieval": False,
            "semantic_or_spatial_top_k_selection": False,
            "control_tokens": 0,
            "environmental_text_inputs": [],
        },
        "strict direct-memory contract",
    )
    pool = config.get("training_pool")
    if not isinstance(pool, Mapping):
        raise TypeError("V96 training pool must be a mapping")
    for key, expected in {
        "scene_count": 40,
        "pair_count": 20,
        "row_count": 960,
        "changed_unit_count": 66,
        "changed_side_count": 132,
        "invariant_unit_count": 414,
        "invariant_side_count": 828,
        "fixed_invariant_subset_unit_count": 96,
        "fixed_invariant_subset_side_count": 192,
        "answer_class_count": 29,
        "canonical_list_answer": "book, cube",
        "canonical_list_answer_row_count": 22,
        "exact_answer_strings_preserved_for_teacher_forcing": True,
        "normalization_used_only_for_answer_class_ids": True,
    }.items():
        _require(pool.get(key), expected, f"training pool {key}")
    for key in (
        "row_inventory_sha256",
        "raw_answer_inventory_sha256",
        "scene_inventory_sha256",
        "pair_inventory_sha256",
        "answer_class_inventory_sha256",
        "balanced_class_weight_inventory_sha256",
        "changed_family_weight_inventory_sha256",
        "invariant_family_weight_inventory_sha256",
    ):
        _require_hash(pool.get(key), key, draft=draft)

    frozen = config.get("frozen_stack")
    if not isinstance(frozen, Mapping):
        raise TypeError("V96 frozen stack must be a mapping")
    for key, expected in {
        "frozen_bank_count": EXPECTED_FROZEN_BANK_COUNT,
        "frozen_adapter_parameter_count": EXPECTED_FROZEN_PARAMETER_COUNT,
        "v95_bank_name": "v95_strict_causal_successor_bridge",
        "v95_bank_state_sha256": "53404c733586ebd25caa440f822a4d4af6cc3dbb71bf4f6b6f94af23f3a2492a",
        "base_gemma_frozen": True,
        "merged_weights": False,
    }.items():
        _require(frozen.get(key), expected, f"frozen stack {key}")

    bridge = config.get("bridge")
    if not isinstance(bridge, Mapping):
        raise TypeError("V96 bridge must be a mapping")
    for key, expected in {
        "bank_name": FRESH_BANK_NAME,
        "target_modules": list(TARGET_MODULES),
        "pinned_weight_shapes": PINNED_TENSORS,
        "pinned_weight_dtype": "BF16",
        "rank": 8,
        "alpha": 16.0,
        "dropout": 0.0,
        "trainable_parameter_count": FRESH_PARAMETER_COUNT,
        "initialization_algorithm": "cpu_kaiming_uniform_a_exact_zero_b",
        "initialization_seed": 960096,
        "disjoint_from_all_frozen_bank_targets": True,
        "total_bank_count_after_install": 10,
        "total_adapter_parameter_count_after_install": (EXPECTED_TOTAL_ADAPTER_PARAMETER_COUNT),
    }.items():
        _require(bridge.get(key), expected, f"bridge {key}")
    _require_hash(
        bridge.get("expected_initial_state_sha256"),
        "initial state",
        draft=draft,
    )
    if bridge.get("expected_initial_state_sha256") != "TO_FILL":
        _require(
            bridge.get("expected_initial_state_sha256"),
            EXPECTED_INITIAL_STATE_SHA256,
            "initial state",
        )

    training = config.get("training")
    if not isinstance(training, Mapping):
        raise TypeError("V96 training protocol must be a mapping")
    for key, expected in {
        "optimizer": "AdamW",
        "retention_passes": 2,
        "retention_rows_per_pass": 960,
        "total_retention_steps": EXPECTED_RETENTION_STEPS,
        "changed_pair_rounds": 4,
        "changed_pair_units_per_round": 66,
        "total_changed_pair_steps": EXPECTED_CHANGED_PAIR_STEPS,
        "pair_question_identity": "byte_exact_left_right_required",
        "invariant_subset_rounds": 1,
        "invariant_subset_units_per_round": 96,
        "total_invariant_pair_steps": EXPECTED_INVARIANT_PAIR_STEPS,
        "invariant_subset_question_identity": "byte_exact_left_right_required",
        "total_micro_steps": EXPECTED_MICRO_STEPS,
        "microbatch_size": 1,
        "gradient_accumulation_steps": 8,
        "optimizer_updates": EXPECTED_OPTIMIZER_UPDATES,
        "schedule_policy": (
            "hash_interleaved_two_full_retention_passes_four_exact_changed_pair_rounds_"
            "one_balanced_invariant_subset"
        ),
        "schedule_seed": 960096,
        "invariant_subset_policy": "answer_independent_answer_type_balanced_hash_selection",
        "learning_rate": 0.000075,
        "weight_decay": 0.0,
        "gradient_clip_norm": 1.0,
        "retention_balanced_ce_weight": 1.0,
        "balanced_ce_formula": "normalized_inverse_sqrt_answer_class_frequency",
        "balanced_ce_mean_over_rows": 1.0,
        "changed_family_weight_formula": "normalized_inverse_sqrt_changed_unit_frequency",
        "changed_family_weight_mean_over_units": 1.0,
        "pair_correct_ce_weight": 1.0,
        "within_memory_answer_margin_weight": 1.5,
        "within_memory_answer_target_margin_nll": 0.75,
        "across_memory_causal_margin_weight": 0.75,
        "across_memory_causal_target_margin_nll": 0.5,
        "pair_side_smoothmax_temperature": 0.25,
        "invariant_family_weight_formula": ("normalized_inverse_sqrt_selected_unit_frequency"),
        "invariant_family_weight_mean_over_units": 1.0,
        "invariant_correct_ce_weight": 1.0,
        "invariant_nll_consistency_weight": 0.5,
        "invariant_nll_consistency_tolerance": 0.1,
        "zero_payload_training_rows": 0,
        "permutation_training_rows": 0,
        "controls_policy": "inherited_frozen_v95_behavior_post_fixed_final_gate_only",
        "broad_nll_forward_evaluations": 1920,
        "changed_pair_nll_forward_evaluations": 1056,
        "invariant_pair_nll_forward_evaluations": 192,
        "auxiliary_nll_forward_evaluations": 888,
        "total_nll_forward_evaluations": EXPECTED_TOTAL_NLL_FORWARDS,
        "v95_measured_seconds_per_nll_forward": 2.2671,
        "estimated_wall_time_seconds_from_v95_measured_ratio": 7182,
        "wall_time_budget_seconds": 9000,
        "checkpoint_every_optimizer_updates": 15,
        "deterministic_resume": True,
        "checkpoint_selection": (
            "fixed_final_update_285_before_known_development_or_deferred_generation"
        ),
        "intermediate_behavior_selection": False,
    }.items():
        _require(training.get(key), expected, f"training {key}")
    for key in ("schedule_sha256", "invariant_subset_sha256"):
        _require_hash(training.get(key), key, draft=draft)

    outputs = config.get("outputs")
    expected_outputs = {
        "preregistration": (
            "reports/gemma4/metrics/gemma4_v96_atomic_pair_repair_preregistration.json"
        ),
        "cpu_preflight": (
            "reports/gemma4/metrics/gemma4_v96_atomic_pair_repair_cpu_preflight.json"
        ),
        "topology_smoke": (
            "reports/gemma4/metrics/gemma4_v96_atomic_pair_repair_topology_smoke.json"
        ),
        "work_root": "data_gemma4/checkpoints/v96_atomic_pair_repair_work",
        "fixed_final_candidate": "reports/gemma4/artifacts/v96_atomic_pair_repair_final",
        "training_report": ("reports/gemma4/metrics/gemma4_v96_atomic_pair_repair_training.json"),
    }
    _require(outputs, expected_outputs, "output paths")
    gate = config.get("known_development_gate")
    if not isinstance(gate, Mapping):
        raise TypeError("V96 known-development gate must be a mapping")
    for key, expected in {
        "row_count": 216,
        "changed_side_total": 24,
        "changed_unit_total": 12,
        "invariant_side_total": 192,
        "v95_reference_correct": 167,
        "v95_reference_changed_side_correct": 13,
        "v95_reference_complete_changed_units": 1,
        "v95_reference_prediction_changed_units": 2,
        "v95_reference_invariant_false_changes": 20,
        "v96_correct_minimum": 160,
        "changed_side_correct_minimum": 15,
        "complete_changed_units_minimum": 4,
        "prediction_changed_units_minimum": 7,
        "invariant_false_change_maximum": 20,
        "fixed_final_checkpoint_may_not_change_after_gate": True,
    }.items():
        _require(gate.get(key), expected, f"known-development gate {key}")
    return dict(config)


def answer_class_id_v96(answer: str) -> str:
    normalized = normalize_answer(answer)
    if not normalized:
        raise ValueError("V96 answer normalizes to empty")
    return "answer_" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:20]


def _row_inventory(rows: Sequence[RowV96]) -> list[list[Any]]:
    return sorted(
        [
            [
                row.scene_id,
                row.question_id,
                row.pair_id,
                row.question_key,
                row.answer,
                row.answer_class,
                row.answer_type,
                row.expected_change,
            ]
            for row in rows
        ]
    )


def load_training_rows_v96(config: Mapping[str, Any]) -> tuple[RowV96, ...]:
    """Load training rows without destroying canonical answer punctuation."""

    source = _leaf_path(config["sources"]["training_qa"])
    if source.is_symlink() or not source.is_file():
        raise FileNotFoundError(source)
    required = {
        "scene_id",
        "question_id",
        "question",
        "answer",
        "answer_type",
        "counterfactual_pair_id",
        "counterfactual_paired_scene_id",
        "counterfactual_question_key",
        "counterfactual_change_type",
        "counterfactual_expected_change",
    }
    rows: list[RowV96] = []
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"V96 invalid training JSON at line {line_number}") from error
        if not isinstance(raw, Mapping) or not required <= set(raw):
            raise ValueError(f"V96 training row fields changed at line {line_number}")
        string_fields = tuple(required - {"counterfactual_expected_change"})
        if any(not isinstance(raw[field], str) or not raw[field] for field in string_fields):
            raise ValueError(f"V96 training string field changed at line {line_number}")
        if type(raw["counterfactual_expected_change"]) is not bool:
            raise TypeError("V96 expected-change field must be boolean")
        canonical_answer = raw["answer"]
        rows.append(
            RowV96(
                scene_id=raw["scene_id"],
                question_id=raw["question_id"],
                question=raw["question"],
                answer=canonical_answer,
                answer_class=answer_class_id_v96(canonical_answer),
                answer_type=raw["answer_type"],
                pair_id=raw["counterfactual_pair_id"],
                paired_scene_id=raw["counterfactual_paired_scene_id"],
                question_key=raw["counterfactual_question_key"],
                change_type=raw["counterfactual_change_type"],
                expected_change=raw["counterfactual_expected_change"],
            )
        )
    scenes = tuple(sorted({row.scene_id for row in rows}))
    pairs = {row.pair_id for row in rows}
    pool = config["training_pool"]
    if (
        len(rows) != 960
        or len({row.key for row in rows}) != 960
        or scenes != TRAINING_SCENES
        or len(pairs) != 20
        or set(scenes).intersection(PRIOR_EVALUATION_SCENES)
        or set(scenes).intersection(DEFERRED_FINAL_SCENES)
        or sum(row.answer == "book, cube" for row in rows) != 22
        or any(row.answer == "book cube" for row in rows)
    ):
        raise ValueError("V96 exact canonical training inventory changed")
    observed = {
        "row_inventory_sha256": canonical_sha256_v85(_row_inventory(rows)),
        "raw_answer_inventory_sha256": canonical_sha256_v85(
            sorted(Counter(row.answer for row in rows).items())
        ),
        "scene_inventory_sha256": canonical_sha256_v85(scenes),
        "pair_inventory_sha256": canonical_sha256_v85(sorted(pairs)),
    }
    for key, value in observed.items():
        expected = pool[key]
        if expected != "TO_FILL" and expected != value:
            raise ValueError(f"V96 {key} changed")
    changed, invariant = pair_units_v96(rows)
    if len(changed) != 66 or len(invariant) != 414:
        raise ValueError("V96 changed/invariant unit counts changed")
    return tuple(rows)


def pair_units_v96(
    rows: Sequence[RowV96],
) -> tuple[tuple[PairUnitV96, ...], tuple[PairUnitV96, ...]]:
    grouped: dict[tuple[str, str], list[RowV96]] = defaultdict(list)
    for row in rows:
        grouped[(row.pair_id, row.question_key)].append(row)
    changed: list[PairUnitV96] = []
    invariant: list[PairUnitV96] = []
    for (pair_id, question_key), members in sorted(grouped.items()):
        if len(members) != 2 or len({row.scene_id for row in members}) != 2:
            raise ValueError("V96 pair unit must contain exactly two scene sides")
        left, right = sorted(members, key=lambda row: row.scene_id)
        if (
            left.paired_scene_id != right.scene_id
            or right.paired_scene_id != left.scene_id
            or left.question != right.question
            or left.answer_type != right.answer_type
            or left.change_type != right.change_type
            or left.expected_change != right.expected_change
        ):
            raise ValueError("V96 atomic counterpart linkage changed")
        unit = PairUnitV96(
            pair_id,
            question_key,
            left.change_type,
            left.answer_type,
            left,
            right,
        )
        if left.expected_change:
            if left.answer_class == right.answer_class:
                raise ValueError("V96 changed pair has no canonical answer contrast")
            changed.append(unit)
        else:
            if left.answer != right.answer:
                raise ValueError("V96 invariant pair changed its exact canonical answer")
            invariant.append(unit)
    if len(changed) != 66 or len(invariant) != 414:
        raise ValueError("V96 pair-unit inventory changed")
    return tuple(changed), tuple(invariant)


def balanced_class_weights_v96(
    config: Mapping[str, Any], rows: Sequence[RowV96]
) -> dict[str, float]:
    counts = Counter(row.answer_class for row in rows)
    raw = {key: 1.0 / math.sqrt(value) for key, value in counts.items()}
    normalizer = len(rows) / sum(raw[row.answer_class] for row in rows)
    weights = {key: raw[key] * normalizer for key in sorted(raw)}
    pool = config["training_pool"]
    hashes = {
        "answer_class_inventory_sha256": canonical_sha256_v85(sorted(counts.items())),
        "balanced_class_weight_inventory_sha256": canonical_sha256_v85(sorted(weights.items())),
    }
    if (
        len(counts) != 29
        or abs(sum(weights[row.answer_class] for row in rows) / len(rows) - 1.0) > 1e-12
    ):
        raise ValueError("V96 balanced class weights changed")
    for key, value in hashes.items():
        expected = pool[key]
        if expected != "TO_FILL" and expected != value:
            raise ValueError(f"V96 {key} changed")
    return weights


def family_weights_v96(units: Sequence[PairUnitV96]) -> dict[str, float]:
    counts = Counter(unit.change_type for unit in units)
    raw = {key: 1.0 / math.sqrt(value) for key, value in counts.items()}
    normalizer = len(units) / sum(raw[unit.change_type] for unit in units)
    result = {key: raw[key] * normalizer for key in sorted(raw)}
    if abs(sum(result[unit.change_type] for unit in units) / len(units) - 1.0) > 1e-12:
        raise RuntimeError("V96 family weights do not have unit mean one")
    return result


def invariant_subset_v96(rows: Sequence[RowV96], *, seed: int = 960096) -> tuple[PairUnitV96, ...]:
    """Choose 96 stable units without consulting answer text or class."""

    _changed, invariant = pair_units_v96(rows)
    by_type: dict[str, list[PairUnitV96]] = defaultdict(list)
    for unit in invariant:
        by_type[unit.answer_type].append(unit)
    selected: list[PairUnitV96] = []
    for answer_type, quota in sorted(_ANSWER_TYPE_QUOTAS.items()):
        candidates = sorted(
            by_type[answer_type],
            key=lambda unit: (
                hashlib.sha256(
                    f"{seed}|stable|{answer_type}|{unit.pair_id}|{unit.question_key}".encode()
                ).hexdigest(),
                unit.key,
            ),
        )
        if len(candidates) < quota:
            raise ValueError("V96 invariant answer-type quota is impossible")
        selected.extend(candidates[:quota])
    result = tuple(sorted(selected, key=lambda unit: unit.key))
    if (
        len(result) != 96
        or len({unit.key for unit in result}) != 96
        or Counter(unit.answer_type for unit in result) != Counter(_ANSWER_TYPE_QUOTAS)
        or len({unit.change_type for unit in result}) != 9
    ):
        raise RuntimeError("V96 invariant subset selection changed")
    return result


def training_schedule_v96(
    rows: Sequence[RowV96], *, seed: int = 960096
) -> tuple[TrainingStepV96, ...]:
    changed, _invariant = pair_units_v96(rows)
    stable = invariant_subset_v96(rows, seed=seed)
    steps: list[TrainingStepV96] = []
    for pass_index in range(2):
        steps.extend(TrainingStepV96("retention", pass_index, row=row) for row in rows)
    for round_index in range(4):
        steps.extend(TrainingStepV96("changed_pair", round_index, unit=unit) for unit in changed)
    steps.extend(TrainingStepV96("invariant_pair", 0, unit=unit) for unit in stable)
    result = tuple(
        sorted(
            steps,
            key=lambda step: (
                hashlib.sha256(
                    (f"{seed}|schedule|" + "|".join(map(str, step.identity()))).encode()
                ).hexdigest(),
                step.identity(),
            ),
        )
    )
    counts = Counter(step.kind for step in result)
    if (
        len(result) != EXPECTED_MICRO_STEPS
        or counts
        != Counter(
            {
                "retention": EXPECTED_RETENTION_STEPS,
                "changed_pair": EXPECTED_CHANGED_PAIR_STEPS,
                "invariant_pair": EXPECTED_INVARIANT_PAIR_STEPS,
            }
        )
        or Counter(step.row.key for step in result if step.row is not None)
        != Counter({row.key: 2 for row in rows})
        or Counter(step.unit.key for step in result if step.kind == "changed_pair")
        != Counter({unit.key: 4 for unit in changed})
    ):
        raise RuntimeError("V96 fixed training schedule changed")
    return result


def load_scene_memories_v96(
    config: Mapping[str, Any], rows: Sequence[RowV96]
) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    memories: dict[str, torch.Tensor] = {}
    hashes: dict[str, str] = {}
    for field in ("train_memory_cache", "development_memory_cache"):
        cache = load_v82_cache(resolve_v85(config["sources"][field]))
        for scene_id, memory in zip(
            cache.metadata["scene_ids"], cache.tensors["scene_memories"], strict=True
        ):
            if scene_id in memories:
                raise ValueError("V96 memory caches overlap")
            fixed = memory.unsqueeze(0).detach().cpu().contiguous()
            if tuple(fixed.shape) != (1, 738, 1536) or fixed.dtype != torch.bfloat16:
                raise ValueError("V96 immutable scene memory shape or dtype changed")
            memories[scene_id] = fixed
            hashes[scene_id] = prefix_sha256(fixed)
    requested = tuple(sorted({row.scene_id for row in rows}))
    if tuple(sorted(memories)) != requested or requested != TRAINING_SCENES:
        raise ValueError("V96 must bind exactly forty training memories")
    return memories, hashes


class _SyntheticAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(1536, 4096, bias=False, dtype=torch.bfloat16)


class _SyntheticLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = _SyntheticAttention()


class _SyntheticGemma(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = nn.Module()
        layers: list[nn.Module] = [nn.Identity() for _ in range(35)]
        layers[9] = _SyntheticLayer()
        self.model.language_model.layers = nn.ModuleList(layers)


def lora_preflight_v96(config: Mapping[str, Any]) -> dict[str, Any]:
    bridge = config["bridge"]
    synthetic = _SyntheticGemma()
    synthetic.requires_grad_(False)
    installation = install_lora_adapters(
        synthetic,
        LoRASettings(
            enabled=True,
            rank=int(bridge["rank"]),
            alpha=float(bridge["alpha"]),
            dropout=float(bridge["dropout"]),
            target_modules=tuple(bridge["target_modules"]),
        ),
    )
    if installation is None:
        raise RuntimeError("V96 synthetic LoRA installation failed")
    initialize_lora_adapter_state(installation, seed=int(bridge["initialization_seed"]))
    observed = installation.state_sha256()
    expected = bridge["expected_initial_state_sha256"]
    if (
        installation.parameter_count != FRESH_PARAMETER_COUNT
        or (expected != "TO_FILL" and expected != observed)
        or any(torch.count_nonzero(adapter.lora_b).item() for adapter in installation.adapters)
    ):
        raise RuntimeError("V96 deterministic zero-output LoRA initialization changed")
    return {
        "bank_name": FRESH_BANK_NAME,
        "target_modules": list(installation.target_names),
        "parameter_count": installation.parameter_count,
        "initial_state_sha256": observed,
        "adapter_shapes": [
            {
                "lora_a": list(adapter.lora_a.shape),
                "lora_b": list(adapter.lora_b.shape),
            }
            for adapter in installation.adapters
        ],
        "exact_zero_output_at_initialization": True,
        "full_gemma_model_loaded": False,
    }


def authenticate_pinned_model_tensors_v96(config: Mapping[str, Any]) -> dict[str, Any]:
    sources = config["sources"]
    snapshot = (
        Path.home()
        / ".cache/huggingface/hub/models--google--gemma-4-E2B-it/snapshots"
        / str(sources["model_revision"])
    )
    model = snapshot / "model.safetensors"
    if model.is_symlink():
        model = model.resolve()
    if not model.is_file() or sha256_file_v85(model) != sources["model_blob_sha256_identity"]:
        raise ValueError("V96 pinned local model blob changed")
    shapes: dict[str, list[int]] = {}
    dtypes: dict[str, str] = {}
    with safe_open(model, framework="pt", device="cpu") as archive:
        keys = set(archive.keys())
        for name, expected_shape in PINNED_TENSORS.items():
            if name not in keys:
                raise KeyError(f"V96 pinned tensor missing: {name}")
            tensor = archive.get_slice(name)
            shapes[name] = list(tensor.get_shape())
            dtypes[name] = str(tensor.get_dtype())
            if shapes[name] != expected_shape or dtypes[name] != "BF16":
                raise ValueError(f"V96 pinned tensor changed: {name}")
    return {
        "model_blob_sha256": sha256_file_v85(model),
        "tensor_shapes": shapes,
        "tensor_dtypes": dtypes,
        "tensor_values_materialized": False,
        "full_gemma_model_loaded": False,
    }


def _deferred_physical_paths(config: Mapping[str, Any]) -> tuple[Path, ...]:
    return tuple(
        _leaf_path(root) / scene_id
        for root in config["deferred_final_lock"]["physical_artifact_roots"]
        for scene_id in DEFERRED_FINAL_SCENES
    )


def assert_deferred_final_absent_v96(config: Mapping[str, Any]) -> dict[str, Any]:
    physical = _deferred_physical_paths(config)
    present = [str(path) for path in physical if path.exists() or path.is_symlink()]
    if present:
        raise RuntimeError(f"V96 refuses existing deferred-final artifacts: {present}")
    placeholders: dict[str, int] = {}
    for raw in config["deferred_final_lock"]["empty_qa_placeholders"]:
        path = _leaf_path(raw)
        if path.is_symlink() or not path.is_file() or path.stat().st_size != 0:
            raise RuntimeError(f"V96 deferred QA placeholder is not empty: {path}")
        placeholders[Path(raw).as_posix()] = 0
    plans = tuple(
        _leaf_path(raw) for raw in config["deferred_final_lock"]["legacy_plan_files_never_opened"]
    )
    if any(not path.is_file() or path.is_symlink() for path in plans):
        raise FileNotFoundError("V96 legacy deferred plan absence-lock changed")
    return {
        "scene_ids": list(DEFERRED_FINAL_SCENES),
        "physical_path_count_checked": len(physical),
        "physical_artifacts_present": [],
        "empty_qa_placeholders": placeholders,
        "legacy_plan_file_count_opened": 0,
        "generation_performed": False,
    }


def assert_initial_outputs_absent_v96(config: Mapping[str, Any]) -> dict[str, Any]:
    """Reject files, directories, valid symlinks, and broken symlinks."""

    outputs = config["outputs"]
    checked = {
        key: _leaf_path(outputs[key])
        for key in ("work_root", "fixed_final_candidate", "training_report")
    }
    present = {
        key: path.as_posix() for key, path in checked.items() if path.exists() or path.is_symlink()
    }
    if present:
        raise FileExistsError(f"V96 initial output already exists: {present}")
    return {
        "checked_paths": {key: path.as_posix() for key, path in checked.items()},
        "work_root_absent": True,
        "fixed_final_candidate_absent": True,
        "training_report_absent": True,
    }


def forbidden_training_roots_v96(config: Mapping[str, Any]) -> list[Path]:
    roots = list(_deferred_physical_paths(config))
    roots.extend(
        _leaf_path(root) / scene_id
        for root in config["deferred_final_lock"]["physical_artifact_roots"]
        for scene_id in PRIOR_EVALUATION_SCENES
    )
    roots.extend(
        _leaf_path(path)
        for path in (
            config["excluded_known_development"]["labels_path"],
            config["excluded_known_development"]["questions_path"],
            *config["deferred_final_lock"]["empty_qa_placeholders"],
            *config["deferred_final_lock"]["legacy_plan_files_never_opened"],
            "reports/gemma4/predictions",
            "data/oracle",
        )
    )
    return list(dict.fromkeys(path.absolute() for path in roots))


def _reject_row_content(value: object) -> None:
    forbidden = {
        "question",
        "questions",
        "question_text",
        "answer",
        "answers",
        "answer_text",
        "reference",
        "reference_answer",
        "rows",
        "predictions",
    }
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).casefold() in forbidden:
                raise ValueError("V96 parent aggregate serializes row-level QA content")
            _reject_row_content(child)
    elif isinstance(value, list):
        for child in value:
            _reject_row_content(child)


def validate_v95_structured_parent_v96(evidence: Mapping[str, Any]) -> None:
    metrics = evidence.get("metrics")
    if not isinstance(metrics, Mapping):
        raise TypeError("V96 V95 parent metrics are absent")
    arms = metrics.get("arms")
    counterfactual = metrics.get("counterfactual")
    if not isinstance(arms, Mapping) or not isinstance(counterfactual, Mapping):
        raise TypeError("V96 V95 parent structured metrics changed")
    primary = arms.get("primary")
    if (
        evidence.get("artifact") != "gemma4_v95_known_development_structured_score_v1"
        or evidence.get("status") != "measured_aggregate_only_not_yet_gated"
        or evidence.get("row_count") != 216
        or evidence.get("scene_count") != 6
        or evidence.get("row_level_content_serialized") is not False
        or evidence.get("labels_opened_only_by_separate_scorer") is not True
        or evidence.get("scorer_loaded_model") is not False
        or evidence.get("runtime_promotion_authorized") is not False
        or not isinstance(primary, Mapping)
        or primary.get("correct") != 167
        or primary.get("total") != 216
        or counterfactual.get("canonical_correct_sides") != 13
        or counterfactual.get("canonical_complete_units") != 1
        or counterfactual.get("canonical_prediction_changed_units") != 2
        or counterfactual.get("side_count") != 24
        or counterfactual.get("unit_count") != 12
    ):
        raise ValueError("V96 rejected V95's aggregate structured parent evidence")
    _reject_row_content(evidence)


def authenticate_parent_v95_v96(config: Mapping[str, Any]) -> dict[str, Any]:
    """Authenticate the exact unpromoted V95 fixed-final and row-free result."""

    from semantic_3d_chat.training.train_v95_strict_causal_successor import (
        authenticate_training_report_v95,
    )

    sources = config["sources"]
    v95_config_path = _leaf_path(sources["frozen_v95_config"])
    v95_config_sha256 = sha256_file_v85(v95_config_path)
    if v95_config_sha256 != sources["frozen_v95_config_sha256"]:
        raise ValueError("V96 frozen V95 config changed")
    v95_config = load_config_v95(v95_config_path, allow_draft=False)
    v95_source_auth = authenticate_training_sources_v95(v95_config)
    training_auth = authenticate_training_report_v95(v95_config, config_path=v95_config_path)
    report = _leaf_path(sources["v95_training_report"])
    if (
        sha256_file_v85(report) != sources["v95_training_report_sha256"]
        or training_auth["training_report_sha256"] != sources["v95_training_report_sha256"]
    ):
        raise ValueError("V96 V95 training report changed")
    training_report = _strict_json(report)
    report_source_hashes = training_report.get("source_hashes")
    if not isinstance(report_source_hashes, Mapping):
        raise TypeError("V96 V95 training source hashes are absent")
    for source_key in ("preflight_source", "trainer_source"):
        raw_source = str(v95_config["sources"][source_key])
        expected_source_hash = str(v95_config["sources"][source_key + "_sha256"])
        actual_source_hash = sha256_file_v85(_leaf_path(raw_source))
        if (
            actual_source_hash != expected_source_hash
            or v95_source_auth.get(raw_source) != expected_source_hash
            or report_source_hashes.get(raw_source) != expected_source_hash
        ):
            raise ValueError(f"V96 current V95 {source_key} chain changed")

    root = _leaf_path(sources["frozen_v95_fixed_final"])
    if root.is_symlink() or not root.is_dir():
        raise ValueError("V96 V95 fixed-final parent is absent or linked")
    if {child.name for child in root.iterdir()} != {
        "bridge.safetensors",
        "runtime_metadata.json",
    }:
        raise ValueError("V96 V95 parent file inventory changed")
    weights = root / "bridge.safetensors"
    metadata_path = root / "runtime_metadata.json"
    metadata = _strict_json(metadata_path)
    candidate = authenticate_fixed_final_candidate_v95(
        v95_config,
        config_path=v95_config_path,
    )
    candidate_report = training_report.get("candidate")
    trainable_bridge = training_report.get("trainable_bridge")
    metadata_bindings = metadata.get("bindings")
    if (
        sha256_file_v85(weights) != sources["frozen_v95_bridge_sha256"]
        or sha256_file_v85(metadata_path) != sources["frozen_v95_bridge_metadata_sha256"]
        or metadata.get("artifact") != "gemma4_v95_strict_causal_successor_fixed_final_v1"
        or metadata.get("status") != "fixed_final_awaiting_known_development_gate"
        or metadata.get("bank_name") != "v95_strict_causal_successor_bridge"
        or metadata.get("state_sha256") != config["frozen_stack"]["v95_bank_state_sha256"]
        or metadata.get("parameter_count") != 143_360
        or metadata.get("runtime_promotion_authorized") is not False
        or metadata.get("deferred_final_generated") is not False
        or not isinstance(metadata_bindings, Mapping)
        or metadata_bindings.get("config_sha256") != v95_config_sha256
        or metadata_bindings.get("trainer_source_sha256")
        != v95_config["sources"]["trainer_source_sha256"]
        or not isinstance(candidate_report, Mapping)
        or not isinstance(trainable_bridge, Mapping)
        or candidate["weights_sha256"] != metadata.get("weights_sha256")
        or candidate["weights_sha256"] != candidate_report.get("weights_sha256")
        or candidate["weights_sha256"] != sources["frozen_v95_bridge_sha256"]
        or candidate["metadata_file_sha256"] != sources["frozen_v95_bridge_metadata_sha256"]
        or candidate["metadata_canonical_sha256"]
        != candidate_report.get("metadata_canonical_sha256")
        or candidate["state_sha256"] != metadata.get("state_sha256")
        or candidate["state_sha256"] != trainable_bridge.get("final_state_sha256")
        or candidate["state_sha256"] != config["frozen_stack"]["v95_bank_state_sha256"]
        or candidate["training_report_sha256"] != training_auth["training_report_sha256"]
        or candidate["config_sha256"] != v95_config_sha256
    ):
        raise ValueError("V96 V95 fixed-final weights/metadata/training chain changed")

    structured_path = _leaf_path(sources["v95_known_development_structured"])
    if sha256_file_v85(structured_path) != sources["v95_known_development_structured_sha256"]:
        raise ValueError("V96 V95 structured parent evidence changed")
    structured = _strict_json(structured_path)
    validate_v95_structured_parent_v96(structured)
    final_score_path = _leaf_path(sources["v95_known_development_final_score"])
    evidence_path = _leaf_path(sources["v95_known_development_evidence"])
    if (
        sha256_file_v85(final_score_path) != sources["v95_known_development_final_score_sha256"]
        or sha256_file_v85(evidence_path) != sources["v95_known_development_evidence_sha256"]
    ):
        raise ValueError("V96 V95 final failed-gate evidence changed")
    final_score = _strict_json(final_score_path)
    evidence = _strict_json(evidence_path)
    nll_path = final_score_path.with_name(final_score_path.stem + "_nll.json")
    nll_access_path = final_score_path.with_name(final_score_path.stem + "_nll_access.json")
    nll_completion_path = final_score_path.with_name(final_score_path.stem + "_nll_completion.json")
    nll = _strict_json(nll_path)
    nll_access = _strict_json(nll_access_path)
    nll_completion = _strict_json(nll_completion_path)
    nll_sha256 = sha256_file_v85(nll_path)
    nll_access_sha256 = sha256_file_v85(nll_access_path)
    nll_completion_sha256 = sha256_file_v85(nll_completion_path)
    gate_results = final_score.get("gate_results")
    if (
        final_score.get("artifact") != "gemma4_v95_known_development_gate_v1"
        or final_score.get("status") != "measured_preregistered_gate_not_passed"
        or final_score.get("known_development_gate_passed") is not False
        or final_score.get("deferred_final_unlock_eligible") is not False
        or final_score.get("row_level_content_serialized") is not False
        or final_score.get("runtime_promotion_authorized") is not False
        or final_score.get("protected_read_count") != 0
        or not isinstance(gate_results, Mapping)
        or gate_results.get("v95_correct_minimum") is not True
        or gate_results.get("prediction_changed_units_minimum") is not False
        or evidence.get("artifact") != "gemma4_v95_known_development_evidence_v1"
        or evidence.get("status") != "sealed_aggregate_evidence"
        or evidence.get("known_development_gate_passed") is not False
        or evidence.get("deferred_final_unlock_eligible") is not False
        or evidence.get("row_level_content_serialized") is not False
        or evidence.get("runtime_promotion_authorized") is not False
        or evidence.get("final_score_sha256") != sha256_file_v85(final_score_path)
        or final_score.get("config_sha256") != v95_config_sha256
        or evidence.get("config_sha256") != v95_config_sha256
        or final_score.get("training_report_sha256") != training_auth["training_report_sha256"]
        or evidence.get("training_report_sha256") != training_auth["training_report_sha256"]
        or final_score.get("structured_score_sha256") != sha256_file_v85(structured_path)
        or evidence.get("structured_score_sha256") != sha256_file_v85(structured_path)
        or final_score.get("candidate_state_sha256") != candidate["state_sha256"]
        or structured.get("candidate_fingerprint_sha256") != candidate["fingerprint_sha256"]
        or final_score.get("candidate_fingerprint_sha256") != candidate["fingerprint_sha256"]
        or evidence.get("candidate_fingerprint_sha256") != candidate["fingerprint_sha256"]
        or final_score.get("structured_metrics") != structured.get("metrics")
        or evidence.get("known_development_gate_results_sha256")
        != canonical_sha256_v85(gate_results)
        or nll.get("artifact") != "gemma4_v95_known_development_nll_aggregate_v1"
        or nll.get("status") != "measured_aggregate_only_not_yet_gated"
        or nll.get("schema_version") != 95
        or nll.get("row_level_content_serialized") is not False
        or nll.get("candidate_fingerprint_sha256") != candidate["fingerprint_sha256"]
        or nll.get("metrics") != final_score.get("nll_metrics")
        or nll_access.get("artifact") != "gemma4_v95_file_access_audit_v1"
        or nll_access.get("schema_version") != 95
        or nll_access.get("passed") is not True
        or nll_access.get("protected_read_count") != 0
        or nll_access.get("forbidden_accesses") != []
        or nll_completion.get("artifact") != "gemma4_v95_known_development_nll_completion_v1"
        or nll_completion.get("schema_version") != 95
        or nll_completion.get("candidate_fingerprint_before") != candidate["fingerprint_sha256"]
        or nll_completion.get("candidate_fingerprint_after") != candidate["fingerprint_sha256"]
        or nll_completion.get("candidate_immutable") is not True
        or nll_completion.get("nll_sha256") != nll_sha256
        or nll_completion.get("nll_access_sha256") != nll_access_sha256
        or final_score.get("nll_sha256") != nll_sha256
        or evidence.get("nll_sha256") != nll_sha256
        or final_score.get("nll_access_sha256") != nll_access_sha256
        or evidence.get("nll_access_sha256") != nll_access_sha256
        or final_score.get("nll_completion_sha256") != nll_completion_sha256
        or evidence.get("nll_completion_sha256") != nll_completion_sha256
    ):
        raise ValueError("V96 rejected V95's sealed failed-gate evidence")
    _reject_row_content(final_score)
    _reject_row_content(evidence)
    _reject_row_content(nll)
    _reject_row_content(nll_completion)
    return {
        "status": "authenticated_fixed_final_not_promoted",
        "frozen_bank_count": EXPECTED_FROZEN_BANK_COUNT,
        "frozen_parameter_count": EXPECTED_FROZEN_PARAMETER_COUNT,
        "v95_state_sha256": metadata["state_sha256"],
        "v95_candidate_fingerprint_sha256": candidate["fingerprint_sha256"],
        "v95_training_report_sha256": training_auth["training_report_sha256"],
        "v95_structured_evidence_sha256": sha256_file_v85(structured_path),
        "v95_final_score_sha256": sha256_file_v85(final_score_path),
        "v95_evidence_sha256": sha256_file_v85(evidence_path),
        "v95_nll_sha256": nll_sha256,
        "v95_nll_access_sha256": nll_access_sha256,
        "v95_nll_completion_sha256": nll_completion_sha256,
        "v95_preflight_source_sha256": v95_config["sources"]["preflight_source_sha256"],
        "v95_trainer_source_sha256": v95_config["sources"]["trainer_source_sha256"],
        "v95_known_development_gate_passed": False,
        "v95_primary_correct": 167,
        "v95_changed_sides": 13,
        "v95_complete_units": 1,
        "v95_prediction_changed_units": 2,
        "v95_invariant_false_changes": 20,
        "runtime_release_loaded": False,
        "row_level_parent_content_loaded": False,
    }


def authenticate_training_sources_v96(config: Mapping[str, Any]) -> dict[str, str]:
    sources = config["sources"]
    bindings = (
        (sources["runtime_config"], sources["runtime_config_sha256"]),
        (sources["training_qa"], sources["training_qa_sha256"]),
        (
            str(Path(sources["train_memory_cache"]) / "training_tensors.safetensors"),
            sources["train_memory_tensor_sha256"],
        ),
        (
            str(Path(sources["train_memory_cache"]) / "metadata.json"),
            sources["train_memory_metadata_sha256"],
        ),
        (
            str(Path(sources["development_memory_cache"]) / "training_tensors.safetensors"),
            sources["development_memory_tensor_sha256"],
        ),
        (
            str(Path(sources["development_memory_cache"]) / "metadata.json"),
            sources["development_memory_metadata_sha256"],
        ),
        (sources["preflight_source"], sources["preflight_source_sha256"]),
        (sources["trainer_source"], sources["trainer_source_sha256"]),
    )
    observed: dict[str, str] = {}
    for raw, expected in bindings:
        _require_hash(expected, str(raw), draft=False)
        value = sha256_file_v85(raw)
        if value != expected:
            raise ValueError(f"V96 pinned training source changed: {raw}")
        observed[str(raw)] = value
    authenticate_parent_v95_v96(config)
    return observed


def derive_contract_v96(config_path: str | Path = CONFIG) -> dict[str, Any]:
    config = load_config_v96(config_path, allow_draft=True)
    audit = FileAccessAudit(
        forbidden_training_roots_v96(config),
        forbidden_component_names=frozenset(),
        block_forbidden=True,
    )
    with audit:
        absence = assert_deferred_final_absent_v96(config)
        initial_outputs = assert_initial_outputs_absent_v96(config)
        rows = load_training_rows_v96(config)
        changed, _invariant = pair_units_v96(rows)
        stable = invariant_subset_v96(rows)
        class_weights = balanced_class_weights_v96(config, rows)
        changed_weights = family_weights_v96(changed)
        invariant_weights = family_weights_v96(stable)
        schedule = training_schedule_v96(rows)
        memories, memory_hashes = load_scene_memories_v96(config, rows)
        lora = lora_preflight_v96(config)
        pinned = authenticate_pinned_model_tensors_v96(config)
        parent = authenticate_parent_v95_v96(config)
    audit.assert_clean()
    schedule_sha = canonical_sha256_v85([step.identity() for step in schedule])
    subset_sha = canonical_sha256_v85([[unit.pair_id, unit.question_key] for unit in stable])
    inventories = {
        "row_inventory_sha256": canonical_sha256_v85(_row_inventory(rows)),
        "raw_answer_inventory_sha256": canonical_sha256_v85(
            sorted(Counter(row.answer for row in rows).items())
        ),
        "scene_inventory_sha256": canonical_sha256_v85(sorted(memories)),
        "pair_inventory_sha256": canonical_sha256_v85(sorted({row.pair_id for row in rows})),
        "answer_class_inventory_sha256": canonical_sha256_v85(
            sorted(Counter(row.answer_class for row in rows).items())
        ),
        "balanced_class_weight_inventory_sha256": canonical_sha256_v85(
            sorted(class_weights.items())
        ),
        "changed_family_weight_inventory_sha256": canonical_sha256_v85(
            sorted(changed_weights.items())
        ),
        "invariant_family_weight_inventory_sha256": canonical_sha256_v85(
            sorted(invariant_weights.items())
        ),
    }
    for key, observed in inventories.items():
        expected = config["training_pool"][key]
        if expected != "TO_FILL" and expected != observed:
            raise ValueError(f"V96 derived {key} changed")
    for key, observed in (
        ("schedule_sha256", schedule_sha),
        ("invariant_subset_sha256", subset_sha),
    ):
        expected = config["training"][key]
        if expected != "TO_FILL" and expected != observed:
            raise ValueError(f"V96 derived {key} changed")
    counts = Counter(step.kind for step in schedule)
    return {
        "schema_version": 96,
        "status": "derived_not_training_authorized",
        "config_status": config["status"],
        "dataset_hashes": inventories,
        "schedule_sha256": schedule_sha,
        "invariant_subset_sha256": subset_sha,
        "training_rows": len(rows),
        "training_scenes": len(memories),
        "training_pairs": len({row.pair_id for row in rows}),
        "changed_units": len(changed),
        "changed_sides": 2 * len(changed),
        "all_changed_pair_questions_byte_identical": all(
            unit.left.question.encode("utf-8") == unit.right.question.encode("utf-8")
            for unit in changed
        ),
        "invariant_subset_units": len(stable),
        "invariant_subset_sides": 2 * len(stable),
        "all_invariant_subset_questions_byte_identical": all(
            unit.left.question.encode("utf-8") == unit.right.question.encode("utf-8")
            for unit in stable
        ),
        "schedule_step_counts": dict(sorted(counts.items())),
        "schedule_micro_steps": len(schedule),
        "optimizer_updates": len(schedule) // 8,
        "total_nll_forward_evaluations": EXPECTED_TOTAL_NLL_FORWARDS,
        "auxiliary_nll_forward_evaluations": 888,
        "estimated_wall_time_seconds": 7182,
        "wall_time_budget_seconds": 9000,
        "all_training_memory_hashes": memory_hashes,
        "frozen_parent": parent,
        "lora_preflight": lora,
        "pinned_model_tensors": pinned,
        "initial_output_absence": initial_outputs,
        "deferred_final_absence": absence,
        "known_development_scene_ids_loaded": [],
        "known_development_labels_opened": False,
        "known_development_questions_opened": False,
        "deferred_final_scene_ids_loaded": [],
        "deferred_final_artifacts_generated": False,
        "file_audit_forbidden_reads": audit.forbidden_accesses(),
        "full_gemma_model_loaded": False,
        "optimizer_constructed": False,
        "optimizer_updates_performed": 0,
        "training_authorized": False,
    }


def derive_preregistration_v96(
    config_path: str | Path = CONFIG,
) -> dict[str, Any]:
    config = load_config_v96(config_path, allow_draft=True)
    derived = derive_contract_v96(config_path)
    return {
        "artifact": PREREG_ARTIFACT,
        "schema_version": 96,
        "status": "draft_not_sealed_training_implementation_pending",
        "config_path": Path(config_path).as_posix(),
        "config_sha256": sha256_file_v85(config_path),
        "derived_contract": derived,
        "strict_input_contract": config["strict_input_contract"],
        "training_pool": config["training_pool"],
        "excluded_known_development": config["excluded_known_development"],
        "deferred_final_lock": config["deferred_final_lock"],
        "frozen_stack": config["frozen_stack"],
        "bridge": config["bridge"],
        "training_protocol": config["training"],
        "known_development_protocol": config["known_development_gate"],
        "initial_output_absence": derived["initial_output_absence"],
        "parent_authenticated": derived["frozen_parent"]["status"]
        == "authenticated_fixed_final_not_promoted",
        "known_development_labels_opened": False,
        "known_development_questions_opened": False,
        "deferred_final_artifacts_generated": False,
        "full_gemma_model_loaded": False,
        "optimizer_constructed": False,
        "optimizer_updates": 0,
        "training_authorized": False,
    }


def _atomic_create_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    destination = _leaf_path(path)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"V96 create-once output exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def build_preregistration_v96(config_path: str | Path = CONFIG) -> dict[str, Any]:
    config = load_config_v96(config_path, allow_draft=False)
    assert_initial_outputs_absent_v96(config)
    sources = authenticate_training_sources_v96(config)
    draft = derive_preregistration_v96(config_path)
    if draft["parent_authenticated"] is not True:
        raise RuntimeError("V96 cannot seal before V95 is authenticated")
    payload = {
        **draft,
        "status": "sealed_before_v96_full_model_load_and_deferred_generation",
        "authenticated_sources": sources,
        "training_authorized": True,
    }
    output = _atomic_create_json(config["outputs"]["preregistration"], payload)
    return {**payload, "output": output.as_posix()}


def authenticate_preregistration_v96(
    config: Mapping[str, Any], *, config_path: str | Path = CONFIG
) -> dict[str, str]:
    path = _leaf_path(config["outputs"]["preregistration"])
    payload = _strict_json(path)
    config_hash = sha256_file_v85(config_path)
    absence = payload.get("initial_output_absence")
    if (
        payload.get("artifact") != PREREG_ARTIFACT
        or payload.get("schema_version") != 96
        or payload.get("status") != "sealed_before_v96_full_model_load_and_deferred_generation"
        or payload.get("config_sha256") != config_hash
        or payload.get("parent_authenticated") is not True
        or payload.get("known_development_labels_opened") is not False
        or payload.get("known_development_questions_opened") is not False
        or payload.get("deferred_final_artifacts_generated") is not False
        or payload.get("full_gemma_model_loaded") is not False
        or payload.get("optimizer_constructed") is not False
        or payload.get("optimizer_updates") != 0
        or payload.get("training_authorized") is not True
        or not isinstance(absence, Mapping)
        or any(
            absence.get(key) is not True
            for key in (
                "work_root_absent",
                "fixed_final_candidate_absent",
                "training_report_absent",
            )
        )
    ):
        raise ValueError("V96 preregistration changed")
    return {
        "config_sha256": config_hash,
        "preregistration_sha256": sha256_file_v85(path),
    }


def run_cpu_preflight_v96(config_path: str | Path = CONFIG) -> dict[str, Any]:
    config = load_config_v96(config_path, allow_draft=False)
    prereg = authenticate_preregistration_v96(config, config_path=config_path)
    absence = assert_initial_outputs_absent_v96(config)
    report = {
        "artifact": PREFLIGHT_ARTIFACT,
        "schema_version": 96,
        "status": "passed",
        "passed": True,
        **prereg,
        "authenticated_sources": authenticate_training_sources_v96(config),
        "derived_contract": derive_contract_v96(config_path),
        "initial_output_absence": absence,
        "parent_authenticated": True,
        "known_development_labels_opened": False,
        "known_development_questions_opened": False,
        "deferred_final_artifacts_generated": False,
        "full_gemma_model_loaded": False,
        "optimizer_constructed": False,
        "optimizer_updates": 0,
        "behavior_scored": False,
        "oracle_loaded": False,
        "runtime_promotion_authorized": False,
    }
    output = _atomic_create_json(config["outputs"]["cpu_preflight"], report)
    return {**report, "output": output.as_posix()}


def authenticate_cpu_preflight_v96(
    config: Mapping[str, Any], *, config_path: str | Path = CONFIG
) -> dict[str, str]:
    prereg = authenticate_preregistration_v96(config, config_path=config_path)
    path = _leaf_path(config["outputs"]["cpu_preflight"])
    payload = _strict_json(path)
    absence = payload.get("initial_output_absence")
    if (
        payload.get("artifact") != PREFLIGHT_ARTIFACT
        or payload.get("schema_version") != 96
        or payload.get("status") != "passed"
        or payload.get("passed") is not True
        or payload.get("config_sha256") != prereg["config_sha256"]
        or payload.get("preregistration_sha256") != prereg["preregistration_sha256"]
        or payload.get("parent_authenticated") is not True
        or payload.get("known_development_labels_opened") is not False
        or payload.get("known_development_questions_opened") is not False
        or payload.get("deferred_final_artifacts_generated") is not False
        or payload.get("full_gemma_model_loaded") is not False
        or payload.get("optimizer_constructed") is not False
        or payload.get("optimizer_updates") != 0
        or payload.get("behavior_scored") is not False
        or payload.get("oracle_loaded") is not False
        or payload.get("runtime_promotion_authorized") is not False
        or not isinstance(absence, Mapping)
        or any(
            absence.get(key) is not True
            for key in (
                "work_root_absent",
                "fixed_final_candidate_absent",
                "training_report_absent",
            )
        )
    ):
        raise ValueError("V96 CPU preflight changed")
    return {**prereg, "cpu_preflight_sha256": sha256_file_v85(path)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "authenticate-parent",
            "derive",
            "derive-preregistration",
            "preregister",
            "cpu-preflight",
            "authenticate",
        ),
    )
    parser.add_argument("--config", default=str(CONFIG))
    args = parser.parse_args(argv)
    if args.command == "authenticate-parent":
        config = load_config_v96(args.config, allow_draft=True)
        result = authenticate_parent_v95_v96(config)
    elif args.command == "derive":
        result = derive_contract_v96(args.config)
    elif args.command == "derive-preregistration":
        result = derive_preregistration_v96(args.config)
    elif args.command == "preregister":
        result = build_preregistration_v96(args.config)
    elif args.command == "cpu-preflight":
        result = run_cpu_preflight_v96(args.config)
    else:
        config = load_config_v96(args.config, allow_draft=False)
        result = authenticate_cpu_preflight_v96(config, config_path=args.config)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CONFIG",
    "EXPECTED_CHANGED_PAIR_STEPS",
    "EXPECTED_FROZEN_BANK_COUNT",
    "EXPECTED_FROZEN_PARAMETER_COUNT",
    "EXPECTED_INITIAL_STATE_SHA256",
    "EXPECTED_INVARIANT_PAIR_STEPS",
    "EXPECTED_MICRO_STEPS",
    "EXPECTED_OPTIMIZER_UPDATES",
    "EXPECTED_RETENTION_STEPS",
    "EXPECTED_TOTAL_ADAPTER_PARAMETER_COUNT",
    "EXPECTED_TOTAL_NLL_FORWARDS",
    "FRESH_BANK_NAME",
    "FRESH_PARAMETER_COUNT",
    "TARGET_MODULES",
    "PairUnitV96",
    "RowV96",
    "TrainingStepV96",
    "answer_class_id_v96",
    "assert_deferred_final_absent_v96",
    "assert_initial_outputs_absent_v96",
    "authenticate_cpu_preflight_v96",
    "authenticate_parent_v95_v96",
    "authenticate_pinned_model_tensors_v96",
    "authenticate_preregistration_v96",
    "authenticate_training_sources_v96",
    "balanced_class_weights_v96",
    "build_preregistration_v96",
    "derive_contract_v96",
    "derive_preregistration_v96",
    "family_weights_v96",
    "forbidden_training_roots_v96",
    "invariant_subset_v96",
    "load_config_v96",
    "load_scene_memories_v96",
    "load_training_rows_v96",
    "lora_preflight_v96",
    "pair_units_v96",
    "training_schedule_v96",
    "validate_v95_structured_parent_v96",
]
