"""Model-free V95 contract and deferred-final absence seal.

V95 continues from V94's exact fixed-final *failed* optimization candidate,
never from a promoted runtime release.  Its only optimization data are the
existing forty training scenes.  Scenes 57--62 are a post-fixed-final
development gate, while opaque scenes 25--30 remain physically absent until
that gate passes and an explicit unlock occurs.  The ordinary preflight opens
only a create-once aggregate V94 evidence seal; it never opens V94 predictions,
scores, or labels.  This module never creates a scene, constructs an optimizer,
or loads the full Gemma model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import torch
import yaml
from safetensors import safe_open
from safetensors.torch import load_file
from torch import nn

from semantic_3d_chat.chat.file_audit import FileAccessAudit
from semantic_3d_chat.chat.v94_strict_multiscene_runtime import (
    EXPECTED_ADAPTER_PARAMETER_COUNT as V94_ADAPTER_PARAMETER_COUNT,
)
from semantic_3d_chat.chat.v94_strict_multiscene_runtime import (
    EXPECTED_BANKS as V94_BANKS,
)
from semantic_3d_chat.chat.v94_strict_multiscene_runtime import (
    V94_STATE_SHA256,
)
from semantic_3d_chat.evaluation.v85_strict_multiscene_preflight import (
    canonical_sha256_v85,
    resolve_v85,
    sha256_file_v85,
)
from semantic_3d_chat.evaluation.v94_strict_multiscene_evidence import (
    ARTIFACT as V94_EVIDENCE_ARTIFACT,
)
from semantic_3d_chat.evaluation.v94_strict_multiscene_evidence import (
    authenticate_v94_evidence,
)
from semantic_3d_chat.language.lora import (
    LoRASettings,
    initialize_lora_adapter_state,
    install_lora_adapters,
    tensor_state_sha256,
)
from semantic_3d_chat.language.prefix_injection import prefix_sha256
from semantic_3d_chat.training.train_question_control_v73 import (
    RowV73,
    changed_units_v73,
    load_training_rows_v73,
)
from semantic_3d_chat.training.v82_reader_artifacts import load_v82_cache

CONFIG: Final[Path] = Path("configs/experiments/gemma4_v95_strict_causal_successor.yaml")
FRESH_BANK_NAME: Final[str] = "v95_strict_causal_successor_bridge"
TARGET_MODULES: Final[tuple[str, ...]] = (
    "model.language_model.layers.9.self_attn.k_proj",
    "model.language_model.layers.9.self_attn.v_proj",
    "model.language_model.layers.34.mlp.up_proj",
)
PINNED_TENSORS: Final[dict[str, list[int]]] = {
    TARGET_MODULES[0] + ".weight": [512, 1536],
    TARGET_MODULES[1] + ".weight": [512, 1536],
    TARGET_MODULES[2] + ".weight": [12288, 1536],
}
FRESH_PARAMETER_COUNT: Final[int] = 143_360
EXPECTED_INITIAL_STATE_SHA256: Final[str] = (
    "ba33a6deb0fab6e8e2f8ef1e8b61636b58110d233dd9084a95fe9af1a4a1a39a"
)
EXPECTED_TOTAL_ADAPTER_PARAMETER_COUNT: Final[int] = 819_200
TRAINING_SCENES: Final[tuple[str, ...]] = tuple(
    [f"scene_{index:06d}" for index in range(11, 25)]
    + [f"scene_{index:06d}" for index in range(31, 57)]
)
PRIOR_EVALUATION_SCENES: Final[tuple[str, ...]] = tuple(
    f"scene_{index:06d}" for index in range(57, 63)
)
DEFERRED_FINAL_SCENES: Final[tuple[str, ...]] = tuple(
    f"scene_{index:06d}" for index in range(25, 31)
)
PREREG_ARTIFACT: Final[str] = "gemma4_v95_strict_causal_successor_preregistration_v1"
PREFLIGHT_ARTIFACT: Final[str] = "gemma4_v95_strict_causal_successor_cpu_preflight_v1"
_DRAFT: Final[str] = "draft_contract_unsealed_training_not_authorized"
_SEALED: Final[str] = "sealed_before_v95_full_model_load"
_HEX64: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{64}")
V94_SCORE_SHA256: Final[str] = "af6433f98c5a7cbdb3c2686fc09d060f164f2b851d7da074736705d86dbab188"
V94_EVIDENCE_BUNDLE_SHA256: Final[str] = (
    "459fe554890c7b44b4f93672e8b153f827ba39c98b22e93d78ccb0ecfd25782a"
)


def _require(value: object, expected: object, label: str) -> None:
    if value != expected:
        raise ValueError(f"V95 {label} changed")


def _require_hash(value: object, label: str, *, draft: bool) -> None:
    if isinstance(value, str) and _HEX64.fullmatch(value):
        return
    if draft and value == "TO_FILL":
        return
    raise ValueError(f"V95 {label} is not sealed")


def _strict_json(path: str | Path) -> dict[str, Any]:
    source = resolve_v85(path)
    if source.is_symlink() or not source.is_file():
        raise FileNotFoundError(source)
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"V95 JSON must contain one object: {source}")
    return value


def load_config_v95(path: str | Path = CONFIG, *, allow_draft: bool = True) -> dict[str, Any]:
    source = resolve_v85(path)
    if source.is_symlink() or not source.is_file():
        raise FileNotFoundError(source)
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or set(payload) != {"v95"}:
        raise ValueError("V95 config must contain exactly one v95 mapping")
    config = payload["v95"]
    if not isinstance(config, Mapping):
        raise TypeError("V95 config payload must be a mapping")
    _require(config.get("schema_version"), 95, "schema version")
    _require(
        config.get("artifact"),
        "gemma4_v95_strict_causal_successor_direct_memory_lora_v1",
        "artifact",
    )
    status = config.get("status")
    if status not in ({_DRAFT, _SEALED} if allow_draft else {_SEALED}):
        raise ValueError("V95 config status is not authorized")
    _require(config.get("seed"), 950095, "seed")

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
        raise TypeError("V95 training pool must be a mapping")
    for key, expected in {
        "scene_count": 40,
        "pair_count": 20,
        "row_count": 960,
        "changed_unit_count": 66,
        "changed_side_count": 132,
        "answer_class_count": 29,
        "all_rows_used_once_per_epoch": True,
        "runtime_serializes_questions_or_answers": False,
    }.items():
        _require(pool.get(key), expected, f"training pool {key}")
    for field in (
        "row_inventory_sha256",
        "scene_inventory_sha256",
        "pair_inventory_sha256",
        "answer_class_inventory_sha256",
        "balanced_class_weight_inventory_sha256",
    ):
        _require_hash(pool.get(field), field, draft=False)

    prior = config.get("excluded_prior_evaluation")
    if not isinstance(prior, Mapping):
        raise TypeError("V95 prior-evaluation exclusion must be a mapping")
    _require(tuple(prior.get("scene_ids", ())), PRIOR_EVALUATION_SCENES, "prior scenes")
    if any(
        prior.get(field) is not False
        for field in (
            "used_for_v95_optimization",
            "used_for_v95_checkpoint_selection",
            "opened_by_v95_preflight",
            "opened_by_v95_training",
        )
    ):
        raise ValueError("V95 prior evaluation isolation changed")

    deferred = config.get("deferred_final_lock")
    if not isinstance(deferred, Mapping):
        raise TypeError("V95 deferred-final lock must be a mapping")
    _require(tuple(deferred.get("scene_ids", ())), DEFERRED_FINAL_SCENES, "deferred IDs")
    _require(
        tuple(deferred.get("physical_artifact_roots", ())),
        ("data/oracle", "data/rendered", "data_gemma4/features", "data_gemma4/maps"),
        "deferred physical roots",
    )
    if any(
        deferred.get(field) is not False
        for field in (
            "generation_before_fixed_final_authorized",
            "rendering_before_fixed_final_authorized",
            "feature_extraction_before_fixed_final_authorized",
            "map_building_before_fixed_final_authorized",
            "qa_generation_before_fixed_final_authorized",
        )
    ) or any(
        deferred.get(field) is not True
        for field in (
            "physical_artifacts_required_absent_through_fixed_final",
            "qa_placeholders_required_zero_bytes_through_fixed_final",
            "only_opaque_ids_and_absence_locks_available_to_v95",
        )
    ):
        raise ValueError("V95 deferred-final boundary changed")

    frozen = config.get("frozen_stack")
    if not isinstance(frozen, Mapping):
        raise TypeError("V95 frozen stack must be a mapping")
    for key, expected in {
        "exact_parent": "fixed_final_nonpromoted_optimization_parent",
        "base_gemma_frozen": True,
        "frozen_bank_count": 8,
        "frozen_adapter_parameter_count": V94_ADAPTER_PARAMETER_COUNT,
        "v94_bank_name": "v94_strict_multiscene_full40_bridge",
        "v94_bank_state_sha256": V94_STATE_SHA256,
        "v94_fixed_final_selected_before_v94_labels_opened": True,
        "v94_behavior_gate_passed_required": False,
        "v94_runtime_release_required_absent": True,
        "merged_weights": False,
    }.items():
        _require(frozen.get(key), expected, f"frozen stack {key}")

    bridge = config.get("bridge")
    if not isinstance(bridge, Mapping):
        raise TypeError("V95 bridge must be a mapping")
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
        "initialization_seed": 950095,
        "expected_initial_state_sha256": EXPECTED_INITIAL_STATE_SHA256,
        "disjoint_from_all_frozen_bank_targets": True,
        "total_bank_count_after_install": 9,
        "total_adapter_parameter_count_after_install": (EXPECTED_TOTAL_ADAPTER_PARAMETER_COUNT),
    }.items():
        _require(bridge.get(key), expected, f"bridge {key}")

    training = config.get("training")
    if not isinstance(training, Mapping):
        raise TypeError("V95 training contract must be a mapping")
    for key, expected in {
        "epochs": 4,
        "rows_per_epoch": 960,
        "total_micro_rows": 3840,
        "gradient_accumulation_rows": 8,
        "optimizer_updates": 480,
        "cross_scene_eligible_row_count": 498,
        "cross_scene_exposures_per_eligible_row": 2,
        "cross_scene_wrong_memory_rows_per_epoch": 249,
        "total_cross_scene_wrong_memory_rows": 996,
        "causal_control_answer_type_count": 7,
        "causal_control_unique_row_count": 498,
        "causal_control_repeated_row_count_per_arm": 2,
        "zero_payload_rows_per_epoch": 125,
        "total_zero_payload_rows": 500,
        "permutation_rows_per_epoch": 125,
        "total_permutation_rows": 500,
        "total_nll_forward_evaluations": 5836,
        "auxiliary_nll_forward_evaluations": 1996,
        "wall_time_budget_seconds": 5400,
        "checkpoint_selection": (
            "fixed_final_update_480_before_known_development_or_deferred_generation"
        ),
        "intermediate_behavior_selection": False,
    }.items():
        _require(training.get(key), expected, f"training {key}")
    for field in (
        "row_order_sha256",
        "cross_scene_schedule_sha256",
        "zero_payload_schedule_sha256",
        "permutation_control_schedule_sha256",
        "payload_permutation_sha256",
    ):
        _require_hash(training.get(field), field, draft=status == _DRAFT)

    development = config.get("known_development_gate")
    if not isinstance(development, Mapping):
        raise TypeError("V95 known-development gate must be a mapping")
    for key, expected in {
        "role": "post_fixed_final_go_no_go_not_checkpoint_selection",
        "scene_ids": list(PRIOR_EVALUATION_SCENES),
        "scene_count": 6,
        "row_count": 216,
        "labels_opened_after_fixed_final_only": True,
        "labels_opened_by_separate_label_isolated_evaluator_only": True,
        "fixed_final_checkpoint_may_not_change_after_gate": True,
        "v94_reference_correct": 143,
        "v94_reference_total": 216,
        "v94_reference_changed_side_correct": 13,
        "v94_reference_complete_changed_units": 2,
        "v94_reference_prediction_changed_units": 4,
        "v95_correct_minimum": 150,
        "v95_accuracy_margin_over_v94_minimum": 0.03,
        "changed_side_correct_minimum": 15,
        "complete_changed_units_minimum": 4,
        "prediction_changed_units_minimum": 7,
        "mean_changed_side_wrong_minus_correct_nll_minimum": 0.2,
        "correct_accuracy_above_zero_payload_required": True,
        "correct_accuracy_above_full_interior_permutation_required": True,
        "correct_nll_below_zero_payload_required": True,
        "correct_nll_below_full_interior_permutation_required": True,
        "pass_required_before_deferred_final_unlock": True,
    }.items():
        _require(development.get(key), expected, f"known development {key}")

    evaluation = config.get("deferred_evaluation")
    if not isinstance(evaluation, Mapping):
        raise TypeError("V95 deferred evaluation must be a mapping")
    for key, expected in {
        "scene_ids": list(DEFERRED_FINAL_SCENES),
        "scene_count": 6,
        "pair_count": 3,
        "expected_row_count_after_unlock": 216,
        "expected_changed_unit_count_after_unlock": 12,
        "expected_changed_side_count_after_unlock": 24,
        "generation_requires_explicit_post_fixed_final_unlock": True,
        "fixed_final_selected_before_generation": True,
        "fixed_final_selected_before_label_creation": True,
        "labels_opened_only_by_separate_final_scorer": True,
        "known_development_gate_must_pass_before_unlock": True,
    }.items():
        _require(evaluation.get(key), expected, f"deferred evaluation {key}")

    sources = config.get("sources")
    if not isinstance(sources, Mapping):
        raise TypeError("V95 sources must be a mapping")
    for field in (
        "v94_hardened_scored_evidence_sha256",
        "preflight_source_sha256",
        "trainer_source_sha256",
    ):
        _require_hash(sources.get(field), field, draft=status == _DRAFT)
    scope = config.get("scope")
    if not isinstance(scope, Mapping) or any(
        scope.get(field) is not False
        for field in (
            "cloud_inference",
            "full_gemma_model_loaded_by_preflight",
            "optimizer_constructed_by_preflight",
            "prior_evaluation_scenes_57_through_62_loaded",
            "deferred_final_scenes_25_through_30_loaded",
            "deferred_final_semantic_plans_loaded",
            "deferred_final_artifacts_generated",
            "oracle_loaded",
            "runtime_promotion_authorized",
        )
    ):
        raise ValueError("V95 protected scope changed")
    return dict(config)


def _row_inventory(rows: Sequence[RowV73]) -> list[list[Any]]:
    return sorted(
        [
            [
                row.scene_id,
                row.question_id,
                row.pair_id,
                row.question_key,
                row.answer_class,
                row.answer_type,
                row.expected_change,
            ]
            for row in rows
        ]
    )


def balanced_class_weights_v95(
    config: Mapping[str, Any], rows: Sequence[RowV73]
) -> dict[str, float]:
    """Return stable inverse-sqrt class weights with row mean exactly one."""

    counts = Counter(row.answer_class for row in rows)
    raw = {key: 1.0 / math.sqrt(value) for key, value in counts.items()}
    normalizer = len(rows) / sum(raw[row.answer_class] for row in rows)
    weights = {key: raw[key] * normalizer for key in sorted(raw)}
    pool = config["training_pool"]
    observed_counts = canonical_sha256_v85(sorted([[key, value] for key, value in counts.items()]))
    observed_weights = canonical_sha256_v85(
        sorted([[key, value] for key, value in weights.items()])
    )
    if (
        len(counts) != 29
        or observed_counts != pool["answer_class_inventory_sha256"]
        or observed_weights != pool["balanced_class_weight_inventory_sha256"]
        or abs(sum(weights[row.answer_class] for row in rows) / len(rows) - 1.0) > 1e-12
    ):
        raise ValueError("V95 balanced CE inventory changed")
    return weights


def load_training_rows_v95(config: Mapping[str, Any]) -> tuple[RowV73, ...]:
    rows = tuple(load_training_rows_v73(config["sources"]["training_qa"]))
    scenes = sorted({row.scene_id for row in rows})
    pairs = sorted({row.pair_id for row in rows})
    units = changed_units_v73(rows)
    forbidden = set(PRIOR_EVALUATION_SCENES) | set(DEFERRED_FINAL_SCENES)
    pool = config["training_pool"]
    hashes = {
        "row_inventory_sha256": canonical_sha256_v85(_row_inventory(rows)),
        "scene_inventory_sha256": canonical_sha256_v85(scenes),
        "pair_inventory_sha256": canonical_sha256_v85(pairs),
    }
    if (
        len(rows) != 960
        or tuple(scenes) != TRAINING_SCENES
        or len(pairs) != 20
        or len(units) != 66
        or sum(row.expected_change for row in rows) != 132
        or forbidden.intersection(scenes)
        or any(pool[key] != value for key, value in hashes.items())
    ):
        raise ValueError("V95 exact training-only row inventory changed")
    balanced_class_weights_v95(config, rows)
    return rows


def training_schedule_v95(
    rows: Sequence[RowV73], *, seed: int = 950095, epochs: int = 4
) -> tuple[tuple[int, RowV73], ...]:
    schedule: list[tuple[int, RowV73]] = []
    for epoch in range(epochs):
        shuffled = sorted(rows, key=lambda row: row.key)
        random.Random(seed + epoch).shuffle(shuffled)
        schedule.extend((epoch, row) for row in shuffled)
    return tuple(schedule)


def cross_scene_wrong_rows_v95(
    rows: Sequence[RowV73],
) -> dict[tuple[str, str], RowV73]:
    """Map all exact-same-question eligible rows to one deterministic negative."""

    candidates = _cross_scene_candidates_v95(rows)
    result = {key: values[0] for key, values in candidates.items()}
    if len(result) != 498:
        raise ValueError("V95 did not causally supervise every eligible train row")
    return result


def _question_identity_v95(row: RowV73) -> str:
    return " ".join(row.question.casefold().split())


def _cross_scene_candidates_v95(
    rows: Sequence[RowV73],
) -> dict[tuple[str, str], tuple[RowV73, ...]]:
    by_question: dict[str, list[RowV73]] = defaultdict(list)
    for row in rows:
        by_question[_question_identity_v95(row)].append(row)
    result: dict[tuple[str, str], tuple[RowV73, ...]] = {}
    for row in rows:
        candidates = tuple(
            sorted(
                (
                    other
                    for other in by_question[_question_identity_v95(row)]
                    if other.scene_id != row.scene_id and other.answer_class != row.answer_class
                ),
                key=lambda other: (other.answer_class, other.scene_id, other.question_id),
            )
        )
        if candidates:
            result[row.key] = candidates
    return result


def cross_scene_schedule_v95(
    rows: Sequence[RowV73], *, seed: int = 950095
) -> tuple[tuple[int, RowV73, RowV73], ...]:
    """Expose each of 498 causal-capable rows exactly twice across four epochs."""

    by_key = {row.key: row for row in rows}
    candidates = _cross_scene_candidates_v95(rows)
    ordered_keys = sorted(
        candidates,
        key=lambda key: (
            hashlib.sha256(f"{seed}|{key[0]}|{key[1]}".encode()).hexdigest(),
            key,
        ),
    )
    if len(ordered_keys) != 498:
        raise ValueError("V95 exact-question causal-capable inventory changed")
    starts = (0, 249, 124, 373)
    schedule: list[tuple[int, RowV73, RowV73]] = []
    for epoch, start in enumerate(starts):
        selected = [ordered_keys[(start + offset) % 498] for offset in range(249)]
        for key in selected:
            row = by_key[key]
            options = candidates[key]
            choice_seed = int(
                hashlib.sha256(
                    f"{seed}|{epoch}|{row.scene_id}|{row.question_id}".encode()
                ).hexdigest()[:16],
                16,
            )
            wrong = options[choice_seed % len(options)]
            schedule.append((epoch, row, wrong))
    exposures = Counter(row.key for _epoch, row, _wrong in schedule)
    if (
        len(schedule) != 996
        or set(exposures) != set(candidates)
        or set(exposures.values()) != {2}
        or any(
            _question_identity_v95(row) != _question_identity_v95(wrong)
            or row.answer_class == wrong.answer_class
            or row.scene_id == wrong.scene_id
            for _epoch, row, wrong in schedule
        )
    ):
        raise RuntimeError("V95 cross-scene causal schedule is incomplete")
    return tuple(schedule)


def causal_control_schedule_v95(
    rows: Sequence[RowV73], *, arm: str, seed: int = 950095
) -> tuple[tuple[int, RowV73], ...]:
    """Rotate 125 rows/epoch across all 498 causal-capable rows.

    Four epochs produce 500 exposures.  Every eligible row is therefore seen
    at least once and exactly two deterministically selected rows are repeated.
    Zero and permutation arms use distinct domain-separated orders.
    """

    if arm not in {"zero_payload", "full_interior_permutation"}:
        raise ValueError(f"V95 unknown causal control arm: {arm}")
    by_key = {row.key: row for row in rows}
    eligible_keys = set(_cross_scene_candidates_v95(rows))
    ordered_keys = sorted(
        eligible_keys,
        key=lambda key: (
            hashlib.sha256(f"{seed}|{arm}|{key[0]}|{key[1]}".encode()).hexdigest(),
            key,
        ),
    )
    if len(ordered_keys) != 498:
        raise ValueError("V95 causal-control eligible inventory changed")
    schedule = tuple(
        (epoch, by_key[ordered_keys[(epoch * 125 + offset) % 498]])
        for epoch in range(4)
        for offset in range(125)
    )
    exposures = Counter(row.key for _epoch, row in schedule)
    repeated = [key for key, count in exposures.items() if count == 2]
    if (
        len(schedule) != 500
        or set(exposures) != eligible_keys
        or set(exposures.values()) != {1, 2}
        or len(repeated) != 2
        or len({row.answer_type for _epoch, row in schedule}) != 7
        or any(sum(epoch == requested for epoch, _row in schedule) != 125 for requested in range(4))
    ):
        raise RuntimeError("V95 causal-control rotation is incomplete")
    return schedule


def load_scene_memories_v95(
    config: Mapping[str, Any], rows: Sequence[RowV73]
) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    memories: dict[str, torch.Tensor] = {}
    hashes: dict[str, str] = {}
    for field in ("train_memory_cache", "development_memory_cache"):
        cache = load_v82_cache(resolve_v85(config["sources"][field]))
        for scene_id, memory in zip(
            cache.metadata["scene_ids"], cache.tensors["scene_memories"], strict=True
        ):
            if scene_id in memories:
                raise ValueError("V95 memory caches overlap")
            fixed = memory.unsqueeze(0).detach().cpu().contiguous()
            if tuple(fixed.shape) != (1, 738, 1536) or fixed.dtype != torch.bfloat16:
                raise ValueError("V95 immutable scene memory shape or dtype changed")
            memories[scene_id] = fixed
            hashes[scene_id] = prefix_sha256(fixed)
    requested = tuple(sorted({row.scene_id for row in rows}))
    if tuple(sorted(memories)) != requested or requested != TRAINING_SCENES:
        raise ValueError("V95 does not bind exactly forty training memories")
    return memories, hashes


def zero_payload_memory_v95(memory: torch.Tensor) -> torch.Tensor:
    if tuple(memory.shape) != (1, 738, 1536) or memory.dtype != torch.bfloat16:
        raise ValueError("V95 zero control requires one BF16 strict memory")
    result = memory.clone()
    result[:, 1:-1].zero_()
    if (
        not torch.equal(result[:, :1], memory[:, :1])
        or not torch.equal(result[:, -1:], memory[:, -1:])
        or torch.count_nonzero(result[:, 1:-1]).item() != 0
    ):
        raise RuntimeError("V95 zero control changed native boundaries")
    return result


def payload_permutation_v95(seed: int = 950095) -> torch.Tensor:
    permutation = torch.randperm(736, generator=torch.Generator().manual_seed(seed))
    if torch.equal(permutation, torch.arange(736)) or len(set(permutation.tolist())) != 736:
        raise RuntimeError("V95 payload permutation is not a bijective shuffle")
    return permutation


def permuted_payload_memory_v95(memory: torch.Tensor, *, seed: int = 950095) -> torch.Tensor:
    if tuple(memory.shape) != (1, 738, 1536) or memory.dtype != torch.bfloat16:
        raise ValueError("V95 permutation control requires one BF16 strict memory")
    result = torch.cat(
        (memory[:, :1], memory[:, 1:-1][:, payload_permutation_v95(seed)], memory[:, -1:]),
        dim=1,
    )
    if (
        not torch.equal(result[:, :1], memory[:, :1])
        or not torch.equal(result[:, -1:], memory[:, -1:])
        or torch.equal(result[:, 1:-1], memory[:, 1:-1])
    ):
        raise RuntimeError("V95 payload permutation changed boundaries or stayed identity")
    return result


class _SyntheticAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.k_proj = nn.Linear(1536, 512, bias=False, dtype=torch.bfloat16)
        self.v_proj = nn.Linear(1536, 512, bias=False, dtype=torch.bfloat16)


class _SyntheticMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.up_proj = nn.Linear(1536, 12288, bias=False, dtype=torch.bfloat16)


class _SyntheticLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = _SyntheticAttention()
        self.mlp = _SyntheticMLP()


class _SyntheticGemma(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = nn.Module()
        layers: list[nn.Module] = [nn.Identity() for _ in range(35)]
        layers[9] = _SyntheticLayer()
        layers[34] = _SyntheticLayer()
        self.model.language_model.layers = nn.ModuleList(layers)


def lora_preflight_v95(config: Mapping[str, Any]) -> dict[str, Any]:
    bridge = config["bridge"]
    synthetic_model = _SyntheticGemma()
    synthetic_model.requires_grad_(False)
    installation = install_lora_adapters(
        synthetic_model,
        LoRASettings(
            enabled=True,
            rank=int(bridge["rank"]),
            alpha=float(bridge["alpha"]),
            dropout=float(bridge["dropout"]),
            target_modules=tuple(bridge["target_modules"]),
        ),
    )
    if installation is None:
        raise RuntimeError("V95 synthetic LoRA installation failed")
    initialize_lora_adapter_state(installation, seed=int(bridge["initialization_seed"]))
    observed = installation.state_sha256()
    if (
        installation.parameter_count != FRESH_PARAMETER_COUNT
        or observed != EXPECTED_INITIAL_STATE_SHA256
        or any(torch.count_nonzero(adapter.lora_b).item() for adapter in installation.adapters)
    ):
        raise RuntimeError("V95 deterministic zero-output LoRA initialization changed")
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


def authenticate_pinned_model_tensors_v95(config: Mapping[str, Any]) -> dict[str, Any]:
    sources = config["sources"]
    snapshot = (
        Path.home()
        / ".cache/huggingface/hub/models--google--gemma-4-E2B-it/snapshots"
        / str(sources["model_revision"])
    )
    blob = (snapshot / "model.safetensors").resolve(strict=True)
    if blob.name != sources["model_blob_sha256_identity"]:
        raise ValueError("V95 pinned Gemma blob identity changed")
    observed: dict[str, dict[str, Any]] = {}
    with safe_open(str(blob), framework="pt", device="cpu") as handle:
        for name, expected_shape in PINNED_TENSORS.items():
            if name not in handle.keys():  # noqa: SIM118
                raise ValueError(f"V95 pinned target is absent: {name}")
            sliced = handle.get_slice(name)
            shape, dtype = list(sliced.get_shape()), str(sliced.get_dtype())
            if shape != expected_shape or dtype != "BF16":
                raise ValueError(f"V95 pinned target topology changed: {name}")
            observed[name] = {"shape": shape, "dtype": dtype}
    return {
        "model_blob_sha256_identity": blob.name,
        "tensors": observed,
        "tensor_materialized": False,
        "full_gemma_model_loaded": False,
    }


def _deferred_physical_paths(config: Mapping[str, Any]) -> tuple[Path, ...]:
    deferred = config["deferred_final_lock"]
    return tuple(
        resolve_v85(root) / scene_id
        for root in deferred["physical_artifact_roots"]
        for scene_id in DEFERRED_FINAL_SCENES
    )


def assert_deferred_final_absent_v95(config: Mapping[str, Any]) -> dict[str, Any]:
    """Use metadata-only filesystem checks; never open a plan or deferred QA."""

    physical = _deferred_physical_paths(config)
    present = [str(path) for path in physical if path.exists() or path.is_symlink()]
    if present:
        raise RuntimeError(f"V95 refuses existing deferred-final artifacts: {present}")
    placeholders: dict[str, int] = {}
    for raw in config["deferred_final_lock"]["empty_qa_placeholders"]:
        path = resolve_v85(raw)
        if path.is_symlink() or not path.is_file() or path.stat().st_size != 0:
            raise RuntimeError(f"V95 deferred QA placeholder is not empty: {path}")
        placeholders[Path(raw).as_posix()] = 0
    plan_paths = tuple(
        resolve_v85(raw) for raw in config["deferred_final_lock"]["legacy_plan_files_never_opened"]
    )
    if any(not path.is_file() or path.is_symlink() for path in plan_paths):
        raise FileNotFoundError("V95 legacy plan absence-lock source changed")
    return {
        "scene_ids": list(DEFERRED_FINAL_SCENES),
        "physical_path_count_checked": len(physical),
        "physical_artifacts_present": [],
        "empty_qa_placeholders": placeholders,
        "legacy_plan_file_count_opened": 0,
        "scene_generation_performed": False,
        "rendering_performed": False,
        "feature_extraction_performed": False,
        "map_building_performed": False,
        "qa_generation_performed": False,
    }


def forbidden_training_roots_v95(config: Mapping[str, Any]) -> list[Path]:
    roots = list(_deferred_physical_paths(config))
    roots.extend(
        resolve_v85(root) / scene_id
        for root in config["deferred_final_lock"]["physical_artifact_roots"]
        for scene_id in PRIOR_EVALUATION_SCENES
    )
    roots.append(resolve_v85(config["excluded_prior_evaluation"]["labels_path"]))
    roots.extend(
        resolve_v85(path) for path in config["deferred_final_lock"]["empty_qa_placeholders"]
    )
    roots.extend(
        resolve_v85(path)
        for path in config["deferred_final_lock"]["legacy_plan_files_never_opened"]
    )
    roots.extend(
        resolve_v85(path)
        for path in (
            "reports/gemma4/questions/v56_fresh_development_validation.json",
            (
                "reports/gemma4/predictions/"
                "gemma4_v94_strict_multiscene_full40_validation_question_only.jsonl"
            ),
            (
                "reports/gemma4/predictions/"
                "gemma4_v94_strict_multiscene_full40_validation_question_only.jsonl.access.json"
            ),
            (
                "reports/gemma4/predictions/"
                "gemma4_v94_strict_multiscene_full40_validation_question_only.jsonl.provenance.json"
            ),
            ("reports/gemma4/metrics/gemma4_v94_strict_multiscene_full40_validation.json"),
        )
    )
    return list(dict.fromkeys(path.resolve() for path in roots))


def validate_parent_evidence_seal_v95(evidence: Mapping[str, Any]) -> None:
    """Reject anything except V94's aggregate, failed-gate evidence bundle."""

    score = evidence.get("score")
    gates = evidence.get("gates")
    if (
        evidence.get("artifact") != V94_EVIDENCE_ARTIFACT
        or evidence.get("schema_version") != 1
        or evidence.get("passed") is not True
        or evidence.get("behavior_score_present") is not True
        or evidence.get("behavior_gate_passed") is not False
        or evidence.get("bundle_sha256") != V94_EVIDENCE_BUNDLE_SHA256
        or evidence.get("candidate_state_sha256") != V94_STATE_SHA256
        or evidence.get("candidate_weights_sha256")
        != "574418f9458610a5d0007001c4ba732a0a8a4fd590e3b9bc4fcdd8481a646079"
        or evidence.get("prediction_row_count") != 216
        or not isinstance(gates, Mapping)
        or not gates
        or any(value is not True for value in gates.values())
        or not isinstance(score, Mapping)
        or score.get("score_sha256") != V94_SCORE_SHA256
        or score.get("behavior_gate_passed") is not False
        or score.get("status") != "measured_gate_not_passed"
    ):
        raise ValueError("V95 rejected the aggregate V94 failure-evidence seal")

    forbidden_keys = {
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

    def walk(value: object) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                if str(key).casefold() in forbidden_keys:
                    raise ValueError(
                        "V95 aggregate parent evidence serializes questions or answers"
                    )
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(evidence)


def seal_parent_evidence_v95(config_path: str | Path = CONFIG) -> dict[str, Any]:
    """Create the one allowed aggregate from direct V94 authentication.

    This is the sole V95 command allowed to traverse V94 prediction/score
    evidence.  Ordinary derivation and training bind only the resulting
    create-once aggregate JSON.
    """

    config = load_config_v95(config_path, allow_draft=True)
    destination = resolve_v85(config["sources"]["v94_hardened_scored_evidence"])
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"V95 create-once parent evidence already exists: {destination}")
    evidence = authenticate_v94_evidence(
        require_score=True,
        require_behavior_pass=False,
    )
    validate_parent_evidence_seal_v95(evidence)
    output = _atomic_create_json(destination, evidence)
    return {
        "artifact": "gemma4_v95_parent_failure_evidence_seal_v1",
        "status": "sealed_create_once",
        "output": output.as_posix(),
        "output_sha256": sha256_file_v85(output),
        "v94_bundle_sha256": evidence["bundle_sha256"],
        "v94_score_sha256": evidence["score"]["score_sha256"],
        "v94_behavior_gate_passed": False,
        "prediction_row_count": evidence["prediction_row_count"],
        "questions_or_answers_serialized": False,
    }


def authenticate_parent_v94_v95(config: Mapping[str, Any]) -> dict[str, Any]:
    """Bind the fixed-final, explicitly non-promoted V94 optimization parent."""

    sources = config["sources"]
    v85 = resolve_v85(sources["frozen_v85_checkpoint"])
    v94 = resolve_v85(sources["frozen_v94_fixed_final"])
    training_path = resolve_v85(sources["v94_training_report"])
    evidence_path = resolve_v85(sources["v94_hardened_scored_evidence"])
    for path, expected in (
        (v85 / "adapter.safetensors", sources["frozen_v85_adapter_sha256"]),
        (v85 / "runtime_metadata.json", sources["frozen_v85_metadata_sha256"]),
        (v94 / "bridge.safetensors", sources["frozen_v94_bridge_sha256"]),
        (
            v94 / "runtime_metadata.json",
            sources["frozen_v94_bridge_metadata_sha256"],
        ),
        (training_path, sources["v94_training_report_sha256"]),
        (evidence_path, sources["v94_hardened_scored_evidence_sha256"]),
    ):
        _require_hash(expected, str(path), draft=False)
        if path.is_symlink() or not path.is_file() or sha256_file_v85(path) != expected:
            raise ValueError(f"V95 frozen optimization-parent bytes changed: {path}")

    v85_metadata = _strict_json(v85 / "runtime_metadata.json")
    v85_lora = v85_metadata.get("lora")
    v85_states = v85_metadata.get("lora_bank_state_sha256")
    if (
        not isinstance(v85_lora, Mapping)
        or v85_lora.get("adapter_parameter_count") != 565_248
        or v85_lora.get("trainable_adapter_parameter_count") != 0
        or tuple(row.get("name") for row in v85_lora.get("banks", ())) != V94_BANKS[:-1]
        or not isinstance(v85_states, Mapping)
        or set(v85_states) != set(V94_BANKS[:-1])
    ):
        raise ValueError("V95 exact frozen seven-bank V85 substrate changed")

    v94_metadata = _strict_json(v94 / "runtime_metadata.json")
    bridge_state = load_file(str(v94 / "bridge.safetensors"), device="cpu")
    normalized_bridge = {f"adapters.0.{name}": value for name, value in bridge_state.items()}
    training = _strict_json(training_path)
    evidence = _strict_json(evidence_path)
    validate_parent_evidence_seal_v95(evidence)
    score = evidence.get("score")
    forbidden_release = resolve_v85(sources["forbidden_v94_release_artifact"])
    if (
        v94_metadata.get("artifact") != "gemma4_v94_strict_multiscene_full40_fixed_final_v1"
        or v94_metadata.get("status") != "fixed_final_awaiting_preregistered_acceptance_gates"
        or v94_metadata.get("state_sha256") != V94_STATE_SHA256
        or v94_metadata.get("parameter_count") != 110_592
        or v94_metadata.get("runtime_promotion_authorized") is not False
        or v94_metadata.get("evaluation_scored") is not False
        or v94_metadata.get("questions_or_answers_serialized") is not False
        or v94_metadata.get("oracle_serialized") is not False
        or set(bridge_state) != {"lora_a", "lora_b"}
        or tensor_state_sha256(normalized_bridge) != V94_STATE_SHA256
        or training.get("artifact") != "gemma4_v94_strict_multiscene_full40_training_v1"
        or training.get("status") != "fixed_final_training_complete_not_promoted"
        or training.get("candidate", {}).get("fixed_final") is not True
        or training.get("candidate", {}).get("runtime_promotion_authorized") is not False
        or training.get("candidate", {}).get("weights_sha256")
        != sources["frozen_v94_bridge_sha256"]
        or evidence.get("bundle_sha256") != V94_EVIDENCE_BUNDLE_SHA256
        or evidence.get("passed") is not True
        or evidence.get("behavior_score_present") is not True
        or evidence.get("behavior_gate_passed") is not False
        or evidence.get("candidate_state_sha256") != V94_STATE_SHA256
        or evidence.get("candidate_weights_sha256") != sources["frozen_v94_bridge_sha256"]
        or not isinstance(score, Mapping)
        or score.get("score_sha256") != V94_SCORE_SHA256
        or score.get("behavior_gate_passed") is not False
        or score.get("status") != "measured_gate_not_passed"
        or forbidden_release.exists()
        or forbidden_release.is_symlink()
    ):
        raise ValueError("V95 parent is not the exact failed fixed-final V94 state")
    parent_targets = {target for row in v85_lora["banks"] for target in row["target_modules"]}
    parent_targets.add(str(v94_metadata["target_module"]))
    if parent_targets.intersection(TARGET_MODULES):
        raise ValueError("V95 fresh targets overlap the frozen V94 stack")
    return {
        "parent": "fixed_final_nonpromoted_optimization_parent",
        "frozen_bank_count": 8,
        "frozen_adapter_parameter_count": V94_ADAPTER_PARAMETER_COUNT,
        "v94_bridge_state_sha256": V94_STATE_SHA256,
        "v94_behavior_gate_passed": False,
        "v94_runtime_release_present": False,
        "runtime_promotion_authorized": False,
        "source_reference_labels_opened": False,
    }


def parent_status_v95(config: Mapping[str, Any]) -> dict[str, Any]:
    sources = config["sources"]
    required = (
        resolve_v85(sources["frozen_v85_checkpoint"]) / "adapter.safetensors",
        resolve_v85(sources["frozen_v85_checkpoint"]) / "runtime_metadata.json",
        resolve_v85(sources["frozen_v94_fixed_final"]) / "bridge.safetensors",
        resolve_v85(sources["frozen_v94_fixed_final"]) / "runtime_metadata.json",
        resolve_v85(sources["v94_training_report"]),
        resolve_v85(sources["v94_hardened_scored_evidence"]),
    )
    if config["status"] == _DRAFT and not all(path.is_file() for path in required):
        return {
            "status": "awaiting_sealed_v94_failure_evidence",
            "required_file_count": len(required),
            "available_file_count": sum(path.is_file() for path in required),
            "parent_bytes_opened": False,
            "training_authorized": False,
        }
    return {"status": "authenticated", **authenticate_parent_v94_v95(config)}


def authenticate_training_sources_v95(config: Mapping[str, Any]) -> dict[str, str]:
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
            raise ValueError(f"V95 pinned training source changed: {raw}")
        observed[str(raw)] = value
    authenticate_parent_v94_v95(config)
    return observed


def derive_contract_v95(config_path: str | Path = CONFIG) -> dict[str, Any]:
    """Derive the fixed schedule under a read-blocking final-scene audit."""

    config = load_config_v95(config_path, allow_draft=True)
    audit = FileAccessAudit(
        forbidden_training_roots_v95(config),
        forbidden_component_names=frozenset(),
        block_forbidden=True,
    )
    with audit:
        absence = assert_deferred_final_absent_v95(config)
        rows = load_training_rows_v95(config)
        weights = balanced_class_weights_v95(config, rows)
        schedule = training_schedule_v95(rows, epochs=int(config["training"]["epochs"]))
        wrong = cross_scene_wrong_rows_v95(rows)
        wrong_schedule = cross_scene_schedule_v95(rows)
        zero_schedule = causal_control_schedule_v95(rows, arm="zero_payload")
        permutation_schedule = causal_control_schedule_v95(rows, arm="full_interior_permutation")
        memories, memory_hashes = load_scene_memories_v95(config, rows)
        lora = lora_preflight_v95(config)
        pinned = authenticate_pinned_model_tensors_v95(config)
        parent = parent_status_v95(config)
    audit.assert_clean()
    schedule_hash = canonical_sha256_v85(
        [[epoch, row.scene_id, row.question_id] for epoch, row in schedule]
    )
    wrong_schedule_hash = canonical_sha256_v85(
        [
            [epoch, row.scene_id, row.question_id, wrong_row.scene_id, wrong_row.question_id]
            for epoch, row, wrong_row in wrong_schedule
        ]
    )
    zero_schedule_hash = canonical_sha256_v85(
        [[epoch, row.scene_id, row.question_id] for epoch, row in zero_schedule]
    )
    permutation_schedule_hash = canonical_sha256_v85(
        [[epoch, row.scene_id, row.question_id] for epoch, row in permutation_schedule]
    )
    permutation_hash = canonical_sha256_v85(payload_permutation_v95().tolist())
    for field, observed in (
        ("row_order_sha256", schedule_hash),
        ("cross_scene_schedule_sha256", wrong_schedule_hash),
        ("zero_payload_schedule_sha256", zero_schedule_hash),
        ("permutation_control_schedule_sha256", permutation_schedule_hash),
        ("payload_permutation_sha256", permutation_hash),
    ):
        expected = config["training"][field]
        if expected != "TO_FILL" and expected != observed:
            raise ValueError(f"V95 derived {field} changed")
    return {
        "schema_version": 95,
        "status": "derived_not_training_authorized",
        "config_status": config["status"],
        "dataset_hashes": {
            "row_inventory_sha256": canonical_sha256_v85(_row_inventory(rows)),
            "scene_inventory_sha256": canonical_sha256_v85(sorted(memories)),
            "pair_inventory_sha256": canonical_sha256_v85(sorted({row.pair_id for row in rows})),
            "answer_class_inventory_sha256": canonical_sha256_v85(
                sorted(Counter(row.answer_class for row in rows).items())
            ),
            "balanced_class_weight_inventory_sha256": canonical_sha256_v85(sorted(weights.items())),
        },
        "training_schedule_sha256": schedule_hash,
        "cross_scene_schedule_sha256": wrong_schedule_hash,
        "cross_scene_wrong_inventory_sha256": canonical_sha256_v85(
            sorted(
                [
                    [key[0], key[1], value.scene_id, value.question_id]
                    for key, value in wrong.items()
                ]
            )
        ),
        "zero_payload_schedule_sha256": zero_schedule_hash,
        "permutation_control_schedule_sha256": permutation_schedule_hash,
        "payload_permutation_sha256": permutation_hash,
        "training_rows": len(rows),
        "training_scenes": len(memories),
        "training_pairs": len({row.pair_id for row in rows}),
        "changed_units": len(changed_units_v73(rows)),
        "cross_scene_eligible_rows": len(wrong),
        "cross_scene_wrong_schedule_rows": len(wrong_schedule),
        "cross_scene_wrong_rows_per_epoch": 249,
        "zero_payload_schedule_rows": len(zero_schedule),
        "zero_payload_rows_per_epoch": 125,
        "zero_payload_unique_rows": len({row.key for _epoch, row in zero_schedule}),
        "zero_payload_repeated_rows": sum(
            count == 2 for count in Counter(row.key for _epoch, row in zero_schedule).values()
        ),
        "permutation_control_schedule_rows": len(permutation_schedule),
        "permutation_rows_per_epoch": 125,
        "permutation_unique_rows": len({row.key for _epoch, row in permutation_schedule}),
        "permutation_repeated_rows": sum(
            count == 2
            for count in Counter(row.key for _epoch, row in permutation_schedule).values()
        ),
        "schedule_micro_rows": len(schedule),
        "optimizer_updates": len(schedule) // 8,
        "total_nll_forward_evaluations": 5836,
        "auxiliary_nll_forward_evaluations": 1996,
        "all_training_memory_hashes": memory_hashes,
        "frozen_parent": parent,
        "lora_preflight": lora,
        "pinned_model_tensors": pinned,
        "deferred_final_absence": absence,
        "prior_evaluation_scene_ids_loaded": [],
        "prior_evaluation_labels_opened": False,
        "deferred_final_scene_ids_loaded": [],
        "deferred_final_semantic_plans_opened": False,
        "deferred_final_artifacts_generated": False,
        "file_audit_forbidden_reads": audit.forbidden_accesses(),
        "full_gemma_model_loaded": False,
        "optimizer_constructed": False,
        "optimizer_updates_performed": 0,
        "training_authorized": False,
    }


def derive_preregistration_v95(
    config_path: str | Path = CONFIG,
) -> dict[str, Any]:
    """Return the complete preregistration payload without writing or sealing it."""

    config = load_config_v95(config_path, allow_draft=True)
    derived = derive_contract_v95(config_path)
    return {
        "artifact": PREREG_ARTIFACT,
        "schema_version": 95,
        "status": "draft_not_sealed_training_implementation_pending",
        "config_path": Path(config_path).as_posix(),
        "config_sha256": sha256_file_v85(config_path),
        "derived_contract": derived,
        "strict_input_contract": config["strict_input_contract"],
        "training_pool": config["training_pool"],
        "excluded_prior_evaluation": config["excluded_prior_evaluation"],
        "deferred_final_lock": config["deferred_final_lock"],
        "frozen_stack": config["frozen_stack"],
        "bridge": config["bridge"],
        "training_protocol": config["training"],
        "known_development_protocol": config["known_development_gate"],
        "deferred_evaluation_protocol": config["deferred_evaluation"],
        "fixed_gates": config["gates"],
        "parent_authenticated": derived["frozen_parent"]["status"] == "authenticated",
        "prior_evaluation_labels_opened": False,
        "deferred_final_labels_opened": False,
        "deferred_final_artifacts_generated": False,
        "full_gemma_model_loaded": False,
        "optimizer_constructed": False,
        "optimizer_updates": 0,
        "training_authorized": False,
    }


def _atomic_create_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    destination = resolve_v85(path)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"V95 create-once output exists: {destination}")
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


def build_preregistration_v95(config_path: str | Path = CONFIG) -> dict[str, Any]:
    """Seal only after V94 exists and every source hash has been pinned."""

    config = load_config_v95(config_path, allow_draft=False)
    sources = authenticate_training_sources_v95(config)
    draft = derive_preregistration_v95(config_path)
    if draft["parent_authenticated"] is not True:
        raise RuntimeError(
            "V95 cannot seal before the failed V94 fixed-final evidence is authenticated"
        )
    payload = {
        **draft,
        "status": "sealed_before_v95_full_model_load_and_deferred_generation",
        "authenticated_sources": sources,
        "parent_authenticated": True,
        "training_authorized": True,
    }
    output = _atomic_create_json(config["outputs"]["preregistration"], payload)
    return {**payload, "output": output.as_posix()}


def authenticate_preregistration_v95(
    config: Mapping[str, Any], *, config_path: str | Path = CONFIG
) -> dict[str, str]:
    path = resolve_v85(config["outputs"]["preregistration"])
    payload = _strict_json(path)
    config_hash = sha256_file_v85(config_path)
    if (
        payload.get("artifact") != PREREG_ARTIFACT
        or payload.get("schema_version") != 95
        or payload.get("status") != "sealed_before_v95_full_model_load_and_deferred_generation"
        or payload.get("config_sha256") != config_hash
        or payload.get("parent_authenticated") is not True
        or payload.get("prior_evaluation_labels_opened") is not False
        or payload.get("deferred_final_labels_opened") is not False
        or payload.get("deferred_final_artifacts_generated") is not False
        or payload.get("full_gemma_model_loaded") is not False
        or payload.get("optimizer_constructed") is not False
        or payload.get("optimizer_updates") != 0
        or payload.get("training_authorized") is not True
    ):
        raise ValueError("V95 preregistration changed")
    return {
        "config_sha256": config_hash,
        "preregistration_sha256": sha256_file_v85(path),
    }


def run_cpu_preflight_v95(config_path: str | Path = CONFIG) -> dict[str, Any]:
    config = load_config_v95(config_path, allow_draft=False)
    prereg = authenticate_preregistration_v95(config, config_path=config_path)
    report = {
        "artifact": PREFLIGHT_ARTIFACT,
        "schema_version": 95,
        "status": "passed",
        "passed": True,
        **prereg,
        "authenticated_sources": authenticate_training_sources_v95(config),
        "derived_contract": derive_contract_v95(config_path),
        "parent_authenticated": True,
        "known_development_labels_opened": False,
        "deferred_final_labels_opened": False,
        "deferred_final_artifacts_generated": False,
        "full_gemma_model_loaded": False,
        "optimizer_constructed": False,
        "optimizer_updates": 0,
        "behavior_scored": False,
        "protected_direct_v94_evidence_opened": [],
        "oracle_loaded": False,
        "runtime_promotion_authorized": False,
    }
    output = _atomic_create_json(config["outputs"]["cpu_preflight"], report)
    return {**report, "output": output.as_posix()}


def authenticate_cpu_preflight_v95(
    config: Mapping[str, Any], *, config_path: str | Path = CONFIG
) -> dict[str, str]:
    prereg = authenticate_preregistration_v95(config, config_path=config_path)
    path = resolve_v85(config["outputs"]["cpu_preflight"])
    payload = _strict_json(path)
    if (
        payload.get("artifact") != PREFLIGHT_ARTIFACT
        or payload.get("schema_version") != 95
        or payload.get("status") != "passed"
        or payload.get("passed") is not True
        or payload.get("config_sha256") != prereg["config_sha256"]
        or payload.get("preregistration_sha256") != prereg["preregistration_sha256"]
        or payload.get("parent_authenticated") is not True
        or payload.get("known_development_labels_opened") is not False
        or payload.get("deferred_final_labels_opened") is not False
        or payload.get("deferred_final_artifacts_generated") is not False
        or payload.get("full_gemma_model_loaded") is not False
        or payload.get("optimizer_constructed") is not False
        or payload.get("optimizer_updates") != 0
        or payload.get("behavior_scored") is not False
        or payload.get("protected_direct_v94_evidence_opened") != []
        or payload.get("oracle_loaded") is not False
        or payload.get("runtime_promotion_authorized") is not False
    ):
        raise ValueError("V95 CPU preflight changed")
    return {**prereg, "cpu_preflight_sha256": sha256_file_v85(path)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "seal-parent-evidence",
            "derive",
            "derive-preregistration",
            "preregister",
            "cpu-preflight",
            "authenticate",
        ),
    )
    parser.add_argument("--config", default=str(CONFIG))
    args = parser.parse_args(argv)
    if args.command == "seal-parent-evidence":
        result = seal_parent_evidence_v95(args.config)
    elif args.command == "derive":
        result = derive_contract_v95(args.config)
    elif args.command == "derive-preregistration":
        result = derive_preregistration_v95(args.config)
    elif args.command == "preregister":
        result = build_preregistration_v95(args.config)
    elif args.command == "cpu-preflight":
        result = run_cpu_preflight_v95(args.config)
    else:
        config = load_config_v95(args.config, allow_draft=False)
        result = authenticate_cpu_preflight_v95(config, config_path=args.config)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CONFIG",
    "DEFERRED_FINAL_SCENES",
    "EXPECTED_INITIAL_STATE_SHA256",
    "EXPECTED_TOTAL_ADAPTER_PARAMETER_COUNT",
    "FRESH_BANK_NAME",
    "FRESH_PARAMETER_COUNT",
    "PRIOR_EVALUATION_SCENES",
    "TARGET_MODULES",
    "TRAINING_SCENES",
    "assert_deferred_final_absent_v95",
    "authenticate_cpu_preflight_v95",
    "authenticate_parent_v94_v95",
    "authenticate_pinned_model_tensors_v95",
    "authenticate_preregistration_v95",
    "balanced_class_weights_v95",
    "build_preregistration_v95",
    "causal_control_schedule_v95",
    "cross_scene_schedule_v95",
    "cross_scene_wrong_rows_v95",
    "derive_contract_v95",
    "derive_preregistration_v95",
    "forbidden_training_roots_v95",
    "load_config_v95",
    "load_scene_memories_v95",
    "load_training_rows_v95",
    "lora_preflight_v95",
    "main",
    "payload_permutation_v95",
    "permuted_payload_memory_v95",
    "run_cpu_preflight_v95",
    "seal_parent_evidence_v95",
    "training_schedule_v95",
    "validate_parent_evidence_seal_v95",
    "zero_payload_memory_v95",
]
