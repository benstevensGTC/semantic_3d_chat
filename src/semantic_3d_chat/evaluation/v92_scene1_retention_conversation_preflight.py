"""Seal V92's retention-aware conversational repair without loading Gemma.

V92 freezes the exact V89 release plus the exact failed V90 and V91 repair
banks.  It trains one disjoint, exact-zero-output bank.  This module binds the
parent evidence, derives the fixed 590-row development schedule, authenticates
the immutable continuous scene memory, and verifies the fresh bank on CPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Final

import torch
import yaml
from safetensors import safe_open
from safetensors.torch import load_file
from torch import nn

from semantic_3d_chat.evaluation.evaluate_v91_scene1_conversational_repair import (
    conversational_match_v91,
)
from semantic_3d_chat.evaluation.metrics import normalize_answer
from semantic_3d_chat.evaluation.v56_fresh_development_score import (
    canonical_type_specific_match,
)
from semantic_3d_chat.evaluation.v85_strict_multiscene_preflight import (
    atomic_create_json_v85,
    canonical_sha256_v85,
    resolve_v85,
    sha256_file_v85,
)
from semantic_3d_chat.evaluation.v86_scene1_demo_preflight import (
    EXPECTED_PREFIX_SHA256,
    load_scene1_memory_v86,
    zero_payload_memory_v86,
)
from semantic_3d_chat.evaluation.v89_scene1_retention_preflight import (
    strict_json_v89,
)
from semantic_3d_chat.evaluation.v91_scene1_conversational_preflight import (
    authenticate_cpu_preflight_v91,
    held_wording_rows_v91,
    load_canonical_rows_v91,
    load_config_v91,
    primary_rows_v91,
    training_wording_rows_v91,
)
from semantic_3d_chat.language.lora import (
    LoRASettings,
    initialize_lora_adapter_state,
    install_lora_adapters,
    tensor_state_sha256,
)
from semantic_3d_chat.language.prefix_injection import prefix_sha256
from semantic_3d_chat.training.train_question_control_v73 import RowV73

CONFIG: Final[Path] = Path(
    "configs/experiments/gemma4_v92_scene1_retention_conversation_repair.yaml"
)
SCENE_ID: Final[str] = "scene_000001"
FRESH_BANK_NAME: Final[str] = "v92_scene1_retention_conversation_repair"
TARGET_MODULE: Final[str] = "model.language_model.layers.29.self_attn.o_proj"
PINNED_MODEL_TENSOR: Final[str] = TARGET_MODULE + ".weight"
TARGET_IN_FEATURES: Final[int] = 4096
TARGET_OUT_FEATURES: Final[int] = 1536
FRESH_PARAMETER_COUNT: Final[int] = 45056
EXPECTED_INITIAL_STATE_SHA256: Final[str] = (
    "c10d38c727df1520418a5bb9be7bac262a2b6acdef07203a669ff52f8cd08cc1"
)
V91_STATE_SHA256: Final[str] = (
    "53022311c3bc5e249a6d262fbb19b6e893a6af085be542e4d6941f7a13ea72cd"
)
PREREG_ARTIFACT: Final[str] = (
    "gemma4_v92_scene1_retention_conversation_repair_preregistration_v1"
)
PREFLIGHT_ARTIFACT: Final[str] = (
    "gemma4_v92_scene1_retention_conversation_repair_cpu_preflight_v1"
)
INTENT_IDS: Final[tuple[str, ...]] = (
    "inventory",
    "chair_presence",
    "bowl_color",
    "bowl_left_chair",
    "table_contents",
    "under_table",
    "closest",
    "wall_object",
    "cube_location",
    "lamp_turn",
    "frame_support",
    "sitting",
    "bowl_contents",
)
PRIMARY_FAILED_INTENTS: Final[tuple[str, ...]] = (
    "inventory",
    "bowl_left_chair",
    "table_contents",
)
CONVERSATIONAL_ERROR_IDS: Final[tuple[str, ...]] = (
    "v91_inventory_existing_00",
    "v91_bowl_left_chair_existing_00",
    "v91_table_contents_existing_00",
    "v91_inventory_new_held_00",
    "v91_closest_new_held_01",
    "v91_cube_location_new_held_00",
)
_HEX64: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{64}")
_DRAFT: Final[str] = "draft_before_sealed_preflight"
_SEALED: Final[str] = "sealed_before_full_model_load"


@dataclass(frozen=True)
class TrainingItemV92:
    schedule_id: str
    kind: str
    source_question_id: str
    row: RowV73
    causal_margin: bool
    intent_id: str | None = None
    wording_ordinal: int | None = None
    copy_ordinal: int | None = None


def _require_exact(value: Any, expected: Any, label: str) -> None:
    if value != expected:
        raise ValueError(f"V92 {label} changed")


def _require_hash(value: Any, label: str, *, allow_draft: bool = False) -> None:
    if isinstance(value, str) and _HEX64.fullmatch(value):
        return
    if allow_draft and value == "TO_FILL":
        return
    raise ValueError(f"V92 {label} is not sealed")


def _answer_class(answer: str) -> str:
    normalized = normalize_answer(answer)
    if not normalized:
        raise ValueError("V92 answer normalizes empty")
    return "answer_" + hashlib.sha256(normalized.encode()).hexdigest()[:20]


def _parent_config(config: Mapping[str, Any]) -> dict[str, Any]:
    return load_config_v91(config["sources"]["parent_v91_config"], allow_draft=False)


def _validate_intents(config: Mapping[str, Any]) -> None:
    intents = config.get("conversational_intents")
    if not isinstance(intents, list) or len(intents) != 13:
        raise ValueError("V92 requires exactly thirteen intents")
    if tuple(raw.get("id") for raw in intents if isinstance(raw, Mapping)) != INTENT_IDS:
        raise ValueError("V92 intent identity/order changed")
    parent = _parent_config(config)["conversational_intents"]
    parent_by_id = {str(raw["id"]): raw for raw in parent}
    held_questions: list[str] = []
    known_questions: list[str] = []
    for raw in intents:
        if not isinstance(raw, Mapping) or set(raw) != {
            "id",
            "family",
            "answer",
            "new_held_wordings",
        }:
            raise ValueError("V92 intent schema changed")
        source = parent_by_id[str(raw["id"])]
        if raw["family"] != source["family"] or str(raw["answer"]) != str(source["answer"]):
            raise ValueError("V92 intent semantics changed")
        held = raw["new_held_wordings"]
        if (
            not isinstance(held, list)
            or len(held) != 2
            or any(not isinstance(value, str) or not value.strip() for value in held)
        ):
            raise ValueError("V92 requires two new held wordings per intent")
        held_questions.extend(held)
        known_questions.extend(source["existing_wordings"])
        known_questions.extend(source["new_held_wordings"])
    normalized_known = {normalize_answer(value) for value in known_questions}
    normalized_held = [normalize_answer(value) for value in held_questions]
    if (
        len(normalized_known) != 104
        or len(normalized_held) != 26
        or len(set(normalized_held)) != 26
        or normalized_known & set(normalized_held)
    ):
        raise ValueError("V92 held wording isolation changed")


def load_config_v92(
    path: str | Path = CONFIG, *, allow_draft: bool = True
) -> dict[str, Any]:
    source = resolve_v85(path)
    if source.is_symlink() or not source.is_file():
        raise FileNotFoundError(source)
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or set(payload) != {"v92"}:
        raise ValueError("V92 config must contain exactly one v92 mapping")
    config = payload["v92"]
    if not isinstance(config, Mapping):
        raise TypeError("V92 config payload must be a mapping")
    _require_exact(config.get("schema_version"), 92, "schema version")
    _require_exact(
        config.get("artifact"),
        "gemma4_v92_scene1_retention_conversation_repair_direct_memory_v1",
        "artifact identity",
    )
    _require_exact(config.get("seed"), 920092, "seed")
    status = config.get("status")
    if status not in {_DRAFT, _SEALED} or (not allow_draft and status != _SEALED):
        raise ValueError("V92 config has not been sealed")
    _require_exact(
        config.get("strict_input_contract"),
        {
            "shape": [1, 738, 1536],
            "native_boi_eoi_retained": True,
            "continuous_environment_payload_tokens": 736,
            "compiled_before_question": True,
            "reused_byte_identically_across_questions": True,
            "supplied_directly_to_native_gemma_image_prefix": True,
            "question_derived_environmental_tokens": 0,
            "question_conditioned_environmental_readout": False,
            "question_dependent_scene_processing": False,
            "question_dependent_retrieval": False,
            "semantic_or_spatial_top_k_selection": False,
            "control_tokens": 0,
            "environmental_text_inputs": [],
        },
        "strict input contract",
    )
    _require_exact(
        config.get("parent_v91_result"),
        {
            "model_acceptance_gate_passed": False,
            "runtime_promotion_authorized": False,
            "frozen_parent_state_invariant": True,
            "canonical_correct": 115,
            "canonical_total": 138,
            "canonical_error_count": 23,
            "canonical_type_correct": {
                "presence": 22,
                "count": 9,
                "metric": 1,
                "attribute": 14,
                "spatial_relation": 69,
                "support": 0,
            },
            "primary_correct": 10,
            "primary_total": 13,
            "primary_failed_intents": list(PRIMARY_FAILED_INTENTS),
            "held_wording_correct": 23,
            "held_wording_total": 26,
            "held_wording_failures": {"inventory": 1, "closest": 1, "cube_location": 1},
            "causal_mean_zero_minus_correct_nll": 1.3944181121218067,
            "causal_prediction_changes": 10,
            "candidate_state_sha256": V91_STATE_SHA256,
            "candidate_checkpoint_sha256": (
                "02c140b1ceaa29252d766aa19931e24a52a8175c17c93f306e510cd1215bf148"
            ),
        },
        "measured V91 parent result",
    )
    dataset = config.get("dataset")
    if not isinstance(dataset, Mapping):
        raise TypeError("V92 dataset must be a mapping")
    expected_dataset = {
        "scene_id": SCENE_ID,
        "canonical_row_count": 138,
        "canonical_row_inventory_sha256": (
            "9919ff1bee4611dce4132d79fa50f6f6b4ace567a6df780a2e0e21bd88237a8e"
        ),
        "parent_correct_count": 115,
        "parent_error_count": 23,
        "parent_error_replay_copies": 5,
        "parent_correct_anchor_replay_copies": 1,
        "conversational_intent_count": 13,
        "known_wordings_per_intent": 8,
        "known_conversational_row_count": 104,
        "v91_conversational_error_row_count": 6,
        "v91_conversational_error_extra_copies": 10,
        "primary_failed_intent_count": 3,
        "primary_failed_known_row_count": 24,
        "primary_failed_known_extra_copies": 2,
        "primary_success_anchor_count": 10,
        "new_held_wordings_per_intent": 2,
        "new_held_wording_row_count": 26,
        "rows_per_epoch": 590,
        "epochs": 3,
        "total_micro_rows": 1770,
        "labels_derived_offline_from_oracle": True,
        "oracle_loaded_during_training": False,
        "questions_or_answers_serialized_at_runtime": False,
        "exact_v91_conversational_error_question_ids": list(CONVERSATIONAL_ERROR_IDS),
    }
    if any(dataset.get(key) != value for key, value in expected_dataset.items()):
        raise ValueError("V92 fixed dataset contract changed")
    for key in ("training_inventory_sha256", "training_schedule_sha256"):
        _require_hash(dataset.get(key), key, allow_draft=allow_draft)
    _require_exact(
        config.get("frozen_stack"),
        {
            "v89_parent_bank_count": 11,
            "v89_parent_adapter_parameter_count": 872448,
            "v90_bank_name": "v90_scene1_conversational_bridge",
            "v90_target_module": "model.language_model.layers.28.self_attn.o_proj",
            "v90_state_sha256": (
                "70e236711d8ac1fe7cf808f6f4e939b29db476016c8ef49db143707df0f3bde7"
            ),
            "v90_parameter_count": 28672,
            "v91_bank_name": "v91_scene1_conversational_repair",
            "v91_target_module": "model.language_model.layers.33.mlp.down_proj",
            "v91_state_sha256": V91_STATE_SHA256,
            "v91_parameter_count": 221184,
            "total_frozen_bank_count": 13,
            "total_frozen_adapter_parameter_count": 1122304,
            "v90_runtime_promotable": False,
            "v91_runtime_promotable": False,
            "exact_failed_candidates_used_as_offline_parent": True,
            "merged_weights": False,
        },
        "frozen stack",
    )
    _require_exact(
        config.get("bridge"),
        {
            "bank_name": FRESH_BANK_NAME,
            "target_module": TARGET_MODULE,
            "target_layer_type": "self_attention_output",
            "pinned_weight_shape": [TARGET_OUT_FEATURES, TARGET_IN_FEATURES],
            "pinned_weight_dtype": "BF16",
            "target_in_features": TARGET_IN_FEATURES,
            "target_out_features": TARGET_OUT_FEATURES,
            "rank": 8,
            "alpha": 16.0,
            "dropout": 0.0,
            "trainable_parameter_count": FRESH_PARAMETER_COUNT,
            "initialization_algorithm": "cpu_kaiming_uniform_a_exact_zero_b",
            "initialization_seed": 920092,
            "expected_initial_state_sha256": EXPECTED_INITIAL_STATE_SHA256,
            "disjoint_from_all_frozen_banks": True,
        },
        "fresh bridge",
    )
    _require_exact(
        config.get("training"),
        {
            "optimizer": "AdamW",
            "learning_rate": 0.0001,
            "weight_decay": 0.0,
            "gradient_accumulation_rows": 6,
            "optimizer_updates": 295,
            "gradient_clip_norm": 1.0,
            "answer_ce_weight": 1.0,
            "zero_payload_margin_weight": 1.0,
            "zero_payload_target_margin_nll": 0.5,
            "row_order_seed": 920092,
            "checkpoint_selection": "fixed_final_update_295",
            "intermediate_behavior_selection": False,
        },
        "training protocol",
    )
    _require_exact(
        config.get("gates"),
        {
            "canonical_correct_minimum": 122,
            "canonical_total": 138,
            "canonical_presence_correct_minimum": 21,
            "canonical_count_correct_minimum": 9,
            "canonical_metric_correct_minimum": 1,
            "canonical_attribute_correct_minimum": 15,
            "canonical_spatial_correct_minimum": 73,
            "canonical_support_correct_minimum": 1,
            "primary_conversational_required_correct": 12,
            "primary_conversational_total": 13,
            "core_actionable_required_correct": 6,
            "core_actionable_total": 6,
            "new_held_wording_required_correct": 22,
            "new_held_wording_total": 26,
            "new_held_wording_each_intent_minimum": 1,
            "causal_mean_zero_minus_correct_nll_minimum": 0.5,
            "causal_prediction_change_minimum": 8,
            "exact_prefix_hash_invariance_required": True,
            "exact_total_environment_input_invariance_required": True,
            "oracle_physically_unavailable_during_runtime_required": True,
            "forbidden_runtime_read_count_maximum": 0,
        },
        "acceptance gates",
    )
    _require_exact(
        config.get("outputs"),
        {
            "preregistration": "reports/gemma4/metrics/gemma4_v92_scene1_retention_conversation_repair_preregistration.json",
            "cpu_preflight": "reports/gemma4/metrics/gemma4_v92_scene1_retention_conversation_repair_cpu_preflight.json",
            "fixed_final_candidate": "reports/gemma4/artifacts/v92_scene1_retention_conversation_repair_final",
            "training_report": "reports/gemma4/metrics/gemma4_v92_scene1_retention_conversation_repair_training.json",
            "evaluation_predictions": "reports/gemma4/predictions/gemma4_v92_scene1_retention_conversation_repair_evaluation.json",
            "evaluation_report": "reports/gemma4/metrics/gemma4_v92_scene1_retention_conversation_repair_evaluation.json",
        },
        "output namespace",
    )
    _require_exact(
        config.get("scope"),
        {
            "post_v91_training_set_development": True,
            "exact_failed_v90_and_v91_candidates_frozen": True,
            "single_scene_conversational_repair": True,
            "development_known_training_wordings": True,
            "newly_held_wording_only_evaluation": True,
            "local_inference_only": True,
            "cloud_inference": False,
            "held_out_scene_generalization_claim": False,
            "official_validation_loaded": False,
            "official_test_loaded": False,
            "deferred_final_loaded": False,
            "runtime_promotion_authorized": False,
        },
        "development scope",
    )
    sources = config.get("sources")
    if not isinstance(sources, Mapping):
        raise TypeError("V92 sources must be a mapping")
    for key, value in sources.items():
        if key.endswith("_sha256"):
            _require_hash(value, key, allow_draft=allow_draft and key in {
                "preflight_source_sha256", "training_source_sha256", "evaluation_source_sha256"
            })
    _validate_intents(config)
    return dict(config)


def load_canonical_rows_v92(config: Mapping[str, Any]) -> tuple[RowV73, ...]:
    rows = load_canonical_rows_v91(_parent_config(config))
    if len(rows) != 138 or {row.scene_id for row in rows} != {SCENE_ID}:
        raise ValueError("V92 canonical inventory changed")
    return rows


def known_wording_rows_v92(config: Mapping[str, Any]) -> tuple[RowV73, ...]:
    parent = _parent_config(config)
    rows = (*training_wording_rows_v91(parent), *held_wording_rows_v91(parent))
    if len(rows) != 104 or len({row.question_id for row in rows}) != 104:
        raise RuntimeError("V92 known conversational inventory changed")
    return rows


training_wording_rows_v92 = known_wording_rows_v92
conversational_rows_v92 = known_wording_rows_v92


def primary_rows_v92(config: Mapping[str, Any]) -> tuple[RowV73, ...]:
    rows = primary_rows_v91(_parent_config(config))
    if len(rows) != 13:
        raise RuntimeError("V92 primary inventory changed")
    return rows


def _held_row(raw: Mapping[str, Any], ordinal: int, question: str) -> RowV73:
    intent = str(raw["id"])
    answer = str(raw["answer"])
    return RowV73(
        scene_id=SCENE_ID,
        question_id=f"v92_{intent}_new_held_{ordinal:02d}",
        question=question.strip(),
        answer=answer,
        answer_class=_answer_class(answer),
        answer_type=str(raw["family"]),
        pair_id=f"v92_conversation_{intent}",
        paired_scene_id=SCENE_ID,
        question_key=f"v92_conversation_{intent}",
        change_type="wording",
        expected_change=False,
    )


def held_wording_rows_v92(config: Mapping[str, Any]) -> tuple[RowV73, ...]:
    rows = tuple(
        _held_row(raw, ordinal, str(question))
        for raw in config["conversational_intents"]
        for ordinal, question in enumerate(raw["new_held_wordings"])
    )
    if len(rows) != 26 or len({row.question_id for row in rows}) != 26:
        raise RuntimeError("V92 new held wording inventory changed")
    return rows


new_held_wording_rows_v92 = held_wording_rows_v92


def _clone(row: RowV73, schedule_id: str) -> RowV73:
    return replace(row, question_id=schedule_id, question_key=schedule_id)


def _parent_prediction(config: Mapping[str, Any]) -> dict[str, Any]:
    payload = strict_json_v89(config["sources"]["parent_v91_predictions"])
    if (
        payload.get("artifact") != "gemma4_v91_scene1_conversational_repair_predictions_v1"
        or payload.get("schema_version") != 91
        or payload.get("status") != "fixed_final_evaluation_only_not_runtime"
        or payload.get("scene_id") != SCENE_ID
        or payload.get("canonical_row_count") != 138
        or payload.get("primary_conversational_row_count") != 13
        or payload.get("new_held_wording_row_count") != 26
        or payload.get("frozen_parent_state_invariant") is not True
        or payload.get("candidate_state_invariant") is not True
        or payload.get("runtime_promotion_authorized") is not False
    ):
        raise ValueError("V92 V91 prediction identity changed")
    return payload


def parent_correct_and_errors_v92(
    config: Mapping[str, Any], rows: Sequence[RowV73]
) -> tuple[tuple[RowV73, ...], tuple[RowV73, ...]]:
    records = _parent_prediction(config).get("canonical_records")
    by_id = {row.question_id: row for row in rows}
    if not isinstance(records, list) or len(records) != 138:
        raise ValueError("V92 V91 canonical predictions changed")
    records_by_id = {str(record.get("question_id")): record for record in records}
    if set(records_by_id) != set(by_id):
        raise ValueError("V92 V91 canonical prediction coverage changed")
    correct: list[RowV73] = []
    errors: list[RowV73] = []
    for question_id in sorted(by_id):
        row = by_id[question_id]
        record = records_by_id[question_id]
        if (
            record.get("scene_id") != SCENE_ID
            or str(record.get("reference_answer")) != row.answer
            or record.get("answer_type") != row.answer_type
            or record.get("scene_memory_sha256") != EXPECTED_PREFIX_SHA256
        ):
            raise ValueError("V92 V91 canonical binding changed")
        target = correct if canonical_type_specific_match(
            row.answer_type, record.get("prediction", ""), row.answer
        ) else errors
        target.append(row)
    if len(correct) != 115 or len(errors) != 23:
        raise ValueError("V92 requires exact V91 115/23 split")
    return tuple(correct), tuple(errors)


def conversational_errors_v92(config: Mapping[str, Any]) -> tuple[RowV73, ...]:
    payload = _parent_prediction(config)
    records = [
        *payload["primary_conversational_records"],
        *payload["new_held_wording_records"],
    ]
    known = {row.question_id: row for row in known_wording_rows_v92(config)}
    errors: list[RowV73] = []
    for record in records:
        question_id = str(record.get("question_id"))
        row = known.get(question_id)
        if row is None:
            raise ValueError("V92 V91 conversational record has unknown row")
        intent_id = row.pair_id.removeprefix("v91_conversation_")
        if not conversational_match_v91(
            intent_id, row.answer_type, record.get("prediction", ""), row.answer
        ):
            errors.append(row)
    if tuple(row.question_id for row in errors) != CONVERSATIONAL_ERROR_IDS:
        raise ValueError("V92 exact six V91 conversational errors changed")
    return tuple(errors)


def derive_training_items_v92(
    config: Mapping[str, Any], canonical: Sequence[RowV73] | None = None
) -> tuple[TrainingItemV92, ...]:
    canonical_rows = tuple(canonical) if canonical is not None else load_canonical_rows_v92(config)
    correct, errors = parent_correct_and_errors_v92(config, canonical_rows)
    known = known_wording_rows_v92(config)
    known_by_id = {row.question_id: row for row in known}
    primary = primary_rows_v92(config)
    items: list[TrainingItemV92] = [
        TrainingItemV92(row.question_id, "canonical", row.question_id, row, False)
        for row in sorted(canonical_rows, key=lambda value: value.question_id)
    ]
    for row in sorted(errors, key=lambda value: value.question_id):
        for copy in range(5):
            schedule_id = f"v92_error_{copy:02d}_{row.question_id}"
            items.append(TrainingItemV92(
                schedule_id, "error_replay", row.question_id,
                _clone(row, schedule_id), False, copy_ordinal=copy,
            ))
    for row in sorted(correct, key=lambda value: value.question_id):
        schedule_id = f"v92_anchor_{row.question_id}"
        items.append(TrainingItemV92(
            schedule_id, "correct_anchor_replay", row.question_id,
            _clone(row, schedule_id), False,
        ))
    for row in known:
        intent = row.pair_id.removeprefix("v91_conversation_")
        is_existing = "_existing_" in row.question_id
        ordinal = int(row.question_id.rsplit("_", 1)[1]) + (0 if is_existing else 6)
        items.append(TrainingItemV92(
            f"v92_known_{row.question_id}", "conversational_known", row.question_id,
            _clone(row, f"v92_known_{row.question_id}"),
            row.question_id.endswith("_existing_00"), intent, ordinal, 0,
        ))
    for row in conversational_errors_v92(config):
        intent = row.pair_id.removeprefix("v91_conversation_")
        ordinal = int(row.question_id.rsplit("_", 1)[1]) + (
            0 if "_existing_" in row.question_id else 6
        )
        for copy in range(10):
            schedule_id = f"v92_conversation_error_{copy:02d}_{row.question_id}"
            items.append(TrainingItemV92(
                schedule_id, "conversational_error_replay", row.question_id,
                _clone(row, schedule_id), False, intent, ordinal, copy,
            ))
    for intent in PRIMARY_FAILED_INTENTS:
        for row in known:
            if row.pair_id != f"v91_conversation_{intent}":
                continue
            ordinal = int(row.question_id.rsplit("_", 1)[1]) + (
                0 if "_existing_" in row.question_id else 6
            )
            for copy in range(2):
                schedule_id = f"v92_failed_intent_{copy:02d}_{row.question_id}"
                items.append(TrainingItemV92(
                    schedule_id, "primary_failed_intent_replay", row.question_id,
                    _clone(row, schedule_id), False, intent, ordinal, copy,
                ))
    for row in primary:
        intent = row.pair_id.removeprefix("v91_conversation_")
        if intent in PRIMARY_FAILED_INTENTS:
            continue
        source = known_by_id[row.question_id]
        schedule_id = f"v92_primary_success_{row.question_id}"
        items.append(TrainingItemV92(
            schedule_id, "primary_success_anchor", row.question_id,
            _clone(source, schedule_id), False, intent, 0, 0,
        ))
    counts = Counter(item.kind for item in items)
    held_ids = {row.question_id for row in held_wording_rows_v92(config)}
    expected = {
        "canonical": 138,
        "error_replay": 115,
        "correct_anchor_replay": 115,
        "conversational_known": 104,
        "conversational_error_replay": 60,
        "primary_failed_intent_replay": 48,
        "primary_success_anchor": 10,
    }
    if (
        len(items) != 590
        or len({item.schedule_id for item in items}) != 590
        or counts != expected
        or held_ids & {item.source_question_id for item in items}
        or sum(item.causal_margin for item in items) != 13
    ):
        raise RuntimeError(f"V92 fixed 590-row inventory changed: {counts}")
    return tuple(items)


def inventory_v92(items: Sequence[TrainingItemV92]) -> list[dict[str, Any]]:
    return [
        {
            "schedule_id": item.schedule_id,
            "kind": item.kind,
            "source_question_id": item.source_question_id,
            "question": item.row.question,
            "answer": item.row.answer,
            "answer_class": item.row.answer_class,
            "answer_type": item.row.answer_type,
            "causal_margin": item.causal_margin,
            "intent_id": item.intent_id,
            "wording_ordinal": item.wording_ordinal,
            "copy_ordinal": item.copy_ordinal,
        }
        for item in items
    ]


training_inventory_v92 = inventory_v92


def schedule_v92(
    items: Sequence[TrainingItemV92], *, seed: int = 920092, epochs: int = 3
) -> tuple[tuple[int, TrainingItemV92], ...]:
    schedule: list[tuple[int, TrainingItemV92]] = []
    for epoch in range(epochs):
        shuffled = sorted(items, key=lambda item: item.schedule_id)
        random.Random(seed + epoch).shuffle(shuffled)
        schedule.extend((epoch, item) for item in shuffled)
    return tuple(schedule)


training_schedule_v92 = schedule_v92


class _SyntheticAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.o_proj = nn.Linear(
            TARGET_IN_FEATURES, TARGET_OUT_FEATURES, bias=False, dtype=torch.bfloat16
        )


class _SyntheticLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = _SyntheticAttention()


class _SyntheticLanguage(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Identity() for _ in range(29)]
            + [_SyntheticLayer()]
            + [nn.Identity() for _ in range(5)]
        )


class _SyntheticGemma(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = _SyntheticLanguage()


def derive_lora_preflight_v92(config: Mapping[str, Any]) -> dict[str, Any]:
    bridge = config["bridge"]
    installation = install_lora_adapters(
        _SyntheticGemma(),
        LoRASettings(
            enabled=True,
            rank=int(bridge["rank"]),
            alpha=float(bridge["alpha"]),
            dropout=float(bridge["dropout"]),
            target_modules=(str(bridge["target_module"]),),
        ),
    )
    if installation is None:
        raise RuntimeError("V92 synthetic LoRA installation failed")
    initialize_lora_adapter_state(installation, seed=int(bridge["initialization_seed"]))
    return {
        "bank_name": FRESH_BANK_NAME,
        "target_modules": list(installation.target_names),
        "parameter_count": installation.parameter_count,
        "initial_state_sha256": installation.state_sha256(),
        "base_projection_type": "torch.nn.Linear",
        "base_projection_weight_shape": [TARGET_OUT_FEATURES, TARGET_IN_FEATURES],
        "lora_a_shape": [8, TARGET_IN_FEATURES],
        "lora_b_shape": [TARGET_OUT_FEATURES, 8],
        "lora_b_nonzero_count": sum(
            int(torch.count_nonzero(adapter.lora_b).item())
            for adapter in installation.adapters
        ),
        "exact_zero_output_at_initialization": True,
        "full_gemma_model_loaded": False,
        "device": "cpu",
    }


def lora_preflight_v92(config: Mapping[str, Any]) -> dict[str, Any]:
    result = derive_lora_preflight_v92(config)
    if (
        result["target_modules"] != [TARGET_MODULE]
        or result["parameter_count"] != FRESH_PARAMETER_COUNT
        or result["base_projection_weight_shape"] != [TARGET_OUT_FEATURES, TARGET_IN_FEATURES]
        or result["lora_a_shape"] != [8, TARGET_IN_FEATURES]
        or result["lora_b_shape"] != [TARGET_OUT_FEATURES, 8]
        or result["initial_state_sha256"] != EXPECTED_INITIAL_STATE_SHA256
        or result["initial_state_sha256"] != config["bridge"]["expected_initial_state_sha256"]
        or result["lora_b_nonzero_count"] != 0
    ):
        raise RuntimeError("V92 deterministic LoRA initialization changed")
    return result


def memory_preflight_v92(config: Mapping[str, Any]) -> dict[str, Any]:
    memory, observed_hash, metadata = load_scene1_memory_v86(config)
    zero = zero_payload_memory_v86(memory)
    result = {
        "scene_id": SCENE_ID,
        "shape": list(memory.shape),
        "dtype": str(memory.dtype),
        "canonical_prefix_sha256": observed_hash,
        "zero_payload_prefix_sha256": prefix_sha256(zero),
        "native_boi_preserved": torch.equal(memory[:, :1], zero[:, :1]),
        "native_eoi_preserved": torch.equal(memory[:, -1:], zero[:, -1:]),
        "zeroed_interior_tokens": 736,
        "compiled_before_question": metadata.get("compiled_before_user_question") is True,
        "question_inputs_used": metadata.get("question_inputs_used_for_compilation") is True,
        "questions_or_answers_serialized": metadata.get("questions_or_answers_serialized") is True,
        "oracle_loaded": metadata.get("oracle_loaded") is True,
        "model_loaded": False,
        "device": "cpu",
    }
    if (
        result["shape"] != [1, 738, 1536]
        or result["dtype"] != "torch.bfloat16"
        or result["canonical_prefix_sha256"] != EXPECTED_PREFIX_SHA256
        or not result["native_boi_preserved"]
        or not result["native_eoi_preserved"]
        or not result["compiled_before_question"]
        or result["question_inputs_used"]
        or result["questions_or_answers_serialized"]
        or result["oracle_loaded"]
    ):
        raise RuntimeError("V92 immutable memory preflight changed")
    return result


def authenticate_parent_v92(config: Mapping[str, Any]) -> dict[str, Any]:
    sources = config["sources"]
    parent_config = _parent_config(config)
    parent_preflight = authenticate_cpu_preflight_v91(
        parent_config, config_path=sources["parent_v91_config"]
    )
    if parent_preflight != {
        "config_sha256": sources["parent_v91_config_sha256"],
        "preregistration_sha256": sources["parent_v91_preregistration_sha256"],
        "cpu_preflight_sha256": sources["parent_v91_cpu_preflight_sha256"],
    }:
        raise ValueError("V92 V91 CPU preflight binding changed")
    root = resolve_v85(sources["parent_v91_candidate"])
    metadata = strict_json_v89(root / "runtime_metadata.json")
    tensors = load_file(str(root / "bridge.safetensors"), device="cpu")
    if (
        metadata.get("artifact") != "gemma4_v91_scene1_conversational_repair_fixed_final_v1"
        or metadata.get("schema_version") != 91
        or metadata.get("bank_name") != "v91_scene1_conversational_repair"
        or metadata.get("target_module") != "model.language_model.layers.33.mlp.down_proj"
        or metadata.get("rank") != 16
        or float(metadata.get("alpha", -1)) != 32.0
        or metadata.get("parameter_count") != 221184
        or metadata.get("state_sha256") != V91_STATE_SHA256
        or metadata.get("weights_sha256") != sources["parent_v91_bridge_sha256"]
        or metadata.get("total_bank_count") != 13
        or metadata.get("total_adapter_parameter_count") != 1122304
        or metadata.get("runtime_promotion_authorized") is not False
        or metadata.get("questions_or_answers_serialized") is not False
        or metadata.get("training_inventory_serialized") is not False
        or metadata.get("oracle_serialized") is not False
        or set(tensors) != {"lora_a", "lora_b"}
        or list(tensors["lora_a"].shape) != [16, 12288]
        or list(tensors["lora_b"].shape) != [1536, 16]
        or tensor_state_sha256({f"adapters.0.{key}": value for key, value in tensors.items()})
        != V91_STATE_SHA256
    ):
        raise ValueError("V92 exact failed V91 bridge authentication failed")
    training = strict_json_v89(sources["parent_v91_training"])
    evaluation = strict_json_v89(sources["parent_v91_evaluation"])
    metrics = evaluation.get("metrics")
    by_type = metrics.get("canonical_accuracy_by_answer_type") if isinstance(metrics, Mapping) else None
    primary = metrics.get("primary_conversational") if isinstance(metrics, Mapping) else None
    held = metrics.get("new_held_wording") if isinstance(metrics, Mapping) else None
    causal = metrics.get("causal_control") if isinstance(metrics, Mapping) else None
    if (
        training.get("artifact") != "gemma4_v91_scene1_conversational_repair_training_v1"
        or training.get("status") != "fixed_final_training_complete_not_promoted"
        or training.get("config_sha256") != sources["parent_v91_config_sha256"]
        or training.get("preregistration_sha256") != sources["parent_v91_preregistration_sha256"]
        or training.get("cpu_preflight_sha256") != sources["parent_v91_cpu_preflight_sha256"]
        or training.get("optimizer_updates") != 295
        or training.get("micro_rows_consumed") != 1770
        or training.get("causal_margin_rows_consumed") != 39
        or training.get("oracle_loaded") is not False
        or training.get("protected_read_count") != 0
        or training.get("runtime_promotion_authorized") is not False
        or evaluation.get("artifact") != "gemma4_v91_scene1_conversational_repair_evaluation_v1"
        or evaluation.get("status") != "model_gates_fail_not_runtime_promotable"
        or evaluation.get("evaluation_predictions_sha256") != sources["parent_v91_predictions_sha256"]
        or evaluation.get("training_report_sha256") != sources["parent_v91_training_sha256"]
        or evaluation.get("runtime_promotion_authorized") is not False
        or evaluation.get("oracle_loaded") is not False
        or not isinstance(metrics, Mapping)
        or metrics.get("model_acceptance_gate_passed") is not False
        or metrics.get("canonical_type_specific") != {"accuracy": 115 / 138, "correct": 115, "total": 138}
        or not isinstance(by_type, Mapping)
        or {key: by_type[key]["correct"] for key in by_type}
        != {"attribute": 14, "count": 9, "metric": 1, "presence": 22, "spatial_relation": 69, "support": 0}
        or not isinstance(primary, Mapping)
        or primary.get("correct") != 10
        or primary.get("total") != 13
        or primary.get("core_actionable_correct") != 5
        or not isinstance(held, Mapping)
        or held.get("correct") != 23
        or held.get("total") != 26
        or not isinstance(causal, Mapping)
        or causal.get("mean_zero_minus_correct_nll") != 1.3944181121218067
        or causal.get("canonical_prediction_changes") != 10
    ):
        raise ValueError("V92 measured V91 evidence authentication failed")
    canonical = load_canonical_rows_v92(config)
    _correct, errors = parent_correct_and_errors_v92(config, canonical)
    conversation_errors = conversational_errors_v92(config)
    return {
        "v89_frozen_bank_count": 11,
        "v90_frozen_bank_count": 1,
        "v91_frozen_bank_count": 1,
        "total_frozen_bank_count": 13,
        "total_frozen_parameter_count": 1122304,
        "v91_state_sha256": V91_STATE_SHA256,
        "v91_model_acceptance_gate_passed": False,
        "v91_runtime_promotion_authorized": False,
        "canonical_correct": 115,
        "canonical_errors": len(errors),
        "primary_correct": 10,
        "held_wording_correct": 23,
        "conversational_error_question_ids": [row.question_id for row in conversation_errors],
        "new_repair_is_post_failure_development": True,
    }


def _source_bindings(config: Mapping[str, Any]) -> tuple[tuple[str, str, bool], ...]:
    sources = config["sources"]
    return (
        (sources["runtime_config"], sources["runtime_config_sha256"], False),
        (sources["scene1_qa"], sources["scene1_qa_sha256"], False),
        (str(Path(sources["scene1_memory"]) / "memory.safetensors"), sources["scene1_memory_tensor_sha256"], False),
        (str(Path(sources["scene1_memory"]) / "runtime_metadata.json"), sources["scene1_memory_metadata_sha256"], False),
        (str(Path(sources["parent_v89_checkpoint"]) / "adapter.safetensors"), sources["parent_v89_adapter_sha256"], False),
        (str(Path(sources["parent_v89_checkpoint"]) / "runtime_metadata.json"), sources["parent_v89_metadata_sha256"], False),
        (sources["parent_v90_config"], sources["parent_v90_config_sha256"], False),
        (str(Path(sources["parent_v90_candidate"]) / "bridge.safetensors"), sources["parent_v90_bridge_sha256"], False),
        (str(Path(sources["parent_v90_candidate"]) / "runtime_metadata.json"), sources["parent_v90_metadata_sha256"], False),
        (sources["parent_v91_config"], sources["parent_v91_config_sha256"], False),
        (sources["parent_v91_preregistration"], sources["parent_v91_preregistration_sha256"], False),
        (sources["parent_v91_cpu_preflight"], sources["parent_v91_cpu_preflight_sha256"], False),
        (sources["parent_v91_training"], sources["parent_v91_training_sha256"], False),
        (str(Path(sources["parent_v91_candidate"]) / "bridge.safetensors"), sources["parent_v91_bridge_sha256"], False),
        (str(Path(sources["parent_v91_candidate"]) / "runtime_metadata.json"), sources["parent_v91_metadata_sha256"], False),
        (sources["parent_v91_predictions"], sources["parent_v91_predictions_sha256"], False),
        (sources["parent_v91_evaluation"], sources["parent_v91_evaluation_sha256"], False),
        (sources["preflight_source"], sources["preflight_source_sha256"], True),
        (sources["training_source"], sources["training_source_sha256"], True),
        (sources["evaluation_source"], sources["evaluation_source_sha256"], True),
    )


def authenticate_pinned_model_tensor_v92(config: Mapping[str, Any]) -> dict[str, Any]:
    sources = config["sources"]
    snapshot = (
        Path.home()
        / ".cache/huggingface/hub/models--google--gemma-4-E2B-it/snapshots"
        / str(sources["model_revision"])
    )
    model_blob = (snapshot / "model.safetensors").resolve(strict=True)
    if model_blob.name != sources["model_blob_sha256_identity"]:
        raise ValueError("V92 pinned Gemma blob identity changed")
    with safe_open(str(model_blob), framework="pt", device="cpu") as handle:
        if PINNED_MODEL_TENSOR not in handle.keys():  # noqa: SIM118
            raise ValueError("V92 pinned layer-29 tensor absent")
        tensor_slice = handle.get_slice(PINNED_MODEL_TENSOR)
        shape = list(tensor_slice.get_shape())
        dtype = str(tensor_slice.get_dtype())
    result = {
        "model_id": sources["model_id"],
        "model_revision": sources["model_revision"],
        "model_blob_sha256_identity": model_blob.name,
        "tensor_name": PINNED_MODEL_TENSOR,
        "shape": shape,
        "dtype": dtype,
        "header_read_via_safe_open": True,
        "tensor_materialized": False,
        "full_gemma_model_loaded": False,
    }
    if shape != [TARGET_OUT_FEATURES, TARGET_IN_FEATURES] or dtype != "BF16":
        raise ValueError("V92 pinned layer-29 projection shape changed")
    return result


def authenticate_sources_v92(
    config: Mapping[str, Any], *, require_implementation_sources: bool = True
) -> dict[str, Any]:
    observed: dict[str, Any] = {}
    for path, expected, implementation in _source_bindings(config):
        if implementation and not require_implementation_sources and expected == "TO_FILL":
            continue
        _require_hash(expected, str(path))
        actual = sha256_file_v85(path)
        if actual != expected:
            raise ValueError(f"V92 pinned source changed: {path}")
        observed[str(path)] = actual
    sources = config["sources"]
    snapshot = (
        Path.home()
        / ".cache/huggingface/hub/models--google--gemma-4-E2B-it/snapshots"
        / str(sources["model_revision"])
    )
    model_config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    text = model_config.get("text_config")
    if (
        not isinstance(text, Mapping)
        or text.get("hidden_size") != 1536
        or text.get("num_hidden_layers") != 35
        or not isinstance(text.get("layer_types"), list)
        or text["layer_types"][29] != "full_attention"
    ):
        raise ValueError("V92 local Gemma layer-29 contract changed")
    observed["gemma_model_blob_sha256_identity"] = sources["model_blob_sha256_identity"]
    observed["pinned_model_tensor"] = authenticate_pinned_model_tensor_v92(config)
    authenticate_parent_v92(config)
    return observed


def derive_contract_v92(config_path: str | Path = CONFIG) -> dict[str, Any]:
    config = load_config_v92(config_path, allow_draft=True)
    canonical = load_canonical_rows_v92(config)
    items = derive_training_items_v92(config, canonical)
    schedule = schedule_v92(
        items,
        seed=int(config["training"]["row_order_seed"]),
        epochs=int(config["dataset"]["epochs"]),
    )
    kinds = Counter(item.kind for item in items)
    return {
        "training_inventory_sha256": canonical_sha256_v85(inventory_v92(items)),
        "training_schedule_sha256": canonical_sha256_v85(
            [[epoch, item.schedule_id] for epoch, item in schedule]
        ),
        "kind_rows_per_epoch": dict(sorted(kinds.items())),
        "new_held_wording_rows": len(held_wording_rows_v92(config)),
        "rows_per_epoch": len(items),
        "epochs": int(config["dataset"]["epochs"]),
        "total_micro_rows": len(schedule),
        "gradient_accumulation_rows": int(config["training"]["gradient_accumulation_rows"]),
        "optimizer_updates": len(schedule) // int(config["training"]["gradient_accumulation_rows"]),
        "primary_causal_rows_per_epoch": sum(item.causal_margin for item in items),
        "total_primary_causal_rows": sum(item.causal_margin for _epoch, item in schedule),
        "lora": lora_preflight_v92(config),
    }


def protocol_v92(
    config: Mapping[str, Any], *, config_path: str | Path = CONFIG,
    require_sealed_hashes: bool = True,
) -> dict[str, Any]:
    contract = derive_contract_v92(config_path)
    expected_kinds = {
        "canonical": 138,
        "conversational_error_replay": 60,
        "conversational_known": 104,
        "correct_anchor_replay": 115,
        "error_replay": 115,
        "primary_failed_intent_replay": 48,
        "primary_success_anchor": 10,
    }
    checks = {
        "kind_counts_exact": contract["kind_rows_per_epoch"] == expected_kinds,
        "new_held_26_excluded": contract["new_held_wording_rows"] == 26,
        "rows_per_epoch_590_exact": contract["rows_per_epoch"] == 590,
        "three_epochs_exact": contract["epochs"] == 3,
        "micro_rows_1770_exact": contract["total_micro_rows"] == 1770,
        "optimizer_updates_295_exact": contract["optimizer_updates"] == 295,
        "primary_causal_13_per_epoch_exact": contract["primary_causal_rows_per_epoch"] == 13,
        "primary_causal_39_total_exact": contract["total_primary_causal_rows"] == 39,
        "oracle_not_loaded_by_trainer": config["dataset"]["oracle_loaded_during_training"] is False,
        "runtime_serializes_no_supervision": config["dataset"]["questions_or_answers_serialized_at_runtime"] is False,
        "fresh_target_disjoint": config["bridge"]["disjoint_from_all_frozen_banks"] is True,
        "failed_v91_not_misrepresented_as_release": config["parent_v91_result"]["runtime_promotion_authorized"] is False,
    }
    if require_sealed_hashes:
        checks.update({
            "inventory_hash_exact": contract["training_inventory_sha256"] == config["dataset"]["training_inventory_sha256"],
            "schedule_hash_exact": contract["training_schedule_sha256"] == config["dataset"]["training_schedule_sha256"],
        })
    if not all(checks.values()):
        raise RuntimeError(f"V92 protocol failed: {checks}")
    return {"checks": checks, **contract}


def build_preregistration_v92(config_path: str | Path = CONFIG) -> dict[str, Any]:
    config = load_config_v92(config_path, allow_draft=False)
    return {
        "artifact": PREREG_ARTIFACT,
        "schema_version": 92,
        "status": _SEALED,
        "config_sha256": sha256_file_v85(config_path),
        "source_sha256": authenticate_sources_v92(config),
        "parent": authenticate_parent_v92(config),
        "protocol": protocol_v92(config, config_path=config_path),
        "strict_input_contract": dict(config["strict_input_contract"]),
        "scope": dict(config["scope"]),
        "model_loaded": False,
        "mps_used": False,
        "oracle_loaded": False,
    }


def _expected_cpu_preflight_v92(
    config: Mapping[str, Any], *, config_path: str | Path,
    preregistration_sha256: str,
) -> dict[str, Any]:
    return {
        "artifact": PREFLIGHT_ARTIFACT,
        "schema_version": 92,
        "status": "cpu_preflight_pass_training_authorized",
        "config_sha256": sha256_file_v85(config_path),
        "preregistration_sha256": preregistration_sha256,
        "protocol": protocol_v92(config, config_path=config_path),
        "memory": memory_preflight_v92(config),
        "parent": authenticate_parent_v92(config),
        "model_loaded": False,
        "mps_used": False,
        "optimizer_constructed": False,
        "oracle_loaded": False,
        "training_authorized": True,
    }


def run_cpu_preflight_v92(config_path: str | Path = CONFIG) -> dict[str, Any]:
    config = load_config_v92(config_path, allow_draft=False)
    prereg = resolve_v85(config["outputs"]["preregistration"])
    expected_prereg = build_preregistration_v92(config_path)
    if prereg.exists():
        if strict_json_v89(prereg) != expected_prereg:
            raise ValueError("V92 preregistration changed")
    else:
        atomic_create_json_v85(prereg, expected_prereg)
    result = _expected_cpu_preflight_v92(
        config, config_path=config_path, preregistration_sha256=sha256_file_v85(prereg)
    )
    output = resolve_v85(config["outputs"]["cpu_preflight"])
    if output.exists():
        if strict_json_v89(output) != result:
            raise ValueError("V92 CPU preflight changed")
    else:
        atomic_create_json_v85(output, result)
    return result


def authenticate_cpu_preflight_v92(
    config: Mapping[str, Any], *, config_path: str | Path = CONFIG
) -> dict[str, str]:
    load_config_v92(config_path, allow_draft=False)
    authenticate_sources_v92(config)
    prereg = resolve_v85(config["outputs"]["preregistration"])
    cpu = resolve_v85(config["outputs"]["cpu_preflight"])
    if strict_json_v89(prereg) != build_preregistration_v92(config_path):
        raise ValueError("V92 preregistration authentication failed")
    expected = _expected_cpu_preflight_v92(
        config, config_path=config_path, preregistration_sha256=sha256_file_v85(prereg)
    )
    if strict_json_v89(cpu) != expected:
        raise ValueError("V92 CPU preflight authentication failed")
    return {
        "config_sha256": sha256_file_v85(config_path),
        "preregistration_sha256": sha256_file_v85(prereg),
        "cpu_preflight_sha256": sha256_file_v85(cpu),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("derive", "preregister", "preflight", "authenticate"))
    parser.add_argument("--config", type=Path, default=CONFIG)
    args = parser.parse_args(argv)
    if args.command == "derive":
        result = derive_contract_v92(args.config)
    elif args.command == "preregister":
        config = load_config_v92(args.config, allow_draft=False)
        path = resolve_v85(config["outputs"]["preregistration"])
        atomic_create_json_v85(path, build_preregistration_v92(args.config))
        result = {"path": str(path), "sha256": sha256_file_v85(path)}
    elif args.command == "preflight":
        result = run_cpu_preflight_v92(args.config)
    else:
        config = load_config_v92(args.config, allow_draft=False)
        result = authenticate_cpu_preflight_v92(config, config_path=args.config)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
