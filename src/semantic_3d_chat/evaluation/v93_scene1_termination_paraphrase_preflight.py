"""Preflight V93's termination/paraphrase repair without loading Gemma.

V93 freezes V92's exact fourteen-bank development stack and trains one
disjoint, exact-zero-output layer-24 bank.  This module authenticates the V92
failure, derives the exact 590-row schedule, isolates 26 new held wordings,
and verifies the immutable continuous memory and fresh bridge on CPU.
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
from semantic_3d_chat.evaluation.v92_scene1_retention_conversation_preflight import (
    authenticate_cpu_preflight_v92,
    held_wording_rows_v92,
    known_wording_rows_v92,
    load_canonical_rows_v92,
    load_config_v92,
    primary_rows_v92,
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
    "configs/experiments/gemma4_v93_scene1_termination_paraphrase_repair.yaml"
)
SCENE_ID: Final[str] = "scene_000001"
FRESH_BANK_NAME: Final[str] = "v93_scene1_termination_paraphrase_repair"
TARGET_MODULE: Final[str] = "model.language_model.layers.24.self_attn.o_proj"
PINNED_MODEL_TENSOR: Final[str] = TARGET_MODULE + ".weight"
TARGET_IN_FEATURES: Final[int] = 4096
TARGET_OUT_FEATURES: Final[int] = 1536
FRESH_PARAMETER_COUNT: Final[int] = 45056
EXPECTED_INITIAL_STATE_SHA256: Final[str] = (
    "23b29db2f3cf2b05e3bc4cd845cc0f85f9c7c3b65bfcd0ab92a2c1e9df6a2e77"
)
V92_STATE_SHA256: Final[str] = "a5544c7256e857d44597118171cffdbfe7349b1293b08d8ed2dbccb5068d57e7"
PREREG_ARTIFACT: Final[str] = "gemma4_v93_scene1_termination_paraphrase_repair_preregistration_v1"
PREFLIGHT_ARTIFACT: Final[str] = "gemma4_v93_scene1_termination_paraphrase_repair_cpu_preflight_v1"
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
PRIMARY_FAILED_INTENTS: Final[tuple[str, ...]] = ("table_contents",)
CONVERSATIONAL_ERROR_IDS: Final[tuple[str, ...]] = (
    "v91_table_contents_existing_00",
    "v92_inventory_new_held_00",
    "v92_inventory_new_held_01",
    "v92_bowl_color_new_held_01",
    "v92_table_contents_new_held_00",
    "v92_closest_new_held_00",
    "v92_closest_new_held_01",
    "v92_cube_location_new_held_00",
    "v92_lamp_turn_new_held_00",
    "v92_frame_support_new_held_00",
)
SUPPORT_ERROR_IDS: Final[tuple[str, ...]] = ("q_000071", "q_000125")
_HEX64: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{64}")
_DRAFT: Final[str] = "draft_before_sealed_preflight"
_SEALED: Final[str] = "sealed_before_full_model_load"


@dataclass(frozen=True)
class TrainingItemV93:
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
        raise ValueError(f"V93 {label} changed")


def _require_hash(value: Any, label: str, *, allow_draft: bool = False) -> None:
    if isinstance(value, str) and _HEX64.fullmatch(value):
        return
    if allow_draft and value == "TO_FILL":
        return
    raise ValueError(f"V93 {label} is not sealed")


def _answer_class(answer: str) -> str:
    normalized = normalize_answer(answer)
    if not normalized:
        raise ValueError("V93 answer normalizes empty")
    return "answer_" + hashlib.sha256(normalized.encode()).hexdigest()[:20]


def _parent_config(config: Mapping[str, Any]) -> dict[str, Any]:
    return load_config_v92(config["sources"]["parent_v92_config"], allow_draft=False)


def _validate_intents(config: Mapping[str, Any]) -> None:
    intents = config.get("conversational_intents")
    if not isinstance(intents, list) or len(intents) != len(INTENT_IDS):
        raise ValueError("V93 requires exactly thirteen intents")
    if tuple(raw.get("id") for raw in intents if isinstance(raw, Mapping)) != INTENT_IDS:
        raise ValueError("V93 intent identity/order changed")
    parent_config = _parent_config(config)
    parent = parent_config["conversational_intents"]
    parent_by_id = {str(raw["id"]): raw for raw in parent}
    parent_rows = (
        *known_wording_rows_v92(parent_config),
        *held_wording_rows_v92(parent_config),
    )
    parent_questions = {normalize_answer(row.question) for row in parent_rows}
    training_questions: list[str] = []
    held_questions: list[str] = []
    for raw in intents:
        if not isinstance(raw, Mapping) or set(raw) != {
            "id",
            "family",
            "answer",
            "training_wordings",
            "new_held_wordings",
        }:
            raise ValueError("V93 intent schema changed")
        source = parent_by_id[str(raw["id"])]
        if raw["family"] != source["family"] or str(raw["answer"]) != str(source["answer"]):
            raise ValueError("V93 intent semantics changed")
        training = raw["training_wordings"]
        held = raw["new_held_wordings"]
        if (
            not isinstance(training, list)
            or len(training) != 6
            or any(not isinstance(value, str) or not value.strip() for value in training)
            or not isinstance(held, list)
            or len(held) != 2
            or any(not isinstance(value, str) or not value.strip() for value in held)
        ):
            raise ValueError("V93 requires six train and two held wordings per intent")
        training_questions.extend(training)
        held_questions.extend(held)
    normalized_training = [normalize_answer(value) for value in training_questions]
    normalized_held = [normalize_answer(value) for value in held_questions]
    if (
        len(parent_questions) != 130
        or len(normalized_training) != 78
        or len(set(normalized_training)) != 78
        or len(normalized_held) != 26
        or len(set(normalized_held)) != 26
        or parent_questions & set(normalized_training)
        or parent_questions & set(normalized_held)
        or set(normalized_training) & set(normalized_held)
    ):
        raise ValueError("V93 paraphrase isolation changed")


def load_config_v93(path: str | Path = CONFIG, *, allow_draft: bool = True) -> dict[str, Any]:
    source = resolve_v85(path)
    if source.is_symlink() or not source.is_file():
        raise FileNotFoundError(source)
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or set(payload) != {"v93"}:
        raise ValueError("V93 config must contain exactly one v93 mapping")
    config = payload["v93"]
    if not isinstance(config, Mapping):
        raise TypeError("V93 config payload must be a mapping")
    _require_exact(config.get("schema_version"), 93, "schema version")
    _require_exact(
        config.get("artifact"),
        "gemma4_v93_scene1_termination_paraphrase_repair_direct_memory_v1",
        "artifact identity",
    )
    _require_exact(config.get("seed"), 930093, "seed")
    _require_exact(config.get("model_id"), "google/gemma-4-E2B-it", "model id")
    _require_exact(
        config.get("revision"),
        "3e22461f65e89153144f8adb70e3b8c2cc9845a7",
        "model revision",
    )
    _require_exact(config.get("dtype"), "bfloat16", "model dtype")
    _require_exact(config.get("max_answer_tokens"), 32, "generation cap")
    expected_prompt = (
        "You answer using only the continuous 3D scene memory supplied before this "
        "conversation. Do not invent objects or relationships that are not supported "
        "by the scene. If the scene does not provide enough evidence, say that it is "
        "unknown. Give only the shortest direct answer."
    )
    _require_exact(config.get("system_prompt"), expected_prompt, "system prompt")
    status = config.get("status")
    if status not in {_DRAFT, _SEALED} or (not allow_draft and status != _SEALED):
        raise ValueError("V93 config has not been sealed")
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
        config.get("parent_v92_result"),
        {
            "model_acceptance_gate_passed": False,
            "runtime_promotion_authorized": False,
            "frozen_parent_state_invariant": True,
            "canonical_correct": 123,
            "canonical_total": 138,
            "canonical_error_count": 15,
            "canonical_type_correct": {
                "presence": 22,
                "count": 9,
                "metric": 1,
                "attribute": 17,
                "spatial_relation": 74,
                "support": 0,
            },
            "primary_correct": 12,
            "primary_total": 13,
            "primary_failed_intents": ["table_contents"],
            "core_actionable_correct": 5,
            "core_actionable_total": 6,
            "held_wording_correct": 17,
            "held_wording_total": 26,
            "held_wording_failure_counts": {
                "inventory": 2,
                "bowl_color": 1,
                "table_contents": 1,
                "closest": 2,
                "cube_location": 1,
                "lamp_turn": 1,
                "frame_support": 1,
            },
            "causal_mean_zero_minus_correct_nll": 1.644125777688784,
            "causal_prediction_changes": 10,
            "candidate_state_sha256": V92_STATE_SHA256,
            "candidate_checkpoint_sha256": (
                "d47ee551c0ec78f4e49a4dc6e8e884b22911d4e43285e5b2e0213d2aad725297"
            ),
        },
        "measured V92 parent result",
    )
    dataset = config.get("dataset")
    if not isinstance(dataset, Mapping):
        raise TypeError("V93 dataset must be a mapping")
    expected_dataset = {
        "scene_id": SCENE_ID,
        "canonical_row_count": 138,
        "canonical_row_inventory_sha256": (
            "9919ff1bee4611dce4132d79fa50f6f6b4ace567a6df780a2e0e21bd88237a8e"
        ),
        "parent_correct_count": 123,
        "parent_error_count": 15,
        "parent_error_extra_copies": 4,
        "parent_correct_anchor_copies": 1,
        "known_conversational_row_count": 130,
        "training_paraphrases_per_intent": 6,
        "training_paraphrase_row_count": 78,
        "parent_conversational_error_count": 10,
        "parent_conversational_error_extra_copies": 5,
        "support_error_row_count": 2,
        "support_error_extra_copies": 5,
        "primary_inventory_anchor_count": 1,
        "new_held_wordings_per_intent": 2,
        "new_held_wording_row_count": 26,
        "rows_per_epoch": 590,
        "epochs": 3,
        "total_micro_rows": 1770,
        "labels_derived_offline_from_oracle": True,
        "oracle_loaded_during_training": False,
        "questions_or_answers_serialized_at_runtime": False,
        "exact_support_error_question_ids": list(SUPPORT_ERROR_IDS),
        "exact_parent_conversational_error_question_ids": list(CONVERSATIONAL_ERROR_IDS),
    }
    if any(dataset.get(key) != value for key, value in expected_dataset.items()):
        raise ValueError("V93 fixed dataset contract changed")
    for key in ("training_inventory_sha256", "training_schedule_sha256"):
        _require_hash(dataset.get(key), key, allow_draft=allow_draft)
    _require_exact(
        config.get("frozen_stack"),
        {
            "v89_parent_bank_count": 11,
            "v89_parent_adapter_parameter_count": 872448,
            "v90_parameter_count": 28672,
            "v91_parameter_count": 221184,
            "v92_bank_name": "v92_scene1_retention_conversation_repair",
            "v92_target_module": "model.language_model.layers.29.self_attn.o_proj",
            "v92_state_sha256": V92_STATE_SHA256,
            "v92_parameter_count": 45056,
            "total_frozen_bank_count": 14,
            "total_frozen_adapter_parameter_count": 1167360,
            "v90_runtime_promotable": False,
            "v91_runtime_promotable": False,
            "v92_runtime_promotable": False,
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
            "initialization_seed": 930093,
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
            "eos_extra_weight": 4.0,
            "zero_payload_margin_weight": 1.0,
            "zero_payload_target_margin_nll": 0.5,
            "row_order_seed": 930093,
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
            "primary_conversational_required_correct": 13,
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
            "preregistration": "reports/gemma4/metrics/gemma4_v93_scene1_termination_paraphrase_repair_preregistration.json",
            "cpu_preflight": "reports/gemma4/metrics/gemma4_v93_scene1_termination_paraphrase_repair_cpu_preflight.json",
            "fixed_final_candidate": "reports/gemma4/artifacts/v93_scene1_termination_paraphrase_repair_final",
            "training_report": "reports/gemma4/metrics/gemma4_v93_scene1_termination_paraphrase_repair_training.json",
            "evaluation_predictions": "reports/gemma4/predictions/gemma4_v93_scene1_termination_paraphrase_repair_evaluation.json",
            "evaluation_report": "reports/gemma4/metrics/gemma4_v93_scene1_termination_paraphrase_repair_evaluation.json",
        },
        "output namespace",
    )
    _require_exact(
        config.get("scope"),
        {
            "post_v92_training_set_development": True,
            "exact_failed_v92_candidate_frozen": True,
            "single_scene_termination_and_paraphrase_repair": True,
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
        raise TypeError("V93 sources must be a mapping")
    for key, value in sources.items():
        if key.endswith("_sha256"):
            _require_hash(
                value,
                key,
                allow_draft=allow_draft
                and key
                in {
                    "preflight_source_sha256",
                    "training_source_sha256",
                    "evaluation_source_sha256",
                },
            )
    _validate_intents(config)
    return dict(config)


def load_canonical_rows_v93(config: Mapping[str, Any]) -> tuple[RowV73, ...]:
    rows = load_canonical_rows_v92(_parent_config(config))
    if len(rows) != 138 or {row.scene_id for row in rows} != {SCENE_ID}:
        raise ValueError("V93 canonical inventory changed")
    return rows


def _intent_from_row(row: RowV73) -> str:
    for prefix in ("v91_conversation_", "v92_conversation_", "v93_conversation_"):
        if row.pair_id.startswith(prefix):
            intent = row.pair_id.removeprefix(prefix)
            if intent in INTENT_IDS:
                return intent
    raise ValueError(f"V93 row has unknown conversational identity: {row.question_id}")


def known_wording_rows_v93(config: Mapping[str, Any]) -> tuple[RowV73, ...]:
    parent = _parent_config(config)
    rows = (*known_wording_rows_v92(parent), *held_wording_rows_v92(parent))
    counts = Counter(_intent_from_row(row) for row in rows)
    if (
        len(rows) != 130
        or len({row.question_id for row in rows}) != 130
        or counts != Counter({intent: 10 for intent in INTENT_IDS})
    ):
        raise RuntimeError("V93 known conversational inventory changed")
    return rows


conversational_rows_v93 = known_wording_rows_v93


def primary_rows_v93(config: Mapping[str, Any]) -> tuple[RowV73, ...]:
    rows = primary_rows_v92(_parent_config(config))
    if (
        len(rows) != 13
        or {_intent_from_row(row) for row in rows} != set(INTENT_IDS)
        or any(not row.question_id.endswith("_existing_00") for row in rows)
    ):
        raise RuntimeError("V93 primary inventory changed")
    return rows


def _wording_row(
    raw: Mapping[str, Any],
    ordinal: int,
    question: str,
    *,
    split: str,
) -> RowV73:
    intent = str(raw["id"])
    answer = str(raw["answer"])
    return RowV73(
        scene_id=SCENE_ID,
        question_id=f"v93_{intent}_{split}_{ordinal:02d}",
        question=question.strip(),
        answer=answer,
        answer_class=_answer_class(answer),
        answer_type=str(raw["family"]),
        pair_id=f"v93_conversation_{intent}",
        paired_scene_id=SCENE_ID,
        question_key=f"v93_conversation_{intent}",
        change_type="wording",
        expected_change=False,
    )


def training_paraphrase_rows_v93(config: Mapping[str, Any]) -> tuple[RowV73, ...]:
    rows = tuple(
        _wording_row(raw, ordinal, str(question), split="training")
        for raw in config["conversational_intents"]
        for ordinal, question in enumerate(raw["training_wordings"])
    )
    if (
        len(rows) != 78
        or len({row.question_id for row in rows}) != 78
        or Counter(_intent_from_row(row) for row in rows)
        != Counter({intent: 6 for intent in INTENT_IDS})
    ):
        raise RuntimeError("V93 training paraphrase inventory changed")
    return rows


training_wording_rows_v93 = training_paraphrase_rows_v93


def held_wording_rows_v93(config: Mapping[str, Any]) -> tuple[RowV73, ...]:
    rows = tuple(
        _wording_row(raw, ordinal, str(question), split="new_held")
        for raw in config["conversational_intents"]
        for ordinal, question in enumerate(raw["new_held_wordings"])
    )
    if (
        len(rows) != 26
        or len({row.question_id for row in rows}) != 26
        or Counter(_intent_from_row(row) for row in rows)
        != Counter({intent: 2 for intent in INTENT_IDS})
    ):
        raise RuntimeError("V93 new held wording inventory changed")
    return rows


new_held_wording_rows_v93 = held_wording_rows_v93


def _clone(row: RowV73, schedule_id: str) -> RowV73:
    return replace(row, question_id=schedule_id, question_key=schedule_id)


def _parent_prediction(config: Mapping[str, Any]) -> dict[str, Any]:
    payload = strict_json_v89(config["sources"]["parent_v92_predictions"])
    if (
        payload.get("artifact") != "gemma4_v92_scene1_retention_conversation_repair_predictions_v1"
        or payload.get("schema_version") != 92
        or payload.get("status") != "fixed_final_evaluation_only_not_runtime"
        or payload.get("scene_id") != SCENE_ID
        or payload.get("canonical_row_count") != 138
        or payload.get("primary_conversational_row_count") != 13
        or payload.get("new_v92_held_wording_row_count") != 26
        or payload.get("frozen_parent_state_invariant") is not True
        or payload.get("candidate_state_invariant") is not True
        or payload.get("runtime_promotion_authorized") is not False
    ):
        raise ValueError("V93 V92 prediction identity changed")
    return payload


def parent_correct_and_errors_v93(
    config: Mapping[str, Any], rows: Sequence[RowV73]
) -> tuple[tuple[RowV73, ...], tuple[RowV73, ...]]:
    records = _parent_prediction(config).get("canonical_records")
    by_id = {row.question_id: row for row in rows}
    if not isinstance(records, list) or len(records) != 138:
        raise ValueError("V93 V92 canonical predictions changed")
    records_by_id = {str(record.get("question_id")): record for record in records}
    if set(records_by_id) != set(by_id):
        raise ValueError("V93 V92 canonical prediction coverage changed")
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
            raise ValueError("V93 V92 canonical binding changed")
        target = (
            correct
            if canonical_type_specific_match(
                row.answer_type, record.get("prediction", ""), row.answer
            )
            else errors
        )
        target.append(row)
    if (
        len(correct) != 123
        or len(errors) != 15
        or not set(SUPPORT_ERROR_IDS).issubset(row.question_id for row in errors)
    ):
        raise ValueError("V93 requires exact V92 123/15 split and support errors")
    return tuple(correct), tuple(errors)


def conversational_errors_v93(config: Mapping[str, Any]) -> tuple[RowV73, ...]:
    payload = _parent_prediction(config)
    records = [
        *payload["primary_conversational_records"],
        *payload["new_v92_held_wording_records"],
    ]
    known = {row.question_id: row for row in known_wording_rows_v93(config)}
    errors: list[RowV73] = []
    for record in records:
        question_id = str(record.get("question_id"))
        row = known.get(question_id)
        if row is None:
            raise ValueError("V93 V92 conversational record has unknown row")
        intent_id = _intent_from_row(row)
        if not conversational_match_v91(
            intent_id, row.answer_type, record.get("prediction", ""), row.answer
        ):
            errors.append(row)
    if tuple(row.question_id for row in errors) != CONVERSATIONAL_ERROR_IDS:
        raise ValueError("V93 exact ten V92 conversational errors changed")
    return tuple(errors)


def derive_training_items_v93(
    config: Mapping[str, Any], canonical: Sequence[RowV73] | None = None
) -> tuple[TrainingItemV93, ...]:
    canonical_rows = tuple(canonical) if canonical is not None else load_canonical_rows_v93(config)
    correct, errors = parent_correct_and_errors_v93(config, canonical_rows)
    known = known_wording_rows_v93(config)
    paraphrases = training_paraphrase_rows_v93(config)
    primary = primary_rows_v93(config)
    items: list[TrainingItemV93] = [
        TrainingItemV93(row.question_id, "canonical", row.question_id, row, False)
        for row in sorted(canonical_rows, key=lambda value: value.question_id)
    ]
    for row in sorted(errors, key=lambda value: value.question_id):
        for copy in range(4):
            schedule_id = f"v93_parent_error_{copy:02d}_{row.question_id}"
            items.append(
                TrainingItemV93(
                    schedule_id,
                    "parent_error_replay",
                    row.question_id,
                    _clone(row, schedule_id),
                    False,
                    copy_ordinal=copy,
                )
            )
    for row in sorted(correct, key=lambda value: value.question_id):
        schedule_id = f"v93_parent_anchor_{row.question_id}"
        items.append(
            TrainingItemV93(
                schedule_id,
                "parent_correct_anchor",
                row.question_id,
                _clone(row, schedule_id),
                False,
            )
        )
    intent_ordinals: Counter[str] = Counter()
    for row in known:
        intent = _intent_from_row(row)
        ordinal = intent_ordinals[intent]
        intent_ordinals[intent] += 1
        schedule_id = f"v93_known_{row.question_id}"
        items.append(
            TrainingItemV93(
                schedule_id,
                "conversational_known",
                row.question_id,
                _clone(row, schedule_id),
                row.question_id.endswith("_existing_00"),
                intent,
                ordinal,
                0,
            )
        )
    for row in paraphrases:
        intent = _intent_from_row(row)
        ordinal = int(row.question_id.rsplit("_", 1)[1])
        schedule_id = f"v93_paraphrase_{row.question_id}"
        items.append(
            TrainingItemV93(
                schedule_id,
                "training_paraphrase",
                row.question_id,
                _clone(row, schedule_id),
                False,
                intent,
                ordinal,
                0,
            )
        )
    for row in conversational_errors_v93(config):
        intent = _intent_from_row(row)
        for copy in range(5):
            schedule_id = f"v93_conversation_error_{copy:02d}_{row.question_id}"
            items.append(
                TrainingItemV93(
                    schedule_id,
                    "conversational_error_replay",
                    row.question_id,
                    _clone(row, schedule_id),
                    False,
                    intent,
                    None,
                    copy,
                )
            )
    canonical_by_id = {row.question_id: row for row in canonical_rows}
    for question_id in SUPPORT_ERROR_IDS:
        row = canonical_by_id[question_id]
        for copy in range(5):
            schedule_id = f"v93_support_error_{copy:02d}_{question_id}"
            items.append(
                TrainingItemV93(
                    schedule_id,
                    "support_error_replay",
                    question_id,
                    _clone(row, schedule_id),
                    False,
                    copy_ordinal=copy,
                )
            )
    inventory = next(row for row in primary if _intent_from_row(row) == "inventory")
    schedule_id = "v93_primary_inventory_anchor"
    items.append(
        TrainingItemV93(
            schedule_id,
            "primary_inventory_anchor",
            inventory.question_id,
            _clone(inventory, schedule_id),
            False,
            "inventory",
            0,
            0,
        )
    )
    counts = Counter(item.kind for item in items)
    held_ids = {row.question_id for row in held_wording_rows_v93(config)}
    expected = {
        "canonical": 138,
        "parent_error_replay": 60,
        "parent_correct_anchor": 123,
        "conversational_known": 130,
        "training_paraphrase": 78,
        "conversational_error_replay": 50,
        "support_error_replay": 10,
        "primary_inventory_anchor": 1,
    }
    if (
        len(items) != 590
        or len({item.schedule_id for item in items}) != 590
        or counts != expected
        or held_ids & {item.source_question_id for item in items}
        or sum(item.causal_margin for item in items) != 13
    ):
        raise RuntimeError(f"V93 fixed 590-row inventory changed: {counts}")
    return tuple(items)


def inventory_v93(items: Sequence[TrainingItemV93]) -> list[dict[str, Any]]:
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


training_inventory_v93 = inventory_v93


def schedule_v93(
    items: Sequence[TrainingItemV93], *, seed: int = 930093, epochs: int = 3
) -> tuple[tuple[int, TrainingItemV93], ...]:
    schedule: list[tuple[int, TrainingItemV93]] = []
    for epoch in range(epochs):
        shuffled = sorted(items, key=lambda item: item.schedule_id)
        random.Random(seed + epoch).shuffle(shuffled)
        schedule.extend((epoch, item) for item in shuffled)
    return tuple(schedule)


training_schedule_v93 = schedule_v93


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
            [nn.Identity() for _ in range(24)]
            + [_SyntheticLayer()]
            + [nn.Identity() for _ in range(10)]
        )


class _SyntheticGemma(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = _SyntheticLanguage()


def derive_lora_preflight_v93(config: Mapping[str, Any]) -> dict[str, Any]:
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
        raise RuntimeError("V93 synthetic LoRA installation failed")
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
            int(torch.count_nonzero(adapter.lora_b).item()) for adapter in installation.adapters
        ),
        "exact_zero_output_at_initialization": True,
        "full_gemma_model_loaded": False,
        "device": "cpu",
    }


def lora_preflight_v93(config: Mapping[str, Any]) -> dict[str, Any]:
    result = derive_lora_preflight_v93(config)
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
        raise RuntimeError("V93 deterministic LoRA initialization changed")
    return result


def memory_preflight_v93(config: Mapping[str, Any]) -> dict[str, Any]:
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
        raise RuntimeError("V93 immutable memory preflight changed")
    return result


def authenticate_parent_v93(config: Mapping[str, Any]) -> dict[str, Any]:
    """Authenticate V92's full failed candidate and measurement, model-free."""

    sources = config["sources"]
    parent_config = _parent_config(config)
    parent_preflight = authenticate_cpu_preflight_v92(
        parent_config, config_path=sources["parent_v92_config"]
    )
    if parent_preflight != {
        "config_sha256": sources["parent_v92_config_sha256"],
        "preregistration_sha256": sources["parent_v92_preregistration_sha256"],
        "cpu_preflight_sha256": sources["parent_v92_cpu_preflight_sha256"],
    }:
        raise ValueError("V93 V92 CPU preflight binding changed")
    root = resolve_v85(sources["parent_v92_candidate"])
    metadata = strict_json_v89(root / "runtime_metadata.json")
    tensors = load_file(str(root / "bridge.safetensors"), device="cpu")
    if (
        metadata.get("artifact") != "gemma4_v92_scene1_retention_conversation_repair_fixed_final_v1"
        or metadata.get("schema_version") != 92
        or metadata.get("bank_name") != "v92_scene1_retention_conversation_repair"
        or metadata.get("target_module") != "model.language_model.layers.29.self_attn.o_proj"
        or metadata.get("rank") != 8
        or float(metadata.get("alpha", -1)) != 16.0
        or metadata.get("parameter_count") != 45056
        or metadata.get("state_sha256") != V92_STATE_SHA256
        or metadata.get("weights_sha256") != sources["parent_v92_bridge_sha256"]
        or metadata.get("total_bank_count") != 14
        or metadata.get("total_adapter_parameter_count") != 1167360
        or metadata.get("runtime_promotion_authorized") is not False
        or metadata.get("questions_or_answers_serialized") is not False
        or metadata.get("training_inventory_serialized") is not False
        or metadata.get("oracle_serialized") is not False
        or set(tensors) != {"lora_a", "lora_b"}
        or list(tensors["lora_a"].shape) != [8, 4096]
        or list(tensors["lora_b"].shape) != [1536, 8]
        or tensor_state_sha256({f"adapters.0.{key}": value for key, value in tensors.items()})
        != V92_STATE_SHA256
    ):
        raise ValueError("V93 exact failed V92 bridge authentication failed")
    training = strict_json_v89(sources["parent_v92_training"])
    evaluation = strict_json_v89(sources["parent_v92_evaluation"])
    metrics = evaluation.get("metrics")
    canonical = (
        metrics.get("canonical_strict_normalized_exact") if isinstance(metrics, Mapping) else None
    )
    by_type = (
        metrics.get("canonical_accuracy_by_answer_type") if isinstance(metrics, Mapping) else None
    )
    primary = metrics.get("primary_conversational") if isinstance(metrics, Mapping) else None
    held = metrics.get("new_held_wording") if isinstance(metrics, Mapping) else None
    causal = metrics.get("causal_control") if isinstance(metrics, Mapping) else None
    if (
        training.get("artifact") != "gemma4_v92_scene1_retention_conversation_repair_training_v1"
        or training.get("status") != "fixed_final_training_complete_not_promoted"
        or training.get("config_sha256") != sources["parent_v92_config_sha256"]
        or training.get("preregistration_sha256") != sources["parent_v92_preregistration_sha256"]
        or training.get("cpu_preflight_sha256") != sources["parent_v92_cpu_preflight_sha256"]
        or training.get("optimizer_updates") != 295
        or training.get("micro_rows_consumed") != 1770
        or training.get("causal_margin_rows_consumed") != 39
        or training.get("oracle_loaded") is not False
        or training.get("protected_read_count") != 0
        or training.get("runtime_promotion_authorized") is not False
        or evaluation.get("artifact")
        != "gemma4_v92_scene1_retention_conversation_repair_evaluation_v1"
        or evaluation.get("status") != "model_gates_fail_not_runtime_promotable"
        or evaluation.get("evaluation_predictions_sha256")
        != sources["parent_v92_predictions_sha256"]
        or evaluation.get("training_report_sha256") != sources["parent_v92_training_sha256"]
        or evaluation.get("runtime_promotion_authorized") is not False
        or evaluation.get("oracle_loaded") is not False
        or not isinstance(metrics, Mapping)
        or metrics.get("model_acceptance_gate_passed") is not False
        or not isinstance(canonical, Mapping)
        or canonical.get("correct") != 123
        or canonical.get("total") != 138
        or not isinstance(by_type, Mapping)
        or {key: by_type[key]["correct"] for key in by_type}
        != {
            "attribute": 17,
            "count": 9,
            "metric": 1,
            "presence": 22,
            "spatial_relation": 74,
            "support": 0,
        }
        or not isinstance(primary, Mapping)
        or primary.get("correct") != 12
        or primary.get("total") != 13
        or primary.get("core_actionable_correct") != 5
        or not isinstance(held, Mapping)
        or held.get("correct") != 17
        or held.get("total") != 26
        or not isinstance(causal, Mapping)
        or causal.get("mean_zero_minus_correct_nll") != 1.644125777688784
        or causal.get("canonical_prediction_changes") != 10
    ):
        raise ValueError("V93 measured V92 evidence authentication failed")
    canonical_rows = load_canonical_rows_v93(config)
    _correct, errors = parent_correct_and_errors_v93(config, canonical_rows)
    conversation_errors = conversational_errors_v93(config)
    return {
        "v89_frozen_bank_count": 11,
        "v90_frozen_bank_count": 1,
        "v91_frozen_bank_count": 1,
        "v92_frozen_bank_count": 1,
        "total_frozen_bank_count": 14,
        "total_frozen_parameter_count": 1167360,
        "v92_state_sha256": V92_STATE_SHA256,
        "v92_model_acceptance_gate_passed": False,
        "v92_runtime_promotion_authorized": False,
        "canonical_correct": 123,
        "canonical_errors": len(errors),
        "primary_correct": 12,
        "semantic_primary_failed_intents": ["table_contents"],
        "held_wording_correct": 17,
        "conversational_error_question_ids": [row.question_id for row in conversation_errors],
        "support_error_question_ids": list(SUPPORT_ERROR_IDS),
        "new_repair_is_post_failure_development": True,
    }


def _source_bindings(config: Mapping[str, Any]) -> tuple[tuple[str, str, bool], ...]:
    sources = config["sources"]
    return (
        (sources["runtime_config"], sources["runtime_config_sha256"], False),
        (sources["scene1_qa"], sources["scene1_qa_sha256"], False),
        (
            str(Path(sources["scene1_memory"]) / "memory.safetensors"),
            sources["scene1_memory_tensor_sha256"],
            False,
        ),
        (
            str(Path(sources["scene1_memory"]) / "runtime_metadata.json"),
            sources["scene1_memory_metadata_sha256"],
            False,
        ),
        (
            str(Path(sources["parent_v89_checkpoint"]) / "adapter.safetensors"),
            sources["parent_v89_adapter_sha256"],
            False,
        ),
        (
            str(Path(sources["parent_v89_checkpoint"]) / "runtime_metadata.json"),
            sources["parent_v89_metadata_sha256"],
            False,
        ),
        (sources["parent_v92_config"], sources["parent_v92_config_sha256"], False),
        (
            sources["parent_v92_preflight_source"],
            sources["parent_v92_preflight_source_sha256"],
            False,
        ),
        (
            sources["parent_v92_training_source"],
            sources["parent_v92_training_source_sha256"],
            False,
        ),
        (
            sources["parent_v92_evaluation_source"],
            sources["parent_v92_evaluation_source_sha256"],
            False,
        ),
        (
            sources["parent_v92_preregistration"],
            sources["parent_v92_preregistration_sha256"],
            False,
        ),
        (
            sources["parent_v92_cpu_preflight"],
            sources["parent_v92_cpu_preflight_sha256"],
            False,
        ),
        (
            sources["parent_v92_training"],
            sources["parent_v92_training_sha256"],
            False,
        ),
        (
            str(Path(sources["parent_v92_candidate"]) / "bridge.safetensors"),
            sources["parent_v92_bridge_sha256"],
            False,
        ),
        (
            str(Path(sources["parent_v92_candidate"]) / "runtime_metadata.json"),
            sources["parent_v92_metadata_sha256"],
            False,
        ),
        (
            sources["parent_v92_predictions"],
            sources["parent_v92_predictions_sha256"],
            False,
        ),
        (
            sources["parent_v92_evaluation"],
            sources["parent_v92_evaluation_sha256"],
            False,
        ),
        (sources["preflight_source"], sources["preflight_source_sha256"], True),
        (sources["training_source"], sources["training_source_sha256"], True),
        (sources["evaluation_source"], sources["evaluation_source_sha256"], True),
    )


def authenticate_pinned_model_tensor_v93(config: Mapping[str, Any]) -> dict[str, Any]:
    sources = config["sources"]
    snapshot = (
        Path.home()
        / ".cache/huggingface/hub/models--google--gemma-4-E2B-it/snapshots"
        / str(sources["model_revision"])
    )
    model_blob = (snapshot / "model.safetensors").resolve(strict=True)
    if model_blob.name != sources["model_blob_sha256_identity"]:
        raise ValueError("V93 pinned Gemma blob identity changed")
    with safe_open(str(model_blob), framework="pt", device="cpu") as handle:
        if PINNED_MODEL_TENSOR not in handle.keys():  # noqa: SIM118
            raise ValueError("V93 pinned layer-24 tensor absent")
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
        raise ValueError("V93 pinned layer-24 projection shape changed")
    return result


def authenticate_sources_v93(
    config: Mapping[str, Any], *, require_implementation_sources: bool = True
) -> dict[str, Any]:
    observed: dict[str, Any] = {}
    for path, expected, implementation in _source_bindings(config):
        if implementation and not require_implementation_sources and expected == "TO_FILL":
            continue
        _require_hash(expected, str(path))
        actual = sha256_file_v85(path)
        if actual != expected:
            raise ValueError(f"V93 pinned source changed: {path}")
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
        or text["layer_types"][24] != "full_attention"
    ):
        raise ValueError("V93 local Gemma layer-24 contract changed")
    observed["gemma_model_blob_sha256_identity"] = sources["model_blob_sha256_identity"]
    observed["pinned_model_tensor"] = authenticate_pinned_model_tensor_v93(config)
    authenticate_parent_v93(config)
    return observed


def derive_contract_v93(config_path: str | Path = CONFIG) -> dict[str, Any]:
    config = load_config_v93(config_path, allow_draft=True)
    canonical = load_canonical_rows_v93(config)
    items = derive_training_items_v93(config, canonical)
    schedule = schedule_v93(
        items,
        seed=int(config["training"]["row_order_seed"]),
        epochs=int(config["dataset"]["epochs"]),
    )
    kinds = Counter(item.kind for item in items)
    return {
        "training_inventory_sha256": canonical_sha256_v85(inventory_v93(items)),
        "training_schedule_sha256": canonical_sha256_v85(
            [[epoch, item.schedule_id] for epoch, item in schedule]
        ),
        "kind_rows_per_epoch": dict(sorted(kinds.items())),
        "new_held_wording_rows": len(held_wording_rows_v93(config)),
        "training_paraphrase_rows": len(training_paraphrase_rows_v93(config)),
        "rows_per_epoch": len(items),
        "epochs": int(config["dataset"]["epochs"]),
        "total_micro_rows": len(schedule),
        "gradient_accumulation_rows": int(config["training"]["gradient_accumulation_rows"]),
        "optimizer_updates": len(schedule) // int(config["training"]["gradient_accumulation_rows"]),
        "primary_causal_rows_per_epoch": sum(item.causal_margin for item in items),
        "total_primary_causal_rows": sum(item.causal_margin for _epoch, item in schedule),
        "eos_supervised_rows": len(schedule),
        "eos_extra_weight": float(config["training"]["eos_extra_weight"]),
        "system_prompt_sha256": canonical_sha256_v85(config["system_prompt"]),
        "max_answer_tokens": int(config["max_answer_tokens"]),
        "lora": lora_preflight_v93(config),
    }


def protocol_v93(
    config: Mapping[str, Any],
    *,
    config_path: str | Path = CONFIG,
    require_sealed_hashes: bool = True,
) -> dict[str, Any]:
    contract = derive_contract_v93(config_path)
    expected_kinds = {
        "canonical": 138,
        "parent_error_replay": 60,
        "parent_correct_anchor": 123,
        "conversational_known": 130,
        "training_paraphrase": 78,
        "conversational_error_replay": 50,
        "support_error_replay": 10,
        "primary_inventory_anchor": 1,
    }
    checks = {
        "kind_counts_exact": contract["kind_rows_per_epoch"] == expected_kinds,
        "new_held_26_excluded": contract["new_held_wording_rows"] == 26,
        "new_training_paraphrases_78_included": contract["training_paraphrase_rows"] == 78,
        "rows_per_epoch_590_exact": contract["rows_per_epoch"] == 590,
        "three_epochs_exact": contract["epochs"] == 3,
        "micro_rows_1770_exact": contract["total_micro_rows"] == 1770,
        "optimizer_updates_295_exact": contract["optimizer_updates"] == 295,
        "primary_causal_13_per_epoch_exact": contract["primary_causal_rows_per_epoch"] == 13,
        "primary_causal_39_total_exact": contract["total_primary_causal_rows"] == 39,
        "all_1770_rows_receive_eos_supervision": contract["eos_supervised_rows"] == 1770,
        "eos_extra_weight_is_four": contract["eos_extra_weight"] == 4.0,
        "termination_prompt_bound": (
            contract["system_prompt_sha256"] == canonical_sha256_v85(config["system_prompt"])
            and contract["max_answer_tokens"] == 32
        ),
        "oracle_not_loaded_by_trainer": config["dataset"]["oracle_loaded_during_training"] is False,
        "runtime_serializes_no_supervision": config["dataset"][
            "questions_or_answers_serialized_at_runtime"
        ]
        is False,
        "fresh_target_disjoint": config["bridge"]["disjoint_from_all_frozen_banks"] is True,
        "failed_v92_not_misrepresented_as_release": config["parent_v92_result"][
            "runtime_promotion_authorized"
        ]
        is False,
    }
    if require_sealed_hashes:
        checks.update(
            {
                "inventory_hash_exact": contract["training_inventory_sha256"]
                == config["dataset"]["training_inventory_sha256"],
                "schedule_hash_exact": contract["training_schedule_sha256"]
                == config["dataset"]["training_schedule_sha256"],
            }
        )
    if not all(checks.values()):
        raise RuntimeError(f"V93 protocol failed: {checks}")
    return {"checks": checks, **contract}


def build_preregistration_v93(config_path: str | Path = CONFIG) -> dict[str, Any]:
    config = load_config_v93(config_path, allow_draft=False)
    return {
        "artifact": PREREG_ARTIFACT,
        "schema_version": 93,
        "status": _SEALED,
        "config_sha256": sha256_file_v85(config_path),
        "source_sha256": authenticate_sources_v93(config),
        "parent": authenticate_parent_v93(config),
        "protocol": protocol_v93(config, config_path=config_path),
        "strict_input_contract": dict(config["strict_input_contract"]),
        "scope": dict(config["scope"]),
        "model_loaded": False,
        "mps_used": False,
        "oracle_loaded": False,
    }


def _expected_cpu_preflight_v93(
    config: Mapping[str, Any],
    *,
    config_path: str | Path,
    preregistration_sha256: str,
) -> dict[str, Any]:
    return {
        "artifact": PREFLIGHT_ARTIFACT,
        "schema_version": 93,
        "status": "cpu_preflight_pass_training_authorized",
        "config_sha256": sha256_file_v85(config_path),
        "preregistration_sha256": preregistration_sha256,
        "protocol": protocol_v93(config, config_path=config_path),
        "memory": memory_preflight_v93(config),
        "parent": authenticate_parent_v93(config),
        "model_loaded": False,
        "mps_used": False,
        "optimizer_constructed": False,
        "oracle_loaded": False,
        "training_authorized": True,
    }


def run_cpu_preflight_v93(config_path: str | Path = CONFIG) -> dict[str, Any]:
    config = load_config_v93(config_path, allow_draft=False)
    prereg = resolve_v85(config["outputs"]["preregistration"])
    expected_prereg = build_preregistration_v93(config_path)
    if prereg.exists():
        if strict_json_v89(prereg) != expected_prereg:
            raise ValueError("V93 preregistration changed")
    else:
        atomic_create_json_v85(prereg, expected_prereg)
    result = _expected_cpu_preflight_v93(
        config, config_path=config_path, preregistration_sha256=sha256_file_v85(prereg)
    )
    output = resolve_v85(config["outputs"]["cpu_preflight"])
    if output.exists():
        if strict_json_v89(output) != result:
            raise ValueError("V93 CPU preflight changed")
    else:
        atomic_create_json_v85(output, result)
    return result


def authenticate_cpu_preflight_v93(
    config: Mapping[str, Any], *, config_path: str | Path = CONFIG
) -> dict[str, str]:
    load_config_v93(config_path, allow_draft=False)
    authenticate_sources_v93(config)
    prereg = resolve_v85(config["outputs"]["preregistration"])
    cpu = resolve_v85(config["outputs"]["cpu_preflight"])
    if strict_json_v89(prereg) != build_preregistration_v93(config_path):
        raise ValueError("V93 preregistration authentication failed")
    expected = _expected_cpu_preflight_v93(
        config, config_path=config_path, preregistration_sha256=sha256_file_v85(prereg)
    )
    if strict_json_v89(cpu) != expected:
        raise ValueError("V93 CPU preflight authentication failed")
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
        result = derive_contract_v93(args.config)
    elif args.command == "preregister":
        config = load_config_v93(args.config, allow_draft=False)
        path = resolve_v85(config["outputs"]["preregistration"])
        atomic_create_json_v85(path, build_preregistration_v93(args.config))
        result = {"path": str(path), "sha256": sha256_file_v85(path)}
    elif args.command == "preflight":
        result = run_cpu_preflight_v93(args.config)
    else:
        config = load_config_v93(args.config, allow_draft=False)
        result = authenticate_cpu_preflight_v93(config, config_path=args.config)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
