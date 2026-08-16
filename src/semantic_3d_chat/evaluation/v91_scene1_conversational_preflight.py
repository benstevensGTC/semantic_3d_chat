"""Seal and CPU-preflight V91's evidence-driven conversational repair.

V91 freezes the released V89 eleven-bank stack and the exact failed V90 bridge
as a twelve-bank offline parent.  It learns one disjoint, exact-zero-output
rank-16 repair bank.  All six V90 wordings per intent become training data;
only two newly preregistered wordings per intent are held from optimization.

This module is model-free.  It authenticates fixed local files, derives the
590-row schedule, validates immutable continuous memory on CPU, and proves the
fresh adapter's deterministic initialization without loading Gemma weights.
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
    load_canonical_rows_v89,
    strict_json_v89,
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
    "configs/experiments/gemma4_v91_scene1_conversational_repair.yaml"
)
SCENE_ID: Final[str] = "scene_000001"
FRESH_BANK_NAME: Final[str] = "v91_scene1_conversational_repair"
TARGET_MODULE: Final[str] = "model.language_model.layers.33.mlp.down_proj"
PREREG_ARTIFACT: Final[str] = (
    "gemma4_v91_scene1_conversational_repair_preregistration_v2"
)
PREFLIGHT_ARTIFACT: Final[str] = (
    "gemma4_v91_scene1_conversational_repair_cpu_preflight_v2"
)
EXPECTED_INITIAL_STATE_SHA256: Final[str] = (
    "0f255efb26255dcac0815511e44aabad5e21820f78f9a7662dc1bf59f627db2b"
)
PINNED_MODEL_TENSOR: Final[str] = (
    "model.language_model.layers.33.mlp.down_proj.weight"
)
TARGET_IN_FEATURES: Final[int] = 12_288
TARGET_OUT_FEATURES: Final[int] = 1_536
FRESH_PARAMETER_COUNT: Final[int] = 221_184
V90_STATE_SHA256: Final[str] = (
    "70e236711d8ac1fe7cf808f6f4e939b29db476016c8ef49db143707df0f3bde7"
)
FAILED_INTENTS: Final[tuple[str, ...]] = (
    "inventory",
    "under_table",
    "wall_object",
    "cube_location",
    "sitting",
    "bowl_contents",
)
SUCCESSFUL_INTENTS: Final[tuple[str, ...]] = (
    "chair_presence",
    "bowl_color",
    "bowl_left_chair",
    "table_contents",
    "closest",
    "lamp_turn",
    "frame_support",
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
_HEX64: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{64}")
_DRAFT: Final[str] = "draft_before_sealed_preflight"
_SEALED: Final[str] = "sealed_before_full_model_load"


@dataclass(frozen=True)
class TrainingItemV91:
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
        raise ValueError(f"V91 {label} changed")


def _require_hash_or_draft(value: Any, label: str, *, allow_draft: bool) -> None:
    if isinstance(value, str) and _HEX64.fullmatch(value) is not None:
        return
    if allow_draft and value == "TO_FILL":
        return
    raise ValueError(f"V91 {label} is not sealed")


def _answer_class(answer: str) -> str:
    normalized = normalize_answer(answer)
    if not normalized:
        raise ValueError("V91 answer normalizes empty")
    return "answer_" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:20]


def _validate_intents(config: Mapping[str, Any]) -> None:
    intents = config.get("conversational_intents")
    if not isinstance(intents, list) or len(intents) != 13:
        raise ValueError("V91 requires exactly 13 conversational intents")
    if tuple(raw.get("id") for raw in intents if isinstance(raw, Mapping)) != INTENT_IDS:
        raise ValueError("V91 conversational intent identity/order changed")
    all_questions: list[str] = []
    for raw in intents:
        if not isinstance(raw, Mapping) or set(raw) != {
            "id",
            "family",
            "answer",
            "existing_wordings",
            "new_held_wordings",
        }:
            raise ValueError("V91 conversational intent schema changed")
        existing = raw["existing_wordings"]
        held = raw["new_held_wordings"]
        if (
            not isinstance(existing, list)
            or len(existing) != 6
            or not isinstance(held, list)
            or len(held) != 2
            or any(
                not isinstance(value, str) or not value.strip()
                for value in existing + held
            )
            or not str(raw["family"]).strip()
            or not str(raw["answer"]).strip()
        ):
            raise ValueError("V91 requires six train and two new held wordings per intent")
        all_questions.extend(existing + held)
    normalized = [normalize_answer(value) for value in all_questions]
    if len(normalized) != 104 or len(set(normalized)) != 104:
        raise ValueError("V91 existing/new-held wordings overlap")


def load_config_v91(
    path: str | Path = CONFIG, *, allow_draft: bool = True
) -> dict[str, Any]:
    source = resolve_v85(path)
    if source.is_symlink() or not source.is_file():
        raise FileNotFoundError(source)
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or set(payload) != {"v91"}:
        raise ValueError("V91 config must contain exactly one v91 mapping")
    config = payload["v91"]
    if not isinstance(config, Mapping):
        raise TypeError("V91 config payload must be a mapping")
    _require_exact(config.get("schema_version"), 91, "schema version")
    _require_exact(
        config.get("artifact"),
        "gemma4_v91_scene1_conversational_repair_direct_memory_v1",
        "artifact identity",
    )
    _require_exact(config.get("seed"), 910091, "seed")
    status = config.get("status")
    if status not in {_DRAFT, _SEALED} or (not allow_draft and status != _SEALED):
        raise ValueError("V91 config has not been sealed")
    _require_exact(
        config.get("topology_amendment"),
        {
            "amendment_id": "v91_layer33_down_proj_shape_correction_v2",
            "amendment_version": 2,
            "reason": (
                "gemma4_swiglu_down_proj_consumes_concatenated_gate_and_up_activations"
            ),
            "superseded_config_sha256": (
                "b5c95ec12fd0040731417700936be94b865abcbfbf16f157be0aedf7d4e76e09"
            ),
            "superseded_preregistration": {
                "path": (
                    "reports/gemma4/metrics/"
                    "gemma4_v91_scene1_conversational_repair_preregistration.json"
                ),
                "sha256": (
                    "9dfbaac24f2c0132cd0189e07ff2ace8e3aada2282f10b1ca5cf4aa027774c3e"
                ),
            },
            "superseded_cpu_preflight": {
                "path": (
                    "reports/gemma4/metrics/"
                    "gemma4_v91_scene1_conversational_repair_cpu_preflight.json"
                ),
                "sha256": (
                    "f341347ea312d1a075647dec3847073ec71f5dc50486ccae02db96e5db6b8711"
                ),
            },
            "superseded_artifacts_preserved": True,
            "pinned_tensor_name": PINNED_MODEL_TENSOR,
            "superseded_synthetic_weight_shape": [1_536, 6_144],
            "pinned_weight_shape": [TARGET_OUT_FEATURES, TARGET_IN_FEATURES],
            "pinned_weight_dtype": "BF16",
            "corrected_trainable_parameter_count": FRESH_PARAMETER_COUNT,
            "corrected_initial_state_sha256": EXPECTED_INITIAL_STATE_SHA256,
            "tensor_header_authenticated_without_full_model_load": True,
            "replacement_output_namespace": "v2",
        },
        "layer-33 topology amendment",
    )
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
        "strict continuous-memory contract",
    )
    _require_exact(
        config.get("parent_v90_result"),
        {
            "model_acceptance_gate_passed": False,
            "runtime_promotion_authorized": False,
            "frozen_parent_state_invariant": True,
            "canonical_correct": 124,
            "canonical_total": 138,
            "canonical_error_count": 14,
            "canonical_type_correct": {
                "presence": 22,
                "count": 9,
                "metric": 1,
                "attribute": 16,
                "spatial_relation": 75,
                "support": 1,
            },
            "primary_correct": 7,
            "primary_total": 13,
            "held_wording_correct": 10,
            "held_wording_total": 26,
            "causal_mean_zero_minus_correct_nll": 0.8939050505152688,
            "causal_prediction_changes": 10,
            "failed_intents": list(FAILED_INTENTS),
            "successful_intents": list(SUCCESSFUL_INTENTS),
            "candidate_state_sha256": V90_STATE_SHA256,
            "candidate_checkpoint_sha256": (
                "351f42673372008b8312e7b56f7d19a36eb1cfbc9ef184025e9bcdac47cdfcd4"
            ),
        },
        "measured V90 parent result",
    )
    dataset = config.get("dataset")
    if not isinstance(dataset, Mapping):
        raise TypeError("V91 dataset must be a mapping")
    expected_dataset = {
        "scene_id": SCENE_ID,
        "canonical_row_count": 138,
        "canonical_row_inventory_sha256": (
            "9919ff1bee4611dce4132d79fa50f6f6b4ace567a6df780a2e0e21bd88237a8e"
        ),
        "parent_correct_count": 124,
        "parent_error_count": 14,
        "parent_error_replay_copies": 2,
        "parent_correct_anchor_replay_copies": 1,
        "conversational_intent_count": 13,
        "existing_wordings_per_intent": 6,
        "newly_held_wordings_per_intent": 2,
        "newly_held_wording_row_count": 26,
        "successful_intent_count": 7,
        "successful_wording_copies": 2,
        "successful_conversational_rows_per_epoch": 84,
        "failed_intent_count": 6,
        "failed_wording_copies": 6,
        "failed_conversational_rows_per_epoch": 216,
        "rows_per_epoch": 590,
        "epochs": 3,
        "total_micro_rows": 1770,
        "labels_derived_offline_from_oracle": True,
        "oracle_loaded_during_training": False,
        "questions_or_answers_serialized_at_runtime": False,
    }
    if any(dataset.get(key) != value for key, value in expected_dataset.items()):
        raise ValueError("V91 fixed repair dataset contract changed")
    for key in ("training_inventory_sha256", "training_schedule_sha256"):
        _require_hash_or_draft(dataset.get(key), key, allow_draft=allow_draft)
    _validate_intents(config)
    _require_exact(
        config.get("frozen_stack"),
        {
            "v89_parent_bank_count": 11,
            "v89_parent_adapter_parameter_count": 872448,
            "v90_bank_name": "v90_scene1_conversational_bridge",
            "v90_target_module": "model.language_model.layers.28.self_attn.o_proj",
            "v90_state_sha256": V90_STATE_SHA256,
            "v90_parameter_count": 28672,
            "total_frozen_bank_count": 12,
            "total_frozen_adapter_parameter_count": 901120,
            "v90_runtime_promotable": False,
            "exact_failed_candidate_used_as_offline_parent": True,
            "merged_weights": False,
        },
        "frozen V89 plus failed-V90 stack",
    )
    _require_exact(
        config.get("bridge"),
        {
            "bank_name": FRESH_BANK_NAME,
            "target_module": TARGET_MODULE,
            "target_layer_type": "mlp",
            "target_in_features": TARGET_IN_FEATURES,
            "target_out_features": TARGET_OUT_FEATURES,
            "rank": 16,
            "alpha": 32.0,
            "dropout": 0.0,
            "trainable_parameter_count": FRESH_PARAMETER_COUNT,
            "initialization_algorithm": "cpu_kaiming_uniform_a_exact_zero_b",
            "initialization_seed": 910091,
            "expected_initial_state_sha256": EXPECTED_INITIAL_STATE_SHA256,
            "disjoint_from_all_frozen_banks": True,
        },
        "fresh repair bridge",
    )
    _require_exact(
        config.get("training"),
        {
            "optimizer": "AdamW",
            "learning_rate": 0.00025,
            "weight_decay": 0.0,
            "gradient_accumulation_rows": 6,
            "optimizer_updates": 295,
            "gradient_clip_norm": 1.0,
            "answer_ce_weight": 1.0,
            "zero_payload_margin_weight": 1.0,
            "zero_payload_target_margin_nll": 0.5,
            "row_order_seed": 910091,
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
            "preregistration": (
                "reports/gemma4/metrics/"
                "gemma4_v91_scene1_conversational_repair_preregistration_v2.json"
            ),
            "cpu_preflight": (
                "reports/gemma4/metrics/"
                "gemma4_v91_scene1_conversational_repair_cpu_preflight_v2.json"
            ),
            "fixed_final_candidate": (
                "reports/gemma4/artifacts/"
                "v91_scene1_conversational_repair_final_v2"
            ),
            "training_report": (
                "reports/gemma4/metrics/"
                "gemma4_v91_scene1_conversational_repair_training_v2.json"
            ),
            "evaluation_predictions": (
                "reports/gemma4/predictions/"
                "gemma4_v91_scene1_conversational_repair_evaluation_v2.json"
            ),
            "evaluation_report": (
                "reports/gemma4/metrics/"
                "gemma4_v91_scene1_conversational_repair_evaluation_v2.json"
            ),
        },
        "v2 create-once output namespace",
    )
    _require_exact(
        config.get("scope"),
        {
            "post_v90_training_set_development": True,
            "exact_failed_v90_candidate_frozen": True,
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
        "development-only scope",
    )
    sources = config.get("sources")
    if not isinstance(sources, Mapping):
        raise TypeError("V91 sources must be a mapping")
    for key in (
        "runtime_config_sha256",
        "scene1_qa_sha256",
        "scene1_memory_tensor_sha256",
        "scene1_memory_metadata_sha256",
        "parent_v89_adapter_sha256",
        "parent_v89_metadata_sha256",
        "parent_v90_config_sha256",
        "parent_v90_preregistration_sha256",
        "parent_v90_cpu_preflight_sha256",
        "parent_v90_training_sha256",
        "parent_v90_bridge_sha256",
        "parent_v90_metadata_sha256",
        "parent_v90_predictions_sha256",
        "parent_v90_evaluation_sha256",
    ):
        _require_hash_or_draft(sources.get(key), key, allow_draft=False)
    for key in (
        "preflight_source_sha256",
        "training_source_sha256",
        "evaluation_source_sha256",
    ):
        _require_hash_or_draft(sources.get(key), key, allow_draft=allow_draft)
    return dict(config)


def load_canonical_rows_v91(config: Mapping[str, Any]) -> tuple[RowV73, ...]:
    rows = load_canonical_rows_v89(config)
    if len(rows) != 138 or {row.scene_id for row in rows} != {SCENE_ID}:
        raise ValueError("V91 canonical scene-one inventory changed")
    return rows


def authenticate_wording_lineage_v91(config: Mapping[str, Any]) -> dict[str, Any]:
    """Prove all six train wordings are exact V90 rows and new holds are disjoint."""

    parent_path = resolve_v85(config["sources"]["parent_v90_config"])
    payload = yaml.safe_load(parent_path.read_text(encoding="utf-8"))
    parent = payload.get("v90") if isinstance(payload, Mapping) else None
    parent_intents = parent.get("conversational_intents") if isinstance(parent, Mapping) else None
    if not isinstance(parent_intents, list) or len(parent_intents) != 13:
        raise ValueError("V91 pinned V90 wording source is malformed")
    parent_by_id = {
        str(raw.get("id")): raw for raw in parent_intents if isinstance(raw, Mapping)
    }
    new_held: list[str] = []
    existing: list[str] = []
    for raw in config["conversational_intents"]:
        intent_id = str(raw["id"])
        source = parent_by_id.get(intent_id)
        if not isinstance(source, Mapping):
            raise TypeError("V91 V90 intent coverage changed")
        expected = [
            str(source["primary"]),
            *[str(value) for value in source["train"]],
            *[str(value) for value in source["held_wording"]],
        ]
        observed = [str(value) for value in raw["existing_wordings"]]
        if observed != expected:
            raise ValueError(f"V91 existing wording lineage changed: {intent_id}")
        existing.extend(observed)
        new_held.extend(str(value) for value in raw["new_held_wordings"])
    if set(map(normalize_answer, existing)) & set(map(normalize_answer, new_held)):
        raise ValueError("V91 new held wordings overlap the V90 training lineage")
    return {
        "source_v90_intent_count": len(parent_by_id),
        "existing_wording_count": len(existing),
        "new_held_wording_count": len(new_held),
        "all_existing_wordings_exact_v90_rows": True,
        "new_held_wordings_disjoint": True,
    }


def _wording_row(
    raw: Mapping[str, Any], *, held: bool, ordinal: int, question: str
) -> RowV73:
    intent_id = str(raw["id"])
    split = "new_held" if held else "existing"
    question_id = f"v91_{intent_id}_{split}_{ordinal:02d}"
    answer = str(raw["answer"]).strip()
    return RowV73(
        scene_id=SCENE_ID,
        question_id=question_id,
        question=question.strip(),
        answer=answer,
        answer_class=_answer_class(answer),
        answer_type=str(raw["family"]),
        pair_id=f"v91_conversation_{intent_id}",
        paired_scene_id=SCENE_ID,
        question_key=f"v91_conversation_{intent_id}",
        change_type="wording",
        expected_change=False,
    )


def training_wording_rows_v91(config: Mapping[str, Any]) -> tuple[RowV73, ...]:
    rows = tuple(
        _wording_row(raw, held=False, ordinal=ordinal, question=str(question))
        for raw in config["conversational_intents"]
        for ordinal, question in enumerate(raw["existing_wordings"])
    )
    if len(rows) != 78 or len({row.question_id for row in rows}) != 78:
        raise RuntimeError("V91 78-row conversational training expansion changed")
    return rows


conversational_rows_v91 = training_wording_rows_v91


def primary_rows_v91(config: Mapping[str, Any]) -> tuple[RowV73, ...]:
    rows = tuple(row for row in training_wording_rows_v91(config) if row.question_id.endswith("_00"))
    if len(rows) != 13:
        raise RuntimeError("V91 primary prompt inventory changed")
    return rows


def held_wording_rows_v91(config: Mapping[str, Any]) -> tuple[RowV73, ...]:
    rows = tuple(
        _wording_row(raw, held=True, ordinal=ordinal, question=str(question))
        for raw in config["conversational_intents"]
        for ordinal, question in enumerate(raw["new_held_wordings"])
    )
    if len(rows) != 26 or len({row.question_id for row in rows}) != 26:
        raise RuntimeError("V91 new held-wording expansion changed")
    return rows


new_held_wording_rows_v91 = held_wording_rows_v91


def _clone(row: RowV73, schedule_id: str) -> RowV73:
    return replace(row, question_id=schedule_id, question_key=schedule_id)


def parent_correct_and_errors_v91(
    config: Mapping[str, Any], rows: Sequence[RowV73]
) -> tuple[tuple[RowV73, ...], tuple[RowV73, ...]]:
    payload = strict_json_v89(config["sources"]["parent_v90_predictions"])
    expected = {
        "artifact": "gemma4_v90_scene1_conversational_predictions_v1",
        "schema_version": 90,
        "status": "fixed_final_evaluation_only_not_runtime",
        "scene_id": SCENE_ID,
        "scene_count": 1,
        "canonical_row_count": 138,
        "primary_conversational_row_count": 13,
        "held_wording_row_count": 26,
        "fixed_checkpoint_selected_before_scoring": True,
        "checkpoint_selection_after_scoring": False,
        "frozen_parent_state_invariant": True,
        "questions_or_answers_serialized_in_runtime_candidate": False,
        "training_inventory_serialized_in_runtime_candidate": False,
        "oracle_serialized_in_runtime_candidate": False,
        "runtime_promotion_authorized": False,
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ValueError("V91 V90 parent prediction identity changed")
    candidate = payload.get("candidate")
    leakage = payload.get("leakage")
    memory = payload.get("scene_memory")
    if (
        not isinstance(candidate, Mapping)
        or candidate.get("state_sha256") != V90_STATE_SHA256
        or candidate.get("weights_sha256")
        != config["sources"]["parent_v90_bridge_sha256"]
        or candidate.get("optimizer_updates") != 172
        or not isinstance(leakage, Mapping)
        or leakage.get("oracle_loaded") is not False
        or leakage.get("protected_read_count") != 0
        or not isinstance(memory, Mapping)
        or memory.get("prefix_hash_invariant") is not True
        or memory.get("environment_conditioned_input_invariant") is not True
        or memory.get("prefix_sha256_before") != EXPECTED_PREFIX_SHA256
        or memory.get("prefix_sha256_after") != EXPECTED_PREFIX_SHA256
        or memory.get("shape") != [1, 738, 1536]
    ):
        raise ValueError("V91 V90 parent candidate/leakage/memory evidence changed")
    records = payload.get("canonical_records")
    if not isinstance(records, list) or len(records) != 138:
        raise ValueError("V91 V90 canonical prediction inventory changed")
    by_id = {row.question_id: row for row in rows}
    records_by_id = {str(record.get("question_id")): record for record in records}
    if len(records_by_id) != 138 or set(records_by_id) != set(by_id):
        raise ValueError("V91 V90 canonical prediction coverage changed")
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
            raise ValueError("V91 V90 canonical prediction binding changed")
        target = (
            correct
            if canonical_type_specific_match(
                row.answer_type, record.get("prediction", ""), row.answer
            )
            else errors
        )
        target.append(row)
    if len(correct) != 124 or len(errors) != 14:
        raise ValueError("V91 requires exact V90 124/14 canonical split")
    return tuple(correct), tuple(errors)


def derive_training_items_v91(
    config: Mapping[str, Any], canonical: Sequence[RowV73] | None = None
) -> tuple[TrainingItemV91, ...]:
    canonical_rows = tuple(canonical) if canonical is not None else load_canonical_rows_v91(config)
    correct, errors = parent_correct_and_errors_v91(config, canonical_rows)
    items: list[TrainingItemV91] = [
        TrainingItemV91(row.question_id, "canonical", row.question_id, row, False)
        for row in sorted(canonical_rows, key=lambda value: value.question_id)
    ]
    for row in sorted(errors, key=lambda value: value.question_id):
        for copy in range(2):
            schedule_id = f"v91_error_{copy}_{row.question_id}"
            items.append(
                TrainingItemV91(
                    schedule_id,
                    "error_replay",
                    row.question_id,
                    _clone(row, schedule_id),
                    False,
                    copy_ordinal=copy,
                )
            )
    for row in sorted(correct, key=lambda value: value.question_id):
        schedule_id = f"v91_anchor_{row.question_id}"
        items.append(
            TrainingItemV91(
                schedule_id,
                "correct_anchor_replay",
                row.question_id,
                _clone(row, schedule_id),
                False,
            )
        )
    copies = {**{value: 2 for value in SUCCESSFUL_INTENTS}, **{value: 6 for value in FAILED_INTENTS}}
    for row in training_wording_rows_v91(config):
        intent_id = row.pair_id.removeprefix("v91_conversation_")
        wording_ordinal = int(row.question_id.rsplit("_", 1)[1])
        kind = (
            "conversational_repair"
            if intent_id in FAILED_INTENTS
            else "conversational_success"
        )
        for copy in range(copies[intent_id]):
            schedule_id = f"{row.question_id}_copy_{copy:02d}"
            items.append(
                TrainingItemV91(
                    schedule_id,
                    kind,
                    row.question_id,
                    _clone(row, schedule_id),
                    wording_ordinal == 0 and copy == 0,
                    intent_id,
                    wording_ordinal,
                    copy,
                )
            )
    counts = Counter(item.kind for item in items)
    held_ids = {row.question_id for row in held_wording_rows_v91(config)}
    if (
        len(items) != 590
        or len({item.schedule_id for item in items}) != 590
        or counts
        != {
            "canonical": 138,
            "error_replay": 28,
            "correct_anchor_replay": 124,
            "conversational_success": 84,
            "conversational_repair": 216,
        }
        or held_ids & {item.schedule_id for item in items}
        or sum(item.causal_margin for item in items) != 13
    ):
        raise RuntimeError("V91 fixed 590-row repair inventory changed")
    return tuple(items)


def inventory_v91(items: Sequence[TrainingItemV91]) -> list[dict[str, Any]]:
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


training_inventory_v91 = inventory_v91


def schedule_v91(
    items: Sequence[TrainingItemV91], *, seed: int = 910091, epochs: int = 3
) -> tuple[tuple[int, TrainingItemV91], ...]:
    schedule: list[tuple[int, TrainingItemV91]] = []
    for epoch in range(epochs):
        shuffled = sorted(items, key=lambda item: item.schedule_id)
        random.Random(seed + epoch).shuffle(shuffled)
        schedule.extend((epoch, item) for item in shuffled)
    return tuple(schedule)


training_schedule_v91 = schedule_v91


class _SyntheticMlp(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.down_proj = nn.Linear(
            TARGET_IN_FEATURES,
            TARGET_OUT_FEATURES,
            bias=False,
            dtype=torch.bfloat16,
        )


class _SyntheticLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = _SyntheticMlp()


class _SyntheticLanguage(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Identity() for _ in range(33)]
            + [_SyntheticLayer()]
            + [nn.Identity()]
        )


class _SyntheticGemma(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = _SyntheticLanguage()


def derive_lora_preflight_v91(config: Mapping[str, Any]) -> dict[str, Any]:
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
        raise RuntimeError("V91 synthetic LoRA installation failed")
    initialize_lora_adapter_state(installation, seed=int(bridge["initialization_seed"]))
    return {
        "bank_name": FRESH_BANK_NAME,
        "target_modules": list(installation.target_names),
        "parameter_count": installation.parameter_count,
        "initial_state_sha256": installation.state_sha256(),
        "base_projection_type": "torch.nn.Linear",
        "base_projection_weight_shape": [TARGET_OUT_FEATURES, TARGET_IN_FEATURES],
        "lora_a_shape": [16, TARGET_IN_FEATURES],
        "lora_b_shape": [TARGET_OUT_FEATURES, 16],
        "lora_b_nonzero_count": sum(
            int(torch.count_nonzero(adapter.lora_b).item())
            for adapter in installation.adapters
        ),
        "exact_zero_output_at_initialization": True,
        "full_gemma_model_loaded": False,
        "device": "cpu",
    }


def lora_preflight_v91(config: Mapping[str, Any]) -> dict[str, Any]:
    result = derive_lora_preflight_v91(config)
    if (
        result["target_modules"] != [TARGET_MODULE]
        or result["parameter_count"] != FRESH_PARAMETER_COUNT
        or result["base_projection_weight_shape"]
        != [TARGET_OUT_FEATURES, TARGET_IN_FEATURES]
        or result["lora_a_shape"] != [16, TARGET_IN_FEATURES]
        or result["lora_b_shape"] != [TARGET_OUT_FEATURES, 16]
        or result["initial_state_sha256"] != EXPECTED_INITIAL_STATE_SHA256
        or result["initial_state_sha256"] != config["bridge"]["expected_initial_state_sha256"]
        or result["lora_b_nonzero_count"] != 0
    ):
        raise RuntimeError("V91 deterministic exact-zero LoRA preflight changed")
    return result


def memory_preflight_v91(config: Mapping[str, Any]) -> dict[str, Any]:
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
        raise RuntimeError("V91 immutable memory CPU preflight changed")
    return result


def authenticate_parent_v91(config: Mapping[str, Any]) -> dict[str, Any]:
    sources = config["sources"]
    candidate_root = Path(sources["parent_v90_candidate"])
    metadata = strict_json_v89(candidate_root / "runtime_metadata.json")
    tensors = load_file(str(resolve_v85(candidate_root / "bridge.safetensors")), device="cpu")
    if (
        metadata.get("artifact")
        != "gemma4_v90_scene1_conversational_fixed_final_v1"
        or metadata.get("schema_version") != 90
        or metadata.get("status")
        != "fixed_final_awaiting_preregistered_acceptance_gates"
        or metadata.get("bank_name") != "v90_scene1_conversational_bridge"
        or metadata.get("target_module")
        != "model.language_model.layers.28.self_attn.o_proj"
        or metadata.get("rank") != 8
        or float(metadata.get("alpha", -1.0)) != 16.0
        or float(metadata.get("dropout", -1.0)) != 0.0
        or metadata.get("parameter_count") != 28672
        or metadata.get("state_sha256") != V90_STATE_SHA256
        or metadata.get("weights_sha256") != sources["parent_v90_bridge_sha256"]
        or metadata.get("frozen_parent_bank_count") != 11
        or metadata.get("total_bank_count") != 12
        or metadata.get("runtime_promotion_authorized") is not False
        or metadata.get("environmental_memory_serialized") is not False
        or metadata.get("environmental_text_serialized") is not False
        or metadata.get("environmental_text_inputs") != []
        or metadata.get("questions_or_answers_serialized") is not False
        or metadata.get("training_metadata_serialized") is not False
        or metadata.get("training_inventory_serialized") is not False
        or metadata.get("oracle_serialized") is not False
        or metadata.get("evaluation_scored") is not False
        or set(tensors) != {"lora_a", "lora_b"}
        or tensors["lora_a"].dtype != torch.float32
        or tensors["lora_b"].dtype != torch.float32
        or list(tensors["lora_a"].shape) != [8, 2048]
        or list(tensors["lora_b"].shape) != [1536, 8]
        or tensor_state_sha256(
            {f"adapters.0.{name}": value for name, value in tensors.items()}
        )
        != V90_STATE_SHA256
    ):
        raise ValueError("V91 exact failed V90 candidate authentication failed")
    preregistration = strict_json_v89(sources["parent_v90_preregistration"])
    cpu_preflight = strict_json_v89(sources["parent_v90_cpu_preflight"])
    training = strict_json_v89(sources["parent_v90_training"])
    evaluation = strict_json_v89(sources["parent_v90_evaluation"])
    metrics = evaluation.get("metrics")
    canonical = metrics.get("canonical_type_specific") if isinstance(metrics, Mapping) else None
    primary = metrics.get("primary_conversational") if isinstance(metrics, Mapping) else None
    held = metrics.get("held_wording") if isinstance(metrics, Mapping) else None
    causal = metrics.get("causal_control") if isinstance(metrics, Mapping) else None
    bindings = metadata.get("bindings")
    if (
        preregistration.get("artifact")
        != "gemma4_v90_scene1_conversational_preregistration_v1"
        or preregistration.get("schema_version") != 90
        or preregistration.get("status") != "sealed_before_full_model_load"
        or preregistration.get("config_sha256")
        != sources["parent_v90_config_sha256"]
        or cpu_preflight.get("artifact")
        != "gemma4_v90_scene1_conversational_cpu_preflight_v1"
        or cpu_preflight.get("schema_version") != 90
        or cpu_preflight.get("status")
        != "cpu_preflight_pass_training_authorized"
        or cpu_preflight.get("config_sha256")
        != sources["parent_v90_config_sha256"]
        or cpu_preflight.get("preregistration_sha256")
        != sources["parent_v90_preregistration_sha256"]
        or cpu_preflight.get("training_authorized") is not True
        or not isinstance(bindings, Mapping)
        or bindings.get("config_sha256") != sources["parent_v90_config_sha256"]
        or bindings.get("preregistration_sha256")
        != sources["parent_v90_preregistration_sha256"]
        or bindings.get("cpu_preflight_sha256")
        != sources["parent_v90_cpu_preflight_sha256"]
        or bindings.get("fixed_final_optimizer_updates") != 172
        or bindings.get("training_inventory_sha256")
        != "8963085d8276f2000480e11651cbe97fcc5b7eb7711c9c3c2cb3e20090fc9cc7"
        or bindings.get("training_schedule_sha256")
        != "3c3b199b723d9bf6be6686a421b997fbac2eb209c5886bcd4544a67fd7954643"
        or bindings.get("scene_memory_prefix_sha256") != EXPECTED_PREFIX_SHA256
        or training.get("artifact")
        != "gemma4_v90_scene1_conversational_training_v1"
        or training.get("status") != "fixed_final_training_complete_not_promoted"
        or training.get("config_sha256") != sources["parent_v90_config_sha256"]
        or training.get("preregistration_sha256")
        != sources["parent_v90_preregistration_sha256"]
        or training.get("cpu_preflight_sha256")
        != sources["parent_v90_cpu_preflight_sha256"]
        or training.get("optimizer_updates") != 172
        or training.get("micro_rows_consumed") != 1032
        or training.get("causal_margin_rows_consumed") != 39
        or training.get("oracle_loaded") is not False
        or training.get("protected_read_count") != 0
        or training.get("runtime_promotion_authorized") is not False
        or evaluation.get("artifact")
        != "gemma4_v90_scene1_conversational_evaluation_v1"
        or evaluation.get("status") != "model_gates_fail_not_runtime_promotable"
        or evaluation.get("config_sha256") != sources["parent_v90_config_sha256"]
        or evaluation.get("preregistration_sha256")
        != sources["parent_v90_preregistration_sha256"]
        or evaluation.get("cpu_preflight_sha256")
        != sources["parent_v90_cpu_preflight_sha256"]
        or evaluation.get("evaluation_predictions_sha256")
        != sources["parent_v90_predictions_sha256"]
        or evaluation.get("training_report_sha256")
        != sources["parent_v90_training_sha256"]
        or evaluation.get("runtime_promotion_authorized") is not False
        or evaluation.get("oracle_loaded") is not False
        or not isinstance(metrics, Mapping)
        or metrics.get("model_acceptance_gate_passed") is not False
        or canonical != {"accuracy": 124 / 138, "correct": 124, "total": 138}
        or not isinstance(primary, Mapping)
        or primary.get("correct") != 7
        or primary.get("total") != 13
        or not isinstance(held, Mapping)
        or held.get("correct") != 10
        or held.get("total") != 26
        or not isinstance(causal, Mapping)
        or causal.get("canonical_prediction_changes") != 10
        or causal.get("mean_zero_minus_correct_nll") != 0.8939050505152688
    ):
        raise ValueError("V91 measured failed-V90 evidence authentication failed")
    return {
        "v89_frozen_bank_count": 11,
        "v89_frozen_parameter_count": 872448,
        "v90_bank_name": metadata["bank_name"],
        "v90_state_sha256": metadata["state_sha256"],
        "v90_parameter_count": metadata["parameter_count"],
        "total_frozen_bank_count": 12,
        "total_frozen_parameter_count": 901120,
        "v90_model_acceptance_gate_passed": False,
        "v90_runtime_promotion_authorized": False,
        "canonical_correct": 124,
        "canonical_errors": 14,
        "primary_correct": 7,
        "new_repair_is_post_failure_development": True,
    }


def _source_bindings(config: Mapping[str, Any]) -> tuple[tuple[str, str, bool], ...]:
    sources = config["sources"]
    return (
        (str(sources["runtime_config"]), str(sources["runtime_config_sha256"]), False),
        (str(sources["scene1_qa"]), str(sources["scene1_qa_sha256"]), False),
        (str(Path(sources["scene1_memory"]) / "memory.safetensors"), str(sources["scene1_memory_tensor_sha256"]), False),
        (str(Path(sources["scene1_memory"]) / "runtime_metadata.json"), str(sources["scene1_memory_metadata_sha256"]), False),
        (str(Path(sources["parent_v89_checkpoint"]) / "adapter.safetensors"), str(sources["parent_v89_adapter_sha256"]), False),
        (str(Path(sources["parent_v89_checkpoint"]) / "runtime_metadata.json"), str(sources["parent_v89_metadata_sha256"]), False),
        (str(sources["parent_v90_config"]), str(sources["parent_v90_config_sha256"]), False),
        (str(sources["parent_v90_preregistration"]), str(sources["parent_v90_preregistration_sha256"]), False),
        (str(sources["parent_v90_cpu_preflight"]), str(sources["parent_v90_cpu_preflight_sha256"]), False),
        (str(sources["parent_v90_training"]), str(sources["parent_v90_training_sha256"]), False),
        (str(Path(sources["parent_v90_candidate"]) / "bridge.safetensors"), str(sources["parent_v90_bridge_sha256"]), False),
        (str(Path(sources["parent_v90_candidate"]) / "runtime_metadata.json"), str(sources["parent_v90_metadata_sha256"]), False),
        (str(sources["parent_v90_predictions"]), str(sources["parent_v90_predictions_sha256"]), False),
        (str(sources["parent_v90_evaluation"]), str(sources["parent_v90_evaluation_sha256"]), False),
        (str(sources["preflight_source"]), str(sources["preflight_source_sha256"]), True),
        (str(sources["training_source"]), str(sources["training_source_sha256"]), True),
        (str(sources["evaluation_source"]), str(sources["evaluation_source_sha256"]), True),
    )


def authenticate_pinned_model_tensor_v91(
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Authenticate the target weight's safetensors header without loading it."""

    sources = config["sources"]
    amendment = config["topology_amendment"]
    snapshot = (
        Path.home()
        / ".cache/huggingface/hub/models--google--gemma-4-E2B-it/snapshots"
        / str(sources["model_revision"])
    )
    model_link = snapshot / "model.safetensors"
    model_blob = model_link.resolve(strict=True)
    if model_blob.name != sources["model_blob_sha256_identity"]:
        raise ValueError("V91 pinned Gemma model blob identity changed")
    tensor_name = str(amendment["pinned_tensor_name"])
    with safe_open(str(model_blob), framework="pt", device="cpu") as handle:
        if tensor_name not in handle.keys():  # noqa: SIM118 - handle is not iterable.
            raise ValueError("V91 pinned layer-33 down-projection tensor is absent")
        tensor_slice = handle.get_slice(tensor_name)
        shape = list(tensor_slice.get_shape())
        dtype = str(tensor_slice.get_dtype())
    result = {
        "model_id": sources["model_id"],
        "model_revision": sources["model_revision"],
        "model_blob_sha256_identity": model_blob.name,
        "tensor_name": tensor_name,
        "shape": shape,
        "dtype": dtype,
        "header_read_via_safe_open": True,
        "tensor_materialized": False,
        "full_gemma_model_loaded": False,
    }
    if (
        shape != [TARGET_OUT_FEATURES, TARGET_IN_FEATURES]
        or shape != amendment["pinned_weight_shape"]
        or dtype != "BF16"
        or dtype != amendment["pinned_weight_dtype"]
    ):
        raise ValueError("V91 pinned layer-33 down-projection shape changed")
    return result


def authenticate_sources_v91(
    config: Mapping[str, Any], *, require_implementation_sources: bool = True
) -> dict[str, Any]:
    observed: dict[str, Any] = {}
    for path, expected, implementation in _source_bindings(config):
        if implementation and not require_implementation_sources and expected == "TO_FILL":
            continue
        if _HEX64.fullmatch(expected) is None:
            raise ValueError(f"V91 source hash is not sealed: {path}")
        actual = sha256_file_v85(path)
        if actual != expected:
            raise ValueError(f"V91 pinned source changed: {path}")
        observed[path] = actual
    sources = config["sources"]
    snapshot = (
        Path.home()
        / ".cache/huggingface/hub/models--google--gemma-4-E2B-it/snapshots"
        / str(sources["model_revision"])
    )
    model_blob = (snapshot / "model.safetensors").resolve(strict=True)
    model_config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    text = model_config.get("text_config")
    if (
        model_blob.name != sources["model_blob_sha256_identity"]
        or not isinstance(text, Mapping)
        or text.get("hidden_size") != 1536
        or text.get("num_hidden_layers") != 35
        or not isinstance(text.get("layer_types"), list)
        or text["layer_types"][33] != "sliding_attention"
        or text.get("intermediate_size") != 6144
    ):
        raise ValueError("V91 local Gemma layer-33 MLP contract changed")
    observed["gemma_model_blob_sha256_identity"] = model_blob.name
    observed["pinned_model_tensor"] = authenticate_pinned_model_tensor_v91(config)
    authenticate_parent_v91(config)
    return observed


def derive_contract_v91(config_path: str | Path = CONFIG) -> dict[str, Any]:
    config = load_config_v91(config_path, allow_draft=True)
    wording_lineage = authenticate_wording_lineage_v91(config)
    canonical = load_canonical_rows_v91(config)
    items = derive_training_items_v91(config, canonical)
    schedule = schedule_v91(
        items,
        seed=int(config["training"]["row_order_seed"]),
        epochs=int(config["dataset"]["epochs"]),
    )
    kinds = Counter(item.kind for item in items)
    return {
        "training_inventory_sha256": canonical_sha256_v85(inventory_v91(items)),
        "training_schedule_sha256": canonical_sha256_v85(
            [[epoch, item.schedule_id] for epoch, item in schedule]
        ),
        "canonical_rows_per_epoch": kinds["canonical"],
        "parent_error_replay_rows_per_epoch": kinds["error_replay"],
        "parent_correct_anchor_rows_per_epoch": kinds["correct_anchor_replay"],
        "successful_conversational_rows_per_epoch": kinds["conversational_success"],
        "failed_conversational_rows_per_epoch": kinds["conversational_repair"],
        "new_held_wording_rows": len(held_wording_rows_v91(config)),
        "rows_per_epoch": len(items),
        "epochs": int(config["dataset"]["epochs"]),
        "total_micro_rows": len(schedule),
        "gradient_accumulation_rows": int(config["training"]["gradient_accumulation_rows"]),
        "optimizer_updates": len(schedule) // int(config["training"]["gradient_accumulation_rows"]),
        "primary_causal_rows_per_epoch": sum(item.causal_margin for item in items),
        "total_primary_causal_rows": sum(item.causal_margin for _epoch, item in schedule),
        "wording_lineage": wording_lineage,
        "lora": lora_preflight_v91(config),
    }


def protocol_v91(
    config: Mapping[str, Any], *, config_path: str | Path = CONFIG, require_sealed_hashes: bool = True
) -> dict[str, Any]:
    contract = derive_contract_v91(config_path)
    dataset = config["dataset"]
    checks = {
        "canonical_138_exact": contract["canonical_rows_per_epoch"] == 138,
        "v90_errors_28_exact": contract["parent_error_replay_rows_per_epoch"] == 28,
        "v90_anchors_124_exact": contract["parent_correct_anchor_rows_per_epoch"] == 124,
        "success_intents_84_exact": contract["successful_conversational_rows_per_epoch"] == 84,
        "failed_intents_216_exact": contract["failed_conversational_rows_per_epoch"] == 216,
        "new_held_26_excluded": contract["new_held_wording_rows"] == 26,
        "rows_per_epoch_590_exact": contract["rows_per_epoch"] == 590,
        "three_epochs_exact": contract["epochs"] == 3,
        "micro_rows_1770_exact": contract["total_micro_rows"] == 1770,
        "accumulation_6_exact": contract["gradient_accumulation_rows"] == 6,
        "optimizer_updates_295_exact": contract["optimizer_updates"] == 295,
        "primary_causal_13_per_epoch_exact": contract["primary_causal_rows_per_epoch"] == 13,
        "primary_causal_39_total_exact": contract["total_primary_causal_rows"] == 39,
        "oracle_not_loaded_by_trainer": dataset["oracle_loaded_during_training"] is False,
        "runtime_serializes_no_supervision": dataset["questions_or_answers_serialized_at_runtime"] is False,
        "fresh_target_disjoint": config["bridge"]["disjoint_from_all_frozen_banks"] is True,
        "failed_v90_parent_not_misrepresented_as_release": config["parent_v90_result"]["runtime_promotion_authorized"] is False,
    }
    if require_sealed_hashes:
        checks.update(
            {
                "inventory_hash_exact": contract["training_inventory_sha256"] == dataset["training_inventory_sha256"],
                "schedule_hash_exact": contract["training_schedule_sha256"] == dataset["training_schedule_sha256"],
            }
        )
    if not all(checks.values()):
        raise RuntimeError(f"V91 protocol failed: {checks}")
    return {"checks": checks, **contract}


def build_preregistration_v91(config_path: str | Path = CONFIG) -> dict[str, Any]:
    config = load_config_v91(config_path, allow_draft=False)
    sources = authenticate_sources_v91(config)
    return {
        "artifact": PREREG_ARTIFACT,
        "schema_version": 91,
        "status": _SEALED,
        "config_sha256": sha256_file_v85(config_path),
        "source_sha256": sources,
        "parent": authenticate_parent_v91(config),
        "protocol": protocol_v91(config, config_path=config_path),
        "strict_input_contract": dict(config["strict_input_contract"]),
        "scope": dict(config["scope"]),
        "model_loaded": False,
        "mps_used": False,
        "oracle_loaded": False,
    }


def _expected_cpu_preflight_v91(
    config: Mapping[str, Any], *, config_path: str | Path, preregistration_sha256: str
) -> dict[str, Any]:
    return {
        "artifact": PREFLIGHT_ARTIFACT,
        "schema_version": 91,
        "status": "cpu_preflight_pass_training_authorized",
        "config_sha256": sha256_file_v85(config_path),
        "preregistration_sha256": preregistration_sha256,
        "protocol": protocol_v91(config, config_path=config_path),
        "memory": memory_preflight_v91(config),
        "parent": authenticate_parent_v91(config),
        "model_loaded": False,
        "mps_used": False,
        "optimizer_constructed": False,
        "oracle_loaded": False,
        "training_authorized": True,
    }


def run_cpu_preflight_v91(config_path: str | Path = CONFIG) -> dict[str, Any]:
    config = load_config_v91(config_path, allow_draft=False)
    prereg_path = resolve_v85(config["outputs"]["preregistration"])
    expected_prereg = build_preregistration_v91(config_path)
    if prereg_path.exists():
        if strict_json_v89(prereg_path) != expected_prereg:
            raise ValueError("V91 preregistration changed")
    else:
        atomic_create_json_v85(prereg_path, expected_prereg)
    result = _expected_cpu_preflight_v91(
        config,
        config_path=config_path,
        preregistration_sha256=sha256_file_v85(prereg_path),
    )
    output = resolve_v85(config["outputs"]["cpu_preflight"])
    if output.exists():
        if strict_json_v89(output) != result:
            raise ValueError("V91 CPU preflight changed")
    else:
        atomic_create_json_v85(output, result)
    return result


def authenticate_cpu_preflight_v91(
    config: Mapping[str, Any], *, config_path: str | Path = CONFIG
) -> dict[str, str]:
    load_config_v91(config_path, allow_draft=False)
    authenticate_sources_v91(config)
    prereg = resolve_v85(config["outputs"]["preregistration"])
    cpu = resolve_v85(config["outputs"]["cpu_preflight"])
    if strict_json_v89(prereg) != build_preregistration_v91(config_path):
        raise ValueError("V91 preregistration authentication failed")
    expected = _expected_cpu_preflight_v91(
        config,
        config_path=config_path,
        preregistration_sha256=sha256_file_v85(prereg),
    )
    if strict_json_v89(cpu) != expected:
        raise ValueError("V91 CPU preflight authentication failed")
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
        result = derive_contract_v91(args.config)
    elif args.command == "preregister":
        config = load_config_v91(args.config, allow_draft=False)
        path = resolve_v85(config["outputs"]["preregistration"])
        atomic_create_json_v85(path, build_preregistration_v91(args.config))
        result = {"path": str(path), "sha256": sha256_file_v85(path)}
    elif args.command == "preflight":
        result = run_cpu_preflight_v91(args.config)
    else:
        config = load_config_v91(args.config, allow_draft=False)
        result = authenticate_cpu_preflight_v91(config, config_path=args.config)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
