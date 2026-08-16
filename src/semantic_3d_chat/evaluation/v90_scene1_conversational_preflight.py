"""Seal and CPU-preflight V90's scene-one conversational continuation.

V90 is a development-known, single-scene conversational overfit experiment.
It freezes the released V89 stack and adds one disjoint, exact-zero-output
rank-8 LoRA bank.  Its environmental input remains the immutable pre-question
``[1, 738, 1536]`` continuous Gemma image prefix; no question controls scene
processing or retrieval.

This module is deliberately model-free.  It authenticates the local Gemma
snapshot identity without loading its weights, derives the fixed offline
training inventory, verifies the immutable scene memory on CPU, and constructs
create-once preregistration artifacts.  Oracle-derived answers are used only
to construct offline training rows and are never serialized into runtime
artifacts.
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
)
from semantic_3d_chat.language.prefix_injection import prefix_sha256
from semantic_3d_chat.training.train_question_control_v73 import RowV73

CONFIG: Final[Path] = Path("configs/experiments/gemma4_v90_scene1_conversational.yaml")
SCENE_ID: Final[str] = "scene_000001"
FRESH_BANK_NAME: Final[str] = "v90_scene1_conversational_bridge"
TARGET_MODULE: Final[str] = "model.language_model.layers.28.self_attn.o_proj"
PREREG_ARTIFACT: Final[str] = "gemma4_v90_scene1_conversational_preregistration_v1"
PREFLIGHT_ARTIFACT: Final[str] = "gemma4_v90_scene1_conversational_cpu_preflight_v1"
EXPECTED_INITIAL_STATE_SHA256: Final[str] = (
    "136039ab015a91dce401bae423f14be0277e253cec677c622bece4fd8a0c6f1a"
)
_HEX64: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{64}")
_DRAFT: Final[str] = "draft_before_sealed_preflight"
_SEALED: Final[str] = "sealed_before_full_model_load"


@dataclass(frozen=True)
class TrainingItemV90:
    """One unique per-epoch row in V90's fixed offline schedule."""

    schedule_id: str
    kind: str
    source_question_id: str
    row: RowV73
    causal_margin: bool
    intent_id: str | None = None
    wording: str | None = None


def _require_exact(value: Any, expected: Any, label: str) -> None:
    if value != expected:
        raise ValueError(f"V90 {label} changed")


def _require_hash_or_draft(value: Any, label: str, *, allow_draft: bool) -> None:
    if isinstance(value, str) and _HEX64.fullmatch(value) is not None:
        return
    if allow_draft and value == "TO_FILL":
        return
    raise ValueError(f"V90 {label} is not sealed")


def _answer_class(answer: str) -> str:
    normalized = normalize_answer(answer)
    if not normalized:
        raise ValueError("V90 answer normalizes empty")
    return "answer_" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:20]


def _validate_intents(config: Mapping[str, Any]) -> None:
    intents = config.get("conversational_intents")
    if not isinstance(intents, list) or len(intents) != 13:
        raise ValueError("V90 requires exactly 13 conversational intents")
    expected_ids = (
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
    if tuple(raw.get("id") for raw in intents if isinstance(raw, Mapping)) != expected_ids:
        raise ValueError("V90 conversational intent identity/order changed")
    questions: list[str] = []
    for raw in intents:
        if not isinstance(raw, Mapping) or set(raw) != {
            "id",
            "family",
            "answer",
            "primary",
            "train",
            "held_wording",
        }:
            raise ValueError("V90 conversational intent schema changed")
        if not all(str(raw[key]).strip() for key in ("id", "family", "answer", "primary")):
            raise ValueError("V90 conversational intent contains an empty scalar")
        train = raw["train"]
        held = raw["held_wording"]
        if (
            not isinstance(train, list)
            or len(train) != 3
            or not isinstance(held, list)
            or len(held) != 2
            or any(not isinstance(value, str) or not value.strip() for value in train + held)
        ):
            raise ValueError("V90 requires three train and two held wordings per intent")
        questions.extend([str(raw["primary"]), *train, *held])
    normalized_questions = [normalize_answer(question) for question in questions]
    if len(normalized_questions) != 78 or len(set(normalized_questions)) != 78:
        raise ValueError("V90 primary/train/held wordings overlap")


def load_config_v90(
    path: str | Path = CONFIG, *, allow_draft: bool = True
) -> dict[str, Any]:
    """Load and strictly validate V90; draft hashes are allowed only on request."""

    source = resolve_v85(path)
    if source.is_symlink() or not source.is_file():
        raise FileNotFoundError(source)
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or set(payload) != {"v90"}:
        raise ValueError("V90 config must contain exactly one v90 mapping")
    config = payload["v90"]
    if not isinstance(config, Mapping):
        raise TypeError("V90 config payload must be a mapping")
    _require_exact(config.get("schema_version"), 90, "schema version")
    _require_exact(
        config.get("artifact"),
        "gemma4_v90_scene1_conversational_direct_memory_v1",
        "artifact identity",
    )
    _require_exact(config.get("seed"), 900090, "seed")
    status = config.get("status")
    if status not in {_DRAFT, _SEALED} or (not allow_draft and status != _SEALED):
        raise ValueError("V90 config has not been sealed")
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
    dataset = config.get("dataset")
    if not isinstance(dataset, Mapping):
        raise TypeError("V90 dataset must be a mapping")
    expected_dataset = {
        "scene_id": SCENE_ID,
        "canonical_row_count": 138,
        "canonical_row_inventory_sha256": (
            "9919ff1bee4611dce4132d79fa50f6f6b4ace567a6df780a2e0e21bd88237a8e"
        ),
        "parent_correct_count": 122,
        "parent_error_count": 16,
        "parent_error_replay_copies": 2,
        "parent_correct_anchor_replay_copies": 1,
        "conversational_intent_count": 13,
        "conversational_rows_per_epoch": 52,
        "held_wording_row_count": 26,
        "rows_per_epoch": 344,
        "epochs": 3,
        "total_micro_rows": 1032,
        "labels_derived_offline_from_oracle": True,
        "oracle_loaded_during_training": False,
        "questions_or_answers_serialized_at_runtime": False,
    }
    if any(dataset.get(key) != value for key, value in expected_dataset.items()):
        raise ValueError("V90 fixed dataset contract changed")
    for key in ("training_inventory_sha256", "training_schedule_sha256"):
        _require_hash_or_draft(dataset.get(key), key, allow_draft=allow_draft)
    _validate_intents(config)
    _require_exact(
        config.get("frozen_stack"),
        {
            "parent_bank_count": 11,
            "parent_adapter_parameter_count": 872448,
            "merged_weights": False,
        },
        "frozen V89 stack",
    )
    _require_exact(
        config.get("bridge"),
        {
            "bank_name": FRESH_BANK_NAME,
            "target_module": TARGET_MODULE,
            "target_layer_type": "sliding_attention",
            "target_in_features": 2048,
            "target_out_features": 1536,
            "rank": 8,
            "alpha": 16.0,
            "dropout": 0.0,
            "trainable_parameter_count": 28672,
            "initialization_algorithm": "cpu_kaiming_uniform_a_exact_zero_b",
            "initialization_seed": 900090,
            "expected_initial_state_sha256": EXPECTED_INITIAL_STATE_SHA256,
            "disjoint_from_all_frozen_banks": True,
        },
        "sole fresh LoRA bridge",
    )
    _require_exact(
        config.get("training"),
        {
            "optimizer": "AdamW",
            "learning_rate": 0.00025,
            "weight_decay": 0.0,
            "gradient_accumulation_rows": 6,
            "optimizer_updates": 172,
            "gradient_clip_norm": 1.0,
            "answer_ce_weight": 1.0,
            "zero_payload_margin_weight": 1.0,
            "zero_payload_target_margin_nll": 0.5,
            "row_order_seed": 900090,
            "checkpoint_selection": "fixed_final_update_172",
            "intermediate_behavior_selection": False,
        },
        "training protocol",
    )
    _require_exact(
        config.get("gates"),
        {
            "canonical_accuracy_minimum": 0.8623,
            "canonical_presence_correct_minimum": 21,
            "canonical_count_correct_minimum": 9,
            "canonical_metric_correct_minimum": 1,
            "canonical_attribute_correct_minimum": 14,
            "canonical_spatial_correct_minimum": 69,
            "canonical_support_correct_minimum": 1,
            "primary_conversational_required_correct": 12,
            "primary_conversational_total": 13,
            "core_actionable_required_correct": 6,
            "held_wording_required_correct": 22,
            "held_wording_total": 26,
            "held_wording_each_intent_minimum": 1,
            "parent_smoke_required_correct": 3,
            "exact_prefix_hash_invariance_required": True,
            "exact_total_environment_input_invariance_required": True,
            "correct_memory_mean_nll_below_zero_payload": True,
            "causal_prediction_change_minimum": 6,
            "oracle_physically_unavailable_during_runtime_required": True,
            "forbidden_runtime_read_count_maximum": 0,
        },
        "acceptance gates",
    )
    _require_exact(
        config.get("scope"),
        {
            "single_scene_conversational_overfit": True,
            "development_known_questions": True,
            "local_inference_only": True,
            "cloud_inference": False,
            "held_out_generalization_claim": False,
            "official_validation_loaded": False,
            "official_test_loaded": False,
            "deferred_final_loaded": False,
            "runtime_promotion_authorized": False,
        },
        "development-only scope",
    )
    sources = config.get("sources")
    if not isinstance(sources, Mapping):
        raise TypeError("V90 sources must be a mapping")
    for key in (
        "runtime_config_sha256",
        "scene1_qa_sha256",
        "scene1_memory_tensor_sha256",
        "scene1_memory_metadata_sha256",
        "parent_adapter_sha256",
        "parent_metadata_sha256",
        "parent_predictions_sha256",
        "parent_evaluation_sha256",
        "parent_release_sha256",
    ):
        _require_hash_or_draft(sources.get(key), key, allow_draft=False)
    for key in ("preflight_source_sha256", "training_source_sha256", "evaluation_source_sha256"):
        _require_hash_or_draft(sources.get(key), key, allow_draft=allow_draft)
    return dict(config)


def load_canonical_rows_v90(config: Mapping[str, Any]) -> tuple[RowV73, ...]:
    """Load the exact 138 V89-authenticated canonical scene-one rows."""

    rows = load_canonical_rows_v89(config)
    if len(rows) != 138 or {row.scene_id for row in rows} != {SCENE_ID}:
        raise ValueError("V90 canonical scene-one inventory changed")
    return rows


def _intent_row(
    raw: Mapping[str, Any], *, wording: str, ordinal: int, question: str
) -> RowV73:
    intent_id = str(raw["id"])
    suffix = wording if ordinal < 0 else f"{wording}_{ordinal:02d}"
    question_id = f"v90_{intent_id}_{suffix}"
    answer = str(raw["answer"]).strip()
    return RowV73(
        scene_id=SCENE_ID,
        question_id=question_id,
        question=question.strip(),
        answer=answer,
        answer_class=_answer_class(answer),
        answer_type=str(raw["family"]),
        pair_id=f"v90_conversation_{intent_id}",
        paired_scene_id=SCENE_ID,
        question_key=f"v90_conversation_{intent_id}",
        change_type="wording",
        expected_change=False,
    )


def primary_rows_v90(config: Mapping[str, Any]) -> tuple[RowV73, ...]:
    """Return the 13 user-facing primary conversational prompts."""

    return tuple(
        _intent_row(raw, wording="primary", ordinal=-1, question=str(raw["primary"]))
        for raw in config["conversational_intents"]
    )


def conversational_rows_v90(config: Mapping[str, Any]) -> tuple[RowV73, ...]:
    """Expand every primary prompt plus its three training paraphrases (52 rows)."""

    rows: list[RowV73] = []
    for raw in config["conversational_intents"]:
        rows.append(
            _intent_row(raw, wording="primary", ordinal=-1, question=str(raw["primary"]))
        )
        rows.extend(
            _intent_row(raw, wording="train", ordinal=ordinal, question=str(question))
            for ordinal, question in enumerate(raw["train"])
        )
    if len(rows) != 52 or len({row.question_id for row in rows}) != 52:
        raise RuntimeError("V90 conversational expansion changed")
    return tuple(rows)


def held_wording_rows_v90(config: Mapping[str, Any]) -> tuple[RowV73, ...]:
    """Return the 26 evaluation-only paraphrases, never training schedule rows."""

    rows = tuple(
        _intent_row(raw, wording="held", ordinal=ordinal, question=str(question))
        for raw in config["conversational_intents"]
        for ordinal, question in enumerate(raw["held_wording"])
    )
    if len(rows) != 26 or len({row.question_id for row in rows}) != 26:
        raise RuntimeError("V90 held-wording expansion changed")
    return rows


def _clone(row: RowV73, schedule_id: str) -> RowV73:
    return replace(row, question_id=schedule_id, question_key=schedule_id)


def parent_correct_and_errors_v90(
    config: Mapping[str, Any], rows: Sequence[RowV73]
) -> tuple[tuple[RowV73, ...], tuple[RowV73, ...]]:
    """Reproduce the sealed V89 canonical 122-correct/16-error partition."""

    payload = strict_json_v89(config["sources"]["parent_predictions"])
    expected_root = {
        "artifact": "gemma4_v89_scene1_retention_predictions_v1",
        "schema_version": 89,
        "status": "fixed_final_evaluation_only_not_runtime",
        "row_count": 138,
        "scene_count": 1,
        "fixed_checkpoint_selected_before_scoring": True,
        "checkpoint_selection_after_scoring": False,
        "development_known_smoke_trained": True,
        "held_out_smoke_claim": False,
        "runtime_promotion_authorized": False,
        "training_references_serialized_in_runtime_candidate": False,
        "error_inventory_serialized_in_runtime_candidate": False,
        "anchor_inventory_serialized_in_runtime_candidate": False,
    }
    if any(payload.get(key) != value for key, value in expected_root.items()):
        raise ValueError("V90 V89 parent prediction identity changed")
    leakage = payload.get("leakage")
    scene_memory = payload.get("scene_memory")
    if (
        not isinstance(leakage, Mapping)
        or leakage.get("oracle_loaded") is not False
        or leakage.get("protected_read_count") != 0
        or not isinstance(scene_memory, Mapping)
        or scene_memory.get("prefix_hash_invariant") is not True
        or scene_memory.get("same_prefix_reused_for_every_question") is not True
        or scene_memory.get("prefix_sha256_before") != EXPECTED_PREFIX_SHA256
        or scene_memory.get("prefix_sha256_after") != EXPECTED_PREFIX_SHA256
        or scene_memory.get("shape") != [1, 738, 1536]
    ):
        raise ValueError("V90 V89 parent leakage/memory evidence changed")
    records = payload.get("records")
    if not isinstance(records, list) or len(records) != 138:
        raise ValueError("V90 V89 parent records changed")
    by_id = {row.question_id: row for row in rows}
    records_by_id = {str(record.get("question_id")): record for record in records}
    if len(records_by_id) != 138 or set(records_by_id) != set(by_id):
        raise ValueError("V90 V89 parent prediction coverage changed")
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
            raise ValueError("V90 V89 parent record binding changed")
        target = (
            correct
            if canonical_type_specific_match(
                row.answer_type, record.get("prediction", ""), row.answer
            )
            else errors
        )
        target.append(row)
    if len(correct) != 122 or len(errors) != 16:
        raise ValueError("V90 requires the exact V89 122/16 canonical split")
    return tuple(correct), tuple(errors)


def derive_training_items_v90(
    config: Mapping[str, Any], canonical: Sequence[RowV73] | None = None
) -> tuple[TrainingItemV90, ...]:
    """Derive the exact 344-row per-epoch training inventory."""

    canonical_rows = tuple(canonical) if canonical is not None else load_canonical_rows_v90(config)
    correct, errors = parent_correct_and_errors_v90(config, canonical_rows)
    items: list[TrainingItemV90] = [
        TrainingItemV90(row.question_id, "canonical", row.question_id, row, False)
        for row in sorted(canonical_rows, key=lambda value: value.question_id)
    ]
    for row in sorted(errors, key=lambda value: value.question_id):
        for copy in ("a", "b"):
            schedule_id = f"v90_error_{copy}_{row.question_id}"
            items.append(
                TrainingItemV90(
                    schedule_id,
                    "error_replay",
                    row.question_id,
                    _clone(row, schedule_id),
                    False,
                )
            )
    for row in sorted(correct, key=lambda value: value.question_id):
        schedule_id = f"v90_anchor_{row.question_id}"
        items.append(
            TrainingItemV90(
                schedule_id,
                "correct_anchor_replay",
                row.question_id,
                _clone(row, schedule_id),
                False,
            )
        )
    for row in conversational_rows_v90(config):
        intent_id = row.pair_id.removeprefix("v90_conversation_")
        wording = "primary" if row.question_id.endswith("_primary") else "train"
        items.append(
            TrainingItemV90(
                row.question_id,
                "conversational",
                row.question_id,
                row,
                wording == "primary",
                intent_id,
                wording,
            )
        )
    counts = Counter(item.kind for item in items)
    held_ids = {row.question_id for row in held_wording_rows_v90(config)}
    if (
        len(items) != 344
        or len({item.schedule_id for item in items}) != 344
        or counts
        != {
            "canonical": 138,
            "error_replay": 32,
            "correct_anchor_replay": 122,
            "conversational": 52,
        }
        or held_ids & {item.schedule_id for item in items}
        or sum(item.causal_margin for item in items) != 13
    ):
        raise RuntimeError("V90 fixed 344-row inventory changed")
    return tuple(items)


def inventory_v90(items: Sequence[TrainingItemV90]) -> list[dict[str, Any]]:
    """Return the deterministic offline inventory used for the sealed hash."""

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
            "wording": item.wording,
        }
        for item in items
    ]


training_inventory_v90 = inventory_v90


def schedule_v90(
    items: Sequence[TrainingItemV90], *, seed: int = 900090, epochs: int = 3
) -> tuple[tuple[int, TrainingItemV90], ...]:
    """Seed-shuffle all 344 rows once per epoch, deterministically."""

    schedule: list[tuple[int, TrainingItemV90]] = []
    for epoch in range(epochs):
        shuffled = sorted(items, key=lambda item: item.schedule_id)
        random.Random(seed + epoch).shuffle(shuffled)
        schedule.extend((epoch, item) for item in shuffled)
    return tuple(schedule)


training_schedule_v90 = schedule_v90


class _SyntheticAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.o_proj = nn.Linear(2048, 1536, bias=False, dtype=torch.bfloat16)


class _SyntheticLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = _SyntheticAttention()


class _SyntheticLanguage(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Identity() for _ in range(28)]
            + [_SyntheticLayer()]
            + [nn.Identity() for _ in range(6)]
        )


class _SyntheticGemma(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = _SyntheticLanguage()


def derive_lora_preflight_v90(config: Mapping[str, Any]) -> dict[str, Any]:
    """Install the fresh bank on a tiny shape-faithful CPU projection."""

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
        raise RuntimeError("V90 synthetic LoRA installation failed")
    initialize_lora_adapter_state(installation, seed=int(bridge["initialization_seed"]))
    return {
        "bank_name": FRESH_BANK_NAME,
        "target_modules": list(installation.target_names),
        "parameter_count": installation.parameter_count,
        "initial_state_sha256": installation.state_sha256(),
        "base_projection_type": "torch.nn.Linear",
        "base_projection_weight_shape": [1536, 2048],
        "lora_a_shape": [8, 2048],
        "lora_b_shape": [1536, 8],
        "lora_b_nonzero_count": sum(
            int(torch.count_nonzero(adapter.lora_b).item())
            for adapter in installation.adapters
        ),
        "exact_zero_output_at_initialization": True,
        "full_gemma_model_loaded": False,
        "device": "cpu",
    }


def lora_preflight_v90(config: Mapping[str, Any]) -> dict[str, Any]:
    result = derive_lora_preflight_v90(config)
    if (
        result["target_modules"] != [TARGET_MODULE]
        or result["parameter_count"] != 28672
        or result["initial_state_sha256"] != EXPECTED_INITIAL_STATE_SHA256
        or result["initial_state_sha256"] != config["bridge"]["expected_initial_state_sha256"]
        or result["lora_b_nonzero_count"] != 0
    ):
        raise RuntimeError("V90 deterministic exact-zero LoRA preflight changed")
    return result


def memory_preflight_v90(config: Mapping[str, Any]) -> dict[str, Any]:
    """Authenticate immutable memory and construct the exact zero-payload control."""

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
    if result != {
        **result,
        "shape": [1, 738, 1536],
        "dtype": "torch.bfloat16",
        "canonical_prefix_sha256": EXPECTED_PREFIX_SHA256,
        "native_boi_preserved": True,
        "native_eoi_preserved": True,
        "zeroed_interior_tokens": 736,
        "compiled_before_question": True,
        "question_inputs_used": False,
        "questions_or_answers_serialized": False,
        "oracle_loaded": False,
        "model_loaded": False,
        "device": "cpu",
    }:
        raise RuntimeError("V90 immutable memory CPU preflight changed")
    return result


def authenticate_parent_v90(config: Mapping[str, Any]) -> dict[str, Any]:
    """Authenticate V89 release gates, bindings, and its 11 frozen LoRA banks."""

    sources = config["sources"]
    release = strict_json_v89(sources["parent_release"])
    evaluation = strict_json_v89(sources["parent_evaluation"])
    metadata = strict_json_v89(Path(sources["parent_checkpoint"]) / "runtime_metadata.json")
    bindings = release.get("bindings")
    checkpoint = release.get("checkpoint")
    release_memory = release.get("scene_memory")
    if (
        release.get("artifact") != "gemma4_v89_strict_runtime_release_v1"
        or release.get("schema_version") != 89
        or release.get("scene_id") != SCENE_ID
        or release.get("promotion_decision") != "strict_scene1_experimental_primary"
        or release.get("all_release_gates_passed") is not True
        or release.get("runtime_checkpoint_contains_supervision") is not False
        or release.get("runtime_checkpoint_contains_environmental_text") is not False
        or release.get("chat_runtime_loads_training_or_evaluation_reports") is not False
        or not isinstance(bindings, Mapping)
        or bindings.get("evaluation_predictions_sha256")
        != sources["parent_predictions_sha256"]
        or bindings.get("model_gate_report_sha256") != sources["parent_evaluation_sha256"]
        or bindings.get("model_acceptance_gate_passed") is not True
        or bindings.get("held_out_generalization_claim") is not False
        or not isinstance(checkpoint, Mapping)
        or checkpoint.get("adapter_sha256") != sources["parent_adapter_sha256"]
        or checkpoint.get("runtime_metadata_sha256") != sources["parent_metadata_sha256"]
        or checkpoint.get("exact_two_file_checkpoint") is not True
        or not isinstance(release_memory, Mapping)
        or release_memory.get("canonical_prefix_sha256") != EXPECTED_PREFIX_SHA256
        or release_memory.get("packaged_memory_tensor_file_sha256")
        != sources["scene1_memory_tensor_sha256"]
        or release_memory.get("question_data_used_for_rebinding") is not False
    ):
        raise ValueError("V90 V89 strict runtime release authentication failed")
    metrics = evaluation.get("metrics")
    canonical = metrics.get("canonical_type_specific") if isinstance(metrics, Mapping) else None
    if (
        evaluation.get("artifact") != "gemma4_v89_scene1_retention_evaluation_v1"
        or evaluation.get("schema_version") != 89
        or evaluation.get("evaluation_predictions_sha256")
        != sources["parent_predictions_sha256"]
        or evaluation.get("oracle_loaded") is not False
        or evaluation.get("held_out_generalization_claim") is not False
        or not isinstance(metrics, Mapping)
        or metrics.get("model_acceptance_gate_passed") is not True
        or canonical != {"accuracy": 0.8840579710144928, "correct": 122, "total": 138}
    ):
        raise ValueError("V90 V89 parent evaluation authentication failed")
    lora = metadata.get("lora")
    provenance = metadata.get("initialization_provenance")
    release_provenance = (
        provenance.get("v89_strict_runtime_release")
        if isinstance(provenance, Mapping)
        else None
    )
    banks = lora.get("banks") if isinstance(lora, Mapping) else None
    if (
        metadata.get("schema_version") != 1
        or metadata.get("language_model_id") != sources["model_id"]
        or metadata.get("language_revision") != sources["model_revision"]
        or metadata.get("question_dependent_scene_processing") is not False
        or metadata.get("lora_parameter_count") != 872448
        or metadata.get("lora_trainable_parameter_count") != 0
        or not isinstance(lora, Mapping)
        or lora.get("adapter_parameter_count") != 872448
        or lora.get("trainable_adapter_parameter_count") != 0
        or not isinstance(banks, list)
        or len(banks) != 11
        or any(bank.get("trainable") is not False for bank in banks)
        or any(TARGET_MODULE in bank.get("target_modules", []) for bank in banks)
        or not isinstance(release_provenance, Mapping)
        or release_provenance.get("schema_version") != 89
        or release_provenance.get("runtime_promotion_authorized") is not True
        or release_provenance.get("model_acceptance_gate_passed") is not True
        or release_provenance.get("held_out_generalization_claim") is not False
        or release_provenance.get("model_gate_report_sha256")
        != sources["parent_evaluation_sha256"]
        or release_provenance.get("evaluation_predictions_sha256")
        != sources["parent_predictions_sha256"]
    ):
        raise ValueError("V90 V89 frozen checkpoint metadata authentication failed")
    return {
        "release_artifact": release["artifact"],
        "promotion_decision": release["promotion_decision"],
        "all_release_gates_passed": True,
        "canonical_correct": 122,
        "canonical_errors": 16,
        "frozen_bank_count": len(banks),
        "frozen_adapter_parameter_count": metadata["lora_parameter_count"],
        "fresh_target_disjoint": True,
        "runtime_promotion_authorized": True,
        "held_out_generalization_claim": False,
    }


def _source_bindings(config: Mapping[str, Any]) -> tuple[tuple[str, str, bool], ...]:
    sources = config["sources"]
    return (
        (str(sources["runtime_config"]), str(sources["runtime_config_sha256"]), False),
        (str(sources["scene1_qa"]), str(sources["scene1_qa_sha256"]), False),
        (
            str(Path(sources["scene1_memory"]) / "memory.safetensors"),
            str(sources["scene1_memory_tensor_sha256"]),
            False,
        ),
        (
            str(Path(sources["scene1_memory"]) / "runtime_metadata.json"),
            str(sources["scene1_memory_metadata_sha256"]),
            False,
        ),
        (
            str(Path(sources["parent_checkpoint"]) / "adapter.safetensors"),
            str(sources["parent_adapter_sha256"]),
            False,
        ),
        (
            str(Path(sources["parent_checkpoint"]) / "runtime_metadata.json"),
            str(sources["parent_metadata_sha256"]),
            False,
        ),
        (str(sources["parent_predictions"]), str(sources["parent_predictions_sha256"]), False),
        (str(sources["parent_evaluation"]), str(sources["parent_evaluation_sha256"]), False),
        (str(sources["parent_release"]), str(sources["parent_release_sha256"]), False),
        (str(sources["preflight_source"]), str(sources["preflight_source_sha256"]), True),
        (str(sources["training_source"]), str(sources["training_source_sha256"]), True),
        (str(sources["evaluation_source"]), str(sources["evaluation_source_sha256"]), True),
    )


def authenticate_sources_v90(
    config: Mapping[str, Any], *, require_implementation_sources: bool = True
) -> dict[str, str]:
    """Hash pinned files and authenticate parent/model identities without model load."""

    observed: dict[str, str] = {}
    for path, expected, implementation_source in _source_bindings(config):
        if implementation_source and not require_implementation_sources and expected == "TO_FILL":
            continue
        if _HEX64.fullmatch(expected) is None:
            raise ValueError(f"V90 source hash is not sealed: {path}")
        actual = sha256_file_v85(path)
        if actual != expected:
            raise ValueError(f"V90 pinned source changed: {path}")
        observed[path] = actual
    sources = config["sources"]
    snapshot = (
        Path.home()
        / ".cache/huggingface/hub"
        / "models--google--gemma-4-E2B-it"
        / "snapshots"
        / str(sources["model_revision"])
    )
    model_blob = (snapshot / "model.safetensors").resolve(strict=True)
    if model_blob.name != sources["model_blob_sha256_identity"]:
        raise ValueError("V90 local Gemma model blob identity changed")
    model_config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    text_config = model_config.get("text_config")
    if (
        not isinstance(text_config, Mapping)
        or text_config.get("hidden_size") != 1536
        or text_config.get("num_hidden_layers") != 35
        or not isinstance(text_config.get("layer_types"), list)
        or text_config["layer_types"][28] != "sliding_attention"
    ):
        raise ValueError("V90 local Gemma layer-28 contract changed")
    observed["gemma_model_blob_sha256_identity"] = model_blob.name
    authenticate_parent_v90(config)
    return observed


def derive_contract_v90(config_path: str | Path = CONFIG) -> dict[str, Any]:
    """Derive all model-free V90 hashes and exact schedule counts, including draft."""

    config = load_config_v90(config_path, allow_draft=True)
    canonical = load_canonical_rows_v90(config)
    items = derive_training_items_v90(config, canonical)
    schedule = schedule_v90(
        items,
        seed=int(config["training"]["row_order_seed"]),
        epochs=int(config["dataset"]["epochs"]),
    )
    kind_counts = Counter(item.kind for item in items)
    return {
        "training_inventory_sha256": canonical_sha256_v85(inventory_v90(items)),
        "training_schedule_sha256": canonical_sha256_v85(
            [[epoch, item.schedule_id] for epoch, item in schedule]
        ),
        "canonical_rows_per_epoch": kind_counts["canonical"],
        "parent_error_replay_rows_per_epoch": kind_counts["error_replay"],
        "parent_correct_anchor_rows_per_epoch": kind_counts["correct_anchor_replay"],
        "conversational_rows_per_epoch": kind_counts["conversational"],
        "held_wording_rows": len(held_wording_rows_v90(config)),
        "rows_per_epoch": len(items),
        "epochs": int(config["dataset"]["epochs"]),
        "total_micro_rows": len(schedule),
        "gradient_accumulation_rows": int(
            config["training"]["gradient_accumulation_rows"]
        ),
        "optimizer_updates": len(schedule)
        // int(config["training"]["gradient_accumulation_rows"]),
        "primary_causal_rows_per_epoch": sum(item.causal_margin for item in items),
        "total_primary_causal_rows": sum(
            item.causal_margin for _epoch, item in schedule
        ),
        "lora": lora_preflight_v90(config),
    }


def protocol_v90(
    config: Mapping[str, Any],
    *,
    config_path: str | Path = CONFIG,
    require_sealed_hashes: bool = True,
) -> dict[str, Any]:
    contract = derive_contract_v90(config_path)
    dataset = config["dataset"]
    checks = {
        "canonical_138_exact": contract["canonical_rows_per_epoch"] == 138,
        "v89_errors_32_exact": contract["parent_error_replay_rows_per_epoch"] == 32,
        "v89_anchors_122_exact": contract["parent_correct_anchor_rows_per_epoch"] == 122,
        "conversational_52_exact": contract["conversational_rows_per_epoch"] == 52,
        "held_wording_26_excluded_from_training": contract["held_wording_rows"] == 26,
        "rows_per_epoch_344_exact": contract["rows_per_epoch"] == 344,
        "three_epochs_exact": contract["epochs"] == 3,
        "micro_rows_1032_exact": contract["total_micro_rows"] == 1032,
        "accumulation_6_exact": contract["gradient_accumulation_rows"] == 6,
        "optimizer_updates_172_exact": contract["optimizer_updates"] == 172,
        "primary_causal_13_per_epoch_exact": contract["primary_causal_rows_per_epoch"] == 13,
        "primary_causal_39_total_exact": contract["total_primary_causal_rows"] == 39,
        "oracle_not_loaded_by_trainer": dataset["oracle_loaded_during_training"] is False,
        "runtime_serializes_no_supervision": dataset[
            "questions_or_answers_serialized_at_runtime"
        ]
        is False,
        "fresh_target_disjoint": config["bridge"]["disjoint_from_all_frozen_banks"]
        is True,
    }
    if require_sealed_hashes:
        checks.update(
            {
                "inventory_hash_exact": contract["training_inventory_sha256"]
                == dataset["training_inventory_sha256"],
                "schedule_hash_exact": contract["training_schedule_sha256"]
                == dataset["training_schedule_sha256"],
            }
        )
    if not all(checks.values()):
        raise RuntimeError(f"V90 protocol failed: {checks}")
    return {"checks": checks, **contract}


def build_preregistration_v90(config_path: str | Path = CONFIG) -> dict[str, Any]:
    config = load_config_v90(config_path, allow_draft=False)
    sources = authenticate_sources_v90(config)
    return {
        "artifact": PREREG_ARTIFACT,
        "schema_version": 90,
        "status": _SEALED,
        "config_sha256": sha256_file_v85(config_path),
        "source_sha256": sources,
        "parent": authenticate_parent_v90(config),
        "protocol": protocol_v90(config, config_path=config_path),
        "strict_input_contract": dict(config["strict_input_contract"]),
        "scope": dict(config["scope"]),
        "model_loaded": False,
        "mps_used": False,
        "oracle_loaded": False,
    }


def _expected_cpu_preflight_v90(
    config: Mapping[str, Any], *, config_path: str | Path, preregistration_sha256: str
) -> dict[str, Any]:
    return {
        "artifact": PREFLIGHT_ARTIFACT,
        "schema_version": 90,
        "status": "cpu_preflight_pass_training_authorized",
        "config_sha256": sha256_file_v85(config_path),
        "preregistration_sha256": preregistration_sha256,
        "protocol": protocol_v90(config, config_path=config_path),
        "memory": memory_preflight_v90(config),
        "parent": authenticate_parent_v90(config),
        "model_loaded": False,
        "mps_used": False,
        "optimizer_constructed": False,
        "oracle_loaded": False,
        "training_authorized": True,
    }


def run_cpu_preflight_v90(config_path: str | Path = CONFIG) -> dict[str, Any]:
    """Create or authenticate sealed preregistration and CPU-preflight artifacts."""

    config = load_config_v90(config_path, allow_draft=False)
    prereg_path = resolve_v85(config["outputs"]["preregistration"])
    expected_prereg = build_preregistration_v90(config_path)
    if prereg_path.exists():
        if strict_json_v89(prereg_path) != expected_prereg:
            raise ValueError("V90 preregistration changed")
    else:
        atomic_create_json_v85(prereg_path, expected_prereg)
    result = _expected_cpu_preflight_v90(
        config,
        config_path=config_path,
        preregistration_sha256=sha256_file_v85(prereg_path),
    )
    output = resolve_v85(config["outputs"]["cpu_preflight"])
    if output.exists():
        if strict_json_v89(output) != result:
            raise ValueError("V90 CPU preflight changed")
    else:
        atomic_create_json_v85(output, result)
    return result


def authenticate_cpu_preflight_v90(
    config: Mapping[str, Any], *, config_path: str | Path = CONFIG
) -> dict[str, str]:
    """Read-only authentication of already-created V90 preflight artifacts."""

    load_config_v90(config_path, allow_draft=False)
    authenticate_sources_v90(config)
    prereg_path = resolve_v85(config["outputs"]["preregistration"])
    cpu_path = resolve_v85(config["outputs"]["cpu_preflight"])
    if strict_json_v89(prereg_path) != build_preregistration_v90(config_path):
        raise ValueError("V90 preregistration authentication failed")
    expected_cpu = _expected_cpu_preflight_v90(
        config,
        config_path=config_path,
        preregistration_sha256=sha256_file_v85(prereg_path),
    )
    if strict_json_v89(cpu_path) != expected_cpu:
        raise ValueError("V90 CPU preflight authentication failed")
    return {
        "config_sha256": sha256_file_v85(config_path),
        "preregistration_sha256": sha256_file_v85(prereg_path),
        "cpu_preflight_sha256": sha256_file_v85(cpu_path),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("derive", "preregister", "preflight", "authenticate"))
    parser.add_argument("--config", type=Path, default=CONFIG)
    args = parser.parse_args(argv)
    if args.command == "derive":
        result = derive_contract_v90(args.config)
    elif args.command == "preregister":
        config = load_config_v90(args.config, allow_draft=False)
        path = resolve_v85(config["outputs"]["preregistration"])
        atomic_create_json_v85(path, build_preregistration_v90(args.config))
        result = {"path": str(path), "sha256": sha256_file_v85(path)}
    elif args.command == "preflight":
        result = run_cpu_preflight_v90(args.config)
    else:
        config = load_config_v90(args.config, allow_draft=False)
        result = authenticate_cpu_preflight_v90(config, config_path=args.config)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
