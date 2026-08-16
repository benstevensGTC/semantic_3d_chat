"""Seal and CPU-preflight V89's retention-aware scene-one correction.

V89 is explicitly post-V88 training-set development, not held-out evidence.
It freezes the exact V85+V86+V87+V88 stack and adds one disjoint, zero-output
rank-8 adapter.  Its fixed schedule gives all 138 canonical rows one exposure,
all 31 sealed V88 errors two extra exposures, all 107 V88-correct rows one
retention exposure, and the three development-known smoke rows one explicit
exposure per epoch.

This module is model-free.  It never loads Gemma weights, tokenizes questions,
constructs an optimizer, measures V89 behavior, or writes a runtime package.
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

from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.evaluation.metrics import normalize_answer
from semantic_3d_chat.evaluation.v56_fresh_development_score import (
    canonical_answer_key,
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
from semantic_3d_chat.evaluation.v88_scene1_augmented_preflight import (
    load_canonical_rows_v88,
)
from semantic_3d_chat.language.lora import (
    LoRASettings,
    initialize_lora_adapter_state,
    install_lora_adapters,
)
from semantic_3d_chat.language.prefix_injection import prefix_sha256
from semantic_3d_chat.training.train_question_control_v73 import RowV73

CONFIG: Final[Path] = Path("configs/experiments/gemma4_v89_scene1_retention_demo.yaml")
SCENE_ID: Final[str] = "scene_000001"
V86_BANK_NAME: Final[str] = "v86_scene1_demo_bridge"
V86_TARGET_MODULE: Final[str] = "model.language_model.layers.34.mlp.up_proj"
V87_BANK_NAME: Final[str] = "v87_scene1_balanced_bridge"
V87_TARGET_MODULE: Final[str] = "model.language_model.layers.34.mlp.gate_proj"
V88_BANK_NAME: Final[str] = "v88_scene1_augmented_bridge"
V88_TARGET_MODULE: Final[str] = "model.language_model.layers.27.self_attn.q_proj"
FRESH_BANK_NAME: Final[str] = "v89_scene1_retention_bridge"
TARGET_MODULE: Final[str] = "model.language_model.layers.27.self_attn.o_proj"
PREREG_ARTIFACT: Final[str] = "gemma4_v89_scene1_retention_preregistration_v1"
PREFLIGHT_ARTIFACT: Final[str] = "gemma4_v89_scene1_retention_cpu_preflight_v1"
CANONICAL_CAUSAL_IDS: Final[tuple[str, ...]] = (
    "q_000080",
    "q_000108",
    "q_000014",
)
SMOKE_CAUSAL_IDS: Final[tuple[str, ...]] = (
    "v89_smoke_00",
    "v89_smoke_01",
    "v89_smoke_02",
)
_HEX64: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class TrainingItemV89:
    schedule_id: str
    kind: str
    source_question_id: str
    row: RowV73
    causal_margin: bool


def strict_json_v89(path: str | Path) -> dict[str, Any]:
    source = resolve_v85(path)
    if source.is_symlink() or not source.is_file():
        raise FileNotFoundError(source)
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"V89 JSON must contain one object: {source}")
    return value


def _answer_class(answer: str) -> str:
    normalized = normalize_answer(answer)
    if not normalized:
        raise ValueError("V89 answer normalizes empty")
    return "answer_" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:20]


def load_canonical_rows_v89(config: Mapping[str, Any]) -> tuple[RowV73, ...]:
    return load_canonical_rows_v88(config)


def load_config_v89(path: str | Path = CONFIG) -> dict[str, Any]:
    source = resolve_v85(path)
    if source.is_symlink() or not source.is_file():
        raise FileNotFoundError(source)
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or set(payload) != {"v89"}:
        raise ValueError("V89 config must contain exactly one v89 mapping")
    config = payload["v89"]
    if not isinstance(config, Mapping):
        raise TypeError("V89 config payload must be a mapping")
    if any(
        config.get(key) != value
        for key, value in {
            "schema_version": 89,
            "artifact": "gemma4_v89_scene1_retention_direct_memory_overfit_v1",
            "status": "preregistered_before_full_model_load",
            "seed": 890089,
        }.items()
    ):
        raise ValueError("V89 experiment identity changed")
    if config.get("strict_input_contract") != {
        "shape": [1, 738, 1536],
        "native_boi_retained": True,
        "native_eoi_retained": True,
        "payload_tokens": 736,
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
    }:
        raise ValueError("V89 strict continuous-memory contract changed")
    dataset = config.get("dataset")
    expected_dataset = {
        "scene_id": SCENE_ID,
        "canonical_row_count": 138,
        "canonical_row_inventory_sha256": (
            "9919ff1bee4611dce4132d79fa50f6f6b4ace567a6df780a2e0e21bd88237a8e"
        ),
        "parent_v87_to_v88_transition_counts": {
            "retained_correct": 83,
            "recovered": 24,
            "regressed": 20,
            "retained_wrong": 11,
        },
        "parent_v88_error_count": 31,
        "parent_v88_error_type_counts": {
            "attribute": 7,
            "presence": 1,
            "spatial_relation": 22,
            "support": 1,
        },
        "parent_v88_correct_anchor_count": 107,
        "canonical_rows_per_epoch": 138,
        "hard_error_replay_copies": 2,
        "hard_error_replay_rows_per_epoch": 62,
        "correct_anchor_replay_rows_per_epoch": 107,
        "development_smoke_rows_per_epoch": 3,
        "total_rows_per_epoch": 310,
        "error_total_exposures_per_epoch": 3,
        "correct_total_exposures_per_epoch": 2,
        "selection_uses_training_metadata_only": True,
        "answer_metadata_training_only": True,
        "runtime_serializes_questions_or_answers": False,
        "runtime_serializes_training_inventory": False,
        "runtime_serializes_error_inventory": False,
        "runtime_serializes_anchor_inventory": False,
    }
    if not isinstance(dataset, Mapping) or any(
        dataset.get(key) != value for key, value in expected_dataset.items()
    ):
        raise ValueError("V89 fixed retention dataset contract changed")
    for key in (
        "parent_v87_to_v88_transition_inventory_sha256",
        "parent_v88_error_inventory_sha256",
        "parent_v88_correct_anchor_inventory_sha256",
        "training_row_inventory_sha256",
    ):
        if not isinstance(dataset.get(key), str) or _HEX64.fullmatch(dataset[key]) is None:
            raise ValueError(f"V89 {key} is not sealed")
    frozen = config.get("frozen_stack")
    expected_frozen = {
        "base_gemma_frozen": True,
        "v54_bank_count": 6,
        "v85_bank_name": "v85_strict_multiscene_bridge",
        "v85_bank_state_sha256": (
            "f31b8f99f77f1b7b92dafd74220e5e12ccfa35cbc8630a6d7640f2fe1f93c581"
        ),
        "v86_bank_name": V86_BANK_NAME,
        "v86_bank_target_module": V86_TARGET_MODULE,
        "v86_bank_state_sha256": (
            "8b6bd801716132c8aac50c6288b9ba588417dc5e6a7c2c15dd9515892f714260"
        ),
        "v87_bank_name": V87_BANK_NAME,
        "v87_bank_target_module": V87_TARGET_MODULE,
        "v87_bank_state_sha256": (
            "618c03e102d9a9eb98d405d5a040cd8285194539b5a4043d34f16356ac08769e"
        ),
        "v88_bank_name": V88_BANK_NAME,
        "v88_bank_target_module": V88_TARGET_MODULE,
        "v88_bank_state_sha256": (
            "ff311624150056c67ad1c0a06752a77af2de89878778049ae886aa59db3376aa"
        ),
        "total_frozen_bank_count": 10,
        "merged_weights": False,
    }
    if not isinstance(frozen, Mapping) or any(
        frozen.get(key) != value for key, value in expected_frozen.items()
    ):
        raise ValueError("V89 frozen V85+V86+V87+V88 stack changed")
    bridge = config.get("bridge")
    expected_bridge = {
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
        "initialization_seed": 890089,
        "disjoint_from_all_frozen_banks": True,
    }
    if not isinstance(bridge, Mapping) or any(
        bridge.get(key) != value for key, value in expected_bridge.items()
    ):
        raise ValueError("V89 sole fresh bridge contract changed")
    if not isinstance(bridge.get("expected_initial_state_sha256"), str) or _HEX64.fullmatch(
        bridge["expected_initial_state_sha256"]
    ) is None:
        raise ValueError("V89 initial bridge state is not sealed")
    training = config.get("training")
    expected_training = {
        "optimizer": "AdamW",
        "epochs": 3,
        "rows_per_epoch": 310,
        "microbatch_size": 1,
        "gradient_accumulation_rows": 6,
        "optimizer_updates": 155,
        "row_order": "deterministic_allrow_seeded_shuffle",
        "row_order_seed": 890089,
        "learning_rate": 0.00025,
        "weight_decay": 0.0,
        "gradient_clip_norm": 1.0,
        "answer_ce_weight": 1.0,
        "class_weighting": "none",
        "zero_payload_margin_weight": 1.0,
        "zero_payload_target_margin_nll": 0.5,
        "canonical_causal_question_ids": list(CANONICAL_CAUSAL_IDS),
        "development_smoke_causal_schedule_ids": list(SMOKE_CAUSAL_IDS),
        "zero_payload_preserves_native_boi_eoi": True,
        "zero_payload_zeros_exactly_736_interior_tokens": True,
        "causal_rows_per_epoch": 6,
        "total_causal_margin_rows": 18,
        "checkpoint_selection": "fixed_final_update_155",
        "intermediate_behavior_selection": False,
    }
    if not isinstance(training, Mapping) or any(
        training.get(key) != value for key, value in expected_training.items()
    ):
        raise ValueError("V89 fixed retention schedule/loss changed")
    if not isinstance(training.get("row_order_sha256"), str) or _HEX64.fullmatch(
        training["row_order_sha256"]
    ) is None:
        raise ValueError("V89 row order is not sealed")
    gates = config.get("gates")
    expected_gates = {
        "all_scene1_canonical_accuracy_minimum": 0.80,
        "attribute_accuracy_minimum": 0.50,
        "presence_accuracy_minimum": 0.75,
        "spatial_relation_accuracy_minimum": 0.60,
        "exact_training_row_count_required": 138,
        "live_smoke_required_correct": 3,
        "live_smoke_total": 3,
        "live_smoke_is_development_known_and_trained": True,
        "live_smoke_is_held_out": False,
        "causal_correct_memory_mean_nll_below_zero_payload": True,
        "causal_prediction_change_minimum": 1,
        "exact_prefix_hash_invariance_required": True,
        "exact_total_environment_input_invariance_required": True,
        "oracle_physically_unavailable_during_runtime_required": True,
        "forbidden_runtime_read_count_maximum": 0,
        "runtime_promotion_only_after_all_gates": True,
    }
    if not isinstance(gates, Mapping) or any(
        gates.get(key) != value for key, value in expected_gates.items()
    ):
        raise ValueError("V89 unchanged acceptance gates changed")
    if gates.get("live_smoke_questions") != [
        {"question": "Is there a chair?", "expected": "yes"},
        {"question": "What color is the bowl?", "expected": "red"},
        {"question": "Is the bowl left or right of the chair?", "expected": "left"},
    ]:
        raise ValueError("V89 development-known smoke changed")
    if config.get("scope") != {
        "post_v88_training_set_development": True,
        "single_scene_overfit_demonstration": True,
        "retention_aware_error_correction": True,
        "development_known_smoke": True,
        "local_inference_only": True,
        "cloud_inference": False,
        "held_out_generalization_claim": False,
        "official_validation_loaded": False,
        "official_test_loaded": False,
        "deferred_final_loaded": False,
        "oracle_loaded_during_training": False,
        "runtime_promotion_authorized": False,
    }:
        raise ValueError("V89 protected development-only scope changed")
    sources = config.get("sources")
    if not isinstance(sources, Mapping):
        raise TypeError("V89 sources must be a mapping")
    for key in ("preflight_source_sha256", "training_source_sha256", "evaluation_source_sha256"):
        if not isinstance(sources.get(key), str) or _HEX64.fullmatch(sources[key]) is None:
            raise ValueError(f"V89 {key} is not sealed")
    return dict(config)


def derive_parent_behavior_v89(
    config: Mapping[str, Any], rows: Sequence[RowV73]
) -> tuple[
    tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...],
    tuple[RowV73, ...],
    tuple[RowV73, ...],
]:
    v87_payload = strict_json_v89(config["sources"]["parent_v87_predictions"])
    v88_payload = strict_json_v89(config["sources"]["parent_v88_predictions"])
    if (
        v87_payload.get("artifact") != "gemma4_v87_scene1_balanced_predictions_v1"
        or v88_payload.get("artifact") != "gemma4_v88_scene1_augmented_predictions_v1"
        or v87_payload.get("row_count") != 138
        or v88_payload.get("row_count") != 138
    ):
        raise ValueError("V89 parent prediction artifact identity changed")
    row_by_id = {row.question_id: row for row in rows}
    v87_by_id = {str(record["question_id"]): record for record in v87_payload["records"]}
    v88_by_id = {str(record["question_id"]): record for record in v88_payload["records"]}
    if set(row_by_id) != set(v87_by_id) or set(row_by_id) != set(v88_by_id):
        raise ValueError("V89 parent predictions do not cover the canonical rows")
    transitions: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    anchors: list[dict[str, Any]] = []
    error_rows: list[RowV73] = []
    anchor_rows: list[RowV73] = []
    for question_id in sorted(row_by_id):
        row = row_by_id[question_id]
        v87 = v87_by_id[question_id]
        v88 = v88_by_id[question_id]
        v87_prediction = str(v87["prediction"])
        v88_prediction = str(v88["prediction"])
        v87_correct = canonical_type_specific_match(
            row.answer_type, v87_prediction, row.answer
        )
        v88_correct = canonical_type_specific_match(
            row.answer_type, v88_prediction, row.answer
        )
        transition = (
            "retained_correct"
            if v87_correct and v88_correct
            else "regressed"
            if v87_correct
            else "recovered"
            if v88_correct
            else "retained_wrong"
        )
        transitions.append(
            {
                "question_id": question_id,
                "answer_type": row.answer_type,
                "transition": transition,
                "v87_prediction": v87_prediction,
                "v87_correct": v87_correct,
                "v88_prediction": v88_prediction,
                "v88_correct": v88_correct,
                "v87_correct_mean_nll": float(v87["correct_mean_nll"]),
                "v88_correct_mean_nll": float(v88["correct_mean_nll"]),
            }
        )
        selection_record = {
            "question_id": question_id,
            "answer_type": row.answer_type,
            "reference_answer": row.answer,
            "v88_prediction": v88_prediction,
            "v88_canonical_prediction": canonical_answer_key(
                row.answer_type, v88_prediction
            ),
            "v88_correct_mean_nll": float(v88["correct_mean_nll"]),
        }
        if v88_correct:
            anchors.append(selection_record)
            anchor_rows.append(row)
        else:
            errors.append(selection_record)
            error_rows.append(row)
    return (
        tuple(transitions),
        tuple(errors),
        tuple(anchors),
        tuple(error_rows),
        tuple(anchor_rows),
    )


def _clone_row(row: RowV73, *, schedule_id: str, question: str, answer: str) -> RowV73:
    normalized = normalize_answer(answer)
    return replace(
        row,
        question_id=schedule_id,
        question=question,
        answer=normalized,
        answer_class=_answer_class(normalized),
        question_key=schedule_id,
    )


def development_smoke_rows_v89(
    config: Mapping[str, Any], rows: Sequence[RowV73]
) -> tuple[RowV73, ...]:
    by_id = {row.question_id: row for row in rows}
    source_ids = ("q_000080", "q_000108", "q_000014")
    answer_types = ("presence", "attribute", "spatial_relation")
    result: list[RowV73] = []
    for ordinal, (raw, source_id, answer_type) in enumerate(
        zip(
            config["gates"]["live_smoke_questions"],
            source_ids,
            answer_types,
            strict=True,
        )
    ):
        row = _clone_row(
            by_id[source_id],
            schedule_id=f"v89_smoke_{ordinal:02d}",
            question=str(raw["question"]),
            answer=str(raw["expected"]),
        )
        if row.answer_type != answer_type:
            raise ValueError("V89 smoke answer type changed")
        result.append(row)
    return tuple(result)


def derive_training_items_v89(
    config: Mapping[str, Any],
    rows: Sequence[RowV73],
    error_rows: Sequence[RowV73],
    anchor_rows: Sequence[RowV73],
) -> tuple[TrainingItemV89, ...]:
    items: list[TrainingItemV89] = []
    for row in sorted(rows, key=lambda value: value.question_id):
        items.append(
            TrainingItemV89(
                schedule_id=row.question_id,
                kind="canonical",
                source_question_id=row.question_id,
                row=row,
                causal_margin=row.question_id in CANONICAL_CAUSAL_IDS,
            )
        )
    for row in sorted(error_rows, key=lambda value: value.question_id):
        for copy in ("a", "b"):
            schedule_id = f"v89_error_{copy}_{row.question_id}"
            items.append(
                TrainingItemV89(
                    schedule_id=schedule_id,
                    kind="error_replay",
                    source_question_id=row.question_id,
                    row=_clone_row(
                        row,
                        schedule_id=schedule_id,
                        question=row.question,
                        answer=row.answer,
                    ),
                    causal_margin=False,
                )
            )
    for row in sorted(anchor_rows, key=lambda value: value.question_id):
        schedule_id = f"v89_anchor_{row.question_id}"
        items.append(
            TrainingItemV89(
                schedule_id=schedule_id,
                kind="correct_anchor_replay",
                source_question_id=row.question_id,
                row=_clone_row(
                    row,
                    schedule_id=schedule_id,
                    question=row.question,
                    answer=row.answer,
                ),
                causal_margin=False,
            )
        )
    for row in development_smoke_rows_v89(config, rows):
        items.append(
            TrainingItemV89(
                schedule_id=row.question_id,
                kind="development_known_smoke",
                source_question_id=(
                    "q_000080"
                    if row.question_id == "v89_smoke_00"
                    else "q_000108"
                    if row.question_id == "v89_smoke_01"
                    else "q_000014"
                ),
                row=row,
                causal_margin=True,
            )
        )
    if len({item.schedule_id for item in items}) != len(items):
        raise ValueError("V89 schedule IDs are not unique")
    return tuple(items)


def training_inventory_v89(items: Sequence[TrainingItemV89]) -> list[dict[str, Any]]:
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
        }
        for item in items
    ]


def training_schedule_v89(
    items: Sequence[TrainingItemV89], *, seed: int = 890089, epochs: int = 3
) -> tuple[tuple[int, TrainingItemV89], ...]:
    schedule: list[tuple[int, TrainingItemV89]] = []
    for epoch in range(epochs):
        shuffled = sorted(items, key=lambda item: item.schedule_id)
        random.Random(seed + epoch).shuffle(shuffled)
        schedule.extend((epoch, item) for item in shuffled)
    return tuple(schedule)


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
            [nn.Identity() for _ in range(27)]
            + [_SyntheticLayer()]
            + [nn.Identity() for _ in range(7)]
        )


class _SyntheticGemma(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = _SyntheticLanguage()


def derive_lora_preflight_v89(config: Mapping[str, Any]) -> dict[str, Any]:
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
        raise RuntimeError("V89 synthetic LoRA installation failed")
    initialize_lora_adapter_state(installation, seed=int(bridge["initialization_seed"]))
    return {
        "bank_name": FRESH_BANK_NAME,
        "target_modules": list(installation.target_names),
        "parameter_count": installation.parameter_count,
        "initial_state_sha256": installation.state_sha256(),
        "base_projection_weight_shape": [1536, 2048],
        "lora_a_shape": [8, 2048],
        "lora_b_shape": [1536, 8],
        "lora_b_nonzero_count": sum(
            int(torch.count_nonzero(adapter.lora_b).item())
            for adapter in installation.adapters
        ),
        "exact_zero_output_at_initialization": True,
        "full_gemma_model_loaded": False,
    }


def lora_preflight_v89(config: Mapping[str, Any]) -> dict[str, Any]:
    result = derive_lora_preflight_v89(config)
    if (
        result["parameter_count"] != config["bridge"]["trainable_parameter_count"]
        or result["initial_state_sha256"]
        != config["bridge"]["expected_initial_state_sha256"]
        or result["lora_b_nonzero_count"] != 0
    ):
        raise RuntimeError("V89 deterministic exact-zero LoRA contract changed")
    return result


def authenticate_sources_v89(config: Mapping[str, Any]) -> dict[str, str]:
    sources = config["sources"]
    expected = {
        sources["runtime_config"]: sources["runtime_config_sha256"],
        sources["scene1_qa"]: sources["scene1_qa_sha256"],
        str(Path(sources["scene1_memory"]) / "memory.safetensors"): sources[
            "scene1_memory_tensor_sha256"
        ],
        str(Path(sources["scene1_memory"]) / "runtime_metadata.json"): sources[
            "scene1_memory_metadata_sha256"
        ],
        str(Path(sources["frozen_v85_checkpoint"]) / "adapter.safetensors"): sources[
            "frozen_v85_adapter_sha256"
        ],
        str(Path(sources["frozen_v85_checkpoint"]) / "runtime_metadata.json"): sources[
            "frozen_v85_metadata_sha256"
        ],
        str(Path(sources["parent_v86_checkpoint"]) / "bridge.safetensors"): sources[
            "parent_v86_bridge_sha256"
        ],
        str(Path(sources["parent_v86_checkpoint"]) / "runtime_metadata.json"): sources[
            "parent_v86_metadata_sha256"
        ],
        str(Path(sources["parent_v87_checkpoint"]) / "bridge.safetensors"): sources[
            "parent_v87_bridge_sha256"
        ],
        str(Path(sources["parent_v87_checkpoint"]) / "runtime_metadata.json"): sources[
            "parent_v87_metadata_sha256"
        ],
        sources["parent_v87_predictions"]: sources["parent_v87_predictions_sha256"],
        sources["parent_v88_config"]: sources["parent_v88_config_sha256"],
        sources["parent_v88_preregistration"]: sources["parent_v88_preregistration_sha256"],
        sources["parent_v88_cpu_preflight"]: sources["parent_v88_cpu_preflight_sha256"],
        sources["parent_v88_training_report"]: sources["parent_v88_training_report_sha256"],
        str(Path(sources["parent_v88_checkpoint"]) / "bridge.safetensors"): sources[
            "parent_v88_bridge_sha256"
        ],
        str(Path(sources["parent_v88_checkpoint"]) / "runtime_metadata.json"): sources[
            "parent_v88_metadata_sha256"
        ],
        sources["parent_v88_predictions"]: sources["parent_v88_predictions_sha256"],
        sources["parent_v88_evaluation"]: sources["parent_v88_evaluation_sha256"],
        sources["preflight_source"]: sources["preflight_source_sha256"],
        sources["training_source"]: sources["training_source_sha256"],
        sources["evaluation_source"]: sources["evaluation_source_sha256"],
    }
    observed: dict[str, str] = {}
    for path, expected_sha256 in expected.items():
        actual = sha256_file_v85(path)
        if actual != expected_sha256:
            raise ValueError(f"V89 pinned source changed: {path}")
        observed[str(path)] = actual
    snapshot = (
        Path.home()
        / ".cache/huggingface/hub/models--google--gemma-4-E2B-it/snapshots"
        / str(sources["model_revision"])
    )
    blob = (snapshot / "model.safetensors").resolve(strict=True)
    if blob.name != sources["model_blob_sha256_identity"]:
        raise ValueError("V89 local Gemma blob identity changed")
    model_config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    text_config = model_config.get("text_config")
    if not isinstance(text_config, Mapping) or (
        text_config.get("hidden_size") != 1536
        or text_config.get("num_hidden_layers") != 35
        or text_config.get("layer_types", [None] * 35)[27] != "sliding_attention"
        or text_config.get("num_attention_heads") != 8
        or text_config.get("head_dim") != 256
    ):
        raise ValueError("V89 pinned layer-27 topology changed")
    v85_metadata = strict_json_v89(
        Path(sources["frozen_v85_checkpoint"]) / "runtime_metadata.json"
    )
    modules = v85_metadata.get("lora_bank_wrapped_modules")
    occupied = [module for values in modules.values() for module in values] if isinstance(
        modules, Mapping
    ) else []
    occupied.extend((V86_TARGET_MODULE, V87_TARGET_MODULE, V88_TARGET_MODULE))
    if len(occupied) != len(set(occupied)) or TARGET_MODULE in occupied:
        raise ValueError("V89 fresh target overlaps a frozen bank")
    observed["gemma_model_blob_sha256_identity"] = blob.name
    return observed


def validate_parent_v88_v89(config: Mapping[str, Any]) -> dict[str, Any]:
    sources = config["sources"]
    training = strict_json_v89(sources["parent_v88_training_report"])
    candidate = strict_json_v89(
        Path(sources["parent_v88_checkpoint"]) / "runtime_metadata.json"
    )
    evaluation = strict_json_v89(sources["parent_v88_evaluation"])
    metrics = evaluation.get("metrics", {})
    gates = metrics.get("model_acceptance_gates", {})
    expected_false = {"all_scene1_canonical_accuracy_at_least_0_80"}
    false_gates = {key for key, value in gates.items() if value is False}
    if (
        training.get("optimizer_updates") != 188
        or training.get("micro_rows_consumed") != 1128
        or training.get("causal_margin_rows_consumed") != 20
        or not all(training.get("gates", {}).values())
        or candidate.get("state_sha256") != config["frozen_stack"]["v88_bank_state_sha256"]
        or candidate.get("questions_or_answers_serialized") is not False
        or candidate.get("training_metadata_serialized") is not False
        or candidate.get("augmentation_inventory_serialized") is not False
        or candidate.get("error_inventory_serialized") is not False
        or candidate.get("environmental_memory_serialized") is not False
        or evaluation.get("status") != "model_gates_fail_not_runtime_promotable"
        or metrics.get("canonical_type_specific", {}).get("correct") != 107
        or metrics.get("canonical_type_specific", {}).get("total") != 138
        or metrics.get("canonical_accuracy_by_answer_type", {}).get("attribute", {}).get(
            "correct"
        )
        != 11
        or metrics.get("canonical_accuracy_by_answer_type", {}).get("presence", {}).get(
            "correct"
        )
        != 21
        or metrics.get("canonical_accuracy_by_answer_type", {}).get(
            "spatial_relation", {}
        ).get("correct")
        != 64
        or metrics.get("generic_smoke", {}).get("correct") != 3
        or metrics.get("generic_smoke", {}).get("development_known_and_trained") is not True
        or metrics.get("generic_smoke", {}).get("held_out") is not False
        or false_gates != expected_false
        or evaluation.get("runtime_promotion_authorized") is not False
    ):
        raise ValueError("V89 parent V88 negative-result contract changed")
    return {
        "parent_v88_optimizer_updates": 188,
        "parent_v88_canonical_correct": 107,
        "parent_v88_canonical_total": 138,
        "parent_v88_canonical_accuracy": 107 / 138,
        "parent_v88_shortfall_to_fixed_gate": 4,
        "parent_v88_attribute_correct": 11,
        "parent_v88_attribute_total": 18,
        "parent_v88_presence_correct": 21,
        "parent_v88_presence_total": 22,
        "parent_v88_spatial_relation_correct": 64,
        "parent_v88_spatial_relation_total": 86,
        "parent_v88_development_known_smoke_correct": 3,
        "parent_v88_development_known_smoke_total": 3,
        "parent_v88_causal_control_passed": True,
        "parent_v88_only_failed_gate": "all_scene1_canonical_accuracy_at_least_0_80",
        "parent_v88_runtime_promoted": False,
        "parent_v88_state_sha256": candidate["state_sha256"],
        "parent_v88_mutated": False,
    }


def protocol_preflight_v89(config: Mapping[str, Any]) -> dict[str, Any]:
    rows = load_canonical_rows_v89(config)
    transitions, errors, anchors, error_rows, anchor_rows = derive_parent_behavior_v89(
        config, rows
    )
    transition_counts = Counter(record["transition"] for record in transitions)
    error_types = Counter(record["answer_type"] for record in errors)
    items = derive_training_items_v89(config, rows, error_rows, anchor_rows)
    inventory = training_inventory_v89(items)
    kind_counts = Counter(item.kind for item in items)
    schedule = training_schedule_v89(
        items,
        seed=int(config["training"]["row_order_seed"]),
        epochs=int(config["training"]["epochs"]),
    )
    schedule_hash = canonical_sha256_v85(
        [[epoch, item.schedule_id] for epoch, item in schedule]
    )
    per_epoch = Counter(epoch for epoch, _item in schedule)
    per_item = Counter(item.schedule_id for _epoch, item in schedule)
    causal = [item for _epoch, item in schedule if item.causal_margin]
    dataset = config["dataset"]
    if (
        len(rows) != 138
        or dict(sorted(transition_counts.items()))
        != dataset["parent_v87_to_v88_transition_counts"]
        or canonical_sha256_v85(transitions)
        != dataset["parent_v87_to_v88_transition_inventory_sha256"]
        or len(errors) != dataset["parent_v88_error_count"]
        or dict(sorted(error_types.items())) != dataset["parent_v88_error_type_counts"]
        or canonical_sha256_v85(errors) != dataset["parent_v88_error_inventory_sha256"]
        or len(anchors) != dataset["parent_v88_correct_anchor_count"]
        or canonical_sha256_v85(anchors)
        != dataset["parent_v88_correct_anchor_inventory_sha256"]
        or len(items) != 310
        or canonical_sha256_v85(inventory) != dataset["training_row_inventory_sha256"]
        or kind_counts
        != Counter(
            {
                "canonical": 138,
                "error_replay": 62,
                "correct_anchor_replay": 107,
                "development_known_smoke": 3,
            }
        )
        or len(schedule) != 930
        or schedule_hash != config["training"]["row_order_sha256"]
        or set(per_epoch.values()) != {310}
        or set(per_item.values()) != {3}
        or len(causal) != 18
    ):
        raise ValueError("V89 sealed behavior/inventory/schedule contract changed")
    causal_ids = Counter(item.schedule_id for item in causal)
    expected_causal = set(CANONICAL_CAUSAL_IDS + SMOKE_CAUSAL_IDS)
    if set(causal_ids) != expected_causal or set(causal_ids.values()) != {3}:
        raise ValueError("V89 exact causal schedule changed")
    canonical_exposures = Counter(
        item.source_question_id
        for item in items
        if item.kind in {"canonical", "error_replay", "correct_anchor_replay"}
    )
    if (
        {canonical_exposures[row.question_id] for row in error_rows} != {3}
        or {canonical_exposures[row.question_id] for row in anchor_rows} != {2}
    ):
        raise ValueError("V89 error/anchor exposure ratio changed")
    smoke = development_smoke_rows_v89(config, rows)
    if [row.question for row in smoke] != [
        "Is there a chair?",
        "What color is the bowl?",
        "Is the bowl left or right of the chair?",
    ] or [row.answer for row in smoke] != ["yes", "red", "left"]:
        raise ValueError("V89 development-known smoke representation changed")
    memory, memory_hash, metadata = load_scene1_memory_v86(config)
    zero = zero_payload_memory_v86(memory)
    zero_hash = prefix_sha256(zero)
    if memory_hash != EXPECTED_PREFIX_SHA256 or zero_hash == memory_hash:
        raise ValueError("V89 fixed memory or zero-payload control changed")
    return {
        "canonical_row_count": len(rows),
        "parent_v87_to_v88_transition_counts": dict(sorted(transition_counts.items())),
        "parent_v87_to_v88_transition_inventory_sha256": canonical_sha256_v85(
            transitions
        ),
        "parent_v87_to_v88_transition_inventory": list(transitions),
        "parent_v88_error_count": len(errors),
        "parent_v88_error_type_counts": dict(sorted(error_types.items())),
        "parent_v88_error_inventory_sha256": canonical_sha256_v85(errors),
        "parent_v88_error_inventory": list(errors),
        "parent_v88_correct_anchor_count": len(anchors),
        "parent_v88_correct_anchor_inventory_sha256": canonical_sha256_v85(anchors),
        "parent_v88_correct_anchor_inventory": list(anchors),
        "training_rows_per_epoch": len(items),
        "training_kind_counts": dict(sorted(kind_counts.items())),
        "training_row_inventory_sha256": canonical_sha256_v85(inventory),
        "training_row_inventory": inventory,
        "error_total_exposures_per_epoch": 3,
        "correct_total_exposures_per_epoch": 2,
        "development_known_smoke": {
            "trained": True,
            "held_out": False,
            "row_count": 3,
            "schedule_ids": list(SMOKE_CAUSAL_IDS),
            "all_receive_zero_payload_margin": True,
        },
        "schedule_rows": len(schedule),
        "rows_each_epoch": 310,
        "appearances_each_schedule_item": 3,
        "optimizer_updates": 155,
        "row_order_sha256": schedule_hash,
        "first_schedule_keys": [
            [epoch, item.schedule_id] for epoch, item in schedule[:3]
        ],
        "last_schedule_keys": [
            [epoch, item.schedule_id] for epoch, item in schedule[-3:]
        ],
        "causal_schedule_ids": sorted(expected_causal),
        "causal_margin_rows_total": len(causal),
        "fixed_memory_shape": list(memory.shape),
        "fixed_memory_dtype": str(memory.dtype),
        "fixed_memory_prefix_sha256": memory_hash,
        "memory_compiled_before_question": metadata["compiled_before_user_question"],
        "zero_payload_prefix_sha256": zero_hash,
        "zero_payload_preserves_native_boi": bool(torch.equal(zero[:, :1], memory[:, :1])),
        "zero_payload_preserves_native_eoi": bool(torch.equal(zero[:, -1:], memory[:, -1:])),
        "zero_payload_token_count": 736,
        "zero_payload_nonzero_scalar_count": int(torch.count_nonzero(zero[:, 1:-1]).item()),
        "selection_uses_training_metadata_only": True,
        "answers_available_to_training_only": True,
        "runtime_serializes_questions_or_answers": False,
        "runtime_serializes_training_error_or_anchor_inventory": False,
        "questions_tokenized": False,
        "full_gemma_model_loaded": False,
    }


def build_preregistration_v89(config_path: str | Path = CONFIG) -> dict[str, Any]:
    config = load_config_v89(config_path)
    report = {
        "artifact": PREREG_ARTIFACT,
        "schema_version": 89,
        "status": "sealed_after_v88_failure_before_first_v89_full_model_load",
        "config_path": Path(config_path).as_posix(),
        "config_sha256": sha256_file_v85(config_path),
        "authenticated_sources": authenticate_sources_v89(config),
        "parent_v88_evidence": validate_parent_v88_v89(config),
        "strict_input_contract": config["strict_input_contract"],
        "dataset_contract": config["dataset"],
        "frozen_stack": config["frozen_stack"],
        "bridge": config["bridge"],
        "training_protocol": config["training"],
        "fixed_unchanged_gates": config["gates"],
        "protocol_preflight": protocol_preflight_v89(config),
        "lora_cpu_preflight": lora_preflight_v89(config),
        "post_v88_training_set_development": True,
        "retention_aware_error_correction": True,
        "development_known_smoke_trained": True,
        "held_out_smoke_claim": False,
        "new_v89_behavior_scored": False,
        "answers_or_questions_serialized_in_runtime_candidate": False,
        "training_error_or_anchor_inventory_serialized_in_runtime_candidate": False,
        "full_gemma_model_loaded": False,
        "optimizer_constructed": False,
        "optimizer_updates": 0,
        "official_validation_loaded": False,
        "official_test_loaded": False,
        "deferred_final_loaded": False,
        "oracle_loaded": False,
        "runtime_promotion_authorized": False,
    }
    output, _sha = atomic_create_json_v85(config["outputs"]["preregistration"], report)
    report["output"] = output.relative_to(PROJECT_ROOT).as_posix()
    return report


def authenticate_preregistration_v89(
    config: Mapping[str, Any], *, config_path: str | Path = CONFIG
) -> dict[str, str]:
    path = resolve_v85(config["outputs"]["preregistration"])
    report = strict_json_v89(path)
    config_sha256 = sha256_file_v85(config_path)
    if (
        report.get("artifact") != PREREG_ARTIFACT
        or report.get("status")
        != "sealed_after_v88_failure_before_first_v89_full_model_load"
        or report.get("config_sha256") != config_sha256
        or report.get("post_v88_training_set_development") is not True
        or report.get("retention_aware_error_correction") is not True
        or report.get("development_known_smoke_trained") is not True
        or report.get("held_out_smoke_claim") is not False
        or report.get("new_v89_behavior_scored") is not False
        or report.get("full_gemma_model_loaded") is not False
        or report.get("optimizer_constructed") is not False
        or report.get("optimizer_updates") != 0
        or report.get("oracle_loaded") is not False
        or report.get("runtime_promotion_authorized") is not False
    ):
        raise ValueError("V89 preregistration changed")
    return {
        "config_sha256": config_sha256,
        "preregistration_sha256": sha256_file_v85(path),
    }


def run_cpu_preflight_v89(config_path: str | Path = CONFIG) -> dict[str, Any]:
    config = load_config_v89(config_path)
    prereg = authenticate_preregistration_v89(config, config_path=config_path)
    report = {
        "artifact": PREFLIGHT_ARTIFACT,
        "schema_version": 89,
        "status": "passed",
        "passed": True,
        **prereg,
        "authenticated_sources": authenticate_sources_v89(config),
        "parent_v88_evidence": validate_parent_v88_v89(config),
        "protocol_preflight": protocol_preflight_v89(config),
        "lora_preflight": lora_preflight_v89(config),
        "fixed_final_optimizer_updates": 155,
        "fixed_final_checkpoint_selection": "fixed_final_update_155",
        "all_138_canonical_rows_evaluated_unchanged": True,
        "development_known_smoke_trained": True,
        "held_out_smoke_claim": False,
        "same_fixed_memory_compiled_before_questions": True,
        "all_738_memory_slots_retained": True,
        "question_derived_environmental_tokens": 0,
        "question_conditioned_environmental_readout": False,
        "runtime_candidate_will_serialize_training_rows": False,
        "runtime_candidate_will_serialize_error_inventory": False,
        "runtime_candidate_will_serialize_anchor_inventory": False,
        "full_gemma_model_loaded": False,
        "optimizer_constructed": False,
        "optimizer_updates": 0,
        "new_v89_behavior_scored": False,
        "official_validation_loaded": False,
        "official_test_loaded": False,
        "deferred_final_loaded": False,
        "oracle_loaded": False,
        "runtime_promotion_authorized": False,
    }
    output, _sha = atomic_create_json_v85(config["outputs"]["cpu_preflight"], report)
    report["output"] = output.relative_to(PROJECT_ROOT).as_posix()
    return report


def authenticate_cpu_preflight_v89(
    config: Mapping[str, Any], *, config_path: str | Path = CONFIG
) -> dict[str, str]:
    prereg = authenticate_preregistration_v89(config, config_path=config_path)
    path = resolve_v85(config["outputs"]["cpu_preflight"])
    report = strict_json_v89(path)
    if (
        report.get("artifact") != PREFLIGHT_ARTIFACT
        or report.get("status") != "passed"
        or report.get("passed") is not True
        or report.get("config_sha256") != prereg["config_sha256"]
        or report.get("preregistration_sha256") != prereg["preregistration_sha256"]
        or report.get("development_known_smoke_trained") is not True
        or report.get("held_out_smoke_claim") is not False
        or report.get("full_gemma_model_loaded") is not False
        or report.get("optimizer_updates") != 0
        or report.get("new_v89_behavior_scored") is not False
        or report.get("oracle_loaded") is not False
        or report.get("runtime_promotion_authorized") is not False
    ):
        raise ValueError("V89 CPU preflight changed")
    return {**prereg, "cpu_preflight_sha256": sha256_file_v85(path)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preregister", "preflight"))
    parser.add_argument("--config", default=str(CONFIG))
    args = parser.parse_args(argv)
    result = (
        build_preregistration_v89(args.config)
        if args.command == "preregister"
        else run_cpu_preflight_v89(args.config)
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CANONICAL_CAUSAL_IDS",
    "CONFIG",
    "FRESH_BANK_NAME",
    "PREFLIGHT_ARTIFACT",
    "PREREG_ARTIFACT",
    "SCENE_ID",
    "SMOKE_CAUSAL_IDS",
    "TARGET_MODULE",
    "V86_BANK_NAME",
    "V86_TARGET_MODULE",
    "V87_BANK_NAME",
    "V87_TARGET_MODULE",
    "V88_BANK_NAME",
    "V88_TARGET_MODULE",
    "TrainingItemV89",
    "authenticate_cpu_preflight_v89",
    "authenticate_preregistration_v89",
    "authenticate_sources_v89",
    "build_preregistration_v89",
    "derive_lora_preflight_v89",
    "derive_parent_behavior_v89",
    "derive_training_items_v89",
    "development_smoke_rows_v89",
    "load_canonical_rows_v89",
    "load_config_v89",
    "lora_preflight_v89",
    "main",
    "protocol_preflight_v89",
    "run_cpu_preflight_v89",
    "strict_json_v89",
    "training_inventory_v89",
    "training_schedule_v89",
    "validate_parent_v88_v89",
]
