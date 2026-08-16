"""Seal and CPU-preflight the development-known V88 scene-one correction.

V88 is not a held-out experiment.  It is a fixed, post-V87 training-set
correction intended to make the scene-one local demonstration reliable while
preserving the strict continuous-memory interface.  The parent V87 failures,
deterministic training-only augmentation, schedule, sole fresh LoRA surface,
and unchanged acceptance gates are all bound before any V88 model load.

This module is model-free: it never loads Gemma weights, tokenizes a question,
constructs an optimizer, measures new behavior, or creates a runtime package.
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
    load_scene1_rows_v86,
    zero_payload_memory_v86,
)
from semantic_3d_chat.language.lora import (
    LoRASettings,
    initialize_lora_adapter_state,
    install_lora_adapters,
)
from semantic_3d_chat.language.prefix_injection import prefix_sha256
from semantic_3d_chat.training.train_question_control_v73 import RowV73

CONFIG: Final[Path] = Path("configs/experiments/gemma4_v88_scene1_augmented_demo.yaml")
SCENE_ID: Final[str] = "scene_000001"
V86_BANK_NAME: Final[str] = "v86_scene1_demo_bridge"
V86_TARGET_MODULE: Final[str] = "model.language_model.layers.34.mlp.up_proj"
V87_BANK_NAME: Final[str] = "v87_scene1_balanced_bridge"
V87_TARGET_MODULE: Final[str] = "model.language_model.layers.34.mlp.gate_proj"
FRESH_BANK_NAME: Final[str] = "v88_scene1_augmented_bridge"
TARGET_MODULE: Final[str] = "model.language_model.layers.27.self_attn.q_proj"
PREREG_ARTIFACT: Final[str] = "gemma4_v88_scene1_augmented_preregistration_v1"
PREFLIGHT_ARTIFACT: Final[str] = "gemma4_v88_scene1_augmented_cpu_preflight_v1"
CANONICAL_CAUSAL_IDS: Final[tuple[str, ...]] = (
    "q_000080",
    "q_000108",
    "q_000014",
)
AUGMENTED_CAUSAL_IDS: Final[tuple[str, ...]] = (
    "v88_inverse_q_000014",
    "v88_smoke_chair",
)
_HEX64: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{64}")
_RELATION: Final[re.Pattern[str]] = re.compile(
    r"^Is the (.+?) (left or right of|above or below|in front of or behind) the (.+?)\?$"
)
_INVERSE_ANSWER: Final[dict[str, str]] = {
    "left": "right",
    "right": "left",
    "above": "below",
    "below": "above",
    "in front": "behind",
    "behind": "in front",
}


@dataclass(frozen=True)
class TrainingItemV88:
    """One deterministic training-only schedule item."""

    schedule_id: str
    kind: str
    source_question_id: str
    row: RowV73
    causal_margin: bool


def strict_json_v88(path: str | Path) -> dict[str, Any]:
    source = resolve_v85(path)
    if source.is_symlink() or not source.is_file():
        raise FileNotFoundError(source)
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"V88 JSON must contain one object: {source}")
    return value


def _answer_class(answer: str) -> str:
    normalized = normalize_answer(answer)
    if not normalized:
        raise ValueError("V88 answer normalizes empty")
    return "answer_" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:20]


def load_canonical_rows_v88(config: Mapping[str, Any]) -> tuple[RowV73, ...]:
    """Project V88's explicit canonical fields into the proven V86 loader.

    The V86 loader is intentionally strict about the exact 138-row dataclass
    inventory.  V88 names those fields ``canonical_*`` to distinguish them
    from the 282 training schedule items, so this wrapper performs the only
    schema projection and preserves the original authentication logic.
    """

    dataset = config["dataset"]
    projected = dict(config)
    projected["dataset"] = {
        "row_count": int(dataset["canonical_row_count"]),
        "row_inventory_sha256": str(dataset["canonical_row_inventory_sha256"]),
    }
    return load_scene1_rows_v86(projected)


def load_config_v88(path: str | Path = CONFIG) -> dict[str, Any]:
    source = resolve_v85(path)
    if source.is_symlink() or not source.is_file():
        raise FileNotFoundError(source)
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or set(payload) != {"v88"}:
        raise ValueError("V88 config must contain exactly one v88 mapping")
    config = payload["v88"]
    if not isinstance(config, Mapping):
        raise TypeError("V88 config payload must be a mapping")
    if any(
        config.get(key) != value
        for key, value in {
            "schema_version": 88,
            "artifact": "gemma4_v88_scene1_augmented_direct_memory_overfit_v1",
            "status": "preregistered_before_full_model_load",
            "seed": 880088,
        }.items()
    ):
        raise ValueError("V88 identity changed")
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
        raise ValueError("V88 strict continuous-memory contract changed")
    dataset = config.get("dataset")
    expected_dataset = {
        "scene_id": SCENE_ID,
        "canonical_row_count": 138,
        "canonical_row_inventory_sha256": (
            "9919ff1bee4611dce4132d79fa50f6f6b4ace567a6df780a2e0e21bd88237a8e"
        ),
        "parent_v87_error_count": 35,
        "parent_v87_error_type_counts": {
            "attribute": 11,
            "spatial_relation": 23,
            "support": 1,
        },
        "canonical_rows_per_epoch": 138,
        "hard_error_replay_rows_per_epoch": 35,
        "inverse_spatial_rows_per_epoch": 86,
        "alternate_attribute_rows_per_epoch": 9,
        "alternate_presence_rows_per_epoch": 13,
        "new_smoke_rows_per_epoch": 1,
        "total_rows_per_epoch": 282,
        "augmentation_uses_training_metadata_only": True,
        "answer_metadata_training_only": True,
        "runtime_serializes_questions_or_answers": False,
        "runtime_serializes_augmentation_inventory": False,
        "runtime_serializes_error_inventory": False,
    }
    if not isinstance(dataset, Mapping) or any(
        dataset.get(key) != value for key, value in expected_dataset.items()
    ):
        raise ValueError("V88 fixed dataset/augmentation counts changed")
    for key in ("parent_v87_error_inventory_sha256", "augmented_row_inventory_sha256"):
        if not isinstance(dataset.get(key), str) or _HEX64.fullmatch(dataset[key]) is None:
            raise ValueError(f"V88 {key} is not sealed")
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
        "total_frozen_bank_count": 9,
        "merged_weights": False,
    }
    if not isinstance(frozen, Mapping) or any(
        frozen.get(key) != value for key, value in expected_frozen.items()
    ):
        raise ValueError("V88 frozen V85+V86+V87 stack changed")
    bridge = config.get("bridge")
    expected_bridge = {
        "bank_name": FRESH_BANK_NAME,
        "target_module": TARGET_MODULE,
        "target_layer_type": "sliding_attention",
        "target_in_features": 1536,
        "target_out_features": 2048,
        "rank": 16,
        "alpha": 32.0,
        "dropout": 0.0,
        "trainable_parameter_count": 57344,
        "initialization_algorithm": "cpu_kaiming_uniform_a_exact_zero_b",
        "initialization_seed": 880088,
        "disjoint_from_all_frozen_banks": True,
    }
    if not isinstance(bridge, Mapping) or any(
        bridge.get(key) != value for key, value in expected_bridge.items()
    ):
        raise ValueError("V88 sole fresh bridge contract changed")
    if not isinstance(bridge.get("expected_initial_state_sha256"), str) or _HEX64.fullmatch(
        bridge["expected_initial_state_sha256"]
    ) is None:
        raise ValueError("V88 initial bridge state is not sealed")
    training = config.get("training")
    expected_training = {
        "optimizer": "AdamW",
        "epochs": 4,
        "rows_per_epoch": 282,
        "microbatch_size": 1,
        "gradient_accumulation_rows": 6,
        "optimizer_updates": 188,
        "row_order": "deterministic_allrow_seeded_shuffle",
        "row_order_seed": 880088,
        "learning_rate": 0.0005,
        "weight_decay": 0.0,
        "gradient_clip_norm": 1.0,
        "answer_ce_weight": 1.0,
        "class_weighting": "none",
        "zero_payload_margin_weight": 1.0,
        "zero_payload_target_margin_nll": 0.5,
        "canonical_causal_question_ids": list(CANONICAL_CAUSAL_IDS),
        "augmented_causal_schedule_ids": list(AUGMENTED_CAUSAL_IDS),
        "zero_payload_preserves_native_boi_eoi": True,
        "zero_payload_zeros_exactly_736_interior_tokens": True,
        "causal_rows_per_epoch": 5,
        "total_causal_margin_rows": 20,
        "checkpoint_selection": "fixed_final_update_188",
        "intermediate_behavior_selection": False,
    }
    if not isinstance(training, Mapping) or any(
        training.get(key) != value for key, value in expected_training.items()
    ):
        raise ValueError("V88 fixed schedule/loss changed")
    if not isinstance(training.get("row_order_sha256"), str) or _HEX64.fullmatch(
        training["row_order_sha256"]
    ) is None:
        raise ValueError("V88 row order is not sealed")
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
        raise ValueError("V88 unchanged acceptance gates changed")
    if gates.get("live_smoke_questions") != [
        {"question": "Is there a chair?", "expected": "yes"},
        {"question": "What color is the bowl?", "expected": "red"},
        {"question": "Is the bowl left or right of the chair?", "expected": "left"},
    ]:
        raise ValueError("V88 development-known smoke changed")
    scope = config.get("scope")
    if scope != {
        "post_v87_training_set_development": True,
        "single_scene_overfit_demonstration": True,
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
        raise ValueError("V88 protected development-only scope changed")
    sources = config.get("sources")
    if not isinstance(sources, Mapping):
        raise TypeError("V88 sources must be a mapping")
    for key in ("preflight_source_sha256", "training_source_sha256", "evaluation_source_sha256"):
        if not isinstance(sources.get(key), str) or _HEX64.fullmatch(sources[key]) is None:
            raise ValueError(f"V88 {key} is not sealed")
    return dict(config)


def derive_v87_error_inventory_v88(
    config: Mapping[str, Any], rows: Sequence[RowV73]
) -> tuple[tuple[dict[str, Any], ...], tuple[RowV73, ...]]:
    payload = strict_json_v88(config["sources"]["parent_v87_predictions"])
    records = payload.get("records")
    if (
        payload.get("artifact") != "gemma4_v87_scene1_balanced_predictions_v1"
        or payload.get("row_count") != 138
        or not isinstance(records, list)
        or len(records) != 138
    ):
        raise ValueError("V88 parent V87 prediction inventory changed")
    row_by_id = {row.question_id: row for row in rows}
    record_by_id = {str(record.get("question_id")): record for record in records}
    if set(row_by_id) != set(record_by_id):
        raise ValueError("V88 parent V87 predictions do not cover canonical rows")
    inventory: list[dict[str, Any]] = []
    hard_rows: list[RowV73] = []
    for question_id in sorted(row_by_id):
        row = row_by_id[question_id]
        record = record_by_id[question_id]
        prediction = str(record["prediction"])
        if canonical_type_specific_match(row.answer_type, prediction, row.answer):
            continue
        hard_rows.append(row)
        inventory.append(
            {
                "question_id": question_id,
                "answer_type": row.answer_type,
                "reference_answer": row.answer,
                "prediction": prediction,
                "canonical_prediction": canonical_answer_key(row.answer_type, prediction),
                "correct_mean_nll": float(record["correct_mean_nll"]),
            }
        )
    return tuple(inventory), tuple(hard_rows)


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


def derive_training_items_v88(
    config: Mapping[str, Any], rows: Sequence[RowV73], hard_rows: Sequence[RowV73]
) -> tuple[TrainingItemV88, ...]:
    """Derive all 282 per-epoch rows exclusively from training metadata."""

    items: list[TrainingItemV88] = []
    represented: dict[tuple[str, str], str] = {}

    def append(
        row: RowV73,
        *,
        schedule_id: str,
        kind: str,
        source_question_id: str,
        question: str | None = None,
        answer: str | None = None,
        causal_margin: bool = False,
    ) -> None:
        candidate = _clone_row(
            row,
            schedule_id=schedule_id,
            question=row.question if question is None else question,
            answer=row.answer if answer is None else answer,
        )
        key = (candidate.question, candidate.answer)
        if key in represented:
            raise ValueError(f"V88 duplicate training semantic row: {key}")
        represented[key] = schedule_id
        items.append(
            TrainingItemV88(
                schedule_id=schedule_id,
                kind=kind,
                source_question_id=source_question_id,
                row=candidate,
                causal_margin=causal_margin,
            )
        )

    canonical = sorted(rows, key=lambda row: row.question_id)
    for row in canonical:
        append(
            row,
            schedule_id=row.question_id,
            kind="canonical",
            source_question_id=row.question_id,
            causal_margin=row.question_id in CANONICAL_CAUSAL_IDS,
        )

    # Hard-error replay is intentionally an extra occurrence, so its schedule
    # ID differs even though its question/answer pair is the same.  Add it
    # without the semantic-dedup guard used for novel paraphrases.
    for row in sorted(hard_rows, key=lambda value: value.question_id):
        schedule_id = f"v88_hard_{row.question_id}"
        candidate = _clone_row(
            row, schedule_id=schedule_id, question=row.question, answer=row.answer
        )
        items.append(
            TrainingItemV88(
                schedule_id=schedule_id,
                kind="hard_error_replay",
                source_question_id=row.question_id,
                row=candidate,
                causal_margin=False,
            )
        )

    for row in canonical:
        if row.answer_type != "spatial_relation":
            continue
        match = _RELATION.fullmatch(row.question)
        if match is None or row.answer not in _INVERSE_ANSWER:
            raise ValueError(f"V88 cannot invert canonical relation: {row.question_id}")
        append(
            row,
            schedule_id=f"v88_inverse_{row.question_id}",
            kind="inverse_spatial",
            source_question_id=row.question_id,
            question=f"Is the {match.group(3)} {match.group(2)} the {match.group(1)}?",
            answer=_INVERSE_ANSWER[row.answer],
            causal_margin=row.question_id == "q_000014",
        )

    attribute_candidates: dict[tuple[str, str], RowV73] = {}
    for row in canonical:
        if row.answer_type != "attribute":
            continue
        match = re.fullmatch(r"What color is the (.+)\?", row.question)
        if match is None:
            match = re.fullmatch(r"Tell me the (.+)'s color\.", row.question)
        if match is None:
            raise ValueError(f"V88 cannot paraphrase attribute: {row.question_id}")
        question = f"Which color is the {match.group(1)}?"
        attribute_candidates.setdefault((question, row.answer), row)
    for ordinal, ((question, answer), row) in enumerate(sorted(attribute_candidates.items())):
        append(
            row,
            schedule_id=f"v88_attribute_alt_{ordinal:02d}",
            kind="alternate_attribute",
            source_question_id=row.question_id,
            question=question,
            answer=answer,
        )

    presence_candidates: dict[tuple[str, str], RowV73] = {}
    for row in canonical:
        if row.answer_type != "presence":
            continue
        match = re.fullmatch(r"Can you find a (.+)\?", row.question)
        if match is None:
            match = re.fullmatch(r"Is there a (.+) in the room\?", row.question)
        if match is None:
            raise ValueError(f"V88 cannot paraphrase presence: {row.question_id}")
        question = f"Is a {match.group(1)} present?"
        presence_candidates.setdefault((question, row.answer), row)
    for ordinal, ((question, answer), row) in enumerate(sorted(presence_candidates.items())):
        append(
            row,
            schedule_id=f"v88_presence_alt_{ordinal:02d}",
            kind="alternate_presence",
            source_question_id=row.question_id,
            question=question,
            answer=answer,
        )

    smoke_source = next(row for row in canonical if row.question_id == "q_000080")
    smoke_resolution: dict[str, str] = {}
    for ordinal, raw in enumerate(config["gates"]["live_smoke_questions"]):
        question = str(raw["question"])
        answer = normalize_answer(raw["expected"])
        existing = represented.get((question, answer))
        if existing is not None:
            smoke_resolution[question] = existing
            continue
        if ordinal != 0:
            raise ValueError("V88 only the chair smoke may require a novel row")
        append(
            smoke_source,
            schedule_id="v88_smoke_chair",
            kind="development_known_smoke",
            source_question_id="q_000080",
            question=question,
            answer=answer,
            causal_margin=True,
        )
        smoke_resolution[question] = "v88_smoke_chair"
    expected_resolution = {
        "Is there a chair?": "v88_smoke_chair",
        "What color is the bowl?": "q_000108",
        "Is the bowl left or right of the chair?": "v88_inverse_q_000014",
    }
    if smoke_resolution != expected_resolution:
        raise ValueError("V88 development-known smoke deduplication changed")
    return tuple(items)


def training_inventory_v88(items: Sequence[TrainingItemV88]) -> list[dict[str, Any]]:
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


def training_schedule_v88(
    items: Sequence[TrainingItemV88], *, seed: int = 880088, epochs: int = 4
) -> tuple[tuple[int, TrainingItemV88], ...]:
    schedule: list[tuple[int, TrainingItemV88]] = []
    for epoch in range(epochs):
        shuffled = sorted(items, key=lambda item: item.schedule_id)
        random.Random(seed + epoch).shuffle(shuffled)
        schedule.extend((epoch, item) for item in shuffled)
    return tuple(schedule)


class _SyntheticAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(1536, 2048, bias=False, dtype=torch.bfloat16)


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


def derive_lora_preflight_v88(config: Mapping[str, Any]) -> dict[str, Any]:
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
        raise RuntimeError("V88 synthetic LoRA installation failed")
    initialize_lora_adapter_state(installation, seed=int(bridge["initialization_seed"]))
    return {
        "bank_name": FRESH_BANK_NAME,
        "target_modules": list(installation.target_names),
        "parameter_count": installation.parameter_count,
        "initial_state_sha256": installation.state_sha256(),
        "base_projection_weight_shape": [2048, 1536],
        "lora_a_shape": [16, 1536],
        "lora_b_shape": [2048, 16],
        "lora_b_nonzero_count": sum(
            int(torch.count_nonzero(adapter.lora_b).item())
            for adapter in installation.adapters
        ),
        "exact_zero_output_at_initialization": True,
        "full_gemma_model_loaded": False,
    }


def lora_preflight_v88(config: Mapping[str, Any]) -> dict[str, Any]:
    result = derive_lora_preflight_v88(config)
    if (
        result["parameter_count"] != config["bridge"]["trainable_parameter_count"]
        or result["initial_state_sha256"]
        != config["bridge"]["expected_initial_state_sha256"]
        or result["lora_b_nonzero_count"] != 0
    ):
        raise RuntimeError("V88 deterministic exact-zero LoRA contract changed")
    return result


def authenticate_sources_v88(config: Mapping[str, Any]) -> dict[str, str]:
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
        sources["parent_v87_config"]: sources["parent_v87_config_sha256"],
        sources["parent_v87_preregistration"]: sources["parent_v87_preregistration_sha256"],
        sources["parent_v87_cpu_preflight"]: sources["parent_v87_cpu_preflight_sha256"],
        sources["parent_v87_training_report"]: sources["parent_v87_training_report_sha256"],
        str(Path(sources["parent_v87_checkpoint"]) / "bridge.safetensors"): sources[
            "parent_v87_bridge_sha256"
        ],
        str(Path(sources["parent_v87_checkpoint"]) / "runtime_metadata.json"): sources[
            "parent_v87_metadata_sha256"
        ],
        sources["parent_v87_predictions"]: sources["parent_v87_predictions_sha256"],
        sources["parent_v87_evaluation"]: sources["parent_v87_evaluation_sha256"],
        sources["preflight_source"]: sources["preflight_source_sha256"],
        sources["training_source"]: sources["training_source_sha256"],
        sources["evaluation_source"]: sources["evaluation_source_sha256"],
    }
    observed: dict[str, str] = {}
    for path, expected_sha256 in expected.items():
        actual = sha256_file_v85(path)
        if actual != expected_sha256:
            raise ValueError(f"V88 pinned source changed: {path}")
        observed[str(path)] = actual
    snapshot = (
        Path.home()
        / ".cache/huggingface/hub/models--google--gemma-4-E2B-it/snapshots"
        / str(sources["model_revision"])
    )
    blob = (snapshot / "model.safetensors").resolve(strict=True)
    if blob.name != sources["model_blob_sha256_identity"]:
        raise ValueError("V88 local Gemma blob identity changed")
    model_config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    text_config = model_config.get("text_config")
    if not isinstance(text_config, Mapping) or (
        text_config.get("hidden_size") != 1536
        or text_config.get("num_hidden_layers") != 35
        or text_config.get("layer_types", [None] * 35)[27] != "sliding_attention"
    ):
        raise ValueError("V88 pinned layer-27 topology changed")
    v85_metadata = strict_json_v88(
        Path(sources["frozen_v85_checkpoint"]) / "runtime_metadata.json"
    )
    modules = v85_metadata.get("lora_bank_wrapped_modules")
    occupied = [module for values in modules.values() for module in values] if isinstance(
        modules, Mapping
    ) else []
    occupied.extend((V86_TARGET_MODULE, V87_TARGET_MODULE))
    if len(occupied) != len(set(occupied)) or TARGET_MODULE in occupied:
        raise ValueError("V88 fresh target overlaps a frozen bank")
    observed["gemma_model_blob_sha256_identity"] = blob.name
    return observed


def validate_parent_v87_v88(config: Mapping[str, Any]) -> dict[str, Any]:
    sources = config["sources"]
    training = strict_json_v88(sources["parent_v87_training_report"])
    candidate = strict_json_v88(
        Path(sources["parent_v87_checkpoint"]) / "runtime_metadata.json"
    )
    evaluation = strict_json_v88(sources["parent_v87_evaluation"])
    metrics = evaluation.get("metrics", {})
    gates = metrics.get("model_acceptance_gates", {})
    if (
        training.get("optimizer_updates") != 184
        or not all(training.get("gates", {}).values())
        or candidate.get("state_sha256") != config["frozen_stack"]["v87_bank_state_sha256"]
        or candidate.get("questions_or_answers_serialized") is not False
        or candidate.get("environmental_memory_serialized") is not False
        or evaluation.get("status") != "model_gates_fail_not_runtime_promotable"
        or metrics.get("canonical_type_specific", {}).get("correct") != 103
        or metrics.get("canonical_type_specific", {}).get("total") != 138
        or metrics.get("canonical_accuracy_by_answer_type", {}).get("attribute", {}).get(
            "correct"
        )
        != 7
        or metrics.get("canonical_accuracy_by_answer_type", {}).get(
            "spatial_relation", {}
        ).get("correct")
        != 63
        or metrics.get("generic_smoke", {}).get("correct") != 0
        or gates.get("all_scene1_canonical_accuracy_at_least_0_80") is not False
        or gates.get("attribute_accuracy_at_least_0_50") is not False
        or gates.get("generic_live_smoke_exactly_3_of_3") is not False
        or gates.get("presence_accuracy_at_least_0_75") is not True
        or gates.get("spatial_relation_accuracy_at_least_0_60") is not True
        or gates.get("causal_correct_memory_mean_nll_below_zero_payload") is not True
        or evaluation.get("runtime_promotion_authorized") is not False
    ):
        raise ValueError("V88 parent V87 negative-result contract changed")
    return {
        "parent_v87_optimizer_updates": 184,
        "parent_v87_canonical_correct": 103,
        "parent_v87_canonical_total": 138,
        "parent_v87_canonical_accuracy": 103 / 138,
        "parent_v87_attribute_correct": 7,
        "parent_v87_attribute_total": 18,
        "parent_v87_presence_correct": 22,
        "parent_v87_presence_total": 22,
        "parent_v87_spatial_relation_correct": 63,
        "parent_v87_spatial_relation_total": 86,
        "parent_v87_development_known_smoke_correct": 0,
        "parent_v87_development_known_smoke_total": 3,
        "parent_v87_runtime_promoted": False,
        "parent_v87_state_sha256": candidate["state_sha256"],
        "parent_v87_mutated": False,
    }


def protocol_preflight_v88(config: Mapping[str, Any]) -> dict[str, Any]:
    rows = load_canonical_rows_v88(config)
    errors, hard_rows = derive_v87_error_inventory_v88(config, rows)
    error_hash = canonical_sha256_v85(errors)
    error_types = Counter(str(record["answer_type"]) for record in errors)
    items = derive_training_items_v88(config, rows, hard_rows)
    inventory = training_inventory_v88(items)
    inventory_hash = canonical_sha256_v85(inventory)
    kind_counts = Counter(item.kind for item in items)
    schedule = training_schedule_v88(
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
    if (
        len(rows) != 138
        or len(errors) != config["dataset"]["parent_v87_error_count"]
        or dict(sorted(error_types.items())) != config["dataset"]["parent_v87_error_type_counts"]
        or error_hash != config["dataset"]["parent_v87_error_inventory_sha256"]
        or len(items) != config["dataset"]["total_rows_per_epoch"]
        or inventory_hash != config["dataset"]["augmented_row_inventory_sha256"]
        or kind_counts
        != Counter(
            {
                "canonical": 138,
                "hard_error_replay": 35,
                "inverse_spatial": 86,
                "alternate_attribute": 9,
                "alternate_presence": 13,
                "development_known_smoke": 1,
            }
        )
        or len(schedule) != 1128
        or schedule_hash != config["training"]["row_order_sha256"]
        or set(per_epoch.values()) != {282}
        or set(per_item.values()) != {4}
        or len(causal) != 20
    ):
        raise ValueError("V88 sealed error/augmentation/schedule contract changed")
    causal_ids = Counter(item.schedule_id for item in causal)
    expected_causal_ids = set(CANONICAL_CAUSAL_IDS + AUGMENTED_CAUSAL_IDS)
    if set(causal_ids) != expected_causal_ids or set(causal_ids.values()) != {4}:
        raise ValueError("V88 exact causal schedule changed")
    memory, memory_hash, metadata = load_scene1_memory_v86(config)
    zero = zero_payload_memory_v86(memory)
    zero_hash = prefix_sha256(zero)
    if memory_hash != EXPECTED_PREFIX_SHA256 or zero_hash == memory_hash:
        raise ValueError("V88 fixed memory or zero-payload control changed")
    return {
        "canonical_row_count": len(rows),
        "parent_v87_error_count": len(errors),
        "parent_v87_error_type_counts": dict(sorted(error_types.items())),
        "parent_v87_error_inventory_sha256": error_hash,
        "parent_v87_error_inventory": list(errors),
        "training_rows_per_epoch": len(items),
        "training_kind_counts": dict(sorted(kind_counts.items())),
        "augmented_row_inventory_sha256": inventory_hash,
        "augmented_row_inventory": inventory,
        "development_known_smoke": {
            "trained": True,
            "held_out": False,
            "three_questions_represented": True,
            "new_rows_after_deduplication": 1,
            "canonical_dedup_schedule_id": "q_000108",
            "inverse_dedup_schedule_id": "v88_inverse_q_000014",
            "new_schedule_id": "v88_smoke_chair",
        },
        "schedule_rows": len(schedule),
        "rows_each_epoch": 282,
        "appearances_each_schedule_item": 4,
        "optimizer_updates": 188,
        "row_order_sha256": schedule_hash,
        "first_schedule_keys": [
            [epoch, item.schedule_id] for epoch, item in schedule[:3]
        ],
        "last_schedule_keys": [
            [epoch, item.schedule_id] for epoch, item in schedule[-3:]
        ],
        "causal_schedule_ids": sorted(expected_causal_ids),
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
        "augmentation_uses_training_metadata_only": True,
        "answers_available_to_training_only": True,
        "runtime_serializes_questions_or_answers": False,
        "runtime_serializes_augmentation_or_error_inventory": False,
        "questions_tokenized": False,
        "full_gemma_model_loaded": False,
    }


def build_preregistration_v88(config_path: str | Path = CONFIG) -> dict[str, Any]:
    config = load_config_v88(config_path)
    report = {
        "artifact": PREREG_ARTIFACT,
        "schema_version": 88,
        "status": "sealed_after_v87_failure_before_first_v88_full_model_load",
        "config_path": Path(config_path).as_posix(),
        "config_sha256": sha256_file_v85(config_path),
        "authenticated_sources": authenticate_sources_v88(config),
        "parent_v87_evidence": validate_parent_v87_v88(config),
        "strict_input_contract": config["strict_input_contract"],
        "dataset_contract": config["dataset"],
        "frozen_stack": config["frozen_stack"],
        "bridge": config["bridge"],
        "training_protocol": config["training"],
        "fixed_unchanged_gates": config["gates"],
        "protocol_preflight": protocol_preflight_v88(config),
        "lora_cpu_preflight": lora_preflight_v88(config),
        "post_v87_training_set_development": True,
        "development_known_smoke_trained": True,
        "held_out_smoke_claim": False,
        "new_v88_behavior_scored": False,
        "answers_or_questions_serialized_in_runtime_candidate": False,
        "augmentation_or_error_inventory_serialized_in_runtime_candidate": False,
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


def authenticate_preregistration_v88(
    config: Mapping[str, Any], *, config_path: str | Path = CONFIG
) -> dict[str, str]:
    path = resolve_v85(config["outputs"]["preregistration"])
    report = strict_json_v88(path)
    config_sha256 = sha256_file_v85(config_path)
    if (
        report.get("artifact") != PREREG_ARTIFACT
        or report.get("status")
        != "sealed_after_v87_failure_before_first_v88_full_model_load"
        or report.get("config_sha256") != config_sha256
        or report.get("post_v87_training_set_development") is not True
        or report.get("development_known_smoke_trained") is not True
        or report.get("held_out_smoke_claim") is not False
        or report.get("new_v88_behavior_scored") is not False
        or report.get("full_gemma_model_loaded") is not False
        or report.get("optimizer_constructed") is not False
        or report.get("optimizer_updates") != 0
        or report.get("oracle_loaded") is not False
        or report.get("runtime_promotion_authorized") is not False
    ):
        raise ValueError("V88 preregistration changed")
    return {
        "config_sha256": config_sha256,
        "preregistration_sha256": sha256_file_v85(path),
    }


def run_cpu_preflight_v88(config_path: str | Path = CONFIG) -> dict[str, Any]:
    config = load_config_v88(config_path)
    prereg = authenticate_preregistration_v88(config, config_path=config_path)
    report = {
        "artifact": PREFLIGHT_ARTIFACT,
        "schema_version": 88,
        "status": "passed",
        "passed": True,
        **prereg,
        "authenticated_sources": authenticate_sources_v88(config),
        "parent_v87_evidence": validate_parent_v87_v88(config),
        "protocol_preflight": protocol_preflight_v88(config),
        "lora_preflight": lora_preflight_v88(config),
        "fixed_final_optimizer_updates": 188,
        "fixed_final_checkpoint_selection": "fixed_final_update_188",
        "all_138_canonical_rows_evaluated_unchanged": True,
        "development_known_smoke_trained": True,
        "held_out_smoke_claim": False,
        "same_fixed_memory_compiled_before_questions": True,
        "all_738_memory_slots_retained": True,
        "question_derived_environmental_tokens": 0,
        "question_conditioned_environmental_readout": False,
        "runtime_candidate_will_serialize_training_rows": False,
        "runtime_candidate_will_serialize_error_inventory": False,
        "runtime_candidate_will_serialize_augmentation_inventory": False,
        "full_gemma_model_loaded": False,
        "optimizer_constructed": False,
        "optimizer_updates": 0,
        "new_v88_behavior_scored": False,
        "official_validation_loaded": False,
        "official_test_loaded": False,
        "deferred_final_loaded": False,
        "oracle_loaded": False,
        "runtime_promotion_authorized": False,
    }
    output, _sha = atomic_create_json_v85(config["outputs"]["cpu_preflight"], report)
    report["output"] = output.relative_to(PROJECT_ROOT).as_posix()
    return report


def authenticate_cpu_preflight_v88(
    config: Mapping[str, Any], *, config_path: str | Path = CONFIG
) -> dict[str, str]:
    prereg = authenticate_preregistration_v88(config, config_path=config_path)
    path = resolve_v85(config["outputs"]["cpu_preflight"])
    report = strict_json_v88(path)
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
        or report.get("new_v88_behavior_scored") is not False
        or report.get("oracle_loaded") is not False
        or report.get("runtime_promotion_authorized") is not False
    ):
        raise ValueError("V88 CPU preflight changed")
    return {**prereg, "cpu_preflight_sha256": sha256_file_v85(path)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preregister", "preflight"))
    parser.add_argument("--config", default=str(CONFIG))
    args = parser.parse_args(argv)
    result = (
        build_preregistration_v88(args.config)
        if args.command == "preregister"
        else run_cpu_preflight_v88(args.config)
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AUGMENTED_CAUSAL_IDS",
    "CANONICAL_CAUSAL_IDS",
    "CONFIG",
    "FRESH_BANK_NAME",
    "PREFLIGHT_ARTIFACT",
    "PREREG_ARTIFACT",
    "SCENE_ID",
    "TARGET_MODULE",
    "V86_BANK_NAME",
    "V86_TARGET_MODULE",
    "V87_BANK_NAME",
    "V87_TARGET_MODULE",
    "TrainingItemV88",
    "authenticate_cpu_preflight_v88",
    "authenticate_preregistration_v88",
    "authenticate_sources_v88",
    "build_preregistration_v88",
    "derive_lora_preflight_v88",
    "derive_training_items_v88",
    "derive_v87_error_inventory_v88",
    "load_canonical_rows_v88",
    "load_config_v88",
    "lora_preflight_v88",
    "main",
    "protocol_preflight_v88",
    "run_cpu_preflight_v88",
    "strict_json_v88",
    "training_inventory_v88",
    "training_schedule_v88",
    "validate_parent_v87_v88",
]
