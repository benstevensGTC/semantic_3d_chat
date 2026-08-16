"""Seal and CPU-preflight the post-V86 class-balanced V87 experiment.

V87 is an explicitly training-set-only development follow-up to V86's sealed
62.32% result.  It never mutates or selects V86.  Instead it freezes the exact
V86 bridge, installs one disjoint zero-output rank-8 bank, and balances the
aggregate CE mass of all nineteen opaque answer classes while consuming every
one of the 138 rows exactly once per epoch.

This module is model-free: no Gemma weights, optimizer, generation, or new
behavior measurement is performed by preregistration or CPU preflight.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import torch
import yaml
from torch import nn

from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.evaluation.v85_strict_multiscene_preflight import (
    atomic_create_json_v85,
    canonical_sha256_v85,
    resolve_v85,
    sha256_file_v85,
)
from semantic_3d_chat.evaluation.v86_scene1_demo_preflight import (
    EXPECTED_PREFIX_SHA256,
    causal_rows_v86,
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

CONFIG: Final[Path] = Path("configs/experiments/gemma4_v87_scene1_balanced_demo.yaml")
SCENE_ID: Final[str] = "scene_000001"
PARENT_BANK_NAME: Final[str] = "v86_scene1_demo_bridge"
PARENT_TARGET_MODULE: Final[str] = "model.language_model.layers.34.mlp.up_proj"
FRESH_BANK_NAME: Final[str] = "v87_scene1_balanced_bridge"
TARGET_MODULE: Final[str] = "model.language_model.layers.34.mlp.gate_proj"
PREREG_ARTIFACT: Final[str] = "gemma4_v87_scene1_balanced_preregistration_v1"
PREFLIGHT_ARTIFACT: Final[str] = "gemma4_v87_scene1_balanced_cpu_preflight_v1"
CAUSAL_IDS: Final[tuple[str, ...]] = ("q_000080", "q_000108", "q_000014")
_HEX64: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{64}")


def _strict_json(path: str | Path) -> dict[str, Any]:
    source = resolve_v85(path)
    if source.is_symlink() or not source.is_file():
        raise FileNotFoundError(source)
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"V87 JSON must contain one object: {source}")
    return value


def load_config_v87(path: str | Path = CONFIG) -> dict[str, Any]:
    source = resolve_v85(path)
    if source.is_symlink() or not source.is_file():
        raise FileNotFoundError(source)
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or set(payload) != {"v87"}:
        raise ValueError("V87 config must contain exactly one v87 mapping")
    config = payload["v87"]
    if not isinstance(config, Mapping):
        raise TypeError("V87 config payload must be a mapping")
    if (
        config.get("schema_version") != 87
        or config.get("artifact") != "gemma4_v87_scene1_balanced_direct_memory_overfit_v1"
        or config.get("status") != "preregistered_before_full_model_load"
        or config.get("seed") != 870087
    ):
        raise ValueError("V87 experiment identity is unsealed or changed")
    strict = config.get("strict_input_contract")
    if strict != {
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
        raise ValueError("V87 direct-memory contract changed")
    dataset = config.get("dataset")
    if not isinstance(dataset, Mapping) or any(
        dataset.get(key) != value
        for key, value in {
            "scene_id": SCENE_ID,
            "row_count": 138,
            "row_inventory_sha256": (
                "9919ff1bee4611dce4132d79fa50f6f6b4ace567a6df780a2e0e21bd88237a8e"
            ),
            "canonical_answer_class_count": 19,
            "answer_class_inventory_sha256": (
                "639cc0f4f843084a6d5dbe7bbf525480acb43aaacd253568837bec9f2e47aa71"
            ),
            "class_weight_inventory_sha256": (
                "2a890ea9ce8314d3404f2e44101453d29574bb174913b05b8746617d300ad874"
            ),
            "all_scene1_rows_used_once_per_epoch": True,
            "answer_metadata_training_only": True,
            "runtime_serializes_questions_or_answers": False,
        }.items()
    ):
        raise ValueError("V87 exact dataset/class-balance contract changed")
    frozen = config.get("frozen_stack")
    if not isinstance(frozen, Mapping) or any(
        frozen.get(key) != value
        for key, value in {
            "base_gemma_frozen": True,
            "v54_bank_count": 6,
            "v85_bank_name": "v85_strict_multiscene_bridge",
            "v85_bank_state_sha256": (
                "f31b8f99f77f1b7b92dafd74220e5e12ccfa35cbc8630a6d7640f2fe1f93c581"
            ),
            "v86_bank_name": PARENT_BANK_NAME,
            "v86_bank_target_module": PARENT_TARGET_MODULE,
            "v86_bank_state_sha256": (
                "8b6bd801716132c8aac50c6288b9ba588417dc5e6a7c2c15dd9515892f714260"
            ),
            "total_frozen_bank_count": 8,
            "merged_weights": False,
        }.items()
    ):
        raise ValueError("V87 frozen V85+V86 stack changed")
    bridge = config.get("bridge")
    if not isinstance(bridge, Mapping) or any(
        bridge.get(key) != value
        for key, value in {
            "bank_name": FRESH_BANK_NAME,
            "target_module": TARGET_MODULE,
            "target_layer_type": "full_attention",
            "target_in_features": 1536,
            "target_out_features": 12288,
            "rank": 8,
            "alpha": 16.0,
            "dropout": 0.0,
            "trainable_parameter_count": 110592,
            "initialization_algorithm": "cpu_kaiming_uniform_a_exact_zero_b",
            "initialization_seed": 870087,
            "expected_initial_state_sha256": (
                "5aff719665064f7d0a3582fc5d67ff330a5044014278bd2ca69bd153ebefbeca"
            ),
            "disjoint_from_all_frozen_banks": True,
        }.items()
    ):
        raise ValueError("V87 sole fresh bridge contract changed")
    training = config.get("training")
    expected_training = {
        "optimizer": "AdamW",
        "epochs": 8,
        "rows_per_epoch": 138,
        "microbatch_size": 1,
        "gradient_accumulation_rows": 6,
        "optimizer_updates": 184,
        "row_order": "opaque_answer_class_round_robin_each_row_once",
        "row_order_seed": 870087,
        "row_order_sha256": ("b1560e3709acf6dff7e7be519a80827ae9a6644c803a8973b5064c0f043180ff"),
        "learning_rate": 0.0005,
        "weight_decay": 0.0,
        "gradient_clip_norm": 1.0,
        "answer_ce_weight": 1.0,
        "class_weight_formula": ("total_rows_divided_by_class_count_times_class_frequency"),
        "class_weights_mean_over_rows": 1.0,
        "equal_aggregate_ce_mass_per_class": True,
        "zero_payload_margin_weight": 1.0,
        "zero_payload_target_margin_nll": 0.5,
        "causal_subset_question_ids": list(CAUSAL_IDS),
        "causal_subset_inventory_sha256": (
            "f1a7b07442a59eba2c6dbfbc4d0ada2066e72bf15bf3b987dd7875d7552a20b3"
        ),
        "zero_payload_preserves_native_boi_eoi": True,
        "zero_payload_zeros_exactly_736_interior_tokens": True,
        "causal_rows_per_epoch": 3,
        "total_causal_margin_rows": 24,
        "checkpoint_selection": "fixed_final_update_184",
        "intermediate_behavior_selection": False,
    }
    if not isinstance(training, Mapping) or any(
        training.get(key) != value for key, value in expected_training.items()
    ):
        raise ValueError("V87 fixed class-balanced training protocol changed")
    gates = config.get("gates")
    if not isinstance(gates, Mapping) or any(
        gates.get(key) != value
        for key, value in {
            "all_scene1_canonical_accuracy_minimum": 0.80,
            "attribute_accuracy_minimum": 0.50,
            "presence_accuracy_minimum": 0.75,
            "spatial_relation_accuracy_minimum": 0.60,
            "exact_training_row_count_required": 138,
            "live_smoke_required_correct": 3,
            "live_smoke_total": 3,
            "causal_correct_memory_mean_nll_below_zero_payload": True,
            "causal_prediction_change_minimum": 1,
            "exact_prefix_hash_invariance_required": True,
            "exact_total_environment_input_invariance_required": True,
            "oracle_physically_unavailable_during_runtime_required": True,
            "forbidden_runtime_read_count_maximum": 0,
            "runtime_promotion_only_after_all_gates": True,
        }.items()
    ):
        raise ValueError("V87 acceptance gates changed")
    if gates.get("live_smoke_questions") != [
        {"question": "Is there a chair?", "expected": "yes"},
        {"question": "What color is the bowl?", "expected": "red"},
        {
            "question": "Is the bowl left or right of the chair?",
            "expected": "left",
        },
    ]:
        raise ValueError("V87 corrected generic smoke changed")
    sources = config.get("sources")
    if not isinstance(sources, Mapping):
        raise TypeError("V87 sources must be a mapping")
    for field in (
        "preflight_source_sha256",
        "training_source_sha256",
        "evaluation_source_sha256",
    ):
        value = sources.get(field)
        if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
            raise ValueError(f"V87 {field} is not sealed")
    scope = config.get("scope")
    if not isinstance(scope, Mapping) or scope != {
        "post_v86_training_set_development": True,
        "single_scene_overfit_demonstration": True,
        "local_inference_only": True,
        "cloud_inference": False,
        "held_out_generalization_claim": False,
        "official_validation_loaded": False,
        "official_test_loaded": False,
        "deferred_final_loaded": False,
        "oracle_loaded_during_training": False,
        "runtime_promotion_authorized": False,
    }:
        raise ValueError("V87 protected scope changed")
    return dict(config)


def answer_class_balance_v87(
    config: Mapping[str, Any], rows: Sequence[RowV73]
) -> tuple[dict[str, int], dict[str, float]]:
    counts = Counter(row.answer_class for row in rows)
    class_count = len(counts)
    weights = {
        class_id: len(rows) / (class_count * frequency) for class_id, frequency in counts.items()
    }
    inventory = sorted([[class_id, frequency] for class_id, frequency in counts.items()])
    weight_inventory = sorted([[class_id, weight] for class_id, weight in weights.items()])
    if (
        class_count != config["dataset"]["canonical_answer_class_count"]
        or canonical_sha256_v85(inventory) != config["dataset"]["answer_class_inventory_sha256"]
        or canonical_sha256_v85(weight_inventory)
        != config["dataset"]["class_weight_inventory_sha256"]
        or abs(sum(weights[row.answer_class] for row in rows) - len(rows)) > 1e-10
        or any(
            abs(weights[class_id] * frequency - len(rows) / class_count) > 1e-10
            for class_id, frequency in counts.items()
        )
    ):
        raise ValueError("V87 exact answer-class balance changed")
    return dict(counts), weights


def balanced_schedule_v87(
    rows: Sequence[RowV73], *, seed: int = 870087, epochs: int = 8
) -> tuple[tuple[int, RowV73], ...]:
    grouped: defaultdict[str, list[RowV73]] = defaultdict(list)
    for row in rows:
        grouped[row.answer_class].append(row)
    classes = sorted(grouped)
    schedule: list[tuple[int, RowV73]] = []
    for epoch in range(epochs):
        per_class: dict[str, list[RowV73]] = {}
        for class_ordinal, class_id in enumerate(classes):
            values = sorted(grouped[class_id], key=lambda row: row.question_id)
            random.Random(seed + epoch * 1000 + class_ordinal).shuffle(values)
            per_class[class_id] = values
        for round_ordinal in range(max(len(values) for values in per_class.values())):
            class_order = list(classes)
            random.Random(seed + epoch * 1000 + 500 + round_ordinal).shuffle(class_order)
            for class_id in class_order:
                if round_ordinal < len(per_class[class_id]):
                    schedule.append((epoch, per_class[class_id][round_ordinal]))
    return tuple(schedule)


class _SyntheticMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(1536, 12288, bias=False, dtype=torch.bfloat16)


class _SyntheticLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = _SyntheticMLP()


class _SyntheticLanguage(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.Identity() for _ in range(34)] + [_SyntheticLayer()])


class _SyntheticGemma(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = _SyntheticLanguage()


def lora_preflight_v87(config: Mapping[str, Any]) -> dict[str, Any]:
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
        raise RuntimeError("V87 synthetic LoRA installation failed")
    initialize_lora_adapter_state(installation, seed=int(bridge["initialization_seed"]))
    state_sha256 = installation.state_sha256()
    if (
        installation.parameter_count != bridge["trainable_parameter_count"]
        or state_sha256 != bridge["expected_initial_state_sha256"]
        or any(torch.count_nonzero(adapter.lora_b).item() for adapter in installation.adapters)
    ):
        raise RuntimeError("V87 deterministic zero-output LoRA preflight failed")
    return {
        "bank_name": FRESH_BANK_NAME,
        "target_modules": list(installation.target_names),
        "parameter_count": installation.parameter_count,
        "initial_state_sha256": state_sha256,
        "base_projection_weight_shape": [12288, 1536],
        "lora_a_shape": [8, 1536],
        "lora_b_shape": [12288, 8],
        "exact_zero_output_at_initialization": True,
        "full_gemma_model_loaded": False,
    }


def authenticate_sources_v87(config: Mapping[str, Any]) -> dict[str, str]:
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
        sources["parent_v86_config"]: sources["parent_v86_config_sha256"],
        sources["parent_v86_preregistration"]: sources["parent_v86_preregistration_sha256"],
        sources["parent_v86_cpu_preflight"]: sources["parent_v86_cpu_preflight_sha256"],
        sources["parent_v86_training_report"]: sources["parent_v86_training_report_sha256"],
        str(Path(sources["parent_v86_checkpoint"]) / "bridge.safetensors"): sources[
            "parent_v86_bridge_sha256"
        ],
        str(Path(sources["parent_v86_checkpoint"]) / "runtime_metadata.json"): sources[
            "parent_v86_metadata_sha256"
        ],
        sources["parent_v86_predictions"]: sources["parent_v86_predictions_sha256"],
        sources["parent_v86_evaluation"]: sources["parent_v86_evaluation_sha256"],
        sources["preflight_source"]: sources["preflight_source_sha256"],
        sources["training_source"]: sources["training_source_sha256"],
        sources["evaluation_source"]: sources["evaluation_source_sha256"],
    }
    observed: dict[str, str] = {}
    for path, expected_sha256 in expected.items():
        actual = sha256_file_v85(path)
        if actual != expected_sha256:
            raise ValueError(f"V87 pinned source changed: {path}")
        observed[str(path)] = actual
    snapshot = (
        Path.home()
        / ".cache/huggingface/hub/models--google--gemma-4-E2B-it/snapshots"
        / str(sources["model_revision"])
    )
    blob = (snapshot / "model.safetensors").resolve(strict=True)
    if blob.name != sources["model_blob_sha256_identity"]:
        raise ValueError("V87 local Gemma blob identity changed")
    model_config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    text = model_config.get("text_config")
    if not isinstance(text, Mapping) or (
        text.get("hidden_size") != 1536
        or text.get("intermediate_size") != 6144
        or text.get("use_double_wide_mlp") is not True
        or text.get("num_hidden_layers") != 35
        or text.get("layer_types", [None] * 35)[34] != "full_attention"
    ):
        raise ValueError("V87 pinned Gemma topology changed")
    observed["gemma_model_blob_sha256_identity"] = blob.name
    return observed


def validate_parent_v86(config: Mapping[str, Any]) -> dict[str, Any]:
    sources = config["sources"]
    training = _strict_json(sources["parent_v86_training_report"])
    evaluation = _strict_json(sources["parent_v86_evaluation"])
    candidate = _strict_json(Path(sources["parent_v86_checkpoint"]) / "runtime_metadata.json")
    training_gates = training.get("gates")
    model_gates = evaluation.get("metrics", {}).get("model_acceptance_gates")
    if (
        not isinstance(training_gates, Mapping)
        or not all(training_gates.values())
        or training.get("optimizer_updates") != 92
        or training.get("runtime_promotion_authorized") is not False
        or candidate.get("state_sha256") != config["frozen_stack"]["v86_bank_state_sha256"]
        or candidate.get("questions_or_answers_serialized") is not False
        or candidate.get("environmental_memory_serialized") is not False
        or evaluation.get("status") != "model_gates_fail_not_runtime_promotable"
        or evaluation.get("metrics", {}).get("canonical_type_specific", {}).get("correct") != 86
        or evaluation.get("metrics", {}).get("canonical_type_specific", {}).get("total") != 138
        or not isinstance(model_gates, Mapping)
        or model_gates.get("all_scene1_canonical_accuracy_at_least_0_80") is not False
        or any(
            value is not True
            for key, value in model_gates.items()
            if key != "all_scene1_canonical_accuracy_at_least_0_80"
        )
        or evaluation.get("runtime_promotion_authorized") is not False
    ):
        raise ValueError("V87 parent V86 failure rationale or frozen state changed")
    return {
        "parent_v86_training_updates": 92,
        "parent_v86_training_gates_all_passed": True,
        "parent_v86_canonical_correct": 86,
        "parent_v86_canonical_total": 138,
        "parent_v86_canonical_accuracy": 86 / 138,
        "parent_v86_generic_smoke_correct": 3,
        "parent_v86_generic_smoke_total": 3,
        "parent_v86_causal_control_passed": True,
        "parent_v86_only_failed_gate": ("all_scene1_canonical_accuracy_at_least_0_80"),
        "parent_v86_runtime_promoted": False,
        "parent_v86_bridge_state_sha256": candidate["state_sha256"],
        "parent_v86_mutated": False,
    }


def protocol_preflight_v87(config: Mapping[str, Any]) -> dict[str, Any]:
    rows = load_scene1_rows_v86(config)
    counts, weights = answer_class_balance_v87(config, rows)
    schedule = balanced_schedule_v87(
        rows,
        seed=int(config["training"]["row_order_seed"]),
        epochs=int(config["training"]["epochs"]),
    )
    schedule_hash = canonical_sha256_v85([[epoch, row.question_id] for epoch, row in schedule])
    per_epoch = Counter(epoch for epoch, _row in schedule)
    per_row = Counter(row.question_id for _epoch, row in schedule)
    if (
        len(schedule) != 1104
        or schedule_hash != config["training"]["row_order_sha256"]
        or set(per_epoch.values()) != {138}
        or set(per_row.values()) != {8}
    ):
        raise ValueError("V87 deterministic balanced schedule changed")
    causal = causal_rows_v86(config, rows)
    memory, memory_hash, metadata = load_scene1_memory_v86(config)
    zero = zero_payload_memory_v86(memory)
    zero_hash = prefix_sha256(zero)
    if zero_hash == memory_hash:
        raise RuntimeError("V87 zero-payload control did not change the payload")
    opaque_classes = sorted(counts)
    return {
        "row_count": len(rows),
        "opaque_answer_class_count": len(opaque_classes),
        "answer_class_inventory_sha256": config["dataset"]["answer_class_inventory_sha256"],
        "class_weight_inventory_sha256": config["dataset"]["class_weight_inventory_sha256"],
        "minimum_class_frequency": min(counts.values()),
        "maximum_class_frequency": max(counts.values()),
        "minimum_class_weight": min(weights.values()),
        "maximum_class_weight": max(weights.values()),
        "mean_class_weight_over_rows": sum(weights[row.answer_class] for row in rows) / len(rows),
        "equal_aggregate_ce_mass_per_class": True,
        "schedule_rows": len(schedule),
        "rows_each_epoch": 138,
        "appearances_each_row": 8,
        "row_order_sha256": schedule_hash,
        "first_schedule_keys": [[epoch, row.question_id] for epoch, row in schedule[:3]],
        "last_schedule_keys": [[epoch, row.question_id] for epoch, row in schedule[-3:]],
        "causal_question_ids": [row.question_id for row in causal],
        "causal_rows_total": len(causal) * 8,
        "fixed_memory_shape": list(memory.shape),
        "fixed_memory_dtype": str(memory.dtype),
        "fixed_memory_prefix_sha256": memory_hash,
        "expected_fixed_memory_prefix_sha256": EXPECTED_PREFIX_SHA256,
        "memory_compiled_before_question": metadata["compiled_before_user_question"],
        "zero_payload_prefix_sha256": zero_hash,
        "zero_payload_preserves_native_boi": bool(torch.equal(zero[:, :1], memory[:, :1])),
        "zero_payload_preserves_native_eoi": bool(torch.equal(zero[:, -1:], memory[:, -1:])),
        "zero_payload_token_count": 736,
        "zero_payload_nonzero_scalar_count": int(torch.count_nonzero(zero[:, 1:-1]).item()),
        "answer_text_serialized": False,
        "questions_tokenized": False,
        "full_gemma_model_loaded": False,
    }


def build_preregistration_v87(config_path: str | Path = CONFIG) -> dict[str, Any]:
    config = load_config_v87(config_path)
    payload = {
        "artifact": PREREG_ARTIFACT,
        "schema_version": 87,
        "status": "sealed_after_v86_failure_before_first_v87_full_model_load",
        "config_path": Path(config_path).as_posix(),
        "config_sha256": sha256_file_v85(config_path),
        "authenticated_sources": authenticate_sources_v87(config),
        "parent_v86_evidence": validate_parent_v86(config),
        "strict_input_contract": config["strict_input_contract"],
        "dataset_contract": config["dataset"],
        "frozen_stack": config["frozen_stack"],
        "bridge": config["bridge"],
        "training_protocol": config["training"],
        "fixed_gates": config["gates"],
        "protocol_preflight": protocol_preflight_v87(config),
        "lora_cpu_preflight": lora_preflight_v87(config),
        "parent_v86_behavior_known_for_post_failure_design": True,
        "new_v87_behavior_scored": False,
        "answers_available_to_training_only": True,
        "answers_or_questions_serialized_in_runtime_candidate": False,
        "full_gemma_model_loaded": False,
        "optimizer_constructed": False,
        "optimizer_updates": 0,
        "official_validation_loaded": False,
        "official_test_loaded": False,
        "deferred_final_loaded": False,
        "oracle_loaded": False,
        "runtime_promotion_authorized": False,
    }
    output, _sha = atomic_create_json_v85(config["outputs"]["preregistration"], payload)
    payload["output"] = output.relative_to(PROJECT_ROOT).as_posix()
    return payload


def authenticate_preregistration_v87(
    config: Mapping[str, Any], *, config_path: str | Path = CONFIG
) -> dict[str, str]:
    path = resolve_v85(config["outputs"]["preregistration"])
    payload = _strict_json(path)
    config_sha256 = sha256_file_v85(config_path)
    if (
        payload.get("artifact") != PREREG_ARTIFACT
        or payload.get("status") != "sealed_after_v86_failure_before_first_v87_full_model_load"
        or payload.get("config_sha256") != config_sha256
        or payload.get("parent_v86_behavior_known_for_post_failure_design") is not True
        or payload.get("new_v87_behavior_scored") is not False
        or payload.get("full_gemma_model_loaded") is not False
        or payload.get("optimizer_constructed") is not False
        or payload.get("optimizer_updates") != 0
        or payload.get("oracle_loaded") is not False
        or payload.get("runtime_promotion_authorized") is not False
    ):
        raise ValueError("V87 preregistration changed")
    return {
        "config_sha256": config_sha256,
        "preregistration_sha256": sha256_file_v85(path),
    }


def run_cpu_preflight_v87(config_path: str | Path = CONFIG) -> dict[str, Any]:
    config = load_config_v87(config_path)
    prereg = authenticate_preregistration_v87(config, config_path=config_path)
    report = {
        "artifact": PREFLIGHT_ARTIFACT,
        "schema_version": 87,
        "status": "passed",
        "passed": True,
        **prereg,
        "authenticated_sources": authenticate_sources_v87(config),
        "parent_v86_evidence": validate_parent_v86(config),
        "protocol_preflight": protocol_preflight_v87(config),
        "lora_preflight": lora_preflight_v87(config),
        "fixed_final_optimizer_updates": 184,
        "fixed_final_checkpoint_selection": "fixed_final_update_184",
        "all_138_rows_used_once_each_epoch": True,
        "all_19_answer_classes_equal_aggregate_ce_mass": True,
        "same_fixed_memory_compiled_before_questions": True,
        "all_738_memory_slots_retained": True,
        "question_derived_environmental_tokens": 0,
        "question_conditioned_environmental_readout": False,
        "full_gemma_model_loaded": False,
        "optimizer_constructed": False,
        "optimizer_updates": 0,
        "new_v87_behavior_scored": False,
        "protected_or_sealed_new_behavior_artifacts_opened": [],
        "official_validation_loaded": False,
        "official_test_loaded": False,
        "deferred_final_loaded": False,
        "oracle_loaded": False,
        "runtime_promotion_authorized": False,
    }
    output, _sha = atomic_create_json_v85(config["outputs"]["cpu_preflight"], report)
    report["output"] = output.relative_to(PROJECT_ROOT).as_posix()
    return report


def authenticate_cpu_preflight_v87(
    config: Mapping[str, Any], *, config_path: str | Path = CONFIG
) -> dict[str, str]:
    prereg = authenticate_preregistration_v87(config, config_path=config_path)
    path = resolve_v85(config["outputs"]["cpu_preflight"])
    payload = _strict_json(path)
    if (
        payload.get("artifact") != PREFLIGHT_ARTIFACT
        or payload.get("status") != "passed"
        or payload.get("passed") is not True
        or payload.get("config_sha256") != prereg["config_sha256"]
        or payload.get("preregistration_sha256") != prereg["preregistration_sha256"]
        or payload.get("full_gemma_model_loaded") is not False
        or payload.get("optimizer_updates") != 0
        or payload.get("new_v87_behavior_scored") is not False
        or payload.get("protected_or_sealed_new_behavior_artifacts_opened") != []
        or payload.get("oracle_loaded") is not False
        or payload.get("runtime_promotion_authorized") is not False
    ):
        raise ValueError("V87 CPU preflight changed")
    return {**prereg, "cpu_preflight_sha256": sha256_file_v85(path)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preregister", "preflight"))
    parser.add_argument("--config", default=str(CONFIG))
    args = parser.parse_args(argv)
    result = (
        build_preregistration_v87(args.config)
        if args.command == "preregister"
        else run_cpu_preflight_v87(args.config)
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CAUSAL_IDS",
    "CONFIG",
    "FRESH_BANK_NAME",
    "PARENT_BANK_NAME",
    "PARENT_TARGET_MODULE",
    "PREFLIGHT_ARTIFACT",
    "PREREG_ARTIFACT",
    "SCENE_ID",
    "TARGET_MODULE",
    "answer_class_balance_v87",
    "authenticate_cpu_preflight_v87",
    "authenticate_preregistration_v87",
    "authenticate_sources_v87",
    "balanced_schedule_v87",
    "build_preregistration_v87",
    "load_config_v87",
    "lora_preflight_v87",
    "main",
    "protocol_preflight_v87",
    "run_cpu_preflight_v87",
    "validate_parent_v86",
]
