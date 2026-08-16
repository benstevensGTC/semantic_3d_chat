"""Model-free contract checks for V94's strict forty-scene continuation.

V94 starts from the exact released V85 seven-bank runtime stack, consumes the
union of V82's former train and development memories as one 40-scene training
pool, and keeps scenes 57--62 label-isolated until a fixed-final checkpoint
exists.  Draft validation is intentionally read-only: this module never opens
the reserved validation labels, constructs an optimizer, or loads Gemma.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import torch
import yaml
from safetensors import safe_open
from torch import nn

from semantic_3d_chat.evaluation.v85_strict_multiscene_preflight import (
    canonical_sha256_v85,
    resolve_v85,
    sha256_file_v85,
)
from semantic_3d_chat.language.lora import (
    LoRASettings,
    initialize_lora_adapter_state,
    install_lora_adapters,
)
from semantic_3d_chat.language.prefix_injection import prefix_sha256
from semantic_3d_chat.training.train_question_control_v73 import (
    RowV73,
    changed_units_v73,
    load_training_rows_v73,
)
from semantic_3d_chat.training.v82_reader_artifacts import load_v82_cache

CONFIG: Final[Path] = Path(
    "configs/experiments/gemma4_v94_strict_multiscene_full40.yaml"
)
FRESH_BANK_NAME: Final[str] = "v94_strict_multiscene_full40_bridge"
TARGET_MODULE: Final[str] = "model.language_model.layers.34.mlp.gate_proj"
PINNED_MODEL_TENSOR: Final[str] = TARGET_MODULE + ".weight"
TARGET_IN_FEATURES: Final[int] = 1536
TARGET_OUT_FEATURES: Final[int] = 12288
FRESH_PARAMETER_COUNT: Final[int] = 110592
EXPECTED_INITIAL_STATE_SHA256: Final[str] = (
    "7d413bc8bf02accb8d870a56e38de383baba6f7028eda54b1283f7994df71628"
)
EXPECTED_EVALUATION_SCENES: Final[tuple[str, ...]] = tuple(
    f"scene_{index:06d}" for index in range(57, 63)
)
_HEX64: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{64}")
_DRAFT: Final[str] = "draft_before_sealed_preflight"
_SEALED: Final[str] = "sealed_before_full_model_load"
PREREG_ARTIFACT: Final[str] = "gemma4_v94_strict_multiscene_full40_preregistration_v1"
PREFLIGHT_ARTIFACT: Final[str] = "gemma4_v94_strict_multiscene_full40_cpu_preflight_v1"


def _require(value: Any, expected: Any, label: str) -> None:
    if value != expected:
        raise ValueError(f"V94 {label} changed")


def _require_hash(value: Any, label: str, *, draft: bool) -> None:
    if isinstance(value, str) and _HEX64.fullmatch(value):
        return
    if draft and value == "TO_FILL":
        return
    raise ValueError(f"V94 {label} is not sealed")


def _strict_json_v94(path: str | Path) -> dict[str, Any]:
    source = resolve_v85(path)
    if source.is_symlink() or not source.is_file():
        raise FileNotFoundError(source)
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"V94 JSON must contain one object: {source}")
    return value


def load_config_v94(
    path: str | Path = CONFIG, *, allow_draft: bool = True
) -> dict[str, Any]:
    source = resolve_v85(path)
    if source.is_symlink() or not source.is_file():
        raise FileNotFoundError(source)
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or set(payload) != {"v94"}:
        raise ValueError("V94 config must contain exactly one v94 mapping")
    config = payload["v94"]
    if not isinstance(config, Mapping):
        raise TypeError("V94 config payload must be a mapping")
    _require(config.get("schema_version"), 94, "schema version")
    _require(
        config.get("artifact"),
        "gemma4_v94_strict_multiscene_full40_direct_memory_lora_v1",
        "artifact",
    )
    status = config.get("status")
    if status not in ({_DRAFT, _SEALED} if allow_draft else {_SEALED}):
        raise ValueError("V94 config status is not authorized")
    _require(config.get("seed"), 940094, "seed")

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
            "memory_projector_enabled": False,
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
    dataset = config.get("dataset")
    if not isinstance(dataset, Mapping):
        raise TypeError("V94 dataset contract must be a mapping")
    for key, expected in {
        "pair_count": 20,
        "scene_count": 40,
        "row_count": 960,
        "changed_unit_count": 66,
        "changed_side_count": 132,
        "answer_class_count": 29,
        "all_rows_used_once_per_epoch": True,
        "old_v82_train_and_development_roles_retired_for_v94_optimization": True,
        "runtime_serializes_questions_or_answers": False,
    }.items():
        _require(dataset.get(key), expected, f"dataset {key}")
    for field in (
        "row_inventory_sha256",
        "scene_inventory_sha256",
        "pair_inventory_sha256",
        "answer_class_inventory_sha256",
        "inverse_sqrt_class_weight_inventory_sha256",
    ):
        _require_hash(dataset.get(field), f"dataset {field}", draft=status == _DRAFT)

    frozen = config.get("frozen_stack")
    if not isinstance(frozen, Mapping):
        raise TypeError("V94 frozen stack must be a mapping")
    for key, expected in {
        "exact_parent": "v85_strict_runtime_candidate",
        "base_gemma_frozen": True,
        "frozen_bank_count": 7,
        "frozen_adapter_parameter_count": 565248,
        "v85_bank_name": "v85_strict_multiscene_bridge",
        "v85_bank_state_sha256": (
            "f31b8f99f77f1b7b92dafd74220e5e12ccfa35cbc8630a6d7640f2fe1f93c581"
        ),
        "merged_weights": False,
    }.items():
        _require(frozen.get(key), expected, f"frozen stack {key}")
    bank_hashes = frozen.get("frozen_bank_state_sha256")
    if not isinstance(bank_hashes, Mapping) or len(bank_hashes) != 7:
        raise ValueError("V94 exact seven-bank parent inventory changed")

    bridge = config.get("bridge")
    if not isinstance(bridge, Mapping):
        raise TypeError("V94 bridge must be a mapping")
    for key, expected in {
        "bank_name": FRESH_BANK_NAME,
        "target_module": TARGET_MODULE,
        "target_layer_type": "full_attention_mlp_gate",
        "pinned_weight_shape": [TARGET_OUT_FEATURES, TARGET_IN_FEATURES],
        "pinned_weight_dtype": "BF16",
        "target_in_features": TARGET_IN_FEATURES,
        "target_out_features": TARGET_OUT_FEATURES,
        "rank": 8,
        "alpha": 16.0,
        "dropout": 0.0,
        "trainable_parameter_count": FRESH_PARAMETER_COUNT,
        "initialization_algorithm": "cpu_kaiming_uniform_a_exact_zero_b",
        "initialization_seed": 940094,
        "expected_initial_state_sha256": EXPECTED_INITIAL_STATE_SHA256,
        "disjoint_from_all_frozen_banks": True,
        "total_bank_count_after_install": 8,
        "total_adapter_parameter_count_after_install": 675840,
    }.items():
        _require(bridge.get(key), expected, f"bridge {key}")

    training = config.get("training")
    if not isinstance(training, Mapping):
        raise TypeError("V94 training contract must be a mapping")
    for key, expected in {
        "optimizer": "AdamW",
        "epochs": 3,
        "rows_per_epoch": 960,
        "total_micro_rows": 2880,
        "microbatch_size": 1,
        "gradient_accumulation_rows": 8,
        "optimizer_updates": 360,
        "row_order": "sorted_keys_then_epoch_seeded_shuffle",
        "row_order_seed": 940094,
        "learning_rate": 0.00025,
        "weight_decay": 0.0,
        "gradient_clip_norm": 1.0,
        "answer_ce_weight": 1.0,
        "class_weight_formula": "normalized_inverse_sqrt_answer_class_frequency",
        "class_weights_mean_over_rows": 1.0,
        "paired_wrong_scene_margin_weight": 1.0,
        "paired_wrong_scene_target_margin_nll": 0.5,
        "paired_wrong_scene_changed_sides_per_epoch": 132,
        "total_paired_wrong_scene_margin_rows": 396,
        "zero_payload_margin_weight": 1.0,
        "zero_payload_target_margin_nll": 0.5,
        "zero_payload_selection": (
            "both_sides_of_lexicographically_first_changed_unit_per_change_family"
        ),
        "zero_payload_causal_sides_per_epoch": 18,
        "total_zero_payload_margin_rows": 54,
        "checkpoint_every_optimizer_updates": 30,
        "deterministic_resume": True,
        "checkpoint_selection": "fixed_final_update_360",
        "intermediate_behavior_selection": False,
    }.items():
        _require(training.get(key), expected, f"training {key}")
    for field in ("row_order_sha256", "zero_payload_side_inventory_sha256"):
        _require_hash(training.get(field), f"training {field}", draft=status == _DRAFT)

    evaluation = config.get("evaluation")
    if not isinstance(evaluation, Mapping):
        raise TypeError("V94 evaluation contract must be a mapping")
    for key, expected in {
        "scene_ids": list(EXPECTED_EVALUATION_SCENES),
        "scene_count": 6,
        "pair_count": 3,
        "row_count": 216,
        "changed_unit_count": 12,
        "changed_side_count": 24,
        "opened_by_preflight": False,
        "opened_by_training": False,
        "labels_opened_by_memory_compiler": False,
        "labels_opened_by_question_only_predictor": False,
        "labels_opened_only_by_separate_scorer": True,
        "fixed_final_selected_before_evaluation": True,
        "question_label_isolation_required": True,
    }.items():
        _require(evaluation.get(key), expected, f"evaluation {key}")

    gates = config.get("gates")
    if not isinstance(gates, Mapping):
        raise TypeError("V94 gates must be a mapping")
    for key, expected in {
        "canonical_accuracy_minimum": 0.65,
        "canonical_accuracy_margin_over_exact_v85_same_216_comparator": 0.05,
        "attribute_correct_minimum": 24,
        "attribute_total": 48,
        "count_correct_minimum": 38,
        "count_total": 42,
        "metric_correct_minimum": 5,
        "metric_total": 6,
        "orientation_correct_minimum": 5,
        "orientation_total": 6,
        "presence_correct_minimum": 30,
        "presence_total": 42,
        "spatial_relation_correct_minimum": 29,
        "spatial_relation_total": 48,
        "support_correct_minimum": 16,
        "support_total": 24,
        "changed_side_correct_minimum": 14,
        "changed_side_total": 24,
        "complete_changed_units_minimum": 6,
        "changed_unit_total": 12,
        "canonical_prediction_changing_units_minimum": 8,
        "mean_changed_side_wrong_minus_correct_nll_minimum": 0.15,
        "zero_payload_mean_nll_gap_minimum": 0.5,
        "zero_payload_prediction_change_minimum": 6,
        "correct_scene_nll_below_zero_payload_required": True,
        "correct_scene_nll_below_shuffled_scene_required": True,
        "exact_prefix_hash_invariance_required": True,
        "every_evaluation_memory_hash_retained_required": True,
        "question_label_isolation_required": True,
        "protected_read_count_maximum": 0,
        "automatic_runtime_promotion": False,
    }.items():
        _require(gates.get(key), expected, f"gate {key}")

    sources = config.get("sources")
    if not isinstance(sources, Mapping):
        raise TypeError("V94 sources must be a mapping")
    for field in (
        "preflight_source_sha256",
        "training_source_sha256",
        "evaluation_source_sha256",
    ):
        _require_hash(sources.get(field), f"source {field}", draft=status == _DRAFT)
    scope = config.get("scope")
    if not isinstance(scope, Mapping) or any(
        scope.get(field) is not False
        for field in (
            "cloud_inference",
            "evaluation_labels_loaded_during_preflight",
            "evaluation_labels_loaded_during_training",
            "official_test_loaded",
            "deferred_final_loaded",
            "oracle_loaded",
            "runtime_promotion_authorized",
        )
    ):
        raise ValueError("V94 protected scope changed")
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


def class_weights_v94(
    config: Mapping[str, Any], rows: Sequence[RowV73]
) -> dict[str, float]:
    """Return inverse-sqrt-frequency weights normalized to row mean one."""

    counts = Counter(row.answer_class for row in rows)
    raw = {class_id: 1.0 / math.sqrt(count) for class_id, count in counts.items()}
    normalizer = len(rows) / sum(raw[row.answer_class] for row in rows)
    weights = {class_id: raw[class_id] * normalizer for class_id in sorted(raw)}
    inventory = sorted([[key, counts[key]] for key in counts])
    weighted = sorted([[key, weights[key]] for key in weights])
    expected = config["dataset"]
    if (
        len(counts) != expected["answer_class_count"]
        or (expected["answer_class_inventory_sha256"] != "TO_FILL" and canonical_sha256_v85(inventory) != expected["answer_class_inventory_sha256"])
        or (expected["inverse_sqrt_class_weight_inventory_sha256"] != "TO_FILL" and canonical_sha256_v85(weighted) != expected["inverse_sqrt_class_weight_inventory_sha256"])
        or abs(sum(weights[row.answer_class] for row in rows) / len(rows) - 1.0) > 1e-12
    ):
        raise ValueError("V94 inverse-sqrt class weighting changed")
    return weights


def load_training_rows_v94(config: Mapping[str, Any]) -> tuple[RowV73, ...]:
    rows = load_training_rows_v73(config["sources"]["training_qa"])
    scenes = sorted({row.scene_id for row in rows})
    pairs = sorted({row.pair_id for row in rows})
    units = changed_units_v73(rows)
    inventory_hashes = {
        "row_inventory_sha256": canonical_sha256_v85(_row_inventory(rows)),
        "scene_inventory_sha256": canonical_sha256_v85(scenes),
        "pair_inventory_sha256": canonical_sha256_v85(pairs),
    }
    contract = config["dataset"]
    if (
        len(rows) != 960
        or len(scenes) != 40
        or len(pairs) != 20
        or len(units) != 66
        or sum(row.expected_change for row in rows) != 132
        or any(
            contract[key] != "TO_FILL" and contract[key] != value
            for key, value in inventory_hashes.items()
        )
    ):
        raise ValueError("V94 exact complete training-pool inventory changed")
    class_weights_v94(config, rows)
    return tuple(rows)


def training_schedule_v94(
    rows: Sequence[RowV73], *, seed: int = 940094, epochs: int = 3
) -> tuple[tuple[int, RowV73], ...]:
    schedule: list[tuple[int, RowV73]] = []
    for epoch in range(epochs):
        shuffled = sorted(rows, key=lambda row: row.key)
        random.Random(seed + epoch).shuffle(shuffled)
        schedule.extend((epoch, row) for row in shuffled)
    return tuple(schedule)


def causal_sides_v94(
    config: Mapping[str, Any], rows: Sequence[RowV73]
) -> tuple[RowV73, ...]:
    units = changed_units_v73(rows)
    by_family: dict[str, list[Any]] = {}
    for unit in units:
        by_family.setdefault(unit.change_type, []).append(unit)
    if len(by_family) != 9:
        raise ValueError("V94 expected exactly nine change families")
    selected: list[RowV73] = []
    for family in sorted(by_family):
        unit = min(by_family[family], key=lambda value: (value.pair_id, value.question_key))
        selected.extend((unit.left, unit.right))
    selected.sort(key=lambda row: row.key)
    inventory = [[row.scene_id, row.question_id] for row in selected]
    observed = canonical_sha256_v85(inventory)
    expected = config["training"]["zero_payload_side_inventory_sha256"]
    if len(selected) != 18 or (expected != "TO_FILL" and expected != observed):
        raise ValueError("V94 fixed eighteen-side zero-payload subset changed")
    return tuple(selected)


def load_scene_memories_v94(
    config: Mapping[str, Any], rows: Sequence[RowV73]
) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    memories: dict[str, torch.Tensor] = {}
    hashes: dict[str, str] = {}
    for source_field in ("train_memory_cache", "development_memory_cache"):
        cache = load_v82_cache(resolve_v85(config["sources"][source_field]))
        for scene_id, memory in zip(
            cache.metadata["scene_ids"], cache.tensors["scene_memories"], strict=True
        ):
            if scene_id in memories:
                raise ValueError("V94 memory caches overlap")
            fixed = memory.unsqueeze(0).detach().cpu().contiguous()
            if tuple(fixed.shape) != (1, 738, 1536) or fixed.dtype != torch.bfloat16:
                raise ValueError("V94 immutable scene memory shape or dtype changed")
            memories[scene_id] = fixed
            hashes[scene_id] = prefix_sha256(fixed)
    requested = sorted({row.scene_id for row in rows})
    if sorted(memories) != requested or len(memories) != 40:
        raise ValueError("V94 does not bind the exact forty training memories")
    return memories, hashes


def zero_payload_memory_v94(memory: torch.Tensor) -> torch.Tensor:
    if tuple(memory.shape) != (1, 738, 1536) or memory.dtype != torch.bfloat16:
        raise ValueError("V94 zero control requires one BF16 strict memory")
    zero = memory.clone()
    zero[:, 1:-1].zero_()
    if (
        not torch.equal(zero[:, :1], memory[:, :1])
        or not torch.equal(zero[:, -1:], memory[:, -1:])
        or torch.count_nonzero(zero[:, 1:-1]).item() != 0
    ):
        raise RuntimeError("V94 zero control changed native boundaries")
    return zero


class _SyntheticMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(
            TARGET_IN_FEATURES,
            TARGET_OUT_FEATURES,
            bias=False,
            dtype=torch.bfloat16,
        )


class _SyntheticLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = _SyntheticMLP()


class _SyntheticLanguage(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Identity() for _ in range(34)] + [_SyntheticLayer()]
        )


class _SyntheticGemma(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = _SyntheticLanguage()


def lora_preflight_v94(config: Mapping[str, Any]) -> dict[str, Any]:
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
        raise RuntimeError("V94 synthetic LoRA installation failed")
    initialize_lora_adapter_state(installation, seed=int(bridge["initialization_seed"]))
    observed = installation.state_sha256()
    if (
        installation.parameter_count != FRESH_PARAMETER_COUNT
        or observed != EXPECTED_INITIAL_STATE_SHA256
        or any(torch.count_nonzero(adapter.lora_b).item() for adapter in installation.adapters)
    ):
        raise RuntimeError("V94 deterministic exact-zero LoRA initialization changed")
    return {
        "bank_name": FRESH_BANK_NAME,
        "target_modules": list(installation.target_names),
        "parameter_count": installation.parameter_count,
        "initial_state_sha256": observed,
        "base_projection_weight_shape": [TARGET_OUT_FEATURES, TARGET_IN_FEATURES],
        "lora_a_shape": [8, TARGET_IN_FEATURES],
        "lora_b_shape": [TARGET_OUT_FEATURES, 8],
        "exact_zero_output_at_initialization": True,
        "full_gemma_model_loaded": False,
    }


def authenticate_pinned_model_tensor_v94(config: Mapping[str, Any]) -> dict[str, Any]:
    sources = config["sources"]
    snapshot = (
        Path.home()
        / ".cache/huggingface/hub/models--google--gemma-4-E2B-it/snapshots"
        / str(sources["model_revision"])
    )
    blob = (snapshot / "model.safetensors").resolve(strict=True)
    if blob.name != sources["model_blob_sha256_identity"]:
        raise ValueError("V94 pinned Gemma blob identity changed")
    with safe_open(str(blob), framework="pt", device="cpu") as handle:
        if PINNED_MODEL_TENSOR not in handle.keys():  # noqa: SIM118
            raise ValueError("V94 pinned gate projection is absent")
        sliced = handle.get_slice(PINNED_MODEL_TENSOR)
        shape, dtype = list(sliced.get_shape()), str(sliced.get_dtype())
    if shape != [TARGET_OUT_FEATURES, TARGET_IN_FEATURES] or dtype != "BF16":
        raise ValueError("V94 pinned gate projection topology changed")
    return {
        "model_blob_sha256_identity": blob.name,
        "tensor_name": PINNED_MODEL_TENSOR,
        "shape": shape,
        "dtype": dtype,
        "tensor_materialized": False,
        "full_gemma_model_loaded": False,
    }


def _source_bindings(config: Mapping[str, Any]) -> tuple[tuple[str, str, bool], ...]:
    sources = config["sources"]
    return (
        (sources["runtime_config"], sources["runtime_config_sha256"], False),
        (sources["training_qa"], sources["training_qa_sha256"], False),
        (sources["split_manifest"], sources["split_manifest_sha256"], False),
        (str(Path(sources["train_memory_cache"]) / "training_tensors.safetensors"), sources["train_memory_tensor_sha256"], False),
        (str(Path(sources["train_memory_cache"]) / "metadata.json"), sources["train_memory_metadata_sha256"], False),
        (str(Path(sources["development_memory_cache"]) / "training_tensors.safetensors"), sources["development_memory_tensor_sha256"], False),
        (str(Path(sources["development_memory_cache"]) / "metadata.json"), sources["development_memory_metadata_sha256"], False),
        (str(Path(sources["frozen_v85_checkpoint"]) / "adapter.safetensors"), sources["frozen_v85_adapter_sha256"], False),
        (str(Path(sources["frozen_v85_checkpoint"]) / "runtime_metadata.json"), sources["frozen_v85_metadata_sha256"], False),
        (sources["parent_v85_config"], sources["parent_v85_config_sha256"], False),
        (sources["parent_v85_preregistration"], sources["parent_v85_preregistration_sha256"], False),
        (sources["parent_v85_cpu_preflight"], sources["parent_v85_cpu_preflight_sha256"], False),
        (sources["parent_v85_training"], sources["parent_v85_training_sha256"], False),
        (sources["parent_v85_fixed_bridge"], sources["parent_v85_fixed_bridge_sha256"], False),
        (sources["parent_v85_fixed_metadata"], sources["parent_v85_fixed_metadata_sha256"], False),
        (sources["parent_v85_development_predictions"], sources["parent_v85_development_predictions_sha256"], False),
        (sources["parent_v85_development_score"], sources["parent_v85_development_score_sha256"], False),
        (sources["sanitized_evaluation_questions"], sources["sanitized_evaluation_questions_sha256"], False),
        (str(Path(sources["evaluation_memory_controller"]) / "control.safetensors"), sources["evaluation_memory_controller_weights_sha256"], False),
        (str(Path(sources["evaluation_memory_controller"]) / "runtime_metadata.json"), sources["evaluation_memory_controller_metadata_sha256"], False),
        (str(Path(sources["evaluation_probe_bank"]) / "probes.safetensors"), sources["evaluation_probe_tensor_sha256"], False),
        (str(Path(sources["evaluation_probe_bank"]) / "runtime_metadata.json"), sources["evaluation_probe_metadata_sha256"], False),
        (sources["preflight_source"], sources["preflight_source_sha256"], True),
        (sources["training_source"], sources["training_source_sha256"], True),
        (sources["evaluation_source"], sources["evaluation_source_sha256"], True),
    )


def authenticate_sources_v94(
    config: Mapping[str, Any], *, require_implementation_sources: bool = True
) -> dict[str, str]:
    """Authenticate permitted sources without ever opening validation labels."""

    observed: dict[str, str] = {}
    for path, expected, implementation in _source_bindings(config):
        if implementation and not require_implementation_sources and expected == "TO_FILL":
            continue
        _require_hash(expected, str(path), draft=False)
        actual = sha256_file_v85(path)
        if actual != expected:
            raise ValueError(f"V94 pinned source changed: {path}")
        observed[path] = actual
    # Deliberately authenticate only the *declared digest* for validation labels.
    # Reading that file is reserved for the post-prediction scorer.
    _require_hash(config["sources"]["evaluation_qa_sha256"], "validation label digest", draft=False)
    observed["evaluation_labels_declared_sha256_not_opened"] = config["sources"]["evaluation_qa_sha256"]
    authenticate_parent_v85_v94(config)
    authenticate_pinned_model_tensor_v94(config)
    return observed


def authenticate_parent_v85_v94(config: Mapping[str, Any]) -> dict[str, Any]:
    """Bind the exact successful V85 release and all seven frozen LoRA banks."""

    sources = config["sources"]
    score = _strict_json_v94(sources["parent_v85_development_score"])
    metrics = score.get("metrics")
    if (
        not isinstance(metrics, Mapping)
        or metrics.get("runtime_candidate_gate_passed") is not True
        or metrics.get("canonical_type_specific", {}).get("correct") != 214
        or metrics.get("canonical_type_specific", {}).get("total") != 384
        or metrics.get("changed_complete_units") != 4
        or metrics.get("canonical_prediction_changing_units") != 8
    ):
        raise ValueError("V94 exact V85 measured parent evidence changed")
    metadata = _strict_json_v94(
        Path(sources["frozen_v85_checkpoint"]) / "runtime_metadata.json"
    )
    hashes = metadata.get("lora_bank_state_sha256")
    lora = metadata.get("lora")
    if (
        hashes != config["frozen_stack"]["frozen_bank_state_sha256"]
        or not isinstance(lora, Mapping)
        or lora.get("adapter_parameter_count") != 565248
        or len(lora.get("banks", ())) != 7
        or lora.get("trainable_adapter_parameter_count") != 0
    ):
        raise ValueError("V94 exact V85 seven-bank runtime parent changed")
    return {
        "parent": "v85_strict_runtime_candidate",
        "frozen_bank_count": 7,
        "frozen_adapter_parameter_count": 565248,
        "frozen_bank_state_sha256": dict(hashes),
        "v85_canonical_accuracy": 214 / 384,
        "v85_runtime_candidate_gate_passed": True,
    }


def derive_contract_v94(config_path: str | Path = CONFIG) -> dict[str, Any]:
    """Derive sealable hashes and prove the complete draft contract read-only."""

    config = load_config_v94(config_path, allow_draft=True)
    rows = load_training_rows_v94(config)
    weights = class_weights_v94(config, rows)
    schedule = training_schedule_v94(
        rows,
        seed=int(config["training"]["row_order_seed"]),
        epochs=int(config["training"]["epochs"]),
    )
    schedule_hash = canonical_sha256_v85(
        [[epoch, row.scene_id, row.question_id] for epoch, row in schedule]
    )
    causal = causal_sides_v94(config, rows)
    memories, memory_hashes = load_scene_memories_v94(config, rows)
    parent = authenticate_parent_v85_v94(config)
    sanitized = _strict_json_v94(config["sources"]["sanitized_evaluation_questions"])
    if (
        sanitized.get("question_count") != 216
        or not isinstance(sanitized.get("questions"), list)
        or len(sanitized["questions"]) != 216
        or any("answer" in item for item in sanitized["questions"] if isinstance(item, Mapping))
    ):
        raise ValueError("V94 sanitized evaluation question manifest changed")
    return {
        "schema_version": 94,
        "status": "derived_draft_not_sealed",
        "config_status": config["status"],
        "dataset_hashes": {
            "row_inventory_sha256": canonical_sha256_v85(_row_inventory(rows)),
            "scene_inventory_sha256": canonical_sha256_v85(sorted(memories)),
            "pair_inventory_sha256": canonical_sha256_v85(sorted({row.pair_id for row in rows})),
            "answer_class_inventory_sha256": canonical_sha256_v85(
                sorted([[key, value] for key, value in Counter(row.answer_class for row in rows).items()])
            ),
            "inverse_sqrt_class_weight_inventory_sha256": canonical_sha256_v85(
                sorted([[key, value] for key, value in weights.items()])
            ),
        },
        "training_schedule_sha256": schedule_hash,
        "zero_payload_side_inventory_sha256": canonical_sha256_v85(
            [[row.scene_id, row.question_id] for row in causal]
        ),
        "training_rows": len(rows),
        "training_scenes": len(memories),
        "training_pairs": len({row.pair_id for row in rows}),
        "changed_units": len(changed_units_v73(rows)),
        "changed_sides": sum(row.expected_change for row in rows),
        "schedule_rows": len(schedule),
        "optimizer_updates": len(schedule) // 8,
        "zero_payload_causal_sides_per_epoch": len(causal),
        "total_zero_payload_margin_rows": len(causal) * 3,
        "all_memory_hashes": memory_hashes,
        "parent_v85": parent,
        "lora_preflight": lora_preflight_v94(config),
        "pinned_model_tensor": authenticate_pinned_model_tensor_v94(config),
        "sanitized_evaluation_question_count": 216,
        "evaluation_scene_ids": list(EXPECTED_EVALUATION_SCENES),
        "evaluation_label_file_opened": False,
        "full_gemma_model_loaded": False,
        "optimizer_constructed": False,
        "optimizer_updates_performed": 0,
        "reports_created": False,
    }


def _atomic_create_json_v94(path: str | Path, payload: Mapping[str, Any]) -> Path:
    destination = resolve_v85(path)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"V94 create-once output exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
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


def build_preregistration_v94(config_path: str | Path = CONFIG) -> dict[str, Any]:
    """Create the seal only after implementation hashes and status are final."""

    config = load_config_v94(config_path, allow_draft=False)
    sources = authenticate_sources_v94(config)
    derived = derive_contract_v94(config_path)
    payload = {
        "artifact": PREREG_ARTIFACT,
        "schema_version": 94,
        "status": "sealed_before_first_v94_full_model_load_or_validation_label_read",
        "config_path": Path(config_path).as_posix(),
        "config_sha256": sha256_file_v85(config_path),
        "authenticated_sources": sources,
        "derived_contract": derived,
        "strict_input_contract": config["strict_input_contract"],
        "dataset_contract": config["dataset"],
        "frozen_stack": config["frozen_stack"],
        "bridge": config["bridge"],
        "training_protocol": config["training"],
        "evaluation_protocol": config["evaluation"],
        "fixed_gates": config["gates"],
        "evaluation_label_file_opened": False,
        "full_gemma_model_loaded": False,
        "optimizer_constructed": False,
        "optimizer_updates": 0,
        "behavior_scored": False,
        "oracle_loaded": False,
        "runtime_promotion_authorized": False,
    }
    output = _atomic_create_json_v94(config["outputs"]["preregistration"], payload)
    return {**payload, "output": output.as_posix()}


def authenticate_preregistration_v94(
    config: Mapping[str, Any], *, config_path: str | Path = CONFIG
) -> dict[str, str]:
    path = resolve_v85(config["outputs"]["preregistration"])
    payload = _strict_json_v94(path)
    config_hash = sha256_file_v85(config_path)
    if (
        payload.get("artifact") != PREREG_ARTIFACT
        or payload.get("status")
        != "sealed_before_first_v94_full_model_load_or_validation_label_read"
        or payload.get("config_sha256") != config_hash
        or payload.get("evaluation_label_file_opened") is not False
        or payload.get("full_gemma_model_loaded") is not False
        or payload.get("optimizer_constructed") is not False
        or payload.get("optimizer_updates") != 0
        or payload.get("behavior_scored") is not False
        or payload.get("oracle_loaded") is not False
        or payload.get("runtime_promotion_authorized") is not False
    ):
        raise ValueError("V94 preregistration changed")
    return {
        "config_sha256": config_hash,
        "preregistration_sha256": sha256_file_v85(path),
    }


def run_cpu_preflight_v94(config_path: str | Path = CONFIG) -> dict[str, Any]:
    config = load_config_v94(config_path, allow_draft=False)
    prereg = authenticate_preregistration_v94(config, config_path=config_path)
    report = {
        "artifact": PREFLIGHT_ARTIFACT,
        "schema_version": 94,
        "status": "passed",
        "passed": True,
        **prereg,
        "authenticated_sources": authenticate_sources_v94(config),
        "derived_contract": derive_contract_v94(config_path),
        "evaluation_label_file_opened": False,
        "full_gemma_model_loaded": False,
        "optimizer_constructed": False,
        "optimizer_updates": 0,
        "behavior_scored": False,
        "protected_or_sealed_behavior_artifacts_opened": [],
        "oracle_loaded": False,
        "runtime_promotion_authorized": False,
    }
    output = _atomic_create_json_v94(config["outputs"]["cpu_preflight"], report)
    return {**report, "output": output.as_posix()}


def authenticate_cpu_preflight_v94(
    config: Mapping[str, Any], *, config_path: str | Path = CONFIG
) -> dict[str, str]:
    prereg = authenticate_preregistration_v94(config, config_path=config_path)
    path = resolve_v85(config["outputs"]["cpu_preflight"])
    payload = _strict_json_v94(path)
    if (
        payload.get("artifact") != PREFLIGHT_ARTIFACT
        or payload.get("status") != "passed"
        or payload.get("passed") is not True
        or payload.get("config_sha256") != prereg["config_sha256"]
        or payload.get("preregistration_sha256") != prereg["preregistration_sha256"]
        or payload.get("evaluation_label_file_opened") is not False
        or payload.get("full_gemma_model_loaded") is not False
        or payload.get("optimizer_constructed") is not False
        or payload.get("optimizer_updates") != 0
        or payload.get("behavior_scored") is not False
        or payload.get("protected_or_sealed_behavior_artifacts_opened") != []
        or payload.get("oracle_loaded") is not False
        or payload.get("runtime_promotion_authorized") is not False
    ):
        raise ValueError("V94 CPU preflight changed")
    return {**prereg, "cpu_preflight_sha256": sha256_file_v85(path)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("derive", "preregister", "cpu-preflight", "authenticate")
    )
    parser.add_argument("--config", default=str(CONFIG))
    args = parser.parse_args(argv)
    if args.command == "derive":
        result: Mapping[str, Any] = derive_contract_v94(args.config)
    elif args.command == "preregister":
        result = build_preregistration_v94(args.config)
    elif args.command == "cpu-preflight":
        result = run_cpu_preflight_v94(args.config)
    else:
        config = load_config_v94(args.config, allow_draft=False)
        result = authenticate_cpu_preflight_v94(config, config_path=args.config)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CONFIG",
    "EXPECTED_EVALUATION_SCENES",
    "EXPECTED_INITIAL_STATE_SHA256",
    "FRESH_BANK_NAME",
    "TARGET_MODULE",
    "authenticate_cpu_preflight_v94",
    "authenticate_parent_v85_v94",
    "authenticate_pinned_model_tensor_v94",
    "authenticate_preregistration_v94",
    "authenticate_sources_v94",
    "build_preregistration_v94",
    "causal_sides_v94",
    "class_weights_v94",
    "derive_contract_v94",
    "load_config_v94",
    "load_scene_memories_v94",
    "load_training_rows_v94",
    "lora_preflight_v94",
    "main",
    "run_cpu_preflight_v94",
    "training_schedule_v94",
    "zero_payload_memory_v94",
]
