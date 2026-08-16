"""Seal and CPU-preflight the strict V85 multi-scene experiment.

Neither command loads Gemma, constructs an optimizer, performs training, or
opens any official/deferred/oracle evaluation source.  The create-once
preregistration binds the exact historical V73 train/development split, the
immutable 738x1536 scene memories, the deterministic one-epoch schedule, and
the sole fresh Gemma-side LoRA surface before MPS is authorized.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final, Literal

import torch
import yaml
from torch import nn

from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.language.lora import (
    LoRASettings,
    initialize_lora_adapter_state,
    install_lora_adapters,
)
from semantic_3d_chat.training.train_question_control_v73 import (
    ChangedUnitV73,
    RowV73,
    changed_units_v73,
    load_config_v73,
    load_training_rows_v73,
    split_rows_v73,
)
from semantic_3d_chat.training.v82_reader_artifacts import load_v82_cache

CONFIG: Final[Path] = Path("configs/experiments/gemma4_v85_strict_multiscene.yaml")
TARGET_MODULE: Final[str] = "model.language_model.layers.34.mlp.down_proj"
FRESH_BANK_NAME: Final[str] = "v85_strict_multiscene_bridge"
PREREG_ARTIFACT: Final[str] = "gemma4_v85_strict_multiscene_preregistration_v1"
PREFLIGHT_ARTIFACT: Final[str] = "gemma4_v85_strict_multiscene_cpu_preflight_v1"
_HEX64: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{64}")
FORBIDDEN_EVALUATION_PATHS: Final[tuple[str, ...]] = (
    "reports/gemma4/predictions/v83_direct_historical_internal.json",
    "reports/gemma4/metrics/v83_direct_historical_internal_score.json",
    "reports/gemma4/predictions/v82_historical_internal.json",
    "reports/gemma4/metrics/v82_historical_internal_score.json",
    "data/oracle",
)


def resolve_v85(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return (value if value.is_absolute() else PROJECT_ROOT / value).resolve()


def sha256_file_v85(path: str | Path) -> str:
    digest = hashlib.sha256()
    with resolve_v85(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256_v85(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_create_json_v85(
    path: str | Path, payload: Mapping[str, Any]
) -> tuple[Path, str]:
    destination = resolve_v85(path)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"V85 create-once JSON exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
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
    return destination, hashlib.sha256(encoded).hexdigest()


def _strict_json(path: str | Path) -> dict[str, Any]:
    source = resolve_v85(path)
    if source.is_symlink() or not source.is_file():
        raise FileNotFoundError(source)
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"V85 JSON must contain one object: {source}")
    return value


def load_config_v85(path: str | Path = CONFIG) -> dict[str, Any]:
    source = resolve_v85(path)
    if source.is_symlink() or not source.is_file():
        raise FileNotFoundError(source)
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or set(payload) != {"v85"}:
        raise ValueError("V85 config must contain exactly one v85 mapping")
    config = payload["v85"]
    if not isinstance(config, Mapping):
        raise TypeError("V85 config payload must be a mapping")
    if (
        config.get("schema_version") != 85
        or config.get("artifact")
        != "gemma4_v85_strict_multiscene_direct_memory_lora_v1"
        or config.get("seed") != 850085
    ):
        raise ValueError("V85 experiment identity changed")
    strict = config.get("strict_input_contract")
    if not isinstance(strict, Mapping) or strict != {
        "layout": [
            "boi",
            "96_groups_each_probe_key_plus_four_scene_values",
            "all_256_base_latents",
            "eoi",
        ],
        "shape_per_scene": [1, 738, 1536],
        "compiled_before_question": True,
        "reused_byte_identically_across_questions": True,
        "supplied_directly_to_native_gemma_image_prefix": True,
        "all_738_memory_slots_retained": True,
        "memory_projector_enabled": False,
        "question_derived_environmental_tokens": 0,
        "question_conditioned_environmental_readout": False,
        "question_dependent_scene_processing": False,
        "question_dependent_retrieval": False,
        "semantic_or_spatial_top_k_selection": False,
        "control_tokens": 0,
        "environmental_text_inputs": [],
        "gemma_is_only_question_dependent_consumer": True,
    }:
        raise ValueError("V85 strict direct-memory contract changed")
    bridge = config.get("bridge")
    if not isinstance(bridge, Mapping) or (
        bridge.get("bank_name") != FRESH_BANK_NAME
        or bridge.get("target_module") != TARGET_MODULE
        or bridge.get("rank") != 4
        or bridge.get("alpha") != 8.0
        or bridge.get("dropout") != 0.0
        or bridge.get("trainable_parameter_count") != 55_296
        or bridge.get("initialization_seed") != 840084
        or bridge.get("expected_initial_state_sha256")
        != "1ec186d64cab68a3ea2000968a0ca643e591cc32669c6b1b7138deb365cc5cc1"
        or bridge.get("starts_from_v84_candidate") is not False
        or bridge.get("base_gemma_frozen") is not True
        or bridge.get("inherited_v54_lora_banks_frozen") is not True
        or bridge.get("merged_weights") is not False
    ):
        raise ValueError("V85 sole fresh bridge contract changed")
    split = config.get("split")
    if not isinstance(split, Mapping) or any(
        split.get(key) != expected
        for key, expected in {
            "train_pair_count": 12,
            "train_scene_count": 24,
            "train_row_count": 576,
            "train_changed_unit_count": 40,
            "train_changed_side_count": 80,
            "development_pair_count": 8,
            "development_scene_count": 16,
            "development_row_count": 384,
            "development_changed_unit_count": 26,
            "development_changed_side_count": 52,
            "pair_disjoint": True,
            "scene_disjoint": True,
            "development_behavior_opened_during_training": False,
        }.items()
    ):
        raise ValueError("V85 exact historical split contract changed")
    training = config.get("training")
    if not isinstance(training, Mapping) or any(
        training.get(key) != expected
        for key, expected in {
            "epochs": 1,
            "all_train_rows_consumed_exactly_once": True,
            "row_order_seed": 850085,
            "row_order_sha256": (
                "f98a22a06988bc2f7656dcbcd8dcb0024a215b11cc45d622be6c820e2107d2ff"
            ),
            "microbatch_size": 1,
            "gradient_accumulation_rows": 8,
            "optimizer_updates": 72,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "gradient_clip_norm": 1.0,
            "correct_scene_answer_ce_weight": 1.0,
            "changed_side_wrong_scene_margin_weight": 1.0,
            "changed_side_wrong_scene_target_margin_nll": 0.5,
            "unchanged_rows_receive_margin": False,
            "checkpoint_every_optimizer_updates": 12,
            "deterministic_resume": True,
            "checkpoint_selection": "fixed_final_update_72",
            "development_driven_checkpoint_selection": False,
            "intermediate_selection": False,
        }.items()
    ):
        raise ValueError("V85 fixed one-epoch training protocol changed")
    gates = config.get("runtime_candidate_gates")
    if not isinstance(gates, Mapping) or gates != {
        "purpose": "authorize_separate_leakage_runtime_packaging_only",
        "canonical_accuracy_minimum": 0.40,
        "canonical_accuracy_margin_over_answer_frequency_majority": 0.05,
        "canonical_accuracy_threshold_rule": (
            "max_fixed_minimum_or_majority_plus_margin"
        ),
        "spatial_relation_accuracy_minimum": 0.45,
        "spatial_relation_minimum_row_count": 20,
        "mean_changed_side_wrong_minus_correct_nll_strictly_positive": True,
        "complete_changed_units_minimum": 4,
        "canonical_prediction_changing_units_minimum": 8,
        "exact_prefix_hash_invariance_required": True,
        "every_development_memory_hash_retained_required": True,
        "protected_read_count_maximum": 0,
        "separate_leakage_runtime_packaging_authorized_on_pass": True,
        "automatic_runtime_promotion": False,
    }:
        raise ValueError("V85 preregistered runtime-candidate gates changed")
    sources = config.get("sources")
    if not isinstance(sources, Mapping):
        raise TypeError("V85 sources must be a mapping")
    for field in (
        "preflight_source_sha256",
        "training_source_sha256",
        "evaluation_source_sha256",
    ):
        value = sources.get(field)
        if not isinstance(value, str) or _HEX64.fullmatch(value) is None or set(value) == {"0"}:
            raise ValueError(f"V85 {field} is not sealed")
    scope = config.get("scope")
    if not isinstance(scope, Mapping) or any(
        scope.get(field) is not False
        for field in (
            "cloud_inference",
            "official_validation_loaded",
            "official_test_loaded",
            "deferred_final_loaded",
            "sealed_historical_16_loaded",
            "oracle_loaded",
            "runtime_promotion_authorized",
        )
    ):
        raise ValueError("V85 protected scope changed")
    return dict(config)


def _row_inventory(rows: Sequence[RowV73]) -> list[list[str]]:
    return sorted([[row.scene_id, row.question_id] for row in rows])


def _changed_inventory(units: Sequence[ChangedUnitV73]) -> list[list[str]]:
    return sorted(
        [
            [
                unit.pair_id,
                unit.question_key,
                unit.left.scene_id,
                unit.left.question_id,
                unit.right.scene_id,
                unit.right.question_id,
            ]
            for unit in units
        ]
    )


def ordered_training_rows_v85(
    rows: Sequence[RowV73], *, seed: int = 850085
) -> tuple[RowV73, ...]:
    ordered = sorted(rows, key=lambda row: row.key)
    random.Random(seed).shuffle(ordered)
    return tuple(ordered)


def _authenticate_sources(config: Mapping[str, Any]) -> dict[str, str]:
    sources = config["sources"]
    expected = {
        sources["parent_v84_config"]: sources["parent_v84_config_sha256"],
        sources["parent_v84_wiring_report"]: sources[
            "parent_v84_wiring_report_sha256"
        ],
        sources["parent_v84_pair_margin_report"]: sources[
            "parent_v84_pair_margin_report_sha256"
        ],
        sources["runtime_config"]: sources["runtime_config_sha256"],
        sources["v73_config"]: sources["v73_config_sha256"],
        sources["historical_qa"]: sources["historical_qa_sha256"],
        str(Path(sources["train_memory_cache"]) / "training_tensors.safetensors"): sources[
            "train_memory_tensor_sha256"
        ],
        str(Path(sources["train_memory_cache"]) / "metadata.json"): sources[
            "train_memory_metadata_sha256"
        ],
        str(
            Path(sources["development_memory_cache"])
            / "training_tensors.safetensors"
        ): sources["development_memory_tensor_sha256"],
        str(Path(sources["development_memory_cache"]) / "metadata.json"): sources[
            "development_memory_metadata_sha256"
        ],
        str(Path(sources["base_checkpoint"]) / "adapter.safetensors"): sources[
            "base_adapter_sha256"
        ],
        str(Path(sources["base_checkpoint"]) / "runtime_metadata.json"): sources[
            "base_runtime_metadata_sha256"
        ],
        sources["v83_runtime_source"]: sources["v83_runtime_source_sha256"],
        sources["preflight_source"]: sources["preflight_source_sha256"],
        sources["training_source"]: sources["training_source_sha256"],
        sources["evaluation_source"]: sources["evaluation_source_sha256"],
    }
    observed: dict[str, str] = {}
    for path, expected_sha256 in expected.items():
        observed_sha256 = sha256_file_v85(path)
        if observed_sha256 != expected_sha256:
            raise ValueError(f"V85 pinned source changed: {path}")
        observed[str(path)] = observed_sha256

    snapshot = (
        Path.home()
        / ".cache/huggingface/hub/models--google--gemma-4-E2B-it/snapshots"
        / str(sources["model_revision"])
    )
    blob = (snapshot / "model.safetensors").resolve(strict=True)
    if blob.name != sources["model_blob_sha256_identity"]:
        raise ValueError("V85 local Gemma blob identity changed")
    model_config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    text = model_config.get("text_config")
    if not isinstance(text, Mapping) or (
        text.get("hidden_size") != 1_536
        or text.get("intermediate_size") != 6_144
        or text.get("use_double_wide_mlp") is not True
        or text.get("num_hidden_layers") != 35
        or text.get("sliding_window") != 512
        or len(text.get("layer_types", ())) != 35
        or text["layer_types"][34] != "full_attention"
    ):
        raise ValueError("V85 pinned Gemma decoder topology changed")
    observed["gemma_model_blob_sha256_identity"] = blob.name
    return observed


def split_preflight_v85(
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], tuple[RowV73, ...], tuple[RowV73, ...]]:
    sources = config["sources"]
    load_config_v73(sources["v73_config"])
    train, development = split_rows_v73(
        load_training_rows_v73(sources["historical_qa"])
    )
    train_units = changed_units_v73(train)
    development_units = changed_units_v73(development)
    train_scenes = sorted({row.scene_id for row in train})
    development_scenes = sorted({row.scene_id for row in development})
    train_pairs = sorted({row.pair_id for row in train})
    development_pairs = sorted({row.pair_id for row in development})
    split = config["split"]
    observed_hashes = {
        "train_inventory_sha256": canonical_sha256_v85(_row_inventory(train)),
        "train_changed_units_sha256": canonical_sha256_v85(
            _changed_inventory(train_units)
        ),
        "development_inventory_sha256": canonical_sha256_v85(
            _row_inventory(development)
        ),
        "development_changed_units_sha256": canonical_sha256_v85(
            _changed_inventory(development_units)
        ),
    }
    if any(observed_hashes[key] != split[key] for key in observed_hashes):
        raise ValueError("V85 row or changed-unit inventory changed")
    if (
        len(train) != split["train_row_count"]
        or len(train_scenes) != split["train_scene_count"]
        or len(train_pairs) != split["train_pair_count"]
        or len(train_units) != split["train_changed_unit_count"]
        or sum(row.expected_change for row in train)
        != split["train_changed_side_count"]
        or len(development) != split["development_row_count"]
        or len(development_scenes) != split["development_scene_count"]
        or len(development_pairs) != split["development_pair_count"]
        or len(development_units) != split["development_changed_unit_count"]
        or sum(row.expected_change for row in development)
        != split["development_changed_side_count"]
        or set(train_scenes) & set(development_scenes)
        or set(train_pairs) & set(development_pairs)
    ):
        raise ValueError("V85 train/development split changed")

    schedule = ordered_training_rows_v85(
        train, seed=int(config["training"]["row_order_seed"])
    )
    schedule_hash = canonical_sha256_v85(
        [[row.scene_id, row.question_id] for row in schedule]
    )
    if schedule_hash != config["training"]["row_order_sha256"]:
        raise ValueError("V85 deterministic row schedule changed")

    cache_records: dict[str, Any] = {}
    for split_name, rows, scene_ids, cache_path in (
        ("train", train, train_scenes, sources["train_memory_cache"]),
        (
            "development",
            development,
            development_scenes,
            sources["development_memory_cache"],
        ),
    ):
        cache = load_v82_cache(resolve_v85(cache_path))
        if cache.metadata["scene_ids"] != scene_ids:
            raise ValueError(f"V85 {split_name} memory scene inventory changed")
        expected_indices = torch.tensor(
            [scene_ids.index(row.scene_id) for row in rows], dtype=torch.int64
        )
        if not torch.equal(cache.tensors["row_scene_indices"], expected_indices):
            raise ValueError(f"V85 {split_name} QA/memory row alignment changed")
        memories = cache.tensors["scene_memories"]
        if tuple(memories.shape[1:]) != (738, 1536) or memories.dtype != torch.bfloat16:
            raise ValueError(f"V85 {split_name} memory shape or dtype changed")
        cache_records[split_name] = {
            "scene_memory_shape": list(memories.shape),
            "scene_memory_dtype": str(memories.dtype),
            "row_scene_indices_shape": list(cache.tensors["row_scene_indices"].shape),
        }

    answer_counts = Counter(row.answer_class for row in development)
    majority = max(answer_counts.values()) / len(development)
    report = {
        "train_rows": len(train),
        "train_scenes": train_scenes,
        "train_pairs": train_pairs,
        "train_changed_units": len(train_units),
        "train_changed_sides": sum(row.expected_change for row in train),
        "development_rows": len(development),
        "development_scenes": development_scenes,
        "development_pairs": development_pairs,
        "development_changed_units": len(development_units),
        "development_changed_sides": sum(row.expected_change for row in development),
        "pair_and_scene_disjoint": True,
        "inventory_hashes": observed_hashes,
        "row_order_sha256": schedule_hash,
        "row_order_first_keys": [list(row.key) for row in schedule[:3]],
        "row_order_last_keys": [list(row.key) for row in schedule[-3:]],
        "cache": cache_records,
        "development_answer_frequency_majority_baseline": majority,
        "development_answer_text_serialized": False,
        "development_behavior_scored": False,
    }
    return report, tuple(train), tuple(development)


def load_scene_memories_v85(
    config: Mapping[str, Any],
    rows: Sequence[RowV73],
    *,
    split_name: Literal["train", "development"],
) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    """Compile every requested immutable memory before question tokenization."""

    cache_key = (
        "train_memory_cache"
        if split_name == "train"
        else "development_memory_cache"
    )
    cache = load_v82_cache(resolve_v85(config["sources"][cache_key]))
    scene_ids = list(cache.metadata["scene_ids"])
    requested = sorted({row.scene_id for row in rows})
    if requested != scene_ids:
        raise ValueError(f"V85 {split_name} requested scene inventory changed")
    memories: dict[str, torch.Tensor] = {}
    hashes: dict[str, str] = {}
    from semantic_3d_chat.language.prefix_injection import prefix_sha256

    for scene_id, memory in zip(scene_ids, cache.tensors["scene_memories"], strict=True):
        fixed = memory.unsqueeze(0).detach().cpu().contiguous()
        if tuple(fixed.shape) != (1, 738, 1536) or fixed.dtype != torch.bfloat16:
            raise ValueError("V85 immutable scene-memory contract changed")
        memories[scene_id] = fixed
        hashes[scene_id] = prefix_sha256(fixed)
    return memories, hashes


class _SyntheticDecoderLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.down_proj = nn.Linear(12_288, 1_536, bias=False, dtype=torch.bfloat16)


class _SyntheticMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = _SyntheticDecoderLayer()


class _SyntheticLanguage(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.Identity() for _ in range(34)] + [_SyntheticMLP()])


class _SyntheticGemma(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = _SyntheticLanguage()


def lora_preflight_v85(config: Mapping[str, Any]) -> dict[str, Any]:
    bridge = config["bridge"]
    settings = LoRASettings(
        enabled=True,
        rank=int(bridge["rank"]),
        alpha=float(bridge["alpha"]),
        dropout=float(bridge["dropout"]),
        target_modules=(str(bridge["target_module"]),),
    )
    installation = install_lora_adapters(_SyntheticGemma(), settings)
    if installation is None:
        raise RuntimeError("V85 synthetic LoRA installation failed")
    initialize_lora_adapter_state(
        installation, seed=int(bridge["initialization_seed"])
    )
    state_sha256 = installation.state_sha256()
    if (
        installation.parameter_count != bridge["trainable_parameter_count"]
        or state_sha256 != bridge["expected_initial_state_sha256"]
        or any(
            torch.count_nonzero(adapter.lora_b).item() != 0
            for adapter in installation.adapters
        )
    ):
        raise RuntimeError("V85 deterministic zero-output LoRA preflight failed")
    return {
        "bank_name": FRESH_BANK_NAME,
        "target_modules": list(installation.target_names),
        "parameter_count": installation.parameter_count,
        "initial_state_sha256": state_sha256,
        "exact_zero_output_at_initialization": True,
        "base_projection_shape": [1536, 12288],
        "full_gemma_model_loaded": False,
    }


def _validate_parent_v84(config: Mapping[str, Any]) -> dict[str, Any]:
    wiring = _strict_json(config["sources"]["parent_v84_wiring_report"])
    pair = _strict_json(config["sources"]["parent_v84_pair_margin_report"])
    if (
        wiring.get("optimizer_updates") != 4
        or wiring.get("runtime_promotion_authorized") is not False
        or pair.get("optimizer_updates") != 32
        or pair.get("passed") is not True
        or pair.get("runtime_promotion_authorized") is not False
        or pair.get("development_behavior_scored") is not False
    ):
        raise ValueError("V85 V84 rationale or non-promotion boundary changed")
    return {
        "v84_wiring_optimizer_updates": 4,
        "v84_pair_margin_optimizer_updates": 32,
        "v84_pair_margin_train_only_wiring_passed": True,
        "v84_candidates_runtime_promoted": False,
        "v85_starts_from_v84_candidate": False,
    }


def build_preregistration_v85(
    config_path: str | Path = CONFIG,
) -> dict[str, Any]:
    config = load_config_v85(config_path)
    sources = _authenticate_sources(config)
    split, train, development = split_preflight_v85(config)
    lora = lora_preflight_v85(config)
    parent = _validate_parent_v84(config)
    train_memories, train_hashes = load_scene_memories_v85(
        config, train, split_name="train"
    )
    development_memories, development_hashes = load_scene_memories_v85(
        config, development, split_name="development"
    )
    if len(train_memories) != 24 or len(development_memories) != 16:
        raise RuntimeError("V85 did not bind all forty pre-question scene memories")
    config_sha256 = sha256_file_v85(config_path)
    payload = {
        "artifact": PREREG_ARTIFACT,
        "schema_version": 85,
        "status": "sealed_before_first_v85_model_training_or_development_behavior",
        "config_path": Path(config_path).as_posix(),
        "config_sha256": config_sha256,
        "authenticated_sources": sources,
        "parent_v84_evidence": parent,
        "strict_input_contract": config["strict_input_contract"],
        "split_contract": config["split"],
        "split_preflight": split,
        "bridge": config["bridge"],
        "lora_cpu_preflight": lora,
        "fixed_training_protocol": config["training"],
        "fixed_development_protocol": config["development"],
        "fixed_runtime_candidate_gates": config["runtime_candidate_gates"],
        "scene_memories": {
            "compiled_before_question_tokenization": True,
            "shape_each": [1, 738, 1536],
            "train_hashes": train_hashes,
            "development_hashes": development_hashes,
            "all_memory_slots_retained": True,
        },
        "forbidden_evaluation_paths_not_opened": list(FORBIDDEN_EVALUATION_PATHS),
        "full_gemma_model_loaded": False,
        "optimizer_constructed": False,
        "optimizer_updates": 0,
        "development_behavior_scored": False,
        "official_validation_loaded": False,
        "official_test_loaded": False,
        "deferred_final_loaded": False,
        "sealed_historical_16_loaded": False,
        "oracle_loaded": False,
        "runtime_promotion_authorized": False,
    }
    output, _sha = atomic_create_json_v85(config["outputs"]["preregistration"], payload)
    payload["output"] = output.relative_to(PROJECT_ROOT).as_posix()
    return payload


def authenticate_preregistration_v85(
    config: Mapping[str, Any], *, config_path: str | Path = CONFIG
) -> dict[str, str]:
    path = resolve_v85(config["outputs"]["preregistration"])
    payload = _strict_json(path)
    config_sha256 = sha256_file_v85(config_path)
    if (
        payload.get("artifact") != PREREG_ARTIFACT
        or payload.get("status")
        != "sealed_before_first_v85_model_training_or_development_behavior"
        or payload.get("config_sha256") != config_sha256
        or payload.get("full_gemma_model_loaded") is not False
        or payload.get("optimizer_constructed") is not False
        or payload.get("optimizer_updates") != 0
        or payload.get("development_behavior_scored") is not False
        or payload.get("official_validation_loaded") is not False
        or payload.get("official_test_loaded") is not False
        or payload.get("deferred_final_loaded") is not False
        or payload.get("sealed_historical_16_loaded") is not False
        or payload.get("oracle_loaded") is not False
        or payload.get("runtime_promotion_authorized") is not False
    ):
        raise ValueError("V85 preregistration changed")
    return {
        "config_sha256": config_sha256,
        "preregistration_sha256": sha256_file_v85(path),
    }


def run_cpu_preflight_v85(
    config_path: str | Path = CONFIG,
) -> dict[str, Any]:
    config = load_config_v85(config_path)
    prereg = authenticate_preregistration_v85(config, config_path=config_path)
    sources = _authenticate_sources(config)
    split, _train, _development = split_preflight_v85(config)
    lora = lora_preflight_v85(config)
    report = {
        "artifact": PREFLIGHT_ARTIFACT,
        "schema_version": 85,
        "status": "passed",
        "passed": True,
        **prereg,
        "authenticated_sources": sources,
        "split_preflight": split,
        "lora_preflight": lora,
        "fixed_final_optimizer_updates": config["training"]["optimizer_updates"],
        "fixed_final_checkpoint_selection": config["training"]["checkpoint_selection"],
        "deterministic_resume_preregistered": config["training"]["deterministic_resume"],
        "all_train_rows_consumed_exactly_once": True,
        "fixed_memory_compiled_before_questions": True,
        "all_memory_slots_retained": True,
        "question_derived_environmental_tokens": 0,
        "question_conditioned_environmental_readout": False,
        "full_gemma_model_loaded": False,
        "optimizer_constructed": False,
        "optimizer_updates": 0,
        "development_behavior_scored": False,
        "protected_or_sealed_behavior_artifacts_opened": [],
        "official_validation_loaded": False,
        "official_test_loaded": False,
        "deferred_final_loaded": False,
        "sealed_historical_16_loaded": False,
        "oracle_loaded": False,
        "runtime_promotion_authorized": False,
    }
    output, _sha = atomic_create_json_v85(config["outputs"]["cpu_preflight"], report)
    report["output"] = output.relative_to(PROJECT_ROOT).as_posix()
    return report


def authenticate_cpu_preflight_v85(
    config: Mapping[str, Any], *, config_path: str | Path = CONFIG
) -> dict[str, str]:
    prereg = authenticate_preregistration_v85(config, config_path=config_path)
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
        or payload.get("development_behavior_scored") is not False
        or payload.get("protected_or_sealed_behavior_artifacts_opened") != []
        or payload.get("oracle_loaded") is not False
        or payload.get("runtime_promotion_authorized") is not False
    ):
        raise ValueError("V85 CPU preflight changed")
    return {
        **prereg,
        "cpu_preflight_sha256": sha256_file_v85(path),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preregister", "preflight"))
    parser.add_argument("--config", default=str(CONFIG))
    args = parser.parse_args(argv)
    result = (
        build_preregistration_v85(args.config)
        if args.command == "preregister"
        else run_cpu_preflight_v85(args.config)
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CONFIG",
    "FORBIDDEN_EVALUATION_PATHS",
    "FRESH_BANK_NAME",
    "PREFLIGHT_ARTIFACT",
    "PREREG_ARTIFACT",
    "TARGET_MODULE",
    "atomic_create_json_v85",
    "authenticate_cpu_preflight_v85",
    "authenticate_preregistration_v85",
    "build_preregistration_v85",
    "canonical_sha256_v85",
    "load_config_v85",
    "load_scene_memories_v85",
    "lora_preflight_v85",
    "main",
    "ordered_training_rows_v85",
    "resolve_v85",
    "run_cpu_preflight_v85",
    "sha256_file_v85",
    "split_preflight_v85",
]
