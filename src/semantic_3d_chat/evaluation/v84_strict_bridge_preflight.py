"""Preregister and CPU-preflight the strict fixed-memory V84 bridge.

This phase never loads Gemma weights, executes optimization, opens protected
evaluation artifacts, or reads oracle data.  It binds the train/development
split, exact 738-token memories, one fixed wiring unit, and the sole fresh LoRA
surface before the first full-model measurement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

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
    changed_units_v73,
    load_config_v73,
    load_training_rows_v73,
    split_rows_v73,
)
from semantic_3d_chat.training.v82_reader_artifacts import load_v82_cache

CONFIG: Final[Path] = Path(
    "configs/experiments/gemma4_v84_strict_fixed_memory_bridge.yaml"
)
TARGET_MODULE: Final[str] = "model.language_model.layers.34.mlp.down_proj"
FORBIDDEN_EVALUATION_PATHS: Final[tuple[str, ...]] = (
    "reports/gemma4/predictions/v83_direct_historical_internal.json",
    "reports/gemma4/metrics/v83_direct_historical_internal_score.json",
    "reports/gemma4/predictions/v82_historical_internal.json",
    "reports/gemma4/metrics/v82_historical_internal_score.json",
)


def _resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return (value if value.is_absolute() else PROJECT_ROOT / value).resolve()


def sha256_file_v84(path: str | Path) -> str:
    digest = hashlib.sha256()
    with _resolve(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_create_json(path: str | Path, payload: Mapping[str, Any]) -> tuple[Path, str]:
    destination = _resolve(path)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"V84 create-once output exists: {destination}")
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


def load_config_v84(path: str | Path = CONFIG) -> dict[str, Any]:
    source = _resolve(path)
    if source.is_symlink() or not source.is_file():
        raise FileNotFoundError(source)
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or set(payload) != {"v84"}:
        raise ValueError("V84 config must contain exactly one v84 mapping")
    config = payload["v84"]
    if not isinstance(config, Mapping):
        raise TypeError("V84 config payload must be a mapping")
    if (
        config.get("schema_version") != 84
        or config.get("artifact") != "gemma4_v84_strict_fixed_total_input_bridge_v1"
        or config.get("seed") != 840084
    ):
        raise ValueError("V84 experiment identity changed")
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
        raise ValueError("V84 strict total-input contract changed")
    bridge = config.get("bridge")
    if not isinstance(bridge, Mapping) or (
        bridge.get("target_module") != TARGET_MODULE
        or bridge.get("rank") != 4
        or bridge.get("alpha") != 8.0
        or bridge.get("trainable_parameter_count") != 55_296
        or bridge.get("expected_initial_state_sha256")
        != "1ec186d64cab68a3ea2000968a0ca643e591cc32669c6b1b7138deb365cc5cc1"
    ):
        raise ValueError("V84 bridge surface changed")
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
        raise ValueError("V84 protected scope changed")
    return dict(config)


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


def _preflight_lora(config: Mapping[str, Any]) -> dict[str, Any]:
    bridge = config["bridge"]
    settings = LoRASettings(
        enabled=True,
        rank=int(bridge["rank"]),
        alpha=float(bridge["alpha"]),
        dropout=float(bridge["dropout"]),
        target_modules=(str(bridge["target_module"]),),
    )
    model = _SyntheticGemma()
    installation = install_lora_adapters(model, settings)
    if installation is None:
        raise RuntimeError("V84 synthetic LoRA installation failed")
    initialize_lora_adapter_state(installation, seed=int(bridge["initialization_seed"]))
    state_sha256 = installation.state_sha256()
    if (
        installation.parameter_count != int(bridge["trainable_parameter_count"])
        or state_sha256 != bridge["expected_initial_state_sha256"]
        or any(torch.count_nonzero(adapter.lora_b) for adapter in installation.adapters)
    ):
        raise RuntimeError("V84 deterministic zero-output LoRA preflight failed")
    return {
        "target_modules": list(installation.target_names),
        "parameter_count": installation.parameter_count,
        "initial_state_sha256": state_sha256,
        "exact_zero_output_at_initialization": True,
        "base_projection_shape": [1_536, 12_288],
        "base_projection_materialized_for_cpu_shape_preflight_only": True,
    }


def _authenticate_sources(config: Mapping[str, Any]) -> dict[str, str]:
    sources = config["sources"]
    expected = {
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
            Path(sources["development_memory_cache"]) / "training_tensors.safetensors"
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
    }
    observed: dict[str, str] = {}
    for path, expected_sha256 in expected.items():
        observed_sha256 = sha256_file_v84(path)
        if observed_sha256 != expected_sha256:
            raise ValueError(f"V84 pinned source changed: {path}")
        observed[str(path)] = observed_sha256
    snapshot = (
        Path.home()
        / ".cache/huggingface/hub/models--google--gemma-4-E2B-it/snapshots"
        / str(sources["model_revision"])
    )
    blob = (snapshot / "model.safetensors").resolve(strict=True)
    if blob.name != sources["model_blob_sha256_identity"]:
        raise ValueError("V84 local Gemma blob identity changed")
    observed["gemma_model_blob_sha256_identity"] = blob.name
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
        raise ValueError("V84 pinned Gemma decoder topology changed")
    return observed


def _split_preflight(config: Mapping[str, Any]) -> dict[str, Any]:
    sources = config["sources"]
    load_config_v73(sources["v73_config"])
    rows = load_training_rows_v73(sources["historical_qa"])
    train, development = split_rows_v73(rows)
    train_cache = load_v82_cache(_resolve(sources["train_memory_cache"]))
    development_cache = load_v82_cache(_resolve(sources["development_memory_cache"]))
    expected = config["split"]
    train_scenes = sorted({row.scene_id for row in train})
    development_scenes = sorted({row.scene_id for row in development})
    if (
        len(train) != expected["train_row_count"]
        or len(train_scenes) != expected["train_scene_count"]
        or len(development) != expected["development_row_count"]
        or len(development_scenes) != expected["development_scene_count"]
        or train_cache.metadata["scene_ids"] != train_scenes
        or development_cache.metadata["scene_ids"] != development_scenes
        or set(train_scenes) & set(development_scenes)
    ):
        raise ValueError("V84 train/development memory split changed")
    for selected_rows, cache, scene_ids in (
        (train, train_cache, train_scenes),
        (development, development_cache, development_scenes),
    ):
        expected_indices = torch.tensor(
            [scene_ids.index(row.scene_id) for row in selected_rows], dtype=torch.int64
        )
        if not torch.equal(cache.tensors["row_scene_indices"], expected_indices):
            raise ValueError("V84 QA/memory row alignment changed")

    train_units = sorted(
        changed_units_v73(train), key=lambda unit: (unit.change_type, unit.pair_id, unit.question_key)
    )
    wiring = train_units[0]
    configured_wiring = config["wiring"]
    wiring_inventory = [
        [wiring.left.scene_id, wiring.left.question_id],
        [wiring.right.scene_id, wiring.right.question_id],
    ]
    if (
        wiring.change_type != configured_wiring["selected_change_type"]
        or wiring.pair_id != configured_wiring["selected_pair_id"]
        or wiring.question_key != configured_wiring["selected_question_key"]
        or wiring_inventory != configured_wiring["selected_rows"]
        or wiring.left.question != wiring.right.question
        or wiring.left.answer == wiring.right.answer
    ):
        raise ValueError("V84 fixed wiring-unit selection changed")

    development_units = changed_units_v73(development)
    families = sorted({unit.change_type for unit in development_units})
    selected_development = [
        min(
            (unit for unit in development_units if unit.change_type == family),
            key=lambda unit: (unit.pair_id, unit.question_key),
        )
        for family in families
    ]
    if len(selected_development) != config["development"]["selected_change_family_count"]:
        raise ValueError("V84 fixed development-family selection changed")
    return {
        "train_rows": len(train),
        "train_scenes": train_scenes,
        "development_rows": len(development),
        "development_scenes": development_scenes,
        "pair_and_scene_disjoint": True,
        "train_memory_shape": list(train_cache.tensors["scene_memories"].shape),
        "development_memory_shape": list(
            development_cache.tensors["scene_memories"].shape
        ),
        "wiring_unit": {
            "change_type": wiring.change_type,
            "pair_id": wiring.pair_id,
            "question_key": wiring.question_key,
            "question_sha256": hashlib.sha256(wiring.left.question.encode()).hexdigest(),
            "row_inventory": wiring_inventory,
            "answer_text_serialized": False,
        },
        "development_selection": [
            {
                "change_type": unit.change_type,
                "pair_id": unit.pair_id,
                "question_key": unit.question_key,
                "row_inventory": [
                    [unit.left.scene_id, unit.left.question_id],
                    [unit.right.scene_id, unit.right.question_id],
                ],
            }
            for unit in selected_development
        ],
    }


def run_preflight(config_path: str | Path = CONFIG) -> dict[str, Any]:
    config = load_config_v84(config_path)
    sources = _authenticate_sources(config)
    split = _split_preflight(config)
    lora = _preflight_lora(config)
    config_sha256 = sha256_file_v84(config_path)
    preregistration = {
        "artifact": "gemma4_v84_strict_bridge_preregistration_v1",
        "schema_version": 84,
        "status": "sealed_before_first_full_model_measurement",
        "config_path": Path(config_path).as_posix(),
        "config_sha256": config_sha256,
        "strict_input_contract": config["strict_input_contract"],
        "split": config["split"],
        "bridge": config["bridge"],
        "wiring": config["wiring"],
        "development": config["development"],
        "gates": config["gates"],
        "forbidden_evaluation_paths_not_opened": list(FORBIDDEN_EVALUATION_PATHS),
        "full_gemma_model_loaded": False,
        "optimizer_constructed": False,
        "optimizer_updates": 0,
        "official_validation_loaded": False,
        "official_test_loaded": False,
        "deferred_final_loaded": False,
        "sealed_historical_16_loaded": False,
        "oracle_loaded": False,
        "runtime_promotion_authorized": False,
    }
    prereg_path, prereg_sha = _atomic_create_json(
        config["outputs"]["preregistration"], preregistration
    )
    report = {
        "artifact": "gemma4_v84_strict_bridge_cpu_preflight_v1",
        "schema_version": 84,
        "status": "passed",
        "config_sha256": config_sha256,
        "preregistration_path": prereg_path.relative_to(PROJECT_ROOT).as_posix(),
        "preregistration_sha256": prereg_sha,
        "authenticated_sources": sources,
        "split_preflight": split,
        "lora_preflight": lora,
        "fixed_memory_compiled_before_questions": True,
        "all_memory_slots_retained": True,
        "question_derived_environmental_tokens": 0,
        "question_conditioned_environmental_readout": False,
        "full_gemma_model_loaded": False,
        "optimizer_constructed": False,
        "optimizer_updates": 0,
        "protected_or_sealed_behavior_artifacts_opened": [],
        "oracle_loaded": False,
        "runtime_promotion_authorized": False,
        "passed": True,
    }
    output, _sha = _atomic_create_json(config["outputs"]["cpu_preflight"], report)
    report["output"] = output.relative_to(PROJECT_ROOT).as_posix()
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(CONFIG))
    args = parser.parse_args(argv)
    report = run_preflight(args.config)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CONFIG",
    "FORBIDDEN_EVALUATION_PATHS",
    "TARGET_MODULE",
    "load_config_v84",
    "main",
    "run_preflight",
    "sha256_file_v84",
]
