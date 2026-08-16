"""Seal and CPU-preflight the strict V86 scene-000001 demonstration.

V86 is deliberately a single-scene overfit acceptance experiment.  It keeps
the already-measured V85 scene-disjoint result as separate evidence and asks a
narrower engineering question: can one fixed 738-token continuous scene memory
drive a useful local chat demo without any environmental text at inference?

The commands in this module are model-free.  They authenticate every input,
bind all 138 training rows and their fixed schedule, prove the native BOI/EOI
zero-payload control, and verify the sole trainable LoRA surface before the one
authorized full-model training run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any, Final

import torch
import yaml
from safetensors.torch import load_file
from torch import nn

from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.evaluation.metrics import normalize_answer
from semantic_3d_chat.evaluation.v85_strict_multiscene_preflight import (
    atomic_create_json_v85,
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
from semantic_3d_chat.training.train_question_control_v73 import RowV73

CONFIG: Final[Path] = Path("configs/experiments/gemma4_v86_scene1_strict_demo.yaml")
SCENE_ID: Final[str] = "scene_000001"
FRESH_BANK_NAME: Final[str] = "v86_scene1_demo_bridge"
TARGET_MODULE: Final[str] = "model.language_model.layers.34.mlp.up_proj"
PREREG_ARTIFACT: Final[str] = "gemma4_v86_scene1_demo_preregistration_v1"
PREFLIGHT_ARTIFACT: Final[str] = "gemma4_v86_scene1_demo_cpu_preflight_v1"
EXPECTED_PREFIX_SHA256: Final[str] = (
    "a428f5147c815839ae7315a0adab952ab210814fb21dcdc5bf13b167f28a6e37"
)
CAUSAL_IDS: Final[tuple[str, ...]] = ("q_000080", "q_000108", "q_000014")
_HEX64: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{64}")


def _class_id(answer: str) -> str:
    return "answer_" + hashlib.sha256(answer.encode("utf-8")).hexdigest()[:20]


def _strict_json(path: str | Path) -> dict[str, Any]:
    source = resolve_v85(path)
    if source.is_symlink() or not source.is_file():
        raise FileNotFoundError(source)
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"V86 JSON must contain one object: {source}")
    return value


def load_config_v86(path: str | Path = CONFIG) -> dict[str, Any]:
    source = resolve_v85(path)
    if source.is_symlink() or not source.is_file():
        raise FileNotFoundError(source)
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or set(payload) != {"v86"}:
        raise ValueError("V86 config must contain exactly one v86 mapping")
    config = payload["v86"]
    if not isinstance(config, Mapping):
        raise TypeError("V86 config payload must be a mapping")
    if (
        config.get("schema_version") != 86
        or config.get("artifact") != "gemma4_v86_scene1_strict_direct_memory_overfit_v1"
        or config.get("status") != "preregistered_before_full_model_load"
        or config.get("seed") != 860086
    ):
        raise ValueError("V86 experiment identity is unsealed or changed")
    strict = config.get("strict_input_contract")
    required_strict = {
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
    }
    if strict != required_strict:
        raise ValueError("V86 direct-memory contract changed")
    dataset = config.get("dataset")
    if not isinstance(dataset, Mapping) or any(
        dataset.get(key) != value
        for key, value in {
            "scene_id": SCENE_ID,
            "row_count": 138,
            "all_scene1_rows_used": True,
            "answer_metadata_training_only": True,
            "runtime_serializes_questions_or_answers": False,
        }.items()
    ):
        raise ValueError("V86 scene-one dataset contract changed")
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
            "initialization_seed": 860086,
            "expected_initial_state_sha256": (
                "b5ec50969d4ce1d34fca9544248e208e48e74f03445c8efeea1c8e08202c4161"
            ),
            "disjoint_from_all_frozen_banks": True,
        }.items()
    ):
        raise ValueError("V86 sole fresh bridge contract changed")
    training = config.get("training")
    expected_training = {
        "optimizer": "AdamW",
        "epochs": 4,
        "rows_per_epoch": 138,
        "microbatch_size": 1,
        "gradient_accumulation_rows": 6,
        "optimizer_updates": 92,
        "row_order": "sorted_then_epoch_seeded_shuffle",
        "row_order_seed": 860086,
        "row_order_sha256": ("8f83098dc620b2576412168712d2e40e4ba6b1c1f5b1ecfa0be36d0402b3268d"),
        "learning_rate": 0.001,
        "weight_decay": 0.0,
        "gradient_clip_norm": 1.0,
        "answer_ce_weight": 1.0,
        "zero_payload_margin_weight": 1.0,
        "zero_payload_target_margin_nll": 0.5,
        "causal_subset_question_ids": list(CAUSAL_IDS),
        "zero_payload_preserves_native_boi_eoi": True,
        "zero_payload_zeros_exactly_736_interior_tokens": True,
        "causal_rows_per_epoch": 3,
        "total_causal_margin_rows": 12,
        "checkpoint_selection": "fixed_final_update_92",
        "intermediate_behavior_selection": False,
    }
    if not isinstance(training, Mapping) or any(
        training.get(key) != value for key, value in expected_training.items()
    ):
        raise ValueError("V86 fixed training protocol changed")
    gates = config.get("gates")
    if not isinstance(gates, Mapping) or any(
        gates.get(key) != value
        for key, value in {
            "all_scene1_canonical_accuracy_minimum": 0.80,
            "exact_training_row_count_required": 138,
            "live_smoke_minimum_correct": 2,
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
        raise ValueError("V86 fixed acceptance gates changed")
    if gates.get("live_smoke_questions") != [
        {"question": "Is there a chair?", "expected": "yes"},
        {"question": "What color is the bowl?", "expected": "red"},
        {
            "question": "Is the bowl left or right of the chair?",
            "expected": "left",
        },
    ]:
        raise ValueError("V86 corrected live-smoke oracle changed")
    sources = config.get("sources")
    if not isinstance(sources, Mapping):
        raise TypeError("V86 sources must be a mapping")
    for field in (
        "preflight_source_sha256",
        "training_source_sha256",
        "evaluation_source_sha256",
    ):
        value = sources.get(field)
        if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
            raise ValueError(f"V86 {field} is not sealed")
    scope = config.get("scope")
    if not isinstance(scope, Mapping) or scope != {
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
        raise ValueError("V86 protected scope changed")
    return dict(config)


def load_scene1_rows_v86(config: Mapping[str, Any]) -> tuple[RowV73, ...]:
    """Load only the explicitly training-authorized QA file and sanitize rows."""

    source = resolve_v85(config["sources"]["scene1_qa"])
    rows: list[RowV73] = []
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        raw = json.loads(line)
        if not isinstance(raw, Mapping):
            raise TypeError(f"V86 QA line {line_number} is not an object")
        if raw.get("scene_id") != SCENE_ID:
            continue
        required = {"scene_id", "question_id", "question", "answer", "answer_type"}
        if not required <= set(raw):
            raise ValueError(f"V86 QA fields changed at line {line_number}")
        if any(not isinstance(raw[key], str) or not raw[key] for key in required):
            raise TypeError(f"V86 QA string changed at line {line_number}")
        answer = normalize_answer(str(raw["answer"]))
        if not answer:
            raise ValueError(f"V86 QA answer normalizes empty at line {line_number}")
        rows.append(
            RowV73(
                scene_id=SCENE_ID,
                question_id=str(raw["question_id"]),
                question=str(raw["question"]),
                answer=answer,
                answer_class=_class_id(answer),
                answer_type=str(raw["answer_type"]),
                pair_id="v86_scene1_only",
                paired_scene_id=SCENE_ID,
                question_key=str(raw["question_id"]),
                change_type="none",
                expected_change=False,
            )
        )
    rows.sort(key=lambda row: row.question_id)
    dataset = config["dataset"]
    inventory_sha256 = canonical_sha256_v85([asdict(row) for row in rows])
    if (
        len(rows) != dataset["row_count"]
        or len({row.key for row in rows}) != len(rows)
        or inventory_sha256 != dataset["row_inventory_sha256"]
    ):
        raise ValueError("V86 exact 138-row scene-one inventory changed")
    return tuple(rows)


def training_schedule_v86(
    rows: Sequence[RowV73], *, seed: int = 860086, epochs: int = 4
) -> tuple[tuple[int, RowV73], ...]:
    schedule: list[tuple[int, RowV73]] = []
    for epoch in range(epochs):
        shuffled = sorted(rows, key=lambda row: row.question_id)
        random.Random(seed + epoch).shuffle(shuffled)
        schedule.extend((epoch, row) for row in shuffled)
    return tuple(schedule)


def causal_rows_v86(config: Mapping[str, Any], rows: Sequence[RowV73]) -> tuple[RowV73, ...]:
    by_id = {row.question_id: row for row in rows}
    selected = tuple(by_id[question_id] for question_id in CAUSAL_IDS)
    observed = canonical_sha256_v85([asdict(row) for row in selected])
    if observed != config["training"]["causal_subset_inventory_sha256"]:
        raise ValueError("V86 causal subset changed")
    expected = {
        "q_000080": ("Is there a chair in the room?", "yes"),
        "q_000108": ("What color is the bowl?", "red"),
        "q_000014": ("Is the chair left or right of the bowl?", "right"),
    }
    if any((row.question, row.answer) != expected[row.question_id] for row in selected):
        raise ValueError("V86 causal row semantics changed")
    return selected


def load_scene1_memory_v86(
    config: Mapping[str, Any],
) -> tuple[torch.Tensor, str, dict[str, Any]]:
    root = resolve_v85(config["sources"]["scene1_memory"])
    metadata = _strict_json(root / "runtime_metadata.json")
    tensors = load_file(str(root / "memory.safetensors"), device="cpu")
    if set(tensors) != {"fixed_scene_memory"}:
        raise ValueError("V86 memory tensor inventory changed")
    memory = tensors["fixed_scene_memory"].detach().cpu().contiguous()
    observed_hash = prefix_sha256(memory)
    if (
        tuple(memory.shape) != (1, 738, 1536)
        or memory.dtype != torch.bfloat16
        or observed_hash != EXPECTED_PREFIX_SHA256
        or metadata.get("canonical_prefix_sha256") != observed_hash
        or metadata.get("compiled_before_user_question") is not True
        or metadata.get("question_inputs_used_for_compilation") is not False
        or metadata.get("questions_or_answers_serialized") is not False
        or metadata.get("oracle_loaded") is not False
    ):
        raise ValueError("V86 immutable scene-memory contract changed")
    return memory, observed_hash, metadata


def zero_payload_memory_v86(memory: torch.Tensor) -> torch.Tensor:
    if tuple(memory.shape) != (1, 738, 1536):
        raise ValueError("V86 zero-payload control requires [1,738,1536]")
    result = torch.cat(
        (memory[:, :1], torch.zeros_like(memory[:, 1:-1]), memory[:, -1:]), dim=1
    ).contiguous()
    if (
        not torch.equal(result[:, :1], memory[:, :1])
        or not torch.equal(result[:, -1:], memory[:, -1:])
        or torch.count_nonzero(result[:, 1:-1]).item() != 0
        or result[:, 1:-1].numel() != 736 * 1536
    ):
        raise RuntimeError("V86 zero-payload control construction failed")
    return result


class _SyntheticMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.up_proj = nn.Linear(1536, 12288, bias=False, dtype=torch.bfloat16)


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


def lora_preflight_v86(config: Mapping[str, Any]) -> dict[str, Any]:
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
        raise RuntimeError("V86 synthetic LoRA installation failed")
    initialize_lora_adapter_state(installation, seed=int(bridge["initialization_seed"]))
    observed = installation.state_sha256()
    if (
        installation.parameter_count != bridge["trainable_parameter_count"]
        or observed != bridge["expected_initial_state_sha256"]
        or any(torch.count_nonzero(adapter.lora_b).item() for adapter in installation.adapters)
    ):
        raise RuntimeError("V86 deterministic exact-zero LoRA preflight failed")
    return {
        "bank_name": FRESH_BANK_NAME,
        "target_modules": list(installation.target_names),
        "parameter_count": installation.parameter_count,
        "initial_state_sha256": observed,
        "base_projection_weight_shape": [12288, 1536],
        "lora_a_shape": [8, 1536],
        "lora_b_shape": [12288, 8],
        "exact_zero_output_at_initialization": True,
        "full_gemma_model_loaded": False,
    }


def authenticate_sources_v86(config: Mapping[str, Any]) -> dict[str, str]:
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
        str(Path(sources["frozen_checkpoint"]) / "adapter.safetensors"): sources[
            "frozen_adapter_sha256"
        ],
        str(Path(sources["frozen_checkpoint"]) / "runtime_metadata.json"): sources[
            "frozen_runtime_metadata_sha256"
        ],
        sources["v85_equivalence_report"]: sources["v85_equivalence_report_sha256"],
        sources["preflight_source"]: sources["preflight_source_sha256"],
        sources["training_source"]: sources["training_source_sha256"],
        sources["evaluation_source"]: sources["evaluation_source_sha256"],
    }
    observed: dict[str, str] = {}
    for path, expected_sha256 in expected.items():
        value = sha256_file_v85(path)
        if value != expected_sha256:
            raise ValueError(f"V86 pinned source changed: {path}")
        observed[str(path)] = value
    snapshot = (
        Path.home()
        / ".cache/huggingface/hub/models--google--gemma-4-E2B-it/snapshots"
        / str(sources["model_revision"])
    )
    blob = (snapshot / "model.safetensors").resolve(strict=True)
    if blob.name != sources["model_blob_sha256_identity"]:
        raise ValueError("V86 local Gemma blob identity changed")
    model_config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    text = model_config.get("text_config")
    if not isinstance(text, Mapping) or (
        text.get("hidden_size") != 1536
        or text.get("intermediate_size") != 6144
        or text.get("use_double_wide_mlp") is not True
        or text.get("num_hidden_layers") != 35
        or text.get("layer_types", [None] * 35)[34] != "full_attention"
    ):
        raise ValueError("V86 pinned Gemma topology changed")
    frozen_metadata = _strict_json(Path(sources["frozen_checkpoint"]) / "runtime_metadata.json")
    hashes = frozen_metadata.get("lora_bank_state_sha256")
    if (
        not isinstance(hashes, Mapping)
        or len(hashes) != 7
        or hashes.get("v85_strict_multiscene_bridge")
        != config["frozen_stack"]["v85_bank_state_sha256"]
    ):
        raise ValueError("V86 frozen seven-bank source changed")
    observed["gemma_model_blob_sha256_identity"] = blob.name
    return observed


def _protocol_preflight(config: Mapping[str, Any]) -> dict[str, Any]:
    rows = load_scene1_rows_v86(config)
    schedule = training_schedule_v86(
        rows,
        seed=int(config["training"]["row_order_seed"]),
        epochs=int(config["training"]["epochs"]),
    )
    schedule_hash = canonical_sha256_v85([[epoch, row.question_id] for epoch, row in schedule])
    if (
        len(schedule) != 552
        or schedule_hash != config["training"]["row_order_sha256"]
        or any(
            sum(row.question_id == question_id for _epoch, row in schedule) != 4
            for question_id in {row.question_id for row in rows}
        )
    ):
        raise ValueError("V86 deterministic four-epoch schedule changed")
    causal = causal_rows_v86(config, rows)
    memory, memory_hash, metadata = load_scene1_memory_v86(config)
    zero = zero_payload_memory_v86(memory)
    zero_hash = prefix_sha256(zero)
    if zero_hash == memory_hash:
        raise RuntimeError("V86 zero-payload control did not change the payload")
    return {
        "row_count": len(rows),
        "row_inventory_sha256": config["dataset"]["row_inventory_sha256"],
        "schedule_rows": len(schedule),
        "row_order_sha256": schedule_hash,
        "first_schedule_keys": [[epoch, row.question_id] for epoch, row in schedule[:3]],
        "last_schedule_keys": [[epoch, row.question_id] for epoch, row in schedule[-3:]],
        "causal_question_ids": [row.question_id for row in causal],
        "causal_subset_inventory_sha256": config["training"]["causal_subset_inventory_sha256"],
        "fixed_memory_shape": list(memory.shape),
        "fixed_memory_dtype": str(memory.dtype),
        "fixed_memory_prefix_sha256": memory_hash,
        "memory_compiled_before_question": metadata["compiled_before_user_question"],
        "zero_payload_prefix_sha256": zero_hash,
        "zero_payload_preserves_native_boi": bool(torch.equal(zero[:, :1], memory[:, :1])),
        "zero_payload_preserves_native_eoi": bool(torch.equal(zero[:, -1:], memory[:, -1:])),
        "zero_payload_token_count": 736,
        "zero_payload_scalar_count": int(zero[:, 1:-1].numel()),
        "zero_payload_nonzero_scalar_count": int(torch.count_nonzero(zero[:, 1:-1]).item()),
        "questions_tokenized": False,
        "full_gemma_model_loaded": False,
    }


def build_preregistration_v86(config_path: str | Path = CONFIG) -> dict[str, Any]:
    config = load_config_v86(config_path)
    sources = authenticate_sources_v86(config)
    protocol = _protocol_preflight(config)
    lora = lora_preflight_v86(config)
    payload = {
        "artifact": PREREG_ARTIFACT,
        "schema_version": 86,
        "status": "sealed_before_first_v86_full_model_load",
        "config_path": Path(config_path).as_posix(),
        "config_sha256": sha256_file_v85(config_path),
        "authenticated_sources": sources,
        "strict_input_contract": config["strict_input_contract"],
        "dataset_contract": config["dataset"],
        "frozen_stack": config["frozen_stack"],
        "bridge": config["bridge"],
        "training_protocol": config["training"],
        "fixed_gates": config["gates"],
        "protocol_preflight": protocol,
        "lora_cpu_preflight": lora,
        "v85_held_evidence_mutated": False,
        "answers_available_to_training_only": True,
        "answers_or_questions_serialized_in_runtime_candidate": False,
        "full_gemma_model_loaded": False,
        "optimizer_constructed": False,
        "optimizer_updates": 0,
        "behavior_scored": False,
        "official_validation_loaded": False,
        "official_test_loaded": False,
        "deferred_final_loaded": False,
        "oracle_loaded": False,
        "runtime_promotion_authorized": False,
    }
    output, _sha = atomic_create_json_v85(config["outputs"]["preregistration"], payload)
    payload["output"] = output.relative_to(PROJECT_ROOT).as_posix()
    return payload


def authenticate_preregistration_v86(
    config: Mapping[str, Any], *, config_path: str | Path = CONFIG
) -> dict[str, str]:
    path = resolve_v85(config["outputs"]["preregistration"])
    payload = _strict_json(path)
    config_sha256 = sha256_file_v85(config_path)
    if (
        payload.get("artifact") != PREREG_ARTIFACT
        or payload.get("status") != "sealed_before_first_v86_full_model_load"
        or payload.get("config_sha256") != config_sha256
        or payload.get("full_gemma_model_loaded") is not False
        or payload.get("optimizer_constructed") is not False
        or payload.get("optimizer_updates") != 0
        or payload.get("behavior_scored") is not False
        or payload.get("oracle_loaded") is not False
        or payload.get("runtime_promotion_authorized") is not False
    ):
        raise ValueError("V86 preregistration changed")
    return {
        "config_sha256": config_sha256,
        "preregistration_sha256": sha256_file_v85(path),
    }


def run_cpu_preflight_v86(config_path: str | Path = CONFIG) -> dict[str, Any]:
    config = load_config_v86(config_path)
    prereg = authenticate_preregistration_v86(config, config_path=config_path)
    report = {
        "artifact": PREFLIGHT_ARTIFACT,
        "schema_version": 86,
        "status": "passed",
        "passed": True,
        **prereg,
        "authenticated_sources": authenticate_sources_v86(config),
        "protocol_preflight": _protocol_preflight(config),
        "lora_preflight": lora_preflight_v86(config),
        "fixed_final_optimizer_updates": 92,
        "fixed_final_checkpoint_selection": "fixed_final_update_92",
        "all_138_rows_used_each_epoch": True,
        "same_fixed_memory_compiled_before_questions": True,
        "all_738_memory_slots_retained": True,
        "question_derived_environmental_tokens": 0,
        "question_conditioned_environmental_readout": False,
        "full_gemma_model_loaded": False,
        "optimizer_constructed": False,
        "optimizer_updates": 0,
        "behavior_scored": False,
        "protected_or_sealed_behavior_artifacts_opened": [],
        "official_validation_loaded": False,
        "official_test_loaded": False,
        "deferred_final_loaded": False,
        "oracle_loaded": False,
        "runtime_promotion_authorized": False,
    }
    output, _sha = atomic_create_json_v85(config["outputs"]["cpu_preflight"], report)
    report["output"] = output.relative_to(PROJECT_ROOT).as_posix()
    return report


def authenticate_cpu_preflight_v86(
    config: Mapping[str, Any], *, config_path: str | Path = CONFIG
) -> dict[str, str]:
    prereg = authenticate_preregistration_v86(config, config_path=config_path)
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
        or payload.get("behavior_scored") is not False
        or payload.get("protected_or_sealed_behavior_artifacts_opened") != []
        or payload.get("oracle_loaded") is not False
        or payload.get("runtime_promotion_authorized") is not False
    ):
        raise ValueError("V86 CPU preflight changed")
    return {**prereg, "cpu_preflight_sha256": sha256_file_v85(path)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preregister", "preflight"))
    parser.add_argument("--config", default=str(CONFIG))
    args = parser.parse_args(argv)
    result = (
        build_preregistration_v86(args.config)
        if args.command == "preregister"
        else run_cpu_preflight_v86(args.config)
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CAUSAL_IDS",
    "CONFIG",
    "EXPECTED_PREFIX_SHA256",
    "FRESH_BANK_NAME",
    "PREFLIGHT_ARTIFACT",
    "PREREG_ARTIFACT",
    "SCENE_ID",
    "TARGET_MODULE",
    "authenticate_cpu_preflight_v86",
    "authenticate_preregistration_v86",
    "authenticate_sources_v86",
    "build_preregistration_v86",
    "causal_rows_v86",
    "load_config_v86",
    "load_scene1_memory_v86",
    "load_scene1_rows_v86",
    "lora_preflight_v86",
    "main",
    "run_cpu_preflight_v86",
    "training_schedule_v86",
    "zero_payload_memory_v86",
]
