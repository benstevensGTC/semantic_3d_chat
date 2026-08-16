"""Train V91's fixed-memory single-scene conversational repair bridge.

The exact promoted V89 eleven-bank checkpoint and the exact failed-but-
authenticated V90 two-tensor bridge form twelve immutable parent banks.  V91
optimizes only a fresh rank-16 LoRA bank on layer 33's MLP down projection.
Every training row receives the same pre-question 738-token continuous scene
memory.  Only the thirteen primary rows per epoch receive the preregistered
zero-payload causal margin.  The create-once candidate contains only the fresh
two-tensor V91 bridge and sanitized numeric/hash metadata.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import tempfile
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from semantic_3d_chat.chat.file_audit import FileAccessAudit
from semantic_3d_chat.chat.runtime_config import load_runtime_config
from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.evaluation.prediction_artifacts import checkpoint_fingerprint
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
from semantic_3d_chat.evaluation.v89_scene1_retention_preflight import strict_json_v89
from semantic_3d_chat.evaluation.v91_scene1_conversational_preflight import (
    CONFIG,
    FRESH_BANK_NAME,
    TARGET_MODULE,
    authenticate_cpu_preflight_v91,
    authenticate_sources_v91,
    derive_training_items_v91,
    inventory_v91,
    load_canonical_rows_v91,
    load_config_v91,
    schedule_v91,
)
from semantic_3d_chat.language.local_lm import load_local_language_model
from semantic_3d_chat.language.lora import (
    LoRABankCollection,
    LoRABankSettings,
    LoRABanksSettings,
    LoRASettings,
    install_lora_banks,
    lora_banks_settings,
    tensor_state_sha256,
)
from semantic_3d_chat.language.prefix_injection import prefix_sha256
from semantic_3d_chat.training.train_v84_strict_bridge import _prepared_v84
from semantic_3d_chat.training.train_v86_scene1_demo import (
    zero_payload_margin_objective_v86,
)

TRAINING_ARTIFACT: Final[str] = (
    "gemma4_v91_scene1_conversational_repair_training_v1"
)
CANDIDATE_ARTIFACT: Final[str] = (
    "gemma4_v91_scene1_conversational_repair_fixed_final_v1"
)
WEIGHTS_FILENAME: Final[str] = "bridge.safetensors"
METADATA_FILENAME: Final[str] = "runtime_metadata.json"
PARENT_BANK_COUNT: Final[int] = 12
TOTAL_BANK_COUNT: Final[int] = 13
V89_PARAMETER_COUNT: Final[int] = 872_448
V90_PARAMETER_COUNT: Final[int] = 28_672
PARENT_PARAMETER_COUNT: Final[int] = 901_120
FRESH_PARAMETER_COUNT: Final[int] = 221_184
TOTAL_PARAMETER_COUNT: Final[int] = 1_122_304
EXPECTED_ROWS_PER_EPOCH: Final[int] = 590
EXPECTED_MICRO_ROWS: Final[int] = 1_770
EXPECTED_OPTIMIZER_UPDATES: Final[int] = 295
EXPECTED_CAUSAL_ROWS: Final[int] = 39
EXPECTED_FRESH_TARGET: Final[str] = (
    "model.language_model.layers.33.mlp.down_proj"
)
EXPECTED_FRESH_INITIAL_STATE_SHA256: Final[str] = (
    "0f255efb26255dcac0815511e44aabad5e21820f78f9a7662dc1bf59f627db2b"
)
V90_BANK_NAME: Final[str] = "v90_scene1_conversational_bridge"
V90_TARGET: Final[str] = "model.language_model.layers.28.self_attn.o_proj"
V90_ARTIFACT: Final[str] = "gemma4_v90_scene1_conversational_fixed_final_v1"
V90_WEIGHTS_SHA256: Final[str] = (
    "be8a6fa9b633dc52ca962393170623c2707e339206ced627e6439ce7db0a7f94"
)
V90_METADATA_SHA256: Final[str] = (
    "c2dba75094c829c7fadaadf8111a712921f70372ac84e2073714530519349acc"
)
V90_STATE_SHA256: Final[str] = (
    "70e236711d8ac1fe7cf808f6f4e939b29db476016c8ef49db143707df0f3bde7"
)
V90_TRAINING_REPORT_SHA256: Final[str] = (
    "021d87d56fa6d898e255222661f7c381c500478a403b11c6f5bb3ff6228f791c"
)
V90_PREDICTIONS_SHA256: Final[str] = (
    "04dc5b74a1dd5bae9643b7cf0695724adf32c5ed070a6e73cc1623d47fdeef14"
)
V90_EVALUATION_SHA256: Final[str] = (
    "e2102928af6590db9d206ab5d53c13492b4c73b2736ca740dbfbeb5cc85850e7"
)
EXPECTED_V89_BANKS: Final[tuple[str, ...]] = (
    "inherited_v12",
    "extension_v13",
    "extension_v23_shared_kv",
    "extension_v24_shared_query",
    "extension_v28_stage_b_query",
    "extension_v30_joint_pair_query",
    "v85_strict_multiscene_bridge",
    "v86_scene1_demo_bridge",
    "v87_scene1_balanced_bridge",
    "v88_scene1_augmented_bridge",
    "v89_scene1_retention_bridge",
)
EXPECTED_PARENT_BANKS: Final[tuple[str, ...]] = (
    *EXPECTED_V89_BANKS,
    V90_BANK_NAME,
)
_FORBIDDEN_COMPONENTS: Final[frozenset[str]] = frozenset(
    {"oracle", "validation", "test", "deferred"}
)


def _source_value(
    sources: Mapping[str, Any], *keys: str, required: bool = True
) -> Any:
    for key in keys:
        if key in sources:
            return sources[key]
    if required:
        raise KeyError(f"V91 config omitted source aliases: {keys}")
    return None


def _source_path(sources: Mapping[str, Any], *keys: str) -> Path:
    return resolve_v85(str(_source_value(sources, *keys)))


def _v90_paths(sources: Mapping[str, Any]) -> dict[str, Path]:
    return {
        "candidate": _source_path(
            sources,
            "parent_v90_candidate",
            "v90_candidate",
            "failed_v90_candidate",
            "v90_fixed_final_candidate",
        ),
        "training": _source_path(
            sources,
            "parent_v90_training",
            "v90_training_report",
            "parent_v90_training_report",
            "failed_v90_training_report",
        ),
        "predictions": _source_path(
            sources,
            "parent_v90_predictions",
            "v90_predictions",
            "parent_v90_predictions",
            "failed_v90_predictions",
        ),
        "evaluation": _source_path(
            sources,
            "parent_v90_evaluation",
            "v90_evaluation",
            "parent_v90_evaluation",
            "failed_v90_evaluation",
        ),
    }


def _copy_bank_state_v91(
    installation: Any, archive: Mapping[str, torch.Tensor], *, context: str
) -> None:
    expected_keys = set(installation.state_module.state_dict())
    if set(archive) != expected_keys:
        raise ValueError(f"{context} tensor inventory changed")
    if any(value.dtype != torch.float32 for value in archive.values()):
        raise TypeError(f"{context} tensors must be float32")
    installation.state_module.load_state_dict(dict(archive), strict=True)
    installation.validate_state()


def _fresh_state_v91(collection: LoRABankCollection) -> dict[str, torch.Tensor]:
    fresh = collection.bank(FRESH_BANK_NAME).installation
    if len(fresh.adapters) != 1:
        raise ValueError("V91 fresh bank must wrap exactly one module")
    adapter = fresh.adapters[0]
    state = {
        "lora_a": adapter.lora_a.detach().cpu().contiguous(),
        "lora_b": adapter.lora_b.detach().cpu().contiguous(),
    }
    if (
        tuple(state["lora_a"].shape) != (16, 12_288)
        or tuple(state["lora_b"].shape) != (1_536, 16)
        or any(value.dtype != torch.float32 for value in state.values())
        or sum(value.numel() for value in state.values()) != FRESH_PARAMETER_COUNT
    ):
        raise ValueError("V91 fresh bridge tensor topology changed")
    return state


def _load_fresh_state_v91(
    collection: LoRABankCollection, archive: Mapping[str, torch.Tensor]
) -> None:
    fresh = collection.bank(FRESH_BANK_NAME).installation
    if set(archive) != {"lora_a", "lora_b"} or len(fresh.adapters) != 1:
        raise ValueError("V91 fresh bridge tensor inventory changed")
    adapter = fresh.adapters[0]
    with torch.no_grad():
        for name, parameter in (
            ("lora_a", adapter.lora_a),
            ("lora_b", adapter.lora_b),
        ):
            value = archive[name]
            if value.dtype != torch.float32 or value.shape != parameter.shape:
                raise ValueError(f"V91 fresh bridge {name} shape or dtype changed")
            parameter.copy_(value.to(parameter.device))
    collection.validate_state()


def combined_lora_settings_v91(
    runtime_config: Mapping[str, Any], experiment: Mapping[str, Any]
) -> LoRABanksSettings:
    """Return exact frozen V89+V90 banks plus V91's sole trainable bank."""

    v89 = lora_banks_settings(runtime_config)
    if (
        v89.legacy_single_bank
        or tuple(bank.name for bank in v89.banks) != EXPECTED_V89_BANKS
        or any(bank.trainable for bank in v89.banks)
    ):
        raise ValueError("V91 requires the exact promoted frozen V89 bank stack")
    sources = experiment["sources"]
    v90_state = str(
        _source_value(
            sources,
            "v90_state_sha256",
            "v90_bridge_state_sha256",
            "parent_v90_state_sha256",
            required=False,
        )
        or V90_STATE_SHA256
    )
    if v90_state != V90_STATE_SHA256:
        raise ValueError("V91 sealed V90 parent state binding changed")
    v90 = LoRABankSettings(
        name=V90_BANK_NAME,
        trainable=False,
        adapter=LoRASettings(
            enabled=True,
            rank=8,
            alpha=16.0,
            dropout=0.0,
            target_modules=(V90_TARGET,),
        ),
        initialization_algorithm="checkpoint_overwrite",
        initialization_seed=None,
        expected_initial_state_sha256=v90_state,
    )
    bridge = experiment["bridge"]
    if (
        str(bridge["bank_name"]) != FRESH_BANK_NAME
        or str(bridge["target_module"]) != TARGET_MODULE
        or TARGET_MODULE != EXPECTED_FRESH_TARGET
        or int(bridge["rank"]) != 16
        or float(bridge["alpha"]) != 32.0
        or float(bridge["dropout"]) != 0.0
        or int(bridge["trainable_parameter_count"]) != FRESH_PARAMETER_COUNT
        or str(bridge["expected_initial_state_sha256"])
        != EXPECTED_FRESH_INITIAL_STATE_SHA256
    ):
        raise ValueError("V91 fresh repair bridge topology changed")
    fresh = LoRABankSettings(
        name=FRESH_BANK_NAME,
        trainable=True,
        adapter=LoRASettings(
            enabled=True,
            rank=16,
            alpha=32.0,
            dropout=0.0,
            target_modules=(EXPECTED_FRESH_TARGET,),
        ),
        initialization_algorithm=str(bridge["initialization_algorithm"]),
        initialization_seed=int(bridge["initialization_seed"]),
        expected_initial_state_sha256=str(bridge["expected_initial_state_sha256"]),
    )
    settings = LoRABanksSettings(v89.banks + (v90, fresh))
    if (
        tuple(bank.name for bank in settings.banks)
        != (*EXPECTED_PARENT_BANKS, FRESH_BANK_NAME)
        or sum(bank.trainable for bank in settings.banks) != 1
        or settings.bank(FRESH_BANK_NAME).trainable is not True
    ):
        raise RuntimeError("V91 exact thirteen-bank optimizer surface changed")
    return settings


def _authenticate_failed_v90(config: Mapping[str, Any]) -> dict[str, Any]:
    """Authenticate the fixed V90 failure and its safe two-tensor bridge."""

    sources = config["sources"]
    paths = _v90_paths(sources)
    pinned = {
        paths["candidate"] / WEIGHTS_FILENAME: V90_WEIGHTS_SHA256,
        paths["candidate"] / METADATA_FILENAME: V90_METADATA_SHA256,
        paths["training"]: V90_TRAINING_REPORT_SHA256,
        paths["predictions"]: V90_PREDICTIONS_SHA256,
        paths["evaluation"]: V90_EVALUATION_SHA256,
    }
    mismatches = {
        str(path): {
            "expected": expected,
            "observed": sha256_file_v85(path) if path.is_file() else None,
        }
        for path, expected in pinned.items()
        if path.is_symlink() or not path.is_file() or sha256_file_v85(path) != expected
    }
    if mismatches:
        raise ValueError(f"V91 fixed V90 parent evidence changed: {mismatches}")
    metadata = strict_json_v89(paths["candidate"] / METADATA_FILENAME)
    training = strict_json_v89(paths["training"])
    predictions = strict_json_v89(paths["predictions"])
    evaluation = strict_json_v89(paths["evaluation"])
    metrics = evaluation.get("metrics")
    model_gates = (
        metrics.get("model_acceptance_gates")
        if isinstance(metrics, Mapping)
        else None
    )
    primary = metrics.get("primary_conversational") if isinstance(metrics, Mapping) else None
    held = metrics.get("held_wording") if isinstance(metrics, Mapping) else None
    causal = metrics.get("causal_control") if isinstance(metrics, Mapping) else None
    failed_gate_names = (
        {name for name, passed in model_gates.items() if passed is False}
        if isinstance(model_gates, Mapping)
        else None
    )
    expected_failed = {
        "primary_conversational_correct_at_least_required",
        "all_six_core_actionable_intents_correct",
        "held_wording_correct_at_least_required",
        "held_wording_each_intent_at_least_minimum",
    }
    if (
        metadata.get("artifact") != V90_ARTIFACT
        or metadata.get("schema_version") != 90
        or metadata.get("status")
        != "fixed_final_awaiting_preregistered_acceptance_gates"
        or metadata.get("bank_name") != V90_BANK_NAME
        or metadata.get("target_module") != V90_TARGET
        or metadata.get("rank") != 8
        or float(metadata.get("alpha", -1.0)) != 16.0
        or float(metadata.get("dropout", -1.0)) != 0.0
        or metadata.get("parameter_count") != V90_PARAMETER_COUNT
        or metadata.get("state_sha256") != V90_STATE_SHA256
        or metadata.get("weights_sha256") != V90_WEIGHTS_SHA256
        or metadata.get("environmental_memory_serialized") is not False
        or metadata.get("environmental_text_serialized") is not False
        or metadata.get("environmental_text_inputs") != []
        or metadata.get("questions_or_answers_serialized") is not False
        or metadata.get("training_metadata_serialized") is not False
        or metadata.get("training_inventory_serialized") is not False
        or metadata.get("oracle_serialized") is not False
        or metadata.get("evaluation_scored") is not False
        or metadata.get("runtime_promotion_authorized") is not False
        or training.get("artifact")
        != "gemma4_v90_scene1_conversational_training_v1"
        or training.get("status") != "fixed_final_training_complete_not_promoted"
        or training.get("optimizer_updates") != 172
        or training.get("micro_rows_consumed") != 1_032
        or training.get("protected_read_count") != 0
        or training.get("oracle_loaded") is not False
        or evaluation.get("artifact")
        != "gemma4_v90_scene1_conversational_evaluation_v1"
        or evaluation.get("status") != "model_gates_fail_not_runtime_promotable"
        or not isinstance(metrics, Mapping)
        or metrics.get("model_acceptance_gate_passed") is not False
        or metrics.get("separate_runtime_packaging_authorized") is not False
        or metrics.get("runtime_promotion_authorized") is not False
        or failed_gate_names != expected_failed
        or not isinstance(primary, Mapping)
        or primary.get("correct") != 7
        or primary.get("total") != 13
        or primary.get("core_actionable_correct") != 1
        or primary.get("core_actionable_total") != 6
        or not isinstance(held, Mapping)
        or held.get("correct") != 10
        or held.get("total") != 26
        or not isinstance(causal, Mapping)
        or causal.get("row_count") != 13
        or float(causal.get("mean_zero_minus_correct_nll", 0.0)) < 0.5
        or int(causal.get("canonical_prediction_changes", 0)) < 6
        or evaluation.get("oracle_loaded") is not False
        or evaluation.get("held_out_scene") is not False
        or evaluation.get("held_out_generalization_claim") is not False
        or evaluation.get("runtime_promotion_authorized") is not False
        or predictions.get("candidate", {}).get("weights_sha256")
        != V90_WEIGHTS_SHA256
        or predictions.get("candidate", {}).get("state_sha256")
        != V90_STATE_SHA256
        or predictions.get("frozen_parent_state_invariant") is not True
        or predictions.get("runtime_promotion_authorized") is not False
    ):
        raise ValueError("V91 failed V90 parent authentication changed")
    with safe_open(
        str(paths["candidate"] / WEIGHTS_FILENAME), framework="pt", device="cpu"
    ) as handle:
        tensor_metadata = handle.metadata()
        raw = {
            name: handle.get_tensor(name)
            for name in handle.keys()  # noqa: SIM118 - safe_open is not iterable.
        }
    if (
        set(raw) != {"lora_a", "lora_b"}
        or tuple(raw["lora_a"].shape) != (8, 2_048)
        or tuple(raw["lora_b"].shape) != (1_536, 8)
        or any(value.dtype != torch.float32 for value in raw.values())
        or not isinstance(tensor_metadata, dict)
        or tensor_metadata.get("environmental_memory_serialized") != "false"
        or tensor_metadata.get("questions_or_answers_serialized") != "false"
        or tensor_metadata.get("oracle_serialized") != "false"
    ):
        raise ValueError("V91 V90 parent tensor artifact is unsafe")
    state = {
        "adapters.0.lora_a": raw["lora_a"],
        "adapters.0.lora_b": raw["lora_b"],
    }
    if tensor_state_sha256(state) != V90_STATE_SHA256:
        raise ValueError("V91 V90 parent tensor state changed")
    return {
        "candidate_path": str(paths["candidate"]),
        "weights_sha256": V90_WEIGHTS_SHA256,
        "metadata_sha256": V90_METADATA_SHA256,
        "state_sha256": V90_STATE_SHA256,
        "training_report_sha256": V90_TRAINING_REPORT_SHA256,
        "predictions_sha256": V90_PREDICTIONS_SHA256,
        "evaluation_sha256": V90_EVALUATION_SHA256,
        "failed_model_gate_names": sorted(expected_failed),
        "primary_correct": 7,
        "held_wording_correct": 10,
        "failed_but_authenticated": True,
        "runtime_promotion_authorized": False,
        "held_out_generalization_claim": False,
    }


def load_frozen_parent_v91(
    collection: LoRABankCollection,
    parent_checkpoint: str | Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Load V89's promoted archive plus the exact failed V90 bridge frozen."""

    sources = config["sources"]
    v89_root = resolve_v85(parent_checkpoint)
    configured_parent = _source_path(sources, "parent_v89_checkpoint")
    if v89_root != configured_parent:
        raise ValueError("V91 V89 parent checkpoint path changed")
    v89_weights = v89_root / "adapter.safetensors"
    v89_metadata_path = v89_root / METADATA_FILENAME
    if (
        sha256_file_v85(v89_weights) != sources["parent_v89_adapter_sha256"]
        or sha256_file_v85(v89_metadata_path)
        != sources["parent_v89_metadata_sha256"]
    ):
        raise ValueError("V91 promoted V89 parent bytes changed")
    v89_metadata = strict_json_v89(v89_metadata_path)
    v89_lora = v89_metadata.get("lora")
    v89_banks = v89_lora.get("banks") if isinstance(v89_lora, Mapping) else None
    v89_states = v89_metadata.get("lora_bank_state_sha256")
    v89_modules = v89_metadata.get("lora_bank_wrapped_modules")
    v89_counts = v89_metadata.get("lora_bank_parameter_counts")
    provenance_root = v89_metadata.get("initialization_provenance")
    v89_release = (
        provenance_root.get("v89_strict_runtime_release")
        if isinstance(provenance_root, Mapping)
        else None
    )
    frozen = tuple(bank for bank in collection.banks if not bank.settings.trainable)
    trainable = tuple(bank for bank in collection.banks if bank.settings.trainable)
    if (
        not isinstance(v89_banks, list)
        or tuple(str(row.get("name")) for row in v89_banks) != EXPECTED_V89_BANKS
        or not isinstance(v89_states, Mapping)
        or set(v89_states) != set(EXPECTED_V89_BANKS)
        or not isinstance(v89_modules, Mapping)
        or set(v89_modules) != set(EXPECTED_V89_BANKS)
        or not isinstance(v89_counts, Mapping)
        or set(v89_counts) != set(EXPECTED_V89_BANKS)
        or v89_lora.get("adapter_parameter_count") != V89_PARAMETER_COUNT
        or v89_lora.get("trainable_adapter_parameter_count") != 0
        or v89_metadata.get("lora_parameter_count") != V89_PARAMETER_COUNT
        or v89_metadata.get("lora_trainable_parameter_count") != 0
        or not isinstance(v89_release, Mapping)
        or v89_release.get("schema_version") != 89
        or v89_release.get("runtime_promotion_authorized") is not True
        or v89_release.get("model_acceptance_gate_passed") is not True
        or v89_release.get("model_gate_report_authenticated") is not True
        or v89_release.get("held_out_generalization_claim") is not False
        or len(frozen) != PARENT_BANK_COUNT
        or tuple(bank.settings.name for bank in frozen) != EXPECTED_PARENT_BANKS
        or len(trainable) != 1
        or trainable[0].settings.name != FRESH_BANK_NAME
    ):
        raise ValueError("V91 exact frozen V89+V90 bank inventory changed")

    v89_archive = load_file(str(v89_weights), device="cpu")
    observed_states: dict[str, str] = {}
    for bank in frozen[: len(EXPECTED_V89_BANKS)]:
        name = bank.settings.name
        prefix = f"lora_banks.{name}."
        state = {
            key[len(prefix) :]: value
            for key, value in v89_archive.items()
            if key.startswith(prefix)
        }
        if (
            bank.settings.trainable
            or list(bank.installation.target_names) != v89_modules[name]
            or sum(int(value) for value in v89_counts[name].values())
            != bank.installation.parameter_count
            or tensor_state_sha256(state) != v89_states[name]
        ):
            raise ValueError(f"V91 frozen V89 bank changed: {name}")
        _copy_bank_state_v91(
            bank.installation,
            state,
            context=f"V91 frozen V89 bank {name}",
        )
        observed_states[name] = bank.installation.state_sha256()
        if observed_states[name] != v89_states[name]:
            raise ValueError(f"V91 frozen V89 state changed after load: {name}")

    v90_evidence = _authenticate_failed_v90(config)
    v90_bank = collection.bank(V90_BANK_NAME)
    v90_root = Path(v90_evidence["candidate_path"])
    raw_v90 = load_file(str(v90_root / WEIGHTS_FILENAME), device="cpu")
    v90_state = {
        "adapters.0.lora_a": raw_v90["lora_a"],
        "adapters.0.lora_b": raw_v90["lora_b"],
    }
    if (
        v90_bank.settings.trainable
        or v90_bank.installation.target_names != (V90_TARGET,)
        or v90_bank.installation.parameter_count != V90_PARAMETER_COUNT
        or tensor_state_sha256(v90_state) != V90_STATE_SHA256
    ):
        raise ValueError("V91 frozen V90 bank topology changed")
    _copy_bank_state_v91(
        v90_bank.installation,
        v90_state,
        context="V91 frozen V90 bank",
    )
    if v90_bank.installation.state_sha256() != V90_STATE_SHA256:
        raise ValueError("V91 frozen V90 state changed after load")
    observed_states[V90_BANK_NAME] = V90_STATE_SHA256

    fresh = trainable[0].installation
    bridge = config["bridge"]
    if (
        fresh.target_names != (EXPECTED_FRESH_TARGET,)
        or fresh.parameter_count != FRESH_PARAMETER_COUNT
        or fresh.state_sha256() != bridge["expected_initial_state_sha256"]
        or fresh.state_sha256() != EXPECTED_FRESH_INITIAL_STATE_SHA256
        or any(
            int(torch.count_nonzero(adapter.lora_b).item()) != 0
            for adapter in fresh.adapters
        )
    ):
        raise ValueError("V91 fresh repair bridge did not start at exact zero output")
    collection.validate_state()
    fingerprint, files = checkpoint_fingerprint(v89_root)
    return {
        "v89_checkpoint_sha256": fingerprint,
        "v89_checkpoint_files": files,
        "v89_adapter_sha256": sha256_file_v85(v89_weights),
        "v89_runtime_metadata_sha256": sha256_file_v85(v89_metadata_path),
        "v89_bank_count": len(EXPECTED_V89_BANKS),
        "v89_parameter_count": V89_PARAMETER_COUNT,
        "v89_release_provenance_sha256": canonical_sha256_v85(v89_release),
        "v90": v90_evidence,
        "frozen_bank_state_sha256": observed_states,
        "frozen_bank_count": PARENT_BANK_COUNT,
        "frozen_parameter_count": PARENT_PARAMETER_COUNT,
        "fresh_initial_state_sha256": fresh.state_sha256(),
        "parent_tensors_loaded_byte_exactly": True,
        "failed_v90_parent_loaded_unmerged": True,
    }


def _candidate_fingerprint_v91(root: Path) -> tuple[str, list[dict[str, Any]]]:
    files = [root / WEIGHTS_FILENAME, root / METADATA_FILENAME]
    if root.is_symlink() or not root.is_dir() or any(
        path.is_symlink() or not path.is_file() for path in files
    ):
        raise FileNotFoundError("V91 candidate is not an exact physical artifact")
    if {item.name for item in root.iterdir()} != {
        WEIGHTS_FILENAME,
        METADATA_FILENAME,
    }:
        raise ValueError("V91 candidate contains unexpected files")
    entries = [
        {
            "path": path.name,
            "sha256": sha256_file_v85(path),
            "size_bytes": path.stat().st_size,
        }
        for path in files
    ]
    return canonical_sha256_v85(entries), entries


def publish_fixed_final_candidate_v91(
    destination: str | Path,
    collection: LoRABankCollection,
    *,
    bindings: Mapping[str, str | int | bool],
    experiment: Mapping[str, Any],
) -> dict[str, Any]:
    """Atomically publish only V91's fresh two-tensor repair bridge."""

    root = resolve_v85(destination)
    if root.exists() or root.is_symlink():
        raise FileExistsError(f"V91 fixed-final candidate exists: {root}")
    root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{root.name}.", dir=root.parent))
    try:
        weights = temporary / WEIGHTS_FILENAME
        save_file(
            _fresh_state_v91(collection),
            str(weights),
            metadata={
                "artifact": CANDIDATE_ARTIFACT,
                "environmental_memory_serialized": "false",
                "environmental_text_serialized": "false",
                "questions_or_answers_serialized": "false",
                "training_metadata_serialized": "false",
                "oracle_serialized": "false",
            },
        )
        fresh = collection.bank(FRESH_BANK_NAME).installation
        bridge = experiment["bridge"]
        metadata = {
            "artifact": CANDIDATE_ARTIFACT,
            "schema_version": 91,
            "status": "fixed_final_awaiting_preregistered_acceptance_gates",
            "bank_name": FRESH_BANK_NAME,
            "target_module": EXPECTED_FRESH_TARGET,
            "rank": 16,
            "alpha": 32.0,
            "dropout": 0.0,
            "parameter_count": fresh.parameter_count,
            "initialization_algorithm": bridge["initialization_algorithm"],
            "initialization_seed": int(bridge["initialization_seed"]),
            "state_sha256": fresh.state_sha256(),
            "weights_sha256": sha256_file_v85(weights),
            "frozen_parent_bank_count": PARENT_BANK_COUNT,
            "total_bank_count": TOTAL_BANK_COUNT,
            "frozen_parent_parameter_count": PARENT_PARAMETER_COUNT,
            "total_adapter_parameter_count": TOTAL_PARAMETER_COUNT,
            "v90_parent_state_sha256": V90_STATE_SHA256,
            "v90_parent_runtime_promotable": False,
            "environmental_memory_serialized": False,
            "environmental_text_serialized": False,
            "environmental_text_inputs": [],
            "questions_or_answers_serialized": False,
            "training_metadata_serialized": False,
            "training_inventory_serialized": False,
            "oracle_serialized": False,
            "evaluation_scored": False,
            "runtime_promotion_authorized": False,
            "bindings": dict(bindings),
        }
        (temporary / METADATA_FILENAME).write_text(
            json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.rename(temporary, root)
        return metadata
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _authenticate_fixed_final_candidate_v91(
    candidate: str | Path,
) -> tuple[dict[str, Any], str, list[dict[str, Any]]]:
    """Authenticate candidate bytes and tensors without loading Gemma."""

    root = resolve_v85(candidate)
    fingerprint, files = _candidate_fingerprint_v91(root)
    metadata = strict_json_v89(root / METADATA_FILENAME)
    weights = root / WEIGHTS_FILENAME
    expected_metadata_keys = {
        "artifact",
        "schema_version",
        "status",
        "bank_name",
        "target_module",
        "rank",
        "alpha",
        "dropout",
        "parameter_count",
        "initialization_algorithm",
        "initialization_seed",
        "state_sha256",
        "weights_sha256",
        "frozen_parent_bank_count",
        "total_bank_count",
        "frozen_parent_parameter_count",
        "total_adapter_parameter_count",
        "v90_parent_state_sha256",
        "v90_parent_runtime_promotable",
        "environmental_memory_serialized",
        "environmental_text_serialized",
        "environmental_text_inputs",
        "questions_or_answers_serialized",
        "training_metadata_serialized",
        "training_inventory_serialized",
        "oracle_serialized",
        "evaluation_scored",
        "runtime_promotion_authorized",
        "bindings",
    }
    bindings = metadata.get("bindings")
    if (
        set(metadata) != expected_metadata_keys
        or metadata.get("artifact") != CANDIDATE_ARTIFACT
        or metadata.get("schema_version") != 91
        or metadata.get("status")
        != "fixed_final_awaiting_preregistered_acceptance_gates"
        or metadata.get("bank_name") != FRESH_BANK_NAME
        or metadata.get("target_module") != EXPECTED_FRESH_TARGET
        or metadata.get("rank") != 16
        or float(metadata.get("alpha", -1.0)) != 32.0
        or float(metadata.get("dropout", -1.0)) != 0.0
        or metadata.get("parameter_count") != FRESH_PARAMETER_COUNT
        or metadata.get("initialization_algorithm")
        != "cpu_kaiming_uniform_a_exact_zero_b"
        or metadata.get("initialization_seed") != 910091
        or metadata.get("weights_sha256") != sha256_file_v85(weights)
        or metadata.get("frozen_parent_bank_count") != PARENT_BANK_COUNT
        or metadata.get("total_bank_count") != TOTAL_BANK_COUNT
        or metadata.get("frozen_parent_parameter_count")
        != PARENT_PARAMETER_COUNT
        or metadata.get("total_adapter_parameter_count")
        != TOTAL_PARAMETER_COUNT
        or metadata.get("v90_parent_state_sha256") != V90_STATE_SHA256
        or metadata.get("v90_parent_runtime_promotable") is not False
        or metadata.get("environmental_memory_serialized") is not False
        or metadata.get("environmental_text_serialized") is not False
        or metadata.get("environmental_text_inputs") != []
        or metadata.get("questions_or_answers_serialized") is not False
        or metadata.get("training_metadata_serialized") is not False
        or metadata.get("training_inventory_serialized") is not False
        or metadata.get("oracle_serialized") is not False
        or metadata.get("evaluation_scored") is not False
        or metadata.get("runtime_promotion_authorized") is not False
        or not isinstance(bindings, Mapping)
    ):
        raise ValueError("V91 fixed-final candidate authentication failed")
    with safe_open(str(weights), framework="pt", device="cpu") as handle:
        tensor_metadata = handle.metadata()
        tensors = {
            name: handle.get_tensor(name)
            for name in handle.keys()  # noqa: SIM118 - safe_open is not iterable.
        }
    expected_tensor_metadata = {
        "artifact": CANDIDATE_ARTIFACT,
        "environmental_memory_serialized": "false",
        "environmental_text_serialized": "false",
        "questions_or_answers_serialized": "false",
        "training_metadata_serialized": "false",
        "oracle_serialized": "false",
    }
    state = {
        "adapters.0.lora_a": tensors.get("lora_a"),
        "adapters.0.lora_b": tensors.get("lora_b"),
    }
    if (
        tensor_metadata != expected_tensor_metadata
        or set(tensors) != {"lora_a", "lora_b"}
        or tuple(tensors["lora_a"].shape) != (16, 12_288)
        or tuple(tensors["lora_b"].shape) != (1_536, 16)
        or any(value.dtype != torch.float32 for value in tensors.values())
        or tensor_state_sha256(state) != metadata.get("state_sha256")
    ):
        raise ValueError("V91 fixed-final candidate tensor contract changed")
    return metadata, fingerprint, files


def load_fixed_final_bridge_v91(
    collection: LoRABankCollection, candidate: str | Path
) -> dict[str, Any]:
    """Authenticate and load the exact two-tensor V91 fixed-final bridge."""

    root = resolve_v85(candidate)
    metadata, _fingerprint, _files = _authenticate_fixed_final_candidate_v91(
        root
    )
    weights = root / WEIGHTS_FILENAME
    _load_fresh_state_v91(collection, load_file(str(weights), device="cpu"))
    if (
        collection.bank(FRESH_BANK_NAME).installation.state_sha256()
        != metadata.get("state_sha256")
    ):
        raise ValueError("V91 fixed-final bridge state changed")
    return metadata


def authenticate_training_report_v91(
    config: Mapping[str, Any], *, config_path: str | Path = CONFIG
) -> dict[str, str]:
    """Bind the fixed training report to its exact candidate without Gemma."""

    preflight = authenticate_cpu_preflight_v91(config, config_path=config_path)
    path = resolve_v85(config["outputs"]["training_report"])
    report = strict_json_v89(path)
    candidate_root = resolve_v85(config["outputs"]["fixed_final_candidate"])
    candidate_metadata, candidate_fingerprint, candidate_files = (
        _authenticate_fixed_final_candidate_v91(candidate_root)
    )
    gates = report.get("gates")
    report_candidate = report.get("candidate")
    bridge = report.get("trainable_bridge")
    parent = report.get("frozen_parent")
    inventory = report.get("training_inventory")
    scene_memory = report.get("scene_memory")
    candidate_bindings = candidate_metadata.get("bindings")
    if (
        not isinstance(report_candidate, Mapping)
        or not isinstance(bridge, Mapping)
        or not isinstance(parent, Mapping)
        or not isinstance(inventory, Mapping)
        or not isinstance(scene_memory, Mapping)
        or not isinstance(candidate_bindings, Mapping)
        or not isinstance(parent.get("v90"), Mapping)
    ):
        raise TypeError("V91 training report candidate lineage is malformed")
    expected_v89_fingerprint, _v89_files = checkpoint_fingerprint(
        resolve_v85(config["sources"]["parent_v89_checkpoint"])
    )
    expected_bindings: dict[str, str | int | bool] = {
        "config_sha256": preflight["config_sha256"],
        "preregistration_sha256": preflight["preregistration_sha256"],
        "cpu_preflight_sha256": preflight["cpu_preflight_sha256"],
        "v89_checkpoint_sha256": expected_v89_fingerprint,
        "v89_adapter_sha256": str(
            config["sources"]["parent_v89_adapter_sha256"]
        ),
        "v89_runtime_metadata_sha256": str(
            config["sources"]["parent_v89_metadata_sha256"]
        ),
        "v90_weights_sha256": V90_WEIGHTS_SHA256,
        "v90_metadata_sha256": V90_METADATA_SHA256,
        "v90_state_sha256": V90_STATE_SHA256,
        "v90_training_report_sha256": V90_TRAINING_REPORT_SHA256,
        "v90_predictions_sha256": V90_PREDICTIONS_SHA256,
        "v90_evaluation_sha256": V90_EVALUATION_SHA256,
        "fixed_final_optimizer_updates": EXPECTED_OPTIMIZER_UPDATES,
        "training_schedule_sha256": str(
            config["dataset"]["training_schedule_sha256"]
        ),
        "training_inventory_sha256": str(
            config["dataset"]["training_inventory_sha256"]
        ),
        "scene_memory_prefix_sha256": EXPECTED_PREFIX_SHA256,
        "development_known_questions_trained": True,
        "held_wordings_excluded_from_optimization": True,
        "failed_v90_parent_runtime_promotable": False,
    }
    try:
        expected_candidate_path = candidate_root.relative_to(PROJECT_ROOT).as_posix()
    except ValueError as exc:
        raise ValueError("V91 candidate escaped the project root") from exc
    if (
        report.get("artifact") != TRAINING_ARTIFACT
        or report.get("schema_version") != 91
        or report.get("status") != "fixed_final_training_complete_not_promoted"
        or report.get("config_sha256") != preflight["config_sha256"]
        or report.get("preregistration_sha256")
        != preflight["preregistration_sha256"]
        or report.get("cpu_preflight_sha256")
        != preflight["cpu_preflight_sha256"]
        or report.get("optimizer_updates") != EXPECTED_OPTIMIZER_UPDATES
        or report.get("micro_rows_consumed") != EXPECTED_MICRO_ROWS
        or report.get("causal_margin_rows_consumed") != EXPECTED_CAUSAL_ROWS
        or report.get("protected_read_count") != 0
        or report.get("oracle_loaded") is not False
        or report.get("official_validation_loaded") is not False
        or report.get("official_test_loaded") is not False
        or report.get("deferred_final_loaded") is not False
        or report.get("held_out_scene_generalization_claim") is not False
        or report.get("runtime_promotion_authorized") is not False
        or not isinstance(gates, Mapping)
        or not gates
        or not all(value is True for value in gates.values())
        or candidate_bindings != expected_bindings
        or report_candidate.get("path") != expected_candidate_path
        or report_candidate.get("weights_sha256")
        != candidate_metadata["weights_sha256"]
        or report_candidate.get("metadata_canonical_sha256")
        != canonical_sha256_v85(candidate_metadata)
        or report_candidate.get("checkpoint_sha256") != candidate_fingerprint
        or report_candidate.get("checkpoint_files") != candidate_files
        or report_candidate.get("fixed_final") is not True
        or report_candidate.get("runtime_promotion_authorized") is not False
        or bridge.get("bank_name") != FRESH_BANK_NAME
        or bridge.get("target_module") != EXPECTED_FRESH_TARGET
        or bridge.get("rank") != 16
        or float(bridge.get("alpha", -1.0)) != 32.0
        or bridge.get("parameter_count") != FRESH_PARAMETER_COUNT
        or bridge.get("initial_state_sha256")
        != EXPECTED_FRESH_INITIAL_STATE_SHA256
        or bridge.get("final_state_sha256")
        != candidate_metadata["state_sha256"]
        or bridge.get("unmerged") is not True
        or parent.get("v89_checkpoint_sha256") != expected_v89_fingerprint
        or parent.get("v89_adapter_sha256")
        != config["sources"]["parent_v89_adapter_sha256"]
        or parent.get("v89_runtime_metadata_sha256")
        != config["sources"]["parent_v89_metadata_sha256"]
        or parent["v90"].get("weights_sha256") != V90_WEIGHTS_SHA256
        or parent["v90"].get("metadata_sha256") != V90_METADATA_SHA256
        or parent["v90"].get("state_sha256") != V90_STATE_SHA256
        or parent["v90"].get("failed_but_authenticated") is not True
        or parent.get("frozen_bank_count") != PARENT_BANK_COUNT
        or parent.get("frozen_parameter_count") != PARENT_PARAMETER_COUNT
        or parent.get("parent_tensors_loaded_byte_exactly") is not True
        or parent.get("failed_v90_parent_loaded_unmerged") is not True
        or inventory.get("inventory_sha256")
        != config["dataset"]["training_inventory_sha256"]
        or inventory.get("schedule_sha256")
        != config["dataset"]["training_schedule_sha256"]
        or scene_memory.get("prefix_sha256_before") != EXPECTED_PREFIX_SHA256
        or scene_memory.get("prefix_sha256_after") != EXPECTED_PREFIX_SHA256
    ):
        raise ValueError("V91 fixed-final training report changed")
    return {
        **preflight,
        "training_report_sha256": sha256_file_v85(path),
        "candidate_weights_sha256": str(candidate_metadata["weights_sha256"]),
        "candidate_state_sha256": str(candidate_metadata["state_sha256"]),
        "candidate_checkpoint_sha256": candidate_fingerprint,
        "candidate_metadata_canonical_sha256": canonical_sha256_v85(
            candidate_metadata
        ),
    }


def _answer_tail(language: Any, prepared: Any) -> Any:
    from semantic_3d_chat.language.gemma4_answer_tail import answer_tail_forward

    return answer_tail_forward(language, prepared)


def _empty_mps_cache(device: torch.device) -> None:
    if device.type == "mps" and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def _frozen_state_v91(collection: LoRABankCollection) -> dict[str, str]:
    return {
        bank.settings.name: bank.installation.state_sha256()
        for bank in collection.banks
        if not bank.settings.trainable
    }


def run_training_v91(config_path: str | Path = CONFIG) -> dict[str, Any]:
    """Execute the sealed 1,770-row, 295-update V91 repair schedule."""

    started = time.monotonic()
    config = load_config_v91(config_path, allow_draft=False)
    report_path = resolve_v85(config["outputs"]["training_report"])
    candidate_path = resolve_v85(config["outputs"]["fixed_final_candidate"])
    if report_path.exists() or report_path.is_symlink():
        raise FileExistsError(f"V91 fixed training report exists: {report_path}")
    if candidate_path.exists() or candidate_path.is_symlink():
        raise FileExistsError(f"V91 fixed-final candidate exists: {candidate_path}")

    audit = FileAccessAudit(
        forbidden_component_names=_FORBIDDEN_COMPONENTS,
        block_forbidden=True,
    )
    with audit:
        source_hashes = authenticate_sources_v91(config)
        preflight = authenticate_cpu_preflight_v91(config, config_path=config_path)
        canonical = load_canonical_rows_v91(config)
        items = derive_training_items_v91(config, canonical)
        inventory_hash = canonical_sha256_v85(inventory_v91(items))
        schedule = schedule_v91(
            items,
            seed=int(config["training"]["row_order_seed"]),
            epochs=int(config["dataset"]["epochs"]),
        )
        schedule_hash = canonical_sha256_v85(
            [[epoch, item.schedule_id] for epoch, item in schedule]
        )
        if (
            inventory_hash != config["dataset"]["training_inventory_sha256"]
            or schedule_hash != config["dataset"]["training_schedule_sha256"]
            or len(items) != EXPECTED_ROWS_PER_EPOCH
            or len(schedule) != EXPECTED_MICRO_ROWS
        ):
            raise RuntimeError("V91 fixed training inventory or schedule changed")

        cpu_memory, memory_hash_before, _memory_metadata = load_scene1_memory_v86(
            config
        )
        cpu_zero_memory = zero_payload_memory_v86(cpu_memory)
        zero_hash_before = prefix_sha256(cpu_zero_memory)
        runtime = load_runtime_config(config["sources"]["runtime_config"])
        language_config = runtime["language"]
        language = load_local_language_model(
            str(language_config["model_id"]),
            str(language_config["revision"]),
            str(language_config["dtype"]),
            freeze=True,
            local_files_only=True,
            backend="gemma4",
            decoder_gradient_checkpointing=True,
        )
        if language.device.type not in {"mps", "cpu"}:
            raise RuntimeError("V91 requires local MPS or CPU execution")
        collection = install_lora_banks(
            language.model,
            combined_lora_settings_v91(runtime, config),
        )
        if not isinstance(collection, LoRABankCollection):
            raise TypeError("V91 LoRA bank installation failed")
        parent = load_frozen_parent_v91(
            collection,
            config["sources"]["parent_v89_checkpoint"],
            config,
        )
        collection.assert_trainable_surface(language.model)
        if (
            collection.bank_names != (*EXPECTED_PARENT_BANKS, FRESH_BANK_NAME)
            or collection.parameter_count != TOTAL_PARAMETER_COUNT
            or collection.trainable_parameter_count != FRESH_PARAMETER_COUNT
        ):
            raise RuntimeError("V91 exact adapter parameter surface changed")
        frozen_state_before = _frozen_state_v91(collection)
        if frozen_state_before != parent["frozen_bank_state_sha256"]:
            raise RuntimeError("V91 frozen parent state changed before optimization")

        fresh = collection.bank(FRESH_BANK_NAME).installation
        initial_state_sha256 = fresh.state_sha256()
        memory = cpu_memory.to(device=language.device, dtype=torch.bfloat16)
        zero_memory = cpu_zero_memory.to(
            device=language.device,
            dtype=torch.bfloat16,
        )
        system_prompt = str(language_config["system_prompt"])
        training = config["training"]
        parameters = collection.parameters()
        optimizer = torch.optim.AdamW(
            parameters,
            lr=float(training["learning_rate"]),
            weight_decay=float(training["weight_decay"]),
        )
        language.decoder_module.train()
        collection.train()
        optimizer.zero_grad(set_to_none=True)
        accumulation = int(training["gradient_accumulation_rows"])
        if (
            accumulation != 6
            or len(schedule) % accumulation != 0
            or int(training["optimizer_updates"]) != EXPECTED_OPTIMIZER_UPDATES
            or len(schedule) // accumulation != EXPECTED_OPTIMIZER_UPDATES
        ):
            raise RuntimeError("V91 optimizer schedule changed")
        target_margin = float(training["zero_payload_target_margin_nll"])
        ce_weight = float(training["answer_ce_weight"])
        margin_weight = float(training["zero_payload_margin_weight"])
        history: list[dict[str, Any]] = []
        interval: list[dict[str, float | bool]] = []
        optimizer_update = 0
        causal_seen = 0
        kind_seen: Counter[str] = Counter()
        answer_only_seen = 0

        for cursor, (epoch, item) in enumerate(schedule, 1):
            row = item.row
            kind_seen[item.kind] += 1
            prepared, layout = _prepared_v84(language, system_prompt, memory, row)
            if (
                layout.get("answer_only_supervision") is not True
                or layout.get("memory_supplied_directly") is not True
                or layout.get("question_derived_environmental_tokens") != 0
                or layout.get("control_tokens") != 0
            ):
                raise RuntimeError("V91 answer-only direct-memory layout changed")
            answer_only_seen += 1
            tail = _answer_tail(language, prepared)
            correct_nll = tail.mean_nll.float()
            weighted_ce = ce_weight * correct_nll
            if item.causal_margin:
                zero_prepared, zero_layout = _prepared_v84(
                    language,
                    system_prompt,
                    zero_memory,
                    row,
                )
                if (
                    zero_layout.get("answer_only_supervision") is not True
                    or zero_layout.get("memory_supplied_directly") is not True
                    or zero_layout.get("question_derived_environmental_tokens") != 0
                    or zero_layout.get("control_tokens") != 0
                ):
                    raise RuntimeError("V91 zero-memory answer-only layout changed")
                zero_tail = _answer_tail(language, zero_prepared)
                _unused, observed_margin, penalty = (
                    zero_payload_margin_objective_v86(
                        correct_nll,
                        zero_tail.mean_nll.float(),
                        target_margin=target_margin,
                        ce_weight=ce_weight,
                        margin_weight=margin_weight,
                    )
                )
                objective = weighted_ce + margin_weight * penalty
                causal_seen += 1
                interval.append(
                    {
                        "correct_nll": float(correct_nll.detach().cpu()),
                        "objective": float(objective.detach().cpu()),
                        "causal": True,
                        "zero_minus_correct_nll": float(
                            observed_margin.detach().cpu()
                        ),
                        "margin_penalty": float(penalty.detach().cpu()),
                    }
                )
                del zero_prepared, zero_tail, observed_margin, penalty
            else:
                objective = weighted_ce
                interval.append(
                    {
                        "correct_nll": float(correct_nll.detach().cpu()),
                        "objective": float(objective.detach().cpu()),
                        "causal": False,
                        "zero_minus_correct_nll": 0.0,
                        "margin_penalty": 0.0,
                    }
                )
            if not torch.isfinite(objective):
                raise RuntimeError("V91 objective is nonfinite")
            (objective / accumulation).backward()
            del prepared, tail, correct_nll, weighted_ce, objective
            if cursor % accumulation:
                continue

            gradient_l2 = float(collection.gradient_norms()["total_l2"])
            if not math.isfinite(gradient_l2) or gradient_l2 <= 0.0:
                raise RuntimeError("V91 gradient is zero or nonfinite")
            clipped = torch.nn.utils.clip_grad_norm_(
                parameters,
                float(training["gradient_clip_norm"]),
            )
            clip_l2 = float(clipped.detach().cpu())
            if not math.isfinite(clip_l2):
                raise RuntimeError("V91 clipped gradient is nonfinite")
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_update += 1
            collection.validate_state()
            causal_interval = [record for record in interval if record["causal"]]
            record = {
                "update": optimizer_update,
                "row_cursor": cursor,
                "epoch": epoch,
                "mean_correct_nll": sum(
                    float(value["correct_nll"]) for value in interval
                )
                / len(interval),
                "mean_objective": sum(
                    float(value["objective"]) for value in interval
                )
                / len(interval),
                "causal_rows": len(causal_interval),
                "mean_causal_zero_minus_correct_nll": (
                    sum(
                        float(value["zero_minus_correct_nll"])
                        for value in causal_interval
                    )
                    / len(causal_interval)
                    if causal_interval
                    else None
                ),
                "mean_causal_margin_penalty": (
                    sum(
                        float(value["margin_penalty"])
                        for value in causal_interval
                    )
                    / len(causal_interval)
                    if causal_interval
                    else None
                ),
                "gradient_l2_before_clip": gradient_l2,
                "clip_return_l2": clip_l2,
                "state_sha256": fresh.state_sha256(),
            }
            history.append(record)
            interval.clear()
            if optimizer_update in {1, 148, EXPECTED_OPTIMIZER_UPDATES} or (
                optimizer_update % 12 == 0
            ):
                print(
                    json.dumps(
                        {
                            "event": "v91_train_update",
                            "update": optimizer_update,
                            "total_updates": EXPECTED_OPTIMIZER_UPDATES,
                            "row_cursor": cursor,
                            "epoch": epoch,
                            "mean_correct_nll": record["mean_correct_nll"],
                            "causal_rows_seen": causal_seen,
                            "elapsed_seconds": time.monotonic() - started,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            _empty_mps_cache(language.device)

        expected_kinds = Counter(
            {
                "canonical": 414,
                "error_replay": 84,
                "correct_anchor_replay": 372,
                "conversational_success": 252,
                "conversational_repair": 648,
            }
        )
        if (
            optimizer_update != EXPECTED_OPTIMIZER_UPDATES
            or len(history) != EXPECTED_OPTIMIZER_UPDATES
            or causal_seen != EXPECTED_CAUSAL_ROWS
            or answer_only_seen != EXPECTED_MICRO_ROWS
            or kind_seen != expected_kinds
            or interval
        ):
            raise RuntimeError("V91 fixed conversational repair schedule did not complete")
        language.decoder_module.eval()
        collection.eval()
        memory_hash_after = prefix_sha256(memory.detach().cpu())
        zero_hash_after = prefix_sha256(zero_memory.detach().cpu())
        frozen_state_after = _frozen_state_v91(collection)
        if (
            memory_hash_after != memory_hash_before
            or zero_hash_after != zero_hash_before
            or frozen_state_after != frozen_state_before
        ):
            raise RuntimeError("V91 immutable parent or environmental inputs mutated")

        bindings: dict[str, str | int | bool] = {
            "config_sha256": preflight["config_sha256"],
            "preregistration_sha256": preflight["preregistration_sha256"],
            "cpu_preflight_sha256": preflight["cpu_preflight_sha256"],
            "v89_checkpoint_sha256": parent["v89_checkpoint_sha256"],
            "v89_adapter_sha256": parent["v89_adapter_sha256"],
            "v89_runtime_metadata_sha256": parent[
                "v89_runtime_metadata_sha256"
            ],
            "v90_weights_sha256": parent["v90"]["weights_sha256"],
            "v90_metadata_sha256": parent["v90"]["metadata_sha256"],
            "v90_state_sha256": parent["v90"]["state_sha256"],
            "v90_training_report_sha256": parent["v90"][
                "training_report_sha256"
            ],
            "v90_predictions_sha256": parent["v90"]["predictions_sha256"],
            "v90_evaluation_sha256": parent["v90"]["evaluation_sha256"],
            "fixed_final_optimizer_updates": optimizer_update,
            "training_schedule_sha256": schedule_hash,
            "training_inventory_sha256": inventory_hash,
            "scene_memory_prefix_sha256": memory_hash_before,
            "development_known_questions_trained": True,
            "held_wordings_excluded_from_optimization": True,
            "failed_v90_parent_runtime_promotable": False,
        }
        candidate_metadata = publish_fixed_final_candidate_v91(
            candidate_path,
            collection,
            bindings=bindings,
            experiment=config,
        )
        candidate_fingerprint, candidate_files = _candidate_fingerprint_v91(
            candidate_path
        )

    audit.assert_clean()
    gates = {
        "all_1770_sealed_micro_rows_consumed": len(schedule)
        == EXPECTED_MICRO_ROWS,
        "all_138_canonical_rows_consumed_once_each_epoch": kind_seen["canonical"]
        == 414,
        "all_14_parent_errors_replayed_twice_each_epoch": kind_seen[
            "error_replay"
        ]
        == 84,
        "all_124_parent_correct_anchors_replayed_once_each_epoch": kind_seen[
            "correct_anchor_replay"
        ]
        == 372,
        "all_84_success_conversational_rows_consumed_each_epoch": kind_seen[
            "conversational_success"
        ]
        == 252,
        "all_216_repair_conversational_rows_consumed_each_epoch": kind_seen[
            "conversational_repair"
        ]
        == 648,
        "all_39_primary_causal_margin_rows_consumed": causal_seen
        == EXPECTED_CAUSAL_ROWS,
        "answer_only_ce_on_every_micro_row": answer_only_seen
        == EXPECTED_MICRO_ROWS,
        "fixed_final_update_295_reached": optimizer_update
        == EXPECTED_OPTIMIZER_UPDATES,
        "exact_v89_plus_failed_v90_twelve_bank_parent_frozen": (
            parent["frozen_bank_count"] == PARENT_BANK_COUNT
            and parent["parent_tensors_loaded_byte_exactly"] is True
            and parent["failed_v90_parent_loaded_unmerged"] is True
            and parent["v90"]["failed_but_authenticated"] is True
            and frozen_state_after == frozen_state_before
        ),
        "sole_trainable_surface_is_v91_repair_bridge": (
            collection.trainable_parameter_count == FRESH_PARAMETER_COUNT
            and collection.bank(FRESH_BANK_NAME).settings.trainable
            and sum(bank.settings.trainable for bank in collection.banks) == 1
        ),
        "nonzero_finite_gradient_every_update": all(
            math.isfinite(float(row["gradient_l2_before_clip"]))
            and float(row["gradient_l2_before_clip"]) > 0.0
            for row in history
        ),
        "memory_hash_invariant": memory_hash_after == memory_hash_before,
        "zero_payload_hash_invariant": zero_hash_after == zero_hash_before,
        "protected_read_count_zero": len(audit.forbidden_accesses()) == 0,
        "runtime_candidate_contains_no_supervision_or_environment": (
            candidate_metadata["questions_or_answers_serialized"] is False
            and candidate_metadata["training_metadata_serialized"] is False
            and candidate_metadata["training_inventory_serialized"] is False
            and candidate_metadata["environmental_memory_serialized"] is False
            and candidate_metadata["environmental_text_serialized"] is False
            and candidate_metadata["environmental_text_inputs"] == []
            and candidate_metadata["oracle_serialized"] is False
        ),
    }
    if not all(gates.values()):
        raise RuntimeError(f"V91 training gate failed: {gates}")

    report = {
        "artifact": TRAINING_ARTIFACT,
        "schema_version": 91,
        "status": "fixed_final_training_complete_not_promoted",
        "config_sha256": preflight["config_sha256"],
        "preregistration_sha256": preflight["preregistration_sha256"],
        "cpu_preflight_sha256": preflight["cpu_preflight_sha256"],
        "device": language.device.type,
        "model_id": config["sources"]["model_id"],
        "model_revision": config["sources"]["model_revision"],
        "strict_input_contract": config["strict_input_contract"],
        "source_hashes": source_hashes,
        "frozen_parent": {
            **parent,
            "frozen_bank_state_before": frozen_state_before,
            "frozen_bank_state_after": frozen_state_after,
            "frozen_bank_state_invariant": frozen_state_after
            == frozen_state_before,
        },
        "trainable_bridge": {
            "bank_name": FRESH_BANK_NAME,
            "target_module": EXPECTED_FRESH_TARGET,
            "rank": 16,
            "alpha": 32.0,
            "parameter_count": FRESH_PARAMETER_COUNT,
            "initial_state_sha256": initial_state_sha256,
            "final_state_sha256": candidate_metadata["state_sha256"],
            "unmerged": True,
        },
        "training_protocol": config["training"],
        "training_inventory": {
            "canonical_unique_rows": 138,
            "parent_errors": 14,
            "parent_error_replays_per_epoch": 28,
            "parent_correct_anchors": 124,
            "successful_conversational_rows_per_epoch": 84,
            "repair_conversational_rows_per_epoch": 216,
            "primary_causal_rows_per_epoch": 13,
            "unique_schedule_items_per_epoch": EXPECTED_ROWS_PER_EPOCH,
            "inventory_sha256": inventory_hash,
            "schedule_sha256": schedule_hash,
            "development_known_questions_trained": True,
            "new_held_wording_rows_excluded": 26,
            "held_out_scene_generalization_claim": False,
            "answers_or_questions_serialized_in_candidate": False,
            "inventory_serialized_in_candidate": False,
        },
        "micro_rows_consumed": len(schedule),
        "causal_margin_rows_consumed": causal_seen,
        "optimizer_updates": optimizer_update,
        "training_history": history,
        "scene_memory": {
            "compiled_before_question_tokenization": True,
            "shape": [1, 738, 1536],
            "prefix_sha256_before": memory_hash_before,
            "prefix_sha256_after": memory_hash_after,
            "zero_payload_prefix_sha256_before": zero_hash_before,
            "zero_payload_prefix_sha256_after": zero_hash_after,
            "native_boi_eoi_retained_in_zero_control": True,
            "zero_payload_tokens": 736,
            "question_derived_environmental_tokens": 0,
            "question_conditioned_environmental_readout": False,
            "question_dependent_retrieval": False,
        },
        "gates": gates,
        "candidate": {
            "path": candidate_path.relative_to(PROJECT_ROOT).as_posix(),
            "weights_sha256": candidate_metadata["weights_sha256"],
            "metadata_canonical_sha256": canonical_sha256_v85(
                candidate_metadata
            ),
            "checkpoint_sha256": candidate_fingerprint,
            "checkpoint_files": candidate_files,
            "fixed_final": True,
            "runtime_promotion_authorized": False,
        },
        "loaded_file_count": len(audit.unique_paths),
        "loaded_file_inventory_sha256": canonical_sha256_v85(audit.unique_paths),
        "protected_read_count": len(audit.forbidden_accesses()),
        "official_validation_loaded": False,
        "official_test_loaded": False,
        "deferred_final_loaded": False,
        "oracle_loaded": False,
        "held_out_scene_generalization_claim": False,
        "runtime_promotion_authorized": False,
        "elapsed_seconds": time.monotonic() - started,
    }
    atomic_create_json_v85(report_path, report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(CONFIG))
    args = parser.parse_args(argv)
    report = run_training_v91(args.config)
    print(
        json.dumps(
            {
                "status": report["status"],
                "optimizer_updates": report["optimizer_updates"],
                "micro_rows_consumed": report["micro_rows_consumed"],
                "candidate": report["candidate"],
                "elapsed_seconds": report["elapsed_seconds"],
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CANDIDATE_ARTIFACT",
    "METADATA_FILENAME",
    "TRAINING_ARTIFACT",
    "WEIGHTS_FILENAME",
    "authenticate_training_report_v91",
    "combined_lora_settings_v91",
    "load_fixed_final_bridge_v91",
    "load_frozen_parent_v91",
    "main",
    "publish_fixed_final_candidate_v91",
    "run_training_v91",
]
