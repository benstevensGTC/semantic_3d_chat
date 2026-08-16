"""Train V92's retention-aware conversational repair bridge.

V92 freezes the exact V89 eleven-bank release and the exact V90 and V91
development bridges as thirteen immutable, unmerged LoRA banks.  It optimizes
only a fresh rank-8 bank on layer 29's attention output projection.  Every row
uses the same pre-question 738-token continuous scene memory.  The create-once
candidate contains only V92's two fresh tensors and sanitized provenance.
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
    load_config_v91,
)
from semantic_3d_chat.evaluation.v92_scene1_retention_conversation_preflight import (
    CONFIG,
    EXPECTED_INITIAL_STATE_SHA256,
    FRESH_BANK_NAME,
    TARGET_MODULE,
    authenticate_cpu_preflight_v92,
    authenticate_sources_v92,
    derive_training_items_v92,
    inventory_v92,
    load_canonical_rows_v92,
    load_config_v92,
    schedule_v92,
)
from semantic_3d_chat.language.local_lm import load_local_language_model
from semantic_3d_chat.language.lora import (
    InstalledLoRABank,
    LoRABankCollection,
    LoRABankSettings,
    LoRABanksSettings,
    LoRASettings,
    install_lora_banks,
    tensor_state_sha256,
)
from semantic_3d_chat.language.prefix_injection import prefix_sha256
from semantic_3d_chat.training.train_v84_strict_bridge import _prepared_v84
from semantic_3d_chat.training.train_v86_scene1_demo import (
    zero_payload_margin_objective_v86,
)
from semantic_3d_chat.training.train_v91_scene1_conversational_repair import (
    EXPECTED_PARENT_BANKS as V91_FIRST_TWELVE_BANKS,
)
from semantic_3d_chat.training.train_v91_scene1_conversational_repair import (
    FRESH_BANK_NAME as V91_BANK_NAME,
)
from semantic_3d_chat.training.train_v91_scene1_conversational_repair import (
    TOTAL_PARAMETER_COUNT as V91_TOTAL_PARAMETER_COUNT,
)
from semantic_3d_chat.training.train_v91_scene1_conversational_repair import (
    authenticate_training_report_v91,
    combined_lora_settings_v91,
    load_fixed_final_bridge_v91,
    load_frozen_parent_v91,
)

TRAINING_ARTIFACT: Final[str] = "gemma4_v92_scene1_retention_conversation_repair_training_v1"
CANDIDATE_ARTIFACT: Final[str] = "gemma4_v92_scene1_retention_conversation_repair_fixed_final_v1"
WEIGHTS_FILENAME: Final[str] = "bridge.safetensors"
METADATA_FILENAME: Final[str] = "runtime_metadata.json"
PARENT_BANK_COUNT: Final[int] = 13
TOTAL_BANK_COUNT: Final[int] = 14
PARENT_PARAMETER_COUNT: Final[int] = 1_122_304
FRESH_PARAMETER_COUNT: Final[int] = 45_056
TOTAL_PARAMETER_COUNT: Final[int] = 1_167_360
EXPECTED_ROWS_PER_EPOCH: Final[int] = 590
EXPECTED_MICRO_ROWS: Final[int] = 1_770
EXPECTED_OPTIMIZER_UPDATES: Final[int] = 295
EXPECTED_CAUSAL_ROWS: Final[int] = 39
EXPECTED_FRESH_TARGET: Final[str] = "model.language_model.layers.29.self_attn.o_proj"
EXPECTED_FRESH_INITIAL_STATE_SHA256: Final[str] = (
    "c10d38c727df1520418a5bb9be7bac262a2b6acdef07203a669ff52f8cd08cc1"
)

V91_CONFIG_SHA256: Final[str] = "0270a5731059b286cc26657cc3b10846397a1ba67054f93ebfd3adaa25e9885f"
V91_PREFLIGHT_SOURCE_SHA256: Final[str] = (
    "5e98e7efd59133e76ce71553ac2f0d36dad5fc774cf281e3742f2b5d0fb61c5d"
)
V91_TRAINING_SOURCE_SHA256: Final[str] = (
    "843613f3953e4bcd2d580aa6d7ab4b9197cfd825bfe787b9b6072a3a672ce67b"
)
V91_EVALUATION_SOURCE_SHA256: Final[str] = (
    "6939b9f3e5ed3b6d3088281ce13951515ebb99d40b7efe616df5563ef42bf2a4"
)
V91_PREREGISTRATION_SHA256: Final[str] = (
    "151d82517b6a902ba8eb11bb37a90c9eaa4e9f0ad6ba017b68a6bb1c384a23f0"
)
V91_CPU_PREFLIGHT_SHA256: Final[str] = (
    "79da2ca3bf54bc1fb808e839a00e5e6af4a74f0fcaad5a8f72b0c7ee1b2fc6f9"
)
V91_TRAINING_REPORT_SHA256: Final[str] = (
    "2cf3a4d9576cb318c4d4c7a04808a44d5ad1014df5c5c33de275b7eb85337fdc"
)
V91_WEIGHTS_SHA256: Final[str] = "1cccb4e25d11ce88e881d9b7df2ba4c521aceef55a7b6617a1172ee367f782d0"
V91_METADATA_SHA256: Final[str] = "9cd41f9d3bec7f6189e2613d59dccd9e23db213434443528b89743430dfc3ec7"
V91_STATE_SHA256: Final[str] = "53022311c3bc5e249a6d262fbb19b6e893a6af085be542e4d6941f7a13ea72cd"
V91_CHECKPOINT_SHA256: Final[str] = (
    "02c140b1ceaa29252d766aa19931e24a52a8175c17c93f306e510cd1215bf148"
)
V91_PREDICTIONS_SHA256: Final[str] = (
    "401e4e9fb71ce8c242afe4a2ae699f1007b6ecabf53ce7afd158393ad8c431a5"
)
V91_EVALUATION_SHA256: Final[str] = (
    "7d92df4e32ca11b99cdce48c94f2742c743037411f84346f97b0fae45ee57d2f"
)
V91_ARTIFACT: Final[str] = "gemma4_v91_scene1_conversational_repair_fixed_final_v1"
EXPECTED_PARENT_BANKS: Final[tuple[str, ...]] = (
    *V91_FIRST_TWELVE_BANKS,
    V91_BANK_NAME,
)
_FORBIDDEN_COMPONENTS: Final[frozenset[str]] = frozenset(
    {"oracle", "validation", "test", "deferred"}
)


def _source_value(sources: Mapping[str, Any], *keys: str, required: bool = True) -> Any:
    for key in keys:
        if key in sources:
            return sources[key]
    if required:
        raise KeyError(f"V92 config omitted source aliases: {keys}")
    return None


def _source_path(sources: Mapping[str, Any], *keys: str) -> Path:
    return resolve_v85(str(_source_value(sources, *keys)))


def _v91_paths(sources: Mapping[str, Any]) -> dict[str, Path]:
    return {
        "config": _source_path(sources, "parent_v91_config", "v91_config"),
        "preregistration": _source_path(
            sources, "parent_v91_preregistration", "v91_preregistration"
        ),
        "cpu_preflight": _source_path(sources, "parent_v91_cpu_preflight", "v91_cpu_preflight"),
        "training": _source_path(sources, "parent_v91_training", "v91_training_report"),
        "candidate": _source_path(sources, "parent_v91_candidate", "v91_candidate"),
        "predictions": _source_path(sources, "parent_v91_predictions", "v91_predictions"),
        "evaluation": _source_path(sources, "parent_v91_evaluation", "v91_evaluation"),
    }


def _copy_bank_state_v92(
    installation: Any, archive: Mapping[str, torch.Tensor], *, context: str
) -> None:
    expected_keys = set(installation.state_module.state_dict())
    if set(archive) != expected_keys:
        raise ValueError(f"{context} tensor inventory changed")
    if any(value.dtype != torch.float32 for value in archive.values()):
        raise TypeError(f"{context} tensors must be float32")
    installation.state_module.load_state_dict(dict(archive), strict=True)
    installation.validate_state()


def _fresh_state_v92(collection: LoRABankCollection) -> dict[str, torch.Tensor]:
    fresh = collection.bank(FRESH_BANK_NAME).installation
    if len(fresh.adapters) != 1:
        raise ValueError("V92 fresh bank must wrap exactly one module")
    adapter = fresh.adapters[0]
    state = {
        "lora_a": adapter.lora_a.detach().cpu().contiguous(),
        "lora_b": adapter.lora_b.detach().cpu().contiguous(),
    }
    if (
        tuple(state["lora_a"].shape) != (8, 4_096)
        or tuple(state["lora_b"].shape) != (1_536, 8)
        or any(value.dtype != torch.float32 for value in state.values())
        or sum(value.numel() for value in state.values()) != FRESH_PARAMETER_COUNT
    ):
        raise ValueError("V92 fresh bridge tensor topology changed")
    return state


def _load_fresh_state_v92(
    collection: LoRABankCollection, archive: Mapping[str, torch.Tensor]
) -> None:
    fresh = collection.bank(FRESH_BANK_NAME).installation
    if set(archive) != {"lora_a", "lora_b"} or len(fresh.adapters) != 1:
        raise ValueError("V92 fresh bridge tensor inventory changed")
    adapter = fresh.adapters[0]
    with torch.no_grad():
        for name, parameter in (
            ("lora_a", adapter.lora_a),
            ("lora_b", adapter.lora_b),
        ):
            value = archive[name]
            if value.dtype != torch.float32 or value.shape != parameter.shape:
                raise ValueError(f"V92 fresh bridge {name} shape or dtype changed")
            parameter.copy_(value.to(parameter.device))
    collection.validate_state()


def _authenticate_failed_v91(config: Mapping[str, Any]) -> dict[str, Any]:
    """Authenticate the full V91 lineage and its measured failure, model-free."""

    paths = _v91_paths(config["sources"])
    pinned = {
        paths["config"]: V91_CONFIG_SHA256,
        paths["preregistration"]: V91_PREREGISTRATION_SHA256,
        paths["cpu_preflight"]: V91_CPU_PREFLIGHT_SHA256,
        paths["training"]: V91_TRAINING_REPORT_SHA256,
        paths["candidate"] / WEIGHTS_FILENAME: V91_WEIGHTS_SHA256,
        paths["candidate"] / METADATA_FILENAME: V91_METADATA_SHA256,
        paths["predictions"]: V91_PREDICTIONS_SHA256,
        paths["evaluation"]: V91_EVALUATION_SHA256,
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
        raise ValueError(f"V92 exact V91 evidence changed: {mismatches}")

    v91_config = load_config_v91(paths["config"], allow_draft=False)
    training_evidence = authenticate_training_report_v91(
        v91_config,
        config_path=paths["config"],
    )
    metadata = strict_json_v89(paths["candidate"] / METADATA_FILENAME)
    training = strict_json_v89(paths["training"])
    predictions = strict_json_v89(paths["predictions"])
    evaluation = strict_json_v89(paths["evaluation"])
    metrics = evaluation.get("metrics")
    gates = metrics.get("model_acceptance_gates") if isinstance(metrics, Mapping) else None
    canonical = (
        metrics.get("canonical_strict_normalized_exact") if isinstance(metrics, Mapping) else None
    )
    by_type = (
        metrics.get("canonical_accuracy_by_answer_type") if isinstance(metrics, Mapping) else None
    )
    primary = metrics.get("primary_conversational") if isinstance(metrics, Mapping) else None
    held = metrics.get("new_held_wording") if isinstance(metrics, Mapping) else None
    causal = metrics.get("causal_control") if isinstance(metrics, Mapping) else None
    failed_gate_names = (
        {name for name, passed in gates.items() if passed is False}
        if isinstance(gates, Mapping)
        else None
    )
    expected_failed_gates = {
        "all_six_core_actionable_intents_correct",
        "canonical_attribute_correct_at_least_minimum",
        "canonical_correct_at_least_preregistered_minimum",
        "canonical_spatial_correct_at_least_minimum",
        "canonical_support_correct_at_least_minimum",
        "primary_conversational_correct_at_least_required",
    }
    primary_records = primary.get("records") if isinstance(primary, Mapping) else None
    primary_failures = (
        {
            str(row.get("intent_id"))
            for row in primary_records
            if isinstance(row, Mapping) and row.get("strict_normalized_exact") is False
        }
        if isinstance(primary_records, list)
        else None
    )
    expected_type_counts = {
        "presence": (22, 22),
        "count": (9, 9),
        "metric": (1, 1),
        "attribute": (14, 18),
        "spatial_relation": (69, 86),
        "support": (0, 2),
    }
    observed_type_counts = (
        {
            name: (int(row.get("correct", -1)), int(row.get("total", -1)))
            for name, row in by_type.items()
            if isinstance(row, Mapping)
        }
        if isinstance(by_type, Mapping)
        else None
    )
    if (
        metadata.get("artifact") != V91_ARTIFACT
        or metadata.get("schema_version") != 91
        or metadata.get("status") != "fixed_final_awaiting_preregistered_acceptance_gates"
        or metadata.get("bank_name") != V91_BANK_NAME
        or metadata.get("target_module") != "model.language_model.layers.33.mlp.down_proj"
        or metadata.get("rank") != 16
        or float(metadata.get("alpha", -1.0)) != 32.0
        or metadata.get("parameter_count") != 221_184
        or metadata.get("state_sha256") != V91_STATE_SHA256
        or metadata.get("weights_sha256") != V91_WEIGHTS_SHA256
        or metadata.get("runtime_promotion_authorized") is not False
        or metadata.get("environmental_memory_serialized") is not False
        or metadata.get("environmental_text_serialized") is not False
        or metadata.get("environmental_text_inputs") != []
        or metadata.get("questions_or_answers_serialized") is not False
        or metadata.get("training_metadata_serialized") is not False
        or metadata.get("training_inventory_serialized") is not False
        or metadata.get("oracle_serialized") is not False
        or training_evidence.get("training_report_sha256") != V91_TRAINING_REPORT_SHA256
        or training_evidence.get("candidate_weights_sha256") != V91_WEIGHTS_SHA256
        or training_evidence.get("candidate_state_sha256") != V91_STATE_SHA256
        or training_evidence.get("candidate_checkpoint_sha256") != V91_CHECKPOINT_SHA256
        or training.get("artifact") != "gemma4_v91_scene1_conversational_repair_training_v1"
        or training.get("status") != "fixed_final_training_complete_not_promoted"
        or training.get("optimizer_updates") != 295
        or training.get("micro_rows_consumed") != 1_770
        or training.get("protected_read_count") != 0
        or training.get("oracle_loaded") is not False
        or evaluation.get("artifact") != "gemma4_v91_scene1_conversational_repair_evaluation_v1"
        or evaluation.get("status") != "model_gates_fail_not_runtime_promotable"
        or not isinstance(metrics, Mapping)
        or metrics.get("model_acceptance_gate_passed") is not False
        or metrics.get("separate_runtime_packaging_authorized") is not False
        or metrics.get("runtime_promotion_authorized") is not False
        or failed_gate_names != expected_failed_gates
        or not isinstance(canonical, Mapping)
        or canonical.get("correct") != 115
        or canonical.get("total") != 138
        or observed_type_counts != expected_type_counts
        or not isinstance(primary, Mapping)
        or primary.get("correct") != 10
        or primary.get("total") != 13
        or primary.get("core_actionable_correct") != 5
        or primary.get("core_actionable_total") != 6
        or primary_failures != {"inventory", "bowl_left_chair", "table_contents"}
        or not isinstance(held, Mapping)
        or held.get("correct") != 23
        or held.get("total") != 26
        or not isinstance(causal, Mapping)
        or causal.get("row_count") != 13
        or float(causal.get("mean_zero_minus_correct_nll", 0.0)) != 1.3944181121218067
        or causal.get("canonical_prediction_changes") != 10
        or evaluation.get("oracle_loaded") is not False
        or evaluation.get("held_out_scene") is not False
        or evaluation.get("runtime_promotion_authorized") is not False
        or predictions.get("candidate", {}).get("weights_sha256") != V91_WEIGHTS_SHA256
        or predictions.get("candidate", {}).get("state_sha256") != V91_STATE_SHA256
        or predictions.get("training_report_sha256") != V91_TRAINING_REPORT_SHA256
        or predictions.get("frozen_parent_state_invariant") is not True
        or predictions.get("runtime_promotion_authorized") is not False
    ):
        raise ValueError("V92 measured V91 failure authentication changed")

    with safe_open(
        str(paths["candidate"] / WEIGHTS_FILENAME), framework="pt", device="cpu"
    ) as handle:
        tensor_metadata = handle.metadata()
        raw = {
            name: handle.get_tensor(name)
            for name in handle.keys()  # noqa: SIM118 - safe_open is not iterable.
        }
    state = {
        "adapters.0.lora_a": raw.get("lora_a"),
        "adapters.0.lora_b": raw.get("lora_b"),
    }
    if (
        set(raw) != {"lora_a", "lora_b"}
        or tuple(raw["lora_a"].shape) != (16, 12_288)
        or tuple(raw["lora_b"].shape) != (1_536, 16)
        or any(value.dtype != torch.float32 for value in raw.values())
        or tensor_state_sha256(state) != V91_STATE_SHA256
        or not isinstance(tensor_metadata, dict)
        or tensor_metadata.get("environmental_memory_serialized") != "false"
        or tensor_metadata.get("questions_or_answers_serialized") != "false"
        or tensor_metadata.get("oracle_serialized") != "false"
    ):
        raise ValueError("V92 V91 parent tensor artifact is unsafe")
    return {
        "config_path": str(paths["config"]),
        "candidate_path": str(paths["candidate"]),
        "config_sha256": V91_CONFIG_SHA256,
        "preregistration_sha256": V91_PREREGISTRATION_SHA256,
        "cpu_preflight_sha256": V91_CPU_PREFLIGHT_SHA256,
        "preflight_source_sha256": V91_PREFLIGHT_SOURCE_SHA256,
        "training_source_sha256": V91_TRAINING_SOURCE_SHA256,
        "evaluation_source_sha256": V91_EVALUATION_SOURCE_SHA256,
        "training_report_sha256": V91_TRAINING_REPORT_SHA256,
        "weights_sha256": V91_WEIGHTS_SHA256,
        "metadata_sha256": V91_METADATA_SHA256,
        "state_sha256": V91_STATE_SHA256,
        "checkpoint_sha256": V91_CHECKPOINT_SHA256,
        "predictions_sha256": V91_PREDICTIONS_SHA256,
        "evaluation_sha256": V91_EVALUATION_SHA256,
        "canonical_correct": 115,
        "canonical_errors": 23,
        "primary_correct": 10,
        "primary_failures": ["inventory", "bowl_left_chair", "table_contents"],
        "held_wording_correct": 23,
        "failed_but_authenticated": True,
        "runtime_promotion_authorized": False,
        "held_out_generalization_claim": False,
    }


def combined_lora_settings_v92(
    runtime_config: Mapping[str, Any], experiment: Mapping[str, Any]
) -> LoRABanksSettings:
    """Return exact V89+V90+V91 frozen banks and V92's sole trainable bank."""

    v91_paths = _v91_paths(experiment["sources"])
    v91_config = load_config_v91(v91_paths["config"], allow_draft=False)
    inherited = combined_lora_settings_v91(runtime_config, v91_config)
    if (
        tuple(bank.name for bank in inherited.banks[:-1]) != V91_FIRST_TWELVE_BANKS
        or inherited.banks[-1].name != V91_BANK_NAME
        or not inherited.banks[-1].trainable
    ):
        raise ValueError("V92 requires the exact V91 thirteen-bank topology")
    v91_bank = LoRABankSettings(
        name=V91_BANK_NAME,
        trainable=False,
        adapter=inherited.banks[-1].adapter,
        initialization_algorithm=inherited.banks[-1].initialization_algorithm,
        initialization_seed=inherited.banks[-1].initialization_seed,
        expected_initial_state_sha256=inherited.banks[-1].expected_initial_state_sha256,
    )
    bridge = experiment["bridge"]
    if (
        str(bridge["bank_name"]) != FRESH_BANK_NAME
        or str(bridge["target_module"]) != TARGET_MODULE
        or TARGET_MODULE != EXPECTED_FRESH_TARGET
        or int(bridge["rank"]) != 8
        or float(bridge["alpha"]) != 16.0
        or float(bridge["dropout"]) != 0.0
        or int(bridge["trainable_parameter_count"]) != FRESH_PARAMETER_COUNT
        or str(bridge["expected_initial_state_sha256"]) != EXPECTED_FRESH_INITIAL_STATE_SHA256
        or str(bridge["expected_initial_state_sha256"]) != EXPECTED_INITIAL_STATE_SHA256
    ):
        raise ValueError("V92 fresh repair bridge topology changed")
    fresh = LoRABankSettings(
        name=FRESH_BANK_NAME,
        trainable=True,
        adapter=LoRASettings(
            enabled=True,
            rank=8,
            alpha=16.0,
            dropout=0.0,
            target_modules=(EXPECTED_FRESH_TARGET,),
        ),
        initialization_algorithm=str(bridge["initialization_algorithm"]),
        initialization_seed=int(bridge["initialization_seed"]),
        expected_initial_state_sha256=str(bridge["expected_initial_state_sha256"]),
    )
    settings = LoRABanksSettings((*inherited.banks[:-1], v91_bank, fresh))
    if (
        tuple(bank.name for bank in settings.banks) != (*EXPECTED_PARENT_BANKS, FRESH_BANK_NAME)
        or sum(bank.trainable for bank in settings.banks) != 1
        or not settings.bank(FRESH_BANK_NAME).trainable
    ):
        raise RuntimeError("V92 exact fourteen-bank optimizer surface changed")
    return settings


def _v91_proxy_collection(
    collection: LoRABankCollection,
    runtime_config: Mapping[str, Any],
    v91_config: Mapping[str, Any],
) -> LoRABankCollection:
    """Expose the first thirteen installations under V91's loader contract."""

    settings = combined_lora_settings_v91(runtime_config, v91_config)
    banks = tuple(
        InstalledLoRABank(
            setting,
            collection.bank(setting.name).installation,
        )
        for setting in settings.banks
    )
    return LoRABankCollection(settings=settings, banks=banks)


def load_frozen_parent_v92(
    collection: LoRABankCollection,
    parent_checkpoint: str | Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Load the exact V89+V90+V91 thirteen-bank parent byte-for-byte."""

    v91_evidence = _authenticate_failed_v91(config)
    v91_config = load_config_v91(v91_evidence["config_path"], allow_draft=False)
    runtime_config = load_runtime_config(config["sources"]["runtime_config"])
    proxy = _v91_proxy_collection(collection, runtime_config, v91_config)
    lineage = load_frozen_parent_v91(
        proxy,
        parent_checkpoint,
        v91_config,
    )
    v91_metadata = load_fixed_final_bridge_v91(
        proxy,
        v91_evidence["candidate_path"],
    )
    v91_bank = collection.bank(V91_BANK_NAME)
    frozen = tuple(bank for bank in collection.banks if not bank.settings.trainable)
    trainable = tuple(bank for bank in collection.banks if bank.settings.trainable)
    if (
        tuple(bank.settings.name for bank in frozen) != EXPECTED_PARENT_BANKS
        or len(frozen) != PARENT_BANK_COUNT
        or len(trainable) != 1
        or trainable[0].settings.name != FRESH_BANK_NAME
        or v91_bank.settings.trainable
        or v91_bank.installation.target_names != ("model.language_model.layers.33.mlp.down_proj",)
        or v91_bank.installation.parameter_count != 221_184
        or v91_bank.installation.state_sha256() != V91_STATE_SHA256
        or v91_metadata.get("state_sha256") != V91_STATE_SHA256
        or V91_TOTAL_PARAMETER_COUNT != PARENT_PARAMETER_COUNT
    ):
        raise ValueError("V92 exact frozen V91 parent load changed")
    fresh = trainable[0].installation
    if (
        fresh.target_names != (EXPECTED_FRESH_TARGET,)
        or fresh.parameter_count != FRESH_PARAMETER_COUNT
        or fresh.state_sha256() != EXPECTED_FRESH_INITIAL_STATE_SHA256
        or any(int(torch.count_nonzero(adapter.lora_b).item()) != 0 for adapter in fresh.adapters)
    ):
        raise ValueError("V92 fresh bridge did not start at exact zero output")
    collection.validate_state()
    observed_states = {bank.settings.name: bank.installation.state_sha256() for bank in frozen}
    if observed_states[V91_BANK_NAME] != V91_STATE_SHA256:
        raise ValueError("V92 V91 parent state changed after load")
    return {
        "v89_checkpoint_sha256": lineage["v89_checkpoint_sha256"],
        "v89_checkpoint_files": lineage["v89_checkpoint_files"],
        "v89_adapter_sha256": lineage["v89_adapter_sha256"],
        "v89_runtime_metadata_sha256": lineage["v89_runtime_metadata_sha256"],
        "v89_bank_count": lineage["v89_bank_count"],
        "v89_parameter_count": lineage["v89_parameter_count"],
        "v89_release_provenance_sha256": lineage["v89_release_provenance_sha256"],
        "v90": lineage["v90"],
        "v91": v91_evidence,
        "frozen_bank_state_sha256": observed_states,
        "frozen_bank_count": PARENT_BANK_COUNT,
        "frozen_parameter_count": PARENT_PARAMETER_COUNT,
        "fresh_initial_state_sha256": fresh.state_sha256(),
        "parent_tensors_loaded_byte_exactly": True,
        "failed_v90_parent_loaded_unmerged": True,
        "failed_v91_parent_loaded_unmerged": True,
    }


def _candidate_fingerprint_v92(root: Path) -> tuple[str, list[dict[str, Any]]]:
    files = [root / WEIGHTS_FILENAME, root / METADATA_FILENAME]
    if (
        root.is_symlink()
        or not root.is_dir()
        or any(path.is_symlink() or not path.is_file() for path in files)
    ):
        raise FileNotFoundError("V92 candidate is not an exact physical artifact")
    if {item.name for item in root.iterdir()} != {
        WEIGHTS_FILENAME,
        METADATA_FILENAME,
    }:
        raise ValueError("V92 candidate contains unexpected files")
    entries = [
        {
            "path": path.name,
            "sha256": sha256_file_v85(path),
            "size_bytes": path.stat().st_size,
        }
        for path in files
    ]
    return canonical_sha256_v85(entries), entries


def publish_fixed_final_candidate_v92(
    destination: str | Path,
    collection: LoRABankCollection,
    *,
    bindings: Mapping[str, str | int | bool],
    experiment: Mapping[str, Any],
) -> dict[str, Any]:
    """Atomically publish only V92's fresh two-tensor bridge."""

    root = resolve_v85(destination)
    if root.exists() or root.is_symlink():
        raise FileExistsError(f"V92 fixed-final candidate exists: {root}")
    root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{root.name}.", dir=root.parent))
    try:
        weights = temporary / WEIGHTS_FILENAME
        save_file(
            _fresh_state_v92(collection),
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
            "schema_version": 92,
            "status": "fixed_final_awaiting_preregistered_acceptance_gates",
            "bank_name": FRESH_BANK_NAME,
            "target_module": EXPECTED_FRESH_TARGET,
            "rank": 8,
            "alpha": 16.0,
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
            "v91_parent_state_sha256": V91_STATE_SHA256,
            "v91_parent_runtime_promotable": False,
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


def _authenticate_fixed_final_candidate_v92(
    candidate: str | Path,
) -> tuple[dict[str, Any], str, list[dict[str, Any]]]:
    """Authenticate candidate bytes and tensors without loading Gemma."""

    root = resolve_v85(candidate)
    fingerprint, files = _candidate_fingerprint_v92(root)
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
        "v91_parent_state_sha256",
        "v91_parent_runtime_promotable",
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
        or metadata.get("schema_version") != 92
        or metadata.get("status") != "fixed_final_awaiting_preregistered_acceptance_gates"
        or metadata.get("bank_name") != FRESH_BANK_NAME
        or metadata.get("target_module") != EXPECTED_FRESH_TARGET
        or metadata.get("rank") != 8
        or float(metadata.get("alpha", -1.0)) != 16.0
        or float(metadata.get("dropout", -1.0)) != 0.0
        or metadata.get("parameter_count") != FRESH_PARAMETER_COUNT
        or metadata.get("initialization_algorithm") != "cpu_kaiming_uniform_a_exact_zero_b"
        or metadata.get("initialization_seed") != 920092
        or metadata.get("weights_sha256") != sha256_file_v85(weights)
        or metadata.get("frozen_parent_bank_count") != PARENT_BANK_COUNT
        or metadata.get("total_bank_count") != TOTAL_BANK_COUNT
        or metadata.get("frozen_parent_parameter_count") != PARENT_PARAMETER_COUNT
        or metadata.get("total_adapter_parameter_count") != TOTAL_PARAMETER_COUNT
        or metadata.get("v91_parent_state_sha256") != V91_STATE_SHA256
        or metadata.get("v91_parent_runtime_promotable") is not False
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
        raise ValueError("V92 fixed-final candidate authentication failed")
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
        or tuple(tensors["lora_a"].shape) != (8, 4_096)
        or tuple(tensors["lora_b"].shape) != (1_536, 8)
        or any(value.dtype != torch.float32 for value in tensors.values())
        or tensor_state_sha256(state) != metadata.get("state_sha256")
    ):
        raise ValueError("V92 fixed-final candidate tensor contract changed")
    return metadata, fingerprint, files


def load_fixed_final_bridge_v92(
    collection: LoRABankCollection, candidate: str | Path
) -> dict[str, Any]:
    """Authenticate and load the exact two-tensor V92 fixed-final bridge."""

    root = resolve_v85(candidate)
    metadata, _fingerprint, _files = _authenticate_fixed_final_candidate_v92(root)
    _load_fresh_state_v92(
        collection,
        load_file(str(root / WEIGHTS_FILENAME), device="cpu"),
    )
    if collection.bank(FRESH_BANK_NAME).installation.state_sha256() != metadata.get("state_sha256"):
        raise ValueError("V92 fixed-final bridge state changed")
    return metadata


def _expected_candidate_bindings_v92(
    config: Mapping[str, Any], preflight: Mapping[str, str]
) -> dict[str, str | int | bool]:
    v89_fingerprint, _files = checkpoint_fingerprint(
        resolve_v85(config["sources"]["parent_v89_checkpoint"])
    )
    return {
        "config_sha256": preflight["config_sha256"],
        "preregistration_sha256": preflight["preregistration_sha256"],
        "cpu_preflight_sha256": preflight["cpu_preflight_sha256"],
        "v89_checkpoint_sha256": v89_fingerprint,
        "v89_adapter_sha256": str(config["sources"]["parent_v89_adapter_sha256"]),
        "v89_runtime_metadata_sha256": str(config["sources"]["parent_v89_metadata_sha256"]),
        "v91_config_sha256": V91_CONFIG_SHA256,
        "v91_preregistration_sha256": V91_PREREGISTRATION_SHA256,
        "v91_cpu_preflight_sha256": V91_CPU_PREFLIGHT_SHA256,
        "v91_preflight_source_sha256": V91_PREFLIGHT_SOURCE_SHA256,
        "v91_training_source_sha256": V91_TRAINING_SOURCE_SHA256,
        "v91_evaluation_source_sha256": V91_EVALUATION_SOURCE_SHA256,
        "v91_training_report_sha256": V91_TRAINING_REPORT_SHA256,
        "v91_weights_sha256": V91_WEIGHTS_SHA256,
        "v91_metadata_sha256": V91_METADATA_SHA256,
        "v91_state_sha256": V91_STATE_SHA256,
        "v91_checkpoint_sha256": V91_CHECKPOINT_SHA256,
        "v91_predictions_sha256": V91_PREDICTIONS_SHA256,
        "v91_evaluation_sha256": V91_EVALUATION_SHA256,
        "fixed_final_optimizer_updates": EXPECTED_OPTIMIZER_UPDATES,
        "training_schedule_sha256": str(config["dataset"]["training_schedule_sha256"]),
        "training_inventory_sha256": str(config["dataset"]["training_inventory_sha256"]),
        "scene_memory_prefix_sha256": EXPECTED_PREFIX_SHA256,
        "development_known_questions_trained": True,
        "new_held_wordings_excluded_from_optimization": True,
        "failed_v91_parent_runtime_promotable": False,
    }


def authenticate_training_report_v92(
    config: Mapping[str, Any], *, config_path: str | Path = CONFIG
) -> dict[str, str]:
    """Bind V92's fixed report and candidate without loading Gemma."""

    preflight = authenticate_cpu_preflight_v92(config, config_path=config_path)
    v91_evidence = _authenticate_failed_v91(config)
    path = resolve_v85(config["outputs"]["training_report"])
    report = strict_json_v89(path)
    candidate_root = resolve_v85(config["outputs"]["fixed_final_candidate"])
    candidate_metadata, candidate_fingerprint, candidate_files = (
        _authenticate_fixed_final_candidate_v92(candidate_root)
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
        or not isinstance(parent.get("v91"), Mapping)
    ):
        raise TypeError("V92 training report candidate lineage is malformed")
    expected_bindings = _expected_candidate_bindings_v92(config, preflight)
    try:
        expected_candidate_path = candidate_root.relative_to(PROJECT_ROOT).as_posix()
    except ValueError as exc:
        raise ValueError("V92 candidate escaped the project root") from exc
    if (
        report.get("artifact") != TRAINING_ARTIFACT
        or report.get("schema_version") != 92
        or report.get("status") != "fixed_final_training_complete_not_promoted"
        or report.get("config_sha256") != preflight["config_sha256"]
        or report.get("preregistration_sha256") != preflight["preregistration_sha256"]
        or report.get("cpu_preflight_sha256") != preflight["cpu_preflight_sha256"]
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
        or report_candidate.get("weights_sha256") != candidate_metadata["weights_sha256"]
        or report_candidate.get("metadata_canonical_sha256")
        != canonical_sha256_v85(candidate_metadata)
        or report_candidate.get("checkpoint_sha256") != candidate_fingerprint
        or report_candidate.get("checkpoint_files") != candidate_files
        or report_candidate.get("fixed_final") is not True
        or report_candidate.get("runtime_promotion_authorized") is not False
        or bridge.get("bank_name") != FRESH_BANK_NAME
        or bridge.get("target_module") != EXPECTED_FRESH_TARGET
        or bridge.get("rank") != 8
        or float(bridge.get("alpha", -1.0)) != 16.0
        or bridge.get("parameter_count") != FRESH_PARAMETER_COUNT
        or bridge.get("initial_state_sha256") != EXPECTED_FRESH_INITIAL_STATE_SHA256
        or bridge.get("final_state_sha256") != candidate_metadata["state_sha256"]
        or bridge.get("unmerged") is not True
        or parent.get("v89_checkpoint_sha256") != expected_bindings["v89_checkpoint_sha256"]
        or parent.get("v89_adapter_sha256") != config["sources"]["parent_v89_adapter_sha256"]
        or parent.get("v89_runtime_metadata_sha256")
        != config["sources"]["parent_v89_metadata_sha256"]
        or parent["v91"].get("config_sha256") != V91_CONFIG_SHA256
        or parent["v91"].get("weights_sha256") != V91_WEIGHTS_SHA256
        or parent["v91"].get("metadata_sha256") != V91_METADATA_SHA256
        or parent["v91"].get("state_sha256") != V91_STATE_SHA256
        or parent["v91"].get("checkpoint_sha256") != V91_CHECKPOINT_SHA256
        or parent["v91"].get("predictions_sha256") != V91_PREDICTIONS_SHA256
        or parent["v91"].get("evaluation_sha256") != V91_EVALUATION_SHA256
        or parent["v91"].get("failed_but_authenticated") is not True
        or parent["v91"].get("runtime_promotion_authorized") is not False
        or parent.get("frozen_bank_count") != PARENT_BANK_COUNT
        or parent.get("frozen_parameter_count") != PARENT_PARAMETER_COUNT
        or parent.get("parent_tensors_loaded_byte_exactly") is not True
        or parent.get("failed_v91_parent_loaded_unmerged") is not True
        or inventory.get("inventory_sha256") != config["dataset"]["training_inventory_sha256"]
        or inventory.get("schedule_sha256") != config["dataset"]["training_schedule_sha256"]
        or scene_memory.get("prefix_sha256_before") != EXPECTED_PREFIX_SHA256
        or scene_memory.get("prefix_sha256_after") != EXPECTED_PREFIX_SHA256
        or v91_evidence["failed_but_authenticated"] is not True
    ):
        raise ValueError("V92 fixed-final training report changed")
    return {
        **preflight,
        "training_report_sha256": sha256_file_v85(path),
        "candidate_weights_sha256": str(candidate_metadata["weights_sha256"]),
        "candidate_state_sha256": str(candidate_metadata["state_sha256"]),
        "candidate_checkpoint_sha256": candidate_fingerprint,
        "candidate_metadata_canonical_sha256": canonical_sha256_v85(candidate_metadata),
    }


def _answer_tail(language: Any, prepared: Any) -> Any:
    from semantic_3d_chat.language.gemma4_answer_tail import answer_tail_forward

    return answer_tail_forward(language, prepared)


def _empty_mps_cache(device: torch.device) -> None:
    if device.type == "mps" and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def _frozen_state_v92(collection: LoRABankCollection) -> dict[str, str]:
    return {
        bank.settings.name: bank.installation.state_sha256()
        for bank in collection.banks
        if not bank.settings.trainable
    }


def run_training_v92(config_path: str | Path = CONFIG) -> dict[str, Any]:
    """Execute the sealed 1,770-row, 295-update V92 repair schedule."""

    started = time.monotonic()
    config = load_config_v92(config_path, allow_draft=False)
    report_path = resolve_v85(config["outputs"]["training_report"])
    candidate_path = resolve_v85(config["outputs"]["fixed_final_candidate"])
    if report_path.exists() or report_path.is_symlink():
        raise FileExistsError(f"V92 fixed training report exists: {report_path}")
    if candidate_path.exists() or candidate_path.is_symlink():
        raise FileExistsError(f"V92 fixed-final candidate exists: {candidate_path}")

    audit = FileAccessAudit(
        forbidden_component_names=_FORBIDDEN_COMPONENTS,
        block_forbidden=True,
    )
    with audit:
        source_hashes = authenticate_sources_v92(config)
        preflight = authenticate_cpu_preflight_v92(config, config_path=config_path)
        v91_evidence_before_model = _authenticate_failed_v91(config)
        canonical = load_canonical_rows_v92(config)
        items = derive_training_items_v92(config, canonical)
        inventory_hash = canonical_sha256_v85(inventory_v92(items))
        schedule = schedule_v92(
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
            or not v91_evidence_before_model["failed_but_authenticated"]
        ):
            raise RuntimeError("V92 fixed training inventory or schedule changed")

        cpu_memory, memory_hash_before, _memory_metadata = load_scene1_memory_v86(config)
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
            raise RuntimeError("V92 requires local MPS or CPU execution")
        collection = install_lora_banks(
            language.model,
            combined_lora_settings_v92(runtime, config),
        )
        if not isinstance(collection, LoRABankCollection):
            raise TypeError("V92 LoRA bank installation failed")
        parent = load_frozen_parent_v92(
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
            raise RuntimeError("V92 exact adapter parameter surface changed")
        frozen_state_before = _frozen_state_v92(collection)
        if frozen_state_before != parent["frozen_bank_state_sha256"]:
            raise RuntimeError("V92 frozen parent state changed before optimization")

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
            raise RuntimeError("V92 optimizer schedule changed")
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
                raise RuntimeError("V92 answer-only direct-memory layout changed")
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
                    raise RuntimeError("V92 zero-memory answer-only layout changed")
                zero_tail = _answer_tail(language, zero_prepared)
                _unused, observed_margin, penalty = zero_payload_margin_objective_v86(
                    correct_nll,
                    zero_tail.mean_nll.float(),
                    target_margin=target_margin,
                    ce_weight=ce_weight,
                    margin_weight=margin_weight,
                )
                objective = weighted_ce + margin_weight * penalty
                causal_seen += 1
                interval.append(
                    {
                        "correct_nll": float(correct_nll.detach().cpu()),
                        "objective": float(objective.detach().cpu()),
                        "causal": True,
                        "zero_minus_correct_nll": float(observed_margin.detach().cpu()),
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
                raise RuntimeError("V92 objective is nonfinite")
            (objective / accumulation).backward()
            del prepared, tail, correct_nll, weighted_ce, objective
            if cursor % accumulation:
                continue

            gradient_l2 = float(collection.gradient_norms()["total_l2"])
            if not math.isfinite(gradient_l2) or gradient_l2 <= 0.0:
                raise RuntimeError("V92 gradient is zero or nonfinite")
            clipped = torch.nn.utils.clip_grad_norm_(
                parameters,
                float(training["gradient_clip_norm"]),
            )
            clip_l2 = float(clipped.detach().cpu())
            if not math.isfinite(clip_l2):
                raise RuntimeError("V92 clipped gradient is nonfinite")
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_update += 1
            collection.validate_state()
            causal_interval = [record for record in interval if record["causal"]]
            record = {
                "update": optimizer_update,
                "row_cursor": cursor,
                "epoch": epoch,
                "mean_correct_nll": sum(float(value["correct_nll"]) for value in interval)
                / len(interval),
                "mean_objective": sum(float(value["objective"]) for value in interval)
                / len(interval),
                "causal_rows": len(causal_interval),
                "mean_causal_zero_minus_correct_nll": (
                    sum(float(value["zero_minus_correct_nll"]) for value in causal_interval)
                    / len(causal_interval)
                    if causal_interval
                    else None
                ),
                "mean_causal_margin_penalty": (
                    sum(float(value["margin_penalty"]) for value in causal_interval)
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
                            "event": "v92_train_update",
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
                "error_replay": 345,
                "correct_anchor_replay": 345,
                "conversational_known": 312,
                "conversational_error_replay": 180,
                "primary_failed_intent_replay": 144,
                "primary_success_anchor": 30,
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
            raise RuntimeError("V92 fixed retention-repair schedule did not complete")
        language.decoder_module.eval()
        collection.eval()
        memory_hash_after = prefix_sha256(memory.detach().cpu())
        zero_hash_after = prefix_sha256(zero_memory.detach().cpu())
        frozen_state_after = _frozen_state_v92(collection)
        if (
            memory_hash_after != memory_hash_before
            or zero_hash_after != zero_hash_before
            or frozen_state_after != frozen_state_before
        ):
            raise RuntimeError("V92 immutable parent or environmental inputs mutated")

        bindings = _expected_candidate_bindings_v92(config, preflight)
        if (
            bindings["v89_checkpoint_sha256"] != parent["v89_checkpoint_sha256"]
            or bindings["v91_state_sha256"] != parent["v91"]["state_sha256"]
            or bindings["v91_evaluation_sha256"] != parent["v91"]["evaluation_sha256"]
        ):
            raise RuntimeError("V92 candidate parent bindings changed")
        candidate_metadata = publish_fixed_final_candidate_v92(
            candidate_path,
            collection,
            bindings=bindings,
            experiment=config,
        )
        candidate_fingerprint, candidate_files = _candidate_fingerprint_v92(candidate_path)

    audit.assert_clean()
    gates = {
        "all_1770_sealed_micro_rows_consumed": len(schedule) == EXPECTED_MICRO_ROWS,
        "all_138_canonical_rows_consumed_once_each_epoch": kind_seen["canonical"] == 414,
        "all_23_v91_canonical_errors_replayed_five_times_each_epoch": kind_seen["error_replay"]
        == 345,
        "all_115_v91_canonical_correct_anchors_consumed_each_epoch": kind_seen[
            "correct_anchor_replay"
        ]
        == 345,
        "all_104_known_conversational_rows_consumed_each_epoch": kind_seen["conversational_known"]
        == 312,
        "all_6_exact_v91_conversation_errors_replayed_ten_times_each_epoch": kind_seen[
            "conversational_error_replay"
        ]
        == 180,
        "all_24_failed_primary_intent_rows_replayed_twice_each_epoch": kind_seen[
            "primary_failed_intent_replay"
        ]
        == 144,
        "all_10_successful_primary_rows_anchored_each_epoch": kind_seen["primary_success_anchor"]
        == 30,
        "all_39_primary_causal_margin_rows_consumed": causal_seen == EXPECTED_CAUSAL_ROWS,
        "answer_only_ce_on_every_micro_row": answer_only_seen == EXPECTED_MICRO_ROWS,
        "fixed_final_update_295_reached": optimizer_update == EXPECTED_OPTIMIZER_UPDATES,
        "exact_v89_v90_v91_thirteen_bank_parent_frozen": (
            parent["frozen_bank_count"] == PARENT_BANK_COUNT
            and parent["parent_tensors_loaded_byte_exactly"] is True
            and parent["failed_v90_parent_loaded_unmerged"] is True
            and parent["failed_v91_parent_loaded_unmerged"] is True
            and parent["v91"]["failed_but_authenticated"] is True
            and frozen_state_after == frozen_state_before
        ),
        "sole_trainable_surface_is_v92_repair_bridge": (
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
        raise RuntimeError(f"V92 training gate failed: {gates}")

    report = {
        "artifact": TRAINING_ARTIFACT,
        "schema_version": 92,
        "status": "fixed_final_training_complete_not_promoted",
        "config_sha256": preflight["config_sha256"],
        "preregistration_sha256": preflight["preregistration_sha256"],
        "cpu_preflight_sha256": preflight["cpu_preflight_sha256"],
        "device": language.device.type,
        "model_id": config["sources"]["model_id"],
        "model_revision": config["sources"]["model_revision"],
        "strict_input_contract": config["strict_input_contract"],
        "source_hashes": source_hashes,
        "v91_failure_authenticated_before_model_allocation": True,
        "frozen_parent": {
            **parent,
            "frozen_bank_state_before": frozen_state_before,
            "frozen_bank_state_after": frozen_state_after,
            "frozen_bank_state_invariant": frozen_state_after == frozen_state_before,
        },
        "trainable_bridge": {
            "bank_name": FRESH_BANK_NAME,
            "target_module": EXPECTED_FRESH_TARGET,
            "rank": 8,
            "alpha": 16.0,
            "parameter_count": FRESH_PARAMETER_COUNT,
            "initial_state_sha256": initial_state_sha256,
            "final_state_sha256": candidate_metadata["state_sha256"],
            "unmerged": True,
        },
        "training_protocol": config["training"],
        "training_inventory": {
            "canonical_unique_rows": 138,
            "v91_canonical_errors": 23,
            "v91_canonical_error_replays_per_epoch": 115,
            "v91_canonical_correct_anchors_per_epoch": 115,
            "known_conversational_rows_per_epoch": 104,
            "exact_v91_conversation_errors": 6,
            "v91_conversation_error_replays_per_epoch": 60,
            "failed_primary_intent_known_rows": 24,
            "failed_primary_intent_replays_per_epoch": 48,
            "successful_primary_anchors_per_epoch": 10,
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
            "metadata_canonical_sha256": canonical_sha256_v85(candidate_metadata),
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
    report = run_training_v92(args.config)
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
    "authenticate_training_report_v92",
    "combined_lora_settings_v92",
    "load_fixed_final_bridge_v92",
    "load_frozen_parent_v92",
    "main",
    "publish_fixed_final_candidate_v92",
    "run_training_v92",
]
