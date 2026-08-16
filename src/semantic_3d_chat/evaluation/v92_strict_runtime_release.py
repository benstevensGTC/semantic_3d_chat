"""Fail-closed strict-runtime packaging scaffold for V92.

This offline module authenticates the sealed V92 experiment, passing fixed
evaluation, exact V89 runtime parent, exact failed V90/V91 development
bridges, and exact V92 bridge before any runtime file can be created.  Chat
never imports this module.

The isolated smoke child receives only a sanitized runtime YAML, frozen
fourteen-bank checkpoint, immutable continuous scene-memory artifact, and
thirteen questions.  Expected answers remain in this parent process and are
applied only after the child exits.  ``data/oracle`` is physically renamed for
the complete child lifetime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import torch
import yaml
from safetensors.torch import load_file, save_file

from semantic_3d_chat.chat.runtime_config import (
    effective_runtime_config_sha256,
    load_runtime_config,
)
from semantic_3d_chat.chat.v92_strict_scene1_runtime import (
    EXPECTED_ADAPTER_PARAMETER_COUNT,
    EXPECTED_BANKS,
    PROMOTION_DECISION,
    V90_BANK,
    V90_STATE_SHA256,
    V90_TARGET,
    V91_BANK,
    V91_STATE_SHA256,
    V91_TARGET,
    V92_BANK,
    V92_TARGET,
    validate_v92_runtime_contract,
)
from semantic_3d_chat.config import PROJECT_ROOT, config_hash
from semantic_3d_chat.evaluation.prediction_artifacts import checkpoint_fingerprint
from semantic_3d_chat.evaluation.strict_direct_release_core import (
    BridgeSourceContract,
    compose_exact_bank_archive,
    extend_runtime_lora_config,
    extend_runtime_metadata,
    load_bridge_source,
    sha256_file,
    validate_runtime_bank_inventory,
)
from semantic_3d_chat.evaluation.v92_scene1_retention_conversation_preflight import (
    authenticate_sources_v92,
    load_config_v92,
    primary_rows_v92,
)
from semantic_3d_chat.language.lora import tensor_state_sha256
from semantic_3d_chat.scene_encoder.v81_scene_memory_artifact import (
    MEMORY_FILENAME,
    METADATA_FILENAME,
    load_v81_scene_memory,
)
from semantic_3d_chat.training.checkpointing import (
    RUNTIME_METADATA_FILENAME,
    validate_runtime_checkpoint_metadata,
)

SCHEMA_VERSION: Final[int] = 92
SCENE_ID: Final[str] = "scene_000001"
_SHA256: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{64}")

EXPERIMENT_CONFIG_SHA256: Final[str] = (
    "cc05107f4bec837f78d7b50d8467819faa3dcb0a9595929b79ca09a496618915"
)
PREREGISTRATION_SHA256: Final[str] = (
    "acf0ece8cfc6a2e0c812810d66f31e940fac530a99fb9ca91ccd98b40570840a"
)
CPU_PREFLIGHT_SHA256: Final[str] = (
    "dbc702b1c7c3a7ae42ff22dfc58d44109b5a4b318eacd5ac297838206637f467"
)
SOURCE_MEMORY_PREFIX_SHA256: Final[str] = (
    "a428f5147c815839ae7315a0adab952ab210814fb21dcdc5bf13b167f28a6e37"
)
SOURCE_MEMORY_TENSOR_FILE_SHA256: Final[str] = (
    "3427851432d8f2a3609b6205b18b8c0d9a0fcf68d8f3bf0c98c758ac64209ffb"
)

PARENT_BANKS: Final[tuple[str, ...]] = EXPECTED_BANKS[:11]
V90_WEIGHTS_SHA256: Final[str] = (
    "be8a6fa9b633dc52ca962393170623c2707e339206ced627e6439ce7db0a7f94"
)
V90_METADATA_SHA256: Final[str] = (
    "c2dba75094c829c7fadaadf8111a712921f70372ac84e2073714530519349acc"
)
V91_WEIGHTS_SHA256: Final[str] = (
    "1cccb4e25d11ce88e881d9b7df2ba4c521aceef55a7b6617a1172ee367f782d0"
)
V91_METADATA_SHA256: Final[str] = (
    "9cd41f9d3bec7f6189e2613d59dccd9e23db213434443528b89743430dfc3ec7"
)

_BASE_MODEL_GATES: Final[frozenset[str]] = frozenset(
    {
        "canonical_correct_at_least_preregistered_minimum",
        "canonical_presence_correct_at_least_minimum",
        "canonical_count_correct_at_least_minimum",
        "canonical_metric_correct_at_least_minimum",
        "canonical_attribute_correct_at_least_minimum",
        "canonical_spatial_correct_at_least_minimum",
        "canonical_support_correct_at_least_minimum",
        "primary_conversational_correct_at_least_required",
        "all_six_core_actionable_intents_correct",
        "new_held_wording_correct_at_least_required",
        "new_held_wording_each_intent_at_least_minimum",
        "causal_correct_memory_mean_nll_at_least_preregistered_margin_below_zero_payload",
        "causal_prediction_changes_at_least_required",
        "exact_prefix_hash_invariance",
        "exact_total_environment_input_invariance",
        "frozen_parent_state_invariance",
        "protected_read_count_at_most_preregistered_maximum",
    }
)
_REQUIRED_MODEL_GATES: Final[frozenset[str]] = frozenset(
    {*_BASE_MODEL_GATES, "fixed_final_candidate_state_invariance"}
)
_REQUIRED_RUNTIME_GATES: Final[frozenset[str]] = frozenset(
    {
        "model_acceptance_gate_authenticated_and_passed",
        "runtime_process_exit_zero",
        "at_least_twelve_of_thirteen_behavior_assertions_pass",
        "all_six_core_actionable_intents_pass",
        "oracle_physically_unavailable",
        "oracle_restored_after_runtime",
        "child_audit_completion_passed",
        "child_used_exact_fourteen_frozen_banks",
        "child_parameter_inventory_exact",
        "file_audit_forbidden_read_count_zero",
        "file_audit_protected_read_count_zero",
        "prefix_hash_identical_for_every_question",
        "total_environment_conditioned_input_identical",
        "prefix_and_environment_input_identical",
        "expected_immutable_scene_prefix",
        "exact_direct_memory_layout_every_question",
        "source_memory_bytes_unchanged",
        "expectations_absent_from_child_protocol",
    }
)
CORE_ACTIONABLE_INTENTS: Final[frozenset[str]] = frozenset(
    {"table_contents", "under_table", "wall_object", "cube_location", "sitting", "bowl_contents"}
)

EXPERIMENT_CONFIG: Final[Path] = PROJECT_ROOT / (
    "configs/experiments/gemma4_v92_scene1_retention_conversation_repair.yaml"
)
PREREGISTRATION: Final[Path] = PROJECT_ROOT / (
    "reports/gemma4/metrics/gemma4_v92_scene1_retention_conversation_repair_preregistration.json"
)
CPU_PREFLIGHT: Final[Path] = PROJECT_ROOT / (
    "reports/gemma4/metrics/gemma4_v92_scene1_retention_conversation_repair_cpu_preflight.json"
)
TRAINING_REPORT: Final[Path] = PROJECT_ROOT / (
    "reports/gemma4/metrics/gemma4_v92_scene1_retention_conversation_repair_training.json"
)
PREDICTIONS: Final[Path] = PROJECT_ROOT / (
    "reports/gemma4/predictions/gemma4_v92_scene1_retention_conversation_repair_evaluation.json"
)
MODEL_GATE_REPORT: Final[Path] = PROJECT_ROOT / (
    "reports/gemma4/metrics/gemma4_v92_scene1_retention_conversation_repair_evaluation.json"
)
PARENT_RUNTIME_CONFIG: Final[Path] = PROJECT_ROOT / "configs/runtime/gemma4_v89_strict_scene1.yaml"
PARENT_CHECKPOINT: Final[Path] = PROJECT_ROOT / (
    "data_gemma4/runtime/checkpoints/gemma4_v89_strict_scene1_release_v1"
)
PARENT_RELEASE_REPORT: Final[Path] = PROJECT_ROOT / (
    "reports/gemma4/metrics/gemma4_v89_strict_runtime_release.json"
)
V90_BRIDGE_CANDIDATE: Final[Path] = PROJECT_ROOT / (
    "reports/gemma4/artifacts/v90_scene1_conversational_final"
)
V91_BRIDGE_CANDIDATE: Final[Path] = PROJECT_ROOT / (
    "reports/gemma4/artifacts/v91_scene1_conversational_repair_final_v2"
)
V92_BRIDGE_CANDIDATE: Final[Path] = PROJECT_ROOT / (
    "reports/gemma4/artifacts/v92_scene1_retention_conversation_repair_final"
)
SOURCE_MEMORY: Final[Path] = PROJECT_ROOT / (
    "data_gemma4/runtime/scene_memories/v89/scene_000001"
)

RUNTIME_CONFIG: Final[Path] = PROJECT_ROOT / "configs/runtime/gemma4_v92_strict_scene1.yaml"
CANDIDATE_CHECKPOINT: Final[Path] = PROJECT_ROOT / (
    "reports/gemma4/artifacts/v92_scene1_retention_conversation_repair_runtime_v1"
)
CANDIDATE_MEMORY: Final[Path] = PROJECT_ROOT / (
    "reports/gemma4/artifacts/v92_scene1_retention_conversation_repair_runtime_memory_v1/scene_000001"
)
RELEASE_CHECKPOINT: Final[Path] = PROJECT_ROOT / (
    "data_gemma4/runtime/checkpoints/gemma4_v92_strict_scene1_release_v1"
)
RELEASE_MEMORY: Final[Path] = PROJECT_ROOT / (
    "data_gemma4/runtime/scene_memories/v92/scene_000001"
)
SMOKE_CHAT: Final[Path] = PROJECT_ROOT / "reports/gemma4/examples/v92_strict_runtime_smoke.jsonl"
SMOKE_AUDIT: Final[Path] = PROJECT_ROOT / (
    "reports/gemma4/metrics/v92_strict_runtime_smoke_access.json"
)
SMOKE_REPORT: Final[Path] = PROJECT_ROOT / (
    "reports/gemma4/metrics/gemma4_v92_scene1_retention_conversation_repair_runtime_smoke.json"
)
RELEASE_REPORT: Final[Path] = PROJECT_ROOT / (
    "reports/gemma4/metrics/gemma4_v92_strict_runtime_release.json"
)


def _read_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(path)

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON field in {path}: {key}")
            result[key] = value
        return result

    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates)
    if not isinstance(value, dict):
        raise TypeError(f"Expected one JSON object: {path}")
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_hash(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"V92 {field} is not a lowercase SHA-256 digest")
    return value


def _relative(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def _load_experiment() -> dict[str, Any]:
    if sha256_file(EXPERIMENT_CONFIG) != EXPERIMENT_CONFIG_SHA256:
        raise ValueError("V92 sealed experiment config changed")
    raw = EXPERIMENT_CONFIG.read_text(encoding="utf-8")
    if "REPLACE_" in raw or "TO_FILL" in raw:
        raise ValueError("V92 sealed experiment config contains a placeholder")
    payload = yaml.safe_load(raw)
    if not isinstance(payload, dict) or set(payload) != {"v92"}:
        raise ValueError("V92 sealed experiment identity changed")
    experiment = payload["v92"]
    bridge = experiment.get("bridge") if isinstance(experiment, dict) else None
    scope = experiment.get("scope") if isinstance(experiment, dict) else None
    if (
        not isinstance(experiment, dict)
        or experiment.get("schema_version") != 92
        or experiment.get("artifact")
        != "gemma4_v92_scene1_retention_conversation_repair_direct_memory_v1"
        or experiment.get("status") != "sealed_before_full_model_load"
        or not isinstance(bridge, dict)
        or bridge.get("bank_name") != V92_BANK
        or bridge.get("target_module") != V92_TARGET
        or bridge.get("rank") != 8
        or float(bridge.get("alpha", -1.0)) != 16.0
        or bridge.get("trainable_parameter_count") != 45_056
        or not isinstance(scope, dict)
        or scope.get("post_v91_training_set_development") is not True
        or scope.get("exact_failed_v90_and_v91_candidates_frozen") is not True
        or scope.get("held_out_scene_generalization_claim") is not False
        or scope.get("runtime_promotion_authorized") is not False
    ):
        raise ValueError("V92 sealed experiment contract changed")
    return experiment


def _authenticate_preflight(experiment: Mapping[str, Any]) -> dict[str, str]:
    for path, expected in (
        (PREREGISTRATION, PREREGISTRATION_SHA256),
        (CPU_PREFLIGHT, CPU_PREFLIGHT_SHA256),
    ):
        if path.is_symlink() or not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"V92 fixed pre-model evidence changed: {_relative(path)}")
    config = load_config_v92(EXPERIMENT_CONFIG, allow_draft=False)
    if config != experiment:
        raise ValueError("V92 release and preflight parsed different configs")
    from semantic_3d_chat.evaluation.v92_scene1_retention_conversation_preflight import (
        authenticate_cpu_preflight_v92,
    )

    bindings = authenticate_cpu_preflight_v92(config, config_path=EXPERIMENT_CONFIG)
    if (
        bindings.get("config_sha256") != EXPERIMENT_CONFIG_SHA256
        or bindings.get("preregistration_sha256") != PREREGISTRATION_SHA256
        or bindings.get("cpu_preflight_sha256") != CPU_PREFLIGHT_SHA256
    ):
        raise ValueError("V92 pre-model evidence bindings changed")
    return bindings


def _bucket(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"V92 metric bucket is malformed: {label}")
    return value


def validate_model_gate_contract_v92(
    report: Mapping[str, Any], experiment: Mapping[str, Any] | None = None
) -> None:
    """Fail closed unless every sealed V92 model gate passed exactly."""

    experiment = _load_experiment() if experiment is None else experiment
    thresholds = _bucket(experiment.get("gates"), "preregistered gates")
    metrics = _bucket(report.get("metrics"), "metrics")
    gates = _bucket(metrics.get("model_acceptance_gates"), "model gates")
    canonical = _bucket(metrics.get("canonical_type_specific"), "canonical")
    by_type = _bucket(metrics.get("canonical_accuracy_by_answer_type"), "canonical type")
    primary = _bucket(metrics.get("primary_conversational"), "primary")
    primary_by_intent = _bucket(primary.get("by_intent"), "primary intents")
    held = _bucket(metrics.get("new_held_wording"), "held wording")
    held_by_intent = _bucket(held.get("by_intent"), "held intents")
    causal = _bucket(metrics.get("causal_control"), "causal")
    intent_ids = {str(row["id"]) for row in experiment["conversational_intents"]}
    type_minima = {
        "presence": int(thresholds["canonical_presence_correct_minimum"]),
        "count": int(thresholds["canonical_count_correct_minimum"]),
        "metric": int(thresholds["canonical_metric_correct_minimum"]),
        "attribute": int(thresholds["canonical_attribute_correct_minimum"]),
        "spatial_relation": int(thresholds["canonical_spatial_correct_minimum"]),
        "support": int(thresholds["canonical_support_correct_minimum"]),
    }
    type_pass = all(
        int(_bucket(by_type.get(name), name).get("correct", -1)) >= minimum
        for name, minimum in type_minima.items()
    )
    intents_pass = (
        set(primary_by_intent) == intent_ids
        and all(_bucket(primary_by_intent[name], name).get("total") == 1 for name in intent_ids)
        and all(
            _bucket(primary_by_intent[name], name).get("correct") == 1
            for name in CORE_ACTIONABLE_INTENTS
        )
        and set(held_by_intent) == intent_ids
        and all(
            _bucket(held_by_intent[name], name).get("total") == 2
            and int(_bucket(held_by_intent[name], name).get("correct", -1))
            >= int(thresholds["new_held_wording_each_intent_minimum"])
            for name in intent_ids
        )
    )
    if (
        report.get("artifact")
        != "gemma4_v92_scene1_retention_conversation_repair_evaluation_v1"
        or report.get("schema_version") != 92
        or report.get("status") != "model_gates_pass_separate_runtime_packaging_required"
        or set(gates) != _REQUIRED_MODEL_GATES
        or any(value is not True for value in gates.values())
        or metrics.get("model_acceptance_gate_passed") is not True
        or metrics.get("separate_runtime_packaging_authorized") is not True
        or metrics.get("runtime_oracle_unavailable_gate_pending") is not True
        or metrics.get("runtime_file_audit_gate_pending") is not True
        or metrics.get("automatic_runtime_promotion") is not False
        or metrics.get("runtime_promotion_authorized") is not False
        or report.get("separate_runtime_packaging_authorized") is not True
        or report.get("runtime_oracle_unavailable_gate_pending") is not True
        or report.get("runtime_file_audit_gate_pending") is not True
        or report.get("automatic_runtime_promotion") is not False
        or report.get("runtime_promotion_authorized") is not False
        or report.get("fixed_checkpoint_selected_before_scoring") is not True
        or report.get("checkpoint_selection_after_scoring") is not False
        or report.get("post_v91_training_set_development") is not True
        or report.get("single_scene_retention_conversation_repair") is not True
        or report.get("development_known_primary_questions") is not True
        or report.get("newly_held_wording_only") is not True
        or report.get("held_out_scene") is not False
        or report.get("held_out_scene_generalization_claim") is not False
        or report.get("frozen_thirteen_bank_parent_mutated") is not False
        or report.get("fixed_final_candidate_state_invariant") is not True
        or report.get("oracle_loaded") is not False
        or report.get("official_validation_loaded") is not False
        or report.get("official_test_loaded") is not False
        or report.get("deferred_final_loaded") is not False
        or canonical.get("total") != int(thresholds["canonical_total"])
        or int(canonical.get("correct", -1)) < int(thresholds["canonical_correct_minimum"])
        or not type_pass
        or primary.get("total") != int(thresholds["primary_conversational_total"])
        or int(primary.get("correct", -1))
        < int(thresholds["primary_conversational_required_correct"])
        or primary.get("core_actionable_correct")
        != int(thresholds["core_actionable_required_correct"])
        or not intents_pass
        or held.get("total") != int(thresholds["new_held_wording_total"])
        or int(held.get("correct", -1))
        < int(thresholds["new_held_wording_required_correct"])
        or causal.get("row_count") != 13
        or float(causal.get("mean_zero_minus_correct_nll", -1.0))
        < float(thresholds["causal_mean_zero_minus_correct_nll_minimum"])
        or int(causal.get("canonical_prediction_changes", -1))
        < int(thresholds["causal_prediction_change_minimum"])
    ):
        raise ValueError("V92 model-level acceptance report did not pass exactly")

    memory = _bucket(report.get("scene_memory"), "scene memory")
    leakage = _bucket(report.get("leakage"), "leakage")
    if (
        memory.get("compiled_before_question_tokenization") is not True
        or memory.get("shape") != [1, 738, 1536]
        or memory.get("continuous_environment_payload_tokens") != 736
        or memory.get("prefix_sha256_before") != SOURCE_MEMORY_PREFIX_SHA256
        or memory.get("prefix_sha256_after") != SOURCE_MEMORY_PREFIX_SHA256
        or memory.get("prefix_hash_invariant") is not True
        or memory.get("environment_conditioned_input_invariant") is not True
        or memory.get("same_exact_memory_reused_for_all_177_questions") is not True
        or memory.get("question_derived_environmental_tokens") != 0
        or memory.get("question_conditioned_environmental_readout") is not False
        or memory.get("question_dependent_scene_processing") is not False
        or memory.get("question_dependent_retrieval") is not False
        or memory.get("control_tokens") != 0
        or memory.get("environmental_text_inputs") != []
        or leakage.get("protected_read_count") != 0
        or leakage.get("protected_reads") != []
        or leakage.get("oracle_loaded") is not False
    ):
        raise ValueError("V92 model leakage or immutable-memory evidence changed")


def _fixed_contracts() -> tuple[BridgeSourceContract, BridgeSourceContract]:
    contracts = (
        BridgeSourceContract(
            root=V90_BRIDGE_CANDIDATE,
            artifact="gemma4_v90_scene1_conversational_fixed_final_v1",
            bank_name=V90_BANK,
            target_module=V90_TARGET,
            rank=8,
            alpha=16.0,
            dropout=0.0,
            parameter_count=28_672,
            state_sha256=V90_STATE_SHA256,
            weights_sha256=V90_WEIGHTS_SHA256,
            metadata_sha256=V90_METADATA_SHA256,
        ),
        BridgeSourceContract(
            root=V91_BRIDGE_CANDIDATE,
            artifact="gemma4_v91_scene1_conversational_repair_fixed_final_v1",
            bank_name=V91_BANK,
            target_module=V91_TARGET,
            rank=16,
            alpha=32.0,
            dropout=0.0,
            parameter_count=221_184,
            state_sha256=V91_STATE_SHA256,
            weights_sha256=V91_WEIGHTS_SHA256,
            metadata_sha256=V91_METADATA_SHA256,
        ),
    )
    v90, v91 = (load_bridge_source(contract) for contract in contracts)
    if (
        tuple(v90.state["adapters.0.lora_a"].shape) != (8, 2_048)
        or tuple(v90.state["adapters.0.lora_b"].shape) != (1_536, 8)
        or tuple(v91.state["adapters.0.lora_a"].shape) != (16, 12_288)
        or tuple(v91.state["adapters.0.lora_b"].shape) != (1_536, 16)
    ):
        raise ValueError("V92 frozen V90/V91 bridge topology changed")
    return contracts


def _v92_contract(evidence: Mapping[str, Any]) -> BridgeSourceContract:
    contract = BridgeSourceContract(
        root=V92_BRIDGE_CANDIDATE,
        artifact="gemma4_v92_scene1_retention_conversation_repair_fixed_final_v1",
        bank_name=V92_BANK,
        target_module=V92_TARGET,
        rank=8,
        alpha=16.0,
        dropout=0.0,
        parameter_count=45_056,
        state_sha256=_require_hash(
            evidence.get("v92_bridge_state_sha256"), "V92 bridge state"
        ),
        weights_sha256=_require_hash(
            evidence.get("v92_bridge_file_sha256"), "V92 bridge file"
        ),
        metadata_sha256=_require_hash(
            evidence.get("v92_bridge_metadata_sha256"), "V92 bridge metadata"
        ),
    )
    loaded = load_bridge_source(contract)
    if (
        tuple(loaded.state["adapters.0.lora_a"].shape) != (8, 4_096)
        or tuple(loaded.state["adapters.0.lora_b"].shape) != (1_536, 8)
    ):
        raise ValueError("V92 bridge topology changed")
    return contract


def _contracts(evidence: Mapping[str, Any]) -> tuple[BridgeSourceContract, ...]:
    return (*_fixed_contracts(), _v92_contract(evidence))


def authenticate_v92_model_gate() -> dict[str, Any]:
    """Authenticate passing V92 evidence and all three added bridge banks."""

    from semantic_3d_chat.training.train_v92_scene1_retention_conversation_repair import (
        authenticate_training_report_v92,
    )

    experiment = _load_experiment()
    _authenticate_preflight(experiment)
    bindings = authenticate_training_report_v92(
        experiment, config_path=EXPERIMENT_CONFIG
    )
    training = _read_json(TRAINING_REPORT)
    predictions = _read_json(PREDICTIONS)
    report = _read_json(MODEL_GATE_REPORT)
    validate_model_gate_contract_v92(report, experiment)
    sources = authenticate_sources_v92(experiment)
    if report.get("source_hashes") != sources or training.get("source_hashes") != sources:
        raise ValueError("V92 training/evaluation source inventory changed")
    training_sha = sha256_file(TRAINING_REPORT)
    predictions_sha = sha256_file(PREDICTIONS)
    candidate_metadata = _read_json(V92_BRIDGE_CANDIDATE / RUNTIME_METADATA_FILENAME)
    candidate_metadata_sha = sha256_file(V92_BRIDGE_CANDIDATE / RUNTIME_METADATA_FILENAME)
    if (
        bindings.get("config_sha256") != EXPERIMENT_CONFIG_SHA256
        or bindings.get("preregistration_sha256") != PREREGISTRATION_SHA256
        or bindings.get("cpu_preflight_sha256") != CPU_PREFLIGHT_SHA256
        or bindings.get("training_report_sha256") != training_sha
        or report.get("config_sha256") != EXPERIMENT_CONFIG_SHA256
        or report.get("preregistration_sha256") != PREREGISTRATION_SHA256
        or report.get("cpu_preflight_sha256") != CPU_PREFLIGHT_SHA256
        or report.get("training_report_sha256") != training_sha
        or report.get("evaluation_predictions_sha256") != predictions_sha
        or report.get("evaluation_predictions_path") != _relative(PREDICTIONS)
        or predictions.get("artifact")
        != "gemma4_v92_scene1_retention_conversation_repair_predictions_v1"
        or predictions.get("schema_version") != 92
        or predictions.get("status") != "fixed_final_evaluation_only_not_runtime"
        or predictions.get("config_sha256") != EXPERIMENT_CONFIG_SHA256
        or predictions.get("training_report_sha256") != training_sha
        or predictions.get("scene_id") != SCENE_ID
        or predictions.get("scene_count") != 1
        or predictions.get("canonical_row_count") != 138
        or predictions.get("primary_conversational_row_count") != 13
        or predictions.get("new_v92_held_wording_row_count") != 26
        or predictions.get("causal_control_row_count") != 13
        or predictions.get("fixed_checkpoint_selected_before_scoring") is not True
        or predictions.get("checkpoint_selection_after_scoring") is not False
        or predictions.get("frozen_parent_bank_count") != 13
        or predictions.get("frozen_parent_state_invariant") is not True
        or predictions.get("candidate_state_invariant") is not True
        or predictions.get("questions_or_answers_serialized_in_runtime_candidate") is not False
        or predictions.get("training_inventory_serialized_in_runtime_candidate") is not False
        or predictions.get("oracle_serialized_in_runtime_candidate") is not False
        or predictions.get("runtime_promotion_authorized") is not False
        or predictions.get("scene_memory") != report.get("scene_memory")
        or predictions.get("leakage") != report.get("leakage")
        or candidate_metadata.get("state_sha256")
        != bindings.get("candidate_state_sha256")
        or candidate_metadata.get("weights_sha256")
        != bindings.get("candidate_weights_sha256")
    ):
        raise ValueError("V92 fixed predictions, candidate, or hash binding changed")
    records = (
        predictions.get("canonical_records"),
        predictions.get("primary_conversational_records"),
        predictions.get("new_v92_held_wording_records"),
        predictions.get("causal_records"),
    )
    if any(not isinstance(rows, list) for rows in records) or tuple(
        len(rows) for rows in records
    ) != (138, 13, 26, 13):
        raise ValueError("V92 fixed prediction record inventory changed")

    parent_release = _read_json(PARENT_RELEASE_REPORT)
    parent_fingerprint, parent_files = checkpoint_fingerprint(PARENT_CHECKPOINT)
    parent_checkpoint = parent_release.get("checkpoint")
    if (
        parent_release.get("artifact") != "gemma4_v89_strict_runtime_release_v1"
        or parent_release.get("schema_version") != 89
        or parent_release.get("all_release_gates_passed") is not True
        or not isinstance(parent_checkpoint, Mapping)
        or parent_checkpoint.get("checkpoint_sha256") != parent_fingerprint
        or parent_checkpoint.get("checkpoint_files") != parent_files
        or sha256_file(PARENT_RUNTIME_CONFIG) != experiment["sources"]["runtime_config_sha256"]
        or sha256_file(PARENT_CHECKPOINT / "adapter.safetensors")
        != experiment["sources"]["parent_v89_adapter_sha256"]
        or sha256_file(PARENT_CHECKPOINT / RUNTIME_METADATA_FILENAME)
        != experiment["sources"]["parent_v89_metadata_sha256"]
    ):
        raise ValueError("V92 promoted V89 parent binding changed")

    fixed = _fixed_contracts()
    evidence = {
        "schema_version": SCHEMA_VERSION,
        "experiment_config_sha256": EXPERIMENT_CONFIG_SHA256,
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "cpu_preflight_sha256": CPU_PREFLIGHT_SHA256,
        "training_report_sha256": training_sha,
        "evaluation_predictions_sha256": predictions_sha,
        "model_gate_report_sha256": sha256_file(MODEL_GATE_REPORT),
        "parent_release_report_sha256": sha256_file(PARENT_RELEASE_REPORT),
        "parent_checkpoint_sha256": parent_fingerprint,
        "v90_bridge_file_sha256": fixed[0].weights_sha256,
        "v90_bridge_metadata_sha256": fixed[0].metadata_sha256,
        "v90_bridge_state_sha256": fixed[0].state_sha256,
        "v91_bridge_file_sha256": fixed[1].weights_sha256,
        "v91_bridge_metadata_sha256": fixed[1].metadata_sha256,
        "v91_bridge_state_sha256": fixed[1].state_sha256,
        "v92_bridge_file_sha256": str(bindings["candidate_weights_sha256"]),
        "v92_bridge_metadata_sha256": candidate_metadata_sha,
        "v92_bridge_state_sha256": str(bindings["candidate_state_sha256"]),
        "model_acceptance_gate_passed": True,
        "model_gate_report_authenticated": True,
        "held_out_scene": False,
        "held_out_generalization_claim": False,
    }
    _v92_contract(evidence)
    return evidence


def build_runtime_config_payload(evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Build sanitized exact V89+V90+V91+V92 runtime configuration."""

    parent = load_runtime_config(PARENT_RUNTIME_CONFIG)
    parent.pop("_config_path", None)
    configured = parent.get("language", {}).get("lora_banks")
    if not isinstance(configured, dict) or tuple(configured) != PARENT_BANKS:
        raise ValueError("V92 authenticated V89 parent bank inventory changed")
    payload = extend_runtime_lora_config(
        parent_runtime_config=parent,
        added_bridges=_contracts(evidence),
        expected_final_banks=EXPECTED_BANKS,
    )
    banks = payload["language"]["lora_banks"]
    if (
        tuple(banks) != EXPECTED_BANKS
        or len(banks) != 14
        or any(row.get("trainable") is not False for row in banks.values())
    ):
        raise RuntimeError("V92 runtime payload lost exact frozen fourteen-bank order")
    return payload


def materialize_runtime_config(evidence: Mapping[str, Any]) -> dict[str, Any]:
    authenticated = authenticate_v92_model_gate()
    if dict(evidence) != authenticated:
        raise ValueError("V92 runtime-config evidence is not current")
    payload = build_runtime_config_payload(authenticated)
    encoded = yaml.safe_dump(payload, sort_keys=False, allow_unicode=False)
    if RUNTIME_CONFIG.exists():
        if RUNTIME_CONFIG.is_symlink() or RUNTIME_CONFIG.read_text(encoding="utf-8") != encoded:
            raise ValueError("Existing V92 runtime config differs from authenticated gate")
    else:
        RUNTIME_CONFIG.parent.mkdir(parents=True, exist_ok=True)
        with RUNTIME_CONFIG.open("x", encoding="utf-8") as handle:
            handle.write(encoded)
    return load_runtime_config(RUNTIME_CONFIG)


def _composed_adapter(
    evidence: Mapping[str, Any],
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    return compose_exact_bank_archive(
        base_checkpoint=PARENT_CHECKPOINT,
        expected_base_banks=PARENT_BANKS,
        added_bridges=_contracts(evidence),
        expected_final_banks=EXPECTED_BANKS,
    )


def _source_stack_sha256(tensors: Mapping[str, torch.Tensor]) -> str:
    source = {
        name: value
        for name, value in tensors.items()
        if not name.startswith("block_cross_residual.")
    }
    if not source or len(source) >= len(tensors):
        raise RuntimeError("V92 frozen source-stack inventory is invalid")
    return tensor_state_sha256(source)


def build_runtime_metadata(
    evidence: Mapping[str, Any],
    *,
    promotion: str,
    smoke_report_sha256: str | None,
) -> dict[str, Any]:
    allowed = {"pending_isolated_runtime_smoke", PROMOTION_DECISION}
    if promotion not in allowed:
        raise ValueError("Unknown V92 runtime promotion state")
    promoted = promotion == PROMOTION_DECISION
    if promoted != (smoke_report_sha256 is not None):
        raise ValueError("V92 promotion and smoke binding disagree")
    config = build_runtime_config_payload(evidence)
    parent = _read_json(PARENT_CHECKPOINT / RUNTIME_METADATA_FILENAME)
    metadata = extend_runtime_metadata(
        parent_metadata=parent,
        added_bridges=_contracts(evidence),
        expected_final_banks=EXPECTED_BANKS,
    )
    states = metadata["lora_bank_state_sha256"]
    for row in metadata["lora"]["banks"]:
        row["expected_initial_state_sha256"] = states[str(row["name"])]
    metadata["config_hash"] = config_hash(config)
    tensors, _ = _composed_adapter(evidence)
    metadata["frozen_block_cross_source_stack_state_sha256"] = _source_stack_sha256(tensors)
    provenance = dict(metadata.get("initialization_provenance", {}))
    provenance["v92_strict_runtime_release"] = {
        "schema_version": 92,
        "experiment_config_sha256": evidence["experiment_config_sha256"],
        "preregistration_sha256": evidence["preregistration_sha256"],
        "cpu_preflight_sha256": evidence["cpu_preflight_sha256"],
        "training_report_sha256": evidence["training_report_sha256"],
        "evaluation_predictions_sha256": evidence["evaluation_predictions_sha256"],
        "model_gate_report_sha256": evidence["model_gate_report_sha256"],
        "v90_bridge_state_sha256": evidence["v90_bridge_state_sha256"],
        "v91_bridge_state_sha256": evidence["v91_bridge_state_sha256"],
        "v92_bridge_state_sha256": evidence["v92_bridge_state_sha256"],
        "model_acceptance_gate_passed": True,
        "model_gate_report_authenticated": True,
        "promotion_decision": promotion,
        "runtime_promotion_authorized": promoted,
        "smoke_report_sha256": smoke_report_sha256,
        "held_out_scene": False,
        "held_out_generalization_claim": False,
    }
    metadata["initialization_provenance"] = provenance
    validate_runtime_checkpoint_metadata(metadata)
    validate_runtime_bank_inventory(
        runtime_config=config,
        checkpoint_metadata=metadata,
        expected_bank_order=EXPECTED_BANKS,
        expected_states={str(name): str(state) for name, state in states.items()},
    )
    if metadata["lora_parameter_count"] != EXPECTED_ADAPTER_PARAMETER_COUNT:
        raise RuntimeError("V92 runtime parameter inventory changed")
    if promoted:
        contract = validate_v92_runtime_contract(
            scene_id=SCENE_ID,
            runtime_config=config,
            checkpoint_metadata=metadata,
        )
        if contract["v92_bridge_state_sha256"] != evidence["v92_bridge_state_sha256"]:
            raise RuntimeError("V92 promoted metadata failed runtime-only validation")
    return metadata


def _atomic_checkpoint(
    destination: Path,
    *,
    metadata: Mapping[str, Any],
    evidence: Mapping[str, Any],
    source_adapter: Path | None = None,
) -> dict[str, Any]:
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    try:
        if source_adapter is None:
            tensors, inheritance = _composed_adapter(evidence)
            save_file(tensors, str(temporary / "adapter.safetensors"))
        else:
            if source_adapter.is_symlink() or not source_adapter.is_file():
                raise FileNotFoundError(source_adapter)
            shutil.copyfile(source_adapter, temporary / "adapter.safetensors")
            inheritance = {"candidate_adapter_bytes_reused_exactly": True}
        _write_json(temporary / RUNTIME_METADATA_FILENAME, metadata)
        if {item.name for item in temporary.iterdir()} != {
            "adapter.safetensors",
            RUNTIME_METADATA_FILENAME,
        }:
            raise RuntimeError("V92 checkpoint is not an exact two-file package")
        os.rename(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    fingerprint, files = checkpoint_fingerprint(destination)
    return {
        **inheritance,
        "checkpoint_sha256": fingerprint,
        "checkpoint_files": files,
        "adapter_sha256": sha256_file(destination / "adapter.safetensors"),
        "runtime_metadata_sha256": sha256_file(
            destination / RUNTIME_METADATA_FILENAME
        ),
        "exact_two_file_checkpoint": True,
    }


def _rebind_memory(
    destination: Path, *, checkpoint_sha256: str, runtime_config_sha256: str
) -> dict[str, Any]:
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    experiment = _load_experiment()
    source_metadata = _read_json(SOURCE_MEMORY / METADATA_FILENAME)
    if (
        SOURCE_MEMORY.is_symlink()
        or not SOURCE_MEMORY.is_dir()
        or {item.name for item in SOURCE_MEMORY.iterdir()}
        != {MEMORY_FILENAME, METADATA_FILENAME}
        or sha256_file(SOURCE_MEMORY / METADATA_FILENAME)
        != experiment["sources"]["scene1_memory_metadata_sha256"]
        or source_metadata.get("scene_id") != SCENE_ID
        or source_metadata.get("canonical_prefix_sha256")
        != SOURCE_MEMORY_PREFIX_SHA256
        or source_metadata.get("tensor_file_sha256")
        != SOURCE_MEMORY_TENSOR_FILE_SHA256
        or sha256_file(SOURCE_MEMORY / MEMORY_FILENAME)
        != SOURCE_MEMORY_TENSOR_FILE_SHA256
    ):
        raise ValueError("V92 source scene-memory bytes changed")
    rebound = dict(source_metadata)
    rebound["source_base_checkpoint_sha256"] = checkpoint_sha256
    rebound["runtime_config_sha256"] = runtime_config_sha256
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    try:
        shutil.copyfile(SOURCE_MEMORY / MEMORY_FILENAME, temporary / MEMORY_FILENAME)
        _write_json(temporary / METADATA_FILENAME, rebound)
        if sha256_file(temporary / MEMORY_FILENAME) != SOURCE_MEMORY_TENSOR_FILE_SHA256:
            raise RuntimeError("V92 scene-memory bytes changed during rebinding")
        os.rename(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    loaded = load_v81_scene_memory(
        destination,
        expected_scene_id=SCENE_ID,
        expected_base_checkpoint_sha256=checkpoint_sha256,
        expected_runtime_config_sha256=runtime_config_sha256,
        expected_model_device="cpu",
    )
    if loaded.metadata["canonical_prefix_sha256"] != SOURCE_MEMORY_PREFIX_SHA256:
        raise RuntimeError("V92 canonical scene prefix changed during rebinding")
    return {
        "source_memory_tensor_file_sha256": SOURCE_MEMORY_TENSOR_FILE_SHA256,
        "packaged_memory_tensor_file_sha256": sha256_file(
            destination / MEMORY_FILENAME
        ),
        "canonical_prefix_sha256": loaded.metadata["canonical_prefix_sha256"],
        "metadata_only_rebinding": True,
        "memory_tensor_file_bytes_unchanged": True,
        "question_data_used_for_rebinding": False,
        "exact_two_file_scene_memory": {item.name for item in destination.iterdir()}
        == {MEMORY_FILENAME, METADATA_FILENAME},
    }


def cleanup_failed_candidate() -> None:
    """Remove only an un-smoked, unpromoted partial V92 package."""

    if any(
        path.exists()
        for path in (SMOKE_REPORT, RELEASE_REPORT, RELEASE_CHECKPOINT, RELEASE_MEMORY)
    ):
        raise RuntimeError("Refusing V92 cleanup after smoke or release evidence exists")
    for root in (CANDIDATE_CHECKPOINT, CANDIDATE_MEMORY):
        if root.is_symlink():
            raise ValueError(f"Refusing to clean symbolic-link candidate: {root}")
        if root.exists():
            shutil.rmtree(root)
    if RUNTIME_CONFIG.exists():
        if RUNTIME_CONFIG.is_symlink():
            raise ValueError("Refusing to clean symbolic-link V92 runtime config")
        RUNTIME_CONFIG.unlink()


def prepare_candidate() -> dict[str, Any]:
    """Package only after every sealed V92 model gate passes exactly."""

    destinations = (
        CANDIDATE_CHECKPOINT,
        CANDIDATE_MEMORY,
        SMOKE_REPORT,
        RELEASE_CHECKPOINT,
        RELEASE_MEMORY,
        RELEASE_REPORT,
    )
    if any(path.exists() or path.is_symlink() for path in destinations):
        raise FileExistsError("V92 runtime candidate destination is not pristine")
    evidence = authenticate_v92_model_gate()
    config = materialize_runtime_config(evidence)
    metadata = build_runtime_metadata(
        evidence,
        promotion="pending_isolated_runtime_smoke",
        smoke_report_sha256=None,
    )
    checkpoint = _atomic_checkpoint(
        CANDIDATE_CHECKPOINT, metadata=metadata, evidence=evidence
    )
    memory = _rebind_memory(
        CANDIDATE_MEMORY,
        checkpoint_sha256=str(checkpoint["checkpoint_sha256"]),
        runtime_config_sha256=effective_runtime_config_sha256(config),
    )
    return {
        "schema_version": 92,
        "phase": "v92_strict_runtime_candidate_prepared",
        "candidate_checkpoint": _relative(CANDIDATE_CHECKPOINT),
        "candidate_memory": _relative(CANDIDATE_MEMORY),
        "runtime_config": _relative(RUNTIME_CONFIG),
        "runtime_config_sha256": effective_runtime_config_sha256(config),
        "checkpoint": checkpoint,
        "scene_memory": memory,
        "model_gate_evidence": evidence,
        "runtime_metadata_contains_supervision": False,
        "promotion_decision": "pending_isolated_runtime_smoke",
    }


def verify_candidate() -> dict[str, Any]:
    evidence = authenticate_v92_model_gate()
    if not CANDIDATE_CHECKPOINT.is_dir() or not CANDIDATE_MEMORY.is_dir():
        raise FileNotFoundError("V92 strict runtime candidate package is incomplete")
    expected_metadata = build_runtime_metadata(
        evidence,
        promotion="pending_isolated_runtime_smoke",
        smoke_report_sha256=None,
    )
    metadata = _read_json(CANDIDATE_CHECKPOINT / RUNTIME_METADATA_FILENAME)
    if metadata != expected_metadata:
        raise ValueError("V92 candidate runtime metadata changed")
    fingerprint, files = checkpoint_fingerprint(CANDIDATE_CHECKPOINT)
    candidate = load_file(
        str(CANDIDATE_CHECKPOINT / "adapter.safetensors"), device="cpu"
    )
    parent = load_file(str(PARENT_CHECKPOINT / "adapter.safetensors"), device="cpu")
    expected_tensors, composition = _composed_adapter(evidence)
    config = load_runtime_config(RUNTIME_CONFIG)
    validate_runtime_bank_inventory(
        runtime_config=config,
        checkpoint_metadata=metadata,
        expected_bank_order=EXPECTED_BANKS,
        expected_states={
            str(name): str(value)
            for name, value in metadata["lora_bank_state_sha256"].items()
        },
    )
    checks = {
        "exact_two_file_checkpoint": {item["path"] for item in files}
        == {"adapter.safetensors", RUNTIME_METADATA_FILENAME},
        "exact_fourteen_bank_order": composition["final_bank_order"]
        == list(EXPECTED_BANKS),
        "exact_six_v90_v91_v92_bridge_tensors_added": composition["added_tensor_count"]
        == 6,
        "v89_parent_tensors_byte_identical": composition["base_tensors_byte_identical"]
        is True
        and set(parent).issubset(candidate)
        and all(torch.equal(candidate[name], value) for name, value in parent.items()),
        "exact_tensor_inventory": set(candidate) == set(expected_tensors),
        "all_tensor_values_equal": set(candidate) == set(expected_tensors)
        and all(
            torch.equal(candidate[name], expected_tensors[name]) for name in candidate
        ),
        "exact_adapter_parameter_count": metadata["lora_parameter_count"]
        == EXPECTED_ADAPTER_PARAMETER_COUNT,
        "zero_trainable_runtime_parameters": metadata[
            "lora_trainable_parameter_count"
        ]
        == 0,
        "scene_memory_bytes_unchanged": sha256_file(
            CANDIDATE_MEMORY / MEMORY_FILENAME
        )
        == SOURCE_MEMORY_TENSOR_FILE_SHA256,
    }
    load_v81_scene_memory(
        CANDIDATE_MEMORY,
        expected_scene_id=SCENE_ID,
        expected_base_checkpoint_sha256=fingerprint,
        expected_runtime_config_sha256=effective_runtime_config_sha256(config),
        expected_model_device="cpu",
    )
    if not all(checks.values()):
        raise RuntimeError(f"V92 strict candidate verification failed: {checks}")
    return {
        "phase": "v92_strict_runtime_candidate_verified",
        "checkpoint_sha256": fingerprint,
        "adapter_sha256": sha256_file(
            CANDIDATE_CHECKPOINT / "adapter.safetensors"
        ),
        "memory_tensor_sha256": sha256_file(CANDIDATE_MEMORY / MEMORY_FILENAME),
        "v92_bridge_state_sha256": evidence["v92_bridge_state_sha256"],
        "checks": checks,
        "passed": True,
    }


def _primary_cases(
    experiment: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], ...]:
    """Return evaluation-only cases; this payload must never enter chat."""

    experiment = _load_experiment() if experiment is None else experiment
    rows = primary_rows_v92(experiment)
    cases: list[dict[str, Any]] = []
    for row in rows:
        identifier = row.question_id.removeprefix("v91_").rsplit("_existing_", 1)[0]
        if not identifier or not row.question.strip() or not row.answer.strip():
            raise ValueError("V92 primary conversational case is malformed")
        cases.append(
            {
                "intent_id": identifier,
                "family": row.answer_type,
                "question": row.question,
                "expected": row.answer,
                "core_actionable": identifier in CORE_ACTIONABLE_INTENTS,
            }
        )
    identifiers = {str(row["intent_id"]) for row in cases}
    if (
        len(cases) != 13
        or len(identifiers) != 13
        or len({str(row["question"]) for row in cases}) != 13
        or {str(row["intent_id"]) for row in cases if row["core_actionable"]}
        != CORE_ACTIONABLE_INTENTS
    ):
        raise ValueError("V92 primary conversational case coverage changed")
    return tuple(cases)


def _smoke_command(questions: Sequence[str]) -> list[str]:
    """Build the child protocol from questions only, never expectations."""

    python = PROJECT_ROOT / ".venv-gemma4/bin/python"
    command = [
        str(python),
        "-m",
        "semantic_3d_chat.chat.v83_direct_scene_memory_cli",
        "--config",
        str(RUNTIME_CONFIG),
        "--scene",
        SCENE_ID,
        "--base-checkpoint",
        str(CANDIDATE_CHECKPOINT),
        "--scene-memory",
        str(CANDIDATE_MEMORY),
        "--audit-log",
        str(SMOKE_AUDIT),
        "--chat-log",
        str(SMOKE_CHAT),
    ]
    for question in questions:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("V92 smoke question is empty")
        command.extend(("--question", question))
    return command


def _protected_smoke_reads(audit: Mapping[str, Any]) -> list[str]:
    loaded = audit.get("loaded_files")
    if not isinstance(loaded, list) or not all(isinstance(path, str) for path in loaded):
        raise ValueError("V92 child file audit is malformed")
    exact_files = {
        path.resolve()
        for path in (
            EXPERIMENT_CONFIG,
            PREREGISTRATION,
            CPU_PREFLIGHT,
            TRAINING_REPORT,
            PREDICTIONS,
            MODEL_GATE_REPORT,
            PARENT_RELEASE_REPORT,
        )
    }
    protected_roots = {
        V90_BRIDGE_CANDIDATE.resolve(),
        V91_BRIDGE_CANDIDATE.resolve(),
        V92_BRIDGE_CANDIDATE.resolve(),
    }
    forbidden_components = {"oracle", "qa", "validation", "test", "deferred"}
    forbidden_module_fragments = {
        "v90_scene1_conversational_preflight.py",
        "train_v90_scene1_conversational.py",
        "evaluate_v90_scene1_conversational.py",
        "v91_scene1_conversational_preflight.py",
        "train_v91_scene1_conversational_repair.py",
        "evaluate_v91_scene1_conversational_repair.py",
        "v92_scene1_retention_conversation_preflight.py",
        "train_v92_scene1_retention_conversation_repair.py",
        "evaluate_v92_scene1_retention_conversation_repair.py",
        "v92_strict_runtime_release",
    }
    violations: list[str] = []
    for raw in loaded:
        path = Path(raw).expanduser().resolve()
        component_violation = bool(
            forbidden_components.intersection(part.casefold() for part in path.parts)
        )
        module_violation = any(
            fragment in str(path).casefold() for fragment in forbidden_module_fragments
        )
        root_violation = any(
            path == root or root in path.parents for root in protected_roots
        )
        if path in exact_files or component_violation or module_violation or root_violation:
            violations.append(str(path))
    return sorted(set(violations))


def _score_behavior(
    rows: Sequence[Mapping[str, Any]], cases: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    # This evaluation-only scorer is intentionally imported after child exit.
    from semantic_3d_chat.evaluation.evaluate_v91_scene1_conversational_repair import (
        conversational_match_v91,
    )

    if len(rows) != len(cases):
        raise ValueError("V92 smoke result count differs from primary inventory")
    behavior: list[dict[str, Any]] = []
    for row, case in zip(rows, cases, strict=True):
        observed = str(row.get("answer", "")).strip()
        passed = conversational_match_v91(
            str(case["intent_id"]),
            str(case["family"]),
            observed,
            case["expected"],
        )
        behavior.append({**dict(case), "observed": observed, "passed": passed})
    return behavior


def _direct_layout_passes(row: Mapping[str, Any]) -> bool:
    layout = row.get("prepared_layout_audit")
    return bool(
        isinstance(layout, Mapping)
        and layout.get("fixed_scene_memory_tokens_supplied_to_gemma") == 738
        and layout.get("continuous_environment_payload_tokens") == 736
        and layout.get("native_boi_tokens") == 1
        and layout.get("native_eoi_tokens") == 1
        and layout.get("payload_image_modality_exact") is True
        and layout.get("payload_pad_ple_exact") is True
        and layout.get("boi_eoi_native_ple_exact") is True
        and layout.get("all_payload_tokens_unmasked") is True
        and layout.get("control_activation_tokens") == 0
        and layout.get("question_derived_environmental_tokens") == 0
    )


def validate_runtime_smoke_report_v92(
    smoke: Mapping[str, Any], evidence: Mapping[str, Any]
) -> None:
    """Authenticate one create-once oracle-absent child result."""

    cases = _primary_cases()
    behavior = smoke.get("behavior")
    gates = smoke.get("gates")
    if not isinstance(behavior, list) or len(behavior) != 13:
        raise ValueError("V92 runtime smoke behavior inventory changed")
    case_fields_match = all(
        isinstance(row, Mapping)
        and {
            key: row.get(key)
            for key in ("intent_id", "family", "question", "expected", "core_actionable")
        }
        == dict(case)
        and isinstance(row.get("observed"), str)
        and isinstance(row.get("passed"), bool)
        for row, case in zip(behavior, cases, strict=True)
    )
    correct = sum(row.get("passed") is True for row in behavior)
    core = sum(
        row.get("passed") is True and row.get("core_actionable") is True
        for row in behavior
    )
    if (
        smoke.get("schema_version") != 92
        or smoke.get("artifact") != "gemma4_v92_strict_runtime_smoke_v1"
        or smoke.get("model_gate_evidence_sha256")
        != _canonical_sha256(dict(evidence))
        or smoke.get("model_gate_report_sha256")
        != evidence.get("model_gate_report_sha256")
        or smoke.get("training_report_sha256")
        != evidence.get("training_report_sha256")
        or smoke.get("evaluation_predictions_sha256")
        != evidence.get("evaluation_predictions_sha256")
        or smoke.get("v92_bridge_state_sha256")
        != evidence.get("v92_bridge_state_sha256")
        or smoke.get("v91_bridge_state_sha256") != V91_STATE_SHA256
        or smoke.get("v90_bridge_state_sha256") != V90_STATE_SHA256
        or smoke.get("behavior_assertions_applied_after_chat_process_exit") is not True
        or smoke.get("expected_answers_supplied_to_chat_child") is not False
        or smoke.get("expected_behavior_not_loaded_by_chat_runtime") is not True
        or not case_fields_match
        or smoke.get("behavior_correct") != correct
        or smoke.get("behavior_total") != 13
        or smoke.get("core_actionable_correct") != core
        or smoke.get("core_actionable_total") != 6
        or correct < 12
        or core != 6
        or smoke.get("prefix_hashes") != [SOURCE_MEMORY_PREFIX_SHA256] * 13
        or smoke.get("environment_conditioned_input_hashes")
        != [SOURCE_MEMORY_PREFIX_SHA256] * 13
        or not isinstance(gates, Mapping)
        or set(gates) != _REQUIRED_RUNTIME_GATES
        or any(value is not True for value in gates.values())
        or smoke.get("passed") is not True
        or smoke.get("promotion_authorized") is not True
        or smoke.get("held_out_scene") is not False
        or smoke.get("held_out_generalization_claim") is not False
        or smoke.get("chat_log_sha256") != sha256_file(SMOKE_CHAT)
        or smoke.get("file_audit_sha256") != sha256_file(SMOKE_AUDIT)
    ):
        raise ValueError("V92 runtime smoke evidence did not pass exactly")


def run_smoke() -> dict[str, Any]:
    """Run the generic direct-memory child with the oracle renamed away."""

    if SMOKE_REPORT.is_file():
        existing = _read_json(SMOKE_REPORT)
        validate_runtime_smoke_report_v92(existing, authenticate_v92_model_gate())
        return existing
    if SMOKE_CHAT.exists() or SMOKE_AUDIT.exists():
        raise FileExistsError("V92 smoke artifacts exist; results are create-once")
    evidence = authenticate_v92_model_gate()
    candidate = verify_candidate()
    cases = _primary_cases()
    questions = [str(case["question"]) for case in cases]
    command = _smoke_command(questions)
    if command.count("--question") != 13 or any(
        token in command for token in ("--expected", "--answer", "--reference")
    ):
        raise RuntimeError("V92 child protocol contains an expectation channel")

    oracle = PROJECT_ROOT / "data/oracle"
    unavailable = PROJECT_ROOT / f"data/.oracle-unavailable-v92-{os.getpid()}"
    if not oracle.is_dir() or oracle.is_symlink() or unavailable.exists():
        raise FileNotFoundError("V92 oracle cannot be made physically unavailable")
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONPATH": str(PROJECT_ROOT / "src"),
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_OFFLINE": "1",
        }
    )
    completed: subprocess.CompletedProcess[str] | None = None
    oracle_unavailable = False
    try:
        os.rename(oracle, unavailable)
        oracle_unavailable = not oracle.exists() and unavailable.is_dir()
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
    finally:
        if unavailable.exists():
            os.rename(unavailable, oracle)
    if completed is None or completed.returncode != 0:
        stdout = completed.stdout if completed is not None else ""
        stderr = completed.stderr if completed is not None else "child did not start"
        code = completed.returncode if completed is not None else None
        raise RuntimeError(
            f"V92 strict runtime smoke failed: returncode={code}\n"
            f"stdout={stdout}\nstderr={stderr}"
        )

    rows = [
        json.loads(line)
        for line in SMOKE_CHAT.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if (
        len(rows) != 13
        or not all(isinstance(row, dict) for row in rows)
        or [row.get("question") for row in rows] != questions
    ):
        raise RuntimeError("V92 smoke chat rows differ from question protocol")
    stdout_records: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            stdout_records.append(value)
    startup_rows = [
        row
        for row in stdout_records
        if row.get("phase") == "v83_direct_fixed_scene_memory_ready"
    ]
    completion_rows = [
        row for row in stdout_records if row.get("phase") == "v83_chat_audit_complete"
    ]
    startup = startup_rows[0] if len(startup_rows) == 1 else {}
    completion = completion_rows[0] if len(completion_rows) == 1 else {}

    # Expectations first enter here, after subprocess.run has completed.
    behavior = _score_behavior(rows, cases)
    correct = sum(row["passed"] is True for row in behavior)
    core_correct = sum(
        row["passed"] is True and row["core_actionable"] is True for row in behavior
    )
    audit = _read_json(SMOKE_AUDIT)
    protected_reads = _protected_smoke_reads(audit)
    prefix_hashes = [row.get("prefix_hash") for row in rows]
    input_hashes = [
        row.get("environment_conditioned_input_sha256", row.get("prefix_hash"))
        for row in rows
    ]
    lora = startup.get("lora")
    lora_banks = lora.get("banks") if isinstance(lora, Mapping) else None
    exact_banks = bool(
        isinstance(lora_banks, list)
        and [row.get("name") for row in lora_banks if isinstance(row, Mapping)]
        == list(EXPECTED_BANKS)
        and all(
            isinstance(row, Mapping) and row.get("trainable") is False
            for row in lora_banks
        )
    )
    expectations_absent = bool(
        command.count("--question") == 13
        and [
            command[index + 1]
            for index, value in enumerate(command)
            if value == "--question"
        ]
        == questions
        and all(
            flag not in command for flag in ("--expected", "--answer", "--reference")
        )
    )
    gates = {
        "model_acceptance_gate_authenticated_and_passed": evidence[
            "model_acceptance_gate_passed"
        ]
        is True,
        "runtime_process_exit_zero": completed.returncode == 0,
        "at_least_twelve_of_thirteen_behavior_assertions_pass": correct >= 12,
        "all_six_core_actionable_intents_pass": core_correct == 6,
        "oracle_physically_unavailable": oracle_unavailable,
        "oracle_restored_after_runtime": oracle.is_dir(),
        "child_audit_completion_passed": completion.get("passed") is True
        and completion.get("fixed_memory_invariant") is True,
        "child_used_exact_fourteen_frozen_banks": exact_banks,
        "child_parameter_inventory_exact": isinstance(lora, Mapping)
        and lora.get("adapter_parameter_count") == EXPECTED_ADAPTER_PARAMETER_COUNT
        and lora.get("trainable_adapter_parameter_count") == 0,
        "file_audit_forbidden_read_count_zero": audit.get("forbidden_accesses") == []
        and audit.get("passed") is True
        and completion.get("forbidden_access_count") == 0,
        "file_audit_protected_read_count_zero": protected_reads == [],
        "prefix_hash_identical_for_every_question": len(set(prefix_hashes)) == 1,
        "total_environment_conditioned_input_identical": len(set(input_hashes)) == 1,
        "prefix_and_environment_input_identical": prefix_hashes == input_hashes,
        "expected_immutable_scene_prefix": set(prefix_hashes)
        == {SOURCE_MEMORY_PREFIX_SHA256},
        "exact_direct_memory_layout_every_question": all(
            _direct_layout_passes(row) for row in rows
        ),
        "source_memory_bytes_unchanged": sha256_file(
            CANDIDATE_MEMORY / MEMORY_FILENAME
        )
        == SOURCE_MEMORY_TENSOR_FILE_SHA256,
        "expectations_absent_from_child_protocol": expectations_absent,
    }
    report = {
        "schema_version": 92,
        "artifact": "gemma4_v92_strict_runtime_smoke_v1",
        "model_gate_evidence_sha256": _canonical_sha256(evidence),
        "model_gate_report_sha256": evidence["model_gate_report_sha256"],
        "training_report_sha256": evidence["training_report_sha256"],
        "evaluation_predictions_sha256": evidence["evaluation_predictions_sha256"],
        "v92_bridge_state_sha256": evidence["v92_bridge_state_sha256"],
        "v91_bridge_state_sha256": V91_STATE_SHA256,
        "v90_bridge_state_sha256": V90_STATE_SHA256,
        "candidate_checkpoint_sha256": candidate["checkpoint_sha256"],
        "candidate_adapter_sha256": candidate["adapter_sha256"],
        "candidate_memory_tensor_sha256": candidate["memory_tensor_sha256"],
        "runtime_config_sha256": effective_runtime_config_sha256(
            load_runtime_config(RUNTIME_CONFIG)
        ),
        "chat_log_sha256": sha256_file(SMOKE_CHAT),
        "file_audit_sha256": sha256_file(SMOKE_AUDIT),
        "chat_process_stdout_sha256": hashlib.sha256(
            completed.stdout.encode()
        ).hexdigest(),
        "chat_process_stderr_sha256": hashlib.sha256(
            completed.stderr.encode()
        ).hexdigest(),
        "behavior_assertions_applied_after_chat_process_exit": True,
        "expected_answers_supplied_to_chat_child": False,
        "expected_behavior_not_loaded_by_chat_runtime": True,
        "behavior": behavior,
        "behavior_correct": correct,
        "behavior_total": 13,
        "core_actionable_correct": core_correct,
        "core_actionable_total": 6,
        "prefix_hashes": prefix_hashes,
        "environment_conditioned_input_hashes": input_hashes,
        "protected_reads": protected_reads,
        "gates": gates,
        "passed": all(gates.values()),
        "promotion_authorized": all(gates.values()),
        "held_out_scene": False,
        "held_out_generalization_claim": False,
    }
    _write_json(SMOKE_REPORT, report)
    return report


def promote_release() -> dict[str, Any]:
    """Promote create-once only after the exact isolated smoke passes."""

    evidence = authenticate_v92_model_gate()
    if any(
        path.exists() or path.is_symlink()
        for path in (RELEASE_CHECKPOINT, RELEASE_MEMORY, RELEASE_REPORT)
    ):
        raise FileExistsError("V92 strict runtime release destination already exists")
    smoke = _read_json(SMOKE_REPORT)
    validate_runtime_smoke_report_v92(smoke, evidence)
    candidate = verify_candidate()
    if (
        smoke.get("candidate_checkpoint_sha256") != candidate["checkpoint_sha256"]
        or smoke.get("candidate_adapter_sha256") != candidate["adapter_sha256"]
        or smoke.get("candidate_memory_tensor_sha256")
        != candidate["memory_tensor_sha256"]
        or smoke.get("runtime_config_sha256")
        != effective_runtime_config_sha256(load_runtime_config(RUNTIME_CONFIG))
    ):
        raise ValueError("V92 smoked candidate bytes changed before promotion")
    smoke_sha = sha256_file(SMOKE_REPORT)
    metadata = build_runtime_metadata(
        evidence,
        promotion=PROMOTION_DECISION,
        smoke_report_sha256=smoke_sha,
    )
    checkpoint = _atomic_checkpoint(
        RELEASE_CHECKPOINT,
        metadata=metadata,
        evidence=evidence,
        source_adapter=CANDIDATE_CHECKPOINT / "adapter.safetensors",
    )
    if checkpoint["adapter_sha256"] != candidate["adapter_sha256"]:
        raise RuntimeError("Promoted V92 adapter differs from smoked candidate")
    config = load_runtime_config(RUNTIME_CONFIG)
    memory = _rebind_memory(
        RELEASE_MEMORY,
        checkpoint_sha256=str(checkpoint["checkpoint_sha256"]),
        runtime_config_sha256=effective_runtime_config_sha256(config),
    )
    release = {
        "schema_version": 92,
        "artifact": "gemma4_v92_strict_runtime_release_v1",
        "promotion_decision": PROMOTION_DECISION,
        "promotion_scope": "strict_direct_continuous_scene_memory_scene1_chat",
        "scene_id": SCENE_ID,
        "strict_input_contract": {
            "shape": [1, 738, 1536],
            "continuous_environment_payload_tokens": 736,
            "native_boi_eoi_retained": True,
            "compiled_before_question": True,
            "same_exact_memory_reused_for_every_question": True,
            "question_derived_environmental_tokens": 0,
            "question_conditioned_environmental_readout": False,
            "question_dependent_scene_processing": False,
            "question_dependent_retrieval": False,
            "control_tokens": 0,
            "environmental_text_inputs": [],
        },
        "held_out_scene": False,
        "held_out_generalization_claim": False,
        "runtime_config": _relative(RUNTIME_CONFIG),
        "runtime_config_sha256": effective_runtime_config_sha256(config),
        "checkpoint": checkpoint,
        "scene_memory": memory,
        "bindings": {**evidence, "runtime_smoke_sha256": smoke_sha},
        "chat_runtime_loads_training_or_evaluation_reports": False,
        "runtime_checkpoint_contains_environmental_text": False,
        "runtime_checkpoint_contains_supervision": False,
        "scene_memory_metadata_only_rebinding": True,
        "scene_memory_tensor_bytes_unchanged": True,
        "all_release_gates_passed": True,
    }
    _write_json(RELEASE_REPORT, release)
    return release


def verify_release() -> dict[str, Any]:
    evidence = authenticate_v92_model_gate()
    smoke = _read_json(SMOKE_REPORT)
    validate_runtime_smoke_report_v92(smoke, evidence)
    release = _read_json(RELEASE_REPORT)
    metadata = _read_json(RELEASE_CHECKPOINT / RUNTIME_METADATA_FILENAME)
    smoke_sha = sha256_file(SMOKE_REPORT)
    expected_metadata = build_runtime_metadata(
        evidence,
        promotion=PROMOTION_DECISION,
        smoke_report_sha256=smoke_sha,
    )
    if metadata != expected_metadata:
        raise ValueError("V92 promoted runtime metadata changed")
    fingerprint, files = checkpoint_fingerprint(RELEASE_CHECKPOINT)
    config = load_runtime_config(RUNTIME_CONFIG)
    loaded = load_v81_scene_memory(
        RELEASE_MEMORY,
        expected_scene_id=SCENE_ID,
        expected_base_checkpoint_sha256=fingerprint,
        expected_runtime_config_sha256=effective_runtime_config_sha256(config),
        expected_model_device="cpu",
    )
    provenance = metadata["initialization_provenance"]["v92_strict_runtime_release"]
    validate_runtime_bank_inventory(
        runtime_config=config,
        checkpoint_metadata=metadata,
        expected_bank_order=EXPECTED_BANKS,
        expected_states={
            str(name): str(value)
            for name, value in metadata["lora_bank_state_sha256"].items()
        },
    )
    parent = load_file(str(PARENT_CHECKPOINT / "adapter.safetensors"), device="cpu")
    promoted = load_file(
        str(RELEASE_CHECKPOINT / "adapter.safetensors"), device="cpu"
    )
    bindings = release.get("bindings")
    checks = {
        "release_report_identity": release.get("artifact")
        == "gemma4_v92_strict_runtime_release_v1"
        and release.get("schema_version") == 92
        and release.get("all_release_gates_passed") is True,
        "release_report_promoted": release.get("promotion_decision")
        == PROMOTION_DECISION,
        "exact_two_file_checkpoint": {item["path"] for item in files}
        == {"adapter.safetensors", RUNTIME_METADATA_FILENAME},
        "checkpoint_fingerprint_matches_release": isinstance(
            release.get("checkpoint"), Mapping
        )
        and fingerprint == release["checkpoint"].get("checkpoint_sha256"),
        "adapter_matches_smoked_candidate": sha256_file(
            RELEASE_CHECKPOINT / "adapter.safetensors"
        )
        == sha256_file(CANDIDATE_CHECKPOINT / "adapter.safetensors")
        == smoke.get("candidate_adapter_sha256"),
        "v89_parent_tensors_byte_identical": set(parent).issubset(promoted)
        and all(torch.equal(promoted[name], value) for name, value in parent.items()),
        "exact_six_v90_v91_v92_tensors_added": len(set(promoted) - set(parent)) == 6,
        "memory_bytes_match_source": sha256_file(RELEASE_MEMORY / MEMORY_FILENAME)
        == SOURCE_MEMORY_TENSOR_FILE_SHA256,
        "memory_prefix_match_source": loaded.metadata["canonical_prefix_sha256"]
        == SOURCE_MEMORY_PREFIX_SHA256,
        "model_gate_binding_exact": provenance["model_gate_report_sha256"]
        == evidence["model_gate_report_sha256"],
        "runtime_smoke_binding_exact": provenance["smoke_report_sha256"]
        == smoke_sha
        and isinstance(bindings, Mapping)
        and bindings.get("runtime_smoke_sha256") == smoke_sha,
        "v92_state_binding_exact": provenance["v92_bridge_state_sha256"]
        == evidence["v92_bridge_state_sha256"],
        "v91_state_binding_exact": provenance["v91_bridge_state_sha256"]
        == V91_STATE_SHA256,
        "v90_state_binding_exact": provenance["v90_bridge_state_sha256"]
        == V90_STATE_SHA256,
        "exact_fourteen_frozen_banks": tuple(
            row["name"] for row in metadata["lora"]["banks"]
        )
        == EXPECTED_BANKS
        and metadata["lora"]["trainable_adapter_parameter_count"] == 0
        and metadata["lora"]["adapter_parameter_count"]
        == EXPECTED_ADAPTER_PARAMETER_COUNT,
        "runtime_promotion_authorized": provenance["runtime_promotion_authorized"]
        is True,
        "no_held_out_scene_claim": release.get("held_out_scene") is False
        and release.get("held_out_generalization_claim") is False,
    }
    if not all(checks.values()):
        raise RuntimeError(f"V92 strict runtime release verification failed: {checks}")
    return {"phase": "v92_strict_runtime_release_verified", "checks": checks, "passed": True}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "authenticate",
            "prepare",
            "verify-candidate",
            "smoke",
            "promote",
            "verify",
            "cleanup-failed-candidate",
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    functions = {
        "authenticate": authenticate_v92_model_gate,
        "prepare": prepare_candidate,
        "verify-candidate": verify_candidate,
        "smoke": run_smoke,
        "promote": promote_release,
        "verify": verify_release,
        "cleanup-failed-candidate": lambda: (
            cleanup_failed_candidate()
            or {"phase": "v92_failed_candidate_cleaned", "passed": True}
        ),
    }
    try:
        result = functions[args.command]()
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"V92 strict runtime {args.command} refused: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 2 if args.command == "smoke" and result.get("passed") is not True else 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CANDIDATE_CHECKPOINT",
    "CANDIDATE_MEMORY",
    "CPU_PREFLIGHT_SHA256",
    "EXPECTED_BANKS",
    "EXPERIMENT_CONFIG_SHA256",
    "MODEL_GATE_REPORT",
    "PREREGISTRATION_SHA256",
    "RELEASE_CHECKPOINT",
    "RELEASE_MEMORY",
    "RUNTIME_CONFIG",
    "SMOKE_REPORT",
    "SOURCE_MEMORY_PREFIX_SHA256",
    "SOURCE_MEMORY_TENSOR_FILE_SHA256",
    "V92_BANK",
    "V92_BRIDGE_CANDIDATE",
    "authenticate_v92_model_gate",
    "build_runtime_config_payload",
    "build_runtime_metadata",
    "cleanup_failed_candidate",
    "main",
    "materialize_runtime_config",
    "prepare_candidate",
    "promote_release",
    "run_smoke",
    "validate_model_gate_contract_v92",
    "validate_runtime_smoke_report_v92",
    "verify_candidate",
    "verify_release",
]
