"""Authenticate, package, isolate-smoke, and promote V89's strict runtime.

This module is an evaluation/release surface and is never imported by chat.
It refuses every write unless the fixed V89 evaluation has passed every
preregistered gate.  The external chat child receives only a standalone
runtime YAML, an exact two-file eleven-bank checkpoint, and an exact two-file
continuous scene memory.  Semantic expectations are applied here only after
the child process exits.
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
from semantic_3d_chat.chat.v89_strict_scene1_runtime import (
    EXPECTED_ADAPTER_PARAMETER_COUNT,
    EXPECTED_BANKS,
    SCENE_ID,
    V86_BANK,
    V86_STATE_SHA256,
    V86_TARGET,
    V87_BANK,
    V87_STATE_SHA256,
    V87_TARGET,
    V88_BANK,
    V88_STATE_SHA256,
    V88_TARGET,
    V89_BANK,
    V89_TARGET,
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

SCHEMA_VERSION: Final[int] = 89
_SHA256: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{64}")
EXPERIMENT_CONFIG_SHA256: Final[str] = (
    "6781126de7e378a27e9a4140d2e47efb7b673c8a0b3522dd762fea5214312e2c"
)
PREREGISTRATION_SHA256: Final[str] = (
    "493208eb96b6bfe14267ebc05612441457a2b52751f0ca06e4fb90fab84d94a9"
)
CPU_PREFLIGHT_SHA256: Final[str] = (
    "bc063b1cbad1d05a53e0044bfcc80f6d52f994e1ef85e0b5ed351469a987e256"
)
if any(
    _SHA256.fullmatch(value) is None
    for value in (
        EXPERIMENT_CONFIG_SHA256,
        PREREGISTRATION_SHA256,
        CPU_PREFLIGHT_SHA256,
    )
):
    raise RuntimeError("V89 fixed evidence constants must be 64-character SHA-256")
SOURCE_MEMORY_PREFIX_SHA256: Final[str] = (
    "a428f5147c815839ae7315a0adab952ab210814fb21dcdc5bf13b167f28a6e37"
)
SOURCE_MEMORY_TENSOR_FILE_SHA256: Final[str] = (
    "3427851432d8f2a3609b6205b18b8c0d9a0fcf68d8f3bf0c98c758ac64209ffb"
)
_BASE_BANKS: Final[tuple[str, ...]] = EXPECTED_BANKS[:7]
_REQUIRED_MODEL_GATES: Final[frozenset[str]] = frozenset(
    {
        "all_scene1_canonical_accuracy_at_least_0_80",
        "attribute_accuracy_at_least_0_50",
        "presence_accuracy_at_least_0_75",
        "spatial_relation_accuracy_at_least_0_60",
        "exact_training_row_count_138",
        "generic_live_smoke_exactly_3_of_3",
        "causal_correct_memory_mean_nll_below_zero_payload",
        "causal_prediction_change_at_least_1",
        "exact_prefix_hash_invariance",
        "exact_total_environment_input_invariance",
        "protected_read_count_zero",
    }
)
_REQUIRED_RUNTIME_GATES: Final[frozenset[str]] = frozenset(
    {
        "model_acceptance_gate_authenticated_and_passed",
        "runtime_process_exit_zero",
        "three_behavior_assertions_pass",
        "oracle_physically_unavailable",
        "child_reported_oracle_unavailable_at_start",
        "oracle_restored_after_runtime",
        "child_audit_completion_passed",
        "child_loaded_no_training_or_evaluation_report",
        "child_used_exact_eleven_frozen_banks",
        "file_audit_forbidden_read_count_zero",
        "prefix_hash_identical_for_every_question",
        "total_environment_conditioned_input_identical",
        "prefix_and_environment_input_identical",
        "expected_immutable_scene_prefix",
        "source_memory_bytes_unchanged",
    }
)

EXPERIMENT_CONFIG: Final[Path] = (
    PROJECT_ROOT / "configs/experiments/gemma4_v89_scene1_retention_demo.yaml"
)
BASE_RUNTIME_CONFIG: Final[Path] = (
    PROJECT_ROOT / "configs/runtime/gemma4_v85_strict_multiscene.yaml"
)
RUNTIME_CONFIG: Final[Path] = (
    PROJECT_ROOT / "configs/runtime/gemma4_v89_strict_scene1.yaml"
)
V85_CHECKPOINT: Final[Path] = (
    PROJECT_ROOT / "reports/gemma4/artifacts/v85_strict_runtime_candidate"
)
V86_BRIDGE_CANDIDATE: Final[Path] = (
    PROJECT_ROOT / "reports/gemma4/artifacts/v86_scene1_demo_final"
)
V87_BRIDGE_CANDIDATE: Final[Path] = (
    PROJECT_ROOT / "reports/gemma4/artifacts/v87_scene1_balanced_final"
)
V88_BRIDGE_CANDIDATE: Final[Path] = (
    PROJECT_ROOT / "reports/gemma4/artifacts/v88_scene1_augmented_final"
)
V89_BRIDGE_CANDIDATE: Final[Path] = (
    PROJECT_ROOT / "reports/gemma4/artifacts/v89_scene1_retention_final"
)
PREREGISTRATION: Final[Path] = (
    PROJECT_ROOT
    / "reports/gemma4/metrics/gemma4_v89_scene1_retention_preregistration.json"
)
CPU_PREFLIGHT: Final[Path] = (
    PROJECT_ROOT
    / "reports/gemma4/metrics/gemma4_v89_scene1_retention_cpu_preflight.json"
)
TRAINING_REPORT: Final[Path] = (
    PROJECT_ROOT / "reports/gemma4/metrics/gemma4_v89_scene1_retention_training.json"
)
PREDICTIONS: Final[Path] = (
    PROJECT_ROOT
    / "reports/gemma4/predictions/gemma4_v89_scene1_retention_evaluation.json"
)
MODEL_GATE_REPORT: Final[Path] = (
    PROJECT_ROOT / "reports/gemma4/metrics/gemma4_v89_scene1_retention_evaluation.json"
)
SOURCE_MEMORY: Final[Path] = (
    PROJECT_ROOT / "data_gemma4/runtime/scene_memories/v81/scene_000001"
)
CANDIDATE_CHECKPOINT: Final[Path] = (
    PROJECT_ROOT / "reports/gemma4/artifacts/v89_scene1_retention_runtime"
)
CANDIDATE_MEMORY: Final[Path] = (
    PROJECT_ROOT
    / "reports/gemma4/artifacts/v89_scene1_retention_runtime_memory/scene_000001"
)
RELEASE_CHECKPOINT: Final[Path] = (
    PROJECT_ROOT
    / "data_gemma4/runtime/checkpoints/gemma4_v89_strict_scene1_release_v1"
)
RELEASE_MEMORY: Final[Path] = (
    PROJECT_ROOT / "data_gemma4/runtime/scene_memories/v89/scene_000001"
)
SMOKE_CHAT: Final[Path] = (
    PROJECT_ROOT / "reports/gemma4/examples/v89_strict_runtime_smoke.jsonl"
)
SMOKE_AUDIT: Final[Path] = (
    PROJECT_ROOT / "reports/gemma4/metrics/v89_strict_runtime_smoke_access.json"
)
SMOKE_REPORT: Final[Path] = (
    PROJECT_ROOT
    / "reports/gemma4/metrics/gemma4_v89_scene1_retention_runtime_smoke.json"
)
RELEASE_REPORT: Final[Path] = (
    PROJECT_ROOT / "reports/gemma4/metrics/gemma4_v89_strict_runtime_release.json"
)

# These expectations remain evaluation-only and never enter the chat child.
_SMOKE_CASES: Final[tuple[tuple[str, str], ...]] = (
    ("Is there a chair?", "yes"),
    ("What color is the bowl?", "red"),
    ("Is the bowl left or right of the chair?", "left"),
)

V86_CONTRACT: Final[BridgeSourceContract] = BridgeSourceContract(
    root=V86_BRIDGE_CANDIDATE,
    artifact="gemma4_v86_scene1_demo_fixed_final_v1",
    bank_name=V86_BANK,
    target_module=V86_TARGET,
    rank=8,
    alpha=16.0,
    dropout=0.0,
    parameter_count=110_592,
    state_sha256=V86_STATE_SHA256,
    weights_sha256="3e4db9a621e915da341ba2163b7c541863c067eafc5abc55e53a50f594476015",
    metadata_sha256="ad9534737bbe1cb443960159461f5ec9c78609cb1f2d311d05b245261f6ac54d",
)
V87_CONTRACT: Final[BridgeSourceContract] = BridgeSourceContract(
    root=V87_BRIDGE_CANDIDATE,
    artifact="gemma4_v87_scene1_balanced_fixed_final_v1",
    bank_name=V87_BANK,
    target_module=V87_TARGET,
    rank=8,
    alpha=16.0,
    dropout=0.0,
    parameter_count=110_592,
    state_sha256=V87_STATE_SHA256,
    weights_sha256="3dc027ec236347b09feeb1052476078e8a577f9df4b65b13a29a41d4b959578f",
    metadata_sha256="f82bed9eef8f215187730de036a34a50afa6e5ed88bf65e639f5fe3d6b11c136",
)
V88_CONTRACT: Final[BridgeSourceContract] = BridgeSourceContract(
    root=V88_BRIDGE_CANDIDATE,
    artifact="gemma4_v88_scene1_augmented_fixed_final_v1",
    bank_name=V88_BANK,
    target_module=V88_TARGET,
    rank=16,
    alpha=32.0,
    dropout=0.0,
    parameter_count=57_344,
    state_sha256=V88_STATE_SHA256,
    weights_sha256="95d4aaf9c42cbf796dc047b3e622cf92247c898987ab159570944861ef698cf1",
    metadata_sha256="670ef58de0ac3b9d1e9a141292ffea900848cc512b77fc12c69d9def706d3d41",
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

    value = json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates
    )
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
        raise ValueError(f"V89 {field} is not a lowercase SHA-256 digest")
    return value


def _load_experiment() -> dict[str, Any]:
    if EXPERIMENT_CONFIG.is_symlink() or not EXPERIMENT_CONFIG.is_file():
        raise FileNotFoundError("V89 sealed experiment config is not a physical file")
    if sha256_file(EXPERIMENT_CONFIG) != EXPERIMENT_CONFIG_SHA256:
        raise ValueError("V89 sealed experiment config changed")
    raw = EXPERIMENT_CONFIG.read_text(encoding="utf-8")
    if "REPLACE_" in raw:
        raise ValueError("V89 sealed experiment config contains a placeholder")
    payload = yaml.safe_load(raw)
    if not isinstance(payload, dict) or set(payload) != {"v89"}:
        raise ValueError("V89 experiment config identity changed")
    experiment = payload["v89"]
    dataset = experiment.get("dataset") if isinstance(experiment, dict) else None
    outputs = experiment.get("outputs") if isinstance(experiment, dict) else None
    if (
        not isinstance(experiment, dict)
        or experiment.get("schema_version") != 89
        or experiment.get("artifact")
        != "gemma4_v89_scene1_retention_direct_memory_overfit_v1"
        or experiment.get("status") != "preregistered_before_full_model_load"
        or not isinstance(dataset, dict)
        or dataset.get("scene_id") != SCENE_ID
        or dataset.get("canonical_row_count") != 138
        or dataset.get("runtime_serializes_questions_or_answers") is not False
        or dataset.get("runtime_serializes_training_inventory") is not False
        or dataset.get("runtime_serializes_error_inventory") is not False
        or dataset.get("runtime_serializes_anchor_inventory") is not False
        or not isinstance(outputs, dict)
        or outputs.get("runtime_candidate")
        != CANDIDATE_CHECKPOINT.relative_to(PROJECT_ROOT).as_posix()
        or outputs.get("runtime_smoke_report")
        != SMOKE_REPORT.relative_to(PROJECT_ROOT).as_posix()
    ):
        raise ValueError("V89 sealed single-scene experiment contract changed")
    return experiment


def validate_model_gate_contract_v89(report: Mapping[str, Any]) -> None:
    """Fail closed unless every preregistered V89 model gate is exactly true."""

    metrics = report.get("metrics")
    gates = metrics.get("model_acceptance_gates") if isinstance(metrics, dict) else None
    canonical = (
        metrics.get("canonical_type_specific") if isinstance(metrics, dict) else None
    )
    by_type = (
        metrics.get("canonical_accuracy_by_answer_type")
        if isinstance(metrics, dict)
        else None
    )
    causal = metrics.get("causal_control") if isinstance(metrics, dict) else None
    smoke = metrics.get("generic_smoke") if isinstance(metrics, dict) else None
    if (
        report.get("artifact") != "gemma4_v89_scene1_retention_evaluation_v1"
        or report.get("schema_version") != 89
        or report.get("status")
        != "model_gates_pass_separate_runtime_packaging_required"
        or not isinstance(metrics, dict)
        or metrics.get("model_acceptance_gate_passed") is not True
        or not isinstance(gates, dict)
        or set(gates) != _REQUIRED_MODEL_GATES
        or any(value is not True for value in gates.values())
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
        or report.get("development_known_smoke_trained") is not True
        or report.get("held_out_smoke_claim") is not False
        or report.get("held_out_generalization_claim") is not False
        or report.get("parent_v85_v86_v87_v88_mutated") is not False
        or report.get("oracle_loaded") is not False
        or report.get("official_validation_loaded") is not False
        or report.get("official_test_loaded") is not False
        or report.get("deferred_final_loaded") is not False
        or not isinstance(canonical, dict)
        or canonical.get("total") != 138
        or int(canonical.get("correct", -1)) < 111
        or float(canonical.get("accuracy", -1.0)) < 0.80
        or not isinstance(by_type, dict)
        or float(by_type.get("attribute", {}).get("accuracy", -1.0)) < 0.50
        or float(by_type.get("presence", {}).get("accuracy", -1.0)) < 0.75
        or float(by_type.get("spatial_relation", {}).get("accuracy", -1.0)) < 0.60
        or not isinstance(causal, dict)
        or float(causal.get("mean_zero_minus_correct_nll", 0.0)) <= 0.0
        or int(causal.get("canonical_prediction_changes", 0)) < 1
        or not isinstance(smoke, dict)
        or smoke.get("correct") != 3
        or smoke.get("total") != 3
        or float(smoke.get("accuracy", 0.0)) != 1.0
        or smoke.get("development_known_and_trained") is not True
        or smoke.get("held_out") is not False
    ):
        raise ValueError("V89 model-level acceptance report did not pass exactly")
    smoke_records = smoke.get("records")
    smoke_contract = (
        [
            (
                row.get("question"),
                row.get("expected"),
                row.get("normalized_prediction"),
                row.get("exact_correct"),
                row.get("development_known_and_trained"),
                row.get("held_out"),
            )
            for row in smoke_records
        ]
        if isinstance(smoke_records, list)
        and all(isinstance(row, dict) for row in smoke_records)
        else None
    )
    expected_smoke = [
        (question, expected, expected, True, True, False)
        for question, expected in _SMOKE_CASES
    ]
    leakage = report.get("leakage")
    memory = report.get("scene_memory")
    if (
        smoke_contract != expected_smoke
        or not isinstance(leakage, dict)
        or leakage.get("protected_read_count") != 0
        or leakage.get("protected_reads") != []
        or leakage.get("oracle_loaded") is not False
        or not isinstance(memory, dict)
        or memory.get("prefix_hash_invariant") is not True
        or memory.get("same_prefix_reused_for_every_question") is not True
        or memory.get("question_derived_environmental_tokens") != 0
        or memory.get("question_conditioned_environmental_readout") is not False
        or memory.get("question_dependent_scene_processing") is not False
        or memory.get("question_dependent_retrieval") is not False
        or memory.get("prefix_sha256_before") != SOURCE_MEMORY_PREFIX_SHA256
        or memory.get("prefix_sha256_after") != SOURCE_MEMORY_PREFIX_SHA256
    ):
        raise ValueError("V89 model leakage, smoke, or memory evidence changed")


def _authenticate_source_hashes(
    report: Mapping[str, Any], experiment: Mapping[str, Any]
) -> None:
    observed = report.get("source_hashes")
    sources = experiment.get("sources")
    if not isinstance(observed, dict) or not observed or not isinstance(sources, dict):
        raise ValueError("V89 authenticated source inventory is missing")
    expected = {
        str(sources["runtime_config"]): str(sources["runtime_config_sha256"]),
        str(sources["scene1_qa"]): str(sources["scene1_qa_sha256"]),
        str(Path(str(sources["scene1_memory"])) / MEMORY_FILENAME): str(
            sources["scene1_memory_tensor_sha256"]
        ),
        str(Path(str(sources["scene1_memory"])) / METADATA_FILENAME): str(
            sources["scene1_memory_metadata_sha256"]
        ),
        str(Path(str(sources["frozen_v85_checkpoint"])) / "adapter.safetensors"): str(
            sources["frozen_v85_adapter_sha256"]
        ),
        str(
            Path(str(sources["frozen_v85_checkpoint"])) / RUNTIME_METADATA_FILENAME
        ): str(sources["frozen_v85_metadata_sha256"]),
        str(Path(str(sources["parent_v86_checkpoint"])) / "bridge.safetensors"): str(
            sources["parent_v86_bridge_sha256"]
        ),
        str(
            Path(str(sources["parent_v86_checkpoint"])) / RUNTIME_METADATA_FILENAME
        ): str(sources["parent_v86_metadata_sha256"]),
        str(Path(str(sources["parent_v87_checkpoint"])) / "bridge.safetensors"): str(
            sources["parent_v87_bridge_sha256"]
        ),
        str(
            Path(str(sources["parent_v87_checkpoint"])) / RUNTIME_METADATA_FILENAME
        ): str(sources["parent_v87_metadata_sha256"]),
        str(sources["parent_v87_predictions"]): str(
            sources["parent_v87_predictions_sha256"]
        ),
        str(sources["parent_v88_config"]): str(sources["parent_v88_config_sha256"]),
        str(sources["parent_v88_preregistration"]): str(
            sources["parent_v88_preregistration_sha256"]
        ),
        str(sources["parent_v88_cpu_preflight"]): str(
            sources["parent_v88_cpu_preflight_sha256"]
        ),
        str(sources["parent_v88_training_report"]): str(
            sources["parent_v88_training_report_sha256"]
        ),
        str(Path(str(sources["parent_v88_checkpoint"])) / "bridge.safetensors"): str(
            sources["parent_v88_bridge_sha256"]
        ),
        str(
            Path(str(sources["parent_v88_checkpoint"])) / RUNTIME_METADATA_FILENAME
        ): str(sources["parent_v88_metadata_sha256"]),
        str(sources["parent_v88_predictions"]): str(
            sources["parent_v88_predictions_sha256"]
        ),
        str(sources["parent_v88_evaluation"]): str(
            sources["parent_v88_evaluation_sha256"]
        ),
        str(sources["preflight_source"]): str(sources["preflight_source_sha256"]),
        str(sources["training_source"]): str(sources["training_source_sha256"]),
        str(sources["evaluation_source"]): str(sources["evaluation_source_sha256"]),
        "gemma_model_blob_sha256_identity": str(
            sources["model_blob_sha256_identity"]
        ),
    }
    if observed != expected:
        raise ValueError("V89 report source inventory differs from sealed config")
    mismatches: dict[str, Any] = {}
    for relative, expected in observed.items():
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise TypeError("V89 source-hash inventory is malformed")
        if relative == "gemma_model_blob_sha256_identity":
            if expected != sources.get("model_blob_sha256_identity"):
                mismatches[relative] = {
                    "expected": sources.get("model_blob_sha256_identity"),
                    "observed": expected,
                }
            continue
        path = (PROJECT_ROOT / relative).resolve()
        try:
            path.relative_to(PROJECT_ROOT.resolve())
        except ValueError as error:
            raise ValueError("V89 authenticated source escaped project root") from error
        actual = sha256_file(path) if path.is_file() else None
        if actual != expected:
            mismatches[relative] = {"expected": expected, "observed": actual}
    if mismatches:
        raise ValueError(f"V89 authenticated sources changed: {mismatches}")


def _v89_contract(
    experiment: Mapping[str, Any],
    training: Mapping[str, Any],
    predictions: Mapping[str, Any],
) -> BridgeSourceContract:
    metadata_path = V89_BRIDGE_CANDIDATE / RUNTIME_METADATA_FILENAME
    weights_path = V89_BRIDGE_CANDIDATE / "bridge.safetensors"
    metadata = _read_json(metadata_path)
    bridge = experiment.get("bridge")
    bindings = metadata.get("bindings")
    training_candidate = training.get("candidate")
    predicted_candidate = predictions.get("candidate")
    if (
        not isinstance(bridge, dict)
        or metadata.get("artifact")
        != "gemma4_v89_scene1_retention_fixed_final_v1"
        or metadata.get("schema_version") != 89
        or metadata.get("status")
        != "fixed_final_awaiting_preregistered_acceptance_gates"
        or metadata.get("bank_name") != V89_BANK
        or metadata.get("target_module") != V89_TARGET
        or metadata.get("rank") != 8
        or float(metadata.get("alpha", -1.0)) != 16.0
        or float(metadata.get("dropout", -1.0)) != 0.0
        or metadata.get("parameter_count") != 28_672
        or metadata.get("frozen_bank_count") != 10
        or metadata.get("total_bank_count") != 11
        or metadata.get("environmental_memory_serialized") is not False
        or metadata.get("questions_or_answers_serialized") is not False
        or metadata.get("training_metadata_serialized") is not False
        or metadata.get("error_inventory_serialized") is not False
        or metadata.get("anchor_inventory_serialized") is not False
        or metadata.get("oracle_serialized") is not False
        or metadata.get("evaluation_scored") is not False
        or metadata.get("runtime_promotion_authorized") is not False
        or not isinstance(bindings, dict)
        or bindings.get("fixed_final_optimizer_updates") != 155
        or bindings.get("scene_memory_prefix_sha256")
        != SOURCE_MEMORY_PREFIX_SHA256
        or bindings.get("development_known_smoke_trained") is not True
        or not isinstance(training_candidate, dict)
        or training_candidate.get("weights_sha256") != metadata.get("weights_sha256")
        or training_candidate.get("metadata_canonical_sha256")
        != _canonical_sha256(metadata)
        or training_candidate.get("fixed_final") is not True
        or training_candidate.get("runtime_promotion_authorized") is not False
        or not isinstance(predicted_candidate, dict)
        or predicted_candidate.get("path")
        != V89_BRIDGE_CANDIDATE.relative_to(PROJECT_ROOT).as_posix()
        or predicted_candidate.get("weights_sha256") != metadata.get("weights_sha256")
        or predicted_candidate.get("state_sha256") != metadata.get("state_sha256")
        or predicted_candidate.get("optimizer_updates") != 155
    ):
        raise ValueError("V89 fixed-final bridge binding changed")
    state_sha256 = _require_hash(metadata.get("state_sha256"), "bridge state")
    weights_sha256 = _require_hash(metadata.get("weights_sha256"), "bridge weights")
    if not weights_path.is_file() or sha256_file(weights_path) != weights_sha256:
        raise ValueError("V89 fixed-final bridge weights changed")
    contract = BridgeSourceContract(
        root=V89_BRIDGE_CANDIDATE,
        artifact="gemma4_v89_scene1_retention_fixed_final_v1",
        bank_name=V89_BANK,
        target_module=V89_TARGET,
        rank=8,
        alpha=16.0,
        dropout=0.0,
        parameter_count=28_672,
        state_sha256=state_sha256,
        weights_sha256=weights_sha256,
        metadata_sha256=sha256_file(metadata_path),
    )
    loaded = load_bridge_source(contract)
    if (
        tuple(loaded.state["adapters.0.lora_a"].shape) != (8, 2_048)
        or tuple(loaded.state["adapters.0.lora_b"].shape) != (1_536, 8)
    ):
        raise ValueError("V89 fixed-final bridge tensor topology changed")
    return contract


def _contracts_from_evidence(
    evidence: Mapping[str, Any],
) -> tuple[BridgeSourceContract, ...]:
    return (
        V86_CONTRACT,
        V87_CONTRACT,
        V88_CONTRACT,
        BridgeSourceContract(
            root=V89_BRIDGE_CANDIDATE,
            artifact="gemma4_v89_scene1_retention_fixed_final_v1",
            bank_name=V89_BANK,
            target_module=V89_TARGET,
            rank=8,
            alpha=16.0,
            dropout=0.0,
            parameter_count=28_672,
            state_sha256=_require_hash(
                evidence.get("v89_bridge_state_sha256"), "bridge state"
            ),
            weights_sha256=_require_hash(
                evidence.get("v89_bridge_file_sha256"), "bridge weights"
            ),
            metadata_sha256=_require_hash(
                evidence.get("v89_bridge_metadata_sha256"), "bridge metadata"
            ),
        ),
    )


def authenticate_v89_model_gate() -> dict[str, Any]:
    """Authenticate all fixed and create-once V89 evidence without imports."""

    experiment = _load_experiment()
    outputs = experiment["outputs"]
    sources = experiment["sources"]
    fixed = {
        PREREGISTRATION: PREREGISTRATION_SHA256,
        CPU_PREFLIGHT: CPU_PREFLIGHT_SHA256,
    }
    mismatches = {
        path.relative_to(PROJECT_ROOT).as_posix(): {
            "expected": expected,
            "observed": sha256_file(path) if path.is_file() else None,
        }
        for path, expected in fixed.items()
        if path.is_symlink() or not path.is_file() or sha256_file(path) != expected
    }
    if mismatches:
        raise ValueError(f"V89 fixed preregistration evidence changed: {mismatches}")
    expected_paths = {
        "training_report": TRAINING_REPORT,
        "evaluation_predictions": PREDICTIONS,
        "evaluation_report": MODEL_GATE_REPORT,
        "fixed_final_candidate": V89_BRIDGE_CANDIDATE,
    }
    if any(
        outputs.get(key) != path.relative_to(PROJECT_ROOT).as_posix()
        for key, path in expected_paths.items()
    ):
        raise ValueError("V89 sealed output paths changed")

    training = _read_json(TRAINING_REPORT)
    predictions = _read_json(PREDICTIONS)
    report = _read_json(MODEL_GATE_REPORT)
    if report.get("preregistered_gates") != experiment.get("gates"):
        raise ValueError("V89 model gate is not bound to preregistered gates")
    validate_model_gate_contract_v89(report)
    dynamic_hashes = {
        "config_sha256": (EXPERIMENT_CONFIG_SHA256, report.get("config_sha256")),
        "preregistration_sha256": (
            PREREGISTRATION_SHA256,
            report.get("preregistration_sha256"),
        ),
        "cpu_preflight_sha256": (
            CPU_PREFLIGHT_SHA256,
            report.get("cpu_preflight_sha256"),
        ),
        "training_report_sha256": (
            sha256_file(TRAINING_REPORT),
            report.get("training_report_sha256"),
        ),
        "evaluation_predictions_sha256": (
            sha256_file(PREDICTIONS),
            report.get("evaluation_predictions_sha256"),
        ),
    }
    hash_mismatches = {
        key: {"observed": observed, "bound": bound}
        for key, (observed, bound) in dynamic_hashes.items()
        if observed != bound or _SHA256.fullmatch(observed) is None
    }
    if hash_mismatches:
        raise ValueError(f"V89 bound final evidence changed: {hash_mismatches}")
    if (
        report.get("evaluation_predictions_path")
        != PREDICTIONS.relative_to(PROJECT_ROOT).as_posix()
        or training.get("artifact")
        != "gemma4_v89_scene1_retention_training_v1"
        or training.get("schema_version") != 89
        or training.get("status") != "fixed_final_training_complete_not_promoted"
        or training.get("config_sha256") != EXPERIMENT_CONFIG_SHA256
        or training.get("preregistration_sha256") != PREREGISTRATION_SHA256
        or training.get("cpu_preflight_sha256") != CPU_PREFLIGHT_SHA256
        or training.get("optimizer_updates") != 155
        or training.get("micro_rows_consumed") != 930
        or training.get("causal_margin_rows_consumed") != 18
        or training.get("protected_read_count") != 0
        or training.get("oracle_loaded") is not False
        or training.get("official_validation_loaded") is not False
        or training.get("official_test_loaded") is not False
        or training.get("deferred_final_loaded") is not False
        or training.get("held_out_generalization_claim") is not False
        or training.get("runtime_promotion_authorized") is not False
        or not isinstance(training.get("gates"), dict)
        or not training["gates"]
        or any(value is not True for value in training["gates"].values())
        or predictions.get("artifact")
        != "gemma4_v89_scene1_retention_predictions_v1"
        or predictions.get("schema_version") != 89
        or predictions.get("status") != "fixed_final_evaluation_only_not_runtime"
        or predictions.get("config_sha256") != EXPERIMENT_CONFIG_SHA256
        or predictions.get("training_report_sha256")
        != dynamic_hashes["training_report_sha256"][0]
        or predictions.get("row_count") != 138
        or predictions.get("scene_count") != 1
        or predictions.get("fixed_checkpoint_selected_before_scoring") is not True
        or predictions.get("checkpoint_selection_after_scoring") is not False
        or predictions.get("development_known_smoke_trained") is not True
        or predictions.get("held_out_smoke_claim") is not False
        or predictions.get("training_references_serialized_in_runtime_candidate")
        is not False
        or predictions.get("error_inventory_serialized_in_runtime_candidate")
        is not False
        or predictions.get("anchor_inventory_serialized_in_runtime_candidate")
        is not False
        or predictions.get("runtime_promotion_authorized") is not False
    ):
        raise ValueError("V89 training or fixed-prediction identity changed")
    records = predictions.get("records")
    if (
        not isinstance(records, list)
        or len(records) != 138
        or any(
            not isinstance(record, dict) or record.get("scene_id") != SCENE_ID
            for record in records
        )
    ):
        raise ValueError("V89 fixed predictions are not exactly scene one")
    predicted_memory = predictions.get("scene_memory")
    if predicted_memory != report.get("scene_memory"):
        raise ValueError("V89 report and prediction scene-memory bindings differ")
    _authenticate_source_hashes(report, experiment)
    if training.get("source_hashes") != report.get("source_hashes"):
        raise ValueError("V89 training and evaluation source inventories differ")

    contract = _v89_contract(experiment, training, predictions)
    for source_contract in (V86_CONTRACT, V87_CONTRACT, V88_CONTRACT):
        load_bridge_source(source_contract)
    if (
        sha256_file(BASE_RUNTIME_CONFIG) != sources["runtime_config_sha256"]
        or sha256_file(V85_CHECKPOINT / "adapter.safetensors")
        != sources["frozen_v85_adapter_sha256"]
        or sha256_file(V85_CHECKPOINT / RUNTIME_METADATA_FILENAME)
        != sources["frozen_v85_metadata_sha256"]
    ):
        raise ValueError("V89 pinned runtime source bytes changed")
    bindings = _read_json(V89_BRIDGE_CANDIDATE / RUNTIME_METADATA_FILENAME).get(
        "bindings"
    )
    if (
        not isinstance(bindings, dict)
        or bindings.get("config_sha256") != EXPERIMENT_CONFIG_SHA256
        or bindings.get("preregistration_sha256") != PREREGISTRATION_SHA256
        or bindings.get("cpu_preflight_sha256") != CPU_PREFLIGHT_SHA256
        or bindings.get("v85_adapter_sha256")
        != sources["frozen_v85_adapter_sha256"]
        or bindings.get("v86_bridge_sha256") != V86_CONTRACT.weights_sha256
        or bindings.get("v87_bridge_sha256") != V87_CONTRACT.weights_sha256
        or bindings.get("v88_bridge_sha256") != V88_CONTRACT.weights_sha256
        or bindings.get("row_order_sha256")
        != experiment["training"]["row_order_sha256"]
        or bindings.get("training_inventory_sha256")
        != experiment["dataset"]["training_row_inventory_sha256"]
    ):
        raise ValueError("V89 bridge scalar source bindings changed")
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_config_sha256": EXPERIMENT_CONFIG_SHA256,
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "cpu_preflight_sha256": CPU_PREFLIGHT_SHA256,
        "training_report_sha256": dynamic_hashes["training_report_sha256"][0],
        "model_gate_report_sha256": sha256_file(MODEL_GATE_REPORT),
        "evaluation_predictions_sha256": dynamic_hashes[
            "evaluation_predictions_sha256"
        ][0],
        "v85_adapter_sha256": sources["frozen_v85_adapter_sha256"],
        "v86_bridge_file_sha256": V86_CONTRACT.weights_sha256,
        "v86_bridge_state_sha256": V86_STATE_SHA256,
        "v87_bridge_file_sha256": V87_CONTRACT.weights_sha256,
        "v87_bridge_state_sha256": V87_STATE_SHA256,
        "v88_bridge_file_sha256": V88_CONTRACT.weights_sha256,
        "v88_bridge_state_sha256": V88_STATE_SHA256,
        "v89_bridge_file_sha256": contract.weights_sha256,
        "v89_bridge_metadata_sha256": contract.metadata_sha256,
        "v89_bridge_state_sha256": contract.state_sha256,
        "model_acceptance_gate_passed": True,
        "model_gate_report_authenticated": True,
        "development_known_smoke_trained": True,
        "held_out_smoke_claim": False,
        "held_out_generalization_claim": False,
    }


def build_runtime_config_payload(evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Build the exact standalone eleven-bank runtime payload in memory."""

    parent = load_runtime_config(BASE_RUNTIME_CONFIG)
    parent.pop("_config_path", None)
    # The historical V85 YAML intentionally leaves one checkpoint-overwrite
    # bank's expected initialization digest null.  A standalone immutable
    # runtime must instead bind every frozen bank to the authenticated final
    # state carried by the exact V85 checkpoint metadata.
    parent_metadata = _read_json(V85_CHECKPOINT / RUNTIME_METADATA_FILENAME)
    parent_states = parent_metadata.get("lora_bank_state_sha256")
    parent_banks = parent.get("language", {}).get("lora_banks")
    if (
        not isinstance(parent_states, dict)
        or set(parent_states) != set(_BASE_BANKS)
        or not isinstance(parent_banks, dict)
        or tuple(parent_banks) != _BASE_BANKS
    ):
        raise ValueError("V89 authenticated V85 base-bank bindings changed")
    for name in _BASE_BANKS:
        state = _require_hash(parent_states.get(name), f"V85 {name} state")
        row = parent_banks[name]
        if not isinstance(row, dict) or row.get("trainable") is not False:
            raise ValueError(f"V89 V85 base bank is not frozen: {name}")
        row["expected_initial_state_sha256"] = state
    payload = extend_runtime_lora_config(
        parent_runtime_config=parent,
        added_bridges=_contracts_from_evidence(evidence),
        expected_final_banks=EXPECTED_BANKS,
    )
    banks = payload.get("language", {}).get("lora_banks")
    if not isinstance(banks, dict) or tuple(banks) != EXPECTED_BANKS:
        raise RuntimeError("V89 runtime payload lost exact eleven-bank order")
    return payload


def materialize_runtime_config(evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Create a sanitized standalone config only after gate authentication."""

    authenticated = authenticate_v89_model_gate()
    if dict(evidence) != authenticated:
        raise ValueError("V89 runtime-config evidence is not current")
    payload = build_runtime_config_payload(authenticated)
    encoded = yaml.safe_dump(payload, sort_keys=False, allow_unicode=False)
    if RUNTIME_CONFIG.exists():
        if (
            RUNTIME_CONFIG.is_symlink()
            or RUNTIME_CONFIG.read_text(encoding="utf-8") != encoded
        ):
            raise ValueError(
                "Existing V89 runtime config differs from authenticated gate"
            )
    else:
        with RUNTIME_CONFIG.open("x", encoding="utf-8") as handle:
            handle.write(encoded)
    loaded = load_runtime_config(RUNTIME_CONFIG)
    banks = loaded["language"]["lora_banks"]
    if tuple(banks) != EXPECTED_BANKS or any(
        row.get("trainable") is not False for row in banks.values()
    ):
        raise RuntimeError("Materialized V89 runtime config is not exactly frozen")
    return loaded


def _composed_adapter(
    evidence: Mapping[str, Any],
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    return compose_exact_bank_archive(
        base_checkpoint=V85_CHECKPOINT,
        expected_base_banks=_BASE_BANKS,
        added_bridges=_contracts_from_evidence(evidence),
        expected_final_banks=EXPECTED_BANKS,
    )


def _source_stack_sha256(tensors: Mapping[str, torch.Tensor]) -> str:
    source = {
        name: value
        for name, value in tensors.items()
        if not name.startswith("block_cross_residual.")
    }
    if not source or len(source) >= len(tensors):
        raise RuntimeError("V89 frozen source-stack inventory is invalid")
    return tensor_state_sha256(source)


def build_runtime_metadata(
    evidence: Mapping[str, Any],
    *,
    promotion: str,
    smoke_report_sha256: str | None,
) -> dict[str, Any]:
    """Build runtime-only metadata; no question, answer, or oracle fields."""

    if promotion not in {
        "pending_isolated_runtime_smoke",
        "strict_scene1_experimental_primary",
    }:
        raise ValueError("Unknown V89 runtime promotion state")
    if (promotion == "strict_scene1_experimental_primary") != (
        smoke_report_sha256 is not None
    ):
        raise ValueError("V89 promotion and smoke binding disagree")
    if smoke_report_sha256 is not None:
        _require_hash(smoke_report_sha256, "runtime smoke binding")
    config = build_runtime_config_payload(evidence)
    parent = _read_json(V85_CHECKPOINT / RUNTIME_METADATA_FILENAME)
    contracts = _contracts_from_evidence(evidence)
    metadata = extend_runtime_metadata(
        parent_metadata=parent,
        added_bridges=contracts,
        expected_final_banks=EXPECTED_BANKS,
    )
    # Mirror the standalone config's final-state binding in checkpoint
    # architecture metadata.  This intentionally upgrades V85's one legacy
    # null initialization digest without changing any tensor bytes.
    states = metadata["lora_bank_state_sha256"]
    for row in metadata["lora"]["banks"]:
        row["expected_initial_state_sha256"] = states[str(row["name"])]
    metadata["config_hash"] = config_hash(config)
    tensors, _composition = _composed_adapter(evidence)
    metadata["frozen_block_cross_source_stack_state_sha256"] = (
        _source_stack_sha256(tensors)
    )
    provenance = dict(metadata.get("initialization_provenance", {}))
    provenance["v89_strict_runtime_release"] = {
        "schema_version": SCHEMA_VERSION,
        "experiment_config_sha256": evidence["experiment_config_sha256"],
        "preregistration_sha256": evidence["preregistration_sha256"],
        "cpu_preflight_sha256": evidence["cpu_preflight_sha256"],
        "training_report_sha256": evidence["training_report_sha256"],
        "model_gate_report_sha256": evidence["model_gate_report_sha256"],
        "evaluation_predictions_sha256": evidence[
            "evaluation_predictions_sha256"
        ],
        "v86_bridge_state_sha256": V86_STATE_SHA256,
        "v87_bridge_state_sha256": V87_STATE_SHA256,
        "v88_bridge_state_sha256": V88_STATE_SHA256,
        "v89_bridge_state_sha256": evidence["v89_bridge_state_sha256"],
        "model_acceptance_gate_passed": True,
        "model_gate_report_authenticated": True,
        "promotion_decision": promotion,
        "runtime_promotion_authorized": promotion
        == "strict_scene1_experimental_primary",
        "smoke_report_sha256": smoke_report_sha256,
        "development_known_smoke_trained": True,
        "held_out_smoke_claim": False,
        "held_out_generalization_claim": False,
        "v75_comparator_retained": True,
    }
    metadata["initialization_provenance"] = provenance
    validate_runtime_checkpoint_metadata(metadata)
    validate_runtime_bank_inventory(
        runtime_config=config,
        checkpoint_metadata=metadata,
        expected_bank_order=EXPECTED_BANKS,
        expected_states={
            str(name): str(state)
            for name, state in metadata["lora_bank_state_sha256"].items()
        },
    )
    if (
        metadata["lora_parameter_count"] != EXPECTED_ADAPTER_PARAMETER_COUNT
        or metadata["lora_trainable_parameter_count"] != 0
    ):
        raise RuntimeError("V89 runtime parameter inventory changed")
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
            raise RuntimeError(
                "V89 runtime checkpoint is not an exact two-file package"
            )
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
    if (
        not SOURCE_MEMORY.is_dir()
        or SOURCE_MEMORY.is_symlink()
        or {item.name for item in SOURCE_MEMORY.iterdir()}
        != {MEMORY_FILENAME, METADATA_FILENAME}
    ):
        raise ValueError("Source V81 scene memory is not an exact two-file artifact")
    source_metadata = _read_json(SOURCE_MEMORY / METADATA_FILENAME)
    if (
        source_metadata.get("scene_id") != SCENE_ID
        or source_metadata.get("canonical_prefix_sha256")
        != SOURCE_MEMORY_PREFIX_SHA256
        or source_metadata.get("tensor_file_sha256")
        != SOURCE_MEMORY_TENSOR_FILE_SHA256
        or sha256_file(SOURCE_MEMORY / MEMORY_FILENAME)
        != SOURCE_MEMORY_TENSOR_FILE_SHA256
    ):
        raise ValueError("Source V81 scene-memory bytes changed")
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
            raise RuntimeError("V89 scene-memory bytes changed during rebinding")
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
        raise RuntimeError("V89 canonical scene prefix changed during rebinding")
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
    """Remove only an un-smoked, unpromoted partial V89 runtime candidate."""

    if (
        SMOKE_REPORT.exists()
        or RELEASE_REPORT.exists()
        or RELEASE_CHECKPOINT.exists()
        or RELEASE_MEMORY.exists()
    ):
        raise RuntimeError(
            "Refusing V89 cleanup after smoke or release evidence exists"
        )
    for root in (CANDIDATE_CHECKPOINT, CANDIDATE_MEMORY):
        if root.is_symlink():
            raise ValueError(f"Refusing to clean symbolic-link candidate: {root}")
        if root.exists():
            shutil.rmtree(root)
    if RUNTIME_CONFIG.exists():
        if RUNTIME_CONFIG.is_symlink():
            raise ValueError("Refusing to clean symbolic-link V89 runtime config")
        RUNTIME_CONFIG.unlink()


def prepare_candidate() -> dict[str, Any]:
    """Package only after the fixed V89 model gate has passed exactly."""

    if (
        CANDIDATE_CHECKPOINT.exists()
        or CANDIDATE_MEMORY.exists()
        or SMOKE_REPORT.exists()
        or RELEASE_CHECKPOINT.exists()
        or RELEASE_MEMORY.exists()
        or RELEASE_REPORT.exists()
    ):
        raise FileExistsError("V89 runtime candidate destination is not pristine")
    evidence = authenticate_v89_model_gate()
    config = materialize_runtime_config(evidence)
    metadata = build_runtime_metadata(
        evidence,
        promotion="pending_isolated_runtime_smoke",
        smoke_report_sha256=None,
    )
    checkpoint = _atomic_checkpoint(
        CANDIDATE_CHECKPOINT,
        metadata=metadata,
        evidence=evidence,
    )
    memory = _rebind_memory(
        CANDIDATE_MEMORY,
        checkpoint_sha256=str(checkpoint["checkpoint_sha256"]),
        runtime_config_sha256=effective_runtime_config_sha256(config),
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "phase": "v89_strict_runtime_candidate_prepared",
        "candidate_checkpoint": CANDIDATE_CHECKPOINT.relative_to(
            PROJECT_ROOT
        ).as_posix(),
        "candidate_memory": CANDIDATE_MEMORY.relative_to(PROJECT_ROOT).as_posix(),
        "runtime_config": RUNTIME_CONFIG.relative_to(PROJECT_ROOT).as_posix(),
        "runtime_config_sha256": effective_runtime_config_sha256(config),
        "checkpoint": checkpoint,
        "scene_memory": memory,
        "model_gate_evidence": evidence,
        "runtime_metadata_contains_supervision": False,
        "promotion_decision": "pending_isolated_runtime_smoke",
    }


def verify_candidate() -> dict[str, Any]:
    evidence = authenticate_v89_model_gate()
    if not CANDIDATE_CHECKPOINT.is_dir() or not CANDIDATE_MEMORY.is_dir():
        raise FileNotFoundError("V89 strict runtime candidate package is incomplete")
    expected = build_runtime_metadata(
        evidence,
        promotion="pending_isolated_runtime_smoke",
        smoke_report_sha256=None,
    )
    metadata = _read_json(CANDIDATE_CHECKPOINT / RUNTIME_METADATA_FILENAME)
    if metadata != expected:
        raise ValueError("V89 candidate runtime metadata changed")
    fingerprint, files = checkpoint_fingerprint(CANDIDATE_CHECKPOINT)
    candidate = load_file(
        str(CANDIDATE_CHECKPOINT / "adapter.safetensors"), device="cpu"
    )
    expected_tensors, composition = _composed_adapter(evidence)
    config = load_runtime_config(RUNTIME_CONFIG)
    expected_states = {
        str(name): str(value)
        for name, value in metadata["lora_bank_state_sha256"].items()
    }
    validate_runtime_bank_inventory(
        runtime_config=config,
        checkpoint_metadata=metadata,
        expected_bank_order=EXPECTED_BANKS,
        expected_states=expected_states,
    )
    checks = {
        "exact_two_file_checkpoint": {item["path"] for item in files}
        == {"adapter.safetensors", RUNTIME_METADATA_FILENAME},
        "exact_eleven_bank_order": composition["final_bank_order"]
        == list(EXPECTED_BANKS),
        "exact_eight_bridge_tensors_added": composition["added_tensor_count"] == 8,
        "v85_base_tensors_byte_identical": composition[
            "base_tensors_byte_identical"
        ]
        is True,
        "exact_tensor_inventory": set(candidate) == set(expected_tensors),
        "all_tensor_values_equal": set(candidate) == set(expected_tensors)
        and all(
            torch.equal(candidate[name], expected_tensors[name]) for name in candidate
        ),
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
        raise RuntimeError(f"V89 strict candidate verification failed: {checks}")
    return {
        "phase": "v89_strict_runtime_candidate_verified",
        "checkpoint_sha256": fingerprint,
        "v89_bridge_state_sha256": evidence["v89_bridge_state_sha256"],
        "checks": checks,
        "passed": True,
    }


def _normalized_answer(value: object) -> str:
    return str(value).strip().casefold().rstrip(".!?")


def validate_runtime_smoke_report_v89(
    smoke: Mapping[str, Any], evidence: Mapping[str, Any]
) -> None:
    gates = smoke.get("gates")
    expected_behavior = [
        {
            "question": question,
            "expected": expected,
            "observed": expected,
            "passed": True,
        }
        for question, expected in _SMOKE_CASES
    ]
    if (
        smoke.get("schema_version") != 89
        or smoke.get("artifact") != "gemma4_v89_strict_runtime_smoke_v1"
        or smoke.get("model_gate_report_sha256")
        != evidence.get("model_gate_report_sha256")
        or smoke.get("v89_bridge_state_sha256")
        != evidence.get("v89_bridge_state_sha256")
        or smoke.get("behavior_assertions_applied_after_chat_process_exit") is not True
        or smoke.get("expected_behavior_not_loaded_by_chat_runtime") is not True
        or smoke.get("behavior") != expected_behavior
        or smoke.get("prefix_hashes") != [SOURCE_MEMORY_PREFIX_SHA256] * 3
        or smoke.get("environment_conditioned_input_hashes")
        != [SOURCE_MEMORY_PREFIX_SHA256] * 3
        or not isinstance(gates, dict)
        or set(gates) != _REQUIRED_RUNTIME_GATES
        or any(value is not True for value in gates.values())
        or smoke.get("passed") is not True
        or smoke.get("promotion_authorized") is not True
        or smoke.get("development_known_smoke_trained") is not True
        or smoke.get("held_out_smoke_claim") is not False
        or smoke.get("held_out_generalization_claim") is not False
        or smoke.get("chat_log_sha256") != sha256_file(SMOKE_CHAT)
        or smoke.get("file_audit_sha256") != sha256_file(SMOKE_AUDIT)
    ):
        raise ValueError("V89 runtime smoke evidence did not pass exactly")


def run_smoke() -> dict[str, Any]:
    """Run an external candidate child while the oracle is physically renamed."""

    if SMOKE_REPORT.is_file():
        existing = _read_json(SMOKE_REPORT)
        validate_runtime_smoke_report_v89(existing, authenticate_v89_model_gate())
        return existing
    if SMOKE_CHAT.exists() or SMOKE_AUDIT.exists():
        raise FileExistsError("V89 smoke artifacts exist; results are create-once")
    evidence = authenticate_v89_model_gate()
    candidate = verify_candidate()
    oracle = PROJECT_ROOT / "data/oracle"
    unavailable = PROJECT_ROOT / f"data/.oracle-unavailable-v89-{os.getpid()}"
    python = PROJECT_ROOT / ".venv-gemma4/bin/python"
    if not python.is_file():
        raise FileNotFoundError("V89 local Gemma Python environment is unavailable")
    if not oracle.is_dir() or oracle.is_symlink() or unavailable.exists():
        raise FileNotFoundError("V89 oracle cannot be made physically unavailable")
    command = [
        str(python),
        "-m",
        "semantic_3d_chat.chat.v89_strict_scene1_cli",
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
        "--allow-candidate",
    ]
    for question, _expected in _SMOKE_CASES:
        command.extend(("--question", question))
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
            "V89 strict runtime smoke failed: "
            f"returncode={code}\nstdout={stdout}\nstderr={stderr}"
        )
    rows = [
        json.loads(line)
        for line in SMOKE_CHAT.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != len(_SMOKE_CASES) or not all(
        isinstance(row, dict) for row in rows
    ):
        raise RuntimeError("V89 smoke chat row count or shape changed")
    stdout_records: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            stdout_records.append(value)
    startup_records = [
        row for row in stdout_records if row.get("phase") == "v89_strict_scene1_ready"
    ]
    completion_records = [
        row for row in stdout_records if row.get("phase") == "v89_chat_audit_complete"
    ]
    startup = startup_records[0] if len(startup_records) == 1 else {}
    completion = completion_records[0] if len(completion_records) == 1 else {}
    behavior = [
        {
            "question": question,
            "expected": expected,
            "observed": _normalized_answer(row.get("answer")),
            "passed": _normalized_answer(row.get("answer")) == expected,
        }
        for row, (question, expected) in zip(rows, _SMOKE_CASES, strict=True)
    ]
    audit = _read_json(SMOKE_AUDIT)
    prefix_hashes = [row.get("prefix_hash") for row in rows]
    input_hashes = [row.get("environment_conditioned_input_sha256") for row in rows]
    gates = {
        "model_acceptance_gate_authenticated_and_passed": evidence[
            "model_acceptance_gate_passed"
        ]
        is True,
        "runtime_process_exit_zero": completed.returncode == 0,
        "three_behavior_assertions_pass": all(row["passed"] for row in behavior),
        "oracle_physically_unavailable": oracle_unavailable,
        "child_reported_oracle_unavailable_at_start": startup.get(
            "oracle_directory_available_at_runtime_start"
        )
        is False,
        "oracle_restored_after_runtime": oracle.is_dir(),
        "child_audit_completion_passed": completion.get("passed") is True,
        "child_loaded_no_training_or_evaluation_report": completion.get(
            "training_or_evaluation_report_loaded"
        )
        is False,
        "child_used_exact_eleven_frozen_banks": startup.get(
            "frozen_lora_bank_count"
        )
        == 11
        and startup.get("trainable_runtime_parameter_count") == 0
        and startup.get("v89_bridge_state_sha256")
        == evidence["v89_bridge_state_sha256"],
        "file_audit_forbidden_read_count_zero": not audit.get("forbidden_accesses"),
        "prefix_hash_identical_for_every_question": len(set(prefix_hashes)) == 1,
        "total_environment_conditioned_input_identical": len(set(input_hashes)) == 1,
        "prefix_and_environment_input_identical": prefix_hashes == input_hashes,
        "expected_immutable_scene_prefix": set(prefix_hashes)
        == {SOURCE_MEMORY_PREFIX_SHA256},
        "source_memory_bytes_unchanged": sha256_file(
            CANDIDATE_MEMORY / MEMORY_FILENAME
        )
        == SOURCE_MEMORY_TENSOR_FILE_SHA256,
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "artifact": "gemma4_v89_strict_runtime_smoke_v1",
        "model_gate_report_sha256": evidence["model_gate_report_sha256"],
        "v89_bridge_state_sha256": evidence["v89_bridge_state_sha256"],
        "candidate_checkpoint_sha256": candidate["checkpoint_sha256"],
        "candidate_adapter_sha256": sha256_file(
            CANDIDATE_CHECKPOINT / "adapter.safetensors"
        ),
        "candidate_memory_tensor_sha256": sha256_file(
            CANDIDATE_MEMORY / MEMORY_FILENAME
        ),
        "chat_log_sha256": sha256_file(SMOKE_CHAT),
        "file_audit_sha256": sha256_file(SMOKE_AUDIT),
        "behavior_assertions_applied_after_chat_process_exit": True,
        "expected_behavior_not_loaded_by_chat_runtime": True,
        "chat_process_stdout_sha256": hashlib.sha256(
            completed.stdout.encode()
        ).hexdigest(),
        "chat_process_stderr_sha256": hashlib.sha256(
            completed.stderr.encode()
        ).hexdigest(),
        "behavior": behavior,
        "prefix_hashes": prefix_hashes,
        "environment_conditioned_input_hashes": input_hashes,
        "gates": gates,
        "passed": all(gates.values()),
        "promotion_authorized": all(gates.values()),
        "development_known_smoke_trained": True,
        "held_out_smoke_claim": False,
        "held_out_generalization_claim": False,
    }
    _write_json(SMOKE_REPORT, report)
    return report


def promote_release() -> dict[str, Any]:
    evidence = authenticate_v89_model_gate()
    if (
        RELEASE_CHECKPOINT.exists()
        or RELEASE_MEMORY.exists()
        or RELEASE_REPORT.exists()
    ):
        raise FileExistsError("V89 strict runtime release destination already exists")
    smoke = _read_json(SMOKE_REPORT)
    validate_runtime_smoke_report_v89(smoke, evidence)
    candidate = verify_candidate()
    if (
        smoke.get("candidate_checkpoint_sha256") != candidate["checkpoint_sha256"]
        or smoke.get("candidate_adapter_sha256")
        != sha256_file(CANDIDATE_CHECKPOINT / "adapter.safetensors")
        or smoke.get("candidate_memory_tensor_sha256")
        != sha256_file(CANDIDATE_MEMORY / MEMORY_FILENAME)
    ):
        raise ValueError("V89 smoked candidate bytes changed before promotion")
    smoke_sha256 = sha256_file(SMOKE_REPORT)
    metadata = build_runtime_metadata(
        evidence,
        promotion="strict_scene1_experimental_primary",
        smoke_report_sha256=smoke_sha256,
    )
    checkpoint = _atomic_checkpoint(
        RELEASE_CHECKPOINT,
        metadata=metadata,
        evidence=evidence,
        source_adapter=CANDIDATE_CHECKPOINT / "adapter.safetensors",
    )
    if checkpoint["adapter_sha256"] != sha256_file(
        CANDIDATE_CHECKPOINT / "adapter.safetensors"
    ):
        raise RuntimeError("Promoted V89 adapter differs from smoked candidate")
    config = load_runtime_config(RUNTIME_CONFIG)
    memory = _rebind_memory(
        RELEASE_MEMORY,
        checkpoint_sha256=str(checkpoint["checkpoint_sha256"]),
        runtime_config_sha256=effective_runtime_config_sha256(config),
    )
    release = {
        "schema_version": SCHEMA_VERSION,
        "artifact": "gemma4_v89_strict_runtime_release_v1",
        "promotion_decision": "strict_scene1_experimental_primary",
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
            "question_dependent_retrieval": False,
            "environmental_text_inputs": [],
        },
        "development_known_smoke_trained": True,
        "held_out_smoke_claim": False,
        "held_out_generalization_claim": False,
        "v75_comparator_retained": True,
        "runtime_config": RUNTIME_CONFIG.relative_to(PROJECT_ROOT).as_posix(),
        "runtime_config_sha256": effective_runtime_config_sha256(config),
        "checkpoint": checkpoint,
        "scene_memory": memory,
        "bindings": {**evidence, "runtime_smoke_sha256": smoke_sha256},
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
    evidence = authenticate_v89_model_gate()
    release = _read_json(RELEASE_REPORT)
    metadata = _read_json(RELEASE_CHECKPOINT / RUNTIME_METADATA_FILENAME)
    validate_runtime_checkpoint_metadata(metadata)
    fingerprint, files = checkpoint_fingerprint(RELEASE_CHECKPOINT)
    config = load_runtime_config(RUNTIME_CONFIG)
    loaded = load_v81_scene_memory(
        RELEASE_MEMORY,
        expected_scene_id=SCENE_ID,
        expected_base_checkpoint_sha256=fingerprint,
        expected_runtime_config_sha256=effective_runtime_config_sha256(config),
        expected_model_device="cpu",
    )
    provenance = metadata["initialization_provenance"][
        "v89_strict_runtime_release"
    ]
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
        "release_report_promoted": release.get("promotion_decision")
        == "strict_scene1_experimental_primary",
        "exact_two_file_checkpoint": {item["path"] for item in files}
        == {"adapter.safetensors", RUNTIME_METADATA_FILENAME},
        "checkpoint_fingerprint_matches_release": fingerprint
        == release["checkpoint"]["checkpoint_sha256"],
        "adapter_matches_smoked_candidate": sha256_file(
            RELEASE_CHECKPOINT / "adapter.safetensors"
        )
        == sha256_file(CANDIDATE_CHECKPOINT / "adapter.safetensors"),
        "memory_bytes_match_source": sha256_file(RELEASE_MEMORY / MEMORY_FILENAME)
        == SOURCE_MEMORY_TENSOR_FILE_SHA256,
        "memory_prefix_match_source": loaded.metadata["canonical_prefix_sha256"]
        == SOURCE_MEMORY_PREFIX_SHA256,
        "model_gate_binding_exact": provenance["model_gate_report_sha256"]
        == evidence["model_gate_report_sha256"],
        "runtime_smoke_binding_exact": provenance["smoke_report_sha256"]
        == sha256_file(SMOKE_REPORT)
        == release["bindings"]["runtime_smoke_sha256"],
        "v89_state_binding_exact": provenance["v89_bridge_state_sha256"]
        == evidence["v89_bridge_state_sha256"],
        "exact_eleven_frozen_banks": tuple(
            row["name"] for row in metadata["lora"]["banks"]
        )
        == EXPECTED_BANKS
        and metadata["lora"]["trainable_adapter_parameter_count"] == 0,
        "runtime_promotion_authorized": provenance[
            "runtime_promotion_authorized"
        ]
        is True,
    }
    if not all(checks.values()):
        raise RuntimeError(f"V89 strict runtime release verification failed: {checks}")
    return {
        "phase": "v89_strict_runtime_release_verified",
        "checks": checks,
        "passed": True,
    }


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
        "authenticate": authenticate_v89_model_gate,
        "prepare": prepare_candidate,
        "verify-candidate": verify_candidate,
        "smoke": run_smoke,
        "promote": promote_release,
        "verify": verify_release,
        "cleanup-failed-candidate": lambda: (
            cleanup_failed_candidate()
            or {"phase": "v89_failed_candidate_cleaned", "passed": True}
        ),
    }
    try:
        result = functions[args.command]()
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"V89 strict runtime {args.command} refused: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 2 if args.command == "smoke" and result.get("passed") is not True else 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CANDIDATE_CHECKPOINT",
    "CANDIDATE_MEMORY",
    "MODEL_GATE_REPORT",
    "RELEASE_CHECKPOINT",
    "RELEASE_MEMORY",
    "RUNTIME_CONFIG",
    "SMOKE_REPORT",
    "V86_CONTRACT",
    "V87_CONTRACT",
    "V88_CONTRACT",
    "authenticate_v89_model_gate",
    "build_runtime_config_payload",
    "build_runtime_metadata",
    "cleanup_failed_candidate",
    "main",
    "materialize_runtime_config",
    "prepare_candidate",
    "promote_release",
    "run_smoke",
    "validate_model_gate_contract_v89",
    "validate_runtime_smoke_report_v89",
    "verify_candidate",
    "verify_release",
]
