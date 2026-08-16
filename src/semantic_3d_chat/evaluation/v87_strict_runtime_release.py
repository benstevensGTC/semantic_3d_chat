"""Package, isolate-smoke, and promote V87's strict direct-memory runtime.

This is an evaluation/release surface, never a chat dependency.  The child chat
process receives only a standalone sanitized runtime YAML, an exact two-file
checkpoint, and an exact two-file continuous scene-memory artifact.  Semantic
smoke expectations are applied here only after that child has exited.
"""

from __future__ import annotations

import argparse
import copy
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
from semantic_3d_chat.chat.v87_strict_scene1_runtime import (
    V86_BANK,
    V86_STATE_SHA256,
    V86_TARGET,
    V87_BANK,
    V87_TARGET,
)
from semantic_3d_chat.config import PROJECT_ROOT, config_hash
from semantic_3d_chat.evaluation.prediction_artifacts import checkpoint_fingerprint
from semantic_3d_chat.evaluation.v85_strict_runtime_release import (
    CANDIDATE_CHECKPOINT as V85_RUNTIME_CHECKPOINT,
)
from semantic_3d_chat.evaluation.v85_strict_runtime_release import (
    verify_candidate as verify_v85_candidate,
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

SCHEMA_VERSION: Final[int] = 87
SOURCE_MEMORY_PREFIX_SHA256: Final[str] = (
    "a428f5147c815839ae7315a0adab952ab210814fb21dcdc5bf13b167f28a6e37"
)
SOURCE_MEMORY_TENSOR_FILE_SHA256: Final[str] = (
    "3427851432d8f2a3609b6205b18b8c0d9a0fcf68d8f3bf0c98c758ac64209ffb"
)
V86_PARAMETER_COUNT: Final[int] = 110_592
V87_PARAMETER_COUNT: Final[int] = 110_592
_SHA256: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{64}")
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
        "child_used_exact_nine_frozen_banks",
        "file_audit_forbidden_read_count_zero",
        "prefix_hash_identical_for_every_question",
        "total_environment_conditioned_input_identical",
        "prefix_and_environment_input_identical",
        "expected_immutable_scene_prefix",
        "source_memory_bytes_unchanged",
    }
)

EXPERIMENT_CONFIG: Final[Path] = (
    PROJECT_ROOT / "configs/experiments/gemma4_v87_scene1_balanced_demo.yaml"
)
BASE_RUNTIME_CONFIG: Final[Path] = (
    PROJECT_ROOT / "configs/runtime/gemma4_v85_strict_multiscene.yaml"
)
RUNTIME_CONFIG: Final[Path] = (
    PROJECT_ROOT / "configs/runtime/gemma4_v87_strict_scene1.yaml"
)
V86_BRIDGE_CANDIDATE: Final[Path] = (
    PROJECT_ROOT / "reports/gemma4/artifacts/v86_scene1_demo_final"
)
V87_BRIDGE_CANDIDATE: Final[Path] = (
    PROJECT_ROOT / "reports/gemma4/artifacts/v87_scene1_balanced_final"
)
V87_MODEL_GATE_REPORT: Final[Path] = (
    PROJECT_ROOT / "reports/gemma4/metrics/gemma4_v87_scene1_balanced_evaluation.json"
)
SOURCE_MEMORY: Final[Path] = (
    PROJECT_ROOT / "data_gemma4/runtime/scene_memories/v81/scene_000001"
)
CANDIDATE_CHECKPOINT: Final[Path] = (
    PROJECT_ROOT / "reports/gemma4/artifacts/v87_strict_runtime_candidate"
)
CANDIDATE_MEMORY: Final[Path] = (
    PROJECT_ROOT / "reports/gemma4/artifacts/v87_strict_runtime_candidate_memory/scene_000001"
)
RELEASE_CHECKPOINT: Final[Path] = (
    PROJECT_ROOT / "data_gemma4/runtime/checkpoints/gemma4_v87_strict_scene1_release_v1"
)
RELEASE_MEMORY: Final[Path] = (
    PROJECT_ROOT / "data_gemma4/runtime/scene_memories/v87/scene_000001"
)
SMOKE_CHAT: Final[Path] = (
    PROJECT_ROOT / "reports/gemma4/examples/v87_strict_runtime_smoke.jsonl"
)
SMOKE_AUDIT: Final[Path] = (
    PROJECT_ROOT / "reports/gemma4/metrics/v87_strict_runtime_smoke_access.json"
)
SMOKE_REPORT: Final[Path] = (
    PROJECT_ROOT / "reports/gemma4/metrics/v87_strict_runtime_smoke.json"
)
RELEASE_REPORT: Final[Path] = (
    PROJECT_ROOT / "reports/gemma4/metrics/gemma4_v87_strict_runtime_release.json"
)

# Evaluation-only assertions.  The chat CLI cannot import this module.
_SMOKE_CASES: Final[tuple[tuple[str, str], ...]] = (
    ("Is there a chair?", "yes"),
    ("What color is the bowl?", "red"),
    ("Is the bowl left or right of the chair?", "left"),
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected one JSON object: {path}")
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def cleanup_failed_candidate() -> None:
    """Remove only unpromoted V87 runtime outputs after a failed preparation.

    The operation is deliberately unavailable once a smoke or release record
    exists.  This gives an explicit recovery path for partial I/O failures
    without weakening the create-once behavioral evidence.
    """

    if SMOKE_REPORT.exists() or RELEASE_REPORT.exists() or RELEASE_CHECKPOINT.exists():
        raise RuntimeError("Refusing cleanup after V87 behavioral or release evidence exists")
    for root in (CANDIDATE_CHECKPOINT, CANDIDATE_MEMORY):
        if root.is_symlink():
            raise ValueError(f"Refusing to clean a symbolic-link V87 candidate: {root}")
        if root.exists():
            shutil.rmtree(root)
    if RUNTIME_CONFIG.exists():
        if RUNTIME_CONFIG.is_symlink():
            raise ValueError("Refusing to clean a symbolic-link V87 runtime config")
        RUNTIME_CONFIG.unlink()


def _load_experiment() -> dict[str, Any]:
    payload = yaml.safe_load(EXPERIMENT_CONFIG.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != {"v87"}:
        raise ValueError("V87 experiment config identity changed")
    experiment = payload["v87"]
    if not isinstance(experiment, dict):
        raise TypeError("V87 experiment config must contain one mapping")
    dataset = experiment.get("dataset")
    if (
        experiment.get("schema_version") != 87
        or experiment.get("artifact")
        != "gemma4_v87_scene1_balanced_direct_memory_overfit_v1"
        or experiment.get("status") != "preregistered_before_full_model_load"
        or not isinstance(dataset, dict)
        or dataset.get("scene_id") != "scene_000001"
        or dataset.get("row_count") != 138
    ):
        raise ValueError("V87 sealed single-scene experiment identity changed")
    if "REPLACE_" in EXPERIMENT_CONFIG.read_text(encoding="utf-8"):
        raise ValueError("V87 experiment config still contains unsealed hash placeholders")
    return experiment


def _require_hash(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"V87 {field} is not a lowercase SHA-256 digest")
    return value


def _bridge_state(
    root: Path,
    *,
    artifact: str,
    bank: str,
    target: str,
    expected_state: str | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    metadata = _read_json(root / "runtime_metadata.json")
    weights = root / "bridge.safetensors"
    if (
        metadata.get("artifact") != artifact
        or metadata.get("status")
        != "fixed_final_awaiting_preregistered_acceptance_gates"
        or metadata.get("bank_name") != bank
        or metadata.get("target_module") != target
        or metadata.get("rank") != 8
        or float(metadata.get("alpha", -1.0)) != 16.0
        or float(metadata.get("dropout", -1.0)) != 0.0
        or metadata.get("parameter_count") != 110_592
        or metadata.get("weights_sha256") != sha256_file(weights)
        or metadata.get("environmental_memory_serialized") is not False
        or metadata.get("questions_or_answers_serialized") is not False
        or metadata.get("oracle_serialized") is not False
        or metadata.get("evaluation_scored") is not False
        or metadata.get("runtime_promotion_authorized") is not False
    ):
        raise ValueError(f"V87 {bank} source candidate authentication failed")
    archive = load_file(str(weights), device="cpu")
    if set(archive) != {"lora_a", "lora_b"}:
        raise ValueError(f"V87 {bank} source tensor inventory changed")
    if (
        tuple(archive["lora_a"].shape) != (8, 1_536)
        or tuple(archive["lora_b"].shape) != (12_288, 8)
        or archive["lora_a"].dtype != torch.float32
        or archive["lora_b"].dtype != torch.float32
    ):
        raise ValueError(f"V87 {bank} source tensor shape or dtype changed")
    state = {
        "adapters.0.lora_a": archive["lora_a"].contiguous(),
        "adapters.0.lora_b": archive["lora_b"].contiguous(),
    }
    observed = tensor_state_sha256(state)
    if metadata.get("state_sha256") != observed or (
        expected_state is not None and observed != expected_state
    ):
        raise ValueError(f"V87 {bank} source tensor state changed")
    return state, metadata


def validate_model_gate_contract(report: Mapping[str, Any]) -> None:
    """Fail closed unless every preregistered model-level gate is exactly true."""

    metrics = report.get("metrics")
    gates = metrics.get("model_acceptance_gates") if isinstance(metrics, dict) else None
    if (
        report.get("artifact") != "gemma4_v87_scene1_balanced_evaluation_v1"
        or report.get("schema_version") != 87
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
        or report.get("held_out_generalization_claim") is not False
        or report.get("parent_v86_mutated") is not False
        or report.get("oracle_loaded") is not False
        or report.get("official_validation_loaded") is not False
        or report.get("official_test_loaded") is not False
        or report.get("deferred_final_loaded") is not False
    ):
        raise ValueError("V87 model-level acceptance report did not pass exactly")
    canonical = metrics.get("canonical_type_specific")
    by_type = metrics.get("canonical_accuracy_by_answer_type")
    causal = metrics.get("causal_control")
    smoke = metrics.get("generic_smoke")
    type_rows = (
        [by_type.get(name) for name in ("attribute", "presence", "spatial_relation")]
        if isinstance(by_type, dict)
        else []
    )
    smoke_records = smoke.get("records") if isinstance(smoke, dict) else None
    smoke_contract = (
        [
            (
                record.get("question"),
                record.get("expected"),
                record.get("normalized_prediction"),
                record.get("exact_correct"),
            )
            for record in smoke_records
        ]
        if isinstance(smoke_records, list)
        and all(isinstance(record, dict) for record in smoke_records)
        else None
    )
    if (
        not isinstance(canonical, dict)
        or canonical.get("total") != 138
        or int(canonical.get("correct", -1)) < 111
        or float(canonical.get("accuracy", -1.0)) < 0.80
        or not isinstance(by_type, dict)
        or len(type_rows) != 3
        or not all(isinstance(row, dict) for row in type_rows)
        or float(type_rows[0].get("accuracy", -1.0)) < 0.50
        or float(type_rows[1].get("accuracy", -1.0)) < 0.75
        or float(type_rows[2].get("accuracy", -1.0)) < 0.60
        or not isinstance(causal, dict)
        or float(causal.get("mean_zero_minus_correct_nll", 0.0)) <= 0.0
        or int(causal.get("canonical_prediction_changes", 0)) < 1
        or not isinstance(smoke, dict)
        or smoke.get("correct") != 3
        or smoke.get("total") != 3
        or float(smoke.get("accuracy", 0.0)) != 1.0
        or smoke_contract
        != [(question, expected, expected, True) for question, expected in _SMOKE_CASES]
    ):
        raise ValueError("V87 numeric model metrics contradict their passing gates")
    leakage = report.get("leakage")
    scene_memory = report.get("scene_memory")
    if (
        not isinstance(leakage, dict)
        or leakage.get("protected_read_count") != 0
        or leakage.get("protected_reads") != []
        or leakage.get("oracle_loaded") is not False
        or not isinstance(scene_memory, dict)
        or scene_memory.get("prefix_hash_invariant") is not True
        or scene_memory.get("same_prefix_reused_for_every_question") is not True
        or scene_memory.get("question_derived_environmental_tokens") != 0
        or scene_memory.get("prefix_sha256_before") != SOURCE_MEMORY_PREFIX_SHA256
        or scene_memory.get("prefix_sha256_after") != SOURCE_MEMORY_PREFIX_SHA256
    ):
        raise ValueError("V87 model-level leakage or memory gate changed")


def authenticate_v87_model_gate() -> dict[str, Any]:
    """Authenticate the create-once V87 gate and every artifact it binds."""

    experiment = _load_experiment()
    outputs = experiment["outputs"]
    sources = experiment["sources"]
    expected_report = (PROJECT_ROOT / str(outputs["evaluation_report"])).resolve()
    if expected_report != V87_MODEL_GATE_REPORT.resolve():
        raise ValueError("V87 model-gate report path changed")
    report = _read_json(V87_MODEL_GATE_REPORT)
    if report.get("preregistered_gates") != experiment.get("gates"):
        raise ValueError("V87 model gate is not bound to the sealed acceptance gates")
    validate_model_gate_contract(report)

    config_digest = sha256_file(EXPERIMENT_CONFIG)
    prereg = PROJECT_ROOT / str(outputs["preregistration"])
    preflight = PROJECT_ROOT / str(outputs["cpu_preflight"])
    training = PROJECT_ROOT / str(outputs["training_report"])
    predictions = PROJECT_ROOT / str(outputs["evaluation_predictions"])
    bound_files = {
        "config_sha256": (config_digest, report.get("config_sha256")),
        "preregistration_sha256": (sha256_file(prereg), report.get("preregistration_sha256")),
        "cpu_preflight_sha256": (sha256_file(preflight), report.get("cpu_preflight_sha256")),
        "training_report_sha256": (sha256_file(training), report.get("training_report_sha256")),
        "evaluation_predictions_sha256": (
            sha256_file(predictions),
            report.get("evaluation_predictions_sha256"),
        ),
    }
    mismatches = {
        key: {"observed": observed, "bound": bound}
        for key, (observed, bound) in bound_files.items()
        if observed != bound or _SHA256.fullmatch(observed) is None
    }
    if mismatches:
        raise ValueError(f"V87 model-gate bound artifact changed: {mismatches}")
    if report.get("evaluation_predictions_path") != str(outputs["evaluation_predictions"]):
        raise ValueError("V87 model-gate predictions path changed")
    if config_digest != _require_hash(report["config_sha256"], "config binding"):
        raise ValueError("V87 experiment config changed after scoring")
    source_hashes = report.get("source_hashes")
    if not isinstance(source_hashes, dict) or not source_hashes:
        raise ValueError("V87 model-gate authenticated source inventory is missing")
    source_mismatches: dict[str, Any] = {}
    for relative, expected in source_hashes.items():
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise TypeError("V87 model-gate source-hash inventory is malformed")
        if relative == "gemma_model_blob_sha256_identity":
            if expected != sources["model_blob_sha256_identity"]:
                source_mismatches[relative] = {
                    "expected": sources["model_blob_sha256_identity"],
                    "observed": expected,
                }
            continue
        source_path = (PROJECT_ROOT / relative).resolve()
        try:
            source_path.relative_to(PROJECT_ROOT.resolve())
        except ValueError as error:
            raise ValueError("V87 model-gate source escaped the project root") from error
        observed = sha256_file(source_path) if source_path.is_file() else None
        if observed != expected:
            source_mismatches[relative] = {"expected": expected, "observed": observed}
    if source_mismatches:
        raise ValueError(f"V87 model-gate authenticated sources changed: {source_mismatches}")

    prediction_payload = _read_json(predictions)
    candidate_binding = prediction_payload.get("candidate")
    v87_state, v87_metadata = _bridge_state(
        V87_BRIDGE_CANDIDATE,
        artifact="gemma4_v87_scene1_balanced_fixed_final_v1",
        bank=V87_BANK,
        target=V87_TARGET,
    )
    v87_state_sha256 = tensor_state_sha256(v87_state)
    if (
        prediction_payload.get("artifact")
        != "gemma4_v87_scene1_balanced_predictions_v1"
        or prediction_payload.get("schema_version") != 87
        or prediction_payload.get("row_count") != 138
        or prediction_payload.get("scene_count") != 1
        or prediction_payload.get("runtime_promotion_authorized") is not False
        or not isinstance(candidate_binding, dict)
        or candidate_binding.get("path") != str(outputs["fixed_final_candidate"])
        or candidate_binding.get("weights_sha256") != v87_metadata["weights_sha256"]
        or candidate_binding.get("state_sha256") != v87_state_sha256
        or candidate_binding.get("optimizer_updates") != 184
    ):
        raise ValueError("V87 scored predictions are not bound to the fixed candidate")
    rows = prediction_payload.get("records")
    if (
        not isinstance(rows, list)
        or len(rows) != 138
        or any(
            not isinstance(row, dict) or row.get("scene_id") != "scene_000001"
            for row in rows
        )
    ):
        raise ValueError("V87 fixed model gate includes a non-scene-one record")

    v86_state, v86_metadata = _bridge_state(
        V86_BRIDGE_CANDIDATE,
        artifact="gemma4_v86_scene1_demo_fixed_final_v1",
        bank=V86_BANK,
        target=V86_TARGET,
        expected_state=V86_STATE_SHA256,
    )
    if tensor_state_sha256(v86_state) != V86_STATE_SHA256:
        raise ValueError("V86 parent bridge state changed")
    if (
        sha256_file(V86_BRIDGE_CANDIDATE / "bridge.safetensors")
        != sources["parent_v86_bridge_sha256"]
        or sha256_file(V86_BRIDGE_CANDIDATE / "runtime_metadata.json")
        != sources["parent_v86_metadata_sha256"]
    ):
        raise ValueError("V87 pinned V86 parent bytes changed")

    verify_v85_candidate()
    if (
        sha256_file(V85_RUNTIME_CHECKPOINT / "adapter.safetensors")
        != sources["frozen_v85_adapter_sha256"]
        or sha256_file(V85_RUNTIME_CHECKPOINT / RUNTIME_METADATA_FILENAME)
        != sources["frozen_v85_metadata_sha256"]
        or sha256_file(BASE_RUNTIME_CONFIG) != sources["runtime_config_sha256"]
    ):
        raise ValueError("V87 pinned V85 runtime bytes changed")
    return {
        "experiment_config_sha256": config_digest,
        "preregistration_sha256": bound_files["preregistration_sha256"][0],
        "cpu_preflight_sha256": bound_files["cpu_preflight_sha256"][0],
        "training_report_sha256": bound_files["training_report_sha256"][0],
        "model_gate_report_sha256": sha256_file(V87_MODEL_GATE_REPORT),
        "evaluation_predictions_sha256": bound_files[
            "evaluation_predictions_sha256"
        ][0],
        "v85_adapter_sha256": sha256_file(
            V85_RUNTIME_CHECKPOINT / "adapter.safetensors"
        ),
        "v86_bridge_file_sha256": v86_metadata["weights_sha256"],
        "v86_bridge_state_sha256": V86_STATE_SHA256,
        "v87_bridge_file_sha256": v87_metadata["weights_sha256"],
        "v87_bridge_state_sha256": v87_state_sha256,
        "model_acceptance_gate_passed": True,
        "model_gate_report_authenticated": True,
    }


def build_runtime_config_payload(v87_state_sha256: str) -> dict[str, Any]:
    """Create a standalone nine-bank payload without evaluation vocabulary."""

    _require_hash(v87_state_sha256, "runtime V87 state")
    payload = yaml.safe_load(BASE_RUNTIME_CONFIG.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("V85 runtime config is malformed")
    result = copy.deepcopy(payload)
    banks = result["language"]["lora_banks"]
    if not isinstance(banks, dict) or len(banks) != 7:
        raise ValueError("V87 runtime requires the exact seven-bank V85 base config")
    banks[V86_BANK] = {
        "trainable": False,
        "rank": 8,
        "alpha": 16.0,
        "dropout": 0.0,
        "initialization_algorithm": "checkpoint_overwrite",
        "initialization_seed": None,
        "expected_initial_state_sha256": V86_STATE_SHA256,
        "target_modules": [V86_TARGET],
    }
    banks[V87_BANK] = {
        "trainable": False,
        "rank": 8,
        "alpha": 16.0,
        "dropout": 0.0,
        "initialization_algorithm": "checkpoint_overwrite",
        "initialization_seed": None,
        "expected_initial_state_sha256": v87_state_sha256,
        "target_modules": [V87_TARGET],
    }
    return result


def materialize_runtime_config(evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Materialize only after the final V87 state and gate report authenticate."""

    state = _require_hash(evidence.get("v87_bridge_state_sha256"), "bridge state")
    payload = build_runtime_config_payload(state)
    encoded = yaml.safe_dump(payload, sort_keys=False, allow_unicode=False)
    if RUNTIME_CONFIG.exists():
        if RUNTIME_CONFIG.read_text(encoding="utf-8") != encoded:
            raise ValueError("Existing V87 standalone runtime config differs from final gate")
    else:
        with RUNTIME_CONFIG.open("x", encoding="utf-8") as handle:
            handle.write(encoded)
    loaded = load_runtime_config(RUNTIME_CONFIG)
    banks = loaded["language"]["lora_banks"]
    if len(banks) != 9 or banks[V87_BANK]["expected_initial_state_sha256"] != state:
        raise RuntimeError("Materialized V87 standalone runtime config changed")
    return loaded


def _merged_adapter() -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    base = load_file(
        str(V85_RUNTIME_CHECKPOINT / "adapter.safetensors"), device="cpu"
    )
    merged = {name: value.detach().cpu().contiguous() for name, value in base.items()}
    additions: dict[str, str] = {}
    for root, artifact, bank, target, expected in (
        (
            V86_BRIDGE_CANDIDATE,
            "gemma4_v86_scene1_demo_fixed_final_v1",
            V86_BANK,
            V86_TARGET,
            V86_STATE_SHA256,
        ),
        (
            V87_BRIDGE_CANDIDATE,
            "gemma4_v87_scene1_balanced_fixed_final_v1",
            V87_BANK,
            V87_TARGET,
            None,
        ),
    ):
        state, metadata = _bridge_state(
            root,
            artifact=artifact,
            bank=bank,
            target=target,
            expected_state=expected,
        )
        for suffix, value in state.items():
            key = f"lora_banks.{bank}.{suffix}"
            if key in merged:
                raise ValueError(f"V87 runtime bridge key already exists: {key}")
            merged[key] = value
        additions[bank] = str(metadata["state_sha256"])
    retained = {name: merged[name] for name in base}
    if tensor_state_sha256(retained) != tensor_state_sha256(base):
        raise RuntimeError("V85 candidate tensor bytes changed while adding V86/V87")
    return merged, {
        "v85_tensor_count": len(base),
        "packaged_tensor_count": len(merged),
        "v85_base_tensors_byte_identical": True,
        "added_bank_state_sha256": additions,
    }


def _source_stack_sha256(tensors: Mapping[str, torch.Tensor]) -> str:
    source = {
        name: value
        for name, value in tensors.items()
        if not name.startswith("block_cross_residual.")
    }
    if not source or len(source) >= len(tensors):
        raise RuntimeError("V87 frozen source-stack inventory is invalid")
    return tensor_state_sha256(source)


def build_runtime_metadata(
    evidence: Mapping[str, Any],
    *,
    promotion: str,
    smoke_report_sha256: str | None,
) -> dict[str, Any]:
    config = materialize_runtime_config(evidence)
    metadata = _read_json(V85_RUNTIME_CHECKPOINT / RUNTIME_METADATA_FILENAME)
    metadata["config_hash"] = config_hash(config)
    banks = [dict(record) for record in metadata["lora"]["banks"]]
    states = dict(metadata["lora_bank_state_sha256"])
    modules = dict(metadata["lora_bank_wrapped_modules"])
    counts = dict(metadata["lora_bank_parameter_counts"])
    for name, target, state, parameter_count in (
        (V86_BANK, V86_TARGET, V86_STATE_SHA256, V86_PARAMETER_COUNT),
        (
            V87_BANK,
            V87_TARGET,
            _require_hash(evidence.get("v87_bridge_state_sha256"), "bridge state"),
            V87_PARAMETER_COUNT,
        ),
    ):
        banks.append(
            {
                "name": name,
                "trainable": False,
                "rank": 8,
                "alpha": 16.0,
                "dropout": 0.0,
                "target_modules": [target],
                "initialization_algorithm": "checkpoint_overwrite",
                "initialization_seed": None,
                "expected_initial_state_sha256": state,
                "adapter_parameter_count": parameter_count,
            }
        )
        states[name] = state
        modules[name] = [target]
        counts[name] = {target: parameter_count}
    total = int(metadata["lora"]["adapter_parameter_count"]) + 2 * 110_592
    metadata["lora"] = {
        "schema_version": 2,
        "enabled": True,
        "banks": banks,
        "adapter_parameter_count": total,
        "trainable_adapter_parameter_count": 0,
    }
    metadata["lora_bank_state_sha256"] = states
    metadata["lora_bank_wrapped_modules"] = modules
    metadata["lora_bank_parameter_counts"] = counts
    metadata["lora_parameter_count"] = total
    metadata["lora_trainable_parameter_count"] = 0
    tensors, _inheritance = _merged_adapter()
    metadata["frozen_block_cross_source_stack_state_sha256"] = (
        _source_stack_sha256(tensors)
    )
    provenance = dict(metadata["initialization_provenance"])
    provenance["v87_strict_runtime_release"] = {
        "schema_version": SCHEMA_VERSION,
        "experiment_config_sha256": evidence["experiment_config_sha256"],
        "preregistration_sha256": evidence["preregistration_sha256"],
        "cpu_preflight_sha256": evidence["cpu_preflight_sha256"],
        "training_report_sha256": evidence["training_report_sha256"],
        "model_gate_report_sha256": evidence["model_gate_report_sha256"],
        "v86_bridge_state_sha256": V86_STATE_SHA256,
        "v87_bridge_state_sha256": evidence["v87_bridge_state_sha256"],
        "model_acceptance_gate_passed": True,
        "model_gate_report_authenticated": True,
        "promotion_decision": promotion,
        "runtime_promotion_authorized": promotion
        == "strict_scene1_experimental_primary",
        "smoke_report_sha256": smoke_report_sha256,
        "v75_comparator_retained": True,
        "held_out_generalization_claim": False,
    }
    metadata["initialization_provenance"] = provenance
    validate_runtime_checkpoint_metadata(metadata)
    return metadata


def _atomic_checkpoint(
    destination: Path,
    *,
    metadata: Mapping[str, Any],
    source_adapter: Path | None = None,
) -> dict[str, Any]:
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}.", dir=destination.parent
    ) as raw:
        temporary = Path(raw)
        if source_adapter is None:
            tensors, inheritance = _merged_adapter()
            save_file(tensors, str(temporary / "adapter.safetensors"))
        else:
            shutil.copyfile(source_adapter, temporary / "adapter.safetensors")
            inheritance = {"candidate_adapter_bytes_reused_exactly": True}
        _write_json(temporary / RUNTIME_METADATA_FILENAME, metadata)
        if {item.name for item in temporary.iterdir()} != {
            "adapter.safetensors",
            RUNTIME_METADATA_FILENAME,
        }:
            raise RuntimeError("V87 runtime checkpoint is not an exact two-file package")
        os.replace(temporary, destination)
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
    if destination.exists():
        raise FileExistsError(destination)
    source_metadata = _read_json(SOURCE_MEMORY / METADATA_FILENAME)
    if (
        source_metadata.get("canonical_prefix_sha256") != SOURCE_MEMORY_PREFIX_SHA256
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
    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}.", dir=destination.parent
    ) as raw:
        temporary = Path(raw)
        shutil.copyfile(SOURCE_MEMORY / MEMORY_FILENAME, temporary / MEMORY_FILENAME)
        _write_json(temporary / METADATA_FILENAME, rebound)
        if sha256_file(temporary / MEMORY_FILENAME) != SOURCE_MEMORY_TENSOR_FILE_SHA256:
            raise RuntimeError("V87 scene-memory bytes changed during metadata rebinding")
        os.replace(temporary, destination)
    loaded = load_v81_scene_memory(
        destination,
        expected_scene_id="scene_000001",
        expected_base_checkpoint_sha256=checkpoint_sha256,
        expected_runtime_config_sha256=runtime_config_sha256,
        expected_model_device="cpu",
    )
    if loaded.metadata["canonical_prefix_sha256"] != SOURCE_MEMORY_PREFIX_SHA256:
        raise RuntimeError("V87 canonical scene prefix changed during rebinding")
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


def prepare_candidate() -> dict[str, Any]:
    if CANDIDATE_CHECKPOINT.exists() or CANDIDATE_MEMORY.exists():
        raise FileExistsError("V87 runtime candidate destination is not pristine")
    evidence = authenticate_v87_model_gate()
    config = materialize_runtime_config(evidence)
    metadata = build_runtime_metadata(
        evidence,
        promotion="pending_oracle_isolated_runtime_smoke",
        smoke_report_sha256=None,
    )
    checkpoint = _atomic_checkpoint(CANDIDATE_CHECKPOINT, metadata=metadata)
    memory = _rebind_memory(
        CANDIDATE_MEMORY,
        checkpoint_sha256=str(checkpoint["checkpoint_sha256"]),
        runtime_config_sha256=effective_runtime_config_sha256(config),
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "phase": "v87_strict_runtime_candidate_prepared",
        "candidate_checkpoint": str(CANDIDATE_CHECKPOINT.relative_to(PROJECT_ROOT)),
        "candidate_memory": str(CANDIDATE_MEMORY.relative_to(PROJECT_ROOT)),
        "runtime_config": str(RUNTIME_CONFIG.relative_to(PROJECT_ROOT)),
        "runtime_config_sha256": effective_runtime_config_sha256(config),
        "checkpoint": checkpoint,
        "scene_memory": memory,
        "model_gate_evidence": evidence,
        "runtime_metadata_contains_supervision": False,
        "promotion_decision": "pending_oracle_isolated_runtime_smoke",
    }


def verify_candidate() -> dict[str, Any]:
    evidence = authenticate_v87_model_gate()
    if not CANDIDATE_CHECKPOINT.is_dir() or not CANDIDATE_MEMORY.is_dir():
        raise FileNotFoundError("V87 strict runtime candidate package is incomplete")
    expected = build_runtime_metadata(
        evidence,
        promotion="pending_oracle_isolated_runtime_smoke",
        smoke_report_sha256=None,
    )
    if _read_json(CANDIDATE_CHECKPOINT / RUNTIME_METADATA_FILENAME) != expected:
        raise ValueError("V87 candidate runtime metadata changed")
    fingerprint, files = checkpoint_fingerprint(CANDIDATE_CHECKPOINT)
    candidate = load_file(
        str(CANDIDATE_CHECKPOINT / "adapter.safetensors"), device="cpu"
    )
    expected_tensors, _inheritance = _merged_adapter()
    checks = {
        "exact_two_file_checkpoint": {item["path"] for item in files}
        == {"adapter.safetensors", RUNTIME_METADATA_FILENAME},
        "exact_tensor_inventory": set(candidate) == set(expected_tensors),
        "all_tensor_values_equal": set(candidate) == set(expected_tensors)
        and all(torch.equal(candidate[name], expected_tensors[name]) for name in candidate),
        "scene_memory_bytes_unchanged": sha256_file(
            CANDIDATE_MEMORY / MEMORY_FILENAME
        )
        == SOURCE_MEMORY_TENSOR_FILE_SHA256,
    }
    load_v81_scene_memory(
        CANDIDATE_MEMORY,
        expected_scene_id="scene_000001",
        expected_base_checkpoint_sha256=fingerprint,
        expected_runtime_config_sha256=effective_runtime_config_sha256(
            load_runtime_config(RUNTIME_CONFIG)
        ),
        expected_model_device="cpu",
    )
    if not all(checks.values()):
        raise RuntimeError(f"V87 strict candidate verification failed: {checks}")
    return {
        "phase": "v87_strict_runtime_candidate_verified",
        "checkpoint_sha256": fingerprint,
        "v87_bridge_state_sha256": evidence["v87_bridge_state_sha256"],
        "checks": checks,
        "passed": True,
    }


def _normalized_answer(value: object) -> str:
    return str(value).strip().casefold().rstrip(".!?")


def validate_runtime_smoke_report(
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
        smoke.get("schema_version") != 87
        or smoke.get("artifact") != "gemma4_v87_strict_runtime_smoke_v1"
        or smoke.get("model_gate_report_sha256")
        != evidence.get("model_gate_report_sha256")
        or smoke.get("v87_bridge_state_sha256")
        != evidence.get("v87_bridge_state_sha256")
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
        or smoke.get("held_out_generalization_claim") is not False
        or smoke.get("chat_log_sha256") != sha256_file(SMOKE_CHAT)
        or smoke.get("file_audit_sha256") != sha256_file(SMOKE_AUDIT)
    ):
        raise ValueError("V87 runtime smoke evidence did not pass exactly")


def run_smoke() -> dict[str, Any]:
    """Run the candidate in an external process with oracle physically absent."""

    if SMOKE_REPORT.is_file():
        existing = _read_json(SMOKE_REPORT)
        validate_runtime_smoke_report(existing, authenticate_v87_model_gate())
        return existing
    if SMOKE_CHAT.exists() or SMOKE_AUDIT.exists():
        raise FileExistsError("V87 smoke artifacts already exist; results are create-once")
    evidence = authenticate_v87_model_gate()
    candidate = verify_candidate()
    oracle = PROJECT_ROOT / "data/oracle"
    unavailable = PROJECT_ROOT / f"data/.oracle-unavailable-v87-{os.getpid()}"
    if not oracle.is_dir() or unavailable.exists():
        raise FileNotFoundError("The oracle directory cannot be made physically unavailable")
    command = [
        str(PROJECT_ROOT / ".venv-gemma4/bin/python"),
        "-m",
        "semantic_3d_chat.chat.v87_strict_scene1_cli",
        "--config",
        str(RUNTIME_CONFIG),
        "--scene",
        "scene_000001",
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
    if completed.returncode != 0:
        raise RuntimeError(
            "V87 strict runtime smoke failed: "
            f"returncode={completed.returncode}\nstdout={completed.stdout}\n"
            f"stderr={completed.stderr}"
        )
    rows = [
        json.loads(line)
        for line in SMOKE_CHAT.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != len(_SMOKE_CASES):
        raise RuntimeError("V87 smoke chat row count changed")
    stdout_records: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            stdout_records.append(value)
    startup_records = [
        row for row in stdout_records if row.get("phase") == "v87_strict_scene1_ready"
    ]
    completion_records = [
        row for row in stdout_records if row.get("phase") == "v87_chat_audit_complete"
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
        "child_used_exact_nine_frozen_banks": startup.get("frozen_lora_bank_count")
        == 9
        and startup.get("trainable_runtime_parameter_count") == 0
        and startup.get("v87_bridge_state_sha256")
        == evidence["v87_bridge_state_sha256"],
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
        "artifact": "gemma4_v87_strict_runtime_smoke_v1",
        "model_gate_report_sha256": evidence["model_gate_report_sha256"],
        "v87_bridge_state_sha256": evidence["v87_bridge_state_sha256"],
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
        "chat_process_stdout_sha256": hashlib.sha256(completed.stdout.encode()).hexdigest(),
        "chat_process_stderr_sha256": hashlib.sha256(completed.stderr.encode()).hexdigest(),
        "behavior": behavior,
        "prefix_hashes": prefix_hashes,
        "environment_conditioned_input_hashes": input_hashes,
        "gates": gates,
        "passed": all(gates.values()),
        "promotion_authorized": all(gates.values()),
        "held_out_generalization_claim": False,
    }
    _write_json(SMOKE_REPORT, report)
    return report


def promote_release() -> dict[str, Any]:
    evidence = authenticate_v87_model_gate()
    if RELEASE_CHECKPOINT.exists() or RELEASE_MEMORY.exists() or RELEASE_REPORT.exists():
        raise FileExistsError("V87 strict runtime release destination already exists")
    smoke = _read_json(SMOKE_REPORT)
    validate_runtime_smoke_report(smoke, evidence)
    candidate = verify_candidate()
    if (
        smoke.get("candidate_checkpoint_sha256") != candidate["checkpoint_sha256"]
        or smoke.get("candidate_adapter_sha256")
        != sha256_file(CANDIDATE_CHECKPOINT / "adapter.safetensors")
        or smoke.get("candidate_memory_tensor_sha256")
        != sha256_file(CANDIDATE_MEMORY / MEMORY_FILENAME)
    ):
        raise ValueError("V87 smoked candidate bytes changed before promotion")
    smoke_sha256 = sha256_file(SMOKE_REPORT)
    metadata = build_runtime_metadata(
        evidence,
        promotion="strict_scene1_experimental_primary",
        smoke_report_sha256=smoke_sha256,
    )
    checkpoint = _atomic_checkpoint(
        RELEASE_CHECKPOINT,
        metadata=metadata,
        source_adapter=CANDIDATE_CHECKPOINT / "adapter.safetensors",
    )
    if checkpoint["adapter_sha256"] != sha256_file(
        CANDIDATE_CHECKPOINT / "adapter.safetensors"
    ):
        raise RuntimeError("Promoted V87 adapter bytes differ from smoked candidate")
    config = load_runtime_config(RUNTIME_CONFIG)
    memory = _rebind_memory(
        RELEASE_MEMORY,
        checkpoint_sha256=str(checkpoint["checkpoint_sha256"]),
        runtime_config_sha256=effective_runtime_config_sha256(config),
    )
    release = {
        "schema_version": SCHEMA_VERSION,
        "artifact": "gemma4_v87_strict_runtime_release_v1",
        "promotion_decision": "strict_scene1_experimental_primary",
        "promotion_scope": "strict_direct_continuous_scene_memory_scene1_chat",
        "scene_id": "scene_000001",
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
        "held_out_generalization_claim": False,
        "v75_comparator_retained": True,
        "runtime_config": str(RUNTIME_CONFIG.relative_to(PROJECT_ROOT)),
        "runtime_config_sha256": effective_runtime_config_sha256(config),
        "checkpoint": checkpoint,
        "scene_memory": memory,
        "bindings": {**evidence, "runtime_smoke_sha256": smoke_sha256},
        "chat_runtime_loads_evaluation_reports": False,
        "runtime_checkpoint_contains_environmental_text": False,
        "runtime_checkpoint_contains_supervision": False,
        "scene_memory_metadata_only_rebinding": True,
        "scene_memory_tensor_bytes_unchanged": True,
        "all_release_gates_passed": True,
    }
    _write_json(RELEASE_REPORT, release)
    return release


def verify_release() -> dict[str, Any]:
    evidence = authenticate_v87_model_gate()
    release = _read_json(RELEASE_REPORT)
    metadata = _read_json(RELEASE_CHECKPOINT / RUNTIME_METADATA_FILENAME)
    validate_runtime_checkpoint_metadata(metadata)
    fingerprint, files = checkpoint_fingerprint(RELEASE_CHECKPOINT)
    loaded = load_v81_scene_memory(
        RELEASE_MEMORY,
        expected_scene_id="scene_000001",
        expected_base_checkpoint_sha256=fingerprint,
        expected_runtime_config_sha256=effective_runtime_config_sha256(
            load_runtime_config(RUNTIME_CONFIG)
        ),
        expected_model_device="cpu",
    )
    provenance = metadata["initialization_provenance"]["v87_strict_runtime_release"]
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
        "v87_state_binding_exact": provenance["v87_bridge_state_sha256"]
        == evidence["v87_bridge_state_sha256"],
        "runtime_promotion_authorized": provenance["runtime_promotion_authorized"]
        is True,
    }
    if not all(checks.values()):
        raise RuntimeError(f"V87 strict runtime release verification failed: {checks}")
    return {
        "phase": "v87_strict_runtime_release_verified",
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
        "authenticate": authenticate_v87_model_gate,
        "prepare": prepare_candidate,
        "verify-candidate": verify_candidate,
        "smoke": run_smoke,
        "promote": promote_release,
        "verify": verify_release,
        "cleanup-failed-candidate": lambda: (
            cleanup_failed_candidate()
            or {"phase": "v87_failed_candidate_cleaned", "passed": True}
        ),
    }
    try:
        result = functions[args.command]()
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"V87 strict runtime {args.command} refused: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 2 if args.command == "smoke" and result.get("passed") is not True else 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CANDIDATE_CHECKPOINT",
    "CANDIDATE_MEMORY",
    "RELEASE_CHECKPOINT",
    "RELEASE_MEMORY",
    "RUNTIME_CONFIG",
    "V87_MODEL_GATE_REPORT",
    "authenticate_v87_model_gate",
    "build_runtime_config_payload",
    "build_runtime_metadata",
    "cleanup_failed_candidate",
    "materialize_runtime_config",
    "prepare_candidate",
    "promote_release",
    "run_smoke",
    "sha256_file",
    "validate_model_gate_contract",
    "validate_runtime_smoke_report",
    "verify_candidate",
    "verify_release",
]
