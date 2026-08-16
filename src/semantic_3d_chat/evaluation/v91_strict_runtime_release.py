"""Fail-closed strict-runtime packaging for V91's repair bridge.

V91 is a development-known, single-scene conversational repair.  This
module does not train or score the model and is never imported by chat.  It
authenticates the sealed experiment, preregistration, CPU preflight, fixed
training result, fixed evaluation result, promoted V89 parent, exact frozen
failed V90 bridge, and exact two-tensor V91 bridge before it can write a
standalone runtime artifact.

The isolated smoke child receives only a sanitized runtime YAML, a frozen
thirteen-bank checkpoint, one immutable continuous scene-memory artifact, and
the thirteen user questions.  Expected answers remain in this parent release
process and are applied only after the child has exited.  The oracle directory
is physically renamed for the entire child lifetime.
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
from semantic_3d_chat.chat.v91_strict_scene1_runtime import (
    PROMOTION_DECISION,
    validate_v91_runtime_contract,
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
from semantic_3d_chat.evaluation.v91_scene1_conversational_preflight import (
    authenticate_sources_v91,
    load_config_v91,
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

SCHEMA_VERSION: Final[int] = 91
SCENE_ID: Final[str] = "scene_000001"
_SHA256: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{64}")

# These three artifacts were sealed before the full model was loaded.  Final
# training/evaluation/candidate hashes are deliberately not source constants:
# their exact identities are authenticated through the create-once reports and
# the candidate's mutually bound metadata after those artifacts exist.
EXPERIMENT_CONFIG_SHA256: Final[str] = (
    "0270a5731059b286cc26657cc3b10846397a1ba67054f93ebfd3adaa25e9885f"
)
PREREGISTRATION_SHA256: Final[str] = (
    "151d82517b6a902ba8eb11bb37a90c9eaa4e9f0ad6ba017b68a6bb1c384a23f0"
)
CPU_PREFLIGHT_SHA256: Final[str] = (
    "79da2ca3bf54bc1fb808e839a00e5e6af4a74f0fcaad5a8f72b0c7ee1b2fc6f9"
)
SOURCE_MEMORY_PREFIX_SHA256: Final[str] = (
    "a428f5147c815839ae7315a0adab952ab210814fb21dcdc5bf13b167f28a6e37"
)
SOURCE_MEMORY_TENSOR_FILE_SHA256: Final[str] = (
    "3427851432d8f2a3609b6205b18b8c0d9a0fcf68d8f3bf0c98c758ac64209ffb"
)
if any(
    _SHA256.fullmatch(value) is None
    for value in (
        EXPERIMENT_CONFIG_SHA256,
        PREREGISTRATION_SHA256,
        CPU_PREFLIGHT_SHA256,
        SOURCE_MEMORY_PREFIX_SHA256,
        SOURCE_MEMORY_TENSOR_FILE_SHA256,
    )
):
    raise RuntimeError("V91 fixed evidence constants must be lowercase SHA-256")

V90_BANK: Final[str] = "v90_scene1_conversational_bridge"
V90_TARGET: Final[str] = "model.language_model.layers.28.self_attn.o_proj"
V90_STATE_SHA256: Final[str] = (
    "70e236711d8ac1fe7cf808f6f4e939b29db476016c8ef49db143707df0f3bde7"
)
V90_WEIGHTS_SHA256: Final[str] = (
    "be8a6fa9b633dc52ca962393170623c2707e339206ced627e6439ce7db0a7f94"
)
V90_METADATA_SHA256: Final[str] = (
    "c2dba75094c829c7fadaadf8111a712921f70372ac84e2073714530519349acc"
)
V91_BANK: Final[str] = "v91_scene1_conversational_repair"
V91_TARGET: Final[str] = "model.language_model.layers.33.mlp.down_proj"
PARENT_BANKS: Final[tuple[str, ...]] = (
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
EXPECTED_BANKS: Final[tuple[str, ...]] = (*PARENT_BANKS, V90_BANK, V91_BANK)
PARENT_ADAPTER_PARAMETER_COUNT: Final[int] = 872_448
V90_ADAPTER_PARAMETER_COUNT: Final[int] = 28_672
V91_ADAPTER_PARAMETER_COUNT: Final[int] = 221_184
EXPECTED_ADAPTER_PARAMETER_COUNT: Final[int] = 1_122_304

CORE_ACTIONABLE_INTENTS: Final[frozenset[str]] = frozenset(
    {
        "table_contents",
        "under_table",
        "wall_object",
        "cube_location",
        "sitting",
        "bowl_contents",
    }
)
_REQUIRED_MODEL_GATES: Final[frozenset[str]] = frozenset(
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
_REQUIRED_TRAINING_GATES: Final[frozenset[str]] = frozenset(
    {
        "all_1770_sealed_micro_rows_consumed",
        "all_138_canonical_rows_consumed_once_each_epoch",
        "all_14_parent_errors_replayed_twice_each_epoch",
        "all_124_parent_correct_anchors_replayed_once_each_epoch",
        "all_84_success_conversational_rows_consumed_each_epoch",
        "all_216_repair_conversational_rows_consumed_each_epoch",
        "all_39_primary_causal_margin_rows_consumed",
        "answer_only_ce_on_every_micro_row",
        "fixed_final_update_295_reached",
        "exact_v89_plus_failed_v90_twelve_bank_parent_frozen",
        "sole_trainable_surface_is_v91_repair_bridge",
        "nonzero_finite_gradient_every_update",
        "memory_hash_invariant",
        "zero_payload_hash_invariant",
        "protected_read_count_zero",
        "runtime_candidate_contains_no_supervision_or_environment",
    }
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
        "child_used_exact_thirteen_frozen_banks",
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

EXPERIMENT_CONFIG: Final[Path] = (
    PROJECT_ROOT / "configs/experiments/gemma4_v91_scene1_conversational_repair.yaml"
)
PREREGISTRATION: Final[Path] = (
    PROJECT_ROOT / "reports/gemma4/metrics/gemma4_v91_scene1_conversational_repair_preregistration_v2.json"
)
CPU_PREFLIGHT: Final[Path] = (
    PROJECT_ROOT / "reports/gemma4/metrics/gemma4_v91_scene1_conversational_repair_cpu_preflight_v2.json"
)
TRAINING_REPORT: Final[Path] = (
    PROJECT_ROOT / "reports/gemma4/metrics/gemma4_v91_scene1_conversational_repair_training_v2.json"
)
PREDICTIONS: Final[Path] = (
    PROJECT_ROOT / "reports/gemma4/predictions/gemma4_v91_scene1_conversational_repair_evaluation_v2.json"
)
MODEL_GATE_REPORT: Final[Path] = (
    PROJECT_ROOT / "reports/gemma4/metrics/gemma4_v91_scene1_conversational_repair_evaluation_v2.json"
)
PARENT_RUNTIME_CONFIG: Final[Path] = PROJECT_ROOT / "configs/runtime/gemma4_v89_strict_scene1.yaml"
PARENT_CHECKPOINT: Final[Path] = (
    PROJECT_ROOT / "data_gemma4/runtime/checkpoints/gemma4_v89_strict_scene1_release_v1"
)
PARENT_RELEASE_REPORT: Final[Path] = (
    PROJECT_ROOT / "reports/gemma4/metrics/gemma4_v89_strict_runtime_release.json"
)
V91_BRIDGE_CANDIDATE: Final[Path] = (
    PROJECT_ROOT / "reports/gemma4/artifacts/v91_scene1_conversational_repair_final_v2"
)
V90_BRIDGE_CANDIDATE: Final[Path] = (
    PROJECT_ROOT / "reports/gemma4/artifacts/v90_scene1_conversational_final"
)
SOURCE_MEMORY: Final[Path] = PROJECT_ROOT / "data_gemma4/runtime/scene_memories/v89/scene_000001"

RUNTIME_CONFIG: Final[Path] = PROJECT_ROOT / "configs/runtime/gemma4_v91_strict_scene1.yaml"
CANDIDATE_CHECKPOINT: Final[Path] = (
    PROJECT_ROOT / "reports/gemma4/artifacts/v91_scene1_conversational_repair_runtime_v1"
)
CANDIDATE_MEMORY: Final[Path] = (
    PROJECT_ROOT / "reports/gemma4/artifacts/v91_scene1_conversational_repair_runtime_memory_v1/scene_000001"
)
RELEASE_CHECKPOINT: Final[Path] = (
    PROJECT_ROOT / "data_gemma4/runtime/checkpoints/gemma4_v91_strict_scene1_release_v1"
)
RELEASE_MEMORY: Final[Path] = PROJECT_ROOT / "data_gemma4/runtime/scene_memories/v91/scene_000001"
SMOKE_CHAT: Final[Path] = PROJECT_ROOT / "reports/gemma4/examples/v91_strict_runtime_smoke.jsonl"
SMOKE_AUDIT: Final[Path] = (
    PROJECT_ROOT / "reports/gemma4/metrics/v91_strict_runtime_smoke_access.json"
)
SMOKE_REPORT: Final[Path] = (
    PROJECT_ROOT / "reports/gemma4/metrics/gemma4_v91_scene1_conversational_repair_runtime_smoke.json"
)
RELEASE_REPORT: Final[Path] = (
    PROJECT_ROOT / "reports/gemma4/metrics/gemma4_v91_strict_runtime_release.json"
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
        raise ValueError(f"V91 {field} is not a lowercase SHA-256 digest")
    return value


def _relative(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def _load_experiment() -> dict[str, Any]:
    if EXPERIMENT_CONFIG.is_symlink() or not EXPERIMENT_CONFIG.is_file():
        raise FileNotFoundError("V91 sealed experiment config is not a physical file")
    if sha256_file(EXPERIMENT_CONFIG) != EXPERIMENT_CONFIG_SHA256:
        raise ValueError("V91 sealed experiment config changed")
    raw = EXPERIMENT_CONFIG.read_text(encoding="utf-8")
    if "REPLACE_" in raw:
        raise ValueError("V91 sealed experiment config contains a placeholder")
    payload = yaml.safe_load(raw)
    if not isinstance(payload, dict) or set(payload) != {"v91"}:
        raise ValueError("V91 sealed experiment identity changed")
    experiment = payload["v91"]
    outputs = experiment.get("outputs") if isinstance(experiment, dict) else None
    dataset = experiment.get("dataset") if isinstance(experiment, dict) else None
    bridge = experiment.get("bridge") if isinstance(experiment, dict) else None
    scope = experiment.get("scope") if isinstance(experiment, dict) else None
    expected_outputs = {
        "preregistration": _relative(PREREGISTRATION),
        "cpu_preflight": _relative(CPU_PREFLIGHT),
        "fixed_final_candidate": _relative(V91_BRIDGE_CANDIDATE),
        "training_report": _relative(TRAINING_REPORT),
        "evaluation_predictions": _relative(PREDICTIONS),
        "evaluation_report": _relative(MODEL_GATE_REPORT),
    }
    if (
        not isinstance(experiment, dict)
        or experiment.get("schema_version") != 91
        or experiment.get("artifact")
        != "gemma4_v91_scene1_conversational_repair_direct_memory_v1"
        or experiment.get("status") != "sealed_before_full_model_load"
        or outputs != expected_outputs
        or not isinstance(dataset, dict)
        or dataset.get("scene_id") != SCENE_ID
        or dataset.get("canonical_row_count") != 138
        or dataset.get("conversational_intent_count") != 13
        or dataset.get("newly_held_wording_row_count") != 26
        or dataset.get("total_micro_rows") != 1_770
        or dataset.get("questions_or_answers_serialized_at_runtime") is not False
        or not isinstance(bridge, dict)
        or bridge.get("bank_name") != V91_BANK
        or bridge.get("target_module") != V91_TARGET
        or bridge.get("rank") != 16
        or float(bridge.get("alpha", -1.0)) != 32.0
        or float(bridge.get("dropout", -1.0)) != 0.0
        or bridge.get("trainable_parameter_count") != V91_ADAPTER_PARAMETER_COUNT
        or not isinstance(scope, dict)
        or scope.get("post_v90_training_set_development") is not True
        or scope.get("exact_failed_v90_candidate_frozen") is not True
        or scope.get("single_scene_conversational_repair") is not True
        or scope.get("development_known_training_wordings") is not True
        or scope.get("held_out_scene_generalization_claim") is not False
        or scope.get("runtime_promotion_authorized") is not False
    ):
        raise ValueError("V91 sealed experiment contract changed")
    return experiment


def _expected_source_hashes(experiment: Mapping[str, Any]) -> dict[str, Any]:
    config = load_config_v91(EXPERIMENT_CONFIG, allow_draft=False)
    if config != experiment:
        raise ValueError("V91 release and preflight parsed different sealed configs")
    return authenticate_sources_v91(config)


def _authenticate_source_hashes(
    report: Mapping[str, Any], experiment: Mapping[str, Any]
) -> dict[str, Any]:
    expected = _expected_source_hashes(experiment)
    observed = report.get("source_hashes")
    if observed != expected:
        raise ValueError("V91 report source inventory differs from sealed config")
    return expected


def _authenticate_preflight(experiment: Mapping[str, Any]) -> None:
    fixed = {
        PREREGISTRATION: PREREGISTRATION_SHA256,
        CPU_PREFLIGHT: CPU_PREFLIGHT_SHA256,
    }
    mismatches = {
        _relative(path): {
            "expected": expected,
            "observed": sha256_file(path) if path.is_file() else None,
        }
        for path, expected in fixed.items()
        if path.is_symlink() or not path.is_file() or sha256_file(path) != expected
    }
    if mismatches:
        raise ValueError(f"V91 fixed preregistration evidence changed: {mismatches}")
    preregistration = _read_json(PREREGISTRATION)
    preflight = _read_json(CPU_PREFLIGHT)
    prereg_protocol = preregistration.get("protocol")
    preflight_protocol = preflight.get("protocol")
    if (
        preregistration.get("artifact")
        != "gemma4_v91_scene1_conversational_repair_preregistration_v2"
        or preregistration.get("schema_version") != 91
        or preregistration.get("status") != "sealed_before_full_model_load"
        or preregistration.get("config_sha256") != EXPERIMENT_CONFIG_SHA256
        or preregistration.get("model_loaded") is not False
        or preregistration.get("oracle_loaded") is not False
        or not isinstance(prereg_protocol, dict)
        or prereg_protocol.get("total_micro_rows") != 1_770
        or prereg_protocol.get("optimizer_updates") != 295
        or prereg_protocol.get("total_primary_causal_rows") != 39
        or preflight.get("artifact")
        != "gemma4_v91_scene1_conversational_repair_cpu_preflight_v2"
        or preflight.get("schema_version") != 91
        or preflight.get("status") != "cpu_preflight_pass_training_authorized"
        or preflight.get("training_authorized") is not True
        or preflight.get("config_sha256") != EXPERIMENT_CONFIG_SHA256
        or preflight.get("preregistration_sha256") != PREREGISTRATION_SHA256
        or preflight.get("model_loaded") is not False
        or preflight.get("oracle_loaded") is not False
        or preflight_protocol != prereg_protocol
        or preregistration.get("scope") != experiment.get("scope")
    ):
        raise ValueError("V91 fixed preregistration or preflight contract changed")


def _metric_bucket(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"V91 metric bucket is malformed: {name}")
    return value




def validate_model_gate_contract_v91(
    report: Mapping[str, Any], experiment: Mapping[str, Any] | None = None
) -> None:
    """Fail closed unless the exact preregistered V91 repair gates passed."""

    experiment = _load_experiment() if experiment is None else experiment
    thresholds = _metric_bucket(experiment.get("gates"), "preregistered gates")
    intents = experiment.get("conversational_intents")
    if not isinstance(intents, list) or not all(
        isinstance(row, Mapping) for row in intents
    ):
        raise ValueError("V91 conversational intent contract is malformed")
    intent_ids = {str(row.get("id")) for row in intents}
    metrics = _metric_bucket(report.get("metrics"), "metrics")
    gates = _metric_bucket(metrics.get("model_acceptance_gates"), "model gates")
    canonical = _metric_bucket(
        metrics.get("canonical_type_specific"), "canonical type-specific"
    )
    by_type = _metric_bucket(
        metrics.get("canonical_accuracy_by_answer_type"), "canonical by type"
    )
    primary = _metric_bucket(
        metrics.get("primary_conversational"), "primary conversational"
    )
    primary_by_intent = _metric_bucket(primary.get("by_intent"), "primary by intent")
    held = _metric_bucket(metrics.get("new_held_wording"), "new held wording")
    held_by_intent = _metric_bucket(held.get("by_intent"), "new held by intent")
    causal = _metric_bucket(metrics.get("causal_control"), "causal control")
    type_minima = {
        "presence": int(thresholds["canonical_presence_correct_minimum"]),
        "count": int(thresholds["canonical_count_correct_minimum"]),
        "metric": int(thresholds["canonical_metric_correct_minimum"]),
        "attribute": int(thresholds["canonical_attribute_correct_minimum"]),
        "spatial_relation": int(thresholds["canonical_spatial_correct_minimum"]),
        "support": int(thresholds["canonical_support_correct_minimum"]),
    }
    type_counts_pass = all(
        int(_metric_bucket(by_type.get(name), name).get("correct", -1)) >= minimum
        for name, minimum in type_minima.items()
    )
    primary_intents_pass = bool(
        set(primary_by_intent) == intent_ids
        and all(
            _metric_bucket(primary_by_intent[name], name).get("total") == 1
            for name in intent_ids
        )
    )
    core_pass = bool(
        CORE_ACTIONABLE_INTENTS.issubset(intent_ids)
        and all(
            _metric_bucket(primary_by_intent.get(name), name).get("correct") == 1
            for name in CORE_ACTIONABLE_INTENTS
        )
    )
    held_intents_pass = bool(
        set(held_by_intent) == intent_ids
        and all(
            _metric_bucket(held_by_intent[name], name).get("total") == 2
            and int(_metric_bucket(held_by_intent[name], name).get("correct", -1))
            >= int(thresholds["new_held_wording_each_intent_minimum"])
            for name in intent_ids
        )
    )
    if (
        report.get("artifact")
        != "gemma4_v91_scene1_conversational_repair_evaluation_v1"
        or report.get("schema_version") != 91
        or report.get("status")
        != "model_gates_pass_separate_runtime_packaging_required"
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
        or report.get("post_v90_training_set_development") is not True
        or report.get("single_scene_conversational_repair") is not True
        or report.get("development_known_primary_questions") is not True
        or report.get("newly_held_wording_only") is not True
        or report.get("held_out_scene") is not False
        or report.get("held_out_scene_generalization_claim") is not False
        or report.get("parent_v89_runtime_checkpoint_mutated") is not False
        or report.get("parent_v90_failed_candidate_mutated") is not False
        or report.get("fixed_final_candidate_state_invariant") is not True
        or report.get("oracle_loaded") is not False
        or report.get("official_validation_loaded") is not False
        or report.get("official_test_loaded") is not False
        or report.get("deferred_final_loaded") is not False
        or canonical.get("total") != int(thresholds["canonical_total"])
        or int(canonical.get("correct", -1))
        < int(thresholds["canonical_correct_minimum"])
        or not type_counts_pass
        or primary.get("total")
        != int(thresholds["primary_conversational_total"])
        or int(primary.get("correct", -1))
        < int(thresholds["primary_conversational_required_correct"])
        or primary.get("core_actionable_total") != len(CORE_ACTIONABLE_INTENTS)
        or primary.get("core_actionable_correct")
        != int(thresholds["core_actionable_required_correct"])
        or not primary_intents_pass
        or not core_pass
        or held.get("total") != int(thresholds["new_held_wording_total"])
        or int(held.get("correct", -1))
        < int(thresholds["new_held_wording_required_correct"])
        or held.get("newly_held_wording_only") is not True
        or held.get("held_out_scene") is not False
        or not held_intents_pass
        or causal.get("row_count") != 13
        or float(causal.get("mean_zero_minus_correct_nll", -1.0))
        < float(thresholds["causal_mean_zero_minus_correct_nll_minimum"])
        or float(causal.get("required_mean_margin_nll", -1.0))
        != float(thresholds["causal_mean_zero_minus_correct_nll_minimum"])
        or int(causal.get("canonical_prediction_changes", -1))
        < int(thresholds["causal_prediction_change_minimum"])
    ):
        raise ValueError("V91 model-level acceptance report did not pass exactly")

    memory = _metric_bucket(report.get("scene_memory"), "scene memory")
    leakage = _metric_bucket(report.get("leakage"), "leakage")
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
        raise ValueError("V91 model leakage or immutable-memory evidence changed")


def _candidate_fingerprint(root: Path) -> tuple[str, list[dict[str, Any]]]:
    if root.is_symlink() or not root.is_dir():
        raise FileNotFoundError("V91 bridge candidate is unavailable")
    names = {item.name for item in root.iterdir()}
    if names != {"bridge.safetensors", RUNTIME_METADATA_FILENAME} or any(
        item.is_symlink() or not item.is_file() for item in root.iterdir()
    ):
        raise ValueError("V91 bridge candidate is not an exact two-file artifact")
    files = [root / "bridge.safetensors", root / RUNTIME_METADATA_FILENAME]
    entries = [
        {
            "path": path.name,
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in files
    ]
    return _canonical_sha256(entries), entries




def _contract_from_evidence(evidence: Mapping[str, Any]) -> BridgeSourceContract:
    return BridgeSourceContract(
        root=V91_BRIDGE_CANDIDATE,
        artifact="gemma4_v91_scene1_conversational_repair_fixed_final_v1",
        bank_name=V91_BANK,
        target_module=V91_TARGET,
        rank=16,
        alpha=32.0,
        dropout=0.0,
        parameter_count=V91_ADAPTER_PARAMETER_COUNT,
        state_sha256=_require_hash(evidence.get("v91_bridge_state_sha256"), "bridge state"),
        weights_sha256=_require_hash(evidence.get("v91_bridge_file_sha256"), "bridge weights"),
        metadata_sha256=_require_hash(
            evidence.get("v91_bridge_metadata_sha256"), "bridge metadata"
        ),
    )


def _v90_contract() -> BridgeSourceContract:
    contract = BridgeSourceContract(
        root=V90_BRIDGE_CANDIDATE,
        artifact="gemma4_v90_scene1_conversational_fixed_final_v1",
        bank_name=V90_BANK,
        target_module=V90_TARGET,
        rank=8,
        alpha=16.0,
        dropout=0.0,
        parameter_count=V90_ADAPTER_PARAMETER_COUNT,
        state_sha256=V90_STATE_SHA256,
        weights_sha256=V90_WEIGHTS_SHA256,
        metadata_sha256=V90_METADATA_SHA256,
    )
    loaded = load_bridge_source(contract)
    if tuple(loaded.state["adapters.0.lora_a"].shape) != (8, 2_048) or tuple(
        loaded.state["adapters.0.lora_b"].shape
    ) != (1_536, 8):
        raise ValueError("V91 frozen V90 bridge tensor topology changed")
    return contract


def _contracts_from_evidence(
    evidence: Mapping[str, Any],
) -> tuple[BridgeSourceContract, BridgeSourceContract]:
    return _v90_contract(), _contract_from_evidence(evidence)




def authenticate_v91_model_gate() -> dict[str, Any]:
    """Authenticate the passing V91 gate and both exact added bridge banks."""

    from semantic_3d_chat.training.train_v91_scene1_conversational_repair import (
        authenticate_training_report_v91,
    )

    experiment = _load_experiment()
    _authenticate_preflight(experiment)
    bindings = authenticate_training_report_v91(
        experiment, config_path=EXPERIMENT_CONFIG
    )
    training = _read_json(TRAINING_REPORT)
    predictions = _read_json(PREDICTIONS)
    report = _read_json(MODEL_GATE_REPORT)
    if report.get("preregistered_gates") != experiment.get("gates"):
        raise ValueError("V91 evaluation is not bound to preregistered gates")
    validate_model_gate_contract_v91(report, experiment)
    expected_sources = _authenticate_source_hashes(report, experiment)
    if training.get("source_hashes") != expected_sources:
        raise ValueError("V91 training and evaluation source inventories differ")
    training_sha = sha256_file(TRAINING_REPORT)
    predictions_sha = sha256_file(PREDICTIONS)
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
        != "gemma4_v91_scene1_conversational_repair_predictions_v1"
        or predictions.get("schema_version") != 91
        or predictions.get("status") != "fixed_final_evaluation_only_not_runtime"
        or predictions.get("config_sha256") != EXPERIMENT_CONFIG_SHA256
        or predictions.get("training_report_sha256") != training_sha
        or predictions.get("scene_id") != SCENE_ID
        or predictions.get("scene_count") != 1
        or predictions.get("canonical_row_count") != 138
        or predictions.get("primary_conversational_row_count") != 13
        or predictions.get("new_held_wording_row_count") != 26
        or predictions.get("causal_control_row_count") != 13
        or predictions.get("fixed_checkpoint_selected_before_scoring") is not True
        or predictions.get("checkpoint_selection_after_scoring") is not False
        or predictions.get("newly_held_wording_only") is not True
        or predictions.get("held_out_scene") is not False
        or predictions.get("frozen_parent_state_invariant") is not True
        or predictions.get("frozen_parent_state_before")
        != predictions.get("frozen_parent_state_after")
        or predictions.get("candidate_state_invariant") is not True
        or predictions.get("questions_or_answers_serialized_in_runtime_candidate")
        is not False
        or predictions.get("training_inventory_serialized_in_runtime_candidate")
        is not False
        or predictions.get("oracle_serialized_in_runtime_candidate") is not False
        or predictions.get("runtime_promotion_authorized") is not False
        or predictions.get("scene_memory") != report.get("scene_memory")
        or predictions.get("leakage") != report.get("leakage")
    ):
        raise ValueError("V91 fixed predictions or hash binding changed")
    record_contract = (
        predictions.get("canonical_records"),
        predictions.get("primary_conversational_records"),
        predictions.get("new_held_wording_records"),
        predictions.get("causal_records"),
    )
    if any(not isinstance(rows, list) for rows in record_contract) or tuple(
        len(rows) for rows in record_contract
    ) != (138, 13, 26, 13):
        raise ValueError("V91 fixed prediction record inventory changed")

    parent_release = _read_json(PARENT_RELEASE_REPORT)
    parent_fingerprint, parent_files = checkpoint_fingerprint(PARENT_CHECKPOINT)
    parent_checkpoint = parent_release.get("checkpoint")
    sources = experiment["sources"]
    if (
        sha256_file(PARENT_RUNTIME_CONFIG) != sources["runtime_config_sha256"]
        or sha256_file(PARENT_CHECKPOINT / "adapter.safetensors")
        != sources["parent_v89_adapter_sha256"]
        or sha256_file(PARENT_CHECKPOINT / RUNTIME_METADATA_FILENAME)
        != sources["parent_v89_metadata_sha256"]
        or parent_release.get("artifact")
        != "gemma4_v89_strict_runtime_release_v1"
        or parent_release.get("schema_version") != 89
        or parent_release.get("promotion_decision")
        != "strict_scene1_experimental_primary"
        or parent_release.get("all_release_gates_passed") is not True
        or parent_release.get("held_out_generalization_claim") is not False
        or not isinstance(parent_checkpoint, Mapping)
        or parent_checkpoint.get("checkpoint_sha256") != parent_fingerprint
        or parent_checkpoint.get("checkpoint_files") != parent_files
        or parent_checkpoint.get("adapter_sha256")
        != sources["parent_v89_adapter_sha256"]
        or parent_checkpoint.get("runtime_metadata_sha256")
        != sources["parent_v89_metadata_sha256"]
    ):
        raise ValueError("V91 promoted V89 parent binding changed")

    v90_contract = _v90_contract()
    v91_metadata = _read_json(V91_BRIDGE_CANDIDATE / RUNTIME_METADATA_FILENAME)
    v91_contract = _contract_from_evidence(
        {
            "v91_bridge_state_sha256": bindings["candidate_state_sha256"],
            "v91_bridge_file_sha256": bindings["candidate_weights_sha256"],
            "v91_bridge_metadata_sha256": sha256_file(
                V91_BRIDGE_CANDIDATE / RUNTIME_METADATA_FILENAME
            ),
        }
    )
    loaded_v91 = load_bridge_source(v91_contract)
    if (
        v91_metadata.get("schema_version") != 91
        or v91_metadata.get("frozen_parent_bank_count") != 12
        or v91_metadata.get("total_bank_count") != 13
        or v91_metadata.get("frozen_parent_parameter_count") != 901_120
        or v91_metadata.get("total_adapter_parameter_count")
        != EXPECTED_ADAPTER_PARAMETER_COUNT
        or v91_metadata.get("v90_parent_state_sha256") != V90_STATE_SHA256
        or v91_metadata.get("v90_parent_runtime_promotable") is not False
        or tuple(loaded_v91.state["adapters.0.lora_a"].shape) != (16, 12_288)
        or tuple(loaded_v91.state["adapters.0.lora_b"].shape) != (1_536, 16)
    ):
        raise ValueError("V91 fixed-final bridge or frozen V90 lineage changed")
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_config_sha256": EXPERIMENT_CONFIG_SHA256,
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "cpu_preflight_sha256": CPU_PREFLIGHT_SHA256,
        "training_report_sha256": training_sha,
        "evaluation_predictions_sha256": predictions_sha,
        "model_gate_report_sha256": sha256_file(MODEL_GATE_REPORT),
        "parent_release_report_sha256": sha256_file(PARENT_RELEASE_REPORT),
        "parent_checkpoint_sha256": parent_fingerprint,
        "parent_adapter_sha256": sources["parent_v89_adapter_sha256"],
        "parent_metadata_sha256": sources["parent_v89_metadata_sha256"],
        "v90_bridge_file_sha256": v90_contract.weights_sha256,
        "v90_bridge_metadata_sha256": v90_contract.metadata_sha256,
        "v90_bridge_state_sha256": v90_contract.state_sha256,
        "v91_bridge_file_sha256": v91_contract.weights_sha256,
        "v91_bridge_metadata_sha256": v91_contract.metadata_sha256,
        "v91_bridge_state_sha256": v91_contract.state_sha256,
        "model_acceptance_gate_passed": True,
        "model_gate_report_authenticated": True,
        "post_v90_training_set_development": True,
        "single_scene_conversational_repair": True,
        "development_known_primary_questions": True,
        "newly_held_wording_only": True,
        "held_out_scene": False,
        "held_out_generalization_claim": False,
    }


def build_runtime_config_payload(evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Build the sanitized, frozen parent-plus-V91 runtime configuration."""

    parent = load_runtime_config(PARENT_RUNTIME_CONFIG)
    parent.pop("_config_path", None)
    configured = parent.get("language", {}).get("lora_banks")
    parent_metadata = _read_json(PARENT_CHECKPOINT / RUNTIME_METADATA_FILENAME)
    parent_states = parent_metadata.get("lora_bank_state_sha256")
    if (
        not isinstance(configured, dict)
        or tuple(configured) != PARENT_BANKS
        or any(row.get("trainable") is not False for row in configured.values())
        or not isinstance(parent_states, dict)
        or set(parent_states) != set(PARENT_BANKS)
    ):
        raise ValueError("V91 authenticated V89 parent bank inventory changed")
    for name in PARENT_BANKS:
        state = _require_hash(parent_states.get(name), f"parent {name} state")
        if configured[name].get("expected_initial_state_sha256") != state:
            raise ValueError(f"V91 parent runtime state binding changed: {name}")
    payload = extend_runtime_lora_config(
        parent_runtime_config=parent,
        added_bridges=_contracts_from_evidence(evidence),
        expected_final_banks=EXPECTED_BANKS,
    )
    banks = payload.get("language", {}).get("lora_banks")
    if (
        not isinstance(banks, dict)
        or tuple(banks) != EXPECTED_BANKS
        or len(banks) != 13
        or any(row.get("trainable") is not False for row in banks.values())
    ):
        raise RuntimeError("V91 runtime payload lost exact thirteen-bank order")
    return payload


def materialize_runtime_config(evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Create the standalone YAML only after current gate authentication."""

    authenticated = authenticate_v91_model_gate()
    if dict(evidence) != authenticated:
        raise ValueError("V91 runtime-config evidence is not current")
    payload = build_runtime_config_payload(authenticated)
    encoded = yaml.safe_dump(payload, sort_keys=False, allow_unicode=False)
    if RUNTIME_CONFIG.exists():
        if (
            RUNTIME_CONFIG.is_symlink()
            or not RUNTIME_CONFIG.is_file()
            or RUNTIME_CONFIG.read_text(encoding="utf-8") != encoded
        ):
            raise ValueError("Existing V91 runtime config differs from authenticated gate")
    else:
        RUNTIME_CONFIG.parent.mkdir(parents=True, exist_ok=True)
        with RUNTIME_CONFIG.open("x", encoding="utf-8") as handle:
            handle.write(encoded)
    loaded = load_runtime_config(RUNTIME_CONFIG)
    banks = loaded["language"]["lora_banks"]
    if tuple(banks) != EXPECTED_BANKS or any(
        row.get("trainable") is not False for row in banks.values()
    ):
        raise RuntimeError("Materialized V91 runtime config is not exactly frozen")
    return loaded


def _composed_adapter(
    evidence: Mapping[str, Any],
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    return compose_exact_bank_archive(
        base_checkpoint=PARENT_CHECKPOINT,
        expected_base_banks=PARENT_BANKS,
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
        raise RuntimeError("V91 frozen source-stack inventory is invalid")
    return tensor_state_sha256(source)


def build_runtime_metadata(
    evidence: Mapping[str, Any],
    *,
    promotion: str,
    smoke_report_sha256: str | None,
) -> dict[str, Any]:
    """Build runtime-only metadata containing no supervision or scene text."""

    allowed = {
        "pending_isolated_runtime_smoke",
        PROMOTION_DECISION,
    }
    if promotion not in allowed:
        raise ValueError("Unknown V91 runtime promotion state")
    promoted = promotion == PROMOTION_DECISION
    if promoted != (smoke_report_sha256 is not None):
        raise ValueError("V91 promotion and smoke binding disagree")
    if smoke_report_sha256 is not None:
        _require_hash(smoke_report_sha256, "runtime smoke binding")

    config = build_runtime_config_payload(evidence)
    parent = _read_json(PARENT_CHECKPOINT / RUNTIME_METADATA_FILENAME)
    contracts = _contracts_from_evidence(evidence)
    metadata = extend_runtime_metadata(
        parent_metadata=parent,
        added_bridges=contracts,
        expected_final_banks=EXPECTED_BANKS,
    )
    states = metadata["lora_bank_state_sha256"]
    for row in metadata["lora"]["banks"]:
        row["expected_initial_state_sha256"] = states[str(row["name"])]
    metadata["config_hash"] = config_hash(config)
    tensors, _composition = _composed_adapter(evidence)
    metadata["frozen_block_cross_source_stack_state_sha256"] = _source_stack_sha256(tensors)
    provenance = dict(metadata.get("initialization_provenance", {}))
    provenance["v91_strict_runtime_release"] = {
        "schema_version": SCHEMA_VERSION,
        "experiment_config_sha256": evidence["experiment_config_sha256"],
        "preregistration_sha256": evidence["preregistration_sha256"],
        "cpu_preflight_sha256": evidence["cpu_preflight_sha256"],
        "training_report_sha256": evidence["training_report_sha256"],
        "evaluation_predictions_sha256": evidence["evaluation_predictions_sha256"],
        "model_gate_report_sha256": evidence["model_gate_report_sha256"],
        "parent_release_report_sha256": evidence["parent_release_report_sha256"],
        "parent_checkpoint_sha256": evidence["parent_checkpoint_sha256"],
        "v90_bridge_state_sha256": evidence["v90_bridge_state_sha256"],
        "v91_bridge_state_sha256": evidence["v91_bridge_state_sha256"],
        "model_acceptance_gate_passed": True,
        "model_gate_report_authenticated": True,
        "promotion_decision": promotion,
        "runtime_promotion_authorized": promoted,
        "smoke_report_sha256": smoke_report_sha256,
        "post_v90_training_set_development": True,
        "single_scene_conversational_repair": True,
        "development_known_primary_questions": True,
        "newly_held_wording_only": True,
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
    if (
        metadata["lora_parameter_count"] != EXPECTED_ADAPTER_PARAMETER_COUNT
        or metadata["lora_trainable_parameter_count"] != 0
        or metadata["lora"]["adapter_parameter_count"] != EXPECTED_ADAPTER_PARAMETER_COUNT
        or metadata["lora"]["trainable_adapter_parameter_count"] != 0
    ):
        raise RuntimeError("V91 runtime parameter inventory changed")
    if promoted:
        # The release packager and runtime are separate security boundaries.
        # Validate the final metadata through the actual runtime-only contract
        # before any promoted checkpoint can be written.
        runtime_contract = validate_v91_runtime_contract(
            scene_id=SCENE_ID,
            runtime_config=config,
            checkpoint_metadata=metadata,
        )
        if (
            runtime_contract.get("v91_bridge_state_sha256") != evidence["v91_bridge_state_sha256"]
            or runtime_contract.get("runtime_promotion_authorized") is not True
        ):
            raise RuntimeError("V91 promoted metadata failed its chat-runtime contract")
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
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
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
            raise RuntimeError("V91 checkpoint is not an exact two-file package")
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
        "runtime_metadata_sha256": sha256_file(destination / RUNTIME_METADATA_FILENAME),
        "exact_two_file_checkpoint": True,
    }


def _rebind_memory(
    destination: Path, *, checkpoint_sha256: str, runtime_config_sha256: str
) -> dict[str, Any]:
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    experiment = _load_experiment()
    source_metadata = _read_json(SOURCE_MEMORY / METADATA_FILENAME)
    expected_metadata_sha = experiment["sources"]["scene1_memory_metadata_sha256"]
    if (
        SOURCE_MEMORY.is_symlink()
        or not SOURCE_MEMORY.is_dir()
        or {item.name for item in SOURCE_MEMORY.iterdir()} != {MEMORY_FILENAME, METADATA_FILENAME}
        or any(item.is_symlink() or not item.is_file() for item in SOURCE_MEMORY.iterdir())
        or sha256_file(SOURCE_MEMORY / METADATA_FILENAME) != expected_metadata_sha
        or source_metadata.get("scene_id") != SCENE_ID
        or source_metadata.get("canonical_prefix_sha256") != SOURCE_MEMORY_PREFIX_SHA256
        or source_metadata.get("tensor_file_sha256") != SOURCE_MEMORY_TENSOR_FILE_SHA256
        or sha256_file(SOURCE_MEMORY / MEMORY_FILENAME) != SOURCE_MEMORY_TENSOR_FILE_SHA256
    ):
        raise ValueError("Source V89 scene-memory bytes changed")
    rebound = dict(source_metadata)
    rebound["source_base_checkpoint_sha256"] = checkpoint_sha256
    rebound["runtime_config_sha256"] = runtime_config_sha256
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        shutil.copyfile(SOURCE_MEMORY / MEMORY_FILENAME, temporary / MEMORY_FILENAME)
        _write_json(temporary / METADATA_FILENAME, rebound)
        if sha256_file(temporary / MEMORY_FILENAME) != SOURCE_MEMORY_TENSOR_FILE_SHA256:
            raise RuntimeError("V91 scene-memory bytes changed during rebinding")
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
        raise RuntimeError("V91 canonical scene prefix changed during rebinding")
    return {
        "source_memory_tensor_file_sha256": SOURCE_MEMORY_TENSOR_FILE_SHA256,
        "packaged_memory_tensor_file_sha256": sha256_file(destination / MEMORY_FILENAME),
        "canonical_prefix_sha256": loaded.metadata["canonical_prefix_sha256"],
        "metadata_only_rebinding": True,
        "memory_tensor_file_bytes_unchanged": True,
        "question_data_used_for_rebinding": False,
        "exact_two_file_scene_memory": {item.name for item in destination.iterdir()}
        == {MEMORY_FILENAME, METADATA_FILENAME},
    }


def cleanup_failed_candidate() -> None:
    """Remove only an un-smoked, unpromoted partial V91 package."""

    if (
        SMOKE_REPORT.exists()
        or RELEASE_REPORT.exists()
        or RELEASE_CHECKPOINT.exists()
        or RELEASE_MEMORY.exists()
    ):
        raise RuntimeError("Refusing V91 cleanup after smoke or release evidence exists")
    for root in (CANDIDATE_CHECKPOINT, CANDIDATE_MEMORY):
        if root.is_symlink():
            raise ValueError(f"Refusing to clean symbolic-link candidate: {root}")
        if root.exists():
            shutil.rmtree(root)
    if RUNTIME_CONFIG.exists():
        if RUNTIME_CONFIG.is_symlink():
            raise ValueError("Refusing to clean symbolic-link V91 runtime config")
        RUNTIME_CONFIG.unlink()


def prepare_candidate() -> dict[str, Any]:
    """Package only after all sealed V91 model gates pass exactly."""

    destinations = (
        CANDIDATE_CHECKPOINT,
        CANDIDATE_MEMORY,
        SMOKE_REPORT,
        RELEASE_CHECKPOINT,
        RELEASE_MEMORY,
        RELEASE_REPORT,
    )
    if any(path.exists() or path.is_symlink() for path in destinations):
        raise FileExistsError("V91 runtime candidate destination is not pristine")
    evidence = authenticate_v91_model_gate()
    config = materialize_runtime_config(evidence)
    metadata = build_runtime_metadata(
        evidence,
        promotion="pending_isolated_runtime_smoke",
        smoke_report_sha256=None,
    )
    checkpoint = _atomic_checkpoint(CANDIDATE_CHECKPOINT, metadata=metadata, evidence=evidence)
    memory = _rebind_memory(
        CANDIDATE_MEMORY,
        checkpoint_sha256=str(checkpoint["checkpoint_sha256"]),
        runtime_config_sha256=effective_runtime_config_sha256(config),
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "phase": "v91_strict_runtime_candidate_prepared",
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
    evidence = authenticate_v91_model_gate()
    if not CANDIDATE_CHECKPOINT.is_dir() or not CANDIDATE_MEMORY.is_dir():
        raise FileNotFoundError("V91 strict runtime candidate package is incomplete")
    expected_metadata = build_runtime_metadata(
        evidence,
        promotion="pending_isolated_runtime_smoke",
        smoke_report_sha256=None,
    )
    metadata = _read_json(CANDIDATE_CHECKPOINT / RUNTIME_METADATA_FILENAME)
    if metadata != expected_metadata:
        raise ValueError("V91 candidate runtime metadata changed")
    fingerprint, files = checkpoint_fingerprint(CANDIDATE_CHECKPOINT)
    candidate = load_file(str(CANDIDATE_CHECKPOINT / "adapter.safetensors"), device="cpu")
    parent = load_file(str(PARENT_CHECKPOINT / "adapter.safetensors"), device="cpu")
    expected_tensors, composition = _composed_adapter(evidence)
    config = load_runtime_config(RUNTIME_CONFIG)
    validate_runtime_bank_inventory(
        runtime_config=config,
        checkpoint_metadata=metadata,
        expected_bank_order=EXPECTED_BANKS,
        expected_states={
            str(name): str(value) for name, value in metadata["lora_bank_state_sha256"].items()
        },
    )
    checks = {
        "exact_two_file_checkpoint": {item["path"] for item in files}
        == {"adapter.safetensors", RUNTIME_METADATA_FILENAME},
        "exact_thirteen_bank_order": composition["final_bank_order"]
        == list(EXPECTED_BANKS),
        "exact_four_v90_v91_bridge_tensors_added": composition["added_tensor_count"]
        == 4,
        "v89_parent_tensors_byte_identical": composition["base_tensors_byte_identical"] is True
        and set(parent).issubset(candidate)
        and all(torch.equal(candidate[name], value) for name, value in parent.items()),
        "exact_tensor_inventory": set(candidate) == set(expected_tensors),
        "all_tensor_values_equal": set(candidate) == set(expected_tensors)
        and all(torch.equal(candidate[name], expected_tensors[name]) for name in candidate),
        "exact_adapter_parameter_count": metadata["lora_parameter_count"]
        == EXPECTED_ADAPTER_PARAMETER_COUNT,
        "zero_trainable_runtime_parameters": metadata["lora_trainable_parameter_count"] == 0,
        "scene_memory_bytes_unchanged": sha256_file(CANDIDATE_MEMORY / MEMORY_FILENAME)
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
        raise RuntimeError(f"V91 strict candidate verification failed: {checks}")
    return {
        "phase": "v91_strict_runtime_candidate_verified",
        "checkpoint_sha256": fingerprint,
        "adapter_sha256": sha256_file(CANDIDATE_CHECKPOINT / "adapter.safetensors"),
        "memory_tensor_sha256": sha256_file(CANDIDATE_MEMORY / MEMORY_FILENAME),
        "v91_bridge_state_sha256": evidence["v91_bridge_state_sha256"],
        "checks": checks,
        "passed": True,
    }


def _primary_cases(
    experiment: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], ...]:
    """Return evaluation-only cases; this payload must never enter chat."""

    experiment = _load_experiment() if experiment is None else experiment
    rows = experiment.get("conversational_intents")
    if not isinstance(rows, list) or len(rows) != 13:
        raise ValueError("V91 primary conversational inventory changed")
    cases: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise TypeError("V91 conversational intent row is malformed")
        identifier = row.get("id")
        family = row.get("family")
        wordings = row.get("existing_wordings")
        question = wordings[0] if isinstance(wordings, list) and wordings else None
        expected = row.get("answer")
        if not all(
            isinstance(value, str) and bool(value.strip())
            for value in (identifier, family, question, expected)
        ):
            raise ValueError("V91 primary conversational case is malformed")
        cases.append(
            {
                "intent_id": identifier,
                "family": family,
                "question": question,
                "expected": expected,
                "core_actionable": identifier in CORE_ACTIONABLE_INTENTS,
            }
        )
    identifiers = {str(row["intent_id"]) for row in cases}
    if (
        len(identifiers) != 13
        or len({str(row["question"]) for row in cases}) != 13
        or {str(row["intent_id"]) for row in cases if row["core_actionable"]}
        != CORE_ACTIONABLE_INTENTS
    ):
        raise ValueError("V91 primary conversational case coverage changed")
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
            raise ValueError("V91 smoke question is empty")
        command.extend(("--question", question))
    return command


def _protected_smoke_reads(audit: Mapping[str, Any]) -> list[str]:
    loaded = audit.get("loaded_files")
    if not isinstance(loaded, list) or not all(isinstance(path, str) for path in loaded):
        raise ValueError("V91 child file audit is malformed")
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
    }
    forbidden_components = {"oracle", "qa", "validation", "test", "deferred"}
    forbidden_module_fragments = {
        "v91_scene1_conversational_preflight",
        "train_v91_scene1_conversational_repair",
        "evaluate_v91_scene1_conversational_repair",
        "v91_strict_runtime_release",
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
        root_violation = False
        for root in protected_roots:
            try:
                path.relative_to(root)
            except ValueError:
                continue
            root_violation = True
            break
        if path in exact_files or component_violation or module_violation or root_violation:
            violations.append(str(path))
    return sorted(set(violations))


def _score_behavior(
    rows: Sequence[Mapping[str, Any]], cases: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    # Importing this scorer also imports evaluation-only labels and helpers, so
    # it is intentionally deferred until the external chat process has exited.
    from semantic_3d_chat.evaluation.evaluate_v91_scene1_conversational_repair import (
        conversational_match_v91,
    )

    if len(rows) != len(cases):
        raise ValueError("V91 smoke result count differs from primary inventory")
    behavior: list[dict[str, Any]] = []
    for row, case in zip(rows, cases, strict=True):
        observed = str(row.get("answer", "")).strip()
        passed = conversational_match_v91(
            str(case["intent_id"]),
            str(case["family"]),
            observed,
            case["expected"],
        )
        behavior.append(
            {
                **dict(case),
                "observed": observed,
                "passed": passed,
            }
        )
    return behavior


def validate_runtime_smoke_report_v91(
    smoke: Mapping[str, Any], evidence: Mapping[str, Any]
) -> None:
    """Authenticate a create-once isolated runtime smoke result."""

    cases = _primary_cases()
    behavior = smoke.get("behavior")
    gates = smoke.get("gates")
    if not isinstance(behavior, list) or len(behavior) != 13:
        raise ValueError("V91 runtime smoke behavior inventory changed")
    case_fields_match = all(
        isinstance(row, Mapping)
        and {
            key: row.get(key)
            for key in (
                "intent_id",
                "family",
                "question",
                "expected",
                "core_actionable",
            )
        }
        == dict(case)
        and isinstance(row.get("observed"), str)
        and isinstance(row.get("passed"), bool)
        for row, case in zip(behavior, cases, strict=True)
    )
    correct = sum(row.get("passed") is True for row in behavior)
    core = sum(row.get("passed") is True and row.get("core_actionable") is True for row in behavior)
    if (
        smoke.get("schema_version") != 91
        or smoke.get("artifact") != "gemma4_v91_strict_runtime_smoke_v1"
        or smoke.get("model_gate_evidence_sha256") != _canonical_sha256(dict(evidence))
        or smoke.get("model_gate_report_sha256") != evidence.get("model_gate_report_sha256")
        or smoke.get("training_report_sha256") != evidence.get("training_report_sha256")
        or smoke.get("evaluation_predictions_sha256")
        != evidence.get("evaluation_predictions_sha256")
        or smoke.get("v91_bridge_state_sha256") != evidence.get("v91_bridge_state_sha256")
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
        or smoke.get("environment_conditioned_input_hashes") != [SOURCE_MEMORY_PREFIX_SHA256] * 13
        or not isinstance(gates, Mapping)
        or set(gates) != _REQUIRED_RUNTIME_GATES
        or any(value is not True for value in gates.values())
        or smoke.get("passed") is not True
        or smoke.get("promotion_authorized") is not True
        or smoke.get("post_v90_training_set_development") is not True
        or smoke.get("single_scene_conversational_repair") is not True
        or smoke.get("development_known_primary_questions") is not True
        or smoke.get("newly_held_wording_only") is not True
        or smoke.get("held_out_scene") is not False
        or smoke.get("held_out_generalization_claim") is not False
        or smoke.get("chat_log_sha256") != sha256_file(SMOKE_CHAT)
        or smoke.get("file_audit_sha256") != sha256_file(SMOKE_AUDIT)
    ):
        raise ValueError("V91 runtime smoke evidence did not pass exactly")


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


def run_smoke() -> dict[str, Any]:
    """Run the generic direct-memory child with the oracle renamed away."""

    if SMOKE_REPORT.is_file():
        existing = _read_json(SMOKE_REPORT)
        validate_runtime_smoke_report_v91(existing, authenticate_v91_model_gate())
        return existing
    if SMOKE_CHAT.exists() or SMOKE_AUDIT.exists():
        raise FileExistsError("V91 smoke artifacts exist; results are create-once")
    evidence = authenticate_v91_model_gate()
    candidate = verify_candidate()
    cases = _primary_cases()
    questions = [str(case["question"]) for case in cases]
    command = _smoke_command(questions)
    python = Path(command[0])
    if not python.is_file():
        raise FileNotFoundError("V91 local Gemma Python environment is unavailable")
    if command.count("--question") != 13 or any(
        token in command for token in ("--expected", "--answer", "--reference")
    ):
        raise RuntimeError("V91 child protocol contains an expectation channel")

    oracle = PROJECT_ROOT / "data/oracle"
    unavailable = PROJECT_ROOT / f"data/.oracle-unavailable-v91-{os.getpid()}"
    if not oracle.is_dir() or oracle.is_symlink() or unavailable.exists():
        raise FileNotFoundError("V91 oracle cannot be made physically unavailable")
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
            f"V91 strict runtime smoke failed: returncode={code}\nstdout={stdout}\nstderr={stderr}"
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
        raise RuntimeError("V91 smoke chat rows differ from the question protocol")
    stdout_records: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            stdout_records.append(value)
    startup_records = [
        row for row in stdout_records if row.get("phase") == "v83_direct_fixed_scene_memory_ready"
    ]
    completion_records = [
        row for row in stdout_records if row.get("phase") == "v83_chat_audit_complete"
    ]
    startup = startup_records[0] if len(startup_records) == 1 else {}
    completion = completion_records[0] if len(completion_records) == 1 else {}

    # Evaluation-only expectations enter only now, after subprocess.run ended.
    behavior = _score_behavior(rows, cases)
    correct = sum(row["passed"] is True for row in behavior)
    core_correct = sum(row["passed"] is True and row["core_actionable"] is True for row in behavior)
    audit = _read_json(SMOKE_AUDIT)
    protected_reads = _protected_smoke_reads(audit)
    prefix_hashes = [row.get("prefix_hash") for row in rows]
    # The generic V83 result row names the direct environmental tensor
    # ``prefix_hash``.  There is no separate readout tensor in this runtime, so
    # that same digest is the total environment-conditioned input identity.
    input_hashes = [
        row.get("environment_conditioned_input_sha256", row.get("prefix_hash")) for row in rows
    ]
    lora = startup.get("lora")
    lora_banks = lora.get("banks") if isinstance(lora, Mapping) else None
    exact_banks = bool(
        isinstance(lora_banks, list)
        and [row.get("name") for row in lora_banks if isinstance(row, Mapping)]
        == list(EXPECTED_BANKS)
        and all(isinstance(row, Mapping) and row.get("trainable") is False for row in lora_banks)
    )
    expectations_absent = bool(
        command.count("--question") == 13
        and [command[index + 1] for index, value in enumerate(command) if value == "--question"]
        == questions
        and all(flag not in command for flag in ("--expected", "--answer", "--reference"))
    )
    gates = {
        "model_acceptance_gate_authenticated_and_passed": evidence["model_acceptance_gate_passed"]
        is True,
        "runtime_process_exit_zero": completed.returncode == 0,
        "at_least_twelve_of_thirteen_behavior_assertions_pass": correct >= 12,
        "all_six_core_actionable_intents_pass": core_correct == 6,
        "oracle_physically_unavailable": oracle_unavailable,
        "oracle_restored_after_runtime": oracle.is_dir(),
        "child_audit_completion_passed": completion.get("passed") is True
        and completion.get("fixed_memory_invariant") is True,
        "child_used_exact_thirteen_frozen_banks": exact_banks,
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
        "expected_immutable_scene_prefix": set(prefix_hashes) == {SOURCE_MEMORY_PREFIX_SHA256},
        "exact_direct_memory_layout_every_question": all(
            _direct_layout_passes(row) for row in rows
        ),
        "source_memory_bytes_unchanged": sha256_file(CANDIDATE_MEMORY / MEMORY_FILENAME)
        == SOURCE_MEMORY_TENSOR_FILE_SHA256,
        "expectations_absent_from_child_protocol": expectations_absent,
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "artifact": "gemma4_v91_strict_runtime_smoke_v1",
        "model_gate_evidence_sha256": _canonical_sha256(evidence),
        "model_gate_report_sha256": evidence["model_gate_report_sha256"],
        "training_report_sha256": evidence["training_report_sha256"],
        "evaluation_predictions_sha256": evidence["evaluation_predictions_sha256"],
        "v91_bridge_state_sha256": evidence["v91_bridge_state_sha256"],
        "v90_bridge_state_sha256": evidence["v90_bridge_state_sha256"],
        "candidate_checkpoint_sha256": candidate["checkpoint_sha256"],
        "candidate_adapter_sha256": candidate["adapter_sha256"],
        "candidate_memory_tensor_sha256": candidate["memory_tensor_sha256"],
        "runtime_config_sha256": effective_runtime_config_sha256(
            load_runtime_config(RUNTIME_CONFIG)
        ),
        "chat_log_sha256": sha256_file(SMOKE_CHAT),
        "file_audit_sha256": sha256_file(SMOKE_AUDIT),
        "chat_process_stdout_sha256": hashlib.sha256(completed.stdout.encode()).hexdigest(),
        "chat_process_stderr_sha256": hashlib.sha256(completed.stderr.encode()).hexdigest(),
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
        "post_v90_training_set_development": True,
        "single_scene_conversational_repair": True,
        "development_known_primary_questions": True,
        "newly_held_wording_only": True,
        "held_out_scene": False,
        "held_out_generalization_claim": False,
    }
    _write_json(SMOKE_REPORT, report)
    return report


def promote_release() -> dict[str, Any]:
    """Promote create-once only after the exact isolated smoke passes."""

    evidence = authenticate_v91_model_gate()
    if (
        RELEASE_CHECKPOINT.exists()
        or RELEASE_CHECKPOINT.is_symlink()
        or RELEASE_MEMORY.exists()
        or RELEASE_MEMORY.is_symlink()
        or RELEASE_REPORT.exists()
        or RELEASE_REPORT.is_symlink()
    ):
        raise FileExistsError("V91 strict runtime release destination already exists")
    smoke = _read_json(SMOKE_REPORT)
    validate_runtime_smoke_report_v91(smoke, evidence)
    candidate = verify_candidate()
    if (
        smoke.get("candidate_checkpoint_sha256") != candidate["checkpoint_sha256"]
        or smoke.get("candidate_adapter_sha256") != candidate["adapter_sha256"]
        or smoke.get("candidate_memory_tensor_sha256") != candidate["memory_tensor_sha256"]
        or smoke.get("runtime_config_sha256")
        != effective_runtime_config_sha256(load_runtime_config(RUNTIME_CONFIG))
    ):
        raise ValueError("V91 smoked candidate bytes changed before promotion")
    smoke_sha256 = sha256_file(SMOKE_REPORT)
    metadata = build_runtime_metadata(
        evidence,
        promotion=PROMOTION_DECISION,
        smoke_report_sha256=smoke_sha256,
    )
    checkpoint = _atomic_checkpoint(
        RELEASE_CHECKPOINT,
        metadata=metadata,
        evidence=evidence,
        source_adapter=CANDIDATE_CHECKPOINT / "adapter.safetensors",
    )
    if checkpoint["adapter_sha256"] != candidate["adapter_sha256"]:
        raise RuntimeError("Promoted V91 adapter differs from smoked candidate")
    config = load_runtime_config(RUNTIME_CONFIG)
    memory = _rebind_memory(
        RELEASE_MEMORY,
        checkpoint_sha256=str(checkpoint["checkpoint_sha256"]),
        runtime_config_sha256=effective_runtime_config_sha256(config),
    )
    release = {
        "schema_version": SCHEMA_VERSION,
        "artifact": "gemma4_v91_strict_runtime_release_v1",
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
        "post_v90_training_set_development": True,
        "single_scene_conversational_repair": True,
        "development_known_primary_questions": True,
        "newly_held_wording_only": True,
        "held_out_scene": False,
        "held_out_generalization_claim": False,
        "runtime_config": _relative(RUNTIME_CONFIG),
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
    evidence = authenticate_v91_model_gate()
    smoke = _read_json(SMOKE_REPORT)
    validate_runtime_smoke_report_v91(smoke, evidence)
    release = _read_json(RELEASE_REPORT)
    metadata = _read_json(RELEASE_CHECKPOINT / RUNTIME_METADATA_FILENAME)
    smoke_sha256 = sha256_file(SMOKE_REPORT)
    expected_metadata = build_runtime_metadata(
        evidence,
        promotion=PROMOTION_DECISION,
        smoke_report_sha256=smoke_sha256,
    )
    if metadata != expected_metadata:
        raise ValueError("V91 promoted runtime metadata changed")
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
    provenance = metadata["initialization_provenance"]["v91_strict_runtime_release"]
    validate_runtime_bank_inventory(
        runtime_config=config,
        checkpoint_metadata=metadata,
        expected_bank_order=EXPECTED_BANKS,
        expected_states={
            str(name): str(value) for name, value in metadata["lora_bank_state_sha256"].items()
        },
    )
    parent = load_file(str(PARENT_CHECKPOINT / "adapter.safetensors"), device="cpu")
    promoted = load_file(str(RELEASE_CHECKPOINT / "adapter.safetensors"), device="cpu")
    release_bindings = release.get("bindings")
    checks = {
        "release_report_identity": release.get("artifact") == "gemma4_v91_strict_runtime_release_v1"
        and release.get("schema_version") == 91
        and release.get("all_release_gates_passed") is True,
        "release_report_promoted": release.get("promotion_decision") == PROMOTION_DECISION,
        "exact_two_file_checkpoint": {item["path"] for item in files}
        == {"adapter.safetensors", RUNTIME_METADATA_FILENAME},
        "checkpoint_fingerprint_matches_release": isinstance(release.get("checkpoint"), Mapping)
        and fingerprint == release["checkpoint"].get("checkpoint_sha256"),
        "adapter_matches_smoked_candidate": sha256_file(RELEASE_CHECKPOINT / "adapter.safetensors")
        == sha256_file(CANDIDATE_CHECKPOINT / "adapter.safetensors")
        == smoke.get("candidate_adapter_sha256"),
        "v89_parent_tensors_byte_identical": set(parent).issubset(promoted)
        and all(torch.equal(promoted[name], value) for name, value in parent.items()),
        "exact_four_v90_v91_tensors_added": len(set(promoted) - set(parent)) == 4,
        "memory_bytes_match_source": sha256_file(RELEASE_MEMORY / MEMORY_FILENAME)
        == SOURCE_MEMORY_TENSOR_FILE_SHA256,
        "memory_prefix_match_source": loaded.metadata["canonical_prefix_sha256"]
        == SOURCE_MEMORY_PREFIX_SHA256,
        "model_gate_binding_exact": provenance["model_gate_report_sha256"]
        == evidence["model_gate_report_sha256"],
        "runtime_smoke_binding_exact": provenance["smoke_report_sha256"] == smoke_sha256
        and isinstance(release_bindings, Mapping)
        and release_bindings.get("runtime_smoke_sha256") == smoke_sha256,
        "v91_state_binding_exact": provenance["v91_bridge_state_sha256"]
        == evidence["v91_bridge_state_sha256"],
        "v90_state_binding_exact": provenance["v90_bridge_state_sha256"]
        == V90_STATE_SHA256,
        "exact_thirteen_frozen_banks": tuple(
            row["name"] for row in metadata["lora"]["banks"]
        )
        == EXPECTED_BANKS
        and metadata["lora"]["trainable_adapter_parameter_count"] == 0
        and metadata["lora"]["adapter_parameter_count"] == EXPECTED_ADAPTER_PARAMETER_COUNT,
        "runtime_promotion_authorized": provenance["runtime_promotion_authorized"] is True,
        "no_held_out_scene_claim": release.get("held_out_scene") is False
        and release.get("held_out_generalization_claim") is False,
    }
    if not all(checks.values()):
        raise RuntimeError(f"V91 strict runtime release verification failed: {checks}")
    return {
        "phase": "v91_strict_runtime_release_verified",
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
        "authenticate": authenticate_v91_model_gate,
        "prepare": prepare_candidate,
        "verify-candidate": verify_candidate,
        "smoke": run_smoke,
        "promote": promote_release,
        "verify": verify_release,
        "cleanup-failed-candidate": lambda: (
            cleanup_failed_candidate() or {"phase": "v91_failed_candidate_cleaned", "passed": True}
        ),
    }
    try:
        result = functions[args.command]()
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"V91 strict runtime {args.command} refused: {error}", file=sys.stderr)
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
    "PARENT_CHECKPOINT",
    "PREREGISTRATION_SHA256",
    "RELEASE_CHECKPOINT",
    "RELEASE_MEMORY",
    "RUNTIME_CONFIG",
    "SMOKE_REPORT",
    "SOURCE_MEMORY_PREFIX_SHA256",
    "SOURCE_MEMORY_TENSOR_FILE_SHA256",
    "V91_BANK",
    "V91_BRIDGE_CANDIDATE",
    "authenticate_v91_model_gate",
    "build_runtime_config_payload",
    "build_runtime_metadata",
    "cleanup_failed_candidate",
    "main",
    "materialize_runtime_config",
    "prepare_candidate",
    "promote_release",
    "run_smoke",
    "validate_model_gate_contract_v91",
    "validate_runtime_smoke_report_v91",
    "verify_candidate",
    "verify_release",
]
