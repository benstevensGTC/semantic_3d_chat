"""Claim-bounded, artifact-only reporting support for V89.

The inspector authenticates exact source, training, candidate, prediction,
evaluation, and post-hoc-figure bytes without loading Gemma, QA source rows,
scene memory, or oracle data. A passing model-level evaluation is intentionally
insufficient for a ``runtime_ready`` claim: an independently authenticated
strict-runtime smoke must also pass while the oracle is physically unavailable
and the runtime file audit remains clean.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

import yaml

from semantic_3d_chat.config import PROJECT_ROOT

V89_CONFIG: Final[Path] = Path("configs/experiments/gemma4_v89_scene1_retention_demo.yaml")
V89_PREREGISTRATION: Final[Path] = Path(
    "reports/gemma4/metrics/gemma4_v89_scene1_retention_preregistration.json"
)
V89_CPU_PREFLIGHT: Final[Path] = Path(
    "reports/gemma4/metrics/gemma4_v89_scene1_retention_cpu_preflight.json"
)
V89_TRAINING: Final[Path] = Path("reports/gemma4/metrics/gemma4_v89_scene1_retention_training.json")
V89_FIXED_FINAL_WEIGHTS: Final[Path] = Path(
    "reports/gemma4/artifacts/v89_scene1_retention_final/bridge.safetensors"
)
V89_FIXED_FINAL_METADATA: Final[Path] = Path(
    "reports/gemma4/artifacts/v89_scene1_retention_final/runtime_metadata.json"
)
V89_EVALUATION_PREDICTIONS: Final[Path] = Path(
    "reports/gemma4/predictions/gemma4_v89_scene1_retention_evaluation.json"
)
V89_EVALUATION: Final[Path] = Path(
    "reports/gemma4/metrics/gemma4_v89_scene1_retention_evaluation.json"
)
V89_RUNTIME_CANDIDATE: Final[Path] = Path("reports/gemma4/artifacts/v89_scene1_retention_runtime")
V89_RUNTIME_SMOKE: Final[Path] = Path(
    "reports/gemma4/metrics/gemma4_v89_scene1_retention_runtime_smoke.json"
)
V89_RUNTIME_FILE_AUDIT: Final[Path] = Path(
    "reports/gemma4/metrics/v89_strict_runtime_smoke_access.json"
)
V89_RUNTIME_RELEASE: Final[Path] = Path(
    "reports/gemma4/metrics/gemma4_v89_strict_runtime_release.json"
)
V89_RELEASE_CONFIG: Final[Path] = Path("configs/runtime/gemma4_v89_strict_scene1.yaml")
V89_RELEASE_ADAPTER: Final[Path] = Path(
    "data_gemma4/runtime/checkpoints/gemma4_v89_strict_scene1_release_v1/adapter.safetensors"
)
V89_RELEASE_METADATA: Final[Path] = Path(
    "data_gemma4/runtime/checkpoints/gemma4_v89_strict_scene1_release_v1/runtime_metadata.json"
)
V89_RELEASE_MEMORY: Final[Path] = Path(
    "data_gemma4/runtime/scene_memories/v89/scene_000001/memory.safetensors"
)
V89_RELEASE_MEMORY_METADATA: Final[Path] = Path(
    "data_gemma4/runtime/scene_memories/v89/scene_000001/runtime_metadata.json"
)
V89_ACCURACY_FIGURE: Final[Path] = Path("reports/gemma4/figures/v89_scene1_accuracy_by_type.png")
V89_ACCURACY_FIGURE_SUMMARY: Final[Path] = Path(
    "reports/gemma4/examples/v89_scene1_accuracy_by_type.json"
)

V89_SEALED_SOURCE_SHA256: Final[dict[Path, str]] = {
    V89_CONFIG: "6781126de7e378a27e9a4140d2e47efb7b673c8a0b3522dd762fea5214312e2c",
    Path("src/semantic_3d_chat/evaluation/v89_scene1_retention_preflight.py"): (
        "c64d63309676fae32df10f38debaa0131409347822c1f5578690a09b0c345e29"
    ),
    Path("src/semantic_3d_chat/training/train_v89_scene1_retention.py"): (
        "51e13fbd1109f4f28d2775fa0c03868be8e4ea50590a8e854b9c4fa436c55c07"
    ),
    Path("src/semantic_3d_chat/evaluation/evaluate_v89_scene1_retention.py"): (
        "de2ba05a30e7a3d3a1bc77e2b6a30f1c2c8240c4729b5a5f4dfdb1f9dcd00c1d"
    ),
    V89_PREREGISTRATION: ("493208eb96b6bfe14267ebc05612441457a2b52751f0ca06e4fb90fab84d94a9"),
    V89_CPU_PREFLIGHT: ("bc063b1cbad1d05a53e0044bfcc80f6d52f994e1ef85e0b5ed351469a987e256"),
}

V89_SEALED_RESULT_SHA256: Final[dict[Path, str]] = {
    V89_TRAINING: "1980a694bca8268056ca7485c755127c9fc94fd2cd377de3e5556ccaba887ea6",
    V89_FIXED_FINAL_WEIGHTS: ("570e6e582664d21964e110e93036cfb24f35aff59f598f049f87008bffda89b6"),
    V89_FIXED_FINAL_METADATA: ("dcc62e09e83ae4649c6f30b2a350047a8077e57355258580f5ddc802a2ffa727"),
    V89_EVALUATION_PREDICTIONS: (
        "49ff7b9fd4010f6d3201c5ee22ca25ed11d035617910b722a53c52956b9df640"
    ),
    V89_EVALUATION: ("c880c0707c6c87783a20134984914e4a0c6cace4d4d70c7d202ee2a527ee87a2"),
    Path("src/semantic_3d_chat/evaluation/v89_accuracy_figure.py"): (
        "44ea9a6e53cf322e641d0193c671b2bb521822c91f1e11973ff74a902e57d186"
    ),
    Path("tests/test_v89_accuracy_figure.py"): (
        "cc9129224a0243b8a49bfa4bf71a6ea5d81caf004f277ebccbcbcbfc58180177"
    ),
    V89_ACCURACY_FIGURE: ("3442492e4fe6bf86bba21a85f6c1864ee74ece625854a794adad34dafc859e46"),
    V89_ACCURACY_FIGURE_SUMMARY: (
        "e2d97a66739c0e58677bee264a34ccf092bda997c5328f7095e369ec575a7d08"
    ),
}

V89_SEALED_RUNTIME_SHA256: Final[dict[Path, str]] = {
    V89_RUNTIME_SMOKE: ("99e9cf6e631ceb0cbdc4d26adc5218c7d84fcd3d78f23e050d6a486e07c0c0b3"),
    V89_RUNTIME_FILE_AUDIT: ("adf71c8afd7c327302bb5c905d5caeb9a2a3af5171270aea6f059a64e4a7a00e"),
    V89_RUNTIME_RELEASE: ("fe414043ad80c6e43fdc424c695e6856600c96a44446b6fab5409d0ac06d45cb"),
    V89_RELEASE_ADAPTER: ("78856650751f8c700a651553e02d95ac389be53e12a56c39576a6d2f23d7d386"),
    V89_RELEASE_METADATA: ("c375d729d0a23e515eef979129a7032479fe153ad1b5491d594bf61a9f1f9a72"),
    V89_RELEASE_MEMORY: ("3427851432d8f2a3609b6205b18b8c0d9a0fcf68d8f3bf0c98c758ac64209ffb"),
    V89_RELEASE_MEMORY_METADATA: (
        "ea3a45080b3a0c519292a5e230fa1c4e17a1caafa6f3ff9d23adfd9ef1c2befd"
    ),
}

V89_PENDING_RUNTIME_PATHS: Final[tuple[Path, ...]] = (V89_RUNTIME_CANDIDATE,)
# Backward-compatible name retained for callers created before measurements
# completed; only genuinely pending runtime evidence is listed now.
V89_PENDING_RESULT_PATHS: Final[tuple[Path, ...]] = V89_PENDING_RUNTIME_PATHS

_EXPECTED_TRANSITIONS: Final[dict[str, int]] = {
    "retained_correct": 83,
    "recovered": 24,
    "regressed": 20,
    "retained_wrong": 11,
}
_EXPECTED_ERROR_TYPES: Final[dict[str, int]] = {
    "attribute": 7,
    "presence": 1,
    "spatial_relation": 22,
    "support": 1,
}
_EXPECTED_ACCURACY_BY_TYPE: Final[dict[str, tuple[int, int]]] = {
    "attribute": (15, 18),
    "count": (9, 9),
    "metric": (1, 1),
    "presence": (22, 22),
    "spatial_relation": (74, 86),
    "support": (1, 2),
}
_EXPECTED_SMOKE_QUESTIONS: Final[tuple[str, str, str]] = (
    "Is there a chair?",
    "What color is the bowl?",
    "Is the bowl left or right of the chair?",
)
_EXPECTED_SMOKE_ANSWERS: Final[tuple[str, str, str]] = ("yes", "red", "left")
_REQUIRED_TRAINING_GATES: Final[frozenset[str]] = frozenset(
    {
        "all_107_v88_correct_anchors_replayed_once_each_epoch",
        "all_138_canonical_rows_consumed_once_each_epoch",
        "all_18_causal_margin_rows_consumed",
        "all_31_v88_errors_replayed_twice_each_epoch",
        "all_3_development_smoke_rows_consumed_once_each_epoch",
        "all_930_sealed_micro_rows_consumed",
        "fixed_final_update_155_reached",
        "memory_hash_invariant",
        "nonzero_finite_gradient_every_update",
        "protected_read_count_zero",
        "runtime_candidate_contains_no_training_rows_or_answers",
        "zero_payload_hash_invariant",
    }
)
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


def _resolve(root: Path, path: Path) -> Path:
    resolved = (root / path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"V89 reporting path escaped project root: {path}") from error
    return resolved


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected one JSON object: {path}")
    return value


def classify_v89_runtime_readiness(
    *,
    evaluation: Mapping[str, Any] | None,
    runtime_smoke: Mapping[str, Any] | None,
    evaluation_authenticated: bool,
    runtime_smoke_authenticated: bool,
) -> dict[str, Any]:
    """Return a fail-closed readiness classification from already sealed evidence."""

    metrics = evaluation.get("metrics") if isinstance(evaluation, Mapping) else None
    model_gates = metrics.get("model_acceptance_gates") if isinstance(metrics, Mapping) else None
    canonical = metrics.get("canonical_type_specific") if isinstance(metrics, Mapping) else None
    smoke = metrics.get("generic_smoke") if isinstance(metrics, Mapping) else None
    model_gate_passed = bool(
        evaluation_authenticated
        and isinstance(evaluation, Mapping)
        and evaluation.get("artifact") == "gemma4_v89_scene1_retention_evaluation_v1"
        and evaluation.get("schema_version") == 89
        and evaluation.get("status") == "model_gates_pass_separate_runtime_packaging_required"
        and isinstance(metrics, Mapping)
        and metrics.get("model_acceptance_gate_passed") is True
        and isinstance(model_gates, Mapping)
        and set(model_gates) == _REQUIRED_MODEL_GATES
        and all(value is True for value in model_gates.values())
        and isinstance(canonical, Mapping)
        and canonical.get("total") == 138
        and int(canonical.get("correct", -1)) >= 111
        and float(canonical.get("accuracy", -1.0)) >= 0.80
        and isinstance(smoke, Mapping)
        and smoke.get("correct") == 3
        and smoke.get("total") == 3
        and smoke.get("development_known_and_trained") is True
        and smoke.get("held_out") is False
        and evaluation.get("development_known_smoke_trained") is True
        and evaluation.get("held_out_smoke_claim") is False
        and evaluation.get("held_out_generalization_claim") is False
        and evaluation.get("parent_v85_v86_v87_v88_mutated") is False
        and evaluation.get("separate_runtime_packaging_authorized") is True
        and evaluation.get("automatic_runtime_promotion") is False
        and evaluation.get("runtime_promotion_authorized") is False
        and evaluation.get("oracle_loaded") is False
    )

    runtime_gates = runtime_smoke.get("gates") if isinstance(runtime_smoke, Mapping) else None
    separate_runtime_smoke_passed = bool(
        model_gate_passed
        and runtime_smoke_authenticated
        and isinstance(runtime_smoke, Mapping)
        and runtime_smoke.get("artifact") == "gemma4_v89_strict_runtime_smoke_v1"
        and runtime_smoke.get("schema_version") == 89
        and isinstance(runtime_gates, Mapping)
        and set(runtime_gates) == _REQUIRED_RUNTIME_GATES
        and all(runtime_gates[name] is True for name in _REQUIRED_RUNTIME_GATES)
        and runtime_smoke.get("passed") is True
        and runtime_smoke.get("promotion_authorized") is True
        and runtime_smoke.get("expected_behavior_not_loaded_by_chat_runtime") is True
        and runtime_smoke.get("held_out_generalization_claim") is False
    )
    runtime_ready = model_gate_passed and separate_runtime_smoke_passed
    if runtime_ready:
        reason = "authenticated_model_gate_and_separate_strict_runtime_smoke_passed"
    elif not model_gate_passed:
        reason = "authenticated_passing_model_gate_not_available"
    else:
        reason = "separate_authenticated_strict_runtime_smoke_not_passed"
    return {
        "model_evaluation_authenticated": evaluation_authenticated,
        "model_acceptance_gate_passed": model_gate_passed,
        "separate_runtime_smoke_authenticated": runtime_smoke_authenticated,
        "separate_runtime_smoke_passed": separate_runtime_smoke_passed,
        "runtime_ready": runtime_ready,
        "runtime_ready_reason": reason,
    }


def _authenticate_files(
    *, root: Path, expected_hashes: Mapping[Path, str], label: str
) -> dict[str, str]:
    observed: dict[str, str] = {}
    for relative, expected in expected_hashes.items():
        source = _resolve(root, relative)
        if not source.is_file() or source.is_symlink():
            raise FileNotFoundError(f"V89 {label} is missing or unsafe: {relative}")
        digest = _sha256(source)
        if digest != expected:
            raise ValueError(f"V89 {label} digest differs: {relative}")
        observed[relative.as_posix()] = digest
    return observed


def inspect_v89_reporting_state(
    project_root: str | Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Authenticate all sealed V89 evidence and return bounded measured claims."""

    root = Path(project_root).resolve()
    required_paths = [path.as_posix() for path in V89_SEALED_SOURCE_SHA256]
    try:
        observed_hashes = _authenticate_files(
            root=root,
            expected_hashes=V89_SEALED_SOURCE_SHA256,
            label="sealed source",
        )

        config_payload = yaml.safe_load(_resolve(root, V89_CONFIG).read_text(encoding="utf-8"))
        if not isinstance(config_payload, Mapping) or set(config_payload) != {"v89"}:
            raise ValueError("V89 configuration root differs")
        config = config_payload["v89"]
        preregistration = _read_object(_resolve(root, V89_PREREGISTRATION))
        preflight = _read_object(_resolve(root, V89_CPU_PREFLIGHT))
        if not isinstance(config, Mapping):
            raise TypeError("V89 configuration is not a mapping")
        contract = config.get("strict_input_contract")
        dataset = config.get("dataset")
        bridge = config.get("bridge")
        training = config.get("training")
        gates = config.get("gates")
        scope = config.get("scope")
        sources = config.get("sources")
        if not all(
            isinstance(value, Mapping)
            for value in (contract, dataset, bridge, training, gates, scope, sources)
        ):
            raise TypeError("V89 sealed configuration sections differ")

        config_ok = (
            config.get("schema_version") == 89
            and config.get("artifact") == "gemma4_v89_scene1_retention_direct_memory_overfit_v1"
            and config.get("status") == "preregistered_before_full_model_load"
            and contract.get("shape") == [1, 738, 1536]
            and contract.get("payload_tokens") == 736
            and contract.get("compiled_before_question") is True
            and contract.get("reused_byte_identically_across_questions") is True
            and contract.get("all_memory_slots_retained") is True
            and contract.get("question_derived_environmental_tokens") == 0
            and contract.get("question_conditioned_environmental_readout") is False
            and contract.get("question_dependent_retrieval") is False
            and contract.get("control_tokens") == 0
            and contract.get("environmental_text_inputs") == []
            and dataset.get("scene_id") == "scene_000001"
            and dataset.get("canonical_row_count") == 138
            and dataset.get("parent_v87_to_v88_transition_counts") == _EXPECTED_TRANSITIONS
            and dataset.get("parent_v88_error_count") == 31
            and dataset.get("parent_v88_error_type_counts") == _EXPECTED_ERROR_TYPES
            and dataset.get("parent_v88_correct_anchor_count") == 107
            and dataset.get("total_rows_per_epoch") == 310
            and dataset.get("runtime_serializes_questions_or_answers") is False
            and dataset.get("runtime_serializes_error_inventory") is False
            and dataset.get("runtime_serializes_anchor_inventory") is False
            and bridge.get("bank_name") == "v89_scene1_retention_bridge"
            and bridge.get("target_module") == "model.language_model.layers.27.self_attn.o_proj"
            and bridge.get("rank") == 8
            and bridge.get("trainable_parameter_count") == 28_672
            and training.get("epochs") == 3
            and training.get("rows_per_epoch") == 310
            and training.get("optimizer_updates") == 155
            and training.get("total_causal_margin_rows") == 18
            and training.get("checkpoint_selection") == "fixed_final_update_155"
            and training.get("intermediate_behavior_selection") is False
            and gates.get("live_smoke_required_correct") == 3
            and gates.get("live_smoke_total") == 3
            and gates.get("live_smoke_is_development_known_and_trained") is True
            and gates.get("live_smoke_is_held_out") is False
            and gates.get("runtime_promotion_only_after_all_gates") is True
            and scope.get("post_v88_training_set_development") is True
            and scope.get("single_scene_overfit_demonstration") is True
            and scope.get("retention_aware_error_correction") is True
            and scope.get("development_known_smoke") is True
            and scope.get("held_out_generalization_claim") is False
            and scope.get("runtime_promotion_authorized") is False
            and sources.get("preflight_source_sha256")
            == V89_SEALED_SOURCE_SHA256[
                Path("src/semantic_3d_chat/evaluation/v89_scene1_retention_preflight.py")
            ]
            and sources.get("training_source_sha256")
            == V89_SEALED_SOURCE_SHA256[
                Path("src/semantic_3d_chat/training/train_v89_scene1_retention.py")
            ]
            and sources.get("evaluation_source_sha256")
            == V89_SEALED_SOURCE_SHA256[
                Path("src/semantic_3d_chat/evaluation/evaluate_v89_scene1_retention.py")
            ]
        )
        protocol = preregistration.get("protocol_preflight")
        lora = preregistration.get("lora_cpu_preflight")
        preregistration_ok = (
            preregistration.get("artifact") == "gemma4_v89_scene1_retention_preregistration_v1"
            and preregistration.get("schema_version") == 89
            and preregistration.get("status")
            == "sealed_after_v88_failure_before_first_v89_full_model_load"
            and preregistration.get("config_path") == V89_CONFIG.as_posix()
            and preregistration.get("config_sha256") == V89_SEALED_SOURCE_SHA256[V89_CONFIG]
            and preregistration.get("strict_input_contract") == contract
            and preregistration.get("dataset_contract") == dataset
            and preregistration.get("training_protocol") == training
            and preregistration.get("fixed_unchanged_gates") == gates
            and preregistration.get("post_v88_training_set_development") is True
            and preregistration.get("retention_aware_error_correction") is True
            and preregistration.get("development_known_smoke_trained") is True
            and preregistration.get("held_out_smoke_claim") is False
            and preregistration.get("new_v89_behavior_scored") is False
            and preregistration.get("full_gemma_model_loaded") is False
            and preregistration.get("optimizer_constructed") is False
            and preregistration.get("optimizer_updates") == 0
            and preregistration.get("oracle_loaded") is False
            and preregistration.get("runtime_promotion_authorized") is False
            and isinstance(protocol, Mapping)
            and protocol.get("parent_v87_to_v88_transition_counts") == _EXPECTED_TRANSITIONS
            and protocol.get("parent_v88_error_count") == 31
            and protocol.get("parent_v88_correct_anchor_count") == 107
            and protocol.get("training_rows_per_epoch") == 310
            and protocol.get("schedule_rows") == 930
            and protocol.get("optimizer_updates") == 155
            and protocol.get("causal_margin_rows_total") == 18
            and isinstance(lora, Mapping)
            and lora.get("target_modules") == ["model.language_model.layers.27.self_attn.o_proj"]
            and lora.get("parameter_count") == 28_672
            and lora.get("lora_b_nonzero_count") == 0
            and lora.get("exact_zero_output_at_initialization") is True
        )
        preflight_ok = (
            preflight.get("artifact") == "gemma4_v89_scene1_retention_cpu_preflight_v1"
            and preflight.get("schema_version") == 89
            and preflight.get("status") == "passed"
            and preflight.get("passed") is True
            and preflight.get("config_sha256") == V89_SEALED_SOURCE_SHA256[V89_CONFIG]
            and preflight.get("preregistration_sha256")
            == V89_SEALED_SOURCE_SHA256[V89_PREREGISTRATION]
            and preflight.get("fixed_final_optimizer_updates") == 155
            and preflight.get("fixed_final_checkpoint_selection") == "fixed_final_update_155"
            and preflight.get("development_known_smoke_trained") is True
            and preflight.get("held_out_smoke_claim") is False
            and preflight.get("all_738_memory_slots_retained") is True
            and preflight.get("question_derived_environmental_tokens") == 0
            and preflight.get("question_conditioned_environmental_readout") is False
            and preflight.get("full_gemma_model_loaded") is False
            and preflight.get("optimizer_constructed") is False
            and preflight.get("optimizer_updates") == 0
            and preflight.get("new_v89_behavior_scored") is False
            and preflight.get("oracle_loaded") is False
            and preflight.get("runtime_promotion_authorized") is False
        )
        if not all((config_ok, preregistration_ok, preflight_ok)):
            raise ValueError("V89 sealed source/preregistration semantics differ")

        result_hashes = _authenticate_files(
            root=root,
            expected_hashes=V89_SEALED_RESULT_SHA256,
            label="sealed result",
        )
        runtime_hashes = _authenticate_files(
            root=root,
            expected_hashes=V89_SEALED_RUNTIME_SHA256,
            label="sealed runtime evidence",
        )
        training_result = _read_object(_resolve(root, V89_TRAINING))
        candidate = _read_object(_resolve(root, V89_FIXED_FINAL_METADATA))
        predictions = _read_object(_resolve(root, V89_EVALUATION_PREDICTIONS))
        evaluation = _read_object(_resolve(root, V89_EVALUATION))
        figure_summary = _read_object(_resolve(root, V89_ACCURACY_FIGURE_SUMMARY))
        runtime_smoke = _read_object(_resolve(root, V89_RUNTIME_SMOKE))
        runtime_audit = _read_object(_resolve(root, V89_RUNTIME_FILE_AUDIT))
        runtime_release = _read_object(_resolve(root, V89_RUNTIME_RELEASE))
        release_metadata = _read_object(_resolve(root, V89_RELEASE_METADATA))
        release_memory_metadata = _read_object(_resolve(root, V89_RELEASE_MEMORY_METADATA))

        config_sha256 = V89_SEALED_SOURCE_SHA256[V89_CONFIG]
        prereg_sha256 = V89_SEALED_SOURCE_SHA256[V89_PREREGISTRATION]
        preflight_sha256 = V89_SEALED_SOURCE_SHA256[V89_CPU_PREFLIGHT]
        training_sha256 = V89_SEALED_RESULT_SHA256[V89_TRAINING]
        weights_sha256 = V89_SEALED_RESULT_SHA256[V89_FIXED_FINAL_WEIGHTS]
        predictions_sha256 = V89_SEALED_RESULT_SHA256[V89_EVALUATION_PREDICTIONS]
        evaluation_sha256 = V89_SEALED_RESULT_SHA256[V89_EVALUATION]
        smoke_sha256 = V89_SEALED_RUNTIME_SHA256[V89_RUNTIME_SMOKE]

        history = training_result.get("training_history")
        training_gates = training_result.get("gates")
        inventory = training_result.get("training_inventory")
        trained_bridge = training_result.get("trainable_bridge")
        trained_candidate = training_result.get("candidate")
        trained_memory = training_result.get("scene_memory")
        training_ok = bool(
            isinstance(history, list)
            and isinstance(training_gates, Mapping)
            and isinstance(inventory, Mapping)
            and isinstance(trained_bridge, Mapping)
            and isinstance(trained_candidate, Mapping)
            and isinstance(trained_memory, Mapping)
            and training_result.get("artifact") == "gemma4_v89_scene1_retention_training_v1"
            and training_result.get("schema_version") == 89
            and training_result.get("status") == "fixed_final_training_complete_not_promoted"
            and training_result.get("config_sha256") == config_sha256
            and training_result.get("preregistration_sha256") == prereg_sha256
            and training_result.get("cpu_preflight_sha256") == preflight_sha256
            and training_result.get("device") == "mps"
            and training_result.get("micro_rows_consumed") == 930
            and training_result.get("optimizer_updates") == 155
            and training_result.get("causal_margin_rows_consumed") == 18
            and len(history) == 155
            and all(
                row.get("update") == index
                and math.isfinite(float(row.get("gradient_l2_before_clip")))
                and float(row["gradient_l2_before_clip"]) > 0.0
                for index, row in enumerate(history, start=1)
            )
            and set(training_gates) == _REQUIRED_TRAINING_GATES
            and all(value is True for value in training_gates.values())
            and inventory.get("canonical_unique_rows") == 138
            and inventory.get("parent_v88_errors") == 31
            and inventory.get("parent_v88_correct_anchors") == 107
            and inventory.get("unique_schedule_items_per_epoch") == 310
            and inventory.get("development_known_smoke_trained") is True
            and inventory.get("held_out_smoke_claim") is False
            and inventory.get("answers_or_questions_serialized_in_candidate") is False
            and inventory.get("inventory_serialized_in_candidate") is False
            and trained_bridge.get("parameter_count") == 28_672
            and trained_bridge.get("target_module")
            == "model.language_model.layers.27.self_attn.o_proj"
            and trained_bridge.get("initial_state_sha256")
            == "5686d458589b7b39599eb9e865d08ef709a43e7ca36b7496d212b1f7581dc83d"
            and trained_bridge.get("final_state_sha256")
            == "de2388828b4a95770e6e55639baa4538a8360ab1323b68b05ab915aaaba68bd8"
            and trained_candidate.get("fixed_final") is True
            and trained_candidate.get("weights_sha256") == weights_sha256
            and trained_candidate.get("runtime_promotion_authorized") is False
            and trained_memory.get("shape") == [1, 738, 1536]
            and trained_memory.get("prefix_sha256_before")
            == trained_memory.get("prefix_sha256_after")
            and trained_memory.get("zero_payload_prefix_sha256_before")
            == trained_memory.get("zero_payload_prefix_sha256_after")
            and training_result.get("loaded_file_count") == 87
            and training_result.get("protected_read_count") == 0
            and training_result.get("oracle_loaded") is False
            and training_result.get("official_validation_loaded") is False
            and training_result.get("official_test_loaded") is False
            and training_result.get("deferred_final_loaded") is False
            and training_result.get("held_out_generalization_claim") is False
            and training_result.get("runtime_promotion_authorized") is False
            and math.isfinite(float(training_result.get("elapsed_seconds")))
            and float(training_result["elapsed_seconds"]) > 0.0
        )

        bindings = candidate.get("bindings")
        candidate_ok = bool(
            isinstance(bindings, Mapping)
            and candidate.get("artifact") == "gemma4_v89_scene1_retention_fixed_final_v1"
            and candidate.get("schema_version") == 89
            and candidate.get("status") == "fixed_final_awaiting_preregistered_acceptance_gates"
            and candidate.get("bank_name") == "v89_scene1_retention_bridge"
            and candidate.get("target_module") == "model.language_model.layers.27.self_attn.o_proj"
            and candidate.get("rank") == 8
            and float(candidate.get("alpha")) == 16.0
            and float(candidate.get("dropout")) == 0.0
            and candidate.get("parameter_count") == 28_672
            and candidate.get("frozen_bank_count") == 10
            and candidate.get("total_bank_count") == 11
            and candidate.get("state_sha256") == trained_bridge.get("final_state_sha256")
            and candidate.get("weights_sha256") == weights_sha256
            and all(
                candidate.get(name) is False
                for name in (
                    "environmental_memory_serialized",
                    "questions_or_answers_serialized",
                    "training_metadata_serialized",
                    "error_inventory_serialized",
                    "anchor_inventory_serialized",
                    "oracle_serialized",
                    "evaluation_scored",
                    "runtime_promotion_authorized",
                )
            )
            and bindings.get("config_sha256") == config_sha256
            and bindings.get("preregistration_sha256") == prereg_sha256
            and bindings.get("cpu_preflight_sha256") == preflight_sha256
            and bindings.get("fixed_final_optimizer_updates") == 155
            and bindings.get("development_known_smoke_trained") is True
        )

        prediction_rows = predictions.get("records")
        prediction_candidate = predictions.get("candidate")
        prediction_memory = predictions.get("scene_memory")
        prediction_leakage = predictions.get("leakage")
        smoke_records = predictions.get("smoke_records")
        causal_records = predictions.get("causal_records")
        if not all(
            isinstance(value, expected_type)
            for value, expected_type in (
                (prediction_rows, list),
                (prediction_candidate, Mapping),
                (prediction_memory, Mapping),
                (prediction_leakage, Mapping),
                (smoke_records, list),
                (causal_records, list),
            )
        ):
            raise TypeError("V89 prediction structure differs")
        observed_types = Counter(row.get("answer_type") for row in prediction_rows)
        correct_types = Counter(
            row.get("answer_type")
            for row in prediction_rows
            if row.get("normalized_prediction") == row.get("reference_answer")
        )
        predictions_ok = bool(
            predictions.get("artifact") == "gemma4_v89_scene1_retention_predictions_v1"
            and predictions.get("schema_version") == 89
            and predictions.get("status") == "fixed_final_evaluation_only_not_runtime"
            and predictions.get("config_sha256") == config_sha256
            and predictions.get("training_report_sha256") == training_sha256
            and predictions.get("row_count") == 138
            and predictions.get("scene_count") == 1
            and len(prediction_rows) == 138
            and len({row.get("question_id") for row in prediction_rows}) == 138
            and {
                name: (correct_types[name], observed_types[name])
                for name in _EXPECTED_ACCURACY_BY_TYPE
            }
            == _EXPECTED_ACCURACY_BY_TYPE
            and all(row.get("scene_id") == "scene_000001" for row in prediction_rows)
            and all(
                row.get("scene_memory_sha256") == prediction_memory.get("prefix_sha256_before")
                for row in prediction_rows
            )
            and prediction_candidate.get("optimizer_updates") == 155
            and prediction_candidate.get("weights_sha256") == weights_sha256
            and prediction_candidate.get("state_sha256") == trained_bridge.get("final_state_sha256")
            and prediction_memory.get("shape") == [1, 738, 1536]
            and prediction_memory.get("prefix_hash_invariant") is True
            and prediction_memory.get("same_prefix_reused_for_every_question") is True
            and prediction_memory.get("question_derived_environmental_tokens") == 0
            and prediction_memory.get("question_conditioned_environmental_readout") is False
            and prediction_memory.get("question_dependent_retrieval") is False
            and prediction_leakage.get("loaded_file_count") == 87
            and prediction_leakage.get("protected_read_count") == 0
            and prediction_leakage.get("protected_reads") == []
            and prediction_leakage.get("oracle_loaded") is False
            and len(causal_records) == 3
            and len(smoke_records) == 3
            and [row.get("normalized_prediction") for row in smoke_records]
            == ["yes", "red", "left"]
            and all(row.get("exact_correct") is True for row in smoke_records)
            and all(
                row.get("development_known_and_trained") is True and row.get("held_out") is False
                for row in smoke_records
            )
            and predictions.get("development_known_smoke_trained") is True
            and predictions.get("held_out_smoke_claim") is False
            and predictions.get("training_references_serialized_in_runtime_candidate") is False
            and predictions.get("error_inventory_serialized_in_runtime_candidate") is False
            and predictions.get("anchor_inventory_serialized_in_runtime_candidate") is False
            and predictions.get("fixed_checkpoint_selected_before_scoring") is True
            and predictions.get("checkpoint_selection_after_scoring") is False
            and predictions.get("runtime_promotion_authorized") is False
        )

        metrics = evaluation.get("metrics")
        overall = metrics.get("canonical_type_specific") if isinstance(metrics, Mapping) else None
        strict = metrics.get("strict_normalized_exact") if isinstance(metrics, Mapping) else None
        by_type = (
            metrics.get("canonical_accuracy_by_answer_type")
            if isinstance(metrics, Mapping)
            else None
        )
        smoke = metrics.get("generic_smoke") if isinstance(metrics, Mapping) else None
        causal = metrics.get("causal_control") if isinstance(metrics, Mapping) else None
        model_gates = (
            metrics.get("model_acceptance_gates") if isinstance(metrics, Mapping) else None
        )
        evaluation_memory = evaluation.get("scene_memory")
        leakage = evaluation.get("leakage")
        if not all(
            isinstance(value, Mapping)
            for value in (
                metrics,
                overall,
                strict,
                by_type,
                smoke,
                causal,
                model_gates,
                evaluation_memory,
                leakage,
            )
        ):
            raise TypeError("V89 evaluation structure differs")
        evaluation_ok = bool(
            evaluation.get("artifact") == "gemma4_v89_scene1_retention_evaluation_v1"
            and evaluation.get("schema_version") == 89
            and evaluation.get("status") == "model_gates_pass_separate_runtime_packaging_required"
            and evaluation.get("config_sha256") == config_sha256
            and evaluation.get("preregistration_sha256") == prereg_sha256
            and evaluation.get("cpu_preflight_sha256") == preflight_sha256
            and evaluation.get("training_report_sha256") == training_sha256
            and evaluation.get("evaluation_predictions_path")
            == V89_EVALUATION_PREDICTIONS.as_posix()
            and evaluation.get("evaluation_predictions_sha256") == predictions_sha256
            and evaluation.get("fixed_checkpoint_selected_before_scoring") is True
            and evaluation.get("checkpoint_selection_after_scoring") is False
            and evaluation.get("development_known_smoke_trained") is True
            and evaluation.get("held_out_smoke_claim") is False
            and evaluation.get("held_out_generalization_claim") is False
            and evaluation.get("official_validation_loaded") is False
            and evaluation.get("official_test_loaded") is False
            and evaluation.get("deferred_final_loaded") is False
            and evaluation.get("oracle_loaded") is False
            and evaluation.get("parent_v85_v86_v87_v88_mutated") is False
            and evaluation.get("automatic_runtime_promotion") is False
            and evaluation.get("separate_runtime_packaging_authorized") is True
            and evaluation.get("runtime_promotion_authorized") is False
            and overall == {"accuracy": 122 / 138, "correct": 122, "total": 138}
            and strict == overall
            and {name: (row.get("correct"), row.get("total")) for name, row in by_type.items()}
            == _EXPECTED_ACCURACY_BY_TYPE
            and metrics.get("model_acceptance_gate_passed") is True
            and set(model_gates) == _REQUIRED_MODEL_GATES
            and all(value is True for value in model_gates.values())
            and smoke.get("correct") == 3
            and smoke.get("total") == 3
            and smoke.get("development_known_and_trained") is True
            and smoke.get("held_out") is False
            and [row.get("normalized_prediction") for row in smoke.get("records", [])]
            == ["yes", "red", "left"]
            and causal.get("row_count") == 3
            and causal.get("canonical_prediction_changes") == 2
            and math.isclose(
                float(causal.get("mean_correct_memory_nll")),
                0.5105967409908772,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            and math.isclose(
                float(causal.get("mean_zero_payload_nll")),
                2.5385982990264893,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            and math.isclose(
                float(causal.get("mean_zero_minus_correct_nll")),
                2.028001558035612,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            and evaluation_memory == prediction_memory
            and leakage == prediction_leakage
            and math.isfinite(float(evaluation.get("elapsed_seconds")))
            and float(evaluation["elapsed_seconds"]) > 0.0
        )

        figure_source = figure_summary.get("source")
        figure_scope = figure_summary.get("scope")
        figure_metrics = figure_summary.get("metrics")
        figure = figure_summary.get("figure")
        figure_ok = bool(
            all(
                isinstance(value, Mapping)
                for value in (figure_source, figure_scope, figure_metrics, figure)
            )
            and figure_summary.get("artifact") == "v89_scene1_accuracy_by_type_posthoc_figure_v1"
            and figure_source.get("path") == V89_EVALUATION.as_posix()
            and figure_source.get("sha256") == evaluation_sha256
            and figure_scope.get("development_known_smoke_trained") is True
            and figure_scope.get("held_out_smoke") is False
            and figure_scope.get("held_out_generalization") is False
            and figure_scope.get("new_inference") is False
            and figure_scope.get("qa_or_oracle_loaded") is False
            and figure_scope.get("runtime_smoke_evidence") is False
            and figure_scope.get("runtime_promotion_authorized") is False
            and figure_metrics.get("overall") == overall
            and figure_metrics.get("acceptance_passed") is True
            and figure_metrics.get("all_model_gates_passed") is True
            and figure.get("path") == V89_ACCURACY_FIGURE.as_posix()
            and figure.get("sha256") == V89_SEALED_RESULT_SHA256[V89_ACCURACY_FIGURE]
        )

        readiness = classify_v89_runtime_readiness(
            evaluation=evaluation,
            runtime_smoke=runtime_smoke,
            evaluation_authenticated=evaluation_ok,
            runtime_smoke_authenticated=True,
        )
        smoke_gates = runtime_smoke.get("gates")
        behavior = runtime_smoke.get("behavior")
        runtime_smoke_ok = bool(
            readiness["runtime_ready"]
            and isinstance(smoke_gates, Mapping)
            and set(smoke_gates) == _REQUIRED_RUNTIME_GATES
            and isinstance(behavior, list)
            and len(behavior) == 3
            and tuple(row.get("question") for row in behavior) == _EXPECTED_SMOKE_QUESTIONS
            and tuple(row.get("expected") for row in behavior) == _EXPECTED_SMOKE_ANSWERS
            and tuple(row.get("observed") for row in behavior) == _EXPECTED_SMOKE_ANSWERS
            and all(row.get("passed") is True for row in behavior)
            and runtime_smoke.get("model_gate_report_sha256") == evaluation_sha256
            and runtime_smoke.get("file_audit_sha256")
            == V89_SEALED_RUNTIME_SHA256[V89_RUNTIME_FILE_AUDIT]
            and runtime_smoke.get("candidate_adapter_sha256")
            == V89_SEALED_RUNTIME_SHA256[V89_RELEASE_ADAPTER]
            and runtime_smoke.get("candidate_memory_tensor_sha256")
            == V89_SEALED_RUNTIME_SHA256[V89_RELEASE_MEMORY]
            and runtime_smoke.get("v89_bridge_state_sha256")
            == trained_bridge.get("final_state_sha256")
            and runtime_smoke.get("development_known_smoke_trained") is True
            and runtime_smoke.get("held_out_smoke_claim") is False
            and runtime_smoke.get("held_out_generalization_claim") is False
            and runtime_smoke.get("expected_behavior_not_loaded_by_chat_runtime") is True
            and len(set(runtime_smoke.get("prefix_hashes", []))) == 1
            and len(set(runtime_smoke.get("environment_conditioned_input_hashes", []))) == 1
            and runtime_smoke.get("prefix_hashes")
            == runtime_smoke.get("environment_conditioned_input_hashes")
            and runtime_audit.get("passed") is True
            and runtime_audit.get("forbidden_accesses") == []
        )

        release_bindings = runtime_release.get("bindings")
        release_checkpoint = runtime_release.get("checkpoint")
        release_contract = runtime_release.get("strict_input_contract")
        release_memory = runtime_release.get("scene_memory")
        release_provenance = release_metadata.get("initialization_provenance", {}).get(
            "v89_strict_runtime_release"
        )
        runtime_release_ok = bool(
            runtime_smoke_ok
            and all(
                isinstance(value, Mapping)
                for value in (
                    release_bindings,
                    release_checkpoint,
                    release_contract,
                    release_memory,
                    release_provenance,
                )
            )
            and runtime_release.get("artifact") == "gemma4_v89_strict_runtime_release_v1"
            and runtime_release.get("schema_version") == 89
            and runtime_release.get("all_release_gates_passed") is True
            and runtime_release.get("promotion_decision") == "strict_scene1_experimental_primary"
            and runtime_release.get("promotion_scope")
            == "strict_direct_continuous_scene_memory_scene1_chat"
            and runtime_release.get("runtime_checkpoint_contains_environmental_text") is False
            and runtime_release.get("runtime_checkpoint_contains_supervision") is False
            and runtime_release.get("chat_runtime_loads_training_or_evaluation_reports") is False
            and runtime_release.get("development_known_smoke_trained") is True
            and runtime_release.get("held_out_smoke_claim") is False
            and runtime_release.get("held_out_generalization_claim") is False
            and release_bindings.get("experiment_config_sha256") == config_sha256
            and release_bindings.get("preregistration_sha256") == prereg_sha256
            and release_bindings.get("cpu_preflight_sha256") == preflight_sha256
            and release_bindings.get("training_report_sha256") == training_sha256
            and release_bindings.get("evaluation_predictions_sha256") == predictions_sha256
            and release_bindings.get("model_gate_report_sha256") == evaluation_sha256
            and release_bindings.get("runtime_smoke_sha256") == smoke_sha256
            and release_bindings.get("model_acceptance_gate_passed") is True
            and release_checkpoint.get("adapter_sha256")
            == V89_SEALED_RUNTIME_SHA256[V89_RELEASE_ADAPTER]
            and release_checkpoint.get("runtime_metadata_sha256")
            == V89_SEALED_RUNTIME_SHA256[V89_RELEASE_METADATA]
            and release_checkpoint.get("checkpoint_sha256")
            == "9408092e589834671c79394260b67198262e4d2a4f1fe01f3f772fed6b4c2b1b"
            and release_checkpoint.get("exact_two_file_checkpoint") is True
            and release_contract.get("shape") == [1, 738, 1536]
            and release_contract.get("continuous_environment_payload_tokens") == 736
            and release_contract.get("environmental_text_inputs") == []
            and release_contract.get("same_exact_memory_reused_for_every_question") is True
            and release_contract.get("question_derived_environmental_tokens") == 0
            and release_contract.get("question_conditioned_environmental_readout") is False
            and release_contract.get("question_dependent_retrieval") is False
            and release_memory.get("memory_tensor_file_bytes_unchanged") is True
            and release_memory.get("question_data_used_for_rebinding") is False
            and release_memory.get("packaged_memory_tensor_file_sha256")
            == V89_SEALED_RUNTIME_SHA256[V89_RELEASE_MEMORY]
            and release_metadata.get("lora_trainable_parameter_count") == 0
            and release_metadata.get("scene_latents") == 256
            and release_metadata.get("scene_model_dim") == 384
            and release_metadata.get("language_model_id") == "google/gemma-4-E2B-it"
            and release_metadata.get("question_dependent_scene_processing") is False
            and release_metadata.get("all_voxels_transformed") is True
            and len(release_metadata.get("lora_bank_state_sha256", {})) == 11
            and release_metadata.get("lora_bank_state_sha256", {}).get(
                "v89_scene1_retention_bridge"
            )
            == trained_bridge.get("final_state_sha256")
            and release_provenance.get("runtime_promotion_authorized") is True
            and release_provenance.get("smoke_report_sha256") == smoke_sha256
            and release_provenance.get("model_gate_report_sha256") == evaluation_sha256
            and release_memory_metadata.get("canonical_prefix_sha256")
            == prediction_memory.get("prefix_sha256_before")
            and release_memory_metadata.get("source_base_checkpoint_sha256")
            == release_checkpoint.get("checkpoint_sha256")
            and release_memory_metadata.get("question_inputs_used_for_compilation") is False
            and release_memory_metadata.get("questions_or_answers_serialized") is False
            and release_memory_metadata.get("oracle_loaded") is False
        )

        checks = {
            "sealed_configuration": config_ok,
            "preregistration": preregistration_ok,
            "cpu_preflight": preflight_ok,
            "fixed_final_training": training_ok,
            "fixed_final_candidate": candidate_ok,
            "evaluation_predictions": predictions_ok,
            "single_scene_evaluation": evaluation_ok,
            "accuracy_figure": figure_ok,
            "strict_runtime_smoke": runtime_smoke_ok,
            "strict_runtime_release": runtime_release_ok,
        }
        if not all(checks.values()):
            failed = sorted(name for name, passed in checks.items() if not passed)
            raise ValueError(f"V89 sealed result semantics differ: {failed}")

        measurement_paths = [
            path.as_posix()
            for path in (
                *V89_SEALED_RESULT_SHA256,
                *V89_SEALED_RUNTIME_SHA256,
            )
        ]
        status = "authenticated_runtime_ready_single_scene_122_of_138_promoted"
        return {
            "status": status,
            "evidence_authenticated": True,
            "source_bundle_authenticated": True,
            "preregistration_authenticated": True,
            "cpu_preflight_authenticated": True,
            "post_v88_training_set_development": True,
            "single_scene_training_set_development": True,
            "single_scene_overfit_only": True,
            "development_known_smoke_trained": True,
            "held_out_smoke": False,
            "terminal_result": True,
            "held_out_generalization_measured": False,
            "official_validation_measured": False,
            "training_measured": True,
            "evaluation_measured": True,
            "result_evidence_authenticated": True,
            "runtime_smoke_measured": True,
            "runtime_smoke_authenticated": True,
            "runtime_release_authenticated": True,
            "runtime_promotion_authorized": True,
            "default_runtime_changed": True,
            "runtime_readiness": readiness,
            "runtime_ready": True,
            "canonical_report_generation_authorized": True,
            "runtime_ready_requires": [
                "sealed all-pass V89 model evaluation",
                "separately authenticated strict runtime package",
                "separate oracle-unavailable runtime smoke with clean file audit",
            ],
            "sealed_design": {
                "scene_id": "scene_000001",
                "strict_memory_shape": [1, 738, 1536],
                "parent_v88_errors": 31,
                "parent_v88_correct_anchors": 107,
                "rows_per_epoch": 310,
                "epochs": 3,
                "micro_rows": 930,
                "optimizer_updates": 155,
                "causal_margin_rows": 18,
                "fresh_target": "model.language_model.layers.27.self_attn.o_proj",
                "fresh_rank": 8,
                "fresh_parameter_count": 28_672,
                "smoke_expected_answers": ["yes", "red", "left"],
                "smoke_is_trained": True,
                "smoke_is_held_out": False,
            },
            "training_result": {
                "status": training_result["status"],
                "device": training_result["device"],
                "canonical_unique_training_rows": 138,
                "unique_schedule_items_per_epoch": 310,
                "micro_rows_consumed": 930,
                "epochs": 3,
                "optimizer_updates": 155,
                "causal_margin_rows_consumed": 18,
                "parent_v88_errors_replayed_twice_per_epoch": 31,
                "parent_v88_correct_anchors_replayed_per_epoch": 107,
                "trainable_parameter_count": 28_672,
                "initial_bridge_state_sha256": trained_bridge["initial_state_sha256"],
                "final_bridge_state_sha256": trained_bridge["final_state_sha256"],
                "fixed_final_weights_sha256": weights_sha256,
                "elapsed_seconds": training_result["elapsed_seconds"],
                "loaded_file_count": 87,
                "protected_read_count": 0,
                "all_training_gates_passed": True,
            },
            "single_scene_evaluation": {
                "status": evaluation["status"],
                "scene_id": "scene_000001",
                "training_authorized_scene": True,
                "canonical_exact": dict(overall),
                "strict_normalized_exact": dict(strict),
                "canonical_accuracy_by_answer_type": dict(by_type),
                "answer_token_mean_nll": metrics["answer_token_mean_nll"],
                "overall_acceptance_threshold": 0.8,
                "correct_above_acceptance_threshold": 11,
                "model_acceptance_gates": dict(model_gates),
                "model_acceptance_gate_passed": True,
                "elapsed_seconds": evaluation["elapsed_seconds"],
                "held_out_generalization": False,
                "official_validation": False,
            },
            "generic_scene1_smoke": {
                "correct": 3,
                "total": 3,
                "accuracy": 1.0,
                "expected_answers": ["yes", "red", "left"],
                "observed_answers": ["yes", "red", "left"],
                "records": [
                    {
                        "question": row["question"],
                        "answer": row["observed"],
                        "exact_correct": row["passed"],
                    }
                    for row in behavior
                ],
                "development_known_and_trained": True,
                "held_out": False,
                "runtime_oracle_unavailable_audit_run": True,
                "caveat": (
                    "All three behavior questions were represented in V89's training "
                    "schedule. The strict runtime smoke proves packaging, isolation, "
                    "and invariant input reuse—not held-out generalization."
                ),
            },
            "zero_payload_causal_control": {
                "row_count": 3,
                "mean_correct_memory_nll": causal["mean_correct_memory_nll"],
                "mean_zero_payload_nll": causal["mean_zero_payload_nll"],
                "mean_zero_minus_correct_nll": causal["mean_zero_minus_correct_nll"],
                "canonical_prediction_changes": 2,
                "positive_nll_advantage_gate_passed": True,
                "prediction_change_gate_passed": True,
            },
            "strict_input_invariance": {
                "shape": [1, 738, 1536],
                "prefix_sha256": prediction_memory["prefix_sha256_before"],
                "prefix_hash_invariant": True,
                "same_prefix_reused_for_every_question": True,
                "exact_total_environment_input_invariant": True,
                "question_derived_environmental_tokens": 0,
                "question_conditioned_environmental_readout": False,
                "question_dependent_retrieval": False,
                "control_tokens": 0,
                "environmental_text_inputs": [],
            },
            "evaluation_isolation": {
                "loaded_file_count": 87,
                "protected_read_count": 0,
                "oracle_loaded": False,
                "runtime_oracle_physically_unavailable_test_run": True,
                "runtime_file_audit_forbidden_read_count": 0,
                "chat_loaded_training_or_evaluation_report": False,
            },
            "runtime_release": {
                "promotion_decision": runtime_release["promotion_decision"],
                "promotion_scope": runtime_release["promotion_scope"],
                "checkpoint_path": (
                    "data_gemma4/runtime/checkpoints/gemma4_v89_strict_scene1_release_v1"
                ),
                "checkpoint_sha256": release_checkpoint["checkpoint_sha256"],
                "adapter_sha256": release_checkpoint["adapter_sha256"],
                "runtime_metadata_sha256": release_checkpoint["runtime_metadata_sha256"],
                "exact_two_file_checkpoint": True,
                "exact_eleven_frozen_banks": True,
                "all_release_gates_passed": True,
                "runtime_promotion_authorized": True,
            },
            "accuracy_figure": {
                "status": "authenticated_posthoc_development_known_result_visualization",
                "path": V89_ACCURACY_FIGURE.as_posix(),
                "sha256": V89_SEALED_RESULT_SHA256[V89_ACCURACY_FIGURE],
                "summary_path": V89_ACCURACY_FIGURE_SUMMARY.as_posix(),
                "summary_sha256": V89_SEALED_RESULT_SHA256[V89_ACCURACY_FIGURE_SUMMARY],
                "source_report_sha256": evaluation_sha256,
                "new_inference": False,
                "held_out_generalization": False,
                "runtime_promotion_evidence": False,
            },
            "comparison_to_v88": {
                "v88_correct": 107,
                "v89_correct": 122,
                "correct_answer_gain": 15,
                "v88_accuracy": 107 / 138,
                "v89_accuracy": 122 / 138,
                "accuracy_point_gain": 100.0 * (15 / 138),
                "v88_spatial_relation_correct": 64,
                "v89_spatial_relation_correct": 74,
                "v88_attribute_correct": 11,
                "v89_attribute_correct": 15,
                "smoke_was_trained_in_both": True,
            },
            "checks": checks,
            "source_evidence_paths": required_paths,
            "source_evidence_sha256": observed_hashes,
            "pending_result_paths": [],
            "present_unsealed_result_paths": [],
            "measurement_evidence_paths": measurement_paths,
            "measurement_evidence_sha256": {**result_hashes, **runtime_hashes},
            "scope_warning": (
                "V89 is an authenticated post-V88 single-scene training-set "
                "development result. It scored 122/138 (88.41%) and passed every "
                "locked model gate plus the separate oracle-unavailable strict "
                "runtime gate. Its 3/3 smoke was explicitly trained and is not held "
                "out. Promotion is a local scene-one experimental runtime, not "
                "held-out generalization or official validation."
            ),
        }
    except (KeyError, TypeError, ValueError, FileNotFoundError, OSError) as error:
        return {
            "status": "source_bundle_authentication_failed",
            "evidence_authenticated": False,
            "source_bundle_authenticated": False,
            "post_v88_training_set_development": True,
            "single_scene_training_set_development": True,
            "development_known_smoke_trained": True,
            "held_out_smoke": False,
            "training_measured": False,
            "evaluation_measured": False,
            "result_evidence_authenticated": False,
            "runtime_smoke_measured": False,
            "runtime_promotion_authorized": False,
            "runtime_ready": False,
            "canonical_report_generation_authorized": False,
            "source_evidence_paths": required_paths,
            "measurement_evidence_paths": [],
            "source_evidence_error": f"{type(error).__name__}: {error}",
        }


__all__ = [
    "V89_ACCURACY_FIGURE",
    "V89_ACCURACY_FIGURE_SUMMARY",
    "V89_CONFIG",
    "V89_CPU_PREFLIGHT",
    "V89_EVALUATION",
    "V89_EVALUATION_PREDICTIONS",
    "V89_FIXED_FINAL_METADATA",
    "V89_FIXED_FINAL_WEIGHTS",
    "V89_PENDING_RESULT_PATHS",
    "V89_PREREGISTRATION",
    "V89_RUNTIME_CANDIDATE",
    "V89_RUNTIME_FILE_AUDIT",
    "V89_RUNTIME_RELEASE",
    "V89_RUNTIME_SMOKE",
    "V89_SEALED_RESULT_SHA256",
    "V89_SEALED_RUNTIME_SHA256",
    "V89_SEALED_SOURCE_SHA256",
    "V89_TRAINING",
    "classify_v89_runtime_readiness",
    "inspect_v89_reporting_state",
]
