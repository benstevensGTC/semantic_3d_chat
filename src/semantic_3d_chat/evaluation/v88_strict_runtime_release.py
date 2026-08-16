"""Fail-closed post-gate surface for V88's strict scene-one runtime.

V88 did not pass its sealed model gate, so this module currently authenticates
and refuses it.  The model-free composition helpers are complete for audit and
future reuse, but no function here writes a runtime YAML, candidate, smoke, or
release artifact.  The chat runtime never imports this module.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import yaml

from semantic_3d_chat.chat.runtime_config import load_runtime_config
from semantic_3d_chat.chat.v88_strict_scene1_runtime import (
    V86_BANK,
    V86_STATE_SHA256,
    V86_TARGET,
    V87_BANK,
    V87_STATE_SHA256,
    V87_TARGET,
    V88_BANK,
    V88_TARGET,
)
from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.evaluation.strict_direct_release_core import (
    BridgeSourceContract,
    compose_exact_bank_archive,
    extend_runtime_lora_config,
    load_bridge_source,
    sha256_file,
)

SCHEMA_VERSION: Final[int] = 88
SCENE_ID: Final[str] = "scene_000001"
EXPERIMENT_CONFIG_SHA256: Final[str] = (
    "c9812271b5834f605fc7158b0fdcf5c6eab269efbc1e8fa0b5aec62c7e9b20fb"
)
PREREGISTRATION_SHA256: Final[str] = (
    "765db3b7420d6bf8b1c4a9122ca211497ab9985ca8a85ae7bd524ca523c0657d"
)
CPU_PREFLIGHT_SHA256: Final[str] = (
    "0d352bc80c6537351ccd1d5cea83d029eff12af14757a8ff1e7ac5ceea1d4fd5"
)
TRAINING_REPORT_SHA256: Final[str] = (
    "d356e641d84de01aa89484f5ec4b7034b3dadc3719a9a2488696537ad5b05d43"
)
EVALUATION_REPORT_SHA256: Final[str] = (
    "40b4c591d84a3b1ae99c301d017e9c6212308203a319e3c511f22583e5a78641"
)
PREDICTIONS_SHA256: Final[str] = (
    "a7c815fde05383aa2f4aa07ec0127ab7950470d91b1f34f5b171e5f688850f56"
)
SOURCE_MEMORY_PREFIX_SHA256: Final[str] = (
    "a428f5147c815839ae7315a0adab952ab210814fb21dcdc5bf13b167f28a6e37"
)

EXPERIMENT_CONFIG: Final[Path] = (
    PROJECT_ROOT / "configs/experiments/gemma4_v88_scene1_augmented_demo.yaml"
)
BASE_RUNTIME_CONFIG: Final[Path] = (
    PROJECT_ROOT / "configs/runtime/gemma4_v85_strict_multiscene.yaml"
)
V85_CHECKPOINT: Final[Path] = (
    PROJECT_ROOT / "reports/gemma4/artifacts/v85_strict_runtime_candidate"
)
PREREGISTRATION: Final[Path] = (
    PROJECT_ROOT / "reports/gemma4/metrics/gemma4_v88_scene1_augmented_preregistration.json"
)
CPU_PREFLIGHT: Final[Path] = (
    PROJECT_ROOT / "reports/gemma4/metrics/gemma4_v88_scene1_augmented_cpu_preflight.json"
)
TRAINING_REPORT: Final[Path] = (
    PROJECT_ROOT / "reports/gemma4/metrics/gemma4_v88_scene1_augmented_training.json"
)
EVALUATION_REPORT: Final[Path] = (
    PROJECT_ROOT / "reports/gemma4/metrics/gemma4_v88_scene1_augmented_evaluation.json"
)
PREDICTIONS: Final[Path] = (
    PROJECT_ROOT / "reports/gemma4/predictions/gemma4_v88_scene1_augmented_evaluation.json"
)
RUNTIME_CONFIG: Final[Path] = (
    PROJECT_ROOT / "configs/runtime/gemma4_v88_strict_scene1.yaml"
)
CANDIDATE_CHECKPOINT: Final[Path] = (
    PROJECT_ROOT / "reports/gemma4/artifacts/v88_scene1_augmented_runtime"
)
RELEASE_CHECKPOINT: Final[Path] = (
    PROJECT_ROOT / "data_gemma4/runtime/checkpoints/gemma4_v88_strict_scene1_release_v1"
)

V86_CONTRACT: Final[BridgeSourceContract] = BridgeSourceContract(
    root=PROJECT_ROOT / "reports/gemma4/artifacts/v86_scene1_demo_final",
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
    root=PROJECT_ROOT / "reports/gemma4/artifacts/v87_scene1_balanced_final",
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
    root=PROJECT_ROOT / "reports/gemma4/artifacts/v88_scene1_augmented_final",
    artifact="gemma4_v88_scene1_augmented_fixed_final_v1",
    bank_name=V88_BANK,
    target_module=V88_TARGET,
    rank=16,
    alpha=32.0,
    dropout=0.0,
    parameter_count=57_344,
    state_sha256="ff311624150056c67ad1c0a06752a77af2de89878778049ae886aa59db3376aa",
    weights_sha256="95d4aaf9c42cbf796dc047b3e622cf92247c898987ab159570944861ef698cf1",
    metadata_sha256="670ef58de0ac3b9d1e9a141292ffea900848cc512b77fc12c69d9def706d3d41",
)
_BASE_BANKS: Final[tuple[str, ...]] = (
    "inherited_v12",
    "extension_v13",
    "extension_v23_shared_kv",
    "extension_v24_shared_query",
    "extension_v28_stage_b_query",
    "extension_v30_joint_pair_query",
    "v85_strict_multiscene_bridge",
)
FINAL_BANKS: Final[tuple[str, ...]] = _BASE_BANKS + (
    V86_BANK,
    V87_BANK,
    V88_BANK,
)
_REQUIRED_GATES: Final[frozenset[str]] = frozenset(
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


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected one JSON object: {path}")
    return value


def _load_experiment() -> dict[str, Any]:
    if sha256_file(EXPERIMENT_CONFIG) != EXPERIMENT_CONFIG_SHA256:
        raise ValueError("V88 sealed experiment config changed")
    payload = yaml.safe_load(EXPERIMENT_CONFIG.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != {"v88"}:
        raise ValueError("V88 experiment config identity changed")
    experiment = payload["v88"]
    dataset = experiment.get("dataset") if isinstance(experiment, dict) else None
    if (
        not isinstance(experiment, dict)
        or experiment.get("schema_version") != 88
        or experiment.get("artifact")
        != "gemma4_v88_scene1_augmented_direct_memory_overfit_v1"
        or experiment.get("status") != "preregistered_before_full_model_load"
        or not isinstance(dataset, dict)
        or dataset.get("scene_id") != SCENE_ID
        or dataset.get("canonical_row_count") != 138
        or dataset.get("runtime_serializes_questions_or_answers") is not False
        or dataset.get("runtime_serializes_augmentation_inventory") is not False
        or dataset.get("runtime_serializes_error_inventory") is not False
    ):
        raise ValueError("V88 sealed single-scene experiment identity changed")
    return experiment


def validate_model_gate_contract_v88(report: Mapping[str, Any]) -> None:
    """Accept only an exact all-pass, explicitly development-known V88 result."""

    metrics = report.get("metrics")
    gates = metrics.get("model_acceptance_gates") if isinstance(metrics, dict) else None
    canonical = metrics.get("canonical_type_specific") if isinstance(metrics, dict) else None
    by_type = (
        metrics.get("canonical_accuracy_by_answer_type")
        if isinstance(metrics, dict)
        else None
    )
    causal = metrics.get("causal_control") if isinstance(metrics, dict) else None
    smoke = metrics.get("generic_smoke") if isinstance(metrics, dict) else None
    if (
        report.get("artifact") != "gemma4_v88_scene1_augmented_evaluation_v1"
        or report.get("schema_version") != 88
        or report.get("status")
        != "model_gates_pass_separate_runtime_packaging_required"
        or not isinstance(metrics, dict)
        or metrics.get("model_acceptance_gate_passed") is not True
        or not isinstance(gates, dict)
        or set(gates) != _REQUIRED_GATES
        or any(value is not True for value in gates.values())
        or metrics.get("separate_runtime_packaging_authorized") is not True
        or metrics.get("runtime_promotion_authorized") is not False
        or report.get("separate_runtime_packaging_authorized") is not True
        or report.get("automatic_runtime_promotion") is not False
        or report.get("runtime_promotion_authorized") is not False
        or report.get("fixed_checkpoint_selected_before_scoring") is not True
        or report.get("checkpoint_selection_after_scoring") is not False
        or report.get("development_known_smoke_trained") is not True
        or report.get("held_out_smoke_claim") is not False
        or report.get("held_out_generalization_claim") is not False
        or report.get("parent_v85_v86_v87_mutated") is not False
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
        or smoke.get("development_known_and_trained") is not True
        or smoke.get("held_out") is not False
    ):
        raise ValueError("V88 model-level acceptance report did not pass exactly")
    records = smoke.get("records")
    if not isinstance(records, list) or len(records) != 3 or any(
        not isinstance(record, dict)
        or record.get("exact_correct") is not True
        or record.get("development_known_and_trained") is not True
        or record.get("held_out") is not False
        for record in records
    ):
        raise ValueError("V88 development-known smoke evidence changed")
    leakage = report.get("leakage")
    memory = report.get("scene_memory")
    if (
        not isinstance(leakage, dict)
        or leakage.get("protected_read_count") != 0
        or leakage.get("protected_reads") != []
        or leakage.get("oracle_loaded") is not False
        or not isinstance(memory, dict)
        or memory.get("prefix_hash_invariant") is not True
        or memory.get("same_prefix_reused_for_every_question") is not True
        or memory.get("question_derived_environmental_tokens") != 0
        or memory.get("prefix_sha256_before") != SOURCE_MEMORY_PREFIX_SHA256
        or memory.get("prefix_sha256_after") != SOURCE_MEMORY_PREFIX_SHA256
    ):
        raise ValueError("V88 model-level leakage or memory gate changed")


def authenticate_v88_model_gate() -> dict[str, Any]:
    """Authenticate all sealed bytes, then fail unless every model gate passed."""

    experiment = _load_experiment()
    fixed_hashes = {
        PREREGISTRATION: PREREGISTRATION_SHA256,
        CPU_PREFLIGHT: CPU_PREFLIGHT_SHA256,
        TRAINING_REPORT: TRAINING_REPORT_SHA256,
        EVALUATION_REPORT: EVALUATION_REPORT_SHA256,
        PREDICTIONS: PREDICTIONS_SHA256,
    }
    mismatches = {
        str(path.relative_to(PROJECT_ROOT)): {
            "expected": expected,
            "observed": sha256_file(path) if path.is_file() else None,
        }
        for path, expected in fixed_hashes.items()
        if not path.is_file() or sha256_file(path) != expected
    }
    if mismatches:
        raise ValueError(f"V88 sealed evidence changed: {mismatches}")
    report = _read_json(EVALUATION_REPORT)
    if report.get("preregistered_gates") != experiment.get("gates"):
        raise ValueError("V88 model gate is not bound to preregistered gates")
    validate_model_gate_contract_v88(report)
    for contract in (V86_CONTRACT, V87_CONTRACT, V88_CONTRACT):
        load_bridge_source(contract)
    predictions = _read_json(PREDICTIONS)
    candidate = predictions.get("candidate")
    records = predictions.get("records")
    if (
        predictions.get("artifact") != "gemma4_v88_scene1_augmented_predictions_v1"
        or predictions.get("schema_version") != 88
        or predictions.get("row_count") != 138
        or predictions.get("scene_count") != 1
        or predictions.get("development_known_smoke_trained") is not True
        or predictions.get("held_out_smoke_claim") is not False
        or predictions.get("runtime_promotion_authorized") is not False
        or not isinstance(candidate, dict)
        or candidate.get("weights_sha256") != V88_CONTRACT.weights_sha256
        or candidate.get("state_sha256") != V88_CONTRACT.state_sha256
        or candidate.get("optimizer_updates") != 188
        or not isinstance(records, list)
        or len(records) != 138
        or any(
            not isinstance(record, dict) or record.get("scene_id") != SCENE_ID
            for record in records
        )
    ):
        raise ValueError("V88 fixed predictions or scene-one binding changed")
    return {
        "schema_version": 88,
        "model_acceptance_gate_passed": True,
        "model_gate_report_authenticated": True,
        "experiment_config_sha256": EXPERIMENT_CONFIG_SHA256,
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "cpu_preflight_sha256": CPU_PREFLIGHT_SHA256,
        "training_report_sha256": TRAINING_REPORT_SHA256,
        "model_gate_report_sha256": EVALUATION_REPORT_SHA256,
        "evaluation_predictions_sha256": PREDICTIONS_SHA256,
        "v86_bridge_state_sha256": V86_STATE_SHA256,
        "v87_bridge_state_sha256": V87_STATE_SHA256,
        "v88_bridge_state_sha256": V88_CONTRACT.state_sha256,
        "development_known_smoke_trained": True,
        "held_out_smoke_claim": False,
        "held_out_generalization_claim": False,
    }


def build_post_gate_runtime_payload() -> dict[str, Any]:
    """Build in memory only; authentication executes before any composition."""

    authenticate_v88_model_gate()
    parent = load_runtime_config(BASE_RUNTIME_CONFIG)
    payload = extend_runtime_lora_config(
        parent_runtime_config=parent,
        added_bridges=(V86_CONTRACT, V87_CONTRACT, V88_CONTRACT),
        expected_final_banks=FINAL_BANKS,
    )
    if tuple(payload["language"]["lora_banks"]) != FINAL_BANKS:
        raise RuntimeError("V88 post-gate runtime payload is not exact ten-bank order")
    return payload


def verify_post_gate_composition() -> dict[str, Any]:
    """Compose in CPU memory only after a passing gate; never writes artifacts."""

    authenticate_v88_model_gate()
    archive, evidence = compose_exact_bank_archive(
        base_checkpoint=V85_CHECKPOINT,
        expected_base_banks=_BASE_BANKS,
        added_bridges=(V86_CONTRACT, V87_CONTRACT, V88_CONTRACT),
        expected_final_banks=FINAL_BANKS,
    )
    return {
        "schema_version": 88,
        "phase": "v88_post_gate_composition_verified",
        "bank_count": len(FINAL_BANKS),
        "tensor_count": len(archive),
        "evidence": evidence,
        "artifact_written": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("authenticate", "verify-composition"))
    args = parser.parse_args(argv)
    try:
        result = (
            authenticate_v88_model_gate()
            if args.command == "authenticate"
            else verify_post_gate_composition()
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"V88 strict runtime {args.command} refused: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CANDIDATE_CHECKPOINT",
    "FINAL_BANKS",
    "RELEASE_CHECKPOINT",
    "RUNTIME_CONFIG",
    "V86_CONTRACT",
    "V87_CONTRACT",
    "V88_CONTRACT",
    "authenticate_v88_model_gate",
    "build_post_gate_runtime_payload",
    "main",
    "validate_model_gate_contract_v88",
    "verify_post_gate_composition",
]
