from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from semantic_3d_chat.evaluation import v92_strict_runtime_release as release
from semantic_3d_chat.evaluation.strict_direct_release_core import BridgeSourceContract


def _passing_report() -> dict[str, object]:
    experiment = release._load_experiment()
    intent_ids = [str(row["id"]) for row in experiment["conversational_intents"]]
    primary = {
        identifier: {"correct": 1, "total": 1, "accuracy": 1.0}
        for identifier in intent_ids
    }
    held = {
        identifier: {"correct": 2, "total": 2, "accuracy": 1.0}
        for identifier in intent_ids
    }
    model_gates = {name: True for name in release._REQUIRED_MODEL_GATES}
    memory = {
        "compiled_before_question_tokenization": True,
        "shape": [1, 738, 1536],
        "continuous_environment_payload_tokens": 736,
        "prefix_sha256_before": release.SOURCE_MEMORY_PREFIX_SHA256,
        "prefix_sha256_after": release.SOURCE_MEMORY_PREFIX_SHA256,
        "prefix_hash_invariant": True,
        "environment_conditioned_input_invariant": True,
        "same_exact_memory_reused_for_all_177_questions": True,
        "question_derived_environmental_tokens": 0,
        "question_conditioned_environmental_readout": False,
        "question_dependent_scene_processing": False,
        "question_dependent_retrieval": False,
        "control_tokens": 0,
        "environmental_text_inputs": [],
    }
    metrics = {
        "canonical_type_specific": {"correct": 122, "total": 138, "accuracy": 122 / 138},
        "canonical_accuracy_by_answer_type": {
            "presence": {"correct": 21, "total": 23, "accuracy": 21 / 23},
            "count": {"correct": 9, "total": 10, "accuracy": 0.9},
            "metric": {"correct": 1, "total": 1, "accuracy": 1.0},
            "attribute": {"correct": 15, "total": 15, "accuracy": 1.0},
            "spatial_relation": {"correct": 73, "total": 82, "accuracy": 73 / 82},
            "support": {"correct": 1, "total": 7, "accuracy": 1 / 7},
        },
        "primary_conversational": {
            "correct": 13,
            "total": 13,
            "accuracy": 1.0,
            "core_actionable_correct": 6,
            "core_actionable_total": 6,
            "by_intent": primary,
            "records": [],
        },
        "new_held_wording": {
            "correct": 26,
            "total": 26,
            "accuracy": 1.0,
            "newly_held_wording_only": True,
            "held_out_scene": False,
            "by_intent": held,
            "records": [],
        },
        "causal_control": {
            "row_count": 13,
            "mean_zero_minus_correct_nll": 0.5,
            "required_mean_margin_nll": 0.5,
            "canonical_prediction_changes": 13,
            "records": [],
        },
        "model_acceptance_gates": model_gates,
        "model_acceptance_gate_passed": True,
        "separate_runtime_packaging_authorized": True,
        "runtime_oracle_unavailable_gate_pending": True,
        "runtime_file_audit_gate_pending": True,
        "automatic_runtime_promotion": False,
        "runtime_promotion_authorized": False,
    }
    return {
        "artifact": "gemma4_v92_scene1_retention_conversation_repair_evaluation_v1",
        "schema_version": 92,
        "status": "model_gates_pass_separate_runtime_packaging_required",
        "metrics": metrics,
        "scene_memory": memory,
        "leakage": {"protected_read_count": 0, "protected_reads": [], "oracle_loaded": False},
        "fixed_checkpoint_selected_before_scoring": True,
        "checkpoint_selection_after_scoring": False,
        "post_v91_training_set_development": True,
        "single_scene_retention_conversation_repair": True,
        "development_known_primary_questions": True,
        "newly_held_wording_only": True,
        "held_out_scene": False,
        "held_out_scene_generalization_claim": False,
        "frozen_thirteen_bank_parent_mutated": False,
        "fixed_final_candidate_state_invariant": True,
        "separate_runtime_packaging_authorized": True,
        "runtime_oracle_unavailable_gate_pending": True,
        "runtime_file_audit_gate_pending": True,
        "automatic_runtime_promotion": False,
        "runtime_promotion_authorized": False,
        "official_validation_loaded": False,
        "official_test_loaded": False,
        "deferred_final_loaded": False,
        "oracle_loaded": False,
    }


def _contracts() -> tuple[BridgeSourceContract, ...]:
    return (
        BridgeSourceContract(
            root=release.PROJECT_ROOT,
            artifact="v90",
            bank_name=release.V90_BANK,
            target_module=release.V90_TARGET,
            rank=8,
            alpha=16.0,
            dropout=0.0,
            parameter_count=28_672,
            state_sha256=release.V90_STATE_SHA256,
            weights_sha256="1" * 64,
            metadata_sha256="2" * 64,
        ),
        BridgeSourceContract(
            root=release.PROJECT_ROOT,
            artifact="v91",
            bank_name=release.V91_BANK,
            target_module=release.V91_TARGET,
            rank=16,
            alpha=32.0,
            dropout=0.0,
            parameter_count=221_184,
            state_sha256=release.V91_STATE_SHA256,
            weights_sha256="3" * 64,
            metadata_sha256="4" * 64,
        ),
        BridgeSourceContract(
            root=release.PROJECT_ROOT,
            artifact="v92",
            bank_name=release.V92_BANK,
            target_module=release.V92_TARGET,
            rank=8,
            alpha=16.0,
            dropout=0.0,
            parameter_count=45_056,
            state_sha256="a" * 64,
            weights_sha256="5" * 64,
            metadata_sha256="6" * 64,
        ),
    )


def test_v92_fixed_pre_model_evidence_hashes_are_exact() -> None:
    expected = {
        release.EXPERIMENT_CONFIG: release.EXPERIMENT_CONFIG_SHA256,
        release.PREREGISTRATION: release.PREREGISTRATION_SHA256,
        release.CPU_PREFLIGHT: release.CPU_PREFLIGHT_SHA256,
    }
    assert {path: release.sha256_file(path) for path in expected} == expected
    release._authenticate_preflight(release._load_experiment())


def test_v92_runtime_payload_is_v89_plus_exact_three_bridge_banks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(release, "_contracts", lambda _evidence: _contracts())
    payload = release.build_runtime_config_payload({})
    banks = payload["language"]["lora_banks"]
    assert tuple(banks) == release.EXPECTED_BANKS
    assert len(banks) == 14
    assert all(bank["trainable"] is False for bank in banks.values())
    assert banks[release.V90_BANK]["expected_initial_state_sha256"] == release.V90_STATE_SHA256
    assert banks[release.V91_BANK]["expected_initial_state_sha256"] == release.V91_STATE_SHA256
    assert banks[release.V92_BANK]["expected_initial_state_sha256"] == "a" * 64


def test_v92_model_gate_requires_every_gate_and_candidate_invariance() -> None:
    passing = _passing_report()
    release.validate_model_gate_contract_v92(passing)

    failed = copy.deepcopy(passing)
    failed["metrics"]["model_acceptance_gates"][
        "fixed_final_candidate_state_invariance"
    ] = False
    with pytest.raises(ValueError, match="did not pass exactly"):
        release.validate_model_gate_contract_v92(failed)

    leaked = copy.deepcopy(passing)
    leaked["leakage"]["protected_read_count"] = 1
    with pytest.raises(ValueError, match="leakage or immutable-memory"):
        release.validate_model_gate_contract_v92(leaked)


def test_v92_child_protocol_has_questions_and_no_expectation_channel() -> None:
    cases = release._primary_cases()
    questions = [str(case["question"]) for case in cases]
    answers = {str(case["expected"]) for case in cases}
    command = release._smoke_command(questions)
    assert command.count("--question") == 13
    assert answers.isdisjoint(command)
    assert all(flag not in command for flag in ("--expected", "--answer", "--reference"))


def test_v92_postprocess_audit_protects_all_offline_bridges() -> None:
    safe = {
        "loaded_files": [
            str(release.RUNTIME_CONFIG),
            str(release.CANDIDATE_CHECKPOINT / "adapter.safetensors"),
            str(release.CANDIDATE_MEMORY / "memory.safetensors"),
        ]
    }
    unsafe = {
        "loaded_files": [
            *safe["loaded_files"],
            str(release.V90_BRIDGE_CANDIDATE / "bridge.safetensors"),
            str(release.V91_BRIDGE_CANDIDATE / "bridge.safetensors"),
            str(release.V92_BRIDGE_CANDIDATE / "bridge.safetensors"),
            str(release.PROJECT_ROOT / "data/oracle/scene_000001.json"),
        ]
    }
    assert release._protected_smoke_reads(safe) == []
    assert len(release._protected_smoke_reads(unsafe)) == 4


def test_v92_promotion_fails_closed_without_passing_smoke(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    smoke = tmp_path / "smoke.json"
    smoke.write_text(
        json.dumps({"passed": False, "promotion_authorized": False}),
        encoding="utf-8",
    )
    monkeypatch.setattr(release, "SMOKE_REPORT", smoke)
    monkeypatch.setattr(release, "RELEASE_CHECKPOINT", tmp_path / "release")
    monkeypatch.setattr(release, "RELEASE_MEMORY", tmp_path / "memory")
    monkeypatch.setattr(release, "RELEASE_REPORT", tmp_path / "release.json")
    monkeypatch.setattr(release, "authenticate_v92_model_gate", dict)
    with pytest.raises(ValueError, match="runtime smoke"):
        release.promote_release()


def test_v92_authentication_refuses_before_fixed_evaluation_exists() -> None:
    if release.MODEL_GATE_REPORT.is_file():
        pytest.skip("V92 evaluation now exists; model-gate tests cover fail-closed report logic")
    with pytest.raises(FileNotFoundError):
        release.authenticate_v92_model_gate()
