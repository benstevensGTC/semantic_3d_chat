from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from safetensors.torch import load_file

from semantic_3d_chat.chat.v90_strict_scene1_runtime import (
    PROMOTION_DECISION as RUNTIME_PROMOTION_DECISION,
)
from semantic_3d_chat.evaluation import v90_strict_runtime_release as release


def _evidence() -> dict[str, object]:
    return {
        "v90_bridge_state_sha256": "a" * 64,
        "v90_bridge_file_sha256": "b" * 64,
        "v90_bridge_metadata_sha256": "c" * 64,
    }


def _passing_report() -> dict[str, object]:
    experiment = release._load_experiment()
    intent_ids = [str(row["id"]) for row in experiment["conversational_intents"]]
    primary_by_intent = {
        identifier: {"correct": 1, "total": 1, "accuracy": 1.0} for identifier in intent_ids
    }
    held_by_intent = {
        identifier: {"correct": 2, "total": 2, "accuracy": 1.0} for identifier in intent_ids
    }
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
        "canonical_type_specific": {
            "correct": 119,
            "total": 138,
            "accuracy": 119 / 138,
        },
        "canonical_accuracy_by_answer_type": {
            "presence": {"correct": 21, "total": 23, "accuracy": 21 / 23},
            "count": {"correct": 9, "total": 10, "accuracy": 0.9},
            "metric": {"correct": 1, "total": 1, "accuracy": 1.0},
            "attribute": {"correct": 14, "total": 15, "accuracy": 14 / 15},
            "spatial_relation": {
                "correct": 69,
                "total": 82,
                "accuracy": 69 / 82,
            },
            "support": {"correct": 1, "total": 7, "accuracy": 1 / 7},
        },
        "primary_conversational": {
            "correct": 13,
            "total": 13,
            "accuracy": 1.0,
            "core_actionable_correct": 6,
            "core_actionable_total": 6,
            "parent_smoke_correct": 3,
            "parent_smoke_total": 3,
            "by_intent": primary_by_intent,
            "records": [],
        },
        "held_wording": {
            "correct": 26,
            "total": 26,
            "accuracy": 1.0,
            "held_out_wording_only": True,
            "held_out_scene": False,
            "by_intent": held_by_intent,
            "records": [],
        },
        "causal_control": {
            "row_count": 13,
            "mean_zero_minus_correct_nll": 0.5,
            "required_mean_margin_nll": 0.5,
            "canonical_prediction_changes": 6,
            "records": [],
        },
        "model_acceptance_gates": {name: True for name in release._REQUIRED_MODEL_GATES},
        "model_acceptance_gate_passed": True,
        "separate_runtime_packaging_authorized": True,
        "runtime_oracle_unavailable_gate_pending": True,
        "runtime_file_audit_gate_pending": True,
        "automatic_runtime_promotion": False,
        "runtime_promotion_authorized": False,
    }
    return {
        "artifact": "gemma4_v90_scene1_conversational_evaluation_v1",
        "schema_version": 90,
        "status": "model_gates_pass_separate_runtime_packaging_required",
        "metrics": metrics,
        "scene_memory": memory,
        "leakage": {
            "protected_read_count": 0,
            "protected_reads": [],
            "oracle_loaded": False,
        },
        "fixed_checkpoint_selected_before_scoring": True,
        "checkpoint_selection_after_scoring": False,
        "single_scene_conversational_overfit": True,
        "development_known_primary_questions": True,
        "held_out_wording_only": True,
        "held_out_scene": False,
        "held_out_generalization_claim": False,
        "parent_v89_runtime_checkpoint_mutated": False,
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


def test_v90_fixed_pre_model_evidence_hashes_are_exact() -> None:
    expected = {
        release.EXPERIMENT_CONFIG: release.EXPERIMENT_CONFIG_SHA256,
        release.PREREGISTRATION: release.PREREGISTRATION_SHA256,
        release.CPU_PREFLIGHT: release.CPU_PREFLIGHT_SHA256,
    }

    assert all(len(value) == 64 for value in expected.values())
    assert all(set(value) <= set("0123456789abcdef") for value in expected.values())
    assert {path: release.sha256_file(path) for path in expected} == expected


def test_v90_runtime_payload_is_exact_frozen_parent_plus_fresh_bank() -> None:
    payload = release.build_runtime_config_payload(_evidence())
    banks = payload["language"]["lora_banks"]

    assert tuple(banks) == release.EXPECTED_BANKS
    assert len(banks) == 12
    assert all(bank["trainable"] is False for bank in banks.values())
    assert banks[release.V90_BANK] == {
        "trainable": False,
        "rank": 8,
        "alpha": 16.0,
        "dropout": 0.0,
        "initialization_algorithm": "checkpoint_overwrite",
        "initialization_seed": None,
        "expected_initial_state_sha256": "a" * 64,
        "target_modules": [release.V90_TARGET],
    }
    assert release.PROMOTION_DECISION == RUNTIME_PROMOTION_DECISION


def test_v90_model_gate_requires_exact_thresholds_core_and_leakage() -> None:
    passing = _passing_report()
    release.validate_model_gate_contract_v90(passing)

    failed_gate = copy.deepcopy(passing)
    failed_gate["metrics"]["model_acceptance_gates"][
        "primary_conversational_correct_at_least_required"
    ] = False
    with pytest.raises(ValueError, match="did not pass exactly"):
        release.validate_model_gate_contract_v90(failed_gate)

    only_eleven = copy.deepcopy(passing)
    only_eleven["metrics"]["primary_conversational"]["correct"] = 11
    with pytest.raises(ValueError, match="did not pass exactly"):
        release.validate_model_gate_contract_v90(only_eleven)

    missing_core = copy.deepcopy(passing)
    missing_core["metrics"]["primary_conversational"]["by_intent"]["table_contents"]["correct"] = 0
    with pytest.raises(ValueError, match="did not pass exactly"):
        release.validate_model_gate_contract_v90(missing_core)

    leaked = copy.deepcopy(passing)
    leaked["leakage"]["protected_read_count"] = 1
    with pytest.raises(ValueError, match="leakage or immutable-memory"):
        release.validate_model_gate_contract_v90(leaked)


def test_v90_child_command_contains_questions_but_no_expectation_channel() -> None:
    cases = release._primary_cases()
    questions = [str(case["question"]) for case in cases]
    answers = {str(case["expected"]) for case in cases}
    command = release._smoke_command(questions)

    assert command.count("--question") == 13
    assert [
        command[index + 1] for index, value in enumerate(command) if value == "--question"
    ] == questions
    assert answers.isdisjoint(command)
    assert "--expected" not in command
    assert "--answer" not in command
    assert "--reference" not in command


def test_v90_postprocess_audit_rejects_oracle_and_model_evidence_reads() -> None:
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
            str(release.TRAINING_REPORT),
            str(release.PROJECT_ROOT / "data/oracle/scene_000001.json"),
        ]
    }

    assert release._protected_smoke_reads(safe) == []
    violations = release._protected_smoke_reads(unsafe)
    assert str(release.TRAINING_REPORT.resolve()) in violations
    assert any("/data/oracle/" in path for path in violations)


def test_v90_final_bridge_hashes_are_runtime_evidence_not_source_constants() -> None:
    contract = release._contract_from_evidence(_evidence())

    assert contract.state_sha256 == "a" * 64
    assert contract.weights_sha256 == "b" * 64
    assert contract.metadata_sha256 == "c" * 64
    assert not hasattr(release, "V90_STATE_SHA256")


@pytest.mark.skipif(
    not release.V90_BRIDGE_CANDIDATE.is_dir(),
    reason="fixed-final V90 conversational bridge has not finished training",
)
def test_v90_promoted_metadata_passes_the_actual_chat_runtime_contract() -> None:
    bridge_metadata = json.loads(
        (release.V90_BRIDGE_CANDIDATE / release.RUNTIME_METADATA_FILENAME).read_text(
            encoding="utf-8"
        )
    )
    parent_fingerprint, _files = release.checkpoint_fingerprint(release.PARENT_CHECKPOINT)
    evidence = {
        "experiment_config_sha256": release.EXPERIMENT_CONFIG_SHA256,
        "preregistration_sha256": release.PREREGISTRATION_SHA256,
        "cpu_preflight_sha256": release.CPU_PREFLIGHT_SHA256,
        "training_report_sha256": "1" * 64,
        "evaluation_predictions_sha256": "2" * 64,
        "model_gate_report_sha256": "3" * 64,
        "parent_release_report_sha256": "4" * 64,
        "parent_checkpoint_sha256": parent_fingerprint,
        "parent_adapter_sha256": "5" * 64,
        "parent_metadata_sha256": "6" * 64,
        "v90_bridge_file_sha256": bridge_metadata["weights_sha256"],
        "v90_bridge_metadata_sha256": release.sha256_file(
            release.V90_BRIDGE_CANDIDATE / release.RUNTIME_METADATA_FILENAME
        ),
        "v90_bridge_state_sha256": bridge_metadata["state_sha256"],
    }

    metadata = release.build_runtime_metadata(
        evidence,
        promotion=RUNTIME_PROMOTION_DECISION,
        smoke_report_sha256="7" * 64,
    )
    contract = release.validate_v90_runtime_contract(
        scene_id=release.SCENE_ID,
        runtime_config=release.build_runtime_config_payload(evidence),
        checkpoint_metadata=metadata,
    )

    assert contract["runtime_promotion_authorized"] is True
    assert contract["v90_bridge_state_sha256"] == bridge_metadata["state_sha256"]
    assert metadata["lora"]["adapter_parameter_count"] == 901_120
    assert metadata["lora"]["trainable_adapter_parameter_count"] == 0


def test_v90_promotion_fails_closed_without_exact_passing_smoke(
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
    monkeypatch.setattr(
        release,
        "authenticate_v90_model_gate",
        lambda: {
            "model_gate_report_sha256": "7" * 64,
            "training_report_sha256": "8" * 64,
            "evaluation_predictions_sha256": "9" * 64,
            "v90_bridge_state_sha256": "a" * 64,
        },
    )

    with pytest.raises(ValueError, match="runtime smoke"):
        release.promote_release()


@pytest.mark.skipif(
    not release.CANDIDATE_CHECKPOINT.is_dir(),
    reason="passing-gate V90 runtime candidate has not been packaged",
)
def test_v90_candidate_is_exact_v89_parent_plus_two_tensors() -> None:
    result = release.verify_candidate()
    candidate = load_file(str(release.CANDIDATE_CHECKPOINT / "adapter.safetensors"), device="cpu")
    parent = load_file(str(release.PARENT_CHECKPOINT / "adapter.safetensors"), device="cpu")
    metadata = json.loads(
        (release.CANDIDATE_CHECKPOINT / release.RUNTIME_METADATA_FILENAME).read_text(
            encoding="utf-8"
        )
    )

    assert result["passed"] is True
    assert set(parent).issubset(candidate)
    assert all(candidate[name].equal(value) for name, value in parent.items())
    assert len(set(candidate) - set(parent)) == 2
    assert tuple(row["name"] for row in metadata["lora"]["banks"]) == (release.EXPECTED_BANKS)
    assert metadata["lora_parameter_count"] == 901_120
    assert metadata["lora_trainable_parameter_count"] == 0


def test_v90_release_cannot_exist_without_passing_external_smoke() -> None:
    if release.RELEASE_CHECKPOINT.exists():
        smoke = json.loads(release.SMOKE_REPORT.read_text(encoding="utf-8"))
        assert smoke["passed"] is True
        assert smoke["promotion_authorized"] is True
        assert release.RUNTIME_CONFIG.is_file()
