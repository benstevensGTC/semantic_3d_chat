from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from semantic_3d_chat.evaluation import v91_strict_runtime_release as release


def _evidence() -> dict[str, object]:
    return {
        "v91_bridge_state_sha256": "a" * 64,
        "v91_bridge_file_sha256": "b" * 64,
        "v91_bridge_metadata_sha256": "c" * 64,
    }


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
        "artifact": "gemma4_v91_scene1_conversational_repair_evaluation_v1",
        "schema_version": 91,
        "status": "model_gates_pass_separate_runtime_packaging_required",
        "metrics": metrics,
        "scene_memory": memory,
        "leakage": {"protected_read_count": 0, "protected_reads": [], "oracle_loaded": False},
        "fixed_checkpoint_selected_before_scoring": True,
        "checkpoint_selection_after_scoring": False,
        "post_v90_training_set_development": True,
        "single_scene_conversational_repair": True,
        "development_known_primary_questions": True,
        "newly_held_wording_only": True,
        "held_out_scene": False,
        "held_out_scene_generalization_claim": False,
        "parent_v89_runtime_checkpoint_mutated": False,
        "parent_v90_failed_candidate_mutated": False,
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


def test_v91_fixed_pre_model_evidence_hashes_are_exact() -> None:
    expected = {
        release.EXPERIMENT_CONFIG: release.EXPERIMENT_CONFIG_SHA256,
        release.PREREGISTRATION: release.PREREGISTRATION_SHA256,
        release.CPU_PREFLIGHT: release.CPU_PREFLIGHT_SHA256,
    }
    assert {path: release.sha256_file(path) for path in expected} == expected
    release._authenticate_preflight(release._load_experiment())


def test_v91_runtime_payload_is_v89_plus_exact_v90_and_v91_banks() -> None:
    payload = release.build_runtime_config_payload(_evidence())
    banks = payload["language"]["lora_banks"]
    assert tuple(banks) == release.EXPECTED_BANKS
    assert len(banks) == 13
    assert all(bank["trainable"] is False for bank in banks.values())
    assert banks[release.V90_BANK]["expected_initial_state_sha256"] == release.V90_STATE_SHA256
    assert banks[release.V91_BANK] == {
        "trainable": False,
        "rank": 16,
        "alpha": 32.0,
        "dropout": 0.0,
        "initialization_algorithm": "checkpoint_overwrite",
        "initialization_seed": None,
        "expected_initial_state_sha256": "a" * 64,
        "target_modules": [release.V91_TARGET],
    }


def test_v91_model_gate_requires_all_repair_core_and_leakage_gates() -> None:
    passing = _passing_report()
    release.validate_model_gate_contract_v91(passing)

    failed = copy.deepcopy(passing)
    failed["metrics"]["model_acceptance_gates"][
        "primary_conversational_correct_at_least_required"
    ] = False
    with pytest.raises(ValueError, match="did not pass exactly"):
        release.validate_model_gate_contract_v91(failed)

    missing_core = copy.deepcopy(passing)
    missing_core["metrics"]["primary_conversational"]["by_intent"]["under_table"][
        "correct"
    ] = 0
    with pytest.raises(ValueError, match="did not pass exactly"):
        release.validate_model_gate_contract_v91(missing_core)

    leaked = copy.deepcopy(passing)
    leaked["leakage"]["protected_read_count"] = 1
    with pytest.raises(ValueError, match="leakage or immutable-memory"):
        release.validate_model_gate_contract_v91(leaked)


def test_v91_child_protocol_has_questions_and_no_expectation_channel() -> None:
    cases = release._primary_cases()
    questions = [str(case["question"]) for case in cases]
    answers = {str(case["expected"]) for case in cases}
    command = release._smoke_command(questions)
    assert command.count("--question") == 13
    assert answers.isdisjoint(command)
    assert all(flag not in command for flag in ("--expected", "--answer", "--reference"))


def test_v91_postprocess_audit_protects_both_offline_bridges() -> None:
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
            str(release.PROJECT_ROOT / "data/oracle/scene_000001.json"),
        ]
    }
    assert release._protected_smoke_reads(safe) == []
    violations = release._protected_smoke_reads(unsafe)
    assert len(violations) == 3


def test_v91_promotion_fails_closed_without_exact_passing_smoke(
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
    monkeypatch.setattr(release, "authenticate_v91_model_gate", dict)
    with pytest.raises(ValueError, match="runtime smoke"):
        release.promote_release()


def test_v91_candidate_composes_four_tensors_over_unchanged_v89_parent() -> None:
    if not release.V91_BRIDGE_CANDIDATE.is_dir() or not release.MODEL_GATE_REPORT.is_file():
        pytest.skip("V91 training and evaluation have not both finished")
    model_gate = json.loads(release.MODEL_GATE_REPORT.read_text(encoding="utf-8"))
    if model_gate.get("metrics", {}).get("model_acceptance_gate_passed") is not True:
        pytest.skip("V91 model gate did not authorize runtime packaging")
    evidence = release.authenticate_v91_model_gate()
    tensors, composition = release._composed_adapter(evidence)
    assert composition["base_tensors_byte_identical"] is True
    assert composition["added_tensor_count"] == 4
    assert composition["final_bank_order"] == list(release.EXPECTED_BANKS)
    assert len(tensors) == composition["final_tensor_count"]
