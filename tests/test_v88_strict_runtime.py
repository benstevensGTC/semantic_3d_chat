from __future__ import annotations

import copy
import inspect
import json
from pathlib import Path

import pytest

from semantic_3d_chat.chat import v88_strict_scene1_runtime
from semantic_3d_chat.evaluation import v88_strict_runtime_release as release
from semantic_3d_chat.evaluation.v88_strict_runtime_release import (
    CANDIDATE_CHECKPOINT,
    FINAL_BANKS,
    RELEASE_CHECKPOINT,
    RUNTIME_CONFIG,
    V88_CONTRACT,
    authenticate_v88_model_gate,
    validate_model_gate_contract_v88,
)


def _passing_report() -> dict[str, object]:
    gates = {name: True for name in release._REQUIRED_GATES}
    smoke_records = [
        {
            "exact_correct": True,
            "development_known_and_trained": True,
            "held_out": False,
        }
        for _ in range(3)
    ]
    return {
        "artifact": "gemma4_v88_scene1_augmented_evaluation_v1",
        "schema_version": 88,
        "status": "model_gates_pass_separate_runtime_packaging_required",
        "metrics": {
            "model_acceptance_gates": gates,
            "model_acceptance_gate_passed": True,
            "separate_runtime_packaging_authorized": True,
            "runtime_promotion_authorized": False,
            "canonical_type_specific": {
                "correct": 111,
                "total": 138,
                "accuracy": 111 / 138,
            },
            "canonical_accuracy_by_answer_type": {
                "attribute": {"accuracy": 0.50},
                "presence": {"accuracy": 0.75},
                "spatial_relation": {"accuracy": 0.60},
            },
            "causal_control": {
                "mean_zero_minus_correct_nll": 0.1,
                "canonical_prediction_changes": 1,
            },
            "generic_smoke": {
                "correct": 3,
                "total": 3,
                "accuracy": 1.0,
                "development_known_and_trained": True,
                "held_out": False,
                "records": smoke_records,
            },
        },
        "separate_runtime_packaging_authorized": True,
        "automatic_runtime_promotion": False,
        "runtime_promotion_authorized": False,
        "fixed_checkpoint_selected_before_scoring": True,
        "checkpoint_selection_after_scoring": False,
        "development_known_smoke_trained": True,
        "held_out_smoke_claim": False,
        "held_out_generalization_claim": False,
        "parent_v85_v86_v87_mutated": False,
        "oracle_loaded": False,
        "official_validation_loaded": False,
        "official_test_loaded": False,
        "deferred_final_loaded": False,
        "leakage": {
            "protected_read_count": 0,
            "protected_reads": [],
            "oracle_loaded": False,
        },
        "scene_memory": {
            "prefix_hash_invariant": True,
            "same_prefix_reused_for_every_question": True,
            "question_derived_environmental_tokens": 0,
            "prefix_sha256_before": release.SOURCE_MEMORY_PREFIX_SHA256,
            "prefix_sha256_after": release.SOURCE_MEMORY_PREFIX_SHA256,
        },
    }


def test_v88_runtime_module_has_no_training_or_evaluation_imports() -> None:
    source = inspect.getsource(v88_strict_scene1_runtime).casefold()

    for forbidden in (
        "semantic_3d_chat.evaluation.evaluate_v88_scene1_augmented",
        "semantic_3d_chat.training.train_v88_scene1_augmented",
        "semantic_3d_chat.evaluation.v88_scene1_augmented_preflight",
        "semantic_3d_chat.evaluation.v88_strict_runtime_release",
        "question_answer",
        "error_inventory",
        "augmentation_inventory",
        "_smoke_cases",
    ):
        assert forbidden not in source


def test_v88_runtime_is_scene1_only_and_exact_ten_bank_contract() -> None:
    assert v88_strict_scene1_runtime.SCENE_ID == "scene_000001"
    assert len(FINAL_BANKS) == 10
    assert FINAL_BANKS[-3:] == (
        "v86_scene1_demo_bridge",
        "v87_scene1_balanced_bridge",
        "v88_scene1_augmented_bridge",
    )
    assert V88_CONTRACT.target_module == (
        "model.language_model.layers.27.self_attn.q_proj"
    )
    assert V88_CONTRACT.rank == 16
    assert V88_CONTRACT.alpha == 32.0
    assert V88_CONTRACT.parameter_count == 57_344


def test_v88_model_gate_accepts_only_all_true_development_known_result() -> None:
    passing = _passing_report()
    validate_model_gate_contract_v88(passing)

    failed = copy.deepcopy(passing)
    failed["metrics"]["model_acceptance_gates"][
        "all_scene1_canonical_accuracy_at_least_0_80"
    ] = False
    with pytest.raises(ValueError, match="did not pass exactly"):
        validate_model_gate_contract_v88(failed)

    false_held = copy.deepcopy(passing)
    false_held["held_out_smoke_claim"] = True
    with pytest.raises(ValueError, match="did not pass exactly"):
        validate_model_gate_contract_v88(false_held)


def test_sealed_v88_result_refuses_runtime_packaging() -> None:
    with pytest.raises(ValueError, match="did not pass exactly"):
        authenticate_v88_model_gate()

    assert not RUNTIME_CONFIG.exists()
    assert not CANDIDATE_CHECKPOINT.exists()
    assert not RELEASE_CHECKPOINT.exists()


def test_v88_release_source_exposes_no_write_or_promotion_command() -> None:
    source = inspect.getsource(release)
    parser_choices = inspect.getsource(release.main)

    assert 'choices=("authenticate", "verify-composition")' in parser_choices
    assert "def promote_release" not in source
    assert "def prepare_candidate" not in source
    assert "save_file(" not in source
    assert "write_text(" not in source


def test_v88_failed_experiment_created_no_runtime_artifact() -> None:
    absent = (
        RUNTIME_CONFIG,
        CANDIDATE_CHECKPOINT,
        RELEASE_CHECKPOINT,
        Path("data_gemma4/runtime/scene_memories/v88/scene_000001"),
    )
    assert all(not path.exists() for path in absent)


def test_v88_runtime_checkpoint_metadata_cannot_serialize_supervision() -> None:
    encoded = json.dumps(
        {
            "bank_name": V88_CONTRACT.bank_name,
            "target_module": V88_CONTRACT.target_module,
            "state_sha256": V88_CONTRACT.state_sha256,
        },
        sort_keys=True,
    ).casefold()
    assert all(
        word not in encoded
        for word in (
            "oracle",
            "question",
            "answer",
            "error_inventory",
            "augmentation_inventory",
            "object_name",
        )
    )
