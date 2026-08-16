from __future__ import annotations

import copy
import inspect
import json
from pathlib import Path

import pytest
from safetensors.torch import load_file

from semantic_3d_chat.chat.v87_strict_scene1_cli import _parser, _run
from semantic_3d_chat.chat.v87_strict_scene1_runtime import (
    V86_BANK,
    V86_STATE_SHA256,
    V87_BANK,
    V87_TARGET,
)
from semantic_3d_chat.evaluation import v87_strict_runtime_release as release
from semantic_3d_chat.evaluation.v87_strict_runtime_release import (
    CANDIDATE_CHECKPOINT,
    CANDIDATE_MEMORY,
    RELEASE_CHECKPOINT,
    RUNTIME_CONFIG,
    SOURCE_MEMORY_PREFIX_SHA256,
    SOURCE_MEMORY_TENSOR_FILE_SHA256,
    build_runtime_config_payload,
    promote_release,
    sha256_file,
    validate_model_gate_contract,
    verify_candidate,
)
from semantic_3d_chat.scene_encoder.v81_scene_memory_artifact import MEMORY_FILENAME


def _passing_model_gate() -> dict[str, object]:
    gates = {
        "all_scene1_canonical_accuracy_at_least_0_80": True,
        "attribute_accuracy_at_least_0_50": True,
        "presence_accuracy_at_least_0_75": True,
        "spatial_relation_accuracy_at_least_0_60": True,
        "exact_training_row_count_138": True,
        "generic_live_smoke_exactly_3_of_3": True,
        "causal_correct_memory_mean_nll_below_zero_payload": True,
        "causal_prediction_change_at_least_1": True,
        "exact_prefix_hash_invariance": True,
        "exact_total_environment_input_invariance": True,
        "protected_read_count_zero": True,
    }
    return {
        "artifact": "gemma4_v87_scene1_balanced_evaluation_v1",
        "schema_version": 87,
        "status": "model_gates_pass_separate_runtime_packaging_required",
        "metrics": {
            "model_acceptance_gates": gates,
            "model_acceptance_gate_passed": True,
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
                "records": [
                    {
                        "question": "Is there a chair?",
                        "expected": "yes",
                        "normalized_prediction": "yes",
                        "exact_correct": True,
                    },
                    {
                        "question": "What color is the bowl?",
                        "expected": "red",
                        "normalized_prediction": "red",
                        "exact_correct": True,
                    },
                    {
                        "question": "Is the bowl left or right of the chair?",
                        "expected": "left",
                        "normalized_prediction": "left",
                        "exact_correct": True,
                    },
                ],
            },
            "separate_runtime_packaging_authorized": True,
            "runtime_oracle_unavailable_gate_pending": True,
            "runtime_file_audit_gate_pending": True,
            "automatic_runtime_promotion": False,
            "runtime_promotion_authorized": False,
        },
        "leakage": {
            "protected_read_count": 0,
            "protected_reads": [],
            "oracle_loaded": False,
        },
        "scene_memory": {
            "prefix_hash_invariant": True,
            "same_prefix_reused_for_every_question": True,
            "question_derived_environmental_tokens": 0,
            "prefix_sha256_before": SOURCE_MEMORY_PREFIX_SHA256,
            "prefix_sha256_after": SOURCE_MEMORY_PREFIX_SHA256,
        },
        "separate_runtime_packaging_authorized": True,
        "runtime_oracle_unavailable_gate_pending": True,
        "runtime_file_audit_gate_pending": True,
        "automatic_runtime_promotion": False,
        "runtime_promotion_authorized": False,
        "fixed_checkpoint_selected_before_scoring": True,
        "checkpoint_selection_after_scoring": False,
        "held_out_generalization_claim": False,
        "parent_v86_mutated": False,
        "oracle_loaded": False,
        "official_validation_loaded": False,
        "official_test_loaded": False,
        "deferred_final_loaded": False,
    }


def test_v87_runtime_payload_has_exact_nine_frozen_banks() -> None:
    state = "7" * 64
    payload = build_runtime_config_payload(state)
    banks = payload["language"]["lora_banks"]

    assert len(banks) == 9
    assert all(bank["trainable"] is False for bank in banks.values())
    assert banks[V86_BANK]["expected_initial_state_sha256"] == V86_STATE_SHA256
    assert banks[V87_BANK]["expected_initial_state_sha256"] == state
    assert banks[V87_BANK]["target_modules"] == [V87_TARGET]
    assert banks[V87_BANK]["rank"] == 8
    assert banks[V87_BANK]["alpha"] == 16.0


def test_v87_runtime_payload_has_no_environmental_labels_or_supervision() -> None:
    encoded = json.dumps(build_runtime_config_payload("7" * 64), sort_keys=True).casefold()

    for forbidden in (
        "oracle",
        "chair",
        "bowl",
        "picture frame",
        "red cube",
        "scene graph",
        "target_instance",
        "reference_answer",
    ):
        assert forbidden not in encoded


def test_v87_model_gate_contract_accepts_only_all_true_exact_inventory() -> None:
    report = _passing_model_gate()
    validate_model_gate_contract(report)

    failed = copy.deepcopy(report)
    failed["metrics"]["model_acceptance_gates"][
        "attribute_accuracy_at_least_0_50"
    ] = False
    with pytest.raises(ValueError, match="did not pass exactly"):
        validate_model_gate_contract(failed)

    extra = copy.deepcopy(report)
    extra["metrics"]["model_acceptance_gates"]["unregistered_gate"] = True
    with pytest.raises(ValueError, match="did not pass exactly"):
        validate_model_gate_contract(extra)


def test_v87_model_gate_contract_rejects_leakage_and_automatic_promotion() -> None:
    leaked = _passing_model_gate()
    leaked["leakage"]["protected_read_count"] = 1
    with pytest.raises(ValueError, match="leakage or memory"):
        validate_model_gate_contract(leaked)

    promoted = _passing_model_gate()
    promoted["automatic_runtime_promotion"] = True
    with pytest.raises(ValueError, match="did not pass exactly"):
        validate_model_gate_contract(promoted)


def test_v87_cli_defaults_to_release_and_candidate_requires_flag() -> None:
    defaults = _parser().parse_args([])
    candidate = _parser().parse_args(["--allow-candidate"])

    assert defaults.allow_candidate is False
    assert defaults.config.endswith("gemma4_v87_strict_scene1.yaml")
    assert defaults.base_checkpoint.endswith("gemma4_v87_strict_scene1_release_v1")
    assert defaults.scene_memory.endswith("runtime/scene_memories/v87/scene_000001")
    assert candidate.allow_candidate is True


def test_v87_cli_refuses_every_scene_except_scene_one() -> None:
    with pytest.raises(ValueError, match="only scene_000001"):
        _run(["--scene", "scene_000039", "--question", "anything"])


def test_v87_chat_process_does_not_import_release_or_evaluation_surface() -> None:
    from semantic_3d_chat.chat import v87_strict_scene1_cli

    source = inspect.getsource(v87_strict_scene1_cli)
    assert "v87_strict_runtime_release" not in source
    assert "evaluate_v87" not in source
    assert "_SMOKE_CASES" not in source


def test_v87_promotion_fails_closed_when_smoke_did_not_pass(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    smoke = tmp_path / "smoke.json"
    smoke.write_text(
        json.dumps({"passed": False, "promotion_authorized": False}),
        encoding="utf-8",
    )
    monkeypatch.setattr(release, "SMOKE_REPORT", smoke)
    monkeypatch.setattr(
        release,
        "authenticate_v87_model_gate",
        lambda: {"model_gate_report_sha256": "7" * 64},
    )

    with pytest.raises(ValueError, match="did not pass exactly"):
        promote_release()


@pytest.mark.skipif(
    not CANDIDATE_CHECKPOINT.is_dir(), reason="post-gate V87 candidate not packaged"
)
def test_v87_candidate_is_exact_and_preserves_v85_subset() -> None:
    result = verify_candidate()
    candidate = load_file(str(CANDIDATE_CHECKPOINT / "adapter.safetensors"), device="cpu")
    base = load_file(
        str(release.V85_RUNTIME_CHECKPOINT / "adapter.safetensors"), device="cpu"
    )

    assert result["passed"] is True
    assert set(base).issubset(candidate)
    assert all(candidate[name].equal(value) for name, value in base.items())
    assert len(set(candidate) - set(base)) == 4
    assert {item.name for item in CANDIDATE_CHECKPOINT.iterdir()} == {
        "adapter.safetensors",
        "runtime_metadata.json",
    }
    encoded_metadata = (
        CANDIDATE_CHECKPOINT / "runtime_metadata.json"
    ).read_text(encoding="utf-8").casefold()
    assert all(
        token not in encoded_metadata
        for token in (
            "oracle",
            "answer_text",
            "question_text",
            "object_name",
            "scene_graph",
            "target_instance",
        )
    )


@pytest.mark.skipif(
    not CANDIDATE_MEMORY.is_dir(), reason="post-gate V87 memory not packaged"
)
def test_v87_candidate_memory_rebinding_changes_metadata_only() -> None:
    assert sha256_file(CANDIDATE_MEMORY / MEMORY_FILENAME) == (
        SOURCE_MEMORY_TENSOR_FILE_SHA256
    )
    assert {item.name for item in CANDIDATE_MEMORY.iterdir()} == {
        "memory.safetensors",
        "runtime_metadata.json",
    }


def test_v87_release_cannot_exist_before_runtime_smoke_passes() -> None:
    if RELEASE_CHECKPOINT.exists():
        smoke = json.loads(release.SMOKE_REPORT.read_text(encoding="utf-8"))
        assert smoke["passed"] is True
        assert smoke["promotion_authorized"] is True
        assert RUNTIME_CONFIG.is_file()
