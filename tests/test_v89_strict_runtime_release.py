from __future__ import annotations

import copy
import inspect
import json
from pathlib import Path

import pytest
from safetensors.torch import load_file

from semantic_3d_chat.chat import (
    v89_strict_scene1_cli,
    v89_strict_scene1_runtime,
)
from semantic_3d_chat.chat.v89_strict_scene1_cli import _parser, _run
from semantic_3d_chat.evaluation import v89_strict_runtime_release as release


def _evidence() -> dict[str, object]:
    return {
        "v89_bridge_state_sha256": "a" * 64,
        "v89_bridge_file_sha256": "b" * 64,
        "v89_bridge_metadata_sha256": "c" * 64,
    }


def _passing_report() -> dict[str, object]:
    gates = {name: True for name in release._REQUIRED_MODEL_GATES}
    smoke_records = [
        {
            "question": question,
            "expected": expected,
            "normalized_prediction": expected,
            "exact_correct": True,
            "development_known_and_trained": True,
            "held_out": False,
        }
        for question, expected in release._SMOKE_CASES
    ]
    memory = {
        "prefix_hash_invariant": True,
        "same_prefix_reused_for_every_question": True,
        "question_derived_environmental_tokens": 0,
        "question_conditioned_environmental_readout": False,
        "question_dependent_scene_processing": False,
        "question_dependent_retrieval": False,
        "prefix_sha256_before": release.SOURCE_MEMORY_PREFIX_SHA256,
        "prefix_sha256_after": release.SOURCE_MEMORY_PREFIX_SHA256,
    }
    return {
        "artifact": "gemma4_v89_scene1_retention_evaluation_v1",
        "schema_version": 89,
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
                "development_known_and_trained": True,
                "held_out": False,
                "records": smoke_records,
            },
            "separate_runtime_packaging_authorized": True,
            "runtime_oracle_unavailable_gate_pending": True,
            "runtime_file_audit_gate_pending": True,
            "automatic_runtime_promotion": False,
            "runtime_promotion_authorized": False,
        },
        "scene_memory": memory,
        "leakage": {
            "protected_read_count": 0,
            "protected_reads": [],
            "oracle_loaded": False,
        },
        "fixed_checkpoint_selected_before_scoring": True,
        "checkpoint_selection_after_scoring": False,
        "development_known_smoke_trained": True,
        "held_out_smoke_claim": False,
        "held_out_generalization_claim": False,
        "parent_v85_v86_v87_v88_mutated": False,
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


def test_v89_fixed_evidence_constants_are_exact_sha256() -> None:
    values = (
        release.EXPERIMENT_CONFIG_SHA256,
        release.PREREGISTRATION_SHA256,
        release.CPU_PREFLIGHT_SHA256,
        release.SOURCE_MEMORY_PREFIX_SHA256,
        release.SOURCE_MEMORY_TENSOR_FILE_SHA256,
    )

    assert all(len(value) == 64 for value in values)
    assert all(set(value) <= set("0123456789abcdef") for value in values)


def test_v89_runtime_payload_has_exact_eleven_frozen_banks() -> None:
    payload = release.build_runtime_config_payload(_evidence())
    banks = payload["language"]["lora_banks"]

    assert tuple(banks) == v89_strict_scene1_runtime.EXPECTED_BANKS
    assert len(banks) == 11
    assert all(bank["trainable"] is False for bank in banks.values())
    assert banks["v89_scene1_retention_bridge"] == {
        "trainable": False,
        "rank": 8,
        "alpha": 16.0,
        "dropout": 0.0,
        "initialization_algorithm": "checkpoint_overwrite",
        "initialization_seed": None,
        "expected_initial_state_sha256": "a" * 64,
        "target_modules": [
            "model.language_model.layers.27.self_attn.o_proj"
        ],
    }


def test_v89_payload_rebinds_every_v85_bank_to_authenticated_final_state() -> None:
    payload = release.build_runtime_config_payload(_evidence())
    configured = payload["language"]["lora_banks"]
    parent_metadata = json.loads(
        (release.V85_CHECKPOINT / "runtime_metadata.json").read_text(
            encoding="utf-8"
        )
    )
    states = parent_metadata["lora_bank_state_sha256"]

    assert tuple(configured)[:7] == v89_strict_scene1_runtime.EXPECTED_BANKS[:7]
    assert all(
        configured[name]["expected_initial_state_sha256"] == states[name]
        for name in v89_strict_scene1_runtime.EXPECTED_BANKS[:7]
    )


def test_v89_gate_contract_requires_exact_all_pass_result() -> None:
    passing = _passing_report()
    release.validate_model_gate_contract_v89(passing)

    failed = copy.deepcopy(passing)
    failed["metrics"]["model_acceptance_gates"][
        "all_scene1_canonical_accuracy_at_least_0_80"
    ] = False
    with pytest.raises(ValueError, match="did not pass exactly"):
        release.validate_model_gate_contract_v89(failed)

    leaked = copy.deepcopy(passing)
    leaked["leakage"]["protected_read_count"] = 1
    with pytest.raises(ValueError, match="leakage, smoke, or memory"):
        release.validate_model_gate_contract_v89(leaked)

    false_held_out = copy.deepcopy(passing)
    false_held_out["held_out_smoke_claim"] = True
    with pytest.raises(ValueError, match="did not pass exactly"):
        release.validate_model_gate_contract_v89(false_held_out)


def test_v89_chat_runtime_imports_no_v89_training_evaluation_or_release() -> None:
    source = inspect.getsource(v89_strict_scene1_runtime).casefold()
    cli_source = inspect.getsource(v89_strict_scene1_cli).casefold()

    for forbidden in (
        "train_v89_scene1_retention",
        "evaluate_v89_scene1_retention",
        "v89_scene1_retention_preflight",
        "_smoke_cases",
        "reference_answer",
        "target_instance",
        "error_inventory",
        "anchor_inventory",
    ):
        assert forbidden not in source
        assert forbidden not in cli_source
    assert (
        "semantic_3d_chat.evaluation.v89_strict_runtime_release" not in source
    )
    assert (
        "semantic_3d_chat.evaluation.v89_strict_runtime_release" not in cli_source
    )


def test_v89_cli_is_scene1_only_and_defaults_to_promoted_release() -> None:
    defaults = _parser().parse_args([])
    candidate = _parser().parse_args(["--allow-candidate"])

    assert defaults.allow_candidate is False
    assert defaults.config.endswith("gemma4_v89_strict_scene1.yaml")
    assert defaults.base_checkpoint.endswith("gemma4_v89_strict_scene1_release_v1")
    assert defaults.scene_memory.endswith("runtime/scene_memories/v89/scene_000001")
    assert candidate.allow_candidate is True
    with pytest.raises(ValueError, match="only scene_000001"):
        _run(["--scene", "scene_000039", "--question", "anything"])


def test_v89_promotion_fails_closed_without_exact_passing_smoke(
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
        "authenticate_v89_model_gate",
        lambda: {
            "model_gate_report_sha256": "7" * 64,
            "v89_bridge_state_sha256": "8" * 64,
        },
    )

    with pytest.raises(ValueError, match="did not pass exactly"):
        release.promote_release()


@pytest.mark.skipif(
    not release.CANDIDATE_CHECKPOINT.is_dir(),
    reason="passing-gate V89 runtime candidate not packaged",
)
def test_v89_candidate_is_exact_eleven_bank_parent_plus_eight_tensors() -> None:
    result = release.verify_candidate()
    candidate = load_file(
        str(release.CANDIDATE_CHECKPOINT / "adapter.safetensors"), device="cpu"
    )
    parent = load_file(
        str(release.V85_CHECKPOINT / "adapter.safetensors"), device="cpu"
    )

    assert result["passed"] is True
    assert set(parent).issubset(candidate)
    assert all(candidate[name].equal(value) for name, value in parent.items())
    assert len(set(candidate) - set(parent)) == 8
    metadata = json.loads(
        (release.CANDIDATE_CHECKPOINT / "runtime_metadata.json").read_text(
            encoding="utf-8"
        )
    )
    assert tuple(row["name"] for row in metadata["lora"]["banks"]) == (
        v89_strict_scene1_runtime.EXPECTED_BANKS
    )
    assert metadata["lora_parameter_count"] == 872_448
    assert metadata["lora_trainable_parameter_count"] == 0


def test_v89_release_cannot_exist_without_passing_external_smoke() -> None:
    if release.RELEASE_CHECKPOINT.exists():
        smoke = json.loads(release.SMOKE_REPORT.read_text(encoding="utf-8"))
        assert smoke["passed"] is True
        assert smoke["promotion_authorized"] is True
        assert release.RUNTIME_CONFIG.is_file()
