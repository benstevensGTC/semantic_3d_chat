from __future__ import annotations

import json
from pathlib import Path

import pytest

from semantic_3d_chat.evaluation import v41_retry1_update8_terminal_gate as gate


@pytest.fixture(scope="module")
def terminal() -> dict:
    return gate.audit_v41_retry1_update8()


def test_terminal_seals_failed_update8_and_denies_broader_actions(
    terminal: dict,
) -> None:
    assert terminal["artifact"] == "v41_retry1_update8_terminal_gate"
    assert terminal["seal_revision"] == 2
    assert terminal["passed"] is True
    assert terminal["terminal_conclusion"] == (
        "update8_train_only_gate_failed_stop_is_final"
    )
    assert terminal["only_exact_successor_authorized"] == (
        "v42_train_only_no_step_diagnostic_screen"
    )
    assert terminal["v42_train_only_no_step_diagnostic_screen_authorized"] is True
    for field in (
        "arbitrary_training_authorized",
        "resume_v41_training_authorized",
        "parameter_update_authorized",
        "validation_access_authorized",
        "oracle_access_authorized",
        "final_test_access_authorized",
        "selector_execution_authorized",
        "chat_or_runtime_promotion_authorized",
        "embodied_agent_promotion_authorized",
    ):
        assert terminal[field] is False


def test_exact_checkpoint_inventory_is_hash_bound(terminal: dict) -> None:
    inventory = terminal["checkpoint_inventory"]
    assert inventory["root_entries"] == ["update_000", "update_008"]
    assert inventory["update_000_entries"] == [
        "adapter.safetensors",
        "metadata.json",
        "runtime_metadata.json",
    ]
    assert inventory["update_008_entries"] == [
        "adapter.safetensors",
        "metadata.json",
        "optimizer.pt",
        "optimizer_audit.json",
        "runtime_metadata.json",
    ]
    assert inventory["file_sha256"] == gate._FILES
    assert inventory["no_update_after_eight_persisted"] is True


def test_only_exact_layer14_lora_b_changed(terminal: dict) -> None:
    transition = terminal["tensor_transition"]
    assert transition["tensor_count"] == 179
    assert transition["changed_tensor_names"] == [gate._TARGET]
    assert transition["changed_tensor_count"] == 1
    assert transition["changed_parameter_count"] == 16_384
    assert transition["target_shape"] == [4096, 4]
    assert transition["target_lora_b_state_sha256"] == {
        "update_000": gate._TARGET_U0_SHA256,
        "update_008": gate._TARGET_U8_SHA256,
    }
    assert transition["frozen_excluding_target_state_sha256"] == {
        "update_000": gate._FROZEN_SHA256,
        "update_008": gate._FROZEN_SHA256,
    }
    assert transition["only_existing_layer14_q_proj_lora_b_changed"] is True
    assert transition["every_other_tensor_bit_exact"] is True
    assert transition["all_tensors_finite"] is True


def test_optimizer_is_exact_stateless_sgd_step8_payload(terminal: dict) -> None:
    optimizer = terminal["optimizer_integrity"]
    assert optimizer["optimizer_step"] == 8
    assert optimizer["optimizer_sha256"] == gate._FILES[
        "update_008/optimizer.pt"
    ]
    assert optimizer["implementation"] == "torch.optim.SGD"
    assert optimizer["stateless_momentum_free_payload"] is True
    assert optimizer["state_entry_count"] == 0
    assert optimizer["parameter_group_count"] == 1
    assert optimizer["parameter_names"] == [gate._TARGET]
    assert optimizer["learning_rate"] == 0.003
    assert optimizer["momentum"] == 0.0
    assert optimizer["foreach"] is False
    assert optimizer["fused"] is False
    assert optimizer["update_zero_optimizer_absent"] is True


def test_projection_history_replays_all_eight_steps(terminal: dict) -> None:
    replay = terminal["projection_history_replay"]
    assert replay["validated_optimizer_steps"] == 8
    assert replay["all_projected_microsteps_authenticated"] is True
    assert replay["optimizer_steps"] == list(range(1, 9))
    assert replay["selected_masks"] == [0, 0, 1, 0, 0, 0, 0, 1]
    assert replay["active_constraint_counts"] == [4, 4, 4, 4, 4, 3, 4, 4]
    assert replay["projected_steps"] == [3, 8]
    assert replay["all_target_and_frozen_hash_chains_authenticated"] is True
    assert replay["all_device_cast_and_clip_attestations_authenticated"] is True


def test_update8_gate_replays_exact_failure(terminal: dict) -> None:
    replay = terminal["update8_gate_replay"]
    assert replay["passed"] is False
    assert replay["replayed_exactly"] is True
    assert replay["failed_checks"] == [
        "priority_teacher_deficit_improved_at_least_0_5",
        "teacher_cross_complete_units_at_least_17",
        "teacher_positive_sides_at_least_34",
    ]
    assert replay["complete_units"] == 9
    assert replay["positive_sides"] == 33
    assert replay["cross_prefix_complete_units"] == 16
    assert replay["source_broad_nll"] == gate._SOURCE_BROAD_NLL
    assert replay["update8_broad_nll"] == gate._UPDATE8_BROAD_NLL
    assert replay["priority_side_deficit_improvement"] == (
        gate._UPDATE8_PRIORITY_IMPROVEMENT
    )
    assert replay["stopped_before_update_nine"] is True
    assert replay["update16_gate_absent"] is True
    assert replay["update41_gate_absent"] is True


def test_retry_provenance_and_runtime_boundaries_are_exact(terminal: dict) -> None:
    provenance = terminal["retry1_provenance"]
    assert provenance["retry1_terminal_report_sha256"] == (
        gate._RETRY1_TERMINAL_SHA256
    )
    assert provenance["authorization_id"] == (
        "v41_retry1_cpu_first_projected_gradient_l14_lora_b"
    )
    assert provenance[
        "same_authorization_and_predecessor_persisted_at_u0_and_u8"
    ] is True
    assert provenance["cpu_first_mps_conversion_required"] is True
    assert provenance["restricted_access_remained_denied"] is True
    for arm in ("update_000", "update_008"):
        assert terminal["runtime_metadata_audit"][arm][
            "sanitized_runtime_exact"
        ] is True
    access = terminal["terminal_process_access_audit"]
    for field in (
        "gemma_loaded",
        "qa_loaded",
        "maps_loaded",
        "validation_loaded",
        "oracle_loaded",
        "final_test_loaded",
        "optimizer_step_executed",
    ):
        assert access[field] is False


def test_only_one_train_only_no_step_v42_screen_is_authorized(
    terminal: dict,
) -> None:
    authorization = terminal["conditional_successor_authorization"]
    assert authorization["authorization_id"] == (
        "v42_v41_retry1_update8_no_step_diagnostic_screen"
    )
    assert authorization["only_exact_action"] == (
        "one_train_only_no_step_diagnostic_screen"
    )
    scope = authorization["diagnostic_scope"]
    assert scope["single_report_only"] is True
    assert scope["exact_training_qa_and_maps_read_only"] is True
    assert scope["forward_only_candidate_diagnostics_allowed"] is True
    assert scope["gradient_measurement_authorized"] is False
    assert scope["temporary_target_b_substitution_authorized"] is True
    assert scope[
        "temporary_substitution_must_restore_exact_u0_after_each_candidate"
    ] is True
    assert scope["fixed_alpha_grid"] == gate._V42_ALPHA_GRID
    assert scope["fixed_candidate_state_sha256"] == gate._V42_CANDIDATE_HASHES
    assert scope["adaptive_candidate_refinement_authorized"] is False
    for field in (
        "parameter_mutation_authorized",
        "persistent_parameter_mutation_authorized",
        "optimizer_construction_authorized",
        "optimizer_deserialization_authorized",
        "optimizer_step_authorized",
        "checkpoint_write_authorized",
        "resume_training_authorized",
        "candidate_training_authorized",
    ):
        assert scope[field] is False
    assert authorization["selector_execution_authorized"] is False
    assert authorization["chat_or_runtime_promotion_authorized"] is False
    assert authorization["new_terminal_seal_required_after_diagnostic"] is True


def test_materialized_report_is_exact_replay() -> None:
    report_path = gate._resolve(gate.DEFAULT_OUTPUT)
    if not report_path.is_file():
        pytest.skip("V41 retry1 update-eight terminal has not been materialized")
    assert json.loads(report_path.read_text(encoding="utf-8")) == (
        gate.audit_v41_retry1_update8()
    )


def test_terminal_write_is_atomic_and_source_artifacts_remain_exact(
    tmp_path: Path,
) -> None:
    before = {
        relative: gate._sha256(gate._resolve(gate.DEFAULT_CHECKPOINT_ROOT) / relative)
        for relative in gate._FILES
    }
    output = tmp_path / "terminal.json"
    report = gate.write_report(output)
    assert json.loads(output.read_text(encoding="utf-8")) == report
    assert list(tmp_path.iterdir()) == [output]
    after = {
        relative: gate._sha256(gate._resolve(gate.DEFAULT_CHECKPOINT_ROOT) / relative)
        for relative in gate._FILES
    }
    assert before == after == gate._FILES


def test_terminal_fails_closed_on_changed_artifact_pin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(gate._FILES, "update_008/metadata.json", "0" * 64)
    with pytest.raises(ValueError, match="metadata\\.json bytes changed"):
        gate.audit_v41_retry1_update8()
