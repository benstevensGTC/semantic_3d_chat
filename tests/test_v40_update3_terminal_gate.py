from __future__ import annotations

import json
from pathlib import Path

import pytest

from semantic_3d_chat.evaluation.v40_update3_terminal_gate import (
    audit_v40_update3_failure,
)


@pytest.fixture(scope="module")
def terminal() -> dict:
    return audit_v40_update3_failure()


def test_terminal_authenticates_exact_v40_inputs(terminal: dict) -> None:
    assert terminal["passed"] is True
    assert terminal["artifact"] == "v40_update3_terminal_gate"
    assert terminal["seal_revision"] == 3
    assert terminal["supersedes_revision1_sha256"] == (
        "9ce66c309adeb9e81636dc45cc1237e20cb7f050ed6ba9fe0cfa762c6c33660e"
    )
    assert terminal["supersedes_revision2_sha256"] == (
        "63b22d16303018d0482710a34c1a848d131a0e65a34dd96735fcfb241deba844"
    )
    integrity = terminal["input_integrity"]
    assert integrity["v39_terminal"]["sha256"] == (
        "fcf0494c18ed13c3f1fe54eb109a51391183a4eeb14abb9dbd2ad0ad0ca448c3"
    )
    assert integrity["v40_config"]["sha256"] == (
        "5e7d67a91a10f65e44699a7af1644fffff481dcd21ce34a028cf371048f1c9bc"
    )
    assert integrity["v40_trainer"]["sha256"] == (
        "0801d580e4903c82a481f81383a7319e883e431b211fb18761d4af2a72d1fdaa"
    )
    assert integrity["guard_failure"]["sha256"] == (
        "0136edab8669346e4c24163650659608a7bf728fb8244ad81366a8efb5fa1f61"
    )


def test_v40_failure_is_only_broad_direction_at_pre_step_three(
    terminal: dict,
) -> None:
    failure = terminal["failure_replay"]
    assert failure["failed_before_optimizer_step"] == 3
    assert failure["optimizer_step_three_executed"] is False
    assert failure["clip_applied_for_update_three"] is False
    assert failure["checkpoint_written_for_update_three"] is False
    assert failure["only_failed_direction"] == "broad"
    assert failure["broad_dot_with_raw_total"] == -0.06791611851052101
    assert failure["broad_cosine_with_raw_total"] == -0.038372349473092954
    assert failure["target_bit_exact_across_failed_attempt"] is True
    assert failure["frozen_surface_bit_exact_across_failed_attempt"] is True


def test_v40_only_persisted_exact_update_zero(terminal: dict) -> None:
    envelope = terminal["stopped_envelope"]
    assert envelope["root_entries"] == [
        "guard_failure_update_003.json",
        "update_000",
    ]
    assert envelope["persisted_update_directories"] == ["update_000"]
    assert envelope["only_update_zero_persisted"] is True
    assert envelope["no_update_three_or_later_checkpoint"] is True
    assert envelope["no_trained_optimizer_file_persisted"] is True
    source = terminal["persisted_update_zero"]
    assert source["optimizer_step"] == 0
    assert source["only_update_zero_history_persisted"] is True
    assert source["adapter_sha256"] == (
        "b0bdceb7699e9d97467915c69186f433d3b0fac2b09144d38c3078afe1f70cb0"
    )
    assert source["metadata_sha256"] == (
        "ee74b1572061cae09d20bdf2b07e5f94ce9ef5c3ebfb6908131448bf8e5b484d"
    )
    assert source["runtime_metadata_sha256"] == (
        "209858f923ffa0916484209aeefad6f56a2cb4902bbd0dacd29decc222245c49"
    )


def test_v40_transient_updates_one_two_are_discarded_and_u3_never_executes(
    terminal: dict,
) -> None:
    conclusion = terminal["execution_conclusion"]
    assert conclusion["optimizer_steps_one_and_two_executed_in_memory"] is True
    assert conclusion["optimizer_step_three_executed"] is False
    assert conclusion["optimizer_step_three_failed_before_clip"] is True
    assert conclusion["optimizer_step_three_target_state_unchanged"] is True
    assert conclusion["only_update_zero_checkpoint_persisted"] is True
    assert conclusion["no_trained_checkpoint_persisted"] is True
    assert conclusion["transient_updates_discarded_on_process_exit"] is True
    assert conclusion["v41_must_restart_exact_update_zero"] is True
    assert terminal["failure_replay"]["transient_pre_update3_target_sha256"] == (
        "8c50aa3a5975f450c3c95fb00dbf077a33285bf22ac3208f5d745cd617bd8d48"
    )


def test_v41_source_is_exact_v40_and_v38_update_zero(terminal: dict) -> None:
    source = terminal["conditional_successor_authorization"]["source"]
    assert source["checkpoint"].endswith(
        "gemma4_v40_diverse28_cross_preserving_l14_query/update_000"
    )
    assert source["optimizer_step"] == 0
    assert source["source_is_exact_v40_and_v38_update_zero"] is True
    assert source["v40_transient_pre_update3_state_is_not_a_source_checkpoint"] is True
    assert source["full_tensor_state_sha256"] == (
        "7b951c6d7ae4f7b50603159f0bc4dfb4d50b5b40f9325134d78d1de1dae87fc0"
    )
    assert source["target_lora_b_state_sha256"] == (
        "d0834cc588ee2a9edf08aabedfd01e0a6d2b01c6b6ae7e3a3d764eaddf58cc3e"
    )
    assert source["source_optimizer_access_authorized"] is False


def test_v41_authorizes_same_b_only_surface_and_weighted_components(
    terminal: dict,
) -> None:
    authorization = terminal["conditional_successor_authorization"]
    target = authorization["target_surface"]
    assert target["parameter_names"] == [
        "lora_banks.extension_v28_stage_b_query.adapters.1.lora_b"
    ]
    assert target["parameter_shapes"] == [[4096, 4]]
    assert target["tensor_count"] == 1
    assert target["parameter_count"] == 16_384
    assert target["lora_b_only"] is True
    assert target["lora_a_frozen_learned_basis"] is True
    assert authorization["weighted_objective_components"] == {
        "broad": 1.0,
        "answer": 0.5,
        "side": 8.0,
        "cross": 56.0,
        "side_hinge_margin": 0.5,
        "cross_prefix_margin": 0.1,
        "scene_formula": "side + cross",
        "raw_total_formula": "broad + answer + side + cross",
    }


def test_v41_projection_qp_is_closed_deterministic_and_exact(terminal: dict) -> None:
    solver = terminal["conditional_successor_authorization"][
        "projected_gradient_solver"
    ]
    assert solver["solver_device"] == "cpu"
    assert solver["solver_dtype"] == "torch.float64"
    assert solver["constraint_direction_order"] == [
        "broad",
        "answer",
        "scene",
        "cross",
    ]
    assert solver["beta_formula"] == "max(1e-12, 1e-4 * l2(g_raw))"
    assert solver["beta_absolute_floor"] == 1e-12
    assert solver["beta_raw_norm_multiplier"] == 1e-4
    active = solver["active_set_enumeration"]
    assert active["maximum_constraint_count"] == 4
    assert active["active_constraint_count_allowed"] == [1, 2, 3, 4]
    assert active["mask_count_formula"] == "2 ** active_constraint_count"
    assert active["mask_count_allowed"] == [2, 4, 8, 16]
    assert active["mask_order"] == (
        "ascending_integer_over_canonical_active_direction_order"
    )
    assert active["independent_active_subsets_only"] is True
    assert active["rank_absolute_tolerance"] == 1e-12
    assert active["rank_relative_tolerance"] == 1e-10
    assert active["dual_lambda_lower_tolerance"] == -1e-10
    assert active["kkt_absolute_tolerance"] == 1e-10
    assert active["kkt_relative_tolerance"] == 1e-8
    assert active["require_primal_feasibility"] is True
    assert active["require_dual_feasibility"] is True
    assert active["require_active_equality_feasibility"] is True
    assert active["require_stationarity"] is True
    assert active["require_complementarity"] is True
    assert active["selection"] == "minimum_objective_then_lowest_mask"


def test_v41_zero_hinge_directions_are_explicitly_inactive_not_silent(
    terminal: dict,
) -> None:
    solver = terminal["conditional_successor_authorization"][
        "projected_gradient_solver"
    ]
    assert solver["authorization_revision"] == 3
    assert solver["require_all_raw_components_finite"] is True
    assert solver["nonfinite_component_action"] == "fail_stop_before_mutation"
    assert solver["constraint_activity_norm_floor_inclusive"] == 0.0
    assert solver["active_constraint_rule"] == (
        "finite_l2_norm_strictly_greater_than_zero"
    )
    assert solver["active_constraint_direction_order"] == (
        "stable_subsequence_of_broad_answer_scene_cross_with_positive_norm"
    )
    assert solver["inactive_constraint_direction_order"] == (
        "stable_subsequence_of_broad_answer_scene_cross_with_exact_zero_norm"
    )
    assert solver["all_constraint_directions_may_be_zero_and_inactive"] is True
    assert solver["minimum_active_constraint_count"] == 1
    assert solver["zero_constraint_policy"] == (
        "record_inactive_and_first_order_satisfied_without_normalization"
    )
    assert solver["standalone_side_is_not_a_constraint"] is True
    assert solver["standalone_side_may_be_zero"] is True
    assert solver["scene_may_be_zero_and_explicitly_inactive"] is True
    assert solver["cross_may_be_zero_and_explicitly_inactive"] is True
    assert solver["inactive_constraints_must_be_persisted_with_norm_and_reason"] is True
    assert solver["raw_total_must_be_finite"] is True
    assert solver["raw_total_norm_minimum_exclusive"] == 1e-12


def test_v41_projection_contract_has_no_stale_fixed_16_candidate_text(
    terminal: dict,
) -> None:
    solver = terminal["conditional_successor_authorization"][
        "projected_gradient_solver"
    ]
    assert solver["optimization_problem"] == (
        "minimize 0.5*l2(d-g_raw)^2 subject to u_i dot d >= beta for every "
        "active nonzero constraint direction in canonical filtered order"
    )
    assert (
        "all_2**active_constraint_count_candidate_feasibility_records"
        in solver["persist_every_microstep"]
    )
    assert "all_16_candidate_feasibility_records" not in json.dumps(solver)


def test_v41_projection_has_replay_cast_and_clip_safety(terminal: dict) -> None:
    solver = terminal["conditional_successor_authorization"][
        "projected_gradient_solver"
    ]
    determinism = solver["determinism"]
    assert determinism["solve_twice_from_independent_cpu_float64_clones"] is True
    assert determinism["selected_mask_must_match"] is True
    assert determinism["lambdas_must_be_bit_exact"] is True
    assert determinism["projected_direction_must_be_bit_exact"] is True
    cpu = solver["cpu_solution_safety"]
    assert cpu["all_active_constraint_dots_at_least_beta"] is True
    assert cpu["inactive_exact_zero_directions_recorded_satisfied"] is True
    assert cpu["projected_to_raw_cosine_minimum"] == 0.95
    assert cpu["correction_ratio_maximum"] == 0.25
    cast = solver["device_cast_safety"]
    assert cast["normalized_constraint_margin_minimum"] == "beta/2"
    assert cast["all_active_dots_and_cosines_finite_and_strictly_positive"] is True
    assert cast["inactive_exact_zero_directions_remain_recorded_satisfied"] is True
    clip = solver["scalar_clip_safety"]
    assert clip["clip_norm"] == 1.0
    assert clip["single_global_scalar_over_lora_b"] is True
    assert clip["projected_to_clipped_cosine_minimum"] == 0.9999999
    assert clip["inactive_exact_zero_directions_remain_recorded_satisfied"] is True


def test_v41_must_replay_exact_pre_update3_target_hash(terminal: dict) -> None:
    replay = terminal["conditional_successor_authorization"][
        "transient_replay_gate"
    ]
    assert replay["required_before_optimizer_step_three"] is True
    assert replay["exact_target_hash_after_replayed_steps_one_and_two"] == (
        "8c50aa3a5975f450c3c95fb00dbf077a33285bf22ac3208f5d745cd617bd8d48"
    )
    assert replay["source_failure_artifact_sha256"] == (
        "0136edab8669346e4c24163650659608a7bf728fb8244ad81366a8efb5fa1f61"
    )
    assert replay["fail_before_step_three_if_hash_differs"] is True


def test_v41_uses_fresh_stateless_sgd_and_unchanged_schedule_gates(
    terminal: dict,
) -> None:
    authorization = terminal["conditional_successor_authorization"]
    optimizer = authorization["optimizer"]
    assert optimizer == {
        "implementation": "torch.optim.SGD",
        "fresh": True,
        "learning_rate": 0.003,
        "momentum": 0.0,
        "dampening": 0.0,
        "weight_decay": 0.0,
        "nesterov": False,
        "foreach": False,
        "fused": False,
        "adam_or_adamw_authorized": False,
        "resume_only_self_hashed_v41_sgd_state": True,
    }
    schedule = authorization["schedule"]
    assert schedule["maximum_optimizer_step"] == 41
    assert schedule["saved_optimizer_steps"] == [0, 8, 16, 24, 32, 40, 41]
    assert schedule["diagnostic_steps"] == [0, 8, 16, 41]
    assert schedule["full_schedule_sha256"] == (
        "2e7a2a8136c968739a8aaaa1138be87bec322fefa0fb340e6fb9bc6e07278bea"
    )
    gates = authorization["hard_train_only_gates"]
    assert gates["unchanged_from_v38_v40"] is True
    assert gates["update8"]["priority_side_deficit_minimum_improvement"] == 0.5
    assert gates["update16"]["priority_side_deficit_minimum_improvement"] == 3.12
    assert gates["update41"]["priority_side_deficit_minimum_improvement"] == 6.24


def test_v41_is_train_only_output_isolated_and_fail_closed(terminal: dict) -> None:
    authorization = terminal["conditional_successor_authorization"]
    isolation = authorization["stop_and_isolation"]
    assert isolation["authorized_output_root"].endswith(
        "gemma4_v41_diverse28_projected_gradient_l14_query"
    )
    assert isolation["v40_checkpoint_root_write_authorized"] is False
    assert isolation["stop_before_mutation_on_projection_or_attestation_failure"] is True
    assert isolation["stop_at_failed_update8_gate"] is True
    assert isolation["stop_at_failed_update16_gate"] is True
    assert isolation["stop_at_failed_update41_gate"] is True
    assert authorization["selector_execution_authorized"] is False
    assert authorization["chat_or_runtime_promotion_authorized"] is False
    assert terminal["validation_access_authorized"] is False
    assert terminal["oracle_access_authorized"] is False
    assert terminal["final_test_access_authorized"] is False


def test_terminal_process_loaded_no_environment_or_optimizer(terminal: dict) -> None:
    access = terminal["terminal_process_access_audit"]
    assert access["gemma_loaded"] is False
    assert access["qa_loaded"] is False
    assert access["maps_loaded"] is False
    assert access["validation_loaded"] is False
    assert access["oracle_loaded"] is False
    assert access["final_test_loaded"] is False
    assert access["optimizer_deserialized"] is False
    assert access["adapter_access"] == "bytes_hashed_only"
    assert access["loaded_file_count"] == 10


def test_protected_artifact_remains_exact(terminal: dict) -> None:
    protected = terminal["input_integrity"]["protected_artifact"]
    assert protected["sha256"] == (
        "c0086f66edbb8854a7938e09c57535bfd47100adbaf3b3c95eeb4b08014ce2f8"
    )
    assert protected["access"] == "bytes_hashed_only"
    assert protected["unchanged"] is True


def test_persisted_terminal_report_is_exact_replay() -> None:
    path = Path("reports/gemma4/metrics/v40_update3_terminal_gate.json")
    if not path.is_file():
        pytest.skip("V40 terminal report has not been materialized")
    assert json.loads(path.read_text(encoding="utf-8")) == audit_v40_update3_failure()


def test_terminal_rejects_changed_config_bytes(tmp_path: Path) -> None:
    source = Path("configs/experiments/gemma4_diverse28_cross_preserving_v40.yaml")
    changed = tmp_path / source.name
    changed.write_bytes(source.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="V40 config bytes changed"):
        audit_v40_update3_failure(config_path=changed)


def test_terminal_rejects_changed_trainer_bytes(tmp_path: Path) -> None:
    source = Path("src/semantic_3d_chat/training/train_cross_preserving_v40.py")
    changed = tmp_path / source.name
    changed.write_bytes(source.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="V40 trainer bytes changed"):
        audit_v40_update3_failure(trainer_path=changed)
