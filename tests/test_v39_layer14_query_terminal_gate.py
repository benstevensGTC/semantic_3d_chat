from __future__ import annotations

import json
from pathlib import Path

import pytest

from semantic_3d_chat.evaluation.v39_layer14_query_terminal_gate import (
    audit_v39_layer14_query_screen,
)


@pytest.fixture(scope="module")
def terminal() -> dict:
    return audit_v39_layer14_query_screen()


def test_terminal_authenticates_exact_v39_inputs(terminal: dict) -> None:
    assert terminal["passed"] is True
    assert terminal["artifact"] == "v39_layer14_query_terminal_gate"
    integrity = terminal["input_integrity"]
    assert integrity["screen_report"]["sha256"] == (
        "d10ee48738864bb8e1b8136d7f16a2176e6e46e8adea5a9faa05bf974eb4bdbe"
    )
    assert integrity["screen_module"]["sha256"] == (
        "49477f22908c7b9230a4f8f0862824b427967c48dfd181a7dca1a812ad365c5c"
    )
    assert integrity["screen_tests"]["sha256"] == (
        "bff3612e5433c39cb5429f6e13934291fe8080c62a7934ef4cb89c41b2633f6b"
    )
    assert integrity["v38_terminal_seal"]["sha256"] == (
        "1015949e802abccd562f7762cc01111818646527f3366aeaf01de3854bbe164a"
    )


def test_terminal_replays_v39_only_contract_failure(terminal: dict) -> None:
    failure = terminal["single_failure_replay"]
    assert failure["diagnostic_passed"] is False
    assert failure["false_contract_checks"] == [
        "all_predeclared_directional_compatibility_checks_passed"
    ]
    assert failure["failed_directional_pairs"] == [
        "proposed_training_aggregate__cross_prefix_maintenance_aggregate"
    ]
    assert failure["failed_cross_prefix_cosine"] == -0.30298788564923285
    assert failure["failed_cross_prefix_dot_product"] == -0.013504734244905985
    assert failure["all_other_predeclared_directions_passed"] is True


def test_terminal_solves_global_and_per_tensor_strict_intervals(
    terminal: dict,
) -> None:
    math_replay = terminal["cross_preserving_gradient_math"]
    surfaces = math_replay["surfaces"]
    expected = {
        "global": (4.834377059531332, 29.880385532751976),
        "lora_a": (12.566257500163564, 13.859227571889285),
        "lora_b": (3.545042356673292, 37.89459979731636),
    }
    for name, (lower, upper) in expected.items():
        interval = surfaces[name]["strict_feasible_interval"]
        assert interval["lower_exclusive"] == pytest.approx(lower, abs=1e-12)
        assert interval["upper_exclusive"] == pytest.approx(upper, abs=1e-12)
    joint = math_replay["joint_global_and_per_tensor_strict_feasible_interval"]
    assert joint == pytest.approx(
        {
            "lower_exclusive": 12.566257500163564,
            "upper_exclusive": 13.859227571889285,
        },
        abs=1e-12,
    )


def test_t6_weight28_is_rejected_and_t13_weight56_is_strictly_interior(
    terminal: dict,
) -> None:
    math_replay = terminal["cross_preserving_gradient_math"]
    rejected = math_replay["rejected_t6_control"]
    assert rejected["t"] == 6.0
    assert rejected["effective_cross_weight"] == 28.0
    assert rejected["lora_a_interval_contains_t"] is False
    assert rejected["lora_a_cross_prefix_dot_product"] < 0.0
    assert rejected["rejected"] is True
    assert math_replay["selected_t"] == 13.0
    assert math_replay["selected_effective_cross_weight"] == 56.0
    assert math_replay["selected_t_strictly_inside_joint_interval"] is True
    for surface in math_replay["surfaces"].values():
        assert surface["all_authorized_dots_and_cosines_strictly_positive"] is True
        for direction in surface["authorized_directional_compatibility"].values():
            assert direction["dot_product"] > 0.0
            assert direction["cosine"] > 0.0


def test_v40_authorizes_only_existing_lora_b_and_freezes_lora_a(
    terminal: dict,
) -> None:
    target = terminal["conditional_successor_authorization"]["target_surface"]
    assert target["parameter_names"] == [
        "lora_banks.extension_v28_stage_b_query.adapters.1.lora_b"
    ]
    assert target["parameter_shapes"] == [[4096, 4]]
    assert target["tensor_count"] == 1
    assert target["parameter_count"] == 16_384
    assert target["lora_b_only"] is True
    assert target["lora_a_learned_basis_frozen"] is True
    assert target["lora_a_write_authorized"] is False
    assert target["source_local_b_state_sha256"] == (
        "d0834cc588ee2a9edf08aabedfd01e0a6d2b01c6b6ae7e3a3d764eaddf58cc3e"
    )
    assert target["source_local_a_state_sha256"] == (
        "9f0ee5f9bbb9ec07bd42aaca1e0817be567a11c396c693e6412e5f2b08f37403"
    )
    assert target["source_full_checkpoint_key_b_state_sha256"] == (
        "1cdda782f0caf121c743d36d8b122e9480aa8300453c03872da37dfe81556799"
    )
    assert target["source_frozen_excluding_b_state_sha256"] == (
        "cec01bc088bb87c6bb44e0659eb03aa766f951ddeee706ca9a70edaa080dea5e"
    )
    assert target["source_frozen_excluding_b_tensor_count"] == 178
    assert target["source_frozen_excluding_b_parameter_count"] == 13_969_292


def test_v40_uses_direction_preserving_sgd_not_adam(terminal: dict) -> None:
    authorization = terminal["conditional_successor_authorization"]
    optimizer = authorization["optimizer"]
    assert optimizer == {
        "type": "SGD",
        "implementation": "torch.optim.SGD",
        "fresh": True,
        "learning_rate": 0.003,
        "momentum": 0.0,
        "dampening": 0.0,
        "nesterov": False,
        "weight_decay": 0.0,
        "gradient_clip_norm": 1.0,
        "clip_is_one_global_scalar_over_the_single_b_tensor": True,
        "direction_preserving_after_scalar_clip": True,
        "adam_or_adamw_authorized": False,
        "source_optimizer_state_loaded": False,
        "source_optimizer_file_opened": False,
        "trainable_parameter_count": 16_384,
        "resume_only_self_hashed_v40_sgd_state": True,
        "resume_optimizer_type_must_remain_exact": True,
    }
    objective = authorization["objective"]
    assert objective["cross_prefix_hinge_weight"] == 56.0
    assert objective["gradient_construction_t"] == 13.0
    assert objective["cross_prefix_weight_28_authorized"] is False


def test_v40_requires_raw_component_direction_guard_before_every_step(
    terminal: dict,
) -> None:
    guard = terminal["conditional_successor_authorization"][
        "per_microstep_raw_gradient_guard"
    ]
    assert guard["required_before_every_optimizer_step"] is True
    assert guard["component_gradient_api"] == "torch.autograd.grad"
    assert guard["backward_for_component_construction_authorized"] is False
    assert guard["weighted_components"] == {
        "broad": 1.0,
        "answer": 0.5,
        "side": 8.0,
        "cross": 56.0,
    }
    assert guard["total_gradient_formula"] == "broad + answer + side + cross"
    assert guard["scene_gradient_formula"] == "side + cross"
    assert guard["directions_checked_if_nonzero"] == [
        "broad",
        "answer",
        "scene",
        "cross",
    ]
    assert guard["require_strictly_positive_total_direction_dot"] is True
    assert guard["require_strictly_positive_total_direction_cosine"] is True
    assert guard["fail_stop_before_clip_or_step_on_any_guard_failure"] is True
    assert guard["scalar_global_clip_only_after_guard_passes"] is True
    assert guard["momentum_free_sgd_step_only_after_guard_passes"] is True


def test_v40_reuses_exact_v38_schedule_and_hard_gates(terminal: dict) -> None:
    authorization = terminal["conditional_successor_authorization"]
    schedule = authorization["schedule"]
    assert schedule["maximum_optimizer_step"] == 41
    assert schedule["saved_optimizer_steps"] == [0, 8, 16, 24, 32, 40, 41]
    assert schedule["per_unit_nll_diagnostics_required_at_steps"] == [0, 8, 16, 41]
    assert schedule["pair_schedule_sha256"] == (
        "4e26160c9a36e20ac894ff5e26604e6fad985ff5c84211282fd9807c9e738b3f"
    )
    assert schedule["full_schedule_sha256"] == (
        "2e7a2a8136c968739a8aaaa1138be87bec322fefa0fb340e6fb9bc6e07278bea"
    )
    gates = authorization["hard_train_only_gates"]
    assert gates["unchanged_from_exact_v38"] is True
    assert gates["update8"]["priority_side_deficit_minimum_improvement"] == 0.5
    assert gates["update16"]["priority_side_deficit_minimum_improvement"] == 3.12
    assert gates["update41"]["priority_side_deficit_minimum_improvement"] == 6.24
    stop = authorization["stop_protocol"]
    assert stop["stop_at_update8_if_gate_fails"] is True
    assert stop["stop_at_update16_if_gate_fails"] is True
    assert stop["stop_at_update41_if_gate_fails"] is True
    assert stop["no_gate_relaxation_authorized"] is True


def test_terminal_denies_environment_evaluation_and_promotion(terminal: dict) -> None:
    authorization = terminal["conditional_successor_authorization"]
    assert authorization["validation_access_authorized"] is False
    assert authorization["final_test_access_authorized"] is False
    assert authorization["oracle_access_authorized"] is False
    assert authorization["selector_execution_authorized"] is False
    assert authorization["chat_or_runtime_promotion_authorized"] is False
    access = terminal["terminal_process_access_audit"]
    assert access["loaded_file_count"] == 5
    assert access["gemma_loaded"] is False
    assert access["checkpoint_tensor_or_metadata_loaded"] is False
    assert access["optimizer_opened"] is False
    assert access["qa_loaded"] is False
    assert access["scene_maps_loaded"] is False
    assert access["validation_loaded"] is False
    assert access["oracle_loaded"] is False
    assert access["final_test_loaded"] is False


def test_terminal_protected_artifact_remains_exact(terminal: dict) -> None:
    protected = terminal["input_integrity"]["protected_artifact"]
    assert protected["sha256"] == (
        "c0086f66edbb8854a7938e09c57535bfd47100adbaf3b3c95eeb4b08014ce2f8"
    )
    assert protected["access"] == "bytes_hashed_only"
    assert protected["unchanged"] is True


def test_saved_terminal_report_is_exact_replay() -> None:
    path = Path("reports/gemma4/metrics/v39_layer14_query_terminal_gate.json")
    if not path.is_file():
        pytest.skip("terminal report has not been materialized yet")
    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert persisted == audit_v39_layer14_query_screen()


def test_terminal_rejects_changed_screen_bytes(tmp_path: Path) -> None:
    source = Path("reports/gemma4/metrics/v39_layer14_query_gradient_screen.json")
    changed = tmp_path / source.name
    changed.write_bytes(source.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="V39 gradient-screen report bytes changed"):
        audit_v39_layer14_query_screen(screen_report=changed)


def test_terminal_rejects_changed_v38_seal_bytes(tmp_path: Path) -> None:
    source = Path("reports/gemma4/metrics/v38_update8_terminal_gate.json")
    changed = tmp_path / source.name
    changed.write_bytes(source.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="V38 revision-2 terminal seal bytes changed"):
        audit_v39_layer14_query_screen(v38_terminal=changed)
