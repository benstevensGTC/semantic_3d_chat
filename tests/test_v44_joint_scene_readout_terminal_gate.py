from __future__ import annotations

import json
from pathlib import Path

import pytest

from semantic_3d_chat.evaluation import v44_joint_scene_readout_terminal_gate as gate


@pytest.fixture(scope="module")
def report() -> dict[str, object]:
    return gate.audit_v44_joint_scene_readout()


def test_terminal_replays_exact_negative_v44(report: dict[str, object]) -> None:
    assert report["passed"] is True
    assert report["terminal_conclusion"] == (
        "update8_train_only_gate_failed_stop_is_final"
    )
    replay = report["update8_gate_replay"]
    assert replay["passed"] is False
    assert replay["failed_checks"] == [
        "teacher_complete_units_at_least_9",
        "teacher_positive_sides_at_least_34",
    ]
    assert replay["complete_units"] == 7
    assert replay["positive_sides"] == 32
    assert replay["cross_prefix_complete_units"] == 18
    assert replay["complete_physical_pair_coverage"] == 3
    assert replay["update16_gate_absent"] is True
    assert report["execution_conclusion"]["update_nine_executed"] is False


def test_exact_checkpoint_and_runtime_envelopes_are_authenticated(
    report: dict[str, object],
) -> None:
    inventory = report["checkpoint_inventory"]
    tensors = report["tensor_transition"]
    assert inventory["root_entries"] == ["update_000", "update_004", "update_008"]
    assert inventory["no_update_after_eight_persisted"] is True
    assert tensors["tensor_count_each_checkpoint"] == 179
    assert tensors["authorized_parameter_count"] == 415_744
    assert tensors["only_three_authorized_tensors_changed"] is True
    assert tensors["frozen_state_bit_exact_through_update_eight"] is True
    assert tensors["state_sha256"] == gate._STATE_HASHES
    runtime = report["runtime_metadata_audit"]
    assert sorted(runtime) == ["update_000", "update_004", "update_008"]
    assert all(value["sanitized_runtime_exact"] for value in runtime.values())


def test_v45_authorization_is_exact_and_train_only(
    report: dict[str, object],
) -> None:
    auth = report["conditional_successor_authorization"]
    assert auth["authorization_id"] == "v45_train_only_retention_repair_pilot"
    assert auth["authorized_config"] == str(gate.V45_CONFIG)
    assert auth["authorized_output_root"] == str(gate.V45_OUTPUT)
    assert auth["source_checkpoint"] == str(
        gate.DEFAULT_CHECKPOINT_ROOT / "update_008"
    )
    assert auth["source_full_tensor_state_sha256"] == gate._STATE_HASHES[
        "update_008"
    ]["full"]
    assert auth["trainable_surface"]["parameter_names"] == list(
        gate._PARAMETER_NAMES
    )
    assert auth["trainable_surface"]["total_parameter_count"] == 415_744
    assert auth["optimizer"] == {
        "implementation": "fresh_torch_adamw_two_groups",
        "source_optimizer_loaded": False,
        "scene_readout_learning_rate": 1e-5,
        "query_learning_rate": 8e-6,
        "weight_decay": 0.0,
        "foreach": False,
        "fused": False,
        "per_group_gradient_clip_norm": 1.0,
    }
    assert auth["schedule"]["checkpoint_steps"] == [0, 2, 4, 6, 8]
    assert auth["target_schedule"] == gate._TARGET_SCHEDULE
    assert auth["source_optimizer_policy"] == {
        "source_optimizer_file_present_and_authenticated": True,
        "source_optimizer_file_open_authorized_by_v45": False,
        "source_optimizer_deserialization_authorized": False,
        "source_optimizer_state_loading_authorized": False,
        "fresh_optimizer_required": True,
    }
    assert all(
        auth["scope"][field] is False
        for field in (
            "validation_access_authorized",
            "oracle_access_authorized",
            "final_test_access_authorized",
            "selector_execution_authorized",
            "runtime_promotion_authorized",
            "chat_promotion_authorized",
        )
    )


def test_retention_controls_and_gates_are_machine_readable(
    report: dict[str, object],
) -> None:
    auth = report["conditional_successor_authorization"]
    retention = auth["retention_control"]
    assert retention["fragile_side_constraint_count"] == 8
    assert retention["book_cross_constraint_count"] == 4
    assert retention["fragile_side_floor"] == 0.125
    assert retention["book_cross_floor"] == 0.025
    assert retention["two_normalized_means_summed_before_single_weight"] is True
    assert auth["objective"]["retention_weight"] == 8.0
    assert auth["objective"]["retention_formula"] == (
        "mean(relu(0.125-fragile_side_margin),8)"
        "+mean(relu(0.025-book_cross_prefix_margin),4)"
    )
    assert [row["question_key"] for row in retention["lost_side_gate4_constraints"]] == [
        "cfq_5c84a2c27d2be251",
        "cfq_699675ceeaf65406",
    ]
    gate4 = auth["update4_gate"]
    gate8 = auth["update8_gate"]
    assert gate4["complete_units_minimum"] == 9
    assert gate4["positive_sides_minimum"] == 34
    assert gate4["complete_physical_pair_id_coverage_minimum"] == 4
    assert gate4["broad_nll_maximum"] == 2.9213306349515915
    assert gate8["require_recorded_update4_gate_passed"] is True
    assert gate8["complete_units_minimum"] == 10
    assert gate8["positive_sides_minimum"] == 35
    assert gate8["complete_physical_pair_id_coverage_minimum"] == 5
    assert gate8["mirror_complete_units_minimum"] == 2
    assert gate8["book_complete_units_minimum"] == 1
    assert gate8["book_cross_prefix_complete_units_minimum"] == 1
    assert gate8["train_greedy_complete_units_minimum"] == 5
    assert gate8["broad_greedy_exact_correct_minimum"] == 23
    assert gate8["broad_greedy_row_count_exact"] == 48
    assert gate8["u8_prefix_trust_rms_maximum"] == 0.002
    assert gate8["lost_side_margins_must_remain_strictly_positive"] == (
        gate4["lost_side_margins_must_both_be_strictly_positive"]
    )


def test_materialized_terminal_replays_exactly() -> None:
    path = gate._resolve(gate.DEFAULT_OUTPUT)
    if not path.exists():
        pytest.skip("terminal not materialized yet")
    assert json.loads(path.read_text(encoding="utf-8")) == (
        gate.audit_v44_joint_scene_readout()
    )


def test_write_is_one_shot_and_output_is_pinned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(ValueError, match="pinned"):
        gate.write_report(tmp_path / "other.json")
    existing = tmp_path / "terminal.json"
    existing.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(gate, "DEFAULT_OUTPUT", existing)
    with pytest.raises(FileExistsError, match="one-shot"):
        gate.write_report(existing)


def test_changed_pinned_input_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(gate._PINNED_INPUTS, str(gate.V44_TRAINER), "0" * 64)
    with pytest.raises(ValueError, match="bytes changed"):
        gate.audit_v44_joint_scene_readout()
