from __future__ import annotations

import json
from pathlib import Path

import pytest

from semantic_3d_chat.evaluation import v45_retention_repair_terminal_gate as gate


@pytest.fixture(scope="module")
def report() -> dict[str, object]:
    return gate.audit_v45_retention_repair()


def test_terminal_replays_exact_negative_v45(report: dict[str, object]) -> None:
    assert report["passed"] is True
    assert report["terminal_conclusion"] == (
        "update4_train_only_gate_failed_stop_is_final"
    )
    replay = report["update4_gate_replay"]
    assert replay["passed"] is False
    assert replay["failed_checks"] == [
        "both_lost_side_margins_strictly_positive",
        "complete_physical_pair_id_coverage_at_least_4",
        "teacher_complete_units_at_least_9",
        "teacher_positive_sides_at_least_34",
    ]
    assert replay["complete_units"] == 8
    assert replay["positive_sides"] == 32
    assert replay["cross_prefix_complete_units"] == 17
    assert replay["complete_physical_pair_coverage"] == 3
    assert replay["complete_units_by_family"] == {
        "book_support": 0,
        "mirror_lr": 2,
        "picture_support": 0,
    }
    assert replay["cross_prefix_complete_units_by_family"] == {
        "book_support": 1,
        "mirror_lr": 4,
        "picture_support": 3,
    }
    assert replay["lost_side_evidence"] == [
        {
            "pair_id": "pair_000006",
            "question_key": "cfq_5c84a2c27d2be251",
            "side_index": 0,
            "margin": -0.0625,
            "strictly_positive": False,
        },
        {
            "pair_id": "pair_000016",
            "question_key": "cfq_699675ceeaf65406",
            "side_index": 1,
            "margin": 0.24999994039535522,
            "strictly_positive": True,
        },
    ]
    assert report["execution_conclusion"]["update_five_executed"] is False


def test_exact_checkpoint_and_tensor_envelopes_are_authenticated(
    report: dict[str, object],
) -> None:
    inventory = report["checkpoint_inventory"]
    assert inventory["root_entries"] == ["update_000", "update_002", "update_004"]
    assert inventory["update_006_absent"] is True
    assert inventory["update_008_absent"] is True
    assert inventory["no_checkpoint_after_failed_update_four"] is True
    tensors = report["tensor_transition"]
    assert tensors["tensor_count_each_checkpoint"] == 179
    assert tensors["authorized_parameter_count"] == 415_744
    assert tensors["state_sha256"] == gate._STATE_HASHES
    assert tensors["update_zero_adapter_bytes_equal_v44_source"] is True
    assert tensors["only_three_authorized_tensors_changed"] is True
    assert tensors["frozen_state_bit_exact_through_update_four"] is True
    assert sorted(tensors["changed_tensor_names"]) == ["update_002", "update_004"]
    assert all(
        names == sorted(gate._PARAMETER_NAMES)
        for names in tensors["changed_tensor_names"].values()
    )
    runtime = report["runtime_metadata_audit"]
    assert sorted(runtime) == ["update_000", "update_002", "update_004"]
    assert all(value["sanitized_runtime_exact"] for value in runtime.values())


def test_v44_source_and_protected_artifact_are_pinned(
    report: dict[str, object],
) -> None:
    source = report["v44_source_authentication"]
    assert source["terminal_sha256"] == gate._PINNED_INPUTS[str(gate.V44_TERMINAL)]
    assert source["source_file_sha256"] == gate._V44_SOURCE_FILES
    assert source["source_full_tensor_state_sha256"] == gate._STATE_HASHES[
        "update_000"
    ]["full"]
    assert source["source_optimizer_bytes_authenticated_but_not_deserialized"] is True
    assert report["input_integrity"]["file_sha256"][str(gate.PROTECTED_REPORT)] == (
        "c0086f66edbb8854a7938e09c57535bfd47100adbaf3b3c95eeb4b08014ce2f8"
    )
    assert report["terminal_process_access_audit"]["protected_report_access"] == (
        "bytes_hashed_only"
    )


def test_history_and_access_boundary_are_exact(report: dict[str, object]) -> None:
    history = report["history_audit"]
    assert history["optimizer_updates_executed"] == [1, 2, 3, 4]
    assert history["checkpoint_steps_persisted"] == [0, 2, 4]
    assert history["history_prefixes_bit_exact"] is True
    assert history["history_frozen_hash_exact_every_step"] is True
    provenance = report["v45_provenance"]
    assert provenance["exact_train_scene_count"] == 16
    assert provenance["train_question_count"] == 384
    assert provenance["train_changed_pair_unit_count"] == 25
    assert provenance["source_optimizer_files_or_states_loaded"] is False
    assert provenance["validation_oracle_and_final_access"] is False
    process = report["terminal_process_access_audit"]
    assert process["optimizer_files_bytes_hashed_only"] is True
    assert process["optimizer_deserialized"] is False
    for field in (
        "gemma_loaded",
        "qa_loaded",
        "maps_loaded",
        "validation_loaded",
        "oracle_loaded",
        "final_test_loaded",
        "optimizer_step_executed",
    ):
        assert process[field] is False


def test_v46_authorization_is_exact_report_only_and_train_only(
    report: dict[str, object],
) -> None:
    auth = report["conditional_successor_authorization"]
    assert auth["authorization_id"] == (
        "v46_train_only_checkpoint_gradient_diagnostic"
    )
    assert auth["authorized_script"] == str(gate.V46_DIAGNOSTIC)
    assert auth["authorized_test"] == str(gate.V46_TEST)
    assert auth["authorized_report"] == str(gate.V46_OUTPUT)
    assert auth["explicit_terminal_sha256_cli_required"] is True
    assert auth["implementation_integrity"] == {
        "script_sha256": gate._PINNED_INPUTS[str(gate.V46_DIAGNOSTIC)],
        "test_sha256": gate._PINNED_INPUTS[str(gate.V46_TEST)],
        "config_sha256": gate._PINNED_INPUTS[str(gate.DEFAULT_CONFIG)],
    }
    assert auth["invocation_contract"] == {
        "terminal_path": str(gate.DEFAULT_OUTPUT),
        "required_cli_argument": "--expected-v45-terminal-sha256",
        "expected_value": "sha256_of_materialized_v45_terminal_passed_explicitly",
        "v46_must_not_embed_terminal_sha256": True,
        "v46_must_authenticate_terminal_bytes_and_exact_authorization": True,
    }
    assert auth["source"]["checkpoint"] == str(
        gate.DEFAULT_CHECKPOINT_ROOT / "update_004"
    )
    assert auth["source"]["full_tensor_state_sha256"] == gate._STATE_HASHES[
        "update_004"
    ]["full"]
    data = auth["fixed_data_boundary"]
    assert data["scene_count"] == 16
    assert data["train_question_count"] == 384
    assert data["changed_pair_unit_count"] == 25
    assert data["broad_nll_row_count"] == 48
    assert data["blocking_file_access_audit_required"] is True
    specs = auth["measurements"]["isolated_side_gradient_specs"]
    assert [(value["question_key"], value["side_index"]) for value in specs] == [
        ("cfq_5c84a2c27d2be251", 0),
        ("cfq_699675ceeaf65406", 1),
        ("cfq_0a79d507273195ef", 0),
    ]
    assert specs[0]["role"] == "g5_candidate_direction_source"
    assert all("diagnostic_only" in value["role"] for value in specs[1:])


def test_v46_sign_line_is_fixed_and_cannot_select_or_write(
    report: dict[str, object],
) -> None:
    auth = report["conditional_successor_authorization"]
    line = auth["fresh_adam_sign_line"]
    assert line["direction_ids"] == [
        "g5_scene_sign",
        "g5_query_sign",
        "g5_both_sign",
    ]
    assert line["candidate_formula"] == "float32_P0-alpha*lr_group*sign(g5)"
    assert line["scene_readout_learning_rate"] == 1e-5
    assert line["query_learning_rate"] == 8e-6
    assert line["alpha_grid"] == [0.125, 0.25, 0.5, 1.0, 2.0]
    assert line["exact_candidate_count"] == 15
    assert line["exact_u4_restoration_before_and_after_every_probe"] is True
    assert line["adaptive_direction_or_scalar_selection"] is False
    assert line["diagnostic_gradient_q699_and_q0a79_used_as_directions"] is False
    assert all(auth["forbidden_actions"].values())
    scope = auth["scope"]
    assert scope["read_only_except_single_report"] is True
    assert scope["report_only_output"] is True
    assert scope["no_candidate_is_authorized_by_this_diagnostic"] is True
    assert all(
        scope[field] is False
        for field in (
            "validation_access_authorized",
            "oracle_access_authorized",
            "final_test_access_authorized",
            "selector_execution_authorized",
            "runtime_promotion_authorized",
            "chat_promotion_authorized",
        )
    )


def test_materialized_terminal_replays_exactly() -> None:
    path = gate._resolve(gate.DEFAULT_OUTPUT)
    if not path.exists():
        pytest.skip("terminal not materialized yet")
    assert json.loads(path.read_text(encoding="utf-8")) == (
        gate.audit_v45_retention_repair()
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
    monkeypatch.setitem(gate._PINNED_INPUTS, str(gate.V45_TRAINER), "0" * 64)
    with pytest.raises(ValueError, match="bytes changed"):
        gate.audit_v45_retention_repair()
