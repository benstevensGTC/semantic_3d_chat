from __future__ import annotations

import json
from pathlib import Path

import pytest

from semantic_3d_chat.evaluation import v47_book_continuation_terminal_gate as gate
from semantic_3d_chat.evaluation import v48_v47_u4_dual_margin_screen as v48

_V48_SCRIPT_SHA256 = "c132e66568b626658315dc90c3638e52677d299900d863270ddfced07c580611"
_V48_TEST_SHA256 = "e522471a3ca88ea363bf59c2de0bb5f2a9d1cee628a8e874264fa2d2db52e31a"


@pytest.fixture(scope="module")
def scaffold() -> dict[str, object]:
    return gate.build_terminal_scaffold()


@pytest.fixture(scope="module")
def terminal() -> dict[str, object]:
    return gate.build_terminal_report(_V48_SCRIPT_SHA256, _V48_TEST_SHA256)


def test_placeholder_scaffold_authenticates_v47_but_cannot_materialize(
    scaffold: dict[str, object],
) -> None:
    assert scaffold["passed"] is True
    assert scaffold["artifact"] == "v47_book_continuation_terminal_gate_scaffold"
    assert scaffold["terminal_materialization_authorized"] is False
    assert scaffold["v47_final_train_only_gate_passed"] is False
    assert scaffold["only_exact_successor_authorized"] is None
    reference = scaffold["v48_implementation_reference"]
    assert isinstance(reference, dict)
    assert reference["status"] == "pending_stable_v48_implementation_bytes"
    assert reference["implementation_files_opened"] is False
    assert reference["implementation_authenticated"] is False


def test_placeholder_mode_does_not_require_v48_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gate, "V48_DIAGNOSTIC", tmp_path / "missing_screen.py")
    monkeypatch.setattr(gate, "V48_TEST", tmp_path / "missing_test.py")
    result = gate.build_terminal_scaffold()
    reference = result["v48_implementation_reference"]
    assert isinstance(reference, dict)
    assert reference["implementation_files_opened"] is False


def test_exact_inputs_predecessor_and_protected_v11_are_authenticated(
    scaffold: dict[str, object],
) -> None:
    inputs = scaffold["input_integrity"]
    assert isinstance(inputs, dict)
    assert inputs["file_sha256"][str(gate.PROTECTED_REPORT)] == (
        "c0086f66edbb8854a7938e09c57535bfd47100adbaf3b3c95eeb4b08014ce2f8"
    )
    assert inputs["protected_v11_access"] == "bytes_hashed_only"
    assert inputs["normalized_config_hash"] == "9c79d3cb5af0"
    predecessor = scaffold["v46_predecessor_authentication"]
    assert isinstance(predecessor, dict)
    assert predecessor["terminal_sha256"] == (
        "de66c9844786c8718399c75162e0b13313b778c1b7d5fa7edcd4133d4d31b60d"
    )
    assert predecessor["exact_v47_authorization_authenticated"] is True
    assert all(predecessor["authorization_checks"].values())


def test_all_checkpoint_files_inventory_and_optimizer_steps_are_sealed(
    scaffold: dict[str, object],
) -> None:
    inventory = scaffold["checkpoint_inventory"]
    assert isinstance(inventory, dict)
    assert inventory["root_entries"] == ["update_000", "update_002", "update_004"]
    assert inventory["checkpoint_steps_persisted"] == [0, 2, 4]
    assert inventory["manifest_sha256"] == (
        "0c457e644a41e4e6af0e31fb64a5bb60655a46f9828e664aae3119256d6e8b64"
    )
    assert inventory["file_sha256"] == gate._CHECKPOINT_FILES
    assert inventory["optimizer_files_bytes_hashed_only"] is True
    assert inventory["optimizer_state_deserialized"] is False
    assert inventory["update_006_absent"] is True
    assert inventory["update_008_absent"] is True


def test_exact_tensor_hashes_and_frozen_integrity_are_recomputed(
    scaffold: dict[str, object],
) -> None:
    tensors = scaffold["tensor_transition"]
    assert isinstance(tensors, dict)
    assert tensors["tensor_count_each_checkpoint"] == 179
    assert tensors["authorized_parameter_count"] == 415_744
    assert tensors["state_sha256"] == gate._STATE_HASHES
    assert tensors["state_sha256"]["update_004"] == {
        "full": "adfc0400d1a3bb49b278cd3012ab571d01465f2380881f986c085a25474276e5",
        "authorized": ("a23de4988774a966c0d7aac378ede5d15a3fa1d96093c5039f181a62b0bb09b0"),
        "frozen": "31cb215cf0d1623886d4a79203c501912e66537021da53dd711124abdc2e36fa",
    }
    assert tensors["only_three_authorized_tensors_changed"] is True
    assert tensors["frozen_state_bit_exact_through_update_four"] is True
    assert tensors["all_tensors_finite"] is True


def test_runtime_metadata_is_exact_sanitization_at_every_checkpoint(
    scaffold: dict[str, object],
) -> None:
    runtime = scaffold["runtime_metadata_audit"]
    assert isinstance(runtime, dict)
    assert [runtime[label]["optimizer_step"] for label in sorted(runtime)] == [0, 2, 4]
    assert all(value["sanitized_runtime_exact"] for value in runtime.values())
    assert all(value["training_history_qa_and_gate_fields_absent"] for value in runtime.values())


def test_history_prefixes_schedule_and_every_update_hash_are_exact(
    scaffold: dict[str, object],
) -> None:
    history = scaffold["history_audit"]
    assert isinstance(history, dict)
    assert history["optimizer_updates_executed"] == [1, 2, 3, 4]
    assert history["checkpoint_steps_persisted"] == [0, 2, 4]
    assert history["history_lengths"] == [1, 3, 5]
    assert history["history_prefixes_bit_exact"] is True
    assert history["history_frozen_hash_exact_every_step"] is True
    assert history["authorized_state_sha256_by_update"] == {
        str(step): value for step, value in gate._AUTHORIZED_HISTORY_HASHES.items()
    }
    assert history["query_state_sha256_by_update"] == {
        str(step): value for step, value in gate._QUERY_HASHES.items()
    }
    assert history["scene_readout_state_sha256_by_update"] == {
        str(step): value for step, value in gate._SCENE_READOUT_HASHES.items()
    }
    assert history["prefix_trust_rms_by_update"] == {
        str(step): value for step, value in gate._PREFIX_TRUST_RMS.items()
    }
    assert history["fixed_broad_question_ids"] == list(gate._BROAD_QUESTION_IDS)
    assert history["update2_integrity_gate_passed"] is True


def test_negative_teacher_gate_is_replayed_exactly(scaffold: dict[str, object]) -> None:
    replay = scaffold["update4_gate_replay"]
    assert isinstance(replay, dict)
    teacher = replay["teacher_forced"]
    assert teacher["complete_units"] == 8
    assert teacher["teacher_complete_threshold"] == 10
    assert teacher["positive_sides"] == 33
    assert teacher["teacher_positive_threshold"] == 35
    assert teacher["cross_prefix_complete_units"] == 17
    assert teacher["teacher_cross_threshold"] == 17
    assert teacher["complete_physical_pair_coverage"] == 4
    assert teacher["physical_pair_threshold"] == 5
    assert teacher["complete_units_by_family"] == {
        "book_support": 0,
        "mirror_lr": 1,
        "picture_support": 0,
    }
    assert teacher["cross_prefix_complete_units_by_family"] == {
        "book_support": 1,
        "mirror_lr": 4,
        "picture_support": 2,
    }
    assert replay["passed"] is False


def test_continuous_and_greedy_gate_results_are_replayed_exactly(
    scaffold: dict[str, object],
) -> None:
    replay = scaffold["update4_gate_replay"]
    assert isinstance(replay, dict)
    assert replay["priority_side_deficit_improvement"] == 0.7275158166885376
    assert replay["checks"]["priority_teacher_deficit_improved_at_least_0_5_vs_original_v41_u0"]
    assert replay["update4_broad_nll"] == 2.9172145972649255
    assert replay["checks"]["broad_nll_at_most_authorized_maximum"]
    assert replay["update4_source_prefix_trust_rms"] == 0.001376520493067801
    assert replay["checks"]["source_prefix_trust_rms_at_most_0_002"]
    greedy = replay["greedy"]
    assert greedy["complete_units"] == 4
    assert greedy["complete_units_threshold"] == 5
    assert greedy["broad_exact_correct"] == 23
    assert greedy["broad_row_count"] == 48
    assert replay["checks"]["broad_greedy_exact_correct_at_least_23_of_48"]
    assert replay["checks"]["train_greedy_complete_units_at_least_5"] is False


def test_exact_failure_set_and_lost_side_evidence_are_sealed(
    scaffold: dict[str, object],
) -> None:
    replay = scaffold["update4_gate_replay"]
    assert isinstance(replay, dict)
    assert replay["failed_checks"] == [
        "book_complete_units_at_least_1",
        "both_lost_side_margins_remain_strictly_positive",
        "complete_physical_pair_id_coverage_at_least_5",
        "mirror_complete_units_at_least_2",
        "teacher_complete_units_at_least_10",
        "teacher_positive_sides_at_least_35",
        "train_greedy_complete_units_at_least_5",
    ]
    assert replay["lost_side_evidence"] == [
        {
            "pair_id": "pair_000006",
            "question_key": "cfq_5c84a2c27d2be251",
            "side_index": 0,
            "margin": 0.0625,
            "strictly_positive": True,
        },
        {
            "pair_id": "pair_000016",
            "question_key": "cfq_699675ceeaf65406",
            "side_index": 1,
            "margin": 0.0,
            "strictly_positive": False,
        },
    ]


def test_no_restricted_access_or_promotion_is_authorized(
    scaffold: dict[str, object],
) -> None:
    for field in (
        "arbitrary_training_authorized",
        "resume_v47_training_authorized",
        "candidate_checkpoint_write_authorized",
        "validation_access_authorized",
        "oracle_access_authorized",
        "final_test_access_authorized",
        "selector_execution_authorized",
        "runtime_promotion_authorized",
        "chat_promotion_authorized",
        "embodied_promotion_authorized",
    ):
        assert scaffold[field] is False
    access = scaffold["terminal_process_access_audit"]
    assert isinstance(access, dict)
    assert access["gemma_loaded"] is False
    assert access["qa_loaded"] is False
    assert access["maps_loaded"] is False
    assert access["validation_loaded"] is False
    assert access["oracle_loaded"] is False
    assert access["final_test_loaded"] is False
    assert access["optimizer_deserialized"] is False
    assert access["optimizer_step_executed"] is False


def test_real_v48_hashes_produce_exact_authorization_accepted_by_v48(
    terminal: dict[str, object],
) -> None:
    assert terminal["terminal_materialization_authorized"] is True
    assert terminal["only_exact_successor_authorized"] == v48._AUTHORIZATION_ID
    assert terminal["v47_final_train_only_gate_passed"] is False
    authorization = terminal["conditional_successor_authorization"]
    assert isinstance(authorization, dict)
    assert authorization["implementation_integrity"] == {
        "script_sha256": _V48_SCRIPT_SHA256,
        "test_sha256": _V48_TEST_SHA256,
        "config_sha256": gate._PINNED_INPUTS[str(gate.DEFAULT_CONFIG)],
    }
    assert authorization["measurements"]["exact_three_torch_autograd_grad_probes_authorized"]
    assert authorization["forbidden_actions"]["backward_or_parameter_gradient_accumulation"]
    assert authorization["forbidden_actions"]["gradient_outside_exact_three_specs"]
    checks = v48._validate_authorization(terminal, authorization)
    assert all(checks.values())


def test_placeholder_mixed_or_malformed_hashes_fail_closed() -> None:
    with pytest.raises(ValueError, match="explicit stable V48"):
        gate.build_terminal_report(
            gate.IMPLEMENTATION_SHA256_PLACEHOLDER,
            gate.IMPLEMENTATION_SHA256_PLACEHOLDER,
        )
    with pytest.raises(ValueError, match="both be placeholders or real"):
        gate.build_terminal_scaffold(
            _V48_SCRIPT_SHA256,
            gate.IMPLEMENTATION_SHA256_PLACEHOLDER,
        )
    with pytest.raises(ValueError, match="64 lowercase"):
        gate.build_terminal_scaffold("not-a-hash", _V48_TEST_SHA256)


def test_tampered_v47_or_v48_bytes_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(gate._PINNED_INPUTS, str(gate.V47_TRAINER), "0" * 64)
    with pytest.raises(ValueError, match="bytes changed"):
        gate.build_terminal_scaffold()


def test_terminal_writer_is_pinned_atomic_and_one_shot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "terminal.json"
    monkeypatch.setattr(gate, "DEFAULT_OUTPUT", output)
    report = gate.write_report(
        output,
        expected_v48_diagnostic_sha256=_V48_SCRIPT_SHA256,
        expected_v48_test_sha256=_V48_TEST_SHA256,
    )
    assert json.loads(output.read_text(encoding="utf-8")) == report
    with pytest.raises(FileExistsError, match="one-shot"):
        gate.write_report(
            output,
            expected_v48_diagnostic_sha256=_V48_SCRIPT_SHA256,
            expected_v48_test_sha256=_V48_TEST_SHA256,
        )
    with pytest.raises(ValueError, match="pinned"):
        gate.write_report(
            tmp_path / "other.json",
            expected_v48_diagnostic_sha256=_V48_SCRIPT_SHA256,
            expected_v48_test_sha256=_V48_TEST_SHA256,
        )


def test_materialized_terminal_replays_exactly_if_present() -> None:
    path = gate._resolve(gate.DEFAULT_OUTPUT)
    if not path.is_file():
        pytest.skip("V47 terminal is materialized only after focused tests pass")
    assert json.loads(path.read_text(encoding="utf-8")) == gate.build_terminal_report(
        _V48_SCRIPT_SHA256,
        _V48_TEST_SHA256,
    )
