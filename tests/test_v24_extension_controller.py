from __future__ import annotations

import copy
from pathlib import Path

import pytest

from semantic_3d_chat.evaluation import v24_extension_controller as controller


def _controller_provenance() -> dict[str, object]:
    value = copy.deepcopy(controller.EXPECTED_TRAINING_SOURCE_PROVENANCE)
    value["head_commit"] = "c" * 40
    value["head_tree"] = "d" * 40
    return value


def _metrics(
    *,
    sides: int,
    units: int,
    minimum_candidate: float,
    minimum_full: float,
    mean_candidate: float = 1.0,
    mean_full: float = 1.0,
) -> dict[str, object]:
    return {
        "full_vocab_sides": sides,
        "full_vocab_units": units,
        "mean_candidate_margin": mean_candidate,
        "minimum_candidate_margin": minimum_candidate,
        "mean_full_vocab_margin": mean_full,
        "minimum_full_vocab_margin": minimum_full,
    }


def _candidate_row(epoch: int, color: dict[str, object], mirror: dict[str, object]) -> dict:
    return {
        "epoch": epoch,
        "optimizer_step": epoch,
        "cumulative_microsteps": epoch * controller.MICROSTEPS_PER_UPDATE,
        "metadata_path": f"metadata_{epoch}.json",
        "metadata_sha256": f"{epoch:064x}",
        "adapter_sha256": f"{epoch + 20:064x}",
        "optimizer_sha256": f"{epoch + 40:064x}",
        "new_bank_state_sha256": f"{epoch + 60:064x}",
        "recomputed_payload_hashes": {"tensor": f"{epoch + 80:064x}"},
        "optimizer_manifest": {"all_state_tensors_sha256": f"{epoch + 100:064x}"},
        "color": color,
        "mirror": mirror,
        "opaque_unit_margin_detail": {"epoch": epoch},
    }


def test_exact_screen_selected_epoch_and_continuation_state_are_pinned() -> None:
    evidence = controller._load_exact_screen(controller.CONFIG_PATH, controller.SCREEN_PATH)

    assert evidence["screen_sha256"] == controller.EXPECTED_SCREEN_SHA256
    assert evidence["training_source_provenance"] == (
        controller.EXPECTED_TRAINING_SOURCE_PROVENANCE
    )
    assert evidence["selected"]["epoch"] == 1
    assert {
        "adapter_sha256": evidence["selected"]["adapter_sha256"],
        "metadata_sha256": evidence["selected"]["metadata_sha256"],
        "optimizer_sha256": evidence["selected"]["optimizer_sha256"],
    } == controller.EXPECTED_SELECTED_ARTIFACTS
    assert (
        evidence["selected"]["optimizer_manifest"]["all_state_tensors_sha256"]
        == controller.EXPECTED_SELECTED_OPTIMIZER_STATE_SHA256
    )


def test_control_plane_transition_is_exact_and_committed() -> None:
    current = _controller_provenance()
    accepted, transition = controller._validate_control_plane_transition(
        controller.EXPECTED_TRAINING_SOURCE_PROVENANCE,
        current_provenance=current,
        transition=controller.EXPECTED_CONTROL_PLANE_TRANSITION,
    )
    assert accepted == current
    assert transition == {
        "Makefile": "M",
        "src/semantic_3d_chat/evaluation/v24_extension_controller.py": "A",
        "tests/test_v24_extension_controller.py": "A",
    }

    changed = dict(controller.EXPECTED_CONTROL_PLANE_TRANSITION)
    changed["src/semantic_3d_chat/training/train_adapter.py"] = "M"
    with pytest.raises(controller.V24ExtensionViolation, match="control-plane transition"):
        controller._validate_control_plane_transition(
            controller.EXPECTED_TRAINING_SOURCE_PROVENANCE,
            current_provenance=current,
            transition=changed,
        )


def test_manifest_binds_selected_optimizer_history_and_two_stage_training() -> None:
    evidence = controller._load_exact_screen(controller.CONFIG_PATH, controller.SCREEN_PATH)
    current = _controller_provenance()
    manifest = controller._manifest_body(
        evidence,
        current,
        controller.EXPECTED_CONTROL_PLANE_TRANSITION,
    )

    assert manifest["training_source_provenance"] == (
        controller.EXPECTED_TRAINING_SOURCE_PROVENANCE
    )
    assert manifest["controller_source_provenance"] == current
    assert manifest["selected_epoch"] == 1
    assert manifest["selected_checkpoint_artifact_hashes"] == (
        controller.EXPECTED_SELECTED_ARTIFACTS
    )
    assert manifest["selected_optimizer_manifest"]["all_state_tensors_sha256"] == (
        controller.EXPECTED_SELECTED_OPTIMIZER_STATE_SHA256
    )
    assert len(manifest["selected_history_sha256"]) == 64
    assert manifest["replay_epochs"] == [2, 3, 4]
    assert manifest["novel_epochs"] == [5, 6, 7, 8]
    assert manifest["final_selection_epochs"] == list(range(1, 9))
    assert manifest["stage_b_authorized"] is False
    assert manifest["trainer"]["requires_exact_provenance_preflight"] is True
    assert manifest["trainer"]["replay_argv"][-2:] == ["--epochs", "4"]
    assert manifest["trainer"]["novel_argv"][-2:] == ["--epochs", "8"]
    assert manifest["trainer"]["replay_resume"].endswith("epoch_001")
    assert manifest["trainer"]["novel_resume"].endswith("epoch_004")


def test_replay_normalization_allows_only_output_namespace() -> None:
    primary = {"epoch": 2, "output_namespace": controller.PRIMARY_NAMESPACE, "loss": 1.25}
    replay = {
        "epoch": 2,
        "output_namespace": controller.EXTENSION_NAMESPACE,
        "loss": 1.25,
    }
    assert controller._normalized_replay_metadata(primary) == (
        controller._normalized_replay_metadata(replay)
    )
    replay["loss"] = 1.0
    assert controller._normalized_replay_metadata(primary) != (
        controller._normalized_replay_metadata(replay)
    )


def test_replay_binds_decoded_optimizer_not_torch_save_container_bytes() -> None:
    semantic = {
        "adapter_sha256": "a" * 64,
        "new_bank_state_sha256": "b" * 64,
        "recomputed_payload_hashes": {"scene_state_sha256": "c" * 64},
        "optimizer_manifest": {"all_state_tensors_sha256": "d" * 64},
        "color": {"full_vocab_sides": 12},
        "mirror": {"full_vocab_sides": 10},
        "opaque_unit_margin_detail": {"source_detail_sha256": "e" * 64},
    }
    primary = {**copy.deepcopy(semantic), "optimizer_sha256": "f" * 64}
    replay = {**copy.deepcopy(semantic), "optimizer_sha256": "0" * 64}
    audit = controller._require_replay_semantic_equivalence(replay, primary, epoch=2)
    assert audit["container_byte_difference_present"] is True
    assert audit["decoded_optimizer_manifest_exact"] is True
    assert audit["decoded_all_state_tensors_sha256"] == "d" * 64

    replay["optimizer_manifest"]["all_state_tensors_sha256"] = "1" * 64
    with pytest.raises(controller.V24ExtensionViolation, match="optimizer_manifest"):
        controller._require_replay_semantic_equivalence(replay, primary, epoch=2)


def test_extension_layout_accepts_only_replay_or_complete_bounded_branch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / controller.EXTENSION_NAMESPACE
    root.mkdir()
    for epoch in (2, 3, 4):
        (root / f"epoch_{epoch:03d}").mkdir()
    monkeypatch.setattr(controller, "_extension_root", lambda _config: root)
    controller._require_replay_or_final_layout({}, field="replay")

    (root / "epoch_005").mkdir()
    with pytest.raises(controller.V24ExtensionViolation, match="must be exactly"):
        controller._require_replay_or_final_layout({}, field="partial novel branch")

    for epoch in (6, 7, 8):
        (root / f"epoch_{epoch:03d}").mkdir()
    controller._require_replay_or_final_layout({}, field="final")

    (root / "epoch_009").mkdir()
    with pytest.raises(controller.V24ExtensionViolation, match="must be exactly"):
        controller._require_replay_or_final_layout({}, field="overrun")


def test_history_contract_and_lora_a_moments_fail_closed() -> None:
    gate = {"by_pair": {"pair_000001": {"accuracy": 1.0}}}
    metadata = {
        "history": [
            {"epoch": 1, "pair_candidate_gate": {"old": 1}},
            {"epoch": 2, "pair_candidate_gate": gate},
        ],
        "pair_candidate_gate": gate,
    }
    assert len(controller._validate_history_contract(metadata, epoch=2)) == 2
    metadata["history"][-1]["epoch"] = 3
    with pytest.raises(controller.V24ExtensionViolation, match="history labels"):
        controller._validate_history_contract(metadata, epoch=2)

    manifest = {
        "parameter_states": [
            {"role": "A", "exp_avg_nonzero": 1, "exp_avg_sq_nonzero": 1},
            {"role": "B", "exp_avg_nonzero": 1, "exp_avg_sq_nonzero": 1},
        ]
    }
    controller._require_nonreset_a_moments(manifest, epoch=2)
    manifest["parameter_states"][0]["exp_avg_nonzero"] = 0
    with pytest.raises(controller.V24ExtensionViolation, match="reset"):
        controller._require_nonreset_a_moments(manifest, epoch=2)


def test_final_ranking_covers_complete_trajectory_and_strict_gate() -> None:
    color = _metrics(sides=12, units=6, minimum_candidate=0.5, minimum_full=0.5)
    weak = _metrics(
        sides=9,
        units=3,
        minimum_candidate=-0.5,
        minimum_full=-0.5,
        mean_candidate=0.5,
        mean_full=0.5,
    )
    rows = [_candidate_row(epoch, color, weak) for epoch in range(1, 9)]
    rows[0]["mirror"] = _metrics(
        sides=10,
        units=4,
        minimum_candidate=-0.8,
        minimum_full=-0.8,
        mean_candidate=0.6,
        mean_full=0.6,
    )
    result = controller._select_final_candidates(rows)
    assert result["selected"]["epoch"] == 1
    assert result["greedy_audit_authorized"] is False

    rows[4]["mirror"] = _metrics(
        sides=12,
        units=6,
        minimum_candidate=0.1,
        minimum_full=0.1,
    )
    result = controller._select_final_candidates(rows)
    assert result["selected"]["epoch"] == 5
    assert result["full_teacher_gate_passed"] is True
    assert result["greedy_audit_authorized"] is True

    rows[4]["mirror"]["minimum_candidate_margin"] = 0.0
    result = controller._select_final_candidates(rows)
    assert result["selected"]["epoch"] == 5
    assert result["full_teacher_gate_passed"] is False
    assert result["greedy_audit_authorized"] is False
