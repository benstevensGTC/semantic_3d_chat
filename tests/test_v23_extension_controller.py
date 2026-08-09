from __future__ import annotations

import copy
import subprocess
from pathlib import Path

import pytest

from semantic_3d_chat.config import load_config
from semantic_3d_chat.evaluation import v23_extension_controller as controller


def _controller_provenance() -> dict[str, object]:
    value = copy.deepcopy(controller.EXPECTED_TRAINING_SOURCE_PROVENANCE)
    value["head_commit"] = "c" * 40
    value["head_tree"] = "d" * 40
    return value


def test_exact_screen_and_selected_epoch_are_externally_pinned() -> None:
    evidence = controller._load_exact_screen(controller.CONFIG_PATH, controller.SCREEN_PATH)

    assert evidence["screen_sha256"] == controller.EXPECTED_SCREEN_SHA256
    assert evidence["training_source_provenance"] == (
        controller.EXPECTED_TRAINING_SOURCE_PROVENANCE
    )
    assert evidence["selected"]["epoch"] == 2
    assert {
        "adapter_sha256": evidence["selected"]["adapter_sha256"],
        "metadata_sha256": evidence["selected"]["metadata_sha256"],
        "optimizer_sha256": evidence["selected"]["optimizer_sha256"],
    } == controller.EXPECTED_SELECTED_ARTIFACTS


def test_control_plane_transition_is_exact_not_an_open_allowlist() -> None:
    current = _controller_provenance()
    accepted, records = controller._validate_control_plane_transition(
        controller.EXPECTED_TRAINING_SOURCE_PROVENANCE,
        current_provenance=current,
        transition=controller.EXPECTED_CONTROL_PLANE_TRANSITION,
    )
    assert accepted == current
    assert records == controller.EXPECTED_CONTROL_PLANE_TRANSITION

    changed = dict(controller.EXPECTED_CONTROL_PLANE_TRANSITION)
    changed["src/semantic_3d_chat/training/train_adapter.py"] = "M"
    with pytest.raises(controller.V23ExtensionViolation, match="control-plane transition"):
        controller._validate_control_plane_transition(
            controller.EXPECTED_TRAINING_SOURCE_PROVENANCE,
            current_provenance=current,
            transition=changed,
        )


def test_manifest_binds_two_stage_argv_and_distinct_source_provenance() -> None:
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
    assert manifest["replay_epochs"] == [3, 4]
    assert manifest["novel_epochs"] == [5, 6, 7, 8]
    assert manifest["stage_b_authorized"] is False
    assert manifest["trainer"]["requires_exact_provenance_preflight"] is True
    assert manifest["trainer"]["replay_argv"][-2:] == ["--epochs", "4"]
    assert manifest["trainer"]["novel_argv"][-2:] == ["--epochs", "8"]
    assert manifest["trainer"]["replay_resume"].endswith("epoch_002")
    assert manifest["trainer"]["novel_resume"].endswith("epoch_004")


def test_replay_normalization_allows_only_namespace() -> None:
    primary = {"epoch": 3, "output_namespace": controller.PRIMARY_NAMESPACE, "loss": 1.25}
    replay = {
        "epoch": 3,
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


def test_replay_accepts_noncanonical_optimizer_container_but_not_state_drift() -> None:
    semantic = {
        "adapter_sha256": "a" * 64,
        "new_bank_state_sha256": "b" * 64,
        "recomputed_payload_hashes": {"scene_state_sha256": "c" * 64},
        "optimizer_manifest": {"all_state_tensors_sha256": "d" * 64},
        "color": {"full_vocab_sides": 12},
        "mirror": {"full_vocab_sides": 10},
    }
    primary = {**copy.deepcopy(semantic), "optimizer_sha256": "e" * 64}
    replay = {**copy.deepcopy(semantic), "optimizer_sha256": "f" * 64}

    audit = controller._require_replay_semantic_equivalence(replay, primary, epoch=3)

    assert audit == {
        "replay_optimizer_sha256": "f" * 64,
        "primary_optimizer_sha256": "e" * 64,
        "container_bytes_equal": False,
        "container_byte_difference_present": True,
        "container_byte_difference_classification": (
            "expected_non_semantic_torch_save_reserialization"
        ),
        "decoded_optimizer_manifest_exact": True,
        "decoded_all_state_tensors_sha256": "d" * 64,
    }

    replay["optimizer_manifest"]["all_state_tensors_sha256"] = "0" * 64
    with pytest.raises(controller.V23ExtensionViolation, match="optimizer_manifest"):
        controller._require_replay_semantic_equivalence(replay, primary, epoch=3)


def test_extension_layout_rejects_extra_epoch_and_parent_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / controller.EXTENSION_NAMESPACE
    root.mkdir()
    (root / "epoch_003").mkdir()
    (root / "epoch_004").mkdir()
    monkeypatch.setattr(controller, "_extension_root", lambda _config: root)
    controller._require_extension_layout({}, {3, 4}, field="replay")

    (root / "epoch_005").mkdir()
    with pytest.raises(controller.V23ExtensionViolation, match="epoch directory set"):
        controller._require_extension_layout({}, {3, 4}, field="replay")

    checkpoint_root = tmp_path / "checkpoints"
    target = tmp_path / "target"
    checkpoint_root.mkdir()
    target.mkdir()
    (checkpoint_root / controller.EXTENSION_NAMESPACE).symlink_to(target, target_is_directory=True)
    monkeypatch.undo()
    monkeypatch.setattr(controller, "artifact_root", lambda _config, _kind: checkpoint_root)
    with pytest.raises(controller.V23ExtensionViolation, match="symlink"):
        controller._extension_root(load_config(controller.CONFIG_PATH))


def test_cached_replay_rerun_accepts_only_replay_or_complete_final_layout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / controller.EXTENSION_NAMESPACE
    root.mkdir()
    for epoch in (3, 4):
        (root / f"epoch_{epoch:03d}").mkdir()
    monkeypatch.setattr(controller, "_extension_root", lambda _config: root)

    controller._require_replay_or_final_layout({}, field="cached replay")

    (root / "epoch_005").mkdir()
    with pytest.raises(controller.V23ExtensionViolation, match="must be exactly"):
        controller._require_replay_or_final_layout({}, field="partial novel branch")

    for epoch in (6, 7, 8):
        (root / f"epoch_{epoch:03d}").mkdir()
    controller._require_replay_or_final_layout({}, field="cached final branch")

    (root / "epoch_009").mkdir()
    with pytest.raises(controller.V23ExtensionViolation, match="must be exactly"):
        controller._require_replay_or_final_layout({}, field="overrun branch")


def test_lora_a_optimizer_moments_cannot_reset_after_selected_epoch() -> None:
    manifest = {
        "parameter_states": [
            {"role": "A", "exp_avg_nonzero": 1, "exp_avg_sq_nonzero": 1},
            {"role": "B", "exp_avg_nonzero": 1, "exp_avg_sq_nonzero": 1},
        ]
    }
    controller._require_nonreset_a_moments(manifest, epoch=3)
    manifest["parameter_states"][0]["exp_avg_nonzero"] = 0
    with pytest.raises(controller.V23ExtensionViolation, match="reset"):
        controller._require_nonreset_a_moments(manifest, epoch=3)


def test_branch_history_requires_exact_labels_and_top_level_gate_binding() -> None:
    gate = {"by_pair": {"pair_000001": {"accuracy": 1.0}}}
    metadata = {
        "history": [
            {"epoch": 1, "pair_candidate_gate": {"old": 1}},
            {"epoch": 2, "pair_candidate_gate": gate},
        ],
        "pair_candidate_gate": gate,
    }
    assert len(controller._validate_history_contract(metadata, epoch=2)) == 2

    mislabeled = copy.deepcopy(metadata)
    mislabeled["history"][-1]["epoch"] = 3
    with pytest.raises(controller.V23ExtensionViolation, match="history labels"):
        controller._validate_history_contract(mislabeled, epoch=2)

    detached_gate = copy.deepcopy(metadata)
    detached_gate["pair_candidate_gate"] = {"by_pair": {}}
    with pytest.raises(controller.V23ExtensionViolation, match="pair gate"):
        controller._validate_history_contract(detached_gate, epoch=2)


def test_make_dry_run_contains_two_separate_pinned_training_stages() -> None:
    result = subprocess.run(
        ["make", "-n", "gemma4-v23-extension"],
        cwd=controller.PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "git worktree add --detach" in result.stdout
    assert "requires_exact_provenance" not in result.stdout  # guard is executable Python
    assert "--resume \"data_gemma4/checkpoints/gemma4_v23_shared_kv/epoch_002\"" in result.stdout
    assert "--epochs \"4\"" in result.stdout
    assert "--resume \"data_gemma4/checkpoints/gemma4_v23_shared_kv_extension_u8/epoch_004\"" in result.stdout
    assert "--epochs \"8\"" in result.stdout
