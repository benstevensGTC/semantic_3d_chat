from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from semantic_3d_chat.evaluation import v21_extension_controller as controller


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _pair_metrics(*, full: bool) -> dict[str, Any]:
    sides = 12 if full else 8
    units = 6 if full else 2
    minimum = 0.5 if full else -0.5
    return {
        "candidate_sides": sides,
        "candidate_units": units,
        "full_vocab_sides": sides,
        "full_vocab_units": units,
        "mean_candidate_margin": 1.0 if full else 0.2,
        "mean_full_vocab_margin": 1.0 if full else 0.2,
        "minimum_candidate_margin": minimum,
        "minimum_full_vocab_margin": minimum,
    }


def _screen_epoch(epoch: int, *, mirror_full: bool = False) -> dict[str, Any]:
    return {
        "epoch": epoch,
        "color": _pair_metrics(full=True),
        "mirror": _pair_metrics(full=mirror_full),
    }


def _contract() -> dict[str, Any]:
    return {
        "config_hash": "a" * 12,
        "config_hash_full": "a" * 64,
        "preflight_contract_sha256": "b" * 64,
    }


def _screen(**overrides: Any) -> dict[str, Any]:
    value = {
        "selector_type": "strict_v21_signed_x_local_field_phase_aware_epoch_selector",
        "report_only": True,
        "model_inference_executed": False,
        "gemma_model_loaded": False,
        "checkpoint_tensor_state_loaded": True,
        "checkpoint_tensor_state_safely_inspected": True,
        "optimizer_deserialized": True,
        "optimizer_deserialization_weights_only": True,
        "model_dtype": "bfloat16",
        "continuation_authorized": True,
        "continuation_gate_passed": True,
        "full_teacher_gate_passed": False,
        "greedy_audit_authorized": False,
        "greedy_audit_forbidden": True,
        "decision": "continue_selected_epoch_no_greedy_audit",
        "conditional_max_optimizer_updates": 8,
        "selected_epoch": 3,
        "update1_authorization": {"report_path": "reports/v21_update1_match.json"},
        "selected_signed_x_state_sha256": _digest("selected-state"),
        "selected_optimizer_state_sha256": _digest("selected-optimizer-state"),
        "selection_policy": {
            "continuation_requires": {
                "mirror_minimum_full_vocab_sides": 8,
                "mirror_minimum_full_vocab_units": 2,
            }
        },
        "epochs": [_screen_epoch(epoch) for epoch in controller.EXPECTED_EPOCHS],
    }
    value.update(overrides)
    return value


def test_only_continuation_without_full_teacher_authorizes_extension() -> None:
    assert controller._require_extension_decision(_screen()) == 3
    for mutation in (
        {"continuation_authorized": False},
        {"full_teacher_gate_passed": True},
        {"greedy_audit_authorized": True},
        {"conditional_max_optimizer_updates": 12},
        {"model_dtype": "float16"},
    ):
        with pytest.raises(controller.V21ExtensionViolation):
            controller._require_extension_decision(_screen(**mutation))


def test_behavioral_gate_is_recomputed_instead_of_trusting_boolean_aliases() -> None:
    weak = [_screen_epoch(epoch) for epoch in controller.EXPECTED_EPOCHS]
    weak[2]["mirror"]["full_vocab_sides"] = 7
    with pytest.raises(controller.V21ExtensionViolation, match="8-side/2-unit"):
        controller._require_extension_decision(_screen(epochs=weak))

    already_full = [_screen_epoch(epoch) for epoch in controller.EXPECTED_EPOCHS]
    already_full[2] = _screen_epoch(3, mirror_full=True)
    with pytest.raises(controller.V21ExtensionViolation, match="already meets"):
        controller._require_extension_decision(_screen(epochs=already_full))


def test_launch_manifest_targets_only_isolated_update_eight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    selected = checkpoint_root / controller.OUTPUT_NAMESPACE / "epoch_003"
    selected.mkdir(parents=True)
    for name in ("adapter.safetensors", "metadata.json", "optimizer.pt"):
        (selected / name).write_bytes(name.encode())
    metadata_hash = controller._file_sha256(selected / "metadata.json", "metadata")
    artifact_hashes = controller._checkpoint_hashes(selected, "selected")
    screen = _screen(
        selected_checkpoint=str(selected),
        selected_checkpoint_metadata_path=str(selected / "metadata.json"),
        selected_checkpoint_metadata_sha256=metadata_hash,
        selected_checkpoint_artifact_hashes=artifact_hashes,
    )
    monkeypatch.setattr(controller, "artifact_root", lambda _config, _kind: checkpoint_root)
    monkeypatch.setattr(controller, "_validate_config", lambda _config: _contract())
    report = controller._build_launch_manifest(
        config_path=tmp_path / "v21.yaml",
        screen_path=tmp_path / "screen.json",
        screen=screen,
        screen_sha256=_digest("screen"),
        config={},
        source_provenance={"clean": True},
        require_namespace_absent=True,
    )
    assert report["target_optimizer_update"] == 8
    assert report["model_dtype"] == "bfloat16"
    assert report["config_hash_full"] == "a" * 64
    assert report["preflight_contract_sha256"] == "b" * 64
    assert report["frozen_scene_state_sha256"] == controller.EXPECTED_FROZEN_SCENE_SHA256
    assert report["start_optimizer_update"] == 4
    assert report["expected_extension_epochs"] == [4, 5, 6, 7, 8]
    assert report["extension_output_namespace"].endswith("_extension_u8")
    assert report["overwrite_original_namespace"] is False
    assert report["greedy_audit_forbidden_during_extension"] is True
    assert report["trainer"]["executes_on_prepare"] is False
    assert report["trainer"]["argv"][-2:] == ["--epochs", "8"]
    assert report["update1_authorization"] == screen["update1_authorization"]
    assert report["selected_checkpoint_artifact_hashes"] == artifact_hashes


def test_existing_extension_namespace_is_never_overwritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    selected = checkpoint_root / controller.OUTPUT_NAMESPACE / "epoch_003"
    selected.mkdir(parents=True)
    for name in ("adapter.safetensors", "metadata.json", "optimizer.pt"):
        (selected / name).write_bytes(name.encode())
    extension = checkpoint_root / controller.EXTENSION_NAMESPACE
    extension.mkdir(parents=True)
    monkeypatch.setattr(controller, "artifact_root", lambda _config, _kind: checkpoint_root)
    monkeypatch.setattr(controller, "_validate_config", lambda _config: _contract())
    screen = _screen(
        selected_checkpoint=str(selected),
        selected_checkpoint_metadata_path=str(selected / "metadata.json"),
        selected_checkpoint_metadata_sha256=controller._file_sha256(
            selected / "metadata.json", "metadata"
        ),
        selected_checkpoint_artifact_hashes=controller._checkpoint_hashes(selected, "selected"),
    )
    with pytest.raises(controller.V21ExtensionViolation, match="overwrite"):
        controller._build_launch_manifest(
            config_path=tmp_path / "v21.yaml",
            screen_path=tmp_path / "screen.json",
            screen=screen,
            screen_sha256=_digest("screen"),
            config={},
            source_provenance={"clean": True},
            require_namespace_absent=True,
        )


def test_candidate_never_authorizes_greedy_before_full_teacher() -> None:
    metrics = {
        "color": {
            "candidate_sides": 12,
            "candidate_units": 6,
            "full_vocab_sides": 12,
            "full_vocab_units": 6,
            "mean_candidate_margin": 1.0,
            "mean_full_vocab_margin": 1.0,
            "minimum_candidate_margin": 0.5,
            "minimum_full_vocab_margin": 0.5,
        },
        "mirror": {
            "candidate_sides": 8,
            "candidate_units": 2,
            "full_vocab_sides": 8,
            "full_vocab_units": 2,
            "mean_candidate_margin": 0.2,
            "mean_full_vocab_margin": 0.2,
            "minimum_candidate_margin": -0.5,
            "minimum_full_vocab_margin": -0.5,
        },
    }
    candidate = controller._candidate(
        4,
        metrics,
        metadata_path="epoch_004/metadata.json",
        metadata_sha256=_digest("metadata"),
        artifact_hashes={
            "adapter_sha256": _digest("adapter"),
            "metadata_sha256": _digest("metadata"),
            "optimizer_sha256": _digest("optimizer"),
        },
        state_sha256=_digest("state"),
        optimizer_state_sha256=_digest("optimizer-state"),
        screen={
            "selection_policy": {
                "continuation_requires": {
                    "mirror_minimum_full_vocab_sides": 8,
                    "mirror_minimum_full_vocab_units": 2,
                }
            }
        },
    )
    assert candidate["continuation_gate_passed"] is True
    assert candidate["full_teacher_gate_passed"] is False
    assert candidate["model_dtype"] == "bfloat16"
    assert candidate["frozen_scene_state_sha256"] == controller.EXPECTED_FROZEN_SCENE_SHA256


@pytest.mark.parametrize("artifact", ["adapter.safetensors", "optimizer.pt"])
def test_selected_adapter_or_optimizer_cannot_differ_from_screen(
    artifact: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    selected = checkpoint_root / controller.OUTPUT_NAMESPACE / "epoch_003"
    selected.mkdir(parents=True)
    for name in ("adapter.safetensors", "metadata.json", "optimizer.pt"):
        (selected / name).write_bytes(name.encode())
    expected_hashes = controller._checkpoint_hashes(selected, "selected")
    (selected / artifact).write_bytes(b"changed after screen")
    screen = _screen(
        selected_checkpoint=str(selected),
        selected_checkpoint_metadata_path=str(selected / "metadata.json"),
        selected_checkpoint_metadata_sha256=expected_hashes["metadata_sha256"],
        selected_checkpoint_artifact_hashes=expected_hashes,
    )
    monkeypatch.setattr(controller, "artifact_root", lambda _config, _kind: checkpoint_root)
    monkeypatch.setattr(controller, "_validate_config", lambda _config: _contract())
    with pytest.raises(controller.V21ExtensionViolation, match="adapter/metadata/optimizer"):
        controller._build_launch_manifest(
            config_path=tmp_path / "v21.yaml",
            screen_path=tmp_path / "screen.json",
            screen=screen,
            screen_sha256=_digest("screen"),
            config={},
            source_provenance={"clean": True},
            require_namespace_absent=True,
        )


def test_extension_recomputes_selector_with_bound_update1_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "v21.yaml"
    config_path.write_text("seed: 1\n", encoding="utf-8")
    selection_path = tmp_path / "selection.json"
    selection_path.write_text("{}\n", encoding="utf-8")
    epoch_paths: dict[int, Path] = {}
    rows: list[dict[str, Any]] = []
    for epoch in controller.EXPECTED_EPOCHS:
        path = tmp_path / f"epoch_{epoch:03d}" / "metadata.json"
        path.parent.mkdir()
        path.write_text("{}\n", encoding="utf-8")
        epoch_paths[epoch] = path
        rows.append({"epoch": epoch, "checkpoint_metadata_path": str(path)})
    update1_path = tmp_path / "v21_update1_match.json"
    update1_path.write_text("{}\n", encoding="utf-8")
    screen = {
        "selection_artifact_path": str(selection_path),
        "update1_authorization": {"report_path": str(update1_path)},
        "epochs": rows,
    }
    screen_path = tmp_path / "screen.json"
    screen_path.write_text(json.dumps(screen) + "\n", encoding="utf-8")
    calls: list[str] = []

    def recompute(
        _config: dict[str, Any],
        _selection: dict[str, Any],
        _epochs: dict[int, dict[str, Any]],
        **kwargs: Any,
    ) -> dict[str, Any]:
        calls.append(str(kwargs["update1_report_path"]))
        return copy.deepcopy(screen)

    monkeypatch.setattr(controller, "load_config", lambda _path: {})
    monkeypatch.setattr(controller, "summarize_v21_epochs", recompute)
    loaded, _digest_value, _config, _selection = controller._load_exact_screen_report(
        config_path, screen_path
    )
    assert loaded == screen
    assert calls == [str(update1_path)]


def _final_selector_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    full_at_update8: bool = True,
    full_at_update7: bool = False,
    fork_epoch: int | None = None,
    repeated_pre_gate_epoch: int | None = None,
    metric_drift_epoch: int | None = None,
) -> dict[str, Any]:
    source = {
        "schema_version": 1,
        "repository_root": str(tmp_path),
        "commit_sha": "c" * 40,
        "tree_sha": "d" * 40,
        "diff_sha256": hashlib.sha256(b"").hexdigest(),
        "diff_bytes": 0,
        "dirty": False,
        "tracked_dirty": False,
        "untracked_paths": [],
    }
    selected_epoch = 4
    selected_checkpoint = tmp_path / controller.OUTPUT_NAMESPACE / "epoch_004"
    selected_checkpoint.mkdir(parents=True)
    (selected_checkpoint / "metadata.json").write_text("{}\n", encoding="utf-8")
    extension_root = tmp_path / controller.EXTENSION_NAMESPACE
    selected_hashes = {
        "adapter_sha256": "a" * 64,
        "metadata_sha256": "b" * 64,
        "optimizer_sha256": "c" * 64,
    }
    manifest = {
        "selected_epoch": selected_epoch,
        "selected_checkpoint": str(selected_checkpoint),
        "selected_checkpoint_artifact_hashes": selected_hashes,
        "extension_checkpoint_root": str(extension_root),
        "expected_extension_epochs": [5, 6, 7, 8],
        "source_provenance": source,
        "config_path": "configs/v21.yaml",
        "config_hash": "a" * 12,
        "config_hash_full": "a" * 64,
        "preflight_contract_sha256": "b" * 64,
        "screen_report_path": "screen.json",
        "screen_report_sha256": "d" * 64,
        "update1_authorization": {"report_path": "update1.json"},
    }
    selection_policy = {
        "continuation_requires": {
            "mirror_minimum_full_vocab_sides": 8,
            "mirror_minimum_full_vocab_units": 2,
        }
    }
    screen_epochs: list[dict[str, Any]] = []
    for epoch in range(1, 5):
        screen_epochs.append(
            {
                **_screen_epoch(epoch),
                "checkpoint_metadata_path": (
                    f"{controller.OUTPUT_NAMESPACE}/epoch_{epoch:03d}/metadata.json"
                ),
                "checkpoint_metadata_sha256": f"{epoch:x}" * 64,
                "checkpoint_artifact_hashes": {
                    "adapter_sha256": f"{epoch:x}" * 64,
                    "metadata_sha256": f"{epoch:x}" * 64,
                    "optimizer_sha256": f"{epoch:x}" * 64,
                },
                "signed_x_state_sha256": f"{epoch:x}" * 64,
                "optimizer_state_sha256": f"{epoch + 8:x}" * 64,
            }
        )
    screen = {"selection_policy": selection_policy, "epochs": screen_epochs}
    history = [{"epoch": epoch} for epoch in range(1, 9)]
    selected_metadata = {
        "history": copy.deepcopy(history[:4]),
        "initialization_provenance": {"source": "v18"},
        "signed_x_scene_residual_zero_output_equivalence": {"verified": True},
    }

    def extension_row(epoch: int, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
        epoch_history = copy.deepcopy(history[:epoch])
        if fork_epoch == epoch:
            epoch_history[0] = {"epoch": 999}
        state = f"{epoch:x}" * 64
        if repeated_pre_gate_epoch == epoch:
            state = f"{epoch - 1:x}" * 64
        full = (full_at_update8 and epoch == 8) or (full_at_update7 and epoch in {7, 8})
        mirror = _pair_metrics(full=full)
        if metric_drift_epoch == epoch:
            mirror["mean_candidate_margin"] += 0.125
        if not full:
            sides = min(11, epoch + 3)
            units = min(5, epoch - 3)
            mirror.update(
                {
                    "candidate_sides": sides,
                    "full_vocab_sides": sides,
                    "candidate_units": units,
                    "full_vocab_units": units,
                }
            )
        return {
            "epoch": epoch,
            "path": str(extension_root / f"epoch_{epoch:03d}" / "metadata.json"),
            "artifact_sha256": f"{epoch:x}" * 64,
            "signed_x_state_sha256": state,
            "optimizer_state_sha256": f"{epoch + 8:x}" * 64,
            "history": epoch_history,
            "initialization_provenance": {"source": "v18"},
            "zero_output_equivalence": {"verified": True},
            "metrics": {"color": _pair_metrics(full=True), "mirror": mirror},
            "checkpoint_artifact_hashes": {
                "adapter_sha256": f"{epoch:x}" * 64,
                "metadata_sha256": f"{epoch:x}" * 64,
                "optimizer_sha256": f"{epoch:x}" * 64,
            },
        }

    monkeypatch.setattr(
        controller,
        "_validate_launch_manifest",
        lambda *_args, **_kwargs: (manifest, screen, {}),
    )
    monkeypatch.setattr(controller, "_validate_config", lambda _config: _contract())
    monkeypatch.setattr(controller, "artifact_root", lambda _config, _kind: tmp_path)
    monkeypatch.setattr(
        controller,
        "_load_json_strict",
        lambda _path: (selected_metadata, selected_hashes["metadata_sha256"]),
    )
    monkeypatch.setattr(controller, "_checkpoint_hashes", lambda *_args: selected_hashes)
    monkeypatch.setattr(controller, "_validate_extension_epoch", extension_row)
    monkeypatch.setattr(controller, "_file_sha256", lambda *_args: "f" * 64)
    return source


def test_final_selector_revalidates_full_gate_before_unlocking_greedy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _final_selector_fixture(tmp_path, monkeypatch)
    report = controller.select_final_extension(tmp_path / "launch.json", current_provenance=source)
    assert report["epoch_count"] == 8
    assert report["selected_epoch"] == 8
    assert report["conditional_limit_reached"] is True
    assert report["continuation_authorized"] is False
    assert report["full_teacher_gate_passed"] is True
    assert report["greedy_audit_authorized"] is True
    assert report["decision"] == "full_teacher_gate_passed_greedy_audit_allowed"
    assert report["model_dtype"] == "bfloat16"
    assert report["cumulative_update_evidence"]["extension_history_prefixes_exact"] is True


def test_conditional_limit_stays_closed_without_full_teacher_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _final_selector_fixture(tmp_path, monkeypatch, full_at_update8=False)
    report = controller.select_final_extension(tmp_path / "launch.json", current_provenance=source)
    assert report["selected_epoch"] == 8
    assert report["full_teacher_gate_passed"] is False
    assert report["greedy_audit_authorized"] is False
    assert report["greedy_audit_forbidden"] is True
    assert report["decision"] == "conditional_limit_reached_no_greedy_audit"


@pytest.mark.parametrize("mode", ["fork", "repeat"])
def test_final_selector_rejects_history_fork_and_pre_gate_state_repeat(
    mode: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kwargs = {"fork_epoch": 7} if mode == "fork" else {"repeated_pre_gate_epoch": 7}
    source = _final_selector_fixture(tmp_path, monkeypatch, **kwargs)
    pattern = "forks cumulative history" if mode == "fork" else "before a full-teacher plateau"
    with pytest.raises(controller.V21ExtensionViolation, match=pattern):
        controller.select_final_extension(tmp_path / "launch.json", current_provenance=source)


def test_identical_state_cannot_newly_claim_full_teacher_at_update8(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _final_selector_fixture(
        tmp_path,
        monkeypatch,
        full_at_update8=True,
        repeated_pre_gate_epoch=8,
    )
    with pytest.raises(controller.V21ExtensionViolation, match="before a full-teacher plateau"):
        controller.select_final_extension(tmp_path / "launch.json", current_provenance=source)


def test_identical_state_plateau_requires_prior_full_gate_and_exact_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _final_selector_fixture(
        tmp_path,
        monkeypatch,
        full_at_update7=True,
        repeated_pre_gate_epoch=8,
    )
    report = controller.select_final_extension(tmp_path / "launch.json", current_provenance=source)
    assert report["greedy_audit_authorized"] is True
    assert report["cumulative_update_evidence"]["repeated_full_teacher_plateau_epochs"] == [8]


def test_identical_full_state_with_metric_drift_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _final_selector_fixture(
        tmp_path,
        monkeypatch,
        full_at_update7=True,
        repeated_pre_gate_epoch=8,
        metric_drift_epoch=8,
    )
    with pytest.raises(controller.V21ExtensionViolation, match="before a full-teacher plateau"):
        controller.select_final_extension(tmp_path / "launch.json", current_provenance=source)


@pytest.mark.parametrize("entrypoint", ["screen", "manifest"])
@pytest.mark.parametrize("link_location", ["file", "parent"])
def test_top_level_controller_inputs_reject_symlinks_before_read(
    entrypoint: str,
    link_location: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "v21.yaml"
    config_path.write_text("seed: 1\n", encoding="utf-8")
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    real_input = real_parent / f"{entrypoint}.json"
    real_input.write_text("{}\n", encoding="utf-8")
    if link_location == "file":
        exposed = tmp_path / f"{entrypoint}_alias.json"
        exposed.symlink_to(real_input)
    else:
        alias_parent = tmp_path / "alias_parent"
        alias_parent.symlink_to(real_parent, target_is_directory=True)
        exposed = alias_parent / real_input.name

    reads: list[Path] = []

    def forbidden_read(path: Path) -> tuple[dict[str, Any], str]:
        reads.append(path)
        raise AssertionError("symlinked input was read")

    monkeypatch.setattr(controller, "_load_json_strict", forbidden_read)
    with pytest.raises(controller.V21ExtensionViolation, match="symbolic link"):
        if entrypoint == "screen":
            controller._load_exact_screen_report(config_path, exposed)
        else:
            controller._validate_launch_manifest(exposed)
    assert reads == []


def test_controller_itself_has_no_model_or_unsafe_tensor_imports() -> None:
    source_path = Path(controller.__file__ or "")
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".")[0])
    assert imported_roots.isdisjoint(
        {"torch", "transformers", "safetensors", "numpy", "PIL", "blender"}
    )
    text = source_path.read_text(encoding="utf-8")
    assert "torch.load" not in text
    assert "weights_only=False" not in text


def test_controller_rejects_oracle_paths_before_loading(tmp_path: Path) -> None:
    forbidden = tmp_path / "oracle" / "screen.json"
    with pytest.raises(controller.V21ExtensionViolation, match="oracle"):
        controller._reject_path(forbidden)
