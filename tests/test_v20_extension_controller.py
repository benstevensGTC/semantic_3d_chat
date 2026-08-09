from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from semantic_3d_chat.evaluation import v20_extension_controller as controller


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _screen(**overrides: Any) -> dict[str, Any]:
    value = {
        "selector_type": "strict_v20_signed_x_local_field_epoch_selector",
        "report_only": True,
        "model_inference_executed": False,
        "gemma_model_loaded": False,
        "checkpoint_tensor_state_loaded": True,
        "checkpoint_tensor_state_safely_inspected": True,
        "optimizer_deserialized": True,
        "optimizer_deserialization_weights_only": True,
        "continuation_authorized": True,
        "continuation_gate_passed": True,
        "full_teacher_gate_passed": False,
        "greedy_audit_authorized": False,
        "greedy_audit_forbidden": True,
        "decision": "continue_selected_epoch_no_greedy_audit",
        "conditional_max_optimizer_updates": 8,
        "selected_epoch": 3,
        "update1_authorization": {"report_path": "reports/v20_update1_match.json"},
        "selected_signed_x_state_sha256": _digest("selected-state"),
        "selected_optimizer_state_sha256": _digest("selected-optimizer-state"),
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
    ):
        with pytest.raises(controller.V20ExtensionViolation):
            controller._require_extension_decision(_screen(**mutation))


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
    monkeypatch.setattr(controller, "_validate_config", lambda _config: {"config_hash": "a" * 12})
    report = controller._build_launch_manifest(
        config_path=tmp_path / "v20.yaml",
        screen_path=tmp_path / "screen.json",
        screen=screen,
        screen_sha256=_digest("screen"),
        config={},
        source_provenance={"clean": True},
        require_namespace_absent=True,
    )
    assert report["target_optimizer_update"] == 8
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
    monkeypatch.setattr(controller, "_validate_config", lambda _config: {"config_hash": "a" * 12})
    screen = _screen(
        selected_checkpoint=str(selected),
        selected_checkpoint_metadata_path=str(selected / "metadata.json"),
        selected_checkpoint_metadata_sha256=controller._file_sha256(
            selected / "metadata.json", "metadata"
        ),
        selected_checkpoint_artifact_hashes=controller._checkpoint_hashes(selected, "selected"),
    )
    with pytest.raises(controller.V20ExtensionViolation, match="overwrite"):
        controller._build_launch_manifest(
            config_path=tmp_path / "v20.yaml",
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
    monkeypatch.setattr(controller, "_validate_config", lambda _config: {"config_hash": "a" * 12})
    with pytest.raises(controller.V20ExtensionViolation, match="adapter/metadata/optimizer"):
        controller._build_launch_manifest(
            config_path=tmp_path / "v20.yaml",
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
    update1_path = tmp_path / "v20_update1_match.json"
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
    monkeypatch.setattr(controller, "summarize_v20_epochs", recompute)
    loaded, _digest_value, _config, _selection = controller._load_exact_screen_report(
        tmp_path / "v20.yaml", screen_path
    )
    assert loaded == screen
    assert calls == [str(update1_path)]
