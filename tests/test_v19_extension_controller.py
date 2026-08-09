from __future__ import annotations

import ast
import hashlib
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from semantic_3d_chat.config import load_config
from semantic_3d_chat.evaluation import v19_extension_controller as controller
from semantic_3d_chat.evaluation.v19_extension_controller import (
    EXTENSION_NAMESPACE,
    TARGET_OPTIMIZER_UPDATE,
    V19ExtensionViolation,
    _build_launch_manifest,
    _load_exact_screen_report,
    _require_extension_decision,
    _validate_extension_epoch,
    prepare_extension_launch,
    select_final_extension,
)

CONFIG_PATH = "configs/experiments/gemma4_color_mirror_signed_x_moment_v19.yaml"
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()


def _source() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "scope": "repository_excluding_generated_artifacts_v1",
        "available": True,
        "head_commit": "a" * 40,
        "head_tree": "b" * 40,
        "is_clean": True,
        "tracked_diff_sha256": EMPTY_SHA256,
    }


def _screen_decision(selected_epoch: int = 4) -> dict[str, Any]:
    return {
        "selector_type": "strict_v19_signed_x_moment_epoch_selector",
        "report_only": True,
        "model_inference_executed": False,
        "checkpoint_tensor_state_loaded": False,
        "continuation_authorized": True,
        "continuation_gate_passed": True,
        "full_teacher_gate_passed": False,
        "greedy_audit_authorized": False,
        "greedy_audit_forbidden": True,
        "decision": "continue_selected_epoch_no_greedy_audit",
        "conditional_max_optimizer_updates": 12,
        "selected_epoch": selected_epoch,
    }


def _write_checkpoint(directory: Path, metadata: bytes = b"metadata\n") -> dict[str, str]:
    directory.mkdir(parents=True)
    (directory / "adapter.safetensors").write_bytes(b"adapter")
    (directory / "metadata.json").write_bytes(metadata)
    (directory / "optimizer.pt").write_bytes(b"optimizer")
    return {
        "adapter_sha256": hashlib.sha256(b"adapter").hexdigest(),
        "metadata_sha256": hashlib.sha256(metadata).hexdigest(),
        "optimizer_sha256": hashlib.sha256(b"optimizer").hexdigest(),
    }


def test_prepare_builds_exact_isolated_trainer_argv_and_refuses_reuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_config(CONFIG_PATH)
    checkpoint_root = tmp_path / "checkpoints"
    selected = checkpoint_root / controller.OUTPUT_NAMESPACE / "epoch_004"
    hashes = _write_checkpoint(selected)
    python = tmp_path / controller.PYTHON_EXECUTABLE
    python.parent.mkdir(parents=True)
    python.write_bytes(b"python")
    screen = {
        **_screen_decision(),
        "source_provenance": _source(),
        "selected_checkpoint_metadata_path": str(selected / "metadata.json"),
        "selected_checkpoint_metadata_sha256": hashes["metadata_sha256"],
    }
    monkeypatch.setattr(controller, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(controller, "artifact_root", lambda _config, _kind: checkpoint_root)
    report = _build_launch_manifest(
        config_path=tmp_path / "config.yaml",
        screen_path=tmp_path / "screen.json",
        screen=screen,
        screen_sha256="c" * 64,
        config=config,
        source_provenance=_source(),
        require_namespace_absent=True,
    )

    assert report["selected_checkpoint_artifact_hashes"] == hashes
    assert report["extension_output_namespace"] == EXTENSION_NAMESPACE
    assert report["extension_output_namespace"] != report["original_output_namespace"]
    assert report["expected_extension_epochs"] == list(range(5, 13))
    assert report["overwrite_original_namespace"] is False
    assert report["trainer"]["argv"] == [
        controller.PYTHON_EXECUTABLE,
        "-m",
        controller.TRAINING_MODULE,
        "--config",
        "config.yaml",
        "--resume",
        f"checkpoints/{controller.OUTPUT_NAMESPACE}/epoch_004",
        "--output-namespace",
        EXTENSION_NAMESPACE,
        "--epochs",
        "12",
    ]
    assert report["trainer"]["executes_on_prepare"] is False

    (checkpoint_root / EXTENSION_NAMESPACE).mkdir()
    with pytest.raises(V19ExtensionViolation, match="Refusing to reuse or overwrite"):
        _build_launch_manifest(
            config_path=tmp_path / "config.yaml",
            screen_path=tmp_path / "screen.json",
            screen=screen,
            screen_sha256="c" * 64,
            config=config,
            source_provenance=_source(),
            require_namespace_absent=True,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("continuation_authorized", False, "not authorized"),
        ("continuation_gate_passed", False, "did not pass"),
        ("full_teacher_gate_passed", True, "already passed"),
        ("greedy_audit_authorized", True, "already authorized"),
        ("greedy_audit_forbidden", False, "does not explicitly forbid"),
        ("conditional_max_optimizer_updates", 13, "must equal 12"),
    ],
)
def test_extension_decision_is_fail_closed(field: str, value: Any, message: str) -> None:
    screen = _screen_decision()
    screen[field] = value
    with pytest.raises(V19ExtensionViolation, match=message):
        _require_extension_decision(screen)


def test_exact_screen_is_recomputed_from_all_bound_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selection_path = tmp_path / "selection.json"
    epoch_paths = {epoch: tmp_path / f"epoch_{epoch}.json" for epoch in range(1, 5)}
    screen_path = tmp_path / "screen.json"
    screen = {
        "selection_artifact_path": str(selection_path),
        "epochs": [
            {"epoch": epoch, "checkpoint_metadata_path": str(path)}
            for epoch, path in epoch_paths.items()
        ],
    }
    values: dict[Path, tuple[dict[str, Any], str]] = {
        screen_path.resolve(): (screen, "1" * 64),
        selection_path.resolve(): ({"selection": True}, "2" * 64),
        **{
            path.resolve(): ({"epoch": epoch}, str(epoch) * 64)
            for epoch, path in epoch_paths.items()
        },
    }
    monkeypatch.setattr(controller, "load_config", lambda _path: {"config": True})
    monkeypatch.setattr(controller, "_load_json_strict", lambda path: values[path.resolve()])
    monkeypatch.setattr(controller, "summarize_v19_epochs", lambda *_args, **_kwargs: screen)
    observed, digest, config, selection = _load_exact_screen_report(
        tmp_path / "config.yaml", screen_path
    )
    assert observed == screen
    assert digest == "1" * 64
    assert config == {"config": True}
    assert selection == {"selection": True}

    monkeypatch.setattr(
        controller,
        "summarize_v19_epochs",
        lambda *_args, **_kwargs: {**screen, "continuation_authorized": False},
    )
    with pytest.raises(V19ExtensionViolation, match="differs from exact recomputation"):
        _load_exact_screen_report(tmp_path / "config.yaml", screen_path)


def test_prepare_requires_exact_current_clean_source(monkeypatch: pytest.MonkeyPatch) -> None:
    screen = {**_screen_decision(), "source_provenance": _source()}
    monkeypatch.setattr(
        controller,
        "_load_exact_screen_report",
        lambda *_args: (screen, "c" * 64, {}, {}),
    )
    changed = _source()
    changed["head_commit"] = "d" * 40
    with pytest.raises(V19ExtensionViolation, match="differs from the exact V19 screen"):
        prepare_extension_launch("config", "screen", current_provenance=changed)
    dirty = _source()
    dirty["is_clean"] = False
    with pytest.raises(V19ExtensionViolation, match="clean committed source"):
        prepare_extension_launch("config", "screen", current_provenance=dirty)


def test_extension_epoch_normalizes_only_the_isolated_namespace_for_strict_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    metadata_path = tmp_path / "epoch_005" / "metadata.json"
    raw = {"output_namespace": EXTENSION_NAMESPACE, "source_provenance": _source()}
    seen: dict[str, Any] = {}
    monkeypatch.setattr(controller, "_load_json_strict", lambda _path: (raw, "a" * 64))

    def validate(epoch: int, artifact: dict, _contract: dict, **kwargs: Any) -> dict:
        seen.update({"epoch": epoch, "artifact": artifact, **kwargs})
        return {
            "epoch": epoch,
            "history": [],
            "initialization_provenance": {},
            "zero_output_equivalence": {},
            "metrics": {},
            "signed_x_state_sha256": "b" * 64,
            "path": kwargs["path"],
            "artifact_sha256": kwargs["artifact_sha256"],
        }

    monkeypatch.setattr(controller, "_validate_epoch_artifact", validate)
    monkeypatch.setattr(
        controller,
        "_checkpoint_hashes",
        lambda *_args: {
            "adapter_sha256": "c" * 64,
            "metadata_sha256": "a" * 64,
            "optimizer_sha256": "d" * 64,
        },
    )
    row = _validate_extension_epoch(5, metadata_path, {}, _source())
    assert seen["artifact"]["output_namespace"] == controller.OUTPUT_NAMESPACE
    assert raw["output_namespace"] == EXTENSION_NAMESPACE
    assert row["checkpoint_artifact_hashes"]["metadata_sha256"] == "a" * 64

    raw["output_namespace"] = controller.OUTPUT_NAMESPACE
    with pytest.raises(V19ExtensionViolation, match="isolated output namespace"):
        _validate_extension_epoch(5, metadata_path, {}, _source())


def _pair_metrics(*, full: bool, partial_score: int = 8) -> dict[str, Any]:
    sides = 12 if full else partial_score
    units = 6 if full else 2
    minimum = 0.5 if full else -0.5
    return {
        "full_vocab_units": units,
        "full_vocab_sides": sides,
        "candidate_units": units,
        "candidate_sides": sides,
        "mean_full_vocab_margin": 1.0 if full else -0.1,
        "minimum_full_vocab_margin": minimum,
        "mean_candidate_margin": 1.0 if full else -0.1,
        "minimum_candidate_margin": minimum,
    }


def _color_metrics() -> dict[str, Any]:
    return _pair_metrics(full=True)


def _final_selector_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    fork_epoch: int | None = None,
    repeated_pre_gate_epoch: int | None = None,
) -> None:
    source = _source()
    selected_epoch = 4
    selected_checkpoint = tmp_path / "original" / "epoch_004"
    extension_root = tmp_path / EXTENSION_NAMESPACE
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
        "expected_extension_epochs": list(range(5, 13)),
        "source_provenance": source,
        "config_path": CONFIG_PATH,
        "config_hash": "26e2e91e5daf",
        "screen_report_path": "screen.json",
        "screen_report_sha256": "d" * 64,
    }
    selection_policy = {
        "continuation_requires": {
            "mirror_minimum_full_vocab_sides": 8,
            "mirror_minimum_full_vocab_units": 2,
        }
    }
    screen_epochs = []
    for epoch in range(1, 5):
        screen_epochs.append(
            {
                "epoch": epoch,
                "checkpoint_metadata_path": f"original/epoch_{epoch:03d}/metadata.json",
                "checkpoint_metadata_sha256": str(epoch) * 64,
                "signed_x_state_sha256": f"{epoch:x}" * 64,
                "color": _color_metrics(),
                "mirror": _pair_metrics(full=False),
            }
        )
    screen = {"selection_policy": selection_policy, "epochs": screen_epochs}
    history = [{"epoch": epoch} for epoch in range(1, 13)]
    selected_metadata = {
        "history": deepcopy(history[:4]),
        "initialization_provenance": {"source": "v18"},
        "signed_x_scene_residual_zero_output_equivalence": {"verified": True},
    }

    def extension_row(epoch: int, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
        epoch_history = deepcopy(history[:epoch])
        if fork_epoch == epoch:
            epoch_history[0] = {"epoch": 999}
        state = f"{epoch:x}" * 64
        if repeated_pre_gate_epoch == epoch:
            state = f"{epoch - 1:x}" * 64
        return {
            "epoch": epoch,
            "path": str(extension_root / f"epoch_{epoch:03d}" / "metadata.json"),
            "artifact_sha256": f"{epoch:x}" * 64,
            "signed_x_state_sha256": state,
            "history": epoch_history,
            "initialization_provenance": {"source": "v18"},
            "zero_output_equivalence": {"verified": True},
            "metrics": {
                "color": _color_metrics(),
                "mirror": _pair_metrics(full=epoch == 12, partial_score=min(11, epoch + 3)),
            },
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
    monkeypatch.setattr(controller, "_validate_config", lambda _config: {})
    monkeypatch.setattr(controller, "artifact_root", lambda _config, _kind: tmp_path)
    monkeypatch.setattr(
        controller,
        "_load_json_strict",
        lambda _path: (selected_metadata, selected_hashes["metadata_sha256"]),
    )
    monkeypatch.setattr(controller, "_checkpoint_hashes", lambda *_args: selected_hashes)
    monkeypatch.setattr(controller, "_validate_extension_epoch", extension_row)
    monkeypatch.setattr(controller, "_file_sha256", lambda *_args: "f" * 64)


def test_final_selector_validates_cumulative_branch_and_unlocks_only_at_full_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _final_selector_fixture(tmp_path, monkeypatch)
    report = select_final_extension(tmp_path / "launch.json", current_provenance=_source())
    assert report["epoch_count"] == TARGET_OPTIMIZER_UPDATE
    assert report["selected_epoch"] == 12
    assert report["conditional_limit_reached"] is True
    assert report["continuation_authorized"] is False
    assert report["full_teacher_gate_passed"] is True
    assert report["greedy_audit_authorized"] is True
    assert report["decision"] == "full_teacher_gate_passed_greedy_audit_allowed"
    assert report["cumulative_update_evidence"]["extension_history_prefixes_exact"] is True


def test_final_selector_rejects_history_fork_and_pre_gate_state_repeat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _final_selector_fixture(tmp_path, monkeypatch, fork_epoch=7)
    with pytest.raises(V19ExtensionViolation, match="forks or rewrites cumulative history"):
        select_final_extension(tmp_path / "launch.json", current_provenance=_source())

    monkeypatch.undo()
    _final_selector_fixture(tmp_path, monkeypatch, repeated_pre_gate_epoch=7)
    with pytest.raises(V19ExtensionViolation, match="before a full-teacher plateau"):
        select_final_extension(tmp_path / "launch.json", current_provenance=_source())


def test_controller_has_no_model_tensor_or_environment_data_imports() -> None:
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
    assert "load_file(" not in text
