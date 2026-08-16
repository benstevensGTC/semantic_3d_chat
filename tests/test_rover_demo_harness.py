from __future__ import annotations

import argparse
import os
from pathlib import Path

import pytest

from semantic_3d_chat.robot import rover_demo

ROOT = Path(__file__).parents[1]


def _args(**overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "config": "configs/runtime/embodied_live.yaml",
        "control_runtime_config": "configs/runtime/gemma4_v56_question_control.yaml",
        "scene": "scene_000001",
        "base_checkpoint": "data_gemma4/runtime/checkpoints/gemma4_v54_release_v1",
        "control_checkpoint": (
            "data_gemma4/runtime/checkpoints/gemma4_v75_nll_control_release_v1"
        ),
        "runtime_asset": None,
        "robot_state_checkpoint": "data_gemma4/checkpoints/robot_state_numeric_v1",
        "navigation_checkpoint": "data_gemma4/checkpoints/navigation_policy_v3",
        "map": None,
        "map_visual": None,
        "scan_visual": "reports/gemma4/figures/scan_montage.png",
        "audit_report": None,
        "host": "127.0.0.1",
        "port": 8770,
        "no_open": False,
        "check": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_rover_demo_derives_only_opaque_scene_paths() -> None:
    settings = rover_demo._settings(_args(scene="scene_000123", no_open=True))

    assert settings.runtime_asset == (
        ROOT / "data/runtime_assets/scene_000123/s_000123.blend"
    ).resolve()
    assert settings.map_path == (
        ROOT / "data_gemma4/maps/scene_000123/voxel_map.npz"
    ).resolve()
    assert settings.map_visual == (
        ROOT / "reports/gemma4/figures/scene_000123/map_rgb.png"
    ).resolve()
    assert settings.open_browser is False
    assert settings.audit_output == (
        ROOT / "reports/gemma4/metrics/practical_rover_access_scene_000123.json"
    ).resolve()


@pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.2", "example.com"])
def test_rover_demo_refuses_non_loopback_bind(host: str) -> None:
    with pytest.raises(ValueError, match="non-loopback"):
        rover_demo._settings(_args(host=host))


def test_integrated_launcher_forwards_the_exact_backend_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from semantic_3d_chat.robot import practical_rover

    sentinel = object()
    observed: dict[str, object] = {}

    def fake_factory(**kwargs: object) -> object:
        observed.update(kwargs)
        return sentinel

    monkeypatch.setattr(practical_rover, "build_local_practical_rover", fake_factory)
    settings = rover_demo._settings(_args(no_open=True))
    persistent_map = tmp_path / "fresh-session" / "semantic_map.npz"

    assert rover_demo._build_session(settings, persistent_map=persistent_map) is sentinel
    assert observed == {
        "config": settings.config,
        "control_config": settings.control_runtime_config,
        "scene_id": settings.scene_id,
        "base_checkpoint": settings.base_checkpoint,
        "control_checkpoint": settings.control_checkpoint,
        "runtime_asset": settings.runtime_asset,
        "robot_state_checkpoint": settings.robot_state_checkpoint,
        "navigation_checkpoint": settings.navigation_checkpoint,
        "persistent_map": persistent_map.resolve(),
        "audit_output": settings.audit_output,
        "initial_scan": False,
    }


def test_integrated_launcher_refuses_an_existing_persistent_map(
    tmp_path: Path,
) -> None:
    persistent_map = tmp_path / "semantic_map.npz"
    persistent_map.write_bytes(b"prior-session")

    with pytest.raises(FileExistsError, match="cannot inherit"):
        rover_demo._build_session(
            rover_demo._settings(_args(no_open=True)),
            persistent_map=persistent_map,
        )


def test_model_free_preflight_covers_room_scan_map_and_rover_assets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(rover_demo, "PROJECT_ROOT", tmp_path)
    files = {
        "config": tmp_path / "configs/embodied.yaml",
        "control": tmp_path / "configs/control.yaml",
        "asset": tmp_path / "data/runtime_assets/scene_000001/s_000001.blend",
        "map": tmp_path / "data_gemma4/maps/scene_000001/voxel_map.npz",
        "map_visual": tmp_path / "reports/gemma4/figures/scene_000001/map_rgb.png",
        "scan_visual": tmp_path / "reports/gemma4/figures/scan_montage.png",
    }
    for path in files.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"prepared")
    inventories = {
        "base": ("adapter.safetensors", "runtime_metadata.json"),
        "control_checkpoint": ("control.safetensors", "runtime_metadata.json"),
        "robot_state": ("state.safetensors", "runtime_metadata.json"),
        "navigation": ("policy.safetensors", "runtime_metadata.json"),
    }
    for directory, names in inventories.items():
        root = tmp_path / directory
        root.mkdir()
        for name in names:
            (root / name).write_bytes(b"prepared")
    blender = tmp_path / "blender"
    blender.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    blender.chmod(0o755)
    monkeypatch.setenv("BLENDER", str(blender))
    monkeypatch.setattr(rover_demo, "_module_available", lambda _name: True)
    monkeypatch.setattr(
        rover_demo,
        "load_config",
        lambda _path: {"scene": {"room_size_m": [6.0, 5.0, 3.0]}},
    )
    monkeypatch.setattr(
        rover_demo,
        "_backend_preflight",
        lambda _settings: {
            "ready": True,
            "continuous_map": str(files["map"]),
            "navigation_checkpoint_sha256": "d" * 64,
            "gemma_runtime_binding_sha256": "e" * 64,
        },
    )
    settings = rover_demo.RoverDemoSettings(
        config=files["config"],
        control_runtime_config=files["control"],
        scene_id="scene_000001",
        base_checkpoint=tmp_path / "base",
        control_checkpoint=tmp_path / "control_checkpoint",
        runtime_asset=files["asset"],
        robot_state_checkpoint=tmp_path / "robot_state",
        navigation_checkpoint=tmp_path / "navigation",
        map_path=files["map"],
        map_visual=files["map_visual"],
        scan_visual=files["scan_visual"],
        audit_output=tmp_path / "reports/audit.json",
        host="127.0.0.1",
        port=8770,
        open_browser=False,
    )

    result = rover_demo.check_rover_demo(settings)

    assert result["passed"] is True
    assert result["loads_model"] is False
    assert result["runs_blender"] is False
    assert result["starts_server"] is False
    assert result["local_inference"] is True
    assert result["human_visuals_are_model_inputs"] is False
    assert result["starts_from_precomputed_static_map"] is True
    assert result["inherits_prior_persistent_map"] is False
    assert result["initial_rgbd_scan"] is False
    assert result["runtime_rgbd_scans"] is False
    assert result["task_trained_navigation"] is True
    assert result["navigation_checkpoint_sha256"] == "d" * 64
    assert result["gemma_runtime_binding_sha256"] == "e" * 64
    assert result["high_level_natural_language_only"] is True
    assert result["untrained_json_backend_enabled"] is False
    assert result["environmental_text_inputs"] == []
    assert result["room_size_m"] == [6.0, 5.0, 3.0]


def test_one_command_launcher_authenticates_before_starting_server() -> None:
    launcher = ROOT / "scripts/run_local_rover_demo.sh"
    source = launcher.read_text(encoding="utf-8")

    assert os.access(launcher, os.X_OK)
    assert "prepare_demo_runtime.py --check" in source
    assert "check_demo_artifacts.py --fast" in source
    assert "TRANSFORMERS_OFFLINE=1" in source
    assert "HF_HUB_OFFLINE=1" in source
    module = "-m semantic_3d_chat.robot.rover_demo"
    assert module in source
    assert source.index("check_demo_artifacts.py --fast") < source.rindex(module)


def test_make_exposes_local_ui_and_separate_memory_safe_mcp_target() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")

    assert "ROVER_DEMO_HOST ?= 127.0.0.1" in makefile
    assert "ROVER_DEMO_PORT ?= 8770" in makefile
    assert "rover-demo-check:" in makefile
    assert "run_local_rover_demo.sh --check" in makefile
    assert "rover-demo:" in makefile
    assert "run_local_rover_demo.sh --scene" in makefile
    assert "rover-demo-mcp: rover-demo-check" in makefile
    assert "gemma4-embodied-mcp" in makefile


def test_server_setup_failure_closes_heavy_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from semantic_3d_chat.robot import rover_web_app

    class FakeSession:
        def __init__(self) -> None:
            self.closed = 0

        def close(self) -> None:
            self.closed += 1

    session = FakeSession()
    settings = rover_demo._settings(_args(no_open=True))
    monkeypatch.setattr(
        rover_demo,
        "load_config",
        lambda _path: {"scene": {"room_size_m": [6.0, 5.0, 3.0]}},
    )
    persistent_paths: list[Path] = []

    def fake_build(_settings: rover_demo.RoverDemoSettings, *, persistent_map: Path) -> FakeSession:
        assert not persistent_map.exists()
        persistent_paths.append(persistent_map)
        return session

    monkeypatch.setattr(rover_demo, "_build_session", fake_build)

    def fail_app(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("app setup failed")

    monkeypatch.setattr(rover_web_app, "create_rover_web_app", fail_app)
    with pytest.raises(RuntimeError, match="app setup failed"):
        rover_demo._serve(settings, {"passed": True})

    assert session.closed == 1
    assert len(persistent_paths) == 1
    assert not persistent_paths[0].parent.exists()
