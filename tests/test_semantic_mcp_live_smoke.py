from __future__ import annotations

from pathlib import Path

import pytest

from semantic_3d_chat.evaluation.semantic_mcp_live_smoke import (
    semantic_server_parameters,
    validate_live_refresh,
)
from semantic_3d_chat.mcp_server.server import ToolResponse


def _receipt(version: int, marker: str, *, yaw: float, depth: int) -> dict[str, object]:
    payload: dict[str, object] = {
        "success": True,
        "error_code": None,
        "scene_id": "scene_000001",
        "seed": 17,
        "scene_version": version,
        "position_m": [0.0, 0.0, 0.0],
        "camera_position_m": [0.0, 0.0, 1.2],
        "body_yaw_degrees": yaw,
        "camera_yaw_degrees": yaw,
        "pitch_degrees": 0.0,
        "linear_velocity_xy_m": [0.0, 0.0],
        "angular_velocity_degrees": 0.0,
        "collision": False,
        "last_movement_delta_m": [0.0, 0.0, 0.0],
        "distance_moved": 0.0,
        "turn_degrees": 0.0,
        "scan_coverage": 0.1 * version,
        "scan_count": version,
        "visible_voxels": 0,
        "valid_depth_pixels": depth,
        "observation_id": None if version == 0 else f"o_{version:06d}",
        "clearance_m": None,
        "action_count": version,
        "stopped": False,
        "schema": "semantic_3d_chat.scene_prefix_binding.v2",
        "map_version": version,
        "source_voxels": 100 + version,
        "processed_voxels": 100 + version,
    }
    fields = (
        "map_sha256",
        "scene_prefix_sha256",
        "scene_control_signature_sha256",
        "binding_sha256",
        "active_prefix_sha256",
        "robot_state_sha256",
        "robot_tokens_sha256",
        "active_binding_sha256",
    )
    for index, field in enumerate(fields):
        payload[field] = f"{(int(marker, 16) + index) % 16:x}" * 64
    payload["robot_state_encoder_sha256"] = "f" * 64
    return payload


def test_live_refresh_contract_tracks_map_prefix_controller_and_robot_tokens() -> None:
    evidence = validate_live_refresh(
        _receipt(0, "1", yaw=0.0, depth=0),
        _receipt(1, "2", yaw=0.0, depth=123),
        _receipt(2, "3", yaw=15.0, depth=117),
        turn_degrees=15.0,
    )

    assert evidence["scan_map_version"] == 1
    assert evidence["turn_map_version"] == 2
    assert evidence["scan_changed_scene_control_signature_sha256"] is True
    assert evidence["turn_changed_scene_control_signature_sha256"] is True


def test_live_refresh_rejects_an_unchanged_scene_prefix() -> None:
    initial = _receipt(0, "1", yaw=0.0, depth=0)
    scanned = _receipt(1, "2", yaw=0.0, depth=123)
    scanned["scene_prefix_sha256"] = initial["scene_prefix_sha256"]

    with pytest.raises(AssertionError, match="continuous refresh hashes did not change"):
        validate_live_refresh(
            initial,
            scanned,
            _receipt(2, "3", yaw=15.0, depth=117),
            turn_degrees=15.0,
        )


def test_v2_tool_schema_accepts_safe_controller_signature_hash() -> None:
    validated = ToolResponse.model_validate(_receipt(1, "2", yaw=0.0, depth=123))
    assert validated.scene_control_signature_sha256 == "4" * 64


def test_semantic_stdio_parameters_are_explicit_offline_and_controlled(
    tmp_path: Path,
) -> None:
    parameters = semantic_server_parameters(
        python_executable=Path(".venv-gemma4/bin/python").absolute(),
        config=tmp_path / "runtime.yaml",
        scene_id="scene_000001",
        base_checkpoint=tmp_path / "base",
        control_checkpoint=tmp_path / "control",
        control_runtime_config=tmp_path / "control.yaml",
        runtime_asset=tmp_path / "s_000001.blend",
        robot_state_checkpoint=tmp_path / "robot",
        persistent_map=tmp_path / "map.npz",
        audit_report=tmp_path / "audit.json",
    )

    assert parameters.args[-2:] == ["--transport", "stdio"]
    assert "--control-checkpoint" in parameters.args
    assert "--robot-state-checkpoint" in parameters.args
    assert parameters.env is not None
    assert parameters.env["TRANSFORMERS_OFFLINE"] == "1"
    assert parameters.env["HF_HUB_OFFLINE"] == "1"
