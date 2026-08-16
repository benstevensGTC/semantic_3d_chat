from __future__ import annotations

import hashlib
import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from mcp import StdioServerParameters

from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.robot.conversation import ConversationalEmbodiedAgent
from semantic_3d_chat.robot.mcp_stdio_runtime import (
    MCPActionTransportError,
    MCPConversationRuntime,
    MCPStdioToolClient,
    validate_numeric_tool_receipt,
)
from semantic_3d_chat.robot.tools import TOOL_ARGUMENTS


def _sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(payload),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _receipt(
    version: int,
    scene_marker: str,
    robot_marker: str,
    *,
    yaw: float = 0.0,
    action_count: int = 0,
    stopped: bool = False,
) -> dict[str, Any]:
    scene_identity: dict[str, Any] = {
        "schema": "semantic_3d_chat.scene_prefix_binding.v2",
        "scene_id": "scene_000001",
        "map_version": version,
        "map_sha256": scene_marker * 64,
        "scene_prefix_sha256": f"{(int(scene_marker, 16) + 1) % 16:x}" * 64,
        "scene_control_signature_sha256": (
            f"{(int(scene_marker, 16) + 2) % 16:x}" * 64
        ),
        "source_voxels": 12,
        "processed_voxels": 12,
    }
    binding_sha256 = _sha256(scene_identity)
    active_identity = {
        **scene_identity,
        "binding_sha256": binding_sha256,
        "active_prefix_sha256": robot_marker * 64,
        "robot_state_sha256": f"{(int(robot_marker, 16) + 1) % 16:x}" * 64,
        "robot_tokens_sha256": f"{(int(robot_marker, 16) + 2) % 16:x}" * 64,
        "robot_state_encoder_sha256": "f" * 64,
    }
    return {
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
        "scan_coverage": version / 10.0,
        "scan_count": version,
        "visible_voxels": 12 if version else 0,
        "valid_depth_pixels": 12 if version else 0,
        "observation_id": None if version == 0 else f"o_{version:06d}",
        "clearance_m": None,
        "action_count": action_count,
        "stopped": stopped,
        **active_identity,
        "active_binding_sha256": _sha256(active_identity),
    }


def _config(tmp_path: Path) -> dict[str, Any]:
    return {
        "seed": 17,
        "paths": {"data_root": str(tmp_path / "data")},
        "scene": {"room_size_m": [6.0, 5.0, 3.0]},
        "render": {"resolution": [16, 16], "horizontal_fov_degrees": 72.0},
        "robot": {
            "auto_scan_after_motion": True,
            "radius_m": 0.2,
            "camera_height_m": 1.2,
            "max_move_m": 0.5,
            "max_move_to_m": 1.0,
            "max_turn_degrees": 45.0,
            "max_look_delta_degrees": 30.0,
            "max_camera_yaw_offset_degrees": 60.0,
            "max_pitch_degrees": 45.0,
            "collision_z_min_m": 0.12,
            "collision_z_max_m": 1.8,
            "surface_padding_m": 0.02,
        },
    }


def _write_map(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        centers_world=np.asarray(
            [[-2.9, -2.4, 0.7], [2.9, 2.4, 0.7], [1.2, 1.0, 0.7]],
            dtype=np.float32,
        ),
        mean_rgb=np.full((3, 3), 127.0, dtype=np.float32),
        observation_count=np.ones(3, dtype=np.int32),
    )


class _FakeToolClient:
    tool_names = frozenset(TOOL_ARGUMENTS)

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.responses = [
            _receipt(0, "1", "4"),
            _receipt(1, "2", "5", action_count=1),
            _receipt(2, "3", "6", yaw=15.0, action_count=2),
            _receipt(2, "3", "7", yaw=15.0, action_count=3, stopped=True),
        ]

    def call_tool(self, name: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
        self.calls.append((name, dict(arguments)))
        return self.responses.pop(0)


class _UnusedTextEncoder:
    output_dim = 1

    def encode_queries(self, queries: list[str]) -> np.ndarray:
        raise AssertionError(f"numeric smoke unexpectedly grounded text: {queries!r}")


def test_conversation_agent_actions_cross_mcp_and_refresh_binding(tmp_path: Path) -> None:
    base_map = tmp_path / "maps" / "scene_000001" / "voxel_map.npz"
    persistent_map = tmp_path / "runtime" / "scene_000001" / "semantic_map.npz"
    _write_map(base_map)
    client = _FakeToolClient()
    runtime = MCPConversationRuntime(
        client,
        _config(tmp_path),
        base_map_path=base_map,
        persistent_map_path=persistent_map,
    )
    agent = ConversationalEmbodiedAgent(
        runtime,
        _UnusedTextEncoder(),
        room_size_m=[6.0, 5.0, 3.0],
        feature_start=0,
        feature_dim=1,
    )

    initial = runtime.prefix_binding()
    scanned = agent.handle("Scan.")
    turned = agent.handle("Turn right 15 degrees.")
    stopped = agent.handle("Stop.")

    assert [name for name, _arguments in client.calls] == [
        "get_robot_state",
        "scan",
        "turn",
        "stop",
    ]
    assert runtime.binding_refresh_count == 4
    assert scanned["prefix_binding"]["map_version"] == 1
    assert turned["prefix_binding"]["map_version"] == 2
    assert stopped["prefix_binding"]["map_version"] == 2
    assert scanned["prefix_binding"]["scene_prefix_sha256"] != initial[
        "scene_prefix_sha256"
    ]
    assert turned["prefix_binding"]["scene_prefix_sha256"] != scanned[
        "prefix_binding"
    ]["scene_prefix_sha256"]
    assert stopped["prefix_binding"]["active_binding_sha256"] != turned[
        "prefix_binding"
    ]["active_binding_sha256"]
    assert runtime.simulator.state.body_yaw_degrees == 15.0
    assert runtime.simulator.state.stopped is True
    assert all(result["environmental_text_inputs"] == [] for result in (scanned, turned, stopped))


def test_numeric_receipt_rejects_extra_semantic_fields_and_corrupt_bindings() -> None:
    extra = _receipt(0, "1", "4")
    extra["category"] = "fixture"
    with pytest.raises(MCPActionTransportError, match="numeric schema"):
        validate_numeric_tool_receipt(extra, require_continuous_binding=True)

    corrupt = _receipt(0, "1", "4")
    corrupt["binding_sha256"] = "0" * 64
    with pytest.raises(MCPActionTransportError, match="binding hash"):
        validate_numeric_tool_receipt(corrupt, require_continuous_binding=True)


def _write_numeric_server_config(tmp_path: Path) -> Path:
    data_root = tmp_path / "server_data"
    map_path = data_root / "maps" / "scene_000001" / "voxel_map.npz"
    _write_map(map_path)
    config = _config(tmp_path)
    config["paths"]["data_root"] = str(data_root)
    config["paths"]["reports_root"] = str(tmp_path / "reports")
    config["robot"]["auto_scan_after_motion"] = False
    config["robot"].update(
        {
            "initial_position_xy_m": [0.0, 0.0],
            "scan_depth_min_m": 0.1,
            "scan_depth_max_m": 6.0,
            "history_length": 16,
        }
    )
    destination = tmp_path / "stdio_config.json"
    destination.write_text(json.dumps(config), encoding="utf-8")
    return destination


def test_official_stdio_client_keeps_session_alive_for_multiple_calls(tmp_path: Path) -> None:
    environment = dict(os.environ)
    source_root = str(PROJECT_ROOT / "src")
    inherited = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        source_root if not inherited else os.pathsep.join((source_root, inherited))
    )
    parameters = StdioServerParameters(
        command=sys.executable,
        args=[
            "-m",
            "semantic_3d_chat.mcp_server.server",
            "--config",
            str(_write_numeric_server_config(tmp_path)),
            "--scene",
            "scene_000001",
            "--transport",
            "stdio",
        ],
        cwd=PROJECT_ROOT,
        env=environment,
    )

    with MCPStdioToolClient(
        parameters,
        startup_timeout_seconds=30.0,
        call_timeout_seconds=30.0,
        read_timeout_seconds=30.0,
    ) as client:
        assert client.tool_names == frozenset(TOOL_ARGUMENTS)
        before = client.call_tool("get_robot_state", {})
        turned = client.call_tool("turn", {"angle_degrees": 15.0})
        after = client.call_tool("get_robot_state", {})

    assert before["body_yaw_degrees"] == 0.0
    assert turned["success"] is True
    assert turned["body_yaw_degrees"] == 15.0
    assert after["body_yaw_degrees"] == 15.0
