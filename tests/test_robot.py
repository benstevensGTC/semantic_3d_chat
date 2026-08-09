from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from mcp.server.mcpserver.exceptions import ToolError

from semantic_3d_chat.mcp_server.server import build_server
from semantic_3d_chat.robot.collision import NumericCollisionMap
from semantic_3d_chat.robot.simulator import EmbodiedCameraSimulator
from semantic_3d_chat.robot.state_encoder import (
    ROBOT_STATE_FEATURE_DIM,
    NumericRobotState,
    RobotStateEncoder,
    append_robot_state_tokens,
    robot_state_vector,
)
from semantic_3d_chat.robot.tools import RobotToolController, tool_schemas


def _numeric_map(path: Path) -> None:
    wall_values = np.linspace(-2.5, 2.5, 51, dtype=np.float32)
    walls = []
    for value in wall_values:
        walls.extend(
            [
                (-3.0, value, 0.7),
                (3.0, value, 0.7),
                (value, -2.5, 0.7),
                (value, 2.5, 0.7),
            ]
        )
    visible = [
        (x, y, z)
        for y in (1.0, 1.2, 1.4)
        for x in np.linspace(-0.5, 0.5, 11)
        for z in (0.8, 1.0, 1.2, 1.4)
    ]
    centers = np.asarray([*walls, (0.6, 0.0, 0.7), *visible], dtype=np.float32)
    rgb = np.tile(np.asarray([[64.0, 128.0, 192.0]], dtype=np.float32), (len(centers), 1))
    path.parent.mkdir(parents=True)
    np.savez_compressed(
        path,
        centers_world=centers,
        mean_rgb=rgb,
        observation_count=np.ones(len(centers), dtype=np.int32),
    )


def _config(tmp_path: Path) -> dict:
    _numeric_map(tmp_path / "maps" / "scene_000001" / "voxel_map.npz")
    return {
        "seed": 17,
        "paths": {"data_root": str(tmp_path)},
        "scene": {"room_size_m": [6.0, 5.0, 3.0]},
        "render": {"resolution": [64, 48], "horizontal_fov_degrees": 72.0},
        "robot": {
            "radius_m": 0.20,
            "camera_height_m": 1.20,
            "max_move_m": 0.50,
            "max_move_to_m": 1.0,
            "max_turn_degrees": 45.0,
            "max_look_delta_degrees": 30.0,
            "max_camera_yaw_offset_degrees": 60.0,
            "max_pitch_degrees": 45.0,
            "collision_z_min_m": 0.12,
            "collision_z_max_m": 1.80,
            "surface_padding_m": 0.02,
            "scan_depth_min_m": 0.10,
            "scan_depth_max_m": 6.0,
            "initial_position_xy_m": [0.0, 0.0],
            "history_length": 16,
        },
    }


def test_collision_checks_complete_segment() -> None:
    collision = NumericCollisionMap(
        np.asarray([[0.5, 0.0]], dtype=np.float32),
        room_min_xy_m=(-2.0, -2.0),
        room_max_xy_m=(2.0, 2.0),
        robot_radius_m=0.2,
        surface_padding_m=0.01,
    )
    assert not collision.point_check((0.0, 0.0)).collision
    assert collision.segment_check((0.0, 0.0), (1.0, 0.0)).collision
    assert not collision.segment_check((0.0, 0.0), (0.0, 1.0)).collision
    assert collision.point_check((1.9, 0.0)).collision


def test_robot_actions_collision_scan_and_reset(tmp_path: Path) -> None:
    simulator = EmbodiedCameraSimulator(_config(tmp_path), "scene_000001")
    initial = simulator.get_robot_state()
    assert initial["success"] and initial["position_m"] == [0.0, 0.0, 0.0]

    assert simulator.turn(46)["error_code"] == "E_LIMIT"
    assert simulator.turn(30)["success"]
    looked = simulator.look(10, -5)
    assert looked["success"] and looked["pitch_degrees"] == -5

    simulator.reset_scene("scene_000001", 17)
    moved = simulator.move_forward(0.25)
    assert moved["success"] and np.isclose(moved["distance_moved"], 0.25)

    simulator.reset_scene("scene_000001", 17)
    blocked = simulator.move_to(0.6, 0.0)
    assert not blocked["success"] and blocked["collision"]
    assert blocked["position_m"] == [0.0, 0.0, 0.0]

    scanned = simulator.scan()
    assert scanned["success"] and scanned["visible_voxels"] > 0
    assert scanned["scene_version"] == 1 and scanned["scan_coverage"] > 0
    scan_file = tmp_path / "robot" / "scene_000001" / "scans" / "o_000001.npz"
    with np.load(scan_file, allow_pickle=False) as archive:
        assert set(archive.files) == {
            "rgb",
            "depth_m",
            "intrinsics",
            "camera_to_world",
            "visible_voxel_indices",
        }
        assert archive["rgb"].shape == (48, 64, 3)

    assert simulator.stop()["stopped"]
    assert simulator.move_forward(0.1)["error_code"] == "E_STOPPED"
    reset = simulator.reset_scene("scene_000001", 18)
    assert reset["success"] and not reset["stopped"] and reset["scene_version"] == 0


def test_failed_reset_is_atomic_and_seed_is_integral(tmp_path: Path) -> None:
    config = _config(tmp_path)
    simulator = EmbodiedCameraSimulator(config, "scene_000001")
    original_collision = simulator.collision_map
    original_scanner = simulator.scanner
    original_state = simulator.get_robot_state()

    malformed = tmp_path / "maps" / "scene_000002" / "voxel_map.npz"
    malformed.parent.mkdir(parents=True)
    np.savez_compressed(
        malformed,
        centers_world=np.asarray([[0.5, 0.5, 0.7]], dtype=np.float32),
    )
    rejected = simulator.reset_scene("scene_000002", 18)
    assert rejected["error_code"] == "E_SCENE_UNAVAILABLE"
    assert simulator.collision_map is original_collision
    assert simulator.scanner is original_scanner
    assert simulator.get_robot_state()["scene_id"] == original_state["scene_id"]
    assert simulator.reset_scene("scene_000001", 1.5)["error_code"] == "E_NUMERIC"


def test_strict_direct_tool_schema(tmp_path: Path) -> None:
    config = _config(tmp_path)
    controller = RobotToolController(EmbodiedCameraSimulator(config, "scene_000001"))
    assert controller.dispatch({"tool": "turn", "arguments": {"angle_degrees": 10}})["success"]
    invalid = controller.dispatch(
        {"tool": "turn", "arguments": {"angle_degrees": 10, "extra": 1}}
    )
    assert invalid["error_code"] == "E_SCHEMA"
    assert controller.dispatch({"tool": "unavailable", "arguments": {}})["error_code"] == "E_TOOL"
    schemas = {item["name"]: item["inputSchema"] for item in tool_schemas(config)}
    assert set(schemas) == {
        "get_robot_state",
        "look",
        "turn",
        "move_forward",
        "move_backward",
        "move_to",
        "scan",
        "stop",
        "reset_scene",
    }
    assert schemas["turn"]["additionalProperties"] is False
    assert schemas["turn"]["properties"]["angle_degrees"]["maximum"] == 45.0


def test_robot_state_encoder_is_continuous_and_composable() -> None:
    state = NumericRobotState(
        position_m=(0.1, -0.2, 0.0),
        body_yaw_degrees=30,
        camera_yaw_degrees=40,
        pitch_degrees=-5,
        linear_velocity_xy_m=(0.1, 0.0),
        angular_velocity_degrees=10,
        collision=False,
        last_movement_delta_m=(0.1, 0.0, 0.0),
        scan_coverage=0.2,
        stopped=False,
    )
    features = robot_state_vector(
        state, torch.tensor([-3.0, -2.5, 0.0]), torch.tensor([3.0, 2.5, 3.0])
    )
    assert features.shape == (ROBOT_STATE_FEATURE_DIM,)
    encoder = RobotStateEncoder(32, hidden_dim=24, token_count=4)
    tokens = encoder(features)
    assert tokens.shape == (1, 4, 32) and torch.isfinite(tokens).all()
    prefix = torch.zeros(1, 10, 32)
    assert append_robot_state_tokens(prefix, tokens).shape == (1, 14, 32)


def test_official_mcp_server_schemas_and_structured_result(tmp_path: Path) -> None:
    simulator = EmbodiedCameraSimulator(_config(tmp_path), "scene_000001")
    server = build_server(simulator)

    async def exercise() -> None:
        tools = await server.list_tools()
        by_name = {tool.name: tool for tool in tools}
        assert set(by_name) == set(TOOL_NAMES)
        assert by_name["move_forward"].input_schema["properties"]["distance_meters"][
            "type"
        ] == "number"
        assert by_name["move_forward"].input_schema["properties"]["distance_meters"][
            "maximum"
        ] == 0.5
        assert by_name["turn"].input_schema["additionalProperties"] is False
        state_before = simulator.get_robot_state()
        with pytest.raises(ToolError):
            await server.call_tool("turn", {"angle_degrees": 10.0, "extra": 1})
        assert simulator.get_robot_state()["body_yaw_degrees"] == state_before[
            "body_yaw_degrees"
        ]
        result = await server.call_tool("turn", {"angle_degrees": 15.0})
        assert result.structured_content is not None
        payload = result.structured_content
        assert payload["success"] is True
        assert payload["body_yaw_degrees"] == 15.0
        encoded = json.dumps(payload)
        for prohibited in ("category", "caption", "relationship", "object_name"):
            assert prohibited not in encoded

    asyncio.run(exercise())


TOOL_NAMES = (
    "get_robot_state",
    "look",
    "turn",
    "move_forward",
    "move_backward",
    "move_to",
    "scan",
    "stop",
    "reset_scene",
)
