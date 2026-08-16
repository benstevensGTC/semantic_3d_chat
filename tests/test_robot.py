from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from mcp.server.mcpserver.exceptions import ToolError

from semantic_3d_chat.chat.file_audit import FileAccessAudit
from semantic_3d_chat.mcp_server import server as mcp_server
from semantic_3d_chat.mcp_server.server import (
    _run_semantic_server_lifetime,
    _runtime_asset_preflight,
    _validate_navigation_safety_metadata,
    _validate_semantic_config_isolation,
    build_server,
)
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


def test_configurable_initial_body_yaw_is_applied_on_every_reset(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config["robot"]["initial_body_yaw_degrees"] = 30.0
    simulator = EmbodiedCameraSimulator(config, "scene_000001")

    assert simulator.get_robot_state()["body_yaw_degrees"] == pytest.approx(30.0)
    moved = simulator.move_forward(0.25)
    assert moved["success"] is True
    assert moved["position_m"][:2] == pytest.approx([-0.125, 0.21650635])

    simulator.turn(-15.0)
    reset = simulator.reset_scene("scene_000001", 18)
    assert reset["success"] is True
    assert reset["body_yaw_degrees"] == pytest.approx(30.0)


def test_invalid_initial_body_yaw_fails_before_scene_commit(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config["robot"]["initial_body_yaw_degrees"] = float("inf")

    with pytest.raises(ValueError, match="initial_body_yaw_degrees"):
        EmbodiedCameraSimulator(config, "scene_000001")


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


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS regression requires Apple Silicon"
)
def test_robot_state_vector_defaults_to_cpu_with_mps_map_bounds() -> None:
    state = NumericRobotState(
        position_m=(0.0, 0.0, 0.0),
        body_yaw_degrees=0.0,
        camera_yaw_degrees=0.0,
        pitch_degrees=0.0,
        linear_velocity_xy_m=(0.0, 0.0),
        angular_velocity_degrees=0.0,
        collision=False,
        last_movement_delta_m=(0.0, 0.0, 0.0),
        scan_coverage=0.0,
        stopped=False,
    )
    features = robot_state_vector(
        state,
        torch.tensor([-3.0, -2.5, 0.0], device="mps"),
        torch.tensor([3.0, 2.5, 3.0], device="mps"),
    )

    assert features.device.type == "cpu"
    assert features.shape == (ROBOT_STATE_FEATURE_DIM,)
    assert torch.isfinite(features).all()


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
        assert {
            "schema",
            "map_version",
            "map_sha256",
            "scene_prefix_sha256",
            "scene_control_signature_sha256",
            "binding_sha256",
            "active_prefix_sha256",
            "robot_state_sha256",
            "robot_tokens_sha256",
            "robot_state_encoder_sha256",
            "active_binding_sha256",
        }.isdisjoint(payload)
        encoded = json.dumps(payload)
        for prohibited in ("category", "caption", "relationship", "object_name"):
            assert prohibited not in encoded

    asyncio.run(exercise())


class FakeRefreshingRuntime:
    """Binding-shaped wrapper used to exercise strict MCP compatibility."""

    def __init__(self, simulator: EmbodiedCameraSimulator) -> None:
        self.simulator = simulator

    def _binding(self, payload: dict) -> dict:
        map_version = int(payload["scene_version"])
        return {
            **payload,
            "schema": "semantic_3d_chat.scene_prefix_binding.v1",
            "map_version": map_version,
            "map_sha256": "1" * 64,
            "scene_prefix_sha256": "2" * 64,
            "source_voxels": 100,
            "processed_voxels": 50,
            "binding_sha256": "3" * 64,
            "active_prefix_sha256": "2" * 64,
            "robot_state_sha256": None,
            "robot_tokens_sha256": None,
            "robot_state_encoder_sha256": None,
            "active_binding_sha256": "4" * 64,
        }

    def get_robot_state(self) -> dict:
        return self._binding(self.simulator.get_robot_state())

    def look(self, yaw_delta_degrees: float, pitch_delta_degrees: float) -> dict:
        return self._binding(self.simulator.look(yaw_delta_degrees, pitch_delta_degrees))

    def turn(self, angle_degrees: float) -> dict:
        return self._binding(self.simulator.turn(angle_degrees))

    def move_forward(self, distance_meters: float) -> dict:
        return self._binding(self.simulator.move_forward(distance_meters))

    def move_backward(self, distance_meters: float) -> dict:
        return self._binding(self.simulator.move_backward(distance_meters))

    def move_to(self, x: float, y: float) -> dict:
        return self._binding(self.simulator.move_to(x, y))

    def scan(self) -> dict:
        return self._binding(self.simulator.scan())

    def stop(self) -> dict:
        return self._binding(self.simulator.stop())

    def reset_scene(self, scene_id: str, seed: int) -> dict:
        del scene_id, seed
        return self._binding(self.simulator.protocol_error("E_RESET_UNSUPPORTED"))


class FakeRefreshingRuntimeV2(FakeRefreshingRuntime):
    """Current refreshed-map receipt emitted by the production runtime."""

    def _binding(self, payload: dict) -> dict:
        return {
            **super()._binding(payload),
            "schema": "semantic_3d_chat.scene_prefix_binding.v2",
        }


def test_official_mcp_server_accepts_refreshing_prefix_bindings(tmp_path: Path) -> None:
    simulator = EmbodiedCameraSimulator(_config(tmp_path), "scene_000001")
    runtime = FakeRefreshingRuntime(simulator)
    server = build_server(runtime)

    async def exercise() -> None:
        result = await server.call_tool("turn", {"angle_degrees": 15.0})
        assert result.structured_content is not None
        payload = result.structured_content
        assert payload["schema"] == "semantic_3d_chat.scene_prefix_binding.v1"
        assert payload["scene_prefix_sha256"] == "2" * 64
        assert payload["active_binding_sha256"] == "4" * 64
        before = runtime.get_robot_state()
        reset = await server.call_tool(
            "reset_scene", {"scene_id": "scene_000001", "seed": 99}
        )
        assert reset.structured_content is not None
        assert reset.structured_content["success"] is False
        assert reset.structured_content["error_code"] == "E_RESET_UNSUPPORTED"
        after = runtime.get_robot_state()
        assert after["scene_id"] == before["scene_id"]
        assert after["scene_prefix_sha256"] == before["scene_prefix_sha256"]
        encoded = json.dumps(reset.structured_content)
        for prohibited in ("category", "caption", "relationship", "object_name"):
            assert prohibited not in encoded

    asyncio.run(exercise())


def test_official_mcp_server_accepts_current_v2_prefix_binding(tmp_path: Path) -> None:
    simulator = EmbodiedCameraSimulator(_config(tmp_path), "scene_000001")
    runtime = FakeRefreshingRuntimeV2(simulator)
    server = build_server(runtime)

    async def exercise() -> None:
        result = await server.call_tool("scan", {})
        assert result.structured_content is not None
        assert (
            result.structured_content["schema"]
            == "semantic_3d_chat.scene_prefix_binding.v2"
        )
        assert result.structured_content["map_version"] == 1

    asyncio.run(exercise())


def test_semantic_mcp_audit_remains_active_during_server_requests(tmp_path: Path) -> None:
    forbidden_root = tmp_path / "data" / "oracle"
    forbidden_file = forbidden_root / "hidden.json"
    forbidden_file.parent.mkdir(parents=True)
    forbidden_file.write_text("do not load", encoding="utf-8")
    continuous_map = tmp_path / "data_gemma4" / "features" / "scene_000001.npz"
    continuous_map.parent.mkdir(parents=True)
    continuous_map.write_bytes(b"continuous numeric payload")
    audit_report = tmp_path / "reports" / "embodied_mcp_file_access.json"
    audit = FileAccessAudit([forbidden_root], block_forbidden=True)
    lifecycle: list[str] = []

    class RequestServingServer:
        def run(self, transport: str, **kwargs: object) -> None:
            assert transport == "stdio"
            assert not kwargs
            assert audit.active is True
            lifecycle.append("server_run")
            assert continuous_map.read_bytes() == b"continuous numeric payload"
            forbidden_file.read_text(encoding="utf-8")

    def factory() -> RequestServingServer:
        assert audit.active is True
        lifecycle.append("server_factory")
        assert continuous_map.read_bytes() == b"continuous numeric payload"
        return RequestServingServer()

    with pytest.raises(PermissionError, match="Blocked forbidden runtime file read"):
        _run_semantic_server_lifetime(
            factory,
            audit=audit,
            audit_report=audit_report,
            transport="stdio",
            host="127.0.0.1",
            port=8766,
        )

    assert lifecycle == ["server_factory", "server_run"]
    assert audit.active is False
    report = json.loads(audit_report.read_text(encoding="utf-8"))
    assert report["passed"] is False
    assert report["block_forbidden"] is True
    assert str(continuous_map.resolve()) in report["loaded_files"]
    assert report["forbidden_accesses"] == [str(forbidden_file.resolve())]


def test_semantic_mcp_component_policy_blocks_qa_under_any_data_root(
    tmp_path: Path,
) -> None:
    qa_file = tmp_path / "data_diverse52" / "qa" / "questions.jsonl"
    qa_file.parent.mkdir(parents=True)
    qa_file.write_text("runtime must not read this", encoding="utf-8")
    feature_file = tmp_path / "data_gemma4" / "features" / "f_000001.npy"
    feature_file.parent.mkdir(parents=True)
    feature_file.write_bytes(b"continuous feature")
    audit = FileAccessAudit(
        [],
        forbidden_component_names=frozenset({"oracle", "qa"}),
        block_forbidden=True,
    )

    with audit:
        assert feature_file.read_bytes() == b"continuous feature"
        with pytest.raises(PermissionError, match="Blocked forbidden runtime file read"):
            qa_file.read_text(encoding="utf-8")

    assert audit.forbidden_accesses() == [str(qa_file.resolve())]


def _write_mcp_check_config(tmp_path: Path) -> Path:
    config = _config(tmp_path)
    config["paths"]["reports_root"] = str(tmp_path / "reports")
    config["render"].update({"engine": "NUMERIC_TEST", "samples": 1})
    config["mapping"] = {"depth_max_m": 6.0}
    path = tmp_path / "mcp_check.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
    return path


def test_semantic_mcp_accepts_and_bounds_navigation_safety_metadata() -> None:
    from semantic_3d_chat.config import load_config

    config = load_config("configs/runtime/embodied_live.yaml")
    _validate_semantic_config_isolation(config)
    _validate_navigation_safety_metadata(config)

    invalid = {**config, "robot": dict(config["robot"])}
    invalid["robot"]["approach_minimum_safe_step_m"] = invalid["robot"]["max_move_m"] + 0.01
    with pytest.raises(ValueError, match="distance safety metadata exceeds move bound"):
        _validate_navigation_safety_metadata(invalid)


def test_numeric_mcp_check_is_finite_audited_and_state_preserving(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = _write_mcp_check_config(tmp_path)
    map_path = tmp_path / "maps" / "scene_000001" / "voxel_map.npz"
    before = (map_path.stat().st_mtime_ns, map_path.read_bytes())
    audit_report = tmp_path / "reports" / "numeric_mcp_check.json"

    def reject_transport(*args: object, **kwargs: object) -> None:
        del args, kwargs
        pytest.fail("--check must not start MCP transport")

    monkeypatch.setattr(mcp_server, "_serve", reject_transport)
    status = mcp_server.main(
        [
            "--config",
            str(config_path),
            "--scene",
            "scene_000001",
            "--check",
            "--audit-report",
            str(audit_report),
        ]
    )

    assert status == 0
    output = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert output["passed"] is True
    assert output["mode"] == "numeric_only"
    assert output["starts_transport"] is False
    assert output["loads_language_model"] is False
    assert output["loads_blender"] is False
    assert output["changes_robot_or_map_state"] is False
    assert output["action_protocol"]["tool_count"] == 9
    assert output["action_protocol"]["strict_input_schemas"] is True
    assert (map_path.stat().st_mtime_ns, map_path.read_bytes()) == before
    assert not (tmp_path / "robot").exists()

    audit = json.loads(audit_report.read_text(encoding="utf-8"))
    assert audit["passed"] is True
    assert audit["forbidden_accesses"] == []
    assert str(config_path.resolve()) in audit["loaded_files"]
    assert str(map_path.resolve()) in audit["loaded_files"]


def test_semantic_live_config_load_is_inside_lifetime_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _write_mcp_check_config(tmp_path)
    audit_report = tmp_path / "reports" / "semantic_lifetime.json"

    class ConfigWasAudited(RuntimeError):
        pass

    def stop_after_config(config: dict) -> dict:
        assert config["_config_path"] == str(config_path.resolve())
        raise ConfigWasAudited

    monkeypatch.setattr(
        mcp_server,
        "_validate_configured_action_protocol",
        stop_after_config,
    )
    monkeypatch.setattr(
        mcp_server,
        "_validate_semantic_config_isolation",
        lambda config: None,
    )
    monkeypatch.setattr(
        mcp_server,
        "_validate_navigation_safety_metadata",
        lambda config: None,
    )
    with pytest.raises(ConfigWasAudited):
        mcp_server.main(
            [
                "--config",
                str(config_path),
                "--scene",
                "scene_000001",
                "--checkpoint",
                str(tmp_path / "unused_checkpoint"),
                "--runtime-asset",
                str(tmp_path / "unused_asset.blend"),
                "--audit-report",
                str(audit_report),
            ]
        )

    report = json.loads(audit_report.read_text(encoding="utf-8"))
    assert report["passed"] is True
    assert report["forbidden_accesses"] == []
    assert str(config_path.resolve()) in report["loaded_files"]


def test_semantic_mcp_asset_preflight_authenticates_without_blender(
    tmp_path: Path,
) -> None:
    asset = tmp_path / "runtime_assets" / "scene_000001" / "s_000001.blend"
    asset.parent.mkdir(parents=True)
    asset.write_bytes(b"opaque deterministic blender payload")
    manifest = {
        "schema": "semantic_3d_chat.runtime_scene.v2",
        "scene_id": "scene_000001",
        "asset_file": asset.name,
        "asset_sha256": hashlib.sha256(asset.read_bytes()).hexdigest(),
        "object_names_opaque": True,
        "nested_names_opaque": True,
        "custom_properties_present": False,
        "external_assets_present": False,
        "automation_present": False,
        "animation_present": False,
        "unsupported_datablocks_present": False,
        "strict_nested_datablock_audit_passed": True,
        "mesh_objects": 4,
        "light_objects": 1,
        "materials": 2,
        "collections": 0,
        "node_trees": 2,
    }
    manifest_path = asset.with_suffix(".json")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    audit = FileAccessAudit(block_forbidden=True)

    with audit:
        result = _runtime_asset_preflight("scene_000001", asset, audit=audit)

    assert result["strict_manifest"] is True
    assert result["blender_loaded"] is False
    assert result["asset_sha256"] == manifest["asset_sha256"]
    assert str(asset.resolve()) in audit.unique_paths
    assert str(manifest_path.resolve()) in audit.unique_paths

    asset.write_bytes(b"tampered")
    with (
        FileAccessAudit(block_forbidden=True) as tamper_audit,
        pytest.raises(ValueError, match="hash differs"),
    ):
        _runtime_asset_preflight("scene_000001", asset, audit=tamper_audit)


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
