from __future__ import annotations

import hashlib
import inspect
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from semantic_3d_chat.evaluation.conversational_mcp_session_inspect import build_inspection
from semantic_3d_chat.robot import conversational_mcp_agent
from semantic_3d_chat.robot.conversational_mcp_agent import (
    PersistentMCPConversationSession,
    parse_interactive_mcp_command,
    run_face_instruction,
)
from semantic_3d_chat.robot.mcp_stdio_runtime import (
    MCPActionTransportError,
    MCPConversationRuntime,
)
from semantic_3d_chat.robot.semantic_mapping import semantic_map_content_hash
from semantic_3d_chat.robot.tools import TOOL_ARGUMENTS


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(payload), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_runtime_map(path: Path, *, version: int) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    xyz = np.asarray(
        [
            [-1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.5, -1.0, 1.0],
        ],
        dtype=np.float32,
    )
    features = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    header = {
        "schema_version": 1,
        "voxel_size_m": 0.05,
        "occupied_voxels": 3,
        "feature_dim": 4,
        "semantic_dtype_on_disk": "float32",
        "codec": "identity",
        "total_observations": 3 * (version + 1),
        "max_voxels": 100,
        "metadata": {"scene_id": "scene_000001"},
    }
    np.savez_compressed(
        path,
        voxel_coordinates=np.floor(xyz / 0.05).astype(np.int32),
        centers_world=xyz,
        observation_count=np.full(3, version + 1, dtype=np.int32),
        weight_sum=np.full(3, version + 1, dtype=np.float32),
        mean_rgb=np.full((3, 3), 127.0, dtype=np.float32),
        semantic_features=features,
        semantic_feature_m2=np.zeros(3, dtype=np.float32),
        semantic_variance=np.zeros(3, dtype=np.float32),
        normal=np.tile(np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32), (3, 1)),
        normal_valid=np.ones(3, dtype=bool),
        view_direction=np.tile(np.asarray([[0.0, 1.0, 0.0]], dtype=np.float32), (3, 1)),
        view_direction_valid=np.ones(3, dtype=bool),
        confidence=np.ones(3, dtype=np.float32),
        last_frame=np.full(3, f"o_{version:06d}", dtype="<U8"),
        metadata_json=np.asarray(json.dumps(header, sort_keys=True)),
    )
    return semantic_map_content_hash(path)


def _receipt(
    *,
    version: int,
    map_sha256: str,
    yaw: float,
    action_count: int,
    stopped: bool,
    position_xy_m: tuple[float, float] = (0.0, 0.0),
    last_movement_delta_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
    distance_moved: float = 0.0,
    turn_degrees: float = 0.0,
    observation: bool = False,
) -> dict[str, Any]:
    marker = f"{version % 16:x}"
    robot_marker = f"{(action_count + 8) % 16:x}"
    scene_identity: dict[str, Any] = {
        "schema": "semantic_3d_chat.scene_prefix_binding.v2",
        "scene_id": "scene_000001",
        "map_version": version,
        "map_sha256": map_sha256,
        "scene_prefix_sha256": marker * 64,
        "scene_control_signature_sha256": f"{(version + 1) % 16:x}" * 64,
        "source_voxels": 3,
        "processed_voxels": 3,
    }
    binding_sha256 = _canonical_sha256(scene_identity)
    active_identity = {
        **scene_identity,
        "binding_sha256": binding_sha256,
        "active_prefix_sha256": robot_marker * 64,
        "robot_state_sha256": f"{(action_count + 9) % 16:x}" * 64,
        "robot_tokens_sha256": f"{(action_count + 10) % 16:x}" * 64,
        "robot_state_encoder_sha256": "f" * 64,
    }
    return {
        "success": True,
        "error_code": None,
        "scene_id": "scene_000001",
        "seed": 17,
        "scene_version": version,
        "position_m": [position_xy_m[0], position_xy_m[1], 0.0],
        "camera_position_m": [position_xy_m[0], position_xy_m[1], 1.2],
        "body_yaw_degrees": yaw,
        "camera_yaw_degrees": yaw,
        "pitch_degrees": 0.0,
        "linear_velocity_xy_m": [0.0, 0.0],
        "angular_velocity_degrees": 0.0,
        "collision": False,
        "last_movement_delta_m": list(last_movement_delta_m),
        "distance_moved": distance_moved,
        "turn_degrees": turn_degrees,
        "scan_coverage": min(1.0, version / 10.0),
        "scan_count": version,
        "visible_voxels": 3 if observation else 0,
        "valid_depth_pixels": 3 if observation else 0,
        "observation_id": f"o_{version:06d}" if observation else None,
        "clearance_m": None,
        "action_count": action_count,
        "stopped": stopped,
        **active_identity,
        "active_binding_sha256": _canonical_sha256(active_identity),
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
            "face_alignment_deadband_degrees": 3.0,
            "face_alignment_stalled_turn_degrees": 1.0,
            "approach_heading_deadband_degrees": 15.0,
            "approach_target_standoff_m": 0.5,
            "approach_minimum_progress_m": 0.15,
            "approach_minimum_safe_step_m": 0.02,
        },
    }


class _TextEncoder:
    output_dim = 4

    def __init__(self) -> None:
        self.queries: list[tuple[str, ...]] = []

    def encode_queries(self, queries: Sequence[str]) -> np.ndarray:
        self.queries.append(tuple(queries))
        return np.tile(np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (len(queries), 1))


class _StatefulClient:
    tool_names = frozenset(TOOL_ARGUMENTS)

    def __init__(self, map_path: Path) -> None:
        self.map_path = map_path
        self.version = 0
        self.yaw = 0.0
        self.action_count = 0
        self.stopped = False
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.map_sha256 = _write_runtime_map(map_path, version=0)

    def call_tool(self, name: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
        self.calls.append((name, dict(arguments)))
        observation = False
        if name == "scan":
            observation = True
            self.version += 1
            self.action_count += 1
            self.map_sha256 = _write_runtime_map(self.map_path, version=self.version)
        elif name == "turn":
            observation = True
            self.yaw += float(arguments["angle_degrees"])
            self.version += 1
            self.action_count += 1
            self.map_sha256 = _write_runtime_map(self.map_path, version=self.version)
        elif name == "stop":
            self.action_count += 1
            self.stopped = True
        elif name != "get_robot_state":
            raise AssertionError(f"unexpected fake tool call: {name}")
        return _receipt(
            version=self.version,
            map_sha256=self.map_sha256,
            yaw=self.yaw,
            action_count=self.action_count,
            stopped=self.stopped,
            observation=observation,
        )


class _PersistentSessionClient(_StatefulClient):
    def __init__(self, map_path: Path) -> None:
        super().__init__(map_path)
        self.position = np.zeros(2, dtype=np.float64)

    def call_tool(self, name: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
        self.calls.append((name, dict(arguments)))
        movement = np.zeros(3, dtype=np.float64)
        distance = 0.0
        turn = 0.0
        observation = False
        if name == "scan":
            observation = True
            self.version += 1
            self.action_count += 1
            self.map_sha256 = _write_runtime_map(self.map_path, version=self.version)
        elif name == "turn":
            observation = True
            turn = float(arguments["angle_degrees"])
            self.yaw += turn
            self.version += 1
            self.action_count += 1
            self.map_sha256 = _write_runtime_map(self.map_path, version=self.version)
        elif name == "move_forward":
            observation = True
            distance = float(arguments["distance_meters"])
            direction = np.asarray(
                [
                    -np.sin(np.radians(self.yaw)),
                    np.cos(np.radians(self.yaw)),
                ],
                dtype=np.float64,
            )
            delta = direction * distance
            self.position += delta
            movement[:2] = delta
            self.version += 1
            self.action_count += 1
            self.map_sha256 = _write_runtime_map(self.map_path, version=self.version)
        elif name == "stop":
            self.action_count += 1
            self.stopped = True
        elif name != "get_robot_state":
            raise AssertionError(f"unexpected fake tool call: {name}")
        return _receipt(
            version=self.version,
            map_sha256=self.map_sha256,
            yaw=self.yaw,
            action_count=self.action_count,
            stopped=self.stopped,
            position_xy_m=(float(self.position[0]), float(self.position[1])),
            last_movement_delta_m=tuple(float(value) for value in movement),
            distance_moved=distance,
            turn_degrees=turn,
            observation=observation,
        )


def test_natural_language_face_loop_crosses_numeric_tool_seam_and_stops(tmp_path: Path) -> None:
    active_map = tmp_path / "runtime" / "scene_000001" / "semantic_map.npz"
    client = _StatefulClient(active_map)
    runtime = MCPConversationRuntime(
        client,
        _config(tmp_path),
        base_map_path=active_map,
        persistent_map_path=active_map,
    )
    encoder = _TextEncoder()

    result = run_face_instruction(
        runtime,
        encoder,
        "Face the chair, then stop.",
        room_size_m=[6.0, 5.0, 3.0],
        feature_start=0,
        feature_dim=4,
        max_steps=4,
    )

    assert result["passed"] is True
    assert result["termination_reason"] == "fresh_grounding_inside_deadband"
    assert [name for name, _arguments in client.calls] == [
        "get_robot_state",
        "scan",
        "turn",
        "turn",
        "stop",
    ]
    assert [step["mcp_call"]["tool"] for step in result["steps"]] == [
        "turn",
        "turn",
        "stop",
    ]
    assert result["final_body_yaw_degrees"] == pytest.approx(90.0)
    assert result["final_stopped"] is True
    assert result["all_decisions_used_fresh_all_voxel_grounding"] is True
    assert result["semantic_leaks_in_numeric_tool_receipts"] == []
    assert result["policy"]["learned_v3_action_head_used"] is False
    assert result["policy"]["gemma_native_function_calling_used"] is False
    assert result["policy"]["official_mcp_sdk_stdio_action_execution"] is False
    assert result["transport"]["implementation"] == "structured_tool_client_test_seam"
    assert result["target_phrase_retained_in_tool_output"] is False
    assert all(step["continuous_binding_transition"]["passed"] is True for step in result["steps"])
    assert all(step["grounding"]["scored_voxels"] == 3 for step in result["steps"])
    assert len(encoder.queries) == 3
    assert all(queries == ("chair", "a chair") for queries in encoder.queries)
    # The only label-bearing text is the user's instruction/grounding query;
    # validated MCP receipts remain strictly numeric/protocol-only.
    serialized_receipts = json.dumps(
        [result["initial_observation"]] + [step["numeric_tool_receipt"] for step in result["steps"]]
    ).casefold()
    assert "chair" not in serialized_receipts


def test_face_loop_requires_explicit_terminal_stop(tmp_path: Path) -> None:
    active_map = tmp_path / "runtime" / "scene_000001" / "semantic_map.npz"
    client = _StatefulClient(active_map)
    runtime = MCPConversationRuntime(
        client,
        _config(tmp_path),
        base_map_path=active_map,
        persistent_map_path=active_map,
    )

    with pytest.raises(ValueError, match="explicit terminal"):
        run_face_instruction(
            runtime,
            _TextEncoder(),
            "Face the chair.",
            room_size_m=[6.0, 5.0, 3.0],
            feature_start=0,
            feature_dim=4,
            max_steps=4,
        )

    # Fail before any state-changing tool call for a command outside the
    # terminal alignment grammar.
    assert [name for name, _arguments in client.calls] == ["get_robot_state"]


@pytest.mark.parametrize(
    ("text", "kind", "target"),
    [
        ("Face the translucent wobble sculpture.", "face", "translucent wobble sculpture"),
        ("Look at the chair, then stop.", "face", "chair"),
        ("Look toward the bronze shape.", "face", "bronze shape"),
        ("Turn toward cobalt thing and stop", "face", "cobalt thing"),
        ("Move closer to the red cube, then stop.", "approach", "red cube"),
        ("Walk toward bowl and stop", "approach", "bowl"),
        ("Approach the side table.", "approach", "side table"),
        ("scan", "scan", None),
        ("look around", "scan", None),
        ("get robot state", "state", None),
        ("stop", "stop", None),
    ],
)
def test_interactive_parser_is_bounded_and_has_no_object_inventory(
    text: str, kind: str, target: str | None
) -> None:
    parsed = parse_interactive_mcp_command(text)
    assert parsed.kind == kind
    assert parsed.target_text == target
    if target is not None:
        assert target in str(parsed.terminal_instruction)


def test_live_conversational_module_imports_no_evaluator_or_object_inventory() -> None:
    source = inspect.getsource(conversational_mcp_agent)
    assert "semantic_3d_chat.evaluation" not in source
    inventory = (
        "bowl",
        "book",
        "cabinet",
        "chair",
        "cube",
        "door",
        "frame",
        "lamp",
        "picture",
        "plant",
        "table",
        "window",
    )
    assert not any(re.search(rf"\b{word}\b", source, re.IGNORECASE) for word in inventory)


def test_persistent_session_reuses_one_client_for_face_approach_scan_state_stop(
    tmp_path: Path,
) -> None:
    active_map = tmp_path / "runtime" / "scene_000001" / "semantic_map.npz"
    client = _PersistentSessionClient(active_map)
    runtime = MCPConversationRuntime(
        client,
        _config(tmp_path),
        base_map_path=active_map,
        persistent_map_path=active_map,
    )
    encoder = _TextEncoder()
    session = PersistentMCPConversationSession(
        runtime,
        encoder,
        room_size_m=[6.0, 5.0, 3.0],
        feature_start=0,
        feature_dim=4,
        max_steps=6,
    )

    startup = session.start()
    face = session.handle("Face the chair, then stop.")
    assert startup["passed"] is True
    assert face["passed"] is True
    assert face["episode_stop_latched"] is False
    assert face["goal_settled_without_episode_latch"] is True
    assert face["final_body_yaw_degrees"] == pytest.approx(90.0)

    approach = session.handle("Walk toward the chair and stop.")
    assert approach["passed"] is True
    assert approach["termination_reason"] == "semantic_standoff"
    assert approach["actual_progress_m"] == pytest.approx(0.5)
    assert approach["episode_stop_latched"] is False
    assert approach["goal_settled_without_episode_latch"] is True

    scan = session.handle("scan")
    state = session.handle("get robot state")
    stop = session.handle("stop")
    shutdown = session.shutdown()
    summary = session.summary()

    assert scan["passed"] is True
    assert state["passed"] is True
    assert stop["passed"] is True
    assert stop["episode_stop_latched"] is True
    assert shutdown["passed"] is True
    assert shutdown["mcp_stop_called"] is False
    assert summary["passed"] is True
    assert summary["turn_count"] == 5
    assert summary["final_stopped"] is True
    assert summary["policy"]["learned_v3_action_head_used"] is False
    assert summary["policy"]["gemma_native_function_calling_used"] is False
    assert summary["policy"]["one_persistent_stdio_session"] is True

    names = [name for name, _arguments in client.calls]
    assert names == [
        "get_robot_state",  # runtime handshake
        "scan",  # one session-start observation
        "turn",
        "turn",
        "get_robot_state",  # stationary face-goal acknowledgment, not stop
        "move_forward",
        "get_robot_state",  # stationary approach-goal acknowledgment, not stop
        "scan",
        "get_robot_state",
        "stop",  # only the user's standalone stop consumes the latch
        "get_robot_state",  # shutdown verifies the already-latched state
    ]
    assert all(
        step["continuous_binding_transition"]["passed"] is True
        for turn in (face, approach)
        for step in turn["steps"]
    )
    assert all(
        step["grounding"]["all_map_voxels_scored"] is True
        for turn in (face, approach)
        for step in turn["steps"]
    )
    receipts = [startup["initial_observation"]["numeric_tool_receipt"]]
    receipts.extend(step["numeric_tool_receipt"] for step in face["steps"])
    receipts.extend(step["numeric_tool_receipt"] for step in approach["steps"])
    receipts.extend(
        (scan["numeric_tool_receipt"], state["numeric_tool_receipt"], stop["numeric_tool_receipt"])
    )
    serialized = json.dumps(receipts).casefold()
    assert "chair" not in serialized
    assert "oracle" not in serialized


def test_invalid_or_post_stop_command_is_fail_closed_without_motion(tmp_path: Path) -> None:
    active_map = tmp_path / "runtime" / "scene_000001" / "semantic_map.npz"
    client = _PersistentSessionClient(active_map)
    runtime = MCPConversationRuntime(
        client,
        _config(tmp_path),
        base_map_path=active_map,
        persistent_map_path=active_map,
    )
    session = PersistentMCPConversationSession(
        runtime,
        _TextEncoder(),
        room_size_m=[6.0, 5.0, 3.0],
        feature_start=0,
        feature_dim=4,
    )
    session.start()
    calls_before = len(client.calls)
    with pytest.raises(ValueError, match="Supported commands"):
        session.handle("please invent an unsafe motor command")
    assert len(client.calls) == calls_before

    assert session.handle("stop")["passed"] is True
    calls_before = len(client.calls)
    blocked = session.handle("Approach the chair")
    assert blocked["passed"] is False
    assert blocked["error_code"] == "E_STOPPED"
    assert blocked["tool_executed"] is False
    assert len(client.calls) == calls_before


def _write_clean_session_audit(
    path: Path, *, training_blocked_by_root: bool = False
) -> dict[str, Any]:
    forbidden_names = [
        "oracle",
        "qa",
        "scorer-only",
        "scorer_only",
    ]
    if not training_blocked_by_root:
        forbidden_names.append("training")
    loaded = (
        path.parent / "src" / "semantic_3d_chat" / "training" / "checkpointing.pyc"
        if training_blocked_by_root
        else path.parent / "runtime_model.safetensors"
    )
    payload = {
        "loaded_files": [str(loaded)],
        "forbidden_roots": [
            str(path.parent / "data" / "oracle"),
            str(path.parent / "data" / "qa"),
            str(path.parent / "data_gemma4" / "training"),
            str(path.parent / "reports" / "scorer_only"),
        ],
        "forbidden_component_names": forbidden_names,
        "block_forbidden": True,
        "forbidden_accesses": [],
        "passed": True,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return {
        "path": str(path),
        "sha256": _file_sha256(path),
        "loaded_file_count": 1,
        "forbidden_access_count": 0,
        "passed": True,
    }


def _write_inspectable_session(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    active_map = tmp_path / "runtime" / "scene_000001" / "semantic_map.npz"
    client = _PersistentSessionClient(active_map)
    runtime = MCPConversationRuntime(
        client,
        _config(tmp_path),
        base_map_path=active_map,
        persistent_map_path=active_map,
    )
    session = PersistentMCPConversationSession(
        runtime,
        _TextEncoder(),
        room_size_m=[6.0, 5.0, 3.0],
        feature_start=0,
        feature_dim=4,
        max_steps=6,
    )
    session.start()
    for command in (
        "Face the chair, then stop.",
        "Move closer to the chair, then stop.",
        "scan",
        "get robot state",
        "stop",
    ):
        assert session.handle(command)["passed"] is True
    session.shutdown()
    report = session.summary()
    report["policy"]["official_mcp_sdk_stdio_action_execution"] = True
    report["transport"].update(
        {
            "implementation": "official_python_mcp_sdk_stdio",
            "process_boundary": True,
            "persistent_connection": True,
        }
    )
    report["client_access_audit"] = _write_clean_session_audit(tmp_path / "client.json")
    report["server_access_audit"] = _write_clean_session_audit(
        tmp_path / "server.json", training_blocked_by_root=True
    )
    runtime_result = tmp_path / "session.json"
    runtime_result.write_text(json.dumps(report), encoding="utf-8")
    return runtime_result, report


def test_model_free_session_inspector_authenticates_finite_same_stdio_proof(
    tmp_path: Path,
) -> None:
    runtime_result, _report = _write_inspectable_session(tmp_path)

    inspection = build_inspection(runtime_result)

    assert inspection["passed"] is True
    assert inspection["transcript"]["command_order"] == [
        "face",
        "approach",
        "scan",
        "state",
        "stop",
    ]
    assert inspection["transcript"]["move_count"] == 1
    assert inspection["transcript"]["distance_moved_m"] == pytest.approx(0.5)
    assert inspection["transcript"]["final_stopped"] is True
    assert inspection["client_access_audit"]["forbidden_access_count"] == 0
    assert inspection["server_access_audit"]["forbidden_access_count"] == 0
    assert inspection["environmental_text_inputs"] == []
    assert inspection["oracle_inputs_opened"] is False
    assert inspection["oracle_target_distance_scoring_deferred"] is True
    assert all(inspection["checks"].values())


def test_session_inspector_rejects_reordered_or_zero_movement_transcript(
    tmp_path: Path,
) -> None:
    runtime_result, report = _write_inspectable_session(tmp_path)
    report["turns"][0], report["turns"][1] = report["turns"][1], report["turns"][0]
    runtime_result.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="command order"):
        build_inspection(runtime_result)

    runtime_result, report = _write_inspectable_session(tmp_path / "zero_move")
    movement = next(
        step for step in report["turns"][1]["steps"] if step["mcp_call"]["tool"] == "move_forward"
    )
    movement["numeric_tool_receipt"]["distance_moved"] = 0.0
    runtime_result.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="positive numeric translation"):
        build_inspection(runtime_result)


def test_session_inspector_rejects_semantic_receipt_or_rehashed_forbidden_audit(
    tmp_path: Path,
) -> None:
    runtime_result, report = _write_inspectable_session(tmp_path)
    report["turns"][2]["numeric_tool_receipt"]["object_name"] = "chair"
    runtime_result.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(MCPActionTransportError, match="numeric schema"):
        build_inspection(runtime_result)

    runtime_result, report = _write_inspectable_session(tmp_path / "bad_audit")
    client_path = Path(report["client_access_audit"]["path"])
    audit = json.loads(client_path.read_text(encoding="utf-8"))
    audit["loaded_files"].append(str(tmp_path / "data" / "oracle" / "scene.json"))
    audit["forbidden_accesses"] = [audit["loaded_files"][-1]]
    audit["passed"] = False
    client_path.write_text(json.dumps(audit), encoding="utf-8")
    report["client_access_audit"].update(
        {
            "sha256": _file_sha256(client_path),
            "loaded_file_count": 2,
            "forbidden_access_count": 1,
            "passed": False,
        }
    )
    runtime_result.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="client audit"):
        build_inspection(runtime_result)
