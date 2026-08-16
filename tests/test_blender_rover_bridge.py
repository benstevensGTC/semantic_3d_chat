from __future__ import annotations

import math
from pathlib import Path

import pytest

from semantic_3d_chat.robot.blender_rover_bridge import (
    LoopbackRoverClient,
    interpolate_pose,
    normalize_loopback_url,
    parse_rover_response,
    shortest_yaw_delta,
)

ROOT = Path(__file__).parents[1]


def _response(**state_overrides: object) -> dict[str, object]:
    state: dict[str, object] = {
        "scene_id": "scene_000001",
        "position_xy_m": [0.25, -0.5],
        "body_yaw_degrees": -30.0,
        "camera_yaw_degrees": -10.0,
        "pitch_degrees": 5.0,
        "collision": False,
        "stopped": False,
        "scene_version": 2,
        "map_version": 2,
        "scan_count": 3,
    }
    state.update(state_overrides)
    return {
        "state": state,
        "reply": "Movement complete.",
        "scene_prefix_hash": "a" * 64,
        "control": {
            "control_mode": "actual_local_gemma_model_only_waypoint_policy",
            "navigation_control_mode": "actual_local_gemma_model_only_waypoint_policy",
            "model_selects_every_waypoint_and_heading": True,
            "model_selects_stop": True,
            "deterministic_route_planner_used": False,
            "fallback_used": False,
            "substitution_applied": False,
            "synthetic_stop_applied": False,
        },
        "scene_memory": {
            "schema": "semantic_3d_chat.scene_memory_diagnostics.v1",
            "tensor_shape": [1, 738, 1536],
            "sha256": "a" * 64,
            "l2_norm": 412.25,
            "token_count": 738,
            "model_dim": 1536,
            "map_version": 2,
            "all_runtime_voxels_encoded": True,
            "question_dependent_scene_retrieval": False,
            "environmental_text_inputs": [],
            "base_adapter_weights_loaded": True,
            "control_weights_loaded": True,
            "control_training_gate_passed": True,
            "loaded_file_audit": {"passed": True, "forbidden_access_count": 0},
        },
    }


def _decision(
    step: int,
    *,
    action: str = "move_to",
    accepted: bool = True,
    executed: bool = True,
    error_code: str | None = None,
) -> dict[str, object]:
    primitive = {"move_to": "move_to", "face": "turn", "stop": "stop"}[action]
    arguments: dict[str, float]
    if primitive == "move_to":
        arguments = {"x": 0.25, "y": -0.5}
    elif primitive == "turn":
        arguments = {"angle_degrees": -30.0}
    else:
        arguments = {}
    action_index = ("move_to", "face", "stop").index(action)
    logits = [-8.0, -8.0, -8.0]
    probabilities = [0.0, 0.0, 0.0]
    logits[action_index] = 8.0
    probabilities[action_index] = 1.0
    return {
        "step": step,
        "model_action": action,
        "model_action_logits": logits,
        "model_action_probabilities": probabilities,
        "scene_token_count": 738,
        "robot_token_count": 4,
        "history_token_count": step - 1,
        "prompt_token_count": 12,
        "decision_position": 753 + step,
        "decision_tensor_sha256": f"{step % 10}" * 64,
        "instruction_sha256": "1" * 64,
        "active_prefix_sha256": "2" * 64,
        "scene_prefix_sha256": "a" * 64,
        "robot_tokens_sha256": "3" * 64,
        "checkpoint_sha256": "d" * 64,
        "model_waypoint_delta_robot_m": [0.1, 0.4],
        "model_turn_delta_degrees": -30.0,
        "model_desired_heading_degrees": -30.0,
        "derived_absolute_facing_heading_degrees": -30.0,
        "derived_world_waypoint_xy_m": (
            [arguments["x"], arguments["y"]] if primitive == "move_to" else None
        ),
        "primitive_tool": primitive,
        "primitive_arguments": arguments,
        "accepted": accepted,
        "executed": executed,
        "error_code": error_code,
        "actual_gemma_causal_forward": True,
        "model_selected_every_waypoint_and_heading": True,
        "deterministic_route_planner_used": False,
        "substitution_applied": False,
        "synthetic_stop_applied": False,
    }


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("http://127.0.0.1:8770/", "http://127.0.0.1:8770"),
        ("http://LOCALHOST:9000", "http://localhost:9000"),
        ("http://[::1]:8770", "http://[::1]:8770"),
    ],
)
def test_loopback_url_normalization(raw: str, expected: str) -> None:
    assert normalize_loopback_url(raw) == expected


@pytest.mark.parametrize(
    "url",
    [
        "https://127.0.0.1:8770",
        "http://192.168.1.5:8770",
        "http://example.com",
        "http://127.0.0.1:8770/api/state",
        "http://user:pass@127.0.0.1:8770",
    ],
)
def test_bridge_refuses_non_origin_or_non_loopback_urls(url: str) -> None:
    with pytest.raises(ValueError, match="loopback"):
        normalize_loopback_url(url)


def test_public_numeric_response_becomes_blender_pose() -> None:
    response = parse_rover_response(_response())

    assert response.pose.scene_id == "scene_000001"
    assert (response.pose.x_m, response.pose.y_m) == (0.25, -0.5)
    assert response.pose.body_yaw_degrees == -30.0
    assert response.pose.camera_yaw_degrees == -10.0
    assert response.pose.scene_version == 2
    assert response.reply == "Movement complete."
    assert response.trajectory == (response.pose,)
    assert response.scene_memory is not None
    assert response.scene_memory.shape == (1, 738, 1536)
    assert response.scene_memory.sha256 == "a" * 64
    assert response.scene_memory.l2_norm == pytest.approx(412.25)


def test_blender_bridge_rejects_legacy_or_fallback_navigation() -> None:
    legacy = _response()
    legacy["control"]["navigation_control_mode"] = "legacy_navigation_backend"
    with pytest.raises(ValueError, match="model-only navigation backend"):
        parse_rover_response(legacy)

    fallback = _response()
    fallback["control"]["fallback_used"] = True
    with pytest.raises(ValueError, match="movement attestation failed"):
        parse_rover_response(fallback)


def test_numeric_action_receipts_preserve_compound_trajectory() -> None:
    payload = _response()
    payload["actions"] = [
        {
            "scene_id": "scene_000001",
            "position_m": [0.0, 0.5, 0.0],
            "body_yaw_degrees": 0.0,
            "camera_yaw_degrees": 0.0,
            "pitch_degrees": 0.0,
            "collision": False,
            "stopped": False,
            "scene_version": 1,
            "scan_count": 1,
            "distance_moved": 0.5,
        },
        {
            "scene_id": "scene_000001",
            "position_m": [0.25, -0.5, 0.0],
            "body_yaw_degrees": -30.0,
            "camera_yaw_degrees": -10.0,
            "pitch_degrees": 5.0,
            "collision": False,
            "stopped": False,
            "scene_version": 2,
            "scan_count": 3,
            "turn_degrees": -30.0,
            "valid_depth_pixels": 50176,
        },
    ]

    response = parse_rover_response(payload)

    assert [(pose.x_m, pose.y_m) for pose in response.trajectory] == [
        (0.0, 0.5),
        (0.25, -0.5),
    ]
    assert response.trajectory[0].body_yaw_degrees == 0.0
    assert response.trajectory[-1] == response.pose
    assert response.events == (
        "Translated 0.50 m to x=+0.00, y=+0.50 m.",
        "Rotated -30.0°.",
    )


def test_actual_gemma_decisions_replace_generic_events_with_numeric_receipts() -> None:
    payload = _response()
    payload["model_decisions"] = [
        _decision(1, action="face"),
        _decision(
            2,
            action="move_to",
            accepted=False,
            executed=False,
            error_code="E_MODEL_COLLISION",
        ),
        _decision(3, action="stop"),
    ]
    payload["actions"] = [
        {
            "scene_id": "scene_000001",
            "position_m": [0.25, -0.5, 0.0],
            "body_yaw_degrees": -30.0,
            "camera_yaw_degrees": -10.0,
            "pitch_degrees": 5.0,
            "scene_version": 2,
            "scan_count": 3,
            "turn_degrees": -30.0,
        }
    ]

    response = parse_rover_response(payload)

    assert len(response.decisions) == 3
    assert response.decisions[0].model_action == "face"
    assert response.decisions[0].turn_delta_degrees == pytest.approx(-30.0)
    assert response.decisions[0].desired_heading_degrees == pytest.approx(-30.0)
    assert response.decisions[1].accepted is False
    assert response.decisions[1].error_code == "E_MODEL_COLLISION"
    assert response.events[0].startswith(
        "Gemma step 001 · FACE · Gemma raw turn Δ -30.000° · "
        "deterministic absolute facing -30.000° · executed exact raw Δ · "
        "ACCEPTED · EXECUTED"
    )
    assert "Gemma step 002 · MOVE_TO" in response.events[1]
    assert "REJECTED · NOT EXECUTED · E_MODEL_COLLISION" in response.events[1]
    assert "Gemma step 003 · STOP" in response.events[2]
    for event in response.events:
        assert "causal context 738 scene + 4 robot" in event
        assert "prompt tokens · decision position" in event
        assert "p(move/face/stop)" in event
        assert "raw logits [" in event
        assert "output " in event
        assert "active prefix " in event
        assert "checkpoint " in event


def test_bridge_rejects_tampered_raw_face_or_world_waypoint_provenance() -> None:
    face_payload = _response()
    face = _decision(1, action="face")
    face["model_turn_delta_degrees"] = -29.0
    face_payload["model_decisions"] = [face]
    with pytest.raises(ValueError, match="differs from Gemma's raw turn delta"):
        parse_rover_response(face_payload)

    move_payload = _response()
    move = _decision(1, action="move_to")
    move["derived_world_waypoint_xy_m"] = [9.0, 9.0]
    move_payload["model_decisions"] = [move]
    with pytest.raises(ValueError, match="differs from execution target"):
        parse_rover_response(move_payload)


def test_bridge_accepts_128_decisions_and_rejects_129() -> None:
    payload = _response()
    payload["model_decisions"] = [_decision(step) for step in range(1, 129)]
    assert len(parse_rover_response(payload).decisions) == 128

    payload["model_decisions"] = [_decision(step) for step in range(1, 130)]
    with pytest.raises(ValueError, match="decision log is too long"):
        parse_rover_response(payload)


@pytest.mark.parametrize(
    "memory",
    [
        {"tensor_shape": [1, 738, 1536], "sha256": "bad", "l2_norm": 1.0},
        {"tensor_shape": [1, 0, 1536], "sha256": "a" * 64, "l2_norm": 1.0},
        {"tensor_shape": [1, 738, 1536], "sha256": "a" * 64, "l2_norm": math.nan},
        {
            "tensor_shape": [1, 738, 1536],
            "sha256": "a" * 64,
            "l2_norm": 1.0,
            "token_count": 64,
        },
    ],
)
def test_invalid_scene_memory_diagnostics_fail_closed(memory: dict[str, object]) -> None:
    payload = _response()
    payload["scene_memory"] = memory
    with pytest.raises((TypeError, ValueError)):
        parse_rover_response(payload)


def test_scene_memory_hash_must_match_public_prefix_binding() -> None:
    payload = _response()
    payload["scene_prefix_hash"] = "b" * 64
    with pytest.raises(ValueError, match="public scene-prefix binding"):
        parse_rover_response(payload)


@pytest.mark.parametrize(
    "override",
    [
        {"scene_id": "chair_room"},
        {"position_xy_m": [0.0]},
        {"position_xy_m": [math.nan, 0.0]},
        {"scan_count": -1},
        {"body_yaw_degrees": math.inf},
    ],
)
def test_invalid_numeric_state_fails_closed(override: dict[str, object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        parse_rover_response(_response(**override))


def test_yaw_interpolation_uses_the_shortest_visible_arc() -> None:
    assert shortest_yaw_delta(170.0, -170.0) == pytest.approx(20.0)
    assert shortest_yaw_delta(-170.0, 170.0) == pytest.approx(-20.0)

    halfway = interpolate_pose((0.0, 0.0), (2.0, -1.0), 170.0, -170.0, 0.5)
    assert halfway == pytest.approx((1.0, -0.5, 180.0))
    assert interpolate_pose((0, 0), (2, 2), 0, 90, -1) == pytest.approx((0, 0, 0))
    assert interpolate_pose((0, 0), (2, 2), 0, 90, 2) == pytest.approx((2, 2, 90))


def test_client_exposes_only_state_and_high_level_instruction_methods() -> None:
    client = LoopbackRoverClient("http://127.0.0.1:8770")
    assert callable(client.state)
    assert callable(client.instruct)
    assert not hasattr(client, "tool")


def test_blender_script_is_a_real_3d_animated_operator_surface() -> None:
    source = (ROOT / "blender/rover_control_ui.py").read_text(encoding="utf-8")
    bridge = (ROOT / "src/semantic_3d_chat/robot/blender_rover_bridge.py").read_text(
        encoding="utf-8"
    )

    assert "primitive_cube_add" in source
    assert "primitive_cylinder_add" in source
    assert "primitive_uv_sphere_add" in source
    assert "primitive_cone_add" in source
    assert "interpolate_pose" in source
    assert "response.trajectory" in source
    assert "target_camera_yaw_offset" in source
    assert "space.show_region_ui = True" in source
    assert "active_panel_category" not in source
    assert 'bl_space_type = "VIEW_3D"' in source
    assert 'bl_category = "Gemma Rover"' in source
    assert '"/api/instruction"' not in source  # endpoints stay in the pure bridge
    assert "Gemma is thinking locally" in source
    assert "Replaying already-returned Gemma decisions" in source
    assert "Agent executing" not in source
    assert "GEMMA_ROVER_UL_transcript" in source
    assert "template_list" in source
    assert "_MAX_TRANSCRIPT_LINES = 4_096" in source
    assert '"agent": "Gemma decision"' in source
    assert "Send goal to local Gemma" in source
    assert "Gemma chooses every waypoint, heading, and stop." in source
    assert "No direct driving controls." in source
    assert "GeometryNodeMeshToPoints" in source
    assert '_MAX_MAP_POINTS = 25_000' in source
    assert 'required = {"centers_world", "mean_rgb", "semantic_features", "confidence"}' in source
    assert '"data_gemma4/maps"' in source
    assert '"data_gemma4/robot/practical_rover"' not in source
    assert "robot-camera refresh" in source
    assert source.index('overlay.prop(scene, "gemma_rover_show_map"') < source.index(
        "if scene.gemma_rover_show_technical_details:"
    ) < source.index('memory.label(text=scene.gemma_rover_map_points')
    assert "Embedded 3D map overlay" in source
    assert "Continuous scene tokens" in source
    assert "pre-scanned before any goal" in source
    assert "no rover camera is used as an agent input" in source
    assert "Rover Camera" not in source
    assert "what I can perceive" not in source
    assert "gemma_rover.tool" not in source
    assert "Turn Left" not in source
    assert "Forward" not in source
    assert "reports/gemma4" not in source
    assert "data/oracle" not in source
    assert "causal context" in bridge
    assert "p(move/face/stop)" in bridge
    assert "raw logits" in bridge
    assert "decision_tensor_sha256" in bridge
