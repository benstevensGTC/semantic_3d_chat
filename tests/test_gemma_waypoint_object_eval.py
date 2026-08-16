from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from semantic_3d_chat.evaluation import gemma_waypoint_object_eval as evaluation


def _control(*, attempted: bool) -> dict[str, Any]:
    return {
        "control_mode": evaluation.CONTROL_MODE,
        "gemma_attempted": attempted,
        "gemma_accepted": attempted,
        "fallback_used": False,
        "local_inference": True,
        "cloud_model_used": False,
        "high_level_natural_language_only": True,
        "task_trained_navigation": True,
        "untrained_json_backend_enabled": False,
        "static_precomputed_scene_memory": True,
        "camera_control_input": False,
    }


def _memory() -> dict[str, Any]:
    return {
        "tensor_shape": [1, 258, 1536],
        "active_tensor_shape": [1, 262, 1536],
        "sha256": "a" * 64,
        "token_count": 258,
        "model_dim": 1536,
        "robot_state_token_count": 4,
        "source_voxels": 74_699,
        "processed_voxels": 8_422,
        "semantic_feature_dim": 3_072,
        "map_version": 1,
        "all_runtime_voxels_encoded": True,
        "base_adapter_weights_loaded": True,
        "control_weights_loaded": True,
        "control_training_gate_passed": True,
        "question_dependent_scene_retrieval": False,
        "environmental_text_inputs": [],
        "loaded_file_audit": {"passed": True, "forbidden_access_count": 0},
    }


def _state(*, xy: tuple[float, float], yaw: float, action_count: int) -> dict[str, Any]:
    return {
        "scene_id": "scene_000001",
        "position_xy_m": list(xy),
        "body_yaw_degrees": yaw,
        "collision": False,
        "stopped": action_count > 0,
        "action_count": action_count,
        "map_version": 1,
        "scan_count": 0,
        "scene_prefix_hash": "a" * 64,
    }


def _decision(step: int, action: str) -> dict[str, Any]:
    return {
        "step": step,
        "model_action": action,
        "model_waypoint_delta_robot_m": [0.0, 0.0],
        "model_desired_heading_degrees": -45.0,
        "primitive_tool": {"face": "turn", "move_to": "move_to", "stop": "stop"}[action],
        "primitive_arguments": {},
        "accepted": True,
        "executed": True,
        "error_code": None,
        "actual_gemma_causal_forward": True,
        "model_selected_every_waypoint_and_heading": True,
        "deterministic_route_planner_used": False,
        "substitution_applied": False,
        "synthetic_stop_applied": False,
    }


def _payload(
    *,
    attempted: bool,
    xy: tuple[float, float] = (-0.5, -0.25),
    yaw: float = -90.0,
    first_action: str = "face",
) -> dict[str, Any]:
    action_count = 2 if attempted else 0
    return {
        "state": _state(xy=xy, yaw=yaw, action_count=action_count),
        "scene_memory": _memory(),
        "control": _control(attempted=attempted),
        "model_decisions": (
            [_decision(1, first_action), _decision(2, "stop")] if attempted else []
        ),
        "actions": (
            [
                {
                    "position_xy_m": list(xy),
                    "distance_moved": 0.4 if first_action == "move_to" else 0.0,
                    "success": True,
                    "collision": False,
                    "stopped": False,
                },
                {
                    "position_xy_m": list(xy),
                    "distance_moved": 0.0,
                    "success": True,
                    "collision": False,
                    "stopped": True,
                },
            ]
            if attempted
            else []
        ),
    }


def _capture(goal_id: str, final: dict[str, Any]) -> dict[str, Any]:
    calls: list[tuple[str, object]] = []

    def request(_origin: str, path: str, payload: object, _timeout: float) -> dict[str, Any]:
        calls.append((path, payload))
        return _payload(attempted=False) if path == "/api/state" else final

    result = evaluation.capture_live_goal("http://127.0.0.1:8770", goal_id, request_fn=request)
    assert calls == [
        ("/api/state", None),
        ("/api/instruction", {"instruction": evaluation.GOALS[goal_id].instruction}),
    ]
    return result


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _oracle() -> dict[str, Any]:
    return {
        "scene_id": "scene_000001",
        "instances": [
            {
                "instance_id": "i_cube",
                "category": "cube",
                "expected_center_xyz_m": [0.56, 0.61, 0.96],
                "bbox": {
                    "min_xyz_m": [0.42, 0.47, 0.81],
                    "max_xyz_m": [0.70, 0.75, 1.11],
                },
            },
            {
                "instance_id": "i_chair",
                "category": "chair",
                "expected_center_xyz_m": [-1.24, 0.52, 0.63],
                "bbox": {
                    "min_xyz_m": [-1.51, 0.25, 0.0],
                    "max_xyz_m": [-0.97, 0.79, 1.26],
                },
            },
        ],
    }


def test_capture_uses_only_public_loopback_api_and_contains_no_oracle_metadata() -> None:
    cube = (0.56, 0.61)
    robot = (-0.5, -0.25)
    expected_yaw = math.degrees(math.atan2(-(cube[0] - robot[0]), cube[1] - robot[1]))
    runtime = _capture("face_cube", _payload(attempted=True, xy=robot, yaw=expected_yaw))

    assert runtime["model_selected_every_waypoint_and_heading"] is True
    assert runtime["deterministic_route_planner_used"] is False
    assert runtime["model_selected_terminal_stop"] is True
    assert "oracle" not in json.dumps(runtime, sort_keys=True).casefold()


def test_capture_rejects_non_loopback_before_making_a_request() -> None:
    called = False

    def request(*_args: object) -> dict[str, Any]:
        nonlocal called
        called = True
        return {}

    with pytest.raises(ValueError, match="loopback"):
        evaluation.capture_live_goal("https://example.com", "face_cube", request_fn=request)
    assert called is False


def test_scores_face_and_approach_after_both_runtime_captures_complete(
    tmp_path: Path,
) -> None:
    cube = (0.56, 0.61)
    start = (-0.5, -0.25)
    expected_yaw = math.degrees(math.atan2(-(cube[0] - start[0]), cube[1] - start[1]))
    face = _capture("face_cube", _payload(attempted=True, xy=start, yaw=expected_yaw))
    approach = _capture(
        "approach_chair",
        _payload(attempted=True, xy=(-1.0, 0.0), yaw=0.0, first_action="move_to"),
    )
    face_path = tmp_path / "face.json"
    approach_path = tmp_path / "approach.json"
    oracle_path = tmp_path / "oracle" / "scene_000001" / "oracle.json"
    _write(face_path, face)
    _write(approach_path, approach)
    _write(oracle_path, _oracle())

    report = evaluation.score_runtime_files(
        [face_path, approach_path], oracle_root=tmp_path / "oracle"
    )

    assert report["all_passed"] is True
    assert report["all_runtime_evidence_validated_before_oracle_open"] is True
    assert report["runtime_process_read_oracle"] is False
    rows = {row["goal_id"]: row for row in report["goals"]}
    assert rows["face_cube"]["oracle_yaw_error_degrees"] == pytest.approx(0.0)
    assert rows["approach_chair"]["target_center_progress_m"] > 0.25
    assert rows["approach_chair"]["final_oracle_bbox_standoff_m"] == pytest.approx(0.25)


def test_invalid_runtime_is_rejected_before_oracle_path_is_opened(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime_path = tmp_path / "bad_runtime.json"
    oracle_path = tmp_path / "oracle" / "scene_000001" / "oracle.json"
    _write(runtime_path, {"schema": "wrong"})
    _write(oracle_path, _oracle())
    original = evaluation._read_object
    opened: list[Path] = []

    def tracked(path: Path) -> dict[str, Any]:
        opened.append(path)
        return original(path)

    monkeypatch.setattr(evaluation, "_read_object", tracked)
    with pytest.raises(ValueError, match="wrong schema"):
        evaluation.score_runtime_files([runtime_path], oracle_root=tmp_path / "oracle")

    assert opened == [runtime_path.resolve()]
