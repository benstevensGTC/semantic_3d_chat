from __future__ import annotations

import hashlib
import json
import math
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest

from semantic_3d_chat.robot.model_bound_action import (
    ModelBoundActionExecutor,
    bind_model_tool_call,
)

_CHECKPOINT = "c" * 64


def _config() -> dict[str, Any]:
    return {
        "scene": {"room_size_m": [6.0, 5.0, 3.0]},
        "robot": {
            "radius_m": 0.25,
            "max_move_m": 0.5,
            "max_move_to_m": 1.0,
            "max_turn_degrees": 45.0,
            "max_look_delta_degrees": 30.0,
            "max_camera_yaw_offset_degrees": 60.0,
            "max_pitch_degrees": 45.0,
        },
    }


class _CollisionMap:
    def __init__(self, *, blocked_x: float | None = None) -> None:
        self.blocked_x = blocked_x
        self.segments: list[tuple[tuple[float, float], tuple[float, float]]] = []

    def segment_check(
        self,
        start: tuple[float, float],
        target: tuple[float, float],
    ) -> SimpleNamespace:
        normalized_start = (float(start[0]), float(start[1]))
        normalized_target = (float(target[0]), float(target[1]))
        self.segments.append((normalized_start, normalized_target))
        collision = self.blocked_x is not None and normalized_target[0] >= self.blocked_x
        return SimpleNamespace(collision=collision, clearance_m=0.0 if collision else 1.0)


class _Runtime:
    def __init__(self, *, collision_map: _CollisionMap | None = None) -> None:
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.stopped = False
        self.version = 0
        self.calls: list[tuple[str, tuple[Any, ...]]] = []
        self.simulator = SimpleNamespace(collision_map=collision_map or _CollisionMap())

    def _digest(self, offset: int) -> str:
        return hashlib.sha256(f"{self.version}:{offset}".encode()).hexdigest()

    def prefix_binding(self) -> dict[str, str]:
        return {
            "active_prefix_sha256": self._digest(1),
            "scene_prefix_sha256": "a" * 64,
            "robot_tokens_sha256": self._digest(2),
        }

    def get_robot_state(self) -> dict[str, Any]:
        return {
            "scene_id": "scene_000001",
            "position_m": [self.x, self.y, 0.0],
            "body_yaw_degrees": self.yaw,
            "camera_yaw_degrees": self.yaw,
            "pitch_degrees": 0.0,
            "stopped": self.stopped,
            **self.prefix_binding(),
        }

    def _receipt(self, name: str) -> dict[str, Any]:
        self.version += 1
        return {"success": True, "error_code": None, "command": name}

    def turn(self, angle_degrees: float) -> dict[str, Any]:
        self.calls.append(("turn", (angle_degrees,)))
        self.yaw += angle_degrees
        return self._receipt("turn")

    def move_forward(self, distance_meters: float) -> dict[str, Any]:
        self.calls.append(("move_forward", (distance_meters,)))
        radians = math.radians(self.yaw)
        self.x += distance_meters * -math.sin(radians)
        self.y += distance_meters * math.cos(radians)
        return self._receipt("move_forward")

    def move_backward(self, distance_meters: float) -> dict[str, Any]:
        self.calls.append(("move_backward", (distance_meters,)))
        radians = math.radians(self.yaw)
        self.x -= distance_meters * -math.sin(radians)
        self.y -= distance_meters * math.cos(radians)
        return self._receipt("move_backward")

    def move_to(self, x: float, y: float) -> dict[str, Any]:
        self.calls.append(("move_to", (x, y)))
        self.x, self.y = x, y
        return self._receipt("move_to")

    def stop(self) -> dict[str, Any]:
        self.calls.append(("stop", ()))
        self.stopped = True
        return self._receipt("stop")


def _bind(
    runtime: _Runtime,
    payload: dict[str, Any],
    *,
    decision_id: str = "decision_00000001",
    checkpoint: str = _CHECKPOINT,
):
    return bind_model_tool_call(
        json.dumps(payload),
        _config(),
        robot_state=runtime.get_robot_state(),
        binding=runtime.prefix_binding(),
        checkpoint_sha256=checkpoint,
        decision_id=decision_id,
    )


@pytest.mark.parametrize(
    ("payload", "expected_call"),
    [
        (
            {"tool": "turn", "arguments": {"angle_degrees": 17.25}},
            ("turn", (17.25,)),
        ),
        (
            {"tool": "move_forward", "arguments": {"distance_meters": 0.37}},
            ("move_forward", (0.37,)),
        ),
        (
            {"tool": "move_backward", "arguments": {"distance_meters": 0.19}},
            ("move_backward", (0.19,)),
        ),
        (
            {"tool": "move_to", "arguments": {"x": 0.37, "y": -0.22}},
            ("move_to", (0.37, -0.22)),
        ),
        ({"tool": "stop", "arguments": {}}, ("stop", ())),
    ],
)
def test_executor_dispatches_exact_model_action_without_substitution(
    payload: dict[str, Any],
    expected_call: tuple[str, tuple[Any, ...]],
) -> None:
    runtime = _Runtime()
    proposal = _bind(runtime, payload)
    result = ModelBoundActionExecutor(
        _config(), checkpoint_sha256=_CHECKPOINT
    ).execute(runtime, proposal)

    assert result["success"] is True
    assert result["executed"] is True
    assert result["executed_tool"] == payload["tool"]
    assert result["executed_arguments"] == payload["arguments"]
    assert result["substitution_applied"] is False
    assert result["synthetic_stop_applied"] is False
    assert runtime.calls == [expected_call]


def test_bound_call_records_canonical_action_and_complete_provenance() -> None:
    runtime = _Runtime()
    raw = '{ "arguments": {"y":-0.22,"x":0.37}, "tool":"move_to" }'
    proposal = bind_model_tool_call(
        raw,
        _config(),
        robot_state=runtime.get_robot_state(),
        binding=runtime.prefix_binding(),
        checkpoint_sha256=_CHECKPOINT,
        decision_id="decision_provenance_01",
    )

    assert proposal.integrity_error() is None
    assert proposal.action() == {
        "tool": "move_to",
        "arguments": {"x": 0.37, "y": -0.22},
    }
    assert proposal.action_sha256 == hashlib.sha256(
        proposal.canonical_action_json.encode()
    ).hexdigest()
    assert proposal.model_output_sha256 == hashlib.sha256(raw.encode()).hexdigest()
    assert proposal.as_dict()["checkpoint_sha256"] == _CHECKPOINT
    assert proposal.as_dict()["decision_id"] == "decision_provenance_01"
    assert len(proposal.provenance_sha256) == 64


def test_tampered_action_is_rejected_without_motion_or_synthetic_stop() -> None:
    runtime = _Runtime()
    proposal = _bind(
        runtime,
        {"tool": "turn", "arguments": {"angle_degrees": 10.0}},
    )
    tampered = replace(
        proposal,
        canonical_action_json=(
            '{"arguments":{"angle_degrees":20.0},"tool":"turn"}'
        ),
    )

    result = ModelBoundActionExecutor(
        _config(), checkpoint_sha256=_CHECKPOINT
    ).execute(runtime, tampered)

    assert result["error_code"] == "E_MODEL_ACTION_TAMPERED"
    assert result["executed"] is False
    assert result["synthetic_stop_applied"] is False
    assert runtime.calls == []


def test_stale_context_and_wrong_checkpoint_are_rejected_before_motion() -> None:
    runtime = _Runtime()
    stale = _bind(
        runtime,
        {"tool": "turn", "arguments": {"angle_degrees": 10.0}},
        decision_id="decision_stale_01",
    )
    runtime.version += 1
    stale_result = ModelBoundActionExecutor(
        _config(), checkpoint_sha256=_CHECKPOINT
    ).execute(runtime, stale)

    fresh_runtime = _Runtime()
    wrong_checkpoint = _bind(
        fresh_runtime,
        {"tool": "stop", "arguments": {}},
        decision_id="decision_checkpoint_01",
        checkpoint="d" * 64,
    )
    checkpoint_result = ModelBoundActionExecutor(
        _config(), checkpoint_sha256=_CHECKPOINT
    ).execute(fresh_runtime, wrong_checkpoint)

    assert stale_result["error_code"] == "E_MODEL_CONTEXT_STALE"
    assert checkpoint_result["error_code"] == "E_MODEL_CHECKPOINT"
    assert runtime.calls == fresh_runtime.calls == []


def test_colliding_model_move_is_rejected_without_reroute_or_stop() -> None:
    collision = _CollisionMap(blocked_x=0.3)
    runtime = _Runtime(collision_map=collision)
    proposal = _bind(
        runtime,
        {"tool": "move_to", "arguments": {"x": 0.4, "y": 0.0}},
    )

    result = ModelBoundActionExecutor(
        _config(), checkpoint_sha256=_CHECKPOINT
    ).execute(runtime, proposal)

    assert result["error_code"] == "E_MODEL_COLLISION"
    assert result["executed"] is False
    assert result["executed_tool"] is None
    assert result["substitution_applied"] is False
    assert result["synthetic_stop_applied"] is False
    assert runtime.calls == []
    assert collision.segments == [((0.0, 0.0), (0.4, 0.0))]


@pytest.mark.parametrize("tool", ["scan", "look", "get_robot_state", "reset_scene"])
def test_non_motion_production_actions_are_rejected_without_dispatch(tool: str) -> None:
    runtime = _Runtime()
    arguments: dict[str, Any]
    if tool == "look":
        arguments = {"yaw_delta_degrees": 5.0, "pitch_delta_degrees": 0.0}
    elif tool == "reset_scene":
        arguments = {"scene_id": "scene_000001", "seed": 7}
    else:
        arguments = {}
    proposal = _bind(runtime, {"tool": tool, "arguments": arguments})

    result = ModelBoundActionExecutor(
        _config(), checkpoint_sha256=_CHECKPOINT
    ).execute(runtime, proposal)

    assert result["error_code"] == "E_MODEL_ACTION_FORBIDDEN"
    assert result["executed"] is False
    assert runtime.calls == []


def test_model_decision_is_one_time_and_replay_does_not_repeat_action() -> None:
    runtime = _Runtime()
    proposal = _bind(
        runtime,
        {"tool": "turn", "arguments": {"angle_degrees": -12.5}},
    )
    executor = ModelBoundActionExecutor(_config(), checkpoint_sha256=_CHECKPOINT)

    first = executor.execute(runtime, proposal)
    second = executor.execute(runtime, proposal)

    assert first["success"] is True
    assert second["error_code"] == "E_MODEL_DECISION_REPLAY"
    assert runtime.calls == [("turn", (-12.5,))]


def test_out_of_bounds_output_never_becomes_a_bound_action() -> None:
    runtime = _Runtime()

    with pytest.raises(ValueError, match="E_LIMIT"):
        _bind(
            runtime,
            {"tool": "turn", "arguments": {"angle_degrees": 90.0}},
        )

    assert runtime.calls == []
