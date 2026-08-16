from __future__ import annotations

import hashlib

import pytest
import torch

from semantic_3d_chat.language.prefix_injection import prefix_sha256
from semantic_3d_chat.robot.action_context import (
    capture_continuous_action_context,
    require_grounding_map_binding,
)
from semantic_3d_chat.robot.state_encoder import (
    NumericRobotState,
    robot_state_vector,
    robot_state_vector_sha256,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _state(*, x: float = 0.25, yaw: float = 15.0) -> NumericRobotState:
    return NumericRobotState(
        position_m=(x, -0.5, 0.0),
        body_yaw_degrees=yaw,
        camera_yaw_degrees=yaw,
        pitch_degrees=0.0,
        linear_velocity_xy_m=(0.0, 0.0),
        angular_velocity_degrees=0.0,
        collision=False,
        last_movement_delta_m=(0.0, 0.0, 0.0),
        scan_coverage=0.25,
        stopped=False,
    )


def _state_hash(state: NumericRobotState) -> str:
    features = robot_state_vector(
        state,
        torch.tensor([-3.0, -2.5, 0.0]),
        torch.tensor([3.0, 2.5, 3.0]),
    )
    return robot_state_vector_sha256(features)


class _Runtime:
    def __init__(self, state: NumericRobotState) -> None:
        self.prefix = torch.arange(72, dtype=torch.float32).reshape(1, 6, 12)
        self.state = state
        self.binding = {
            "active_prefix_sha256": prefix_sha256(self.prefix),
            "scene_prefix_sha256": _digest("scene"),
            "map_sha256": _digest("map"),
            "robot_state_sha256": _state_hash(state),
            "robot_tokens_sha256": _digest("robot tokens"),
        }

    def continuous_action_context_snapshot(self):
        return self.prefix.clone(), dict(self.binding), self.state


def test_context_binds_exact_prefix_map_and_numeric_robot_state() -> None:
    runtime = _Runtime(_state())
    context = capture_continuous_action_context(runtime, [6.0, 5.0, 3.0])

    assert torch.equal(context.active_prefix, runtime.prefix)
    assert context.map_sha256 == runtime.binding["map_sha256"]
    assert robot_state_vector_sha256(context.state_features) == runtime.binding[
        "robot_state_sha256"
    ]
    require_grounding_map_binding(
        context,
        grounding_map_sha256=context.map_sha256,
        scored_voxels=123,
        available_voxels=123,
    )


def test_context_rejects_pose_not_used_to_build_robot_tokens() -> None:
    runtime = _Runtime(_state())
    runtime.state = _state(x=0.75)

    with pytest.raises(RuntimeError, match="differs from the bound robot tokens"):
        capture_continuous_action_context(runtime, [6.0, 5.0, 3.0])


def test_context_rejects_wrong_or_incomplete_grounding_map() -> None:
    context = capture_continuous_action_context(_Runtime(_state()), [6.0, 5.0, 3.0])

    with pytest.raises(RuntimeError, match="differs from the action scene prefix"):
        require_grounding_map_binding(
            context,
            grounding_map_sha256=_digest("other map"),
            scored_voxels=123,
            available_voxels=123,
        )
    with pytest.raises(RuntimeError, match="complete active map"):
        require_grounding_map_binding(
            context,
            grounding_map_sha256=context.map_sha256,
            scored_voxels=122,
            available_voxels=123,
        )
