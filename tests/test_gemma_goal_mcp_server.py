from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Mapping
from typing import Any

import pytest
from mcp.server.mcpserver.exceptions import ToolError

from semantic_3d_chat.mcp_server.gemma_goal_server import (
    SCHEMA,
    build_gemma_goal_server,
    goal_response,
)
from semantic_3d_chat.robot.gemma_waypoint_runtime import (
    GOAL_RESULT_SCHEMA,
    STEP_RECEIPT_SCHEMA,
)
from semantic_3d_chat.robot.practical_rover import PracticalRoverController

SCENE_HASH = "1" * 64
ACTIVE_HASH = "2" * 64
ROBOT_HASH = "3" * 64
CHECKPOINT_HASH = "4" * 64
RUNTIME_BINDING_HASH = "5" * 64


def _step(
    goal: str,
    step: int,
    action: str,
    *,
    success: bool,
    executed: bool,
    error_code: str | None = None,
) -> dict[str, Any]:
    primitive = {"face": "turn", "move_to": "move_to", "stop": "stop"}[action]
    arguments: dict[str, float]
    if action == "face":
        arguments = {"angle_degrees": 22.5}
    elif action == "move_to":
        arguments = {"x": -0.25 + step / 10.0, "y": 0.5}
    else:
        arguments = {}
    return {
        "schema": STEP_RECEIPT_SCHEMA,
        "step": step,
        "decision_id": f"gwp_00000001_{step:03d}",
        "model_action": action,
        "primitive_tool": primitive,
        "primitive_arguments": arguments,
        "model_action_logits": [1.0, 0.0, -1.0],
        "model_action_probabilities": [0.7, 0.2, 0.1],
        "decision_tensor_sha256": f"{step + 5:x}" * 64,
        "instruction_sha256": hashlib.sha256(goal.encode()).hexdigest(),
        "active_prefix_sha256": ACTIVE_HASH,
        "scene_prefix_sha256": SCENE_HASH,
        "robot_tokens_sha256": ROBOT_HASH,
        "checkpoint_sha256": CHECKPOINT_HASH,
        "scene_token_count": 258,
        "robot_token_count": 4,
        "history_token_count": step - 1,
        "prompt_token_count": 64,
        "decision_position": 300 + step,
        "model_waypoint_delta_robot_m": [0.1, 0.2],
        "model_turn_delta_degrees": 22.5,
        "model_desired_heading_degrees": -67.5,
        "bound_proposal": None,
        "history_row": [0.0],
        "execution": {
            "success": success,
            "executed": executed,
            "error_code": error_code,
            "substitution_applied": False,
            "synthetic_stop_applied": False,
        },
        "actual_gemma_causal_forward": True,
        "model_selected_every_waypoint_and_heading": True,
        "deterministic_route_planner_used": False,
        "substitution_applied": False,
        "synthetic_stop_applied": False,
    }


def _payload(goal: str) -> dict[str, Any]:
    decisions = [
        _step(goal, 1, "face", success=True, executed=True),
        _step(
            goal,
            2,
            "move_to",
            success=False,
            executed=False,
            error_code="E_MODEL_COLLISION",
        ),
        _step(goal, 3, "move_to", success=True, executed=True),
        _step(goal, 4, "stop", success=True, executed=True),
    ]
    return {
        "schema": "semantic_3d_chat.practical_rover.v1",
        "success": True,
        "error_code": None,
        "state": {
            "scene_id": "scene_000001",
            "scene_version": 0,
            "position_m": [-0.05, 0.5, 0.0],
            "body_yaw_degrees": -67.5,
            "collision": False,
            "stopped": False,
            "action_count": 3,
        },
        "model_decisions": decisions,
        "scene_prefix_hash": SCENE_HASH,
        "active_prefix_hash": ACTIVE_HASH,
        "scene_memory": {
            "sha256": SCENE_HASH,
            "source_voxels": 74_699,
            "processed_voxels": 8_422,
            "semantic_feature_dim": 3_072,
            "token_count": 258,
            "model_dim": 1_536,
            "robot_state_token_count": 4,
            "all_runtime_voxels_encoded": True,
            "question_dependent_scene_retrieval": False,
            "environmental_text_inputs": [],
            "loaded_file_audit": {
                "forbidden_access_count": 0,
                "passed": True,
            },
        },
        "decision_source": "actual_local_gemma_model_only_waypoint_policy",
        "navigation_control_mode": "actual_local_gemma_model_only_waypoint_policy",
        "navigation_checkpoint_sha256": CHECKPOINT_HASH,
        "gemma_runtime_binding_sha256": RUNTIME_BINDING_HASH,
        "high_level_natural_language_only": True,
        "model_selects_every_waypoint_and_heading": True,
        "model_selects_stop": True,
        "model_stop_emitted": True,
        "local_inference": True,
        "cloud_model_used": False,
        "continuous_scene_memory": True,
        "continuous_robot_state": True,
        "fallback_used": False,
        "deterministic_route_planner_used": False,
        "synthetic_stop_applied": False,
        "substitution_applied": False,
        "environmental_text_inputs": [],
    }


def _startup() -> dict[str, Any]:
    return {
        "ready": True,
        "control_mode": "actual_local_gemma_model_only_waypoint_policy",
        "high_level_natural_language_only": True,
        "model_selects_every_waypoint_and_heading": True,
        "model_selects_stop": True,
        "local_inference": True,
        "cloud_model_used": False,
        "fallback_used": False,
        "deterministic_route_planner_used": False,
        "synthetic_stop_applied": False,
        "substitution_applied": False,
        "environmental_text_inputs": [],
        "scene_prefix_hash": SCENE_HASH,
        "navigation_checkpoint_sha256": CHECKPOINT_HASH,
    }


class _FakeController:
    def __init__(self) -> None:
        self.goals: list[str] = []

    def startup(self) -> dict[str, Any]:
        return _startup()

    def navigate_goal(self, goal: str) -> dict[str, Any]:
        self.goals.append(goal)
        return _payload(goal)


def test_goal_mcp_exposes_only_one_high_level_tool_and_gemma_receipts() -> None:
    controller = _FakeController()
    server = build_gemma_goal_server(controller)  # type: ignore[arg-type]

    async def exercise() -> None:
        tools = await server.list_tools()
        assert [tool.name for tool in tools] == ["navigate"]
        assert tools[0].input_schema["additionalProperties"] is False
        assert set(tools[0].input_schema["properties"]) == {"goal"}
        for forbidden in (
            "turn",
            "move_to",
            "move_forward",
            "move_backward",
            "look",
            "stop",
            "scan",
        ):
            assert forbidden not in {tool.name for tool in tools}
        with pytest.raises(ToolError):
            await server.call_tool("navigate", {"goal": "Do a lap", "x": 1.0})
        result = await server.call_tool(
            "navigate", {"goal": "Go around the obstacle and stop by the goal"}
        )
        assert result.structured_content is not None
        payload = result.structured_content
        assert payload["schema"] == SCHEMA
        assert payload["success"] is True
        assert payload["model_decision_count"] == 4
        assert payload["accepted_decision_count"] == 3
        assert payload["rejected_decision_count"] == 1
        assert [item["model_action"] for item in payload["decisions"]] == [
            "face",
            "move_to",
            "move_to",
            "stop",
        ]
        assert payload["decisions"][1]["accepted"] is False
        assert payload["decisions"][1]["error_code"] == "E_MODEL_COLLISION"
        assert payload["fallback_used"] is False
        assert payload["substitution_applied"] is False
        assert payload["synthetic_stop_applied"] is False
        assert payload["deterministic_route_planner_used"] is False
        assert payload["scene_prefix_sha256"] == SCENE_HASH
        assert controller.goals == ["Go around the obstacle and stop by the goal"]
        serialized = str(payload).casefold()
        for prohibited in (
            "object_name",
            "caption",
            "scene_graph",
            "oracle",
            "relationship",
        ):
            assert prohibited not in serialized

    asyncio.run(exercise())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("fallback_used", True),
        ("substitution_applied", True),
        ("synthetic_stop_applied", True),
        ("deterministic_route_planner_used", True),
    ],
)
def test_goal_response_fails_closed_on_any_non_model_control(
    field: str,
    value: bool,
) -> None:
    goal = "Complete a lap and stop"
    payload = _payload(goal)
    payload[field] = value

    with pytest.raises(RuntimeError, match="nonexclusive Gemma"):
        goal_response(payload, goal)


def test_goal_response_requires_an_executed_gemma_terminal_stop() -> None:
    goal = "Face the goal and stop"
    payload = _payload(goal)
    payload["model_decisions"][-1]["model_action"] = "face"
    payload["model_decisions"][-1]["primitive_tool"] = "turn"
    payload["model_decisions"][-1]["primitive_arguments"] = {"angle_degrees": 1.0}
    payload["model_decisions"][-1]["model_turn_delta_degrees"] = 1.0

    with pytest.raises(RuntimeError, match="accepted terminal STOP"):
        goal_response(payload, goal)


class _GoalResult:
    def __init__(self, payload: Mapping[str, Any]) -> None:
        self.payload = dict(payload)

    def as_dict(self) -> dict[str, Any]:
        return self.payload


class _WaypointController:
    checkpoint_sha256 = CHECKPOINT_HASH

    def __init__(self, goal: str) -> None:
        self.goal = goal
        self.calls: list[tuple[str, int]] = []
        step = _step(goal, 1, "stop", success=True, executed=True)
        self.result = _GoalResult(
            {
                "schema": GOAL_RESULT_SCHEMA,
                "success": True,
                "termination": "model_stop",
                "error_code": None,
                "instruction_sha256": hashlib.sha256(goal.encode()).hexdigest(),
                "checkpoint_sha256": CHECKPOINT_HASH,
                "model_stop_emitted": True,
                "step_count": 1,
                "steps": [step],
                "synthetic_stop_applied": False,
                "substitution_applied": False,
                "deterministic_route_planner_used": False,
            }
        )

    def run(self, goal: str, *, max_steps: int) -> _GoalResult:
        self.calls.append((goal, max_steps))
        return self.result


class _NavigateHarness:
    def __init__(self, goal: str) -> None:
        self._lock = __import__("threading").RLock()
        self._model_only_waypoint_controller = _WaypointController(goal)
        self.gemma_waypoint_controller = self._model_only_waypoint_controller
        self.envelope_result: Mapping[str, Any] | None = None

    def _assert_open(self) -> None:
        return None

    def _history_index(self) -> int:
        return 0

    def _envelope(self, result: Mapping[str, Any], **kwargs: Any) -> dict[str, Any]:
        self.envelope_result = result
        return {"result": dict(result), "envelope": kwargs}


def test_transport_navigation_entrypoint_passes_original_goal_directly_to_gemma() -> None:
    goal = "Figure out a safe multipath lap, return here, and stop"
    harness = _NavigateHarness(goal)

    response = PracticalRoverController.navigate_goal(harness, f"  {goal}  ")  # type: ignore[arg-type]

    assert harness._model_only_waypoint_controller.calls == [(goal, 128)]
    assert response["envelope"] == {
        "history_index": 0,
        "decision_source": "actual_local_gemma_model_only_waypoint_policy",
        "gemma_attempted": True,
        "gemma_accepted": True,
        "fallback_used": False,
    }
    assert response["result"]["deterministic_route_planner_used"] is False
    assert response["result"]["synthetic_stop_applied"] is False
    assert response["result"]["substitution_applied"] is False
