"""One-tool MCP surface for the production local-Gemma rover.

The legacy numerical primitive MCP server remains in :mod:`server`.  This
module is deliberately separate: its only tool accepts an outcome-level text
goal, then the promoted causal Gemma waypoint policy chooses every exact FACE
angle, robot-relative MOVE_TO waypoint, retry, and STOP.  Deterministic code is
limited to coordinate conversion, bounded execution, and safety rejection.

Run locally over stdio with::

    python -m semantic_3d_chat.mcp_server.gemma_goal_server

No environmental text, label, caption, object inventory, or oracle metadata is
returned through MCP.  The response contains numeric state plus authenticated
continuous-prefix and model-decision provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from typing import Annotated, Any, Final, Literal

from mcp.server.mcpserver import MCPServer
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from semantic_3d_chat.robot.practical_rover import (
    DEFAULT_ASSET,
    DEFAULT_BASE_CHECKPOINT,
    DEFAULT_CONFIG,
    DEFAULT_CONTROL_CHECKPOINT,
    DEFAULT_CONTROL_CONFIG,
    DEFAULT_NAVIGATION_CHECKPOINT,
    DEFAULT_ROBOT_STATE_CHECKPOINT,
    DEFAULT_SCENE,
    PracticalRoverController,
    build_local_practical_rover,
    practical_rover_preflight,
)

SCHEMA: Final[str] = "semantic_3d_chat.gemma_goal_mcp.v1"
TOOL_NAME: Final[str] = "navigate"
SERVER_NAME: Final[str] = "semantic-3d-gemma-goal-rover"
MAX_GOAL_CHARACTERS: Final[int] = 4096

_SHA256 = re.compile(r"[0-9a-f]{64}")
_SCENE_ID = re.compile(r"scene_[0-9]{6}")
_MODEL_ACTION_TO_PRIMITIVE: Final[dict[str, str]] = {
    "face": "turn",
    "move_to": "move_to",
    "stop": "stop",
}

FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
OpaqueSceneId = Annotated[str, StringConstraints(pattern=r"^scene_[0-9]{6}$")]
ProtocolErrorCode = Annotated[str, StringConstraints(pattern=r"^E_[A-Z0-9_]+$")]
GoalText = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=MAX_GOAL_CHARACTERS,
    ),
]


class GemmaDecisionReceipt(BaseModel):
    """Authenticated, label-free summary of one causal Gemma motion decision."""

    model_config = ConfigDict(extra="forbid")

    step: Annotated[int, Field(ge=1, le=128)]
    model_action: Literal["face", "move_to", "stop"]
    primitive_tool: Literal["turn", "move_to", "stop"]
    accepted: bool
    executed: bool
    error_code: ProtocolErrorCode | None
    turn_delta_degrees: FiniteFloat | None
    waypoint_xy_m: Annotated[list[FiniteFloat], Field(min_length=2, max_length=2)] | None
    desired_heading_degrees: FiniteFloat
    waypoint_delta_robot_m: Annotated[list[FiniteFloat], Field(min_length=2, max_length=2)]
    actual_gemma_causal_forward: Literal[True]
    decision_tensor_sha256: Sha256
    active_prefix_sha256: Sha256
    robot_tokens_sha256: Sha256
    scene_prefix_sha256: Sha256
    navigation_checkpoint_sha256: Sha256
    substitution_applied: Literal[False]
    synthetic_stop_applied: Literal[False]
    deterministic_route_planner_used: Literal[False]


class GemmaGoalResponse(BaseModel):
    """Structured MCP result with numeric state and continuous-policy evidence."""

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        serialize_by_alias=True,
    )

    schema_name: Literal["semantic_3d_chat.gemma_goal_mcp.v1"] = Field(alias="schema")
    success: bool
    error_code: ProtocolErrorCode | None
    scene_id: OpaqueSceneId
    goal_sha256: Sha256
    scene_prefix_sha256: Sha256
    active_prefix_sha256: Sha256
    navigation_checkpoint_sha256: Sha256
    gemma_runtime_binding_sha256: Sha256
    position_m: Annotated[list[FiniteFloat], Field(min_length=3, max_length=3)]
    body_yaw_degrees: FiniteFloat
    collision: bool
    stopped: bool
    action_count: Annotated[int, Field(ge=0)]
    map_version: Annotated[int, Field(ge=0)]
    source_voxels: Annotated[int, Field(ge=1)]
    processed_voxels: Annotated[int, Field(ge=1)]
    semantic_feature_dim: Annotated[int, Field(ge=1)]
    scene_token_count: Annotated[int, Field(ge=3)]
    scene_model_dim: Annotated[int, Field(ge=1)]
    robot_state_token_count: Annotated[int, Field(ge=1)]
    model_decision_count: Annotated[int, Field(ge=1, le=128)]
    accepted_decision_count: Annotated[int, Field(ge=0, le=128)]
    rejected_decision_count: Annotated[int, Field(ge=0, le=128)]
    model_stop_emitted: bool
    local_inference: Literal[True]
    cloud_model_used: Literal[False]
    continuous_scene_memory: Literal[True]
    continuous_robot_state: Literal[True]
    model_selects_every_waypoint_and_heading: Literal[True]
    model_selects_stop: Literal[True]
    fallback_used: Literal[False]
    substitution_applied: Literal[False]
    synthetic_stop_applied: Literal[False]
    deterministic_route_planner_used: Literal[False]
    question_dependent_scene_retrieval: Literal[False]
    runtime_forbidden_access_count: Literal[0]
    decisions: Annotated[list[GemmaDecisionReceipt], Field(min_length=1, max_length=128)]


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _scene_id(value: object) -> str:
    if not isinstance(value, str) or _SCENE_ID.fullmatch(value) is None:
        raise ValueError("MCP response scene_id must be opaque")
    return value


def _strict_numeric_vector(
    value: object,
    name: str,
    *,
    length: int,
) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{name} must contain exactly {length} values")
    return [_finite(item, f"{name}[{index}]") for index, item in enumerate(value)]


def _decision_receipt(
    value: Mapping[str, Any],
    *,
    expected_step: int,
    scene_prefix_sha256: str,
    navigation_checkpoint_sha256: str,
) -> GemmaDecisionReceipt:
    action = value.get("model_action")
    primitive = value.get("primitive_tool")
    if (
        action not in _MODEL_ACTION_TO_PRIMITIVE
        or primitive != _MODEL_ACTION_TO_PRIMITIVE[action]
        or value.get("step") != expected_step
        or value.get("actual_gemma_causal_forward") is not True
        or value.get("model_selected_every_waypoint_and_heading") is not True
        or value.get("deterministic_route_planner_used") is not False
        or value.get("substitution_applied") is not False
        or value.get("synthetic_stop_applied") is not False
        or value.get("scene_prefix_sha256") != scene_prefix_sha256
        or value.get("checkpoint_sha256") != navigation_checkpoint_sha256
    ):
        raise RuntimeError("MCP rejected a decision without exact Gemma provenance")
    execution = value.get("execution")
    arguments = value.get("primitive_arguments")
    if not isinstance(execution, Mapping) or not isinstance(arguments, Mapping):
        raise TypeError("MCP received a malformed Gemma execution receipt")
    if (
        type(execution.get("success")) is not bool
        or type(execution.get("executed")) is not bool
        or execution.get("substitution_applied") is not False
        or execution.get("synthetic_stop_applied") is not False
    ):
        raise RuntimeError("MCP rejected an unauthenticated Gemma execution receipt")
    if execution["success"] is True and (
        execution["executed"] is not True or execution.get("error_code") is not None
    ):
        raise RuntimeError("MCP rejected an inconsistent accepted Gemma decision")
    if execution["success"] is False and execution.get("error_code") is None:
        raise RuntimeError("MCP rejected a safety rejection without an error code")
    expected_argument_names = {
        "face": {"angle_degrees"},
        "move_to": {"x", "y"},
        "stop": set(),
    }[action]
    if set(arguments) != expected_argument_names:
        raise RuntimeError("MCP rejected changed Gemma primitive arguments")
    turn_delta = _finite(arguments["angle_degrees"], "angle_degrees") if action == "face" else None
    raw_turn_delta = _finite(
        value.get("model_turn_delta_degrees"),
        "model_turn_delta_degrees",
    )
    if turn_delta is not None and not math.isclose(
        turn_delta,
        raw_turn_delta,
        rel_tol=0.0,
        abs_tol=1e-8,
    ):
        raise RuntimeError("MCP rejected a FACE angle changed after Gemma inference")
    waypoint = (
        [
            _finite(arguments["x"], "waypoint x"),
            _finite(arguments["y"], "waypoint y"),
        ]
        if action == "move_to"
        else None
    )
    error_code = execution.get("error_code")
    if error_code is not None and (
        not isinstance(error_code, str) or re.fullmatch(r"E_[A-Z0-9_]+", error_code) is None
    ):
        raise ValueError("MCP execution error_code is invalid")
    return GemmaDecisionReceipt.model_validate(
        {
            "step": expected_step,
            "model_action": action,
            "primitive_tool": primitive,
            "accepted": execution["success"],
            "executed": execution["executed"],
            "error_code": error_code,
            "turn_delta_degrees": turn_delta,
            "waypoint_xy_m": waypoint,
            "desired_heading_degrees": _finite(
                value.get("model_desired_heading_degrees"),
                "model_desired_heading_degrees",
            ),
            "waypoint_delta_robot_m": _strict_numeric_vector(
                value.get("model_waypoint_delta_robot_m"),
                "model_waypoint_delta_robot_m",
                length=2,
            ),
            "actual_gemma_causal_forward": True,
            "decision_tensor_sha256": _sha256(
                value.get("decision_tensor_sha256"), "decision_tensor_sha256"
            ),
            "active_prefix_sha256": _sha256(
                value.get("active_prefix_sha256"), "active_prefix_sha256"
            ),
            "robot_tokens_sha256": _sha256(value.get("robot_tokens_sha256"), "robot_tokens_sha256"),
            "scene_prefix_sha256": scene_prefix_sha256,
            "navigation_checkpoint_sha256": navigation_checkpoint_sha256,
            "substitution_applied": False,
            "synthetic_stop_applied": False,
            "deterministic_route_planner_used": False,
        }
    )


def goal_response(payload: Mapping[str, Any], submitted_goal: str) -> GemmaGoalResponse:
    """Fail closed and reduce a practical-rover envelope to the MCP allowlist."""

    if not isinstance(submitted_goal, str) or not submitted_goal.strip():
        raise ValueError("submitted_goal must be non-empty text")
    if submitted_goal != submitted_goal.strip():
        raise ValueError("submitted_goal must already be stripped")
    if (
        payload.get("decision_source") != "actual_local_gemma_model_only_waypoint_policy"
        or payload.get("navigation_control_mode") != "actual_local_gemma_model_only_waypoint_policy"
        or payload.get("high_level_natural_language_only") is not True
        or payload.get("model_selects_every_waypoint_and_heading") is not True
        or payload.get("model_selects_stop") is not True
        or payload.get("local_inference") is not True
        or payload.get("cloud_model_used") is not False
        or payload.get("continuous_scene_memory") is not True
        or payload.get("continuous_robot_state") is not True
        or payload.get("fallback_used") is not False
        or payload.get("deterministic_route_planner_used") is not False
        or payload.get("synthetic_stop_applied") is not False
        or payload.get("substitution_applied") is not False
        or payload.get("environmental_text_inputs") != []
    ):
        raise RuntimeError("MCP rejected a nonexclusive Gemma navigation result")
    state = payload.get("state")
    memory = payload.get("scene_memory")
    raw_decisions = payload.get("model_decisions")
    if (
        not isinstance(state, Mapping)
        or not isinstance(memory, Mapping)
        or not isinstance(raw_decisions, list)
        or not raw_decisions
        or len(raw_decisions) > 128
    ):
        raise TypeError("MCP received an incomplete Gemma navigation result")
    scene_hash = _sha256(payload.get("scene_prefix_hash"), "scene_prefix_hash")
    active_hash = _sha256(payload.get("active_prefix_hash"), "active_prefix_hash")
    navigation_hash = _sha256(
        payload.get("navigation_checkpoint_sha256"),
        "navigation_checkpoint_sha256",
    )
    runtime_binding_hash = _sha256(
        payload.get("gemma_runtime_binding_sha256"),
        "gemma_runtime_binding_sha256",
    )
    audit = memory.get("loaded_file_audit")
    if (
        memory.get("sha256") != scene_hash
        or memory.get("question_dependent_scene_retrieval") is not False
        or memory.get("environmental_text_inputs") != []
        or memory.get("all_runtime_voxels_encoded") is not True
        or not isinstance(audit, Mapping)
        or audit.get("forbidden_access_count") != 0
        or audit.get("passed") is not True
    ):
        raise RuntimeError("MCP rejected invalid continuous-scene provenance")
    decisions = [
        _decision_receipt(
            item,
            expected_step=index,
            scene_prefix_sha256=scene_hash,
            navigation_checkpoint_sha256=navigation_hash,
        )
        for index, item in enumerate(raw_decisions, start=1)
        if isinstance(item, Mapping)
    ]
    if len(decisions) != len(raw_decisions):
        raise TypeError("MCP received a malformed Gemma decision list")
    accepted_count = sum(decision.accepted for decision in decisions)
    model_stop_emitted = payload.get("model_stop_emitted")
    success = payload.get("success")
    if type(success) is not bool or type(model_stop_emitted) is not bool:
        raise TypeError("MCP result lacks boolean completion state")
    if success and (
        model_stop_emitted is not True
        or decisions[-1].model_action != "stop"
        or decisions[-1].accepted is not True
        or decisions[-1].executed is not True
    ):
        raise RuntimeError("MCP success lacks Gemma's accepted terminal STOP")
    error_code = payload.get("error_code")
    if error_code is not None and (
        not isinstance(error_code, str) or re.fullmatch(r"E_[A-Z0-9_]+", error_code) is None
    ):
        raise ValueError("MCP result error_code is invalid")
    scene_id = _scene_id(state.get("scene_id"))
    map_version = state.get("scene_version", payload.get("map_version"))
    if isinstance(map_version, bool) or not isinstance(map_version, int):
        raise TypeError("MCP result map version must be an integer")
    action_count = state.get("action_count")
    if isinstance(action_count, bool) or not isinstance(action_count, int):
        raise TypeError("MCP result action_count must be an integer")
    if type(state.get("collision")) is not bool or type(state.get("stopped")) is not bool:
        raise TypeError("MCP collision and stopped state must be boolean")
    integer_memory_fields = (
        "source_voxels",
        "processed_voxels",
        "semantic_feature_dim",
        "token_count",
        "model_dim",
        "robot_state_token_count",
    )
    for name in integer_memory_fields:
        value = memory.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"MCP scene-memory {name} must be positive")
    return GemmaGoalResponse.model_validate(
        {
            "schema": SCHEMA,
            "success": success,
            "error_code": error_code,
            "scene_id": scene_id,
            "goal_sha256": hashlib.sha256(submitted_goal.encode("utf-8")).hexdigest(),
            "scene_prefix_sha256": scene_hash,
            "active_prefix_sha256": active_hash,
            "navigation_checkpoint_sha256": navigation_hash,
            "gemma_runtime_binding_sha256": runtime_binding_hash,
            "position_m": _strict_numeric_vector(state.get("position_m"), "position_m", length=3),
            "body_yaw_degrees": _finite(state.get("body_yaw_degrees"), "body_yaw_degrees"),
            "collision": state.get("collision"),
            "stopped": state.get("stopped"),
            "action_count": action_count,
            "map_version": map_version,
            "source_voxels": memory["source_voxels"],
            "processed_voxels": memory["processed_voxels"],
            "semantic_feature_dim": memory["semantic_feature_dim"],
            "scene_token_count": memory["token_count"],
            "scene_model_dim": memory["model_dim"],
            "robot_state_token_count": memory["robot_state_token_count"],
            "model_decision_count": len(decisions),
            "accepted_decision_count": accepted_count,
            "rejected_decision_count": len(decisions) - accepted_count,
            "model_stop_emitted": model_stop_emitted,
            "local_inference": True,
            "cloud_model_used": False,
            "continuous_scene_memory": True,
            "continuous_robot_state": True,
            "model_selects_every_waypoint_and_heading": True,
            "model_selects_stop": True,
            "fallback_used": False,
            "substitution_applied": False,
            "synthetic_stop_applied": False,
            "deterministic_route_planner_used": False,
            "question_dependent_scene_retrieval": False,
            "runtime_forbidden_access_count": 0,
            "decisions": decisions,
        }
    )


def _validate_startup(payload: Mapping[str, Any]) -> tuple[str, str]:
    if (
        payload.get("ready") is not True
        or payload.get("control_mode") != "actual_local_gemma_model_only_waypoint_policy"
        or payload.get("high_level_natural_language_only") is not True
        or payload.get("model_selects_every_waypoint_and_heading") is not True
        or payload.get("model_selects_stop") is not True
        or payload.get("local_inference") is not True
        or payload.get("cloud_model_used") is not False
        or payload.get("fallback_used") is not False
        or payload.get("deterministic_route_planner_used") is not False
        or payload.get("synthetic_stop_applied") is not False
        or payload.get("substitution_applied") is not False
        or payload.get("environmental_text_inputs") != []
    ):
        raise RuntimeError("MCP requires the exclusive local-Gemma startup contract")
    return (
        _sha256(payload.get("scene_prefix_hash"), "startup scene_prefix_hash"),
        _sha256(
            payload.get("navigation_checkpoint_sha256"),
            "startup navigation_checkpoint_sha256",
        ),
    )


def build_gemma_goal_server(controller: PracticalRoverController) -> MCPServer[None]:
    """Build the one-tool MCP server over an already loaded rover controller."""

    startup = controller.startup()
    scene_hash, checkpoint_hash = _validate_startup(startup)
    server: MCPServer[None] = MCPServer(
        SERVER_NAME,
        version="0.1.0",
        description=("Outcome-level local-Gemma control over continuous 3D scene memory."),
        instructions=(
            "Submit one outcome-level natural-language goal. Gemma selects every exact "
            "FACE angle, MOVE_TO waypoint, recovery decision, and STOP. No direct motor "
            "tools, route planner, fallback, substitution, or synthetic STOP are exposed."
        ),
    )

    @server.tool(structured_output=True)
    def navigate(goal: GoalText) -> GemmaGoalResponse:
        """Run one high-level goal; local Gemma chooses the complete motion sequence."""

        normalized_goal = goal.strip()
        result = controller.navigate_goal(normalized_goal)
        response = goal_response(result, normalized_goal)
        if (
            response.scene_prefix_sha256 != scene_hash
            or response.navigation_checkpoint_sha256 != checkpoint_hash
        ):
            raise RuntimeError("MCP continuous scene or Gemma checkpoint identity changed")
        return response

    registered = server._tool_manager.get_tool(TOOL_NAME)
    if registered is None:  # pragma: no cover - registration programming error
        raise RuntimeError("Gemma navigation MCP tool was not registered")
    argument_model = registered.fn_metadata.arg_model
    argument_model.model_config["extra"] = "forbid"
    argument_model.model_rebuild(force=True)
    registered.parameters = {
        **registered.parameters,
        "additionalProperties": False,
    }
    return server


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--control-config", default=DEFAULT_CONTROL_CONFIG)
    parser.add_argument("--scene", default=DEFAULT_SCENE)
    parser.add_argument("--base-checkpoint", default=DEFAULT_BASE_CHECKPOINT)
    parser.add_argument("--control-checkpoint", default=DEFAULT_CONTROL_CHECKPOINT)
    parser.add_argument("--runtime-asset", default=DEFAULT_ASSET)
    parser.add_argument("--robot-state-checkpoint", default=DEFAULT_ROBOT_STATE_CHECKPOINT)
    parser.add_argument("--navigation-checkpoint", default=DEFAULT_NAVIGATION_CHECKPOINT)
    parser.add_argument("--audit-output")
    parser.add_argument("--transport", choices=("stdio", "streamable-http"), default="stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8771)
    parser.add_argument("--check", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    inputs = {
        "config": args.config,
        "control_config": args.control_config,
        "scene_id": args.scene,
        "base_checkpoint": args.base_checkpoint,
        "control_checkpoint": args.control_checkpoint,
        "runtime_asset": args.runtime_asset,
        "robot_state_checkpoint": args.robot_state_checkpoint,
        "navigation_checkpoint": args.navigation_checkpoint,
    }
    if args.check:
        preflight = practical_rover_preflight(**inputs)
        print(
            json.dumps(
                {
                    "schema": SCHEMA,
                    "passed": True,
                    "tool_names": [TOOL_NAME],
                    "tool_count": 1,
                    "direct_motor_tools_exposed": False,
                    "legacy_numeric_mcp_separate": True,
                    "preflight": preflight,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    controller = build_local_practical_rover(
        **inputs,
        audit_output=args.audit_output,
        enable_gemma_json_fallback=False,
        initial_scan=False,
    )
    try:
        server = build_gemma_goal_server(controller)
        if args.transport == "stdio":
            server.run("stdio")
        else:
            server.run("streamable-http", host=args.host, port=args.port)
    finally:
        controller.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "SCHEMA",
    "TOOL_NAME",
    "GemmaDecisionReceipt",
    "GemmaGoalResponse",
    "build_gemma_goal_server",
    "goal_response",
    "main",
]
