"""Strict JSON tool protocol for the direct embodied-camera precursor."""

from __future__ import annotations

from typing import Any

from semantic_3d_chat.robot.simulator import EmbodiedCameraSimulator

TOOL_ARGUMENTS: dict[str, frozenset[str]] = {
    "get_robot_state": frozenset(),
    "look": frozenset({"yaw_delta_degrees", "pitch_delta_degrees"}),
    "turn": frozenset({"angle_degrees"}),
    "move_forward": frozenset({"distance_meters"}),
    "move_backward": frozenset({"distance_meters"}),
    "move_to": frozenset({"x", "y"}),
    "scan": frozenset(),
    "stop": frozenset(),
    "reset_scene": frozenset({"scene_id", "seed"}),
}


def tool_schemas(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Return constrained JSON schemas for local model tool generation."""

    robot = config["robot"]
    max_turn = float(robot["max_turn_degrees"])
    max_look = float(robot.get("max_look_delta_degrees", 45.0))
    max_move = float(robot["max_move_m"])
    result: list[dict[str, Any]] = []
    definitions: dict[str, dict[str, Any]] = {
        "get_robot_state": {},
        "look": {
            "yaw_delta_degrees": {"type": "number", "minimum": -max_look, "maximum": max_look},
            "pitch_delta_degrees": {
                "type": "number",
                "minimum": -max_look,
                "maximum": max_look,
            },
        },
        "turn": {
            "angle_degrees": {"type": "number", "minimum": -max_turn, "maximum": max_turn}
        },
        "move_forward": {
            "distance_meters": {"type": "number", "minimum": 0.0, "maximum": max_move}
        },
        "move_backward": {
            "distance_meters": {"type": "number", "minimum": 0.0, "maximum": max_move}
        },
        "move_to": {
            "x": {"type": "number"},
            "y": {"type": "number"},
        },
        "scan": {},
        "stop": {},
        "reset_scene": {
            "scene_id": {"type": "string", "pattern": r"^scene_[0-9]{6}$"},
            "seed": {"type": "integer", "minimum": 0, "maximum": 2**32 - 1},
        },
    }
    for name, properties in definitions.items():
        result.append(
            {
                "name": name,
                "inputSchema": {
                    "type": "object",
                    "properties": properties,
                    "required": sorted(properties),
                    "additionalProperties": False,
                },
            }
        )
    return result


class RobotToolController:
    def __init__(self, simulator: EmbodiedCameraSimulator) -> None:
        self.simulator = simulator

    def call(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        if name not in TOOL_ARGUMENTS:
            return self.simulator.protocol_error("E_TOOL")
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, dict) or frozenset(arguments) != TOOL_ARGUMENTS[name]:
            return self.simulator.protocol_error("E_SCHEMA")
        if name == "get_robot_state":
            return self.simulator.get_robot_state()
        if name == "look":
            return self.simulator.look(
                arguments["yaw_delta_degrees"], arguments["pitch_delta_degrees"]
            )
        if name == "turn":
            return self.simulator.turn(arguments["angle_degrees"])
        if name == "move_forward":
            return self.simulator.move_forward(arguments["distance_meters"])
        if name == "move_backward":
            return self.simulator.move_backward(arguments["distance_meters"])
        if name == "move_to":
            return self.simulator.move_to(arguments["x"], arguments["y"])
        if name == "scan":
            return self.simulator.scan()
        if name == "stop":
            return self.simulator.stop()
        return self.simulator.reset_scene(str(arguments["scene_id"]), arguments["seed"])

    def dispatch(self, payload: Any) -> dict[str, Any]:
        """Validate the exact constrained JSON envelope before executing it."""

        if not isinstance(payload, dict) or set(payload) != {"tool", "arguments"}:
            return self.simulator.protocol_error("E_PROTOCOL")
        if not isinstance(payload["tool"], str):
            return self.simulator.protocol_error("E_PROTOCOL")
        return self.call(payload["tool"], payload["arguments"])
