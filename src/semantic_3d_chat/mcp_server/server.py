"""Official MCP Python SDK wrapper around the tested numerical robot actions."""

from __future__ import annotations

import argparse
from typing import Any

from mcp.server.mcpserver import MCPServer
from pydantic import BaseModel, ConfigDict

from semantic_3d_chat.config import load_config
from semantic_3d_chat.robot.simulator import EmbodiedCameraSimulator
from semantic_3d_chat.robot.tools import tool_schemas


class ToolResponse(BaseModel):
    """Common structured output containing protocol and numerical state only."""

    model_config = ConfigDict(extra="forbid")

    success: bool
    error_code: str | None
    scene_id: str
    seed: int
    scene_version: int
    position_m: list[float]
    camera_position_m: list[float]
    body_yaw_degrees: float
    camera_yaw_degrees: float
    pitch_degrees: float
    linear_velocity_xy_m: list[float]
    angular_velocity_degrees: float
    collision: bool
    last_movement_delta_m: list[float]
    distance_moved: float
    turn_degrees: float
    scan_coverage: float
    scan_count: int
    visible_voxels: int
    observation_id: str | None
    clearance_m: float | None
    action_count: int
    stopped: bool


def _response(payload: dict[str, Any]) -> ToolResponse:
    return ToolResponse.model_validate(payload)


def _harden_input_schemas(server: MCPServer[None], simulator: EmbodiedCameraSimulator) -> None:
    """Apply the direct protocol's strict, configured schemas to MCP tools.

    MCP SDK 2.0 derives function-argument models with Pydantic's default
    ``extra='ignore'`` policy. That would advertise no action bounds and would
    silently discard unexpected fields. The project pins this SDK version, so
    we explicitly forbid extras on each generated argument model and expose the
    same configured limits used by the already-tested direct protocol. Runtime
    methods still validate every value independently before changing state.
    """

    schemas = {item["name"]: item["inputSchema"] for item in tool_schemas(simulator.config)}
    for name, input_schema in schemas.items():
        registered = server._tool_manager.get_tool(name)
        if registered is None:  # pragma: no cover - programming error during registration
            raise RuntimeError(f"MCP tool was not registered: {name}")
        argument_model = registered.fn_metadata.arg_model
        argument_model.model_config["extra"] = "forbid"
        argument_model.model_rebuild(force=True)
        registered.parameters = {
            **input_schema,
            "title": f"{name}Arguments",
        }


def build_server(simulator: EmbodiedCameraSimulator) -> MCPServer[None]:
    """Build an in-process server so schemas and actions can be tested directly."""

    server: MCPServer[None] = MCPServer(
        "semantic-3d-robot",
        version="0.1.0",
        description="Bounded numerical embodied-camera actions over continuous scene memory.",
        instructions=(
            "Tool results contain only protocol status, opaque identifiers, numerical pose, "
            "collision state, scan coverage, and scene version."
        ),
    )

    @server.tool(structured_output=True)
    def get_robot_state() -> ToolResponse:
        """Return the current numerical robot and camera state."""

        return _response(simulator.get_robot_state())

    @server.tool(structured_output=True)
    def look(yaw_delta_degrees: float, pitch_delta_degrees: float) -> ToolResponse:
        """Rotate the camera within the configured per-call and total limits."""

        return _response(simulator.look(yaw_delta_degrees, pitch_delta_degrees))

    @server.tool(structured_output=True)
    def turn(angle_degrees: float) -> ToolResponse:
        """Rotate the robot body by one bounded angle."""

        return _response(simulator.turn(angle_degrees))

    @server.tool(structured_output=True)
    def move_forward(distance_meters: float) -> ToolResponse:
        """Attempt one bounded forward translation with swept collision checking."""

        return _response(simulator.move_forward(distance_meters))

    @server.tool(structured_output=True)
    def move_backward(distance_meters: float) -> ToolResponse:
        """Attempt one bounded backward translation with swept collision checking."""

        return _response(simulator.move_backward(distance_meters))

    @server.tool(structured_output=True)
    def move_to(x: float, y: float) -> ToolResponse:
        """Attempt one bounded straight-line movement to a numerical world coordinate."""

        return _response(simulator.move_to(x, y))

    @server.tool(structured_output=True)
    def scan() -> ToolResponse:
        """Capture numerical RGB-D, update observation counts, and advance scene version."""

        return _response(simulator.scan())

    @server.tool(structured_output=True)
    def stop() -> ToolResponse:
        """Stop the current episode; reset_scene starts a new one."""

        return _response(simulator.stop())

    @server.tool(structured_output=True)
    def reset_scene(scene_id: str, seed: int) -> ToolResponse:
        """Reset to an opaque scene ID and deterministic numerical start state."""

        return _response(simulator.reset_scene(scene_id, seed))

    _harden_input_schemas(server, simulator)
    return server


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", default="configs/default.yaml")
    result.add_argument("--scene", default="scene_000001")
    result.add_argument("--seed", type=int)
    result.add_argument("--transport", choices=("stdio", "streamable-http"), default="stdio")
    result.add_argument("--host", default="127.0.0.1")
    result.add_argument("--port", type=int, default=8766)
    return result


def main() -> None:
    args = parser().parse_args()
    config = load_config(args.config)
    simulator = EmbodiedCameraSimulator(config, args.scene, seed=args.seed)
    server = build_server(simulator)
    if args.transport == "stdio":
        server.run("stdio")
    else:
        server.run("streamable-http", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
