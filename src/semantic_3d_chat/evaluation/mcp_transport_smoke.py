"""Exercise the robot MCP server through the official SDK's stdio transport."""

from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from semantic_3d_chat.config import PROJECT_ROOT

EXPECTED_TOOLS = frozenset(
    {
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
)

_FORBIDDEN_RESULT_KEYS = frozenset(
    {
        "category",
        "categories",
        "caption",
        "captions",
        "label",
        "labels",
        "object",
        "objects",
        "object_id",
        "object_name",
        "relationship",
        "relationships",
        "scene_description",
        "scene_graph",
    }
)
_FORBIDDEN_SEMANTIC_WORDS = (
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


def _tool_content_payload(result: Any) -> list[dict[str, Any]]:
    return [item.model_dump(mode="json", by_alias=True) for item in result.content]


def _semantic_leaks(payloads: list[Any]) -> list[str]:
    """Return exact forbidden keys or semantic label words found in tool results."""

    leaks: set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                normalized = str(key).lower()
                if normalized in _FORBIDDEN_RESULT_KEYS:
                    leaks.add(f"key:{normalized}")
                visit(child)
            return
        if isinstance(value, list):
            for child in value:
                visit(child)
            return
        if isinstance(value, str):
            lowered = value.lower()
            for word in _FORBIDDEN_SEMANTIC_WORDS:
                if re.search(rf"\b{re.escape(word)}\b", lowered):
                    leaks.add(f"word:{word}")

    for payload in payloads:
        visit(payload)
    return sorted(leaks)


async def run_stdio_transport_smoke(
    config: str | Path,
    scene_id: str,
    *,
    python_executable: str | Path | None = None,
) -> dict[str, Any]:
    """Spawn the production server module and validate its JSON-RPC boundary."""

    executable = Path(sys.executable if python_executable is None else python_executable)
    if not executable.is_absolute():
        executable = (Path.cwd() / executable).absolute()
    config_path = Path(config)
    if not config_path.is_absolute():
        config_path = (PROJECT_ROOT / config_path).absolute()

    child_environment = dict(os.environ)
    source_root = str(PROJECT_ROOT / "src")
    inherited_pythonpath = child_environment.get("PYTHONPATH")
    child_environment["PYTHONPATH"] = (
        source_root
        if not inherited_pythonpath
        else os.pathsep.join((source_root, inherited_pythonpath))
    )
    parameters = StdioServerParameters(
        # Do not resolve a virtual-environment symlink: doing so bypasses that
        # environment's site-packages when Python starts the child process.
        command=str(executable),
        args=[
            "-m",
            "semantic_3d_chat.mcp_server.server",
            "--config",
            str(config_path),
            "--scene",
            scene_id,
            "--transport",
            "stdio",
        ],
        cwd=PROJECT_ROOT,
        env=child_environment,
    )

    # Keep the transport and session scopes explicit: transport teardown owns
    # the subprocess, while session teardown owns the negotiated MCP channel.
    async with stdio_client(parameters) as (read_stream, write_stream):  # noqa: SIM117
        async with ClientSession(read_stream, write_stream) as session:
            initialized = await session.initialize()
            listed = await session.list_tools()
            tools = {tool.name: tool for tool in listed.tools}
            if set(tools) != EXPECTED_TOOLS:
                raise AssertionError(
                    f"MCP tool inventory mismatch: {sorted(tools)} != {sorted(EXPECTED_TOOLS)}"
                )

            turn_schema = tools["turn"].input_schema
            angle_schema = turn_schema["properties"]["angle_degrees"]
            maximum = float(angle_schema["maximum"])
            if turn_schema.get("additionalProperties") is not False:
                raise AssertionError("turn must reject additional properties")
            if maximum <= 0.0:
                raise AssertionError("turn must advertise a positive configured bound")

            state = await session.call_tool("get_robot_state", {})
            if state.is_error or state.structured_content is None:
                raise AssertionError("get_robot_state failed across stdio")
            initial_yaw = float(state.structured_content["body_yaw_degrees"])

            requested_turn = min(15.0, maximum)
            turned = await session.call_tool("turn", {"angle_degrees": requested_turn})
            if turned.is_error or turned.structured_content is None:
                raise AssertionError("bounded turn failed across stdio")
            expected_yaw = (initial_yaw + requested_turn + 180.0) % 360.0 - 180.0
            actual_yaw = float(turned.structured_content["body_yaw_degrees"])
            if abs(actual_yaw - expected_yaw) > 1e-6:
                raise AssertionError(f"turn produced yaw {actual_yaw}, expected {expected_yaw}")

            extra = await session.call_tool(
                "turn", {"angle_degrees": 1.0, "unexpected": 1}
            )
            if not extra.is_error:
                raise AssertionError("MCP server silently accepted an extra argument")

            after_extra = await session.call_tool("get_robot_state", {})
            if after_extra.structured_content is None:
                raise AssertionError("state unavailable after rejected extra argument")
            if float(after_extra.structured_content["body_yaw_degrees"]) != actual_yaw:
                raise AssertionError("rejected extra argument changed robot state")

            out_of_bounds = await session.call_tool(
                "turn", {"angle_degrees": maximum + 1.0}
            )
            if out_of_bounds.structured_content is None:
                raise AssertionError("bounded action rejection lacked structured output")
            if out_of_bounds.structured_content.get("success") is not False:
                raise AssertionError("out-of-bounds turn was not rejected")
            if out_of_bounds.structured_content.get("error_code") != "E_LIMIT":
                raise AssertionError("out-of-bounds turn did not return E_LIMIT")
            if float(out_of_bounds.structured_content["body_yaw_degrees"]) != actual_yaw:
                raise AssertionError("out-of-bounds turn changed robot state")

            reset_seed = 23
            reset = await session.call_tool(
                "reset_scene", {"scene_id": scene_id, "seed": reset_seed}
            )
            if reset.is_error or reset.structured_content is None:
                raise AssertionError("same-scene reset failed across stdio")
            if (
                reset.structured_content.get("success") is not True
                or reset.structured_content.get("seed") != reset_seed
                or reset.structured_content.get("scene_version") != 0
                or reset.structured_content.get("scan_count") != 0
                or float(reset.structured_content["body_yaw_degrees"]) != 0.0
            ):
                raise AssertionError("same-scene reset did not restore a clean episode")

            checked_payloads = [
                state.structured_content,
                _tool_content_payload(state),
                turned.structured_content,
                _tool_content_payload(turned),
                _tool_content_payload(extra),
                after_extra.structured_content,
                out_of_bounds.structured_content,
                _tool_content_payload(out_of_bounds),
                reset.structured_content,
                _tool_content_payload(reset),
            ]
            leaks = _semantic_leaks(checked_payloads)
            if leaks:
                raise AssertionError(f"semantic data leaked through MCP tool results: {leaks}")

            return {
                "schema": "semantic_3d_chat.mcp_stdio_transport_smoke.v1",
                "passed": True,
                "transport": "stdio",
                "mcp_sdk_version": importlib.metadata.version("mcp"),
                "protocol_version": str(initialized.protocol_version),
                "server_name": initialized.server_info.name,
                "server_version": initialized.server_info.version,
                "scene_id": scene_id,
                "tool_count": len(tools),
                "tools": sorted(tools),
                "additional_properties_forbidden": True,
                "bounded_turn_degrees": requested_turn,
                "resulting_body_yaw_degrees": actual_yaw,
                "extra_argument_rejected": bool(extra.is_error),
                "out_of_bounds_rejected": True,
                "out_of_bounds_error_code": "E_LIMIT",
                "state_unchanged_after_rejections": True,
                "same_scene_reset_passed": True,
                "reset_seed": reset_seed,
                "reset_scene_version": 0,
                "semantic_result_leaks": leaks,
            }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", default="configs/default.yaml")
    result.add_argument("--scene", default="scene_000001")
    result.add_argument("--output", type=Path)
    return result


def main() -> None:
    args = parser().parse_args()
    report = asyncio.run(run_stdio_transport_smoke(args.config, args.scene))
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
