"""Official-MCP stdio action runtime for :mod:`robot.conversation`.

The production MCP subprocess owns rendering, semantic-map fusion, scene-token
refresh, and numeric robot-state encoding.  This module deliberately mirrors
only the resulting numeric/protocol receipt in the conversation process.  It
never parses the MCP text content and never exposes an environmental caption,
label, object ID, or scene graph to the agent.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import math
import os
import threading
from collections.abc import Mapping
from pathlib import Path
from types import TracebackType
from typing import Any, Protocol, Self

import numpy as np
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from pydantic import ValidationError

from semantic_3d_chat.mcp_server.server import ToolResponse
from semantic_3d_chat.robot.collision import NumericCollisionMap
from semantic_3d_chat.robot.simulator import RobotState
from semantic_3d_chat.robot.tools import TOOL_ARGUMENTS

_EXPECTED_TOOLS = frozenset(TOOL_ARGUMENTS)
_AUTO_SCAN_ACTIONS = frozenset(
    {"look", "turn", "move_forward", "move_backward", "move_to"}
)
_BINDING_FIELDS = (
    "schema",
    "scene_id",
    "map_version",
    "map_sha256",
    "scene_prefix_sha256",
    "scene_control_signature_sha256",
    "source_voxels",
    "processed_voxels",
    "binding_sha256",
    "active_prefix_sha256",
    "robot_state_sha256",
    "robot_tokens_sha256",
    "robot_state_encoder_sha256",
    "active_binding_sha256",
)
_KNOWN_ACTION_ERRORS = frozenset(
    {
        "E_COLLISION",
        "E_EMPTY_SCAN",
        "E_LIMIT",
        "E_MAP_RESET",
        "E_MAP_UPDATE",
        "E_NUMERIC",
        "E_PROTOCOL",
        "E_SCENE_ID",
        "E_SCENE_UNAVAILABLE",
        "E_SCHEMA",
        "E_STOPPED",
        "E_TOOL",
    }
)
_FORBIDDEN_PATH_COMPONENTS = frozenset(
    {"oracle", "qa", "training", "scorer_only", "scorer-only"}
)


class MCPActionTransportError(RuntimeError):
    """A fail-closed error at the MCP process or JSON-RPC boundary."""


class StructuredToolClient(Protocol):
    """Small synchronous seam implemented by the real stdio client and tests."""

    @property
    def tool_names(self) -> frozenset[str]: ...

    def call_tool(self, name: str, arguments: Mapping[str, Any]) -> dict[str, Any]: ...


class ConversationAnswerBackend(Protocol):
    """Optional local answer seam; action MCP deliberately has no text tool."""

    def answer(self, question: str) -> Any: ...


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(payload),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _safe_map_path(path: str | Path, *, purpose: str, must_exist: bool) -> Path:
    candidate = Path(path).expanduser()
    unresolved = Path(os.path.abspath(candidate))
    current = Path(unresolved.anchor)
    for component in unresolved.parts[1:]:
        current /= component
        if current.is_symlink():
            raise ValueError(f"{purpose} cannot use a symbolic-link path")
    if _FORBIDDEN_PATH_COMPONENTS.intersection(
        component.casefold() for component in unresolved.parts
    ):
        raise ValueError(f"{purpose} cannot be stored with environmental supervision")
    if must_exist and not unresolved.is_file():
        raise FileNotFoundError(f"{purpose} is unavailable: {unresolved}")
    return unresolved


def validate_numeric_tool_receipt(
    payload: Mapping[str, Any],
    *,
    require_continuous_binding: bool = False,
) -> dict[str, Any]:
    """Validate and normalize the structured result without reading text content.

    ``ToolResponse`` forbids unknown keys, non-finite numbers, non-opaque IDs,
    malformed hashes, and arbitrary strings.  The additional checks below pin
    the finite set of protocol errors and authenticate both nested prefix
    bindings, so a mixed or stale receipt fails before it reaches the agent.
    """

    if not isinstance(payload, Mapping):
        raise MCPActionTransportError("MCP structured output is not an object")
    try:
        validated = ToolResponse.model_validate(dict(payload))
    except ValidationError as error:
        raise MCPActionTransportError("MCP structured output failed its numeric schema") from error
    receipt = validated.model_dump(mode="json", by_alias=True)
    error_code = receipt.get("error_code")
    if error_code is not None and error_code not in _KNOWN_ACTION_ERRORS:
        raise MCPActionTransportError("MCP returned an unrecognized protocol error")
    if not require_continuous_binding:
        return receipt

    missing = [field for field in _BINDING_FIELDS if receipt.get(field) is None]
    if missing:
        raise MCPActionTransportError(
            f"MCP receipt omitted continuous-prefix binding fields: {missing}"
        )
    if receipt["schema"] != "semantic_3d_chat.scene_prefix_binding.v2":
        raise MCPActionTransportError("MCP receipt is not a v2 continuous-prefix binding")
    if receipt["scene_version"] != receipt["map_version"]:
        raise MCPActionTransportError("MCP scene and map versions diverged")
    if receipt["source_voxels"] < 1 or receipt["processed_voxels"] < 1:
        raise MCPActionTransportError("MCP continuous-prefix binding is empty")

    scene_identity = {
        field: receipt[field]
        for field in (
            "schema",
            "scene_id",
            "map_version",
            "map_sha256",
            "scene_prefix_sha256",
            "scene_control_signature_sha256",
            "source_voxels",
            "processed_voxels",
        )
    }
    if _canonical_sha256(scene_identity) != receipt["binding_sha256"]:
        raise MCPActionTransportError("MCP scene-prefix binding hash is invalid")
    active_identity = {
        **scene_identity,
        "binding_sha256": receipt["binding_sha256"],
        "active_prefix_sha256": receipt["active_prefix_sha256"],
        "robot_state_sha256": receipt["robot_state_sha256"],
        "robot_tokens_sha256": receipt["robot_tokens_sha256"],
        "robot_state_encoder_sha256": receipt["robot_state_encoder_sha256"],
    }
    if _canonical_sha256(active_identity) != receipt["active_binding_sha256"]:
        raise MCPActionTransportError("MCP active-prefix binding hash is invalid")
    return receipt


class MCPStdioToolClient:
    """Keep one official-SDK stdio session alive behind a synchronous seam."""

    def __init__(
        self,
        parameters: StdioServerParameters,
        *,
        read_timeout_seconds: float = 600.0,
        startup_timeout_seconds: float = 600.0,
        call_timeout_seconds: float = 600.0,
    ) -> None:
        if any(
            not math.isfinite(value) or value <= 0
            for value in (
                read_timeout_seconds,
                startup_timeout_seconds,
                call_timeout_seconds,
            )
        ):
            raise ValueError("MCP timeouts must be finite and positive")
        self.parameters = parameters
        self.read_timeout_seconds = float(read_timeout_seconds)
        self.startup_timeout_seconds = float(startup_timeout_seconds)
        self.call_timeout_seconds = float(call_timeout_seconds)
        self._ready = threading.Event()
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._shutdown: asyncio.Event | None = None
        self._session: ClientSession | None = None
        self._tool_names = frozenset[str]()
        self._error: BaseException | None = None
        self._call_lock = threading.Lock()

    @property
    def tool_names(self) -> frozenset[str]:
        if self._thread is None or not self._thread.is_alive():
            if self._error is not None:
                raise MCPActionTransportError("MCP stdio session is unavailable") from self._error
            raise MCPActionTransportError("MCP stdio session has not been started")
        return self._tool_names

    async def _serve(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._shutdown = asyncio.Event()
        async with stdio_client(self.parameters) as (read_stream, write_stream):  # noqa: SIM117
            async with ClientSession(
                read_stream,
                write_stream,
                read_timeout_seconds=self.read_timeout_seconds,
            ) as session:
                await session.initialize()
                listed = await session.list_tools()
                names = frozenset(tool.name for tool in listed.tools)
                if names != _EXPECTED_TOOLS:
                    raise MCPActionTransportError(
                        "MCP tool inventory differs from the nine bounded robot tools"
                    )
                self._session = session
                self._tool_names = names
                self._ready.set()
                await self._shutdown.wait()

    def _thread_main(self) -> None:
        try:
            asyncio.run(self._serve())
        except BaseException as error:  # noqa: BLE001 - propagate worker failures safely
            self._error = error
        finally:
            self._session = None
            self._ready.set()

    def start(self) -> Self:
        if self._thread is not None:
            raise RuntimeError("MCP stdio client cannot be started twice")
        self._thread = threading.Thread(
            target=self._thread_main,
            name="semantic-3d-mcp-stdio",
            daemon=True,
        )
        self._thread.start()
        if not self._ready.wait(self.startup_timeout_seconds):
            self.close()
            raise MCPActionTransportError("MCP stdio server startup timed out")
        if self._error is not None or self._session is None:
            raise MCPActionTransportError("MCP stdio server startup failed") from self._error
        return self

    async def _call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if self._session is None:
            raise MCPActionTransportError("MCP stdio session is closed")
        result = await self._session.call_tool(name, arguments)
        if result.is_error:
            # Do not surface free-form MCP text content through the action seam.
            raise MCPActionTransportError(f"MCP tool transport failed: {name}")
        if result.structured_content is None:
            raise MCPActionTransportError(f"MCP tool omitted structured output: {name}")
        return validate_numeric_tool_receipt(result.structured_content)

    def call_tool(self, name: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
        if name not in TOOL_ARGUMENTS:
            raise ValueError("Unknown MCP robot tool")
        if not isinstance(arguments, Mapping) or frozenset(arguments) != TOOL_ARGUMENTS[name]:
            raise ValueError("MCP robot tool arguments do not match the strict schema")
        loop = self._loop
        if loop is None or self._session is None or self._error is not None:
            raise MCPActionTransportError("MCP stdio session is unavailable") from self._error
        with self._call_lock:
            future = asyncio.run_coroutine_threadsafe(
                self._call(name, dict(arguments)),
                loop,
            )
            try:
                return future.result(timeout=self.call_timeout_seconds)
            except TimeoutError as error:
                future.cancel()
                raise MCPActionTransportError(f"MCP tool timed out: {name}") from error

    def close(self) -> None:
        thread = self._thread
        loop = self._loop
        shutdown = self._shutdown
        if thread is None:
            return
        if loop is not None and shutdown is not None and thread.is_alive():
            loop.call_soon_threadsafe(shutdown.set)
        thread.join(timeout=min(self.call_timeout_seconds, 60.0))
        if thread.is_alive():
            raise MCPActionTransportError("MCP stdio server did not terminate")
        self._thread = None

    def __enter__(self) -> Self:
        return self.start()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


class _RemoteMapUpdaterView:
    def __init__(self, base_map_path: Path, persistent_map_path: Path) -> None:
        self.base_map_path = base_map_path
        self.persistent_map_path = persistent_map_path


def _normalize_degrees(value: float) -> float:
    return (float(value) + 180.0) % 360.0 - 180.0


class _RemoteSimulatorView:
    """Numeric state/collision view required by the existing semantic planner."""

    def __init__(self, config: Mapping[str, Any], receipt: Mapping[str, Any]) -> None:
        robot = config.get("robot")
        if not isinstance(robot, Mapping):
            raise TypeError("MCP conversation runtime requires robot configuration")
        self.config = dict(config)
        self.settings = dict(robot)
        self.collision_map: NumericCollisionMap
        self._receipt: dict[str, Any] = {}
        self.state = RobotState(
            scene_id=str(receipt["scene_id"]),
            seed=int(receipt["seed"]),
            position_xy_m=np.asarray(receipt["position_m"][:2], dtype=np.float64),
        )
        self.update(receipt)

    def update(self, receipt: Mapping[str, Any]) -> None:
        body_yaw = float(receipt["body_yaw_degrees"])
        camera_yaw = float(receipt["camera_yaw_degrees"])
        self.state = RobotState(
            scene_id=str(receipt["scene_id"]),
            seed=int(receipt["seed"]),
            position_xy_m=np.asarray(receipt["position_m"][:2], dtype=np.float64),
            body_yaw_degrees=body_yaw,
            camera_yaw_offset_degrees=_normalize_degrees(camera_yaw - body_yaw),
            pitch_degrees=float(receipt["pitch_degrees"]),
            linear_velocity_xy_m=np.asarray(
                receipt["linear_velocity_xy_m"], dtype=np.float64
            ),
            angular_velocity_degrees=float(receipt["angular_velocity_degrees"]),
            collision=bool(receipt["collision"]),
            last_movement_delta_m=np.asarray(
                receipt["last_movement_delta_m"], dtype=np.float64
            ),
            scan_coverage=float(receipt["scan_coverage"]),
            scan_count=int(receipt["scan_count"]),
            scene_version=int(receipt["scene_version"]),
            map_sha256=str(receipt["map_sha256"]),
            action_count=int(receipt["action_count"]),
            stopped=bool(receipt["stopped"]),
        )
        self._receipt = copy.deepcopy(dict(receipt))

    def get_robot_state(self) -> dict[str, Any]:
        return copy.deepcopy(self._receipt)


class MCPConversationRuntime:
    """Drop-in action runtime for :class:`ConversationalEmbodiedAgent`.

    The server remains authoritative.  The local simulator view exists only so
    the established label-free grounding/planning code can read the latest
    numeric pose and anonymous collision geometry.  Every call replaces the
    cached state and continuous-prefix binding with the authenticated receipt.
    """

    def __init__(
        self,
        tool_client: StructuredToolClient,
        config: Mapping[str, Any],
        *,
        base_map_path: str | Path,
        persistent_map_path: str | Path,
        answer_backend: ConversationAnswerBackend | None = None,
        owned_client: MCPStdioToolClient | None = None,
    ) -> None:
        if tool_client.tool_names != _EXPECTED_TOOLS:
            raise MCPActionTransportError("MCP client does not expose the bounded tool set")
        self.config = copy.deepcopy(dict(config))
        self.map_updater = _RemoteMapUpdaterView(
            _safe_map_path(
                base_map_path,
                purpose="sanitized base map",
                must_exist=True,
            ),
            _safe_map_path(
                persistent_map_path,
                purpose="sanitized persistent map",
                must_exist=False,
            ),
        )
        self._tool_client = tool_client
        self._owned_client = owned_client
        self._answer_backend = answer_backend
        self._binding: dict[str, Any] = {}
        self._binding_refresh_count = 0
        initial = validate_numeric_tool_receipt(
            tool_client.call_tool("get_robot_state", {}),
            require_continuous_binding=True,
        )
        self.simulator = _RemoteSimulatorView(self.config, initial)
        self._accept_receipt("get_robot_state", initial, previous=None)
        self._refresh_collision_map()

    @classmethod
    def connect_stdio(
        cls,
        parameters: StdioServerParameters,
        config: Mapping[str, Any],
        *,
        base_map_path: str | Path,
        persistent_map_path: str | Path,
        answer_backend: ConversationAnswerBackend | None = None,
        read_timeout_seconds: float = 600.0,
        startup_timeout_seconds: float = 600.0,
        call_timeout_seconds: float = 600.0,
    ) -> MCPConversationRuntime:
        client = MCPStdioToolClient(
            parameters,
            read_timeout_seconds=read_timeout_seconds,
            startup_timeout_seconds=startup_timeout_seconds,
            call_timeout_seconds=call_timeout_seconds,
        ).start()
        try:
            return cls(
                client,
                config,
                base_map_path=base_map_path,
                persistent_map_path=persistent_map_path,
                answer_backend=answer_backend,
                owned_client=client,
            )
        except BaseException:
            client.close()
            raise

    @property
    def binding_refresh_count(self) -> int:
        return self._binding_refresh_count

    def _active_map_path(self) -> Path:
        persistent = self.map_updater.persistent_map_path
        return persistent if persistent.is_file() else self.map_updater.base_map_path

    def _refresh_collision_map(self) -> None:
        settings = self.simulator.settings
        self.simulator.collision_map = NumericCollisionMap.from_voxel_map(
            self._active_map_path(),
            room_size_m=self.config["scene"]["room_size_m"],
            robot_radius_m=float(settings["radius_m"]),
            collision_z_min_m=float(settings.get("collision_z_min_m", 0.12)),
            collision_z_max_m=float(settings.get("collision_z_max_m", 1.80)),
            surface_padding_m=float(settings.get("surface_padding_m", 0.035)),
        )

    def _accept_receipt(
        self,
        tool_name: str,
        payload: Mapping[str, Any],
        *,
        previous: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        receipt = validate_numeric_tool_receipt(
            payload,
            require_continuous_binding=True,
        )
        if previous is not None:
            if receipt["scene_id"] != previous["scene_id"] and tool_name != "reset_scene":
                raise MCPActionTransportError("MCP action changed the opaque scene unexpectedly")
            prior_version = int(previous["map_version"])
            current_version = int(receipt["map_version"])
            if tool_name != "reset_scene" and current_version < prior_version:
                raise MCPActionTransportError("MCP map version moved backward")
            auto_scan = bool(self.simulator.settings.get("auto_scan_after_motion", False))
            should_commit = tool_name == "scan" or (
                auto_scan and tool_name in _AUTO_SCAN_ACTIONS
            )
            if receipt["success"] and should_commit and current_version != prior_version + 1:
                raise MCPActionTransportError(
                    "Successful observation action did not refresh the scene prefix"
                )
        prior_map_version = None if previous is None else int(previous["map_version"])
        self._binding = {field: receipt[field] for field in _BINDING_FIELDS}
        self._binding_refresh_count += 1
        self.simulator.update(receipt)
        if prior_map_version is not None and int(receipt["map_version"]) != prior_map_version:
            self._refresh_collision_map()
        return receipt

    def _call(self, name: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
        previous = copy.deepcopy(self._binding)
        payload = self._tool_client.call_tool(name, arguments)
        return self._accept_receipt(name, payload, previous=previous)

    def prefix_binding(self) -> dict[str, Any]:
        return copy.deepcopy(self._binding)

    def get_robot_state(self) -> dict[str, Any]:
        return self._call("get_robot_state", {})

    def look(self, yaw_delta_degrees: float, pitch_delta_degrees: float) -> dict[str, Any]:
        return self._call(
            "look",
            {
                "yaw_delta_degrees": yaw_delta_degrees,
                "pitch_delta_degrees": pitch_delta_degrees,
            },
        )

    def turn(self, angle_degrees: float) -> dict[str, Any]:
        return self._call("turn", {"angle_degrees": angle_degrees})

    def move_forward(self, distance_meters: float) -> dict[str, Any]:
        return self._call("move_forward", {"distance_meters": distance_meters})

    def move_backward(self, distance_meters: float) -> dict[str, Any]:
        return self._call("move_backward", {"distance_meters": distance_meters})

    def move_to(self, x: float, y: float) -> dict[str, Any]:
        return self._call("move_to", {"x": x, "y": y})

    def scan(self) -> dict[str, Any]:
        return self._call("scan", {})

    def stop(self) -> dict[str, Any]:
        return self._call("stop", {})

    def reset_scene(self, scene_id: str, seed: int) -> dict[str, Any]:
        return self._call("reset_scene", {"scene_id": scene_id, "seed": seed})

    def answer(self, question: str) -> Any:
        if self._answer_backend is None:
            raise RuntimeError(
                "This MCP runtime exposes numeric robot actions only; configure a local "
                "continuous-prefix answer backend for non-navigation turns"
            )
        return self._answer_backend.answer(question)

    def close(self) -> None:
        if self._owned_client is not None:
            client, self._owned_client = self._owned_client, None
            client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


__all__ = [
    "MCPActionTransportError",
    "MCPConversationRuntime",
    "MCPStdioToolClient",
    "StructuredToolClient",
    "validate_numeric_tool_receipt",
]
