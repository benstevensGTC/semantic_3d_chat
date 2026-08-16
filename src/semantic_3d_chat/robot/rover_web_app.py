"""Offline browser console for a locally controlled semantic-map rover.

The UI is intentionally a thin operator surface.  A caller supplies a
``RoverSession`` that owns Gemma, the continuous scene memory, simulation, and
MCP/direct action execution.  This module only serializes bounded user input
and renders a curated numeric state; it never imports datasets, evaluators, or
simulator metadata.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import re
import sys
from collections.abc import Mapping, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlsplit

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, HTMLResponse, JSONResponse, Response
from starlette.routing import Route

from semantic_3d_chat.chat.web_app import validate_visual_assets

PROJECT_ROOT = Path(__file__).resolve().parents[3]

_OPAQUE_SCENE_ID = re.compile(r"scene_[0-9]{6}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_MAX_REQUEST_BYTES = 16 * 1024
_MAX_INSTRUCTION_CHARACTERS = 2_048
_MAX_REPLY_CHARACTERS = 12_000
_MAX_ACTIONS_PER_TURN = 128
_MAX_MODEL_DECISIONS_PER_TURN = 128
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
_SCENE_MEMORY_SCHEMA = "semantic_3d_chat.scene_memory_diagnostics.v1"
_MODEL_ACTION_ORDER = ("move_to", "face", "stop")
_MODEL_ACTIONS = frozenset(_MODEL_ACTION_ORDER)
_PRIMITIVE_TOOLS = frozenset({"move_to", "turn", "stop"})
_MODEL_ONLY_NAVIGATION_MODE = "actual_local_gemma_model_only_waypoint_policy"
_PROTOCOL_CODE = re.compile(r"[A-Z][A-Z0-9_]{0,63}")
class RoverSession(Protocol):
    """Synchronous seam implemented by either MCP or a direct local runtime."""

    def startup(self) -> Mapping[str, Any]: ...

    def handle_instruction(self, text: str) -> Mapping[str, Any]: ...

    def close(self) -> None: ...


def _finite(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _integer(value: Any, *, name: str, minimum: int = 0) -> int:
    result = _finite(value, name=name)
    if not result.is_integer() or result < minimum:
        raise ValueError(f"{name} must be an integer greater than or equal to {minimum}")
    return int(result)


def _pair(value: Any, *, name: str) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        raise ValueError(f"{name} must contain two finite numbers")
    return [_finite(item, name=f"{name}[{index}]") for index, item in enumerate(value)]


def _finite_vector(value: Any, *, name: str, size: int) -> list[float]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != size
    ):
        raise ValueError(f"{name} must contain {size} finite numbers")
    return [_finite(item, name=f"{name}[{index}]") for index, item in enumerate(value)]


def _room_size(value: Sequence[float]) -> list[float]:
    if len(value) != 3:
        raise ValueError("room_size_m must contain three finite positive numbers")
    result = [_finite(item, name=f"room_size_m[{index}]") for index, item in enumerate(value)]
    if any(item <= 0.0 for item in result):
        raise ValueError("room_size_m must contain three finite positive numbers")
    return result


def _state_source(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    candidate = payload.get("state")
    return candidate if isinstance(candidate, Mapping) else payload


def _first(mapping: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in mapping:
            return mapping[name]
    return default


def _public_state(payload: Mapping[str, Any], room_size_m: Sequence[float]) -> dict[str, Any]:
    """Project a backend result onto the numeric fields the browser is allowed to see."""

    source = _state_source(payload)
    scene_id = str(_first(source, "scene_id", default=payload.get("scene_id", "")))
    if not _OPAQUE_SCENE_ID.fullmatch(scene_id):
        raise ValueError("Rover state requires an opaque scene_id")
    position = _first(source, "position_xy_m", "position_m", default=(0.0, 0.0))
    if isinstance(position, Sequence) and not isinstance(position, (str, bytes)) and len(position) == 3:
        position = position[:2]
    position_xy = _pair(position, name="position_xy_m")
    room = _room_size(room_size_m)
    if abs(position_xy[0]) > room[0] / 2.0 + 1e-6 or abs(position_xy[1]) > room[1] / 2.0 + 1e-6:
        raise ValueError("Rover position lies outside room bounds")

    prefix_hash = str(
        _first(
            source,
            "scene_prefix_hash",
            "scene_prefix_sha256",
            "prefix_hash",
            default=_first(
                payload,
                "scene_prefix_hash",
                "scene_prefix_sha256",
                "prefix_hash",
                default="",
            ),
        )
    ).casefold()
    if not _SHA256.fullmatch(prefix_hash):
        raise ValueError("Rover state requires a SHA-256 scene-prefix binding")

    coverage = _finite(_first(source, "scan_coverage", default=0.0), name="scan_coverage")
    if not 0.0 <= coverage <= 1.0:
        raise ValueError("scan_coverage must be in [0, 1]")
    return {
        "scene_id": scene_id,
        "position_xy_m": position_xy,
        "body_yaw_degrees": _finite(
            _first(source, "body_yaw_degrees", "yaw_degrees", default=0.0),
            name="body_yaw_degrees",
        ),
        "camera_yaw_degrees": _finite(
            _first(
                source,
                "camera_yaw_degrees",
                default=_first(source, "body_yaw_degrees", "yaw_degrees", default=0.0),
            ),
            name="camera_yaw_degrees",
        ),
        "pitch_degrees": _finite(
            _first(source, "pitch_degrees", "camera_pitch_degrees", default=0.0),
            name="pitch_degrees",
        ),
        "collision": bool(_first(source, "collision", "collision_flag", default=False)),
        "stopped": bool(_first(source, "stopped", default=False)),
        "scan_coverage": coverage,
        "scan_count": _integer(_first(source, "scan_count", default=0), name="scan_count"),
        "scene_version": _integer(
            _first(source, "scene_version", default=0), name="scene_version"
        ),
        "map_version": _integer(
            _first(source, "map_version", default=payload.get("map_version", 0)),
            name="map_version",
        ),
        "action_count": _integer(
            _first(source, "action_count", default=0), name="action_count"
        ),
        "scene_prefix_hash": prefix_hash,
        "room_size_m": room,
    }


def _optional_protocol_code(value: Any, *, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or _PROTOCOL_CODE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a bounded protocol code")
    return value


def _required_sha256(value: Any, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a SHA-256 digest")
    result = value.casefold()
    if _SHA256.fullmatch(result) is None:
        raise ValueError(f"{name} must be a SHA-256 digest")
    return result


def _public_action(value: Any) -> dict[str, Any]:
    """Whitelist numeric motion-state fields from one simulator receipt."""

    if not isinstance(value, Mapping):
        raise TypeError("Rover action receipts must be objects")
    result: dict[str, Any] = {}
    scene_id = value.get("scene_id")
    if scene_id is not None:
        if not isinstance(scene_id, str) or _OPAQUE_SCENE_ID.fullmatch(scene_id) is None:
            raise ValueError("Rover action receipt has an invalid opaque scene ID")
        result["scene_id"] = scene_id
    for name, size in (
        ("position_xy_m", 2),
        ("position_m", 3),
        ("camera_position_m", 3),
        ("linear_velocity_xy_m", 2),
        ("last_movement_delta_m", 3),
    ):
        raw = value.get(name)
        if raw is None:
            continue
        if (
            not isinstance(raw, Sequence)
            or isinstance(raw, (str, bytes))
            or len(raw) != size
        ):
            raise ValueError(f"actions[].{name} must contain {size} finite numbers")
        result[name] = [
            _finite(item, name=f"actions[].{name}[{index}]")
            for index, item in enumerate(raw)
        ]
    for name in (
        "body_yaw_degrees",
        "camera_yaw_degrees",
        "pitch_degrees",
        "angular_velocity_degrees",
        "distance_moved",
        "turn_degrees",
        "scan_coverage",
        "clearance_m",
    ):
        raw = value.get(name)
        if raw is not None:
            result[name] = _finite(raw, name=f"actions[].{name}")
    for name in (
        "scene_version",
        "map_version",
        "scan_count",
        "visible_voxels",
        "valid_depth_pixels",
        "action_count",
    ):
        if name in value:
            result[name] = _integer(value[name], name=f"actions[].{name}")
    for name in ("collision", "stopped", "success"):
        if name in value:
            if type(value[name]) is not bool:
                raise TypeError(f"actions[].{name} must be boolean")
            result[name] = value[name]
    if "error_code" in value:
        result["error_code"] = _optional_protocol_code(
            value["error_code"], name="actions[].error_code"
        )
    return result


def _public_model_decision(
    value: Any,
    *,
    expected_step: int,
) -> dict[str, Any]:
    """Expose the exact numeric Gemma decision, causal provenance, and status.

    Prompt text, arbitrary backend strings, simulator metadata, and
    environmental fields are deliberately omitted. Cryptographic bindings and
    numeric causal-token counts remain visible so an operator can verify that
    each motion came from a fresh local Gemma forward pass.
    """

    if not isinstance(value, Mapping):
        raise TypeError("Gemma model decisions must be objects")
    step = _integer(value.get("step"), name="model_decisions[].step", minimum=1)
    if step != expected_step:
        raise ValueError("Gemma model decisions must be complete and ordered")
    action = value.get("model_action")
    primitive = value.get("primitive_tool")
    if action not in _MODEL_ACTIONS or primitive not in _PRIMITIVE_TOOLS:
        raise ValueError("Gemma model decision has an invalid action protocol")
    if primitive != {"move_to": "move_to", "face": "turn", "stop": "stop"}[action]:
        raise ValueError("Gemma model decision and primitive disagree")
    logits = _finite_vector(
        value.get("model_action_logits"),
        name="model_decisions[].model_action_logits",
        size=len(_MODEL_ACTION_ORDER),
    )
    probabilities = _finite_vector(
        value.get("model_action_probabilities"),
        name="model_decisions[].model_action_probabilities",
        size=len(_MODEL_ACTION_ORDER),
    )
    if any(item < 0.0 or item > 1.0 for item in probabilities) or not math.isclose(
        sum(probabilities), 1.0, rel_tol=0.0, abs_tol=1e-5
    ):
        raise ValueError("Gemma action probabilities are invalid")
    maximum = max(logits)
    expected_probabilities = [math.exp(item - maximum) for item in logits]
    denominator = sum(expected_probabilities)
    expected_probabilities = [item / denominator for item in expected_probabilities]
    if any(
        not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-5)
        for observed, expected in zip(probabilities, expected_probabilities, strict=True)
    ):
        raise ValueError("Gemma action probabilities differ from its raw logits")
    if _MODEL_ACTION_ORDER[max(range(len(logits)), key=logits.__getitem__)] != action:
        raise ValueError("Gemma selected action differs from its raw action logits")
    causal_counts = {
        "scene_token_count": _integer(
            value.get("scene_token_count"),
            name="model_decisions[].scene_token_count",
            minimum=1,
        ),
        "robot_token_count": _integer(
            value.get("robot_token_count"),
            name="model_decisions[].robot_token_count",
            minimum=1,
        ),
        "history_token_count": _integer(
            value.get("history_token_count"),
            name="model_decisions[].history_token_count",
        ),
        "prompt_token_count": _integer(
            value.get("prompt_token_count"),
            name="model_decisions[].prompt_token_count",
            minimum=1,
        ),
        "decision_position": _integer(
            value.get("decision_position"),
            name="model_decisions[].decision_position",
        ),
    }
    causal_hashes = {
        name: _required_sha256(value.get(name), name=f"model_decisions[].{name}")
        for name in (
            "decision_tensor_sha256",
            "instruction_sha256",
            "active_prefix_sha256",
            "scene_prefix_sha256",
            "robot_tokens_sha256",
            "checkpoint_sha256",
        )
    }
    waypoint = _pair(
        value.get("model_waypoint_delta_robot_m"),
        name="model_decisions[].model_waypoint_delta_robot_m",
    )
    turn_delta = _finite(
        value.get("model_turn_delta_degrees"),
        name="model_decisions[].model_turn_delta_degrees",
    )
    heading = _finite(
        value.get("model_desired_heading_degrees"),
        name="model_decisions[].model_desired_heading_degrees",
    )
    raw_arguments = value.get("primitive_arguments")
    if not isinstance(raw_arguments, Mapping):
        raise TypeError("Gemma primitive arguments must be an object")
    expected_argument_names = {
        "move_to": {"x", "y"},
        "turn": {"angle_degrees"},
        "stop": set(),
    }[primitive]
    if set(raw_arguments) != expected_argument_names:
        raise ValueError("Gemma primitive argument schema changed")
    arguments = {
        str(name): _finite(raw_arguments[name], name=f"model_decisions[].{name}")
        for name in sorted(expected_argument_names)
    }
    if primitive == "turn" and not math.isclose(
        arguments["angle_degrees"], turn_delta, rel_tol=0.0, abs_tol=1e-8
    ):
        raise ValueError("Executed FACE turn differs from Gemma's raw turn delta")
    world_waypoint = (
        [arguments["x"], arguments["y"]] if primitive == "move_to" else None
    )
    execution = value.get("execution")
    if not isinstance(execution, Mapping):
        raise TypeError("Gemma model decision lacks an execution receipt")
    success = execution.get("success")
    executed = execution.get("executed")
    if type(success) is not bool or type(executed) is not bool:
        raise TypeError("Gemma execution status must be boolean")
    error_code = _optional_protocol_code(
        execution.get("error_code"), name="model_decisions[].execution.error_code"
    )
    if success and (not executed or error_code is not None):
        raise ValueError("Accepted Gemma decision has an inconsistent execution receipt")
    for name, expected in (
        ("actual_gemma_causal_forward", True),
        ("model_selected_every_waypoint_and_heading", True),
        ("deterministic_route_planner_used", False),
        ("substitution_applied", False),
        ("synthetic_stop_applied", False),
    ):
        if type(value.get(name)) is not bool or value.get(name) is not expected:
            raise ValueError(f"Gemma model decision attestation failed: {name}")
    if execution.get("substitution_applied") is not False:
        raise ValueError("Gemma execution substituted a different action")
    if execution.get("synthetic_stop_applied") is not False:
        raise ValueError("Gemma execution synthesized a stop")
    return {
        "step": step,
        "model_action": action,
        "model_action_logits": logits,
        "model_action_probabilities": probabilities,
        **causal_counts,
        **causal_hashes,
        "model_waypoint_delta_robot_m": waypoint,
        "model_turn_delta_degrees": turn_delta,
        "model_desired_heading_degrees": heading,
        "derived_absolute_facing_heading_degrees": heading,
        "derived_world_waypoint_xy_m": world_waypoint,
        "primitive_tool": primitive,
        "primitive_arguments": arguments,
        "accepted": success,
        "executed": executed,
        "error_code": error_code,
        "actual_gemma_causal_forward": True,
        "model_selected_every_waypoint_and_heading": True,
        "deterministic_route_planner_used": False,
        "substitution_applied": False,
        "synthetic_stop_applied": False,
    }


def _public_scene_memory(
    payload: Mapping[str, Any],
    state: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and whitelist nonsemantic continuous-memory evidence."""

    raw = payload.get("scene_memory")
    if not isinstance(raw, Mapping):
        raise TypeError("Rover result lacks continuous scene-memory diagnostics")
    if raw.get("schema") != _SCENE_MEMORY_SCHEMA:
        raise ValueError("Rover scene-memory diagnostic schema changed")

    def shape(name: str) -> list[int]:
        value = raw.get(name)
        if (
            not isinstance(value, Sequence)
            or isinstance(value, (str, bytes))
            or len(value) != 3
        ):
            raise ValueError(f"scene_memory.{name} must be a rank-three shape")
        result = [
            _integer(item, name=f"scene_memory.{name}[{index}]", minimum=1)
            for index, item in enumerate(value)
        ]
        if result[0] != 1:
            raise ValueError(f"scene_memory.{name} must have batch size one")
        return result

    tensor_shape = shape("tensor_shape")
    active_shape = shape("active_tensor_shape")
    token_count = _integer(
        raw.get("token_count"), name="scene_memory.token_count", minimum=1
    )
    model_dim = _integer(raw.get("model_dim"), name="scene_memory.model_dim", minimum=1)
    robot_tokens = _integer(
        raw.get("robot_state_token_count"),
        name="scene_memory.robot_state_token_count",
    )
    if tensor_shape != [1, token_count, model_dim]:
        raise ValueError("Scene-memory token shape disagrees with its dimensions")
    if active_shape != [1, token_count + robot_tokens, model_dim]:
        raise ValueError("Active scene-memory shape disagrees with numeric-state tokens")

    scene_hash = str(raw.get("sha256", "")).casefold()
    active_hash = str(raw.get("active_sha256", "")).casefold()
    if not _SHA256.fullmatch(scene_hash) or not _SHA256.fullmatch(active_hash):
        raise ValueError("Scene-memory tensor hashes must be SHA-256")
    if scene_hash != state.get("scene_prefix_hash"):
        raise ValueError("Scene-memory tensor hash differs from the active map binding")
    l2_norm = _finite(raw.get("l2_norm"), name="scene_memory.l2_norm")
    rms = _finite(raw.get("rms"), name="scene_memory.rms")
    active_l2 = _finite(
        raw.get("active_l2_norm"), name="scene_memory.active_l2_norm"
    )
    if min(l2_norm, rms, active_l2) <= 0.0:
        raise ValueError("Scene-memory tensor norms must be positive")

    map_version = _integer(raw.get("map_version"), name="scene_memory.map_version")
    if map_version != state.get("map_version"):
        raise ValueError("Scene-memory map version differs from rover state")
    source_voxels = _integer(
        raw.get("source_voxels"), name="scene_memory.source_voxels", minimum=1
    )
    processed_voxels = _integer(
        raw.get("processed_voxels"),
        name="scene_memory.processed_voxels",
        minimum=1,
    )
    semantic_feature_dim = _integer(
        raw.get("semantic_feature_dim"),
        name="scene_memory.semantic_feature_dim",
        minimum=1,
    )
    required_true = (
        "all_runtime_voxels_encoded",
        "base_adapter_weights_loaded",
        "control_weights_loaded",
        "control_training_gate_passed",
    )
    if any(raw.get(name) is not True for name in required_true):
        raise ValueError("Continuous scene-memory checkpoint attestation failed")
    if raw.get("question_dependent_scene_retrieval") is not False:
        raise ValueError("Question-dependent scene retrieval is prohibited")
    if raw.get("environmental_text_inputs") != []:
        raise ValueError("Scene-memory diagnostics must contain no environmental text")

    raw_audit = raw.get("loaded_file_audit")
    if not isinstance(raw_audit, Mapping):
        raise TypeError("Scene-memory loaded-file audit is missing")
    audit_enabled = raw_audit.get("enabled")
    audit_passed = raw_audit.get("passed")
    if type(audit_enabled) is not bool or audit_passed is not True:
        raise ValueError("Scene-memory loaded-file audit did not pass")
    loaded_file_count = _integer(
        raw_audit.get("loaded_file_count"),
        name="scene_memory.loaded_file_audit.loaded_file_count",
    )
    forbidden_access_count = _integer(
        raw_audit.get("forbidden_access_count"),
        name="scene_memory.loaded_file_audit.forbidden_access_count",
    )
    inventory_hash = str(
        raw_audit.get("loaded_file_inventory_sha256", "")
    ).casefold()
    if forbidden_access_count != 0 or not _SHA256.fullmatch(inventory_hash):
        raise ValueError("Scene-memory loaded-file inventory is invalid")
    return {
        "schema": _SCENE_MEMORY_SCHEMA,
        "tensor_shape": tensor_shape,
        "sha256": scene_hash,
        "l2_norm": l2_norm,
        "rms": rms,
        "token_count": token_count,
        "model_dim": model_dim,
        "active_tensor_shape": active_shape,
        "active_sha256": active_hash,
        "active_l2_norm": active_l2,
        "robot_state_token_count": robot_tokens,
        "map_version": map_version,
        "source_voxels": source_voxels,
        "processed_voxels": processed_voxels,
        "semantic_feature_dim": semantic_feature_dim,
        **{name: True for name in required_true},
        "question_dependent_scene_retrieval": False,
        "loaded_file_audit": {
            "enabled": audit_enabled,
            "loaded_file_count": loaded_file_count,
            "loaded_file_inventory_sha256": inventory_hash,
            "forbidden_access_count": 0,
            "passed": True,
        },
        "environmental_text_inputs": [],
    }


def _public_result(payload: Mapping[str, Any], room_size_m: Sequence[float]) -> dict[str, Any]:
    state = _public_state(payload, room_size_m)
    scene_memory = _public_scene_memory(payload, state)
    reply = payload.get("reply", "")
    if not isinstance(reply, str):
        raise TypeError("Rover result reply must be text")
    actions = payload.get("actions", ())
    if not isinstance(actions, Sequence) or isinstance(actions, (str, bytes)):
        raise TypeError("Rover result actions must be a sequence")
    if len(actions) > _MAX_ACTIONS_PER_TURN:
        raise ValueError("Rover result contains too many numeric action receipts")
    raw_decisions = payload.get("model_decisions", ())
    if not isinstance(raw_decisions, Sequence) or isinstance(
        raw_decisions, (str, bytes)
    ):
        raise TypeError("Rover result model_decisions must be a sequence")
    if len(raw_decisions) > _MAX_MODEL_DECISIONS_PER_TURN:
        raise ValueError("Rover result contains too many Gemma decisions")
    model_decisions = [
        _public_model_decision(value, expected_step=index)
        for index, value in enumerate(raw_decisions, start=1)
    ]
    control_mode = payload.get("control_mode", "unspecified")
    if not isinstance(control_mode, str) or not re.fullmatch(r"[a-z0-9_.-]{1,64}", control_mode):
        raise ValueError("Rover result control_mode must be an opaque protocol name")
    navigation_control_mode = payload.get("navigation_control_mode")
    if navigation_control_mode != _MODEL_ONLY_NAVIGATION_MODE:
        raise ValueError("Rover result is not bound to the model-only navigation backend")
    navigation_checkpoint_sha256 = _required_sha256(
        payload.get("navigation_checkpoint_sha256"),
        name="navigation_checkpoint_sha256",
    )
    gemma_runtime_binding_sha256 = _required_sha256(
        payload.get("gemma_runtime_binding_sha256"),
        name="gemma_runtime_binding_sha256",
    )
    instruction_hashes = {item["instruction_sha256"] for item in model_decisions}
    if len(instruction_hashes) > 1:
        raise ValueError("Gemma decisions in one turn changed the user-instruction binding")
    for item in model_decisions:
        if item["checkpoint_sha256"] != navigation_checkpoint_sha256:
            raise ValueError("Gemma decision checkpoint differs from the active controller")
        if item["scene_prefix_sha256"] != state["scene_prefix_hash"]:
            raise ValueError("Gemma decision scene prefix differs from the active map")
        if item["scene_token_count"] != scene_memory["token_count"]:
            raise ValueError("Gemma decision scene-token count differs from the active map")
        if item["robot_token_count"] != scene_memory["robot_state_token_count"]:
            raise ValueError("Gemma decision robot-token count differs from the active state")
    required_control_flags = {
        "high_level_natural_language_only": True,
        "task_trained_navigation": True,
        "model_selects_every_waypoint_and_heading": True,
        "model_selects_stop": True,
        "deterministic_route_planner_used": False,
        "fallback_used": False,
        "substitution_applied": False,
        "synthetic_stop_applied": False,
        "untrained_json_backend_enabled": False,
        "static_precomputed_scene_memory": True,
        "camera_control_input": False,
    }
    for name, expected in required_control_flags.items():
        observed = payload.get(name)
        if type(observed) is not bool or observed is not expected:
            raise ValueError(
                f"Rover result failed the high-level static-map control gate: {name}"
            )
    dynamic_control_flags: dict[str, bool] = {}
    for name in (
        "gemma_attempted",
        "gemma_accepted",
        "local_inference",
        "cloud_model_used",
        "initial_scan_performed",
        "scene_memory_refreshed",
    ):
        observed = payload.get(name)
        if type(observed) is not bool:
            raise TypeError(f"Rover result control flag must be boolean: {name}")
        dynamic_control_flags[name] = observed
    if dynamic_control_flags["gemma_accepted"] and not dynamic_control_flags[
        "gemma_attempted"
    ]:
        raise ValueError("Rover result accepted Gemma without running Gemma")
    if (
        control_mode == _MODEL_ONLY_NAVIGATION_MODE
        and dynamic_control_flags["gemma_accepted"]
    ):
        if not model_decisions:
            raise ValueError(
                "Successful Gemma navigation requires a nonempty model decision log"
            )
        terminal = model_decisions[-1]
        if not (
            terminal["model_action"] == "stop"
            and terminal["primitive_tool"] == "stop"
            and terminal["accepted"] is True
            and terminal["executed"] is True
        ):
            raise ValueError(
                "Successful Gemma navigation must end in accepted, executed Gemma STOP"
            )
        if terminal["active_prefix_sha256"] != scene_memory["active_sha256"]:
            raise ValueError("Terminal Gemma STOP differs from the final causal prefix")
    control = {
        "control_mode": control_mode,
        "navigation_control_mode": navigation_control_mode,
        "navigation_checkpoint_sha256": navigation_checkpoint_sha256,
        "gemma_runtime_binding_sha256": gemma_runtime_binding_sha256,
        **dynamic_control_flags,
        **required_control_flags,
    }
    return {
        "reply": reply[:_MAX_REPLY_CHARACTERS],
        "state": state,
        "actions": [_public_action(action) for action in actions],
        "model_decisions": model_decisions,
        "scene_prefix_hash": state["scene_prefix_hash"],
        "map_version": state["map_version"],
        "scene_memory": scene_memory,
        "control": control,
    }


async def _json_body(request: Request) -> dict[str, Any]:
    try:
        length = int(request.headers.get("content-length", "0"))
    except ValueError:
        length = _MAX_REQUEST_BYTES + 1
    if length > _MAX_REQUEST_BYTES:
        raise ValueError("request_too_large")
    raw = await request.body()
    if len(raw) > _MAX_REQUEST_BYTES:
        raise ValueError("request_too_large")
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError("invalid_json") from exc
    if not isinstance(payload, dict):
        raise TypeError("request_must_be_an_object")
    return payload


def _effective_port(scheme: str, port: int | None) -> int | None:
    if port is not None:
        return port
    return 443 if scheme == "https" else 80 if scheme == "http" else None


def _require_loopback_host(request: Request) -> None:
    """Reject DNS-rebinding Host values before any state is returned or changed."""

    host = request.headers.get("host", "")
    try:
        parsed = urlsplit(f"//{host}")
        hostname = parsed.hostname
        _ = parsed.port
    except ValueError as exc:
        raise ValueError("invalid_host") from exc
    if not hostname or hostname.casefold() not in _LOOPBACK_HOSTS:
        raise ValueError("loopback_host_required")


def _require_json_mutation(request: Request) -> None:
    """Require a non-simple, same-origin browser request for rover mutations."""

    _require_loopback_host(request)
    media_type = request.headers.get("content-type", "").split(";", 1)[0].strip()
    if media_type.casefold() != "application/json":
        raise ValueError("application_json_required")
    origin = request.headers.get("origin")
    if origin is None:
        return
    try:
        source = urlsplit(origin)
        target = request.url
        source_host = source.hostname
        target_host = target.hostname
        source_port = source.port
        target_port = target.port
    except ValueError as exc:
        raise ValueError("same_origin_required") from exc
    if (
        source.scheme not in {"http", "https"}
        or source_host is None
        or target_host is None
        or source_host.casefold() not in _LOOPBACK_HOSTS
        or source.scheme.casefold() != target.scheme.casefold()
        or source_host.casefold() != target_host.casefold()
        or _effective_port(source.scheme.casefold(), source_port)
        != _effective_port(target.scheme.casefold(), target_port)
    ):
        raise ValueError("same_origin_required")


_PAGE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="dark">
  <title>Local Semantic 3D Rover</title>
  <style>
    :root { color-scheme:dark; --ink:#eef8f5; --muted:#8da6a3; --bg:#06110f;
      --panel:#0b1d1a; --panel2:#102824; --line:#21453e; --mint:#64f0c1;
      --cyan:#66cbe8; --amber:#ffc66d; --red:#ff776f; --shadow:#0008; }
    * { box-sizing:border-box; }
    body { margin:0; min-height:100vh; color:var(--ink); font:14px/1.45 Inter,ui-sans-serif,
      system-ui,-apple-system,sans-serif; background:radial-gradient(circle at 20% -10%,#153d34 0,
      #071511 35%,var(--bg) 75%); }
    button,textarea { font:inherit; }
    header { height:74px; padding:14px 22px; display:flex; align-items:center;
      justify-content:space-between; gap:18px; border-bottom:1px solid var(--line);
      background:#06110fe8; backdrop-filter:blur(14px); position:sticky; top:0; z-index:5; }
    h1 { margin:0; font-size:19px; letter-spacing:.02em; }
    .eyebrow { color:var(--mint); text-transform:uppercase; font:700 10px/1.2 ui-monospace,monospace;
      letter-spacing:.16em; margin-bottom:4px; }
    .badges { display:flex; align-items:center; justify-content:flex-end; gap:8px; flex-wrap:wrap; }
    .badge { border:1px solid var(--line); border-radius:999px; padding:5px 9px; color:var(--muted);
      font:600 11px ui-monospace,monospace; background:#0b1d1a; }
    .badge.live { color:var(--mint); border-color:#2d7964; }
    .pulse { width:7px; height:7px; border-radius:50%; display:inline-block; margin-right:6px;
      background:currentColor; box-shadow:0 0 10px currentColor; }
    main { max-width:1280px; margin:auto; padding:16px; display:grid;
      grid-template-columns:minmax(420px,1fr) minmax(380px,.85fr); gap:16px; }
    .stack { display:grid; gap:16px; align-content:start; }
    .panel { background:linear-gradient(145deg,#0d211de8,#081714ef); border:1px solid var(--line);
      border-radius:14px; overflow:hidden; box-shadow:0 18px 50px var(--shadow); }
    .panel-head { min-height:43px; padding:10px 13px; border-bottom:1px solid var(--line);
      display:flex; align-items:center; justify-content:space-between; gap:12px; }
    .panel-title { color:#b9cbc8; font:750 11px ui-monospace,monospace; letter-spacing:.1em;
      text-transform:uppercase; }
    .hint { color:var(--muted); font-size:11px; }
    .visuals { display:grid; grid-template-columns:1fr; gap:16px; }
    .visual img { display:block; width:100%; max-height:410px; aspect-ratio:16/10;
      object-fit:contain; background:#040a09; }
    .memory-strip { padding:10px 13px; display:flex; flex-wrap:wrap; gap:8px;
      color:var(--muted); font:11px ui-monospace,monospace; }
    .memory-strip span { border:1px solid var(--line); border-radius:999px; padding:5px 8px; }
    button { border:1px solid #326158; border-radius:9px; background:#102c26; color:var(--ink);
      min-height:38px; padding:8px 9px; cursor:pointer; font-weight:700; }
    button:hover { border-color:var(--mint); background:#153a32; }
    button:focus-visible,textarea:focus-visible { outline:2px solid var(--cyan); outline-offset:2px; }
    button:disabled { opacity:.45; cursor:wait; }
    .chat { height:min(760px,calc(100vh - 106px)); min-height:560px; display:grid;
      grid-template-rows:auto 1fr auto; }
    #messages { min-height:0; overflow:auto; padding:13px; }
    .message { max-width:92%; margin:0 0 10px; padding:10px 11px; border-radius:11px;
      white-space:pre-wrap; overflow-wrap:anywhere; }
    .message.user { margin-left:auto; background:#15334a; border:1px solid #315c7c; }
    .message.gemma { background:#103129; border:1px solid #296c59; }
    .message.agent { max-width:100%; color:#ffe1a8; background:#261c0d; border:1px solid #72501c;
      font:11px ui-monospace,monospace; }
    .message.system { max-width:100%; color:var(--muted); background:#0a1715; border:1px dashed #29433e;
      font:11px ui-monospace,monospace; }
    form { border-top:1px solid var(--line); padding:12px; display:grid; gap:8px; }
    textarea { width:100%; min-height:78px; max-height:180px; resize:vertical; color:var(--ink);
      background:#06120f; border:1px solid #31574f; border-radius:9px; padding:10px; }
    .form-row { display:flex; justify-content:space-between; align-items:center; gap:10px; }
    #error { color:var(--red); font-size:11px; min-height:16px; }
    #send { color:#062018; background:var(--mint); border-color:var(--mint); min-width:116px; }
    .no { color:var(--red); }
    @media (max-width:900px) { main { grid-template-columns:1fr; } .chat { height:720px; } }
    @media (max-width:700px) { header { height:auto; align-items:flex-start; } main { padding:9px; }
      .visuals { grid-template-columns:1fr; } .chat { min-height:620px; }
      .badge.hash { display:none; } }
  </style>
</head>
<body>
<header>
  <div><div class="eyebrow">Pre-scanned semantic map · local-only runtime</div><h1>Local Semantic 3D Rover</h1></div>
  <div class="badges"><span class="badge live"><i class="pulse"></i><span id="online">starting</span></span>
    <span class="badge" id="control-mode">controller starting</span><span class="badge hash" id="binding">prefix —</span></div>
</header>
<main>
  <section class="stack">
    <div class="visuals">
      <article class="panel visual" id="map-card"><div class="panel-head"><span class="panel-title">Precomputed embedded 3D map</span><span class="hint">the controller uses continuous map tokens</span></div><img id="map" alt="Fused semantic point-map preview"><div class="memory-strip"><span id="token-shape">tokens —</span><span id="prefix-binding">prefix —</span><span id="pose-summary">pose —</span></div></article>
      <article class="panel visual" id="overview-card"><div class="panel-head"><span class="panel-title">Pre-scan overview</span><span class="hint">human display only · never a live control input</span></div><img id="overview" alt="RGB room scan montage"></article>
    </div>
  </section>
  <aside class="stack">
    <article class="panel chat">
      <div class="panel-head"><span class="panel-title">Free-form high-level goal</span><span class="hint">Gemma chooses every waypoint, turn, recovery, and STOP</span></div>
      <div id="messages"><div class="message system">Starting the local scene memory and rover session…</div></div>
      <form id="chat-form"><textarea id="instruction" maxlength="2048" required placeholder="Describe the outcome you want in ordinary language."></textarea>
        <div class="form-row"><span id="error"></span><button id="send" type="submit">Send free-form goal to Gemma</button></div></form>
    </article>
  </aside>
</main>
<script>
const el = id => document.getElementById(id); let busy = false;
function renderState(state,memory) {
  const [x,y]=state.position_xy_m;
  el('binding').textContent=`prefix ${state.scene_prefix_hash.slice(0,10)}…`;
  el('prefix-binding').textContent=`prefix ${state.scene_prefix_hash.slice(0,12)}…`;
  el('pose-summary').textContent=`pose x ${x.toFixed(2)} · y ${y.toFixed(2)} · yaw ${state.body_yaw_degrees.toFixed(0)}°`;
  el('token-shape').textContent=`tokens ${memory.tensor_shape.join(' × ')}`;
  el('online').textContent=state.stopped?'local · stopped':'local · ready';
}
function addMessage(kind,text) { const node=document.createElement('div'); node.className=`message ${kind}`; node.textContent=text;
  el('messages').appendChild(node); el('messages').scrollTop=el('messages').scrollHeight; }
function renderDecision(value) {
  const status=value.accepted&&value.executed?'ACCEPTED · EXECUTED':value.executed?'REJECTED AFTER EXECUTION':'REJECTED · NOT EXECUTED';
  const [right,forward]=value.model_waypoint_delta_robot_m;
  let exact='';
  if(value.primitive_tool==='move_to') {
    const [worldX,worldY]=value.derived_world_waypoint_xy_m;
    exact=`Gemma raw waypoint · robot-frame Δ right ${right>=0?'+':''}${right.toFixed(3)} m, forward ${forward>=0?'+':''}${forward.toFixed(3)} m · deterministic frame transform → world x=${worldX>=0?'+':''}${worldX.toFixed(3)}, y=${worldY>=0?'+':''}${worldY.toFixed(3)} m`;
  } else if(value.primitive_tool==='turn') {
    const rawTurn=value.model_turn_delta_degrees;
    exact=`Gemma raw turn Δ ${rawTurn>=0?'+':''}${rawTurn.toFixed(3)}° · deterministic absolute facing ${value.derived_absolute_facing_heading_degrees>=0?'+':''}${value.derived_absolute_facing_heading_degrees.toFixed(3)}° · executed exact raw Δ`;
  } else exact='Gemma raw action · model-selected STOP';
  const error=value.error_code?` · ${value.error_code}`:'';
  const [moveP,faceP,stopP]=value.model_action_probabilities;
  const logits=value.model_action_logits.map(number=>number.toFixed(3)).join(', ');
  const causal=`causal context ${value.scene_token_count} scene + ${value.robot_token_count} robot + ${value.history_token_count} history + ${value.prompt_token_count} prompt tokens · decision position ${value.decision_position}`;
  const provenance=`p(move/face/stop) ${moveP.toFixed(3)} / ${faceP.toFixed(3)} / ${stopP.toFixed(3)} · raw logits [${logits}] · output ${value.decision_tensor_sha256.slice(0,12)}… · active prefix ${value.active_prefix_sha256.slice(0,12)}… · checkpoint ${value.checkpoint_sha256.slice(0,12)}…`;
  return `Gemma step ${String(value.step).padStart(3,'0')} · ${value.model_action.toUpperCase()} · ${exact} · ${status}${error}\n${causal}\n${provenance}`;
}
function renderControl(control) { let label='local controller';
  if(control.cloud_model_used) label='control gate failed';
  else if(control.gemma_attempted&&control.gemma_accepted) label='actual Gemma decisions · local';
  else if(control.local_inference) label='semantic map · local';
  el('control-mode').textContent=label; el('control-mode').className=`badge ${control.cloud_model_used?'no':''}`; }
function setBusy(value) { busy=value; document.querySelectorAll('button').forEach(node=>node.disabled=value); el('instruction').disabled=value; }
async function request(path,payload) { const response=await fetch(path,{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(payload)});
  const body=await response.json(); if(!response.ok) throw new Error(body.error||`request failed (${response.status})`); renderState(body.state,body.scene_memory); renderControl(body.control); return body; }
async function sendInstruction(text) { if(busy||!text.trim())return; addMessage('user',text.trim()); setBusy(true); el('error').textContent='';
  addMessage('system','Gemma is reasoning over the fixed continuous 3D scene memory…');
  try { const body=await request('/api/instruction',{instruction:text.trim()});
    for(const decision of body.model_decisions||[]) addMessage('agent',renderDecision(decision));
    addMessage('gemma',body.reply||'Goal processing finished.'); }
  catch(error){ el('error').textContent=String(error); addMessage('system',`Request failed: ${error}`); } finally { setBusy(false); el('instruction').focus(); } }
el('chat-form').addEventListener('submit',event=>{event.preventDefault();const text=el('instruction').value;el('instruction').value='';sendInstruction(text);});
async function initialize(){const response=await fetch('/api/state');const body=await response.json();if(!response.ok)throw new Error(body.error||'startup failed');
  renderState(body.state,body.scene_memory);renderControl(body.control); for(const name of ['overview','map']){if(body.visuals[name])el(name).src=body.visuals[name];else el(`${name}-card`).hidden=true;}
  el('messages').innerHTML='';addMessage('system',`Ready in ${body.state.scene_id}. The room was embedded before this dialogue; the rover camera is not a control input.`);
  if(body.reply)addMessage('gemma',body.reply);}
initialize().catch(error=>{el('online').textContent='startup error';el('error').textContent=String(error);});
</script>
</body>
</html>
"""


def create_rover_web_app(
    session: RoverSession,
    room_size_m: Sequence[float],
    *,
    visual_assets: Mapping[str, str | Path] | None = None,
    figure_root: str | Path = PROJECT_ROOT / "reports" / "gemma4" / "figures",
) -> Starlette:
    """Create a loopback-ready ASGI app around an unopened rover session."""

    room = _room_size(room_size_m)
    assets = validate_visual_assets(visual_assets or {}, Path(figure_root).resolve())
    session_lock = asyncio.Lock()
    public: dict[str, Any] | None = None
    closed = False

    async def run_session(method: Any, *args: Any) -> Mapping[str, Any]:
        nonlocal public
        async with session_lock:
            result = await asyncio.to_thread(method, *args)
            if not isinstance(result, Mapping):
                raise TypeError("Rover session returned a non-object result")
            public = _public_result(result, room)
            return result

    @asynccontextmanager
    async def lifespan(_app: Starlette):
        nonlocal closed
        try:
            await run_session(session.startup)
            yield
        finally:
            if not closed:
                async with session_lock:
                    await asyncio.to_thread(session.close)
                    closed = True

    def response_payload() -> dict[str, Any]:
        if public is None:
            raise RuntimeError("Rover session is not ready")
        return {**public, "visuals": {name: f"/assets/{name}" for name in assets}}

    async def index(request: Request) -> Response:
        try:
            _require_loopback_host(request)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        return HTMLResponse(
            _PAGE,
            headers={
                "Cache-Control": "no-store",
                "X-Content-Type-Options": "nosniff",
                "X-Frame-Options": "DENY",
                "Content-Security-Policy": "default-src 'self'; img-src 'self' data:; style-src 'unsafe-inline'; script-src 'unsafe-inline'",
            },
        )

    async def state(request: Request) -> Response:
        try:
            _require_loopback_host(request)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        return JSONResponse(response_payload(), headers={"Cache-Control": "no-store"})

    async def instruction(request: Request) -> Response:
        try:
            _require_json_mutation(request)
            payload = await _json_body(request)
            text = payload.get("instruction")
            if not isinstance(text, str) or not text.strip():
                raise ValueError("instruction_must_be_nonempty_text")
            if len(text) > _MAX_INSTRUCTION_CHARACTERS:
                raise ValueError("instruction_too_long")
            await run_session(session.handle_instruction, text.strip())
            return JSONResponse(response_payload(), headers={"Cache-Control": "no-store"})
        except (TypeError, ValueError) as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        except RuntimeError as exc:
            return JSONResponse({"error": str(exc)}, status_code=409)

    async def close(request: Request) -> Response:
        nonlocal closed
        try:
            _require_json_mutation(request)
            payload = await _json_body(request)
            if payload:
                raise ValueError("close_requires_empty_object")
        except (TypeError, ValueError) as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        if not closed:
            async with session_lock:
                await asyncio.to_thread(session.close)
                closed = True
        return JSONResponse({"closed": True}, headers={"Cache-Control": "no-store"})

    async def asset(request: Request) -> Response:
        try:
            _require_loopback_host(request)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        name = str(request.path_params["asset_name"])
        path = assets.get(name)
        if path is None:
            return JSONResponse({"error": "unknown_asset"}, status_code=404)
        return FileResponse(
            path,
            headers={"Cache-Control": "public, max-age=300", "X-Content-Type-Options": "nosniff"},
        )

    app = Starlette(
        debug=False,
        lifespan=lifespan,
        routes=[
            Route("/", index, methods=["GET"]),
            Route("/api/state", state, methods=["GET"]),
            Route("/api/instruction", instruction, methods=["POST"]),
            Route("/api/close", close, methods=["POST"]),
            Route("/assets/{asset_name:str}", asset, methods=["GET"]),
        ],
    )
    app.state.rover_session = session
    app.state.room_size_m = room
    app.state.visual_assets = dict(assets)
    return app


def serve_rover_web_app(
    app: Starlette,
    *,
    host: str = "127.0.0.1",
    port: int = 8770,
) -> None:
    """Serve an already constructed rover app, refusing non-loopback binds."""

    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("Rover web UI accepts loopback binds only")
    if not 1 <= int(port) <= 65_535:
        raise ValueError("port must be between 1 and 65535")
    import uvicorn

    uvicorn.run(app, host=host, port=int(port), log_level="info", access_log=False)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--host", default="127.0.0.1")
    result.add_argument("--port", type=int, default=8770)
    return result


def main(argv: list[str] | None = None) -> int:
    """Explain the injection requirement when invoked without the project launcher."""

    args = parser().parse_args(argv)
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        print("Rover web UI accepts loopback binds only", file=sys.stderr)
        return 2
    print(
        "Use the project rover-demo launcher; it constructs the local Gemma/MCP "
        f"session before serving http://{args.host}:{args.port}.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "RoverSession",
    "create_rover_web_app",
    "serve_rover_web_app",
]
