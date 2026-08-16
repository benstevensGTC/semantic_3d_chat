"""Small, dependency-free bridge between Blender and the loopback rover API.

Blender ships its own Python interpreter, so the operator UI deliberately uses
only the standard library.  This module owns URL, request, response, and pose
validation and can therefore be tested without importing ``bpy`` or loading a
model.  Environmental information never crosses this bridge as labels: the UI
receives only the public reply plus numeric robot state already exposed by the
local rover process.
"""

from __future__ import annotations

import json
import math
import re
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final
from urllib.parse import urlsplit, urlunsplit

_LOOPBACK_HOSTS: Final[frozenset[str]] = frozenset({"127.0.0.1", "localhost", "::1"})
_OPAQUE_SCENE_ID: Final[re.Pattern[str]] = re.compile(r"scene_[0-9]{6}")
_SHA256: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{64}")
_MAX_RESPONSE_BYTES: Final[int] = 2 * 1024 * 1024
_MAX_INSTRUCTION_CHARACTERS: Final[int] = 2_048
_MAX_ACTIONS_PER_TURN: Final[int] = 128
_MAX_MODEL_DECISIONS_PER_TURN: Final[int] = 128
_MODEL_ACTION_ORDER: Final[tuple[str, str, str]] = ("move_to", "face", "stop")
_MODEL_ACTIONS: Final[frozenset[str]] = frozenset(_MODEL_ACTION_ORDER)
_MODEL_ONLY_NAVIGATION_MODE: Final[str] = (
    "actual_local_gemma_model_only_waypoint_policy"
)
_PROTOCOL_CODE: Final[re.Pattern[str]] = re.compile(r"[A-Z][A-Z0-9_]{0,63}")
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


def normalize_loopback_url(value: str) -> str:
    """Return a canonical HTTP loopback origin and reject every remote URL."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError("A loopback backend URL is required")
    parsed = urlsplit(value.strip())
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("The backend URL has an invalid port") from exc
    if (
        parsed.scheme.casefold() != "http"
        or parsed.hostname is None
        or parsed.hostname.casefold() not in _LOOPBACK_HOSTS
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
    ):
        raise ValueError("The Blender rover accepts only a plain HTTP loopback origin")
    if port is not None and not 1 <= port <= 65_535:
        raise ValueError("The backend port must be in [1, 65535]")
    host = parsed.hostname.casefold()
    authority = f"[{host}]" if ":" in host else host
    if port is not None:
        authority = f"{authority}:{port}"
    return urlunsplit(("http", authority, "", "", ""))


@dataclass(frozen=True, slots=True)
class RoverPose:
    """Validated numeric pose used to animate the operator-only rover mesh."""

    scene_id: str
    x_m: float
    y_m: float
    body_yaw_degrees: float
    camera_yaw_degrees: float
    pitch_degrees: float
    collision: bool
    stopped: bool
    scene_version: int
    map_version: int
    scan_count: int


@dataclass(frozen=True, slots=True)
class SceneMemoryDiagnostics:
    """Nonsemantic diagnostics for the active continuous prefix."""

    shape: tuple[int, ...]
    sha256: str
    l2_norm: float
    token_count: int
    model_dim: int


@dataclass(frozen=True, slots=True)
class GemmaDecisionReceipt:
    """Strict numeric record of one actual-Gemma motion decision."""

    step: int
    model_action: str
    action_logits: tuple[float, float, float]
    action_probabilities: tuple[float, float, float]
    waypoint_delta_robot_m: tuple[float, float]
    turn_delta_degrees: float
    desired_heading_degrees: float
    scene_token_count: int
    robot_token_count: int
    history_token_count: int
    prompt_token_count: int
    decision_position: int
    decision_tensor_sha256: str
    active_prefix_sha256: str
    checkpoint_sha256: str
    primitive_tool: str
    target_xy_m: tuple[float, float] | None
    turn_degrees: float | None
    accepted: bool
    executed: bool
    error_code: str | None


@dataclass(frozen=True, slots=True)
class RoverResponse:
    """Human reply and numeric state returned by the local controller."""

    pose: RoverPose
    reply: str
    control_mode: str
    trajectory: tuple[RoverPose, ...] = ()
    events: tuple[str, ...] = ()
    decisions: tuple[GemmaDecisionReceipt, ...] = ()
    scene_memory: SceneMemoryDiagnostics | None = None


def _action_pose(value: Any, final_pose: RoverPose) -> RoverPose | None:
    """Convert one numeric action receipt into a pose without reading labels."""

    if not isinstance(value, Mapping):
        return None
    raw_position = value.get("position_xy_m", value.get("position_m"))
    if (
        not isinstance(raw_position, Sequence)
        or isinstance(raw_position, (str, bytes))
        or len(raw_position) not in {2, 3}
    ):
        return None
    scene_id = value.get("scene_id", final_pose.scene_id)
    if scene_id != final_pose.scene_id:
        raise ValueError("A rover action receipt changed the opaque scene ID")

    def integer(name: str, fallback: int) -> int:
        raw = value.get(name, fallback)
        numeric = _finite(raw, name=f"actions[].{name}")
        if not numeric.is_integer() or numeric < 0:
            raise ValueError(f"actions[].{name} must be a nonnegative integer")
        return int(numeric)

    return RoverPose(
        scene_id=final_pose.scene_id,
        x_m=_finite(raw_position[0], name="actions[].position[0]"),
        y_m=_finite(raw_position[1], name="actions[].position[1]"),
        body_yaw_degrees=_finite(
            value.get("body_yaw_degrees", final_pose.body_yaw_degrees),
            name="actions[].body_yaw_degrees",
        ),
        camera_yaw_degrees=_finite(
            value.get("camera_yaw_degrees", final_pose.camera_yaw_degrees),
            name="actions[].camera_yaw_degrees",
        ),
        pitch_degrees=_finite(
            value.get("pitch_degrees", final_pose.pitch_degrees),
            name="actions[].pitch_degrees",
        ),
        collision=bool(value.get("collision", False)),
        stopped=bool(value.get("stopped", final_pose.stopped)),
        scene_version=integer("scene_version", final_pose.scene_version),
        map_version=integer("map_version", final_pose.map_version),
        scan_count=integer("scan_count", final_pose.scan_count),
    )


def _memory_diagnostics(
    payload: Mapping[str, Any], pose: RoverPose
) -> SceneMemoryDiagnostics | None:
    value = payload.get("scene_memory")
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError("scene_memory diagnostics must be an object")
    if value.get("schema") != "semantic_3d_chat.scene_memory_diagnostics.v1":
        raise ValueError("scene_memory diagnostics have an unsupported schema")
    if value.get("environmental_text_inputs") != []:
        raise ValueError("scene_memory diagnostics report environmental text inputs")
    if value.get("question_dependent_scene_retrieval") is not False:
        raise ValueError("scene_memory diagnostics are not question independent")
    if value.get("all_runtime_voxels_encoded") is not True:
        raise ValueError("scene_memory diagnostics do not cover every runtime voxel")
    raw_shape = value.get("tensor_shape")
    if (
        not isinstance(raw_shape, Sequence)
        or isinstance(raw_shape, (str, bytes))
        or not 2 <= len(raw_shape) <= 4
    ):
        raise ValueError("scene_memory.tensor_shape must contain two to four dimensions")
    shape: list[int] = []
    for index, raw in enumerate(raw_shape):
        numeric = _finite(raw, name=f"scene_memory.tensor_shape[{index}]")
        if not numeric.is_integer() or numeric < 1:
            raise ValueError("scene_memory.tensor_shape values must be positive integers")
        shape.append(int(numeric))
    sha256 = value.get("sha256")
    if not isinstance(sha256, str) or _SHA256.fullmatch(sha256.casefold()) is None:
        raise ValueError("scene_memory.sha256 must be a lowercase SHA-256 digest")
    if payload.get("scene_prefix_hash") != sha256.casefold():
        raise ValueError("scene_memory SHA-256 differs from the public scene-prefix binding")
    norm = _finite(value.get("l2_norm"), name="scene_memory.l2_norm")
    if norm <= 0.0:
        raise ValueError("scene_memory.l2_norm must be positive")
    token_count = _finite(
        value.get("token_count", shape[-2]), name="scene_memory.token_count"
    )
    model_dim = _finite(value.get("model_dim", shape[-1]), name="scene_memory.model_dim")
    map_version = _finite(value.get("map_version"), name="scene_memory.map_version")
    if (
        not token_count.is_integer()
        or token_count < 1
        or not model_dim.is_integer()
        or model_dim < 1
        or int(token_count) != shape[-2]
        or int(model_dim) != shape[-1]
        or not map_version.is_integer()
        or int(map_version) != pose.map_version
    ):
        raise ValueError("scene_memory token count or model dimension disagrees with shape")
    for gate in (
        "base_adapter_weights_loaded",
        "control_weights_loaded",
        "control_training_gate_passed",
    ):
        if value.get(gate) is not True:
            raise ValueError(f"scene_memory diagnostic gate failed: {gate}")
    audit = value.get("loaded_file_audit")
    if (
        not isinstance(audit, Mapping)
        or audit.get("passed") is not True
        or audit.get("forbidden_access_count") != 0
    ):
        raise ValueError("scene_memory loaded-file audit did not pass")
    return SceneMemoryDiagnostics(
        shape=tuple(shape),
        sha256=sha256.casefold(),
        l2_norm=norm,
        token_count=int(token_count),
        model_dim=int(model_dim),
    )


def _action_event(value: Any) -> str | None:
    """Describe an action using numeric protocol values, never scene labels."""

    if not isinstance(value, Mapping):
        return None
    if value.get("collision") is True or value.get("error_code") == "E_COLLISION":
        return "Safety stop: collision detected; pose unchanged."
    distance = value.get("distance_moved")
    if distance is not None and abs(_finite(distance, name="actions[].distance_moved")) > 1e-6:
        position = value.get("position_xy_m", value.get("position_m"))
        suffix = ""
        if (
            isinstance(position, Sequence)
            and not isinstance(position, (str, bytes))
            and len(position) in {2, 3}
        ):
            suffix = (
                f" to x={_finite(position[0], name='actions[].position[0]'):+.2f},"
                f" y={_finite(position[1], name='actions[].position[1]'):+.2f} m"
            )
        return f"Translated {abs(float(distance)):.2f} m{suffix}."
    turn = value.get("turn_degrees")
    if turn is not None and abs(_finite(turn, name="actions[].turn_degrees")) > 1e-6:
        return f"Rotated {_finite(turn, name='actions[].turn_degrees'):+.1f}°."
    depth = value.get("valid_depth_pixels")
    if depth is not None:
        count = _finite(depth, name="actions[].valid_depth_pixels")
        if count.is_integer() and count > 0:
            return f"Fused a new RGB-D observation ({int(count):,} valid depth pixels)."
    if value.get("success") is False:
        code = value.get("error_code", "E_ACTION")
        return f"Action rejected safely ({str(code)[:32]})."
    return None


def _model_decision(value: Any, *, expected_step: int) -> GemmaDecisionReceipt:
    """Validate the browser's numeric-only actual-Gemma decision record."""

    if not isinstance(value, Mapping):
        raise TypeError("Gemma model decisions must be objects")
    step_number = _finite(value.get("step"), name="model_decisions[].step")
    if not step_number.is_integer() or int(step_number) != expected_step:
        raise ValueError("Gemma model decisions must be complete and ordered")
    action = value.get("model_action")
    primitive = value.get("primitive_tool")
    expected_primitive = {"move_to": "move_to", "face": "turn", "stop": "stop"}
    if action not in _MODEL_ACTIONS or primitive != expected_primitive.get(action):
        raise ValueError("Gemma model decision has an invalid action protocol")
    raw_logits = value.get("model_action_logits")
    raw_probabilities = value.get("model_action_probabilities")
    if (
        not isinstance(raw_logits, Sequence)
        or isinstance(raw_logits, (str, bytes))
        or len(raw_logits) != len(_MODEL_ACTION_ORDER)
        or not isinstance(raw_probabilities, Sequence)
        or isinstance(raw_probabilities, (str, bytes))
        or len(raw_probabilities) != len(_MODEL_ACTION_ORDER)
    ):
        raise ValueError("Gemma action-head output must contain three numeric values")
    logits = tuple(
        _finite(item, name=f"model_decisions[].logits[{index}]")
        for index, item in enumerate(raw_logits)
    )
    probabilities = tuple(
        _finite(item, name=f"model_decisions[].probabilities[{index}]")
        for index, item in enumerate(raw_probabilities)
    )
    if any(item < 0.0 or item > 1.0 for item in probabilities) or not math.isclose(
        sum(probabilities), 1.0, rel_tol=0.0, abs_tol=1e-5
    ):
        raise ValueError("Gemma action probabilities are invalid")
    maximum = max(logits)
    expected_probabilities = tuple(math.exp(item - maximum) for item in logits)
    denominator = sum(expected_probabilities)
    expected_probabilities = tuple(item / denominator for item in expected_probabilities)
    if any(
        not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-5)
        for observed, expected in zip(probabilities, expected_probabilities, strict=True)
    ):
        raise ValueError("Gemma action probabilities differ from its raw logits")
    if _MODEL_ACTION_ORDER[max(range(len(logits)), key=logits.__getitem__)] != action:
        raise ValueError("Gemma selected action differs from its raw action logits")

    def nonnegative_integer(name: str, *, minimum: int = 0) -> int:
        numeric = _finite(value.get(name), name=f"model_decisions[].{name}")
        if not numeric.is_integer() or numeric < minimum:
            raise ValueError(f"model_decisions[].{name} must be an integer >= {minimum}")
        return int(numeric)

    scene_token_count = nonnegative_integer("scene_token_count", minimum=1)
    robot_token_count = nonnegative_integer("robot_token_count", minimum=1)
    history_token_count = nonnegative_integer("history_token_count")
    prompt_token_count = nonnegative_integer("prompt_token_count", minimum=1)
    decision_position = nonnegative_integer("decision_position")

    def digest(name: str) -> str:
        raw = value.get(name)
        if not isinstance(raw, str) or _SHA256.fullmatch(raw.casefold()) is None:
            raise ValueError(f"model_decisions[].{name} must be a SHA-256 digest")
        return raw.casefold()

    decision_tensor_sha256 = digest("decision_tensor_sha256")
    active_prefix_sha256 = digest("active_prefix_sha256")
    checkpoint_sha256 = digest("checkpoint_sha256")
    digest("instruction_sha256")
    digest("scene_prefix_sha256")
    digest("robot_tokens_sha256")
    raw_delta = value.get("model_waypoint_delta_robot_m")
    if (
        not isinstance(raw_delta, Sequence)
        or isinstance(raw_delta, (str, bytes))
        or len(raw_delta) != 2
    ):
        raise ValueError("Gemma waypoint delta must contain two finite numbers")
    delta = (
        _finite(raw_delta[0], name="model_decisions[].delta[0]"),
        _finite(raw_delta[1], name="model_decisions[].delta[1]"),
    )
    heading = _finite(
        value.get("model_desired_heading_degrees"),
        name="model_decisions[].model_desired_heading_degrees",
    )
    turn_delta = _finite(
        value.get("model_turn_delta_degrees"),
        name="model_decisions[].model_turn_delta_degrees",
    )
    derived_heading = _finite(
        value.get("derived_absolute_facing_heading_degrees"),
        name="model_decisions[].derived_absolute_facing_heading_degrees",
    )
    if not math.isclose(heading, derived_heading, rel_tol=0.0, abs_tol=1e-8):
        raise ValueError("Gemma heading compatibility field differs from its derivation")
    arguments = value.get("primitive_arguments")
    if not isinstance(arguments, Mapping):
        raise TypeError("Gemma primitive arguments must be an object")
    target: tuple[float, float] | None = None
    turn: float | None = None
    if primitive == "move_to":
        if set(arguments) != {"x", "y"}:
            raise ValueError("Gemma MOVE_TO arguments changed")
        target = (
            _finite(arguments["x"], name="model_decisions[].x"),
            _finite(arguments["y"], name="model_decisions[].y"),
        )
        raw_world_target = value.get("derived_world_waypoint_xy_m")
        if (
            not isinstance(raw_world_target, Sequence)
            or isinstance(raw_world_target, (str, bytes))
            or len(raw_world_target) != 2
        ):
            raise ValueError("Derived world waypoint must contain two finite numbers")
        derived_target = (
            _finite(raw_world_target[0], name="model_decisions[].world_x"),
            _finite(raw_world_target[1], name="model_decisions[].world_y"),
        )
        if target != derived_target:
            raise ValueError("Gemma world-waypoint derivation differs from execution target")
    elif primitive == "turn":
        if value.get("derived_world_waypoint_xy_m") is not None:
            raise ValueError("FACE must not claim a derived world waypoint")
        if set(arguments) != {"angle_degrees"}:
            raise ValueError("Gemma FACE arguments changed")
        turn = _finite(
            arguments["angle_degrees"],
            name="model_decisions[].angle_degrees",
        )
        if not math.isclose(turn, turn_delta, rel_tol=0.0, abs_tol=1e-8):
            raise ValueError("Executed FACE turn differs from Gemma's raw turn delta")
    else:
        if arguments:
            raise ValueError("Gemma STOP must not contain primitive arguments")
        if value.get("derived_world_waypoint_xy_m") is not None:
            raise ValueError("STOP must not claim a derived world waypoint")
    accepted = value.get("accepted")
    executed = value.get("executed")
    if type(accepted) is not bool or type(executed) is not bool:
        raise TypeError("Gemma decision status must be boolean")
    raw_error = value.get("error_code")
    if raw_error is not None and (
        not isinstance(raw_error, str) or _PROTOCOL_CODE.fullmatch(raw_error) is None
    ):
        raise ValueError("Gemma rejection has an invalid protocol code")
    if accepted and (not executed or raw_error is not None):
        raise ValueError("Accepted Gemma decision has inconsistent status")
    for name, expected in (
        ("actual_gemma_causal_forward", True),
        ("model_selected_every_waypoint_and_heading", True),
        ("deterministic_route_planner_used", False),
        ("substitution_applied", False),
        ("synthetic_stop_applied", False),
    ):
        if type(value.get(name)) is not bool or value.get(name) is not expected:
            raise ValueError(f"Gemma decision attestation failed: {name}")
    return GemmaDecisionReceipt(
        step=expected_step,
        model_action=action,
        action_logits=logits,
        action_probabilities=probabilities,
        waypoint_delta_robot_m=delta,
        turn_delta_degrees=turn_delta,
        desired_heading_degrees=derived_heading,
        scene_token_count=scene_token_count,
        robot_token_count=robot_token_count,
        history_token_count=history_token_count,
        prompt_token_count=prompt_token_count,
        decision_position=decision_position,
        decision_tensor_sha256=decision_tensor_sha256,
        active_prefix_sha256=active_prefix_sha256,
        checkpoint_sha256=checkpoint_sha256,
        primitive_tool=primitive,
        target_xy_m=target,
        turn_degrees=turn,
        accepted=accepted,
        executed=executed,
        error_code=raw_error,
    )


def _model_decision_event(value: GemmaDecisionReceipt) -> str:
    right, forward = value.waypoint_delta_robot_m
    if value.primitive_tool == "move_to":
        assert value.target_xy_m is not None
        target = value.target_xy_m
        exact = (
            f"Gemma raw waypoint · robot-frame Δ right {right:+.3f} m, "
            f"forward {forward:+.3f} m · deterministic frame transform → "
            f"world x={target[0]:+.3f}, y={target[1]:+.3f} m"
        )
    elif value.primitive_tool == "turn":
        assert value.turn_degrees is not None
        exact = (
            f"Gemma raw turn Δ {value.turn_delta_degrees:+.3f}° · "
            f"deterministic absolute facing {value.desired_heading_degrees:+.3f}° · "
            "executed exact raw Δ"
        )
    else:
        exact = "Gemma raw action · model-selected STOP"
    if value.accepted and value.executed:
        status = "ACCEPTED · EXECUTED"
    elif value.executed:
        status = "REJECTED AFTER EXECUTION"
    else:
        status = "REJECTED · NOT EXECUTED"
    error = "" if value.error_code is None else f" · {value.error_code}"
    probabilities = " / ".join(f"{item:.3f}" for item in value.action_probabilities)
    logits = ", ".join(f"{item:.3f}" for item in value.action_logits)
    return (
        f"Gemma step {value.step:03d} · {value.model_action.upper()} · "
        f"{exact} · {status}{error}\n"
        f"causal context {value.scene_token_count} scene + "
        f"{value.robot_token_count} robot + {value.history_token_count} history + "
        f"{value.prompt_token_count} prompt tokens · decision position "
        f"{value.decision_position}\n"
        f"p(move/face/stop) {probabilities} · raw logits [{logits}] · "
        f"output {value.decision_tensor_sha256[:12]}… · active prefix "
        f"{value.active_prefix_sha256[:12]}… · checkpoint "
        f"{value.checkpoint_sha256[:12]}…"
    )


def parse_rover_response(payload: Mapping[str, Any]) -> RoverResponse:
    """Validate the browser API's compact public response."""

    if not isinstance(payload, Mapping):
        raise TypeError("The rover response must be a JSON object")
    state = payload.get("state")
    if not isinstance(state, Mapping):
        raise TypeError("The rover response has no numeric state")
    scene_id = state.get("scene_id")
    if not isinstance(scene_id, str) or _OPAQUE_SCENE_ID.fullmatch(scene_id) is None:
        raise ValueError("The rover response has an invalid opaque scene ID")
    position = state.get("position_xy_m")
    if (
        not isinstance(position, Sequence)
        or isinstance(position, (str, bytes))
        or len(position) != 2
    ):
        raise ValueError("The rover response has no two-dimensional position")

    def integer(name: str) -> int:
        value = _finite(state.get(name), name=name)
        if not value.is_integer() or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
        return int(value)

    pose = RoverPose(
        scene_id=scene_id,
        x_m=_finite(position[0], name="position_xy_m[0]"),
        y_m=_finite(position[1], name="position_xy_m[1]"),
        body_yaw_degrees=_finite(
            state.get("body_yaw_degrees"), name="body_yaw_degrees"
        ),
        camera_yaw_degrees=_finite(
            state.get("camera_yaw_degrees"), name="camera_yaw_degrees"
        ),
        pitch_degrees=_finite(state.get("pitch_degrees"), name="pitch_degrees"),
        collision=bool(state.get("collision", False)),
        stopped=bool(state.get("stopped", False)),
        scene_version=integer("scene_version"),
        map_version=integer("map_version"),
        scan_count=integer("scan_count"),
    )
    reply = payload.get("reply", "")
    if not isinstance(reply, str):
        raise TypeError("The rover reply must be text")
    control = payload.get("control", {})
    if not isinstance(control, Mapping):
        raise TypeError("The rover control receipt must be an object")
    mode = control.get("control_mode", "local")
    if not isinstance(mode, str) or len(mode) > 64:
        raise ValueError("The rover control mode is invalid")
    if control.get("navigation_control_mode") != _MODEL_ONLY_NAVIGATION_MODE:
        raise ValueError("The rover is not using the model-only navigation backend")
    required_control = {
        "model_selects_every_waypoint_and_heading": True,
        "model_selects_stop": True,
        "deterministic_route_planner_used": False,
        "fallback_used": False,
        "substitution_applied": False,
        "synthetic_stop_applied": False,
    }
    if any(control.get(name) is not expected for name, expected in required_control.items()):
        raise ValueError("The rover model-only movement attestation failed")
    raw_actions = payload.get("actions", ())
    if not isinstance(raw_actions, Sequence) or isinstance(raw_actions, (str, bytes)):
        raise TypeError("The rover action trajectory must be a sequence")
    if len(raw_actions) > _MAX_ACTIONS_PER_TURN:
        raise ValueError("The rover action trajectory is too long")
    trajectory = tuple(
        action_pose
        for raw_action in raw_actions
        if (action_pose := _action_pose(raw_action, pose)) is not None
    )
    if not trajectory or trajectory[-1] != pose:
        trajectory = (*trajectory, pose)
    action_events = tuple(
        event
        for raw_action in raw_actions
        if (event := _action_event(raw_action)) is not None
    )
    raw_decisions = payload.get("model_decisions", ())
    if not isinstance(raw_decisions, Sequence) or isinstance(
        raw_decisions, (str, bytes)
    ):
        raise TypeError("The Gemma model decision log must be a sequence")
    if len(raw_decisions) > _MAX_MODEL_DECISIONS_PER_TURN:
        raise ValueError("The Gemma model decision log is too long")
    decisions = tuple(
        _model_decision(value, expected_step=index)
        for index, value in enumerate(raw_decisions, start=1)
    )
    # Actual model decisions supersede generic movement summaries in the
    # dialogue.  The complete numeric action trajectory remains available for
    # animation without duplicating every step in the transcript.
    events = (
        tuple(_model_decision_event(value) for value in decisions)
        if decisions
        else action_events
    )
    return RoverResponse(
        pose=pose,
        reply=reply[:12_000],
        control_mode=mode,
        trajectory=trajectory,
        events=events,
        decisions=decisions,
        scene_memory=_memory_diagnostics(payload, pose),
    )


def shortest_yaw_delta(start_degrees: float, target_degrees: float) -> float:
    """Return the signed shortest rotation from start to target."""

    start = _finite(start_degrees, name="start_degrees")
    target = _finite(target_degrees, name="target_degrees")
    return (target - start + 180.0) % 360.0 - 180.0


def interpolate_pose(
    start_xy: Sequence[float],
    target_xy: Sequence[float],
    start_yaw_degrees: float,
    target_yaw_degrees: float,
    fraction: float,
) -> tuple[float, float, float]:
    """Smoothly interpolate a visible Blender pose over the shortest yaw arc."""

    if len(start_xy) != 2 or len(target_xy) != 2:
        raise ValueError("Pose positions must contain exactly two values")
    amount = min(1.0, max(0.0, _finite(fraction, name="fraction")))
    smooth = amount * amount * (3.0 - 2.0 * amount)
    start_x = _finite(start_xy[0], name="start_xy[0]")
    start_y = _finite(start_xy[1], name="start_xy[1]")
    target_x = _finite(target_xy[0], name="target_xy[0]")
    target_y = _finite(target_xy[1], name="target_xy[1]")
    yaw = _finite(start_yaw_degrees, name="start_yaw_degrees") + smooth * shortest_yaw_delta(
        start_yaw_degrees, target_yaw_degrees
    )
    return (
        start_x + smooth * (target_x - start_x),
        start_y + smooth * (target_y - start_y),
        yaw,
    )


class LoopbackRoverClient:
    """Synchronous standard-library client intended for a Blender worker thread."""

    def __init__(self, backend_url: str, *, timeout_seconds: float = 180.0) -> None:
        self.backend_url = normalize_loopback_url(backend_url)
        self.timeout_seconds = _finite(timeout_seconds, name="timeout_seconds")
        if not 0.1 <= self.timeout_seconds <= 600.0:
            raise ValueError("timeout_seconds must be in [0.1, 600]")

    def _request(self, path: str, payload: Mapping[str, Any] | None = None) -> RoverResponse:
        body = None
        method = "GET"
        headers = {"Accept": "application/json", "Connection": "close"}
        if payload is not None:
            body = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode("utf-8")
            method = "POST"
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            f"{self.backend_url}{path}", data=body, headers=headers, method=method
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                raw = response.read(_MAX_RESPONSE_BYTES + 1)
                media_type = response.headers.get_content_type()
        except urllib.error.HTTPError as exc:
            detail = exc.read(512).decode("utf-8", errors="replace")
            raise RuntimeError(f"Rover backend rejected the request ({exc.code}): {detail}") from exc
        except urllib.error.URLError as exc:
            raise ConnectionError(f"Cannot reach local rover backend: {exc.reason}") from exc
        if len(raw) > _MAX_RESPONSE_BYTES:
            raise RuntimeError("Rover backend response is too large")
        if media_type != "application/json":
            raise RuntimeError("Rover backend returned a non-JSON response")
        try:
            decoded = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("Rover backend returned invalid JSON") from exc
        return parse_rover_response(decoded)

    def state(self) -> RoverResponse:
        return self._request("/api/state")

    def instruct(self, text: str) -> RoverResponse:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Enter a rover instruction first")
        normalized = text.strip()
        if len(normalized) > _MAX_INSTRUCTION_CHARACTERS:
            raise ValueError("The rover instruction is too long")
        return self._request("/api/instruction", {"instruction": normalized})

__all__ = [
    "GemmaDecisionReceipt",
    "LoopbackRoverClient",
    "RoverPose",
    "RoverResponse",
    "SceneMemoryDiagnostics",
    "interpolate_pose",
    "normalize_loopback_url",
    "parse_rover_response",
    "shortest_yaw_delta",
]
