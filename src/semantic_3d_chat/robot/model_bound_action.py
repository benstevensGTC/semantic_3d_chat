"""Model-provenance gate for exact, bounded robot actions.

This module deliberately does not plan.  It binds one model output to the
continuous scene/robot context that produced it and later either executes that
exact action or rejects it.  Validation is not allowed to clamp an argument,
choose a waypoint, reroute a motion, or synthesize a ``stop`` action.

The binder is intended to be called at the model-policy boundary.  Production
integration can then pass only :class:`ModelBoundToolCall` objects to the
executor instead of passing freely constructible JSON or ``ValidatedToolCall``
objects to the simulator.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from semantic_3d_chat.robot.llm_tool_policy import validate_tool_call_text

_SHA256: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{64}")
_DECISION_ID: Final[re.Pattern[str]] = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{7,127}")
_EXECUTABLE_ACTIONS: Final[frozenset[str]] = frozenset(
    {"turn", "move_forward", "move_backward", "move_to", "stop"}
)
_SCHEMA: Final[str] = "semantic_3d_chat.model_bound_tool_call.v1"
_RESULT_SCHEMA: Final[str] = "semantic_3d_chat.model_bound_action_result.v1"


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _require_sha256(value: object, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _provenance_payload(
    *,
    canonical_action_json: str,
    action_sha256: str,
    model_output_sha256: str,
    checkpoint_sha256: str,
    decision_id: str,
    active_prefix_sha256: str,
    scene_prefix_sha256: str,
    robot_tokens_sha256: str,
) -> dict[str, str]:
    return {
        "schema": _SCHEMA,
        "canonical_action_json": canonical_action_json,
        "action_sha256": action_sha256,
        "model_output_sha256": model_output_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "decision_id": decision_id,
        "active_prefix_sha256": active_prefix_sha256,
        "scene_prefix_sha256": scene_prefix_sha256,
        "robot_tokens_sha256": robot_tokens_sha256,
    }


@dataclass(frozen=True, slots=True)
class ModelBoundToolCall:
    """One canonical model action bound to its exact continuous context."""

    canonical_action_json: str
    action_sha256: str
    model_output_sha256: str
    checkpoint_sha256: str
    decision_id: str
    active_prefix_sha256: str
    scene_prefix_sha256: str
    robot_tokens_sha256: str
    provenance_sha256: str
    schema: str = _SCHEMA

    def integrity_error(self) -> str | None:
        """Return a terse integrity error without interpreting the action."""

        if self.schema != _SCHEMA:
            return "E_MODEL_PROVENANCE_SCHEMA"
        if not isinstance(self.canonical_action_json, str):
            return "E_MODEL_PROVENANCE_INTEGRITY"
        if not isinstance(self.decision_id, str) or _DECISION_ID.fullmatch(self.decision_id) is None:
            return "E_MODEL_DECISION_ID"
        digests = (
            self.action_sha256,
            self.model_output_sha256,
            self.checkpoint_sha256,
            self.active_prefix_sha256,
            self.scene_prefix_sha256,
            self.robot_tokens_sha256,
            self.provenance_sha256,
        )
        if any(not isinstance(value, str) or _SHA256.fullmatch(value) is None for value in digests):
            return "E_MODEL_PROVENANCE_INTEGRITY"
        if _sha256_text(self.canonical_action_json) != self.action_sha256:
            return "E_MODEL_ACTION_TAMPERED"
        payload = _provenance_payload(
            canonical_action_json=self.canonical_action_json,
            action_sha256=self.action_sha256,
            model_output_sha256=self.model_output_sha256,
            checkpoint_sha256=self.checkpoint_sha256,
            decision_id=self.decision_id,
            active_prefix_sha256=self.active_prefix_sha256,
            scene_prefix_sha256=self.scene_prefix_sha256,
            robot_tokens_sha256=self.robot_tokens_sha256,
        )
        if _sha256_text(_canonical_json(payload)) != self.provenance_sha256:
            return "E_MODEL_PROVENANCE_TAMPERED"
        return None

    def action(self) -> dict[str, Any]:
        """Decode the integrity-checked canonical action envelope."""

        if self.integrity_error() is not None:
            raise ValueError("Model-bound action failed its provenance integrity check")
        value = json.loads(self.canonical_action_json)
        if not isinstance(value, dict):
            raise TypeError("Canonical model action is not an object")
        return value

    def as_dict(self) -> dict[str, str]:
        return {
            **_provenance_payload(
                canonical_action_json=self.canonical_action_json,
                action_sha256=self.action_sha256,
                model_output_sha256=self.model_output_sha256,
                checkpoint_sha256=self.checkpoint_sha256,
                decision_id=self.decision_id,
                active_prefix_sha256=self.active_prefix_sha256,
                scene_prefix_sha256=self.scene_prefix_sha256,
                robot_tokens_sha256=self.robot_tokens_sha256,
            ),
            "provenance_sha256": self.provenance_sha256,
        }


def bind_model_tool_call(
    model_output: str,
    config: Mapping[str, Any],
    *,
    robot_state: Mapping[str, Any],
    binding: Mapping[str, Any],
    checkpoint_sha256: str,
    decision_id: str,
) -> ModelBoundToolCall:
    """Validate and bind one raw model output without changing its semantics.

    The normal strict tool validator establishes the exact bounded envelope.
    This function only canonicalizes JSON key ordering; it never alters a tool
    name or numeric argument.  Unsupported production actions such as ``scan``
    remain bindable so the executor can explicitly reject and audit them.
    """

    if not isinstance(model_output, str):
        raise TypeError("model_output must be text")
    validation = validate_tool_call_text(model_output, config, robot_state=robot_state)
    if validation.call is None:
        raise ValueError(f"Model tool output is invalid: {validation.error_code or 'E_SCHEMA'}")
    if not isinstance(decision_id, str) or _DECISION_ID.fullmatch(decision_id) is None:
        raise ValueError("decision_id must be an opaque 8-128 character identifier")
    checkpoint = _require_sha256(checkpoint_sha256, name="checkpoint_sha256")
    hashes = {
        name: _require_sha256(binding.get(name), name=name)
        for name in (
            "active_prefix_sha256",
            "scene_prefix_sha256",
            "robot_tokens_sha256",
        )
    }
    canonical = validation.call.canonical_json
    action_digest = _sha256_text(canonical)
    if action_digest != validation.call.call_sha256:
        raise RuntimeError("Strict tool validation returned an inconsistent action hash")
    payload = _provenance_payload(
        canonical_action_json=canonical,
        action_sha256=action_digest,
        model_output_sha256=_sha256_text(model_output),
        checkpoint_sha256=checkpoint,
        decision_id=decision_id,
        active_prefix_sha256=hashes["active_prefix_sha256"],
        scene_prefix_sha256=hashes["scene_prefix_sha256"],
        robot_tokens_sha256=hashes["robot_tokens_sha256"],
    )
    return ModelBoundToolCall(
        canonical_action_json=canonical,
        action_sha256=action_digest,
        model_output_sha256=payload["model_output_sha256"],
        checkpoint_sha256=checkpoint,
        decision_id=decision_id,
        active_prefix_sha256=hashes["active_prefix_sha256"],
        scene_prefix_sha256=hashes["scene_prefix_sha256"],
        robot_tokens_sha256=hashes["robot_tokens_sha256"],
        provenance_sha256=_sha256_text(_canonical_json(payload)),
    )


def _state_xy_yaw(state: Mapping[str, Any]) -> tuple[tuple[float, float], float] | None:
    position = state.get("position_m")
    yaw = state.get("body_yaw_degrees")
    if (
        not isinstance(position, Sequence)
        or isinstance(position, (str, bytes))
        or len(position) < 2
        or isinstance(yaw, bool)
    ):
        return None
    try:
        x, y, angle = float(position[0]), float(position[1]), float(yaw)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in (x, y, angle)):
        return None
    return (x, y), angle


def _target_xy(
    action: Mapping[str, Any],
    state: Mapping[str, Any],
) -> tuple[float, float] | None:
    state_values = _state_xy_yaw(state)
    if state_values is None:
        return None
    (x, y), yaw = state_values
    name = action.get("tool")
    arguments = action.get("arguments")
    if not isinstance(arguments, Mapping):
        return None
    if name == "move_to":
        try:
            return float(arguments["x"]), float(arguments["y"])
        except (KeyError, TypeError, ValueError):
            return None
    if name not in {"move_forward", "move_backward"}:
        return None
    try:
        distance = float(arguments["distance_meters"])
    except (KeyError, TypeError, ValueError):
        return None
    direction = -1.0 if name == "move_backward" else 1.0
    radians = math.radians(yaw)
    return (
        x + direction * distance * -math.sin(radians),
        y + direction * distance * math.cos(radians),
    )


def _collision_map(runtime: Any) -> Any | None:
    simulator = getattr(runtime, "simulator", None)
    candidate = getattr(simulator, "collision_map", None)
    if candidate is None:
        candidate = getattr(runtime, "collision_map", None)
    return candidate if callable(getattr(candidate, "segment_check", None)) else None


def _collision_predicted(
    runtime: Any,
    action: Mapping[str, Any],
    state: Mapping[str, Any],
) -> bool | None:
    target = _target_xy(action, state)
    if target is None:
        return None
    state_values = _state_xy_yaw(state)
    collision_map = _collision_map(runtime)
    if state_values is None or collision_map is None:
        return None
    start, _yaw = state_values
    check = collision_map.segment_check(start, target)
    if isinstance(check, Mapping):
        value = check.get("collision")
    else:
        value = getattr(check, "collision", None)
    return value if isinstance(value, bool) else None


def _result(
    proposal: ModelBoundToolCall | object,
    *,
    success: bool,
    executed: bool,
    error_code: str | None,
    tool: str | None = None,
    arguments: Mapping[str, Any] | None = None,
    receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema": _RESULT_SCHEMA,
        "success": success,
        "executed": executed,
        "error_code": error_code,
        "decision_id": (
            proposal.decision_id if isinstance(proposal, ModelBoundToolCall) else None
        ),
        "action_sha256": (
            proposal.action_sha256 if isinstance(proposal, ModelBoundToolCall) else None
        ),
        "checkpoint_sha256": (
            proposal.checkpoint_sha256 if isinstance(proposal, ModelBoundToolCall) else None
        ),
        "model_tool": tool,
        "model_arguments": None if arguments is None else dict(arguments),
        "executed_tool": tool if executed else None,
        "executed_arguments": dict(arguments) if executed and arguments is not None else None,
        "substitution_applied": False,
        "synthetic_stop_applied": False,
        "numeric_tool_receipt": None if receipt is None else dict(receipt),
    }


class ModelBoundActionExecutor:
    """Execute an exact model action once, or reject it without substitution."""

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        checkpoint_sha256: str,
    ) -> None:
        if not isinstance(config, Mapping):
            raise TypeError("Model-bound executor config must be a mapping")
        self.config = dict(config)
        self.checkpoint_sha256 = _require_sha256(
            checkpoint_sha256,
            name="checkpoint_sha256",
        )
        self._consumed_decision_ids: set[str] = set()
        self._lock = threading.RLock()

    def execute(self, runtime: Any, proposal: ModelBoundToolCall) -> dict[str, Any]:
        """Dispatch only the exact, current, collision-free model proposal."""

        if not isinstance(proposal, ModelBoundToolCall):
            return _result(
                proposal,
                success=False,
                executed=False,
                error_code="E_MODEL_PROVENANCE_TYPE",
            )
        integrity_error = proposal.integrity_error()
        if integrity_error is not None:
            return _result(
                proposal,
                success=False,
                executed=False,
                error_code=integrity_error,
            )
        if proposal.checkpoint_sha256 != self.checkpoint_sha256:
            return _result(
                proposal,
                success=False,
                executed=False,
                error_code="E_MODEL_CHECKPOINT",
            )
        with self._lock:
            if proposal.decision_id in self._consumed_decision_ids:
                return _result(
                    proposal,
                    success=False,
                    executed=False,
                    error_code="E_MODEL_DECISION_REPLAY",
                )
            self._consumed_decision_ids.add(proposal.decision_id)

            binding_method = getattr(runtime, "prefix_binding", None)
            try:
                binding = binding_method() if callable(binding_method) else None
            except Exception:  # noqa: BLE001 - any unavailable binding fails closed
                binding = None
            expected_context = (
                proposal.active_prefix_sha256,
                proposal.scene_prefix_sha256,
                proposal.robot_tokens_sha256,
            )
            observed_context = (
                binding.get("active_prefix_sha256") if isinstance(binding, Mapping) else None,
                binding.get("scene_prefix_sha256") if isinstance(binding, Mapping) else None,
                binding.get("robot_tokens_sha256") if isinstance(binding, Mapping) else None,
            )
            if observed_context != expected_context:
                return _result(
                    proposal,
                    success=False,
                    executed=False,
                    error_code="E_MODEL_CONTEXT_STALE",
                )

            try:
                action = proposal.action()
            except (TypeError, ValueError, json.JSONDecodeError):
                return _result(
                    proposal,
                    success=False,
                    executed=False,
                    error_code="E_MODEL_ACTION_TAMPERED",
                )
            name = action.get("tool")
            arguments = action.get("arguments")
            if not isinstance(name, str) or not isinstance(arguments, Mapping):
                return _result(
                    proposal,
                    success=False,
                    executed=False,
                    error_code="E_MODEL_ACTION_INVALID",
                )
            if name not in _EXECUTABLE_ACTIONS:
                return _result(
                    proposal,
                    success=False,
                    executed=False,
                    error_code="E_MODEL_ACTION_FORBIDDEN",
                    tool=name,
                    arguments=arguments,
                )

            try:
                state = runtime.get_robot_state()
            except Exception:  # noqa: BLE001 - state failures must not reach motion
                state = None
            if not isinstance(state, Mapping):
                return _result(
                    proposal,
                    success=False,
                    executed=False,
                    error_code="E_MODEL_STATE",
                    tool=name,
                    arguments=arguments,
                )
            validation = validate_tool_call_text(
                proposal.canonical_action_json,
                self.config,
                robot_state=state,
            )
            if (
                validation.call is None
                or validation.error_code is not None
                or validation.call.call_sha256 != proposal.action_sha256
                or validation.call.canonical_json != proposal.canonical_action_json
            ):
                return _result(
                    proposal,
                    success=False,
                    executed=False,
                    error_code=validation.error_code or "E_MODEL_ACTION_INVALID",
                    tool=name,
                    arguments=arguments,
                )

            if name in {"move_forward", "move_backward", "move_to"}:
                try:
                    collision = _collision_predicted(runtime, action, state)
                except Exception:  # noqa: BLE001 - geometry failure must fail closed
                    collision = None
                if collision is None:
                    return _result(
                        proposal,
                        success=False,
                        executed=False,
                        error_code="E_MODEL_COLLISION_CHECK",
                        tool=name,
                        arguments=arguments,
                    )
                if collision:
                    return _result(
                        proposal,
                        success=False,
                        executed=False,
                        error_code="E_MODEL_COLLISION",
                        tool=name,
                        arguments=arguments,
                    )

            try:
                if name == "turn":
                    receipt = runtime.turn(arguments["angle_degrees"])
                elif name == "move_forward":
                    receipt = runtime.move_forward(arguments["distance_meters"])
                elif name == "move_backward":
                    receipt = runtime.move_backward(arguments["distance_meters"])
                elif name == "move_to":
                    receipt = runtime.move_to(arguments["x"], arguments["y"])
                else:
                    receipt = runtime.stop()
            except Exception:  # noqa: BLE001 - never replace a failed exact action
                return _result(
                    proposal,
                    success=False,
                    executed=False,
                    error_code="E_MODEL_RUNTIME",
                    tool=name,
                    arguments=arguments,
                )
            if not isinstance(receipt, Mapping):
                return _result(
                    proposal,
                    success=False,
                    executed=True,
                    error_code="E_MODEL_RECEIPT",
                    tool=name,
                    arguments=arguments,
                )
            succeeded = receipt.get("success") is True
            error = None if succeeded else str(receipt.get("error_code") or "E_MODEL_ACTION")
            return _result(
                proposal,
                success=succeeded,
                executed=True,
                error_code=error,
                tool=name,
                arguments=arguments,
                receipt=receipt,
            )


__all__ = [
    "ModelBoundActionExecutor",
    "ModelBoundToolCall",
    "bind_model_tool_call",
]
