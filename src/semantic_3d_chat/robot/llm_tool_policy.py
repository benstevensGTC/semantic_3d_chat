"""Fail-closed local-Gemma selection for the bounded numeric robot tools.

The model receives the user's instruction as text and the environment only as
the already-built continuous scene plus robot-state prefix.  Model output is a
proposal, never an executable command: an exact JSON parser, the configured
tool schemas, and dynamic pose limits all have to accept it before dispatch.

This module is an inference seam, not evidence of a trained action policy.  The
current adapter has not been trained on tool traces; callers must report that
fact and choose an explicit failure fallback.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import torch

from semantic_3d_chat.language.local_lm import prompt_token_ids
from semantic_3d_chat.language.prefix_injection import (
    prefix_sha256,
    scene_boundary_mode_setting,
    scene_prefix_after_bos_setting,
)
from semantic_3d_chat.robot.tools import TOOL_ARGUMENTS, tool_schemas

FallbackPolicy = Literal["fail_closed", "deterministic_parser"]

_SHA256 = re.compile(r"[0-9a-f]{64}")
_SCENE_ID = re.compile(r"scene_[0-9]{6}")
_MAX_MODEL_OUTPUT_BYTES = 4096
_MAX_INSTRUCTION_CHARACTERS = 4096
_MAX_GENERATION_TOKENS = 256


class ToolProposalBackend(Protocol):
    """Text-generation backend which must attest its continuous context."""

    def generate(
        self,
        instruction: str,
        *,
        correction_code: str | None,
    ) -> GeneratedToolProposal: ...


@dataclass(frozen=True)
class GeneratedToolProposal:
    text: str
    active_prefix_sha256: str
    scene_prefix_sha256: str
    robot_tokens_sha256: str | None
    local_inference: bool
    used_continuous_scene_prefix: bool
    used_continuous_robot_tokens: bool
    training_status: str = "untrained_tool_selection_seam"


@dataclass(frozen=True)
class ValidatedToolCall:
    """A tool envelope accepted by both static and dynamic validation."""

    name: str
    arguments: dict[str, Any]
    canonical_json: str
    call_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {"tool": self.name, "arguments": dict(self.arguments)}


@dataclass(frozen=True)
class ToolCallValidation:
    call: ValidatedToolCall | None
    error_code: str | None

    @property
    def valid(self) -> bool:
        return self.call is not None and self.error_code is None


@dataclass(frozen=True)
class ToolPolicyDecision:
    call: ValidatedToolCall | None
    attempts: int
    validation_errors: tuple[str, ...]
    proposal_sha256: tuple[str, ...]
    active_prefix_sha256: str | None
    scene_prefix_sha256: str | None
    robot_tokens_sha256: str | None
    fallback_policy: FallbackPolicy
    training_status: str = "untrained_tool_selection_seam"

    @property
    def accepted(self) -> bool:
        return self.call is not None

    def audit_payload(self) -> dict[str, Any]:
        return {
            "schema": "semantic_3d_chat.local_gemma_tool_decision.v1",
            "accepted": self.accepted,
            "call": None if self.call is None else self.call.as_dict(),
            "call_sha256": None if self.call is None else self.call.call_sha256,
            "attempts": self.attempts,
            "retries": max(0, self.attempts - 1),
            "validation_errors": list(self.validation_errors),
            "proposal_sha256": list(self.proposal_sha256),
            "active_prefix_sha256": self.active_prefix_sha256,
            "scene_prefix_sha256": self.scene_prefix_sha256,
            "robot_tokens_sha256": self.robot_tokens_sha256,
            "used_continuous_scene_prefix": self.scene_prefix_sha256 is not None,
            "used_continuous_robot_tokens": self.robot_tokens_sha256 is not None,
            "local_inference": True,
            "training_status": self.training_status,
            "fallback_policy": self.fallback_policy,
            "raw_model_output_logged": False,
            "environmental_text_inputs": [],
        }


class _DuplicateKey(ValueError):
    pass


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKey(key)
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON constant is forbidden: {value}")


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def tool_protocol_system_prompt(config: Mapping[str, Any]) -> str:
    """Build the environment-free, deterministic tool-selection instruction."""

    schemas = tool_schemas(dict(config))
    return (
        "Select exactly one bounded robot action. The environment and robot state are "
        "available only in the continuous prefix before this prompt. Never describe the "
        "environment. Return exactly one JSON object with keys tool and arguments, with "
        "no Markdown, prose, or extra keys. The only permitted tools and argument schemas "
        f"are: {_canonical_json(schemas)}"
    )


def tool_protocol_sha256(config: Mapping[str, Any]) -> str:
    return hashlib.sha256(tool_protocol_system_prompt(config).encode("utf-8")).hexdigest()


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _validate_schema_value(value: object, schema: Mapping[str, Any]) -> str | None:
    kind = schema.get("type")
    if kind == "number":
        numeric = _number(value)
        if numeric is None:
            return "E_NUMERIC"
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        if minimum is not None and numeric < float(minimum):
            return "E_LIMIT"
        if maximum is not None and numeric > float(maximum):
            return "E_LIMIT"
        return None
    if kind == "integer":
        if isinstance(value, bool) or not isinstance(value, int):
            return "E_NUMERIC"
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        if minimum is not None and value < int(minimum):
            return "E_LIMIT"
        if maximum is not None and value > int(maximum):
            return "E_LIMIT"
        return None
    if kind == "string":
        if not isinstance(value, str):
            return "E_SCHEMA"
        pattern = schema.get("pattern")
        if isinstance(pattern, str) and re.fullmatch(pattern, value) is None:
            return "E_SCENE_ID"
        return None
    return "E_SCHEMA"


def _dynamic_limit_error(
    name: str,
    arguments: Mapping[str, Any],
    config: Mapping[str, Any],
    robot_state: Mapping[str, Any] | None,
) -> str | None:
    if robot_state is None:
        return None
    robot = config.get("robot")
    if not isinstance(robot, Mapping):
        return "E_CONFIG"
    if name == "look":
        body = _number(robot_state.get("body_yaw_degrees"))
        camera = _number(robot_state.get("camera_yaw_degrees"))
        pitch = _number(robot_state.get("pitch_degrees"))
        if body is None or camera is None or pitch is None:
            return "E_STATE"
        offset = (camera - body + 180.0) % 360.0 - 180.0
        next_offset = offset + float(arguments["yaw_delta_degrees"])
        next_pitch = pitch + float(arguments["pitch_delta_degrees"])
        if abs(next_offset) > float(robot.get("max_camera_yaw_offset_degrees", 90.0)):
            return "E_LIMIT"
        if abs(next_pitch) > float(robot.get("max_pitch_degrees", 45.0)):
            return "E_LIMIT"
    elif name == "move_to":
        position = robot_state.get("position_m")
        if (
            not isinstance(position, (list, tuple))
            or len(position) < 2
            or _number(position[0]) is None
            or _number(position[1]) is None
        ):
            return "E_STATE"
        target_x = float(arguments["x"])
        target_y = float(arguments["y"])
        distance = math.hypot(target_x - float(position[0]), target_y - float(position[1]))
        if distance > float(robot.get("max_move_to_m", 1.0)):
            return "E_LIMIT"
        scene = config.get("scene")
        room = scene.get("room_size_m") if isinstance(scene, Mapping) else None
        if isinstance(room, (list, tuple)) and len(room) >= 2:
            radius = float(robot.get("radius_m", 0.0))
            if (
                abs(target_x) > float(room[0]) / 2.0 - radius
                or abs(target_y) > float(room[1]) / 2.0 - radius
            ):
                return "E_LIMIT"
    elif name == "reset_scene":
        current = robot_state.get("scene_id")
        if isinstance(current, str) and arguments["scene_id"] != current:
            return "E_SCENE_UNAVAILABLE"
    return None


def validate_tool_call_text(
    text: object,
    config: Mapping[str, Any],
    *,
    robot_state: Mapping[str, Any] | None = None,
) -> ToolCallValidation:
    """Parse one complete JSON value and validate it before any tool executes."""

    if not isinstance(text, str) or not text.strip():
        return ToolCallValidation(None, "E_EMPTY")
    if len(text.encode("utf-8")) > _MAX_MODEL_OUTPUT_BYTES:
        return ToolCallValidation(None, "E_OUTPUT_SIZE")
    try:
        payload = json.loads(
            text,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except _DuplicateKey:
        return ToolCallValidation(None, "E_DUPLICATE_KEY")
    except (TypeError, ValueError, json.JSONDecodeError):
        return ToolCallValidation(None, "E_JSON")
    if not isinstance(payload, dict) or set(payload) != {"tool", "arguments"}:
        return ToolCallValidation(None, "E_ENVELOPE")
    name = payload["tool"]
    arguments = payload["arguments"]
    if not isinstance(name, str) or name not in TOOL_ARGUMENTS:
        return ToolCallValidation(None, "E_TOOL")
    if not isinstance(arguments, dict):
        return ToolCallValidation(None, "E_ARGUMENTS")
    schemas = {entry["name"]: entry["inputSchema"] for entry in tool_schemas(dict(config))}
    schema = schemas[name]
    properties = schema["properties"]
    if set(arguments) != set(properties):
        return ToolCallValidation(None, "E_SCHEMA")
    for argument_name, argument_schema in properties.items():
        error = _validate_schema_value(arguments[argument_name], argument_schema)
        if error is not None:
            return ToolCallValidation(None, error)
    dynamic_error = _dynamic_limit_error(name, arguments, config, robot_state)
    if dynamic_error is not None:
        return ToolCallValidation(None, dynamic_error)
    canonical = _canonical_json({"tool": name, "arguments": arguments})
    call = ValidatedToolCall(
        name=name,
        arguments=dict(arguments),
        canonical_json=canonical,
        call_sha256=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    )
    return ToolCallValidation(call, None)


class ContinuousPrefixGemmaToolBackend:
    """Generate a JSON proposal with the loaded local Gemma causal decoder."""

    def __init__(
        self,
        runtime: Any,
        config: Mapping[str, Any],
        *,
        max_new_tokens: int = 96,
    ) -> None:
        if (
            isinstance(max_new_tokens, bool)
            or not isinstance(max_new_tokens, int)
            or not 1 <= max_new_tokens <= _MAX_GENERATION_TOKENS
        ):
            raise ValueError(f"max_new_tokens must be in [1, {_MAX_GENERATION_TOKENS}]")
        if not callable(getattr(runtime, "active_prefix_snapshot", None)):
            raise TypeError("Runtime lacks the continuous action-prefix snapshot interface")
        prefix_refresher = getattr(runtime, "prefix_refresher", None)
        wrapped = getattr(prefix_refresher, "runtime", None)
        base = getattr(wrapped, "base", wrapped)
        if base is None or getattr(base, "language", None) is None:
            raise TypeError("Runtime lacks a loaded local causal language backend")
        if getattr(base.language, "backend_name", None) != "gemma4":
            raise ValueError("Continuous tool selection requires the local Gemma 4 backend")
        self.runtime = runtime
        self.base = base
        self.system_prompt = tool_protocol_system_prompt(config)
        self.max_new_tokens = max_new_tokens

    def generate(
        self,
        instruction: str,
        *,
        correction_code: str | None,
    ) -> GeneratedToolProposal:
        prefix, binding = self.runtime.active_prefix_snapshot()
        if not isinstance(prefix, torch.Tensor) or prefix.ndim != 3:
            raise RuntimeError("Runtime returned an invalid continuous action prefix")
        if not torch.isfinite(prefix).all():
            raise RuntimeError("Continuous action prefix contains NaN or infinity")
        observed_hash = prefix_sha256(prefix)
        if binding.get("active_prefix_sha256") != observed_hash:
            raise RuntimeError("Continuous action prefix differs from its runtime binding")
        retry_note = (
            ""
            if correction_code is None
            else (
                "\nThe previous proposal was not executed because validation returned "
                f"{correction_code}. Produce a fresh valid JSON object."
            )
        )
        prompt_ids = prompt_token_ids(
            self.base.language.tokenizer,
            self.system_prompt,
            instruction + retry_note,
            self.base.language.device,
        )
        with torch.inference_mode():
            generated = self.base.language.generate_from_scene_prefix(
                prefix,
                prompt_ids,
                max_new_tokens=self.max_new_tokens,
                eos_token_ids=self.base._eos_token_ids(),
                scene_prefix_after_bos=scene_prefix_after_bos_setting(self.base.config),
                scene_boundary_mode=scene_boundary_mode_setting(self.base.config),
                fallback=self.base._generation_function,
            )
        if not isinstance(generated, torch.Tensor) or generated.ndim != 2:
            raise RuntimeError("Local Gemma returned invalid generated token IDs")
        decoded = self.base.language.tokenizer.decode(
            generated[0].detach().cpu().tolist(),
            skip_special_tokens=True,
        ).strip()
        scene_hash = binding.get("scene_prefix_sha256")
        robot_hash = binding.get("robot_tokens_sha256")
        return GeneratedToolProposal(
            text=decoded,
            active_prefix_sha256=observed_hash,
            scene_prefix_sha256=scene_hash if isinstance(scene_hash, str) else "",
            robot_tokens_sha256=robot_hash if isinstance(robot_hash, str) else None,
            local_inference=True,
            used_continuous_scene_prefix=True,
            used_continuous_robot_tokens=robot_hash is not None,
        )


class LocalGemmaToolPolicy:
    """Retry-bounded proposal policy with explicit fail-closed behavior."""

    def __init__(
        self,
        backend: ToolProposalBackend,
        config: Mapping[str, Any],
        *,
        robot_state_provider: Callable[[], Mapping[str, Any]],
        max_retries: int = 1,
        fallback_policy: FallbackPolicy = "fail_closed",
        propagate_backend_exceptions: bool = False,
    ) -> None:
        if (
            isinstance(max_retries, bool)
            or not isinstance(max_retries, int)
            or not 0 <= max_retries <= 2
        ):
            raise ValueError("max_retries must be an integer in [0, 2]")
        if fallback_policy not in {"fail_closed", "deterministic_parser"}:
            raise ValueError("Unsupported tool-policy fallback")
        if not callable(robot_state_provider):
            raise TypeError("robot_state_provider must be callable")
        if not isinstance(propagate_backend_exceptions, bool):
            raise TypeError("propagate_backend_exceptions must be a boolean")
        self.backend = backend
        self.config = config
        self.robot_state_provider = robot_state_provider
        self.max_retries = max_retries
        self.fallback_policy = fallback_policy
        self.propagate_backend_exceptions = propagate_backend_exceptions

    @staticmethod
    def _context_error(proposal: GeneratedToolProposal) -> str | None:
        if not isinstance(proposal, GeneratedToolProposal):
            return "E_GENERATION"
        if not isinstance(proposal.text, str):
            return "E_GENERATION"
        hashes = (proposal.active_prefix_sha256, proposal.scene_prefix_sha256)
        if any(not isinstance(value, str) or _SHA256.fullmatch(value) is None for value in hashes):
            return "E_CONTEXT"
        if (
            not isinstance(proposal.robot_tokens_sha256, str)
            or _SHA256.fullmatch(proposal.robot_tokens_sha256) is None
        ):
            return "E_ROBOT_CONTEXT"
        if (
            proposal.local_inference is not True
            or proposal.used_continuous_scene_prefix is not True
            or proposal.used_continuous_robot_tokens is not True
        ):
            return "E_CONTEXT"
        if proposal.training_status not in {
            "untrained_tool_selection_seam",
            "supervised_continuous_navigation_policy_v1",
            "supervised_continuous_semantic_grounded_navigation_policy_v3",
            "supervised_continuous_semantic_clearance_navigation_policy_v4",
            "supervised_continuous_gemma4_tool_decoder_v2",
        }:
            return "E_TRAINING_STATUS"
        return None

    def select(self, instruction: str) -> ToolPolicyDecision:
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("Tool-selection instruction must be non-empty text")
        if len(instruction) > _MAX_INSTRUCTION_CHARACTERS:
            raise ValueError("Tool-selection instruction is too long")
        errors: list[str] = []
        proposal_hashes: list[str] = []
        context: tuple[str, str, str] | None = None
        accepted: ValidatedToolCall | None = None
        attempts = 0
        correction: str | None = None
        training_status: str | None = None
        for _ in range(self.max_retries + 1):
            attempts += 1
            try:
                proposal = self.backend.generate(
                    instruction.strip(),
                    correction_code=correction,
                )
            except Exception:
                if self.propagate_backend_exceptions:
                    raise
                error = "E_GENERATION"
                errors.append(error)
                correction = error
                continue
            text = (
                proposal.text
                if isinstance(proposal, GeneratedToolProposal) and isinstance(proposal.text, str)
                else ""
            )
            proposal_hashes.append(hashlib.sha256(text.encode("utf-8")).hexdigest())
            error = self._context_error(proposal)
            if error is None:
                if training_status is None:
                    training_status = proposal.training_status
                elif training_status != proposal.training_status:
                    error = "E_TRAINING_STATUS_CHANGED"
            if error is None:
                current = (
                    proposal.active_prefix_sha256,
                    proposal.scene_prefix_sha256,
                    proposal.robot_tokens_sha256 or "",
                )
                if context is None:
                    context = current
                elif current != context:
                    error = "E_CONTEXT_CHANGED"
            if error is None:
                try:
                    state = self.robot_state_provider()
                except Exception:  # noqa: BLE001 - every state failure must fail closed
                    error = "E_STATE"
                else:
                    if not isinstance(state, Mapping):
                        error = "E_STATE"
                    else:
                        state_context = (
                            state.get("active_prefix_sha256"),
                            state.get("scene_prefix_sha256"),
                            state.get("robot_tokens_sha256"),
                        )
                        comparable = tuple(
                            value if isinstance(value, str) and _SHA256.fullmatch(value) else None
                            for value in state_context
                        )
                        if any(value is not None for value in comparable) and comparable != current:
                            error = "E_CONTEXT_CHANGED"
                        else:
                            validation = validate_tool_call_text(
                                proposal.text,
                                self.config,
                                robot_state=state,
                            )
                            error = validation.error_code
                            accepted = validation.call
            if error is None and accepted is not None:
                break
            errors.append(error or "E_SCHEMA")
            correction = errors[-1]
            accepted = None
            if error == "E_CONTEXT_CHANGED":
                break
        return ToolPolicyDecision(
            call=accepted,
            attempts=attempts,
            validation_errors=tuple(errors),
            proposal_sha256=tuple(proposal_hashes),
            active_prefix_sha256=None if context is None else context[0],
            scene_prefix_sha256=None if context is None else context[1],
            robot_tokens_sha256=None if context is None else context[2],
            fallback_policy=self.fallback_policy,
            training_status=training_status or "untrained_tool_selection_seam",
        )


def execute_validated_tool_call(
    runtime: Any,
    call: ValidatedToolCall,
    *,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Execute only a validator-issued call against the refreshing runtime."""

    if not isinstance(call, ValidatedToolCall):
        raise TypeError("Only a ValidatedToolCall may reach robot execution")
    expected_canonical = _canonical_json(call.as_dict())
    expected_hash = hashlib.sha256(expected_canonical.encode("utf-8")).hexdigest()
    if call.canonical_json != expected_canonical or call.call_sha256 != expected_hash:
        raise ValueError("Validated tool-call integrity check failed")
    state = runtime.get_robot_state()
    if not isinstance(state, Mapping):
        raise TypeError("Robot state provider returned a non-object response")
    revalidated = validate_tool_call_text(
        call.canonical_json,
        config,
        robot_state=state,
    )
    if (
        revalidated.call is None
        or revalidated.error_code is not None
        or revalidated.call.call_sha256 != call.call_sha256
    ):
        raise ValueError(
            "Validated tool call is malformed, out of bounds, or stale: "
            f"{revalidated.error_code or 'E_INTEGRITY'}"
        )
    arguments = dict(call.arguments)
    if call.name == "get_robot_state":
        result = runtime.get_robot_state()
    elif call.name in {"scan", "stop"}:
        result = getattr(runtime, call.name)()
    elif call.name == "look":
        result = runtime.look(
            arguments["yaw_delta_degrees"],
            arguments["pitch_delta_degrees"],
        )
    elif call.name == "turn":
        result = runtime.turn(arguments["angle_degrees"])
    elif call.name in {"move_forward", "move_backward"}:
        result = getattr(runtime, call.name)(arguments["distance_meters"])
    elif call.name == "move_to":
        result = runtime.move_to(arguments["x"], arguments["y"])
    elif call.name == "reset_scene":
        result = runtime.reset_scene(arguments["scene_id"], arguments["seed"])
    else:  # impossible for a validator-issued object; defense in depth
        raise RuntimeError("Validated tool call contains an unsupported tool")
    if not isinstance(result, Mapping):
        raise TypeError("Robot tool returned a non-object response")
    return dict(result)


__all__ = [
    "ContinuousPrefixGemmaToolBackend",
    "FallbackPolicy",
    "GeneratedToolProposal",
    "LocalGemmaToolPolicy",
    "ToolCallValidation",
    "ToolPolicyDecision",
    "ToolProposalBackend",
    "ValidatedToolCall",
    "execute_validated_tool_call",
    "tool_protocol_sha256",
    "tool_protocol_system_prompt",
    "validate_tool_call_text",
]
