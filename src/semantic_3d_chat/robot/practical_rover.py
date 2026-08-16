"""Practical local-Gemma rover backend for the interactive room demo.

The production operator surface has one hard navigation boundary:

* questions use the local Gemma causal decoder and the continuous scene prefix;
* every navigation turn is supplied verbatim to the task-trained Gemma waypoint
  controller together with all fixed scene tokens and numeric robot tokens;
* Gemma chooses every waypoint, absolute heading, retry, and STOP decision;
* deterministic code may only convert those exact numeric decisions into the
  bounded kinematic primitives, validate them, and either execute or reject;
* no route planner, target grounder, patrol macro, action parser, or fallback is
  reachable from the production high-level path;
* control uses the precomputed full-scene map without taking rover-camera
  observations; only numeric robot-state tokens refresh after motion.

No oracle, QA, caption, object inventory, or simulator semantic label is read.
The compact response contract is intentionally convenient for a local web UI.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Self

import torch

from semantic_3d_chat.chat.file_audit import FileAccessAudit
from semantic_3d_chat.chat.question_control_runtime import QuestionControlledChatRuntime
from semantic_3d_chat.chat.runtime_config import (
    effective_runtime_config_sha256,
    load_runtime_config,
)
from semantic_3d_chat.config import PROJECT_ROOT, load_config, project_path
from semantic_3d_chat.language.prefix_injection import prefix_sha256
from semantic_3d_chat.robot.blender_scanner import SanitizedBlenderScanner
from semantic_3d_chat.robot.conversation import (
    ConversationalEmbodiedAgent,
    parse_navigation_instruction,
    should_offer_llm_tool_policy,
)
from semantic_3d_chat.robot.gemma_runtime_binding import (
    checkpoint_fingerprint_sha256,
    control_checkpoint_fingerprint_sha256,
    gemma_runtime_binding_sha256,
    question_controlled_gemma_runtime_binding,
    validate_gemma_runtime_binding,
)
from semantic_3d_chat.robot.gemma_waypoint_runtime import (
    CHECKPOINT_ARCHITECTURE as GEMMA_WAYPOINT_CHECKPOINT_ARCHITECTURE,
)
from semantic_3d_chat.robot.gemma_waypoint_runtime import (
    CHECKPOINT_SCHEMA_V1 as GEMMA_WAYPOINT_CHECKPOINT_SCHEMA_V1,
)
from semantic_3d_chat.robot.gemma_waypoint_runtime import (
    CHECKPOINT_SCHEMA_V2 as GEMMA_WAYPOINT_CHECKPOINT_SCHEMA_V2,
)
from semantic_3d_chat.robot.gemma_waypoint_runtime import (
    GOAL_RESULT_SCHEMA as GEMMA_WAYPOINT_GOAL_RESULT_SCHEMA,
)
from semantic_3d_chat.robot.gemma_waypoint_runtime import (
    HISTORY_PARAMETERIZATION_V1 as GEMMA_WAYPOINT_HISTORY_PARAMETERIZATION_V1,
)
from semantic_3d_chat.robot.gemma_waypoint_runtime import (
    HISTORY_PARAMETERIZATION_V2 as GEMMA_WAYPOINT_HISTORY_PARAMETERIZATION_V2,
)
from semantic_3d_chat.robot.gemma_waypoint_runtime import (
    STEP_RECEIPT_SCHEMA as GEMMA_WAYPOINT_STEP_RECEIPT_SCHEMA,
)
from semantic_3d_chat.robot.gemma_waypoint_runtime import (
    GemmaWaypointClosedLoopController,
    load_gemma_waypoint_policy_checkpoint,
)
from semantic_3d_chat.robot.goal_router import parse_semantic_goal
from semantic_3d_chat.robot.llm_tool_policy import (
    LocalGemmaToolPolicy,
    ToolPolicyDecision,
    ValidatedToolCall,
    execute_validated_tool_call,
    validate_tool_call_text,
)
from semantic_3d_chat.robot.runtime_refresh import build_refreshing_embodied_runtime
from semantic_3d_chat.robot.semantic_agent import GemmaProjectedTextEncoder
from semantic_3d_chat.robot.semantic_between import execute_semantic_between_goal
from semantic_3d_chat.robot.semantic_goal_fallback import execute_grounded_goal_fallback
from semantic_3d_chat.robot.semantic_patrol import execute_numeric_patrol
from semantic_3d_chat.robot.trained_goal_policy import (
    TrainedGoalPolicyBundle,
    execute_trained_goal,
)
from semantic_3d_chat.robot.waypoint_history import (
    HISTORY_FEATURE_DIM_V1 as GEMMA_WAYPOINT_HISTORY_FEATURE_DIM_V1,
)
from semantic_3d_chat.robot.waypoint_history import (
    HISTORY_FEATURE_DIM_V2 as GEMMA_WAYPOINT_HISTORY_FEATURE_DIM_V2,
)

SCHEMA: Final[str] = "semantic_3d_chat.practical_rover.v1"
DEFAULT_CONFIG: Final[str] = "configs/runtime/embodied_live.yaml"
DEFAULT_CONTROL_CONFIG: Final[str] = "configs/runtime/gemma4_v56_question_control.yaml"
DEFAULT_SCENE: Final[str] = "scene_000001"
DEFAULT_BASE_CHECKPOINT: Final[str] = (
    "data_gemma4/runtime/checkpoints/gemma4_v54_release_v1"
)
DEFAULT_CONTROL_CHECKPOINT: Final[str] = (
    "data_gemma4/runtime/checkpoints/gemma4_v75_nll_control_release_v1"
)
DEFAULT_ASSET: Final[str] = "data/runtime_assets/scene_000001/s_000001.blend"
DEFAULT_ROBOT_STATE_CHECKPOINT: Final[str] = (
    "data_gemma4/checkpoints/robot_state_numeric_v1"
)
DEFAULT_NAVIGATION_CHECKPOINT: Final[str] = (
    "data_gemma4/checkpoints/"
    "gemma_waypoint_policy_v2_operator_dagger_v14_runtime_aligned"
)

_SHA256 = re.compile(r"[0-9a-f]{64}")
_SCENE_ID = re.compile(r"scene_[0-9]{6}")
_STATE_REQUEST = re.compile(
    r"^(?:(?:get|show|tell me)(?:\s+the)?\s+)?(?:robot|rover)\s+state[.!?]?\s*$",
    re.IGNORECASE,
)
_GREETING = re.compile(
    r"^(?:hi|hello|hey|hiya|good\s+(?:morning|afternoon|evening))"
    r"(?:\s+(?:there|rover))?[.!?]*\s*$",
    re.IGNORECASE,
)
_NAVIGATION_REQUEST = re.compile(
    r"^(?:(?:please\s+)?(?:can|could|would|will)\s+you(?:\s+please)?\s+|"
    r"please\s+|i\s+want\s+you\s+to\s+)?(?:approach|circle|complete\s+(?:a|one)\s+lap|"
    r"do\s+(?:a|one)\s+lap|drive|explore|face|go|look|make\s+(?:a|one)\s+(?:lap|circuit)|"
    r"move|navigate|orient|park|patrol|scan|stand|stop|take\s+(?:(?:a|one)\s+lap|me\s+to)|"
    r"tour|turn|walk|between)\b",
    re.IGNORECASE,
)
_SIMPLE_TURN = re.compile(
    r"^(?:please\s+)?(?:turn|rotate)\s+(left|right)"
    r"(?:\s+(\d+(?:\.\d+)?)\s*(?:degrees?|deg)?)?[.!]?\s*$",
    re.IGNORECASE,
)
_SIMPLE_MOVE = re.compile(
    r"^(?:please\s+)?(?:move|go|drive)\s+(forward|ahead|backward|back|in reverse)"
    r"(?:\s+(\d+(?:\.\d+)?)\s*(?:meters?|metres?|m)?)?[.!]?\s*$",
    re.IGNORECASE,
)
_SIMPLE_LOOK = re.compile(
    r"^(?:please\s+)?look\s+(left|right|up|down)"
    r"(?:\s+(\d+(?:\.\d+)?)\s*(?:degrees?|deg)?)?[.!]?\s*$",
    re.IGNORECASE,
)
_PROTECTED_COMPONENTS: Final[frozenset[str]] = frozenset(
    {"oracle", "qa", "training", "scorer", "scorer_only", "scorer-only"}
)
_MAX_TEXT_CHARACTERS = 4096
_DEFAULT_RIGHT_ANGLE_DEGREES = 90.0
_DEFAULT_MOVE_METERS = 0.25
_DEFAULT_LOOK_DEGREES = 20.0
_DEFAULT_MAX_PLAN_ACTIONS = 64
_GEMMA_WAYPOINT_MAX_STEPS = 128
_DEFERRED_SCAN_ACTIONS: Final[frozenset[str]] = frozenset(
    {"look", "turn", "move_forward", "move_backward", "move_to"}
)
_SCENE_MEMORY_SCHEMA: Final[str] = "semantic_3d_chat.scene_memory_diagnostics.v1"
_GEMMA_WAYPOINT_HISTORY_CONTRACTS: Final[
    frozenset[tuple[str, int, str]]
] = frozenset(
    {
        (
            GEMMA_WAYPOINT_CHECKPOINT_SCHEMA_V1,
            GEMMA_WAYPOINT_HISTORY_FEATURE_DIM_V1,
            GEMMA_WAYPOINT_HISTORY_PARAMETERIZATION_V1,
        ),
        (
            GEMMA_WAYPOINT_CHECKPOINT_SCHEMA_V2,
            GEMMA_WAYPOINT_HISTORY_FEATURE_DIM_V2,
            GEMMA_WAYPOINT_HISTORY_PARAMETERIZATION_V2,
        ),
    }
)


def _validate_gemma_waypoint_history_contract(
    metadata: Mapping[str, Any],
) -> tuple[str, int, str]:
    """Accept only the two versioned history contracts supported by runtime."""

    contract = (
        metadata.get("schema"),
        metadata.get("history_dim"),
        metadata.get("history_parameterization"),
    )
    if contract not in _GEMMA_WAYPOINT_HISTORY_CONTRACTS:
        raise ValueError(
            "Task-trained Gemma waypoint checkpoint history contract is invalid"
        )
    schema, history_dim, parameterization = contract
    assert isinstance(schema, str)
    assert isinstance(history_dim, int) and not isinstance(history_dim, bool)
    assert isinstance(parameterization, str)
    return schema, history_dim, parameterization


@dataclass(frozen=True)
class _DirectActionPlan:
    """Question-free sequence of individually bounded numeric tool calls."""

    actions: tuple[tuple[str, dict[str, Any]], ...]

    @property
    def command_name(self) -> str:
        return self.actions[0][0] if len(self.actions) == 1 else "compound"


def _rooted(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    rooted = candidate if candidate.is_absolute() else PROJECT_ROOT / candidate
    return Path(os.path.abspath(rooted))


def _safe_input(path: str | Path, *, purpose: str, kind: str) -> Path:
    candidate = _rooted(path)
    current = Path(candidate.anchor)
    for component in candidate.parts[1:]:
        current /= component
        if current.is_symlink():
            raise ValueError(f"{purpose} cannot use a symbolic-link path")
    if _PROTECTED_COMPONENTS.intersection(part.casefold() for part in candidate.parts):
        raise ValueError(f"{purpose} cannot enter protected supervision trees")
    if kind == "file" and not candidate.is_file():
        raise FileNotFoundError(f"{purpose} is unavailable: {candidate}")
    if kind == "directory" and not candidate.is_dir():
        raise FileNotFoundError(f"{purpose} is unavailable: {candidate}")
    if kind not in {"file", "directory"}:
        raise ValueError("kind must be file or directory")
    return candidate


def _runtime_audit() -> FileAccessAudit:
    return FileAccessAudit(
        [
            PROJECT_ROOT / "data" / "oracle",
            PROJECT_ROOT / "data" / "qa",
            PROJECT_ROOT / "data_gemma4" / "training",
            PROJECT_ROOT / "reports" / "gemma4" / "scorer_only",
        ],
        forbidden_component_names=_PROTECTED_COMPONENTS,
        block_forbidden=True,
    )


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    converted = float(value)
    return converted if math.isfinite(converted) else None


def _numeric_state(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return the numeric/protocol state allowlist consumed by the UI."""

    permitted = (
        "scene_id",
        "scene_version",
        "position_m",
        "camera_position_m",
        "body_yaw_degrees",
        "camera_yaw_degrees",
        "pitch_degrees",
        "linear_velocity_xy_m",
        "angular_velocity_degrees",
        "collision",
        "last_movement_delta_m",
        "distance_moved",
        "turn_degrees",
        "scan_coverage",
        "scan_count",
        "visible_voxels",
        "valid_depth_pixels",
        "observation_id",
        "clearance_m",
        "action_count",
        "stopped",
        "success",
        "error_code",
        "map_sha256",
    )
    result = {name: value[name] for name in permitted if name in value}
    # Round-trip through strict JSON so tensors, numpy scalars, NaN, and Inf
    # cannot accidentally reach the browser contract.
    return json.loads(json.dumps(result, sort_keys=True, allow_nan=False))


def _binding(runtime: Any) -> dict[str, Any]:
    value = runtime.prefix_binding()
    if not isinstance(value, Mapping):
        raise TypeError("Rover runtime returned an invalid prefix binding")
    scene_hash = value.get("scene_prefix_sha256")
    active_hash = value.get("active_prefix_sha256")
    if (
        not isinstance(scene_hash, str)
        or _SHA256.fullmatch(scene_hash) is None
        or not isinstance(active_hash, str)
        or _SHA256.fullmatch(active_hash) is None
    ):
        raise RuntimeError("Rover runtime returned invalid continuous-prefix hashes")
    return dict(value)


def _scene_memory_diagnostics(
    runtime: Any,
    config: Mapping[str, Any],
    audit: FileAccessAudit | None,
) -> dict[str, Any]:
    """Attest the exact nontextual full-scene tensor supplied to local Gemma.

    ``active_prefix_snapshot`` is the runtime's serialized copy of the complete
    environment prefix plus optional numeric robot-state tokens.  Those numeric
    tokens are inserted immediately before the immutable scene-end boundary, so
    the original question-independent scene tensor can be reconstructed without
    reading a map path, label, caption, or simulator metadata.  Both hashes are
    checked against the transaction binding before this diagnostic is returned.
    """

    snapshot = getattr(runtime, "active_prefix_snapshot", None)
    if not callable(snapshot):
        raise TypeError("Rover runtime lacks the continuous-prefix snapshot interface")
    active_prefix, raw_binding = snapshot()
    if not isinstance(active_prefix, torch.Tensor) or active_prefix.ndim != 3:
        raise TypeError("Rover runtime returned an invalid active-prefix tensor")
    if active_prefix.shape[0] != 1 or active_prefix.shape[1] < 3 or active_prefix.shape[2] < 1:
        raise ValueError("Rover active prefix has an invalid shape")
    if not torch.is_floating_point(active_prefix) or not torch.isfinite(
        active_prefix.float()
    ).all():
        raise ValueError("Rover active prefix must be finite floating point")
    if not isinstance(raw_binding, Mapping):
        raise TypeError("Rover active-prefix binding is invalid")
    binding = dict(raw_binding)
    active_hash = prefix_sha256(active_prefix)
    if active_hash != binding.get("active_prefix_sha256"):
        raise RuntimeError("Active continuous-prefix tensor differs from its binding")

    scene_encoder = config.get("scene_encoder")
    if not isinstance(scene_encoder, Mapping):
        raise TypeError("Rover configuration lacks scene-encoder settings")
    latent_count = scene_encoder.get("global_latents")
    if isinstance(latent_count, bool) or not isinstance(latent_count, int) or latent_count < 1:
        raise ValueError("scene_encoder.global_latents must be a positive integer")
    scene_token_count = latent_count + 2
    if active_prefix.shape[1] < scene_token_count:
        raise RuntimeError("Active prefix omitted configured full-scene tokens")
    robot_state_token_count = int(active_prefix.shape[1]) - scene_token_count
    scene_prefix = (
        active_prefix
        if robot_state_token_count == 0
        else torch.cat(
            (
                active_prefix[:, : scene_token_count - 1],
                active_prefix[:, -1:],
            ),
            dim=1,
        )
    )
    scene_hash = prefix_sha256(scene_prefix)
    if scene_hash != binding.get("scene_prefix_sha256"):
        raise RuntimeError("Reconstructed full-scene tensor differs from its map binding")

    scene_fp32 = scene_prefix.detach().float()
    active_fp32 = active_prefix.detach().float()
    scene_l2 = float(torch.linalg.vector_norm(scene_fp32).cpu())
    scene_rms = float(torch.sqrt(torch.mean(scene_fp32.square())).cpu())
    active_l2 = float(torch.linalg.vector_norm(active_fp32).cpu())
    if not all(math.isfinite(value) and value > 0.0 for value in (scene_l2, scene_rms, active_l2)):
        raise RuntimeError("Continuous-prefix tensor norms are invalid")

    source_voxels = binding.get("source_voxels")
    processed_voxels = binding.get("processed_voxels")
    map_version = binding.get("map_version")
    for name, value in (
        ("source_voxels", source_voxels),
        ("processed_voxels", processed_voxels),
        ("map_version", map_version),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise RuntimeError(f"Continuous-prefix binding has invalid {name}")
    if source_voxels < 1 or processed_voxels < 1:
        raise RuntimeError("Continuous scene memory cannot be empty")

    wrapped = getattr(getattr(runtime, "prefix_refresher", None), "runtime", None)
    base = getattr(wrapped, "base", wrapped)
    map_data = getattr(base, "map_data", None)
    semantic_feature_dim = getattr(map_data, "feature_dim", None)
    runtime_voxels = getattr(map_data, "voxel_count", None)
    if (
        isinstance(semantic_feature_dim, bool)
        or not isinstance(semantic_feature_dim, int)
        or semantic_feature_dim < 1
    ):
        raise RuntimeError("Loaded semantic map has no positive feature dimension")
    if (
        isinstance(runtime_voxels, bool)
        or not isinstance(runtime_voxels, int)
        or runtime_voxels != processed_voxels
    ):
        raise RuntimeError("Loaded semantic-map voxel count differs from its prefix binding")

    base_metadata = getattr(base, "checkpoint_metadata", None)
    control_metadata = getattr(wrapped, "control_metadata", None)
    if not isinstance(base_metadata, Mapping) or not isinstance(control_metadata, Mapping):
        raise TypeError("Continuous scene runtime lacks loaded checkpoint attestations")
    if base_metadata.get("semantic_dim") != semantic_feature_dim:
        raise RuntimeError("Base checkpoint semantic dimension differs from the loaded map")
    control_training_gate = control_metadata.get("saved_runtime_training_gate_passed") is True
    if not control_training_gate:
        raise RuntimeError("Question-control checkpoint lacks its saved training gate")

    loaded_paths = [] if audit is None else audit.unique_paths
    loaded_inventory_hash = hashlib.sha256(
        "\n".join(loaded_paths).encode("utf-8")
    ).hexdigest()
    forbidden_accesses = [] if audit is None else audit.forbidden_accesses()
    return {
        "schema": _SCENE_MEMORY_SCHEMA,
        "tensor_shape": [int(value) for value in scene_prefix.shape],
        "sha256": scene_hash,
        "l2_norm": scene_l2,
        "rms": scene_rms,
        "token_count": scene_token_count,
        "model_dim": int(scene_prefix.shape[2]),
        "active_tensor_shape": [int(value) for value in active_prefix.shape],
        "active_sha256": active_hash,
        "active_l2_norm": active_l2,
        "robot_state_token_count": robot_state_token_count,
        "map_version": map_version,
        "source_voxels": source_voxels,
        "processed_voxels": processed_voxels,
        "semantic_feature_dim": semantic_feature_dim,
        "all_runtime_voxels_encoded": runtime_voxels == processed_voxels,
        "base_adapter_weights_loaded": True,
        "control_weights_loaded": True,
        "control_training_gate_passed": control_training_gate,
        "question_dependent_scene_retrieval": False,
        "loaded_file_audit": {
            "enabled": audit is not None,
            "loaded_file_count": len(loaded_paths),
            "loaded_file_inventory_sha256": loaded_inventory_hash,
            "forbidden_access_count": len(forbidden_accesses),
            "passed": not forbidden_accesses,
        },
        "environmental_text_inputs": [],
    }


def _simple_tool_call(text: str, config: Mapping[str, Any]) -> tuple[str, dict[str, Any]] | None:
    """Parse friendly label-free shorthands into conservative bounded calls."""

    robot = config.get("robot")
    if not isinstance(robot, Mapping):
        raise TypeError("Rover configuration has no robot settings")
    if match := _SIMPLE_TURN.fullmatch(text):
        maximum = float(robot["max_turn_degrees"])
        magnitude = min(float(match.group(2) or 30.0), maximum)
        return "turn", {
            "angle_degrees": magnitude
            if match.group(1).casefold() == "left"
            else -magnitude
        }
    if match := _SIMPLE_MOVE.fullmatch(text):
        maximum = float(robot["max_move_m"])
        magnitude = min(float(match.group(2) or 0.25), maximum)
        backwards = match.group(1).casefold() in {"backward", "back", "in reverse"}
        return (
            "move_backward" if backwards else "move_forward",
            {"distance_meters": magnitude},
        )
    if match := _SIMPLE_LOOK.fullmatch(text):
        maximum = float(robot.get("max_look_delta_degrees", 45.0))
        magnitude = min(float(match.group(2) or 20.0), maximum)
        direction = match.group(1).casefold()
        yaw = magnitude if direction == "left" else -magnitude if direction == "right" else 0.0
        pitch = magnitude if direction == "up" else -magnitude if direction == "down" else 0.0
        return "look", {"yaw_delta_degrees": yaw, "pitch_delta_degrees": pitch}
    return None


def _without_polite_wrapper(text: str) -> str:
    """Remove conversational politeness without changing action semantics."""

    value = text.strip()
    prefixes = (
        r"^please\s+(?:can|could|would|will)\s+you(?:\s+please)?\s+",
        r"^(?:can|could|would|will)\s+you(?:\s+please)?\s+",
        r"^please\s+",
    )
    for pattern in prefixes:
        updated = re.sub(pattern, "", value, count=1, flags=re.IGNORECASE)
        if updated != value:
            value = updated
            break
    value = re.sub(r"\s*,?\s+please\s*$", "", value, flags=re.IGNORECASE)
    return re.sub(r"[.!?]+\s*$", "", value).strip()


def _is_navigation_request(text: str) -> bool:
    """Classify action versus question without entering any legacy router."""

    return _NAVIGATION_REQUEST.match(text) is not None


def _assert_model_only_goal_payload(
    result: Mapping[str, Any],
    *,
    instruction: str,
    checkpoint_sha256: str,
) -> None:
    """Fail closed unless a complete goal came only from exact Gemma decisions."""

    steps = result.get("steps")
    if (
        result.get("schema") != GEMMA_WAYPOINT_GOAL_RESULT_SCHEMA
        or type(result.get("success")) is not bool
        or type(result.get("model_stop_emitted")) is not bool
        or result.get("instruction_sha256")
        != hashlib.sha256(instruction.encode("utf-8")).hexdigest()
        or result.get("checkpoint_sha256") != checkpoint_sha256
        or not isinstance(steps, list)
        or result.get("step_count") != len(steps)
        or not 1 <= len(steps) <= _GEMMA_WAYPOINT_MAX_STEPS
    ):
        raise RuntimeError("Production Gemma goal provenance is incomplete")
    for expected_step, item in enumerate(steps, start=1):
        if not isinstance(item, Mapping):
            raise TypeError("Production Gemma goal has a malformed decision log")
        execution = item.get("execution")
        action = item.get("model_action")
        primitive = item.get("primitive_tool")
        if (
            item.get("schema") != GEMMA_WAYPOINT_STEP_RECEIPT_SCHEMA
            or item.get("step") != expected_step
            or action not in {"move_to", "face", "stop"}
            or primitive != {"move_to": "move_to", "face": "turn", "stop": "stop"}.get(
                action
            )
            or not isinstance(execution, Mapping)
            or type(execution.get("success")) is not bool
            or type(execution.get("executed")) is not bool
            or item.get("actual_gemma_causal_forward") is not True
            or item.get("model_selected_every_waypoint_and_heading") is not True
            or item.get("deterministic_route_planner_used") is not False
            or item.get("substitution_applied") is not False
            or item.get("synthetic_stop_applied") is not False
            or execution.get("substitution_applied") is not False
            or execution.get("synthetic_stop_applied") is not False
        ):
            raise RuntimeError("Production Gemma decision provenance is invalid")
    if (
        result.get("deterministic_route_planner_used") is not False
        or result.get("synthetic_stop_applied") is not False
        or result.get("substitution_applied") is not False
    ):
        raise RuntimeError("Production Gemma controller violated its no-substitution contract")
    if result.get("success") is True:
        terminal = steps[-1]
        execution = terminal["execution"]
        if (
            result.get("termination") != "model_stop"
            or result.get("model_stop_emitted") is not True
            or terminal.get("model_action") != "stop"
            or terminal.get("primitive_tool") != "stop"
            or execution.get("success") is not True
            or execution.get("executed") is not True
        ):
            raise RuntimeError(
                "Successful production navigation lacks an executed Gemma STOP"
            )


def _direct_action_clause(text: str) -> tuple[str, float | None] | None:
    """Parse one label-free numeric/body-control clause before chunking."""

    turn = re.fullmatch(
        r"(?:turn|rotate)\s+(left|right)"
        r"(?:\s+(?:by\s+)?(\d+(?:\.\d+)?)\s*(?:degrees?|deg)?)?",
        text,
        flags=re.IGNORECASE,
    )
    if turn:
        magnitude = float(turn.group(2) or _DEFAULT_RIGHT_ANGLE_DEGREES)
        return "turn", magnitude if turn.group(1).casefold() == "left" else -magnitude
    turn = re.fullmatch(
        r"(?:turn|rotate)\s+(?:by\s+)?(\d+(?:\.\d+)?)\s*(?:degrees?|deg)?"
        r"\s+(?:to\s+the\s+)?(left|right)",
        text,
        flags=re.IGNORECASE,
    )
    if turn:
        magnitude = float(turn.group(1))
        return "turn", magnitude if turn.group(2).casefold() == "left" else -magnitude
    move = re.fullmatch(
        r"(?:move|go|drive)\s+(forward|ahead|backward|back|in\s+reverse)"
        r"(?:\s+(?:by\s+)?(\d+(?:\.\d+)?)\s*(?:meters?|metres?|m)?)?",
        text,
        flags=re.IGNORECASE,
    )
    if move:
        backward = move.group(1).casefold() in {"backward", "back", "in reverse"}
        return (
            "move_backward" if backward else "move_forward",
            float(move.group(2) or _DEFAULT_MOVE_METERS),
        )
    look = re.fullmatch(
        r"look\s+(left|right|up|down)"
        r"(?:\s+(?:by\s+)?(\d+(?:\.\d+)?)\s*(?:degrees?|deg)?)?",
        text,
        flags=re.IGNORECASE,
    )
    if look:
        direction = look.group(1).casefold()
        magnitude = float(look.group(2) or _DEFAULT_LOOK_DEGREES)
        if direction in {"right", "down"}:
            magnitude = -magnitude
        return f"look_{direction}", magnitude
    if re.fullmatch(r"(?:scan|scan\s+the\s+room|look\s+around)", text, re.IGNORECASE):
        return "scan", None
    if re.fullmatch(r"stop", text, re.IGNORECASE):
        return "stop", None
    return None


def _bounded_values(
    value: float,
    maximum: float,
    *,
    max_pieces: int = _DEFAULT_MAX_PLAN_ACTIONS,
) -> list[float]:
    if not math.isfinite(value) or not math.isfinite(maximum) or maximum <= 0.0:
        raise ValueError("Numeric action limits must be finite and positive")
    if isinstance(max_pieces, bool) or not isinstance(max_pieces, int) or max_pieces < 1:
        raise ValueError("max_pieces must be a positive integer")
    if value == 0.0:
        return [0.0]
    sign = -1.0 if value < 0.0 else 1.0
    magnitude = abs(value)
    pieces = max(1, math.ceil(magnitude / maximum))
    if pieces > max_pieces:
        raise OverflowError("E_PLAN_LIMIT")
    values = [sign * maximum] * (pieces - 1)
    values.append(sign * (magnitude - maximum * (pieces - 1)))
    return values


def _direct_action_plan(
    text: str,
    config: Mapping[str, Any],
) -> _DirectActionPlan | None:
    """Build an ordered plan whose every call independently satisfies static limits."""

    robot = config.get("robot")
    if not isinstance(robot, Mapping):
        raise TypeError("Rover configuration has no robot settings")
    unwrapped = _without_polite_wrapper(text)
    clauses = re.split(
        r"\s*(?:,\s*)?(?:and\s+then|then|and)\s+",
        unwrapped,
        flags=re.IGNORECASE,
    )
    parsed = [_direct_action_clause(clause.strip()) for clause in clauses]
    if not parsed or any(value is None for value in parsed):
        return None

    configured_limit = robot.get("max_compound_actions", _DEFAULT_MAX_PLAN_ACTIONS)
    if (
        isinstance(configured_limit, bool)
        or not isinstance(configured_limit, int)
        or configured_limit < 1
    ):
        raise ValueError("robot.max_compound_actions must be a positive integer")
    actions: list[tuple[str, dict[str, Any]]] = []
    for parsed_clause in parsed:
        assert parsed_clause is not None
        name, value = parsed_clause
        remaining_slots = configured_limit - len(actions)
        if remaining_slots < 1:
            raise OverflowError("E_PLAN_LIMIT")
        if name == "turn":
            assert value is not None
            actions.extend(
                ("turn", {"angle_degrees": step})
                for step in _bounded_values(
                    value,
                    float(robot["max_turn_degrees"]),
                    max_pieces=remaining_slots,
                )
            )
        elif name in {"move_forward", "move_backward"}:
            assert value is not None
            actions.extend(
                (name, {"distance_meters": step})
                for step in _bounded_values(
                    value,
                    float(robot["max_move_m"]),
                    max_pieces=remaining_slots,
                )
            )
        elif name.startswith("look_"):
            assert value is not None
            maximum = float(robot.get("max_look_delta_degrees", 45.0))
            for step in _bounded_values(
                value,
                maximum,
                max_pieces=remaining_slots,
            ):
                direction = name.removeprefix("look_")
                actions.append(
                    (
                        "look",
                        {
                            "yaw_delta_degrees": step if direction in {"left", "right"} else 0.0,
                            "pitch_delta_degrees": step if direction in {"up", "down"} else 0.0,
                        },
                    )
                )
        else:
            actions.append((name, {}))

    return _DirectActionPlan(tuple(actions))


def _execute_direct_action_plan(
    runtime: Any,
    plan: _DirectActionPlan,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Revalidate every chunk, with one final observation for a motion batch.

    The live refreshing runtime normally scans after every successful motion.
    That remains the contract for one-action calls and every MCP call.  A direct
    compound plan is different: its chunks are an implementation detail used
    to retain collision checks and configured action bounds.  When the runtime
    exposes its serialized auto-scan seam, hold that same runtime lock, execute
    all-body-action chunks with auto-scan temporarily disabled, then restore
    the flag and take one observation at the last reached pose.  The final scan
    also happens after partial progress followed by a collision.

    Plans containing an explicit ``scan`` or ``stop`` retain their literal
    ordering and the ordinary per-call behavior.  A runtime without the
    serialized seam also falls back to ordinary per-call behavior.
    """

    receipts: list[dict[str, Any]] = []
    error_code: str | None = None
    completed_action_count = 0
    changed_pose = False
    final_refresh_performed = False
    final_refresh_succeeded: bool | None = None
    final_refresh_error_code: str | None = None

    runtime_lock = getattr(runtime, "_lock", None)
    auto_scan = getattr(runtime, "auto_scan_after_motion", None)
    lock_acquire = getattr(runtime_lock, "acquire", None)
    lock_release = getattr(runtime_lock, "release", None)
    defer_scan = bool(
        len(plan.actions) > 1
        and auto_scan is True
        and all(name in _DEFERRED_SCAN_ACTIONS for name, _arguments in plan.actions)
        and callable(lock_acquire)
        and callable(lock_release)
    )

    if defer_scan:
        lock_acquire()
    try:
        if defer_scan:
            runtime.auto_scan_after_motion = False
        try:
            for name, arguments in plan.actions:
                candidate = json.dumps(
                    {"tool": name, "arguments": arguments},
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                validation = validate_tool_call_text(
                    candidate,
                    config,
                    robot_state=runtime.get_robot_state(),
                )
                if validation.call is None:
                    error_code = validation.error_code or "E_SCHEMA"
                    break
                receipt = execute_validated_tool_call(
                    runtime,
                    validation.call,
                    config=config,
                )
                receipts.append(receipt)
                if receipt.get("success") is not True:
                    error_code = str(receipt.get("error_code") or "E_ACTION")
                    break
                completed_action_count += 1
                changed_pose = changed_pose or name in _DEFERRED_SCAN_ACTIONS
        finally:
            if defer_scan:
                runtime.auto_scan_after_motion = True

        # Keep the runtime lock held between the final chunk and this
        # observation: no MCP or UI action can interleave at the temporarily
        # unobserved pose.
        if defer_scan and changed_pose:
            refresh_receipt = runtime.scan()
            receipts.append(refresh_receipt)
            final_refresh_performed = True
            final_refresh_succeeded = refresh_receipt.get("success") is True
            if not final_refresh_succeeded:
                final_refresh_error_code = str(
                    refresh_receipt.get("error_code") or "E_SCAN"
                )
                if error_code is None:
                    error_code = final_refresh_error_code
    finally:
        if defer_scan:
            lock_release()

    planned_actions_succeeded = completed_action_count == len(plan.actions)
    return {
        "kind": "navigation",
        "command": plan.command_name,
        "success": planned_actions_succeeded
        and (final_refresh_succeeded is not False),
        "error_code": error_code,
        "action_receipts": receipts,
        "planned_action_count": len(plan.actions),
        "completed_action_count": completed_action_count,
        "final_refresh_performed": final_refresh_performed,
        "final_refresh_succeeded": final_refresh_succeeded,
        "final_refresh_error_code": final_refresh_error_code,
    }


def _parsed_expected_tool(command: Any) -> tuple[str, dict[str, Any]] | None:
    """Return the exact call implied by a non-semantic parsed instruction."""

    if command.kind in {"scan", "stop"}:
        return command.kind, {}
    if command.kind == "turn":
        return "turn", {"angle_degrees": float(command.value)}
    if command.kind in {"move_forward", "move_backward"}:
        return command.kind, {"distance_meters": float(command.value)}
    # A face/approach/between request needs a continuously grounded XYZ target
    # and cannot be verified from a single untrained JSON proposal.
    return None


def _call_matches_expected(
    call: ValidatedToolCall,
    expected: tuple[str, dict[str, Any]],
) -> bool:
    name, arguments = expected
    if call.name != name or set(call.arguments) != set(arguments):
        return False
    for key, expected_value in arguments.items():
        observed = call.arguments[key]
        if isinstance(expected_value, (int, float)) and not isinstance(expected_value, bool):
            numeric = _finite_number(observed)
            if numeric is None or not math.isclose(
                numeric, float(expected_value), rel_tol=0.0, abs_tol=1e-6
            ):
                return False
        elif observed != expected_value:
            return False
    return True


def _reply_for_result(result: Mapping[str, Any], actions: Sequence[Mapping[str, Any]]) -> str:
    if result.get("kind") == "answer":
        answer = result.get("answer")
        return str(answer).strip() if isinstance(answer, str) and answer.strip() else "Unknown."
    if result.get("success") is True:
        command = str(result.get("command", "action")).replace("_", " ")
        if command == "gemma waypoint closed loop":
            step_count = result.get("step_count")
            if isinstance(step_count, int):
                return (
                    f"Gemma chose and executed {step_count} decisions from the "
                    "continuous 3D memory, including its own STOP decision."
                )
            return "Gemma completed the goal and selected STOP."
        if command == "semantic map patrol":
            completed = result.get("completed_action_count")
            plan = result.get("plan")
            distance = plan.get("path_length_m") if isinstance(plan, Mapping) else None
            if isinstance(completed, int) and isinstance(distance, (int, float)):
                return (
                    f"Patrol complete — I followed {completed} internally planned "
                    f"waypoints over {float(distance):.1f} m using the fixed 3D map."
                )
            return "Patrol complete using the fixed continuous 3D map."
        if command == "semantic between":
            return (
                "Goal complete — I reached a collision-free point between the two "
                "regions grounded in the fixed semantic 3D map."
            )
        if command in {"semantic grounded face", "semantic grounded approach"}:
            goal = command.removeprefix("semantic grounded ")
            return (
                f"Goal complete — I used all-voxel semantic grounding and the fixed "
                f"3D geometry to {goal} the requested region."
            )
        if command == "scan":
            return "Scan complete. I refreshed my continuous 3D memory."
        if command in {"get robot state", "get_robot_state", "state"}:
            return "Here is my current numeric state."
        if command == "compound":
            completed = result.get("completed_action_count")
            if isinstance(completed, int):
                return f"Done — I completed {completed} bounded actions in order."
        return f"Done — I completed the bounded {command} action."
    error = result.get("error_code")
    if error is None and actions:
        error = actions[-1].get("error_code")
    if error == "E_COLLISION":
        return "I stopped safely because the requested path was blocked."
    if error == "E_STOPPED":
        return "The rover is stopped. Reset it before requesting more motion."
    if error == "E_TOOL_POLICY_REJECTED":
        return "I could not turn that request into a safe bounded action."
    if error == "E_HIGH_LEVEL_ONLY":
        return (
            "Give me an outcome-level goal, such as patrolling the room, facing "
            "something, or approaching something you name."
        )
    if error == "E_SENSOR_ACTION":
        return "Camera-driven control is disabled; I use the precomputed global 3D memory."
    if error == "E_GOAL_UNSUPPORTED":
        return "That outcome is not supported safely yet; try patrol, face, or approach."
    if error == "E_MAX_STEPS":
        return (
            "Gemma did not select STOP within the bounded decision budget; "
            "no fallback route or synthetic stop was substituted."
        )
    if error == "E_HIGH_LEVEL_ONLY":
        return "Give me an outcome-level goal; I will choose the bounded actions."
    return "I could not complete that action safely."


class PracticalRoverController:
    """Thread-safe UI backend over a loaded local continuous-scene runtime."""

    def __init__(
        self,
        runtime: Any,
        semantic_agent: ConversationalEmbodiedAgent,
        config: Mapping[str, Any],
        *,
        llm_tool_policy: LocalGemmaToolPolicy | None = None,
        trained_goal_policy: TrainedGoalPolicyBundle | None = None,
        gemma_waypoint_controller: GemmaWaypointClosedLoopController | None = None,
        high_level_only: bool = False,
        audit: FileAccessAudit | None = None,
        audit_output: str | Path | None = None,
        initial_scan: bool = False,
    ) -> None:
        if not isinstance(initial_scan, bool):
            raise TypeError("initial_scan must be boolean")
        if not isinstance(high_level_only, bool):
            raise TypeError("high_level_only must be boolean")
        if high_level_only and initial_scan:
            raise ValueError("High-level static-map control forbids an initial scan")
        if high_level_only and llm_tool_policy is not None:
            raise ValueError("High-level control forbids the untrained JSON policy")
        if gemma_waypoint_controller is not None and not high_level_only:
            raise ValueError(
                "Gemma waypoint control requires the exclusive high-level mode"
            )
        if (
            gemma_waypoint_controller is not None
            and trained_goal_policy is not None
        ):
            raise ValueError(
                "Gemma waypoint control cannot coexist with a legacy goal planner"
            )
        if (
            high_level_only
            and trained_goal_policy is None
            and gemma_waypoint_controller is None
        ):
            raise ValueError("High-level control requires a task-trained Gemma policy")
        auto_scan = getattr(runtime, "auto_scan_after_motion", None)
        if high_level_only and auto_scan is not False:
            raise ValueError("High-level control requires auto_scan_after_motion=false")
        self.runtime = runtime
        self.semantic_agent = semantic_agent
        self.config = config
        self.llm_tool_policy = llm_tool_policy
        self.trained_goal_policy = trained_goal_policy
        self.gemma_waypoint_controller = gemma_waypoint_controller
        self._model_only_waypoint_controller = gemma_waypoint_controller
        self.high_level_only = high_level_only
        self.audit = audit
        self.audit_output = None if audit_output is None else _rooted(audit_output)
        self.initial_scan = initial_scan
        self._lock = threading.RLock()
        self._closed = False
        self._started = False
        self._startup_payload: dict[str, Any] | None = None

    def _assert_open(self) -> None:
        if self._closed:
            raise RuntimeError("Rover controller is closed")

    def _history_index(self) -> int | None:
        history = getattr(getattr(self.runtime, "simulator", None), "history", None)
        try:
            return len(history) if history is not None else None
        except TypeError:
            return None

    def _new_history(self, index: int | None) -> list[dict[str, Any]]:
        history = getattr(getattr(self.runtime, "simulator", None), "history", None)
        if index is None or history is None:
            return []
        try:
            values = list(history)[index:]
        except TypeError:
            return []
        return [_numeric_state(value) for value in values if isinstance(value, Mapping)]

    def _navigation_identity(self) -> tuple[str | None, str | None]:
        controller = self._model_only_waypoint_controller
        if controller is None:
            return None, None
        if self.gemma_waypoint_controller is not controller:
            raise RuntimeError("Loaded model-only Gemma controller identity changed")
        checkpoint_sha256 = getattr(controller, "checkpoint_sha256", None)
        metadata = getattr(controller, "metadata", None)
        runtime_binding_sha256 = (
            metadata.get("gemma_runtime_binding_sha256")
            if isinstance(metadata, Mapping)
            else None
        )
        if (
            not isinstance(checkpoint_sha256, str)
            or _SHA256.fullmatch(checkpoint_sha256) is None
            or not isinstance(runtime_binding_sha256, str)
            or _SHA256.fullmatch(runtime_binding_sha256) is None
        ):
            raise RuntimeError("Loaded Gemma waypoint identity is incomplete")
        return checkpoint_sha256, runtime_binding_sha256

    def _envelope(
        self,
        result: Mapping[str, Any],
        *,
        history_index: int | None,
        decision_source: str,
        gemma_attempted: bool,
        gemma_accepted: bool,
        fallback_used: bool,
    ) -> dict[str, Any]:
        binding = _binding(self.runtime)
        navigation_checkpoint_sha256, runtime_binding_sha256 = (
            self._navigation_identity()
        )
        scene_memory = _scene_memory_diagnostics(self.runtime, self.config, self.audit)
        if scene_memory["sha256"] != binding["scene_prefix_sha256"]:
            raise RuntimeError("Scene-memory diagnostic changed during response assembly")
        state = _numeric_state(self.runtime.get_robot_state())
        actions = self._new_history(history_index)
        receipts = result.get("action_receipts")
        if not actions and isinstance(receipts, list):
            actions = [_numeric_state(item) for item in receipts if isinstance(item, Mapping)]
        map_version = state.get("scene_version", binding.get("map_version", 0))
        raw_model_decisions = result.get("steps", [])
        if not isinstance(raw_model_decisions, list) or any(
            not isinstance(item, Mapping) for item in raw_model_decisions
        ):
            raise TypeError("Gemma waypoint decisions must be a list of objects")
        model_decisions = [dict(item) for item in raw_model_decisions]
        return {
            "schema": SCHEMA,
            "ready": True,
            "reply": _reply_for_result(result, actions),
            "state": state,
            "actions": actions,
            "model_decisions": model_decisions,
            "scene_prefix_hash": binding["scene_prefix_sha256"],
            "active_prefix_hash": binding["active_prefix_sha256"],
            "scene_memory": scene_memory,
            "map_version": int(map_version),
            "success": bool(result.get("success", result.get("kind") == "answer")),
            "decision_source": decision_source,
            "control_mode": decision_source,
            "navigation_control_mode": (
                "actual_local_gemma_model_only_waypoint_policy"
                if self.gemma_waypoint_controller is not None
                else "legacy_navigation_backend"
            ),
            "navigation_checkpoint_sha256": navigation_checkpoint_sha256,
            "gemma_runtime_binding_sha256": runtime_binding_sha256,
            "gemma_attempted": gemma_attempted,
            "gemma_accepted": gemma_accepted,
            "fallback_used": fallback_used,
            "local_inference": True,
            "cloud_model_used": False,
            "continuous_scene_memory": True,
            "continuous_robot_state": binding.get("robot_tokens_sha256") is not None,
            "high_level_natural_language_only": self.high_level_only,
            "task_trained_navigation": (
                self.gemma_waypoint_controller is not None
                or self.trained_goal_policy is not None
            ),
            "model_selects_every_waypoint_and_heading": (
                self.gemma_waypoint_controller is not None
            ),
            "model_selects_stop": self.gemma_waypoint_controller is not None,
            "deterministic_route_planner_used": bool(
                result.get("deterministic_route_planner_used", False)
            ),
            "synthetic_stop_applied": bool(
                result.get("synthetic_stop_applied", False)
            ),
            "substitution_applied": bool(
                result.get("substitution_applied", False)
            ),
            "model_stop_emitted": bool(result.get("model_stop_emitted", False)),
            "untrained_json_backend_enabled": self.llm_tool_policy is not None,
            "static_precomputed_scene_memory": not self.initial_scan,
            "camera_control_input": False,
            "initial_scan_performed": self.initial_scan,
            "scene_memory_refreshed": bool(actions) and any(
                item.get("valid_depth_pixels", 0) for item in actions
            ),
            "planned_action_count": result.get("planned_action_count"),
            "completed_action_count": result.get("completed_action_count"),
            "final_refresh_performed": result.get("final_refresh_performed", False),
            "final_refresh_succeeded": result.get("final_refresh_succeeded"),
            "final_refresh_error_code": result.get("final_refresh_error_code"),
            "environmental_text_inputs": [],
            "error_code": result.get("error_code"),
        }

    def startup(self) -> dict[str, Any]:
        with self._lock:
            self._assert_open()
            if self._startup_payload is not None:
                return json.loads(json.dumps(self._startup_payload, allow_nan=False))
            history_index = self._history_index()
            scan_receipt: Mapping[str, Any] | None = None
            if self.initial_scan:
                scan_receipt = self.runtime.scan()
                if not isinstance(scan_receipt, Mapping):
                    raise TypeError("Initial rover scan returned an invalid receipt")
                if scan_receipt.get("success") is not True:
                    raise RuntimeError(
                        "Initial RGB-D scan failed: "
                        f"{scan_receipt.get('error_code', 'E_SCAN')}"
                    )
            binding = _binding(self.runtime)
            navigation_checkpoint_sha256, runtime_binding_sha256 = (
                self._navigation_identity()
            )
            scene_memory = _scene_memory_diagnostics(self.runtime, self.config, self.audit)
            if scene_memory["sha256"] != binding["scene_prefix_sha256"]:
                raise RuntimeError("Scene-memory diagnostic changed during startup")
            state = _numeric_state(self.runtime.get_robot_state())
            actions = self._new_history(history_index)
            if not actions and scan_receipt is not None:
                actions = [_numeric_state(scan_receipt)]
            self._started = True
            self._startup_payload = {
                "schema": SCHEMA,
                "ready": True,
                "reply": (
                    "Local Gemma rover is ready; the initial RGB-D scan is fused."
                    if self.initial_scan
                    else "Local Gemma rover is ready."
                ),
                "state": state,
                "actions": actions,
                "scene_prefix_hash": binding["scene_prefix_sha256"],
                "active_prefix_hash": binding["active_prefix_sha256"],
                "scene_memory": scene_memory,
                "map_version": int(state.get("scene_version", binding.get("map_version", 0))),
                "local_inference": True,
                "cloud_model_used": False,
                "continuous_scene_memory": True,
                "continuous_robot_state": binding.get("robot_tokens_sha256") is not None,
                "action_modes": (
                    [
                        "actual_local_gemma_causal_waypoint_policy",
                        "gemma_selects_every_waypoint_heading_and_stop",
                        "static_precomputed_continuous_3d_memory",
                        "exact_action_conversion_and_rejection_only",
                    ]
                    if self.gemma_waypoint_controller is not None
                    else [
                        "legacy_task_trained_local_gemma_goal_policy",
                        "static_precomputed_continuous_3d_memory",
                    ]
                    if self.trained_goal_policy is not None
                    else [
                        "local_gemma_continuous_semantic_grounding",
                        "deterministic_bounded_safety_fallback",
                    ]
                ),
                "control_mode": (
                    "actual_local_gemma_model_only_waypoint_policy"
                    if self.gemma_waypoint_controller is not None
                    else "high_level_task_trained_local_gemma_static_scene"
                    if self.high_level_only
                    else "local_gemma_with_bounded_safety_fallback"
                ),
                "navigation_control_mode": (
                    "actual_local_gemma_model_only_waypoint_policy"
                    if self.gemma_waypoint_controller is not None
                    else "legacy_navigation_backend"
                ),
                "navigation_checkpoint_sha256": navigation_checkpoint_sha256,
                "gemma_runtime_binding_sha256": runtime_binding_sha256,
                "high_level_natural_language_only": self.high_level_only,
                "task_trained_navigation": (
                    self.gemma_waypoint_controller is not None
                    or self.trained_goal_policy is not None
                ),
                "model_selects_every_waypoint_and_heading": (
                    self.gemma_waypoint_controller is not None
                ),
                "model_selects_stop": self.gemma_waypoint_controller is not None,
                "deterministic_route_planner_used": False,
                "synthetic_stop_applied": False,
                "substitution_applied": False,
                "untrained_json_backend_enabled": self.llm_tool_policy is not None,
                "static_precomputed_scene_memory": not self.initial_scan,
                "camera_control_input": False,
                "gemma_attempted": False,
                "gemma_accepted": False,
                "fallback_used": False,
                "initial_scan_performed": self.initial_scan,
                "scene_memory_refreshed": self.initial_scan,
                "environmental_text_inputs": [],
            }
            return json.loads(json.dumps(self._startup_payload, allow_nan=False))

    def dispatch_tool(self, tool: str, args: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Validate and execute one explicit UI control without model generation."""

        with self._lock:
            self._assert_open()
            if self.high_level_only:
                result = {
                    "kind": "navigation",
                    "command": "high_level_only",
                    "success": False,
                    "error_code": "E_HIGH_LEVEL_ONLY",
                    "action_receipts": [],
                }
                return self._envelope(
                    result,
                    history_index=self._history_index(),
                    decision_source="direct_tool_disabled_high_level_only",
                    gemma_attempted=False,
                    gemma_accepted=False,
                    fallback_used=False,
                )
            arguments = {} if args is None else dict(args)
            candidate = json.dumps(
                {"tool": tool, "arguments": arguments},
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            validation = validate_tool_call_text(
                candidate,
                self.config,
                robot_state=self.runtime.get_robot_state(),
            )
            history_index = self._history_index()
            if validation.call is None:
                result = {
                    "kind": "navigation",
                    "command": str(tool),
                    "success": False,
                    "error_code": validation.error_code or "E_SCHEMA",
                    "action_receipts": [],
                }
            else:
                receipt = execute_validated_tool_call(
                    self.runtime,
                    validation.call,
                    config=self.config,
                )
                result = {
                    "kind": "navigation",
                    "command": validation.call.name,
                    "success": bool(receipt.get("success")),
                    "error_code": receipt.get("error_code"),
                    "action_receipts": [receipt],
                }
            return self._envelope(
                result,
                history_index=history_index,
                decision_source="validated_ui_tool",
                gemma_attempted=False,
                gemma_accepted=False,
                fallback_used=False,
            )

    def _gemma_decision(self, text: str) -> ToolPolicyDecision | None:
        if self.llm_tool_policy is None:
            return None
        return self.llm_tool_policy.select(text)

    def navigate_goal(self, text: str) -> dict[str, Any]:
        """Execute one outcome-level goal through the exclusive Gemma policy.

        This entrypoint exists for non-UI transports such as MCP.  Unlike the
        conversational handler it performs no question/action classification:
        every non-empty input is passed verbatim to the loaded causal waypoint
        controller.  The deterministic runtime can execute or reject Gemma's
        exact FACE/MOVE_TO/STOP proposal, but it cannot plan, substitute, fall
        back, or append a terminal STOP.
        """

        if not isinstance(text, str) or not text.strip():
            raise ValueError("Rover navigation goal must be non-empty text")
        raw_instruction = text.strip()
        if len(raw_instruction) > _MAX_TEXT_CHARACTERS:
            raise ValueError("Rover navigation goal is too long")
        with self._lock:
            self._assert_open()
            model_only_controller = self._model_only_waypoint_controller
            if model_only_controller is None:
                raise RuntimeError(
                    "Goal transport requires the model-only Gemma waypoint controller"
                )
            if self.gemma_waypoint_controller is not model_only_controller:
                raise RuntimeError("Loaded model-only Gemma controller identity changed")
            history_index = self._history_index()
            goal = model_only_controller.run(
                raw_instruction,
                max_steps=_GEMMA_WAYPOINT_MAX_STEPS,
            )
            result = goal.as_dict()
            _assert_model_only_goal_payload(
                result,
                instruction=raw_instruction,
                checkpoint_sha256=model_only_controller.checkpoint_sha256,
            )
            result.update(
                {
                    "kind": "navigation",
                    "command": "gemma_waypoint_closed_loop",
                    "planned_action_count": result.get("step_count"),
                    "completed_action_count": sum(
                        1
                        for item in result.get("steps", [])
                        if isinstance(item, Mapping)
                        and isinstance(item.get("execution"), Mapping)
                        and item["execution"].get("success") is True
                    ),
                    "action_receipts": [],
                }
            )
            return self._envelope(
                result,
                history_index=history_index,
                decision_source="actual_local_gemma_model_only_waypoint_policy",
                gemma_attempted=True,
                gemma_accepted=bool(result.get("success")),
                fallback_used=False,
            )

    def _execute_target_goal(
        self,
        kind: str,
        target_text: str,
    ) -> tuple[dict[str, Any], str, bool]:
        if kind not in {"face", "approach"} or self.trained_goal_policy is None:
            raise ValueError("Target goal requires a trained face/approach policy")
        trained = execute_trained_goal(
            self.runtime,
            self.trained_goal_policy,
            kind=kind,  # type: ignore[arg-type]
            target_text=target_text,
        )
        if trained.get("success") is True:
            return (
                trained,
                "task_trained_local_gemma_full_scene_goal_policy_v3",
                False,
            )
        fallback = execute_grounded_goal_fallback(
            self.runtime,
            self.config,
            kind=kind,  # type: ignore[arg-type]
            target_text=target_text,
            text_encoder=self.semantic_agent.text_encoder,
        )
        fallback["task_trained_policy_attempt"] = {
            name: trained.get(name)
            for name in (
                "schema",
                "success",
                "error_code",
                "termination_reason",
                "step_count",
                "training_status",
                "camera_observations_during_goal",
                "static_scene_prefix_unchanged",
            )
        }
        return (
            fallback,
            "task_trained_policy_then_all_voxel_semantic_geometry_fallback",
            True,
        )

    def handle_instruction(self, text: str) -> dict[str, Any]:
        """Handle one chat turn without allowing a model string to execute directly."""

        if not isinstance(text, str) or not text.strip():
            raise ValueError("Rover instruction must be non-empty text")
        raw_instruction = text.strip()
        normalized = " ".join(text.strip().split())
        if len(normalized) > _MAX_TEXT_CHARACTERS:
            raise ValueError("Rover instruction is too long")
        with self._lock:
            self._assert_open()
            history_index = self._history_index()
            if _GREETING.fullmatch(normalized):
                result = {
                    "kind": "answer",
                    "answer": (
                        "Hi! Give me an outcome-level goal—such as facing a landmark, "
                        "approaching something you name, or patrolling the room."
                    ),
                    "success": True,
                }
                source = "deterministic_non_environmental_greeting"
                gemma_attempted = False
                gemma_accepted = False
                fallback_used = False
            elif _STATE_REQUEST.fullmatch(normalized):
                result: dict[str, Any] = {
                    "kind": "navigation",
                    "command": "state",
                    "success": True,
                    "action_receipts": [],
                }
                source = "numeric_state_query"
                gemma_attempted = False
                gemma_accepted = False
                fallback_used = False
            else:
                # Production high-level navigation is a closed, exclusive
                # branch. Classification decides only whether this is an
                # action turn; it supplies no target, waypoint, heading, route,
                # completion test, or fallback action. The original user text
                # goes directly to actual Gemma, which is rerun after every
                # executed or rejected proposal until Gemma itself emits STOP.
                model_only_controller = self._model_only_waypoint_controller
                if model_only_controller is not None:
                    if self.gemma_waypoint_controller is not model_only_controller:
                        raise RuntimeError(
                            "Loaded model-only Gemma controller identity changed"
                        )
                    if _is_navigation_request(normalized):
                        goal = model_only_controller.run(
                            raw_instruction,
                            max_steps=_GEMMA_WAYPOINT_MAX_STEPS,
                        )
                        result = goal.as_dict()
                        _assert_model_only_goal_payload(
                            result,
                            instruction=raw_instruction,
                            checkpoint_sha256=model_only_controller.checkpoint_sha256,
                        )
                        result.update(
                            {
                                "kind": "navigation",
                                "command": "gemma_waypoint_closed_loop",
                                "planned_action_count": result.get("step_count"),
                                "completed_action_count": sum(
                                    1
                                    for item in result.get("steps", [])
                                    if isinstance(item, Mapping)
                                    and isinstance(item.get("execution"), Mapping)
                                    and item["execution"].get("success") is True
                                ),
                                "action_receipts": [],
                            }
                        )
                        return self._envelope(
                            result,
                            history_index=history_index,
                            decision_source=(
                                "actual_local_gemma_model_only_waypoint_policy"
                            ),
                            gemma_attempted=True,
                            gemma_accepted=bool(result.get("success")),
                            fallback_used=False,
                        )
                    answer_question = getattr(
                        self.semantic_agent, "answer_question", None
                    )
                    if not callable(answer_question):
                        raise RuntimeError(
                            "Production scene-question path lacks an answer-only interface"
                        )
                    result = answer_question(raw_instruction)
                    if result.get("kind") != "answer":
                        raise RuntimeError(
                            "Production scene-question path attempted a robot action"
                        )
                    return self._envelope(
                        result,
                        history_index=history_index,
                        decision_source="local_gemma_continuous_scene_answer",
                        gemma_attempted=True,
                        gemma_accepted=True,
                        fallback_used=False,
                    )
                semantic_goal = parse_semantic_goal(normalized)
                if semantic_goal is not None and semantic_goal.kind == "lap":
                    result = execute_numeric_patrol(self.runtime, self.config)
                    return self._envelope(
                        result,
                        history_index=history_index,
                        decision_source="global_static_3d_map_patrol_planner",
                        gemma_attempted=False,
                        gemma_accepted=False,
                        fallback_used=False,
                    )
                if (
                    semantic_goal is not None
                    and semantic_goal.kind in {"face", "approach"}
                    and self.trained_goal_policy is not None
                ):
                    result, source, fallback_used = self._execute_target_goal(
                        semantic_goal.kind,
                        semantic_goal.targets[0],
                    )
                    return self._envelope(
                        result,
                        history_index=history_index,
                        decision_source=source,
                        gemma_attempted=True,
                        gemma_accepted=bool(result.get("success")),
                        fallback_used=fallback_used,
                    )
                if semantic_goal is not None and semantic_goal.kind == "between":
                    result = execute_semantic_between_goal(
                        self.runtime,
                        self.config,
                        first_target_text=semantic_goal.targets[0],
                        second_target_text=semantic_goal.targets[1],
                        text_encoder=self.semantic_agent.text_encoder,
                    )
                    return self._envelope(
                        result,
                        history_index=history_index,
                        decision_source="local_gemma_all_voxel_between_goal_planner",
                        gemma_attempted=True,
                        gemma_accepted=bool(result.get("success")),
                        fallback_used=False,
                    )
                command_text = _without_polite_wrapper(normalized)
                command = (
                    parse_navigation_instruction(command_text) if command_text else None
                )
                if (
                    command is not None
                    and command.kind in {"face", "approach"}
                    and self.trained_goal_policy is not None
                ):
                    result, source, fallback_used = self._execute_target_goal(
                        command.kind,
                        command.targets[0],
                    )
                    return self._envelope(
                        result,
                        history_index=history_index,
                        decision_source=source,
                        gemma_attempted=True,
                        gemma_accepted=bool(result.get("success")),
                        fallback_used=fallback_used,
                    )
                if (
                    self.high_level_only
                    and command is not None
                    and command.kind not in {"face", "approach"}
                ):
                    result = {
                        "kind": "navigation",
                        "command": "high_level_only",
                        "success": False,
                        "error_code": "E_HIGH_LEVEL_ONLY",
                        "action_receipts": [],
                    }
                    return self._envelope(
                        result,
                        history_index=history_index,
                        decision_source="low_level_chat_command_disabled",
                        gemma_attempted=False,
                        gemma_accepted=False,
                        fallback_used=False,
                    )
                try:
                    direct_plan = _direct_action_plan(normalized, self.config)
                except OverflowError:
                    direct_plan = None
                    result = {
                        "kind": "navigation",
                        "command": (
                            "high_level_only" if self.high_level_only else "compound"
                        ),
                        "success": False,
                        "error_code": (
                            "E_HIGH_LEVEL_ONLY" if self.high_level_only else "E_PLAN_LIMIT"
                        ),
                        "action_receipts": [],
                    }
                    source = (
                        "low_level_chat_command_disabled"
                        if self.high_level_only
                        else "deterministic_plan_limit_rejection"
                    )
                    gemma_attempted = False
                    gemma_accepted = False
                    fallback_used = False
                    return self._envelope(
                        result,
                        history_index=history_index,
                        decision_source=source,
                        gemma_attempted=gemma_attempted,
                        gemma_accepted=gemma_accepted,
                        fallback_used=fallback_used,
                    )
                if self.high_level_only and direct_plan is not None:
                    result = {
                        "kind": "navigation",
                        "command": "high_level_only",
                        "success": False,
                        "error_code": "E_HIGH_LEVEL_ONLY",
                        "action_receipts": [],
                    }
                    return self._envelope(
                        result,
                        history_index=history_index,
                        decision_source="low_level_chat_command_disabled",
                        gemma_attempted=False,
                        gemma_accepted=False,
                        fallback_used=False,
                    )
                action_requested = (
                    command is not None
                    or direct_plan is not None
                    or bool(command_text)
                    and should_offer_llm_tool_policy(command_text)
                )
                decision = self._gemma_decision(normalized) if action_requested else None
                gemma_attempted = decision is not None
                gemma_accepted = False
                fallback_used = False
                expected = None
                if direct_plan is not None and len(direct_plan.actions) == 1:
                    expected = direct_plan.actions[0]
                elif direct_plan is None and command is not None:
                    expected = _parsed_expected_tool(command)
                # For an explicit numeric/simple request, a generated proposal
                # executes only when it agrees exactly with the deterministic
                # interpretation. A single model call cannot stand in for a
                # compound plan. Target-bearing requests deliberately fall
                # through to all-voxel semantic grounding and geometry.
                conforming = bool(
                    decision is not None
                    and decision.call is not None
                    and (
                        expected is not None
                        and _call_matches_expected(decision.call, expected)
                        or expected is None
                        and direct_plan is None
                        and command is None
                    )
                )
                if conforming:
                    assert decision is not None and decision.call is not None
                    receipt = execute_validated_tool_call(
                        self.runtime,
                        decision.call,
                        config=self.config,
                    )
                    result = {
                        "kind": "navigation",
                        "command": decision.call.name,
                        "success": bool(receipt.get("success")),
                        "error_code": receipt.get("error_code"),
                        "action_receipts": [receipt],
                        "tool_selection": decision.audit_payload(),
                    }
                    source = "local_gemma_constrained_json"
                    gemma_accepted = True
                elif direct_plan is not None:
                    result = _execute_direct_action_plan(
                        self.runtime,
                        direct_plan,
                        self.config,
                    )
                    if decision is not None:
                        result["tool_selection"] = decision.audit_payload()
                    source = (
                        "deterministic_compound_parser_fallback"
                        if gemma_attempted and len(direct_plan.actions) > 1
                        else "deterministic_bounded_parser_fallback"
                        if gemma_attempted
                        else "deterministic_compound_parser"
                        if len(direct_plan.actions) > 1
                        else "deterministic_bounded_parser"
                    )
                    fallback_used = gemma_attempted
                elif command is not None:
                    # This is an explicit, recorded fallback. Target-bearing
                    # commands still use local Gemma semantics by embedding the
                    # user's phrase and scoring every active-map voxel.
                    result = self.semantic_agent.handle(command_text)
                    source = (
                        "local_gemma_all_voxel_semantic_grounding_bounded_planner"
                        if command.targets
                        else "deterministic_bounded_parser_fallback"
                        if gemma_attempted
                        else "deterministic_bounded_parser"
                    )
                    fallback_used = gemma_attempted
                elif decision is not None:
                    result = {
                        "kind": "navigation",
                        "command": "local_gemma_tool_policy",
                        "success": False,
                        "error_code": "E_TOOL_POLICY_REJECTED",
                        "action_receipts": [],
                        "tool_selection": decision.audit_payload(),
                    }
                    source = "local_gemma_constrained_json_rejected"
                else:
                    result = self.semantic_agent.handle(normalized)
                    source = "local_gemma_continuous_scene_answer"
                    gemma_attempted = True
                    gemma_accepted = True
            return self._envelope(
                result,
                history_index=history_index,
                decision_source=source,
                gemma_attempted=gemma_attempted,
                gemma_accepted=gemma_accepted,
                fallback_used=fallback_used,
            )

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            try:
                if self.audit is not None:
                    self.audit.assert_clean()
                    if self.audit_output is not None:
                        self.audit.save(self.audit_output)
            finally:
                if self.audit is not None:
                    self.audit.__exit__(None, None, None)
                close = getattr(self.runtime, "close", None)
                if callable(close):
                    close()
                self._closed = True

    def __enter__(self) -> Self:
        self.startup()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def practical_rover_preflight(
    *,
    config: str | Path = DEFAULT_CONFIG,
    control_config: str | Path = DEFAULT_CONTROL_CONFIG,
    scene_id: str = DEFAULT_SCENE,
    base_checkpoint: str | Path = DEFAULT_BASE_CHECKPOINT,
    control_checkpoint: str | Path = DEFAULT_CONTROL_CHECKPOINT,
    runtime_asset: str | Path = DEFAULT_ASSET,
    robot_state_checkpoint: str | Path = DEFAULT_ROBOT_STATE_CHECKPOINT,
    navigation_checkpoint: str | Path = DEFAULT_NAVIGATION_CHECKPOINT,
) -> dict[str, Any]:
    """Model-free, renderer-free verification of the practical demo inputs."""

    if _SCENE_ID.fullmatch(scene_id) is None:
        raise ValueError("scene_id must be opaque")
    config_path = _safe_input(config, purpose="embodied config", kind="file")
    control_config_path = _safe_input(
        control_config, purpose="control runtime config", kind="file"
    )
    base = _safe_input(base_checkpoint, purpose="base checkpoint", kind="directory")
    control = _safe_input(
        control_checkpoint, purpose="control checkpoint", kind="directory"
    )
    asset = _safe_input(runtime_asset, purpose="runtime scene asset", kind="file")
    robot = _safe_input(
        robot_state_checkpoint, purpose="robot-state checkpoint", kind="directory"
    )
    navigation = _safe_input(
        navigation_checkpoint,
        purpose="task-trained Gemma waypoint checkpoint",
        kind="directory",
    )
    loaded = load_config(config_path)
    control_loaded = load_runtime_config(control_config_path)
    for label, root, inventory in (
        ("base", base, {"adapter.safetensors", "runtime_metadata.json"}),
        ("control", control, {"control.safetensors", "runtime_metadata.json"}),
        ("robot_state", robot, {"state.safetensors", "runtime_metadata.json"}),
        ("navigation", navigation, {"policy.safetensors", "runtime_metadata.json"}),
    ):
        observed = {item.name for item in root.iterdir()}
        if observed != inventory:
            raise ValueError(
                f"Practical rover {label} checkpoint inventory changed: "
                f"expected={sorted(inventory)} observed={sorted(observed)}"
            )
    render = loaded.get("render")
    mapping = loaded.get("mapping")
    if not isinstance(render, Mapping) or not isinstance(mapping, Mapping):
        raise TypeError("Embodied config lacks render or mapping settings")
    robot_settings = loaded.get("robot")
    if (
        not isinstance(robot_settings, Mapping)
        or robot_settings.get("auto_scan_after_motion") is not False
    ):
        raise ValueError("Practical rover requires static-map auto_scan_after_motion=false")
    resolution = tuple(int(value) for value in render["resolution"])
    scanner = SanitizedBlenderScanner(
        scene_id,
        asset,
        resolution=resolution,
        horizontal_fov_degrees=float(render["horizontal_fov_degrees"]),
        engine=str(render["engine"]),
        samples=int(render["samples"]),
        max_depth_m=float(mapping["depth_max_m"]),
        output_directory=PROJECT_ROOT / "data_gemma4" / "robot" / "practical_rover" / scene_id / "scans",
    )
    language = control_loaded.get("language")
    if not isinstance(language, Mapping) or language.get("backend") != "gemma4":
        raise ValueError("Practical rover requires a local Gemma runtime config")
    model_id = language.get("model_id")
    revision = language.get("revision")
    if not isinstance(model_id, str) or not isinstance(revision, str):
        raise TypeError("Local Gemma model identity must be pinned")
    with (navigation / "runtime_metadata.json").open("r", encoding="utf-8") as handle:
        navigation_metadata = json.load(handle)
    with (robot / "runtime_metadata.json").open("r", encoding="utf-8") as handle:
        robot_metadata = json.load(handle)
    if not isinstance(navigation_metadata, Mapping):
        raise TypeError("Gemma waypoint metadata must be an object")
    navigation_history_contract = _validate_gemma_waypoint_history_contract(
        navigation_metadata
    )
    navigation_weights_sha256 = navigation_metadata.get("weights_sha256")
    if (
        not isinstance(navigation_weights_sha256, str)
        or _SHA256.fullmatch(navigation_weights_sha256) is None
    ):
        raise ValueError("Gemma waypoint metadata has no valid weights identity")
    weights_digest = hashlib.sha256()
    with (navigation / "policy.safetensors").open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            weights_digest.update(chunk)
    if weights_digest.hexdigest() != navigation_weights_sha256:
        raise ValueError("Gemma waypoint checkpoint weights changed")
    navigation_runtime_binding = validate_gemma_runtime_binding(
        navigation_metadata.get("gemma_runtime_binding")
    )
    navigation_runtime_binding_sha256 = gemma_runtime_binding_sha256(
        navigation_runtime_binding
    )
    if (
        navigation_metadata.get("gemma_runtime_binding_sha256")
        != navigation_runtime_binding_sha256
        or navigation_runtime_binding.get("effective_runtime_config_sha256")
        != effective_runtime_config_sha256(control_loaded)
        or navigation_runtime_binding.get("base_checkpoint_sha256")
        != checkpoint_fingerprint_sha256(base)
        or navigation_runtime_binding.get("control_checkpoint_sha256")
        != control_checkpoint_fingerprint_sha256(control)
    ):
        raise ValueError("Gemma waypoint runtime binding differs from local preflight")
    required_navigation_flags = (
        "actual_gemma_causal_forward",
        "gemma_output_hidden_states",
        "complete_scene_prefix_required",
        "every_scene_token_processed",
        "numeric_state_and_history_required",
        "model_selects_every_waypoint_and_heading",
        "saved_controller_tensors_only",
        "scene_splits_disjoint",
        "context_projection_frozen_during_training",
    )
    if (
        any(navigation_metadata.get(field) is not True for field in required_navigation_flags)
        or navigation_metadata.get("architecture")
        != GEMMA_WAYPOINT_CHECKPOINT_ARCHITECTURE
        or navigation_metadata.get("action_names") != ["move_to", "face", "stop"]
        or navigation_metadata.get("runtime_required_files")
        != ["policy.safetensors", "runtime_metadata.json"]
        or navigation_metadata.get("frozen_gemma_weights_saved") is not False
        or navigation_metadata.get("deterministic_route_planner_allowed_at_runtime")
        is not False
        or navigation_metadata.get("model_id") != model_id
        or navigation_metadata.get("model_revision") != revision
        or navigation_metadata.get("environmental_text_inputs") != []
        or navigation_metadata.get("oracle_inputs_at_runtime") is not False
    ):
        raise ValueError("Task-trained Gemma waypoint checkpoint contract is invalid")
    for field in (
        "scene_token_count",
        "robot_token_count",
        "history_dim",
        "max_history_tokens",
        "hidden_size",
        "training_sample_count",
        "validation_sample_count",
    ):
        value = navigation_metadata.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"Gemma waypoint metadata has invalid {field}")
    scene_encoder = loaded.get("scene_encoder")
    configured_latents = (
        scene_encoder.get("global_latents")
        if isinstance(scene_encoder, Mapping)
        else None
    )
    if (
        isinstance(configured_latents, bool)
        or not isinstance(configured_latents, int)
        or not isinstance(robot_metadata, Mapping)
        or navigation_metadata.get("scene_token_count") != configured_latents + 2
        or navigation_metadata.get("robot_token_count")
        != robot_metadata.get("token_count")
        or navigation_metadata.get("hidden_size")
        != robot_metadata.get("output_dim")
        or (
            navigation_metadata.get("schema"),
            navigation_metadata.get("history_dim"),
            navigation_metadata.get("history_parameterization"),
        )
        != navigation_history_contract
    ):
        raise ValueError("Gemma waypoint checkpoint dimensions differ from live runtime")
    return {
        "schema": SCHEMA,
        "ready": True,
        "loads_language_model": False,
        "renders_scene": False,
        "changes_robot_or_map_state": False,
        "scene_id": scene_id,
        "model_id": model_id,
        "model_revision": revision,
        "local_model_snapshot_required": True,
        "blender_executable": str(scanner.blender_executable),
        "asset_sha256": scanner.asset_sha256,
        "continuous_map": str(project_path(loaded, "maps", scene_id, "voxel_map.npz")),
        "navigation_checkpoint": str(navigation),
        "navigation_checkpoint_sha256": navigation_weights_sha256,
        "gemma_runtime_binding_sha256": navigation_runtime_binding_sha256,
        "navigation_policy_task_trained": True,
        "navigation_policy_actual_gemma_causal_forward": True,
        "navigation_scene_token_count": int(navigation_metadata["scene_token_count"]),
        "navigation_robot_token_count": int(navigation_metadata["robot_token_count"]),
        "navigation_history_feature_dim": int(navigation_metadata["history_dim"]),
        "navigation_history_parameterization": str(
            navigation_metadata["history_parameterization"]
        ),
        "navigation_max_history_tokens": int(
            navigation_metadata["max_history_tokens"]
        ),
        "every_scene_token_processed_per_navigation_decision": True,
        "every_map_voxel_influences_scene_prefix": True,
        "question_dependent_target_grounding": False,
        "model_selects_every_waypoint_and_heading": True,
        "model_selects_stop": True,
        "deterministic_route_planner_allowed_at_runtime": False,
        "high_level_natural_language_only": True,
        "initial_scan": False,
        "auto_scan_after_motion": False,
        "untrained_json_backend_enabled": False,
        "action_validation": "exact_model_bound_action_or_rejection",
        "fallback": None,
        "environmental_text_inputs": [],
    }


def build_local_practical_rover(
    *,
    config: str | Path = DEFAULT_CONFIG,
    control_config: str | Path = DEFAULT_CONTROL_CONFIG,
    scene_id: str = DEFAULT_SCENE,
    base_checkpoint: str | Path = DEFAULT_BASE_CHECKPOINT,
    control_checkpoint: str | Path = DEFAULT_CONTROL_CHECKPOINT,
    runtime_asset: str | Path = DEFAULT_ASSET,
    robot_state_checkpoint: str | Path = DEFAULT_ROBOT_STATE_CHECKPOINT,
    navigation_checkpoint: str | Path = DEFAULT_NAVIGATION_CHECKPOINT,
    persistent_map: str | Path | None = None,
    audit_output: str | Path | None = None,
    enable_gemma_json_fallback: bool = False,
    initial_scan: bool = False,
) -> PracticalRoverController:
    """Load the actual local Gemma, renderer, continuous map, and rover runtime."""

    if enable_gemma_json_fallback:
        raise ValueError("The untrained Gemma JSON backend is disabled")
    if initial_scan:
        raise ValueError("The practical rover uses precomputed memory without an initial scan")
    practical_rover_preflight(
        config=config,
        control_config=control_config,
        scene_id=scene_id,
        base_checkpoint=base_checkpoint,
        control_checkpoint=control_checkpoint,
        runtime_asset=runtime_asset,
        robot_state_checkpoint=robot_state_checkpoint,
        navigation_checkpoint=navigation_checkpoint,
    )
    audit = _runtime_audit()
    audit.__enter__()
    try:
        embodied_config = load_config(_safe_input(config, purpose="embodied config", kind="file"))
        control_runtime_config = load_runtime_config(
            _safe_input(control_config, purpose="control config", kind="file"),
            record_file=audit.record,
        )
        chat = QuestionControlledChatRuntime.load(
            control_runtime_config,
            scene_id,
            base_checkpoint=_safe_input(
                base_checkpoint, purpose="base checkpoint", kind="directory"
            ),
            control_checkpoint=_safe_input(
                control_checkpoint, purpose="control checkpoint", kind="directory"
            ),
            audit=audit,
        )
        render = embodied_config["render"]
        scanner = SanitizedBlenderScanner(
            scene_id,
            _safe_input(runtime_asset, purpose="runtime scene asset", kind="file"),
            resolution=tuple(int(value) for value in render["resolution"]),
            horizontal_fov_degrees=float(render["horizontal_fov_degrees"]),
            engine=str(render["engine"]),
            samples=int(render["samples"]),
            max_depth_m=float(embodied_config["mapping"]["depth_max_m"]),
            output_directory=(
                PROJECT_ROOT
                / "data_gemma4"
                / "robot"
                / "practical_rover"
                / scene_id
                / "scans"
            ),
        )
        persistent = (
            PROJECT_ROOT
            / "data_gemma4"
            / "robot"
            / "practical_rover"
            / scene_id
            / "semantic_map.npz"
            if persistent_map is None
            else _rooted(persistent_map)
        )
        runtime = build_refreshing_embodied_runtime(
            embodied_config,
            scene_id,
            checkpoint=_safe_input(
                base_checkpoint, purpose="base checkpoint", kind="directory"
            ),
            chat_runtime=chat,
            persistent_map_path=persistent,
            observation_scanner=scanner,
            robot_state_checkpoint=_safe_input(
                robot_state_checkpoint,
                purpose="robot-state checkpoint",
                kind="directory",
            ),
            audit=audit,
            local_files_only=True,
        )
        text_encoder = GemmaProjectedTextEncoder.from_config(embodied_config)
        semantic_agent = ConversationalEmbodiedAgent(
            runtime,
            text_encoder,
            room_size_m=embodied_config["scene"]["room_size_m"],
        )
        wrapped_chat = getattr(
            getattr(runtime, "prefix_refresher", None), "runtime", None
        )
        base_chat = getattr(wrapped_chat, "base", wrapped_chat)
        language_runtime = getattr(base_chat, "language", None)
        prefix_backend = getattr(language_runtime, "prefix_backend", None)
        tokenizer = getattr(language_runtime, "tokenizer", None)
        language_settings = control_runtime_config.get("language")
        if not isinstance(language_settings, Mapping):
            raise TypeError("Control runtime lacks pinned local Gemma settings")
        model_id = language_settings.get("model_id")
        model_revision = language_settings.get("revision")
        if not isinstance(model_id, str) or not isinstance(model_revision, str):
            raise TypeError("Control runtime lacks a pinned local Gemma identity")
        loaded_waypoint_policy = load_gemma_waypoint_policy_checkpoint(
            _safe_input(
                navigation_checkpoint,
                purpose="task-trained Gemma waypoint checkpoint",
                kind="directory",
            ),
            prefix_backend=prefix_backend,
            tokenizer=tokenizer,
            expected_model_id=model_id,
            expected_model_revision=model_revision,
            expected_gemma_runtime_binding=question_controlled_gemma_runtime_binding(
                chat,
                control_runtime_config,
                base_checkpoint=base_checkpoint,
                control_checkpoint=control_checkpoint,
            ),
            audit=audit,
        )
        gemma_waypoint_controller = GemmaWaypointClosedLoopController.from_loaded(
            runtime=runtime,
            config=embodied_config,
            loaded=loaded_waypoint_policy,
        )
        controller = PracticalRoverController(
            runtime,
            semantic_agent,
            embodied_config,
            llm_tool_policy=None,
            trained_goal_policy=None,
            gemma_waypoint_controller=gemma_waypoint_controller,
            high_level_only=True,
            audit=audit,
            audit_output=audit_output,
            initial_scan=initial_scan,
        )
        controller.startup()
        audit.assert_clean()
        return controller
    except BaseException:
        audit.__exit__(None, None, None)
        raise


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
    parser.add_argument("--check", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    kwargs = {
        "config": args.config,
        "control_config": args.control_config,
        "scene_id": args.scene,
        "base_checkpoint": args.base_checkpoint,
        "control_checkpoint": args.control_checkpoint,
        "runtime_asset": args.runtime_asset,
        "robot_state_checkpoint": args.robot_state_checkpoint,
        "navigation_checkpoint": args.navigation_checkpoint,
    }
    if not args.check:
        raise ValueError("Use the local rover web UI for live inference; this CLI supports --check")
    print(json.dumps(practical_rover_preflight(**kwargs), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_NAVIGATION_CHECKPOINT",
    "PracticalRoverController",
    "build_local_practical_rover",
    "practical_rover_preflight",
]
