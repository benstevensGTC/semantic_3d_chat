"""V3 navigation controller with label-free continuous semantic grounding.

Static scene-QA remains question-independent.  This module is only for embodied
navigation, where the user's target phrase is embedded locally, compared with
every language-aligned voxel in the active 3D map, and converted to a numeric
grounded-target state.  No object label inventory, simulator metadata, oracle,
caption, or scene graph is accepted at runtime.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from torch import nn

from semantic_3d_chat.chat.file_audit import FileAccessAudit
from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.evaluation.gemma4_semantic_sanity import (
    GEMMA4_PROJECTED_DIM,
    GEMMA4_PROJECTED_START,
)
from semantic_3d_chat.language.prefix_injection import prefix_sha256
from semantic_3d_chat.robot.action_context import (
    ContinuousActionContext,
    capture_continuous_action_context,
    require_grounding_map_binding,
)
from semantic_3d_chat.robot.llm_tool_policy import GeneratedToolProposal
from semantic_3d_chat.robot.navigation_policy import (
    ACTION_NAMES,
    split_active_prefix,
    tool_call_from_prediction,
)
from semantic_3d_chat.robot.semantic_agent import (
    ContinuousSemanticGrounding,
    ContinuousSemanticTargetGrounder,
    ContinuousTextEncoder,
    GemmaProjectedTextEncoder,
)
from semantic_3d_chat.robot.state_encoder import robot_state_vector

TARGET_STATE_DIM: Final[int] = 10
ARCHITECTURE: Final[str] = "continuous_semantic_grounded_navigation_controller_v3"
TRAINING_STATUS: Final[str] = "supervised_continuous_semantic_grounded_navigation_policy_v3"
SCHEMA_VERSION: Final[int] = 3
# The accepted checkpoint and learned controller remain V3.  This independently
# versioned runtime contract records the bounded, label-free convergence fix for
# compound ``scan, then approach ... and stop`` instructions.  Keeping the
# version separate prevents current behavior from being confused with the
# immutable historical V3 live journal.
RUNTIME_INTERLOCK_VERSION: Final[str] = "v3.1"
_FILES: Final[frozenset[str]] = frozenset({"policy.safetensors", "runtime_metadata.json"})
_SHA256 = re.compile(r"[0-9a-f]{64}")
_BLOCKED = frozenset({"oracle", "qa", "training", "scorer_only"})
_METADATA_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "architecture",
        "hidden_size",
        "model_dim",
        "target_state_dim",
        "scene_token_count",
        "robot_token_count",
        "action_names",
        "model_id",
        "model_revision",
        "max_turn_degrees",
        "max_move_m",
        "room_size_m",
        "grounding_feature_start",
        "grounding_feature_dim",
        "task_trained",
        "training_dataset_sha256",
        "train_scene_count",
        "validation_scene_count",
        "scene_splits_disjoint",
        "complete_scene_prefix_required",
        "question_independent_static_scene_prefix_required",
        "every_scene_token_processed",
        "numeric_robot_tokens_required",
        "continuous_semantic_grounding_required",
        "all_map_voxels_scored_for_grounding",
        "query_dependent_grounding_navigation_only",
        "environmental_text_inputs",
        "oracle_inputs_at_runtime",
        "runtime_required_files",
        "collision_interlock_required",
        "weights_sha256",
    }
)
_SPACE = re.compile(r"\s+")
_TRAILING_STOP = re.compile(r"(?:,?\s+then\s+stop|\s+and\s+stop)[.!]?\s*$", re.IGNORECASE)
_DIRECTION_SUFFIX = re.compile(r"\s+using\s+the\s+shorter\s+direction\s*$", re.IGNORECASE)
_ALIGNMENT_PREFIX = re.compile(
    r"^(?:face|turn\s+toward|turn\s+to\s+face|look\s+at)\b",
    re.IGNORECASE,
)
_APPROACH_PREFIX = re.compile(
    r"^(?:move\s+closer\s+to|approach|walk\s+toward|move\s+toward)\b",
    re.IGNORECASE,
)
_SCAN_THEN_APPROACH_PREFIX = re.compile(
    r"^(?:scan(?:\s+the\s+room)?|look\s+around)\s*,?\s+then\s+"
    r"(?:move\s+closer\s+to|approach|walk\s+toward|move\s+toward)\b",
    re.IGNORECASE,
)
_DEFAULT_ALIGNMENT_DEADBAND_DEGREES: Final[float] = 3.0
_DEFAULT_STALLED_TURN_DEGREES: Final[float] = 1.0
_DEFAULT_APPROACH_HEADING_DEADBAND_DEGREES: Final[float] = 15.0
_DEFAULT_APPROACH_TARGET_STANDOFF_M: Final[float] = 0.5
_DEFAULT_APPROACH_MINIMUM_PROGRESS_M: Final[float] = 0.15
_DEFAULT_APPROACH_MINIMUM_SAFE_STEP_M: Final[float] = 0.02


def _rooted(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    rooted = candidate if candidate.is_absolute() else PROJECT_ROOT / candidate
    return Path(os.path.abspath(rooted))


def _reject_runtime_path(path: Path) -> None:
    if _BLOCKED & {part.casefold() for part in path.parts}:
        raise ValueError("V3 runtime paths cannot enter blocked data trees")
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current /= component
        if current.is_symlink():
            raise ValueError("V3 runtime paths cannot contain symbolic links")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _positive_int(value: object, name: str, maximum: int = 16384) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
        raise ValueError(f"V3 {name} must be in [1, {maximum}]")
    return value


def _finite_positive(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"V3 {name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"V3 {name} must be finite and positive")
    return result


def _finite_nonnegative(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"V3 {name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"V3 {name} must be finite and nonnegative")
    return result


def _terminal_alignment_requested(instruction: str) -> bool:
    """Recognize a protocol-level face/turn/look goal with an explicit stop.

    This deliberately examines only the action grammar supplied by the user.
    The matched target phrase is neither interpreted nor retained, so this
    cannot introduce an environmental category inventory into the runtime.
    """

    literal = _SPACE.sub(" ", _literal_instruction(instruction)).strip()
    without_stop = _TRAILING_STOP.sub("", literal).strip()
    return without_stop != literal and _ALIGNMENT_PREFIX.match(without_stop) is not None


def _terminal_approach_requested(instruction: str) -> bool:
    """Recognize a user-requested approach goal that explicitly ends in stop.

    V3.1 also recognizes a protocol-level scan/look preamble before the final
    approach action.  The grammar contains action verbs only: the arbitrary
    target phrase is neither classified nor retained, so this fix introduces
    no category names, object IDs, captions, or oracle inputs.
    """

    literal = _SPACE.sub(" ", _literal_instruction(instruction)).strip()
    without_stop = _TRAILING_STOP.sub("", literal).strip()
    return without_stop != literal and bool(
        _APPROACH_PREFIX.match(without_stop)
        or _SCAN_THEN_APPROACH_PREFIX.match(without_stop)
    )


def apply_numeric_alignment_interlock(
    instruction: str,
    target_state: torch.Tensor,
    learned_call: Mapping[str, Any],
    *,
    target_xyz_m: Sequence[float] | None,
    robot_position_xy_m: Sequence[float],
    robot_yaw_degrees: float,
    max_turn_degrees: float,
    deadband_degrees: float = _DEFAULT_ALIGNMENT_DEADBAND_DEGREES,
    stalled_turn_degrees: float = _DEFAULT_STALLED_TURN_DEGREES,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Complete terminal alignment goals using only grounded numeric geometry.

    V3 predicts both a discrete action and a bounded argument.  A trained
    classifier can keep selecting ``turn`` after its argument regressor has
    converged to nearly zero.  For an explicit ``face/turn/look ... then stop``
    instruction, this interlock detects that numerical stall, applies the
    remaining angle encoded in the continuous grounded-target state, and only
    permits ``stop`` after a newly grounded residual enters the deadband.

    The target state and explicit geometric arguments contain no labels.  The
    remaining yaw is independently recomputed from numeric target XYZ, robot
    XY, and robot yaw rather than trusting the learned argument regressor.
    """

    maximum = _finite_positive(max_turn_degrees, "max_turn_degrees")
    deadband = _finite_positive(deadband_degrees, "alignment_deadband_degrees")
    stalled = _finite_positive(stalled_turn_degrees, "stalled_turn_degrees")
    if deadband >= maximum or stalled >= maximum:
        raise ValueError("V3 alignment thresholds must be smaller than the turn bound")
    state = torch.as_tensor(target_state, dtype=torch.float32).detach().cpu()
    if state.shape != (1, TARGET_STATE_DIM) or not torch.isfinite(state).all():
        raise ValueError("V3 alignment interlock requires one finite target state")
    call = dict(learned_call)
    arguments = call.get("arguments")
    if not isinstance(call.get("tool"), str) or not isinstance(arguments, Mapping):
        raise TypeError("V3 learned call has an invalid shape")

    requested = _terminal_alignment_requested(instruction)
    target_available = bool(float(state[0, 0].item()) == 1.0)
    position = np.asarray(robot_position_xy_m, dtype=np.float64)
    yaw = float(robot_yaw_degrees)
    if position.shape != (2,) or not np.isfinite(position).all() or not math.isfinite(yaw):
        raise ValueError("V3 alignment interlock robot pose is invalid")
    target: np.ndarray | None = None
    desired_yaw: float | None = None
    residual: float | None = None
    if target_available:
        if target_xyz_m is None:
            raise ValueError("V3 available alignment target has no numeric XYZ")
        target = np.asarray(target_xyz_m, dtype=np.float64)
        if target.shape != (3,) or not np.isfinite(target).all():
            raise ValueError("V3 alignment target XYZ is invalid")
        delta = target[:2] - position
        if float(np.linalg.norm(delta)) <= 1e-8:
            raise ValueError("V3 alignment target is coincident with robot XY")
        desired_yaw = math.degrees(math.atan2(-float(delta[0]), float(delta[1])))
        raw_residual = desired_yaw - yaw
        residual = math.degrees(
            math.atan2(math.sin(math.radians(raw_residual)), math.cos(math.radians(raw_residual)))
        )
        encoded_residual = math.degrees(
            math.atan2(float(state[0, 8].item()), float(state[0, 9].item()))
        )
        disagreement = math.degrees(
            math.atan2(
                math.sin(math.radians(residual - encoded_residual)),
                math.cos(math.radians(residual - encoded_residual)),
            )
        )
        if abs(disagreement) > 1e-3:
            raise RuntimeError("V3 target-state heading differs from numeric target geometry")
    elif target_xyz_m is not None:
        raise ValueError("V3 unavailable alignment target unexpectedly has numeric XYZ")
    learned_turn: float | None = None
    if call["tool"] == "turn":
        value = arguments.get("angle_degrees")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("V3 learned turn argument is not numeric")
        learned_turn = float(value)
        if not math.isfinite(learned_turn):
            raise ValueError("V3 learned turn argument is not finite")

    output = {"tool": call["tool"], "arguments": dict(arguments)}
    correction_applied = False
    stop_applied = False
    reason = "not_terminal_alignment"
    if requested and target_available:
        assert residual is not None
        if abs(residual) <= deadband:
            output = {"tool": "stop", "arguments": {}}
            stop_applied = True
            reason = "fresh_grounding_inside_deadband"
        elif call["tool"] == "stop" or (
            learned_turn is not None and abs(learned_turn) <= stalled
        ):
            correction = max(-maximum, min(maximum, residual))
            output = {"tool": "turn", "arguments": {"angle_degrees": correction}}
            correction_applied = True
            reason = (
                "premature_learned_stop"
                if call["tool"] == "stop"
                else "stalled_learned_turn"
            )
        else:
            reason = "learned_action_not_stalled"
    elif requested:
        reason = "target_unavailable"

    audit = {
        "schema": "semantic_3d_chat.numeric_alignment_interlock.v1",
        "enabled": True,
        "terminal_alignment_requested": requested,
        "target_available": target_available,
        "target_xyz_m": None if target is None else target.tolist(),
        "robot_position_xy_m": position.tolist(),
        "robot_yaw_degrees": yaw,
        "desired_yaw_degrees": desired_yaw,
        "angular_residual_degrees": residual,
        "deadband_degrees": deadband,
        "stalled_turn_degrees": stalled,
        "learned_tool": call["tool"],
        "learned_turn_degrees": learned_turn,
        "correction_applied": correction_applied,
        "corrected_turn_degrees": (
            output["arguments"].get("angle_degrees") if correction_applied else None
        ),
        "stop_applied": stop_applied,
        "reason": reason,
        "numeric_target_state_only": True,
        "environmental_text_inputs": [],
        "oracle_inputs_at_runtime": False,
    }
    return output, audit


def apply_numeric_approach_interlock(
    instruction: str,
    target_state: torch.Tensor,
    learned_call: Mapping[str, Any],
    *,
    target_xyz_m: Sequence[float] | None,
    initial_robot_position_xy_m: Sequence[float],
    robot_position_xy_m: Sequence[float],
    robot_yaw_degrees: float,
    max_turn_degrees: float,
    max_move_m: float,
    heading_deadband_degrees: float = _DEFAULT_APPROACH_HEADING_DEADBAND_DEGREES,
    target_standoff_m: float = _DEFAULT_APPROACH_TARGET_STANDOFF_M,
    minimum_progress_m: float = _DEFAULT_APPROACH_MINIMUM_PROGRESS_M,
    stalled_turn_degrees: float = _DEFAULT_STALLED_TURN_DEGREES,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Reject a premature approach ``stop`` using only numeric scene geometry.

    The learned controller remains the primary policy.  This bounded interlock
    acts only on an explicit ``move closer ... then stop`` protocol goal.  A
    stop is accepted only after the robot has actually translated and fresh
    all-voxel grounding places it within the configured target standoff.
    Premature stops become either a bounded heading correction or a bounded
    forward step.  Collision rejection remains authoritative downstream.
    """

    maximum_turn = _finite_positive(max_turn_degrees, "max_turn_degrees")
    maximum_move = _finite_positive(max_move_m, "max_move_m")
    heading_deadband = _finite_positive(
        heading_deadband_degrees, "approach_heading_deadband_degrees"
    )
    standoff = _finite_positive(target_standoff_m, "approach_target_standoff_m")
    minimum_progress = _finite_positive(
        minimum_progress_m, "approach_minimum_progress_m"
    )
    stalled = _finite_positive(stalled_turn_degrees, "stalled_turn_degrees")
    if heading_deadband >= maximum_turn or stalled >= maximum_turn:
        raise ValueError("V3 approach angular thresholds exceed the turn bound")
    if minimum_progress > maximum_move:
        raise ValueError("V3 approach minimum progress exceeds one bounded move")

    state = torch.as_tensor(target_state, dtype=torch.float32).detach().cpu()
    if state.shape != (1, TARGET_STATE_DIM) or not torch.isfinite(state).all():
        raise ValueError("V3 approach interlock requires one finite target state")
    call = dict(learned_call)
    arguments = call.get("arguments")
    if not isinstance(call.get("tool"), str) or not isinstance(arguments, Mapping):
        raise TypeError("V3 learned approach call has an invalid shape")

    requested = _terminal_approach_requested(instruction)
    target_available = bool(float(state[0, 0].item()) == 1.0)
    initial = np.asarray(initial_robot_position_xy_m, dtype=np.float64)
    position = np.asarray(robot_position_xy_m, dtype=np.float64)
    yaw = float(robot_yaw_degrees)
    if (
        initial.shape != (2,)
        or position.shape != (2,)
        or not np.isfinite(initial).all()
        or not np.isfinite(position).all()
        or not math.isfinite(yaw)
    ):
        raise ValueError("V3 approach robot pose is invalid")

    target: np.ndarray | None = None
    target_distance: float | None = None
    desired_yaw: float | None = None
    residual: float | None = None
    if target_available:
        if target_xyz_m is None:
            raise ValueError("V3 available approach target has no numeric XYZ")
        target = np.asarray(target_xyz_m, dtype=np.float64)
        if target.shape != (3,) or not np.isfinite(target).all():
            raise ValueError("V3 approach target XYZ is invalid")
        delta = target[:2] - position
        target_distance = float(np.linalg.norm(delta))
        if target_distance <= 1e-8:
            raise ValueError("V3 approach target is coincident with robot XY")
        desired_yaw = math.degrees(math.atan2(-float(delta[0]), float(delta[1])))
        raw_residual = desired_yaw - yaw
        residual = math.degrees(
            math.atan2(
                math.sin(math.radians(raw_residual)),
                math.cos(math.radians(raw_residual)),
            )
        )
    elif target_xyz_m is not None:
        raise ValueError("V3 unavailable approach target unexpectedly has numeric XYZ")

    progress = float(np.linalg.norm(position - initial))
    goal_satisfied = bool(
        target_available
        and target_distance is not None
        and progress >= minimum_progress - 1e-6
        and target_distance <= standoff + 1e-6
    )
    learned_turn: float | None = None
    if call["tool"] == "turn":
        value = arguments.get("angle_degrees")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("V3 learned approach turn argument is not numeric")
        learned_turn = float(value)
        if not math.isfinite(learned_turn):
            raise ValueError("V3 learned approach turn argument is not finite")

    output = {"tool": call["tool"], "arguments": dict(arguments)}
    correction_applied = False
    stop_applied = False
    reason = "not_terminal_approach"
    if requested and target_available:
        assert residual is not None and target_distance is not None
        stalled_turn = learned_turn is not None and abs(learned_turn) <= stalled
        if goal_satisfied:
            output = {"tool": "stop", "arguments": {}}
            stop_applied = True
            reason = "fresh_grounding_approach_goal_satisfied"
        elif abs(residual) > heading_deadband and (
            call["tool"] in {"stop", "move_forward", "move_backward"} or stalled_turn
        ):
            correction = max(-maximum_turn, min(maximum_turn, residual))
            output = {"tool": "turn", "arguments": {"angle_degrees": correction}}
            correction_applied = True
            reason = "premature_stop_heading_correction" if call["tool"] == "stop" else (
                "unsafe_move_heading_correction"
                if call["tool"] in {"move_forward", "move_backward"}
                else "stalled_turn_heading_correction"
            )
        elif call["tool"] == "stop" or stalled_turn:
            required_progress = max(0.0, minimum_progress - progress)
            required_standoff = max(0.0, target_distance - standoff)
            distance = min(maximum_move, max(required_progress, required_standoff))
            if distance <= 1e-6:
                raise RuntimeError("V3 approach goal is incomplete but no safe progress remains")
            output = {
                "tool": "move_forward",
                "arguments": {"distance_meters": distance},
            }
            correction_applied = True
            reason = (
                "premature_stop_forward_progress"
                if call["tool"] == "stop"
                else "stalled_turn_forward_progress"
            )
        else:
            reason = "learned_action_not_stalled"
    elif requested:
        reason = "target_unavailable"

    audit = {
        "schema": "semantic_3d_chat.numeric_approach_interlock.v1",
        "runtime_interlock_version": RUNTIME_INTERLOCK_VERSION,
        "enabled": True,
        "terminal_approach_requested": requested,
        "target_available": target_available,
        "target_xyz_m": None if target is None else target.tolist(),
        "initial_robot_position_xy_m": initial.tolist(),
        "robot_position_xy_m": position.tolist(),
        "robot_yaw_degrees": yaw,
        "desired_yaw_degrees": desired_yaw,
        "angular_residual_degrees": residual,
        "target_distance_m": target_distance,
        "actual_progress_m": progress,
        "minimum_progress_m": minimum_progress,
        "target_standoff_m": standoff,
        "heading_deadband_degrees": heading_deadband,
        "learned_tool": call["tool"],
        "learned_turn_degrees": learned_turn,
        "goal_satisfied": goal_satisfied,
        "completion_satisfied": goal_satisfied,
        "completion_mode": "semantic_standoff" if goal_satisfied else None,
        "correction_applied": correction_applied,
        "corrected_tool": output["tool"] if correction_applied else None,
        "corrected_arguments": output["arguments"] if correction_applied else None,
        "stop_applied": stop_applied,
        "reason": reason,
        "numeric_target_state_only": True,
        "environmental_text_inputs": [],
        "oracle_inputs_at_runtime": False,
    }
    return output, audit


def apply_collision_limited_approach_interlock(
    learned_call: Mapping[str, Any],
    approach_audit: Mapping[str, Any],
    *,
    collision_map: Any,
    robot_position_xy_m: Sequence[float],
    robot_yaw_degrees: float,
    minimum_safe_step_m: float = _DEFAULT_APPROACH_MINIMUM_SAFE_STEP_M,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Clip a terminal approach step or stop at the closest safe pose.

    This interlock uses only the anonymous numeric collision field, the current
    numeric robot pose, and the already selected bounded movement.  If the full
    step would collide, it first searches for a shorter collision-free step. If
    less than ``minimum_safe_step_m`` remains after the robot has made the
    required approach progress, it stops without contact and records a distinct
    collision-limited completion mode.  No object identity or oracle geometry
    is available here.
    """

    minimum_safe_step = _finite_positive(
        minimum_safe_step_m, "approach_minimum_safe_step_m"
    )
    call = {
        "tool": learned_call.get("tool"),
        "arguments": dict(learned_call.get("arguments", {})),
    }
    audit = dict(approach_audit)
    collision_audit: dict[str, Any] = {
        "schema": "semantic_3d_chat.collision_limited_approach_interlock.v1",
        "enabled": True,
        "numeric_collision_map_only": True,
        "oracle_inputs_at_runtime": False,
        "environmental_text_inputs": [],
        "collision_predicted": False,
        "requested_distance_m": None,
        "maximum_collision_free_distance_m": None,
        "executed_safe_distance_m": None,
        "minimum_safe_step_m": minimum_safe_step,
        "safe_closest_reachable": False,
        "reason": "not_applicable",
    }
    audit["collision_limited_interlock"] = collision_audit
    audit["collision_precheck_predicted_rejection"] = False
    audit["collision_rejection_deferred_to_exact_simulator"] = False

    if audit.get("terminal_approach_requested") is not True or call["tool"] not in {
        "move_forward",
        "move_backward",
    }:
        return call, audit
    if collision_map is None or not callable(getattr(collision_map, "segment_check", None)):
        raise TypeError("V3 collision-limited approach requires a numeric collision map")
    distance = call["arguments"].get("distance_meters")
    if isinstance(distance, bool) or not isinstance(distance, (int, float)):
        raise TypeError("V3 collision-limited approach distance is not numeric")
    requested_distance = _finite_positive(distance, "approach movement distance")
    start = np.asarray(robot_position_xy_m, dtype=np.float64)
    yaw = float(robot_yaw_degrees)
    if start.shape != (2,) or not np.isfinite(start).all() or not math.isfinite(yaw):
        raise ValueError("V3 collision-limited approach robot pose is invalid")
    direction = np.asarray(
        [-math.sin(math.radians(yaw)), math.cos(math.radians(yaw))], dtype=np.float64
    )
    if call["tool"] == "move_backward":
        direction *= -1.0
    collision_audit["requested_distance_m"] = requested_distance
    full_check = collision_map.segment_check(start, start + requested_distance * direction)
    if full_check.collision is not True:
        collision_audit["reason"] = "full_step_collision_free"
        return call, audit

    collision_audit["collision_predicted"] = True
    audit["collision_precheck_predicted_rejection"] = True
    low = 0.0
    high = requested_distance
    for _ in range(24):
        midpoint = (low + high) / 2.0
        check = collision_map.segment_check(start, start + midpoint * direction)
        if check.collision:
            high = midpoint
        else:
            low = midpoint
    # Stay measurably inside the numeric free-space boundary instead of relying
    # on floating-point equality at the inflated collision radius.
    safe_distance = max(0.0, low - min(0.005, low * 0.10))
    collision_audit["maximum_collision_free_distance_m"] = low
    progress = _finite_nonnegative(
        audit.get("actual_progress_m"), "approach actual progress"
    )
    minimum_progress = _finite_nonnegative(
        audit.get("minimum_progress_m"), "approach minimum progress"
    )
    if safe_distance >= minimum_safe_step:
        output = {
            "tool": call["tool"],
            "arguments": {"distance_meters": safe_distance},
        }
        collision_audit["executed_safe_distance_m"] = safe_distance
        collision_audit["reason"] = "clipped_to_collision_free_step"
        audit["correction_applied"] = True
        audit["corrected_tool"] = output["tool"]
        audit["corrected_arguments"] = output["arguments"]
        audit["reason"] = "collision_limited_safe_progress"
        return output, audit

    if progress >= minimum_progress - 1e-6:
        output = {"tool": "stop", "arguments": {}}
        collision_audit["safe_closest_reachable"] = True
        collision_audit["reason"] = "no_material_collision_free_step_remains"
        audit["correction_applied"] = True
        audit["corrected_tool"] = "stop"
        audit["corrected_arguments"] = {}
        audit["stop_applied"] = True
        audit["completion_satisfied"] = True
        audit["completion_mode"] = "collision_limited_safe_stop"
        audit["reason"] = "collision_limited_safe_stop"
        return output, audit

    # Before any meaningful translation, a blocked path is not a successful
    # approach. Preserve the original move so the exact simulator returns its
    # fail-closed action receipt instead of misreporting success.
    collision_audit["reason"] = "blocked_before_minimum_progress"
    audit["collision_rejection_deferred_to_exact_simulator"] = True
    return call, audit


def _literal_instruction(policy_input: str) -> str:
    if not isinstance(policy_input, str):
        raise TypeError("V3 navigation instruction must be text")
    prefix = "User navigation instruction: "
    stripped = policy_input.strip()
    first_line = stripped.splitlines()[0] if stripped else ""
    value = first_line[len(prefix) :].strip() if first_line.startswith(prefix) else stripped
    if not value or len(value) > 1024:
        raise ValueError("V3 navigation instruction is empty or too long")
    return value


def target_text_from_navigation_instruction(instruction: str) -> str | None:
    """Extract only a user-supplied target phrase with no scene inventory.

    The grammar is protocol-level: it recognizes action phrases but contains no
    object/category names.  Explicit metric moves, scan, and stop have no target.
    """

    literal = _SPACE.sub(" ", _literal_instruction(instruction)).strip()
    lower = literal.casefold()
    if re.fullmatch(
        r"(?:stop|scan(?: the room)?|look around|move (?:forward|backward) .+?)[.!]?",
        lower,
    ):
        return None
    # Update-after-scan instructions ground the final requested destination.
    update = re.search(
        r"\bthen\s+((?:move\s+closer\s+to|walk\s+toward|approach)\s+.+)$", literal, re.IGNORECASE
    )
    candidate = update.group(1) if update else literal
    candidate = _TRAILING_STOP.sub("", candidate).strip()
    candidate = _DIRECTION_SUFFIX.sub("", candidate).strip()
    patterns = (
        r"^go\s+around\s+.+?\s+and\s+stop\s+(?:beside|near)\s+(?:the\s+)?(.+?)[.!]?$",
        r"^(?:move\s+closer\s+to|walk\s+toward|move\s+toward|go\s+to|approach)\s+(?:the\s+)?(.+?)[.!]?$",
        r"^(?:turn\s+toward|turn\s+to\s+face|face|look\s+at)\s+(?:the\s+)?(.+?)[.!]?$",
    )
    for pattern in patterns:
        match = re.fullmatch(pattern, candidate, flags=re.IGNORECASE)
        if match:
            result = " ".join(match.group(1).strip().split()).rstrip(".!?")
            if not result or len(result) > 256:
                raise ValueError("V3 parsed target phrase is invalid")
            return result
    return None


def grounded_target_state(
    target_xyz_m: torch.Tensor,
    robot_state_features: torch.Tensor,
    available: torch.Tensor,
    *,
    room_size_m: Sequence[float],
) -> torch.Tensor:
    """Encode numeric world/relative target geometry without prose or labels."""

    targets = torch.as_tensor(target_xyz_m, dtype=torch.float32)
    states = torch.as_tensor(robot_state_features, dtype=torch.float32)
    valid = torch.as_tensor(available, dtype=torch.float32)
    if targets.ndim == 1:
        targets = targets.unsqueeze(0)
    if states.ndim == 1:
        states = states.unsqueeze(0)
    if valid.ndim == 0:
        valid = valid.unsqueeze(0)
    if targets.ndim != 2 or targets.shape[1] != 3:
        raise ValueError("V3 targets must have shape [B,3]")
    if states.ndim != 2 or states.shape != (len(targets), 18):
        raise ValueError("V3 robot_state_features must have shape [B,18]")
    if valid.shape != (len(targets),) or torch.any((valid != 0.0) & (valid != 1.0)):
        raise ValueError("V3 target availability must be a binary [B] tensor")
    room = torch.as_tensor(room_size_m, dtype=torch.float32)
    if room.shape != (3,) or torch.any(room <= 0.0) or not torch.isfinite(room).all():
        raise ValueError("V3 room_size_m must contain three finite positive values")
    if not torch.isfinite(targets).all() or not torch.isfinite(states).all():
        raise ValueError("V3 target state inputs contain NaN or infinity")
    minimum = torch.tensor([-room[0] / 2.0, -room[1] / 2.0, 0.0])
    maximum = torch.tensor([room[0] / 2.0, room[1] / 2.0, room[2]])
    target_normalized = 2.0 * (targets - minimum) / (maximum - minimum) - 1.0
    robot_normalized = states[:, :3]
    relative_normalized = (target_normalized - robot_normalized) * 0.5
    robot_world = (robot_normalized + 1.0) * 0.5 * (maximum - minimum) + minimum
    delta = targets - robot_world
    distance = torch.linalg.vector_norm(delta[:, :2], dim=-1) / torch.linalg.vector_norm(room[:2])
    desired_yaw = torch.atan2(-delta[:, 0], delta[:, 1])
    robot_yaw = torch.atan2(states[:, 3], states[:, 4])
    error = desired_yaw - robot_yaw
    mask = valid.unsqueeze(1)
    encoded = torch.cat(
        (
            mask,
            target_normalized * mask,
            relative_normalized * mask,
            distance.unsqueeze(1) * mask,
            torch.sin(error).unsqueeze(1) * mask,
            torch.cos(error).unsqueeze(1) * mask,
        ),
        dim=1,
    )
    if encoded.shape != (len(targets), TARGET_STATE_DIM) or not torch.isfinite(encoded).all():
        raise RuntimeError("V3 grounded target state is invalid")
    return encoded


class GroundedContinuousNavigationControllerV3(nn.Module):
    """All-scene controller with an explicit numeric grounded-target branch."""

    def __init__(self, hidden_size: int, *, model_dim: int = 128) -> None:
        super().__init__()
        self.hidden_size = _positive_int(hidden_size, "hidden_size")
        self.model_dim = _positive_int(model_dim, "model_dim", 2048)
        self.scene_key = nn.Linear(self.hidden_size, self.model_dim, bias=False)
        self.scene_value = nn.Linear(self.hidden_size, self.model_dim, bias=False)
        self.instruction_query = nn.Linear(self.hidden_size, self.model_dim, bias=False)
        self.instruction_value = nn.Linear(self.hidden_size, self.model_dim)
        self.robot_value = nn.Linear(self.hidden_size, self.model_dim)
        self.target_value = nn.Sequential(
            nn.Linear(TARGET_STATE_DIM, self.model_dim),
            nn.GELU(),
            nn.LayerNorm(self.model_dim),
            nn.Linear(self.model_dim, self.model_dim),
        )
        self.fusion = nn.Sequential(
            nn.LayerNorm(self.model_dim * 5),
            nn.Linear(self.model_dim * 5, self.model_dim * 2),
            nn.GELU(),
            nn.Linear(self.model_dim * 2, self.model_dim),
            nn.GELU(),
        )
        self.action_head = nn.Linear(self.model_dim, len(ACTION_NAMES))
        self.argument_head = nn.Linear(self.model_dim, len(ACTION_NAMES))

    def forward(
        self,
        scene_prefix: torch.Tensor,
        robot_tokens: torch.Tensor,
        instruction_embedding: torch.Tensor,
        target_state: torch.Tensor,
        *,
        scene_batch_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if scene_prefix.ndim != 3 or scene_prefix.shape[-1] != self.hidden_size:
            raise ValueError("V3 scene_prefix shape differs")
        if robot_tokens.ndim != 3 or robot_tokens.shape[-1] != self.hidden_size:
            raise ValueError("V3 robot_tokens shape differs")
        if instruction_embedding.ndim == 3:
            instruction_embedding = instruction_embedding.mean(dim=1)
        batch = robot_tokens.shape[0]
        if instruction_embedding.shape != (batch, self.hidden_size):
            raise ValueError("V3 instruction embedding shape differs")
        if target_state.shape != (batch, TARGET_STATE_DIM):
            raise ValueError("V3 target_state shape differs")
        if scene_batch_indices is None:
            if scene_prefix.shape[0] != batch:
                raise ValueError("V3 scene and action batches differ")
        elif (
            scene_batch_indices.shape != (batch,)
            or scene_batch_indices.dtype != torch.long
            or torch.any(scene_batch_indices < 0)
            or torch.any(scene_batch_indices >= scene_prefix.shape[0])
        ):
            raise ValueError("V3 scene batch indices are invalid")
        if scene_prefix.shape[1] < 3 or robot_tokens.shape[1] < 1:
            raise ValueError("V3 scene or robot token sequence is empty")
        if not all(
            torch.isfinite(value).all()
            for value in (scene_prefix, robot_tokens, instruction_embedding, target_state)
        ):
            raise ValueError("V3 inputs contain NaN or infinity")

        scene = torch.nn.functional.layer_norm(scene_prefix.float(), (self.hidden_size,))
        robot = torch.nn.functional.layer_norm(robot_tokens.float(), (self.hidden_size,))
        instruction = torch.nn.functional.layer_norm(
            instruction_embedding.float(), (self.hidden_size,)
        )
        keys = self.scene_key(scene)
        values = self.scene_value(scene)
        if scene_batch_indices is not None:
            selected = scene_batch_indices.to(keys.device)
            keys = keys[selected]
            values = values[selected]
        query = self.instruction_query(instruction)
        attention = torch.softmax(
            torch.einsum("bd,bsd->bs", query, keys) / math.sqrt(self.model_dim), dim=-1
        )
        attended = torch.einsum("bs,bsd->bd", attention, values)
        fused = self.fusion(
            torch.cat(
                (
                    attended,
                    values.mean(dim=1),
                    self.instruction_value(instruction),
                    self.robot_value(robot).mean(dim=1),
                    self.target_value(target_state.float()),
                ),
                dim=-1,
            )
        )
        logits = self.action_head(fused)
        arguments = torch.tanh(self.argument_head(fused))
        if not torch.isfinite(logits).all() or not torch.isfinite(arguments).all():
            raise RuntimeError("V3 controller output contains NaN or infinity")
        return logits, arguments


def _validated_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(metadata)
    integers = {
        name: _positive_int(value.get(name), name)
        for name in (
            "hidden_size",
            "model_dim",
            "scene_token_count",
            "robot_token_count",
            "train_scene_count",
            "validation_scene_count",
        )
    }
    room = value.get("room_size_m")
    if not isinstance(room, list) or len(room) != 3:
        raise ValueError("V3 room size metadata differs")
    room_values = [_finite_positive(item, "room_size_m") for item in room]
    digest = value.get("training_dataset_sha256")
    weights_digest = value.get("weights_sha256")
    if (
        set(value) != _METADATA_FIELDS
        or value.get("schema_version") != SCHEMA_VERSION
        or value.get("architecture") != ARCHITECTURE
        or value.get("target_state_dim") != TARGET_STATE_DIM
        or value.get("action_names") != list(ACTION_NAMES)
        or not isinstance(value.get("model_id"), str)
        or not value["model_id"]
        or not isinstance(value.get("model_revision"), str)
        or not value["model_revision"]
        or value.get("grounding_feature_start") != GEMMA4_PROJECTED_START
        or value.get("grounding_feature_dim") != GEMMA4_PROJECTED_DIM
        or value.get("task_trained") is not True
        or not isinstance(digest, str)
        or _SHA256.fullmatch(digest) is None
        or not isinstance(weights_digest, str)
        or _SHA256.fullmatch(weights_digest) is None
        or any(
            value.get(field) is not True
            for field in (
                "scene_splits_disjoint",
                "complete_scene_prefix_required",
                "question_independent_static_scene_prefix_required",
                "every_scene_token_processed",
                "numeric_robot_tokens_required",
                "continuous_semantic_grounding_required",
                "all_map_voxels_scored_for_grounding",
                "query_dependent_grounding_navigation_only",
                "collision_interlock_required",
            )
        )
        or value.get("environmental_text_inputs") != []
        or value.get("oracle_inputs_at_runtime") is not False
        or value.get("runtime_required_files") != ["policy.safetensors", "runtime_metadata.json"]
    ):
        raise ValueError("V3 checkpoint contract is not inference safe")
    value.update(integers)
    value["room_size_m"] = room_values
    value["max_turn_degrees"] = _finite_positive(value.get("max_turn_degrees"), "max_turn_degrees")
    value["max_move_m"] = _finite_positive(value.get("max_move_m"), "max_move_m")
    return value


def save_navigation_policy_v3_checkpoint(
    destination: str | Path,
    controller: GroundedContinuousNavigationControllerV3,
    *,
    runtime_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    root = _rooted(destination)
    _reject_runtime_path(root)
    if root.exists():
        raise FileExistsError(f"V3 checkpoint already exists: {root}")
    root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{root.name}.", dir=root.parent))
    try:
        weights_path = temporary / "policy.safetensors"
        save_file(
            {
                name: tensor.detach().cpu().contiguous()
                for name, tensor in controller.state_dict().items()
            },
            str(weights_path),
        )
        metadata = {
            **dict(runtime_metadata),
            "schema_version": SCHEMA_VERSION,
            "architecture": ARCHITECTURE,
            "hidden_size": controller.hidden_size,
            "model_dim": controller.model_dim,
            "target_state_dim": TARGET_STATE_DIM,
            "action_names": list(ACTION_NAMES),
            "weights_sha256": _sha256_file(weights_path),
        }
        metadata = _validated_metadata(metadata)
        (temporary / "runtime_metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, root)
        return {**metadata, "checkpoint": str(root)}
    finally:
        if temporary.exists():
            for child in temporary.iterdir():
                child.unlink(missing_ok=True)
            temporary.rmdir()


def load_navigation_policy_v3_checkpoint(
    checkpoint: str | Path,
    *,
    expected_hidden_size: int,
    expected_model_id: str,
    expected_model_revision: str,
    device: torch.device | str = "cpu",
    audit: FileAccessAudit | None = None,
) -> tuple[GroundedContinuousNavigationControllerV3, dict[str, Any]]:
    root = _rooted(checkpoint)
    _reject_runtime_path(root)
    if not root.is_dir() or {entry.name for entry in root.iterdir()} != _FILES:
        raise ValueError("V3 checkpoint must contain exactly two files")
    weights_path = root / "policy.safetensors"
    metadata_path = root / "runtime_metadata.json"
    if any(path.is_symlink() or not path.is_file() for path in (weights_path, metadata_path)):
        raise ValueError("V3 checkpoint entries must be regular files")
    if audit is not None:
        audit.record(weights_path)
        audit.record(metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise TypeError("V3 runtime metadata must be an object")
    metadata = _validated_metadata(metadata)
    if _sha256_file(weights_path) != metadata["weights_sha256"]:
        raise ValueError("V3 checkpoint weights hash differs")
    if (
        metadata["hidden_size"] != expected_hidden_size
        or metadata["model_id"] != expected_model_id
        or metadata["model_revision"] != expected_model_revision
    ):
        raise ValueError("V3 checkpoint local-model binding differs")
    controller = GroundedContinuousNavigationControllerV3(
        metadata["hidden_size"], model_dim=metadata["model_dim"]
    )
    state = load_file(str(weights_path), device="cpu")
    expected = controller.state_dict()
    if set(state) != set(expected) or any(
        state[name].shape != expected[name].shape for name in expected
    ):
        raise ValueError("V3 checkpoint tensor inventory differs")
    if any(not torch.isfinite(tensor).all() for tensor in state.values()):
        raise ValueError("V3 checkpoint contains NaN or infinity")
    controller.load_state_dict(state, strict=True)
    controller.to(device).eval()
    for parameter in controller.parameters():
        parameter.requires_grad_(False)
    return controller, metadata


def _active_map_path(runtime: Any) -> Path:
    updater = getattr(runtime, "map_updater", None)
    if updater is None:
        raise TypeError("V3 runtime has no semantic map updater")
    persistent = Path(updater.persistent_map_path)
    base = Path(updater.base_map_path)
    selected = persistent if persistent.is_file() else base
    _reject_runtime_path(_rooted(selected))
    if not selected.is_file():
        raise FileNotFoundError("V3 active semantic map is unavailable")
    return selected


def _collision_safe_or_stop(runtime: Any, call: dict[str, Any]) -> dict[str, Any]:
    name = call["tool"]
    if name not in {"move_forward", "move_backward"}:
        return call
    simulator = getattr(runtime, "simulator", None)
    if simulator is None:
        return {"tool": "stop", "arguments": {}}
    start = np.asarray(simulator.state.position_xy_m, dtype=np.float64)
    yaw = float(simulator.state.body_yaw_degrees)
    distance = float(call["arguments"]["distance_meters"])
    direction = np.asarray(
        [-math.sin(math.radians(yaw)), math.cos(math.radians(yaw))], dtype=np.float64
    )
    if name == "move_backward":
        direction *= -1.0
    if simulator.collision_map.segment_check(start, start + distance * direction).collision:
        return {"tool": "stop", "arguments": {}}
    return call


class SemanticGroundedActionBackendV3:
    """Runtime backend that grounds each embodied target against all active voxels."""

    def __init__(
        self,
        runtime: Any,
        controller: GroundedContinuousNavigationControllerV3,
        metadata: Mapping[str, Any],
        config: Mapping[str, Any],
        *,
        text_encoder: ContinuousTextEncoder | None = None,
    ) -> None:
        self.runtime = runtime
        self.controller = controller.eval()
        self.metadata = _validated_metadata(metadata)
        self.config = dict(config)
        prefix_refresher = getattr(runtime, "prefix_refresher", None)
        wrapped = getattr(prefix_refresher, "runtime", None)
        self.base = getattr(wrapped, "base", wrapped)
        if self.base is None or getattr(self.base, "language", None) is None:
            raise TypeError("V3 backend requires a loaded local language model")
        if self.base.language.hidden_size != controller.hidden_size:
            raise ValueError("V3 controller width differs from local Gemma")
        self.text_encoder = text_encoder or GemmaProjectedTextEncoder.from_config(config)
        if self.text_encoder.output_dim != int(self.metadata["grounding_feature_dim"]):
            raise ValueError("V3 grounding text encoder width differs")
        robot_config = self.config.get("robot")
        if not isinstance(robot_config, Mapping):
            robot_config = {}
        self.alignment_deadband_degrees = _finite_positive(
            robot_config.get(
                "face_alignment_deadband_degrees",
                _DEFAULT_ALIGNMENT_DEADBAND_DEGREES,
            ),
            "face_alignment_deadband_degrees",
        )
        self.stalled_turn_degrees = _finite_positive(
            robot_config.get(
                "face_alignment_stalled_turn_degrees",
                _DEFAULT_STALLED_TURN_DEGREES,
            ),
            "face_alignment_stalled_turn_degrees",
        )
        self.approach_heading_deadband_degrees = _finite_positive(
            robot_config.get(
                "approach_heading_deadband_degrees",
                _DEFAULT_APPROACH_HEADING_DEADBAND_DEGREES,
            ),
            "approach_heading_deadband_degrees",
        )
        self.approach_target_standoff_m = _finite_positive(
            robot_config.get(
                "approach_target_standoff_m",
                _DEFAULT_APPROACH_TARGET_STANDOFF_M,
            ),
            "approach_target_standoff_m",
        )
        self.approach_minimum_progress_m = _finite_positive(
            robot_config.get(
                "approach_minimum_progress_m",
                _DEFAULT_APPROACH_MINIMUM_PROGRESS_M,
            ),
            "approach_minimum_progress_m",
        )
        self.approach_minimum_safe_step_m = _finite_positive(
            robot_config.get(
                "approach_minimum_safe_step_m",
                _DEFAULT_APPROACH_MINIMUM_SAFE_STEP_M,
            ),
            "approach_minimum_safe_step_m",
        )
        if (
            self.alignment_deadband_degrees >= float(self.metadata["max_turn_degrees"])
            or self.stalled_turn_degrees >= float(self.metadata["max_turn_degrees"])
            or self.approach_heading_deadband_degrees
            >= float(self.metadata["max_turn_degrees"])
            or self.approach_minimum_progress_m > float(self.metadata["max_move_m"])
            or self.approach_minimum_safe_step_m > float(self.metadata["max_move_m"])
        ):
            raise ValueError("V3 live convergence thresholds exceed bounded actions")
        self.last_grounding: dict[str, Any] | None = None
        self._approach_instruction: str | None = None
        self._approach_initial_position_xy_m: tuple[float, float] | None = None

    def numeric_alignment_interlock_summary(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "runtime_interlock_version": RUNTIME_INTERLOCK_VERSION,
            "action_modes": ["face", "turn_toward", "turn_to_face", "look_at"],
            "explicit_terminal_stop_required": True,
            "deadband_degrees": self.alignment_deadband_degrees,
            "stalled_turn_degrees": self.stalled_turn_degrees,
            "remaining_angle_source": "continuous_target_xyz_and_numeric_robot_yaw",
            "bounded_by_max_turn_degrees": True,
            "fresh_grounding_required_before_stop": True,
            "approach": {
                "action_modes": ["move_closer", "approach", "move_toward"],
                "compound_scan_then_approach_supported": True,
                "explicit_terminal_stop_required": True,
                "heading_deadband_degrees": self.approach_heading_deadband_degrees,
                "target_standoff_m": self.approach_target_standoff_m,
                "minimum_progress_m": self.approach_minimum_progress_m,
                "minimum_safe_step_m": self.approach_minimum_safe_step_m,
                "fresh_grounding_required_before_stop": True,
                "numeric_translation_required_before_stop": True,
                "collision_limited_safe_stop": True,
            },
            "environmental_text_inputs": [],
            "oracle_inputs_at_runtime": False,
        }

    def _ground(
        self,
        target_text: str | None,
        *,
        context: ContinuousActionContext | None = None,
    ) -> tuple[torch.Tensor, ContinuousSemanticGrounding | None]:
        simulator = self.runtime.simulator
        room = self.metadata["room_size_m"]
        minimum = torch.tensor([-room[0] / 2.0, -room[1] / 2.0, 0.0])
        maximum = torch.tensor([room[0] / 2.0, room[1] / 2.0, room[2]])
        state_features = (
            robot_state_vector(simulator.numeric_state(), minimum, maximum)
            if context is None
            else context.state_features
        )
        if target_text is None:
            self.last_grounding = {
                "target_available": False,
                "query_embedding_sha256": None,
                "map_sha256": (
                    self.runtime.prefix_binding().get("map_sha256")
                    if context is None
                    else context.map_sha256
                ),
                "continuous_context_verified": context is not None,
            }
            if context is not None:
                self.last_grounding.update(
                    {
                        "active_prefix_sha256": context.binding[
                            "active_prefix_sha256"
                        ],
                        "scene_prefix_sha256": context.binding[
                            "scene_prefix_sha256"
                        ],
                        "robot_state_sha256": context.binding[
                            "robot_state_sha256"
                        ],
                        "robot_tokens_sha256": context.binding[
                            "robot_tokens_sha256"
                        ],
                    }
                )
            return (
                grounded_target_state(
                    torch.zeros(3), state_features, torch.tensor(0.0), room_size_m=room
                ),
                None,
            )
        grounder = ContinuousSemanticTargetGrounder(
            _active_map_path(self.runtime),
            self.text_encoder,
            room_size_m=room,
            feature_start=int(self.metadata["grounding_feature_start"]),
            feature_dim=int(self.metadata["grounding_feature_dim"]),
        )
        grounding = grounder.ground(target_text)
        if context is None:
            if grounding.scored_voxels != len(grounder.xyz):
                raise RuntimeError("V3 grounding did not score the complete voxel map")
        else:
            require_grounding_map_binding(
                context,
                grounding_map_sha256=grounding.map_sha256,
                scored_voxels=grounding.scored_voxels,
                available_voxels=len(grounder.xyz),
            )
        if grounder.scene_id != simulator.state.scene_id:
            raise RuntimeError("V3 grounding map scene differs from robot state")
        target = grounded_target_state(
            torch.tensor(grounding.target_xyz_m),
            state_features,
            torch.tensor(1.0),
            room_size_m=room,
        )
        self.last_grounding = {
            "target_available": True,
            "target_xyz_m": list(grounding.target_xyz_m),
            "target_state_sha256": hashlib.sha256(
                target.detach().cpu().contiguous().numpy().tobytes()
            ).hexdigest(),
            "query_embedding_sha256": grounding.query_embedding_sha256,
            "map_sha256": grounding.map_sha256,
            "scored_voxels": grounding.scored_voxels,
            "eligible_voxels": grounding.eligible_voxels,
            "continuous_context_verified": context is not None,
        }
        if context is not None:
            self.last_grounding.update(
                {
                    "active_prefix_sha256": context.binding["active_prefix_sha256"],
                    "scene_prefix_sha256": context.binding["scene_prefix_sha256"],
                    "robot_state_sha256": context.binding["robot_state_sha256"],
                    "robot_tokens_sha256": context.binding["robot_tokens_sha256"],
                }
            )
        return target, grounding

    @torch.inference_mode()
    def generate(self, instruction: str, *, correction_code: str | None) -> GeneratedToolProposal:
        del correction_code
        context = capture_continuous_action_context(
            self.runtime,
            self.metadata["room_size_m"],
        )
        active = context.active_prefix
        binding = context.binding
        observed_hash = prefix_sha256(active)
        if binding.get("active_prefix_sha256") != observed_hash:
            raise RuntimeError("V3 active prefix differs from runtime binding")
        scene, robot = split_active_prefix(
            active,
            scene_token_count=int(self.metadata["scene_token_count"]),
            robot_token_count=int(self.metadata["robot_token_count"]),
        )
        literal = _literal_instruction(instruction)
        if _terminal_approach_requested(literal):
            previous = (
                None
                if self.last_grounding is None
                else self.last_grounding.get("numeric_approach_interlock")
            )
            previous_complete = bool(
                isinstance(previous, Mapping)
                and previous.get("completion_satisfied") is True
            )
            if self._approach_instruction != literal or previous_complete:
                current_xy = context.numeric_state.position_m[:2]
                self._approach_instruction = literal
                self._approach_initial_position_xy_m = (
                    float(current_xy[0]),
                    float(current_xy[1]),
                )
        target_text = target_text_from_navigation_instruction(literal)
        target_state, _grounding = self._ground(target_text, context=context)
        encoded = self.base.language.tokenizer(
            literal, add_special_tokens=False, return_tensors="pt"
        )
        token_ids = encoded.get("input_ids") if isinstance(encoded, Mapping) else None
        if not isinstance(token_ids, torch.Tensor) or token_ids.ndim != 2 or not token_ids.numel():
            raise ValueError("V3 tokenizer returned no instruction tokens")
        embedding_layer = self.base.language.model.get_input_embeddings()
        instruction_embedding = embedding_layer(token_ids.to(embedding_layer.weight.device))
        device = next(self.controller.parameters()).device
        logits, arguments = self.controller(
            scene.to(device),
            robot.to(device),
            instruction_embedding.float().mean(dim=1).to(device),
            target_state.to(device),
        )
        action_index = int(torch.argmax(logits[0]).item())
        call = tool_call_from_prediction(
            action_index,
            float(arguments[0, action_index].item()),
            max_turn_degrees=float(self.metadata["max_turn_degrees"]),
            max_move_m=float(self.metadata["max_move_m"]),
        )
        call, convergence = apply_numeric_alignment_interlock(
            literal,
            target_state,
            call,
            target_xyz_m=(
                None if _grounding is None else _grounding.target_xyz_m
            ),
            robot_position_xy_m=context.numeric_state.position_m[:2],
            robot_yaw_degrees=context.numeric_state.body_yaw_degrees,
            max_turn_degrees=float(self.metadata["max_turn_degrees"]),
            deadband_degrees=self.alignment_deadband_degrees,
            stalled_turn_degrees=self.stalled_turn_degrees,
        )
        if self.last_grounding is not None:
            self.last_grounding["numeric_alignment_interlock"] = convergence
        if self._approach_initial_position_xy_m is None:
            current_xy = context.numeric_state.position_m[:2]
            initial_xy = (float(current_xy[0]), float(current_xy[1]))
        else:
            initial_xy = self._approach_initial_position_xy_m
        call, approach = apply_numeric_approach_interlock(
            literal,
            target_state,
            call,
            target_xyz_m=(None if _grounding is None else _grounding.target_xyz_m),
            initial_robot_position_xy_m=initial_xy,
            robot_position_xy_m=context.numeric_state.position_m[:2],
            robot_yaw_degrees=context.numeric_state.body_yaw_degrees,
            max_turn_degrees=float(self.metadata["max_turn_degrees"]),
            max_move_m=float(self.metadata["max_move_m"]),
            heading_deadband_degrees=self.approach_heading_deadband_degrees,
            target_standoff_m=self.approach_target_standoff_m,
            minimum_progress_m=self.approach_minimum_progress_m,
            stalled_turn_degrees=self.stalled_turn_degrees,
        )
        if (
            approach["terminal_approach_requested"] is True
            and approach["target_available"] is True
        ):
            call, approach = apply_collision_limited_approach_interlock(
                call,
                approach,
                collision_map=self.runtime.simulator.collision_map,
                robot_position_xy_m=context.numeric_state.position_m[:2],
                robot_yaw_degrees=context.numeric_state.body_yaw_degrees,
                minimum_safe_step_m=self.approach_minimum_safe_step_m,
            )
        else:
            call = _collision_safe_or_stop(self.runtime, call)
        if self.last_grounding is not None:
            self.last_grounding["numeric_approach_interlock"] = approach
        scene_hash = binding.get("scene_prefix_sha256")
        robot_hash = binding.get("robot_tokens_sha256")
        return GeneratedToolProposal(
            text=json.dumps(call, sort_keys=True, separators=(",", ":"), allow_nan=False),
            active_prefix_sha256=observed_hash,
            scene_prefix_sha256=scene_hash if isinstance(scene_hash, str) else "",
            robot_tokens_sha256=robot_hash if isinstance(robot_hash, str) else None,
            local_inference=True,
            used_continuous_scene_prefix=True,
            used_continuous_robot_tokens=True,
            training_status=TRAINING_STATUS,
        )


__all__ = [
    "ARCHITECTURE",
    "RUNTIME_INTERLOCK_VERSION",
    "SCHEMA_VERSION",
    "TARGET_STATE_DIM",
    "TRAINING_STATUS",
    "GroundedContinuousNavigationControllerV3",
    "SemanticGroundedActionBackendV3",
    "apply_collision_limited_approach_interlock",
    "apply_numeric_alignment_interlock",
    "apply_numeric_approach_interlock",
    "grounded_target_state",
    "load_navigation_policy_v3_checkpoint",
    "save_navigation_policy_v3_checkpoint",
    "target_text_from_navigation_instruction",
]
