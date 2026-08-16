"""Natural-language navigation over one official MCP stdio session.

This is the deliberately small, honest embodied integration path.  A user's
``face ... then stop`` instruction is reduced to its user-supplied target
phrase by the label-free V3 action grammar.  Selectively loaded local Gemma 4
token rows ground that phrase against *every* voxel in the active continuous
semantic map.  The V3 numeric alignment interlock then chooses bounded turns
from target XYZ and numeric robot yaw until a fresh grounding is inside the
configured deadband, followed by ``stop``.

The legacy :func:`run_face_instruction` API remains a finite, safety-latched
single-goal proof.  :class:`PersistentMCPConversationSession` adds the useful
interactive surface: repeated face, approach, scan, and state turns share one
stdio subprocess and one persistent continuous map.  Reaching a face/approach
goal leaves the kinematic robot stationary without latching the episode-wide
``stop`` flag; an explicit standalone ``stop`` remains safety-latched.

Every scan, turn, movement, state query, and explicit stop crosses an official
Python MCP SDK stdio subprocess
boundary.  The subprocess owns RGB-D rendering, full-image vision encoding,
map fusion, scene-prefix rebuilding, and numeric robot-token rebuilding.  Tool
results contain only the strict numeric/protocol receipt.  This module does
not use Gemma function calling and does not claim to run the learned V3 action
head: a neutral stop sentinel is supplied to the independently tested numeric
alignment interlock so the integration does not pretend that unavailable
prefix tensors can be reconstructed from MCP hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import re
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch
from mcp import StdioServerParameters

from semantic_3d_chat.chat.file_audit import FileAccessAudit
from semantic_3d_chat.config import PROJECT_ROOT, load_config, project_path
from semantic_3d_chat.robot.mcp_stdio_runtime import (
    MCPConversationRuntime,
    MCPStdioToolClient,
    validate_numeric_tool_receipt,
)
from semantic_3d_chat.robot.navigation_policy_v3 import (
    TARGET_STATE_DIM,
    apply_collision_limited_approach_interlock,
    apply_numeric_alignment_interlock,
    apply_numeric_approach_interlock,
    grounded_target_state,
    target_text_from_navigation_instruction,
)
from semantic_3d_chat.robot.semantic_agent import (
    ContinuousSemanticTargetGrounder,
    ContinuousTextEncoder,
    GemmaProjectedTextEncoder,
)
from semantic_3d_chat.robot.state_encoder import NumericRobotState, robot_state_vector

SCHEMA: Final[str] = "semantic_3d_chat.conversational_mcp_face.v1"
SESSION_SCHEMA: Final[str] = "semantic_3d_chat.conversational_mcp_session.v1"
POLICY_NAME: Final[str] = "selective_gemma_all_voxel_v3_numeric_alignment"
SESSION_POLICY_NAME: Final[str] = "selective_gemma_all_voxel_v3_numeric_navigation"
_BLOCKED_COMPONENTS: Final[frozenset[str]] = frozenset(
    {"oracle", "qa", "training", "scorer_only", "scorer-only"}
)
_TERMINAL_FACE = re.compile(
    r"^(?:face|turn\s+toward|turn\s+to\s+face|look\s+at)\s+.+?"
    r"(?:,?\s+then\s+stop|\s+and\s+stop)[.!]?\s*$",
    re.IGNORECASE,
)
_TERMINAL_SUFFIX = re.compile(
    r"(?:,?\s+then\s+stop|\s+and\s+stop)[.!]?\s*$",
    re.IGNORECASE,
)
_INTERACTIVE_FACE = re.compile(
    r"^(?:face|turn\s+toward|turn\s+to\s+face|look\s+at|look\s+toward)\s+"
    r"(?:the\s+)?(.+?)[.!]?\s*$",
    re.IGNORECASE,
)
_INTERACTIVE_APPROACH = re.compile(
    r"^(?:move\s+closer\s+to|walk\s+toward|move\s+toward|go\s+to|approach)\s+"
    r"(?:the\s+)?(.+?)[.!]?\s*$",
    re.IGNORECASE,
)
_INTERACTIVE_STATE = re.compile(
    r"^(?:get|show)(?:\s+the)?\s+robot\s+state[.!]?\s*$|^robot\s+state[.!]?\s*$",
    re.IGNORECASE,
)
_INTERACTIVE_SCAN = re.compile(
    r"^(?:scan|scan\s+the\s+room|look\s+around)[.!]?\s*$",
    re.IGNORECASE,
)
_INTERACTIVE_STOP = re.compile(r"^stop[.!]?\s*$", re.IGNORECASE)


@dataclass(frozen=True)
class InteractiveMCPCommand:
    """One label-free command parsed only from the user's own text."""

    kind: str
    target_text: str | None = None
    terminal_instruction: str | None = None


def parse_interactive_mcp_command(text: str) -> InteractiveMCPCommand:
    """Parse the deliberately bounded interactive MCP action language.

    The grammar contains action phrases but no object/category vocabulary.  A
    target is copied only from the user's instruction and is never sent to the
    MCP tool server; only the resulting numerical action arguments cross stdio.
    """

    if not isinstance(text, str) or not text.strip():
        raise ValueError("Interactive MCP instruction must be non-empty text")
    normalized = " ".join(text.strip().split())
    if _INTERACTIVE_STOP.fullmatch(normalized):
        return InteractiveMCPCommand("stop")
    if _INTERACTIVE_SCAN.fullmatch(normalized):
        return InteractiveMCPCommand("scan")
    if _INTERACTIVE_STATE.fullmatch(normalized):
        return InteractiveMCPCommand("state")

    action = _TERMINAL_SUFFIX.sub("", normalized).strip()
    for kind, pattern, canonical in (
        ("face", _INTERACTIVE_FACE, "Face {target}, then stop."),
        ("approach", _INTERACTIVE_APPROACH, "Approach {target}, then stop."),
    ):
        match = pattern.fullmatch(action)
        if match is None:
            continue
        target = " ".join(match.group(1).strip().split()).rstrip(".!?")
        if not target or len(target) > 256:
            raise ValueError("Interactive MCP target phrase is invalid")
        terminal = canonical.format(target=target)
        parsed = target_text_from_navigation_instruction(terminal)
        if parsed != target:
            raise RuntimeError("Interactive and V3 target grammars disagree")
        return InteractiveMCPCommand(kind, target, terminal)
    raise ValueError(
        "Supported commands are face/look/turn toward TARGET, approach/move closer/"
        "walk toward TARGET, scan, get robot state, and stop"
    )


def _rooted(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    return Path(os.path.abspath(candidate if candidate.is_absolute() else PROJECT_ROOT / candidate))


def _server_parameters(
    *,
    python_executable: str | Path,
    config: str | Path,
    scene_id: str,
    base_checkpoint: str | Path,
    control_checkpoint: str | Path,
    control_runtime_config: str | Path,
    runtime_asset: str | Path,
    robot_state_checkpoint: str | Path,
    persistent_map: str | Path,
    audit_report: str | Path,
) -> StdioServerParameters:
    """Construct the production stdio command without importing evaluators."""

    executable = Path(python_executable)
    if not executable.is_absolute():
        executable = (Path.cwd() / executable).absolute()
    child_environment = dict(os.environ)
    source_root = str(PROJECT_ROOT / "src")
    inherited_pythonpath = child_environment.get("PYTHONPATH")
    child_environment["PYTHONPATH"] = (
        source_root
        if not inherited_pythonpath
        else os.pathsep.join((source_root, inherited_pythonpath))
    )
    child_environment.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    return StdioServerParameters(
        command=str(executable),
        args=[
            "-m",
            "semantic_3d_chat.mcp_server.server",
            "--config",
            str(_rooted(config)),
            "--scene",
            scene_id,
            "--checkpoint",
            str(_rooted(base_checkpoint)),
            "--control-checkpoint",
            str(_rooted(control_checkpoint)),
            "--control-runtime-config",
            str(_rooted(control_runtime_config)),
            "--runtime-asset",
            str(_rooted(runtime_asset)),
            "--robot-state-checkpoint",
            str(_rooted(robot_state_checkpoint)),
            "--persistent-map",
            str(_rooted(persistent_map)),
            "--audit-report",
            str(_rooted(audit_report)),
            "--transport",
            "stdio",
        ],
        cwd=PROJECT_ROOT,
        env=child_environment,
    )


def _numeric_protocol_violations(receipts: Sequence[Mapping[str, Any]]) -> list[str]:
    """Inventory-free proof that tool outputs contain only the strict schema."""

    violations: list[str] = []
    for index, receipt in enumerate(receipts):
        try:
            validate_numeric_tool_receipt(receipt, require_continuous_binding=True)
        except (RuntimeError, TypeError, ValueError):
            violations.append(f"receipt_{index}:strict_numeric_schema")
    return violations


def _safe_input(path: str | Path, *, purpose: str, kind: str) -> Path:
    candidate = _rooted(path)
    current = Path(candidate.anchor)
    for component in candidate.parts[1:]:
        current /= component
        if current.is_symlink():
            raise ValueError(f"{purpose} cannot use a symbolic-link path")
    if _BLOCKED_COMPONENTS.intersection(component.casefold() for component in candidate.parts):
        raise ValueError(f"{purpose} cannot enter environmental supervision trees")
    if kind == "file" and not candidate.is_file():
        raise FileNotFoundError(f"{purpose} is unavailable: {candidate}")
    if kind == "directory" and not candidate.is_dir():
        raise FileNotFoundError(f"{purpose} is unavailable: {candidate}")
    if kind not in {"file", "directory"}:
        raise ValueError("input kind must be file or directory")
    return candidate


def _atomic_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    destination = _rooted(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    finally:
        Path(temporary_name).unlink(missing_ok=True)
    return destination


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _client_audit() -> FileAccessAudit:
    return FileAccessAudit(
        [
            PROJECT_ROOT / "data" / "oracle",
            PROJECT_ROOT / "data" / "qa",
            PROJECT_ROOT / "data_gemma4" / "training",
            PROJECT_ROOT / "reports" / "gemma4" / "scorer_only",
        ],
        forbidden_component_names=_BLOCKED_COMPONENTS,
        block_forbidden=True,
    )


def _numeric_state(runtime: MCPConversationRuntime) -> NumericRobotState:
    state = runtime.simulator.state
    movement = np.asarray(state.last_movement_delta_m, dtype=np.float64)
    velocity = np.asarray(state.linear_velocity_xy_m, dtype=np.float64)
    if movement.shape != (3,) or velocity.shape != (2,):
        raise RuntimeError("Remote MCP numeric robot state has an invalid shape")
    return NumericRobotState(
        position_m=(float(state.position_xy_m[0]), float(state.position_xy_m[1]), 0.0),
        body_yaw_degrees=float(state.body_yaw_degrees),
        camera_yaw_degrees=float(state.camera_yaw_degrees),
        pitch_degrees=float(state.pitch_degrees),
        linear_velocity_xy_m=(float(velocity[0]), float(velocity[1])),
        angular_velocity_degrees=float(state.angular_velocity_degrees),
        collision=bool(state.collision),
        last_movement_delta_m=tuple(float(value) for value in movement),
        scan_coverage=float(state.scan_coverage),
        stopped=bool(state.stopped),
    )


def _active_map_path(runtime: MCPConversationRuntime) -> Path:
    persistent = Path(runtime.map_updater.persistent_map_path)
    base = Path(runtime.map_updater.base_map_path)
    selected = persistent if persistent.is_file() else base
    return _safe_input(selected, purpose="active continuous semantic map", kind="file")


def _binding_transition(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    *,
    observation_required: bool,
) -> dict[str, Any]:
    before_version = int(before["map_version"])
    after_version = int(after["map_version"])
    map_advanced = after_version == before_version + 1
    scene_prefix_changed = before["scene_prefix_sha256"] != after["scene_prefix_sha256"]
    map_changed = before["map_sha256"] != after["map_sha256"]
    active_changed = before["active_prefix_sha256"] != after["active_prefix_sha256"]
    passed = (
        map_advanced and map_changed and scene_prefix_changed and active_changed
        if observation_required
        else after_version == before_version and active_changed
    )
    return {
        "observation_required": observation_required,
        "before_map_version": before_version,
        "after_map_version": after_version,
        "map_advanced_exactly_once": map_advanced,
        "map_sha256_changed": map_changed,
        "scene_prefix_sha256_changed": scene_prefix_changed,
        "active_prefix_sha256_changed": active_changed,
        "passed": passed,
    }


def _stable_binding_transition(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> dict[str, Any]:
    """Authenticate a read-only state receipt without demanding a new scan."""

    fields = (
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
    stable = {field: before.get(field) == after.get(field) for field in fields}
    return {
        "observation_required": False,
        "read_only_state_query": True,
        "stable_fields": stable,
        "passed": all(stable.values()),
    }


def _compact_interlock(value: Mapping[str, Any]) -> dict[str, Any]:
    """Remove the legacy learned-call name from a deterministic sentinel use."""

    fields = (
        "terminal_alignment_requested",
        "target_available",
        "target_xyz_m",
        "robot_position_xy_m",
        "robot_yaw_degrees",
        "desired_yaw_degrees",
        "angular_residual_degrees",
        "deadband_degrees",
        "correction_applied",
        "corrected_turn_degrees",
        "stop_applied",
        "reason",
        "numeric_target_state_only",
        "oracle_inputs_at_runtime",
    )
    return {
        "schema": "semantic_3d_chat.deterministic_numeric_alignment_decision.v1",
        "proposal_source": "neutral_stop_sentinel",
        **{field: value[field] for field in fields},
        "environmental_text_inputs": [],
    }


def _compact_approach_interlock(value: Mapping[str, Any]) -> dict[str, Any]:
    """Expose the deterministic numeric decision without learned-policy claims."""

    fields = (
        "terminal_approach_requested",
        "target_available",
        "target_xyz_m",
        "initial_robot_position_xy_m",
        "robot_position_xy_m",
        "robot_yaw_degrees",
        "desired_yaw_degrees",
        "angular_residual_degrees",
        "target_distance_m",
        "actual_progress_m",
        "minimum_progress_m",
        "target_standoff_m",
        "heading_deadband_degrees",
        "goal_satisfied",
        "completion_satisfied",
        "completion_mode",
        "correction_applied",
        "corrected_tool",
        "corrected_arguments",
        "stop_applied",
        "reason",
        "numeric_target_state_only",
        "collision_precheck_predicted_rejection",
        "collision_rejection_deferred_to_exact_simulator",
        "oracle_inputs_at_runtime",
    )
    collision = value.get("collision_limited_interlock")
    if not isinstance(collision, Mapping):
        raise TypeError("Approach decision omitted its collision-limited interlock")
    collision_fields = (
        "numeric_collision_map_only",
        "collision_predicted",
        "requested_distance_m",
        "maximum_collision_free_distance_m",
        "executed_safe_distance_m",
        "minimum_safe_step_m",
        "safe_closest_reachable",
        "reason",
        "oracle_inputs_at_runtime",
    )
    return {
        "schema": "semantic_3d_chat.deterministic_numeric_approach_decision.v1",
        "proposal_source": "neutral_stop_sentinel",
        **{field: value[field] for field in fields},
        "collision_limited_interlock": {
            "schema": "semantic_3d_chat.numeric_collision_decision.v1",
            **{field: collision[field] for field in collision_fields},
            "environmental_text_inputs": [],
        },
        "environmental_text_inputs": [],
    }


def _grounding_summary(grounding: Any, binding: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "target_xyz_m": list(grounding.target_xyz_m),
        "seed_xyz_m": list(grounding.seed_xyz_m),
        "cosine_similarity": grounding.cosine_similarity,
        "similarity_q99": grounding.similarity_q99,
        "peak_margin_over_q99": grounding.peak_margin_over_q99,
        "local_support_voxels": grounding.local_support_voxels,
        "scored_voxels": grounding.scored_voxels,
        "eligible_voxels": grounding.eligible_voxels,
        "all_map_voxels_scored": grounding.scored_voxels == int(binding["source_voxels"]),
    }


def _session_policy_summary(*, max_steps: int, official_stdio_actions: bool) -> dict[str, Any]:
    return {
        "name": SESSION_POLICY_NAME,
        "natural_language_parser": "bounded_label_free_action_grammar",
        "continuous_grounding": "selective_local_gemma_tied_token_embeddings",
        "grounding_scope": "every_active_map_voxel",
        "action_controller": "v3_numeric_alignment_and_approach_interlocks",
        "collision_controller": "anonymous_numeric_voxel_collision_map",
        "interlock_proposal_source": "neutral_stop_sentinel",
        "learned_v3_action_head_used": False,
        "gemma_native_function_calling_used": False,
        "direct_function_action_execution_used": False,
        "official_mcp_sdk_stdio_action_execution": official_stdio_actions,
        "one_persistent_stdio_session": True,
        "initial_observation_before_navigation": True,
        "fresh_all_voxel_grounding_before_every_navigation_decision": True,
        "successful_motion_refreshes_map_and_complete_scene_prefix": True,
        "goal_convergence_stationary_without_episode_stop_latch": True,
        "standalone_stop_is_episode_safety_latched": True,
        "maximum_steps_per_navigation_turn": max_steps,
        "environmental_text_inputs": [],
        "oracle_inputs_at_runtime": False,
    }


def _policy_summary(*, max_steps: int, official_stdio_actions: bool) -> dict[str, Any]:
    return {
        "name": POLICY_NAME,
        "natural_language_parser": "v3_action_grammar_target_phrase_only",
        "continuous_grounding": "selective_local_gemma_tied_token_embeddings",
        "grounding_scope": "every_active_map_voxel",
        "action_controller": "v3_numeric_alignment_convergence_interlock",
        "interlock_proposal_source": "neutral_stop_sentinel",
        "learned_v3_action_head_used": False,
        "gemma_native_function_calling_used": False,
        "direct_function_action_execution_used": False,
        "official_mcp_sdk_stdio_action_execution": official_stdio_actions,
        "initial_observation_before_alignment": True,
        "fresh_all_voxel_grounding_before_every_decision": True,
        "maximum_alignment_steps": max_steps,
        "environmental_text_inputs": [],
        "oracle_inputs_at_runtime": False,
    }


def _require_terminal_face_instruction(instruction: str) -> str:
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("Instruction must be non-empty text")
    normalized = " ".join(instruction.strip().split())
    target_text = target_text_from_navigation_instruction(normalized)
    if target_text is None or _TERMINAL_FACE.fullmatch(normalized) is None:
        raise ValueError("Face integration requires an explicit terminal 'then stop' goal")
    return target_text


def run_face_instruction(
    runtime: MCPConversationRuntime,
    text_encoder: ContinuousTextEncoder,
    instruction: str,
    *,
    room_size_m: Sequence[float],
    feature_start: int,
    feature_dim: int,
    max_steps: int = 8,
) -> dict[str, Any]:
    """Run one bounded face-target loop over an already connected MCP runtime."""

    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or not 1 <= max_steps <= 32:
        raise ValueError("max_steps must be an integer in [1, 32]")
    room = np.asarray(room_size_m, dtype=np.float32)
    if room.shape != (3,) or not np.isfinite(room).all() or np.any(room <= 0.0):
        raise ValueError("room_size_m must contain three finite positive values")
    target_text = _require_terminal_face_instruction(instruction)
    request_sha256 = hashlib.sha256(instruction.strip().encode("utf-8")).hexdigest()
    target_text_sha256 = hashlib.sha256(target_text.encode("utf-8")).hexdigest()
    settings = runtime.simulator.settings
    maximum_turn = float(settings["max_turn_degrees"])
    deadband = float(settings.get("face_alignment_deadband_degrees", 3.0))
    stalled = float(settings.get("face_alignment_stalled_turn_degrees", 1.0))
    if not all(math.isfinite(value) and value > 0.0 for value in (maximum_turn, deadband, stalled)):
        raise ValueError("Configured face-alignment bounds must be finite and positive")

    official_stdio = isinstance(runtime._tool_client, MCPStdioToolClient)
    initial_binding = runtime.prefix_binding()
    before_scan = runtime.prefix_binding()
    scan_receipt = runtime.scan()
    if scan_receipt.get("success") is not True:
        raise RuntimeError("Initial MCP observation failed")
    after_scan = runtime.prefix_binding()
    scan_transition = _binding_transition(
        before_scan,
        after_scan,
        observation_required=True,
    )
    if not scan_transition["passed"]:
        raise RuntimeError("Initial MCP scan did not refresh map and scene prefix")

    steps: list[dict[str, Any]] = []
    receipts: list[dict[str, Any]] = [dict(scan_receipt)]
    termination_reason = "maximum_steps"
    for step_index in range(1, max_steps + 1):
        map_path = _active_map_path(runtime)
        grounding = ContinuousSemanticTargetGrounder(
            map_path,
            text_encoder,
            room_size_m=room,
            feature_start=feature_start,
            feature_dim=feature_dim,
        ).ground(target_text)
        binding_before = runtime.prefix_binding()
        if grounding.map_sha256 != binding_before["map_sha256"]:
            raise RuntimeError("Grounded active map differs from authenticated MCP binding")
        minimum = torch.tensor([-room[0] / 2.0, -room[1] / 2.0, 0.0])
        maximum = torch.tensor([room[0] / 2.0, room[1] / 2.0, room[2]])
        state_features = robot_state_vector(_numeric_state(runtime), minimum, maximum)
        target_state = grounded_target_state(
            torch.tensor(grounding.target_xyz_m),
            state_features,
            torch.tensor(1.0),
            room_size_m=room,
        )
        if target_state.shape != (1, TARGET_STATE_DIM):
            raise RuntimeError("V3 numeric target state shape changed")
        call, raw_interlock = apply_numeric_alignment_interlock(
            instruction,
            target_state,
            {"tool": "stop", "arguments": {}},
            target_xyz_m=grounding.target_xyz_m,
            robot_position_xy_m=runtime.simulator.state.position_xy_m,
            robot_yaw_degrees=runtime.simulator.state.body_yaw_degrees,
            max_turn_degrees=maximum_turn,
            deadband_degrees=deadband,
            stalled_turn_degrees=stalled,
        )
        if raw_interlock["terminal_alignment_requested"] is not True:
            raise ValueError("Face integration requires an explicit terminal 'then stop' goal")
        tool = call.get("tool")
        arguments = call.get("arguments")
        if tool == "turn" and isinstance(arguments, Mapping):
            receipt = runtime.turn(float(arguments["angle_degrees"]))
            observation_required = True
        elif tool == "stop" and arguments == {}:
            receipt = runtime.stop()
            observation_required = False
        else:
            raise RuntimeError("Numeric face controller emitted a non-face action")
        binding_after = runtime.prefix_binding()
        transition = _binding_transition(
            binding_before,
            binding_after,
            observation_required=observation_required,
        )
        if receipt.get("success") is not True or not transition["passed"]:
            raise RuntimeError("MCP face action failed or did not refresh its continuous binding")
        receipts.append(dict(receipt))
        steps.append(
            {
                "step": step_index,
                "active_map_sha256": grounding.map_sha256,
                "query_embedding_sha256": grounding.query_embedding_sha256,
                "target_state_sha256": hashlib.sha256(
                    target_state.detach().cpu().contiguous().numpy().tobytes()
                ).hexdigest(),
                "grounding": {
                    "target_xyz_m": list(grounding.target_xyz_m),
                    "seed_xyz_m": list(grounding.seed_xyz_m),
                    "cosine_similarity": grounding.cosine_similarity,
                    "similarity_q99": grounding.similarity_q99,
                    "peak_margin_over_q99": grounding.peak_margin_over_q99,
                    "local_support_voxels": grounding.local_support_voxels,
                    "scored_voxels": grounding.scored_voxels,
                    "eligible_voxels": grounding.eligible_voxels,
                    "all_map_voxels_scored": grounding.scored_voxels
                    == int(binding_before["source_voxels"]),
                },
                "numeric_alignment": _compact_interlock(raw_interlock),
                "mcp_call": {"tool": tool, "arguments": dict(arguments)},
                "numeric_tool_receipt": dict(receipt),
                "continuous_binding_transition": transition,
            }
        )
        if tool == "stop":
            termination_reason = "fresh_grounding_inside_deadband"
            break

    final_binding = runtime.prefix_binding()
    leaks = _numeric_protocol_violations(receipts)
    every_voxel = bool(steps) and all(
        step["grounding"]["all_map_voxels_scored"] is True for step in steps
    )
    stopped = bool(runtime.simulator.state.stopped)
    success = (
        termination_reason == "fresh_grounding_inside_deadband"
        and stopped
        and every_voxel
        and not leaks
    )
    return {
        "schema": SCHEMA,
        "passed": success,
        "scene_id": runtime.simulator.state.scene_id,
        "instruction": instruction.strip(),
        "instruction_source": "user_text_only",
        "request_sha256": request_sha256,
        "target_text_sha256": target_text_sha256,
        "target_phrase_retained_in_tool_output": False,
        "policy": _policy_summary(
            max_steps=max_steps,
            official_stdio_actions=official_stdio,
        ),
        "transport": {
            "implementation": (
                "official_python_mcp_sdk_stdio"
                if official_stdio
                else "structured_tool_client_test_seam"
            ),
            "mcp_sdk_version": importlib.metadata.version("mcp"),
            "process_boundary": official_stdio,
            "tool_count": len(runtime._tool_client.tool_names),
            "tools": sorted(runtime._tool_client.tool_names),
            "numeric_structured_output_only": True,
        },
        "initial_binding": initial_binding,
        "initial_observation": {
            "numeric_tool_receipt": dict(scan_receipt),
            "continuous_binding_transition": scan_transition,
        },
        "steps": steps,
        "step_count": len(steps),
        "termination_reason": termination_reason,
        "final_binding": final_binding,
        "final_body_yaw_degrees": float(runtime.simulator.state.body_yaw_degrees),
        "final_stopped": stopped,
        "all_decisions_used_fresh_all_voxel_grounding": every_voxel,
        "prefix_binding_refresh_count": runtime.binding_refresh_count,
        "semantic_leaks_in_numeric_tool_receipts": leaks,
        "environmental_text_inputs": [],
        "oracle_inputs_at_runtime": False,
    }


class PersistentMCPConversationSession:
    """Repeated semantic navigation turns over one connected MCP subprocess.

    Target phrases exist only on this client side.  Each navigation decision
    embeds the current user target, scores every active-map voxel, derives a
    numerical XYZ target, and sends only bounded numerical tool arguments to
    the MCP server.  The server remains authoritative for motion, collision,
    observations, map fusion, and continuous-prefix reconstruction.
    """

    def __init__(
        self,
        runtime: MCPConversationRuntime,
        text_encoder: ContinuousTextEncoder,
        *,
        room_size_m: Sequence[float],
        feature_start: int,
        feature_dim: int,
        max_steps: int = 12,
    ) -> None:
        if (
            isinstance(max_steps, bool)
            or not isinstance(max_steps, int)
            or not 1 <= max_steps <= 32
        ):
            raise ValueError("max_steps must be an integer in [1, 32]")
        room = np.asarray(room_size_m, dtype=np.float32)
        if room.shape != (3,) or not np.isfinite(room).all() or np.any(room <= 0.0):
            raise ValueError("room_size_m must contain three finite positive values")
        if (
            isinstance(feature_start, bool)
            or not isinstance(feature_start, int)
            or feature_start < 0
        ):
            raise ValueError("feature_start must be a nonnegative integer")
        if isinstance(feature_dim, bool) or not isinstance(feature_dim, int) or feature_dim < 1:
            raise ValueError("feature_dim must be a positive integer")
        if int(text_encoder.output_dim) != feature_dim:
            raise ValueError("Text encoder and grounding feature dimensions differ")
        settings = runtime.simulator.settings
        if settings.get("auto_scan_after_motion") is not True:
            raise ValueError("Persistent MCP navigation requires auto_scan_after_motion=true")

        required = {
            "max_turn_degrees": float(settings["max_turn_degrees"]),
            "max_move_m": float(settings["max_move_m"]),
            "face_alignment_deadband_degrees": float(
                settings.get("face_alignment_deadband_degrees", 3.0)
            ),
            "face_alignment_stalled_turn_degrees": float(
                settings.get("face_alignment_stalled_turn_degrees", 1.0)
            ),
            "approach_heading_deadband_degrees": float(
                settings.get("approach_heading_deadband_degrees", 15.0)
            ),
            "approach_target_standoff_m": float(settings.get("approach_target_standoff_m", 0.5)),
            "approach_minimum_progress_m": float(settings.get("approach_minimum_progress_m", 0.15)),
            "approach_minimum_safe_step_m": float(
                settings.get("approach_minimum_safe_step_m", 0.02)
            ),
        }
        if not all(math.isfinite(value) and value > 0.0 for value in required.values()):
            raise ValueError("Persistent MCP navigation bounds must be finite and positive")
        if (
            required["face_alignment_deadband_degrees"] >= required["max_turn_degrees"]
            or required["face_alignment_stalled_turn_degrees"] >= required["max_turn_degrees"]
            or required["approach_heading_deadband_degrees"] >= required["max_turn_degrees"]
            or required["approach_minimum_progress_m"] > required["max_move_m"]
            or required["approach_minimum_safe_step_m"] > required["max_move_m"]
        ):
            raise ValueError("Persistent MCP navigation thresholds exceed action bounds")

        self.runtime = runtime
        self.text_encoder = text_encoder
        self.room = room
        self.feature_start = feature_start
        self.feature_dim = feature_dim
        self.max_steps = max_steps
        self.settings = required
        self.official_stdio = isinstance(runtime._tool_client, MCPStdioToolClient)
        self.started = False
        self.initial_binding: dict[str, Any] | None = None
        self.initial_observation: dict[str, Any] | None = None
        self.turns: list[dict[str, Any]] = []
        self.shutdown_record: dict[str, Any] | None = None

    def _transport(self) -> dict[str, Any]:
        return {
            "implementation": (
                "official_python_mcp_sdk_stdio"
                if self.official_stdio
                else "structured_tool_client_test_seam"
            ),
            "mcp_sdk_version": importlib.metadata.version("mcp"),
            "process_boundary": self.official_stdio,
            "tool_count": len(self.runtime._tool_client.tool_names),
            "tools": sorted(self.runtime._tool_client.tool_names),
            "numeric_structured_output_only": True,
            "persistent_connection": True,
        }

    def start(self) -> dict[str, Any]:
        if self.started:
            raise RuntimeError("Persistent MCP conversation session was already started")
        before = self.runtime.prefix_binding()
        receipt = self.runtime.scan()
        after = self.runtime.prefix_binding()
        transition = _binding_transition(before, after, observation_required=True)
        if receipt.get("success") is not True or not transition["passed"]:
            raise RuntimeError("Initial MCP scan did not refresh the continuous scene binding")
        if _numeric_protocol_violations([receipt]):
            raise RuntimeError("Initial numeric MCP receipt contains semantic text")
        self.started = True
        self.initial_binding = before
        self.initial_observation = {
            "numeric_tool_receipt": dict(receipt),
            "continuous_binding_transition": transition,
        }
        return {
            "schema": SESSION_SCHEMA,
            "phase": "started",
            "passed": True,
            "scene_id": self.runtime.simulator.state.scene_id,
            "policy": _session_policy_summary(
                max_steps=self.max_steps,
                official_stdio_actions=self.official_stdio,
            ),
            "transport": self._transport(),
            "initial_observation": self.initial_observation,
            "supported_commands": [
                "face/look/turn toward TARGET [then stop]",
                "approach/move closer/walk toward TARGET [then stop]",
                "scan",
                "get robot state",
                "stop",
            ],
            "environmental_text_inputs": [],
            "oracle_inputs_at_runtime": False,
        }

    def _require_started(self) -> None:
        if not self.started:
            raise RuntimeError("Persistent MCP conversation session has not started")

    def _target_state(self, target_text: str) -> tuple[Any, torch.Tensor, dict[str, Any]]:
        binding = self.runtime.prefix_binding()
        grounding = ContinuousSemanticTargetGrounder(
            _active_map_path(self.runtime),
            self.text_encoder,
            room_size_m=self.room,
            feature_start=self.feature_start,
            feature_dim=self.feature_dim,
        ).ground(target_text)
        if grounding.map_sha256 != binding["map_sha256"]:
            raise RuntimeError("Grounded active map differs from authenticated MCP binding")
        minimum = torch.tensor([-self.room[0] / 2.0, -self.room[1] / 2.0, 0.0], dtype=torch.float32)
        maximum = torch.tensor(
            [self.room[0] / 2.0, self.room[1] / 2.0, self.room[2]],
            dtype=torch.float32,
        )
        state_features = robot_state_vector(_numeric_state(self.runtime), minimum, maximum)
        target_state = grounded_target_state(
            torch.tensor(grounding.target_xyz_m, dtype=torch.float32),
            state_features,
            torch.tensor(1.0, dtype=torch.float32),
            room_size_m=self.room,
        )
        if target_state.shape != (1, TARGET_STATE_DIM) or not torch.isfinite(target_state).all():
            raise RuntimeError("V3 numeric target state is invalid")
        return grounding, target_state, binding

    def _settle_goal(self) -> tuple[dict[str, Any], dict[str, Any]]:
        state = self.runtime.simulator.state
        linear = np.asarray(state.linear_velocity_xy_m, dtype=np.float64)
        if (
            linear.shape != (2,)
            or not np.isfinite(linear).all()
            or np.linalg.norm(linear) > 1e-8
            or not math.isfinite(float(state.angular_velocity_degrees))
            or abs(float(state.angular_velocity_degrees)) > 1e-8
        ):
            raise RuntimeError("Navigation goal cannot settle while the robot is moving")
        if state.stopped:
            raise RuntimeError("Navigation goal unexpectedly consumed the episode stop latch")
        before = self.runtime.prefix_binding()
        receipt = self.runtime.get_robot_state()
        after = self.runtime.prefix_binding()
        transition = _stable_binding_transition(before, after)
        if receipt.get("success") is not True or not transition["passed"]:
            raise RuntimeError("MCP goal-settle state acknowledgment was not read-only")
        return dict(receipt), transition

    @staticmethod
    def _target_hashes(instruction: str, target_text: str) -> dict[str, Any]:
        return {
            "instruction": instruction.strip(),
            "instruction_source": "user_text_only",
            "request_sha256": hashlib.sha256(instruction.strip().encode("utf-8")).hexdigest(),
            "target_text_sha256": hashlib.sha256(target_text.encode("utf-8")).hexdigest(),
            "target_phrase_retained_in_tool_output": False,
        }

    def _face(self, command: InteractiveMCPCommand, instruction: str) -> dict[str, Any]:
        assert command.target_text is not None and command.terminal_instruction is not None
        steps: list[dict[str, Any]] = []
        receipts: list[dict[str, Any]] = []
        termination = "maximum_steps"
        for step_index in range(1, self.max_steps + 1):
            grounding, target_state, binding_before = self._target_state(command.target_text)
            call, raw = apply_numeric_alignment_interlock(
                command.terminal_instruction,
                target_state,
                {"tool": "stop", "arguments": {}},
                target_xyz_m=grounding.target_xyz_m,
                robot_position_xy_m=self.runtime.simulator.state.position_xy_m,
                robot_yaw_degrees=self.runtime.simulator.state.body_yaw_degrees,
                max_turn_degrees=self.settings["max_turn_degrees"],
                deadband_degrees=self.settings["face_alignment_deadband_degrees"],
                stalled_turn_degrees=self.settings["face_alignment_stalled_turn_degrees"],
            )
            tool = call.get("tool")
            arguments = call.get("arguments")
            if tool == "turn" and isinstance(arguments, Mapping):
                receipt = self.runtime.turn(float(arguments["angle_degrees"]))
                transition = _binding_transition(
                    binding_before, self.runtime.prefix_binding(), observation_required=True
                )
                actual_call = {"tool": "turn", "arguments": dict(arguments)}
                goal_settled = False
            elif tool == "stop" and arguments == {}:
                receipt, transition = self._settle_goal()
                actual_call = {"tool": "get_robot_state", "arguments": {}}
                goal_settled = True
            else:
                raise RuntimeError("Numeric face controller emitted a non-face action")
            if receipt.get("success") is not True or not transition["passed"]:
                raise RuntimeError("MCP face action failed or did not refresh its binding")
            receipts.append(dict(receipt))
            steps.append(
                {
                    "step": step_index,
                    "active_map_sha256": grounding.map_sha256,
                    "query_embedding_sha256": grounding.query_embedding_sha256,
                    "target_state_sha256": hashlib.sha256(
                        target_state.detach().cpu().contiguous().numpy().tobytes()
                    ).hexdigest(),
                    "grounding": _grounding_summary(grounding, binding_before),
                    "numeric_alignment": _compact_interlock(raw),
                    "controller_call": {"tool": tool, "arguments": dict(arguments)},
                    "mcp_call": actual_call,
                    "goal_settled_without_episode_latch": goal_settled,
                    "numeric_tool_receipt": dict(receipt),
                    "continuous_binding_transition": transition,
                }
            )
            if goal_settled:
                termination = "fresh_grounding_inside_deadband"
                break
        every_voxel = bool(steps) and all(
            step["grounding"]["all_map_voxels_scored"] is True for step in steps
        )
        leaks = _numeric_protocol_violations(receipts)
        passed = (
            termination == "fresh_grounding_inside_deadband"
            and not self.runtime.simulator.state.stopped
            and every_voxel
            and not leaks
        )
        return {
            "schema": "semantic_3d_chat.conversational_mcp_session_turn.v1",
            "passed": passed,
            "command_kind": "face",
            **self._target_hashes(instruction, command.target_text),
            "steps": steps,
            "step_count": len(steps),
            "termination_reason": termination,
            "final_binding": self.runtime.prefix_binding(),
            "final_position_xy_m": self.runtime.simulator.state.position_xy_m.tolist(),
            "final_body_yaw_degrees": float(self.runtime.simulator.state.body_yaw_degrees),
            "episode_stop_latched": bool(self.runtime.simulator.state.stopped),
            "goal_settled_without_episode_latch": passed,
            "all_decisions_used_fresh_all_voxel_grounding": every_voxel,
            "semantic_leaks_in_numeric_tool_receipts": leaks,
            "environmental_text_inputs": [],
            "oracle_inputs_at_runtime": False,
        }

    def _approach(self, command: InteractiveMCPCommand, instruction: str) -> dict[str, Any]:
        assert command.target_text is not None and command.terminal_instruction is not None
        initial_xy = np.asarray(self.runtime.simulator.state.position_xy_m, dtype=np.float64).copy()
        steps: list[dict[str, Any]] = []
        receipts: list[dict[str, Any]] = []
        termination = "maximum_steps"
        for step_index in range(1, self.max_steps + 1):
            grounding, target_state, binding_before = self._target_state(command.target_text)
            current_xy = np.asarray(
                self.runtime.simulator.state.position_xy_m, dtype=np.float64
            ).copy()
            call, raw = apply_numeric_approach_interlock(
                command.terminal_instruction,
                target_state,
                {"tool": "stop", "arguments": {}},
                target_xyz_m=grounding.target_xyz_m,
                initial_robot_position_xy_m=initial_xy,
                robot_position_xy_m=current_xy,
                robot_yaw_degrees=self.runtime.simulator.state.body_yaw_degrees,
                max_turn_degrees=self.settings["max_turn_degrees"],
                max_move_m=self.settings["max_move_m"],
                heading_deadband_degrees=self.settings["approach_heading_deadband_degrees"],
                target_standoff_m=self.settings["approach_target_standoff_m"],
                minimum_progress_m=self.settings["approach_minimum_progress_m"],
                stalled_turn_degrees=self.settings["face_alignment_stalled_turn_degrees"],
            )
            call, raw = apply_collision_limited_approach_interlock(
                call,
                raw,
                collision_map=self.runtime.simulator.collision_map,
                robot_position_xy_m=current_xy,
                robot_yaw_degrees=self.runtime.simulator.state.body_yaw_degrees,
                minimum_safe_step_m=self.settings["approach_minimum_safe_step_m"],
            )
            tool = call.get("tool")
            arguments = call.get("arguments")
            if tool == "turn" and isinstance(arguments, Mapping):
                receipt = self.runtime.turn(float(arguments["angle_degrees"]))
                transition = _binding_transition(
                    binding_before, self.runtime.prefix_binding(), observation_required=True
                )
                actual_call = {"tool": "turn", "arguments": dict(arguments)}
                goal_settled = False
            elif tool == "move_forward" and isinstance(arguments, Mapping):
                receipt = self.runtime.move_forward(float(arguments["distance_meters"]))
                if receipt.get("success") is not True:
                    raise RuntimeError("Exact MCP simulator rejected collision-limited movement")
                transition = _binding_transition(
                    binding_before, self.runtime.prefix_binding(), observation_required=True
                )
                actual_call = {"tool": "move_forward", "arguments": dict(arguments)}
                goal_settled = False
            elif tool == "stop" and arguments == {}:
                if raw.get("completion_satisfied") is not True:
                    raise RuntimeError("Numeric approach controller attempted a premature stop")
                receipt, transition = self._settle_goal()
                actual_call = {"tool": "get_robot_state", "arguments": {}}
                goal_settled = True
            else:
                raise RuntimeError("Numeric approach controller emitted an unsupported action")
            if receipt.get("success") is not True or not transition["passed"]:
                raise RuntimeError("MCP approach action failed or did not refresh its binding")
            receipts.append(dict(receipt))
            steps.append(
                {
                    "step": step_index,
                    "active_map_sha256": grounding.map_sha256,
                    "query_embedding_sha256": grounding.query_embedding_sha256,
                    "target_state_sha256": hashlib.sha256(
                        target_state.detach().cpu().contiguous().numpy().tobytes()
                    ).hexdigest(),
                    "grounding": _grounding_summary(grounding, binding_before),
                    "numeric_approach": _compact_approach_interlock(raw),
                    "controller_call": {"tool": tool, "arguments": dict(arguments)},
                    "mcp_call": actual_call,
                    "goal_settled_without_episode_latch": goal_settled,
                    "numeric_tool_receipt": dict(receipt),
                    "continuous_binding_transition": transition,
                }
            )
            if goal_settled:
                termination = str(raw["completion_mode"])
                break
        every_voxel = bool(steps) and all(
            step["grounding"]["all_map_voxels_scored"] is True for step in steps
        )
        leaks = _numeric_protocol_violations(receipts)
        completed = termination in {"semantic_standoff", "collision_limited_safe_stop"}
        passed = (
            completed and not self.runtime.simulator.state.stopped and every_voxel and not leaks
        )
        final_xy = np.asarray(self.runtime.simulator.state.position_xy_m, dtype=np.float64)
        return {
            "schema": "semantic_3d_chat.conversational_mcp_session_turn.v1",
            "passed": passed,
            "command_kind": "approach",
            **self._target_hashes(instruction, command.target_text),
            "steps": steps,
            "step_count": len(steps),
            "termination_reason": termination,
            "final_binding": self.runtime.prefix_binding(),
            "initial_position_xy_m": initial_xy.tolist(),
            "final_position_xy_m": final_xy.tolist(),
            "actual_progress_m": float(np.linalg.norm(final_xy - initial_xy)),
            "episode_stop_latched": bool(self.runtime.simulator.state.stopped),
            "goal_settled_without_episode_latch": passed,
            "all_decisions_used_fresh_all_voxel_grounding": every_voxel,
            "semantic_leaks_in_numeric_tool_receipts": leaks,
            "environmental_text_inputs": [],
            "oracle_inputs_at_runtime": False,
        }

    def _simple(self, command: InteractiveMCPCommand, instruction: str) -> dict[str, Any]:
        before = self.runtime.prefix_binding()
        if command.kind == "state":
            receipt = self.runtime.get_robot_state()
            transition = _stable_binding_transition(before, self.runtime.prefix_binding())
            actual_tool = "get_robot_state"
        elif command.kind == "scan":
            receipt = self.runtime.scan()
            transition = _binding_transition(
                before, self.runtime.prefix_binding(), observation_required=True
            )
            actual_tool = "scan"
        elif command.kind == "stop":
            if self.runtime.simulator.state.stopped:
                receipt = self.runtime.get_robot_state()
                transition = _stable_binding_transition(before, self.runtime.prefix_binding())
                actual_tool = "get_robot_state"
            else:
                receipt = self.runtime.stop()
                transition = _binding_transition(
                    before, self.runtime.prefix_binding(), observation_required=False
                )
                actual_tool = "stop"
        else:  # pragma: no cover - parser and dispatcher are exhaustive
            raise RuntimeError("Unsupported simple MCP command")
        leaks = _numeric_protocol_violations([receipt])
        passed = (
            receipt.get("success") is True
            and transition["passed"] is True
            and not leaks
            and (command.kind != "stop" or self.runtime.simulator.state.stopped)
        )
        return {
            "schema": "semantic_3d_chat.conversational_mcp_session_turn.v1",
            "passed": passed,
            "command_kind": command.kind,
            "instruction": instruction.strip(),
            "instruction_source": "user_text_only",
            "request_sha256": hashlib.sha256(instruction.strip().encode("utf-8")).hexdigest(),
            "mcp_call": {"tool": actual_tool, "arguments": {}},
            "numeric_tool_receipt": dict(receipt),
            "continuous_binding_transition": transition,
            "final_binding": self.runtime.prefix_binding(),
            "episode_stop_latched": bool(self.runtime.simulator.state.stopped),
            "semantic_leaks_in_numeric_tool_receipts": leaks,
            "environmental_text_inputs": [],
            "oracle_inputs_at_runtime": False,
        }

    def handle(self, instruction: str) -> dict[str, Any]:
        self._require_started()
        command = parse_interactive_mcp_command(instruction)
        if self.runtime.simulator.state.stopped and command.kind not in {"state", "stop"}:
            result = {
                "schema": "semantic_3d_chat.conversational_mcp_session_turn.v1",
                "passed": False,
                "command_kind": command.kind,
                "instruction": instruction.strip(),
                "instruction_source": "user_text_only",
                "request_sha256": hashlib.sha256(instruction.strip().encode("utf-8")).hexdigest(),
                "error_code": "E_STOPPED",
                "tool_executed": False,
                "episode_stop_latched": True,
                "environmental_text_inputs": [],
                "oracle_inputs_at_runtime": False,
            }
        elif command.kind == "face":
            result = self._face(command, instruction)
        elif command.kind == "approach":
            result = self._approach(command, instruction)
        else:
            result = self._simple(command, instruction)
        result["turn_index"] = len(self.turns) + 1
        self.turns.append(result)
        return result

    def shutdown(self, *, reason: str = "client_exit") -> dict[str, Any]:
        """Latch safe stop before closing stdio, unless already explicitly stopped."""

        self._require_started()
        if self.shutdown_record is not None:
            return self.shutdown_record
        before = self.runtime.prefix_binding()
        if self.runtime.simulator.state.stopped:
            receipt = self.runtime.get_robot_state()
            transition = _stable_binding_transition(before, self.runtime.prefix_binding())
            stop_called = False
        else:
            receipt = self.runtime.stop()
            transition = _binding_transition(
                before, self.runtime.prefix_binding(), observation_required=False
            )
            stop_called = True
        passed = (
            receipt.get("success") is True
            and transition["passed"] is True
            and self.runtime.simulator.state.stopped
            and not _numeric_protocol_violations([receipt])
        )
        self.shutdown_record = {
            "schema": "semantic_3d_chat.conversational_mcp_session_shutdown.v1",
            "passed": passed,
            "reason": reason,
            "mcp_stop_called": stop_called,
            "numeric_tool_receipt": dict(receipt),
            "continuous_binding_transition": transition,
            "episode_stop_latched": bool(self.runtime.simulator.state.stopped),
            "environmental_text_inputs": [],
            "oracle_inputs_at_runtime": False,
        }
        if not passed:
            raise RuntimeError("Persistent MCP session failed to stop safely")
        return self.shutdown_record

    def summary(self) -> dict[str, Any]:
        self._require_started()
        turns_passed = all(turn.get("passed") is True for turn in self.turns)
        shutdown_passed = self.shutdown_record is None or self.shutdown_record.get("passed") is True
        return {
            "schema": SESSION_SCHEMA,
            "passed": turns_passed and shutdown_passed,
            "scene_id": self.runtime.simulator.state.scene_id,
            "policy": _session_policy_summary(
                max_steps=self.max_steps,
                official_stdio_actions=self.official_stdio,
            ),
            "transport": self._transport(),
            "initial_binding": self.initial_binding,
            "initial_observation": self.initial_observation,
            "turns": self.turns,
            "turn_count": len(self.turns),
            "shutdown": self.shutdown_record,
            "final_binding": self.runtime.prefix_binding(),
            "final_position_xy_m": self.runtime.simulator.state.position_xy_m.tolist(),
            "final_body_yaw_degrees": float(self.runtime.simulator.state.body_yaw_degrees),
            "final_stopped": bool(self.runtime.simulator.state.stopped),
            "prefix_binding_refresh_count": self.runtime.binding_refresh_count,
            "environmental_text_inputs": [],
            "oracle_inputs_at_runtime": False,
        }


def _preflight(
    *,
    config_path: str | Path,
    scene_id: str,
    instruction: str,
    base_checkpoint: str | Path,
    control_checkpoint: str | Path,
    control_runtime_config: str | Path,
    runtime_asset: str | Path,
    robot_state_checkpoint: str | Path,
    max_steps: int,
) -> dict[str, Any]:
    started = time.monotonic()
    config_file = _safe_input(config_path, purpose="embodied config", kind="file")
    config = load_config(config_file)
    target = _require_terminal_face_instruction(instruction)
    for path, purpose, kind in (
        (base_checkpoint, "base checkpoint", "directory"),
        (control_checkpoint, "control checkpoint", "directory"),
        (control_runtime_config, "control runtime config", "file"),
        (runtime_asset, "sanitized runtime asset", "file"),
        (robot_state_checkpoint, "numeric robot-state checkpoint", "directory"),
        (project_path(config, "maps", scene_id, "voxel_map.npz"), "base map", "file"),
    ):
        _safe_input(path, purpose=purpose, kind=kind)
    encoder = GemmaProjectedTextEncoder.from_config(config)
    if max_steps < 1 or max_steps > 32:
        raise ValueError("max_steps must be in [1, 32]")
    return {
        "schema": SCHEMA,
        "phase": "preflight",
        "passed": True,
        "scene_id": scene_id,
        "request_sha256": hashlib.sha256(instruction.strip().encode("utf-8")).hexdigest(),
        "target_text_sha256": hashlib.sha256(target.encode("utf-8")).hexdigest(),
        "policy": _policy_summary(
            max_steps=max_steps,
            official_stdio_actions=False,
        ),
        "official_mcp_sdk_stdio_action_execution_planned": True,
        "local_model_snapshot": str(encoder.snapshot),
        "loads_full_gemma_model": False,
        "loads_blender": False,
        "starts_mcp_transport": False,
        "changes_robot_or_map_state": False,
        "environmental_text_inputs": [],
        "oracle_inputs_at_runtime": False,
        "elapsed_seconds": time.monotonic() - started,
    }


def _session_preflight(
    *,
    config_path: str | Path,
    scene_id: str,
    commands: Sequence[str],
    base_checkpoint: str | Path,
    control_checkpoint: str | Path,
    control_runtime_config: str | Path,
    runtime_asset: str | Path,
    robot_state_checkpoint: str | Path,
    max_steps: int,
) -> dict[str, Any]:
    """Read-only validation for the persistent conversational MCP surface."""

    started = time.monotonic()
    config_file = _safe_input(config_path, purpose="embodied config", kind="file")
    config = load_config(config_file)
    parsed = [parse_interactive_mcp_command(command) for command in commands]
    for path, purpose, kind in (
        (base_checkpoint, "base checkpoint", "directory"),
        (control_checkpoint, "control checkpoint", "directory"),
        (control_runtime_config, "control runtime config", "file"),
        (runtime_asset, "sanitized runtime asset", "file"),
        (robot_state_checkpoint, "numeric robot-state checkpoint", "directory"),
        (project_path(config, "maps", scene_id, "voxel_map.npz"), "base map", "file"),
    ):
        _safe_input(path, purpose=purpose, kind=kind)
    encoder = GemmaProjectedTextEncoder.from_config(config)
    if isinstance(max_steps, bool) or not 1 <= max_steps <= 32:
        raise ValueError("max_steps must be in [1, 32]")
    robot = config.get("robot")
    if not isinstance(robot, Mapping) or robot.get("auto_scan_after_motion") is not True:
        raise ValueError("Interactive MCP requires robot.auto_scan_after_motion=true")
    return {
        "schema": SESSION_SCHEMA,
        "phase": "preflight",
        "passed": True,
        "scene_id": scene_id,
        "command_count": len(commands),
        "command_kinds": [command.kind for command in parsed],
        "command_request_sha256": [
            hashlib.sha256(command.strip().encode("utf-8")).hexdigest() for command in commands
        ],
        "policy": _session_policy_summary(
            max_steps=max_steps,
            official_stdio_actions=False,
        ),
        "official_mcp_sdk_stdio_action_execution_planned": True,
        "local_model_snapshot": str(encoder.snapshot),
        "loads_full_gemma_model": False,
        "loads_gemma_tensor_rows": False,
        "loads_blender": False,
        "starts_mcp_transport": False,
        "changes_robot_or_map_state": False,
        "supported_commands": [
            "face/look/turn toward TARGET [then stop]",
            "approach/move closer/walk toward TARGET [then stop]",
            "scan",
            "get robot state",
            "stop",
        ],
        "environmental_text_inputs": [],
        "oracle_inputs_at_runtime": False,
        "elapsed_seconds": time.monotonic() - started,
    }


def run_live(
    *,
    config_path: str | Path,
    scene_id: str,
    instruction: str,
    base_checkpoint: str | Path,
    control_checkpoint: str | Path,
    control_runtime_config: str | Path,
    runtime_asset: str | Path,
    robot_state_checkpoint: str | Path,
    client_audit_report: str | Path,
    server_audit_report: str | Path,
    python_executable: str | Path | None = None,
    max_steps: int = 8,
) -> dict[str, Any]:
    """Start production MCP, run one instruction, and authenticate both audits."""

    started = time.monotonic()
    audit = _client_audit()
    result: dict[str, Any] | None = None
    error: BaseException | None = None
    workspace = PROJECT_ROOT / "reports" / "gemma4" / "artifacts"
    workspace.mkdir(parents=True, exist_ok=True)
    try:
        with (
            audit,
            tempfile.TemporaryDirectory(
                prefix=f"conversational_mcp_{scene_id}_",
                dir=workspace,
            ) as temporary,
        ):
            config_file = _safe_input(config_path, purpose="embodied config", kind="file")
            config = load_config(config_file)
            base_map = _safe_input(
                project_path(config, "maps", scene_id, "voxel_map.npz"),
                purpose="base continuous semantic map",
                kind="file",
            )
            persistent_map = Path(temporary) / "semantic_map.npz"
            parameters: StdioServerParameters = _server_parameters(
                python_executable=(
                    sys.executable if python_executable is None else python_executable
                ),
                config=config_file,
                scene_id=scene_id,
                base_checkpoint=_safe_input(
                    base_checkpoint, purpose="base checkpoint", kind="directory"
                ),
                control_checkpoint=_safe_input(
                    control_checkpoint, purpose="control checkpoint", kind="directory"
                ),
                control_runtime_config=_safe_input(
                    control_runtime_config,
                    purpose="control runtime config",
                    kind="file",
                ),
                runtime_asset=_safe_input(
                    runtime_asset, purpose="sanitized runtime asset", kind="file"
                ),
                robot_state_checkpoint=_safe_input(
                    robot_state_checkpoint,
                    purpose="numeric robot-state checkpoint",
                    kind="directory",
                ),
                persistent_map=persistent_map,
                audit_report=_rooted(server_audit_report),
            )
            encoder = GemmaProjectedTextEncoder.from_config(config)
            with MCPConversationRuntime.connect_stdio(
                parameters,
                config,
                base_map_path=base_map,
                persistent_map_path=persistent_map,
            ) as runtime:
                result = run_face_instruction(
                    runtime,
                    encoder,
                    instruction,
                    room_size_m=config["scene"]["room_size_m"],
                    feature_start=int(config["vision"].get("aligned_feature_start", 1536)),
                    feature_dim=encoder.output_dim,
                    max_steps=max_steps,
                )
    except BaseException as caught:  # noqa: BLE001 - preserve audit before re-raising
        error = caught
    finally:
        audit.save(_rooted(client_audit_report))
    audit.assert_clean()
    if error is not None:
        raise error
    if result is None:  # pragma: no cover - defensive guard
        raise RuntimeError("Conversational MCP run produced no result")

    server_report = _safe_input(
        server_audit_report,
        purpose="server lifetime access audit",
        kind="file",
    )
    server_audit = json.loads(server_report.read_text(encoding="utf-8"))
    if (
        not isinstance(server_audit, dict)
        or server_audit.get("passed") is not True
        or server_audit.get("forbidden_accesses") != []
    ):
        raise RuntimeError("Semantic MCP server lifetime access audit failed")
    client_report = _safe_input(
        client_audit_report,
        purpose="client access audit",
        kind="file",
    )
    result.update(
        {
            "client_access_audit": {
                "path": str(client_report),
                "sha256": _sha256_file(client_report),
                "loaded_file_count": len(audit.unique_paths),
                "forbidden_access_count": 0,
                "passed": True,
            },
            "server_access_audit": {
                "path": str(server_report),
                "sha256": _sha256_file(server_report),
                "loaded_file_count": len(server_audit.get("loaded_files", [])),
                "forbidden_access_count": 0,
                "passed": True,
            },
            "elapsed_seconds": time.monotonic() - started,
        }
    )
    return result


def _print_interactive_turn(result: Mapping[str, Any]) -> None:
    receipt = result.get("numeric_tool_receipt")
    if not isinstance(receipt, Mapping):
        steps = result.get("steps")
        if isinstance(steps, list) and steps and isinstance(steps[-1], Mapping):
            receipt = steps[-1].get("numeric_tool_receipt")
    position = None
    yaw = None
    stopped = result.get("episode_stop_latched")
    if isinstance(receipt, Mapping):
        position = receipt.get("position_m")
        yaw = receipt.get("body_yaw_degrees")
        stopped = receipt.get("stopped", stopped)
    print(
        json.dumps(
            {
                "turn": result.get("turn_index"),
                "command": result.get("command_kind"),
                "passed": result.get("passed"),
                "termination": result.get("termination_reason"),
                "position_m": position,
                "body_yaw_degrees": yaw,
                "stopped": stopped,
            },
            sort_keys=True,
            allow_nan=False,
        ),
        flush=True,
    )


def run_session_live(
    *,
    config_path: str | Path,
    scene_id: str,
    commands: Sequence[str],
    interactive: bool,
    base_checkpoint: str | Path,
    control_checkpoint: str | Path,
    control_runtime_config: str | Path,
    runtime_asset: str | Path,
    robot_state_checkpoint: str | Path,
    client_audit_report: str | Path,
    server_audit_report: str | Path,
    python_executable: str | Path | None = None,
    max_steps: int = 12,
) -> dict[str, Any]:
    """Run repeated commands through one production MCP stdio subprocess."""

    if not interactive and not commands:
        raise ValueError("Persistent MCP run needs --interactive or at least one --command")
    started = time.monotonic()
    audit = _client_audit()
    result: dict[str, Any] | None = None
    error: BaseException | None = None
    workspace = PROJECT_ROOT / "reports" / "gemma4" / "artifacts"
    workspace.mkdir(parents=True, exist_ok=True)
    try:
        with (
            audit,
            tempfile.TemporaryDirectory(
                prefix=f"conversational_mcp_session_{scene_id}_",
                dir=workspace,
            ) as temporary,
        ):
            config_file = _safe_input(config_path, purpose="embodied config", kind="file")
            config = load_config(config_file)
            base_map = _safe_input(
                project_path(config, "maps", scene_id, "voxel_map.npz"),
                purpose="base continuous semantic map",
                kind="file",
            )
            persistent_map = Path(temporary) / "semantic_map.npz"
            parameters: StdioServerParameters = _server_parameters(
                python_executable=(
                    sys.executable if python_executable is None else python_executable
                ),
                config=config_file,
                scene_id=scene_id,
                base_checkpoint=_safe_input(
                    base_checkpoint, purpose="base checkpoint", kind="directory"
                ),
                control_checkpoint=_safe_input(
                    control_checkpoint, purpose="control checkpoint", kind="directory"
                ),
                control_runtime_config=_safe_input(
                    control_runtime_config,
                    purpose="control runtime config",
                    kind="file",
                ),
                runtime_asset=_safe_input(
                    runtime_asset, purpose="sanitized runtime asset", kind="file"
                ),
                robot_state_checkpoint=_safe_input(
                    robot_state_checkpoint,
                    purpose="numeric robot-state checkpoint",
                    kind="directory",
                ),
                persistent_map=persistent_map,
                audit_report=_rooted(server_audit_report),
            )
            encoder = GemmaProjectedTextEncoder.from_config(config)
            with MCPConversationRuntime.connect_stdio(
                parameters,
                config,
                base_map_path=base_map,
                persistent_map_path=persistent_map,
            ) as runtime:
                session = PersistentMCPConversationSession(
                    runtime,
                    encoder,
                    room_size_m=config["scene"]["room_size_m"],
                    feature_start=int(config["vision"].get("aligned_feature_start", 1536)),
                    feature_dim=encoder.output_dim,
                    max_steps=max_steps,
                )
                startup = session.start()
                if interactive:
                    print(
                        json.dumps(
                            {
                                "phase": "ready",
                                "scene_id": scene_id,
                                "policy": startup["policy"]["name"],
                                "prompt": "embodied> ",
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                try:
                    for command in commands:
                        turn = session.handle(command)
                        _print_interactive_turn(turn)
                        if runtime.simulator.state.stopped:
                            break
                    while interactive and not runtime.simulator.state.stopped:
                        try:
                            command = input("embodied> ")
                        except EOFError:
                            break
                        if command.strip().casefold() in {"exit", "quit"}:
                            break
                        try:
                            turn = session.handle(command)
                        except ValueError as invalid:
                            print(
                                json.dumps(
                                    {
                                        "passed": False,
                                        "error_code": "E_INSTRUCTION",
                                        "message": str(invalid),
                                    },
                                    sort_keys=True,
                                ),
                                flush=True,
                            )
                            continue
                        _print_interactive_turn(turn)
                finally:
                    session.shutdown(reason="client_exit")
                result = session.summary()
    except BaseException as caught:  # noqa: BLE001 - preserve audit before re-raising
        error = caught
    finally:
        audit.save(_rooted(client_audit_report))
    audit.assert_clean()
    if error is not None:
        raise error
    if result is None:  # pragma: no cover - defensive guard
        raise RuntimeError("Persistent conversational MCP run produced no result")

    server_report = _safe_input(
        server_audit_report,
        purpose="server lifetime access audit",
        kind="file",
    )
    server_audit = json.loads(server_report.read_text(encoding="utf-8"))
    if (
        not isinstance(server_audit, dict)
        or server_audit.get("passed") is not True
        or server_audit.get("forbidden_accesses") != []
    ):
        raise RuntimeError("Semantic MCP server lifetime access audit failed")
    client_report = _safe_input(
        client_audit_report,
        purpose="client access audit",
        kind="file",
    )
    result.update(
        {
            "client_access_audit": {
                "path": str(client_report),
                "sha256": _sha256_file(client_report),
                "loaded_file_count": len(audit.unique_paths),
                "forbidden_access_count": 0,
                "passed": True,
            },
            "server_access_audit": {
                "path": str(server_report),
                "sha256": _sha256_file(server_report),
                "loaded_file_count": len(server_audit.get("loaded_files", [])),
                "forbidden_access_count": 0,
                "passed": True,
            },
            "elapsed_seconds": time.monotonic() - started,
        }
    )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/runtime/embodied_live.yaml")
    parser.add_argument("--scene", default="scene_000001")
    parser.add_argument("--instruction", default="Face the target, then stop.")
    parser.add_argument(
        "--base-checkpoint",
        default="data_gemma4/runtime/checkpoints/gemma4_v54_release_v1",
    )
    parser.add_argument(
        "--control-checkpoint",
        default="data_gemma4/runtime/checkpoints/gemma4_v75_nll_control_release_v1",
    )
    parser.add_argument(
        "--control-runtime-config",
        default="configs/runtime/gemma4_v56_question_control.yaml",
    )
    parser.add_argument("--runtime-asset", required=True)
    parser.add_argument(
        "--robot-state-checkpoint",
        default="data_gemma4/checkpoints/robot_state_numeric_v1",
    )
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="keep one MCP stdio session open for repeated conversational commands",
    )
    parser.add_argument(
        "--command",
        action="append",
        default=[],
        help="execute one persistent-session command; repeat to run a finite sequence",
    )
    parser.add_argument("--output")
    parser.add_argument("--client-audit-report")
    parser.add_argument("--server-audit-report")
    parser.add_argument("--check", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    session_mode = bool(args.interactive or args.command)
    output = _rooted(
        args.output
        or (
            f"reports/gemma4/metrics/conversational_mcp_session_{args.scene}.json"
            if session_mode
            else f"reports/gemma4/metrics/conversational_mcp_face_{args.scene}.json"
        )
    )
    client_audit_report = _rooted(
        args.client_audit_report
        or (
            f"reports/gemma4/metrics/conversational_mcp_session_client_access_{args.scene}.json"
            if session_mode
            else f"reports/gemma4/metrics/conversational_mcp_client_access_{args.scene}.json"
        )
    )
    server_audit_report = _rooted(
        args.server_audit_report
        or (
            f"reports/gemma4/metrics/conversational_mcp_session_server_access_{args.scene}.json"
            if session_mode
            else f"reports/gemma4/metrics/conversational_mcp_server_access_{args.scene}.json"
        )
    )
    finite_common = {
        "config_path": args.config,
        "scene_id": args.scene,
        "instruction": args.instruction,
        "base_checkpoint": args.base_checkpoint,
        "control_checkpoint": args.control_checkpoint,
        "control_runtime_config": args.control_runtime_config,
        "runtime_asset": args.runtime_asset,
        "robot_state_checkpoint": args.robot_state_checkpoint,
        "max_steps": args.max_steps,
    }
    if args.check:
        if session_mode:
            result = _session_preflight(
                config_path=args.config,
                scene_id=args.scene,
                commands=args.command,
                base_checkpoint=args.base_checkpoint,
                control_checkpoint=args.control_checkpoint,
                control_runtime_config=args.control_runtime_config,
                runtime_asset=args.runtime_asset,
                robot_state_checkpoint=args.robot_state_checkpoint,
                max_steps=args.max_steps,
            )
        else:
            result = _preflight(**finite_common)
    elif session_mode:
        result = run_session_live(
            config_path=args.config,
            scene_id=args.scene,
            commands=args.command,
            interactive=args.interactive,
            base_checkpoint=args.base_checkpoint,
            control_checkpoint=args.control_checkpoint,
            control_runtime_config=args.control_runtime_config,
            runtime_asset=args.runtime_asset,
            robot_state_checkpoint=args.robot_state_checkpoint,
            client_audit_report=client_audit_report,
            server_audit_report=server_audit_report,
            max_steps=args.max_steps,
        )
    else:
        result = run_live(
            **finite_common,
            client_audit_report=client_audit_report,
            server_audit_report=server_audit_report,
        )
    destination = _atomic_json(output, result)
    print(
        json.dumps(
            {
                "schema": result["schema"],
                "passed": result["passed"],
                "scene_id": result["scene_id"],
                "phase": result.get("phase", "live"),
                "output": str(destination),
                "policy": result["policy"]["name"],
                "learned_v3_action_head_used": result["policy"]["learned_v3_action_head_used"],
                "gemma_native_function_calling_used": result["policy"][
                    "gemma_native_function_calling_used"
                ],
            },
            sort_keys=True,
            allow_nan=False,
        ),
        flush=True,
    )
    return 0 if result["passed"] is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
