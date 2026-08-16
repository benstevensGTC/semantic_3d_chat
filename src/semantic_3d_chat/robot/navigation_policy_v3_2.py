"""V3.2 development calibration for compound scan-then-approach goals.

The accepted V3 learned checkpoint still evaluates every complete scene token,
the local Gemma instruction embedding, freshly grounded all-voxel target state,
and numeric robot tokens on every decision.  This successor changes only the
post-policy bounded runtime interlock: after a requested scan has completed, a
deterministic geometry-only planner may convert the grounded XYZ into safe
``move_to`` waypoints.  No category inventory, task ID, simulator label,
caption, scene graph, oracle relationship, or oracle coordinate is available.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import replace
from pathlib import Path
from typing import Any, Final

import numpy as np

from semantic_3d_chat.robot.collision import NumericCollisionMap
from semantic_3d_chat.robot.llm_tool_policy import GeneratedToolProposal
from semantic_3d_chat.robot.navigation_policy_v3 import (
    SemanticGroundedActionBackendV3,
)
from semantic_3d_chat.robot.planner import NumericPathPlan, NumericWaypointPlanner
from semantic_3d_chat.robot.semantic_agent import ContinuousTextEncoder
from semantic_3d_chat.robot.semantic_mapping import semantic_map_content_hash

RUNTIME_INTERLOCK_VERSION: Final[str] = "v3.2"
_SPACE = re.compile(r"\s+")
_COMPOUND_APPROACH = re.compile(
    r"^(?:scan(?:\s+the\s+room)?|look\s+around)\s*,?\s+then\s+"
    r"(?:move\s+closer\s+to|approach|walk\s+toward|move\s+toward)\b"
    r".+(?:,?\s+then\s+stop|\s+and\s+stop)[.!]?\s*$",
    re.IGNORECASE,
)
_DEFAULT_CALIBRATED_STANDOFF_M: Final[float] = 0.35
_DEFAULT_GRID_RESOLUTION_M: Final[float] = 0.15
_DEFAULT_PLANNER_STANDOFF_TOLERANCE_M: Final[float] = 0.20
_WAYPOINT_REACHED_M: Final[float] = 1e-4


def is_compound_scan_approach_instruction(instruction: str) -> bool:
    """Match only action grammar while treating the target phrase as opaque."""

    literal = literal_navigation_instruction(instruction)
    return _COMPOUND_APPROACH.fullmatch(literal) is not None


def literal_navigation_instruction(policy_input: str) -> str:
    """Unwrap the exact benchmark/runtime protocol envelope, if present."""

    if not isinstance(policy_input, str):
        raise TypeError("V3.2 navigation instruction must be text")
    if not policy_input.strip():
        raise ValueError("V3.2 navigation instruction is empty")
    prefix = "User navigation instruction: "
    stripped = policy_input.strip()
    first_line = stripped.splitlines()[0]
    literal = first_line[len(prefix) :].strip() if first_line.startswith(prefix) else stripped
    literal = _SPACE.sub(" ", literal).strip()
    if not literal or len(literal) > 1024:
        raise ValueError("V3.2 literal navigation instruction is invalid")
    return literal


def _active_map_path(runtime: Any) -> Path:
    updater = getattr(runtime, "map_updater", None)
    persistent = getattr(updater, "persistent_map_path", None)
    base = getattr(updater, "base_map_path", None)
    if not isinstance(persistent, Path) or not isinstance(base, Path):
        raise TypeError("V3.2 runtime has no authenticated semantic-map paths")
    active = persistent if persistent.is_file() else base
    if not active.is_file() or active.is_symlink():
        raise FileNotFoundError(active)
    return active


def _finite_positive(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"V3.2 {name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"V3.2 {name} must be finite and positive")
    return result


def _collision_map_for_active_memory(
    runtime: Any,
    config: dict[str, Any],
    *,
    expected_map_sha256: str,
) -> NumericCollisionMap:
    path = _active_map_path(runtime)
    if semantic_map_content_hash(path) != expected_map_sha256:
        raise RuntimeError("V3.2 planner map differs from grounded continuous memory")
    robot = config["robot"]
    collision = NumericCollisionMap.from_voxel_map(
        path,
        room_size_m=config["scene"]["room_size_m"],
        robot_radius_m=float(robot["radius_m"]),
        collision_z_min_m=float(robot.get("collision_z_min_m", 0.12)),
        collision_z_max_m=float(robot.get("collision_z_max_m", 1.80)),
        surface_padding_m=float(robot.get("surface_padding_m", 0.035)),
    )
    # Dispatch and planning must use the same freshly rebuilt anonymous field.
    runtime.simulator.collision_map = collision
    return collision


def _plan_sha256(plan: NumericPathPlan) -> str:
    encoded = json.dumps(
        plan.as_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class SemanticGroundedActionBackendV32(SemanticGroundedActionBackendV3):
    """V3 checkpoint plus a generic continuous-grounding waypoint interlock."""

    runtime_interlock_version = RUNTIME_INTERLOCK_VERSION

    def __init__(
        self,
        runtime: Any,
        controller: Any,
        metadata: dict[str, Any],
        config: dict[str, Any],
        *,
        text_encoder: ContinuousTextEncoder | None = None,
    ) -> None:
        super().__init__(
            runtime,
            controller,
            metadata,
            config,
            text_encoder=text_encoder,
        )
        self._v32_config = config
        robot = config["robot"]
        self.compound_approach_standoff_m = _finite_positive(
            robot.get(
                "compound_scan_approach_standoff_m",
                _DEFAULT_CALIBRATED_STANDOFF_M,
            ),
            "compound_scan_approach_standoff_m",
        )
        if self.compound_approach_standoff_m >= self.approach_target_standoff_m:
            raise ValueError("V3.2 compound approach standoff must be below the V3 default")
        self.planner_grid_resolution_m = _finite_positive(
            robot.get("compound_scan_planner_grid_resolution_m", _DEFAULT_GRID_RESOLUTION_M),
            "compound_scan_planner_grid_resolution_m",
        )
        self.planner_standoff_tolerance_m = _finite_positive(
            robot.get(
                "compound_scan_planner_standoff_tolerance_m",
                _DEFAULT_PLANNER_STANDOFF_TOLERANCE_M,
            ),
            "compound_scan_planner_standoff_tolerance_m",
        )
        self._v32_instruction: str | None = None
        self._v32_scene_version: int | None = None
        self._v32_target_sha256: str | None = None
        self._v32_collision_map_sha256: str | None = None
        self._v32_collision_map: NumericCollisionMap | None = None
        self._v32_plan: NumericPathPlan | None = None
        self._v32_waypoints: list[tuple[float, float]] = []

    def numeric_alignment_interlock_summary(self) -> dict[str, Any]:
        summary = super().numeric_alignment_interlock_summary()
        summary["runtime_interlock_version"] = self.runtime_interlock_version
        summary["compound_scan_approach"] = {
            "enabled": True,
            "calibrated_semantic_standoff_m": self.compound_approach_standoff_m,
            "geometry_only_waypoint_planner": True,
            "all_segments_collision_checked": True,
            "maximum_waypoint_step_m": min(
                0.5, float(self._v32_config["robot"].get("max_move_to_m", 1.0))
            ),
            "target_source": "fresh_all_voxel_continuous_grounding_xyz",
            "environmental_text_inputs": [],
            "oracle_inputs_at_runtime": False,
        }
        return summary

    def _reset_v32(self, instruction: str | None) -> None:
        self._v32_instruction = instruction
        self._v32_scene_version = None
        self._v32_target_sha256 = None
        self._v32_collision_map_sha256 = None
        self._v32_collision_map = None
        self._v32_plan = None
        self._v32_waypoints = []

    def _planner_action(
        self,
        instruction: str,
    ) -> tuple[dict[str, Any], dict[str, Any]] | None:
        grounding = self.last_grounding
        state = self.runtime.simulator.state
        if not isinstance(grounding, dict) or grounding.get("target_available") is not True:
            return None
        target = np.asarray(grounding.get("target_xyz_m"), dtype=np.float64)
        if target.shape != (3,) or not np.isfinite(target).all():
            raise ValueError("V3.2 continuous grounding target is invalid")
        map_sha256 = grounding.get("map_sha256")
        if not isinstance(map_sha256, str):
            raise TypeError("V3.2 grounding lacks a map binding")
        position = np.asarray(state.position_xy_m, dtype=np.float64)
        target_digest = hashlib.sha256(target.astype(np.float32).tobytes()).hexdigest()
        version = int(state.scene_version)
        must_plan = (
            self._v32_plan is None
            or self._v32_scene_version != version
            or self._v32_target_sha256 != target_digest
        )
        if must_plan:
            collision = _collision_map_for_active_memory(
                self.runtime,
                self._v32_config,
                expected_map_sha256=map_sha256,
            )
            self._v32_collision_map = collision
            self._v32_collision_map_sha256 = map_sha256
            maximum_step = min(
                0.5,
                float(self._v32_config["robot"].get("max_move_to_m", 1.0)),
            )
            planner = NumericWaypointPlanner(
                collision,
                grid_resolution_m=self.planner_grid_resolution_m,
                standoff_m=self.compound_approach_standoff_m,
                standoff_tolerance_m=self.planner_standoff_tolerance_m,
                max_waypoint_step_m=maximum_step,
            )
            self._v32_plan = planner.plan(position, target[:2])
            self._v32_waypoints = list(self._v32_plan.waypoints_xy_m)
            self._v32_scene_version = version
            self._v32_target_sha256 = target_digest
        collision = self._v32_collision_map
        if collision is None or self._v32_collision_map_sha256 != map_sha256:
            raise RuntimeError("V3.2 cached collision field differs from active memory")

        while self._v32_waypoints:
            waypoint = np.asarray(self._v32_waypoints[0], dtype=np.float64)
            if float(np.linalg.norm(waypoint - position)) > _WAYPOINT_REACHED_M:
                break
            self._v32_waypoints.pop(0)
        assert self._v32_plan is not None
        plan_audit: dict[str, Any] = {
            "schema": "semantic_3d_chat.numeric_compound_approach_planner.v1",
            "runtime_interlock_version": self.runtime_interlock_version,
            "enabled": True,
            "instruction_sha256": hashlib.sha256(instruction.encode("utf-8")).hexdigest(),
            "target_xyz_m": target.tolist(),
            "target_sha256": target_digest,
            "map_sha256": map_sha256,
            "scene_version": version,
            "plan_sha256": _plan_sha256(self._v32_plan),
            "planned_waypoint_count": len(self._v32_plan.waypoints_xy_m),
            "remaining_waypoint_count": len(self._v32_waypoints),
            "calibrated_semantic_standoff_m": self.compound_approach_standoff_m,
            "collision_capped_target_distance_m": self._v32_plan.target_distance_m,
            "numeric_map_only": True,
            "all_map_voxels_scored_for_grounding": True,
            "environmental_text_inputs": [],
            "oracle_inputs_at_runtime": False,
        }
        if not self._v32_waypoints:
            plan_audit.update(
                {
                    "action": "stop",
                    "completion_satisfied": True,
                    "completion_mode": "collision_capped_numeric_waypoint_standoff",
                }
            )
            return {"tool": "stop", "arguments": {}}, plan_audit
        waypoint = np.asarray(self._v32_waypoints[0], dtype=np.float64)
        segment = collision.segment_check(position, waypoint)
        distance = float(np.linalg.norm(waypoint - position))
        maximum = float(self._v32_config["robot"].get("max_move_to_m", 1.0))
        if segment.collision or distance <= 0.0 or distance > maximum + 1e-9:
            raise RuntimeError("V3.2 planner released an unsafe bounded waypoint")
        plan_audit.update(
            {
                "action": "move_to",
                "next_waypoint_xy_m": waypoint.tolist(),
                "next_waypoint_distance_m": distance,
                "next_waypoint_clearance_m": segment.clearance_m,
                "completion_satisfied": False,
                "completion_mode": None,
            }
        )
        return {
            "tool": "move_to",
            "arguments": {"x": float(waypoint[0]), "y": float(waypoint[1])},
        }, plan_audit

    def generate(
        self,
        instruction: str,
        *,
        correction_code: str | None,
    ) -> GeneratedToolProposal:
        compound = is_compound_scan_approach_instruction(instruction)
        normalized = literal_navigation_instruction(instruction)
        if not compound:
            self._reset_v32(None)
            return super().generate(instruction, correction_code=correction_code)
        if self._v32_instruction != normalized or self.runtime.simulator.state.scan_count < 1:
            self._reset_v32(normalized)

        original_standoff = self.approach_target_standoff_m
        self.approach_target_standoff_m = self.compound_approach_standoff_m
        try:
            proposal = super().generate(instruction, correction_code=correction_code)
        finally:
            self.approach_target_standoff_m = original_standoff

        if self.runtime.simulator.state.scan_count < 1:
            return proposal
        replacement = self._planner_action(normalized)
        if replacement is None:
            return proposal
        call, planner_audit = replacement
        if isinstance(self.last_grounding, dict):
            self.last_grounding["numeric_compound_approach_planner"] = planner_audit
            approach = self.last_grounding.get("numeric_approach_interlock")
            if isinstance(approach, dict):
                approach["runtime_interlock_version"] = self.runtime_interlock_version
                approach["target_standoff_m"] = self.compound_approach_standoff_m
        return replace(
            proposal,
            text=json.dumps(call, sort_keys=True, separators=(",", ":"), allow_nan=False),
        )


__all__ = [
    "RUNTIME_INTERLOCK_VERSION",
    "SemanticGroundedActionBackendV32",
    "is_compound_scan_approach_instruction",
    "literal_navigation_instruction",
]
