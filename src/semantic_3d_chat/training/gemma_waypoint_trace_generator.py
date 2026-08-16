"""Training-only expert traces for a Gemma absolute-waypoint policy.

The deployable controller is expected to choose every absolute heading and XY
waypoint itself.  Deterministic geometry planners are used here only as offline
teachers: they are not a runtime fallback and their plans are never copied into
runtime artifacts.  Environmental words and oracle coordinates therefore stay
inside a path containing a ``training`` component.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations, pairwise
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch

from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.evaluation.gemma_waypoint_object_eval import (
    RUNTIME_SCHEMA as OBJECT_RUNTIME_SCHEMA,
)
from semantic_3d_chat.evaluation.gemma_waypoint_object_eval import (
    SCORE_SCHEMA as OBJECT_SCORE_SCHEMA,
)
from semantic_3d_chat.evaluation.gemma_waypoint_object_eval import (
    validate_runtime_record as validate_object_runtime_record,
)
from semantic_3d_chat.robot.collision import NumericCollisionMap
from semantic_3d_chat.robot.planner import NumericWaypointPlanner
from semantic_3d_chat.robot.semantic_patrol import NumericPatrolPlanner
from semantic_3d_chat.robot.state_encoder import (
    ROBOT_STATE_FEATURE_DIM,
    NumericRobotState,
    robot_state_vector,
)
from semantic_3d_chat.robot.waypoint_history import (
    HISTORY_FEATURE_DIM,
    HISTORY_FEATURE_DIM_V2,
    HISTORY_PARAMETERIZATION,
    HISTORY_PARAMETERIZATION_V2,
    WaypointGoalProgressLedger,
    encode_waypoint_history_transition,
    encode_waypoint_history_transition_v2,
)
from semantic_3d_chat.training.navigation_target_trace_v3 import (
    load_navigation_target_trace_v3,
)

TRACE_SCHEMA: Final[str] = "semantic_3d_chat.gemma_waypoint_trace_sample.v1"
MANIFEST_SCHEMA: Final[str] = "semantic_3d_chat.gemma_waypoint_trace_dataset.v1"
ACTION_NAMES: Final[tuple[str, ...]] = ("MOVE_TO", "FACE", "STOP")
ACTION_TO_CODE: Final[dict[str, int]] = {
    name: index for index, name in enumerate(ACTION_NAMES)
}
_SUPPORTED_HISTORY_CONTRACTS: Final[dict[str, int]] = {
    HISTORY_PARAMETERIZATION: HISTORY_FEATURE_DIM,
    HISTORY_PARAMETERIZATION_V2: HISTORY_FEATURE_DIM_V2,
}
_SCENE_ID = re.compile(r"scene_[0-9]{6}")
_STRUCTURAL_CATEGORIES: Final[frozenset[str]] = frozenset(
    {"floor", "wall", "ceiling"}
)
_ROW_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema",
        "sample_id",
        "episode_id",
        "scene_id",
        "split",
        "family",
        "task_variant",
        "instruction",
        "step_index",
        "state_features",
        "history_pose_xy_yaw",
        "history_action_codes",
        "history",
        "action",
        "expert_action",
        "expert_action_code",
        "expert_heading_degrees",
        "expert_xy_m",
        "waypoint_delta_robot_m",
        "heading_degrees",
        "episode_goal_xy_m",
        "training_target_xyz_m",
        "source_sample_sha256",
        "collision_safe_target",
        "environmental_text_training_only",
        "oracle_available_at_runtime",
        "expert_planner_available_at_runtime",
    }
)


def _rooted(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    rooted = candidate if candidate.is_absolute() else PROJECT_ROOT / candidate
    return Path(os.path.abspath(rooted))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_training_path(path: Path) -> None:
    if "training" not in {part.casefold() for part in path.parts}:
        raise ValueError("Gemma waypoint traces must remain under a training tree")
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current /= component
        if current.is_symlink():
            raise ValueError("Gemma waypoint trace paths cannot contain symbolic links")


def _normalize_degrees(value: float) -> float:
    result = (float(value) + 180.0) % 360.0 - 180.0
    # Avoid a platform-dependent negative zero in the authenticated JSON.
    return 0.0 if abs(result) < 1e-12 else result


def _heading_between(start_xy: Sequence[float], end_xy: Sequence[float]) -> float:
    start = np.asarray(start_xy, dtype=np.float64)
    end = np.asarray(end_xy, dtype=np.float64)
    delta = end - start
    if start.shape != (2,) or end.shape != (2,) or not np.isfinite(delta).all():
        raise ValueError("Waypoint heading requires two finite XY points")
    if float(np.linalg.norm(delta)) <= 1e-12:
        raise ValueError("A zero-length movement has no expert heading")
    # The simulator's forward vector is [-sin(yaw), cos(yaw)].
    return _normalize_degrees(
        math.degrees(math.atan2(-float(delta[0]), float(delta[1])))
    )


@dataclass(frozen=True)
class AbsoluteWaypointAction:
    """One V3 bounded action converted to an absolute policy target."""

    action: str
    current_pose_xy_yaw: tuple[float, float, float]
    completion_pose_xy_yaw: tuple[float, float, float]
    heading_degrees: float | None
    xy_m: tuple[float, float] | None


def convert_v3_action_to_absolute(
    row: Mapping[str, Any],
    *,
    room_size_m: Sequence[float],
    max_turn_degrees: float,
    max_move_m: float,
) -> AbsoluteWaypointAction | None:
    """Convert one authenticated V3 target to FACE/MOVE_TO/STOP.

    ``scan`` is intentionally omitted: newly observed continuous scene tokens
    are a runtime state update, not a spatial waypoint chosen by the policy.
    """

    room = np.asarray(room_size_m, dtype=np.float64)
    state = np.asarray(row.get("state_features"), dtype=np.float64)
    argument = row.get("argument_target_normalized")
    action_name = row.get("action_name")
    if (
        room.shape != (3,)
        or not np.isfinite(room).all()
        or np.any(room <= 0)
        or state.shape != (ROBOT_STATE_FEATURE_DIM,)
        or not np.isfinite(state).all()
        or isinstance(argument, bool)
        or not isinstance(argument, (int, float))
        or not math.isfinite(float(argument))
        or not math.isfinite(float(max_turn_degrees))
        or max_turn_degrees <= 0
        or not math.isfinite(float(max_move_m))
        or max_move_m <= 0
    ):
        raise ValueError("V3 action cannot be converted from invalid numeric fields")
    normalized = float(argument)
    if abs(normalized) > 1.0 + 1e-6:
        raise ValueError("V3 normalized action target lies outside [-1, 1]")
    normalized = min(1.0, max(-1.0, normalized))
    lower = np.asarray([-room[0] / 2.0, -room[1] / 2.0, 0.0])
    position = lower + (state[:3] + 1.0) * 0.5 * room
    yaw = _normalize_degrees(math.degrees(math.atan2(state[3], state[4])))
    current = (float(position[0]), float(position[1]), yaw)

    if action_name == "scan":
        return None
    if action_name == "stop":
        return AbsoluteWaypointAction("STOP", current, current, None, None)
    if action_name == "turn":
        target = _normalize_degrees(yaw + normalized * float(max_turn_degrees))
        return AbsoluteWaypointAction(
            "FACE", current, (current[0], current[1], target), target, None
        )
    if action_name not in {"move_forward", "move_backward"}:
        raise ValueError(f"Unsupported V3 source action: {action_name!r}")
    distance = (normalized + 1.0) * 0.5 * float(max_move_m)
    radians = math.radians(yaw)
    direction = np.asarray([-math.sin(radians), math.cos(radians)])
    if action_name == "move_backward":
        direction *= -1.0
    endpoint = position[:2] + distance * direction
    target_xy = (float(endpoint[0]), float(endpoint[1]))
    return AbsoluteWaypointAction(
        "MOVE_TO",
        current,
        (target_xy[0], target_xy[1], yaw),
        None,
        target_xy,
    )


@dataclass
class _Pose:
    x: float
    y: float
    yaw: float
    last_delta: tuple[float, float, float] = (0.0, 0.0, 0.0)
    linear_velocity: tuple[float, float] = (0.0, 0.0)
    angular_velocity: float = 0.0

    def triple(self) -> tuple[float, float, float]:
        return (float(self.x), float(self.y), _normalize_degrees(self.yaw))


@dataclass(frozen=True)
class _HistoryEncoding:
    """Exact offline/live numeric-history contract for one dataset build."""

    parameterization: str
    feature_dim: int
    rejection_streak_scale: int

    @classmethod
    def from_settings(
        cls, settings: Mapping[str, Any], *, history_length: int
    ) -> _HistoryEncoding:
        parameterization = str(
            settings.get("history_parameterization", HISTORY_PARAMETERIZATION)
        )
        expected_dim = _SUPPORTED_HISTORY_CONTRACTS.get(parameterization)
        configured_dim = settings.get("history_feature_dim", expected_dim)
        if (
            expected_dim is None
            or isinstance(configured_dim, bool)
            or not isinstance(configured_dim, int)
            or configured_dim != expected_dim
            or history_length < 1
        ):
            raise ValueError("Waypoint history parameterization/dimension pair differs")
        return cls(
            parameterization=parameterization,
            feature_dim=expected_dim,
            rejection_streak_scale=history_length,
        )

    @property
    def is_v2(self) -> bool:
        return self.parameterization == HISTORY_PARAMETERIZATION_V2

    def new_ledger(
        self, initial_pose_xy_yaw: Sequence[float]
    ) -> WaypointGoalProgressLedger:
        return WaypointGoalProgressLedger.from_initial_pose(initial_pose_xy_yaw)

    def encode_receipt(
        self,
        *,
        ledger: WaypointGoalProgressLedger,
        action: str,
        before_pose_xy_yaw: Sequence[float],
        result_pose_xy_yaw: Sequence[float],
        requested_waypoint_delta_robot_m: Sequence[float],
        requested_heading_degrees: float,
        room_size_m: Sequence[float],
        max_waypoint_step_m: float,
        success: bool,
    ) -> tuple[float, ...]:
        ledger.record_receipt(
            before_pose_xy_yaw=before_pose_xy_yaw,
            after_pose_xy_yaw=result_pose_xy_yaw,
            success=success,
        )
        common = {
            "action": action,
            "result_pose_xy_yaw": result_pose_xy_yaw,
            "requested_waypoint_delta_robot_m": requested_waypoint_delta_robot_m,
            "requested_heading_degrees": requested_heading_degrees,
            "room_size_m": room_size_m,
            "max_waypoint_step_m": max_waypoint_step_m,
            "success": success,
        }
        if not self.is_v2:
            return encode_waypoint_history_transition(**common)
        return encode_waypoint_history_transition_v2(
            **common,
            goal_progress=ledger.normalized_features(
                room_size_m=room_size_m,
                rejection_streak_scale=self.rejection_streak_scale,
            ),
        )


def _copy_goal_progress_ledger(
    ledger: WaypointGoalProgressLedger,
) -> WaypointGoalProgressLedger:
    return WaypointGoalProgressLedger(
        start_xy_m=ledger.start_xy_m,
        current_xy_m=ledger.current_xy_m,
        cumulative_accepted_path_m=ledger.cumulative_accepted_path_m,
        accepted_edge_cross_sum_m2=ledger.accepted_edge_cross_sum_m2,
        consecutive_rejections=ledger.consecutive_rejections,
    )


def _v1_history_encoding(history_length: int) -> _HistoryEncoding:
    return _HistoryEncoding(
        parameterization=HISTORY_PARAMETERIZATION,
        feature_dim=HISTORY_FEATURE_DIM,
        rejection_streak_scale=history_length,
    )


def _numeric_state(pose: _Pose, room_size_m: Sequence[float]) -> list[float]:
    room = torch.tensor(room_size_m, dtype=torch.float32)
    state = NumericRobotState(
        position_m=(pose.x, pose.y, 0.0),
        body_yaw_degrees=pose.yaw,
        camera_yaw_degrees=pose.yaw,
        pitch_degrees=0.0,
        linear_velocity_xy_m=pose.linear_velocity,
        angular_velocity_degrees=pose.angular_velocity,
        collision=False,
        last_movement_delta_m=pose.last_delta,
        scan_coverage=0.0,
        stopped=False,
    )
    vector = robot_state_vector(
        state,
        torch.tensor([-room[0] / 2.0, -room[1] / 2.0, 0.0]),
        torch.tensor([room[0] / 2.0, room[1] / 2.0, room[2]]),
    )
    return [float(value) for value in vector.tolist()]


def _pose_close(first: Sequence[float], second: Sequence[float]) -> bool:
    return (
        math.dist(first[:2], second[:2]) <= 2e-5
        and abs(_normalize_degrees(float(first[2]) - float(second[2]))) <= 2e-4
    )


def _history_payload(
    poses: list[tuple[float, float, float]],
    action_codes: list[int],
    history_length: int,
    *,
    room_size_m: Sequence[float],
    max_waypoint_step_m: float,
    history_encoding: _HistoryEncoding,
    numeric_rows: Sequence[Sequence[float]] | None = None,
) -> tuple[list[list[float]], list[int], list[list[float]]]:
    if len(poses) != len(action_codes) + 1:
        raise RuntimeError("Numeric action history is not transition-aligned")
    explicit = None
    if numeric_rows is not None:
        explicit = (
            np.empty((0, history_encoding.feature_dim), dtype=np.float64)
            if not action_codes and len(numeric_rows) == 0
            else np.asarray(numeric_rows, dtype=np.float64)
        )
        if explicit.shape != (len(action_codes), history_encoding.feature_dim):
            raise ValueError("Explicit numeric history is not action-aligned")
        if not np.isfinite(explicit).all():
            raise ValueError("Explicit numeric history contains non-finite values")
    keep_actions = min(len(action_codes), history_length)
    pose_tail = poses[-(keep_actions + 1) :]
    action_tail = action_codes[-keep_actions:] if keep_actions else []
    if explicit is not None:
        numeric_tail = explicit[-keep_actions:] if keep_actions else explicit[:0]
        return (
            [list(value) for value in pose_tail],
            list(action_tail),
            [[float(value) for value in row] for row in numeric_tail],
        )
    if history_encoding.is_v2 and action_tail:
        raise ValueError("V2 history reconstruction requires the goal-scoped ledger")
    numeric_history: list[list[float]] = []
    for index, action_code in enumerate(action_tail):
        start = pose_tail[index]
        result = pose_tail[index + 1]
        action = ACTION_NAMES[action_code]
        if action == "MOVE_TO":
            delta_world = np.asarray(result[:2]) - np.asarray(start[:2])
            radians = math.radians(float(start[2]))
            right = np.asarray([math.cos(radians), math.sin(radians)])
            forward = np.asarray([-math.sin(radians), math.cos(radians)])
            requested_delta = (
                float(np.dot(delta_world, right)),
                float(np.dot(delta_world, forward)),
            )
            requested_heading = float(result[2])
        else:
            requested_delta = (0.0, 0.0)
            requested_heading = float(result[2])
        numeric_history.append(
            list(
                encode_waypoint_history_transition(
                    action=action,
                    result_pose_xy_yaw=result,
                    requested_waypoint_delta_robot_m=requested_delta,
                    requested_heading_degrees=requested_heading,
                    room_size_m=room_size_m,
                    max_waypoint_step_m=max_waypoint_step_m,
                    success=True,
                )
            )
        )
    return ([list(value) for value in pose_tail], list(action_tail), numeric_history)


def _world_target_to_robot_delta(
    current_pose_xy_yaw: Sequence[float], target_xy_m: Sequence[float]
) -> tuple[float, float]:
    current = np.asarray(current_pose_xy_yaw, dtype=np.float64)
    target = np.asarray(target_xy_m, dtype=np.float64)
    delta_world = target - current[:2]
    radians = math.radians(float(current[2]))
    right = np.asarray([math.cos(radians), math.sin(radians)])
    forward = np.asarray([-math.sin(radians), math.cos(radians)])
    return (
        float(np.dot(delta_world, right)),
        float(np.dot(delta_world, forward)),
    )


def _make_row(
    *,
    episode_id: str,
    scene_id: str,
    split: str,
    family: str,
    task_variant: str,
    instruction: str,
    step_index: int,
    state_features: Sequence[float],
    history_poses: list[tuple[float, float, float]],
    history_action_codes: list[int],
    history_length: int,
    room_size_m: Sequence[float],
    max_waypoint_step_m: float,
    history_encoding: _HistoryEncoding,
    action: str,
    heading_degrees: float | None,
    xy_m: tuple[float, float] | None,
    episode_goal_xy_m: tuple[float, float] | None,
    training_target_xyz_m: Sequence[float] | None,
    source_sample_sha256: str | None,
    history_numeric_rows: Sequence[Sequence[float]] | None = None,
) -> dict[str, Any]:
    poses, action_codes, numeric_history = _history_payload(
        history_poses,
        history_action_codes,
        history_length,
        room_size_m=room_size_m,
        max_waypoint_step_m=max_waypoint_step_m,
        history_encoding=history_encoding,
        numeric_rows=history_numeric_rows,
    )
    current_pose = history_poses[-1]
    waypoint_delta = (
        None
        if action != "MOVE_TO" or xy_m is None
        else list(_world_target_to_robot_delta(current_pose, xy_m))
    )
    row = {
        "schema": TRACE_SCHEMA,
        "sample_id": "pending",
        "episode_id": episode_id,
        "scene_id": scene_id,
        "split": split,
        "family": family,
        "task_variant": task_variant,
        "instruction": instruction,
        "step_index": int(step_index),
        "state_features": [float(value) for value in state_features],
        "history_pose_xy_yaw": poses,
        "history_action_codes": action_codes,
        "history": numeric_history,
        "action": action.casefold(),
        "expert_action": action,
        "expert_action_code": ACTION_TO_CODE[action],
        "expert_heading_degrees": (
            None if heading_degrees is None else float(heading_degrees)
        ),
        "expert_xy_m": None if xy_m is None else [float(xy_m[0]), float(xy_m[1])],
        "waypoint_delta_robot_m": waypoint_delta,
        "heading_degrees": (
            None if heading_degrees is None else float(heading_degrees)
        ),
        "episode_goal_xy_m": (
            None
            if episode_goal_xy_m is None
            else [float(episode_goal_xy_m[0]), float(episode_goal_xy_m[1])]
        ),
        "training_target_xyz_m": (
            None
            if training_target_xyz_m is None
            else [float(value) for value in training_target_xyz_m]
        ),
        "source_sample_sha256": source_sample_sha256,
        "collision_safe_target": True,
        "environmental_text_training_only": True,
        "oracle_available_at_runtime": False,
        "expert_planner_available_at_runtime": False,
    }
    if set(row) != _ROW_FIELDS:
        raise RuntimeError("Internal waypoint row fields differ from the schema")
    return row


class _SyntheticTraceBuilder:
    def __init__(
        self,
        *,
        episode_id: str,
        scene_id: str,
        split: str,
        family: str,
        task_variant: str,
        instruction: str,
        start_xy: Sequence[float],
        initial_yaw: float,
        episode_goal_xy_m: tuple[float, float] | None,
        training_target_xyz_m: tuple[float, float, float] | None,
        collision_map: NumericCollisionMap,
        room_size_m: Sequence[float],
        max_waypoint_step_m: float,
        max_turn_degrees: float,
        history_length: int,
        history_encoding: _HistoryEncoding | None = None,
        source_sample_sha256: str | None = None,
    ) -> None:
        self.episode_id = episode_id
        self.scene_id = scene_id
        self.split = split
        self.family = family
        self.task_variant = task_variant
        self.instruction = instruction
        self.pose = _Pose(float(start_xy[0]), float(start_xy[1]), initial_yaw)
        self.episode_goal = episode_goal_xy_m
        self.training_target_xyz = training_target_xyz_m
        self.collision_map = collision_map
        self.room_size = room_size_m
        self.max_step = float(max_waypoint_step_m)
        self.max_turn = float(max_turn_degrees)
        if not math.isfinite(self.max_turn) or self.max_turn <= 0.0:
            raise ValueError("Synthetic FACE bound must be finite and positive")
        self.history_length = history_length
        self.history_encoding = history_encoding or _HistoryEncoding(
            parameterization=HISTORY_PARAMETERIZATION,
            feature_dim=HISTORY_FEATURE_DIM,
            rejection_streak_scale=history_length,
        )
        self.source_sample_sha256 = source_sample_sha256
        self.history_poses = [self.pose.triple()]
        self.history_actions: list[int] = []
        self.history_numeric_rows: list[list[float]] = []
        self.goal_progress_ledger = self.history_encoding.new_ledger(
            self.pose.triple()
        )
        self.rows: list[dict[str, Any]] = []

    def _record(
        self,
        action: str,
        *,
        heading: float | None = None,
        xy: tuple[float, float] | None = None,
    ) -> None:
        self.rows.append(
            _make_row(
                episode_id=self.episode_id,
                scene_id=self.scene_id,
                split=self.split,
                family=self.family,
                task_variant=self.task_variant,
                instruction=self.instruction,
                step_index=len(self.rows),
                state_features=_numeric_state(self.pose, self.room_size),
                history_poses=self.history_poses,
                history_action_codes=self.history_actions,
                history_length=self.history_length,
                room_size_m=self.room_size,
                max_waypoint_step_m=self.max_step,
                history_encoding=self.history_encoding,
                action=action,
                heading_degrees=heading,
                xy_m=xy,
                episode_goal_xy_m=self.episode_goal,
                training_target_xyz_m=self.training_target_xyz,
                source_sample_sha256=self.source_sample_sha256,
                history_numeric_rows=self.history_numeric_rows,
            )
        )

    def seed_training_history(
        self,
        *,
        poses: Sequence[Sequence[float]],
        action_codes: Sequence[int],
        numeric_rows: Sequence[Sequence[float]],
        goal_progress_ledger: WaypointGoalProgressLedger | None = None,
    ) -> None:
        """Seed an offline continuation with authenticated numeric history."""

        seeded_poses = [tuple(float(value) for value in pose) for pose in poses]
        seeded_actions = [int(value) for value in action_codes]
        seeded_numeric = [[float(value) for value in row] for row in numeric_rows]
        if (
            len(seeded_poses) != len(seeded_actions) + 1
            or len(seeded_numeric) != len(seeded_actions)
            or not _pose_close(seeded_poses[-1], self.pose.triple())
            or np.asarray(seeded_numeric, dtype=np.float64).shape
            != (len(seeded_actions), self.history_encoding.feature_dim)
            or (
                self.history_encoding.is_v2
                and goal_progress_ledger is None
            )
        ):
            raise ValueError("Seeded recovery history is not transition-aligned")
        self.history_poses = seeded_poses
        self.history_actions = seeded_actions
        self.history_numeric_rows = seeded_numeric
        if goal_progress_ledger is not None:
            if not _pose_close(
                (*goal_progress_ledger.current_xy_m, self.pose.yaw),
                self.pose.triple(),
            ):
                raise ValueError("Seeded goal-progress ledger pose differs")
            self.goal_progress_ledger = _copy_goal_progress_ledger(
                goal_progress_ledger
            )

    def _append_success_history(
        self,
        *,
        action: str,
        completion_pose: tuple[float, float, float],
        requested_waypoint_delta_robot_m: Sequence[float],
        requested_heading_degrees: float,
    ) -> None:
        before_pose = self.history_poses[-1]
        self.history_poses.append(completion_pose)
        self.history_actions.append(ACTION_TO_CODE[action])
        self.history_numeric_rows.append(
            list(
                self.history_encoding.encode_receipt(
                    ledger=self.goal_progress_ledger,
                    action=action.casefold(),
                    before_pose_xy_yaw=before_pose,
                    result_pose_xy_yaw=completion_pose,
                    requested_waypoint_delta_robot_m=requested_waypoint_delta_robot_m,
                    requested_heading_degrees=requested_heading_degrees,
                    room_size_m=self.room_size,
                    max_waypoint_step_m=self.max_step,
                    success=True,
                )
            )
        )

    def face(self, heading_degrees: float) -> None:
        heading = _normalize_degrees(heading_degrees)
        if abs(_normalize_degrees(heading - self.pose.yaw)) > self.max_turn + 1e-7:
            raise RuntimeError("Synthetic FACE target exceeds the bounded turn step")
        self._record("FACE", heading=heading)
        delta = _normalize_degrees(heading - self.pose.yaw)
        self.pose = _Pose(
            self.pose.x,
            self.pose.y,
            heading,
            last_delta=self.pose.last_delta,
            angular_velocity=delta,
        )
        completion = self.pose.triple()
        self._append_success_history(
            action="FACE",
            completion_pose=completion,
            requested_waypoint_delta_robot_m=(0.0, 0.0),
            requested_heading_degrees=heading,
        )

    def face_to(self, heading_degrees: float, *, ensure_action: bool = False) -> None:
        target = _normalize_degrees(heading_degrees)
        delta = _normalize_degrees(target - self.pose.yaw)
        emitted = False
        while abs(delta) > 1e-7:
            amount = math.copysign(min(abs(delta), self.max_turn), delta)
            self.face(_normalize_degrees(self.pose.yaw + amount))
            emitted = True
            delta = _normalize_degrees(target - self.pose.yaw)
        if ensure_action and not emitted:
            self.face(target)

    def face_toward_in_fixed_steps(
        self,
        heading_degrees: float,
        *,
        step_degrees: float,
    ) -> None:
        """Emit only full fixed turns toward a heading and omit the remainder.

        This is used only by the offline lap teacher.  A following MOVE_TO row
        is encoded relative to the coarse heading actually reached here, so no
        deterministic correction or residual turn is required at runtime.
        """

        step = float(step_degrees)
        if (
            not math.isfinite(step)
            or step <= 0.0
            or step > self.max_turn + 1e-7
        ):
            raise ValueError("Fixed FACE step is outside the bounded turn range")
        target = _normalize_degrees(heading_degrees)
        delta = _normalize_degrees(target - self.pose.yaw)
        while abs(delta) >= step - 1e-7:
            amount = math.copysign(step, delta)
            self.face(_normalize_degrees(self.pose.yaw + amount))
            delta = _normalize_degrees(target - self.pose.yaw)

    def face_with_execution_drift(
        self,
        heading_degrees: float,
        *,
        executed_delta_degrees: float,
    ) -> None:
        """Label the expert FACE while replaying a bounded behavior turn.

        This is an offline DAgger primitive.  The row target remains the exact
        expert heading, but the following numeric state/history records the
        slightly different turn actually emitted by a behavior policy.  No
        execution drift or correction is available to the deployed runtime.
        """

        heading = _normalize_degrees(heading_degrees)
        desired_delta = _normalize_degrees(heading - self.pose.yaw)
        executed_delta = float(executed_delta_degrees)
        if (
            not math.isfinite(executed_delta)
            or abs(desired_delta) > self.max_turn + 1e-7
            or abs(executed_delta) > self.max_turn + 1e-7
            or abs(executed_delta) <= 1e-9
            or desired_delta * executed_delta <= 0.0
            or abs(executed_delta) > abs(desired_delta) + 1e-7
        ):
            raise ValueError("Synthetic FACE execution drift is outside its label")
        self._record("FACE", heading=heading)
        actual_heading = _normalize_degrees(self.pose.yaw + executed_delta)
        self.pose = _Pose(
            self.pose.x,
            self.pose.y,
            actual_heading,
            last_delta=self.pose.last_delta,
            angular_velocity=executed_delta,
        )
        completion = self.pose.triple()
        self._append_success_history(
            action="FACE",
            completion_pose=completion,
            requested_waypoint_delta_robot_m=(0.0, 0.0),
            requested_heading_degrees=actual_heading,
        )

    def face_toward_with_execution_drift(
        self,
        heading_degrees: float,
        *,
        step_degrees: float,
        executed_magnitudes_degrees: Sequence[float],
        profile_index: int,
    ) -> int:
        """Replay cumulative, transition-aligned under-turns toward a target."""

        step = float(step_degrees)
        magnitudes = tuple(float(value) for value in executed_magnitudes_degrees)
        if (
            not math.isfinite(step)
            or step <= 0.0
            or step > self.max_turn + 1e-7
            or not magnitudes
            or any(
                not math.isfinite(value) or value <= 0.0 or value > step + 1e-7
                for value in magnitudes
            )
            or profile_index < 0
        ):
            raise ValueError("FACE execution-drift profile is invalid")
        target = _normalize_degrees(heading_degrees)
        delta = _normalize_degrees(target - self.pose.yaw)
        while abs(delta) >= step - 1e-7:
            desired_amount = math.copysign(step, delta)
            magnitude = magnitudes[profile_index % len(magnitudes)]
            self.face_with_execution_drift(
                _normalize_degrees(self.pose.yaw + desired_amount),
                executed_delta_degrees=math.copysign(magnitude, desired_amount),
            )
            profile_index += 1
            delta = _normalize_degrees(target - self.pose.yaw)
        return profile_index

    def _validated_move_geometry(
        self, target_xy: Sequence[float]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        target = np.asarray(target_xy, dtype=np.float64)
        start = np.asarray([self.pose.x, self.pose.y], dtype=np.float64)
        if target.shape != (2,) or not np.isfinite(target).all():
            raise ValueError("Synthetic expert waypoint is invalid")
        delta = target - start
        distance = float(np.linalg.norm(delta))
        if distance <= 1e-9:
            return target, start, delta
        if distance > self.max_step + 1e-7:
            raise RuntimeError("Synthetic expert waypoint exceeds the bounded step")
        if self.collision_map.segment_check(start, target).collision:
            raise RuntimeError("Synthetic expert waypoint segment is in collision")
        return target, start, delta

    def _move_after_facing(
        self,
        *,
        target: np.ndarray,
        delta: np.ndarray,
    ) -> None:
        self._record("MOVE_TO", xy=(float(target[0]), float(target[1])))
        requested_delta = _world_target_to_robot_delta(self.pose.triple(), target)
        self.pose = _Pose(
            float(target[0]),
            float(target[1]),
            self.pose.yaw,
            last_delta=(float(delta[0]), float(delta[1]), 0.0),
            linear_velocity=(float(delta[0]), float(delta[1])),
        )
        self._append_success_history(
            action="MOVE_TO",
            completion_pose=self.pose.triple(),
            requested_waypoint_delta_robot_m=requested_delta,
            requested_heading_degrees=self.pose.yaw,
        )

    def move_to(
        self,
        target_xy: Sequence[float],
        *,
        fixed_face_step_degrees: float | None = None,
    ) -> None:
        target, start, delta = self._validated_move_geometry(target_xy)
        distance = float(np.linalg.norm(delta))
        if distance <= 1e-9:
            return
        target_heading = _heading_between(start, target)
        if fixed_face_step_degrees is None:
            self.face_to(target_heading, ensure_action=True)
        else:
            self.face_toward_in_fixed_steps(
                target_heading,
                step_degrees=fixed_face_step_degrees,
            )
        self._move_after_facing(target=target, delta=delta)

    def move_to_direct(self, target_xy: Sequence[float]) -> None:
        """Emit one bounded MOVE_TO without inserting an offline FACE label.

        The deployed Gemma head natively selects a robot-relative 2D delta, so
        MOVE_TO is not constrained to the current forward axis. This primitive
        is used for a live pre-divergence correction whose correct class is
        MOVE_TO at the exact current heading.
        """

        target, _, delta = self._validated_move_geometry(target_xy)
        if float(np.linalg.norm(delta)) <= 1e-9:
            raise RuntimeError("Direct synthetic MOVE_TO target is unchanged")
        self._move_after_facing(target=target, delta=delta)

    def move_to_with_execution_drift(
        self,
        target_xy: Sequence[float],
        *,
        fixed_face_step_degrees: float,
        executed_magnitudes_degrees: Sequence[float],
        profile_index: int,
    ) -> int:
        """Move after a cumulative model-like FACE execution-drift prefix."""

        target, start, delta = self._validated_move_geometry(target_xy)
        if float(np.linalg.norm(delta)) <= 1e-9:
            return profile_index
        target_heading = _heading_between(start, target)
        profile_index = self.face_toward_with_execution_drift(
            target_heading,
            step_degrees=fixed_face_step_degrees,
            executed_magnitudes_degrees=executed_magnitudes_degrees,
            profile_index=profile_index,
        )
        self._move_after_facing(target=target, delta=delta)
        return profile_index

    def stop(self) -> None:
        self._record("STOP")
        self._append_success_history(
            action="STOP",
            completion_pose=self.pose.triple(),
            requested_waypoint_delta_robot_m=(0.0, 0.0),
            requested_heading_degrees=self.pose.yaw,
        )


def _boundary_recovery_geometry(
    current_xy: np.ndarray,
    *,
    collision_map: NumericCollisionMap,
    position_perturbation_m: float,
    rejected_proposal_distance_m: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Find a safe near-boundary state and a genuinely rejected outward target."""

    lower = collision_map.room_min_xy_m + collision_map.robot_radius_m
    upper = collision_map.room_max_xy_m - collision_map.robot_radius_m
    choices = (
        (float(current_xy[0] - lower[0]), np.asarray([-1.0, 0.0])),
        (float(upper[0] - current_xy[0]), np.asarray([1.0, 0.0])),
        (float(current_xy[1] - lower[1]), np.asarray([0.0, -1.0])),
        (float(upper[1] - current_xy[1]), np.asarray([0.0, 1.0])),
    )
    for _distance, direction in sorted(choices, key=lambda item: item[0]):
        for scale in (1.0, 0.75, 0.5, 0.25, 0.0):
            perturbed = current_xy + direction * position_perturbation_m * scale
            if (
                collision_map.point_check(perturbed).collision
                or collision_map.segment_check(current_xy, perturbed).collision
            ):
                continue
            rejected_target = perturbed + direction * rejected_proposal_distance_m
            if collision_map.segment_check(perturbed, rejected_target).collision:
                return perturbed, rejected_target
    return None


def _evenly_spaced_indices(length: int, limit: int) -> tuple[int, ...]:
    if length <= limit:
        return tuple(range(length))
    return tuple((index * length) // limit for index in range(limit))


def _replay_expert_goal_progress(
    rows: Sequence[Mapping[str, Any]],
    *,
    transition_count: int,
    history_encoding: _HistoryEncoding,
    room_size_m: Sequence[float],
    max_waypoint_step_m: float,
) -> WaypointGoalProgressLedger:
    """Replay untruncated numeric expert receipts through one decision prefix."""

    if not rows or not 0 <= transition_count <= len(rows):
        raise ValueError("Expert goal-progress replay range is invalid")
    initial_pose = tuple(float(value) for value in rows[0]["history_pose_xy_yaw"][-1])
    ledger = history_encoding.new_ledger(initial_pose)
    current = initial_pose
    for index, row in enumerate(rows[:transition_count]):
        observed = tuple(float(value) for value in row["history_pose_xy_yaw"][-1])
        if not _pose_close(observed, current):
            raise ValueError("Expert goal-progress replay pose is discontinuous")
        action = str(row["expert_action"])
        if action == "MOVE_TO":
            target = tuple(float(value) for value in row["expert_xy_m"])
            completion = (target[0], target[1], current[2])
            requested_delta = _world_target_to_robot_delta(current, target)
            requested_heading = current[2]
        elif action == "FACE":
            requested_heading = float(row["expert_heading_degrees"])
            completion = (current[0], current[1], requested_heading)
            requested_delta = (0.0, 0.0)
        elif action == "STOP":
            completion = current
            requested_delta = (0.0, 0.0)
            requested_heading = current[2]
        else:
            raise ValueError(f"Unsupported expert action in replay: {action!r}")
        history_encoding.encode_receipt(
            ledger=ledger,
            action=action,
            before_pose_xy_yaw=current,
            result_pose_xy_yaw=completion,
            requested_waypoint_delta_robot_m=requested_delta,
            requested_heading_degrees=requested_heading,
            room_size_m=room_size_m,
            max_waypoint_step_m=max_waypoint_step_m,
            success=True,
        )
        current = completion
        if index + 1 < transition_count:
            following = tuple(
                float(value) for value in rows[index + 1]["history_pose_xy_yaw"][-1]
            )
            if not _pose_close(following, current):
                raise ValueError("Expert goal-progress completion pose differs")
    return ledger


def _lap_recovery_rows(
    nominal_rows: Sequence[Mapping[str, Any]],
    *,
    collision_map: NumericCollisionMap,
    recovery_planner: NumericWaypointPlanner,
    room_size_m: Sequence[float],
    max_waypoint_step_m: float,
    fixed_face_step_degrees: float,
    history_length: int,
    history_encoding: _HistoryEncoding | None = None,
    augmentation: _LapRecoveryAugmentation,
) -> list[dict[str, Any]]:
    """Create complete deterministic DAgger episodes after unsafe proposals.

    The collision map and planner are used only to label these training rows.
    Every episode starts with the same ``success=0`` numeric history used by
    the deployed rejection loop, rejoins the nominal route, completes every
    remaining lap waypoint, and finally teaches Gemma to select STOP. No route
    or planner state is serialized into the rows or deployable checkpoint.
    """

    history_encoding = history_encoding or _v1_history_encoding(history_length)
    candidates: list[
        tuple[
            int,
            Mapping[str, Any],
            np.ndarray,
            np.ndarray,
            tuple[tuple[float, float], ...],
        ]
    ] = []
    for source_index, row in enumerate(nominal_rows):
        if row.get("expert_action") != "MOVE_TO":
            continue
        current_pose = np.asarray(row.get("history_pose_xy_yaw", [])[-1], dtype=np.float64)
        target = np.asarray(row.get("expert_xy_m"), dtype=np.float64)
        if current_pose.shape != (3,) or target.shape != (2,):
            raise RuntimeError("Lap recovery source row has invalid geometry")
        boundary = _boundary_recovery_geometry(
            current_pose[:2],
            collision_map=collision_map,
            position_perturbation_m=augmentation.position_perturbation_m,
            rejected_proposal_distance_m=augmentation.rejected_proposal_distance_m,
        )
        if boundary is None:
            continue
        perturbed_xy, rejected_target = boundary
        try:
            recovery_plan = recovery_planner.plan_to_free_point(perturbed_xy, target)
        except (RuntimeError, ValueError):
            continue
        if not recovery_plan.waypoints_xy_m:
            continue
        safe_target = np.asarray(recovery_plan.waypoints_xy_m[0], dtype=np.float64)
        if (
            float(np.linalg.norm(safe_target - perturbed_xy)) <= 1e-9
            or collision_map.segment_check(perturbed_xy, safe_target).collision
        ):
            continue
        candidates.append(
            (
                source_index,
                row,
                perturbed_xy,
                rejected_target,
                recovery_plan.waypoints_xy_m,
            )
        )
    if len(candidates) < augmentation.minimum_states_per_episode:
        raise RuntimeError("Lap recovery teacher found too few boundary states")

    selected = [
        candidates[index]
        for index in _evenly_spaced_indices(
            len(candidates), augmentation.max_states_per_episode
        )
    ]
    rows: list[dict[str, Any]] = []
    for candidate_index, (
        source_index,
        source,
        perturbed_xy,
        rejected_target,
        recovery_waypoints,
    ) in enumerate(selected):
        source_poses = [
            tuple(float(value) for value in pose)
            for pose in source["history_pose_xy_yaw"]
        ]
        action_codes = [int(value) for value in source["history_action_codes"]]
        numeric_history = [
            [float(value) for value in history_row]
            for history_row in source["history"]
        ]
        progress_ledger = _replay_expert_goal_progress(
            nominal_rows,
            transition_count=source_index,
            history_encoding=history_encoding,
            room_size_m=room_size_m,
            max_waypoint_step_m=max_waypoint_step_m,
        )
        nominal_pose = source_poses[-1]
        working_pose = nominal_pose
        perturb_delta = perturbed_xy - np.asarray(nominal_pose[:2], dtype=np.float64)
        if float(np.linalg.norm(perturb_delta)) > 1e-9:
            moved_pose = (
                float(perturbed_xy[0]),
                float(perturbed_xy[1]),
                float(nominal_pose[2]),
            )
            numeric_history.append(
                list(
                    history_encoding.encode_receipt(
                        ledger=progress_ledger,
                        action="move_to",
                        before_pose_xy_yaw=nominal_pose,
                        result_pose_xy_yaw=moved_pose,
                        requested_waypoint_delta_robot_m=_world_target_to_robot_delta(
                            nominal_pose, perturbed_xy
                        ),
                        requested_heading_degrees=float(nominal_pose[2]),
                        room_size_m=room_size_m,
                        max_waypoint_step_m=max_waypoint_step_m,
                        success=True,
                    )
                )
            )
            source_poses.append(moved_pose)
            action_codes.append(ACTION_TO_CODE["MOVE_TO"])
            working_pose = moved_pose

        yaw_offset = augmentation.yaw_offsets_degrees[
            candidate_index % len(augmentation.yaw_offsets_degrees)
        ]
        perturbed_yaw = _normalize_degrees(float(working_pose[2]) + yaw_offset)
        if abs(yaw_offset) > 1e-9:
            faced_pose = (
                float(working_pose[0]),
                float(working_pose[1]),
                perturbed_yaw,
            )
            numeric_history.append(
                list(
                    history_encoding.encode_receipt(
                        ledger=progress_ledger,
                        action="face",
                        before_pose_xy_yaw=working_pose,
                        result_pose_xy_yaw=faced_pose,
                        requested_waypoint_delta_robot_m=(0.0, 0.0),
                        requested_heading_degrees=perturbed_yaw,
                        room_size_m=room_size_m,
                        max_waypoint_step_m=max_waypoint_step_m,
                        success=True,
                    )
                )
            )
            source_poses.append(faced_pose)
            action_codes.append(ACTION_TO_CODE["FACE"])
            working_pose = faced_pose

        rejected_delta = _world_target_to_robot_delta(working_pose, rejected_target)
        rejected_heading = _heading_between(working_pose[:2], rejected_target)
        for streak in augmentation.rejection_streak_lengths:
            poses = list(source_poses)
            actions = list(action_codes)
            history = list(numeric_history)
            branch_ledger = _copy_goal_progress_ledger(progress_ledger)
            for _ in range(streak):
                poses.append(working_pose)
                actions.append(ACTION_TO_CODE["MOVE_TO"])
                history.append(
                    list(
                        history_encoding.encode_receipt(
                            ledger=branch_ledger,
                            action="move_to",
                            before_pose_xy_yaw=working_pose,
                            result_pose_xy_yaw=working_pose,
                            requested_waypoint_delta_robot_m=rejected_delta,
                            requested_heading_degrees=rejected_heading,
                            room_size_m=room_size_m,
                            max_waypoint_step_m=max_waypoint_step_m,
                            success=False,
                        )
                    )
                )
            recovery_pose = _Pose(
                float(working_pose[0]),
                float(working_pose[1]),
                float(working_pose[2]),
                last_delta=(float(perturb_delta[0]), float(perturb_delta[1]), 0.0),
            )
            episode_id = (
                f"{source['episode_id']}_dagger_{source_index:03d}_r{streak:02d}"
            )
            builder = _SyntheticTraceBuilder(
                episode_id=episode_id,
                scene_id=str(source["scene_id"]),
                split=str(source["split"]),
                family="lap_recovery",
                task_variant=f"{source['task_variant']}_dagger_recovery",
                instruction=str(source["instruction"]),
                start_xy=working_pose[:2],
                initial_yaw=float(working_pose[2]),
                episode_goal_xy_m=tuple(source["episode_goal_xy_m"]),
                training_target_xyz_m=None,
                collision_map=collision_map,
                room_size_m=room_size_m,
                max_waypoint_step_m=max_waypoint_step_m,
                max_turn_degrees=fixed_face_step_degrees,
                history_length=history_length,
                history_encoding=history_encoding,
                source_sample_sha256=_canonical_sha256(source),
            )
            builder.pose = recovery_pose
            builder.seed_training_history(
                poses=poses,
                action_codes=actions,
                numeric_rows=history,
                goal_progress_ledger=branch_ledger,
            )
            for waypoint in recovery_waypoints:
                builder.move_to(
                    waypoint,
                    fixed_face_step_degrees=fixed_face_step_degrees,
                )
            for remaining in nominal_rows[source_index + 1 :]:
                if remaining.get("expert_action") == "MOVE_TO":
                    builder.move_to(
                        remaining["expert_xy_m"],
                        fixed_face_step_degrees=fixed_face_step_degrees,
                    )
            builder.stop()
            if not builder.rows or builder.rows[-1]["expert_action"] != "STOP":
                raise RuntimeError("Lap recovery continuation has no model-labeled STOP")
            rows.extend(builder.rows)
    return rows


def _safe_starts(
    collision_map: NumericCollisionMap,
    *,
    count: int,
    seed: int,
    required_first: Sequence[float] | None = None,
) -> list[np.ndarray]:
    if count < 1:
        raise ValueError("start_pose_count must be positive")
    lower = collision_map.room_min_xy_m + collision_map.robot_radius_m + 0.20
    upper = collision_map.room_max_xy_m - collision_map.robot_radius_m - 0.20
    xs = np.linspace(lower[0], upper[0], 9)
    ys = np.linspace(lower[1], upper[1], 7)
    required = None
    if required_first is not None:
        required = np.asarray(required_first, dtype=np.float64)
        if required.shape != (2,) or not np.isfinite(required).all():
            raise ValueError("Required waypoint-training start must be finite XY")
        check = collision_map.point_check(required)
        if check.collision or check.clearance_m < 0.05:
            raise RuntimeError("Required waypoint-training start is not collision-free")
    candidates = ([] if required is None else [required])
    candidates.append(np.zeros(2, dtype=np.float64))
    candidates.extend(
        np.asarray([x, y], dtype=np.float64) for y in ys for x in xs
    )
    safe: list[np.ndarray] = []
    seen: set[tuple[float, float]] = set()
    for point in candidates:
        key = (round(float(point[0]), 8), round(float(point[1]), 8))
        check = collision_map.point_check(point)
        if key not in seen and not check.collision and check.clearance_m >= 0.05:
            safe.append(point)
            seen.add(key)
    if len(safe) < count:
        raise RuntimeError("Too few collision-free waypoint-training start poses")
    first: np.ndarray | None = None
    if required is not None:
        for index, point in enumerate(safe):
            if np.allclose(point, required, atol=1e-9):
                first = safe.pop(index)
                break
        if first is None:
            raise RuntimeError("Required waypoint-training start was lost")
    center: np.ndarray | None = None
    for index, point in enumerate(safe):
        if np.allclose(point, 0.0):
            center = safe.pop(index)
            break
    rng = np.random.default_rng(seed)
    rng.shuffle(safe)
    ordered = ([] if first is None else [first]) + ([] if center is None else [center]) + safe
    return ordered[:count]


def _signed_route_area(
    start_xy: Sequence[float], waypoints: Sequence[Sequence[float]]
) -> float:
    points = [np.asarray(start_xy, dtype=np.float64)]
    points.extend(np.asarray(point, dtype=np.float64) for point in waypoints)
    if len(points) < 4 or not np.allclose(points[0], points[-1], atol=1e-6):
        raise ValueError("Lap expert route must be closed")
    area = 0.0
    for first, second in pairwise(points):
        area += float(first[0] * second[1] - first[1] * second[0])
    return 0.5 * area


def _lap_routes(
    start: np.ndarray,
    planner: NumericPatrolPlanner,
) -> dict[str, list[tuple[float, float]]]:
    plan = planner.plan(start)
    forward = list(plan.waypoints_xy_m)
    route = [tuple(float(value) for value in start), *forward]
    reverse = list(reversed(route[:-1]))
    if abs(_signed_route_area(start, forward)) <= 0.5:
        raise RuntimeError("Lap expert route has insufficient room-scale winding")
    routes = {
        ("counterclockwise" if _signed_route_area(start, forward) > 0 else "clockwise"): forward,
        ("clockwise" if _signed_route_area(start, forward) > 0 else "counterclockwise"): reverse,
    }
    for waypoints in routes.values():
        if abs(_signed_route_area(start, waypoints)) <= 0.5:
            raise RuntimeError("Reversed lap expert route lost its winding")
        cursor = np.asarray(start, dtype=np.float64)
        for point in waypoints:
            target = np.asarray(point, dtype=np.float64)
            if planner.collision_map.segment_check(cursor, target).collision:
                raise RuntimeError("Lap expert route contains a colliding segment")
            cursor = target
    return routes


def _load_oracle_objects(
    path: Path, scene_id: str
) -> dict[str, tuple[float, float, float]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("scene_id") != scene_id:
        raise ValueError("Waypoint-training oracle scene identity differs")
    instances = value.get("instances")
    if not isinstance(instances, list):
        raise TypeError("Waypoint-training oracle has no instance inventory")
    objects: dict[str, tuple[float, float, float]] = {}
    for instance in instances:
        if not isinstance(instance, Mapping):
            continue
        category = instance.get("category")
        center = np.asarray(instance.get("expected_center_xyz_m"), dtype=np.float64)
        if (
            isinstance(category, str)
            and category not in _STRUCTURAL_CATEGORIES
            and category not in objects
            and center.shape == (3,)
            and np.isfinite(center).all()
        ):
            objects[category] = tuple(float(item) for item in center)
    if len(objects) < 2:
        raise RuntimeError("Between traces require at least two nonstructural objects")
    return objects


def _between_pairs(
    objects: Mapping[str, tuple[float, float, float]],
    *,
    count: int,
    seed: int,
) -> list[tuple[str, str]]:
    pairs = list(combinations(sorted(objects), 2))
    rng = np.random.default_rng(seed)
    rng.shuffle(pairs)
    if count < 1 or not pairs:
        raise ValueError("between_pairs_per_scene must be positive")
    return pairs[: min(count, len(pairs))]


def _object_targets(
    objects: Mapping[str, tuple[float, float, float]],
    *,
    count: int,
    seed: int,
) -> list[str]:
    names = sorted(objects)
    if count < 1 or not names:
        raise ValueError("object_targets_per_scene must be positive")
    rng = np.random.default_rng(seed)
    rng.shuffle(names)
    return names[: min(count, len(names))]


def _plan_between(
    *,
    start: np.ndarray,
    first_xyz: Sequence[float],
    second_xyz: Sequence[float],
    planner: NumericWaypointPlanner,
    search_resolution_m: float,
) -> tuple[np.ndarray, tuple[tuple[float, float], ...]]:
    midpoint = (
        np.asarray(first_xyz, dtype=np.float64)[:2]
        + np.asarray(second_xyz, dtype=np.float64)[:2]
    ) / 2.0
    candidates = [midpoint]
    for radius in np.arange(search_resolution_m, 0.80 + 1e-9, search_resolution_m):
        for angle_index in range(24):
            angle = 2.0 * math.pi * angle_index / 24.0
            candidates.append(
                midpoint
                + radius
                * np.asarray([math.cos(angle), math.sin(angle)], dtype=np.float64)
            )
    for candidate in candidates:
        if planner.collision_map.point_check(candidate).collision:
            continue
        try:
            plan = planner.plan_to_free_point(start, candidate)
        except (RuntimeError, ValueError):
            continue
        if plan.waypoints_xy_m:
            return np.asarray(plan.goal_xy_m, dtype=np.float64), plan.waypoints_xy_m
    raise RuntimeError("No collision-free expert path reaches a between-object goal")


@dataclass(frozen=True)
class _LapRecoveryAugmentation:
    enabled: bool
    full_post_recovery_continuation: bool
    nominal_initial_yaws_only: bool
    max_states_per_episode: int
    minimum_states_per_episode: int
    position_perturbation_m: float
    yaw_offsets_degrees: tuple[float, ...]
    rejected_proposal_distance_m: float
    rejection_streak_lengths: tuple[int, ...]


@dataclass(frozen=True)
class _LapExecutionDriftAugmentation:
    enabled: bool
    nominal_initial_yaws_only: bool
    post_face_rows_only: bool
    turn_magnitude_profiles_degrees: tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class _LiveFailureDaggerAugmentation:
    """Authenticated train-only relabeling of one preserved live failure."""

    enabled: bool
    report_path: Path | None
    report_sha256: str | None
    scene_id: str | None
    instruction: str | None
    initial_pose_xy_yaw: tuple[float, float, float] | None
    checkpoint_sha256: str | None
    scene_prefix_sha256: str | None
    first_rejection_step: int | None
    first_rejection_error_code: str | None
    failed_target_xy_m: tuple[float, float] | None
    failed_target_tolerance_m: float
    route_direction: str | None
    resume_waypoint_index: int | None
    maximum_total_decisions: int
    minimum_path_length_m: float
    minimum_abs_winding_area_m2: float
    maximum_return_error_m: float


@dataclass(frozen=True)
class _LivePreDivergenceDaggerAugmentation:
    """One authenticated correction immediately before a live class error."""

    source: _LiveFailureDaggerAugmentation
    branch_id: str
    correction_decision_step: int
    observed_model_action: str
    expected_pose_xy_yaw: tuple[float, float, float]
    expected_input_sha256: str
    resume_waypoint_index: int
    maximum_total_decisions: int
    expected_expert_first_action: str = "MOVE_TO"
    recovery_plan_to_resume: bool = False
    force_first_face: bool = False


@dataclass(frozen=True)
class _LiveObjectGoalDaggerAugmentation:
    """Authenticated train-only correction of one semantic object goal.

    The two reports are immutable evidence: the runtime report proves the
    deployed policy's continuous-input decisions, while the separately scored
    report binds the opaque target instance to training-only oracle geometry.
    Neither report nor any oracle field is available to the deployed policy.
    """

    enabled: bool
    runtime_report_path: Path | None
    runtime_report_sha256: str | None
    score_report_path: Path | None
    score_report_sha256: str | None
    scene_id: str | None
    goal_id: str | None
    instruction: str | None
    initial_pose_xy_yaw: tuple[float, float, float] | None
    checkpoint_sha256: str | None
    scene_prefix_sha256: str | None
    correction_decision_step: int | None
    observed_model_action: str | None
    observed_action_accepted: bool
    expected_pose_xy_yaw: tuple[float, float, float] | None
    expected_input_sha256: str | None
    expected_expert_first_action: str | None
    offline_expert_standoff_m: float | None
    stop_position_neighborhood_offsets_m: tuple[tuple[float, float], ...]
    target_category: str | None
    target_instance_id: str | None
    maximum_total_decisions: int
    minimum_center_progress_m: float
    maximum_bbox_standoff_m: float


@dataclass(frozen=True)
class _AuthenticatedLivePrefix:
    pose: _Pose
    poses: tuple[tuple[float, float, float], ...]
    action_codes: tuple[int, ...]
    numeric_history: tuple[tuple[float, ...], ...]
    trajectory_xy: tuple[tuple[float, float], ...]
    terminal_rejected_target_xy: tuple[float, float] | None
    goal_progress_ledger: WaypointGoalProgressLedger


def _lap_recovery_augmentation(
    settings: Mapping[str, Any],
    *,
    max_waypoint_step_m: float,
    max_turn_degrees: float,
    history_length: int,
) -> _LapRecoveryAugmentation:
    raw = settings.get("lap_recovery_augmentation", {"enabled": False})
    if not isinstance(raw, Mapping) or not isinstance(raw.get("enabled", False), bool):
        raise TypeError("lap_recovery_augmentation must be a mapping with boolean enabled")
    enabled = bool(raw.get("enabled", False))
    continuation = raw.get("full_post_recovery_continuation", False)
    nominal_yaws_only = raw.get("nominal_initial_yaws_only", False)
    if not isinstance(continuation, bool) or not isinstance(nominal_yaws_only, bool):
        raise TypeError("Lap recovery continuation flags must be boolean")
    max_states = int(raw.get("max_states_per_episode", 12))
    minimum_states = int(raw.get("minimum_states_per_episode", 1))
    position = float(raw.get("position_perturbation_m", 0.10))
    rejected_distance = float(raw.get("rejected_proposal_distance_m", 0.45))
    yaw_raw = raw.get("yaw_offsets_degrees", [-8.0, 8.0])
    default_streaks = list(
        dict.fromkeys((1, min(4, history_length), min(8, history_length), history_length))
    )
    streak_raw = raw.get("rejection_streak_lengths", default_streaks)
    if (
        not isinstance(yaw_raw, list)
        or not yaw_raw
        or not isinstance(streak_raw, list)
        or not streak_raw
    ):
        raise TypeError("Lap recovery yaw offsets and rejection streaks must be lists")
    yaw_offsets = tuple(float(value) for value in yaw_raw)
    streaks = tuple(int(value) for value in streak_raw)
    if (
        (enabled and not continuation)
        or
        max_states < 1
        or minimum_states < 1
        or minimum_states > max_states
        or not math.isfinite(position)
        or position < 0.0
        or position > max_waypoint_step_m
        or not math.isfinite(rejected_distance)
        or rejected_distance <= 0.0
        or rejected_distance > max_waypoint_step_m
        or any(not math.isfinite(value) or abs(value) > max_turn_degrees for value in yaw_offsets)
        or any(value < 1 or value > history_length for value in streaks)
        or len(set(streaks)) != len(streaks)
    ):
        raise ValueError("Lap recovery augmentation bounds or continuation are invalid")
    return _LapRecoveryAugmentation(
        enabled=enabled,
        full_post_recovery_continuation=continuation,
        nominal_initial_yaws_only=nominal_yaws_only,
        max_states_per_episode=max_states,
        minimum_states_per_episode=minimum_states,
        position_perturbation_m=position,
        yaw_offsets_degrees=yaw_offsets,
        rejected_proposal_distance_m=rejected_distance,
        rejection_streak_lengths=streaks,
    )


def _lap_execution_drift_augmentation(
    settings: Mapping[str, Any],
    *,
    fixed_face_step_degrees: float,
) -> _LapExecutionDriftAugmentation:
    raw = settings.get("lap_execution_drift_augmentation", {"enabled": False})
    if not isinstance(raw, Mapping) or not isinstance(raw.get("enabled", False), bool):
        raise TypeError(
            "lap_execution_drift_augmentation must have a boolean enabled flag"
        )
    enabled = bool(raw.get("enabled", False))
    nominal_yaws_only = raw.get("nominal_initial_yaws_only", True)
    post_face_only = raw.get("post_face_rows_only", True)
    profiles_raw = raw.get("turn_magnitude_profiles_degrees", [])
    if (
        not isinstance(nominal_yaws_only, bool)
        or not isinstance(post_face_only, bool)
        or not isinstance(profiles_raw, list)
        or any(not isinstance(profile, list) for profile in profiles_raw)
    ):
        raise TypeError("Lap execution-drift settings have invalid types")
    profiles = tuple(
        tuple(float(value) for value in profile) for profile in profiles_raw
    )
    if (
        (enabled and (not profiles or not post_face_only))
        or any(not profile for profile in profiles)
        or any(
            not math.isfinite(value)
            or value <= 0.0
            or value > fixed_face_step_degrees + 1e-7
            for profile in profiles
            for value in profile
        )
        or len(set(profiles)) != len(profiles)
    ):
        raise ValueError("Lap execution-drift profiles are invalid")
    return _LapExecutionDriftAugmentation(
        enabled=enabled,
        nominal_initial_yaws_only=nominal_yaws_only,
        post_face_rows_only=post_face_only,
        turn_magnitude_profiles_degrees=profiles,
    )


def _live_failure_dagger_augmentation(
    settings: Mapping[str, Any],
) -> _LiveFailureDaggerAugmentation:
    raw = settings.get("live_failure_dagger_augmentation", {"enabled": False})
    if not isinstance(raw, Mapping) or not isinstance(raw.get("enabled", False), bool):
        raise TypeError(
            "live_failure_dagger_augmentation must have a boolean enabled flag"
        )
    enabled = bool(raw.get("enabled", False))
    if not enabled:
        return _LiveFailureDaggerAugmentation(
            enabled=False,
            report_path=None,
            report_sha256=None,
            scene_id=None,
            instruction=None,
            initial_pose_xy_yaw=None,
            checkpoint_sha256=None,
            scene_prefix_sha256=None,
            first_rejection_step=None,
            first_rejection_error_code=None,
            failed_target_xy_m=None,
            failed_target_tolerance_m=0.0,
            route_direction=None,
            resume_waypoint_index=None,
            maximum_total_decisions=128,
            minimum_path_length_m=5.0,
            minimum_abs_winding_area_m2=0.5,
            maximum_return_error_m=0.35,
        )

    report_path = _rooted(str(raw.get("report_path", "")))
    current = Path(report_path.anchor)
    for component in report_path.parts[1:]:
        current /= component
        if current.is_symlink():
            raise ValueError("Live-failure DAgger report path cannot contain symlinks")
    digest = raw.get("report_sha256")
    checkpoint = raw.get("checkpoint_sha256")
    scene_prefix = raw.get("scene_prefix_sha256")
    scene_id = raw.get("scene_id")
    instruction = raw.get("instruction")
    initial = raw.get("initial_pose_xy_yaw")
    failed_target = raw.get("failed_target_xy_m")
    route_direction = raw.get("route_direction")
    error_code = raw.get("first_rejection_error_code")
    numeric = np.asarray(initial, dtype=np.float64)
    target = np.asarray(failed_target, dtype=np.float64)
    digest_pattern = re.compile(r"[0-9a-f]{64}")
    first_step = raw.get("first_rejection_step")
    resume_index = raw.get("resume_waypoint_index")
    maximum_steps = raw.get("maximum_total_decisions", 128)
    target_tolerance = float(raw.get("failed_target_tolerance_m", 0.01))
    minimum_path = float(raw.get("minimum_path_length_m", 5.0))
    minimum_area = float(raw.get("minimum_abs_winding_area_m2", 0.5))
    maximum_return = float(raw.get("maximum_return_error_m", 0.35))
    if (
        not report_path.is_file()
        or report_path.is_symlink()
        or "reports" not in {part.casefold() for part in report_path.parts}
        or not isinstance(digest, str)
        or digest_pattern.fullmatch(digest) is None
        or not isinstance(checkpoint, str)
        or digest_pattern.fullmatch(checkpoint) is None
        or not isinstance(scene_prefix, str)
        or digest_pattern.fullmatch(scene_prefix) is None
        or not isinstance(scene_id, str)
        or _SCENE_ID.fullmatch(scene_id) is None
        or not isinstance(instruction, str)
        or not instruction.strip()
        or instruction != instruction.strip()
        or numeric.shape != (3,)
        or not np.isfinite(numeric).all()
        or target.shape != (2,)
        or not np.isfinite(target).all()
        or route_direction not in {"clockwise", "counterclockwise"}
        or not isinstance(error_code, str)
        or not error_code.startswith("E_")
        or isinstance(first_step, bool)
        or not isinstance(first_step, int)
        or first_step < 1
        or isinstance(resume_index, bool)
        or not isinstance(resume_index, int)
        or resume_index < 0
        or isinstance(maximum_steps, bool)
        or not isinstance(maximum_steps, int)
        or not 1 <= maximum_steps <= 128
        or not math.isfinite(target_tolerance)
        or not 0.0 < target_tolerance <= 0.10
        or not math.isfinite(minimum_path)
        or minimum_path <= 0.0
        or not math.isfinite(minimum_area)
        or minimum_area <= 0.0
        or not math.isfinite(maximum_return)
        or maximum_return < 0.0
    ):
        raise ValueError("Live-failure DAgger augmentation contract is invalid")
    return _LiveFailureDaggerAugmentation(
        enabled=True,
        report_path=report_path,
        report_sha256=digest,
        scene_id=scene_id,
        instruction=instruction,
        initial_pose_xy_yaw=tuple(float(value) for value in numeric),
        checkpoint_sha256=checkpoint,
        scene_prefix_sha256=scene_prefix,
        first_rejection_step=first_step,
        first_rejection_error_code=error_code,
        failed_target_xy_m=tuple(float(value) for value in target),
        failed_target_tolerance_m=target_tolerance,
        route_direction=route_direction,
        resume_waypoint_index=resume_index,
        maximum_total_decisions=maximum_steps,
        minimum_path_length_m=minimum_path,
        minimum_abs_winding_area_m2=minimum_area,
        maximum_return_error_m=maximum_return,
    )


def _live_failure_dagger_augmentations(
    settings: Mapping[str, Any],
) -> tuple[_LiveFailureDaggerAugmentation, ...]:
    """Load an append-only sequence of authenticated live-failure sources.

    ``live_failure_dagger_augmentation`` remains the primary, backwards-
    compatible entry. Successor iterations may add sealed reports under
    ``additional_live_failure_dagger_augmentations`` without replacing prior
    DAgger supervision. A derived profile can append another generation under
    ``successor_live_failure_dagger_augmentations``; this separate key avoids
    YAML profile replacement dropping the inherited additional sources. Every
    entry uses the same fail-closed contract and duplicate reports are rejected.
    """

    primary = _live_failure_dagger_augmentation(settings)
    raw_additional = settings.get(
        "additional_live_failure_dagger_augmentations", []
    )
    raw_successor = settings.get(
        "successor_live_failure_dagger_augmentations", []
    )
    for field, values in (
        ("additional_live_failure_dagger_augmentations", raw_additional),
        ("successor_live_failure_dagger_augmentations", raw_successor),
    ):
        if not isinstance(values, list) or any(
            not isinstance(item, Mapping) for item in values
        ):
            raise TypeError(f"{field} must be a list of mappings")
    parsed: list[_LiveFailureDaggerAugmentation] = []
    if primary.enabled:
        parsed.append(primary)
    for raw in (*raw_additional, *raw_successor):
        nested_settings = dict(settings)
        nested_settings["live_failure_dagger_augmentation"] = dict(raw)
        augmentation = _live_failure_dagger_augmentation(nested_settings)
        if not augmentation.enabled:
            raise ValueError("Additional live-failure DAgger entries must be enabled")
        parsed.append(augmentation)
    report_digests = [item.report_sha256 for item in parsed]
    if len(set(report_digests)) != len(report_digests):
        raise ValueError("Live-failure DAgger report SHA-256 values must be unique")
    return tuple(parsed)


def _live_pre_divergence_dagger_augmentations(
    settings: Mapping[str, Any],
    live_sources: Sequence[_LiveFailureDaggerAugmentation],
) -> tuple[_LivePreDivergenceDaggerAugmentation, ...]:
    """Parse every train-only correction branch attached to each sealed source."""

    primary_raw = settings.get("live_failure_dagger_augmentation", {})
    additional_raw = settings.get("additional_live_failure_dagger_augmentations", [])
    successor_raw = settings.get("successor_live_failure_dagger_augmentations", [])
    raw_sources = [primary_raw, *additional_raw, *successor_raw]
    sources_by_digest = {str(source.report_sha256): source for source in live_sources}
    digest_pattern = re.compile(r"[0-9a-f]{64}")
    parsed: list[_LivePreDivergenceDaggerAugmentation] = []
    for raw_source in raw_sources:
        if not isinstance(raw_source, Mapping):
            continue
        singular = raw_source.get("pre_divergence_dagger_augmentation")
        plural = raw_source.get("pre_divergence_dagger_augmentations", [])
        if not isinstance(plural, list) or any(
            not isinstance(value, Mapping) for value in plural
        ):
            raise TypeError("pre_divergence_dagger_augmentations must be a list")
        raw_branches: list[Mapping[str, Any]] = []
        if singular is not None:
            if not isinstance(singular, Mapping):
                raise TypeError("pre_divergence_dagger_augmentation must be a mapping")
            raw_branches.append(singular)
        raw_branches.extend(plural)
        source = sources_by_digest.get(str(raw_source.get("report_sha256")))
        for raw in raw_branches:
            branch_id = raw.get("branch_id")
            correction_step = raw.get("correction_decision_step")
            observed_action = raw.get("observed_model_action")
            expected_pose = np.asarray(
                raw.get("expected_pose_xy_yaw"), dtype=np.float64
            )
            expected_input = raw.get("expected_input_sha256")
            resume_index = raw.get("resume_waypoint_index")
            maximum_steps = raw.get("maximum_total_decisions", 128)
            expected_expert = raw.get("expected_expert_first_action", "MOVE_TO")
            recovery_plan_to_resume = raw.get("recovery_plan_to_resume", False)
            force_first_face = raw.get("force_first_face", False)
            if (
                raw.get("enabled") is not True
                or source is None
                or not isinstance(branch_id, str)
                or re.fullmatch(r"[0-9a-z_]+", branch_id) is None
                or isinstance(correction_step, bool)
                or not isinstance(correction_step, int)
                or correction_step < 2
                or observed_action not in {"move_to", "face", "stop"}
                or expected_pose.shape != (3,)
                or not np.isfinite(expected_pose).all()
                or not isinstance(expected_input, str)
                or digest_pattern.fullmatch(expected_input) is None
                or isinstance(resume_index, bool)
                or not isinstance(resume_index, int)
                or resume_index < 0
                or isinstance(maximum_steps, bool)
                or not isinstance(maximum_steps, int)
                or not 1 <= maximum_steps <= 128
                or expected_expert not in {"MOVE_TO", "FACE"}
                or not isinstance(recovery_plan_to_resume, bool)
                or not isinstance(force_first_face, bool)
                or (
                    force_first_face
                    and (
                        expected_expert != "FACE"
                        or recovery_plan_to_resume
                    )
                )
            ):
                raise ValueError("Live divergence DAgger contract is invalid")
            parsed.append(
                _LivePreDivergenceDaggerAugmentation(
                    source=source,
                    branch_id=branch_id,
                    correction_decision_step=correction_step,
                    observed_model_action=observed_action,
                    expected_pose_xy_yaw=tuple(float(value) for value in expected_pose),
                    expected_input_sha256=expected_input,
                    resume_waypoint_index=resume_index,
                    maximum_total_decisions=maximum_steps,
                    expected_expert_first_action=expected_expert,
                    recovery_plan_to_resume=recovery_plan_to_resume,
                    force_first_face=force_first_face,
                )
            )
    identities = [
        (str(item.source.report_sha256), item.branch_id) for item in parsed
    ]
    if len(set(identities)) != len(identities):
        raise ValueError("Live pre-divergence DAgger branch identities must be unique")
    return tuple(parsed)


def _live_object_goal_dagger_augmentation(
    settings: Mapping[str, Any],
) -> _LiveObjectGoalDaggerAugmentation:
    """Parse a sealed object-goal correction without exposing it at runtime."""

    raw = settings.get("live_object_goal_dagger_augmentation", {"enabled": False})
    if not isinstance(raw, Mapping) or not isinstance(raw.get("enabled", False), bool):
        raise TypeError(
            "live_object_goal_dagger_augmentation must have a boolean enabled flag"
        )
    enabled = bool(raw.get("enabled", False))
    if not enabled:
        return _LiveObjectGoalDaggerAugmentation(
            enabled=False,
            runtime_report_path=None,
            runtime_report_sha256=None,
            score_report_path=None,
            score_report_sha256=None,
            scene_id=None,
            goal_id=None,
            instruction=None,
            initial_pose_xy_yaw=None,
            checkpoint_sha256=None,
            scene_prefix_sha256=None,
            correction_decision_step=None,
            observed_model_action=None,
            observed_action_accepted=True,
            expected_pose_xy_yaw=None,
            expected_input_sha256=None,
            expected_expert_first_action=None,
            offline_expert_standoff_m=None,
            stop_position_neighborhood_offsets_m=(),
            target_category=None,
            target_instance_id=None,
            maximum_total_decisions=64,
            minimum_center_progress_m=0.25,
            maximum_bbox_standoff_m=0.60,
        )

    runtime_path = _rooted(str(raw.get("runtime_report_path", "")))
    score_path = _rooted(str(raw.get("score_report_path", "")))
    for path in (runtime_path, score_path):
        current = Path(path.anchor)
        for component in path.parts[1:]:
            current /= component
            if current.is_symlink():
                raise ValueError("Live object-goal report path cannot contain symlinks")
    runtime_digest = raw.get("runtime_report_sha256")
    score_digest = raw.get("score_report_sha256")
    checkpoint = raw.get("checkpoint_sha256")
    scene_prefix = raw.get("scene_prefix_sha256")
    expected_input = raw.get("expected_input_sha256")
    digest_pattern = re.compile(r"[0-9a-f]{64}")
    scene_id = raw.get("scene_id")
    goal_id = raw.get("goal_id")
    instruction = raw.get("instruction")
    initial = np.asarray(raw.get("initial_pose_xy_yaw"), dtype=np.float64)
    expected_pose = np.asarray(raw.get("expected_pose_xy_yaw"), dtype=np.float64)
    correction_step = raw.get("correction_decision_step")
    observed_action = raw.get("observed_model_action")
    observed_accepted = raw.get("observed_action_accepted", True)
    expert_action = raw.get("expected_expert_first_action")
    expert_standoff = raw.get("offline_expert_standoff_m")
    neighborhood_raw = raw.get("stop_position_neighborhood_offsets_m", [])
    if not isinstance(neighborhood_raw, list) or any(
        not isinstance(value, list) for value in neighborhood_raw
    ):
        raise TypeError("STOP-neighborhood offsets must be a list of XY lists")
    neighborhood_arrays = tuple(
        np.asarray(value, dtype=np.float64) for value in neighborhood_raw
    )
    target_category = raw.get("target_category")
    target_instance_id = raw.get("target_instance_id")
    maximum_steps = raw.get("maximum_total_decisions", 64)
    minimum_progress = float(raw.get("minimum_center_progress_m", 0.25))
    maximum_standoff = float(raw.get("maximum_bbox_standoff_m", 0.60))
    digests = (
        runtime_digest,
        score_digest,
        checkpoint,
        scene_prefix,
        expected_input,
    )
    if (
        any(
            not isinstance(value, str) or digest_pattern.fullmatch(value) is None
            for value in digests
        )
        or any(
            not path.is_file()
            or path.is_symlink()
            or "reports" not in {part.casefold() for part in path.parts}
            for path in (runtime_path, score_path)
        )
        or scene_id is None
        or not isinstance(scene_id, str)
        or _SCENE_ID.fullmatch(scene_id) is None
        or not isinstance(goal_id, str)
        or re.fullmatch(r"[0-9a-z_]+", goal_id) is None
        or not isinstance(instruction, str)
        or not instruction.strip()
        or instruction != instruction.strip()
        or initial.shape != (3,)
        or not np.isfinite(initial).all()
        or expected_pose.shape != (3,)
        or not np.isfinite(expected_pose).all()
        or isinstance(correction_step, bool)
        or not isinstance(correction_step, int)
        or correction_step < 1
        or observed_action not in {"move_to", "face", "stop"}
        or not isinstance(observed_accepted, bool)
        or expert_action not in {"MOVE_TO", "FACE", "STOP"}
        or observed_action.casefold() == str(expert_action).casefold()
        or (
            expert_standoff is not None
            and (
                isinstance(expert_standoff, bool)
                or not isinstance(expert_standoff, (int, float))
                or not math.isfinite(float(expert_standoff))
                or not 0.0 < float(expert_standoff) <= 5.0
                or expert_action == "STOP"
            )
        )
        or any(
            offset.shape != (2,)
            or not np.isfinite(offset).all()
            or float(np.linalg.norm(offset)) <= 0.0
            or float(np.linalg.norm(offset)) > 0.05
            for offset in neighborhood_arrays
        )
        or len(neighborhood_arrays) > 8
        or len({tuple(float(value) for value in offset) for offset in neighborhood_arrays})
        != len(neighborhood_arrays)
        or (expert_action != "STOP" and bool(neighborhood_arrays))
        or (bool(neighborhood_arrays) and correction_step < 3)
        or not isinstance(target_category, str)
        or re.fullmatch(r"[a-z][a-z0-9 ]*", target_category) is None
        or not isinstance(target_instance_id, str)
        or re.fullmatch(r"i_[0-9]{6}", target_instance_id) is None
        or isinstance(maximum_steps, bool)
        or not isinstance(maximum_steps, int)
        or not 1 <= maximum_steps <= 128
        or not math.isfinite(minimum_progress)
        or minimum_progress <= 0.0
        or not math.isfinite(maximum_standoff)
        or maximum_standoff <= 0.0
    ):
        raise ValueError("Live object-goal DAgger augmentation contract is invalid")
    return _LiveObjectGoalDaggerAugmentation(
        enabled=True,
        runtime_report_path=runtime_path,
        runtime_report_sha256=str(runtime_digest),
        score_report_path=score_path,
        score_report_sha256=str(score_digest),
        scene_id=scene_id,
        goal_id=goal_id,
        instruction=instruction,
        initial_pose_xy_yaw=tuple(float(value) for value in initial),
        checkpoint_sha256=str(checkpoint),
        scene_prefix_sha256=str(scene_prefix),
        correction_decision_step=correction_step,
        observed_model_action=observed_action,
        observed_action_accepted=observed_accepted,
        expected_pose_xy_yaw=tuple(float(value) for value in expected_pose),
        expected_input_sha256=str(expected_input),
        expected_expert_first_action=expert_action,
        offline_expert_standoff_m=(
            None if expert_standoff is None else float(expert_standoff)
        ),
        stop_position_neighborhood_offsets_m=tuple(
            tuple(float(value) for value in offset) for offset in neighborhood_arrays
        ),
        target_category=target_category,
        target_instance_id=target_instance_id,
        maximum_total_decisions=maximum_steps,
        minimum_center_progress_m=minimum_progress,
        maximum_bbox_standoff_m=maximum_standoff,
    )


def _live_object_goal_dagger_augmentations(
    settings: Mapping[str, Any],
) -> tuple[_LiveObjectGoalDaggerAugmentation, ...]:
    """Return append-only object-goal sources with unique sealed runtimes."""

    primary = _live_object_goal_dagger_augmentation(settings)
    raw_additional = settings.get(
        "additional_live_object_goal_dagger_augmentations", []
    )
    if not isinstance(raw_additional, list) or any(
        not isinstance(item, Mapping) for item in raw_additional
    ):
        raise TypeError(
            "additional_live_object_goal_dagger_augmentations must be a list of mappings"
        )
    parsed: list[_LiveObjectGoalDaggerAugmentation] = []
    if primary.enabled:
        parsed.append(primary)
    for raw in raw_additional:
        nested = dict(settings)
        nested["live_object_goal_dagger_augmentation"] = dict(raw)
        augmentation = _live_object_goal_dagger_augmentation(nested)
        if not augmentation.enabled:
            raise ValueError(
                "Additional live object-goal DAgger entries must be enabled"
            )
        parsed.append(augmentation)
    digests = [item.runtime_report_sha256 for item in parsed]
    if len(set(digests)) != len(digests):
        raise ValueError("Live object-goal runtime report SHA-256 values must be unique")
    return tuple(parsed)


def _load_authenticated_live_failure_report(
    augmentation: _LiveFailureDaggerAugmentation,
) -> dict[str, Any]:
    if not augmentation.enabled or augmentation.report_path is None:
        raise ValueError("Live-failure report loading requires an enabled augmentation")
    if _sha256(augmentation.report_path) != augmentation.report_sha256:
        raise ValueError("Preserved live-failure report SHA-256 differs")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate live-failure report field: {key}")
            result[key] = value
        return result

    report = json.loads(
        augmentation.report_path.read_text(encoding="utf-8"),
        object_pairs_hook=unique_object,
    )
    snapshot = report.get("runtime_snapshot") if isinstance(report, dict) else None
    control = snapshot.get("control") if isinstance(snapshot, Mapping) else None
    state = snapshot.get("state") if isinstance(snapshot, Mapping) else None
    decisions = snapshot.get("model_decisions") if isinstance(snapshot, Mapping) else None
    prohibition_keys = (
        "deterministic_route_planner_used",
        "fallback_used",
        "substitution_applied",
        "synthetic_stop_applied",
    )
    top_level_prohibitions = (
        tuple(report.get(key) for key in prohibition_keys)
        if isinstance(report, Mapping)
        else ()
    )
    no_stop_verifier_failure = (
        top_level_prohibitions == (None, None, None, None)
        and report.get("model_decisions_authenticated") is False
        and report.get("decision_authentication_failure")
        == "AssertionError: Gemma did not select and execute the terminal STOP"
        and report.get("failure_type") == "AssertionError"
        and isinstance(decisions, list)
        and report.get("model_decision_count") == len(decisions)
        and report.get("executed_decision_count")
        == sum(
            1
            for decision in decisions
            if isinstance(decision, Mapping)
            and decision.get("accepted") is True
            and decision.get("executed") is True
        )
        and report.get("rejected_decision_count")
        == sum(
            1
            for decision in decisions
            if isinstance(decision, Mapping)
            and decision.get("accepted") is False
            and decision.get("executed") is False
        )
        and not any(
            isinstance(decision, Mapping)
            and decision.get("model_action") == "stop"
            for decision in decisions
        )
        and isinstance(control, Mapping)
        and all(control.get(key) is False for key in prohibition_keys)
        and all(
            isinstance(decision, Mapping)
            and decision.get("deterministic_route_planner_used") is False
            and decision.get("substitution_applied") is False
            and decision.get("synthetic_stop_applied") is False
            for decision in decisions
        )
    )
    top_level_provenance_is_safe = (
        top_level_prohibitions == (False, False, False, False)
        or no_stop_verifier_failure
    )
    if (
        report.get("schema") != "semantic_3d_chat.gemma_waypoint_live_acceptance.v2"
        or report.get("passed") is not False
        or report.get("local_inference") is not True
        or report.get("cloud_model_used") is not False
        or not top_level_provenance_is_safe
        or report.get("instruction") != augmentation.instruction
        or not isinstance(snapshot, Mapping)
        or not isinstance(control, Mapping)
        or control.get("navigation_checkpoint_sha256")
        != augmentation.checkpoint_sha256
        or control.get("gemma_attempted") is not True
        or control.get("local_inference") is not True
        or control.get("deterministic_route_planner_used") is not False
        or control.get("fallback_used") is not False
        or control.get("substitution_applied") is not False
        or control.get("synthetic_stop_applied") is not False
        or not isinstance(state, Mapping)
        or state.get("scene_id") != augmentation.scene_id
        or state.get("scene_prefix_hash") != augmentation.scene_prefix_sha256
        or snapshot.get("scene_prefix_hash") != augmentation.scene_prefix_sha256
        or not isinstance(decisions, list)
        or len(decisions) < int(augmentation.first_rejection_step or 0)
    ):
        raise ValueError("Preserved live-failure report provenance differs")
    return report


def _read_unique_json_object(path: Path, *, artifact: str) -> dict[str, Any]:
    """Read one sealed report while rejecting duplicate-key ambiguity."""

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate {artifact} field: {key}")
            result[key] = value
        return result

    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_object)
    if not isinstance(value, dict):
        raise TypeError(f"{artifact} must be a JSON object")
    return value


def _load_authenticated_live_object_goal_reports(
    augmentation: _LiveObjectGoalDaggerAugmentation,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Authenticate continuous-runtime evidence before consuming oracle score data."""

    if (
        not augmentation.enabled
        or augmentation.runtime_report_path is None
        or augmentation.score_report_path is None
    ):
        raise ValueError("Live object-goal report loading requires an enabled source")
    if _sha256(augmentation.runtime_report_path) != augmentation.runtime_report_sha256:
        raise ValueError("Preserved object-goal runtime report SHA-256 differs")
    runtime = _read_unique_json_object(
        augmentation.runtime_report_path, artifact="object-goal runtime report"
    )
    # This validator proves that the complete persisted episode came from the
    # local model-only public runtime and had no oracle/environmental fields.
    validated = validate_object_runtime_record(runtime)
    decisions = runtime.get("model_decisions")
    assert augmentation.correction_decision_step is not None
    if (
        runtime.get("schema") != OBJECT_RUNTIME_SCHEMA
        or validated["goal"].goal_id != augmentation.goal_id
        or runtime.get("instruction") != augmentation.instruction
        or runtime.get("scene_id") != augmentation.scene_id
        or validated.get("scene_prefix_sha256") != augmentation.scene_prefix_sha256
        or not isinstance(decisions, list)
        or len(decisions) < augmentation.correction_decision_step
    ):
        raise ValueError("Preserved object-goal runtime provenance differs")
    digest_pattern = re.compile(r"[0-9a-f]{64}")
    for decision in decisions:
        if (
            not isinstance(decision, Mapping)
            or decision.get("checkpoint_sha256") != augmentation.checkpoint_sha256
            or decision.get("scene_prefix_sha256") != augmentation.scene_prefix_sha256
            or any(
                not isinstance(decision.get(field), str)
                or digest_pattern.fullmatch(str(decision.get(field))) is None
                for field in (
                    "active_prefix_sha256",
                    "decision_tensor_sha256",
                    "robot_tokens_sha256",
                )
            )
        ):
            raise ValueError("Preserved object-goal decision binding differs")
    startup = runtime.get("startup_state")
    assert augmentation.initial_pose_xy_yaw is not None
    if (
        not isinstance(startup, Mapping)
        or not np.allclose(
            np.asarray(
                [
                    *startup.get("position_xy_m", []),
                    startup.get("body_yaw_degrees"),
                ],
                dtype=np.float64,
            ),
            np.asarray(augmentation.initial_pose_xy_yaw, dtype=np.float64),
            atol=1e-9,
        )
    ):
        raise ValueError("Preserved object-goal initial pose differs")

    # Only after all runtime provenance is valid may the separately generated
    # evaluation score (which contains oracle geometry) be opened.
    if _sha256(augmentation.score_report_path) != augmentation.score_report_sha256:
        raise ValueError("Preserved object-goal score report SHA-256 differs")
    score = _read_unique_json_object(
        augmentation.score_report_path, artifact="object-goal score report"
    )
    goals = score.get("goals")
    matches = (
        [
            goal
            for goal in goals
            if isinstance(goal, Mapping) and goal.get("goal_id") == augmentation.goal_id
        ]
        if isinstance(goals, list)
        else []
    )
    if (
        score.get("schema") != OBJECT_SCORE_SCHEMA
        or score.get("status") != "evaluation_only_oracle_score"
        or score.get("all_runtime_evidence_validated_before_oracle_open") is not True
        or score.get("runtime_capture_process_accepts_oracle_arguments") is not False
        or score.get("runtime_process_read_oracle") is not False
        or score.get("scorer_reads_oracle") is not True
        or score.get("scene_id") != augmentation.scene_id
        or len(matches) != 1
    ):
        raise ValueError("Preserved object-goal score provenance differs")
    goal = dict(matches[0])
    checks = goal.get("checks")
    target_center = np.asarray(goal.get("target_center_xyz_m"), dtype=np.float64)
    required_checks = {
        "accepted_gemma_stop",
        "minimum_oracle_center_progress",
        "maximum_oracle_bbox_standoff",
    }
    progress = goal.get("target_center_progress_m")
    standoff = goal.get("final_oracle_bbox_standoff_m")
    minimum_progress = goal.get("minimum_center_progress_m")
    maximum_standoff = goal.get("maximum_bbox_standoff_m")
    numeric_metrics = (progress, standoff, minimum_progress, maximum_standoff)
    metrics_valid = all(
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        for value in numeric_metrics
    )
    expected_progress_check = (
        metrics_valid and float(progress) >= float(minimum_progress)
    )
    expected_standoff_check = (
        metrics_valid and float(standoff) <= float(maximum_standoff)
    )
    checks_valid = (
        isinstance(checks, Mapping)
        and set(checks) == required_checks
        and all(isinstance(checks[name], bool) for name in required_checks)
        and checks["accepted_gemma_stop"] is True
        and checks["minimum_oracle_center_progress"] is expected_progress_check
        and checks["maximum_oracle_bbox_standoff"] is expected_standoff_check
        and not (expected_progress_check and expected_standoff_check)
    )
    score_rows_valid = (
        isinstance(goals, list)
        and score.get("goal_count") == len(goals)
        and score.get("passed_count")
        == sum(item.get("passed") is True for item in goals if isinstance(item, Mapping))
        and score.get("all_passed")
        is all(
            isinstance(item, Mapping) and item.get("passed") is True for item in goals
        )
    )
    if (
        goal.get("passed") is not False
        or goal.get("instruction") != augmentation.instruction
        or goal.get("runtime_file_sha256") != augmentation.runtime_report_sha256
        or goal.get("scene_prefix_sha256") != augmentation.scene_prefix_sha256
        or goal.get("target_category") != augmentation.target_category
        or goal.get("target_instance_id") != augmentation.target_instance_id
        or goal.get("metric") != "approach_standoff"
        or target_center.shape != (3,)
        or not np.isfinite(target_center).all()
        or not checks_valid
        or not score_rows_valid
        or not metrics_valid
        or float(minimum_progress) != augmentation.minimum_center_progress_m
        or float(maximum_standoff) != augmentation.maximum_bbox_standoff_m
        or float(standoff) < 0.0
        or goal.get("model_selected_terminal_stop") is not True
        or goal.get("model_decision_count") != runtime.get("model_decision_count")
        or goal.get("accepted_decision_count")
        != runtime.get("accepted_decision_count")
        or goal.get("rejected_decision_count")
        != runtime.get("rejected_decision_count")
    ):
        raise ValueError("Preserved object-goal failed score differs")
    return runtime, score, goal


def _load_training_oracle_object_geometry(
    oracle_path: Path,
    *,
    scene_id: str,
    category: str,
    instance_id: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return one exact target center and XY box from training-only metadata."""

    oracle = _read_unique_json_object(oracle_path, artifact="training oracle")
    instances = oracle.get("instances")
    matches = (
        [
            item
            for item in instances
            if isinstance(item, Mapping)
            and item.get("category") == category
            and item.get("instance_id") == instance_id
        ]
        if isinstance(instances, list)
        else []
    )
    if oracle.get("scene_id") != scene_id or len(matches) != 1:
        raise ValueError("Training oracle object-goal target differs")
    target = matches[0]
    center = np.asarray(target.get("expected_center_xyz_m"), dtype=np.float64)
    bbox = target.get("bbox")
    minimum = np.asarray(
        bbox.get("min_xyz_m") if isinstance(bbox, Mapping) else None,
        dtype=np.float64,
    )
    maximum = np.asarray(
        bbox.get("max_xyz_m") if isinstance(bbox, Mapping) else None,
        dtype=np.float64,
    )
    if (
        center.shape != (3,)
        or minimum.shape != (3,)
        or maximum.shape != (3,)
        or not np.isfinite(center).all()
        or not np.isfinite(minimum).all()
        or not np.isfinite(maximum).all()
        or np.any(maximum <= minimum)
    ):
        raise ValueError("Training oracle object-goal geometry is invalid")
    return center, minimum[:2], maximum[:2]


def _lap_execution_drift_rows(
    *,
    episode_id: str,
    scene_id: str,
    split: str,
    direction: str,
    instruction: str,
    start_xy: Sequence[float],
    initial_yaw: float,
    route_waypoints: Sequence[Sequence[float]],
    collision_map: NumericCollisionMap,
    room_size_m: Sequence[float],
    max_waypoint_step_m: float,
    fixed_face_step_degrees: float,
    history_length: int,
    history_encoding: _HistoryEncoding | None = None,
    augmentation: _LapExecutionDriftAugmentation,
) -> tuple[list[dict[str, Any]], int]:
    """Create train-only post-FACE rows from cumulative behavior drift.

    Each hidden-history prefix is generated by a complete numeric rollout. The
    behavior executes the configured under-turn magnitudes, while every row is
    labeled with the unchanged bounded expert action and its robot-frame target.
    Only rows immediately following a FACE transition are retained, keeping the
    cache practical while directly covering every observed action boundary.
    """

    history_encoding = history_encoding or _v1_history_encoding(history_length)
    rows: list[dict[str, Any]] = []
    episode_count = 0
    for profile_index, profile in enumerate(
        augmentation.turn_magnitude_profiles_degrees
    ):
        drift_episode_id = f"{episode_id}_execution_drift_{profile_index:02d}"
        builder = _SyntheticTraceBuilder(
            episode_id=drift_episode_id,
            scene_id=scene_id,
            split=split,
            family="lap_execution_drift",
            task_variant=f"lap_{direction}_execution_drift",
            instruction=instruction,
            start_xy=start_xy,
            initial_yaw=initial_yaw,
            episode_goal_xy_m=(float(start_xy[0]), float(start_xy[1])),
            training_target_xyz_m=None,
            collision_map=collision_map,
            room_size_m=room_size_m,
            max_waypoint_step_m=max_waypoint_step_m,
            max_turn_degrees=fixed_face_step_degrees,
            history_length=history_length,
            history_encoding=history_encoding,
        )
        turn_index = 0
        for waypoint in route_waypoints:
            turn_index = builder.move_to_with_execution_drift(
                waypoint,
                fixed_face_step_degrees=fixed_face_step_degrees,
                executed_magnitudes_degrees=profile,
                profile_index=turn_index,
            )
        builder.stop()
        selected = [
            row
            for row in builder.rows
            if row["history_action_codes"]
            and int(row["history_action_codes"][-1]) == ACTION_TO_CODE["FACE"]
        ]
        if not selected:
            raise RuntimeError("Lap execution-drift rollout has no post-FACE rows")
        for selected_index, row in enumerate(selected):
            row["step_index"] = selected_index
        rows.extend(selected)
        episode_count += 1
    return rows, episode_count


def _replay_authenticated_live_prefix(
    *,
    augmentation: _LiveFailureDaggerAugmentation,
    decisions: Sequence[Mapping[str, Any]],
    transition_count: int,
    terminal_rejection_error_code: str | None,
    room_size_m: Sequence[float],
    max_waypoint_step_m: float,
    fixed_face_step_degrees: float,
    history_length: int = 16,
    history_encoding: _HistoryEncoding | None = None,
) -> _AuthenticatedLivePrefix:
    """Replay a sealed Gemma prefix into exact numeric state and history."""

    history_encoding = history_encoding or _v1_history_encoding(history_length)
    if augmentation.initial_pose_xy_yaw is None:
        raise ValueError("Authenticated live prefix has no initial pose")
    initial = augmentation.initial_pose_xy_yaw
    pose = _Pose(float(initial[0]), float(initial[1]), float(initial[2]))
    poses: list[tuple[float, float, float]] = [pose.triple()]
    actions: list[int] = []
    numeric_history: list[tuple[float, ...]] = []
    trajectory = [tuple(float(value) for value in initial[:2])]
    rejected_target: tuple[float, float] | None = None
    progress_ledger = history_encoding.new_ledger(initial)
    if transition_count < 1 or len(decisions) < transition_count:
        raise ValueError("Authenticated live prefix length is invalid")
    for decision_index, decision in enumerate(decisions[:transition_count]):
        if not isinstance(decision, Mapping):
            raise TypeError("Live-failure decision must be a mapping")
        action = decision.get("model_action")
        accepted = decision.get("accepted")
        executed = decision.get("executed")
        step = decision_index + 1
        terminal_rejection = (
            terminal_rejection_error_code is not None
            and step == transition_count
        )
        if (
            decision.get("step") != step
            or decision.get("actual_gemma_causal_forward") is not True
            or decision.get("checkpoint_sha256") != augmentation.checkpoint_sha256
            or decision.get("scene_prefix_sha256")
            != augmentation.scene_prefix_sha256
            or decision.get("deterministic_route_planner_used") is not False
            or decision.get("substitution_applied") is not False
            or decision.get("synthetic_stop_applied") is not False
            or action not in {"move_to", "face", "stop"}
            or not isinstance(accepted, bool)
            or not isinstance(executed, bool)
            or accepted != executed
            or (terminal_rejection and accepted)
        ):
            raise ValueError("Live-failure decision provenance or ordering differs")
        turn_delta = decision.get("model_turn_delta_degrees")
        desired_heading = decision.get("model_desired_heading_degrees")
        waypoint_delta = np.asarray(
            decision.get("model_waypoint_delta_robot_m"), dtype=np.float64
        )
        if (
            isinstance(turn_delta, bool)
            or not isinstance(turn_delta, (int, float))
            or not math.isfinite(float(turn_delta))
            or abs(float(turn_delta)) > fixed_face_step_degrees + 1e-6
            or isinstance(desired_heading, bool)
            or not isinstance(desired_heading, (int, float))
            or not math.isfinite(float(desired_heading))
            or waypoint_delta.shape != (2,)
            or not np.isfinite(waypoint_delta).all()
            or np.any(np.abs(waypoint_delta) > max_waypoint_step_m + 1e-6)
            or abs(
                _normalize_degrees(
                    pose.yaw + float(turn_delta) - float(desired_heading)
                )
            )
            > 2e-4
        ):
            raise ValueError("Live-failure decision numeric heads differ")

        before = pose.triple()
        result = before
        if action == "face":
            if decision.get("derived_world_waypoint_xy_m") is not None:
                raise ValueError("Live FACE unexpectedly carries a world waypoint")
            if accepted:
                pose = _Pose(
                    pose.x,
                    pose.y,
                    float(desired_heading),
                    last_delta=pose.last_delta,
                    linear_velocity=(0.0, 0.0),
                    angular_velocity=float(turn_delta),
                )
                result = pose.triple()
        elif action == "move_to":
            world_target = np.asarray(
                decision.get("derived_world_waypoint_xy_m"), dtype=np.float64
            )
            expected_world = np.asarray(before[:2], dtype=np.float64)
            radians = math.radians(float(before[2]))
            expected_world += (
                waypoint_delta[0]
                * np.asarray([math.cos(radians), math.sin(radians)])
                + waypoint_delta[1]
                * np.asarray([-math.sin(radians), math.cos(radians)])
            )
            if (
                world_target.shape != (2,)
                or not np.isfinite(world_target).all()
                or not np.allclose(world_target, expected_world, atol=2e-6)
            ):
                raise ValueError("Live MOVE world/robot target binding differs")
            if accepted:
                movement = world_target - np.asarray(before[:2], dtype=np.float64)
                pose = _Pose(
                    float(world_target[0]),
                    float(world_target[1]),
                    float(before[2]),
                    last_delta=(float(movement[0]), float(movement[1]), 0.0),
                    linear_velocity=(float(movement[0]), float(movement[1])),
                    angular_velocity=0.0,
                )
                result = pose.triple()
                trajectory.append(tuple(float(value) for value in world_target))
            elif terminal_rejection:
                rejected_target = tuple(float(value) for value in world_target)
        elif accepted:
            raise ValueError("The preserved live prefix cannot stop before correction")

        numeric_history.append(
            tuple(
                history_encoding.encode_receipt(
                    ledger=progress_ledger,
                    action=str(action),
                    before_pose_xy_yaw=before,
                    result_pose_xy_yaw=result if accepted else before,
                    requested_waypoint_delta_robot_m=waypoint_delta,
                    requested_heading_degrees=float(desired_heading),
                    room_size_m=room_size_m,
                    max_waypoint_step_m=max_waypoint_step_m,
                    success=accepted,
                )
            )
        )
        poses.append(result if accepted else before)
        actions.append(ACTION_TO_CODE[str(action).upper()])
        if not accepted and (
            not isinstance(decision.get("error_code"), str)
            or not str(decision.get("error_code")).startswith("E_")
        ):
            raise ValueError("Rejected live prefix action has no opaque error code")
        if terminal_rejection and (
            action != "move_to"
            or decision.get("error_code") != terminal_rejection_error_code
        ):
            raise ValueError("Terminal live rejection differs from its contract")

    if (terminal_rejection_error_code is None) != (rejected_target is None):
        raise RuntimeError("Authenticated live prefix rejection state differs")
    return _AuthenticatedLivePrefix(
        pose=pose,
        poses=tuple(poses),
        action_codes=tuple(actions),
        numeric_history=tuple(numeric_history),
        trajectory_xy=tuple(trajectory),
        terminal_rejected_target_xy=rejected_target,
        goal_progress_ledger=progress_ledger,
    )


def _validate_live_history_window(
    history: object,
    *,
    exact_prefix_transition_count: int,
    history_length: int,
    expected_terminal_success: bool,
    history_encoding: _HistoryEncoding | None = None,
) -> None:
    """Require the runtime's exact growing-then-rolling history window."""

    history_encoding = history_encoding or _v1_history_encoding(history_length)
    if exact_prefix_transition_count < 1 or history_length < 1:
        raise ValueError("Live-failure history bounds must be positive")
    expected_length = min(exact_prefix_transition_count, history_length)
    if (
        not isinstance(history, list)
        or len(history) != expected_length
        or not history
        or not isinstance(history[-1], list)
        or len(history[-1]) != history_encoding.feature_dim
        or history[-1][11] != float(expected_terminal_success)
    ):
        raise RuntimeError(
            "Live correction history is not an exact transition window"
        )


def _validate_live_failure_history_window(
    history: object,
    *,
    exact_prefix_transition_count: int,
    history_length: int,
    history_encoding: _HistoryEncoding | None = None,
) -> None:
    _validate_live_history_window(
        history,
        exact_prefix_transition_count=exact_prefix_transition_count,
        history_length=history_length,
        expected_terminal_success=False,
        history_encoding=history_encoding,
    )


def _live_failure_dagger_rows(
    *,
    augmentation: _LiveFailureDaggerAugmentation,
    route_waypoints: Sequence[Sequence[float]],
    collision_map: NumericCollisionMap,
    recovery_planner: NumericWaypointPlanner,
    room_size_m: Sequence[float],
    max_waypoint_step_m: float,
    fixed_face_step_degrees: float,
    history_length: int,
    history_encoding: _HistoryEncoding | None = None,
    episode_discriminator: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Relabel the exact first rejected live state and finish the lap safely.

    All decisions before and including the first rejection are replayed only
    into numeric pose/history. The next row is labeled by the offline expert;
    the deployed controller never imports this parser, collision map, or route
    planner. The continuation reaches the configured padded route, processes
    every remaining waypoint, returns to the original live start, and includes
    an explicit model-supervised STOP row.
    """

    history_encoding = history_encoding or _v1_history_encoding(history_length)
    report = _load_authenticated_live_failure_report(augmentation)
    snapshot = report["runtime_snapshot"]
    decisions = snapshot["model_decisions"]
    assert augmentation.initial_pose_xy_yaw is not None
    assert augmentation.first_rejection_step is not None
    assert augmentation.first_rejection_error_code is not None
    assert augmentation.failed_target_xy_m is not None
    assert augmentation.resume_waypoint_index is not None
    assert augmentation.route_direction is not None
    initial = augmentation.initial_pose_xy_yaw
    reported_initial = np.asarray(report.get("initial_position_xy_m"), dtype=np.float64)
    if (
        reported_initial.shape != (2,)
        or not np.isfinite(reported_initial).all()
        or not np.allclose(reported_initial, np.asarray(initial[:2]), atol=1e-9)
    ):
        raise ValueError("Live-failure report initial position differs")

    replay = _replay_authenticated_live_prefix(
        augmentation=augmentation,
        decisions=decisions,
        transition_count=augmentation.first_rejection_step,
        terminal_rejection_error_code=augmentation.first_rejection_error_code,
        room_size_m=room_size_m,
        max_waypoint_step_m=max_waypoint_step_m,
        fixed_face_step_degrees=fixed_face_step_degrees,
        history_length=history_length,
        history_encoding=history_encoding,
    )
    pose = replay.pose
    poses = list(replay.poses)
    actions = list(replay.action_codes)
    numeric_history = [list(value) for value in replay.numeric_history]
    trajectory = [np.asarray(value, dtype=np.float64) for value in replay.trajectory_xy]
    rejected_target = np.asarray(
        replay.terminal_rejected_target_xy, dtype=np.float64
    )
    if not np.allclose(
        rejected_target,
        np.asarray(augmentation.failed_target_xy_m),
        atol=augmentation.failed_target_tolerance_m,
    ):
        raise ValueError("First rejected live waypoint differs from its sealed target")
    if not collision_map.segment_check(np.asarray(pose.triple()[:2]), rejected_target).collision:
        raise RuntimeError("Padded offline collision map does not reject the live failure")
    if augmentation.resume_waypoint_index >= len(route_waypoints) - 1:
        raise ValueError("Live-failure resume waypoint cannot finish the configured route")
    resume_target = np.asarray(
        route_waypoints[augmentation.resume_waypoint_index], dtype=np.float64
    )
    recovery_plan = recovery_planner.plan_to_free_point(
        np.asarray(pose.triple()[:2], dtype=np.float64), resume_target
    )
    if not recovery_plan.waypoints_xy_m:
        raise RuntimeError("Live-failure offline expert produced no recovery waypoints")

    if episode_discriminator is not None and re.fullmatch(
        r"[0-9a-z_]+", episode_discriminator
    ) is None:
        raise ValueError("Live-failure episode discriminator is invalid")
    episode_suffix = (
        f"_{episode_discriminator}" if episode_discriminator is not None else ""
    )
    builder = _SyntheticTraceBuilder(
        episode_id=(
            f"live_failure_{augmentation.scene_id}_"
            f"step_{augmentation.first_rejection_step:03d}{episode_suffix}"
        ),
        scene_id=str(augmentation.scene_id),
        split="train",
        family="lap_live_failure_recovery",
        task_variant=f"lap_{augmentation.route_direction}_live_failure_recovery",
        instruction=str(augmentation.instruction),
        start_xy=pose.triple()[:2],
        initial_yaw=pose.yaw,
        episode_goal_xy_m=(float(initial[0]), float(initial[1])),
        training_target_xyz_m=None,
        collision_map=collision_map,
        room_size_m=room_size_m,
        max_waypoint_step_m=max_waypoint_step_m,
        max_turn_degrees=fixed_face_step_degrees,
        history_length=history_length,
        history_encoding=history_encoding,
        source_sample_sha256=str(augmentation.report_sha256),
    )
    builder.pose = pose
    builder.seed_training_history(
        poses=poses,
        action_codes=actions,
        numeric_rows=numeric_history,
        goal_progress_ledger=replay.goal_progress_ledger,
    )
    continuation_waypoints = [
        *recovery_plan.waypoints_xy_m,
        *route_waypoints[augmentation.resume_waypoint_index + 1 :],
    ]
    minimum_clearance = math.inf
    cursor = np.asarray(pose.triple()[:2], dtype=np.float64)
    for waypoint in continuation_waypoints:
        target = np.asarray(waypoint, dtype=np.float64)
        check = collision_map.segment_check(cursor, target)
        if check.collision:
            raise RuntimeError("Live-failure expert continuation contains a collision")
        minimum_clearance = min(minimum_clearance, check.clearance_m)
        builder.move_to(
            waypoint,
            fixed_face_step_degrees=fixed_face_step_degrees,
        )
        trajectory.append(target.copy())
        cursor = target
    builder.stop()
    if (
        not builder.rows
        or builder.rows[-1]["expert_action"] != "STOP"
        or builder.rows[-1]["history"][-1][11] != 1.0
    ):
        raise RuntimeError("Live-failure continuation lacks a model-labeled STOP")
    total_decisions = augmentation.first_rejection_step + len(builder.rows)
    if total_decisions > augmentation.maximum_total_decisions:
        raise RuntimeError("Live-failure expert continuation exceeds the runtime budget")

    path_length = sum(
        float(np.linalg.norm(second - first))
        for first, second in pairwise(trajectory)
    )
    signed_area = 0.5 * sum(
        float(first[0] * second[1] - first[1] * second[0])
        for first, second in pairwise(trajectory)
    )
    return_error = float(np.linalg.norm(trajectory[-1] - trajectory[0]))
    expected_sign = -1.0 if augmentation.route_direction == "clockwise" else 1.0
    if (
        path_length < augmentation.minimum_path_length_m
        or abs(signed_area) < augmentation.minimum_abs_winding_area_m2
        or signed_area * expected_sign <= 0.0
        or return_error > augmentation.maximum_return_error_m
    ):
        raise RuntimeError("Live-failure expert continuation does not form a valid lap")
    first = builder.rows[0]
    _validate_live_failure_history_window(
        first["history"],
        exact_prefix_transition_count=len(numeric_history),
        history_length=history_length,
        history_encoding=history_encoding,
    )
    if (
        not np.allclose(
            np.asarray(first["state_features"], dtype=np.float64),
            np.asarray(_numeric_state(pose, room_size_m), dtype=np.float64),
            atol=0.0,
        )
    ):
        raise RuntimeError("Live-failure recovery row is not exact runtime-aligned state")
    metrics = {
        "source_report_sha256": augmentation.report_sha256,
        "source_checkpoint_sha256": augmentation.checkpoint_sha256,
        "source_scene_prefix_sha256": augmentation.scene_prefix_sha256,
        "first_rejection_step": augmentation.first_rejection_step,
        "first_rejection_error_code": augmentation.first_rejection_error_code,
        "first_rejected_target_xy_m": [float(value) for value in rejected_target],
        "exact_prefix_transition_count": len(numeric_history),
        "exact_history_parameterization": history_encoding.parameterization,
        "first_recovery_input_sha256": _canonical_sha256(
            {
                "state_features": first["state_features"],
                "history": first["history"],
            }
        ),
        "continuation_sample_count": len(builder.rows),
        "total_decision_count": total_decisions,
        "path_length_m": path_length,
        "signed_winding_area_m2": signed_area,
        "abs_winding_area_m2": abs(signed_area),
        "return_error_m": return_error,
        "minimum_padded_map_clearance_m": minimum_clearance,
        "model_labeled_stop": True,
        "offline_planner_used_for_labels_only": True,
        "runtime_planner_available": False,
    }
    return builder.rows, metrics


def _live_pre_divergence_dagger_rows(
    *,
    augmentation: _LivePreDivergenceDaggerAugmentation,
    route_waypoints: Sequence[Sequence[float]],
    collision_map: NumericCollisionMap,
    recovery_planner: NumericWaypointPlanner | None = None,
    room_size_m: Sequence[float],
    max_waypoint_step_m: float,
    fixed_face_step_degrees: float,
    history_length: int,
    history_encoding: _HistoryEncoding | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Correct an exact accepted live state before its first class divergence."""

    history_encoding = history_encoding or _v1_history_encoding(history_length)
    source = augmentation.source
    report = _load_authenticated_live_failure_report(source)
    snapshot = report["runtime_snapshot"]
    decisions = snapshot["model_decisions"]
    prefix_count = augmentation.correction_decision_step - 1
    replay = _replay_authenticated_live_prefix(
        augmentation=source,
        decisions=decisions,
        transition_count=prefix_count,
        terminal_rejection_error_code=None,
        room_size_m=room_size_m,
        max_waypoint_step_m=max_waypoint_step_m,
        fixed_face_step_degrees=fixed_face_step_degrees,
        history_length=history_length,
        history_encoding=history_encoding,
    )
    if not np.allclose(
        np.asarray(replay.pose.triple(), dtype=np.float64),
        np.asarray(augmentation.expected_pose_xy_yaw, dtype=np.float64),
        atol=1e-9,
    ):
        raise ValueError("Live pre-divergence pose differs from its sealed contract")
    observed = decisions[prefix_count]
    if (
        not isinstance(observed, Mapping)
        or observed.get("step") != augmentation.correction_decision_step
        or observed.get("model_action") != augmentation.observed_model_action
        or observed.get("accepted") is not True
        or observed.get("executed") is not True
        or observed.get("actual_gemma_causal_forward") is not True
        or observed.get("checkpoint_sha256") != source.checkpoint_sha256
        or observed.get("scene_prefix_sha256") != source.scene_prefix_sha256
        or observed.get("deterministic_route_planner_used") is not False
        or observed.get("substitution_applied") is not False
        or observed.get("synthetic_stop_applied") is not False
    ):
        raise ValueError("Live pre-divergence observed action differs")
    if augmentation.resume_waypoint_index >= len(route_waypoints) - 1:
        raise ValueError("Live pre-divergence resume waypoint cannot finish route")

    builder = _SyntheticTraceBuilder(
        episode_id=(
            f"live_divergence_{source.scene_id}_"
            f"d{augmentation.correction_decision_step:03d}_"
            f"{augmentation.branch_id}_{str(source.report_sha256)[:12]}"
        ),
        scene_id=str(source.scene_id),
        split="train",
        family="lap_live_divergence_correction",
        task_variant=(
            f"lap_{source.route_direction}_live_divergence_correction"
        ),
        instruction=str(source.instruction),
        start_xy=replay.pose.triple()[:2],
        initial_yaw=replay.pose.yaw,
        episode_goal_xy_m=(
            float(source.initial_pose_xy_yaw[0]),
            float(source.initial_pose_xy_yaw[1]),
        ),
        training_target_xyz_m=None,
        collision_map=collision_map,
        room_size_m=room_size_m,
        max_waypoint_step_m=max_waypoint_step_m,
        max_turn_degrees=fixed_face_step_degrees,
        history_length=history_length,
        history_encoding=history_encoding,
        source_sample_sha256=str(source.report_sha256),
    )
    builder.pose = replay.pose
    builder.seed_training_history(
        poses=replay.poses,
        action_codes=replay.action_codes,
        numeric_rows=replay.numeric_history,
        goal_progress_ledger=replay.goal_progress_ledger,
    )
    resume_target = route_waypoints[augmentation.resume_waypoint_index]
    if augmentation.recovery_plan_to_resume:
        if recovery_planner is None:
            raise ValueError("Live divergence recovery labels require a planner")
        recovery = recovery_planner.plan_to_free_point(
            np.asarray(replay.pose.triple()[:2], dtype=np.float64),
            np.asarray(resume_target, dtype=np.float64),
        )
        if not recovery.waypoints_xy_m:
            raise RuntimeError("Live divergence recovery planner returned no labels")
        continuation_waypoints = [
            *recovery.waypoints_xy_m,
            *route_waypoints[augmentation.resume_waypoint_index + 1 :],
        ]
    else:
        continuation_waypoints = [
            resume_target,
            *route_waypoints[augmentation.resume_waypoint_index + 1 :],
        ]
    if augmentation.force_first_face:
        bearing = _heading_between(
            replay.pose.triple()[:2],
            resume_target,
        )
        bearing_delta = _normalize_degrees(bearing - replay.pose.yaw)
        if abs(bearing_delta) <= 1e-7:
            raise RuntimeError("Forced live divergence FACE has no bearing direction")
        builder.face(
            _normalize_degrees(
                replay.pose.yaw
                + math.copysign(fixed_face_step_degrees, bearing_delta)
            )
        )
    trajectory = [np.asarray(point, dtype=np.float64) for point in replay.trajectory_xy]
    minimum_clearance = math.inf
    cursor = np.asarray(replay.pose.triple()[:2], dtype=np.float64)
    for waypoint_index, waypoint in enumerate(continuation_waypoints):
        target = np.asarray(waypoint, dtype=np.float64)
        check = collision_map.segment_check(cursor, target)
        if check.collision:
            raise RuntimeError("Live pre-divergence continuation contains a collision")
        minimum_clearance = min(minimum_clearance, check.clearance_m)
        builder.move_to(
            waypoint,
            fixed_face_step_degrees=fixed_face_step_degrees,
        )
        trajectory.append(target.copy())
        cursor = target
    builder.stop()
    if (
        not builder.rows
        or builder.rows[0]["expert_action"]
        != augmentation.expected_expert_first_action
        or builder.rows[-1]["expert_action"] != "STOP"
        or builder.rows[-1]["history"][-1][11] != 1.0
    ):
        raise RuntimeError("Live divergence branch action contract differs")
    total_decisions = prefix_count + len(builder.rows)
    if total_decisions > augmentation.maximum_total_decisions:
        raise RuntimeError("Live pre-divergence continuation exceeds runtime budget")

    path_length = sum(
        float(np.linalg.norm(second - first))
        for first, second in pairwise(trajectory)
    )
    signed_area = 0.5 * sum(
        float(first[0] * second[1] - first[1] * second[0])
        for first, second in pairwise(trajectory)
    )
    return_error = float(np.linalg.norm(trajectory[-1] - trajectory[0]))
    expected_sign = -1.0 if source.route_direction == "clockwise" else 1.0
    if (
        path_length < source.minimum_path_length_m
        or abs(signed_area) < source.minimum_abs_winding_area_m2
        or signed_area * expected_sign <= 0.0
        or return_error > source.maximum_return_error_m
    ):
        raise RuntimeError("Live pre-divergence continuation is not a valid lap")
    first = builder.rows[0]
    first_input_sha256 = _canonical_sha256(
        {
            "state_features": first["state_features"],
            "history": first["history"],
        }
    )
    _validate_live_history_window(
        first["history"],
        exact_prefix_transition_count=prefix_count,
        history_length=history_length,
        expected_terminal_success=bool(decisions[prefix_count - 1]["accepted"]),
        history_encoding=history_encoding,
    )
    if (
        first_input_sha256 != augmentation.expected_input_sha256
        or not np.allclose(
            np.asarray(first["state_features"], dtype=np.float64),
            np.asarray(_numeric_state(replay.pose, room_size_m), dtype=np.float64),
            atol=0.0,
        )
    ):
        raise RuntimeError("Live pre-divergence correction input differs")
    metrics = {
        "branch_id": augmentation.branch_id,
        "source_report_sha256": source.report_sha256,
        "source_checkpoint_sha256": source.checkpoint_sha256,
        "source_scene_prefix_sha256": source.scene_prefix_sha256,
        "correction_decision_step": augmentation.correction_decision_step,
        "observed_model_action": augmentation.observed_model_action,
        "expert_first_action": augmentation.expected_expert_first_action,
        "exact_prefix_transition_count": prefix_count,
        "exact_history_parameterization": history_encoding.parameterization,
        "first_correction_input_sha256": first_input_sha256,
        "continuation_sample_count": len(builder.rows),
        "total_decision_count": total_decisions,
        "path_length_m": path_length,
        "signed_winding_area_m2": signed_area,
        "abs_winding_area_m2": abs(signed_area),
        "return_error_m": return_error,
        "minimum_padded_map_clearance_m": minimum_clearance,
        "model_labeled_stop": True,
        "offline_planner_used_for_labels_only": True,
        "offline_recovery_plan_to_resume": augmentation.recovery_plan_to_resume,
        "offline_forced_first_face": augmentation.force_first_face,
        "runtime_planner_available": False,
    }
    return builder.rows, metrics


def _point_to_axis_aligned_xy_box_distance(
    point: Sequence[float], minimum: Sequence[float], maximum: Sequence[float]
) -> float:
    xy = np.asarray(point, dtype=np.float64)
    lower = np.asarray(minimum, dtype=np.float64)
    upper = np.asarray(maximum, dtype=np.float64)
    if (
        xy.shape != (2,)
        or lower.shape != (2,)
        or upper.shape != (2,)
        or not np.isfinite(xy).all()
        or not np.isfinite(lower).all()
        or not np.isfinite(upper).all()
        or np.any(upper <= lower)
    ):
        raise ValueError("Object-goal XY box geometry is invalid")
    delta = np.maximum(np.maximum(lower - xy, 0.0), xy - upper)
    return float(np.linalg.norm(delta))


def _live_object_goal_dagger_rows(
    *,
    augmentation: _LiveObjectGoalDaggerAugmentation,
    target_xyz_m: Sequence[float],
    target_bbox_min_xy_m: Sequence[float],
    target_bbox_max_xy_m: Sequence[float],
    collision_map: NumericCollisionMap,
    runtime_collision_map: NumericCollisionMap | None = None,
    approach_planner: NumericWaypointPlanner,
    room_size_m: Sequence[float],
    max_waypoint_step_m: float,
    max_turn_degrees: float,
    history_length: int,
    history_encoding: _HistoryEncoding,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Relabel one exact live object-goal divergence with a safe continuation.

    The runtime episode is replayed strictly as numeric pose and action receipts
    through the transition immediately before the configured divergence. The
    local planner and oracle are used only here to produce training labels.
    """

    runtime, _score, scored_goal = _load_authenticated_live_object_goal_reports(
        augmentation
    )
    assert augmentation.initial_pose_xy_yaw is not None
    assert augmentation.correction_decision_step is not None
    assert augmentation.expected_pose_xy_yaw is not None
    assert augmentation.expected_input_sha256 is not None
    assert augmentation.expected_expert_first_action is not None
    assert augmentation.scene_id is not None
    assert augmentation.instruction is not None
    assert augmentation.checkpoint_sha256 is not None
    assert augmentation.scene_prefix_sha256 is not None
    assert augmentation.runtime_report_path is not None
    assert augmentation.runtime_report_sha256 is not None
    decisions = runtime["model_decisions"]
    prefix_count = augmentation.correction_decision_step - 1
    replay_source = _LiveFailureDaggerAugmentation(
        enabled=True,
        report_path=augmentation.runtime_report_path,
        report_sha256=augmentation.runtime_report_sha256,
        scene_id=augmentation.scene_id,
        instruction=augmentation.instruction,
        initial_pose_xy_yaw=augmentation.initial_pose_xy_yaw,
        checkpoint_sha256=augmentation.checkpoint_sha256,
        scene_prefix_sha256=augmentation.scene_prefix_sha256,
        first_rejection_step=1,
        first_rejection_error_code="E_UNUSED_TRAINING_REPLAY",
        failed_target_xy_m=(0.0, 0.0),
        failed_target_tolerance_m=1e-6,
        route_direction="clockwise",
        resume_waypoint_index=0,
        maximum_total_decisions=augmentation.maximum_total_decisions,
        minimum_path_length_m=1e-6,
        minimum_abs_winding_area_m2=1e-6,
        maximum_return_error_m=1e6,
    )
    if prefix_count:
        replay = _replay_authenticated_live_prefix(
            augmentation=replay_source,
            decisions=decisions,
            transition_count=prefix_count,
            terminal_rejection_error_code=None,
            room_size_m=room_size_m,
            max_waypoint_step_m=max_waypoint_step_m,
            fixed_face_step_degrees=max_turn_degrees,
            history_length=history_length,
            history_encoding=history_encoding,
        )
    else:
        initial = augmentation.initial_pose_xy_yaw
        replay = _AuthenticatedLivePrefix(
            pose=_Pose(*initial),
            poses=(initial,),
            action_codes=(),
            numeric_history=(),
            trajectory_xy=(initial[:2],),
            terminal_rejected_target_xy=None,
            goal_progress_ledger=history_encoding.new_ledger(initial),
        )
    if not np.allclose(
        np.asarray(replay.pose.triple(), dtype=np.float64),
        np.asarray(augmentation.expected_pose_xy_yaw, dtype=np.float64),
        atol=1e-9,
    ):
        raise ValueError("Live object-goal correction pose differs")
    observed = decisions[prefix_count]
    if (
        not isinstance(observed, Mapping)
        or observed.get("step") != augmentation.correction_decision_step
        or observed.get("model_action") != augmentation.observed_model_action
        or observed.get("accepted") is not augmentation.observed_action_accepted
        or observed.get("executed") is not augmentation.observed_action_accepted
        or observed.get("actual_gemma_causal_forward") is not True
        or observed.get("checkpoint_sha256") != augmentation.checkpoint_sha256
        or observed.get("scene_prefix_sha256") != augmentation.scene_prefix_sha256
        or observed.get("deterministic_route_planner_used") is not False
        or observed.get("substitution_applied") is not False
        or observed.get("synthetic_stop_applied") is not False
    ):
        raise ValueError("Live object-goal observed divergence differs")

    target = np.asarray(target_xyz_m, dtype=np.float64)
    scored_target = np.asarray(scored_goal["target_center_xyz_m"], dtype=np.float64)
    if (
        target.shape != (3,)
        or not np.isfinite(target).all()
        or not np.allclose(target, scored_target, atol=1e-9)
    ):
        raise ValueError("Live object-goal oracle and score targets differ")
    start_xy = np.asarray(replay.pose.triple()[:2], dtype=np.float64)
    stop_correction = augmentation.expected_expert_first_action == "STOP"
    source_approach_planner = approach_planner
    if augmentation.offline_expert_standoff_m is not None:
        source_approach_planner = NumericWaypointPlanner(
            approach_planner.collision_map,
            grid_resolution_m=approach_planner.grid_resolution_m,
            standoff_m=augmentation.offline_expert_standoff_m,
            # A per-source safety override is an exact preregistered ring, not
            # permission for the generic planner to try closer fallback radii.
            standoff_tolerance_m=0.0,
            max_waypoint_step_m=approach_planner.max_waypoint_step_m,
            angular_samples=approach_planner.angular_samples,
            max_expanded_nodes=approach_planner.max_expanded_nodes,
        )
    approach = (
        None
        if stop_correction
        else source_approach_planner.plan(start_xy, target[:2])
    )
    if approach is not None and not approach.waypoints_xy_m:
        raise RuntimeError("Object-goal offline expert produced no approach waypoints")

    def new_builder(
        *,
        suffix: str,
        pose: _Pose,
        poses: Sequence[Sequence[float]],
        action_codes: Sequence[int],
        numeric_history: Sequence[Sequence[float]],
        ledger: WaypointGoalProgressLedger,
        episode_goal_xy_m: Sequence[float],
    ) -> _SyntheticTraceBuilder:
        value = _SyntheticTraceBuilder(
            episode_id=(
                f"live_object_goal_{augmentation.scene_id}_"
                f"{augmentation.goal_id}_d{augmentation.correction_decision_step:03d}_"
                f"{augmentation.runtime_report_sha256[:12]}{suffix}"
            ),
            scene_id=augmentation.scene_id,
            split="train",
            family="object_goal_live_divergence_correction",
            task_variant="approach_object_live_divergence_correction",
            instruction=augmentation.instruction,
            start_xy=pose.triple()[:2],
            initial_yaw=pose.yaw,
            episode_goal_xy_m=episode_goal_xy_m,
            training_target_xyz_m=target,
            collision_map=collision_map,
            room_size_m=room_size_m,
            max_waypoint_step_m=max_waypoint_step_m,
            max_turn_degrees=max_turn_degrees,
            history_length=history_length,
            history_encoding=history_encoding,
            source_sample_sha256=augmentation.runtime_report_sha256,
        )
        value.pose = pose
        value.seed_training_history(
            poses=poses,
            action_codes=action_codes,
            numeric_rows=numeric_history,
            goal_progress_ledger=ledger,
        )
        return value

    builder = new_builder(
        suffix="",
        pose=replay.pose,
        poses=replay.poses,
        action_codes=replay.action_codes,
        numeric_history=replay.numeric_history,
        ledger=replay.goal_progress_ledger,
        episode_goal_xy_m=(start_xy if approach is None else approach.goal_xy_m),
    )
    trajectory = [np.asarray(value, dtype=np.float64) for value in replay.trajectory_xy]
    stop_collision_map = runtime_collision_map or collision_map
    minimum_clearance = (
        stop_collision_map.point_check(start_xy).clearance_m
        if stop_correction
        else collision_map.point_check(start_xy).clearance_m
    )
    cursor = start_xy.copy()
    if approach is not None:
        for waypoint in approach.waypoints_xy_m:
            destination = np.asarray(waypoint, dtype=np.float64)
            check = collision_map.segment_check(cursor, destination)
            if check.collision:
                raise RuntimeError("Object-goal offline expert continuation collides")
            minimum_clearance = min(minimum_clearance, check.clearance_m)
            heading = _heading_between(cursor, destination)
            builder.face_to(heading)
            builder.move_to_direct(destination)
            trajectory.append(destination.copy())
            cursor = destination
    builder.stop()
    first = builder.rows[0]
    first_input_sha256 = _canonical_sha256(
        {"state_features": first["state_features"], "history": first["history"]}
    )
    if prefix_count:
        _validate_live_history_window(
            first["history"],
            exact_prefix_transition_count=prefix_count,
            history_length=history_length,
            expected_terminal_success=True,
            history_encoding=history_encoding,
        )
    elif first["history"] != []:
        raise RuntimeError("Decision-one object-goal correction has nonempty history")
    total_decisions = prefix_count + len(builder.rows)
    final_xy = trajectory[-1]
    initial_center_distance = float(
        np.linalg.norm(
            np.asarray(augmentation.initial_pose_xy_yaw[:2], dtype=np.float64)
            - target[:2]
        )
    )
    final_center_distance = float(np.linalg.norm(final_xy - target[:2]))
    center_progress = initial_center_distance - final_center_distance
    bbox_standoff = _point_to_axis_aligned_xy_box_distance(
        final_xy, target_bbox_min_xy_m, target_bbox_max_xy_m
    )
    path_length = sum(
        float(np.linalg.norm(second - first_point))
        for first_point, second in pairwise(trajectory)
    )
    all_rows = list(builder.rows)
    neighborhood_hashes: list[str] = []
    neighborhood_geometry: list[dict[str, Any]] = []
    if augmentation.stop_position_neighborhood_offsets_m:
        if (
            not stop_correction
            or prefix_count < 1
            or decisions[prefix_count - 1].get("model_action") != "move_to"
            or decisions[prefix_count - 1].get("accepted") is not True
        ):
            raise RuntimeError("STOP-neighborhood source is not an accepted MOVE state")
        previous = _replay_authenticated_live_prefix(
            augmentation=replay_source,
            decisions=decisions,
            transition_count=prefix_count - 1,
            terminal_rejection_error_code=None,
            room_size_m=room_size_m,
            max_waypoint_step_m=max_waypoint_step_m,
            fixed_face_step_degrees=max_turn_degrees,
            history_length=history_length,
            history_encoding=history_encoding,
        )
        source_move = decisions[prefix_count - 1]
        for neighbor_index, offset in enumerate(
            augmentation.stop_position_neighborhood_offsets_m
        ):
            neighbor_xy = start_xy + np.asarray(offset, dtype=np.float64)
            movement = neighbor_xy - np.asarray(previous.pose.triple()[:2])
            point_check = stop_collision_map.point_check(neighbor_xy)
            segment_check = stop_collision_map.segment_check(
                np.asarray(previous.pose.triple()[:2]), neighbor_xy
            )
            neighbor_center_distance = float(np.linalg.norm(neighbor_xy - target[:2]))
            neighbor_progress = initial_center_distance - neighbor_center_distance
            neighbor_standoff = _point_to_axis_aligned_xy_box_distance(
                neighbor_xy, target_bbox_min_xy_m, target_bbox_max_xy_m
            )
            if (
                point_check.collision
                or segment_check.collision
                or float(np.linalg.norm(movement)) > max_waypoint_step_m + 1e-7
                or neighbor_progress < augmentation.minimum_center_progress_m
                or neighbor_standoff > augmentation.maximum_bbox_standoff_m
            ):
                raise RuntimeError("STOP-neighborhood state is not a safe success state")
            neighbor_pose = _Pose(
                float(neighbor_xy[0]),
                float(neighbor_xy[1]),
                previous.pose.yaw,
                last_delta=(float(movement[0]), float(movement[1]), 0.0),
                linear_velocity=(float(movement[0]), float(movement[1])),
                angular_velocity=0.0,
            )
            neighbor_ledger = _copy_goal_progress_ledger(
                previous.goal_progress_ledger
            )
            neighbor_receipt = history_encoding.encode_receipt(
                ledger=neighbor_ledger,
                action="move_to",
                before_pose_xy_yaw=previous.pose.triple(),
                result_pose_xy_yaw=neighbor_pose.triple(),
                requested_waypoint_delta_robot_m=_world_target_to_robot_delta(
                    previous.pose.triple(), neighbor_xy
                ),
                requested_heading_degrees=float(
                    source_move["model_desired_heading_degrees"]
                ),
                room_size_m=room_size_m,
                max_waypoint_step_m=max_waypoint_step_m,
                success=True,
            )
            neighbor_builder = new_builder(
                suffix=f"_neighbor_{neighbor_index:02d}",
                pose=neighbor_pose,
                poses=(*previous.poses, neighbor_pose.triple()),
                action_codes=(*previous.action_codes, ACTION_TO_CODE["MOVE_TO"]),
                numeric_history=(*previous.numeric_history, tuple(neighbor_receipt)),
                ledger=neighbor_ledger,
                episode_goal_xy_m=neighbor_xy,
            )
            neighbor_builder.stop()
            neighbor_row = neighbor_builder.rows[0]
            neighbor_hash = _canonical_sha256(
                {
                    "state_features": neighbor_row["state_features"],
                    "history": neighbor_row["history"],
                }
            )
            _validate_live_history_window(
                neighbor_row["history"],
                exact_prefix_transition_count=prefix_count,
                history_length=history_length,
                expected_terminal_success=True,
                history_encoding=history_encoding,
            )
            if neighbor_row["expert_action"] != "STOP":
                raise RuntimeError("STOP-neighborhood row is not STOP-supervised")
            all_rows.append(neighbor_row)
            neighborhood_hashes.append(neighbor_hash)
            neighborhood_geometry.append(
                {
                    "offset_xy_m": [float(value) for value in offset],
                    "position_xy_m": [float(value) for value in neighbor_xy],
                    "input_sha256": neighbor_hash,
                    "target_center_progress_m": neighbor_progress,
                    "oracle_bbox_standoff_m": neighbor_standoff,
                    "padded_map_clearance_m": point_check.clearance_m,
                }
            )
            minimum_clearance = min(
                minimum_clearance,
                point_check.clearance_m,
                segment_check.clearance_m,
            )
    if (
        first_input_sha256 != augmentation.expected_input_sha256
        or first["expert_action"] != augmentation.expected_expert_first_action
        or builder.rows[-1]["expert_action"] != "STOP"
        or builder.rows[-1]["history"][-1][11] != 1.0
        or total_decisions > augmentation.maximum_total_decisions
        or center_progress < augmentation.minimum_center_progress_m
        or bbox_standoff > augmentation.maximum_bbox_standoff_m
        or minimum_clearance <= 0.0
    ):
        raise RuntimeError("Live object-goal expert continuation contract differs")
    metrics = {
        "source_runtime_report_sha256": augmentation.runtime_report_sha256,
        "source_score_report_sha256": augmentation.score_report_sha256,
        "source_checkpoint_sha256": augmentation.checkpoint_sha256,
        "source_scene_prefix_sha256": augmentation.scene_prefix_sha256,
        "goal_id": augmentation.goal_id,
        "correction_decision_step": augmentation.correction_decision_step,
        "observed_model_action": augmentation.observed_model_action,
        "expert_first_action": augmentation.expected_expert_first_action,
        "exact_prefix_transition_count": prefix_count,
        "exact_history_parameterization": history_encoding.parameterization,
        "first_correction_input_sha256": first_input_sha256,
        "continuation_sample_count": len(all_rows),
        "continuation_episode_count": 1 + len(neighborhood_hashes),
        "total_decision_count": total_decisions,
        "path_length_m": path_length,
        "initial_target_center_distance_m": initial_center_distance,
        "final_target_center_distance_m": final_center_distance,
        "target_center_progress_m": center_progress,
        "final_oracle_bbox_standoff_m": bbox_standoff,
        "minimum_padded_map_clearance_m": minimum_clearance,
        "stop_neighborhood_input_sha256": neighborhood_hashes,
        "stop_neighborhood_geometry": neighborhood_geometry,
        "model_labeled_stop": True,
        "offline_expert_standoff_m": (
            None
            if stop_correction
            else source_approach_planner.standoff_m
        ),
        "offline_expert_standoff_override": (
            augmentation.offline_expert_standoff_m is not None
        ),
        "offline_planner_used_for_labels_only": approach is not None,
        "runtime_planner_available": False,
        "runtime_oracle_available": False,
    }
    return all_rows, metrics


def _effective_settings(
    config: Mapping[str, Any], profile_name: str
) -> tuple[dict[str, Any], list[str], list[str]]:
    raw = config.get("gemma_waypoint_traces")
    if not isinstance(raw, Mapping):
        raise TypeError("Config has no gemma_waypoint_traces mapping")
    profiles = raw.get("profiles")
    if not isinstance(profiles, Mapping) or not isinstance(
        profiles.get(profile_name), Mapping
    ):
        raise TypeError(f"Unknown waypoint trace profile: {profile_name}")
    effective = {key: value for key, value in raw.items() if key != "profiles"}
    effective.update(dict(profiles[profile_name]))
    train = list(effective.get("train_scene_ids", []))
    validation = list(effective.get("validation_scene_ids", []))
    train_limit = effective.get("train_scene_limit")
    validation_limit = effective.get("validation_scene_limit")
    if train_limit is not None:
        train = train[: int(train_limit)]
    if validation_limit is not None:
        validation = validation[: int(validation_limit)]
    if (
        not train
        or not validation
        or set(train) & set(validation)
        or any(_SCENE_ID.fullmatch(scene) is None for scene in [*train, *validation])
    ):
        raise ValueError("Waypoint training requires nonempty disjoint opaque scene splits")
    return effective, train, validation


def _converted_source_rows(
    source_rows: Sequence[Mapping[str, Any]],
    *,
    selected_scene_splits: Mapping[str, str],
    room_size_m: Sequence[float],
    max_turn_degrees: float,
    max_move_m: float,
    history_length: int,
    history_encoding: _HistoryEncoding,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    histories: dict[str, list[tuple[float, float, float]]] = {}
    action_histories: dict[str, list[int]] = {}
    numeric_histories: dict[str, list[list[float]]] = {}
    ledgers: dict[str, WaypointGoalProgressLedger] = {}
    episode_steps: defaultdict[str, int] = defaultdict(int)
    rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for source in source_rows:
        scene_id = str(source.get("scene_id"))
        if scene_id not in selected_scene_splits:
            continue
        expected_split = selected_scene_splits[scene_id]
        if source.get("split") != expected_split:
            raise ValueError("V3 source row crosses the declared scene split")
        converted = convert_v3_action_to_absolute(
            source,
            room_size_m=room_size_m,
            max_turn_degrees=max_turn_degrees,
            max_move_m=max_move_m,
        )
        if converted is None:
            counts["dropped_scan"] += 1
            continue
        source_episode = str(source.get("episode_id"))
        episode_id = f"source_{source_episode}"
        history = histories.setdefault(
            episode_id, [converted.current_pose_xy_yaw]
        )
        actions = action_histories.setdefault(episode_id, [])
        numeric_history = numeric_histories.setdefault(episode_id, [])
        ledger = ledgers.setdefault(
            episode_id,
            history_encoding.new_ledger(converted.current_pose_xy_yaw),
        )
        if not _pose_close(history[-1], converted.current_pose_xy_yaw):
            raise ValueError("Converted V3 episode has a discontinuous numeric pose")
        target = source.get("oracle_target_xyz_m")
        target_xyz = (
            [float(value) for value in target]
            if source.get("target_state_available") is True
            else None
        )
        row = _make_row(
            episode_id=episode_id,
            scene_id=scene_id,
            split=expected_split,
            family=str(source.get("family")),
            task_variant="converted_v3",
            instruction=str(source.get("instruction")),
            step_index=episode_steps[episode_id],
            state_features=source.get("state_features", []),
            history_poses=history,
            history_action_codes=actions,
            history_length=history_length,
            room_size_m=room_size_m,
            max_waypoint_step_m=max_move_m,
            history_encoding=history_encoding,
            action=converted.action,
            heading_degrees=converted.heading_degrees,
            xy_m=converted.xy_m,
            episode_goal_xy_m=(
                None
                if target_xyz is None
                else (float(target_xyz[0]), float(target_xyz[1]))
            ),
            training_target_xyz_m=target_xyz,
            source_sample_sha256=_canonical_sha256(source),
            history_numeric_rows=numeric_history,
        )
        rows.append(row)
        episode_steps[episode_id] += 1
        history.append(converted.completion_pose_xy_yaw)
        actions.append(ACTION_TO_CODE[converted.action])
        requested_delta = (
            _world_target_to_robot_delta(
                converted.current_pose_xy_yaw,
                converted.xy_m,
            )
            if converted.action == "MOVE_TO" and converted.xy_m is not None
            else (0.0, 0.0)
        )
        numeric_history.append(
            list(
                history_encoding.encode_receipt(
                    ledger=ledger,
                    action=converted.action,
                    before_pose_xy_yaw=(
                        float(ledger.current_xy_m[0]),
                        float(ledger.current_xy_m[1]),
                        float(converted.current_pose_xy_yaw[2]),
                    ),
                    result_pose_xy_yaw=converted.completion_pose_xy_yaw,
                    requested_waypoint_delta_robot_m=requested_delta,
                    requested_heading_degrees=(
                        float(converted.heading_degrees)
                        if converted.heading_degrees is not None
                        else float(converted.current_pose_xy_yaw[2])
                    ),
                    room_size_m=room_size_m,
                    max_waypoint_step_m=max_move_m,
                    success=True,
                )
            )
        )
        counts[f"converted_{source.get('action_name')!s}"] += 1
    if not rows:
        raise RuntimeError("No V3 rows were selected for absolute conversion")
    return rows, counts


def generate_gemma_waypoint_trace_dataset(
    config: Mapping[str, Any],
    destination: str | Path,
    *,
    profile: str = "smoke",
) -> dict[str, Any]:
    """Build authenticated conversion, lap, and between-object teacher traces."""

    settings, train_scenes, validation_scenes = _effective_settings(config, profile)
    root = _rooted(destination)
    _require_training_path(root)
    if root.exists():
        raise FileExistsError(f"Gemma waypoint trace dataset already exists: {root}")
    source_root = _rooted(str(settings["source_trace_dataset"]))
    _require_training_path(source_root)
    source_manifest, source_rows = load_navigation_target_trace_v3(source_root)
    configured_train = list(config["gemma_waypoint_traces"]["train_scene_ids"])
    configured_validation = list(
        config["gemma_waypoint_traces"]["validation_scene_ids"]
    )
    if (
        source_manifest.get("train_scene_ids") != configured_train
        or source_manifest.get("validation_scene_ids") != configured_validation
        or set(configured_train) & set(configured_validation)
    ):
        raise ValueError("V3 source traces differ from configured disjoint splits")

    room_size = tuple(float(value) for value in config["scene"]["room_size_m"])
    robot = config["robot"]
    executor_max_turn = float(robot["max_turn_degrees"])
    turn_margin = float(settings.get("expert_turn_margin_degrees", 0.0))
    if (
        not math.isfinite(turn_margin)
        or turn_margin < 0.0
        or turn_margin >= executor_max_turn
    ):
        raise ValueError("Waypoint expert turn margin is outside runtime bounds")
    max_turn = executor_max_turn - turn_margin
    lap_fixed_face_step = float(
        settings.get("lap_fixed_face_step_degrees", min(40.0, max_turn))
    )
    if (
        not math.isfinite(lap_fixed_face_step)
        or lap_fixed_face_step <= 0.0
        or lap_fixed_face_step > max_turn
    ):
        raise ValueError("Lap fixed FACE step is outside expert turn bounds")
    max_move = float(robot["max_move_m"])
    max_waypoint_step = float(settings.get("max_waypoint_step_m", max_move))
    history_length = int(settings.get("history_length", 16))
    if history_length < 2 or max_waypoint_step <= 0 or max_waypoint_step > float(
        robot.get("max_move_to_m", 1.0)
    ):
        raise ValueError("Waypoint trace bounds or history length are invalid")
    history_encoding = _HistoryEncoding.from_settings(
        settings,
        history_length=history_length,
    )
    yaws = [float(value) for value in settings["initial_yaw_degrees"]]
    if not yaws or not all(math.isfinite(value) for value in yaws):
        raise ValueError("Waypoint trace initial yaws are invalid")
    yaw_jitter_raw = settings.get("training_initial_yaw_jitter_degrees", [0.0])
    if (
        not isinstance(yaw_jitter_raw, list)
        or not yaw_jitter_raw
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or abs(float(value)) > max_turn
            for value in yaw_jitter_raw
        )
    ):
        raise ValueError("Training initial-yaw jitter is invalid")
    yaw_jitter = tuple(float(value) for value in yaw_jitter_raw)
    if 0.0 not in yaw_jitter or len(set(yaw_jitter)) != len(yaw_jitter):
        raise ValueError("Training initial-yaw jitter must uniquely include zero")
    recovery_augmentation = _lap_recovery_augmentation(
        settings,
        max_waypoint_step_m=max_waypoint_step,
        max_turn_degrees=max_turn,
        history_length=history_length,
    )
    execution_drift_augmentation = _lap_execution_drift_augmentation(
        settings,
        fixed_face_step_degrees=lap_fixed_face_step,
    )
    live_failure_augmentations = _live_failure_dagger_augmentations(settings)
    live_pre_divergence_augmentations = (
        _live_pre_divergence_dagger_augmentations(
            settings, live_failure_augmentations
        )
    )
    live_object_goal_augmentations = _live_object_goal_dagger_augmentations(settings)
    runtime_surface_padding = float(robot.get("surface_padding_m", 0.035))
    expert_surface_padding = float(
        settings.get("expert_surface_padding_m", runtime_surface_padding)
    )
    minimum_runtime_clearance_margin = float(
        settings.get("minimum_runtime_clearance_margin_m", 0.0)
    )
    if (
        not math.isfinite(runtime_surface_padding)
        or runtime_surface_padding < 0.0
        or not math.isfinite(expert_surface_padding)
        or expert_surface_padding < runtime_surface_padding
        or not math.isfinite(minimum_runtime_clearance_margin)
        or minimum_runtime_clearance_margin < 0.0
        or expert_surface_padding + 1e-12
        < runtime_surface_padding + minimum_runtime_clearance_margin
    ):
        raise ValueError("Offline expert surface-clearance margin is invalid")
    start_count = int(settings["start_pose_count"])
    between_count = int(settings["between_pairs_per_scene"])
    object_target_count = int(settings.get("object_targets_per_scene", 1))
    selected_splits = {
        **{scene: "train" for scene in train_scenes},
        **{scene: "validation" for scene in validation_scenes},
    }
    include_source_rows = bool(settings.get("include_source_rows", True))
    if include_source_rows:
        rows, source_counts = _converted_source_rows(
            source_rows,
            selected_scene_splits=selected_splits,
            room_size_m=room_size,
            max_turn_degrees=max_turn,
            max_move_m=max_move,
            history_length=history_length,
            history_encoding=history_encoding,
        )
    else:
        rows = []
        source_counts = Counter(
            {
                "excluded_by_profile": sum(
                    1
                    for row in source_rows
                    if str(row.get("scene_id")) in selected_splits
                )
            }
        )

    oracle_root = _rooted(str(settings["oracle_root"]))
    map_root = _rooted(str(settings["map_root"]))
    oracle_hashes: dict[str, str] = {}
    map_hashes: dict[str, str] = {}
    family_episodes: Counter[str] = Counter()
    variant_episodes: Counter[str] = Counter()
    synthetic_episode_counter = 0
    lap_recovery_sample_count = 0
    lap_recovery_episode_count = 0
    lap_execution_drift_sample_count = 0
    lap_execution_drift_episode_count = 0
    live_failure_dagger_sample_count = 0
    live_failure_dagger_episode_count = 0
    live_failure_dagger_metrics_by_report: dict[str, dict[str, Any]] = {}
    live_pre_divergence_sample_count = 0
    live_pre_divergence_episode_count = 0
    live_pre_divergence_metrics_by_branch: dict[str, dict[str, Any]] = {}
    live_object_goal_dagger_sample_count = 0
    live_object_goal_dagger_episode_count = 0
    live_object_goal_dagger_metrics_by_report: dict[str, dict[str, Any]] = {}
    seed = int(settings.get("seed", config.get("seed", 0)))
    all_scenes = [*train_scenes, *validation_scenes]
    for scene_index, scene_id in enumerate(all_scenes):
        split = selected_splits[scene_id]
        lap_yaws = yaws
        if split == "train":
            lap_yaws = list(
                dict.fromkeys(
                    _normalize_degrees(yaw + offset)
                    for yaw in yaws
                    for offset in yaw_jitter
                )
            )
        oracle_path = oracle_root / scene_id / "oracle.json"
        map_path = map_root / scene_id / "voxel_map.npz"
        if (
            not oracle_path.is_file()
            or oracle_path.is_symlink()
            or not map_path.is_file()
            or map_path.is_symlink()
        ):
            raise FileNotFoundError(
                f"Waypoint training input is unavailable for {scene_id}"
            )
        objects = _load_oracle_objects(oracle_path, scene_id)
        collision_map = NumericCollisionMap.from_voxel_map(
            map_path,
            room_size_m=room_size,
            robot_radius_m=float(robot["radius_m"]),
            collision_z_min_m=float(robot.get("collision_z_min_m", 0.12)),
            collision_z_max_m=float(robot.get("collision_z_max_m", 1.80)),
            surface_padding_m=expert_surface_padding,
        )
        runtime_collision_map = NumericCollisionMap.from_voxel_map(
            map_path,
            room_size_m=room_size,
            robot_radius_m=float(robot["radius_m"]),
            collision_z_min_m=float(robot.get("collision_z_min_m", 0.12)),
            collision_z_max_m=float(robot.get("collision_z_max_m", 1.80)),
            surface_padding_m=runtime_surface_padding,
        )
        required_starts = settings.get("required_start_pose_xy_by_scene", {})
        if not isinstance(required_starts, Mapping):
            raise TypeError("required_start_pose_xy_by_scene must be a mapping")
        starts = _safe_starts(
            collision_map,
            count=start_count,
            seed=seed + scene_index * 101,
            required_first=required_starts.get(scene_id),
        )
        patrol = NumericPatrolPlanner(
            collision_map,
            anchor_count=int(settings.get("lap_anchor_count", 8)),
            wall_margin_m=float(settings.get("lap_wall_margin_m", 0.45)),
            grid_resolution_m=float(settings.get("planner_grid_resolution_m", 0.15)),
            max_waypoint_step_m=max_waypoint_step,
            max_waypoints=int(settings.get("lap_max_waypoints", 64)),
        )
        between_planner = NumericWaypointPlanner(
            collision_map,
            grid_resolution_m=float(settings.get("planner_grid_resolution_m", 0.15)),
            standoff_m=max(0.20, collision_map.inflated_radius_m),
            standoff_tolerance_m=0.05,
            max_waypoint_step_m=max_waypoint_step,
        )
        approach_planner = NumericWaypointPlanner(
            collision_map,
            grid_resolution_m=float(settings.get("planner_grid_resolution_m", 0.15)),
            standoff_m=float(settings.get("approach_standoff_m", 0.60)),
            standoff_tolerance_m=float(
                settings.get("approach_standoff_tolerance_m", 0.20)
            ),
            max_waypoint_step_m=max_waypoint_step,
            angular_samples=int(settings.get("approach_angular_samples", 48)),
        )
        pairs = _between_pairs(
            objects,
            count=between_count,
            seed=seed + scene_index * 101 + 37,
        )
        object_targets = _object_targets(
            objects,
            count=object_target_count,
            seed=seed + scene_index * 101 + 73,
        )
        scene_between_episodes = 0
        scene_face_episodes = 0
        scene_approach_episodes = 0

        for start_index, start in enumerate(starts):
            try:
                routes = _lap_routes(start, patrol)
            except RuntimeError as error:
                raise RuntimeError(
                    f"No complete lap teacher route for {scene_id} start {start_index}"
                ) from error
            for yaw_index, yaw in enumerate(lap_yaws):
                for direction in ("clockwise", "counterclockwise"):
                    generic_lap = settings.get("generic_clockwise_lap_instructions")
                    if generic_lap is not None and (
                        not isinstance(generic_lap, list)
                        or not generic_lap
                        or any(not isinstance(value, str) or not value.strip() for value in generic_lap)
                    ):
                        raise ValueError("generic_clockwise_lap_instructions must be text")
                    lap_instruction = (
                        str(generic_lap[yaw_index % len(generic_lap)])
                        if direction == "clockwise" and generic_lap
                        else (
                            f"Make one complete {direction} lap around the room and "
                            "return to the starting position, then stop."
                        )
                    )
                    episode_id = (
                        f"lap_{scene_id[6:]}_{start_index:02d}_{yaw_index:02d}_"
                        f"{'cw' if direction == 'clockwise' else 'ccw'}"
                    )
                    builder = _SyntheticTraceBuilder(
                        episode_id=episode_id,
                        scene_id=scene_id,
                        split=split,
                        family="lap",
                        task_variant=f"lap_{direction}",
                        instruction=lap_instruction,
                        start_xy=start,
                        initial_yaw=yaw,
                        episode_goal_xy_m=(float(start[0]), float(start[1])),
                        training_target_xyz_m=None,
                        collision_map=collision_map,
                        room_size_m=room_size,
                        max_waypoint_step_m=max_waypoint_step,
                        max_turn_degrees=max_turn,
                        history_length=history_length,
                        history_encoding=history_encoding,
                    )
                    for waypoint in routes[direction]:
                        builder.move_to(
                            waypoint,
                            fixed_face_step_degrees=lap_fixed_face_step,
                        )
                    builder.stop()
                    if not builder.rows:
                        raise RuntimeError("Lap teacher produced an empty episode")
                    rows.extend(builder.rows)
                    synthetic_episode_counter += 1
                    family_episodes["lap"] += 1
                    variant_episodes[f"lap_{direction}"] += 1
                    execution_drift_yaw_allowed = (
                        not execution_drift_augmentation.nominal_initial_yaws_only
                        or any(
                            abs(_normalize_degrees(yaw - nominal_yaw)) <= 1e-9
                            for nominal_yaw in yaws
                        )
                    )
                    if (
                        execution_drift_augmentation.enabled
                        and split == "train"
                        and execution_drift_yaw_allowed
                    ):
                        drift_rows, drift_episode_count = (
                            _lap_execution_drift_rows(
                                episode_id=episode_id,
                                scene_id=scene_id,
                                split=split,
                                direction=direction,
                                instruction=lap_instruction,
                                start_xy=start,
                                initial_yaw=yaw,
                                route_waypoints=routes[direction],
                                collision_map=collision_map,
                                room_size_m=room_size,
                                max_waypoint_step_m=max_waypoint_step,
                                fixed_face_step_degrees=lap_fixed_face_step,
                                history_length=history_length,
                                history_encoding=history_encoding,
                                augmentation=execution_drift_augmentation,
                            )
                        )
                        rows.extend(drift_rows)
                        lap_execution_drift_sample_count += len(drift_rows)
                        lap_execution_drift_episode_count += drift_episode_count
                        synthetic_episode_counter += drift_episode_count
                        family_episodes[
                            "lap_execution_drift"
                        ] += drift_episode_count
                        variant_episodes[
                            f"lap_{direction}_execution_drift"
                        ] += drift_episode_count
                    recovery_yaw_allowed = (
                        not recovery_augmentation.nominal_initial_yaws_only
                        or any(
                            abs(_normalize_degrees(yaw - nominal_yaw)) <= 1e-9
                            for nominal_yaw in yaws
                        )
                    )
                    if (
                        recovery_augmentation.enabled
                        and split == "train"
                        and recovery_yaw_allowed
                    ):
                        recovery_rows = _lap_recovery_rows(
                            builder.rows,
                            collision_map=collision_map,
                            recovery_planner=between_planner,
                            room_size_m=room_size,
                            max_waypoint_step_m=max_waypoint_step,
                            fixed_face_step_degrees=lap_fixed_face_step,
                            history_length=history_length,
                            history_encoding=history_encoding,
                            augmentation=recovery_augmentation,
                        )
                        rows.extend(recovery_rows)
                        new_recovery_episode_count = len(
                            {str(row["episode_id"]) for row in recovery_rows}
                        )
                        lap_recovery_sample_count += len(recovery_rows)
                        lap_recovery_episode_count += new_recovery_episode_count
                        synthetic_episode_counter += new_recovery_episode_count
                        family_episodes[
                            "lap_recovery"
                        ] += new_recovery_episode_count
                        variant_episodes[
                            f"lap_{direction}_dagger_recovery"
                        ] += new_recovery_episode_count
                    for live_failure_augmentation in live_failure_augmentations:
                        live_initial = (
                            live_failure_augmentation.initial_pose_xy_yaw
                        )
                        if (
                            live_initial is not None
                            and scene_id == live_failure_augmentation.scene_id
                            and split == "train"
                            and direction
                            == live_failure_augmentation.route_direction
                            and np.allclose(
                                start,
                                np.asarray(live_initial[:2], dtype=np.float64),
                                atol=1e-9,
                            )
                            and abs(
                                _normalize_degrees(
                                    float(yaw) - float(live_initial[2])
                                )
                            )
                            <= 1e-9
                            and lap_instruction
                            == live_failure_augmentation.instruction
                        ):
                            report_digest = str(
                                live_failure_augmentation.report_sha256
                            )
                            if report_digest in live_failure_dagger_metrics_by_report:
                                raise RuntimeError(
                                    "Live-failure DAgger augmentation matched twice"
                                )
                            discriminator = (
                                report_digest[:12]
                                if len(live_failure_augmentations) > 1
                                else None
                            )
                            failure_rows, failure_metrics = (
                                _live_failure_dagger_rows(
                                    augmentation=live_failure_augmentation,
                                    route_waypoints=routes[direction],
                                    collision_map=collision_map,
                                    recovery_planner=between_planner,
                                    room_size_m=room_size,
                                    max_waypoint_step_m=max_waypoint_step,
                                    fixed_face_step_degrees=lap_fixed_face_step,
                                    history_length=history_length,
                                    history_encoding=history_encoding,
                                    episode_discriminator=discriminator,
                                )
                            )
                            rows.extend(failure_rows)
                            live_failure_dagger_sample_count += len(failure_rows)
                            live_failure_dagger_episode_count += 1
                            live_failure_dagger_metrics_by_report[
                                report_digest
                            ] = failure_metrics
                            synthetic_episode_counter += 1
                            family_episodes["lap_live_failure_recovery"] += 1
                            variant_episodes[
                                f"lap_{direction}_live_failure_recovery"
                            ] += 1
                    for divergence_augmentation in (
                        live_pre_divergence_augmentations
                    ):
                        live_source = divergence_augmentation.source
                        live_initial = live_source.initial_pose_xy_yaw
                        if (
                            live_initial is not None
                            and scene_id == live_source.scene_id
                            and split == "train"
                            and direction == live_source.route_direction
                            and np.allclose(
                                start,
                                np.asarray(live_initial[:2], dtype=np.float64),
                                atol=1e-9,
                            )
                            and abs(
                                _normalize_degrees(
                                    float(yaw) - float(live_initial[2])
                                )
                            )
                            <= 1e-9
                            and lap_instruction == live_source.instruction
                        ):
                            branch_key = (
                                f"{live_source.report_sha256}:"
                                f"{divergence_augmentation.branch_id}"
                            )
                            if branch_key in live_pre_divergence_metrics_by_branch:
                                raise RuntimeError(
                                    "Live pre-divergence branch matched twice"
                                )
                            divergence_rows, divergence_metrics = (
                                _live_pre_divergence_dagger_rows(
                                    augmentation=divergence_augmentation,
                                    route_waypoints=routes[direction],
                                    collision_map=collision_map,
                                    recovery_planner=between_planner,
                                    room_size_m=room_size,
                                    max_waypoint_step_m=max_waypoint_step,
                                    fixed_face_step_degrees=lap_fixed_face_step,
                                    history_length=history_length,
                                    history_encoding=history_encoding,
                                )
                            )
                            rows.extend(divergence_rows)
                            live_pre_divergence_sample_count += len(
                                divergence_rows
                            )
                            live_pre_divergence_episode_count += 1
                            live_pre_divergence_metrics_by_branch[
                                branch_key
                            ] = divergence_metrics
                            synthetic_episode_counter += 1
                            family_episodes[
                                "lap_live_divergence_correction"
                            ] += 1
                            variant_episodes[
                                f"lap_{direction}_live_divergence_correction"
                            ] += 1

            for pair_index, (first_name, second_name) in enumerate(pairs):
                for yaw_index, yaw in enumerate(yaws):
                    try:
                        goal, waypoints = _plan_between(
                            start=start,
                            first_xyz=objects[first_name],
                            second_xyz=objects[second_name],
                            planner=between_planner,
                            search_resolution_m=float(
                                settings.get("between_search_resolution_m", 0.05)
                            ),
                        )
                    except RuntimeError:
                        continue
                    episode_id = (
                        f"between_{scene_id[6:]}_{start_index:02d}_"
                        f"{yaw_index:02d}_{pair_index:02d}"
                    )
                    builder = _SyntheticTraceBuilder(
                        episode_id=episode_id,
                        scene_id=scene_id,
                        split=split,
                        family="between",
                        task_variant="between_objects",
                        instruction=(
                            f"Move between the {first_name} and the {second_name}, "
                            "then stop."
                        ),
                        start_xy=start,
                        initial_yaw=yaw,
                        episode_goal_xy_m=(float(goal[0]), float(goal[1])),
                        training_target_xyz_m=None,
                        collision_map=collision_map,
                        room_size_m=room_size,
                        max_waypoint_step_m=max_waypoint_step,
                        max_turn_degrees=max_turn,
                        history_length=history_length,
                        history_encoding=history_encoding,
                    )
                    for waypoint in waypoints:
                        builder.move_to(waypoint)
                    builder.stop()
                    rows.extend(builder.rows)
                    synthetic_episode_counter += 1
                    family_episodes["between"] += 1
                    variant_episodes["between_objects"] += 1
                    scene_between_episodes += 1

            for target_index, target_name in enumerate(object_targets):
                target_xyz = objects[target_name]
                target_xy = np.asarray(target_xyz[:2], dtype=np.float64)
                try:
                    approach_plan = approach_planner.plan(start, target_xy)
                except (RuntimeError, ValueError):
                    continue
                for yaw_index, yaw in enumerate(yaws):
                    face_id = (
                        f"face_{scene_id[6:]}_{start_index:02d}_"
                        f"{yaw_index:02d}_{target_index:02d}"
                    )
                    face_builder = _SyntheticTraceBuilder(
                        episode_id=face_id,
                        scene_id=scene_id,
                        split=split,
                        family="face",
                        task_variant="face_object",
                        instruction=f"Face the {target_name}, then stop.",
                        start_xy=start,
                        initial_yaw=yaw,
                        episode_goal_xy_m=(float(target_xy[0]), float(target_xy[1])),
                        training_target_xyz_m=target_xyz,
                        collision_map=collision_map,
                        room_size_m=room_size,
                        max_waypoint_step_m=max_waypoint_step,
                        max_turn_degrees=max_turn,
                        history_length=history_length,
                        history_encoding=history_encoding,
                    )
                    face_builder.face_to(
                        _heading_between(start, target_xy), ensure_action=True
                    )
                    face_builder.stop()
                    rows.extend(face_builder.rows)
                    synthetic_episode_counter += 1
                    family_episodes["face"] += 1
                    variant_episodes["face_object"] += 1
                    scene_face_episodes += 1

                    approach_id = (
                        f"approach_{scene_id[6:]}_{start_index:02d}_"
                        f"{yaw_index:02d}_{target_index:02d}"
                    )
                    approach_builder = _SyntheticTraceBuilder(
                        episode_id=approach_id,
                        scene_id=scene_id,
                        split=split,
                        family="approach",
                        task_variant="approach_object",
                        instruction=f"Move near the {target_name}, face it, then stop.",
                        start_xy=start,
                        initial_yaw=yaw,
                        episode_goal_xy_m=approach_plan.goal_xy_m,
                        training_target_xyz_m=target_xyz,
                        collision_map=collision_map,
                        room_size_m=room_size,
                        max_waypoint_step_m=max_waypoint_step,
                        max_turn_degrees=max_turn,
                        history_length=history_length,
                        history_encoding=history_encoding,
                    )
                    for waypoint in approach_plan.waypoints_xy_m:
                        approach_builder.move_to(waypoint)
                    approach_position = np.asarray(
                        [approach_builder.pose.x, approach_builder.pose.y],
                        dtype=np.float64,
                    )
                    approach_builder.face_to(
                        _heading_between(approach_position, target_xy),
                        ensure_action=True,
                    )
                    approach_builder.stop()
                    rows.extend(approach_builder.rows)
                    synthetic_episode_counter += 1
                    family_episodes["approach"] += 1
                    variant_episodes["approach_object"] += 1
                    scene_approach_episodes += 1

            for live_object_goal_augmentation in live_object_goal_augmentations:
                live_object_initial = (
                    live_object_goal_augmentation.initial_pose_xy_yaw
                )
                if (
                    live_object_initial is not None
                    and split == "train"
                    and scene_id == live_object_goal_augmentation.scene_id
                    and np.allclose(
                        start,
                        np.asarray(live_object_initial[:2], dtype=np.float64),
                        atol=1e-9,
                    )
                ):
                    report_digest = str(
                        live_object_goal_augmentation.runtime_report_sha256
                    )
                    if report_digest in live_object_goal_dagger_metrics_by_report:
                        raise RuntimeError(
                            "Live object-goal DAgger augmentation matched twice"
                        )
                    assert live_object_goal_augmentation.target_category is not None
                    assert live_object_goal_augmentation.target_instance_id is not None
                    target_center, bbox_min_xy, bbox_max_xy = (
                        _load_training_oracle_object_geometry(
                            oracle_path,
                            scene_id=scene_id,
                            category=live_object_goal_augmentation.target_category,
                            instance_id=live_object_goal_augmentation.target_instance_id,
                        )
                    )
                    object_rows, object_metrics = _live_object_goal_dagger_rows(
                        augmentation=live_object_goal_augmentation,
                        target_xyz_m=target_center,
                        target_bbox_min_xy_m=bbox_min_xy,
                        target_bbox_max_xy_m=bbox_max_xy,
                        collision_map=collision_map,
                        runtime_collision_map=runtime_collision_map,
                        approach_planner=approach_planner,
                        room_size_m=room_size,
                        max_waypoint_step_m=max_waypoint_step,
                        max_turn_degrees=max_turn,
                        history_length=history_length,
                        history_encoding=history_encoding,
                    )
                    object_episode_count = int(
                        object_metrics["continuation_episode_count"]
                    )
                    rows.extend(object_rows)
                    live_object_goal_dagger_sample_count += len(object_rows)
                    live_object_goal_dagger_episode_count += object_episode_count
                    live_object_goal_dagger_metrics_by_report[
                        report_digest
                    ] = object_metrics
                    synthetic_episode_counter += object_episode_count
                    family_episodes[
                        "object_goal_live_divergence_correction"
                    ] += object_episode_count
                    variant_episodes[
                        "approach_object_live_divergence_correction"
                    ] += object_episode_count

        if scene_between_episodes < 1:
            raise RuntimeError(f"No between-object expert trace was built for {scene_id}")
        if scene_face_episodes < 1 or scene_approach_episodes < 1:
            raise RuntimeError(f"No object-facing/approach trace was built for {scene_id}")
        oracle_hashes[scene_id] = _sha256(oracle_path)
        map_hashes[scene_id] = _sha256(map_path)

    if live_failure_augmentations and (
        live_failure_dagger_episode_count != len(live_failure_augmentations)
        or len(live_failure_dagger_metrics_by_report)
        != len(live_failure_augmentations)
    ):
        raise RuntimeError(
            "Each live-failure DAgger augmentation must match one train episode"
        )
    if live_pre_divergence_augmentations and (
        live_pre_divergence_episode_count
        != len(live_pre_divergence_augmentations)
        or len(live_pre_divergence_metrics_by_branch)
        != len(live_pre_divergence_augmentations)
    ):
        raise RuntimeError(
            "Each live pre-divergence branch must match one train episode"
        )
    if live_object_goal_augmentations and (
        len(live_object_goal_dagger_metrics_by_report)
        != len(live_object_goal_augmentations)
    ):
        raise RuntimeError(
            "Each live object-goal DAgger augmentation must match one train source"
        )

    if (
        family_episodes["lap"] < 2 * len(all_scenes)
        or variant_episodes["lap_clockwise"] < len(all_scenes)
        or variant_episodes["lap_counterclockwise"] < len(all_scenes)
        or family_episodes["between"] < len(all_scenes)
        or family_episodes["face"] < len(all_scenes)
        or family_episodes["approach"] < len(all_scenes)
    ):
        raise RuntimeError("Waypoint teacher did not cover every synthetic task family")

    for index, row in enumerate(rows):
        row["sample_id"] = f"w_{index:08d}"
    exact_input_targets: dict[str, str] = {}
    for row in rows:
        input_digest = _canonical_sha256(
            {
                "scene_id": row["scene_id"],
                "instruction": row["instruction"],
                "state_features": row["state_features"],
                "history": row["history"],
            }
        )
        target_digest = _canonical_sha256(
            {
                "expert_action": row["expert_action"],
                "waypoint_delta_robot_m": row["waypoint_delta_robot_m"],
                "heading_degrees": row["heading_degrees"],
            }
        )
        previous = exact_input_targets.setdefault(input_digest, target_digest)
        if previous != target_digest:
            raise RuntimeError("Exact Gemma input has contradictory expert labels")
    observed_splits = {
        scene: {str(row["split"]) for row in rows if row["scene_id"] == scene}
        for scene in all_scenes
    }
    if any(observed_splits[scene] != {selected_splits[scene]} for scene in all_scenes):
        raise RuntimeError("Waypoint rows violate scene-level split isolation")

    root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{root.name}.", dir=root.parent))
    try:
        traces_path = temporary / "traces.jsonl"
        with traces_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        action_counts = Counter(str(row["expert_action"]) for row in rows)
        family_counts = Counter(str(row["family"]) for row in rows)
        episode_ids = {str(row["episode_id"]) for row in rows}
        source_episode_count = len(
            {episode for episode in episode_ids if episode.startswith("source_")}
        )
        stable_settings = {
            key: value
            for key, value in settings.items()
            if key
            not in {
                "source_trace_dataset",
                "oracle_root",
                "map_root",
                "train_scene_ids",
                "validation_scene_ids",
            }
        }
        live_failure_enabled = bool(live_failure_augmentations)
        primary_live_failure_metrics = (
            live_failure_dagger_metrics_by_report.get(
                str(live_failure_augmentations[0].report_sha256)
            )
            if live_failure_augmentations
            else None
        )
        live_object_goal_enabled = bool(live_object_goal_augmentations)
        primary_live_object_goal_metrics = (
            live_object_goal_dagger_metrics_by_report.get(
                str(live_object_goal_augmentations[0].runtime_report_sha256)
            )
            if live_object_goal_augmentations
            else None
        )
        body: dict[str, Any] = {
            "schema": MANIFEST_SCHEMA,
            "profile": profile,
            "sample_count": len(rows),
            "episode_count": len(episode_ids),
            "source_converted_episode_count": source_episode_count,
            "source_rows_included": include_source_rows,
            "synthetic_expert_episode_count": synthetic_episode_counter,
            "train_scene_ids": train_scenes,
            "validation_scene_ids": validation_scenes,
            "train_scene_count": len(train_scenes),
            "validation_scene_count": len(validation_scenes),
            "scene_splits_disjoint": True,
            "contradictory_exact_input_count": 0,
            "action_names": list(ACTION_NAMES),
            "action_sample_counts": {
                name: int(action_counts[name]) for name in ACTION_NAMES
            },
            "family_sample_counts": {
                key: int(value) for key, value in sorted(family_counts.items())
            },
            "synthetic_family_episode_counts": {
                key: int(value) for key, value in sorted(family_episodes.items())
            },
            "synthetic_variant_episode_counts": {
                key: int(value) for key, value in sorted(variant_episodes.items())
            },
            "source_action_conversion_counts": {
                key: int(value) for key, value in sorted(source_counts.items())
            },
            "state_feature_dim": ROBOT_STATE_FEATURE_DIM,
            "history_length": history_length,
            "history_feature_dim": history_encoding.feature_dim,
            "history_parameterization": history_encoding.parameterization,
            "history_goal_progress_from_numeric_receipts_only": (
                history_encoding.is_v2
            ),
            "history_goal_progress_question_independent": history_encoding.is_v2,
            "history_goal_progress_rejection_streak_scale": (
                history_encoding.rejection_streak_scale
                if history_encoding.is_v2
                else None
            ),
            "max_waypoint_step_m": max_waypoint_step,
            "max_face_step_degrees": max_turn,
            "runtime_max_face_step_degrees": executor_max_turn,
            "expert_turn_margin_degrees": turn_margin,
            "runtime_surface_padding_m": runtime_surface_padding,
            "expert_surface_padding_m": expert_surface_padding,
            "minimum_runtime_clearance_margin_m": (
                minimum_runtime_clearance_margin
            ),
            "expert_routes_use_padded_numeric_collision_map": True,
            "lap_fixed_face_step_degrees": lap_fixed_face_step,
            "lap_face_actions_fixed_magnitude": True,
            "lap_residual_face_actions_omitted": True,
            "training_initial_yaw_jitter_degrees": list(yaw_jitter),
            "training_initial_yaw_jitter_train_split_only": True,
            "lap_execution_drift_augmentation_enabled": (
                execution_drift_augmentation.enabled
            ),
            "lap_execution_drift_nominal_initial_yaws_only": (
                execution_drift_augmentation.nominal_initial_yaws_only
            ),
            "lap_execution_drift_post_face_rows_only": (
                execution_drift_augmentation.post_face_rows_only
            ),
            "lap_execution_drift_turn_magnitude_profiles_degrees": [
                list(profile)
                for profile in (
                    execution_drift_augmentation.turn_magnitude_profiles_degrees
                )
            ],
            "lap_execution_drift_sample_count": lap_execution_drift_sample_count,
            "lap_execution_drift_episode_count": lap_execution_drift_episode_count,
            "lap_execution_drift_train_split_only": True,
            "lap_execution_drift_transition_aligned_histories": True,
            "lap_recovery_augmentation_enabled": recovery_augmentation.enabled,
            "lap_recovery_full_post_recovery_continuation": (
                recovery_augmentation.full_post_recovery_continuation
            ),
            "lap_recovery_nominal_initial_yaws_only": (
                recovery_augmentation.nominal_initial_yaws_only
            ),
            "lap_recovery_sample_count": lap_recovery_sample_count,
            "lap_recovery_episode_count": lap_recovery_episode_count,
            "lap_recovery_rejection_streak_lengths": list(
                recovery_augmentation.rejection_streak_lengths
            ),
            "lap_recovery_train_split_only": True,
            "lap_recovery_correct_stop_supervised": True,
            "live_failure_dagger_augmentation_enabled": (
                live_failure_enabled
            ),
            "live_failure_dagger_sample_count": live_failure_dagger_sample_count,
            "live_failure_dagger_episode_count": live_failure_dagger_episode_count,
            "live_failure_dagger_train_split_only": True,
            "live_failure_dagger_exact_runtime_history": (
                live_failure_enabled
            ),
            "live_failure_dagger_full_collision_safe_continuation": (
                live_failure_enabled
            ),
            "live_failure_dagger_correct_stop_supervised": (
                live_failure_enabled
            ),
            "live_failure_dagger_metrics": primary_live_failure_metrics,
            "live_divergence_dagger_augmentation_enabled": bool(
                live_pre_divergence_augmentations
            ),
            "live_divergence_dagger_sample_count": (
                live_pre_divergence_sample_count
            ),
            "live_divergence_dagger_episode_count": (
                live_pre_divergence_episode_count
            ),
            "live_divergence_dagger_train_split_only": True,
            "live_divergence_dagger_exact_runtime_history": bool(
                live_pre_divergence_augmentations
            ),
            "live_divergence_dagger_direct_class_correction": bool(
                live_pre_divergence_augmentations
            ),
            "live_divergence_dagger_correct_stop_supervised": bool(
                live_pre_divergence_augmentations
            ),
            "live_divergence_dagger_metrics_by_branch_id": (
                live_pre_divergence_metrics_by_branch
            ),
            "live_object_goal_dagger_augmentation_enabled": (
                live_object_goal_enabled
            ),
            "live_object_goal_dagger_sample_count": (
                live_object_goal_dagger_sample_count
            ),
            "live_object_goal_dagger_episode_count": (
                live_object_goal_dagger_episode_count
            ),
            "live_object_goal_dagger_train_split_only": True,
            "live_object_goal_dagger_exact_runtime_history": (
                live_object_goal_enabled
            ),
            "live_object_goal_dagger_score_authenticated_after_runtime": (
                live_object_goal_enabled
            ),
            "live_object_goal_dagger_correct_stop_supervised": (
                live_object_goal_enabled
            ),
            "live_object_goal_dagger_metrics": primary_live_object_goal_metrics,
            "offline_dagger_recovery_labels_only": True,
            "runtime_recovery_planner_available": False,
            "absolute_heading_targets": True,
            "absolute_world_xy_targets": True,
            "all_lap_waypoints_model_supervised": True,
            "all_between_waypoints_model_supervised": True,
            "all_object_face_and_approach_steps_model_supervised": True,
            "both_lap_winding_directions": True,
            "policy_selects_all_headings_and_waypoints_at_runtime": True,
            "offline_numeric_expert_planners_only": True,
            "expert_planners_available_at_runtime": False,
            "runtime_preprogrammed_lap_function": False,
            "environmental_text_training_only": True,
            "semantic_training_metadata": True,
            "runtime_compatible": False,
            "runtime_must_block_parent_tree": True,
            "oracle_inputs_used_for_training_only": True,
            "oracle_inputs_at_runtime": False,
            "checkpoint_contains_trace_rows": False,
            "checkpoint_contains_object_labels": False,
            "source_trace_dataset_sha256": str(source_manifest["dataset_sha256"]),
            "source_trace_manifest_sha256": _sha256(source_root / "manifest.json"),
            "source_trace_rows_sha256": _sha256(source_root / "traces.jsonl"),
            "oracle_source_sha256": oracle_hashes,
            "map_source_sha256": map_hashes,
            "effective_settings_sha256": _canonical_sha256(stable_settings),
            "traces_sha256": _sha256(traces_path),
        }
        if len(live_failure_augmentations) > 1:
            report_digests = [
                str(augmentation.report_sha256)
                for augmentation in live_failure_augmentations
            ]
            body.update(
                {
                    "live_failure_dagger_cumulative_sources": True,
                    "live_failure_dagger_source_report_sha256": report_digests,
                    "live_failure_dagger_metrics_by_report_sha256": {
                        digest: live_failure_dagger_metrics_by_report[digest]
                        for digest in report_digests
                    },
                    "live_failure_dagger_all_sources_retained": (
                        set(report_digests)
                        == set(live_failure_dagger_metrics_by_report)
                    ),
                }
            )
        if len(live_object_goal_augmentations) > 1:
            object_report_digests = [
                str(augmentation.runtime_report_sha256)
                for augmentation in live_object_goal_augmentations
            ]
            body.update(
                {
                    "live_object_goal_dagger_cumulative_sources": True,
                    "live_object_goal_dagger_runtime_report_sha256": (
                        object_report_digests
                    ),
                    "live_object_goal_dagger_metrics_by_report_sha256": {
                        digest: live_object_goal_dagger_metrics_by_report[digest]
                        for digest in object_report_digests
                    },
                    "live_object_goal_dagger_all_sources_retained": (
                        set(object_report_digests)
                        == set(live_object_goal_dagger_metrics_by_report)
                    ),
                }
            )
        manifest = {**body, "dataset_sha256": _canonical_sha256(body)}
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, root)
        return manifest
    finally:
        if temporary.exists():
            for child in temporary.iterdir():
                child.unlink(missing_ok=True)
            temporary.rmdir()


def load_gemma_waypoint_trace_dataset(
    source: str | Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Strictly authenticate a training-only absolute-waypoint dataset."""

    root = _rooted(source)
    _require_training_path(root)
    manifest_path = root / "manifest.json"
    traces_path = root / "traces.jsonl"
    if (
        not root.is_dir()
        or {entry.name for entry in root.iterdir()} != {"manifest.json", "traces.jsonl"}
        or any(path.is_symlink() or not path.is_file() for path in (manifest_path, traces_path))
    ):
        raise ValueError("Waypoint dataset must contain exactly two regular files")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError("Waypoint trace manifest schema differs")
    body = {key: value for key, value in manifest.items() if key != "dataset_sha256"}
    if (
        manifest.get("dataset_sha256") != _canonical_sha256(body)
        or manifest.get("traces_sha256") != _sha256(traces_path)
        or manifest.get("scene_splits_disjoint") is not True
        or manifest.get("runtime_compatible") is not False
        or manifest.get("runtime_must_block_parent_tree") is not True
        or manifest.get("environmental_text_training_only") is not True
        or manifest.get("oracle_inputs_at_runtime") is not False
        or manifest.get("expert_planners_available_at_runtime") is not False
        or manifest.get("runtime_preprogrammed_lap_function") is not False
        or manifest.get("action_names") != list(ACTION_NAMES)
        or _SUPPORTED_HISTORY_CONTRACTS.get(
            str(manifest.get("history_parameterization"))
        )
        != manifest.get("history_feature_dim")
    ):
        raise ValueError("Waypoint trace manifest integrity or isolation differs")
    train = set(manifest.get("train_scene_ids", []))
    validation = set(manifest.get("validation_scene_ids", []))
    if not train or not validation or train & validation:
        raise ValueError("Waypoint trace scene splits are invalid")
    rows: list[dict[str, Any]] = []
    with traces_path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            row = json.loads(line)
            if not isinstance(row, dict) or set(row) != _ROW_FIELDS:
                raise ValueError(f"Waypoint trace row {index} fields differ")
            action = row.get("expert_action")
            heading = row.get("expert_heading_degrees")
            xy = row.get("expert_xy_m")
            state = np.asarray(row.get("state_features"), dtype=np.float64)
            poses = np.asarray(row.get("history_pose_xy_yaw"), dtype=np.float64)
            history_actions = row.get("history_action_codes")
            history_value = row.get("history")
            numeric_history = (
                np.empty((0, int(manifest["history_feature_dim"])), dtype=np.float64)
                if history_value == []
                else np.asarray(history_value, dtype=np.float64)
            )
            trainer_action = row.get("action")
            trainer_waypoint = row.get("waypoint_delta_robot_m")
            trainer_heading = row.get("heading_degrees")
            scene_id = row.get("scene_id")
            if scene_id not in train | validation:
                raise ValueError(f"Waypoint trace row {index} has an undeclared scene")
            expected_split = "train" if scene_id in train else "validation"
            if (
                row.get("schema") != TRACE_SCHEMA
                or row.get("sample_id") != f"w_{index:08d}"
                or row.get("split") != expected_split
                or row.get("expert_action_code") != ACTION_TO_CODE.get(str(action), -1)
                or state.shape != (ROBOT_STATE_FEATURE_DIM,)
                or not np.isfinite(state).all()
                or poses.ndim != 2
                or poses.shape[1:] != (3,)
                or not 1 <= len(poses) <= int(manifest["history_length"]) + 1
                or not np.isfinite(poses).all()
                or not isinstance(history_actions, list)
                or len(history_actions) != len(poses) - 1
                or any(value not in ACTION_TO_CODE.values() for value in history_actions)
                or numeric_history.shape
                != (len(history_actions), int(manifest["history_feature_dim"]))
                or not np.isfinite(numeric_history).all()
                or trainer_action != str(action).casefold()
                or row.get("collision_safe_target") is not True
                or row.get("environmental_text_training_only") is not True
                or row.get("oracle_available_at_runtime") is not False
                or row.get("expert_planner_available_at_runtime") is not False
            ):
                raise ValueError(f"Waypoint trace row {index} violates its contract")
            if (
                manifest["history_parameterization"] == HISTORY_PARAMETERIZATION_V2
                and len(numeric_history)
                and (
                    np.any(numeric_history[:, 12] < 0.0)
                    or np.any(numeric_history[:, 12] >= 1.0)
                    or np.any(np.abs(numeric_history[:, 13]) >= 1.0)
                    or np.any(numeric_history[:, 14] < 0.0)
                    or np.any(numeric_history[:, 14] > 1.0)
                    or np.any(numeric_history[:, 15] < 0.0)
                    or np.any(numeric_history[:, 15] >= 1.0)
                )
            ):
                raise ValueError(f"Waypoint V2 history row {index} is out of bounds")
            if action == "FACE":
                if (
                    isinstance(heading, bool)
                    or not isinstance(heading, (int, float))
                    or not math.isfinite(float(heading))
                    or not -180.0 <= float(heading) < 180.0
                    or xy is not None
                    or trainer_waypoint is not None
                    or trainer_heading != heading
                ):
                    raise ValueError(f"Waypoint FACE row {index} target is invalid")
            elif action == "MOVE_TO":
                target = np.asarray(xy, dtype=np.float64)
                relative = np.asarray(trainer_waypoint, dtype=np.float64)
                if (
                    target.shape != (2,)
                    or not np.isfinite(target).all()
                    or relative.shape != (2,)
                    or not np.isfinite(relative).all()
                    or float(np.linalg.norm(relative))
                    > float(manifest["max_waypoint_step_m"]) + 1e-6
                    or heading is not None
                    or trainer_heading is not None
                ):
                    raise ValueError(f"Waypoint MOVE_TO row {index} target is invalid")
            elif action == "STOP":
                if (
                    heading is not None
                    or xy is not None
                    or trainer_waypoint is not None
                    or trainer_heading is not None
                ):
                    raise ValueError(f"Waypoint STOP row {index} target is invalid")
            else:
                raise ValueError(f"Waypoint row {index} has an unknown action")
            rows.append(row)
    if len(rows) != manifest.get("sample_count"):
        raise ValueError("Waypoint trace sample count differs")
    return manifest, rows


__all__ = [
    "ACTION_NAMES",
    "ACTION_TO_CODE",
    "MANIFEST_SCHEMA",
    "TRACE_SCHEMA",
    "AbsoluteWaypointAction",
    "convert_v3_action_to_absolute",
    "generate_gemma_waypoint_trace_dataset",
    "load_gemma_waypoint_trace_dataset",
]
