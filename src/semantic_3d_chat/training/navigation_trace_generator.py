"""Oracle-side generation of bounded navigation supervision.

This module is training-only by design.  It opens semantic scene metadata to
construct collision-checked expert traces, and therefore refuses to write
outside a path containing a ``training`` component.  The deployable policy
loader has the inverse rule and rejects this entire tree.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch

from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.robot.collision import NumericCollisionMap
from semantic_3d_chat.robot.navigation_policy import (
    ACTION_TO_INDEX,
    normalized_argument_for_action,
)
from semantic_3d_chat.robot.planner import NumericWaypointPlanner
from semantic_3d_chat.robot.state_encoder import NumericRobotState, robot_state_vector

_SCENE_ID = re.compile(r"scene_[0-9]{6}")
_TRACE_SCHEMA: Final[str] = "semantic_3d_chat.navigation_trace_sample.v1"
_MANIFEST_SCHEMA: Final[str] = "semantic_3d_chat.navigation_trace_dataset.v1"
_FAMILIES: Final[tuple[str, ...]] = (
    "face",
    "approach",
    "stop",
    "obstacle",
    "left_right",
    "update_after_scan",
    "collision_recovery",
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


def _require_training_output(path: Path) -> None:
    if "training" not in {part.casefold() for part in path.parts}:
        raise ValueError("Oracle-derived navigation traces must live under a training tree")
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current /= component
        if current.is_symlink():
            raise ValueError("Navigation training outputs cannot contain symbolic links")


def _atomic_json(path: Path, payload: object) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def _normalize_degrees(value: float) -> float:
    return (float(value) + 180.0) % 360.0 - 180.0


def _desired_heading(start: np.ndarray, target: np.ndarray) -> float:
    delta = target[:2] - start[:2]
    if float(np.linalg.norm(delta)) <= 1e-9:
        return 0.0
    # Simulator forward is [-sin(yaw), cos(yaw)].
    return math.degrees(math.atan2(-float(delta[0]), float(delta[1])))


@dataclass(frozen=True)
class _ExpertState:
    position_xy_m: tuple[float, float]
    body_yaw_degrees: float
    collision: bool = False
    last_delta_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    velocity_xy_m: tuple[float, float] = (0.0, 0.0)
    angular_velocity_degrees: float = 0.0
    scan_coverage: float = 0.0
    stopped: bool = False

    def numeric(self) -> NumericRobotState:
        return NumericRobotState(
            position_m=(self.position_xy_m[0], self.position_xy_m[1], 0.0),
            body_yaw_degrees=self.body_yaw_degrees,
            camera_yaw_degrees=self.body_yaw_degrees,
            pitch_degrees=0.0,
            linear_velocity_xy_m=self.velocity_xy_m,
            angular_velocity_degrees=self.angular_velocity_degrees,
            collision=self.collision,
            last_movement_delta_m=self.last_delta_m,
            scan_coverage=self.scan_coverage,
            stopped=self.stopped,
        )


@dataclass(frozen=True)
class _Episode:
    family: str
    instruction: str
    initial_state: _ExpertState
    target_xy_m: tuple[float, float] | None = None
    obstacle_xy_m: tuple[float, float] | None = None


class _TraceBuilder:
    def __init__(
        self,
        *,
        scene_id: str,
        split: str,
        episode_id: str,
        family: str,
        instruction: str,
        initial_state: _ExpertState,
        collision_map: NumericCollisionMap,
        room_min: torch.Tensor,
        room_max: torch.Tensor,
        max_turn_degrees: float,
        max_move_m: float,
        scan_coverage_increment: float,
    ) -> None:
        self.scene_id = scene_id
        self.split = split
        self.episode_id = episode_id
        self.family = family
        self.instruction = instruction
        self.state = initial_state
        self.collision_map = collision_map
        self.room_min = room_min
        self.room_max = room_max
        self.max_turn = float(max_turn_degrees)
        self.max_move = float(max_move_m)
        self.scan_coverage_increment = float(scan_coverage_increment)
        if not 0.0 < self.scan_coverage_increment <= 1.0:
            raise ValueError("Scan-coverage increment must be in (0, 1]")
        self.rows: list[dict[str, Any]] = []

    def _record(self, action_name: str, argument: float) -> None:
        vector = robot_state_vector(
            self.state.numeric(), self.room_min, self.room_max
        )
        normalized = normalized_argument_for_action(
            action_name,
            argument,
            max_turn_degrees=self.max_turn,
            max_move_m=self.max_move,
        )
        row = {
            "schema": _TRACE_SCHEMA,
            "sample_id": "pending",
            "episode_id": self.episode_id,
            "scene_id": self.scene_id,
            "split": self.split,
            "family": self.family,
            "instruction": self.instruction,
            "step_index": len(self.rows),
            "state_features": [float(value) for value in vector.tolist()],
            "action_index": ACTION_TO_INDEX[action_name],
            "action_name": action_name,
            "argument_target_normalized": float(normalized),
            "collision_safe_target": True,
            "oracle_available_at_runtime": False,
        }
        self.rows.append(row)

    def turn_to(self, target_yaw: float) -> None:
        delta = _normalize_degrees(target_yaw - self.state.body_yaw_degrees)
        while abs(delta) > 1e-7:
            amount = math.copysign(min(abs(delta), self.max_turn), delta)
            self._record("turn", amount)
            self.state = replace(
                self.state,
                body_yaw_degrees=_normalize_degrees(
                    self.state.body_yaw_degrees + amount
                ),
                collision=False,
                velocity_xy_m=(0.0, 0.0),
                angular_velocity_degrees=amount,
                last_delta_m=(0.0, 0.0, 0.0),
            )
            delta = _normalize_degrees(target_yaw - self.state.body_yaw_degrees)

    def move_to(self, target_xy: np.ndarray) -> None:
        start = np.asarray(self.state.position_xy_m, dtype=np.float64)
        delta = np.asarray(target_xy, dtype=np.float64) - start
        distance = float(np.linalg.norm(delta))
        if distance <= 1e-8:
            return
        if distance > self.max_move + 1e-7:
            raise RuntimeError("Expert planner emitted an over-limit movement step")
        heading = _desired_heading(start, np.asarray(target_xy, dtype=np.float64))
        self.turn_to(heading)
        check = self.collision_map.segment_check(start, target_xy)
        if check.collision:
            raise RuntimeError("Expert trace attempted a collision-bearing movement")
        self._record("move_forward", distance)
        self.state = replace(
            self.state,
            position_xy_m=(float(target_xy[0]), float(target_xy[1])),
            collision=False,
            velocity_xy_m=(float(delta[0]), float(delta[1])),
            angular_velocity_degrees=0.0,
            last_delta_m=(float(delta[0]), float(delta[1]), 0.0),
        )

    def scan(self) -> None:
        self._record("scan", 0.0)
        self.state = replace(
            self.state,
            collision=False,
            velocity_xy_m=(0.0, 0.0),
            angular_velocity_degrees=0.0,
            last_delta_m=(0.0, 0.0, 0.0),
            scan_coverage=min(
                1.0, self.state.scan_coverage + self.scan_coverage_increment
            ),
        )

    def move_backward(self, distance: float) -> bool:
        yaw = math.radians(self.state.body_yaw_degrees)
        direction = np.asarray([-math.sin(yaw), math.cos(yaw)], dtype=np.float64)
        start = np.asarray(self.state.position_xy_m, dtype=np.float64)
        target = start - float(distance) * direction
        check = self.collision_map.segment_check(start, target)
        if check.collision:
            return False
        self._record("move_backward", float(distance))
        delta = target - start
        self.state = replace(
            self.state,
            position_xy_m=(float(target[0]), float(target[1])),
            collision=False,
            velocity_xy_m=(float(delta[0]), float(delta[1])),
            angular_velocity_degrees=0.0,
            last_delta_m=(float(delta[0]), float(delta[1]), 0.0),
        )
        return True

    def stop(self) -> None:
        self._record("stop", 0.0)
        self.state = replace(
            self.state,
            velocity_xy_m=(0.0, 0.0),
            angular_velocity_degrees=0.0,
            last_delta_m=(0.0, 0.0, 0.0),
            stopped=True,
        )


def _load_oracle(path: Path, scene_id: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("scene_id") != scene_id:
        raise ValueError("Navigation trace oracle differs from the requested scene")
    instances = value.get("instances")
    if not isinstance(instances, list):
        raise TypeError("Navigation trace oracle has no instance list")
    return value


def _instance_centers(oracle: dict[str, Any]) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for value in oracle["instances"]:
        if not isinstance(value, dict):
            continue
        category = value.get("category")
        center = np.asarray(value.get("expected_center_xyz_m"), dtype=np.float64)
        if (
            isinstance(category, str)
            and center.shape == (3,)
            and np.isfinite(center).all()
            and category not in result
        ):
            result[category] = center
    return result


def _safe_starts(
    collision_map: NumericCollisionMap,
    *,
    count: int,
    seed: int,
) -> list[np.ndarray]:
    candidates = [np.zeros(2, dtype=np.float64)]
    for y in np.arange(-1.5, 1.51, 0.5):
        for x in np.arange(-2.0, 2.01, 0.5):
            candidates.append(np.asarray([x, y], dtype=np.float64))
    safe = [
        point
        for point in candidates
        if not collision_map.point_check(point).collision
        and collision_map.point_check(point).clearance_m >= 0.05
    ]
    if not safe:
        raise RuntimeError("No collision-free trace-generation start pose is available")
    center = safe.pop(0) if np.allclose(safe[0], 0.0) else None
    rng = np.random.default_rng(seed)
    rng.shuffle(safe)
    selected = ([] if center is None else [center]) + safe
    return selected[:count]


def _episode_rows(
    episode: _Episode,
    *,
    scene_id: str,
    split: str,
    episode_id: str,
    collision_map: NumericCollisionMap,
    room_size_m: tuple[float, float, float],
    max_turn_degrees: float,
    max_move_m: float,
    scan_coverage_increment: float,
    planner_settings: dict[str, Any],
) -> list[dict[str, Any]]:
    room = torch.tensor(room_size_m, dtype=torch.float32)
    builder = _TraceBuilder(
        scene_id=scene_id,
        split=split,
        episode_id=episode_id,
        family=episode.family,
        instruction=episode.instruction,
        initial_state=episode.initial_state,
        collision_map=collision_map,
        room_min=torch.tensor(
            [-room[0] / 2.0, -room[1] / 2.0, 0.0], dtype=torch.float32
        ),
        room_max=torch.tensor(
            [room[0] / 2.0, room[1] / 2.0, room[2]], dtype=torch.float32
        ),
        max_turn_degrees=max_turn_degrees,
        max_move_m=max_move_m,
        scan_coverage_increment=scan_coverage_increment,
    )
    if episode.family in {"face", "left_right"}:
        assert episode.target_xy_m is not None
        builder.turn_to(
            _desired_heading(
                np.asarray(builder.state.position_xy_m),
                np.asarray(episode.target_xy_m),
            )
        )
        builder.stop()
    elif episode.family in {"approach", "obstacle", "update_after_scan"}:
        assert episode.target_xy_m is not None
        if episode.family == "update_after_scan":
            builder.scan()
        planner = NumericWaypointPlanner(
            collision_map,
            max_waypoint_step_m=max_move_m,
            **planner_settings,
        )
        plan = planner.plan(builder.state.position_xy_m, episode.target_xy_m)
        for waypoint in plan.waypoints_xy_m:
            builder.move_to(np.asarray(waypoint, dtype=np.float64))
        builder.stop()
    elif episode.family == "stop":
        if episode.instruction.startswith("Move backward"):
            if not builder.move_backward(0.25):
                return []
            builder.stop()
            return builder.rows
        yaw = builder.state.body_yaw_degrees
        start = np.asarray(builder.state.position_xy_m, dtype=np.float64)
        direction = np.asarray(
            [-math.sin(math.radians(yaw)), math.cos(math.radians(yaw))]
        )
        target = start + 0.25 * direction
        if collision_map.segment_check(start, target).collision:
            return []
        builder.move_to(target)
        builder.stop()
    elif episode.family == "collision_recovery":
        builder.stop()
    else:
        raise AssertionError(f"Unhandled navigation training family: {episode.family}")
    return builder.rows


def _episodes_for_pose(
    centers: dict[str, np.ndarray],
    state: _ExpertState,
) -> list[_Episode]:
    required = {"chair", "table", "bowl", "floor lamp"}
    if not required <= set(centers):
        return []
    return [
        _Episode(
            "face",
            "Face the chair, then stop.",
            state,
            tuple(float(value) for value in centers["chair"][:2]),
        ),
        _Episode(
            "approach",
            "Move closer to the table, then stop.",
            state,
            tuple(float(value) for value in centers["table"][:2]),
        ),
        _Episode("stop", "Move forward 0.25 meters, then stop.", state),
        _Episode("stop", "Move backward 0.25 meters, then stop.", state),
        _Episode(
            "obstacle",
            "Go around the chair and stop beside the bowl.",
            state,
            tuple(float(value) for value in centers["bowl"][:2]),
            tuple(float(value) for value in centers["chair"][:2]),
        ),
        _Episode(
            "left_right",
            "Turn toward the bowl using the shorter direction, then stop.",
            state,
            tuple(float(value) for value in centers["bowl"][:2]),
        ),
        _Episode(
            "update_after_scan",
            "Scan the room, then move closer to the floor lamp and stop.",
            state,
            tuple(float(value) for value in centers["floor lamp"][:2]),
        ),
        _Episode(
            "collision_recovery",
            "Stop immediately because movement was blocked.",
            replace(state, collision=True),
        ),
    ]


def generate_navigation_trace_dataset(
    config: dict[str, Any],
    destination: str | Path,
) -> dict[str, Any]:
    """Generate scene-disjoint, collision-checked expert action traces."""

    settings = config.get("navigation_policy")
    if not isinstance(settings, dict):
        raise TypeError("Config has no navigation_policy mapping")
    root = _rooted(destination)
    _require_training_output(root)
    if root.exists():
        raise FileExistsError(f"Navigation trace dataset already exists: {root}")
    train_scenes = settings.get("train_scene_ids")
    validation_scenes = settings.get("validation_scene_ids")
    if not isinstance(train_scenes, list) or not isinstance(validation_scenes, list):
        raise TypeError("Navigation trace scene splits must be lists")
    if not train_scenes or not validation_scenes:
        raise ValueError("Navigation trace train and validation splits cannot be empty")
    if set(train_scenes) & set(validation_scenes):
        raise ValueError("Navigation trace scene splits must be disjoint")
    all_scenes = [*train_scenes, *validation_scenes]
    if any(not isinstance(scene, str) or _SCENE_ID.fullmatch(scene) is None for scene in all_scenes):
        raise ValueError("Navigation trace scene IDs must be opaque")
    if len(set(all_scenes)) != len(all_scenes):
        raise ValueError("Navigation trace scene IDs must be unique")

    oracle_root = _rooted(str(settings["oracle_root"]))
    map_root = _rooted(str(settings["map_root"]))
    prefix_root = _rooted(str(settings["prefix_cache_root"]))
    prefix_manifest_path = prefix_root / "manifest.json"
    prefix_manifest = json.loads(prefix_manifest_path.read_text(encoding="utf-8"))
    prefix_scenes = prefix_manifest.get("scenes") if isinstance(prefix_manifest, dict) else None
    if not isinstance(prefix_scenes, dict):
        raise TypeError("Question-independent prefix cache manifest is invalid")
    room_size = tuple(float(value) for value in config["scene"]["room_size_m"])
    robot = config["robot"]
    max_turn = float(robot["max_turn_degrees"])
    max_move = float(robot["max_move_m"])
    scan_coverage_increment = float(
        settings.get("simulated_scan_coverage_increment", 0.10)
    )
    if not math.isfinite(scan_coverage_increment) or not (
        0.0 < scan_coverage_increment <= 1.0
    ):
        raise ValueError(
            "navigation_policy.simulated_scan_coverage_increment must be in (0, 1]"
        )
    start_pose_count = int(settings.get("start_pose_count", 1))
    yaws = settings.get("initial_yaw_degrees", [0.0, -90.0, 90.0, 180.0])
    if start_pose_count < 1 or not isinstance(yaws, list) or not yaws:
        raise ValueError("Navigation trace pose augmentation is invalid")
    yaws = [float(value) for value in yaws]
    if not all(math.isfinite(value) for value in yaws):
        raise ValueError("Navigation trace yaw values must be finite")
    planner_settings = {
        "grid_resolution_m": float(settings.get("planner_grid_resolution_m", 0.15)),
        "standoff_m": float(settings.get("planner_standoff_m", 0.60)),
        "standoff_tolerance_m": float(
            settings.get("planner_standoff_tolerance_m", 0.20)
        ),
        "angular_samples": int(settings.get("planner_angular_samples", 48)),
    }

    root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{root.name}.", dir=root.parent))
    rows: list[dict[str, Any]] = []
    oracle_hashes: dict[str, str] = {}
    map_hashes: dict[str, str] = {}
    prefix_hashes: dict[str, str] = {}
    family_counts = {name: 0 for name in _FAMILIES}
    episode_count = 0
    try:
        for scene_index, scene_id in enumerate(all_scenes):
            split = "train" if scene_id in train_scenes else "validation"
            oracle_path = oracle_root / scene_id / "oracle.json"
            map_path = map_root / scene_id / "voxel_map.npz"
            prefix_entry = prefix_scenes.get(scene_id)
            if not oracle_path.is_file() or not map_path.is_file() or not isinstance(
                prefix_entry, dict
            ):
                raise FileNotFoundError(
                    f"Navigation trace source artifact is unavailable for {scene_id}"
                )
            prefix_file = prefix_root / str(prefix_entry.get("filename"))
            if not prefix_file.is_file() or _sha256(prefix_file) != prefix_entry.get(
                "file_sha256"
            ):
                raise ValueError(f"Question-independent prefix cache changed for {scene_id}")
            oracle = _load_oracle(oracle_path, scene_id)
            centers = _instance_centers(oracle)
            collision_map = NumericCollisionMap.from_voxel_map(
                map_path,
                room_size_m=room_size,
                robot_radius_m=float(robot["radius_m"]),
                collision_z_min_m=float(robot.get("collision_z_min_m", 0.12)),
                collision_z_max_m=float(robot.get("collision_z_max_m", 1.80)),
                surface_padding_m=float(robot.get("surface_padding_m", 0.035)),
            )
            starts = _safe_starts(
                collision_map,
                count=start_pose_count,
                seed=int(config["seed"]) + scene_index,
            )
            for pose_index, start in enumerate(starts):
                for yaw_index, yaw in enumerate(yaws):
                    initial = _ExpertState(
                        position_xy_m=(float(start[0]), float(start[1])),
                        body_yaw_degrees=_normalize_degrees(yaw),
                    )
                    for family_index, episode in enumerate(
                        _episodes_for_pose(centers, initial)
                    ):
                        episode_id = (
                            f"e_{scene_id[6:]}_{pose_index:02d}_{yaw_index:02d}_"
                            f"{family_index:02d}"
                        )
                        generated = _episode_rows(
                            episode,
                            scene_id=scene_id,
                            split=split,
                            episode_id=episode_id,
                            collision_map=collision_map,
                            room_size_m=room_size,
                            max_turn_degrees=max_turn,
                            max_move_m=max_move,
                            scan_coverage_increment=scan_coverage_increment,
                            planner_settings=planner_settings,
                        )
                        if not generated:
                            continue
                        episode_count += 1
                        family_counts[episode.family] += 1
                        rows.extend(generated)
            oracle_hashes[scene_id] = _sha256(oracle_path)
            map_hashes[scene_id] = _sha256(map_path)
            prefix_hashes[scene_id] = str(prefix_entry["prefix_sha256"])

        if not rows or any(family_counts[name] < 1 for name in _FAMILIES):
            raise RuntimeError("Navigation trace generator did not cover every family")
        observed_scenes = {str(row["scene_id"]) for row in rows}
        if observed_scenes != set(all_scenes):
            missing = sorted(set(all_scenes) - observed_scenes)
            raise RuntimeError(
                "Navigation trace generator produced no samples for declared scenes: "
                f"{missing}"
            )
        for index, row in enumerate(rows):
            row["sample_id"] = f"n_{index:08d}"
        traces_path = temporary / "traces.jsonl"
        with traces_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        manifest: dict[str, Any] = {
            "schema": _MANIFEST_SCHEMA,
            "semantic_training_metadata": True,
            "runtime_compatible": False,
            "runtime_must_block_parent_tree": True,
            "oracle_inputs_used_for_training_only": True,
            "oracle_inputs_at_runtime": False,
            "scene_splits_disjoint": True,
            "train_scene_ids": train_scenes,
            "validation_scene_ids": validation_scenes,
            "train_scene_count": len(train_scenes),
            "validation_scene_count": len(validation_scenes),
            "sample_count": len(rows),
            "episode_count": episode_count,
            "family_episode_counts": family_counts,
            "action_names": list(ACTION_TO_INDEX),
            "state_feature_dim": 18,
            "complete_scene_prefixes": True,
            "question_dependent_scene_retrieval": False,
            "every_scene_token_processed_by_controller": True,
            "collision_checked_movement_targets": True,
            "bounded_action_targets": True,
            "simulated_scan_coverage_increment": scan_coverage_increment,
            "stop_targets_included": family_counts["collision_recovery"] > 0,
            "oracle_source_sha256": oracle_hashes,
            "map_source_sha256": map_hashes,
            "scene_prefix_sha256": prefix_hashes,
            "prefix_cache_manifest_sha256": _sha256(prefix_manifest_path),
            "traces_sha256": _sha256(traces_path),
            "config_sha256": _canonical_sha256(
                {key: value for key, value in config.items() if not key.startswith("_")}
            ),
        }
        manifest["dataset_sha256"] = _canonical_sha256(
            {key: value for key, value in manifest.items() if key != "dataset_sha256"}
        )
        _atomic_json(temporary / "manifest.json", manifest)
        os.replace(temporary, root)
        return {**manifest, "dataset": str(root)}
    finally:
        if temporary.exists():
            for child in temporary.iterdir():
                child.unlink(missing_ok=True)
            temporary.rmdir()


def load_navigation_trace_dataset(
    dataset: str | Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Strict training/evaluation loader; never import this from runtime code."""

    root = _rooted(dataset)
    _require_training_output(root)
    manifest_path = root / "manifest.json"
    traces_path = root / "traces.jsonl"
    if not root.is_dir() or {entry.name for entry in root.iterdir()} != {
        "manifest.json",
        "traces.jsonl",
    }:
        raise ValueError("Navigation trace dataset must contain exactly two files")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema") != _MANIFEST_SCHEMA:
        raise ValueError("Navigation trace dataset manifest is invalid")
    if (
        manifest.get("runtime_compatible") is not False
        or manifest.get("oracle_inputs_at_runtime") is not False
        or manifest.get("scene_splits_disjoint") is not True
        or manifest.get("traces_sha256") != _sha256(traces_path)
    ):
        raise ValueError("Navigation trace dataset contract changed")
    claimed = manifest.get("dataset_sha256")
    observed = _canonical_sha256(
        {key: value for key, value in manifest.items() if key != "dataset_sha256"}
    )
    if claimed != observed:
        raise ValueError("Navigation trace dataset root hash mismatch")
    rows: list[dict[str, Any]] = []
    with traces_path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            value = json.loads(line)
            if not isinstance(value, dict) or value.get("schema") != _TRACE_SCHEMA:
                raise ValueError(f"Navigation trace sample {index} is invalid")
            if value.get("sample_id") != f"n_{index:08d}":
                raise ValueError("Navigation trace sample ordering changed")
            action_name = value.get("action_name")
            action_index = value.get("action_index")
            state = value.get("state_features")
            argument = value.get("argument_target_normalized")
            if (
                action_name not in ACTION_TO_INDEX
                or action_index != ACTION_TO_INDEX[action_name]
                or not isinstance(state, list)
                or len(state) != 18
                or not np.isfinite(np.asarray(state, dtype=np.float64)).all()
                or isinstance(argument, bool)
                or not isinstance(argument, (int, float))
                or not math.isfinite(float(argument))
                or not -1.0 <= float(argument) <= 1.0
                or value.get("collision_safe_target") is not True
                or value.get("oracle_available_at_runtime") is not False
            ):
                raise ValueError(f"Navigation trace sample {index} violates its contract")
            rows.append(value)
    if len(rows) != manifest.get("sample_count"):
        raise ValueError("Navigation trace sample count changed")
    return manifest, rows


__all__ = [
    "generate_navigation_trace_dataset",
    "load_navigation_trace_dataset",
]
