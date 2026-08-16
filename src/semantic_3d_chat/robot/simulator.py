"""Bounded embodied-camera simulator over an anonymous numerical scene map."""

from __future__ import annotations

import math
import os
import re
import tempfile
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from semantic_3d_chat.config import PROJECT_ROOT, project_path
from semantic_3d_chat.robot.collision import CollisionCheck, NumericCollisionMap
from semantic_3d_chat.robot.state_encoder import NumericRobotState

_OPAQUE_SCENE_ID = re.compile(r"scene_[0-9]{6}")


def _number(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a finite number, not a boolean")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite number") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _seed(value: Any) -> int:
    numeric = _number(value, name="seed")
    if not numeric.is_integer() or numeric < 0 or numeric > 2**32 - 1:
        raise ValueError("seed must be an integer in [0, 2^32 - 1]")
    return int(numeric)


def _normalize_degrees(angle_degrees: float) -> float:
    return (float(angle_degrees) + 180.0) % 360.0 - 180.0


def _camera_basis(yaw_degrees: float, pitch_degrees: float) -> tuple[np.ndarray, ...]:
    """Return CV camera right/down/forward axes in canonical world coordinates."""

    yaw = math.radians(yaw_degrees)
    pitch = math.radians(pitch_degrees)
    right = np.array([math.cos(yaw), math.sin(yaw), 0.0], dtype=np.float64)
    forward = np.array(
        [-math.sin(yaw) * math.cos(pitch), math.cos(yaw) * math.cos(pitch), math.sin(pitch)],
        dtype=np.float64,
    )
    down = np.cross(forward, right)
    return right, down, forward


@dataclass(frozen=True)
class NumericObservation:
    observation_id: str
    rgb: np.ndarray
    depth_m: np.ndarray
    intrinsics: np.ndarray
    camera_to_world: np.ndarray
    visible_voxel_indices: np.ndarray


class NumericMapScanner:
    """Render a deterministic point-splat RGB-D observation from the fused map.

    This direct-function precursor reobserves the existing anonymous numerical
    map.  It exercises pose-dependent visibility, RGB-D persistence, observation
    fusion, and the scene-update seam without reading scene-generation metadata.
    A later renderer/vision callback can replace it through ``map_update_hook``.
    """

    def __init__(
        self,
        map_path: str | Path,
        *,
        resolution: tuple[int, int],
        horizontal_fov_degrees: float,
        depth_min_m: float,
        depth_max_m: float,
        output_directory: str | Path,
    ) -> None:
        with np.load(Path(map_path), allow_pickle=False) as archive:
            required = {"centers_world", "mean_rgb", "observation_count"}
            missing = required - set(archive.files)
            if missing:
                raise ValueError(f"Numeric voxel map is missing fields: {sorted(missing)}")
            self.centers_world = archive["centers_world"].astype(np.float32)
            self.mean_rgb = archive["mean_rgb"].astype(np.float32)
            self.observation_count = archive["observation_count"].astype(np.int64)
        self._initial_observation_count = self.observation_count.copy()
        count = len(self.centers_world)
        if (
            self.centers_world.shape != (count, 3)
            or self.mean_rgb.shape != (count, 3)
            or self.observation_count.shape != (count,)
            or count == 0
        ):
            raise ValueError("Invalid numerical map arrays for scanning")
        if not np.isfinite(self.centers_world).all() or not np.isfinite(self.mean_rgb).all():
            raise ValueError("Numerical map contains NaN or infinity")
        width, height = (int(value) for value in resolution)
        if width < 2 or height < 2:
            raise ValueError("Scan resolution must be at least 2x2")
        if not 1.0 <= horizontal_fov_degrees < 179.0:
            raise ValueError("horizontal_fov_degrees must be in [1, 179)")
        if depth_min_m <= 0 or depth_max_m <= depth_min_m:
            raise ValueError("Invalid scan depth bounds")
        self.width = width
        self.height = height
        self.horizontal_fov_degrees = float(horizontal_fov_degrees)
        self.depth_min_m = float(depth_min_m)
        self.depth_max_m = float(depth_max_m)
        self.output_directory = Path(output_directory)
        focal = width / (2.0 * math.tan(math.radians(horizontal_fov_degrees) / 2.0))
        self.intrinsics = np.array(
            [[focal, 0.0, (width - 1) / 2.0], [0.0, focal, (height - 1) / 2.0], [0, 0, 1]],
            dtype=np.float64,
        )

    def capture(
        self,
        *,
        observation_index: int,
        camera_position_m: tuple[float, float, float],
        yaw_degrees: float,
        pitch_degrees: float,
    ) -> NumericObservation:
        origin = np.asarray(camera_position_m, dtype=np.float64)
        if origin.shape != (3,) or not np.isfinite(origin).all():
            raise ValueError("camera_position_m must be a finite three-vector")
        right, down, forward = _camera_basis(yaw_degrees, pitch_degrees)
        rotation = np.stack((right, down, forward), axis=1)
        relative = self.centers_world.astype(np.float64) - origin
        camera_points = relative @ rotation
        z = camera_points[:, 2]
        valid = (z >= self.depth_min_m) & (z <= self.depth_max_m)
        x_projected = self.intrinsics[0, 0] * camera_points[:, 0] / np.maximum(z, 1e-12)
        y_projected = self.intrinsics[1, 1] * camera_points[:, 1] / np.maximum(z, 1e-12)
        u = np.rint(x_projected + self.intrinsics[0, 2]).astype(np.int64)
        v = np.rint(y_projected + self.intrinsics[1, 2]).astype(np.int64)
        valid &= (u >= 0) & (u < self.width) & (v >= 0) & (v < self.height)
        candidate_indices = np.flatnonzero(valid)
        rgb = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        depth = np.zeros((self.height, self.width), dtype=np.float32)
        if len(candidate_indices):
            flat_pixels = v[candidate_indices] * self.width + u[candidate_indices]
            order = np.lexsort((candidate_indices, z[candidate_indices], flat_pixels))
            ordered_pixels = flat_pixels[order]
            first = np.concatenate(([True], ordered_pixels[1:] != ordered_pixels[:-1]))
            visible_indices = candidate_indices[order[first]]
            rgb[v[visible_indices], u[visible_indices]] = np.clip(
                np.rint(self.mean_rgb[visible_indices]), 0, 255
            ).astype(np.uint8)
            depth[v[visible_indices], u[visible_indices]] = z[visible_indices].astype(np.float32)
        else:
            visible_indices = np.empty(0, dtype=np.int64)

        camera_to_world = np.eye(4, dtype=np.float64)
        camera_to_world[:3, :3] = rotation
        camera_to_world[:3, 3] = origin
        observation_id = f"o_{observation_index:06d}"
        observation = NumericObservation(
            observation_id=observation_id,
            rgb=rgb,
            depth_m=depth,
            intrinsics=self.intrinsics.copy(),
            camera_to_world=camera_to_world,
            visible_voxel_indices=visible_indices.astype(np.int64, copy=False),
        )
        self._save(observation)
        return observation

    def _save(self, observation: NumericObservation) -> None:
        self.output_directory.mkdir(parents=True, exist_ok=True)
        destination = self.output_directory / f"{observation.observation_id}.npz"
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=self.output_directory
        )
        try:
            with os.fdopen(descriptor, "wb") as handle:
                np.savez_compressed(
                    handle,
                    rgb=observation.rgb,
                    depth_m=observation.depth_m,
                    intrinsics=observation.intrinsics,
                    camera_to_world=observation.camera_to_world,
                    visible_voxel_indices=observation.visible_voxel_indices,
                )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, destination)
        finally:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass

    def integrate(self, observation: NumericObservation, observed_mask: np.ndarray) -> int:
        indices = observation.visible_voxel_indices
        if len(indices):
            np.add.at(self.observation_count, indices, 1)
            observed_mask[indices] = True
        return len(indices)

    def reset_episode(self) -> None:
        """Restore scanner-local numeric coverage for a deterministic reset."""

        self.observation_count = self._initial_observation_count.copy()

    def snapshot_episode_state(self) -> np.ndarray:
        return self.observation_count.copy()

    def restore_episode_state(self, state: np.ndarray) -> None:
        value = np.asarray(state, dtype=np.int64)
        if value.shape != self.observation_count.shape:
            raise ValueError("Scanner episode snapshot shape changed")
        self.observation_count = value.copy()


@dataclass
class RobotState:
    scene_id: str
    seed: int
    position_xy_m: np.ndarray
    body_yaw_degrees: float = 0.0
    camera_yaw_offset_degrees: float = 0.0
    pitch_degrees: float = 0.0
    linear_velocity_xy_m: np.ndarray | None = None
    angular_velocity_degrees: float = 0.0
    collision: bool = False
    last_movement_delta_m: np.ndarray | None = None
    scan_coverage: float = 0.0
    scan_count: int = 0
    scene_version: int = 0
    map_sha256: str | None = None
    action_count: int = 0
    stopped: bool = False

    def __post_init__(self) -> None:
        if self.linear_velocity_xy_m is None:
            self.linear_velocity_xy_m = np.zeros(2, dtype=np.float64)
        if self.last_movement_delta_m is None:
            self.last_movement_delta_m = np.zeros(3, dtype=np.float64)

    @property
    def camera_yaw_degrees(self) -> float:
        return _normalize_degrees(self.body_yaw_degrees + self.camera_yaw_offset_degrees)


@dataclass(frozen=True)
class PreparedSceneReset:
    """Side-effect-free numerical scene state ready for one atomic swap."""

    scene_id: str
    seed: int
    collision_map: NumericCollisionMap
    scanner: NumericMapScanner
    observed_mask: np.ndarray
    position_xy_m: np.ndarray
    body_yaw_degrees: float


MapUpdateHook = Callable[[NumericObservation, int], Mapping[str, Any] | None]


class EmbodiedCameraSimulator:
    """Stateful, collision-safe action API with numerical-only observations."""

    def __init__(
        self,
        config: dict[str, Any],
        scene_id: str,
        *,
        seed: int | None = None,
        map_update_hook: MapUpdateHook | None = None,
    ) -> None:
        self.config = config
        self.settings = dict(config["robot"])
        self.data_root = (PROJECT_ROOT / str(config["paths"]["data_root"])).resolve()
        self.map_update_hook = map_update_hook
        self.history: deque[dict[str, Any]] = deque(maxlen=int(self.settings.get("history_length", 64)))
        self.collision_map: NumericCollisionMap
        self.scanner: NumericMapScanner
        self.observed_mask: np.ndarray
        self.state: RobotState
        self.reset_scene(scene_id, int(config["seed"] if seed is None else seed))

    @property
    def robot_radius_m(self) -> float:
        return float(self.settings["radius_m"])

    @property
    def camera_height_m(self) -> float:
        return float(self.settings["camera_height_m"])

    def _map_path(self, scene_id: str) -> Path:
        if not _OPAQUE_SCENE_ID.fullmatch(scene_id):
            raise ValueError("scene_id must match scene_ followed by six digits")
        path = project_path(self.config, "maps", scene_id, "voxel_map.npz").resolve()
        expected_parent = project_path(self.config, "maps", scene_id).resolve()
        if path.parent != expected_parent or not path.is_file():
            raise FileNotFoundError(f"Numeric map is unavailable for {scene_id}")
        return path

    def _build_scene_runtime(
        self, scene_id: str
    ) -> tuple[NumericCollisionMap, NumericMapScanner, np.ndarray]:
        """Construct a scene runtime without mutating the active episode.

        Keeping construction side-effect free makes ``reset_scene`` atomic: a
        missing or malformed replacement map cannot leave the old robot state
        paired with a partially replaced collision map or scanner.
        """

        map_path = self._map_path(scene_id)
        room_size = self.config["scene"]["room_size_m"]
        collision_map = NumericCollisionMap.from_voxel_map(
            map_path,
            room_size_m=room_size,
            robot_radius_m=self.robot_radius_m,
            collision_z_min_m=float(self.settings.get("collision_z_min_m", 0.12)),
            collision_z_max_m=float(self.settings.get("collision_z_max_m", 1.80)),
            surface_padding_m=float(self.settings.get("surface_padding_m", 0.035)),
        )
        resolution = self.settings.get("scan_resolution", self.config["render"]["resolution"])
        scanner = NumericMapScanner(
            map_path,
            resolution=(int(resolution[0]), int(resolution[1])),
            horizontal_fov_degrees=float(
                self.settings.get(
                    "scan_horizontal_fov_degrees", self.config["render"]["horizontal_fov_degrees"]
                )
            ),
            depth_min_m=float(self.settings.get("scan_depth_min_m", 0.10)),
            depth_max_m=float(self.settings.get("scan_depth_max_m", 10.0)),
            output_directory=self.data_root / "robot" / scene_id / "scans",
        )
        observed_mask = np.zeros(len(scanner.centers_world), dtype=bool)
        return collision_map, scanner, observed_mask

    def _safe_initial_xy(self, seed: int, collision_map: NumericCollisionMap) -> np.ndarray:
        configured = np.asarray(self.settings.get("initial_position_xy_m", [0.0, 0.0]), dtype=float)
        if configured.shape != (2,) or not np.isfinite(configured).all():
            raise ValueError("initial_position_xy_m must contain two finite values")
        candidates = [configured]
        rng = np.random.default_rng(seed)
        angle_offset = float(rng.uniform(-math.pi, math.pi))
        for radius in np.arange(0.25, 1.76, 0.25):
            for angle in angle_offset + np.arange(16) * (2.0 * math.pi / 16.0):
                candidates.append(configured + radius * np.array([math.cos(angle), math.sin(angle)]))
        for candidate in candidates:
            check = collision_map.point_check(candidate)
            if not check.collision and check.clearance_m >= 0.01:
                return candidate.astype(np.float64)
        raise RuntimeError("No collision-free numerical start pose was found")

    def numeric_state(self) -> NumericRobotState:
        movement = self.state.last_movement_delta_m
        velocity = self.state.linear_velocity_xy_m
        assert movement is not None and velocity is not None
        return NumericRobotState(
            position_m=(
                float(self.state.position_xy_m[0]),
                float(self.state.position_xy_m[1]),
                0.0,
            ),
            body_yaw_degrees=float(self.state.body_yaw_degrees),
            camera_yaw_degrees=float(self.state.camera_yaw_degrees),
            pitch_degrees=float(self.state.pitch_degrees),
            linear_velocity_xy_m=(float(velocity[0]), float(velocity[1])),
            angular_velocity_degrees=float(self.state.angular_velocity_degrees),
            collision=bool(self.state.collision),
            last_movement_delta_m=tuple(float(value) for value in movement),
            scan_coverage=float(self.state.scan_coverage),
            stopped=bool(self.state.stopped),
        )

    def _result(
        self,
        success: bool,
        *,
        error_code: str | None = None,
        distance_moved: float = 0.0,
        turn_degrees: float = 0.0,
        visible_voxels: int = 0,
        valid_depth_pixels: int = 0,
        observation_id: str | None = None,
        clearance_m: float | None = None,
        record: bool = True,
    ) -> dict[str, Any]:
        velocity = self.state.linear_velocity_xy_m
        movement = self.state.last_movement_delta_m
        assert velocity is not None and movement is not None
        result: dict[str, Any] = {
            "success": bool(success),
            "error_code": error_code,
            "scene_id": self.state.scene_id,
            "seed": int(self.state.seed),
            "scene_version": int(self.state.scene_version),
            "position_m": [
                float(self.state.position_xy_m[0]),
                float(self.state.position_xy_m[1]),
                0.0,
            ],
            "camera_position_m": [
                float(self.state.position_xy_m[0]),
                float(self.state.position_xy_m[1]),
                self.camera_height_m,
            ],
            "body_yaw_degrees": float(self.state.body_yaw_degrees),
            "camera_yaw_degrees": float(self.state.camera_yaw_degrees),
            "pitch_degrees": float(self.state.pitch_degrees),
            "linear_velocity_xy_m": [float(velocity[0]), float(velocity[1])],
            "angular_velocity_degrees": float(self.state.angular_velocity_degrees),
            "collision": bool(self.state.collision),
            "last_movement_delta_m": [float(value) for value in movement],
            "distance_moved": float(distance_moved),
            "turn_degrees": float(turn_degrees),
            "scan_coverage": float(self.state.scan_coverage),
            "scan_count": int(self.state.scan_count),
            "visible_voxels": int(visible_voxels),
            "valid_depth_pixels": int(valid_depth_pixels),
            "observation_id": observation_id,
            "clearance_m": None if clearance_m is None else float(clearance_m),
            "action_count": int(self.state.action_count),
            "stopped": bool(self.state.stopped),
        }
        if self.state.map_sha256 is not None:
            result["map_sha256"] = self.state.map_sha256
        if record:
            self.history.append(result.copy())
        return result

    def get_robot_state(self) -> dict[str, Any]:
        return self._result(True, record=False)

    def _protocol_failure(self, code: str) -> dict[str, Any]:
        self.state.action_count += 1
        self.state.linear_velocity_xy_m = np.zeros(2, dtype=np.float64)
        self.state.angular_velocity_degrees = 0.0
        return self._result(False, error_code=code)

    def protocol_error(self, code: str = "E_PROTOCOL") -> dict[str, Any]:
        """Return a state-preserving, concise protocol failure."""

        return self._protocol_failure(code)

    def look(self, yaw_delta_degrees: Any, pitch_delta_degrees: Any) -> dict[str, Any]:
        try:
            yaw_delta = _number(yaw_delta_degrees, name="yaw_delta_degrees")
            pitch_delta = _number(pitch_delta_degrees, name="pitch_delta_degrees")
        except (TypeError, ValueError):
            return self._protocol_failure("E_NUMERIC")
        if self.state.stopped:
            return self._protocol_failure("E_STOPPED")
        max_delta = float(self.settings.get("max_look_delta_degrees", 45.0))
        next_offset = self.state.camera_yaw_offset_degrees + yaw_delta
        next_pitch = self.state.pitch_degrees + pitch_delta
        if (
            abs(yaw_delta) > max_delta
            or abs(pitch_delta) > max_delta
            or abs(next_offset) > float(self.settings.get("max_camera_yaw_offset_degrees", 90.0))
            or abs(next_pitch) > float(self.settings["max_pitch_degrees"])
        ):
            return self._protocol_failure("E_LIMIT")
        self.state.camera_yaw_offset_degrees = next_offset
        self.state.pitch_degrees = next_pitch
        self.state.action_count += 1
        self.state.collision = False
        self.state.linear_velocity_xy_m = np.zeros(2, dtype=np.float64)
        self.state.angular_velocity_degrees = yaw_delta
        return self._result(True, turn_degrees=yaw_delta)

    def turn(self, angle_degrees: Any) -> dict[str, Any]:
        try:
            angle = _number(angle_degrees, name="angle_degrees")
        except (TypeError, ValueError):
            return self._protocol_failure("E_NUMERIC")
        if self.state.stopped:
            return self._protocol_failure("E_STOPPED")
        if abs(angle) > float(self.settings["max_turn_degrees"]):
            return self._protocol_failure("E_LIMIT")
        self.state.body_yaw_degrees = _normalize_degrees(self.state.body_yaw_degrees + angle)
        self.state.action_count += 1
        self.state.collision = False
        self.state.linear_velocity_xy_m = np.zeros(2, dtype=np.float64)
        self.state.angular_velocity_degrees = angle
        return self._result(True, turn_degrees=angle)

    def _attempt_move(self, target_xy_m: np.ndarray) -> dict[str, Any]:
        start = self.state.position_xy_m.copy()
        delta = target_xy_m - start
        distance = float(np.linalg.norm(delta))
        check: CollisionCheck = self.collision_map.segment_check(start, target_xy_m)
        self.state.action_count += 1
        self.state.angular_velocity_degrees = 0.0
        if check.collision:
            self.state.collision = True
            self.state.linear_velocity_xy_m = np.zeros(2, dtype=np.float64)
            self.state.last_movement_delta_m = np.zeros(3, dtype=np.float64)
            return self._result(False, error_code="E_COLLISION", clearance_m=check.clearance_m)
        self.state.position_xy_m = target_xy_m.astype(np.float64)
        self.state.linear_velocity_xy_m = delta.astype(np.float64)
        self.state.last_movement_delta_m = np.array([delta[0], delta[1], 0.0], dtype=np.float64)
        self.state.collision = False
        return self._result(True, distance_moved=distance, clearance_m=check.clearance_m)

    def move_forward(self, distance_meters: Any) -> dict[str, Any]:
        try:
            distance = _number(distance_meters, name="distance_meters")
        except (TypeError, ValueError):
            return self._protocol_failure("E_NUMERIC")
        if self.state.stopped:
            return self._protocol_failure("E_STOPPED")
        if distance < 0 or distance > float(self.settings["max_move_m"]):
            return self._protocol_failure("E_LIMIT")
        yaw = math.radians(self.state.body_yaw_degrees)
        forward_xy = np.array([-math.sin(yaw), math.cos(yaw)], dtype=np.float64)
        return self._attempt_move(self.state.position_xy_m + distance * forward_xy)

    def move_backward(self, distance_meters: Any) -> dict[str, Any]:
        try:
            distance = _number(distance_meters, name="distance_meters")
        except (TypeError, ValueError):
            return self._protocol_failure("E_NUMERIC")
        if self.state.stopped:
            return self._protocol_failure("E_STOPPED")
        if distance < 0 or distance > float(self.settings["max_move_m"]):
            return self._protocol_failure("E_LIMIT")
        yaw = math.radians(self.state.body_yaw_degrees)
        forward_xy = np.array([-math.sin(yaw), math.cos(yaw)], dtype=np.float64)
        return self._attempt_move(self.state.position_xy_m - distance * forward_xy)

    def move_to(self, x: Any, y: Any) -> dict[str, Any]:
        try:
            target = np.array([_number(x, name="x"), _number(y, name="y")], dtype=np.float64)
        except (TypeError, ValueError):
            return self._protocol_failure("E_NUMERIC")
        if self.state.stopped:
            return self._protocol_failure("E_STOPPED")
        if np.linalg.norm(target - self.state.position_xy_m) > float(
            self.settings.get("max_move_to_m", 1.0)
        ):
            return self._protocol_failure("E_LIMIT")
        return self._attempt_move(target)

    def scan(self) -> dict[str, Any]:
        if self.state.stopped:
            return self._protocol_failure("E_STOPPED")
        observation_index = self.state.scan_count + 1
        observation = self.scanner.capture(
            observation_index=observation_index,
            camera_position_m=(
                float(self.state.position_xy_m[0]),
                float(self.state.position_xy_m[1]),
                self.camera_height_m,
            ),
            yaw_degrees=self.state.camera_yaw_degrees,
            pitch_degrees=self.state.pitch_degrees,
        )
        valid_depth_pixels = int(np.count_nonzero(observation.depth_m > 0))
        # Point-splat observations carry exact source-voxel indices.  A fresh
        # renderer has no such identity channel, so do not mislabel its valid
        # depth pixels as visible map voxels.
        visible_voxels = len(observation.visible_voxel_indices)
        proposed_version = self.state.scene_version + int(bool(valid_depth_pixels))
        update_result: Mapping[str, Any] | None = None
        if valid_depth_pixels and self.map_update_hook is not None:
            try:
                update_result = self.map_update_hook(observation, proposed_version)
                if update_result is not None:
                    returned_version = update_result.get("map_version")
                    returned_hash = update_result.get("map_sha256")
                    if returned_version != proposed_version or (
                        not isinstance(returned_hash, str)
                        or re.fullmatch(r"[0-9a-f]{64}", returned_hash) is None
                    ):
                        raise ValueError("Map update hook returned an invalid receipt")
            except (OSError, RuntimeError, TypeError, ValueError):
                # The observation is a disposable sanitized artifact.  Map,
                # coverage, scan count, and scene version remain unchanged so
                # the same numeric pose can retry safely.
                self.state.action_count += 1
                self.state.linear_velocity_xy_m = np.zeros(2, dtype=np.float64)
                self.state.angular_velocity_degrees = 0.0
                self.state.collision = False
                return self._result(False, error_code="E_MAP_UPDATE")
        integrated = self.scanner.integrate(observation, self.observed_mask)
        if integrated != valid_depth_pixels:
            raise RuntimeError("Numeric observation integration count changed")
        self.state.scan_count = observation_index
        self.state.action_count += 1
        self.state.linear_velocity_xy_m = np.zeros(2, dtype=np.float64)
        self.state.angular_velocity_degrees = 0.0
        self.state.collision = False
        if valid_depth_pixels:
            self.state.scene_version = proposed_version
            if update_result is not None:
                self.state.map_sha256 = str(update_result["map_sha256"])
            directional_coverage = getattr(self.scanner, "directional_coverage", None)
            self.state.scan_coverage = (
                float(directional_coverage)
                if directional_coverage is not None
                else float(np.mean(self.observed_mask))
            )
        return self._result(
            bool(valid_depth_pixels),
            error_code=None if valid_depth_pixels else "E_EMPTY_SCAN",
            visible_voxels=visible_voxels,
            valid_depth_pixels=valid_depth_pixels,
            observation_id=observation.observation_id,
        )

    def stop(self) -> dict[str, Any]:
        self.state.action_count += 1
        self.state.stopped = True
        self.state.linear_velocity_xy_m = np.zeros(2, dtype=np.float64)
        self.state.angular_velocity_degrees = 0.0
        return self._result(True)

    def reset_scene(self, scene_id: str, seed: Any) -> dict[str, Any]:
        if not isinstance(scene_id, str) or not _OPAQUE_SCENE_ID.fullmatch(scene_id):
            if hasattr(self, "state"):
                return self._protocol_failure("E_SCENE_ID")
            raise ValueError("scene_id must match scene_ followed by six digits")
        try:
            seed_value = _seed(seed)
        except (TypeError, ValueError):
            if hasattr(self, "state"):
                return self._protocol_failure("E_NUMERIC")
            raise
        try:
            prepared = self.prepare_scene_reset(scene_id, seed_value)
        except (FileNotFoundError, ValueError, RuntimeError):
            if hasattr(self, "state"):
                return self._protocol_failure("E_SCENE_UNAVAILABLE")
            raise
        return self.commit_scene_reset(prepared)

    def prepare_scene_reset(self, scene_id: str, seed: Any) -> PreparedSceneReset:
        """Validate and build a replacement episode without mutating current state."""

        if not isinstance(scene_id, str) or not _OPAQUE_SCENE_ID.fullmatch(scene_id):
            raise ValueError("scene_id must match scene_ followed by six digits")
        seed_value = _seed(seed)
        collision_map, scanner, observed_mask = self._build_scene_runtime(scene_id)
        position = self._safe_initial_xy(seed_value, collision_map)
        body_yaw_degrees = _normalize_degrees(
            _number(
                self.settings.get("initial_body_yaw_degrees", 0.0),
                name="initial_body_yaw_degrees",
            )
        )
        return PreparedSceneReset(
            scene_id=scene_id,
            seed=seed_value,
            collision_map=collision_map,
            scanner=scanner,
            observed_mask=observed_mask,
            position_xy_m=position,
            body_yaw_degrees=body_yaw_degrees,
        )

    def commit_scene_reset(
        self,
        prepared: PreparedSceneReset,
        *,
        scanner_override: Any | None = None,
        reset_scanner: bool = True,
    ) -> dict[str, Any]:
        """Commit a prepared reset, optionally retaining a sanitized renderer."""

        if not isinstance(prepared, PreparedSceneReset):
            raise TypeError("prepared reset has an invalid type")
        scanner = prepared.scanner if scanner_override is None else scanner_override
        if reset_scanner:
            reset_episode = getattr(scanner, "reset_episode", None)
            if callable(reset_episode):
                reset_episode()
        self.collision_map = prepared.collision_map
        self.scanner = scanner
        self.observed_mask = prepared.observed_mask.copy()
        self.state = RobotState(
            scene_id=prepared.scene_id,
            seed=prepared.seed,
            position_xy_m=prepared.position_xy_m.copy(),
            body_yaw_degrees=prepared.body_yaw_degrees,
        )
        self.history.clear()
        return self._result(True)
