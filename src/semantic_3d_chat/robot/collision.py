"""Collision checks derived exclusively from numerical voxel-map geometry."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class CollisionCheck:
    collision: bool
    clearance_m: float
    collision_xy_m: tuple[float, float] | None


def _finite_xy(value: np.ndarray | tuple[float, float], *, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (2,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain two finite coordinates")
    return result


class NumericCollisionMap:
    """A conservative 2D collision field built from anonymous 3D map points.

    Points between ``collision_z_min_m`` and ``collision_z_max_m`` are treated
    as physical surfaces.  The floor and ceiling are therefore excluded while
    walls and furniture remain.  No object IDs, category names, segmentation,
    or scene-generation metadata are loaded.
    """

    def __init__(
        self,
        obstacle_points_xy_m: np.ndarray,
        *,
        room_min_xy_m: tuple[float, float],
        room_max_xy_m: tuple[float, float],
        robot_radius_m: float,
        surface_padding_m: float,
    ) -> None:
        points = np.asarray(obstacle_points_xy_m, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
            raise ValueError("obstacle_points_xy_m must be a finite [N, 2] array")
        if not len(points):
            raise ValueError("Collision geometry cannot be empty")
        self.obstacle_points_xy_m = points
        self.room_min_xy_m = _finite_xy(room_min_xy_m, name="room_min_xy_m")
        self.room_max_xy_m = _finite_xy(room_max_xy_m, name="room_max_xy_m")
        if np.any(self.room_max_xy_m <= self.room_min_xy_m):
            raise ValueError("Room maximum must exceed room minimum")
        if not np.isfinite(robot_radius_m) or robot_radius_m <= 0:
            raise ValueError("robot_radius_m must be finite and positive")
        if not np.isfinite(surface_padding_m) or surface_padding_m < 0:
            raise ValueError("surface_padding_m must be finite and nonnegative")
        self.robot_radius_m = float(robot_radius_m)
        self.surface_padding_m = float(surface_padding_m)
        self.inflated_radius_m = self.robot_radius_m + self.surface_padding_m

    @classmethod
    def from_voxel_map(
        cls,
        path: str | Path,
        *,
        room_size_m: tuple[float, float, float] | list[float],
        robot_radius_m: float,
        collision_z_min_m: float,
        collision_z_max_m: float,
        surface_padding_m: float,
    ) -> NumericCollisionMap:
        size = np.asarray(room_size_m, dtype=np.float64)
        if size.shape != (3,) or not np.isfinite(size).all() or np.any(size <= 0):
            raise ValueError("room_size_m must contain three finite positive values")
        if (
            not np.isfinite(collision_z_min_m)
            or not np.isfinite(collision_z_max_m)
            or collision_z_min_m < 0
            or collision_z_max_m <= collision_z_min_m
        ):
            raise ValueError("Invalid collision-height interval")
        with np.load(Path(path), allow_pickle=False) as archive:
            if "centers_world" not in archive.files:
                raise ValueError("Numeric voxel map has no centers_world field")
            centers = archive["centers_world"].astype(np.float32)
        if centers.ndim != 2 or centers.shape[1] != 3 or not np.isfinite(centers).all():
            raise ValueError("Invalid centers_world array")
        mask = (centers[:, 2] >= collision_z_min_m) & (centers[:, 2] <= collision_z_max_m)
        return cls(
            centers[mask, :2],
            room_min_xy_m=(-size[0] / 2.0, -size[1] / 2.0),
            room_max_xy_m=(size[0] / 2.0, size[1] / 2.0),
            robot_radius_m=robot_radius_m,
            surface_padding_m=surface_padding_m,
        )

    def _inside_room(self, point_xy_m: np.ndarray) -> bool:
        lower = self.room_min_xy_m + self.robot_radius_m
        upper = self.room_max_xy_m - self.robot_radius_m
        return bool(np.all(point_xy_m >= lower) and np.all(point_xy_m <= upper))

    def point_check(self, point_xy_m: np.ndarray | tuple[float, float]) -> CollisionCheck:
        point = _finite_xy(point_xy_m, name="point_xy_m")
        if not self._inside_room(point):
            return CollisionCheck(True, 0.0, (float(point[0]), float(point[1])))
        differences = self.obstacle_points_xy_m.astype(np.float64) - point
        distances = np.linalg.norm(differences, axis=1)
        nearest_index = int(np.argmin(distances))
        surface_distance = float(distances[nearest_index])
        clearance = max(0.0, surface_distance - self.inflated_radius_m)
        collision_point = None
        if surface_distance <= self.inflated_radius_m:
            nearest = self.obstacle_points_xy_m[nearest_index]
            collision_point = (float(nearest[0]), float(nearest[1]))
        return CollisionCheck(collision_point is not None, clearance, collision_point)

    def segment_check(
        self,
        start_xy_m: np.ndarray | tuple[float, float],
        end_xy_m: np.ndarray | tuple[float, float],
    ) -> CollisionCheck:
        """Check the complete swept segment without discretization gaps."""

        start = _finite_xy(start_xy_m, name="start_xy_m")
        end = _finite_xy(end_xy_m, name="end_xy_m")
        if not self._inside_room(end):
            return CollisionCheck(True, 0.0, (float(end[0]), float(end[1])))
        delta = end - start
        squared_length = float(np.dot(delta, delta))
        points = self.obstacle_points_xy_m.astype(np.float64)
        if squared_length <= np.finfo(np.float64).eps:
            return self.point_check(end)
        fractions = np.clip(((points - start) @ delta) / squared_length, 0.0, 1.0)
        closest_on_segment = start + fractions[:, None] * delta
        distances = np.linalg.norm(points - closest_on_segment, axis=1)
        nearest_index = int(np.argmin(distances))
        surface_distance = float(distances[nearest_index])
        clearance = max(0.0, surface_distance - self.inflated_radius_m)
        collision_point = None
        if surface_distance <= self.inflated_radius_m:
            nearest = points[nearest_index]
            collision_point = (float(nearest[0]), float(nearest[1]))
        return CollisionCheck(collision_point is not None, clearance, collision_point)
