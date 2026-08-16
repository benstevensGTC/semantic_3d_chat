"""Deterministic geometry-only waypoint planning for grounded numeric targets.

The planner deliberately knows nothing about object identity or language.  Its
only environmental inputs are a collision field and a target coordinate that a
separate continuous scene/question model may predict.  This keeps path planning
on the inference-safe side of the project boundary while providing a reliable
post-grounding route around anonymous occupied geometry.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from itertools import count, pairwise

import numpy as np

from semantic_3d_chat.robot.collision import NumericCollisionMap


def _finite_xy(value: np.ndarray | tuple[float, float], *, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (2,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain two finite coordinates")
    return result


@dataclass(frozen=True)
class NumericPathPlan:
    """A label-free sequence of collision-checked numerical waypoints."""

    target_xy_m: tuple[float, float]
    goal_xy_m: tuple[float, float]
    waypoints_xy_m: tuple[tuple[float, float], ...]
    path_length_m: float
    target_distance_m: float
    direct_path: bool
    expanded_nodes: int

    def as_dict(self) -> dict[str, object]:
        return {
            "target_xy_m": list(self.target_xy_m),
            "goal_xy_m": list(self.goal_xy_m),
            "waypoints_xy_m": [list(point) for point in self.waypoints_xy_m],
            "path_length_m": self.path_length_m,
            "target_distance_m": self.target_distance_m,
            "direct_path": self.direct_path,
            "expanded_nodes": self.expanded_nodes,
        }


class NumericWaypointPlanner:
    """Plan a bounded approach path over anonymous voxel-derived occupancy.

    The requested target coordinate is normally on an occupied surface, so the
    planner selects a collision-free goal on a configurable standoff ring.  It
    first tries exact line-of-sight candidates, then falls back to deterministic
    eight-connected A* over a conservative occupancy grid.  Every returned
    continuous segment is checked by :class:`NumericCollisionMap` before the
    plan is released.
    """

    def __init__(
        self,
        collision_map: NumericCollisionMap,
        *,
        grid_resolution_m: float = 0.15,
        standoff_m: float = 0.60,
        standoff_tolerance_m: float = 0.20,
        max_waypoint_step_m: float = 0.50,
        angular_samples: int = 48,
        max_expanded_nodes: int = 50_000,
    ) -> None:
        values = (
            grid_resolution_m,
            standoff_m,
            standoff_tolerance_m,
            max_waypoint_step_m,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Planner distances must be finite")
        if grid_resolution_m <= 0 or standoff_m <= 0 or standoff_tolerance_m < 0:
            raise ValueError("Planner resolution/standoff values are invalid")
        if max_waypoint_step_m <= 0:
            raise ValueError("max_waypoint_step_m must be positive")
        if angular_samples < 8 or max_expanded_nodes < 1:
            raise ValueError("Planner search limits are invalid")
        self.collision_map = collision_map
        self.grid_resolution_m = float(grid_resolution_m)
        self.standoff_m = float(standoff_m)
        self.standoff_tolerance_m = float(standoff_tolerance_m)
        self.max_waypoint_step_m = float(max_waypoint_step_m)
        self.angular_samples = int(angular_samples)
        self.max_expanded_nodes = int(max_expanded_nodes)

    def _approach_candidates(self, start: np.ndarray, target: np.ndarray) -> list[np.ndarray]:
        direction = start - target
        base_angle = math.atan2(float(direction[1]), float(direction[0]))
        angular_step = 2.0 * math.pi / self.angular_samples
        offsets = [0]
        for index in range(1, self.angular_samples // 2 + 1):
            offsets.extend((index, -index))
        offsets = offsets[: self.angular_samples]
        radii = [self.standoff_m]
        if self.standoff_tolerance_m:
            radii.extend(
                (
                    self.standoff_m + self.standoff_tolerance_m,
                    max(
                        self.collision_map.inflated_radius_m + self.grid_resolution_m,
                        self.standoff_m - self.standoff_tolerance_m,
                    ),
                )
            )
        candidates: list[np.ndarray] = []
        seen: set[tuple[float, float]] = set()
        for radius in radii:
            for offset in offsets:
                angle = base_angle + offset * angular_step
                candidate = target + radius * np.asarray(
                    [math.cos(angle), math.sin(angle)], dtype=np.float64
                )
                key = (round(float(candidate[0]), 9), round(float(candidate[1]), 9))
                if key in seen or self.collision_map.point_check(candidate).collision:
                    continue
                seen.add(key)
                candidates.append(candidate)
        return candidates

    def _grid(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        lower = self.collision_map.room_min_xy_m + self.collision_map.robot_radius_m
        upper = self.collision_map.room_max_xy_m - self.collision_map.robot_radius_m
        xs = np.arange(lower[0], upper[0] + self.grid_resolution_m * 0.25, self.grid_resolution_m)
        ys = np.arange(lower[1], upper[1] + self.grid_resolution_m * 0.25, self.grid_resolution_m)
        if len(xs) < 2 or len(ys) < 2:
            raise RuntimeError("Room is too small for the configured planning grid")
        free = np.ones((len(ys), len(xs)), dtype=bool)

        # Inflate by half a cell diagonal as well as the robot radius.  Then an
        # edge between neighboring free cell centers cannot slip past an
        # occupied sample between the centers.  The operation is vectorized
        # over all anonymous surface samples for each small neighboring-cell
        # offset, avoiding a dense [grid, surface] distance matrix.
        points = self.collision_map.obstacle_points_xy_m.astype(np.float64)
        nearest_x = np.rint((points[:, 0] - xs[0]) / self.grid_resolution_m).astype(np.int64)
        nearest_y = np.rint((points[:, 1] - ys[0]) / self.grid_resolution_m).astype(np.int64)
        grid_inflated_radius = (
            self.collision_map.inflated_radius_m
            + self.grid_resolution_m / math.sqrt(2.0)
        )
        radius_cells = math.ceil(grid_inflated_radius / self.grid_resolution_m) + 1
        squared_radius = grid_inflated_radius**2
        for delta_y in range(-radius_cells, radius_cells + 1):
            grid_y = nearest_y + delta_y
            valid_y = (grid_y >= 0) & (grid_y < len(ys))
            if not np.any(valid_y):
                continue
            for delta_x in range(-radius_cells, radius_cells + 1):
                grid_x = nearest_x + delta_x
                valid = valid_y & (grid_x >= 0) & (grid_x < len(xs))
                if not np.any(valid):
                    continue
                selected_x = grid_x[valid]
                selected_y = grid_y[valid]
                dx = xs[selected_x] - points[valid, 0]
                dy = ys[selected_y] - points[valid, 1]
                colliding = dx * dx + dy * dy <= squared_radius + 1e-12
                free[selected_y[colliding], selected_x[colliding]] = False
        return xs, ys, free

    def _nearest_reachable_free_cell(
        self,
        point: np.ndarray,
        xs: np.ndarray,
        ys: np.ndarray,
        free: np.ndarray,
    ) -> tuple[int, int] | None:
        distances = (xs[None, :] - point[0]) ** 2 + (ys[:, None] - point[1]) ** 2
        distances = np.where(free, distances, np.inf)
        for flat in np.argsort(distances, axis=None):
            if not math.isfinite(float(distances.flat[flat])):
                break
            y_index, x_index = np.unravel_index(flat, free.shape)
            center = np.asarray([xs[x_index], ys[y_index]], dtype=np.float64)
            if not self.collision_map.segment_check(point, center).collision:
                return int(y_index), int(x_index)
        return None

    def _a_star(
        self,
        start: np.ndarray,
        candidates: list[np.ndarray],
    ) -> tuple[list[np.ndarray], np.ndarray, int]:
        xs, ys, free = self._grid()
        start_cell = self._nearest_reachable_free_cell(start, xs, ys, free)
        if start_cell is None:
            raise RuntimeError("No free planning cell is available near the robot")
        goal_for_cell: dict[tuple[int, int], np.ndarray] = {}
        for candidate in candidates:
            cell = self._nearest_reachable_free_cell(candidate, xs, ys, free)
            if cell is None:
                continue
            prior = goal_for_cell.get(cell)
            if prior is None or float(np.linalg.norm(candidate - start)) < float(
                np.linalg.norm(prior - start)
            ):
                goal_for_cell[cell] = candidate
        if not goal_for_cell:
            raise RuntimeError("No collision-free planning goal is available")

        goal_cells = tuple(sorted(goal_for_cell))

        def heuristic(cell: tuple[int, int]) -> float:
            y_index, x_index = cell
            return min(
                math.hypot(x_index - goal_x, y_index - goal_y) * self.grid_resolution_m
                for goal_y, goal_x in goal_cells
            )

        frontier: list[tuple[float, float, int, tuple[int, int]]] = []
        sequence = count()
        heapq.heappush(frontier, (heuristic(start_cell), 0.0, next(sequence), start_cell))
        costs: dict[tuple[int, int], float] = {start_cell: 0.0}
        parent: dict[tuple[int, int], tuple[int, int]] = {}
        expanded = 0
        reached: tuple[int, int] | None = None
        neighbors = (
            (-1, 0, 1.0),
            (0, -1, 1.0),
            (0, 1, 1.0),
            (1, 0, 1.0),
            (-1, -1, math.sqrt(2.0)),
            (-1, 1, math.sqrt(2.0)),
            (1, -1, math.sqrt(2.0)),
            (1, 1, math.sqrt(2.0)),
        )
        while frontier:
            _priority, queued_cost, _order, current = heapq.heappop(frontier)
            if queued_cost > costs.get(current, math.inf) + 1e-12:
                continue
            expanded += 1
            if expanded > self.max_expanded_nodes:
                raise RuntimeError("Numerical waypoint search exceeded its node limit")
            if current in goal_for_cell:
                reached = current
                break
            current_y, current_x = current
            for delta_y, delta_x, scale in neighbors:
                neighbor = (current_y + delta_y, current_x + delta_x)
                neighbor_y, neighbor_x = neighbor
                if (
                    neighbor_y < 0
                    or neighbor_y >= free.shape[0]
                    or neighbor_x < 0
                    or neighbor_x >= free.shape[1]
                    or not free[neighbor]
                ):
                    continue
                # Do not cut diagonally through two occupied orthogonal cells.
                if delta_y and delta_x and (
                    not free[current_y + delta_y, current_x]
                    or not free[current_y, current_x + delta_x]
                ):
                    continue
                tentative = queued_cost + scale * self.grid_resolution_m
                if tentative + 1e-12 >= costs.get(neighbor, math.inf):
                    continue
                costs[neighbor] = tentative
                parent[neighbor] = current
                heapq.heappush(
                    frontier,
                    (tentative + heuristic(neighbor), tentative, next(sequence), neighbor),
                )
        if reached is None:
            raise RuntimeError("No collision-free route to the numeric target was found")

        cells = [reached]
        while cells[-1] != start_cell:
            cells.append(parent[cells[-1]])
        cells.reverse()
        points = [np.asarray([xs[x_index], ys[y_index]], dtype=np.float64) for y_index, x_index in cells]
        return points, goal_for_cell[reached], expanded

    def _simplify(self, points: list[np.ndarray]) -> list[np.ndarray]:
        if len(points) < 3:
            return points
        simplified = [points[0]]
        index = 0
        while index < len(points) - 1:
            candidate = len(points) - 1
            while candidate > index + 1:
                if not self.collision_map.segment_check(points[index], points[candidate]).collision:
                    break
                candidate -= 1
            simplified.append(points[candidate])
            index = candidate
        return simplified

    def _subdivide(self, points: list[np.ndarray]) -> list[np.ndarray]:
        output = [points[0]]
        for endpoint in points[1:]:
            start = output[-1]
            delta = endpoint - start
            distance = float(np.linalg.norm(delta))
            pieces = max(1, math.ceil(distance / self.max_waypoint_step_m))
            for step in range(1, pieces + 1):
                output.append(start + delta * (step / pieces))
        return output

    def plan(
        self,
        start_xy_m: np.ndarray | tuple[float, float],
        target_xy_m: np.ndarray | tuple[float, float],
    ) -> NumericPathPlan:
        start = _finite_xy(start_xy_m, name="start_xy_m")
        target = _finite_xy(target_xy_m, name="target_xy_m")
        if self.collision_map.point_check(start).collision:
            raise ValueError("Robot start position is in collision")
        candidates = self._approach_candidates(start, target)
        if not candidates:
            raise RuntimeError("No collision-free standoff point exists around the numeric target")

        visible = [
            candidate
            for candidate in candidates
            if not self.collision_map.segment_check(start, candidate).collision
        ]
        direct = bool(visible)
        expanded = 0
        if visible:
            goal = min(
                visible,
                key=lambda candidate: (
                    abs(float(np.linalg.norm(candidate - target)) - self.standoff_m),
                    float(np.linalg.norm(candidate - start)),
                    float(candidate[0]),
                    float(candidate[1]),
                ),
            )
            route = [start, goal]
        else:
            grid_route, goal, expanded = self._a_star(start, candidates)
            route = [start, *grid_route, goal]
        route = self._subdivide(self._simplify(route))
        for first, second in pairwise(route):
            if self.collision_map.segment_check(first, second).collision:
                raise RuntimeError("Planner produced a collision-bearing continuous segment")
            if float(np.linalg.norm(second - first)) > self.max_waypoint_step_m + 1e-9:
                raise RuntimeError("Planner produced an over-limit waypoint step")
        path_length = sum(
            float(np.linalg.norm(second - first))
            for first, second in pairwise(route)
        )
        goal = route[-1]
        return NumericPathPlan(
            target_xy_m=(float(target[0]), float(target[1])),
            goal_xy_m=(float(goal[0]), float(goal[1])),
            waypoints_xy_m=tuple(
                (float(point[0]), float(point[1])) for point in route[1:]
            ),
            path_length_m=path_length,
            target_distance_m=float(np.linalg.norm(goal - target)),
            direct_path=direct,
            expanded_nodes=expanded,
        )

    def plan_to_free_point(
        self,
        start_xy_m: np.ndarray | tuple[float, float],
        goal_xy_m: np.ndarray | tuple[float, float],
    ) -> NumericPathPlan:
        """Plan to an exact anonymous free-space coordinate.

        Semantic object targets normally lie on occupied surfaces and therefore
        use :meth:`plan` with a standoff ring.  Patrol and coverage objectives
        instead provide already-free numeric coordinates derived from the
        occupancy field.  This method keeps those two contracts distinct while
        using the same deterministic A* search, swept-segment validation, and
        bounded waypoint subdivision.
        """

        start = _finite_xy(start_xy_m, name="start_xy_m")
        goal = _finite_xy(goal_xy_m, name="goal_xy_m")
        if self.collision_map.point_check(start).collision:
            raise ValueError("Robot start position is in collision")
        if self.collision_map.point_check(goal).collision:
            raise ValueError("Requested free-space goal is in collision")

        direct = not self.collision_map.segment_check(start, goal).collision
        expanded = 0
        if direct:
            route = [start, goal]
        else:
            grid_route, reached_goal, expanded = self._a_star(start, [goal])
            if float(np.linalg.norm(reached_goal - goal)) > 1e-8:
                raise RuntimeError("Free-space planner changed the requested goal")
            route = [start, *grid_route, goal]

        route = self._subdivide(self._simplify(route))
        for first, second in pairwise(route):
            if self.collision_map.segment_check(first, second).collision:
                raise RuntimeError(
                    "Free-space planner produced a collision-bearing segment"
                )
            if float(np.linalg.norm(second - first)) > self.max_waypoint_step_m + 1e-9:
                raise RuntimeError("Free-space planner produced an over-limit step")
        path_length = sum(
            float(np.linalg.norm(second - first))
            for first, second in pairwise(route)
        )
        return NumericPathPlan(
            target_xy_m=(float(goal[0]), float(goal[1])),
            goal_xy_m=(float(goal[0]), float(goal[1])),
            waypoints_xy_m=tuple(
                (float(point[0]), float(point[1])) for point in route[1:]
            ),
            path_length_m=path_length,
            target_distance_m=0.0,
            direct_path=direct,
            expanded_nodes=expanded,
        )


__all__ = ["NumericPathPlan", "NumericWaypointPlanner"]
