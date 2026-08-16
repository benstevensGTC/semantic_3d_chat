"""The room as the system actually perceived it: named objects with metrics.

A scene graph is the handover point between perception and reasoning.  It holds
what Gemma decided each blob is, where it is in metres, how big it is, and which
floor area a rover can actually occupy.  Everything in it was derived from
images; nothing was looked up.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

GRAPH_SCHEMA = "semantic_3d_chat.spatial_lens.scene_graph.v1"


@dataclass(frozen=True)
class SceneObject:
    object_id: str
    name: str
    center_m: tuple[float, float, float]
    bbox_min_m: tuple[float, float, float]
    bbox_max_m: tuple[float, float, float]
    mean_rgb: tuple[float, float, float]
    voxel_count: int
    name_confidence: float
    name_votes: dict[str, int] = field(default_factory=dict)

    @property
    def footprint_m(self) -> tuple[float, float]:
        return (
            self.bbox_max_m[0] - self.bbox_min_m[0],
            self.bbox_max_m[1] - self.bbox_min_m[1],
        )

    @property
    def height_m(self) -> float:
        return self.bbox_max_m[2] - self.bbox_min_m[2]

    def as_dict(self) -> dict[str, Any]:
        return {
            "object_id": self.object_id,
            "name": self.name,
            "center_m": [round(v, 4) for v in self.center_m],
            "bbox_min_m": [round(v, 4) for v in self.bbox_min_m],
            "bbox_max_m": [round(v, 4) for v in self.bbox_max_m],
            "footprint_m": [round(v, 4) for v in self.footprint_m],
            "height_m": round(self.height_m, 4),
            "mean_rgb": [round(v, 4) for v in self.mean_rgb],
            "voxel_count": self.voxel_count,
            "name_confidence": round(self.name_confidence, 4),
            "name_votes": self.name_votes,
        }


@dataclass(frozen=True)
class SceneGraph:
    room: str
    room_size_m: tuple[float, float, float]
    objects: tuple[SceneObject, ...]
    free_grid: np.ndarray  # [ny, nx] bool: True where the rover body fits
    grid_resolution_m: float
    rover_radius_m: float

    # ------------------------------------------------------------------ io
    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": GRAPH_SCHEMA,
            "room": self.room,
            "room_size_m": [float(v) for v in self.room_size_m],
            "grid_resolution_m": self.grid_resolution_m,
            "rover_radius_m": self.rover_radius_m,
            "free_fraction": float(self.free_grid.mean()),
            "objects": [item.as_dict() for item in self.objects],
        }

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.as_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        np.save(destination.with_suffix(".freegrid.npy"), self.free_grid)
        return destination

    @classmethod
    def load(cls, path: str | Path) -> SceneGraph:
        source = Path(path)
        payload = json.loads(source.read_text(encoding="utf-8"))
        if payload.get("schema") != GRAPH_SCHEMA:
            raise ValueError("Unexpected scene-graph schema")
        grid = np.load(source.with_suffix(".freegrid.npy"))
        return cls(
            room=payload["room"],
            room_size_m=tuple(float(v) for v in payload["room_size_m"]),
            objects=tuple(
                SceneObject(
                    object_id=item["object_id"],
                    name=item["name"],
                    center_m=tuple(item["center_m"]),
                    bbox_min_m=tuple(item["bbox_min_m"]),
                    bbox_max_m=tuple(item["bbox_max_m"]),
                    mean_rgb=tuple(item["mean_rgb"]),
                    voxel_count=int(item["voxel_count"]),
                    name_confidence=float(item["name_confidence"]),
                    name_votes=dict(item.get("name_votes", {})),
                )
                for item in payload["objects"]
            ),
            free_grid=grid,
            grid_resolution_m=float(payload["grid_resolution_m"]),
            rover_radius_m=float(payload["rover_radius_m"]),
        )

    # -------------------------------------------------------------- queries
    def find(self, name: str) -> SceneObject | None:
        """Loose lookup by perceived name; returns the largest match."""

        needle = name.strip().lower()
        matches = [
            item
            for item in self.objects
            if needle == item.name.lower()
            or needle in item.name.lower()
            or item.name.lower() in needle
        ]
        if not matches:
            return None
        return max(matches, key=lambda item: item.voxel_count)

    def is_free(self, x: float, y: float) -> bool:
        column, row = self._cell(x, y)
        if row < 0 or column < 0 or row >= self.free_grid.shape[0]:
            return False
        if column >= self.free_grid.shape[1]:
            return False
        return bool(self.free_grid[row, column])

    def _cell(self, x: float, y: float) -> tuple[int, int]:
        half_x = self.room_size_m[0] / 2.0
        half_y = self.room_size_m[1] / 2.0
        column = int((x + half_x) / self.grid_resolution_m)
        row = int((y + half_y) / self.grid_resolution_m)
        return column, row

    def cell_center(self, column: int, row: int) -> tuple[float, float]:
        half_x = self.room_size_m[0] / 2.0
        half_y = self.room_size_m[1] / 2.0
        return (
            (column + 0.5) * self.grid_resolution_m - half_x,
            (row + 0.5) * self.grid_resolution_m - half_y,
        )

    def nearest_free(self, x: float, y: float, *, max_radius_m: float = 2.0) -> tuple[float, float] | None:
        """Closest standable point to a request, for goals inside furniture."""

        if self.is_free(x, y):
            return (x, y)
        steps = int(max_radius_m / self.grid_resolution_m)
        column, row = self._cell(x, y)
        for ring in range(1, steps + 1):
            best: tuple[float, tuple[float, float]] | None = None
            for dc in range(-ring, ring + 1):
                for dr in (-ring, ring):
                    for cc, rr in ((column + dc, row + dr), (column + dr, row + dc)):
                        if (
                            0 <= rr < self.free_grid.shape[0]
                            and 0 <= cc < self.free_grid.shape[1]
                            and self.free_grid[rr, cc]
                        ):
                            point = self.cell_center(cc, rr)
                            distance = (point[0] - x) ** 2 + (point[1] - y) ** 2
                            if best is None or distance < best[0]:
                                best = (distance, point)
            if best is not None:
                return best[1]
        return None

    def approach_point(self, item: SceneObject) -> tuple[float, float] | None:
        """A free cell adjacent to an object, where a rover could stand.

        This is measured geometry offered to the model as information -- "here
        is somewhere you could stand next to the lamp" -- not a route. The model
        still decides whether to go there, and every step is still its own.
        """

        best: tuple[float, tuple[float, float]] | None = None
        rows, columns = self.free_grid.shape
        for row in range(rows):
            for column in range(columns):
                if not self.free_grid[row, column]:
                    continue
                x, y = self.cell_center(column, row)
                dx = max(item.bbox_min_m[0] - x, 0.0, x - item.bbox_max_m[0])
                dy = max(item.bbox_min_m[1] - y, 0.0, y - item.bbox_max_m[1])
                gap = math.hypot(dx, dy)
                if best is None or gap < best[0]:
                    best = (gap, (x, y))
        return None if best is None else best[1]

    # ----------------------------------------------------------- rendering
    def describe(self, *, include_grid_summary: bool = True) -> str:
        """A compact, metric description for the language model to reason over.

        Coordinates are the room's own frame: +X right, +Y forward, origin at
        the room centre, metres.  Headings follow the rover convention so the
        model can answer with numbers the executor accepts directly.
        """

        width, depth, height = self.room_size_m
        lines = [
            "ROOM",
            f"  size: {width:.2f} m (X) by {depth:.2f} m (Y), {height:.2f} m tall",
            (
                f"  coordinates: origin at room centre, X in "
                f"[{-width / 2:.2f}, {width / 2:.2f}], Y in "
                f"[{-depth / 2:.2f}, {depth / 2:.2f}], metres"
            ),
            (
                "  headings: yaw 0 faces +Y, yaw -90 faces +X, "
                "yaw +90 faces -X, yaw 180 faces -Y"
            ),
            "",
            "OBJECTS (perceived from the scan, with measured extents)",
        ]
        for item in sorted(self.objects, key=lambda o: o.center_m[0]):
            x0, y0 = item.bbox_min_m[0], item.bbox_min_m[1]
            x1, y1 = item.bbox_max_m[0], item.bbox_max_m[1]
            lines.append(
                f"  {item.object_id} {item.name}: centre ({item.center_m[0]:+.2f}, "
                f"{item.center_m[1]:+.2f}), occupies X [{x0:+.2f}, {x1:+.2f}] and "
                f"Y [{y0:+.2f}, {y1:+.2f}], height {item.height_m:.2f} m"
            )
        if include_grid_summary:
            lines.extend(
                [
                    "",
                    "FLOOR",
                    (
                        f"  a rover of radius {self.rover_radius_m:.2f} m can "
                        f"stand on {100.0 * float(self.free_grid.mean()):.0f}% "
                        f"of the floor area"
                    ),
                ]
            )
        return "\n".join(lines)


def build_free_grid(
    objects: list[SceneObject],
    room_size_m: tuple[float, float, float],
    *,
    resolution_m: float,
    rover_radius_m: float,
    ignore_height_m: float,
) -> np.ndarray:
    """Mark floor cells where the rover's body fits without touching anything.

    Objects shorter than ``ignore_height_m`` are treated as drivable (a rug does
    not block a rover), and every obstacle is inflated by the rover radius so a
    free cell is a legal centre position rather than merely an empty point.
    """

    width, depth, _height = room_size_m
    columns = max(1, round(width / resolution_m))
    rows = max(1, round(depth / resolution_m))
    grid = np.ones((rows, columns), dtype=bool)
    xs = (np.arange(columns) + 0.5) * resolution_m - width / 2.0
    ys = (np.arange(rows) + 0.5) * resolution_m - depth / 2.0
    mesh_x, mesh_y = np.meshgrid(xs, ys)

    # Walls: the rover centre must stay a full radius from every wall plane.
    grid &= np.abs(mesh_x) <= (width / 2.0 - rover_radius_m)
    grid &= np.abs(mesh_y) <= (depth / 2.0 - rover_radius_m)

    for item in objects:
        if item.height_m < ignore_height_m:
            continue
        x0 = item.bbox_min_m[0] - rover_radius_m
        x1 = item.bbox_max_m[0] + rover_radius_m
        y0 = item.bbox_min_m[1] - rover_radius_m
        y1 = item.bbox_max_m[1] + rover_radius_m
        blocked = (mesh_x >= x0) & (mesh_x <= x1) & (mesh_y >= y0) & (mesh_y <= y1)
        grid &= ~blocked
    return grid


__all__ = [
    "GRAPH_SCHEMA",
    "SceneGraph",
    "SceneObject",
    "build_free_grid",
]
