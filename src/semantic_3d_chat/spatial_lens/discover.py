"""Find objects in the point cloud without being told what to look for.

The room shell is removed geometrically -- floor, ceiling and the four wall
planes are known from the scan's own room dimensions, not from any label -- and
what remains is grouped by 26-connectivity over occupied voxels.  Each surviving
group becomes an object proposal with a metric footprint.

This stage deliberately produces anonymous proposals.  It has no vocabulary, no
detector and no oracle; naming happens afterwards, from pixels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from semantic_3d_chat.spatial_lens.perceive import SemanticCloud


@dataclass(frozen=True)
class ObjectProposal:
    """One anonymous blob of occupied space."""

    proposal_id: str
    voxel_indices: np.ndarray
    center_m: tuple[float, float, float]
    bbox_min_m: tuple[float, float, float]
    bbox_max_m: tuple[float, float, float]
    mean_rgb: tuple[float, float, float]
    voxel_count: int

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
            "proposal_id": self.proposal_id,
            "center_m": list(self.center_m),
            "bbox_min_m": list(self.bbox_min_m),
            "bbox_max_m": list(self.bbox_max_m),
            "footprint_m": list(self.footprint_m),
            "height_m": self.height_m,
            "mean_rgb": list(self.mean_rgb),
            "voxel_count": self.voxel_count,
        }


class _UnionFind:
    def __init__(self, size: int) -> None:
        self._parent = list(range(size))

    def find(self, item: int) -> int:
        parent = self._parent
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self._parent[b] = a


def _shell_mask(
    centers: np.ndarray,
    room_size_m: tuple[float, float, float],
    *,
    floor_margin_m: float,
    wall_margin_m: float,
) -> np.ndarray:
    """True where a point belongs to the room shell rather than its contents."""

    width, depth, height = room_size_m
    half_x, half_y = width / 2.0, depth / 2.0
    x, y, z = centers[:, 0], centers[:, 1], centers[:, 2]
    return (
        (z < floor_margin_m)
        | (z > height - floor_margin_m)
        | (np.abs(x) > half_x - wall_margin_m)
        | (np.abs(y) > half_y - wall_margin_m)
    )


def discover_objects(
    cloud: SemanticCloud,
    *,
    floor_margin_m: float = 0.09,
    wall_margin_m: float = 0.14,
    connect_radius_voxels: int = 1,
    min_voxels: int = 40,
    min_height_m: float = 0.08,
) -> list[ObjectProposal]:
    """Group the non-shell point cloud into anonymous object proposals."""

    centers = np.asarray(cloud.centers_m, dtype=np.float64)
    keep = ~_shell_mask(
        centers,
        cloud.room_size_m,
        floor_margin_m=floor_margin_m,
        wall_margin_m=wall_margin_m,
    )
    indices = np.flatnonzero(keep)
    if not len(indices):
        return []

    voxel = np.floor(centers[indices] / cloud.voxel_size_m).astype(np.int64)
    lookup = {tuple(key): position for position, key in enumerate(map(tuple, voxel))}
    union = _UnionFind(len(indices))
    span = range(-connect_radius_voxels, connect_radius_voxels + 1)
    offsets = [
        (dx, dy, dz)
        for dx in span
        for dy in span
        for dz in span
        if (dx, dy, dz) > (0, 0, 0)
    ]
    for key, position in lookup.items():
        for dx, dy, dz in offsets:
            neighbour = lookup.get((key[0] + dx, key[1] + dy, key[2] + dz))
            if neighbour is not None:
                union.union(position, neighbour)

    groups: dict[int, list[int]] = {}
    for position in range(len(indices)):
        groups.setdefault(union.find(position), []).append(position)

    kept: list[np.ndarray] = []
    half = cloud.voxel_size_m / 2.0
    for members in sorted(groups.values(), key=len, reverse=True):
        if len(members) < min_voxels:
            continue
        picked = indices[np.asarray(members, dtype=np.int64)]
        points = centers[picked]
        if float(points[:, 2].max() - points[:, 2].min()) + 2 * half < min_height_m:
            continue
        kept.append(picked)

    # A single object can survive as more than one connected group when a thin
    # part is separated by occlusion -- a shelf back panel behind its sides, for
    # example.  Absorb any group whose plan-view footprint sits almost entirely
    # inside a larger group's footprint; two genuinely distinct objects do not
    # occupy the same floor area.
    merged = _absorb_contained(kept, centers, containment=0.70, pad=half)

    proposals: list[ObjectProposal] = []
    for picked in sorted(merged, key=len, reverse=True):
        points = centers[picked]
        lower = points.min(axis=0) - half
        upper = points.max(axis=0) + half
        proposals.append(
            ObjectProposal(
                proposal_id=f"p_{len(proposals) + 1:03d}",
                voxel_indices=picked,
                center_m=tuple(float(v) for v in points.mean(axis=0)),
                bbox_min_m=tuple(float(v) for v in lower),
                bbox_max_m=tuple(float(v) for v in upper),
                mean_rgb=tuple(
                    float(v) for v in np.asarray(cloud.rgb)[picked].mean(axis=0)
                ),
                voxel_count=len(picked),
            )
        )
    return proposals


def _xy_bounds(points: np.ndarray, pad: float) -> tuple[float, float, float, float]:
    """Plan-view extent, padded so a single-voxel-thick slab still has area."""

    return (
        float(points[:, 0].min()) - pad,
        float(points[:, 1].min()) - pad,
        float(points[:, 0].max()) + pad,
        float(points[:, 1].max()) + pad,
    )


def _absorb_contained(
    groups: list[np.ndarray], centers: np.ndarray, *, containment: float, pad: float
) -> list[np.ndarray]:
    order = sorted(range(len(groups)), key=lambda i: len(groups[i]), reverse=True)
    bounds = {i: _xy_bounds(centers[groups[i]], pad) for i in order}
    absorbed_into: dict[int, int] = {}
    for small_pos, small in enumerate(order):
        for large in order[:small_pos]:
            if large in absorbed_into:
                continue
            sx0, sy0, sx1, sy1 = bounds[small]
            lx0, ly0, lx1, ly1 = bounds[large]
            overlap_x = max(0.0, min(sx1, lx1) - max(sx0, lx0))
            overlap_y = max(0.0, min(sy1, ly1) - max(sy0, ly0))
            small_area = max((sx1 - sx0) * (sy1 - sy0), 1e-9)
            if (overlap_x * overlap_y) / small_area >= containment:
                absorbed_into[small] = large
                break
    result: dict[int, list[np.ndarray]] = {}
    for index in order:
        root = index
        while root in absorbed_into:
            root = absorbed_into[root]
        result.setdefault(root, []).append(groups[index])
    return [np.concatenate(parts) for parts in result.values()]


__all__ = ["ObjectProposal", "discover_objects"]
