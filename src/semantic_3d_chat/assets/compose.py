"""Compose a room out of real furniture, with the properties the study needs.

Three things the hand-built primitive rooms could not give:

*Duplicates.* Every one of the twenty-seven primitive rooms held at most one
object of any kind, so "the chair nearest the bookshelf" had a single answer for
a reason that had nothing to do with geometry. Categories are deliberately
repeated here, which is what makes a relation the only way to tell two things
apart.

*Variety.* Rooms differ in extent, proportion and how full they are, so nothing
can be learned from the shape of the room itself.

*Occlusion.* Small objects stand on top of larger ones, which is both how rooms
actually look and the thing that makes a single viewpoint insufficient.

Placement is rejection-sampled against the walls and against everything already
placed, so nothing intersects anything -- an object clipping through a wall
would be a geometry error the scan would faithfully record and the model would
have to explain.

The category strings here never reach the model. They pick what goes in the
room and they tell the scorer what it placed; every name the system reasons
about still comes from Gemma looking at the object.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# What can stand on the floor, what belongs on a surface, what hangs.
SURFACE_CATEGORIES = frozenset(
    {"vase", "books", "bottle", "pot", "clock", "camera", "basket", "speaker", "lamp"}
)
CEILING_CATEGORIES = frozenset({"chandelier"})
WALL_CATEGORIES = frozenset({"painting", "mirror"})
# Things wide and flat enough to put something else on.
SUPPORT_CATEGORIES = frozenset({"table", "desk", "cabinet", "bookshelf"})

# What each word actually implies about size, as (min, max) largest extent in
# metres. The tag-based classifier is generous -- a 17 cm decorative ledge and a
# two-metre bookcase both answer to "shelf" -- and an object labelled one thing
# while looking like another poisons the scoring without ever reaching the
# model. Anything outside its category's range is dropped rather than renamed.
CATEGORY_SIZE_M: dict[str, tuple[float, float]] = {
    "bed": (1.4, 2.6),
    "sofa": (1.2, 2.6),
    "armchair": (0.6, 1.4),
    "chair": (0.4, 1.4),
    "desk": (0.7, 2.2),
    "table": (0.4, 2.4),
    "cabinet": (0.5, 2.6),
    "bookshelf": (0.7, 2.6),
    "television": (0.3, 1.8),
    "chandelier": (0.3, 1.4),
    "lamp": (0.15, 1.9),
    "speaker": (0.15, 1.4),
    "barrel": (0.4, 1.2),
    "crate": (0.2, 1.4),
    "basket": (0.2, 1.0),
    "vase": (0.15, 1.0),
    "pot": (0.12, 0.8),
    "bottle": (0.1, 0.6),
    "books": (0.15, 0.7),
    "plant": (0.2, 2.0),
    "clock": (0.1, 0.6),
    "bucket": (0.15, 0.7),
    "toolbox": (0.2, 0.9),
    "suitcase": (0.3, 1.1),
    "rug": (0.8, 3.0),
    "mirror": (0.3, 2.2),
    "painting": (0.3, 2.2),
    "ball": (0.15, 0.5),
    "guitar": (0.5, 1.3),
    "camera": (0.1, 0.5),
    "fan": (0.2, 1.6),
    "heater": (0.3, 1.6),
    "sign": (0.3, 1.4),
}

# Below this an object is a few voxels across and neither the vision encoder nor
# the point cloud can say anything reliable about it, so it may sit on a surface
# as clutter but never stands alone on the floor as a target.
MIN_FLOOR_EXTENT_M = 0.35

BUILD_SCHEMA = "semantic_3d_chat.assets.room_build.v1"
KEY_SCHEMA = "semantic_3d_chat.assets.room_key.v1"


@dataclass(frozen=True)
class Placement:
    """One asset, where it goes, and how big it is once placed."""

    instance_id: str
    asset_id: str
    category: str
    gltf: str
    position_m: tuple[float, float, float]
    yaw_degrees: float
    scale: float
    size_m: tuple[float, float, float]
    support: str | None = None

    def footprint_m(self) -> tuple[float, float]:
        """Extent along world x and y after the yaw is applied."""

        angle = math.radians(self.yaw_degrees)
        cos, sin = abs(math.cos(angle)), abs(math.sin(angle))
        width, depth, _ = self.size_m
        return (width * cos + depth * sin, width * sin + depth * cos)


@dataclass(frozen=True)
class ComposedRoom:
    """A whole room: its shell, its contents, and how it was drawn."""

    name: str
    size_m: tuple[float, float, float]
    placements: tuple[Placement, ...]
    seed: int
    clutter: float
    recipe: dict[str, Any] = field(default_factory=dict)

    def build_payload(self) -> dict[str, Any]:
        """Geometry and asset paths. No category names anywhere."""

        return {
            "schema": BUILD_SCHEMA,
            "name": self.name,
            "room_size_m": list(self.size_m),
            "objects": [
                {
                    "instance_id": p.instance_id,
                    "gltf": p.gltf,
                    "position_m": [round(v, 4) for v in p.position_m],
                    "yaw_degrees": round(p.yaw_degrees, 2),
                    "scale": round(p.scale, 4),
                }
                for p in self.placements
            ],
        }

    def key_payload(self) -> dict[str, Any]:
        """What was actually placed. Scorer-only; the model never sees it."""

        return {
            "schema": KEY_SCHEMA,
            "name": self.name,
            "seed": self.seed,
            "clutter": round(self.clutter, 3),
            "room_size_m": [round(v, 3) for v in self.size_m],
            "objects": [
                {
                    "instance_id": p.instance_id,
                    "asset_id": p.asset_id,
                    "category": p.category,
                    "position_m": [round(v, 4) for v in p.position_m],
                    "size_m": [round(v, 4) for v in p.size_m],
                    "yaw_degrees": round(p.yaw_degrees, 2),
                    "support": p.support,
                }
                for p in self.placements
            ],
            "recipe": self.recipe,
        }


def _overlaps(
    centre: tuple[float, float],
    extent: tuple[float, float],
    placed: list[tuple[tuple[float, float], tuple[float, float]]],
    clearance: float,
) -> bool:
    for other_centre, other_extent in placed:
        gap_x = abs(centre[0] - other_centre[0]) - (extent[0] + other_extent[0]) / 2.0
        gap_y = abs(centre[1] - other_centre[1]) - (extent[1] + other_extent[1]) / 2.0
        if gap_x < clearance and gap_y < clearance:
            return True
    return False


def load_manifest(root: Path) -> list[dict[str, Any]]:
    path = root / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"no asset manifest at {path}; run lens_fetch_assets.py")
    return json.loads(path.read_text(encoding="utf-8"))


def compose_room(
    name: str,
    manifest: list[dict[str, Any]],
    *,
    seed: int,
    min_extent_m: float = 4.0,
    max_extent_m: float = 9.0,
    wall_clearance_m: float = 0.25,
    object_clearance_m: float = 0.22,
    duplicate_categories: tuple[int, int] = (1, 3),
    min_objects: int = 5,
    attempts: int = 400,
    tries: int = 12,
) -> ComposedRoom:
    """Draw one room: its size, its contents, and where everything stands."""

    rng = random.Random(seed)
    width = round(rng.uniform(min_extent_m, max_extent_m), 2)
    depth = round(rng.uniform(min_extent_m, max_extent_m), 2)
    height = round(rng.uniform(2.6, 3.4), 2)
    # How full the room is, as a fraction of floor area the furniture may cover.
    clutter = rng.uniform(0.10, 0.26)

    by_category: dict[str, list[dict[str, Any]]] = {}
    for entry in manifest:
        category = entry["category"]
        extent = max(float(v) for v in entry["size_m"])
        low, high = CATEGORY_SIZE_M.get(category, (0.1, 3.0))
        if not (low <= extent <= high):
            continue
        by_category.setdefault(category, []).append(entry)

    floor_pool = [c for c in by_category if c not in CEILING_CATEGORIES | WALL_CATEGORIES]
    surface_pool = [c for c in floor_pool if c in SURFACE_CATEGORIES]
    standing_pool = [
        c
        for c in floor_pool
        if c not in SURFACE_CATEGORIES
        and any(
            max(float(v) for v in e["size_m"]) >= MIN_FLOOR_EXTENT_M
            for e in by_category[c]
        )
    ]
    if not standing_pool:
        raise ValueError("asset manifest has nothing that can stand on a floor")

    # Repeat at least one category on purpose, so a relation is the only way to
    # say which one is meant. Left to chance this happens in some rooms and not
    # others, and the rooms without it teach the model that a category name is
    # always sufficient -- which is the habit the relational task has to break.
    low, high = duplicate_categories
    repeatable = [c for c in standing_pool if len(by_category[c]) >= 2]
    if not repeatable:
        raise ValueError("no category has two assets; cannot guarantee a duplicate")
    guaranteed = rng.choice(repeatable)
    wanted = rng.sample(standing_pool, k=min(len(standing_pool), rng.randint(4, 7)))
    if guaranteed not in wanted:
        wanted.append(guaranteed)
    plan: list[str] = []
    for category in wanted:
        available = len(by_category[category])
        if category == guaranteed:
            count = rng.randint(2, min(max(high, 2), available))
        else:
            count = rng.randint(low, min(high, available))
        plan += [category] * count
    # The guaranteed duplicate goes first so the clutter budget cannot squeeze
    # it out in favour of whatever happened to be drawn earlier.
    plan.sort(key=lambda c: c != guaranteed)

    budget = width * depth * clutter
    placed_boxes: list[tuple[tuple[float, float], tuple[float, float]]] = []
    placements: list[Placement] = []
    used_assets: set[str] = set()
    covered = 0.0
    counter = 0

    for category in plan:
        options = [e for e in by_category[category] if e["asset_id"] not in used_assets]
        if not options:
            continue
        options = [
            e for e in options
            if max(float(v) for v in e["size_m"]) >= MIN_FLOOR_EXTENT_M
        ]
        if not options:
            continue
        entry = rng.choice(options)
        scale = rng.uniform(0.85, 1.18)
        size = tuple(float(v) * scale for v in entry["size_m"])
        if size[0] > width - 2 * wall_clearance_m or size[1] > depth - 2 * wall_clearance_m:
            continue
        yaw = rng.uniform(0.0, 360.0)
        counter += 1
        candidate = Placement(
            instance_id=f"a_{counter:04d}",
            asset_id=entry["asset_id"],
            category=category,
            gltf=f"{entry['asset_id']}/{entry['gltf']}",
            position_m=(0.0, 0.0, 0.0),
            yaw_degrees=yaw,
            scale=scale,
            size_m=size,  # type: ignore[arg-type]
        )
        extent = candidate.footprint_m()
        if covered + extent[0] * extent[1] > budget and placements:
            continue

        half_x = max(width / 2.0 - wall_clearance_m - extent[0] / 2.0, 0.0)
        half_y = max(depth / 2.0 - wall_clearance_m - extent[1] / 2.0, 0.0)
        if half_x <= 0.0 or half_y <= 0.0:
            continue
        for _ in range(attempts):
            spot = (rng.uniform(-half_x, half_x), rng.uniform(-half_y, half_y))
            if not _overlaps(spot, extent, placed_boxes, object_clearance_m):
                placed_boxes.append((spot, extent))
                placements.append(
                    Placement(
                        instance_id=candidate.instance_id,
                        asset_id=candidate.asset_id,
                        category=category,
                        gltf=candidate.gltf,
                        position_m=(round(spot[0], 4), round(spot[1], 4), 0.0),
                        yaw_degrees=yaw,
                        scale=scale,
                        size_m=size,  # type: ignore[arg-type]
                    )
                )
                covered += extent[0] * extent[1]
                used_assets.add(entry["asset_id"])
                break

    # Small things on top of big things: real rooms look like this, and it is
    # what makes one viewpoint insufficient.
    supports = [p for p in placements if p.category in SUPPORT_CATEGORIES]
    for support in supports:
        if not surface_pool or rng.random() > 0.65:
            continue
        category = rng.choice(surface_pool)
        options = [e for e in by_category[category] if e["asset_id"] not in used_assets]
        if not options:
            continue
        entry = rng.choice(options)
        scale = rng.uniform(0.85, 1.1)
        size = tuple(float(v) * scale for v in entry["size_m"])
        support_extent = support.footprint_m()
        if size[0] > support_extent[0] * 0.7 or size[1] > support_extent[1] * 0.7:
            continue
        counter += 1
        jitter_x = (support_extent[0] - size[0]) / 2.0 * 0.6
        jitter_y = (support_extent[1] - size[1]) / 2.0 * 0.6
        placements.append(
            Placement(
                instance_id=f"a_{counter:04d}",
                asset_id=entry["asset_id"],
                category=category,
                gltf=f"{entry['asset_id']}/{entry['gltf']}",
                position_m=(
                    round(support.position_m[0] + rng.uniform(-jitter_x, jitter_x), 4),
                    round(support.position_m[1] + rng.uniform(-jitter_y, jitter_y), 4),
                    round(support.size_m[2], 4),
                ),
                yaw_degrees=rng.uniform(0.0, 360.0),
                scale=scale,
                size_m=size,  # type: ignore[arg-type]
                support=support.instance_id,
            )
        )
        used_assets.add(entry["asset_id"])

    categories_placed = [p.category for p in placements]
    duplicated = {c for c in categories_placed if categories_placed.count(c) > 1}
    if (len(placements) < min_objects or not duplicated) and tries > 1:
        # A thin or duplicate-free draw is not worth keeping; redraw rather than
        # quietly ship a room that cannot support the questions asked of it.
        return compose_room(
            name,
            manifest,
            seed=seed + 7919,
            min_extent_m=min_extent_m,
            max_extent_m=max_extent_m,
            wall_clearance_m=wall_clearance_m,
            object_clearance_m=object_clearance_m,
            duplicate_categories=duplicate_categories,
            min_objects=min_objects,
            attempts=attempts,
            tries=tries - 1,
        )
    if len(placements) < 3:
        raise ValueError(f"{name}: only {len(placements)} objects placed; loosen the budget")

    categories = categories_placed
    return ComposedRoom(
        name=name,
        size_m=(width, depth, height),
        placements=tuple(placements),
        seed=seed,
        clutter=clutter,
        recipe={
            "objects": len(placements),
            "distinct_categories": len(set(categories)),
            "duplicated_categories": sorted(
                {c for c in categories if categories.count(c) > 1}
            ),
            "floor_coverage": round(covered / (width * depth), 4),
            "on_supports": sum(1 for p in placements if p.support),
        },
    )


__all__ = [
    "BUILD_SCHEMA",
    "CATEGORY_SIZE_M",
    "CEILING_CATEGORIES",
    "KEY_SCHEMA",
    "MIN_FLOOR_EXTENT_M",
    "SUPPORT_CATEGORIES",
    "SURFACE_CATEGORIES",
    "ComposedRoom",
    "Placement",
    "compose_room",
    "load_manifest",
]
