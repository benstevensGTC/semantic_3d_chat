"""A hand-authorable room description, and the split that keeps it honest.

The point of this module is that a person can write a room by hand -- a few
lines of JSON per object -- and the perception stack still has to *discover*
what is in it.  That only means something if the words the author used never
reach the model, so a spec is compiled into two separate artifacts:

``build`` (geometry only)
    Sizes, positions, rotations and RGBA colors.  This is what Blender needs
    and it contains no category name at all: an object is ``i_000003``, a stack
    of primitives, and a color triple.

``key`` (author intent)
    The ``name`` the author gave each object, kept aside for scoring only.  It
    is written under a ``scorer_only`` path that the runtime file audit blocks,
    exactly like the procedural scenes' oracle data.

So "the author called it a table" and "the perception stack decided it is a
table" stay independent, and agreement between them is a measurement rather
than a tautology.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

SPEC_SCHEMA: Final[str] = "semantic_3d_chat.spatial_lens.room_spec.v1"
BUILD_SCHEMA: Final[str] = "semantic_3d_chat.spatial_lens.room_build.v1"
KEY_SCHEMA: Final[str] = "semantic_3d_chat.spatial_lens.room_key.v1"

_NAME = re.compile(r"[a-z0-9][a-z0-9_]{0,62}")

# Authoring colors. The builder needs RGBA; the author writes a word. These are
# deliberately ordinary furniture colors so a scan looks like a room.
COLORS: Final[dict[str, tuple[float, float, float, float]]] = {
    "white": (0.86, 0.85, 0.82, 1.0),
    "cream": (0.88, 0.79, 0.57, 1.0),
    "gray": (0.62, 0.65, 0.67, 1.0),
    "charcoal": (0.10, 0.12, 0.14, 1.0),
    "black": (0.04, 0.04, 0.05, 1.0),
    "wood": (0.40, 0.19, 0.07, 1.0),
    "dark_wood": (0.20, 0.08, 0.025, 1.0),
    "red": (0.78, 0.035, 0.025, 1.0),
    "orange": (0.85, 0.35, 0.04, 1.0),
    "yellow": (0.90, 0.58, 0.035, 1.0),
    "green": (0.035, 0.42, 0.10, 1.0),
    "teal": (0.05, 0.45, 0.45, 1.0),
    "blue": (0.025, 0.18, 0.78, 1.0),
    "purple": (0.35, 0.10, 0.55, 1.0),
    "pink": (0.85, 0.45, 0.60, 1.0),
    "terracotta": (0.58, 0.20, 0.09, 1.0),
}

# Primitive shapes the author can place directly.
PRIMITIVES: Final[frozenset[str]] = frozenset({"box", "cylinder", "sphere"})

# Composite furniture, expanded into primitives by the builder.  The author
# gives a footprint and height; the recipe decides the parts.  These names are
# authoring conveniences only -- they are stripped before Blender sees them.
COMPOSITES: Final[frozenset[str]] = frozenset(
    {"table", "chair", "shelf", "cabinet", "lamp", "bed", "rug", "screen"}
)

SHAPES: Final[frozenset[str]] = PRIMITIVES | COMPOSITES


@dataclass(frozen=True)
class Part:
    """One Blender primitive: the only thing the builder ever sees."""

    kind: str
    center_m: tuple[float, float, float]
    size_m: tuple[float, float, float]
    yaw_degrees: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "center_m": [float(v) for v in self.center_m],
            "size_m": [float(v) for v in self.size_m],
            "yaw_degrees": float(self.yaw_degrees),
        }


@dataclass(frozen=True)
class AuthoredObject:
    instance_id: str
    name: str
    shape: str
    color: str
    rgba: tuple[float, float, float, float]
    position_m: tuple[float, float]
    size_m: tuple[float, float, float]
    yaw_degrees: float
    parts: tuple[Part, ...] = field(default_factory=tuple)

    @property
    def footprint_m(self) -> tuple[float, float]:
        return (self.size_m[0], self.size_m[1])


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _vector(value: object, name: str, size: int) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a list of {size} numbers")
    if len(value) != size:
        raise ValueError(f"{name} must have exactly {size} entries")
    return tuple(_number(item, f"{name}[{index}]") for index, item in enumerate(value))


def _composite_parts(
    shape: str,
    position: tuple[float, float],
    size: tuple[float, float, float],
    yaw: float,
) -> tuple[Part, ...]:
    """Expand authoring furniture into primitives.

    Recipes are intentionally plain and blocky. The perception stack has to
    recognize these from rendered pixels, so they need recognizable silhouettes
    rather than detail: a table is a slab on legs, a chair is a seat with a
    back, a lamp is a pole with a shade.
    """

    x, y = position
    width, depth, height = size
    parts: list[Part] = []
    if shape == "table":
        top = 0.06
        parts.append(Part("box", (x, y, height - top / 2), (width, depth, top), yaw))
        leg = 0.07
        inset = 0.10
        for dx, dy in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
            parts.append(
                Part(
                    "box",
                    (
                        x + dx * (width / 2 - inset),
                        y + dy * (depth / 2 - inset),
                        (height - top) / 2,
                    ),
                    (leg, leg, height - top),
                    yaw,
                )
            )
    elif shape == "chair":
        seat_h = height * 0.45
        seat = 0.06
        parts.append(Part("box", (x, y, seat_h), (width, depth, seat), yaw))
        back = 0.06
        radians = math.radians(yaw)
        # Back panel sits at the -Y edge of the seat, rotated with the chair.
        offset = depth / 2 - back / 2
        parts.append(
            Part(
                "box",
                (
                    x + math.sin(radians) * offset,
                    y - math.cos(radians) * offset,
                    seat_h + height * 0.30,
                ),
                (width, back, height * 0.55),
                yaw,
            )
        )
        leg = 0.05
        for dx, dy in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
            parts.append(
                Part(
                    "box",
                    (
                        x + dx * (width / 2 - 0.05),
                        y + dy * (depth / 2 - 0.05),
                        seat_h / 2,
                    ),
                    (leg, leg, seat_h),
                    yaw,
                )
            )
    elif shape in {"shelf", "cabinet"}:
        wall = 0.05
        parts.append(Part("box", (x, y, height / 2), (width, wall, height), yaw))
        shelves = max(2, int(height / 0.45))
        for index in range(shelves):
            level = height * (index + 0.5) / shelves
            parts.append(Part("box", (x, y, level), (width, depth, wall), yaw))
        for side in (-1, 1):
            parts.append(
                Part(
                    "box",
                    (x + side * (width / 2 - wall / 2), y, height / 2),
                    (wall, depth, height),
                    yaw,
                )
            )
    elif shape == "lamp":
        base = 0.05
        pole = min(width, depth) * 0.12
        parts.append(
            Part("cylinder", (x, y, base / 2), (width * 0.7, depth * 0.7, base), yaw)
        )
        parts.append(
            Part("cylinder", (x, y, height * 0.5), (pole, pole, height * 0.85), yaw)
        )
        parts.append(
            Part(
                "cylinder",
                (x, y, height - height * 0.09),
                (width, depth, height * 0.18),
                yaw,
            )
        )
    elif shape == "bed":
        frame = height * 0.55
        parts.append(Part("box", (x, y, frame / 2), (width, depth, frame), yaw))
        parts.append(
            Part(
                "box",
                (x, y, frame + (height - frame) / 2),
                (width * 0.98, depth * 0.98, height - frame),
                yaw,
            )
        )
    elif shape == "rug":
        parts.append(Part("box", (x, y, max(height, 0.01) / 2), size, yaw))
    elif shape == "screen":
        stand = height * 0.18
        parts.append(
            Part("box", (x, y, stand / 2), (width * 0.35, depth, stand), yaw)
        )
        parts.append(
            Part(
                "box",
                (x, y, stand + (height - stand) / 2),
                (width, max(depth * 0.25, 0.04), height - stand),
                yaw,
            )
        )
    else:  # pragma: no cover - guarded by validation
        raise ValueError(f"Unknown composite shape: {shape}")
    return tuple(parts)


@dataclass(frozen=True)
class RoomSpec:
    name: str
    size_m: tuple[float, float, float]
    objects: tuple[AuthoredObject, ...]
    wall_color: str = "warm_gray"
    floor_color: str = "wood"

    @property
    def bounds_m(self) -> tuple[tuple[float, float], tuple[float, float]]:
        half_x = self.size_m[0] / 2.0
        half_y = self.size_m[1] / 2.0
        return ((-half_x, -half_y), (half_x, half_y))

    def build_payload(self) -> dict[str, Any]:
        """Geometry only. Deliberately contains no author-supplied name."""

        return {
            "schema": BUILD_SCHEMA,
            "room_size_m": [float(v) for v in self.size_m],
            "floor_rgba": list(COLORS.get(self.floor_color, COLORS["wood"])),
            "wall_rgba": list(COLORS.get(self.wall_color, COLORS["gray"])),
            "instances": [
                {
                    "instance_id": item.instance_id,
                    "rgba": [float(v) for v in item.rgba],
                    "parts": [part.as_dict() for part in item.parts],
                }
                for item in self.objects
            ],
        }

    def key_payload(self) -> dict[str, Any]:
        """Author intent, for scoring only. Never given to perception."""

        return {
            "schema": KEY_SCHEMA,
            "room_name": self.name,
            "room_size_m": [float(v) for v in self.size_m],
            "instances": [
                {
                    "instance_id": item.instance_id,
                    "authored_name": item.name,
                    "authored_shape": item.shape,
                    "authored_color": item.color,
                    "position_m": [float(v) for v in item.position_m],
                    "size_m": [float(v) for v in item.size_m],
                    "yaw_degrees": float(item.yaw_degrees),
                }
                for item in self.objects
            ],
        }


def parse_room_spec(payload: Mapping[str, Any]) -> RoomSpec:
    """Validate a hand-written room and expand its furniture into primitives."""

    if payload.get("schema") not in {SPEC_SCHEMA, None}:
        raise ValueError(f"Room spec schema must be {SPEC_SCHEMA}")
    name = payload.get("name")
    if not isinstance(name, str) or _NAME.fullmatch(name) is None:
        raise ValueError("Room 'name' must be lowercase letters, digits, underscore")
    size = _vector(payload.get("size_m", [6.0, 5.0, 2.8]), "size_m", 3)
    if any(value < 1.0 for value in size):
        raise ValueError("Room size_m must be at least 1 m on every axis")
    raw_objects = payload.get("objects")
    if not isinstance(raw_objects, list) or not raw_objects:
        raise ValueError("Room spec needs a nonempty 'objects' list")

    half_x, half_y = size[0] / 2.0, size[1] / 2.0
    objects: list[AuthoredObject] = []
    used: set[str] = set()
    for index, raw in enumerate(raw_objects):
        if not isinstance(raw, Mapping):
            raise TypeError(f"objects[{index}] must be an object")
        label = raw.get("name")
        if not isinstance(label, str) or not label.strip():
            raise ValueError(f"objects[{index}] needs a 'name'")
        label = label.strip()
        shape = str(raw.get("shape", label.split()[-1])).strip().lower()
        if shape not in SHAPES:
            raise ValueError(
                f"objects[{index}] shape {shape!r} must be one of {sorted(SHAPES)}"
            )
        color = str(raw.get("color", "gray")).strip().lower()
        if color not in COLORS:
            raise ValueError(
                f"objects[{index}] color {color!r} must be one of {sorted(COLORS)}"
            )
        position = _vector(raw.get("position_m"), f"objects[{index}].position_m", 2)
        default_size = {
            "table": [1.2, 0.8, 0.74],
            "chair": [0.5, 0.5, 0.9],
            "shelf": [0.9, 0.35, 1.8],
            "cabinet": [1.0, 0.45, 1.1],
            "lamp": [0.35, 0.35, 1.6],
            "bed": [1.4, 2.0, 0.55],
            "rug": [1.6, 1.1, 0.02],
            "screen": [1.1, 0.2, 0.7],
        }.get(shape, [0.4, 0.4, 0.4])
        size_m = _vector(raw.get("size_m", default_size), f"objects[{index}].size_m", 3)
        if any(value <= 0.0 for value in size_m):
            raise ValueError(f"objects[{index}].size_m must be positive")
        yaw = _number(raw.get("yaw_degrees", 0.0), f"objects[{index}].yaw_degrees")

        # Keep furniture inside the room so a scan can actually see it.
        if (
            abs(position[0]) + size_m[0] / 2.0 > half_x + 1e-6
            or abs(position[1]) + size_m[1] / 2.0 > half_y + 1e-6
        ):
            raise ValueError(
                f"objects[{index}] ({label!r}) extends outside the room footprint"
            )
        if size_m[2] > size[2]:
            raise ValueError(f"objects[{index}] ({label!r}) is taller than the room")

        instance_id = f"i_{index + 1:06d}"
        if instance_id in used:  # pragma: no cover - index is unique by construction
            raise ValueError("duplicate instance id")
        used.add(instance_id)
        rgba = COLORS[color]
        if shape in PRIMITIVES:
            parts = (Part(shape, (position[0], position[1], size_m[2] / 2), size_m, yaw),)
        else:
            parts = _composite_parts(shape, position, size_m, yaw)
        objects.append(
            AuthoredObject(
                instance_id=instance_id,
                name=label,
                shape=shape,
                color=color,
                rgba=rgba,
                position_m=(position[0], position[1]),
                size_m=size_m,
                yaw_degrees=yaw,
                parts=parts,
            )
        )

    overlaps = _overlapping_pairs(objects)
    if overlaps:
        pairs = ", ".join(f"{a!r}/{b!r}" for a, b in overlaps)
        raise ValueError(f"Authored objects overlap in plan view: {pairs}")

    return RoomSpec(
        name=name,
        size_m=size,
        objects=tuple(objects),
        wall_color=str(payload.get("wall_color", "gray")).strip().lower(),
        floor_color=str(payload.get("floor_color", "wood")).strip().lower(),
    )


def _overlapping_pairs(
    objects: Sequence[AuthoredObject],
) -> list[tuple[str, str]]:
    """Reject stacked furniture: flat rugs and low objects may overlap."""

    clashes: list[tuple[str, str]] = []
    for i, first in enumerate(objects):
        for second in objects[i + 1 :]:
            # Anything under 5 cm is a floor covering; things may sit on it.
            if min(first.size_m[2], second.size_m[2]) <= 0.05:
                continue
            dx = abs(first.position_m[0] - second.position_m[0])
            dy = abs(first.position_m[1] - second.position_m[1])
            if (
                dx < (first.size_m[0] + second.size_m[0]) / 2.0 - 1e-6
                and dy < (first.size_m[1] + second.size_m[1]) / 2.0 - 1e-6
            ):
                clashes.append((first.name, second.name))
    return clashes


def load_room_spec(path: str | Path) -> RoomSpec:
    source = Path(path).expanduser()
    return parse_room_spec(json.loads(source.read_text(encoding="utf-8")))


__all__ = [
    "BUILD_SCHEMA",
    "COLORS",
    "COMPOSITES",
    "KEY_SCHEMA",
    "PRIMITIVES",
    "SHAPES",
    "SPEC_SCHEMA",
    "AuthoredObject",
    "Part",
    "RoomSpec",
    "load_room_spec",
    "parse_room_spec",
]
