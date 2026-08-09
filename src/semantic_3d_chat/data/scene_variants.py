"""Deterministic scene plans and oracle-only counterfactual controls.

This module deliberately uses only the Python standard library so Blender's
bundled interpreter and the project's normal test environment share one source
of truth.  Nothing here is imported by chat inference.
"""

from __future__ import annotations

import copy
import math
import re
from dataclasses import dataclass
from typing import Any

SCENE_ID_PATTERN = re.compile(r"scene_[0-9]{6}")
PAIR_ID_PATTERN = re.compile(r"pair_[0-9]{6}")
INSTANCE_ID_PATTERN = re.compile(r"i_[0-9]{6}")

COLOR_VARIANTS = frozenset({"base", "swap_red_blue"})
LAYOUT_VARIANTS = frozenset({"base", "cube_on", "cube_under", "mirror_lr"})
PAIR_ROLES = frozenset({"reference", "counterfactual"})
CHANGE_TYPES = frozenset({"color_swap", "cube_support", "mirror_lr", "object_removal"})
REMOVABLE_INSTANCE_IDS = frozenset(
    {
        "i_000101",  # chair
        "i_000102",  # picture frame
        "i_000103",  # bowl
        "i_000104",  # floor lamp
        "i_000105",  # cube
        "i_000106",  # book
        "i_000107",  # cabinet
        "i_000108",  # plant pot
    }
)


def validate_scene_id(scene_id: str) -> str:
    if not SCENE_ID_PATTERN.fullmatch(scene_id):
        raise ValueError("scene ID must match scene_ followed by six digits")
    return scene_id


def scene_number(scene_id: str) -> int:
    validate_scene_id(scene_id)
    return int(scene_id.removeprefix("scene_"))


def derive_scene_seed(
    base_seed: int,
    scene_id: str,
    *,
    seed_stride: int = 1,
    explicit_seed: int | None = None,
) -> int:
    """Derive stable variation while preserving scene_000001 exactly."""

    if explicit_seed is not None:
        if explicit_seed < 0:
            raise ValueError("scene seed must be non-negative")
        return int(explicit_seed)
    if seed_stride < 1:
        raise ValueError("seed_stride must be positive")
    return int(base_seed) + (scene_number(scene_id) - 1) * int(seed_stride)


@dataclass(frozen=True)
class ScenePlan:
    """All semantic generation controls, stored only in the oracle tree."""

    scene_id: str
    seed: int
    color_variant: str = "base"
    layout_variant: str = "base"
    remove_instance_ids: tuple[str, ...] = ()
    pair_id: str | None = None
    paired_scene_id: str | None = None
    change_type: str | None = None
    pair_role: str | None = None

    def __post_init__(self) -> None:
        validate_scene_id(self.scene_id)
        if self.seed < 0:
            raise ValueError("scene seed must be non-negative")
        if self.color_variant not in COLOR_VARIANTS:
            raise ValueError(f"Unknown color variant: {self.color_variant}")
        if self.layout_variant not in LAYOUT_VARIANTS:
            raise ValueError(f"Unknown layout variant: {self.layout_variant}")
        removals = tuple(sorted(set(self.remove_instance_ids)))
        if any(not INSTANCE_ID_PATTERN.fullmatch(value) for value in removals):
            raise ValueError("removed instances must use opaque i_XXXXXX IDs")
        unsupported = set(removals) - REMOVABLE_INSTANCE_IDS
        if unsupported:
            raise ValueError(f"Unsupported removable instance IDs: {sorted(unsupported)}")
        object.__setattr__(self, "remove_instance_ids", removals)

        pair_fields = (self.pair_id, self.paired_scene_id, self.change_type, self.pair_role)
        if any(value is not None for value in pair_fields) and not all(
            value is not None for value in pair_fields
        ):
            raise ValueError("Counterfactual pair metadata must be provided together")
        if self.pair_id is not None:
            if not PAIR_ID_PATTERN.fullmatch(self.pair_id):
                raise ValueError("pair ID must match pair_ followed by six digits")
            validate_scene_id(str(self.paired_scene_id))
            if self.paired_scene_id == self.scene_id:
                raise ValueError("paired_scene_id must differ from scene_id")
            if self.change_type not in CHANGE_TYPES:
                raise ValueError(f"Unknown counterfactual change type: {self.change_type}")
            if self.pair_role not in PAIR_ROLES:
                raise ValueError(f"Unknown counterfactual role: {self.pair_role}")

    @property
    def is_default_scene_one(self) -> bool:
        return (
            self.scene_id == "scene_000001"
            and self.color_variant == "base"
            and self.layout_variant == "base"
            and not self.remove_instance_ids
            and self.pair_id is None
        )

    @property
    def mirrored(self) -> bool:
        return self.layout_variant == "mirror_lr"

    @property
    def control_signature(self) -> tuple[str, str, tuple[str, ...]]:
        return self.color_variant, self.layout_variant, self.remove_instance_ids

    def resolved_color(self, color_name: str) -> str:
        if self.color_variant != "swap_red_blue":
            return color_name
        return {"red": "blue", "blue": "red"}.get(color_name, color_name)

    def mirror_x(self, x: float) -> float:
        return -float(x) if self.mirrored else float(x)

    def offset_x(self, x: float, offset: float) -> float:
        return float(x) - float(offset) if self.mirrored else float(x) + float(offset)

    def removes(self, instance_id: str) -> bool:
        return instance_id in self.remove_instance_ids

    def oracle_metadata(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "seed": self.seed,
            "color_variant": self.color_variant,
            "layout_variant": self.layout_variant,
            "removed_instance_ids": list(self.remove_instance_ids),
        }
        if self.pair_id is not None:
            result["counterfactual_pair"] = {
                "pair_id": self.pair_id,
                "paired_scene_id": self.paired_scene_id,
                "change_type": self.change_type,
                "role": self.pair_role,
                "held_constant": [
                    "seed",
                    "room_dimensions",
                    "object_jitter",
                    "lighting",
                    "camera_scan",
                ],
            }
        return result


def scene_plan_from_mapping(
    scene_id: str,
    values: dict[str, Any],
    *,
    base_seed: int,
    seed_stride: int,
) -> ScenePlan:
    explicit_seed = values.get("seed")
    if explicit_seed is None and "seed_offset" in values:
        explicit_seed = int(base_seed) + int(values["seed_offset"])
    raw_removals = values.get("remove_instance_ids", [])
    if raw_removals is None:
        raw_removals = []
    if not isinstance(raw_removals, (list, tuple)):
        raise TypeError(f"remove_instance_ids for {scene_id} must be a list")
    return ScenePlan(
        scene_id=scene_id,
        seed=derive_scene_seed(
            base_seed,
            scene_id,
            seed_stride=seed_stride,
            explicit_seed=None if explicit_seed is None else int(explicit_seed),
        ),
        color_variant=str(values.get("color_variant", "base")),
        layout_variant=str(values.get("layout_variant", "base")),
        remove_instance_ids=tuple(str(value) for value in raw_removals),
        pair_id=None if values.get("pair_id") is None else str(values["pair_id"]),
        paired_scene_id=(
            None if values.get("paired_scene_id") is None else str(values["paired_scene_id"])
        ),
        change_type=None if values.get("change_type") is None else str(values["change_type"]),
        pair_role=None if values.get("pair_role") is None else str(values["pair_role"]),
    )


def batch_scene_plans(config: dict[str, Any]) -> tuple[ScenePlan, ...]:
    batch = config.get("batch")
    if not isinstance(batch, dict):
        raise TypeError("Batch configuration requires a batch mapping")
    raw_scenes = batch.get("scenes")
    if not isinstance(raw_scenes, dict) or not raw_scenes:
        raise ValueError("batch.scenes must be a non-empty mapping")
    base_seed = int(config["seed"])
    seed_stride = int(batch.get("seed_stride", 1009))
    plans = tuple(
        scene_plan_from_mapping(
            str(scene_id),
            values,
            base_seed=base_seed,
            seed_stride=seed_stride,
        )
        for scene_id, values in raw_scenes.items()
        if isinstance(values, dict)
    )
    invalid_entries = [
        str(scene_id) for scene_id, values in raw_scenes.items() if not isinstance(values, dict)
    ]
    if invalid_entries:
        raise TypeError(f"batch scene entries must be mappings: {invalid_entries}")
    if len({plan.scene_id for plan in plans}) != len(plans):
        raise ValueError("batch.scenes contains duplicate scene IDs")
    _validate_counterfactual_pairs(plans)
    return plans


def _validate_counterfactual_pairs(plans: tuple[ScenePlan, ...]) -> None:
    groups: dict[str, list[ScenePlan]] = {}
    for plan in plans:
        if plan.pair_id is not None:
            groups.setdefault(plan.pair_id, []).append(plan)
    expected_control = {
        "color_swap": "color_variant",
        "cube_support": "layout_variant",
        "mirror_lr": "layout_variant",
        "object_removal": "remove_instance_ids",
    }
    for pair_id, members in groups.items():
        if len(members) != 2:
            raise ValueError(f"{pair_id} must contain exactly two scenes")
        first, second = members
        if first.seed != second.seed:
            raise ValueError(f"{pair_id} members must use the same seed")
        if first.change_type != second.change_type:
            raise ValueError(f"{pair_id} members disagree on change_type")
        if {first.pair_role, second.pair_role} != PAIR_ROLES:
            raise ValueError(f"{pair_id} needs one reference and one counterfactual")
        if first.paired_scene_id != second.scene_id or second.paired_scene_id != first.scene_id:
            raise ValueError(f"{pair_id} paired scene references must be reciprocal")
        controls = ("color_variant", "layout_variant", "remove_instance_ids")
        changed = {
            name for name in controls if getattr(first, name) != getattr(second, name)
        }
        expected = {expected_control[str(first.change_type)]}
        if changed != expected:
            raise ValueError(
                f"{pair_id} must change only {sorted(expected)}, changed {sorted(changed)}"
            )


def oracle_control_facts(oracle: dict[str, Any]) -> dict[str, Any]:
    """Extract the exact facts changed by the initial counterfactual controls."""

    instances = {
        entry["instance_id"]: entry
        for entry in oracle.get("instances", [])
        if entry.get("kind") == "object"
    }
    cube = instances.get("i_000105")
    return {
        "present_instance_ids": sorted(instances),
        "colors": {
            instance_id: entry["color"]["name"]
            for instance_id, entry in sorted(instances.items())
        },
        "center_x_m": {
            instance_id: float(entry["expected_center_xyz_m"][0])
            for instance_id, entry in sorted(instances.items())
        },
        "cube_support_surface": None if cube is None else cube.get("support_surface"),
        "relationships": sorted(
            (
                relationship["subject_instance_id"],
                relationship["predicate"],
                relationship["object_instance_id"],
            )
            for relationship in oracle.get("relationships", [])
        ),
    }


def project_oracle_counterfactual(
    reference_oracle: dict[str, Any],
    counterfactual_plan: ScenePlan,
) -> dict[str, Any]:
    """Project oracle facts for deterministic unit tests and dataset checks.

    Blender remains the source of rendered geometry.  This oracle-only helper
    makes the expected paired delta independently inspectable without exposing
    any of it to the runtime manifest.
    """

    result = copy.deepcopy(reference_oracle)
    result["scene_id"] = counterfactual_plan.scene_id
    result["seed"] = counterfactual_plan.seed
    result["generation"] = counterfactual_plan.oracle_metadata()

    if counterfactual_plan.color_variant == "swap_red_blue":
        for instance in result.get("instances", []):
            color = instance.get("color", {})
            old_name = color.get("name")
            new_name = counterfactual_plan.resolved_color(str(old_name))
            if new_name != old_name:
                color["name"] = new_name
                color["rgba"] = {
                    "red": [0.78, 0.035, 0.025, 1.0],
                    "blue": [0.025, 0.18, 0.78, 1.0],
                }[new_name]

    if counterfactual_plan.mirrored:
        for instance in result.get("instances", []):
            for key in ("expected_center_xyz_m",):
                if key in instance:
                    instance[key][0] = -float(instance[key][0])
            pose = instance.get("pose", {})
            if "center_xyz_m" in pose:
                pose["center_xyz_m"][0] = -float(pose["center_xyz_m"][0])
            bbox = instance.get("bbox", {})
            if "min_xyz_m" in bbox and "max_xyz_m" in bbox:
                old_min = float(bbox["min_xyz_m"][0])
                old_max = float(bbox["max_xyz_m"][0])
                bbox["min_xyz_m"][0] = -old_max
                bbox["max_xyz_m"][0] = -old_min
        for relationship in result.get("relationships", []):
            if relationship.get("predicate") == "left_of":
                relationship["predicate"] = "right_of"
            elif relationship.get("predicate") == "right_of":
                relationship["predicate"] = "left_of"

    if counterfactual_plan.layout_variant == "cube_under":
        for instance in result.get("instances", []):
            if instance.get("instance_id") == "i_000105":
                instance["support_surface"] = "i_000001"
                center_z = 0.15
                instance["expected_center_xyz_m"][2] = center_z
                instance["pose"]["center_xyz_m"][2] = center_z
                instance["bbox"]["min_xyz_m"][2] = 0.0
                instance["bbox"]["max_xyz_m"][2] = 0.30
        relationships = [
            relationship
            for relationship in result.get("relationships", [])
            if not (
                relationship.get("subject_instance_id") in {"i_000100", "i_000105"}
                and relationship.get("object_instance_id") in {"i_000100", "i_000105"}
            )
        ]
        relationships.extend(
            [
                {
                    "subject_instance_id": "i_000105",
                    "predicate": "on",
                    "object_instance_id": "i_000001",
                },
                {
                    "subject_instance_id": "i_000105",
                    "predicate": "under",
                    "object_instance_id": "i_000100",
                },
                {
                    "subject_instance_id": "i_000100",
                    "predicate": "above",
                    "object_instance_id": "i_000105",
                },
            ]
        )
        result["relationships"] = relationships

    removals = set(counterfactual_plan.remove_instance_ids)
    if removals:
        result["instances"] = [
            instance
            for instance in result.get("instances", [])
            if instance.get("instance_id") not in removals
        ]
        result["relationships"] = [
            relationship
            for relationship in result.get("relationships", [])
            if relationship.get("subject_instance_id") not in removals
            and relationship.get("object_instance_id") not in removals
        ]
    return result


def validate_oracle_geometry(
    oracle: dict[str, Any],
    *,
    camera_position_m: tuple[float, float, float],
    pitch_degrees: tuple[float, ...],
    horizontal_fov_degrees: float,
    image_size: tuple[int, int],
    tolerance_m: float = 0.015,
) -> dict[str, Any]:
    """Pragmatic inside-room, AABB-separation, and scan-coverage checks."""

    room = oracle["room"]
    room_min = tuple(float(value) for value in room["bounds_min_m"])
    room_max = tuple(float(value) for value in room["bounds_max_m"])
    objects = [entry for entry in oracle["instances"] if entry.get("kind") == "object"]
    if not objects:
        raise ValueError("Scene has no object instances")

    minimum_clearance = float("inf")
    for entry in objects:
        bbox_min = tuple(float(value) for value in entry["bbox"]["min_xyz_m"])
        bbox_max = tuple(float(value) for value in entry["bbox"]["max_xyz_m"])
        for axis in range(3):
            if bbox_min[axis] < room_min[axis] - tolerance_m:
                raise ValueError(f"{entry['instance_id']} extends below room bound on axis {axis}")
            if bbox_max[axis] > room_max[axis] + tolerance_m:
                raise ValueError(f"{entry['instance_id']} extends above room bound on axis {axis}")
            minimum_clearance = min(
                minimum_clearance,
                bbox_min[axis] - room_min[axis],
                room_max[axis] - bbox_max[axis],
            )

    checked_pairs = 0
    for first_index, first in enumerate(objects):
        for second in objects[first_index + 1 :]:
            if "table" in {first.get("category"), second.get("category")}:
                # A table's combined AABB fills the free volume between legs.
                continue
            if first.get("support_surface") == second.get("instance_id") or second.get(
                "support_surface"
            ) == first.get("instance_id"):
                continue
            checked_pairs += 1
            first_min = first["bbox"]["min_xyz_m"]
            first_max = first["bbox"]["max_xyz_m"]
            second_min = second["bbox"]["min_xyz_m"]
            second_max = second["bbox"]["max_xyz_m"]
            overlap = [
                min(float(first_max[axis]), float(second_max[axis]))
                - max(float(first_min[axis]), float(second_min[axis]))
                for axis in range(3)
            ]
            if all(value > tolerance_m for value in overlap):
                raise ValueError(
                    f"Object AABBs intersect: {first['instance_id']} and {second['instance_id']}"
                )

    width, height = image_size
    if width < 1 or height < 1:
        raise ValueError("image_size must be positive")
    horizontal_fov = math.radians(float(horizontal_fov_degrees))
    vertical_fov = 2.0 * math.atan(math.tan(horizontal_fov / 2.0) * height / width)
    pitch_min = math.radians(min(pitch_degrees)) - vertical_fov / 2.0
    pitch_max = math.radians(max(pitch_degrees)) + vertical_fov / 2.0
    for entry in objects:
        center = entry["expected_center_xyz_m"]
        dx = float(center[0]) - camera_position_m[0]
        dy = float(center[1]) - camera_position_m[1]
        dz = float(center[2]) - camera_position_m[2]
        planar = math.hypot(dx, dy)
        distance = math.sqrt(dx * dx + dy * dy + dz * dz)
        elevation = math.atan2(dz, planar)
        if distance <= 0.10 or not pitch_min <= elevation <= pitch_max:
            raise ValueError(f"{entry['instance_id']} is outside center-scan angular coverage")
        if not bool(entry.get("visible_from_center_scan", False)):
            raise ValueError(f"{entry['instance_id']} is not marked visible from center scan")

    return {
        "inside_room": True,
        "nonintersection": True,
        "center_scan_angular_coverage": True,
        "object_count": len(objects),
        "checked_object_pairs": checked_pairs,
        "minimum_room_clearance_m": float(minimum_clearance),
    }


__all__ = [
    "CHANGE_TYPES",
    "COLOR_VARIANTS",
    "LAYOUT_VARIANTS",
    "REMOVABLE_INSTANCE_IDS",
    "ScenePlan",
    "batch_scene_plans",
    "derive_scene_seed",
    "oracle_control_facts",
    "project_oracle_counterfactual",
    "scene_plan_from_mapping",
    "validate_oracle_geometry",
]
