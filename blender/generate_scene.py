"""Generate a deterministic furnished room and isolated oracle artifacts."""

# Blender does not add a ``--python`` script's directory to sys.path.  The
# path insertion below must therefore precede the local helper import.
# ruff: noqa: I001

from __future__ import annotations

import argparse
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import bpy

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from semantic_3d_chat.data.scene_variants import (
    COLOR_VARIANTS,
    LAYOUT_VARIANTS,
    ScenePlan,
    derive_scene_seed,
    validate_oracle_geometry,
)

from scene_utils import (
    add_area_light,
    add_box,
    add_cone,
    add_cylinder,
    add_torus,
    add_uv_sphere,
    assign_instance,
    atomic_json,
    blender_cli_args,
    config_hash,
    configure_render,
    create_camera,
    create_material,
    load_config,
    oracle_instance,
    point_camera_at,
    render_png,
    reset_scene,
    scene_paths,
    validate_scene_id,
)


COLORS: dict[str, tuple[float, float, float, float]] = {
    "warm_white": (0.82, 0.80, 0.73, 1.0),
    "light_gray": (0.62, 0.65, 0.67, 1.0),
    "charcoal": (0.10, 0.12, 0.14, 1.0),
    "wood": (0.40, 0.19, 0.07, 1.0),
    "dark_wood": (0.20, 0.08, 0.025, 1.0),
    "red": (0.78, 0.035, 0.025, 1.0),
    "blue": (0.025, 0.18, 0.78, 1.0),
    "yellow": (0.90, 0.58, 0.035, 1.0),
    "green": (0.035, 0.42, 0.10, 1.0),
    "terracotta": (0.58, 0.20, 0.09, 1.0),
    "cream": (0.88, 0.79, 0.57, 1.0),
    "glass_blue": (0.16, 0.46, 0.62, 1.0),
}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", default="configs/default.yaml")
    result.add_argument("--scene", default="scene_000001")
    result.add_argument("--seed", type=int)
    result.add_argument("--color-variant", choices=sorted(COLOR_VARIANTS))
    result.add_argument("--layout-variant", choices=sorted(LAYOUT_VARIANTS))
    result.add_argument("--remove-instance", action="append")
    result.add_argument("--pair-id")
    result.add_argument("--paired-scene")
    result.add_argument("--change-type")
    result.add_argument("--pair-role")
    return result


class RoomBuilder:
    def __init__(self, config: dict[str, Any], scene_id: str, plan: ScenePlan) -> None:
        self.config = config
        self.scene_id = scene_id
        self.plan = plan
        self.seed = plan.seed
        self.rng = random.Random(self.seed)
        self.scene = reset_scene()
        self.room_size = tuple(float(value) for value in config["scene"]["room_size_m"])
        self.jitter = float(config["scene"].get("object_jitter_m", 0.0))
        self.materials = {
            name: create_material(f"m_{index:04d}", COLORS[plan.resolved_color(name)])
            for index, name in enumerate(COLORS)
        }
        self.instances: list[dict[str, Any]] = []
        self.relationships: list[dict[str, str]] = []

    def _xy(self, x: float, y: float, *, amount: float = 1.0) -> tuple[float, float]:
        magnitude = self.jitter * amount
        resolved_x = x + self.rng.uniform(-magnitude, magnitude)
        resolved_y = y + self.rng.uniform(-magnitude, magnitude)
        if self.plan.mirrored:
            resolved_x = -resolved_x
        return resolved_x, resolved_y

    def _add_instance(
        self,
        *,
        instance_id: str,
        kind: str,
        category: str,
        color_name: str,
        parts: list[bpy.types.Object],
        support_surface: str | None,
        rotation_euler_degrees: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        part_names = assign_instance(parts, instance_id)
        resolved_color = self.plan.resolved_color(color_name)
        self.instances.append(
            oracle_instance(
                instance_id=instance_id,
                kind=kind,
                category=category,
                color_name=resolved_color,
                rgba=COLORS[resolved_color],
                part_names=part_names,
                support_surface=support_surface,
                rotation_euler_degrees=rotation_euler_degrees,
            )
        )

    def _relationship(self, subject: str, predicate: str, target: str) -> None:
        self.relationships.append(
            {
                "subject_instance_id": subject,
                "predicate": predicate,
                "object_instance_id": target,
            }
        )

    def build_room_shell(self) -> None:
        width, depth, height = self.room_size
        wall = self.materials["warm_white"]
        floor = self.materials["light_gray"]
        thickness = 0.10
        surface_specs = [
            ("i_000001", "floor", (0.0, 0.0, -thickness / 2), (width, depth, thickness), floor),
            (
                "i_000002",
                "wall",
                (-width / 2 - thickness / 2, 0.0, height / 2),
                (thickness, depth + 2 * thickness, height),
                wall,
            ),
            (
                "i_000003",
                "wall",
                (width / 2 + thickness / 2, 0.0, height / 2),
                (thickness, depth + 2 * thickness, height),
                wall,
            ),
            (
                "i_000004",
                "wall",
                (0.0, depth / 2 + thickness / 2, height / 2),
                (width + 2 * thickness, thickness, height),
                wall,
            ),
            (
                "i_000005",
                "wall",
                (0.0, -depth / 2 - thickness / 2, height / 2),
                (width + 2 * thickness, thickness, height),
                wall,
            ),
            (
                "i_000006",
                "ceiling",
                (0.0, 0.0, height + thickness / 2),
                (width, depth, thickness),
                wall,
            ),
        ]
        for instance_id, category, center, dimensions, material in surface_specs:
            part = add_box("pending", center, dimensions, material)
            self._add_instance(
                instance_id=instance_id,
                kind="surface",
                category=category,
                color_name="light_gray" if category == "floor" else "warm_white",
                parts=[part],
                support_surface=None,
            )

        # A visible inset door and handle.  It is a fixture, not a hole, so all
        # scan rays still terminate on room geometry.
        door_x = self.plan.mirror_x(-1.85)
        door_y = -depth / 2 + 0.012
        door_parts = [
            add_box(
                "pending",
                (door_x, door_y, 1.05),
                (0.96, 0.055, 2.10),
                self.materials["wood"],
            ),
            add_uv_sphere(
                "pending",
                (self.plan.offset_x(door_x, 0.34), door_y + 0.045, 1.02),
                (0.055, 0.035, 0.055),
                self.materials["yellow"],
            ),
        ]
        self._add_instance(
            instance_id="i_000010",
            kind="fixture",
            category="door",
            color_name="wood",
            parts=door_parts,
            support_surface="i_000005",
        )
        self._relationship("i_000010", "mounted_on", "i_000005")

    def build_table(self) -> tuple[float, float]:
        x, y = self._xy(0.85, 0.70)
        parts = [add_box("pending", (x, y, 0.76), (1.42, 0.82, 0.10), self.materials["wood"])]
        for dx in (-0.60, 0.60):
            for dy in (-0.30, 0.30):
                parts.append(
                    add_box(
                        "pending",
                        (x + dx, y + dy, 0.355),
                        (0.10, 0.10, 0.71),
                        self.materials["dark_wood"],
                    )
                )
        self._add_instance(
            instance_id="i_000100",
            kind="object",
            category="table",
            color_name="wood",
            parts=parts,
            support_surface="i_000001",
        )
        self._relationship("i_000100", "on", "i_000001")
        return x, y

    def build_chair(self) -> tuple[float, float]:
        x, y = self._xy(-1.20, 0.48)
        if self.plan.removes("i_000101"):
            return x, y
        parts = [
            add_box("pending", (x, y, 0.48), (0.54, 0.54, 0.09), self.materials["blue"]),
            add_box("pending", (x, y + 0.235, 0.88), (0.54, 0.075, 0.76), self.materials["blue"]),
        ]
        for dx in (-0.21, 0.21):
            for dy in (-0.21, 0.21):
                parts.append(
                    add_box(
                        "pending",
                        (x + dx, y + dy, 0.235),
                        (0.075, 0.075, 0.47),
                        self.materials["charcoal"],
                    )
                )
        self._add_instance(
            instance_id="i_000101",
            kind="object",
            category="chair",
            color_name="blue",
            parts=parts,
            support_surface="i_000001",
        )
        self._relationship("i_000101", "on", "i_000001")
        return x, y

    def build_picture_frame(self) -> None:
        if self.plan.removes("i_000102"):
            return
        width, depth, _ = self.room_size
        del width
        x = self.plan.mirror_x(-0.45)
        y = depth / 2 - 0.028
        z = 1.72
        border = self.materials["dark_wood"]
        parts = [
            add_box("pending", (x, y, z), (0.88, 0.045, 0.58), self.materials["yellow"]),
            add_box(
                "pending",
                (self.plan.offset_x(x, -0.49), y - 0.025, z),
                (0.10, 0.09, 0.78),
                border,
            ),
            add_box(
                "pending",
                (self.plan.offset_x(x, 0.49), y - 0.025, z),
                (0.10, 0.09, 0.78),
                border,
            ),
            add_box("pending", (x, y - 0.025, z - 0.34), (1.08, 0.09, 0.10), border),
            add_box("pending", (x, y - 0.025, z + 0.34), (1.08, 0.09, 0.10), border),
        ]
        self._add_instance(
            instance_id="i_000102",
            kind="object",
            category="picture frame",
            color_name="yellow",
            parts=parts,
            support_surface="i_000004",
        )
        self._relationship("i_000102", "mounted_on", "i_000004")

    def build_bowl(self) -> tuple[float, float]:
        x, y = self._xy(-1.46, -1.18, amount=0.6)
        if self.plan.removes("i_000103"):
            return x, y
        parts = [
            add_torus("pending", (x, y, 0.115), 0.20, 0.052, self.materials["red"]),
            add_cylinder("pending", (x, y, 0.035), 0.18, 0.05, self.materials["red"]),
        ]
        self._add_instance(
            instance_id="i_000103",
            kind="object",
            category="bowl",
            color_name="red",
            parts=parts,
            support_surface="i_000001",
        )
        self._relationship("i_000103", "on", "i_000001")
        return x, y

    def build_lamp(self) -> tuple[float, float]:
        x, y = self._xy(-2.16, 1.45, amount=0.5)
        if self.plan.removes("i_000104"):
            return x, y
        parts = [
            add_cylinder("pending", (x, y, 0.055), 0.30, 0.11, self.materials["charcoal"]),
            add_cylinder("pending", (x, y, 1.00), 0.045, 1.90, self.materials["charcoal"]),
            add_cone("pending", (x, y, 2.02), 0.37, 0.19, 0.46, self.materials["cream"]),
        ]
        self._add_instance(
            instance_id="i_000104",
            kind="object",
            category="floor lamp",
            color_name="cream",
            parts=parts,
            support_surface="i_000001",
        )
        self._relationship("i_000104", "on", "i_000001")
        return x, y

    def build_tabletop_objects(self, table_xy: tuple[float, float]) -> None:
        table_x, table_y = table_xy
        cube_z = 0.15 if self.plan.layout_variant == "cube_under" else 0.96
        cube_center = (self.plan.offset_x(table_x, -0.32), table_y + 0.02, cube_z)
        if not self.plan.removes("i_000105"):
            cube_parts = [
                add_box("pending", cube_center, (0.28, 0.28, 0.30), self.materials["red"])
            ]
            cube_support = (
                "i_000001" if self.plan.layout_variant == "cube_under" else "i_000100"
            )
            self._add_instance(
                instance_id="i_000105",
                kind="object",
                category="cube",
                color_name="red",
                parts=cube_parts,
                support_surface=cube_support,
            )
            if self.plan.layout_variant == "cube_under":
                self._relationship("i_000105", "on", "i_000001")
                self._relationship("i_000105", "under", "i_000100")
                self._relationship("i_000100", "above", "i_000105")
            else:
                self._relationship("i_000105", "on", "i_000100")
                self._relationship("i_000100", "under", "i_000105")

        if not self.plan.removes("i_000106"):
            book_parts = [
                add_box(
                    "pending",
                    (self.plan.offset_x(table_x, 0.34), table_y - 0.08, 0.845),
                    (0.38, 0.25, 0.07),
                    self.materials["green"],
                    rotation_degrees=(0.0, 0.0, -12.0 if self.plan.mirrored else 12.0),
                )
            ]
            book_rotation = (0.0, 0.0, -12.0 if self.plan.mirrored else 12.0)
            self._add_instance(
                instance_id="i_000106",
                kind="object",
                category="book",
                color_name="green",
                parts=book_parts,
                support_surface="i_000100",
                rotation_euler_degrees=book_rotation,
            )
            self._relationship("i_000106", "on", "i_000100")
            self._relationship("i_000100", "under", "i_000106")

    def build_cabinet(self) -> tuple[float, float]:
        x, y = self._xy(2.12, -1.22, amount=0.5)
        if self.plan.removes("i_000107"):
            return x, y
        parts = [
            add_box("pending", (x, y, 0.48), (0.78, 0.50, 0.96), self.materials["wood"]),
            add_box(
                "pending", (x, y - 0.265, 0.48), (0.68, 0.035, 0.84), self.materials["dark_wood"]
            ),
            add_uv_sphere(
                "pending",
                (self.plan.offset_x(x, 0.22), y - 0.30, 0.52),
                (0.045, 0.03, 0.045),
                self.materials["yellow"],
            ),
        ]
        self._add_instance(
            instance_id="i_000107",
            kind="object",
            category="cabinet",
            color_name="wood",
            parts=parts,
            support_surface="i_000001",
        )
        self._relationship("i_000107", "on", "i_000001")
        return x, y

    def build_plant(self) -> tuple[float, float]:
        x, y = self._xy(2.14, 1.57, amount=0.45)
        if self.plan.removes("i_000108"):
            return x, y
        parts = [
            add_cone("pending", (x, y, 0.25), 0.30, 0.24, 0.50, self.materials["terracotta"]),
            add_cylinder("pending", (x, y, 0.67), 0.045, 0.48, self.materials["green"]),
            add_uv_sphere(
                "pending",
                (self.plan.offset_x(x, -0.15), y, 0.92),
                (0.15, 0.07, 0.36),
                self.materials["green"],
            ),
            add_uv_sphere(
                "pending",
                (self.plan.offset_x(x, 0.15), y, 0.96),
                (0.15, 0.07, 0.38),
                self.materials["green"],
            ),
            add_uv_sphere(
                "pending", (x, y - 0.10, 1.05), (0.12, 0.08, 0.40), self.materials["green"]
            ),
        ]
        self._add_instance(
            instance_id="i_000108",
            kind="object",
            category="plant pot",
            color_name="terracotta",
            parts=parts,
            support_surface="i_000001",
        )
        self._relationship("i_000108", "on", "i_000001")
        return x, y

    def add_pairwise_relationships(self) -> None:
        objects = [entry for entry in self.instances if entry["kind"] == "object"]
        epsilon = 0.20
        for first_index, first in enumerate(objects):
            first_center = first["expected_center_xyz_m"]
            for second in objects[first_index + 1 :]:
                second_center = second["expected_center_xyz_m"]
                dx = first_center[0] - second_center[0]
                dy = first_center[1] - second_center[1]
                dz = first_center[2] - second_center[2]
                if abs(dx) > epsilon:
                    if dx < 0:
                        self._relationship(first["instance_id"], "left_of", second["instance_id"])
                        self._relationship(second["instance_id"], "right_of", first["instance_id"])
                    else:
                        self._relationship(first["instance_id"], "right_of", second["instance_id"])
                        self._relationship(second["instance_id"], "left_of", first["instance_id"])
                if abs(dy) > epsilon:
                    if dy > 0:
                        self._relationship(
                            first["instance_id"], "in_front_of", second["instance_id"]
                        )
                        self._relationship(second["instance_id"], "behind", first["instance_id"])
                    else:
                        self._relationship(first["instance_id"], "behind", second["instance_id"])
                        self._relationship(
                            second["instance_id"], "in_front_of", first["instance_id"]
                        )
                if abs(dz) > 0.35:
                    if dz > 0:
                        self._relationship(first["instance_id"], "above", second["instance_id"])
                        self._relationship(second["instance_id"], "below", first["instance_id"])
                    else:
                        self._relationship(first["instance_id"], "below", second["instance_id"])
                        self._relationship(second["instance_id"], "above", first["instance_id"])
                planar_distance = math.hypot(dx, dy)
                if planar_distance < 1.10:
                    self._relationship(first["instance_id"], "near", second["instance_id"])
                    self._relationship(second["instance_id"], "near", first["instance_id"])

    def add_lighting(self) -> None:
        if self.scene.world is None:
            self.scene.world = bpy.data.worlds.new("w_0001")
        self.scene.world.color = (0.045, 0.045, 0.045)
        add_area_light(
            "l_0001",
            (self.plan.mirror_x(-1.45), -0.90, 2.82),
            energy=310.0,
            size=2.5,
            color=(1.0, 0.84, 0.68),
        )
        add_area_light(
            "l_0002",
            (self.plan.mirror_x(1.55), 1.05, 2.75),
            energy=270.0,
            size=2.2,
            color=(0.72, 0.84, 1.0),
        )

    def build(self) -> dict[str, Any]:
        self.build_room_shell()
        table_xy = self.build_table()
        self.build_chair()
        self.build_picture_frame()
        self.build_bowl()
        self.build_lamp()
        self.build_tabletop_objects(table_xy)
        self.build_cabinet()
        self.build_plant()
        self.add_pairwise_relationships()
        self.add_lighting()
        width, depth, height = self.room_size
        oracle = {
            "schema_version": 1,
            "scene_id": self.scene_id,
            "seed": self.seed,
            "coordinate_system": {
                "units": "meters",
                "x_axis": "right",
                "y_axis": "forward",
                "z_axis": "up",
                "camera_axes": "x_right_y_down_z_forward",
            },
            "room": {
                "size_m": [width, depth, height],
                "bounds_min_m": [-width / 2, -depth / 2, 0.0],
                "bounds_max_m": [width / 2, depth / 2, height],
            },
            "instances": self.instances,
            "relationships": self.relationships,
        }
        render = self.config["render"]
        validation = validate_oracle_geometry(
            oracle,
            camera_position_m=tuple(float(value) for value in render["camera_position_m"]),
            pitch_degrees=tuple(float(value) for value in render["pitch_degrees"]),
            horizontal_fov_degrees=float(render["horizontal_fov_degrees"]),
            image_size=tuple(int(value) for value in render["resolution"]),
        )
        # Preserve scene_000001's original oracle bytes.  New scene records get
        # explicit oracle-only generation and validation metadata.
        preserve_original_scene_one = (
            self.plan.is_default_scene_one and self.plan.seed == int(self.config["seed"])
        )
        if not preserve_original_scene_one:
            oracle["generation"] = self.plan.oracle_metadata()
            oracle["validation"] = validation
        return oracle


def _scene_plan(config: dict[str, Any], args: argparse.Namespace, scene_id: str) -> ScenePlan:
    scene_config = config["scene"]
    legacy_variant = str(scene_config.get("variant", "base"))
    presets = {
        "base": ("base", "base", ()),
        "color_swap": ("swap_red_blue", "base", ()),
        "cube_on": ("base", "cube_on", ()),
        "cube_under": ("base", "cube_under", ()),
        "mirror_lr": ("base", "mirror_lr", ()),
        "remove_book": ("base", "base", ("i_000106",)),
    }
    if legacy_variant not in presets:
        raise ValueError(f"Unknown scene.variant preset: {legacy_variant}")
    preset_color, preset_layout, preset_removals = presets[legacy_variant]
    configured_removals = scene_config.get("remove_instance_ids", preset_removals)
    if not isinstance(configured_removals, (list, tuple)):
        raise TypeError("scene.remove_instance_ids must be a list")
    removals = (
        tuple(args.remove_instance)
        if args.remove_instance is not None
        else tuple(str(value) for value in configured_removals)
    )
    return ScenePlan(
        scene_id=scene_id,
        seed=derive_scene_seed(
            int(config["seed"]),
            scene_id,
            seed_stride=int(scene_config.get("seed_stride", 1)),
            explicit_seed=args.seed,
        ),
        color_variant=str(
            args.color_variant
            or scene_config.get("color_variant", preset_color)
        ),
        layout_variant=str(
            args.layout_variant
            or scene_config.get("layout_variant", preset_layout)
        ),
        remove_instance_ids=removals,
        pair_id=args.pair_id,
        paired_scene_id=args.paired_scene,
        change_type=args.change_type,
        pair_role=args.pair_role,
    )


def main() -> None:
    started = time.perf_counter()
    args = parser().parse_args(blender_cli_args())
    scene_id = validate_scene_id(args.scene)
    config, _ = load_config(args.config)
    plan = _scene_plan(config, args, scene_id)
    paths = scene_paths(config, scene_id)
    builder = RoomBuilder(config, scene_id, plan)
    oracle = builder.build()

    render = config["render"]
    width, height = (int(value) for value in render["resolution"])
    resolved_engine = configure_render(
        builder.scene,
        width=width,
        height=height,
        engine=str(render["engine"]),
        samples=int(render["samples"]),
    )
    camera = create_camera("c_preview", 78.0)
    camera.location = (2.48, -2.02, 2.55)
    point_camera_at(camera, (0.15, 0.28, 0.72))
    builder.scene.camera = camera
    render_png(builder.scene, paths["rendered"] / "p_000000.png")

    # Save only the semantic oracle and Blender source in the isolated oracle
    # tree.  The runtime tree receives pixels and a sanitized manifest later.
    atomic_json(paths["oracle"] / "oracle.json", oracle)
    blend_path = paths["oracle"] / "scene.blend"
    bpy.context.preferences.filepaths.save_version = 0
    blend_path.with_suffix(".blend1").unlink(missing_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path), check_existing=False)

    elapsed = time.perf_counter() - started
    print(
        "SCENE_GENERATED "
        f"scene={scene_id} instances={len(oracle['instances'])} "
        f"relationships={len(oracle['relationships'])} engine={resolved_engine} "
        f"seed={plan.seed} color_variant={plan.color_variant} "
        f"layout_variant={plan.layout_variant} "
        f"config={config_hash(config)} seconds={elapsed:.3f} "
        f"blend={blend_path}"
    )


if __name__ == "__main__":
    main()
