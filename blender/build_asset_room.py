"""Assemble a room from downloaded glTF furniture, inside Blender.

Like the primitive builder, this script is handed geometry only: instance ids,
positions, rotations, scales and a path to a mesh. It is never told what any of
the meshes are. Blender object names stay opaque, so nothing downstream can read
a category out of the saved scene.

Two things differ from the primitive path. Each asset arrives at its own
authored size, so it is rescaled to the metres the composer asked for rather
than trusted. And an asset can import as many meshes, which all take the same
instance index, because "the chair" has to be one thing to the depth pass.

Run through Blender:

    blender --background --python blender/build_asset_room.py -- \
        --build data/spatial_lens/<room>/build.json \
        --assets data/assets \
        --output data/spatial_lens/<room>/scene.blend \
        --measured data/spatial_lens/<room>/measured_geometry.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import bpy
import mathutils

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scene_utils import (  # type: ignore[import-not-found]
    add_area_light,
    add_box,
    assign_instance,
    atomic_json,
    blender_cli_args,
    combined_bbox,
    create_material,
    reset_scene,
)

WALL_THICKNESS = 0.10
FLOOR_RGBA = (0.55, 0.52, 0.48, 1.0)
WALL_RGBA = (0.80, 0.78, 0.74, 1.0)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", required=True)
    parser.add_argument("--assets", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--measured", required=True)
    return parser


def _shell(scene_size: list[float]) -> None:
    width, depth, height = (float(value) for value in scene_size)
    floor_material = create_material("m_floor", FLOOR_RGBA, roughness=0.78)
    wall_material = create_material("m_wall", WALL_RGBA, roughness=0.88)
    ceiling_material = create_material("m_ceiling", (0.92, 0.92, 0.92, 1.0))
    half_x, half_y = width / 2.0, depth / 2.0

    # Reserved 900-block ids: the depth pass needs every mesh to carry an opaque
    # numeric instance index, and these are numbers, not names.
    shell = (
        ("floor", (0.0, 0.0, -WALL_THICKNESS / 2), (width, depth, WALL_THICKNESS), floor_material),
        ("ceiling", (0.0, 0.0, height + WALL_THICKNESS / 2),
         (width, depth, WALL_THICKNESS), ceiling_material),
        ("xn", (-half_x - WALL_THICKNESS / 2, 0.0, height / 2),
         (WALL_THICKNESS, depth, height), wall_material),
        ("xp", (half_x + WALL_THICKNESS / 2, 0.0, height / 2),
         (WALL_THICKNESS, depth, height), wall_material),
        ("yn", (0.0, -half_y - WALL_THICKNESS / 2, height / 2),
         (width, WALL_THICKNESS, height), wall_material),
        ("yp", (0.0, half_y + WALL_THICKNESS / 2, height / 2),
         (width, WALL_THICKNESS, height), wall_material),
    )
    for index, (suffix, center, size, material) in enumerate(shell):
        obj = add_box(f"shell_tmp_{suffix}", center, size, material)
        assign_instance([obj], f"i_000{901 + index}")

    for index, (x, y) in enumerate(((-width / 4, -depth / 4), (width / 4, depth / 4))):
        add_area_light(
            f"shell_light_{index}",
            (x, y, height - 0.25),
            # A third of what the primitive rooms used. Those were dark
            # painted boxes; photographed furniture against pale walls blows
            # out to flat white at that budget, and a white frame carries no
            # information for the vision encoder to fuse.
            energy=60.0 * (width * depth) / 20.0,
            size=max(width, depth) * 0.5,
            color=(1.0, 0.98, 0.94),
        )


def _import_gltf(path: Path) -> list[bpy.types.Object]:
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=str(path))
    fresh = [obj for obj in bpy.data.objects if obj not in before]
    return [obj for obj in fresh if obj.type == "MESH"]


def _world_bounds(objects: list[bpy.types.Object]) -> tuple[list[float], list[float]]:
    bpy.context.view_layer.update()
    corners = []
    for obj in objects:
        corners.extend(
            obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box
        )
    minimum = [min(c[axis] for c in corners) for axis in range(3)]
    maximum = [max(c[axis] for c in corners) for axis in range(3)]
    return minimum, maximum


def _place(entry: dict[str, Any], assets_root: Path, index: int) -> dict[str, Any]:
    instance_id = str(entry["instance_id"])
    source = assets_root / str(entry["gltf"])
    if not source.is_file():
        raise FileNotFoundError(f"missing asset mesh: {source}")
    meshes = _import_gltf(source)
    if not meshes:
        raise ValueError(f"{source} imported no meshes")

    # Detach from whatever parenting the glTF brought, so the transforms below
    # are the only thing deciding where this ends up.
    for obj in meshes:
        if obj.parent is not None:
            matrix = obj.matrix_world.copy()
            obj.parent = None
            obj.matrix_world = matrix

    minimum, maximum = _world_bounds(meshes)
    extent = [maximum[axis] - minimum[axis] for axis in range(3)]
    wanted = [float(v) for v in entry["size_m"]]
    # Rescale to the size the composer asked for rather than trusting the file:
    # one factor for all axes, so nothing is stretched out of proportion.
    ratios = [wanted[axis] / extent[axis] for axis in range(3) if extent[axis] > 1e-6]
    factor = min(ratios) if ratios else 1.0

    centre_x = (minimum[0] + maximum[0]) / 2.0
    centre_y = (minimum[1] + maximum[1]) / 2.0
    base_z = minimum[2]
    yaw = math.radians(float(entry["yaw_degrees"]))
    target = [float(v) for v in entry["position_m"]]

    pivot = mathutils.Vector((centre_x, centre_y, base_z))
    rotation = mathutils.Matrix.Rotation(yaw, 4, "Z")
    scaling = mathutils.Matrix.Diagonal((factor, factor, factor, 1.0))
    move = mathutils.Matrix.Translation(mathutils.Vector(target))
    transform = move @ rotation @ scaling @ mathutils.Matrix.Translation(-pivot)
    for obj in meshes:
        obj.matrix_world = transform @ obj.matrix_world

    names = assign_instance(meshes, instance_id)
    low, high = combined_bbox(names)
    return {
        "instance_id": instance_id,
        "instance_index": index,
        "bbox_min_m": [round(float(v), 5) for v in low],
        "bbox_max_m": [round(float(v), 5) for v in high],
        "part_count": len(names),
    }


def main() -> None:
    args = _parser().parse_args(blender_cli_args())
    build = json.loads(Path(args.build).read_text(encoding="utf-8"))
    assets_root = Path(args.assets)

    reset_scene()
    _shell(build["room_size_m"])
    measured = [
        _place(entry, assets_root, index)
        for index, entry in enumerate(build["objects"], start=1)
    ]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".blend1").unlink(missing_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(output), check_existing=False)

    atomic_json(
        Path(args.measured),
        {
            "schema": "semantic_3d_chat.spatial_lens.measured_geometry.v1",
            "room_size_m": [float(value) for value in build["room_size_m"]],
            "instances": measured,
        },
    )
    print(json.dumps({"blend": str(output), "instance_count": len(measured)}, indent=2))


if __name__ == "__main__":
    main()
