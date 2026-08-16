"""Compile a hand-authored room description into a .blend, inside Blender.

This script is handed geometry only.  Its input JSON has instance ids, RGBA
colors and primitive boxes/cylinders/spheres -- never the words the author used
for the furniture.  Blender object names are opaque (``i_000003_p_01``), so the
saved scene cannot leak a category into anything that later renders it.

Run through Blender:

    blender --background --python blender/build_authored_room.py -- \
        --build data/spatial_lens/<room>/build.json \
        --output data/spatial_lens/<room>/scene.blend \
        --measured data/spatial_lens/<room>/measured_geometry.json
"""


from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import bpy

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scene_utils import (  # type: ignore[import-not-found]
    add_area_light,
    add_box,
    add_cylinder,
    add_uv_sphere,
    assign_instance,
    atomic_json,
    blender_cli_args,
    combined_bbox,
    create_material,
    reset_scene,
)

WALL_THICKNESS = 0.10


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--measured", required=True)
    return parser


def _shell(scene_size: list[float], floor_rgba, wall_rgba) -> None:
    width, depth, height = (float(value) for value in scene_size)
    floor_material = create_material("m_floor", tuple(floor_rgba), roughness=0.78)
    wall_material = create_material("m_wall", tuple(wall_rgba), roughness=0.88)
    ceiling_material = create_material("m_ceiling", (0.92, 0.92, 0.92, 1.0))
    half_x, half_y = width / 2.0, depth / 2.0

    # The ray-cast depth pass requires every mesh to carry an opaque numeric
    # instance index, so the shell gets reserved ids in the 900 block. They are
    # numbers, not names: nothing here says "floor" or "wall" to perception.
    shell = (
        ("floor", (0.0, 0.0, -WALL_THICKNESS / 2), (width, depth, WALL_THICKNESS), floor_material),
        (
            "ceiling",
            (0.0, 0.0, height + WALL_THICKNESS / 2),
            (width, depth, WALL_THICKNESS),
            ceiling_material,
        ),
        (
            "xn",
            (-half_x - WALL_THICKNESS / 2, 0.0, height / 2),
            (WALL_THICKNESS, depth, height),
            wall_material,
        ),
        (
            "xp",
            (half_x + WALL_THICKNESS / 2, 0.0, height / 2),
            (WALL_THICKNESS, depth, height),
            wall_material,
        ),
        (
            "yn",
            (0.0, -half_y - WALL_THICKNESS / 2, height / 2),
            (width, WALL_THICKNESS, height),
            wall_material,
        ),
        (
            "yp",
            (0.0, half_y + WALL_THICKNESS / 2, height / 2),
            (width, WALL_THICKNESS, height),
            wall_material,
        ),
    )
    for index, (suffix, center, size, material) in enumerate(shell):
        obj = add_box(f"shell_tmp_{suffix}", center, size, material)
        assign_instance([obj], f"i_000{901 + index}")
    # Two broad lights keep every wall usable for reconstruction; a single
    # centre lamp leaves the corners too dark for the vision encoder.
    for index, (x, y) in enumerate(((-width / 4, -depth / 4), (width / 4, depth / 4))):
        add_area_light(
            f"shell_light_{index}",
            (x, y, height - 0.25),
            energy=180.0 * (width * depth) / 20.0,
            size=max(width, depth) * 0.5,
            color=(1.0, 0.98, 0.94),
        )


def _instance(entry: dict[str, Any]) -> dict[str, Any]:
    instance_id = str(entry["instance_id"])
    rgba = tuple(float(value) for value in entry["rgba"])
    material = create_material(f"m_{instance_id}", rgba)
    parts = []
    for part in entry["parts"]:
        kind = str(part["kind"])
        center = tuple(float(value) for value in part["center_m"])
        size = tuple(float(value) for value in part["size_m"])
        yaw = float(part.get("yaw_degrees", 0.0))
        name = f"{instance_id}_tmp_{len(parts):02d}"
        if kind == "box":
            obj = add_box(name, center, size, material, rotation_degrees=(0.0, 0.0, yaw))
        elif kind == "cylinder":
            obj = add_cylinder(
                name,
                center,
                radius=max(size[0], size[1]) / 2.0,
                depth=size[2],
                material=material,
                rotation_degrees=(0.0, 0.0, yaw),
            )
        elif kind == "sphere":
            obj = add_uv_sphere(
                name, center, (size[0] / 2.0, size[1] / 2.0, size[2] / 2.0), material
            )
        else:
            raise ValueError(f"Unsupported primitive kind: {kind}")
        parts.append(obj)
    names = assign_instance(parts, instance_id)
    bbox_min, bbox_max = combined_bbox(names)
    return {
        "instance_id": instance_id,
        "part_names": names,
        "bbox": {"min_xyz_m": bbox_min, "max_xyz_m": bbox_max},
        "center_xyz_m": [
            (low + high) / 2.0 for low, high in zip(bbox_min, bbox_max)
        ],
        "dimensions_m": [high - low for low, high in zip(bbox_min, bbox_max)],
        "rgba": [float(value) for value in rgba],
    }


def main() -> None:
    args = _parser().parse_args(blender_cli_args())
    build = json.loads(Path(args.build).read_text(encoding="utf-8"))

    reset_scene()
    _shell(build["room_size_m"], build["floor_rgba"], build["wall_rgba"])
    measured = [_instance(entry) for entry in build["instances"]]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".blend1").unlink(missing_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(output), check_existing=False)

    # Measured geometry is ground truth for scoring perception. It carries no
    # author words -- only instance ids and the boxes Blender actually made.
    atomic_json(
        Path(args.measured),
        {
            "schema": "semantic_3d_chat.spatial_lens.measured_geometry.v1",
            "room_size_m": [float(value) for value in build["room_size_m"]],
            "instances": measured,
        },
    )
    print(
        json.dumps(
            {
                "blend": str(output),
                "instance_count": len(measured),
                "room_size_m": build["room_size_m"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
