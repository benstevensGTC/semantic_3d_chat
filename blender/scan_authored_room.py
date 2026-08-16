"""Scan an authored room into RGB + metric depth + exact camera poses.

This is the only place the environment is observed.  It emits pixels and
geometry and nothing else: no instance ids, no visibility counts, no category
of any kind.  Everything downstream has to work from these images.

Poses are laid out on a ring plus the room centre, each looking slightly down
and sweeping several yaws, which is enough to cover the walls and the middle of
an ordinary room.

Run through Blender with the room's .blend already open:

    blender --background data/spatial_lens/<room>/scene.blend \
        --python blender/scan_authored_room.py -- \
        --output data/spatial_lens/<room>/scans --room-size 6 5 2.8
"""

# ruff: noqa: I001

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import bpy
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scene_utils import (  # type: ignore[import-not-found]
    atomic_json,
    blender_cli_args,
    camera_intrinsics,
    camera_to_world_cv,
    configure_render,
    create_camera,
    render_png,
    set_camera_yaw_pitch,
)
from render_scan import (  # type: ignore[import-not-found]
    axial_depth_from_bvh,
    build_world_bvh,
    pixel_rays_cv,
    sanitize_scene_for_scan,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--room-size", nargs=3, type=float, required=True)
    parser.add_argument("--resolution", nargs=2, type=int, default=[448, 448])
    parser.add_argument("--fov-degrees", type=float, default=72.0)
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--engine", default="BLENDER_EEVEE")
    parser.add_argument("--camera-height", type=float, default=1.15)
    parser.add_argument("--ring-count", type=int, default=8)
    parser.add_argument("--yaws-per-station", type=int, default=3)
    parser.add_argument("--pitch-degrees", type=float, default=-14.0)
    return parser


def _stations(width: float, depth: float, height: float, ring: int) -> list[tuple[float, float]]:
    """Ring of standing points plus the centre, all inside the walls."""

    margin = 0.9
    half_x = max(width / 2.0 - margin, 0.2)
    half_y = max(depth / 2.0 - margin, 0.2)
    points: list[tuple[float, float]] = [(0.0, 0.0)]
    for index in range(ring):
        angle = 2.0 * math.pi * index / ring
        points.append((half_x * math.cos(angle), half_y * math.sin(angle)))
    return points


def main() -> None:
    args = _parser().parse_args(blender_cli_args())
    width_px, height_px = (int(value) for value in args.resolution)
    room_w, room_d, room_h = (float(value) for value in args.room_size)

    scene = bpy.context.scene
    sanitize_scene_for_scan(scene)
    configure_render(
        scene,
        width=width_px,
        height=height_px,
        engine=str(args.engine),
        samples=int(args.samples),
    )
    camera = create_camera("c_lens_scan", float(args.fov_degrees))
    scene.camera = camera
    intrinsics = camera_intrinsics(width_px, height_px, float(args.fov_degrees))
    rays = pixel_rays_cv(intrinsics, width_px, height_px)
    # Instance indices are deliberately discarded: per-instance visibility is
    # oracle information, and this stage must emit pixels and geometry only.
    bvh, _objects, _tris, _tri_instances, _obj_instances = build_world_bvh(scene)
    max_distance = float(camera.data.clip_end)

    output = Path(args.output)
    (output / "rgb").mkdir(parents=True, exist_ok=True)
    (output / "depth").mkdir(parents=True, exist_ok=True)

    frames: list[dict[str, Any]] = []
    stations = _stations(room_w, room_d, room_h, int(args.ring_count))
    frame_index = 0
    for station in stations:
        # Face the room centre from the ring, and sweep all round from inside.
        if station == (0.0, 0.0):
            base_yaws = [i * 360.0 / (args.yaws_per_station * 2) for i in range(args.yaws_per_station * 2)]
        else:
            inward = math.degrees(math.atan2(-station[0], -station[1]))
            spread = 42.0
            base_yaws = [
                inward + spread * (i - (args.yaws_per_station - 1) / 2.0)
                for i in range(args.yaws_per_station)
            ]
        for yaw in base_yaws:
            position = (station[0], station[1], float(args.camera_height))
            set_camera_yaw_pitch(camera, position, yaw, float(args.pitch_degrees))
            camera_to_world = camera_to_world_cv(camera)
            frame_id = f"o_{frame_index + 1:06d}"
            rgb_relative = f"rgb/{frame_id}.png"
            depth_relative = f"depth/{frame_id}.npy"
            render_png(scene, output / rgb_relative)
            depth = axial_depth_from_bvh(
                bvh,
                camera_to_world,
                rays,
                width=width_px,
                height=height_px,
                max_distance_m=max_distance,
            )
            if not np.isfinite(depth).all() or np.any(depth < 0):
                raise RuntimeError(f"Invalid depth in {frame_id}")
            np.save(output / depth_relative, depth.astype(np.float32, copy=False))
            frames.append(
                {
                    "frame_id": frame_id,
                    "rgb_path": rgb_relative,
                    "depth_path": depth_relative,
                    "width": width_px,
                    "height": height_px,
                    "intrinsics": intrinsics.tolist(),
                    "camera_to_world": camera_to_world.tolist(),
                    "camera_position_m": list(position),
                    "camera_yaw_degrees": float(yaw),
                    "camera_pitch_degrees": float(args.pitch_degrees),
                }
            )
            frame_index += 1

    atomic_json(
        output / "manifest.json",
        {
            "schema": "semantic_3d_chat.spatial_lens.scan.v1",
            "frame_count": len(frames),
            "room_size_m": [room_w, room_d, room_h],
            "resolution": [width_px, height_px],
            "horizontal_fov_degrees": float(args.fov_degrees),
            "camera_convention": "opencv_camera_to_world",
            "contains_instance_labels": False,
            "frames": frames,
        },
    )
    print(json.dumps({"frames": len(frames), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
