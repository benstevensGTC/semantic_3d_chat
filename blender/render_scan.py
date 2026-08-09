"""Render an opaque RGB scan and exact axial ray-cast depth from a room .blend."""

# Blender does not add a ``--python`` script's directory to sys.path.  The
# path insertion below must therefore precede the local helper import.

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scene_utils import (
    atomic_json,
    blender_cli_args,
    camera_intrinsics,
    camera_to_world_cv,
    config_hash,
    configure_render,
    create_camera,
    load_config,
    render_png,
    scene_paths,
    set_camera_yaw_pitch,
    validate_scene_id,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", default="configs/default.yaml")
    result.add_argument("--scene", default="scene_000001")
    return result


def build_world_bvh(scene: bpy.types.Scene) -> tuple[BVHTree, int, int]:
    """Build one evaluated world-space triangle BVH for deterministic depth."""

    depsgraph = bpy.context.evaluated_depsgraph_get()
    vertices: list[tuple[float, float, float]] = []
    triangles: list[tuple[int, int, int]] = []
    mesh_objects = 0
    for obj in sorted(scene.objects, key=lambda candidate: candidate.name):
        if obj.type != "MESH" or obj.hide_render:
            continue
        evaluated = obj.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh()
        try:
            mesh.calc_loop_triangles()
            offset = len(vertices)
            vertices.extend(tuple(evaluated.matrix_world @ vertex.co) for vertex in mesh.vertices)
            triangles.extend(
                tuple(offset + int(index) for index in triangle.vertices)
                for triangle in mesh.loop_triangles
            )
            mesh_objects += 1
        finally:
            evaluated.to_mesh_clear()
    if not vertices or not triangles:
        raise RuntimeError("No evaluated mesh geometry was available for depth ray casting")
    bvh = BVHTree.FromPolygons(vertices, triangles, all_triangles=True, epsilon=0.0)
    if bvh is None:
        raise RuntimeError("Blender failed to construct the depth ray-casting BVH")
    return bvh, mesh_objects, len(triangles)


def pixel_rays_cv(intrinsics: np.ndarray, width: int, height: int) -> np.ndarray:
    """Return one normalized CV ray per complete-image pixel in row-major order."""

    yy, xx = np.meshgrid(
        np.arange(height, dtype=np.float64),
        np.arange(width, dtype=np.float64),
        indexing="ij",
    )
    rays = np.stack(
        (
            (xx - intrinsics[0, 2]) / intrinsics[0, 0],
            (yy - intrinsics[1, 2]) / intrinsics[1, 1],
            np.ones_like(xx),
        ),
        axis=-1,
    )
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    return rays.reshape(-1, 3)


def axial_depth_from_bvh(
    bvh: BVHTree,
    camera_to_world: np.ndarray,
    rays_cv: np.ndarray,
    *,
    width: int,
    height: int,
    max_distance_m: float,
) -> np.ndarray:
    """Cast every pixel ray once and store z-forward, not Euclidean range."""

    rotation = camera_to_world[:3, :3]
    origin = camera_to_world[:3, 3]
    world_rays = rays_cv @ rotation.T
    depth = np.zeros(len(rays_cv), dtype=np.float32)
    origin_vector = Vector(tuple(float(value) for value in origin))
    for index, world_direction in enumerate(world_rays):
        _, _, _, distance = bvh.ray_cast(
            origin_vector,
            Vector(tuple(float(value) for value in world_direction)),
            max_distance_m,
        )
        if distance is None:
            continue
        # The normalized camera ray's z component converts slant range to the
        # exact optical-axis depth consumed by pinhole back-projection.
        axial_depth = float(distance) * float(rays_cv[index, 2])
        if axial_depth > 0.0 and np.isfinite(axial_depth):
            depth[index] = axial_depth
    return depth.reshape(height, width)


def atomic_numpy(path: Path, array: np.ndarray) -> None:
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npy")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        np.save(temporary, array, allow_pickle=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def sanitize_scene_for_scan(scene: bpy.types.Scene) -> None:
    # Remove the preview camera.  Geometry and lights retain opaque names.
    for obj in list(scene.objects):
        if obj.type == "CAMERA":
            bpy.data.objects.remove(obj, do_unlink=True)


def main() -> None:
    started = time.perf_counter()
    args = parser().parse_args(blender_cli_args())
    scene_id = validate_scene_id(args.scene)
    config, config_path = load_config(args.config)
    del config_path
    paths = scene_paths(config, scene_id)
    scene = bpy.context.scene
    sanitize_scene_for_scan(scene)

    render_config = config["render"]
    width, height = (int(value) for value in render_config["resolution"])
    horizontal_fov = float(render_config["horizontal_fov_degrees"])
    resolved_engine = configure_render(
        scene,
        width=width,
        height=height,
        engine=str(render_config["engine"]),
        samples=int(render_config["samples"]),
    )
    camera = create_camera("c_scan", horizontal_fov)
    scene.camera = camera
    intrinsics = camera_intrinsics(width, height, horizontal_fov)
    rays_cv = pixel_rays_cv(intrinsics, width, height)
    bvh, mesh_objects, triangle_count = build_world_bvh(scene)

    position = tuple(float(value) for value in render_config["camera_position_m"])
    yaws = [float(value) for value in render_config["yaw_degrees"]]
    pitches = [float(value) for value in render_config["pitch_degrees"]]
    max_distance = float(camera.data.clip_end)
    frames: list[dict[str, Any]] = []
    depth_min = float("inf")
    depth_max = 0.0
    valid_pixels = 0

    frame_index = 0
    for pitch in pitches:
        for yaw in yaws:
            frame_started = time.perf_counter()
            frame_id = f"f_{frame_index:06d}"
            camera_id = f"c_{frame_index:06d}"
            set_camera_yaw_pitch(camera, position, yaw, pitch)
            camera_to_world = camera_to_world_cv(camera)
            rgb_relative = Path("rgb") / f"{frame_id}.png"
            depth_relative = Path("depth") / f"{frame_id}.npy"
            render_png(scene, paths["rendered"] / rgb_relative)
            depth = axial_depth_from_bvh(
                bvh,
                camera_to_world,
                rays_cv,
                width=width,
                height=height,
                max_distance_m=max_distance,
            )
            if not np.isfinite(depth).all() or np.any(depth < 0):
                raise RuntimeError(f"Invalid depth values in {frame_id}")
            atomic_numpy(paths["rendered"] / depth_relative, depth.astype(np.float32, copy=False))
            valid = depth > 0
            if np.any(valid):
                depth_min = min(depth_min, float(depth[valid].min()))
                depth_max = max(depth_max, float(depth[valid].max()))
                valid_pixels += int(valid.sum())
            frames.append(
                {
                    "frame_id": frame_id,
                    "camera_id": camera_id,
                    "frame_number": frame_index,
                    "rgb_path": rgb_relative.as_posix(),
                    "depth_path": depth_relative.as_posix(),
                    "intrinsics": intrinsics.tolist(),
                    "camera_to_world": camera_to_world.tolist(),
                }
            )
            print(
                "FRAME_RENDERED "
                f"frame={frame_id} valid={int(valid.sum())}/{width * height} "
                f"seconds={time.perf_counter() - frame_started:.3f}"
            )
            frame_index += 1

    manifest = {
        "schema_version": 1,
        "scene_id": scene_id,
        "config_hash": config_hash(config),
        "coordinate_system": {
            "world": "x_right_y_forward_z_up",
            "camera": "x_right_y_down_z_forward",
            "units": "meters",
            "depth": "axial_camera_z",
        },
        "image_size": {"width": width, "height": height},
        "horizontal_fov_degrees": horizontal_fov,
        "frames": frames,
    }
    atomic_json(paths["rendered"] / "manifest.json", manifest)
    elapsed = time.perf_counter() - started
    total_pixels = len(frames) * width * height
    print(
        "SCAN_RENDERED "
        f"scene={scene_id} frames={len(frames)} valid={valid_pixels}/{total_pixels} "
        f"depth_min={depth_min:.6f} depth_max={depth_max:.6f} "
        f"meshes={mesh_objects} triangles={triangle_count} engine={resolved_engine} "
        f"seconds={elapsed:.3f} manifest={paths['rendered'] / 'manifest.json'}"
    )


if __name__ == "__main__":
    main()
