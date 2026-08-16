"""Render an opaque RGB scan and exact axial ray-cast depth from a room .blend."""

# Blender does not add a ``--python`` script's directory to sys.path.  The
# path insertion below must therefore precede the local helper import.

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import Counter
from collections.abc import MutableMapping, Sequence
from pathlib import Path
from typing import Any

import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

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

from semantic_3d_chat.data.scene_variants import validate_visibility_evidence
from semantic_3d_chat.scan_plan import (
    build_runtime_frame,
    build_runtime_manifest,
    expand_scan_poses,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", default="configs/default.yaml")
    result.add_argument("--scene", default="scene_000001")
    return result


def build_world_bvh(
    scene: bpy.types.Scene,
) -> tuple[BVHTree, int, int, tuple[int, ...], tuple[int, ...]]:
    """Build one evaluated world-space triangle BVH for deterministic depth."""

    depsgraph = bpy.context.evaluated_depsgraph_get()
    vertices: list[tuple[float, float, float]] = []
    triangles: list[tuple[int, int, int]] = []
    triangle_instance_indices: list[int] = []
    object_instance_indices: set[int] = set()
    mesh_objects = 0
    for obj in sorted(scene.objects, key=lambda candidate: candidate.name):
        if obj.type != "MESH" or obj.hide_render:
            continue
        evaluated = obj.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh()
        try:
            raw_instance_index = obj.get("instance_index")
            if not isinstance(raw_instance_index, int):
                raise TypeError(f"Mesh {obj.name} lacks an opaque numeric instance index")
            instance_index = int(raw_instance_index)
            mesh.calc_loop_triangles()
            offset = len(vertices)
            vertices.extend(tuple(evaluated.matrix_world @ vertex.co) for vertex in mesh.vertices)
            object_triangles = [
                tuple(offset + int(index) for index in triangle.vertices)
                for triangle in mesh.loop_triangles
            ]
            triangles.extend(object_triangles)
            triangle_instance_indices.extend([instance_index] * len(object_triangles))
            if instance_index >= 100:
                object_instance_indices.add(instance_index)
            mesh_objects += 1
        finally:
            evaluated.to_mesh_clear()
    if not vertices or not triangles:
        raise RuntimeError("No evaluated mesh geometry was available for depth ray casting")
    bvh = BVHTree.FromPolygons(vertices, triangles, all_triangles=True, epsilon=0.0)
    if bvh is None:
        raise RuntimeError("Blender failed to construct the depth ray-casting BVH")
    return (
        bvh,
        mesh_objects,
        len(triangles),
        tuple(triangle_instance_indices),
        tuple(sorted(object_instance_indices)),
    )


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
    triangle_instance_indices: Sequence[int] | None = None,
    visible_pixel_counts: MutableMapping[int, int] | None = None,
) -> np.ndarray:
    """Cast every pixel ray once and store z-forward, not Euclidean range."""

    rotation = camera_to_world[:3, :3]
    origin = camera_to_world[:3, 3]
    world_rays = rays_cv @ rotation.T
    depth = np.zeros(len(rays_cv), dtype=np.float32)
    origin_vector = Vector(tuple(float(value) for value in origin))
    if triangle_instance_indices is not None and visible_pixel_counts is None:
        raise ValueError("visible_pixel_counts is required with triangle instance indices")
    for index, world_direction in enumerate(world_rays):
        _, _, face_index, distance = bvh.ray_cast(
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
            if triangle_instance_indices is not None:
                if face_index is None or not 0 <= int(face_index) < len(triangle_instance_indices):
                    raise RuntimeError("Depth ray returned an invalid BVH face index")
                instance_index = int(triangle_instance_indices[int(face_index)])
                visible_pixel_counts[instance_index] = (
                    int(visible_pixel_counts.get(instance_index, 0)) + 1
                )
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
    (
        bvh,
        mesh_objects,
        triangle_count,
        triangle_instance_indices,
        object_instance_indices,
    ) = build_world_bvh(scene)

    scan_poses = expand_scan_poses(render_config)
    max_distance = float(camera.data.clip_end)
    frames: list[dict[str, Any]] = []
    depth_min = float("inf")
    depth_max = 0.0
    valid_pixels = 0
    visible_pixel_counts: Counter[int] = Counter()

    for frame_index, pose in enumerate(scan_poses):
        frame_started = time.perf_counter()
        set_camera_yaw_pitch(
            camera,
            pose.position_m,
            pose.yaw_degrees,
            pose.pitch_degrees,
        )
        camera_to_world = camera_to_world_cv(camera)
        frame = build_runtime_frame(
            frame_index,
            intrinsics=intrinsics.tolist(),
            camera_to_world=camera_to_world.tolist(),
        )
        frame_id = str(frame["frame_id"])
        rgb_relative = Path(str(frame["rgb_path"]))
        depth_relative = Path(str(frame["depth_path"]))
        render_png(scene, paths["rendered"] / rgb_relative)
        depth = axial_depth_from_bvh(
            bvh,
            camera_to_world,
            rays_cv,
            width=width,
            height=height,
            max_distance_m=max_distance,
            triangle_instance_indices=triangle_instance_indices,
            visible_pixel_counts=visible_pixel_counts,
        )
        if not np.isfinite(depth).all() or np.any(depth < 0):
            raise RuntimeError(f"Invalid depth values in {frame_id}")
        atomic_numpy(
            paths["rendered"] / depth_relative,
            depth.astype(np.float32, copy=False),
        )
        valid = depth > 0
        if np.any(valid):
            depth_min = min(depth_min, float(depth[valid].min()))
            depth_max = max(depth_max, float(depth[valid].max()))
            valid_pixels += int(valid.sum())
        frames.append(frame)
        print(
            "FRAME_RENDERED "
            f"frame={frame_id} valid={int(valid.sum())}/{width * height} "
            f"seconds={time.perf_counter() - frame_started:.3f}"
        )

    minimum_visible_pixels = 1
    expected_instance_ids = [f"i_{value:06d}" for value in object_instance_indices]
    visibility = {
        "schema_version": 1,
        "scene_id": scene_id,
        "method": "exact_depth_raycast",
        "minimum_visible_pixels": minimum_visible_pixels,
        "expected_instance_ids": expected_instance_ids,
        "visible_pixel_counts": {
            instance_id: int(visible_pixel_counts.get(instance_index, 0))
            for instance_id, instance_index in zip(
                expected_instance_ids, object_instance_indices, strict=True
            )
        },
        "all_required_visible": all(
            int(visible_pixel_counts.get(instance_index, 0)) >= minimum_visible_pixels
            for instance_index in object_instance_indices
        ),
    }
    validate_visibility_evidence(visibility)
    atomic_json(paths["oracle"] / "visibility.json", visibility)

    manifest = build_runtime_manifest(
        scene_id=scene_id,
        config_digest=config_hash(config),
        width=width,
        height=height,
        horizontal_fov_degrees=horizontal_fov,
        frames=frames,
    )
    atomic_json(paths["rendered"] / "manifest.json", manifest)
    elapsed = time.perf_counter() - started
    total_pixels = len(frames) * width * height
    print(
        "SCAN_RENDERED "
        f"scene={scene_id} frames={len(frames)} valid={valid_pixels}/{total_pixels} "
        f"depth_min={depth_min:.6f} depth_max={depth_max:.6f} "
        f"meshes={mesh_objects} triangles={triangle_count} engine={resolved_engine} "
        f"visible_objects={len(object_instance_indices)} "
        f"seconds={elapsed:.3f} manifest={paths['rendered'] / 'manifest.json'} "
        f"visibility={paths['oracle'] / 'visibility.json'}"
    )


if __name__ == "__main__":
    main()
