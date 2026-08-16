"""Render one exact RGB-D observation from a sanitized runtime Blender asset."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

sys.path.insert(0, str(Path(__file__).resolve().parent))

from runtime_scene_contract import SHA256, audit_runtime_scene, file_sha256
from scene_utils import (
    blender_cli_args,
    camera_intrinsics,
    camera_to_world_cv,
    configure_render,
    create_camera,
    render_png,
    set_camera_yaw_pitch,
    validate_scene_id,
)

_OBSERVATION_ID = re.compile(r"o_[0-9]{6}")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--scene", required=True)
    result.add_argument("--observation", required=True)
    result.add_argument("--asset-sha256", required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--x", type=float, required=True)
    result.add_argument("--y", type=float, required=True)
    result.add_argument("--z", type=float, required=True)
    result.add_argument("--yaw", type=float, required=True)
    result.add_argument("--pitch", type=float, required=True)
    result.add_argument("--width", type=int, required=True)
    result.add_argument("--height", type=int, required=True)
    result.add_argument("--horizontal-fov", type=float, required=True)
    result.add_argument("--engine", required=True)
    result.add_argument("--samples", type=int, required=True)
    result.add_argument("--max-depth", type=float, default=30.0)
    return result


def _safe_runtime_input(expected_sha256: str) -> None:
    source = Path(os.path.abspath(bpy.data.filepath))
    if {"oracle", "qa"} & {part.casefold() for part in source.parts}:
        raise ValueError("Runtime renderer cannot open an oracle or QA scene")
    if SHA256.fullmatch(expected_sha256) is None or file_sha256(source) != expected_sha256:
        raise ValueError("Opened runtime scene differs from its authenticated asset hash")
    audit_runtime_scene(audit_names=True)


def _world_bvh() -> BVHTree:
    graph = bpy.context.evaluated_depsgraph_get()
    vertices: list[tuple[float, float, float]] = []
    triangles: list[tuple[int, int, int]] = []
    for obj in sorted(bpy.context.scene.objects, key=lambda item: item.name):
        if obj.type != "MESH" or obj.hide_render:
            continue
        evaluated = obj.evaluated_get(graph)
        mesh = evaluated.to_mesh()
        try:
            mesh.calc_loop_triangles()
            offset = len(vertices)
            vertices.extend(tuple(evaluated.matrix_world @ vertex.co) for vertex in mesh.vertices)
            triangles.extend(
                tuple(offset + int(index) for index in triangle.vertices)
                for triangle in mesh.loop_triangles
            )
        finally:
            evaluated.to_mesh_clear()
    if not vertices or not triangles:
        raise RuntimeError("Sanitized runtime asset contains no renderable geometry")
    result = BVHTree.FromPolygons(vertices, triangles, all_triangles=True, epsilon=0.0)
    if result is None:
        raise RuntimeError("Blender failed to build the runtime depth BVH")
    return result


def _pixel_rays(intrinsics: np.ndarray, width: int, height: int) -> np.ndarray:
    yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    rays = np.stack(
        (
            (xx - intrinsics[0, 2]) / intrinsics[0, 0],
            (yy - intrinsics[1, 2]) / intrinsics[1, 1],
            np.ones_like(xx),
        ),
        axis=-1,
    ).astype(np.float64)
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    return rays.reshape(-1, 3)


def _depth(
    bvh: BVHTree,
    camera_to_world: np.ndarray,
    rays: np.ndarray,
    width: int,
    height: int,
    max_depth: float,
) -> np.ndarray:
    origin = camera_to_world[:3, 3]
    world_rays = rays @ camera_to_world[:3, :3].T
    values = np.zeros(len(rays), dtype=np.float32)
    origin_vector = Vector(tuple(float(value) for value in origin))
    for index, direction in enumerate(world_rays):
        _point, _normal, _face, distance = bvh.ray_cast(
            origin_vector,
            Vector(tuple(float(value) for value in direction)),
            max_depth,
        )
        if distance is not None:
            axial = float(distance) * float(rays[index, 2])
            if axial > 0.0 and math.isfinite(axial):
                values[index] = axial
    return values.reshape(height, width)


def _atomic_npy(path: Path, value: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npy")
    try:
        np.save(temporary, value, allow_pickle=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, value: dict[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    args = parser().parse_args(blender_cli_args())
    if bpy.context.preferences.filepaths.use_scripts_auto_execute:
        raise RuntimeError("Runtime rendering requires Blender --disable-autoexec")
    scene_id = validate_scene_id(args.scene)
    if _OBSERVATION_ID.fullmatch(args.observation) is None:
        raise ValueError("Observation ID must be opaque")
    values = (args.x, args.y, args.z, args.yaw, args.pitch, args.horizontal_fov, args.max_depth)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Runtime render pose and camera values must be finite")
    if args.width < 2 or args.height < 2 or args.samples < 1:
        raise ValueError("Runtime render dimensions and samples must be positive")
    if not 1.0 <= args.horizontal_fov < 179.0 or args.max_depth <= 0:
        raise ValueError("Runtime camera FOV or max depth is invalid")
    output = Path(os.path.abspath(args.output.expanduser()))
    if {"oracle", "qa"} & {part.casefold() for part in output.parts}:
        raise ValueError("Runtime observation cannot be written under oracle or QA")
    output.mkdir(parents=True, exist_ok=True)
    _safe_runtime_input(args.asset_sha256)

    scene = bpy.context.scene
    configure_render(
        scene,
        width=args.width,
        height=args.height,
        engine=args.engine,
        samples=args.samples,
    )
    camera = create_camera("c_000000", args.horizontal_fov)
    scene.camera = camera
    set_camera_yaw_pitch(
        camera,
        (args.x, args.y, args.z),
        args.yaw,
        args.pitch,
    )
    intrinsics = camera_intrinsics(args.width, args.height, args.horizontal_fov)
    camera_to_world = camera_to_world_cv(camera)
    rays = _pixel_rays(intrinsics, args.width, args.height)
    bvh = _world_bvh()
    rgb_name = f"{args.observation}.png"
    depth_name = f"{args.observation}.npy"
    render_png(scene, output / rgb_name)
    depth = _depth(
        bvh,
        camera_to_world,
        rays,
        args.width,
        args.height,
        args.max_depth,
    )
    if not np.isfinite(depth).all() or np.any(depth < 0) or not np.any(depth > 0):
        raise RuntimeError("Runtime render produced invalid or empty metric depth")
    _atomic_npy(output / depth_name, depth)
    receipt = {
        "schema": "semantic_3d_chat.runtime_observation.v1",
        "scene_id": scene_id,
        "observation_id": args.observation,
        "rgb_path": rgb_name,
        "depth_path": depth_name,
        "intrinsics": intrinsics.tolist(),
        "camera_to_world": camera_to_world.tolist(),
        "width": args.width,
        "height": args.height,
        "valid_depth_pixels": int(np.count_nonzero(depth > 0)),
    }
    _atomic_json(output / f"{args.observation}.json", receipt)
    print(json.dumps(receipt, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
