"""Choose viewpoints by how much of the room they actually see.

The old scan was thirty views on one ring at one height with one pitch, and
thirty was a number somebody picked. It is the wrong kind of number: whether a
room is covered depends on how cluttered it is, how tall the furniture is and
what stands in front of what, none of which a constant knows about.

This measures instead. Surface points are sampled over every object, a large
pool of candidate viewpoints is generated across positions, heights and pitches,
visibility is resolved by ray-cast against the scene's own BVH, and views are
then chosen greedily -- each time the one that adds the most unseen surface --
until the curve flattens. The result is a per-room view list and the coverage
curve that justifies its length.

Occlusion is handled honestly: a surface point counts as seen only if the ray
from the camera reaches it without hitting something else first, so a chair
behind a table is not credited to a view that cannot see it.

Run through Blender:

    blender --background scene.blend --python blender/plan_scan_coverage.py -- \
        --room-size W D H --output plan.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import bpy
import mathutils

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scene_utils import (  # type: ignore[import-not-found]
    atomic_json,
    blender_cli_args,
)

SHELL_INDEX_FLOOR = 901


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--room-size", nargs=3, type=float, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--target-coverage", type=float, default=0.99)
    parser.add_argument("--max-views", type=int, default=64)
    parser.add_argument("--surface-samples", type=int, default=4000)
    parser.add_argument("--fov-degrees", type=float, default=72.0)
    parser.add_argument("--seed", type=int, default=17)
    return parser


def _object_meshes() -> list[bpy.types.Object]:
    """Everything that is furniture rather than floor, wall or ceiling."""

    out = []
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        index = obj.get("instance_index")
        if index is None or int(index) >= SHELL_INDEX_FLOOR:
            continue
        out.append(obj)
    return out


def _sample_surface(objects, count: int, rng: random.Random):
    """Points spread over the furniture, in proportion to area."""

    depsgraph = bpy.context.evaluated_depsgraph_get()
    triangles = []
    for obj in objects:
        evaluated = obj.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh()
        mesh.calc_loop_triangles()
        matrix = evaluated.matrix_world
        for tri in mesh.loop_triangles:
            a, b, c = (matrix @ mesh.vertices[i].co for i in tri.vertices)
            area = (b - a).cross(c - a).length / 2.0
            if area > 1e-9:
                triangles.append((area, a, b, c))
        evaluated.to_mesh_clear()
    if not triangles:
        return []
    total = sum(t[0] for t in triangles)
    points = []
    for _ in range(count):
        pick = rng.random() * total
        running = 0.0
        for area, a, b, c in triangles:
            running += area
            if running >= pick:
                u, v = rng.random(), rng.random()
                if u + v > 1.0:
                    u, v = 1.0 - u, 1.0 - v
                points.append(a + (b - a) * u + (c - a) * v)
                break
    return points


def _object_boxes(objects) -> list[tuple[list[float], list[float]]]:
    """World-space bounds of every piece of furniture."""

    bpy.context.view_layer.update()
    boxes = []
    for obj in objects:
        corners = [obj.matrix_world @ mathutils.Vector(c) for c in obj.bound_box]
        boxes.append(
            (
                [min(c[a] for c in corners) for a in range(3)],
                [max(c[a] for c in corners) for a in range(3)],
            )
        )
    return boxes


def _in_free_space(point, boxes, clearance: float) -> bool:
    """A camera inside a sofa sees the inside of a sofa.

    The greedy selector has no way to know this on its own -- an occluded view
    simply covers nothing, but a view *embedded* in geometry covers whatever
    happens to be within a few centimetres and scores well for it.
    """

    for low, high in boxes:
        if all(low[a] - clearance <= point[a] <= high[a] + clearance for a in range(3)):
            return False
    return True


def _candidates(width: float, depth: float, height: float, rng: random.Random, boxes=None):
    """Where a camera could stand, at several heights and pitches."""

    margin = 0.75
    half_x = max(width / 2.0 - margin, 0.25)
    half_y = max(depth / 2.0 - margin, 0.25)
    stations = [(0.0, 0.0)]
    for ring, count in ((1.0, 10), (0.55, 6)):
        for index in range(count):
            angle = 2.0 * math.pi * index / count + ring
            stations.append((half_x * ring * math.cos(angle), half_y * ring * math.sin(angle)))
    # Eye level sees the room; low sees under tables; high sees the tops of
    # things, which a single standing height never does.
    heights = [0.75, 1.25, 1.85]
    pitches = [-30.0, -10.0, 10.0]
    views = []
    for station in stations:
        for camera_height in heights:
            if boxes is not None and not _in_free_space(
                (station[0], station[1], camera_height), boxes, 0.35
            ):
                continue
            for yaw_index in range(6):
                yaw = yaw_index * 60.0 + rng.uniform(-8.0, 8.0)
                for pitch in pitches:
                    views.append(
                        {
                            "position_m": [station[0], station[1], camera_height],
                            "yaw_degrees": yaw,
                            "pitch_degrees": pitch,
                        }
                    )
    return views


def _direction(yaw_degrees: float, pitch_degrees: float) -> mathutils.Vector:
    """Must match scene_utils.set_camera_yaw_pitch exactly.

    That helper defines yaw=0 as looking along +Y with positive yaw turning
    right-handed about +Z, which puts a MINUS on the x term. Getting the sign
    wrong mirrors every planned view about the y axis: the planner then scores
    directions the camera never looks in, and reports coverage for a scan that
    was never taken.
    """

    yaw = math.radians(yaw_degrees)
    pitch = math.radians(pitch_degrees)
    return mathutils.Vector(
        (
            -math.sin(yaw) * math.cos(pitch),
            math.cos(yaw) * math.cos(pitch),
            math.sin(pitch),
        )
    ).normalized()


def _visible(view, points, half_fov_cos: float, scene, depsgraph) -> set[int]:
    """Which sampled points this view can actually see, occlusion included."""

    origin = mathutils.Vector(view["position_m"])
    forward = _direction(view["yaw_degrees"], view["pitch_degrees"])
    seen: set[int] = set()
    for index, point in enumerate(points):
        offset = point - origin
        distance = offset.length
        if distance < 1e-4 or distance > 14.0:
            continue
        direction = offset / distance
        if direction.dot(forward) < half_fov_cos:
            continue
        hit, location, _normal, _face, _obj, _matrix = scene.ray_cast(
            depsgraph, origin, direction, distance=distance * 1.02
        )
        # Reaching within a centimetre of the sample means nothing blocked it.
        if hit and (location - point).length < 0.01:
            seen.add(index)
    return seen


def main() -> None:
    args = _parser().parse_args(blender_cli_args())
    rng = random.Random(args.seed)
    width, depth, height = (float(v) for v in args.room_size)

    objects = _object_meshes()
    points = _sample_surface(objects, int(args.surface_samples), rng)
    if not points:
        raise SystemExit("no furniture surface to cover")

    scene = bpy.context.scene
    depsgraph = bpy.context.evaluated_depsgraph_get()
    # A generous cone: the real camera is rectangular, so requiring the point
    # inside the inscribed cone under-counts rather than over-counts.
    half_fov_cos = math.cos(math.radians(float(args.fov_degrees) * 0.5))

    boxes = _object_boxes(objects)
    candidates = _candidates(width, depth, height, rng, boxes)
    if not candidates:
        raise SystemExit("every candidate viewpoint is inside furniture")
    visibility = [_visible(view, points, half_fov_cos, scene, depsgraph) for view in candidates]
    reachable = set().union(*visibility) if visibility else set()
    if not reachable:
        raise SystemExit("no sampled surface is visible from anywhere")

    chosen: list[int] = []
    covered: set[int] = set()
    curve = []
    while len(chosen) < int(args.max_views):
        best, gain = None, 0
        for index, seen in enumerate(visibility):
            if index in chosen:
                continue
            fresh = len(seen - covered)
            if fresh > gain:
                best, gain = index, fresh
        if best is None or gain == 0:
            break
        chosen.append(best)
        covered |= visibility[best]
        curve.append(round(len(covered) / len(reachable), 5))
        if len(covered) / len(reachable) >= float(args.target_coverage):
            break

    atomic_json(
        Path(args.output),
        {
            "schema": "semantic_3d_chat.spatial_lens.scan_plan.v1",
            "room_size_m": [width, depth, height],
            "surface_samples": len(points),
            # Coverage is quoted against what any camera in the room can reach:
            # the inside of a drawer is not a failure of the view plan.
            "reachable_samples": len(reachable),
            "reachable_fraction_of_sampled": round(len(reachable) / len(points), 4),
            "views": [candidates[index] for index in chosen],
            "coverage_curve": curve,
            "final_coverage": curve[-1] if curve else 0.0,
            "candidates_considered": len(candidates),
            "free_space_clearance_m": 0.35,
        },
    )
    print(
        json.dumps(
            {
                "views": len(chosen),
                "coverage": curve[-1] if curve else 0.0,
                "reachable_fraction": round(len(reachable) / len(points), 4),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
