from __future__ import annotations

import argparse
import json

import numpy as np

from semantic_3d_chat.config import PROJECT_ROOT, load_config
from semantic_3d_chat.mapping.depth_projection import (
    project_depth_to_world,
    project_world_points_to_pixels,
)
from semantic_3d_chat.rendering_io import iter_frames, load_rgb_depth


def distance_to_box_surface(points: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    distances = np.stack(
        [np.abs(points[:, axis] - bound) for axis in range(3) for bound in (lower[axis], upper[axis])],
        axis=1,
    )
    return distances.min(axis=1)


def validate_scene(scene_id: str, config: dict) -> dict:
    data_root = PROJECT_ROOT / config["paths"]["data_root"]
    manifest_path = data_root / "rendered" / scene_id / "manifest.json"
    oracle_path = data_root / "oracle" / scene_id / "oracle.json"
    oracle = json.loads(oracle_path.read_text(encoding="utf-8"))
    all_points = []
    reprojection_errors = []
    depth_errors = []
    for frame in iter_frames(manifest_path):
        _, depth = load_rgb_depth(frame)
        projection = project_depth_to_world(
            depth,
            frame.intrinsics,
            frame.camera_to_world,
            min_depth_m=float(config["mapping"]["depth_min_m"]),
            max_depth_m=float(config["mapping"]["depth_max_m"]),
            pixel_stride=4,
        )
        projected_uv, projected_depth = project_world_points_to_pixels(
            projection.points_world, frame.intrinsics, frame.camera_to_world
        )
        reprojection_errors.append(
            np.linalg.norm(projected_uv - projection.pixels_uv.astype(np.float64), axis=1)
        )
        depth_errors.append(np.abs(projected_depth - projection.depth_m))
        all_points.append(projection.points_world)
    points = np.concatenate(all_points, axis=0)
    room_min = np.asarray(oracle["room"]["bounds_min_m"], dtype=np.float32)
    room_max = np.asarray(oracle["room"]["bounds_max_m"], dtype=np.float32)
    inside = np.all(points >= room_min - 0.03, axis=1) & np.all(points <= room_max + 0.03, axis=1)
    floor_candidates = np.abs(points[:, 2] - room_min[2]) < 0.08

    cube = next(item for item in oracle["instances"] if item["category"] == "cube")
    cube_min = np.asarray(cube["bbox"]["min_xyz_m"], dtype=np.float32)
    cube_max = np.asarray(cube["bbox"]["max_xyz_m"], dtype=np.float32)
    tolerance = 0.04
    cube_mask = np.all(points >= cube_min - tolerance, axis=1) & np.all(
        points <= cube_max + tolerance, axis=1
    )
    cube_points = points[cube_mask]
    cube_surface_error = (
        distance_to_box_surface(cube_points, cube_min, cube_max)
        if len(cube_points)
        else np.array([np.inf])
    )
    reprojection = np.concatenate(reprojection_errors)
    depth_roundtrip = np.concatenate(depth_errors)
    metrics = {
        "scene_id": scene_id,
        "sampled_points": len(points),
        "inside_room_fraction": float(inside.mean()),
        "floor_point_count": int(floor_candidates.sum()),
        "floor_mean_abs_error_m": float(np.abs(points[floor_candidates, 2]).mean())
        if floor_candidates.any()
        else None,
        "reprojection_rmse_pixels": float(np.sqrt(np.mean(reprojection**2))),
        "depth_roundtrip_rmse_m": float(np.sqrt(np.mean(depth_roundtrip**2))),
        "cube_surface_point_count": len(cube_points),
        "cube_surface_median_error_m": float(np.median(cube_surface_error)),
    }
    metrics["passed"] = bool(
        metrics["inside_room_fraction"] > 0.995
        and metrics["reprojection_rmse_pixels"] < 1e-3
        and metrics["depth_roundtrip_rmse_m"] < 1e-5
        and metrics["cube_surface_point_count"] >= 10
        and metrics["cube_surface_median_error_m"] < 0.03
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--scene", default="scene_000001")
    args = parser.parse_args()
    config = load_config(args.config)
    metrics = validate_scene(args.scene, config)
    output = PROJECT_ROOT / config["paths"]["reports_root"] / "metrics" / "geometry.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    if not metrics["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
