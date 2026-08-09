from __future__ import annotations

import argparse
import json
import time

from semantic_3d_chat.config import PROJECT_ROOT, load_config, project_path
from semantic_3d_chat.mapping.fusion import fuse_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--scene", default="scene_000001")
    args = parser.parse_args()
    config = load_config(args.config)
    started = time.perf_counter()
    voxel_map, frame_stats = fuse_manifest(
        project_path(config, "rendered", args.scene, "manifest.json"),
        project_path(config, "features", args.scene),
        voxel_size_m=float(config["mapping"]["voxel_size_m"]),
        depth_min_m=float(config["mapping"]["depth_min_m"]),
        depth_max_m=float(config["mapping"]["depth_max_m"]),
        pixel_stride=int(config["mapping"]["pixel_stride"]),
        max_voxels=int(config["mapping"]["max_voxels"]),
        confidence_distance_scale_m=float(config["mapping"]["confidence_distance_scale_m"]),
    )
    map_directory = project_path(config, "maps", args.scene)
    output = voxel_map.save(map_directory / "voxel_map.npz", metadata={"scene_id": args.scene})
    preview_directory = PROJECT_ROOT / config["paths"]["reports_root"] / "figures" / args.scene
    previews = voxel_map.export_previews(preview_directory)
    summary = {
        "phase": "mapping_complete",
        "scene_id": args.scene,
        **voxel_map.summary(),
        "content_hash": voxel_map.content_hash(),
        "frame_count": len(frame_stats),
        "elapsed_seconds": time.perf_counter() - started,
        "map_path": str(output.relative_to(PROJECT_ROOT)),
        "preview_paths": [str(path.relative_to(PROJECT_ROOT)) for path in previews],
    }
    metrics_path = PROJECT_ROOT / config["paths"]["reports_root"] / "metrics" / f"map_{args.scene}.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
