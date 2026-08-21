#!/usr/bin/env python3
"""Fuse a room's scan into a Gemma semantic point cloud."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.language.model_choice import add_model_arguments
from semantic_3d_chat.spatial_lens.perceive import build_semantic_cloud


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--room", required=True)
    parser.add_argument("--voxel-size-m", type=float, default=0.05)
    parser.add_argument("--pixel-stride", type=int, default=5)
    parser.add_argument("--force", action="store_true")
    # A cloud belongs to the model that built it: its feature width is that
    # decoder's hidden size. Writing an E4B map over an E2B one would silently
    # break every cached result, so a different model gets a different name.
    parser.add_argument("--out-name", default="point_cloud.npz")
    add_model_arguments(parser)
    args = parser.parse_args()

    destination = (
        PROJECT_ROOT / "data" / "spatial_lens" / args.room / args.out_name
    )
    if destination.is_file() and not args.force:
        raise SystemExit(f"{destination} exists; pass --force to rebuild")

    def report(done: int, total: int, voxels: int) -> None:
        print(f"frame {done}/{total} voxels={voxels}", flush=True)

    cloud = build_semantic_cloud(
        args.room,
        voxel_size_m=args.voxel_size_m,
        pixel_stride=args.pixel_stride,
        progress=report,
        model_id=args.model,
        revision=args.revision,
    )
    cloud.save(destination)
    print(
        json.dumps(
            {
                "room": args.room,
                "model": args.model,
                "voxels": len(cloud),
                "feature_dim": int(cloud.features.shape[1]),
                "voxel_size_m": cloud.voxel_size_m,
                "megabytes": round(Path(destination).stat().st_size / 1e6, 1),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
