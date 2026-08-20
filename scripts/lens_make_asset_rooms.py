#!/usr/bin/env python3
"""Compose asset rooms and build them in Blender.

Writes the geometry-only build payload the builder consumes and the
category key the scorer keeps, then invokes Blender per room.
"""

from __future__ import annotations

import argparse
import json
import subprocess

from semantic_3d_chat.assets.compose import compose_room, load_manifest
from semantic_3d_chat.config import PROJECT_ROOT

BLENDER = "/Applications/Blender.app/Contents/MacOS/Blender"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=40)
    parser.add_argument("--prefix", default="asset")
    parser.add_argument("--seed", type=int, default=90210)
    parser.add_argument("--blender", default=BLENDER)
    parser.add_argument("--compose-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    assets_root = PROJECT_ROOT / "data" / "assets"
    manifest = load_manifest(assets_root)
    built = 0
    for index in range(args.count):
        name = f"{args.prefix}{index:03d}"
        room_dir = PROJECT_ROOT / "data" / "spatial_lens" / name
        blend = room_dir / "scene.blend"
        if blend.is_file() and not args.force:
            print(f"skip {name}")
            continue
        room = compose_room(name, manifest, seed=args.seed + index)
        room_dir.mkdir(parents=True, exist_ok=True)

        # Geometry for Blender: instance ids, transforms and mesh paths only.
        build = room.build_payload()
        for entry, placement in zip(build["objects"], room.placements, strict=True):
            entry["size_m"] = [round(v, 5) for v in placement.size_m]
        (room_dir / "build.json").write_text(
            json.dumps(build, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        # Category names live under scorer_only, which the runtime audit blocks.
        key_dir = room_dir / "scorer_only"
        key_dir.mkdir(exist_ok=True)
        (key_dir / "room_key.json").write_text(
            json.dumps(room.key_payload(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(
            f"{name}: {room.recipe['objects']} objects, "
            f"{room.size_m[0]:.1f}x{room.size_m[1]:.1f}m, "
            f"dups={room.recipe['duplicated_categories']}"
        )
        if args.compose_only:
            continue

        result = subprocess.run(
            [
                args.blender, "--background", "--python",
                str(PROJECT_ROOT / "blender" / "build_asset_room.py"), "--",
                "--build", str(room_dir / "build.json"),
                "--assets", str(assets_root),
                "--output", str(blend),
                "--measured", str(room_dir / "measured_geometry.json"),
            ],
            capture_output=True, text=True, check=False,
        )
        if result.returncode != 0 or not blend.is_file():
            print(f"  FAILED: {result.stdout[-1500:]}\n{result.stderr[-1500:]}")
            continue
        built += 1
    print(f"\nbuilt {built} rooms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
