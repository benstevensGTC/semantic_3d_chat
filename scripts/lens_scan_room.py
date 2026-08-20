#!/usr/bin/env python3
"""Scan an authored room: RGB, metric depth, and exact camera poses only."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from semantic_3d_chat.config import PROJECT_ROOT


def _blender() -> str:
    candidate = os.environ.get("BLENDER", "blender")
    resolved = shutil.which(candidate)
    if resolved:
        return resolved
    mac = Path("/Applications/Blender.app/Contents/MacOS/Blender")
    if mac.is_file():
        return str(mac)
    raise SystemExit("Blender not found; set BLENDER=/path/to/blender")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--room", required=True)
    parser.add_argument("--resolution", type=int, default=448)
    parser.add_argument("--ring-count", type=int, default=8)
    parser.add_argument("--yaws-per-station", type=int, default=3)
    # With a coverage plan the ring arguments are ignored: the views were
    # chosen for this room's geometry rather than drawn on a circle.
    parser.add_argument("--plan", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    room_root = PROJECT_ROOT / "data" / "spatial_lens" / args.room
    blend = room_root / "scene.blend"
    if not blend.is_file():
        raise SystemExit(f"No built room at {blend}; run lens_build_room.py first")
    build = json.loads((room_root / "build.json").read_text(encoding="utf-8"))
    scans = room_root / "scans"
    if (scans / "manifest.json").is_file() and not args.force:
        raise SystemExit(f"{scans} already scanned; pass --force to rescan")

    command = [
        _blender(),
        "--background",
        str(blend),
        "--python",
        str(PROJECT_ROOT / "blender" / "scan_authored_room.py"),
        "--",
        "--output",
        str(scans),
        "--room-size",
        *[str(value) for value in build["room_size_m"]],
        "--resolution",
        str(args.resolution),
        str(args.resolution),
        "--ring-count",
        str(args.ring_count),
        "--yaws-per-station",
        str(args.yaws_per_station),
    ]
    if args.plan:
        command += ["--plan", str(args.plan)]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0 or not (scans / "manifest.json").is_file():
        sys.stderr.write(completed.stdout[-4000:] + "\n" + completed.stderr[-4000:])
        raise SystemExit("Blender failed to scan the authored room")

    manifest = json.loads((scans / "manifest.json").read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "room": args.room,
                "frames": manifest["frame_count"],
                "resolution": manifest["resolution"],
                "contains_instance_labels": manifest["contains_instance_labels"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
