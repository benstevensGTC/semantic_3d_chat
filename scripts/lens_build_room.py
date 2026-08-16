#!/usr/bin/env python3
"""Compile a hand-authored room spec into a Blender scene.

Writes three artifacts and keeps them deliberately separate:

    data/spatial_lens/<room>/build.json              geometry only, no words
    data/spatial_lens/<room>/scene.blend             the scene Blender renders
    data/spatial_lens/<room>/measured_geometry.json  boxes Blender actually made
    reports/gemma4/scorer_only/spatial_lens/<room>/key.json   author intent

Only the first three are ever visible to perception.  The key lives under a
``scorer_only`` path that the runtime file audit blocks.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.spatial_lens.room_spec import load_room_spec


def _blender() -> str:
    candidate = os.environ.get("BLENDER", "blender")
    resolved = shutil.which(candidate)
    if resolved:
        return resolved
    mac = Path("/Applications/Blender.app/Contents/MacOS/Blender")
    if mac.is_file():
        return str(mac)
    raise SystemExit(
        "Blender not found. Install it, or set BLENDER=/path/to/blender "
        "(on macOS usually /Applications/Blender.app/Contents/MacOS/Blender)."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, help="hand-authored room JSON")
    parser.add_argument("--force", action="store_true", help="rebuild if present")
    args = parser.parse_args()

    spec = load_room_spec(args.spec)
    room_root = PROJECT_ROOT / "data" / "spatial_lens" / spec.name
    key_root = (
        PROJECT_ROOT / "reports" / "gemma4" / "scorer_only" / "spatial_lens" / spec.name
    )
    blend = room_root / "scene.blend"
    if blend.is_file() and not args.force:
        raise SystemExit(f"{blend} already exists; pass --force to rebuild")
    room_root.mkdir(parents=True, exist_ok=True)
    key_root.mkdir(parents=True, exist_ok=True)

    build_path = room_root / "build.json"
    build_path.write_text(
        json.dumps(spec.build_payload(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (key_root / "key.json").write_text(
        json.dumps(spec.key_payload(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    measured = room_root / "measured_geometry.json"
    command = [
        _blender(),
        "--background",
        "--python",
        str(PROJECT_ROOT / "blender" / "build_authored_room.py"),
        "--",
        "--build",
        str(build_path),
        "--output",
        str(blend),
        "--measured",
        str(measured),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0 or not blend.is_file():
        sys.stderr.write(completed.stdout[-4000:] + "\n" + completed.stderr[-4000:])
        raise SystemExit("Blender failed to build the authored room")

    print(
        json.dumps(
            {
                "room": spec.name,
                "blend": str(blend.relative_to(PROJECT_ROOT)),
                "objects": len(spec.objects),
                "room_size_m": list(spec.size_m),
                "author_key_is_scorer_only": True,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
