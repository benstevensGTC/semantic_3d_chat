#!/usr/bin/env python3
"""Build, scan and perceive many authored rooms in one pass.

Each stage is skipped when its artifact already exists, so the pipeline can be
resumed after an interruption without redoing hours of scanning.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from semantic_3d_chat.config import PROJECT_ROOT

GEMMA_PY = PROJECT_ROOT / ".venv-gemma4" / "bin" / "python"
MAIN_PY = PROJECT_ROOT / ".venv" / "bin" / "python"


def run(python: Path, script: str, *args: str) -> bool:
    completed = subprocess.run(
        [str(python), str(PROJECT_ROOT / "scripts" / script), *args],
        capture_output=True, text=True, check=False, cwd=PROJECT_ROOT,
        env={"PYTHONPATH": "src", "PATH": "/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin"},
    )
    if completed.returncode != 0:
        sys.stderr.write(f"\n[{script} {' '.join(args)}] failed:\n")
        sys.stderr.write(completed.stdout[-1500:] + completed.stderr[-1500:] + "\n")
    return completed.returncode == 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rooms", nargs="+", required=True)
    parser.add_argument("--resolution", type=int, default=384)
    parser.add_argument("--ring-count", type=int, default=6)
    parser.add_argument("--yaws", type=int, default=3)
    parser.add_argument("--voxel-size-m", type=float, default=0.06)
    parser.add_argument("--pixel-stride", type=int, default=6)
    parser.add_argument("--views-per-object", type=int, default=2)
    args = parser.parse_args()

    done, failed = [], []
    for name in args.rooms:
        started = time.perf_counter()
        root = PROJECT_ROOT / "data" / "spatial_lens" / name
        spec = PROJECT_ROOT / "rooms" / f"{name}.json"
        if not spec.is_file():
            print(f"{name}: no spec"); failed.append(name); continue

        stages = [
            (root / "scene.blend", MAIN_PY, ("lens_build_room.py", "--spec", str(spec))),
            (root / "scans" / "manifest.json", MAIN_PY,
             ("lens_scan_room.py", "--room", name,
              "--resolution", str(args.resolution),
              "--ring-count", str(args.ring_count),
              "--yaws-per-station", str(args.yaws))),
            (root / "point_cloud.npz", GEMMA_PY,
             ("lens_perceive.py", "--room", name,
              "--voxel-size-m", str(args.voxel_size_m),
              "--pixel-stride", str(args.pixel_stride))),
            # Naming closes the loop: it is what turns anonymous footprints
            # into the (phrase, cells) pairs the grounding head trains on.
            (root / "scene_graph.json", GEMMA_PY,
             ("lens_understand.py", "--room", name,
              "--views-per-object", str(args.views_per_object))),
        ]
        broke = False
        for artifact, python, command in stages:
            if artifact.is_file():
                continue
            if not run(python, *command):
                failed.append(name); broke = True; break
        if broke:
            continue

        cloud_mb = (root / "point_cloud.npz").stat().st_size / 1e6
        frames = json.loads((root / "scans" / "manifest.json").read_text())["frame_count"]
        objects = len(json.loads((root / "scene_graph.json").read_text())["objects"])
        print(f"{name}: {frames} frames, cloud {cloud_mb:.0f} MB, {objects} objects, "
              f"{time.perf_counter() - started:.0f}s", flush=True)
        done.append(name)

    print(f"\nperceived {len(done)}/{len(args.rooms)}")
    if failed:
        print("failed:", ", ".join(failed))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
