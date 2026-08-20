#!/usr/bin/env python3
"""Compose, build, plan, scan, perceive and name a set of asset rooms.

Each stage writes an artifact and is skipped when that artifact already exists,
so an interrupted run resumes rather than restarting. The stages that need
Gemma run in the transformers venv; everything else runs in the main one.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time

from semantic_3d_chat.assets.compose import compose_room, load_manifest
from semantic_3d_chat.config import PROJECT_ROOT

BLENDER = "/Applications/Blender.app/Contents/MacOS/Blender"
MAIN_PY = PROJECT_ROOT / ".venv" / "bin" / "python"
GEMMA_PY = PROJECT_ROOT / ".venv-gemma4" / "bin" / "python"
ENV = {"PYTHONPATH": "src", "PATH": "/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin"}


def run(command: list[str], label: str) -> bool:
    done = subprocess.run(
        command, capture_output=True, text=True, check=False, cwd=PROJECT_ROOT, env=ENV
    )
    if done.returncode != 0:
        sys.stderr.write(f"\n[{label}] failed:\n{done.stdout[-1200:]}{done.stderr[-1200:]}\n")
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=40)
    parser.add_argument("--prefix", default="asset")
    parser.add_argument("--seed", type=int, default=90210)
    parser.add_argument("--resolution", type=int, default=384)
    parser.add_argument("--voxel-size-m", type=float, default=0.05)
    parser.add_argument("--pixel-stride", type=int, default=6)
    parser.add_argument("--views-per-object", type=int, default=2)
    parser.add_argument("--target-coverage", type=float, default=0.99)
    parser.add_argument("--max-views", type=int, default=48)
    args = parser.parse_args()

    assets_root = PROJECT_ROOT / "data" / "assets"
    manifest = load_manifest(assets_root)
    finished, failed = [], []

    for index in range(args.count):
        name = f"{args.prefix}{index:03d}"
        root = PROJECT_ROOT / "data" / "spatial_lens" / name
        started = time.perf_counter()

        if not (root / "build.json").is_file():
            room = compose_room(name, manifest, seed=args.seed + index)
            root.mkdir(parents=True, exist_ok=True)
            build = room.build_payload()
            (root / "build.json").write_text(
                json.dumps(build, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            key_dir = root / "scorer_only"
            key_dir.mkdir(exist_ok=True)
            (key_dir / "room_key.json").write_text(
                json.dumps(room.key_payload(), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

        size = json.loads((root / "build.json").read_text(encoding="utf-8"))["room_size_m"]
        stages = [
            (root / "scene.blend", [
                BLENDER, "--background", "--python",
                str(PROJECT_ROOT / "blender" / "build_asset_room.py"), "--",
                "--build", str(root / "build.json"),
                "--assets", str(assets_root),
                "--output", str(root / "scene.blend"),
                "--measured", str(root / "measured_geometry.json"),
            ], "build"),
            (root / "scan_plan.json", [
                BLENDER, "--background", str(root / "scene.blend"), "--python",
                str(PROJECT_ROOT / "blender" / "plan_scan_coverage.py"), "--",
                "--room-size", *[str(v) for v in size],
                "--output", str(root / "scan_plan.json"),
                "--target-coverage", str(args.target_coverage),
                "--max-views", str(args.max_views),
            ], "plan"),
            (root / "scans" / "manifest.json", [
                str(MAIN_PY), str(PROJECT_ROOT / "scripts" / "lens_scan_room.py"),
                "--room", name, "--resolution", str(args.resolution),
                "--plan", str(root / "scan_plan.json"),
            ], "scan"),
            (root / "point_cloud.npz", [
                str(GEMMA_PY), str(PROJECT_ROOT / "scripts" / "lens_perceive.py"),
                "--room", name, "--voxel-size-m", str(args.voxel_size_m),
                "--pixel-stride", str(args.pixel_stride),
            ], "perceive"),
            (root / "scene_graph.json", [
                str(GEMMA_PY), str(PROJECT_ROOT / "scripts" / "lens_understand.py"),
                "--room", name, "--views-per-object", str(args.views_per_object),
            ], "name"),
        ]

        broke = False
        for artifact, command, label in stages:
            if artifact.exists():
                continue
            if not run(command, f"{name}:{label}"):
                failed.append(f"{name}:{label}")
                broke = True
                break
        if broke:
            continue

        plan = json.loads((root / "scan_plan.json").read_text(encoding="utf-8"))
        graph = json.loads((root / "scene_graph.json").read_text(encoding="utf-8"))
        finished.append(name)
        print(
            f"{name}: {len(plan['views'])} views @ {plan['final_coverage']:.1%} coverage, "
            f"{len(graph['objects'])} objects named, "
            f"{time.perf_counter() - started:.0f}s"
        )

    print(f"\nfinished {len(finished)}, failed {len(failed)}")
    if failed:
        print("failures:", failed[:12])
    return 0 if finished else 1


if __name__ == "__main__":
    raise SystemExit(main())
