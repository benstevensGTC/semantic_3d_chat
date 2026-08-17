#!/usr/bin/env python3
"""Score how well Gemma can locate objects using only the 3D semantic field.

For each object the perception stack discovered, ask where it is, reading the
answer out of the bird's-eye 3D tokens alone. The reply is a grid cell, which is
converted to metres and compared against the object's measured position.

Scrambled and zeroed copies of the same tokens are scored identically, because
a localization accuracy that survives those was not grounded in the scene.
"""

from __future__ import annotations

import argparse
import json
import math

from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.spatial_lens.perceive import SemanticCloud
from semantic_3d_chat.spatial_lens.reason_3d import locate_3d
from semantic_3d_chat.spatial_lens.scene_graph import SceneGraph
from semantic_3d_chat.spatial_lens.scene_tokens_3d import (
    build_scene_tokens_3d,
    shuffled,
    zeroed,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--room", required=True)
    parser.add_argument("--grid", type=int, default=16)
    parser.add_argument("--tolerance-m", type=float, default=1.0)
    parser.add_argument("--controls", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()

    from semantic_3d_chat.language.local_lm import load_local_language_model

    room_root = PROJECT_ROOT / "data" / "spatial_lens" / args.room
    cloud = SemanticCloud.load(room_root / "point_cloud.npz")
    graph = SceneGraph.load(room_root / "scene_graph.json")
    tokens = build_scene_tokens_3d(cloud, grid=args.grid)
    print(f"3D tokens: {tokens.token_count}, cell = "
          f"{cloud.room_size_m[0] / args.grid:.2f} x "
          f"{cloud.room_size_m[1] / args.grid:.2f} m")

    language = load_local_language_model(
        "google/gemma-4-E2B-it",
        revision="3e22461f65e89153144f8adb70e3b8c2cc9845a7",
        requested_dtype="bfloat16",
        local_files_only=True,
    )

    conditions = {"scene": tokens}
    if args.controls:
        conditions["shuffled"] = shuffled(tokens)
        conditions["zeroed"] = zeroed(tokens)

    rows = []
    summary = {}
    for condition, variant in conditions.items():
        hits = 0
        errors = []
        print(f"\n[{condition}]")
        for item in graph.objects:
            truth = (item.center_m[0], item.center_m[1])
            found = locate_3d(language, variant, item.name)
            if found is None:
                print(f"  {item.name:<16} no answer")
                rows.append(
                    {"condition": condition, "object": item.name, "located": False}
                )
                continue
            error = math.dist(found, truth)
            within = error <= args.tolerance_m
            hits += int(within)
            errors.append(error)
            print(
                f"  {item.name:<16} said ({found[0]:+.2f}, {found[1]:+.2f}) "
                f"true ({truth[0]:+.2f}, {truth[1]:+.2f})  err {error:.2f} m"
                f"  {'HIT' if within else ''}"
            )
            rows.append(
                {
                    "condition": condition,
                    "object": item.name,
                    "located": True,
                    "said_m": [round(v, 3) for v in found],
                    "true_m": [round(v, 3) for v in truth],
                    "error_m": round(error, 3),
                    "within_tolerance": within,
                }
            )
        total = len(graph.objects)
        summary[condition] = {
            "objects": total,
            "within_tolerance": hits,
            "accuracy": hits / max(1, total),
            "median_error_m": (
                round(sorted(errors)[len(errors) // 2], 3) if errors else None
            ),
        }
        print(f"  -> {hits}/{total} within {args.tolerance_m} m")

    # Chance: a uniformly random cell landing within tolerance of the truth.
    width, depth, _ = cloud.room_size_m
    chance = min(1.0, math.pi * args.tolerance_m**2 / (width * depth))
    print(f"\nrandom-guess baseline at this tolerance: {chance:.1%}")
    for name, value in summary.items():
        print(f"  {name:<9} {value['accuracy']:.0%} "
              f"(median error {value['median_error_m']} m)")

    if args.output:
        destination = PROJECT_ROOT / args.output
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(
                {
                    "schema": "semantic_3d_chat.spatial_lens.locate_3d.v1",
                    "room": args.room,
                    "grid": args.grid,
                    "tolerance_m": args.tolerance_m,
                    "random_baseline": round(chance, 4),
                    "scene_graph_used_in_prompt": False,
                    "summary": summary,
                    "results": rows,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
