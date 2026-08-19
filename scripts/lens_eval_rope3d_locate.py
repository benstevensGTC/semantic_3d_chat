#!/usr/bin/env python3
"""Measure whether 3D rotary position lets Gemma say *where* something is.

Describing the room from the 3D field already worked; reading a position out of
it did not, scoring at the random baseline whether the tokens were real,
scrambled or zero. That is the failure this is aimed at, so it is measured the
same way and on the same readout.

Four conditions, all sharing one set of scene tokens:

  raster            position implied by the order the tokens arrive in
  rope3d            position drives the decoder's own rotary encoding
  rope3d_scrambled  same, with the positions permuted among the tokens
  zeroed            no scene at all

The third is the one that matters. It changes nothing except which place each
token claims to be, so if it does not undo the gain, the gain was not geometry.
"""

from __future__ import annotations

import argparse
import json
import random

import numpy as np

from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.spatial_lens.discover import discover_objects
from semantic_3d_chat.spatial_lens.grounding_data import available_rooms
from semantic_3d_chat.spatial_lens.perceive import SemanticCloud
from semantic_3d_chat.spatial_lens.reason_3d import locate_3d
from semantic_3d_chat.spatial_lens.scene_tokens_3d import build_scene_tokens_3d, zeroed

CONDITIONS = (
    "raster",
    "rope3d_xyz",
    "rope3d_z_only",
    "rope3d_scrambled",
    "zeroed",
)


def scrambled_positions(tokens, seed: int = 20260818):
    """Same semantics, wrong places: the control that isolates the geometry."""

    from dataclasses import replace

    order = np.random.default_rng(seed).permutation(tokens.centroids_m.shape[0])
    return replace(tokens, centroids_m=tokens.centroids_m[order].copy())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdout", type=int, default=8)
    parser.add_argument("--grid", type=int, default=16)
    parser.add_argument("--span-units", type=float, default=256.0)
    parser.add_argument("--tolerance-m", type=float, default=1.0)
    parser.add_argument("--report",
                        default="reports/gemma4/metrics/rope3d_locate.json")
    args = parser.parse_args()

    shuffled = list(available_rooms())
    random.Random(20260818).shuffle(shuffled)
    rooms = sorted(shuffled[: args.holdout])
    print(f"held-out rooms: {rooms}")

    from semantic_3d_chat.language.local_lm import load_local_language_model

    language = load_local_language_model(
        "google/gemma-4-E2B-it",
        revision="3e22461f65e89153144f8adb70e3b8c2cc9845a7",
        requested_dtype="bfloat16",
        local_files_only=True,
    )

    results = {name: {"errors": [], "hits": [], "refused": 0} for name in CONDITIONS}
    chance = []
    trials = []
    for room in rooms:
        root = PROJECT_ROOT / "data" / "spatial_lens" / room
        cloud = SemanticCloud.load(root / "point_cloud.npz")
        tokens = build_scene_tokens_3d(cloud, grid=args.grid)
        named = {
            item["object_id"]: item["name"]
            for item in json.loads((root / "scene_graph.json").read_text())["objects"]
        }
        width, depth, _ = cloud.room_size_m
        centers = np.asarray(cloud.centers_m, dtype=np.float64)
        variants = {
            "raster": (tokens, False, "xyz"),
            "rope3d_xyz": (tokens, True, "xyz"),
            "rope3d_z_only": (tokens, True, "z_only"),
            "rope3d_scrambled": (scrambled_positions(tokens), True, "xyz"),
            "zeroed": (zeroed(tokens), False, "xyz"),
        }
        for proposal in discover_objects(cloud):
            name = named.get(proposal.proposal_id)
            if not name or name == "unidentified object":
                continue
            footprint = centers[proposal.voxel_indices][:, :2]
            # A random cell's chance of landing within tolerance of this object.
            cells = np.stack(
                np.meshgrid(
                    (np.arange(args.grid) + 0.5) * width / args.grid - width / 2,
                    (np.arange(args.grid) + 0.5) * depth / args.grid - depth / 2,
                    indexing="xy",
                ),
                axis=-1,
            ).reshape(-1, 2)
            near = np.linalg.norm(
                cells[:, None, :] - footprint[None, :, :], axis=2
            ).min(axis=1)
            chance.append(float((near <= args.tolerance_m).mean()))

            for condition, (field, use_rope, axes) in variants.items():
                answer = locate_3d(language, field, name, rope3d=use_rope, axes=axes)
                if answer is None:
                    results[condition]["refused"] += 1
                    results[condition]["errors"].append(float("inf"))
                    results[condition]["hits"].append(0.0)
                    continue
                gap = float(
                    np.linalg.norm(np.asarray(answer)[None, :] - footprint, axis=1).min()
                )
                results[condition]["errors"].append(gap)
                results[condition]["hits"].append(float(gap <= args.tolerance_m))
                trials.append(
                    {"room": room, "object": name, "condition": condition,
                     "answer_m": [round(v, 3) for v in answer], "gap_m": round(gap, 3)}
                )
            print(f"  {room:8s} {name:22s} " + "  ".join(
                f"{c}={results[c]['hits'][-1]:.0f}" for c in CONDITIONS))

    summary = {
        "held_out_rooms": rooms,
        "tolerance_m": args.tolerance_m,
        "span_units": args.span_units,
        "queries_per_condition": len(results["raster"]["hits"]),
        "random_baseline": round(float(np.mean(chance)), 4),
        "conditions": {
            name: {
                "within_tolerance": round(float(np.mean(data["hits"])), 4),
                "median_gap_m": round(
                    float(np.median([e for e in data["errors"] if np.isfinite(e)]) )
                    if any(np.isfinite(e) for e in data["errors"]) else float("nan"), 3),
                "refused": data["refused"],
            }
            for name, data in results.items()
        },
        "trials": trials,
    }
    destination = PROJECT_ROOT / args.report
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "trials"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
