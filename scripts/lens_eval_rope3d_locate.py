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
from semantic_3d_chat.language.model_choice import add_model_arguments, revision_for
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
    # Nothing on this path is trained, so there is nothing to hold out from --
    # and at 53 queries the intervals were wide enough to swallow every effect
    # being argued about. All 27 rooms roughly triples the sample.
    parser.add_argument("--rooms", type=int, default=0,
                        help="cap on rooms measured; 0 uses every scanned room")
    parser.add_argument("--room-prefix", default=None,
                        help="only measure rooms whose name starts with this")
    add_model_arguments(parser)
    # A cloud belongs to the model that built it, so the map has to be selected
    # alongside the model that reads it.
    parser.add_argument("--cloud-name", default="point_cloud.npz")
    parser.add_argument("--grid", type=int, default=16)
    parser.add_argument("--span-units", type=float, default=256.0)
    parser.add_argument("--tolerance-m", type=float, default=1.0)
    parser.add_argument("--report",
                        default="reports/gemma4/metrics/rope3d_locate.json")
    args = parser.parse_args()

    shuffled = list(available_rooms())
    if args.room_prefix:
        shuffled = [r for r in shuffled if r.startswith(args.room_prefix)]
    random.Random(20260818).shuffle(shuffled)
    have = [
        r for r in shuffled
        if (PROJECT_ROOT / "data" / "spatial_lens" / r / args.cloud_name).is_file()
    ]
    rooms = sorted(have[: args.rooms] if args.rooms else have)
    print(f"measuring {len(rooms)} rooms")

    from semantic_3d_chat.language.local_lm import load_local_language_model

    language = load_local_language_model(
        args.model,
        revision=args.revision or revision_for(args.model),
        requested_dtype="bfloat16",
        local_files_only=True,
    )

    results = {
        name: {"errors": [], "hits": [], "answered": [], "refused": 0}
        for name in CONDITIONS
    }
    chance = []
    occupancy = []
    trials = []
    for room in rooms:
        root = PROJECT_ROOT / "data" / "spatial_lens" / room
        cloud = SemanticCloud.load(root / args.cloud_name)
        tokens = build_scene_tokens_3d(cloud, grid=args.grid)
        occupancy.append(tokens.occupied_fraction)
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

            outcome: dict[str, str] = {}
            for condition, (field, use_rope, axes) in variants.items():
                answer = locate_3d(
                    language, field, name,
                    rope3d=use_rope, axes=axes, span_units=args.span_units,
                )
                # A refusal is scored wrong so the arms share a denominator, and
                # recorded separately so a condition that merely declines more
                # often is not mistaken for one that reasons worse.
                if answer is None:
                    results[condition]["refused"] += 1
                    results[condition]["errors"].append(float("inf"))
                    results[condition]["hits"].append(0.0)
                    results[condition]["answered"].append(0.0)
                    outcome[condition] = "-"
                    trials.append(
                        {"room": room, "object": name, "condition": condition,
                         "answer_m": None, "gap_m": None, "refused": True}
                    )
                    continue
                gap = float(
                    np.linalg.norm(np.asarray(answer)[None, :] - footprint, axis=1).min()
                )
                results[condition]["errors"].append(gap)
                results[condition]["hits"].append(float(gap <= args.tolerance_m))
                results[condition]["answered"].append(1.0)
                outcome[condition] = f"{float(gap <= args.tolerance_m):.0f}"
                trials.append(
                    {"room": room, "object": name, "condition": condition,
                     "answer_m": [round(v, 3) for v in answer],
                     "gap_m": round(gap, 3), "refused": False}
                )
            print(f"  {room:8s} {name:22s} " + "  ".join(
                f"{c}={outcome[c]}" for c in CONDITIONS))

    from semantic_3d_chat.evaluation.proportions import (
        holm_adjust,
        mcnemar_exact,
        wilson_interval,
    )

    def condition_summary(data: dict) -> dict[str, object]:
        total = len(data["hits"])
        hits = int(sum(data["hits"]))
        answered_count = int(sum(data["answered"]))
        finite = [e for e in data["errors"] if np.isfinite(e)]
        return {
            "queries": total,
            "within_tolerance": round(hits / total, 3) if total else None,
            "interval_95": wilson_interval(hits, total),
            "refused": data["refused"],
            # Conditioned on the model answering at all, so it is NOT comparable
            # across conditions with different refusal rates -- it is here to
            # show whether a low score is bad aim or bad compliance.
            "within_tolerance_when_answered": (
                round(
                    sum(
                        h for h, a in zip(data["hits"], data["answered"], strict=True) if a
                    ) / answered_count, 3
                ) if answered_count else None
            ),
            "median_gap_m_answered": (
                round(float(np.median(finite)), 3) if finite else None
            ),
        }

    paired = {
        name: mcnemar_exact(results[name]["hits"], results["raster"]["hits"])
        for name in CONDITIONS if name != "raster"
    }
    summary = {
        "rooms": rooms,
        "room_prefix": args.room_prefix,
        "model": args.model,
        "cloud": args.cloud_name,
        "tolerance_m": args.tolerance_m,
        "span_units": args.span_units,
        "queries_per_condition": len(results["raster"]["hits"]),
        "random_baseline": round(float(np.mean(chance)), 4),
        # Empty columns are zero vectors, so this much of a "real" scene differs
        # from the zeroed control at all. Read that control accordingly.
        "occupied_fraction": round(float(np.mean(occupancy)), 3),
        "conditions": {
            name: condition_summary(data) for name, data in results.items()
        },
        "mcnemar_vs_raster": paired,
        "holm_adjusted_p": holm_adjust({k: v["p_value"] for k, v in paired.items()}),
        "trials": trials,
    }
    destination = PROJECT_ROOT / args.report
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "trials"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
