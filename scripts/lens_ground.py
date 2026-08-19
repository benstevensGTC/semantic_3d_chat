#!/usr/bin/env python3
"""Locate objects in a room using the trained grounding head.

This is the payoff of the whole pipeline: a phrase goes in, the head reads the
room's per-cell semantic embeddings, and a metric position comes out -- in rooms
the head has never seen. It answers "where is the lamp" from the 3D field, which
is the piece a frozen decoder could not do.
"""

from __future__ import annotations

import argparse
import json
import math

import numpy as np
import torch

from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.spatial_lens.grounding import load_head
from semantic_3d_chat.spatial_lens.grounding_data import embed_phrases
from semantic_3d_chat.spatial_lens.perceive import SemanticCloud
from semantic_3d_chat.spatial_lens.scene_graph import SceneGraph
from semantic_3d_chat.spatial_lens.scene_tokens_3d import (
    build_scene_tokens_3d,
    shuffled,
    zeroed,
)


def ground(head, tokens, query_vector) -> tuple[float, float, np.ndarray]:
    with torch.no_grad():
        logits = head(
            torch.from_numpy(tokens.tokens).unsqueeze(0).float(),
            torch.from_numpy(query_vector).unsqueeze(0).float(),
        )[0]
    index = int(logits.argmax())
    x, y = tokens.cell_center_m(index)
    return x, y, torch.softmax(logits, -1).numpy()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--room", required=True)
    parser.add_argument("--phrase", action="append", default=None)
    parser.add_argument("--head", default="data_gemma4/checkpoints/spatial_grounding_v1")
    parser.add_argument("--controls", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()

    head, metadata = load_head(PROJECT_ROOT / args.head)
    if args.room in metadata.get("train_rooms", []):
        print(f"NOTE: {args.room} was a TRAINING room for this head\n")
    elif args.room in metadata.get("heldout_rooms", []):
        print(f"{args.room} was HELD OUT during training\n")

    root = PROJECT_ROOT / "data" / "spatial_lens" / args.room
    cloud = SemanticCloud.load(root / "point_cloud.npz")
    tokens = build_scene_tokens_3d(cloud, grid=int(metadata["grid"]))
    graph = SceneGraph.load(root / "scene_graph.json")

    phrases = args.phrase or [f"the {item.name}" for item in graph.objects]
    truth = {f"the {item.name}": item.center_m[:2] for item in graph.objects}

    from semantic_3d_chat.language.local_lm import load_local_language_model

    language = load_local_language_model(
        "google/gemma-4-E2B-it",
        revision="3e22461f65e89153144f8adb70e3b8c2cc9845a7",
        requested_dtype="bfloat16", local_files_only=True,
    )
    vectors = embed_phrases(language, phrases)
    del language

    variants = {"scene": tokens}
    if args.controls:
        variants["shuffled"] = shuffled(tokens)
        variants["zeroed"] = zeroed(tokens)

    rows, summary = [], {}
    for condition, variant in variants.items():
        errors = []
        print(f"[{condition}]")
        for phrase, vector in zip(phrases, vectors, strict=True):
            x, y, _ = ground(head, variant, vector)
            actual = truth.get(phrase)
            if actual is None:
                print(f"  {phrase:<24} -> ({x:+.2f}, {y:+.2f})")
                continue
            error = math.dist((x, y), actual)
            errors.append(error)
            print(f"  {phrase:<24} -> ({x:+.2f}, {y:+.2f})  true "
                  f"({actual[0]:+.2f}, {actual[1]:+.2f})  err {error:.2f} m")
            rows.append({"condition": condition, "phrase": phrase,
                         "predicted_m": [round(x, 3), round(y, 3)],
                         "true_m": [round(v, 3) for v in actual],
                         "error_m": round(error, 3)})
        if errors:
            summary[condition] = {
                "median_error_m": round(float(np.median(errors)), 3),
                "within_1m": round(float(np.mean([e <= 1.0 for e in errors])), 3),
            }
            print(f"  -> median {summary[condition]['median_error_m']:.2f} m, "
                  f"{summary[condition]['within_1m']:.0%} within 1 m\n")

    if args.output:
        destination = PROJECT_ROOT / args.output
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps({
            "schema": "semantic_3d_chat.spatial_lens.ground_eval.v1",
            "room": args.room,
            "was_held_out": args.room in metadata.get("heldout_rooms", []),
            "head": args.head, "summary": summary, "results": rows,
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
