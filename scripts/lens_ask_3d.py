#!/usr/bin/env python3
"""Ask Gemma about a room using only the 3D semantic field as evidence.

No scene graph, no object list, no caption. The prompt is Gemma's own fused 3D
tokens plus a question. Every question is repeated against a scrambled-layout
copy and a zeroed copy, because an answer that survives those was not grounded.
"""

from __future__ import annotations

import argparse
import json

from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.spatial_lens.perceive import SemanticCloud
from semantic_3d_chat.spatial_lens.reason_3d import ask_3d
from semantic_3d_chat.spatial_lens.scene_tokens_3d import (
    build_scene_tokens_3d,
    shuffled,
    zeroed,
)

DEFAULT_QUESTIONS = (
    "List the objects you can see in this room.",
    "Is there any furniture on the left side of the room?",
    "What is the largest object in the room?",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--room", required=True)
    parser.add_argument("--question", action="append", default=None)
    parser.add_argument("--grid", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--controls", action="store_true", help="also run scrambled/zeroed")
    parser.add_argument("--output")
    args = parser.parse_args()

    from semantic_3d_chat.language.local_lm import load_local_language_model

    cloud = SemanticCloud.load(
        PROJECT_ROOT / "data" / "spatial_lens" / args.room / "point_cloud.npz"
    )
    tokens = build_scene_tokens_3d(cloud, grid=args.grid)
    print(
        f"3D scene tokens: {tokens.token_count} "
        f"({tokens.grid}x{tokens.grid} columns), "
        f"{int(tokens.occupancy.sum())} occupied"
    )

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
    for question in (args.question or list(DEFAULT_QUESTIONS)):
        print(f"\nQ: {question}")
        for name, variant in conditions.items():
            answer = ask_3d(
                language, variant, question, max_new_tokens=args.max_new_tokens
            )
            rows.append({"question": question, "condition": name, "answer": answer})
            print(f"  [{name:8}] {answer}", flush=True)

    if args.output:
        destination = PROJECT_ROOT / args.output
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(
                {
                    "schema": "semantic_3d_chat.spatial_lens.qa_3d.v1",
                    "room": args.room,
                    "grid": args.grid,
                    "token_count": tokens.token_count,
                    "occupied_columns": int(tokens.occupancy.sum()),
                    "scene_graph_used": False,
                    "object_list_in_prompt": False,
                    "exchanges": rows,
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
