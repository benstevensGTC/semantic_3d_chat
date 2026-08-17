#!/usr/bin/env python3
"""Ask Gemma spatial questions about a room it perceived.

    lens_ask.py --room studio --question "What is between the bookshelf and the ball?"
    lens_ask.py --room studio            # interactive
"""

from __future__ import annotations

import argparse
import json

from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.spatial_lens.gemma_client import GemmaChat, OllamaChat
from semantic_3d_chat.spatial_lens.reasoning import answer_question
from semantic_3d_chat.spatial_lens.scene_graph import SceneGraph


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--room", required=True)
    parser.add_argument("--question", action="append", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=220)
    parser.add_argument("--show-map", action="store_true")
    parser.add_argument(
        "--reasoner",
        choices=("gemma", "ollama"),
        default="gemma",
        help=(
            "which local model does the reasoning. Perception always stays on "
            "Gemma; 'ollama' swaps only the reasoning layer for a larger local "
            "model, which is what makes unaided step-by-step navigation work."
        ),
    )
    parser.add_argument("--ollama-model", default="qwen3.8:27b")
    parser.add_argument("--output")
    args = parser.parse_args()

    graph_path = PROJECT_ROOT / "data" / "spatial_lens" / args.room / "scene_graph.json"
    graph = SceneGraph.load(graph_path)
    if args.show_map:
        print(graph.describe())
        print()

    chat = (
        OllamaChat.load(model=args.ollama_model)
        if args.reasoner == "ollama"
        else GemmaChat.load()
    )
    transcript = []

    def run(question: str) -> None:
        answer = answer_question(
            chat, graph, question, max_new_tokens=args.max_new_tokens
        )
        transcript.append({"question": question, "answer": answer})
        print(f"\nQ: {question}\nA: {answer}\n", flush=True)

    if args.question:
        for question in args.question:
            run(question)
    else:
        print("Ask about the room. Blank line or Ctrl-D exits.\n")
        while True:
            try:
                question = input("> ").strip()
            except EOFError:
                break
            if not question:
                break
            run(question)

    if args.output:
        destination = PROJECT_ROOT / args.output
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(
                {
                    "schema": "semantic_3d_chat.spatial_lens.qa.v1",
                    "room": args.room,
                    "object_names": [item.name for item in graph.objects],
                    "exchanges": transcript,
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
