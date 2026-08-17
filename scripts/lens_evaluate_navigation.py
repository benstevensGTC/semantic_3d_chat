#!/usr/bin/env python3
"""Drive to every perceived object in a room and score whether it arrived.

One goal per object, phrased in the model's own perceived vocabulary. A goal
counts as reached only if the model stopped itself AND ended within the
threshold of the object it was asked for -- ending up near something else, or
running out of steps in the right place, both count as failures.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys

from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.spatial_lens.scene_graph import SceneGraph


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--room", required=True)
    parser.add_argument("--step-selection", choices=("assisted", "model"), default="assisted")
    parser.add_argument("--max-steps", type=int, default=16)
    parser.add_argument("--arrival-m", type=float, default=1.2)
    parser.add_argument("--reasoner", choices=("gemma", "ollama"), default="gemma")
    parser.add_argument("--ollama-model", default="qwen3.8:27b")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    graph = SceneGraph.load(
        PROJECT_ROOT / "data" / "spatial_lens" / args.room / "scene_graph.json"
    )
    scratch = PROJECT_ROOT / "data" / "spatial_lens" / args.room / "nav_eval"
    scratch.mkdir(parents=True, exist_ok=True)

    rows = []
    for item in graph.objects:
        goal = f"Drive to the {item.name} and stop next to it."
        record = scratch / f"{item.object_id}.json"
        completed = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "lens_drive.py"),
                "--room", args.room,
                "--goal", goal,
                "--max-steps", str(args.max_steps),
                "--step-selection", args.step_selection,
                "--reasoner", args.reasoner,
                "--ollama-model", args.ollama_model,
                "--output", str(record.relative_to(PROJECT_ROOT)),
            ],
            capture_output=True,
            text=True,
            check=False,
            cwd=PROJECT_ROOT,
        )
        if completed.returncode != 0 or not record.is_file():
            rows.append({"object": item.name, "error": completed.stderr[-400:]})
            print(f"  {item.name:<16} ERROR", flush=True)
            continue
        drive = json.loads(record.read_text(encoding="utf-8"))
        final = (drive["final_pose"]["x_m"], drive["final_pose"]["y_m"])
        distance = math.dist(final, item.center_m[:2])
        # Nearest perceived object to where it actually stopped.
        nearest = min(
            graph.objects, key=lambda o: math.dist(final, o.center_m[:2])
        )
        stopped = drive["termination"] == "model_stop"
        reached = bool(stopped and distance <= args.arrival_m)
        rows.append(
            {
                "object": item.name,
                "goal": goal,
                "model_stopped": stopped,
                "final_distance_m": round(distance, 3),
                "nearest_object": nearest.name,
                "correct_object_is_nearest": nearest.name == item.name,
                "decisions": len(drive["steps"]),
                "rejected": drive["rejected"],
                "path_length_m": drive["path_length_m"],
                "reached": reached,
            }
        )
        print(
            f"  {item.name:<16} {'REACHED' if reached else 'failed ':<8} "
            f"{distance:5.2f} m  stop={stopped}  "
            f"{len(drive['steps'])} decisions, {drive['rejected']} rejected",
            flush=True,
        )

    scored = [row for row in rows if "reached" in row]
    summary = {
        "schema": "semantic_3d_chat.spatial_lens.navigation_eval.v1",
        "room": args.room,
        "step_selection": args.step_selection,
        "reasoner": args.reasoner,
        "arrival_threshold_m": args.arrival_m,
        "goal_count": len(scored),
        "reached_count": sum(int(row["reached"]) for row in scored),
        "reached_rate": (
            sum(int(row["reached"]) for row in scored) / max(1, len(scored))
        ),
        "model_stop_rate": (
            sum(int(row["model_stopped"]) for row in scored) / max(1, len(scored))
        ),
        "goals": rows,
    }
    destination = PROJECT_ROOT / args.output
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"\nreached {summary['reached_count']}/{summary['goal_count']} "
        f"({summary['reached_rate']:.0%}), model stopped itself "
        f"{summary['model_stop_rate']:.0%} of the time"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
