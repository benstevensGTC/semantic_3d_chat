#!/usr/bin/env python3
"""Drive the rover from Gemma's spatial reasoning over the perceived room.

Gemma chooses every destination and heading. This script only checks whether a
proposal is physically legal, executes it exactly, and hands the outcome back.

    lens_drive.py --room studio --goal "Drive to the ball and stop next to it."
"""

from __future__ import annotations

import argparse
import json
import math

from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.spatial_lens.gemma_client import GemmaChat, OllamaChat
from semantic_3d_chat.spatial_lens.reasoning import (
    ASSISTED_NAV_SYSTEM,
    NAV_SYSTEM,
    navigation_prompt,
    parse_decision,
)
from semantic_3d_chat.spatial_lens.rover import (
    Rover,
    StepReceipt,
    choose_start_pose,
    follow_toward,
    probe_directions,
)
from semantic_3d_chat.spatial_lens.scene_graph import SceneGraph


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--room", required=True)
    parser.add_argument("--goal", required=True)
    parser.add_argument("--max-steps", type=int, default=16)
    parser.add_argument("--max-step-m", type=float, default=0.5)
    parser.add_argument("--start-x", type=float, default=None)
    parser.add_argument("--start-y", type=float, default=None)
    parser.add_argument(
        "--step-selection",
        choices=("assisted", "model"),
        default="assisted",
        help=(
            "assisted: Gemma picks the target and decides when to stop, and a "
            "local planner handles getting round furniture (this is what a real "
            "robot does, and it works). model: Gemma also chooses every metric "
            "step, which is the purer measurement and fails on detours."
        ),
    )
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

    graph = SceneGraph.load(
        PROJECT_ROOT / "data" / "spatial_lens" / args.room / "scene_graph.json"
    )
    rover = Rover(graph=graph, pose=choose_start_pose(graph), max_step_m=args.max_step_m)
    if args.start_x is not None and args.start_y is not None:
        placed = graph.nearest_free(args.start_x, args.start_y)
        if placed is None:
            raise SystemExit("Requested start pose has no free floor nearby")
        rover.pose = type(rover.pose)(placed[0], placed[1], 0.0)
        rover.path = [placed]

    chat = (
        OllamaChat.load(model=args.ollama_model)
        if args.reasoner == "ollama"
        else GemmaChat.load()
    )
    nav_system = (
        ASSISTED_NAV_SYSTEM if args.step_selection == "assisted" else NAV_SYSTEM
    )
    print(f"goal: {args.goal}")
    print(f"step selection: {args.step_selection}  reasoner: {args.reasoner}")
    print(f"start: ({rover.pose.x_m:+.2f}, {rover.pose.y_m:+.2f}) yaw {rover.pose.yaw_degrees:+.1f}\n")

    approach_points = {
        item.name: point
        for item in graph.objects
        if (point := graph.approach_point(item)) is not None
    }
    history: list[str] = []
    receipts: list[StepReceipt] = []
    termination = "max_steps"
    # The model names its target once; the harness carries it forward so that
    # per-direction progress can be measured for it on every later step.
    target_name: str | None = None
    target_point: tuple[float, float] | None = None
    for step in range(1, args.max_steps + 1):
        prompt = navigation_prompt(
            graph,
            rover.pose,
            args.goal,
            history,
            max_step_m=args.max_step_m,
            approach_points=approach_points,
            open_directions=probe_directions(rover),
            target=target_point,
            target_name=target_name,
            visited=rover.path,
        )
        reply = chat.ask_text(prompt, system=nav_system, max_new_tokens=160)
        before = rover.pose.as_dict()
        try:
            decision = parse_decision(reply)
        except ValueError as error:
            history.append(f"reply was not a usable action ({error})")
            print(f"  {step}: unparsable reply", flush=True)
            continue

        if decision.target:
            resolved = graph.find(decision.target)
            if resolved is not None:
                target_name = resolved.name
                target_point = approach_points.get(
                    resolved.name, (resolved.center_m[0], resolved.center_m[1])
                )

        if decision.action in {"MOVE_TO", "MOVE_TOWARD"}:
            if decision.x_m is None or decision.y_m is None:
                accepted, code, distance = False, "E_MISSING_TARGET", 0.0
            elif args.step_selection == "assisted":
                accepted, code, distance = follow_toward(
                    rover, decision.x_m, decision.y_m
                )
            elif decision.action == "MOVE_TOWARD":
                accepted, code, distance = rover.move_toward(decision.x_m, decision.y_m)
            else:
                accepted, code, distance = rover.move_to(decision.x_m, decision.y_m)
            requested = {"x": decision.x_m, "y": decision.y_m}
        elif decision.action == "FACE":
            if decision.yaw_degrees is None:
                accepted, code, distance = False, "E_MISSING_YAW", 0.0
            else:
                accepted, code, distance = rover.face(decision.yaw_degrees)
            requested = {"yaw_degrees": decision.yaw_degrees}
        else:
            accepted, code, distance = True, None, 0.0
            requested = {}

        receipt = StepReceipt(
            step=step,
            action=decision.action,
            requested=requested,
            accepted=accepted,
            error_code=code,
            pose_before=before,
            pose_after=rover.pose.as_dict(),
            distance_m=distance,
            reasoning=decision.reasoning,
        )
        receipts.append(receipt)
        history.append(receipt.summary())
        print(
            f"  {step}: {decision.action:<8} {receipt.summary():<52} :: {decision.reasoning}",
            flush=True,
        )
        if decision.action == "STOP":
            termination = "model_stop"
            break

    final = rover.pose
    print(
        f"\nfinished: {termination} at ({final.x_m:+.2f}, {final.y_m:+.2f}) "
        f"yaw {final.yaw_degrees:+.1f}, path {rover.path_length_m:.2f} m, "
        f"{sum(1 for r in receipts if not r.accepted)} rejected of {len(receipts)}"
    )

    report = {
        "schema": "semantic_3d_chat.spatial_lens.drive.v1",
        "room": args.room,
        "goal": args.goal,
        "termination": termination,
        "step_selection": args.step_selection,
        "reasoner": args.reasoner,
        "model_selected_target_and_termination": True,
        "model_selected_every_metric_step": args.step_selection == "model",
        "deterministic_local_planner_used": args.step_selection == "assisted",
        "final_pose": final.as_dict(),
        "path_length_m": round(rover.path_length_m, 4),
        "path": [[round(x, 4), round(y, 4)] for x, y in rover.path],
        "accepted": sum(1 for r in receipts if r.accepted),
        "rejected": sum(1 for r in receipts if not r.accepted),
        "steps": [r.as_dict() for r in receipts],
        "final_target": target_name,
        "objects": [
            {"name": item.name, "center_m": list(item.center_m[:2])}
            for item in graph.objects
        ],
    }
    if args.output:
        destination = PROJECT_ROOT / args.output
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    # A convenience read-out: how close the rover ended to each named object.
    print("\ndistance from final pose to each perceived object:")
    for item in sorted(
        graph.objects,
        key=lambda o: math.dist((final.x_m, final.y_m), o.center_m[:2]),
    ):
        print(
            f"  {item.name:<16} {math.dist((final.x_m, final.y_m), item.center_m[:2]):.2f} m"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
