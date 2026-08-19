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


def _grounding_locator(room: str, head_path: str) -> dict:
    """Load the grounding head and return a phrase -> metres callable."""

    import numpy as np
    import torch

    from semantic_3d_chat.spatial_lens.grounding import load_head
    from semantic_3d_chat.spatial_lens.grounding_data import embed_phrases
    from semantic_3d_chat.spatial_lens.perceive import SemanticCloud
    from semantic_3d_chat.spatial_lens.scene_tokens_3d import build_scene_tokens_3d

    head, metadata = load_head(PROJECT_ROOT / head_path)
    cloud = SemanticCloud.load(
        PROJECT_ROOT / "data" / "spatial_lens" / room / "point_cloud.npz"
    )
    tokens = build_scene_tokens_3d(cloud, grid=int(metadata["grid"]))
    status = (
        "room was HELD OUT of head training"
        if room in metadata.get("heldout_rooms", [])
        else "room was in head training"
        if room in metadata.get("train_rooms", [])
        else "room is new to the head"
    )

    from semantic_3d_chat.language.local_lm import load_local_language_model

    language = load_local_language_model(
        "google/gemma-4-E2B-it",
        revision="3e22461f65e89153144f8adb70e3b8c2cc9845a7",
        requested_dtype="bfloat16", local_files_only=True,
    )
    cache: dict[str, tuple[float, float]] = {}

    def locate(phrase: str) -> tuple[float, float]:
        if phrase not in cache:
            vector = embed_phrases(language, [phrase])
            with torch.no_grad():
                logits = head(
                    torch.from_numpy(tokens.tokens).unsqueeze(0).float(),
                    torch.from_numpy(vector).float(),
                )[0]
            cache[phrase] = tokens.cell_center_m(int(np.asarray(logits).argmax()))
        return cache[phrase]

    return {"locate": locate, "status": status}


def _point_locator(room: str, model_path: str) -> dict:
    """Read the target's position out of the point cloud itself.

    The grid locator above answers with a cell centre, so its precision stops
    at a third of a metre no matter how sure it is. This one answers with a
    weighted position over real points, which is a place in the room rather
    than a box the place falls inside.
    """

    import torch

    from semantic_3d_chat.spatial_lens.grounding_data import embed_phrases
    from semantic_3d_chat.spatial_lens.perceive import SemanticCloud
    from semantic_3d_chat.spatial_lens.point_grounding import load_model
    from semantic_3d_chat.spatial_lens.point_grounding_data import downsample

    model, metadata = load_model(PROJECT_ROOT / model_path)
    cloud = SemanticCloud.load(
        PROJECT_ROOT / "data" / "spatial_lens" / room / "point_cloud.npz"
    )
    budget = int(metadata.get("token_budget", 1024))
    chosen = downsample(cloud, token_budget=budget, cell_m=0.14, seed=0)
    points = torch.from_numpy(cloud.centers_m[chosen]).unsqueeze(0).float()
    features = torch.from_numpy(
        cloud.features[chosen].astype("float32")
    ).unsqueeze(0).float()
    status = (
        "room was HELD OUT of training"
        if room in metadata.get("held_out_rooms", [])
        else "room was in training"
    )

    from semantic_3d_chat.language.local_lm import load_local_language_model

    language = load_local_language_model(
        "google/gemma-4-E2B-it",
        revision="3e22461f65e89153144f8adb70e3b8c2cc9845a7",
        requested_dtype="bfloat16", local_files_only=True,
    )
    cache: dict[str, tuple[float, float]] = {}

    def locate(phrase: str) -> tuple[float, float]:
        if phrase not in cache:
            vector = torch.from_numpy(embed_phrases(language, [phrase])).float()
            with torch.no_grad():
                where = model.predict_position(features, points, vector)[0]
            cache[phrase] = (float(where[0]), float(where[1]))
        return cache[phrase]

    return {"locate": locate, "status": status}


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
    parser.add_argument(
        "--target-source",
        choices=("graph", "grounding", "points"),
        default="graph",
        help=(
            "graph: the target's position comes from the metric scene graph. "
            "points: the same, read straight off the point cloud by the "
            "3D-rotary model, which answers with a position rather than a cell. "
            "grounding: it is read out of the 3D semantic field by the trained "
            "grounding head, so the rover is driving to a place the model "
            "located in the point cloud rather than looked up in a list."
        ),
    )
    parser.add_argument(
        "--grounding-head",
        default="data_gemma4/checkpoints/spatial_grounding_v1",
    )
    parser.add_argument(
        "--point-model",
        default="data_gemma4/checkpoints/point_grounding_rope3d",
    )
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

    locator = None
    if args.target_source == "grounding":
        locator = _grounding_locator(args.room, args.grounding_head)
        print(f"targets located by the grounding head ({locator['status']})\n")
    elif args.target_source == "points":
        locator = _point_locator(args.room, args.point_model)
        print(f"targets located in the point cloud ({locator['status']})\n")
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
                if locator is not None:
                    # Read the target's position out of the 3D field, then take
                    # the nearest standable cell to it. The scene graph supplies
                    # only the vocabulary, never the coordinates.
                    located = locator["locate"](f"the {resolved.name}")
                    target_point = graph.nearest_free(*located) or located
                else:
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
        "target_source": args.target_source,
        "target_located_in_3d_field": args.target_source in {"grounding", "points"},
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
