#!/usr/bin/env python3
"""Plan, run, and score model-only closed-loop goals in unseen rooms.

The three subcommands are deliberately separate processes so the rollout stage
can be audited for oracle isolation:

    plan     reads evaluation-only oracle geometry; writes tasks + targets
    rollout  reads ONLY the tasks file; blocks every oracle path at runtime
    score    joins rollouts with targets and applies fixed thresholds
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from semantic_3d_chat.evaluation.v15_heldout_closed_loop import (
    DEFAULT_BASE_CHECKPOINT,
    DEFAULT_CONFIG,
    DEFAULT_CONTROL_CHECKPOINT,
    DEFAULT_CONTROL_CONFIG,
    DEFAULT_ROBOT_STATE_CHECKPOINT,
    _atomic_json,
    plan_heldout_tasks,
    run_heldout_rollouts,
    score_heldout_rollouts,
)

SEALED_SCENES = (
    "scene_000051",
    "scene_000052",
    "scene_000053",
    "scene_000054",
    "scene_000055",
    "scene_000056",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    plan = commands.add_parser("plan")
    plan.add_argument("--scene", action="append", default=None)
    plan.add_argument("--oracle-root", default="data/oracle")
    plan.add_argument("--object-goals-per-scene", type=int, default=2)
    plan.add_argument("--max-steps-object", type=int, default=32)
    plan.add_argument("--max-steps-lap", type=int, default=128)
    plan.add_argument("--no-lap", action="store_true")
    plan.add_argument("--tasks", type=Path, required=True)
    plan.add_argument("--targets", type=Path, required=True)

    rollout = commands.add_parser("rollout")
    rollout.add_argument("--tasks", type=Path, required=True)
    rollout.add_argument("--navigation-checkpoint", required=True)
    rollout.add_argument("--config", default=DEFAULT_CONFIG)
    rollout.add_argument("--control-config", default=DEFAULT_CONTROL_CONFIG)
    rollout.add_argument("--base-checkpoint", default=DEFAULT_BASE_CHECKPOINT)
    rollout.add_argument("--control-checkpoint", default=DEFAULT_CONTROL_CHECKPOINT)
    rollout.add_argument(
        "--robot-state-checkpoint", default=DEFAULT_ROBOT_STATE_CHECKPOINT
    )
    rollout.add_argument("--output", type=Path, required=True)

    score = commands.add_parser("score")
    score.add_argument("--rollouts", type=Path, required=True)
    score.add_argument("--targets", type=Path, required=True)
    score.add_argument("--maximum-face-yaw-error-degrees", type=float, default=20.0)
    score.add_argument("--minimum-center-progress-m", type=float, default=0.25)
    score.add_argument("--maximum-bbox-standoff-m", type=float, default=0.60)
    score.add_argument("--minimum-lap-path-length-m", type=float, default=5.0)
    score.add_argument("--minimum-lap-absolute-area-m2", type=float, default=0.5)
    score.add_argument("--maximum-lap-return-error-m", type=float, default=0.75)
    score.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "plan":
        scenes = tuple(args.scene) if args.scene else SEALED_SCENES
        tasks, targets = plan_heldout_tasks(
            scenes,
            oracle_root=args.oracle_root,
            object_goals_per_scene=args.object_goals_per_scene,
            include_lap=not args.no_lap,
            max_steps_object=args.max_steps_object,
            max_steps_lap=args.max_steps_lap,
        )
        _atomic_json(args.tasks.resolve(), tasks)
        _atomic_json(args.targets.resolve(), targets)
        print(json.dumps({"tasks": tasks["task_count"], "scenes": list(scenes)}, indent=2))
        return 0
    if args.command == "rollout":
        plan = json.loads(args.tasks.read_text(encoding="utf-8"))

        def report(record: dict[str, object]) -> None:
            print(
                f"rollout goal={record['goal_id']} termination={record['termination']} "
                f"decisions={record['decision_count']} "
                f"rejected={record['rejected_decision_count']}",
                flush=True,
            )

        result = run_heldout_rollouts(
            plan,
            navigation_checkpoint=args.navigation_checkpoint,
            config=args.config,
            control_config=args.control_config,
            base_checkpoint=args.base_checkpoint,
            control_checkpoint=args.control_checkpoint,
            robot_state_checkpoint=args.robot_state_checkpoint,
            progress=report,
        )
        _atomic_json(args.output.resolve(), result)
        print(json.dumps({"rollouts": result["rollout_count"]}, indent=2))
        return 0
    rollouts = json.loads(args.rollouts.read_text(encoding="utf-8"))
    targets = json.loads(args.targets.read_text(encoding="utf-8"))
    report = score_heldout_rollouts(
        rollouts,
        targets,
        maximum_face_yaw_error_degrees=args.maximum_face_yaw_error_degrees,
        minimum_center_progress_m=args.minimum_center_progress_m,
        maximum_bbox_standoff_m=args.maximum_bbox_standoff_m,
        minimum_lap_path_length_m=args.minimum_lap_path_length_m,
        minimum_lap_absolute_area_m2=args.minimum_lap_absolute_area_m2,
        maximum_lap_return_error_m=args.maximum_lap_return_error_m,
    )
    _atomic_json(args.output.resolve(), report)
    print(json.dumps({key: report[key] for key in ("goal_count", "passed_count", "pass_rate", "per_metric")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
