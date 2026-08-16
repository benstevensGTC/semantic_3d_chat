"""Model-only closed-loop navigation in rooms the controller never trained on.

Every previous live acceptance ran in `scene_000001`, the single room the V4-V14
operator policy was fitted to.  That measures execution, not generalization.
This module runs the *same deployed* closed-loop controller
(:class:`GemmaWaypointClosedLoopController`) in arbitrary rooms, so the only
thing that changes between the accepted live demo and an unseen-room rollout is
which 258-token scene prefix and which numeric collision map are loaded.

Three deliberately separated stages keep the oracle out of inference:

``plan``
    Reads evaluation-only oracle geometry and emits two files: a ``tasks``
    file containing nothing but scene IDs and natural-language goals, and a
    ``targets`` file containing the geometry needed to score them.

``rollout``
    Consumes only the ``tasks`` file.  It runs under a
    :class:`FileAccessAudit` that *blocks* `data/oracle`, `data/qa`, and
    `data_gemma4/training`, so a rollout that touched oracle geometry would
    raise rather than silently score well.  No Blender asset and no rendering
    are required: model-only control keeps the map static, so the runtime is
    built with ``observation_scanner=None``.

``score``
    Joins completed rollouts with the ``targets`` file and applies fixed
    thresholds.

The rover is never handed a route, a waypoint list, a target coordinate, or a
stop rule.  It receives the room's continuous scene prefix, its own numeric
state and action history, and the goal text; Gemma emits every FACE, MOVE_TO,
and STOP.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from semantic_3d_chat.chat.file_audit import FileAccessAudit
from semantic_3d_chat.chat.question_control_runtime import QuestionControlledChatRuntime
from semantic_3d_chat.chat.runtime_config import load_runtime_config
from semantic_3d_chat.config import PROJECT_ROOT, load_config
from semantic_3d_chat.robot.gemma_runtime_binding import (
    question_controlled_gemma_runtime_binding,
)
from semantic_3d_chat.robot.gemma_waypoint_runtime import (
    GemmaWaypointClosedLoopController,
    load_gemma_waypoint_policy_checkpoint,
)
from semantic_3d_chat.robot.runtime_refresh import build_refreshing_embodied_runtime

PLAN_SCHEMA: Final[str] = "semantic_3d_chat.v15_heldout_closed_loop_plan.v1"
TARGET_SCHEMA: Final[str] = "semantic_3d_chat.v15_heldout_closed_loop_targets.v1"
ROLLOUT_SCHEMA: Final[str] = "semantic_3d_chat.v15_heldout_closed_loop_rollout.v1"
SCORE_SCHEMA: Final[str] = "semantic_3d_chat.v15_heldout_closed_loop_score.v1"

# Room shell and openings are not navigation targets.
STRUCTURAL_CATEGORIES: Final[frozenset[str]] = frozenset(
    {"ceiling", "door", "floor", "wall"}
)
_PROTECTED_COMPONENTS: Final[frozenset[str]] = frozenset(
    {"oracle", "qa", "training", "scorer", "scorer_only", "scorer-only"}
)

DEFAULT_CONFIG: Final[str] = "configs/runtime/embodied_live.yaml"
DEFAULT_CONTROL_CONFIG: Final[str] = "configs/runtime/gemma4_v56_question_control.yaml"
DEFAULT_BASE_CHECKPOINT: Final[str] = (
    "data_gemma4/runtime/checkpoints/gemma4_v54_release_v1"
)
DEFAULT_CONTROL_CHECKPOINT: Final[str] = (
    "data_gemma4/runtime/checkpoints/gemma4_v75_nll_control_release_v1"
)
DEFAULT_ROBOT_STATE_CHECKPOINT: Final[str] = (
    "data_gemma4/checkpoints/robot_state_numeric_v1"
)


def _rooted(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    rooted = candidate if candidate.is_absolute() else PROJECT_ROOT / candidate
    return Path(os.path.abspath(rooted))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, suffix=".partial")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _planar_distance(left: Sequence[float], right: Sequence[float]) -> float:
    return math.hypot(float(left[0]) - float(right[0]), float(left[1]) - float(right[1]))


def _point_to_xy_box_distance(
    point: Sequence[float], minimum: Sequence[float], maximum: Sequence[float]
) -> float:
    dx = max(float(minimum[0]) - float(point[0]), 0.0, float(point[0]) - float(maximum[0]))
    dy = max(float(minimum[1]) - float(point[1]), 0.0, float(point[1]) - float(maximum[1]))
    return math.hypot(dx, dy)


def _shortest_angle_error_degrees(observed: float, expected: float) -> float:
    return abs((float(observed) - float(expected) + 180.0) % 360.0 - 180.0)


# An object whose bounding box reaches close to the floor is something the
# rover can drive up to; anything resting on furniture is not.
_FLOOR_STANDING_MAX_Z_M: Final[float] = 0.20


def _is_floor_standing(instance: Mapping[str, Any]) -> bool:
    bbox = instance.get("bbox")
    if not isinstance(bbox, Mapping):
        return False
    minimum = bbox.get("min_xyz_m")
    if not isinstance(minimum, Sequence) or len(minimum) < 3:
        return False
    return float(minimum[2]) <= _FLOOR_STANDING_MAX_Z_M


def _footprint_area_m2(instance: Mapping[str, Any]) -> float:
    bbox = instance.get("bbox")
    if not isinstance(bbox, Mapping):
        return 0.0
    minimum = bbox.get("min_xyz_m")
    maximum = bbox.get("max_xyz_m")
    if (
        not isinstance(minimum, Sequence)
        or not isinstance(maximum, Sequence)
        or len(minimum) < 2
        or len(maximum) < 2
    ):
        return 0.0
    return abs(float(maximum[0]) - float(minimum[0])) * abs(
        float(maximum[1]) - float(minimum[1])
    )


def _target_heading_degrees(
    origin_xy: Sequence[float], target_xy: Sequence[float]
) -> float:
    """Project convention: yaw 0 faces +Y and yaw -90 faces +X."""

    dx = float(target_xy[0]) - float(origin_xy[0])
    dy = float(target_xy[1]) - float(origin_xy[1])
    return math.degrees(math.atan2(-dx, dy))


# ---------------------------------------------------------------------------
# Stage 1: plan (evaluation-only oracle read)
# ---------------------------------------------------------------------------


def plan_heldout_tasks(
    scene_ids: Sequence[str],
    *,
    oracle_root: str | Path = "data/oracle",
    object_goals_per_scene: int = 2,
    include_lap: bool = True,
    max_steps_object: int = 32,
    max_steps_lap: int = 128,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Derive goal text plus a separate scoring key from oracle geometry.

    The returned pair is ``(tasks, targets)``.  ``tasks`` is what the rollout
    process is allowed to see: opaque scene IDs and English goals.  ``targets``
    holds the geometry and never reaches inference.
    """

    if not scene_ids:
        raise ValueError("At least one held-out scene is required")
    if object_goals_per_scene < 0 or max_steps_object < 1 or max_steps_lap < 1:
        raise ValueError("Held-out plan bounds must be positive")
    root = _rooted(oracle_root)
    tasks: list[dict[str, Any]] = []
    targets: list[dict[str, Any]] = []
    oracle_files: dict[str, str] = {}
    for scene_id in scene_ids:
        oracle_path = root / scene_id / "oracle.json"
        oracle = json.loads(oracle_path.read_text(encoding="utf-8"))
        if oracle.get("scene_id") != scene_id:
            raise ValueError(f"Oracle scene identity differs for {scene_id}")
        oracle_files[scene_id] = _sha256_file(oracle_path)
        instances = oracle.get("instances")
        if not isinstance(instances, list):
            raise TypeError(f"Oracle instances must be a list for {scene_id}")
        counts: dict[str, int] = {}
        for item in instances:
            if not isinstance(item, Mapping):
                continue
            category = item.get("category")
            if isinstance(category, str) and category not in STRUCTURAL_CATEGORIES:
                counts[category] = counts.get(category, 0) + 1
        # Only unambiguous categories can be scored, and only an unambiguous
        # phrase is a fair instruction: "the chair" must denote one chair.
        unique = sorted(name for name, count in counts.items() if count == 1)
        by_category = {
            str(item["category"]): item
            for item in instances
            if isinstance(item, Mapping) and item.get("category") in unique
        }
        # Prefer floor-standing furniture. A book resting on a table has a tiny
        # footprint that the table itself makes physically unreachable, so an
        # approach standoff threshold against it would score the room's
        # geometry rather than the controller.
        chosen = sorted(
            unique,
            key=lambda name: (
                -int(_is_floor_standing(by_category[name])),
                -_footprint_area_m2(by_category[name]),
                name,
            ),
        )[:object_goals_per_scene]
        for category in chosen:
            match = by_category[category]
            center = [float(value) for value in match["expected_center_xyz_m"]]
            bbox = match["bbox"]
            for kind, instruction, metric in (
                ("face", f"Face the {category}, then stop.", "face_yaw"),
                (
                    "approach",
                    f"Move close to the {category}, then stop.",
                    "approach_standoff",
                ),
            ):
                goal_id = f"{scene_id}_{kind}_{category.replace(' ', '_')}"
                tasks.append(
                    {
                        "goal_id": goal_id,
                        "scene_id": scene_id,
                        "instruction": instruction,
                        "max_steps": int(max_steps_object),
                    }
                )
                targets.append(
                    {
                        "goal_id": goal_id,
                        "scene_id": scene_id,
                        "metric": metric,
                        "target_category": category,
                        "target_instance_id": match.get("instance_id"),
                        "target_center_xyz_m": center,
                        "target_bbox_min_xyz_m": [
                            float(value) for value in bbox["min_xyz_m"]
                        ],
                        "target_bbox_max_xyz_m": [
                            float(value) for value in bbox["max_xyz_m"]
                        ],
                    }
                )
        if include_lap:
            goal_id = f"{scene_id}_lap"
            tasks.append(
                {
                    "goal_id": goal_id,
                    "scene_id": scene_id,
                    "instruction": "Do a lap around the room.",
                    "max_steps": int(max_steps_lap),
                }
            )
            targets.append(
                {
                    "goal_id": goal_id,
                    "scene_id": scene_id,
                    "metric": "lap_circuit",
                }
            )
    plan = {
        "schema": PLAN_SCHEMA,
        "scene_ids": list(scene_ids),
        "task_count": len(tasks),
        "contains_target_geometry": False,
        "tasks": tasks,
    }
    key = {
        "schema": TARGET_SCHEMA,
        "scene_ids": list(scene_ids),
        "oracle_root": root.as_posix(),
        "oracle_file_sha256": oracle_files,
        "targets": targets,
    }
    return plan, key


# ---------------------------------------------------------------------------
# Stage 2: rollout (no oracle, no renderer)
# ---------------------------------------------------------------------------


def _heldout_audit() -> FileAccessAudit:
    return FileAccessAudit(
        [
            PROJECT_ROOT / "data" / "oracle",
            PROJECT_ROOT / "data" / "qa",
            PROJECT_ROOT / "data_gemma4" / "training",
            PROJECT_ROOT / "reports" / "gemma4" / "scorer_only",
        ],
        forbidden_component_names=_PROTECTED_COMPONENTS,
        block_forbidden=True,
    )


def build_headless_waypoint_controller(
    scene_id: str,
    *,
    config: str | Path = DEFAULT_CONFIG,
    control_config: str | Path = DEFAULT_CONTROL_CONFIG,
    base_checkpoint: str | Path = DEFAULT_BASE_CHECKPOINT,
    control_checkpoint: str | Path = DEFAULT_CONTROL_CHECKPOINT,
    robot_state_checkpoint: str | Path = DEFAULT_ROBOT_STATE_CHECKPOINT,
    navigation_checkpoint: str | Path,
    persistent_map_root: str | Path = "data_gemma4/robot/heldout_closed_loop",
    audit: FileAccessAudit | None = None,
) -> tuple[GemmaWaypointClosedLoopController, Any, dict[str, Any]]:
    """Load the deployed closed-loop controller for one arbitrary room.

    This mirrors ``build_local_practical_rover`` with two differences: no
    Blender asset or scanner is required (model-only control never scans), and
    no conversational wrapper is constructed.  The waypoint controller itself,
    its checkpoint authentication, and its runtime binding checks are the exact
    deployed ones.
    """

    embodied_config = load_config(_rooted(config))
    control_runtime_config = load_runtime_config(
        str(_rooted(control_config)),
        record_file=(None if audit is None else audit.record),
    )
    base = _rooted(base_checkpoint)
    control = _rooted(control_checkpoint)
    chat = QuestionControlledChatRuntime.load(
        control_runtime_config,
        scene_id,
        base_checkpoint=base,
        control_checkpoint=control,
        audit=audit,
    )
    runtime = build_refreshing_embodied_runtime(
        embodied_config,
        scene_id,
        checkpoint=base,
        chat_runtime=chat,
        persistent_map_path=_rooted(persistent_map_root)
        / scene_id
        / "semantic_map.npz",
        observation_scanner=None,
        robot_state_checkpoint=_rooted(robot_state_checkpoint),
        audit=audit,
        local_files_only=True,
    )
    if getattr(runtime, "auto_scan_after_motion", None) is not False:
        raise ValueError("Held-out rollouts require a static map")
    base_chat = getattr(chat, "base", chat)
    language_runtime = getattr(base_chat, "language", None)
    language_settings = control_runtime_config["language"]
    loaded = load_gemma_waypoint_policy_checkpoint(
        _rooted(navigation_checkpoint),
        prefix_backend=language_runtime.prefix_backend,
        tokenizer=language_runtime.tokenizer,
        expected_model_id=str(language_settings["model_id"]),
        expected_model_revision=str(language_settings["revision"]),
        expected_gemma_runtime_binding=question_controlled_gemma_runtime_binding(
            chat,
            control_runtime_config,
            base_checkpoint=base,
            control_checkpoint=control,
        ),
        audit=audit,
    )
    controller = GemmaWaypointClosedLoopController.from_loaded(
        runtime=runtime,
        config=embodied_config,
        loaded=loaded,
    )
    identity = {
        "scene_id": scene_id,
        "checkpoint": _rooted(navigation_checkpoint).as_posix(),
        "checkpoint_sha256": loaded.checkpoint_sha256,
        "scene_prefix_sha256": controller.scene_prefix_sha256,
    }
    return controller, runtime, identity


def _pose(state: Mapping[str, Any]) -> tuple[float, float, float]:
    return (
        float(state["position_m"][0]),
        float(state["position_m"][1]),
        float(state["body_yaw_degrees"]),
    )


def _trajectory_metrics(poses: Sequence[tuple[float, float, float]]) -> dict[str, Any]:
    path_length = sum(
        _planar_distance(poses[index - 1], poses[index]) for index in range(1, len(poses))
    )
    # Signed shoelace area of the closed polygon the rover actually traced.
    area = 0.0
    for index in range(len(poses)):
        x0, y0, _ = poses[index]
        x1, y1, _ = poses[(index + 1) % len(poses)]
        area += x0 * y1 - x1 * y0
    return {
        "path_length_m": path_length,
        "signed_swept_area_m2": 0.5 * area,
        "absolute_swept_area_m2": abs(0.5 * area),
        "return_distance_m": _planar_distance(poses[0], poses[-1]),
    }


def run_heldout_rollouts(
    plan: Mapping[str, Any],
    *,
    navigation_checkpoint: str | Path,
    config: str | Path = DEFAULT_CONFIG,
    control_config: str | Path = DEFAULT_CONTROL_CONFIG,
    base_checkpoint: str | Path = DEFAULT_BASE_CHECKPOINT,
    control_checkpoint: str | Path = DEFAULT_CONTROL_CHECKPOINT,
    robot_state_checkpoint: str | Path = DEFAULT_ROBOT_STATE_CHECKPOINT,
    progress: Any | None = None,
) -> dict[str, Any]:
    """Run every planned goal, rebuilding the runtime once per room."""

    if plan.get("schema") != PLAN_SCHEMA:
        raise ValueError("Held-out rollout requires a v15 plan file")
    if plan.get("contains_target_geometry") is not False:
        raise ValueError("Held-out rollout refuses a plan carrying target geometry")
    tasks = list(plan["tasks"])
    by_scene: dict[str, list[Mapping[str, Any]]] = {}
    for task in tasks:
        by_scene.setdefault(str(task["scene_id"]), []).append(task)

    audit = _heldout_audit()
    audit.__enter__()
    started = time.monotonic()
    rollouts: list[dict[str, Any]] = []
    identities: dict[str, Any] = {}
    try:
        for scene_id in sorted(by_scene):
            controller, runtime, identity = build_headless_waypoint_controller(
                scene_id,
                config=config,
                control_config=control_config,
                base_checkpoint=base_checkpoint,
                control_checkpoint=control_checkpoint,
                robot_state_checkpoint=robot_state_checkpoint,
                navigation_checkpoint=navigation_checkpoint,
                audit=audit,
            )
            identities[scene_id] = identity
            for task in by_scene[scene_id]:
                start_state = runtime.get_robot_state()
                start_pose = _pose(start_state)
                poses: list[tuple[float, float, float]] = [start_pose]
                begin = time.monotonic()
                result = controller.run(
                    str(task["instruction"]), max_steps=int(task["max_steps"])
                )
                accepted = 0
                rejected = 0
                actions: list[str] = []
                for receipt in result.receipts:
                    actions.append(receipt.action)
                    if receipt.execution.get("success") is True:
                        accepted += 1
                        # Accepted primitives return the simulator's exact
                        # post-action numeric state; rejections carry no pose.
                        position = receipt.execution.get("position_m")
                        poses.append(
                            (
                                float(position[0])
                                if isinstance(position, Sequence)
                                else poses[-1][0],
                                float(position[1])
                                if isinstance(position, Sequence)
                                else poses[-1][1],
                                float(
                                    receipt.execution.get(
                                        "body_yaw_degrees", poses[-1][2]
                                    )
                                ),
                            )
                        )
                    else:
                        rejected += 1
                final_pose = _pose(runtime.get_robot_state())
                poses.append(final_pose)
                record = {
                    "goal_id": str(task["goal_id"]),
                    "scene_id": scene_id,
                    "instruction": str(task["instruction"]),
                    "max_steps": int(task["max_steps"]),
                    "termination": result.termination,
                    "error_code": result.error_code,
                    "model_stop_emitted": bool(result.model_stop_emitted),
                    "controller_success": bool(result.success),
                    "decision_count": len(result.receipts),
                    "accepted_decision_count": accepted,
                    "rejected_decision_count": rejected,
                    "action_counts": {
                        name: actions.count(name)
                        for name in sorted(set(actions))
                    },
                    "start_pose_xy_yaw": list(start_pose),
                    "final_pose_xy_yaw": list(final_pose),
                    "elapsed_seconds": time.monotonic() - begin,
                    "checkpoint_sha256": identity["checkpoint_sha256"],
                    "scene_prefix_sha256": identity["scene_prefix_sha256"],
                    **_trajectory_metrics(poses),
                }
                rollouts.append(record)
                if progress is not None:
                    progress(record)
            # Free the room's Gemma runtime before loading the next room.
            del controller, runtime
    finally:
        audit.__exit__(None, None, None)
    return {
        "schema": ROLLOUT_SCHEMA,
        "rollout_process_read_oracle": False,
        "oracle_reads_blocked_by_audit": True,
        "renderer_used": False,
        "deterministic_route_planner_used": False,
        "model_selected_every_action": True,
        "navigation_checkpoint": _rooted(navigation_checkpoint).as_posix(),
        "scene_identities": identities,
        "elapsed_seconds": time.monotonic() - started,
        "rollout_count": len(rollouts),
        "rollouts": rollouts,
    }


# ---------------------------------------------------------------------------
# Stage 3: score (evaluation-only oracle geometry)
# ---------------------------------------------------------------------------


def score_heldout_rollouts(
    rollout_report: Mapping[str, Any],
    targets: Mapping[str, Any],
    *,
    maximum_face_yaw_error_degrees: float = 20.0,
    minimum_center_progress_m: float = 0.25,
    maximum_bbox_standoff_m: float = 0.60,
    minimum_lap_path_length_m: float = 5.0,
    minimum_lap_absolute_area_m2: float = 0.5,
    maximum_lap_return_error_m: float = 0.75,
) -> dict[str, Any]:
    """Score completed rollouts against the sealed geometry key."""

    if rollout_report.get("schema") != ROLLOUT_SCHEMA:
        raise ValueError("Score requires a v15 rollout report")
    if targets.get("schema") != TARGET_SCHEMA:
        raise ValueError("Score requires a v15 target key")
    key = {str(row["goal_id"]): row for row in targets["targets"]}
    rows: list[dict[str, Any]] = []
    for rollout in rollout_report["rollouts"]:
        goal_id = str(rollout["goal_id"])
        target = key.get(goal_id)
        if target is None:
            raise ValueError(f"No scoring target for completed rollout {goal_id}")
        final_xy = rollout["final_pose_xy_yaw"][:2]
        start_xy = rollout["start_pose_xy_yaw"][:2]
        # A goal the model never terminated is a failure regardless of pose.
        stopped = bool(rollout["model_stop_emitted"]) and bool(
            rollout["controller_success"]
        )
        common = {
            "goal_id": goal_id,
            "scene_id": rollout["scene_id"],
            "instruction": rollout["instruction"],
            "metric": target["metric"],
            "termination": rollout["termination"],
            "decision_count": rollout["decision_count"],
            "accepted_decision_count": rollout["accepted_decision_count"],
            "rejected_decision_count": rollout["rejected_decision_count"],
            "path_length_m": rollout["path_length_m"],
            "scene_prefix_sha256": rollout["scene_prefix_sha256"],
        }
        if target["metric"] in {"face_yaw", "approach_standoff"}:
            center_xy = target["target_center_xyz_m"][:2]
            initial_distance = _planar_distance(start_xy, center_xy)
            final_distance = _planar_distance(final_xy, center_xy)
            common.update(
                {
                    "target_category": target["target_category"],
                    "target_instance_id": target["target_instance_id"],
                    "initial_target_center_distance_m": initial_distance,
                    "final_target_center_distance_m": final_distance,
                    "target_center_progress_m": initial_distance - final_distance,
                }
            )
        if target["metric"] == "face_yaw":
            expected = _target_heading_degrees(final_xy, center_xy)
            error = _shortest_angle_error_degrees(
                rollout["final_pose_xy_yaw"][2], expected
            )
            checks = {
                "model_selected_terminal_stop": stopped,
                "maximum_oracle_yaw_error": error <= maximum_face_yaw_error_degrees,
            }
            row = {
                **common,
                "final_body_yaw_degrees": rollout["final_pose_xy_yaw"][2],
                "oracle_target_heading_degrees": expected,
                "oracle_yaw_error_degrees": error,
                "checks": checks,
            }
        elif target["metric"] == "approach_standoff":
            standoff = _point_to_xy_box_distance(
                final_xy,
                target["target_bbox_min_xyz_m"],
                target["target_bbox_max_xyz_m"],
            )
            checks = {
                "model_selected_terminal_stop": stopped,
                "minimum_oracle_center_progress": (
                    common["target_center_progress_m"] >= minimum_center_progress_m
                ),
                "maximum_oracle_bbox_standoff": standoff <= maximum_bbox_standoff_m,
            }
            row = {
                **common,
                "final_oracle_bbox_standoff_m": standoff,
                "checks": checks,
            }
        elif target["metric"] == "lap_circuit":
            checks = {
                "model_selected_terminal_stop": stopped,
                "minimum_path_length": (
                    rollout["path_length_m"] >= minimum_lap_path_length_m
                ),
                "minimum_absolute_swept_area": (
                    rollout["absolute_swept_area_m2"] >= minimum_lap_absolute_area_m2
                ),
                "maximum_return_error": (
                    rollout["return_distance_m"] <= maximum_lap_return_error_m
                ),
            }
            row = {
                **common,
                "absolute_swept_area_m2": rollout["absolute_swept_area_m2"],
                "return_distance_m": rollout["return_distance_m"],
                "checks": checks,
            }
        else:
            raise ValueError(f"Unsupported held-out metric: {target['metric']}")
        row["passed"] = all(row["checks"].values())
        rows.append(row)

    by_metric: dict[str, dict[str, int]] = {}
    for row in rows:
        bucket = by_metric.setdefault(row["metric"], {"passed": 0, "total": 0})
        bucket["total"] += 1
        bucket["passed"] += int(row["passed"])
    stop_rate = sum(
        1 for row in rows if row["checks"]["model_selected_terminal_stop"]
    ) / max(1, len(rows))
    return {
        "schema": SCORE_SCHEMA,
        "status": "evaluation_only_oracle_score",
        "rollout_process_read_oracle": False,
        "scorer_reads_oracle": True,
        "scene_ids": list(targets["scene_ids"]),
        "oracle_file_sha256": targets["oracle_file_sha256"],
        "navigation_checkpoint": rollout_report["navigation_checkpoint"],
        "goal_count": len(rows),
        "passed_count": sum(int(row["passed"]) for row in rows),
        "pass_rate": sum(int(row["passed"]) for row in rows) / max(1, len(rows)),
        "model_selected_terminal_stop_rate": stop_rate,
        "per_metric": {
            name: {
                **value,
                "pass_rate": value["passed"] / max(1, value["total"]),
            }
            for name, value in sorted(by_metric.items())
        },
        "thresholds": {
            "maximum_face_yaw_error_degrees": maximum_face_yaw_error_degrees,
            "minimum_center_progress_m": minimum_center_progress_m,
            "maximum_bbox_standoff_m": maximum_bbox_standoff_m,
            "minimum_lap_path_length_m": minimum_lap_path_length_m,
            "minimum_lap_absolute_area_m2": minimum_lap_absolute_area_m2,
            "maximum_lap_return_error_m": maximum_lap_return_error_m,
        },
        "goals": rows,
    }


__all__ = [
    "PLAN_SCHEMA",
    "ROLLOUT_SCHEMA",
    "SCORE_SCHEMA",
    "TARGET_SCHEMA",
    "build_headless_waypoint_controller",
    "plan_heldout_tasks",
    "run_heldout_rollouts",
    "score_heldout_rollouts",
]
