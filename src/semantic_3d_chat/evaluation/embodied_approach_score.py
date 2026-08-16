"""Post-inference oracle scorer for continuous-semantic approach episodes.

The conversational runtime is validated completely before this module opens an
oracle file.  This evaluator is therefore deliberately outside the chat/robot
process and its output is always labelled evaluation-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _xy(value: object, name: str) -> tuple[float, float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) < 2:
        raise TypeError(f"{name} must contain at least two numbers")
    result = (float(value[0]), float(value[1]))
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{name} contains a non-finite value")
    return result


def _finite(value: object, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} is not finite")
    return result


def _runtime_attestation(path: Path, scene_id: str) -> dict[str, Any]:
    """Validate inference evidence without opening any oracle resource."""

    value = _read_object(path)
    turns = value.get("turns")
    if not isinstance(turns, list) or len(turns) != 1 or not isinstance(turns[0], Mapping):
        raise ValueError(f"approach result must contain exactly one turn: {path}")
    turn = turns[0]
    receipts = turn.get("action_receipts")
    groundings = turn.get("continuous_grounding_attestations")
    if not isinstance(receipts, list) or not receipts or not all(isinstance(row, Mapping) for row in receipts):
        raise ValueError(f"approach result has no valid action receipts: {path}")
    if not isinstance(groundings, list) or not groundings or not all(isinstance(row, Mapping) for row in groundings):
        raise ValueError(f"approach result has no valid grounding attestations: {path}")
    final_grounding = groundings[-1]
    approach = final_grounding.get("numeric_approach_interlock")
    if not isinstance(approach, Mapping):
        raise TypeError(f"final grounding has no numeric approach attestation: {path}")
    final_receipt = receipts[-1]
    collision_limited = approach.get("collision_limited_interlock")
    semantic_completion = (
        approach.get("goal_satisfied") is True
        and approach.get("stop_applied") is True
        and approach.get("reason") == "fresh_grounding_approach_goal_satisfied"
    )
    safe_collision_limited_completion = (
        approach.get("completion_satisfied") is True
        and approach.get("completion_mode") == "collision_limited_safe_stop"
        and approach.get("stop_applied") is True
        and approach.get("reason") == "collision_limited_safe_stop"
        and isinstance(collision_limited, Mapping)
        and collision_limited.get("safe_closest_reachable") is True
        and collision_limited.get("numeric_collision_map_only") is True
    )
    stopped = final_receipt.get("stopped") is True
    collision = any(row.get("collision") is True for row in receipts)
    all_receipts_successful = all(row.get("success") is True for row in receipts)
    runtime_completed = (
        turn.get("success") is True
        and turn.get("termination_reason") == "stop"
        and stopped
        and not collision
        and all_receipts_successful
        and (semantic_completion or safe_collision_limited_completion)
    )
    valid_failure = (
        turn.get("success") is False
        and turn.get("termination_reason") in {"action_failure", "max_steps"}
        and not stopped
        and (
            turn.get("termination_reason") == "max_steps"
            or not all_receipts_successful
        )
    )
    runtime_ok = (
        value.get("schema") == "semantic_3d_chat.embodied_conversation_result.v1"
        and value.get("scene_id") == scene_id
        and value.get("environmental_text_inputs") == []
        and value.get("passed_runtime_audit") is True
        and value.get("forbidden_access_count") == 0
        and turn.get("prefix_refresh_verified") is True
        and turn.get("primary_static_scene_retrieval") is False
        and turn.get("static_scene_prefix_question_independent") is True
        and turn.get("environmental_text_inputs") == []
        and all(row.get("all_map_voxels_scored") is True for row in groundings)
        and all(row.get("environmental_text_inputs") == [] for row in groundings)
        and all(row.get("oracle_inputs_at_runtime") is False for row in groundings)
        and approach.get("terminal_approach_requested") is True
        and approach.get("environmental_text_inputs") == []
        and approach.get("oracle_inputs_at_runtime") is False
        and (runtime_completed or valid_failure)
    )
    if not runtime_ok:
        raise ValueError(f"runtime approach contract did not pass: {path}")
    semantic_distance = _finite(approach.get("target_distance_m"), "semantic target distance")
    semantic_standoff = _finite(approach.get("target_standoff_m"), "semantic target standoff")
    if semantic_completion and semantic_distance > semantic_standoff + 1e-6:
        raise ValueError(f"runtime stopped outside its continuous target standoff: {path}")
    return {
        "scene_id": scene_id,
        "result_path": path.as_posix(),
        "result_sha256": _sha256(path),
        "step_count": int(turn["step_count"]),
        "initial_xy_m": _xy(approach.get("initial_robot_position_xy_m"), "initial robot position"),
        "final_xy_m": _xy(final_receipt.get("position_m"), "final robot position"),
        "semantic_target_xyz_m": list(approach["target_xyz_m"]),
        "semantic_target_distance_m": semantic_distance,
        "semantic_target_standoff_m": semantic_standoff,
        "numeric_translation_m": _finite(approach.get("actual_progress_m"), "numeric translation"),
        "map_version": int(final_receipt["map_version"]),
        "processed_voxels": int(final_receipt["processed_voxels"]),
        "runtime_forbidden_access_count": 0,
        "runtime_environmental_text_inputs": [],
        "runtime_completed": runtime_completed,
        "semantic_standoff_completed": semantic_completion,
        "safe_collision_limited_completion": safe_collision_limited_completion,
        "completion_mode": approach.get("completion_mode"),
        "termination_reason": turn.get("termination_reason"),
        "stopped": stopped,
        "collision": collision,
        "all_action_receipts_successful": all_receipts_successful,
    }


def _planar_distance(left: tuple[float, float], right: tuple[float, float]) -> float:
    return math.hypot(left[0] - right[0], left[1] - right[1])


def _point_to_xy_box_distance(point: tuple[float, float], minimum: tuple[float, float], maximum: tuple[float, float]) -> float:
    dx = max(minimum[0] - point[0], 0.0, point[0] - maximum[0])
    dy = max(minimum[1] - point[1], 0.0, point[1] - maximum[1])
    return math.hypot(dx, dy)


def score_approach_results(
    cases: Sequence[tuple[str, str | Path]],
    *,
    oracle_root: str | Path,
    target_category: str,
    minimum_center_progress_m: float = 0.25,
    maximum_bbox_standoff_m: float = 0.60,
) -> dict[str, Any]:
    """Score validated runtime results in a physically separate oracle stage."""

    if not cases:
        raise ValueError("at least one approach case is required")
    progress_threshold = _finite(minimum_center_progress_m, "minimum center progress")
    standoff_threshold = _finite(maximum_bbox_standoff_m, "maximum bbox standoff")
    if progress_threshold < 0.0 or standoff_threshold < 0.0:
        raise ValueError("approach thresholds must be non-negative")

    # This ordering is intentional and tested: reject malformed inference
    # evidence before making any oracle path available to the scorer.
    runtime_rows = [
        _runtime_attestation(Path(os.path.abspath(Path(result).expanduser())), scene_id)
        for scene_id, result in cases
    ]

    root = Path(os.path.abspath(Path(oracle_root).expanduser()))
    scored: list[dict[str, Any]] = []
    for runtime in runtime_rows:
        oracle_path = root / runtime["scene_id"] / "oracle.json"
        oracle = _read_object(oracle_path)
        instances = oracle.get("instances")
        if oracle.get("scene_id") != runtime["scene_id"] or not isinstance(instances, list):
            raise ValueError(f"oracle scene contract differs: {oracle_path}")
        matches = [row for row in instances if isinstance(row, Mapping) and row.get("category") == target_category]
        if len(matches) != 1:
            raise ValueError(f"expected one oracle {target_category!r} in {runtime['scene_id']}")
        target = matches[0]
        center = _xy(target.get("expected_center_xyz_m"), "oracle target center")
        bbox = target.get("bbox")
        if not isinstance(bbox, Mapping):
            raise TypeError(f"oracle target has no bounding box: {oracle_path}")
        minimum = _xy(bbox.get("min_xyz_m"), "oracle bounding-box minimum")
        maximum = _xy(bbox.get("max_xyz_m"), "oracle bounding-box maximum")
        initial_center_distance = _planar_distance(runtime["initial_xy_m"], center)
        final_center_distance = _planar_distance(runtime["final_xy_m"], center)
        center_progress = initial_center_distance - final_center_distance
        bbox_standoff = _point_to_xy_box_distance(runtime["final_xy_m"], minimum, maximum)
        checks = {
            "runtime_contract": True,
            "runtime_completed": runtime["runtime_completed"],
            "minimum_oracle_center_progress": center_progress >= progress_threshold,
            "maximum_oracle_bbox_standoff": bbox_standoff <= standoff_threshold,
            "continuous_completion": runtime["semantic_standoff_completed"]
            or runtime["safe_collision_limited_completion"],
            "stopped": runtime["stopped"],
            "collision_free": not runtime["collision"],
        }
        scored.append(
            {
                **runtime,
                "oracle_file": oracle_path.as_posix(),
                "oracle_file_sha256": _sha256(oracle_path),
                "oracle_instance_id": target.get("instance_id"),
                "oracle_target_category": target_category,
                "oracle_target_center_xyz_m": list(target["expected_center_xyz_m"]),
                "initial_oracle_center_distance_m": initial_center_distance,
                "final_oracle_center_distance_m": final_center_distance,
                "oracle_center_progress_m": center_progress,
                "final_oracle_bbox_standoff_m": bbox_standoff,
                "checks": checks,
                "passed": all(checks.values()),
            }
        )
    passed = sum(row["passed"] for row in scored)
    return {
        "schema": "semantic_3d_chat.embodied_approach_oracle_score.v1",
        "status": "evaluation_only_oracle_score",
        "scorer_reads_oracle": True,
        "runtime_process_read_oracle": False,
        "runtime_evidence_validated_before_oracle_open": True,
        "runtime_environmental_text_inputs": [],
        "target_category_used_only_by_scorer": target_category,
        "minimum_center_progress_m": progress_threshold,
        "maximum_bbox_standoff_m": standoff_threshold,
        "scene_count": len(scored),
        "passed_count": passed,
        "all_passed": passed == len(scored),
        "scenes": scored,
    }


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", required=True, metavar="SCENE=RESULT")
    parser.add_argument("--oracle-root", default="data/oracle")
    parser.add_argument("--target-category", required=True)
    parser.add_argument("--minimum-center-progress-m", type=float, default=0.25)
    parser.add_argument("--maximum-bbox-standoff-m", type=float, default=0.60)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    cases: list[tuple[str, str]] = []
    for value in args.case:
        scene_id, separator, result = value.partition("=")
        if not separator or not scene_id or not result:
            parser.error("--case must have the form SCENE=RESULT")
        cases.append((scene_id, result))
    report = score_approach_results(
        cases,
        oracle_root=args.oracle_root,
        target_category=args.target_category,
        minimum_center_progress_m=args.minimum_center_progress_m,
        maximum_bbox_standoff_m=args.maximum_bbox_standoff_m,
    )
    _atomic_json(Path(args.output), report)
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
