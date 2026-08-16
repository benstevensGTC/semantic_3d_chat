"""Physically separate oracle score for an authenticated MCP session smoke.

The runtime result, model-free inspection, and both process-lifetime access
audits are authenticated before this process opens the scoring specification or
scene oracle.  Oracle values are never returned to the robot runtime, grounder,
MCP server, semantic map, or continuous prefix.
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
from typing import Any, Final

from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.evaluation.conversational_mcp_session_inspect import (
    EXPECTED_COMMAND_ORDER,
    build_inspection,
)
from semantic_3d_chat.evaluation.conversational_mcp_session_inspect import (
    SCHEMA as INSPECTION_SCHEMA,
)

SCHEMA: Final[str] = "semantic_3d_chat.conversational_mcp_session_oracle_score.v1"
SPEC_SCHEMA: Final[str] = "semantic_3d_chat.conversational_mcp_session_scoring_spec.v1"
RUNTIME_SCHEMA: Final[str] = "semantic_3d_chat.conversational_mcp_session.v1"


def _rooted(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    return Path(os.path.abspath(candidate if candidate.is_absolute() else PROJECT_ROOT / candidate))


def _read_object(path: Path, *, purpose: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{purpose} must be a regular file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{purpose} must contain one JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _finite_vector(value: object, width: int, *, name: str) -> list[float]:
    if (
        not isinstance(value, list)
        or len(value) != width
        or any(
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
            for item in value
        )
    ):
        raise ValueError(f"{name} must be a finite {width}-vector")
    return [float(item) for item in value]


def _finite_nonnegative(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def _wrapped_degrees(value: float) -> float:
    return math.degrees(math.atan2(math.sin(math.radians(value)), math.cos(math.radians(value))))


def _distance_xy(first: Sequence[float], second: Sequence[float]) -> float:
    return math.hypot(float(first[0]) - float(second[0]), float(first[1]) - float(second[1]))


def _desired_yaw(robot: Sequence[float], target: Sequence[float]) -> float:
    delta_x = float(target[0]) - float(robot[0])
    delta_y = float(target[1]) - float(robot[1])
    if math.hypot(delta_x, delta_y) <= 1e-9:
        raise ValueError("Oracle target is coincident with the robot XY")
    return math.degrees(math.atan2(-delta_x, delta_y))


def _all_runtime_receipts(runtime: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    initial = runtime["initial_observation"]["numeric_tool_receipt"]
    receipts: list[Mapping[str, Any]] = [initial]
    for turn in runtime["turns"]:
        steps = turn.get("steps")
        if isinstance(steps, list):
            receipts.extend(step["numeric_tool_receipt"] for step in steps)
        elif isinstance(turn.get("numeric_tool_receipt"), Mapping):
            receipts.append(turn["numeric_tool_receipt"])
    receipts.append(runtime["shutdown"]["numeric_tool_receipt"])
    return receipts


def _authenticate_before_oracle(
    runtime_result: str | Path,
    inspection_result: str | Path,
) -> tuple[Path, Path, dict[str, Any], dict[str, Any]]:
    """Recompute the inspector and bind the supplied artifact before oracle I/O."""

    runtime_path = _rooted(runtime_result)
    inspection_path = _rooted(inspection_result)
    # build_inspection is the model-free gate. It authenticates the runtime
    # transcript and both access-audit hashes, and refuses oracle/QA paths.
    recomputed = build_inspection(runtime_path)
    saved = _read_object(inspection_path, purpose="model-free session inspection")
    _require(saved == recomputed, "Saved session inspection differs from a fresh authentication")
    _require(
        saved.get("schema") == INSPECTION_SCHEMA
        and saved.get("passed") is True
        and saved.get("oracle_inputs_opened") is False
        and saved.get("runtime_result", {}).get("sha256") == _sha256(runtime_path),
        "Session inspection is not a clean runtime binding",
    )
    runtime = _read_object(runtime_path, purpose="conversational MCP runtime result")
    _require(
        runtime.get("schema") == RUNTIME_SCHEMA
        and runtime.get("passed") is True
        and runtime.get("final_stopped") is True,
        "Runtime result is not a completed passing session",
    )
    return runtime_path, inspection_path, runtime, saved


def build_score(
    runtime_result: str | Path,
    inspection_result: str | Path,
    scene_oracle: str | Path,
    scoring_spec: str | Path,
) -> dict[str, Any]:
    """Authenticate runtime evidence, then measure isolated oracle geometry."""

    runtime_path, inspection_path, runtime, inspection = _authenticate_before_oracle(
        runtime_result,
        inspection_result,
    )

    # These two files are opened only after the complete runtime and inspector
    # gate above succeeds. Nothing below calls or modifies the runtime.
    spec_path = _rooted(scoring_spec)
    oracle_path = _rooted(scene_oracle)
    spec = _read_object(spec_path, purpose="oracle-only session scoring specification")
    oracle = _read_object(oracle_path, purpose="scene oracle")
    scene_id = runtime["scene_id"]
    _require(
        spec.get("schema") == SPEC_SCHEMA
        and spec.get("scene_id") == scene_id
        and oracle.get("scene_id") == scene_id,
        "Runtime, scoring specification, and oracle identities differ",
    )
    _require(
        tuple(spec.get("expected_command_order", [])) == EXPECTED_COMMAND_ORDER
        and inspection["transcript"]["command_order"] == list(EXPECTED_COMMAND_ORDER),
        "Oracle scoring spec command order differs from authenticated runtime",
    )
    target_id = spec.get("target_instance_id")
    instances = oracle.get("instances")
    _require(isinstance(target_id, str), "Scoring spec target instance is invalid")
    _require(isinstance(instances, list), "Scene oracle instances are invalid")
    target_rows = [
        row for row in instances if isinstance(row, Mapping) and row.get("instance_id") == target_id
    ]
    _require(len(target_rows) == 1, "Scene oracle lacks exactly one requested target instance")
    target_pose = target_rows[0].get("pose")
    _require(isinstance(target_pose, Mapping), "Oracle target pose is invalid")
    assert isinstance(target_pose, Mapping)
    target = _finite_vector(target_pose.get("center_xyz_m"), 3, name="oracle target center")

    initial_receipt = runtime["initial_observation"]["numeric_tool_receipt"]
    face_turn = runtime["turns"][0]
    face_receipt = face_turn["steps"][-1]["numeric_tool_receipt"]
    final_position_xy = _finite_vector(runtime.get("final_position_xy_m"), 2, name="final XY")
    initial_position = _finite_vector(initial_receipt.get("position_m"), 3, name="initial pose")
    face_position = _finite_vector(face_receipt.get("position_m"), 3, name="face pose")
    final_yaw_value = runtime.get("final_body_yaw_degrees")
    face_yaw_value = face_receipt.get("body_yaw_degrees")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in (face_yaw_value, final_yaw_value)
    ):
        raise ValueError("Runtime face/final yaw is invalid")
    face_yaw = float(face_yaw_value)
    final_yaw = float(final_yaw_value)
    initial_distance = _distance_xy(initial_position, target)
    face_distance = _distance_xy(face_position, target)
    final_distance = _distance_xy(final_position_xy, target)
    target_progress = initial_distance - final_distance
    displacement = _distance_xy(initial_position, final_position_xy)
    face_desired_yaw = _desired_yaw(face_position, target)
    final_desired_yaw = _desired_yaw(final_position_xy, target)
    face_heading_error = _wrapped_degrees(face_desired_yaw - face_yaw)
    final_heading_error = _wrapped_degrees(final_desired_yaw - final_yaw)
    receipts = _all_runtime_receipts(runtime)
    collision_count = sum(receipt.get("collision") is True for receipt in receipts)

    maximum_face_heading = _finite_nonnegative(
        spec.get("maximum_face_heading_error_degrees"),
        name="maximum face heading error",
    )
    maximum_final_heading = _finite_nonnegative(
        spec.get("maximum_final_heading_error_degrees"),
        name="maximum final heading error",
    )
    maximum_standoff = _finite_nonnegative(
        spec.get("maximum_final_target_standoff_m"),
        name="maximum final target standoff",
    )
    minimum_progress = _finite_nonnegative(
        spec.get("minimum_target_progress_m"),
        name="minimum target progress",
    )
    maximum_collisions_value = spec.get("maximum_collisions")
    if (
        isinstance(maximum_collisions_value, bool)
        or not isinstance(maximum_collisions_value, int)
        or maximum_collisions_value < 0
    ):
        raise ValueError("Maximum collisions must be a nonnegative integer")
    gates = {
        "runtime_and_inspection_authenticated_before_oracle_open": True,
        "face_heading_within_bound": abs(face_heading_error) <= maximum_face_heading,
        "final_heading_within_bound": abs(final_heading_error) <= maximum_final_heading,
        "final_target_standoff_within_bound": final_distance <= maximum_standoff,
        "target_progress_above_minimum": target_progress >= minimum_progress,
        "collision_count_within_bound": collision_count <= maximum_collisions_value,
        "final_stop_latched": (
            runtime.get("final_stopped") is True if spec.get("require_final_stop") is True else True
        ),
    }
    passed = all(gates.values())
    client = inspection["client_access_audit"]
    server = inspection["server_access_audit"]
    return {
        "schema": SCHEMA,
        "status": "passed" if passed else "failed",
        "passed": passed,
        "scene_id": scene_id,
        "runtime_evidence": {
            "runtime_result_path": _relative(runtime_path),
            "runtime_result_sha256": _sha256(runtime_path),
            "inspection_path": _relative(inspection_path),
            "inspection_sha256": _sha256(inspection_path),
            "inspection_runtime_sha256": inspection["runtime_result"]["sha256"],
            "client_access_audit_sha256": client["sha256"],
            "server_access_audit_sha256": server["sha256"],
            "client_forbidden_access_count": client["forbidden_access_count"],
            "server_forbidden_access_count": server["forbidden_access_count"],
            "official_single_persistent_stdio_session": True,
            "learned_v3_action_head_used": False,
            "gemma_native_function_calling_used": False,
        },
        "oracle_target": {
            "opaque_instance_id": target_id,
            "center_xyz_m": target,
        },
        "distance": {
            "initial_xy_m": initial_position[:2],
            "face_xy_m": face_position[:2],
            "final_xy_m": final_position_xy,
            "initial_target_distance_m": initial_distance,
            "face_target_distance_m": face_distance,
            "final_target_distance_m": final_distance,
            "target_progress_m": target_progress,
            "minimum_target_progress_m": minimum_progress,
            "robot_displacement_m": displacement,
            "maximum_final_target_standoff_m": maximum_standoff,
        },
        "heading": {
            "face_body_yaw_degrees": face_yaw,
            "face_desired_yaw_degrees": face_desired_yaw,
            "face_signed_error_degrees": face_heading_error,
            "face_absolute_error_degrees": abs(face_heading_error),
            "maximum_face_error_degrees": maximum_face_heading,
            "final_body_yaw_degrees": final_yaw,
            "final_desired_yaw_degrees": final_desired_yaw,
            "final_signed_error_degrees": final_heading_error,
            "final_absolute_error_degrees": abs(final_heading_error),
            "maximum_final_error_degrees": maximum_final_heading,
        },
        "collision": {
            "receipt_count": len(receipts),
            "collision_count": collision_count,
            "maximum_collisions": maximum_collisions_value,
        },
        "gates": gates,
        "oracle_only_scorer_attestation": {
            "evaluation_only": True,
            "runtime_and_inspection_validated_before_oracle_open": True,
            "runtime_process_read_oracle": False,
            "oracle_geometry_loaded_by_scorer_only": True,
            "score_fed_back_to_runtime": False,
            "runtime_result_modified": False,
            "inspection_result_modified": False,
            "scene_oracle_path": _relative(oracle_path),
            "scene_oracle_sha256": _sha256(oracle_path),
            "scoring_spec_path": _relative(spec_path),
            "scoring_spec_sha256": _sha256(spec_path),
        },
    }


def _atomic_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    destination = _rooted(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    finally:
        Path(temporary_name).unlink(missing_ok=True)
    return destination


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-result", required=True)
    parser.add_argument("--inspection-result", required=True)
    parser.add_argument("--scene-oracle", required=True)
    parser.add_argument("--scoring-spec", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    payload = build_score(
        args.runtime_result,
        args.inspection_result,
        args.scene_oracle,
        args.scoring_spec,
    )
    destination = _atomic_json(args.output, payload)
    print(
        json.dumps(
            {
                "schema": payload["schema"],
                "passed": payload["passed"],
                "scene_id": payload["scene_id"],
                "final_target_distance_m": payload["distance"]["final_target_distance_m"],
                "target_progress_m": payload["distance"]["target_progress_m"],
                "face_absolute_heading_error_degrees": payload["heading"][
                    "face_absolute_error_degrees"
                ],
                "collision_count": payload["collision"]["collision_count"],
                "output": str(destination),
            },
            sort_keys=True,
            allow_nan=False,
        ),
        flush=True,
    )
    return 0 if payload["passed"] is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
