"""Evaluation-only oracle heading score for a completed MCP face episode.

Runtime evidence and both process access audits are fully authenticated before
this scorer opens the physically separate task specification or scene oracle.
No oracle value is returned to the robot runtime, semantic grounder, controller,
MCP server, or persistent map.
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
from semantic_3d_chat.evaluation.mcp_transport_smoke import _semantic_leaks

SCHEMA: Final[str] = "semantic_3d_chat.conversational_mcp_face_oracle_score.v1"
RUNTIME_SCHEMA: Final[str] = "semantic_3d_chat.conversational_mcp_face.v1"
POLICY_NAME: Final[str] = "selective_gemma_all_voxel_v3_numeric_alignment"


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


def _wrapped_degrees(value: float) -> float:
    return math.degrees(math.atan2(math.sin(math.radians(value)), math.cos(math.radians(value))))


def _validate_audit(reference: Mapping[str, Any], *, purpose: str) -> dict[str, Any]:
    path_value = reference.get("path")
    expected_hash = reference.get("sha256")
    if not isinstance(path_value, str) or not isinstance(expected_hash, str):
        raise TypeError(f"{purpose} audit reference is invalid")
    path = _rooted(path_value)
    if _sha256(path) != expected_hash:
        raise ValueError(f"{purpose} audit hash differs from runtime evidence")
    audit = _read_object(path, purpose=f"{purpose} access audit")
    loaded = audit.get("loaded_files")
    if (
        audit.get("passed") is not True
        or audit.get("block_forbidden") is not True
        or audit.get("forbidden_accesses") != []
        or not isinstance(loaded, list)
        or len(loaded) != reference.get("loaded_file_count")
        or reference.get("forbidden_access_count") != 0
        or reference.get("passed") is not True
    ):
        raise ValueError(f"{purpose} access audit does not attest a clean runtime")
    return {
        "path": _relative(path),
        "sha256": expected_hash,
        "loaded_file_count": len(loaded),
        "forbidden_access_count": 0,
        "passed": True,
    }


def _validate_runtime_before_oracle(
    runtime_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    result = _read_object(runtime_path, purpose="MCP face runtime result")
    policy = result.get("policy")
    transport = result.get("transport")
    steps = result.get("steps")
    initial = result.get("initial_observation")
    if (
        result.get("schema") != RUNTIME_SCHEMA
        or result.get("passed") is not True
        or result.get("scene_id") is None
        or result.get("termination_reason") != "fresh_grounding_inside_deadband"
        or result.get("final_stopped") is not True
        or result.get("all_decisions_used_fresh_all_voxel_grounding") is not True
        or result.get("environmental_text_inputs") != []
        or result.get("oracle_inputs_at_runtime") is not False
        or result.get("semantic_leaks_in_numeric_tool_receipts") != []
        or not isinstance(policy, Mapping)
        or policy.get("name") != POLICY_NAME
        or policy.get("official_mcp_sdk_stdio_action_execution") is not True
        or policy.get("learned_v3_action_head_used") is not False
        or policy.get("gemma_native_function_calling_used") is not False
        or not isinstance(transport, Mapping)
        or transport.get("implementation") != "official_python_mcp_sdk_stdio"
        or transport.get("process_boundary") is not True
        or transport.get("numeric_structured_output_only") is not True
        or not isinstance(steps, list)
        or not steps
        or result.get("step_count") != len(steps)
        or not isinstance(initial, Mapping)
    ):
        raise ValueError("MCP face runtime result lacks its clean integration attestation")

    initial_receipt = initial.get("numeric_tool_receipt")
    initial_transition = initial.get("continuous_binding_transition")
    receipts: list[Mapping[str, Any]] = []
    if not isinstance(initial_receipt, Mapping) or not isinstance(initial_transition, Mapping):
        raise TypeError("MCP face runtime lacks its initial numeric observation")
    if (
        initial_receipt.get("success") is not True
        or initial_transition.get("passed") is not True
        or initial_transition.get("observation_required") is not True
    ):
        raise ValueError("MCP face initial observation did not refresh the continuous map")
    receipts.append(initial_receipt)

    for index, step in enumerate(steps, start=1):
        if not isinstance(step, Mapping) or step.get("step") != index:
            raise ValueError("MCP face steps are not a consecutive finite transcript")
        grounding = step.get("grounding")
        call = step.get("mcp_call")
        receipt = step.get("numeric_tool_receipt")
        transition = step.get("continuous_binding_transition")
        if (
            not isinstance(grounding, Mapping)
            or grounding.get("all_map_voxels_scored") is not True
            or not isinstance(grounding.get("scored_voxels"), int)
            or grounding["scored_voxels"] < 1
            or not isinstance(call, Mapping)
            or call.get("tool") not in {"turn", "stop"}
            or not isinstance(receipt, Mapping)
            or receipt.get("success") is not True
            or not isinstance(transition, Mapping)
            or transition.get("passed") is not True
        ):
            raise ValueError("MCP face transcript has an invalid grounded action step")
        receipts.append(receipt)
    if steps[-1]["mcp_call"].get("tool") != "stop":
        raise ValueError("MCP face transcript does not terminate with stop")
    if _semantic_leaks(list(receipts)):
        raise ValueError("MCP face numeric tool receipts contain semantic text")

    client = result.get("client_access_audit")
    server = result.get("server_access_audit")
    if not isinstance(client, Mapping) or not isinstance(server, Mapping):
        raise TypeError("MCP face runtime lacks both process access audits")
    client_audit = _validate_audit(client, purpose="client")
    server_audit = _validate_audit(server, purpose="server")
    return result, client_audit, server_audit


def build_score(
    runtime_result: str | Path,
    scene_oracle: str | Path,
    scoring_spec: str | Path,
    *,
    task_id: str = "nav_000",
) -> dict[str, Any]:
    """Authenticate completed runtime evidence, then score isolated oracle geometry."""

    runtime_path = _rooted(runtime_result)
    result, client_audit, server_audit = _validate_runtime_before_oracle(runtime_path)

    # Oracle/evaluation inputs are intentionally opened only after every runtime
    # artifact and both process-lifetime access audits have passed validation.
    spec_path = _rooted(scoring_spec)
    oracle_path = _rooted(scene_oracle)
    spec = _read_object(spec_path, purpose="oracle scoring specification")
    oracle = _read_object(oracle_path, purpose="scene oracle")
    scene_id = result["scene_id"]
    if spec.get("scene_id") != scene_id or oracle.get("scene_id") != scene_id:
        raise ValueError("Runtime, scoring specification, and oracle scenes differ")
    tasks = spec.get("tasks")
    instances = oracle.get("instances")
    if not isinstance(tasks, list) or not isinstance(instances, list):
        raise TypeError("Oracle scoring inputs lack task or instance rows")
    task_rows = [row for row in tasks if isinstance(row, Mapping) and row.get("task_id") == task_id]
    if len(task_rows) != 1 or task_rows[0].get("family") != "face":
        raise ValueError("Oracle scoring spec lacks one requested face task")
    task = task_rows[0]
    target_id = task.get("target_instance_id")
    target_rows = [
        row for row in instances if isinstance(row, Mapping) and row.get("instance_id") == target_id
    ]
    if len(target_rows) != 1 or not isinstance(target_rows[0].get("pose"), Mapping):
        raise ValueError("Scene oracle lacks one numeric target instance")
    target = _finite_vector(
        target_rows[0]["pose"].get("center_xyz_m"),
        3,
        name="oracle target center",
    )

    final_step = result["steps"][-1]
    final_receipt = final_step["numeric_tool_receipt"]
    robot = _finite_vector(final_receipt.get("position_m"), 3, name="final robot pose")
    yaw_value = final_receipt.get("body_yaw_degrees")
    if (
        isinstance(yaw_value, bool)
        or not isinstance(yaw_value, (int, float))
        or not math.isfinite(float(yaw_value))
    ):
        raise ValueError("Final robot yaw is invalid")
    final_yaw = float(yaw_value)
    desired_yaw = math.degrees(math.atan2(-(target[0] - robot[0]), target[1] - robot[1]))
    signed_error = _wrapped_degrees(desired_yaw - final_yaw)
    maximum_error = float(task["maximum_heading_error_degrees"])
    maximum_collisions = int(task["maximum_collisions"])
    receipts = [result["initial_observation"]["numeric_tool_receipt"]] + [
        step["numeric_tool_receipt"] for step in result["steps"]
    ]
    collision_count = sum(receipt.get("collision") is True for receipt in receipts)
    passed = bool(
        result["passed"] is True
        and final_receipt.get("stopped") is True
        and abs(signed_error) <= maximum_error
        and collision_count <= maximum_collisions
    )
    return {
        "schema": SCHEMA,
        "status": "passed" if passed else "failed",
        "passed": passed,
        "scene_id": scene_id,
        "task_id": task_id,
        "family": "face",
        "runtime_result": {
            "path": _relative(runtime_path),
            "sha256": _sha256(runtime_path),
            "policy": POLICY_NAME,
            "official_mcp_sdk_stdio_action_execution": True,
            "learned_v3_action_head_used": False,
            "gemma_native_function_calling_used": False,
            "client_access_audit": client_audit,
            "server_access_audit": server_audit,
        },
        "final_pose": {
            "position_xyz_m": robot,
            "body_yaw_degrees": final_yaw,
            "stopped": final_receipt.get("stopped") is True,
        },
        "oracle_target": {
            "opaque_instance_id": target_id,
            "center_xyz_m": target,
            "desired_yaw_degrees": desired_yaw,
        },
        "heading": {
            "signed_error_degrees": signed_error,
            "absolute_error_degrees": abs(signed_error),
            "maximum_error_degrees": maximum_error,
        },
        "collision": {
            "count": collision_count,
            "maximum_count": maximum_collisions,
            "passed": collision_count <= maximum_collisions,
        },
        "oracle_only_scorer_attestation": {
            "evaluation_only": True,
            "runtime_validated_before_oracle_open": True,
            "runtime_process_read_oracle": False,
            "oracle_geometry_loaded_by_scorer_only": True,
            "score_fed_back_to_runtime": False,
            "runtime_result_modified": False,
            "scene_oracle_sha256": _sha256(oracle_path),
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
    parser.add_argument("--scene-oracle", required=True)
    parser.add_argument("--scoring-spec", required=True)
    parser.add_argument("--task-id", default="nav_000")
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    payload = build_score(
        args.runtime_result,
        args.scene_oracle,
        args.scoring_spec,
        task_id=args.task_id,
    )
    destination = _atomic_json(args.output, payload)
    print(
        json.dumps(
            {
                "schema": payload["schema"],
                "passed": payload["passed"],
                "scene_id": payload["scene_id"],
                "absolute_heading_error_degrees": payload["heading"]["absolute_error_degrees"],
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


__all__ = ["POLICY_NAME", "RUNTIME_SCHEMA", "SCHEMA", "build_score"]
