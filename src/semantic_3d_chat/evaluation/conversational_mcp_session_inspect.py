"""Model-free authentication of the finite conversational MCP session smoke.

This inspector reads only the saved runtime transcript and its two access-audit
files.  It does not load Gemma, Blender, a semantic map, QA data, training data,
or oracle geometry.  Physical target-distance scoring intentionally belongs to
a later, physically separate oracle-only evaluator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np

from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.evaluation.mcp_transport_smoke import _semantic_leaks
from semantic_3d_chat.robot.mcp_stdio_runtime import validate_numeric_tool_receipt

SCHEMA: Final[str] = "semantic_3d_chat.conversational_mcp_session_inspection.v1"
RUNTIME_SCHEMA: Final[str] = "semantic_3d_chat.conversational_mcp_session.v1"
TURN_SCHEMA: Final[str] = "semantic_3d_chat.conversational_mcp_session_turn.v1"
SHUTDOWN_SCHEMA: Final[str] = "semantic_3d_chat.conversational_mcp_session_shutdown.v1"
POLICY_NAME: Final[str] = "selective_gemma_all_voxel_v3_numeric_navigation"
EXPECTED_COMMAND_ORDER: Final[tuple[str, ...]] = (
    "face",
    "approach",
    "scan",
    "state",
    "stop",
)
_SHA256 = re.compile(r"[0-9a-f]{64}")
_BLOCKED_COMPONENTS: Final[frozenset[str]] = frozenset(
    {"oracle", "qa", "training", "scorer_only", "scorer-only"}
)
_BINDING_FIELDS: Final[tuple[str, ...]] = (
    "schema",
    "scene_id",
    "map_version",
    "map_sha256",
    "scene_prefix_sha256",
    "scene_control_signature_sha256",
    "source_voxels",
    "processed_voxels",
    "binding_sha256",
    "active_prefix_sha256",
    "robot_state_sha256",
    "robot_tokens_sha256",
    "robot_state_encoder_sha256",
    "active_binding_sha256",
)
_SCENE_BINDING_FIELDS: Final[tuple[str, ...]] = (
    "schema",
    "scene_id",
    "map_version",
    "map_sha256",
    "scene_prefix_sha256",
    "scene_control_signature_sha256",
    "source_voxels",
    "processed_voxels",
    "binding_sha256",
)


def _rooted(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    return Path(os.path.abspath(candidate if candidate.is_absolute() else PROJECT_ROOT / candidate))


def _safe_file(path: str | Path, *, purpose: str) -> Path:
    candidate = _rooted(path)
    current = Path(candidate.anchor)
    for component in candidate.parts[1:]:
        current /= component
        if current.is_symlink():
            raise ValueError(f"{purpose} cannot use a symbolic-link path")
    if _BLOCKED_COMPONENTS.intersection(component.casefold() for component in candidate.parts):
        raise ValueError(f"{purpose} cannot enter environmental supervision trees")
    if not candidate.is_file():
        raise FileNotFoundError(f"{purpose} is unavailable: {candidate}")
    return candidate


def _read_object(path: Path, *, purpose: str) -> dict[str, Any]:
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


def _binding(value: Mapping[str, Any]) -> dict[str, Any]:
    missing = [field for field in _BINDING_FIELDS if field not in value]
    if missing:
        raise ValueError(f"Continuous binding omitted fields: {missing}")
    return {field: value[field] for field in _BINDING_FIELDS}


def _validate_audit(reference: Mapping[str, Any], *, purpose: str) -> dict[str, Any]:
    path_value = reference.get("path")
    expected_hash = reference.get("sha256")
    if (
        not isinstance(path_value, str)
        or not isinstance(expected_hash, str)
        or _SHA256.fullmatch(expected_hash) is None
    ):
        raise TypeError(f"{purpose} audit reference is invalid")
    path = _safe_file(path_value, purpose=f"{purpose} access audit")
    _require(_sha256(path) == expected_hash, f"{purpose} audit hash differs")
    audit = _read_object(path, purpose=f"{purpose} access audit")
    loaded = audit.get("loaded_files")
    forbidden_names = audit.get("forbidden_component_names")
    forbidden_roots = audit.get("forbidden_roots")
    _require(audit.get("passed") is True, f"{purpose} audit did not pass")
    _require(audit.get("block_forbidden") is True, f"{purpose} audit was not blocking")
    _require(audit.get("forbidden_accesses") == [], f"{purpose} audit records forbidden reads")
    _require(isinstance(loaded, list), f"{purpose} audit loaded-file list is invalid")
    _require(
        isinstance(forbidden_roots, list)
        and all(isinstance(item, str) for item in forbidden_roots),
        f"{purpose} audit forbidden roots are invalid",
    )
    _require(
        isinstance(forbidden_names, list)
        and all(isinstance(item, str) for item in forbidden_names),
        f"{purpose} audit forbidden-component policy is invalid",
    )
    blocked_name_components = {str(item).casefold() for item in forbidden_names}
    covered_components = set(blocked_name_components)
    for root in forbidden_roots:
        covered_components.update(component.casefold() for component in Path(root).parts)
    _require(
        _BLOCKED_COMPONENTS.issubset(covered_components),
        f"{purpose} audit forbidden path policy is incomplete",
    )
    _require(
        len(loaded) == reference.get("loaded_file_count"),
        f"{purpose} loaded-file count differs",
    )
    _require(
        reference.get("forbidden_access_count") == 0 and reference.get("passed") is True,
        f"{purpose} runtime audit reference is not clean",
    )
    for item in loaded:
        _require(isinstance(item, str), f"{purpose} audit contains a non-path entry")
        loaded_path = Path(item)
        components = {component.casefold() for component in loaded_path.parts}
        _require(
            not blocked_name_components.intersection(components),
            f"{purpose} audit independently reveals a component-blocked loaded path",
        )
        for root in forbidden_roots:
            try:
                loaded_path.relative_to(Path(root))
            except ValueError:
                continue
            raise ValueError(f"{purpose} audit independently reveals a root-blocked loaded path")
    return {
        "path": _relative(path),
        "sha256": expected_hash,
        "loaded_file_count": len(loaded),
        "forbidden_access_count": 0,
        "passed": True,
    }


def _validate_transition_claim(
    transition: object,
    *,
    observation_required: bool,
) -> Mapping[str, Any]:
    _require(isinstance(transition, Mapping), "Continuous binding transition is missing")
    assert isinstance(transition, Mapping)
    _require(transition.get("passed") is True, "Continuous binding transition did not pass")
    _require(
        transition.get("observation_required") is observation_required,
        "Continuous binding transition observation mode differs",
    )
    return transition


def _validate_observation_transition(
    previous: Mapping[str, Any],
    current: Mapping[str, Any],
    transition: object,
) -> None:
    claim = _validate_transition_claim(transition, observation_required=True)
    _require(current["scene_id"] == previous["scene_id"], "Observation changed opaque scene")
    _require(
        int(current["map_version"]) == int(previous["map_version"]) + 1,
        "Observation did not advance map version exactly once",
    )
    _require(current["map_sha256"] != previous["map_sha256"], "Observation map hash did not change")
    _require(
        current["scene_prefix_sha256"] != previous["scene_prefix_sha256"],
        "Observation scene-prefix hash did not change",
    )
    _require(
        current["active_prefix_sha256"] != previous["active_prefix_sha256"],
        "Observation active-prefix hash did not change",
    )
    _require(
        isinstance(current.get("observation_id"), str)
        and int(current.get("valid_depth_pixels", 0)) > 0
        and int(current.get("scan_count", 0)) > int(previous.get("scan_count", -1)),
        "Observation transition lacks a fresh nonempty RGB-D receipt",
    )
    _require(
        claim.get("before_map_version") == previous["map_version"]
        and claim.get("after_map_version") == current["map_version"]
        and claim.get("map_advanced_exactly_once") is True
        and claim.get("map_sha256_changed") is True
        and claim.get("scene_prefix_sha256_changed") is True
        and claim.get("active_prefix_sha256_changed") is True,
        "Observation transition claim disagrees with authenticated receipts",
    )


def _validate_stable_transition(
    previous: Mapping[str, Any],
    current: Mapping[str, Any],
    transition: object,
) -> None:
    claim = _validate_transition_claim(transition, observation_required=False)
    _require(claim.get("read_only_state_query") is True, "State query was not marked read-only")
    _require(
        all(current[field] == previous[field] for field in _BINDING_FIELDS),
        "Read-only state query changed a continuous binding",
    )
    _require(current.get("observation_id") is None, "Read-only state query claims an observation")
    stable = claim.get("stable_fields")
    _require(
        isinstance(stable, Mapping) and stable and all(value is True for value in stable.values()),
        "Read-only state transition did not attest stable fields",
    )


def _validate_stop_transition(
    previous: Mapping[str, Any],
    current: Mapping[str, Any],
    transition: object,
) -> None:
    claim = _validate_transition_claim(transition, observation_required=False)
    _require(
        all(current[field] == previous[field] for field in _SCENE_BINDING_FIELDS),
        "Standalone stop changed the scene/map binding",
    )
    _require(
        current["active_prefix_sha256"] != previous["active_prefix_sha256"],
        "Stop did not refresh robot prefix",
    )
    _require(current.get("stopped") is True, "Standalone stop did not latch the episode")
    _require(current.get("observation_id") is None, "Standalone stop claims an observation")
    _require(
        claim.get("before_map_version") == previous["map_version"]
        and claim.get("after_map_version") == current["map_version"]
        and claim.get("map_sha256_changed") is False
        and claim.get("scene_prefix_sha256_changed") is False
        and claim.get("active_prefix_sha256_changed") is True,
        "Stop transition claim disagrees with authenticated receipts",
    )


def _numeric_receipt(value: object) -> dict[str, Any]:
    _require(isinstance(value, Mapping), "MCP step omitted its numeric receipt")
    assert isinstance(value, Mapping)
    receipt = validate_numeric_tool_receipt(value, require_continuous_binding=True)
    _require(
        receipt.get("success") is True and receipt.get("error_code") is None,
        "Finite session contains a failed or contradictory MCP receipt",
    )
    return receipt


def _validate_grounded_steps(
    turn: Mapping[str, Any],
    previous: dict[str, Any],
    *,
    family: str,
    receipts: list[dict[str, Any]],
    action_sequence: list[str],
) -> tuple[dict[str, Any], int, float, int]:
    steps = turn.get("steps")
    _require(isinstance(steps, list) and steps, f"{family} turn has no grounded steps")
    assert isinstance(steps, list)
    _require(turn.get("step_count") == len(steps), f"{family} step count differs")
    move_count = 0
    distance_moved = 0.0
    observation_count = 0
    for index, step in enumerate(steps, start=1):
        _require(
            isinstance(step, Mapping) and step.get("step") == index,
            f"{family} steps are not consecutive",
        )
        assert isinstance(step, Mapping)
        grounding = step.get("grounding")
        _require(isinstance(grounding, Mapping), f"{family} step omitted grounding")
        assert isinstance(grounding, Mapping)
        _require(
            grounding.get("all_map_voxels_scored") is True
            and grounding.get("scored_voxels") == previous["source_voxels"]
            and step.get("active_map_sha256") == previous["map_sha256"],
            f"{family} step did not ground against every active-map voxel",
        )
        _require(
            isinstance(step.get("query_embedding_sha256"), str)
            and _SHA256.fullmatch(str(step["query_embedding_sha256"])) is not None
            and isinstance(step.get("target_state_sha256"), str)
            and _SHA256.fullmatch(str(step["target_state_sha256"])) is not None,
            f"{family} step omitted continuous grounding hashes",
        )
        call = step.get("mcp_call")
        _require(isinstance(call, Mapping), f"{family} step omitted MCP call")
        assert isinstance(call, Mapping)
        tool = call.get("tool")
        allowed = (
            {"turn", "get_robot_state"}
            if family == "face"
            else {
                "turn",
                "move_forward",
                "get_robot_state",
            }
        )
        _require(tool in allowed, f"{family} step used an unexpected MCP tool")
        arguments = call.get("arguments")
        _require(isinstance(arguments, Mapping), f"{family} MCP arguments are invalid")
        assert isinstance(arguments, Mapping)
        if tool == "turn":
            value = arguments.get("angle_degrees")
            _require(
                set(arguments) == {"angle_degrees"}
                and not isinstance(value, bool)
                and isinstance(value, (int, float))
                and math.isfinite(float(value)),
                f"{family} turn arguments are not finite numeric protocol data",
            )
        elif tool == "move_forward":
            value = arguments.get("distance_meters")
            _require(
                set(arguments) == {"distance_meters"}
                and not isinstance(value, bool)
                and isinstance(value, (int, float))
                and math.isfinite(float(value))
                and float(value) > 0.0,
                "Approach move arguments are not positive numeric protocol data",
            )
        else:
            _require(arguments == {}, f"{family} state acknowledgment has arguments")
        controller = step.get("controller_call")
        _require(isinstance(controller, Mapping), f"{family} controller call is missing")
        assert isinstance(controller, Mapping)
        expected_controller = (
            {"tool": "stop", "arguments": {}}
            if tool == "get_robot_state"
            else {"tool": tool, "arguments": dict(arguments)}
        )
        _require(
            dict(controller) == expected_controller,
            f"{family} MCP call differs from its deterministic numeric controller decision",
        )
        decision_key = "numeric_alignment" if family == "face" else "numeric_approach"
        decision = step.get(decision_key)
        _require(isinstance(decision, Mapping), f"{family} numeric decision audit is missing")
        assert isinstance(decision, Mapping)
        _require(
            decision.get("proposal_source") == "neutral_stop_sentinel"
            and decision.get("numeric_target_state_only") is True
            and decision.get("environmental_text_inputs") == []
            and decision.get("oracle_inputs_at_runtime") is False,
            f"{family} numeric controller audit overclaims learned or textual inputs",
        )
        if family == "approach":
            collision = decision.get("collision_limited_interlock")
            _require(
                isinstance(collision, Mapping)
                and collision.get("numeric_collision_map_only") is True
                and collision.get("environmental_text_inputs") == []
                and collision.get("oracle_inputs_at_runtime") is False,
                "Approach collision audit is not anonymous numeric geometry",
            )
        current = _numeric_receipt(step.get("numeric_tool_receipt"))
        if tool in {"turn", "move_forward"}:
            _validate_observation_transition(
                previous, current, step.get("continuous_binding_transition")
            )
            observation_count += 1
        else:
            _validate_stable_transition(
                previous, current, step.get("continuous_binding_transition")
            )
            _require(
                step.get("goal_settled_without_episode_latch") is True
                and current.get("stopped") is False,
                f"{family} goal-settle incorrectly latched stop",
            )
            _require(index == len(steps), f"{family} state acknowledgment was not terminal")
        if tool == "move_forward":
            moved = float(current["distance_moved"])
            delta = np.asarray(current["last_movement_delta_m"], dtype=np.float64)
            prior_xy = np.asarray(previous["position_m"][:2], dtype=np.float64)
            current_xy = np.asarray(current["position_m"][:2], dtype=np.float64)
            _require(
                math.isfinite(moved)
                and moved > 0.0
                and delta.shape == (3,)
                and np.linalg.norm(delta[:2]) > 0.0
                and np.linalg.norm(current_xy - prior_xy) > 0.0,
                "Approach movement did not produce positive numeric translation",
            )
            move_count += 1
            distance_moved += moved
        receipts.append(current)
        action_sequence.append(str(tool))
        previous = current
    _require(
        steps[-1]["mcp_call"].get("tool") == "get_robot_state"
        and turn.get("goal_settled_without_episode_latch") is True
        and turn.get("episode_stop_latched") is False,
        f"{family} turn did not settle safely without the stop latch",
    )
    return previous, move_count, distance_moved, observation_count


def build_inspection(runtime_result: str | Path) -> dict[str, Any]:
    """Authenticate one completed finite session without opening oracle inputs."""

    runtime_path = _safe_file(runtime_result, purpose="conversational MCP session result")
    result = _read_object(runtime_path, purpose="conversational MCP session result")
    policy = result.get("policy")
    transport = result.get("transport")
    turns = result.get("turns")
    _require(result.get("schema") == RUNTIME_SCHEMA, "Session runtime schema differs")
    _require(result.get("passed") is True, "Session runtime did not pass")
    _require(isinstance(result.get("scene_id"), str), "Session runtime scene ID is invalid")
    _require(result.get("environmental_text_inputs") == [], "Session runtime claims text leakage")
    _require(result.get("oracle_inputs_at_runtime") is False, "Session runtime claims oracle input")
    _require(isinstance(policy, Mapping), "Session runtime policy is missing")
    assert isinstance(policy, Mapping)
    _require(
        policy.get("name") == POLICY_NAME
        and policy.get("official_mcp_sdk_stdio_action_execution") is True
        and policy.get("one_persistent_stdio_session") is True
        and policy.get("fresh_all_voxel_grounding_before_every_navigation_decision") is True
        and policy.get("successful_motion_refreshes_map_and_complete_scene_prefix") is True
        and policy.get("learned_v3_action_head_used") is False
        and policy.get("gemma_native_function_calling_used") is False
        and policy.get("direct_function_action_execution_used") is False
        and policy.get("environmental_text_inputs") == []
        and policy.get("oracle_inputs_at_runtime") is False,
        "Session policy attestation is incomplete or overclaims learned action selection",
    )
    _require(isinstance(transport, Mapping), "Session transport attestation is missing")
    assert isinstance(transport, Mapping)
    _require(
        transport.get("implementation") == "official_python_mcp_sdk_stdio"
        and transport.get("process_boundary") is True
        and transport.get("persistent_connection") is True
        and transport.get("numeric_structured_output_only") is True
        and transport.get("tool_count") == 9,
        "Session did not use one authenticated official-MCP stdio transport",
    )
    _require(isinstance(turns, list), "Session turn transcript is invalid")
    assert isinstance(turns, list)
    command_order = [
        turn.get("command_kind") if isinstance(turn, Mapping) else None for turn in turns
    ]
    _require(tuple(command_order) == EXPECTED_COMMAND_ORDER, "Finite session command order differs")
    _require(result.get("turn_count") == len(turns), "Session turn count differs")

    initial = result.get("initial_observation")
    initial_binding = result.get("initial_binding")
    _require(isinstance(initial, Mapping), "Session initial observation is missing")
    _require(isinstance(initial_binding, Mapping), "Session initial binding is missing")
    assert isinstance(initial, Mapping) and isinstance(initial_binding, Mapping)
    initial_receipt = _numeric_receipt(initial.get("numeric_tool_receipt"))
    _validate_observation_transition(
        _binding(initial_binding),
        initial_receipt,
        initial.get("continuous_binding_transition"),
    )
    receipts: list[dict[str, Any]] = [initial_receipt]
    action_sequence = ["scan"]
    previous = initial_receipt
    total_moves = 0
    total_distance = 0.0
    observation_refreshes = 1

    for index, turn in enumerate(turns):
        _require(isinstance(turn, Mapping), "Session turn is not an object")
        assert isinstance(turn, Mapping)
        family = EXPECTED_COMMAND_ORDER[index]
        _require(
            turn.get("schema") == TURN_SCHEMA
            and turn.get("passed") is True
            and turn.get("turn_index") == index + 1
            and turn.get("environmental_text_inputs") == []
            and turn.get("oracle_inputs_at_runtime") is False,
            f"Session {family} turn did not pass its runtime contract",
        )
        if family in {"face", "approach"}:
            previous, moves, distance, observations = _validate_grounded_steps(
                turn,
                previous,
                family=family,
                receipts=receipts,
                action_sequence=action_sequence,
            )
            total_moves += moves
            total_distance += distance
            observation_refreshes += observations
            _require(
                turn.get("all_decisions_used_fresh_all_voxel_grounding") is True
                and turn.get("semantic_leaks_in_numeric_tool_receipts") == [],
                f"Session {family} turn lacks all-voxel/no-leak attestation",
            )
            continue

        call = turn.get("mcp_call")
        _require(isinstance(call, Mapping), f"Session {family} turn omitted MCP call")
        assert isinstance(call, Mapping)
        expected_tool = "get_robot_state" if family == "state" else family
        _require(
            call.get("tool") == expected_tool and call.get("arguments") == {},
            f"Session {family} turn used a different MCP action",
        )
        current = _numeric_receipt(turn.get("numeric_tool_receipt"))
        if family == "scan":
            _validate_observation_transition(
                previous, current, turn.get("continuous_binding_transition")
            )
            observation_refreshes += 1
        elif family == "state":
            _validate_stable_transition(
                previous, current, turn.get("continuous_binding_transition")
            )
        else:
            _validate_stop_transition(previous, current, turn.get("continuous_binding_transition"))
        receipts.append(current)
        action_sequence.append(str(expected_tool))
        previous = current

    _require(total_moves >= 1 and total_distance > 0.0, "Finite session contains no real movement")
    shutdown = result.get("shutdown")
    _require(
        isinstance(shutdown, Mapping)
        and shutdown.get("schema") == SHUTDOWN_SCHEMA
        and shutdown.get("passed") is True
        and shutdown.get("mcp_stop_called") is False
        and shutdown.get("episode_stop_latched") is True,
        "Session shutdown did not verify the already-latched standalone stop",
    )
    assert isinstance(shutdown, Mapping)
    shutdown_receipt = _numeric_receipt(shutdown.get("numeric_tool_receipt"))
    _validate_stable_transition(
        previous,
        shutdown_receipt,
        shutdown.get("continuous_binding_transition"),
    )
    receipts.append(shutdown_receipt)
    action_sequence.append("get_robot_state")
    previous = shutdown_receipt

    _require(
        result.get("final_stopped") is True and previous.get("stopped") is True,
        "Final stop is not latched",
    )
    _require(
        isinstance(result.get("final_binding"), Mapping)
        and _binding(result["final_binding"]) == _binding(previous),
        "Saved final binding differs from the final numeric receipt",
    )
    expected_refresh_count = 1 + len(receipts)  # initial runtime state handshake + visible calls
    _require(
        result.get("prefix_binding_refresh_count") == expected_refresh_count,
        "Prefix-binding refresh count differs from the complete MCP transcript",
    )
    leaks = _semantic_leaks(receipts)
    _require(not leaks, "Numeric MCP receipts contain environmental semantic text")

    client_reference = result.get("client_access_audit")
    server_reference = result.get("server_access_audit")
    _require(
        isinstance(client_reference, Mapping) and isinstance(server_reference, Mapping),
        "Session result lacks both process-lifetime access audits",
    )
    assert isinstance(client_reference, Mapping) and isinstance(server_reference, Mapping)
    client_audit = _validate_audit(client_reference, purpose="client")
    server_audit = _validate_audit(server_reference, purpose="server")
    _require(
        client_audit["path"] != server_audit["path"],
        "Client and server must provide distinct process-lifetime audits",
    )
    collision_count = sum(receipt.get("collision") is True for receipt in receipts)
    return {
        "schema": SCHEMA,
        "status": "passed",
        "passed": True,
        "scene_id": result["scene_id"],
        "runtime_result": {
            "path": _relative(runtime_path),
            "sha256": _sha256(runtime_path),
        },
        "checks": {
            "runtime_schema_and_pass": True,
            "exact_command_order": True,
            "every_turn_passed": True,
            "official_single_persistent_stdio_session": True,
            "honest_nonlearned_action_policy": True,
            "at_least_one_positive_translation": True,
            "all_navigation_decisions_scored_every_voxel": True,
            "every_observation_refreshed_map_and_scene_prefix": True,
            "read_only_state_queries_preserved_binding": True,
            "strict_numeric_receipt_schema": True,
            "numeric_receipts_have_no_semantic_leak": True,
            "prefix_refresh_count_matches_transcript": True,
            "client_access_audit_zero_forbidden": True,
            "server_access_audit_zero_forbidden": True,
            "standalone_stop_finally_latched": True,
        },
        "transcript": {
            "command_order": command_order,
            "action_sequence": action_sequence,
            "turn_count": len(turns),
            "numeric_receipt_count": len(receipts),
            "observation_refresh_count": observation_refreshes,
            "move_count": total_moves,
            "distance_moved_m": total_distance,
            "collision_count": collision_count,
            "prefix_binding_refresh_count": expected_refresh_count,
            "final_stopped": True,
        },
        "client_access_audit": client_audit,
        "server_access_audit": server_audit,
        "environmental_text_inputs": [],
        "oracle_inputs_opened": False,
        "oracle_target_distance_scoring_deferred": True,
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
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    payload = build_inspection(args.runtime_result)
    destination = _atomic_json(args.output, payload)
    print(
        json.dumps(
            {
                "schema": payload["schema"],
                "passed": payload["passed"],
                "scene_id": payload["scene_id"],
                "move_count": payload["transcript"]["move_count"],
                "distance_moved_m": payload["transcript"]["distance_moved_m"],
                "output": str(destination),
            },
            sort_keys=True,
            allow_nan=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
