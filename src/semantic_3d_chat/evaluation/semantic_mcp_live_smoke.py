"""Finite official-SDK smoke test for the live continuous semantic MCP server."""

from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.evaluation.mcp_transport_smoke import (
    EXPECTED_TOOLS,
    _semantic_leaks,
    _tool_content_payload,
)

_SHA256_LENGTH = 64
_HASH_FIELDS = (
    "map_sha256",
    "scene_prefix_sha256",
    "scene_control_signature_sha256",
    "binding_sha256",
    "active_prefix_sha256",
    "robot_state_sha256",
    "robot_tokens_sha256",
    "robot_state_encoder_sha256",
    "active_binding_sha256",
)


def _absolute(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    return (
        candidate.absolute() if candidate.is_absolute() else (PROJECT_ROOT / candidate).absolute()
    )


def _required_hash(payload: dict[str, Any], field: str) -> str:
    value = payload.get(field)
    if (
        not isinstance(value, str)
        or len(value) != _SHA256_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise AssertionError(f"{field} is not a lowercase SHA-256")
    return value


def validate_live_refresh(
    initial: dict[str, Any],
    scanned: dict[str, Any],
    turned: dict[str, Any],
    *,
    turn_degrees: float,
) -> dict[str, Any]:
    """Validate two real observation commits and return compact evidence."""

    for name, payload in (("initial", initial), ("scan", scanned), ("turn", turned)):
        if payload.get("success") is not True:
            raise AssertionError(f"{name} MCP receipt was unsuccessful")
        for field in _HASH_FIELDS:
            _required_hash(payload, field)
        if payload.get("schema") != "semantic_3d_chat.scene_prefix_binding.v2":
            raise AssertionError(f"{name} did not expose a v2 continuous-prefix binding")
        if int(payload.get("source_voxels", 0)) < 1:
            raise AssertionError(f"{name} source voxel count is empty")
        if int(payload.get("processed_voxels", 0)) < 1:
            raise AssertionError(f"{name} processed voxel count is empty")

    initial_version = int(initial["map_version"])
    scan_version = int(scanned["map_version"])
    turn_version = int(turned["map_version"])
    if scan_version != initial_version + 1 or turn_version != scan_version + 1:
        raise AssertionError("scan and auto-scan turn did not commit consecutive map versions")
    if int(scanned["scene_version"]) != scan_version:
        raise AssertionError("scan scene version differs from its persistent-map binding")
    if int(turned["scene_version"]) != turn_version:
        raise AssertionError("turn scene version differs from its persistent-map binding")
    if int(scanned["scan_count"]) != int(initial["scan_count"]) + 1:
        raise AssertionError("explicit scan did not advance the scan count")
    if int(turned["scan_count"]) != int(scanned["scan_count"]) + 1:
        raise AssertionError("turn did not trigger its configured automatic scan")
    if int(scanned["valid_depth_pixels"]) < 1:
        raise AssertionError("explicit scan returned no valid metric depth")
    if int(turned["valid_depth_pixels"]) < 1:
        raise AssertionError("turn's automatic scan returned no valid metric depth")
    expected_yaw = (float(initial["body_yaw_degrees"]) + turn_degrees + 180.0) % 360.0 - 180.0
    if abs(float(turned["body_yaw_degrees"]) - expected_yaw) > 1e-6:
        raise AssertionError("bounded turn returned an unexpected numeric pose")

    changed_fields: dict[str, bool] = {}
    for field in (
        "map_sha256",
        "scene_prefix_sha256",
        "scene_control_signature_sha256",
        "binding_sha256",
        "active_prefix_sha256",
        "robot_state_sha256",
        "robot_tokens_sha256",
        "active_binding_sha256",
    ):
        changed_fields[f"scan_changed_{field}"] = scanned[field] != initial[field]
        changed_fields[f"turn_changed_{field}"] = turned[field] != scanned[field]
    required_changes = tuple(changed_fields)
    if not all(changed_fields[field] for field in required_changes):
        unchanged = sorted(field for field in required_changes if not changed_fields[field])
        raise AssertionError(f"continuous refresh hashes did not change: {unchanged}")
    if turned["robot_state_encoder_sha256"] != initial["robot_state_encoder_sha256"]:
        raise AssertionError("robot-state encoder identity changed during the episode")

    return {
        "initial_map_version": initial_version,
        "scan_map_version": scan_version,
        "turn_map_version": turn_version,
        "explicit_scan_valid_depth_pixels": int(scanned["valid_depth_pixels"]),
        "turn_auto_scan_valid_depth_pixels": int(turned["valid_depth_pixels"]),
        "initial_source_voxels": int(initial["source_voxels"]),
        "scan_source_voxels": int(scanned["source_voxels"]),
        "turn_source_voxels": int(turned["source_voxels"]),
        "initial_processed_voxels": int(initial["processed_voxels"]),
        "scan_processed_voxels": int(scanned["processed_voxels"]),
        "turn_processed_voxels": int(turned["processed_voxels"]),
        "bounded_turn_degrees": turn_degrees,
        "resulting_body_yaw_degrees": float(turned["body_yaw_degrees"]),
        **changed_fields,
        "robot_state_encoder_identity_invariant": True,
        "initial_hashes": {field: initial[field] for field in _HASH_FIELDS},
        "scan_hashes": {field: scanned[field] for field in _HASH_FIELDS},
        "turn_hashes": {field: turned[field] for field in _HASH_FIELDS},
    }


def semantic_server_parameters(
    *,
    python_executable: str | Path,
    config: str | Path,
    scene_id: str,
    base_checkpoint: str | Path,
    control_checkpoint: str | Path,
    control_runtime_config: str | Path,
    runtime_asset: str | Path,
    robot_state_checkpoint: str | Path,
    persistent_map: str | Path,
    audit_report: str | Path,
) -> StdioServerParameters:
    """Construct explicit fail-closed arguments for the production server."""

    executable = Path(python_executable)
    if not executable.is_absolute():
        executable = (Path.cwd() / executable).absolute()
    child_environment = dict(os.environ)
    source_root = str(PROJECT_ROOT / "src")
    inherited_pythonpath = child_environment.get("PYTHONPATH")
    child_environment["PYTHONPATH"] = (
        source_root
        if not inherited_pythonpath
        else os.pathsep.join((source_root, inherited_pythonpath))
    )
    child_environment.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    args = [
        "-m",
        "semantic_3d_chat.mcp_server.server",
        "--config",
        str(_absolute(config)),
        "--scene",
        scene_id,
        "--checkpoint",
        str(_absolute(base_checkpoint)),
        "--control-checkpoint",
        str(_absolute(control_checkpoint)),
        "--control-runtime-config",
        str(_absolute(control_runtime_config)),
        "--runtime-asset",
        str(_absolute(runtime_asset)),
        "--robot-state-checkpoint",
        str(_absolute(robot_state_checkpoint)),
        "--persistent-map",
        str(_absolute(persistent_map)),
        "--audit-report",
        str(_absolute(audit_report)),
        "--transport",
        "stdio",
    ]
    return StdioServerParameters(
        command=str(executable),
        args=args,
        cwd=PROJECT_ROOT,
        env=child_environment,
    )


async def run_semantic_mcp_live_smoke(
    *,
    config: str | Path,
    scene_id: str,
    base_checkpoint: str | Path,
    control_checkpoint: str | Path,
    control_runtime_config: str | Path,
    runtime_asset: str | Path,
    robot_state_checkpoint: str | Path,
    audit_report: str | Path,
    python_executable: str | Path | None = None,
    turn_degrees: float = 15.0,
) -> dict[str, Any]:
    """Start the real server, execute two RGB-D updates, and close it cleanly."""

    started = time.monotonic()
    audit_path = _absolute(audit_report)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    workspace_root = PROJECT_ROOT / "reports" / "gemma4" / "artifacts"
    workspace_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="semantic_3d_mcp_live_",
        dir=workspace_root,
    ) as temporary:
        persistent_map = Path(temporary) / "semantic_map.npz"
        parameters = semantic_server_parameters(
            python_executable=sys.executable if python_executable is None else python_executable,
            config=config,
            scene_id=scene_id,
            base_checkpoint=base_checkpoint,
            control_checkpoint=control_checkpoint,
            control_runtime_config=control_runtime_config,
            runtime_asset=runtime_asset,
            robot_state_checkpoint=robot_state_checkpoint,
            persistent_map=persistent_map,
            audit_report=audit_path,
        )
        checked_payloads: list[Any] = []
        async with stdio_client(parameters) as (read_stream, write_stream):  # noqa: SIM117
            async with ClientSession(
                read_stream,
                write_stream,
                read_timeout_seconds=600.0,
            ) as session:
                initialized = await session.initialize()
                listed = await session.list_tools()
                tools = {tool.name: tool for tool in listed.tools}
                if set(tools) != EXPECTED_TOOLS:
                    raise AssertionError("live semantic MCP tool inventory changed")

                initial_result = await session.call_tool("get_robot_state", {})
                scan_result = await session.call_tool("scan", {})
                turn_result = await session.call_tool("turn", {"angle_degrees": turn_degrees})
                results = (initial_result, scan_result, turn_result)
                if any(result.is_error or result.structured_content is None for result in results):
                    raise AssertionError("live semantic MCP request failed")
                for result in results:
                    checked_payloads.extend(
                        (result.structured_content, _tool_content_payload(result))
                    )
                leaks = _semantic_leaks(checked_payloads)
                if leaks:
                    raise AssertionError(
                        f"semantic data leaked through live MCP tool results: {leaks}"
                    )
                evidence = validate_live_refresh(
                    initial_result.structured_content,
                    scan_result.structured_content,
                    turn_result.structured_content,
                    turn_degrees=turn_degrees,
                )

        if not audit_path.is_file():
            raise AssertionError("semantic MCP server did not persist its lifetime file audit")
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if (
            audit.get("passed") is not True
            or audit.get("block_forbidden") is not True
            or audit.get("forbidden_accesses") != []
        ):
            raise AssertionError("semantic MCP lifetime file audit did not pass")
        loaded_files = audit.get("loaded_files")
        if not isinstance(loaded_files, list) or not loaded_files:
            raise AssertionError("semantic MCP lifetime file audit is empty")

    return {
        "schema": "semantic_3d_chat.semantic_mcp_live_smoke.v1",
        "passed": True,
        "transport": "stdio",
        "mcp_sdk_version": importlib.metadata.version("mcp"),
        "protocol_version": str(initialized.protocol_version),
        "server_name": initialized.server_info.name,
        "server_version": initialized.server_info.version,
        "scene_id": scene_id,
        "tool_count": len(tools),
        "tools": sorted(tools),
        "base_checkpoint": str(_absolute(base_checkpoint)),
        "control_checkpoint": str(_absolute(control_checkpoint)),
        "continuous_controller_active": True,
        "full_image_explicit_scan_count": 1,
        "full_image_auto_scan_count": 1,
        "semantic_result_leaks": [],
        "environmental_text_in_tool_results": False,
        "audit_report": str(audit_path),
        "loaded_file_count": len(loaded_files),
        "forbidden_access_count": 0,
        "elapsed_seconds": time.monotonic() - started,
        **evidence,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", default="configs/runtime/embodied_live.yaml")
    result.add_argument("--scene", default="scene_000001")
    result.add_argument(
        "--base-checkpoint",
        default="data_gemma4/runtime/checkpoints/gemma4_v54_release_v1",
    )
    result.add_argument(
        "--control-checkpoint",
        default="data_gemma4/runtime/checkpoints/gemma4_v75_nll_control_release_v1",
    )
    result.add_argument(
        "--control-runtime-config",
        default="configs/runtime/gemma4_v56_question_control.yaml",
    )
    result.add_argument(
        "--runtime-asset",
        default="data/runtime_assets/scene_000001/s_000001.blend",
    )
    result.add_argument(
        "--robot-state-checkpoint",
        default="data_gemma4/checkpoints/robot_state_numeric_v1",
    )
    result.add_argument(
        "--audit-report",
        type=Path,
        default=Path("reports/gemma4/metrics/v75_semantic_mcp_live_file_access_scene_000001.json"),
    )
    result.add_argument(
        "--output",
        type=Path,
        default=Path("reports/gemma4/metrics/v75_semantic_mcp_live_smoke_scene_000001.json"),
    )
    result.add_argument("--turn-degrees", type=float, default=15.0)
    return result


def main() -> None:
    args = parser().parse_args()
    report = asyncio.run(
        run_semantic_mcp_live_smoke(
            config=args.config,
            scene_id=args.scene,
            base_checkpoint=args.base_checkpoint,
            control_checkpoint=args.control_checkpoint,
            control_runtime_config=args.control_runtime_config,
            runtime_asset=args.runtime_asset,
            robot_state_checkpoint=args.robot_state_checkpoint,
            audit_report=args.audit_report,
            turn_degrees=args.turn_degrees,
        )
    )
    output = _absolute(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
