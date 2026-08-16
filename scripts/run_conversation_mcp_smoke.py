"""Finite end-to-end conversation-agent smoke over the real semantic MCP stdio server.

This writes all mutable map/audit state to a system temporary directory and
prints one machine-readable result.  It performs no oracle or QA reads.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from semantic_3d_chat.config import PROJECT_ROOT, load_config, project_path
from semantic_3d_chat.evaluation.semantic_mcp_live_smoke import (
    semantic_server_parameters,
)
from semantic_3d_chat.robot.conversation import ConversationalEmbodiedAgent
from semantic_3d_chat.robot.mcp_stdio_runtime import MCPConversationRuntime


class _UnusedTextEncoder:
    """Numeric commands in this transport smoke never invoke semantic grounding."""

    output_dim = 1

    def encode_queries(self, queries: list[str]) -> np.ndarray:
        raise AssertionError(f"finite numeric MCP smoke unexpectedly grounded: {queries!r}")


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
        type=Path,
        default=None,
    )
    result.add_argument(
        "--robot-state-checkpoint",
        default="data_gemma4/checkpoints/robot_state_numeric_v1",
    )
    result.add_argument("--turn-degrees", type=float, default=15.0)
    result.add_argument("--timeout-seconds", type=float, default=600.0)
    result.add_argument("--python-executable", type=Path, default=Path(sys.executable))
    result.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path for an atomically published JSON evidence report.",
    )
    return result


def _runtime_asset(scene_id: str, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit if explicit.is_absolute() else PROJECT_ROOT / explicit
    filename = f"{scene_id.replace('scene_', 's_', 1)}.blend"
    return PROJECT_ROOT / "data" / "runtime_assets" / scene_id / filename


def _atomic_json(path: Path, value: dict[str, object]) -> None:
    resolved = path if path.is_absolute() else PROJECT_ROOT / path
    resolved.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{resolved.name}.", suffix=".tmp", dir=resolved.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, resolved)
    finally:
        Path(temporary).unlink(missing_ok=True)


def run(args: argparse.Namespace) -> dict[str, object]:
    started = time.monotonic()
    config = load_config(args.config)
    if config.get("robot", {}).get("auto_scan_after_motion") is not True:
        raise ValueError("finite MCP smoke requires robot.auto_scan_after_motion=true")
    base_map = project_path(config, "maps", args.scene, "voxel_map.npz").resolve()
    if not base_map.is_file():
        raise FileNotFoundError(f"sanitized base map is unavailable: {base_map}")
    runtime_asset = _runtime_asset(args.scene, args.runtime_asset).resolve()
    if not runtime_asset.is_file():
        raise FileNotFoundError(f"sanitized runtime scene asset is unavailable: {runtime_asset}")

    with tempfile.TemporaryDirectory(prefix="semantic_3d_conversation_mcp_") as temporary:
        # macOS exposes /var as a symlink to /private/var. The inference-safe
        # runtime intentionally rejects every symlink component, so pass the
        # canonical temporary path across the MCP process boundary.
        work = Path(temporary).resolve()
        persistent_map = work / "semantic_map.npz"
        audit_path = work / "mcp_file_access.json"
        parameters = semantic_server_parameters(
            python_executable=args.python_executable,
            config=args.config,
            scene_id=args.scene,
            base_checkpoint=args.base_checkpoint,
            control_checkpoint=args.control_checkpoint,
            control_runtime_config=args.control_runtime_config,
            runtime_asset=runtime_asset,
            robot_state_checkpoint=args.robot_state_checkpoint,
            persistent_map=persistent_map,
            audit_report=audit_path,
        )
        runtime = MCPConversationRuntime.connect_stdio(
            parameters,
            config,
            base_map_path=base_map,
            persistent_map_path=persistent_map,
            read_timeout_seconds=args.timeout_seconds,
            startup_timeout_seconds=args.timeout_seconds,
            call_timeout_seconds=args.timeout_seconds,
        )
        with runtime:
            agent = ConversationalEmbodiedAgent(
                runtime,
                _UnusedTextEncoder(),
                room_size_m=config["scene"]["room_size_m"],
                feature_start=0,
                feature_dim=1,
            )
            initial = runtime.prefix_binding()
            scanned = agent.handle("Scan.")
            turned = agent.handle(f"Turn right {args.turn_degrees:g} degrees.")
            stopped = agent.handle("Stop.")
            final = runtime.prefix_binding()
            refresh_count = runtime.binding_refresh_count

        if not all(value.get("success") is True for value in (scanned, turned, stopped)):
            raise AssertionError("conversation agent returned an unsuccessful MCP action")
        bindings = [
            initial,
            scanned["prefix_binding"],
            turned["prefix_binding"],
            stopped["prefix_binding"],
        ]
        versions = [int(value["map_version"]) for value in bindings]
        if versions != [0, 1, 2, 2]:
            raise AssertionError(f"unexpected action/map refresh sequence: {versions}")
        if bindings[0]["scene_prefix_sha256"] == bindings[1]["scene_prefix_sha256"]:
            raise AssertionError("explicit scan did not refresh the scene prefix")
        if bindings[1]["scene_prefix_sha256"] == bindings[2]["scene_prefix_sha256"]:
            raise AssertionError("turn auto-scan did not refresh the scene prefix")
        if bindings[2]["scene_prefix_sha256"] != bindings[3]["scene_prefix_sha256"]:
            raise AssertionError("stop unexpectedly changed the unchanged semantic map")
        if bindings[2]["active_binding_sha256"] == bindings[3]["active_binding_sha256"]:
            raise AssertionError("stop did not refresh the continuous numeric robot-state binding")
        if refresh_count != 4 or final != bindings[-1]:
            raise AssertionError("client did not accept exactly one binding per MCP result")
        if any(
            value.get("environmental_text_inputs") != []
            for value in (scanned, turned, stopped)
        ):
            raise AssertionError("conversation action result exposed environmental text")

        if not audit_path.is_file():
            raise AssertionError("MCP server did not save its lifetime file audit")
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if audit.get("passed") is not True or audit.get("forbidden_accesses") != []:
            raise AssertionError("MCP server file-access audit failed")
        loaded_files = audit.get("loaded_files")
        if not isinstance(loaded_files, list) or not loaded_files:
            raise AssertionError("MCP server file-access audit is empty")

        return {
            "schema": "semantic_3d_chat.conversation_mcp_stdio_smoke.v1",
            "passed": True,
            "transport": "stdio",
            "scene_id": args.scene,
            "agent": "ConversationalEmbodiedAgent",
            "action_commands": ["scan", "turn", "stop"],
            "map_versions": versions,
            "binding_refresh_count": refresh_count,
            "explicit_scan_scene_prefix_changed": True,
            "turn_auto_scan_scene_prefix_changed": True,
            "stop_robot_binding_changed": True,
            "final_scene_prefix_sha256": final["scene_prefix_sha256"],
            "final_active_binding_sha256": final["active_binding_sha256"],
            "numeric_structured_receipts_only": True,
            "environmental_text_inputs": [],
            "forbidden_access_count": 0,
            "loaded_file_count": len(loaded_files),
            "elapsed_seconds": time.monotonic() - started,
        }


def main() -> None:
    args = parser().parse_args()
    result = run(args)
    if args.output is not None:
        _atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
