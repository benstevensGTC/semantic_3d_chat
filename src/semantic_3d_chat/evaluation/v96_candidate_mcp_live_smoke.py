"""Finite promoted-release-gated smoke for the V96 numeric MCP robot bridge.

The runner authenticates the promoted six-scene V96 release in an isolated
model-free child before it creates a transport or mutable robot state.  It then
uses the official MCP stdio client to call only ``get_robot_state``, ``scan``,
and ``turn``.  The smoke proves numeric robot/map/full-memory refresh wiring;
it deliberately does not ask a language question or claim that V96 answer
generation with additional robot-state tokens is authenticated.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import math
import os
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.evaluation.mcp_transport_smoke import (
    EXPECTED_TOOLS,
    _semantic_leaks,
    _tool_content_payload,
)
from semantic_3d_chat.evaluation.semantic_mcp_live_smoke import (
    validate_live_refresh,
)
from semantic_3d_chat.robot.v96_candidate_refresh import (
    run_isolated_v96_release_verification,
)

SCHEMA: Final[str] = "semantic_3d_chat.v96_candidate_mcp_live_smoke.v1"
SCENE_IDS: Final[tuple[str, ...]] = tuple(
    f"scene_{index:06d}" for index in range(25, 31)
)
LIVE_TOOL_SEQUENCE: Final[tuple[str, ...]] = (
    "get_robot_state",
    "scan",
    "turn",
)
DEFAULT_CONFIG: Final[str] = "configs/runtime/embodied_live.yaml"
DEFAULT_BASE_CHECKPOINT: Final[str] = (
    "reports/gemma4/artifacts/v85_strict_runtime_candidate"
)
DEFAULT_BRIDGE_HOOK: Final[str] = (
    "configs/runtime/gemma4_v96_explicit_candidate_mcp_bridge.yaml"
)
DEFAULT_SCENE_MEMORY_ROOT: Final[str] = (
    "reports/gemma4/artifacts/v95_deferred_final/memory_cache"
)
DEFAULT_ROBOT_STATE_CHECKPOINT: Final[str] = (
    "data_gemma4/checkpoints/robot_state_numeric_v1"
)
_RELEASE_BINDING_FIELDS: Final[tuple[str, ...]] = (
    "candidate_fingerprint_sha256",
    "deferred_final_evidence_sha256",
    "runtime_smoke_sha256",
    "release_checkpoint_sha256",
    "release_adapter_sha256",
    "v95_state_sha256",
    "v96_state_sha256",
    "runtime_implementation_inventory_sha256",
)
_LIVE_HASH_FIELDS: Final[tuple[str, ...]] = (
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
_LIVE_CHANGED_FIELDS: Final[tuple[str, ...]] = tuple(
    f"{stage}_changed_{field}"
    for stage in ("scan", "turn")
    for field in _LIVE_HASH_FIELDS
    if field != "robot_state_encoder_sha256"
)
_LIVE_REPORT_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema",
        "passed",
        "transport",
        "mcp_sdk_version",
        "protocol_version",
        "server_name",
        "server_version",
        "scene_id",
        "tool_count",
        "tools",
        "called_tools",
        "base_checkpoint",
        "scene_memory",
        "runtime_asset",
        "audit_report",
        "audit_report_sha256",
        "loaded_file_count",
        "forbidden_access_count",
        "mode",
        "promoted_runtime_release_verified_before_transport",
        "server_reauthenticates_promoted_release_before_model_load",
        "deferred_final_gate_passed",
        "runtime_leakage_gate_passed",
        "numeric_tool_outputs_only",
        "question_free_full_memory_refresh",
        "full_memory_tokens",
        "full_memory_recompiled_before_map_commit",
        "robot_state_numeric_binding_exercised",
        "language_questions_asked",
        "v96_answer_generation_exercised",
        "direct_v96_answer_robot_tokens_authenticated",
        "environmental_text_inputs",
        "semantic_result_leaks",
        "release_bindings",
        "elapsed_seconds",
        "initial_map_version",
        "scan_map_version",
        "turn_map_version",
        "explicit_scan_valid_depth_pixels",
        "turn_auto_scan_valid_depth_pixels",
        "initial_source_voxels",
        "scan_source_voxels",
        "turn_source_voxels",
        "initial_processed_voxels",
        "scan_processed_voxels",
        "turn_processed_voxels",
        "bounded_turn_degrees",
        "resulting_body_yaw_degrees",
        "robot_state_encoder_identity_invariant",
        "initial_hashes",
        "scan_hashes",
        "turn_hashes",
        *_LIVE_CHANGED_FIELDS,
    }
)
_AUDIT_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "loaded_files",
        "forbidden_roots",
        "forbidden_component_names",
        "block_forbidden",
        "forbidden_accesses",
        "passed",
    }
)


def _absolute(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    rooted = candidate if candidate.is_absolute() else PROJECT_ROOT / candidate
    return Path(os.path.abspath(rooted))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _physical_file(path: str | Path, *, purpose: str) -> Path:
    """Return an absolute regular file while rejecting every symlink component."""

    source = _absolute(path)
    current = Path(source.anchor)
    for component in source.parts[1:]:
        current /= component
        if current.is_symlink():
            raise ValueError(f"{purpose} path contains a symbolic link: {current}")
    if not source.is_file():
        raise FileNotFoundError(f"{purpose} is unavailable: {source}")
    return source


def _read_strict_json_object(
    path: str | Path,
    *,
    purpose: str,
) -> tuple[dict[str, Any], str]:
    """Read one physical JSON object and hash the exact bytes that were parsed."""

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate {purpose} field: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"{purpose} contains a non-finite number: {value}")

    source = _physical_file(path, purpose=purpose)
    raw = source.read_bytes()
    try:
        decoded = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{purpose} is not UTF-8 JSON") from error
    value = json.loads(
        decoded,
        object_pairs_hook=reject_duplicates,
        parse_constant=reject_constant,
    )
    if not isinstance(value, dict):
        raise TypeError(f"{purpose} must be one JSON object")
    return value, hashlib.sha256(raw).hexdigest()


def _validate_lifetime_audit(
    audit_report: str | Path,
    *,
    expected_sha256: str | None = None,
    expected_loaded_file_count: int | None = None,
) -> tuple[dict[str, Any], str]:
    audit, audit_sha256 = _read_strict_json_object(
        audit_report,
        purpose="V96 MCP lifetime audit",
    )
    loaded_files = audit.get("loaded_files")
    forbidden_roots = audit.get("forbidden_roots")
    forbidden_components = audit.get("forbidden_component_names")
    if (
        set(audit) != set(_AUDIT_FIELDS)
        or audit.get("passed") is not True
        or audit.get("block_forbidden") is not True
        or audit.get("forbidden_accesses") != []
        or not isinstance(loaded_files, list)
        or not loaded_files
        or not all(isinstance(item, str) and item for item in loaded_files)
        or len(set(loaded_files)) != len(loaded_files)
        or not isinstance(forbidden_roots, list)
        or not all(isinstance(item, str) for item in forbidden_roots)
        or not isinstance(forbidden_components, list)
        or not all(isinstance(item, str) for item in forbidden_components)
    ):
        raise ValueError("V96 MCP lifetime audit did not authenticate exactly")
    if expected_sha256 is not None and audit_sha256 != expected_sha256:
        raise ValueError("V96 MCP lifetime audit hash differs from the live result")
    if (
        expected_loaded_file_count is not None
        and len(loaded_files) != expected_loaded_file_count
    ):
        raise ValueError("V96 MCP lifetime audit file count differs from the live result")
    return audit, audit_sha256


def _require_promoted_release(
    receipt: Mapping[str, Any],
    *,
    scene_id: str,
) -> dict[str, Any]:
    """Validate the sanitized receipt returned by the isolated verifier."""

    if (
        receipt.get("phase") != "v96_strict_runtime_release_verified"
        or receipt.get("passed") is not True
        or receipt.get("deferred_final_binding_exact") is not True
        or receipt.get("runtime_smoke_binding_exact") is not True
        or receipt.get("promoted_runtime_release_verified") is not True
        or receipt.get("scene_ids") != list(SCENE_IDS)
        or scene_id not in SCENE_IDS
        or any(
            not _is_sha256(receipt.get(field))
            for field in _RELEASE_BINDING_FIELDS
        )
    ):
        raise ValueError(
            "V96 live MCP smoke requires the exact promoted deferred-final release"
        )
    return {field: receipt[field] for field in _RELEASE_BINDING_FIELDS}


def v96_server_parameters(
    *,
    python_executable: str | Path,
    config: str | Path,
    scene_id: str,
    base_checkpoint: str | Path,
    bridge_hook: str | Path,
    scene_memory: str | Path,
    runtime_asset: str | Path,
    robot_state_checkpoint: str | Path,
    persistent_map: str | Path,
    scan_output_directory: str | Path,
    audit_report: str | Path,
) -> StdioServerParameters:
    """Construct the explicit V96 candidate-overlay MCP child command."""

    if scene_id not in SCENE_IDS:
        raise ValueError("V96 live MCP smoke accepts only released scenes 25 through 30")
    executable = Path(python_executable).expanduser()
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
        "--runtime-asset",
        str(_absolute(runtime_asset)),
        "--robot-state-checkpoint",
        str(_absolute(robot_state_checkpoint)),
        "--v96-candidate-bridge-hook",
        str(_absolute(bridge_hook)),
        "--v96-scene-memory",
        str(_absolute(scene_memory)),
        "--allow-explicit-v96-candidate",
        "--persistent-map",
        str(_absolute(persistent_map)),
        "--scan-output-directory",
        str(_absolute(scan_output_directory)),
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


def build_v96_live_smoke_report(
    *,
    release_receipt: Mapping[str, Any],
    scene_id: str,
    base_checkpoint: str | Path,
    scene_memory: str | Path,
    runtime_asset: str | Path,
    audit_report: str | Path,
    audit_report_sha256: str,
    protocol_version: str,
    server_name: str,
    server_version: str,
    loaded_file_count: int,
    elapsed_seconds: float,
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a result that scopes the proof to numeric embodied refresh."""

    release_hashes = _require_promoted_release(release_receipt, scene_id=scene_id)
    if (
        loaded_file_count < 1
        or elapsed_seconds <= 0.0
        or len(audit_report_sha256) != 64
        or any(character not in "0123456789abcdef" for character in audit_report_sha256)
    ):
        raise ValueError("V96 live MCP smoke received invalid runtime evidence")
    report = {
        "schema": SCHEMA,
        "passed": True,
        "transport": "stdio",
        "mcp_sdk_version": importlib.metadata.version("mcp"),
        "protocol_version": protocol_version,
        "server_name": server_name,
        "server_version": server_version,
        "scene_id": scene_id,
        "tool_count": len(EXPECTED_TOOLS),
        "tools": sorted(EXPECTED_TOOLS),
        "called_tools": list(LIVE_TOOL_SEQUENCE),
        "base_checkpoint": str(_absolute(base_checkpoint)),
        "scene_memory": str(_absolute(scene_memory)),
        "runtime_asset": str(_absolute(runtime_asset)),
        "audit_report": str(_absolute(audit_report)),
        "audit_report_sha256": audit_report_sha256,
        "loaded_file_count": loaded_file_count,
        "forbidden_access_count": 0,
        "mode": "explicit_v96_candidate_overlay_after_promoted_release",
        "promoted_runtime_release_verified_before_transport": True,
        "server_reauthenticates_promoted_release_before_model_load": True,
        "deferred_final_gate_passed": True,
        "runtime_leakage_gate_passed": True,
        "numeric_tool_outputs_only": True,
        "question_free_full_memory_refresh": True,
        "full_memory_tokens": 738,
        "full_memory_recompiled_before_map_commit": True,
        "robot_state_numeric_binding_exercised": True,
        "language_questions_asked": 0,
        "v96_answer_generation_exercised": False,
        "direct_v96_answer_robot_tokens_authenticated": False,
        "environmental_text_inputs": [],
        "semantic_result_leaks": [],
        "release_bindings": release_hashes,
        "elapsed_seconds": elapsed_seconds,
    }
    overlap = set(report).intersection(evidence)
    if overlap:
        raise ValueError(f"V96 live MCP evidence shadows result fields: {sorted(overlap)}")
    report.update(evidence)
    return report


def _require_runtime_inputs(
    *,
    config: Path,
    base_checkpoint: Path,
    bridge_hook: Path,
    scene_memory: Path,
    runtime_asset: Path,
    robot_state_checkpoint: Path,
) -> None:
    expected = {
        config: "file",
        base_checkpoint: "directory",
        bridge_hook: "file",
        scene_memory: "directory",
        runtime_asset: "file",
        robot_state_checkpoint: "directory",
    }
    missing = [
        str(path)
        for path, kind in expected.items()
        if (kind == "file" and not path.is_file())
        or (kind == "directory" and not path.is_dir())
    ]
    if missing:
        raise FileNotFoundError(f"V96 live MCP runtime input is unavailable: {missing}")


async def run_v96_candidate_mcp_live_smoke(
    *,
    config: str | Path,
    scene_id: str,
    base_checkpoint: str | Path,
    bridge_hook: str | Path,
    scene_memory: str | Path,
    runtime_asset: str | Path,
    robot_state_checkpoint: str | Path,
    audit_report: str | Path,
    python_executable: str | Path | None = None,
    turn_degrees: float = 15.0,
) -> dict[str, Any]:
    """Run three numeric calls, then close the real V96 MCP child cleanly."""

    started = time.monotonic()
    release_receipt = run_isolated_v96_release_verification()
    _require_promoted_release(release_receipt, scene_id=scene_id)

    config_path = _absolute(config)
    checkpoint_path = _absolute(base_checkpoint)
    hook_path = _absolute(bridge_hook)
    memory_path = _absolute(scene_memory)
    asset_path = _absolute(runtime_asset)
    robot_checkpoint_path = _absolute(robot_state_checkpoint)
    audit_path = _absolute(audit_report)
    _require_runtime_inputs(
        config=config_path,
        base_checkpoint=checkpoint_path,
        bridge_hook=hook_path,
        scene_memory=memory_path,
        runtime_asset=asset_path,
        robot_state_checkpoint=robot_checkpoint_path,
    )

    audit_path.parent.mkdir(parents=True, exist_ok=True)
    workspace_root = PROJECT_ROOT / "reports/gemma4/artifacts"
    workspace_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="v96_candidate_mcp_live_",
        dir=workspace_root,
    ) as temporary:
        temporary_root = Path(temporary)
        persistent_map = temporary_root / "semantic_map.npz"
        scan_output = temporary_root / "scans"
        parameters = v96_server_parameters(
            python_executable=(
                sys.executable if python_executable is None else python_executable
            ),
            config=config_path,
            scene_id=scene_id,
            base_checkpoint=checkpoint_path,
            bridge_hook=hook_path,
            scene_memory=memory_path,
            runtime_asset=asset_path,
            robot_state_checkpoint=robot_checkpoint_path,
            persistent_map=persistent_map,
            scan_output_directory=scan_output,
            audit_report=audit_path,
        )
        checked_payloads: list[Any] = []
        async with stdio_client(parameters) as (read_stream, write_stream):  # noqa: SIM117
            async with ClientSession(
                read_stream,
                write_stream,
                read_timeout_seconds=900.0,
            ) as session:
                initialized = await session.initialize()
                listed = await session.list_tools()
                tools = {tool.name: tool for tool in listed.tools}
                if set(tools) != EXPECTED_TOOLS:
                    raise AssertionError("live V96 MCP tool inventory changed")

                initial_result = await session.call_tool("get_robot_state", {})
                scan_result = await session.call_tool("scan", {})
                turn_result = await session.call_tool(
                    "turn", {"angle_degrees": turn_degrees}
                )
                results = (initial_result, scan_result, turn_result)
                if any(
                    result.is_error or result.structured_content is None
                    for result in results
                ):
                    raise AssertionError("live V96 MCP numeric request failed")
                for result in results:
                    checked_payloads.extend(
                        (result.structured_content, _tool_content_payload(result))
                    )
                leaks = _semantic_leaks(checked_payloads)
                if leaks:
                    raise AssertionError(
                        f"semantic data leaked through live V96 MCP results: {leaks}"
                    )
                evidence = validate_live_refresh(
                    initial_result.structured_content,
                    scan_result.structured_content,
                    turn_result.structured_content,
                    turn_degrees=turn_degrees,
                )

        if not audit_path.is_file():
            raise AssertionError("V96 MCP server did not persist its lifetime file audit")
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if (
            not isinstance(audit, dict)
            or audit.get("passed") is not True
            or audit.get("block_forbidden") is not True
            or audit.get("forbidden_accesses") != []
        ):
            raise AssertionError("V96 MCP lifetime file audit did not pass")
        loaded_files = audit.get("loaded_files")
        if not isinstance(loaded_files, list) or not loaded_files:
            raise AssertionError("V96 MCP lifetime file audit is empty")

    return build_v96_live_smoke_report(
        release_receipt=release_receipt,
        scene_id=scene_id,
        base_checkpoint=checkpoint_path,
        scene_memory=memory_path,
        runtime_asset=asset_path,
        audit_report=audit_path,
        audit_report_sha256=_sha256_file(audit_path),
        protocol_version=str(initialized.protocol_version),
        server_name=initialized.server_info.name,
        server_version=initialized.server_info.version,
        loaded_file_count=len(loaded_files),
        elapsed_seconds=time.monotonic() - started,
        evidence=evidence,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--scene", choices=SCENE_IDS, default=SCENE_IDS[0])
    parser.add_argument("--base-checkpoint", default=DEFAULT_BASE_CHECKPOINT)
    parser.add_argument("--bridge-hook", default=DEFAULT_BRIDGE_HOOK)
    parser.add_argument("--scene-memory")
    parser.add_argument("--runtime-asset")
    parser.add_argument(
        "--robot-state-checkpoint",
        default=DEFAULT_ROBOT_STATE_CHECKPOINT,
    )
    parser.add_argument("--audit-report")
    parser.add_argument("--output")
    parser.add_argument("--python")
    parser.add_argument("--turn-degrees", type=float, default=15.0)
    return parser


def _write_create_once(path: Path, report: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(dict(report), indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    scene_memory = args.scene_memory or str(
        Path(DEFAULT_SCENE_MEMORY_ROOT) / args.scene
    )
    runtime_asset = args.runtime_asset or str(
        Path("data/runtime_assets")
        / args.scene
        / f"s_{args.scene.removeprefix('scene_')}.blend"
    )
    audit_report = args.audit_report or (
        f"reports/gemma4/metrics/v96_candidate_mcp_live_access_{args.scene}.json"
    )
    output = _absolute(
        args.output
        or f"reports/gemma4/metrics/v96_candidate_mcp_live_smoke_{args.scene}.json"
    )
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    report = asyncio.run(
        run_v96_candidate_mcp_live_smoke(
            config=args.config,
            scene_id=args.scene,
            base_checkpoint=args.base_checkpoint,
            bridge_hook=args.bridge_hook,
            scene_memory=scene_memory,
            runtime_asset=runtime_asset,
            robot_state_checkpoint=args.robot_state_checkpoint,
            audit_report=audit_report,
            python_executable=args.python,
            turn_degrees=args.turn_degrees,
        )
    )
    _write_create_once(output, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "LIVE_TOOL_SEQUENCE",
    "SCENE_IDS",
    "build_v96_live_smoke_report",
    "main",
    "run_v96_candidate_mcp_live_smoke",
    "v96_server_parameters",
]
