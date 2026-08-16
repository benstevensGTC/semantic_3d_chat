"""Run one promoted-V96 conversational navigation instruction locally."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from semantic_3d_chat.chat.file_audit import FileAccessAudit
from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.robot.v96_co_resident_mcp_agent import (
    V96CoResidentMCPAgent,
)
from semantic_3d_chat.robot.v96_release_action import (
    V3_NAVIGATION_CHECKPOINT,
    build_v96_release_action_backend,
)
from semantic_3d_chat.robot.v96_release_embodied import (
    build_promoted_v96_embodied_runtime,
)


def _absolute(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    return Path(os.path.abspath(candidate if candidate.is_absolute() else PROJECT_ROOT / candidate))


def _runtime_audit() -> FileAccessAudit:
    roots = [
        PROJECT_ROOT / "data/oracle",
        PROJECT_ROOT / "data/qa",
        PROJECT_ROOT / "data_gemma4/oracle",
        PROJECT_ROOT / "data_gemma4/qa",
        PROJECT_ROOT / "reports/gemma4/predictions",
        PROJECT_ROOT / "reports/gemma4/questions",
        PROJECT_ROOT / "reports/gemma4/scorer_only",
        PROJECT_ROOT / "configs/experiments",
    ]
    return FileAccessAudit(roots, block_forbidden=True)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


async def run(args: argparse.Namespace) -> dict[str, Any]:
    audit = _runtime_audit()
    with audit:
        runtime = build_promoted_v96_embodied_runtime(
            args.scene,
            runtime_asset=args.runtime_asset,
            persistent_map_path=args.persistent_map,
            scan_output_directory=args.scan_output,
            audit=audit,
            local_files_only=True,
        )
        backend = build_v96_release_action_backend(
            runtime,
            runtime.config,
            navigation_checkpoint=V3_NAVIGATION_CHECKPOINT,
            audit=audit,
        )
        agent = V96CoResidentMCPAgent(runtime, backend, runtime.config)
        result = await agent.run_instruction(
            args.instruction,
            max_steps=args.max_steps,
        )
    audit.assert_clean()
    payload = result.as_dict()
    payload.update(
        {
            "phase": "v96_promoted_embodied_navigation",
            "scene_id": args.scene,
            "runtime_forbidden_access_count": 0,
            "runtime_loaded_file_count": len(audit.unique_paths),
            "promoted_static_release_required": True,
            "source_policy_retrained_on_v96": False,
            "navigation_held_out_claim": False,
        }
    )
    if args.audit_report is not None:
        audit.save(_absolute(args.audit_report))
    if args.output is not None:
        _write_json_atomic(_absolute(args.output), payload)
    return payload


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--scene", default="scene_000025")
    result.add_argument("--runtime-asset", required=True)
    result.add_argument("--persistent-map")
    result.add_argument("--scan-output")
    result.add_argument("--instruction", required=True)
    result.add_argument("--max-steps", type=int, default=24)
    result.add_argument("--output")
    result.add_argument("--audit-report")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    payload = asyncio.run(run(args))
    print(json.dumps(payload, sort_keys=True, allow_nan=False))
    return 0 if payload["termination_reason"] == "stop" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "parser", "run"]
