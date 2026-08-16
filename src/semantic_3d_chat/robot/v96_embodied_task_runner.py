"""Sanitized per-scene V96 held-out navigation runtime.

This process never imports the scorer module and never receives an oracle
category, target coordinate, expected answer, or scene label inventory.  Its
only task-bearing file contains an opaque scene ID plus literal user navigation
instructions.  All environmental state reaches the frozen controller through
the 738 continuous scene tokens, four numeric robot tokens, and all-voxel
continuous target grounding.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from semantic_3d_chat.chat.file_audit import FileAccessAudit
from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.mcp_server.server import build_server
from semantic_3d_chat.robot.mcp_stdio_runtime import validate_numeric_tool_receipt
from semantic_3d_chat.robot.v96_candidate_refresh import (
    run_isolated_v96_release_verification,
)
from semantic_3d_chat.robot.v96_co_resident_mcp_agent import V96CoResidentMCPAgent
from semantic_3d_chat.robot.v96_release_action import (
    build_v96_release_action_backend,
)
from semantic_3d_chat.robot.v96_release_embodied import (
    DEFAULT_ROBOT_STATE_CHECKPOINT,
    RELEASE_SCENE_IDS,
    build_promoted_v96_embodied_runtime,
    validate_promoted_v96_release_receipt,
)
from semantic_3d_chat.robot.v96_runtime_source_contract import runtime_source_paths

TASK_INPUT_SCHEMA: Final[str] = (
    "semantic_3d_chat.v96_embodied_runtime_task_input.v1"
)
EVIDENCE_SCHEMA: Final[str] = (
    "semantic_3d_chat.v96_embodied_navigation_runtime_evidence.v2"
)
ACCESS_LOG_SCHEMA: Final[str] = (
    "semantic_3d_chat.v96_embodied_runtime_access_log.v1"
)
RESET_SEED: Final[int] = 20260814
NAVIGATION_CHECKPOINT: Final[Path] = (
    PROJECT_ROOT / "data_gemma4/checkpoints/navigation_policy_v3"
)
_SCENE_ID: Final[re.Pattern[str]] = re.compile(r"scene_[0-9]{6}")
_SHA256: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{64}")
_RUNTIME_SOURCE_PATHS: Final[tuple[str, ...]] = runtime_source_paths()
_RUNTIME_CONFIG_PATHS: Final[tuple[str, ...]] = (
    "configs/runtime/embodied_live.yaml",
    "configs/runtime/embodied_v54.yaml",
    "configs/runtime/gemma4_v54.yaml",
    "configs/runtime/gemma4_v96_strict_multiscene.yaml",
)


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _absolute_nonsymlink(path: str | Path, *, purpose: str) -> Path:
    candidate = Path(path).expanduser()
    rooted = candidate if candidate.is_absolute() else PROJECT_ROOT / candidate
    absolute = Path(os.path.abspath(rooted))
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        if current.is_symlink():
            raise ValueError(f"{purpose} path contains a symbolic link: {current}")
    return absolute


def _strict_object(path: Path) -> dict[str, Any]:
    path = _absolute_nonsymlink(path, purpose="V96 runtime task input")
    if not path.is_file():
        raise FileNotFoundError(path)

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate V96 runtime task field: {key}")
            result[key] = value
        return result

    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=unique,
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"Non-finite JSON constant is forbidden: {token}")
        ),
    )
    if not isinstance(value, dict):
        raise TypeError("V96 runtime task input must be one JSON object")
    return value


def _validated_task_input(
    value: Mapping[str, Any],
    *,
    expected_sha256: str,
) -> tuple[str, list[dict[str, Any]]]:
    if _canonical_sha256(value) != expected_sha256:
        raise ValueError("V96 runtime task input differs from preregistration")
    scene_id = value.get("scene_id")
    tasks = value.get("tasks")
    if (
        set(value) != {"schema", "scene_id", "tasks"}
        or value.get("schema") != TASK_INPUT_SCHEMA
        or not isinstance(scene_id, str)
        or _SCENE_ID.fullmatch(scene_id) is None
        or scene_id not in RELEASE_SCENE_IDS
        or not isinstance(tasks, list)
        or len(tasks) != 6
    ):
        raise ValueError("V96 runtime task input schema changed")
    checked: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in tasks:
        if not isinstance(row, Mapping):
            raise TypeError("V96 runtime task row must be an object")
        task = dict(row)
        task_id = task.get("task_id")
        instruction = task.get("instruction")
        max_steps = task.get("max_steps")
        if (
            set(task) != {"scene_id", "task_id", "instruction", "max_steps"}
            or task.get("scene_id") != scene_id
            or not isinstance(task_id, str)
            or not task_id.startswith(f"{scene_id}:")
            or task_id in seen
            or not isinstance(instruction, str)
            or not instruction.strip()
            or len(instruction) > 1024
            or isinstance(max_steps, bool)
            or not isinstance(max_steps, int)
            or not 1 <= max_steps <= 128
        ):
            raise ValueError("V96 runtime task row changed")
        seen.add(task_id)
        checked.append(task)
    return scene_id, checked


def _source_inventory_sha256(relative_paths: Sequence[str]) -> str:
    if len(relative_paths) != len(set(relative_paths)):
        raise RuntimeError("V96 runtime source inventory contains duplicate paths")
    entries: list[dict[str, Any]] = []
    for relative in sorted(relative_paths):
        path = _absolute_nonsymlink(
            PROJECT_ROOT / relative,
            purpose="V96 runtime source inventory",
        )
        if not path.is_file():
            raise FileNotFoundError(path)
        entries.append(
            {
                "path": relative,
                "sha256": _file_sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return _canonical_sha256(entries)


def _runtime_audit() -> FileAccessAudit:
    roots = [
        PROJECT_ROOT / "data/oracle",
        PROJECT_ROOT / "data/qa",
        PROJECT_ROOT / "data/rendered",
        PROJECT_ROOT / "data_gemma4/oracle",
        PROJECT_ROOT / "data_gemma4/qa",
        PROJECT_ROOT / "data_gemma4/features",
        PROJECT_ROOT / "data_gemma4/training",
        PROJECT_ROOT / "reports/gemma4/predictions",
        PROJECT_ROOT / "reports/gemma4/questions",
        PROJECT_ROOT / "reports/gemma4/scorer_only",
        PROJECT_ROOT / "configs/experiments",
        PROJECT_ROOT
        / "src/semantic_3d_chat/evaluation/v96_embodied_heldout.py",
    ]
    return FileAccessAudit(
        roots,
        forbidden_component_names={"oracle", "qa", "scorer_only"},
        block_forbidden=True,
    )


def _access_log(paths: Sequence[str]) -> dict[str, Any]:
    loaded = sorted(set(paths))
    return {
        "schema": ACCESS_LOG_SCHEMA,
        "loaded_files": loaded,
        "loaded_file_count": len(loaded),
        "loaded_file_inventory_sha256": _canonical_sha256(loaded),
        "forbidden_accesses": [],
        "oracle_reads": 0,
        "qa_reads": 0,
        "training_reads": 0,
        "scorer_reads": 0,
        "block_forbidden": True,
    }


def _evidence(
    *,
    task_id: str,
    scene_id: str,
    navigation: Mapping[str, Any],
    args: argparse.Namespace,
    release_receipt_sha256: str,
    access_log: Mapping[str, Any],
) -> dict[str, Any]:
    navigation_value = dict(navigation)
    access_value = dict(access_log)
    value: dict[str, Any] = {
        "schema": EVIDENCE_SCHEMA,
        "task_id": task_id,
        "scene_id": scene_id,
        "navigation": navigation_value,
        "navigation_sha256": _canonical_sha256(navigation_value),
        "preregistration_sha256": args.preregistration_sha256,
        "release_receipt_sha256": release_receipt_sha256,
        "dependency_contract_sha256": args.dependency_contract_sha256,
        "runtime_config_inventory_sha256": args.runtime_config_inventory_sha256,
        "runtime_source_inventory_sha256": args.runtime_source_inventory_sha256,
        "implementation_source_inventory_sha256": (
            args.implementation_source_inventory_sha256
        ),
        "runtime_asset_contract_sha256": args.runtime_asset_contract_sha256,
        "runtime_task_input_sha256": args.expected_task_input_sha256,
        "runtime_access_log": access_value,
        "runtime_access_log_sha256": _canonical_sha256(access_value),
        "forbidden_runtime_reads": 0,
        "oracle_runtime_reads": 0,
        "environmental_text_inputs": [],
        "scorer_only_target_category_loaded": False,
        "runtime_result_closed": True,
    }
    value["evidence_identity_sha256"] = _canonical_sha256(value)
    return value


def _write_output_directory(
    destination: Path,
    evidence: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}.", dir=destination.parent
    ) as raw:
        stage = Path(raw)
        rows: list[dict[str, Any]] = []
        for value in evidence:
            task_id = str(value["task_id"])
            opaque = hashlib.sha256(task_id.encode("utf-8")).hexdigest()[:16]
            filename = f"r_{opaque}.json"
            path = stage / filename
            path.write_text(
                json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False)
                + "\n",
                encoding="utf-8",
            )
            rows.append(
                {
                    "task_id": task_id,
                    "file": filename,
                    "sha256": _file_sha256(path),
                }
            )
        manifest: dict[str, Any] = {
            "schema": "semantic_3d_chat.v96_embodied_runtime_result_manifest.v1",
            "scene_id": evidence[0]["scene_id"],
            "results": rows,
        }
        manifest["inventory_sha256"] = _canonical_sha256(rows)
        (stage / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(stage, destination)
    return manifest


async def run(args: argparse.Namespace) -> dict[str, Any]:
    for name in (
        "expected_task_input_sha256",
        "preregistration_sha256",
        "expected_release_receipt_sha256",
        "dependency_contract_sha256",
        "runtime_config_inventory_sha256",
        "runtime_source_inventory_sha256",
        "implementation_source_inventory_sha256",
        "runtime_asset_contract_sha256",
        "expected_runtime_asset_sha256",
        "expected_runtime_manifest_sha256",
    ):
        if _SHA256.fullmatch(str(getattr(args, name))) is None:
            raise ValueError(f"V96 runtime binding is not SHA-256: {name}")
    if (
        _source_inventory_sha256(_RUNTIME_SOURCE_PATHS)
        != args.runtime_source_inventory_sha256
    ):
        raise ValueError("V96 embodied runtime source changed after preregistration")
    if (
        _source_inventory_sha256(_RUNTIME_CONFIG_PATHS)
        != args.runtime_config_inventory_sha256
    ):
        raise ValueError("V96 embodied runtime config changed after preregistration")
    asset = _absolute_nonsymlink(
        args.runtime_asset,
        purpose="V96 sanitized runtime asset",
    )
    manifest_path = asset.with_suffix(".json")
    if (
        asset.is_symlink()
        or manifest_path.is_symlink()
        or _file_sha256(asset) != args.expected_runtime_asset_sha256
        or _file_sha256(manifest_path) != args.expected_runtime_manifest_sha256
    ):
        raise ValueError("V96 sanitized runtime asset differs from preregistration")
    task_path = _absolute_nonsymlink(
        args.task_input,
        purpose="V96 runtime task input",
    )
    persistent_map = _absolute_nonsymlink(
        args.persistent_map,
        purpose="V96 held-out persistent map",
    )
    scan_root = _absolute_nonsymlink(
        args.scan_output,
        purpose="V96 held-out scan output",
    )
    output = _absolute_nonsymlink(
        args.output,
        purpose="V96 held-out runtime output",
    )
    if persistent_map.exists() or persistent_map.is_symlink():
        raise FileExistsError("V96 held-out runtime requires an empty persistent-map path")
    audit = _runtime_audit()
    results: list[tuple[str, dict[str, Any]]] = []
    with audit:
        audit.record(task_path)
        audit.record(asset)
        audit.record(manifest_path)
        task_input = _strict_object(task_path)
        scene_id, tasks = _validated_task_input(
            task_input,
            expected_sha256=args.expected_task_input_sha256,
        )
        receipt = validate_promoted_v96_release_receipt(
            run_isolated_v96_release_verification()
        )
        receipt_sha256 = _canonical_sha256(receipt)
        if receipt_sha256 != args.expected_release_receipt_sha256:
            raise ValueError("V96 promoted release differs from preregistration")
        runtime = build_promoted_v96_embodied_runtime(
            scene_id,
            runtime_asset=asset,
            robot_state_checkpoint=DEFAULT_ROBOT_STATE_CHECKPOINT,
            persistent_map_path=persistent_map,
            scan_output_directory=scan_root,
            audit=audit,
            release_verifier=lambda: receipt,
            local_files_only=True,
        )
        server = build_server(runtime)
        for task in tasks:
            reset = await server.call_tool(
                "reset_scene", {"scene_id": scene_id, "seed": RESET_SEED}
            )
            if reset.is_error or not isinstance(reset.structured_content, Mapping):
                raise RuntimeError("V96 held-out reset failed through MCP")
            reset_receipt = validate_numeric_tool_receipt(
                reset.structured_content,
                require_continuous_binding=True,
            )
            if (
                reset_receipt["success"] is not True
                or reset_receipt["map_version"] != 0
                or reset_receipt["action_count"] != 0
            ):
                raise RuntimeError("V96 held-out reset did not isolate the task")
            scanner = runtime.simulator.scanner
            scanner.output_directory = scan_root / hashlib.sha256(
                str(task["task_id"]).encode("utf-8")
            ).hexdigest()[:16]
            backend = build_v96_release_action_backend(
                runtime,
                runtime.config,
                navigation_checkpoint=NAVIGATION_CHECKPOINT,
                audit=audit,
            )
            agent = V96CoResidentMCPAgent(
                runtime,
                backend,
                runtime.config,
                server=server,
            )
            result = await agent.run_instruction(
                str(task["instruction"]),
                max_steps=int(task["max_steps"]),
            )
            results.append((str(task["task_id"]), result.as_dict()))
    audit.assert_clean()
    log = _access_log(audit.unique_paths)
    evidence = [
        _evidence(
            task_id=task_id,
            scene_id=scene_id,
            navigation=navigation,
            args=args,
            release_receipt_sha256=receipt_sha256,
            access_log=log,
        )
        for task_id, navigation in results
    ]
    manifest = _write_output_directory(output, evidence)
    return {
        "phase": "v96_embodied_scene_runtime_complete",
        "scene_id": scene_id,
        "task_count": len(evidence),
        "output": str(output),
        "inventory_sha256": manifest["inventory_sha256"],
        "forbidden_runtime_reads": 0,
        "oracle_runtime_reads": 0,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--task-input", required=True)
    result.add_argument("--expected-task-input-sha256", required=True)
    result.add_argument("--preregistration-sha256", required=True)
    result.add_argument("--expected-release-receipt-sha256", required=True)
    result.add_argument("--dependency-contract-sha256", required=True)
    result.add_argument("--runtime-config-inventory-sha256", required=True)
    result.add_argument("--runtime-source-inventory-sha256", required=True)
    result.add_argument("--implementation-source-inventory-sha256", required=True)
    result.add_argument("--runtime-asset-contract-sha256", required=True)
    result.add_argument("--runtime-asset", required=True)
    result.add_argument("--expected-runtime-asset-sha256", required=True)
    result.add_argument("--expected-runtime-manifest-sha256", required=True)
    result.add_argument("--persistent-map", required=True)
    result.add_argument("--scan-output", required=True)
    result.add_argument("--output", required=True)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    payload = asyncio.run(run(args))
    print(json.dumps(payload, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["ACCESS_LOG_SCHEMA", "EVIDENCE_SCHEMA", "RESET_SEED", "main", "run"]
