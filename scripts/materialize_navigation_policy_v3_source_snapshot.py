#!/usr/bin/env python3
"""Materialize exact shared-source bytes bound by the historical V3 run."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Final

from semantic_3d_chat.config import PROJECT_ROOT

DESTINATION: Final[Path] = (
    PROJECT_ROOT / "reports/gemma4/evidence/navigation_policy_v3_sources"
)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _reverse_inference_v4(source: str) -> str:
    result = source.replace(
        "from semantic_3d_chat.robot.navigation_policy_v4 import (\n"
        "    SemanticClearanceActionBackendV4,\n"
        "    load_navigation_policy_v4_checkpoint,\n"
        ")\n",
        "",
    ).replace("choices=(1, 3, 4),", "choices=(1, 3),")
    start = result.index("        if navigation_policy_version == 4:\n")
    end = result.index("        elif navigation_policy_version == 3:\n", start)
    result = result[:start] + result[end:]
    result = result.replace(
        "        elif navigation_policy_version == 3:\n",
        "        if navigation_policy_version == 3:\n",
        1,
    ).replace(
        "        if navigation_policy_version in {3, 4}:\n",
        "        if navigation_policy_version == 3:\n",
    )
    start = result.index(
        "        if navigation_policy_version == 4:\n",
        result.index("def _run_contract"),
    )
    end = result.index("    return result\n", start)
    result = result[:start] + result[end:]
    start = result.index("                if args.navigation_policy_version == 4:\n")
    end = result.index(
        "                elif args.navigation_policy_version == 3:\n", start
    )
    result = result[:start] + result[end:]
    return result.replace(
        "                elif args.navigation_policy_version == 3:\n",
        "                if args.navigation_policy_version == 3:\n",
        1,
    )


def _reverse_tool_policy_v4(source: str) -> str:
    return source.replace(
        '            "supervised_continuous_semantic_clearance_navigation_policy_v4",\n',
        "",
    )


def _reverse_benchmark_v4(source: str) -> str:
    result = source.replace(
        '        "supervised_continuous_semantic_clearance_navigation_policy_v4",\n',
        "",
    ).replace(
        "        if status in {\n"
        '            "supervised_continuous_semantic_grounded_navigation_policy_v3",\n'
        "            } and (\n",
        "        if status == "
        '"supervised_continuous_semantic_grounded_navigation_policy_v3" and (\n',
    )
    start = result.index(
        '        if status == "supervised_continuous_semantic_clearance_navigation_policy_v4" and (\n'
    )
    terminal = (
        '            raise ValueError("V4 navigation journal lacks clearance-safety provenance")\n'
    )
    end = result.index(terminal, start) + len(terminal)
    return result[:start] + result[end:]


_SOURCES: Final[
    tuple[tuple[str, str, str, Callable[[str], str]], ...]
] = (
    (
        "inference_cli",
        "scripts/run_llm_navigation_inference.py",
        "df19394a1add0ace5dd4aa542989233ad2dea0ecdde1c115819828a71f44a31c",
        _reverse_inference_v4,
    ),
    (
        "tool_policy",
        "src/semantic_3d_chat/robot/llm_tool_policy.py",
        "93100da22d93124a8928285bfba90f67dd1dfc38094de1d18ead9ae688140e25",
        _reverse_tool_policy_v4,
    ),
    (
        "benchmark",
        "src/semantic_3d_chat/evaluation/llm_navigation_benchmark.py",
        "0fcd836c56d15a96d905c8da078c1cf56dd2959eb94d1e0684cddd5ad2f20fa4",
        _reverse_benchmark_v4,
    ),
)


def _create_once(path: Path, payload: bytes) -> None:
    if path.exists():
        raise FileExistsError(f"Historical source snapshot already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    if DESTINATION.exists():
        raise FileExistsError(f"Historical V3 source tree already exists: {DESTINATION}")
    rows: dict[str, dict[str, str]] = {}
    materialized: list[tuple[Path, bytes]] = []
    for name, relative, expected, transform in _SOURCES:
        source = PROJECT_ROOT / relative
        reconstructed = transform(source.read_text(encoding="utf-8")).encode("utf-8")
        observed = _sha256(reconstructed)
        if observed != expected:
            raise ValueError(
                f"Historical V3 reconstruction differs for {name}: {observed}"
            )
        destination = DESTINATION / Path(relative).name
        materialized.append((destination, reconstructed))
        rows[name] = {
            "historical_path": str(destination.relative_to(PROJECT_ROOT)),
            "sealed_sha256": expected,
            "current_successor_path": relative,
            "current_successor_sha256": _sha256(source.read_bytes()),
        }
    manifest = {
        "schema": "semantic_3d_chat.navigation_policy_v3_source_snapshot.v1",
        "status": "historical_shared_source_bytes_materialized",
        "scope": "sealed_v3_run_only",
        "current_runtime_source_claimed": False,
        "historical_journal_sha256": (
            "865e829fdcd6cf0bd0bb05c7f18f30fa57269d139649abe00104d0d983c55aa6"
        ),
        "sources": rows,
    }
    for path, payload in materialized:
        _create_once(path, payload)
    _create_once(
        DESTINATION / "manifest.json",
        (
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8"),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
