"""Audited local chat over an immutable scene-only continuous prefix atlas."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from semantic_3d_chat.chat.file_audit import FileAccessAudit
from semantic_3d_chat.config import PROJECT_ROOT, artifact_root, reports_root


def _rooted(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    rooted = candidate if candidate.is_absolute() else PROJECT_ROOT / candidate
    return Path(os.path.abspath(rooted))


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--atlas-checkpoint", required=True)
    parser.add_argument("--question", action="append")
    parser.add_argument("--audit-log")
    parser.add_argument("--chat-log")
    return parser


def _run(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    default_data = PROJECT_ROOT / "data"
    audit = FileAccessAudit(
        [
            *(default_data / name for name in ("oracle", "qa", "rendered", "features")),
            PROJECT_ROOT / "data_gemma4" / "training",
            PROJECT_ROOT / "reports" / "gemma4" / "scorer_only",
        ],
        forbidden_component_names=frozenset({"oracle", "qa", "rendered", "features"}),
        block_forbidden=True,
    )
    audit_path = _rooted(
        args.audit_log or "reports/gemma4/metrics/fixed_prefix_chat_access.json"
    )
    chat_path = _rooted(
        args.chat_log or "reports/gemma4/examples/fixed_prefix_chat.jsonl"
    )
    completed = False
    try:
        with audit:
            from semantic_3d_chat.chat.fixed_prefix_runtime import (
                FixedPrefixAtlasChatRuntime,
            )
            from semantic_3d_chat.chat.runtime_config import load_runtime_config

            config = load_runtime_config(args.config, record_file=audit.record)
            for kind in ("oracle", "qa", "rendered", "features"):
                root = artifact_root(config, kind).resolve()
                if root not in audit.forbidden_roots:
                    audit.forbidden_roots.append(root)
            training_root = artifact_root(config, "checkpoints").resolve().parent / "training"
            if training_root not in audit.forbidden_roots:
                audit.forbidden_roots.append(training_root)
            if args.audit_log is None:
                audit_path = reports_root(config) / "metrics" / "fixed_prefix_chat_access.json"
            if args.chat_log is None:
                chat_path = reports_root(config) / "examples" / "fixed_prefix_chat.jsonl"
            runtime = FixedPrefixAtlasChatRuntime.load(
                config,
                args.scene,
                base_checkpoint=_rooted(args.base_checkpoint),
                atlas_checkpoint=_rooted(args.atlas_checkpoint),
                audit=audit,
            )
            startup = runtime.startup_summary()
            if runtime.questions_answered != 0:
                raise RuntimeError("Fixed prefix was not finalized before question input")
            print(json.dumps(startup, sort_keys=True, allow_nan=False), flush=True)

            def answer(question: str) -> None:
                result = runtime.answer(question)
                payload = {"phase": "answer", "scene_id": args.scene, **result.to_dict()}
                _append_jsonl(chat_path, payload)
                print(json.dumps(payload, sort_keys=True, allow_nan=False), flush=True)

            if args.question:
                for question in args.question:
                    answer(question)
            else:
                print("Ask about the fixed continuous room memory. Type 'quit' to exit.")
                while True:
                    try:
                        question = input("You> ").strip()
                    except EOFError:
                        print()
                        break
                    if question.casefold() in {"quit", "exit"}:
                        break
                    if question:
                        answer(question)
            runtime.assert_prefix_unchanged()
            completed = True
    finally:
        audit.save(audit_path)
    audit.assert_clean()
    if completed:
        print(
            json.dumps(
                {
                    "phase": "fixed_prefix_audit_complete",
                    "passed": True,
                    "prefix_invariant": True,
                    "forbidden_access_count": len(audit.forbidden_accesses()),
                    "loaded_file_count": len(audit.unique_paths),
                    "audit_log": str(audit_path),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        return _run(argv)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"Fixed-prefix chat refused: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
