"""Audited local chat using the exact direct 738-token V83 scene memory."""

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
    value = Path(path).expanduser()
    rooted = value if value.is_absolute() else PROJECT_ROOT / value
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
    parser.add_argument("--scene-memory", required=True)
    parser.add_argument("--question", action="append")
    parser.add_argument("--audit-log")
    parser.add_argument("--chat-log")
    return parser


def _run(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    compiler_roots = [
        PROJECT_ROOT
        / "data_gemma4"
        / "runtime"
        / "checkpoints"
        / "gemma4_v75_nll_control_release_v1",
        PROJECT_ROOT
        / "reports"
        / "gemma4"
        / "artifacts"
        / "v75_fixed_atlas_historical_internal_v1"
        / "probe_bank",
        PROJECT_ROOT / "data_gemma4" / "training",
        PROJECT_ROOT / "reports" / "gemma4" / "scorer_only",
    ]
    default_data = PROJECT_ROOT / "data"
    audit = FileAccessAudit(
        [
            *(default_data / name for name in ("oracle", "qa", "rendered", "features")),
            *compiler_roots,
        ],
        forbidden_component_names=frozenset(
            {"oracle", "qa", "rendered", "features", "scorer"}
        ),
        block_forbidden=True,
    )
    audit_path = _rooted(
        args.audit_log or f"reports/gemma4/metrics/v83_chat_access_{args.scene}.json"
    )
    chat_path = _rooted(
        args.chat_log or f"reports/gemma4/examples/v83_chat_{args.scene}.jsonl"
    )
    completed = False
    try:
        with audit:
            from semantic_3d_chat.chat.runtime_config import load_runtime_config
            from semantic_3d_chat.chat.v83_direct_scene_memory_runtime import (
                V83DirectSceneMemoryChatRuntime,
            )

            config = load_runtime_config(args.config, record_file=audit.record)
            for kind in ("oracle", "qa", "rendered", "features"):
                root = artifact_root(config, kind).resolve()
                if root not in audit.forbidden_roots:
                    audit.forbidden_roots.append(root)
            if args.audit_log is None:
                audit_path = (
                    reports_root(config)
                    / "metrics"
                    / f"v83_chat_access_{args.scene}.json"
                )
            if args.chat_log is None:
                chat_path = (
                    reports_root(config)
                    / "examples"
                    / f"v83_chat_{args.scene}.jsonl"
                )
            runtime = V83DirectSceneMemoryChatRuntime.load(
                config,
                args.scene,
                base_checkpoint=_rooted(args.base_checkpoint),
                scene_memory=_rooted(args.scene_memory),
                audit=audit,
                local_files_only=True,
            )
            startup = runtime.startup_summary()
            if runtime.questions_answered != 0:
                raise RuntimeError("V83 fixed memory was not bound before question input")
            print(json.dumps(startup, sort_keys=True, allow_nan=False), flush=True)

            def answer(question: str) -> None:
                result = runtime.answer(question)
                payload = {
                    "phase": "answer",
                    "scene_id": args.scene,
                    **result.to_dict(),
                    "fixed_scene_memory_sha256": runtime.scene_prefix_hash,
                    "reader_audit": None,
                    "grounding_audit": None,
                    "prepared_layout_audit": runtime.last_prepared_layout_audit,
                }
                _append_jsonl(chat_path, payload)
                print(json.dumps(payload, sort_keys=True, allow_nan=False), flush=True)

            if args.question:
                for question in args.question:
                    answer(question)
            else:
                print("Ask about the embedded room. Type 'quit' or press Ctrl-D to exit.")
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
                    "phase": "v83_chat_audit_complete",
                    "passed": True,
                    "fixed_memory_invariant": True,
                    "exact_738_token_memory_supplied_directly_to_gemma": True,
                    "question_derived_environmental_tokens": 0,
                    "compiler_or_probe_bank_loaded": False,
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
        print(f"V83 chat refused: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
