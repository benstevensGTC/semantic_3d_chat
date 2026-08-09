"""Interactive and one-shot local chat over a precomputed continuous 3D prefix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from semantic_3d_chat.chat.file_audit import FileAccessAudit

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", default="configs/default.yaml")
    result.add_argument("--scene", default="scene_000001")
    result.add_argument("--checkpoint")
    result.add_argument(
        "--question",
        action="append",
        help="Ask one question noninteractively; repeat the flag for multiple questions.",
    )
    result.add_argument(
        "--audit-log",
        default=None,
        help="JSON record of every path opened while loading and querying the runtime.",
    )
    result.add_argument(
        "--chat-log",
        default=None,
        help="Append answers, numeric grounding, confidence, and prefix hashes here.",
    )
    return result


def _rooted(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    return candidate.resolve() if candidate.is_absolute() else (PROJECT_ROOT / candidate).resolve()


def _forbidden_roots(data_root: Path) -> list[Path]:
    return [data_root / name for name in ("oracle", "qa", "rendered", "features")]


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()


def _emit_answer(runtime: Any, question: str, chat_log: Path, *, interactive: bool) -> None:
    result = runtime.answer(question)
    payload = {"phase": "answer", "scene_id": runtime.scene_id, **result.to_dict()}
    _append_jsonl(chat_log, payload)
    if interactive:
        print(f"Assistant> {result.answer}")
        print(
            "Grounding> "
            + json.dumps(
                {
                    "xyz_m": list(result.grounding_xyz_m),
                    "confidence": result.grounding_confidence,
                    "support_distance_m": result.grounding_support_distance_m,
                    "prefix_hash": result.prefix_hash,
                },
                sort_keys=True,
            )
        )
    else:
        print(json.dumps(payload, sort_keys=True, allow_nan=False), flush=True)


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    config_path = _rooted(args.config)
    audit_path = _rooted(args.audit_log or "reports/metrics/chat_file_access.json")
    chat_log = _rooted(args.chat_log or "reports/examples/sample_chats.jsonl")
    default_data_root = PROJECT_ROOT / "data"
    audit = FileAccessAudit(_forbidden_roots(default_data_root))
    completed = False
    try:
        with audit:
            # Import after activating the process-wide hook so runtime module and
            # model/config reads are represented in the access log.
            from semantic_3d_chat.chat.runtime import StaticChatRuntime
            from semantic_3d_chat.config import (
                artifact_root,
                default_checkpoint_path,
                load_config,
                reports_root,
            )

            audit.record(config_path)
            config = load_config(config_path)
            configured_reports = reports_root(config)
            if args.audit_log is None:
                audit_path = configured_reports / "metrics" / "chat_file_access.json"
            if args.chat_log is None:
                chat_log = configured_reports / "examples" / "sample_chats.jsonl"
            configured_forbidden_roots = [
                artifact_root(config, kind).resolve()
                for kind in ("oracle", "qa", "rendered", "features")
            ]
            for root in configured_forbidden_roots:
                if root not in audit.forbidden_roots:
                    audit.forbidden_roots.append(root)
            runtime = StaticChatRuntime.load(
                config,
                args.scene,
                checkpoint=args.checkpoint or default_checkpoint_path(config),
                audit=audit,
                local_files_only=True,
            )
            print(
                json.dumps(runtime.startup_summary(), sort_keys=True, allow_nan=False), flush=True
            )
            if args.question:
                for question in args.question:
                    _emit_answer(runtime, question, chat_log, interactive=False)
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
                    if not question:
                        continue
                    _emit_answer(runtime, question, chat_log, interactive=True)
            runtime.assert_prefix_unchanged()
            completed = True
    finally:
        audit.save(audit_path)
    audit.assert_clean()
    if completed:
        print(
            json.dumps(
                {
                    "phase": "audit_complete",
                    "passed": True,
                    "loaded_file_count": len(audit.unique_paths),
                    "audit_log": str(audit_path),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return 0


# Retain compatibility with console scripts generated before the entry point was
# corrected from ``cli:app`` to ``cli:main`` in pyproject.toml.
app = main


if __name__ == "__main__":
    raise SystemExit(main())
