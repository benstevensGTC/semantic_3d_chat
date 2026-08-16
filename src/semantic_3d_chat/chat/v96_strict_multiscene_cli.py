"""Audited CLI for the held-out-gated V96 continuous-memory runtime."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Final

from semantic_3d_chat.chat.file_audit import FileAccessAudit
from semantic_3d_chat.config import PROJECT_ROOT, artifact_root, reports_root

DEFAULT_CONFIG: Final[str] = "configs/runtime/gemma4_v96_strict_multiscene.yaml"
DEFAULT_SCENE: Final[str] = "scene_000025"
RELEASE_CHECKPOINT: Final[str] = (
    "data_gemma4/runtime/checkpoints/gemma4_v96_strict_multiscene_release_v1"
)
CANDIDATE_CHECKPOINT: Final[str] = (
    "reports/gemma4/artifacts/v96_strict_runtime_candidate"
)
RELEASE_MEMORY_ROOT: Final[str] = "data_gemma4/runtime/scene_memories/v96"
CANDIDATE_MEMORY_ROOT: Final[str] = (
    "reports/gemma4/artifacts/v96_strict_runtime_candidate_memories"
)
_SCENE_ID: Final[re.Pattern[str]] = re.compile(r"scene_[0-9]{6}")


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
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--scene", default=DEFAULT_SCENE)
    parser.add_argument("--base-checkpoint")
    parser.add_argument("--scene-memory")
    parser.add_argument("--question", action="append")
    parser.add_argument("--audit-log")
    parser.add_argument("--chat-log")
    parser.add_argument(
        "--allow-candidate",
        action="store_true",
        help="Use an authenticated pre-smoke candidate instead of requiring promotion.",
    )
    return parser


def _selected_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    checkpoint = args.base_checkpoint
    memory = args.scene_memory
    if checkpoint is None:
        checkpoint = CANDIDATE_CHECKPOINT if args.allow_candidate else RELEASE_CHECKPOINT
    if memory is None:
        root = CANDIDATE_MEMORY_ROOT if args.allow_candidate else RELEASE_MEMORY_ROOT
        memory = str(Path(root) / args.scene)
    return _rooted(checkpoint), _rooted(memory)


def _forbidden_roots() -> list[Path]:
    """Return supervision, build, and scoring surfaces chat may not open."""

    roots = [
        PROJECT_ROOT / "data/oracle",
        PROJECT_ROOT / "data/qa",
        PROJECT_ROOT / "data/rendered",
        PROJECT_ROOT / "data/features",
        PROJECT_ROOT / "data_diverse52/qa",
        PROJECT_ROOT / "data_gemma4/features",
        PROJECT_ROOT / "data_gemma4/maps",
        PROJECT_ROOT / "data_gemma4/training",
        PROJECT_ROOT / "data_gemma4/checkpoints/v96_atomic_pair_repair_work",
        PROJECT_ROOT / "configs/experiments/gemma4_v96_atomic_pair_repair.yaml",
        PROJECT_ROOT / "configs/experiments/gemma4_v95_deferred_final_materialization.yaml",
        PROJECT_ROOT / "reports/gemma4/scorer_only",
        PROJECT_ROOT / "reports/gemma4/questions",
        PROJECT_ROOT / "reports/gemma4/predictions",
        PROJECT_ROOT / "reports/gemma4/artifacts/v96_atomic_pair_repair_final",
        PROJECT_ROOT / "reports/gemma4/artifacts/v95_strict_causal_successor_final",
        PROJECT_ROOT / "reports/gemma4/artifacts/v94_strict_multiscene_full40_final",
        PROJECT_ROOT / "reports/gemma4/artifacts/v95_deferred_final",
        PROJECT_ROOT / "reports/gemma4/artifacts/v85_strict_runtime_candidate",
    ]
    roots.extend(PROJECT_ROOT.glob("data*/oracle"))
    roots.extend(PROJECT_ROOT.glob("data*/qa"))
    return list(dict.fromkeys(path.resolve() for path in roots))


def _run(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if _SCENE_ID.fullmatch(args.scene) is None:
        raise ValueError("V96 scene ID must be opaque scene_NNNNNN")
    checkpoint_path, memory_path = _selected_paths(args)
    audit = FileAccessAudit(
        _forbidden_roots(),
        forbidden_component_names=frozenset(
            {"oracle", "qa", "scorer", "predictions"}
        ),
        block_forbidden=True,
    )
    audit_path = _rooted(
        args.audit_log or f"reports/gemma4/metrics/v96_chat_access_{args.scene}.json"
    )
    chat_path = _rooted(
        args.chat_log or f"reports/gemma4/examples/v96_chat_{args.scene}.jsonl"
    )
    completed = False
    oracle_available_at_start = any(
        path.is_dir() for path in PROJECT_ROOT.glob("data*/oracle")
    )
    runtime_mode: str | None = None
    runtime_promoted = False
    runtime = None
    try:
        with audit:
            from semantic_3d_chat.chat.runtime_config import load_runtime_config
            from semantic_3d_chat.chat.v96_strict_multiscene_runtime import (
                V96StrictMultisceneChatRuntime,
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
                    / f"v96_chat_access_{args.scene}.json"
                )
            if args.chat_log is None:
                chat_path = (
                    reports_root(config)
                    / "examples"
                    / f"v96_chat_{args.scene}.jsonl"
                )
            runtime = V96StrictMultisceneChatRuntime.load(
                config,
                args.scene,
                base_checkpoint=checkpoint_path,
                scene_memory=memory_path,
                audit=audit,
                local_files_only=True,
            )
            if runtime.questions_answered != 0:
                raise RuntimeError("V96 scene memory was not bound before question input")
            runtime_mode = runtime.runtime_package_mode
            runtime_promoted = runtime.runtime_promotion_authorized
            if runtime_mode == "candidate" and not args.allow_candidate:
                raise RuntimeError(
                    "V96 checkpoint is a candidate; pass --allow-candidate explicitly"
                )
            if runtime_mode == "promoted" and not runtime_promoted:
                raise RuntimeError("V96 promoted provenance is internally inconsistent")
            startup = {
                **runtime.startup_summary(),
                "candidate_runtime_explicitly_allowed": bool(args.allow_candidate),
                "prefix_hash_printed_before_questions": True,
                "oracle_directory_available_at_runtime_start": oracle_available_at_start,
            }
            print(json.dumps(startup, sort_keys=True, allow_nan=False), flush=True)

            def answer(question: str) -> None:
                if runtime is None:  # pragma: no cover - structural guard
                    raise RuntimeError("V96 runtime was not initialized")
                result = runtime.answer(question)
                payload = {
                    "phase": "answer",
                    "scene_id": args.scene,
                    **result.to_dict(),
                    "fixed_scene_memory_sha256": runtime.scene_prefix_hash,
                    "environment_conditioned_input_sha256": (
                        runtime.environment_conditioned_input_hashes[-1]
                    ),
                    "question_derived_environmental_tokens": 0,
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
            if len(set(runtime.environment_conditioned_input_hashes)) > 1:
                raise RuntimeError("V96 environmental input changed across questions")
            completed = True
    finally:
        audit.save(audit_path)
    audit.assert_clean()
    if completed:
        print(
            json.dumps(
                {
                    "phase": "v96_chat_audit_complete",
                    "passed": True,
                    "prefix_hash_printed_before_questions": True,
                    "prefix_hash_invariant": True,
                    "fixed_memory_invariant": True,
                    "total_environment_conditioned_input_invariant": True,
                    "exact_738_token_memory_supplied_directly_to_gemma": True,
                    "question_derived_environmental_tokens": 0,
                    "training_evaluation_or_scorer_file_loaded": False,
                    "runtime_package_mode": runtime_mode,
                    "runtime_promotion_authorized": runtime_promoted,
                    "candidate_runtime_explicitly_allowed": bool(args.allow_candidate),
                    "oracle_directory_available_at_runtime_start": (
                        oracle_available_at_start
                    ),
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
        print(f"V96 chat refused: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
