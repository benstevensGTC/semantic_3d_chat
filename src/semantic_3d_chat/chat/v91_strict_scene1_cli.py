"""Audited local CLI for the promoted V91 continuous-memory scene-one runtime."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Final

from semantic_3d_chat.chat.file_audit import FileAccessAudit
from semantic_3d_chat.config import PROJECT_ROOT, artifact_root, reports_root

DEFAULT_CONFIG: Final[str] = "configs/runtime/gemma4_v91_strict_scene1.yaml"
DEFAULT_CHECKPOINT: Final[str] = (
    "data_gemma4/runtime/checkpoints/gemma4_v91_strict_scene1_release_v1"
)
DEFAULT_SCENE_MEMORY: Final[str] = (
    "data_gemma4/runtime/scene_memories/v91/scene_000001"
)
_V91_OFFLINE_PATHS: Final[tuple[str, ...]] = (
    "configs/experiments/gemma4_v90_scene1_conversational.yaml",
    "reports/gemma4/artifacts/v90_scene1_conversational_final",
    "reports/gemma4/predictions/gemma4_v90_scene1_conversational_evaluation.json",
    "reports/gemma4/metrics/gemma4_v90_scene1_conversational_training.json",
    "reports/gemma4/metrics/gemma4_v90_scene1_conversational_evaluation.json",
    "configs/experiments/gemma4_v91_scene1_conversational_repair.yaml",
    "data_gemma4/checkpoints/v91_scene1_conversational_repair_work_v2",
    "reports/gemma4/artifacts/v91_scene1_conversational_repair_final_v2",
    "reports/gemma4/predictions/gemma4_v91_scene1_conversational_repair_evaluation_v2.json",
    "reports/gemma4/metrics/gemma4_v91_scene1_conversational_repair_preregistration_v2.json",
    "reports/gemma4/metrics/gemma4_v91_scene1_conversational_repair_cpu_preflight_v2.json",
    "reports/gemma4/metrics/gemma4_v91_scene1_conversational_repair_training_v2.json",
    "reports/gemma4/metrics/gemma4_v91_scene1_conversational_repair_evaluation_v2.json",
)


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
    parser.add_argument("--scene", default="scene_000001")
    parser.add_argument("--base-checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--scene-memory", default=DEFAULT_SCENE_MEMORY)
    parser.add_argument("--question", action="append")
    parser.add_argument("--audit-log")
    parser.add_argument("--chat-log")
    return parser


def _forbidden_roots() -> list[Path]:
    """Offline supervision/evaluation surfaces the chat child may not open."""

    inherited = (
        "data/oracle",
        "data/qa",
        "data/rendered",
        "data/features",
        "data_diverse52/qa",
        "data_gemma4/training",
        "reports/gemma4/scorer_only",
        "reports/gemma4/artifacts/v89_scene1_retention_final",
        "reports/gemma4/artifacts/v88_scene1_augmented_final",
        "reports/gemma4/artifacts/v87_scene1_balanced_final",
        "reports/gemma4/artifacts/v86_scene1_demo_final",
        "reports/gemma4/artifacts/v85_strict_runtime_candidate",
        "reports/gemma4/artifacts/v85_strict_multiscene_final",
        "data_gemma4/runtime/checkpoints/gemma4_v75_nll_control_release_v1",
    )
    return [PROJECT_ROOT / path for path in (*inherited, *_V91_OFFLINE_PATHS)]


def _run(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.scene != "scene_000001":
        raise ValueError("V91 strict scene-one chat accepts only scene_000001")
    audit = FileAccessAudit(
        _forbidden_roots(),
        forbidden_component_names=frozenset(
            {"oracle", "qa", "rendered", "features", "scorer", "predictions"}
        ),
        block_forbidden=True,
    )
    audit_path = _rooted(
        args.audit_log or f"reports/gemma4/metrics/v91_chat_access_{args.scene}.json"
    )
    chat_path = _rooted(
        args.chat_log or f"reports/gemma4/examples/v91_chat_{args.scene}.jsonl"
    )
    completed = False
    oracle_available_at_start = (PROJECT_ROOT / "data/oracle").exists()
    try:
        with audit:
            from semantic_3d_chat.chat.runtime_config import load_runtime_config
            from semantic_3d_chat.chat.v91_strict_scene1_runtime import (
                V91StrictScene1ChatRuntime,
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
                    / f"v91_chat_access_{args.scene}.json"
                )
            if args.chat_log is None:
                chat_path = (
                    reports_root(config)
                    / "examples"
                    / f"v91_chat_{args.scene}.jsonl"
                )
            runtime = V91StrictScene1ChatRuntime.load(
                config,
                args.scene,
                base_checkpoint=_rooted(args.base_checkpoint),
                scene_memory=_rooted(args.scene_memory),
                audit=audit,
                local_files_only=True,
            )
            if runtime.questions_answered != 0:
                raise RuntimeError("V91 scene memory was not bound before question input")
            if not runtime.runtime_promotion_authorized:
                raise RuntimeError("V91 checkpoint is not the promoted strict release")
            startup = {
                **runtime.startup_summary(),
                "oracle_directory_available_at_runtime_start": oracle_available_at_start,
            }
            print(json.dumps(startup, sort_keys=True, allow_nan=False), flush=True)

            def answer(question: str) -> None:
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
                raise RuntimeError("V91 total environmental input changed across questions")
            completed = True
    finally:
        audit.save(audit_path)
    audit.assert_clean()
    if completed:
        print(
            json.dumps(
                {
                    "phase": "v91_chat_audit_complete",
                    "passed": True,
                    "prefix_hash_printed_before_questions": True,
                    "fixed_memory_invariant": True,
                    "total_environment_conditioned_input_invariant": True,
                    "exact_738_token_memory_supplied_directly_to_gemma": True,
                    "question_derived_environmental_tokens": 0,
                    "compiler_or_probe_bank_loaded": False,
                    "training_or_evaluation_report_loaded": False,
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
        print(f"V91 chat refused: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
