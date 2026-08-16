"""Audited local chat over one strictly invariant continuous 3D scene prefix.

This research launcher deliberately accepts an explicit, unpromoted checkpoint
so a failed-but-runnable fixed-prefix baseline remains inspectable.  It never
adds question-conditioned environmental tokens: the exact complete prefix is
built once before any question and reused byte-identically for every turn.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from semantic_3d_chat.chat.file_audit import FileAccessAudit
from semantic_3d_chat.config import PROJECT_ROOT, artifact_root, reports_root


def _rooted(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (PROJECT_ROOT / value).resolve()


def _parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", required=True)
    result.add_argument("--scene", required=True)
    result.add_argument("--checkpoint", required=True)
    result.add_argument("--question", action="append")
    result.add_argument("--audit-log")
    result.add_argument("--chat-log")
    output = result.add_mutually_exclusive_group()
    output.add_argument(
        "--human",
        action="store_true",
        help="Print concise human-readable status and answers",
    )
    output.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Print JSON lines (the default for finite --question runs)",
    )
    result.add_argument(
        "--replace-chat-log",
        action="store_true",
        help="Replace the selected transcript before a finite --question run",
    )
    return result


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()


def _human_output_requested(args: argparse.Namespace) -> bool:
    """Use human output explicitly, or by default for an interactive session."""

    return bool(args.human or (not args.question and not args.json_output))


def _human_startup(startup: dict[str, Any]) -> None:
    shape = startup.get("prefix_shape")
    if isinstance(shape, list) and len(shape) >= 2:
        token_count = shape[-2]
        hidden_dim = shape[-1]
        memory = f"{token_count} tokens x {hidden_dim} dimensions"
    else:
        memory = "continuous tokens"
    backend = startup.get("language_backend", "local Gemma")
    device = startup.get("device", "local device")
    print("Semantic 3D Chat ready")
    print(f"  Scene: {startup['scene_id']}")
    print(f"  Continuous memory: {memory}")
    print(
        "  Fixed prefix: "
        f"{startup['environment_conditioned_input_sha256']} "
        "(built before questions)"
    )
    print(f"  Inference: {backend} on {device}")
    print("  Status: development checkpoint; research acceptance gate not passed")


def _human_answer(payload: dict[str, Any], *, echo_question: bool) -> None:
    if echo_question:
        print(f"\nYou> {payload['question']}")
    print(f"Assistant> {payload['answer']}")
    xyz = payload.get("grounding_xyz_m")
    confidence = payload.get("grounding_confidence")
    if (
        isinstance(xyz, list)
        and len(xyz) == 3
        and all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in xyz)
    ):
        grounding = f"({float(xyz[0]):+.2f}, {float(xyz[1]):+.2f}, {float(xyz[2]):+.2f}) m"
        if isinstance(confidence, (int, float)) and not isinstance(confidence, bool):
            grounding += f", confidence {float(confidence):.2f}"
        print(f"  Grounding: {grounding}")
    generated = payload.get("generated_tokens")
    elapsed = payload.get("elapsed_seconds")
    if (
        isinstance(generated, int)
        and not isinstance(generated, bool)
        and isinstance(elapsed, (int, float))
        and not isinstance(elapsed, bool)
    ):
        print(f"  Generation: {generated} tokens in {float(elapsed):.2f} s")


def _human_audit(payload: dict[str, Any]) -> None:
    print(
        "\nVerification: PASS - fixed prefix unchanged; "
        f"{payload['forbidden_access_count']} forbidden file reads"
    )
    print(
        f"  Audited {payload['loaded_file_count']} unique files; "
        f"log: {payload['audit_log']}"
    )


def _emit(
    runtime: Any,
    scene_id: str,
    question: str,
    chat_log: Path,
    *,
    human_output: bool,
    echo_question: bool,
) -> None:
    answer = runtime.answer(question)
    if answer.prefix_hash != runtime.scene_prefix_hash:
        raise RuntimeError("Answer used a different environmental prefix")
    payload = {
        "phase": "answer",
        "scene_id": scene_id,
        "strict_fixed_environment_embedding_input": True,
        "environment_conditioned_input_sha256": runtime.scene_prefix_hash,
        **answer.to_dict(),
    }
    _append_jsonl(chat_log, payload)
    if human_output:
        _human_answer(payload, echo_question=echo_question)
    else:
        print(json.dumps(payload, sort_keys=True, allow_nan=False), flush=True)


def _run(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    human_output = _human_output_requested(args)
    if args.replace_chat_log and not args.question:
        raise ValueError("--replace-chat-log requires at least one finite --question")
    checkpoint = _rooted(args.checkpoint)
    default_data = PROJECT_ROOT / "data"
    forbidden = [
        *(default_data / name for name in ("oracle", "qa", "rendered", "features")),
        PROJECT_ROOT / "data_gemma4" / "training",
        PROJECT_ROOT / "reports" / "gemma4" / "scorer_only",
        # Offline training history lives beside the sanitized runtime metadata
        # but is never an inference input.
        checkpoint / "metadata.json",
    ]
    audit = FileAccessAudit(
        forbidden,
        forbidden_component_names=frozenset({"oracle", "qa"}),
        block_forbidden=True,
    )
    audit_path = _rooted(
        args.audit_log or "reports/gemma4/metrics/strict_prefix_chat_access.json"
    )
    chat_log = _rooted(
        args.chat_log or "reports/gemma4/examples/strict_prefix_chat.jsonl"
    )
    completed = False
    runtime: Any | None = None
    try:
        with audit:
            from semantic_3d_chat.chat.runtime import StaticChatRuntime
            from semantic_3d_chat.chat.runtime_config import load_runtime_config

            config = load_runtime_config(args.config, record_file=audit.record)
            for kind in ("oracle", "qa", "rendered", "features"):
                root = artifact_root(config, kind).resolve()
                if root not in audit.forbidden_roots:
                    audit.forbidden_roots.append(root)
            if args.audit_log is None:
                audit_path = reports_root(config) / "metrics" / "strict_prefix_chat_access.json"
            if args.chat_log is None:
                chat_log = reports_root(config) / "examples" / "strict_prefix_chat.jsonl"
            if args.replace_chat_log:
                if chat_log.is_symlink():
                    raise ValueError("Replacement chat log cannot be a symbolic link")
                chat_log.parent.mkdir(parents=True, exist_ok=True)
                chat_log.write_text("", encoding="utf-8")
            runtime = StaticChatRuntime.load(
                config,
                args.scene,
                checkpoint=checkpoint,
                audit=audit,
                local_files_only=True,
            )
            prefix_hash = runtime.scene_prefix_hash
            if runtime.questions_answered != 0 or runtime.current_prefix_hash() != prefix_hash:
                raise RuntimeError("Scene prefix was not finalized before questions")
            startup = {
                **runtime.startup_summary(),
                "phase": "strict_fixed_prefix_chat_ready",
                "scene_id": args.scene,
                "research_checkpoint_behaviorally_promoted": False,
                "scene_prefix_computed_before_question": True,
                "strict_fixed_environment_embedding_input": True,
                "environment_conditioned_input_sha256": prefix_hash,
                "question_conditioned_scene_readout_tokens": False,
                "question_dependent_scene_retrieval": False,
                "environmental_text_inputs": [],
            }
            if human_output:
                _human_startup(startup)
            else:
                print(json.dumps(startup, sort_keys=True, allow_nan=False), flush=True)
            if args.question:
                for question in args.question:
                    _emit(
                        runtime,
                        args.scene,
                        question,
                        chat_log,
                        human_output=human_output,
                        echo_question=human_output,
                    )
            else:
                print(
                    "Ask about the embedded room. Type 'quit' or press Ctrl-D to exit."
                )
                while True:
                    try:
                        question = input("You> ").strip()
                    except EOFError:
                        print()
                        break
                    if question.casefold() in {"quit", "exit"}:
                        break
                    if question:
                        _emit(
                            runtime,
                            args.scene,
                            question,
                            chat_log,
                            human_output=human_output,
                            echo_question=False,
                        )
            runtime.assert_prefix_unchanged()
            completed = True
    finally:
        audit.save(audit_path)
    audit.assert_clean()
    if completed and runtime is not None:
        audit_payload = {
            "phase": "audit_complete",
            "passed": True,
            "prefix_invariant": True,
            "strict_fixed_environment_embedding_input": True,
            "environment_conditioned_input_sha256": runtime.scene_prefix_hash,
            "loaded_file_count": len(audit.unique_paths),
            "forbidden_access_count": len(audit.forbidden_accesses()),
            "audit_log": str(audit_path),
        }
        if human_output:
            _human_audit(audit_payload)
        else:
            print(json.dumps(audit_payload, sort_keys=True), flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        return _run(argv)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"Strict-prefix chat refused: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
