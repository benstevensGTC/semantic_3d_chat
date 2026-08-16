"""Audited interactive chat over a fixed 3D prefix and continuous control head.

This research entry point deliberately requires both checkpoint paths.  The
base checkpoint constructs the complete, question-independent scene prefix;
the small control checkpoint maps that prefix and the user's question directly
to continuous Gemma inputs.  Neither checkpoint may contain QA or oracle files.
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--control-checkpoint", required=True)
    parser.add_argument(
        "--grounding-checkpoint",
        help="Optional authenticated two-file V78 numeric-grounding diagnostic.",
    )
    parser.add_argument("--question", action="append")
    parser.add_argument("--audit-log")
    parser.add_argument("--chat-log")
    return parser


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()


def _emit(runtime: Any, scene_id: str, question: str, chat_log: Path) -> None:
    answer = runtime.answer(question)
    payload = {"phase": "answer", "scene_id": scene_id, **answer.to_dict()}
    grounding_audit = getattr(runtime, "last_grounding_audit", None)
    if grounding_audit is not None:
        payload["grounding_audit"] = dict(grounding_audit)
    _append_jsonl(chat_log, payload)
    print(json.dumps(payload, sort_keys=True, allow_nan=False), flush=True)


def _run(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    default_data = PROJECT_ROOT / "data"
    audit = FileAccessAudit(
        [default_data / name for name in ("oracle", "qa", "rendered", "features")],
        forbidden_component_names=frozenset({"oracle", "qa", "rendered", "features"}),
        block_forbidden=True,
    )
    audit_path = _rooted(args.audit_log or "reports/gemma4/metrics/control_chat_access.json")
    chat_log = _rooted(args.chat_log or "reports/gemma4/examples/control_chat.jsonl")
    completed = False
    try:
        with audit:
            # These imports occur after the audit hook is active so the report
            # covers runtime code, config, model, map, and checkpoint reads.
            from semantic_3d_chat.chat.question_control_runtime import (
                QuestionControlledChatRuntime,
                block_question_control_training_artifacts,
            )
            from semantic_3d_chat.chat.runtime_config import load_runtime_config
            from semantic_3d_chat.evaluation.predict_question_control import (
                _control_checkpoint_sha256,
            )
            from semantic_3d_chat.evaluation.prediction_artifacts import (
                checkpoint_fingerprint,
            )

            config = load_runtime_config(args.config, record_file=audit.record)
            training_artifact_root = block_question_control_training_artifacts(audit, config)
            configured_forbidden = [
                artifact_root(config, kind).resolve()
                for kind in ("oracle", "qa", "rendered", "features")
            ]
            for root in configured_forbidden:
                if root not in audit.forbidden_roots:
                    audit.forbidden_roots.append(root)
            configured_reports = reports_root(config)
            if args.audit_log is None:
                audit_path = configured_reports / "metrics" / "control_chat_access.json"
            if args.chat_log is None:
                chat_log = configured_reports / "examples" / "control_chat.jsonl"

            base_checkpoint = _rooted(args.base_checkpoint)
            control_checkpoint = _rooted(args.control_checkpoint)
            grounding_checkpoint = (
                None
                if args.grounding_checkpoint is None
                else _rooted(args.grounding_checkpoint)
            )
            runtime = QuestionControlledChatRuntime.load(
                config,
                args.scene,
                base_checkpoint=base_checkpoint,
                control_checkpoint=control_checkpoint,
                grounding_checkpoint=grounding_checkpoint,
                audit=audit,
            )
            base_sha256, _base_files = checkpoint_fingerprint(base_checkpoint)
            base_startup = runtime.startup_summary()
            grounding_startup = getattr(runtime, "grounding_sidecar_startup_audit", None)
            prefix_computed_before_question = bool(
                runtime.questions_answered == 0
                and runtime.current_prefix_hash() == runtime.scene_prefix_hash
            )
            if not prefix_computed_before_question:
                raise RuntimeError("Scene prefix was not finalized before accepting user questions")
            startup = {
                "phase": "question_control_chat_ready",
                "scene_id": args.scene,
                "device": base_startup.get("device"),
                "source_voxels": base_startup.get("source_voxels"),
                "processed_voxels": base_startup.get("processed_voxels"),
                "occupied_blocks": base_startup.get("occupied_blocks"),
                "scene_latents": base_startup.get("scene_latents"),
                "language_hidden_dim": base_startup.get("language_hidden_dim"),
                "scene_prefix_shape": base_startup.get("prefix_shape"),
                "scene_prefix_hash": runtime.scene_prefix_hash,
                "scene_control_signature_sha256": (runtime.scene_control_signature_hash),
                "scene_prefix_computed_before_question": (prefix_computed_before_question),
                "questions_answered": runtime.questions_answered,
                "control_architecture": runtime.control_metadata.get("architecture"),
                "control_schema_version": runtime.control_metadata.get("schema_version"),
                "base_checkpoint_sha256": base_sha256,
                "control_checkpoint_sha256": _control_checkpoint_sha256(control_checkpoint),
                "optional_v78_grounding": (
                    grounding_startup() if callable(grounding_startup) else None
                ),
                "blocked_training_artifact_root": str(training_artifact_root),
                "environmental_text_inputs": [],
                "question_dependent_scene_retrieval": False,
            }
            print(json.dumps(startup, sort_keys=True, allow_nan=False), flush=True)
            if args.question:
                for question in args.question:
                    _emit(runtime, args.scene, question, chat_log)
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
                        _emit(runtime, args.scene, question, chat_log)
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
                    "prefix_invariant": True,
                    "loaded_file_count": len(audit.unique_paths),
                    "forbidden_access_count": len(audit.forbidden_accesses()),
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
        print(f"Question-control chat refused: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
