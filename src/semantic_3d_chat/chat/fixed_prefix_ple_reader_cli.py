"""Audited local chat with the hard-gated V54 fixed-prefix PLE reader."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from semantic_3d_chat.chat.file_audit import FileAccessAudit
from semantic_3d_chat.config import PROJECT_ROOT, artifact_root, project_path, reports_root

_OPAQUE_SCENE_ID = re.compile(r"scene_[0-9]{6}")
_DEFAULT_CONFIG = "configs/runtime/gemma4_v54.yaml"
_DEFAULT_BASE_CHECKPOINT = (
    "data_gemma4/checkpoints/gemma4_v54_semantic_greedy_gate/update_000"
)
_DEFAULT_READER_CHECKPOINT = (
    "data_gemma4/checkpoints/gemma4_v54_fixed_prefix_ple_reader_v4"
)


def _rooted(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    rooted = candidate if candidate.is_absolute() else PROJECT_ROOT / candidate
    return Path(os.path.abspath(rooted))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=_DEFAULT_CONFIG)
    parser.add_argument("--scene", default="scene_000001")
    parser.add_argument("--base-checkpoint", default=_DEFAULT_BASE_CHECKPOINT)
    parser.add_argument("--reader-checkpoint", default=_DEFAULT_READER_CHECKPOINT)
    parser.add_argument("--question", action="append")
    parser.add_argument("--audit-log")
    parser.add_argument("--chat-log")
    parser.add_argument(
        "--replace-chat-log",
        action="store_true",
        help="Replace the selected transcript before a finite --question run",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Authenticate sanitized inputs without loading Gemma weights",
    )
    return parser


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()


def _runtime_audit() -> FileAccessAudit:
    default_data = PROJECT_ROOT / "data"
    return FileAccessAudit(
        [
            *(default_data / name for name in ("oracle", "qa", "rendered", "features")),
            PROJECT_ROOT / "data_gemma4" / "training",
            PROJECT_ROOT / "reports" / "gemma4" / "scorer_only",
        ],
        forbidden_component_names=frozenset(
            {"oracle", "qa", "rendered", "features", "scorer_only", "scorer-only"}
        ),
        block_forbidden=True,
    )


def _extend_forbidden_roots(audit: FileAccessAudit, config: dict[str, Any]) -> None:
    for kind in ("oracle", "qa", "rendered", "features"):
        root = artifact_root(config, kind).resolve()
        if root not in audit.forbidden_roots:
            audit.forbidden_roots.append(root)
    training_root = artifact_root(config, "checkpoints").resolve().parent / "training"
    scorer_root = reports_root(config).resolve() / "scorer_only"
    for root in (training_root, scorer_root):
        if root not in audit.forbidden_roots:
            audit.forbidden_roots.append(root)


def _load_config(path: Path, audit: FileAccessAudit) -> dict[str, Any]:
    from semantic_3d_chat.chat.runtime_config import load_runtime_config

    return load_runtime_config(path, record_file=audit.record)


def _authenticate_inputs(
    config: dict[str, Any],
    scene_id: str,
    base_checkpoint: Path,
    reader_checkpoint: Path,
    audit: FileAccessAudit,
) -> dict[str, Any]:
    from semantic_3d_chat.chat.fixed_prefix_ple_reader_runtime import (
        validate_ple_reader_checkpoint,
        validate_v54_checkpoint,
        validate_v54_runtime_config,
    )

    config_sha256 = validate_v54_runtime_config(config)
    base = validate_v54_checkpoint(base_checkpoint, audit=audit)
    reader = validate_ple_reader_checkpoint(reader_checkpoint, audit=audit)
    map_path = project_path(config, "maps", scene_id, "voxel_map.npz").resolve()
    if not map_path.is_file() or map_path.is_symlink():
        raise FileNotFoundError(f"Sanitized numeric voxel map is unavailable: {map_path}")
    audit.record(map_path)
    if (
        reader.metadata["base_runtime_config_effective_sha256"] != config_sha256
        or reader.metadata["base_checkpoint_sha256"]
        != "3e128b40c1b73bb32750285679cda6b1bea364e67465e986a94a81dfc95e81e8"
    ):
        raise ValueError("PLE reader checkpoint is not bound to authenticated V54 inputs")
    return {
        "phase": "fixed_prefix_ple_reader_preflight",
        "passed": True,
        "scene_id": scene_id,
        "config_effective_sha256": config_sha256,
        "base_checkpoint": str(base.root),
        "base_adapter_sha256": base.adapter_sha256,
        "base_runtime_metadata_sha256": base.runtime_metadata_sha256,
        "base_training_metadata_opened": False,
        "reader_checkpoint": str(reader.root),
        "reader_artifact": reader.metadata["artifact"],
        "reader_adapter_file_sha256": reader.metadata["adapter_file_sha256"],
        "reader_adapter_state_sha256": reader.metadata["adapter_state_sha256"],
        "sanitized_numeric_map": str(map_path),
        "scene_prefix_computed_before_question_required": True,
        "question_dependent_scene_retrieval": False,
        "environmental_text_inputs": [],
    }


def _load_runtime(
    config: dict[str, Any],
    scene_id: str,
    base_checkpoint: Path,
    reader_checkpoint: Path,
    audit: FileAccessAudit,
) -> Any:
    from semantic_3d_chat.chat.fixed_prefix_ple_reader_runtime import (
        FixedPrefixPLEReaderChatRuntime,
    )

    return FixedPrefixPLEReaderChatRuntime.load(
        config,
        scene_id,
        base_checkpoint=base_checkpoint,
        reader_checkpoint=reader_checkpoint,
        audit=audit,
        local_files_only=True,
    )


def _emit(runtime: Any, scene_id: str, question: str, chat_log: Path) -> None:
    before = runtime.current_prefix_hash()
    if before != runtime.scene_prefix_hash:
        raise RuntimeError("PLE reader scene prefix changed before question input")
    answer = runtime.answer(question)
    after = runtime.current_prefix_hash()
    if before != after or answer.prefix_hash != before:
        raise RuntimeError("PLE reader did not reuse the exact fixed scene prefix")
    payload = {
        "phase": "answer",
        "scene_id": scene_id,
        "strict_fixed_environment_embedding_input": True,
        "environment_conditioned_input_sha256": before,
        "question_dependent_scene_retrieval": False,
        "environmental_text_inputs": [],
        **answer.to_dict(),
    }
    _append_jsonl(chat_log, payload)
    print(json.dumps(payload, sort_keys=True, allow_nan=False), flush=True)


def _run(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if _OPAQUE_SCENE_ID.fullmatch(args.scene) is None:
        raise ValueError("scene_id must be opaque and match scene_ followed by six digits")
    if args.replace_chat_log and (args.check or not args.question):
        raise ValueError("--replace-chat-log requires a finite non-check --question run")

    config_path = _rooted(args.config)
    base_checkpoint = _rooted(args.base_checkpoint)
    reader_checkpoint = _rooted(args.reader_checkpoint)
    audit_path = _rooted(
        args.audit_log
        or "reports/gemma4/metrics/fixed_prefix_ple_reader_chat_access.json"
    )
    chat_log = _rooted(
        args.chat_log
        or "reports/gemma4/examples/fixed_prefix_ple_reader_chat.jsonl"
    )
    audit = _runtime_audit()
    completed = False
    runtime: Any | None = None
    try:
        with audit:
            config = _load_config(config_path, audit)
            _extend_forbidden_roots(audit, config)
            preflight = _authenticate_inputs(
                config,
                args.scene,
                base_checkpoint,
                reader_checkpoint,
                audit,
            )
            if args.audit_log is None:
                audit_path = reports_root(config) / "metrics" / audit_path.name
            if args.chat_log is None:
                chat_log = reports_root(config) / "examples" / chat_log.name
            print(json.dumps(preflight, sort_keys=True, allow_nan=False), flush=True)
            if not args.check:
                if args.replace_chat_log:
                    if chat_log.is_symlink():
                        raise ValueError("Replacement chat log cannot be a symbolic link")
                    chat_log.parent.mkdir(parents=True, exist_ok=True)
                    chat_log.write_text("", encoding="utf-8")
                runtime = _load_runtime(
                    config,
                    args.scene,
                    base_checkpoint,
                    reader_checkpoint,
                    audit,
                )
                if runtime.questions_answered != 0:
                    raise RuntimeError("PLE reader accepted questions before startup completed")
                startup = runtime.startup_summary()
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
    print(
        json.dumps(
            {
                "phase": "fixed_prefix_ple_reader_audit_complete",
                "passed": completed,
                "check_only": bool(args.check),
                "prefix_invariant": (
                    None
                    if runtime is None
                    else runtime.current_prefix_hash() == runtime.scene_prefix_hash
                ),
                "strict_fixed_environment_embedding_input": True,
                "environment_conditioned_input_sha256": (
                    None if runtime is None else runtime.scene_prefix_hash
                ),
                "questions_answered": 0 if runtime is None else runtime.questions_answered,
                "forbidden_access_count": len(audit.forbidden_accesses()),
                "loaded_file_count": len(audit.unique_paths),
                "audit_log": str(audit_path),
            },
            sort_keys=True,
            allow_nan=False,
        ),
        flush=True,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        return _run(argv)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"Fixed-prefix PLE reader chat refused: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
