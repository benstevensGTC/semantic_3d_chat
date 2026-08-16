"""Audited CLI for the PASS-gated, unpromoted V96 candidate runtime."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import yaml

from semantic_3d_chat.chat.file_audit import FileAccessAudit
from semantic_3d_chat.chat.v96_explicit_candidate_runtime import (
    V96CandidateAuthorization,
)
from semantic_3d_chat.config import PROJECT_ROOT, reports_root

DEFAULT_HOOK: Final[str] = "configs/runtime/gemma4_v96_explicit_candidate_hook.yaml"
HOOK_ARTIFACT: Final[str] = "gemma4_v96_explicit_candidate_runtime_hook_v1"
HOOK_MODE: Final[str] = "explicit_candidate_only_not_default"
_SCENE_ID = re.compile(r"scene_[0-9]{6}")


@dataclass(frozen=True)
class V96RuntimeHook:
    path: Path
    authorization_config: Path
    runtime_config: Path
    default_scene: str
    scene_memory_root: Path
    local_files_only: bool


def _absolute(path: str | Path) -> Path:
    value = Path(path).expanduser()
    rooted = value if value.is_absolute() else PROJECT_ROOT / value
    return Path(os.path.abspath(rooted))


def _strict_json_object(raw: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate V96 authorization field: {key}")
            result[key] = value
        return result

    value = json.loads(raw, object_pairs_hook=reject_duplicates)
    if not isinstance(value, dict):
        raise TypeError("V96 authorization subprocess must return one JSON object")
    return value


def load_v96_runtime_hook(path: str | Path = DEFAULT_HOOK) -> V96RuntimeHook:
    source = _absolute(path)
    if source.is_symlink() or not source.is_file():
        raise FileNotFoundError(source)
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping) or set(raw) != {"v96_candidate_runtime"}:
        raise ValueError("V96 runtime hook must contain one v96_candidate_runtime mapping")
    hook = raw["v96_candidate_runtime"]
    required = {
        "schema_version",
        "artifact",
        "mode",
        "authorization_config",
        "runtime_config",
        "default_scene",
        "scene_memory_root",
        "local_files_only",
        "require_authenticated_pass_evidence",
        "require_explicit_candidate_flag",
        "default_runtime_pointer_modified",
        "runtime_promotion_authorized",
        "environmental_text_inputs",
    }
    if not isinstance(hook, Mapping) or set(hook) != required:
        raise ValueError("V96 runtime hook fields changed")
    default_scene = hook.get("default_scene")
    if (
        hook.get("schema_version") != 96
        or hook.get("artifact") != HOOK_ARTIFACT
        or hook.get("mode") != HOOK_MODE
        or not isinstance(default_scene, str)
        or _SCENE_ID.fullmatch(default_scene) is None
        or hook.get("local_files_only") is not True
        or hook.get("require_authenticated_pass_evidence") is not True
        or hook.get("require_explicit_candidate_flag") is not True
        or hook.get("default_runtime_pointer_modified") is not False
        or hook.get("runtime_promotion_authorized") is not False
        or hook.get("environmental_text_inputs") != []
    ):
        raise ValueError("V96 runtime hook is not a safe explicit-candidate contract")
    return V96RuntimeHook(
        path=source,
        authorization_config=_absolute(str(hook["authorization_config"])),
        runtime_config=_absolute(str(hook["runtime_config"])),
        default_scene=default_scene,
        scene_memory_root=_absolute(str(hook["scene_memory_root"])),
        local_files_only=True,
    )


def run_isolated_v96_authorization(
    config_path: str | Path,
    *,
    timeout_seconds: float = 600.0,
) -> V96CandidateAuthorization:
    """Authenticate in a child that cannot expose evaluation text to Gemma."""

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "semantic_3d_chat.chat.v96_explicit_candidate_authorize",
            "--config",
            str(_absolute(config_path)),
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or "authorization subprocess failed"
        raise RuntimeError(detail)
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise ValueError("V96 authorization subprocess returned unexpected output")
    authorization = V96CandidateAuthorization.from_payload(
        _strict_json_object(lines[0])
    )
    if Path(authorization.authorization_config_path).resolve() != _absolute(config_path):
        raise ValueError("V96 authorization was produced for a different sealed config")
    return authorization


def _forbidden_roots(authorization: V96CandidateAuthorization) -> list[Path]:
    roots = [
        PROJECT_ROOT / "data/oracle",
        PROJECT_ROOT / "data/qa",
        PROJECT_ROOT / "data/rendered",
        PROJECT_ROOT / "data/features",
        PROJECT_ROOT / "data_diverse52/qa",
        PROJECT_ROOT / "data_gemma4/features",
        PROJECT_ROOT / "data_gemma4/training",
        PROJECT_ROOT / "reports/gemma4/questions",
        PROJECT_ROOT / "reports/gemma4/predictions",
        Path(authorization.authorization_config_path),
        Path(authorization.final_score_path),
        Path(authorization.evidence_path),
    ]
    roots.extend(PROJECT_ROOT.glob("data*/oracle"))
    roots.extend(PROJECT_ROOT.glob("data*/qa"))
    return list(dict.fromkeys(path.resolve() for path in roots))


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), sort_keys=True, allow_nan=False) + "\n")
        handle.flush()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hook", default=DEFAULT_HOOK)
    parser.add_argument("--scene")
    parser.add_argument("--scene-memory")
    parser.add_argument("--question", action="append")
    parser.add_argument("--audit-log")
    parser.add_argument("--chat-log")
    parser.add_argument("--authorization-timeout-seconds", type=float, default=600.0)
    parser.add_argument(
        "--allow-explicit-candidate",
        action="store_true",
        help=(
            "Acknowledge that authenticated PASS evidence does not automatically "
            "promote V96 or change the default runtime."
        ),
    )
    return parser


def _run(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.allow_explicit_candidate:
        raise ValueError("pass --allow-explicit-candidate to use unpromoted V96")
    hook = load_v96_runtime_hook(args.hook)
    scene_id = args.scene or hook.default_scene
    if _SCENE_ID.fullmatch(scene_id) is None:
        raise ValueError("V96 scene ID must be opaque scene_NNNNNN")
    memory_path = (
        _absolute(args.scene_memory)
        if args.scene_memory is not None
        else hook.scene_memory_root / scene_id
    )

    # This finishes before runtime/model imports and returns no row content.
    authorization = run_isolated_v96_authorization(
        hook.authorization_config,
        timeout_seconds=args.authorization_timeout_seconds,
    )
    if Path(authorization.runtime_config_path).resolve() != hook.runtime_config:
        raise ValueError("V96 hook runtime config differs from the authenticated source")

    audit = FileAccessAudit(
        _forbidden_roots(authorization),
        forbidden_component_names=frozenset(
            {"oracle", "qa", "training", "scorer", "predictions"}
        ),
        block_forbidden=True,
    )
    audit_path = _absolute(
        args.audit_log
        or f"reports/gemma4/metrics/v96_explicit_chat_access_{scene_id}.json"
    )
    chat_path = _absolute(
        args.chat_log
        or f"reports/gemma4/examples/v96_explicit_chat_{scene_id}.jsonl"
    )
    completed = False
    oracle_available_at_start = (PROJECT_ROOT / "data/oracle").exists()
    runtime = None
    try:
        with audit:
            from semantic_3d_chat.chat.runtime_config import load_runtime_config
            from semantic_3d_chat.chat.v96_explicit_candidate_runtime import (
                V96ExplicitCandidateChatRuntime,
            )

            audit.record(hook.path)
            config = load_runtime_config(hook.runtime_config, record_file=audit.record)
            if args.audit_log is None:
                audit_path = (
                    reports_root(config)
                    / "metrics"
                    / f"v96_explicit_chat_access_{scene_id}.json"
                )
            if args.chat_log is None:
                chat_path = (
                    reports_root(config)
                    / "examples"
                    / f"v96_explicit_chat_{scene_id}.jsonl"
                )
            runtime = V96ExplicitCandidateChatRuntime.load(
                config,
                scene_id,
                authorization=authorization,
                scene_memory=memory_path,
                audit=audit,
                local_files_only=hook.local_files_only,
            )
            if runtime.questions_answered != 0:
                raise RuntimeError("V96 scene memory was not bound before question input")
            startup = {
                **runtime.startup_summary(),
                "explicit_candidate_allowed": True,
                "prefix_hash_printed_before_questions": True,
                "oracle_directory_available_at_runtime_start": (
                    oracle_available_at_start
                ),
            }
            print(json.dumps(startup, sort_keys=True, allow_nan=False), flush=True)

            def answer(question: str) -> None:
                result = runtime.answer(question)
                payload = {
                    "phase": "answer",
                    "scene_id": scene_id,
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
                raise RuntimeError("V96 scene prefix changed across questions")
            completed = True
    finally:
        audit.save(audit_path)
    audit.assert_clean()
    if completed and runtime is not None:
        print(
            json.dumps(
                {
                    "phase": "v96_explicit_chat_audit_complete",
                    "passed": True,
                    "authenticated_pass_evidence": True,
                    "prefix_hash_printed_before_questions": True,
                    "prefix_hash_invariant": True,
                    "exact_738_token_memory_supplied_directly_to_gemma": True,
                    "frozen_lora_bank_count": 10,
                    "frozen_lora_parameter_count": 864_256,
                    "question_derived_environmental_tokens": 0,
                    "oracle_or_text_metadata_loaded_by_chat": False,
                    "automatic_runtime_promotion": False,
                    "runtime_promotion_authorized": False,
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
    except (
        OSError,
        RuntimeError,
        subprocess.SubprocessError,
        TypeError,
        ValueError,
    ) as error:
        print(f"V96 explicit chat refused: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_HOOK",
    "V96RuntimeHook",
    "load_v96_runtime_hook",
    "main",
    "run_isolated_v96_authorization",
]
