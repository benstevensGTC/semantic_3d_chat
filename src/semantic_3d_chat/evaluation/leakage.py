"""Executable oracle-isolation and question-independent-prefix evaluation."""

from __future__ import annotations

import argparse
import json
import os
import uuid
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from semantic_3d_chat.chat.file_audit import FileAccessAudit
from semantic_3d_chat.config import (
    PROJECT_ROOT,
    artifact_root,
    default_checkpoint_path,
    load_config,
    reports_root,
)

DEFAULT_QUESTIONS = (
    "Is there a chair?",
    "What is on the table?",
    "Which direction would you turn to face the lamp?",
)


@dataclass(frozen=True)
class OracleIsolation:
    original: Path
    hidden: Path | None
    renamed: bool


@contextmanager
def oracle_temporarily_unavailable(oracle_directory: str | Path) -> Iterator[OracleIsolation]:
    """Atomically hide an exact oracle directory and restore it on every exit path."""

    original = Path(oracle_directory).expanduser().resolve()
    if original.name != "oracle":
        raise ValueError(f"Refusing to rename a non-oracle directory: {original}")
    if not original.exists():
        yield OracleIsolation(original=original, hidden=None, renamed=False)
        return
    hidden = original.with_name(f".oracle-unavailable-{os.getpid()}-{uuid.uuid4().hex}")
    if hidden.exists():
        raise FileExistsError(f"Temporary oracle destination already exists: {hidden}")
    original.rename(hidden)
    try:
        yield OracleIsolation(original=original, hidden=hidden, renamed=True)
    finally:
        if original.exists():
            raise RuntimeError(
                "Cannot safely restore oracle because its original path was recreated; "
                f"preserved original data at {hidden}"
            )
        hidden.rename(original)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _default_runtime_loader(
    config: dict[str, Any],
    scene_id: str,
    checkpoint: str | Path,
    audit: FileAccessAudit,
) -> Any:
    # Delay this import until the oracle has been hidden and auditing is active.
    from semantic_3d_chat.chat.runtime import StaticChatRuntime

    return StaticChatRuntime.load(
        config,
        scene_id,
        checkpoint=checkpoint,
        audit=audit,
        local_files_only=True,
    )


def run_leakage_evaluation(
    *,
    config_path: str | Path = "configs/default.yaml",
    scene_id: str = "scene_000001",
    checkpoint: str | Path | None = None,
    questions: Sequence[str] = DEFAULT_QUESTIONS,
    report_path: str | Path | None = None,
    runtime_loader: Callable[[dict[str, Any], str, str | Path, FileAccessAudit], Any] | None = None,
) -> dict[str, Any]:
    if not questions:
        raise ValueError("Leakage evaluation requires at least one question")
    resolved_config = Path(config_path).expanduser()
    if not resolved_config.is_absolute():
        resolved_config = (PROJECT_ROOT / resolved_config).resolve()
    loader = runtime_loader or _default_runtime_loader
    default_data_root = PROJECT_ROOT / "data"
    audit = FileAccessAudit(
        [default_data_root / name for name in ("oracle", "qa", "rendered", "features")]
    )
    isolation: OracleIsolation | None = None
    answers: list[dict[str, Any]] = []
    prefix_hashes: list[str] = []
    startup: dict[str, Any] | None = None
    prefix_computed_before_question = False
    oracle_unavailable_during_inference = False
    failure: BaseException | None = None

    # Activate auditing before resolving recursive config inheritance. This
    # makes the report a complete record of the leakage command's reads, not
    # only of files opened after the final merged config already existed.
    with audit:
        config = load_config(resolved_config)
        audit.record(resolved_config)
        configured_reports_root = reports_root(config).resolve()
        resolved_checkpoint = checkpoint or default_checkpoint_path(config)
        output = (
            Path(report_path).expanduser().resolve()
            if report_path is not None
            else configured_reports_root / "metrics" / "leakage.json"
        )
        oracle = artifact_root(config, "oracle").resolve()
        configured_forbidden = [
            oracle,
            artifact_root(config, "qa").resolve(),
            artifact_root(config, "rendered").resolve(),
            artifact_root(config, "features").resolve(),
        ]
        for root in configured_forbidden:
            if root not in audit.forbidden_roots:
                audit.forbidden_roots.append(root)

        try:
            with oracle_temporarily_unavailable(oracle) as isolation_state:
                isolation = isolation_state
                if isolation.hidden is not None:
                    audit.forbidden_roots.append(isolation.hidden)
                oracle_unavailable_during_inference = not oracle.exists()
                runtime = loader(config, scene_id, resolved_checkpoint, audit)
                startup = runtime.startup_summary()
                prefix_computed_before_question = (
                    runtime.questions_answered == 0
                    and runtime.current_prefix_hash() == runtime.scene_prefix_hash
                )
                prefix_hashes.append(runtime.scene_prefix_hash)
                for question in questions:
                    result = runtime.answer(str(question))
                    answers.append(result.to_dict())
                    prefix_hashes.append(runtime.current_prefix_hash())
                runtime.assert_prefix_unchanged()
            audit.assert_clean()
        except Exception as exc:  # noqa: BLE001 - always emit a failure report after safe restore
            failure = exc

    oracle_restored = oracle.exists() if isolation is not None and isolation.renamed else True
    forbidden_accesses = audit.forbidden_accesses()
    loaded_files = audit.unique_paths
    prefix_invariant = bool(prefix_hashes) and len(set(prefix_hashes)) == 1
    passed = bool(
        failure is None
        and oracle_unavailable_during_inference
        and oracle_restored
        and prefix_computed_before_question
        and prefix_invariant
        and not forbidden_accesses
        and len(answers) == len(questions)
    )
    report = {
        "schema_version": 1,
        "passed": passed,
        "scene_id": scene_id,
        "checkpoint": str(resolved_checkpoint),
        "oracle_directory": str(oracle),
        "oracle_was_renamed": bool(isolation and isolation.renamed),
        "oracle_unavailable_during_inference": oracle_unavailable_during_inference,
        "oracle_restored": oracle_restored,
        "prefix_computed_before_first_question": prefix_computed_before_question,
        "prefix_invariant": prefix_invariant,
        "prefix_hash": prefix_hashes[0] if prefix_hashes else None,
        "prefix_hashes": prefix_hashes,
        "question_count": len(questions),
        "answers": answers,
        "startup": startup,
        "loaded_files": loaded_files,
        "forbidden_accesses": forbidden_accesses,
        "failure": None if failure is None else f"{type(failure).__name__}: {failure}",
    }
    _atomic_json(output, report)
    if failure is not None:
        raise RuntimeError(f"Leakage evaluation failed; report written to {output}") from failure
    if not passed:
        raise RuntimeError(f"Leakage evaluation did not pass; report written to {output}")
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", default="configs/default.yaml")
    result.add_argument("--scene", default="scene_000001")
    result.add_argument("--checkpoint")
    result.add_argument("--question", action="append")
    result.add_argument("--output", default=None)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    report = run_leakage_evaluation(
        config_path=args.config,
        scene_id=args.scene,
        checkpoint=args.checkpoint,
        questions=args.question or DEFAULT_QUESTIONS,
        report_path=args.output,
    )
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "scene_id": report["scene_id"],
                "oracle_was_renamed": report["oracle_was_renamed"],
                "oracle_restored": report["oracle_restored"],
                "oracle_unavailable_during_inference": report[
                    "oracle_unavailable_during_inference"
                ],
                "prefix_computed_before_first_question": report[
                    "prefix_computed_before_first_question"
                ],
                "prefix_invariant": report["prefix_invariant"],
                "prefix_hash": report["prefix_hash"],
                "question_count": report["question_count"],
                "loaded_file_count": len(report["loaded_files"]),
                "forbidden_access_count": len(report["forbidden_accesses"]),
            },
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
