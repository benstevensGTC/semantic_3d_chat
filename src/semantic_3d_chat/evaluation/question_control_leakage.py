"""Oracle-deletion and prefix-invariance test for continuous-control chat."""

from __future__ import annotations

import argparse
import json
import os
import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.evaluation.leakage import (
    DEFAULT_QUESTIONS,
    _atomic_json,
    run_leakage_evaluation,
)


@dataclass(frozen=True)
class TeacherArtifactIsolation:
    """State for one exact, atomically hidden training-only teacher directory."""

    original: Path
    training_root: Path
    hidden: Path | None
    renamed: bool


def _unresolved_project_path(path: str | Path) -> Path:
    value = Path(path).expanduser()
    rooted = value if value.is_absolute() else PROJECT_ROOT / value
    return Path(os.path.abspath(rooted))


def _reject_symlink_components(path: Path) -> None:
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current /= component
        if current.is_symlink():
            raise ValueError(
                f"Teacher artifact isolation refuses symbolic-link path components: {current}"
            )


@contextmanager
def teacher_artifact_temporarily_unavailable(
    teacher_directory: str | Path,
) -> Iterator[TeacherArtifactIsolation]:
    """Atomically hide one direct child of an exact ``training`` artifact root.

    The containing training root remains an explicit file-audit denylist.  Only
    the selected numeric teacher directory is renamed; Python source modules in
    ``src/semantic_3d_chat/training`` are unrelated and remain importable.
    """

    original = _unresolved_project_path(teacher_directory)
    _reject_symlink_components(original)
    training_root = original.parent
    if training_root.name.casefold() != "training" or not original.name:
        raise ValueError("Teacher artifact must be one exact direct child of a training root")
    if original.exists() and not original.is_dir():
        raise ValueError(f"Teacher artifact is not a directory: {original}")
    if not original.exists():
        yield TeacherArtifactIsolation(
            original=original,
            training_root=training_root,
            hidden=None,
            renamed=False,
        )
        return
    hidden = original.with_name(f".{original.name}-unavailable-{os.getpid()}-{uuid.uuid4().hex}")
    if hidden.exists():
        raise FileExistsError(f"Temporary teacher destination exists: {hidden}")
    original.rename(hidden)
    try:
        yield TeacherArtifactIsolation(
            original=original,
            training_root=training_root,
            hidden=hidden,
            renamed=True,
        )
    finally:
        if original.exists():
            raise RuntimeError(
                "Cannot safely restore teacher artifact because its original path was "
                f"recreated; preserved original data at {hidden}"
            )
        hidden.rename(original)


def _path_is_within(path: str | Path, root: Path) -> bool:
    try:
        Path(path).expanduser().resolve().relative_to(root.resolve())
    except (OSError, RuntimeError, ValueError):
        return False
    return True


class _LeakageRuntimeAdapter:
    """Expose the generic leakage-runner protocol without mutating the runtime."""

    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime
        self.scene_prefix_hash = runtime.scene_prefix_hash
        self.questions_answered = 0
        self.control_token_hashes: list[str | None] = []
        self.environment_conditioned_input_hashes: list[str] = []
        self.grounding_scene_token_hashes: list[str | None] = []
        self.control_contract = _validated_control_runtime_contract(runtime)

    def startup_summary(self) -> dict[str, Any]:
        runtime_summary = getattr(self.runtime, "startup_summary", None)
        summary = dict(
            runtime_summary() if callable(runtime_summary) else self.runtime.base.startup_summary()
        )
        summary.update(
            {
                "runtime_kind": "continuous_scene_question_control",
                "scene_prefix_hash": self.scene_prefix_hash,
                **self.control_contract,
                "environmental_text_inputs": [],
                "question_dependent_scene_retrieval": False,
            }
        )
        return summary

    def current_prefix_hash(self) -> str:
        current = getattr(self.runtime, "current_prefix_hash", None)
        return current() if callable(current) else self.runtime.scene_prefix_hash

    def answer(self, question: str) -> Any:
        result = self.runtime.answer(question)
        self.questions_answered += 1
        self.control_token_hashes.append(
            getattr(self.runtime, "last_control_tokens_sha256", None)
        )
        self.environment_conditioned_input_hashes.append(
            getattr(
                self.runtime,
                "last_environment_conditioned_input_sha256",
                self.scene_prefix_hash,
            )
        )
        grounding_audit = getattr(self.runtime, "last_grounding_audit", None)
        self.grounding_scene_token_hashes.append(
            None
            if not isinstance(grounding_audit, Mapping)
            else grounding_audit.get("scene_tokens_sha256")
        )
        return result

    def assert_prefix_unchanged(self) -> None:
        self.runtime.assert_prefix_unchanged()


def _validated_control_runtime_contract(runtime: Any) -> dict[str, Any]:
    """Return architecture-neutral safety facts from sanitized runtime metadata.

    The leakage evaluator deliberately does not enumerate V1/V2/V3/V4.  The
    production loader owns each architecture's exact schema validation; this
    boundary verifies only the invariants every continuous-control runtime must
    expose before it is allowed into the generic oracle-deletion runner.
    """

    metadata = getattr(runtime, "control_metadata", None)
    if not isinstance(metadata, Mapping):
        raise TypeError("Question-control runtime metadata must be a mapping")
    architecture = metadata.get("architecture")
    schema_version = metadata.get("schema_version")
    if not isinstance(architecture, str) or not architecture.strip():
        raise ValueError("Question-control runtime architecture is missing")
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version < 1
    ):
        raise ValueError("Question-control runtime schema version is invalid")
    if metadata.get("environmental_text_inputs") != []:
        raise ValueError("Question-control runtime declares environmental text inputs")
    if metadata.get("question_dependent_scene_retrieval") is not False:
        raise ValueError("Question-control runtime permits question-dependent retrieval")
    if metadata.get("complete_scene_prefix_required") is not True:
        raise ValueError("Question-control runtime does not require the complete scene prefix")
    return {
        "control_architecture": architecture,
        "control_schema_version": schema_version,
        "control_runtime_contract_safe": True,
        "complete_scene_prefix_required": True,
        "question_conditioned_scene_readout_tokens": True,
        "strict_fixed_environment_embedding_input": False,
    }


def run_question_control_leakage(
    *,
    config_path: str | Path,
    scene_id: str,
    base_checkpoint: str | Path,
    control_checkpoint: str | Path,
    grounding_checkpoint: str | Path | None = None,
    teacher_artifact: str | Path | None = None,
    training_artifact: str | Path | None = None,
    questions: Sequence[str] = DEFAULT_QUESTIONS,
    report_path: str | Path,
) -> dict[str, Any]:
    if teacher_artifact is not None and training_artifact is not None:
        raise ValueError("Select only one training artifact to isolate")
    selected_training_artifact = (
        training_artifact if training_artifact is not None else teacher_artifact
    )
    control_path = Path(control_checkpoint).expanduser().resolve()
    grounding_path = (
        None
        if grounding_checkpoint is None
        else Path(grounding_checkpoint).expanduser().resolve()
    )
    control_hash: list[str] = []
    control_contracts: list[dict[str, Any]] = []
    runtime_adapters: list[_LeakageRuntimeAdapter] = []
    runtime_training_roots: list[Path] = []
    isolation: TeacherArtifactIsolation | None = None

    def load_runtime(
        config: dict[str, Any],
        selected_scene: str,
        selected_base: str | Path,
        audit: Any,
    ) -> _LeakageRuntimeAdapter:
        # Import the production runtime only after the generic runner has
        # hidden the oracle and activated its process-wide file audit.
        from semantic_3d_chat.chat.question_control_runtime import (
            QuestionControlledChatRuntime,
            question_control_training_artifact_root,
        )
        from semantic_3d_chat.evaluation.predict_question_control import (
            _control_checkpoint_sha256,
        )

        training_root = question_control_training_artifact_root(config).resolve()
        if training_root not in runtime_training_roots:
            runtime_training_roots.append(training_root)
        if isolation is not None and isolation.training_root.resolve() != training_root:
            raise ValueError(
                "Isolated training artifact is outside the runtime's derived training root"
            )
        if training_root not in audit.forbidden_roots:
            audit.forbidden_roots.append(training_root)
        if (
            isolation is not None
            and isolation.hidden is not None
            and isolation.hidden not in audit.forbidden_roots
        ):
            audit.forbidden_roots.append(isolation.hidden)
        runtime = QuestionControlledChatRuntime.load(
            config,
            selected_scene,
            base_checkpoint=selected_base,
            control_checkpoint=control_path,
            grounding_checkpoint=grounding_path,
            audit=audit,
        )
        control_hash.append(_control_checkpoint_sha256(control_path))
        adapter = _LeakageRuntimeAdapter(runtime)
        control_contracts.append(adapter.control_contract)
        runtime_adapters.append(adapter)
        return adapter

    output = Path(report_path).expanduser().resolve()
    training_artifact_unavailable_during_inference: bool | None = None
    if selected_training_artifact is None:
        report = run_leakage_evaluation(
            config_path=config_path,
            scene_id=scene_id,
            checkpoint=base_checkpoint,
            questions=questions,
            report_path=output,
            runtime_loader=load_runtime,
        )
    else:
        with teacher_artifact_temporarily_unavailable(selected_training_artifact) as state:
            isolation = state
            training_artifact_unavailable_during_inference = not state.original.exists()
            report = run_leakage_evaluation(
                config_path=config_path,
                scene_id=scene_id,
                checkpoint=base_checkpoint,
                questions=questions,
                report_path=output,
                runtime_loader=load_runtime,
            )
    if len(runtime_training_roots) != 1:
        raise RuntimeError("Question-control leakage loader did not resolve one training root")
    training_root = runtime_training_roots[0]
    training_artifact_restored = bool(
        isolation is None
        or (isolation.original.is_dir() if isolation.renamed else not isolation.original.exists())
    )
    report["runtime_kind"] = "continuous_scene_question_control"
    report["control_checkpoint"] = str(control_path)
    report["grounding_checkpoint"] = (
        None if grounding_path is None else str(grounding_path)
    )
    if len(control_hash) != 1 or len(control_contracts) != 1 or len(runtime_adapters) != 1:
        raise RuntimeError("Question-control leakage loader did not run exactly once")
    report["control_checkpoint_sha256"] = control_hash[0]
    report.update(control_contracts[0])
    adapter = runtime_adapters[0]
    report["control_token_sha256_by_question"] = adapter.control_token_hashes
    report["environment_conditioned_input_sha256_by_question"] = (
        adapter.environment_conditioned_input_hashes
    )
    report["environment_conditioned_input_invariant"] = (
        len(set(adapter.environment_conditioned_input_hashes)) <= 1
    )
    report["grounding_scene_token_sha256_by_question"] = (
        adapter.grounding_scene_token_hashes
    )
    nonempty_grounding_hashes = [
        value for value in adapter.grounding_scene_token_hashes if value is not None
    ]
    report["grounding_scene_tokens_invariant"] = bool(
        grounding_path is None
        or (
            len(nonempty_grounding_hashes) == len(questions)
            and len(set(nonempty_grounding_hashes)) == 1
        )
    )
    expected_control_files = [
        str((control_path / name).resolve())
        for name in ("control.safetensors", "runtime_metadata.json")
    ]
    loaded_files = set(report["loaded_files"])
    loaded_control_files = [path for path in expected_control_files if path in loaded_files]
    expected_grounding_files = (
        []
        if grounding_path is None
        else [
            str((grounding_path / name).resolve())
            for name in ("grounding.safetensors", "metadata.json")
        ]
    )
    loaded_grounding_files = [
        path for path in expected_grounding_files if path in loaded_files
    ]
    isolated_roots: list[Path] = []
    if isolation is not None:
        isolated_roots.append(isolation.original)
        if isolation.hidden is not None:
            isolated_roots.append(isolation.hidden)
    isolated_artifact_loaded_paths = [
        loaded
        for loaded in report["loaded_files"]
        if any(_path_is_within(loaded, root) for root in isolated_roots)
    ]
    training_artifact_loaded_paths = [
        loaded for loaded in report["loaded_files"] if _path_is_within(loaded, training_root)
    ]
    report["runtime_checkpoint_files_expected"] = expected_control_files
    report["runtime_checkpoint_files_loaded"] = loaded_control_files
    report["runtime_checkpoint_files_complete"] = loaded_control_files == expected_control_files
    report["grounding_checkpoint_files_expected"] = expected_grounding_files
    report["grounding_checkpoint_files_loaded"] = loaded_grounding_files
    report["grounding_checkpoint_files_complete"] = (
        loaded_grounding_files == expected_grounding_files
    )
    report["training_artifact_root"] = str(training_root)
    report["training_artifact_isolation_requested"] = isolation is not None
    report["isolated_training_artifact_directory"] = (
        None if isolation is None else str(isolation.original)
    )
    report["training_artifact_was_renamed"] = bool(isolation and isolation.renamed)
    report["training_artifact_unavailable_during_inference"] = (
        training_artifact_unavailable_during_inference
    )
    report["training_artifact_restored"] = training_artifact_restored
    report["isolated_training_artifact_loaded_paths"] = isolated_artifact_loaded_paths
    # Backward-compatible V58 report aliases.  New V4 callers can omit a
    # teacher-specific path and rely on the exact training-root audit above.
    report["teacher_artifact_directory"] = None if isolation is None else str(isolation.original)
    report["teacher_artifact_was_renamed"] = bool(isolation and isolation.renamed)
    report["teacher_artifact_unavailable_during_inference"] = (
        training_artifact_unavailable_during_inference
    )
    report["teacher_artifact_restored"] = training_artifact_restored
    report["teacher_artifact_loaded_paths"] = isolated_artifact_loaded_paths
    report["training_artifact_loaded_paths"] = training_artifact_loaded_paths
    report["teacher_artifact_loaded"] = bool(isolated_artifact_loaded_paths)
    report["qa_or_oracle_loaded"] = any(
        {"qa", "oracle"} & {part.casefold() for part in Path(loaded).parts}
        for loaded in report["loaded_files"]
    )
    report["passed"] = bool(
        report["passed"]
        and report["runtime_checkpoint_files_complete"]
        and report["grounding_checkpoint_files_complete"]
        and report["grounding_scene_tokens_invariant"]
        and (isolation is None or report["training_artifact_unavailable_during_inference"] is True)
        and report["training_artifact_restored"]
        and not report["teacher_artifact_loaded"]
        and not report["training_artifact_loaded_paths"]
        and not report["qa_or_oracle_loaded"]
    )
    _atomic_json(output, report)
    if not report["passed"]:
        raise RuntimeError(f"Question-control leakage test failed: {output}")
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--control-checkpoint", required=True)
    parser.add_argument("--grounding-checkpoint")
    artifact = parser.add_mutually_exclusive_group()
    artifact.add_argument("--training-artifact")
    artifact.add_argument("--teacher-artifact")
    parser.add_argument("--question", action="append")
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = run_question_control_leakage(
        config_path=args.config,
        scene_id=args.scene,
        base_checkpoint=args.base_checkpoint,
        control_checkpoint=args.control_checkpoint,
        grounding_checkpoint=args.grounding_checkpoint,
        teacher_artifact=args.teacher_artifact,
        training_artifact=args.training_artifact,
        questions=args.question or DEFAULT_QUESTIONS,
        report_path=args.output,
    )
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "oracle_was_renamed": report["oracle_was_renamed"],
                "oracle_restored": report["oracle_restored"],
                "prefix_computed_before_first_question": report[
                    "prefix_computed_before_first_question"
                ],
                "prefix_invariant": report["prefix_invariant"],
                "question_count": report["question_count"],
                "loaded_file_count": len(report["loaded_files"]),
                "forbidden_access_count": len(report["forbidden_accesses"]),
                "teacher_artifact_loaded": report["teacher_artifact_loaded"],
                "control_architecture": report["control_architecture"],
                "control_schema_version": report["control_schema_version"],
                "training_artifact_isolation_requested": report[
                    "training_artifact_isolation_requested"
                ],
                "teacher_artifact_was_renamed": report["teacher_artifact_was_renamed"],
                "teacher_artifact_restored": report["teacher_artifact_restored"],
                "runtime_checkpoint_files_complete": report["runtime_checkpoint_files_complete"],
                "grounding_checkpoint_files_complete": report[
                    "grounding_checkpoint_files_complete"
                ],
                "grounding_scene_tokens_invariant": report[
                    "grounding_scene_tokens_invariant"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "TeacherArtifactIsolation",
    "run_question_control_leakage",
    "teacher_artifact_temporarily_unavailable",
]
