"""Oracle/training-removal audit for the strict fixed-prefix atlas runtime."""

from __future__ import annotations

import argparse
import json
import os
import uuid
from collections.abc import Iterator, Sequence
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
class TrainingRootIsolation:
    original: Path
    hidden: Path | None
    renamed: bool


def _unresolved_rooted(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    rooted = candidate if candidate.is_absolute() else PROJECT_ROOT / candidate
    return Path(os.path.abspath(rooted))


def _reject_symlink_components(path: Path, label: str) -> None:
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current /= component
        if current.is_symlink():
            raise ValueError(f"{label} path contains a symbolic-link component: {current}")


@contextmanager
def training_root_temporarily_unavailable(
    training_directory: str | Path,
) -> Iterator[TrainingRootIsolation]:
    """Atomically hide one exact directory named ``training`` and restore it."""

    original = _unresolved_rooted(training_directory)
    _reject_symlink_components(original, "Training isolation")
    if original.name.casefold() != "training":
        raise ValueError("Training isolation accepts only an exact directory named training")
    if original.exists() and not original.is_dir():
        raise ValueError(f"Training root is not a directory: {original}")
    if not original.exists():
        yield TrainingRootIsolation(original=original, hidden=None, renamed=False)
        return
    hidden = original.with_name(
        f".training-unavailable-{os.getpid()}-{uuid.uuid4().hex}"
    )
    if hidden.exists():
        raise FileExistsError(f"Temporary training destination exists: {hidden}")
    original.rename(hidden)
    try:
        yield TrainingRootIsolation(original=original, hidden=hidden, renamed=True)
    finally:
        if original.exists():
            raise RuntimeError(
                "Cannot restore training artifacts because the original path was "
                f"recreated; preserved original data at {hidden}"
            )
        hidden.rename(original)


def _path_is_within(path: str | Path, root: Path) -> bool:
    try:
        Path(path).expanduser().resolve().relative_to(root.resolve())
    except (OSError, RuntimeError, ValueError):
        return False
    return True


def run_fixed_prefix_atlas_leakage(
    *,
    config_path: str | Path,
    scene_id: str,
    base_checkpoint: str | Path,
    atlas_checkpoint: str | Path,
    training_directory: str | Path,
    questions: Sequence[str] = DEFAULT_QUESTIONS,
    report_path: str | Path,
) -> dict[str, Any]:
    """Hide oracle and training artifacts while exercising immutable atlas chat."""

    atlas = _unresolved_rooted(atlas_checkpoint).resolve()
    training = _unresolved_rooted(training_directory)
    output = _unresolved_rooted(report_path)
    runtime_instances: list[Any] = []
    isolation: TrainingRootIsolation | None = None

    def load_runtime(
        config: dict[str, Any],
        selected_scene: str,
        selected_base: str | Path,
        audit: Any,
    ) -> Any:
        from semantic_3d_chat.chat.fixed_prefix_runtime import (
            FixedPrefixAtlasChatRuntime,
        )

        if training not in audit.forbidden_roots:
            audit.forbidden_roots.append(training)
        if (
            isolation is not None
            and isolation.hidden is not None
            and isolation.hidden not in audit.forbidden_roots
        ):
            audit.forbidden_roots.append(isolation.hidden)
        runtime = FixedPrefixAtlasChatRuntime.load(
            config,
            selected_scene,
            base_checkpoint=selected_base,
            atlas_checkpoint=atlas,
            audit=audit,
            local_files_only=True,
        )
        runtime_instances.append(runtime)
        return runtime

    with training_root_temporarily_unavailable(training) as state:
        isolation = state
        training_unavailable = not state.original.exists()
        report = run_leakage_evaluation(
            config_path=config_path,
            scene_id=scene_id,
            checkpoint=base_checkpoint,
            questions=questions,
            report_path=output,
            runtime_loader=load_runtime,
            require_strict_fixed_environment_input=True,
        )

    if len(runtime_instances) != 1:
        raise RuntimeError("Fixed-atlas leakage runtime did not load exactly once")
    training_restored = bool(
        isolation is not None
        and (
            isolation.original.is_dir()
            if isolation.renamed
            else not isolation.original.exists()
        )
    )
    expected_atlas_files = [
        str((atlas / name).resolve())
        for name in ("atlas.safetensors", "runtime_metadata.json")
    ]
    loaded_files = set(report["loaded_files"])
    loaded_atlas_files = [path for path in expected_atlas_files if path in loaded_files]
    isolated_roots = [training]
    if isolation is not None and isolation.hidden is not None:
        isolated_roots.append(isolation.hidden)
    training_loaded_paths = [
        loaded
        for loaded in report["loaded_files"]
        if any(_path_is_within(loaded, root) for root in isolated_roots)
    ]
    report.update(
        {
            "schema": "semantic_3d_chat.fixed_prefix_atlas_leakage.v1",
            "runtime_kind": "strict_fixed_continuous_scene_atlas",
            "atlas_checkpoint": str(atlas),
            "atlas_checkpoint_files_expected": expected_atlas_files,
            "atlas_checkpoint_files_loaded": loaded_atlas_files,
            "atlas_checkpoint_files_complete": (
                loaded_atlas_files == expected_atlas_files
            ),
            "training_directory": str(training),
            "training_directory_was_renamed": bool(isolation and isolation.renamed),
            "training_directory_unavailable_during_inference": training_unavailable,
            "training_directory_restored": training_restored,
            "training_artifact_loaded_paths": training_loaded_paths,
            "question_dependent_scene_processing": False,
            "question_dependent_retrieval": False,
            "strict_fixed_environment_embedding_input": True,
        }
    )
    report["passed"] = bool(
        report["passed"]
        and report["atlas_checkpoint_files_complete"]
        and report["training_directory_unavailable_during_inference"]
        and report["training_directory_restored"]
        and not report["training_artifact_loaded_paths"]
        and report["prefix_invariant"]
        and report["prefix_computed_before_first_question"]
        and report["oracle_unavailable_during_inference"]
        and report["oracle_restored"]
    )
    _atomic_json(output, report)
    if not report["passed"]:
        raise RuntimeError(f"Fixed-atlas leakage audit failed: {output}")
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--atlas-checkpoint", required=True)
    parser.add_argument(
        "--training-directory", default="data_gemma4/training"
    )
    parser.add_argument("--question", action="append")
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = run_fixed_prefix_atlas_leakage(
        config_path=args.config,
        scene_id=args.scene,
        base_checkpoint=args.base_checkpoint,
        atlas_checkpoint=args.atlas_checkpoint,
        training_directory=args.training_directory,
        questions=args.question or DEFAULT_QUESTIONS,
        report_path=args.output,
    )
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "oracle_unavailable_during_inference": report[
                    "oracle_unavailable_during_inference"
                ],
                "training_directory_unavailable_during_inference": report[
                    "training_directory_unavailable_during_inference"
                ],
                "prefix_invariant": report["prefix_invariant"],
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


__all__ = [
    "TrainingRootIsolation",
    "run_fixed_prefix_atlas_leakage",
    "training_root_temporarily_unavailable",
]
