"""Generate questions-only predictions from the strict fixed-prefix runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

from semantic_3d_chat.chat.fixed_prefix_runtime import FixedPrefixAtlasChatRuntime
from semantic_3d_chat.chat.runtime_config import load_runtime_config
from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.evaluation.baseline_io import sha256_file
from semantic_3d_chat.evaluation.prediction_artifacts import (
    AtomicPredictionJournal,
    PredictionProvenance,
    build_scene_map_manifest,
    checkpoint_fingerprint,
    effective_config_sha256,
    scene_map_manifest_sha256,
)
from semantic_3d_chat.evaluation.question_manifest import load_question_manifest
from semantic_3d_chat.training.fixed_prefix_atlas_checkpoint import (
    two_file_checkpoint_fingerprint,
)


def _rooted(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    rooted = candidate if candidate.is_absolute() else PROJECT_ROOT / candidate
    return Path(os.path.abspath(rooted))


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _provenance(
    *,
    config: dict,
    config_path: Path,
    base_checkpoint: Path,
    atlas_checkpoint: Path,
    questions_path: Path,
    scene_ids: list[str],
    split: str,
) -> PredictionProvenance:
    base_sha256, base_files = checkpoint_fingerprint(base_checkpoint)
    atlas_sha256, atlas_files = two_file_checkpoint_fingerprint(atlas_checkpoint)
    combined = {
        "base_checkpoint_sha256": base_sha256,
        "atlas_checkpoint_sha256": atlas_sha256,
    }
    files = tuple(
        [
            {**entry, "path": f"base/{entry['path']}"}
            for entry in base_files
        ]
        + [
            {
                "path": f"atlas/{name}",
                "sha256": digest,
                "size_bytes": (atlas_checkpoint / name).stat().st_size,
            }
            for name, digest in sorted(atlas_files.items())
        ]
    )
    maps = build_scene_map_manifest(config, scene_ids)
    return PredictionProvenance(
        config_path=str(config_path),
        config_sha256=effective_config_sha256(config),
        config_file_sha256=sha256_file(config_path),
        checkpoint_path=f"{base_checkpoint} + {atlas_checkpoint}",
        checkpoint_sha256=_canonical_sha256(combined),
        checkpoint_files=files,
        references_path=str(questions_path),
        references_sha256=sha256_file(questions_path),
        scene_map_manifest_sha256=scene_map_manifest_sha256(maps),
        scene_map_manifest=maps,
        split=split,
        run_kind="strict_fixed_continuous_scene_atlas",
        condition="same_complete_prefix_every_question",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/runtime/gemma4_v54.yaml")
    parser.add_argument("--split", default="validation", choices=("train", "validation", "test"))
    parser.add_argument("--questions-manifest", type=Path, required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--atlas-checkpoint", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-questions-per-scene", type=int)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args(argv)
    if args.max_questions_per_scene is not None and args.max_questions_per_scene < 1:
        raise ValueError("max-questions-per-scene must be positive")

    config_path = _rooted(args.config)
    config = load_runtime_config(config_path)
    questions = load_question_manifest(_rooted(args.questions_manifest))
    by_scene = questions.by_scene()
    base_checkpoint = _rooted(args.base_checkpoint)
    atlas_checkpoint = _rooted(args.atlas_checkpoint)
    provenance = _provenance(
        config=config,
        config_path=config_path,
        base_checkpoint=base_checkpoint,
        atlas_checkpoint=atlas_checkpoint,
        questions_path=questions.manifest_path,
        scene_ids=sorted(by_scene),
        split=args.split,
    )
    journal = AtomicPredictionJournal(
        _rooted(args.output), provenance, resume=not args.no_resume
    )
    initial_count = len(journal.records)
    prefix_hashes: dict[str, str] = {}
    started = time.perf_counter()
    for scene_id, records in sorted(by_scene.items()):
        selected = (
            records[: args.max_questions_per_scene]
            if args.max_questions_per_scene is not None
            else records
        )
        pending = [
            row for row in selected if not journal.contains(scene_id, row.question_id)
        ]
        if not pending:
            cached = {
                str(row["prefix_hash"])
                for row in journal.records
                if row.get("scene_id") == scene_id
            }
            if len(cached) == 1:
                prefix_hashes[scene_id] = cached.pop()
            continue
        runtime = FixedPrefixAtlasChatRuntime.load(
            config,
            scene_id,
            base_checkpoint=base_checkpoint,
            atlas_checkpoint=atlas_checkpoint,
            local_files_only=True,
        )
        if runtime.questions_answered != 0:
            raise RuntimeError("Fixed scene prefix was not compiled before evaluation")
        prefix_hashes[scene_id] = runtime.scene_prefix_hash
        for row in pending:
            answer = runtime.answer(row.question)
            if answer.prefix_hash != runtime.scene_prefix_hash:
                raise RuntimeError("Question changed the fixed scene prefix")
            journal.append(
                {
                    "scene_id": scene_id,
                    "question_id": row.question_id,
                    "predicted_answer": answer.answer,
                    "grounding_xyz": list(answer.grounding_xyz_m),
                    "grounding_confidence": answer.grounding_confidence,
                    "prefix_hash": answer.prefix_hash,
                    "generated_tokens": answer.generated_tokens,
                    "elapsed_seconds": answer.elapsed_seconds,
                    "question_dependent_scene_processing": False,
                    "language_model_environment_conditioning_question_dependent": False,
                    "auxiliary_grounding_question_conditioned": True,
                    "auxiliary_grounding_affects_language_model": False,
                }
            )
        runtime.assert_prefix_unchanged()
    final_count = len(journal.records)
    print(
        json.dumps(
            {
                "phase": "strict_fixed_prefix_prediction_complete",
                "predictions": final_count,
                "new_predictions": final_count - initial_count,
                "scene_count": len(prefix_hashes),
                "prefix_hashes": prefix_hashes,
                "same_prefix_every_question": True,
                "question_dependent_scene_processing": False,
                "question_dependent_retrieval": False,
                "elapsed_seconds": time.perf_counter() - started,
                "output": str(_rooted(args.output)),
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
