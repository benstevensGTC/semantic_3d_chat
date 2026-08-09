"""Generate evaluation predictions from continuous scene prefixes.

This driver reads a strict questions-only manifest after each scene runtime has
built its question-independent prefix. It never opens QA supervision. Reference
answers and oracle targets are joined to predictions only by the separate scorer.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from semantic_3d_chat.chat.runtime import StaticChatRuntime
from semantic_3d_chat.config import default_checkpoint_path, load_config, reports_root
from semantic_3d_chat.evaluation.prediction_artifacts import (
    AtomicPredictionJournal,
    build_prediction_provenance,
)
from semantic_3d_chat.evaluation.question_manifest import load_question_manifest


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--split", default="test", choices=("train", "validation", "test"))
    parser.add_argument(
        "--questions-manifest",
        type=Path,
        help="Strict manifest produced by semantic_3d_chat.evaluation.prepare_questions",
    )
    parser.add_argument("--checkpoint")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-questions-per-scene", type=int)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.max_questions_per_scene is not None and args.max_questions_per_scene < 1:
        raise ValueError("max_questions_per_scene must be positive")
    config = load_config(args.config)
    questions_path = (
        args.questions_manifest
        or reports_root(config) / "questions" / f"{args.split}.json"
    )
    question_manifest = load_question_manifest(questions_path)
    by_scene = question_manifest.by_scene()
    output = args.output or reports_root(config) / "predictions" / f"{args.split}.jsonl"
    output = output.resolve()
    checkpoint = Path(
        args.checkpoint or default_checkpoint_path(config)
    ).expanduser().resolve()
    provenance = build_prediction_provenance(
        config,
        config_path=args.config,
        checkpoint_path=checkpoint,
        references_path=question_manifest.manifest_path,
        split=args.split,
        run_kind="continuous_scene_static",
        condition=(
            "all_questions"
            if args.max_questions_per_scene is None
            else f"max_questions_per_scene={args.max_questions_per_scene}"
        ),
    )
    journal = AtomicPredictionJournal(output, provenance, resume=not args.no_resume)
    started = time.perf_counter()
    initial_count = len(journal.records)
    count = initial_count
    prefix_hashes: dict[str, str] = {}
    for scene_id, scene_records in sorted(by_scene.items()):
        selected = (
            scene_records[: args.max_questions_per_scene]
            if args.max_questions_per_scene
            else scene_records
        )
        pending = [
            record
            for record in selected
            if not journal.contains(scene_id, record.question_id)
        ]
        if not pending:
            cached_hashes = {
                str(record["prefix_hash"])
                for record in journal.records
                if record.get("scene_id") == scene_id and record.get("prefix_hash")
            }
            if len(cached_hashes) == 1:
                prefix_hashes[scene_id] = cached_hashes.pop()
            continue
        runtime = StaticChatRuntime.load(
            config,
            scene_id,
            checkpoint=checkpoint,
            local_files_only=True,
        )
        prefix_hashes[scene_id] = runtime.scene_prefix_hash
        cached_hashes = {
            str(record["prefix_hash"])
            for record in journal.records
            if record.get("scene_id") == scene_id and record.get("prefix_hash")
        }
        if cached_hashes and cached_hashes != {runtime.scene_prefix_hash}:
            raise RuntimeError(f"Cached prefix hash differs for static scene {scene_id}")
        for record in pending:
            answer = runtime.answer(record.question)
            prediction = {
                "scene_id": scene_id,
                "question_id": record.question_id,
                "predicted_answer": answer.answer,
                "grounding_xyz": list(answer.grounding_xyz_m),
                "grounding_confidence": answer.grounding_confidence,
                "prefix_hash": answer.prefix_hash,
                "generated_tokens": answer.generated_tokens,
                "elapsed_seconds": answer.elapsed_seconds,
            }
            journal.append(prediction)
            count += 1
            print(
                json.dumps(
                    {
                        "phase": "evaluation_inference",
                        "scene": scene_id,
                        "completed": count,
                        "answer": answer.answer,
                    }
                ),
                flush=True,
            )
        runtime.assert_prefix_unchanged()
    print(
        json.dumps(
            {
                "phase": "evaluation_inference_complete",
                "split": args.split,
                "predictions": count,
                "resumed_predictions": initial_count,
                "new_predictions": count - initial_count,
                "scene_count": len(prefix_hashes),
                "prefix_hashes": prefix_hashes,
                "questions_manifest_path": str(question_manifest.manifest_path),
                "questions_manifest_sha256": question_manifest.manifest_sha256,
                "questions_sha256": question_manifest.questions_sha256,
                "prediction_provenance_sha256": provenance.sha256,
                "elapsed_seconds": time.perf_counter() - started,
                "output": str(output),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
