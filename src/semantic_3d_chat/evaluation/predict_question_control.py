"""Inference-only prediction with a fixed scene prefix and continuous control head.

This process accepts only a sanitized questions manifest.  It never opens QA
answers, oracle metadata, captions, labels, or scene-generation files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from collections.abc import Sequence
from pathlib import Path

from semantic_3d_chat.chat.question_control_runtime import (
    QuestionControlledChatRuntime,
)
from semantic_3d_chat.chat.runtime_config import load_runtime_config
from semantic_3d_chat.evaluation.prediction_artifacts import (
    AtomicPredictionJournal,
    build_prediction_provenance,
)
from semantic_3d_chat.evaluation.question_manifest import load_question_manifest

_SHA256 = re.compile(r"[0-9a-f]{64}")


def _prediction_condition(
    max_questions_per_scene: int | None,
    control_checkpoint_sha256: str,
) -> str:
    if _SHA256.fullmatch(control_checkpoint_sha256) is None:
        raise ValueError("Control checkpoint fingerprint must be lowercase SHA-256")
    selection = (
        "all_questions"
        if max_questions_per_scene is None
        else f"max_questions_per_scene={max_questions_per_scene}"
    )
    return f"{selection};control_checkpoint_sha256={control_checkpoint_sha256}"


def _cached_prefix_hashes(
    records: Sequence[dict[str, object]], scene_id: str
) -> set[str]:
    scene_records = [record for record in records if record.get("scene_id") == scene_id]
    hashes: set[str] = set()
    for record in scene_records:
        value = record.get("prefix_hash")
        if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
            raise RuntimeError(f"Cached prediction has an invalid prefix hash for {scene_id}")
        hashes.add(value)
    return hashes


def _control_checkpoint_sha256(path: str | Path) -> str:
    source = Path(os.path.abspath(Path(path).expanduser()))
    current = Path(source.anchor)
    for component in source.parts[1:]:
        current /= component
        if current.is_symlink():
            raise ValueError(
                f"Control checkpoint path must not contain symlinks: {current}"
            )
    if {"oracle", "qa"}.intersection(component.casefold() for component in source.parts):
        raise ValueError("Control checkpoint must be separate from QA/oracle data")
    if not source.is_dir():
        raise FileNotFoundError(f"Control checkpoint is unavailable: {source}")
    expected = ("control.safetensors", "runtime_metadata.json")
    if sorted(item.name for item in source.iterdir()) != sorted(expected):
        raise ValueError("Control checkpoint inventory is not runtime-minimal")
    entries = []
    for name in expected:
        item = source / name
        if item.is_symlink() or not item.is_file():
            raise ValueError(f"Control checkpoint entry is not a regular file: {item}")
        digest = hashlib.sha256(item.read_bytes()).hexdigest()
        entries.append({"name": name, "sha256": digest, "size_bytes": item.stat().st_size})
    payload = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--questions-manifest", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--control-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="validation", choices=("train", "validation", "test"))
    parser.add_argument("--max-questions-per-scene", type=int)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.max_questions_per_scene is not None and args.max_questions_per_scene < 1:
        raise ValueError("max_questions_per_scene must be positive")
    config = load_runtime_config(args.config)
    manifest = load_question_manifest(args.questions_manifest)
    by_scene = manifest.by_scene()
    base_checkpoint = args.base_checkpoint.expanduser().resolve()
    control_checkpoint = args.control_checkpoint.expanduser().resolve()
    control_sha256 = _control_checkpoint_sha256(control_checkpoint)
    provenance = build_prediction_provenance(
        config,
        config_path=args.config,
        checkpoint_path=base_checkpoint,
        references_path=manifest.manifest_path,
        scene_ids=sorted(by_scene),
        split=args.split,
        run_kind="continuous_scene_question_control_v1",
        condition=_prediction_condition(args.max_questions_per_scene, control_sha256),
    )
    journal = AtomicPredictionJournal(
        args.output.expanduser().resolve(),
        provenance,
        resume=not args.no_resume,
    )
    initial_count = len(journal.records)
    started = time.perf_counter()
    prefix_hashes: dict[str, str] = {}
    for scene_id, records in sorted(by_scene.items()):
        selected = (
            records[: args.max_questions_per_scene]
            if args.max_questions_per_scene is not None
            else records
        )
        pending = [
            record
            for record in selected
            if not journal.contains(scene_id, record.question_id)
        ]
        if not pending:
            hashes = _cached_prefix_hashes(list(journal.records), scene_id)
            if len(hashes) != 1:
                raise RuntimeError(
                    f"Cached predictions do not prove one static prefix for {scene_id}"
                )
            prefix_hashes[scene_id] = next(iter(hashes))
            continue
        runtime = QuestionControlledChatRuntime.load(
            config,
            scene_id,
            base_checkpoint=base_checkpoint,
            control_checkpoint=control_checkpoint,
        )
        prefix_hashes[scene_id] = runtime.scene_prefix_hash
        cached_hashes = _cached_prefix_hashes(list(journal.records), scene_id)
        if cached_hashes and cached_hashes != {runtime.scene_prefix_hash}:
            raise RuntimeError(
                f"Cached prefix hash differs for question-controlled scene {scene_id}"
            )
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
                    "control_checkpoint_sha256": control_sha256,
                }
            if runtime.last_control_audit is not None:
                prediction["control_audit"] = dict(runtime.last_control_audit)
            journal.append(prediction)
            print(
                json.dumps(
                    {
                        "phase": "question_control_inference",
                        "scene": scene_id,
                        "completed": len(journal.records),
                    }
                ),
                flush=True,
            )
        runtime.assert_prefix_unchanged()
    if len(prefix_hashes) != len(by_scene):
        raise RuntimeError("Question-control inference did not attest every scene prefix")
    expected_keys = {
        (scene_id, record.question_id)
        for scene_id, records in by_scene.items()
        for record in (
            records[: args.max_questions_per_scene]
            if args.max_questions_per_scene is not None
            else records
        )
    }
    if journal.completed_keys != expected_keys:
        raise RuntimeError("Prediction journal does not exactly match selected question keys")
    summary = {
        "phase": "question_control_inference_complete",
        "split": args.split,
        "predictions": len(journal.records),
        "new_predictions": len(journal.records) - initial_count,
        "scene_count": len(prefix_hashes),
        "prefix_hashes": prefix_hashes,
        "control_checkpoint_sha256": control_sha256,
        "prediction_provenance_sha256": provenance.sha256,
        "elapsed_seconds": time.perf_counter() - started,
        "output": str(args.output),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
