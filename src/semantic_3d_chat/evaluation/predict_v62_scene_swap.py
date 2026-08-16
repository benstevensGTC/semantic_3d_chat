"""Inference-only blind scene-prefix swap for the V62 control population.

Every one of the 384 questions is asked with the continuous prefix of its
opaque paired scene.  The process never loads route labels, expected answers,
change types, QA files, oracle files, or the V62 scorer.  Scoring later selects
the 52 changed sides behind the one-shot terminal boundary.

Before creating an output journal, the process authenticates that the control
artifact is the sealed schema-6 V65 magnitude-gated checkpoint whose public
metadata says its saved-runtime training gate passed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Final

from semantic_3d_chat.chat.question_control_runtime import QuestionControlledChatRuntime
from semantic_3d_chat.chat.runtime_config import (
    effective_runtime_config_sha256,
    load_runtime_config,
)
from semantic_3d_chat.evaluation.predict_question_control import (
    _cached_prefix_hashes,
)
from semantic_3d_chat.evaluation.prediction_artifacts import (
    AtomicPredictionJournal,
    build_prediction_provenance,
    checkpoint_fingerprint,
)
from semantic_3d_chat.evaluation.question_manifest import load_question_manifest
from semantic_3d_chat.evaluation.v65_candidate_contract import (
    validate_sealed_v65_checkpoint,
)

RUN_KIND: Final[str] = "continuous_scene_question_control_scene_swap_v1"
CONDITION_PREFIX: Final[str] = "all_questions_bidirectional_scene_swap"

# Opaque evaluation pairing only.  No semantic type, route label, answer, or
# relationship is present in this inference module.
V62_INTERNAL_SCENE_PAIRS: Final[tuple[tuple[str, str], ...]] = (
    ("scene_000039", "scene_000040"),
    ("scene_000041", "scene_000042"),
    ("scene_000043", "scene_000044"),
    ("scene_000045", "scene_000046"),
    ("scene_000047", "scene_000048"),
    ("scene_000049", "scene_000050"),
    ("scene_000051", "scene_000052"),
    ("scene_000055", "scene_000056"),
)


def paired_scene_ids() -> dict[str, str]:
    result: dict[str, str] = {}
    for first, second in V62_INTERNAL_SCENE_PAIRS:
        result[first] = second
        result[second] = first
    if len(result) != 16 or any(result.get(value) != key for key, value in result.items()):
        raise AssertionError("V62 opaque scene pairing is not reciprocal")
    return result


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--questions-manifest", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--control-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="train", choices=("train", "validation"))
    parser.add_argument("--no-resume", action="store_true")
    destinations = {action.dest for action in parser._actions}
    if destinations & {
        "scorer_references",
        "qa",
        "oracle",
        "route_labels",
        "changed_questions",
    }:
        raise AssertionError("V62 scene-swap runner exposes a forbidden input")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    config = load_runtime_config(args.config)
    manifest = load_question_manifest(args.questions_manifest)
    pairing = paired_scene_ids()
    by_scene = manifest.by_scene()
    if set(by_scene) != set(pairing) or any(len(rows) != 24 for rows in by_scene.values()):
        raise ValueError("V62 blind scene-swap requires the exact 16-scene/384-question manifest")

    base_checkpoint = args.base_checkpoint.expanduser().resolve()
    control_checkpoint = args.control_checkpoint.expanduser().resolve()
    base_checkpoint_sha256, _base_files = checkpoint_fingerprint(base_checkpoint)
    sealed_control = validate_sealed_v65_checkpoint(
        control_checkpoint,
        base_checkpoint_sha256=base_checkpoint_sha256,
        runtime_config_sha256=effective_runtime_config_sha256(config),
    )
    control_sha256 = sealed_control.fingerprint_sha256
    condition = f"{CONDITION_PREFIX};control_checkpoint_sha256={control_sha256}"
    provenance = build_prediction_provenance(
        config,
        config_path=args.config,
        checkpoint_path=base_checkpoint,
        references_path=manifest.manifest_path,
        scene_ids=sorted(by_scene),
        split=args.split,
        run_kind=RUN_KIND,
        condition=condition,
    )
    journal = AtomicPredictionJournal(
        args.output.expanduser().resolve(),
        provenance,
        resume=not args.no_resume,
    )
    initial_count = len(journal.records)
    started = time.perf_counter()
    prefix_hashes: dict[str, str] = {}
    for source_scene_id, questions in by_scene.items():
        injected_scene_id = pairing[source_scene_id]
        pending = [
            question
            for question in questions
            if not journal.contains(source_scene_id, question.question_id)
        ]
        if not pending:
            hashes = _cached_prefix_hashes(list(journal.records), source_scene_id)
            if len(hashes) != 1:
                raise RuntimeError("Cached V62 swap predictions lack one injected prefix")
            prefix_hashes[source_scene_id] = next(iter(hashes))
            continue
        runtime = QuestionControlledChatRuntime.load(
            config,
            injected_scene_id,
            base_checkpoint=base_checkpoint,
            control_checkpoint=control_checkpoint,
        )
        prefix_hashes[source_scene_id] = runtime.scene_prefix_hash
        cached = _cached_prefix_hashes(list(journal.records), source_scene_id)
        if cached and cached != {runtime.scene_prefix_hash}:
            raise RuntimeError("Cached V62 scene-swap prefix changed")
        for question in pending:
            answer = runtime.answer(question.question)
            if runtime.last_control_audit is None:
                raise RuntimeError("V62 scene-swap answer lacks a continuous-control audit")
            journal.append(
                {
                    "scene_id": source_scene_id,
                    "question_id": question.question_id,
                    "injected_scene_id": injected_scene_id,
                    "predicted_answer": answer.answer,
                    "grounding_xyz": list(answer.grounding_xyz_m),
                    "grounding_confidence": answer.grounding_confidence,
                    "prefix_hash": answer.prefix_hash,
                    "question_sha256": hashlib.sha256(
                        question.question.encode("utf-8")
                    ).hexdigest(),
                    "generated_tokens": answer.generated_tokens,
                    "elapsed_seconds": answer.elapsed_seconds,
                    "control_checkpoint_sha256": control_sha256,
                    "control_audit": dict(runtime.last_control_audit),
                }
            )
            print(
                json.dumps(
                    {
                        "phase": "v62_blind_scene_swap_inference",
                        "source_scene": source_scene_id,
                        "completed": len(journal.records),
                    }
                ),
                flush=True,
            )
        runtime.assert_prefix_unchanged()

    expected_keys = {
        (scene_id, question.question_id)
        for scene_id, questions in by_scene.items()
        for question in questions
    }
    if journal.completed_keys != expected_keys or len(journal.records) != 384:
        raise RuntimeError("V62 blind scene-swap journal is incomplete")
    print(
        json.dumps(
            {
                "phase": "v62_blind_scene_swap_complete",
                "predictions": len(journal.records),
                "new_predictions": len(journal.records) - initial_count,
                "scene_count": len(prefix_hashes),
                "control_checkpoint_sha256": control_sha256,
                "prediction_provenance_sha256": provenance.sha256,
                "elapsed_seconds": time.perf_counter() - started,
                "route_labels_loaded": False,
                "scorer_references_loaded": False,
                "qa_or_oracle_loaded": False,
                "output": str(args.output),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["CONDITION_PREFIX", "RUN_KIND", "V62_INTERNAL_SCENE_PAIRS", "main", "paired_scene_ids"]
