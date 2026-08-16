"""Blind V66 schema-7 natural and paired-scene-swap inference.

The runner accepts only the sanitized questions manifest, runtime config,
frozen base adapter, and sealed always-on V7 control checkpoint.  It never
accepts or opens answers, route labels, scorer references, QA, oracle, fresh
development, or deferred-final data.  Each row records both the immutable
full-scene prefix hash and the cached all-256-latent scene-signature hash.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import torch

from semantic_3d_chat.chat.question_control_runtime import (
    QuestionControlledChatRuntime,
    _load_control_head,
)
from semantic_3d_chat.chat.runtime_config import (
    effective_runtime_config_sha256,
    load_runtime_config,
)
from semantic_3d_chat.evaluation.predict_question_control import (
    _control_checkpoint_sha256,
)
from semantic_3d_chat.evaluation.prediction_artifacts import (
    AtomicPredictionJournal,
    build_prediction_provenance,
    checkpoint_fingerprint,
)
from semantic_3d_chat.evaluation.question_manifest import (
    QuestionManifest,
    QuestionRecord,
    load_question_manifest,
)
from semantic_3d_chat.scene_encoder.question_control_v7 import (
    AlwaysOnTeacherBasisFullSceneQuestionControlV7,
)
from semantic_3d_chat.training.question_control_v7_checkpoint import (
    v7_value_state_sha256,
)

NATURAL_RUN_KIND: Final[str] = "continuous_scene_question_control_v1"
SWAP_RUN_KIND: Final[str] = "continuous_scene_question_control_scene_swap_v1"
NATURAL_CONDITION_PREFIX: Final[str] = "all_questions"
SWAP_CONDITION_PREFIX: Final[str] = "all_questions_bidirectional_scene_swap"
ARCHITECTURE: Final[str] = "always_on_teacher_basis_full_scene_control_v7"
INTERNAL_SCENE_PAIRS: Final[tuple[tuple[str, str], ...]] = (
    ("scene_000039", "scene_000040"),
    ("scene_000041", "scene_000042"),
    ("scene_000043", "scene_000044"),
    ("scene_000045", "scene_000046"),
    ("scene_000047", "scene_000048"),
    ("scene_000049", "scene_000050"),
    ("scene_000051", "scene_000052"),
    ("scene_000055", "scene_000056"),
)
_FORBIDDEN_PATH_PARTS: Final[frozenset[str]] = frozenset(
    {
        "oracle",
        "qa",
        "scorer_only",
        "scorer-only",
        "fresh_development",
        "deferred_final",
    }
)


def paired_scene_ids() -> dict[str, str]:
    result: dict[str, str] = {}
    for first, second in INTERNAL_SCENE_PAIRS:
        result[first] = second
        result[second] = first
    if len(result) != 16 or any(result.get(value) != key for key, value in result.items()):
        raise AssertionError("V66 opaque internal scene pairing is not reciprocal")
    return result


def _safe_manifest_path(path: str | Path) -> Path:
    source = Path(path).expanduser().resolve()
    parts = {part.casefold() for part in source.parts}
    if parts & _FORBIDDEN_PATH_PARTS:
        raise ValueError("V66 inference refuses QA/oracle/scorer/protected paths")
    if not source.is_file() or source.is_symlink():
        raise FileNotFoundError(f"V66 questions-only manifest is unavailable: {source}")
    return source


def _validate_manifest(manifest: QuestionManifest) -> dict[str, tuple[QuestionRecord, ...]]:
    by_scene = manifest.by_scene()
    if (
        set(by_scene) != set(paired_scene_ids())
        or manifest.question_count != 384
        or manifest.scene_count != 16
        or any(len(rows) != 24 for rows in by_scene.values())
    ):
        raise ValueError("V66 inference requires the exact 16-scene/384-question manifest")
    return by_scene


def _authenticate_sealed_v7(
    checkpoint: Path,
    *,
    base_checkpoint_sha256: str,
    runtime_config_sha256: str,
) -> tuple[str, dict[str, Any]]:
    fingerprint = _control_checkpoint_sha256(checkpoint)
    control, metadata = _load_control_head(
        checkpoint,
        hidden_size=1536,
        device=torch.device("cpu"),
    )
    if (
        type(control) is not AlwaysOnTeacherBasisFullSceneQuestionControlV7
        or metadata.get("architecture") != ARCHITECTURE
        or metadata.get("schema_version") != 7
        or metadata.get("base_checkpoint_sha256") != base_checkpoint_sha256
        or metadata.get("base_runtime_config_sha256") != runtime_config_sha256
        or metadata.get("expected_environment_latents") != 256
        or metadata.get("control_tokens") != 4
        or metadata.get("always_on_continuous_control") is not True
        or metadata.get("complete_scene_prefix_required") is not True
        or metadata.get("question_dependent_scene_retrieval") is not False
        or metadata.get("environmental_text_inputs") != []
        or metadata.get("training_answers_runtime_loaded") is not False
        or metadata.get("answer_class_codebook_runtime_loaded") is not False
        or metadata.get("saved_runtime_training_gate_required") is not True
        or metadata.get("saved_runtime_training_gate_passed") is not True
        or v7_value_state_sha256(control) != metadata.get("source_v66_training_fit_state_sha256")
    ):
        raise ValueError("V66 inference requires the exact sealed schema-7 checkpoint")
    return fingerprint, dict(metadata)


def _cached_scene_identities(
    records: Sequence[Mapping[str, Any]], scene_id: str
) -> tuple[set[str], set[str]]:
    prefixes: set[str] = set()
    signatures: set[str] = set()
    for record in records:
        if record.get("scene_id") != scene_id:
            continue
        prefix = record.get("prefix_hash")
        signature = record.get("scene_control_signature_sha256")
        if (
            not isinstance(prefix, str)
            or len(prefix) != 64
            or not isinstance(signature, str)
            or len(signature) != 64
        ):
            raise RuntimeError(f"Cached V66 identity is invalid for {scene_id}")
        prefixes.add(prefix)
        signatures.add(signature)
    return prefixes, signatures


def _runtime_identities(runtime: QuestionControlledChatRuntime) -> tuple[str, str]:
    runtime.assert_prefix_unchanged()
    signature = runtime.scene_control_signature_hash
    if not isinstance(signature, str) or len(signature) != 64:
        raise RuntimeError("V66 runtime lacks its pre-question full-scene signature hash")
    return runtime.scene_prefix_hash, signature


def _prediction_record(
    *,
    source_scene_id: str,
    question: QuestionRecord,
    injected_scene_id: str,
    runtime: QuestionControlledChatRuntime,
    swap: bool,
) -> dict[str, Any]:
    prefix_before, signature_before = _runtime_identities(runtime)
    answer = runtime.answer(question.question)
    prefix_after, signature_after = _runtime_identities(runtime)
    if prefix_before != prefix_after or signature_before != signature_after:
        raise RuntimeError("V66 prefix or scene signature changed while answering")
    if runtime.last_control_audit is None:
        raise RuntimeError("V66 answer lacks an always-on continuous-control audit")
    audit = dict(runtime.last_control_audit)
    if (
        audit.get("architecture") != ARCHITECTURE
        or audit.get("environment_latent_count") != 256
        or audit.get("every_scene_token_influenced_output") is not True
        or audit.get("question_dependent_scene_retrieval") is not False
        or audit.get("control_used") is not True
        or audit.get("always_on_continuous_control") is not True
    ):
        raise RuntimeError("V66 runtime audit violates the schema-7 contract")
    row: dict[str, Any] = {
        "scene_id": source_scene_id,
        "question_id": question.question_id,
        "predicted_answer": answer.answer,
        "grounding_xyz": list(answer.grounding_xyz_m),
        "grounding_confidence": answer.grounding_confidence,
        "prefix_hash": prefix_after,
        "scene_control_signature_sha256": signature_after,
        "generated_tokens": answer.generated_tokens,
        "elapsed_seconds": answer.elapsed_seconds,
        "control_audit": audit,
    }
    if swap:
        row.update(
            {
                "injected_scene_id": injected_scene_id,
                "question_sha256": hashlib.sha256(question.question.encode("utf-8")).hexdigest(),
            }
        )
    return row


def _run(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = _safe_manifest_path(args.questions_manifest)
    config = load_runtime_config(args.config)
    manifest = load_question_manifest(manifest_path)
    by_scene = _validate_manifest(manifest)
    pairing = paired_scene_ids()
    base_checkpoint = Path(args.base_checkpoint).expanduser().resolve()
    control_checkpoint = Path(args.control_checkpoint).expanduser().resolve()
    base_sha256, _base_files = checkpoint_fingerprint(base_checkpoint)

    # Authenticate the sealed schema-7 adapter before creating an output
    # journal.  A rejected checkpoint cannot leave a reusable prediction file.
    control_sha256, _metadata = _authenticate_sealed_v7(
        control_checkpoint,
        base_checkpoint_sha256=base_sha256,
        runtime_config_sha256=effective_runtime_config_sha256(config),
    )
    swap = args.mode == "swap"
    condition_prefix = SWAP_CONDITION_PREFIX if swap else NATURAL_CONDITION_PREFIX
    provenance = build_prediction_provenance(
        config,
        config_path=args.config,
        checkpoint_path=base_checkpoint,
        references_path=manifest_path,
        scene_ids=sorted(by_scene),
        split=args.split,
        run_kind=SWAP_RUN_KIND if swap else NATURAL_RUN_KIND,
        condition=f"{condition_prefix};control_checkpoint_sha256={control_sha256}",
    )
    journal = AtomicPredictionJournal(
        Path(args.output).expanduser().resolve(),
        provenance,
        resume=not args.no_resume,
    )
    started = time.perf_counter()
    initial_count = len(journal.records)
    prefix_by_source: dict[str, str] = {}
    signature_by_source: dict[str, str] = {}
    for source_scene_id, questions in sorted(by_scene.items()):
        injected_scene_id = pairing[source_scene_id] if swap else source_scene_id
        pending = [
            question
            for question in questions
            if not journal.contains(source_scene_id, question.question_id)
        ]
        if not pending:
            prefixes, signatures = _cached_scene_identities(journal.records, source_scene_id)
            if len(prefixes) != 1 or len(signatures) != 1:
                raise RuntimeError("Cached V66 rows lack one immutable scene identity")
            prefix_by_source[source_scene_id] = next(iter(prefixes))
            signature_by_source[source_scene_id] = next(iter(signatures))
            continue
        runtime = QuestionControlledChatRuntime.load(
            config,
            injected_scene_id,
            base_checkpoint=base_checkpoint,
            control_checkpoint=control_checkpoint,
        )
        prefix_hash, signature_hash = _runtime_identities(runtime)
        prefix_by_source[source_scene_id] = prefix_hash
        signature_by_source[source_scene_id] = signature_hash
        cached_prefixes, cached_signatures = _cached_scene_identities(
            journal.records, source_scene_id
        )
        if cached_prefixes not in (set(), {prefix_hash}) or cached_signatures not in (
            set(),
            {signature_hash},
        ):
            raise RuntimeError("Cached V66 scene identity differs from loaded runtime")
        for question in pending:
            row = _prediction_record(
                source_scene_id=source_scene_id,
                question=question,
                injected_scene_id=injected_scene_id,
                runtime=runtime,
                swap=swap,
            )
            row["control_checkpoint_sha256"] = control_sha256
            journal.append(row)
            print(
                json.dumps(
                    {
                        "phase": f"v66_blind_{args.mode}_inference",
                        "source_scene": source_scene_id,
                        "completed": len(journal.records),
                    }
                ),
                flush=True,
            )
        runtime.assert_prefix_unchanged()

    expected = {
        (scene_id, question.question_id)
        for scene_id, questions in by_scene.items()
        for question in questions
    }
    if journal.completed_keys != expected or len(journal.records) != 384:
        raise RuntimeError("V66 blind prediction journal is incomplete")
    return {
        "phase": f"v66_blind_{args.mode}_inference_complete",
        "mode": args.mode,
        "predictions": len(journal.records),
        "new_predictions": len(journal.records) - initial_count,
        "scene_count": len(prefix_by_source),
        "control_checkpoint_sha256": control_sha256,
        "prediction_provenance_sha256": provenance.sha256,
        "prefix_sha256_by_source_scene": prefix_by_source,
        "scene_signature_sha256_by_source_scene": signature_by_source,
        "elapsed_seconds": time.perf_counter() - started,
        "answers_or_route_labels_loaded": False,
        "scorer_references_loaded": False,
        "qa_or_oracle_loaded": False,
        "fresh_or_final_protected_data_loaded": False,
        "output": str(Path(args.output).expanduser().resolve()),
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("natural", "swap"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--questions-manifest", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--control-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="validation", choices=("train", "validation"))
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args(argv)
    forbidden = {
        "scorer_references",
        "qa",
        "oracle",
        "route_labels",
        "answers",
        "changed_questions",
    }
    if {action.dest for action in parser._actions} & forbidden:
        raise AssertionError("V66 blind runner exposes a forbidden input")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    summary = _run(_parse_args(argv))
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ARCHITECTURE",
    "INTERNAL_SCENE_PAIRS",
    "NATURAL_CONDITION_PREFIX",
    "NATURAL_RUN_KIND",
    "SWAP_CONDITION_PREFIX",
    "SWAP_RUN_KIND",
    "main",
    "paired_scene_ids",
]
