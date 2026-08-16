"""Answer questions with local Gemma from evaluation-only oracle scene text.

This process is intentionally answer-blind: it rejects QA/oracle paths, loads a
strict question-only manifest plus the prepared scene-text control artifact,
and never imports the preparation or scoring implementations.  This is a
prohibited upper bound, not a valid primary scene-memory path.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable, Mapping, Sequence
from importlib.metadata import version
from pathlib import Path
from typing import Any

import torch

from semantic_3d_chat.config import PROJECT_ROOT, load_config
from semantic_3d_chat.evaluation.baseline_io import (
    atomic_write_jsonl,
    sha256_file,
    text_fingerprint,
)
from semantic_3d_chat.evaluation.oracle_text_artifacts import (
    PREDICTION_BASELINE,
    PREDICTION_PROVENANCE_SCHEMA,
    PREDICTION_REPORT_SCHEMA,
    SceneTextBundle,
    atomic_write_json,
    canonical_json_sha256,
    default_prediction_report_path,
    default_provenance_path,
    load_prediction_provenance,
    load_prediction_records,
    load_scene_text_bundle,
    validate_v55_development_scope,
)
from semantic_3d_chat.evaluation.question_manifest import (
    QuestionManifest,
    load_question_manifest,
)
from semantic_3d_chat.language.local_lm import load_local_language_model

TextAnswerer = Callable[[str, str], str]

DEFAULT_CONFIG = Path("configs/experiments/gemma4_oracle_text_v55.yaml")
DEFAULT_QUESTIONS = Path("reports/gemma4/questions/v55_development_validation.json")
DEFAULT_SCENES = Path(
    "reports/gemma4/evaluation_only/oracle_text_upper_bound/v55_scene_descriptions.json"
)
DEFAULT_PREDICTIONS = Path(
    "reports/gemma4/evaluation_only/oracle_text_upper_bound/v55_predictions.jsonl"
)


def _inference_contract(
    config: Mapping[str, Any], *, injected_test_answerer: bool
) -> dict[str, Any]:
    language = config.get("language")
    evaluation = config.get("evaluation")
    if not isinstance(language, Mapping) or not isinstance(evaluation, Mapping):
        raise TypeError("Config requires language and evaluation mappings")
    baselines = evaluation.get("baselines")
    baseline = baselines.get("oracle_text") if isinstance(baselines, Mapping) else None
    if not isinstance(baseline, Mapping):
        raise TypeError("Config requires evaluation.baselines.oracle_text")
    backend = str(language.get("backend", "auto")).casefold()
    model_id = str(language.get("model_id", ""))
    revision = str(language.get("revision", ""))
    if backend not in {"auto", "gemma4"} or "gemma-4" not in model_id.casefold():
        raise ValueError("The V55 oracle-text control requires the local Gemma 4 backend")
    if not revision or revision == "main":
        raise ValueError("The local Gemma 4 revision must be pinned")
    max_answer_tokens = baseline.get("max_answer_tokens")
    if (
        isinstance(max_answer_tokens, bool)
        or not isinstance(max_answer_tokens, int)
        or max_answer_tokens < 1
    ):
        raise ValueError("oracle_text.max_answer_tokens must be a positive integer")
    system_prompt = baseline.get("system_prompt")
    if not isinstance(system_prompt, str) or not system_prompt.strip():
        raise ValueError("oracle_text.system_prompt must be non-empty")
    return {
        "backend": "gemma4",
        "model_id": model_id,
        "model_revision": revision,
        "model_license": "Gemma Terms of Use",
        "dtype": str(language.get("dtype", "float16")),
        "local_files_only": True,
        "generation_backend": (
            "injected_test_answerer" if injected_test_answerer else "local_gemma"
        ),
        "scientific_measurement_eligible": not injected_test_answerer,
        "greedy_decoding": True,
        "max_answer_tokens": max_answer_tokens,
        "system_prompt": system_prompt.strip(),
        "scene_text_before_question": True,
        "software_versions": {
            "torch": torch.__version__,
            "transformers": version("transformers"),
        },
    }


class LocalGemmaOracleTextAnswerer:
    """Greedy local Gemma generation for the explicitly prohibited control."""

    def __init__(self, contract: Mapping[str, Any]) -> None:
        self.system_prompt = str(contract["system_prompt"])
        self.max_answer_tokens = int(contract["max_answer_tokens"])
        self.local = load_local_language_model(
            str(contract["model_id"]),
            revision=str(contract["model_revision"]),
            requested_dtype=str(contract["dtype"]),
            freeze=True,
            local_files_only=True,
            backend="gemma4",
        )

    def __call__(self, scene_text: str, question: str) -> str:
        messages = [
            {
                "role": "system",
                "content": f"{self.system_prompt}\n\n{scene_text}",
            },
            {"role": "user", "content": question},
        ]
        tokenizer = self.local.tokenizer
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        input_ids = (
            encoded.input_ids
            if hasattr(encoded, "input_ids")
            else encoded["input_ids"]
            if isinstance(encoded, dict)
            else encoded
        ).to(self.local.device)
        if input_ids.ndim != 2 or input_ids.shape[1] == 0:
            raise ValueError("Oracle-text chat template returned no token IDs")
        attention_mask = torch.ones_like(input_ids)
        generation: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "do_sample": False,
            "max_new_tokens": self.max_answer_tokens,
        }
        if tokenizer.pad_token_id is not None:
            generation["pad_token_id"] = tokenizer.pad_token_id
        if tokenizer.eos_token_id is not None:
            generation["eos_token_id"] = tokenizer.eos_token_id
        with torch.inference_mode():
            output = self.local.model.generate(**generation)
        continuation = output[0, input_ids.shape[1] :].detach().cpu()
        return tokenizer.decode(continuation, skip_special_tokens=True).strip()


def _build_provenance(
    contract: Mapping[str, Any],
    questions: QuestionManifest,
    scenes: SceneTextBundle,
    *,
    implementation_path: Path,
) -> dict[str, Any]:
    if questions.manifest_path is None or questions.manifest_sha256 is None:
        raise AssertionError("Question manifest must have on-disk provenance")
    if scenes.path is None or scenes.file_sha256 is None:
        raise AssertionError("Scene-text bundle must have on-disk provenance")
    identity = {
        "baseline": PREDICTION_BASELINE,
        "evaluation_only": True,
        "primary_path_eligible": False,
        "prohibited_primary_input": True,
        "question_manifest_sha256": questions.manifest_sha256,
        "questions_sha256": questions.questions_sha256,
        "source_qa_sha256": questions.source_qa_sha256,
        "scene_text_bundle_sha256": scenes.file_sha256,
        "scene_descriptions_sha256": scenes.scene_descriptions_sha256,
        "scene_ids": sorted(scenes.by_scene()),
        "question_count": questions.question_count,
        "inference_contract": dict(contract),
        "implementation_files": {
            "oracle_text_predict.py": sha256_file(implementation_path),
            "oracle_text_artifacts.py": sha256_file(
                implementation_path.with_name("oracle_text_artifacts.py")
            ),
            "language/local_lm.py": sha256_file(
                implementation_path.parents[1] / "language" / "local_lm.py"
            ),
            "language/gemma4_backend.py": sha256_file(
                implementation_path.parents[1] / "language" / "gemma4_backend.py"
            ),
        },
    }
    return {
        "schema": PREDICTION_PROVENANCE_SCHEMA,
        "schema_version": 1,
        "inference_provenance_sha256": canonical_json_sha256(identity),
        "identity": identity,
        "inputs": {
            "question_manifest_path": str(questions.manifest_path),
            "scene_text_bundle_path": str(scenes.path),
        },
    }


def _validate_inputs(questions: QuestionManifest, scenes: SceneTextBundle) -> None:
    if questions.manifest_sha256 != scenes.question_manifest_sha256:
        raise ValueError("Scene text was prepared from a different question manifest")
    if questions.questions_sha256 != scenes.questions_sha256:
        raise ValueError("Scene text was prepared for different questions")
    if questions.source_qa_sha256 != scenes.source_qa_sha256:
        raise ValueError("Scene text and question manifest source hashes disagree")
    question_scenes = set(questions.by_scene())
    description_scenes = set(scenes.by_scene())
    if question_scenes != description_scenes:
        raise ValueError(
            "Scene text must cover exactly the question scenes; "
            f"missing={sorted(question_scenes - description_scenes)} "
            f"extra={sorted(description_scenes - question_scenes)}"
        )


def _validate_cached_record(
    record: Mapping[str, Any],
    *,
    scene_id: str,
    question_id: str,
    question_sha256: str,
    scene_text_sha256: str,
) -> None:
    if (
        record["scene_id"] != scene_id
        or record["question_id"] != question_id
        or record["question_sha256"] != question_sha256
        or record["scene_text_sha256"] != scene_text_sha256
    ):
        raise RuntimeError(
            f"Cached oracle-text prediction does not match current input: {scene_id}/{question_id}"
        )


def run_oracle_text_predictions(
    config: Mapping[str, Any],
    question_manifest_path: str | Path,
    scene_text_bundle_path: str | Path,
    predictions_path: str | Path,
    *,
    provenance_path: str | Path | None = None,
    report_path: str | Path | None = None,
    answerer: TextAnswerer | None = None,
    allow_test_answerer: bool = False,
    resume: bool = True,
    require_v55_development: bool = True,
) -> dict[str, Any]:
    """Generate authenticated predictions without opening answer-bearing files."""

    if answerer is not None and not allow_test_answerer:
        raise ValueError(
            "Injected answerers are test-only and require allow_test_answerer=True; "
            "the CLI always uses local Gemma"
        )
    questions = load_question_manifest(question_manifest_path)
    scenes = load_scene_text_bundle(scene_text_bundle_path)
    _validate_inputs(questions, scenes)
    validate_v55_development_scope(
        list(questions.by_scene()),
        questions.question_count,
        required=require_v55_development,
    )
    contract = _inference_contract(
        config,
        injected_test_answerer=answerer is not None,
    )
    implementation_path = Path(__file__).resolve()
    expected_provenance = _build_provenance(
        contract,
        questions,
        scenes,
        implementation_path=implementation_path,
    )
    provenance_sha256 = str(expected_provenance["inference_provenance_sha256"])
    destination = Path(predictions_path).expanduser().resolve()
    provenance_destination = (
        Path(provenance_path).expanduser().resolve()
        if provenance_path is not None
        else default_provenance_path(destination)
    )
    report_destination = (
        Path(report_path).expanduser().resolve()
        if report_path is not None
        else default_prediction_report_path(destination)
    )

    if resume and (destination.exists() or provenance_destination.exists()):
        if not destination.is_file() or not provenance_destination.is_file():
            raise RuntimeError("Cannot resume an incomplete predictions/provenance artifact pair")
        stored_provenance = load_prediction_provenance(provenance_destination)
        if stored_provenance != expected_provenance:
            raise RuntimeError(
                "Oracle-text resume provenance mismatch; use --no-resume or a new output path"
            )
        records = load_prediction_records(
            destination,
            provenance_sha256=provenance_sha256,
        )
    else:
        atomic_write_json(provenance_destination, expected_provenance)
        atomic_write_jsonl(destination, ())
        report_destination.unlink(missing_ok=True)
        records = []

    by_key = {(row["scene_id"], row["question_id"]): row for row in records}
    scene_index = scenes.by_scene()
    model_answerer = answerer
    new_prediction_count = 0
    inference_seconds = 0.0
    ordered_records: list[dict[str, Any]] = []
    for question in questions.questions:
        scene = scene_index[question.scene_id]
        question_hash = text_fingerprint(
            question.scene_id,
            question.question_id,
            question.question,
        )
        key = (question.scene_id, question.question_id)
        cached = by_key.get(key)
        if cached is not None:
            _validate_cached_record(
                cached,
                scene_id=question.scene_id,
                question_id=question.question_id,
                question_sha256=question_hash,
                scene_text_sha256=scene.scene_text_sha256,
            )
            ordered_records.append(cached)
            continue
        if model_answerer is None:
            model_answerer = LocalGemmaOracleTextAnswerer(contract)
        started = time.perf_counter()
        answer = model_answerer(scene.scene_text, question.question)
        inference_seconds += time.perf_counter() - started
        if not isinstance(answer, str):
            raise TypeError("Oracle-text answerer must return text")
        record = {
            "answer": answer.strip(),
            "baseline": PREDICTION_BASELINE,
            "evaluation_only": True,
            "inference_provenance_sha256": provenance_sha256,
            "primary_path_eligible": False,
            "prohibited_primary_input": True,
            "question_id": question.question_id,
            "question_sha256": question_hash,
            "scene_id": question.scene_id,
            "scene_text_sha256": scene.scene_text_sha256,
        }
        ordered_records.append(record)
        by_key[key] = record
        new_prediction_count += 1
        # Preserve all prior cached rows as well as newly completed rows after
        # each answer. A killed multi-hour local run loses at most one question.
        completed_in_manifest_order = [
            by_key[(item.scene_id, item.question_id)]
            for item in questions.questions
            if (item.scene_id, item.question_id) in by_key
        ]
        atomic_write_jsonl(destination, completed_in_manifest_order)

    if len(by_key) != questions.question_count:
        raise RuntimeError("Oracle-text inference did not cover every question")
    atomic_write_jsonl(destination, ordered_records)
    report = {
        "schema": PREDICTION_REPORT_SCHEMA,
        "schema_version": 1,
        "status": "complete",
        "baseline": PREDICTION_BASELINE,
        "evaluation_only": True,
        "primary_path_eligible": False,
        "prohibited_primary_input": True,
        "scientific_measurement_eligible": answerer is None,
        "question_count": questions.question_count,
        "scene_count": len(scene_index),
        "new_prediction_count": new_prediction_count,
        "resumed_prediction_count": questions.question_count - new_prediction_count,
        "inference_seconds_new_predictions": inference_seconds,
        "mean_seconds_per_new_prediction": (
            inference_seconds / new_prediction_count if new_prediction_count else 0.0
        ),
        "inference_provenance_sha256": provenance_sha256,
        "predictions_path": str(destination),
        "predictions_sha256": sha256_file(destination),
        "provenance_path": str(provenance_destination),
        "provenance_file_sha256": sha256_file(provenance_destination),
    }
    atomic_write_json(report_destination, report)
    return {**report, "report_path": str(report_destination)}


def _project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--scene-text", type=Path, default=DEFAULT_SCENES)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--provenance", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--allow-non-v55-scope", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    config_path = _project_path(args.config)
    report = run_oracle_text_predictions(
        load_config(config_path),
        _project_path(args.questions),
        _project_path(args.scene_text),
        _project_path(args.predictions),
        provenance_path=_project_path(args.provenance) if args.provenance else None,
        report_path=_project_path(args.report) if args.report else None,
        resume=not args.no_resume,
        require_v55_development=not args.allow_non_v55_scope,
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
