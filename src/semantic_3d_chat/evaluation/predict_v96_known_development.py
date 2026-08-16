"""Label-blind four-arm predictor for V96's fixed-final candidate."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from semantic_3d_chat.chat.file_audit import FileAccessAudit
from semantic_3d_chat.chat.runtime_config import load_runtime_config
from semantic_3d_chat.evaluation.v85_strict_multiscene_preflight import sha256_file_v85
from semantic_3d_chat.evaluation.v96_atomic_pair_repair_preflight import (
    CONFIG,
    load_config_v96,
)
from semantic_3d_chat.evaluation.v96_known_development_common import (
    ARMS,
    PREDICTION_ARTIFACT,
    PREDICTION_COMPLETION_ARTIFACT,
    QUESTION_COUNT,
    SCHEMA_VERSION,
    assert_bound_config_path_v96,
    assert_output_bundle_state_v96,
    assert_same_candidate_v96,
    audit_report_v96,
    authenticate_fixed_final_candidate_v96,
    authenticate_fixed_inputs_before_questions_v96,
    authenticate_prediction_bundle_v96,
    canonical_sha256_v96,
    evaluation_paths_v96,
    load_future_trainer_v96,
    load_known_questions_v96,
    prediction_forbidden_roots_v96,
    prediction_provenance_v96,
    prediction_row_v96,
    validate_prediction_rows_v96,
    write_json_create_once_v96,
    write_jsonl_create_once_v96,
)
from semantic_3d_chat.evaluation.v96_known_development_implementation import (
    hardened_evaluation_stage_v96,
)
from semantic_3d_chat.language.local_lm import load_local_language_model
from semantic_3d_chat.language.lora import LoRABankCollection, install_lora_banks
from semantic_3d_chat.language.prefix_injection import prefix_sha256
from semantic_3d_chat.training.train_v84_strict_bridge import _generate_v84


@dataclass(frozen=True)
class PredictorStackV96:
    language: Any
    collection: LoRABankCollection
    candidate: Mapping[str, Any]
    system_prompt: str
    max_new_tokens: int


def load_predictor_stack_v96(
    config: Mapping[str, Any], *, expected_candidate_state_sha256: str
) -> PredictorStackV96:
    """Load frozen Gemma+V95 and the sole fresh V96 fixed-final bank."""

    trainer = load_future_trainer_v96()
    required = (
        "combined_lora_settings_v96",
        "load_frozen_parent_v96",
        "load_fixed_final_bridge_v96",
    )
    if any(not callable(getattr(trainer, name, None)) for name in required):
        raise RuntimeError(f"V96 trainer does not expose evaluator contract: {required}")
    runtime = load_runtime_config(config["sources"]["runtime_config"])
    language_config = runtime["language"]
    language = load_local_language_model(
        str(language_config["model_id"]),
        str(language_config["revision"]),
        str(language_config["dtype"]),
        freeze=True,
        local_files_only=True,
        backend="gemma4",
        decoder_gradient_checkpointing=False,
    )
    settings = trainer.combined_lora_settings_v96(runtime, config)
    collection = install_lora_banks(language.model, settings)
    if not isinstance(collection, LoRABankCollection):
        raise TypeError("V96 known-development LoRA installation failed")
    trainer.load_frozen_parent_v96(collection, config)
    candidate = trainer.load_fixed_final_bridge_v96(
        collection, config["outputs"]["fixed_final_candidate"]
    )
    if candidate.get("state_sha256") != expected_candidate_state_sha256:
        raise ValueError("V96 predictor loaded a different fixed-final candidate")
    collection.eval()
    language.model.eval()
    language.decoder_module.eval()
    return PredictorStackV96(
        language=language,
        collection=collection,
        candidate=candidate,
        system_prompt=str(language_config["system_prompt"]),
        max_new_tokens=int(language_config["max_answer_tokens"]),
    )


def generate_arm_v96(stack: PredictorStackV96, memory: torch.Tensor, question: str) -> str:
    return _generate_v84(
        stack.language,
        stack.system_prompt,
        memory,
        SimpleNamespace(question=question, answer="unknown"),
        max_new_tokens=stack.max_new_tokens,
    )


def _all_outputs(paths: Any) -> tuple[Path, ...]:
    return (
        paths.predictions,
        paths.provenance,
        paths.prediction_access,
        paths.prediction_completion,
    )


@hardened_evaluation_stage_v96
def predict_known_development_v96(
    config_path: str | Path = CONFIG,
) -> dict[str, Any]:
    """Generate 216 four-arm rows without opening labels or prior behavior."""

    assert_bound_config_path_v96(config_path)
    initial = load_config_v96(config_path, allow_draft=False)
    paths = evaluation_paths_v96(initial)
    states = [path.exists() or path.is_symlink() for path in _all_outputs(paths)]
    if all(states):
        bundle = authenticate_prediction_bundle_v96(config_path)
        return {
            "artifact": PREDICTION_ARTIFACT,
            "row_count": len(bundle["rows"]),
            "prediction_sha256": bundle["prediction_sha256"],
            "completed": True,
            "reused_authenticated_create_once_bundle": True,
            "runtime_promotion_authorized": False,
        }
    assert_output_bundle_state_v96(_all_outputs(paths), complete=False)

    started = time.monotonic()
    audit = FileAccessAudit(
        prediction_forbidden_roots_v96(initial),
        forbidden_component_names={"oracle"},
        block_forbidden=True,
    )
    with audit:
        config = load_config_v96(config_path, allow_draft=False)
        # All 24 scene/control memories exist before the first question read.
        fixed = authenticate_fixed_inputs_before_questions_v96(
            config, config_path=config_path, audit=audit
        )
        stack = load_predictor_stack_v96(
            config,
            expected_candidate_state_sha256=str(fixed.candidate["state_sha256"]),
        )
        questions = load_known_questions_v96()
        provenance = prediction_provenance_v96(config_path, fixed, questions)
        rows: list[dict[str, Any]] = []
        for ordinal, question in enumerate(questions.questions, 1):
            scene_id = question.scene_id
            predictions = {
                arm: generate_arm_v96(
                    stack, fixed.memories[arm][scene_id], question.question
                )
                for arm in ARMS
            }
            current_hashes = {
                arm: prefix_sha256(fixed.memories[arm][scene_id]) for arm in ARMS
            }
            rows.append(
                prediction_row_v96(
                    scene_id=scene_id,
                    question_id=question.question_id,
                    predictions=predictions,
                    memory_hashes=current_hashes,
                    provenance_sha256=provenance["provenance_sha256"],
                    unchanged=all(
                        current_hashes[arm] == fixed.memory_hashes[arm][scene_id]
                        for arm in ARMS
                    ),
                )
            )
            if ordinal == 1 or ordinal % 12 == 0 or ordinal == QUESTION_COUNT:
                print(
                    json.dumps(
                        {
                            "event": "v96_known_development_prediction",
                            "ordinal": ordinal,
                            "total": QUESTION_COUNT,
                            "scene_id": scene_id,
                            "question_id": question.question_id,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        hashes_after = {
            arm: {
                scene_id: prefix_sha256(fixed.memories[arm][scene_id])
                for scene_id in fixed.memories[arm]
            }
            for arm in ARMS
        }
        candidate_after = authenticate_fixed_final_candidate_v96(
            config, config_path=config_path, audit=audit
        )
    audit.assert_clean()
    assert_same_candidate_v96(fixed.candidate, candidate_after)
    if hashes_after != fixed.memory_hashes:
        raise RuntimeError("V96 predictor mutated a fixed scene/control memory")
    validate_prediction_rows_v96(
        rows,
        questions=questions,
        memory_hashes=fixed.memory_hashes,
        provenance_sha256=provenance["provenance_sha256"],
    )
    access = audit_report_v96(audit)
    if access["passed"] is not True or access["protected_read_count"] != 0:
        raise RuntimeError("V96 predictor protected-file audit failed")

    write_json_create_once_v96(paths.provenance, provenance)
    write_jsonl_create_once_v96(paths.predictions, rows)
    write_json_create_once_v96(paths.prediction_access, access)
    completion = {
        "artifact": PREDICTION_COMPLETION_ARTIFACT,
        "schema_version": SCHEMA_VERSION,
        "candidate_fingerprint_before": fixed.candidate["fingerprint_sha256"],
        "candidate_fingerprint_after": candidate_after["fingerprint_sha256"],
        "candidate_immutable": True,
        "frozen_v95_parent_immutable": (
            fixed.candidate["frozen_v95_state_sha256"]
            == candidate_after["frozen_v95_state_sha256"]
        ),
        "memory_manifest_sha256": fixed.memory_manifest_sha256,
        "bound_memory_inventory_sha256": canonical_sha256_v96(fixed.memory_hashes),
        "question_manifest_sha256": questions.manifest_sha256,
        "questions_sha256": questions.questions_sha256,
        "prediction_provenance_sha256": provenance["provenance_sha256"],
        "prediction_provenance_file_sha256": sha256_file_v85(paths.provenance),
        "prediction_sha256": sha256_file_v85(paths.predictions),
        "prediction_access_sha256": sha256_file_v85(paths.prediction_access),
        "row_count": len(rows),
        "scene_count": 6,
        "arms": list(ARMS),
        "all_memories_bound_before_questions": True,
        "all_memory_hashes_invariant": True,
        "labels_opened": False,
        "oracle_opened": False,
        "protected_read_count": 0,
        "runtime_promotion_authorized": False,
        "elapsed_seconds": time.monotonic() - started,
    }
    write_json_create_once_v96(paths.prediction_completion, completion)
    bundle = authenticate_prediction_bundle_v96(config_path)
    return {
        **completion,
        "completed": True,
        "reused_authenticated_create_once_bundle": False,
        "prediction_sha256": bundle["prediction_sha256"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(CONFIG))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    print(json.dumps(predict_known_development_v96(args.config), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "PredictorStackV96",
    "generate_arm_v96",
    "load_predictor_stack_v96",
    "main",
    "predict_known_development_v96",
]
