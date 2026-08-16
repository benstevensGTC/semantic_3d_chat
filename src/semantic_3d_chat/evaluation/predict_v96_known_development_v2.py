"""Label-blind four-arm predictor using V96's inference-safe v2 attestation."""

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
    FRESH_BANK_NAME,
)
from semantic_3d_chat.evaluation.v96_known_development_common_v2 import (
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
    load_config_v96_v2,
    load_future_trainer_v96,
    load_known_questions_v96,
    mandatory_fixed_input_reads_v96,
    prediction_forbidden_roots_v96,
    prediction_provenance_v96,
    prediction_row_v96,
    validate_access_receipt_v96_v2,
    validate_prediction_completion_schema_v96_v2,
    validate_prediction_rows_v96,
    write_json_create_once_v96,
    write_jsonl_create_once_v96,
)
from semantic_3d_chat.evaluation.v96_known_development_implementation_v2 import (
    authenticate_evaluation_implementation_v96_v2,
    authenticate_model_snapshot_v96_v2,
    hardened_evaluation_stage_v96_v2,
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
    bank_state_sha256: Mapping[str, str]
    expected_frozen_bank_state_sha256: Mapping[str, str]
    lora_bank_topology: Mapping[str, Any]


def authenticate_loaded_lora_bank_states_v96_v2(
    collection: LoRABankCollection,
    *,
    expected_frozen_bank_state_sha256: Mapping[str, str],
    expected_candidate_state_sha256: str,
    expected_lora_bank_topology: Mapping[str, Any],
) -> dict[str, str]:
    """Authenticate all live adapter tensors against pre-question sealed hashes."""

    expected_frozen = dict(expected_frozen_bank_state_sha256)
    if (
        len(expected_frozen) != 9
        or FRESH_BANK_NAME in expected_frozen
        or any(
            not isinstance(name, str)
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            for name, digest in expected_frozen.items()
        )
        or len(expected_candidate_state_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in expected_candidate_state_sha256
        )
    ):
        raise ValueError("V96 v2 sealed LoRA-bank state inventory is malformed")
    collection.validate_state()
    installed_names = tuple(bank.settings.name for bank in collection.banks)
    observed = dict(collection.state_sha256())
    expected = {**expected_frozen, FRESH_BANK_NAME: expected_candidate_state_sha256}
    if (
        len(collection.banks) != 10
        or len(installed_names) != len(set(installed_names))
        or installed_names[-1:] != (FRESH_BANK_NAME,)
        or set(installed_names[:-1]) != set(expected_frozen)
        or any(bank.settings.trainable for bank in collection.banks[:-1])
        or collection.banks[-1].settings.trainable is not True
        or collection.settings.contract() != dict(expected_lora_bank_topology)
        or set(observed) != set(installed_names)
        or observed != expected
    ):
        raise ValueError("V96 v2 loaded LoRA-bank states differ from the sealed inventory")
    return {name: observed[name] for name in installed_names}


def load_predictor_stack_v96(
    config: Mapping[str, Any],
    *,
    expected_candidate_state_sha256: str,
    expected_frozen_bank_state_sha256: Mapping[str, str],
    expected_lora_bank_topology: Mapping[str, Any],
) -> PredictorStackV96:
    """Load frozen Gemma+V95 and V96 without invoking training authentication."""

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
    if settings.contract() != dict(expected_lora_bank_topology):
        raise ValueError("V96 v2 live LoRA-bank topology differs from its seal")
    collection = install_lora_banks(language.model, settings)
    if not isinstance(collection, LoRABankCollection):
        raise TypeError("V96 v2 known-development LoRA installation failed")
    trainer.load_frozen_parent_v96(collection, config)
    candidate = trainer.load_fixed_final_bridge_v96(
        collection, config["outputs"]["fixed_final_candidate"]
    )
    if candidate.get("state_sha256") != expected_candidate_state_sha256:
        raise ValueError("V96 v2 predictor loaded a different fixed-final candidate")
    bank_states = authenticate_loaded_lora_bank_states_v96_v2(
        collection,
        expected_frozen_bank_state_sha256=expected_frozen_bank_state_sha256,
        expected_candidate_state_sha256=expected_candidate_state_sha256,
        expected_lora_bank_topology=expected_lora_bank_topology,
    )
    collection.eval()
    language.model.eval()
    language.decoder_module.eval()
    return PredictorStackV96(
        language=language,
        collection=collection,
        candidate=candidate,
        system_prompt=str(language_config["system_prompt"]),
        max_new_tokens=int(language_config["max_answer_tokens"]),
        bank_state_sha256=bank_states,
        expected_frozen_bank_state_sha256=dict(
            expected_frozen_bank_state_sha256
        ),
        lora_bank_topology=dict(expected_lora_bank_topology),
    )


def generate_arm_v96(stack: PredictorStackV96, memory: torch.Tensor, question: str) -> str:
    return _generate_v84(
        stack.language,
        stack.system_prompt,
        memory,
        SimpleNamespace(question=question, answer="unknown"),
        max_new_tokens=stack.max_new_tokens,
    )


def _outputs(paths: Any) -> tuple[Path, ...]:
    return (
        paths.predictions,
        paths.provenance,
        paths.prediction_access,
        paths.prediction_completion,
    )


@hardened_evaluation_stage_v96_v2
def predict_known_development_v96(
    config_path: str | Path = CONFIG,
) -> dict[str, Any]:
    """Generate all 216 four-arm rows without any training-QA read."""

    assert_bound_config_path_v96(config_path)
    implementation = authenticate_evaluation_implementation_v96_v2(
        config_path=config_path
    )
    initial = load_config_v96_v2(config_path, allow_draft=False)
    paths = evaluation_paths_v96(initial)
    states = [path.exists() or path.is_symlink() for path in _outputs(paths)]
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
    assert_output_bundle_state_v96(_outputs(paths), complete=False)

    started = time.monotonic()
    audit = FileAccessAudit(
        prediction_forbidden_roots_v96(initial),
        forbidden_component_names={"oracle"},
        block_forbidden=True,
    )
    with audit:
        implementation_inside = authenticate_evaluation_implementation_v96_v2(
            config_path=config_path
        )
        if implementation_inside != implementation:
            raise RuntimeError("V96 v2 implementation changed across audit boundary")
        config = load_config_v96_v2(config_path, allow_draft=False)
        fixed = authenticate_fixed_inputs_before_questions_v96(
            config,
            config_path=config_path,
            audit=audit,
            implementation=implementation_inside,
        )
        model_snapshot_before = authenticate_model_snapshot_v96_v2(
            config,
            expected=implementation_inside["model_snapshot_binding"],
            audit=audit,
        )
        stack = load_predictor_stack_v96(
            config,
            expected_candidate_state_sha256=str(fixed.candidate["state_sha256"]),
            expected_frozen_bank_state_sha256=implementation_inside[
                "frozen_bank_expected_states"
            ],
            expected_lora_bank_topology=implementation_inside[
                "lora_bank_topology"
            ],
        )
        questions = load_known_questions_v96()
        provenance = prediction_provenance_v96(
            config_path, fixed, questions, implementation=implementation_inside
        )
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
                            "event": "v96_v2_known_development_prediction",
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
        bank_states_after = authenticate_loaded_lora_bank_states_v96_v2(
            stack.collection,
            expected_frozen_bank_state_sha256=(
                stack.expected_frozen_bank_state_sha256
            ),
            expected_candidate_state_sha256=str(fixed.candidate["state_sha256"]),
            expected_lora_bank_topology=stack.lora_bank_topology,
        )
        if bank_states_after != stack.bank_state_sha256:
            raise RuntimeError("V96 v2 LoRA-bank states changed during prediction")
        candidate_after = authenticate_fixed_final_candidate_v96(
            config,
            config_path=config_path,
            audit=audit,
            implementation=implementation_inside,
        )
        model_snapshot_after = authenticate_model_snapshot_v96_v2(
            config,
            expected=implementation_inside["model_snapshot_binding"],
            audit=audit,
        )
        if model_snapshot_after != model_snapshot_before:
            raise RuntimeError("V96 v2 base-model snapshot changed during prediction")
        implementation_after = authenticate_evaluation_implementation_v96_v2(
            config_path=config_path
        )
        if implementation_after != implementation_inside:
            raise RuntimeError("V96 v2 implementation changed during prediction")
    audit.assert_clean()
    assert_same_candidate_v96(fixed.candidate, candidate_after)
    if hashes_after != fixed.memory_hashes:
        raise RuntimeError("V96 v2 predictor mutated a fixed memory")
    validate_prediction_rows_v96(
        rows,
        questions=questions,
        memory_hashes=fixed.memory_hashes,
        provenance_sha256=provenance["provenance_sha256"],
    )
    access = audit_report_v96(audit)
    mandatory = mandatory_fixed_input_reads_v96(
        config,
        fixed,
        config_path=config_path,
        implementation=implementation_after,
    )
    validate_access_receipt_v96_v2(
        access,
        forbidden_roots=prediction_forbidden_roots_v96(config),
        mandatory=mandatory,
    )
    training_qa = str(
        Path(config["sources"]["training_qa"])
        if Path(config["sources"]["training_qa"]).is_absolute()
        else Path.cwd() / str(config["sources"]["training_qa"])
    )
    if (
        access["passed"] is not True
        or access["protected_read_count"] != 0
        or training_qa in set(access["loaded_files"])
    ):
        raise RuntimeError("V96 v2 predictor file-access boundary failed")

    write_json_create_once_v96(paths.provenance, provenance)
    write_jsonl_create_once_v96(paths.predictions, rows)
    write_json_create_once_v96(paths.prediction_access, access)
    completion = {
        "artifact": PREDICTION_COMPLETION_ARTIFACT,
        "schema_version": SCHEMA_VERSION,
        "candidate_fingerprint_before": fixed.candidate["fingerprint_sha256"],
        "candidate_fingerprint_after": candidate_after["fingerprint_sha256"],
        "candidate_attestation_file_sha256": fixed.candidate[
            "attestation_file_sha256"
        ],
        "model_snapshot_inventory_sha256": model_snapshot_before[
            "inventory_sha256"
        ],
        "model_snapshot_hashes_invariant": True,
        "lora_bank_state_inventory_sha256": canonical_sha256_v96(
            stack.bank_state_sha256
        ),
        "lora_bank_topology_sha256": implementation_after[
            "lora_bank_topology_sha256"
        ],
        "lora_bank_states_invariant": True,
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
        "implementation_reauthenticated_inside_audit": True,
        "training_qa_opened": False,
        "labels_opened": False,
        "oracle_opened": False,
        "protected_read_count": 0,
        "runtime_promotion_authorized": False,
        "elapsed_seconds": time.monotonic() - started,
    }
    validate_prediction_completion_schema_v96_v2(completion)
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
    result = predict_known_development_v96(args.config)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "PredictorStackV96",
    "authenticate_loaded_lora_bank_states_v96_v2",
    "generate_arm_v96",
    "load_predictor_stack_v96",
    "main",
    "predict_known_development_v96",
]
