"""Evaluate V88's one fixed-final development-known scene-one checkpoint.

The scorer keeps V87's acceptance contract unchanged: all 138 canonical rows,
the three canonical native-boundary zero-payload controls, and the exact three
user-facing smoke phrasings.  The smoke is explicitly training-known and is a
runnable-demo check, not held-out evidence.  Passing authorizes only a separate
oracle-unavailable runtime package; this command never promotes a checkpoint.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import torch

from semantic_3d_chat.chat.file_audit import FileAccessAudit
from semantic_3d_chat.chat.runtime_config import load_runtime_config
from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.evaluation.evaluate_v87_scene1_balanced import score_records_v87
from semantic_3d_chat.evaluation.metrics import normalize_answer
from semantic_3d_chat.evaluation.v85_strict_multiscene_preflight import (
    atomic_create_json_v85,
    canonical_sha256_v85,
    resolve_v85,
)
from semantic_3d_chat.evaluation.v86_scene1_demo_preflight import (
    load_scene1_memory_v86,
    zero_payload_memory_v86,
)
from semantic_3d_chat.evaluation.v88_scene1_augmented_preflight import (
    CANONICAL_CAUSAL_IDS,
    CONFIG,
    SCENE_ID,
    authenticate_sources_v88,
    load_canonical_rows_v88,
    load_config_v88,
)
from semantic_3d_chat.language.local_lm import load_local_language_model
from semantic_3d_chat.language.lora import LoRABankCollection, install_lora_banks
from semantic_3d_chat.language.prefix_injection import prefix_sha256
from semantic_3d_chat.training.train_question_control_v73 import RowV73
from semantic_3d_chat.training.train_v84_strict_bridge import (
    _generate_v84,
    _measure_nll_v84,
)
from semantic_3d_chat.training.train_v88_scene1_augmented import (
    authenticate_training_report_v88,
    combined_lora_settings_v88,
    load_fixed_final_bridge_v88,
    load_frozen_stack_v88,
)

PREDICTIONS_ARTIFACT: Final[str] = "gemma4_v88_scene1_augmented_predictions_v1"
EVALUATION_ARTIFACT: Final[str] = "gemma4_v88_scene1_augmented_evaluation_v1"
_FORBIDDEN_COMPONENTS: Final[frozenset[str]] = frozenset(
    {"oracle", "validation", "test", "deferred"}
)


def _canonical_causal_rows_v88(rows: Sequence[RowV73]) -> tuple[RowV73, ...]:
    by_id = {row.question_id: row for row in rows}
    selected = tuple(by_id[question_id] for question_id in CANONICAL_CAUSAL_IDS)
    expected = {
        "q_000080": ("Is there a chair in the room?", "yes"),
        "q_000108": ("What color is the bowl?", "red"),
        "q_000014": ("Is the chair left or right of the bowl?", "right"),
    }
    if any((row.question, row.answer) != expected[row.question_id] for row in selected):
        raise ValueError("V88 canonical causal reference changed")
    return selected


def _smoke_rows_v88(config: Mapping[str, Any]) -> tuple[RowV73, ...]:
    answer_types = ("presence", "attribute", "spatial_relation")
    return tuple(
        RowV73(
            scene_id=SCENE_ID,
            question_id=f"v88_smoke_{ordinal:02d}",
            question=str(raw["question"]),
            answer=normalize_answer(raw["expected"]),
            answer_class=f"v88_smoke_class_{ordinal:02d}",
            answer_type=answer_type,
            pair_id="v88_development_known_smoke",
            paired_scene_id=SCENE_ID,
            question_key=f"v88_smoke_{ordinal:02d}",
            change_type="none",
            expected_change=False,
        )
        for ordinal, (raw, answer_type) in enumerate(
            zip(config["gates"]["live_smoke_questions"], answer_types, strict=True)
        )
    )


def run_evaluation_v88(config_path: str | Path = CONFIG) -> dict[str, Any]:
    started = time.monotonic()
    config = load_config_v88(config_path)
    source_hashes = authenticate_sources_v88(config)
    training_bindings = authenticate_training_report_v88(config, config_path=config_path)
    rows = load_canonical_rows_v88(config)
    causal_rows = _canonical_causal_rows_v88(rows)
    smoke_rows = _smoke_rows_v88(config)
    predictions_path = resolve_v85(config["outputs"]["evaluation_predictions"])
    report_path = resolve_v85(config["outputs"]["evaluation_report"])
    if predictions_path.exists() or report_path.exists():
        raise FileExistsError("V88 create-once evaluation output exists")

    cpu_memory, memory_hash_before, _metadata = load_scene1_memory_v86(config)
    cpu_zero_memory = zero_payload_memory_v86(cpu_memory)
    zero_hash_before = prefix_sha256(cpu_zero_memory)
    audit = FileAccessAudit(
        forbidden_component_names=_FORBIDDEN_COMPONENTS,
        block_forbidden=True,
    )
    with audit:
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
        if language.device.type != "mps":
            raise RuntimeError("V88 fixed-final evaluation requires local MPS")
        collection = install_lora_banks(
            language.model, combined_lora_settings_v88(runtime, config)
        )
        if not isinstance(collection, LoRABankCollection):
            raise TypeError("V88 evaluation LoRA installation failed")
        frozen_source = load_frozen_stack_v88(
            collection,
            v85_checkpoint=config["sources"]["frozen_v85_checkpoint"],
            v86_checkpoint=config["sources"]["parent_v86_checkpoint"],
            v87_checkpoint=config["sources"]["parent_v87_checkpoint"],
            experiment=config,
        )
        candidate = load_fixed_final_bridge_v88(
            collection, config["outputs"]["fixed_final_candidate"]
        )
        collection.eval()
        language.decoder_module.eval()
        memory = cpu_memory.to(device=language.device, dtype=torch.bfloat16)
        zero_memory = cpu_zero_memory.to(device=language.device, dtype=torch.bfloat16)
        system_prompt = str(language_config["system_prompt"])
        max_new_tokens = int(language_config["max_answer_tokens"])
        records: list[dict[str, Any]] = []
        for ordinal, row in enumerate(rows, 1):
            measured, layout = _measure_nll_v84(language, system_prompt, memory, row)
            prediction = _generate_v84(
                language,
                system_prompt,
                memory,
                row,
                max_new_tokens=max_new_tokens,
            )
            records.append(
                {
                    "scene_id": row.scene_id,
                    "question_id": row.question_id,
                    "answer_type": row.answer_type,
                    "reference_answer": row.answer,
                    "prediction": prediction,
                    "normalized_prediction": normalize_answer(prediction),
                    "correct_mean_nll": measured["mean_nll"],
                    "correct_answer_token_top1_accuracy": measured[
                        "answer_token_top1_accuracy"
                    ],
                    "scene_memory_sha256": memory_hash_before,
                    "layout_audit": layout,
                }
            )
            if ordinal == 1 or ordinal % 12 == 0 or ordinal == len(rows):
                print(
                    json.dumps(
                        {
                            "event": "v88_evaluation_row",
                            "ordinal": ordinal,
                            "total": len(rows),
                            "question_id": row.question_id,
                            "elapsed_seconds": time.monotonic() - started,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            torch.mps.empty_cache()

        prediction_by_id = {str(record["question_id"]): record for record in records}
        causal_records: list[dict[str, Any]] = []
        for row in causal_rows:
            zero_measured, zero_layout = _measure_nll_v84(
                language, system_prompt, zero_memory, row
            )
            zero_prediction = _generate_v84(
                language,
                system_prompt,
                zero_memory,
                row,
                max_new_tokens=max_new_tokens,
            )
            correct = prediction_by_id[row.question_id]
            causal_records.append(
                {
                    "scene_id": SCENE_ID,
                    "question_id": row.question_id,
                    "answer_type": row.answer_type,
                    "reference_answer": row.answer,
                    "correct_prediction": correct["prediction"],
                    "zero_prediction": zero_prediction,
                    "correct_mean_nll": correct["correct_mean_nll"],
                    "zero_mean_nll": zero_measured["mean_nll"],
                    "zero_minus_correct_nll": zero_measured["mean_nll"]
                    - float(correct["correct_mean_nll"]),
                    "correct_memory_sha256": memory_hash_before,
                    "zero_payload_memory_sha256": zero_hash_before,
                    "zero_layout_audit": zero_layout,
                }
            )
        smoke_records: list[dict[str, Any]] = []
        for row in smoke_rows:
            prediction = _generate_v84(
                language,
                system_prompt,
                memory,
                row,
                max_new_tokens=max_new_tokens,
            )
            normalized = normalize_answer(prediction)
            smoke_records.append(
                {
                    "question_id": row.question_id,
                    "question": row.question,
                    "expected": row.answer,
                    "prediction": prediction,
                    "normalized_prediction": normalized,
                    "exact_correct": normalized == row.answer,
                    "development_known_and_trained": True,
                    "held_out": False,
                    "scene_memory_sha256": memory_hash_before,
                }
            )
        memory_hash_after = prefix_sha256(memory.detach().cpu())
        zero_hash_after = prefix_sha256(zero_memory.detach().cpu())
    audit.assert_clean()

    prefix_hash_invariant = (
        memory_hash_after == memory_hash_before
        and all(record["scene_memory_sha256"] == memory_hash_before for record in records)
        and all(
            record["scene_memory_sha256"] == memory_hash_before for record in smoke_records
        )
    )
    environment_input_invariant = (
        prefix_hash_invariant
        and zero_hash_after == zero_hash_before
        and all(
            record["correct_memory_sha256"] == memory_hash_before
            and record["zero_payload_memory_sha256"] == zero_hash_before
            for record in causal_records
        )
    )
    protected_count = len(audit.forbidden_accesses())
    score = score_records_v87(
        rows,
        records,
        causal_records,
        smoke_records,
        gates=config["gates"],
        prefix_hash_invariant=prefix_hash_invariant,
        environment_input_invariant=environment_input_invariant,
        protected_read_count=protected_count,
    )
    score["generic_smoke"]["development_known_and_trained"] = True
    score["generic_smoke"]["held_out"] = False
    predictions = {
        "artifact": PREDICTIONS_ARTIFACT,
        "schema_version": 88,
        "status": "fixed_final_evaluation_only_not_runtime",
        "config_sha256": training_bindings["config_sha256"],
        "training_report_sha256": training_bindings["training_report_sha256"],
        "row_count": len(records),
        "scene_count": 1,
        "fixed_checkpoint_selected_before_scoring": True,
        "checkpoint_selection_after_scoring": False,
        "development_known_smoke_trained": True,
        "held_out_smoke_claim": False,
        "scene_memory": {
            "compiled_before_question_tokenization": True,
            "shape": [1, 738, 1536],
            "prefix_sha256_before": memory_hash_before,
            "prefix_sha256_after": memory_hash_after,
            "zero_payload_prefix_sha256_before": zero_hash_before,
            "zero_payload_prefix_sha256_after": zero_hash_after,
            "prefix_hash_invariant": prefix_hash_invariant,
            "same_prefix_reused_for_every_question": True,
            "question_derived_environmental_tokens": 0,
            "question_conditioned_environmental_readout": False,
            "question_dependent_scene_processing": False,
            "question_dependent_retrieval": False,
        },
        "candidate": {
            "path": config["outputs"]["fixed_final_candidate"],
            "weights_sha256": candidate["weights_sha256"],
            "state_sha256": candidate["state_sha256"],
            "optimizer_updates": 188,
        },
        "leakage": {
            "loaded_file_count": len(audit.unique_paths),
            "loaded_file_inventory_sha256": canonical_sha256_v85(audit.unique_paths),
            "protected_read_count": protected_count,
            "protected_reads": audit.forbidden_accesses(),
            "oracle_loaded": False,
        },
        "training_references_serialized_in_runtime_candidate": False,
        "augmentation_inventory_serialized_in_runtime_candidate": False,
        "error_inventory_serialized_in_runtime_candidate": False,
        "runtime_promotion_authorized": False,
        "records": records,
        "causal_records": causal_records,
        "smoke_records": smoke_records,
    }
    prediction_output, prediction_sha = atomic_create_json_v85(predictions_path, predictions)
    report = {
        "artifact": EVALUATION_ARTIFACT,
        "schema_version": 88,
        "status": (
            "model_gates_pass_separate_runtime_packaging_required"
            if score["model_acceptance_gate_passed"]
            else "model_gates_fail_not_runtime_promotable"
        ),
        "config_sha256": training_bindings["config_sha256"],
        "preregistration_sha256": training_bindings["preregistration_sha256"],
        "cpu_preflight_sha256": training_bindings["cpu_preflight_sha256"],
        "training_report_sha256": training_bindings["training_report_sha256"],
        "evaluation_predictions_path": prediction_output.relative_to(PROJECT_ROOT).as_posix(),
        "evaluation_predictions_sha256": prediction_sha,
        "source_hashes": source_hashes,
        "frozen_source": frozen_source,
        "fixed_checkpoint_selected_before_scoring": True,
        "checkpoint_selection_after_scoring": False,
        "preregistered_gates": config["gates"],
        "metrics": score,
        "scene_memory": predictions["scene_memory"],
        "leakage": predictions["leakage"],
        "development_known_smoke_trained": True,
        "held_out_smoke_claim": False,
        "held_out_generalization_claim": False,
        "parent_v85_v86_v87_mutated": False,
        "separate_runtime_packaging_authorized": score["model_acceptance_gate_passed"],
        "runtime_oracle_unavailable_gate_pending": score["model_acceptance_gate_passed"],
        "runtime_file_audit_gate_pending": score["model_acceptance_gate_passed"],
        "automatic_runtime_promotion": False,
        "runtime_promotion_authorized": False,
        "official_validation_loaded": False,
        "official_test_loaded": False,
        "deferred_final_loaded": False,
        "oracle_loaded": False,
        "elapsed_seconds": time.monotonic() - started,
    }
    atomic_create_json_v85(report_path, report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(CONFIG))
    args = parser.parse_args(argv)
    report = run_evaluation_v88(args.config)
    print(
        json.dumps(
            {
                "status": report["status"],
                "metrics": report["metrics"],
                "development_known_smoke_trained": True,
                "held_out_smoke_claim": False,
                "runtime_promotion_authorized": False,
                "elapsed_seconds": report["elapsed_seconds"],
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EVALUATION_ARTIFACT",
    "PREDICTIONS_ARTIFACT",
    "main",
    "run_evaluation_v88",
]
