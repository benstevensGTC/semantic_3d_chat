"""Evaluate V86's one fixed-final scene-000001 demonstration checkpoint.

This scorer opens all 138 training-authorized references only for evaluation.
It cannot mutate or select a checkpoint.  It measures canonical answer
accuracy, the preregistered native-boundary zero-payload control, and the three
generic chat questions.  Passing these model-level gates still does not promote
the runtime: an independently audited oracle-unavailable live smoke must pass.
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
from semantic_3d_chat.evaluation.metrics import normalize_answer
from semantic_3d_chat.evaluation.v56_fresh_development_score import (
    canonical_answer_key,
    canonical_type_specific_match,
)
from semantic_3d_chat.evaluation.v85_strict_multiscene_preflight import (
    atomic_create_json_v85,
    canonical_sha256_v85,
    resolve_v85,
)
from semantic_3d_chat.evaluation.v86_scene1_demo_preflight import (
    CONFIG,
    SCENE_ID,
    authenticate_sources_v86,
    causal_rows_v86,
    load_config_v86,
    load_scene1_memory_v86,
    load_scene1_rows_v86,
    zero_payload_memory_v86,
)
from semantic_3d_chat.language.local_lm import load_local_language_model
from semantic_3d_chat.language.lora import LoRABankCollection, install_lora_banks
from semantic_3d_chat.language.prefix_injection import prefix_sha256
from semantic_3d_chat.training.train_question_control_v73 import RowV73
from semantic_3d_chat.training.train_v84_strict_bridge import (
    _generate_v84,
    _measure_nll_v84,
)
from semantic_3d_chat.training.train_v86_scene1_demo import (
    authenticate_training_report_v86,
    combined_lora_settings_v86,
    load_fixed_final_bridge_v86,
    load_frozen_v85_stack_v86,
)

PREDICTIONS_ARTIFACT: Final[str] = "gemma4_v86_scene1_demo_predictions_v1"
EVALUATION_ARTIFACT: Final[str] = "gemma4_v86_scene1_demo_evaluation_v1"
_FORBIDDEN_COMPONENTS: Final[frozenset[str]] = frozenset(
    {"oracle", "validation", "test", "deferred"}
)


def _smoke_rows(config: Mapping[str, Any]) -> tuple[RowV73, ...]:
    result: list[RowV73] = []
    answer_types = ("presence", "attribute", "spatial_relation")
    for ordinal, (raw, answer_type) in enumerate(
        zip(config["gates"]["live_smoke_questions"], answer_types, strict=True)
    ):
        result.append(
            RowV73(
                scene_id=SCENE_ID,
                question_id=f"v86_smoke_{ordinal:02d}",
                question=str(raw["question"]),
                answer=normalize_answer(raw["expected"]),
                answer_class=f"v86_smoke_class_{ordinal:02d}",
                answer_type=answer_type,
                pair_id="v86_smoke",
                paired_scene_id=SCENE_ID,
                question_key=f"v86_smoke_{ordinal:02d}",
                change_type="none",
                expected_change=False,
            )
        )
    return tuple(result)


def score_records_v86(
    rows: Sequence[RowV73],
    records: Sequence[Mapping[str, Any]],
    causal_records: Sequence[Mapping[str, Any]],
    smoke_records: Sequence[Mapping[str, Any]],
    *,
    gates: Mapping[str, Any],
    prefix_hash_invariant: bool,
    environment_input_invariant: bool,
    protected_read_count: int,
) -> dict[str, Any]:
    if len(rows) != 138 or len(records) != 138:
        raise ValueError("V86 score requires all 138 scene-one rows")
    expected_ids = {row.question_id for row in rows}
    if {str(record["question_id"]) for record in records} != expected_ids:
        raise ValueError("V86 prediction inventory differs from the fixed references")
    row_by_id = {row.question_id: row for row in rows}
    scored: list[dict[str, Any]] = []
    for record in records:
        row = row_by_id[str(record["question_id"])]
        prediction = str(record["prediction"])
        scored.append(
            {
                **record,
                "canonical_prediction": canonical_answer_key(row.answer_type, prediction),
                "canonical_correct": canonical_type_specific_match(
                    row.answer_type, prediction, row.answer
                ),
                "strict_normalized_exact": normalize_answer(prediction) == row.answer,
            }
        )
    by_type: dict[str, Any] = {}
    for answer_type in sorted({row.answer_type for row in rows}):
        selected = [
            record
            for record in scored
            if row_by_id[str(record["question_id"])].answer_type == answer_type
        ]
        correct = sum(bool(record["canonical_correct"]) for record in selected)
        by_type[answer_type] = {
            "correct": correct,
            "total": len(selected),
            "accuracy": correct / len(selected),
        }
    canonical_correct = sum(bool(record["canonical_correct"]) for record in scored)
    strict_correct = sum(bool(record["strict_normalized_exact"]) for record in scored)
    if len(causal_records) != 3:
        raise ValueError("V86 causal control requires exactly three rows")
    causal_mean_margin = sum(
        float(record["zero_minus_correct_nll"]) for record in causal_records
    ) / len(causal_records)
    causal_prediction_changes = sum(
        canonical_answer_key(str(record["answer_type"]), record["correct_prediction"])
        != canonical_answer_key(str(record["answer_type"]), record["zero_prediction"])
        for record in causal_records
    )
    if len(smoke_records) != 3:
        raise ValueError("V86 generic smoke requires exactly three rows")
    smoke_correct = sum(bool(record["exact_correct"]) for record in smoke_records)
    model_gates = {
        "all_scene1_canonical_accuracy_at_least_0_80": canonical_correct / 138
        >= float(gates["all_scene1_canonical_accuracy_minimum"]),
        "exact_training_row_count_138": len(scored)
        == int(gates["exact_training_row_count_required"]),
        "generic_live_smoke_at_least_2_of_3": smoke_correct
        >= int(gates["live_smoke_minimum_correct"]),
        "causal_correct_memory_mean_nll_below_zero_payload": causal_mean_margin > 0.0,
        "causal_prediction_change_at_least_1": causal_prediction_changes
        >= int(gates["causal_prediction_change_minimum"]),
        "exact_prefix_hash_invariance": prefix_hash_invariant,
        "exact_total_environment_input_invariance": environment_input_invariant,
        "protected_read_count_zero": protected_read_count
        <= int(gates["forbidden_runtime_read_count_maximum"]),
    }
    passed = all(model_gates.values())
    return {
        "canonical_type_specific": {
            "correct": canonical_correct,
            "total": len(scored),
            "accuracy": canonical_correct / len(scored),
        },
        "strict_normalized_exact": {
            "correct": strict_correct,
            "total": len(scored),
            "accuracy": strict_correct / len(scored),
        },
        "canonical_accuracy_by_answer_type": by_type,
        "answer_token_mean_nll": sum(float(record["correct_mean_nll"]) for record in records)
        / len(records),
        "causal_control": {
            "row_count": len(causal_records),
            "mean_correct_memory_nll": sum(
                float(record["correct_mean_nll"]) for record in causal_records
            )
            / len(causal_records),
            "mean_zero_payload_nll": sum(
                float(record["zero_mean_nll"]) for record in causal_records
            )
            / len(causal_records),
            "mean_zero_minus_correct_nll": causal_mean_margin,
            "canonical_prediction_changes": causal_prediction_changes,
            "records": list(causal_records),
        },
        "generic_smoke": {
            "correct": smoke_correct,
            "total": len(smoke_records),
            "accuracy": smoke_correct / len(smoke_records),
            "records": list(smoke_records),
        },
        "model_acceptance_gates": model_gates,
        "model_acceptance_gate_passed": passed,
        "runtime_oracle_unavailable_gate_pending": passed,
        "runtime_file_audit_gate_pending": passed,
        "runtime_promotion_authorized": False,
    }


def run_evaluation_v86(config_path: str | Path = CONFIG) -> dict[str, Any]:
    started = time.monotonic()
    config = load_config_v86(config_path)
    source_hashes = authenticate_sources_v86(config)
    training_bindings = authenticate_training_report_v86(config, config_path=config_path)
    rows = load_scene1_rows_v86(config)
    causal_rows = causal_rows_v86(config, rows)
    smoke_rows = _smoke_rows(config)
    predictions_path = resolve_v85(config["outputs"]["evaluation_predictions"])
    report_path = resolve_v85(config["outputs"]["evaluation_report"])
    if predictions_path.exists() or report_path.exists():
        raise FileExistsError("V86 create-once evaluation output exists")

    # Compile both fixed environmental inputs before any question tokenization.
    cpu_memory, memory_hash_before, _memory_metadata = load_scene1_memory_v86(config)
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
            raise RuntimeError("V86 fixed-final evaluation requires local MPS")
        collection = install_lora_banks(language.model, combined_lora_settings_v86(runtime, config))
        if not isinstance(collection, LoRABankCollection):
            raise TypeError("V86 evaluation LoRA installation failed")
        frozen_source = load_frozen_v85_stack_v86(
            collection, config["sources"]["frozen_checkpoint"]
        )
        candidate = load_fixed_final_bridge_v86(
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
                    "correct_answer_token_top1_accuracy": measured["answer_token_top1_accuracy"],
                    "scene_memory_sha256": memory_hash_before,
                    "layout_audit": layout,
                }
            )
            if ordinal == 1 or ordinal % 12 == 0 or ordinal == len(rows):
                print(
                    json.dumps(
                        {
                            "event": "v86_evaluation_row",
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

        causal_records: list[dict[str, Any]] = []
        prediction_by_id = {str(record["question_id"]): record for record in records}
        for row in causal_rows:
            zero_measured, zero_layout = _measure_nll_v84(language, system_prompt, zero_memory, row)
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
                    "scene_memory_sha256": memory_hash_before,
                }
            )
        memory_hash_after = prefix_sha256(memory.detach().cpu())
        zero_hash_after = prefix_sha256(zero_memory.detach().cpu())
    audit.assert_clean()

    prefix_hash_invariant = (
        memory_hash_after == memory_hash_before
        and all(record["scene_memory_sha256"] == memory_hash_before for record in records)
        and all(record["scene_memory_sha256"] == memory_hash_before for record in smoke_records)
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
    score = score_records_v86(
        rows,
        records,
        causal_records,
        smoke_records,
        gates=config["gates"],
        prefix_hash_invariant=prefix_hash_invariant,
        environment_input_invariant=environment_input_invariant,
        protected_read_count=protected_count,
    )
    predictions = {
        "artifact": PREDICTIONS_ARTIFACT,
        "schema_version": 86,
        "status": "fixed_final_evaluation_only_not_runtime",
        "config_sha256": training_bindings["config_sha256"],
        "training_report_sha256": training_bindings["training_report_sha256"],
        "row_count": len(records),
        "scene_count": 1,
        "fixed_checkpoint_selected_before_scoring": True,
        "checkpoint_selection_after_scoring": False,
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
            "optimizer_updates": 92,
        },
        "leakage": {
            "loaded_file_count": len(audit.unique_paths),
            "loaded_file_inventory_sha256": canonical_sha256_v85(audit.unique_paths),
            "protected_read_count": protected_count,
            "protected_reads": audit.forbidden_accesses(),
            "oracle_loaded": False,
        },
        "training_references_serialized_in_runtime_candidate": False,
        "runtime_promotion_authorized": False,
        "records": records,
        "causal_records": causal_records,
        "smoke_records": smoke_records,
    }
    prediction_output, prediction_sha = atomic_create_json_v85(predictions_path, predictions)
    report = {
        "artifact": EVALUATION_ARTIFACT,
        "schema_version": 86,
        "status": (
            "model_gates_pass_runtime_leakage_smoke_required"
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
        "held_out_generalization_claim": False,
        "v85_held_evidence_mutated": False,
        "runtime_oracle_unavailable_gate_pending": score["model_acceptance_gate_passed"],
        "runtime_file_audit_gate_pending": score["model_acceptance_gate_passed"],
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
    report = run_evaluation_v86(args.config)
    print(
        json.dumps(
            {
                "status": report["status"],
                "metrics": report["metrics"],
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
    "run_evaluation_v86",
    "score_records_v86",
]
