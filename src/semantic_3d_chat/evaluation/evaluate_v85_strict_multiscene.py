"""Evaluate the one fixed-final V85 checkpoint on pair/scene-disjoint development.

This command is permitted only after all 576 training rows and fixed update 72
have been published.  It evaluates that checkpoint exactly once, cannot select
or mutate a checkpoint, and leaves runtime promotion disabled.  Passing the
preregistered gates authorizes only a later, separate leakage/runtime-packaging
step.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import torch

from semantic_3d_chat.chat.file_audit import FileAccessAudit
from semantic_3d_chat.chat.runtime_config import load_runtime_config
from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.evaluation.v56_fresh_development_score import (
    canonical_answer_key,
    canonical_type_specific_match,
)
from semantic_3d_chat.evaluation.v85_strict_multiscene_preflight import (
    CONFIG,
    _authenticate_sources,
    atomic_create_json_v85,
    canonical_sha256_v85,
    load_config_v85,
    load_scene_memories_v85,
    resolve_v85,
    split_preflight_v85,
)
from semantic_3d_chat.language.local_lm import load_local_language_model
from semantic_3d_chat.language.lora import LoRABankCollection, install_lora_banks
from semantic_3d_chat.language.prefix_injection import prefix_sha256
from semantic_3d_chat.training.train_question_control_v73 import (
    ChangedUnitV73,
    RowV73,
    changed_units_v73,
)
from semantic_3d_chat.training.train_v84_strict_bridge import (
    _generate_v84,
    _measure_nll_v84,
)
from semantic_3d_chat.training.train_v85_strict_multiscene import (
    authenticate_training_report_v85,
    combined_lora_settings_v85,
    load_fixed_final_bridge_v85,
    load_frozen_v54_banks_v85,
)

PREDICTIONS_ARTIFACT: Final[str] = (
    "gemma4_v85_strict_multiscene_development_predictions_v1"
)
SCORE_ARTIFACT: Final[str] = "gemma4_v85_strict_multiscene_development_score_v1"
_FORBIDDEN_COMPONENTS: Final[frozenset[str]] = frozenset(
    {"oracle", "validation", "test", "deferred", "final"}
)


def _aggregate_accuracy(
    records: Sequence[Mapping[str, Any]], field: str
) -> dict[str, Any]:
    correct = sum(bool(record[field]) for record in records)
    return {
        "correct": correct,
        "total": len(records),
        "accuracy": correct / len(records) if records else 0.0,
    }


def _canonical_key(row: RowV73, prediction: str) -> object | None:
    return canonical_answer_key(row.answer_type, prediction)


def _unit_index(
    units: Sequence[ChangedUnitV73],
) -> dict[tuple[str, str], tuple[ChangedUnitV73, RowV73]]:
    result: dict[tuple[str, str], tuple[ChangedUnitV73, RowV73]] = {}
    for unit in units:
        result[unit.left.key] = unit, unit.right
        result[unit.right.key] = unit, unit.left
    return result


def score_records_v85(
    rows: Sequence[RowV73],
    records: Sequence[Mapping[str, Any]],
    *,
    gates: Mapping[str, Any],
    prefix_hash_invariant: bool,
    every_memory_hash_retained: bool,
    protected_read_count: int,
) -> dict[str, Any]:
    if len(rows) != 384 or len(records) != len(rows):
        raise ValueError("V85 development score requires all 384 rows")
    row_by_key = {row.key: row for row in rows}
    record_by_key = {
        (str(record["scene_id"]), str(record["question_id"])): record
        for record in records
    }
    if set(row_by_key) != set(record_by_key):
        raise ValueError("V85 development prediction keys changed")
    units = changed_units_v73(rows)
    unit_lookup = _unit_index(units)

    scored: list[dict[str, Any]] = []
    for key, row in row_by_key.items():
        source = record_by_key[key]
        prediction = str(source["correct_scene_prediction"])
        canonical_correct = canonical_type_specific_match(
            row.answer_type, prediction, row.answer
        )
        strict_exact = str(source["normalized_prediction"]) == row.answer
        item = {
            **source,
            "canonical_correct": canonical_correct,
            "strict_normalized_exact": strict_exact,
        }
        if row.expected_change:
            _unit, opposite = unit_lookup[row.key]
            wrong_prediction = str(source["paired_wrong_scene_prediction"])
            item["wrong_scene_matches_opposite_target"] = canonical_type_specific_match(
                opposite.answer_type, wrong_prediction, opposite.answer
            )
        scored.append(item)

    canonical = _aggregate_accuracy(scored, "canonical_correct")
    strict = _aggregate_accuracy(scored, "strict_normalized_exact")
    by_answer_type: dict[str, Any] = {}
    for answer_type in sorted({row.answer_type for row in rows}):
        selected = [
            record
            for record in scored
            if row_by_key[(str(record["scene_id"]), str(record["question_id"]))].answer_type
            == answer_type
        ]
        by_answer_type[answer_type] = _aggregate_accuracy(selected, "canonical_correct")

    changed_records = [
        record
        for record in scored
        if row_by_key[(str(record["scene_id"]), str(record["question_id"]))].expected_change
    ]
    changed_correct = _aggregate_accuracy(changed_records, "canonical_correct")
    wrong_opposite = _aggregate_accuracy(
        changed_records, "wrong_scene_matches_opposite_target"
    )
    complete_units = 0
    prediction_changing_units = 0
    unit_details: list[dict[str, Any]] = []
    family_values: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for unit in units:
        left = record_by_key[unit.left.key]
        right = record_by_key[unit.right.key]
        left_correct = canonical_type_specific_match(
            unit.left.answer_type,
            str(left["correct_scene_prediction"]),
            unit.left.answer,
        )
        right_correct = canonical_type_specific_match(
            unit.right.answer_type,
            str(right["correct_scene_prediction"]),
            unit.right.answer,
        )
        changed = _canonical_key(
            unit.left, str(left["correct_scene_prediction"])
        ) != _canonical_key(unit.right, str(right["correct_scene_prediction"]))
        complete_units += int(left_correct and right_correct)
        prediction_changing_units += int(changed)
        detail = {
            "pair_id": unit.pair_id,
            "question_key": unit.question_key,
            "change_type": unit.change_type,
            "both_correct": left_correct and right_correct,
            "canonical_prediction_changed": changed,
            "mean_wrong_minus_correct_nll": (
                float(left["wrong_minus_correct_nll"])
                + float(right["wrong_minus_correct_nll"])
            )
            / 2,
        }
        unit_details.append(detail)
        family_values[unit.change_type].append(detail)

    by_change_family = {
        family: {
            "units": len(values),
            "complete_units": sum(bool(value["both_correct"]) for value in values),
            "prediction_changing_units": sum(
                bool(value["canonical_prediction_changed"]) for value in values
            ),
            "mean_wrong_minus_correct_nll": sum(
                float(value["mean_wrong_minus_correct_nll"]) for value in values
            )
            / len(values),
        }
        for family, values in sorted(family_values.items())
    }
    mean_nll = sum(float(record["correct_scene_mean_nll"]) for record in scored) / len(
        scored
    )
    mean_wrong_minus_correct = sum(
        float(record["wrong_minus_correct_nll"]) for record in scored
    ) / len(scored)
    mean_changed_margin = sum(
        float(record["wrong_minus_correct_nll"]) for record in changed_records
    ) / len(changed_records)
    majority_count = max(Counter(row.answer_class for row in rows).values())
    majority_accuracy = majority_count / len(rows)
    canonical_threshold = max(
        float(gates["canonical_accuracy_minimum"]),
        majority_accuracy
        + float(gates["canonical_accuracy_margin_over_answer_frequency_majority"]),
    )
    spatial = by_answer_type.get("spatial_relation", {"total": 0, "accuracy": 0.0})
    enough_spatial = int(spatial["total"]) >= int(
        gates["spatial_relation_minimum_row_count"]
    )
    gate_results = {
        "canonical_accuracy_at_least_preregistered_threshold": canonical["accuracy"]
        >= canonical_threshold,
        "spatial_relation_accuracy_at_least_0_45": enough_spatial
        and float(spatial["accuracy"])
        >= float(gates["spatial_relation_accuracy_minimum"]),
        "mean_changed_side_wrong_minus_correct_nll_strictly_positive": mean_changed_margin
        > 0.0,
        "complete_changed_units_at_least_4": complete_units
        >= int(gates["complete_changed_units_minimum"]),
        "canonical_prediction_changing_units_at_least_8": prediction_changing_units
        >= int(gates["canonical_prediction_changing_units_minimum"]),
        "exact_prefix_hash_invariance": prefix_hash_invariant,
        "every_development_memory_hash_retained": every_memory_hash_retained,
        "protected_read_count_zero": protected_read_count
        <= int(gates["protected_read_count_maximum"]),
    }
    passed = all(gate_results.values())
    return {
        "strict_normalized_exact": strict,
        "canonical_type_specific": canonical,
        "canonical_accuracy_by_answer_type": by_answer_type,
        "answer_token_mean_nll": mean_nll,
        "mean_wrong_minus_correct_nll_all_rows": mean_wrong_minus_correct,
        "mean_wrong_minus_correct_nll_changed_sides": mean_changed_margin,
        "changed_side_correct_scene": changed_correct,
        "changed_side_wrong_scene_matches_opposite_target": wrong_opposite,
        "changed_complete_units": complete_units,
        "canonical_prediction_changing_units": prediction_changing_units,
        "changed_unit_total": len(units),
        "changed_metrics_by_change_family": by_change_family,
        "unit_details": unit_details,
        "answer_frequency_majority_baseline": {
            "correct": majority_count,
            "total": len(rows),
            "accuracy": majority_accuracy,
        },
        "preregistered_canonical_accuracy_threshold": canonical_threshold,
        "spatial_relation_row_count_sufficient": enough_spatial,
        "runtime_candidate_gates": gate_results,
        "runtime_candidate_gate_passed": passed,
        "separate_leakage_runtime_packaging_authorized": passed,
        "automatic_runtime_promotion": False,
    }


def run_development_evaluation_v85(
    config_path: str | Path = CONFIG,
) -> dict[str, Any]:
    started = time.monotonic()
    config = load_config_v85(config_path)
    source_hashes = _authenticate_sources(config)
    split_report, _train_rows, development_rows = split_preflight_v85(config)
    training_bindings = authenticate_training_report_v85(
        config, config_path=config_path
    )
    predictions_path = resolve_v85(config["outputs"]["development_predictions"])
    score_path = resolve_v85(config["outputs"]["development_score"])
    if predictions_path.exists() or score_path.exists():
        raise FileExistsError("V85 create-once development output exists")

    # Compile and hash all sixteen full memories before Gemma sees a question.
    cpu_memories, memory_hashes_before = load_scene_memories_v85(
        config, development_rows, split_name="development"
    )
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
            raise RuntimeError("V85 fixed-final development evaluation requires local MPS")
        collection = install_lora_banks(
            language.model, combined_lora_settings_v85(runtime, config)
        )
        if not isinstance(collection, LoRABankCollection):
            raise TypeError("V85 development LoRA installation failed")
        frozen_source = load_frozen_v54_banks_v85(
            collection, config["sources"]["base_checkpoint"]
        )
        candidate = load_fixed_final_bridge_v85(
            collection, config["outputs"]["fixed_final_candidate"]
        )
        collection.assert_trainable_surface(language.model)
        collection.eval()
        language.decoder_module.eval()
        memory_by_scene = {
            scene_id: memory.to(device=language.device, dtype=torch.bfloat16)
            for scene_id, memory in cpu_memories.items()
        }
        system_prompt = str(language_config["system_prompt"])
        max_new_tokens = int(language_config["max_answer_tokens"])
        changed_keys = {
            row.key for unit in changed_units_v73(development_rows) for row in (unit.left, unit.right)
        }
        records: list[dict[str, Any]] = []
        from semantic_3d_chat.evaluation.metrics import normalize_answer

        for ordinal, row in enumerate(development_rows, 1):
            correct, layout = _measure_nll_v84(
                language, system_prompt, memory_by_scene[row.scene_id], row
            )
            wrong, _wrong_layout = _measure_nll_v84(
                language, system_prompt, memory_by_scene[row.paired_scene_id], row
            )
            prediction = _generate_v84(
                language,
                system_prompt,
                memory_by_scene[row.scene_id],
                row,
                max_new_tokens=max_new_tokens,
            )
            wrong_prediction = (
                _generate_v84(
                    language,
                    system_prompt,
                    memory_by_scene[row.paired_scene_id],
                    row,
                    max_new_tokens=max_new_tokens,
                )
                if row.key in changed_keys
                else None
            )
            records.append(
                {
                    "scene_id": row.scene_id,
                    "question_id": row.question_id,
                    "pair_id": row.pair_id,
                    "paired_scene_id": row.paired_scene_id,
                    "question_key": row.question_key,
                    "change_type": row.change_type,
                    "answer_type": row.answer_type,
                    "expected_change": row.expected_change,
                    "reference_answer": row.answer,
                    "correct_scene_prediction": prediction,
                    "normalized_prediction": normalize_answer(prediction),
                    "paired_wrong_scene_prediction": wrong_prediction,
                    "correct_scene_mean_nll": correct["mean_nll"],
                    "correct_scene_answer_token_top1_accuracy": correct[
                        "answer_token_top1_accuracy"
                    ],
                    "paired_wrong_scene_mean_nll": wrong["mean_nll"],
                    "wrong_minus_correct_nll": wrong["mean_nll"]
                    - correct["mean_nll"],
                    "scene_memory_sha256": memory_hashes_before[row.scene_id],
                    "paired_wrong_memory_sha256": memory_hashes_before[
                        row.paired_scene_id
                    ],
                    "layout_audit": layout,
                }
            )
            if ordinal == 1 or ordinal % 24 == 0 or ordinal == len(development_rows):
                print(
                    json.dumps(
                        {
                            "event": "v85_development_row",
                            "ordinal": ordinal,
                            "total": len(development_rows),
                            "scene_id": row.scene_id,
                            "question_id": row.question_id,
                            "elapsed_seconds": time.monotonic() - started,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            torch.mps.empty_cache()

        memory_hashes_after = {
            scene_id: prefix_sha256(memory.detach().cpu())
            for scene_id, memory in memory_by_scene.items()
        }
    audit.assert_clean()
    prefix_hash_invariant = memory_hashes_after == memory_hashes_before and all(
        record["scene_memory_sha256"]
        == memory_hashes_before[str(record["scene_id"])]
        for record in records
    )
    every_memory_hash_retained = (
        len(memory_hashes_before) == config["split"]["development_scene_count"]
        and set(memory_hashes_before) == {row.scene_id for row in development_rows}
        and all(len(value) == 64 for value in memory_hashes_before.values())
    )
    protected_count = len(audit.forbidden_accesses())
    score = score_records_v85(
        development_rows,
        records,
        gates=config["runtime_candidate_gates"],
        prefix_hash_invariant=prefix_hash_invariant,
        every_memory_hash_retained=every_memory_hash_retained,
        protected_read_count=protected_count,
    )
    predictions = {
        "artifact": PREDICTIONS_ARTIFACT,
        "schema_version": 85,
        "status": "fixed_final_development_predictions_not_runtime",
        "config_sha256": training_bindings["config_sha256"],
        "training_report_sha256": training_bindings["training_report_sha256"],
        "fixed_checkpoint_selected_before_development": True,
        "development_driven_checkpoint_selection": False,
        "row_count": len(records),
        "scene_count": len(memory_hashes_before),
        "changed_side_count": len(changed_keys),
        "scene_memory": {
            "compiled_before_question_tokenization": True,
            "shape_each": [1, 738, 1536],
            "hashes_before": memory_hashes_before,
            "hashes_after": memory_hashes_after,
            "prefix_hash_invariant": prefix_hash_invariant,
            "every_development_memory_hash_retained": every_memory_hash_retained,
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
            "optimizer_updates": 72,
        },
        "leakage": {
            "loaded_file_count": len(audit.unique_paths),
            "loaded_file_inventory_sha256": canonical_sha256_v85(audit.unique_paths),
            "protected_read_count": protected_count,
            "protected_reads": audit.forbidden_accesses(),
            "official_validation_loaded": False,
            "official_test_loaded": False,
            "deferred_final_loaded": False,
            "sealed_historical_16_loaded": False,
            "oracle_loaded": False,
        },
        "behavior_scored_in_prediction_process": True,
        "runtime_promotion_authorized": False,
        "records": records,
    }
    prediction_output, prediction_sha = atomic_create_json_v85(
        predictions_path, predictions
    )
    report = {
        "artifact": SCORE_ARTIFACT,
        "schema_version": 85,
        "status": (
            "runtime_candidate_gate_pass_separate_packaging_required"
            if score["runtime_candidate_gate_passed"]
            else "runtime_candidate_gate_fail_diagnostic_only"
        ),
        "config_sha256": training_bindings["config_sha256"],
        "preregistration_sha256": training_bindings["preregistration_sha256"],
        "cpu_preflight_sha256": training_bindings["cpu_preflight_sha256"],
        "training_report_sha256": training_bindings["training_report_sha256"],
        "development_predictions_path": prediction_output.relative_to(
            PROJECT_ROOT
        ).as_posix(),
        "development_predictions_sha256": prediction_sha,
        "source_hashes": source_hashes,
        "frozen_source": frozen_source,
        "split_preflight": split_report,
        "fixed_checkpoint_selected_before_development": True,
        "checkpoint_selection_after_scoring": False,
        "preregistered_runtime_candidate_gates": config["runtime_candidate_gates"],
        "metrics": score,
        "scene_memory": predictions["scene_memory"],
        "leakage": predictions["leakage"],
        "separate_leakage_runtime_packaging_authorized": score[
            "separate_leakage_runtime_packaging_authorized"
        ],
        "automatic_runtime_promotion": False,
        "runtime_promotion_authorized": False,
        "official_validation_loaded": False,
        "official_test_loaded": False,
        "deferred_final_loaded": False,
        "sealed_historical_16_loaded": False,
        "oracle_loaded": False,
        "elapsed_seconds": time.monotonic() - started,
    }
    atomic_create_json_v85(score_path, report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(CONFIG))
    args = parser.parse_args(argv)
    report = run_development_evaluation_v85(args.config)
    print(
        json.dumps(
            {
                "status": report["status"],
                "metrics": report["metrics"],
                "separate_leakage_runtime_packaging_authorized": report[
                    "separate_leakage_runtime_packaging_authorized"
                ],
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
    "PREDICTIONS_ARTIFACT",
    "SCORE_ARTIFACT",
    "main",
    "run_development_evaluation_v85",
    "score_records_v85",
]
