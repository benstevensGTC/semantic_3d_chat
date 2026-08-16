"""Evaluate V93's sealed scene-one termination/paraphrase repair candidate.

The fixed-final checkpoint is selected and authenticated before this module
scores anything.  Evaluation covers the unchanged 138-row canonical set,
thirteen development-known primary chat questions, exactly two entirely new
V93 wordings for each chat intent, and thirteen zero-payload causal controls.
Every conditioned call receives the same pre-question ``[1, 738, 1536]``
continuous scene memory; questions never select, retrieve, or rewrite scene
tokens.

This module only creates acceptance evidence.  It cannot package or promote a
runtime checkpoint, even when every preregistered model gate passes.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import torch

from semantic_3d_chat.chat.file_audit import FileAccessAudit
from semantic_3d_chat.chat.runtime_config import load_runtime_config
from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.evaluation.evaluate_v91_scene1_conversational_repair import (
    CORE_ACTIONABLE_INTENTS,
    INTENT_IDS,
    score_records_v91,
)
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
from semantic_3d_chat.evaluation.v93_scene1_termination_paraphrase_preflight import (
    CONFIG,
    SCENE_ID,
    load_config_v93,
)
from semantic_3d_chat.evaluation.v93_scene1_termination_paraphrase_preflight import (
    held_wording_rows_v93 as _preflight_held_wording_rows_v93,
)
from semantic_3d_chat.evaluation.v93_scene1_termination_paraphrase_preflight import (
    primary_rows_v93 as _preflight_primary_rows_v93,
)
from semantic_3d_chat.language.local_lm import load_local_language_model
from semantic_3d_chat.language.lora import LoRABankCollection, install_lora_banks
from semantic_3d_chat.language.prefix_injection import prefix_sha256
from semantic_3d_chat.training.train_question_control_v73 import RowV73
from semantic_3d_chat.training.train_v84_strict_bridge import (
    _generate_v84,
    _measure_nll_v84,
)

PREDICTIONS_ARTIFACT: Final[str] = "gemma4_v93_scene1_termination_paraphrase_repair_predictions_v1"
EVALUATION_ARTIFACT: Final[str] = "gemma4_v93_scene1_termination_paraphrase_repair_evaluation_v1"
EXPECTED_PREFIX_SHAPE: Final[tuple[int, int, int]] = (1, 738, 1536)
EXPECTED_EVALUATED_QUESTION_COUNT: Final[int] = 177
EXPECTED_CAUSAL_CONTROL_COUNT: Final[int] = 13
EXPECTED_FROZEN_PARENT_BANK_COUNT: Final[int] = 14
_FORBIDDEN_COMPONENTS: Final[frozenset[str]] = frozenset(
    {"oracle", "validation", "test", "deferred"}
)
_QUESTION_ID: Final[re.Pattern[str]] = re.compile(
    r"v(?:91|92|93)_(.+)_(?:existing_0[0-5]|new_held_0[01])"
)


def _intent_id_v93(record_or_row: Mapping[str, Any] | RowV73) -> str:
    """Resolve an intent from sealed opaque IDs, never scene metadata."""

    if isinstance(record_or_row, Mapping):
        explicit = record_or_row.get("intent_id")
        if isinstance(explicit, str) and explicit in INTENT_IDS:
            return explicit
    question_id = (
        record_or_row.question_id
        if isinstance(record_or_row, RowV73)
        else str(record_or_row.get("question_id", ""))
    )
    match = _QUESTION_ID.fullmatch(question_id)
    value = match.group(1) if match else None
    if value in INTENT_IDS:
        return str(value)
    raise ValueError(f"Malformed V93 conversational question ID: {question_id}")


def primary_rows_v93(config: Mapping[str, Any]) -> tuple[RowV73, ...]:
    """Return the exact thirteen development-known primary questions."""

    rows = _preflight_primary_rows_v93(config)
    if (
        len(rows) != 13
        or len({row.question_id for row in rows}) != 13
        or {_intent_id_v93(row) for row in rows} != set(INTENT_IDS)
        or any(not row.question_id.startswith("v91_") for row in rows)
    ):
        raise ValueError("V93 primary conversational inventory changed")
    return rows


def held_wording_rows_v93(config: Mapping[str, Any]) -> tuple[RowV73, ...]:
    """Return exactly twenty-six entirely new V93 wording-only holds."""

    rows = _preflight_held_wording_rows_v93(config)
    if (
        len(rows) != 26
        or len({row.question_id for row in rows}) != 26
        or Counter(_intent_id_v93(row) for row in rows)
        != Counter({identifier: 2 for identifier in INTENT_IDS})
        or any(not row.question_id.startswith("v93_") for row in rows)
    ):
        raise ValueError("V93 new held-wording inventory changed")
    return rows


def _with_intents(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Attach preregistered intent identity for V91's stable scorer."""

    return [{**record, "intent_id": _intent_id_v93(record)} for record in records]


def score_records_v93(
    canonical_rows: Sequence[RowV73],
    canonical_records: Sequence[Mapping[str, Any]],
    primary_records: Sequence[Mapping[str, Any]],
    held_records: Sequence[Mapping[str, Any]],
    causal_records: Sequence[Mapping[str, Any]],
    *,
    gates: Mapping[str, Any],
    prefix_hash_invariant: bool,
    environment_input_invariant: bool,
    parent_state_invariant: bool,
    candidate_state_invariant: bool,
    protected_read_count: int,
) -> dict[str, Any]:
    """Score V93 gates while retaining the audited V91 answer semantics."""

    score = score_records_v91(
        canonical_rows,
        canonical_records,
        _with_intents(primary_records),
        _with_intents(held_records),
        _with_intents(causal_records),
        gates=gates,
        prefix_hash_invariant=prefix_hash_invariant,
        environment_input_invariant=environment_input_invariant,
        parent_state_invariant=parent_state_invariant,
        protected_read_count=protected_read_count,
    )
    model_gates = dict(score["model_acceptance_gates"])
    model_gates["fixed_final_candidate_state_invariance"] = candidate_state_invariant
    passed = all(model_gates.values())
    score.update(
        {
            "model_acceptance_gates": model_gates,
            "model_acceptance_gate_passed": passed,
            "separate_runtime_packaging_authorized": passed,
            "runtime_oracle_unavailable_gate_pending": passed,
            "runtime_file_audit_gate_pending": passed,
            "automatic_runtime_promotion": False,
            "runtime_promotion_authorized": False,
        }
    )
    return score


def _evaluate_rows_v93(
    *,
    language: Any,
    system_prompt: str,
    memory: torch.Tensor,
    memory_hash: str,
    rows: Sequence[RowV73],
    max_new_tokens: int,
    phase: str,
    started: float,
) -> list[dict[str, Any]]:
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
        record: dict[str, Any] = {
            "scene_id": row.scene_id,
            "question_id": row.question_id,
            "answer_type": row.answer_type,
            "reference_answer": row.answer,
            "prediction": prediction,
            "normalized_prediction": normalize_answer(prediction),
            "correct_mean_nll": measured["mean_nll"],
            "correct_answer_token_top1_accuracy": measured["answer_token_top1_accuracy"],
            "scene_memory_sha256": memory_hash,
            "layout_audit": layout,
        }
        if phase != "canonical":
            record["question"] = row.question
            record["intent_id"] = _intent_id_v93(row)
        records.append(record)
        if ordinal == 1 or ordinal % 12 == 0 or ordinal == len(rows):
            print(
                json.dumps(
                    {
                        "event": "v93_evaluation_row",
                        "phase": phase,
                        "ordinal": ordinal,
                        "total": len(rows),
                        "question_id": row.question_id,
                        "elapsed_seconds": time.monotonic() - started,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    return records


def _layout_is_strict(record: Mapping[str, Any]) -> bool:
    layout = record.get("layout_audit")
    return bool(
        isinstance(layout, Mapping)
        and layout.get("memory_supplied_directly") is True
        and layout.get("memory_tokens") == 738
        and layout.get("control_tokens") == 0
        and layout.get("question_derived_environmental_tokens") == 0
        and layout.get("answer_only_supervision") is True
    )


def prompt_contract_v93(
    config: Mapping[str, Any],
    runtime_language: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind evaluation to V93's termination prompt and local model identity."""

    prompt = config.get("system_prompt")
    max_answer_tokens = config.get("max_answer_tokens")
    expected_model = {
        "model_id": config.get("model_id"),
        "revision": config.get("revision"),
        "dtype": config.get("dtype"),
    }
    runtime_model = {
        "model_id": runtime_language.get("model_id"),
        "revision": runtime_language.get("revision"),
        "dtype": runtime_language.get("dtype"),
    }
    if (
        not isinstance(prompt, str)
        or not prompt.strip()
        or prompt == runtime_language.get("system_prompt")
        or isinstance(max_answer_tokens, bool)
        or max_answer_tokens != 32
        or runtime_language.get("max_answer_tokens") != 32
        or expected_model != runtime_model
    ):
        raise ValueError("V93 evaluation prompt, generation cap, or model identity changed")
    return {
        "system_prompt_sha256": canonical_sha256_v85(prompt),
        "max_answer_tokens": max_answer_tokens,
        "model_id": expected_model["model_id"],
        "model_revision": expected_model["revision"],
        "dtype": expected_model["dtype"],
        "differs_from_v89_runtime_baseline": True,
    }


def run_evaluation_v93(config_path: str | Path = CONFIG) -> dict[str, Any]:
    """Run create-once V93 fixed-final evaluation with local Gemma/MPS."""

    # Deferred imports preserve model-free import/preflight behavior while the
    # independently sealed trainer and evaluator are assembled.
    from semantic_3d_chat.evaluation.v93_scene1_termination_paraphrase_preflight import (
        authenticate_sources_v93,
        load_canonical_rows_v93,
    )
    from semantic_3d_chat.training.train_v93_scene1_termination_paraphrase_repair import (
        authenticate_training_report_v93,
        combined_lora_settings_v93,
        load_fixed_final_bridge_v93,
        load_frozen_parent_v93,
    )

    started = time.monotonic()
    config = load_config_v93(config_path, allow_draft=False)
    source_hashes = authenticate_sources_v93(
        config,
        require_implementation_sources=True,
    )
    training_bindings = authenticate_training_report_v93(config, config_path=config_path)
    canonical_rows = load_canonical_rows_v93(config)
    primary_rows = primary_rows_v93(config)
    held_rows = held_wording_rows_v93(config)
    predictions_path = resolve_v85(config["outputs"]["evaluation_predictions"])
    report_path = resolve_v85(config["outputs"]["evaluation_report"])
    if predictions_path.exists() or report_path.exists():
        raise FileExistsError("V93 create-once evaluation output exists")

    cpu_memory, memory_hash_before, _metadata = load_scene1_memory_v86(config)
    if tuple(cpu_memory.shape) != EXPECTED_PREFIX_SHAPE:
        raise ValueError("V93 immutable scene-memory shape changed")
    cpu_zero_memory = zero_payload_memory_v86(cpu_memory)
    zero_hash_before = prefix_sha256(cpu_zero_memory)
    audit = FileAccessAudit(
        forbidden_component_names=_FORBIDDEN_COMPONENTS,
        block_forbidden=True,
    )
    with audit:
        runtime = load_runtime_config(config["sources"]["runtime_config"])
        language_config = runtime["language"]
        prompt_contract = prompt_contract_v93(config, language_config)
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
            raise RuntimeError("V93 fixed-final evaluation requires local MPS")
        collection = install_lora_banks(language.model, combined_lora_settings_v93(runtime, config))
        if not isinstance(collection, LoRABankCollection):
            raise TypeError("V93 evaluation LoRA installation failed")
        frozen_source = load_frozen_parent_v93(
            collection,
            config["sources"]["parent_v89_checkpoint"],
            config,
        )
        candidate = load_fixed_final_bridge_v93(
            collection, config["outputs"]["fixed_final_candidate"]
        )
        if (
            candidate.get("weights_sha256") != training_bindings["candidate_weights_sha256"]
            or candidate.get("state_sha256") != training_bindings["candidate_state_sha256"]
            or canonical_sha256_v85(candidate)
            != training_bindings["candidate_metadata_canonical_sha256"]
            or not isinstance(candidate.get("bindings"), Mapping)
            or candidate["bindings"].get("system_prompt_sha256")
            != prompt_contract["system_prompt_sha256"]
            or candidate["bindings"].get("max_answer_tokens")
            != prompt_contract["max_answer_tokens"]
        ):
            raise ValueError(
                "V93 loaded candidate or prompt differs from authenticated training evidence"
            )
        collection.eval()
        language.decoder_module.eval()
        frozen_states_before = {
            bank.settings.name: bank.installation.state_sha256()
            for bank in collection.banks
            if not bank.settings.trainable
        }
        if len(frozen_states_before) != EXPECTED_FROZEN_PARENT_BANK_COUNT:
            raise ValueError("V93 requires the exact frozen fourteen-bank parent")
        candidate_bank_name = str(config["bridge"]["bank_name"])
        candidate_state_before = collection.bank(candidate_bank_name).installation.state_sha256()
        memory = cpu_memory.to(device=language.device, dtype=torch.bfloat16)
        zero_memory = cpu_zero_memory.to(device=language.device, dtype=torch.bfloat16)
        system_prompt = str(config["system_prompt"])
        max_new_tokens = int(prompt_contract["max_answer_tokens"])
        canonical_records = _evaluate_rows_v93(
            language=language,
            system_prompt=system_prompt,
            memory=memory,
            memory_hash=memory_hash_before,
            rows=canonical_rows,
            max_new_tokens=max_new_tokens,
            phase="canonical",
            started=started,
        )
        primary_records = _evaluate_rows_v93(
            language=language,
            system_prompt=system_prompt,
            memory=memory,
            memory_hash=memory_hash_before,
            rows=primary_rows,
            max_new_tokens=max_new_tokens,
            phase="primary_conversational",
            started=started,
        )
        held_records = _evaluate_rows_v93(
            language=language,
            system_prompt=system_prompt,
            memory=memory,
            memory_hash=memory_hash_before,
            rows=held_rows,
            max_new_tokens=max_new_tokens,
            phase="new_v93_held_wording",
            started=started,
        )
        primary_by_id = {str(record["question_id"]): record for record in primary_records}
        causal_records: list[dict[str, Any]] = []
        for row in primary_rows:
            zero_measured, zero_layout = _measure_nll_v84(language, system_prompt, zero_memory, row)
            zero_prediction = _generate_v84(
                language,
                system_prompt,
                zero_memory,
                row,
                max_new_tokens=max_new_tokens,
            )
            correct = primary_by_id[row.question_id]
            causal_records.append(
                {
                    "scene_id": SCENE_ID,
                    "question_id": row.question_id,
                    "intent_id": _intent_id_v93(row),
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
            torch.mps.empty_cache()
        memory_hash_after = prefix_sha256(memory.detach().cpu())
        zero_hash_after = prefix_sha256(zero_memory.detach().cpu())
        frozen_states_after = {
            bank.settings.name: bank.installation.state_sha256()
            for bank in collection.banks
            if not bank.settings.trainable
        }
        candidate_state_after = collection.bank(candidate_bank_name).installation.state_sha256()
    audit.assert_clean()

    all_records = canonical_records + primary_records + held_records
    if len(all_records) != EXPECTED_EVALUATED_QUESTION_COUNT:
        raise RuntimeError("V93 evaluated question inventory changed")
    if len(causal_records) != EXPECTED_CAUSAL_CONTROL_COUNT:
        raise RuntimeError("V93 causal-control inventory changed")
    prefix_hash_invariant = bool(
        memory_hash_after == memory_hash_before
        and all(record["scene_memory_sha256"] == memory_hash_before for record in all_records)
    )
    environment_input_invariant = bool(
        prefix_hash_invariant
        and zero_hash_after == zero_hash_before
        and all(_layout_is_strict(record) for record in all_records)
        and all(
            record["correct_memory_sha256"] == memory_hash_before
            and record["zero_payload_memory_sha256"] == zero_hash_before
            and _layout_is_strict({"layout_audit": record["zero_layout_audit"]})
            for record in causal_records
        )
    )
    parent_state_invariant = frozen_states_after == frozen_states_before
    candidate_state_invariant = candidate_state_after == candidate_state_before
    protected_count = len(audit.forbidden_accesses())
    score = score_records_v93(
        canonical_rows,
        canonical_records,
        primary_records,
        held_records,
        causal_records,
        gates=config["gates"],
        prefix_hash_invariant=prefix_hash_invariant,
        environment_input_invariant=environment_input_invariant,
        parent_state_invariant=parent_state_invariant,
        candidate_state_invariant=candidate_state_invariant,
        protected_read_count=protected_count,
    )
    scene_memory = {
        "compiled_before_question_tokenization": True,
        "shape": list(EXPECTED_PREFIX_SHAPE),
        "continuous_environment_payload_tokens": 736,
        "prefix_sha256_before": memory_hash_before,
        "prefix_sha256_after": memory_hash_after,
        "zero_payload_prefix_sha256_before": zero_hash_before,
        "zero_payload_prefix_sha256_after": zero_hash_after,
        "prefix_hash_invariant": prefix_hash_invariant,
        "environment_conditioned_input_invariant": environment_input_invariant,
        "same_exact_memory_reused_for_all_177_questions": True,
        "question_derived_environmental_tokens": 0,
        "question_conditioned_environmental_readout": False,
        "question_dependent_scene_processing": False,
        "question_dependent_retrieval": False,
        "control_tokens": 0,
        "environmental_text_inputs": [],
    }
    leakage = {
        "loaded_file_count": len(audit.unique_paths),
        "loaded_file_inventory_sha256": canonical_sha256_v85(audit.unique_paths),
        "protected_read_count": protected_count,
        "protected_reads": audit.forbidden_accesses(),
        "forbidden_component_names": sorted(_FORBIDDEN_COMPONENTS),
        "oracle_loaded": False,
    }
    predictions = {
        "artifact": PREDICTIONS_ARTIFACT,
        "schema_version": 93,
        "status": "fixed_final_evaluation_only_not_runtime",
        "config_sha256": training_bindings["config_sha256"],
        "training_report_sha256": training_bindings["training_report_sha256"],
        "scene_id": SCENE_ID,
        "scene_count": 1,
        "canonical_row_count": len(canonical_records),
        "primary_conversational_row_count": len(primary_records),
        "new_v93_held_wording_row_count": len(held_records),
        "causal_control_row_count": len(causal_records),
        "fixed_checkpoint_selected_before_scoring": True,
        "checkpoint_selection_after_scoring": False,
        "newly_held_wording_only": True,
        "held_out_scene": False,
        "prompt_contract": prompt_contract,
        "scene_memory": scene_memory,
        "candidate": {
            "path": config["outputs"]["fixed_final_candidate"],
            "weights_sha256": candidate["weights_sha256"],
            "state_sha256": candidate["state_sha256"],
            "optimizer_updates": int(config["training"]["optimizer_updates"]),
        },
        "frozen_parent_bank_count": len(frozen_states_before),
        "frozen_parent_state_before": frozen_states_before,
        "frozen_parent_state_after": frozen_states_after,
        "frozen_parent_state_invariant": parent_state_invariant,
        "candidate_state_before": candidate_state_before,
        "candidate_state_after": candidate_state_after,
        "candidate_state_invariant": candidate_state_invariant,
        "leakage": leakage,
        "questions_or_answers_serialized_in_runtime_candidate": False,
        "training_inventory_serialized_in_runtime_candidate": False,
        "oracle_serialized_in_runtime_candidate": False,
        "runtime_promotion_authorized": False,
        "canonical_records": canonical_records,
        "primary_conversational_records": primary_records,
        "new_v93_held_wording_records": held_records,
        "causal_records": causal_records,
    }
    prediction_output, prediction_sha = atomic_create_json_v85(predictions_path, predictions)
    report = {
        "artifact": EVALUATION_ARTIFACT,
        "schema_version": 93,
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
        "prompt_contract": prompt_contract,
        "metrics": score,
        "scene_memory": scene_memory,
        "leakage": leakage,
        "post_v92_training_set_development": True,
        "single_scene_termination_paraphrase_repair": True,
        "development_known_primary_questions": True,
        "newly_held_wording_only": True,
        "held_out_scene": False,
        "held_out_scene_generalization_claim": False,
        "frozen_fourteen_bank_parent_mutated": not parent_state_invariant,
        "fixed_final_candidate_state_invariant": candidate_state_invariant,
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
    report = run_evaluation_v93(args.config)
    print(
        json.dumps(
            {
                "status": report["status"],
                "metrics": report["metrics"],
                "held_out_scene": False,
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
    "CORE_ACTIONABLE_INTENTS",
    "EVALUATION_ARTIFACT",
    "PREDICTIONS_ARTIFACT",
    "held_wording_rows_v93",
    "main",
    "primary_rows_v93",
    "prompt_contract_v93",
    "run_evaluation_v93",
    "score_records_v93",
]
