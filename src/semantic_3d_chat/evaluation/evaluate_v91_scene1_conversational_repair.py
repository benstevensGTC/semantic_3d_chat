"""Evaluate V91's fixed-final scene-one conversational repair candidate.

This is an offline, development-known evaluation.  It scores the unchanged
138 canonical questions, the thirteen primary conversational questions, two
new wording-only holds for each of the thirteen intents, and a zero-payload
control for every primary question.  Every model call receives the same
pre-question ``[1, 738, 1536]`` continuous scene memory.  No scene caption,
object list, oracle label, or question-conditioned scene readout is supplied.

The evaluator is deliberately create-once and never promotes a runtime
checkpoint.  Passing model gates can only authorize a later, separately
audited packaging step in which the oracle directory is physically absent.
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
from semantic_3d_chat.evaluation.metrics import (
    canonical_presence,
    canonical_relation,
    normalize_answer,
)
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
    load_scene1_memory_v86,
    zero_payload_memory_v86,
)
from semantic_3d_chat.evaluation.v91_scene1_conversational_preflight import (
    CONFIG,
    SCENE_ID,
    load_config_v91,
)
from semantic_3d_chat.evaluation.v91_scene1_conversational_preflight import (
    held_wording_rows_v91 as _preflight_held_wording_rows_v91,
)
from semantic_3d_chat.evaluation.v91_scene1_conversational_preflight import (
    primary_rows_v91 as _preflight_primary_rows_v91,
)
from semantic_3d_chat.language.local_lm import load_local_language_model
from semantic_3d_chat.language.lora import LoRABankCollection, install_lora_banks
from semantic_3d_chat.language.prefix_injection import prefix_sha256
from semantic_3d_chat.training.train_question_control_v73 import RowV73
from semantic_3d_chat.training.train_v84_strict_bridge import (
    _generate_v84,
    _measure_nll_v84,
)

PREDICTIONS_ARTIFACT: Final[str] = (
    "gemma4_v91_scene1_conversational_repair_predictions_v1"
)
EVALUATION_ARTIFACT: Final[str] = (
    "gemma4_v91_scene1_conversational_repair_evaluation_v1"
)
EXPECTED_PREFIX_SHAPE: Final[tuple[int, int, int]] = (1, 738, 1536)
EXPECTED_EVALUATED_QUESTION_COUNT: Final[int] = 177
_FORBIDDEN_COMPONENTS: Final[frozenset[str]] = frozenset(
    {"oracle", "validation", "test", "deferred"}
)
CORE_ACTIONABLE_INTENTS: Final[frozenset[str]] = frozenset(
    {
        "table_contents",
        "under_table",
        "wall_object",
        "cube_location",
        "sitting",
        "bowl_contents",
    }
)
INTENT_IDS: Final[frozenset[str]] = frozenset(
    {
        "inventory",
        "chair_presence",
        "bowl_color",
        "bowl_left_chair",
        "table_contents",
        "under_table",
        "closest",
        "wall_object",
        "cube_location",
        "lamp_turn",
        "frame_support",
        "sitting",
        "bowl_contents",
    }
)
_LIST_INTENTS: Final[frozenset[str]] = frozenset(
    {"inventory", "table_contents"}
)
_OBJECT_VOCABULARY: Final[tuple[str, ...]] = (
    "picture frame",
    "floor lamp",
    "plant pot",
    "television",
    "cabinet",
    "ceiling",
    "window",
    "chair",
    "table",
    "bowl",
    "cube",
    "book",
    "floor",
    "wall",
    "door",
    "sofa",
    "sink",
    "bed",
)
_EXPECTED_LIST_ITEMS: Final[dict[str, frozenset[str]]] = {
    "inventory": frozenset(
        {
            "table",
            "chair",
            "picture frame",
            "bowl",
            "floor lamp",
            "cube",
            "book",
            "cabinet",
            "plant pot",
        }
    ),
    "table_contents": frozenset({"book", "cube"}),
}
_EMPTY_ANSWERS: Final[frozenset[str]] = frozenset(
    {"nothing", "none", "no", "no object", "no objects", "empty"}
)


def _intent_id(record_or_row: Mapping[str, Any] | RowV73) -> str:
    """Resolve a V91 intent without consulting environmental metadata."""

    if isinstance(record_or_row, Mapping):
        explicit = record_or_row.get("intent_id")
        if isinstance(explicit, str) and explicit in INTENT_IDS:
            return explicit
    question_id = (
        record_or_row.question_id
        if isinstance(record_or_row, RowV73)
        else str(record_or_row.get("question_id", ""))
    )
    match = re.fullmatch(
        r"v91_(.+)_(?:existing_(?:00)|new_held_(?:00|01))", question_id
    )
    value = match.group(1) if match else None
    if value in INTENT_IDS:
        return value
    raise ValueError(f"Malformed V91 conversational question ID: {question_id}")


def primary_rows_v91(config: Mapping[str, Any]) -> tuple[RowV73, ...]:
    """Return the exact thirteen preflight-sealed primary questions."""

    rows = _preflight_primary_rows_v91(config)
    if (
        len(rows) != 13
        or len({row.question_id for row in rows}) != 13
        or {_intent_id(row) for row in rows} != INTENT_IDS
    ):
        raise ValueError("V91 primary conversational inventory changed")
    return rows


def held_wording_rows_v91(config: Mapping[str, Any]) -> tuple[RowV73, ...]:
    """Return the twenty-six new, training-disjoint wording-only holds."""

    rows = _preflight_held_wording_rows_v91(config)
    if len(rows) != 26 or len({row.question_id for row in rows}) != 26:
        raise ValueError("V91 new held-wording inventory changed")
    if Counter(_intent_id(row) for row in rows) != Counter(
        {identifier: 2 for identifier in INTENT_IDS}
    ):
        raise ValueError("V91 new held-wording intent coverage changed")
    return rows


def _extract_object_items(value: Any) -> frozenset[str]:
    normalized = normalize_answer(value)
    remaining = f" {normalized} "
    found: set[str] = set()
    # Longer names must be consumed first: ``floor lamp`` cannot also count as
    # an independent ``floor`` item unless floor appears elsewhere in the text.
    for item in sorted(_OBJECT_VOCABULARY, key=lambda name: (-len(name), name)):
        pattern = re.compile(rf"(?<!\S){re.escape(item)}(?!\S)")
        if pattern.search(remaining):
            found.add(item)
            remaining = pattern.sub(" ", remaining)
    return frozenset(found)


def conversational_answer_key_v91(
    intent_id: str,
    family: str,
    value: Any,
) -> object | None:
    """Canonicalize chat answers with V90's robust semantic conventions."""

    if intent_id not in INTENT_IDS:
        raise ValueError(f"Unknown V91 conversational intent: {intent_id}")
    if intent_id in _LIST_INTENTS:
        return tuple(sorted(_extract_object_items(value)))
    if family in {"presence", "containment"}:
        return canonical_presence(value)
    if family in {"spatial_relation", "viewpoint_direction"}:
        return canonical_relation(value)
    normalized = normalize_answer(value)
    if family == "empty_support" and normalized in _EMPTY_ANSWERS:
        return "nothing"
    if family == "wall_object" and normalized in {"frame", "picture frame"}:
        return "picture frame"
    if family == "object_location" and normalized in {
        "table",
        "on table",
        "on tabletop",
        "on top of table",
    }:
        return "on table"
    if family == "frame_support" and normalized in {
        "wall",
        "on wall",
        "wall mounted",
        "mounted on wall",
    }:
        return "wall"
    return normalized if normalized else None


def conversational_match_v91(
    intent_id: str,
    family: str,
    prediction: Any,
    reference: Any,
) -> bool:
    """Apply deterministic semantic matching and reject extra list objects."""

    expected = conversational_answer_key_v91(intent_id, family, reference)
    observed = conversational_answer_key_v91(intent_id, family, prediction)
    if intent_id in _LIST_INTENTS:
        expected_items = _EXPECTED_LIST_ITEMS[intent_id]
        return expected == tuple(sorted(expected_items)) and observed == expected
    return expected is not None and observed == expected


def _score_conversational_records(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_per_intent: int,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    scored: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for record in records:
        identifier = _intent_id(record)
        family = str(record["answer_type"])
        counts[identifier] += 1
        correct = conversational_match_v91(
            identifier,
            family,
            record["prediction"],
            record["reference_answer"],
        )
        scored.append(
            {
                **record,
                "intent_id": identifier,
                "canonical_prediction": conversational_answer_key_v91(
                    identifier, family, record["prediction"]
                ),
                "canonical_correct": correct,
                "strict_normalized_exact": normalize_answer(record["prediction"])
                == normalize_answer(record["reference_answer"]),
            }
        )
    if counts != Counter(
        {identifier: expected_per_intent for identifier in INTENT_IDS}
    ):
        raise ValueError(
            "V91 conversational prediction coverage differs from thirteen intents"
        )
    by_intent: dict[str, dict[str, Any]] = {}
    for identifier in sorted(INTENT_IDS):
        selected = [record for record in scored if record["intent_id"] == identifier]
        correct = sum(bool(record["canonical_correct"]) for record in selected)
        by_intent[identifier] = {
            "correct": correct,
            "total": len(selected),
            "accuracy": correct / len(selected),
        }
    return scored, by_intent


def score_records_v91(
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
    protected_read_count: int,
) -> dict[str, Any]:
    """Score every preregistered V91 model gate without writing artifacts."""

    if int(gates["canonical_total"]) != 138:
        raise ValueError("V91 preregistered canonical total changed")
    if len(canonical_rows) != 138 or len(canonical_records) != 138:
        raise ValueError("V91 canonical retention requires all 138 rows")
    row_by_id = {row.question_id: row for row in canonical_rows}
    record_ids = {str(record.get("question_id")) for record in canonical_records}
    if len(row_by_id) != 138 or record_ids != set(row_by_id):
        raise ValueError("V91 canonical prediction inventory differs")

    canonical_scored: list[dict[str, Any]] = []
    for record in canonical_records:
        row = row_by_id[str(record["question_id"])]
        prediction = str(record["prediction"])
        canonical_scored.append(
            {
                **record,
                "canonical_prediction": canonical_answer_key(
                    row.answer_type, prediction
                ),
                "canonical_correct": canonical_type_specific_match(
                    row.answer_type, prediction, row.answer
                ),
                "strict_normalized_exact": normalize_answer(prediction)
                == normalize_answer(row.answer),
            }
        )
    canonical_by_type: dict[str, dict[str, Any]] = {}
    for answer_type in sorted({row.answer_type for row in canonical_rows}):
        selected = [
            record
            for record in canonical_scored
            if row_by_id[str(record["question_id"])].answer_type == answer_type
        ]
        correct = sum(bool(record["canonical_correct"]) for record in selected)
        canonical_by_type[answer_type] = {
            "correct": correct,
            "total": len(selected),
            "accuracy": correct / len(selected),
        }
    expected_types = {
        "presence",
        "count",
        "metric",
        "attribute",
        "spatial_relation",
        "support",
    }
    if set(canonical_by_type) != expected_types:
        raise ValueError("V91 canonical answer-type inventory changed")
    canonical_correct = sum(
        bool(record["canonical_correct"]) for record in canonical_scored
    )
    canonical_strict = sum(
        bool(record["strict_normalized_exact"]) for record in canonical_scored
    )

    if len(primary_records) != int(gates["primary_conversational_total"]):
        raise ValueError("V91 primary conversational row count differs")
    if len(held_records) != int(gates["new_held_wording_total"]):
        raise ValueError("V91 new held-wording row count differs")
    primary_scored, primary_by_intent = _score_conversational_records(
        primary_records, expected_per_intent=1
    )
    held_scored, held_by_intent = _score_conversational_records(
        held_records, expected_per_intent=2
    )
    primary_correct = sum(
        bool(record["canonical_correct"]) for record in primary_scored
    )
    held_correct = sum(bool(record["canonical_correct"]) for record in held_scored)
    if int(gates["core_actionable_total"]) != len(CORE_ACTIONABLE_INTENTS):
        raise ValueError("V91 core actionable gate inventory changed")
    core_correct = sum(
        primary_by_intent[identifier]["correct"]
        for identifier in CORE_ACTIONABLE_INTENTS
    )

    if len(causal_records) != 13 or {
        _intent_id(record) for record in causal_records
    } != INTENT_IDS:
        raise ValueError("V91 causal control requires all thirteen primary intents")
    causal_margin = sum(
        float(record["zero_minus_correct_nll"]) for record in causal_records
    ) / len(causal_records)
    causal_changes = sum(
        conversational_answer_key_v91(
            _intent_id(record),
            str(record["answer_type"]),
            record["correct_prediction"],
        )
        != conversational_answer_key_v91(
            _intent_id(record),
            str(record["answer_type"]),
            record["zero_prediction"],
        )
        for record in causal_records
    )
    target_margin = float(gates["causal_mean_zero_minus_correct_nll_minimum"])
    model_gates = {
        "canonical_correct_at_least_preregistered_minimum": canonical_correct
        >= int(gates["canonical_correct_minimum"]),
        "canonical_presence_correct_at_least_minimum": canonical_by_type[
            "presence"
        ]["correct"]
        >= int(gates["canonical_presence_correct_minimum"]),
        "canonical_count_correct_at_least_minimum": canonical_by_type["count"][
            "correct"
        ]
        >= int(gates["canonical_count_correct_minimum"]),
        "canonical_metric_correct_at_least_minimum": canonical_by_type["metric"][
            "correct"
        ]
        >= int(gates["canonical_metric_correct_minimum"]),
        "canonical_attribute_correct_at_least_minimum": canonical_by_type[
            "attribute"
        ]["correct"]
        >= int(gates["canonical_attribute_correct_minimum"]),
        "canonical_spatial_correct_at_least_minimum": canonical_by_type[
            "spatial_relation"
        ]["correct"]
        >= int(gates["canonical_spatial_correct_minimum"]),
        "canonical_support_correct_at_least_minimum": canonical_by_type[
            "support"
        ]["correct"]
        >= int(gates["canonical_support_correct_minimum"]),
        "primary_conversational_correct_at_least_required": primary_correct
        >= int(gates["primary_conversational_required_correct"]),
        "all_six_core_actionable_intents_correct": core_correct
        == int(gates["core_actionable_required_correct"]),
        "new_held_wording_correct_at_least_required": held_correct
        >= int(gates["new_held_wording_required_correct"]),
        "new_held_wording_each_intent_at_least_minimum": all(
            result["correct"]
            >= int(gates["new_held_wording_each_intent_minimum"])
            for result in held_by_intent.values()
        ),
        "causal_correct_memory_mean_nll_at_least_preregistered_margin_below_zero_payload": causal_margin
        >= target_margin,
        "causal_prediction_changes_at_least_required": causal_changes
        >= int(gates["causal_prediction_change_minimum"]),
        "exact_prefix_hash_invariance": prefix_hash_invariant,
        "exact_total_environment_input_invariance": environment_input_invariant,
        "frozen_parent_state_invariance": parent_state_invariant,
        "protected_read_count_at_most_preregistered_maximum": protected_read_count
        <= int(gates["forbidden_runtime_read_count_maximum"]),
    }
    passed = all(model_gates.values())
    return {
        "canonical_type_specific": {
            "correct": canonical_correct,
            "total": len(canonical_scored),
            "accuracy": canonical_correct / len(canonical_scored),
        },
        "canonical_strict_normalized_exact": {
            "correct": canonical_strict,
            "total": len(canonical_scored),
            "accuracy": canonical_strict / len(canonical_scored),
        },
        "canonical_accuracy_by_answer_type": canonical_by_type,
        "canonical_answer_token_mean_nll": sum(
            float(record["correct_mean_nll"]) for record in canonical_records
        )
        / len(canonical_records),
        "primary_conversational": {
            "correct": primary_correct,
            "total": len(primary_scored),
            "accuracy": primary_correct / len(primary_scored),
            "core_actionable_correct": core_correct,
            "core_actionable_total": len(CORE_ACTIONABLE_INTENTS),
            "by_intent": primary_by_intent,
            "records": primary_scored,
        },
        "new_held_wording": {
            "correct": held_correct,
            "total": len(held_scored),
            "accuracy": held_correct / len(held_scored),
            "newly_held_wording_only": True,
            "held_out_scene": False,
            "by_intent": held_by_intent,
            "records": held_scored,
        },
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
            "mean_zero_minus_correct_nll": causal_margin,
            "required_mean_margin_nll": target_margin,
            "canonical_prediction_changes": causal_changes,
            "records": list(causal_records),
        },
        "model_acceptance_gates": model_gates,
        "model_acceptance_gate_passed": passed,
        "separate_runtime_packaging_authorized": passed,
        "runtime_oracle_unavailable_gate_pending": passed,
        "runtime_file_audit_gate_pending": passed,
        "automatic_runtime_promotion": False,
        "runtime_promotion_authorized": False,
    }


def _evaluate_rows(
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
            "correct_answer_token_top1_accuracy": measured[
                "answer_token_top1_accuracy"
            ],
            "scene_memory_sha256": memory_hash,
            "layout_audit": layout,
        }
        if phase != "canonical":
            record["question"] = row.question
            record["intent_id"] = _intent_id(row)
        records.append(record)
        if ordinal == 1 or ordinal % 12 == 0 or ordinal == len(rows):
            print(
                json.dumps(
                    {
                        "event": "v91_evaluation_row",
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


def run_evaluation_v91(config_path: str | Path = CONFIG) -> dict[str, Any]:
    """Run V91's create-once fixed-final evaluation on local MPS."""

    # Trainer imports are deferred so this evaluator remains importable while
    # independently sealed V91 implementation files are being assembled.
    from semantic_3d_chat.evaluation.v91_scene1_conversational_preflight import (
        authenticate_sources_v91,
        load_canonical_rows_v91,
    )
    from semantic_3d_chat.training.train_v91_scene1_conversational_repair import (
        authenticate_training_report_v91,
        combined_lora_settings_v91,
        load_fixed_final_bridge_v91,
        load_frozen_parent_v91,
    )

    started = time.monotonic()
    config = load_config_v91(config_path, allow_draft=False)
    source_hashes = authenticate_sources_v91(config)
    training_bindings = authenticate_training_report_v91(
        config, config_path=config_path
    )
    canonical_rows = load_canonical_rows_v91(config)
    primary_rows = primary_rows_v91(config)
    held_rows = held_wording_rows_v91(config)
    predictions_path = resolve_v85(config["outputs"]["evaluation_predictions"])
    report_path = resolve_v85(config["outputs"]["evaluation_report"])
    if predictions_path.exists() or report_path.exists():
        raise FileExistsError("V91 create-once evaluation output exists")

    cpu_memory, memory_hash_before, _metadata = load_scene1_memory_v86(config)
    if tuple(cpu_memory.shape) != EXPECTED_PREFIX_SHAPE:
        raise ValueError("V91 immutable scene-memory shape changed")
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
            raise RuntimeError("V91 fixed-final evaluation requires local MPS")
        collection = install_lora_banks(
            language.model, combined_lora_settings_v91(runtime, config)
        )
        if not isinstance(collection, LoRABankCollection):
            raise TypeError("V91 evaluation LoRA installation failed")
        frozen_source = load_frozen_parent_v91(
            collection,
            config["sources"]["parent_v89_checkpoint"],
            config,
        )
        candidate = load_fixed_final_bridge_v91(
            collection, config["outputs"]["fixed_final_candidate"]
        )
        if (
            candidate.get("weights_sha256")
            != training_bindings["candidate_weights_sha256"]
            or candidate.get("state_sha256")
            != training_bindings["candidate_state_sha256"]
            or canonical_sha256_v85(candidate)
            != training_bindings["candidate_metadata_canonical_sha256"]
        ):
            raise ValueError(
                "V91 loaded candidate differs from the authenticated training result"
            )
        collection.eval()
        language.decoder_module.eval()
        frozen_states_before = {
            bank.settings.name: bank.installation.state_sha256()
            for bank in collection.banks
            if not bank.settings.trainable
        }
        candidate_state_before = collection.bank(
            str(config["bridge"]["bank_name"])
        ).installation.state_sha256()
        memory = cpu_memory.to(device=language.device, dtype=torch.bfloat16)
        zero_memory = cpu_zero_memory.to(
            device=language.device, dtype=torch.bfloat16
        )
        system_prompt = str(language_config["system_prompt"])
        max_new_tokens = int(language_config["max_answer_tokens"])
        canonical_records = _evaluate_rows(
            language=language,
            system_prompt=system_prompt,
            memory=memory,
            memory_hash=memory_hash_before,
            rows=canonical_rows,
            max_new_tokens=max_new_tokens,
            phase="canonical",
            started=started,
        )
        primary_records = _evaluate_rows(
            language=language,
            system_prompt=system_prompt,
            memory=memory,
            memory_hash=memory_hash_before,
            rows=primary_rows,
            max_new_tokens=max_new_tokens,
            phase="primary_conversational",
            started=started,
        )
        held_records = _evaluate_rows(
            language=language,
            system_prompt=system_prompt,
            memory=memory,
            memory_hash=memory_hash_before,
            rows=held_rows,
            max_new_tokens=max_new_tokens,
            phase="new_held_wording",
            started=started,
        )
        primary_by_id = {
            str(record["question_id"]): record for record in primary_records
        }
        causal_records: list[dict[str, Any]] = []
        for row in primary_rows:
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
            correct = primary_by_id[row.question_id]
            causal_records.append(
                {
                    "scene_id": SCENE_ID,
                    "question_id": row.question_id,
                    "intent_id": _intent_id(row),
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
        candidate_state_after = collection.bank(
            str(config["bridge"]["bank_name"])
        ).installation.state_sha256()
    audit.assert_clean()

    all_records = canonical_records + primary_records + held_records
    if len(all_records) != EXPECTED_EVALUATED_QUESTION_COUNT:
        raise RuntimeError("V91 evaluated question inventory changed")
    prefix_hash_invariant = bool(
        memory_hash_after == memory_hash_before
        and all(
            record["scene_memory_sha256"] == memory_hash_before
            for record in all_records
        )
    )
    environment_input_invariant = bool(
        prefix_hash_invariant
        and zero_hash_after == zero_hash_before
        and all(_layout_is_strict(record) for record in all_records)
        and all(
            record["correct_memory_sha256"] == memory_hash_before
            and record["zero_payload_memory_sha256"] == zero_hash_before
            and _layout_is_strict(
                {"layout_audit": record["zero_layout_audit"]}
            )
            for record in causal_records
        )
    )
    parent_state_invariant = frozen_states_after == frozen_states_before
    candidate_state_invariant = candidate_state_after == candidate_state_before
    if not candidate_state_invariant:
        raise RuntimeError("V91 fixed-final candidate changed during evaluation")
    protected_count = len(audit.forbidden_accesses())
    score = score_records_v91(
        canonical_rows,
        canonical_records,
        primary_records,
        held_records,
        causal_records,
        gates=config["gates"],
        prefix_hash_invariant=prefix_hash_invariant,
        environment_input_invariant=environment_input_invariant,
        parent_state_invariant=parent_state_invariant,
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
        "schema_version": 91,
        "status": "fixed_final_evaluation_only_not_runtime",
        "config_sha256": training_bindings["config_sha256"],
        "training_report_sha256": training_bindings["training_report_sha256"],
        "scene_id": SCENE_ID,
        "scene_count": 1,
        "canonical_row_count": len(canonical_records),
        "primary_conversational_row_count": len(primary_records),
        "new_held_wording_row_count": len(held_records),
        "causal_control_row_count": len(causal_records),
        "fixed_checkpoint_selected_before_scoring": True,
        "checkpoint_selection_after_scoring": False,
        "newly_held_wording_only": True,
        "held_out_scene": False,
        "scene_memory": scene_memory,
        "candidate": {
            "path": config["outputs"]["fixed_final_candidate"],
            "weights_sha256": candidate["weights_sha256"],
            "state_sha256": candidate["state_sha256"],
            "optimizer_updates": int(config["training"]["optimizer_updates"]),
        },
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
        "new_held_wording_records": held_records,
        "causal_records": causal_records,
    }
    prediction_output, prediction_sha = atomic_create_json_v85(
        predictions_path, predictions
    )
    report = {
        "artifact": EVALUATION_ARTIFACT,
        "schema_version": 91,
        "status": (
            "model_gates_pass_separate_runtime_packaging_required"
            if score["model_acceptance_gate_passed"]
            else "model_gates_fail_not_runtime_promotable"
        ),
        "config_sha256": training_bindings["config_sha256"],
        "preregistration_sha256": training_bindings["preregistration_sha256"],
        "cpu_preflight_sha256": training_bindings["cpu_preflight_sha256"],
        "training_report_sha256": training_bindings["training_report_sha256"],
        "evaluation_predictions_path": prediction_output.relative_to(
            PROJECT_ROOT
        ).as_posix(),
        "evaluation_predictions_sha256": prediction_sha,
        "source_hashes": source_hashes,
        "frozen_source": frozen_source,
        "fixed_checkpoint_selected_before_scoring": True,
        "checkpoint_selection_after_scoring": False,
        "preregistered_gates": config["gates"],
        "metrics": score,
        "scene_memory": scene_memory,
        "leakage": leakage,
        "post_v90_training_set_development": True,
        "single_scene_conversational_repair": True,
        "development_known_primary_questions": True,
        "newly_held_wording_only": True,
        "held_out_scene": False,
        "held_out_scene_generalization_claim": False,
        "parent_v89_runtime_checkpoint_mutated": False,
        "parent_v90_failed_candidate_mutated": False,
        "fixed_final_candidate_state_invariant": candidate_state_invariant,
        "separate_runtime_packaging_authorized": score[
            "model_acceptance_gate_passed"
        ],
        "runtime_oracle_unavailable_gate_pending": score[
            "model_acceptance_gate_passed"
        ],
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
    report = run_evaluation_v91(args.config)
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
    "conversational_answer_key_v91",
    "conversational_match_v91",
    "held_wording_rows_v91",
    "main",
    "primary_rows_v91",
    "run_evaluation_v91",
    "score_records_v91",
]
