from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from semantic_3d_chat.evaluation.v62_pair_disjoint_preregistration import TRAIN_PAIR_IDS
from semantic_3d_chat.training import train_question_control_v66 as v66
from semantic_3d_chat.training.question_control_v66_prototypes import answer_class_id_v66
from semantic_3d_chat.training.train_question_control_v63 import V63Row


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _record(
    row: V63Row,
    *,
    exact: bool,
    prediction_id: str,
) -> dict[str, object]:
    reference = v66._row_reference_sha256(row)
    canonical = reference if exact else _digest(f"wrong:{prediction_id}")
    return {
        "scene_id": row.scene_id,
        "question_id": row.question_id,
        "pair_id": row.pair_id,
        "question_key": row.question_key,
        "answer_class_id": answer_class_id_v66(row.answer),
        "answer_type_sha256": _digest(row.answer_type),
        "list_scoring": row.answer_type in v66.LIST_ANSWER_TYPES,
        "counterfactual_changed_side": row.route_label,
        "fold_class_supported": True,
        "raw_prediction_sha256": _digest(f"raw:{prediction_id}"),
        "canonical_prediction_sha256": canonical,
        "reference_canonical_sha256": reference,
        "scoring_contract_sha256": v66._row_scoring_contract_sha256(row),
        "canonical_exact": exact,
    }


def _row(
    ordinal: int,
    *,
    pair_id: str,
    answer: str,
    answer_type: str,
    changed: bool = False,
    question_key: str | None = None,
    question: str | None = None,
) -> V63Row:
    return V63Row(
        scene_id=f"scene_{ordinal:06d}",
        question_id=f"q_{ordinal:06d}",
        question=question or f"Question {ordinal}?",
        pair_id=pair_id,
        question_key=question_key or f"cfq_{ordinal:06d}",
        route_label=changed,
        answer=answer,
        answer_type=answer_type,
    )


def test_v66_parser_exposes_training_only_boundary_and_no_protected_inputs() -> None:
    destinations = {action.dest for action in v66._parser()._actions}

    assert {
        "baseline_lock",
        "preregistration",
        "training_baseline_lock",
        "filtered_train_qa",
        "teacher_cache",
        "supplemental_teacher_cache",
        "prefix_cache",
        "base_runtime_config",
        "base_checkpoint",
        "source_v60_checkpoint",
        "work_directory",
        "output_checkpoint",
        "training_report",
    } <= destinations
    assert {
        "validation_questions",
        "scorer_references",
        "oracle",
        "fresh_development",
        "final_scenes",
        "internal_validation",
    }.isdisjoint(destinations)


def test_v66_thresholds_are_preregistered_above_v54_training_baseline() -> None:
    threshold = v66.V66B_BEHAVIOR_THRESHOLDS

    assert threshold.held_supported_exact_minimum == 300
    assert threshold.held_supported_exact_minimum > 227
    assert threshold.held_supported_total == 571
    assert threshold.held_unsupported_total == 5
    assert threshold.held_changed_side_exact_minimum == 45
    assert threshold.held_changed_side_exact_minimum > 35
    assert threshold.held_complete_unit_minimum == 15
    assert threshold.held_complete_unit_minimum > 5
    assert threshold.held_prediction_change_unit_minimum == 20
    assert threshold.held_prediction_change_unit_minimum > 9
    assert threshold.final_exact_minimum == 520
    assert threshold.final_complete_unit_minimum == 36
    assert threshold.paired_opposite_follows_side_minimum == 60
    assert threshold.paired_opposite_follows_complete_minimum == 25
    assert threshold.paired_opposite_original_exact_maximum == 20
    assert threshold.paired_opposite_original_complete_maximum == 5
    assert dict(threshold.per_type_minimum_exact)["orientation"] == 20


def test_v66_trainer_authenticates_pre_run_preregistration() -> None:
    payload = v66.validate_v66_preregistration(
        "reports/gemma4/metrics/v66b_paired_opposite_preregistration.json"
    )

    assert payload["thresholds"]["held_supported_exact_minimum"] == 300
    assert payload["invalidated_predecessor"]["sha256"].startswith("974f7049")
    assert payload["controls"][
        "exact_paired_opposite_scene_prefix_and_signature"
    ] is True
    assert payload["controls"][
        "unverified_native_answer_embedding_fallback_permitted"
    ] is False


def test_v66_behavior_metrics_recompute_changed_and_type_controls() -> None:
    rows = [
        _row(1, pair_id=TRAIN_PAIR_IDS[0], answer="left", answer_type="spatial_relation", changed=True, question_key="pair_question"),
        _row(2, pair_id=TRAIN_PAIR_IDS[0], answer="right", answer_type="spatial_relation", changed=True, question_key="pair_question"),
        _row(3, pair_id=TRAIN_PAIR_IDS[0], answer="yes", answer_type="presence"),
    ]
    records = [
        _record(rows[0], exact=True, prediction_id="left"),
        _record(rows[1], exact=True, prediction_id="right"),
        _record(rows[2], exact=False, prediction_id="no"),
    ]

    metrics = v66.behavior_metrics_v66(records, unsupported_count=1)

    assert metrics["supported_exact"] == 2
    assert metrics["supported_total"] == 3
    assert metrics["inventory_total"] == 4
    assert metrics["changed_side_exact"] == 2
    assert metrics["complete_changed_units"] == 1
    assert metrics["prediction_change_units"] == 1
    assert metrics["changed_unit_total"] == 1
    assert metrics["per_type_by_sha256"][_digest("spatial_relation")] == {
        "exact": 2,
        "total": 2,
    }


def test_v66_primary_cv_assessment_enforces_every_preregistered_dimension() -> None:
    threshold = v66.V66B_BEHAVIOR_THRESHOLDS
    metrics = {
        "supported_exact": threshold.held_supported_exact_minimum,
        "supported_total": 571,
        "unsupported_total": 5,
        "inventory_total": 576,
        "eligible_fold_total": 12,
        "eligible_folds_with_exact_hit": 12,
        "changed_side_exact": threshold.held_changed_side_exact_minimum,
        "changed_side_total": 75,
        "complete_changed_units": threshold.held_complete_unit_minimum,
        "changed_unit_total": 35,
        "prediction_change_units": threshold.held_prediction_change_unit_minimum,
        "per_type_by_sha256": {
            _digest(answer_type): {"exact": minimum, "total": minimum}
            for answer_type, minimum in threshold.per_type_minimum_exact
        },
    }

    assert all(v66.assess_cv_v66(metrics).values())
    metrics["prediction_change_units"] -= 1
    assert v66.assess_cv_v66(metrics)["held_prediction_change_units"] is False


def test_v66_cached_fold_is_recomputed_and_detects_tampering(tmp_path: Path) -> None:
    held_pair = TRAIN_PAIR_IDS[0]
    rows: list[V63Row] = []
    for index in range(48):
        rows.append(
            _row(
                index,
                pair_id=held_pair,
                answer="yes",
                answer_type="presence",
            )
        )
    rows.append(
        _row(100, pair_id=TRAIN_PAIR_IDS[1], answer="yes", answer_type="presence")
    )
    records = [_record(row, exact=True, prediction_id=row.question_id) for row in rows[:48]]
    behavior = v66.behavior_metrics_v66(records, unsupported_count=0)
    payload = {
        "schema_version": 1,
        "artifact": v66._FOLD_ARTIFACT,
        "run_signature_sha256": "a" * 64,
        "held_pair_id": held_pair,
        "held_rows_used_for_optimization": False,
        "held_teacher_sources_used": False,
        "fold_codebook_sha256": "b" * 64,
        "fold_basis_sha256": "c" * 64,
        "fold_class_count": 1,
        "fold_train_target_count": 1,
        "fit": {
            "optimizer_steps": 1,
            "classification_optimizer_steps": 1,
            "elapsed_seconds": 0.1,
            "question_norm_sha256": "d" * 64,
            "question_norm_frozen": True,
            "numeric_prototype_top1_accuracy": 1.0,
            "numeric_prototype_mean_margin": 0.5,
        },
        "behavior": behavior,
        "records": records,
    }

    validated = v66._validate_cached_fold_v66(
        payload,
        held_pair_id=held_pair,
        run_signature_sha256="a" * 64,
        rows=rows,
    )
    assert validated["behavior"] == behavior
    tampered = json.loads(json.dumps(payload))
    tampered["behavior"]["supported_exact"] -= 1
    with pytest.raises(ValueError, match="metrics differ"):
        v66._validate_cached_fold_v66(
            tampered,
            held_pair_id=held_pair,
            run_signature_sha256="a" * 64,
            rows=rows,
        )


def test_v66_combined_teacher_loader_requires_all_classes_and_no_overlap() -> None:
    primary_rows = tuple(
        _row(
            index,
            pair_id=TRAIN_PAIR_IDS[index % 2],
            answer=f"class {index}",
            answer_type="attribute",
            changed=index < 21,
        )
        for index in range(28)
    )
    verification_rows = tuple(
        _row(
            100 + index,
            pair_id=primary_rows[index].pair_id,
            answer=primary_rows[index].answer,
            answer_type="attribute",
        )
        for index in range(21, 28)
    )
    rows = primary_rows + verification_rows
    original = {row.key: torch.ones(1, 4, 1536) for row in primary_rows[:21]}
    preflight = type("Preflight", (), {
        "rows": rows,
        "teacher_targets": original,
        "teacher_metadata_sha256": "1" * 64,
        "teacher_weights_sha256": "2" * 64,
    })()
    supplemental_sources = primary_rows[21:]
    supplemental = {
        row.key: torch.full((1, 4, 1536), 0.1) for row in supplemental_sources
    }
    metadata = {
        "records": [
            {
                "source_scene_id": source.scene_id,
                "source_question_id": source.question_id,
                "source_pair_id": source.pair_id,
                "answer_class_id": answer_class_id_v66(source.answer),
                "verification_keys": [
                    {
                        "scene_id": source.scene_id,
                        "question_id": source.question_id,
                    },
                    {
                        "scene_id": verification.scene_id,
                        "question_id": verification.question_id,
                    },
                ],
            }
            for source, verification in zip(
                supplemental_sources,
                verification_rows,
                strict=True,
            )
        ]
    }

    combined, audit = v66.load_combined_verified_teachers_v66(
        preflight,
        "unused",
        supplemental_loader=lambda _path: (supplemental, metadata),
    )

    assert len(combined) == 28
    assert audit["answer_class_count"] == 28
    assert audit["every_answer_class_has_verified_teacher"] is True


def test_v66b_paired_opposite_gate_enforces_preregistered_dimensions() -> None:
    threshold = v66.V66B_BEHAVIOR_THRESHOLDS
    metrics = {
        "answer_follows_injected_scene": 60,
        "paired_opposite_side_total": 80,
        "answer_follows_injected_scene_complete_units": 25,
        "paired_opposite_unit_total": 40,
        "answer_matches_original_reference": 20,
        "answer_matches_original_reference_complete_units": 5,
        "question_identity_count": 80,
        "exact_paired_scene_count": 80,
        "exact_paired_scene_prefix_count": 80,
        "exact_paired_scene_signature_count": 80,
        "differing_reference_count": 80,
        "cross_swap_complete_units": 40,
        "answer_or_question_text_stored": False,
    }

    assert all(v66.paired_opposite_checks_v66(metrics).values())
    metrics["answer_follows_injected_scene"] = (
        threshold.paired_opposite_follows_side_minimum - 1
    )
    assert v66.paired_opposite_checks_v66(metrics)[
        "follows_injected_side_minimum"
    ] is False


def test_v66b_generates_all_exact_paired_opposite_scene_injections() -> None:
    rows: list[V63Row] = []
    prefixes: dict[str, torch.Tensor] = {}
    answers_by_prefix_value: dict[int, str] = {}
    questions: dict[tuple[str, str], torch.Tensor] = {}
    for unit_index in range(40):
        pair_id = TRAIN_PAIR_IDS[unit_index % len(TRAIN_PAIR_IDS)]
        question_key = f"changed_unit_{unit_index:03d}"
        question = f"Identical paired question {unit_index}?"
        left = _row(
            2 * unit_index,
            pair_id=pair_id,
            answer="left",
            answer_type="spatial_relation",
            changed=True,
            question_key=question_key,
            question=question,
        )
        right = _row(
            2 * unit_index + 1,
            pair_id=pair_id,
            answer="right",
            answer_type="spatial_relation",
            changed=True,
            question_key=question_key,
            question=question,
        )
        rows.extend((left, right))
        for row in (left, right):
            prefix_value = int(row.scene_id.removeprefix("scene_")) + 1
            prefixes[row.scene_id] = torch.full(
                (1, 256, 4), float(prefix_value), dtype=torch.float32
            )
            answers_by_prefix_value[prefix_value] = row.answer
            questions[row.key] = torch.ones(1, 3, 4)

    class FakeControl:
        def __init__(self) -> None:
            self._used = False

        def encode_scene(self, prefix: torch.Tensor) -> torch.Tensor:
            return prefix.mean(dim=1)

        def forward_from_signature(
            self, signature: torch.Tensor, question: torch.Tensor
        ) -> SimpleNamespace:
            assert question.shape == (1, 3, 4)
            self._used = True
            return SimpleNamespace(control_tokens=signature.unsqueeze(1))

        def audit(self) -> SimpleNamespace:
            return SimpleNamespace(control_used=self._used)

    seen_prefix_values: list[int] = []

    def generator_fn(**kwargs: object) -> str:
        prefix = kwargs["scene_prefix"]
        assert isinstance(prefix, torch.Tensor)
        value = int(prefix[0, 0, 0].item())
        seen_prefix_values.append(value)
        return answers_by_prefix_value[value]

    control = FakeControl()
    signatures = {
        scene_id: control.encode_scene(prefix) for scene_id, prefix in prefixes.items()
    }
    records = v66.generate_paired_opposite_scene_rows_v66(
        SimpleNamespace(control=control, signatures=signatures),
        rows,
        questions=questions,
        bundle=SimpleNamespace(
            runtime=object(), device=torch.device("cpu"), model_dtype=torch.float32
        ),
        prefixes=prefixes,
        generator_fn=generator_fn,
    )
    metrics = v66.paired_opposite_metrics_v66(records)

    assert len(records) == 80
    assert len(seen_prefix_values) == 80
    assert metrics["answer_follows_injected_scene"] == 80
    assert metrics["answer_follows_injected_scene_complete_units"] == 40
    assert metrics["answer_matches_original_reference"] == 0
    assert metrics["question_identity_count"] == 80
    assert metrics["exact_paired_scene_prefix_count"] == 80
    assert metrics["exact_paired_scene_signature_count"] == 80
    assert all(v66.paired_opposite_checks_v66(metrics).values())
    tampered = [dict(record) for record in records]
    tampered[0]["question_sha256"] = _digest("not-the-paired-question")
    with pytest.raises(ValueError, match="boolean differs"):
        v66.paired_opposite_metrics_v66(tampered)
