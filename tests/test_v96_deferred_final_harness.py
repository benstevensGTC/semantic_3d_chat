from __future__ import annotations

import copy
import hashlib
import inspect
from pathlib import Path
from typing import Any

import pytest

from semantic_3d_chat.evaluation import nll_v96_deferred_final as nll_module
from semantic_3d_chat.evaluation import predict_v96_deferred_final as predictor
from semantic_3d_chat.evaluation import score_v96_deferred_final as scorer
from semantic_3d_chat.evaluation import v96_deferred_final_common as common
from semantic_3d_chat.evaluation import (
    v96_deferred_final_evaluation as evaluation,
)
from semantic_3d_chat.evaluation.question_manifest import (
    QuestionManifest,
    QuestionRecord,
    questions_sha256,
)
from semantic_3d_chat.evaluation.score_v96_deferred_final import (
    structured_metrics_v96_final,
)
from semantic_3d_chat.evaluation.v95_deferred_final_qa import (
    PAIR_SCENES,
    PAIR_UNIT_QUOTAS,
)
from semantic_3d_chat.evaluation.v96_deferred_final_evaluation import (
    ANSWER_TYPE_TOTALS,
    CHANGED_SIDE_COUNT,
    CHANGED_UNIT_COUNT,
    FINAL_GATE_CONTRACT,
    PAIR_SCENE,
    QUESTION_COUNT,
    SCENE_IDS,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _answer_types() -> tuple[str, ...]:
    return tuple(
        answer_type for answer_type, count in PAIR_UNIT_QUOTAS.items() for _ in range(count)
    )


def _answer(answer_type: str, *, alternate: bool = False) -> str:
    values = {
        "attribute": ("red", "blue"),
        "count": ("one", "two"),
        "metric": ("2 meters", "3 meters"),
        "orientation": ("upright", "overturned"),
        "presence": ("yes", "no"),
        "spatial_relation": ("left", "right"),
        "support": ("table", "floor"),
    }
    return values[answer_type][int(alternate)]


def _references_and_rows() -> tuple[
    list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]
]:
    references: list[dict[str, Any]] = []
    v96_rows: list[dict[str, Any]] = []
    v94_rows: list[dict[str, Any]] = []
    types = _answer_types()
    global_scene_offset = {scene: index for index, scene in enumerate(SCENE_IDS)}
    for pair_id, scenes in PAIR_SCENES.items():
        for unit_index, answer_type in enumerate(types):
            changed = unit_index < 4
            question_key = f"unit_{unit_index:02d}"
            for side, scene_id in enumerate(scenes):
                question_id = f"q_{global_scene_offset[scene_id] * len(types) + unit_index:06d}"
                answer = _answer(
                    answer_type,
                    alternate=changed and side == 1,
                )
                reference = {
                    "scene_id": scene_id,
                    "question_id": question_id,
                    "question": f"Synthetic paired question {question_key}?",
                    "answer_type": answer_type,
                    "answer": answer,
                    "counterfactual_pair_id": pair_id,
                    "counterfactual_paired_scene_id": PAIR_SCENE[scene_id],
                    "counterfactual_question_key": question_key,
                    "counterfactual_change_type": "synthetic_change",
                    "counterfactual_expected_change": changed,
                }
                references.append(reference)
                v96_rows.append(
                    {
                        "scene_id": scene_id,
                        "question_id": question_id,
                        f"{common.PRIMARY}_prediction": answer,
                        f"{common.ZERO_PAYLOAD}_prediction": "unknown",
                        f"{common.FULL_INTERIOR_PERMUTATION}_prediction": "unknown",
                        f"{common.PAIRED_WRONG_SCENE}_prediction": "unknown",
                    }
                )
                v94_rows.append(
                    {
                        "scene_id": scene_id,
                        "question_id": question_id,
                        "prediction": answer,
                    }
                )
    return references, v96_rows, v94_rows


def _questions() -> QuestionManifest:
    references, _v96, _v94 = _references_and_rows()
    records = tuple(
        QuestionRecord(
            scene_id=str(row["scene_id"]),
            question_id=str(row["question_id"]),
            question=str(row["question"]),
        )
        for row in references
    )
    return QuestionManifest(
        questions=records,
        questions_sha256=questions_sha256(records),
        source_qa_sha256=_digest("references"),
        manifest_path=Path("/synthetic/v96-final-questions.json"),
        manifest_sha256=_digest("manifest"),
    )


def _fixed() -> common.FixedFinalInputs:
    hashes = {arm: {scene: _digest(f"{arm}:{scene}") for scene in SCENE_IDS} for arm in common.ARMS}
    return common.FixedFinalInputs(
        candidate={"fingerprint_sha256": _digest("candidate")},
        memories={},
        memory_hashes=hashes,
        memory_inventory_sha256=_digest("memory-inventory"),
        memory_paths={},
        materialized={},
    )


def _prediction_rows(model: str) -> list[dict[str, Any]]:
    fixed = _fixed()
    provenance = _digest(f"{model}-provenance")
    rows: list[dict[str, Any]] = []
    for question in _questions().questions:
        if model == "v96":
            rows.append(
                {
                    "artifact": common.V96_PREDICTION_ARTIFACT,
                    "schema_version": common.SCHEMA_VERSION,
                    "scene_id": question.scene_id,
                    "question_id": question.question_id,
                    "paired_scene_id": PAIR_SCENE[question.scene_id],
                    **{f"{arm}_prediction": f"synthetic {arm} answer" for arm in common.ARMS},
                    **{
                        f"{arm}_memory_sha256": fixed.memory_hashes[arm][question.scene_id]
                        for arm in common.ARMS
                    },
                    "all_memory_hashes_unchanged": True,
                    "provenance_sha256": provenance,
                }
            )
        else:
            rows.append(
                {
                    "artifact": common.V94_PREDICTION_ARTIFACT,
                    "schema_version": common.SCHEMA_VERSION,
                    "scene_id": question.scene_id,
                    "question_id": question.question_id,
                    "memory_sha256": fixed.memory_hashes[common.PRIMARY][question.scene_id],
                    "prediction": "synthetic v94 answer",
                    "provenance_sha256": provenance,
                }
            )
    return rows


@pytest.mark.parametrize("model", ["v96", "v94"])
def test_deferred_prediction_rows_are_exact_label_free_schemas(model: str) -> None:
    rows = _prediction_rows(model)
    common.validate_prediction_rows_v96_final(
        rows,
        model=model,
        fixed=_fixed(),
        questions=_questions(),
        provenance_sha256=_digest(f"{model}-provenance"),
    )

    expected_fields = common.V96_ROW_FIELDS if model == "v96" else common.V94_ROW_FIELDS
    assert len(rows) == QUESTION_COUNT
    assert all(set(row) == expected_fields for row in rows)
    prohibited = {
        "answer",
        "reference",
        "target_xyz",
        "target_instance",
        "counterfactual_expected_change",
    }
    assert all(not prohibited.intersection(row) for row in rows)


@pytest.mark.parametrize("model", ["v96", "v94"])
def test_deferred_prediction_schema_fails_closed_on_missing_or_extra_row(
    model: str,
) -> None:
    rows = _prediction_rows(model)
    with pytest.raises(ValueError):
        common.validate_prediction_rows_v96_final(
            rows[:-1],
            model=model,
            fixed=_fixed(),
            questions=_questions(),
            provenance_sha256=_digest(f"{model}-provenance"),
        )

    tampered = copy.deepcopy(rows)
    tampered[0]["answer"] = "leaked label"
    with pytest.raises(ValueError):
        common.validate_prediction_rows_v96_final(
            tampered,
            model=model,
            fixed=_fixed(),
            questions=_questions(),
            provenance_sha256=_digest(f"{model}-provenance"),
        )


def test_deferred_structured_metrics_cover_exact_same_216_rows() -> None:
    references, v96_rows, v94_rows = _references_and_rows()

    metrics = structured_metrics_v96_final(references, v96_rows, v94_rows)

    assert len(references) == QUESTION_COUNT
    assert metrics["arms"][common.PRIMARY]["correct"] == QUESTION_COUNT
    assert {
        key: value["total"]
        for key, value in metrics["arms"][common.PRIMARY]["by_answer_type"].items()
    } == ANSWER_TYPE_TOTALS
    assert metrics["v94_same_rows"]["correct"] == QUESTION_COUNT
    assert metrics["v96_accuracy_margin_over_v94_same_rows"] == 0.0
    assert metrics["counterfactual"]["unit_count"] == CHANGED_UNIT_COUNT
    assert metrics["counterfactual"]["side_count"] == CHANGED_SIDE_COUNT
    assert metrics["counterfactual"]["canonical_correct_sides"] == CHANGED_SIDE_COUNT
    assert metrics["counterfactual"]["canonical_complete_units"] == CHANGED_UNIT_COUNT
    assert metrics["counterfactual"]["canonical_prediction_changed_units"] == CHANGED_UNIT_COUNT
    assert (
        sum(
            int(value["correct_sides"]) for value in metrics["counterfactual"]["by_family"].values()
        )
        == CHANGED_SIDE_COUNT
    )
    assert set(metrics["counterfactual"]["by_family"]) == set(PAIR_SCENES)
    assert all(
        value
        == {
            "unit_count": 4,
            "correct_sides": 8,
            "complete_units": 4,
            "prediction_changed_units": 4,
        }
        for value in metrics["counterfactual"]["by_family"].values()
    )
    assert metrics["stable_invariant"]["invariant_false_change_count"] == 0
    common.assert_aggregate_only_v96_final(metrics)


def test_deferred_structured_metrics_reject_nonidentical_comparator_rows() -> None:
    references, v96_rows, v94_rows = _references_and_rows()
    with pytest.raises(ValueError, match="exact same 216 keys"):
        structured_metrics_v96_final(references, v96_rows, v94_rows[:-1])


def test_deferred_stable_units_must_have_identical_reference_semantics() -> None:
    references, v96_rows, v94_rows = _references_and_rows()
    stable = next(row for row in references if row["counterfactual_expected_change"] is False)
    stable["answer"] = _answer(str(stable["answer_type"]), alternate=True)
    with pytest.raises(ValueError, match="invariant unit changed"):
        structured_metrics_v96_final(references, v96_rows, v94_rows)


def test_deferred_aggregate_guard_accepts_only_aggregate_content() -> None:
    common.assert_aggregate_only_v96_final(
        {
            "metrics": {
                "accuracy": 0.65,
                "by_answer_type": {"attribute": {"correct": 24, "total": 48}},
            },
            "v96_prediction_sha256": _digest("predictions"),
            "row_level_content_serialized": False,
        }
    )
    for key in (
        "question",
        "answer",
        "prediction",
        "predictions",
        "reference",
        "references",
        "rows",
        "scene_id",
        "question_id",
        "target_xyz",
        "target_instance",
    ):
        with pytest.raises(ValueError, match="aggregate"):
            common.assert_aggregate_only_v96_final({"metrics": {key: "leak"}})


def test_deferred_final_gate_contract_is_the_exact_sealed_v95_contract() -> None:
    assert common.canonical_sha256_v96(FINAL_GATE_CONTRACT) == (
        "fec89e831c55be7a7c057b40940f47eb279024a5c2365c96946b6eada747c068"
    )
    assert FINAL_GATE_CONTRACT["canonical_accuracy_minimum"] == 0.65
    assert FINAL_GATE_CONTRACT["canonical_accuracy_margin_over_fixed_v94_same_rows"] == 0.03
    assert FINAL_GATE_CONTRACT["changed_side_total"] == CHANGED_SIDE_COUNT
    assert FINAL_GATE_CONTRACT["changed_unit_total"] == CHANGED_UNIT_COUNT
    assert FINAL_GATE_CONTRACT["automatic_runtime_promotion"] is False


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), "1.0"])
def test_deferred_finite_number_rejects_nonfinite_or_nonnumeric(value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        common.finite_number(value, "synthetic")


def test_predictor_authenticates_question_receipt_without_opening_question_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, bool]] = []
    monkeypatch.setattr(
        evaluation,
        "authenticate_preregistration_v96_final",
        lambda: {"authenticated": True},
    )
    monkeypatch.setattr(
        evaluation,
        "authenticate_unlock_blind_v96_final",
        lambda _preregistration: {"authenticated": True},
    )

    def stage(
        _preregistration: dict[str, Any],
        _unlock: dict[str, Any],
        name: str,
        *,
        hash_outputs: bool,
    ) -> dict[str, Any]:
        calls.append((name, hash_outputs))
        return {"authenticated": True}

    monkeypatch.setattr(evaluation, "authenticate_stage_receipt_v96_final", stage)

    result = evaluation.authenticate_materialized_inputs_v96_final(label_process=False)

    assert result["authenticated"] is True
    assert calls == [("memory", True), ("questions", False)]


def test_source_order_proves_memories_and_bundles_precede_question_or_label_reads() -> None:
    prediction_source = inspect.getsource(predictor.predict_deferred_final_v96)
    assert prediction_source.index(
        "authenticate_fixed_inputs_before_questions_v96_final"
    ) < prediction_source.index("load_questions_v96_final")
    assert "load_references_v96_final" not in prediction_source
    assert "FINAL_QA" not in prediction_source

    score_source = inspect.getsource(scorer.score_deferred_final_v96)
    assert score_source.index(
        'authenticate_prediction_bundle_v96_final("v96")'
    ) < score_source.index("load_references_v96_final")
    assert score_source.index(
        'authenticate_prediction_bundle_v96_final("v94")'
    ) < score_source.index("load_references_v96_final")

    nll_source = inspect.getsource(nll_module.measure_nll_v96_final)
    assert nll_source.index('authenticate_prediction_bundle_v96_final("v96")') < nll_source.index(
        "load_references_v96_final"
    )
    assert nll_source.index('authenticate_prediction_bundle_v96_final("v94")') < nll_source.index(
        "load_references_v96_final"
    )


def _boundary_structured_metrics() -> dict[str, Any]:
    by_answer_type = {
        answer_type: {
            "correct": FINAL_GATE_CONTRACT[f"{answer_type}_correct_minimum"],
            "total": FINAL_GATE_CONTRACT[f"{answer_type}_total"],
            "accuracy": (
                FINAL_GATE_CONTRACT[f"{answer_type}_correct_minimum"]
                / FINAL_GATE_CONTRACT[f"{answer_type}_total"]
            ),
        }
        for answer_type in ANSWER_TYPE_TOTALS
    }
    primary_correct = sum(int(value["correct"]) for value in by_answer_type.values())
    primary_accuracy = primary_correct / QUESTION_COUNT
    # Both accuracy thresholds are continuous, while the fixed 216-row sample
    # admits only integer-count steps.  Seven rows is the first attainable
    # V96-over-V94 margin at or above 0.03; six rows is below it.
    v94_correct = primary_correct - 7
    v94_accuracy = v94_correct / QUESTION_COUNT
    return {
        "arms": {
            common.PRIMARY: {
                "correct": primary_correct,
                "total": QUESTION_COUNT,
                "accuracy": primary_accuracy,
                "by_answer_type": by_answer_type,
            },
            common.ZERO_PAYLOAD: {
                "correct": 0,
                "total": QUESTION_COUNT,
                "accuracy": 0.0,
                "by_answer_type": {},
            },
            common.FULL_INTERIOR_PERMUTATION: {
                "correct": 0,
                "total": QUESTION_COUNT,
                "accuracy": 0.0,
                "by_answer_type": {},
            },
            common.PAIRED_WRONG_SCENE: {
                "correct": 0,
                "total": QUESTION_COUNT,
                "accuracy": 0.0,
                "by_answer_type": {},
            },
        },
        "v94_same_rows": {
            "correct": v94_correct,
            "total": QUESTION_COUNT,
            "accuracy": v94_accuracy,
            "by_answer_type": {},
        },
        "v96_accuracy_margin_over_v94_same_rows": primary_accuracy - v94_accuracy,
        "counterfactual": {
            "unit_count": CHANGED_UNIT_COUNT,
            "side_count": CHANGED_SIDE_COUNT,
            "canonical_correct_sides": FINAL_GATE_CONTRACT["changed_side_correct_minimum"],
            "canonical_complete_units": FINAL_GATE_CONTRACT["complete_changed_units_minimum"],
            "canonical_prediction_changed_units": FINAL_GATE_CONTRACT[
                "canonical_prediction_changing_units_minimum"
            ],
        },
        "comparisons": {},
        "stable_invariant": {
            "side_count": QUESTION_COUNT - CHANGED_SIDE_COUNT,
            "unit_count": QUESTION_COUNT // 2 - CHANGED_UNIT_COUNT,
            "invariant_false_change_count": 0,
            "invariant_false_change_rate": 0.0,
        },
    }


def _boundary_nll_metrics() -> dict[str, Any]:
    primary = 1.0
    return {
        "primary_mean_nll": primary,
        "paired_wrong_scene_mean_nll": primary
        + FINAL_GATE_CONTRACT["mean_changed_side_wrong_minus_correct_nll_minimum"],
        "zero_payload_mean_nll": primary + FINAL_GATE_CONTRACT["zero_payload_mean_nll_gap_minimum"],
        "full_interior_permutation_mean_nll": primary
        + FINAL_GATE_CONTRACT["permutation_mean_nll_gap_minimum"],
        "mean_wrong_minus_primary_nll": FINAL_GATE_CONTRACT[
            "mean_changed_side_wrong_minus_correct_nll_minimum"
        ],
        "mean_changed_wrong_minus_primary_nll": FINAL_GATE_CONTRACT[
            "mean_changed_side_wrong_minus_correct_nll_minimum"
        ],
        "zero_payload_mean_nll_gap": FINAL_GATE_CONTRACT["zero_payload_mean_nll_gap_minimum"],
        "full_interior_permutation_mean_nll_gap": FINAL_GATE_CONTRACT[
            "permutation_mean_nll_gap_minimum"
        ],
        "row_count_per_arm": QUESTION_COUNT,
        "changed_row_count": CHANGED_SIDE_COUNT,
    }


def test_deferred_nll_aggregate_schema_is_exact_finite_and_count_bound() -> None:
    metrics = _boundary_nll_metrics()
    nll_module.validate_nll_metrics_v96_final(metrics)
    common.assert_aggregate_only_v96_final(metrics)

    extra = {**metrics, "per_row_nll": [1.0]}
    with pytest.raises(ValueError, match="fields changed"):
        nll_module.validate_nll_metrics_v96_final(extra)

    nonfinite = {**metrics, "primary_mean_nll": float("nan")}
    with pytest.raises(ValueError, match="finite"):
        nll_module.validate_nll_metrics_v96_final(nonfinite)

    wrong_count = {**metrics, "changed_row_count": CHANGED_SIDE_COUNT - 1}
    with pytest.raises(ValueError, match="row counts"):
        nll_module.validate_nll_metrics_v96_final(wrong_count)


def _final_gates(
    structured: dict[str, Any] | None = None,
    nll: dict[str, Any] | None = None,
    **controls: Any,
) -> dict[str, bool]:
    # Imported lazily so this test module remains useful while the final sealer
    # is being added in the same implementation milestone.
    from semantic_3d_chat.evaluation.seal_v96_deferred_final import (
        final_gate_results_v96_final,
    )

    return final_gate_results_v96_final(
        structured or _boundary_structured_metrics(),
        nll or _boundary_nll_metrics(),
        FINAL_GATE_CONTRACT,
        fixed_candidate_immutable=controls.get("fixed_candidate_immutable", True),
        prefix_invariant=controls.get("prefix_invariant", True),
        label_isolation_proven=controls.get("label_isolation_proven", True),
        protected_read_count=controls.get("protected_read_count", 0),
        separate_leakage_gate_required=controls.get("separate_leakage_gate_required", True),
    )


def test_every_deferred_final_gate_boundary_passes_inclusively() -> None:
    gates = _final_gates()
    assert gates
    assert all(gates.values()), gates


def test_deferred_final_gate_rejects_a_weakened_contract() -> None:
    from semantic_3d_chat.evaluation.seal_v96_deferred_final import (
        final_gate_results_v96_final,
    )

    weakened = {
        **FINAL_GATE_CONTRACT,
        "canonical_accuracy_minimum": 0.0,
    }
    with pytest.raises(ValueError, match="contract"):
        final_gate_results_v96_final(
            _boundary_structured_metrics(),
            _boundary_nll_metrics(),
            weakened,
            fixed_candidate_immutable=True,
            prefix_invariant=True,
            label_isolation_proven=True,
            protected_read_count=0,
            separate_leakage_gate_required=True,
        )


@pytest.mark.parametrize(
    ("section", "path", "below"),
    [
        *[
            (
                "structured",
                ("arms", common.PRIMARY, "by_answer_type", answer_type, "correct"),
                FINAL_GATE_CONTRACT[f"{answer_type}_correct_minimum"] - 1,
            )
            for answer_type in ANSWER_TYPE_TOTALS
        ],
        (
            "structured",
            ("counterfactual", "canonical_correct_sides"),
            FINAL_GATE_CONTRACT["changed_side_correct_minimum"] - 1,
        ),
        (
            "structured",
            ("counterfactual", "canonical_complete_units"),
            FINAL_GATE_CONTRACT["complete_changed_units_minimum"] - 1,
        ),
        (
            "structured",
            ("counterfactual", "canonical_prediction_changed_units"),
            FINAL_GATE_CONTRACT["canonical_prediction_changing_units_minimum"] - 1,
        ),
        (
            "nll",
            ("mean_changed_wrong_minus_primary_nll",),
            0.199999,
        ),
        ("nll", ("zero_payload_mean_nll_gap",), 0.499999),
        (
            "nll",
            ("full_interior_permutation_mean_nll_gap",),
            0.349999,
        ),
    ],
)
def test_each_numeric_deferred_final_gate_fails_immediately_below_threshold(
    section: str,
    path: tuple[str, ...],
    below: float,
) -> None:
    structured = _boundary_structured_metrics()
    nll = _boundary_nll_metrics()
    target: dict[str, Any] = structured if section == "structured" else nll
    for component in path[:-1]:
        target = target[component]
    target[path[-1]] = below
    if section == "structured" and "by_answer_type" in path:
        target["accuracy"] = below / target["total"]
    gates = _final_gates(structured, nll)
    assert not all(gates.values()), gates


def test_discrete_216_row_accuracy_and_margin_edges_fail_below_threshold() -> None:
    structured = _boundary_structured_metrics()
    primary = structured["arms"][common.PRIMARY]
    primary["correct"] = 140
    primary["accuracy"] = 140 / QUESTION_COUNT
    structured["v96_accuracy_margin_over_v94_same_rows"] = (
        primary["accuracy"] - structured["v94_same_rows"]["accuracy"]
    )
    assert not all(_final_gates(structured=structured).values())

    structured = _boundary_structured_metrics()
    comparator = structured["v94_same_rows"]
    comparator["correct"] = structured["arms"][common.PRIMARY]["correct"] - 6
    comparator["accuracy"] = comparator["correct"] / QUESTION_COUNT
    structured["v96_accuracy_margin_over_v94_same_rows"] = (
        structured["arms"][common.PRIMARY]["accuracy"] - comparator["accuracy"]
    )
    assert not all(_final_gates(structured=structured).values())


@pytest.mark.parametrize(
    ("path", "wrong_total"),
    [
        *[
            (
                ("arms", common.PRIMARY, "by_answer_type", answer_type, "total"),
                FINAL_GATE_CONTRACT[f"{answer_type}_total"] - 1,
            )
            for answer_type in ANSWER_TYPE_TOTALS
        ],
        (
            ("counterfactual", "side_count"),
            FINAL_GATE_CONTRACT["changed_side_total"] - 1,
        ),
        (
            ("counterfactual", "unit_count"),
            FINAL_GATE_CONTRACT["changed_unit_total"] - 1,
        ),
    ],
)
def test_deferred_gate_rejects_incomplete_metric_denominators(
    path: tuple[str, ...], wrong_total: int
) -> None:
    structured = _boundary_structured_metrics()
    target: dict[str, Any] = structured
    for component in path[:-1]:
        target = target[component]
    target[path[-1]] = wrong_total
    with pytest.raises(ValueError, match="count/accuracy|scope changed"):
        _final_gates(structured=structured)


@pytest.mark.parametrize(
    ("control", "value"),
    [
        ("fixed_candidate_immutable", False),
        ("prefix_invariant", False),
        ("label_isolation_proven", False),
        ("protected_read_count", 1),
        ("separate_leakage_gate_required", False),
    ],
)
def test_each_nonmetric_deferred_final_control_fails_closed(control: str, value: object) -> None:
    assert not all(_final_gates(**{control: value}).values())


def test_nll_ordering_controls_fail_even_when_reported_gap_is_positive() -> None:
    nll = _boundary_nll_metrics()
    nll["zero_payload_mean_nll"] = nll["primary_mean_nll"]
    assert not all(_final_gates(nll=nll).values())

    nll = _boundary_nll_metrics()
    nll["full_interior_permutation_mean_nll"] = nll["primary_mean_nll"]
    assert not all(_final_gates(nll=nll).values())
