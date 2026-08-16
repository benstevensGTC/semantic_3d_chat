from __future__ import annotations

from pathlib import Path

import pytest
import torch

from semantic_3d_chat.data.dataset import QARecord
from semantic_3d_chat.evaluation.v29_counterfactual_representation_audit import (
    CounterfactualScenePair,
    ExpectedChangeUnit,
    SceneRepresentation,
    _diagnosis,
    _guard_runtime_path,
    compare_paired_to_unrelated,
    counterfactual_scene_pairs,
    representation_delta_report,
    summarize_teacher_forced_margins,
    tensor_delta_metrics,
    unrelated_scene_pairs,
)
from semantic_3d_chat.language.prefix_injection import prefix_sha256


def _record(
    scene_id: str,
    *,
    pair_id: str,
    role: str,
    key: str,
    answer: str,
    expected_change: bool = True,
    question: str = "Is it left or right?",
) -> QARecord:
    return QARecord(
        scene_id=scene_id,
        question_id=f"{scene_id}_{key}",
        question=question,
        answer=answer,
        answer_type="spatial_relation",
        target_xyz=None,
        counterfactual_pair_id=pair_id,
        counterfactual_question_key=key,
        counterfactual_expected_change=expected_change,
        counterfactual_role=role,
        counterfactual_change_type="mirror",
    )


def _pair(
    pair_id: str,
    reference_scene: str,
    counterfactual_scene: str,
) -> CounterfactualScenePair:
    unit = ExpectedChangeUnit(
        pair_id=pair_id,
        question_key=f"{pair_id}_question",
        change_type="mirror",
        question="Is it left or right?",
        reference_scene_id=reference_scene,
        counterfactual_scene_id=counterfactual_scene,
        reference_answer="left",
        counterfactual_answer="right",
    )
    return CounterfactualScenePair(
        pair_id=pair_id,
        change_type="mirror",
        reference_scene_id=reference_scene,
        counterfactual_scene_id=counterfactual_scene,
        expected_change_units=(unit,),
    )


def _representation(scene_id: str, value: float) -> SceneRepresentation:
    tokens = torch.full((1, 3, 4), value, dtype=torch.float32)
    # Fixed boundary rows ensure prefix deltas are measured over the complete
    # scene prefix while preserving the question-independent markers.
    boundary = torch.zeros(1, 1, 4)
    prefix = torch.cat((boundary, tokens, boundary), dim=1)
    return SceneRepresentation(
        scene_id=scene_id,
        scene_tokens=tokens,
        prefix=prefix,
        scene_token_sha256=prefix_sha256(tokens),
        prefix_sha256=prefix_sha256(prefix),
        source_voxel_count=20,
        input_voxel_count=10,
        processed_voxel_count=10,
        all_voxels_covered=True,
    )


def test_counterfactual_scene_pairs_keeps_qa_outside_runtime_shape() -> None:
    records = [
        _record(
            "scene_000019",
            pair_id="pair_000009",
            role="reference",
            key="changed",
            answer="left",
        ),
        _record(
            "scene_000020",
            pair_id="pair_000009",
            role="counterfactual",
            key="changed",
            answer="right",
        ),
        _record(
            "scene_000019",
            pair_id="pair_000009",
            role="reference",
            key="stable",
            answer="yes",
            expected_change=False,
            question="Is there a chair?",
        ),
        _record(
            "scene_000020",
            pair_id="pair_000009",
            role="counterfactual",
            key="stable",
            answer="yes",
            expected_change=False,
            question="Is there a chair?",
        ),
    ]

    pairs = counterfactual_scene_pairs(
        records,
        expected_scene_ids=("scene_000019", "scene_000020"),
    )

    assert len(pairs) == 1
    assert pairs[0].scene_ids == ("scene_000019", "scene_000020")
    assert len(pairs[0].expected_change_units) == 1
    unit = pairs[0].expected_change_units[0]
    assert unit.reference_answer == "left"
    assert unit.counterfactual_answer == "right"


def test_counterfactual_scene_pairs_rejects_changed_unit_without_answer_change() -> None:
    records = [
        _record(
            "scene_000019",
            pair_id="pair_000009",
            role="reference",
            key="changed",
            answer="left",
        ),
        _record(
            "scene_000020",
            pair_id="pair_000009",
            role="counterfactual",
            key="changed",
            answer="left",
        ),
    ]

    with pytest.raises(ValueError, match="identical answers"):
        counterfactual_scene_pairs(records)


def test_tensor_delta_metrics_reports_every_changed_token() -> None:
    first = torch.zeros(1, 3, 2)
    second = first.clone()
    second[0, 1] = torch.tensor([3.0, 4.0])

    metrics = tensor_delta_metrics(first, second)

    assert metrics["shape"] == [1, 3, 2]
    assert metrics["delta_rms"] == pytest.approx((25.0 / 6.0) ** 0.5)
    assert metrics["changed_token_count_exact"] == 1
    assert metrics["changed_token_fraction_exact"] == pytest.approx(1 / 3)
    assert metrics["per_token_rms"] == pytest.approx([0.0, (25.0 / 2.0) ** 0.5, 0.0])
    assert len(metrics["per_token_relative_rms"]) == 3
    assert len(metrics["per_token_cosine"]) == 3


def test_unrelated_pairs_exclude_only_configured_physical_pairs() -> None:
    pairs = (
        _pair("pair_a", "scene_000019", "scene_000020"),
        _pair("pair_b", "scene_000021", "scene_000022"),
    )

    unrelated = unrelated_scene_pairs(
        ["scene_000019", "scene_000020", "scene_000021", "scene_000022"],
        pairs,
    )

    assert len(unrelated) == 4
    assert ("scene_000019", "scene_000020") not in unrelated
    assert ("scene_000021", "scene_000022") not in unrelated


def test_representation_report_compares_pair_deltas_to_unrelated_scenes() -> None:
    pairs = (
        _pair("pair_a", "scene_000019", "scene_000020"),
        _pair("pair_b", "scene_000021", "scene_000022"),
    )
    representations = {
        "scene_000019": _representation("scene_000019", 0.00),
        "scene_000020": _representation("scene_000020", 0.01),
        "scene_000021": _representation("scene_000021", 1.00),
        "scene_000022": _representation("scene_000022", 1.02),
    }

    report = representation_delta_report(representations, pairs)

    assert report["configured_pair_count"] == 2
    assert report["unrelated_pair_count"] == 4
    assert report["all_configured_scene_tokens_distinct"] is True
    assert report["all_configured_prefixes_distinct"] is True
    contrast = report["contrast"]["scene_tokens"]
    assert contrast["paired_mean_rms_to_unrelated_mean_ratio"] < 0.1
    assert len(contrast["per_pair_unrelated_percentile"]) == 2
    assert len(report["paired"][0]["scene_tokens"]["per_token_rms"]) == 3
    assert "per_token_rms" not in report["unrelated"][0]["scene_tokens"]


def test_compare_paired_to_unrelated_rejects_empty_control_distribution() -> None:
    paired = [{"prefix": {"delta_rms": 1.0, "cosine": 0.9}, "pair_id": "pair_a"}]
    with pytest.raises(ValueError, match="cannot be empty"):
        compare_paired_to_unrelated(paired, [], representation_key="prefix")


def test_teacher_margin_summary_separates_flips_from_correct_binding() -> None:
    rows = [
        {
            "reference_scene": {
                "first_minus_second_mean_log_probability_margin": 2.0,
                "first_differing_token": {"first_minus_second_logit_margin": 3.0},
            },
            "counterfactual_scene": {
                "first_minus_second_mean_log_probability_margin": -1.0,
                "first_differing_token": {"first_minus_second_logit_margin": -2.0},
            },
        },
        {
            "reference_scene": {
                "first_minus_second_mean_log_probability_margin": 1.5,
                "first_differing_token": {"first_minus_second_logit_margin": 2.5},
            },
            "counterfactual_scene": {
                "first_minus_second_mean_log_probability_margin": 0.5,
                "first_differing_token": {"first_minus_second_logit_margin": 1.0},
            },
        },
    ]

    summary = summarize_teacher_forced_margins(rows)

    assert summary["two_sided_sequence_preference_accuracy"] == 0.5
    assert summary["per_side_sequence_preference_accuracy"] == 0.75
    assert summary["same_question_preference_flip_rate"] == 0.5
    assert summary["two_sided_first_differing_token_accuracy"] == 0.5
    assert summary["mean_absolute_scene_induced_sequence_log_odds_shift"] == 2.0


def test_diagnosis_localizes_distinct_prefix_without_decoder_flip() -> None:
    representation = {
        "all_configured_prefixes_distinct": True,
        "contrast": {
            "prefix": {
                "paired_mean_rms_to_unrelated_mean_ratio": 0.15,
                "per_pair_unrelated_percentile": [
                    {"pair_id": "pair_a", "percentile_among_unrelated": 0.0}
                ],
            }
        },
    }
    teacher = {
        "summary": {
            "same_question_preference_flip_rate": 0.0,
            "two_sided_sequence_preference_accuracy": 0.0,
        }
    }

    diagnosis = _diagnosis(representation, teacher)

    assert diagnosis["counterfactual_change_is_diluted_relative_to_scene_identity"] is True
    assert diagnosis["pair_ids_below_every_unrelated_prefix_delta"] == ["pair_a"]
    assert diagnosis["decoder_preference_flip_rate"] == 0.0
    assert "downstream of prefix construction" in diagnosis["failure_localization"]


def test_runtime_path_guard_rejects_oracle_qa_and_features(tmp_path: Path) -> None:
    for forbidden in ("oracle", "qa", "features", "rendered"):
        path = tmp_path / forbidden / "payload.npz"
        with pytest.raises(ValueError, match="forbidden runtime path"):
            _guard_runtime_path(path, purpose="test")

    safe = tmp_path / "maps" / "scene_000019" / "voxel_map.npz"
    assert _guard_runtime_path(safe, purpose="test") == safe.resolve()
