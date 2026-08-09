from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

from semantic_3d_chat.evaluation.candidate_gate_detail import (
    build_candidate_gate_detail,
)
from semantic_3d_chat.training.pair_curriculum import pair_gate_metrics


@dataclass(frozen=True)
class _Record:
    scene_id: str
    question_id: str
    answer: str
    question: str = "forbidden question text"
    answer_type: str = "forbidden answer type"
    target_xyz: tuple[float, float, float] = (9.0, 8.0, 7.0)


@dataclass(frozen=True)
class _Unit:
    records: tuple[_Record, _Record]


def _units() -> list[_Unit]:
    return [
        _Unit(
            records=(
                _Record("scene_000003", "q_000101", "red"),
                _Record("scene_000004", "q_000201", "blue"),
            )
        ),
        _Unit(
            records=(
                _Record("scene_000003", "q_000102", "left"),
                _Record("scene_000004", "q_000202", "right"),
            )
        ),
    ]


def test_canonical_target_detail_expands_every_unit_and_side_without_metadata() -> None:
    detail = build_candidate_gate_detail(
        _units(),
        torch.tensor([[0.75, -0.25], [0.6, 0.5]], requires_grad=True),
        ranking_margin=0.5,
    )

    assert detail["candidate_representation"] == "canonical_training_targets"
    assert detail["contains_canonical_training_targets"] is True
    assert detail["unit_count"] == 2
    assert detail["side_count"] == 4
    assert detail["summary_counts"] == {
        "changed_units_passed": 1,
        "side_preferences_passed": 3,
        "prediction_flips_passed": 1,
        "wrong_prefix_flips_passed": 1,
        "sides_at_configured_margin": 3,
    }
    first, second = detail["units"]
    assert first["changed_unit_passed"] is False
    assert first["prediction_flip_passed"] is False
    assert first["wrong_prefix_flip_passed"] is False
    assert first["configured_margin_passed"] is False
    assert first["sides"][0] == {
        "side_index": 0,
        "scene_id": "scene_000003",
        "question_id": "q_000101",
        "own_vs_alternate_candidate_logit_margin": 0.75,
        "predicted_preference": "own",
        "own_preference_passed": True,
        "configured_margin_passed": True,
        "own_canonical_target": "red",
        "alternate_canonical_target": "blue",
        "predicted_canonical_target": "red",
    }
    assert first["sides"][1]["predicted_preference"] == "alternate"
    assert first["sides"][1]["predicted_canonical_target"] == "red"
    assert second["changed_unit_passed"] is True
    assert second["prediction_flip_passed"] is True
    assert second["configured_margin_passed"] is True

    serialized = json.dumps(detail, sort_keys=True)
    for forbidden in (
        "forbidden question text",
        "forbidden answer type",
        "target_xyz",
        "counterfactual_change_type",
        "pair_id",
        "question_key",
    ):
        assert forbidden not in serialized


class _TokenOnlyRecord:
    def __init__(self, scene_id: str, question_id: str) -> None:
        self.scene_id = scene_id
        self.question_id = question_id

    @property
    def answer(self) -> str:
        raise AssertionError("token-ID detail must not read canonical answer text")


@dataclass(frozen=True)
class _TokenOnlyUnit:
    records: tuple[_TokenOnlyRecord, _TokenOnlyRecord]


def test_token_id_detail_never_reads_or_serializes_target_text() -> None:
    units = [
        _TokenOnlyUnit(
            (
                _TokenOnlyRecord("scene_000003", "q_000101"),
                _TokenOnlyRecord("scene_000004", "q_000201"),
            )
        )
    ]
    detail = build_candidate_gate_detail(
        units,
        [[0.0, 1.25]],
        ranking_margin=0.5,
        candidate_token_ids=[[[1192, 9503], [9503, 1192]]],
    )

    assert detail["candidate_representation"] == "candidate_token_ids"
    assert detail["contains_canonical_training_targets"] is False
    unit = detail["units"][0]
    assert unit["changed_unit_passed"] is False
    assert unit["prediction_flip_passed"] is False
    assert unit["configured_margin_passed"] is False
    first, second = unit["sides"]
    assert first["predicted_preference"] == "alternate"
    assert first["predicted_candidate_token_id"] == 9503
    assert first["own_preference_passed"] is False
    assert second["predicted_preference"] == "own"
    assert second["predicted_candidate_token_id"] == 9503
    assert second["configured_margin_passed"] is True
    serialized = json.dumps(detail, sort_keys=True)
    assert "canonical_target" not in serialized
    assert "answer" not in serialized


def test_full_vocab_margins_expand_every_opaque_side_and_unit() -> None:
    detail = build_candidate_gate_detail(
        _units(),
        [[0.75, 0.25], [0.6, -0.5]],
        ranking_margin=0.5,
        candidate_token_ids=[
            [[1192, 9503], [9503, 1192]],
            [[4432, 8841], [8841, 4432]],
        ],
        full_vocab_margins=[[0.5, 0.125], [1.0, -0.25]],
    )

    assert detail["full_vocab_first_token_evaluated"] is True
    assert detail["contains_canonical_training_targets"] is False
    assert detail["summary_counts"]["full_vocab_top1_sides_passed"] == 3
    assert detail["summary_counts"]["full_vocab_top1_units_passed"] == 1
    first, second = detail["units"]
    assert first["full_vocab_top1_unit_passed"] is True
    assert second["full_vocab_top1_unit_passed"] is False
    assert first["sides"][0]["first_token_target_vs_best_other_logit_margin"] == pytest.approx(0.5)
    assert second["sides"][1]["full_vocab_top1_passed"] is False


def test_full_vocab_detail_rejects_misaligned_or_nonfinite_evidence() -> None:
    with pytest.raises(ValueError, match="shape"):
        build_candidate_gate_detail(
            _units(),
            [[0.1, 0.2], [0.3, 0.4]],
            full_vocab_margins=[[0.1, 0.2]],
        )
    with pytest.raises(ValueError, match="NaN or infinity"):
        build_candidate_gate_detail(
            _units()[:1],
            [[0.1, 0.2]],
            full_vocab_margins=[[0.1, float("inf")]],
        )


def test_detail_counts_exactly_expand_existing_aggregate_gate() -> None:
    margins = [[0.75, -0.25], [0.6, 0.5]]
    detail = build_candidate_gate_detail(_units(), margins, ranking_margin=0.5)
    aggregate = pair_gate_metrics(
        margins,
        ranking_margin=0.5,
        ranking_mode="candidate_logit",
    )
    counts = detail["summary_counts"]

    assert counts["changed_units_passed"] / detail["unit_count"] == pytest.approx(
        aggregate["changed_unit_accuracy"]
    )
    assert counts["side_preferences_passed"] / detail["side_count"] == pytest.approx(
        aggregate["side_accuracy"]
    )
    assert counts["prediction_flips_passed"] / detail["unit_count"] == pytest.approx(
        aggregate["prediction_flip_rate"]
    )
    assert counts["wrong_prefix_flips_passed"] / detail["unit_count"] == pytest.approx(
        aggregate["wrong_prefix_flip_rate"]
    )


@pytest.mark.parametrize(
    ("margins", "token_ids", "match"),
    [
        ([[0.1]], None, "shape"),
        ([[float("nan"), 0.2]], None, "NaN or infinity"),
        ([[0.1, 0.2]], [[[1, 2], [1, 2]]], "reverse"),
        ([[0.1, 0.2]], [[[1, 1], [1, 1]]], "must differ"),
        ([[0.1, 0.2]], [[[-1, 2], [2, -1]]], "cannot be negative"),
    ],
)
def test_candidate_detail_rejects_misaligned_or_invalid_evidence(
    margins: list[list[float]],
    token_ids: list[list[list[int]]] | None,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        build_candidate_gate_detail(
            _units()[:1],
            margins,
            candidate_token_ids=token_ids,
        )


def test_candidate_detail_rejects_nonopaque_identifiers() -> None:
    leaked = [
        _Unit(
            (
                _Record("red_cube_scene", "what_color", "red"),
                _Record("scene_000004", "q_000201", "blue"),
            )
        )
    ]
    with pytest.raises(ValueError, match="opaque"):
        build_candidate_gate_detail(leaked, [[0.2, 0.3]])


def test_chat_package_never_imports_training_candidate_detail() -> None:
    project_root = Path(__file__).parents[1]
    chat_root = project_root / "src" / "semantic_3d_chat" / "chat"
    for source_path in chat_root.glob("*.py"):
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
        imported_modules = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        }
        imported_modules.update(
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        )
        assert "semantic_3d_chat.evaluation.candidate_gate_detail" not in imported_modules

    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(project_root / "src")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import semantic_3d_chat.chat.runtime; "
                "assert 'semantic_3d_chat.evaluation.candidate_gate_detail' "
                "not in sys.modules"
            ),
        ],
        cwd=project_root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
