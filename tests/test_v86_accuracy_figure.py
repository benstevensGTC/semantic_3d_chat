from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from semantic_3d_chat.evaluation.v86_accuracy_figure import (
    ANSWER_TYPE_ORDER,
    DEFAULT_EVALUATION_REPORT,
    DEFAULT_FIGURE,
    DEFAULT_SUMMARY,
    SEALED_EVALUATION_SHA256,
    generate_figure,
    load_sealed_evaluation,
)

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / DEFAULT_EVALUATION_REPORT


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _generate(root: Path) -> dict[str, Any]:
    return generate_figure(REPORT, root / "figure.png", root / "summary.json")


def test_v86_figure_source_is_exact_terminal_evaluation() -> None:
    report = load_sealed_evaluation(REPORT)

    assert _sha256(REPORT) == SEALED_EVALUATION_SHA256
    assert report["status"] == "model_gates_fail_not_runtime_promotable"
    assert report["metrics"]["canonical_type_specific"] == {
        "accuracy": 86 / 138,
        "correct": 86,
        "total": 138,
    }
    assert report["metrics"]["generic_smoke"]["correct"] == 3
    assert report["metrics"]["causal_control"]["canonical_prediction_changes"] == 2


def test_v86_figure_is_deterministic_and_valid_png(tmp_path: Path) -> None:
    first = _generate(tmp_path / "first")
    second = _generate(tmp_path / "second")
    first_figure = Path(first["figure"]["path"])
    second_figure = Path(second["figure"]["path"])

    assert _sha256(first_figure) == _sha256(second_figure)
    with Image.open(first_figure) as image:
        assert image.format == "PNG"
        assert image.width >= 1_500
        assert image.height >= 850


def test_v86_figure_summary_is_scope_bounded(tmp_path: Path) -> None:
    summary = _generate(tmp_path)

    assert summary["source"]["sha256"] == SEALED_EVALUATION_SHA256
    assert summary["scope"]["single_scene_training_authorized_evaluation"] is True
    assert summary["scope"]["held_out_generalization"] is False
    assert summary["scope"]["new_inference"] is False
    assert summary["scope"]["qa_or_oracle_loaded"] is False
    assert summary["scope"]["runtime_promotion_authorized"] is False
    assert tuple(summary["metrics"]["canonical_accuracy_by_answer_type"]) == (ANSWER_TYPE_ORDER)
    assert summary["metrics"]["overall"]["correct"] == 86
    assert summary["metrics"]["acceptance_threshold"] == 0.8
    assert summary["metrics"]["acceptance_passed"] is False
    assert "not held-out generalization" in summary["figure"]["caption"].lower()


def test_v86_figure_fails_before_parsing_modified_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    modified = tmp_path / "evaluation.json"
    modified.write_bytes(REPORT.read_bytes() + b" ")
    parsed = False

    def unexpected_parse(_payload: object) -> object:
        nonlocal parsed
        parsed = True
        raise AssertionError("modified report must not be parsed")

    monkeypatch.setattr(json, "loads", unexpected_parse)
    with pytest.raises(ValueError, match="evaluation digest differs"):
        load_sealed_evaluation(modified)
    assert parsed is False


def test_default_v86_figure_and_summary_match_generated_hashes() -> None:
    summary_path = ROOT / DEFAULT_SUMMARY
    figure_path = ROOT / DEFAULT_FIGURE
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert summary["source"]["path"] == DEFAULT_EVALUATION_REPORT.as_posix()
    assert summary["figure"]["path"] == DEFAULT_FIGURE.as_posix()
    assert summary["figure"]["sha256"] == _sha256(figure_path)
