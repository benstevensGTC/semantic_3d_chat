from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from semantic_3d_chat.evaluation.v85_development_figure import (
    ANSWER_TYPE_ORDER,
    DEFAULT_DEVELOPMENT_REPORT,
    DEFAULT_FIGURE,
    DEFAULT_SUMMARY,
    SEALED_DEVELOPMENT_SHA256,
    generate_figure,
    load_sealed_development_report,
)

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / DEFAULT_DEVELOPMENT_REPORT


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _generate(root: Path) -> dict[str, Any]:
    return generate_figure(REPORT, root / "figure.png", root / "summary.json")


def test_v85_figure_source_is_exact_sealed_development_aggregate() -> None:
    report = load_sealed_development_report(REPORT)

    assert _sha256(REPORT) == SEALED_DEVELOPMENT_SHA256
    assert report["artifact"] == "gemma4_v85_strict_multiscene_development_score_v1"
    assert report["metrics"]["canonical_type_specific"] == {
        "accuracy": 214 / 384,
        "correct": 214,
        "total": 384,
    }
    assert report["metrics"]["answer_frequency_majority_baseline"] == {
        "accuracy": 62 / 384,
        "correct": 62,
        "total": 384,
    }


def test_v85_development_figure_is_deterministic_and_valid_png(tmp_path: Path) -> None:
    first = _generate(tmp_path / "first")
    second = _generate(tmp_path / "second")
    first_figure = Path(first["figure"]["path"])
    second_figure = Path(second["figure"]["path"])

    assert _sha256(first_figure) == _sha256(second_figure)
    with Image.open(first_figure) as image:
        assert image.format == "PNG"
        assert image.width >= 1_500
        assert image.height >= 900


def test_v85_figure_summary_is_machine_bound_and_scope_bounded(tmp_path: Path) -> None:
    summary = _generate(tmp_path)

    assert summary["source"]["sha256"] == SEALED_DEVELOPMENT_SHA256
    assert summary["scope"] == {
        "development_only": True,
        "pair_and_scene_disjoint": True,
        "development_scene_count": 16,
        "development_question_count": 384,
        "official_validation": False,
        "official_test": False,
        "deferred_final": False,
        "post_hoc_visualization_only": True,
        "new_evaluation": False,
        "new_inference": False,
        "model_loaded": False,
        "predictions_or_references_loaded": False,
        "scene_memory_or_map_loaded": False,
        "qa_or_oracle_loaded": False,
        "runtime_promotion_authorized": False,
    }
    assert tuple(summary["metrics"]["canonical_accuracy_by_answer_type"]) == (
        ANSWER_TYPE_ORDER
    )
    assert summary["metrics"]["overall"]["correct"] == 214
    assert summary["metrics"]["answer_frequency_majority_baseline"]["correct"] == 62
    assert "development evidence only" in summary["figure"]["caption"].lower()


def test_v85_figure_fails_closed_before_parsing_modified_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    modified = tmp_path / "development.json"
    modified.write_bytes(REPORT.read_bytes() + b" ")
    parsed = False

    def unexpected_parse(_payload: object) -> object:
        nonlocal parsed
        parsed = True
        raise AssertionError("modified report must not be parsed")

    monkeypatch.setattr(json, "loads", unexpected_parse)
    with pytest.raises(ValueError, match="development report digest differs"):
        load_sealed_development_report(modified)
    assert parsed is False


def test_default_v85_figure_and_summary_match_generated_hashes() -> None:
    summary_path = ROOT / DEFAULT_SUMMARY
    figure_path = ROOT / DEFAULT_FIGURE
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert summary["source"]["path"] == DEFAULT_DEVELOPMENT_REPORT.as_posix()
    assert summary["figure"]["path"] == DEFAULT_FIGURE.as_posix()
    assert summary["figure"]["sha256"] == _sha256(figure_path)
