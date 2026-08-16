from __future__ import annotations

import hashlib
import json
import runpy
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PLOTTER = runpy.run_path(str(ROOT / "scripts/plot_v75_official_validation.py"))
SCORE = ROOT / "reports/gemma4/metrics/v75_official_validation_score.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _generate(root: Path) -> dict[str, Any]:
    return PLOTTER["generate_figures"](
        SCORE,
        root / "figures",
        root / "manifest.json",
    )


def test_plotter_accepts_only_the_exact_sealed_aggregate() -> None:
    score = PLOTTER["load_sealed_score"](SCORE)

    assert _sha256(SCORE) == PLOTTER["SEALED_SCORE_SHA256"]
    assert score["artifact"] == "v75_official_validation_score_v1"
    assert score["scope"]["split"] == "validation"
    assert score["scope"]["candidate_count"] == 1
    assert score["scope"]["question_count"] == 216
    assert score["scope"]["model_loaded"] is False
    assert score["scope"]["scene_map_loaded"] is False
    assert score["scope"]["simulator_oracle_loaded"] is False
    assert score["scope"]["question_or_answer_text_serialized"] is False


def test_generated_figures_are_deterministic_and_valid_pngs(tmp_path: Path) -> None:
    first = _generate(tmp_path / "first")
    second = _generate(tmp_path / "second")

    assert tuple(first["figures"]) == tuple(PLOTTER["FIGURE_FILENAMES"])
    for name in PLOTTER["FIGURE_FILENAMES"]:
        first_path = Path(first["figures"][name]["path"])
        second_path = Path(second["figures"][name]["path"])
        assert _sha256(first_path) == _sha256(second_path)
        with Image.open(first_path) as image:
            assert image.format == "PNG"
            assert image.width >= 1_600
            assert image.height >= 900


def test_manifest_bounds_plotting_to_post_hoc_aggregate_visualization(
    tmp_path: Path,
) -> None:
    manifest = _generate(tmp_path)
    scope = manifest["scope"]

    assert manifest["artifact"] == "v75_official_validation_posthoc_figures_v1"
    assert manifest["source"]["sha256"] == PLOTTER["SEALED_SCORE_SHA256"]
    assert scope == {
        "post_hoc_visualization_only": True,
        "new_evaluation": False,
        "source_file_count": 1,
        "model_loaded": False,
        "predictions_or_references_loaded": False,
        "scene_map_loaded": False,
        "qa_or_oracle_loaded": False,
        "unopened_split_loaded": False,
        "per_example_grounding_errors_available": False,
        "grounding_visualization": "aggregate_summary_only_no_distribution",
    }
    grounding = manifest["figures"]["grounding_aggregate_summary"]
    assert "no per-example errors" in grounding["caption"].lower()
    assert "no distribution is inferred" in grounding["caption"].lower()


def test_plotter_fails_closed_before_parsing_a_modified_score(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    modified = tmp_path / "score.json"
    modified.write_bytes(SCORE.read_bytes() + b" ")
    parsed = False

    def unexpected_parse(_payload: object) -> object:
        nonlocal parsed
        parsed = True
        raise AssertionError("modified score must not be parsed")

    monkeypatch.setitem(
        PLOTTER["load_sealed_score"].__globals__,
        "json",
        type(
            "RejectingJSON",
            (),
            {"loads": staticmethod(unexpected_parse)},
        ),
    )

    with pytest.raises(ValueError, match="score digest differs"):
        PLOTTER["load_sealed_score"](modified)
    assert parsed is False


def test_default_manifest_matches_generated_figure_hashes() -> None:
    manifest_path = ROOT / PLOTTER["DEFAULT_MANIFEST"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["source"]["path"] == PLOTTER["DEFAULT_SCORE"].as_posix()
    for name, filename in PLOTTER["FIGURE_FILENAMES"].items():
        path = ROOT / PLOTTER["DEFAULT_OUTPUT_DIR"] / filename
        assert manifest["figures"][name]["path"] == path.relative_to(ROOT).as_posix()
        assert manifest["figures"][name]["sha256"] == _sha256(path)
