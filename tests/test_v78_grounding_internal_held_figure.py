from __future__ import annotations

import hashlib
import json
import runpy
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PLOTTER = runpy.run_path(str(ROOT / "scripts/plot_v78_grounding_internal_held.py"))
REPORT = ROOT / "reports/gemma4/metrics/v78_grounding_sidecar_internal_held.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _generate(root: Path) -> dict[str, Any]:
    return PLOTTER["generate_figure"](
        REPORT,
        root / "figure.png",
        root / "manifest.json",
    )


def test_plotter_authenticates_internal_historical_scope() -> None:
    report = PLOTTER["load_sealed_report"](REPORT)

    assert _sha256(REPORT) == PLOTTER["SEALED_REPORT_SHA256"]
    assert report["status"] == "internal_historical_diagnostic_only"
    assert report["runtime_promotion_authorized"] is False
    assert report["official_validation_loaded"] is False
    assert report["official_test_loaded"] is False
    assert report["deferred_final_loaded"] is False
    assert report["oracle_files_loaded"] is False
    assert report["split"]["pair_disjoint"] is True
    assert report["split"]["scene_disjoint"] is True
    assert report["split"]["held_grounded_rows"] == 94


def test_figure_is_deterministic_and_valid_png(tmp_path: Path) -> None:
    first = _generate(tmp_path / "first")
    second = _generate(tmp_path / "second")
    first_path = Path(first["figure"]["path"])
    second_path = Path(second["figure"]["path"])

    assert _sha256(first_path) == _sha256(second_path)
    with Image.open(first_path) as image:
        assert image.format == "PNG"
        assert image.size == (1_952, 1_008)


def test_manifest_forbids_official_runtime_or_promotion_claims(tmp_path: Path) -> None:
    manifest = _generate(tmp_path)

    assert manifest["artifact"] == "v78_historical_held_grounding_posthoc_figure_v1"
    assert manifest["source"]["sha256"] == PLOTTER["SEALED_REPORT_SHA256"]
    assert manifest["scope"] == {
        "historical_internal_held_only": True,
        "post_hoc_visualization_only": True,
        "new_evaluation": False,
        "official_validation": False,
        "runtime_evidence": False,
        "promotion_evidence": False,
        "runtime_promotion_authorized": False,
        "source_file_count": 1,
        "model_loaded": False,
        "predictions_or_references_loaded": False,
        "qa_or_oracle_loaded": False,
        "unopened_split_loaded": False,
    }
    caption = manifest["figure"]["caption"].lower()
    assert "internal diagnostic evidence only" in caption
    assert "paired-wrong-scene aggregate is nearly unchanged" in caption
    assert "10 changed-target sides show 90% correct-scene preference" in caption
    assert "position/question shuffles and zero scene are the stronger controls" in caption


def test_modified_report_is_rejected_before_plotting(tmp_path: Path) -> None:
    modified = tmp_path / "report.json"
    modified.write_bytes(REPORT.read_bytes() + b" ")

    with pytest.raises(ValueError, match="report digest differs"):
        PLOTTER["load_sealed_report"](modified)


def test_default_manifest_matches_generated_figure() -> None:
    manifest = json.loads((ROOT / PLOTTER["DEFAULT_MANIFEST"]).read_text())
    figure_path = ROOT / PLOTTER["DEFAULT_FIGURE"]

    assert manifest["source"]["path"] == PLOTTER["DEFAULT_REPORT"].as_posix()
    assert manifest["figure"]["path"] == PLOTTER["DEFAULT_FIGURE"].as_posix()
    assert manifest["figure"]["sha256"] == _sha256(figure_path)
