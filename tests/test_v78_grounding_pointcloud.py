from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from semantic_3d_chat.evaluation import v78_grounding_pointcloud as evidence
from semantic_3d_chat.training.grounding_sidecar_v78 import GroundingRecord

ROOT = Path(__file__).resolve().parents[1]


def _record(scene: str, question_id: str, question: str = "Where is it?") -> GroundingRecord:
    return GroundingRecord(
        question_id=question_id,
        scene_id=scene,
        pair_id="pair_opaque",
        paired_scene_id="scene_paired",
        question_key=f"key_{question_id}",
        question=question,
        target_xyz=(0.0, 0.0, 0.0),
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_visualization_selection_is_scene_complete_and_prediction_independent() -> None:
    records = [
        _record("scene_000002", "q_000009"),
        _record("scene_000001", "q_000010"),
        _record("scene_000002", "q_000001"),
        _record("scene_000001", "q_000003"),
    ]

    selected = evidence.select_visualization_indices(records)

    assert selected == [3, 2]
    assert [(records[index].scene_id, records[index].question_id) for index in selected] == [
        ("scene_000001", "q_000003"),
        ("scene_000002", "q_000001"),
    ]


def test_oracle_bearing_output_is_refused_below_runtime_tree() -> None:
    with pytest.raises(ValueError, match="must not be written to runtime data"):
        evidence._assert_evaluation_output(
            ROOT / "data_gemma4/runtime/should_not_exist.json"
        )


def test_synthetic_pointcloud_plot_is_valid_png(tmp_path: Path) -> None:
    output = tmp_path / "grounding.png"
    xyz = np.asarray(
        [[-1.0, -1.0, 0.0], [0.0, 0.0, 0.5], [1.0, 1.0, 1.0]], dtype=np.float32
    )
    rgb = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    examples = [
        {
            "scene_id": "scene_000001",
            "question_id": "q_000001",
            "question": "Where is the continuous target?",
            "predicted_xyz_m": [0.1, -0.1, 0.5],
            "target_xyz_m": [0.0, 0.0, 0.5],
            "coordinate_error_m": 0.141421,
        }
    ]

    evidence.plot_pointcloud_examples(
        examples,
        {"scene_000001": (xyz, rgb)},
        output,
        maximum_points_per_scene=100,
    )

    with Image.open(output) as image:
        assert image.format == "PNG"
        assert image.width > 800
        assert image.height > 700


def test_sealed_report_authentication_rejects_modified_bytes(tmp_path: Path) -> None:
    source = ROOT / evidence.DEFAULT_SEALED_REPORT
    modified = tmp_path / "modified.json"
    modified.write_bytes(source.read_bytes() + b" ")

    with pytest.raises(ValueError, match="sealed report digest differs"):
        evidence.load_sealed_report(modified)


def test_generated_evidence_is_explicitly_evaluation_only() -> None:
    metrics_path = ROOT / evidence.DEFAULT_METRICS
    figure_path = ROOT / evidence.DEFAULT_FIGURE
    if not metrics_path.is_file() or not figure_path.is_file():
        pytest.skip("Local V78 point-cloud evidence has not been generated")
    report = json.loads(metrics_path.read_text(encoding="utf-8"))

    assert report["artifact"] == "v78_historical_held_pointcloud_evaluation_v1"
    assert report["status"] == "internal_historical_evaluation_only"
    assert report["scope"]["runtime_path_executed"] is False
    assert report["scope"]["runtime_artifacts_written"] == []
    assert report["scope"]["oracle_target_coordinates_loaded_by_evaluator"] is True
    assert report["scope"]["oracle_or_qa_loaded_by_primary_runtime"] is False
    assert report["scope"]["environmental_text_inputs_to_primary_runtime"] == []
    assert report["scope"]["full_gemma_model_loaded"] is False
    assert report["scope"]["post_seal_optimization_or_tuning"] is False
    assert report["sealed_metrics_reproduced_within_1e_7"] is True
    assert report["reproduced_metrics"]["count"] == 94
    assert report["visualization"]["selected_count"] == 6
    assert report["visualization"]["sha256"] == _sha256(figure_path)
    assert len(report["per_example"]) == 94
    assert all(
        source["semantic_features_loaded"] is False
        for source in report["sources"]["numeric_rgb_maps"].values()
    )
