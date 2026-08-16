from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "plot_navigation_journal.py"
JOURNAL = (
    ROOT
    / "reports/gemma4/predictions/llm_navigation_scene_000001_learned_v3.json"
)


def _module():
    spec = importlib.util.spec_from_file_location("plot_navigation_journal", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v3_navigation_trajectory_is_numeric_and_renderable(tmp_path: Path) -> None:
    module = _module()
    journal = json.loads(JOURNAL.read_text(encoding="utf-8"))
    trajectories = module.numeric_trajectories(journal)

    assert len(trajectories) == 6
    assert {row["family"] for row in trajectories} == {
        "approach",
        "face",
        "left_right",
        "obstacle",
        "stop",
        "update_after_scan",
    }
    assert all(
        np.isfinite(np.asarray(row["positions_m"], dtype=np.float64)).all()
        for row in trajectories
    )

    image = tmp_path / "trajectory.png"
    output = tmp_path / "trajectory.json"
    artifact = module.render(JOURNAL, image, output)
    assert artifact["source_journal_root_sha256"] == journal["journal_sha256"]
    assert image.stat().st_size > 10_000
    assert json.loads(output.read_text(encoding="utf-8"))["trajectories"] == trajectories
