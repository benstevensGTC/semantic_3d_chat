"""The V15 summary must be derived from artifacts, never asserted by hand.

The V14 release summary was hand-authored, so nothing structurally prevented it
from disagreeing with the files it cited.  The V15 builder reads every number
back out of a produced artifact and hashes every file it references, and it
reports missing stages instead of quietly dropping them.
"""

from __future__ import annotations

import hashlib
import json
import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BUILDER = runpy.run_path(str(ROOT / "scripts" / "build_gemma_waypoint_v15_summary.py"))


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def _controls(primary_accuracy: float) -> dict:
    def condition(accuracy: float) -> dict:
        return {
            "sample_count": 24,
            "action_accuracy": accuracy,
            "action_macro_recall": accuracy,
            "action_recall": {"move_to": accuracy, "face": accuracy, "stop": accuracy},
            "stop_recall": accuracy,
            "stop_precision": accuracy,
            "waypoint_error_m_mean": 0.05,
            "heading_error_degrees_mean": 3.0,
        }

    conditions = {
        "primary": condition(primary_accuracy),
        "wrong_scene_prefix": condition(0.2),
        "zero_scene_prefix": condition(0.1),
        "shuffled_scene_prefix": condition(0.3),
        "zero_history": condition(0.4),
    }
    return {
        "evaluated_conditions": list(conditions),
        "conditions": conditions,
        "accuracy_drop_from_primary": {
            name: primary_accuracy - value["action_accuracy"]
            for name, value in conditions.items()
            if name != "primary"
        },
        "output_change_from_primary": {
            name: {"action_change_fraction": 0.5}
            for name in conditions
            if name != "primary"
        },
    }


@pytest.fixture
def artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    # runpy.run_path returns a *copy* of the executed namespace, so the builder
    # functions still close over their own module globals. Patch those.
    monkeypatch.setitem(BUILDER["build_summary"].__globals__, "ROOT", tmp_path)
    checkpoint = Path("checkpoints/v15")
    (tmp_path / checkpoint).mkdir(parents=True)
    (tmp_path / checkpoint / "policy.safetensors").write_bytes(b"weights")
    _write(
        tmp_path / checkpoint / "runtime_metadata.json",
        {
            "weights_sha256": "a" * 64,
            "gemma_runtime_binding_sha256": "b" * 64,
            "model_id": "google/gemma-4-E2B-it",
            "model_revision": "3e22461f65e89153144f8adb70e3b8c2cc9845a7",
            "schema": "semantic_3d_chat.gemma_waypoint_checkpoint.v4",
            "history_dim": 16,
            "history_parameterization": "selected_action_parameters_goal_progress_v2",
            "scene_token_count": 258,
            "robot_token_count": 4,
            "actual_gemma_causal_forward": True,
            "complete_scene_prefix_required": True,
            "every_scene_token_processed": True,
            "model_selects_every_waypoint_and_heading": True,
            "deterministic_route_planner_allowed_at_runtime": False,
            "oracle_inputs_at_runtime": False,
            "environmental_text_inputs": [],
            "training_scene_count": 27,
            "validation_scene_count": 8,
            "training_sample_count": 5400,
            "validation_sample_count": 1400,
        },
    )
    _write(
        tmp_path / "reports/training.json",
        {
            "dataset_sha256": "c" * 64,
            "best_epoch": 42,
            "epochs_completed": 160,
            "elapsed_seconds": 12.5,
            "gemma_hidden_cache": {"training_rows": 5400, "validation_rows": 1400},
            "training_metrics": {"action_accuracy": 0.99, "stop_recall": 0.98},
            "controls": _controls(0.9),
        },
    )
    _write(
        tmp_path / "data/general/manifest.json",
        {
            "train_scene_ids": ["scene_000001"],
            "validation_scene_ids": ["scene_000031"],
            "train_scene_count": 27,
            "validation_scene_count": 8,
            "sample_count": 67331,
            "episode_count": 3592,
            "action_sample_counts": {"FACE": 1, "MOVE_TO": 2, "STOP": 3},
            "unroutable_lap_start_count": 6,
        },
    )
    _write(
        tmp_path / "data/sealed/manifest.json",
        {
            "validation_scene_ids": ["scene_000051"],
            "validation_scene_count": 6,
        },
    )
    return {
        "checkpoint": checkpoint,
        "training_metrics": Path("reports/training.json"),
        "general_dataset": Path("data/general"),
        "sealed_dataset": Path("data/sealed"),
        "sealed_controls": Path("reports/sealed_controls.json"),
        "heldout_score": Path("reports/heldout_score.json"),
        "_root": tmp_path,
    }


def test_absent_stages_are_reported_not_dropped(artifacts: dict[str, Path]) -> None:
    root = artifacts.pop("_root")
    summary = BUILDER["build_summary"](**artifacts)
    assert summary["complete"] is False
    assert summary["stages_present"]["sealed_controls"] is False
    assert summary["stages_present"]["heldout_closed_loop_score"] is False
    assert summary["stages_present"]["training_report"] is True
    # An absent stage still appears in the evidence block with its path.
    assert summary["evidence"]["heldout_closed_loop_score"]["present"] is False
    assert "sha256" not in summary["evidence"]["heldout_closed_loop_score"]
    assert root.is_dir()


def test_every_present_evidence_file_is_hashed(artifacts: dict[str, Path]) -> None:
    root = artifacts.pop("_root")
    _write(root / "reports/sealed_controls.json", _controls(0.8))
    _write(
        root / "reports/heldout_score.json",
        {
            "goal_count": 30,
            "passed_count": 21,
            "pass_rate": 0.7,
            "model_selected_terminal_stop_rate": 0.9,
            "per_metric": {"face_yaw": {"passed": 8, "total": 12, "pass_rate": 2 / 3}},
            "scene_ids": ["scene_000051"],
            "rollout_process_read_oracle": False,
        },
    )
    summary = BUILDER["build_summary"](**artifacts)
    assert summary["complete"] is True
    for name, entry in summary["evidence"].items():
        assert entry["present"] is True, name
        digest = hashlib.sha256((root / entry["path"]).read_bytes()).hexdigest()
        assert entry["sha256"] == digest, name


def test_summary_separates_development_from_sealed_measurements(
    artifacts: dict[str, Path],
) -> None:
    root = artifacts.pop("_root")
    _write(root / "reports/sealed_controls.json", _controls(0.8))
    summary = BUILDER["build_summary"](**artifacts)

    # Checkpoint selection saw the development rooms; the sealed rooms are a
    # separate, later measurement. Conflating them is the failure mode this
    # structure exists to prevent.
    assert summary["development_controls"]["primary"]["action_accuracy"] == 0.9
    assert summary["sealed_controls"]["primary"]["action_accuracy"] == 0.8
    assert summary["rooms"]["development_scene_ids"] == ["scene_000031"]
    assert summary["rooms"]["sealed_scene_ids"] == ["scene_000051"]
    assert not set(summary["rooms"]["development_scene_ids"]) & set(
        summary["rooms"]["sealed_scene_ids"]
    )
    assert "shuffled_scene_prefix" in summary["sealed_controls"][
        "accuracy_drop_from_primary"
    ]


def test_runtime_contract_is_copied_from_the_checkpoint(
    artifacts: dict[str, Path],
) -> None:
    artifacts.pop("_root")
    summary = BUILDER["build_summary"](**artifacts)
    contract = summary["runtime_contract"]
    assert contract["checkpoint_schema"] == "semantic_3d_chat.gemma_waypoint_checkpoint.v4"
    assert contract["history_dim"] == 16
    assert contract["scene_token_count"] == 258
    assert contract["oracle_inputs_at_runtime"] is False
    assert contract["deterministic_route_planner_allowed_at_runtime"] is False
    assert summary["navigation_checkpoint_sha256"] == "a" * 64
