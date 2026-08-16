from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from semantic_3d_chat.evaluation.v68_regularized_pair_preregistration import (
    V68_ARM_GRID,
)
from semantic_3d_chat.evaluation.v69_pair_augmentation_preregistration import (
    V69_ARM_GRID,
    V69_FOUNDATION_ARM,
    build_v69_preregistration,
    implementation_source_hashes_v69,
    write_v69_preregistration,
)

_LOCKED_SHA256 = "5cd567a129e083600b8913aa0438c0a8115aba83bd70c24c40ce5475a4bcfb3e"


def test_v69_preregistration_is_failure_driven_and_keeps_every_gate() -> None:
    payload = build_v69_preregistration()
    predecessor = payload["failed_predecessor"]

    assert payload["status"] == "locked_before_v69_training_screen_or_generation"
    assert predecessor["held_complete_units_observed"] == 14
    assert predecessor["held_complete_units_required"] == 15
    assert predecessor["held_prediction_changes_observed"] == 17
    assert predecessor["held_prediction_changes_required"] == 20
    assert predecessor["held_positive_margins_observed"] == 50
    assert predecessor["held_positive_margins_required"] == 53
    assert predecessor["predecessor_artifact_bytes_modified"] is False
    assert payload["numeric_screen"]["thresholds_unchanged_from_v68_v67"] is True
    assert payload["behavioral_gates"]["unchanged_from_v68_v67_and_v66b"] is True
    assert payload["arm_selection"]["best_metric_cherry_picking"] is False
    assert payload["arm_selection"]["greedy_generation_during_selection"] is False
    assert payload["implementation_source_hashes"] == implementation_source_hashes_v69()
    assert all(
        payload["scope"][field] is False
        for field in (
            "validation_inputs_used",
            "scorer_inputs_used",
            "oracle_loaded",
            "fresh_development_loaded",
            "internal_validation_loaded",
            "deferred_final_loaded",
        )
    )


def test_v69_grid_is_small_ordered_and_uses_exact_strong_v68_foundation() -> None:
    assert V69_FOUNDATION_ARM == dict(V68_ARM_GRID[2])
    assert [arm["arm_id"] for arm in V69_ARM_GRID] == [
        "balanced_extrapolation_010",
        "balanced_extrapolation_010_question_mix_010",
        "balanced_extrapolation_020_question_mix_015",
    ]
    assert len(V69_ARM_GRID) == 3
    assert [float(arm["signature_expansion"]) for arm in V69_ARM_GRID] == [
        0.10,
        0.10,
        0.20,
    ]
    assert [float(arm["question_mix_weight"]) for arm in V69_ARM_GRID] == [
        0.0,
        0.10,
        0.15,
    ]


def test_v69_preregistration_is_create_once(tmp_path: Path) -> None:
    destination = tmp_path / "v69.json"
    path, digest = write_v69_preregistration(destination)

    assert path == destination
    assert digest == hashlib.sha256(destination.read_bytes()).hexdigest()
    assert json.loads(destination.read_text(encoding="utf-8")) == build_v69_preregistration()
    with pytest.raises(FileExistsError):
        write_v69_preregistration(destination)

    locked = Path("reports/gemma4/metrics/v69_pair_augmentation_preregistration.json")
    assert hashlib.sha256(locked.read_bytes()).hexdigest() == _LOCKED_SHA256
    assert json.loads(locked.read_text(encoding="utf-8")) == build_v69_preregistration()


def test_v69_preserves_locked_v68_bytes() -> None:
    payload = build_v69_preregistration()
    for relative, expected in payload["preserved_v68_path_hashes"].items():
        assert hashlib.sha256(Path(relative).read_bytes()).hexdigest() == expected
