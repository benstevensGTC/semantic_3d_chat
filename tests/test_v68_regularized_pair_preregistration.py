from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from semantic_3d_chat.evaluation.v68_regularized_pair_preregistration import (
    V68_ARM_GRID,
    build_v68_preregistration,
    implementation_source_hashes_v68,
    write_v68_preregistration,
)

_LOCKED_SHA256 = "6642b16b38e169df0059b2ccfb6aba0f8b1315052f3aad0e2871b30eeda6811f"


def test_v68_preregistration_is_failure_driven_and_keeps_gates() -> None:
    payload = build_v68_preregistration()

    assert payload["status"] == "locked_before_v68_training_screen_or_generation"
    assert payload["failed_predecessor"]["held_complete_units"] == 13
    assert payload["failed_predecessor"]["held_prediction_change_units"] == 14
    assert payload["failed_predecessor"]["positive_own_over_opposite_sides"] == 47
    assert payload["failed_predecessor"]["predecessor_artifact_bytes_modified"] is False
    assert payload["ordered_arm_grid"] == [dict(arm) for arm in V68_ARM_GRID]
    assert len(V68_ARM_GRID) == 3
    assert payload["arm_selection"]["rule"] == (
        "run_in_declared_order_and_select_first_all_gate_pass"
    )
    assert payload["arm_selection"]["best_metric_cherry_picking"] is False
    assert payload["arm_selection"]["greedy_generation_during_selection"] is False
    assert payload["numeric_screen"]["thresholds_unchanged_from_v67"] is True
    assert payload["behavioral_gates"]["unchanged_from_v67_and_v66b"] is True
    assert payload["implementation_source_hashes"] == (implementation_source_hashes_v68())
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


def test_v68_arm_grid_is_small_ordered_and_regularized() -> None:
    assert [arm["arm_id"] for arm in V68_ARM_GRID] == [
        "balanced_all_value_anchor",
        "interaction_only_anchor",
        "strong_all_value_anchor",
    ]
    assert {arm["optimizer_scope"] for arm in V68_ARM_GRID} == {
        "all_value",
        "interaction_only",
    }
    for arm in V68_ARM_GRID:
        assert float(arm["hard_negative_weight"]) > 0.0
        assert float(arm["hard_negative_margin"]) > 0.0
        assert float(arm["anchor_weight"]) > 0.0


def test_v68_preregistration_is_create_once(tmp_path: Path) -> None:
    destination = tmp_path / "v68.json"
    path, digest = write_v68_preregistration(destination)

    assert path == destination
    assert digest == hashlib.sha256(destination.read_bytes()).hexdigest()
    assert json.loads(destination.read_text(encoding="utf-8")) == (build_v68_preregistration())
    with pytest.raises(FileExistsError):
        write_v68_preregistration(destination)

    locked = Path("reports/gemma4/metrics/v68_regularized_pair_preregistration.json")
    assert hashlib.sha256(locked.read_bytes()).hexdigest() == _LOCKED_SHA256
    assert json.loads(locked.read_text(encoding="utf-8")) == (build_v68_preregistration())
