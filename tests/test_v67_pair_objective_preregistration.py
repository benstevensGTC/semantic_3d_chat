from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from semantic_3d_chat.evaluation.v67_pair_objective_preregistration import (
    V67_HYPERPARAMETERS,
    build_v67_preregistration,
    write_v67_preregistration,
)

_LOCKED_SHA256 = "a87ad59102c48da95390659839b76707c3d32af726034ab930fae5e01ba7ab8f"


def test_v67_preregistration_locks_failure_driven_pair_objective() -> None:
    payload = build_v67_preregistration()

    assert payload["status"] == "locked_before_v67_training_screen_or_generation"
    assert payload["failed_predecessor"]["held_supported_exact"] == 409
    assert payload["failed_predecessor"]["held_changed_side_exact"] == 37
    assert payload["failed_predecessor"]["held_complete_units"] == 5
    assert payload["failed_predecessor"]["held_prediction_change_units"] == 16
    assert payload["failed_predecessor"]["spatial_relation_exact"] == 49
    assert payload["diagnosed_mechanism"]["v66_pair_delta_weight"] == 0.0
    assert payload["fixed_hyperparameters"] == V67_HYPERPARAMETERS
    assert payload["pair_objective"][
        "own_teacher_closer_than_exact_paired_opposite_teacher"
    ] is True
    assert payload["pair_objective"]["changed_units_oversampled_atomically"] is True
    assert payload["pair_objective"]["runtime_answer_codebook"] is False
    assert payload["numeric_screen"][
        "required_before_any_greedy_generation"
    ] is True
    assert payload["behavioral_gates"]["unchanged_from_v66b"] is True
    assert payload["publication"][
        "screen_pass_authorization_required_for_full_run"
    ] is True
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


def test_v67_preregistration_is_create_once_and_matches_locked_file(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "v67.json"
    path, digest = write_v67_preregistration(destination)

    assert path == destination
    assert digest == hashlib.sha256(destination.read_bytes()).hexdigest()
    assert json.loads(destination.read_text(encoding="utf-8")) == (
        build_v67_preregistration()
    )
    with pytest.raises(FileExistsError):
        write_v67_preregistration(destination)

    locked = Path("reports/gemma4/metrics/v67_pair_objective_preregistration.json")
    assert hashlib.sha256(locked.read_bytes()).hexdigest() == _LOCKED_SHA256
    assert json.loads(locked.read_text(encoding="utf-8")) == (
        build_v67_preregistration()
    )
