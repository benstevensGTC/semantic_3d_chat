from __future__ import annotations

import json
from pathlib import Path

import pytest

from semantic_3d_chat.evaluation.fixed_prefix_ple_v54_v2_preregistration import (
    V1_PREREGISTRATION_SHA256,
    V1_SMOKE_SHA256,
    authenticate_v1_failure,
    build_preregistration,
    v2_implementation_hashes,
    write_preregistration,
)
from semantic_3d_chat.training.train_fixed_prefix_ple_v54_v2 import (
    RETENTION_SELF_KL_ABSOLUTE_TOLERANCE,
)


def test_v2_authenticates_exact_v1_failure_and_changes_one_tolerance() -> None:
    failure = authenticate_v1_failure()
    contract = build_preregistration()

    assert failure["passed"] is False
    assert failure["initial_retention_kl"] == 1.7583897715667263e-06
    assert failure["gradient_l2"] == 0.2632919251918793
    assert contract["v1_failure"]["preregistration_sha256"] == V1_PREREGISTRATION_SHA256
    assert contract["v1_failure"]["smoke_sha256"] == V1_SMOKE_SHA256
    assert contract["only_change"] == {
        "field": "gradient_smoke.retention_self_kl_absolute_tolerance",
        "v1": 1e-06,
        "v2": 1e-05,
        "reason": "observed_finite_mps_repeat_forward_noise_1.7583897715667263e-6",
    }
    assert RETENTION_SELF_KL_ABSOLUTE_TOLERANCE == 1e-05


def test_v2_preserves_every_training_objective_gate_and_runtime_contract() -> None:
    unchanged = build_preregistration()["unchanged_v1_contract"]

    assert unchanged["trainable_surface"]["parameter_count"] == 41_984
    assert unchanged["objective"]["same_question_wrong_prefix_hinge_weight"] == 1.0
    assert unchanged["optimization"]["maximum_updates"] == 40
    assert unchanged["selection"]["greedy_exact_accuracy_delta_minimum"] == 0.02
    assert unchanged["selection"]["all_gates_required"] is True
    assert unchanged["runtime_contract"]["environmental_text_inputs"] == []
    assert unchanged["runtime_contract"]["question_dependent_retrieval"] is False
    assert unchanged["publication"]["failed_run_publishes_no_checkpoint"] is True


def test_v2_has_unique_paths_and_no_atlas_or_question_controller_dependency() -> None:
    contract = build_preregistration()

    assert contract["output_paths"] == {
        "smoke": "reports/gemma4/metrics/gemma4_v54_fixed_prefix_ple_reader_v2_smoke.json",
        "result": "reports/gemma4/metrics/gemma4_v54_fixed_prefix_ple_reader_v2_result.json",
        "checkpoint": "data_gemma4/checkpoints/gemma4_v54_fixed_prefix_ple_reader_v2",
    }
    assert contract["unchanged_v1_contract"]["independence"][
        "depends_on_failed_atlas"
    ] is False
    assert contract["unchanged_v1_contract"]["independence"][
        "depends_on_failed_question_controllers"
    ] is False
    assert contract["v2_implementation_source_hashes"] == v2_implementation_hashes()


def test_v2_preregistration_is_create_once(tmp_path: Path) -> None:
    destination = tmp_path / "v2.json"
    path, digest = write_preregistration(destination)

    assert path == destination.resolve()
    assert len(digest) == 64
    assert json.loads(destination.read_text(encoding="utf-8")) == build_preregistration()
    with pytest.raises(FileExistsError, match="exists"):
        write_preregistration(destination)


def test_v2_launcher_and_wrapper_do_not_redefine_training_hyperparameters() -> None:
    wrapper = Path(
        "src/semantic_3d_chat/training/train_fixed_prefix_ple_v54_v2.py"
    ).read_text(encoding="utf-8")
    launcher = Path("scripts/run_gemma4_v54_fixed_prefix_ple_reader_v2.sh").read_text(
        encoding="utf-8"
    )

    assert "RETENTION_SELF_KL_ABSOLUTE_TOLERANCE = 1e-05" in wrapper
    assert "learning_rate" not in wrapper
    assert "maximum_updates" not in wrapper
    assert "same_question_wrong_prefix" not in wrapper
    assert "preregister|preflight|smoke|train|authenticate" in launcher
