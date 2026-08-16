from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from semantic_3d_chat.evaluation.v68_regularized_pair_preregistration import (
    V68_ARM_GRID,
    V68_COMMON_HYPERPARAMETERS,
)
from semantic_3d_chat.evaluation.v70_low_frequency_moments_preregistration import (
    V70_ARM,
    V70_COMMON_HYPERPARAMETERS,
    build_v70_preregistration,
    implementation_source_hashes_v70,
    write_v70_preregistration,
)
from semantic_3d_chat.scene_encoder.question_control_v7 import (
    AlwaysOnTeacherBasisFullSceneQuestionControlV7,
)
from semantic_3d_chat.training import train_question_control_v70 as v70


def test_v70_is_one_variable_failure_driven_ablation() -> None:
    payload = build_v70_preregistration()

    assert payload["status"] == "locked_before_v70_numeric_screen"
    assert payload["failed_predecessor"]["sha256"] == (
        "6f6a6af8ab0c254bd8ea1704593770c8445aebaad02bbb55478b94f61103e2a8"
    )
    assert payload["failed_predecessor"]["training_identity_sha256"] == (
        "0d16b6cae0ad5984860f4e94aa0cd0dc029d17b187d582dcc8c2cec8e3094a9e"
    )
    assert V70_ARM == dict(V68_ARM_GRID[2])
    assert V70_COMMON_HYPERPARAMETERS == {
        **V68_COMMON_HYPERPARAMETERS,
        "moment_count": 32,
    }
    assert payload["controlled_ablation"]["only_preregistered_variable_changed"] == (
        "moment_count"
    )
    assert payload["controlled_ablation"]["v69_augmentation_used"] is False
    assert payload["controlled_ablation"]["question_mixing_used"] is False
    assert payload["numeric_screen"]["wall_time_budget_seconds"] == 1200
    assert payload["numeric_screen"]["thresholds_unchanged_from_v69_v68_v67"] is True
    assert payload["implementation_source_hashes"] == implementation_source_hashes_v70()


def test_v70_preregistration_is_create_once(tmp_path: Path) -> None:
    destination = tmp_path / "v70.json"
    path, digest = write_v70_preregistration(destination)

    assert path == destination.resolve()
    assert len(digest) == 64
    assert json.loads(destination.read_text(encoding="utf-8")) == (
        build_v70_preregistration()
    )
    with pytest.raises(FileExistsError, match="already exists"):
        write_v70_preregistration(destination)


def test_v70_fit_lock_changes_only_moment_count() -> None:
    source = v70._preflight_args(SimpleNamespace(marker=70))
    candidate = v70._fit_args(SimpleNamespace(marker=70))

    assert source.moment_count == 8
    assert candidate.moment_count == 32
    source_values = vars(source).copy()
    candidate_values = vars(candidate).copy()
    source_values.pop("moment_count")
    candidate_values.pop("moment_count")
    assert candidate_values == source_values
    assert candidate.v68_arm == dict(V68_ARM_GRID[2])


def test_v70_32_moment_signature_uses_complete_scene() -> None:
    basis = torch.eye(4, 16)
    control = AlwaysOnTeacherBasisFullSceneQuestionControlV7(
        16,
        basis,
        expected_environment_latents=256,
        moment_count=32,
        interaction_dim=4,
        trunk_dim=8,
    )
    torch.manual_seed(70)
    prefix = torch.randn(1, 258, 16)
    baseline = control.encode_scene(prefix)
    changed = prefix.clone()
    changed[:, -2, 0] += 2.0
    modified = control.encode_scene(changed)

    assert baseline.shape == (1, 32, 16)
    assert not torch.equal(baseline[:, 0], modified[:, 0])


def test_v70_cli_and_launcher_have_no_full_or_generation_mode() -> None:
    destinations = {action.dest for action in v70._parser()._actions}
    assert "mode" not in destinations
    assert "screen_authorization" not in destinations
    assert "training_report" in destinations
    assert "output_checkpoint" in destinations

    launcher = Path("scripts/run_gemma4_v70_low_frequency_moments.sh").read_text(
        encoding="utf-8"
    )
    assert "numeric screen only" in launcher
    assert "--screen-authorization" not in launcher
    assert " full" not in launcher
    assert "1200 seconds" in launcher
    assert v70._WALL_TIME_BUDGET_SECONDS == 1200


def test_v70_preserves_exact_v68_sources() -> None:
    payload = build_v70_preregistration()
    for relative, expected in payload["preserved_v68_path_hashes"].items():
        assert v70._sha256_file(Path(relative).resolve()) == expected
