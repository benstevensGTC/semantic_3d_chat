from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from semantic_3d_chat.evaluation.v69_pair_augmentation_preregistration import (
    V69_ARM_GRID,
)
from semantic_3d_chat.evaluation.v71_multiscale_preregistration import (
    V71_AUGMENTATION_ARM,
    build_v71_preregistration,
    implementation_source_hashes_v71,
    write_v71_preregistration,
)
from semantic_3d_chat.evaluation.v71_result_authentication import (
    V71_NUMERIC_SCREEN_SHA256,
    authenticate_v71_result,
)
from semantic_3d_chat.scene_encoder.question_control_v7 import (
    AlwaysOnTeacherBasisFullSceneQuestionControlV7,
)
from semantic_3d_chat.scene_encoder.question_control_v71 import (
    MultiscaleAlwaysOnTeacherBasisControlV71,
)
from semantic_3d_chat.training import train_question_control_v68 as v68
from semantic_3d_chat.training import train_question_control_v71 as v71


def _controller() -> MultiscaleAlwaysOnTeacherBasisControlV71:
    basis = torch.eye(4, 16)
    torch.manual_seed(7108)
    branch_8 = AlwaysOnTeacherBasisFullSceneQuestionControlV7(
        16,
        basis,
        expected_environment_latents=256,
        moment_count=8,
        interaction_dim=4,
        trunk_dim=8,
    )
    torch.manual_seed(7132)
    branch_32 = AlwaysOnTeacherBasisFullSceneQuestionControlV7(
        16,
        basis,
        expected_environment_latents=256,
        moment_count=32,
        interaction_dim=4,
        trunk_dim=8,
    )
    return MultiscaleAlwaysOnTeacherBasisControlV71(branch_8, branch_32)


def test_v71_signature_is_true_first_8_plus_first_32_over_complete_scene() -> None:
    control = _controller()
    torch.manual_seed(71)
    prefix = torch.randn(1, 258, 16)
    signature = control.encode_scene(prefix)

    assert signature.shape == (1, 40, 16)
    assert torch.equal(signature[:, :8], signature[:, 8:16])
    changed = prefix.clone()
    changed[:, -2, 0] += 2.0
    modified = control.encode_scene(changed)
    assert not torch.equal(signature[:, 0], modified[:, 0])
    assert not torch.equal(signature[:, 8], modified[:, 8])


def test_v71_branches_are_independent_and_fusion_is_bounded() -> None:
    control = _controller()
    assert (
        control.scene_projection["branch_8"].weight.data_ptr()
        != control.scene_projection["branch_32"].weight.data_ptr()
    )
    assert (
        control.question_projection["branch_8"].weight.data_ptr()
        != control.question_projection["branch_32"].weight.data_ptr()
    )
    assert control.fusion_weight().item() == pytest.approx(0.5)
    with torch.no_grad():
        control.coefficient_output.fusion_logit.fill_(100.0)
    assert control.fusion_weight().item() == pytest.approx(0.9)
    with torch.no_grad():
        control.coefficient_output.fusion_logit.fill_(-100.0)
    assert control.fusion_weight().item() == pytest.approx(0.1)


def test_v71_forward_is_continuous_bounded_and_uses_both_branches() -> None:
    control = _controller().eval()
    prefix = torch.randn(1, 258, 16)
    question = torch.randn(1, 3, 16)
    signature = control.encode_scene(prefix)
    output = control.forward_from_signature(signature, question)

    assert output.control_tokens.shape == (1, 4, 16)
    assert output.control_rms.max().item() <= control.maximum_control_rms + 1e-6
    audit = control.audit()
    assert audit.branch_moment_counts == (8, 32)
    assert audit.every_environment_latent_influences_both_branches is True
    assert audit.question_dependent_scene_retrieval is False

    baseline = output.control_tokens.detach().clone()
    with torch.no_grad():
        control.scene_projection["branch_32"].weight.zero_()
    changed = control.forward_from_signature(signature, question).control_tokens
    assert not torch.equal(baseline, changed)


def test_v71_exact_v69_first_arm_and_joint_optimizer_scope() -> None:
    control = _controller()
    selected = v68._regularized_parameters(control, optimizer_scope="all_value")
    names = {name for name, _parameter in selected}

    assert V71_AUGMENTATION_ARM == dict(V69_ARM_GRID[0])
    assert "coefficient_output.fusion_logit" in names
    assert any(name.startswith("scene_projection.branch_8") for name in names)
    assert any(name.startswith("scene_projection.branch_32") for name in names)
    assert any(name.startswith("control_trunk.branch_8") for name in names)
    assert any(name.startswith("control_trunk.branch_32") for name in names)
    assert not any(name.startswith("question_norm") for name in names)


def test_v71_preregistration_is_create_once_and_source_locked(tmp_path: Path) -> None:
    payload = build_v71_preregistration()
    assert payload["status"] == "locked_before_v71_numeric_screen"
    assert payload["architecture"]["branch_moment_counts"] == [8, 32]
    assert payload["training_protocol"][
        "augmentation_arm_is_exact_v69_balanced_extrapolation_010"
    ] is True
    assert payload["numeric_screen"]["wall_time_budget_seconds"] == 1200
    assert payload["implementation_source_hashes"] == implementation_source_hashes_v71()

    destination = tmp_path / "v71.json"
    path, digest = write_v71_preregistration(destination)
    assert path == destination.resolve()
    assert len(digest) == 64
    assert json.loads(destination.read_text(encoding="utf-8")) == payload
    with pytest.raises(FileExistsError, match="already exists"):
        write_v71_preregistration(destination)


def test_v71_cli_has_no_full_generation_or_checkpoint_publication_mode() -> None:
    destinations = {action.dest for action in v71._parser()._actions}
    assert "mode" not in destinations
    assert "screen_authorization" not in destinations
    assert "output_checkpoint" in destinations
    assert v71._WALL_TIME_BUDGET_SECONDS == 1200
    launcher = Path("scripts/run_gemma4_v71_multiscale.sh").read_text(
        encoding="utf-8"
    )
    assert "numeric screen only" in launcher
    assert "balanced_extrapolation_010" in launcher
    assert "--screen-authorization" not in launcher
    assert "1200 seconds" in launcher


def test_v71_sealed_failure_authenticates_and_remains_unpublished() -> None:
    result = authenticate_v71_result()

    assert result["measurement_authenticated"] is True
    assert result["status"] == "authenticated_numeric_screen_failed_no_publication"
    assert result["numeric_screen_sha256"] == V71_NUMERIC_SCREEN_SHA256
    assert result["passed"] is False
    assert result["checkpoint_absent"] is True
    assert result["gemma_generation_used"] is False
    assert result["full_behavioral_run_executed"] is False
    assert result["atlas_compilation_executed"] is False
    assert result["fold_count"] == 12
    assert result["metrics"]["complete_class_units"] == 17
    assert result["metrics"]["prediction_change_units"] == 17
    assert result["metrics"]["positive_own_over_opposite_sides"] == 52
    assert result["failed_checks"] == [
        "held_prediction_change_units",
        "positive_own_over_opposite_sides",
    ]
    assert all(result["authentication_checks"].values())


def test_v71_authentication_fails_closed_on_report_tamper(tmp_path: Path) -> None:
    tampered = tmp_path / "screen.json"
    payload = json.loads(
        Path("reports/gemma4/metrics/v71_multiscale_numeric_screen.json").read_text(
            encoding="utf-8"
        )
    )
    payload["result"]["metrics"]["prediction_change_units"] = 20
    tampered.write_text(json.dumps(payload), encoding="utf-8")

    result = authenticate_v71_result(screen_path=tampered)
    assert result["measurement_authenticated"] is False
    assert "digest differs" in result["measurement_evidence_error"]


def test_v71_authentication_fails_closed_if_checkpoint_exists(tmp_path: Path) -> None:
    checkpoint = tmp_path / "forbidden_checkpoint"
    checkpoint.mkdir()
    result = authenticate_v71_result(checkpoint_path=checkpoint)

    assert result["measurement_authenticated"] is False
    assert "checkpoint exists" in result["measurement_evidence_error"]
