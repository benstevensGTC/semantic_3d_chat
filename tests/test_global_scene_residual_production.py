from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from semantic_3d_chat.scene_encoder.global_residual import (
    ZERO_SPATIAL_MEAN_CONTENT_GATE_V1,
    GlobalSceneResidual,
)
from semantic_3d_chat.training.train_adapter import (
    declared_global_scene_residual_parameter_count,
    global_scene_residual_resume_metadata_mismatch,
    validate_global_scene_residual_state,
)


def _residual() -> GlobalSceneResidual:
    return GlobalSceneResidual(
        scene_dim=8,
        latent_count=4,
        width=3,
        fourier_bands=2,
        initialization_seed=18101,
        architecture_version=ZERO_SPATIAL_MEAN_CONTENT_GATE_V1,
        gate_temperature=0.75,
    )


def test_declared_residual_parameter_count_is_optional_and_strict() -> None:
    assert declared_global_scene_residual_parameter_count({}) is None
    assert declared_global_scene_residual_parameter_count({"experiment": {}}) is None
    assert (
        declared_global_scene_residual_parameter_count(
            {"experiment": {"residual_parameter_count": 400_128}}
        )
        == 400_128
    )
    for value in (True, 0, -1, 4.5, "400128"):
        with pytest.raises(ValueError, match="positive integer"):
            declared_global_scene_residual_parameter_count(
                {"experiment": {"residual_parameter_count": value}}
            )
    with pytest.raises(TypeError, match="mapping"):
        declared_global_scene_residual_parameter_count({"experiment": []})


def test_training_state_validation_enforces_declared_count_and_all_state() -> None:
    module = _residual()

    audit = validate_global_scene_residual_state(
        module,
        expected_parameter_count=module.parameter_count,
        context="test",
    )

    assert audit["parameter_count"] == module.parameter_count
    with pytest.raises(ValueError, match="parameter-count mismatch.*expected=1"):
        validate_global_scene_residual_state(
            module,
            expected_parameter_count=1,
            context="test",
        )
    with torch.no_grad():
        module.content_gate_projection.weight.view(-1)[0] = float("nan")
    with pytest.raises(ValueError, match="nonfinite.*content_gate_projection.weight"):
        validate_global_scene_residual_state(
            module,
            expected_parameter_count=module.parameter_count,
            context="test",
        )


def test_resume_metadata_requires_exact_initial_hash_and_parameter_count() -> None:
    module = _residual()
    expected_hash = "a" * 64
    metadata = {
        "global_scene_residual_initial_state_sha256": expected_hash,
        "global_scene_residual_parameter_count": module.parameter_count,
    }

    assert (
        global_scene_residual_resume_metadata_mismatch(
            metadata,
            module,
            expected_initial_state_sha256=expected_hash,
        )
        is None
    )

    wrong = deepcopy(metadata)
    wrong["global_scene_residual_initial_state_sha256"] = "b" * 64
    wrong["global_scene_residual_parameter_count"] = module.parameter_count - 1
    mismatch = global_scene_residual_resume_metadata_mismatch(
        wrong,
        module,
        expected_initial_state_sha256=expected_hash,
    )
    assert mismatch is not None
    assert set(mismatch) == {
        "global_scene_residual_initial_state_sha256",
        "global_scene_residual_parameter_count",
    }
