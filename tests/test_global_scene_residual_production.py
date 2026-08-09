from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from semantic_3d_chat.scene_encoder.global_residual import (
    ZERO_SPATIAL_MEAN_CONTENT_GATE_V1,
    GlobalSceneResidual,
)
from semantic_3d_chat.training.train_adapter import (
    build_adapter_optimizer,
    declared_global_scene_residual_parameter_count,
    explicit_adamw_options,
    global_scene_residual_resume_metadata_mismatch,
    v18_stage_execution_metadata,
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


def test_explicit_adamw_options_are_complete_and_fail_closed() -> None:
    assert explicit_adamw_options({"training": {}}) == {}
    raw = {
        "name": "AdamW",
        "learning_rate": 1.0e-3,
        "betas": [0.9, 0.999],
        "epsilon": 1.0e-8,
        "weight_decay": 0.0,
        "foreach": False,
        "fused": False,
        "capturable": False,
        "maximize": False,
        "amsgrad": False,
        "gradient_clip_norm": 1.0,
        "accumulation_divisor": 12,
        "step_index": 1,
    }
    training = {
        "learning_rate": 1.0e-3,
        "weight_decay": 0.0,
        "gradient_clip_norm": 1.0,
        "gradient_accumulation": 12,
        "optimizer": raw,
    }
    assert explicit_adamw_options({"training": training}) == {
        "betas": (0.9, 0.999),
        "eps": 1.0e-8,
        "foreach": False,
        "fused": False,
        "capturable": False,
        "maximize": False,
        "amsgrad": False,
    }
    with pytest.raises(ValueError, match="keys mismatch"):
        explicit_adamw_options(
            {
                "training": {
                    **training,
                    "optimizer": {key: value for key, value in raw.items() if key != "fused"},
                }
            }
        )
    with pytest.raises(ValueError, match="cannot both"):
        explicit_adamw_options(
            {
                "training": {
                    **training,
                    "optimizer": {**raw, "foreach": True, "fused": True},
                }
            }
        )

    parameter = torch.nn.Parameter(torch.ones(2))
    optimizer, selected = build_adapter_optimizer(
        {"training": training},
        [parameter],
        None,
        None,
    )
    assert len(selected) == 1
    assert selected[0] is parameter
    group = optimizer.param_groups[0]
    for key, expected in {
        "lr": 1.0e-3,
        "weight_decay": 0.0,
        "betas": (0.9, 0.999),
        "eps": 1.0e-8,
        "foreach": False,
        "fused": False,
        "capturable": False,
        "maximize": False,
        "amsgrad": False,
    }.items():
        assert group[key] == expected


def test_v18_stage_execution_metadata_is_exact_and_optional() -> None:
    assert v18_stage_execution_metadata({}) is None
    stages = {
        "stage_1_exact_v14_restart_updates": 1,
        "stage_1_stop_required": True,
        "predicted_preflight_state_must_match_epoch_001": True,
        "stage_2_resume_from_epoch": 1,
        "stage_2_load_optimizer_state": True,
        "stage_2_load_history": True,
        "stage_2_target_total_optimizer_updates": 4,
    }

    observed = v18_stage_execution_metadata(
        {"v18_screen": {"execution_stages": stages}}
    )

    assert observed == {
        key: value
        for key, value in stages.items()
        if key != "predicted_preflight_state_must_match_epoch_001"
    }
    changed = {**stages, "stage_2_load_history": False}
    with pytest.raises(ValueError, match="staged-resume contract"):
        v18_stage_execution_metadata(
            {"v18_screen": {"execution_stages": changed}}
        )


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
