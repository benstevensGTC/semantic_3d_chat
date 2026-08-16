from __future__ import annotations

import pytest

from semantic_3d_chat.robot import practical_rover
from semantic_3d_chat.robot.gemma_waypoint_runtime import (
    CHECKPOINT_SCHEMA_V1,
    CHECKPOINT_SCHEMA_V2,
    HISTORY_PARAMETERIZATION_V1,
    HISTORY_PARAMETERIZATION_V2,
)
from semantic_3d_chat.robot.waypoint_history import (
    HISTORY_FEATURE_DIM_V1,
    HISTORY_FEATURE_DIM_V2,
)


@pytest.mark.parametrize(
    ("schema", "history_dim", "parameterization"),
    [
        (
            CHECKPOINT_SCHEMA_V1,
            HISTORY_FEATURE_DIM_V1,
            HISTORY_PARAMETERIZATION_V1,
        ),
        (
            CHECKPOINT_SCHEMA_V2,
            HISTORY_FEATURE_DIM_V2,
            HISTORY_PARAMETERIZATION_V2,
        ),
    ],
)
def test_practical_rover_accepts_each_exact_versioned_history_contract(
    schema: str,
    history_dim: int,
    parameterization: str,
) -> None:
    metadata = {
        "schema": schema,
        "history_dim": history_dim,
        "history_parameterization": parameterization,
    }

    assert practical_rover._validate_gemma_waypoint_history_contract(metadata) == (
        schema,
        history_dim,
        parameterization,
    )


@pytest.mark.parametrize(
    ("schema", "history_dim", "parameterization"),
    [
        (
            CHECKPOINT_SCHEMA_V1,
            HISTORY_FEATURE_DIM_V2,
            HISTORY_PARAMETERIZATION_V2,
        ),
        (
            CHECKPOINT_SCHEMA_V2,
            HISTORY_FEATURE_DIM_V1,
            HISTORY_PARAMETERIZATION_V1,
        ),
        (
            CHECKPOINT_SCHEMA_V1,
            HISTORY_FEATURE_DIM_V1,
            HISTORY_PARAMETERIZATION_V2,
        ),
        (
            CHECKPOINT_SCHEMA_V2,
            HISTORY_FEATURE_DIM_V2,
            HISTORY_PARAMETERIZATION_V1,
        ),
        (
            CHECKPOINT_SCHEMA_V1,
            HISTORY_FEATURE_DIM_V2,
            HISTORY_PARAMETERIZATION_V1,
        ),
        (
            CHECKPOINT_SCHEMA_V2,
            HISTORY_FEATURE_DIM_V1,
            HISTORY_PARAMETERIZATION_V2,
        ),
    ],
)
def test_practical_rover_rejects_crossed_history_contracts(
    schema: str,
    history_dim: int,
    parameterization: str,
) -> None:
    metadata = {
        "schema": schema,
        "history_dim": history_dim,
        "history_parameterization": parameterization,
    }

    with pytest.raises(ValueError, match="history contract is invalid"):
        practical_rover._validate_gemma_waypoint_history_contract(metadata)


@pytest.mark.parametrize(
    "metadata",
    [
        {},
        {"schema": CHECKPOINT_SCHEMA_V2},
        {
            "schema": CHECKPOINT_SCHEMA_V2,
            "history_dim": True,
            "history_parameterization": HISTORY_PARAMETERIZATION_V2,
        },
        {
            "schema": "semantic_3d_chat.gemma_waypoint_checkpoint.v5",
            "history_dim": HISTORY_FEATURE_DIM_V2,
            "history_parameterization": HISTORY_PARAMETERIZATION_V2,
        },
    ],
)
def test_practical_rover_rejects_missing_or_unknown_history_contracts(
    metadata: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="history contract is invalid"):
        practical_rover._validate_gemma_waypoint_history_contract(metadata)


def test_navigation_default_is_the_live_accepted_v14_checkpoint() -> None:
    assert practical_rover.DEFAULT_NAVIGATION_CHECKPOINT == (
        "data_gemma4/checkpoints/"
        "gemma_waypoint_policy_v2_operator_dagger_v14_runtime_aligned"
    )
