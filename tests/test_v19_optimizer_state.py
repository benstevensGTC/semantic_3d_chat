from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from semantic_3d_chat.evaluation.v19_optimizer_state import (
    V19AdamWStateViolation,
    canonical_v19_adamw_state,
    validate_v19_adamw_state_manifest,
)


def _contract() -> dict[str, object]:
    return {
        "name": "AdamW",
        "learning_rate": 1.0e-4,
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


def _state() -> dict[str, object]:
    parameter = torch.nn.Parameter(torch.zeros(1536, 128))
    optimizer = torch.optim.AdamW(
        [
            {
                "name": "signed_x_output_projection",
                "params": [parameter],
                "lr": 1.0e-4,
                "weight_decay": 0.0,
            }
        ],
        betas=(0.9, 0.999),
        eps=1.0e-8,
        foreach=False,
        fused=False,
        capturable=False,
        maximize=False,
        amsgrad=False,
    )
    parameter.grad = torch.linspace(-1.0, 1.0, parameter.numel()).reshape_as(parameter)
    optimizer.step()
    return optimizer.state_dict()


def test_v19_canonical_optimizer_state_round_trip_is_exact() -> None:
    manifest, digest = canonical_v19_adamw_state(_state(), _contract())

    assert manifest["state_parameter_count"] == 1
    assert manifest["parameter_order"] == [
        {
            "parameter_id": 0,
            "name": "output_projection.weight",
            "shape": [1536, 128],
            "dtype": "float32",
        }
    ]
    assert manifest["param_groups"][0]["name"] == "signed_x_output_projection"
    assert manifest["states"][0]["state"]["step"]["value"] == 1.0
    assert len(digest) == 64
    assert validate_v19_adamw_state_manifest(manifest, _contract()) == digest


@pytest.mark.parametrize(
    ("mutator", "match"),
    [
        (
            lambda state: state["param_groups"][0].update(name="wrong"),
            "group name",
        ),
        (
            lambda state: state["state"][0].update(step=torch.tensor(2.0)),
            "step",
        ),
        (
            lambda state: state["state"][0].update(exp_avg=state["state"][0]["exp_avg"][:-1]),
            "shape mismatch",
        ),
        (
            lambda state: state["state"][0].update(
                exp_avg=state["state"][0]["exp_avg"].to(torch.float64)
            ),
            "dtype mismatch",
        ),
    ],
)
def test_v19_optimizer_state_rejects_tamper(mutator, match: str) -> None:
    state = deepcopy(_state())
    mutator(state)

    with pytest.raises(V19AdamWStateViolation, match=match):
        canonical_v19_adamw_state(state, _contract())


def test_v19_optimizer_manifest_rejects_option_and_digest_tamper() -> None:
    manifest, _ = canonical_v19_adamw_state(_state(), _contract())
    wrong_group = deepcopy(manifest)
    wrong_group["param_groups"][0]["lr"] = 3.0e-4
    with pytest.raises(V19AdamWStateViolation, match="AdamW lr"):
        validate_v19_adamw_state_manifest(wrong_group, _contract())

    wrong_digest = deepcopy(manifest)
    wrong_digest["states"][0]["state"]["exp_avg"]["sha256"] = "bad"
    with pytest.raises(V19AdamWStateViolation, match="digest"):
        validate_v19_adamw_state_manifest(wrong_digest, _contract())
