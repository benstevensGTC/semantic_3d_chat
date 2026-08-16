from __future__ import annotations

import pytest
import torch

from semantic_3d_chat.scene_encoder.question_control import FullSceneQuestionControl


def test_full_scene_question_control_shapes_and_global_coverage() -> None:
    torch.manual_seed(7)
    module = FullSceneQuestionControl(
        32,
        attention_dim=16,
        control_tokens=4,
        uniform_floor=0.1,
    )
    scene = torch.randn(2, 258, 32)
    question = torch.randn(2, 9, 32)

    output = module(scene, question)
    audit = module.audit()

    assert output.shape == (2, 4, 32)
    assert torch.isfinite(output).all()
    assert audit.scene_token_count == 258
    assert audit.control_token_count == 4
    assert audit.every_scene_token_influenced_output is True
    assert audit.minimum_attention_weight >= (0.1 / 258) * (1.0 - 1e-5)


def test_question_changes_control_without_changing_scene() -> None:
    torch.manual_seed(11)
    module = FullSceneQuestionControl(24, attention_dim=12, control_tokens=2)
    scene = torch.randn(1, 10, 24)
    first = module(scene, torch.randn(1, 4, 24))
    second = module(scene, torch.randn(1, 4, 24))

    assert not torch.equal(first, second)
    assert torch.equal(scene, scene.clone())


def test_question_control_validates_masks_and_finiteness() -> None:
    module = FullSceneQuestionControl(8, attention_dim=4)
    scene = torch.randn(1, 5, 8)
    question = torch.randn(1, 3, 8)
    with pytest.raises(ValueError, match="at least one unmasked"):
        module(scene, question, torch.zeros(1, 3))
    invalid = scene.clone()
    invalid[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        module(invalid, question)
    invalid_mask = torch.ones(1, 3)
    invalid_mask[0, 0] = float("nan")
    with pytest.raises(ValueError, match="mask must be finite"):
        module(scene, question, invalid_mask)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"hidden_size": True}, "positive integers"),
        ({"hidden_size": 8, "attention_dim": 0}, "positive integers"),
        ({"hidden_size": 8, "uniform_floor": float("nan")}, "uniform_floor"),
        ({"hidden_size": 8, "output_scale": True}, "output_scale"),
    ],
)
def test_question_control_rejects_ambiguous_constructor_values(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        FullSceneQuestionControl(**kwargs)
