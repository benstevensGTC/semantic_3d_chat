from __future__ import annotations

import pytest
import torch

from semantic_3d_chat.scene_encoder.perceiver import spatial_anchors
from semantic_3d_chat.training.losses import ordered_spatial_relation_contrastive_loss


def test_ordered_relation_soft_pools_every_fixed_anchor_and_aligns_direction() -> None:
    anchors = spatial_anchors(12)
    reference_index = int(anchors[:, 0].argmin())
    target_index = int(anchors[:, 0].argmax())
    scene_tokens = torch.stack((anchors[:, 0], torch.zeros(12)), dim=-1).unsqueeze(0)
    scene_tokens.requires_grad_()
    own = torch.tensor([[1.0, 0.0]])
    alternate = torch.tensor([[-1.0, 0.0]])

    loss, diagnostics = ordered_spatial_relation_contrastive_loss(
        scene_tokens,
        anchors[target_index].unsqueeze(0),
        anchors[reference_index].unsqueeze(0),
        own,
        alternate,
        temperature=0.2,
        margin=0.5,
    )

    assert float(loss.detach()) == pytest.approx(0.0)
    assert diagnostics["eligible_side_count"] == 1
    assert diagnostics["latent_count"] == 12
    assert diagnostics["achieved_margin"].tolist() == pytest.approx([1.0])
    assert diagnostics["relation_answer_cosine"].tolist() == pytest.approx([1.0])
    target_weights = diagnostics["target_weights"]
    reference_weights = diagnostics["reference_weights"]
    assert target_weights.shape == (1, 12)
    assert reference_weights.shape == (1, 12)
    assert bool((target_weights > 0).all())
    assert bool((reference_weights > 0).all())
    assert target_weights.sum(dim=-1).tolist() == pytest.approx([1.0])
    assert reference_weights.sum(dim=-1).tolist() == pytest.approx([1.0])
    assert int(target_weights.argmax(dim=-1)) == target_index
    assert int(reference_weights.argmax(dim=-1)) == reference_index
    assert float(diagnostics["minimum_target_weight"]) > 0.0
    assert float(diagnostics["minimum_reference_weight"]) > 0.0


def test_ordered_relation_hinge_reaches_scene_tokens_but_freezes_answer_embeddings() -> None:
    anchors = spatial_anchors(10)
    target_index = int(anchors[:, 0].argmax())
    reference_index = int(anchors[:, 0].argmin())
    scene_tokens = torch.stack((anchors[:, 0], torch.zeros(10)), dim=-1).unsqueeze(0)
    scene_tokens.requires_grad_()
    # The scene relation starts along X while the desired frozen answer direction
    # is Y, making the hinge active with a nonzero scene-token gradient.
    own = torch.tensor([[0.0, 2.0]], requires_grad=True)
    alternate = torch.tensor([[0.0, -3.0]], requires_grad=True)

    loss, diagnostics = ordered_spatial_relation_contrastive_loss(
        scene_tokens,
        anchors[target_index].unsqueeze(0),
        anchors[reference_index].unsqueeze(0),
        own,
        alternate,
        temperature=0.25,
        margin=0.4,
    )
    loss.backward()

    assert float(loss.detach()) == pytest.approx(0.4)
    assert diagnostics["relation_answer_cosine"].tolist() == pytest.approx([0.0])
    assert scene_tokens.grad is not None
    assert float(scene_tokens.grad.abs().sum()) > 0.0
    assert own.grad is None
    assert alternate.grad is None


@pytest.mark.parametrize("temperature", [0.0, -0.1, float("inf"), float("nan")])
def test_ordered_relation_rejects_invalid_temperature(temperature: float) -> None:
    values = torch.ones(1, 2, 3)
    xyz = torch.zeros(1, 3)
    with pytest.raises(ValueError, match="temperature"):
        ordered_spatial_relation_contrastive_loss(
            values, xyz, xyz, torch.ones(1, 3), -torch.ones(1, 3), temperature=temperature
        )


@pytest.mark.parametrize("margin", [-0.1, 1.1, float("inf"), float("nan")])
def test_ordered_relation_rejects_invalid_margin(margin: float) -> None:
    values = torch.ones(1, 2, 3)
    xyz = torch.zeros(1, 3)
    with pytest.raises(ValueError, match="margin"):
        ordered_spatial_relation_contrastive_loss(
            values, xyz, xyz, torch.ones(1, 3), -torch.ones(1, 3), margin=margin
        )


@pytest.mark.parametrize(
    ("argument", "value", "message"),
    [
        ("scene_tokens", torch.ones(2, 3), "scene_tokens"),
        ("normalized_target_xyz", torch.ones(1, 2), "normalized_target_xyz"),
        ("normalized_reference_xyz", torch.ones(2, 3), "normalized_reference_xyz"),
        ("own_answer_embeddings", torch.ones(1, 2), "own_answer_embeddings"),
        ("alternate_answer_embeddings", torch.ones(2, 3), "alternate_answer_embeddings"),
    ],
)
def test_ordered_relation_validates_tensor_shapes(
    argument: str, value: torch.Tensor, message: str
) -> None:
    arguments = {
        "scene_tokens": torch.ones(1, 2, 3),
        "normalized_target_xyz": torch.zeros(1, 3),
        "normalized_reference_xyz": torch.zeros(1, 3),
        "own_answer_embeddings": torch.ones(1, 3),
        "alternate_answer_embeddings": -torch.ones(1, 3),
    }
    arguments[argument] = value

    with pytest.raises(ValueError, match=message):
        ordered_spatial_relation_contrastive_loss(**arguments)


def test_ordered_relation_rejects_degenerate_frozen_answer_direction() -> None:
    scene_tokens = torch.ones(1, 4, 3, requires_grad=True)
    xyz = torch.zeros(1, 3)
    identical = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)

    with pytest.raises(ValueError, match="nonzero direction"):
        ordered_spatial_relation_contrastive_loss(
            scene_tokens, xyz, xyz, identical, identical.clone()
        )


def test_ordered_relation_keeps_dense_weights_at_extreme_positive_temperature() -> None:
    scene_tokens = torch.randn(1, 32, 4)
    target = torch.tensor([[1.0, 1.0, 1.0]])
    reference = torch.tensor([[-1.0, -1.0, -1.0]])
    own = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    alternate = -own

    _, diagnostics = ordered_spatial_relation_contrastive_loss(
        scene_tokens,
        target,
        reference,
        own,
        alternate,
        temperature=1e-8,
    )

    assert bool((diagnostics["target_weights"] > 0).all())
    assert bool((diagnostics["reference_weights"] > 0).all())
    assert diagnostics["target_weights"].sum().item() == pytest.approx(1.0)
    assert diagnostics["reference_weights"].sum().item() == pytest.approx(1.0)
