from __future__ import annotations

import pytest
import torch

from semantic_3d_chat.language.relation_token import DenseRelationTokenAdapter


def _adapter(*, seed: int = 15008) -> DenseRelationTokenAdapter:
    return DenseRelationTokenAdapter(
        language_hidden_dim=32,
        num_scene_tokens=11,
        adapter_dim=16,
        heads=4,
        fourier_bands=2,
        initialization_seed=seed,
    )


def _inputs() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(1234)
    return (
        torch.randn(2, 11, 32, generator=generator),
        torch.randn(2, 7, 32, generator=generator),
    )


def test_dense_relation_token_shapes_and_strictly_positive_attention() -> None:
    adapter = _adapter()
    scene, question = _inputs()
    output = adapter(scene, question)

    assert output.relation_token.shape == (2, 1, 32)
    assert output.target_xyz_normalized.shape == (2, 3)
    assert output.reference_xyz_normalized.shape == (2, 3)
    assert output.delta_xyz_normalized.shape == (2, 3)
    assert output.target_attention.shape == (2, 11)
    assert output.reference_attention.shape == (2, 11)
    assert torch.all(output.target_attention > 0)
    assert torch.all(output.reference_attention > 0)
    assert torch.allclose(output.target_attention.sum(-1), torch.ones(2))
    assert torch.allclose(output.reference_attention.sum(-1), torch.ones(2))
    assert torch.all(output.target_xyz_normalized.abs() <= 1.0)
    assert torch.all(output.reference_xyz_normalized.abs() <= 1.0)


def test_every_scene_token_has_gradient_to_relation_token() -> None:
    adapter = _adapter()
    scene, question = _inputs()
    scene.requires_grad_(True)
    output = adapter(scene, question)

    output.relation_token.square().sum().backward()

    assert scene.grad is not None
    per_token_gradient = scene.grad.abs().sum(dim=-1)
    assert torch.all(per_token_gradient > 0)


def test_scene_swap_changes_relation_token() -> None:
    adapter = _adapter()
    scene, question = _inputs()
    changed = scene.clone()
    changed[:, 0] += 3.0

    baseline = adapter(scene, question).relation_token
    swapped = adapter(changed, question).relation_token

    assert not torch.allclose(baseline, swapped)


def test_swapping_roles_swaps_geometry_and_negates_delta_and_token() -> None:
    adapter = _adapter()
    scene, question = _inputs()

    normal = adapter(scene, question)
    swapped = adapter(scene, question, swap_roles=True)

    assert torch.allclose(normal.target_attention, swapped.reference_attention)
    assert torch.allclose(normal.reference_attention, swapped.target_attention)
    assert torch.allclose(normal.target_xyz_normalized, swapped.reference_xyz_normalized)
    assert torch.allclose(normal.reference_xyz_normalized, swapped.target_xyz_normalized)
    assert torch.allclose(normal.delta_xyz_normalized, -swapped.delta_xyz_normalized)
    assert torch.allclose(normal.relation_token, -swapped.relation_token, atol=1e-6)


def test_masked_question_embeddings_cannot_change_output() -> None:
    adapter = _adapter()
    scene, question = _inputs()
    mask = torch.ones(2, 7, dtype=torch.bool)
    mask[:, -1] = False
    changed = question.clone()
    changed[:, -1] += 10_000.0

    baseline = adapter(scene, question, mask)
    perturbed = adapter(scene, changed, mask)

    assert torch.equal(baseline.relation_token, perturbed.relation_token)
    assert torch.equal(baseline.target_attention, perturbed.target_attention)
    assert torch.equal(baseline.reference_attention, perturbed.reference_attention)


def test_question_token_order_changes_relation_output() -> None:
    adapter = _adapter()
    scene, question = _inputs()
    permutation = torch.tensor([1, 0, 2, 3, 4, 5, 6])

    baseline = adapter(scene, question).relation_token
    reordered = adapter(scene, question.index_select(1, permutation)).relation_token

    assert not torch.allclose(baseline, reordered)


def test_initialization_is_deterministic_without_consuming_global_rng() -> None:
    torch.manual_seed(7)
    before = torch.random.get_rng_state().clone()
    first = _adapter(seed=91)
    after = torch.random.get_rng_state()
    assert torch.equal(before, after)

    torch.manual_seed(999)
    second = _adapter(seed=91)
    assert all(
        torch.equal(first_value, second.state_dict()[name])
        for name, first_value in first.state_dict().items()
    )
    scene, question = _inputs()
    assert torch.equal(
        first(scene, question).relation_token,
        second(scene, question).relation_token,
    )


def test_gemma_sized_adapter_stays_below_one_million_parameters() -> None:
    adapter = DenseRelationTokenAdapter(
        language_hidden_dim=1536,
        num_scene_tokens=256,
        adapter_dim=128,
        heads=4,
        fourier_bands=4,
    )
    assert adapter.trainable_parameter_count < 1_000_000


@pytest.mark.parametrize(
    ("scene_shape", "question_shape", "message"),
    [
        ((2, 10, 32), (2, 7, 32), "scene_tokens"),
        ((2, 11, 31), (2, 7, 32), "scene_tokens"),
        ((2, 11, 32), (1, 7, 32), "question_embeddings"),
        ((2, 11, 32), (2, 0, 32), "question_embeddings"),
        ((2, 11, 32), (2, 7, 31), "question_embeddings"),
    ],
)
def test_shape_validation(
    scene_shape: tuple[int, ...],
    question_shape: tuple[int, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _adapter()(torch.zeros(scene_shape), torch.zeros(question_shape))


@pytest.mark.parametrize("which", ["scene", "question"])
def test_nonfinite_inputs_are_rejected(which: str) -> None:
    scene, question = _inputs()
    if which == "scene":
        scene[0, 0, 0] = float("nan")
    else:
        question[0, 0, 0] = float("inf")
    with pytest.raises(ValueError, match="NaN or infinity"):
        _adapter()(scene, question)


def test_question_mask_validation() -> None:
    scene, question = _inputs()
    adapter = _adapter()
    with pytest.raises(ValueError, match="shape"):
        adapter(scene, question, torch.ones(2, 6, dtype=torch.bool))
    with pytest.raises(ValueError, match="at least one"):
        adapter(scene, question, torch.zeros(2, 7, dtype=torch.bool))
    with pytest.raises(ValueError, match="zero or one"):
        adapter(scene, question, torch.full((2, 7), 2, dtype=torch.int64))
