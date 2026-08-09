import torch

from semantic_3d_chat.scene_encoder.perceiver import (
    GlobalSceneResampler,
    SignalPreservingProjection,
    spatial_anchors,
    spatial_coverage_weights,
)
from semantic_3d_chat.scene_encoder.point_tokens import PointTokenProjection
from semantic_3d_chat.scene_encoder.projector import SceneTokenizer


def _scene_inputs(count: int, semantic_dim: int) -> dict[str, torch.Tensor]:
    return {
        "semantic": torch.randn(count, semantic_dim),
        "xyz": torch.rand(count, 3) * torch.tensor([6.0, 5.0, 3.0]),
        "rgb": torch.randint(0, 256, (count, 3)),
        "normal": torch.nn.functional.normalize(torch.randn(count, 3), dim=-1),
        "confidence": torch.rand(count),
        "observation_count": torch.randint(1, 8, (count,)),
        "room_min": torch.tensor([0.0, 0.0, 0.0]),
        "room_max": torch.tensor([6.0, 5.0, 3.0]),
    }


def test_scene_tokenizer_is_fixed_size_and_processes_every_voxel() -> None:
    torch.manual_seed(7)
    count, semantic_dim = 17, 32
    xyz = torch.rand(count, 3) * torch.tensor([6.0, 5.0, 3.0])
    model = SceneTokenizer(
        semantic_dim=semantic_dim,
        model_dim=32,
        language_hidden_dim=48,
        block_size_m=0.5,
        tokens_per_block=2,
        global_latents=16,
        heads=4,
        global_layers=1,
        fourier_bands=3,
    )
    output = model(
        semantic=torch.randn(count, semantic_dim),
        xyz=xyz,
        rgb=torch.randint(0, 256, (count, 3)),
        normal=torch.nn.functional.normalize(torch.randn(count, 3), dim=-1),
        confidence=torch.rand(count),
        observation_count=torch.randint(1, 8, (count,)),
        room_min=torch.tensor([0.0, 0.0, 0.0]),
        room_max=torch.tensor([6.0, 5.0, 3.0]),
    )
    assert output.scene_tokens.shape == (1, 16, 48)
    assert output.audit["processed_voxels"].item() == count
    assert output.block_tokens.shape[0] == 2 * output.audit["voxel_counts"].numel()


def test_native_aligned_coverage_assigns_every_voxel_nonzero_weight() -> None:
    positions = torch.tensor(
        [
            [-1.0, -1.0, -1.0],
            [1.0, 1.0, 1.0],
            [-0.4, 0.2, 0.8],
            [0.7, -0.6, 0.1],
            [0.0, 0.0, 0.0],
        ]
    )
    weights = spatial_coverage_weights(
        positions,
        spatial_anchors(16),
        temperature=0.20,
    )

    assert weights.shape == (1, 16, positions.shape[0])
    assert bool(torch.all(weights > 0))
    assert torch.allclose(weights.sum(dim=-1), torch.ones(1, 16))
    assert bool(torch.all(weights.sum(dim=(0, 1)) > 0))


def test_native_aligned_bypass_uses_all_voxels_and_preserves_gradients() -> None:
    torch.manual_seed(11)
    count, semantic_dim, aligned_dim = 19, 24, 8
    inputs = _scene_inputs(count, semantic_dim)
    inputs["semantic"].requires_grad_(True)
    model = SceneTokenizer(
        semantic_dim=semantic_dim,
        model_dim=16,
        language_hidden_dim=aligned_dim,
        block_size_m=0.5,
        tokens_per_block=2,
        global_latents=8,
        heads=4,
        global_layers=1,
        fourier_bands=2,
        language_aligned_tail_dim=aligned_dim,
        native_aligned_coverage_scale=1.0,
        learned_scene_token_scale=0.0,
    )

    output = model(**inputs)
    output.scene_tokens.sum().backward()

    assert output.scene_tokens.shape == (1, 8, aligned_dim)
    assert output.audit["native_aligned_processed_voxels"].item() == count
    assert output.audit["native_aligned_min_weight"].item() > 0
    assert output.audit["native_aligned_min_voxel_contribution"].item() > 0
    aligned_gradient = inputs["semantic"].grad[:, -aligned_dim:]
    assert bool(torch.all(aligned_gradient > 0))


def test_combined_aligned_bypass_keeps_trainable_path_and_controls_rms() -> None:
    torch.manual_seed(13)
    count, semantic_dim, aligned_dim = 23, 24, 8
    inputs = _scene_inputs(count, semantic_dim)
    model = SceneTokenizer(
        semantic_dim=semantic_dim,
        model_dim=16,
        language_hidden_dim=aligned_dim,
        block_size_m=0.5,
        tokens_per_block=2,
        global_latents=8,
        heads=4,
        global_layers=1,
        fourier_bands=2,
        language_aligned_tail_dim=aligned_dim,
        native_aligned_coverage_scale=1.0,
        learned_scene_token_scale=0.1,
        learned_scene_token_rms_target=0.65,
    )

    output = model(**inputs)
    output.scene_tokens.square().mean().backward()

    assert torch.isclose(
        output.audit["learned_scene_token_rms"],
        torch.tensor(0.65),
        rtol=1e-4,
        atol=1e-4,
    )
    trainable_gradients = [
        parameter.grad
        for parameter in model.language_projection.trainable.parameters()
        if parameter.requires_grad
    ]
    assert trainable_gradients
    assert all(gradient is not None for gradient in trainable_gradients)
    assert any(bool(torch.any(gradient != 0)) for gradient in trainable_gradients)


def test_scene_tokenizer_defaults_keep_legacy_single_path() -> None:
    model = SceneTokenizer(
        semantic_dim=12,
        model_dim=16,
        language_hidden_dim=20,
        global_latents=8,
        heads=4,
        global_layers=1,
        fourier_bands=2,
    )

    assert model.language_aligned_tail_dim == 0
    assert model.native_aligned_coverage is None
    assert model.native_aligned_coverage_scale == 0.0
    assert model.learned_scene_token_scale == 1.0
    assert model.learned_scene_token_rms_target is None


def test_scene_tokens_do_not_take_a_question_argument() -> None:
    argument_names = SceneTokenizer.forward.__code__.co_varnames[
        : SceneTokenizer.forward.__code__.co_argcount
    ]
    assert "question" not in argument_names
    assert "query" not in argument_names


def _mean_off_diagonal_cosine(values: torch.Tensor) -> float:
    normalized = torch.nn.functional.normalize(values.detach().float(), dim=-1)
    similarities = normalized @ normalized.transpose(-1, -2)
    count = similarities.shape[-1]
    mask = ~torch.eye(count, dtype=torch.bool)
    return float(similarities[0][mask].mean())


def test_spatial_coverage_resampler_prevents_duplicate_latents() -> None:
    torch.manual_seed(17)
    resampler = GlobalSceneResampler(32, 16, heads=4, layers=2)
    context = torch.randn(1, 80, 32)
    positions = torch.rand(80, 3) * 2.0 - 1.0

    latents = resampler(context, positions)

    assert latents.shape == (1, 16, 32)
    assert _mean_off_diagonal_cosine(latents) < 0.8
    assert float(latents.detach().std(dim=1).mean()) > 0.25


def test_resampler_keeps_local_change_signal_and_all_block_gradients() -> None:
    torch.manual_seed(23)
    resampler = GlobalSceneResampler(32, 16, heads=4, layers=2)
    context = torch.randn(1, 80, 32, requires_grad=True)
    positions = torch.rand(80, 3) * 2.0 - 1.0
    original = resampler(context, positions)
    changed_context = context.detach().clone()
    changed_context[:, 10, 0] += 1.0
    changed = resampler(changed_context, positions)

    relative_l2 = (original - changed).norm() / (0.5 * (original.norm() + changed.norm()))
    assert float(relative_l2.detach()) > 1e-3

    original.sum().backward()
    per_block_gradient = context.grad.norm(dim=-1)
    assert bool(torch.all(per_block_gradient > 0))


def test_projection_fixed_bypass_cannot_erase_input_change() -> None:
    torch.manual_seed(29)
    projection = SignalPreservingProjection(32, 48, skip_scale=0.1)
    # Simulate a collapsed trainable branch; the fixed isometry must still carry
    # input distinctions into the language hidden space.
    for parameter in projection.trainable.parameters():
        parameter.data.zero_()
    first = torch.randn(1, 16, 32)
    second = first.clone()
    second[:, 4, 7] += 0.5

    projected_first = projection(first)
    projected_second = projection(second)

    assert not torch.equal(projected_first, projected_second)
    source_delta = (first - second).norm()
    projected_delta = (projected_first - projected_second).norm()
    assert torch.isclose(projected_delta, source_delta * 0.1, rtol=1e-4, atol=1e-6)


def test_point_projection_fixed_paths_preserve_semantic_and_geometry_changes() -> None:
    torch.manual_seed(31)
    projection = PointTokenProjection(semantic_dim=12, model_dim=16, fourier_bands=2)
    for parameter in projection.network.parameters():
        parameter.data.zero_()
    semantic = torch.randn(3, 12)
    xyz = torch.tensor([[0.1, 0.2, 0.3], [0.5, 0.2, 0.3], [0.1, 0.2, 0.3]])
    common = {
        "rgb": torch.zeros(3, 3),
        "normal": torch.tensor([[0.0, 0.0, 1.0]]).expand(3, -1),
        "confidence": torch.ones(3),
        "observation_count": torch.ones(3),
        "room_min": torch.zeros(3),
        "room_max": torch.ones(3),
    }
    baseline, _ = projection(semantic, xyz, **common)
    semantic_changed = semantic.clone()
    semantic_changed[0, 4] += 2.0
    semantic_output, _ = projection(semantic_changed, xyz, **common)
    geometry_changed = xyz.clone()
    geometry_changed[2, 2] += 0.4
    geometry_output, _ = projection(semantic, geometry_changed, **common)

    assert not torch.equal(baseline[0], semantic_output[0])
    assert not torch.equal(baseline[2], geometry_output[2])
