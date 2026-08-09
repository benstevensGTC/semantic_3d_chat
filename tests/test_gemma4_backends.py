from __future__ import annotations

import importlib.metadata
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from semantic_3d_chat.language.gemma4_backend import Gemma4PrefixBackend
from semantic_3d_chat.language.prefix_injection import (
    SCENE_BOUNDARY_MODE_GEMMA4_NATIVE_IMAGE,
    ContinuousPrefixComposer,
)
from semantic_3d_chat.vision.gemma4_encoder import (
    DenseGemma4Encoder,
    _broadcast_pooled_aligned_grid,
    load_selective_gemma4_vision_bundle,
)
from semantic_3d_chat.vision.gemma4_probe import (
    derive_vision_grid_mapping,
    patchify_complete_image,
)
from semantic_3d_chat.vision.model_registry import GEMMA4_E2B, DenseVisionModelSpec


def _has_gemma4_transformers() -> bool:
    try:
        return importlib.metadata.version("transformers") == "5.14.1"
    except importlib.metadata.PackageNotFoundError:
        return False


def test_production_registry_documents_full_unpooled_e2b_grid() -> None:
    assert GEMMA4_E2B.architecture == "gemma4"
    assert GEMMA4_E2B.image_size == 224
    assert GEMMA4_E2B.grid_size == (48, 48)
    assert GEMMA4_E2B.patch_count == 2304
    assert GEMMA4_E2B.pooling_kernel_size == 3
    assert GEMMA4_E2B.native_dim == 768
    assert GEMMA4_E2B.aligned_dim == 1536
    assert GEMMA4_E2B.hidden_state_index(8) == 7
    assert GEMMA4_E2B.hidden_state_index(16) == 15


def test_all_2304_pre_pool_cells_receive_their_owning_native_projected_token() -> None:
    y, x = torch.meshgrid(torch.arange(48), torch.arange(48), indexing="ij")
    positions = torch.stack((x, y), dim=-1).reshape(1, 2304, 2)
    mapping = derive_vision_grid_mapping(positions, pooling_kernel_size=3)
    projected = torch.arange(256, dtype=torch.float32).reshape(256, 1)
    grid = _broadcast_pooled_aligned_grid(
        projected,
        pre_to_post_token=mapping.pre_to_post_token,
        positions_xy=mapping.pre_xy,
        pre_grid_size=(48, 48),
    )

    expected = (y // 3) * 16 + x // 3
    assert grid.shape == (48, 48, 1)
    assert torch.equal(grid[..., 0], expected.float())
    ownership_counts = torch.bincount(torch.tensor(mapping.pre_to_post_token), minlength=256)
    assert torch.equal(ownership_counts, torch.full((256,), 9))


def _tiny_full_config():
    from transformers import Gemma4Config, Gemma4TextConfig, Gemma4VisionConfig

    text = Gemma4TextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        global_head_dim=8,
        hidden_size_per_layer_input=8,
        vocab_size_per_layer_input=64,
        layer_types=["sliding_attention", "full_attention", "full_attention"],
        sliding_window=8,
        max_position_embeddings=128,
        pad_token_id=0,
        bos_token_id=2,
        eos_token_id=1,
        use_double_wide_mlp=False,
        num_kv_shared_layers=1,
    )
    vision = Gemma4VisionConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        pooling_kernel_size=3,
        patch_size=16,
        position_embedding_size=64,
        use_clipped_linears=False,
        standardize=False,
    )
    return Gemma4Config(
        text_config=text,
        vision_config=vision,
        audio_config=None,
        image_token_id=60,
        video_token_id=61,
        audio_token_id=62,
        boi_token_id=58,
        eoi_token_id=59,
    )


class _TinyCompleteImageProcessor:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, *, images: Image.Image, return_tensors: str):
        assert images.size == (96, 96)
        assert return_tensors == "pt"
        self.calls += 1
        array = torch.from_numpy(__import__("numpy").asarray(images).copy())
        image = array.permute(2, 0, 1).unsqueeze(0).float() / 255.0
        pixel_values, positions = patchify_complete_image(image, patch_size=16)
        return {"pixel_values": pixel_values, "image_position_ids": positions}


@pytest.mark.skipif(
    not _has_gemma4_transformers(),
    reason="Production Gemma 4 tiny models run in .venv-gemma4",
)
def test_production_dense_encoder_keeps_six_by_six_pre_pool_grid_from_one_call() -> None:
    from transformers import Gemma4ForConditionalGeneration

    torch.manual_seed(3301)
    model = Gemma4ForConditionalGeneration(_tiny_full_config()).eval()
    processor = _TinyCompleteImageProcessor()
    spec = DenseVisionModelSpec(
        model_id="local/tiny-gemma4",
        architecture="gemma4",
        image_size=96,
        patch_size=16,
        native_dim=16,
        aligned_dim=32,
        num_hidden_layers=2,
        default_middle_layer=1,
        default_late_layer=2,
        has_cls_token=False,
        license_name="test-only",
        processed_grid_size=(6, 6),
        pooling_kernel_size=3,
        hidden_states_include_input_embedding=False,
    )
    encoder = DenseGemma4Encoder(
        model,
        processor,
        spec,
        device=torch.device("cpu"),
        compute_dtype=torch.float32,
        storage_dtype=torch.float16,
    )
    calls = 0

    def count_call(_module, _inputs, _output):
        nonlocal calls
        calls += 1

    hook = model.model.vision_tower.register_forward_hook(count_call)
    gradient = torch.arange(96, dtype=torch.uint8).view(1, 96).expand(96, 96)
    image = Image.fromarray(torch.stack((gradient, gradient.T, gradient), dim=-1).numpy())
    features = encoder.encode_image(image)
    hook.remove()

    assert processor.calls == 1
    assert calls == 1
    assert features.native_middle.shape == (6, 6, 16)
    assert features.native_late.shape == (6, 6, 16)
    assert features.aligned.shape == (6, 6, 32)
    assert features.spatial_features.shape == (6, 6, 64)
    assert features.native_middle.dtype == torch.float16
    assert not torch.equal(features.native_late[0, 0], features.native_late[0, 1])
    assert torch.equal(features.aligned[0, 0], features.aligned[0, 1])
    assert torch.equal(features.aligned[0, 0], features.aligned[2, 2])
    assert not torch.equal(features.aligned[0, 0], features.aligned[0, 3])


@pytest.mark.skipif(
    not _has_gemma4_transformers(),
    reason="Selective Gemma 4 loader runs in .venv-gemma4",
)
def test_selective_loader_strictly_loads_only_vision_and_projector(tmp_path: Path) -> None:
    from safetensors.torch import save_file
    from transformers import Gemma4ForConditionalGeneration

    torch.manual_seed(5501)
    config = _tiny_full_config()
    source = Gemma4ForConditionalGeneration(config).eval()
    tensors = {
        **{
            f"model.vision_tower.{name}": value.detach().contiguous()
            for name, value in source.model.vision_tower.state_dict().items()
        },
        **{
            f"model.embed_vision.{name}": value.detach().contiguous()
            for name, value in source.model.embed_vision.state_dict().items()
        },
        "model.language_model.norm.weight": torch.ones(32),
    }
    checkpoint = tmp_path / "model.safetensors"
    save_file(tensors, checkpoint)

    bundle = load_selective_gemma4_vision_bundle(
        config,
        checkpoint,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert sum(parameter.numel() for parameter in bundle.vision_tower.parameters()) == sum(
        parameter.numel() for parameter in source.model.vision_tower.parameters()
    )
    assert sum(parameter.numel() for parameter in bundle.embed_vision.parameters()) == sum(
        parameter.numel() for parameter in source.model.embed_vision.parameters()
    )
    source_first = next(source.model.vision_tower.parameters())
    loaded_first = next(bundle.vision_tower.parameters())
    assert torch.equal(source_first, loaded_first)
    assert all(parameter.device.type == "cpu" for parameter in bundle.parameters())

    missing_key = next(key for key in tensors if key.startswith("model.embed_vision."))
    incomplete = dict(tensors)
    incomplete.pop(missing_key)
    incomplete_path = tmp_path / "incomplete.safetensors"
    save_file(incomplete, incomplete_path)
    with pytest.raises(RuntimeError, match="missing="):
        load_selective_gemma4_vision_bundle(
            config,
            incomplete_path,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )


@pytest.mark.skipif(
    not _has_gemma4_transformers(),
    reason="Production Gemma 4 tiny models run in .venv-gemma4",
)
def test_production_prefix_backend_uses_pad_ple_and_extends_cache() -> None:
    from transformers import Gemma4ForConditionalGeneration

    torch.manual_seed(4401)
    model = Gemma4ForConditionalGeneration(_tiny_full_config()).eval()
    backend = Gemma4PrefixBackend(model)
    scene_prefix = torch.randn(1, 5, 32)
    prompt_ids = torch.tensor([[2, 9, 13]], dtype=torch.long)
    prepared = backend.prepare(scene_prefix, prompt_ids)

    pad_ids = torch.zeros((1, 5), dtype=torch.long)
    pad_embeddings = model.get_input_embeddings()(pad_ids)
    expected_scene_ple = model.model.language_model.get_per_layer_inputs(
        pad_ids, pad_embeddings
    )
    assert prepared.inputs_embeds.shape == (1, 8, 32)
    assert prepared.per_layer_inputs.shape == (1, 8, 3, 8)
    assert torch.equal(prepared.per_layer_inputs[:, :5], expected_scene_ple)
    assert torch.equal(prepared.mm_token_type_ids, torch.zeros(1, 8, dtype=torch.long))

    with torch.inference_mode():
        first = backend.prefill(prepared)
        assert first.past_key_values.get_seq_length() == 8
        next_id = first.logits[:, -1].argmax(dim=-1, keepdim=True)
        second = backend.decode_step(
            next_id,
            past_key_values=first.past_key_values,
            attention_mask=torch.ones((1, 9), dtype=torch.long),
        )
    assert second.past_key_values.get_seq_length() == 9
    assert torch.isfinite(second.logits).all()

    generated = backend.generate(prepared, max_new_tokens=3, eos_token_ids=None)
    assert generated.shape == (1, 3)


@pytest.mark.skipif(
    not _has_gemma4_transformers(),
    reason="Production Gemma 4 tiny models run in .venv-gemma4",
)
def test_gemma_bos_first_layout_preserves_native_bos_embedding_and_ple() -> None:
    from transformers import Gemma4ForConditionalGeneration

    torch.manual_seed(4402)
    model = Gemma4ForConditionalGeneration(_tiny_full_config()).eval()
    backend = Gemma4PrefixBackend(model)
    scene_prefix = torch.randn(1, 5, 32)
    prompt_ids = torch.tensor([[2, 9, 13]], dtype=torch.long)
    answer_ids = torch.tensor([[17, 1]], dtype=torch.long)

    prepared = backend.prepare(
        scene_prefix,
        prompt_ids,
        answer_ids,
        scene_prefix_after_bos=True,
    )

    prompt_embeddings = model.get_input_embeddings()(prompt_ids)
    prompt_ple = model.model.language_model.get_per_layer_inputs(
        prompt_ids, prompt_embeddings
    )
    pad_ids = torch.zeros((1, 5), dtype=torch.long)
    pad_embeddings = model.get_input_embeddings()(pad_ids)
    expected_scene_ple = model.model.language_model.get_per_layer_inputs(
        pad_ids, pad_embeddings
    )
    answer_embeddings = model.get_input_embeddings()(answer_ids)
    answer_ple = model.model.language_model.get_per_layer_inputs(
        answer_ids, answer_embeddings
    )

    assert prepared.inputs_embeds.shape == (1, 10, 32)
    assert prepared.per_layer_inputs.shape == (1, 10, 3, 8)
    assert torch.equal(prepared.inputs_embeds[:, :1], prompt_embeddings[:, :1])
    assert torch.equal(prepared.inputs_embeds[:, 1:6], scene_prefix)
    assert torch.equal(prepared.inputs_embeds[:, 6:8], prompt_embeddings[:, 1:])
    assert torch.equal(prepared.inputs_embeds[:, 8:], answer_embeddings)
    assert torch.equal(prepared.per_layer_inputs[:, :1], prompt_ple[:, :1])
    assert torch.equal(prepared.per_layer_inputs[:, 1:6], expected_scene_ple)
    assert torch.equal(prepared.per_layer_inputs[:, 6:8], prompt_ple[:, 1:])
    assert torch.equal(prepared.per_layer_inputs[:, 8:], answer_ple)
    assert torch.equal(prepared.labels[:, :8], torch.full((1, 8), -100))
    assert torch.equal(prepared.labels[:, 8:], answer_ids)
    assert torch.equal(prepared.mm_token_type_ids, torch.zeros(1, 10, dtype=torch.long))

    with pytest.raises(ValueError, match="start with bos_token_id=2"):
        backend.prepare(
            scene_prefix,
            torch.tensor([[9, 2]], dtype=torch.long),
            scene_prefix_after_bos=True,
        )


@pytest.mark.skipif(
    not _has_gemma4_transformers(),
    reason="Production Gemma 4 tiny models run in .venv-gemma4",
)
def test_tiny_production_gemma_native_boundaries_match_visual_protocol() -> None:
    from transformers import Gemma4ForConditionalGeneration

    torch.manual_seed(4403)
    model = Gemma4ForConditionalGeneration(_tiny_full_config()).eval()
    tokenizer = SimpleNamespace(
        bos_token_id=2,
        pad_token_id=0,
        boi_token_id=58,
        image_token_id=60,
        eoi_token_id=59,
    )
    backend = Gemma4PrefixBackend(
        model,
        tokenizer=tokenizer,
        model_revision="tiny-revision",
    )
    composer = ContinuousPrefixComposer(
        32,
        scene_prefix_after_bos=True,
        bos_token_id=2,
        scene_boundary_mode=SCENE_BOUNDARY_MODE_GEMMA4_NATIVE_IMAGE,
        native_boundary_embeddings=backend.native_boundary_embeddings(),
    )
    scene = torch.randn(1, 4, 32, requires_grad=True)
    prompt_ids = torch.tensor([[2, 9, 13]], dtype=torch.long)
    answer_ids = torch.tensor([[17, 1]], dtype=torch.long)
    prepared = composer.compose(
        scene,
        prompt_ids,
        model.get_input_embeddings(),
        answer_ids,
        prefix_backend=backend,
    )

    marker_ids = torch.tensor([[58, 59]], dtype=torch.long)
    marker_embeddings = model.get_input_embeddings()(marker_ids)
    marker_ple = model.model.language_model.get_per_layer_inputs(
        marker_ids, marker_embeddings
    )
    pad_ids = torch.zeros((1, 4), dtype=torch.long)
    pad_embeddings = model.get_input_embeddings()(pad_ids)
    pad_ple = model.model.language_model.get_per_layer_inputs(pad_ids, pad_embeddings)

    assert torch.equal(prepared.inputs_embeds[:, 0:1], model.get_input_embeddings()(prompt_ids[:, :1]))
    assert torch.equal(prepared.inputs_embeds[:, 1:2], marker_embeddings[:, :1])
    assert torch.equal(prepared.inputs_embeds[:, 2:6], scene)
    assert torch.equal(prepared.inputs_embeds[:, 6:7], marker_embeddings[:, 1:])
    assert torch.equal(prepared.per_layer_inputs[:, 1:2], marker_ple[:, :1])
    assert torch.equal(prepared.per_layer_inputs[:, 2:6], pad_ple)
    assert torch.equal(prepared.per_layer_inputs[:, 6:7], marker_ple[:, 1:])
    assert torch.equal(
        prepared.mm_token_type_ids,
        torch.tensor([[0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0]], dtype=torch.long),
    )
    assert torch.equal(prepared.labels[:, :9], torch.full((1, 9), -100))
    assert torch.equal(prepared.labels[:, 9:], answer_ids)

    output = backend.prefill(prepared, use_cache=False)
    assert output.loss is not None and torch.isfinite(output.loss)
    output.loss.backward()
    assert scene.grad is not None and scene.grad.abs().sum() > 0
    assert "scene_start" not in dict(composer.named_parameters())
    assert "scene_end" not in dict(composer.named_parameters())


def test_gemma_backend_rejects_plain_causal_model() -> None:
    with pytest.raises(TypeError, match="language_model"):
        Gemma4PrefixBackend(SimpleNamespace(model=SimpleNamespace()))
