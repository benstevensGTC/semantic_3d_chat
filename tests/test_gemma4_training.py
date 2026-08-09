from __future__ import annotations

import importlib.metadata
from types import SimpleNamespace

import pytest
import torch

import semantic_3d_chat.language.local_lm as local_lm_module
from semantic_3d_chat.language.gemma4_backend import Gemma4PrefixBackend
from semantic_3d_chat.language.local_lm import (
    LocalLanguageModel,
    load_local_language_model,
    prompt_token_ids,
)
from semantic_3d_chat.language.lora import (
    LoRALinear,
    LoRASettings,
    install_lora_adapters,
)
from semantic_3d_chat.language.prefix_injection import (
    ContinuousPrefixComposer,
    stack_prefix_batches,
)


def _has_gemma4_transformers() -> bool:
    try:
        return importlib.metadata.version("transformers") == "5.14.1"
    except importlib.metadata.PackageNotFoundError:
        return False


pytestmark = pytest.mark.skipif(
    not _has_gemma4_transformers(),
    reason="Tiny Gemma 4 training tests run in the isolated .venv-gemma4 environment",
)


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


class _BatchEncodingLike:
    """Mimic the Transformers 5 non-dict chat-template return type."""

    def __init__(self, input_ids: torch.Tensor) -> None:
        self.input_ids = input_ids


class _TemplateTokenizer:
    def apply_chat_template(self, *_args, **_kwargs):
        return _BatchEncodingLike(torch.tensor([[2, 7, 8]], dtype=torch.long))


def test_prompt_token_ids_accepts_transformers5_batch_encoding() -> None:
    ids = prompt_token_ids(
        _TemplateTokenizer(),
        "stable instruction",
        "question",
        torch.device("cpu"),
    )
    assert torch.equal(ids, torch.tensor([[2, 7, 8]], dtype=torch.long))


def test_training_composer_forwards_bos_first_layout_to_gemma_ple_backend() -> None:
    from transformers import Gemma4ForConditionalGeneration

    model = Gemma4ForConditionalGeneration(_tiny_full_config()).eval()
    backend = Gemma4PrefixBackend(model)
    composer = ContinuousPrefixComposer(
        hidden_size=32,
        scene_prefix_after_bos=True,
        bos_token_id=2,
    )
    scene = torch.randn(1, 3, 32)
    prompt_ids = torch.tensor([[2, 7, 8]], dtype=torch.long)
    answer_ids = torch.tensor([[12, 1]], dtype=torch.long)

    batch = composer.compose(
        scene,
        prompt_ids,
        model.get_input_embeddings(),
        answer_ids,
        prefix_backend=backend,
    )
    scene_prefix = composer.scene_prefix(scene)
    prompt_embeddings = model.get_input_embeddings()(prompt_ids)

    assert batch.scene_prefix_length == 5
    assert batch.inputs_embeds.shape == (1, 10, 32)
    assert torch.equal(batch.inputs_embeds[:, :1], prompt_embeddings[:, :1])
    assert torch.equal(batch.inputs_embeds[:, 1:6], scene_prefix)
    assert torch.equal(batch.inputs_embeds[:, 6:8], prompt_embeddings[:, 1:])
    assert batch.per_layer_inputs is not None
    assert batch.per_layer_inputs.shape == (1, 10, 3, 8)
    assert torch.equal(batch.labels[:, :8], torch.full((1, 8), -100))
    assert torch.equal(batch.labels[:, 8:], answer_ids)


def test_gemma4_loader_selects_conditional_model_without_real_weights(monkeypatch) -> None:
    from transformers import Gemma4ForConditionalGeneration

    tiny_model = Gemma4ForConditionalGeneration(_tiny_full_config())
    tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=1)
    captured: dict[str, object] = {}

    def fake_model_load(model_id: str, **kwargs):
        captured["model_id"] = model_id
        captured.update(kwargs)
        return tiny_model

    def fake_tokenizer_load(model_id: str, **kwargs):
        captured["tokenizer_model_id"] = model_id
        captured["tokenizer_kwargs"] = kwargs
        return tokenizer

    monkeypatch.setattr(Gemma4ForConditionalGeneration, "from_pretrained", fake_model_load)
    monkeypatch.setattr(local_lm_module.AutoTokenizer, "from_pretrained", fake_tokenizer_load)
    monkeypatch.setattr(local_lm_module, "select_device", lambda: torch.device("cpu"))

    language = load_local_language_model(
        "local/tiny-gemma-4",
        revision="pinned-revision",
        requested_dtype="bfloat16",
        freeze=True,
        local_files_only=True,
        backend="gemma4",
    )

    assert language.model is tiny_model
    assert language.backend_name == "gemma4"
    assert isinstance(language.prefix_backend, Gemma4PrefixBackend)
    assert language.hidden_size == 32
    assert captured["model_id"] == "local/tiny-gemma-4"
    assert captured["revision"] == "pinned-revision"
    assert captured["local_files_only"] is True
    assert captured["dtype"] is torch.float32
    assert all(not parameter.requires_grad for parameter in tiny_model.parameters())
    assert not tiny_model.training
    assert not language.decoder_gradient_checkpointing_enabled
    assert not language.decoder_module.is_gradient_checkpointing


def test_tiny_gemma4_loader_checkpoints_only_frozen_decoder_recomputation(
    monkeypatch,
) -> None:
    from transformers import Gemma4ForConditionalGeneration

    tiny_model = Gemma4ForConditionalGeneration(_tiny_full_config())
    tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=1)
    monkeypatch.setattr(
        Gemma4ForConditionalGeneration,
        "from_pretrained",
        lambda *_args, **_kwargs: tiny_model,
    )
    monkeypatch.setattr(
        local_lm_module.AutoTokenizer,
        "from_pretrained",
        lambda *_args, **_kwargs: tokenizer,
    )
    monkeypatch.setattr(local_lm_module, "select_device", lambda: torch.device("cpu"))

    language = load_local_language_model(
        "local/tiny-gemma-4",
        requested_dtype="bfloat16",
        freeze=True,
        local_files_only=True,
        backend="gemma4",
        decoder_gradient_checkpointing=True,
    )

    assert language.decoder_gradient_checkpointing_enabled
    assert language.decoder_module.is_gradient_checkpointing
    assert language.decoder_module.training
    assert not tiny_model.training
    assert all(not parameter.requires_grad for parameter in tiny_model.parameters())
    assert all(
        not layer.gradient_checkpointing for layer in tiny_model.model.vision_tower.encoder.layers
    )

    first_layer = language.decoder_module.layers[0]
    original_forward = first_layer.forward
    forward_calls: list[bool] = []

    def tracked_forward(*args, **kwargs):
        forward_calls.append(torch.is_grad_enabled())
        return original_forward(*args, **kwargs)

    first_layer.forward = tracked_forward
    composer = ContinuousPrefixComposer(hidden_size=32)
    scene = torch.randn(1, 4, 32, requires_grad=True)
    prepared = composer.compose(
        scene,
        torch.tensor([[2, 7]], dtype=torch.long),
        tiny_model.get_input_embeddings(),
        torch.tensor([[12, 1]], dtype=torch.long),
        prefix_backend=language.prefix_backend,
    )
    output = language.forward_prefix_batch(prepared, use_cache=False)
    assert len(forward_calls) == 1
    output.loss.backward()

    # Non-reentrant activation checkpointing recomputes the decoder layer in
    # backward while still propagating to the external continuous scene input.
    assert len(forward_calls) == 2
    assert scene.grad is not None and scene.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in tiny_model.parameters())

    with torch.inference_mode():
        language.forward_prefix_batch(prepared, use_cache=False)
    assert language.decoder_module.training
    assert len(forward_calls) == 3


def test_tiny_gemma4_exact_decoder_lora_receives_prefix_gradient_on_cpu() -> None:
    from transformers import Gemma4ForConditionalGeneration

    torch.manual_seed(20260810)
    model = Gemma4ForConditionalGeneration(_tiny_full_config()).eval()
    model.requires_grad_(False)
    settings = LoRASettings(
        enabled=True,
        rank=2,
        alpha=4.0,
        dropout=0.0,
        target_modules=(
            "model.language_model.layers.2.self_attn.q_proj",
            "model.language_model.layers.2.self_attn.o_proj",
        ),
    )
    installation = install_lora_adapters(model, settings)
    assert installation is not None
    layer = model.model.language_model.layers[2]
    assert isinstance(layer.self_attn.q_proj, LoRALinear)
    assert isinstance(layer.self_attn.o_proj, LoRALinear)
    assert isinstance(model.model.language_model.layers[1].self_attn.q_proj, torch.nn.Linear)
    language = LocalLanguageModel(
        model=model,
        tokenizer=SimpleNamespace(),
        device=torch.device("cpu"),
        prefix_backend=Gemma4PrefixBackend(model),
        backend_name="gemma4",
    )
    scene = torch.randn(1, 4, 32, requires_grad=True)
    prepared = ContinuousPrefixComposer(hidden_size=32).compose(
        scene,
        torch.tensor([[2, 7]], dtype=torch.long),
        model.get_input_embeddings(),
        torch.tensor([[12, 1]], dtype=torch.long),
        prefix_backend=language.prefix_backend,
    )

    output = language.forward_prefix_batch(prepared, use_cache=False)
    output.loss.backward()

    assert scene.grad is not None and scene.grad.abs().sum() > 0
    for adapter in installation.adapters:
        assert adapter.lora_b.grad is not None and adapter.lora_b.grad.abs().sum() > 0
        assert adapter.lora_a.grad is not None
        assert torch.count_nonzero(adapter.lora_a.grad) == 0
        assert all(parameter.grad is None for parameter in adapter.base.parameters())
    installation.assert_only_lora_trainable(model)


def test_gemma4_loader_streams_directly_to_mps_without_cpu_duplicate(monkeypatch) -> None:
    from transformers import Gemma4ForConditionalGeneration

    tiny_model = Gemma4ForConditionalGeneration(_tiny_full_config())
    tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=1)
    captured: dict[str, object] = {}

    def fake_model_load(_model_id: str, **kwargs):
        captured.update(kwargs)
        return tiny_model

    monkeypatch.setattr(Gemma4ForConditionalGeneration, "from_pretrained", fake_model_load)
    monkeypatch.setattr(
        local_lm_module.AutoTokenizer,
        "from_pretrained",
        lambda *_args, **_kwargs: tokenizer,
    )
    monkeypatch.setattr(local_lm_module, "select_device", lambda: torch.device("mps"))

    language = load_local_language_model(
        "local/tiny-gemma-4",
        requested_dtype="bfloat16",
        local_files_only=True,
        backend="gemma4",
    )

    assert language.device == torch.device("mps")
    assert captured["device_map"] == {"": torch.device("mps")}
    assert captured["dtype"] is torch.bfloat16


def test_tiny_gemma4_variable_length_training_keeps_pad_ple_and_answer_masks() -> None:
    from transformers import Gemma4ForConditionalGeneration

    torch.manual_seed(20260809)
    model = Gemma4ForConditionalGeneration(_tiny_full_config()).eval()
    model.requires_grad_(False)
    backend = Gemma4PrefixBackend(model)
    language = LocalLanguageModel(
        model=model,
        tokenizer=SimpleNamespace(),
        device=torch.device("cpu"),
        prefix_backend=backend,
        backend_name="gemma4",
    )
    composer = ContinuousPrefixComposer(hidden_size=32)
    first_scene = torch.randn(1, 4, 32, requires_grad=True)
    second_scene = torch.randn(1, 4, 32, requires_grad=True)
    first_prompt = torch.tensor([[2, 7]], dtype=torch.long)
    second_prompt = torch.tensor([[2, 8, 9, 10, 11]], dtype=torch.long)
    first_answer = torch.tensor([[12, 1]], dtype=torch.long)
    second_answer = torch.tensor([[13, 14, 1]], dtype=torch.long)

    first = composer.compose(
        first_scene,
        first_prompt,
        model.get_input_embeddings(),
        first_answer,
        prefix_backend=backend,
    )
    second = composer.compose(
        second_scene,
        second_prompt,
        model.get_input_embeddings(),
        second_answer,
        prefix_backend=backend,
    )
    stacked = stack_prefix_batches([first, second], torch.device("cpu"), prefix_backend=backend)

    first_length = first.inputs_embeds.shape[1]
    maximum_length = second.inputs_embeds.shape[1]
    assert stacked.inputs_embeds.shape == (2, maximum_length, 32)
    assert stacked.per_layer_inputs.shape == (2, maximum_length, 3, 8)
    assert stacked.mm_token_type_ids.shape == (2, maximum_length)
    padding_length = maximum_length - first_length
    assert torch.equal(
        stacked.attention_mask[0, first_length:],
        torch.zeros(padding_length, dtype=torch.long),
    )
    assert torch.equal(stacked.labels[0, first_length:], torch.full((padding_length,), -100))
    assert torch.equal(stacked.labels[0, :8], torch.full((8,), -100))
    assert torch.equal(stacked.labels[0, 8:10], first_answer[0])
    assert torch.equal(stacked.labels[1, :11], torch.full((11,), -100))
    assert torch.equal(stacked.labels[1, 11:14], second_answer[0])

    expected_pad_inputs, expected_pad_ple, expected_pad_mm = backend.padding_values(
        1, padding_length, device=torch.device("cpu")
    )
    assert torch.equal(stacked.inputs_embeds[0, first_length:], expected_pad_inputs[0])
    assert torch.equal(stacked.per_layer_inputs[0, first_length:], expected_pad_ple[0])
    assert torch.equal(stacked.mm_token_type_ids[0, first_length:], expected_pad_mm[0])
    assert torch.equal(stacked.per_layer_inputs[0, :first_length], first.per_layer_inputs[0])
    assert torch.equal(stacked.per_layer_inputs[1], second.per_layer_inputs[0])
    assert torch.equal(stacked.mm_token_type_ids[0, :first_length], first.mm_token_type_ids[0])

    output = language.forward_prefix_batch(stacked, use_cache=False)
    assert output.loss is not None and torch.isfinite(output.loss)
    output.loss.backward()
    assert first_scene.grad is not None and first_scene.grad.abs().sum() > 0
    assert second_scene.grad is not None and second_scene.grad.abs().sum() > 0
    assert composer.scene_start.grad is not None and composer.scene_start.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in model.parameters())
