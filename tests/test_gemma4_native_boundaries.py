from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from semantic_3d_chat.chat.runtime import validate_checkpoint_contract
from semantic_3d_chat.language.gemma4_backend import Gemma4PrefixBackend
from semantic_3d_chat.language.local_lm import LocalLanguageModel
from semantic_3d_chat.language.prefix_injection import (
    SCENE_BOUNDARY_MODE_GEMMA4_NATIVE_IMAGE,
    ContinuousPrefixComposer,
    native_gemma4_image_contract_setting,
    prefix_sha256,
    scene_boundary_contract_mismatch,
    scene_boundary_mode_setting,
    stack_prefix_batches,
)
from semantic_3d_chat.training.checkpointing import (
    load_adapter_checkpoint,
    save_adapter_checkpoint,
)


class _ScaledEmbedding(nn.Embedding):
    def __init__(self, vocabulary: int, width: int, scale: float) -> None:
        super().__init__(vocabulary, width)
        self.scale = scale

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return super().forward(token_ids) * self.scale


class _FakeTextModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            hidden_size=6,
            num_hidden_layers=2,
            hidden_size_per_layer_input=3,
            vocab_size=16,
            pad_token_id=0,
            bos_token_id=2,
            use_bidirectional_attention=None,
        )
        self.embed_tokens_per_layer = _ScaledEmbedding(16, 6, scale=3.0)

    def get_per_layer_inputs(
        self, token_ids: torch.Tensor, _embeddings: torch.Tensor
    ) -> torch.Tensor:
        return self.embed_tokens_per_layer(token_ids).reshape(
            *token_ids.shape,
            self.config.num_hidden_layers,
            self.config.hidden_size_per_layer_input,
        )


class _FakeGemma4(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(9401)
        self.embedding = _ScaledEmbedding(16, 6, scale=2.0)
        self.model = SimpleNamespace(language_model=_FakeTextModel())
        self.config = SimpleNamespace(
            text_config=self.model.language_model.config,
            boi_token_id=10,
            image_token_id=11,
            eoi_token_id=12,
        )

    def get_input_embeddings(self) -> nn.Module:
        return self.embedding


def _tokenizer(**overrides: int) -> SimpleNamespace:
    values = {
        "bos_token_id": 2,
        "pad_token_id": 0,
        "boi_token_id": 10,
        "image_token_id": 11,
        "eoi_token_id": 12,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _backend(**tokenizer_overrides: int) -> Gemma4PrefixBackend:
    return Gemma4PrefixBackend(
        _FakeGemma4(),
        tokenizer=_tokenizer(**tokenizer_overrides),
        model_revision="fake-pinned-revision",
    )


def _native_config() -> dict:
    return {
        "language": {
            "backend": "gemma4",
            "revision": "fake-pinned-revision",
            "scene_prefix_after_bos": True,
            "scene_boundary_mode": SCENE_BOUNDARY_MODE_GEMMA4_NATIVE_IMAGE,
            "gemma4_native_image_contract": {
                "schema_version": 1,
                "model_revision": "fake-pinned-revision",
                "bos_token_id": 2,
                "pad_token_id": 0,
                "boi_token_id": 10,
                "image_token_id": 11,
                "eoi_token_id": 12,
                "use_bidirectional_attention": None,
            },
        }
    }


def test_native_protocol_is_model_derived_and_rejects_tokenizer_drift() -> None:
    backend = _backend()
    assert (
        backend.native_image_contract()
        == _native_config()["language"]["gemma4_native_image_contract"]
    )

    boi, eoi = backend.native_boundary_embeddings()
    ids = torch.tensor([[10, 12]])
    expected = backend.model.get_input_embeddings()(ids)
    assert torch.equal(boi, expected[:, :1])
    assert torch.equal(eoi, expected[:, 1:])
    assert not torch.equal(
        boi,
        backend.model.get_input_embeddings().weight[10].reshape(1, 1, -1),
    )

    with pytest.raises(ValueError, match="tokenizer/config boi_token_id mismatch"):
        _backend(boi_token_id=9).native_image_contract()


def test_native_prefix_exact_order_ple_types_and_answer_alignment() -> None:
    backend = _backend()
    composer = ContinuousPrefixComposer(
        6,
        scene_prefix_after_bos=True,
        bos_token_id=2,
        scene_boundary_mode=SCENE_BOUNDARY_MODE_GEMMA4_NATIVE_IMAGE,
        native_boundary_embeddings=backend.native_boundary_embeddings(),
    )
    assert "scene_start" not in dict(composer.named_parameters())
    assert "scene_end" not in dict(composer.named_parameters())
    assert "scene_start" in dict(composer.named_buffers())
    assert "scene_end" in dict(composer.named_buffers())

    scene = torch.randn(1, 256, 6, requires_grad=True)
    prompt_ids = torch.tensor([[2, 5, 6]])
    answer_ids = torch.tensor([[7, 1]])
    scene_prefix = composer.scene_prefix(scene)
    batch = composer.compose(
        scene,
        prompt_ids,
        backend.model.get_input_embeddings(),
        answer_ids,
        prefix_backend=backend,
    )

    token_embeddings = backend.model.get_input_embeddings()
    text_model = backend.text_model
    bos_embedding = token_embeddings(prompt_ids[:, :1])
    boi_embedding = token_embeddings(torch.tensor([[10]]))
    eoi_embedding = token_embeddings(torch.tensor([[12]]))
    prompt_remainder = token_embeddings(prompt_ids[:, 1:])
    answer_embeddings = token_embeddings(answer_ids)
    prompt_ple = text_model.get_per_layer_inputs(prompt_ids, token_embeddings(prompt_ids))
    pad_ids = torch.zeros((1, 256), dtype=torch.long)
    expected_pad_ple = text_model.get_per_layer_inputs(pad_ids, token_embeddings(pad_ids))
    boundary_ids = torch.tensor([[10, 12]])
    boundary_embeddings = token_embeddings(boundary_ids)
    expected_boundary_ple = text_model.get_per_layer_inputs(boundary_ids, boundary_embeddings)

    assert scene_prefix.shape == (1, 258, 6)
    assert batch.scene_prefix_length == 258
    assert batch.inputs_embeds.shape == (1, 263, 6)
    assert torch.equal(batch.inputs_embeds[:, 0:1], bos_embedding)
    assert torch.equal(batch.inputs_embeds[:, 1:2], boi_embedding)
    assert torch.equal(batch.inputs_embeds[:, 2:258], scene)
    assert torch.equal(batch.inputs_embeds[:, 258:259], eoi_embedding)
    assert torch.equal(batch.inputs_embeds[:, 259:261], prompt_remainder)
    assert torch.equal(batch.inputs_embeds[:, 261:], answer_embeddings)
    assert torch.equal(batch.per_layer_inputs[:, 0:1], prompt_ple[:, :1])
    assert torch.equal(batch.per_layer_inputs[:, 1:2], expected_boundary_ple[:, :1])
    assert torch.equal(batch.per_layer_inputs[:, 2:258], expected_pad_ple)
    assert torch.equal(batch.per_layer_inputs[:, 258:259], expected_boundary_ple[:, 1:])
    assert torch.equal(batch.mm_token_type_ids[:, :2], torch.tensor([[0, 0]], dtype=torch.long))
    assert torch.equal(batch.mm_token_type_ids[:, 2:258], torch.ones((1, 256), dtype=torch.long))
    assert torch.equal(batch.mm_token_type_ids[:, 258:], torch.zeros((1, 5), dtype=torch.long))
    assert torch.equal(batch.labels[:, :261], torch.full((1, 261), -100))
    assert torch.equal(batch.labels[:, 261:], answer_ids)

    batch.inputs_embeds.sum().backward()
    assert scene.grad is not None and scene.grad.abs().sum() > 0


def test_native_control_tokens_use_pad_ple_text_modality_and_preserve_scene_identity() -> None:
    backend = _backend()
    composer = ContinuousPrefixComposer(
        6,
        scene_prefix_after_bos=True,
        bos_token_id=2,
        scene_boundary_mode=SCENE_BOUNDARY_MODE_GEMMA4_NATIVE_IMAGE,
        native_boundary_embeddings=backend.native_boundary_embeddings(),
    )
    scene = torch.randn(1, 4, 6)
    prompt_ids = torch.tensor([[2, 5, 6]])
    answer_ids = torch.tensor([[7, 1]])
    control_tokens = torch.randn(1, 2, 6)
    scene_prefix = composer.scene_prefix(scene)
    scene_hash = prefix_sha256(scene_prefix)

    trained = composer.compose(
        scene,
        prompt_ids,
        backend.model.get_input_embeddings(),
        answer_ids,
        prefix_backend=backend,
        control_tokens=control_tokens,
    )
    generated = composer.compose(
        scene,
        prompt_ids,
        backend.model.get_input_embeddings(),
        prefix_backend=backend,
        control_tokens=control_tokens + 1.0,
    )

    token_embeddings = backend.model.get_input_embeddings()
    prompt_embeddings = token_embeddings(prompt_ids)
    answer_embeddings = token_embeddings(answer_ids)
    pad_ids = torch.zeros((1, 2), dtype=torch.long)
    pad_embeddings = token_embeddings(pad_ids)
    pad_ple = backend.text_model.get_per_layer_inputs(pad_ids, pad_embeddings)

    assert trained.scene_prefix_length == generated.scene_prefix_length == 6
    assert prefix_sha256(composer.scene_prefix(scene)) == scene_hash
    assert torch.equal(trained.inputs_embeds[:, :1], prompt_embeddings[:, :1])
    assert torch.equal(trained.inputs_embeds[:, 1:7], scene_prefix)
    assert torch.equal(trained.inputs_embeds[:, 7:9], prompt_embeddings[:, 1:])
    assert torch.equal(trained.inputs_embeds[:, 9:11], control_tokens)
    assert torch.equal(trained.inputs_embeds[:, 11:], answer_embeddings)
    assert torch.equal(trained.per_layer_inputs[:, 9:11], pad_ple)
    assert torch.equal(trained.mm_token_type_ids[:, 9:11], torch.zeros((1, 2), dtype=torch.long))
    assert torch.equal(trained.labels[:, :11], torch.full((1, 11), -100))
    assert torch.equal(trained.labels[:, 11:], answer_ids)
    assert generated.labels is None
    assert torch.equal(generated.inputs_embeds[:, -2:], control_tokens + 1.0)
    assert torch.equal(generated.per_layer_inputs[:, -2:], pad_ple)
    assert torch.equal(generated.mm_token_type_ids[:, -2:], torch.zeros((1, 2), dtype=torch.long))


@pytest.mark.parametrize(
    ("control_tokens", "message"),
    [
        (torch.randn(1, 6), "shape"),
        (torch.randn(2, 1, 6), "batch sizes"),
        (torch.randn(1, 1, 5), "hidden size"),
        (torch.tensor([[[float("inf"), 0.0, 0.0, 0.0, 0.0, 0.0]]]), "finite"),
    ],
)
def test_gemma_backend_rejects_invalid_control_tokens(
    control_tokens: torch.Tensor,
    message: str,
) -> None:
    backend = _backend()
    with pytest.raises(ValueError, match=message):
        backend.prepare(
            torch.randn(1, 4, 6),
            torch.tensor([[2, 5]]),
            control_tokens=control_tokens,
        )


def test_native_variable_length_stack_preserves_metadata_labels_and_padding() -> None:
    backend = _backend()
    composer = ContinuousPrefixComposer(
        6,
        scene_prefix_after_bos=True,
        bos_token_id=2,
        scene_boundary_mode=SCENE_BOUNDARY_MODE_GEMMA4_NATIVE_IMAGE,
        native_boundary_embeddings=backend.native_boundary_embeddings(),
    )
    first = composer.compose(
        torch.randn(1, 4, 6),
        torch.tensor([[2, 5]]),
        backend.model.get_input_embeddings(),
        torch.tensor([[7]]),
        prefix_backend=backend,
    )
    second = composer.compose(
        torch.randn(1, 4, 6),
        torch.tensor([[2, 5, 6, 7]]),
        backend.model.get_input_embeddings(),
        torch.tensor([[8, 9]]),
        prefix_backend=backend,
    )

    stacked = stack_prefix_batches([first, second], torch.device("cpu"), prefix_backend=backend)
    first_length = first.inputs_embeds.shape[1]
    padding_length = stacked.inputs_embeds.shape[1] - first_length
    expected_pad_inputs, expected_pad_ple, expected_pad_types = backend.padding_values(
        1, padding_length, device=torch.device("cpu")
    )

    assert stacked.scene_prefix_length == 6
    assert torch.equal(
        stacked.attention_mask[0, first_length:],
        torch.zeros(padding_length, dtype=torch.long),
    )
    assert torch.equal(stacked.labels[0, first_length:], torch.full((padding_length,), -100))
    assert torch.equal(stacked.labels[0, first_length - 1 : first_length], torch.tensor([7]))
    assert torch.equal(stacked.inputs_embeds[0, first_length:], expected_pad_inputs[0])
    assert torch.equal(stacked.per_layer_inputs[0, first_length:], expected_pad_ple[0])
    assert torch.equal(stacked.mm_token_type_ids[0, first_length:], expected_pad_types[0])
    assert torch.equal(stacked.mm_token_type_ids[0, 2:6], torch.ones(4, dtype=torch.long))
    assert torch.equal(stacked.mm_token_type_ids[0, 1:2], torch.zeros(1, dtype=torch.long))
    assert torch.equal(stacked.mm_token_type_ids[0, 6:7], torch.zeros(1, dtype=torch.long))


def test_native_boundaries_are_fixed_question_independent_and_tamper_evident(
    tmp_path,
) -> None:
    backend = _backend()
    native = backend.native_boundary_embeddings()
    composer = ContinuousPrefixComposer(
        6,
        scene_prefix_after_bos=True,
        bos_token_id=2,
        scene_boundary_mode=SCENE_BOUNDARY_MODE_GEMMA4_NATIVE_IMAGE,
        native_boundary_embeddings=native,
    )
    scene = torch.randn(1, 4, 6)
    before = prefix_sha256(composer.scene_prefix(scene))
    _questions = ("first question", "unrelated second question")
    after = prefix_sha256(composer.scene_prefix(scene))
    assert before == after
    composer.validate_native_boundary_embeddings(native)

    audit_style_reload = ContinuousPrefixComposer(
        6,
        scene_prefix_after_bos=True,
        bos_token_id=2,
        scene_boundary_mode=SCENE_BOUNDARY_MODE_GEMMA4_NATIVE_IMAGE,
    )
    audit_style_reload.load_state_dict(composer.state_dict(), strict=True)
    audit_style_reload.validate_native_boundary_embeddings(native)
    assert prefix_sha256(audit_style_reload.scene_prefix(scene)) == before

    checkpoint = save_adapter_checkpoint(
        tmp_path / "checkpoint",
        {"composer": composer},
        {"scene_boundary_mode": SCENE_BOUNDARY_MODE_GEMMA4_NATIVE_IMAGE},
    )
    checkpoint_reload = ContinuousPrefixComposer(
        6,
        scene_prefix_after_bos=True,
        bos_token_id=2,
        scene_boundary_mode=SCENE_BOUNDARY_MODE_GEMMA4_NATIVE_IMAGE,
    )
    metadata = load_adapter_checkpoint(checkpoint, {"composer": checkpoint_reload}, device="cpu")
    assert metadata["scene_boundary_mode"] == SCENE_BOUNDARY_MODE_GEMMA4_NATIVE_IMAGE
    checkpoint_reload.validate_native_boundary_embeddings(native)
    assert prefix_sha256(checkpoint_reload.scene_prefix(scene)) == before

    with torch.no_grad():
        composer.scene_end.add_(1.0)
    with pytest.raises(ValueError, match="EOI boundary embedding"):
        composer.validate_native_boundary_embeddings(native)
    with pytest.raises(ValueError, match="EOI embedding"):
        composer.compose(
            scene,
            torch.tensor([[2, 5]]),
            backend.model.get_input_embeddings(),
            prefix_backend=backend,
        )


def test_native_mode_rejects_generic_backend_and_uninitialized_boundaries() -> None:
    backend = _backend()
    native = backend.native_boundary_embeddings()
    composer = ContinuousPrefixComposer(
        6,
        scene_prefix_after_bos=True,
        bos_token_id=2,
        scene_boundary_mode=SCENE_BOUNDARY_MODE_GEMMA4_NATIVE_IMAGE,
        native_boundary_embeddings=native,
    )
    with pytest.raises(ValueError, match="requires the Gemma4 prefix backend"):
        composer.compose(
            torch.randn(1, 2, 6),
            torch.tensor([[2, 5]]),
            backend.model.get_input_embeddings(),
        )

    uninitialized = ContinuousPrefixComposer(
        6,
        scene_prefix_after_bos=True,
        bos_token_id=2,
        scene_boundary_mode=SCENE_BOUNDARY_MODE_GEMMA4_NATIVE_IMAGE,
    )
    with pytest.raises(RuntimeError, match="uninitialized"):
        uninitialized.scene_prefix(torch.randn(1, 2, 6))

    generic_language = LocalLanguageModel(
        model=backend.model,
        tokenizer=_tokenizer(),
        device=torch.device("cpu"),
    )
    with pytest.raises(ValueError, match="requires the Gemma4 prefix backend"):
        generic_language.generate_from_scene_prefix(
            composer.scene_prefix(torch.randn(1, 2, 6)),
            torch.tensor([[2, 5]]),
            max_new_tokens=1,
            eos_token_ids=None,
            scene_prefix_after_bos=True,
            scene_boundary_mode=SCENE_BOUNDARY_MODE_GEMMA4_NATIVE_IMAGE,
            fallback=lambda *_args: torch.tensor([[1]]),
        )


def test_chat_generation_dispatch_uses_same_native_prefix_protocol() -> None:
    backend = _backend()
    language = LocalLanguageModel(
        model=backend.model,
        tokenizer=_tokenizer(),
        device=torch.device("cpu"),
        prefix_backend=backend,
        backend_name="gemma4",
    )
    composer = ContinuousPrefixComposer(
        6,
        scene_prefix_after_bos=True,
        bos_token_id=2,
        scene_boundary_mode=SCENE_BOUNDARY_MODE_GEMMA4_NATIVE_IMAGE,
        native_boundary_embeddings=language.scene_boundary_embeddings(
            SCENE_BOUNDARY_MODE_GEMMA4_NATIVE_IMAGE
        ),
    )
    scene_prefix = composer.scene_prefix(torch.randn(1, 4, 6))
    captured = {}

    def fake_generate(prepared, *, max_new_tokens, eos_token_ids):
        captured["prepared"] = prepared
        captured["max_new_tokens"] = max_new_tokens
        captured["eos_token_ids"] = eos_token_ids
        return torch.tensor([[1]])

    backend.generate = fake_generate
    generated = language.generate_from_scene_prefix(
        scene_prefix,
        torch.tensor([[2, 5, 6]]),
        max_new_tokens=3,
        eos_token_ids=1,
        scene_prefix_after_bos=True,
        scene_boundary_mode=SCENE_BOUNDARY_MODE_GEMMA4_NATIVE_IMAGE,
        fallback=lambda *_args: pytest.fail("generic generation fallback was called"),
    )

    assert torch.equal(generated, torch.tensor([[1]]))
    assert captured["max_new_tokens"] == 3
    assert captured["eos_token_ids"] == 1
    assert torch.equal(
        captured["prepared"].mm_token_type_ids,
        torch.tensor([[0, 0, 1, 1, 1, 1, 0, 0, 0]], dtype=torch.long),
    )


def test_native_config_and_checkpoint_contract_are_strict_and_legacy_safe() -> None:
    config = _native_config()
    contract = native_gemma4_image_contract_setting(config)
    assert scene_boundary_mode_setting(config) == SCENE_BOUNDARY_MODE_GEMMA4_NATIVE_IMAGE
    assert contract == config["language"]["gemma4_native_image_contract"]
    metadata = {
        "scene_boundary_mode": SCENE_BOUNDARY_MODE_GEMMA4_NATIVE_IMAGE,
        "gemma4_native_image_contract": contract,
    }
    assert (
        scene_boundary_contract_mismatch(
            metadata, SCENE_BOUNDARY_MODE_GEMMA4_NATIVE_IMAGE, contract
        )
        is None
    )
    assert scene_boundary_contract_mismatch({}, "learned", None) is None
    mismatch = scene_boundary_contract_mismatch(
        {}, SCENE_BOUNDARY_MODE_GEMMA4_NATIVE_IMAGE, contract
    )
    assert mismatch == {
        "checkpoint": "<missing; legacy learned>",
        "runtime": SCENE_BOUNDARY_MODE_GEMMA4_NATIVE_IMAGE,
    }

    wrong_revision = _native_config()
    wrong_revision["language"]["gemma4_native_image_contract"]["model_revision"] = "wrong"
    with pytest.raises(ValueError, match="must exactly match"):
        native_gemma4_image_contract_setting(wrong_revision)
    invalid_mode = _native_config()
    invalid_mode["language"]["scene_boundary_mode"] = "native-ish"
    with pytest.raises(ValueError, match="must be one of"):
        scene_boundary_mode_setting(invalid_mode)


def test_real_legacy_v6_composer_checkpoint_round_trip_and_v7_rejection(
    tmp_path,
) -> None:
    torch.manual_seed(9601)
    legacy_v6 = ContinuousPrefixComposer(
        6,
        scene_prefix_after_bos=True,
        bos_token_id=2,
    )
    checkpoint = save_adapter_checkpoint(
        tmp_path / "legacy_v6",
        {"composer": legacy_v6},
        {
            "schema_version": 3,
            "scene_prefix_after_bos": True,
            # Deliberately omit scene_boundary_mode: pre-v7 checkpoints used
            # learned delimiters and must retain that exact interpretation.
        },
    )
    current_learned = ContinuousPrefixComposer(
        6,
        scene_prefix_after_bos=True,
        bos_token_id=2,
    )
    metadata = load_adapter_checkpoint(checkpoint, {"composer": current_learned}, device="cpu")

    assert scene_boundary_contract_mismatch(metadata, "learned", None) is None
    assert torch.equal(current_learned.scene_start, legacy_v6.scene_start)
    assert torch.equal(current_learned.scene_end, legacy_v6.scene_end)
    native_contract = _native_config()["language"]["gemma4_native_image_contract"]
    assert scene_boundary_contract_mismatch(
        metadata,
        SCENE_BOUNDARY_MODE_GEMMA4_NATIVE_IMAGE,
        native_contract,
    ) == {
        "checkpoint": "<missing; legacy learned>",
        "runtime": SCENE_BOUNDARY_MODE_GEMMA4_NATIVE_IMAGE,
    }


def test_runtime_checkpoint_contract_covers_native_ids_revision_and_attention() -> None:
    config = _native_config()
    config["language"]["model_id"] = "local/fake-gemma4"
    config["scene_encoder"] = {
        "global_latents": 256,
        "model_dim": 8,
        "input_voxel_size_m": 0.15,
    }
    contract = native_gemma4_image_contract_setting(config)
    metadata = {
        "schema_version": 3,
        "semantic_dim": 12,
        "language_hidden_dim": 6,
        "language_model_id": "local/fake-gemma4",
        "language_revision": "fake-pinned-revision",
        "scene_latents": 256,
        "scene_model_dim": 8,
        "input_voxel_size_m": 0.15,
        "config_hash": "training-hash",
        "scene_prefix_after_bos": True,
        "scene_boundary_mode": SCENE_BOUNDARY_MODE_GEMMA4_NATIVE_IMAGE,
        "gemma4_native_image_contract": contract,
    }
    warnings = validate_checkpoint_contract(
        metadata, config, semantic_dim=12, language_hidden_dim=6
    )
    assert warnings

    tampered = {**metadata, "gemma4_native_image_contract": {**contract, "boi_token_id": 9}}
    with pytest.raises(ValueError, match="scene_boundary_mode"):
        validate_checkpoint_contract(tampered, config, semantic_dim=12, language_hidden_dim=6)
