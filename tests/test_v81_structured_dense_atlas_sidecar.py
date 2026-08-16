from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from semantic_3d_chat.language.gemma4_backend import Gemma4PrefixBackend
from semantic_3d_chat.language.local_lm import question_token_ids
from semantic_3d_chat.language.prefix_injection import (
    SCENE_BOUNDARY_MODE_GEMMA4_NATIVE_IMAGE,
    prefix_sha256,
)
from semantic_3d_chat.language.v81_structured_dense_atlas_sidecar import (
    ATLAS_UNIFORM_FLOOR_MASS,
    BASE_PREFIX_TOKENS,
    FINAL_LOGIT_SOFTCAPPING,
    FIXED_PREFIX_TOKENS,
    HIDDEN_SIZE,
    INPUT_EMBEDDING_TENSOR_NAME,
    MINIMUM_ATLAS_WEIGHT,
    MODEL_BLOB_SHA256_IDENTITY,
    PROBE_COUNT,
    RAW_ATLAS_LOGIT_SCALE,
    VALUES_PER_PROBE,
    assert_prefix_binding_v81,
    audit_v75_v2_prefix_v81,
    bind_fixed_prefix_before_question_v81,
    deterministic_atlas_read_v81,
    frozen_lm_head_logits_v81,
    latest_user_question_query_v81,
    reconstruct_base_v54_prefix_v81,
    split_v75_v2_prefix_v81,
)
from semantic_3d_chat.scene_encoder.fixed_prefix_atlas import tensor_sha256
from semantic_3d_chat.scene_encoder.v81_scene_memory_artifact import (
    MEMORY_FILENAME,
    METADATA_FILENAME,
    load_v81_scene_memory,
    save_v81_scene_memory,
)

_BASE_SHA = "a" * 64
_RUNTIME_SHA = "b" * 64
_CONTROL_SHA = "c" * 64
_PROBE_SHA = "d" * 64


def _memory(seed: int = 8101, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return (
        torch.randn(
            1,
            FIXED_PREFIX_TOKENS,
            HIDDEN_SIZE,
            generator=generator,
            dtype=torch.float32,
        )
        * 0.01
    ).to(dtype=dtype)


class _QuestionTokenizer:
    def __init__(self, token_ids: tuple[int, ...] = (3, 5, 7)) -> None:
        self.token_ids = token_ids
        self.calls: list[dict[str, Any]] = []

    def __call__(self, text: str, **kwargs: Any) -> dict[str, torch.Tensor]:
        self.calls.append({"text": text, **kwargs})
        return {"input_ids": torch.tensor([self.token_ids], dtype=torch.long)}


def test_v81_exact_v75_v2_parse_and_base258_gather() -> None:
    token_values = torch.arange(FIXED_PREFIX_TOKENS, dtype=torch.float32)
    memory = token_values.reshape(1, -1, 1).expand(-1, -1, HIDDEN_SIZE)
    banks = split_v75_v2_prefix_v81(memory)

    expected_keys = 1 + 5 * torch.arange(PROBE_COUNT)
    expected_values = (
        2
        + 5 * torch.arange(PROBE_COUNT)[:, None]
        + torch.arange(VALUES_PER_PROBE)[None, :]
    )
    assert torch.equal(banks.probe_keys[0, :, 0], expected_keys.float())
    assert torch.equal(banks.atlas_values[0, :, :, 0], expected_values.float())
    assert torch.equal(
        banks.base_latents[0, :, 0], torch.arange(481, 737).float()
    )
    assert banks.boi[0, 0, 0].item() == 0.0
    assert banks.eoi[0, 0, 0].item() == 737.0

    base = reconstruct_base_v54_prefix_v81(memory)
    expected_base_indices = torch.cat(
        (torch.tensor([0]), torch.arange(481, 737), torch.tensor([737]))
    )
    assert base.shape == (1, BASE_PREFIX_TOKENS, HIDDEN_SIZE)
    assert torch.equal(base[0, :, 0], expected_base_indices.float())

    audit = audit_v75_v2_prefix_v81(memory)
    assert audit.exact_reconstruction is True
    assert audit.stage_a_positive_floor_value_count == 384
    assert audit.stage_a_dense_score_key_count == 96
    assert audit.all_738_tokens_claimed_strict_positive_payload_influence is False
    assert audit.question_dependent_retrieval is False
    assert audit.top_k_selection is False


def test_latest_user_query_exactly_matches_existing_question_helper() -> None:
    tokenizer = _QuestionTokenizer((2, 4, 9, 11))
    embedding = nn.Embedding(32, HIDDEN_SIZE)
    embedding.requires_grad_(False)
    device = torch.device("cpu")

    expected_ids = question_token_ids(tokenizer, "where now", device)
    result = latest_user_question_query_v81(
        tokenizer=tokenizer,
        embedding_layer=embedding,
        latest_user_question="where now",
        device=device,
        maximum_question_tokens=8,
        model_blob_sha256_identity=MODEL_BLOB_SHA256_IDENTITY,
        embedding_tensor_name=INPUT_EMBEDDING_TENSOR_NAME,
    )

    assert torch.equal(result.token_ids, expected_ids)
    assert torch.equal(
        result.query,
        embedding(expected_ids).detach().float().mean(dim=1),
    )
    assert result.query.shape == (1, HIDDEN_SIZE)
    assert result.query.dtype == torch.float32
    assert result.detached is True
    assert result.add_special_tokens is False
    assert result.included_system_prompt is False
    assert result.included_history is False
    assert result.included_answer is False
    assert all(call["add_special_tokens"] is False for call in tokenizer.calls)
    assert all(call["return_tensors"] == "pt" for call in tokenizer.calls)

    with pytest.raises(ValueError, match="model blob identity"):
        latest_user_question_query_v81(
            tokenizer=tokenizer,
            embedding_layer=embedding,
            latest_user_question="where now",
            device=device,
            maximum_question_tokens=8,
            model_blob_sha256_identity="0" * 64,
            embedding_tensor_name=INPUT_EMBEDDING_TENSOR_NAME,
        )
    embedding.requires_grad_(True)
    with pytest.raises(ValueError, match="frozen embedding"):
        latest_user_question_query_v81(
            tokenizer=tokenizer,
            embedding_layer=embedding,
            latest_user_question="where now",
            device=device,
            maximum_question_tokens=8,
            model_blob_sha256_identity=MODEL_BLOB_SHA256_IDENTITY,
            embedding_tensor_name=INPUT_EMBEDDING_TENSOR_NAME,
        )


def test_dense_stage_a_matches_hand_calculation_and_preserves_four_banks() -> None:
    memory = _memory()
    query = torch.randn(1, HIDDEN_SIZE, generator=torch.Generator().manual_seed(8102))
    binding = bind_fixed_prefix_before_question_v81(memory)
    result = deterministic_atlas_read_v81(memory, query, binding=binding)
    banks = split_v75_v2_prefix_v81(memory)

    logits = (
        torch.einsum(
            "bd,bpd->bp",
            F.normalize(query.float(), dim=-1),
            F.normalize(banks.probe_keys.float(), dim=-1),
        )
        * RAW_ATLAS_LOGIT_SCALE
    )
    weights = ATLAS_UNIFORM_FLOOR_MASS / PROBE_COUNT + (
        1.0 - ATLAS_UNIFORM_FLOOR_MASS
    ) * torch.softmax(logits, dim=-1)
    expected = torch.stack(
        [
            torch.einsum("bp,bph->bh", weights, banks.atlas_values[:, :, bank])
            for bank in range(VALUES_PER_PROBE)
        ],
        dim=1,
    )

    assert torch.equal(result.atlas_logits, logits)
    assert torch.equal(result.atlas_weights, weights)
    assert torch.equal(result.reconstructed_controls, expected)
    assert result.reconstructed_controls.shape == (1, 4, HIDDEN_SIZE)
    assert result.all_96_groups_positive is True
    assert result.all_384_values_receive_positive_floor_weight is True
    assert float(result.atlas_weights.min()) >= MINIMUM_ATLAS_WEIGHT
    assert torch.allclose(
        result.attention_sums, torch.ones(1), atol=1e-6, rtol=0.0
    )
    assert not torch.equal(
        result.reconstructed_controls[:, 0], result.reconstructed_controls[:, 1]
    )


def test_zero_memory_is_exact_zero_for_multiple_questions() -> None:
    memory = torch.zeros(1, FIXED_PREFIX_TOKENS, HIDDEN_SIZE)
    binding = bind_fixed_prefix_before_question_v81(memory)
    outputs = []
    for seed in (8103, 8104):
        question = torch.randn(
            1, HIDDEN_SIZE, generator=torch.Generator().manual_seed(seed)
        )
        outputs.append(
            deterministic_atlas_read_v81(memory, question, binding=binding)
        )

    expected_weights = torch.full((1, PROBE_COUNT), 1.0 / PROBE_COUNT)
    for output in outputs:
        assert torch.count_nonzero(output.reconstructed_controls).item() == 0
        assert torch.count_nonzero(output.atlas_logits).item() == 0
        assert torch.equal(output.atlas_weights, expected_weights)
    assert torch.equal(
        outputs[0].reconstructed_controls, outputs[1].reconstructed_controls
    )


def test_prequestion_binding_rejects_any_memory_mutation() -> None:
    memory = _memory(8105)
    before = memory.clone()
    binding = bind_fixed_prefix_before_question_v81(memory)
    for seed in (8106, 8107):
        query = torch.randn(
            1, HIDDEN_SIZE, generator=torch.Generator().manual_seed(seed)
        )
        result = deterministic_atlas_read_v81(memory, query, binding=binding)
        assert result.fixed_prefix_sha256 == binding.fixed_prefix_sha256
        assert result.atlas_memory_sha256 == binding.atlas_memory_sha256
        assert result.base_prefix_sha256 == binding.base_prefix_sha256
    assert torch.equal(memory, before)

    changed = memory.clone()
    changed[0, 2, 17] += 0.125
    with pytest.raises(ValueError, match="changed after prequestion binding"):
        assert_prefix_binding_v81(changed, binding=binding)
    with pytest.raises(ValueError, match="changed after prequestion binding"):
        deterministic_atlas_read_v81(
            changed,
            torch.randn(1, HIDDEN_SIZE),
            binding=binding,
        )


def _save_scene_memory(destination: Path, memory: torch.Tensor) -> dict[str, Any]:
    return save_v81_scene_memory(
        destination,
        memory,
        scene_id="scene_000001",
        source_base_checkpoint_sha256=_BASE_SHA,
        runtime_config_sha256=_RUNTIME_SHA,
        source_control_checkpoint_sha256=_CONTROL_SHA,
        source_probe_tensor_sha256=_PROBE_SHA,
    )


def _load_scene_memory(destination: Path, *, record_file: Any | None = None):
    return load_v81_scene_memory(
        destination,
        expected_scene_id="scene_000001",
        expected_base_checkpoint_sha256=_BASE_SHA,
        expected_runtime_config_sha256=_RUNTIME_SHA,
        expected_model_device="cpu",
        record_file=record_file,
    )


def test_scene_memory_artifact_is_exactly_two_numeric_files_and_self_sufficient(
    tmp_path: Path,
) -> None:
    memory = _memory(8108, dtype=torch.bfloat16)
    destination = tmp_path / "scene_000001"
    metadata = _save_scene_memory(destination, memory)
    opened: list[Path] = []
    loaded = _load_scene_memory(destination, record_file=lambda path: opened.append(path))

    assert {item.name for item in destination.iterdir()} == {
        MEMORY_FILENAME,
        METADATA_FILENAME,
    }
    assert torch.equal(loaded.memory, memory)
    assert metadata == loaded.metadata
    assert metadata["dtype"] == "torch.bfloat16"
    assert metadata["tensor_sha256"] == tensor_sha256(memory)
    assert metadata["canonical_prefix_sha256"] == prefix_sha256(memory)
    assert metadata["compiled_before_user_question"] is True
    assert metadata["question_inputs_used_for_compilation"] is False
    assert metadata["question_dependent_retrieval"] is False
    assert metadata["environmental_text_inputs"] == []
    assert metadata["questions_or_answers_serialized"] is False
    assert {path.name for path in opened} == {MEMORY_FILENAME, METADATA_FILENAME}
    assert tensor_sha256(memory) != tensor_sha256(memory.float())
    assert prefix_sha256(memory) == prefix_sha256(memory.float())


def test_scene_memory_artifact_rejects_inventory_metadata_and_tensor_tamper(
    tmp_path: Path,
) -> None:
    memory = _memory(8109, dtype=torch.bfloat16)

    extra_root = tmp_path / "extra" / "scene_000001"
    _save_scene_memory(extra_root, memory)
    (extra_root / "extra.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="exactly two files"):
        _load_scene_memory(extra_root)

    metadata_root = tmp_path / "metadata" / "scene_000001"
    _save_scene_memory(metadata_root, memory)
    metadata_path = metadata_root / METADATA_FILENAME
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["reader_logit_scale"] = 80.0
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="runtime contract changed"):
        _load_scene_memory(metadata_root)

    tensor_root = tmp_path / "tensor" / "scene_000001"
    _save_scene_memory(tensor_root, memory)
    tensor_path = tensor_root / MEMORY_FILENAME
    payload = bytearray(tensor_path.read_bytes())
    payload[-1] ^= 1
    tensor_path.write_bytes(payload)
    with pytest.raises(ValueError, match="file digest changed"):
        _load_scene_memory(tensor_root)


class _FakeTextModel:
    def get_per_layer_inputs(
        self, token_ids: torch.Tensor, _embeddings: torch.Tensor
    ) -> torch.Tensor:
        offsets = torch.arange(6, dtype=torch.float32).reshape(1, 1, 2, 3)
        return token_ids.float()[..., None, None] + offsets


class _FakeGemma(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(80, HIDDEN_SIZE)
        self.embedding.requires_grad_(False)
        text_config = SimpleNamespace(
            hidden_size=HIDDEN_SIZE,
            hidden_size_per_layer_input=3,
            num_hidden_layers=2,
            vocab_size=80,
            pad_token_id=0,
            bos_token_id=2,
            use_bidirectional_attention="vision",
        )
        self.config = SimpleNamespace(
            text_config=text_config,
            boi_token_id=58,
            image_token_id=60,
            eoi_token_id=59,
        )
        self.model = SimpleNamespace(language_model=_FakeTextModel())

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embedding


class _NativeTokenizer(_QuestionTokenizer):
    bos_token_id = 2
    pad_token_id = 0
    boi_token_id = 58
    image_token_id = 60
    eoi_token_id = 59


def test_v81_primary_prepared_order_and_pad_ple_modality_are_exact() -> None:
    model = _FakeGemma().eval()
    tokenizer = _NativeTokenizer((9, 13))
    backend = Gemma4PrefixBackend(
        model,
        tokenizer=tokenizer,
        model_revision="model-free-test-revision",
    )
    memory = _memory(8110)
    boi, eoi = backend.native_boundary_embeddings()
    memory[:, :1] = boi
    memory[:, -1:] = eoi
    binding = bind_fixed_prefix_before_question_v81(memory)
    latest = latest_user_question_query_v81(
        tokenizer=tokenizer,
        embedding_layer=model.get_input_embeddings(),
        latest_user_question="where now",
        device=torch.device("cpu"),
        maximum_question_tokens=8,
        model_blob_sha256_identity=MODEL_BLOB_SHA256_IDENTITY,
        embedding_tensor_name=INPUT_EMBEDDING_TENSOR_NAME,
    )
    controls = deterministic_atlas_read_v81(
        memory, latest.query, binding=binding
    ).reconstructed_controls
    base258 = reconstruct_base_v54_prefix_v81(memory)
    prompt_ids = torch.tensor([[2, 17, 19]], dtype=torch.long)
    prepared = backend.prepare(
        base258,
        prompt_ids,
        scene_prefix_after_bos=True,
        scene_boundary_mode=SCENE_BOUNDARY_MODE_GEMMA4_NATIVE_IMAGE,
        control_tokens=controls,
    )

    prompt_embeddings = model.get_input_embeddings()(prompt_ids)
    prompt_ple = model.model.language_model.get_per_layer_inputs(
        prompt_ids, prompt_embeddings
    )
    boundary_ids = torch.tensor([[58, 59]], dtype=torch.long)
    boundary_embeddings = model.get_input_embeddings()(boundary_ids)
    boundary_ple = model.model.language_model.get_per_layer_inputs(
        boundary_ids, boundary_embeddings
    )
    latent_pad_ids = torch.zeros((1, 256), dtype=torch.long)
    latent_pad_embeddings = model.get_input_embeddings()(latent_pad_ids)
    latent_pad_ple = model.model.language_model.get_per_layer_inputs(
        latent_pad_ids, latent_pad_embeddings
    )
    control_pad_ids = torch.zeros((1, 4), dtype=torch.long)
    control_pad_embeddings = model.get_input_embeddings()(control_pad_ids)
    control_pad_ple = model.model.language_model.get_per_layer_inputs(
        control_pad_ids, control_pad_embeddings
    )

    assert prepared.scene_prefix_length == 258
    assert prepared.inputs_embeds.shape == (1, 265, HIDDEN_SIZE)
    assert torch.equal(prepared.inputs_embeds[:, 0:1], prompt_embeddings[:, 0:1])
    assert torch.equal(prepared.inputs_embeds[:, 1:259], base258)
    assert torch.equal(prepared.inputs_embeds[:, 259:261], prompt_embeddings[:, 1:])
    assert torch.equal(prepared.inputs_embeds[:, 261:265], controls)
    assert torch.equal(prepared.per_layer_inputs[:, 0:1], prompt_ple[:, 0:1])
    assert torch.equal(prepared.per_layer_inputs[:, 1:2], boundary_ple[:, 0:1])
    assert torch.equal(prepared.per_layer_inputs[:, 2:258], latent_pad_ple)
    assert torch.equal(prepared.per_layer_inputs[:, 258:259], boundary_ple[:, 1:])
    assert torch.equal(prepared.per_layer_inputs[:, 259:261], prompt_ple[:, 1:])
    assert torch.equal(prepared.per_layer_inputs[:, 261:265], control_pad_ple)
    expected_modality = torch.zeros((1, 265), dtype=torch.long)
    expected_modality[:, 2:258] = 1
    assert torch.equal(prepared.mm_token_type_ids, expected_modality)
    assert torch.equal(prepared.attention_mask, torch.ones((1, 265), dtype=torch.long))
    assert prepared.labels is None
    assert_prefix_binding_v81(memory, binding=binding)


def test_frozen_lm_head_applies_exact_native_gemma_softcap() -> None:
    head = nn.Linear(HIDDEN_SIZE, 13, bias=False)
    with torch.no_grad():
        head.weight.fill_(0.01)
    head.requires_grad_(False)
    hidden = torch.full((2, HIDDEN_SIZE), 3.0, requires_grad=True)

    raw = head(hidden)
    expected = (
        torch.tanh(raw / FINAL_LOGIT_SOFTCAPPING) * FINAL_LOGIT_SOFTCAPPING
    )
    actual = frozen_lm_head_logits_v81(hidden, frozen_lm_head=head)

    assert torch.equal(actual, expected)
    assert not torch.equal(actual, raw)
    assert float(actual.detach().abs().max()) <= FINAL_LOGIT_SOFTCAPPING
    actual.sum().backward()
    assert hidden.grad is not None and float(hidden.grad.norm()) > 0.0
    assert head.weight.grad is None

    head.requires_grad_(True)
    with pytest.raises(ValueError, match="frozen parameters"):
        frozen_lm_head_logits_v81(hidden.detach(), frozen_lm_head=head)
