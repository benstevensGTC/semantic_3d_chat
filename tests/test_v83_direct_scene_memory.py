from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

from semantic_3d_chat.chat.v83_direct_scene_memory_cli import _parser
from semantic_3d_chat.chat.v83_direct_scene_memory_runtime import (
    DIRECT_PAYLOAD_TOKENS,
    RUNTIME_KIND,
    V83DirectSceneMemoryChatRuntime,
    audit_v83_direct_prepared_layout,
)
from semantic_3d_chat.evaluation.v83_direct_historical_behavior import (
    ARTIFACT,
    _shuffled_atlas_memory,
    _zero_payload_memory,
    predict,
)
from semantic_3d_chat.language.gemma4_backend import Gemma4PrefixBackend
from semantic_3d_chat.language.v81_structured_dense_atlas_sidecar import (
    reconstruct_base_v54_prefix_v81,
    split_v75_v2_prefix_v81,
)
from semantic_3d_chat.scene_encoder.v81_scene_memory_artifact import (
    HIDDEN_SIZE,
    LoadedV81SceneMemory,
    build_v81_scene_memory_metadata,
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
            hidden_size=HIDDEN_SIZE,
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
        torch.manual_seed(83)
        self.embedding = _ScaledEmbedding(16, HIDDEN_SIZE, scale=2.0)
        self.model = SimpleNamespace(language_model=_FakeTextModel())
        self.config = SimpleNamespace(
            text_config=self.model.language_model.config,
            boi_token_id=10,
            image_token_id=11,
            eoi_token_id=12,
        )

    def get_input_embeddings(self) -> nn.Module:
        return self.embedding


def _backend() -> Gemma4PrefixBackend:
    tokenizer = SimpleNamespace(
        bos_token_id=2,
        pad_token_id=0,
        boi_token_id=10,
        image_token_id=11,
        eoi_token_id=12,
    )
    return Gemma4PrefixBackend(
        _FakeGemma4().to(torch.bfloat16),
        tokenizer=tokenizer,
        model_revision="v83-fake-revision",
    )


def _memory(backend: Gemma4PrefixBackend, seed: int = 83) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    interior = torch.randn(
        1, DIRECT_PAYLOAD_TOKENS, HIDDEN_SIZE, generator=generator
    ) * 0.01
    boi, eoi = backend.native_boundary_embeddings()
    return torch.cat((boi, interior, eoi), dim=1).to(torch.bfloat16)


def _metadata(memory: torch.Tensor) -> dict[str, Any]:
    return build_v81_scene_memory_metadata(
        memory,
        scene_id="scene_000001",
        tensor_file_sha256="a" * 64,
        source_base_checkpoint_sha256="b" * 64,
        runtime_config_sha256="c" * 64,
        source_control_checkpoint_sha256="d" * 64,
        source_probe_tensor_sha256="e" * 64,
    )


def test_v83_exact_738_prefix_has_native_boundaries_pad_ple_and_no_controls() -> None:
    backend = _backend()
    memory = _memory(backend)
    prompt_ids = torch.tensor([[2, 4, 5]], dtype=torch.long)
    prepared = backend.prepare(
        memory,
        prompt_ids,
        scene_prefix_after_bos=True,
        scene_boundary_mode="gemma4_native_image",
        control_tokens=None,
    )

    audit = audit_v83_direct_prepared_layout(
        backend=backend,
        fixed_memory=memory,
        prompt_ids=prompt_ids,
        prepared=prepared,
    )

    assert audit["fixed_scene_memory_tokens_supplied_to_gemma"] == 738
    assert audit["continuous_environment_payload_tokens"] == 736
    assert audit["control_activation_tokens"] == 0
    assert audit["question_derived_environmental_tokens"] == 0
    assert audit["payload_pad_ple_exact"] is True
    assert audit["boi_eoi_native_ple_exact"] is True
    assert torch.equal(prepared.inputs_embeds[:, 1:739], memory)


def test_v83_layout_audit_rejects_payload_text_modality() -> None:
    backend = _backend()
    memory = _memory(backend)
    prompt_ids = torch.tensor([[2]], dtype=torch.long)
    prepared = backend.prepare(
        memory,
        prompt_ids,
        scene_prefix_after_bos=True,
        scene_boundary_mode="gemma4_native_image",
    )
    prepared.mm_token_type_ids[:, 2] = 0
    with pytest.raises(RuntimeError, match="image-modality IDs changed"):
        audit_v83_direct_prepared_layout(
            backend=backend,
            fixed_memory=memory,
            prompt_ids=prompt_ids,
            prepared=prepared,
        )


def test_v83_zero_and_shuffle_controls_have_exact_meaning() -> None:
    backend = _backend()
    memory = _memory(backend)
    banks = split_v75_v2_prefix_v81(memory)
    zero = _zero_payload_memory(memory)
    shuffled = _shuffled_atlas_memory(memory)
    zero_banks = split_v75_v2_prefix_v81(zero)
    shuffled_banks = split_v75_v2_prefix_v81(shuffled)

    assert torch.equal(zero[:, :1], memory[:, :1])
    assert torch.equal(zero[:, -1:], memory[:, -1:])
    assert torch.count_nonzero(zero[:, 1:-1]).item() == 0
    assert torch.equal(shuffled_banks.boi, banks.boi)
    assert torch.equal(shuffled_banks.eoi, banks.eoi)
    assert torch.equal(shuffled_banks.probe_keys, banks.probe_keys)
    assert torch.equal(shuffled_banks.base_latents, banks.base_latents)
    assert torch.equal(
        shuffled_banks.atlas_values, banks.atlas_values.roll(shifts=1, dims=1)
    )
    assert torch.count_nonzero(zero_banks.base_latents).item() == 0


class _BaseRuntime:
    def __init__(self, memory: torch.Tensor, backend: Gemma4PrefixBackend) -> None:
        self.config = {
            "language": {
                "scene_boundary_mode": "gemma4_native_image",
                "scene_prefix_after_bos": True,
            }
        }
        self.scene_id = "scene_000001"
        self.scene_prefix = reconstruct_base_v54_prefix_v81(memory)
        self.language = SimpleNamespace(
            prefix_backend=backend,
            backend_name="gemma4",
            device=torch.device("cpu"),
        )
        self.assertion_count = 0

    def assert_prefix_unchanged(self) -> None:
        self.assertion_count += 1

    def startup_summary(self) -> dict[str, Any]:
        return {"base_ready": True}


def test_v83_runtime_binds_and_preflights_before_any_question(tmp_path: Path) -> None:
    backend = _backend()
    memory = _memory(backend)
    loaded = LoadedV81SceneMemory(
        root=tmp_path / "scene_000001",
        memory=memory,
        metadata=_metadata(memory),
    )
    runtime = V83DirectSceneMemoryChatRuntime(  # type: ignore[arg-type]
        _BaseRuntime(memory, backend), loaded
    )

    startup = runtime.startup_summary()
    assert runtime.questions_answered == 0
    assert startup["runtime_kind"] == RUNTIME_KIND
    assert startup["exact_738_token_memory_supplied_directly_to_gemma"] is True
    assert startup["prefix_shape"] == [1, 738, HIDDEN_SIZE]
    assert startup["prefix_hash"] == runtime.scene_prefix_hash
    assert startup["environment_conditioned_input_sha256"] == runtime.scene_prefix_hash
    assert startup["question_derived_environmental_tokens"] == 0
    assert startup["reader_enabled"] is False
    assert startup["startup_layout_audit"]["payload_pad_ple_exact"] is True

    runtime.fixed_scene_memory[0, 8, 12] += 0.125
    with pytest.raises((RuntimeError, ValueError), match="changed after"):
        runtime.assert_prefix_unchanged()


def test_v83_cli_requires_memory_and_preserves_question_order() -> None:
    parser = _parser()
    args = parser.parse_args(
        [
            "--config",
            "runtime.yaml",
            "--scene",
            "scene_000001",
            "--base-checkpoint",
            "base",
            "--scene-memory",
            "memory",
            "--question",
            "first",
            "--question",
            "second",
        ]
    )
    assert args.question == ["first", "second"]
    assert args.scene_memory == "memory"


def test_v83_predictor_is_create_once_and_has_no_question_reader(tmp_path: Path) -> None:
    existing = tmp_path / "sealed.json"
    existing.write_text("preserve", encoding="utf-8")
    with pytest.raises(FileExistsError):
        predict(output_path=existing)
    assert existing.read_text(encoding="utf-8") == "preserve"

    runtime_source = Path(
        "src/semantic_3d_chat/chat/v83_direct_scene_memory_runtime.py"
    ).read_text(encoding="utf-8")
    predictor_source = Path(
        "src/semantic_3d_chat/evaluation/v83_direct_historical_behavior.py"
    ).read_text(encoding="utf-8")
    assert "latest_user_question_query" not in runtime_source
    assert "question_derived_environmental_tokens\": 0" in runtime_source
    assert "all_memories_compiled_and_bound_before_questions = True" in predictor_source
    assert predictor_source.index(
        "all_memories_compiled_and_bound_before_questions = True"
    ) < predictor_source.index("_load_predictor_questions(")
    assert '"behavioral_accuracy_scored_in_predictor": False' in predictor_source
    assert ARTIFACT == "v83_direct_historical_internal_predictions_v1"
