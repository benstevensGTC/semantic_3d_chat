from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from semantic_3d_chat.data.dataset import QARecord
from semantic_3d_chat.language.prefix_injection import ContinuousPrefixComposer
from semantic_3d_chat.scene_encoder.dense_sidecar_adapter import DenseSidecarAdapter
from semantic_3d_chat.training.train_post_stack_sidecar import (
    CachedSceneTokens,
    StageABundle,
    answer_token_nll,
    assert_frozen_stage_a_state,
    assert_stage_a_trainable_surface,
    cache_scene_output,
    freeze_for_stage_a,
    frozen_stage_a_state_sha256,
    records_by_scene,
    stage_a_settings,
)


def _sidecar() -> DenseSidecarAdapter:
    return DenseSidecarAdapter(
        scene_dim=4,
        latent_count=2,
        width=3,
        fourier_bands=1,
        max_direct_scale=0.2,
        initialization_seed=28,
    )


class _TinyLanguageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(6, 4)
        self.projection = nn.Linear(4, 6, bias=False)

    def get_input_embeddings(self) -> nn.Module:
        return self.embedding

    def forward(self, *, inputs_embeds, attention_mask, labels, use_cache):
        del attention_mask, use_cache
        logits = self.projection(inputs_embeds.cumsum(dim=1))
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]),
            labels[:, 1:].reshape(-1),
            ignore_index=-100,
        )
        return SimpleNamespace(loss=loss, logits=logits)


class _Tokenizer:
    eos_token = "<eos>"

    def apply_chat_template(self, *_args, **_kwargs) -> torch.Tensor:
        return torch.tensor([[1]], dtype=torch.long)

    def __call__(self, text: str, **_kwargs) -> SimpleNamespace:
        if text.startswith("yes"):
            ids = [2, 3]
        elif text.startswith("no"):
            ids = [4, 3]
        else:
            ids = [1]
        return SimpleNamespace(input_ids=torch.tensor([ids], dtype=torch.long))


def _bundle() -> StageABundle:
    model = _TinyLanguageModel()
    language = SimpleNamespace(
        model=model,
        tokenizer=_Tokenizer(),
        device=torch.device("cpu"),
        forward_prefix_batch=lambda batch, use_cache=False: model(
            inputs_embeds=batch.inputs_embeds,
            attention_mask=batch.attention_mask,
            labels=batch.labels,
            use_cache=use_cache,
        ),
    )
    sidecar = _sidecar()
    composer = ContinuousPrefixComposer(4)
    scene = nn.Linear(4, 4)
    grounding = nn.Linear(4, 4)
    dense = nn.Linear(4, 4)
    global_residual = nn.Linear(4, 4)
    signed = nn.Linear(4, 4)
    modules = {
        "scene_model": scene,
        "composer": composer,
        "grounding": grounding,
        "dense_aligner": dense,
        "global_scene_residual": global_residual,
        "signed_x_scene_residual": signed,
        "dense_sidecar_adapter": sidecar,
    }
    return StageABundle(
        config={"language": {"system_prompt": "stable"}},
        candidate_metadata={
            "dense_sidecar_adapter_parameter_count": sidecar.parameter_count
        },
        language=language,
        scene_model=scene,
        dense_aligner=dense,
        dense_sidecar_adapter=sidecar,
        global_scene_residual=global_residual,
        signed_x_scene_residual=signed,
        composer=composer,
        grounding=grounding,
        lora_installation=None,
        checkpoint_modules=modules,
        frozen_checkpoint_modules={
            name: module for name, module in modules.items() if name != "dense_sidecar_adapter"
        },
    )


def _record(
    scene_id: str = "scene_000001",
    question_id: str = "q_000001",
    answer: str = "yes",
) -> QARecord:
    return QARecord(
        scene_id=scene_id,
        question_id=question_id,
        question="Is it present?",
        answer=answer,
        answer_type="presence",
        target_xyz=None,
    )


def test_stage_a_settings_match_bounded_mac_contract() -> None:
    config = {
        "training": {
            "post_stack_sidecar_stage_a": {
                "enabled": True,
                "max_optimizer_steps": 4,
                "evaluation_interval_steps": 1,
                "batch_size": 1,
                "gradient_accumulation": 12,
                "learning_rate": 1e-4,
                "channel_gain_learning_rate": 2e-4,
                "weight_decay": 0.0,
                "gradient_clip_norm": 1.0,
                "minimum_answer_types": 4,
                "trainable_routes": ["output_projection", "channel_gain"],
            }
        }
    }

    settings = stage_a_settings(config)

    assert settings.enabled is True
    assert settings.max_optimizer_steps == 4
    assert settings.evaluation_interval_steps == 1
    assert settings.gradient_accumulation == 12
    assert settings.output_projection_learning_rate == 1e-4
    assert settings.channel_gain_learning_rate == 2e-4
    assert settings.trainable_routes == ("output_projection", "channel_gain")

    bad = {
        "training": {
            "post_stack_sidecar_stage_a": {
                "trainable_routes": ["output_projection", "base_projection"]
            }
        }
    }
    with pytest.raises(ValueError, match="output_projection and channel_gain"):
        stage_a_settings(bad)
    with pytest.raises(ValueError, match="Unknown"):
        stage_a_settings(
            {"training": {"post_stack_sidecar_stage_a": {"surprise": True}}}
        )


def test_freeze_contract_opens_only_output_projection_and_channel_gain() -> None:
    bundle = _bundle()

    output_projection, channel_gain = freeze_for_stage_a(bundle)
    audit = assert_stage_a_trainable_surface(bundle)

    assert output_projection is bundle.dense_sidecar_adapter.output_projection.weight
    assert channel_gain is bundle.dense_sidecar_adapter.channel_gain
    assert audit["total_trainable_parameters"] == output_projection.numel() + 4
    assert all(not parameter.requires_grad for parameter in bundle.language.model.parameters())
    assert all(
        not parameter.requires_grad
        for name, parameter in bundle.dense_sidecar_adapter.named_parameters()
        if name not in {"output_projection.weight", "channel_gain"}
    )

    bundle.dense_sidecar_adapter.base_projection.weight.requires_grad_(True)
    with pytest.raises(RuntimeError, match="trainable surface mismatch"):
        assert_stage_a_trainable_surface(bundle)


def test_frozen_hash_excludes_authorized_outputs_but_detects_hidden_drift() -> None:
    bundle = _bundle()
    freeze_for_stage_a(bundle)
    expected = frozen_stage_a_state_sha256(bundle)

    with torch.no_grad():
        bundle.dense_sidecar_adapter.output_projection.weight.add_(0.01)
        bundle.dense_sidecar_adapter.channel_gain.add_(0.02)
    assert_frozen_stage_a_state(bundle, expected)

    with torch.no_grad():
        bundle.dense_sidecar_adapter.sidecar_projection.weight[0, 0].add_(0.01)
    with pytest.raises(RuntimeError, match="Frozen V24/V26"):
        assert_frozen_stage_a_state(bundle, expected)


def test_cache_scene_output_proves_update0_identity_and_all_voxel_coverage() -> None:
    sidecar = _sidecar()
    composer = ContinuousPrefixComposer(4)
    base = torch.randn(1, 2, 4)
    aligned = torch.randn(1, 2, 4)
    output = SimpleNamespace(
        scene_tokens=base,
        aligned_sidecar_tokens=aligned,
        audit={
            "processed_voxels": torch.tensor(17),
            "aligned_sidecar_processed_voxels": torch.tensor(17),
            "aligned_sidecar_min_voxel_contribution": torch.tensor(0.01),
        },
    )

    cached = cache_scene_output(
        scene_id="scene_000001",
        output=output,
        voxel_count=17,
        composer=composer,
        sidecar=sidecar,
    )

    assert cached.scene_id == "scene_000001"
    assert cached.processed_voxels == cached.voxel_count == 17
    assert torch.equal(cached.base_scene_tokens, base)
    assert torch.equal(cached.aligned_sidecar_tokens, aligned)
    assert len(cached.base_prefix_sha256) == 64

    incomplete = SimpleNamespace(
        scene_tokens=base,
        aligned_sidecar_tokens=aligned,
        audit={
            **output.audit,
            "aligned_sidecar_processed_voxels": torch.tensor(16),
        },
    )
    with pytest.raises(RuntimeError, match="Incomplete scene cache"):
        cache_scene_output(
            scene_id="scene_000001",
            output=incomplete,
            voxel_count=17,
            composer=composer,
            sidecar=sidecar,
        )


def test_cache_rejects_nonzero_update_as_update0_equivalence() -> None:
    sidecar = _sidecar()
    with torch.no_grad():
        sidecar.channel_gain.fill_(0.1)
    output = SimpleNamespace(
        scene_tokens=torch.randn(1, 2, 4),
        aligned_sidecar_tokens=torch.randn(1, 2, 4),
        audit={
            "processed_voxels": torch.tensor(5),
            "aligned_sidecar_processed_voxels": torch.tensor(5),
            "aligned_sidecar_min_voxel_contribution": torch.tensor(0.1),
        },
    )

    with pytest.raises(RuntimeError, match="update-0 adapter changed"):
        cache_scene_output(
            scene_id="scene_000001",
            output=output,
            voxel_count=5,
            composer=ContinuousPrefixComposer(4),
            sidecar=sidecar,
        )


def test_answer_token_nll_backpropagates_only_from_continuous_cached_scene() -> None:
    bundle = _bundle()
    freeze_for_stage_a(bundle)
    cache = CachedSceneTokens(
        scene_id="scene_000001",
        base_scene_tokens=torch.randn(1, 2, 4),
        aligned_sidecar_tokens=torch.randn(1, 2, 4),
        base_prefix_sha256="0" * 64,
        voxel_count=7,
        processed_voxels=7,
        minimum_voxel_contribution=0.1,
    )

    loss = answer_token_nll(
        cache=cache,
        records=[_record()],
        adapter=bundle.dense_sidecar_adapter,
        language=bundle.language,
        composer=bundle.composer,
        config=bundle.config,
    )
    loss.backward()

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert bundle.dense_sidecar_adapter.output_projection.weight.grad is not None
    assert torch.count_nonzero(
        bundle.dense_sidecar_adapter.output_projection.weight.grad
    ).item() > 0
    assert bundle.dense_sidecar_adapter.channel_gain.grad is not None
    assert torch.count_nonzero(bundle.dense_sidecar_adapter.channel_gain.grad).item() > 0
    assert all(parameter.grad is None for parameter in bundle.language.model.parameters())

    with pytest.raises(ValueError, match="match the cached scene"):
        answer_token_nll(
            cache=cache,
            records=[_record(scene_id="scene_000002")],
            adapter=bundle.dense_sidecar_adapter,
            language=bundle.language,
            composer=bundle.composer,
            config=bundle.config,
        )


def test_records_are_grouped_by_opaque_scene_without_question_selection() -> None:
    records = [
        _record("scene_000002", "q_000002", "no"),
        _record("scene_000001", "q_000003", "yes"),
        _record("scene_000002", "q_000001", "yes"),
    ]

    grouped = records_by_scene(records)

    assert list(grouped) == ["scene_000001", "scene_000002"]
    assert grouped["scene_000001"] == [records[1]]
    assert grouped["scene_000002"] == [records[0], records[2]]
