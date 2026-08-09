from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from semantic_3d_chat.config import load_config
from semantic_3d_chat.data.dataset import QARecord
from semantic_3d_chat.training.pair_curriculum import CounterfactualPairUnit
from semantic_3d_chat.training.train_adapter import (
    deduplicate_spatial_relation_warmup_units,
    pair_spatial_relation_contrastive_objective,
    run_spatial_relation_warmup,
    spatial_relation_contrastive_settings,
    spatial_relation_resume_contract_mismatch,
    spatial_relation_target_audit,
    spatial_relation_warmup_resume_contract_mismatch,
    spatial_relation_warmup_settings,
    spatial_relation_warmup_target_audit,
)


def _record(
    scene_id: str,
    role: str,
    answer: str,
    target_xyz: list[float],
    reference_xyz: list[float],
    question_key: str = "relation",
) -> QARecord:
    return QARecord(
        scene_id=scene_id,
        question_id=f"{question_key}_{role}",
        question="Is the target left or right of the reference?",
        answer=answer,
        answer_type="spatial_relation",
        target_xyz=target_xyz,
        reference_xyz=reference_xyz,
        counterfactual_pair_id="pair_1",
        counterfactual_question_key=question_key,
        counterfactual_expected_change=True,
        counterfactual_role=role,
        counterfactual_change_type="mirror_lr",
    )


def _unit(question_key: str = "relation") -> CounterfactualPairUnit:
    return CounterfactualPairUnit(
        "pair_1",
        question_key,
        _record(
            "scene_a",
            "reference",
            "left",
            [-0.75, 0.0, 0.0],
            [0.75, 0.0, 0.0],
            question_key,
        ),
        _record(
            "scene_b",
            "counterfactual",
            "right",
            [0.75, 0.0, 0.0],
            [-0.75, 0.0, 0.0],
            question_key,
        ),
    )


class _Tokenizer:
    eos_token_id = 2

    def __call__(self, text: str, **kwargs) -> SimpleNamespace:
        assert kwargs["add_special_tokens"] is False
        return SimpleNamespace(input_ids=torch.tensor([[{"left": 0, "right": 1}[text]]]))


class _NoDecoderModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(3, 2)
        self.decoder_calls = 0
        with torch.no_grad():
            self.embedding.weight.copy_(torch.tensor([[-1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]))

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embedding

    def forward(self, *args, **kwargs):  # pragma: no cover - call is the failure
        self.decoder_calls += 1
        raise AssertionError("ordered-relation warmup must not call the decoder")


def _language() -> SimpleNamespace:
    return SimpleNamespace(
        model=_NoDecoderModel(),
        tokenizer=_Tokenizer(),
        device=torch.device("cpu"),
    )


def _map(scene_index: float) -> SimpleNamespace:
    return SimpleNamespace(
        semantic=torch.tensor([scene_index]),
        xyz=torch.zeros(1, 3),
        rgb=torch.zeros(1, 3),
        normal=torch.zeros(1, 3),
        confidence=torch.ones(1),
        observation_count=torch.ones(1),
        room_min=torch.tensor([-1.0, -1.0, -1.0]),
        room_max=torch.tensor([1.0, 1.0, 1.0]),
    )


def test_balanced_shared_candidate_preference_has_zero_hinge_gradient() -> None:
    shared_preference = torch.tensor(0.25, requires_grad=True)
    margins = torch.stack((shared_preference, -shared_preference))
    loss = torch.relu(1.0 - margins).mean()

    loss.backward()

    assert float(loss.detach()) == pytest.approx(1.0)
    assert float(shared_preference.grad) == pytest.approx(0.0)


def test_relation_settings_default_off_and_resume_contract_is_explicit() -> None:
    config = load_config("configs/default.yaml")
    disabled = spatial_relation_contrastive_settings(config)
    warmup_disabled = spatial_relation_warmup_settings(config)

    assert disabled == {"weight": 0.0, "margin": 0.1, "temperature": 0.2}
    assert warmup_disabled == {
        "steps": 0,
        "learning_rate": 0.001,
        "margin_target": 0.1,
        "temperature": 0.2,
        "gradient_clip_norm": 1.0,
    }
    assert spatial_relation_resume_contract_mismatch({}, disabled) is None
    enabled = {**disabled, "weight": 16.0}
    assert spatial_relation_resume_contract_mismatch({}, enabled) == {
        "checkpoint": None,
        "runtime": enabled,
    }

    config["training"]["spatial_relation_contrastive_margin"] = -0.1
    with pytest.raises(ValueError, match="margin must be"):
        spatial_relation_contrastive_settings(config)


def test_relation_target_audit_and_warmup_fingerprint_deduplicate_phrasing() -> None:
    units = [_unit("phrasing_a"), _unit("phrasing_b")]

    selected = deduplicate_spatial_relation_warmup_units(units)
    audit = spatial_relation_target_audit(units)
    warmup_audit = spatial_relation_warmup_target_audit(units)

    assert len(selected) == 1
    assert audit == {
        "eligible_unit_count": 2,
        "eligible_side_count": 4,
        "unique_ordered_region_count": 2,
        "side_to_unique_ordered_region_ratio": 2.0,
    }
    assert warmup_audit["deduplicated_unit_count"] == 1
    assert warmup_audit["eligible_side_count"] == 2
    assert len(warmup_audit["target_sha256"]) == 64


def test_pair_relation_bridge_breaks_balanced_saddle_and_reaches_both_scenes() -> None:
    language = _language()
    # The initial relation lives on Y while left/right answer directions live
    # on X, so both balanced sides have an active non-cancelling spatial loss.
    scene_a = torch.tensor(
        [[[0.0, -1.0], [0.0, -0.25], [0.0, 0.25], [0.0, 1.0]]],
        requires_grad=True,
    )
    scene_b = scene_a.detach().clone().requires_grad_()
    outputs = {
        "scene_a": SimpleNamespace(scene_tokens=scene_a),
        "scene_b": SimpleNamespace(scene_tokens=scene_b),
    }
    maps = {"scene_a": _map(0.0), "scene_b": _map(1.0)}

    loss, diagnostics = pair_spatial_relation_contrastive_objective(
        outputs,
        [_unit()],
        maps,
        language,
        temperature=0.2,
        margin=0.1,
    )
    loss.backward()

    assert float(loss.detach()) == pytest.approx(0.1)
    assert diagnostics["eligible_unit_count"] == 1
    assert diagnostics["unique_ordered_region_count"] == 2
    assert diagnostics["achieved_margin"].tolist() == pytest.approx([0.0, 0.0])
    assert scene_a.grad is not None and float(scene_a.grad.abs().sum()) > 0
    assert scene_b.grad is not None and float(scene_b.grad.abs().sum()) > 0
    assert language.model.embedding.weight.grad is None
    assert bool((diagnostics["target_weights"] > 0).all())
    assert bool((diagnostics["reference_weights"] > 0).all())


class _TinySceneModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        initial = torch.tensor(
            [
                [[0.0, -1.0], [0.0, -0.25], [0.0, 0.25], [0.0, 1.0]],
                [[0.0, -1.0], [0.0, -0.25], [0.0, 0.25], [0.0, 1.0]],
            ]
        )
        self.tokens = nn.Parameter(initial)
        self.forward_counts = [0, 0]

    def forward(self, semantic, xyz, rgb, normal, confidence, observation_count, *bounds):
        scene_index = int(semantic.item())
        self.forward_counts[scene_index] += 1
        scene_tokens = self.tokens[scene_index : scene_index + 1]
        return SimpleNamespace(
            scene_tokens=scene_tokens,
            native_latents=scene_tokens,
            block_tokens=scene_tokens,
            audit={},
        )


def test_relation_warmup_is_scene_only_and_resume_requires_completed_metrics() -> None:
    scene_model = _TinySceneModel()
    language = _language()
    before = scene_model.tokens.detach().clone()
    settings = {
        "steps": 1,
        "learning_rate": 0.1,
        "margin_target": 0.1,
        "temperature": 0.2,
        "gradient_clip_norm": 1.0,
    }

    metrics = run_spatial_relation_warmup(
        scene_model,
        {"scene_a": _map(0.0), "scene_b": _map(1.0)},
        [_unit()],
        language,
        settings=settings,
    )

    assert language.model.decoder_calls == 0
    assert scene_model.forward_counts == [1, 1]
    assert not torch.equal(scene_model.tokens.detach(), before)
    assert metrics["optimizer_steps"] == 1
    assert metrics["final"]["eligible_side_count"] == 2
    assert language.model.embedding.weight.grad is None
    assert all(parameter.grad is None for parameter in scene_model.parameters())

    audit = spatial_relation_warmup_target_audit([_unit()])
    mismatch = spatial_relation_warmup_resume_contract_mismatch(
        {"spatial_relation_warmup": settings}, settings, audit
    )
    assert mismatch is not None
    assert "target_audit" in mismatch
    assert "metrics" in mismatch
    assert (
        spatial_relation_warmup_resume_contract_mismatch(
            {
                "spatial_relation_warmup": settings,
                "spatial_relation_warmup_target_audit": audit,
                "spatial_relation_warmup_metrics": {"completed": True},
            },
            settings,
            audit,
        )
        is None
    )
