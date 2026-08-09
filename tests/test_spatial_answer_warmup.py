from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from semantic_3d_chat.config import load_config
from semantic_3d_chat.data.dataset import QARecord
from semantic_3d_chat.training.pair_curriculum import CounterfactualPairUnit
from semantic_3d_chat.training.train_adapter import (
    deduplicate_spatial_answer_warmup_units,
    gradient_accumulation_resume_contract_mismatch,
    run_spatial_answer_warmup,
    spatial_answer_warmup_resume_contract_mismatch,
    spatial_answer_warmup_settings,
    spatial_answer_warmup_target_audit,
)


def test_gradient_accumulation_resume_contract_preserves_legacy_default() -> None:
    assert gradient_accumulation_resume_contract_mismatch({}, 1) is None
    assert gradient_accumulation_resume_contract_mismatch({"gradient_accumulation": 6}, 6) is None
    assert gradient_accumulation_resume_contract_mismatch({}, 6) == {
        "checkpoint": "<legacy-default:1>",
        "runtime": 6,
    }
    assert gradient_accumulation_resume_contract_mismatch(
        {"gradient_accumulation": 1}, 6
    ) == {"checkpoint": 1, "runtime": 6}
    with pytest.raises(ValueError, match="must be positive"):
        gradient_accumulation_resume_contract_mismatch({}, 0)


def _record(
    scene_id: str,
    question_id: str,
    role: str,
    answer: str,
    target_xyz: list[float],
) -> QARecord:
    return QARecord(
        scene_id=scene_id,
        question_id=question_id,
        question="What color is the target?",
        answer=answer,
        answer_type="attribute",
        target_xyz=target_xyz,
        counterfactual_pair_id="pair_1",
        counterfactual_question_key=question_id,
        counterfactual_expected_change=True,
        counterfactual_role=role,
        counterfactual_change_type="color_swap",
    )


def _unit(question_id: str, target_xyz: list[float] | None = None) -> CounterfactualPairUnit:
    target = [0.0, 0.0, 0.0] if target_xyz is None else target_xyz
    return CounterfactualPairUnit(
        "pair_1",
        question_id,
        _record("scene_a", question_id, "reference", "red", target),
        _record("scene_b", question_id, "counterfactual", "blue", target),
    )


class _NoDecoderModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(3, 3)
        self.decoder_calls = 0
        with torch.no_grad():
            self.embedding.weight.copy_(torch.eye(3))

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embedding

    def forward(self, *args, **kwargs):  # pragma: no cover - failure proves isolation
        self.decoder_calls += 1
        raise AssertionError("warmup must not call the language decoder")


class _Tokenizer:
    eos_token_id = 2

    def __call__(self, text: str, **kwargs) -> SimpleNamespace:
        assert kwargs["add_special_tokens"] is False
        return SimpleNamespace(input_ids=torch.tensor([[{"red": 0, "blue": 1}[text]]]))


class _TinySceneModel(nn.Module):
    def __init__(self, *, already_aligned: bool = False) -> None:
        super().__init__()
        initial = torch.zeros(2, 1, 3)
        if already_aligned:
            initial[0, 0, 0] = 1.0
            initial[1, 0, 1] = 1.0
        else:
            initial[0, 0, 1] = 1.0
            initial[1, 0, 0] = 1.0
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


def _maps() -> dict[str, SimpleNamespace]:
    tensors = {
        "xyz": torch.zeros(1, 3),
        "rgb": torch.zeros(1, 3),
        "normal": torch.zeros(1, 3),
        "confidence": torch.ones(1),
        "observation_count": torch.ones(1),
        "room_min": torch.tensor([-1.0, -1.0, -1.0]),
        "room_max": torch.tensor([1.0, 1.0, 1.0]),
    }
    return {
        "scene_a": SimpleNamespace(semantic=torch.tensor([0.0]), **tensors),
        "scene_b": SimpleNamespace(semantic=torch.tensor([1.0]), **tensors),
    }


def _language() -> SimpleNamespace:
    return SimpleNamespace(
        model=_NoDecoderModel(),
        tokenizer=_Tokenizer(),
        device=torch.device("cpu"),
    )


def test_warmup_defaults_are_off_and_v4_isolated_from_v3() -> None:
    default = spatial_answer_warmup_settings(load_config("configs/default.yaml"))
    v3 = load_config("configs/experiments/gemma4_color_wiring_v3.yaml")
    v4 = load_config("configs/experiments/gemma4_color_wiring_v4.yaml")
    v5 = load_config("configs/experiments/gemma4_color_wiring_v5.yaml")
    v6 = load_config("configs/experiments/gemma4_color_wiring_v6.yaml")
    v7 = load_config("configs/experiments/gemma4_color_wiring_v7.yaml")

    assert default == {
        "steps": 0,
        "learning_rate": 0.001,
        "margin_target": 0.10,
        "gradient_clip_norm": 1.0,
    }
    assert spatial_answer_warmup_settings(v3) == default
    assert spatial_answer_warmup_settings(v4) == {**default, "steps": 50}
    assert v3["training"]["output_namespace"] == "gemma4_color_wiring_v3"
    assert v4["training"]["output_namespace"] == "gemma4_color_wiring_v4"
    assert v5["training"]["output_namespace"] == "gemma4_color_wiring_v5"
    assert v5["language"]["scene_prefix_after_bos"] is False
    assert v6["training"]["output_namespace"] == "gemma4_color_wiring_v6"
    assert v6["language"]["scene_prefix_after_bos"] is True
    assert v7["training"]["output_namespace"] == "gemma4_color_wiring_v7"
    assert v7["language"]["scene_prefix_after_bos"] is True
    assert v7["language"]["scene_boundary_mode"] == "gemma4_native_image"
    assert v7["language"]["gemma4_native_image_contract"]["boi_token_id"] == 255999
    assert v7["language"]["gemma4_native_image_contract"]["eoi_token_id"] == 258882
    assert v5["training"]["gradient_accumulation"] == 6
    assert v5["training"]["pair_units_per_batch"] == 1
    assert v5["training"]["pair_steps_per_epoch"] == 6
    assert v5["training"]["epochs"] == v4["training"]["epochs"] == 12


def test_warmup_deduplicates_paraphrases_by_physical_target() -> None:
    duplicate_units = [_unit("paraphrase_a"), _unit("paraphrase_b")]
    second_target = _unit("another_target", [0.5, 0.0, 0.0])
    third_target = _unit("third_target", [-0.5, 0.0, 0.0])

    all_units = [*duplicate_units, second_target, third_target]
    selected = deduplicate_spatial_answer_warmup_units(all_units)
    audit = spatial_answer_warmup_target_audit(all_units)

    assert {unit.question_key for unit in selected} == {
        "paraphrase_a",
        "another_target",
        "third_target",
    }
    assert audit["deduplicated_unit_count"] == 3
    assert audit["eligible_side_count"] == 6
    assert audit["unique_target_count"] == 3
    assert len(audit["target_sha256"]) == 64


def test_warmup_updates_only_scene_model_without_decoder_and_forwards_each_scene_once() -> None:
    scene_model = _TinySceneModel()
    language = _language()
    before = scene_model.tokens.detach().clone()
    composer = nn.Linear(3, 3)
    grounding = nn.Linear(3, 3)
    composer_before = {
        name: value.detach().clone() for name, value in composer.state_dict().items()
    }
    grounding_before = {
        name: value.detach().clone() for name, value in grounding.state_dict().items()
    }
    # This mirrors main(): the long-lived optimizer can already exist, but it
    # has no moments until its own LM/candidate objective takes a step.
    main_optimizer = torch.optim.AdamW(
        [*scene_model.parameters(), *composer.parameters(), *grounding.parameters()], lr=1e-3
    )
    settings = {
        "steps": 1,
        "learning_rate": 0.1,
        "margin_target": 0.1,
        "gradient_clip_norm": 1.0,
    }

    metrics = run_spatial_answer_warmup(
        scene_model,
        _maps(),
        [_unit("paraphrase_a"), _unit("paraphrase_b")],
        language,
        latent_count=1,
        settings=settings,
    )

    assert language.model.decoder_calls == 0
    assert scene_model.forward_counts == [1, 1]
    assert not torch.equal(scene_model.tokens.detach(), before)
    assert metrics["optimizer_steps"] == 1
    assert metrics["target_audit"]["deduplicated_unit_count"] == 1
    assert metrics["final"]["eligible_side_count"] == 2
    assert all(parameter.grad is None for parameter in scene_model.parameters())
    assert language.model.embedding.weight.grad is None
    assert not main_optimizer.state
    for name, value in composer.state_dict().items():
        assert torch.equal(value, composer_before[name])
    for name, value in grounding.state_dict().items():
        assert torch.equal(value, grounding_before[name])
    assert all(parameter.grad is None for parameter in composer.parameters())
    assert all(parameter.grad is None for parameter in grounding.parameters())


def test_warmup_stops_before_an_update_when_every_unique_side_meets_margin() -> None:
    scene_model = _TinySceneModel(already_aligned=True)
    language = _language()
    before = scene_model.tokens.detach().clone()

    metrics = run_spatial_answer_warmup(
        scene_model,
        _maps(),
        [_unit("target")],
        language,
        latent_count=1,
        settings={
            "steps": 10,
            "learning_rate": 0.1,
            "margin_target": 0.1,
            "gradient_clip_norm": 1.0,
        },
    )

    assert metrics["forward_steps"] == 1
    assert metrics["optimizer_steps"] == 0
    assert metrics["stopped_early"] is True
    assert metrics["final"]["side_success_rate"] == pytest.approx(1.0)
    assert scene_model.forward_counts == [1, 1]
    assert torch.equal(scene_model.tokens.detach(), before)


def test_warmup_resume_contract_accepts_legacy_off_and_requires_completed_enabled_run() -> None:
    disabled = spatial_answer_warmup_settings(load_config("configs/default.yaml"))
    enabled = {**disabled, "steps": 50}
    audit = spatial_answer_warmup_target_audit([_unit("target")])
    completed = {"completed": True, "optimizer_steps": 4}

    assert spatial_answer_warmup_resume_contract_mismatch({}, disabled, audit) is None
    assert (
        spatial_answer_warmup_resume_contract_mismatch(
            {
                "spatial_answer_warmup": enabled,
                "spatial_answer_warmup_target_audit": audit,
                "spatial_answer_warmup_metrics": completed,
            },
            enabled,
            audit,
        )
        is None
    )
    mismatch = spatial_answer_warmup_resume_contract_mismatch(
        {"spatial_answer_warmup": enabled}, enabled, audit
    )
    assert mismatch is not None
    assert "target_audit" in mismatch
    assert "metrics" in mismatch
