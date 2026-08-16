from __future__ import annotations

import copy
import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from semantic_3d_chat.config import load_config
from semantic_3d_chat.data.dataset import QARecord
from semantic_3d_chat.language.lora import lora_banks_settings, tensor_state_sha256
from semantic_3d_chat.language.prefix_injection import ContinuousPrefixComposer
from semantic_3d_chat.scene_encoder.dense_sidecar_adapter import DenseSidecarAdapter
from semantic_3d_chat.training.pair_curriculum import build_exact_question_pair_units
from semantic_3d_chat.training.train_joint_pair_v30 import (
    CachedPreSidecarScene,
    _source_prefix_provenance,
    build_v30_cycle,
    paired_canonical_answer_objective,
    require_approved_v29_source,
    select_balanced_broad_records,
    v30_contract,
    v30_settings,
)

V30_CONFIG = Path("configs/experiments/gemma4_diverse20_joint_pair_v30.yaml")
V29_CONFIG = Path("configs/experiments/gemma4_diverse20_post_stack_decoder_stage_b_v29.yaml")


def _record(
    scene: str,
    question_id: str,
    answer_type: str,
    *,
    answer: str = "yes",
    pair_id: str | None = None,
    question_key: str | None = None,
    role: str | None = None,
) -> QARecord:
    return QARecord(
        scene_id=scene,
        question_id=question_id,
        question="Which side?" if pair_id else f"Question {question_id}?",
        answer=answer,
        answer_type=answer_type,
        target_xyz=None,
        counterfactual_pair_id=pair_id,
        counterfactual_question_key=question_key,
        counterfactual_expected_change=(True if pair_id else None),
        counterfactual_role=role,
        counterfactual_change_type=("test" if pair_id else None),
    )


def _pair_records() -> list[QARecord]:
    return [
        _record(
            "scene_000011",
            "qa",
            "spatial_relation",
            answer="left",
            pair_id="pair_a",
            question_key="key_a",
            role="reference",
        ),
        _record(
            "scene_000012",
            "qb",
            "spatial_relation",
            answer="right",
            pair_id="pair_a",
            question_key="key_a",
            role="counterfactual",
        ),
    ]


def test_v30_is_separate_and_pins_the_joint_surface() -> None:
    v30 = load_config(V30_CONFIG)
    v29 = load_config(V29_CONFIG)
    settings = v30_settings(v30)
    contract = v30_contract(v30)
    banks = lora_banks_settings(v30)

    assert v29.get("v30_joint_pair") is None
    assert settings.pair_repeats_per_cycle == 3
    assert settings.pair_margin == 0.5
    assert settings.pair_margin_weight == 4.0
    assert settings.broad_nll_weight == 1.0
    assert contract["joint_trainable_parameter_count"] == 329_216
    assert contract["sidecar_trainable_parameter_count"] == 198_144
    assert contract["fresh_bank_parameter_count"] == 131_072
    assert contract["promotion_requires"]["validation_changed_complete_pairs_minimum"] == 6
    assert (
        contract["promotion_requires"]["aggregate_validation_exact_accuracy_no_regression"] is True
    )
    assert banks.bank("extension_v28_stage_b_query").trainable is False
    fresh = banks.bank("extension_v30_joint_pair_query")
    assert fresh.trainable is True
    assert fresh.adapter.rank == 8
    assert fresh.adapter.alpha == 16.0
    assert fresh.adapter.target_modules == tuple(
        f"model.language_model.layers.{index}.self_attn.q_proj" for index in range(18, 22)
    )


def test_v30_fresh_bank_hash_is_reproducible_without_loading_gemma() -> None:
    generator = torch.Generator(device="cpu").manual_seed(30030)
    state: dict[str, torch.Tensor] = {}
    for index, output_size in enumerate((2048, 4096, 2048, 2048)):
        lora_a = torch.empty((8, 1536), dtype=torch.float32)
        torch.nn.init.kaiming_uniform_(lora_a, a=math.sqrt(5), generator=generator)
        state[f"adapters.{index}.lora_a"] = lora_a
        state[f"adapters.{index}.lora_b"] = torch.zeros((output_size, 8), dtype=torch.float32)

    assert sum(tensor.numel() for tensor in state.values()) == 131_072
    assert tensor_state_sha256(state) == (
        "2b1d89fbb9189ac551bf12905cf94036ebaa84696449b31c2b37b69d478fb70d"
    )


def test_v30_contract_rejects_surface_and_promotion_relaxation() -> None:
    config = load_config(V30_CONFIG)
    tampered = copy.deepcopy(config)
    tampered["v30_joint_pair"]["joint_trainable_parameter_count"] += 1
    with pytest.raises(ValueError, match="not the sum"):
        v30_contract(tampered)

    tampered = copy.deepcopy(config)
    tampered["v30_joint_pair"]["promotion_requires"][
        "aggregate_validation_exact_accuracy_no_regression"
    ] = False
    with pytest.raises(ValueError, match="forbid aggregate"):
        v30_contract(tampered)


def test_v30_source_is_the_selector_approved_v29_update() -> None:
    source = require_approved_v29_source(load_config(V30_CONFIG))

    assert source.selected_update == 4
    assert source.checkpoint.name == "update_004"
    assert source.selected_arm["eligible"] is True
    assert source.selection_sha256 == (
        "d7acbd7173f079f257619510df36ad3c73f953e7cf0123b7bd383ad01ddfe91a"
    )


def test_source_prefix_provenance_keeps_old_scenes_pinned_and_new_scenes_exact() -> None:
    historical = "a" * 64
    derived = "b" * 64

    assert (
        _source_prefix_provenance(
            scene_id="scene_000019",
            source_prefix=historical,
            expected_prefixes={"scene_000019": historical},
            allowed_unpinned=set(),
        )
        == "historically_pinned"
    )
    assert (
        _source_prefix_provenance(
            scene_id="scene_000031",
            source_prefix=derived,
            expected_prefixes={"scene_000019": historical},
            allowed_unpinned={"scene_000031"},
            repeated_source_prefix=derived,
        )
        == "deterministically_derived"
    )

    with pytest.raises(RuntimeError, match="expected=None"):
        _source_prefix_provenance(
            scene_id="scene_000031",
            source_prefix=derived,
            expected_prefixes={"scene_000019": historical},
            allowed_unpinned=set(),
            repeated_source_prefix=derived,
        )
    with pytest.raises(RuntimeError, match="nondeterministic"):
        _source_prefix_provenance(
            scene_id="scene_000031",
            source_prefix=derived,
            expected_prefixes={"scene_000019": historical},
            allowed_unpinned={"scene_000031"},
            repeated_source_prefix="c" * 64,
        )


def test_balanced_broad_selection_is_deterministic_and_excludes_changed_rows() -> None:
    records = [
        _record(f"scene_{11 + index % 4:06d}", f"q{index}", answer_type)
        for index, answer_type in enumerate(
            ("presence", "count", "attribute", "spatial_relation") * 4
        )
    ]
    records.extend(_pair_records())

    selected = select_balanced_broad_records(
        records, count=8, seed=30, exclude_expected_change=True
    )

    assert selected == select_balanced_broad_records(
        records, count=8, seed=30, exclude_expected_change=True
    )
    assert len(selected) == len({record.question_id for record in selected}) == 8
    assert {record.answer_type for record in selected} == {
        "presence",
        "count",
        "attribute",
        "spatial_relation",
    }
    assert all(record.counterfactual_expected_change is not True for record in selected)


def test_every_v30_cycle_keeps_pair_units_atomic_and_oversampled() -> None:
    records = [
        _record(f"scene_{11 + index % 4:06d}", f"q{index}", answer_type)
        for index, answer_type in enumerate(
            ("presence", "count", "attribute", "spatial_relation") * 4
        )
    ]
    records.extend(_pair_records())
    units = build_exact_question_pair_units(records)
    settings = replace(
        v30_settings(load_config(V30_CONFIG)),
        broad_questions_per_cycle=8,
        pair_repeats_per_cycle=3,
    )

    broad_batches, pair_batches, audit = build_v30_cycle(records, units, settings=settings, seed=31)

    assert sum(len(batch) for _scene, batch in broad_batches) == 8
    assert len(pair_batches) == 3
    assert all(len(batch) == 1 and batch[0] == units[0] for batch in pair_batches)
    assert audit["pair_units_atomic"] is True
    assert audit["every_pair_unit_present_each_cycle"] is True
    assert audit["pair_side_presentations"] == 6


class _TinyTokenizer:
    eos_token = "<eos>"

    def apply_chat_template(self, *args, **kwargs) -> torch.Tensor:
        del args, kwargs
        return torch.tensor([[0]])

    def __call__(self, text: str, **kwargs) -> SimpleNamespace:
        del kwargs
        if text.startswith("left"):
            ids = [1, 3]
        elif text.startswith("right"):
            ids = [2, 3]
        else:
            ids = [0]
        return SimpleNamespace(input_ids=torch.tensor([ids]))


class _TinyCausalModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(4, 2)
        self.projection = torch.nn.Linear(2, 4, bias=False)
        self.forward_calls = 0
        torch.nn.init.zeros_(self.embedding.weight)
        torch.nn.init.zeros_(self.projection.weight)
        with torch.no_grad():
            self.projection.weight[1, 0] = 1.0
            self.projection.weight[2, 0] = -1.0

    def get_input_embeddings(self) -> torch.nn.Module:
        return self.embedding

    def forward(self, *, inputs_embeds, attention_mask, use_cache):
        del attention_mask, use_cache
        self.forward_calls += 1
        return SimpleNamespace(logits=self.projection(inputs_embeds.cumsum(dim=1)))


def test_paired_objective_scores_correct_and_swapped_answers_and_trains_sidecar() -> None:
    unit = build_exact_question_pair_units(_pair_records())[0]
    model = _TinyCausalModel()
    composer = ContinuousPrefixComposer(2)
    with torch.no_grad():
        composer.scene_start.zero_()
        composer.scene_end.zero_()
    sidecar = DenseSidecarAdapter(
        scene_dim=2,
        latent_count=1,
        width=2,
        fourier_bands=1,
        max_direct_scale=0.25,
        initialization_seed=30,
    )
    language = SimpleNamespace(
        model=model,
        tokenizer=_TinyTokenizer(),
        device=torch.device("cpu"),
    )
    bundle = SimpleNamespace(
        dense_sidecar_adapter=sidecar,
        language=language,
        composer=composer,
        config={"language": {"system_prompt": "stable"}},
    )
    caches = {
        "scene_000011": CachedPreSidecarScene(
            "scene_000011",
            torch.tensor([[[0.2, 0.0]]]),
            torch.tensor([[[0.4, -0.2]]]),
            "a" * 64,
            1,
            1,
            1.0,
        ),
        "scene_000012": CachedPreSidecarScene(
            "scene_000012",
            torch.tensor([[[-0.2, 0.0]]]),
            torch.tensor([[[-0.4, 0.2]]]),
            "b" * 64,
            1,
            1,
            1.0,
        ),
    }

    language_nll, margin_loss, diagnostics = paired_canonical_answer_objective(
        units=[unit], caches=caches, bundle=bundle, margin=0.5
    )

    assert diagnostics["margins"].shape == (1, 2)
    assert torch.all(diagnostics["margins"] > 0)
    assert model.forward_calls == 2
    (language_nll + 4.0 * margin_loss).backward()
    assert sidecar.output_projection.weight.grad is not None
    assert sidecar.output_projection.weight.grad.abs().sum() > 0
    assert sidecar.channel_gain.grad is not None
    assert sidecar.channel_gain.grad.abs().sum() > 0


def test_makefile_exposes_v30_development_without_final_test_target() -> None:
    makefile = Path("Makefile").read_text(encoding="utf-8")

    assert "gemma4-v30-train-joint-pair" in makefile
    assert "gemma4-v30-select-joint-pair" in makefile
    assert "gemma4-v30-evaluate-final-test" not in makefile
