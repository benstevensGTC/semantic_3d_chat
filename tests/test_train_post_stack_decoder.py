from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from semantic_3d_chat.config import load_config
from semantic_3d_chat.data.dataset import QARecord
from semantic_3d_chat.language.lora import (
    install_lora_banks,
    lora_banks_settings,
)
from semantic_3d_chat.language.prefix_injection import ContinuousPrefixComposer
from semantic_3d_chat.training.train_post_stack_decoder import (
    ApprovedStageASource,
    CachedFullScenePrefix,
    StageBBundle,
    assert_frozen_stage_b_state,
    assert_stage_b_trainable_surface,
    cached_scene_answer_nll,
    freeze_for_stage_b,
    frozen_stage_b_state_sha256,
    require_approved_stage_a_source,
    stage_b_contract,
    stage_b_settings,
    verify_fresh_bank_update_zero,
)

CONFIG = "configs/experiments/gemma4_color_mirror_post_stack_decoder_stage_b_v28.yaml"
BANK = "extension_v28_stage_b_query"
EXPECTED_HASH = "a6d144a8507a0cf66e70fe9aae8bf2dcc916dc274612456bf5ad22132d5fc795"
TARGETS = (
    "model.language_model.layers.13.self_attn.q_proj",
    "model.language_model.layers.14.self_attn.q_proj",
)


class _ShapeOnlyLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int) -> None:
        nn.Module.__init__(self)
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.zeros(1, dtype=torch.float32).expand(out_features, in_features),
            requires_grad=False,
        )
        self.bias = None


class _ShapeOnlyAttention(nn.Module):
    def __init__(self, layer: int) -> None:
        super().__init__()
        if layer == 13:
            self.q_proj = _ShapeOnlyLinear(1536, 2048)
            self.k_proj = _ShapeOnlyLinear(1536, 256)
            self.v_proj = _ShapeOnlyLinear(1536, 256)
        if layer == 14:
            self.q_proj = _ShapeOnlyLinear(1536, 4096)
            self.k_proj = _ShapeOnlyLinear(1536, 512)
            self.v_proj = _ShapeOnlyLinear(1536, 512)
        if layer == 28:
            self.q_proj = _ShapeOnlyLinear(1536, 2048)
        if layer == 29:
            self.q_proj = _ShapeOnlyLinear(1536, 4096)
        if layer in (30, 31, 32, 33):
            self.q_proj = _ShapeOnlyLinear(1536, 2048)
            self.o_proj = _ShapeOnlyLinear(2048, 1536)
        if layer == 34:
            self.q_proj = _ShapeOnlyLinear(1536, 4096)
            self.o_proj = _ShapeOnlyLinear(4096, 1536)


class _ShapeOnlyLayer(nn.Module):
    def __init__(self, layer: int) -> None:
        super().__init__()
        self.self_attn = _ShapeOnlyAttention(layer)


class _ShapeOnlyGemma(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = nn.Module()
        self.model.language_model.layers = nn.ModuleList(
            _ShapeOnlyLayer(layer) for layer in range(35)
        )


class _TinyAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(4, 4, bias=False)


class _TinyLayer(nn.Module):
    def __init__(self, target: bool) -> None:
        super().__init__()
        self.self_attn = _TinyAttention() if target else nn.Identity()


class _TinyLanguageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = nn.Module()
        self.model.language_model.layers = nn.ModuleList(
            _TinyLayer(index in (13, 14)) for index in range(15)
        )
        self.embedding = nn.Embedding(6, 4)
        self.output = nn.Linear(4, 6, bias=False)

    def get_input_embeddings(self) -> nn.Module:
        return self.embedding

    def forward(self, *, inputs_embeds, attention_mask, labels, use_cache):
        del attention_mask, use_cache
        layers = self.model.language_model.layers
        hidden = layers[13].self_attn.q_proj(inputs_embeds)
        hidden = hidden + layers[14].self_attn.q_proj(inputs_embeds)
        logits = self.output(hidden.cumsum(dim=1))
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
        ids = [2, 3] if text.startswith("yes") else [4, 3]
        return SimpleNamespace(input_ids=torch.tensor([ids], dtype=torch.long))


def _tiny_config() -> dict:
    return {
        "language": {
            "backend": "gemma4",
            "system_prompt": "stable",
            "lora_banks": {
                BANK: {
                    "trainable": True,
                    "rank": 4,
                    "alpha": 8.0,
                    "dropout": 0.0,
                    "target_modules": list(TARGETS),
                    "initialization_algorithm": (
                        "cpu_kaiming_uniform_a_exact_zero_b"
                    ),
                    "initialization_seed": 29029,
                    "expected_initial_state_sha256": (
                        "31f821c62617d95cda772862a0b9b049667462a37f6a062083b352183bd82d3a"
                    ),
                }
            },
        },
        "training": {"lora_learning_rate": 1e-4, "lora_weight_decay": 0.0},
    }


def _tiny_bundle() -> StageBBundle:
    model = _TinyLanguageModel().requires_grad_(False)
    collection = install_lora_banks(model, lora_banks_settings(_tiny_config()))
    assert collection is not None
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
    scene = nn.Linear(4, 4)
    composer = ContinuousPrefixComposer(4)
    grounding = nn.Linear(4, 4)
    dense = nn.Linear(4, 4)
    sidecar = nn.Linear(4, 4)
    global_residual = nn.Linear(4, 4)
    signed = nn.Linear(4, 4)
    modules = {
        "scene_model": scene,
        "composer": composer,
        "grounding": grounding,
        "dense_aligner": dense,
        "dense_sidecar_adapter": sidecar,
        "global_scene_residual": global_residual,
        "signed_x_scene_residual": signed,
        **collection.state_modules(),
    }
    source = ApprovedStageASource(
        checkpoint=Path("/tmp/update_001"),
        selection_report=Path("/tmp/selection.json"),
        selection_sha256="0" * 64,
        selected_update=1,
        selected_arm={"eligible": True},
    )
    return StageBBundle(
        config=_tiny_config(),
        source_config={},
        source_runtime_metadata={},
        source_training_metadata={},
        source=source,
        language=language,
        scene_model=scene,
        dense_aligner=dense,
        dense_sidecar_adapter=sidecar,
        global_scene_residual=global_residual,
        signed_x_scene_residual=signed,
        composer=composer,
        grounding=grounding,
        lora_installation=collection,
        checkpoint_modules=modules,
        frozen_checkpoint_modules={
            name: module
            for name, module in modules.items()
            if name != f"lora_banks.{BANK}"
        },
        trainable_bank_name=BANK,
    )


def _record(scene_id: str = "scene_000001") -> QARecord:
    return QARecord(
        scene_id=scene_id,
        question_id="q_000001",
        question="Is it present?",
        answer="yes",
        answer_type="presence",
        target_xyz=None,
    )


def _approval_config(tmp_path: Path) -> tuple[dict, Path, Path]:
    root = tmp_path / "stage_a"
    checkpoint = root / "update_001"
    checkpoint.mkdir(parents=True)
    for name in ("adapter.safetensors", "metadata.json", "runtime_metadata.json"):
        (checkpoint / name).write_bytes(b"checkpoint")
    report_path = tmp_path / "selection.json"
    report = {
        "schema_version": 1,
        "artifact": "v28_post_stack_sidecar_stage_a_selection",
        "training_evaluation_only": True,
        "question_text_serialized": False,
        "answer_text_serialized": False,
        "oracle_loaded": False,
        "question_dependent_scene_processing": False,
        "passed": True,
        "selected_checkpoint": str(checkpoint),
        "selected_update": 1,
        "arms": [
            {
                "checkpoint": str(checkpoint),
                "update": 1,
                "eligible": True,
            }
        ],
    }
    report_path.write_text(json.dumps(report), encoding="utf-8")
    config = {
        "v28_stage_b": {
            "schema_version": 1,
            "role": "selector_gated_broad_qa_decoder_adaptation",
            "source_config": "unused.yaml",
            "stage_a_selection_report": str(report_path),
            "stage_a_checkpoint_root": str(root),
            "new_bank": BANK,
            "new_bank_parameter_count": 36_864,
            "new_bank_initial_state_sha256": EXPECTED_HASH,
            "update_zero_validation_nll_absolute_tolerance": 1e-7,
            "selection_requires": {
                "color_full_vocab_sides": 12,
                "mirror_full_vocab_sides": 10,
                "no_new_negative_sides": True,
                "validation_nll_must_improve": True,
            },
        }
    }
    return config, report_path, checkpoint


def test_v28_stage_b_config_is_one_exact_fresh_query_bank() -> None:
    config = load_config(CONFIG)
    settings = stage_b_settings(config)
    contract = stage_b_contract(config)
    banks = lora_banks_settings(config)
    fresh = banks.bank(BANK)

    assert settings.enabled is True
    assert settings.max_optimizer_steps == 4
    assert settings.gradient_accumulation == 12
    assert settings.trainable_bank == BANK
    assert contract["new_bank_parameter_count"] == 36_864
    assert contract["new_bank_initial_state_sha256"] == EXPECTED_HASH
    assert [bank.name for bank in banks.banks if bank.trainable] == [BANK]
    assert fresh.adapter.rank == 4
    assert fresh.adapter.alpha == 8.0
    assert fresh.adapter.target_modules == TARGETS
    assert not (set(TARGETS) & set(banks.bank("extension_v23_shared_kv").adapter.target_modules))


def test_makefile_exposes_selector_gated_stage_b_without_changing_stage_a() -> None:
    makefile = Path("Makefile").read_text(encoding="utf-8")

    assert "gemma4-v28-train-stage-b: gemma4-v28-select-stage-a" in makefile
    assert "gemma4-v28-select-stage-b: gemma4-v28-train-stage-b" in makefile
    assert "gemma4-v28-evaluate-stage-b: gemma4-v28-select-stage-b" in makefile
    assert "semantic_3d_chat.training.train_post_stack_decoder" in makefile
    assert "semantic_3d_chat.evaluation.v28_stage_b_selector" in makefile


def test_v28_fresh_bank_has_exact_parameter_count_hash_and_zero_output() -> None:
    config = load_config(CONFIG)
    first = install_lora_banks(_ShapeOnlyGemma(), lora_banks_settings(config))
    second = install_lora_banks(_ShapeOnlyGemma(), lora_banks_settings(config))
    assert first is not None and second is not None
    bank = first.bank(BANK).installation

    assert bank.parameter_count == 36_864
    assert bank.state_sha256() == EXPECTED_HASH
    assert second.bank(BANK).installation.state_sha256() == EXPECTED_HASH
    assert all(torch.count_nonzero(adapter.lora_b).item() == 0 for adapter in bank.adapters)


def test_stage_b_requires_unique_nonzero_selector_approved_stage_a(tmp_path: Path) -> None:
    config, report_path, checkpoint = _approval_config(tmp_path)

    approved = require_approved_stage_a_source(config)

    assert approved.checkpoint == checkpoint.resolve()
    assert approved.selected_update == 1
    assert len(approved.selection_sha256) == 64

    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["passed"] = False
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="did not approve"):
        require_approved_stage_a_source(config)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("update_zero", "nonzero selected"),
        ("ineligible", "unique eligible"),
        ("outside_root", "outside its contracted root"),
    ],
)
def test_stage_b_rejects_invalid_stage_a_selector_decisions(
    tmp_path: Path, mutation: str, message: str
) -> None:
    config, report_path, checkpoint = _approval_config(tmp_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if mutation == "update_zero":
        zero = checkpoint.parent / "update_000"
        checkpoint.rename(zero)
        report["selected_checkpoint"] = str(zero)
        report["selected_update"] = 0
        report["arms"][0].update(checkpoint=str(zero), update=0)
    elif mutation == "ineligible":
        report["arms"][0]["eligible"] = False
    else:
        outside = tmp_path / "outside" / "update_001"
        outside.parent.mkdir()
        checkpoint.rename(outside)
        report["selected_checkpoint"] = str(outside)
        report["arms"][0]["checkpoint"] = str(outside)
    report_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        require_approved_stage_a_source(config)


def test_stage_b_freezes_everything_except_fresh_bank_and_hashes_frozen_state() -> None:
    bundle = _tiny_bundle()
    parameters = freeze_for_stage_b(bundle)
    audit = assert_stage_b_trainable_surface(bundle)
    frozen_hash = frozen_stage_b_state_sha256(bundle)

    assert audit["bank"] == BANK
    assert audit["parameter_count"] == 64
    assert {id(value) for value in parameters} == {
        id(value) for value in bundle.lora_installation.parameters()
    }
    assert all(not value.requires_grad for value in bundle.language.model.embedding.parameters())

    with torch.no_grad():
        parameters[-1].add_(0.01)
    assert_frozen_stage_b_state(bundle, frozen_hash)

    with torch.no_grad():
        bundle.scene_model.weight[0, 0].add_(0.01)
    with pytest.raises(RuntimeError, match="Frozen selected Stage-A"):
        assert_frozen_stage_b_state(bundle, frozen_hash)


def test_update_zero_is_bit_exact_and_rejects_a_nonzero_route() -> None:
    bundle = _tiny_bundle()
    freeze_for_stage_b(bundle)
    bank = bundle.lora_installation.bank(BANK).installation
    audit = verify_fresh_bank_update_zero(
        bundle,
        expected_hash=bank.state_sha256(),
        expected_parameter_count=bank.parameter_count,
    )

    assert audit["verified"] is True
    assert all(audit["target_outputs_bit_exact"].values())

    with torch.no_grad():
        bank.adapters[0].lora_b[0, 0] = 0.01
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_fresh_bank_update_zero(
            bundle,
            expected_hash=audit["state_sha256"],
            expected_parameter_count=bank.parameter_count,
        )


def test_cached_scene_answer_nll_trains_only_fresh_bank() -> None:
    bundle = _tiny_bundle()
    freeze_for_stage_b(bundle)
    cache = CachedFullScenePrefix(
        scene_id="scene_000001",
        scene_tokens=torch.randn(1, 2, 4),
        prefix_sha256="0" * 64,
        voxel_count=7,
        processed_voxels=7,
        minimum_voxel_contribution=0.1,
    )

    loss = cached_scene_answer_nll(cache=cache, records=[_record()], bundle=bundle)
    loss.backward()

    fresh = bundle.lora_installation.bank(BANK).installation
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert all(parameter.grad is not None for parameter in fresh.parameters())
    assert any(
        torch.count_nonzero(adapter.lora_b.grad).item() > 0
        for adapter in fresh.adapters
    )
    assert all(parameter.grad is None for parameter in bundle.scene_model.parameters())
    assert all(parameter.grad is None for parameter in bundle.language.model.embedding.parameters())

    with pytest.raises(ValueError, match="match its cached scene"):
        cached_scene_answer_nll(
            cache=cache,
            records=[_record("scene_000002")],
            bundle=bundle,
        )
