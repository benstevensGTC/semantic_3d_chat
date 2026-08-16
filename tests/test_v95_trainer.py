from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file
from torch import nn

from semantic_3d_chat.chat.runtime_config import load_runtime_config
from semantic_3d_chat.chat.v94_strict_multiscene_runtime import EXPECTED_BANKS
from semantic_3d_chat.evaluation.v95_strict_causal_successor_preflight import (
    FRESH_BANK_NAME,
    TARGET_MODULES,
    load_config_v95,
)
from semantic_3d_chat.language.lora import (
    LoRABankCollection,
    LoRABankSettings,
    LoRABanksSettings,
    LoRASettings,
    install_lora_banks,
)
from semantic_3d_chat.training.train_v95_strict_causal_successor import (
    CANDIDATE_ARTIFACT,
    EXPECTED_FROZEN_BANK_COUNT,
    EXPECTED_MICRO_ROWS,
    EXPECTED_OPTIMIZER_UPDATES,
    EXPECTED_PERMUTATION_ROWS,
    EXPECTED_TOTAL_NLL_FORWARDS,
    EXPECTED_WRONG_MEMORY_ROWS,
    EXPECTED_ZERO_ROWS,
    causal_objective_v95,
    combined_lora_settings_v95,
    discover_resume_checkpoint_v95,
    finalize_fixed_final_candidate_v95,
    load_fixed_final_bridge_v95,
    publish_fixed_final_candidate_v95,
    restore_resume_checkpoint_v95,
    save_resume_checkpoint_v95,
)


def test_v95_combined_settings_are_exact_v85_plus_failed_v94_plus_fresh() -> None:
    config = load_config_v95()
    runtime = load_runtime_config(config["sources"]["runtime_config"])
    settings = combined_lora_settings_v95(runtime, config)

    assert len(settings.banks) == 9
    assert tuple(bank.name for bank in settings.banks[:-1]) == EXPECTED_BANKS
    assert sum(not bank.trainable for bank in settings.banks) == 8
    assert sum(bank.trainable for bank in settings.banks) == 1
    assert settings.bank(EXPECTED_BANKS[-1]).adapter.target_modules == (
        "model.language_model.layers.34.mlp.gate_proj",
    )
    fresh = settings.bank(FRESH_BANK_NAME)
    assert fresh.adapter.target_modules == TARGET_MODULES
    assert fresh.adapter.rank == 8
    assert fresh.adapter.alpha == 16.0
    assert fresh.adapter.dropout == 0.0


def test_v95_four_term_objective_has_expected_gradients() -> None:
    correct = torch.tensor(2.0, requires_grad=True)
    wrong = torch.tensor(1.8, requires_grad=True)
    zero = torch.tensor(1.7, requires_grad=True)
    permutation = torch.tensor(1.6, requires_grad=True)

    objective, records = causal_objective_v95(
        correct,
        class_weight=1.5,
        balanced_ce_weight=1.0,
        wrong_memory_nll=wrong,
        wrong_margin_weight=1.25,
        wrong_target_margin=0.75,
        zero_payload_nll=zero,
        zero_margin_weight=0.75,
        zero_target_margin=0.5,
        permutation_nll=permutation,
        permutation_margin_weight=1.0,
        permutation_target_margin=0.75,
    )
    objective.backward()

    assert objective.item() == pytest.approx(5.9375)
    assert records["wrong_memory_minus_correct_nll"].item() == pytest.approx(-0.2)
    assert records["zero_payload_minus_correct_nll"].item() == pytest.approx(-0.3)
    assert records["permutation_minus_correct_nll"].item() == pytest.approx(-0.4)
    assert correct.grad is not None and correct.grad.item() == pytest.approx(4.5)
    assert wrong.grad is not None and wrong.grad.item() == pytest.approx(-1.25)
    assert zero.grad is not None and zero.grad.item() == pytest.approx(-0.75)
    assert permutation.grad is not None and permutation.grad.item() == pytest.approx(-1.0)


class _TinyAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.k_proj = nn.Linear(5, 3, bias=False)
        self.v_proj = nn.Linear(5, 3, bias=False)


class _TinyMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.up_proj = nn.Linear(5, 7, bias=False)


class _TinyLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = _TinyAttention()
        self.mlp = _TinyMLP()


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = nn.Module()
        layers: list[nn.Module] = [nn.Identity() for _ in range(35)]
        layers[9] = _TinyLayer()
        layers[34] = _TinyLayer()
        self.model.language_model.layers = nn.ModuleList(layers)


def _tiny_collection(seed: int = 95) -> LoRABankCollection:
    settings = LoRABanksSettings(
        (
            LoRABankSettings(
                name=FRESH_BANK_NAME,
                trainable=True,
                adapter=LoRASettings(
                    enabled=True,
                    rank=2,
                    alpha=4.0,
                    dropout=0.0,
                    target_modules=TARGET_MODULES,
                ),
                initialization_algorithm="cpu_kaiming_uniform_a_exact_zero_b",
                initialization_seed=seed,
            ),
        )
    )
    model = _TinyModel()
    model.requires_grad_(False)
    installed = install_lora_banks(model, settings)
    assert isinstance(installed, LoRABankCollection)
    return installed


def test_v95_resume_round_trip_restores_six_tensors_and_adamw(tmp_path: Path) -> None:
    first = _tiny_collection()
    optimizer = torch.optim.AdamW(first.parameters(), lr=0.001)
    loss = sum(parameter.square().mean() for parameter in first.parameters())
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    expected_hash = first.bank(FRESH_BANK_NAME).installation.state_sha256()
    bindings = {"config_sha256": "a" * 64}

    checkpoint = save_resume_checkpoint_v95(
        tmp_path,
        first,
        optimizer,
        update=1,
        row_cursor=8,
        history=[{"update": 1, "row_cursor": 8}],
        bindings=bindings,
        row_order_sha256="b" * 64,
    )
    discovered = discover_resume_checkpoint_v95(
        tmp_path,
        bindings=bindings,
        row_order_sha256="b" * 64,
        gradient_accumulation_rows=8,
    )
    assert discovered is not None and discovered[0] == checkpoint

    second = _tiny_collection(seed=96)
    second_optimizer = torch.optim.AdamW(second.parameters(), lr=0.001)
    restore_resume_checkpoint_v95(discovered[0], discovered[1], second, second_optimizer)
    assert second.bank(FRESH_BANK_NAME).installation.state_sha256() == expected_hash
    assert second_optimizer.state_dict()["state"]


def test_v95_resume_auth_rejects_extra_checkpoint_file(tmp_path: Path) -> None:
    collection = _tiny_collection()
    optimizer = torch.optim.AdamW(collection.parameters(), lr=0.001)
    loss = sum(parameter.square().mean() for parameter in collection.parameters())
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    checkpoint = save_resume_checkpoint_v95(
        tmp_path,
        collection,
        optimizer,
        update=1,
        row_cursor=8,
        history=[{"update": 1, "row_cursor": 8}],
        bindings={"config_sha256": "a" * 64},
        row_order_sha256="b" * 64,
    )
    (checkpoint / "unexpected.txt").write_text("tamper", encoding="utf-8")

    with pytest.raises(ValueError, match="file inventory changed"):
        discover_resume_checkpoint_v95(
            tmp_path,
            bindings={"config_sha256": "a" * 64},
            row_order_sha256="b" * 64,
            gradient_accumulation_rows=8,
        )


def test_v95_candidate_is_create_once_and_six_tensor_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from semantic_3d_chat.training import train_v95_strict_causal_successor as trainer

    collection = _tiny_collection()
    monkeypatch.setattr(trainer, "FRESH_PARAMETER_COUNT", 56)
    candidate = tmp_path / "candidate"
    metadata = publish_fixed_final_candidate_v95(
        candidate,
        collection,
        bindings={"config_sha256": "c" * 64, "fixed_final_optimizer_updates": 480},
    )

    tensors = load_file(str(candidate / "bridge.safetensors"))
    assert metadata["artifact"] == CANDIDATE_ARTIFACT
    assert metadata["parent"] == "fixed_final_nonpromoted_optimization_parent"
    assert len(metadata["tensor_inventory"]) == len(tensors) == 6
    assert set(metadata["tensor_inventory"]) == set(tensors)
    assert metadata["known_development_scored"] is False
    assert metadata["deferred_final_generated"] is False
    assert (
        json.loads((candidate / "runtime_metadata.json").read_text())[
            "questions_or_answers_serialized"
        ]
        is False
    )
    with pytest.raises(FileExistsError):
        publish_fixed_final_candidate_v95(candidate, collection, bindings={})

    reloaded = _tiny_collection(seed=97)
    loaded = load_fixed_final_bridge_v95(reloaded, candidate)
    assert loaded["state_sha256"] == metadata["state_sha256"]


def test_v95_interrupted_finalization_reuses_only_exact_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from semantic_3d_chat.training import train_v95_strict_causal_successor as trainer

    monkeypatch.setattr(trainer, "FRESH_PARAMETER_COUNT", 56)
    bindings = {"config_sha256": "d" * 64, "fixed_final_optimizer_updates": 480}
    first = _tiny_collection()
    candidate = tmp_path / "candidate"
    initial, reused = finalize_fixed_final_candidate_v95(candidate, first, bindings=bindings)
    assert reused is False

    resumed = _tiny_collection()
    resumed.bank(FRESH_BANK_NAME).installation.state_module.load_state_dict(
        first.bank(FRESH_BANK_NAME).installation.state_module.state_dict()
    )
    recovered, reused = finalize_fixed_final_candidate_v95(candidate, resumed, bindings=bindings)
    assert reused is True
    assert recovered["state_sha256"] == initial["state_sha256"]

    wrong = _tiny_collection(seed=1_095)
    with pytest.raises(ValueError, match="bindings changed"):
        finalize_fixed_final_candidate_v95(candidate, wrong, bindings=bindings)


def test_v95_trainer_constants_bind_exact_fixed_schedule() -> None:
    assert EXPECTED_FROZEN_BANK_COUNT == 8
    assert EXPECTED_MICRO_ROWS == 3_840
    assert EXPECTED_OPTIMIZER_UPDATES == 480
    assert EXPECTED_WRONG_MEMORY_ROWS == 996
    assert EXPECTED_ZERO_ROWS == 500
    assert EXPECTED_PERMUTATION_ROWS == 500
    assert EXPECTED_TOTAL_NLL_FORWARDS == 5_836
    assert 3_840 + 996 + 500 + 500 == EXPECTED_TOTAL_NLL_FORWARDS
