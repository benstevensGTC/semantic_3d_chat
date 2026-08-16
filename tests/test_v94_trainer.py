from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file
from torch import nn

from semantic_3d_chat.language.lora import (
    LoRABankCollection,
    LoRABankSettings,
    LoRABanksSettings,
    LoRASettings,
    install_lora_banks,
)
from semantic_3d_chat.training.train_v94_strict_multiscene_full40 import (
    CANDIDATE_ARTIFACT,
    EXPECTED_FRESH_PARAMETER_COUNT,
    FRESH_BANK_NAME,
    TARGET_MODULE,
    combined_lora_settings_v94,
    discover_resume_checkpoint_v94,
    load_fixed_final_bridge_v94,
    multiscene_objective_v94,
    publish_fixed_final_candidate_v94,
    restore_resume_checkpoint_v94,
    save_resume_checkpoint_v94,
)

_INIT_SHA = "7d413bc8bf02accb8d870a56e38de383baba6f7028eda54b1283f7994df71628"


def _experiment() -> dict[str, object]:
    return {
        "bridge": {
            "rank": 8,
            "alpha": 16.0,
            "dropout": 0.0,
            "target_module": TARGET_MODULE,
            "initialization_algorithm": "cpu_kaiming_uniform_a_exact_zero_b",
            "initialization_seed": 940094,
            "expected_initial_state_sha256": _INIT_SHA,
        }
    }


def _runtime() -> dict[str, object]:
    banks = {}
    # Model-free setting construction needs the seven-bank count, not their
    # production module paths.  Each bank is still frozen and exact-path-only.
    for index in range(7):
        banks[f"frozen_{index}"] = {
            "trainable": False,
            "rank": 1,
            "alpha": 1.0,
            "dropout": 0.0,
            "initialization_algorithm": "checkpoint_overwrite",
            "initialization_seed": None,
            "expected_initial_state_sha256": f"{index + 1:064x}",
            "target_modules": [f"frozen.layer_{index}"],
        }
    return {"language": {"backend": "gemma4", "lora_banks": banks}}


def test_v94_combined_settings_are_clean_v85_continuation() -> None:
    settings = combined_lora_settings_v94(_runtime(), _experiment())

    assert len(settings.banks) == 8
    assert sum(bank.trainable for bank in settings.banks) == 1
    fresh = settings.bank(FRESH_BANK_NAME)
    assert fresh.adapter.target_modules == (TARGET_MODULE,)
    assert fresh.adapter.rank == 8
    assert fresh.adapter.alpha == 16.0
    assert fresh.initialization_seed == 940094
    assert fresh.expected_initial_state_sha256 == _INIT_SHA
    assert all(
        not bank.name.startswith(f"v{version}")
        for version in range(86, 94)
        for bank in settings.banks
    )


def test_v94_dual_margin_objective_has_expected_gradients() -> None:
    correct = torch.tensor(2.0, requires_grad=True)
    wrong = torch.tensor(1.8, requires_grad=True)
    zero = torch.tensor(1.7, requires_grad=True)

    objective, records = multiscene_objective_v94(
        correct,
        class_weight=1.5,
        answer_ce_weight=1.0,
        paired_wrong_nll=wrong,
        paired_margin_weight=1.0,
        paired_target_margin=0.5,
        zero_payload_nll=zero,
        zero_margin_weight=1.0,
        zero_target_margin=0.5,
    )
    objective.backward()

    assert objective.item() == pytest.approx(4.5)
    assert records["paired_wrong_minus_correct_nll"].item() == pytest.approx(-0.2)
    assert records["paired_margin_penalty"].item() == pytest.approx(0.7)
    assert records["zero_minus_correct_nll"].item() == pytest.approx(-0.3)
    assert records["zero_margin_penalty"].item() == pytest.approx(0.8)
    assert correct.grad is not None and correct.grad.item() == pytest.approx(3.5)
    assert wrong.grad is not None and wrong.grad.item() == pytest.approx(-1.0)
    assert zero.grad is not None and zero.grad.item() == pytest.approx(-1.0)


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = nn.Module()
        self.model.language_model.layers = nn.ModuleList(
            [nn.Identity() for _ in range(34)] + [nn.Module()]
        )
        layer = self.model.language_model.layers[34]
        layer.mlp = nn.Module()
        layer.mlp.gate_proj = nn.Linear(5, 3, bias=False)


def _tiny_collection(seed: int = 94) -> LoRABankCollection:
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
                    target_modules=(TARGET_MODULE,),
                ),
                initialization_algorithm="cpu_kaiming_uniform_a_exact_zero_b",
                initialization_seed=seed,
            ),
        )
    )
    installed = install_lora_banks(_TinyModel(), settings)
    assert isinstance(installed, LoRABankCollection)
    return installed


def test_v94_authenticated_resume_restores_two_tensors_and_adamw(tmp_path: Path) -> None:
    first = _tiny_collection()
    optimizer = torch.optim.AdamW(first.parameters(), lr=0.001)
    loss = sum(parameter.square().mean() for parameter in first.parameters())
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    expected_hash = first.bank(FRESH_BANK_NAME).installation.state_sha256()
    bindings = {"config_sha256": "a" * 64}

    checkpoint = save_resume_checkpoint_v94(
        tmp_path,
        first,
        optimizer,
        update=1,
        row_cursor=8,
        history=[{"update": 1}],
        bindings=bindings,
        row_order_sha256="b" * 64,
    )
    discovered = discover_resume_checkpoint_v94(
        tmp_path,
        bindings=bindings,
        row_order_sha256="b" * 64,
        gradient_accumulation_rows=8,
    )
    assert discovered is not None and discovered[0] == checkpoint

    second = _tiny_collection()
    second_optimizer = torch.optim.AdamW(second.parameters(), lr=0.001)
    restore_resume_checkpoint_v94(discovered[0], discovered[1], second, second_optimizer)
    assert second.bank(FRESH_BANK_NAME).installation.state_sha256() == expected_hash
    assert second_optimizer.state_dict()["state"]


def test_v94_candidate_is_create_once_and_two_tensor_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from semantic_3d_chat.training import train_v94_strict_multiscene_full40 as trainer

    # The production fixed-final publisher's metadata assumes the production
    # 110,592-parameter bridge. Replace only that numerical check for this tiny
    # state round-trip; serialization and authentication remain unchanged.
    monkeypatch.setattr(trainer, "EXPECTED_FRESH_PARAMETER_COUNT", 16)
    collection = _tiny_collection()
    candidate = tmp_path / "candidate"
    metadata = publish_fixed_final_candidate_v94(
        candidate, collection, bindings={"config_sha256": "c" * 64}
    )
    assert metadata["artifact"] == CANDIDATE_ARTIFACT
    assert metadata["tensor_inventory"] == ["lora_a", "lora_b"]
    assert set(load_file(str(candidate / "bridge.safetensors"))) == {
        "lora_a",
        "lora_b",
    }
    assert (
        json.loads((candidate / "runtime_metadata.json").read_text())[
            "questions_or_answers_serialized"
        ]
        is False
    )
    with pytest.raises(FileExistsError):
        publish_fixed_final_candidate_v94(candidate, collection, bindings={})

    reloaded = _tiny_collection()
    loaded = load_fixed_final_bridge_v94(reloaded, candidate)
    assert loaded["state_sha256"] == metadata["state_sha256"]


def test_v94_production_fresh_parameter_contract_is_exact() -> None:
    assert TARGET_MODULE == "model.language_model.layers.34.mlp.gate_proj"
    assert EXPECTED_FRESH_PARAMETER_COUNT == 110_592
    assert 8 * (1_536 + 12_288) == EXPECTED_FRESH_PARAMETER_COUNT


def test_v94_trainer_constants_bind_fixed_full40_schedule() -> None:
    from semantic_3d_chat.training import train_v94_strict_multiscene_full40 as trainer

    assert trainer.EXPECTED_ROWS_PER_EPOCH == 960
    assert trainer.EXPECTED_EPOCHS == 3
    assert trainer.EXPECTED_MICRO_ROWS == 2_880
    assert trainer.EXPECTED_OPTIMIZER_UPDATES == 360
    assert trainer.EXPECTED_PAIRED_MARGIN_ROWS == 396
    assert trainer.EXPECTED_CAUSAL_MARGIN_ROWS == 54
    assert trainer.EXPECTED_TOTAL_PARAMETER_COUNT == 675_840
    assert "v86" in trainer._FORBIDDEN_COMPONENTS
    assert "v93" in trainer._FORBIDDEN_COMPONENTS
