from __future__ import annotations

import hashlib
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file
from torch import nn

from semantic_3d_chat.config import load_config
from semantic_3d_chat.language.lora import (
    LoRABankSettings,
    LoRABanksSettings,
    LoRALinear,
    LoRASettings,
    initialize_lora_adapter_state,
    install_lora_adapters,
    install_lora_banks,
    lora_banks_checkpoint_contract,
    lora_banks_optimizer_settings,
    lora_banks_settings,
    lora_checkpoint_contract,
    lora_checkpoint_contract_mismatch,
    lora_optimizer_settings,
    lora_settings,
    tensor_state_sha256,
    validate_lora_banks_checkpoint_state,
    validate_lora_checkpoint_state,
)
from semantic_3d_chat.training.checkpointing import (
    load_adapter_checkpoint,
    load_optimizer_checkpoint,
    save_adapter_checkpoint,
    save_optimizer_checkpoint,
)
from semantic_3d_chat.training.train_adapter import (
    assert_zero_output_lora_banks,
    build_adapter_optimizer,
    legacy_lora_bank_source_mismatch,
    optional_sha256_setting,
    staged_legacy_lora_checkpoint_modules,
    training_map_forward,
    verify_initialization_artifact_hashes,
)


class _TinyAttention(nn.Module):
    def __init__(self, hidden_size: int = 6, projected_size: int = 10) -> None:
        super().__init__()
        self.q_proj = nn.Linear(hidden_size, projected_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, projected_size, bias=False)
        self.o_proj = nn.Linear(projected_size, hidden_size, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.o_proj(torch.tanh(self.q_proj(inputs)))


class _TinyLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = _TinyAttention()


class _TinyLanguageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_TinyLayer(), _TinyLayer()])


class _TinyGemma(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = _TinyLanguageModel()


def test_tensor_state_sha256_supports_scalars_without_changing_vector_bytes() -> None:
    scalar = torch.tensor(1.25, dtype=torch.float32)
    vector = scalar.reshape(1)

    scalar_hash = tensor_state_sha256({"value": scalar})
    vector_hash = tensor_state_sha256({"value": vector})

    assert len(scalar_hash) == 64
    assert scalar_hash != vector_hash  # The tensor shape remains part of the identity.
    assert scalar_hash == tensor_state_sha256({"value": scalar.clone()})


def _enabled_config() -> dict:
    return {
        "language": {
            "backend": "gemma4",
            "lora": {
                "enabled": True,
                "rank": 2,
                "alpha": 4.0,
                "dropout": 0.0,
                "target_modules": [
                    "model.language_model.layers.1.self_attn.q_proj",
                    "model.language_model.layers.1.self_attn.o_proj",
                ],
            },
        },
        "training": {
            "lora_learning_rate": 1e-4,
            "lora_weight_decay": 0.0,
        },
    }


def _two_bank_config(*, inherited_hash: str | None = None) -> dict:
    return {
        "language": {
            "backend": "gemma4",
            "lora_banks": {
                "inherited": {
                    "trainable": False,
                    "rank": 2,
                    "alpha": 4.0,
                    "dropout": 0.0,
                    "initialization_algorithm": "checkpoint_overwrite",
                    "initialization_seed": None,
                    "expected_initial_state_sha256": inherited_hash,
                    "target_modules": [
                        "model.language_model.layers.0.self_attn.q_proj",
                        "model.language_model.layers.0.self_attn.o_proj",
                    ],
                },
                "extension": {
                    "trainable": True,
                    "rank": 3,
                    "alpha": 6.0,
                    "dropout": 0.0,
                    "initialization_algorithm": "cpu_kaiming_uniform_a_exact_zero_b",
                    "initialization_seed": 13008,
                    "expected_initial_state_sha256": None,
                    "target_modules": [
                        "model.language_model.layers.1.self_attn.q_proj",
                        "model.language_model.layers.1.self_attn.o_proj",
                    ],
                },
            },
        },
        "training": {
            "learning_rate": 3e-4,
            "weight_decay": 0.01,
            "lora_learning_rate": 5e-5,
            "lora_weight_decay": 0.0,
        },
    }


def test_zero_initialized_lora_is_exact_frozen_baseline() -> None:
    torch.manual_seed(8101)
    base = nn.Linear(5, 7, bias=True)
    inputs = torch.randn(4, 5)
    expected = base(inputs).detach().clone()

    layer = LoRALinear(base, rank=3, alpha=6.0, dropout=0.4).train()
    actual = layer(inputs)

    assert torch.equal(actual, expected)
    assert torch.count_nonzero(layer.lora_b) == 0
    assert layer.lora_a.dtype == layer.lora_b.dtype == torch.float32
    assert all(not parameter.requires_grad for parameter in layer.base.parameters())
    assert all(parameter.requires_grad for parameter in layer.adapter_parameters())


def test_gradients_reach_b_then_a_after_first_optimizer_step() -> None:
    torch.manual_seed(8102)
    layer = LoRALinear(nn.Linear(4, 3), rank=2, alpha=4.0).train()
    base_before = {
        name: parameter.detach().clone() for name, parameter in layer.base.named_parameters()
    }
    inputs = torch.randn(8, 4)
    targets = torch.randn(8, 3)
    optimizer = torch.optim.SGD(layer.adapter_parameters(), lr=0.2)

    first_loss = torch.nn.functional.mse_loss(layer(inputs), targets)
    first_loss.backward()
    assert layer.lora_b.grad is not None and layer.lora_b.grad.abs().sum() > 0
    # With an exactly zero B matrix, the first A gradient is mathematically zero.
    assert layer.lora_a.grad is not None and torch.count_nonzero(layer.lora_a.grad) == 0
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    second_loss = torch.nn.functional.mse_loss(layer(inputs), targets)
    second_loss.backward()
    assert layer.lora_a.grad is not None and layer.lora_a.grad.abs().sum() > 0
    assert layer.lora_b.grad is not None and layer.lora_b.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in layer.base.parameters())
    for name, parameter in layer.base.named_parameters():
        assert torch.equal(parameter, base_before[name])


def test_state_dict_roundtrip_preserves_unmerged_output(tmp_path: Path) -> None:
    torch.manual_seed(8103)
    layer = LoRALinear(nn.Linear(6, 4), rank=2, alpha=3.0, dropout=0.1)
    optimizer = torch.optim.AdamW(layer.adapter_parameters(), lr=0.05)
    layer.train()
    for _ in range(2):
        inputs = torch.randn(5, 6)
        loss = layer(inputs).square().mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    layer.eval()
    probe = torch.randn(3, 6)
    expected = layer(probe).detach()
    state_path = tmp_path / "lora_state.pt"
    torch.save(layer.state_dict(), state_path)

    restored = LoRALinear(nn.Linear(6, 4), rank=2, alpha=3.0, dropout=0.1).eval()
    restored.load_state_dict(torch.load(state_path, map_location="cpu", weights_only=True))

    assert torch.equal(restored(probe), expected)
    assert all(not parameter.requires_grad for parameter in restored.base.parameters())
    assert torch.count_nonzero(restored.lora_b) > 0


def test_trainable_count_and_dtype_cast_contract() -> None:
    layer = LoRALinear(nn.Linear(5, 7), rank=3, alpha=6.0, dropout=0.0)
    expected_count = 3 * (5 + 7)
    actual_count = sum(
        parameter.numel() for parameter in layer.parameters() if parameter.requires_grad
    )

    assert layer.adapter_parameter_count == expected_count == 36
    assert actual_count == expected_count
    assert sum(parameter.numel() for parameter in layer.adapter_parameters()) == expected_count
    assert "merge=False" in repr(layer)

    layer.to(dtype=torch.float16)
    assert layer.base.weight.dtype == torch.float16
    assert layer.lora_a.dtype == layer.lora_b.dtype == torch.float32


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"rank": 0, "alpha": 1.0}, "rank"),
        ({"rank": 2, "alpha": 0.0}, "alpha"),
        ({"rank": 2, "alpha": float("inf")}, "alpha"),
        ({"rank": 2, "alpha": 1.0, "dropout": 1.0}, "dropout"),
    ],
)
def test_invalid_hyperparameters_are_rejected(kwargs: dict, message: str) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        LoRALinear(nn.Linear(3, 2), **kwargs)


def test_non_linear_base_is_rejected() -> None:
    with pytest.raises(TypeError, match="torch.nn.Linear"):
        LoRALinear(nn.Identity(), rank=2, alpha=2.0)


def test_exact_target_installation_freezes_everything_except_a_and_b() -> None:
    torch.manual_seed(8104)
    model = _TinyGemma().eval().requires_grad_(False)
    inputs = torch.randn(3, 6)
    layer = model.model.language_model.layers[1]
    expected = layer.self_attn(inputs).clone()
    untouched_q = model.model.language_model.layers[0].self_attn.q_proj
    settings = lora_settings(_enabled_config())

    installation = install_lora_adapters(model, settings)

    assert installation is not None
    assert torch.equal(layer.self_attn(inputs), expected)
    assert isinstance(layer.self_attn.q_proj, LoRALinear)
    assert isinstance(layer.self_attn.o_proj, LoRALinear)
    assert model.model.language_model.layers[0].self_attn.q_proj is untouched_q
    assert installation.parameter_counts == {
        "model.language_model.layers.1.self_attn.q_proj": 2 * (6 + 10),
        "model.language_model.layers.1.self_attn.o_proj": 2 * (10 + 6),
    }
    assert installation.parameter_count == 64
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    assert trainable == [
        "model.language_model.layers.1.self_attn.q_proj.lora_a",
        "model.language_model.layers.1.self_attn.q_proj.lora_b",
        "model.language_model.layers.1.self_attn.o_proj.lora_a",
        "model.language_model.layers.1.self_attn.o_proj.lora_b",
    ]
    assert all("base" not in key for key in installation.state_module.state_dict())

    assert installation.training is False
    installation.train()
    assert installation.training is True
    assert all(adapter.dropout.training for adapter in installation.adapters)
    installation.eval()
    assert installation.training is False
    assert all(not adapter.dropout.training for adapter in installation.adapters)


def test_exact_target_installation_is_atomic_on_missing_target() -> None:
    model = _TinyGemma().requires_grad_(False)
    original = model.model.language_model.layers[1].self_attn.q_proj
    config = _enabled_config()
    config["language"]["lora"]["target_modules"].append(
        "model.language_model.layers.9.self_attn.q_proj"
    )

    with pytest.raises(ValueError, match="does not exist"):
        install_lora_adapters(model, lora_settings(config))

    assert model.model.language_model.layers[1].self_attn.q_proj is original


def test_compact_checkpoint_roundtrip_and_content_tamper_detection(tmp_path: Path) -> None:
    config = _enabled_config()
    settings = lora_settings(config)
    optimizer_settings = lora_optimizer_settings(config, settings)
    model = _TinyGemma().requires_grad_(False)
    installation = install_lora_adapters(model, settings)
    assert installation is not None and optimizer_settings is not None
    with torch.no_grad():
        installation.adapters[0].lora_b.fill_(0.125)
        installation.adapters[1].lora_b.fill_(-0.25)
    contract = lora_checkpoint_contract(settings, optimizer_settings, installation.parameter_count)
    metadata = {
        "lora": contract,
        "lora_wrapped_modules": list(installation.target_names),
        "lora_trainable_parameter_counts": installation.parameter_counts,
        "lora_trainable_parameter_count": installation.parameter_count,
        "lora_state_sha256": installation.state_sha256(),
    }
    checkpoint = save_adapter_checkpoint(
        tmp_path / "checkpoint", {"lora": installation.state_module}, metadata
    )
    tensors = load_file(checkpoint / "adapter.safetensors")
    assert sorted(tensors) == [
        "lora.adapters.0.lora_a",
        "lora.adapters.0.lora_b",
        "lora.adapters.1.lora_a",
        "lora.adapters.1.lora_b",
    ]

    restored_model = _TinyGemma().requires_grad_(False)
    restored = install_lora_adapters(restored_model, settings)
    assert restored is not None
    loaded_metadata = load_adapter_checkpoint(
        checkpoint, {"lora": restored.state_module}, device="cpu"
    )
    validate_lora_checkpoint_state(loaded_metadata, restored)
    for expected, actual in zip(installation.parameters(), restored.parameters(), strict=True):
        assert torch.equal(expected, actual)

    with torch.no_grad():
        restored.adapters[0].lora_b[0, 0].add_(1.0)
    with pytest.raises(ValueError, match="tamper"):
        validate_lora_checkpoint_state(loaded_metadata, restored)


def test_checkpoint_contract_rejects_lr_target_and_count_drift() -> None:
    config = _enabled_config()
    settings = lora_settings(config)
    optimizer_settings = lora_optimizer_settings(config, settings)
    assert optimizer_settings is not None
    expected = lora_checkpoint_contract(settings, optimizer_settings, 64)

    assert lora_checkpoint_contract_mismatch({"lora": expected}, expected) is None
    for field, value in (
        ("learning_rate", 2e-4),
        ("adapter_parameter_count", 65),
        ("target_modules", list(reversed(expected["target_modules"]))),
    ):
        altered = {**expected, field: value}
        assert lora_checkpoint_contract_mismatch({"lora": altered}, expected) is not None


def test_separate_optimizer_groups_restore_for_exact_resume(tmp_path: Path) -> None:
    config = _enabled_config()
    config["training"].update({"learning_rate": 3e-4, "weight_decay": 0.01})
    settings = lora_settings(config)
    optimizer_settings = lora_optimizer_settings(config, settings)
    model = _TinyGemma().requires_grad_(False)
    installation = install_lora_adapters(model, settings)
    assert installation is not None and optimizer_settings is not None
    scene_parameter = nn.Parameter(torch.tensor([1.0, -1.0]))
    optimizer, clipped_parameters = build_adapter_optimizer(
        config, [scene_parameter], installation, optimizer_settings
    )
    assert [group["name"] for group in optimizer.param_groups] == [
        "scene_adapter",
        "language_lora",
    ]
    assert [group["lr"] for group in optimizer.param_groups] == [3e-4, 1e-4]
    assert [group["weight_decay"] for group in optimizer.param_groups] == [0.01, 0.0]
    assert [id(parameter) for parameter in clipped_parameters] == [
        id(parameter) for parameter in [scene_parameter, *installation.parameters()]
    ]

    inputs = torch.randn(2, 6)
    loss = scene_parameter.square().sum()
    for adapter in installation.adapters:
        adapter_inputs = inputs if adapter.in_features == 6 else torch.randn(2, 10)
        loss = loss + adapter(adapter_inputs).square().mean()
    loss.backward()
    optimizer.step()
    save_optimizer_checkpoint(tmp_path, optimizer)

    restored_model = _TinyGemma().requires_grad_(False)
    restored_installation = install_lora_adapters(restored_model, settings)
    assert restored_installation is not None
    restored_scene_parameter = nn.Parameter(torch.zeros(2))
    restored_optimizer, _ = build_adapter_optimizer(
        config, [restored_scene_parameter], restored_installation, optimizer_settings
    )
    load_optimizer_checkpoint(tmp_path, restored_optimizer, "cpu")

    assert [group["name"] for group in restored_optimizer.param_groups] == [
        "scene_adapter",
        "language_lora",
    ]
    assert len(restored_optimizer.state) == len(optimizer.state)


def test_v8_config_is_strict_opt_in_and_keeps_v7_gate_schedule() -> None:
    v7 = load_config("configs/experiments/gemma4_color_wiring_v7.yaml")
    v8 = load_config("configs/experiments/gemma4_color_wiring_v8.yaml")
    settings = lora_settings(v8)
    optimizer_settings = lora_optimizer_settings(v8, settings)

    assert "lora" not in v7["language"]
    assert settings.target_modules == (
        "model.language_model.layers.34.self_attn.q_proj",
        "model.language_model.layers.34.self_attn.o_proj",
    )
    assert (settings.rank, settings.alpha, settings.dropout) == (4, 8.0, 0.0)
    assert optimizer_settings is not None
    assert optimizer_settings.contract() == {"learning_rate": 1e-4, "weight_decay": 0.0}
    assert v8["training"]["epochs"] == v7["training"]["epochs"]
    assert v8["training"]["pair_steps_per_epoch"] == v7["training"]["pair_steps_per_epoch"]
    assert v8["training"]["gradient_accumulation"] == (v7["training"]["gradient_accumulation"])
    expected_updates = (
        v8["training"]["epochs"]
        * v8["training"]["pair_steps_per_epoch"]
        // v8["training"]["gradient_accumulation"]
    )
    assert expected_updates == 12


def test_legacy_config_through_bank_api_preserves_schema1_contract_and_keys() -> None:
    config = _enabled_config()
    legacy_settings = lora_settings(config)
    legacy_optimizer = lora_optimizer_settings(config, legacy_settings)
    bank_settings = lora_banks_settings(config)
    bank_optimizer = lora_banks_optimizer_settings(config, bank_settings)
    model = _TinyGemma().requires_grad_(False)
    collection = install_lora_banks(model, bank_settings)

    assert collection is not None
    assert collection.bank_names == ("legacy",)
    assert collection.state_modules().keys() == {"lora"}
    expected = lora_checkpoint_contract(
        legacy_settings, legacy_optimizer, collection.parameter_count
    )
    actual = lora_banks_checkpoint_contract(
        bank_settings, bank_optimizer, collection.parameter_counts
    )
    assert actual == expected
    assert collection.checkpoint_metadata() == {
        "lora_wrapped_modules": list(collection.banks[0].installation.target_names),
        "lora_trainable_parameter_counts": collection.banks[0].installation.parameter_counts,
        "lora_trainable_parameter_count": collection.parameter_count,
        "lora_state_sha256": collection.banks[0].installation.state_sha256(),
    }


def test_two_bank_zero_output_optimizer_surface_and_frozen_invariant() -> None:
    torch.manual_seed(8301)
    model = _TinyGemma().eval().requires_grad_(False)
    inputs = torch.randn(4, 6)
    expected = [
        layer.self_attn(inputs).detach().clone() for layer in model.model.language_model.layers
    ]
    config = _two_bank_config()
    settings = lora_banks_settings(config)
    optimizer_settings = lora_banks_optimizer_settings(config, settings)
    collection = install_lora_banks(model, settings)
    assert collection is not None and optimizer_settings is not None

    actual = [layer.self_attn(inputs) for layer in model.model.language_model.layers]
    assert all(torch.equal(left, right) for left, right in zip(actual, expected, strict=True))
    inherited = collection.bank("inherited").installation
    extension = collection.bank("extension").installation
    frozen_hash = inherited.state_sha256()
    trainable_names = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    assert trainable_names == [
        "model.language_model.layers.1.self_attn.q_proj.lora_a",
        "model.language_model.layers.1.self_attn.q_proj.lora_b",
        "model.language_model.layers.1.self_attn.o_proj.lora_a",
        "model.language_model.layers.1.self_attn.o_proj.lora_b",
    ]

    frozen_scene = nn.Parameter(torch.ones(2), requires_grad=False)
    optimizer, clipped = build_adapter_optimizer(
        config, [frozen_scene], collection, optimizer_settings
    )
    assert [group["name"] for group in optimizer.param_groups] == ["language_lora"]
    assert {id(parameter) for parameter in clipped} == {
        id(parameter) for parameter in extension.parameters()
    }
    loss = sum(
        layer.self_attn(inputs).square().mean() for layer in model.model.language_model.layers
    )
    loss.backward()
    assert all(parameter.grad is None for parameter in inherited.parameters())
    assert any(
        adapter.lora_b.grad is not None and torch.count_nonzero(adapter.lora_b.grad)
        for adapter in extension.adapters
    )
    optimizer.step()
    assert inherited.state_sha256() == frozen_hash
    assert any(torch.count_nonzero(adapter.lora_b) for adapter in extension.adapters)


def test_all_frozen_banks_allow_a_trainable_scene_only_optimizer() -> None:
    config = _two_bank_config()
    parsed = lora_banks_settings(config)
    settings = LoRABanksSettings(tuple(replace(bank, trainable=False) for bank in parsed.banks))
    collection = install_lora_banks(_TinyGemma().requires_grad_(False), settings)
    assert collection is not None and collection.parameters() == []
    scene_parameter = nn.Parameter(torch.ones(2))
    optimizer, clipped = build_adapter_optimizer(
        config, [scene_parameter], collection, configured_lora_optimizer=None
    )
    assert [group["name"] for group in optimizer.param_groups] == ["scene_adapter"]
    assert clipped == [scene_parameter]


def test_named_bank_schema2_checkpoint_roundtrip_and_tamper_rejection(tmp_path: Path) -> None:
    config = _two_bank_config()
    settings = lora_banks_settings(config)
    optimizer_settings = lora_banks_optimizer_settings(config, settings)
    source = install_lora_banks(_TinyGemma().requires_grad_(False), settings)
    assert source is not None and optimizer_settings is not None
    with torch.no_grad():
        source.bank("inherited").installation.adapters[0].lora_b.fill_(0.125)
        source.bank("extension").installation.adapters[1].lora_b.fill_(-0.25)
    contract = lora_banks_checkpoint_contract(settings, optimizer_settings, source.parameter_counts)
    assert contract["schema_version"] == 2
    assert contract["adapter_parameter_count"] == source.parameter_count
    assert contract["trainable_adapter_parameter_count"] == source.trainable_parameter_count
    for field, value in (
        ("trainable", True),
        ("rank", 9),
        ("target_modules", ["different.target"]),
        ("initialization_seed", 7),
        ("expected_initial_state_sha256", "0" * 64),
    ):
        altered_banks = [dict(bank) for bank in contract["banks"]]
        altered_banks[0][field] = value
        altered = {**contract, "banks": altered_banks}
        assert lora_checkpoint_contract_mismatch({"lora": altered}, contract) is not None
    metadata = {"lora": contract, **source.checkpoint_metadata()}
    checkpoint = save_adapter_checkpoint(tmp_path / "banks", source.state_modules(), metadata)
    keys = sorted(load_file(checkpoint / "adapter.safetensors"))
    assert keys[0].startswith("lora_banks.extension.")
    assert keys[-1].startswith("lora_banks.inherited.")

    restored = install_lora_banks(_TinyGemma().requires_grad_(False), settings)
    assert restored is not None
    loaded = load_adapter_checkpoint(checkpoint, restored.state_modules(), device="cpu")
    validate_lora_banks_checkpoint_state(loaded, restored)
    assert restored.state_sha256() == source.state_sha256()

    tensors = load_file(checkpoint / "adapter.safetensors")
    tensors["lora_banks.extension.adapters.0.lora_b"][0, 0].add_(1.0)
    save_file(tensors, checkpoint / "adapter.safetensors")
    tampered = install_lora_banks(_TinyGemma().requires_grad_(False), settings)
    assert tampered is not None
    loaded = load_adapter_checkpoint(checkpoint, tampered.state_modules(), device="cpu")
    with pytest.raises(ValueError, match="tamper"):
        validate_lora_banks_checkpoint_state(loaded, tampered)


def test_deterministic_bank_initialization_and_expected_hash_contract() -> None:
    adapter_settings = LoRASettings(
        enabled=True,
        rank=3,
        alpha=6.0,
        dropout=0.0,
        target_modules=("model.language_model.layers.1.self_attn.q_proj",),
    )
    first_model = _TinyGemma().requires_grad_(False)
    first = install_lora_adapters(first_model, adapter_settings)
    assert first is not None
    initialize_lora_adapter_state(first, seed=13008)
    expected_hash = first.state_sha256()
    bank = LoRABankSettings(
        "extension",
        True,
        adapter_settings,
        "cpu_kaiming_uniform_a_exact_zero_b",
        13008,
        expected_hash,
    )
    collection = install_lora_banks(_TinyGemma().requires_grad_(False), LoRABanksSettings((bank,)))
    assert collection is not None
    assert collection.bank("extension").installation.state_sha256() == expected_hash
    bad_bank = replace(bank, expected_initial_state_sha256="0" * 64)
    with pytest.raises(ValueError, match="initial-state hash mismatch"):
        install_lora_banks(_TinyGemma().requires_grad_(False), LoRABanksSettings((bad_bank,)))


def test_legacy_checkpoint_maps_only_into_frozen_named_bank(tmp_path: Path) -> None:
    source_config = _enabled_config()
    source_config["language"]["lora"]["target_modules"] = [
        "model.language_model.layers.0.self_attn.q_proj",
        "model.language_model.layers.0.self_attn.o_proj",
    ]
    source_settings = lora_settings(source_config)
    source_optimizer = lora_optimizer_settings(source_config, source_settings)
    source = install_lora_adapters(_TinyGemma().requires_grad_(False), source_settings)
    assert source is not None and source_optimizer is not None
    with torch.no_grad():
        source.adapters[0].lora_b.fill_(0.375)
    source_metadata = {
        "lora": lora_checkpoint_contract(source_settings, source_optimizer, source.parameter_count),
        "lora_wrapped_modules": list(source.target_names),
        "lora_trainable_parameter_counts": source.parameter_counts,
        "lora_trainable_parameter_count": source.parameter_count,
        "lora_state_sha256": source.state_sha256(),
    }
    checkpoint = save_adapter_checkpoint(
        tmp_path / "legacy", {"lora": source.state_module}, source_metadata
    )

    config = _two_bank_config(inherited_hash=source.state_sha256())
    settings = lora_banks_settings(config)
    collection = install_lora_banks(_TinyGemma().requires_grad_(False), settings)
    assert collection is not None
    inherited = collection.bank("inherited")
    extension_before = collection.bank("extension").installation.state_sha256()
    assert legacy_lora_bank_source_mismatch(source_metadata, inherited) is None
    scene_modules = {
        "scene_model": nn.Identity(),
        "composer": nn.Identity(),
        "grounding": nn.Identity(),
    }
    staged_modules = staged_legacy_lora_checkpoint_modules(scene_modules, inherited)
    assert set(staged_modules) == {"scene_model", "composer", "grounding", "lora"}
    assert all(not key.startswith("lora_banks.") for key in staged_modules)
    loaded = load_adapter_checkpoint(checkpoint, staged_modules, device="cpu")
    validate_lora_checkpoint_state(loaded, inherited.installation)
    assert inherited.installation.state_sha256() == source.state_sha256()
    assert collection.bank("extension").installation.state_sha256() == extension_before
    assert_zero_output_lora_banks(collection, exclude=("inherited",))
    altered = {**source_metadata, "lora_state_sha256": "0" * 64}
    assert "lora_state_sha256" in legacy_lora_bank_source_mismatch(altered, inherited)

    optimizer_settings = lora_banks_optimizer_settings(config, settings)
    assert optimizer_settings is not None
    named_metadata = {
        "lora": lora_banks_checkpoint_contract(
            settings, optimizer_settings, collection.parameter_counts
        ),
        **collection.checkpoint_metadata(),
    }
    named_checkpoint = save_adapter_checkpoint(
        tmp_path / "named", collection.state_modules(), named_metadata
    )
    restored = install_lora_banks(_TinyGemma().requires_grad_(False), settings)
    assert restored is not None
    loaded_named = load_adapter_checkpoint(named_checkpoint, restored.state_modules(), device="cpu")
    validate_lora_banks_checkpoint_state(loaded_named, restored)
    assert restored.state_sha256() == collection.state_sha256()


def test_v13_config_pins_probe_banks_source_and_frozen_objectives() -> None:
    config = load_config("configs/experiments/gemma4_color_mirror_decoder_banks_v13.yaml")
    settings = lora_banks_settings(config)
    optimizer = lora_banks_optimizer_settings(config, settings)
    assert config["language"]["lora"] is None
    assert settings.bank("inherited_v12").adapter.target_modules == (
        "model.language_model.layers.34.self_attn.q_proj",
        "model.language_model.layers.34.self_attn.o_proj",
    )
    assert settings.bank("extension_v13").adapter.target_modules == tuple(
        f"model.language_model.layers.{layer}.self_attn.{projection}"
        for layer in range(30, 34)
        for projection in ("q_proj", "o_proj")
    )
    assert settings.bank("extension_v13").initialization_seed == 13008
    assert settings.bank("extension_v13").expected_initial_state_sha256 == (
        "b4ec0518e4759dda33fc93c9c1d4c76f52f1024fd5b8b1667ad1b4ef5da198af"
    )
    inherited_parameter_count = 2 * 4 * (1536 + 4096)
    extension_parameter_count = 4 * 2 * 8 * (1536 + 2048)
    assert inherited_parameter_count == 45056
    assert extension_parameter_count == 229376
    assert inherited_parameter_count + extension_parameter_count == 274432
    assert optimizer is not None and optimizer.learning_rate == 5e-5
    training = config["training"]
    assert training["initialize_from"].endswith("/epoch_008")
    assert training["initialize_expected_adapter_sha256"] == (
        "a4c85c14a214e4e594992e489a784cb4bacb64d3dfda519ad3da18b1595d9f22"
    )
    assert training["initialize_expected_metadata_sha256"] == (
        "f097c6477546460440e77a3d225afb55818cb13abf9cbb4a90500f75a879b0f5"
    )
    assert training["initialize_legacy_lora_into_bank"] == "inherited_v12"
    assert training["freeze_scene_adapter"] is True
    assert training["epochs"] == training["pair_steps_per_epoch"] == 12
    assert training["gradient_accumulation"] == 12
    for key in (
        "grounding_weight",
        "grounding_anchor_weight",
        "latent_diversity_weight",
        "paired_scene_separation_weight",
        "spatial_answer_contrastive_weight",
        "spatial_answer_warmup_steps",
        "spatial_relation_contrastive_weight",
        "spatial_relation_warmup_steps",
    ):
        assert training[key] == 0


def test_v14_lr_sweep_changes_only_screen_schedule_namespace_and_learning_rate() -> None:
    v13 = load_config("configs/experiments/gemma4_color_mirror_decoder_banks_v13.yaml")
    arms = {
        "lr1e4": 1e-4,
        "lr3e4": 3e-4,
        "lr1e3": 1e-3,
        "lr2e3": 2e-3,
    }
    for suffix, learning_rate in arms.items():
        config = load_config(
            f"configs/experiments/gemma4_color_mirror_decoder_banks_v14_{suffix}.yaml"
        )
        normalized_v13 = deepcopy(v13)
        normalized_arm = deepcopy(config)
        normalized_v13.pop("_config_path", None)
        normalized_arm.pop("_config_path", None)
        sweep = normalized_arm.pop("sweep")
        for key in (
            "output_namespace",
            "lora_learning_rate",
            "epochs",
            "pair_gate_every_epochs",
            "early_stopping_patience",
        ):
            normalized_arm["training"][key] = normalized_v13["training"][key]
        assert normalized_arm == normalized_v13
        assert config["language"] == v13["language"]
        training = config["training"]
        assert training["output_namespace"] == (f"gemma4_color_mirror_decoder_banks_v14_{suffix}")
        assert training["lora_learning_rate"] == learning_rate
        assert training["epochs"] == 4
        assert training["pair_steps_per_epoch"] == 12
        assert training["gradient_accumulation"] == 12
        assert training["pair_gate_every_epochs"] == 1
        assert training["pair_gate_stop_when_passed"] is False
        assert training["early_stopping_patience"] == 0
        for key in (
            "initialize_from",
            "initialize_expected_adapter_sha256",
            "initialize_expected_metadata_sha256",
            "initialize_legacy_lora_into_bank",
            "freeze_scene_adapter",
        ):
            assert training[key] == v13["training"][key]
        assert sweep["arm_learning_rate"] == learning_rate
        assert sweep["updates_per_arm"] == 4
        assert sweep["expected_selection_sha256"] == (
            "7f0714e3151c9ddb57c1da95a457820a833e490c070881a88a9fee4a9168f933"
        )
        assert sweep["expected_frozen_scene_state_sha256"] == (
            "690bd890bfda024dbb5c7d3c68087b8113bc3b8ee81dd6143c7eb2a884e7245b"
        )


def test_initialization_artifact_hash_pins_fail_before_tensor_load(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "adapter.safetensors").write_bytes(b"adapter")
    (checkpoint / "metadata.json").write_bytes(b"metadata")
    adapter_hash = hashlib.sha256(b"adapter").hexdigest()
    metadata_hash = hashlib.sha256(b"metadata").hexdigest()
    assert optional_sha256_setting({"pin": adapter_hash}, "pin") == adapter_hash
    observed = verify_initialization_artifact_hashes(
        checkpoint,
        expected_adapter_sha256=adapter_hash,
        expected_metadata_sha256=metadata_hash,
    )
    assert observed == {
        "adapter_sha256": adapter_hash,
        "metadata_sha256": metadata_hash,
    }
    with pytest.raises(ValueError, match="content hash mismatch"):
        verify_initialization_artifact_hashes(
            checkpoint,
            expected_adapter_sha256="0" * 64,
            expected_metadata_sha256=metadata_hash,
        )
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        optional_sha256_setting({"pin": "invalid"}, "pin")


def test_frozen_scene_no_grad_tokens_support_lora_weight_backward() -> None:
    class FrozenScene(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.projection = nn.Linear(4, 6, bias=False)

        def forward(self, semantic, *_args):
            return SimpleNamespace(scene_tokens=self.projection(semantic))

    scene = FrozenScene().requires_grad_(False).eval()
    data = SimpleNamespace(
        semantic=torch.randn(3, 4),
        xyz=torch.zeros(3, 3),
        rgb=torch.zeros(3, 3),
        normal=torch.zeros(3, 3),
        confidence=torch.ones(3),
        observation_count=torch.ones(3),
        room_min=torch.zeros(3),
        room_max=torch.ones(3),
    )
    output = training_map_forward(scene, data, freeze_scene_adapter=True)
    assert not output.scene_tokens.requires_grad
    assert not output.scene_tokens.is_inference()

    adapter = LoRALinear(nn.Linear(6, 5, bias=False), rank=2, alpha=4.0)
    adapter(output.scene_tokens).square().mean().backward()
    assert all(parameter.grad is None for parameter in scene.parameters())
    assert adapter.lora_b.grad is not None
    assert torch.count_nonzero(adapter.lora_b.grad) > 0
