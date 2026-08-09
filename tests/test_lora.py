from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file
from torch import nn

from semantic_3d_chat.config import load_config
from semantic_3d_chat.language.lora import (
    LoRALinear,
    install_lora_adapters,
    lora_checkpoint_contract,
    lora_checkpoint_contract_mismatch,
    lora_optimizer_settings,
    lora_settings,
    tensor_state_sha256,
    validate_lora_checkpoint_state,
)
from semantic_3d_chat.training.checkpointing import (
    load_adapter_checkpoint,
    load_optimizer_checkpoint,
    save_adapter_checkpoint,
    save_optimizer_checkpoint,
)
from semantic_3d_chat.training.train_adapter import build_adapter_optimizer


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
