from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file
from torch import nn

from semantic_3d_chat.chat.runtime_config import load_runtime_config
from semantic_3d_chat.chat.v94_strict_multiscene_runtime import EXPECTED_BANKS
from semantic_3d_chat.evaluation.v85_strict_multiscene_preflight import sha256_file_v85
from semantic_3d_chat.evaluation.v96_atomic_pair_repair_preflight import (
    FRESH_BANK_NAME,
    TARGET_MODULES,
    load_config_v96,
)
from semantic_3d_chat.language.lora import (
    LoRABankCollection,
    LoRABankSettings,
    LoRABanksSettings,
    LoRASettings,
    install_lora_banks,
)
from semantic_3d_chat.training.train_v96_atomic_pair_repair import (
    CANDIDATE_ARTIFACT,
    EXPECTED_CHANGED_PAIR_STEPS,
    EXPECTED_FROZEN_BANK_COUNT,
    EXPECTED_INVARIANT_PAIR_STEPS,
    EXPECTED_MICRO_STEPS,
    EXPECTED_OPTIMIZER_UPDATES,
    EXPECTED_RETENTION_STEPS,
    EXPECTED_TOTAL_NLL_FORWARDS,
    combined_lora_settings_v96,
    discover_resume_checkpoint_v96,
    invariant_pair_objective_v96,
    load_fixed_final_bridge_v96,
    publish_fixed_final_candidate_v96,
    restore_resume_checkpoint_v96,
    save_resume_checkpoint_v96,
    smoothmax_v96,
    symmetric_pair_objective_v96,
)


def test_v96_combined_settings_are_exact_frozen_v95_plus_fresh_q() -> None:
    config = load_config_v96()
    runtime = load_runtime_config(config["sources"]["runtime_config"])
    settings = combined_lora_settings_v96(runtime, config)

    assert len(settings.banks) == 10
    assert tuple(bank.name for bank in settings.banks[:-2]) == EXPECTED_BANKS
    assert settings.banks[-2].name == "v95_strict_causal_successor_bridge"
    assert settings.banks[-2].trainable is False
    assert settings.banks[-1].name == FRESH_BANK_NAME
    assert settings.banks[-1].trainable is True
    assert sum(not bank.trainable for bank in settings.banks) == EXPECTED_FROZEN_BANK_COUNT == 9
    assert sum(bank.trainable for bank in settings.banks) == 1
    assert settings.banks[-1].adapter.target_modules == TARGET_MODULES
    assert settings.banks[-1].adapter.rank == 8
    assert settings.banks[-1].adapter.alpha == 16.0


def test_v96_symmetric_four_nll_objective_penalizes_worst_pair_side() -> None:
    left = torch.tensor(1.0, requires_grad=True)
    right = torch.tensor(1.2, requires_grad=True)
    left_alt = torch.tensor(0.7, requires_grad=True)
    right_alt = torch.tensor(0.8, requires_grad=True)

    objective, records = symmetric_pair_objective_v96(
        left,
        right,
        left_alt,
        right_alt,
        left_class_weight=1.0,
        right_class_weight=1.0,
        family_weight=1.0,
        correct_ce_weight=1.0,
        answer_margin_weight=1.5,
        answer_target_margin=0.75,
        causal_margin_weight=0.75,
        causal_target_margin=0.5,
        smoothmax_temperature=0.25,
    )
    objective.backward()

    assert objective.item() > 1.1
    assert (
        records["right_answer_margin_penalty"].item() > records["left_answer_margin_penalty"].item()
    )
    assert records["answer_smoothmax_penalty"].item() > records["left_answer_margin_penalty"].item()
    assert all(value.grad is not None for value in (left, right, left_alt, right_alt))
    assert left_alt.grad is not None and left_alt.grad.item() < 0.0
    assert right_alt.grad is not None and right_alt.grad.item() < 0.0


def test_v96_smoothmax_is_zero_preserving() -> None:
    zero = torch.tensor(0.0)
    assert smoothmax_v96(zero, zero, temperature=0.25).item() == pytest.approx(0.0)


def test_v96_invariant_objective_penalizes_only_excess_nll_gap() -> None:
    left = torch.tensor(1.0, requires_grad=True)
    right = torch.tensor(1.4, requires_grad=True)
    objective, records = invariant_pair_objective_v96(
        left,
        right,
        left_class_weight=1.0,
        right_class_weight=1.0,
        family_weight=1.0,
        correct_ce_weight=1.0,
        consistency_weight=0.5,
        consistency_tolerance=0.1,
    )
    objective.backward()

    assert records["absolute_nll_gap"].item() == pytest.approx(0.4)
    assert records["consistency_penalty"].item() == pytest.approx(0.3)
    assert objective.item() == pytest.approx(1.35)
    assert left.grad is not None and right.grad is not None


def test_v96_fixed_schedule_constants_are_self_consistent() -> None:
    assert EXPECTED_RETENTION_STEPS == 1_920
    assert EXPECTED_CHANGED_PAIR_STEPS == 264
    assert EXPECTED_INVARIANT_PAIR_STEPS == 96
    assert EXPECTED_MICRO_STEPS == 2_280
    assert EXPECTED_OPTIMIZER_UPDATES == 285
    assert EXPECTED_TOTAL_NLL_FORWARDS == 3_168


class _TinyAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(5, 7, bias=False)


class _TinyLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = _TinyAttention()


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = nn.Module()
        layers: list[nn.Module] = [nn.Identity() for _ in range(35)]
        layers[9] = _TinyLayer()
        self.model.language_model.layers = nn.ModuleList(layers)


def _tiny_collection(seed: int = 96) -> LoRABankCollection:
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


_RESUME_UPDATE = 15
_RESUME_CURSOR = 120


def _step_optimizer(collection: LoRABankCollection) -> torch.optim.AdamW:
    optimizer = torch.optim.AdamW(collection.parameters(), lr=0.000075, weight_decay=0.0)
    for _step in range(_RESUME_UPDATE):
        loss = sum(parameter.square().mean() for parameter in collection.parameters())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    return optimizer


def _history(collection: LoRABankCollection) -> list[dict[str, object]]:
    history: list[dict[str, object]] = [
        {
            "update": update,
            "row_cursor": update * 8,
            "state_sha256": "a" * 64,
        }
        for update in range(1, _RESUME_UPDATE + 1)
    ]
    history[-1]["state_sha256"] = collection.bank(
        FRESH_BANK_NAME
    ).installation.state_sha256()
    return history


def test_v96_resume_round_trip_restores_two_tensors_and_adamw(tmp_path: Path) -> None:
    first = _tiny_collection()
    optimizer = _step_optimizer(first)
    expected_hash = first.bank(FRESH_BANK_NAME).installation.state_sha256()
    bindings = {"config_sha256": "a" * 64}

    checkpoint = save_resume_checkpoint_v96(
        tmp_path,
        first,
        optimizer,
        update=_RESUME_UPDATE,
        row_cursor=_RESUME_CURSOR,
        history=_history(first),
        bindings=bindings,
        row_order_sha256="b" * 64,
    )
    discovered = discover_resume_checkpoint_v96(
        tmp_path,
        bindings=bindings,
        row_order_sha256="b" * 64,
        gradient_accumulation_rows=8,
    )
    assert discovered is not None and discovered[0] == checkpoint

    second = _tiny_collection(seed=97)
    second_optimizer = torch.optim.AdamW(second.parameters(), lr=0.000075, weight_decay=0.0)
    restore_resume_checkpoint_v96(discovered[0], discovered[1], second, second_optimizer)
    assert second.bank(FRESH_BANK_NAME).installation.state_sha256() == expected_hash
    assert second_optimizer.state_dict()["state"]


def test_v96_resume_auth_rejects_extra_checkpoint_file(tmp_path: Path) -> None:
    collection = _tiny_collection()
    optimizer = _step_optimizer(collection)
    checkpoint = save_resume_checkpoint_v96(
        tmp_path,
        collection,
        optimizer,
        update=_RESUME_UPDATE,
        row_cursor=_RESUME_CURSOR,
        history=_history(collection),
        bindings={"config_sha256": "a" * 64},
        row_order_sha256="b" * 64,
    )
    (checkpoint / "unexpected.txt").write_text("tamper", encoding="utf-8")
    with pytest.raises(ValueError, match="file inventory changed"):
        discover_resume_checkpoint_v96(
            tmp_path,
            bindings={"config_sha256": "a" * 64},
            row_order_sha256="b" * 64,
            gradient_accumulation_rows=8,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", 95),
        ("status", "complete"),
        ("update", 285),
        ("row_cursor", 16),
    ],
)
def test_v96_resume_auth_rejects_metadata_lifecycle_tamper(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    collection = _tiny_collection()
    optimizer = _step_optimizer(collection)
    checkpoint = save_resume_checkpoint_v96(
        tmp_path,
        collection,
        optimizer,
        update=_RESUME_UPDATE,
        row_cursor=_RESUME_CURSOR,
        history=_history(collection),
        bindings={"config_sha256": "a" * 64},
        row_order_sha256="b" * 64,
    )
    state_path = checkpoint / "state.json"
    metadata = json.loads(state_path.read_text(encoding="utf-8"))
    metadata[field] = value
    if field == "update":
        metadata["row_cursor"] = 2_280
        metadata["history"] = [
            {"update": index, "row_cursor": index * 8} for index in range(1, 286)
        ]
    state_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="authentication failed"):
        discover_resume_checkpoint_v96(
            tmp_path,
            bindings={"config_sha256": "a" * 64},
            row_order_sha256="b" * 64,
            gradient_accumulation_rows=8,
        )


def test_v96_resume_auth_rejects_impossible_non_checkpoint_update(tmp_path: Path) -> None:
    collection = _tiny_collection()
    optimizer = _step_optimizer(collection)
    checkpoint = save_resume_checkpoint_v96(
        tmp_path,
        collection,
        optimizer,
        update=_RESUME_UPDATE,
        row_cursor=_RESUME_CURSOR,
        history=_history(collection),
        bindings={"config_sha256": "a" * 64},
        row_order_sha256="b" * 64,
    )
    impossible_update = 16
    destination = tmp_path / f"update_{impossible_update:06d}"
    checkpoint.rename(destination)
    state_path = destination / "state.json"
    metadata = json.loads(state_path.read_text(encoding="utf-8"))
    metadata["update"] = impossible_update
    metadata["row_cursor"] = impossible_update * 8
    state_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="authentication failed"):
        discover_resume_checkpoint_v96(
            tmp_path,
            bindings={"config_sha256": "a" * 64},
            row_order_sha256="b" * 64,
            gradient_accumulation_rows=8,
        )


@pytest.mark.parametrize(
    "target", ["checkpoint", "broken_checkpoint", "state", "tensor", "broken_tensor"]
)
def test_v96_resume_auth_rejects_symlinked_checkpoint_material(
    tmp_path: Path,
    target: str,
) -> None:
    root = tmp_path / "work"
    outside = tmp_path / "outside"
    collection = _tiny_collection()
    optimizer = _step_optimizer(collection)
    checkpoint = save_resume_checkpoint_v96(
        root,
        collection,
        optimizer,
        update=_RESUME_UPDATE,
        row_cursor=_RESUME_CURSOR,
        history=_history(collection),
        bindings={"config_sha256": "a" * 64},
        row_order_sha256="b" * 64,
    )
    if target == "checkpoint":
        checkpoint.rename(outside)
        checkpoint.symlink_to(outside, target_is_directory=True)
    elif target == "broken_checkpoint":
        checkpoint.rename(outside)
        checkpoint.symlink_to(tmp_path / "missing", target_is_directory=True)
    else:
        filename = "state.json" if target == "state" else "state.safetensors"
        original = checkpoint / filename
        if target == "broken_tensor":
            original.unlink()
            original.symlink_to(tmp_path / "missing.safetensors")
        else:
            outside.mkdir()
            moved = outside / filename
            original.rename(moved)
            original.symlink_to(moved)

    with pytest.raises((ValueError, FileNotFoundError), match="symlink|unlinked"):
        discover_resume_checkpoint_v96(
            root,
            bindings={"config_sha256": "a" * 64},
            row_order_sha256="b" * 64,
            gradient_accumulation_rows=8,
        )


def test_v96_resume_auth_rejects_optimizer_group_tamper(tmp_path: Path) -> None:
    collection = _tiny_collection()
    optimizer = _step_optimizer(collection)
    checkpoint = save_resume_checkpoint_v96(
        tmp_path,
        collection,
        optimizer,
        update=_RESUME_UPDATE,
        row_cursor=_RESUME_CURSOR,
        history=_history(collection),
        bindings={"config_sha256": "a" * 64},
        row_order_sha256="b" * 64,
    )
    state_path = checkpoint / "state.json"
    metadata = json.loads(state_path.read_text(encoding="utf-8"))
    metadata["optimizer_param_groups"][0]["lr"] = 0.001
    state_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="authentication failed"):
        discover_resume_checkpoint_v96(
            tmp_path,
            bindings={"config_sha256": "a" * 64},
            row_order_sha256="b" * 64,
            gradient_accumulation_rows=8,
        )


def test_v96_resume_auth_rejects_optimizer_step_tamper(tmp_path: Path) -> None:
    collection = _tiny_collection()
    optimizer = _step_optimizer(collection)
    checkpoint = save_resume_checkpoint_v96(
        tmp_path,
        collection,
        optimizer,
        update=_RESUME_UPDATE,
        row_cursor=_RESUME_CURSOR,
        history=_history(collection),
        bindings={"config_sha256": "a" * 64},
        row_order_sha256="b" * 64,
    )
    tensor_path = checkpoint / "state.safetensors"
    archive = load_file(str(tensor_path), device="cpu")
    archive["optimizer.0.step"] = torch.tensor(2.0)
    save_file(archive, str(tensor_path))
    state_path = checkpoint / "state.json"
    metadata = json.loads(state_path.read_text(encoding="utf-8"))
    metadata["tensor_file_sha256"] = sha256_file_v85(tensor_path)
    state_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="authentication failed"):
        discover_resume_checkpoint_v96(
            tmp_path,
            bindings={"config_sha256": "a" * 64},
            row_order_sha256="b" * 64,
            gradient_accumulation_rows=8,
        )


def test_v96_resume_auth_rejects_history_state_hash_tamper(tmp_path: Path) -> None:
    collection = _tiny_collection()
    optimizer = _step_optimizer(collection)
    checkpoint = save_resume_checkpoint_v96(
        tmp_path,
        collection,
        optimizer,
        update=_RESUME_UPDATE,
        row_cursor=_RESUME_CURSOR,
        history=_history(collection),
        bindings={"config_sha256": "a" * 64},
        row_order_sha256="b" * 64,
    )
    state_path = checkpoint / "state.json"
    metadata = json.loads(state_path.read_text(encoding="utf-8"))
    metadata["history"][-1]["state_sha256"] = "0" * 64
    state_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="authentication failed"):
        discover_resume_checkpoint_v96(
            tmp_path,
            bindings={"config_sha256": "a" * 64},
            row_order_sha256="b" * 64,
            gradient_accumulation_rows=8,
        )


def test_v96_candidate_is_create_once_and_two_tensor_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from semantic_3d_chat.training import train_v96_atomic_pair_repair as trainer

    collection = _tiny_collection()
    monkeypatch.setattr(trainer, "FRESH_PARAMETER_COUNT", 24)
    candidate = tmp_path / "candidate"
    metadata = publish_fixed_final_candidate_v96(
        candidate,
        collection,
        bindings={"config_sha256": "c" * 64, "fixed_final_optimizer_updates": 285},
    )

    tensors = load_file(str(candidate / "bridge.safetensors"))
    assert metadata["artifact"] == CANDIDATE_ARTIFACT
    assert metadata["parent"] == "v95_fixed_final_nonpromoted_optimization_parent"
    assert len(metadata["tensor_inventory"]) == len(tensors) == 2
    assert metadata["known_development_scored"] is False
    assert metadata["deferred_final_generated"] is False
    assert (
        json.loads((candidate / "runtime_metadata.json").read_text())[
            "questions_or_answers_serialized"
        ]
        is False
    )

    loaded = _tiny_collection(seed=98)
    monkeypatch.setattr(trainer, "FRESH_PARAMETER_COUNT", 24)
    loaded_metadata = load_fixed_final_bridge_v96(loaded, candidate)
    assert loaded_metadata["state_sha256"] == metadata["state_sha256"]
    with pytest.raises(FileExistsError, match="create-once"):
        publish_fixed_final_candidate_v96(candidate, collection, bindings={})
