from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file
from torch import nn

from semantic_3d_chat.config import load_config
from semantic_3d_chat.language.lora import (
    LoRAAdapterState,
    LoRAInstallation,
    LoRALinear,
    LoRASettings,
    tensor_state_sha256,
)
from semantic_3d_chat.training.pair_curriculum import build_exact_question_pair_units
from semantic_3d_chat.training.train_block_cross_v35 import v35_settings
from semantic_3d_chat.training.train_joint_block_cross_v36 import (
    _BANK_OPTIMIZER_PARAMETER_NAMES,
    _CORE_PARAMETER_NAMES,
    _optimizer_step_audit,
    assert_v36_trainable_surface,
    build_v35_schedule,
    complete_physical_pair_coverage,
    construct_v36_source_core,
    freeze_for_v36,
    latest_v36_resume_checkpoint,
    load_v35_train_qa_records,
    preflight_v36,
    require_exact_v35_source,
    require_v35_terminal_gate,
    set_v36_optimizer_stage,
    v36_broad_calibration_records,
    v36_contract,
    v36_optimizer,
    v36_settings,
    v36_update16_gate,
    v36_update32_gate,
    v36_update64_gate,
    v36_weighted_objective,
)

V36_CONFIG = Path("configs/experiments/gemma4_diverse28_joint_block_cross_v36.yaml")
BANK_NAME = "extension_v30_joint_pair_query"


def test_v36_contract_pins_terminal_source_surfaces_schedule_and_gates() -> None:
    config = load_config(V36_CONFIG)
    settings = v36_settings(config)
    contract = v36_contract(config)
    terminal = require_v35_terminal_gate(config)
    source, metadata = require_exact_v35_source(config)
    assert source.name == "update_032"
    assert metadata["optimizer_step"] == 32
    assert terminal["sha256"] == (
        "88205d018de14fc0518fe695bf7420c44ac832a1ee95eea0e2ae1f41deff4a27"
    )
    assert settings.saved_optimizer_steps == (*range(0, 97, 8), 100)
    assert settings.decoder_learning_rate == 2e-5
    assert contract.source_tensor_state_sha256 == (
        "1fe8f278460faeb1e13d9da09051a497965a566565c79a4f6ea28c56a9120326"
    )
    assert contract.core_source_state_sha256 == (
        "75af995833d9387e3eb01fb022eaade7327e44960466671123a51aa43afa4cf3"
    )
    assert contract.frozen_nonauthorized_state_sha256 == (
        "b394d502f0c32a694c2d1a448cdf3849c47efc4058cb1f1331fe4a97d381b1dc"
    )


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("training.v36_joint_block_cross", "optimizer_steps", 101),
        ("training.v36_joint_block_cross", "decoder_learning_rate", 2e-4),
        ("training.v36_joint_block_cross", "cross_prefix_flip_weight", 4.0),
        ("v36_joint_block_cross", "source_optimizer_step", 64),
        ("v36_joint_block_cross", "source_block_core_state_sha256", "0" * 64),
        ("v36_joint_block_cross", "source_v35_optimizer_state_loaded", True),
        ("v36_joint_block_cross", "validation_qa_loaded_during_training", True),
        ("v36_joint_block_cross", "frozen_nonauthorized_state_sha256", "0" * 64),
        ("v36_joint_block_cross", "v35_terminal_gate_report_sha256", "0" * 64),
    ],
)
def test_v36_contract_fails_closed(section: str, field: str, value: object) -> None:
    config = copy.deepcopy(load_config(V36_CONFIG))
    target = config
    for key in section.split("."):
        target = target[key]
    target[field] = value
    with pytest.raises((TypeError, ValueError)):
        v36_contract(config)


def test_v36_preflight_is_gemma_map_validation_qa_oracle_and_final_free() -> None:
    report = preflight_v36(load_config(V36_CONFIG))
    assert report["passed"] is True
    assert report["source_optimizer_step"] == 32
    assert report["source_block_core_state_sha256"].startswith("75af9958")
    assert report["exact_stopped_v35_update32_selected_as_source"] is True
    assert report["v35_optimizer_state_loaded"] is False
    assert report["gemma_loaded"] is False
    assert report["scene_maps_loaded"] is False
    assert report["validation_qa_loaded"] is False
    assert report["oracle_loaded"] is False
    assert report["final_test_scenes_touched"] is False


def test_v36_preflight_never_opens_validation_jsonl(monkeypatch: pytest.MonkeyPatch) -> None:
    original = Path.read_text

    def guarded(path: Path, *args, **kwargs):
        if path.name == "validation.jsonl":
            raise AssertionError("V36 training preflight attempted to open validation QA")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded)
    report = preflight_v36(load_config(V36_CONFIG))
    assert report["train_qa_loaded"] is True
    assert report["validation_qa_loaded"] is False


def test_v36_reuses_exact_v35_schedule_and_train_only_broad_calibration() -> None:
    config = load_config(V36_CONFIG)
    records, audit = load_v35_train_qa_records(config)
    units = build_exact_question_pair_units(records)
    schedule, schedule_audit = build_v35_schedule(
        records,
        units,
        settings=v35_settings(config),
        seed=int(config["seed"]),
    )
    broad = v36_broad_calibration_records(schedule)
    terminal = require_v35_terminal_gate(config)["report"]
    assert schedule_audit["schedule_sha256"] == terminal["schedule_sha256"]
    assert len(schedule) == 100
    assert len(broad) == 48
    assert len({(row.scene_id, row.question_id) for row in broad}) == 48
    assert all(row.counterfactual_expected_change is not True for row in broad)
    assert audit["validation_qa_loaded"] is False


def test_v36_source_core_is_learned_v35_not_fresh_zero_core() -> None:
    core = construct_v36_source_core(load_config(V36_CONFIG), device=torch.device("cpu"))
    assert core.state_sha256() == (
        "75af995833d9387e3eb01fb022eaade7327e44960466671123a51aa43afa4cf3"
    )
    assert torch.count_nonzero(dict(core.named_parameters())["w_o"]).item() > 0


def test_v36_frozen_hash_includes_core_buffers_and_excludes_only_authorized_params() -> None:
    tensors = load_file(
        "data_gemma4/checkpoints/gemma4_v35_diverse28_block_cross/update_032/adapter.safetensors",
        device="cpu",
    )
    bank_prefix = f"lora_banks.{BANK_NAME}."
    frozen = {
        name: value
        for name, value in tensors.items()
        if not name.startswith(bank_prefix)
        and name
        not in {
            "block_cross_residual.w_q",
            "block_cross_residual.w_k",
            "block_cross_residual.w_v",
            "block_cross_residual.w_o",
        }
    }
    assert len(frozen) == 167
    assert tensor_state_sha256(frozen) == (
        "b394d502f0c32a694c2d1a448cdf3849c47efc4058cb1f1331fe4a97d381b1dc"
    )
    assert any(name.startswith("block_cross_residual.") for name in frozen)


def _fake_bundle(core: nn.Module) -> SimpleNamespace:
    targets = tuple(
        f"model.language_model.layers.{index}.self_attn.q_proj" for index in range(18, 22)
    )
    outputs = (2048, 4096, 2048, 2048)
    adapters = tuple(
        LoRALinear(
            nn.Linear(1536, out_features, bias=False, device="meta"),
            rank=8,
            alpha=16.0,
        )
        for out_features in outputs
    )
    settings = LoRASettings(
        enabled=True,
        rank=8,
        alpha=16.0,
        dropout=0.0,
        target_modules=targets,
    )
    installation = LoRAInstallation(
        settings=settings,
        adapters=adapters,
        state_module=LoRAAdapterState(targets, adapters),
    )
    language_model = nn.ModuleList(adapters)
    collection = SimpleNamespace(
        bank=lambda name: (
            SimpleNamespace(installation=installation)
            if name == BANK_NAME
            else (_ for _ in ()).throw(KeyError(name))
        )
    )
    return SimpleNamespace(
        trainable_bank_name=BANK_NAME,
        lora_installation=collection,
        language=SimpleNamespace(model=language_model),
        checkpoint_modules={
            "inherited": nn.Linear(2, 2),
            f"lora_banks.{BANK_NAME}": installation.state_module,
            "block_cross_residual": core,
        },
    )


def test_v36_fresh_adam_stages_lora_only_then_joint_exact_surfaces() -> None:
    config = load_config(V36_CONFIG)
    settings = v36_settings(config)
    core = construct_v36_source_core(config, device=torch.device("cpu"))
    bundle = _fake_bundle(core)
    parameters = freeze_for_v36(bundle, core, optimizer_step=0)
    surface0 = assert_v36_trainable_surface(bundle, core, optimizer_step=0)
    optimizer = v36_optimizer(bundle, core, settings)
    assert optimizer.state == {}
    assert sum(parameter.numel() for parameter in parameters) == 1_114_112
    assert surface0["active_stage"] == "lora_only"
    assert [group["name"] for group in optimizer.param_groups] == [
        "block_cross_residual.qkv",
        "block_cross_residual.output",
        f"lora_banks.{BANK_NAME}",
    ]
    assert [group["parameter_names"] for group in optimizer.param_groups] == [
        list(_CORE_PARAMETER_NAMES[:3]),
        list(_CORE_PARAMETER_NAMES[3:]),
        list(_BANK_OPTIMIZER_PARAMETER_NAMES),
    ]
    set_v36_optimizer_stage(
        bundle=bundle,
        block_cross_residual=core,
        optimizer=optimizer,
        optimizer_step_to_run=8,
        settings=settings,
    )
    assert not any(parameter.requires_grad for parameter in core.parameters())
    assert optimizer.param_groups[0]["lr"] == 0.0
    assert optimizer.param_groups[1]["lr"] == 0.0
    assert optimizer.param_groups[2]["lr"] == 2e-5
    set_v36_optimizer_stage(
        bundle=bundle,
        block_cross_residual=core,
        optimizer=optimizer,
        optimizer_step_to_run=9,
        settings=settings,
    )
    assert all(parameter.requires_grad for parameter in core.parameters())
    assert optimizer.param_groups[0]["lr"] == 1e-4
    assert optimizer.param_groups[1]["lr"] == 2.5e-5
    assert_v36_trainable_surface(bundle, core, optimizer_step=8, optimizer=optimizer)


def _pair_metrics(
    *, complete: int = 15, cross: int = 20, positive: int = 40, coverage: int = 7
) -> dict[str, object]:
    pair_ids = [f"pair_{index:06d}" for index in range(coverage)]
    rows = [{"pair_id": pair_id, "complete": True} for pair_id in pair_ids]
    rows.extend({"pair_id": "pair_other", "complete": False} for _ in range(25 - coverage))
    return {
        "complete_units": complete,
        "cross_prefix_complete_units": cross,
        "positive_sides": positive,
        "mean_cross_prefix_margin": 1.5,
        "complete_units_by_family": {
            "book_support": 1,
            "mirror_lr": 1,
            "picture_support": 1,
        },
        "units": rows,
    }


def test_v36_gates_use_train_teacher_broad_greedy_family_and_integrity_evidence() -> None:
    contract = v36_contract(load_config(V36_CONFIG))
    metrics = _pair_metrics()
    assert complete_physical_pair_coverage(metrics) == 7
    gate16 = v36_update16_gate(
        pair_metrics=metrics,
        broad_nll=1.01,
        source_broad_nll=1.0,
        residual_rms=0.05,
        decoder_bank_state_sha256="f" * 64,
        frozen_nonauthorized_state_sha256=contract.frozen_nonauthorized_state_sha256,
        contract=contract,
    )
    assert gate16["passed"] is True
    gate32 = v36_update32_gate(
        update16_gate=gate16,
        pair_metrics=metrics,
        broad_nll=1.02,
        source_broad_nll=1.0,
        residual_rms=0.05,
        contract=contract,
    )
    assert gate32["passed"] is True
    greedy = {
        "complete_units": 6,
        "complete_units_by_family": {
            "book_support": 1,
            "mirror_lr": 1,
            "picture_support": 1,
        },
        "broad_exact_accuracy": 0.58,
    }
    source_greedy = {"broad_exact_accuracy": 0.60}
    gate64 = v36_update64_gate(
        update32_gate=gate32,
        pair_metrics=metrics,
        greedy_metrics=greedy,
        source_greedy_metrics=source_greedy,
        residual_rms=0.09,
        contract=contract,
    )
    assert gate64["passed"] is True
    failed = copy.deepcopy(greedy)
    failed["complete_units_by_family"]["picture_support"] = 0
    assert (
        v36_update64_gate(
            update32_gate=gate32,
            pair_metrics=metrics,
            greedy_metrics=failed,
            source_greedy_metrics=source_greedy,
            residual_rms=0.09,
            contract=contract,
        )["passed"]
        is False
    )


def test_v36_weighted_objective_is_exact_v35_isolation_formula() -> None:
    settings = v36_settings(load_config(V36_CONFIG))
    values = [torch.tensor(value) for value in (2.0, 3.0, 4.0, 5.0, 6.0)]
    observed = v36_weighted_objective(
        broad_nll=values[0],
        pair_correct_nll=values[1],
        side_hinge=values[2],
        cross_prefix_flip_hinge=values[3],
        normalized_residual_penalty=values[4],
        settings=settings,
    )
    assert observed.item() == pytest.approx(0.25 * 2 + 0.5 * 3 + 4 * 4 + 8 * 5 + 0.001 * 6)


def _optimizer_audit_fixture(step: int) -> tuple[dict[str, object], dict[str, torch.Tensor]]:
    ordered_names = {
        "block_cross_residual.qkv": list(_CORE_PARAMETER_NAMES[:3]),
        "block_cross_residual.output": list(_CORE_PARAMETER_NAMES[3:]),
        f"lora_banks.{BANK_NAME}": list(_BANK_OPTIMIZER_PARAMETER_NAMES),
    }
    tensors = {
        name: torch.zeros(index + 1)
        for index, name in enumerate((*_CORE_PARAMETER_NAMES, *_BANK_OPTIMIZER_PARAMETER_NAMES))
    }
    groups: list[dict[str, object]] = []
    state: dict[int, dict[str, torch.Tensor]] = {}
    next_id = 0
    for group_name, parameter_names in ordered_names.items():
        parameter_ids = list(range(next_id, next_id + len(parameter_names)))
        next_id += len(parameter_names)
        groups.append(
            {
                "name": group_name,
                "params": parameter_ids,
                "parameter_names": parameter_names,
                "lr": (
                    2e-5
                    if group_name.startswith("lora_banks")
                    else (0.0 if step <= 8 else (1e-4 if group_name.endswith("qkv") else 2.5e-5))
                ),
                "weight_decay": 0.0,
            }
        )
        for parameter_id, tensor_name in zip(parameter_ids, parameter_names, strict=True):
            if step <= 8 and tensor_name.startswith("block_cross_residual"):
                continue
            tensor_step = step if tensor_name.startswith("lora_banks") else step - 8
            state[parameter_id] = {
                "step": torch.tensor(float(tensor_step)),
                "exp_avg": torch.zeros_like(tensors[tensor_name]),
                "exp_avg_sq": torch.zeros_like(tensors[tensor_name]),
            }
    return {"state": state, "param_groups": groups}, tensors


@pytest.mark.parametrize("step", [8, 16, 100])
def test_v36_optimizer_audit_proves_exact_parameter_moment_mapping(
    tmp_path: Path, step: int
) -> None:
    checkpoint = tmp_path / f"update_{step:03d}"
    checkpoint.mkdir()
    payload, tensors = _optimizer_audit_fixture(step)
    torch.save(payload, checkpoint / "optimizer.pt")
    audit = _optimizer_step_audit(checkpoint, expected_step=step, tensors=tensors)
    assert audit["exact_parameter_order_verified"] is True
    assert audit["lora_optimizer_step"] == step
    assert audit["block_core_optimizer_step"] == (None if step == 8 else step - 8)


def test_v36_optimizer_audit_rejects_same_shape_parameter_reorder(tmp_path: Path) -> None:
    checkpoint = tmp_path / "update_016"
    checkpoint.mkdir()
    payload, tensors = _optimizer_audit_fixture(16)
    group = payload["param_groups"][0]
    group["parameter_names"][0], group["parameter_names"][1] = (
        group["parameter_names"][1],
        group["parameter_names"][0],
    )
    torch.save(payload, checkpoint / "optimizer.pt")
    with pytest.raises(ValueError, match="parameter order changed"):
        _optimizer_step_audit(checkpoint, expected_step=16, tensors=tensors)


def test_v36_optimizer_audit_rejects_same_shape_parameter_id_swap(tmp_path: Path) -> None:
    checkpoint = tmp_path / "update_016"
    checkpoint.mkdir()
    payload, tensors = _optimizer_audit_fixture(16)
    parameter_ids = payload["param_groups"][0]["params"]
    parameter_ids[0], parameter_ids[1] = parameter_ids[1], parameter_ids[0]
    torch.save(payload, checkpoint / "optimizer.pt")
    with pytest.raises(ValueError, match="parameter IDs changed"):
        _optimizer_step_audit(checkpoint, expected_step=16, tensors=tensors)


def test_v36_optimizer_audit_rejects_moment_shape_mismatch(tmp_path: Path) -> None:
    checkpoint = tmp_path / "update_016"
    checkpoint.mkdir()
    payload, tensors = _optimizer_audit_fixture(16)
    first_parameter_id = payload["param_groups"][0]["params"][0]
    payload["state"][first_parameter_id]["exp_avg"] = torch.zeros(999)
    torch.save(payload, checkpoint / "optimizer.pt")
    with pytest.raises(ValueError, match="Adam exp_avg is invalid"):
        _optimizer_step_audit(checkpoint, expected_step=16, tensors=tensors)


def test_v36_optimizer_audit_rejects_missing_or_wrong_staged_state(tmp_path: Path) -> None:
    checkpoint = tmp_path / "update_008"
    checkpoint.mkdir()
    payload, tensors = _optimizer_audit_fixture(8)
    payload["state"].pop(next(iter(payload["state"])))
    torch.save(payload, checkpoint / "optimizer.pt")
    with pytest.raises(ValueError, match="Adam state is incomplete"):
        _optimizer_step_audit(checkpoint, expected_step=8, tensors=tensors)


def test_v36_latest_resume_rejects_incomplete_next_arm(tmp_path: Path) -> None:
    contract = v36_contract(load_config(V36_CONFIG))
    update0 = tmp_path / "update_000"
    update0.mkdir()
    for name in ("adapter.safetensors", "metadata.json", "runtime_metadata.json"):
        (update0 / name).touch()
    assert latest_v36_resume_checkpoint(tmp_path, contract) == update0
    update8 = tmp_path / "update_008"
    update8.mkdir()
    (update8 / "adapter.safetensors").touch()
    with pytest.raises(ValueError, match="incomplete arm"):
        latest_v36_resume_checkpoint(tmp_path, contract)


def test_v36_docs_and_make_have_preflight_train_without_final_bypass() -> None:
    makefile = Path("Makefile").read_text(encoding="utf-8")
    readme = Path("README.md").read_text(encoding="utf-8")
    assert "gemma4-v36-preflight-joint-block-cross" in makefile
    assert "gemma4-v36-train-joint-block-cross" in makefile
    assert "gemma4-v36-evaluate-final" not in makefile
    assert "V36 bounded joint decoder readout" in readme


def test_v36_config_contains_no_validation_or_final_artifact_path() -> None:
    raw = V36_CONFIG.read_text(encoding="utf-8")
    assert "validation.jsonl" not in raw
    assert "scene_000025" in raw  # opaque absence lock, not an artifact path
    assert "data/oracle" not in raw
    assert (
        json.loads(
            Path("reports/gemma4/metrics/v35_update32_terminal_gate.json").read_text(
                encoding="utf-8"
            )
        )["conditional_authorization"]["authorized_existing_lora_bank"]
        == BANK_NAME
    )
