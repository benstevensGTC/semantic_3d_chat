from __future__ import annotations

import copy
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from semantic_3d_chat.config import config_hash, load_config
from semantic_3d_chat.data.dataset import QARecord
from semantic_3d_chat.evaluation.v33_environmental_selector import (
    _validate_optimizer_state_step,
    v33_chat_promotion_checks,
)
from semantic_3d_chat.scene_encoder.dense_sidecar_adapter import DenseSidecarAdapter
from semantic_3d_chat.training.checkpointing import runtime_checkpoint_metadata
from semantic_3d_chat.training.pair_curriculum import CounterfactualPairUnit
from semantic_3d_chat.training.train_environmental_sidecar_v33 import (
    _optimizer,
    assert_deferred_final_scenes_absent,
    assert_v33_trainable_surface,
    build_v33_environmental_schedule,
    freeze_for_v33,
    latest_v33_resume_checkpoint,
    prefix_separation_ratios,
    require_v32_rejection,
    update64_environmental_gate,
    v32_rejection_status,
    v33_contract,
    v33_settings,
    validate_v33_resume_checkpoint,
)

V33_CONFIG = Path("configs/experiments/gemma4_diverse28_environmental_sidecar_v33.yaml")


def _record(
    index: int,
    *,
    scene_id: str,
    answer_type: str,
    pair_id: str | None = None,
    question_key: str | None = None,
    role: str | None = None,
) -> QARecord:
    return QARecord(
        scene_id=scene_id,
        question_id=f"q_{index:04d}",
        question=f"opaque question {question_key or index}",
        answer="left" if role != "counterfactual" else "right",
        answer_type=answer_type,
        target_xyz=None,
        counterfactual_pair_id=pair_id,
        counterfactual_question_key=question_key,
        counterfactual_expected_change=pair_id is not None,
        counterfactual_role=role,
        counterfactual_change_type="opaque_change" if pair_id else None,
    )


def _schedule_fixture() -> tuple[list[QARecord], list[CounterfactualPairUnit]]:
    records = [
        _record(
            index,
            scene_id=f"scene_{11 + index % 8:06d}",
            answer_type=("presence", "count", "attribute", "spatial_relation")[index % 4],
        )
        for index in range(100)
    ]
    units: list[CounterfactualPairUnit] = []
    for index in range(25):
        pair_id = f"pair_{index:03d}"
        question_key = f"unit_{index:03d}"
        first = _record(
            1_000 + 2 * index,
            scene_id=f"scene_{11 + 2 * index % 8:06d}",
            answer_type="spatial_relation",
            pair_id=pair_id,
            question_key=question_key,
            role="reference",
        )
        second = _record(
            1_001 + 2 * index,
            scene_id=f"scene_{11 + (2 * index + 1) % 8:06d}",
            answer_type="spatial_relation",
            pair_id=pair_id,
            question_key=question_key,
            role="counterfactual",
        )
        units.append(CounterfactualPairUnit(pair_id, question_key, first, second))
        records.extend((first, second))
    return records, units


def test_v33_contract_locks_environmental_only_surface_and_schedule() -> None:
    config = load_config(V33_CONFIG)
    settings = v33_settings(config)
    contract = v33_contract(config)
    assert settings.optimizer_steps == 100
    assert settings.saved_optimizer_steps == (*range(0, 100, 8), 100)
    assert settings.output_learning_rate == 2.5e-5
    assert settings.hidden_learning_rate == 1e-4
    assert settings.position_learning_rate == 1e-4
    assert contract.minimum_pair_unit_recurrence == 4
    assert contract.v32_selection_report_sha256 == (
        "2ffeb2655cd6a8627ea9e06c8f261113b0b225a1b39de4eb32126693063c13b7"
    )
    assert contract.v31.deferred_final_scene_ids == tuple(
        f"scene_{index:06d}" for index in range(25, 31)
    )


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("training.v33_environmental", "optimizer_steps", 80),
        ("training.v33_environmental", "hidden_learning_rate", 2.5e-5),
        ("training.v33_environmental", "position_gradient_clip_norm", 2.0),
        ("v33_environmental", "exact_trainable_parameter_count", 329_216),
        ("v33_environmental", "gemma_decoder_frozen", False),
        ("v33_environmental", "saved_optimizer_steps", [0, 8, 100]),
        ("v33_environmental", "v32_selection_report_sha256", "0" * 64),
        ("v33_environmental", "chat_promotion_requires_each_validation_family", False),
    ],
)
def test_v33_contract_fails_closed(section: str, field: str, value: object) -> None:
    config = copy.deepcopy(load_config(V33_CONFIG))
    target = config
    for key in section.split("."):
        target = target[key]
    target[field] = value
    with pytest.raises((TypeError, ValueError)):
        v33_contract(config)


def test_v33_schedule_has_100_true_steps_and_exact_fourfold_pair_recurrence() -> None:
    records, units = _schedule_fixture()
    settings = v33_settings(load_config(V33_CONFIG))
    schedule, audit = build_v33_environmental_schedule(records, units, settings=settings, seed=33)
    assert [row.optimizer_step for row in schedule] == list(range(1, 101))
    appearances = Counter(
        (unit.pair_id, unit.question_key) for row in schedule for unit in row.pair_units
    )
    assert len(appearances) == 25
    assert set(appearances.values()) == {4}
    assert audit["pair_unit_minimum_recurrence"] == 4
    assert audit["pair_unit_maximum_recurrence"] == 4
    assert audit["true_optimizer_step_per_schedule_row"] is True


def test_v33_is_authorized_by_only_the_exact_terminal_v32_rejection(tmp_path: Path) -> None:
    config = load_config(V33_CONFIG)
    assert v32_rejection_status(config)["training_authorized"] is True
    assert require_v32_rejection(config)["status"] == "rejected"

    forged = copy.deepcopy(config)
    report = json.loads(
        Path(config["v33_environmental"]["v32_selection_report"]).read_text(encoding="utf-8")
    )
    report["passed"] = True
    forged_path = tmp_path / "forged.json"
    forged_path.write_text(json.dumps(report), encoding="utf-8")
    forged["v33_environmental"]["v32_selection_report"] = str(forged_path)
    with pytest.raises(ValueError, match="hash differs"):
        v32_rejection_status(forged)


def test_v33_final_footprint_guard_is_fail_closed(tmp_path: Path) -> None:
    config = copy.deepcopy(load_config(V33_CONFIG))
    config["paths"]["data_root"] = str(tmp_path / "data")
    config["paths"]["features_root"] = str(tmp_path / "features")
    config["paths"]["maps_root"] = str(tmp_path / "maps")
    assert_deferred_final_scenes_absent(config)
    forbidden = tmp_path / "maps" / "scene_000025"
    forbidden.mkdir(parents=True)
    with pytest.raises(RuntimeError, match="deferred-final footprint"):
        assert_deferred_final_scenes_absent(config)


class _FakeInstallation(torch.nn.Module):
    pass


def _bundle() -> SimpleNamespace:
    sidecar = DenseSidecarAdapter(
        scene_dim=1536,
        latent_count=256,
        width=128,
        fourier_bands=8,
        max_direct_scale=0.25,
        initialization_seed=28028,
    )
    decoder = torch.nn.Linear(2, 2)
    bank_module = _FakeInstallation()
    bank_module.weight = torch.nn.Parameter(torch.ones(3))
    bank = SimpleNamespace(settings=SimpleNamespace(name="opaque_bank"), installation=bank_module)
    return SimpleNamespace(
        language=SimpleNamespace(model=decoder),
        dense_sidecar_adapter=sidecar,
        checkpoint_modules={
            "dense_sidecar_adapter": sidecar,
            "opaque_decoder_bank": bank_module,
        },
        lora_installation=SimpleNamespace(banks=(bank,)),
    )


def test_v33_freeze_and_optimizer_expose_only_three_environmental_groups() -> None:
    bundle = _bundle()
    trainable = freeze_for_v33(bundle)
    surface = assert_v33_trainable_surface(bundle)
    optimizer = _optimizer(bundle, v33_settings(load_config(V33_CONFIG)))
    assert_v33_trainable_surface(bundle, optimizer)
    assert sum(parameter.numel() for parameter in trainable) == 404_608
    assert surface["group_parameter_counts"] == {
        "output": 198_144,
        "sidecar_hidden": 199_808,
        "position": 6_656,
    }
    assert [group["name"] for group in optimizer.param_groups] == [
        "dense_sidecar_adapter.output",
        "dense_sidecar_adapter.sidecar_hidden",
        "dense_sidecar_adapter.position",
    ]
    assert not any(parameter.requires_grad for parameter in bundle.language.model.parameters())
    assert not any(
        parameter.requires_grad
        for bank in bundle.lora_installation.banks
        for parameter in bank.installation.parameters()
    )


def test_v33_prefix_ratio_and_update64_gates() -> None:
    baseline = {
        "rms_by_validation_family": {
            "book_support": 1.0,
            "mirror_lr": 2.0,
            "picture_support": 1.0,
        },
        "weak_pair_mean_rms": 1.0,
        "unrelated_mean_rms": 4.0,
    }
    current = {
        "rms_by_validation_family": {
            "book_support": 1.3,
            "mirror_lr": 2.2,
            "picture_support": 1.2,
        },
        "weak_pair_mean_rms": 1.25,
        "unrelated_mean_rms": 5.0,
    }
    assert prefix_separation_ratios(current, baseline)["weak_pair_mean"] == 1.25
    base_family = {
        family: {"complete_units": 0, "mean_margin": -1.0}
        for family in ("book_support", "mirror_lr", "picture_support")
    }
    passed_family = copy.deepcopy(base_family)
    passed_family["book_support"] = {"complete_units": 1, "mean_margin": -0.9}
    passed_family["picture_support"]["mean_margin"] = -0.8
    assert update64_environmental_gate(passed_family, base_family)["passed"] is True
    passed_family["picture_support"]["mean_margin"] = -1.1
    assert update64_environmental_gate(passed_family, base_family)["passed"] is False


def _write_resume_u64(root: Path, config: dict, *, gate_passes: bool) -> dict:
    contract = v33_contract(config)
    for step in contract.saved_optimizer_steps:
        if step > 64:
            break
        checkpoint = root / f"update_{step:03d}"
        checkpoint.mkdir(parents=True)
        for name in ("adapter.safetensors", "metadata.json", "runtime_metadata.json"):
            (checkpoint / name).touch()
        if step:
            (checkpoint / "optimizer.pt").touch()
    checkpoint = root / "update_064"
    family0 = {
        family: {"complete_units": 0, "mean_margin": -1.0}
        for family in ("book_support", "mirror_lr", "picture_support")
    }
    family64 = copy.deepcopy(family0)
    family64["book_support"] = {
        "complete_units": 1 if gate_passes else 0,
        "mean_margin": -0.9,
    }
    family64["picture_support"]["mean_margin"] = -0.9 if gate_passes else -1.1
    history = []
    for step in range(65):
        saved = step in contract.saved_optimizer_steps
        family = family0 if step == 0 else family64
        history.append(
            {
                "optimizer_update": step,
                **({} if step == 0 else {"true_optimizer_step": True}),
                "validation_answer_token_nll": 1.0 if saved else None,
                "validation_pair_metrics": {} if saved else None,
                "adapted_prefix_separation": {} if saved else None,
                "adapted_prefix_separation_ratios_from_update0": {} if saved else None,
                "validation_family_teacher_metrics": family if saved else None,
                **({} if step == 0 else {"separate_group_clipping": True}),
            }
        )
    cache = {"scene_count": 22, "source_prefix_sha256_by_scene": {}}
    condition = v32_rejection_status(config)
    metadata = {
        "config_hash": config_hash(config),
        "optimizer_step": 64,
        "best_epoch": 0,
        "best_monitor_loss": 1.0,
        "history": history,
        "v30_joint_pair": {
            "train_scene_ids": list(contract.v31.train_scene_ids),
            "validation_scene_ids": list(contract.v31.validation_scene_ids),
            "final_test_scene_ids_loaded": [],
            "oracle_environment_files_loaded": False,
            "scene_cache": cache,
        },
        "v33_environmental": {
            "conditional_v32_rejection": condition,
            "schedule": {"schedule_sha256": "a" * 64, "optimizer_step_count": 100},
            "optimizer_step": 64,
            "exact_trainable_parameter_count": 404_608,
            "train_scene_ids": list(contract.v31.train_scene_ids),
            "validation_scene_ids": list(contract.v31.validation_scene_ids),
            "deferred_final_scene_ids_loaded": [],
        },
    }
    (checkpoint / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (checkpoint / "runtime_metadata.json").write_text(
        json.dumps(runtime_checkpoint_metadata(metadata)), encoding="utf-8"
    )
    state = {
        parameter_id: {
            "step": torch.tensor(64.0),
            "exp_avg": torch.zeros(1),
            "exp_avg_sq": torch.zeros(1),
        }
        for parameter_id in range(8)
    }
    torch.save(
        {
            "state": state,
            "param_groups": [
                {
                    "name": "dense_sidecar_adapter.output",
                    "lr": 2.5e-5,
                    "weight_decay": 0.0,
                    "params": [0, 1],
                },
                {
                    "name": "dense_sidecar_adapter.sidecar_hidden",
                    "lr": 1e-4,
                    "weight_decay": 0.0,
                    "params": [2, 3, 4, 5],
                },
                {
                    "name": "dense_sidecar_adapter.position",
                    "lr": 1e-4,
                    "weight_decay": 0.0,
                    "params": [6, 7],
                },
            ],
        },
        checkpoint / "optimizer.pt",
    )
    return cache


def test_v33_resume_cannot_bypass_failed_update64_gate(tmp_path: Path) -> None:
    config = load_config(V33_CONFIG)
    contract = v33_contract(config)
    settings = v33_settings(config)
    cache = _write_resume_u64(tmp_path, config, gate_passes=False)
    resume = latest_v33_resume_checkpoint(tmp_path, contract)
    assert resume == tmp_path / "update_064"
    with pytest.raises(RuntimeError, match="cannot resume beyond its failed update-64"):
        validate_v33_resume_checkpoint(
            config=config,
            output=tmp_path,
            resume=resume,
            contract=contract,
            settings=settings,
            condition=v32_rejection_status(config),
            schedule_audit={"schedule_sha256": "a" * 64},
            cache_audit=cache,
        )


def test_v33_selector_audits_three_group_adam_and_complete_family_promotion(
    tmp_path: Path,
) -> None:
    config = load_config(V33_CONFIG)
    _write_resume_u64(tmp_path, config, gate_passes=True)
    _validate_optimizer_state_step(tmp_path / "update_064", 64, v33_settings(config))
    optimizer_path = tmp_path / "update_064" / "optimizer.pt"
    optimizer = torch.load(optimizer_path, map_location="cpu", weights_only=True)
    optimizer["param_groups"][1]["name"] = "decoder"
    torch.save(optimizer, optimizer_path)
    with pytest.raises(ValueError, match="optimizer group 1"):
        _validate_optimizer_state_step(tmp_path / "update_064", 64, v33_settings(config))

    candidate = {
        "greedy_exact_complete_units_correct": 6,
        "greedy_complete_units_by_family": {
            "book_support": 1,
            "mirror_lr": 4,
            "picture_support": 1,
        },
        "color_full_vocab_sides": 12,
        "mirror_full_vocab_sides": 10,
        "new_negative_sides": [],
    }
    assert all(
        v33_chat_promotion_checks(
            candidate, update0_aggregate=(216, 81), selected_aggregate=(216, 81)
        ).values()
    )
    candidate["greedy_complete_units_by_family"]["picture_support"] = 0
    checks = v33_chat_promotion_checks(
        candidate, update0_aggregate=(216, 81), selected_aggregate=(216, 82)
    )
    assert checks["changed_complete_pair_threshold_met"] is True
    assert checks["each_validation_family_demonstrated"] is False


def test_v33_docs_and_make_targets_have_no_final_bypass() -> None:
    makefile = Path("Makefile").read_text(encoding="utf-8")
    readme = Path("README.md").read_text(encoding="utf-8")
    for target in (
        "gemma4-v33-preflight-environmental",
        "gemma4-v33-train-environmental",
        "gemma4-v33-select-environmental",
    ):
        assert target in makefile
    assert "gemma4-v33-evaluate-final" not in makefile
    assert "V33 environmental-only" in readme
