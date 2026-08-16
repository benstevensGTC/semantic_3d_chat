from __future__ import annotations

import copy
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from semantic_3d_chat.config import load_config
from semantic_3d_chat.data.dataset import QARecord
from semantic_3d_chat.scene_encoder.block_cross_residual import (
    construct_block_cross_residual,
    validate_block_cross_residual_state,
)
from semantic_3d_chat.training.pair_curriculum import CounterfactualPairUnit
from semantic_3d_chat.training.train_block_cross_v35 import (
    V35SceneCache,
    _optimizer,
    assert_v35_trainable_surface,
    build_v35_schedule,
    construct_v35_core,
    freeze_for_v35,
    load_v35_train_qa_records,
    pair_and_cross_prefix_hinges,
    pinned_post_v33_prefix_manifest,
    preflight_v35,
    require_v34_terminal_gate,
    set_v35_optimizer_stage,
    v35_contract,
    v35_settings,
    v35_update32_gate,
    v35_update64_gate,
    v35_weighted_objective,
    validate_v35_cache_audit,
    validate_v35_scene_cache,
)

V35_CONFIG = Path("configs/experiments/gemma4_diverse28_block_cross_v35.yaml")


def _record(
    index: int,
    *,
    scene_id: str,
    answer_type: str = "spatial_relation",
    pair_id: str | None = None,
    question_key: str | None = None,
    role: str | None = None,
) -> QARecord:
    return QARecord(
        scene_id=scene_id,
        question_id=f"q_{index:04d}",
        question=f"opaque training question {index}",
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
        for index in range(140)
    ]
    repetitions = (1, 4, 4, 1, 4, 4, 4, 3)
    units: list[CounterfactualPairUnit] = []
    unit_index = 0
    for pair_index, count in enumerate(repetitions):
        pair_id = f"pair_{pair_index:06d}"
        left_scene = f"scene_{11 + 2 * pair_index:06d}"
        right_scene = f"scene_{12 + 2 * pair_index:06d}"
        for _ in range(count):
            key = f"unit_{unit_index:03d}"
            reference = _record(
                1_000 + 2 * unit_index,
                scene_id=left_scene,
                pair_id=pair_id,
                question_key=key,
                role="reference",
            )
            counterfactual = _record(
                1_001 + 2 * unit_index,
                scene_id=right_scene,
                pair_id=pair_id,
                question_key=key,
                role="counterfactual",
            )
            units.append(CounterfactualPairUnit(pair_id, key, reference, counterfactual))
            records.extend((reference, counterfactual))
            unit_index += 1
    assert unit_index == 25
    return records, units


def test_v35_contract_pins_exact_source_core_objective_and_gates() -> None:
    config = load_config(V35_CONFIG)
    settings = v35_settings(config)
    contract = v35_contract(config)
    terminal = require_v34_terminal_gate(config)
    assert settings.saved_optimizer_steps == (*range(0, 97, 8), 100)
    assert settings.broad_nll_weight == 0.25
    assert settings.pair_correct_nll_weight == 0.5
    assert settings.side_hinge_weight == 4.0
    assert settings.cross_prefix_flip_weight == 8.0
    assert settings.residual_penalty_weight == 0.001
    assert contract.source_tensor_state_sha256 == (
        "cb7bb3b48ace60212ee5c7f326839bf2ddd993810417de45c9a9cbc666313fe6"
    )
    assert contract.core_initial_state_sha256 == (
        "72ae7f492f5953e58d809b6782d559dc64669637d5d6a79ae98f3a31296a12cd"
    )
    assert terminal["sha256"] == (
        "b0833a72ba5bc507178fa07cacc8cbef798fce4de94a5f85f2e402aafb46679f"
    )


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("training.v35_block_cross", "optimizer_steps", 101),
        ("training.v35_block_cross", "cross_prefix_flip_weight", 4.0),
        ("training.v35_block_cross", "qkv_learning_rate", 2e-4),
        ("scene_encoder.block_cross_residual", "uniform_floor", 0.0),
        ("scene_encoder.block_cross_residual", "expected_initial_state_sha256", "0" * 64),
        ("v35_block_cross", "source_optimizer_step", 32),
        ("v35_block_cross", "source_v33_tensor_state_sha256", "0" * 64),
        ("v35_block_cross", "validation_qa_loaded_during_training", True),
        ("v35_block_cross", "exact_trainable_parameter_count", 199_808),
        ("v35_block_cross", "v34_terminal_gate_report_sha256", "0" * 64),
    ],
)
def test_v35_contract_fails_closed(section: str, field: str, value: object) -> None:
    config = copy.deepcopy(load_config(V35_CONFIG))
    target = config
    for key in section.split("."):
        target = target[key]
    target[field] = value
    with pytest.raises((TypeError, ValueError)):
        v35_contract(config)


def test_v35_preflight_is_model_map_validation_qa_and_final_free() -> None:
    report = preflight_v35(load_config(V35_CONFIG))
    assert report["passed"] is True
    assert report["gemma_loaded"] is False
    assert report["scene_maps_loaded"] is False
    assert report["validation_qa_loaded"] is False
    assert report["oracle_loaded"] is False
    assert report["final_test_scenes_touched"] is False
    assert report["exact_trainable_parameter_count"] == 983_040


def test_v35_train_loader_never_opens_validation_jsonl(monkeypatch: pytest.MonkeyPatch) -> None:
    config = load_config(V35_CONFIG)
    original = Path.read_text

    def guarded(path: Path, *args, **kwargs):
        if path.name == "validation.jsonl":
            raise AssertionError("V35 training attempted to open validation QA")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded)
    records, audit = load_v35_train_qa_records(config)
    assert len(records) == 384
    assert audit["validation_qa_loaded"] is False
    assert not any(Path(path).name == "validation.jsonl" for path in audit["loaded_files"])


def test_v35_schedule_has_four_exact_recurrences_and_one_unchanged_broad_per_step() -> None:
    records, units = _schedule_fixture()
    schedule, audit = build_v35_schedule(
        records,
        units,
        settings=v35_settings(load_config(V35_CONFIG)),
        seed=35035,
    )
    appearances = Counter(
        (row.pair_unit.pair_id, row.pair_unit.question_key) for row in schedule
    )
    assert len(schedule) == 100
    assert set(appearances.values()) == {4}
    assert len(appearances) == 25
    assert all(row.broad_record.counterfactual_expected_change is not True for row in schedule)
    assert audit["true_optimizer_step_per_schedule_row"] is True
    assert audit["exact_pair_unit_recurrence"] == 4


def test_v35_true_cross_prefix_scores_are_not_within_prefix_aliases() -> None:
    # c=[A@scene1, B@scene2], s=[B@scene1, A@scene2].
    correct = torch.tensor([1.0, 2.0], requires_grad=True)
    swapped = torch.tensor([3.0, 5.0], requires_grad=True)
    side_hinge, side_margins, cross_hinge, cross_margins = pair_and_cross_prefix_hinges(
        correct_rank_nll=correct,
        swapped_rank_nll=swapped,
        side_margin=0.5,
        cross_prefix_margin=0.25,
    )
    assert torch.equal(side_margins, torch.tensor([2.0, 3.0]))
    assert torch.equal(cross_margins, torch.tensor([4.0, 1.0]))
    assert side_hinge.item() == 0.0
    assert cross_hinge.item() == 0.0
    (side_hinge + cross_hinge).backward()
    assert correct.grad is not None and swapped.grad is not None


def test_v35_weighted_objective_is_exact_locked_formula() -> None:
    settings = v35_settings(load_config(V35_CONFIG))
    values = [torch.tensor(value) for value in (2.0, 3.0, 4.0, 5.0, 6.0)]
    observed = v35_weighted_objective(
        broad_nll=values[0],
        pair_correct_nll=values[1],
        side_hinge=values[2],
        cross_prefix_flip_hinge=values[3],
        normalized_residual_penalty=values[4],
        settings=settings,
    )
    expected = 0.25 * 2 + 0.5 * 3 + 4 * 4 + 8 * 5 + 0.001 * 6
    assert observed.item() == pytest.approx(expected)


def _scene_cache() -> V35SceneCache:
    positions = torch.tensor([[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]], dtype=torch.float16)
    return V35SceneCache(
        scene_id="scene_opaque",
        source_scene_tokens=torch.zeros(1, 256, 1536, dtype=torch.float32),
        block_tokens=torch.ones(2, 384, dtype=torch.float16),
        block_positions_normalized=positions,
        source_prefix_sha256="a" * 64,
        voxel_count=7,
        processed_voxels=7,
        occupied_block_count=1,
        tokens_per_block=2,
    )


def test_v35_cache_contract_rejects_omitted_voxels_tokens_and_positions() -> None:
    valid = _scene_cache()
    validate_v35_scene_cache(valid)
    with pytest.raises(ValueError, match="omitted"):
        validate_v35_scene_cache(
            V35SceneCache(**{**valid.__dict__, "processed_voxels": 6})
        )
    with pytest.raises(ValueError, match="omitted an occupied-block token"):
        validate_v35_scene_cache(
            V35SceneCache(**{**valid.__dict__, "occupied_block_count": 2})
        )
    displaced = valid.block_positions_normalized.clone()
    displaced[1, 0] = 0.9
    with pytest.raises(ValueError, match="repeated normalized"):
        validate_v35_scene_cache(
            V35SceneCache(**{**valid.__dict__, "block_positions_normalized": displaced})
        )


def test_v35_cache_audit_is_question_oracle_retrieval_and_validation_qa_free() -> None:
    scene_ids = [f"scene_{index:06d}" for index in range(11, 33)]
    coverage = {
        scene_id: {
            "voxel_count": 10,
            "processed_voxels": 10,
            "occupied_block_count": 3,
            "tokens_per_block": 2,
            "token_count": 6,
        }
        for scene_id in scene_ids
    }
    audit = {
        "cache_boundary": "exact_post_v33_scene_tokens_plus_all_frozen_block_tokens",
        "scene_count": 22,
        "scene_ids": scene_ids,
        "source_scene_tokens_dtype": "torch.float32_cpu",
        "block_tokens_dtype": "torch.float16_cpu",
        "block_positions_dtype": "torch.float16_cpu",
        "all_voxels_covered": True,
        "all_occupied_blocks_processed": True,
        "all_block_tokens_cached": True,
        "all_repeated_normalized_block_positions_cached": True,
        "source_prefixes_match_exact_v33_update64": True,
        "source_prefixes_match_terminal_pinned_post_v33_manifest": True,
        "inherited_prefixes_treated_as_pre_v33_provenance_only": True,
        "question_inputs_to_scene_cache": False,
        "answer_inputs_to_scene_cache": False,
        "question_dependent_scene_processing": False,
        "question_dependent_retrieval": False,
        "oracle_environment_files_loaded": False,
        "validation_qa_loaded": False,
        "coverage_by_scene": coverage,
        "source_prefix_sha256_by_scene": {scene_id: "a" * 64 for scene_id in scene_ids},
        "inherited_pre_v33_prefix_sha256_by_scene": {
            scene_id: "b" * 64 for scene_id in scene_ids
        },
        "post_v33_prefix_manifest_attestation": {
            "attesting_metadata_path": "/tmp/v34/update_032/metadata.json",
            "attesting_metadata_sha256": "c" * 64,
            "attesting_optimizer_step": 0,
            "carrier_checkpoint_optimizer_step": 32,
        },
        "loaded_environment_files": [f"/tmp/maps/{scene_id}/voxel_map.npz" for scene_id in scene_ids],
    }
    validate_v35_cache_audit(audit, expected_scene_ids=scene_ids)
    contaminated = copy.deepcopy(audit)
    contaminated["question_dependent_retrieval"] = True
    with pytest.raises(ValueError, match="failed closed"):
        validate_v35_cache_audit(contaminated, expected_scene_ids=scene_ids)


def test_v35_post_v33_prefixes_come_from_terminal_pinned_v34_update_zero() -> None:
    config = load_config(V35_CONFIG)
    contract = v35_contract(config)
    terminal = require_v34_terminal_gate(config)
    source_metadata = json.loads(
        (contract.source_checkpoint / "metadata.json").read_text(encoding="utf-8")
    )
    manifest = pinned_post_v33_prefix_manifest(
        source_metadata=source_metadata,
        terminal=terminal,
        expected_scene_ids=(
            *contract.v31.train_scene_ids,
            *contract.v31.validation_scene_ids,
        ),
    )
    post = manifest["post_v33_prefix_sha256_by_scene"]
    inherited = manifest["inherited_pre_v33_prefix_sha256_by_scene"]
    assert post["scene_000011"] == (
        "05708c319a90429ee5730edcd3769b9673f63f64ea8b5eb6caf642579888e89f"
    )
    assert inherited["scene_000011"] == (
        "5a423a230a578835882911baab662062dbc8f78a31578112dcec4e942a3756fa"
    )
    assert all(post[scene_id] != inherited[scene_id] for scene_id in post)
    assert manifest["attesting_metadata_sha256"] == (
        "14ba328ab9ac1010b75e40123643e3497c59b2bc1c59bfbe307d05a58cea7719"
    )


def test_v35_post_v33_prefix_manifest_rejects_metadata_tamper(
    tmp_path: Path,
) -> None:
    config = load_config(V35_CONFIG)
    contract = v35_contract(config)
    terminal = require_v34_terminal_gate(config)
    source_metadata = json.loads(
        (contract.source_checkpoint / "metadata.json").read_text(encoding="utf-8")
    )
    original_path = Path(terminal["report"]["update32_checkpoint"])
    if not original_path.is_absolute():
        original_path = V35_CONFIG.resolve().parents[2] / original_path
    tampered = json.loads((original_path / "metadata.json").read_text(encoding="utf-8"))
    tampered["v30_joint_pair"]["update_zero_equivalence"][
        "source_prefix_sha256_by_scene"
    ]["scene_000011"] = "d" * 64
    tampered_root = tmp_path / "update_032"
    tampered_root.mkdir()
    (tampered_root / "metadata.json").write_text(
        json.dumps(tampered, sort_keys=True), encoding="utf-8"
    )
    forged_terminal = copy.deepcopy(terminal)
    forged_terminal["report"]["update32_checkpoint"] = str(tampered_root)
    with pytest.raises(ValueError, match="differs from its terminal pin"):
        pinned_post_v33_prefix_manifest(
            source_metadata=source_metadata,
            terminal=forged_terminal,
            expected_scene_ids=(
                *contract.v31.train_scene_ids,
                *contract.v31.validation_scene_ids,
            ),
        )


def _fake_bundle(core: torch.nn.Module) -> SimpleNamespace:
    inherited = torch.nn.Linear(2, 2)
    model = torch.nn.Linear(3, 3)
    return SimpleNamespace(
        language=SimpleNamespace(model=model),
        checkpoint_modules={"inherited": inherited, "block_cross_residual": core},
    )


def test_v35_surface_and_fresh_adam_stage_output_then_qkv() -> None:
    config = load_config(V35_CONFIG)
    core = construct_v35_core(config, device=torch.device("cpu"))
    bundle = _fake_bundle(core)
    trainable = freeze_for_v35(bundle, core, optimizer_step=0)
    surface0 = assert_v35_trainable_surface(bundle, core, optimizer_step=0)
    optimizer = _optimizer(bundle, core, v35_settings(config))
    assert optimizer.state == {}
    assert sum(parameter.numel() for parameter in trainable) == 983_040
    assert surface0["active_parameter_names"] == ["w_o"]
    assert [group["name"] for group in optimizer.param_groups] == [
        "block_cross_residual.qkv",
        "block_cross_residual.output",
    ]
    assert optimizer.param_groups[0]["lr"] == 0.0
    set_v35_optimizer_stage(
        bundle=bundle,
        block_cross_residual=core,
        optimizer=optimizer,
        optimizer_step_to_run=1,
        settings=v35_settings(config),
    )
    assert [name for name, value in core.named_parameters() if value.requires_grad] == ["w_o"]
    set_v35_optimizer_stage(
        bundle=bundle,
        block_cross_residual=core,
        optimizer=optimizer,
        optimizer_step_to_run=2,
        settings=v35_settings(config),
    )
    assert all(value.requires_grad for value in core.parameters())
    assert optimizer.param_groups[0]["lr"] == 1e-4
    assert not any(bundle.language.model.parameters().__next__().requires_grad for _ in [0])


def test_v35_core_hash_parameter_count_and_zero_identity_are_exact() -> None:
    config = load_config(V35_CONFIG)
    core = construct_block_cross_residual(
        config, scene_dim=1536, block_dim=384, latent_count=256
    )
    assert core is not None
    audit = validate_block_cross_residual_state(core)
    assert audit["parameter_count"] == 983_040
    assert audit["state_sha256"] == (
        "72ae7f492f5953e58d809b6782d559dc64669637d5d6a79ae98f3a31296a12cd"
    )
    base = torch.randn(1, 256, 1536)
    blocks = torch.randn(6, 384)
    positions = torch.rand(6, 3).mul(2).sub(1)
    assert torch.equal(core(base, blocks, positions), base)


def _gate_inputs() -> tuple[dict[str, float | int], dict[str, object], dict[str, object]]:
    separation: dict[str, float | int] = {
        "changed_selectivity_ratio_geometric_mean": 1.03,
        "changed_selectivity_over_1_02_count": 6,
        "changed_selectivity_ratio_minimum": 0.99,
        "unrelated_ratio_median": 1.0,
        "unrelated_abs_log_ratio_p90": 0.01,
    }
    baseline: dict[str, object] = {
        "mean_margin": -0.5,
        "complete_units": 1,
        "complete_units_by_family": {name: 0 for name in ("book_support", "mirror_lr", "picture_support")},
    }
    current: dict[str, object] = {
        "mean_margin": 0.1,
        "complete_units": 8,
        "complete_units_by_family": {name: 1 for name in ("book_support", "mirror_lr", "picture_support")},
    }
    return separation, current, baseline


def test_v35_train_only_gates_require_causal_improvement_families_and_residual_bound() -> None:
    contract = v35_contract(load_config(V35_CONFIG))
    separation, current, baseline = _gate_inputs()
    gate32 = v35_update32_gate(
        separation=separation,
        pair_metrics=current,
        baseline_pair_metrics=baseline,
        residual_rms=0.05,
        contract=contract,
    )
    assert gate32["passed"] is True
    assert gate32["training_scenes_only"] is True
    gate64 = v35_update64_gate(
        update32_gate=gate32,
        pair_metrics=current,
        baseline_pair_metrics=baseline,
        residual_rms=0.05,
        contract=contract,
    )
    assert gate64["passed"] is True
    failed = copy.deepcopy(current)
    failed["complete_units_by_family"]["picture_support"] = 0
    assert v35_update64_gate(
        update32_gate=gate32,
        pair_metrics=failed,
        baseline_pair_metrics=baseline,
        residual_rms=0.05,
        contract=contract,
    )["passed"] is False
    assert v35_update32_gate(
        separation=separation,
        pair_metrics=current,
        baseline_pair_metrics=baseline,
        residual_rms=0.100001,
        contract=contract,
    )["passed"] is False


def test_v35_docs_and_make_have_sealed_preflight_train_without_final_bypass() -> None:
    makefile = Path("Makefile").read_text(encoding="utf-8")
    readme = Path("README.md").read_text(encoding="utf-8")
    for target in (
        "gemma4-v34-seal-update32",
        "gemma4-v35-preflight-block-cross",
        "gemma4-v35-train-block-cross",
    ):
        assert target in makefile
    assert "gemma4-v35-evaluate-final" not in makefile
    assert "V35 bounded all-block cross-residual" in readme
