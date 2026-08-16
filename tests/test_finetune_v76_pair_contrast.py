from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest
import torch
from safetensors import safe_open

import semantic_3d_chat.training.finetune_v76_pair_contrast as v76
from semantic_3d_chat.scene_encoder.question_control_v75 import (
    DenseFullSceneContinuousControlV75,
)
from semantic_3d_chat.training.train_question_control_v73 import (
    HELD_PAIR_IDS,
    TRAIN_PAIR_IDS,
    load_config_v73,
    load_training_rows_v73,
    split_rows_v73,
)


def _split_rows():
    config = load_config_v73("configs/experiments/gemma4_v73_fullscene_controller.yaml")
    return split_rows_v73(load_training_rows_v73(config["training_qa"]))


def _tiny_v75(*, full_contract: bool = False) -> DenseFullSceneContinuousControlV75:
    hidden_size = 1536 if full_contract else 16
    environment_latents = 256 if full_contract else 4
    output_basis = torch.eye(hidden_size, dtype=torch.float32)[:2]
    return DenseFullSceneContinuousControlV75(
        hidden_size,
        output_basis,
        environment_latents=environment_latents,
        query_count=1,
        model_dimension=2,
        coefficient_decoder_hidden_dimension=3,
    )


def test_default_selection_is_all_40_historical_units_and_deterministic() -> None:
    train, held = _split_rows()
    first = v76.select_historical_changed_units_v76(train)
    second = v76.select_historical_changed_units_v76(train)
    assert [(unit.pair_id, unit.question_key) for unit in first] == [
        (unit.pair_id, unit.question_key) for unit in second
    ]
    assert len(first) == v76.EXPECTED_CHANGED_UNIT_COUNT == 40
    assert len({row.key for unit in first for row in (unit.left, unit.right)}) == 80
    assert {unit.pair_id for unit in first} <= set(TRAIN_PAIR_IDS)
    assert not ({unit.pair_id for unit in first} & set(HELD_PAIR_IDS))
    assert Counter(unit.change_type for unit in first) == Counter(v76.EXPECTED_CHANGE_TYPE_COUNTS)
    assert all(unit.left.question == unit.right.question for unit in first)
    assert all(unit.left.answer != unit.right.answer for unit in first)
    with pytest.raises(ValueError, match="escaped historical training pairs"):
        v76.select_historical_changed_units_v76(held)


def test_bounded_selection_and_atomic_schedule_are_deterministic() -> None:
    train, _held = _split_rows()
    units = v76.select_historical_changed_units_v76(train, max_units=7)
    repeated = v76.select_historical_changed_units_v76(train, max_units=7)
    assert [(unit.pair_id, unit.question_key) for unit in units] == [
        (unit.pair_id, unit.question_key) for unit in repeated
    ]
    first = v76.deterministic_pair_schedule_v76(units, cycles=3, seed=760176)
    second = v76.deterministic_pair_schedule_v76(units, cycles=3, seed=760176)
    assert [item.unit.question_key for item in first] == [item.unit.question_key for item in second]
    assert len(first) == 21
    assert set(Counter(item.unit.question_key for item in first).values()) == {3}
    expected = {(unit.pair_id, unit.question_key) for unit in units}
    for cycle in (1, 2, 3):
        observed = {
            (item.unit.pair_id, item.unit.question_key) for item in first if item.cycle == cycle
        }
        assert observed == expected
        assert [item.step_in_cycle for item in first if item.cycle == cycle] == list(range(1, 8))


def test_pair_contrast_formula_and_gradients_use_both_sides() -> None:
    correct = torch.tensor([2.0, 1.0], requires_grad=True)
    alternative = torch.tensor([1.0, 3.0], requires_grad=True)
    anchor = torch.tensor(0.25, requires_grad=True)
    settings = v76.V76LossSettings(
        answer_nll_weight=1.5,
        pair_contrast_weight=2.0,
        pair_contrast_margin=0.5,
        source_anchor_weight=0.1,
    )
    loss, metrics = v76.paired_answer_contrast_objective_v76(correct, alternative, anchor, settings)
    # Correct mean = 1.5, margins = [-1, 2], hinge mean = 0.75.
    assert float(loss.detach()) == pytest.approx(3.775)
    assert float(metrics["correct_answer_nll"].detach()) == pytest.approx(1.5)
    assert float(metrics["alternative_answer_nll"].detach()) == pytest.approx(2.0)
    assert float(metrics["paired_alternative_margin"].detach()) == pytest.approx(0.5)
    assert int(metrics["positive_preference_sides"].detach()) == 1
    assert int(metrics["margin_satisfied_sides"].detach()) == 1
    loss.backward()
    assert correct.grad is not None
    assert alternative.grad is not None
    assert anchor.grad is not None
    assert correct.grad.tolist() == pytest.approx([1.75, 0.75])
    assert alternative.grad.tolist() == pytest.approx([-1.0, 0.0])
    assert float(anchor.grad) == pytest.approx(0.1)


def test_source_anchor_is_zero_at_initialization_and_backpropagates() -> None:
    model = _tiny_v75()
    source = v76.snapshot_source_parameters_v76(model)
    initial = v76.source_weight_anchor_l2_v76(model, source)
    assert torch.equal(initial, torch.zeros_like(initial))
    with torch.no_grad():
        model.key.weight.add_(1.0)
    moved = v76.source_weight_anchor_l2_v76(model, source)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    expected = model.key.weight.numel() / total_parameters
    assert float(moved.detach()) == pytest.approx(expected)
    moved.backward()
    assert model.key.weight.grad is not None
    assert bool(torch.isfinite(model.key.weight.grad).all())
    assert float(model.key.weight.grad.abs().sum()) > 0.0


def test_argument_and_schedule_guards_fail_closed() -> None:
    parser = v76.build_parser()
    args = parser.parse_args([])
    settings = v76.validate_args_v76(args)
    assert args.initial_candidate == v76.V76_INITIAL_CANDIDATE
    assert args.max_units == 40
    assert args.cycles == 2
    assert settings.source_anchor_weight == pytest.approx(0.01)
    args.max_units = 41
    with pytest.raises(ValueError, match="max_units"):
        v76.validate_args_v76(args)
    with pytest.raises(ValueError, match="cycles"):
        v76.deterministic_pair_schedule_v76((), cycles=0, seed=0)
    with pytest.raises(ValueError, match="nonnegative"):
        v76.V76LossSettings(source_anchor_weight=-0.1)
    with pytest.raises(ValueError, match="forbidden split tokens"):
        v76.assert_exact_v75_source_v76("data/oracle/scene.json")


def test_exact_local_v75_candidate_is_authenticated_when_available() -> None:
    source = Path(v76.PROJECT_ROOT) / v76.V76_INITIAL_CANDIDATE
    if not source.is_file():
        pytest.skip("local V75 diagnostic is not installed")
    authenticated, metadata = v76.assert_exact_v75_source_v76(source)
    assert authenticated == source.resolve()
    assert metadata["artifact"] == "v75_verified_teacher_dense_reader_candidate_v1"


def test_input_and_output_guards_reject_symlink_and_project_escape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(v76, "PROJECT_ROOT", tmp_path)
    target = tmp_path / "source.bin"
    target.write_bytes(b"safe")
    link = tmp_path / "source-link.bin"
    link.symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        v76._guard_input_v76(link, "fixture")

    target_directory = tmp_path / "target-directory"
    target_directory.mkdir()
    linked_directory = tmp_path / "linked-directory"
    linked_directory.symlink_to(target_directory, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        v76._guard_output_v76(linked_directory / "candidate.json", suffix=".json")
    with pytest.raises(ValueError, match="project root"):
        v76._guard_output_v76(tmp_path.parent / "escape.json", suffix=".json")


def test_diagnostic_checkpoint_is_minimal_quarantined_and_exact_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(v76, "PROJECT_ROOT", tmp_path)
    model = _tiny_v75(full_contract=True)
    destination = tmp_path / "reports" / "artifacts" / "candidate.safetensors"
    saved = v76.save_v76_diagnostic(
        destination,
        model,
        optimizer_steps=80,
        selected_unit_count=40,
        cycles=2,
    )
    assert saved["exact_zero_audit"]["exact_zero_scene_verified"] is True
    with safe_open(str(destination), framework="pt", device="cpu") as handle:
        assert frozenset(handle.keys()) == v76.V75_STATE_FIELDS
        metadata = dict(handle.metadata() or {})
    assert metadata["artifact"] == "v76_all_historical_pair_contrast_diagnostic_v1"
    assert metadata["runtime_promotion_forbidden_until_gemma_gate"] == "true"
    assert metadata["runtime_publication_artifact"] == "false"
    assert metadata["answer_codebook_serialized"] == "false"
    assert metadata["environmental_text_inputs"] == "0"
    assert metadata["all_256_environment_latents_attended"] == "true"
    assert metadata["exhaustive_historical_changed_units"] == "true"
    assert metadata["official_validation_loaded"] == "false"
    assert metadata["official_test_loaded"] == "false"
    assert metadata["oracle_loaded"] == "false"
    with pytest.raises(FileExistsError):
        v76.save_v76_diagnostic(
            destination,
            model,
            optimizer_steps=80,
            selected_unit_count=40,
            cycles=2,
        )
