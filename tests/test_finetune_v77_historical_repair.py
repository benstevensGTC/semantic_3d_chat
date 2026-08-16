from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import pytest
import torch
from safetensors import safe_open

import semantic_3d_chat.training.finetune_v77_historical_repair as v77
from semantic_3d_chat.scene_encoder.question_control_v75 import (
    DenseFullSceneContinuousControlV75,
)
from semantic_3d_chat.training.train_question_control_v73 import (
    EXPECTED_TRAIN_ROWS,
    HELD_PAIR_IDS,
    TRAIN_PAIR_IDS,
    load_config_v73,
    load_training_rows_v73,
    split_rows_v73,
)


def _split_rows():
    config = load_config_v73("configs/experiments/gemma4_v73_fullscene_controller.yaml")
    return split_rows_v73(load_training_rows_v73(config["training_qa"]))


def _full_v75() -> DenseFullSceneContinuousControlV75:
    basis = torch.eye(1536, dtype=torch.float32)[:112]
    return DenseFullSceneContinuousControlV75(
        1536,
        basis,
        environment_latents=256,
        query_count=4,
        model_dimension=128,
        coefficient_decoder_hidden_dimension=768,
    )


def test_default_selection_is_all_576_historical_rows_and_deterministic() -> None:
    train, held = _split_rows()
    first = v77.select_balanced_historical_rows_v77(
        train, max_rows=EXPECTED_TRAIN_ROWS, seed=770177
    )
    second = v77.select_balanced_historical_rows_v77(
        train, max_rows=EXPECTED_TRAIN_ROWS, seed=770177
    )
    assert [row.key for row in first] == [row.key for row in second]
    assert len(first) == EXPECTED_TRAIN_ROWS == 576
    assert len({row.key for row in first}) == 576
    assert {row.pair_id for row in first} <= set(TRAIN_PAIR_IDS)
    assert not ({row.pair_id for row in first} & set(HELD_PAIR_IDS))
    assert not ({row.scene_id for row in first} & {row.scene_id for row in held})
    assert len({row.answer_class for row in first}) == v77.EXPECTED_ANSWER_CLASS_COUNT == 28
    assert (
        len({v77.question_template_id_v77(row.question) for row in first})
        == v77.EXPECTED_QUESTION_TEMPLATE_COUNT
        == 96
    )


def test_bounded_selection_round_robins_classes_and_templates() -> None:
    train, _held = _split_rows()
    selected = v77.select_balanced_historical_rows_v77(train, max_rows=64, seed=770177)
    # The first round visits every answer class exactly once, even though the
    # original frequencies range from one to 94 rows.
    assert len({row.answer_class for row in selected[:28]}) == 28

    templates_by_class: dict[str, list[str]] = defaultdict(list)
    for row in selected:
        templates_by_class[row.answer_class].append(v77.question_template_id_v77(row.question))
    all_template_counts: dict[str, int] = defaultdict(int)
    for row in train:
        if row.answer_class == "answer_6b86b273ff34fce19d6b":
            all_template_counts[v77.question_template_id_v77(row.question)] += 1
    observed = templates_by_class["answer_6b86b273ff34fce19d6b"]
    distinct_available = len(all_template_counts)
    assert len(set(observed[: min(len(observed), distinct_available)])) == min(
        len(observed), distinct_available
    )


def test_canonical_negative_sampling_is_same_type_deterministic_and_never_correct() -> None:
    train, _held = _split_rows()
    alternatives = v77.canonical_alternatives_v77(train)
    assert alternatives
    for row in train:
        first = v77.sample_negative_answer_v77(row, alternatives, seed=7, cycle=1)
        second = v77.sample_negative_answer_v77(row, alternatives, seed=7, cycle=1)
        assert first == second
        assert first != row.answer
        assert first in alternatives[row.answer_type]


def test_schedule_reuses_the_same_selected_membership_once_per_cycle() -> None:
    train, _held = _split_rows()
    alternatives = v77.canonical_alternatives_v77(train)
    selected = v77.select_balanced_historical_rows_v77(train, max_rows=48, seed=33)
    first = v77.deterministic_training_schedule_v77(selected, alternatives, cycles=3, seed=33)
    repeated = v77.deterministic_training_schedule_v77(selected, alternatives, cycles=3, seed=33)
    assert [item.row.key for item in first] == [item.row.key for item in repeated]
    assert [item.negative_answer for item in first] == [item.negative_answer for item in repeated]
    assert len(first) == 144
    expected = {row.key for row in selected}
    for cycle in (1, 2, 3):
        members = [item for item in first if item.cycle == cycle]
        assert {item.row.key for item in members} == expected
        assert [item.step_in_cycle for item in members] == list(range(1, 49))
        assert all(item.template_id.startswith("template_") for item in members)


def test_changed_opposites_cover_exact_historical_changed_sides() -> None:
    train, _held = _split_rows()
    opposites = v77.changed_opposites_v77(train)
    assert len(opposites) == v77.EXPECTED_CHANGED_SIDE_COUNT == 80
    for key, opposite in opposites.items():
        reverse = opposites[opposite.key]
        assert reverse.key == key
        assert reverse.question == opposite.question
        assert reverse.answer != opposite.answer


def test_row_objective_formula_and_gradients() -> None:
    correct = torch.tensor(2.0, requires_grad=True)
    negative = torch.tensor(1.0, requires_grad=True)
    paired = torch.tensor(3.0, requires_grad=True)
    output_anchor = torch.tensor(0.25, requires_grad=True)
    settings = v77.V77LossSettings(
        answer_nll_weight=1.5,
        negative_margin_weight=2.0,
        negative_margin=0.5,
        changed_pair_margin_weight=3.0,
        changed_pair_margin=1.5,
        source_output_anchor_weight=0.1,
        source_weight_anchor_weight=0.0,
    )
    loss, metrics = v77.row_objective_v77(
        correct_answer_nll=correct,
        negative_answer_nll=negative,
        changed_pair_answer_nll=paired,
        source_output_mse=output_anchor,
        settings=settings,
    )
    # 1.5*2 + 2*(0.5+2-1) + 3*(1.5+2-3) + 0.1*0.25
    assert float(loss.detach()) == pytest.approx(7.525)
    assert float(metrics["negative_answer_margin"].detach()) == pytest.approx(-1.0)
    assert float(metrics["negative_margin_hinge"].detach()) == pytest.approx(1.5)
    assert float(metrics["changed_pair_margin_hinge"].detach()) == pytest.approx(0.5)
    loss.backward()
    assert float(correct.grad) == pytest.approx(6.5)
    assert float(negative.grad) == pytest.approx(-2.0)
    assert float(paired.grad) == pytest.approx(-3.0)
    assert float(output_anchor.grad) == pytest.approx(0.1)


def test_default_args_are_all_rows_but_offer_a_bounded_fast_screen() -> None:
    parser = v77.build_parser()
    args = parser.parse_args([])
    settings = v77.validate_args_v77(args)
    assert args.initial_candidate == v77.V77_INITIAL_CANDIDATE
    assert args.max_rows == 576
    assert args.cycles == 1
    assert args.gradient_accumulation_rows == 8
    assert args.measurement_rows == 48
    assert settings.negative_margin_weight == pytest.approx(0.25)
    fast = parser.parse_args(["--max-rows", "48", "--measurement-rows", "16"])
    v77.validate_args_v77(fast)
    args.measurement_rows = 577
    with pytest.raises(ValueError, match="measurement_rows"):
        v77.validate_args_v77(args)
    with pytest.raises(ValueError, match="positive"):
        v77.V77LossSettings(answer_nll_weight=0.0)


def test_exact_local_d012_candidate_authenticates_without_loading_gemma() -> None:
    source = v77.PROJECT_ROOT / v77.V77_INITIAL_CANDIDATE
    if not source.is_file():
        pytest.skip("exact local V75-NLL diagnostic is not installed")
    authenticated, metadata = v77.assert_exact_v75_nll_source_v77(source)
    assert authenticated == source
    assert metadata == v77._SOURCE_METADATA
    model, _ = v77.load_exact_v75_nll_source_v77(source, torch.device("cpu"))
    assert type(model) is DenseFullSceneContinuousControlV75
    assert model.environment_latents == 256
    assert model.hidden_size == 1536


def test_guards_reject_forbidden_paths_symlinks_and_project_escape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(v77, "PROJECT_ROOT", tmp_path)
    source = tmp_path / "source.bin"
    source.write_bytes(b"safe")
    assert v77._guard_input_v77(source, "fixture") == source
    forbidden = tmp_path / "oracle" / "source.bin"
    forbidden.parent.mkdir()
    forbidden.write_bytes(b"unsafe")
    with pytest.raises(ValueError, match="forbidden"):
        v77._guard_input_v77(forbidden, "fixture")
    link = tmp_path / "source_link.bin"
    link.symlink_to(source)
    with pytest.raises(ValueError, match="symlink"):
        v77._guard_input_v77(link, "fixture")
    with pytest.raises(ValueError, match="project root"):
        v77._guard_output_v77(tmp_path.parent / "escape.json", suffix=".json")
    with pytest.raises(ValueError, match="forbidden"):
        v77._guard_output_v77(tmp_path / "reports" / "runtime" / "candidate.json", suffix=".json")


def test_diagnostic_is_weights_only_quarantined_and_contains_no_answer_codebook(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(v77, "PROJECT_ROOT", tmp_path)
    model = _full_v75()
    destination = tmp_path / "reports" / "artifacts" / "candidate.safetensors"
    saved = v77.save_v77_diagnostic(
        destination,
        model,
        selected_row_count=576,
        cycles=1,
        optimizer_steps=72,
    )
    assert saved["exact_zero_audit"]["exact_zero_scene_verified"] is True
    with safe_open(str(destination), framework="pt", device="cpu") as handle:
        assert frozenset(handle.keys()) == v77.V75_STATE_FIELDS
        metadata = dict(handle.metadata() or {})
    assert metadata["artifact"] == "v77_all_historical_answer_repair_diagnostic_v1"
    assert metadata["exhaustive_576_row_selection"] == "true"
    assert metadata["answer_codebook_serialized"] == "false"
    assert metadata["negative_answer_codebook_serialized"] == "false"
    assert metadata["questions_or_answers_serialized"] == "false"
    assert metadata["runtime_publication_artifact"] == "false"
    assert metadata["official_validation_loaded"] == "false"
    assert metadata["official_test_loaded"] == "false"
    assert metadata["deferred_final_loaded"] == "false"
    assert metadata["oracle_loaded"] == "false"
    serialized_tokens = set(" ".join(metadata.values()).casefold().split())
    for canonical_answer in ("yes", "no", "left", "right", "yellow", "upright"):
        assert canonical_answer not in serialized_tokens
    with pytest.raises(FileExistsError):
        v77.save_v77_diagnostic(
            destination,
            model,
            selected_row_count=576,
            cycles=1,
            optimizer_steps=72,
        )


def test_selection_rejects_internal_held_rows() -> None:
    _train, held = _split_rows()
    with pytest.raises(ValueError, match="576 historical"):
        v77.select_balanced_historical_rows_v77(held, max_rows=1, seed=0)


def test_template_ids_are_opaque_stable_and_text_sensitive() -> None:
    first = v77.question_template_id_v77("Is there a chair?")
    assert first == v77.question_template_id_v77("  IS THERE A CHAIR?  ")
    assert first != v77.question_template_id_v77("Is there a table?")
    assert first.startswith("template_")
    assert "chair" not in first


def test_selected_subset_retains_broad_answer_type_coverage() -> None:
    train, _held = _split_rows()
    selected = v77.select_balanced_historical_rows_v77(train, max_rows=48, seed=770177)
    counts = Counter(row.answer_type for row in selected)
    assert set(counts) == {row.answer_type for row in train}
