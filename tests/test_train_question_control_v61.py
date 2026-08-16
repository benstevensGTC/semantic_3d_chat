from __future__ import annotations

import argparse

import pytest

from semantic_3d_chat.evaluation.v59_multiscene_train_gate import LOCKED_SCENE_IDS
from semantic_3d_chat.training.train_question_control_v61 import _parser, _validate_args


def _arguments(scene_ids: tuple[str, ...] = LOCKED_SCENE_IDS) -> list[str]:
    return [
        "--base-runtime-config",
        "configs/runtime/gemma4_v56_question_control.yaml",
        "--base-checkpoint",
        "data_gemma4/checkpoints/gemma4_v54_semantic_greedy_gate/update_000",
        "--source-v60-checkpoint",
        "data_gemma4/checkpoints/gemma4_v60_teacher_basis_control",
        "--source-v60-report",
        "reports/gemma4/metrics/v60_teacher_basis_control_training.json",
        "--train-qa",
        "data_diverse52/qa/train.jsonl",
        *(value for scene_id in scene_ids for value in ("--scene-id", scene_id)),
        "--prefix-cache",
        "data_gemma4/scene_tokens/v59_locked_six_prefixes",
        "--output-checkpoint",
        "data_gemma4/checkpoints/v61_candidate",
        "--training-report",
        "reports/gemma4/metrics/v61_training.json",
    ]


def test_v61_parser_exposes_only_locked_training_inputs() -> None:
    parser = _parser()
    destinations = {
        action.dest for action in parser._actions if action.dest is not argparse.SUPPRESS
    }
    assert {
        "preregistration",
        "generalization_gate",
        "generalization_questions",
        "baseline",
        "baseline_lock",
        "baseline_predictions",
    }.isdisjoint(destinations)

    args = parser.parse_args(_arguments())
    _validate_args(args)
    assert tuple(sorted(args.scene_id)) == LOCKED_SCENE_IDS


@pytest.mark.parametrize(
    "scene_ids",
    [
        LOCKED_SCENE_IDS[:-1],
        (*LOCKED_SCENE_IDS[:-1], "scene_000039"),
        (*LOCKED_SCENE_IDS, LOCKED_SCENE_IDS[0]),
    ],
)
def test_v61_argument_validation_rejects_any_scene_inventory_drift(
    scene_ids: tuple[str, ...],
) -> None:
    args = _parser().parse_args(_arguments(scene_ids))
    with pytest.raises(ValueError, match="exact locked six training scenes"):
        _validate_args(args)


@pytest.mark.parametrize(
    "prohibited_option",
    ["--preregistration", "--generalization-gate", "--baseline-lock"],
)
def test_v61_parser_rejects_deferred_gate_or_baseline_inputs(
    prohibited_option: str,
) -> None:
    with pytest.raises(SystemExit):
        _parser().parse_args([*_arguments(), prohibited_option, "must-not-be-read.json"])
