from __future__ import annotations

from semantic_3d_chat.training.train_question_control_v58 import (
    _parser,
    _validate_args,
)


def test_v58_cli_defaults_match_proven_free_prompt_recipe() -> None:
    args = _parser().parse_args(
        [
            "--base-runtime-config",
            "configs/runtime/gemma4_v56_question_control.yaml",
            "--base-checkpoint",
            "data_gemma4/checkpoints/gemma4_v54_semantic_greedy_gate/update_000",
            "--source-control-checkpoint",
            "data_gemma4/checkpoints/gemma4_v57_question_control_pair31_32_conditioning_u40",
            "--train-qa",
            "data_diverse52/qa/train.jsonl",
            "--scene-id",
            "scene_000031",
            "--scene-id",
            "scene_000032",
            "--prefix-cache",
            "data_gemma4/scene_tokens/v56_question_control_smoke_prefixes",
            "--teacher-artifact",
            "data_gemma4/training/v58_teachers_pair31_32",
            "--output-checkpoint",
            "data_gemma4/checkpoints/v58_runtime",
            "--training-report",
            "reports/gemma4/metrics/v58_training.json",
        ]
    )
    _validate_args(args)
    assert args.teacher_learning_rate == 0.03
    assert args.teacher_gradient_clip_norm == 1.0
    assert args.teacher_min_steps == 5
    assert args.teacher_max_steps == 20
    assert args.distill_learning_rate == 1e-3
    assert args.distill_epochs == 100
