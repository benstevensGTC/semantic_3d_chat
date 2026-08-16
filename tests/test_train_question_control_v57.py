from __future__ import annotations

import json
from pathlib import Path

import pytest

from semantic_3d_chat.evaluation.predict_question_control import (
    _control_checkpoint_sha256,
)
from semantic_3d_chat.training.train_question_control_v57 import (
    _conditioning_pair_settings,
    _pair_settings,
    _parser,
    _validate_cli_numbers,
)


def _arguments() -> list[str]:
    return [
        "--base-runtime-config",
        "configs/runtime/gemma4_v56_question_control.yaml",
        "--base-checkpoint",
        "data_gemma4/checkpoints/gemma4_v54_semantic_greedy",
        "--train-qa",
        "data_diverse52/qa/train.jsonl",
        "--scene-id",
        "scene_000031",
        "--scene-id",
        "scene_000032",
        "--prefix-cache",
        "data_gemma4/scene_tokens/v56_question_control_smoke_prefixes",
        "--output-checkpoint",
        "data_gemma4/checkpoints/v57_test",
        "--training-report",
        "reports/gemma4/metrics/v57_test.json",
    ]


def test_v57_cli_defaults_encode_delta_sensitive_retry() -> None:
    args = _parser().parse_args(_arguments())
    _validate_cli_numbers(args)
    settings = _pair_settings(args)

    assert args.epochs == 5
    assert args.learning_rate == 3e-5
    assert settings.side_hinge_weight == 0.5
    assert settings.cross_prefix_hinge_weight == 1.0
    assert settings.control_delta_weight == 8.0
    assert settings.minimum_relative_control_delta == 0.03
    assert settings.minimum_normalized_attention_entropy == 0.55
    assert settings.attention_logit_spread_weight == 1.0
    assert settings.answer_alignment_weight == 2.0
    assert settings.answer_absolute_alignment_weight == 1.0
    assert settings.answer_delta_alignment_weight == 2.0

    conditioning = _conditioning_pair_settings(args, settings)
    assert args.conditioning_epochs == 1
    assert conditioning.side_hinge_weight == 0.0
    assert conditioning.cross_prefix_hinge_weight == 0.0
    assert conditioning.answer_nll_weight == 0.25
    assert conditioning.control_delta_weight == 12.0
    assert conditioning.answer_absolute_alignment_weight == 2.0
    assert conditioning.answer_delta_alignment_weight == 4.0

    args.epochs = 1
    args.conditioning_epochs = 1
    _validate_cli_numbers(args)


def test_v57_cli_rejects_invalid_regularizer() -> None:
    args = _parser().parse_args(
        [*_arguments(), "--minimum-normalized-attention-entropy", "1.1"]
    )
    with pytest.raises(ValueError, match="entropy"):
        _validate_cli_numbers(args)


def test_v57_warm_start_is_optional_and_runtime_architecture_stays_v1() -> None:
    fresh = _parser().parse_args(_arguments())
    warm = _parser().parse_args(
        [
            *_arguments(),
            "--initial-control-checkpoint",
            "data_gemma4/checkpoints/gemma4_v56_question_control_overfit_pair31_32",
        ]
    )
    assert fresh.initial_control_checkpoint is None
    assert warm.initial_control_checkpoint.endswith("overfit_pair31_32")
    assert fresh.attention_dim == warm.attention_dim == 256
    assert fresh.control_tokens == warm.control_tokens == 4


def test_v57_warm_start_fingerprint_accepts_exact_two_file_runtime_checkpoint(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "control"
    checkpoint.mkdir()
    (checkpoint / "control.safetensors").write_bytes(b"opaque weights")
    (checkpoint / "runtime_metadata.json").write_text(
        json.dumps({"opaque": True}), encoding="utf-8"
    )
    first = _control_checkpoint_sha256(checkpoint)
    second = _control_checkpoint_sha256(checkpoint)
    assert first == second
    assert len(first) == 64

    (checkpoint / "metadata.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="runtime-minimal"):
        _control_checkpoint_sha256(checkpoint)
