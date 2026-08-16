from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest

from semantic_3d_chat.evaluation import predict_v62_scene_swap as predictor
from semantic_3d_chat.evaluation.question_manifest import QuestionManifest, QuestionRecord


def _manifest() -> QuestionManifest:
    questions = tuple(
        QuestionRecord(scene_id, f"q_{ordinal:06d}", "Where is it?")
        for ordinal, scene_id in enumerate(
            (
                scene_id
                for pair in predictor.V62_INTERNAL_SCENE_PAIRS
                for scene_id in pair
                for _ in range(24)
            ),
            start=1,
        )
    )
    return QuestionManifest(
        questions=questions,
        questions_sha256="1" * 64,
        source_qa_sha256="2" * 64,
    )


def test_scene_swap_cli_exposes_no_answer_or_route_input() -> None:
    args = predictor._parse_args(
        [
            "--config",
            "runtime.yaml",
            "--questions-manifest",
            "questions.json",
            "--base-checkpoint",
            "base",
            "--control-checkpoint",
            "control",
            "--output",
            "predictions.jsonl",
        ]
    )
    assert not any(
        hasattr(args, field)
        for field in ("scorer_references", "qa", "oracle", "route_labels", "changed_questions")
    )


def test_scene_swap_authenticates_sealed_v65_before_creating_journal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = Namespace(
        config="runtime.yaml",
        questions_manifest=tmp_path / "questions.json",
        base_checkpoint=tmp_path / "base",
        control_checkpoint=tmp_path / "control",
        output=tmp_path / "predictions.jsonl",
        split="validation",
        no_resume=False,
    )
    monkeypatch.setattr(predictor, "_parse_args", lambda _argv=None: args)
    monkeypatch.setattr(predictor, "load_runtime_config", lambda _path: {})
    monkeypatch.setattr(predictor, "load_question_manifest", lambda _path: _manifest())
    monkeypatch.setattr(
        predictor,
        "checkpoint_fingerprint",
        lambda _path: ("3" * 64, ()),
    )
    monkeypatch.setattr(
        predictor,
        "effective_runtime_config_sha256",
        lambda _config: "4" * 64,
    )

    observed: dict[str, object] = {}

    def reject_unsealed(
        checkpoint: Path,
        *,
        base_checkpoint_sha256: str,
        runtime_config_sha256: str,
    ) -> object:
        observed.update(
            checkpoint=checkpoint,
            base=base_checkpoint_sha256,
            runtime=runtime_config_sha256,
        )
        raise ValueError("synthetic ungated V65")

    monkeypatch.setattr(predictor, "validate_sealed_v65_checkpoint", reject_unsealed)
    monkeypatch.setattr(
        predictor,
        "AtomicPredictionJournal",
        lambda *_args, **_kwargs: pytest.fail("journal created before V65 authentication"),
    )

    with pytest.raises(ValueError, match="ungated V65"):
        predictor.main([])

    assert observed == {
        "checkpoint": args.control_checkpoint.resolve(),
        "base": "3" * 64,
        "runtime": "4" * 64,
    }
    assert not args.output.exists()
