from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from semantic_3d_chat.evaluation import predict_v66_internal_validation as predictor
from semantic_3d_chat.evaluation.question_manifest import QuestionManifest, QuestionRecord


def _manifest() -> QuestionManifest:
    questions = tuple(
        QuestionRecord(scene_id, f"q_{ordinal:06d}", "Where is it?")
        for ordinal, scene_id in enumerate(
            (
                scene_id
                for pair in predictor.INTERNAL_SCENE_PAIRS
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


def test_v66_runner_cli_exposes_no_reference_or_oracle_input() -> None:
    args = predictor._parse_args(
        [
            "--mode",
            "swap",
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
        for field in (
            "scorer_references",
            "qa",
            "oracle",
            "route_labels",
            "answers",
            "changed_questions",
        )
    )


def test_v66_runner_rejects_questions_under_forbidden_directory(tmp_path: Path) -> None:
    forbidden = tmp_path / "qa" / "questions.json"
    forbidden.parent.mkdir()
    forbidden.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="refuses QA/oracle"):
        predictor._safe_manifest_path(forbidden)


def test_v66_sealed_checkpoint_authentication_precedes_journal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = tmp_path / "questions.json"
    manifest_path.write_text("{}", encoding="utf-8")
    args = Namespace(
        mode="natural",
        config="runtime.yaml",
        questions_manifest=manifest_path,
        base_checkpoint=tmp_path / "base",
        control_checkpoint=tmp_path / "control",
        output=tmp_path / "predictions.jsonl",
        split="validation",
        no_resume=False,
    )
    monkeypatch.setattr(predictor, "load_runtime_config", lambda _path: {})
    monkeypatch.setattr(predictor, "load_question_manifest", lambda _path: _manifest())
    monkeypatch.setattr(predictor, "checkpoint_fingerprint", lambda _path: ("3" * 64, ()))
    observed: dict[str, Any] = {}

    def reject(
        checkpoint: Path,
        *,
        base_checkpoint_sha256: str,
        runtime_config_sha256: str,
    ) -> Any:
        observed.update(
            checkpoint=checkpoint,
            base=base_checkpoint_sha256,
            runtime=runtime_config_sha256,
        )
        raise ValueError("synthetic unsealed V7")

    monkeypatch.setattr(predictor, "_authenticate_sealed_v7", reject)
    monkeypatch.setattr(
        predictor,
        "build_prediction_provenance",
        lambda *_args, **_kwargs: pytest.fail("provenance built before V7 authentication"),
    )
    monkeypatch.setattr(
        predictor,
        "AtomicPredictionJournal",
        lambda *_args, **_kwargs: pytest.fail("journal created before V7 authentication"),
    )

    with pytest.raises(ValueError, match="unsealed V7"):
        predictor._run(args)

    assert observed["checkpoint"] == args.control_checkpoint.resolve()
    assert observed["base"] == "3" * 64
    assert not args.output.exists()


def test_prediction_record_attests_immutable_prefix_and_signature() -> None:
    class Runtime:
        scene_prefix_hash = "a" * 64
        scene_control_signature_hash = "b" * 64
        last_control_audit: dict[str, Any] | None = None

        def assert_prefix_unchanged(self) -> None:
            return None

        def answer(self, _question: str) -> Any:
            self.last_control_audit = {
                "architecture": predictor.ARCHITECTURE,
                "environment_latent_count": 256,
                "every_scene_token_influenced_output": True,
                "question_dependent_scene_retrieval": False,
                "control_used": True,
                "always_on_continuous_control": True,
            }
            return SimpleNamespace(
                answer="left",
                grounding_xyz_m=(1.0, 2.0, 0.5),
                grounding_confidence=0.75,
                generated_tokens=1,
                elapsed_seconds=0.1,
            )

    question = QuestionRecord("scene_000039", "q_000001", "Where is it?")
    row = predictor._prediction_record(
        source_scene_id=question.scene_id,
        question=question,
        injected_scene_id="scene_000040",
        runtime=Runtime(),  # type: ignore[arg-type]
        swap=True,
    )

    assert row["prefix_hash"] == "a" * 64
    assert row["scene_control_signature_sha256"] == "b" * 64
    assert row["injected_scene_id"] == "scene_000040"
    assert row["control_audit"]["control_used"] is True
