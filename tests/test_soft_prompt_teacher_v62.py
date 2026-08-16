from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from semantic_3d_chat.training import soft_prompt_teacher_v62 as v62

_A = "a" * 64
_B = "b" * 64
_C = "c" * 64
_D = "d" * 64
_E = "e" * 64


def _row(scene_id: str, question_id: str, changed: bool) -> dict[str, object]:
    return {
        "scene_id": scene_id,
        "question_id": question_id,
        "question": f"Question {question_id}?",
        "answer": f"answer-{question_id}",
        "answer_type": "attribute",
        "counterfactual_pair_id": "pair_000005",
        "counterfactual_question_key": "cfq_0000000000000001",
        "counterfactual_role": "reference",
        "counterfactual_change_type": "color_swap",
        "counterfactual_expected_change": changed,
    }


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _metrics() -> dict[str, object]:
    return {
        "steps": 2,
        "initial_nll": 1.0,
        "final_nll": 0.1,
        "minimum_nll": 0.1,
        "maximum_preclip_gradient_norm": 0.5,
        "initial_rms": 0.05,
        "final_rms": 0.06,
        "learning_rate": 0.03,
        "attempt_count": 1,
        "attempt_learning_rates": [0.03],
        "total_forward_steps": 2,
    }


def _preflight(tmp_path: Path) -> v62.V62TeacherPreflight:
    records = (
        SimpleNamespace(
            scene_id="scene_000011",
            question_id="q_000001",
            question="Question?",
            answer="answer",
        ),
        SimpleNamespace(
            scene_id="scene_000012",
            question_id="q_000002",
            question="Different question?",
            answer="different",
        ),
    )
    manifest = {
        "run_signature_sha256": _A,
    }
    return v62.V62TeacherPreflight(
        config={"paths": {"checkpoints_root": str(tmp_path / "derived" / "checkpoints")}},
        config_path=tmp_path / "runtime.yaml",
        runtime_config_sha256=_B,
        base_checkpoint=tmp_path / "base",
        base_checkpoint_sha256=_C,
        source_control_checkpoint=tmp_path / "control",
        source_control_checkpoint_sha256=_D,
        source_control=torch.nn.Identity(),
        source_control_metadata={
            "architecture": "synthetic",
            "hidden_size": 1536,
            "control_tokens": 4,
        },
        filtered_train_jsonl=tmp_path / "filtered_train.jsonl",
        filtered_train_jsonl_sha256=_E,
        preregistration_sha256="1" * 64,
        baseline_lock=tmp_path / "baseline.json",
        baseline_lock_sha256="2" * 64,
        baseline_lock_authorization={"v54_checkpoint_sha256": _C},
        scene_ids=("scene_000011", "scene_000012"),
        records=records,  # type: ignore[arg-type]
        total_filtered_rows=4,
        selection_sha256="3" * 64,
        prefixes={
            "scene_000011": torch.zeros(1, 258, 1536),
            "scene_000012": torch.ones(1, 258, 1536),
        },
        prefix_cache=tmp_path / "prefixes",
        prefix_cache_manifest_sha256="4" * 64,
        work_directory=tmp_path / "derived" / "training" / "work",
        output_artifact=tmp_path / "derived" / "training" / "final",
        optimizer={
            "learning_rate": 0.03,
            "minimum_steps": 1,
            "maximum_steps": 2,
            "nll_threshold": 0.001,
            "gradient_clip_norm": 1.0,
        },
        run_manifest=manifest,
    )


def test_changed_selector_uses_exact_true_predicate() -> None:
    rows = [
        _row("scene_000011", "q_000001", True),
        _row("scene_000011", "q_000002", False),
        _row("scene_000012", "q_000003", True),
        _row("scene_000012", "q_000004", False),
    ]

    selected, total, digest = v62._load_changed_training_records(
        rows, scene_ids=("scene_000011", "scene_000012")
    )

    assert total == 4
    assert [(row.scene_id, row.question_id) for row in selected] == [
        ("scene_000011", "q_000001"),
        ("scene_000012", "q_000003"),
    ]
    assert len(digest) == 64


def test_filtered_training_path_requires_explicit_sha_and_boundary_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "filtered.jsonl"
    digest = _write_jsonl(source, [_row("scene_000011", "q_000001", True)])
    calls: list[Path] = []

    def authenticated(path: str | Path) -> tuple[dict[str, object], ...]:
        calls.append(Path(path))
        return (_row("scene_000011", "q_000001", True),)

    monkeypatch.setattr(v62, "load_filtered_training_qa", authenticated)
    resolved, observed, rows = v62._validated_filtered_train_path(source, digest)

    assert resolved == source.resolve()
    assert observed == digest
    assert len(rows) == 1
    assert calls == [source.resolve()]
    with pytest.raises(ValueError, match="digest changed"):
        v62._validated_filtered_train_path(source, "0" * 64)


def test_dry_run_validates_baseline_before_training_and_never_loads_gemma(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    filtered = tmp_path / "filtered.jsonl"
    filtered.write_text("training\n", encoding="utf-8")
    baseline = tmp_path / "baseline.json"
    baseline.write_text("{}\n", encoding="utf-8")
    observed: list[str] = []

    def validate_lock(path: str | Path) -> dict[str, str]:
        observed.append("baseline")
        assert Path(path) == baseline
        return {"v54_checkpoint_sha256": _C}

    def forbidden_training_loader(_path: str | Path) -> tuple[dict[str, object], ...]:
        observed.append("training")
        raise RuntimeError("stop after proving ordering")

    monkeypatch.setattr(v62, "validate_baseline_lock", validate_lock)
    monkeypatch.setattr(v62, "load_filtered_training_qa", forbidden_training_loader)
    args = SimpleNamespace(
        seed=1,
        teacher_min_steps=1,
        teacher_max_steps=2,
        teacher_learning_rate=0.03,
        teacher_nll_threshold=0.001,
        teacher_gradient_clip_norm=1.0,
        baseline_lock=str(baseline),
        scene_id=["scene_000011"],
        filtered_train_qa=str(filtered),
        filtered_train_sha256=hashlib.sha256(filtered.read_bytes()).hexdigest(),
    )

    with pytest.raises(RuntimeError, match="stop after proving ordering"):
        v62.build_v62_teacher_preflight(args)  # type: ignore[arg-type]
    assert observed == ["baseline", "training"]


def test_per_record_cache_resumes_and_final_artifact_is_numeric_and_opaque(
    tmp_path: Path,
) -> None:
    preflight = _preflight(tmp_path)
    v62._prepare_work_directory(preflight)
    first = v62._save_completed_record(
        preflight,
        preflight.records[0],
        torch.full(v62.PROMPT_SHAPE, 0.25),
        _metrics(),
    )

    resumed = v62._existing_completed_records(preflight)
    assert set(resumed) == {first.key}
    assert torch.equal(resumed[first.key].prompt, first.prompt)

    second = v62._save_completed_record(
        preflight,
        preflight.records[1],
        torch.full(v62.PROMPT_SHAPE, -0.125),
        _metrics(),
    )
    metadata = v62.save_v62_teacher_cache(preflight, (first, second))
    loaded, validated = v62.load_v62_teacher_cache(preflight.output_artifact)

    assert metadata == validated
    assert set(loaded) == {first.key, second.key}
    assert validated["runtime_load_permitted"] is False
    assert validated["environmental_text_inputs"] == []
    assert validated["validation_inputs_used"] is False
    assert validated["held_out_inputs_used"] is False
    assert validated["greedy_canonical_exact"] == 2
    serialized = json.dumps(validated, sort_keys=True).casefold()
    assert "question?" not in serialized
    assert "answer-" not in serialized
    assert {item.name for item in preflight.output_artifact.iterdir()} == {
        "metadata.json",
        "teachers.safetensors",
    }


def test_generation_resumes_completed_records_after_interruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    preflight = _preflight(tmp_path)
    first_prompt = torch.full(v62.PROMPT_SHAPE, 0.1)
    second_prompt = torch.full(v62.PROMPT_SHAPE, 0.2)
    calls: list[str] = []

    class Language:
        def enable_decoder_gradient_checkpointing(self) -> None:
            calls.append("enable")

    runtime = SimpleNamespace(language=Language())

    monkeypatch.setattr(v62, "build_v62_teacher_preflight", lambda _args: preflight)
    monkeypatch.setattr(
        v62,
        "_source_initial_prompt",
        lambda *_args: torch.zeros(v62.PROMPT_SHAPE),
    )
    args = argparse.Namespace(dry_run_inventory=False, device="cpu", seed=7)
    optimized = 0

    def optimize(**_kwargs: object) -> tuple[torch.Tensor, dict[str, object]]:
        nonlocal optimized
        optimized += 1
        if optimized == 2:
            raise KeyboardInterrupt
        return first_prompt, _metrics()

    with pytest.raises(KeyboardInterrupt):
        v62.generate_v62_teacher_cache(
            args,
            runtime_provider=lambda *_args: (runtime, torch.device("cpu"), torch.float32),
            question_embedder=lambda *_args: torch.zeros(1, 1, 1536),
            optimizer_fn=optimize,
            generator_fn=lambda **_kwargs: "answer",
            disable_checkpointing_fn=lambda _language: calls.append("disable"),
        )
    assert len(v62._existing_completed_records(preflight)) == 1

    optimized = 0
    result = v62.generate_v62_teacher_cache(
        args,
        runtime_provider=lambda *_args: (runtime, torch.device("cpu"), torch.float32),
        question_embedder=lambda *_args: torch.zeros(1, 1, 1536),
        optimizer_fn=lambda **_kwargs: (second_prompt, _metrics()),
        generator_fn=lambda **kwargs: (
            "answer"
            if kwargs["question"] == preflight.records[0].question
            else "different"
        ),
        disable_checkpointing_fn=lambda _language: calls.append("disable"),
    )

    assert result["mode"] == "complete"
    assert result["resumed_record_count"] == 1
    assert result["new_record_count"] == 1
    loaded, _metadata = v62.load_v62_teacher_cache(preflight.output_artifact)
    assert len(loaded) == 2


def test_greedy_verification_failure_uses_deterministic_low_rate_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    preflight = _preflight(tmp_path)
    calls: list[str] = []

    class Language:
        def enable_decoder_gradient_checkpointing(self) -> None:
            calls.append("enable")

    runtime = SimpleNamespace(language=Language())
    monkeypatch.setattr(v62, "build_v62_teacher_preflight", lambda _args: preflight)
    monkeypatch.setattr(
        v62,
        "_source_initial_prompt",
        lambda *_args: torch.zeros(v62.PROMPT_SHAPE),
    )
    fallback_calls: list[dict[str, object]] = []

    def fallback(**kwargs: object) -> tuple[torch.Tensor, dict[str, object]]:
        fallback_calls.append(kwargs)
        metrics = _metrics()
        for field in ("attempt_count", "attempt_learning_rates", "total_forward_steps"):
            metrics.pop(field)
        metrics.update(
            {
                "steps": 100,
                "learning_rate": 0.0001,
                "final_nll": 0.006,
                "minimum_nll": 0.006,
            }
        )
        return torch.full(v62.PROMPT_SHAPE, 0.75), metrics

    def generate(**kwargs: object) -> str:
        prompt = kwargs["control_tokens"]
        assert isinstance(prompt, torch.Tensor)
        if float(prompt.mean()) < 0.5:
            return "wrong"
        return "answer" if kwargs["question"] == "Question?" else "different"

    result = v62.generate_v62_teacher_cache(
        argparse.Namespace(dry_run_inventory=False, device="cpu", seed=7),
        runtime_provider=lambda *_args: (runtime, torch.device("cpu"), torch.float32),
        question_embedder=lambda *_args: torch.zeros(1, 1, 1536),
        optimizer_fn=lambda **_kwargs: (torch.zeros(v62.PROMPT_SHAPE), _metrics()),
        fallback_optimizer_fn=fallback,
        generator_fn=generate,
        disable_checkpointing_fn=lambda _language: calls.append("disable"),
    )

    assert result["mode"] == "complete"
    assert len(fallback_calls) == 2
    assert all(call["learning_rate"] == pytest.approx(0.0001) for call in fallback_calls)
    assert all(call["max_steps"] == 100 for call in fallback_calls)
    loaded, metadata = v62.load_v62_teacher_cache(preflight.output_artifact)
    assert len(loaded) == 2
    assert all(record["optimization"]["total_forward_steps"] == 100 for record in metadata["records"])
