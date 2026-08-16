from __future__ import annotations

from pathlib import Path

import pytest

from semantic_3d_chat.chat.runtime import ChatAnswer
from semantic_3d_chat.evaluation.leakage import (
    oracle_temporarily_unavailable,
    run_leakage_evaluation,
)


class FakeRuntime:
    def __init__(self) -> None:
        self.scene_prefix_hash = "a" * 64
        self.questions_answered = 0

    def startup_summary(self):
        return {
            "prefix_hash": self.scene_prefix_hash,
            "processed_voxels": 12,
            "strict_fixed_environment_embedding_input": True,
            "environment_conditioned_input_sha256": self.scene_prefix_hash,
            "question_conditioned_scene_readout_tokens": False,
        }

    def current_prefix_hash(self):
        return self.scene_prefix_hash

    def assert_prefix_unchanged(self):
        assert self.current_prefix_hash() == self.scene_prefix_hash

    def answer(self, question: str):
        self.questions_answered += 1
        return ChatAnswer(
            question=question,
            answer="unknown",
            grounding_xyz_m=(0.0, 0.0, 1.0),
            grounding_confidence=0.25,
            grounding_support_distance_m=0.1,
            prefix_hash=self.scene_prefix_hash,
            generated_tokens=1,
            elapsed_seconds=0.01,
        )


def test_oracle_context_restores_after_failure(tmp_path: Path) -> None:
    oracle = tmp_path / "oracle"
    oracle.mkdir()
    truth = oracle / "truth.json"
    truth.write_text("isolated", encoding="utf-8")
    with (
        pytest.raises(RuntimeError, match="deliberate"),
        oracle_temporarily_unavailable(oracle) as state,
    ):
        assert state.renamed
        assert not oracle.exists()
        assert state.hidden is not None and state.hidden.exists()
        raise RuntimeError("deliberate")
    assert truth.read_text(encoding="utf-8") == "isolated"


def test_leakage_evaluation_loads_runtime_while_oracle_is_absent(tmp_path: Path) -> None:
    data = tmp_path / "data"
    oracle = data / "oracle"
    maps = data / "maps" / "scene_000001"
    oracle.mkdir(parents=True)
    maps.mkdir(parents=True)
    (oracle / "truth.json").write_text("never read", encoding="utf-8")
    numeric = maps / "voxel_map.npz"
    numeric.write_bytes(b"numeric-only")
    reports = tmp_path / "reports"
    config = tmp_path / "config.yaml"
    config.write_text(
        f"paths:\n  data_root: {data}\n  reports_root: {reports}\n",
        encoding="utf-8",
    )
    observed = {"oracle_absent": False}

    def loader(_config, _scene_id, _checkpoint, audit):
        observed["oracle_absent"] = not oracle.exists()
        numeric.read_bytes()
        audit.record(numeric)
        return FakeRuntime()

    output = reports / "metrics" / "leakage.json"
    report = run_leakage_evaluation(
        config_path=config,
        questions=("first?", "second?"),
        report_path=output,
        runtime_loader=loader,
    )
    assert observed["oracle_absent"]
    assert oracle.exists()
    assert report["passed"]
    assert report["oracle_was_renamed"]
    assert report["oracle_restored"]
    assert report["prefix_computed_before_first_question"]
    assert report["prefix_invariant"]
    assert report["strict_fixed_environment_embedding_input"] is True
    assert report["forbidden_accesses"] == []
    assert str(numeric.resolve()) in report["loaded_files"]
    assert output.is_file()


def test_leakage_audit_records_recursive_config_sources(tmp_path: Path) -> None:
    data = tmp_path / "data"
    oracle = data / "oracle"
    oracle.mkdir(parents=True)
    reports = tmp_path / "reports"
    base = tmp_path / "base.yaml"
    base.write_text(
        f"paths:\n  data_root: {data}\n  reports_root: {reports}\n",
        encoding="utf-8",
    )
    config = tmp_path / "experiment.yaml"
    config.write_text("_base_: base.yaml\n", encoding="utf-8")

    report = run_leakage_evaluation(
        config_path=config,
        questions=("one?",),
        report_path=reports / "metrics" / "leakage.json",
        runtime_loader=lambda *_args: FakeRuntime(),
    )

    assert report["passed"]
    assert str(config.resolve()) in report["loaded_files"]
    assert str(base.resolve()) in report["loaded_files"]
