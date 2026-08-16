from __future__ import annotations

import json
from pathlib import Path

import pytest

import semantic_3d_chat.training.train_gemma4_tool_decoder_v2 as trainer
from semantic_3d_chat.config import load_config
from semantic_3d_chat.evaluation.gemma4_tool_decoder_training_authorization_v2_2 import (
    AUTHORIZATION_FIELDS,
    SUPERSEDED_AUTHORIZATION_PATH,
    SUPERSEDED_AUTHORIZATION_SHA256,
    build_cpu_authorization_v2_2,
    sha256_file,
    write_cpu_authorization_v2_2,
    write_mps_smoke_release_v2_2,
    write_training_release_v2_2,
)


def _smoke(release_sha256: str) -> dict[str, object]:
    return {
        "schema": "semantic_3d_chat.gemma4_tool_decoder_full_mps_smoke.v2_2",
        "status": "passed",
        "authorization_sha256": release_sha256,
        "device": "mps",
        "full_model_loaded": True,
        "mps_used": True,
        "sample_id": "g_00000000",
        "microbatches": 1,
        "optimizer_steps": 0,
        "loss": 1.0,
        "real_full_vs_tail_answer_nll_absolute_difference": 0.0,
        "real_full_vs_tail_answer_nll_tolerance": 1e-6,
        "training_and_evaluation_use_answer_tail_only": True,
        "lora_gradient_l2": 1.0,
        "projector_gradient_l2": 1.0,
        "trainable_parameter_count": 165888,
        "source": {},
        "elapsed_seconds": 1.0,
        "training_executed": False,
        "checkpoint_published": False,
    }


def test_superseded_v2_1_cpu_authorization_evidence_remains_byte_exact() -> None:
    assert (
        sha256_file(SUPERSEDED_AUTHORIZATION_PATH)
        == SUPERSEDED_AUTHORIZATION_SHA256
    )
    current = build_cpu_authorization_v2_2()
    assert current["supersedes_authorization_path"] == SUPERSEDED_AUTHORIZATION_PATH
    assert current["supersedes_authorization_sha256"] == (
        SUPERSEDED_AUTHORIZATION_SHA256
    )


def test_v2_2_authorization_fields_and_default_cpu_denial_are_exact() -> None:
    payload, digest = trainer.authenticate_training_authorization_v2()
    assert payload == build_cpu_authorization_v2_2()
    assert set(payload) == set(AUTHORIZATION_FIELDS)
    assert len(digest) == 64
    assert payload["authorization_stage"] == "cpu_preparation"
    assert payload["full_model_mps_microbatch_authorized"] is False
    assert payload["multi_update_training_authorized"] is False
    with pytest.raises(PermissionError, match="cannot authorize"):
        trainer.authenticate_training_authorization_v2(
            required_stage="full_model_mps_microbatch"
        )


def test_complete_successor_chain_authenticates_exact_report_sha(tmp_path: Path) -> None:
    cpu_path, cpu_sha = write_cpu_authorization_v2_2(tmp_path / "cpu.json")
    release_path, release_sha = write_mps_smoke_release_v2_2(
        tmp_path / "smoke_release.json", cpu_authorization=cpu_path
    )
    cpu, observed_cpu_sha = trainer.authenticate_training_authorization_v2(
        cpu_path, required_stage="cpu_preparation"
    )
    release, observed_release_sha = trainer.authenticate_training_authorization_v2(
        release_path, required_stage="full_model_mps_microbatch"
    )
    assert observed_cpu_sha == cpu_sha
    assert observed_release_sha == release_sha
    assert release["parent_authorization_sha256"] == cpu_sha
    smoke_path = tmp_path / "smoke.json"
    smoke_path.write_text(
        json.dumps(_smoke(release_sha), sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    training_path, training_sha = write_training_release_v2_2(
        tmp_path / "training_release.json",
        smoke_release=release_path,
        smoke_report=smoke_path,
    )
    training, observed_training_sha = trainer.authenticate_training_authorization_v2(
        training_path, required_stage="multi_update_training"
    )
    assert observed_training_sha == training_sha
    assert training["parent_authorization_sha256"] == release_sha
    assert training["full_model_mps_microbatch_smoke_sha256"] == sha256_file(
        smoke_path
    )
    assert training["full_model_mps_microbatch_smoke"]["optimizer_steps"] == 0
    assert cpu["execution"]["optimizer_steps"] == 0

    # Any byte change after release invalidates the descendant authorization.
    smoke_path.write_text(smoke_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="evidence|bytes"):
        trainer.authenticate_training_authorization_v2(
            training_path, required_stage="multi_update_training"
        )


def test_invalid_smoke_cannot_create_training_release(tmp_path: Path) -> None:
    cpu_path, _ = write_cpu_authorization_v2_2(tmp_path / "cpu.json")
    release_path, release_sha = write_mps_smoke_release_v2_2(
        tmp_path / "smoke_release.json", cpu_authorization=cpu_path
    )
    bad = _smoke(release_sha)
    bad["optimizer_steps"] = 1
    report = tmp_path / "bad_smoke.json"
    report.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="did not pass"):
        write_training_release_v2_2(
            tmp_path / "training_release.json",
            smoke_release=release_path,
            smoke_report=report,
        )
    assert not (tmp_path / "training_release.json").exists()


def test_denied_entrypoints_never_reach_full_model_loader(monkeypatch) -> None:
    reached = False

    def forbidden(*_args, **_kwargs):
        nonlocal reached
        reached = True
        raise AssertionError("full model loader reached")

    monkeypatch.setattr(trainer, "_load_training_bundle", forbidden)
    config = load_config("configs/experiments/gemma4_embodied_tool_decoder_v2.yaml")
    with pytest.raises(PermissionError, match="cannot authorize"):
        trainer.run_full_model_mps_microbatch_smoke_v2(config)
    with pytest.raises(PermissionError, match="cannot authorize"):
        trainer.train_gemma4_tool_decoder_v2(
            config,
            report_path="reports/gemma4/metrics/NEVER.json",
            runtime_checkpoint="data_gemma4/checkpoints/NEVER",
            runtime_probe=lambda _path: {},
        )
    assert reached is False
