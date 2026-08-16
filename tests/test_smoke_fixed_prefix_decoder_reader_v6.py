from __future__ import annotations

from pathlib import Path

import pytest
import torch

from semantic_3d_chat.evaluation.fixed_prefix_decoder_reader_v6_preregistration import (
    INITIALIZATION_SEED,
    _ShapeOnlyDecoder,
    decoder_reader_lora_settings_v6,
)
from semantic_3d_chat.language.gemma4_tool_decoder_v2 import (
    PROJECTOR_INITIALIZATION_SEED,
    tool_decoder_lora_settings_v2,
)
from semantic_3d_chat.language.lora import (
    initialize_lora_adapter_state,
    install_lora_adapters,
)
from semantic_3d_chat.training import smoke_fixed_prefix_decoder_reader_v6 as smoke
from semantic_3d_chat.training.smoke_fixed_prefix_decoder_reader_v6 import (
    _forbidden_evaluation_roots,
    _joint_state_roundtrip,
    _MPSMemorySampler,
)


def test_v6_real_smoke_source_has_no_optimizer_or_checkpoint_writer() -> None:
    source = Path(
        "src/semantic_3d_chat/training/smoke_fixed_prefix_decoder_reader_v6.py"
    ).read_text(encoding="utf-8")
    forbidden = (
        "torch.optim",
        "AdamW(",
        "optimizer.step(",
        "save_file(",
        "OUTPUT_CHECKPOINT",
        "scene_000057",
        "scene_000025",
    )
    assert all(fragment not in source for fragment in forbidden)
    assert "claim_mps_smoke_attempt()" in source
    assert '"optimizer_constructed": False' in source
    assert '"deferred_or_final_qa_accessed": bool(audit.forbidden_accesses())' in source
    assert '"selected_logits_exact"' in source
    assert '"per_token_nll_max_abs_difference"' in source
    assert "FileAccessAudit(" in source
    assert "mps_driver_allocated_bytes_sampled_peak" in source


def test_v6_joint_state_roundtrip_helper_is_strict() -> None:
    model = _ShapeOnlyDecoder().requires_grad_(False)
    reader = install_lora_adapters(model, decoder_reader_lora_settings_v6())
    assert reader is not None
    initialize_lora_adapter_state(reader, seed=INITIALIZATION_SEED)
    for parameter in reader.parameters():
        parameter.requires_grad_(False)
    tool = install_lora_adapters(model, tool_decoder_lora_settings_v2())
    assert tool is not None
    initialize_lora_adapter_state(tool, seed=PROJECTOR_INITIALIZATION_SEED)
    for parameter in tool.parameters():
        parameter.requires_grad_(False)
    before = {
        "reader": reader.state_sha256(),
        "tool": tool.state_sha256(),
    }
    report = _joint_state_roundtrip(reader, tool)
    assert report["strict_state_roundtrip"] is True
    assert report["reader_state_sha256"] == before["reader"]
    assert report["tool_state_sha256"] == before["tool"]
    assert all(not parameter.requires_grad for parameter in model.parameters())
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())


def test_v6_smoke_blocks_deferred_and_final_evaluation_assets() -> None:
    roots = {str(path) for path in _forbidden_evaluation_roots()}
    assert str(Path("data_diverse52/qa/validation.jsonl").resolve()) in roots
    assert str(
        Path("reports/gemma4/questions/v56_fresh_development_validation.json").resolve()
    ) in roots
    assert str(Path("reports/gemma4/questions/test.json").resolve()) in roots
    for scene_id in ("scene_000025", "scene_000057", "scene_000062"):
        assert str(Path("data_gemma4/maps", scene_id).resolve()) in roots
        assert str(Path("data_gemma4/features", scene_id).resolve()) in roots


def test_v6_smoke_memory_sampler_reports_sampled_maximum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    samples = iter((10, 40, 20, 30, 5, 6, 7, 8, 9, 11))
    monkeypatch.setattr(torch.mps, "driver_allocated_memory", lambda: next(samples))
    monkeypatch.setattr(
        smoke.v1,
        "memory_metrics",
        lambda: {
            "peak_process_rss_bytes": 100,
            "mps_current_allocated_bytes": 20,
            "mps_driver_allocated_bytes": 30,
        },
    )
    sampler = _MPSMemorySampler()
    for index in range(10):
        sampler.sample(f"phase_{index}")
    report = sampler.report()
    assert report["mps_driver_allocated_bytes_sampled_peak"] == 40
    assert report["mps_driver_sample_count"] == 10
    assert len(report["mps_driver_samples_by_phase"]) == 10


def test_v6_claimed_smoke_failure_is_terminal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    attempt = tmp_path / "attempt.json"
    report = tmp_path / "report.json"
    release = tmp_path / "release.json"
    release.write_text("{}", encoding="utf-8")
    captured: list[dict[str, object]] = []
    monkeypatch.setattr(smoke, "MPS_SMOKE_ATTEMPT", str(attempt))
    monkeypatch.setattr(smoke, "MPS_SMOKE_REPORT", str(report))
    monkeypatch.setattr(smoke, "MPS_SMOKE_RELEASE", str(release))
    monkeypatch.setattr(
        smoke,
        "claim_mps_smoke_attempt",
        lambda: (attempt, "a" * 64),
    )
    monkeypatch.setattr(smoke, "sha256_file", lambda _path: "b" * 64)
    monkeypatch.setattr(
        smoke,
        "_execute_released_smoke",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("locked failure")),
    )
    monkeypatch.setattr(smoke, "_atomic_create_report", captured.append)

    with pytest.raises(RuntimeError, match="locked failure"):
        smoke.run_released_full_model_mps_smoke()
    assert captured[0]["status"] == "failed_terminal_attempt_consumed"
    assert captured[0]["passed"] is False
    assert captured[0]["optimizer_steps"] == 0
