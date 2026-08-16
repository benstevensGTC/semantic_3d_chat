from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from semantic_3d_chat.evaluation import fixed_prefix_decoder_reader_v6_1_release as release
from semantic_3d_chat.training import smoke_fixed_prefix_decoder_reader_v6_1 as smoke


def test_v6_1_smoke_source_is_zero_update_and_persists_failure_metrics() -> None:
    source = Path(
        "src/semantic_3d_chat/training/smoke_fixed_prefix_decoder_reader_v6_1.py"
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
    assert "claim_v6_1_mps_smoke_attempt()" in source
    assert '"optimizer_constructed": False' in source
    assert '"failure_metrics": getattr(failure, "metrics", {})' in source
    assert "register_forward_hook(capture_hidden)" in source
    assert "model.language_model.norm" in source
    assert "common_shape_reprojection" in source
    assert "softmax_ce_gradient" in source
    assert "target_rank_changes_confined_to_tie_band" in source
    assert "FileAccessAudit(" in source
    assert source.index("_isolated_retention_backward(", source.index("def _clean_gradient_equivalence")) < source.index(
        "_assign_snapshot_gradients(reader, tail_aggregate)",
        source.index("def _clean_gradient_equivalence"),
    )


def test_v6_1_retention_backward_is_fresh_and_leaves_gradients_cleared(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = torch.nn.Linear(1, 1, bias=False)
    parameter = next(model.parameters())
    parameter.data.fill_(1.0)
    parameter.grad = torch.full_like(parameter, 7.0)
    bundle = SimpleNamespace(language=SimpleNamespace(model=model))
    samples: list[str] = []
    sampler = SimpleNamespace(sample=samples.append)
    observed: dict[str, object] = {}

    monkeypatch.setattr(
        smoke.v1,
        "retention_kl_loss",
        lambda _bundle, _row, _teacher: parameter.sum() * 2.0,
    )

    def snapshot(_reader: object) -> dict[str, object]:
        observed["gradient"] = float(parameter.grad.detach())
        return {"a_exact_zero": False, "b_norms": {"opaque": 1.0}}

    monkeypatch.setattr(smoke, "_gradient_snapshot", snapshot)
    loss, captured = smoke._isolated_retention_backward(
        bundle,
        object(),
        {"prompt": "opaque"},
        torch.zeros(1),
        sampler,
    )
    assert float(loss.detach()) == 2.0
    assert observed["gradient"] == 1.0
    assert captured["a_exact_zero"] is False
    assert parameter.grad is None
    assert samples == ["after_retention_forward", "after_retention_backward"]


def test_v6_1_rank_metrics_use_dynamic_two_delta_band() -> None:
    reference = torch.tensor(
        [[1.00, 0.99, 0.40, 0.30, 0.20, 0.10, 0.0, -0.1, -0.2, -0.3, -0.4]]
    )
    selected = reference.clone()
    selected[0, 0] -= 0.01
    selected[0, 1] += 0.01
    metrics = smoke._prediction_rank_metrics(reference, selected, torch.tensor([0]))
    delta = metrics["per_token_max_vocabulary_abs_logit_difference"][0]
    tie_band = metrics["per_token_rank_tie_bands"][0]
    assert delta == pytest.approx(0.01)
    assert tie_band == pytest.approx(2.0 * delta)
    assert metrics["target_rank_changes_confined_to_tie_band"] is True
    assert metrics["strict_above_band_rank_exact"] is True


def test_v6_1_rank_metrics_expose_delta_above_preregistered_bound() -> None:
    reference = torch.tensor(
        [[1.0, 0.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0, -9.0]]
    )
    selected = reference.clone()
    selected[0, 0] -= 0.6
    selected[0, 1] += 0.6
    metrics = smoke._prediction_rank_metrics(reference, selected, torch.tensor([0]))
    # A crossing induced by bounded per-logit errors is necessarily within
    # 2*delta.  The preregistered max-delta gate is what rejects this case.
    assert metrics["target_rank_changes_confined_to_tie_band"] is True
    assert metrics["per_token_max_vocabulary_abs_logit_difference"][0] > 0.25


def test_v6_1_distribution_metrics_are_identity_for_identical_logits() -> None:
    logits = torch.tensor([[2.0, 1.0, 0.0], [0.0, 3.0, 1.0]])
    metrics = smoke._distribution_metrics(logits, logits.clone(), torch.tensor([0, 1]))
    assert metrics["maximum_js_divergence"] == pytest.approx(0.0, abs=1e-15)
    assert metrics["softmax_ce_gradient_max_abs_difference"] == 0.0
    assert metrics["softmax_ce_gradient_cosine_similarity"] == pytest.approx(1.0)


def test_v6_1_memory_contract_has_exactly_19_heavy_phases() -> None:
    assert len(release.MPS_MEMORY_PHASES) == 19
    assert {
        "before_model_load",
        "after_full_vs_tail_equivalence",
        "after_v6_gradient_validation",
        "after_joint_state_roundtrip",
    }.issubset(release.MPS_MEMORY_PHASES)


def test_v6_1_claimed_failure_terminalizes_with_metrics(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    attempt = tmp_path / "attempt.json"
    report = tmp_path / "report.json"
    release_path = tmp_path / "release.json"
    release_path.write_text("{}", encoding="utf-8")
    captured: list[dict[str, object]] = []
    monkeypatch.setattr(smoke, "MPS_SMOKE_ATTEMPT", str(attempt))
    monkeypatch.setattr(smoke, "MPS_SMOKE_REPORT", str(report))
    monkeypatch.setattr(smoke, "MPS_SMOKE_RELEASE", str(release_path))
    monkeypatch.setattr(
        smoke, "claim_v6_1_mps_smoke_attempt", lambda: (attempt, "a" * 64)
    )
    monkeypatch.setattr(smoke, "sha256_file", lambda _path: "b" * 64)
    failure = smoke.V61GateFailure(
        "bounded failure",
        stage="objective_equivalence",
        metrics={"objective_equivalence": {"passed": False}},
    )
    monkeypatch.setattr(
        smoke,
        "_execute_released_smoke",
        lambda **_kwargs: (_ for _ in ()).throw(failure),
    )
    monkeypatch.setattr(smoke, "_atomic_create_report", captured.append)
    with pytest.raises(smoke.V61GateFailure, match="bounded failure"):
        smoke.run_released_full_model_mps_smoke_v6_1()
    terminal = captured[0]
    assert terminal["status"] == "failed_terminal_attempt_consumed"
    assert terminal["failure_stage"] == "objective_equivalence"
    assert terminal["failure_metrics"] == {
        "objective_equivalence": {"passed": False}
    }
    assert terminal["optimizer_steps"] == 0


def test_v6_1_runner_exposes_smoke_and_trainer_modes() -> None:
    source = Path(
        "scripts/run_gemma4_v54_fixed_prefix_decoder_reader_v6_1.sh"
    ).read_text(encoding="utf-8")
    for mode in (
        "release-smoke",
        "authenticate-release-smoke",
        "smoke",
        "authenticate-smoke",
        "preflight",
        "release-training",
        "authenticate-release-training",
        "train",
        "authenticate",
    ):
        assert mode in source
    assert "train_fixed_prefix_decoder_reader_v6_1" in source
