from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open
from torch import nn

from semantic_3d_chat.evaluation.fixed_prefix_decoder_reader_v6_preregistration import (
    answer_varying_wrong_prefixes,
    build_v6_schedule,
)
from semantic_3d_chat.language.fixed_prefix_decoder_reader_v6 import (
    LORA_PARAMETER_COUNT,
    TARGET_MODULES,
)
from semantic_3d_chat.training import train_fixed_prefix_decoder_reader_v6_1 as train
from semantic_3d_chat.training.train_fixed_prefix_decoder_reader_v6_1 import (
    _publish_checkpoint,
    _v6_1_smoke_authentication,
    broad_row_objective,
    contrastive_row_objective,
    optimizer_kwargs,
    retention_objective,
    teacher_and_retention_checks,
)
from semantic_3d_chat.training.train_fixed_prefix_ple_v54 import (
    load_training_records,
    load_validation_records,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _gate_metrics(rate: float, *, answer_nll: float, complete: int) -> dict[str, object]:
    families = {key: rate for key in train._EXPECTED_FAMILIES}
    scopes = {key: rate for key in train._EXPECTED_SCOPES}
    return {
        "answer_nll_mean": answer_nll,
        "curated_positive_margin_rate": rate,
        "curated_complete_units": complete,
        "expanded_positive_margin_rate": rate,
        "family_macro_positive_margin_rate": rate,
        "family_positive_margin_rates": families,
        "scope_macro_positive_margin_rate": rate,
        "scope_positive_margin_rates": scopes,
    }


def test_v6_1_smoke_authentication_requires_exact_tuple_api(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    smoke_path = tmp_path / "smoke.json"
    smoke_path.write_text('{"passed":true,"status":"passed"}\n', encoding="utf-8")
    release_sha256 = "1" * 64
    report = {
        "passed": True,
        "status": "passed",
        "authorization_sha256": release_sha256,
    }
    module = SimpleNamespace(
        authenticate_v6_1_mps_smoke_release=lambda: (
            {"terminal_output": str(smoke_path)},
            release_sha256,
        ),
        authenticate_v6_1_passing_smoke=lambda: (report, _sha256(smoke_path)),
        MPS_SMOKE_REPORT=str(smoke_path),
    )
    monkeypatch.setattr(train.importlib, "import_module", lambda _name: module)

    result = _v6_1_smoke_authentication()

    assert result["passed"] is True
    assert result["report_sha256"] == _sha256(smoke_path)
    assert result["release_sha256"] == release_sha256
    module.authenticate_v6_1_passing_smoke = lambda: report
    with pytest.raises(TypeError, match="return contract"):
        _v6_1_smoke_authentication()
    module.authenticate_v6_1_passing_smoke = lambda: (report, "0" * 64)
    with pytest.raises(ValueError, match="wrong digest"):
        _v6_1_smoke_authentication()


def test_v6_1_schedule_consumes_all_576_rows_once() -> None:
    rows = load_training_records()
    schedule = build_v6_schedule(rows)
    keys = [
        (row.scene_id, row.question_id)
        for update in schedule
        for row in (*update.contrastive, *update.broad)
    ]

    assert len(schedule) == 96
    assert {len(update.contrastive) for update in schedule} == {3}
    assert {len(update.broad) for update in schedule} == {3}
    assert len(keys) == len(set(keys)) == 576
    assert len(answer_varying_wrong_prefixes(rows)) == 288
    validation = load_validation_records()
    assert len(validation) == 384
    assert len(answer_varying_wrong_prefixes(validation)) == 170


def test_v6_1_rowwise_objective_is_the_exact_fixed_weighted_sum() -> None:
    correct = torch.tensor(1.0, requires_grad=True)
    wrong = torch.tensor(1.2, requires_grad=True)
    broad = torch.tensor(2.0, requires_grad=True)
    retention = torch.tensor(0.1, requires_grad=True)

    pair, diagnostics = contrastive_row_objective(correct, wrong)
    total = pair + broad_row_objective(broad) + retention_objective(retention)
    expected = (0.5 / 3.0) * correct + (4.0 / 3.0) * torch.relu(
        torch.tensor(0.5) - (wrong - correct)
    ) + (0.5 / 3.0) * broad + 0.5 * retention

    assert torch.equal(diagnostics["margin"], wrong - correct)
    assert torch.equal(total, expected)
    total.backward()
    assert correct.grad == pytest.approx(1.5)
    assert wrong.grad == pytest.approx(-4.0 / 3.0)
    assert broad.grad == pytest.approx(0.5 / 3.0)
    assert retention.grad == pytest.approx(0.5)


def test_v6_1_optimizer_contract_pins_every_adamw_choice() -> None:
    settings = optimizer_kwargs()
    assert settings == {
        "lr": 1.25e-5,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "weight_decay": 0.0,
        "amsgrad": False,
        "maximize": False,
        "foreach": False,
        "capturable": False,
        "differentiable": False,
        "fused": False,
    }
    parameter = nn.Parameter(torch.ones(1))
    optimizer = torch.optim.AdamW((parameter,), **settings)
    group = optimizer.param_groups[0]
    assert group["betas"] == (0.9, 0.999)
    assert group["eps"] == 1e-8
    assert group["weight_decay"] == 0.0
    assert group["amsgrad"] is False
    assert group["maximize"] is False
    assert group["foreach"] is False
    assert group["capturable"] is False
    assert group["differentiable"] is False
    assert group["fused"] is False


def test_v6_1_all_teacher_family_scope_and_retention_gates_are_required() -> None:
    baseline = _gate_metrics(0.50, answer_nll=3.0, complete=5)
    candidate = _gate_metrics(0.65, answer_nll=2.96, complete=8)
    retention = {
        "mean_ce_increase_nats": 0.03,
        "mean_kl_nats": 0.02,
        "next_token_top1_agreement": 0.98,
    }

    checks = teacher_and_retention_checks(baseline, candidate, retention)

    assert len(checks) == 17
    assert all(checks.values())
    candidate["family_positive_margin_rates"] = {
        **candidate["family_positive_margin_rates"],
        "attribute": 0.49,
    }
    checks = teacher_and_retention_checks(baseline, candidate, retention)
    assert checks["every_family_positive_margin_rate"] is False
    with pytest.raises(ValueError, match="strata"):
        teacher_and_retention_checks(
            baseline,
            {**candidate, "scope_positive_margin_rates": {"cross_pair": 0.7}},
            retention,
        )


def test_v6_1_teacher_evaluator_uses_every_locked_stratum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = load_validation_records()
    scene_ids = sorted({row.scene_id for row in rows})
    bundle = SimpleNamespace(
        installation=SimpleNamespace(eval=lambda: None),
        prefixes={scene_id: torch.tensor([index]) for index, scene_id in enumerate(scene_ids)},
    )
    monkeypatch.setattr(train, "answer_nll", lambda *_args: torch.tensor(1.0))

    metrics = train.evaluate_teacher_forcing_v6_1(bundle, rows)

    assert metrics["answer_nll_count"] == 384
    assert metrics["curated_side_count"] == 52
    assert metrics["curated_unit_count"] == 26
    assert metrics["expanded_side_count"] == 170
    assert metrics["family_counts"] == train._EXPECTED_FAMILIES
    assert metrics["scope_counts"] == train._EXPECTED_SCOPES
    assert metrics["expanded_positive_margin_rate"] == 0.0
    assert metrics["family_macro_positive_margin_rate"] == 0.0
    assert metrics["scope_macro_positive_margin_rate"] == 0.0


def test_v6_1_training_source_has_custom_loader_exact_order_and_delayed_greedy() -> None:
    source = Path(
        "src/semantic_3d_chat/training/train_fixed_prefix_decoder_reader_v6_1.py"
    ).read_text(encoding="utf-8")

    assert "v1._load_bundle" not in source
    assert "gradient_checkpointing=True" not in source
    assert "validate_decoder_reader_surface_v6(runtime.language.model)" in source
    assert "initialize_lora_adapter_state(installation, seed=INITIALIZATION_SEED)" in source
    assert "installation.parameter_count != LORA_PARAMETER_COUNT" in source
    baseline_position = source.index("baseline_teacher = evaluate_teacher_forcing_v6_1")
    optimizer_position = source.index("optimizer = torch.optim.AdamW")
    assert baseline_position < optimizer_position
    assert source.index("for row in update.contrastive:") < source.index(
        "for row in update.broad:"
    ) < source.index("retention_index =")
    assert "if all(checks.values()):\n        greedy = v1.evaluate_greedy" in source
    assert '"intermediate_selection_or_checkpoint": False' in source


def test_v6_1_training_forbidden_roots_cover_both_held_out_sets() -> None:
    roots = {str(path) for path in train.training_forbidden_roots()}
    for scene_id in ("scene_000025", "scene_000030", "scene_000057", "scene_000062"):
        assert str(Path("data/oracle", scene_id).resolve()) in roots
        assert str(Path("data_gemma4/maps", scene_id).resolve()) in roots
        assert str(Path("data_gemma4/features", scene_id).resolve()) in roots


def test_v6_1_training_release_is_create_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    destination = tmp_path / "release.json"
    payload = {
        "schema_version": 1,
        "artifact": "test",
        "status": "released_exactly_one_fixed_96_update_training_run",
    }
    monkeypatch.setattr(train, "build_training_release", lambda: payload)

    path, digest = train.write_training_release(destination)

    assert path == destination.resolve()
    assert digest == _sha256(destination)
    assert json.loads(destination.read_text(encoding="utf-8")) == payload
    with pytest.raises(FileExistsError, match="create-once"):
        train.write_training_release(destination)


class _Pair(nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.lora_a = nn.Parameter(torch.full((2, 3), value))
        self.lora_b = nn.Parameter(torch.full((4, 2), value + 1.0))


class _State(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.adapters = nn.ModuleList((_Pair(1.0), _Pair(2.0)))


def test_v6_1_checkpoint_contains_only_the_fresh_v6_bank_and_is_atomic(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    state = _State()
    installation = SimpleNamespace(
        state_module=state,
        state_sha256=lambda: "a" * 64,
    )
    bundle = SimpleNamespace(installation=installation)
    monkeypatch.setattr(train, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(train, "OUTPUT_CHECKPOINT", "checkpoint")

    result = _publish_checkpoint(
        bundle,
        {"passed": True},
        training_release_sha256="b" * 64,
        parent_smoke_sha256="c" * 64,
    )

    checkpoint = tmp_path / "checkpoint"
    assert result["path"] == "checkpoint"
    assert sorted(path.name for path in checkpoint.iterdir()) == [
        "adapter.safetensors",
        "runtime_metadata.json",
    ]
    with safe_open(checkpoint / "adapter.safetensors", framework="pt") as handle:
        assert set(handle.keys()) == {
            "adapters.0.lora_a",
            "adapters.0.lora_b",
            "adapters.1.lora_a",
            "adapters.1.lora_b",
        }
    metadata = json.loads(
        (checkpoint / "runtime_metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["adapter_type"] == "fresh_v6_only_upper_decoder_lora"
    assert metadata["target_modules"] == list(TARGET_MODULES)
    assert metadata["trainable_parameter_count"] == LORA_PARAMETER_COUNT
    assert metadata["environmental_text_inputs"] == []
    assert metadata["oracle_runtime_access"] is False
    with pytest.raises(FileExistsError, match="already exists"):
        _publish_checkpoint(
            bundle,
            {"passed": True},
            training_release_sha256="b" * 64,
            parent_smoke_sha256="c" * 64,
        )


def test_v6_1_exception_consumes_attempt_but_publishes_no_checkpoint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(train, "TRAINING_ATTEMPT", str(tmp_path / "attempt.json"))
    monkeypatch.setattr(train, "RESULT_REPORT", str(tmp_path / "result.json"))
    monkeypatch.setattr(train, "FILE_AUDIT_REPORT", str(tmp_path / "audit.json"))
    monkeypatch.setattr(train, "OUTPUT_CHECKPOINT", str(tmp_path / "checkpoint"))
    monkeypatch.setattr(train, "training_forbidden_roots", list)
    monkeypatch.setattr(
        train,
        "authenticate_training_release",
        lambda: {"passed": True, "sha256": "d" * 64, "parent_smoke_sha256": "e" * 64},
    )
    monkeypatch.setattr(
        train,
        "_execute_training",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("locked training failure")),
    )

    with pytest.raises(RuntimeError, match="locked training failure"):
        train.train_and_gate()

    result_path = tmp_path / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    audit = json.loads((tmp_path / "audit.json").read_text(encoding="utf-8"))
    assert (tmp_path / "attempt.json").is_file()
    assert result["status"] == "failed_terminal_attempt_consumed_no_checkpoint"
    assert result["checkpoint_published"] is False
    assert result["training_attempt_sha256"] == _sha256(tmp_path / "attempt.json")
    assert audit["passed"] is True
    assert audit["forbidden_accesses"] == []
    assert not (tmp_path / "checkpoint").exists()
    authenticated = train.authenticate_result()
    assert authenticated["passed"] is False
    assert authenticated["checkpoint_exists"] is False
