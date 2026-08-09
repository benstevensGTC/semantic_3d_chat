from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from semantic_3d_chat.config import config_hash, load_config
from semantic_3d_chat.evaluation import v23_shared_kv_controller as controller


def _clean_provenance() -> dict[str, object]:
    return {
        "schema_version": 1,
        "scope": "repository_excluding_generated_artifacts_v1",
        "available": True,
        "is_clean": True,
        "head_commit": "a" * 40,
        "head_tree": "b" * 40,
        "tracked_diff_sha256": hashlib.sha256(b"").hexdigest(),
    }


def _optimizer_group() -> dict[str, object]:
    return {
        "name": "language_lora",
        "lr": 0.0003,
        "weight_decay": 0.0,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "amsgrad": False,
        "maximize": False,
        "foreach": False,
        "capturable": False,
        "differentiable": False,
        "fused": False,
        "decoupled_weight_decay": True,
        "params": list(range(8)),
    }


def _optimizer_state_dict() -> dict[str, object]:
    states: dict[int, dict[str, torch.Tensor]] = {}
    for parameter_id, shape in enumerate(controller.EXPECTED_PARAMETER_SHAPES):
        is_a = parameter_id % 2 == 0
        states[parameter_id] = {
            "step": torch.tensor(1.0, dtype=torch.float32),
            "exp_avg": torch.zeros(shape, dtype=torch.float32)
            if is_a
            else torch.ones(shape, dtype=torch.float32),
            "exp_avg_sq": torch.zeros(shape, dtype=torch.float32)
            if is_a
            else torch.full(shape, 0.5, dtype=torch.float32),
        }
    return {"state": states, "param_groups": [_optimizer_group()]}


def _write_optimizer(path: Path, state: dict[str, object]) -> Path:
    torch.save(state, path)
    return path


def _pair_metrics(
    *,
    sides: int,
    units: int,
    mean_full_vocab: float,
    minimum_full_vocab: float,
    mean_candidate: float | None = None,
    minimum_candidate: float | None = None,
) -> dict[str, float | int]:
    return {
        "full_vocab_sides": sides,
        "full_vocab_units": units,
        "mean_candidate_margin": (mean_full_vocab if mean_candidate is None else mean_candidate),
        "minimum_candidate_margin": (
            minimum_full_vocab if minimum_candidate is None else minimum_candidate
        ),
        "mean_full_vocab_margin": mean_full_vocab,
        "minimum_full_vocab_margin": minimum_full_vocab,
    }


def _epoch(
    epoch: int,
    mirror: dict[str, float | int],
    *,
    color: dict[str, float | int] | None = None,
) -> dict[str, object]:
    bank_sha256 = hashlib.sha256(f"bank-{epoch}".encode()).hexdigest()
    return {
        "epoch": epoch,
        "optimizer_step": epoch,
        "cumulative_microsteps": epoch * 12,
        "metadata_path": (
            f"data_gemma4/checkpoints/{controller.PRIMARY_NAMESPACE}/"
            f"epoch_{epoch:03d}/metadata.json"
        ),
        "metadata_sha256": f"metadata-{epoch}",
        "adapter_sha256": f"adapter-{epoch}",
        "optimizer_sha256": f"optimizer-{epoch}",
        "new_bank_state_sha256": bank_sha256,
        "recomputed_payload_hashes": {
            "scene_state_sha256": "1" * 64,
            "global_scene_residual_state_sha256": "2" * 64,
            "signed_x_scene_residual_state_sha256": "3" * 64,
            "lora_bank_state_sha256": {
                "inherited_v12": "4" * 64,
                "extension_v13": "5" * 64,
                controller.NEW_BANK: bank_sha256,
            },
            "tensor_count": 64,
        },
        "optimizer_manifest": {
            "optimizer": "AdamW",
            "expected_step": epoch,
        },
        "color": color
        or _pair_metrics(
            sides=12,
            units=6,
            mean_full_vocab=1.0,
            minimum_full_vocab=0.25,
        ),
        "mirror": mirror,
    }


def _update1_report(path: Path, epoch1: dict[str, object]) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "audit_type": "v23_shared_kv_update1_verifier",
                "match": True,
                "stage_2_authorized": True,
                "report_only": True,
                "model_loaded": False,
                "oracle_loaded": False,
                "preflight_sha256": "6" * 64,
                "config_sha256": controller.EXPECTED_CONFIG_SHA256,
                "contract_sha256": controller.EXPECTED_CONTRACT_SHA256,
                "checkpoint": (f"data_gemma4/checkpoints/{controller.PRIMARY_NAMESPACE}/epoch_001"),
                "checkpoint_artifact_hashes": {
                    "adapter_sha256": epoch1["adapter_sha256"],
                    "metadata_sha256": epoch1["metadata_sha256"],
                    "optimizer_sha256": epoch1["optimizer_sha256"],
                },
                "new_bank_state_sha256": epoch1["new_bank_state_sha256"],
                "ordered_parameter_shapes": [
                    list(shape) for shape in controller.EXPECTED_PARAMETER_SHAPES
                ],
                "a_tensors_unchanged": True,
                "b_tensors_all_changed": True,
                "optimizer_manifest": epoch1["optimizer_manifest"],
                "recomputed_payload_hashes": epoch1["recomputed_payload_hashes"],
                "color": epoch1["color"],
                "mirror": epoch1["mirror"],
                "source_provenance": _clean_provenance(),
            }
        ),
        encoding="utf-8",
    )
    return path


def test_v23_constants_recompute_from_resolved_config() -> None:
    config = load_config(controller.CONFIG_PATH)
    contract = controller.v23_contract(config)

    assert config_hash(config, length=64) == controller.EXPECTED_CONFIG_SHA256
    assert controller._canonical_sha256(contract) == controller.EXPECTED_CONTRACT_SHA256
    assert contract["source_archive_sha256"] == controller.V21_ARCHIVE_SHA256
    assert contract["new_bank_parameter_count"] == 30_720
    assert contract["optimizer"]["learning_rate"] == 3e-4
    assert contract["optimizer"]["adamw"]["learning_rate"] == 3e-4


def test_v23_preflight_accepts_explicit_clean_provenance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    provenance = _clean_provenance()
    monkeypatch.setattr(
        controller,
        "capture_git_source_provenance",
        lambda _root: deepcopy(provenance),
    )

    report = controller.run_preflight(
        controller.CONFIG_PATH,
        tmp_path / "v23_preflight.json",
    )

    assert report["authorized"] is True
    assert report["stage_1_authorized"] is True
    assert report["source_provenance"] == provenance
    assert report["model_loaded"] is False
    assert report["optimizer_constructed"] is False
    assert report["optimizer_step_executed"] is False
    assert report["oracle_loaded"] is False
    assert Path(report["output"]).is_file()


def test_v23_optimizer_manifest_accepts_exact_eight_parameter_state(tmp_path: Path) -> None:
    manifest = controller._optimizer_manifest(
        _write_optimizer(tmp_path / "optimizer.pt", _optimizer_state_dict())
    )

    assert manifest["optimizer"] == "AdamW"
    assert manifest["group"]["params"] == list(range(8))
    assert manifest["group"]["betas"] == [0.9, 0.999]
    assert [record["role"] for record in manifest["parameter_states"]] == [
        "A",
        "B",
        "A",
        "B",
        "A",
        "B",
        "A",
        "B",
    ]
    assert all(
        record["exp_avg_nonzero"] == 0 and record["exp_avg_sq_nonzero"] == 0
        for record in manifest["parameter_states"]
        if record["role"] == "A"
    )
    assert all(
        record["exp_avg_nonzero"] > 0 and record["exp_avg_sq_nonzero"] > 0
        for record in manifest["parameter_states"]
        if record["role"] == "B"
    )


def test_v23_optimizer_manifest_rejects_parameter_reordering(tmp_path: Path) -> None:
    state = _optimizer_state_dict()
    state["param_groups"][0]["params"] = [1, 0, *range(2, 8)]

    with pytest.raises(controller.V23ControlViolation, match="optimizer group contract"):
        controller._optimizer_manifest(_write_optimizer(tmp_path / "optimizer_reordered.pt", state))


def test_v23_optimizer_manifest_rejects_nonzero_lora_a_moment(tmp_path: Path) -> None:
    state = _optimizer_state_dict()
    state["state"][0]["exp_avg"].reshape(-1)[0] = 1.0

    with pytest.raises(controller.V23ControlViolation, match="LoRA-A optimizer moments"):
        controller._optimizer_manifest(_write_optimizer(tmp_path / "optimizer_nonzero_a.pt", state))


def test_v23_pair_metrics_reject_near_perfect_non_empirical_accuracy() -> None:
    pair = {
        "first_answer_token_top1_accuracy": 0.999,
        "first_answer_token_top1_unit_accuracy": 0.999,
        "mean_own_vs_alternate_candidate_logit_margin": 1.0,
        "minimum_own_vs_alternate_candidate_logit_margin": 0.1,
        "mean_first_answer_token_target_vs_best_other_logit_margin": 1.0,
        "minimum_first_answer_token_target_vs_best_other_logit_margin": 0.1,
    }
    metadata = {"pair_candidate_gate": {"by_pair": {"pair_000001": pair}}}

    with pytest.raises(controller.V23ControlViolation, match="exact 12-way empirical fraction"):
        controller._pair_metrics(metadata, "pair_000001")


def test_v23_pair_metrics_accept_historical_fp32_empirical_fractions() -> None:
    pair = {
        "first_answer_token_top1_accuracy": 0.5833333134651184,
        "first_answer_token_top1_unit_accuracy": 0.1666666716337204,
        "mean_own_vs_alternate_candidate_logit_margin": 1.0,
        "minimum_own_vs_alternate_candidate_logit_margin": 0.1,
        "mean_first_answer_token_target_vs_best_other_logit_margin": 1.0,
        "minimum_first_answer_token_target_vs_best_other_logit_margin": 0.1,
    }
    metadata = {"pair_candidate_gate": {"by_pair": {"pair_000001": pair}}}

    metrics = controller._pair_metrics(metadata, "pair_000001")

    assert metrics["full_vocab_sides"] == 7
    assert metrics["full_vocab_units"] == 1


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_v23_pair_metrics_reject_nonfinite_margins(value: float) -> None:
    pair = {
        "first_answer_token_top1_accuracy": 1.0,
        "first_answer_token_top1_unit_accuracy": 1.0,
        "mean_own_vs_alternate_candidate_logit_margin": 1.0,
        "minimum_own_vs_alternate_candidate_logit_margin": value,
        "mean_first_answer_token_target_vs_best_other_logit_margin": 1.0,
        "minimum_first_answer_token_target_vs_best_other_logit_margin": 0.1,
    }
    metadata = {"pair_candidate_gate": {"by_pair": {"pair_000001": pair}}}

    with pytest.raises(controller.V23ControlViolation, match="finite number"):
        controller._pair_metrics(metadata, "pair_000001")


def test_v23_selector_rejects_forged_minimal_update1_report(tmp_path: Path) -> None:
    update1 = tmp_path / "forged_update1.json"
    update1.write_text(
        json.dumps(
            {
                "match": True,
                "stage_2_authorized": True,
                "contract_sha256": controller.EXPECTED_CONTRACT_SHA256,
                "source_provenance": _clean_provenance(),
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(controller.V23ControlViolation, match="update-1 report root keys"):
        controller.select_epochs(
            controller.CONFIG_PATH,
            update1,
            {epoch: Path(f"unused-{epoch}") for epoch in range(1, 5)},
            tmp_path / "selector.json",
        )


def test_v23_selector_ranks_mirror_units_then_sides_and_uses_earlier_tie(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    records = {
        1: _epoch(
            1,
            _pair_metrics(
                sides=10,
                units=2,
                mean_full_vocab=2.0,
                minimum_full_vocab=-0.1,
            ),
        ),
        2: _epoch(
            2,
            _pair_metrics(
                sides=8,
                units=3,
                mean_full_vocab=0.2,
                minimum_full_vocab=-0.8,
            ),
        ),
        3: _epoch(
            3,
            _pair_metrics(
                sides=9,
                units=3,
                mean_full_vocab=0.5,
                minimum_full_vocab=-0.4,
            ),
        ),
        4: _epoch(
            4,
            _pair_metrics(
                sides=9,
                units=3,
                mean_full_vocab=0.5,
                minimum_full_vocab=-0.4,
            ),
        ),
    }
    monkeypatch.setattr(
        controller,
        "_epoch_record",
        lambda _config, epoch, _path, _source: deepcopy(records[epoch]),
    )
    monkeypatch.setattr(
        controller,
        "capture_git_source_provenance",
        lambda _root: deepcopy(_clean_provenance()),
    )
    update1 = _update1_report(tmp_path / "update1.json", records[1])

    report = controller.select_epochs(
        controller.CONFIG_PATH,
        update1,
        {epoch: Path(f"unused-{epoch}") for epoch in range(1, 5)},
        tmp_path / "selector.json",
    )

    assert report["selected_epoch"] == 3
    assert report["continuation_authorized"] is True
    assert report["full_teacher_gate_passed"] is False
    assert report["greedy_audit_authorized"] is False
    assert report["decision"] == "screen_passed_extension_authorized_no_greedy_audit"


def test_v23_selector_authorizes_greedy_only_for_complete_teacher_gate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    records = {
        epoch: _epoch(
            epoch,
            _pair_metrics(
                sides=12 if epoch == 2 else 8,
                units=6 if epoch == 2 else 2,
                mean_full_vocab=1.25 if epoch == 2 else 0.25,
                minimum_full_vocab=0.125 if epoch == 2 else -0.5,
            ),
        )
        for epoch in range(1, 5)
    }
    monkeypatch.setattr(
        controller,
        "_epoch_record",
        lambda _config, epoch, _path, _source: deepcopy(records[epoch]),
    )
    monkeypatch.setattr(
        controller,
        "capture_git_source_provenance",
        lambda _root: deepcopy(_clean_provenance()),
    )
    update1 = _update1_report(tmp_path / "update1_full.json", records[1])

    report = controller.select_epochs(
        controller.CONFIG_PATH,
        update1,
        {epoch: Path(f"unused-{epoch}") for epoch in range(1, 5)},
        tmp_path / "selector_full.json",
    )

    assert report["selected_epoch"] == 2
    assert report["continuation_authorized"] is False
    assert report["full_teacher_gate_passed"] is True
    assert report["greedy_audit_authorized"] is True
    assert report["decision"] == "full_teacher_gate_passed_greedy_audit_authorized"
