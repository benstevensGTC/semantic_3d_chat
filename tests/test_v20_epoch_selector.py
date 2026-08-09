from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any

import pytest
import torch

from semantic_3d_chat.config import load_config
from semantic_3d_chat.evaluation import v20_epoch_selector as selector

CONFIG_PATH = "configs/experiments/gemma4_color_mirror_signed_x_local_field_v20.yaml"


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _metrics(*, sides: int, units: int, minimum: float, mean: float = 1.0) -> dict[str, Any]:
    return {
        "candidate_sides": sides,
        "candidate_units": units,
        "full_vocab_sides": sides,
        "full_vocab_units": units,
        "mean_candidate_margin": mean,
        "mean_full_vocab_margin": mean,
        "minimum_candidate_margin": minimum,
        "minimum_full_vocab_margin": minimum,
    }


def _install_synthetic_validators(
    monkeypatch: pytest.MonkeyPatch,
    *,
    mirror_by_epoch: dict[int, dict[str, Any]],
    color_by_epoch: dict[int, dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[int, dict[str, Any]]]:
    source = {"synthetic": "clean"}
    screen = selector._expected_screen()
    contract = {"config_hash": "c" * 12, "screen": screen}
    monkeypatch.setattr(selector, "_validate_config", lambda _config: copy.deepcopy(contract))
    monkeypatch.setattr(
        selector,
        "_validate_selection",
        lambda _selection, _contract: {
            "source_provenance": source,
            "selection_sha256": _digest("selection"),
            "pair_selection_sha256": _digest("pair-selection"),
            "pair_membership_sha256": _digest("membership"),
        },
    )
    colors = color_by_epoch or {
        epoch: _metrics(sides=12, units=6, minimum=0.5) for epoch in range(1, 5)
    }

    def validate(
        epoch: int,
        artifact: dict[str, Any],
        _contract: dict[str, Any],
        *,
        path: str,
        artifact_sha256: str,
    ) -> dict[str, Any]:
        return {
            "epoch": epoch,
            "path": path,
            "artifact_sha256": artifact_sha256,
            "source_provenance": source,
            "initialization_provenance": {"source": "v18"},
            "zero_output_equivalence": {"verified": True},
            "signed_x_state_sha256": artifact["state"],
            "history": copy.deepcopy(artifact["history"]),
            "metrics": {"color": colors[epoch], "mirror": mirror_by_epoch[epoch]},
        }

    monkeypatch.setattr(selector, "_validate_epoch_artifact", validate)
    monkeypatch.setattr(selector, "_require_bound_json", lambda *_args, **_kwargs: None)
    history: list[dict[str, Any]] = []
    artifacts: dict[int, dict[str, Any]] = {}
    for epoch in range(1, 5):
        history.append({"epoch": epoch, "loss": float(epoch)})
        artifacts[epoch] = {
            "state": _digest(f"state-{epoch}"),
            "history": copy.deepcopy(history),
        }

    def inspect(
        epoch: int,
        artifact: dict[str, Any],
        *,
        config: dict[str, Any],
        metadata_path: str,
    ) -> dict[str, Any]:
        del config
        return {
            "checkpoint": str(metadata_path).removesuffix("/metadata.json"),
            "checkpoint_artifact_hashes": {
                "adapter_sha256": _digest(f"adapter-{epoch}"),
                "metadata_sha256": selector._canonical_sha256(artifact),
                "optimizer_sha256": _digest(f"optimizer-file-{epoch}"),
            },
            "tensor_evidence": {
                "signed_x_state_sha256": artifact["state"],
                "output_projection_sha256": _digest(f"output-{epoch}"),
                "global_scene_residual_state_sha256": selector.EXPECTED_FROZEN_GLOBAL_RESIDUAL_SHA256,
                "scene_state_sha256": selector.EXPECTED_FROZEN_SCENE_SHA256,
                "lora_bank_state_sha256": selector.EXPECTED_FROZEN_BANKS,
            },
            "optimizer_state_manifest": {"epoch": epoch},
            "optimizer_state_sha256": _digest(f"optimizer-state-{epoch}"),
        }

    monkeypatch.setattr(selector, "_inspect_checkpoint_artifacts", inspect)

    def authorize(_path: str, *, config: dict[str, Any]) -> dict[str, Any]:
        del config
        return {
            "report_path": "v20_update1_match.json",
            "report_sha256": _digest("update1-report"),
            "source_provenance": source,
            "checkpoint": "epoch_001",
            "checkpoint_artifact_hashes": {
                "adapter_sha256": _digest("adapter-1"),
                "metadata_sha256": selector._canonical_sha256(artifacts[1]),
                "optimizer_sha256": _digest("optimizer-file-1"),
            },
            "signed_x_state_sha256": artifacts[1]["state"],
            "output_projection_sha256": _digest("output-1"),
            "frozen_global_scene_residual_state_sha256": selector.EXPECTED_FROZEN_GLOBAL_RESIDUAL_SHA256,
            "frozen_scene_state_sha256": selector.EXPECTED_FROZEN_SCENE_SHA256,
            "frozen_lora_bank_state_sha256": selector.EXPECTED_FROZEN_BANKS,
            "optimizer_state_manifest": {"epoch": 1},
            "optimizer_state_sha256": _digest("optimizer-state-1"),
        }

    monkeypatch.setattr(selector, "_load_update1_authorization", authorize)
    return {}, {}, artifacts


def _summarize(
    config: dict[str, Any],
    selection: dict[str, Any],
    artifacts: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    paths = {epoch: f"epoch_{epoch:03d}/metadata.json" for epoch in selector.EXPECTED_EPOCHS}
    return selector.summarize_v20_epochs(
        config,
        selection,
        artifacts,
        update1_report_path="v20_update1_match.json",
        selection_path="selection.json",
        selection_sha256=_digest("selection-file"),
        epoch_paths=paths,
        epoch_sha256={
            epoch: selector._canonical_sha256(artifact) for epoch, artifact in artifacts.items()
        },
    )


def test_continuation_requires_retention_and_eight_sides_two_units(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mirrors = {
        1: _metrics(sides=6, units=0, minimum=-1.0),
        2: _metrics(sides=8, units=2, minimum=-0.5),
        3: _metrics(sides=7, units=1, minimum=-0.25),
        4: _metrics(sides=6, units=0, minimum=-1.0),
    }
    config, selection, artifacts = _install_synthetic_validators(
        monkeypatch, mirror_by_epoch=mirrors
    )
    report = _summarize(config, selection, artifacts)
    assert report["selected_epoch"] == 2
    assert report["continuation_authorized"] is True
    assert report["full_teacher_gate_passed"] is False
    assert report["greedy_audit_authorized"] is False
    assert report["conditional_max_optimizer_updates"] == 8


def test_full_teacher_gate_is_the_only_greedy_authorization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mirrors = {epoch: _metrics(sides=6, units=0, minimum=-1.0) for epoch in range(1, 5)}
    mirrors[3] = _metrics(sides=12, units=6, minimum=0.125)
    config, selection, artifacts = _install_synthetic_validators(
        monkeypatch, mirror_by_epoch=mirrors
    )
    report = _summarize(config, selection, artifacts)
    assert report["selected_epoch"] == 3
    assert report["full_teacher_gate_passed"] is True
    assert report["greedy_audit_authorized"] is True
    assert report["decision"] == "full_teacher_gate_passed_greedy_audit_allowed"


def test_color_failure_makes_an_epoch_ineligible(monkeypatch: pytest.MonkeyPatch) -> None:
    mirrors = {epoch: _metrics(sides=6, units=0, minimum=-1.0) for epoch in range(1, 5)}
    mirrors[2] = _metrics(sides=12, units=6, minimum=0.25)
    colors = {epoch: _metrics(sides=12, units=6, minimum=0.5) for epoch in range(1, 5)}
    colors[2] = _metrics(sides=11, units=5, minimum=-0.1)
    config, selection, artifacts = _install_synthetic_validators(
        monkeypatch, mirror_by_epoch=mirrors, color_by_epoch=colors
    )
    report = _summarize(config, selection, artifacts)
    assert report["epochs"][1]["color_eligible"] is False
    assert report["selected_epoch"] != 2
    assert report["greedy_audit_authorized"] is False


def test_exact_four_update_set_is_required(monkeypatch: pytest.MonkeyPatch) -> None:
    mirrors = {epoch: _metrics(sides=6, units=0, minimum=-1.0) for epoch in range(1, 5)}
    config, selection, artifacts = _install_synthetic_validators(
        monkeypatch, mirror_by_epoch=mirrors
    )
    artifacts.pop(4)
    with pytest.raises(selector.V20EpochSelectorViolation, match="exactly"):
        _summarize(config, selection, artifacts)


def test_cumulative_history_fork_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    mirrors = {epoch: _metrics(sides=6, units=0, minimum=-1.0) for epoch in range(1, 5)}
    config, selection, artifacts = _install_synthetic_validators(
        monkeypatch, mirror_by_epoch=mirrors
    )
    artifacts[3]["history"][0]["loss"] = 99.0
    with pytest.raises(selector.V20EpochSelectorViolation, match="cumulative history"):
        _summarize(config, selection, artifacts)


def test_repeated_signed_x_state_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    mirrors = {epoch: _metrics(sides=6, units=0, minimum=-1.0) for epoch in range(1, 5)}
    config, selection, artifacts = _install_synthetic_validators(
        monkeypatch, mirror_by_epoch=mirrors
    )
    artifacts[4]["state"] = artifacts[3]["state"]
    with pytest.raises(selector.V20EpochSelectorViolation, match="repeats or rolls back"):
        _summarize(config, selection, artifacts)


def test_direct_selector_invocation_requires_update1_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mirrors = {epoch: _metrics(sides=6, units=0, minimum=-1.0) for epoch in range(1, 5)}
    config, selection, artifacts = _install_synthetic_validators(
        monkeypatch, mirror_by_epoch=mirrors
    )
    with pytest.raises(TypeError, match="update1_report_path"):
        selector.summarize_v20_epochs(config, selection, artifacts)  # type: ignore[call-arg]


def test_missing_and_non_authorizing_update1_reports_fail_closed(tmp_path: Path) -> None:
    config = load_config(CONFIG_PATH)
    with pytest.raises(selector.V20EpochSelectorViolation, match="missing"):
        selector._load_update1_authorization(tmp_path / "missing.json", config=config)
    tampered = tmp_path / "v20_update1_match.json"
    tampered.write_text('{"audit_type":"wrong","match":true}\n', encoding="utf-8")
    with pytest.raises(selector.V20EpochSelectorViolation, match="schema_version"):
        selector._load_update1_authorization(tampered, config=config)


def test_direct_json_binding_rejects_value_that_differs_from_file(tmp_path: Path) -> None:
    path = tmp_path / "metadata.json"
    path.write_text('{"epoch":1}\n', encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(selector.V20EpochSelectorViolation, match="differs"):
        selector._require_bound_json(path, {"epoch": 2}, digest, "epoch metadata")


@pytest.mark.parametrize("case", ["artifact", "state", "optimizer", "source"])
def test_update1_mismatch_cannot_authorize_selection(
    case: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    mirrors = {epoch: _metrics(sides=6, units=0, minimum=-1.0) for epoch in range(1, 5)}
    config, selection, artifacts = _install_synthetic_validators(
        monkeypatch, mirror_by_epoch=mirrors
    )
    original = selector._load_update1_authorization

    def mismatched(path: str, *, config: dict[str, Any]) -> dict[str, Any]:
        evidence = copy.deepcopy(original(path, config=config))
        if case == "artifact":
            evidence["checkpoint_artifact_hashes"]["adapter_sha256"] = _digest("tampered")
        elif case == "state":
            evidence["signed_x_state_sha256"] = _digest("tampered")
        elif case == "optimizer":
            evidence["optimizer_state_sha256"] = _digest("tampered")
        else:
            evidence["source_provenance"] = {"synthetic": "other"}
        return evidence

    monkeypatch.setattr(selector, "_load_update1_authorization", mismatched)
    with pytest.raises(selector.V20EpochSelectorViolation, match="Update-one"):
        _summarize(config, selection, artifacts)


def test_update1_report_rejects_symlink_in_parent_path(tmp_path: Path) -> None:
    config = load_config(CONFIG_PATH)
    real = tmp_path / "real"
    real.mkdir()
    (real / "v20_update1_match.json").write_text("{}\n", encoding="utf-8")
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    with pytest.raises(selector.V20EpochSelectorViolation, match="symbolic link"):
        selector._load_update1_authorization(alias / "v20_update1_match.json", config=config)


def test_checkpoint_optimizer_step_must_equal_epoch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_config(CONFIG_PATH)
    checkpoint = tmp_path / "epoch_001"
    checkpoint.mkdir()
    (checkpoint / "adapter.safetensors").write_bytes(b"synthetic")
    (checkpoint / "metadata.json").write_text("{}\n", encoding="utf-8")
    optimizer_contract = config["training"]["optimizer"]
    parameter = torch.nn.Parameter(torch.zeros(1536, 128, dtype=torch.float32))
    optimizer = torch.optim.AdamW(
        [
            {
                "name": "signed_x_output_projection",
                "params": [parameter],
                "lr": optimizer_contract["learning_rate"],
                "weight_decay": optimizer_contract["weight_decay"],
            }
        ],
        betas=tuple(optimizer_contract["betas"]),
        eps=optimizer_contract["epsilon"],
        foreach=False,
        fused=False,
        capturable=False,
        maximize=False,
        amsgrad=False,
    )
    for _ in range(2):
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()
    torch.save(optimizer.state_dict(), checkpoint / "optimizer.pt")
    monkeypatch.setattr(
        selector,
        "_load_tensor_evidence",
        lambda *_args, **_kwargs: {
            "signed_x_state_sha256": _digest("state"),
            "output_projection_sha256": _digest("output"),
            "global_scene_residual_state_sha256": selector.EXPECTED_FROZEN_GLOBAL_RESIDUAL_SHA256,
            "scene_state_sha256": selector.EXPECTED_FROZEN_SCENE_SHA256,
            "lora_bank_state_sha256": selector.EXPECTED_FROZEN_BANKS,
        },
    )
    with pytest.raises(selector.V20EpochSelectorViolation, match="optimizer violates"):
        selector._inspect_checkpoint_artifacts(
            1,
            {},
            config=config,
            metadata_path=checkpoint / "metadata.json",
        )
