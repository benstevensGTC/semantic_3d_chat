from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any

import pytest
import torch

from semantic_3d_chat.config import load_config
from semantic_3d_chat.evaluation import v21_epoch_selector as selector

CONFIG_PATH = "configs/experiments/gemma4_color_mirror_signed_x_local_field_phase_aware_v21.yaml"
SYNTHETIC_SOURCE = {
    "schema_version": 1,
    "scope": "repository_excluding_generated_artifacts_v1",
    "available": True,
    "head_commit": "a" * 40,
    "head_tree": "b" * 40,
    "is_clean": True,
    "tracked_diff_sha256": hashlib.sha256(b"").hexdigest(),
}


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
    source = copy.deepcopy(SYNTHETIC_SOURCE)
    config_contract = selector._validate_config(load_config(CONFIG_PATH))
    contract = {
        "config_hash": config_contract["config_hash"],
        "config_hash_full": config_contract["config_hash_full"],
        "model_dtype": selector.MODEL_DTYPE,
        "screen": copy.deepcopy(config_contract["screen"]),
        "preflight_contract_sha256": config_contract["preflight_contract_sha256"],
    }
    monkeypatch.setattr(selector, "_validate_config", lambda _config: copy.deepcopy(contract))
    monkeypatch.setattr(
        selector,
        "capture_git_source_provenance",
        lambda _root: copy.deepcopy(source),
    )
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
                "global_scene_residual_state_sha256": (
                    selector.EXPECTED_FROZEN_GLOBAL_RESIDUAL_SHA256
                ),
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
            "report_path": "v21_update1_match.json",
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
            "frozen_global_scene_residual_state_sha256": (
                selector.EXPECTED_FROZEN_GLOBAL_RESIDUAL_SHA256
            ),
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
    return selector.summarize_v21_epochs(
        config,
        selection,
        artifacts,
        update1_report_path="v21_update1_match.json",
        selection_path="selection.json",
        selection_sha256=_digest("selection-file"),
        epoch_paths=paths,
        epoch_sha256={
            epoch: selector._canonical_sha256(artifact) for epoch, artifact in artifacts.items()
        },
    )


def test_config_is_exactly_bfloat16_phase_aware_v21() -> None:
    contract = selector._validate_config(load_config(CONFIG_PATH))
    assert contract["config_hash"] == "ae17da8b9a71"
    assert contract["config_hash_full"] == (
        "ae17da8b9a712e9be89cc7d0f04d6db54bce0c239adf69c3236d848b64d9b04b"
    )
    assert contract["model_dtype"] == "bfloat16"
    assert (
        contract["screen"]["structural_preflight_requires"][
            "legacy_effective_total_norm_selectivity_diagnostic_only"
        ]
        is True
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
    assert report["model_dtype"] == "bfloat16"
    assert report["continuation_authorized"] is True
    assert report["full_teacher_gate_passed"] is False
    assert report["greedy_audit_authorized"] is False


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


def test_exact_zero_color_margin_is_ineligible(monkeypatch: pytest.MonkeyPatch) -> None:
    mirrors = {epoch: _metrics(sides=12, units=6, minimum=0.25) for epoch in range(1, 5)}
    colors = {epoch: _metrics(sides=12, units=6, minimum=0.5) for epoch in range(1, 5)}
    colors[2] = _metrics(sides=12, units=6, minimum=0.0)
    config, selection, artifacts = _install_synthetic_validators(
        monkeypatch,
        mirror_by_epoch=mirrors,
        color_by_epoch=colors,
    )
    report = _summarize(config, selection, artifacts)
    assert report["epochs"][1]["color_eligible"] is False
    assert report["selected_epoch"] != 2


def test_exact_four_updates_unique_states_and_cumulative_history_are_required(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mirrors = {epoch: _metrics(sides=6, units=0, minimum=-1.0) for epoch in range(1, 5)}
    config, selection, artifacts = _install_synthetic_validators(
        monkeypatch, mirror_by_epoch=mirrors
    )
    missing = copy.deepcopy(artifacts)
    missing.pop(4)
    with pytest.raises(selector.V21EpochSelectorViolation, match="exactly"):
        _summarize(config, selection, missing)

    repeated = copy.deepcopy(artifacts)
    repeated[4]["state"] = repeated[3]["state"]
    with pytest.raises(selector.V21EpochSelectorViolation, match="repeats or rolls back"):
        _summarize(config, selection, repeated)

    forked = copy.deepcopy(artifacts)
    forked[3]["history"][0]["loss"] = 99.0
    with pytest.raises(selector.V21EpochSelectorViolation, match="cumulative history"):
        _summarize(config, selection, forked)


@pytest.mark.parametrize("case", ["different_commit", "dirty"])
def test_selector_requires_exact_current_clean_source_provenance(
    monkeypatch: pytest.MonkeyPatch,
    case: str,
) -> None:
    mirrors = {epoch: _metrics(sides=12, units=6, minimum=0.25) for epoch in range(1, 5)}
    config, selection, artifacts = _install_synthetic_validators(
        monkeypatch, mirror_by_epoch=mirrors
    )
    current = copy.deepcopy(SYNTHETIC_SOURCE)
    if case == "different_commit":
        current["head_commit"] = "c" * 40
    else:
        current["is_clean"] = False
    monkeypatch.setattr(
        selector,
        "capture_git_source_provenance",
        lambda _root: copy.deepcopy(current),
    )
    with pytest.raises(selector.V21EpochSelectorViolation, match="current source|Current clean"):
        _summarize(config, selection, artifacts)


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
    with pytest.raises(selector.V21EpochSelectorViolation, match="Update-one"):
        _summarize(config, selection, artifacts)


def _implementation_sources() -> dict[str, Any]:
    result: dict[str, Any] = {}
    for field, relative in selector._IMPLEMENTATION_SOURCES.items():
        result[field] = relative
        result[f"{field}_sha256"] = selector.file_sha256(selector.PROJECT_ROOT / relative)
    return result


def _reduction(sources: dict[str, Any]) -> dict[str, Any]:
    raw = {
        "schema_version": 1,
        "verified": True,
        "model_dtype": "bfloat16",
        "precision_algorithm": "bfloat16_cast_of_fp32_base_plus_fp32_delta",
        "phase_algorithm_family": "phase_aware_precision_pair_v1",
        "phase_algorithm": "phase_aware_bfloat16_pair_v1",
        "legacy_effective_total_norm_selectivity_diagnostic_only": True,
        "preflight_contract_sha256": selector.EXPECTED_V21_CONTRACT_SHA256,
        "scene_ids": list(selector.EXPECTED_TRAIN_SCENES),
        "pair_ids": ["pair_000001", "pair_000003"],
        **{
            key: (
                selector._canonical_sha256(sources)
                if key == "implementation_sources_sha256"
                else _digest(key)
            )
            for key in selector._UPDATE1_REDUCTION_HASH_FIELDS
        },
    }
    return {**raw, "canonical_sha256": selector._canonical_sha256(raw)}


def test_update1_reduction_binds_precision_phase_functional_and_helper_sources() -> None:
    sources = _implementation_sources()
    reduction = _reduction(sources)
    assert (
        selector._validate_rich_preflight_reduction(reduction, implementation_sources=sources)
        == reduction
    )
    for field in (
        "precision_cast_audit_sha256",
        "phase_aware_pair_diagnostics_sha256",
        "predicted_update_functional_audit_sha256",
        "implementation_sources_sha256",
    ):
        tampered = copy.deepcopy(reduction)
        tampered[field] = _digest(f"tampered-{field}")
        with pytest.raises(selector.V21EpochSelectorViolation):
            selector._validate_rich_preflight_reduction(tampered, implementation_sources=sources)


def test_missing_update1_report_and_symlink_fail_before_use(tmp_path: Path) -> None:
    config = load_config(CONFIG_PATH)
    with pytest.raises(selector.V21EpochSelectorViolation, match="missing"):
        selector._load_update1_authorization(tmp_path / "missing.json", config=config)
    real = tmp_path / "real"
    real.mkdir()
    (real / "v21_update1_match.json").write_text("{}\n", encoding="utf-8")
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    with pytest.raises(selector.V21EpochSelectorViolation, match="symbolic link"):
        selector._load_update1_authorization(alias / "v21_update1_match.json", config=config)


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
            "global_scene_residual_state_sha256": (selector.EXPECTED_FROZEN_GLOBAL_RESIDUAL_SHA256),
            "scene_state_sha256": selector.EXPECTED_FROZEN_SCENE_SHA256,
            "lora_bank_state_sha256": selector.EXPECTED_FROZEN_BANKS,
        },
    )
    with pytest.raises(selector.V21EpochSelectorViolation, match="optimizer violates"):
        selector._inspect_checkpoint_artifacts(
            1,
            {},
            config=config,
            metadata_path=checkpoint / "metadata.json",
        )
