"""Seal V34's stopped update-32 failure without loading Gemma or scene data.

The audit reads only immutable configuration/report bytes plus checkpoint
metadata, optimizer state, and adapter tensors.  It proves that V34 stopped at
its causal train-only gate, that only the declared four-tensor base surface
changed, and conditionally authorizes exactly the V35 block cross-residual
experiment.  It cannot authorize chat promotion or held-out final access.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from semantic_3d_chat.config import PROJECT_ROOT, config_hash, load_config
from semantic_3d_chat.language.lora import tensor_state_sha256
from semantic_3d_chat.training.checkpointing import (
    RUNTIME_METADATA_FILENAME,
    TRAINING_METADATA_FILENAME,
    runtime_checkpoint_metadata,
    validate_runtime_checkpoint_metadata,
)
from semantic_3d_chat.training.train_base_surface_v34 import (
    _TRAINABLE_NAMES,
    _optimizer_step_audit,
    assert_deferred_final_scenes_absent,
    require_exact_v33_source,
    require_v33_terminal_gate,
    v34_contract,
    v34_settings,
)

DEFAULT_CONFIG = Path("configs/experiments/gemma4_diverse28_base_surface_v34.yaml")
DEFAULT_CHECKPOINT_ROOT = Path(
    "data_gemma4/checkpoints/gemma4_v34_diverse28_base_surface"
)
DEFAULT_OUTPUT = Path("reports/gemma4/metrics/v34_update32_terminal_gate.json")

_CONFIG_SHA256 = "631d5cee0253efef9060d66bcb66941f3fbcfdae7c38039b80cb88db2d737695"
_V33_TERMINAL_REPORT_SHA256 = (
    "703525975c7a03a9b995c6f950dda92ed2945bd1857008196a1086e2a6c19a49"
)
_SCHEDULE_SHA256 = "48acbf99526e79947166d8b33fcf2c0e4f02982ec55fceb594fb11d091ff4b83"
_SEPARATION_REFERENCE_SHA256 = (
    "c4a444d924ebf6f4591506c4f7dadbf701e6f234c730ec7c9f63c04c236c2e7e"
)
_FROZEN_STATE_SHA256 = "fc47169bdb900ba94ff2c74a24f15ad564849107a9cd0f4c375af2051433ab7b"
_UPDATE0_ADAPTER_SHA256 = (
    "32c071d7acca0e52f8ae4c3dee8cba83319d67b184bbb3ab9957a6f6c4fcf987"
)
_UPDATE32_FILE_SHA256 = {
    "adapter.safetensors": "480a1051c45fce21741ded4d4f41bf915df2c79bb4d70536f3dcf994f6f25131",
    TRAINING_METADATA_FILENAME: "14ba328ab9ac1010b75e40123643e3497c59b2bc1c59bfbe307d05a58cea7719",
    RUNTIME_METADATA_FILENAME: "2503d2c6b31e3151bddb34b9a8681b45ecd58ad62b6fccaebe1785e46c6ee5fe",
    "optimizer.pt": "cc1e8a990ddfc2392fc97650b31ed42081ea09dfd65b9b0af73588cd8e761b46",
}
_EXPECTED_SAVED_STEPS = (0, 8, 16, 24, 32)
_EXPECTED_CHANGED_SELECTIVITY = 1.00003981590271
_EXPECTED_CHANGED_COVERAGE = 0
_EXPECTED_GATE_CHECKS = {
    "at_least_6_of_8_changed_pairs_over_1_02": False,
    "changed_selectivity_geometric_mean_at_least_1_02": False,
    "no_physical_pair_selectivity_below_0_98": True,
    "passed": False,
    "training_scenes_only": True,
    "unrelated_median_two_sided_within_1_02": True,
    "unrelated_p90_abs_log_within_log_1_02": True,
}


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (PROJECT_ROOT / value).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be a mapping")
    return value


def _exact_float(observed: object, expected: float, field: str) -> float:
    if isinstance(observed, bool) or not isinstance(observed, (int, float)):
        raise TypeError(f"{field} must be numeric")
    value = float(observed)
    if not math.isfinite(value) or value != expected:
        raise ValueError(f"{field} changed: expected={expected} observed={value}")
    return value


def _real_file(path: Path, field: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"{field} must be a real file: {path}")


def _relative(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f"{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _validate_checkpoint_sequence(root: Path) -> tuple[Path, ...]:
    if root.is_symlink() or not root.is_dir():
        raise FileNotFoundError(f"V34 checkpoint root must be a real directory: {root}")
    observed = sorted(path.name for path in root.iterdir() if path.name.startswith("update_"))
    expected = [f"update_{step:03d}" for step in _EXPECTED_SAVED_STEPS]
    if observed != expected:
        raise ValueError(
            "V34 must be stopped at its contiguous update-32 early gate: "
            f"observed={observed} expected={expected}"
        )
    paths = tuple(root / name for name in expected)
    for step, path in zip(_EXPECTED_SAVED_STEPS, paths, strict=True):
        if path.is_symlink() or not path.is_dir():
            raise ValueError(f"V34 saved checkpoint must be a real directory: {path}")
        required = [
            "adapter.safetensors",
            TRAINING_METADATA_FILENAME,
            RUNTIME_METADATA_FILENAME,
        ]
        if step:
            required.append("optimizer.pt")
        for filename in required:
            _real_file(path / filename, f"V34 update {step} {filename}")
    return paths


def _validate_saved_metadata(
    paths: tuple[Path, ...], config: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    settings = v34_settings(config)
    expected_config_hash = config_hash(dict(config))
    terminal_metadata: dict[str, Any] | None = None
    terminal_stage: dict[str, Any] | None = None
    for step, path in zip(_EXPECTED_SAVED_STEPS, paths, strict=True):
        metadata_raw = json.loads((path / TRAINING_METADATA_FILENAME).read_text(encoding="utf-8"))
        metadata = dict(_mapping(metadata_raw, f"V34 update-{step} metadata"))
        runtime_raw = json.loads((path / RUNTIME_METADATA_FILENAME).read_text(encoding="utf-8"))
        runtime = _mapping(runtime_raw, f"V34 update-{step} runtime metadata")
        validate_runtime_checkpoint_metadata(runtime)
        if runtime != runtime_checkpoint_metadata(metadata):
            raise ValueError(f"V34 update {step} runtime metadata is not freshly sanitized")
        if metadata.get("config_hash") != expected_config_hash:
            raise ValueError(f"V34 update {step} metadata config hash changed")
        if metadata.get("optimizer_step") != step:
            raise ValueError(f"V34 update {step} metadata optimizer step changed")
        history = metadata.get("history")
        if not isinstance(history, list) or len(history) != step + 1:
            raise ValueError(f"V34 update {step} history must contain update zero through {step}")
        if [row.get("optimizer_update") for row in history if isinstance(row, Mapping)] != list(
            range(step + 1)
        ):
            raise ValueError(f"V34 update {step} history is not contiguous")
        if any(
            not isinstance(row, Mapping) or row.get("true_optimizer_step") is not True
            for row in history[1:]
        ):
            raise ValueError(f"V34 update {step} history does not prove true optimizer steps")

        stage = _mapping(metadata.get("v34_base_surface"), f"V34 update-{step} stage")
        surface = _mapping(stage.get("trainable_surface"), f"V34 update-{step} surface")
        schedule = _mapping(stage.get("schedule"), f"V34 update-{step} schedule")
        legacy = _mapping(metadata.get("v30_joint_pair"), f"V34 update-{step} legacy")
        expected_names = [f"dense_sidecar_adapter.{name}" for name in _TRAINABLE_NAMES]
        if (
            stage.get("artifact") != "v34_diverse28_true_base_surface_training"
            or stage.get("optimizer_step") != step
            or stage.get("exact_trainable_parameter_count") != 199_808
            or stage.get("frozen_state_sha256") != _FROZEN_STATE_SHA256
            or stage.get("gemma_decoder_frozen") is not True
            or stage.get("all_lora_banks_frozen") is not True
            or stage.get("all_v33_learned_tensors_frozen") is not True
            or stage.get("deferred_final_scene_ids_loaded") != []
            or stage.get("oracle_environment_files_loaded") is not False
            or stage.get("question_dependent_scene_processing") is not False
            or stage.get("question_dependent_retrieval") is not False
            or stage.get("development_progress_is_not_chat_promotion") is not True
            or stage.get("separation_uses_training_scenes_only") is not True
            or stage.get("separation_uses_question_or_answer_text") is not False
            or stage.get("separation_uses_oracle_environment_inputs") is not False
            or stage.get("separation_unique_changed_pair_count") != 8
            or stage.get("separation_unrelated_pair_count") != 112
            or stage.get("separation_reference_sha256") != _SEPARATION_REFERENCE_SHA256
            or schedule.get("schedule_sha256") != _SCHEDULE_SHA256
            or surface.get("parameter_names") != expected_names
            or surface.get("total_parameter_count") != 199_808
            or surface.get("gemma_decoder_frozen") is not True
            or surface.get("all_lora_banks_frozen") is not True
            or surface.get("all_v33_learned_tensors_frozen") is not True
            or surface.get("every_other_parameter_frozen") is not True
            or legacy.get("final_test_scene_ids_loaded") != []
            or legacy.get("oracle_environment_files_loaded") is not False
            or legacy.get("question_dependent_scene_processing") is not False
            or legacy.get("question_dependent_retrieval") is not False
        ):
            raise ValueError(f"V34 update {step} violates its frozen/data boundary")
        if stage.get("conditional_v33_terminal_gate") != {
            "path": str(_resolve("reports/gemma4/metrics/v33_update64_terminal_gate.json")),
            "sha256": _V33_TERMINAL_REPORT_SHA256,
        }:
            raise ValueError(f"V34 update {step} V33 terminal provenance changed")
        expected_gate: object = None if step < 32 else _EXPECTED_GATE_CHECKS
        if stage.get("early_training_gate") != expected_gate:
            raise ValueError(f"V34 update {step} early-gate evidence changed")
        if step:
            _optimizer_step_audit(path, step, settings)
        if step == 32:
            terminal_metadata = metadata
            terminal_stage = dict(stage)
    if terminal_metadata is None or terminal_stage is None:
        raise RuntimeError("V34 terminal metadata was not audited")
    return terminal_metadata, terminal_stage


def _validate_tensor_transition(paths: tuple[Path, ...]) -> dict[str, Any]:
    if _sha256(paths[0] / "adapter.safetensors") != _UPDATE0_ADAPTER_SHA256:
        raise ValueError("V34 update zero is not tensor-identical to the exact V33 source")
    initial = load_file(paths[0] / "adapter.safetensors", device="cpu")
    authorized = sorted(f"dense_sidecar_adapter.{name}" for name in _TRAINABLE_NAMES)
    authorized_set = set(authorized)
    frozen_names = sorted(set(initial) - authorized_set)
    if tensor_state_sha256({name: initial[name] for name in frozen_names}) != _FROZEN_STATE_SHA256:
        raise ValueError("V34 update-zero inherited frozen state hash changed")

    terminal: Mapping[str, torch.Tensor] | None = None
    for step, path in zip(_EXPECTED_SAVED_STEPS, paths, strict=True):
        state = load_file(path / "adapter.safetensors", device="cpu")
        if set(state) != set(initial):
            raise ValueError(f"V34 update {step} tensor names differ from update zero")
        changed = sorted(name for name in state if not torch.equal(initial[name], state[name]))
        unauthorized = sorted(set(changed) - authorized_set)
        if unauthorized:
            raise ValueError(f"V34 update {step} changed inherited frozen tensors: {unauthorized}")
        frozen = {name: state[name] for name in frozen_names}
        if tensor_state_sha256(frozen) != _FROZEN_STATE_SHA256:
            raise ValueError(f"V34 update {step} inherited frozen tensor hash changed")
        terminal = state
    if terminal is None:
        raise RuntimeError("V34 tensor transition has no terminal state")
    changed = sorted(name for name in terminal if not torch.equal(initial[name], terminal[name]))
    if changed != authorized:
        raise ValueError(
            "V34 update 0/32 changed tensors differ from its exact four-tensor surface: "
            f"observed={changed} expected={authorized}"
        )
    changed_parameter_count = sum(int(terminal[name].numel()) for name in changed)
    if changed_parameter_count != 199_808:
        raise ValueError("V34 terminal trainable parameter count changed")
    return {
        "changed_tensor_names": changed,
        "changed_tensor_count": len(changed),
        "changed_parameter_count": changed_parameter_count,
        "frozen_tensor_count": len(frozen_names),
        "all_inherited_tensors_frozen_at_every_saved_arm": True,
        "terminal_full_tensor_state_sha256": tensor_state_sha256(terminal),
        "terminal_frozen_tensor_state_sha256": _FROZEN_STATE_SHA256,
    }


def audit_v34_update32(
    config_path: Path = DEFAULT_CONFIG,
    checkpoint_root: Path = DEFAULT_CHECKPOINT_ROOT,
) -> dict[str, Any]:
    """Return V34's immutable terminal failure evidence or fail closed."""

    config_file = _resolve(config_path)
    root = _resolve(checkpoint_root)
    _real_file(config_file, "V34 config")
    observed_config_sha = _sha256(config_file)
    if observed_config_sha != _CONFIG_SHA256:
        raise ValueError("V34 config bytes differ from the stopped update-32 experiment")
    config = load_config(config_file)
    v34_contract(config)
    terminal_v33 = require_v33_terminal_gate(config)
    source, source_metadata = require_exact_v33_source(config)
    assert_deferred_final_scenes_absent(config)
    paths = _validate_checkpoint_sequence(root)
    update32 = paths[-1]

    observed_hashes: dict[str, str] = {}
    for filename, expected in _UPDATE32_FILE_SHA256.items():
        candidate = update32 / filename
        _real_file(candidate, f"V34 update-32 {filename}")
        observed = _sha256(candidate)
        if observed != expected:
            raise ValueError(
                f"V34 update-32 {filename} hash changed: expected={expected} observed={observed}"
            )
        observed_hashes[filename] = observed

    _metadata, stage = _validate_saved_metadata(paths, config)
    separation = _mapping(stage.get("training_separation"), "V34 terminal separation")
    selectivity = _exact_float(
        separation.get("changed_selectivity_ratio_geometric_mean"),
        _EXPECTED_CHANGED_SELECTIVITY,
        "V34 update-32 changed selectivity geometric mean",
    )
    coverage = separation.get("changed_selectivity_over_1_02_count")
    if isinstance(coverage, bool) or coverage != _EXPECTED_CHANGED_COVERAGE:
        raise ValueError(
            "V34 update-32 changed-pair coverage changed: "
            f"expected={_EXPECTED_CHANGED_COVERAGE} observed={coverage}"
        )
    gate = _mapping(stage.get("early_training_gate"), "V34 update-32 early gate")
    if dict(gate) != _EXPECTED_GATE_CHECKS or gate.get("passed") is not False:
        raise ValueError("V34 update-32 causal early-gate outcome changed")
    transition = _validate_tensor_transition(paths)

    return {
        "schema_version": 1,
        "artifact": "v34_update32_terminal_gate",
        "audit_method": "checkpoint_metadata_optimizer_and_tensors_only",
        "gemma_loaded": False,
        "scene_maps_loaded": False,
        "qa_loaded": False,
        "oracle_loaded": False,
        "final_test_scenes_touched": False,
        "config": {"path": _relative(config_file), "sha256": observed_config_sha},
        "v33_terminal_gate": {
            "path": _relative(Path(terminal_v33["path"])),
            "sha256": terminal_v33["sha256"],
        },
        "v33_source": {
            "checkpoint": _relative(source),
            "optimizer_step": source_metadata["optimizer_step"],
        },
        "checkpoint_root": _relative(root),
        "observed_saved_optimizer_steps": list(_EXPECTED_SAVED_STEPS),
        "stopped_at_optimizer_step": 32,
        "no_update_040_or_later": True,
        "update32_checkpoint": _relative(update32),
        "update32_file_sha256": observed_hashes,
        "schedule_sha256": _SCHEDULE_SHA256,
        "separation_reference_sha256": _SEPARATION_REFERENCE_SHA256,
        "true_optimizer_steps_completed": 32,
        "tensor_transition": transition,
        "update32_gate_evidence": {
            "changed_selectivity_ratio_geometric_mean": selectivity,
            "changed_selectivity_over_1_02_count": int(coverage),
            "required_selectivity_ratio": 1.02,
            "required_changed_pair_coverage": 6,
            "checks": dict(gate),
            "training_scenes_only": True,
            "passed": False,
        },
        "v34_development_selection_passed": False,
        "v34_chat_promotion_eligible": False,
        "conditional_authorization": {
            "authorized": True,
            "stage": "v35_block_cross_residual",
            "scope": "exact_zero_block_token_cross_residual_only",
            "all_other_followup_architectures_authorized": False,
            "chat_promotion_authorized": False,
            "final_test_access_authorized": False,
        },
        "conditional_v35_block_cross_residual_authorized": True,
        "passed": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = audit_v34_update32(args.config, args.checkpoint_root)
    _atomic_json(_resolve(args.output), report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["audit_v34_update32"]
