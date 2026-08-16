"""Seal V33's stopped update-64 failure without loading Gemma or scene data.

This audit is intentionally limited to immutable configuration/report bytes and
checkpoint metadata/tensors.  It turns V33's causal early-stop artifact into a
small, hash-pinnable authorization record for V34; it is not a model selector
and can never authorize chat or final-test access.
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
from semantic_3d_chat.training.train_environmental_sidecar_v33 import (
    _TRAINABLE_NAMES,
    _optimizer_checkpoint_step,
    assert_deferred_final_scenes_absent,
    require_v32_rejection,
    v33_contract,
    v33_settings,
)
from semantic_3d_chat.training.train_joint_pair_v30 import require_approved_v29_source

DEFAULT_CONFIG = Path("configs/experiments/gemma4_diverse28_environmental_sidecar_v33.yaml")
DEFAULT_CHECKPOINT_ROOT = Path(
    "data_gemma4/checkpoints/gemma4_v33_diverse28_environmental_sidecar"
)
DEFAULT_OUTPUT = Path("reports/gemma4/metrics/v33_update64_terminal_gate.json")

_CONFIG_SHA256 = "e920d28da8ab0abc3c0ab2c4ad812a2743d1894b769c6302097ac41c31da3905"
_V32_REPORT_SHA256 = "2ffeb2655cd6a8627ea9e06c8f261113b0b225a1b39de4eb32126693063c13b7"
_SCHEDULE_SHA256 = "90b7c3b337f573b47a75ed3faefc915eacd98c9ef11b572ff3c45c4166fc9590"
_UPDATE64_FILE_SHA256 = {
    "adapter.safetensors": "32c071d7acca0e52f8ae4c3dee8cba83319d67b184bbb3ab9957a6f6c4fcf987",
    TRAINING_METADATA_FILENAME: "ef97dfc3415eb4cfbdf30fe952e85db5ea4c54e4dec896a40725fb41fd787c91",
    RUNTIME_METADATA_FILENAME: "fe8df1c8c052ac50899eb19952f96b74ac691780e20b604ba4e11072db32e168",
    "optimizer.pt": "845aa42380b5c8c575162cb003fcadc7761fd615071b9dab71d9da4a85ba3d09",
}
_EXPECTED_SAVED_STEPS = tuple(range(0, 65, 8))
_EXPECTED_GATE_VALUES = {
    "book_update0_mean_margin": 0.06537121534347534,
    "book_update64_mean_margin": -0.006955236196517944,
    "picture_update0_mean_margin": -0.0546875,
    "picture_update64_mean_margin": -0.177734375,
    "weak_pair_prefix_rms_ratio": 1.000267116920406,
    "unrelated_prefix_rms_ratio": 1.000552564097064,
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
    descriptor, temporary = tempfile.mkstemp(prefix=f"{path.name}.", suffix=".tmp", dir=path.parent)
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
        raise FileNotFoundError(f"V33 checkpoint root must be a real directory: {root}")
    observed = sorted(
        path.name for path in root.iterdir() if path.is_dir() and path.name.startswith("update_")
    )
    expected = [f"update_{step:03d}" for step in _EXPECTED_SAVED_STEPS]
    if observed != expected:
        raise ValueError(
            "V33 must be stopped at its contiguous update-64 early gate: "
            f"observed={observed} expected={expected}"
        )
    paths = tuple(root / name for name in expected)
    for step, path in zip(_EXPECTED_SAVED_STEPS, paths, strict=True):
        if path.is_symlink():
            raise ValueError(f"V33 saved checkpoint must not be a symlink: {path}")
        required = ["adapter.safetensors", TRAINING_METADATA_FILENAME, RUNTIME_METADATA_FILENAME]
        if step:
            required.append("optimizer.pt")
        for filename in required:
            _real_file(path / filename, f"V33 update {step} {filename}")
    return paths


def _validate_tensor_transition(update0: Path, update64: Path) -> dict[str, Any]:
    initial = load_file(update0 / "adapter.safetensors", device="cpu")
    terminal = load_file(update64 / "adapter.safetensors", device="cpu")
    if set(initial) != set(terminal):
        raise ValueError("V33 update 0/64 tensor names differ")
    changed = sorted(name for name in initial if not torch.equal(initial[name], terminal[name]))
    authorized = sorted(f"dense_sidecar_adapter.{name}" for name in _TRAINABLE_NAMES)
    if changed != authorized:
        raise ValueError(
            "V33 update 0/64 changed tensors differ from its exact eight-tensor surface: "
            f"observed={changed} expected={authorized}"
        )
    frozen = {name: terminal[name] for name in terminal if name not in set(authorized)}
    return {
        "changed_tensor_names": changed,
        "changed_tensor_count": len(changed),
        "changed_parameter_count": sum(int(terminal[name].numel()) for name in changed),
        "frozen_tensor_count": len(frozen),
        "terminal_full_tensor_state_sha256": tensor_state_sha256(terminal),
        "terminal_frozen_tensor_state_sha256": tensor_state_sha256(frozen),
    }


def audit_v33_update64(
    config_path: Path = DEFAULT_CONFIG,
    checkpoint_root: Path = DEFAULT_CHECKPOINT_ROOT,
) -> dict[str, Any]:
    """Return V33's immutable terminal failure evidence or fail closed."""

    config_file = _resolve(config_path)
    root = _resolve(checkpoint_root)
    _real_file(config_file, "V33 config")
    observed_config_sha = _sha256(config_file)
    if observed_config_sha != _CONFIG_SHA256:
        raise ValueError("V33 config bytes differ from the stopped update-64 experiment")
    config = load_config(config_file)
    v33_contract(config)
    condition = require_v32_rejection(config)
    if condition.get("report_sha256") != _V32_REPORT_SHA256:
        raise ValueError("V33 terminal audit lacks the exact V32 rejection report")
    assert_deferred_final_scenes_absent(config)
    source = require_approved_v29_source(config)
    paths = _validate_checkpoint_sequence(root)
    update0, update64 = paths[0], paths[-1]

    observed_hashes: dict[str, str] = {}
    for filename, expected in _UPDATE64_FILE_SHA256.items():
        candidate = update64 / filename
        _real_file(candidate, f"V33 update-64 {filename}")
        observed = _sha256(candidate)
        if observed != expected:
            raise ValueError(
                f"V33 update-64 {filename} hash changed: expected={expected} observed={observed}"
            )
        observed_hashes[filename] = observed

    metadata = json.loads((update64 / TRAINING_METADATA_FILENAME).read_text(encoding="utf-8"))
    metadata = _mapping(metadata, "V33 update-64 metadata")
    runtime = json.loads((update64 / RUNTIME_METADATA_FILENAME).read_text(encoding="utf-8"))
    runtime = _mapping(runtime, "V33 update-64 runtime metadata")
    validate_runtime_checkpoint_metadata(runtime)
    if runtime != runtime_checkpoint_metadata(metadata):
        raise ValueError("V33 runtime metadata is not the exact sanitized training metadata")
    if metadata.get("config_hash") != config_hash(config):
        raise ValueError("V33 update-64 metadata config hash changed")
    if metadata.get("optimizer_step") != 64:
        raise ValueError("V33 terminal checkpoint is not optimizer update 64")
    history = metadata.get("history")
    if not isinstance(history, list) or len(history) != 65:
        raise ValueError("V33 terminal history must contain update zero plus 64 true updates")
    if [row.get("optimizer_update") for row in history if isinstance(row, Mapping)] != list(
        range(65)
    ):
        raise ValueError("V33 terminal history is not contiguous through update 64")
    if any(
        not isinstance(row, Mapping) or row.get("true_optimizer_step") is not True
        for row in history[1:]
    ):
        raise ValueError("V33 terminal history does not prove 64 true optimizer steps")

    stage = _mapping(metadata.get("v33_environmental"), "metadata.v33_environmental")
    legacy = _mapping(metadata.get("v30_joint_pair"), "metadata.v30_joint_pair")
    if (
        stage.get("artifact") != "v33_diverse28_true_environmental_training"
        or stage.get("optimizer_step") != 64
        or stage.get("source_is_approved_v29_update_004") is not True
        or stage.get("gemma_decoder_frozen") is not True
        or stage.get("all_lora_banks_frozen") is not True
        or stage.get("base_norm_and_projection_frozen") is not True
        or stage.get("deferred_final_scene_ids_loaded") != []
        or stage.get("oracle_environment_files_loaded") is not False
        or stage.get("question_dependent_scene_processing") is not False
        or stage.get("question_dependent_retrieval") is not False
    ):
        raise ValueError("V33 terminal metadata violates its locked causal/data boundary")
    schedule = _mapping(stage.get("schedule"), "v33.schedule")
    if schedule.get("schedule_sha256") != _SCHEDULE_SHA256:
        raise ValueError("V33 schedule hash changed")
    if not (
        schedule.get("optimizer_step_count") == 100
        and schedule.get("pair_unit_count") == 25
        and schedule.get("pair_unit_minimum_recurrence") == 4
        and schedule.get("pair_unit_maximum_recurrence") == 4
        and schedule.get("pair_units_atomic") is True
        and schedule.get("true_optimizer_step_per_schedule_row") is True
    ):
        raise ValueError("V33 schedule contract changed")
    if (
        legacy.get("source_v29_checkpoint") != str(source.checkpoint)
        or legacy.get("source_v29_selection_report") != str(source.selection_report)
        or legacy.get("source_v29_selection_report_sha256") != source.selection_sha256
        or legacy.get("source_v29_selected_update") != 4
        or legacy.get("source_v29_adapter_sha256")
        != config["v30_joint_pair"]["source_adapter_sha256"]
        or legacy.get("source_v29_runtime_metadata_sha256")
        != config["v30_joint_pair"]["source_runtime_metadata_sha256"]
    ):
        raise ValueError("V33 terminal metadata differs from its exact approved V29 source")

    baseline = _mapping(history[0].get("validation_family_teacher_metrics"), "history[0] family")
    terminal = _mapping(history[64].get("validation_family_teacher_metrics"), "history[64] family")
    ratios = _mapping(
        history[64].get("adapted_prefix_separation_ratios_from_update0"),
        "history[64] prefix ratios",
    )
    book0 = _exact_float(
        _mapping(baseline.get("book_support"), "book baseline").get("mean_margin"),
        _EXPECTED_GATE_VALUES["book_update0_mean_margin"],
        "book update-0 mean margin",
    )
    book64 = _exact_float(
        _mapping(terminal.get("book_support"), "book terminal").get("mean_margin"),
        _EXPECTED_GATE_VALUES["book_update64_mean_margin"],
        "book update-64 mean margin",
    )
    picture0 = _exact_float(
        _mapping(baseline.get("picture_support"), "picture baseline").get("mean_margin"),
        _EXPECTED_GATE_VALUES["picture_update0_mean_margin"],
        "picture update-0 mean margin",
    )
    picture64 = _exact_float(
        _mapping(terminal.get("picture_support"), "picture terminal").get("mean_margin"),
        _EXPECTED_GATE_VALUES["picture_update64_mean_margin"],
        "picture update-64 mean margin",
    )
    weak_ratio = _exact_float(
        ratios.get("weak_pair_mean"),
        _EXPECTED_GATE_VALUES["weak_pair_prefix_rms_ratio"],
        "weak-pair prefix RMS ratio",
    )
    unrelated_ratio = _exact_float(
        ratios.get("unrelated_mean"),
        _EXPECTED_GATE_VALUES["unrelated_prefix_rms_ratio"],
        "unrelated prefix RMS ratio",
    )
    nonmirror_complete = sum(
        int(_mapping(terminal.get(family), family).get("complete_units", -1))
        for family in ("book_support", "picture_support")
    )
    if nonmirror_complete != 0:
        raise ValueError("V33 update-64 non-mirror completion evidence changed")
    gate_checks = {
        "nonmirror_teacher_complete": nonmirror_complete >= 1,
        "book_advantage_positive": book64 > book0,
        "picture_advantage_positive": picture64 > picture0,
        "weak_pair_prefix_rms_improved_25pct": weak_ratio >= 1.25,
        "unrelated_prefix_rms_inflation_bounded": unrelated_ratio <= 1.25,
    }
    # The unrelated bound passes, but it is a safety constraint rather than
    # evidence that the environmental early gate passed.
    if gate_checks != {
        "nonmirror_teacher_complete": False,
        "book_advantage_positive": False,
        "picture_advantage_positive": False,
        "weak_pair_prefix_rms_improved_25pct": False,
        "unrelated_prefix_rms_inflation_bounded": True,
    }:
        raise ValueError("V33 update-64 gate outcome differs from the stopped run")
    _optimizer_checkpoint_step(update64, 64, v33_settings(config))
    transition = _validate_tensor_transition(update0, update64)
    if transition["changed_parameter_count"] != 404_608:
        raise ValueError("V33 terminal tensor transition parameter count changed")

    return {
        "schema_version": 1,
        "artifact": "v33_update64_terminal_gate",
        "audit_method": "checkpoint_metadata_and_tensors_only",
        "gemma_loaded": False,
        "scene_maps_loaded": False,
        "qa_loaded": False,
        "oracle_loaded": False,
        "final_test_scenes_touched": False,
        "config": {"path": _relative(config_file), "sha256": observed_config_sha},
        "v32_rejection_report_sha256": condition["report_sha256"],
        "v29_source": {
            "checkpoint": _relative(source.checkpoint),
            "selection_report": _relative(source.selection_report),
            "selection_report_sha256": source.selection_sha256,
            "selected_update": source.selected_update,
            "adapter_sha256": legacy["source_v29_adapter_sha256"],
            "runtime_metadata_sha256": legacy["source_v29_runtime_metadata_sha256"],
        },
        "checkpoint_root": _relative(root),
        "observed_saved_optimizer_steps": list(_EXPECTED_SAVED_STEPS),
        "stopped_at_optimizer_step": 64,
        "no_update_072_or_later": True,
        "update64_checkpoint": _relative(update64),
        "update64_file_sha256": observed_hashes,
        "schedule_sha256": schedule["schedule_sha256"],
        "true_optimizer_steps_completed": 64,
        "tensor_transition": transition,
        "update64_gate_evidence": {
            "book_update0_mean_margin": book0,
            "book_update64_mean_margin": book64,
            "picture_update0_mean_margin": picture0,
            "picture_update64_mean_margin": picture64,
            "nonmirror_teacher_complete_units": nonmirror_complete,
            "weak_pair_prefix_rms_ratio": weak_ratio,
            "unrelated_prefix_rms_ratio": unrelated_ratio,
            "checks": gate_checks,
            "passed": False,
        },
        "v33_development_selection_passed": False,
        "v33_chat_promotion_eligible": False,
        "conditional_v34_base_surface_authorized": True,
        "passed": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = audit_v33_update64(args.config, args.checkpoint_root)
    _atomic_json(_resolve(args.output), report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["audit_v33_update64"]
