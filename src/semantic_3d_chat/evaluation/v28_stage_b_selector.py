"""Select a causally retained V28 Stage-B decoder update.

This is a training/evaluation-only selector.  It loads the complete update-0
runtime once, swaps only the fresh query-LoRA bank, and scores the immutable
color and mirror controls.  No scene region is retrieved from a question and
no evaluation text is copied into a runtime checkpoint.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from semantic_3d_chat.chat.runtime import StaticChatRuntime
from semantic_3d_chat.config import load_config, project_path
from semantic_3d_chat.data.dataset import SceneQADataset
from semantic_3d_chat.evaluation.v27_sidecar_screen import (
    _atomic_json,
    _full_vocab_counts,
    _negative_sides,
    _pair_role_ids,
)
from semantic_3d_chat.evaluation.v28_stage_a_selector import _teacher_gate
from semantic_3d_chat.language.lora import (
    LoRAAdapterState,
    LoRALinear,
    lora_banks_settings,
    tensor_state_sha256,
)
from semantic_3d_chat.scene_encoder.map_io import MapTensorData, load_map_tensors
from semantic_3d_chat.training.checkpointing import (
    RUNTIME_METADATA_FILENAME,
    TRAINING_METADATA_FILENAME,
    runtime_checkpoint_metadata,
    validate_runtime_checkpoint_metadata,
)
from semantic_3d_chat.training.pair_curriculum import (
    build_exact_question_pair_units,
    cap_pair_units_per_pair,
    pair_curriculum_settings,
    select_pair_only_records,
)
from semantic_3d_chat.training.train_post_stack_decoder import (
    v29_development_contract,
)

FRESH_BANK_NAME = "extension_v28_stage_b_query"
FRESH_BANK_PREFIX = f"lora_banks.{FRESH_BANK_NAME}."
_UPDATE_NAME = re.compile(r"update_([0-9]{3})")


def _checkpoint_paths(root: Path, *, expected_final_update: int | None = None) -> list[Path]:
    """Return a complete, gap-free update_000..N inventory."""

    paths = sorted(
        path
        for path in root.glob("update_*")
        if path.is_dir() and _UPDATE_NAME.fullmatch(path.name)
    )
    if not paths or paths[0].name != "update_000":
        raise FileNotFoundError("V28 Stage-B selection requires update_000")
    observed_updates = [int(_UPDATE_NAME.fullmatch(path.name).group(1)) for path in paths]
    final_update = observed_updates[-1]
    expected_updates = list(range(final_update + 1))
    if observed_updates != expected_updates:
        raise FileNotFoundError(
            "V28 Stage-B checkpoints are not contiguous: "
            f"observed={observed_updates} expected={expected_updates}"
        )
    if expected_final_update is not None and final_update != expected_final_update:
        raise FileNotFoundError(
            "V28 Stage-B checkpoint run is incomplete: "
            f"observed_final={final_update} expected_final={expected_final_update}"
        )
    required = (
        "adapter.safetensors",
        TRAINING_METADATA_FILENAME,
        RUNTIME_METADATA_FILENAME,
    )
    for path in paths:
        missing = [name for name in required if not (path / name).is_file()]
        if missing:
            raise FileNotFoundError(f"Incomplete Stage-B checkpoint {path.name}: {missing}")
    return paths


def _metadata(path: Path) -> dict[str, Any]:
    value = json.loads((path / TRAINING_METADATA_FILENAME).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Checkpoint metadata must be a JSON object: {path}")
    return value


def _validate_runtime_metadata(path: Path, training_metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Require the chat payload to be the exact sanitized training state."""

    value = json.loads((path / RUNTIME_METADATA_FILENAME).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Runtime checkpoint metadata must be a JSON object: {path}")
    validate_runtime_checkpoint_metadata(value)
    expected = runtime_checkpoint_metadata(training_metadata)
    if value != expected:
        raise ValueError(f"Runtime/training metadata mismatch in {path.name}")
    return value


def _fresh_bank_state(tensors: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    state = {
        name.removeprefix(FRESH_BANK_PREFIX): value
        for name, value in tensors.items()
        if name.startswith(FRESH_BANK_PREFIX)
    }
    if not state:
        raise ValueError(f"Checkpoint does not contain fresh bank {FRESH_BANK_NAME}")
    return state


def _frozen_tensor_sha256(tensors: Mapping[str, torch.Tensor]) -> str:
    frozen = {
        name: value for name, value in tensors.items() if not name.startswith(FRESH_BANK_PREFIX)
    }
    if not frozen:
        raise ValueError("Stage-B checkpoint has no frozen tensors to audit")
    return tensor_state_sha256(frozen)


def _validation_nll(metadata: Mapping[str, Any]) -> float:
    history = metadata.get("history")
    if not isinstance(history, Sequence) or isinstance(history, (str, bytes)) or not history:
        raise ValueError("Stage-B checkpoint lacks training history")
    final = history[-1]
    if not isinstance(final, Mapping):
        raise TypeError("Stage-B checkpoint history entry must be a mapping")
    value = final.get("validation_answer_token_nll")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("Stage-B checkpoint lacks numeric validation answer NLL")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Stage-B validation NLL is not finite")
    return result


def _fresh_bank_audit(
    tensors: Mapping[str, torch.Tensor],
    metadata: Mapping[str, Any],
    *,
    expected_parameter_count: int,
) -> dict[str, Any]:
    state = _fresh_bank_state(tensors)
    parameter_count = sum(int(value.numel()) for value in state.values())
    if parameter_count != expected_parameter_count:
        raise ValueError(
            "Fresh Stage-B bank parameter-count mismatch: "
            f"observed={parameter_count} expected={expected_parameter_count}"
        )
    state_hash = tensor_state_sha256(state)
    bank_hashes = metadata.get("lora_bank_state_sha256")
    if not isinstance(bank_hashes, Mapping) or bank_hashes.get(FRESH_BANK_NAME) != state_hash:
        raise ValueError("Fresh Stage-B bank metadata hash mismatch")
    return {
        "state": state,
        "state_sha256": state_hash,
        "parameter_count": parameter_count,
        "exact_zero_output": all(
            torch.count_nonzero(value).item() == 0
            for name, value in state.items()
            if name.endswith(".lora_b")
        ),
        "b_tensor_count": sum(name.endswith(".lora_b") for name in state),
    }


def _validate_update_zero(
    metadata: Mapping[str, Any],
    audit: Mapping[str, Any],
    *,
    expected_initial_hash: str,
) -> dict[str, Any]:
    if metadata.get("optimizer_step") != 0:
        raise ValueError("Stage-B update_000 metadata must have optimizer_step 0")
    stage_b = metadata.get("v28_stage_b")
    if not isinstance(stage_b, Mapping):
        raise TypeError("Stage-B update_000 lacks v28_stage_b provenance")
    selected_update = stage_b.get("stage_a_selected_update")
    selected_arm = stage_b.get("stage_a_selected_arm")
    if isinstance(selected_update, bool) or not isinstance(selected_update, int):
        raise TypeError("Stage-B update_000 lacks integer Stage-A selected update")
    if selected_update <= 0:
        raise ValueError("Stage-B must descend from a nonzero Stage-A update")
    if (
        not isinstance(selected_arm, Mapping)
        or selected_arm.get("eligible") is not True
        or selected_arm.get("update") != selected_update
    ):
        raise ValueError("Stage-B update_000 lacks an eligible selected Stage-A arm")
    source = stage_b.get("source_stage_a_checkpoint")
    if not isinstance(source, str) or not source.endswith(f"update_{selected_update:03d}"):
        raise ValueError("Stage-B update_000 Stage-A checkpoint provenance is inconsistent")
    equivalence = stage_b.get("update_zero_equivalence")
    required_equivalence = {
        "verified": True,
        "base": "selector_approved_stage_a_checkpoint",
        "bank": FRESH_BANK_NAME,
        "question_dependent_scene_processing": False,
        "validation_nll_equivalent": True,
    }
    if not isinstance(equivalence, Mapping) or any(
        equivalence.get(key) != value for key, value in required_equivalence.items()
    ):
        raise ValueError("Stage-B update_000 equivalence contract is incomplete")
    if audit["state_sha256"] != expected_initial_hash:
        raise ValueError("Stage-B update_000 fresh bank initial hash mismatch")
    if audit["exact_zero_output"] is not True or int(audit["b_tensor_count"]) < 1:
        raise ValueError("Stage-B update_000 fresh bank is not exact-zero output")
    if equivalence.get("state_sha256") != audit["state_sha256"]:
        raise ValueError("Stage-B update_000 equivalence hash mismatch")
    if equivalence.get("parameter_count") != audit["parameter_count"]:
        raise ValueError("Stage-B update_000 equivalence parameter count mismatch")
    return {
        "verified": True,
        "stage_a_selected_update": selected_update,
        "source_stage_a_checkpoint": source,
        "fresh_bank_initial_state_sha256": audit["state_sha256"],
        "fresh_bank_exact_zero_output": True,
        "validation_nll_equivalent": True,
    }


def _runtime_fresh_bank_state_module(
    runtime: StaticChatRuntime, config: Mapping[str, Any]
) -> LoRAAdapterState:
    settings = lora_banks_settings(config).bank(FRESH_BANK_NAME).adapter
    adapters: list[LoRALinear] = []
    for target in settings.target_modules:
        module = runtime.language.model.get_submodule(target)
        if not isinstance(module, LoRALinear):
            raise TypeError(f"Fresh Stage-B target is not a LoRALinear: {target}")
        adapters.append(module)
    return LoRAAdapterState(settings.target_modules, adapters)


def _selection_requirements(config: Mapping[str, Any]) -> tuple[int, int, int, str, int]:
    contract = config.get("v28_stage_b")
    if not isinstance(contract, Mapping):
        raise TypeError("V28 Stage-B selector requires v28_stage_b")
    if contract.get("new_bank") != FRESH_BANK_NAME:
        raise ValueError("V28 Stage-B selector fresh-bank contract mismatch")
    requirements = contract.get("selection_requires")
    if not isinstance(requirements, Mapping):
        raise TypeError("v28_stage_b.selection_requires must be a mapping")
    if requirements.get("no_new_negative_sides") is not True:
        raise ValueError("V28 Stage-B must require no new negative control sides")
    if requirements.get("validation_nll_must_improve") is not True:
        raise ValueError("V28 Stage-B must require strict validation-NLL improvement")
    training = config.get("training")
    stage_b_training = (
        training.get("post_stack_decoder_stage_b") if isinstance(training, Mapping) else None
    )
    if not isinstance(stage_b_training, Mapping):
        raise TypeError("V28 Stage-B selector lacks bounded training settings")
    return (
        int(requirements.get("color_full_vocab_sides", 12)),
        int(requirements.get("mirror_full_vocab_sides", 10)),
        int(stage_b_training["max_optimizer_steps"]),
        str(contract["new_bank_initial_state_sha256"]),
        int(contract["new_bank_parameter_count"]),
    )


def _select_eligible_arm(arms: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    eligible = [arm for arm in arms if arm.get("eligible") is True]
    return min(
        eligible,
        key=lambda arm: (
            float(arm["validation_answer_token_nll"]),
            int(arm["update"]),
        ),
        default=None,
    )


def _retention_control_config(config: dict[str, Any]) -> dict[str, Any]:
    """Keep V29 selection anchored to the immutable V24/V28 controls."""

    development = v29_development_contract(config)
    if development is None:
        return config
    control = load_config(development.retention_control_config)
    if v29_development_contract(control) is not None:
        raise ValueError("V29 retention controls must come from the pre-V29 config")
    control_qa = project_path(control, "qa").resolve()
    training_qa = project_path(config, "qa").resolve()
    if control_qa == training_qa:
        raise ValueError("V29 retention controls must be separate from diverse20 QA")
    curriculum = pair_curriculum_settings(control)
    required_scene_ids = {
        "scene_000003",
        "scene_000004",
        "scene_000007",
        "scene_000008",
    }
    if set(curriculum.pair_only_scene_ids) != required_scene_ids:
        raise ValueError("V29 retention config does not preserve V24/V28 control scenes")
    return control


def select_stage_b(config_path: Path, checkpoint_root: Path) -> dict[str, Any]:
    config = load_config(config_path)
    control_config = _retention_control_config(config)
    (
        minimum_color,
        minimum_mirror,
        final_update,
        expected_initial_hash,
        expected_parameter_count,
    ) = _selection_requirements(config)
    checkpoints = _checkpoint_paths(checkpoint_root, expected_final_update=final_update)

    curriculum = pair_curriculum_settings(control_config)
    records = list(SceneQADataset(project_path(control_config, "qa", "train.jsonl")).records)
    records = select_pair_only_records(records, curriculum.pair_only_scene_ids)
    records = cap_pair_units_per_pair(
        records, curriculum.max_units_per_pair, seed=int(control_config["seed"])
    )
    units = build_exact_question_pair_units(records)
    if len(units) != 12:
        raise ValueError(f"V28 Stage-B selector requires 12 paired units; got {len(units)}")
    scene_ids = sorted({scene_id for unit in units for scene_id in unit.scene_ids})

    runtime = StaticChatRuntime.load(
        config,
        scene_ids[0],
        checkpoint=checkpoints[0],
        local_files_only=True,
    )
    state_module = _runtime_fresh_bank_state_module(runtime, config)
    maps: dict[str, MapTensorData] = {
        scene_id: load_map_tensors(
            project_path(control_config, "maps", scene_id, "voxel_map.npz"),
            control_config["scene"]["room_size_m"],
            runtime.language.device,
            input_voxel_size_m=control_config["scene_encoder"].get("input_voxel_size_m"),
        )
        for scene_id in scene_ids
    }

    color_pair_id, mirror_pair_id = _pair_role_ids(control_config)
    arms: list[dict[str, Any]] = []
    frozen_hash: str | None = None
    baseline_negatives: set[tuple[str, str]] | None = None
    baseline_validation: float | None = None
    update_zero_provenance: dict[str, Any] | None = None
    for index, checkpoint in enumerate(checkpoints):
        metadata = _metadata(checkpoint)
        _validate_runtime_metadata(checkpoint, metadata)
        tensors = load_file(checkpoint / "adapter.safetensors", device="cpu")
        observed_frozen_hash = _frozen_tensor_sha256(tensors)
        if frozen_hash is None:
            frozen_hash = observed_frozen_hash
        elif observed_frozen_hash != frozen_hash:
            raise RuntimeError(f"Frozen checkpoint tensors changed in {checkpoint.name}")
        recorded_frozen_hash = metadata.get("v28_stage_b", {}).get("frozen_state_sha256")
        if recorded_frozen_hash != observed_frozen_hash:
            raise ValueError(f"Frozen-state metadata hash mismatch in {checkpoint.name}")

        audit = _fresh_bank_audit(
            tensors,
            metadata,
            expected_parameter_count=expected_parameter_count,
        )
        if index == 0:
            update_zero_provenance = _validate_update_zero(
                metadata, audit, expected_initial_hash=expected_initial_hash
            )
        state_module.load_state_dict(audit["state"], strict=True)

        gate = _teacher_gate(
            runtime=runtime,
            units=units,
            maps=maps,
            config=control_config,
        )
        color_sides, color_units = _full_vocab_counts(gate["by_pair"][color_pair_id])
        mirror_sides, mirror_units = _full_vocab_counts(gate["by_pair"][mirror_pair_id])
        observed_negatives = _negative_sides(gate)
        if baseline_negatives is None:
            baseline_negatives = observed_negatives
        new_negatives = sorted(observed_negatives - baseline_negatives)
        validation_nll = _validation_nll(metadata)
        if baseline_validation is None:
            baseline_validation = validation_nll
        checks = {
            "color_retained": color_sides >= minimum_color,
            "mirror_retained": mirror_sides >= minimum_mirror,
            "no_new_negative_sides": not new_negatives,
            "validation_nll_improved": validation_nll < baseline_validation,
        }
        update = metadata.get("optimizer_step")
        if update != index:
            raise ValueError(
                f"Stage-B checkpoint name/optimizer-step mismatch in {checkpoint.name}"
            )
        arms.append(
            {
                "checkpoint": str(checkpoint),
                "update": update,
                "fresh_bank_state_sha256": audit["state_sha256"],
                "fresh_bank_parameter_count": audit["parameter_count"],
                "validation_answer_token_nll": validation_nll,
                "color_full_vocab_sides": color_sides,
                "color_full_vocab_units": color_units,
                "mirror_full_vocab_sides": mirror_sides,
                "mirror_full_vocab_units": mirror_units,
                "new_negative_sides": new_negatives,
                "checks": checks,
                "eligible": index > 0 and all(checks.values()),
            }
        )

    selected = _select_eligible_arm(arms)
    return {
        "schema_version": 1,
        "artifact": "v28_post_stack_decoder_stage_b_selection",
        "training_evaluation_only": True,
        "question_text_serialized": False,
        "answer_text_serialized": False,
        "oracle_loaded": False,
        "question_dependent_scene_processing": False,
        "question_dependent_retrieval": False,
        "model_load_count": 1,
        "scene_ids": scene_ids,
        "retention_control_config": str(control_config.get("_config_path", "<in-memory>")),
        "retention_control_qa_root": str(project_path(control_config, "qa")),
        "training_qa_root": str(project_path(config, "qa")),
        "pair_unit_count": len(units),
        "fresh_bank": FRESH_BANK_NAME,
        "frozen_tensor_sha256": frozen_hash,
        "update_zero_provenance": update_zero_provenance,
        "baseline_validation_answer_token_nll": baseline_validation,
        "requirements": {
            "color_full_vocab_sides": minimum_color,
            "mirror_full_vocab_sides": minimum_mirror,
            "no_new_negative_sides": True,
            "validation_nll_must_improve": True,
        },
        "arms": arms,
        "selected_checkpoint": None if selected is None else selected["checkpoint"],
        "selected_update": None if selected is None else selected["update"],
        "passed": selected is not None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/gemma4_color_mirror_post_stack_decoder_stage_b_v28.yaml"),
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=Path("data_gemma4/checkpoints/gemma4_v28_post_stack_decoder_stage_b"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/gemma4/metrics/v28_stage_b_selection.json"),
    )
    args = parser.parse_args()
    report = select_stage_b(args.config, args.checkpoint_root)
    _atomic_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
