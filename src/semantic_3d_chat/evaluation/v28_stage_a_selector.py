"""Select a bounded V28 Stage-A checkpoint with causal retention gates.

The selector loads Gemma once, swaps only the two authorized post-stack
adapter surfaces, and evaluates every saved update against the immutable V24
color/mirror controls.  It rejects an update that improves validation NLL by
damaging established counterfactual behavior or by perturbing the continuous
prefix beyond the configured bound.
"""

from __future__ import annotations

import argparse
import json
import math
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
from semantic_3d_chat.language.lora import tensor_state_sha256
from semantic_3d_chat.scene_encoder.dense_sidecar_adapter import (
    validate_dense_sidecar_adapter_state,
)
from semantic_3d_chat.scene_encoder.map_io import MapTensorData, load_map_tensors
from semantic_3d_chat.training.checkpointing import TRAINING_METADATA_FILENAME
from semantic_3d_chat.training.pair_curriculum import (
    build_exact_question_pair_units,
    cap_pair_units_per_pair,
    pair_curriculum_settings,
    select_pair_only_records,
)
from semantic_3d_chat.training.train_adapter import (
    evaluate_pair_candidate_gate,
    map_forward,
)

_TRAINABLE_TENSORS = frozenset(
    {
        "dense_sidecar_adapter.output_projection.weight",
        "dense_sidecar_adapter.channel_gain",
    }
)


def _checkpoint_paths(root: Path) -> list[Path]:
    paths = sorted(
        path
        for path in root.glob("update_*")
        if path.is_dir()
        and (path / "adapter.safetensors").is_file()
        and (path / TRAINING_METADATA_FILENAME).is_file()
    )
    if not paths or paths[0].name != "update_000":
        raise FileNotFoundError("V28 Stage-A selection requires update_000")
    return paths


def _metadata(path: Path) -> dict[str, Any]:
    value = json.loads((path / TRAINING_METADATA_FILENAME).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Checkpoint metadata must be a JSON object: {path}")
    return value


def _sidecar_state(tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    prefix = "dense_sidecar_adapter."
    state = {
        name.removeprefix(prefix): value
        for name, value in tensors.items()
        if name.startswith(prefix)
    }
    if not state:
        raise ValueError("Checkpoint does not contain dense_sidecar_adapter tensors")
    return state


def _frozen_tensor_sha256(tensors: dict[str, torch.Tensor]) -> str:
    return tensor_state_sha256(
        {name: value for name, value in tensors.items() if name not in _TRAINABLE_TENSORS}
    )


def _validation_nll(metadata: dict[str, Any]) -> float:
    history = metadata.get("history")
    if not isinstance(history, list) or not history:
        raise ValueError("Stage-A checkpoint lacks training history")
    value = history[-1].get("validation_answer_token_nll")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("Stage-A checkpoint lacks numeric validation answer NLL")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Stage-A validation NLL is not finite")
    return result


def _prefix_metrics(
    *,
    runtime: StaticChatRuntime,
    maps: dict[str, MapTensorData],
    base_prefixes: dict[str, torch.Tensor],
) -> tuple[dict[str, dict[str, float | int]], float, float]:
    model_dtype = next(runtime.language.model.parameters()).dtype
    by_scene: dict[str, dict[str, float | int]] = {}
    with torch.inference_mode():
        for scene_id, data in maps.items():
            output = map_forward(
                runtime.scene_model,
                data,
                runtime.global_scene_residual,
                runtime.signed_x_scene_residual,
                runtime.dense_aligner,
                runtime.dense_sidecar_adapter,
            )
            prefix = runtime.composer.scene_prefix(
                output.scene_tokens.to(model_dtype)
            ).float()
            base = base_prefixes[scene_id]
            delta_rms = float((prefix - base).square().mean().sqrt())
            base_rms = float(base.square().mean().sqrt())
            relative_rms = delta_rms / max(
                base_rms, torch.finfo(torch.float32).eps
            )
            cosine = float(
                torch.nn.functional.cosine_similarity(
                    prefix.flatten(), base.flatten(), dim=0
                )
            )
            if not math.isfinite(relative_rms) or not math.isfinite(cosine):
                raise RuntimeError("V28 Stage-A prefix metric is nonfinite")
            by_scene[scene_id] = {
                "source_voxel_count": int(data.source_voxel_count),
                "coarse_voxel_count": int(data.voxel_count),
                "relative_rms": relative_rms,
                "cosine": cosine,
            }
    return (
        by_scene,
        max(float(item["relative_rms"]) for item in by_scene.values()),
        min(float(item["cosine"]) for item in by_scene.values()),
    )


def _teacher_gate(
    *,
    runtime: StaticChatRuntime,
    units,
    maps,
    config: dict,
) -> dict[str, Any]:
    curriculum = pair_curriculum_settings(config)
    return evaluate_pair_candidate_gate(
        units,
        maps=maps,
        config=config,
        language=runtime.language,
        scene_model=runtime.scene_model,
        global_scene_residual=runtime.global_scene_residual,
        signed_x_scene_residual=runtime.signed_x_scene_residual,
        composer=runtime.composer,
        grounding=runtime.grounding,
        units_per_batch=curriculum.units_per_batch,
        ranking_margin=curriculum.ranking_margin,
        ranking_mode=curriculum.ranking_mode,
        changed_unit_accuracy_threshold=curriculum.changed_unit_accuracy_threshold,
        prediction_flip_threshold=curriculum.prediction_flip_threshold,
        wrong_prefix_flip_threshold=curriculum.wrong_prefix_flip_threshold,
        first_answer_token_top1_accuracy_threshold=(
            curriculum.first_answer_token_top1_accuracy_threshold
        ),
        dense_aligner=runtime.dense_aligner,
        dense_sidecar_adapter=runtime.dense_sidecar_adapter,
    )


def select_stage_a(config_path: Path, checkpoint_root: Path) -> dict[str, Any]:
    config = load_config(config_path)
    contract = config.get("v28_screen")
    if not isinstance(contract, dict):
        raise TypeError("V28 selector requires v28_screen")
    requirements = contract.get("stage_a_selection_requires", {})
    if not isinstance(requirements, dict):
        raise TypeError("v28_screen.stage_a_selection_requires must be a mapping")
    minimum_color = int(requirements.get("color_full_vocab_sides", 12))
    minimum_mirror = int(requirements.get("mirror_full_vocab_sides", 10))
    maximum_relative_rms = float(
        requirements.get("maximum_relative_prefix_rms_drift", 0.10)
    )
    minimum_cosine = float(requirements.get("minimum_prefix_cosine", 0.995))

    curriculum = pair_curriculum_settings(config)
    records = list(SceneQADataset(project_path(config, "qa", "train.jsonl")).records)
    records = select_pair_only_records(records, curriculum.pair_only_scene_ids)
    records = cap_pair_units_per_pair(
        records, curriculum.max_units_per_pair, seed=int(config["seed"])
    )
    units = build_exact_question_pair_units(records)
    if len(units) != 12:
        raise ValueError(f"V28 selector requires 12 paired units; got {len(units)}")
    scene_ids = sorted({scene_id for unit in units for scene_id in unit.scene_ids})
    checkpoints = _checkpoint_paths(checkpoint_root)

    runtime = StaticChatRuntime.load(
        config,
        scene_ids[0],
        checkpoint=checkpoints[0],
        local_files_only=True,
    )
    if runtime.dense_sidecar_adapter is None or runtime.dense_aligner is None:
        raise ValueError("V28 Stage-A runtime lacks its dense sidecar stack")
    maps = {
        scene_id: load_map_tensors(
            project_path(config, "maps", scene_id, "voxel_map.npz"),
            config["scene"]["room_size_m"],
            runtime.language.device,
            input_voxel_size_m=config["scene_encoder"].get("input_voxel_size_m"),
        )
        for scene_id in scene_ids
    }
    model_dtype = next(runtime.language.model.parameters()).dtype
    base_prefixes: dict[str, torch.Tensor] = {}
    with torch.inference_mode():
        for scene_id, data in maps.items():
            base = map_forward(
                runtime.scene_model,
                data,
                runtime.global_scene_residual,
                runtime.signed_x_scene_residual,
            )
            base_prefixes[scene_id] = runtime.composer.scene_prefix(
                base.scene_tokens.to(model_dtype)
            ).float()

    color_pair_id, mirror_pair_id = _pair_role_ids(config)
    arms: list[dict[str, Any]] = []
    frozen_hash: str | None = None
    source_negatives: set[tuple[str, str]] | None = None
    baseline_validation: float | None = None
    for index, checkpoint in enumerate(checkpoints):
        metadata = _metadata(checkpoint)
        tensors = load_file(checkpoint / "adapter.safetensors", device="cpu")
        observed_frozen_hash = _frozen_tensor_sha256(tensors)
        if frozen_hash is None:
            frozen_hash = observed_frozen_hash
        elif observed_frozen_hash != frozen_hash:
            raise RuntimeError(
                f"Frozen checkpoint tensors changed in {checkpoint.name}"
            )
        runtime.dense_sidecar_adapter.load_state_dict(
            _sidecar_state(tensors), strict=True
        )
        audit = validate_dense_sidecar_adapter_state(
            runtime.dense_sidecar_adapter,
            expected_parameter_count=int(
                metadata["dense_sidecar_adapter_parameter_count"]
            ),
            expected_state_sha256=str(
                metadata["dense_sidecar_adapter_state_sha256"]
            ),
            context=f"V28 Stage-A selection {checkpoint.name}",
        )
        prefix_by_scene, maximum_drift, observed_minimum_cosine = _prefix_metrics(
            runtime=runtime,
            maps=maps,
            base_prefixes=base_prefixes,
        )
        gate = _teacher_gate(
            runtime=runtime,
            units=units,
            maps=maps,
            config=config,
        )
        color_sides, color_units = _full_vocab_counts(
            gate["by_pair"][color_pair_id]
        )
        mirror_sides, mirror_units = _full_vocab_counts(
            gate["by_pair"][mirror_pair_id]
        )
        observed_negatives = _negative_sides(gate)
        if source_negatives is None:
            source_negatives = observed_negatives
        new_negatives = sorted(observed_negatives - source_negatives)
        validation_nll = _validation_nll(metadata)
        if baseline_validation is None:
            baseline_validation = validation_nll
        checks = {
            "color_retained": color_sides >= minimum_color,
            "mirror_retained": mirror_sides >= minimum_mirror,
            "no_new_negative_sides": not new_negatives,
            "prefix_drift_bounded": maximum_drift <= maximum_relative_rms,
            "prefix_cosine_bounded": observed_minimum_cosine >= minimum_cosine,
            "validation_nll_improved": (
                index == 0 or validation_nll < baseline_validation
            ),
        }
        arms.append(
            {
                "checkpoint": str(checkpoint),
                "update": int(metadata.get("optimizer_step", index)),
                "sidecar_state_sha256": audit["state_sha256"],
                "validation_answer_token_nll": validation_nll,
                "color_full_vocab_sides": color_sides,
                "color_full_vocab_units": color_units,
                "mirror_full_vocab_sides": mirror_sides,
                "mirror_full_vocab_units": mirror_units,
                "new_negative_sides": new_negatives,
                "maximum_relative_prefix_rms_drift": maximum_drift,
                "minimum_prefix_cosine": observed_minimum_cosine,
                "prefix_drift_by_scene": prefix_by_scene,
                "checks": checks,
                "eligible": index > 0 and all(checks.values()),
            }
        )

    eligible = [arm for arm in arms if arm["eligible"]]
    selected = min(
        eligible,
        key=lambda arm: (
            arm["validation_answer_token_nll"],
            arm["maximum_relative_prefix_rms_drift"],
            arm["update"],
        ),
        default=None,
    )
    return {
        "schema_version": 1,
        "artifact": "v28_post_stack_sidecar_stage_a_selection",
        "training_evaluation_only": True,
        "question_text_serialized": False,
        "answer_text_serialized": False,
        "oracle_loaded": False,
        "question_dependent_scene_processing": False,
        "model_load_count": 1,
        "scene_ids": scene_ids,
        "pair_unit_count": len(units),
        "frozen_tensor_sha256": frozen_hash,
        "baseline_validation_answer_token_nll": baseline_validation,
        "requirements": {
            "color_full_vocab_sides": minimum_color,
            "mirror_full_vocab_sides": minimum_mirror,
            "maximum_relative_prefix_rms_drift": maximum_relative_rms,
            "minimum_prefix_cosine": minimum_cosine,
            "no_new_negative_sides": True,
            "validation_nll_must_improve": True,
        },
        "arms": arms,
        "selected_checkpoint": (
            None if selected is None else selected["checkpoint"]
        ),
        "selected_update": None if selected is None else selected["update"],
        "passed": selected is not None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/experiments/gemma4_color_mirror_post_stack_sidecar_v28.yaml"
        ),
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=Path("data_gemma4/checkpoints/gemma4_v28_post_stack_sidecar/stage_a"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/gemma4/metrics/v28_stage_a_selection.json"),
    )
    args = parser.parse_args()
    report = select_stage_a(args.config, args.checkpoint_root)
    _atomic_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
