"""Sweep bounded V27 sidecar scales with one frozen Gemma runtime load.

This is a training/evaluation-only process.  It uses paired QA records to run
the existing teacher-forced causal-interface gate, but its output contains only
opaque scene/question IDs and numeric measurements.  No question text, answer
text, oracle geometry, or category labels are serialized.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from semantic_3d_chat.chat.runtime import StaticChatRuntime
from semantic_3d_chat.config import load_config, project_path
from semantic_3d_chat.data.dataset import SceneQADataset
from semantic_3d_chat.evaluation.v27_sidecar_screen import (
    _atomic_json,
    _full_vocab_counts,
    _negative_sides,
    _pair_role_ids,
)
from semantic_3d_chat.scene_encoder.map_io import MapTensorData, load_map_tensors
from semantic_3d_chat.training.pair_curriculum import (
    CounterfactualPairUnit,
    PairCurriculumSettings,
    build_exact_question_pair_units,
    cap_pair_units_per_pair,
    pair_curriculum_settings,
    select_pair_only_records,
)
from semantic_3d_chat.training.train_adapter import (
    evaluate_pair_candidate_gate,
    map_forward,
)


def _validated_scales(scales: list[float]) -> list[float]:
    if not scales:
        raise ValueError("At least one sidecar scale is required")
    unique_scales = sorted(set(scales))
    for scale in unique_scales:
        if not math.isfinite(scale) or scale < 0.0:
            raise ValueError(f"Sidecar scales must be finite and non-negative; got {scale}")
    return unique_scales


def _prefixes_at_scale(
    *,
    runtime: StaticChatRuntime,
    maps: dict[str, MapTensorData],
    scale: float,
    model_dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    if runtime.dense_aligner is None:
        raise ValueError("V27 runtime did not load a dense alignment sidecar")
    runtime.dense_aligner.sidecar_scale = scale
    prefixes: dict[str, torch.Tensor] = {}
    with torch.inference_mode():
        for scene_id, data in maps.items():
            encoded = map_forward(
                runtime.scene_model,
                data,
                runtime.global_scene_residual,
                runtime.signed_x_scene_residual,
                runtime.dense_aligner,
            )
            prefixes[scene_id] = runtime.composer.scene_prefix(
                encoded.scene_tokens.to(model_dtype)
            ).float()
    return prefixes


def _prefix_drift(
    *,
    base_prefixes: dict[str, torch.Tensor],
    adapted_prefixes: dict[str, torch.Tensor],
    maps: dict[str, MapTensorData],
) -> tuple[dict[str, dict[str, float | int]], float, float]:
    by_scene: dict[str, dict[str, float | int]] = {}
    for scene_id, base_prefix in base_prefixes.items():
        adapted_prefix = adapted_prefixes[scene_id]
        delta_rms = float((adapted_prefix - base_prefix).square().mean().sqrt())
        base_rms = float(base_prefix.square().mean().sqrt())
        relative_rms = delta_rms / max(base_rms, torch.finfo(torch.float32).eps)
        cosine = float(
            torch.nn.functional.cosine_similarity(
                base_prefix.flatten(), adapted_prefix.flatten(), dim=0
            )
        )
        if not math.isfinite(relative_rms) or not math.isfinite(cosine):
            raise RuntimeError("V27 scale sweep produced a non-finite prefix metric")
        data = maps[scene_id]
        by_scene[scene_id] = {
            "source_voxel_count": data.source_voxel_count,
            "coarse_voxel_count": data.voxel_count,
            "base_prefix_rms": base_rms,
            "delta_rms": delta_rms,
            "relative_rms": relative_rms,
            "cosine": cosine,
        }
    maximum_relative_rms = max(float(value["relative_rms"]) for value in by_scene.values())
    minimum_cosine = min(float(value["cosine"]) for value in by_scene.values())
    return by_scene, maximum_relative_rms, minimum_cosine


def _source_mirror_negatives(
    *,
    contract: dict,
    units: list[CounterfactualPairUnit],
    mirror_pair_id: str,
) -> tuple[set[tuple[str, str]], set[str]]:
    base_checkpoint = Path(contract["base_checkpoint"])
    base_metadata = json.loads((base_checkpoint / "metadata.json").read_text(encoding="utf-8"))
    base_gate = base_metadata["pair_candidate_gate"]
    source_mirror_gate = base_gate["by_pair"][mirror_pair_id]
    mirror_scene_ids = set(
        next(unit.scene_ids for unit in units if unit.pair_id == mirror_pair_id)
    )
    source_negative_sides = _negative_sides(
        {
            **source_mirror_gate,
            "detail": {
                "units": [
                    unit
                    for unit in base_gate["detail"]["units"]
                    if {side["scene_id"] for side in unit["sides"]} == mirror_scene_ids
                ]
            },
        }
    )
    return source_negative_sides, mirror_scene_ids


def _evaluate_scale(
    *,
    scale: float,
    runtime: StaticChatRuntime,
    units: list[CounterfactualPairUnit],
    maps: dict[str, MapTensorData],
    config: dict,
    curriculum: PairCurriculumSettings,
    color_pair_id: str,
    mirror_pair_id: str,
    mirror_scene_ids: set[str],
    source_negative_sides: set[tuple[str, str]],
    base_prefixes: dict[str, torch.Tensor],
    model_dtype: torch.dtype,
    requirements: dict,
) -> dict:
    if runtime.dense_aligner is None:
        raise ValueError("V27 runtime did not load a dense alignment sidecar")
    adapted_prefixes = (
        base_prefixes
        if scale == 0.0
        else _prefixes_at_scale(
            runtime=runtime,
            maps=maps,
            scale=scale,
            model_dtype=model_dtype,
        )
    )
    by_scene, maximum_relative_rms, minimum_cosine = _prefix_drift(
        base_prefixes=base_prefixes,
        adapted_prefixes=adapted_prefixes,
        maps=maps,
    )
    runtime.dense_aligner.sidecar_scale = scale
    gate = evaluate_pair_candidate_gate(
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
    )
    color_gate = gate["by_pair"][color_pair_id]
    mirror_gate = gate["by_pair"][mirror_pair_id]
    color_sides, color_units = _full_vocab_counts(color_gate)
    mirror_sides, mirror_units = _full_vocab_counts(mirror_gate)
    observed_negative_sides = {
        (str(side["scene_id"]), str(side["question_id"]))
        for unit in gate["detail"]["units"]
        for side in unit["sides"]
        if side["scene_id"] in mirror_scene_ids
        and not bool(side["full_vocab_top1_passed"])
    }
    new_negative_sides = sorted(observed_negative_sides - source_negative_sides)
    checks = {
        "color_full_vocab_sides": color_sides
        == int(requirements["color_full_vocab_sides"]),
        "color_full_vocab_units": color_units
        == int(requirements["color_full_vocab_units"]),
        "mirror_minimum_full_vocab_sides": mirror_sides
        >= int(requirements["mirror_minimum_full_vocab_sides"]),
        "mirror_minimum_full_vocab_units": mirror_units
        >= int(requirements["mirror_minimum_full_vocab_units"]),
        "no_new_negative_sides": not new_negative_sides,
        "maximum_relative_prefix_rms_drift": maximum_relative_rms
        <= float(requirements["maximum_relative_prefix_rms_drift"]),
        "minimum_prefix_cosine": minimum_cosine
        >= float(requirements["minimum_prefix_cosine"]),
    }
    teacher_side_gates_pass = (
        checks["color_full_vocab_sides"]
        and checks["mirror_minimum_full_vocab_sides"]
    )
    return {
        "sidecar_scale": scale,
        "color": {
            "pair_id": color_pair_id,
            "full_vocab_sides": color_sides,
            "full_vocab_units": color_units,
            "minimum_candidate_margin": color_gate[
                "minimum_own_vs_alternate_candidate_logit_margin"
            ],
            "minimum_full_vocab_margin": color_gate[
                "minimum_first_answer_token_target_vs_best_other_logit_margin"
            ],
        },
        "mirror": {
            "pair_id": mirror_pair_id,
            "full_vocab_sides": mirror_sides,
            "full_vocab_units": mirror_units,
            "minimum_candidate_margin": mirror_gate[
                "minimum_own_vs_alternate_candidate_logit_margin"
            ],
            "minimum_full_vocab_margin": mirror_gate[
                "minimum_first_answer_token_target_vs_best_other_logit_margin"
            ],
            "source_negative_sides": sorted(source_negative_sides),
            "observed_negative_sides": sorted(observed_negative_sides),
            "new_negative_sides": new_negative_sides,
        },
        "prefix_drift_by_scene": by_scene,
        "maximum_relative_prefix_rms_drift": maximum_relative_rms,
        "minimum_prefix_cosine": minimum_cosine,
        "teacher_side_gates_pass": teacher_side_gates_pass,
        "checks": checks,
        "passed": all(checks.values()),
    }


def run_sweep(config_path: Path, checkpoint: Path, scales: list[float]) -> dict:
    scales = _validated_scales(scales)
    config = load_config(config_path)
    contract = config.get("v27_screen")
    if not isinstance(contract, dict):
        raise TypeError("V27 scale sweep requires a v27_screen config mapping")
    curriculum = pair_curriculum_settings(config)
    records = list(SceneQADataset(project_path(config, "qa", "train.jsonl")).records)
    records = select_pair_only_records(records, curriculum.pair_only_scene_ids)
    records = cap_pair_units_per_pair(
        records,
        curriculum.max_units_per_pair,
        seed=int(config["seed"]),
    )
    units = build_exact_question_pair_units(records)
    if len(units) != 12:
        raise ValueError(f"V27 scale sweep requires exactly 12 paired units; got {len(units)}")
    scene_ids = sorted({scene_id for unit in units for scene_id in unit.scene_ids})

    runtime = StaticChatRuntime.load(
        config,
        scene_ids[0],
        checkpoint=checkpoint,
        local_files_only=True,
    )
    if runtime.dense_aligner is None:
        raise ValueError("V27 runtime did not load a dense alignment sidecar")
    if runtime.dense_aligner.application_mode != "coverage_sidecar":
        raise ValueError("V27 scale sweep requires coverage_sidecar mode")
    configured_scale = runtime.dense_aligner.sidecar_scale
    if configured_scale != float(contract["selected_sidecar_scale"]):
        raise ValueError("Runtime sidecar scale does not match the V27 contract")

    maps = {
        scene_id: load_map_tensors(
            project_path(config, "maps", scene_id, "voxel_map.npz"),
            config["scene"]["room_size_m"],
            device=runtime.language.device,
            input_voxel_size_m=config["scene_encoder"].get("input_voxel_size_m"),
        )
        for scene_id in scene_ids
    }
    model_dtype = next(runtime.language.model.parameters()).dtype
    base_prefixes = _prefixes_at_scale(
        runtime=runtime,
        maps=maps,
        scale=0.0,
        model_dtype=model_dtype,
    )
    color_pair_id, mirror_pair_id = _pair_role_ids(config)
    source_negative_sides, mirror_scene_ids = _source_mirror_negatives(
        contract=contract,
        units=units,
        mirror_pair_id=mirror_pair_id,
    )
    requirements = contract["no_step_requires"]
    scale_results = [
        _evaluate_scale(
            scale=scale,
            runtime=runtime,
            units=units,
            maps=maps,
            config=config,
            curriculum=curriculum,
            color_pair_id=color_pair_id,
            mirror_pair_id=mirror_pair_id,
            mirror_scene_ids=mirror_scene_ids,
            source_negative_sides=source_negative_sides,
            base_prefixes=base_prefixes,
            model_dtype=model_dtype,
            requirements=requirements,
        )
        for scale in scales
    ]
    runtime.dense_aligner.sidecar_scale = configured_scale
    return {
        "schema_version": 1,
        "artifact": "v27_bounded_sidecar_scale_sweep",
        "training_evaluation_only": True,
        "optimizer_steps": 0,
        "model_load_count": 1,
        "question_text_serialized": False,
        "answer_text_serialized": False,
        "oracle_loaded": False,
        "question_dependent_scene_processing": False,
        "base_semantic_path_modified": False,
        "configured_sidecar_scale": configured_scale,
        "scene_ids": scene_ids,
        "pair_unit_count": len(units),
        "swept_scales": scales,
        "scale_results": scale_results,
        "teacher_side_gate_passing_scales": [
            result["sidecar_scale"]
            for result in scale_results
            if result["teacher_side_gates_pass"]
        ],
        "full_contract_passing_scales": [
            result["sidecar_scale"] for result in scale_results if result["passed"]
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/gemma4_color_mirror_dense_sidecar_v27.yaml"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("data_gemma4/checkpoints/gemma4_v27_dense_sidecar/candidate_beta_010"),
    )
    parser.add_argument(
        "--scales",
        type=float,
        nargs="+",
        default=[0.0, 0.01, 0.025, 0.05, 0.075, 0.1],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/gemma4/metrics/v27_sidecar_scale_sweep.json"),
    )
    args = parser.parse_args()
    report = run_sweep(args.config, args.checkpoint, args.scales)
    _atomic_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
