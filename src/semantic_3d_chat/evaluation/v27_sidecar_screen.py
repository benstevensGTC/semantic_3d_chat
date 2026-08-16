"""No-update causal-interface audit for the bounded V27 semantic sidecar.

This is a training/evaluation-only process.  It may read the selected paired QA
records to score teacher-forced behavior, but it never serializes question text,
answers, oracle geometry, or category labels into the runtime checkpoint.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path

import torch

from semantic_3d_chat.chat.runtime import StaticChatRuntime
from semantic_3d_chat.config import load_config, project_path
from semantic_3d_chat.data.dataset import SceneQADataset
from semantic_3d_chat.scene_encoder.map_io import load_map_tensors
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


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def _full_vocab_counts(gate: dict) -> tuple[int, int]:
    side_count = int(gate["side_count"])
    unit_count = int(gate["unit_count"])
    side_passes = round(float(gate["first_answer_token_top1_accuracy"]) * side_count)
    unit_passes = round(float(gate["first_answer_token_top1_unit_accuracy"]) * unit_count)
    return side_passes, unit_passes


def _negative_sides(gate: dict) -> set[tuple[str, str]]:
    return {
        (str(side["scene_id"]), str(side["question_id"]))
        for unit in gate["detail"]["units"]
        for side in unit["sides"]
        if not bool(side["full_vocab_top1_passed"])
    }


def _pair_role_ids(config: dict) -> tuple[str, str]:
    by_pair = config["training"]["pair_objectives"]["by_pair"]
    color = [pair_id for pair_id, value in by_pair.items() if value["role"] == "retention_control"]
    mirror = [pair_id for pair_id, value in by_pair.items() if value["role"] == "signed_target"]
    if len(color) != 1 or len(mirror) != 1:
        raise ValueError("V27 requires exactly one retention-control and one signed-target pair")
    return color[0], mirror[0]


def run_screen(config_path: Path, checkpoint: Path) -> dict:
    config = load_config(config_path)
    contract = config.get("v27_screen")
    if not isinstance(contract, dict):
        raise TypeError("V27 screen requires a v27_screen config mapping")
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
        raise ValueError(f"V27 no-step audit requires exactly 12 paired units; got {len(units)}")
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
        raise ValueError("V27 no-step audit requires coverage_sidecar mode")
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
    prefix_drift_by_scene: dict[str, dict[str, float | int]] = {}
    with torch.inference_mode():
        for scene_id in scene_ids:
            data = maps[scene_id]
            runtime.dense_aligner.sidecar_scale = 0.0
            base = map_forward(
                runtime.scene_model,
                data,
                runtime.global_scene_residual,
                runtime.signed_x_scene_residual,
                runtime.dense_aligner,
            )
            runtime.dense_aligner.sidecar_scale = configured_scale
            adapted = map_forward(
                runtime.scene_model,
                data,
                runtime.global_scene_residual,
                runtime.signed_x_scene_residual,
                runtime.dense_aligner,
            )
            base_prefix = runtime.composer.scene_prefix(base.scene_tokens.to(model_dtype)).float()
            adapted_prefix = runtime.composer.scene_prefix(
                adapted.scene_tokens.to(model_dtype)
            ).float()
            delta_rms = float((adapted_prefix - base_prefix).square().mean().sqrt())
            base_rms = float(base_prefix.square().mean().sqrt())
            relative_rms = delta_rms / max(base_rms, torch.finfo(torch.float32).eps)
            cosine = float(
                torch.nn.functional.cosine_similarity(
                    base_prefix.flatten(), adapted_prefix.flatten(), dim=0
                )
            )
            prefix_drift_by_scene[scene_id] = {
                "source_voxel_count": data.source_voxel_count,
                "coarse_voxel_count": data.voxel_count,
                "base_prefix_rms": base_rms,
                "delta_rms": delta_rms,
                "relative_rms": relative_rms,
                "cosine": cosine,
            }
    runtime.dense_aligner.sidecar_scale = configured_scale

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

    color_pair_id, mirror_pair_id = _pair_role_ids(config)
    color_gate = gate["by_pair"][color_pair_id]
    mirror_gate = gate["by_pair"][mirror_pair_id]
    color_sides, color_units = _full_vocab_counts(color_gate)
    mirror_sides, mirror_units = _full_vocab_counts(mirror_gate)

    base_checkpoint = Path(contract["base_checkpoint"])
    base_metadata = json.loads((base_checkpoint / "metadata.json").read_text(encoding="utf-8"))
    base_gate = base_metadata["pair_candidate_gate"]
    source_mirror_gate = base_gate["by_pair"][mirror_pair_id]
    mirror_scene_ids = set(
        next(unit.scene_ids for unit in units if unit.pair_id == mirror_pair_id)
    )
    source_negative_sides = _negative_sides(
        {**source_mirror_gate, "detail": {
            "units": [
                unit
                for unit in base_gate["detail"]["units"]
                if {side["scene_id"] for side in unit["sides"]} == mirror_scene_ids
            ]
        }}
    )
    observed_negative_sides = {
        (str(side["scene_id"]), str(side["question_id"]))
        for unit in gate["detail"]["units"]
        for side in unit["sides"]
        if side["scene_id"] in mirror_scene_ids
        and not bool(side["full_vocab_top1_passed"])
    }
    new_negative_sides = sorted(observed_negative_sides - source_negative_sides)

    requirements = contract["no_step_requires"]
    maximum_relative_rms = max(
        float(value["relative_rms"]) for value in prefix_drift_by_scene.values()
    )
    minimum_cosine = min(float(value["cosine"]) for value in prefix_drift_by_scene.values())
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
    if not math.isfinite(maximum_relative_rms) or not math.isfinite(minimum_cosine):
        raise RuntimeError("V27 prefix drift audit produced a non-finite metric")

    # The detailed teacher artifact intentionally contains only opaque IDs,
    # numeric token IDs, and margins; no question or answer text is serialized.
    return {
        "schema_version": 1,
        "artifact": "v27_bounded_sidecar_no_step_screen",
        "training_evaluation_only": True,
        "optimizer_steps": 0,
        "question_text_serialized": False,
        "answer_text_serialized": False,
        "oracle_loaded": False,
        "question_dependent_scene_processing": False,
        "base_semantic_path_modified": False,
        "sidecar_scale": configured_scale,
        "scene_ids": scene_ids,
        "pair_unit_count": len(units),
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
        "prefix_drift_by_scene": prefix_drift_by_scene,
        "maximum_relative_prefix_rms_drift": maximum_relative_rms,
        "minimum_prefix_cosine": minimum_cosine,
        "checks": checks,
        "passed": all(checks.values()),
        "teacher_gate_detail": gate["detail"],
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
        "--output",
        type=Path,
        default=Path("reports/gemma4/metrics/v27_sidecar_no_step_screen.json"),
    )
    args = parser.parse_args()
    report = run_screen(args.config, args.checkpoint)
    _atomic_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
