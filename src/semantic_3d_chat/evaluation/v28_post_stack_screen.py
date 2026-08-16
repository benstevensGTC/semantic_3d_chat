"""Exact update-zero audit for the V28 post-stack dense sidecar adapter.

The audit proves that installing the calibrated all-voxel branch and its
zero-output post-stack adapter does not change V24 scene tokens, prefixes, or
teacher-forced decoder results.  QA is read only by this offline evaluator;
question and answer text are never copied into the runtime checkpoint or the
numeric report.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from semantic_3d_chat.chat.runtime import StaticChatRuntime
from semantic_3d_chat.config import load_config, project_path
from semantic_3d_chat.data.dataset import SceneQADataset
from semantic_3d_chat.evaluation.v27_sidecar_screen import (
    _atomic_json,
    _full_vocab_counts,
    _pair_role_ids,
)
from semantic_3d_chat.language.prefix_injection import prefix_sha256
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


def _pair_gate(
    *,
    runtime: StaticChatRuntime,
    units,
    maps,
    config: dict,
    with_sidecar: bool,
) -> dict:
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
        lora_installation=None,
        dense_aligner=runtime.dense_aligner if with_sidecar else None,
        dense_sidecar_adapter=(
            runtime.dense_sidecar_adapter if with_sidecar else None
        ),
    )


def run_screen(config_path: Path, checkpoint: Path) -> dict:
    config = load_config(config_path)
    contract = config.get("v28_screen")
    if not isinstance(contract, dict):
        raise TypeError("V28 screen requires a v28_screen config mapping")
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
        raise ValueError(f"V28 update-zero audit requires 12 paired units; got {len(units)}")
    scene_ids = sorted({scene_id for unit in units for scene_id in unit.scene_ids})

    runtime = StaticChatRuntime.load(
        config,
        scene_ids[0],
        checkpoint=checkpoint,
        local_files_only=True,
    )
    if runtime.dense_aligner is None or runtime.dense_sidecar_adapter is None:
        raise ValueError("V28 runtime must load both dense bridge and post-stack adapter")
    if runtime.dense_aligner.application_mode != "coverage_sidecar":
        raise ValueError("V28 requires coverage_sidecar dense routing")
    if runtime.dense_aligner.sidecar_scale != 0.0:
        raise ValueError("V28 update-zero audit requires an exact zero sidecar scale")
    structural = runtime.dense_sidecar_adapter.validate_structural_state()

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
    equivalence_by_scene: dict[str, dict] = {}
    with torch.inference_mode():
        for scene_id in scene_ids:
            data = maps[scene_id]
            v24_output = map_forward(
                runtime.scene_model,
                data,
                runtime.global_scene_residual,
                runtime.signed_x_scene_residual,
            )
            routed_output = map_forward(
                runtime.scene_model,
                data,
                runtime.global_scene_residual,
                runtime.signed_x_scene_residual,
                runtime.dense_aligner,
            )
            adapted_output = map_forward(
                runtime.scene_model,
                data,
                runtime.global_scene_residual,
                runtime.signed_x_scene_residual,
                runtime.dense_aligner,
                runtime.dense_sidecar_adapter,
            )
            v24_prefix = runtime.composer.scene_prefix(
                v24_output.scene_tokens.to(model_dtype)
            )
            routed_prefix = runtime.composer.scene_prefix(
                routed_output.scene_tokens.to(model_dtype)
            )
            adapted_prefix = runtime.composer.scene_prefix(
                adapted_output.scene_tokens.to(model_dtype)
            )
            processed = int(
                routed_output.audit["aligned_sidecar_processed_voxels"].detach().cpu()
            )
            minimum_contribution = float(
                routed_output.audit["aligned_sidecar_min_voxel_contribution"]
                .detach()
                .float()
                .cpu()
            )
            record = {
                "source_voxel_count": int(data.source_voxel_count),
                "coarse_voxel_count": int(data.voxel_count),
                "sidecar_processed_voxels": processed,
                "minimum_voxel_contribution": minimum_contribution,
                "v24_scene_sha256": prefix_sha256(v24_output.scene_tokens),
                "routed_scene_sha256": prefix_sha256(routed_output.scene_tokens),
                "adapted_scene_sha256": prefix_sha256(adapted_output.scene_tokens),
                "v24_prefix_sha256": prefix_sha256(v24_prefix),
                "routed_prefix_sha256": prefix_sha256(routed_prefix),
                "adapted_prefix_sha256": prefix_sha256(adapted_prefix),
                "routed_scene_exact": torch.equal(
                    v24_output.scene_tokens, routed_output.scene_tokens
                ),
                "adapted_scene_exact": torch.equal(
                    v24_output.scene_tokens, adapted_output.scene_tokens
                ),
                "routed_prefix_exact": torch.equal(v24_prefix, routed_prefix),
                "adapted_prefix_exact": torch.equal(v24_prefix, adapted_prefix),
                "all_voxels_covered": (
                    processed == data.voxel_count and minimum_contribution > 0.0
                ),
            }
            equivalence_by_scene[scene_id] = record

    base_gate = _pair_gate(
        runtime=runtime,
        units=units,
        maps=maps,
        config=config,
        with_sidecar=False,
    )
    adapted_gate = _pair_gate(
        runtime=runtime,
        units=units,
        maps=maps,
        config=config,
        with_sidecar=True,
    )
    color_pair_id, mirror_pair_id = _pair_role_ids(config)
    color_sides, color_units = _full_vocab_counts(adapted_gate["by_pair"][color_pair_id])
    mirror_sides, mirror_units = _full_vocab_counts(
        adapted_gate["by_pair"][mirror_pair_id]
    )
    teacher_gate_exact = base_gate["detail"] == adapted_gate["detail"]
    requirements = contract.get("update_zero_requires", {})
    expected_color_sides = int(requirements.get("color_full_vocab_sides", 12))
    expected_mirror_sides = int(requirements.get("mirror_full_vocab_sides", 10))
    checks = {
        "output_projection_exact_zero": structural["output_projection_exact_zero"],
        "channel_gain_exact_zero": structural["channel_gain_exact_zero"],
        "scene_tokens_exact": all(
            value["routed_scene_exact"] and value["adapted_scene_exact"]
            for value in equivalence_by_scene.values()
        ),
        "scene_prefixes_exact": all(
            value["routed_prefix_exact"] and value["adapted_prefix_exact"]
            for value in equivalence_by_scene.values()
        ),
        "all_voxels_covered": all(
            value["all_voxels_covered"] for value in equivalence_by_scene.values()
        ),
        "teacher_forced_logits_exact": teacher_gate_exact,
        "color_retained": color_sides == expected_color_sides,
        "mirror_retained": mirror_sides >= expected_mirror_sides,
    }
    return {
        "schema_version": 1,
        "artifact": "v28_post_stack_update_zero_screen",
        "training_evaluation_only": True,
        "optimizer_steps": 0,
        "question_text_serialized": False,
        "answer_text_serialized": False,
        "oracle_loaded": False,
        "question_dependent_scene_processing": False,
        "application_order": "after_global_and_signed_x_before_prefix_composer",
        "scene_ids": scene_ids,
        "pair_unit_count": len(units),
        "adapter_state_sha256": structural["state_sha256"],
        "adapter_parameter_count": structural["parameter_count"],
        "equivalence_by_scene": equivalence_by_scene,
        "teacher_gate_detail_exact": teacher_gate_exact,
        "color": {
            "pair_id": color_pair_id,
            "full_vocab_sides": color_sides,
            "full_vocab_units": color_units,
        },
        "mirror": {
            "pair_id": mirror_pair_id,
            "full_vocab_sides": mirror_sides,
            "full_vocab_units": mirror_units,
        },
        "checks": checks,
        "passed": all(checks.values()),
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
        "--checkpoint",
        type=Path,
        default=Path(
            "data_gemma4/checkpoints/gemma4_v28_post_stack_sidecar/candidate_zero"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/gemma4/metrics/v28_post_stack_update_zero_screen.json"),
    )
    args = parser.parse_args()
    report = run_screen(args.config, args.checkpoint)
    _atomic_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
