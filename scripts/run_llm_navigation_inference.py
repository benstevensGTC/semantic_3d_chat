#!/usr/bin/env python3
"""Run the local-Gemma navigation policy without opening any oracle data."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from semantic_3d_chat.chat.file_audit import FileAccessAudit
from semantic_3d_chat.chat.runtime import StaticChatRuntime
from semantic_3d_chat.chat.runtime_config import load_runtime_config
from semantic_3d_chat.config import PROJECT_ROOT, load_config
from semantic_3d_chat.evaluation.llm_navigation_benchmark import (
    canonical_sha256,
    file_sha256,
    finalize_navigation_journal_audit,
    load_task_manifest,
    run_navigation_manifest,
    tree_sha256,
)
from semantic_3d_chat.robot.blender_scanner import SanitizedBlenderScanner
from semantic_3d_chat.robot.llm_tool_policy import (
    ContinuousPrefixGemmaToolBackend,
    LocalGemmaToolPolicy,
    tool_protocol_sha256,
)
from semantic_3d_chat.robot.navigation_policy import (
    LearnedContinuousActionBackend,
    load_navigation_policy_checkpoint,
)
from semantic_3d_chat.robot.navigation_policy_v3 import (
    RUNTIME_INTERLOCK_VERSION,
    SemanticGroundedActionBackendV3,
    load_navigation_policy_v3_checkpoint,
)
from semantic_3d_chat.robot.navigation_policy_v4 import (
    SemanticClearanceActionBackendV4,
    load_navigation_policy_v4_checkpoint,
)
from semantic_3d_chat.robot.runtime_refresh import build_refreshing_embodied_runtime


def _rooted(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return Path(os.path.abspath(value if value.is_absolute() else PROJECT_ROOT / value))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/runtime/embodied_v54.yaml")
    parser.add_argument("--runtime-config", default="configs/runtime/gemma4_v54.yaml")
    parser.add_argument("--scene", default="scene_000001")
    parser.add_argument(
        "--base-checkpoint",
        default="data_gemma4/checkpoints/gemma4_v54_semantic_greedy_gate/update_000",
    )
    parser.add_argument(
        "--runtime-asset",
        default="data/runtime_assets/scene_000001/s_000001.blend",
    )
    parser.add_argument(
        "--robot-state-checkpoint",
        default="data_gemma4/checkpoints/robot_state_numeric_v1",
    )
    parser.add_argument(
        "--tasks",
        default="configs/benchmarks/llm_navigation_scene_000001.json",
    )
    parser.add_argument(
        "--journal",
        default="reports/gemma4/predictions/llm_navigation_scene_000001.json",
    )
    parser.add_argument(
        "--audit-report",
        default="reports/gemma4/metrics/llm_navigation_inference_access.json",
    )
    parser.add_argument(
        "--persistent-map",
        default="data_gemma4/robot_benchmark/scene_000001/semantic_map.npz",
    )
    parser.add_argument("--max-retries", type=int, choices=(0, 1, 2), default=1)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument(
        "--navigation-policy-checkpoint",
        help=(
            "Use a gated supervised continuous-action controller instead of the "
            "untrained Gemma JSON-generation seam"
        ),
    )
    parser.add_argument(
        "--navigation-policy-version",
        type=int,
        choices=(1, 3, 4),
        default=1,
        help="Checkpoint/runtime contract version (default preserves V1 behavior)",
    )
    parser.add_argument("--resume", action="store_true")
    return parser


def _run_contract(
    *,
    config_path: Path,
    runtime_config_path: Path,
    checkpoint: Path,
    runtime_asset: Path,
    robot_state_checkpoint: Path,
    task_source: Path,
    config: dict[str, Any],
    max_retries: int,
    max_new_tokens: int,
    navigation_policy_checkpoint: Path | None,
    navigation_policy_version: int,
) -> dict[str, Any]:
    result = {
        "config_sha256": file_sha256(config_path),
        "runtime_config_sha256": file_sha256(runtime_config_path),
        "base_checkpoint_tree_sha256": tree_sha256(checkpoint),
        "runtime_asset_sha256": file_sha256(runtime_asset),
        "robot_state_checkpoint_tree_sha256": tree_sha256(robot_state_checkpoint),
        "inference_source_sha256": file_sha256(task_source),
        "tool_policy_source_sha256": file_sha256(
            PROJECT_ROOT / "src/semantic_3d_chat/robot/llm_tool_policy.py"
        ),
        "model_id": config["language"]["model_id"],
        "model_revision": config["language"]["revision"],
        "tool_protocol_sha256": tool_protocol_sha256(config),
        "max_retries": max_retries,
        "max_new_tokens": max_new_tokens,
        "fallback_policy": "fail_closed",
        "robot_state_token_training_status": "untrained_numeric_projection",
        "strict_fixed_environment_embedding_input": True,
        "question_conditioned_scene_readout_tokens": False,
    }
    if navigation_policy_checkpoint is not None:
        if navigation_policy_version == 4:
            navigation_source = PROJECT_ROOT / (
                "src/semantic_3d_chat/robot/navigation_policy_v4.py"
            )
            training_status = (
                "supervised_continuous_semantic_clearance_navigation_policy_v4"
            )
        elif navigation_policy_version == 3:
            navigation_source = PROJECT_ROOT / (
                "src/semantic_3d_chat/robot/navigation_policy_v3.py"
            )
            training_status = "supervised_continuous_semantic_grounded_navigation_policy_v3"
        else:
            navigation_source = PROJECT_ROOT / ("src/semantic_3d_chat/robot/navigation_policy.py")
            training_status = "supervised_continuous_navigation_policy_v1"
        result.update(
            {
                "navigation_policy_checkpoint_tree_sha256": tree_sha256(
                    navigation_policy_checkpoint
                ),
                "navigation_policy_source_sha256": file_sha256(navigation_source),
                "tool_policy_training_status": training_status,
            }
        )
        if navigation_policy_version in {3, 4}:
            result.update(
                {
                    "continuous_semantic_grounding_required": True,
                    "all_map_voxels_scored_for_grounding": True,
                    "query_dependent_grounding_navigation_only": True,
                    "oracle_inputs_at_runtime": False,
                    "environmental_text_inputs_at_runtime": [],
                }
            )
        if navigation_policy_version == 3:
            result["navigation_runtime_interlock_version"] = RUNTIME_INTERLOCK_VERSION
        if navigation_policy_version == 4:
            result.update(
                {
                    "numeric_clearance_state_required": True,
                    "clearance_from_sanitized_geometry_only": True,
                    "clearance_ray_count": 24,
                    "clearance_max_range_m": 1.0,
                    "exact_collision_mask_required": True,
                    "unsafe_motion_fallback": "highest_safe_nonterminal_action",
                    "collision_interlock_required": True,
                    "static_scene_prefix_question_independent": True,
                }
            )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config_path = _rooted(args.config)
    runtime_config_path = _rooted(args.runtime_config)
    checkpoint = _rooted(args.base_checkpoint)
    runtime_asset = _rooted(args.runtime_asset)
    robot_state_checkpoint = _rooted(args.robot_state_checkpoint)
    tasks_path = _rooted(args.tasks)
    journal_path = _rooted(args.journal)
    audit_path = _rooted(args.audit_report)
    persistent_map = _rooted(args.persistent_map)
    navigation_policy_checkpoint = (
        None
        if args.navigation_policy_checkpoint is None
        else _rooted(args.navigation_policy_checkpoint)
    )
    source_path = PROJECT_ROOT / "scripts/run_llm_navigation_inference.py"
    forbidden_roots = [
        PROJECT_ROOT / "data" / "oracle",
        PROJECT_ROOT / "data" / "qa",
        PROJECT_ROOT / "data_diverse20" / "qa",
        PROJECT_ROOT / "data_diverse28" / "qa",
        PROJECT_ROOT / "data_gemma4" / "training",
        PROJECT_ROOT / "reports" / "gemma4" / "scorer_only",
    ]
    audit = FileAccessAudit(
        forbidden_roots,
        forbidden_component_names={"oracle", "qa", "scorer_only"},
        block_forbidden=True,
    )
    completed = False
    try:
        with audit:
            manifest = load_task_manifest(tasks_path)
            if manifest.scene_id != args.scene:
                raise ValueError("Opaque scene argument differs from navigation task manifest")
            config = load_config(config_path)
            runtime_config = load_runtime_config(runtime_config_path, record_file=audit.record)
            required_paths = [checkpoint, runtime_asset, robot_state_checkpoint]
            if navigation_policy_checkpoint is not None:
                required_paths.append(navigation_policy_checkpoint)
            for required in required_paths:
                if not required.exists():
                    raise FileNotFoundError(required)
            contract = _run_contract(
                config_path=config_path,
                runtime_config_path=runtime_config_path,
                checkpoint=checkpoint,
                runtime_asset=runtime_asset,
                robot_state_checkpoint=robot_state_checkpoint,
                task_source=source_path,
                config=config,
                max_retries=args.max_retries,
                max_new_tokens=args.max_new_tokens,
                navigation_policy_checkpoint=navigation_policy_checkpoint,
                navigation_policy_version=args.navigation_policy_version,
            )
            chat = StaticChatRuntime.load(
                runtime_config,
                args.scene,
                checkpoint=checkpoint,
                audit=audit,
                local_files_only=True,
            )
            resolution = tuple(int(value) for value in config["render"]["resolution"])
            scanner = SanitizedBlenderScanner(
                args.scene,
                runtime_asset,
                resolution=resolution,
                horizontal_fov_degrees=float(config["render"]["horizontal_fov_degrees"]),
                engine=str(config["render"]["engine"]),
                samples=int(config["render"]["samples"]),
                max_depth_m=float(config["mapping"]["depth_max_m"]),
                output_directory=(persistent_map.parent / "scans"),
            )
            runtime = build_refreshing_embodied_runtime(
                config,
                args.scene,
                checkpoint=checkpoint,
                chat_runtime=chat,
                persistent_map_path=persistent_map,
                observation_scanner=scanner,
                robot_state_checkpoint=robot_state_checkpoint,
                audit=audit,
                local_files_only=True,
            )
            policy_training_status = "untrained_tool_selection_seam"
            proposal_backend: Any = ContinuousPrefixGemmaToolBackend(
                runtime,
                config,
                max_new_tokens=args.max_new_tokens,
            )
            if navigation_policy_checkpoint is not None:
                base = getattr(runtime.prefix_refresher.runtime, "base", None)
                if base is None:
                    base = runtime.prefix_refresher.runtime
                if args.navigation_policy_version == 4:
                    controller, controller_metadata = load_navigation_policy_v4_checkpoint(
                        navigation_policy_checkpoint,
                        expected_hidden_size=int(base.language.hidden_size),
                        expected_model_id=str(config["language"]["model_id"]),
                        expected_model_revision=str(config["language"]["revision"]),
                        device=base.language.device,
                        audit=audit,
                    )
                    proposal_backend = SemanticClearanceActionBackendV4(
                        runtime,
                        controller,
                        controller_metadata,
                        config,
                    )
                    policy_training_status = (
                        "supervised_continuous_semantic_clearance_navigation_policy_v4"
                    )
                elif args.navigation_policy_version == 3:
                    controller, controller_metadata = load_navigation_policy_v3_checkpoint(
                        navigation_policy_checkpoint,
                        expected_hidden_size=int(base.language.hidden_size),
                        expected_model_id=str(config["language"]["model_id"]),
                        expected_model_revision=str(config["language"]["revision"]),
                        device=base.language.device,
                        audit=audit,
                    )
                    proposal_backend = SemanticGroundedActionBackendV3(
                        runtime,
                        controller,
                        controller_metadata,
                        config,
                    )
                    policy_training_status = (
                        "supervised_continuous_semantic_grounded_navigation_policy_v3"
                    )
                else:
                    controller, controller_metadata = load_navigation_policy_checkpoint(
                        navigation_policy_checkpoint,
                        expected_hidden_size=int(base.language.hidden_size),
                        expected_model_id=str(config["language"]["model_id"]),
                        expected_model_revision=str(config["language"]["revision"]),
                        device=base.language.device,
                        audit=audit,
                    )
                    proposal_backend = LearnedContinuousActionBackend(
                        runtime,
                        controller,
                        controller_metadata,
                    )
                    policy_training_status = "supervised_continuous_navigation_policy_v1"
            policy = LocalGemmaToolPolicy(
                proposal_backend,
                config,
                robot_state_provider=runtime.get_robot_state,
                max_retries=args.max_retries,
                fallback_policy="fail_closed",
            )

            def progress(episode: dict[str, Any]) -> None:
                print(
                    json.dumps(
                        {
                            "task_id": episode["task_id"],
                            "termination": episode["termination"],
                            "steps": len(episode["steps"]),
                            "episode_sha256": episode["episode_sha256"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

            result = run_navigation_manifest(
                runtime,
                policy,
                manifest,
                journal_path=journal_path,
                run_contract=contract,
                resume=args.resume,
                runtime_file_audit={"status": "pending_until_runtime_exit"},
                after_episode=progress,
                policy_training_status=policy_training_status,
            )
            audit.assert_clean()
            completed = result["status"] == "complete"
    finally:
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit.save(audit_path)
    audit.assert_clean()
    audit_payload = {
        "passed": not audit.forbidden_accesses(),
        "blocking_enabled": True,
        "loaded_file_count": len(audit.unique_paths),
        "loaded_file_inventory_sha256": canonical_sha256(audit.unique_paths),
        "forbidden_accesses": audit.forbidden_accesses(),
        "audit_report_sha256": file_sha256(audit_path),
    }
    if completed:
        result = finalize_navigation_journal_audit(journal_path, audit_payload)
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "task_count": len(result["episodes"]),
                    "journal": str(journal_path),
                    "journal_sha256": result["journal_sha256"],
                    "oracle_or_labels_loaded": False,
                    "tool_policy_training_status": policy_training_status,
                },
                indent=2,
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
