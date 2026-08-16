"""Local conversational QA and bounded navigation over one continuous 3D memory."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections.abc import Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any

from semantic_3d_chat.chat.file_audit import FileAccessAudit
from semantic_3d_chat.chat.question_control_runtime import QuestionControlledChatRuntime
from semantic_3d_chat.chat.runtime import StaticChatRuntime
from semantic_3d_chat.chat.runtime_config import load_runtime_config
from semantic_3d_chat.config import PROJECT_ROOT, load_config, project_path, reports_root
from semantic_3d_chat.robot.blender_scanner import SanitizedBlenderScanner
from semantic_3d_chat.robot.conversation import (
    ConversationalEmbodiedAgent,
    should_offer_llm_tool_policy,
)
from semantic_3d_chat.robot.conversation_output import render_startup, render_turn
from semantic_3d_chat.robot.gemma4_tool_decoder_v2_integration import (
    inspect_promoted_gemma_tool_decoder_v2,
    load_promoted_gemma_tool_decoder_v2,
)
from semantic_3d_chat.robot.llm_tool_policy import (
    ContinuousPrefixGemmaToolBackend,
    LocalGemmaToolPolicy,
)
from semantic_3d_chat.robot.navigation_policy import (
    LearnedContinuousActionBackend,
    load_navigation_policy_checkpoint,
)
from semantic_3d_chat.robot.navigation_policy_v3 import (
    TRAINING_STATUS as NAVIGATION_POLICY_V3_TRAINING_STATUS,
)
from semantic_3d_chat.robot.navigation_policy_v3 import (
    SemanticGroundedActionBackendV3,
    load_navigation_policy_v3_checkpoint,
)
from semantic_3d_chat.robot.navigation_policy_v4 import (
    TRAINING_STATUS as NAVIGATION_POLICY_V4_TRAINING_STATUS,
)
from semantic_3d_chat.robot.navigation_policy_v4 import (
    SemanticClearanceActionBackendV4,
    load_navigation_policy_v4_checkpoint,
)
from semantic_3d_chat.robot.runtime_refresh import build_refreshing_embodied_runtime
from semantic_3d_chat.robot.semantic_agent import GemmaProjectedTextEncoder

NAVIGATION_POLICY_V1_TRAINING_STATUS = "supervised_continuous_navigation_policy_v1"


def _rooted(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (PROJECT_ROOT / value).resolve()


def _write(payload: object) -> None:
    print(json.dumps(payload, sort_keys=True, allow_nan=False), flush=True)


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/runtime/embodied_live.yaml")
    parser.add_argument("--control-runtime-config", default="configs/runtime/gemma4_v54.yaml")
    parser.add_argument("--scene", default="scene_000001")
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument(
        "--control-checkpoint",
        help=(
            "Optional enhanced question-conditioned readout. Omit it for the strict "
            "fixed environment-prefix runtime."
        ),
    )
    parser.add_argument(
        "--grounding-checkpoint",
        help="Optional authenticated V78 numeric-grounding diagnostic.",
    )
    parser.add_argument("--runtime-asset", required=True)
    parser.add_argument("--robot-state-checkpoint", required=True)
    parser.add_argument("--persistent-map")
    parser.add_argument("--audit-report")
    parser.add_argument(
        "--result-output",
        help="Atomically save finite --command startup and turn evidence as JSON.",
    )
    policy = parser.add_mutually_exclusive_group()
    policy.add_argument(
        "--llm-tool-policy",
        action="store_true",
        help=(
            "Opt in to untrained local-Gemma JSON action selection over the continuous "
            "scene+robot prefix."
        ),
    )
    policy.add_argument(
        "--navigation-policy-checkpoint",
        help=(
            "Sanitized task-trained navigation-controller checkpoint. This is "
            "mutually exclusive with the untrained --llm-tool-policy seam."
        ),
    )
    policy.add_argument(
        "--gemma-tool-decoder-checkpoint",
        help=(
            "Promoted task-trained Gemma JSON tool-decoder checkpoint. It consumes "
            "the complete scene prefix plus numeric robot, target, and clearance tokens."
        ),
    )
    parser.add_argument(
        "--navigation-policy-version",
        type=int,
        choices=(1, 3, 4),
        default=3,
        help="Learned-controller checkpoint schema (default: 3).",
    )
    parser.add_argument(
        "--llm-tool-max-retries",
        type=int,
        choices=(0, 1, 2),
        default=1,
    )
    parser.add_argument(
        "--llm-tool-fallback",
        choices=("fail_closed", "deterministic_parser"),
        default="fail_closed",
    )
    parser.add_argument(
        "--debug-llm-tool-errors",
        action="store_true",
        help=(
            "Developer diagnostic: propagate a local tool-generation exception with a "
            "traceback instead of converting it to the fail-closed E_GENERATION code."
        ),
    )
    parser.add_argument(
        "--navigation-max-steps",
        type=int,
        choices=tuple(range(1, 33)),
        default=12,
        help=(
            "Maximum refreshed policy/action iterations for one learned-navigation "
            "instruction (default: 12; ignored by older one-step modes)."
        ),
    )
    parser.add_argument(
        "--command",
        action="append",
        default=[],
        help="Execute one natural-language turn (repeatable); otherwise enter interactive mode.",
    )
    parser.add_argument(
        "--human",
        action="store_true",
        help=(
            "Render concise human-facing output. Interactive mode enables this "
            "automatically; finite --command runs remain JSON unless requested."
        ),
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help=(
            "Validate the local configuration and learned-policy checkpoint without "
            "loading Gemma, Blender, or changing robot/map state."
        ),
    )
    return parser


def _runtime_file_audit() -> FileAccessAudit:
    """Create the fail-before-open audit used by every conversational mode."""

    return FileAccessAudit(
        [
            PROJECT_ROOT / "data" / "oracle",
            PROJECT_ROOT / "data" / "qa",
            PROJECT_ROOT / "data_gemma4" / "training",
            PROJECT_ROOT / "reports" / "gemma4" / "scorer_only",
        ],
        forbidden_component_names=frozenset({"oracle", "qa", "training", "scorer_only"}),
        block_forbidden=True,
    )


def _configured_model_binding(config: dict[str, Any]) -> tuple[int, str, str]:
    """Return the pinned continuous-prefix width and local Gemma identity."""

    scene_encoder = config.get("scene_encoder")
    language = config.get("language")
    if not isinstance(scene_encoder, dict) or not isinstance(language, dict):
        raise TypeError("Embodied configuration has no scene-encoder/language contract")
    hidden_size = scene_encoder.get("language_aligned_tail_dim")
    model_id = language.get("model_id")
    model_revision = language.get("revision")
    if (
        isinstance(hidden_size, bool)
        or not isinstance(hidden_size, int)
        or hidden_size < 1
        or not isinstance(model_id, str)
        or not model_id
        or not isinstance(model_revision, str)
        or not model_revision
    ):
        raise ValueError("Embodied configuration has an invalid local-model binding")
    return hidden_size, model_id, model_revision


def _runtime_model_binding(runtime: Any) -> tuple[int, str, str]:
    wrapped = runtime.prefix_refresher.runtime
    base = getattr(wrapped, "base", wrapped)
    language = getattr(base, "language", None)
    config = getattr(base, "config", None)
    if language is None or not isinstance(config, dict):
        raise TypeError("Learned navigation requires the loaded local language runtime")
    language_config = config.get("language")
    if not isinstance(language_config, dict):
        raise TypeError("Loaded runtime has no pinned language configuration")
    model_id = language_config.get("model_id")
    model_revision = language_config.get("revision")
    if not isinstance(model_id, str) or not isinstance(model_revision, str):
        raise TypeError("Loaded runtime language identity is invalid")
    return int(language.hidden_size), model_id, model_revision


def _load_navigation_controller(
    checkpoint: str | Path,
    version: int,
    *,
    hidden_size: int,
    model_id: str,
    model_revision: str,
    audit: FileAccessAudit,
) -> tuple[Any, dict[str, Any]]:
    loader = {
        1: load_navigation_policy_checkpoint,
        3: load_navigation_policy_v3_checkpoint,
        4: load_navigation_policy_v4_checkpoint,
    }.get(version)
    if loader is None:
        raise ValueError("navigation policy version must be 1, 3, or 4")
    controller, metadata = loader(
        _rooted(checkpoint),
        expected_hidden_size=hidden_size,
        expected_model_id=model_id,
        expected_model_revision=model_revision,
        device="cpu",
        audit=audit,
    )
    return controller, dict(metadata)


def _load_navigation_backend(
    runtime: Any,
    config: dict[str, Any],
    checkpoint: str | Path,
    version: int,
    *,
    audit: FileAccessAudit,
) -> tuple[Any, dict[str, Any]]:
    hidden_size, model_id, model_revision = _runtime_model_binding(runtime)
    controller, metadata = _load_navigation_controller(
        checkpoint,
        version,
        hidden_size=hidden_size,
        model_id=model_id,
        model_revision=model_revision,
        audit=audit,
    )
    if version == 1:
        backend = LearnedContinuousActionBackend(runtime, controller, metadata)
    elif version == 3:
        backend = SemanticGroundedActionBackendV3(
            runtime,
            controller,
            metadata,
            config,
        )
    elif version == 4:
        backend = SemanticClearanceActionBackendV4(
            runtime,
            controller,
            metadata,
            config,
        )
    else:  # Kept explicit even though argparse and the loader both fail closed.
        raise ValueError("navigation policy version must be 1, 3, or 4")
    return backend, metadata


def _navigation_policy_summary(
    checkpoint: str | Path | None,
    version: int,
    metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    enabled = checkpoint is not None
    is_v3 = enabled and version == 3
    is_v4 = enabled and version == 4
    values = metadata or {}
    return {
        "enabled": enabled,
        "version": version if enabled else None,
        "checkpoint": str(_rooted(checkpoint)) if checkpoint is not None else None,
        "task_trained": bool(values.get("task_trained")) if enabled else False,
        "training_status": (
            NAVIGATION_POLICY_V4_TRAINING_STATUS
            if is_v4
            else NAVIGATION_POLICY_V3_TRAINING_STATUS
            if is_v3
            else NAVIGATION_POLICY_V1_TRAINING_STATUS
            if enabled
            else None
        ),
        "complete_scene_prefix_required": bool(values.get("complete_scene_prefix_required")),
        "every_scene_token_processed": bool(values.get("every_scene_token_processed")),
        "numeric_robot_tokens_required": bool(values.get("numeric_robot_tokens_required")),
        "continuous_semantic_grounding": bool(values.get("continuous_semantic_grounding_required")),
        "all_map_voxels_scored_for_grounding": bool(
            values.get("all_map_voxels_scored_for_grounding")
        ),
        "query_dependent_grounding_navigation_only": bool(
            values.get("query_dependent_grounding_navigation_only")
        ),
        "numeric_clearance_state": bool(values.get("numeric_clearance_state_required")),
        "exact_collision_mask": bool(values.get("exact_collision_mask_required")),
        "grounding_performed_at_startup": False,
        "oracle_inputs_at_runtime": values.get("oracle_inputs_at_runtime", False),
        "environmental_text_inputs": list(values.get("environmental_text_inputs", [])),
    }


def _gemma_tool_decoder_summary(
    checkpoint: str | Path | None,
    metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    enabled = checkpoint is not None
    values = metadata or {}
    return {
        "enabled": enabled,
        "checkpoint": str(_rooted(checkpoint)) if checkpoint is not None else None,
        "task_trained": values.get("task_trained") is True if enabled else False,
        "training_status": values.get("training_status") if enabled else None,
        "promoted_runtime": values.get("status") == "promoted_runtime" if enabled else False,
        "saved_runtime_execution_gate_passed": (
            values.get("saved_runtime_execution_gate_passed") is True if enabled else False
        ),
        "complete_scene_prefix_required": bool(values.get("complete_scene_prefix_required")),
        "question_independent_static_scene_prefix_required": bool(
            values.get("question_independent_static_scene_prefix_required")
        ),
        "numeric_robot_tokens_required": bool(values.get("numeric_robot_tokens_required")),
        "continuous_target_tokens_required": bool(values.get("continuous_target_tokens_required")),
        "numeric_clearance_tokens_required": bool(values.get("numeric_clearance_tokens_required")),
        "collision_interlock_required": bool(values.get("collision_interlock_required")),
        "adapter_inactive_outside_tool_generation": enabled,
        "oracle_inputs_at_runtime": values.get("oracle_inputs_at_runtime", False),
        "environmental_text_inputs": list(values.get("environmental_text_inputs", [])),
    }


def _preflight(
    config: dict[str, Any],
    checkpoint: str | Path | None,
    version: int,
    *,
    gemma_tool_decoder_checkpoint: str | Path | None = None,
    base_checkpoint: str | Path | None = None,
    audit: FileAccessAudit,
) -> dict[str, Any]:
    metadata: dict[str, Any] | None = None
    if checkpoint is not None:
        hidden_size, model_id, model_revision = _configured_model_binding(config)
        _controller, metadata = _load_navigation_controller(
            checkpoint,
            version,
            hidden_size=hidden_size,
            model_id=model_id,
            model_revision=model_revision,
            audit=audit,
        )
    tool_metadata: dict[str, Any] | None = None
    if gemma_tool_decoder_checkpoint is not None:
        hidden_size, model_id, model_revision = _configured_model_binding(config)
        if hidden_size != 1536:
            raise ValueError("Gemma tool decoder requires a 1536-wide scene prefix")
        tool_metadata = inspect_promoted_gemma_tool_decoder_v2(
            gemma_tool_decoder_checkpoint,
            expected_model_id=model_id,
            expected_model_revision=model_revision,
            base_checkpoint=base_checkpoint,
            audit=audit,
        )
    return {
        "phase": "embodied_conversation_preflight",
        "ready": True,
        "loads_language_model": False,
        "changes_robot_or_map_state": False,
        "navigation_policy": _navigation_policy_summary(
            checkpoint,
            version,
            metadata,
        ),
        "gemma_tool_decoder": _gemma_tool_decoder_summary(
            gemma_tool_decoder_checkpoint,
            tool_metadata,
        ),
    }


def _startup(
    runtime: Any,
    scene_id: str,
    *,
    llm_tool_policy: bool = False,
    llm_tool_max_retries: int = 1,
    llm_tool_fallback: str = "fail_closed",
    enhanced_question_conditioned_readout: bool = False,
    navigation_policy_checkpoint: str | Path | None = None,
    navigation_policy_version: int = 3,
    navigation_policy_metadata: dict[str, Any] | None = None,
    gemma_tool_decoder_checkpoint: str | Path | None = None,
    gemma_tool_decoder_metadata: dict[str, Any] | None = None,
    navigation_backend: Any | None = None,
    navigation_max_steps: int = 12,
) -> dict[str, Any]:
    base = runtime.prefix_refresher.runtime
    navigation = _navigation_policy_summary(
        navigation_policy_checkpoint,
        navigation_policy_version,
        navigation_policy_metadata,
    )
    tool_decoder = _gemma_tool_decoder_summary(
        gemma_tool_decoder_checkpoint,
        gemma_tool_decoder_metadata,
    )
    policy_enabled = llm_tool_policy or navigation["enabled"] or tool_decoder["enabled"]
    convergence_summary = getattr(
        navigation_backend,
        "numeric_alignment_interlock_summary",
        None,
    )
    if callable(convergence_summary):
        navigation["numeric_alignment_convergence_interlock"] = convergence_summary()
    training_status = (
        tool_decoder["training_status"]
        if tool_decoder["enabled"]
        else navigation["training_status"]
        if navigation["enabled"]
        else "untrained_tool_selection_seam"
    )
    return {
        "phase": "embodied_conversation_ready",
        "scene_id": scene_id,
        "runtime": base.startup_summary(),
        "prefix_binding": runtime.prefix_binding(),
        "scene_prefix_computed_before_question": True,
        "environmental_text_inputs": [],
        "local_inference": True,
        "bounded_action_protocol": True,
        "strict_fixed_environment_embedding_input": (not enhanced_question_conditioned_readout),
        "question_conditioned_scene_readout_tokens": (enhanced_question_conditioned_readout),
        "llm_tool_policy": {
            "enabled": policy_enabled,
            "backend": (
                "trained_gemma_tool_decoder_v2"
                if tool_decoder["enabled"]
                else f"learned_navigation_v{navigation_policy_version}"
                if navigation["enabled"]
                else "local_gemma_json"
                if llm_tool_policy
                else "disabled"
            ),
            "local_inference": True,
            "continuous_scene_and_robot_prefix": policy_enabled,
            "training_status": training_status,
            "max_retries": llm_tool_max_retries,
            "fallback_policy": llm_tool_fallback,
            "malformed_output_executes": False,
        },
        "navigation_policy": navigation,
        "gemma_tool_decoder": tool_decoder,
        "learned_navigation_closed_loop": {
            "enabled": bool(navigation["enabled"] or tool_decoder["enabled"]),
            "max_steps": navigation_max_steps,
            "reselects_after_each_refreshed_prefix": bool(
                navigation["enabled"] or tool_decoder["enabled"]
            ),
            "termination_conditions": [
                "stop",
                "numeric_alignment_goal_converged",
                "numeric_approach_goal_converged",
                "action_failure",
                "policy_rejection",
                "max_steps",
            ],
        },
    }


def _grounding_attestation(
    backend: Any,
    previous: object,
) -> dict[str, Any] | None:
    current = getattr(backend, "last_grounding", None)
    if current is previous or not isinstance(current, dict):
        return None
    result = dict(current)
    target_available = result.get("target_available") is True
    scored = result.get("scored_voxels")
    result["all_map_voxels_scored"] = bool(
        target_available and isinstance(scored, int) and not isinstance(scored, bool) and scored > 0
    )
    result["environmental_text_inputs"] = []
    result["oracle_inputs_at_runtime"] = False
    return result


def _prefix_refresh_verified(steps: list[dict[str, Any]]) -> bool:
    for previous, current in pairwise(steps):
        binding = previous.get("prefix_binding")
        selection = current.get("tool_selection")
        if not isinstance(binding, dict) or not isinstance(selection, dict):
            return False
        if binding.get("active_prefix_sha256") != selection.get("active_prefix_sha256"):
            return False
    return True


def _handle_conversation_turn(
    agent: ConversationalEmbodiedAgent,
    text: str,
    *,
    navigation_backend: Any | None,
    navigation_max_steps: int,
) -> dict[str, Any]:
    """Run learned navigation to a bounded terminal state over refreshed prefixes.

    Strict QA, deterministic parsing, and the untrained Gemma seam keep their
    historical one-call behavior. An explicitly loaded learned controller or
    promoted Gemma tool decoder may reselect after every refreshed prefix.
    """

    if navigation_backend is None or not should_offer_llm_tool_policy(text):
        return agent.handle(text)

    steps: list[dict[str, Any]] = []
    receipts: list[dict[str, Any]] = []
    termination = "max_steps"
    for index in range(navigation_max_steps):
        previous_grounding = getattr(navigation_backend, "last_grounding", None)
        result = agent.handle(text)
        result["closed_loop_step"] = index + 1
        if grounding := _grounding_attestation(
            navigation_backend,
            previous_grounding,
        ):
            result["continuous_grounding"] = grounding
        steps.append(result)
        action_receipts = result.get("action_receipts")
        if isinstance(action_receipts, list):
            receipts.extend(
                dict(receipt) for receipt in action_receipts if isinstance(receipt, dict)
            )
        if result.get("success") is not True:
            termination = (
                "policy_rejection"
                if result.get("error_code") == "E_TOOL_POLICY_REJECTED"
                else "action_failure"
            )
            break
        if result.get("command") == "stop":
            termination = "stop"
            break

    return {
        "kind": "learned_navigation_closed_loop",
        "success": termination == "stop",
        "termination_reason": termination,
        "step_count": len(steps),
        "max_steps": navigation_max_steps,
        "request_sha256": steps[0]["request_sha256"] if steps else None,
        "steps": steps,
        "action_receipts": receipts,
        "continuous_grounding_attestations": [
            step["continuous_grounding"]
            for step in steps
            if isinstance(step.get("continuous_grounding"), dict)
        ],
        "prefix_refresh_verified": _prefix_refresh_verified(steps),
        "prefix_binding": agent.runtime.prefix_binding(),
        "environmental_text_inputs": [],
        # The static environment prefix remains precomputed and unchanged by
        # the instruction.  Embodied target grounding is intentionally
        # question dependent: it scores every continuous voxel and forms a
        # numeric target state.  It is not the primary static-QA retrieval path.
        "static_scene_prefix_question_independent": True,
        "question_dependent_navigation_grounding": True,
        "primary_static_scene_retrieval": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.result_output is not None and not args.command:
        raise ValueError("--result-output requires at least one finite --command")
    audit = _runtime_file_audit()
    audit_report: Path | None = None
    finite_startup: dict[str, Any] | None = None
    finite_turns: list[dict[str, Any]] = []
    try:
        with audit:
            config = load_config(_rooted(args.config))
            if args.check:
                _write(
                    _preflight(
                        config,
                        args.navigation_policy_checkpoint,
                        args.navigation_policy_version,
                        gemma_tool_decoder_checkpoint=(args.gemma_tool_decoder_checkpoint),
                        base_checkpoint=args.base_checkpoint,
                        audit=audit,
                    )
                )
                audit.assert_clean()
                return 0
            control_config = load_runtime_config(
                _rooted(args.control_runtime_config),
                record_file=audit.record,
            )
            if args.audit_report is None:
                audit_report = (
                    reports_root(config)
                    / "metrics"
                    / f"embodied_conversation_access_{args.scene}.json"
                )
            else:
                audit_report = _rooted(args.audit_report)
            if args.control_checkpoint is None:
                chat = StaticChatRuntime.load(
                    control_config,
                    args.scene,
                    checkpoint=_rooted(args.base_checkpoint),
                    audit=audit,
                    local_files_only=True,
                )
            else:
                chat = QuestionControlledChatRuntime.load(
                    control_config,
                    args.scene,
                    base_checkpoint=_rooted(args.base_checkpoint),
                    control_checkpoint=_rooted(args.control_checkpoint),
                    grounding_checkpoint=(
                        None
                        if args.grounding_checkpoint is None
                        else _rooted(args.grounding_checkpoint)
                    ),
                    audit=audit,
                )
            resolution = tuple(int(value) for value in config["render"]["resolution"])
            scanner = SanitizedBlenderScanner(
                args.scene,
                _rooted(args.runtime_asset),
                resolution=resolution,
                horizontal_fov_degrees=float(config["render"]["horizontal_fov_degrees"]),
                engine=str(config["render"]["engine"]),
                samples=int(config["render"]["samples"]),
                max_depth_m=float(config["mapping"]["depth_max_m"]),
                output_directory=project_path(config, "robot", args.scene, "scans"),
            )
            runtime = build_refreshing_embodied_runtime(
                config,
                args.scene,
                checkpoint=_rooted(args.base_checkpoint),
                chat_runtime=chat,
                persistent_map_path=(
                    None if args.persistent_map is None else _rooted(args.persistent_map)
                ),
                observation_scanner=scanner,
                robot_state_checkpoint=_rooted(args.robot_state_checkpoint),
                audit=audit,
                local_files_only=True,
            )
            tool_policy = None
            navigation_backend = None
            navigation_metadata: dict[str, Any] | None = None
            tool_decoder_metadata: dict[str, Any] | None = None
            text_encoder = GemmaProjectedTextEncoder.from_config(config)
            if args.llm_tool_policy:
                tool_policy = LocalGemmaToolPolicy(
                    ContinuousPrefixGemmaToolBackend(runtime, config),
                    config,
                    robot_state_provider=runtime.get_robot_state,
                    max_retries=args.llm_tool_max_retries,
                    fallback_policy=args.llm_tool_fallback,
                    propagate_backend_exceptions=args.debug_llm_tool_errors,
                )
            elif args.gemma_tool_decoder_checkpoint is not None:
                navigation_backend, tool_decoder_metadata = load_promoted_gemma_tool_decoder_v2(
                    runtime,
                    config,
                    args.gemma_tool_decoder_checkpoint,
                    audit=audit,
                    text_encoder=text_encoder,
                )
                tool_policy = LocalGemmaToolPolicy(
                    navigation_backend,
                    config,
                    robot_state_provider=runtime.get_robot_state,
                    max_retries=args.llm_tool_max_retries,
                    fallback_policy=args.llm_tool_fallback,
                    propagate_backend_exceptions=args.debug_llm_tool_errors,
                )
            elif args.navigation_policy_checkpoint is not None:
                navigation_backend, navigation_metadata = _load_navigation_backend(
                    runtime,
                    config,
                    args.navigation_policy_checkpoint,
                    args.navigation_policy_version,
                    audit=audit,
                )
                tool_policy = LocalGemmaToolPolicy(
                    navigation_backend,
                    config,
                    robot_state_provider=runtime.get_robot_state,
                    max_retries=args.llm_tool_max_retries,
                    fallback_policy=args.llm_tool_fallback,
                    propagate_backend_exceptions=args.debug_llm_tool_errors,
                )
            agent = ConversationalEmbodiedAgent(
                runtime,
                text_encoder,
                room_size_m=config["scene"]["room_size_m"],
                tool_policy=tool_policy,
            )
            startup = _startup(
                runtime,
                args.scene,
                llm_tool_policy=args.llm_tool_policy,
                llm_tool_max_retries=args.llm_tool_max_retries,
                llm_tool_fallback=args.llm_tool_fallback,
                enhanced_question_conditioned_readout=(args.control_checkpoint is not None),
                navigation_policy_checkpoint=args.navigation_policy_checkpoint,
                navigation_policy_version=args.navigation_policy_version,
                navigation_policy_metadata=navigation_metadata,
                gemma_tool_decoder_checkpoint=args.gemma_tool_decoder_checkpoint,
                gemma_tool_decoder_metadata=tool_decoder_metadata,
                navigation_backend=navigation_backend,
                navigation_max_steps=args.navigation_max_steps,
            )
            finite_startup = startup
            human_output = args.human or not args.command
            print(render_startup(startup), flush=True) if human_output else _write(startup)
            if args.command:
                for command in args.command:
                    result = _handle_conversation_turn(
                        agent,
                        command,
                        navigation_backend=navigation_backend,
                        navigation_max_steps=args.navigation_max_steps,
                    )
                    finite_turns.append(result)
                    print(render_turn(result), flush=True) if human_output else _write(result)
            else:
                while True:
                    try:
                        text = input("you> ").strip()
                    except (EOFError, KeyboardInterrupt):
                        print()
                        break
                    if not text:
                        continue
                    if text.casefold() in {"quit", "exit"}:
                        break
                    result = _handle_conversation_turn(
                        agent,
                        text,
                        navigation_backend=navigation_backend,
                        navigation_max_steps=args.navigation_max_steps,
                    )
                    print(render_turn(result), flush=True)
            audit.assert_clean()
    finally:
        if audit_report is not None:
            audit_report.parent.mkdir(parents=True, exist_ok=True)
            audit.save(audit_report)
    audit.assert_clean()
    if args.result_output is not None:
        if finite_startup is None or not finite_turns:
            raise RuntimeError("finite embodied evidence was not produced")
        _atomic_json(
            _rooted(args.result_output),
            {
                "schema": "semantic_3d_chat.embodied_conversation_result.v1",
                "scene_id": args.scene,
                "passed_runtime_audit": True,
                "startup": finite_startup,
                "turns": finite_turns,
                "audit_report": None if audit_report is None else str(audit_report),
                "loaded_file_count": len(audit.unique_paths),
                "forbidden_access_count": len(audit.forbidden_accesses()),
                "environmental_text_inputs": [],
            },
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
