"""Task-trained, full-scene goal policy for the interactive local rover.

This module is the small integration seam that the practical UI was missing.
It loads the accepted V3 navigation checkpoint, routes it through the V3.3
runtime interlocks, and runs a bounded closed loop.  The first policy decision
uses the already-built full-scene prefix; this module never requests a camera
observation before reasoning.  Goal execution is deliberately static-map:
camera scans are forbidden, while numeric robot-state tokens are refreshed
after motion and the precomputed scene prefix remains byte-identical.

No target/category inventory, caption, scene graph, simulator label, oracle
file, or QA artifact is accepted.  A target phrase comes only from the user's
instruction and is never returned in the numeric policy result.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal

from semantic_3d_chat.chat.file_audit import FileAccessAudit
from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.robot.llm_tool_policy import (
    LocalGemmaToolPolicy,
    ToolPolicyDecision,
    execute_validated_tool_call,
)
from semantic_3d_chat.robot.navigation_policy_v3 import (
    TRAINING_STATUS,
    load_navigation_policy_v3_checkpoint,
)
from semantic_3d_chat.robot.navigation_policy_v3_3 import (
    SemanticGroundedActionBackendV33,
)
from semantic_3d_chat.robot.semantic_agent import ContinuousTextEncoder

DEFAULT_TRAINED_GOAL_CHECKPOINT: Final[str] = (
    "data_gemma4/checkpoints/navigation_policy_v3"
)
_BLOCKED_COMPONENTS: Final[frozenset[str]] = frozenset(
    {"oracle", "qa", "training", "scorer_only", "scorer-only"}
)
_GOAL_KINDS: Final[frozenset[str]] = frozenset({"face", "approach"})


def _rooted(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    rooted = candidate if candidate.is_absolute() else PROJECT_ROOT / candidate
    result = Path(os.path.abspath(rooted))
    if _BLOCKED_COMPONENTS & {part.casefold() for part in result.parts}:
        raise ValueError("Goal-policy checkpoint cannot enter a protected data tree")
    current = Path(result.anchor)
    for component in result.parts[1:]:
        current /= component
        if current.is_symlink():
            raise ValueError("Goal-policy checkpoint path cannot contain symbolic links")
    return result


def _runtime_model_binding(runtime: Any) -> tuple[int, str, str]:
    refresher = getattr(runtime, "prefix_refresher", None)
    wrapped = getattr(refresher, "runtime", None)
    base = getattr(wrapped, "base", wrapped)
    language = getattr(base, "language", None)
    config = getattr(base, "config", None)
    if language is None or not isinstance(config, Mapping):
        raise TypeError("Trained goal policy requires the loaded local language runtime")
    language_config = config.get("language")
    if not isinstance(language_config, Mapping):
        raise TypeError("Loaded runtime has no pinned language configuration")
    model_id = language_config.get("model_id")
    revision = language_config.get("revision")
    hidden_size = getattr(language, "hidden_size", None)
    if (
        isinstance(hidden_size, bool)
        or not isinstance(hidden_size, int)
        or hidden_size < 1
        or not isinstance(model_id, str)
        or not model_id
        or not isinstance(revision, str)
        or not revision
    ):
        raise ValueError("Loaded local language-model binding is invalid")
    return hidden_size, model_id, revision


@dataclass(frozen=True)
class TrainedGoalPolicyBundle:
    """Authenticated learned policy plus its full-scene runtime backend."""

    policy: LocalGemmaToolPolicy
    backend: SemanticGroundedActionBackendV33
    metadata: dict[str, Any]
    checkpoint: Path

    def summary(self) -> dict[str, Any]:
        return {
            "task_trained": True,
            "training_status": TRAINING_STATUS,
            "runtime_interlock_version": self.backend.runtime_interlock_version,
            "scene_token_count": int(self.metadata["scene_token_count"]),
            "robot_token_count": int(self.metadata["robot_token_count"]),
            "every_scene_token_processed": True,
            "all_map_voxels_scored_for_target_grounding": True,
            "question_dependent_scene_selection": False,
            "current_camera_observation_required_before_first_decision": False,
            "camera_observations_during_goal": 0,
            "static_precomputed_scene_memory": True,
            "environmental_text_inputs": [],
            "oracle_inputs_at_runtime": False,
        }


def load_trained_goal_policy(
    runtime: Any,
    config: Mapping[str, Any],
    *,
    checkpoint: str | Path = DEFAULT_TRAINED_GOAL_CHECKPOINT,
    text_encoder: ContinuousTextEncoder | None = None,
    device: str = "cpu",
    audit: FileAccessAudit | None = None,
) -> TrainedGoalPolicyBundle:
    """Load the accepted V3 weights and bind the V3.3 closed-loop backend."""

    if not isinstance(config, Mapping):
        raise TypeError("Goal-policy configuration must be a mapping")
    hidden_size, model_id, revision = _runtime_model_binding(runtime)
    root = _rooted(checkpoint)
    controller, metadata = load_navigation_policy_v3_checkpoint(
        root,
        expected_hidden_size=hidden_size,
        expected_model_id=model_id,
        expected_model_revision=revision,
        device=device,
        audit=audit,
    )
    required_true = (
        "task_trained",
        "complete_scene_prefix_required",
        "question_independent_static_scene_prefix_required",
        "every_scene_token_processed",
        "numeric_robot_tokens_required",
        "continuous_semantic_grounding_required",
        "all_map_voxels_scored_for_grounding",
    )
    if (
        any(metadata.get(field) is not True for field in required_true)
        or metadata.get("environmental_text_inputs") != []
        or metadata.get("oracle_inputs_at_runtime") is not False
    ):
        raise ValueError("Navigation checkpoint does not satisfy the trained goal contract")
    backend = SemanticGroundedActionBackendV33(
        runtime,
        controller,
        dict(metadata),
        dict(config),
        text_encoder=text_encoder,
    )
    policy = LocalGemmaToolPolicy(
        backend,
        config,
        robot_state_provider=runtime.get_robot_state,
        max_retries=0,
        fallback_policy="fail_closed",
    )
    return TrainedGoalPolicyBundle(
        policy=policy,
        backend=backend,
        metadata=dict(metadata),
        checkpoint=root,
    )


def canonical_terminal_goal(
    kind: Literal["face", "approach"],
    target_text: str,
) -> str:
    """Create the V3 terminal envelope from only a user-supplied target phrase."""

    if kind not in _GOAL_KINDS:
        raise ValueError("Unsupported semantic goal kind")
    if not isinstance(target_text, str):
        raise TypeError("Semantic goal target must be text")
    target = " ".join(target_text.strip().split()).rstrip(".!?")
    if not target or len(target) > 256 or "\n" in target or "\r" in target:
        raise ValueError("Semantic goal target phrase is invalid")
    if kind == "face":
        return f"Face {target}, then stop."
    if kind == "approach":
        return f"Approach {target}, then stop."
    raise AssertionError("Validated semantic goal kind was not handled")


def _binding(runtime: Any) -> dict[str, Any]:
    value = runtime.prefix_binding()
    if not isinstance(value, Mapping):
        raise TypeError("Goal runtime returned an invalid continuous-prefix binding")
    required = (
        "active_prefix_sha256",
        "scene_prefix_sha256",
        "robot_tokens_sha256",
    )
    if any(not isinstance(value.get(field), str) for field in required):
        raise ValueError("Goal runtime prefix binding is incomplete")
    return dict(value)


def _decision_matches_binding(
    decision: ToolPolicyDecision,
    binding: Mapping[str, Any],
) -> bool:
    return bool(
        decision.active_prefix_sha256 == binding.get("active_prefix_sha256")
        and decision.scene_prefix_sha256 == binding.get("scene_prefix_sha256")
        and decision.robot_tokens_sha256 == binding.get("robot_tokens_sha256")
        and decision.training_status == TRAINING_STATUS
    )


def _complete_grounding(
    grounding: Mapping[str, Any] | None,
    binding: Mapping[str, Any],
) -> bool:
    """Verify that target selection scored the exact bound full voxel map."""

    if not isinstance(grounding, Mapping):
        return False
    source_voxels = binding.get("source_voxels")
    scored_voxels = grounding.get("scored_voxels")
    return bool(
        grounding.get("target_available") is True
        and grounding.get("continuous_context_verified") is True
        and isinstance(source_voxels, int)
        and not isinstance(source_voxels, bool)
        and source_voxels > 0
        and isinstance(scored_voxels, int)
        and not isinstance(scored_voxels, bool)
        and scored_voxels == source_voxels
        and grounding.get("active_prefix_sha256")
        == binding.get("active_prefix_sha256")
        and grounding.get("scene_prefix_sha256")
        == binding.get("scene_prefix_sha256")
        and grounding.get("robot_tokens_sha256")
        == binding.get("robot_tokens_sha256")
    )


def _active_robot_context_advanced(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> bool:
    """Return whether motion refreshed robot tokens without changing scene memory."""

    return bool(
        after.get("active_prefix_sha256") != before.get("active_prefix_sha256")
        and after.get("robot_tokens_sha256") != before.get("robot_tokens_sha256")
    )


def _same_static_scene(
    expected: Mapping[str, Any],
    observed: Mapping[str, Any],
) -> bool:
    """Require the map-backed scene memory to stay byte-identical during control."""

    fields = (
        "scene_id",
        "map_version",
        "map_sha256",
        "scene_prefix_sha256",
        "source_voxels",
        "processed_voxels",
    )
    return all(observed.get(field) == expected.get(field) for field in fields)


def _grounding_snapshot(backend: Any) -> dict[str, Any] | None:
    value = getattr(backend, "last_grounding", None)
    if not isinstance(value, Mapping):
        return None
    permitted = {
        "target_available",
        "target_xyz_m",
        "target_state_sha256",
        "query_embedding_sha256",
        "map_sha256",
        "scored_voxels",
        "eligible_voxels",
        "continuous_context_verified",
        "active_prefix_sha256",
        "scene_prefix_sha256",
        "robot_state_sha256",
        "robot_tokens_sha256",
        "numeric_alignment_interlock",
        "numeric_approach_interlock",
        "numeric_compound_approach_planner",
    }
    return {str(key): item for key, item in value.items() if key in permitted}


def execute_trained_goal(
    runtime: Any,
    bundle: TrainedGoalPolicyBundle,
    *,
    kind: Literal["face", "approach"],
    target_text: str,
    max_steps: int = 12,
) -> dict[str, Any]:
    """Run one goal through repeated full-prefix policy decisions and bounded tools.

    A policy ``stop`` means that this conversational goal is settled.  It is
    intentionally not dispatched to the simulator's episode-wide safety latch,
    so a later natural-language goal can reuse the same local session.
    """

    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or not 1 <= max_steps <= 32:
        raise ValueError("Goal-policy max_steps must be an integer in [1, 32]")
    if not isinstance(bundle, TrainedGoalPolicyBundle):
        raise TypeError("execute_trained_goal requires an authenticated trained bundle")
    settings = getattr(getattr(runtime, "simulator", None), "settings", None)
    runtime_auto_scan = getattr(runtime, "auto_scan_after_motion", None)
    if (
        not isinstance(settings, Mapping)
        or settings.get("auto_scan_after_motion") is not False
        or runtime_auto_scan is not None
        and runtime_auto_scan is not False
    ):
        raise ValueError("Trained goals require static-map auto_scan_after_motion=false")

    instruction = canonical_terminal_goal(kind, target_text)
    target_sha256 = hashlib.sha256(target_text.strip().encode("utf-8")).hexdigest()
    initial_binding = _binding(runtime)
    initial_state = runtime.get_robot_state()
    if not isinstance(initial_state, Mapping):
        raise TypeError("Goal runtime returned invalid numeric robot state")
    initial_scan_count = initial_state.get("scan_count", 0)
    if (
        isinstance(initial_scan_count, bool)
        or not isinstance(initial_scan_count, int)
        or initial_scan_count < 0
    ):
        raise ValueError("Goal runtime returned invalid scan_count")

    steps: list[dict[str, Any]] = []
    receipts: list[dict[str, Any]] = []
    termination = "max_steps"
    error_code: str | None = "E_GOAL_STEPS"
    for index in range(1, max_steps + 1):
        before = _binding(runtime)
        current_state = runtime.get_robot_state()
        if (
            not _same_static_scene(initial_binding, before)
            or not isinstance(current_state, Mapping)
            or current_state.get("scan_count") != initial_scan_count
        ):
            termination = "static_scene_changed"
            error_code = "E_STATIC_SCENE_CHANGED"
            break
        decision = bundle.policy.select(instruction)
        if not _decision_matches_binding(decision, before):
            termination = "policy_context_rejected"
            error_code = "E_CONTEXT"
            break
        call = decision.call
        grounding = _grounding_snapshot(bundle.backend)
        step: dict[str, Any] = {
            "step": index,
            "tool_selection": decision.audit_payload(),
            "continuous_grounding": grounding,
            "prefix_binding_before": before,
            "numeric_tool_receipt": None,
        }
        if call is None:
            steps.append(step)
            termination = "policy_rejected"
            error_code = decision.validation_errors[-1] if decision.validation_errors else "E_POLICY"
            break
        if not _complete_grounding(grounding, before):
            steps.append(step)
            termination = "grounding_rejected"
            error_code = "E_GROUNDING"
            break
        if index == 1:
            decision_state = runtime.get_robot_state()
            if (
                not isinstance(decision_state, Mapping)
                or decision_state.get("scan_count") != initial_scan_count
            ):
                steps.append(step)
                termination = "predecision_sensor_action_rejected"
                error_code = "E_PREDECISION_SENSOR"
                break
        if call.name == "scan":
            steps.append(step)
            termination = "sensor_action_rejected"
            error_code = "E_SENSOR_ACTION"
            break
        if call.name == "stop":
            steps.append(step)
            termination = "goal_settled"
            error_code = None
            break
        receipt = execute_validated_tool_call(runtime, call, config=bundle.policy.config)
        receipts.append(receipt)
        step["numeric_tool_receipt"] = receipt
        after = _binding(runtime)
        step["prefix_binding_after"] = after
        steps.append(step)
        if receipt.get("success") is not True:
            termination = "action_failed"
            error_code = str(receipt.get("error_code") or "E_ACTION")
            break
        after_state = runtime.get_robot_state()
        if (
            not _same_static_scene(initial_binding, after)
            or not isinstance(after_state, Mapping)
            or after_state.get("scan_count") != initial_scan_count
        ):
            termination = "static_scene_changed"
            error_code = "E_STATIC_SCENE_CHANGED"
            break
        if not _active_robot_context_advanced(before, after):
            termination = "robot_prefix_refresh_rejected"
            error_code = "E_ROBOT_PREFIX_STALE"
            break

    success = termination == "goal_settled"
    final_state = runtime.get_robot_state()
    if not isinstance(final_state, Mapping):
        raise TypeError("Goal runtime returned invalid final robot state")
    grounding_complete = bool(steps) and all(
        _complete_grounding(
            step.get("continuous_grounding"),
            step.get("prefix_binding_before", {}),
        )
        for step in steps
    )
    return {
        "schema": "semantic_3d_chat.trained_semantic_goal.v1",
        "kind": "navigation",
        "command": kind,
        "success": success,
        "error_code": error_code,
        "termination_reason": termination,
        "step_count": len(steps),
        "max_steps": max_steps,
        "target_text_sha256": target_sha256,
        "target_text_retained_in_result": False,
        "steps": steps,
        "action_receipts": receipts,
        "initial_prefix_binding": initial_binding,
        "final_prefix_binding": _binding(runtime),
        "initial_scan_count": initial_scan_count,
        "final_scan_count": final_state.get("scan_count"),
        "scene_prefix_computed_before_goal": True,
        "internal_sensor_actions_before_first_policy_decision": 0,
        "camera_observations_during_goal": 0,
        "first_policy_decision_preceded_internal_sensor_actions": False,
        "current_camera_observation_required_before_first_decision": False,
        "static_full_scene_memory_available_to_first_decision": True,
        "static_scene_prefix_unchanged": _same_static_scene(
            initial_binding,
            _binding(runtime),
        ),
        "scene_tokens_processed_per_decision": int(bundle.metadata["scene_token_count"]),
        "every_scene_token_processed": bundle.metadata.get("every_scene_token_processed")
        is True,
        "all_target_groundings_scored_complete_map": grounding_complete,
        "task_trained": True,
        "training_status": TRAINING_STATUS,
        "goal_settled_without_episode_stop_latch": success
        and final_state.get("stopped") is not True,
        "environmental_text_inputs": [],
        "oracle_inputs_at_runtime": False,
    }


__all__ = [
    "DEFAULT_TRAINED_GOAL_CHECKPOINT",
    "TrainedGoalPolicyBundle",
    "canonical_terminal_goal",
    "execute_trained_goal",
    "load_trained_goal_policy",
]
