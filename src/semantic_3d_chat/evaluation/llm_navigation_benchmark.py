"""Sealed inference and physically separate scoring for Gemma robot navigation.

The inference half of this module accepts only user-authored instructions, the
continuous scene/robot prefix owned by ``LocalGemmaToolPolicy``, and numeric
tool receipts.  It deliberately has no representation for target instance IDs,
object boxes, expected directions, or success labels.  The scoring half first
verifies a completed prediction journal and only then opens its oracle inputs.

The harness supports both the original untrained action-selection seam and an
explicitly hash-attested learned controller. It never infers trained status
from fluent output or task score.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal, Protocol

import numpy as np

from semantic_3d_chat.robot.llm_tool_policy import (
    LocalGemmaToolPolicy,
    ToolPolicyDecision,
    execute_validated_tool_call,
)

TaskFamily = Literal[
    "face",
    "approach",
    "stop",
    "obstacle",
    "left_right",
    "update_after_scan",
]

_TASK_FAMILIES: Final[frozenset[str]] = frozenset(
    {"face", "approach", "stop", "obstacle", "left_right", "update_after_scan"}
)
_TASK_ID: Final[re.Pattern[str]] = re.compile(r"nav_[0-9]{3}")
_SCENE_ID: Final[re.Pattern[str]] = re.compile(r"scene_[0-9]{6}")
_INSTANCE_ID: Final[re.Pattern[str]] = re.compile(r"i_[0-9]{6}")
_SHA256: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{64}")
_JOURNAL_SCHEMA: Final[str] = "semantic_3d_chat.llm_navigation_journal.v1"
_TASK_SCHEMA: Final[str] = "semantic_3d_chat.llm_navigation_tasks.v1"
_SCORE_SPEC_SCHEMA: Final[str] = "semantic_3d_chat.llm_navigation_oracle.v1"
_SCORE_SCHEMA: Final[str] = "semantic_3d_chat.llm_navigation_score.v1"


class NavigationRuntime(Protocol):
    def get_robot_state(self) -> Mapping[str, Any]: ...

    def prefix_binding(self) -> Mapping[str, Any]: ...

    def reset_scene(self, scene_id: str, seed: int) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class NavigationTask:
    task_id: str
    family: TaskFamily
    instruction: str
    max_steps: int


@dataclass(frozen=True)
class NavigationTaskManifest:
    scene_id: str
    seed: int
    tasks: tuple[NavigationTask, ...]
    canonical_sha256: str


def canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_sha256(path: str | Path) -> str:
    """Hash one file or a complete regular-file directory tree."""

    root = Path(path)
    if root.is_file():
        return file_sha256(root)
    if not root.is_dir():
        raise FileNotFoundError(root)
    files = sorted(item for item in root.rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"Cannot hash an empty artifact directory: {root}")
    digest = hashlib.sha256()
    for item in files:
        relative = item.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(bytes.fromhex(file_sha256(item)))
    return digest.hexdigest()


def _strict_object(value: object, *, name: str, keys: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"{name} must contain exactly {sorted(keys)}")
    return value


def _finite_number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def parse_task_manifest(payload: object) -> NavigationTaskManifest:
    document = _strict_object(
        payload,
        name="task manifest",
        keys={"schema", "scene_id", "seed", "tasks"},
    )
    if document["schema"] != _TASK_SCHEMA:
        raise ValueError("Unsupported navigation task-manifest schema")
    scene_id = document["scene_id"]
    if not isinstance(scene_id, str) or _SCENE_ID.fullmatch(scene_id) is None:
        raise ValueError("Task manifest scene_id is not opaque")
    seed = _nonnegative_int(document["seed"], name="task manifest seed")
    if seed > 2**32 - 1:
        raise ValueError("Task manifest seed exceeds uint32")
    rows = document["tasks"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("Task manifest must contain at least one task")
    tasks: list[NavigationTask] = []
    seen: set[str] = set()
    for index, raw in enumerate(rows):
        row = _strict_object(
            raw,
            name=f"tasks[{index}]",
            keys={"task_id", "family", "instruction", "max_steps"},
        )
        task_id = row["task_id"]
        family = row["family"]
        instruction = row["instruction"]
        max_steps = row["max_steps"]
        if not isinstance(task_id, str) or _TASK_ID.fullmatch(task_id) is None:
            raise ValueError(f"tasks[{index}].task_id is invalid")
        if task_id in seen:
            raise ValueError("Task IDs must be unique")
        if not isinstance(family, str) or family not in _TASK_FAMILIES:
            raise ValueError(f"tasks[{index}].family is invalid")
        if not isinstance(instruction, str) or not instruction.strip() or len(instruction) > 1024:
            raise ValueError(f"tasks[{index}].instruction is invalid")
        if isinstance(max_steps, bool) or not isinstance(max_steps, int):
            raise TypeError(f"tasks[{index}].max_steps must be an integer")
        if not 1 <= max_steps <= 32:
            raise ValueError(f"tasks[{index}].max_steps must be in [1, 32]")
        seen.add(task_id)
        tasks.append(
            NavigationTask(
                task_id=task_id,
                family=family,  # type: ignore[arg-type]
                instruction=instruction.strip(),
                max_steps=max_steps,
            )
        )
    return NavigationTaskManifest(
        scene_id=scene_id,
        seed=seed,
        tasks=tuple(tasks),
        canonical_sha256=canonical_sha256(document),
    )


def load_task_manifest(path: str | Path) -> NavigationTaskManifest:
    return parse_task_manifest(json.loads(Path(path).read_text(encoding="utf-8")))


_NUMERIC_RECEIPT_KEYS: Final[tuple[str, ...]] = (
    "success",
    "error_code",
    "scene_version",
    "position_m",
    "camera_position_m",
    "body_yaw_degrees",
    "camera_yaw_degrees",
    "pitch_degrees",
    "collision",
    "last_movement_delta_m",
    "distance_moved",
    "turn_degrees",
    "scan_coverage",
    "scan_count",
    "visible_voxels",
    "valid_depth_pixels",
    "clearance_m",
    "action_count",
    "stopped",
)


def numeric_tool_receipt(result: Mapping[str, Any]) -> dict[str, Any]:
    """Return the only action feedback permitted in a subsequent text prompt."""

    receipt = {key: result.get(key) for key in _NUMERIC_RECEIPT_KEYS if key in result}
    for key, value in receipt.items():
        if key == "error_code":
            if value is not None and (
                not isinstance(value, str) or re.fullmatch(r"E_[A-Z0-9_]+", value) is None
            ):
                raise ValueError("Tool result contains a non-protocol error string")
            continue
        if isinstance(value, list):
            for item in value:
                _finite_number(item, name=f"tool receipt {key}")
        elif isinstance(value, bool) or value is None:
            continue
        else:
            _finite_number(value, name=f"tool receipt {key}")
    return receipt


def _policy_instruction(task: NavigationTask, prior_receipt: Mapping[str, Any] | None) -> str:
    prompt = (
        f"User navigation instruction: {task.instruction}\n"
        "Issue exactly one bounded action now. Continue toward the same instruction after "
        "each numeric result, and issue stop only when the instruction is complete."
    )
    if prior_receipt is not None:
        prompt += "\nNumeric result from the preceding bounded action: " + canonical_json(
            prior_receipt
        )
    return prompt


def _binding(value: Mapping[str, Any]) -> dict[str, Any]:
    allowed = (
        "active_prefix_sha256",
        "scene_prefix_sha256",
        "robot_tokens_sha256",
        "map_sha256",
        "map_version",
        "binding_sha256",
    )
    result = {key: value.get(key) for key in allowed if key in value}
    for key, item in result.items():
        if key.endswith("sha256"):
            if not isinstance(item, str) or _SHA256.fullmatch(item) is None:
                raise ValueError(f"Runtime returned an invalid {key}")
        elif key == "map_version":
            _nonnegative_int(item, name="map_version")
    return result


def _decision_payload(decision: ToolPolicyDecision) -> dict[str, Any]:
    payload = decision.audit_payload()
    # The policy already omits raw output.  Assert that invariant at the journal
    # boundary so a future backend cannot accidentally serialize model prose.
    if payload.get("raw_model_output_logged") is not False:
        raise RuntimeError("Tool-policy audit attempted to retain raw model output")
    if payload.get("environmental_text_inputs") != []:
        raise RuntimeError("Tool-policy audit contains environmental text")
    return payload


def _step_hash(previous: str, body: Mapping[str, Any]) -> str:
    if _SHA256.fullmatch(previous) is None:
        raise ValueError("Invalid prior transcript-chain hash")
    return hashlib.sha256(
        bytes.fromhex(previous) + canonical_json(body).encode("utf-8")
    ).hexdigest()


def run_navigation_episode(
    runtime: NavigationRuntime,
    policy: LocalGemmaToolPolicy,
    task: NavigationTask,
    *,
    scene_id: str,
    seed: int,
    policy_training_status: str = "untrained_tool_selection_seam",
) -> dict[str, Any]:
    """Run one reset, retry-bounded closed-loop task without success labels."""

    reset = dict(runtime.reset_scene(scene_id, seed))
    if reset.get("success") is not True:
        raise RuntimeError(f"Episode reset failed closed: {reset.get('error_code')}")
    initial_state = numeric_tool_receipt(runtime.get_robot_state())
    initial_binding = _binding(runtime.prefix_binding())
    chain = "0" * 64
    steps: list[dict[str, Any]] = []
    prior_receipt: dict[str, Any] | None = None
    termination = "max_steps"
    for step_index in range(task.max_steps):
        policy_input = _policy_instruction(task, prior_receipt)
        # Capture the context before generation.  The policy must attest this
        # exact snapshot; taking it afterward would hide a concurrent state or
        # map change during action selection.
        before = _binding(runtime.prefix_binding())
        decision = policy.select(policy_input)
        decision_payload = _decision_payload(decision)
        if decision_payload.get("training_status") != policy_training_status:
            raise RuntimeError("Tool-policy training attestation changed during inference")
        context_pairs = (
            ("active_prefix_sha256", "active_prefix_sha256"),
            ("scene_prefix_sha256", "scene_prefix_sha256"),
            ("robot_tokens_sha256", "robot_tokens_sha256"),
        )
        if any(
            decision_payload.get(decision_key) != before.get(binding_key)
            for decision_key, binding_key in context_pairs
        ):
            raise RuntimeError("Tool policy did not consume the captured continuous context")
        body: dict[str, Any] = {
            "step": step_index,
            "policy_input_sha256": hashlib.sha256(policy_input.encode("utf-8")).hexdigest(),
            "prefix_before": before,
            "decision": decision_payload,
            "receipt": None,
            "prefix_after": before,
        }
        if decision.call is None:
            termination = "policy_rejected"
            chain = _step_hash(chain, body)
            steps.append({**body, "transcript_chain_sha256": chain})
            break
        receipt = numeric_tool_receipt(
            execute_validated_tool_call(runtime, decision.call, config=policy.config)
        )
        after = _binding(runtime.prefix_binding())
        body["receipt"] = receipt
        body["prefix_after"] = after
        chain = _step_hash(chain, body)
        steps.append({**body, "transcript_chain_sha256": chain})
        prior_receipt = receipt
        if decision.call.name == "stop" and receipt.get("success") is True:
            termination = "model_stop"
            break

    final_state = numeric_tool_receipt(runtime.get_robot_state())
    episode_body = {
        "task_id": task.task_id,
        "family": task.family,
        "instruction_sha256": hashlib.sha256(task.instruction.encode("utf-8")).hexdigest(),
        "max_steps": task.max_steps,
        "termination": termination,
        "initial_state": initial_state,
        "initial_prefix_binding": initial_binding,
        "steps": steps,
        "final_state": final_state,
        "final_prefix_binding": _binding(runtime.prefix_binding()),
        "transcript_chain_sha256": chain,
        "task_success_scored_during_inference": False,
        "oracle_or_labels_available": False,
        "environmental_text_inputs": [],
        "tool_policy_training_status": policy_training_status,
    }
    return {**episode_body, "episode_sha256": canonical_sha256(episode_body)}


def _journal_digest(payload: Mapping[str, Any]) -> str:
    return canonical_sha256(
        {key: value for key, value in payload.items() if key != "journal_sha256"}
    )


def validate_navigation_journal(
    payload: object,
    *,
    expected_header: Mapping[str, Any] | None = None,
    require_complete: bool = False,
) -> dict[str, Any]:
    document = _strict_object(
        payload,
        name="navigation journal",
        keys={
            "schema",
            "status",
            "header",
            "episodes",
            "runtime_file_audit",
            "journal_sha256",
        },
    )
    if document["schema"] != _JOURNAL_SCHEMA:
        raise ValueError("Unsupported navigation journal schema")
    if document["status"] not in {"in_progress", "complete"}:
        raise ValueError("Navigation journal status is invalid")
    if require_complete and document["status"] != "complete":
        raise ValueError("Navigation journal is incomplete")
    header = document["header"]
    if not isinstance(header, dict):
        raise TypeError("Navigation journal header is invalid")
    raw_header_training_status = header.get("tool_policy_training_status")
    training_attestation_valid = raw_header_training_status in {
        "untrained_tool_selection_seam",
        "supervised_continuous_navigation_policy_v1",
        "supervised_continuous_semantic_grounded_navigation_policy_v3",
        "supervised_continuous_semantic_clearance_navigation_policy_v4",
    }
    # Preserve the root-hash-first failure contract for a malformed empty
    # journal while still validating the attestation on every sealed artifact.
    header_training_status = (
        raw_header_training_status
        if training_attestation_valid
        else "untrained_tool_selection_seam"
    )
    if expected_header is not None and header != dict(expected_header):
        raise ValueError("Navigation journal run contract changed")
    episodes = document["episodes"]
    if not isinstance(episodes, list):
        raise TypeError("Navigation journal episodes are invalid")
    seen: set[str] = set()
    for episode_index, value in enumerate(episodes):
        if not isinstance(value, dict):
            raise TypeError("Navigation journal episode is invalid")
        task_id = value.get("task_id")
        if not isinstance(task_id, str) or task_id in seen:
            raise ValueError("Navigation journal has duplicate or invalid task IDs")
        seen.add(task_id)
        claimed = value.get("episode_sha256")
        observed = canonical_sha256(
            {key: item for key, item in value.items() if key != "episode_sha256"}
        )
        if claimed != observed:
            raise ValueError(f"Navigation episode {episode_index} hash mismatch")
        previous = "0" * 64
        initial_binding_raw = value.get("initial_prefix_binding")
        if not isinstance(initial_binding_raw, Mapping):
            raise TypeError("Navigation episode initial prefix binding is invalid")
        expected_before = _binding(initial_binding_raw)
        steps = value.get("steps")
        if not isinstance(steps, list):
            raise TypeError("Navigation episode steps are invalid")
        for step_index, step in enumerate(steps):
            if not isinstance(step, dict):
                raise TypeError("Navigation step is invalid")
            claimed_step = step.get("transcript_chain_sha256")
            body = {key: item for key, item in step.items() if key != "transcript_chain_sha256"}
            observed_step = _step_hash(previous, body)
            if claimed_step != observed_step or body.get("step") != step_index:
                raise ValueError("Navigation transcript chain mismatch")
            decision = body.get("decision")
            before = body.get("prefix_before")
            after = body.get("prefix_after")
            if (
                not isinstance(decision, Mapping)
                or not isinstance(before, Mapping)
                or not isinstance(after, Mapping)
            ):
                raise TypeError("Navigation decision or prefix binding is invalid")
            before_binding = _binding(before)
            after_binding = _binding(after)
            if before_binding != expected_before:
                raise ValueError("Navigation prefix chain is discontinuous before a decision")
            if (
                decision.get("local_inference") is not True
                or decision.get("used_continuous_scene_prefix") is not True
                or decision.get("used_continuous_robot_tokens") is not True
                or decision.get("raw_model_output_logged") is not False
                or decision.get("environmental_text_inputs") != []
            ):
                raise ValueError("Navigation decision lacks continuous-context attestation")
            decision_training_status = decision.get("training_status")
            if (
                decision_training_status is None
                and header_training_status == "untrained_tool_selection_seam"
            ):
                decision_training_status = "untrained_tool_selection_seam"
            if decision_training_status != header_training_status:
                raise ValueError("Navigation decision training attestation changed")
            context_pairs = (
                ("active_prefix_sha256", "active_prefix_sha256"),
                ("scene_prefix_sha256", "scene_prefix_sha256"),
                ("robot_tokens_sha256", "robot_tokens_sha256"),
            )
            if any(
                decision.get(left) != before_binding.get(right)
                for left, right in context_pairs
            ):
                raise ValueError("Navigation decision prefix differs from its bound input")
            before_version = int(before_binding.get("map_version", -1))
            after_version = int(after_binding.get("map_version", -1))
            if after_version not in {before_version, before_version + 1}:
                raise ValueError("Navigation map version transition is invalid")
            scene_changed = (
                before_binding.get("scene_prefix_sha256")
                != after_binding.get("scene_prefix_sha256")
            )
            map_changed = before_binding.get("map_sha256") != after_binding.get(
                "map_sha256"
            )
            if (after_version == before_version and (scene_changed or map_changed)) or (
                after_version == before_version + 1
                and (not scene_changed or not map_changed)
            ):
                raise ValueError("Navigation map and scene-prefix transition disagree")
            expected_before = after_binding
            previous = observed_step
        if value.get("transcript_chain_sha256") != previous:
            raise ValueError("Navigation episode chain head mismatch")
        final_binding_raw = value.get("final_prefix_binding")
        if not isinstance(final_binding_raw, Mapping):
            raise TypeError("Navigation episode final prefix binding is invalid")
        if _binding(final_binding_raw) != expected_before:
            raise ValueError("Navigation final prefix differs from the transcript chain")
        if (
            value.get("task_success_scored_during_inference") is not False
            or value.get("oracle_or_labels_available") is not False
            or value.get("environmental_text_inputs") != []
        ):
            raise ValueError("Navigation inference journal contains forbidden supervision")
        if value.get("tool_policy_training_status") != header_training_status:
            raise ValueError("Navigation episode training attestation changed")
    claimed_journal = document["journal_sha256"]
    if claimed_journal != _journal_digest(document):
        raise ValueError("Navigation journal root hash mismatch")
    if not training_attestation_valid:
        raise ValueError("Navigation journal has an invalid policy-training attestation")
    return document


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def write_navigation_journal(path: str | Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    document = {**dict(payload), "journal_sha256": ""}
    document["journal_sha256"] = _journal_digest(document)
    _atomic_json(Path(path), document)
    return document


def run_navigation_manifest(
    runtime: NavigationRuntime,
    policy: LocalGemmaToolPolicy,
    manifest: NavigationTaskManifest,
    *,
    journal_path: str | Path,
    run_contract: Mapping[str, Any],
    resume: bool = False,
    runtime_file_audit: Mapping[str, Any] | None = None,
    after_episode: Callable[[dict[str, Any]], None] | None = None,
    policy_training_status: str = "untrained_tool_selection_seam",
) -> dict[str, Any]:
    """Run/restart whole episodes and seal progress atomically after each one."""

    path = Path(journal_path)
    allowed_training_status = {
        "untrained_tool_selection_seam",
        "supervised_continuous_navigation_policy_v1",
        "supervised_continuous_semantic_grounded_navigation_policy_v3",
        "supervised_continuous_semantic_clearance_navigation_policy_v4",
    }
    if policy_training_status not in allowed_training_status:
        raise ValueError("Unsupported navigation-policy training attestation")
    header = {
        "scene_id": manifest.scene_id,
        "seed": manifest.seed,
        "task_manifest_sha256": manifest.canonical_sha256,
        "task_count": len(manifest.tasks),
        "run_contract": dict(run_contract),
        "local_inference": True,
        "environment_input": "continuous_scene_and_numeric_robot_prefix",
        "feedback_input": "bounded_numeric_tool_receipts",
        "oracle_or_labels_available": False,
        "tool_policy_training_status": policy_training_status,
        "claimed_learned_navigation_success": False,
    }
    episodes: list[dict[str, Any]] = []
    if path.exists():
        if not resume:
            raise FileExistsError(f"Prediction journal exists; pass resume explicitly: {path}")
        existing = validate_navigation_journal(
            json.loads(path.read_text(encoding="utf-8")), expected_header=header
        )
        episodes = list(existing["episodes"])
        if existing["status"] == "complete":
            return existing
    expected_ids = [task.task_id for task in manifest.tasks]
    observed_ids = [episode["task_id"] for episode in episodes]
    if observed_ids != expected_ids[: len(observed_ids)]:
        raise ValueError("Navigation resume journal is not an ordered task prefix")
    base = {
        "schema": _JOURNAL_SCHEMA,
        "status": "in_progress",
        "header": header,
        "episodes": episodes,
        "runtime_file_audit": dict(runtime_file_audit or {"status": "pending"}),
    }
    write_navigation_journal(path, base)
    for task in manifest.tasks[len(episodes) :]:
        episode = run_navigation_episode(
            runtime,
            policy,
            task,
            scene_id=manifest.scene_id,
            seed=manifest.seed,
            policy_training_status=policy_training_status,
        )
        episodes.append(episode)
        base["episodes"] = episodes
        write_navigation_journal(path, base)
        if after_episode is not None:
            after_episode(episode)
    base["status"] = "complete"
    base["runtime_file_audit"] = dict(runtime_file_audit or {"status": "not_supplied"})
    return write_navigation_journal(path, base)


def finalize_navigation_journal_audit(
    path: str | Path,
    audit_payload: Mapping[str, Any],
) -> dict[str, Any]:
    document = validate_navigation_journal(json.loads(Path(path).read_text(encoding="utf-8")))
    if document["status"] != "complete":
        raise ValueError("Cannot attach a final audit to an incomplete journal")
    document["runtime_file_audit"] = dict(audit_payload)
    return write_navigation_journal(path, document)


def _bbox(instance: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    raw = instance.get("bbox")
    if not isinstance(raw, Mapping):
        raise TypeError("Oracle instance lacks a bounding box")
    lower = np.asarray(raw.get("min_xyz_m"), dtype=np.float64)
    upper = np.asarray(raw.get("max_xyz_m"), dtype=np.float64)
    if lower.shape != (3,) or upper.shape != (3,) or not np.isfinite([lower, upper]).all():
        raise ValueError("Oracle instance has an invalid bounding box")
    if np.any(upper < lower):
        raise ValueError("Oracle instance bounding box is inverted")
    return lower, upper


def _position(state: Mapping[str, Any]) -> np.ndarray:
    value = np.asarray(state.get("position_m"), dtype=np.float64)
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError("Journal robot position is invalid")
    return value


def _distance_to_box_xy(point: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    delta = np.maximum(np.maximum(lower[:2] - point[:2], point[:2] - upper[:2]), 0.0)
    return float(np.linalg.norm(delta))


def _normalize_degrees(value: float) -> float:
    return (float(value) + 180.0) % 360.0 - 180.0


def _desired_heading(position: np.ndarray, target: np.ndarray) -> float:
    delta = target[:2] - position[:2]
    return math.degrees(math.atan2(-float(delta[0]), float(delta[1])))


def _calls(episode: Mapping[str, Any]) -> list[tuple[str, Mapping[str, Any], Mapping[str, Any]]]:
    result: list[tuple[str, Mapping[str, Any], Mapping[str, Any]]] = []
    for step in episode["steps"]:
        decision = step["decision"]
        call = decision.get("call")
        receipt = step.get("receipt")
        if isinstance(call, Mapping) and isinstance(receipt, Mapping):
            name = call.get("tool")
            if not isinstance(name, str):
                raise TypeError("Journal tool name is invalid")
            result.append((name, call, receipt))
    return result


_ROBOT_STATE_RECEIPT_KEYS: Final[tuple[str, ...]] = (
    "position_m",
    "body_yaw_degrees",
    "camera_yaw_degrees",
    "pitch_degrees",
    "collision",
    "last_movement_delta_m",
    "scan_coverage",
    "stopped",
)


def continuous_context_metrics(journal: Mapping[str, Any]) -> dict[str, Any]:
    """Measure whether every action used the current scene and robot tokens.

    This is inference-only evidence: it consumes no target IDs, labels, object
    names, or oracle geometry.  State changes are inferred solely from numeric
    receipts and must be accompanied by a new robot-token hash.  Map-version
    changes must update the scene prefix and be consumed by the next decision.
    """

    episodes = journal.get("episodes")
    if not isinstance(episodes, list):
        raise TypeError("Navigation journal episodes are invalid")
    step_count = 0
    decision_context_matches = 0
    prefix_chain_matches = 0
    numeric_state_change_count = 0
    robot_token_refresh_count = 0
    map_update_count = 0
    scene_prefix_refresh_count = 0
    next_decision_count = 0
    refreshed_context_consumed_count = 0
    for episode in episodes:
        if not isinstance(episode, Mapping):
            raise TypeError("Navigation context episode is invalid")
        prior_numeric = episode.get("initial_state")
        prior_binding = episode.get("initial_prefix_binding")
        steps = episode.get("steps")
        if (
            not isinstance(prior_numeric, Mapping)
            or not isinstance(prior_binding, Mapping)
            or not isinstance(steps, list)
        ):
            raise TypeError("Navigation context transcript is incomplete")
        for index, step in enumerate(steps):
            if not isinstance(step, Mapping):
                raise TypeError("Navigation context step is invalid")
            before = step.get("prefix_before")
            after = step.get("prefix_after")
            decision = step.get("decision")
            if not all(isinstance(value, Mapping) for value in (before, after, decision)):
                raise TypeError("Navigation context binding is invalid")
            assert isinstance(before, Mapping)
            assert isinstance(after, Mapping)
            assert isinstance(decision, Mapping)
            step_count += 1
            if before == prior_binding:
                prefix_chain_matches += 1
            if all(
                decision.get(name) == before.get(name)
                for name in (
                    "active_prefix_sha256",
                    "scene_prefix_sha256",
                    "robot_tokens_sha256",
                )
            ):
                decision_context_matches += 1

            receipt = step.get("receipt")
            if isinstance(receipt, Mapping):
                before_numeric = {
                    name: prior_numeric.get(name) for name in _ROBOT_STATE_RECEIPT_KEYS
                }
                after_numeric = {
                    name: receipt.get(name) for name in _ROBOT_STATE_RECEIPT_KEYS
                }
                if before_numeric != after_numeric:
                    numeric_state_change_count += 1
                    if before.get("robot_tokens_sha256") != after.get(
                        "robot_tokens_sha256"
                    ):
                        robot_token_refresh_count += 1
                prior_numeric = receipt

            before_version = int(before.get("map_version", -1))
            after_version = int(after.get("map_version", -1))
            if after_version != before_version:
                map_update_count += 1
                if (
                    before.get("map_sha256") != after.get("map_sha256")
                    and before.get("scene_prefix_sha256")
                    != after.get("scene_prefix_sha256")
                ):
                    scene_prefix_refresh_count += 1
            if index + 1 < len(steps):
                following = steps[index + 1]
                if not isinstance(following, Mapping):
                    raise TypeError("Navigation following step is invalid")
                following_decision = following.get("decision")
                next_decision_count += 1
                if isinstance(following_decision, Mapping) and all(
                    following_decision.get(name) == after.get(name)
                    for name in (
                        "active_prefix_sha256",
                        "scene_prefix_sha256",
                        "robot_tokens_sha256",
                    )
                ):
                    refreshed_context_consumed_count += 1
            prior_binding = after

    passed = bool(
        step_count > 0
        and decision_context_matches == step_count
        and prefix_chain_matches == step_count
        and robot_token_refresh_count == numeric_state_change_count
        and scene_prefix_refresh_count == map_update_count
        and refreshed_context_consumed_count == next_decision_count
    )
    return {
        "passed": passed,
        "oracle_inputs_used": False,
        "environmental_text_inputs": [],
        "step_count": step_count,
        "decision_context_match_count": decision_context_matches,
        "prefix_chain_match_count": prefix_chain_matches,
        "numeric_state_change_count": numeric_state_change_count,
        "robot_token_refresh_count": robot_token_refresh_count,
        "map_update_count": map_update_count,
        "scene_prefix_refresh_count": scene_prefix_refresh_count,
        "next_decision_count": next_decision_count,
        "refreshed_context_consumed_count": refreshed_context_consumed_count,
    }


def _bool_setting(row: Mapping[str, Any], name: str, default: bool = False) -> bool:
    value = row.get(name, default)
    if not isinstance(value, bool):
        raise TypeError(f"Oracle score setting {name} must be boolean")
    return value


def _threshold(row: Mapping[str, Any], name: str, default: float) -> float:
    result = _finite_number(row.get(name, default), name=name)
    if result < 0:
        raise ValueError(f"Oracle score setting {name} must be nonnegative")
    return result


def _target_instance(
    row: Mapping[str, Any], instances: Mapping[str, Mapping[str, Any]]
) -> Mapping[str, Any]:
    target_id = row.get("target_instance_id")
    if not isinstance(target_id, str) or _INSTANCE_ID.fullmatch(target_id) is None:
        raise ValueError("Oracle score task has an invalid target instance")
    try:
        return instances[target_id]
    except KeyError as error:
        raise ValueError("Oracle score target instance is unavailable") from error


def _score_episode(
    episode: Mapping[str, Any],
    score_task: Mapping[str, Any],
    instances: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    family = episode["family"]
    if score_task.get("family") != family:
        raise ValueError("Oracle scoring family differs from inference task")
    initial = _position(episode["initial_state"])
    final = _position(episode["final_state"])
    calls = _calls(episode)
    tool_names = [name for name, _call, _receipt in calls]
    collision_count = sum(bool(receipt.get("collision")) for _name, _call, receipt in calls)
    action_failure_count = sum(
        receipt.get("success") is not True for _name, _call, receipt in calls
    )
    stopped = bool(episode["final_state"].get("stopped"))
    metrics: dict[str, Any] = {
        "collision_count": collision_count,
        "action_failure_count": action_failure_count,
        "executed_action_count": len(calls),
        "policy_rejected": episode["termination"] == "policy_rejected",
        "stopped": stopped,
    }
    checks: dict[str, bool] = {
        "no_collision": collision_count <= int(_threshold(score_task, "maximum_collisions", 0.0)),
        "all_executed_actions_succeeded": action_failure_count == 0,
    }
    require_stopped = _bool_setting(score_task, "require_stopped", True)
    checks["required_stop"] = stopped if require_stopped else True

    if family == "stop":
        displacement = float(np.linalg.norm(final[:2] - initial[:2]))
        minimum = _threshold(score_task, "minimum_displacement_m", 0.0)
        maximum = _threshold(score_task, "maximum_displacement_m", math.inf)
        metrics.update(
            {
                "displacement_m": displacement,
                "minimum_displacement_m": minimum,
                "maximum_displacement_m": maximum,
                "first_executed_tool": tool_names[0] if tool_names else None,
            }
        )
        checks["displacement"] = minimum <= displacement <= maximum
    else:
        target = _target_instance(score_task, instances)
        lower, upper = _bbox(target)
        center = (lower + upper) / 2.0
        initial_distance = _distance_to_box_xy(initial, lower, upper)
        final_distance = _distance_to_box_xy(final, lower, upper)
        progress = initial_distance - final_distance
        metrics.update(
            {
                "initial_target_standoff_m": initial_distance,
                "final_target_standoff_m": final_distance,
                "target_progress_m": progress,
            }
        )
        if family in {"approach", "obstacle", "update_after_scan"}:
            checks["target_standoff"] = final_distance <= _threshold(
                score_task, "maximum_target_standoff_m", 0.85
            )
            checks["target_progress"] = progress >= _threshold(
                score_task, "minimum_target_progress_m", 0.0
            )
        if family in {"face", "left_right"}:
            desired = _desired_heading(final, center)
            final_yaw = _finite_number(
                episode["final_state"].get("camera_yaw_degrees"),
                name="final camera yaw",
            )
            heading_error = abs(_normalize_degrees(final_yaw - desired))
            metrics["heading_error_degrees"] = heading_error
            checks["heading"] = heading_error <= _threshold(
                score_task, "maximum_heading_error_degrees", 20.0
            )
        if family == "left_right":
            initial_desired = _desired_heading(initial, center)
            initial_yaw = _finite_number(
                episode["initial_state"].get("camera_yaw_degrees"),
                name="initial camera yaw",
            )
            expected_delta = _normalize_degrees(initial_desired - initial_yaw)
            turn_values: list[float] = []
            for name, call, _receipt in calls:
                arguments = call.get("arguments")
                if not isinstance(arguments, Mapping):
                    raise TypeError("Journal tool arguments are invalid")
                if name == "turn":
                    turn_values.append(_finite_number(arguments.get("angle_degrees"), name="turn"))
                elif name == "look":
                    turn_values.append(
                        _finite_number(arguments.get("yaw_delta_degrees"), name="look yaw")
                    )
            first_nonzero = next((value for value in turn_values if abs(value) > 1e-9), 0.0)
            expected_sign = (
                0 if abs(expected_delta) <= 1e-9 else int(math.copysign(1, expected_delta))
            )
            actual_sign = 0 if abs(first_nonzero) <= 1e-9 else int(math.copysign(1, first_nonzero))
            metrics.update(
                {
                    "expected_initial_turn_sign": expected_sign,
                    "actual_initial_turn_sign": actual_sign,
                }
            )
            checks["initial_left_right_direction"] = actual_sign == expected_sign
        if family == "obstacle":
            obstacle_id = score_task.get("obstacle_instance_id")
            if not isinstance(obstacle_id, str) or obstacle_id not in instances:
                raise ValueError("Oracle obstacle instance is unavailable")
            obstacle_lower, obstacle_upper = _bbox(instances[obstacle_id])
            positions = [initial]
            positions.extend(_position(receipt) for _name, _call, receipt in calls)
            clearance = min(
                _distance_to_box_xy(point, obstacle_lower, obstacle_upper) for point in positions
            )
            metrics["minimum_obstacle_bbox_clearance_m"] = clearance
            checks["obstacle_clearance"] = clearance >= _threshold(
                score_task, "minimum_obstacle_bbox_clearance_m", 0.0
            )
        if family == "update_after_scan":
            scan_indices = [index for index, name in enumerate(tool_names) if name == "scan"]
            successful_scan_indices = [
                index
                for index in scan_indices
                if calls[index][2].get("success") is True
                and int(calls[index][2].get("scene_version", 0)) > 0
            ]
            changed_and_consumed = False
            post_scan_motion = False
            for index in successful_scan_indices:
                step = episode["steps"][index]
                before_hash = step["prefix_before"].get("scene_prefix_sha256")
                after_hash = step["prefix_after"].get("scene_prefix_sha256")
                changed = before_hash != after_hash
                if index + 1 < len(episode["steps"]):
                    next_hash = episode["steps"][index + 1]["decision"].get("active_prefix_sha256")
                    changed_and_consumed |= changed and next_hash == step["prefix_after"].get(
                        "active_prefix_sha256"
                    )
                post_scan_motion |= any(
                    name in {"move_forward", "move_backward", "move_to"}
                    for name in tool_names[index + 1 :]
                )
            metrics["successful_scan_count"] = len(successful_scan_indices)
            checks["successful_scan"] = bool(successful_scan_indices)
            checks["updated_prefix_consumed"] = changed_and_consumed
            checks["post_scan_motion"] = post_scan_motion
    passed = all(checks.values()) and not metrics["policy_rejected"]
    return {
        "task_id": episode["task_id"],
        "family": family,
        "passed": passed,
        "checks": checks,
        "metrics": metrics,
        "episode_sha256": episode["episode_sha256"],
    }


def _authenticated_policy_provenance(journal: Mapping[str, Any]) -> dict[str, Any]:
    """Derive trained/untrained status from the sealed inference contract."""

    header = journal.get("header")
    if not isinstance(header, Mapping):
        raise TypeError("Navigation journal header is unavailable")
    status = header.get("tool_policy_training_status")
    contract = header.get("run_contract")
    if not isinstance(contract, Mapping):
        raise TypeError("Navigation journal run contract is unavailable")
    if status == "untrained_tool_selection_seam":
        contract_status = contract.get("tool_policy_training_status")
        if contract_status not in {None, status}:
            raise ValueError("Untrained navigation journal has a conflicting run contract")
        if any(
            key in contract
            for key in (
                "navigation_policy_checkpoint_tree_sha256",
                "navigation_policy_source_sha256",
            )
        ):
            raise ValueError("Untrained navigation journal unexpectedly binds a learned policy")
        return {
            "policy_status": status,
            "claimed_trained_navigation_policy": False,
            "navigation_policy_checkpoint_tree_sha256": None,
        }
    if status in {
        "supervised_continuous_navigation_policy_v1",
        "supervised_continuous_semantic_grounded_navigation_policy_v3",
        "supervised_continuous_semantic_clearance_navigation_policy_v4",
    }:
        checkpoint_hash = contract.get("navigation_policy_checkpoint_tree_sha256")
        source_hash = contract.get("navigation_policy_source_sha256")
        if (
            contract.get("tool_policy_training_status") != status
            or not isinstance(checkpoint_hash, str)
            or _SHA256.fullmatch(checkpoint_hash) is None
            or not isinstance(source_hash, str)
            or _SHA256.fullmatch(source_hash) is None
            or contract.get("fallback_policy") != "fail_closed"
            or contract.get("strict_fixed_environment_embedding_input") is not True
            or contract.get("question_conditioned_scene_readout_tokens") is not False
        ):
            raise ValueError("Learned navigation journal lacks authenticated policy provenance")
        if status in {
            "supervised_continuous_semantic_grounded_navigation_policy_v3",
            "supervised_continuous_semantic_clearance_navigation_policy_v4",
        } and (
            contract.get("continuous_semantic_grounding_required") is not True
            or contract.get("all_map_voxels_scored_for_grounding") is not True
            or contract.get("query_dependent_grounding_navigation_only") is not True
            or contract.get("oracle_inputs_at_runtime") is not False
            or contract.get("environmental_text_inputs_at_runtime") != []
        ):
            raise ValueError("V3 navigation journal lacks grounded-target provenance")
        if status == "supervised_continuous_semantic_clearance_navigation_policy_v4" and (
            contract.get("numeric_clearance_state_required") is not True
            or contract.get("clearance_from_sanitized_geometry_only") is not True
            or contract.get("clearance_ray_count") != 24
            or contract.get("clearance_max_range_m") != 1.0
            or contract.get("exact_collision_mask_required") is not True
            or contract.get("unsafe_motion_fallback")
            != "highest_safe_nonterminal_action"
            or contract.get("collision_interlock_required") is not True
            or contract.get("static_scene_prefix_question_independent") is not True
        ):
            raise ValueError("V4 navigation journal lacks clearance-safety provenance")
        return {
            "policy_status": status,
            "claimed_trained_navigation_policy": True,
            "navigation_policy_checkpoint_tree_sha256": checkpoint_hash,
        }
    raise ValueError("Navigation journal policy-training status is unsupported")


def _benchmark_feasibility(
    journal: Mapping[str, Any],
    spec: Mapping[str, Any],
    tasks: Mapping[str, Mapping[str, Any]],
    instances: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate an optional preregistered numeric start and progress criteria."""

    version = spec.get("benchmark_version", 1)
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise ValueError("Navigation benchmark version is invalid")
    if version == 1:
        return {
            "benchmark_version": 1,
            "preregistered_numeric_start": False,
            "all_progress_criteria_feasible": None,
        }
    if version != 2:
        raise ValueError("Unsupported navigation benchmark version")
    expected_raw = spec.get("expected_initial_position_xy_m")
    expected = np.asarray(expected_raw, dtype=np.float64)
    justification = spec.get("feasibility_justification")
    contract = journal["header"].get("run_contract")
    config_sha256 = spec.get("inference_config_sha256")
    if (
        expected.shape != (2,)
        or not np.isfinite(expected).all()
        or not isinstance(justification, Mapping)
        or justification.get("criteria_changed_from_v1") is not False
        or justification.get("prepared_before_v2_inference") is not True
        or not isinstance(contract, Mapping)
        or not isinstance(config_sha256, str)
        or _SHA256.fullmatch(config_sha256) is None
        or contract.get("config_sha256") != config_sha256
    ):
        raise ValueError("Navigation-v2 preregistration contract is invalid")
    feasibility_rows: list[dict[str, Any]] = []
    for episode in journal["episodes"]:
        initial = _position(episode["initial_state"])
        if not np.allclose(initial[:2], expected, rtol=0.0, atol=1e-9):
            raise ValueError("Navigation-v2 episode did not use its preregistered start")
        score_task = tasks[episode["task_id"]]
        if episode["family"] not in {"approach", "obstacle", "update_after_scan"}:
            continue
        target = _target_instance(score_task, instances)
        lower, upper = _bbox(target)
        initial_distance = _distance_to_box_xy(initial, lower, upper)
        required_progress = _threshold(score_task, "minimum_target_progress_m", 0.0)
        feasible = required_progress <= initial_distance + 1e-12
        feasibility_rows.append(
            {
                "task_id": episode["task_id"],
                "initial_target_standoff_m": initial_distance,
                "minimum_target_progress_m": required_progress,
                "progress_feasibility_margin_m": initial_distance - required_progress,
                "feasible": feasible,
            }
        )
        if not feasible:
            raise ValueError("Navigation-v2 minimum progress exceeds physically possible progress")
    return {
        "benchmark_version": version,
        "preregistered_numeric_start": True,
        "expected_initial_position_xy_m": expected.tolist(),
        "inference_config_sha256": config_sha256,
        "criteria_changed_from_v1": False,
        "all_progress_criteria_feasible": all(row["feasible"] for row in feasibility_rows),
        "progress_tasks": feasibility_rows,
    }


def score_navigation_journal(
    journal_path: str | Path,
    scoring_spec_path: str | Path,
    scene_oracle_path: str | Path,
) -> dict[str, Any]:
    """Verify sealed inference first, then open physically separate oracle data."""

    journal_file = Path(journal_path)
    journal = validate_navigation_journal(
        json.loads(journal_file.read_text(encoding="utf-8")), require_complete=True
    )
    policy_provenance = _authenticated_policy_provenance(journal)
    context_evidence = continuous_context_metrics(journal)
    if context_evidence["passed"] is not True:
        raise ValueError("Navigation inference did not preserve continuous context causality")
    runtime_audit = journal.get("runtime_file_audit")
    if (
        not isinstance(runtime_audit, Mapping)
        or runtime_audit.get("passed") is not True
        or runtime_audit.get("blocking_enabled") is not True
        or runtime_audit.get("forbidden_accesses") != []
    ):
        raise ValueError("Navigation inference lacks a sealed clean file audit")
    # Do not move either oracle read above journal validation.  A malformed or
    # partial prediction file must fail without touching evaluation metadata.
    scoring_file = Path(scoring_spec_path)
    oracle_file = Path(scene_oracle_path)
    spec = json.loads(scoring_file.read_text(encoding="utf-8"))
    oracle = json.loads(oracle_file.read_text(encoding="utf-8"))
    if not isinstance(spec, dict) or spec.get("schema") != _SCORE_SPEC_SCHEMA:
        raise ValueError("Unsupported navigation oracle-score schema")
    if spec.get("scene_id") != journal["header"].get("scene_id"):
        raise ValueError("Navigation scoring scene differs from inference")
    if spec.get("task_manifest_sha256") != journal["header"].get("task_manifest_sha256"):
        raise ValueError("Navigation score sidecar is not bound to this task manifest")
    if not isinstance(oracle, dict) or oracle.get("scene_id") != spec.get("scene_id"):
        raise ValueError("Scene oracle differs from navigation scoring sidecar")
    raw_instances = oracle.get("instances")
    if not isinstance(raw_instances, list):
        raise TypeError("Scene oracle instances are unavailable")
    instances: dict[str, Mapping[str, Any]] = {}
    for raw in raw_instances:
        if not isinstance(raw, Mapping):
            raise TypeError("Scene oracle instance is invalid")
        instance_id = raw.get("instance_id")
        if isinstance(instance_id, str):
            instances[instance_id] = raw
    task_rows = spec.get("tasks")
    if not isinstance(task_rows, list):
        raise TypeError("Navigation scoring sidecar tasks are invalid")
    by_task: dict[str, Mapping[str, Any]] = {}
    for row in task_rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("task_id"), str):
            raise TypeError("Navigation scoring task is invalid")
        if row["task_id"] in by_task:
            raise ValueError("Navigation scoring task IDs are duplicated")
        by_task[row["task_id"]] = row
    episode_ids = [episode["task_id"] for episode in journal["episodes"]]
    if set(by_task) != set(episode_ids):
        raise ValueError("Navigation scoring tasks differ from prediction journal")
    feasibility = _benchmark_feasibility(journal, spec, by_task, instances)
    scored = [
        _score_episode(episode, by_task[episode["task_id"]], instances)
        for episode in journal["episodes"]
    ]
    family_metrics: dict[str, dict[str, Any]] = {}
    for family in sorted(_TASK_FAMILIES):
        rows = [row for row in scored if row["family"] == family]
        if rows:
            family_metrics[family] = {
                "task_count": len(rows),
                "success_count": sum(row["passed"] for row in rows),
                "success_rate": sum(row["passed"] for row in rows) / len(rows),
            }
    task_count = len(scored)
    success_count = sum(row["passed"] for row in scored)
    collision_count = sum(row["metrics"]["collision_count"] for row in scored)
    action_failures = sum(row["metrics"]["action_failure_count"] for row in scored)
    policy_rejections = sum(row["metrics"]["policy_rejected"] for row in scored)
    learned_policy = policy_provenance["claimed_trained_navigation_policy"] is True
    limitations = (
        [
            (
                "The compact controller was supervised on oracle-derived traces, but this "
                "six-task benchmark contains only one navigation scene."
            ),
            (
                "The numeric robot-state token projector is deterministic; task learning "
                "occurs in the downstream continuous action controller."
            ),
            "Object identities and geometric tolerances were available only to this scorer.",
        ]
        if learned_policy
        else [
            "The current Gemma adapter was not trained on robot tool traces.",
            "The numeric robot-state token projector is deterministic but task-untrained.",
            "This deterministic development benchmark is not held-out navigation generalization.",
            "Object identities and geometric tolerances were available only to this scorer.",
        ]
    )
    return {
        "schema": _SCORE_SCHEMA,
        "scene_id": spec["scene_id"],
        "benchmark_feasibility": feasibility,
        "passed": success_count == task_count and collision_count == 0,
        **policy_provenance,
        "separation": {
            "inference_journal_validated_before_oracle_open": True,
            "inference_received_oracle_or_labels": False,
            "oracle_used_only_by_post_inference_scorer": True,
            "prediction_journal_sha256": journal["journal_sha256"],
            "scoring_spec_sha256": file_sha256(scoring_file),
            "scene_oracle_sha256": file_sha256(oracle_file),
        },
        "continuous_context_evidence": context_evidence,
        "metrics": {
            "task_count": task_count,
            "success_count": success_count,
            "success_rate": success_count / task_count,
            "collision_count": collision_count,
            "action_failure_count": action_failures,
            "policy_rejection_count": policy_rejections,
            "executed_action_count": sum(row["metrics"]["executed_action_count"] for row in scored),
        },
        "by_family": family_metrics,
        "tasks": scored,
        "limitations": limitations,
    }


__all__ = [
    "NavigationTask",
    "NavigationTaskManifest",
    "canonical_sha256",
    "continuous_context_metrics",
    "file_sha256",
    "finalize_navigation_journal_audit",
    "load_task_manifest",
    "numeric_tool_receipt",
    "parse_task_manifest",
    "run_navigation_episode",
    "run_navigation_manifest",
    "score_navigation_journal",
    "tree_sha256",
    "validate_navigation_journal",
    "write_navigation_journal",
]
