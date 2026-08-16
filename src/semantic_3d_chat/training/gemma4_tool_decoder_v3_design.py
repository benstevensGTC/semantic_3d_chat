"""CPU-only design utilities for the unsealed Gemma-4 tool decoder V3.

This module is deliberately incapable of training.  It contains no model
loader, optimizer, MPS call, checkpoint writer, or held-out evaluator.  It
authenticates the published aggregate V2.2 terminal result, reads only the
4,200 training rows at the beginning of the already-authenticated trace, and
defines the fixed V3 sampling and answer-token weighting policies.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

V2_TERMINAL_PATH: Final[Path] = Path(
    "reports/gemma4/metrics/gemma4_embodied_tool_decoder_training_v2_2.json"
)
V2_TERMINAL_SHA256: Final[str] = (
    "fc6cb4a829e8a69aa94c03c13d79c270b1829bb8cddde500e6b6b0fe10cbfc01"
)
V2_RUNTIME_CHECKPOINT: Final[Path] = Path(
    "data_gemma4/checkpoints/gemma4_embodied_tool_decoder_v2/final"
)
TRACE_PATH: Final[Path] = Path(
    "data_gemma4/training/navigation_policy_v3/traces.jsonl"
)
TRAIN_ROW_COUNT: Final[int] = 4_200
TRAIN_PREFIX_BYTES_SHA256: Final[str] = (
    "66048cbb5438906c97a1731154834beb559012910bd4e98932e47df16828be84"
)
TRAIN_SCENES: Final[tuple[str, ...]] = tuple(
    f"scene_{index:06d}" for index in range(11, 25)
)
ACTION_NAMES: Final[tuple[str, ...]] = (
    "stop",
    "scan",
    "turn",
    "move_forward",
    "move_backward",
)
ARGUMENT_BIN_EDGES: Final[tuple[float, ...]] = (-1.0, -0.6, -0.2, 0.2, 0.6, 1.0)
ARGUMENT_BIN_NAMES: Final[tuple[str, ...]] = (
    "neg_extreme",
    "neg_mid",
    "center",
    "pos_mid",
    "pos_extreme",
)
SCHEDULE_SEED: Final[int] = 2_026_081_223
MICROBATCH_COUNT: Final[int] = 800
GRADIENT_ACCUMULATION: Final[int] = 8
OPTIMIZER_UPDATES: Final[int] = MICROBATCH_COUNT // GRADIENT_ACCUMULATION

TOKEN_ROLE_WEIGHTS: Final[dict[str, float]] = {
    "structure": 1.0,
    "schema_key": 1.0,
    "argument_key": 2.0,
    "action": 8.0,
    "argument_value": 6.0,
    "eos": 2.0,
}
_ROLE_PRIORITY: Final[tuple[str, ...]] = (
    "action",
    "argument_value",
    "argument_key",
    "schema_key",
)


@dataclass(frozen=True)
class TrainingRowV3:
    """The minimal training-only fields needed by the V3 design preflight."""

    sample_id: str
    scene_id: str
    family: str
    action_index: int
    action_name: str
    normalized_argument: float


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    """Return the canonical JSON SHA-256 used for schedules and contracts."""

    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def authenticate_v2_2_terminal_negative(
    project_root: Path,
) -> dict[str, Any]:
    """Authenticate only V2.2's published aggregates and terminal nonpublication."""

    report_path = project_root / V2_TERMINAL_PATH
    observed_sha = _sha256(report_path)
    if observed_sha != V2_TERMINAL_SHA256:
        raise ValueError("Gemma tool-decoder V2.2 terminal result bytes changed")
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    metrics = payload.get("all_heldout_teacher_forced")
    gate = payload.get("teacher_forced_early_gate")
    if not isinstance(metrics, Mapping) or not isinstance(gate, Mapping):
        raise TypeError("V2.2 terminal aggregate sections are missing")
    expected_metrics = {
        "sample_count": 2268,
        "scene_count": 8,
        "answer_token_count": 38054,
        "answer_token_nll": 0.37775762747489017,
        "answer_token_accuracy": 0.8712881694434225,
        "exact_sequence_accuracy": 0.17416225749559083,
        "teacher_forced_argmax_valid_schema_rate": 0.2641093474426808,
        "teacher_forced_argmax_canonical_rate": 0.2641093474426808,
        "teacher_forced_argmax_tool_accuracy": 0.24118165784832452,
    }
    for name, expected in expected_metrics.items():
        observed = metrics.get(name)
        if isinstance(expected, float):
            valid = isinstance(observed, (int, float)) and math.isclose(
                float(observed), expected, rel_tol=0.0, abs_tol=1e-15
            )
        else:
            valid = observed == expected
        if not valid:
            raise ValueError(f"V2.2 aggregate metric changed: {name}")
    failed = gate.get("failed")
    if (
        payload.get("schema") != "semantic_3d_chat.gemma4_tool_decoder_training.v2"
        or payload.get("status")
        != "rejected_before_greedy_generation_no_runtime_checkpoint"
        or payload.get("optimizer_updates") != 64
        or payload.get("selected_update") != 64
        or payload.get("greedy_generation_executed") is not False
        or payload.get("runtime_checkpoint_published") is not False
        or gate.get("passed") is not False
        or failed
        != [
            "exact_sequence_accuracy",
            "teacher_forced_argmax_valid_schema_rate",
            "teacher_forced_argmax_tool_accuracy",
        ]
    ):
        raise ValueError("V2.2 is not the exact terminal negative required by V3")
    checkpoint = project_root / V2_RUNTIME_CHECKPOINT
    if checkpoint.exists():
        raise FileExistsError("V2.2 reports no checkpoint, but its final path exists")
    return {
        "path": str(V2_TERMINAL_PATH),
        "sha256": observed_sha,
        "status": payload["status"],
        "optimizer_updates": 64,
        "aggregate_metrics": expected_metrics,
        "failed_checks": list(failed),
        "greedy_generation_executed": False,
        "runtime_checkpoint_published": False,
        "runtime_checkpoint_path": str(V2_RUNTIME_CHECKPOINT),
        "runtime_checkpoint_absent": True,
    }


def load_training_rows_only(project_root: Path) -> tuple[TrainingRowV3, ...]:
    """Read exactly the first 4,200 trace rows and never enter the held-out suffix."""

    path = project_root / TRACE_PATH
    digest = hashlib.sha256()
    rows: list[TrainingRowV3] = []
    with path.open("rb") as handle:
        for index in range(TRAIN_ROW_COUNT):
            raw = handle.readline()
            if not raw:
                raise EOFError("Navigation trace ended inside the fixed training prefix")
            digest.update(raw)
            value = json.loads(raw)
            action_index = value.get("action_index")
            action_name = value.get("action_name")
            argument = value.get("argument_target_normalized")
            if (
                value.get("sample_id") != f"g_{index:08d}"
                or value.get("split") != "train"
                or value.get("scene_id") not in TRAIN_SCENES
                or isinstance(action_index, bool)
                or not isinstance(action_index, int)
                or not 0 <= action_index < len(ACTION_NAMES)
                or action_name != ACTION_NAMES[action_index]
                or isinstance(argument, bool)
                or not isinstance(argument, (int, float))
                or not math.isfinite(float(argument))
                or not -1.0 <= float(argument) <= 1.0
            ):
                raise ValueError(f"V3 rejected training trace row {index}")
            family = value.get("family")
            if not isinstance(family, str) or not family:
                raise ValueError("V3 training row has no family")
            rows.append(
                TrainingRowV3(
                    sample_id=value["sample_id"],
                    scene_id=value["scene_id"],
                    family=family,
                    action_index=action_index,
                    action_name=action_name,
                    normalized_argument=float(argument),
                )
            )
    if digest.hexdigest() != TRAIN_PREFIX_BYTES_SHA256:
        raise ValueError("The exact 4,200-row training prefix bytes changed")
    return tuple(rows)


def argument_bin_name(action_name: str, normalized_argument: float) -> str:
    """Assign a fixed action-specific argument bin without consulting validation."""

    if action_name not in ACTION_NAMES:
        raise ValueError("Unknown V3 action")
    value = float(normalized_argument)
    if not math.isfinite(value) or not -1.0 <= value <= 1.0:
        raise ValueError("V3 normalized argument is outside [-1, 1]")
    if action_name in {"stop", "scan"}:
        if abs(value) > 1e-12:
            raise ValueError("Argument-free V3 action has a nonzero target")
        return "none"
    for index, name in enumerate(ARGUMENT_BIN_NAMES):
        lower = ARGUMENT_BIN_EDGES[index]
        upper = ARGUMENT_BIN_EDGES[index + 1]
        if lower <= value < upper or (index == len(ARGUMENT_BIN_NAMES) - 1 and value == upper):
            return name
    raise RuntimeError("V3 argument-bin edges do not cover the target")


def balanced_schedule_v3(
    rows: Sequence[TrainingRowV3],
    *,
    microbatch_count: int = MICROBATCH_COUNT,
    seed: int = SCHEDULE_SEED,
) -> tuple[int, ...]:
    """Round-robin actions and each action's occupied argument bins exactly."""

    if isinstance(microbatch_count, bool) or not isinstance(microbatch_count, int):
        raise TypeError("V3 microbatch count must be an integer")
    if microbatch_count < 1 or microbatch_count % len(ACTION_NAMES):
        raise ValueError("V3 microbatches must be positive and divisible by five")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("V3 schedule seed must be a nonnegative integer")
    grouped: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for index, row in enumerate(rows):
        grouped[row.action_name][
            argument_bin_name(row.action_name, row.normalized_argument)
        ].append(index)
    if set(grouped) != set(ACTION_NAMES):
        raise ValueError("V3 training rows do not contain every action")
    randomizer = random.Random(seed)
    for bins in grouped.values():
        for indices in bins.values():
            randomizer.shuffle(indices)
    bin_orders = {
        action: sorted(bins, key=lambda name: (name == "none", name))
        for action, bins in grouped.items()
    }
    cursors: dict[tuple[str, str], int] = defaultdict(int)
    result: list[int] = []
    action_cycles = microbatch_count // len(ACTION_NAMES)
    for cycle in range(action_cycles):
        for action in ACTION_NAMES:
            bins = bin_orders[action]
            bin_name = bins[cycle % len(bins)]
            candidates = grouped[action][bin_name]
            key = (action, bin_name)
            cursor = cursors[key]
            result.append(candidates[cursor % len(candidates)])
            cursors[key] = cursor + 1
    return tuple(result)


def schedule_summary_v3(
    rows: Sequence[TrainingRowV3], schedule: Sequence[int]
) -> dict[str, Any]:
    """Summarize and hash a candidate schedule without exposing row semantics."""

    if not schedule or any(index < 0 or index >= len(rows) for index in schedule):
        raise ValueError("V3 schedule contains an invalid training index")
    action_counts = Counter(rows[index].action_name for index in schedule)
    cell_counts = Counter(
        (
            rows[index].action_name,
            argument_bin_name(
                rows[index].action_name, rows[index].normalized_argument
            ),
        )
        for index in schedule
    )
    scene_counts = Counter(rows[index].scene_id for index in schedule)
    family_counts = Counter(rows[index].family for index in schedule)
    ids = [rows[index].sample_id for index in schedule]
    return {
        "microbatch_count": len(schedule),
        "optimizer_updates": len(schedule) // GRADIENT_ACCUMULATION,
        "sample_ids_sha256": canonical_sha256(ids),
        "unique_sample_count": len(set(ids)),
        "action_counts": dict(sorted(action_counts.items())),
        "action_argument_bin_counts": {
            f"{action}:{bin_name}": count
            for (action, bin_name), count in sorted(cell_counts.items())
        },
        "scene_counts": dict(sorted(scene_counts.items())),
        "family_counts": dict(sorted(family_counts.items())),
    }


def canonical_tool_json_v3(row: TrainingRowV3) -> str:
    """Recreate V2's fixed canonical label from training-only numeric targets."""

    action = row.action_name
    value = min(1.0, max(-1.0, row.normalized_argument))
    if action in {"stop", "scan"}:
        arguments: dict[str, float] = {}
    elif action == "turn":
        angle = round(value * 45.0, 3)
        arguments = {"angle_degrees": 0.0 if angle == 0.0 else angle}
    else:
        distance = min(0.5, max(0.02, (value + 1.0) * 0.25))
        distance = round(distance, 3)
        arguments = {"distance_meters": 0.0 if distance == 0.0 else distance}
    return json.dumps(
        {"tool": action, "arguments": arguments},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def answer_character_spans(canonical_json: str) -> dict[str, tuple[tuple[int, int], ...]]:
    """Locate exact semantic spans in one minified canonical JSON answer."""

    parsed = json.loads(canonical_json)
    if not isinstance(parsed, dict) or list(parsed) != ["arguments", "tool"]:
        raise ValueError("V3 requires canonical sorted tool JSON")
    action = parsed.get("tool")
    arguments = parsed.get("arguments")
    if action not in ACTION_NAMES or not isinstance(arguments, dict):
        raise ValueError("V3 canonical answer has an invalid tool envelope")

    def span_of(fragment: str, start: int = 0) -> tuple[int, int]:
        index = canonical_json.find(fragment, start)
        if index < 0 or canonical_json.find(fragment, index + 1) >= 0:
            raise ValueError(f"V3 answer fragment is missing or ambiguous: {fragment}")
        return index, index + len(fragment)

    schema_spans = (span_of("arguments"), span_of("tool"))
    action_span = span_of(str(action), schema_spans[1][1])
    result: dict[str, tuple[tuple[int, int], ...]] = {
        "schema_key": schema_spans,
        "action": (action_span,),
        "argument_key": (),
        "argument_value": (),
    }
    if arguments:
        if len(arguments) != 1:
            raise ValueError("V3 canonical action has more than one argument")
        name, numeric = next(iter(arguments.items()))
        if name not in {"angle_degrees", "distance_meters"} or isinstance(numeric, bool):
            raise ValueError("V3 canonical action has an unsupported argument")
        key_span = span_of(name)
        value_text = json.dumps(numeric, allow_nan=False, separators=(",", ":"))
        value_start = canonical_json.find(value_text, key_span[1])
        if value_start < 0:
            raise ValueError("V3 canonical numeric value span is missing")
        result["argument_key"] = (key_span,)
        result["argument_value"] = ((value_start, value_start + len(value_text)),)
    return result


def token_roles_and_weights(
    canonical_json: str,
    offsets: Iterable[Sequence[int]],
    *,
    append_eos: bool = True,
) -> tuple[tuple[str, ...], tuple[float, ...]]:
    """Map tokenizer offsets to fixed answer roles with semantic-first priority."""

    spans = answer_character_spans(canonical_json)
    roles: list[str] = []
    previous_end = 0
    for raw in offsets:
        if len(raw) != 2:
            raise ValueError("V3 tokenizer offset must contain start and end")
        start, end = int(raw[0]), int(raw[1])
        if start < previous_end or end <= start or end > len(canonical_json):
            raise ValueError("V3 tokenizer offsets are invalid or reordered")
        previous_end = end
        role = "structure"
        for candidate in _ROLE_PRIORITY:
            if any(start < span_end and end > span_start for span_start, span_end in spans[candidate]):
                role = candidate
                break
        roles.append(role)
    if append_eos:
        roles.append("eos")
    return tuple(roles), tuple(TOKEN_ROLE_WEIGHTS[role] for role in roles)


def weighted_loss_from_token_losses(
    token_losses: Sequence[float], token_weights: Sequence[float]
) -> float:
    """Reference CPU scalar for the preregistered per-answer normalized objective."""

    if len(token_losses) != len(token_weights) or not token_losses:
        raise ValueError("V3 token losses and weights must be nonempty and aligned")
    losses = [float(value) for value in token_losses]
    weights = [float(value) for value in token_weights]
    if any(not math.isfinite(value) or value < 0.0 for value in losses):
        raise ValueError("V3 token loss must be finite and nonnegative")
    if any(not math.isfinite(value) or value <= 0.0 for value in weights):
        raise ValueError("V3 token weight must be finite and positive")
    return sum(loss * weight for loss, weight in zip(losses, weights, strict=True)) / sum(weights)


__all__ = [
    "ACTION_NAMES",
    "ARGUMENT_BIN_NAMES",
    "GRADIENT_ACCUMULATION",
    "MICROBATCH_COUNT",
    "OPTIMIZER_UPDATES",
    "SCHEDULE_SEED",
    "TOKEN_ROLE_WEIGHTS",
    "TRAIN_PREFIX_BYTES_SHA256",
    "TRAIN_ROW_COUNT",
    "V2_RUNTIME_CHECKPOINT",
    "V2_TERMINAL_PATH",
    "V2_TERMINAL_SHA256",
    "TrainingRowV3",
    "answer_character_spans",
    "argument_bin_name",
    "authenticate_v2_2_terminal_negative",
    "balanced_schedule_v3",
    "canonical_sha256",
    "canonical_tool_json_v3",
    "load_training_rows_only",
    "schedule_summary_v3",
    "token_roles_and_weights",
    "weighted_loss_from_token_losses",
]
