"""Exact JSON, causal-control, and collision metrics for tool-decoder V2."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Final

import torch

from semantic_3d_chat.language.gemma4_answer_tail import answer_tail_forward
from semantic_3d_chat.language.gemma4_tool_decoder_v2 import (
    NumericToolContextProjectorV2,
)
from semantic_3d_chat.robot.llm_tool_policy import validate_tool_call_text
from semantic_3d_chat.robot.navigation_policy import (
    ACTION_NAMES,
    ACTION_TO_INDEX,
    normalized_argument_for_action,
)
from semantic_3d_chat.training.gemma4_tool_decoder_v2_data import (
    CAUSAL_VALIDATION_SAMPLE_COUNT,
    CAUSAL_VALIDATION_SAMPLE_IDS_SHA256,
    GREEDY_CONTROL_SAMPLE_COUNT_V2_1,
    GREEDY_CONTROL_SAMPLE_IDS_SHA256_V2_1,
    ToolDecoderDatasetV2,
    causal_validation_indices_v2,
    controlled_sample_inputs_v2,
    greedy_control_validation_indices_v2_1,
    prepare_microbatch_v2,
)

CONTROL_MODES: Final[tuple[str, ...]] = (
    "primary",
    "wrong_scene",
    "zero_scene",
    "wrong_robot",
    "zero_robot",
    "wrong_target",
    "zero_target",
    "wrong_clearance",
    "zero_clearance",
)

GenerationFunction = Callable[[ToolDecoderDatasetV2, int, str], str]
TeacherForcedFunction = Callable[
    [ToolDecoderDatasetV2, int, str], Mapping[str, Any]
]


def _canonical_json(value: object) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )


def _prediction(
    text: object,
    config: Mapping[str, Any],
    *,
    max_turn_degrees: float,
    max_move_m: float,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "valid_schema": False,
        "canonical_output": False,
        "tool": None,
        "action_index": None,
        "normalized_argument": None,
        "error_code": None,
    }
    validation = validate_tool_call_text(text, config)
    result["error_code"] = validation.error_code
    if validation.call is None or validation.error_code is not None:
        return result
    call = validation.call
    if call.name not in ACTION_TO_INDEX:
        result["error_code"] = "E_VOCABULARY"
        return result
    result["valid_schema"] = True
    result["canonical_output"] = text == call.canonical_json
    result["tool"] = call.name
    result["action_index"] = ACTION_TO_INDEX[call.name]
    if call.name == "turn":
        argument = call.arguments["angle_degrees"]
    elif call.name in {"move_forward", "move_backward"}:
        argument = call.arguments["distance_meters"]
    else:
        argument = 0.0
    result["normalized_argument"] = normalized_argument_for_action(
        call.name,
        float(argument),
        max_turn_degrees=max_turn_degrees,
        max_move_m=max_move_m,
    )
    return result


def _unsafe_motion(
    prediction: Mapping[str, Any],
    clearance: torch.Tensor,
    *,
    max_range_m: float = 1.0,
    max_move_m: float,
) -> bool:
    tool = prediction.get("tool")
    normalized = prediction.get("normalized_argument")
    if tool not in {"move_forward", "move_backward"} or not isinstance(
        normalized, (int, float)
    ):
        return False
    distance = max(0.02, (float(normalized) + 1.0) * 0.5 * max_move_m)
    ray = 0 if tool == "move_forward" else len(clearance) // 2
    return distance >= float(clearance[ray]) * max_range_m - 1e-6


def analyze_canonical_json_vocabulary_v2(
    tokenizer: Any,
    *,
    max_turn_degrees: float,
    max_move_m: float,
) -> dict[str, Any]:
    """Prove every fixed JSON grammar form round-trips through the tokenizer."""

    examples = (
        '{"arguments":{},"tool":"stop"}',
        '{"arguments":{},"tool":"scan"}',
        f'{{"arguments":{{"angle_degrees":{-max_turn_degrees:g}}},"tool":"turn"}}',
        f'{{"arguments":{{"angle_degrees":{max_turn_degrees:g}}},"tool":"turn"}}',
        '{"arguments":{"distance_meters":0.02},"tool":"move_forward"}',
        f'{{"arguments":{{"distance_meters":{max_move_m:g}}},"tool":"move_forward"}}',
        '{"arguments":{"distance_meters":0.02},"tool":"move_backward"}',
        f'{{"arguments":{{"distance_meters":{max_move_m:g}}},"tool":"move_backward"}}',
    )
    records: list[dict[str, Any]] = []
    unique_ids: set[int] = set()
    for value in examples:
        encoded = tokenizer(value, add_special_tokens=False, return_tensors="pt")
        ids = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded.input_ids
        if not isinstance(ids, torch.Tensor) or ids.ndim != 2 or ids.shape[1] < 1:
            raise ValueError("V2 tokenizer produced no canonical JSON tokens")
        decoded = tokenizer.decode(ids[0].tolist(), skip_special_tokens=False)
        if decoded != value:
            raise ValueError(f"V2 canonical JSON does not round-trip: {decoded!r} != {value!r}")
        unique_ids.update(int(item) for item in ids[0])
        records.append({"json": value, "token_count": int(ids.shape[1])})
    eos = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos, bool) or not isinstance(eos, int) or eos < 0:
        raise ValueError("V2 tokenizer has no EOS token")
    return {
        "schema": "semantic_3d_chat.gemma4_tool_decoder_json_vocabulary.v2",
        "tool_vocabulary": list(ACTION_NAMES),
        "grammar_form_count": len(records),
        "forms": records,
        "unique_token_id_count": len(unique_ids),
        "eos_token_id": eos,
        "all_forms_exact_roundtrip": True,
        "canonical_key_order": ["arguments", "tool"],
    }


def generate_tool_json_v2(
    dataset: ToolDecoderDatasetV2,
    index: int,
    control: str,
    *,
    language: Any,
    projector: NumericToolContextProjectorV2,
    max_turn_degrees: float,
    max_move_m: float,
    max_new_tokens: int = 24,
) -> str:
    """Greedily decode one JSON proposal from continuous context."""

    prepared, _sample = prepare_microbatch_v2(
        dataset,
        index,
        language=language,
        projector=projector,
        max_turn_degrees=max_turn_degrees,
        max_move_m=max_move_m,
        control=control,
        include_answer=False,
    )
    backend = language.prefix_backend
    generated = backend.generate(
        prepared,
        max_new_tokens=max_new_tokens,
        eos_token_ids=language.tokenizer.eos_token_id,
    )
    if not isinstance(generated, torch.Tensor) or generated.ndim != 2:
        raise RuntimeError("V2 Gemma generation returned invalid token IDs")
    return language.tokenizer.decode(
        generated[0].detach().cpu().tolist(), skip_special_tokens=True
    ).strip()


@torch.inference_mode()
def teacher_forced_row_v2(
    dataset: ToolDecoderDatasetV2,
    index: int,
    control: str,
    *,
    language: Any,
    projector: NumericToolContextProjectorV2,
    config: Mapping[str, Any],
    max_turn_degrees: float,
    max_move_m: float,
) -> dict[str, Any]:
    """Score one complete canonical JSON suffix without autoregressive decoding."""

    prepared, sample = prepare_microbatch_v2(
        dataset,
        index,
        language=language,
        projector=projector,
        max_turn_degrees=max_turn_degrees,
        max_move_m=max_move_m,
        control=control,
        include_answer=True,
    )
    tail = answer_tail_forward(language, prepared)
    token_count = int(tail.targets.numel())
    if token_count < 2:
        raise RuntimeError("V2 teacher-forced row has no JSON answer suffix")
    predicted = tail.logits[0].float().argmax(dim=-1)
    token_correct = int((predicted == tail.targets).sum().item())
    token_nll_sum = tail.per_token_nll.sum()
    if not torch.isfinite(token_nll_sum):
        raise RuntimeError("V2 teacher-forced JSON NLL is nonfinite")
    predicted_text = language.tokenizer.decode(
        predicted[:-1].detach().cpu().tolist(), skip_special_tokens=True
    ).strip()
    validation = validate_tool_call_text(predicted_text, config)
    predicted_tool = None if validation.call is None else validation.call.name
    return {
        "sample_id": sample.sample_id,
        "control": control,
        "token_nll_sum": float(token_nll_sum.detach().cpu()),
        "answer_token_count": token_count,
        "answer_token_correct": token_correct,
        "exact_sequence": token_correct == token_count,
        "teacher_forced_argmax_valid_schema": validation.error_code is None,
        "teacher_forced_argmax_canonical": bool(
            validation.call is not None
            and predicted_text == validation.call.canonical_json
        ),
        "teacher_forced_argmax_tool": predicted_tool,
        "teacher_forced_argmax_tool_correct": predicted_tool == sample.action_name,
    }


def evaluate_all_heldout_teacher_forced_v2(
    dataset: ToolDecoderDatasetV2,
    score: TeacherForcedFunction,
) -> dict[str, Any]:
    """Gate costly generation using every one of the 2,268 held-out rows."""

    nll_sum = 0.0
    token_count = 0
    token_correct = 0
    exact = 0
    schema_valid = 0
    canonical = 0
    tool_correct = 0
    scenes: set[str] = set()
    for index in dataset.validation_indices:
        sample = dataset.samples[index]
        row = score(dataset, index, "primary")
        if row.get("sample_id") != sample.sample_id:
            raise ValueError("V2 teacher-forced callback reordered validation rows")
        row_nll = row.get("token_nll_sum")
        row_tokens = row.get("answer_token_count")
        row_correct = row.get("answer_token_correct")
        if (
            isinstance(row_nll, bool)
            or not isinstance(row_nll, (int, float))
            or not math.isfinite(float(row_nll))
            or float(row_nll) < 0.0
            or isinstance(row_tokens, bool)
            or not isinstance(row_tokens, int)
            or row_tokens < 1
            or isinstance(row_correct, bool)
            or not isinstance(row_correct, int)
            or not 0 <= row_correct <= row_tokens
            or row.get("control") != "primary"
            or not isinstance(row.get("exact_sequence"), bool)
        ):
            raise ValueError("V2 teacher-forced callback returned invalid metrics")
        nll_sum += float(row_nll)
        token_count += row_tokens
        token_correct += row_correct
        exact += int(row["exact_sequence"])
        schema_valid += int(row.get("teacher_forced_argmax_valid_schema") is True)
        canonical += int(row.get("teacher_forced_argmax_canonical") is True)
        tool_correct += int(row.get("teacher_forced_argmax_tool_correct") is True)
        scenes.add(sample.scene_id)
    sample_count = len(dataset.validation_indices)
    return {
        "schema": "semantic_3d_chat.gemma4_tool_decoder_teacher_forced.v2",
        "held_out_scenes_only": True,
        "all_heldout_rows_scored": True,
        "sample_count": sample_count,
        "expected_sample_count": 2268,
        "scene_count": len(scenes),
        "answer_token_count": token_count,
        "answer_token_nll": nll_sum / token_count,
        "answer_token_accuracy": token_correct / token_count,
        "exact_sequence_accuracy": exact / sample_count,
        "teacher_forced_argmax_valid_schema_rate": schema_valid / sample_count,
        "teacher_forced_argmax_canonical_rate": canonical / sample_count,
        "teacher_forced_argmax_tool_accuracy": tool_correct / sample_count,
        "environmental_text_inputs": [],
        "oracle_inputs_at_runtime": False,
    }


def teacher_forced_gate_results_v2(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Fail before greedy generation unless canonical JSON is actually learned."""

    checks = {
        "all_heldout_rows_scored": metrics.get("all_heldout_rows_scored") is True,
        "sample_count": metrics.get("sample_count") == 2268,
        "scene_count": metrics.get("scene_count") == 8,
        "answer_token_nll": float(metrics.get("answer_token_nll", math.inf)) <= 2.0,
        "answer_token_accuracy": float(
            metrics.get("answer_token_accuracy", -math.inf)
        )
        >= 0.80,
        "exact_sequence_accuracy": float(
            metrics.get("exact_sequence_accuracy", -math.inf)
        )
        >= 0.30,
        "teacher_forced_argmax_valid_schema_rate": float(
            metrics.get("teacher_forced_argmax_valid_schema_rate", -math.inf)
        )
        >= 0.80,
        "teacher_forced_argmax_tool_accuracy": float(
            metrics.get("teacher_forced_argmax_tool_accuracy", -math.inf)
        )
        >= 0.70,
    }
    return {
        "schema": "semantic_3d_chat.gemma4_tool_decoder_teacher_forced_gate.v2",
        "checks": checks,
        "passed": all(checks.values()),
        "failed": [name for name, passed in checks.items() if not passed],
        "evaluated_before_greedy_generation": True,
    }


def _teacher_forced_condition_v2(
    dataset: ToolDecoderDatasetV2,
    score: TeacherForcedFunction,
    *,
    control: str,
    indices: Sequence[int],
) -> dict[str, Any]:
    totals: Counter[str] = Counter()
    by_family: dict[str, Counter[str]] = defaultdict(Counter)
    for index in indices:
        sample = dataset.samples[index]
        row = score(dataset, index, control)
        if row.get("sample_id") != sample.sample_id or row.get("control") != control:
            raise ValueError("V2 causal teacher-forced callback reordered rows")
        nll = row.get("token_nll_sum")
        tokens = row.get("answer_token_count")
        correct = row.get("answer_token_correct")
        if (
            isinstance(nll, bool)
            or not isinstance(nll, (int, float))
            or not math.isfinite(float(nll))
            or float(nll) < 0.0
            or isinstance(tokens, bool)
            or not isinstance(tokens, int)
            or tokens < 1
            or isinstance(correct, bool)
            or not isinstance(correct, int)
            or not 0 <= correct <= tokens
        ):
            raise ValueError("V2 causal teacher-forced callback returned invalid metrics")
        increments = {
            "sample_count": 1,
            "token_count": tokens,
            "token_correct": correct,
            "nll_micros": round(float(nll) * 1_000_000),
            "exact": int(row.get("exact_sequence") is True),
            "schema": int(row.get("teacher_forced_argmax_valid_schema") is True),
            "canonical": int(row.get("teacher_forced_argmax_canonical") is True),
            "tool": int(row.get("teacher_forced_argmax_tool_correct") is True),
        }
        totals.update(increments)
        by_family[sample.family].update(increments)

    def metrics(values: Counter[str]) -> dict[str, Any]:
        sample_count = values["sample_count"]
        token_count = values["token_count"]
        return {
            "sample_count": sample_count,
            "answer_token_count": token_count,
            "answer_token_nll": values["nll_micros"] / 1_000_000 / token_count,
            "answer_token_accuracy": values["token_correct"] / token_count,
            "exact_sequence_accuracy": values["exact"] / sample_count,
            "teacher_forced_argmax_valid_schema_rate": values["schema"] / sample_count,
            "teacher_forced_argmax_canonical_rate": values["canonical"] / sample_count,
            "teacher_forced_argmax_tool_accuracy": values["tool"] / sample_count,
        }

    return {
        "control": control,
        **metrics(totals),
        "by_family": {
            family: metrics(values) for family, values in sorted(by_family.items())
        },
    }


def evaluate_teacher_forced_causal_controls_v2(
    dataset: ToolDecoderDatasetV2,
    score: TeacherForcedFunction,
) -> dict[str, Any]:
    """Score all nine contexts on the sealed 448-row causal subset."""

    indices = causal_validation_indices_v2(dataset)
    conditions = {
        control: _teacher_forced_condition_v2(
            dataset, score, control=control, indices=indices
        )
        for control in CONTROL_MODES
    }
    primary = conditions["primary"]
    drops: dict[str, Any] = {}
    for control, condition in conditions.items():
        if control == "primary":
            continue
        drops[control] = {
            "answer_token_nll_increase": (
                condition["answer_token_nll"] - primary["answer_token_nll"]
            ),
            "answer_token_accuracy_drop": (
                primary["answer_token_accuracy"] - condition["answer_token_accuracy"]
            ),
            "exact_sequence_accuracy_drop": (
                primary["exact_sequence_accuracy"]
                - condition["exact_sequence_accuracy"]
            ),
            "teacher_forced_argmax_tool_accuracy_drop": (
                primary["teacher_forced_argmax_tool_accuracy"]
                - condition["teacher_forced_argmax_tool_accuracy"]
            ),
            "by_family": {
                family: {
                    "answer_token_nll_increase": (
                        condition["by_family"][family]["answer_token_nll"]
                        - primary["by_family"][family]["answer_token_nll"]
                    ),
                    "teacher_forced_argmax_tool_accuracy_drop": (
                        primary["by_family"][family][
                            "teacher_forced_argmax_tool_accuracy"
                        ]
                        - condition["by_family"][family][
                            "teacher_forced_argmax_tool_accuracy"
                        ]
                    ),
                }
                for family in primary["by_family"]
            },
        }
    return {
        "schema": "semantic_3d_chat.gemma4_tool_decoder_teacher_causal.v2_1",
        "held_out_scenes_only": True,
        "sample_count_per_condition": len(indices),
        "sample_ids_sha256": CAUSAL_VALIDATION_SAMPLE_IDS_SHA256,
        "condition_count": len(CONTROL_MODES),
        "teacher_forced_forward_count": len(indices) * len(CONTROL_MODES),
        "conditions": conditions,
        "drops_from_primary": drops,
        "environmental_text_inputs": [],
        "oracle_inputs_at_runtime": False,
    }


def teacher_forced_causal_gate_results_v2(
    evaluation: Mapping[str, Any],
) -> dict[str, Any]:
    """Require every continuous modality to change held-out answer likelihood."""

    drops = evaluation.get("drops_from_primary")
    if not isinstance(drops, Mapping):
        raise TypeError("V2.1 teacher-forced causal drops are unavailable")

    def increase(control: str, families: Sequence[str] | None = None) -> float:
        condition = drops.get(control)
        if not isinstance(condition, Mapping):
            return -math.inf
        if families is None:
            return float(condition.get("answer_token_nll_increase", -math.inf))
        by_family = condition.get("by_family")
        if not isinstance(by_family, Mapping):
            return -math.inf
        values = [
            float(by_family.get(family, {}).get("answer_token_nll_increase", -math.inf))
            for family in families
        ]
        return sum(values) / len(values)

    checks = {
        "sample_count_per_condition": evaluation.get("sample_count_per_condition")
        == CAUSAL_VALIDATION_SAMPLE_COUNT,
        "wrong_scene_nll_increase": increase("wrong_scene") >= 0.01,
        "zero_scene_nll_increase": increase("zero_scene") >= 0.01,
        "wrong_robot_targeted_nll_increase": increase(
            "wrong_robot", ("face", "approach", "left_right")
        )
        >= 0.01,
        "zero_robot_targeted_nll_increase": increase(
            "zero_robot", ("face", "approach", "left_right")
        )
        >= 0.01,
        "wrong_target_targeted_nll_increase": increase(
            "wrong_target", ("face", "approach", "left_right")
        )
        >= 0.02,
        "zero_target_targeted_nll_increase": increase(
            "zero_target", ("face", "approach", "left_right")
        )
        >= 0.02,
        "wrong_clearance_targeted_nll_increase": increase(
            "wrong_clearance", ("obstacle", "collision_recovery")
        )
        >= 0.01,
        "zero_clearance_targeted_nll_increase": increase(
            "zero_clearance", ("obstacle", "collision_recovery")
        )
        >= 0.01,
    }
    return {
        "schema": "semantic_3d_chat.gemma4_tool_decoder_teacher_causal_gate.v2_1",
        "checks": checks,
        "passed": all(checks.values()),
        "failed": [name for name, passed in checks.items() if not passed],
        "evaluated_before_greedy_generation": True,
    }
def evaluate_control_v2(
    dataset: ToolDecoderDatasetV2,
    config: Mapping[str, Any],
    generate: GenerationFunction,
    *,
    control: str,
    indices: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Evaluate one condition on held-out scenes with exact structured scoring."""

    if control not in CONTROL_MODES:
        raise ValueError("V2 evaluation control is not preregistered")
    selected = tuple(dataset.validation_indices if indices is None else indices)
    if not selected or any(index not in dataset.validation_indices for index in selected):
        raise ValueError("V2 evaluation is restricted to held-out validation scenes")
    robot = config.get("robot")
    if not isinstance(robot, Mapping):
        raise TypeError("V2 evaluation config has no robot mapping")
    max_turn = float(robot["max_turn_degrees"])
    max_move = float(robot["max_move_m"])
    exact = 0
    schema = 0
    canonical = 0
    tool_correct = 0
    numeric_error: list[float] = []
    turn_total = 0
    turn_sign_correct = 0
    unsafe = 0
    by_family: dict[str, Counter[str]] = defaultdict(Counter)
    errors: Counter[str] = Counter()
    action_predictions: Counter[str] = Counter()
    for index in selected:
        sample = dataset.samples[index]
        raw = generate(dataset, index, control)
        prediction = _prediction(
            raw, config, max_turn_degrees=max_turn, max_move_m=max_move
        )
        exact_match = raw == sample.canonical_answer
        exact += int(exact_match)
        schema += int(prediction["valid_schema"])
        canonical += int(prediction["canonical_output"])
        correct_tool = prediction["tool"] == sample.action_name
        tool_correct += int(correct_tool)
        family = by_family[sample.family]
        family["count"] += 1
        family["exact"] += int(exact_match)
        family["tool"] += int(correct_tool)
        if prediction["error_code"] is not None:
            errors[str(prediction["error_code"])] += 1
        if prediction["tool"] in ACTION_TO_INDEX:
            action_predictions[str(prediction["tool"])] += 1
        predicted_argument = prediction["normalized_argument"]
        if sample.action_name in {"turn", "move_forward", "move_backward"} and isinstance(
            predicted_argument, (int, float)
        ):
            numeric_error.append(abs(float(predicted_argument) - sample.normalized_argument))
        if sample.action_name == "turn":
            turn_total += 1
            if isinstance(predicted_argument, (int, float)) and correct_tool:
                turn_sign_correct += int(
                    math.copysign(1.0, float(predicted_argument))
                    == math.copysign(1.0, sample.normalized_argument)
                )
        _active, _target, clearance, _sample = controlled_sample_inputs_v2(
            dataset, index, control=control
        )
        unsafe += int(
            _unsafe_motion(prediction, clearance[0], max_move_m=max_move)
        )
    count = len(selected)
    return {
        "schema": "semantic_3d_chat.gemma4_tool_decoder_control_metrics.v2",
        "control": control,
        "held_out_scenes_only": True,
        "sample_count": count,
        "scene_count": len({dataset.samples[index].scene_id for index in selected}),
        "exact_json_accuracy": exact / count,
        "valid_schema_rate": schema / count,
        "canonical_json_rate": canonical / count,
        "tool_accuracy": tool_correct / count,
        "argument_mae_normalized": (
            sum(numeric_error) / len(numeric_error) if numeric_error else 1.0
        ),
        "turn_sign_accuracy": (
            turn_sign_correct / turn_total if turn_total else 0.0
        ),
        "unsafe_motion_count": unsafe,
        "collision_risk_rate": unsafe / count,
        "collision_safe_proposal_rate": 1.0 - unsafe / count,
        "predicted_action_counts": {
            name: int(action_predictions.get(name, 0)) for name in ACTION_NAMES
        },
        "validation_error_counts": dict(sorted(errors.items())),
        "by_family": {
            name: {
                "sample_count": values["count"],
                "exact_json_accuracy": values["exact"] / values["count"],
                "tool_accuracy": values["tool"] / values["count"],
            }
            for name, values in sorted(by_family.items())
        },
    }


def evaluate_causal_controls_v2(
    dataset: ToolDecoderDatasetV2,
    config: Mapping[str, Any],
    generate: GenerationFunction,
    *,
    indices: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Run fixed bounded greedy subsets after the all-row teacher-forced gate.

    V2.1 bounds greedy decoding to 896 sequences: 448 stratified primary rows
    plus one row from every scene/family stratum (56) for each of eight altered
    controls. A memoizer prevents their primary rows from being decoded twice.
    """

    if indices is None:
        causal_indices = greedy_control_validation_indices_v2_1(dataset)
        primary_indices = causal_validation_indices_v2(dataset)
    else:
        causal_indices = tuple(indices)
        primary_indices = causal_indices
    cache: dict[tuple[int, str], str] = {}

    def memoized(
        dataset_value: ToolDecoderDatasetV2, index: int, control: str
    ) -> str:
        key = (index, control)
        if key not in cache:
            cache[key] = generate(dataset_value, index, control)
        return cache[key]

    primary_large = evaluate_control_v2(
        dataset, config, memoized, control="primary", indices=primary_indices
    )
    conditions = {
        mode: evaluate_control_v2(
            dataset, config, memoized, control=mode, indices=causal_indices
        )
        for mode in CONTROL_MODES
    }
    primary = conditions["primary"]
    drops = {
        mode: {
            "exact_json_accuracy_drop": primary["exact_json_accuracy"]
            - conditions[mode]["exact_json_accuracy"],
            "tool_accuracy_drop": primary["tool_accuracy"]
            - conditions[mode]["tool_accuracy"],
            "turn_sign_accuracy_drop": primary["turn_sign_accuracy"]
            - conditions[mode]["turn_sign_accuracy"],
            "collision_safe_proposal_rate_drop": primary[
                "collision_safe_proposal_rate"
            ]
            - conditions[mode]["collision_safe_proposal_rate"],
        }
        for mode in CONTROL_MODES
        if mode != "primary"
    }
    greedy_change_rates = {
        mode: sum(
            memoized(dataset, index, mode)
            != memoized(dataset, index, "primary")
            for index in causal_indices
        )
        / len(causal_indices)
        for mode in CONTROL_MODES
        if mode != "primary"
    }
    return {
        "schema": "semantic_3d_chat.gemma4_tool_decoder_causal_controls.v2",
        "held_out_scenes_only": True,
        "train_scene_ids": [f"scene_{index:06d}" for index in range(11, 25)],
        "validation_scene_ids": [
            "scene_000031",
            "scene_000032",
            "scene_000033",
            "scene_000034",
            "scene_000035",
            "scene_000036",
            "scene_000037",
            "scene_000039",
        ],
        "scene_splits_disjoint": True,
        "evaluation_sampling": {
            "selection_timing": "sealed_before_training_or_generation",
            "algorithm": (
                "lexicographic_sample_id_with_action_round_robin_inside_each_"
                "scene_family_stratum"
            ),
            "causal_rows_per_scene_family": 1,
            "causal_sample_count": len(causal_indices),
            "causal_sample_ids_sha256": (
                GREEDY_CONTROL_SAMPLE_IDS_SHA256_V2_1 if indices is None else None
            ),
            "primary_rows_per_scene_family": 8 if indices is None else None,
            "primary_sample_count": len(primary_indices),
            "primary_sample_ids_sha256": (
                CAUSAL_VALIDATION_SAMPLE_IDS_SHA256 if indices is None else None
            ),
            "total_unique_greedy_generations": len(cache),
            "expected_total_unique_greedy_generations": (
                CAUSAL_VALIDATION_SAMPLE_COUNT
                + (len(CONTROL_MODES) - 1) * GREEDY_CONTROL_SAMPLE_COUNT_V2_1
                if indices is None
                else len(causal_indices) * len(CONTROL_MODES)
            ),
        },
        "primary_large": primary_large,
        "conditions": conditions,
        "drops_from_primary": drops,
        "greedy_output_change_rate_from_primary": greedy_change_rates,
        "environmental_text_inputs": [],
        "oracle_inputs_at_runtime": False,
    }


def promotion_gate_results_v2(evaluation: Mapping[str, Any]) -> dict[str, Any]:
    """Apply every preregistered quality and causal-context promotion gate."""

    conditions = evaluation.get("conditions")
    drops = evaluation.get("drops_from_primary")
    if not isinstance(conditions, Mapping) or not isinstance(drops, Mapping):
        raise TypeError("V2 promotion evaluation is incomplete")
    primary = conditions.get("primary")
    primary_quality = evaluation.get("primary_large", primary)
    if not isinstance(primary, Mapping) or not isinstance(primary_quality, Mapping):
        raise TypeError("V2 promotion evaluation has no primary condition")
    teacher_metrics = evaluation.get("all_heldout_teacher_forced")
    teacher_gate = evaluation.get("teacher_forced_early_gate")
    teacher_causal = evaluation.get("teacher_forced_causal_controls")
    teacher_causal_gate = evaluation.get("teacher_forced_causal_gate")
    if not isinstance(teacher_metrics, Mapping) or not isinstance(teacher_gate, Mapping):
        raise TypeError("V2 promotion evaluation lacks its all-row teacher-forced gate")
    if not isinstance(teacher_causal, Mapping) or not isinstance(
        teacher_causal_gate, Mapping
    ):
        raise TypeError("V2.1 promotion evaluation lacks teacher-forced causal gates")
    recomputed_teacher_gate = teacher_forced_gate_results_v2(teacher_metrics)
    recomputed_causal_gate = teacher_forced_causal_gate_results_v2(teacher_causal)
    change_rates = evaluation.get("greedy_output_change_rate_from_primary")
    if not isinstance(change_rates, Mapping):
        raise TypeError("V2.1 promotion evaluation lacks greedy causal diagnostics")
    checks = {
        "teacher_forced_early_gate": (
            teacher_gate == recomputed_teacher_gate
            and recomputed_teacher_gate.get("passed") is True
        ),
        "teacher_forced_causal_gate": (
            teacher_causal_gate == recomputed_causal_gate
            and recomputed_causal_gate.get("passed") is True
        ),
        "exact_json_accuracy": float(
            primary_quality.get("exact_json_accuracy", -1.0)
        )
        >= 0.60,
        "valid_schema_rate": float(primary_quality.get("valid_schema_rate", -1.0))
        >= 0.95,
        "tool_accuracy": float(primary_quality.get("tool_accuracy", -1.0)) >= 0.80,
        "turn_sign_accuracy": float(
            primary_quality.get("turn_sign_accuracy", -1.0)
        )
        >= 0.80,
        "argument_mae_normalized": float(
            primary_quality.get("argument_mae_normalized", math.inf)
        )
        <= 0.25,
        "collision_execution_count": int(
            primary_quality.get("unsafe_motion_count", -1)
        )
        == 0,
        "wrong_clearance_changes_greedy_output": float(
            change_rates.get("wrong_clearance", -math.inf)
        )
        >= 0.10,
        "zero_clearance_changes_greedy_output": float(
            change_rates.get("zero_clearance", -math.inf)
        )
        >= 0.10,
    }
    return {
        "schema": "semantic_3d_chat.gemma4_tool_decoder_promotion_gates.v2",
        "checks": checks,
        "passed": all(checks.values()),
        "failed": [name for name, passed in checks.items() if not passed],
    }


__all__ = [
    "CONTROL_MODES",
    "analyze_canonical_json_vocabulary_v2",
    "evaluate_all_heldout_teacher_forced_v2",
    "evaluate_causal_controls_v2",
    "evaluate_control_v2",
    "generate_tool_json_v2",
    "promotion_gate_results_v2",
    "teacher_forced_gate_results_v2",
    "teacher_forced_row_v2",
]
