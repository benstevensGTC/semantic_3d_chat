"""Model-free sealed contract for V97's V96-continuation repair.

V97 may use only the immutable V96 fixed-final adapter, V96's row-free sealed
aggregate outcome, and the existing forty-scene training pool.  This module
never opens V96 row-level development questions, predictions, labels, NLL
rows, or physically absent deferred-final scenes.  A final update is fixed by
schedule before any V97 development scoring can occur.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import torch
import yaml
from safetensors.torch import load_file
from torch import nn

from semantic_3d_chat.chat.file_audit import FileAccessAudit
from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.evaluation.v85_strict_multiscene_preflight import (
    canonical_sha256_v85,
    sha256_file_v85,
)
from semantic_3d_chat.evaluation.v96_atomic_pair_repair_preflight import (
    DEFERRED_FINAL_SCENES,
    PRIOR_EVALUATION_SCENES,
    PairUnitV96,
    RowV96,
    balanced_class_weights_v96,
    family_weights_v96,
    load_scene_memories_v96,
    load_training_rows_v96,
    pair_units_v96,
)
from semantic_3d_chat.language.lora import LoRASettings, install_lora_adapters

CONFIG: Final[Path] = Path(
    "configs/experiments/gemma4_v97_pair_decision_stability.yaml"
)
FRESH_BANK_NAME: Final[str] = "v97_pair_decision_stability_bridge"
TARGET_MODULES: Final[tuple[str, ...]] = (
    "model.language_model.layers.9.self_attn.q_proj",
)
PINNED_TENSORS: Final[dict[str, list[int]]] = {
    TARGET_MODULES[0] + ".weight": [4096, 1536],
}
FRESH_PARAMETER_COUNT: Final[int] = 45_056
EXPECTED_INITIAL_STATE_SHA256: Final[str] = (
    "4dfd0462668286a47559f7a4688e33aa5485f667b1e5a0f10638bfff7ff18011"
)
EXPECTED_FROZEN_BANK_COUNT: Final[int] = 9
EXPECTED_FROZEN_PARAMETER_COUNT: Final[int] = 819_200
EXPECTED_TOTAL_ADAPTER_PARAMETER_COUNT: Final[int] = 864_256
EXPECTED_RETENTION_STEPS: Final[int] = 960
EXPECTED_CHANGED_PAIR_STEPS: Final[int] = 132
EXPECTED_INVARIANT_PAIR_STEPS: Final[int] = 192
EXPECTED_MICRO_STEPS: Final[int] = 1_284
EXPECTED_OPTIMIZER_UPDATES: Final[int] = 214
EXPECTED_TOTAL_NLL_FORWARDS: Final[int] = 1_872
PREREG_ARTIFACT: Final[str] = (
    "gemma4_v97_pair_decision_stability_preregistration_v1"
)
PREFLIGHT_ARTIFACT: Final[str] = (
    "gemma4_v97_pair_decision_stability_cpu_preflight_v1"
)
_DRAFT: Final[str] = "draft_contract_unsealed_training_not_authorized"
_SEALED: Final[str] = "sealed_before_v97_full_model_load"
_HEX64: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{64}")
_ANSWER_TYPE_QUOTAS: Final[dict[str, int]] = {
    "attribute": 32,
    "count": 32,
    "metric": 18,
    "orientation": 16,
    "presence": 32,
    "spatial_relation": 31,
    "support": 31,
}
_INVARIANT_FAMILY_QUOTAS: Final[dict[str, int]] = {
    "book_support": 24,
    "chair_orientation": 20,
    "color_swap": 20,
    "cube_support": 20,
    "mirror_lr": 24,
    "object_count": 20,
    "object_relocation": 20,
    "object_removal": 20,
    "picture_support": 24,
}
_ROW_CONTENT_KEYS: Final[frozenset[str]] = frozenset(
    {
        "answer",
        "answers",
        "answer_text",
        "label",
        "labels",
        "prediction",
        "predictions",
        "question",
        "questions",
        "question_text",
        "reference",
        "reference_answer",
        "rows",
    }
)


@dataclass(frozen=True)
class TrainingStepV97:
    kind: str
    round_index: int
    row: RowV96 | None = None
    unit: PairUnitV96 | None = None

    def identity(self) -> list[Any]:
        if self.kind == "retention" and self.row is not None and self.unit is None:
            return [self.kind, self.round_index, self.row.scene_id, self.row.question_id]
        if self.kind in {"changed_pair", "invariant_pair"} and self.unit is not None:
            return [self.kind, self.round_index, self.unit.pair_id, self.unit.question_key]
        raise ValueError("V97 malformed training step")


def _require(value: object, expected: object, label: str) -> None:
    if value != expected:
        raise ValueError(f"V97 {label} changed")


def _require_hash(value: object, label: str, *, draft: bool) -> None:
    if isinstance(value, str) and _HEX64.fullmatch(value):
        return
    if draft and value == "TO_FILL":
        return
    raise ValueError(f"V97 {label} is not sealed")


def _leaf_path(path: str | Path) -> Path:
    value = Path(path).expanduser()
    if not value.is_absolute():
        value = PROJECT_ROOT / value
    return Path(os.path.abspath(value))


def _strict_json(path: str | Path) -> dict[str, Any]:
    source = _leaf_path(path)
    if source.is_symlink() or not source.is_file():
        raise FileNotFoundError(source)
    value = json.loads(
        source.read_text(encoding="utf-8"),
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"V97 non-finite JSON number: {token}")
        ),
    )
    if not isinstance(value, dict):
        raise TypeError(f"V97 JSON must contain one object: {source}")
    _assert_finite_json(value)
    return value


def _assert_finite_json(value: object) -> None:
    if isinstance(value, Mapping):
        for child in value.values():
            _assert_finite_json(child)
    elif isinstance(value, list):
        for child in value:
            _assert_finite_json(child)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError("V97 JSON contains non-finite numeric data")


def _reject_row_content(value: object) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).casefold() in _ROW_CONTENT_KEYS:
                raise ValueError("V97 parent aggregate contains row-level content")
            _reject_row_content(child)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for child in value:
            _reject_row_content(child)


def load_config_v97(
    path: str | Path = CONFIG, *, allow_draft: bool = True
) -> dict[str, Any]:
    source = _leaf_path(path)
    if source.is_symlink() or not source.is_file():
        raise FileNotFoundError(source)
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or set(payload) != {"v97"}:
        raise ValueError("V97 config must contain exactly one v97 mapping")
    config = payload["v97"]
    if not isinstance(config, Mapping):
        raise TypeError("V97 config payload must be a mapping")
    _require(config.get("schema_version"), 97, "schema version")
    _require(
        config.get("artifact"),
        "gemma4_v97_pair_decision_stability_direct_memory_lora_v1",
        "artifact",
    )
    status = config.get("status")
    if status not in ({_DRAFT, _SEALED} if allow_draft else {_SEALED}):
        raise ValueError("V97 config status is not authorized")
    _require(config.get("seed"), 970097, "seed")
    draft = status == _DRAFT

    rationale = config.get("rationale")
    if not isinstance(rationale, Mapping):
        raise TypeError("V97 rationale is absent")
    for key, expected in {
        "permitted_v96_aggregate_only": True,
        "v96_primary_correct": 174,
        "v96_primary_total": 216,
        "v96_changed_side_correct": 16,
        "v96_changed_side_total": 24,
        "v96_complete_changed_units": 4,
        "v96_prediction_changed_units": 5,
        "v96_changed_unit_total": 12,
        "v96_invariant_false_changes": 24,
        "v96_invariant_side_total": 192,
        "row_level_v96_development_content_opened": False,
    }.items():
        _require(rationale.get(key), expected, f"rationale {key}")
    _require(
        rationale.get("observed_failed_gates"),
        [
            "prediction_changed_units_minimum_7",
            "invariant_false_change_maximum_20",
        ],
        "exact observed failed gates",
    )

    sources = config.get("sources")
    if not isinstance(sources, Mapping):
        raise TypeError("V97 sources are absent")
    for key in (
        "runtime_config_sha256",
        "training_qa_sha256",
        "train_memory_tensor_sha256",
        "train_memory_metadata_sha256",
        "development_memory_tensor_sha256",
        "development_memory_metadata_sha256",
        "frozen_v96_config_sha256",
        "frozen_v96_preflight_source_sha256",
        "frozen_v96_trainer_source_sha256",
        "frozen_v96_bridge_sha256",
        "frozen_v96_bridge_metadata_sha256",
        "frozen_v96_bridge_state_sha256",
        "v96_training_report_sha256",
        "v96_known_development_final_score_sha256",
        "v96_known_development_evidence_sha256",
        "preflight_source_sha256",
        "trainer_source_sha256",
        "model_blob_sha256_identity",
    ):
        _require_hash(sources.get(key), key, draft=draft)
    _require(sources.get("model_id"), "google/gemma-4-E2B-it", "model ID")
    _require(
        sources.get("model_revision"),
        "3e22461f65e89153144f8adb70e3b8c2cc9845a7",
        "model revision",
    )

    _require(
        config.get("strict_input_contract"),
        {
            "shape_per_scene": [1, 738, 1536],
            "native_boi_retained": True,
            "native_eoi_retained": True,
            "continuous_environment_payload_tokens": 736,
            "compiled_before_question": True,
            "reused_byte_identically_across_questions": True,
            "supplied_directly_to_native_gemma_image_prefix": True,
            "all_memory_slots_retained": True,
            "question_derived_environmental_tokens": 0,
            "question_conditioned_environmental_readout": False,
            "question_dependent_scene_processing": False,
            "question_dependent_retrieval": False,
            "semantic_or_spatial_top_k_selection": False,
            "control_tokens": 0,
            "environmental_text_inputs": [],
        },
        "strict direct-memory contract",
    )
    pool = config.get("training_pool")
    if not isinstance(pool, Mapping):
        raise TypeError("V97 training pool is absent")
    for key, expected in {
        "scene_count": 40,
        "pair_count": 20,
        "row_count": 960,
        "changed_unit_count": 66,
        "changed_side_count": 132,
        "invariant_unit_count": 414,
        "invariant_side_count": 828,
        "fixed_invariant_subset_unit_count": 192,
        "fixed_invariant_subset_side_count": 384,
        "answer_class_count": 29,
        "canonical_list_answer": "book, cube",
        "canonical_list_answer_row_count": 22,
    }.items():
        _require(pool.get(key), expected, f"training pool {key}")
    for key in (
        "row_inventory_sha256",
        "raw_answer_inventory_sha256",
        "scene_inventory_sha256",
        "pair_inventory_sha256",
        "answer_class_inventory_sha256",
        "balanced_class_weight_inventory_sha256",
        "changed_family_weight_inventory_sha256",
        "invariant_family_weight_inventory_sha256",
    ):
        _require_hash(pool.get(key), key, draft=draft)

    parent = config.get("parent_stack")
    if not isinstance(parent, Mapping):
        raise TypeError("V97 parent stack is absent")
    for key, expected in {
        "exact_parent": "v96_fixed_final_failed_known_development_aggregate_parent",
        "base_gemma_frozen": True,
        "older_frozen_bank_count": EXPECTED_FROZEN_BANK_COUNT,
        "older_frozen_adapter_parameter_count": EXPECTED_FROZEN_PARAMETER_COUNT,
        "v96_bank_name": "v96_atomic_pair_repair_bridge",
        "v96_bank_target_modules": list(TARGET_MODULES),
        "v96_bank_parameter_count": FRESH_PARAMETER_COUNT,
        "v96_bank_initial_state_sha256": EXPECTED_INITIAL_STATE_SHA256,
        "v96_fixed_final_selected_before_v96_labels_opened": True,
        "v96_known_development_used_as_row_free_diagnostic_only": True,
        "v96_row_level_content_available_to_v97": False,
        "merged_weights": False,
    }.items():
        _require(parent.get(key), expected, f"parent stack {key}")

    bridge = config.get("bridge")
    if not isinstance(bridge, Mapping):
        raise TypeError("V97 bridge is absent")
    for key, expected in {
        "bank_name": FRESH_BANK_NAME,
        "continuation_mode": "exact_v96_state_checkpoint_overwrite_then_train",
        "target_modules": list(TARGET_MODULES),
        "pinned_weight_shapes": PINNED_TENSORS,
        "pinned_weight_dtype": "BF16",
        "rank": 8,
        "alpha": 16.0,
        "dropout": 0.0,
        "trainable_parameter_count": FRESH_PARAMETER_COUNT,
        "initialization_algorithm": "checkpoint_overwrite",
        "initialization_seed": None,
        "expected_initial_state_sha256": EXPECTED_INITIAL_STATE_SHA256,
        "replaces_v96_bank_at_identical_target": True,
        "disjoint_from_nine_older_frozen_bank_targets": True,
        "total_bank_count_after_install": 10,
        "total_adapter_parameter_count_after_install": (
            EXPECTED_TOTAL_ADAPTER_PARAMETER_COUNT
        ),
    }.items():
        _require(bridge.get(key), expected, f"bridge {key}")

    training = config.get("training")
    if not isinstance(training, Mapping):
        raise TypeError("V97 training protocol is absent")
    for key, expected in {
        "optimizer": "AdamW",
        "retention_passes": 1,
        "retention_rows_per_pass": 960,
        "total_retention_steps": EXPECTED_RETENTION_STEPS,
        "changed_pair_rounds": 2,
        "changed_pair_units_per_round": 66,
        "total_changed_pair_steps": EXPECTED_CHANGED_PAIR_STEPS,
        "invariant_subset_rounds": 1,
        "invariant_subset_units_per_round": 192,
        "total_invariant_pair_steps": EXPECTED_INVARIANT_PAIR_STEPS,
        "total_micro_steps": EXPECTED_MICRO_STEPS,
        "microbatch_size": 1,
        "gradient_accumulation_steps": 6,
        "optimizer_updates": EXPECTED_OPTIMIZER_UPDATES,
        "learning_rate": 0.000025,
        "weight_decay": 0.0,
        "gradient_clip_norm": 0.75,
        "retention_balanced_ce_weight": 1.5,
        "pair_correct_ce_weight": 1.0,
        "within_memory_answer_margin_weight": 1.5,
        "within_memory_answer_target_margin_nll": 0.75,
        "across_memory_causal_margin_weight": 0.75,
        "across_memory_causal_target_margin_nll": 0.5,
        "first_divergent_token_margin_weight": 2.0,
        "first_divergent_token_target_margin_nll": 1.0,
        "pair_side_smoothmax_temperature": 0.25,
        "invariant_correct_ce_weight": 1.5,
        "invariant_nll_consistency_weight": 0.5,
        "invariant_nll_consistency_tolerance": 0.1,
        "invariant_answer_tail_js_weight": 2.0,
        "invariant_answer_tail_js_first_token_weight": 0.5,
        "invariant_answer_tail_js_mean_tail_weight": 0.5,
        "parent_anchor_rms_weight": 0.5,
        "parent_anchor_epsilon": 1e-12,
        "broad_nll_forward_evaluations": 960,
        "changed_pair_nll_forward_evaluations": 528,
        "invariant_pair_nll_forward_evaluations": 384,
        "total_nll_forward_evaluations": EXPECTED_TOTAL_NLL_FORWARDS,
        "checkpoint_every_optimizer_updates": 12,
        "deterministic_resume": True,
        "checkpoint_selection": (
            "fixed_final_update_214_before_any_v97_development_scoring"
        ),
        "intermediate_behavior_selection": False,
    }.items():
        _require(training.get(key), expected, f"training {key}")
    for key in ("schedule_sha256", "invariant_subset_sha256"):
        _require_hash(training.get(key), key, draft=draft)

    gate = config.get("known_development_gate")
    if not isinstance(gate, Mapping):
        raise TypeError("V97 known-development gate is absent")
    for key, expected in {
        "row_count": 216,
        "changed_side_total": 24,
        "changed_unit_total": 12,
        "invariant_side_total": 192,
        "v96_reference_correct": 174,
        "v96_reference_prediction_changed_units": 5,
        "v96_reference_invariant_false_changes": 24,
        "v97_correct_minimum": 160,
        "v96_primary_no_regression_reference": 174,
        "primary_no_regression_reported_separately": True,
        "changed_side_correct_minimum": 15,
        "complete_changed_units_minimum": 4,
        "prediction_changed_units_minimum": 7,
        "invariant_false_change_maximum": 20,
        "mean_changed_side_wrong_minus_correct_nll_minimum": 0.2,
        "thresholds_weakened_from_v96": False,
    }.items():
        _require(gate.get(key), expected, f"known-development gate {key}")
    if int(gate["v97_correct_minimum"]) < 160:
        raise ValueError("V97 primary-accuracy gate weakens V96's preregistered threshold")
    return dict(config)


def _capacity_match_v97(
    capacities: Mapping[tuple[str, str], int], *, seed: int
) -> dict[tuple[str, str], int]:
    """Deterministic integral max-flow for simultaneous type/family quotas."""

    source, sink = "@source", "@sink"
    types = sorted(
        _ANSWER_TYPE_QUOTAS,
        key=lambda name: (hashlib.sha256(f"{seed}|type|{name}".encode()).hexdigest(), name),
    )
    families = sorted(
        _INVARIANT_FAMILY_QUOTAS,
        key=lambda name: (
            hashlib.sha256(f"{seed}|family|{name}".encode()).hexdigest(),
            name,
        ),
    )
    graph: dict[str, list[list[Any]]] = defaultdict(list)

    def add_edge(left: str, right: str, capacity: int) -> list[Any]:
        forward: list[Any] = [right, len(graph[right]), capacity, capacity]
        reverse: list[Any] = [left, len(graph[left]), 0, 0]
        graph[left].append(forward)
        graph[right].append(reverse)
        return forward

    for answer_type in types:
        add_edge(source, "t:" + answer_type, _ANSWER_TYPE_QUOTAS[answer_type])
    cell_edges: dict[tuple[str, str], list[Any]] = {}
    for answer_type in types:
        ordered_families = sorted(
            families,
            key=lambda name: (
                hashlib.sha256(
                    f"{seed}|cell|{answer_type}|{name}".encode()
                ).hexdigest(),
                name,
            ),
        )
        for family in ordered_families:
            capacity = int(capacities.get((answer_type, family), 0))
            if capacity > 0:
                cell_edges[(answer_type, family)] = add_edge(
                    "t:" + answer_type, "f:" + family, capacity
                )
    for family in families:
        add_edge("f:" + family, sink, _INVARIANT_FAMILY_QUOTAS[family])

    flow = 0
    required = sum(_ANSWER_TYPE_QUOTAS.values())
    while True:
        level = {source: 0}
        queue = [source]
        for node in queue:
            for target, _reverse, capacity, _original in graph[node]:
                if capacity > 0 and target not in level:
                    level[target] = level[node] + 1
                    queue.append(target)
        if sink not in level:
            break
        cursor: Counter[str] = Counter()

        def send(
            node: str,
            amount: int,
            *,
            _cursor: Counter[str] = cursor,
            _level: Mapping[str, int] = level,
        ) -> int:
            if node == sink:
                return amount
            while _cursor[node] < len(graph[node]):
                edge = graph[node][_cursor[node]]
                target, reverse_index, capacity, _original = edge
                if capacity > 0 and _level.get(target) == _level[node] + 1:
                    pushed = send(target, min(amount, int(capacity)))
                    if pushed:
                        edge[2] -= pushed
                        graph[target][reverse_index][2] += pushed
                        return pushed
                _cursor[node] += 1
            return 0

        while True:
            pushed = send(source, required - flow)
            if not pushed:
                break
            flow += pushed
    if flow != required:
        raise ValueError("V97 invariant type/family quotas are jointly infeasible")
    return {
        key: int(edge[3]) - int(edge[2])
        for key, edge in cell_edges.items()
        if int(edge[3]) - int(edge[2]) > 0
    }


def invariant_subset_v97(
    rows: Sequence[RowV96], *, seed: int = 970097
) -> tuple[PairUnitV96, ...]:
    """Select 192 stable units under exact, answer-independent dual quotas."""

    _changed, invariant = pair_units_v96(rows)
    cells: dict[tuple[str, str], list[PairUnitV96]] = defaultdict(list)
    for unit in invariant:
        cells[(unit.answer_type, unit.change_type)].append(unit)
    allocations = _capacity_match_v97(
        {key: len(units) for key, units in cells.items()}, seed=seed
    )
    selected: list[PairUnitV96] = []
    for (answer_type, family), quota in sorted(allocations.items()):
        candidates = sorted(
            cells[(answer_type, family)],
            key=lambda unit: (
                hashlib.sha256(
                    (
                        f"{seed}|stable|{answer_type}|{family}|"
                        f"{unit.pair_id}|{unit.question_key}"
                    ).encode()
                ).hexdigest(),
                unit.key,
            ),
        )
        selected.extend(candidates[:quota])
    result = tuple(sorted(selected, key=lambda unit: unit.key))
    if (
        len(result) != EXPECTED_INVARIANT_PAIR_STEPS
        or len({unit.key for unit in result}) != EXPECTED_INVARIANT_PAIR_STEPS
        or Counter(unit.answer_type for unit in result) != Counter(_ANSWER_TYPE_QUOTAS)
        or Counter(unit.change_type for unit in result)
        != Counter(_INVARIANT_FAMILY_QUOTAS)
    ):
        raise RuntimeError("V97 invariant subset selection changed")
    return result


def training_schedule_v97(
    rows: Sequence[RowV96], *, seed: int = 970097
) -> tuple[TrainingStepV97, ...]:
    changed, _invariant = pair_units_v96(rows)
    stable = invariant_subset_v97(rows, seed=seed)
    steps: list[TrainingStepV97] = [
        TrainingStepV97("retention", 0, row=row) for row in rows
    ]
    for round_index in range(2):
        steps.extend(
            TrainingStepV97("changed_pair", round_index, unit=unit)
            for unit in changed
        )
    steps.extend(
        TrainingStepV97("invariant_pair", 0, unit=unit) for unit in stable
    )
    result = tuple(
        sorted(
            steps,
            key=lambda step: (
                hashlib.sha256(
                    (f"{seed}|schedule|" + "|".join(map(str, step.identity()))).encode()
                ).hexdigest(),
                step.identity(),
            ),
        )
    )
    counts = Counter(step.kind for step in result)
    if (
        len(result) != EXPECTED_MICRO_STEPS
        or counts
        != Counter(
            {
                "retention": EXPECTED_RETENTION_STEPS,
                "changed_pair": EXPECTED_CHANGED_PAIR_STEPS,
                "invariant_pair": EXPECTED_INVARIANT_PAIR_STEPS,
            }
        )
        or Counter(step.row.key for step in result if step.row is not None)
        != Counter({row.key: 1 for row in rows})
        or Counter(step.unit.key for step in result if step.kind == "changed_pair")
        != Counter({unit.key: 2 for unit in changed})
        or Counter(step.unit.key for step in result if step.kind == "invariant_pair")
        != Counter({unit.key: 1 for unit in stable})
    ):
        raise RuntimeError("V97 fixed training schedule changed")
    return result


class _SyntheticAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(1536, 4096, bias=False, dtype=torch.bfloat16)


class _SyntheticLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = _SyntheticAttention()


class _SyntheticGemma(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = nn.Module()
        layers: list[nn.Module] = [nn.Identity() for _ in range(35)]
        layers[9] = _SyntheticLayer()
        self.model.language_model.layers = nn.ModuleList(layers)


def lora_preflight_v97(config: Mapping[str, Any]) -> dict[str, Any]:
    bridge = config["bridge"]
    synthetic = _SyntheticGemma()
    synthetic.requires_grad_(False)
    installation = install_lora_adapters(
        synthetic,
        LoRASettings(
            enabled=True,
            rank=int(bridge["rank"]),
            alpha=float(bridge["alpha"]),
            dropout=float(bridge["dropout"]),
            target_modules=tuple(bridge["target_modules"]),
        ),
    )
    if installation is None:
        raise RuntimeError("V97 synthetic LoRA installation failed")
    weights = _leaf_path(config["sources"]["frozen_v96_fixed_final"]) / "bridge.safetensors"
    archive = load_file(str(weights), device="cpu")
    expected_inventory = set(installation.state_module.state_dict())
    if set(archive) != expected_inventory:
        raise ValueError("V97 V96 bridge tensor inventory changed")
    installation.state_module.load_state_dict(archive, strict=True)
    observed = installation.state_sha256()
    if (
        installation.parameter_count != FRESH_PARAMETER_COUNT
        or observed != EXPECTED_INITIAL_STATE_SHA256
        or observed != bridge["expected_initial_state_sha256"]
    ):
        raise RuntimeError("V97 exact V96 initialization changed")
    return {
        "bank_name": FRESH_BANK_NAME,
        "target_modules": list(installation.target_names),
        "parameter_count": installation.parameter_count,
        "initial_state_sha256": observed,
        "adapter_shapes": [
            {"lora_a": list(adapter.lora_a.shape), "lora_b": list(adapter.lora_b.shape)}
            for adapter in installation.adapters
        ],
        "exact_v96_checkpoint_overwrite": True,
        "zero_output_initialization": False,
        "full_gemma_model_loaded": False,
    }


def _deferred_physical_paths(config: Mapping[str, Any]) -> tuple[Path, ...]:
    return tuple(
        _leaf_path(root) / scene_id
        for root in config["deferred_final_lock"]["physical_artifact_roots"]
        for scene_id in DEFERRED_FINAL_SCENES
    )


def assert_deferred_final_absent_v97(config: Mapping[str, Any]) -> dict[str, Any]:
    physical = _deferred_physical_paths(config)
    present = [path.as_posix() for path in physical if path.exists() or path.is_symlink()]
    if present:
        raise RuntimeError(f"V97 refuses existing deferred-final artifacts: {present}")
    placeholders: dict[str, int] = {}
    for raw in config["deferred_final_lock"]["empty_qa_placeholders"]:
        path = _leaf_path(raw)
        if path.is_symlink() or not path.is_file() or path.stat().st_size != 0:
            raise RuntimeError(f"V97 deferred QA placeholder is not empty: {path}")
        placeholders[str(raw)] = 0
    return {
        "scene_ids": list(DEFERRED_FINAL_SCENES),
        "physical_path_count_checked": len(physical),
        "physical_artifacts_present": [],
        "empty_qa_placeholders": placeholders,
        "generation_performed": False,
    }


def assert_initial_outputs_absent_v97(config: Mapping[str, Any]) -> dict[str, Any]:
    outputs = config["outputs"]
    checked = {
        key: _leaf_path(outputs[key])
        for key in ("work_root", "fixed_final_candidate", "training_report")
    }
    present = {
        key: path.as_posix()
        for key, path in checked.items()
        if path.exists() or path.is_symlink()
    }
    if present:
        raise FileExistsError(f"V97 initial output already exists: {present}")
    return {
        "checked_paths": {key: path.as_posix() for key, path in checked.items()},
        "work_root_absent": True,
        "fixed_final_candidate_absent": True,
        "training_report_absent": True,
    }


def forbidden_training_roots_v97(config: Mapping[str, Any]) -> list[Path]:
    roots = list(_deferred_physical_paths(config))
    roots.extend(
        _leaf_path(root) / scene_id
        for root in config["deferred_final_lock"]["physical_artifact_roots"]
        for scene_id in PRIOR_EVALUATION_SCENES
    )
    excluded = config["excluded_known_development"]
    roots.extend(
        _leaf_path(excluded[key])
        for key in (
            "labels_path",
            "questions_path",
            "predictions_path",
            "structured_path",
            "nll_path",
        )
    )
    roots.extend(
        _leaf_path(raw)
        for raw in config["deferred_final_lock"]["empty_qa_placeholders"]
    )
    roots.extend(_leaf_path("data/oracle") / scene for scene in PRIOR_EVALUATION_SCENES)
    return list(dict.fromkeys(path.absolute() for path in roots))


def authenticate_parent_v96_v97(
    config: Mapping[str, Any], *, allow_pending_aggregate: bool
) -> dict[str, Any]:
    """Authenticate V96 without opening any row-level V96 evaluation file."""

    from semantic_3d_chat.evaluation.v96_atomic_pair_repair_preflight import (
        load_config_v96,
    )
    from semantic_3d_chat.training.train_v96_atomic_pair_repair import (
        authenticate_training_report_v96,
    )

    sources = config["sources"]
    pinned_files = (
        ("frozen_v96_config", "frozen_v96_config_sha256"),
        ("frozen_v96_preflight_source", "frozen_v96_preflight_source_sha256"),
        ("frozen_v96_trainer_source", "frozen_v96_trainer_source_sha256"),
        ("v96_training_report", "v96_training_report_sha256"),
    )
    for path_key, hash_key in pinned_files:
        if sha256_file_v85(sources[path_key]) != sources[hash_key]:
            raise ValueError(f"V97 frozen V96 source changed: {path_key}")
    v96_config = load_config_v96(sources["frozen_v96_config"], allow_draft=False)
    training = authenticate_training_report_v96(
        v96_config, config_path=sources["frozen_v96_config"]
    )
    if training["training_report_sha256"] != sources["v96_training_report_sha256"]:
        raise ValueError("V97 V96 training-report authentication changed")
    root = _leaf_path(sources["frozen_v96_fixed_final"])
    if (
        root.is_symlink()
        or not root.is_dir()
        or {child.name for child in root.iterdir()}
        != {"bridge.safetensors", "runtime_metadata.json"}
    ):
        raise ValueError("V97 V96 candidate inventory changed")
    weights = root / "bridge.safetensors"
    metadata_path = root / "runtime_metadata.json"
    if (
        sha256_file_v85(weights) != sources["frozen_v96_bridge_sha256"]
        or sha256_file_v85(metadata_path)
        != sources["frozen_v96_bridge_metadata_sha256"]
    ):
        raise ValueError("V97 V96 candidate bytes changed")
    metadata = _strict_json(metadata_path)
    if (
        metadata.get("artifact") != "gemma4_v96_atomic_pair_repair_fixed_final_v1"
        or metadata.get("schema_version") != 96
        or metadata.get("status") != "fixed_final_awaiting_known_development_gate"
        or metadata.get("bank_name") != "v96_atomic_pair_repair_bridge"
        or metadata.get("target_modules") != list(TARGET_MODULES)
        or metadata.get("parameter_count") != FRESH_PARAMETER_COUNT
        or metadata.get("state_sha256") != EXPECTED_INITIAL_STATE_SHA256
        or metadata.get("environmental_memory_serialized") is not False
        or metadata.get("questions_or_answers_serialized") is not False
        or metadata.get("oracle_serialized") is not False
        or metadata.get("runtime_promotion_authorized") is not False
    ):
        raise ValueError("V97 V96 candidate metadata changed")
    lora = lora_preflight_v97(config)
    if lora["initial_state_sha256"] != metadata["state_sha256"]:
        raise ValueError("V97 V96 candidate tensor state changed")

    final_hash = sources["v96_known_development_final_score_sha256"]
    evidence_hash = sources["v96_known_development_evidence_sha256"]
    if final_hash == "TO_FILL" or evidence_hash == "TO_FILL":
        if not allow_pending_aggregate:
            raise ValueError("V97 row-free V96 aggregate is not sealed")
        return {
            "status": "authenticated_v96_candidate_row_free_aggregate_pending",
            "v96_state_sha256": EXPECTED_INITIAL_STATE_SHA256,
            "v96_training_report_sha256": training["training_report_sha256"],
            "v96_row_level_content_loaded": False,
            "v96_aggregate_loaded": False,
            "training_authorized": False,
        }
    final_path = _leaf_path(sources["v96_known_development_final_score"])
    evidence_path = _leaf_path(sources["v96_known_development_evidence"])
    if (
        sha256_file_v85(final_path) != final_hash
        or sha256_file_v85(evidence_path) != evidence_hash
    ):
        raise ValueError("V97 V96 row-free aggregate bytes changed")
    final = _strict_json(final_path)
    evidence = _strict_json(evidence_path)
    _reject_row_content(final)
    _reject_row_content(evidence)
    metrics = final.get("structured_metrics")
    if not isinstance(metrics, Mapping):
        raise TypeError("V97 V96 structured aggregate is absent")
    arms = metrics.get("arms")
    changed = metrics.get("counterfactual")
    stable = metrics.get("stable_invariant")
    gates = final.get("gate_results")
    if not all(isinstance(value, Mapping) for value in (arms, changed, stable, gates)):
        raise TypeError("V97 V96 row-free aggregate sections changed")
    primary = arms.get("primary")
    failed = sorted(key for key, value in gates.items() if value is False)
    if (
        final.get("artifact") != "gemma4_v96_known_development_gate_v1"
        or final.get("schema_version") != 96
        or final.get("status") != "measured_preregistered_gate_not_passed"
        or final.get("known_development_gate_passed") is not False
        or final.get("deferred_final_unlock_eligible") is not False
        or final.get("row_level_content_serialized") is not False
        or final.get("runtime_promotion_authorized") is not False
        or final.get("candidate_state_sha256") != EXPECTED_INITIAL_STATE_SHA256
        or evidence.get("artifact") != "gemma4_v96_known_development_evidence_v1"
        or evidence.get("status") != "sealed_aggregate_evidence"
        or evidence.get("known_development_gate_passed") is not False
        or evidence.get("row_level_content_serialized") is not False
        or evidence.get("final_score_sha256") != final_hash
        or not isinstance(primary, Mapping)
        or primary.get("correct") != 174
        or primary.get("total") != 216
        or changed.get("canonical_correct_sides") != 16
        or changed.get("canonical_complete_units") != 4
        or changed.get("canonical_prediction_changed_units") != 5
        or stable.get("invariant_false_change_count") != 24
        or stable.get("side_count") != 192
        or failed
        != ["invariant_false_change_maximum", "prediction_changed_units_minimum"]
    ):
        raise ValueError("V97 rejected the sealed V96 aggregate diagnosis")
    return {
        "status": "authenticated_v96_fixed_final_failed_exactly_two_gates",
        "v96_state_sha256": EXPECTED_INITIAL_STATE_SHA256,
        "v96_training_report_sha256": training["training_report_sha256"],
        "v96_final_score_sha256": final_hash,
        "v96_evidence_sha256": evidence_hash,
        "v96_primary_correct": 174,
        "v96_prediction_changed_units": 5,
        "v96_invariant_false_changes": 24,
        "v96_row_level_content_loaded": False,
        "v96_aggregate_loaded": True,
        "training_authorized": True,
    }


def authenticate_training_sources_v97(config: Mapping[str, Any]) -> dict[str, str]:
    sources = config["sources"]
    bindings = (
        (sources["runtime_config"], sources["runtime_config_sha256"]),
        (sources["training_qa"], sources["training_qa_sha256"]),
        (
            str(Path(sources["train_memory_cache"]) / "training_tensors.safetensors"),
            sources["train_memory_tensor_sha256"],
        ),
        (
            str(Path(sources["train_memory_cache"]) / "metadata.json"),
            sources["train_memory_metadata_sha256"],
        ),
        (
            str(Path(sources["development_memory_cache"]) / "training_tensors.safetensors"),
            sources["development_memory_tensor_sha256"],
        ),
        (
            str(Path(sources["development_memory_cache"]) / "metadata.json"),
            sources["development_memory_metadata_sha256"],
        ),
        (sources["preflight_source"], sources["preflight_source_sha256"]),
        (sources["trainer_source"], sources["trainer_source_sha256"]),
    )
    observed: dict[str, str] = {}
    for raw, expected in bindings:
        _require_hash(expected, str(raw), draft=False)
        value = sha256_file_v85(raw)
        if value != expected:
            raise ValueError(f"V97 pinned training source changed: {raw}")
        observed[str(raw)] = value
    authenticate_parent_v96_v97(config, allow_pending_aggregate=False)
    return observed


def derive_contract_v97(config_path: str | Path = CONFIG) -> dict[str, Any]:
    config = load_config_v97(config_path, allow_draft=True)
    draft = config["status"] == _DRAFT
    audit = FileAccessAudit(
        forbidden_training_roots_v97(config),
        forbidden_component_names=frozenset(),
        block_forbidden=True,
    )
    with audit:
        deferred = assert_deferred_final_absent_v97(config)
        outputs = assert_initial_outputs_absent_v97(config)
        rows = load_training_rows_v96(config)
        changed, _invariant = pair_units_v96(rows)
        stable = invariant_subset_v97(rows)
        class_weights = balanced_class_weights_v96(config, rows)
        changed_weights = family_weights_v96(changed)
        invariant_weights = family_weights_v96(stable)
        schedule = training_schedule_v97(rows)
        memories, memory_hashes = load_scene_memories_v96(config, rows)
        lora = lora_preflight_v97(config)
        parent = authenticate_parent_v96_v97(
            config, allow_pending_aggregate=draft
        )
    audit.assert_clean()
    schedule_sha = canonical_sha256_v85([step.identity() for step in schedule])
    subset_sha = canonical_sha256_v85(
        [[unit.pair_id, unit.question_key] for unit in stable]
    )
    inventories = {
        "invariant_family_weight_inventory_sha256": canonical_sha256_v85(
            sorted(invariant_weights.items())
        ),
        "changed_family_weight_inventory_sha256": canonical_sha256_v85(
            sorted(changed_weights.items())
        ),
        "balanced_class_weight_inventory_sha256": canonical_sha256_v85(
            sorted(class_weights.items())
        ),
    }
    for key, observed in inventories.items():
        expected = config["training_pool"][key]
        if expected != "TO_FILL" and expected != observed:
            raise ValueError(f"V97 derived {key} changed")
    for key, observed in (
        ("schedule_sha256", schedule_sha),
        ("invariant_subset_sha256", subset_sha),
    ):
        expected = config["training"][key]
        if expected != "TO_FILL" and expected != observed:
            raise ValueError(f"V97 derived {key} changed")
    counts = Counter(step.kind for step in schedule)
    return {
        "schema_version": 97,
        "status": "derived_not_training_authorized",
        "config_status": config["status"],
        "schedule_sha256": schedule_sha,
        "invariant_subset_sha256": subset_sha,
        "dataset_hashes": inventories,
        "training_rows": len(rows),
        "training_scenes": len(memories),
        "changed_units": len(changed),
        "invariant_subset_units": len(stable),
        "schedule_step_counts": dict(sorted(counts.items())),
        "schedule_micro_steps": len(schedule),
        "optimizer_updates": len(schedule) // 6,
        "total_nll_forward_evaluations": EXPECTED_TOTAL_NLL_FORWARDS,
        "estimated_wall_time_seconds": 4244,
        "wall_time_budget_seconds": 5400,
        "all_training_memory_hashes": memory_hashes,
        "parent": parent,
        "lora_preflight": lora,
        "initial_output_absence": outputs,
        "deferred_final_absence": deferred,
        "all_changed_pair_questions_byte_identical": all(
            unit.left.question.encode() == unit.right.question.encode()
            for unit in changed
        ),
        "all_invariant_subset_questions_byte_identical": all(
            unit.left.question.encode() == unit.right.question.encode()
            for unit in stable
        ),
        "known_development_labels_opened": False,
        "known_development_questions_opened": False,
        "known_development_predictions_opened": False,
        "known_development_structured_rows_opened": False,
        "deferred_final_artifacts_generated": False,
        "file_audit_forbidden_reads": audit.forbidden_accesses(),
        "full_gemma_model_loaded": False,
        "optimizer_constructed": False,
        "optimizer_updates_performed": 0,
        "training_authorized": False,
    }


def derive_preregistration_v97(
    config_path: str | Path = CONFIG,
) -> dict[str, Any]:
    config = load_config_v97(config_path, allow_draft=True)
    derived = derive_contract_v97(config_path)
    return {
        "artifact": PREREG_ARTIFACT,
        "schema_version": 97,
        "status": "draft_not_sealed_training_implementation_pending",
        "config_path": Path(config_path).as_posix(),
        "config_sha256": sha256_file_v85(config_path),
        "derived_contract": derived,
        "strict_input_contract": config["strict_input_contract"],
        "rationale": config["rationale"],
        "training_pool": config["training_pool"],
        "excluded_known_development": config["excluded_known_development"],
        "deferred_final_lock": config["deferred_final_lock"],
        "parent_stack": config["parent_stack"],
        "bridge": config["bridge"],
        "training_protocol": config["training"],
        "known_development_protocol": config["known_development_gate"],
        "initial_output_absence": derived["initial_output_absence"],
        "parent_authenticated": derived["parent"]["training_authorized"],
        "known_development_row_level_content_opened": False,
        "deferred_final_artifacts_generated": False,
        "full_gemma_model_loaded": False,
        "optimizer_constructed": False,
        "optimizer_updates": 0,
        "training_authorized": False,
    }


def _atomic_create_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    destination = _leaf_path(path)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"V97 create-once output exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def build_preregistration_v97(config_path: str | Path = CONFIG) -> dict[str, Any]:
    config = load_config_v97(config_path, allow_draft=False)
    assert_initial_outputs_absent_v97(config)
    sources = authenticate_training_sources_v97(config)
    draft = derive_preregistration_v97(config_path)
    if draft["parent_authenticated"] is not True:
        raise RuntimeError("V97 cannot seal without the row-free V96 aggregate")
    payload = {
        **draft,
        "status": "sealed_before_v97_full_model_load_and_deferred_generation",
        "authenticated_sources": sources,
        "training_authorized": True,
    }
    output = _atomic_create_json(config["outputs"]["preregistration"], payload)
    return {**payload, "output": output.as_posix()}


def authenticate_preregistration_v97(
    config: Mapping[str, Any], *, config_path: str | Path = CONFIG
) -> dict[str, str]:
    path = _leaf_path(config["outputs"]["preregistration"])
    payload = _strict_json(path)
    config_hash = sha256_file_v85(config_path)
    if (
        payload.get("artifact") != PREREG_ARTIFACT
        or payload.get("schema_version") != 97
        or payload.get("status")
        != "sealed_before_v97_full_model_load_and_deferred_generation"
        or payload.get("config_sha256") != config_hash
        or payload.get("parent_authenticated") is not True
        or payload.get("known_development_row_level_content_opened") is not False
        or payload.get("deferred_final_artifacts_generated") is not False
        or payload.get("full_gemma_model_loaded") is not False
        or payload.get("optimizer_constructed") is not False
        or payload.get("optimizer_updates") != 0
        or payload.get("training_authorized") is not True
    ):
        raise ValueError("V97 preregistration changed")
    return {
        "config_sha256": config_hash,
        "preregistration_sha256": sha256_file_v85(path),
    }


def run_cpu_preflight_v97(config_path: str | Path = CONFIG) -> dict[str, Any]:
    config = load_config_v97(config_path, allow_draft=False)
    prereg = authenticate_preregistration_v97(config, config_path=config_path)
    report = {
        "artifact": PREFLIGHT_ARTIFACT,
        "schema_version": 97,
        "status": "passed",
        "passed": True,
        **prereg,
        "authenticated_sources": authenticate_training_sources_v97(config),
        "derived_contract": derive_contract_v97(config_path),
        "parent_authenticated": True,
        "known_development_row_level_content_opened": False,
        "deferred_final_artifacts_generated": False,
        "full_gemma_model_loaded": False,
        "optimizer_constructed": False,
        "optimizer_updates": 0,
        "behavior_scored": False,
        "runtime_promotion_authorized": False,
    }
    output = _atomic_create_json(config["outputs"]["cpu_preflight"], report)
    return {**report, "output": output.as_posix()}


def authenticate_cpu_preflight_v97(
    config: Mapping[str, Any], *, config_path: str | Path = CONFIG
) -> dict[str, str]:
    prereg = authenticate_preregistration_v97(config, config_path=config_path)
    path = _leaf_path(config["outputs"]["cpu_preflight"])
    payload = _strict_json(path)
    if (
        payload.get("artifact") != PREFLIGHT_ARTIFACT
        or payload.get("schema_version") != 97
        or payload.get("status") != "passed"
        or payload.get("passed") is not True
        or payload.get("config_sha256") != prereg["config_sha256"]
        or payload.get("preregistration_sha256") != prereg["preregistration_sha256"]
        or payload.get("parent_authenticated") is not True
        or payload.get("known_development_row_level_content_opened") is not False
        or payload.get("deferred_final_artifacts_generated") is not False
        or payload.get("full_gemma_model_loaded") is not False
        or payload.get("optimizer_constructed") is not False
        or payload.get("optimizer_updates") != 0
        or payload.get("behavior_scored") is not False
        or payload.get("runtime_promotion_authorized") is not False
    ):
        raise ValueError("V97 CPU preflight changed")
    return {
        **prereg,
        "cpu_preflight_sha256": sha256_file_v85(path),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("derive", "preregister", "preflight", "authenticate")
    )
    parser.add_argument("--config", default=str(CONFIG))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "derive":
        result = derive_contract_v97(args.config)
    elif args.command == "preregister":
        result = build_preregistration_v97(args.config)
    elif args.command == "preflight":
        result = run_cpu_preflight_v97(args.config)
    else:
        config = load_config_v97(args.config, allow_draft=False)
        result = authenticate_cpu_preflight_v97(config, config_path=args.config)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CONFIG",
    "EXPECTED_CHANGED_PAIR_STEPS",
    "EXPECTED_INITIAL_STATE_SHA256",
    "EXPECTED_INVARIANT_PAIR_STEPS",
    "EXPECTED_MICRO_STEPS",
    "EXPECTED_OPTIMIZER_UPDATES",
    "EXPECTED_RETENTION_STEPS",
    "EXPECTED_TOTAL_ADAPTER_PARAMETER_COUNT",
    "EXPECTED_TOTAL_NLL_FORWARDS",
    "FRESH_BANK_NAME",
    "FRESH_PARAMETER_COUNT",
    "TARGET_MODULES",
    "TrainingStepV97",
    "assert_deferred_final_absent_v97",
    "assert_initial_outputs_absent_v97",
    "authenticate_cpu_preflight_v97",
    "authenticate_parent_v96_v97",
    "authenticate_preregistration_v97",
    "authenticate_training_sources_v97",
    "build_preregistration_v97",
    "derive_contract_v97",
    "derive_preregistration_v97",
    "forbidden_training_roots_v97",
    "invariant_subset_v97",
    "load_config_v97",
    "lora_preflight_v97",
    "run_cpu_preflight_v97",
    "training_schedule_v97",
]
