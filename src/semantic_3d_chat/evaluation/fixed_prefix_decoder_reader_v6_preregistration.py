"""Unsealed CPU-only proposal for an upper-decoder fixed-prefix reader.

V4 and V5 adapted Gemma-4's per-layer-input projection. Both learned lower
answer NLL without stronger correct-vs-wrong-scene behavior. V6 therefore
proposes one fresh LoRA bank on the layer-32 and layer-33 MLP down projections.
Those residual writes follow their respective causal-attention operations. Its
targets are structurally disjoint from the declared layer-34 tool-decoder V2
target; actual joint runtime behavior is not claimed here. This module can build
and inspect a proposal and run a tiny random-model CPU gradient proof; it has no
full-checkpoint loader, optimizer loop, checkpoint writer, or sealing command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Final

import torch
import torch.nn.functional as F
import yaml
from safetensors import safe_open
from safetensors.torch import load as load_safetensors
from safetensors.torch import save as save_safetensors
from torch import nn

from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.language.fixed_prefix_decoder_reader_v6 import (
    EXPECTED_LAYER_TYPES,
    INITIAL_STATE_SHA256,
    INITIALIZATION_SEED,
    LORA_ALPHA,
    LORA_PARAMETER_COUNT,
    LORA_PARAMETER_COUNT_PER_MODULE,
    LORA_RANK,
    MODEL_ID,
    MODEL_REVISION,
    SLIDING_WINDOW_TOKENS,
    TARGET_IN_FEATURES,
    TARGET_MODULES,
    TARGET_OUT_FEATURES,
    decoder_reader_lora_settings_v6,
    validate_decoder_reader_surface_v6,
)
from semantic_3d_chat.language.gemma4_tool_decoder_v2 import (
    INITIAL_LORA_STATE_SHA256 as TOOL_INITIAL_LORA_STATE_SHA256,
)
from semantic_3d_chat.language.gemma4_tool_decoder_v2 import (
    PROJECTOR_INITIALIZATION_SEED as TOOL_INITIALIZATION_SEED,
)
from semantic_3d_chat.language.gemma4_tool_decoder_v2 import (
    TARGET_PROJECTION as TOOL_TARGET_PROJECTION,
)
from semantic_3d_chat.language.gemma4_tool_decoder_v2 import (
    tool_decoder_lora_settings_v2,
    validate_decoder_surface_v2,
)
from semantic_3d_chat.language.lora import (
    initialize_lora_adapter_state,
    install_lora_adapters,
)
from semantic_3d_chat.training import train_fixed_prefix_ple_v54 as v1

ARTIFACT: Final[str] = "gemma4_v54_fixed_prefix_decoder_reader_v6"
CONFIG: Final[str] = "configs/experiments/gemma4_v54_fixed_prefix_decoder_reader_v6.yaml"
CONFIG_SHA256: Final[str] = "cad5f0af664021b6e5c2bacb2ad1261d3222862e916b320effffe75ae6ab5cf0"
V5_PREREGISTRATION: Final[str] = (
    "reports/gemma4/metrics/gemma4_v54_fixed_prefix_ple_reader_v5_preregistration.json"
)
V5_SMOKE: Final[str] = "reports/gemma4/metrics/gemma4_v54_fixed_prefix_ple_reader_v5_smoke.json"
V5_RESULT: Final[str] = "reports/gemma4/metrics/gemma4_v54_fixed_prefix_ple_reader_v5_result.json"
V5_PREREGISTRATION_SHA256: Final[str] = (
    "7503de97af2d39282ccac3b91566f18bdebd718d81ace56f0cf065bff28db3e6"
)
V5_SMOKE_SHA256: Final[str] = "445c58339b4787d6c30c21b92a976da8bf7bcc1958f2aac4f3b9f8db67371523"
V5_RESULT_SHA256: Final[str] = "a39e3d9720ce595dfbf275cce51cf3e6bdd6c0ac312b6c1c916c82e69f716aa0"
BASE_RUNTIME_CONFIG: Final[str] = "configs/runtime/gemma4_v54.yaml"
BASE_RUNTIME_CONFIG_SHA256: Final[str] = (
    "891c58faaaa5fcd2ed76c7e3871f14c5d8c5ae2e05d9fa4ddd5193773d40e56b"
)
BASE_CHECKPOINT: Final[str] = "data_gemma4/checkpoints/gemma4_v54_semantic_greedy_gate/update_000"
PREFIX_CACHE: Final[str] = "data_gemma4/scene_tokens/v56_question_control_full_prefixes"
TRAIN_QA: Final[str] = "data_gemma4/training/v62_pair_disjoint/train.jsonl"
VALIDATION_QUESTIONS: Final[str] = "reports/gemma4/questions/v62_internal_validation.json"
VALIDATION_REFERENCES: Final[str] = (
    "reports/gemma4/scorer_only/v62_internal_validation_references.json"
)
BASELINE_PREDICTIONS: Final[str] = (
    "reports/gemma4/predictions/v62_v54_no_control_internal_validation.jsonl"
)
RETENTION_CORPUS: Final[str] = (
    "configs/experiments/gemma4_v54_fixed_prefix_ple_reader_v1_retention.json"
)
BASE_RUNTIME_METADATA_SHA256: Final[str] = (
    "807515461c71b08c08dfbd08a184a653e791413748530fa69402512eca6f6fdd"
)
MODEL_CONFIG_SHA256: Final[str] = "1b28f3d2c3100f6c594754b81107428bd7b822a7f48272ca681dae9d2ec38330"
MODEL_TOKENIZER_SHA256: Final[str] = (
    "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f"
)
MODEL_TOKENIZER_CONFIG_SHA256: Final[str] = (
    "9f4fec4b1dc6ecddf8f4a92e9caea5971c0e67d81309f3f9066a2bee8c362633"
)
MODEL_CHAT_TEMPLATE_SHA256: Final[str] = (
    "0a2c8073c878ab1da004bee933a998606537bbb62016310352c7285c3f01c5b5"
)
MODEL_WEIGHTS_BLOB_SHA256: Final[str] = (
    "2db5482b20d746879bb3ef79b5203e9075a2e2b98f54ec7c2f281c1477ddc550"
)
MODEL_WEIGHTS_SIZE_BYTES: Final[int] = 10_246_621_918
RESERVED_TOOL_TARGET: Final[str] = TOOL_TARGET_PROJECTION
_SEED: Final[int] = 720_054
_UPDATES: Final[int] = 96
_CONTRASTIVE_ROWS_PER_UPDATE: Final[int] = 3
_BROAD_ROWS_PER_UPDATE: Final[int] = 3
_PAIR_CE_WEIGHT: Final[float] = 0.5
_HINGE_WEIGHT: Final[float] = 4.0
_HINGE_MARGIN: Final[float] = 0.5
_BROAD_CE_WEIGHT: Final[float] = 0.5
_RETENTION_WEIGHT: Final[float] = 0.5
_WARMUP_UPDATES: Final[int] = 8
_PEAK_LR: Final[float] = 1e-4
_MIN_LR: Final[float] = 1e-5
_TRAIN_WRONG_ASSIGNMENT_SHA256: Final[str] = (
    "875cb3ed4893314494e90d563e1e961358a4fa34ccd6888545a20cfce903c5ff"
)
_VALIDATION_WRONG_ASSIGNMENT_SHA256: Final[str] = (
    "a2eaff713e8a51beec6779fc3d1720f179e2290ecaaf176d13ae1cc8d4362dcd"
)
_UPDATE_SCHEDULE_SHA256: Final[str] = (
    "c9680e1de5cb179a833284da7033a8bddda7bfe23d59c9defc02543804382270"
)
_GREEDY_SUBSET_SHA256: Final[str] = (
    "1d06ad2292635f438af38bfb31f05d0502972244b2c46d9691067fc8fb6756cd"
)

_EXPECTED_INPUT_HASHES: Final[dict[str, str]] = {
    BASE_RUNTIME_CONFIG: BASE_RUNTIME_CONFIG_SHA256,
    f"{BASE_CHECKPOINT}/adapter.safetensors": (
        "6c627f0a0d9efb7100489c24cbd4acafe10456b79280a64a85399a21cb541daf"
    ),
    f"{BASE_CHECKPOINT}/metadata.json": (
        "db1435f8d38ca587e34dcd55dc4d37532efc0504bfb62bc115838dc0ab7a7ece"
    ),
    f"{BASE_CHECKPOINT}/runtime_metadata.json": BASE_RUNTIME_METADATA_SHA256,
    f"{PREFIX_CACHE}/manifest.json": (
        "5a288a7fef65a957ba7b20132c63380cfadc7edbc37b32c1885037f939b9db61"
    ),
    TRAIN_QA: "84b99385fadc5d06e44465ada5902f56131192298ca1539373dc3b334608cbf1",
    VALIDATION_QUESTIONS: (
        "078f65e1402e6e382a7bfdb2ad4b8a65d58e3164705a8a46cd222503aa201052"
    ),
    VALIDATION_REFERENCES: (
        "4202e777ee57ab3f7da329f15589e56b8b0464b782fb4d856dd1a3281ff3115c"
    ),
    BASELINE_PREDICTIONS: (
        "df66de37e918ba068fbcd91308803746122c938ccccadf063d1b1343f1a4c902"
    ),
    RETENTION_CORPUS: (
        "0b2c48236e085960811ac6c9be94440814a141fdc05ed92c1e8f498a2c04f3cb"
    ),
}

_PINNED_CRITICAL_SOURCE_HASHES: Final[dict[str, str]] = {
    "src/semantic_3d_chat/config.py": (
        "f31ba226689b59ed8b6930eafc408e8e0c201090f66d4df86f9ce7a124b78b1e"
    ),
    "src/semantic_3d_chat/evaluation/fixed_prefix_ple_v54_preregistration.py": (
        "29ad92242ba48b8b1caa56bf5888b3d578641843de67b752232a82f75a77d2f2"
    ),
    "src/semantic_3d_chat/evaluation/metrics.py": (
        "08f27c1f7560f7bf7cfb272bc44bb777bcd40c96311e78c092ba5c98028245e3"
    ),
    "src/semantic_3d_chat/evaluation/v55_development_score.py": (
        "23ea365bbcfc06aa7aaa4d0881bdc3973ff3f0381e8464a765697f897909cfc6"
    ),
    "src/semantic_3d_chat/language/gemma4_backend.py": (
        "34edd02ae6c712b9c9cf2ec6586b1819a90631702baae0c2960ecc57bcff60cf"
    ),
    "src/semantic_3d_chat/language/gemma4_tool_decoder_v1.py": (
        "367ffbd07d32d299eebeab8ecbead60da46a218e50e1bea2ed29038711c15ff2"
    ),
    "src/semantic_3d_chat/language/gemma4_tool_decoder_v2.py": (
        "1e9e8d223d9494d8641142ec658fb26261222f3e69d340453fd2951e4367f875"
    ),
    "src/semantic_3d_chat/language/generation.py": (
        "1ecb0154316588f4b47e5a21df61af7f10bd0ae01c147ba2e3eb9a86125d3c44"
    ),
    "src/semantic_3d_chat/language/local_lm.py": (
        "0d8c4de10d8aa2c1c426d4aa48c8059fafd7eb85e3b2f4a02794ea9bea9bb372"
    ),
    "src/semantic_3d_chat/language/lora.py": (
        "f7c3ce3a4d46c0fcfef5e4cd360b4e12387799a9f6ccc0e2ffa74cba6ad56899"
    ),
    "src/semantic_3d_chat/training/train_adapter.py": (
        "726ea1a97e0ee0a874cd885fb94b9dbe356bdcf9a751b99046f6d6c11cd56027"
    ),
    "src/semantic_3d_chat/training/train_fixed_prefix_ple_v54.py": (
        "244519ba4ee1f7aa6e4904e20a9cba76fd0af8b028ad005c04fbd9ee39c1b99d"
    ),
    "src/semantic_3d_chat/training/train_fixed_prefix_ple_v54_v4.py": (
        "6e023ac794b73f063cc89c55955c7b45ecf774a0b0d93fa0750d70487961a9ed"
    ),
    "src/semantic_3d_chat/training/train_fixed_prefix_ple_v54_v5.py": (
        "a6556d13e3e6239d54bfd7c256237c4a8f38acea818d9b97a581f105b604c917"
    ),
}

_IMPLEMENTATION: Final[tuple[str, ...]] = (
    CONFIG,
    "src/semantic_3d_chat/language/fixed_prefix_decoder_reader_v6.py",
    "src/semantic_3d_chat/evaluation/fixed_prefix_decoder_reader_v6_preregistration.py",
    "tests/test_fixed_prefix_decoder_reader_v6.py",
)


@dataclass(frozen=True)
class V6Update:
    contrastive: tuple[v1.ReaderRecord, ...]
    broad: tuple[v1.ReaderRecord, ...]


def _resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return (value if value.is_absolute() else PROJECT_ROOT / value).resolve()


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with _resolve(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def implementation_source_hashes() -> dict[str, str]:
    return {relative: _sha256(relative) for relative in _IMPLEMENTATION}


def _authenticate_pinned_files(expected: Mapping[str, str], *, label: str) -> dict[str, str]:
    observed: dict[str, str] = {}
    for relative, digest in expected.items():
        source = _resolve(relative)
        if not source.is_file() or source.is_symlink():
            raise FileNotFoundError(f"V6 {label} is missing or unsafe: {relative}")
        actual = _sha256(source)
        if actual != digest:
            raise ValueError(f"V6 {label} changed: {relative}: {actual} != {digest}")
        observed[relative] = actual
    return observed


def _authenticate_prefix_cache() -> dict[str, Any]:
    """Authenticate the manifest and every listed train/validation prefix byte."""

    manifest_path = f"{PREFIX_CACHE}/manifest.json"
    manifest = json.loads(_resolve(manifest_path).read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping):
        raise TypeError("V6 fixed-prefix manifest must be a mapping")
    scenes = manifest.get("scenes")
    expected_scenes = set(v1.TRAIN_SCENES) | set(v1.VALIDATION_SCENES)
    if (
        manifest.get("artifact") != "question_independent_scene_prefix_cache_v1"
        or manifest.get("scene_count") != 40
        or manifest.get("complete_scene_prefixes") is not True
        or manifest.get("question_inputs_used") is not False
        or manifest.get("question_dependent_scene_retrieval") is not False
        or manifest.get("environmental_text_inputs") != []
        or not isinstance(scenes, Mapping)
        or set(scenes) != expected_scenes
    ):
        raise ValueError("V6 fixed-prefix manifest contract changed")
    file_hashes: dict[str, str] = {}
    for scene_id in sorted(expected_scenes):
        raw = scenes[scene_id]
        if not isinstance(raw, Mapping):
            raise TypeError(f"V6 prefix entry is invalid: {scene_id}")
        filename = raw.get("filename")
        if not isinstance(filename, str) or filename != f"{scene_id}.safetensors":
            raise ValueError(f"V6 prefix filename changed: {scene_id}")
        source = _resolve(PREFIX_CACHE) / filename
        expected_hash = raw.get("file_sha256")
        if (
            source.is_symlink()
            or not source.is_file()
            or source.stat().st_size != raw.get("file_size_bytes")
            or not isinstance(expected_hash, str)
            or _sha256(source) != expected_hash
            or raw.get("shape") != [1, 258, 1536]
            or raw.get("dtype") != "bfloat16"
        ):
            raise ValueError(f"V6 cached prefix bytes or structure changed: {scene_id}")
        file_hashes[scene_id] = expected_hash
    return {
        "manifest_sha256": _EXPECTED_INPUT_HASHES[manifest_path],
        "scene_count": len(file_hashes),
        "all_40_prefix_files_authenticated": True,
        "prefix_file_hash_inventory_sha256": _canonical_sha256(file_hashes),
    }


def authenticate_frozen_inputs_and_sources() -> dict[str, Any]:
    return {
        "listed_inputs": _authenticate_pinned_files(
            _EXPECTED_INPUT_HASHES, label="pinned input"
        ),
        "prefix_cache": _authenticate_prefix_cache(),
        "critical_sources": _authenticate_pinned_files(
            _PINNED_CRITICAL_SOURCE_HASHES, label="critical upstream source"
        ),
    }


def _record_key(row: v1.ReaderRecord) -> tuple[str, str]:
    return row.scene_id, row.question_id


def answer_varying_wrong_prefixes(
    rows: Sequence[v1.ReaderRecord],
) -> dict[tuple[str, str], str]:
    """Choose one deterministic answer-different scene for every eligible row.

    Curated counterfactual sides retain their exact paired scene when it is a
    valid answer-different candidate. Other rows use a seed-keyed content hash,
    never runtime retrieval or model similarity.
    """

    groups: defaultdict[str, list[v1.ReaderRecord]] = defaultdict(list)
    for row in rows:
        groups[row.question].append(row)
    assignments: dict[tuple[str, str], str] = {}
    for question, group in groups.items():
        if len({row.answer for row in group}) < 2:
            continue
        for row in group:
            candidates = [
                candidate
                for candidate in group
                if candidate.answer != row.answer and candidate.scene_id != row.scene_id
            ]
            if not candidates:
                raise ValueError("V6 answer-varying row has no different-scene negative")
            paired = [
                candidate
                for candidate in candidates
                if row.changed and candidate.scene_id == row.paired_scene_id
            ]
            pool = paired or candidates
            selected = min(
                pool,
                key=lambda candidate: hashlib.sha256(
                    (
                        f"{_SEED}|{question}|{row.scene_id}|{row.question_id}|"
                        f"{candidate.scene_id}|{candidate.question_id}"
                    ).encode()
                ).hexdigest(),
            )
            assignments[_record_key(row)] = selected.scene_id
    return assignments


def answer_varying_inventory(rows: Sequence[v1.ReaderRecord]) -> dict[str, Any]:
    assignments = answer_varying_wrong_prefixes(rows)
    index = {_record_key(row): row for row in rows}
    if len(index) != len(rows):
        raise ValueError("V6 records do not have unique scene/question identifiers")
    scene_answer = {(row.scene_id, row.question): row.answer for row in rows}
    if len(scene_answer) != len(rows):
        raise ValueError("V6 exact questions are not unique within each scene")
    if any(
        wrong_scene == key[0]
        or scene_answer[(wrong_scene, index[key].question)] == index[key].answer
        for key, wrong_scene in assignments.items()
    ):
        raise ValueError("V6 deterministic negative does not differ in scene and answer")
    curated = [row for row in rows if row.changed]
    if any(assignments.get(_record_key(row)) != row.paired_scene_id for row in curated):
        raise ValueError("V6 did not preserve a curated paired-scene negative")
    groups = defaultdict(list)
    for row in rows:
        if _record_key(row) in assignments:
            groups[row.question].append(row)
    candidate_scope_counts: Counter[str] = Counter()
    selected_scope_counts: Counter[str] = Counter()
    selected_answer_pairs: Counter[str] = Counter()
    selected_family_counts: Counter[str] = Counter()
    selected_family_by_scope: Counter[str] = Counter()
    answer_cell_frequencies: list[int] = []
    candidate_counts: list[int] = []
    selected_wrong_scene_reuse: Counter[tuple[str, str]] = Counter()
    for question, group in groups.items():
        answer_cell_frequencies.extend(Counter(row.answer for row in group).values())
        for row in group:
            candidates = [
                candidate
                for candidate in group
                if candidate.answer != row.answer and candidate.scene_id != row.scene_id
            ]
            candidate_counts.append(len(candidates))
            selected_scene = assignments[_record_key(row)]
            selected = next(
                candidate for candidate in group if candidate.scene_id == selected_scene
            )
            selected_wrong_scene_reuse[(question, selected_scene)] += 1
            selected_scope = (
                "same_counterfactual_pair"
                if row.pair_id is not None and selected.pair_id == row.pair_id
                else "cross_pair"
            )
            selected_scope_counts[selected_scope] += 1
            selected_answer_pairs[f"{row.answer} -> {selected.answer}"] += 1
            selected_family_counts[row.answer_type] += 1
            selected_family_by_scope[f"{row.answer_type} / {selected_scope}"] += 1
            same_pair = any(
                row.pair_id is not None and candidate.pair_id == row.pair_id
                for candidate in candidates
            )
            cross_pair = any(candidate.pair_id != row.pair_id for candidate in candidates)
            candidate_scope_counts[
                "same_and_cross_pair_candidates"
                if same_pair and cross_pair
                else "same_pair_candidates_only"
                if same_pair
                else "cross_pair_candidates_only"
            ] += 1
    serialized = [
        {"row": list(key), "wrong_scene_id": assignments[key]} for key in sorted(assignments)
    ]
    return {
        "total_rows": len(rows),
        "answer_varying_exact_question_groups": len(groups),
        "answer_varying_rows": len(assignments),
        "nonvarying_rows": len(rows) - len(assignments),
        "curated_changed_rows": len(curated),
        "curated_pair_preservation": True,
        "wrong_prefix_assignment_sha256": _canonical_sha256(serialized),
        "candidate_scope_counts": dict(sorted(candidate_scope_counts.items())),
        "selected_negative_scope_counts": dict(sorted(selected_scope_counts.items())),
        "selected_correct_to_wrong_answer_pair_distribution": dict(
            sorted(selected_answer_pairs.items())
        ),
        "selected_answer_type_family_distribution": dict(
            sorted(selected_family_counts.items())
        ),
        "selected_answer_type_by_scope_distribution": dict(
            sorted(selected_family_by_scope.items())
        ),
        "answer_frequency_per_group_cell_distribution": dict(
            sorted(Counter(answer_cell_frequencies).items())
        ),
        "answer_frequency_per_group_cell_minimum": min(answer_cell_frequencies),
        "answer_frequency_per_group_cell_maximum": max(answer_cell_frequencies),
        "eligible_candidate_count_per_row_distribution": dict(
            sorted(Counter(candidate_counts).items())
        ),
        "eligible_candidate_count_per_row_minimum": min(candidate_counts),
        "eligible_candidate_count_per_row_maximum": max(candidate_counts),
        "selected_wrong_scene_reuse_within_question_distribution": dict(
            sorted(Counter(selected_wrong_scene_reuse.values()).items())
        ),
        "selected_wrong_scene_reuse_within_question_minimum": min(
            selected_wrong_scene_reuse.values()
        ),
        "selected_wrong_scene_reuse_within_question_maximum": max(
            selected_wrong_scene_reuse.values()
        ),
        "frequency_imbalance_warning": (
            "Canonical-answer frequencies, eligible negative counts, and selected-scene "
            "reuse are imbalanced; all distributions are preserved rather than balanced "
            "or resampled in this one-arm proposal."
        ),
        "scientific_scope": (
            "cross-scene exact-question causality; curated rows separately retain their "
            "physical counterfactual-pair controls"
        ),
        "question_group_size_distribution": dict(
            sorted(Counter(len(group) for group in groups.values()).items())
        ),
    }


def build_v6_schedule(rows: Sequence[v1.ReaderRecord]) -> list[V6Update]:
    assignments = answer_varying_wrong_prefixes(rows)
    varying = [row for row in rows if _record_key(row) in assignments]
    broad = [row for row in rows if _record_key(row) not in assignments]
    if len(varying) != 288 or len(broad) != 288:
        raise ValueError("V6 training inventory is not the pinned 288/288 split")
    rng = random.Random(_SEED)
    rng.shuffle(varying)
    rng.shuffle(broad)
    updates = [
        V6Update(
            contrastive=tuple(
                varying[
                    index * _CONTRASTIVE_ROWS_PER_UPDATE : (index + 1)
                    * _CONTRASTIVE_ROWS_PER_UPDATE
                ]
            ),
            broad=tuple(
                broad[index * _BROAD_ROWS_PER_UPDATE : (index + 1) * _BROAD_ROWS_PER_UPDATE]
            ),
        )
        for index in range(_UPDATES)
    ]
    keys = [_record_key(row) for update in updates for row in (*update.contrastive, *update.broad)]
    if len(keys) != len(set(keys)) or set(keys) != {_record_key(row) for row in rows}:
        raise ValueError("V6 schedule does not consume every training row exactly once")
    serialized = [
        {
            "update": index,
            "contrastive": [list(_record_key(row)) for row in update.contrastive],
            "broad": [list(_record_key(row)) for row in update.broad],
        }
        for index, update in enumerate(updates, start=1)
    ]
    if _canonical_sha256(serialized) != _UPDATE_SCHEDULE_SHA256:
        raise ValueError("V6 deterministic update schedule changed")
    return updates


def learning_rate_v6(update: int) -> float:
    """Return the exact preregistered update-indexed learning rate."""

    if isinstance(update, bool) or not isinstance(update, int) or not 1 <= update <= _UPDATES:
        raise ValueError("V6 update must be an integer in [1, 96]")
    if update <= _WARMUP_UPDATES:
        return _PEAK_LR * update / _WARMUP_UPDATES
    progress = (update - _WARMUP_UPDATES) / (_UPDATES - _WARMUP_UPDATES)
    return _MIN_LR + 0.5 * (_PEAK_LR - _MIN_LR) * (
        1.0 + math.cos(math.pi * progress)
    )


def fixed_greedy_subset_contract(rows: Sequence[v1.ReaderRecord]) -> dict[str, Any]:
    """Pin V5's manifest-order 96-row greedy population by opaque identifiers."""

    by_scene: defaultdict[str, list[v1.ReaderRecord]] = defaultdict(list)
    for row in rows:
        by_scene[row.scene_id].append(row)
    selected = [
        row
        for scene_id in v1.VALIDATION_SCENES
        for row in by_scene[scene_id][:6]
    ]
    serialized = [
        {"scene_id": row.scene_id, "question_id": row.question_id} for row in selected
    ]
    digest = _canonical_sha256(serialized)
    if len(selected) != 96 or digest != _GREEDY_SUBSET_SHA256:
        raise ValueError("V6 fixed greedy subset changed")
    return {
        "selection": "first_6_rows_in_manifest_order_per_pinned_validation_scene",
        "scene_count": 16,
        "rows_per_scene": 6,
        "row_count": 96,
        "opaque_row_key_sha256": digest,
        "baseline_predictions_path": BASELINE_PREDICTIONS,
        "baseline_predictions_sha256": _EXPECTED_INPUT_HASHES[BASELINE_PREDICTIONS],
        "decoding": "deterministic_greedy_do_sample_false",
        "maximum_new_tokens": 32,
        "scoring": "canonical_type_specific_exact_match",
    }


def sequence_length_contract() -> dict[str, Any]:
    """Lengths measured once with the pinned tokenizer and byte-bound corpora."""

    return {
        "measurement_scope": "all_576_train_and_all_384_internal_validation_rows",
        "tokenizer_revision": MODEL_REVISION,
        "tokenizer_json_sha256": MODEL_TOKENIZER_SHA256,
        "tokenizer_config_sha256": MODEL_TOKENIZER_CONFIG_SHA256,
        "chat_template_sha256": MODEL_CHAT_TEMPLATE_SHA256,
        "prompt_builder_source_sha256": _PINNED_CRITICAL_SOURCE_HASHES[
            "src/semantic_3d_chat/language/local_lm.py"
        ],
        "answer_tokenizer_source_sha256": _PINNED_CRITICAL_SOURCE_HASHES[
            "src/semantic_3d_chat/training/train_adapter.py"
        ],
        "fixed_prefix_tokens": 258,
        "maximum_train_prompt_tokens": 63,
        "maximum_train_answer_tokens": 4,
        "maximum_train_teacher_sequence_tokens": 324,
        "maximum_validation_prompt_tokens": 64,
        "maximum_validation_answer_tokens": 4,
        "maximum_validation_teacher_sequence_tokens": 325,
        "maximum_retention_prompt_tokens": 13,
        "maximum_greedy_new_tokens": 32,
        "maximum_validation_greedy_sequence_tokens": 354,
        "layer_32_and_33_sliding_window_tokens": SLIDING_WINDOW_TOKENS,
        "all_teacher_and_preregistered_greedy_sequences_within_window": True,
    }


def measure_sequence_lengths_v6(tokenizer: Any) -> dict[str, int]:
    """Independently recompute the pinned maxima without loading model weights."""

    from semantic_3d_chat.language.local_lm import prompt_token_ids
    from semantic_3d_chat.training.train_adapter import tokenize_answer

    runtime = yaml.safe_load(_resolve(BASE_RUNTIME_CONFIG).read_text(encoding="utf-8"))
    system_prompt = runtime["language"]["system_prompt"]
    measurements: dict[str, int] = {}
    for label, rows in (
        ("train", v1.load_training_records()),
        ("validation", v1.load_validation_records()),
    ):
        triples: list[tuple[int, int, int]] = []
        for row in rows:
            prompt = prompt_token_ids(
                tokenizer, system_prompt, row.question, torch.device("cpu")
            )
            answer = tokenize_answer(tokenizer, row.answer, torch.device("cpu"))
            triples.append((prompt.shape[1], answer.shape[1], 258 + prompt.shape[1] + answer.shape[1]))
        measurements[f"maximum_{label}_prompt_tokens"] = max(item[0] for item in triples)
        measurements[f"maximum_{label}_answer_tokens"] = max(item[1] for item in triples)
        measurements[f"maximum_{label}_teacher_sequence_tokens"] = max(
            item[2] for item in triples
        )
    retention = v1.load_retention_corpus()
    measurements["maximum_retention_prompt_tokens"] = max(
        int(tokenizer(row["prompt"], return_tensors="pt").input_ids.shape[1])
        for row in retention
    )
    measurements["maximum_greedy_new_tokens"] = 32
    measurements["maximum_validation_greedy_sequence_tokens"] = (
        258 + measurements["maximum_validation_prompt_tokens"] + 32
    )
    expected = {
        key: value
        for key, value in sequence_length_contract().items()
        if key.startswith("maximum_") and isinstance(value, int)
    }
    if measurements != expected:
        raise ValueError(f"V6 measured sequence lengths changed: {measurements} != {expected}")
    return measurements


def decoder_contrastive_objective(
    correct_nll: torch.Tensor,
    wrong_nll: torch.Tensor,
    broad_nll: torch.Tensor,
    retention_kl: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Apply the V5 weights to V6's expanded answer-varying controls."""

    if (
        correct_nll.ndim != 1
        or wrong_nll.shape != correct_nll.shape
        or correct_nll.numel() < 1
        or broad_nll.ndim != 1
        or broad_nll.numel() < 1
        or retention_kl.ndim != 0
    ):
        raise ValueError("V6 objective tensor shapes are invalid")
    tensors = (correct_nll, wrong_nll, broad_nll, retention_kl.reshape(1))
    if any(not torch.isfinite(tensor).all() for tensor in tensors):
        raise ValueError("V6 objective received NaN or infinity")
    margins = wrong_nll - correct_nll
    hinge = F.relu(_HINGE_MARGIN - margins).mean()
    loss = (
        _PAIR_CE_WEIGHT * correct_nll.mean()
        + _HINGE_WEIGHT * hinge
        + _BROAD_CE_WEIGHT * broad_nll.mean()
        + _RETENTION_WEIGHT * retention_kl
    )
    return loss, {
        "correct_answer_ce": correct_nll.mean(),
        "wrong_prefix_hinge": hinge,
        "wrong_prefix_margins": margins,
        "broad_answer_ce": broad_nll.mean(),
        "retention_kl": retention_kl,
    }


class _ShapeOnlyDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = nn.Module()
        layers: list[nn.Module] = []
        for index in range(35):
            layer = nn.Module()
            if index in (32, 33, 34):
                layer.mlp = nn.Module()
                layer.mlp.down_proj = nn.Linear(TARGET_IN_FEATURES, TARGET_OUT_FEATURES, bias=False)
            layers.append(layer)
        self.model.language_model.layers = nn.ModuleList(layers)
        self.config = SimpleNamespace(
            text_config=SimpleNamespace(
                layer_types=EXPECTED_LAYER_TYPES,
                sliding_window=SLIDING_WINDOW_TOKENS,
            )
        )


def _initial_state_contract() -> dict[str, Any]:
    model = _ShapeOnlyDecoder().requires_grad_(False)
    validate_decoder_reader_surface_v6(model)
    installation = install_lora_adapters(model, decoder_reader_lora_settings_v6())
    if installation is None:
        raise RuntimeError("V6 decoder LoRA unexpectedly disabled")
    initialize_lora_adapter_state(installation, seed=INITIALIZATION_SEED)
    installation.assert_only_lora_trainable(model)
    return {
        "target_modules": list(installation.target_names),
        "parameter_counts": installation.parameter_counts,
        "parameter_count": installation.parameter_count,
        "initial_state_sha256": installation.state_sha256(),
        "only_authorized_parameters_trainable": True,
        "exact_zero_b": all(
            torch.count_nonzero(adapter.lora_b).item() == 0 for adapter in installation.adapters
        ),
    }


def _shape_only_joint_install_roundtrip() -> dict[str, Any]:
    """Prove only structural coexistence of V6 and the disjoint V2 tool bank."""

    model = _ShapeOnlyDecoder().cpu().requires_grad_(False)
    validate_decoder_reader_surface_v6(model)
    validate_decoder_surface_v2(model)
    reader = install_lora_adapters(model, decoder_reader_lora_settings_v6())
    if reader is None:
        raise RuntimeError("V6 shape-only reader adapter unexpectedly disabled")
    initialize_lora_adapter_state(reader, seed=INITIALIZATION_SEED)
    reader_hash = reader.state_sha256()
    for parameter in reader.parameters():
        parameter.requires_grad_(False)
    tool = install_lora_adapters(model, tool_decoder_lora_settings_v2())
    if tool is None:
        raise RuntimeError("V6 shape-only tool adapter unexpectedly disabled")
    initialize_lora_adapter_state(tool, seed=TOOL_INITIALIZATION_SEED)
    tool_hash = tool.state_sha256()
    if reader_hash != INITIAL_STATE_SHA256 or tool_hash != TOOL_INITIAL_LORA_STATE_SHA256:
        raise ValueError("V6 shape-only joint adapter initialization changed")
    for parameter in tool.parameters():
        parameter.requires_grad_(False)
    state = {
        **{
            f"reader.{key}": value.detach().cpu().clone()
            for key, value in reader.state_module.state_dict().items()
        },
        **{
            f"tool.{key}": value.detach().cpu().clone()
            for key, value in tool.state_module.state_dict().items()
        },
    }
    payload = save_safetensors(state)
    with torch.no_grad():
        for installation in (reader, tool):
            for parameter in installation.parameters():
                parameter.add_(1.0)
    restored = load_safetensors(payload)
    reader.state_module.load_state_dict(
        {key.removeprefix("reader."): value for key, value in restored.items() if key.startswith("reader.")},
        strict=True,
    )
    tool.state_module.load_state_dict(
        {key.removeprefix("tool."): value for key, value in restored.items() if key.startswith("tool.")},
        strict=True,
    )
    if reader.state_sha256() != reader_hash or tool.state_sha256() != tool_hash:
        raise RuntimeError("V6 shape-only joint checkpoint state did not round-trip")
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("V6 shape-only runtime coexistence proof must finish fully frozen")
    return {
        "claim_scope": "shape_only_structural_target_and_state_coexistence",
        "real_gemma_checkpoint_loaded": False,
        "runtime_semantic_or_tool_behavior_proven": False,
        "device": "cpu",
        "reader_targets": list(TARGET_MODULES),
        "tool_target": RESERVED_TOOL_TARGET,
        "target_sets_disjoint": True,
        "reader_state_sha256": reader_hash,
        "tool_state_sha256": tool_hash,
        "serialized_roundtrip_bytes": len(payload),
        "serialized_roundtrip_payload_sha256": hashlib.sha256(payload).hexdigest(),
        "strict_state_load": True,
        "all_joint_runtime_parameters_frozen_after_load": True,
    }


def _model_snapshot() -> Path:
    configured = os.environ.get("HF_HUB_CACHE")
    if configured:
        hub = Path(configured).expanduser()
    else:
        hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
        hub = hf_home.expanduser() / "hub"
    return hub / f"models--{MODEL_ID.replace('/', '--')}" / "snapshots" / MODEL_REVISION


def _snapshot_structure() -> dict[str, Any]:
    snapshot = _model_snapshot()
    config_path = snapshot / "config.json"
    weights = snapshot / "model.safetensors"
    tokenizer = snapshot / "tokenizer.json"
    tokenizer_config = snapshot / "tokenizer_config.json"
    chat_template = snapshot / "chat_template.jinja"
    if not all(
        path.is_file()
        for path in (config_path, weights, tokenizer, tokenizer_config, chat_template)
    ):
        raise FileNotFoundError("Pinned local Gemma-4 snapshot is incomplete")
    if _sha256(config_path) != MODEL_CONFIG_SHA256:
        raise ValueError("Pinned Gemma-4 config hash changed")
    for path, expected in (
        (tokenizer, MODEL_TOKENIZER_SHA256),
        (tokenizer_config, MODEL_TOKENIZER_CONFIG_SHA256),
        (chat_template, MODEL_CHAT_TEMPLATE_SHA256),
    ):
        if _sha256(path) != expected:
            raise ValueError(f"Pinned Gemma-4 tokenizer input changed: {path.name}")
    resolved_weights = weights.resolve()
    if (
        resolved_weights.name != MODEL_WEIGHTS_BLOB_SHA256
        or resolved_weights.stat().st_size != MODEL_WEIGHTS_SIZE_BYTES
    ):
        raise ValueError("Pinned Gemma-4 weight blob identity or size changed")
    config = json.loads(config_path.read_text(encoding="utf-8"))["text_config"]
    layer_types = config["layer_types"]
    if (
        config["hidden_size"] != 1536
        or config["intermediate_size"] != 6144
        or config["use_double_wide_mlp"] is not True
        or config["num_hidden_layers"] != 35
        or config["sliding_window"] != SLIDING_WINDOW_TOKENS
        or tuple(layer_types) != EXPECTED_LAYER_TYPES
        or config["max_position_embeddings"] != 131_072
    ):
        raise ValueError("Pinned Gemma-4 complete decoder architecture changed")
    shapes: dict[str, list[int]] = {}
    dtypes: dict[str, str] = {}
    with safe_open(weights, framework="pt", device="cpu") as archive:
        for target in (*TARGET_MODULES, RESERVED_TOOL_TARGET):
            name = f"{target}.weight"
            shapes[name] = list(archive.get_slice(name).get_shape())
            dtypes[name] = archive.get_slice(name).get_dtype()
    if any(shape != [1536, 12288] for shape in shapes.values()):
        raise ValueError("Gemma-4 decoder down-projection shape changed")
    return {
        "snapshot": str(snapshot.resolve()),
        "checkpoint_parameters_loaded": False,
        "safe_open_metadata_only": True,
        "model_weights_blob_sha256": MODEL_WEIGHTS_BLOB_SHA256,
        "model_weights_size_bytes": MODEL_WEIGHTS_SIZE_BYTES,
        "target_shapes": shapes,
        "target_dtypes": dtypes,
        "sliding_window": SLIDING_WINDOW_TOKENS,
        "layer_types": layer_types,
        "layer_type_count": len(layer_types),
        "max_position_embeddings": config["max_position_embeddings"],
        "tokenizer_json_sha256": MODEL_TOKENIZER_SHA256,
        "tokenizer_config_sha256": MODEL_TOKENIZER_CONFIG_SHA256,
        "chat_template_sha256": MODEL_CHAT_TEMPLATE_SHA256,
    }


def _authenticate_v5_failure() -> dict[str, Any]:
    for relative, expected in (
        (V5_PREREGISTRATION, V5_PREREGISTRATION_SHA256),
        (V5_SMOKE, V5_SMOKE_SHA256),
        (V5_RESULT, V5_RESULT_SHA256),
        (BASE_RUNTIME_CONFIG, BASE_RUNTIME_CONFIG_SHA256),
        (f"{BASE_CHECKPOINT}/runtime_metadata.json", BASE_RUNTIME_METADATA_SHA256),
    ):
        if _sha256(relative) != expected:
            raise ValueError(f"V6 pinned predecessor/input changed: {relative}")
    result = json.loads(_resolve(V5_RESULT).read_text(encoding="utf-8"))
    if (
        result.get("status") != "failed_no_checkpoint"
        or result.get("checkpoint_published") is not False
        or result.get("deferred_holdout", {}).get("accessed") is not False
        or result.get("final_scenes_000025_through_000030_accessed") is not False
        or _resolve("data_gemma4/checkpoints/gemma4_v54_fixed_prefix_ple_reader_v5").exists()
    ):
        raise ValueError("V5 no-checkpoint terminal contract changed")
    return {
        "preregistration_sha256": V5_PREREGISTRATION_SHA256,
        "smoke_sha256": V5_SMOKE_SHA256,
        "result_sha256": V5_RESULT_SHA256,
        "status": result["status"],
        "checkpoint_absent": True,
        "answer_nll_before": result["selection"]["baseline_teacher"]["answer_nll_mean"],
        "answer_nll_after": result["selection"]["candidate_teacher"]["answer_nll_mean"],
        "positive_margin_sides_before": result["selection"]["baseline_teacher"][
            "changed_positive_margin_sides"
        ],
        "positive_margin_sides_after": result["selection"]["candidate_teacher"][
            "changed_positive_margin_sides"
        ],
        "complete_units_before": result["selection"]["baseline_teacher"]["changed_complete_units"],
        "complete_units_after": result["selection"]["candidate_teacher"]["changed_complete_units"],
    }


def _base_decoder_contract() -> dict[str, Any]:
    raw = yaml.safe_load(_resolve(BASE_RUNTIME_CONFIG).read_text(encoding="utf-8"))
    banks = raw["language"]["lora_banks"]
    existing_targets = {target for bank in banks.values() for target in bank["target_modules"]}
    metadata = json.loads(
        _resolve(f"{BASE_CHECKPOINT}/runtime_metadata.json").read_text(encoding="utf-8")
    )
    if (
        set(TARGET_MODULES) & existing_targets
        or RESERVED_TOOL_TARGET in existing_targets
        or metadata.get("lora_parameter_count") != 509_952
        or metadata.get("lora_trainable_parameter_count") != 0
    ):
        raise ValueError("V6 target coexistence with the frozen V54 decoder changed")
    return {
        "existing_v54_target_count": len(existing_targets),
        "existing_v54_lora_parameter_count": metadata["lora_parameter_count"],
        "existing_v54_lora_trainable_parameter_count": 0,
        "v6_targets_disjoint_from_v54": True,
        "declared_tool_v2_target": RESERVED_TOOL_TARGET,
        "v6_targets_disjoint_from_declared_tool_v2_target": (
            RESERVED_TOOL_TARGET not in TARGET_MODULES
        ),
        "coexistence_claim_scope": (
            "structural_target_disjointness_only; real joint runtime behavior remains unproven"
        ),
        "shape_only_joint_install_and_state_roundtrip": _shape_only_joint_install_roundtrip(),
    }


def _validate_draft_config() -> dict[str, Any]:
    source = _resolve(CONFIG)
    if not source.is_file() or source.is_symlink():
        raise FileNotFoundError("V6 draft config is missing or unsafe")
    observed_sha = _sha256(source)
    if observed_sha != CONFIG_SHA256:
        raise ValueError(
            f"V6 complete draft YAML changed: {observed_sha} != {CONFIG_SHA256}"
        )
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise TypeError("V6 draft config must be a mapping")
    # The byte hash above authenticates every key, value, omission, and ordering
    # choice. These assertions additionally bind critical code constants to it.
    if (
        raw["experiment"]["artifact"] != ARTIFACT
        or tuple(raw["decoder_adapter"]["exact_targets"]) != TARGET_MODULES
        or raw["decoder_adapter"]["rank"] != LORA_RANK
        or raw["decoder_adapter"]["trainable_parameter_count"] != LORA_PARAMETER_COUNT
        or raw["schedule"]["wrong_prefix_assignment_sha256"]
        != _TRAIN_WRONG_ASSIGNMENT_SHA256
        or raw["schedule"]["update_schedule_sha256"] != _UPDATE_SCHEDULE_SHA256
        or raw["selection"]["expanded_wrong_prefix_assignment_sha256"]
        != _VALIDATION_WRONG_ASSIGNMENT_SHA256
        or raw["selection"]["greedy_subset_sha256"] != _GREEDY_SUBSET_SHA256
        or raw["execution"]
        != {
            "preregistration_sealed": False,
            "full_gemma_loaded": False,
            "mps_used": False,
            "optimizer_constructed": False,
            "training_authorized": False,
            "training_executed": False,
            "checkpoint_published": False,
        }
    ):
        raise ValueError("V6 draft YAML and code constants disagree")
    return raw


def build_preregistration_draft() -> dict[str, Any]:
    """Build but never seal the one-arm V6 proposal."""

    config = _validate_draft_config()
    authenticated = authenticate_frozen_inputs_and_sources()
    train = v1.load_training_records()
    validation = v1.load_validation_records()
    train_inventory = answer_varying_inventory(train)
    validation_inventory = answer_varying_inventory(validation)
    schedule = build_v6_schedule(train)
    greedy = fixed_greedy_subset_contract(validation)
    lengths = sequence_length_contract()
    initial = _initial_state_contract()
    if (
        train_inventory["answer_varying_rows"] != 288
        or validation_inventory["answer_varying_rows"] != 170
        or train_inventory["wrong_prefix_assignment_sha256"]
        != _TRAIN_WRONG_ASSIGNMENT_SHA256
        or validation_inventory["wrong_prefix_assignment_sha256"]
        != _VALIDATION_WRONG_ASSIGNMENT_SHA256
        or train_inventory["selected_negative_scope_counts"]
        != {"cross_pair": 208, "same_counterfactual_pair": 80}
        or validation_inventory["selected_negative_scope_counts"]
        != {"cross_pair": 118, "same_counterfactual_pair": 52}
        or train_inventory["selected_answer_type_family_distribution"]
        != {
            "attribute": 70,
            "count": 28,
            "metric": 24,
            "orientation": 22,
            "presence": 12,
            "spatial_relation": 62,
            "support": 70,
        }
        or validation_inventory["selected_answer_type_family_distribution"]
        != {
            "attribute": 44,
            "count": 12,
            "metric": 16,
            "orientation": 14,
            "presence": 8,
            "spatial_relation": 32,
            "support": 44,
        }
        or len(schedule) != 96
        or learning_rate_v6(1) != 0.0000125
        or learning_rate_v6(8) != _PEAK_LR
        or learning_rate_v6(96) != _MIN_LR
        or initial["parameter_count"] != 110_592
        or initial["initial_state_sha256"] != INITIAL_STATE_SHA256
    ):
        raise ValueError("V6 proposal inventory or initialization changed")
    return {
        "schema_version": 1,
        "artifact": ARTIFACT,
        "status": "unsealed_cpu_only_draft_training_not_authorized",
        "research_question": (
            "Can two fresh upper-decoder residual-write adapters make local Gemma-4 "
            "causally prefer answers supported by the correct complete continuous V54 "
            "scene prefix over an answer-different wrong-scene prefix?"
        ),
        "failure_driven_transition": {
            "predecessor": _authenticate_v5_failure(),
            "diagnosis": (
                "PLE adaptation reduced answer NLL but twice failed to improve scene "
                "discrimination, so V6 moves the trainable surface after causal attention."
            ),
            "exact_inheritance_claim": False,
            "unchanged_from_v5": {
                "numeric_objective_coefficients": True,
                "curated_gate_thresholds": True,
                "seed_peak_minimum_lr_warmup_weight_decay_and_clip": True,
                "answer_suffix_and_retention_metric_definitions": True,
            },
            "changed_from_v5": {
                "trainable_surface": "PLE projection to layer-32/33 MLP down projections",
                "updates": "80 to 96",
                "contrastive_rows": "80 curated sides twice to 288 varying rows once",
                "broad_rows": "496 rows once to 288 nonvarying rows once",
                "schedule": "one pair unit plus 6/7 broad to 3 varying plus 3 nonvarying",
                "gradient_checkpointing": "true to false as a resource-only change",
                "new_internal_gates": "expanded aggregate, answer-type macro, and scope macro",
            },
        },
        "model": {
            "model_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "local_files_only": True,
            "base_dtype": "bfloat16",
            "architecture": _snapshot_structure(),
            "sequence_lengths": lengths,
            "maximum_observed_train_or_internal_validation_sequence_tokens": lengths[
                "maximum_validation_teacher_sequence_tokens"
            ],
            "maximum_preregistered_greedy_sequence_tokens": lengths[
                "maximum_validation_greedy_sequence_tokens"
            ],
            "upper_sliding_window_tokens": SLIDING_WINDOW_TOKENS,
            "all_preregistered_sequences_inside_layers_32_33_window": True,
        },
        "continuous_scene_contract": {
            "source": "structurally_authenticated_v54_fixed_prefix_cache",
            "shape": [1, 258, 1536],
            "complete_scene_latents": 256,
            "computed_before_question": True,
            "same_prefix_for_unchanged_scene": True,
            "question_dependent_retrieval": False,
            "all_scene_tokens_present": True,
            "environmental_text_inputs": [],
            "oracle_runtime_access": False,
        },
        "trainable_surface": {
            "type": "one_fresh_unmerged_fp32_lora_bank",
            "exact_target_modules": list(TARGET_MODULES),
            "module_shape_out_in": [1536, 12288],
            "rank": LORA_RANK,
            "alpha": LORA_ALPHA,
            "dropout": 0.0,
            "parameter_count_per_module": LORA_PARAMETER_COUNT_PER_MODULE,
            "parameter_count": LORA_PARAMETER_COUNT,
            "initialization_seed": INITIALIZATION_SEED,
            "initial_state_sha256": INITIAL_STATE_SHA256,
            "initial_state": initial,
            "base_model_frozen": True,
            "all_v54_decoder_banks_frozen": True,
            "v54_scene_stack_and_prefixes_frozen": True,
            "layer_34_claim_scope": (
                "structural target disjointness only until a real joint-runtime smoke"
            ),
            "coexistence": _base_decoder_contract(),
            "why_layers_32_and_33_down_projection": (
                "Each MLP receives that layer's post-attention token state. Because every "
                "preregistered sequence is shorter than the 512-token sliding window, the "
                "answer-position states can causally contain the complete earlier scene "
                "prefix. The down projections then write into the residual stream before "
                "layer 34's final full-attention block."
            ),
        },
        "data": {
            "training": train_inventory,
            "internal_validation": validation_inventory,
            "training_validation_scene_disjoint": True,
            "curated_changed_train_rows": 80,
            "curated_changed_validation_rows": 52,
            "deferred_scenes_57_through_62_accessed": False,
            "final_scenes_25_through_30_accessed": False,
        },
        "deterministic_negative_control": {
            "group_key": "exact_user_question_text",
            "eligibility": "at_least_two_distinct_canonical_answers",
            "wrong_prefix_constraint": "different_scene_and_different_answer",
            "curated_changed_rows_keep_exact_paired_scene": True,
            "other_rows": "minimum_seed_keyed_sha256_candidate",
            "model_similarity_or_retrieval_used": False,
            "training_rows": 288,
            "internal_validation_rows": 170,
            "actual_selected_training_scope": {
                "same_counterfactual_pair": 80,
                "cross_pair": 208,
            },
            "actual_selected_internal_validation_scope": {
                "same_counterfactual_pair": 52,
                "cross_pair": 118,
            },
            "scientific_scope": (
                "This expanded control is cross-scene exact-question causality, not a "
                "physical counterfactual-pair claim; the curated 52-side gate remains "
                "separate and unchanged."
            ),
            "answer_and_candidate_frequency_imbalance_is_preserved_and_reported": True,
        },
        "objective": {
            "numeric_coefficients_reused_from_v5": True,
            "objective_is_not_exactly_inherited_from_v5": True,
            "answer_token_normalized_ce_on_contrastive_correct_prefix_weight": 0.5,
            "answer_token_normalized_ce_on_nonvarying_rows_weight": 0.5,
            "answer_different_wrong_prefix_hinge_weight": 4.0,
            "wrong_prefix_margin_nats_per_answer_token": 0.5,
            "retention_next_token_kl_weight": 0.5,
            "answer_suffix_only": True,
            "answer_logit_positions_only": True,
            "labels_before_answer_suffix": -100,
            "per_update_exact_gradient_sum": {
                "contrastive": (
                    "for each of 3 rows, backward 0.5/3*correct_nll + "
                    "4.0/3*relu(0.5-(wrong_nll-correct_nll))"
                ),
                "nonvarying": "for each of 3 rows, backward 0.5/3*correct_nll",
                "retention": "backward 0.5*KL(frozen_teacher || current)",
                "divide_total_by_component_count": False,
                "optimizer_steps_per_update": 1,
            },
            "environmental_caption_loss": False,
        },
        "optimization": {
            "seed": _SEED,
            "optimizer": "adamw",
            "adamw_beta1": 0.9,
            "adamw_beta2": 0.999,
            "adamw_epsilon": 1e-8,
            "adamw_amsgrad": False,
            "adamw_maximize": False,
            "adamw_foreach": False,
            "adamw_capturable": False,
            "adamw_differentiable": False,
            "adamw_fused": False,
            "updates": 96,
            "contrastive_rows_per_update": 3,
            "nonvarying_rows_per_update": 3,
            "every_training_row_exactly_once": True,
            "peak_learning_rate": 0.0001,
            "linear_warmup_updates": 8,
            "cosine_minimum_learning_rate": 0.00001,
            "learning_rate_formula": (
                "u<=8: 1e-4*u/8; u>8: 1e-5 + 0.5*(1e-4-1e-5)*"
                "(1+cos(pi*(u-8)/88))"
            ),
            "learning_rate_endpoints": {
                "update_1": learning_rate_v6(1),
                "update_8": learning_rate_v6(8),
                "update_96": learning_rate_v6(96),
            },
            "weight_decay": 0.0,
            "gradient_clip_l2": 1.0,
            "gradient_clip_scope": "global_l2_over_all_110592_v6_adapter_parameters",
            "gradient_nonfinite_or_nonpositive_is_fatal": True,
            "zero_grad_set_to_none": True,
            "optimizer_steps_per_update": 1,
            "gradient_term_order": (
                "3 contrastive rows in schedule order, 3 nonvarying rows in schedule "
                "order, then the cyclic retention row"
            ),
            "decoder_gradient_checkpointing": False,
            "gradient_checkpointing_change_from_v5_is_resource_only": True,
            "reason_checkpointing_disabled": (
                "The frozen lower decoder can remain outside autograd until the layer-32 "
                "adapter; only layers 32-34 need activation graphs. Forward values and the "
                "locked objective are unchanged."
            ),
            "final_state_after_update_96_is_only_candidate": True,
            "intermediate_or_best_loss_selection": False,
            "one_arm_only": True,
            "schedule_sha256": _UPDATE_SCHEDULE_SHA256,
            "retention_teacher": {
                "population": "all_16_pinned_non_environmental_prompts",
                "capture": "before_optimizer_construction_from_frozen_baseline",
                "stored_dtype_and_device": "float32_cpu",
                "distribution": "softmax_of_full_next_token_logit_vector",
                "loss_direction": "KL(frozen_teacher || current)",
                "update_index": "(update-1) modulo 16",
                "cycles": 6,
                "each_example_exposures": 6,
            },
        },
        "promotion_gates": {
            "evaluation_states": (
                "frozen baseline and the single final state after update 96; no intermediate"
            ),
            "teacher_forcing": {
                "evaluation_microbatch_size": 1,
                "answer_logit_positions_only": True,
                "correct_answer_population": "all_384_internal_validation_rows",
                "correct_answer_metric": "mean answer-token-normalized NLL",
                "minimum_baseline_minus_candidate_nll": 0.03,
                "wrong_prefix_margin": (
                    "answer-token-normalized wrong-prefix NLL minus correct-prefix NLL"
                ),
                "positive_comparison": "strictly_greater_than_zero",
            },
            "v5_curated_52_side_gates_unchanged": {
                "population": "all_52_sides_in_26_physical_counterfactual_units",
                "wrong_prefix": "the exact paired physical-counterfactual scene",
                "validation_answer_nll_improvement_minimum": 0.03,
                "positive_margin_rate_minimum": 0.65,
                "positive_margin_rate_delta_minimum": 0.10,
                "complete_pair_unit_delta_minimum": 3,
                "complete_unit_definition": "both side margins strictly greater than zero",
            },
            "expanded_170_side_gates": {
                "population": "all_170_fixed_answer-varying validation rows",
                "wrong_prefix_assignment_sha256": _VALIDATION_WRONG_ASSIGNMENT_SHA256,
                "positive_margin_rate_minimum": 0.65,
                "positive_margin_rate_delta_minimum": 0.10,
                "answer_type_strata": {
                    "exact_populations": validation_inventory[
                        "selected_answer_type_family_distribution"
                    ],
                    "macro_definition": "unweighted mean of all 7 answer-type rates",
                    "macro_positive_margin_rate_minimum": 0.65,
                    "macro_positive_margin_rate_delta_minimum": 0.10,
                    "minimum_each_family_positive_margin_rate": 0.50,
                    "minimum_each_family_positive_margin_rate_delta": 0.0,
                    "all_7_families_required": True,
                },
                "selected_negative_scope_strata": {
                    "exact_populations": validation_inventory[
                        "selected_negative_scope_counts"
                    ],
                    "macro_definition": "unweighted mean of same-pair and cross-pair rates",
                    "macro_positive_margin_rate_minimum": 0.65,
                    "macro_positive_margin_rate_delta_minimum": 0.10,
                    "minimum_each_scope_positive_margin_rate": 0.55,
                    "minimum_each_scope_positive_margin_rate_delta": 0.0,
                    "both_scopes_required": True,
                },
            },
            "greedy": {
                **greedy,
                "minimum_candidate_minus_pinned_baseline_exact_accuracy": 0.02,
                "run_only_after_all_teacher_and_retention_gates_pass": True,
            },
            "retention": {
                "population": "all_16_pinned_non_environmental_prompts",
                "teacher": "pre-optimizer frozen full next-token logits",
                "continuation_metric": "first continuation-token cross entropy",
                "kl_direction": "KL(frozen_teacher || candidate)",
                "mean_ce_increase_nats_maximum": 0.03,
                "mean_kl_nats_maximum": 0.02,
                "next_token_top1_agreement_minimum": 0.98,
            },
            "all_required": True,
            "failed_run_publishes_no_checkpoint": True,
        },
        "required_preseal_proofs": {
            "independent_surface_and_objective_audit": True,
            "tiny_true_gemma_cpu_zero_noop_and_gradient": True,
            "real_full_model_mps_answer_tail_gradient_smoke": True,
            "real_full_model_joint_v6_and_tool_v2_runtime_smoke": True,
            "real_mps_driver_allocation_maximum_bytes": 25_000_000_000,
            "no_optimizer_before_all_smokes_pass": True,
        },
        "deferred_and_final_policy": {
            "scenes_57_through_62": "forbidden_until_all_internal_gates_pass",
            "scenes_25_through_30": "final_once_forbidden_during_v6_selection",
        },
        "execution": {
            "preregistration_sealed": False,
            "full_gemma_loaded": False,
            "mps_used": False,
            "optimizer_constructed": False,
            "training_authorized": False,
            "training_executed": False,
            "checkpoint_published": False,
        },
        "configuration": {
            "path": CONFIG,
            "sha256": CONFIG_SHA256,
            "complete_yaml_byte_hash_authenticated": True,
            "parsed": config,
        },
        "authenticated_frozen_inputs_and_sources": authenticated,
        "implementation_source_hashes": implementation_source_hashes(),
    }


def structural_preflight() -> dict[str, Any]:
    proposal = build_preregistration_draft()
    return {
        "schema_version": 1,
        "artifact": f"{ARTIFACT}_structural_preflight",
        "status": "passed_cpu_no_model_draft_training_not_authorized",
        "passed": True,
        "proposal_sha256": _canonical_sha256(proposal),
        "targets": list(TARGET_MODULES),
        "trainable_parameter_count": LORA_PARAMETER_COUNT,
        "training_answer_varying_rows": proposal["data"]["training"]["answer_varying_rows"],
        "validation_answer_varying_rows": proposal["data"]["internal_validation"][
            "answer_varying_rows"
        ],
        "schedule_updates": proposal["optimization"]["updates"],
        "full_checkpoint_loaded": False,
        "mps_used": False,
        "optimizer_constructed": False,
        "training_authorized": False,
        "deferred_or_final_access": False,
    }


def _tiny_selected_nll(
    model: nn.Module,
    backend: Any,
    scene_prefix: torch.Tensor,
    prompt_ids: torch.Tensor,
    answer_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    prepared = backend.prepare(
        scene_prefix,
        prompt_ids,
        answer_ids,
        scene_prefix_after_bos=True,
        scene_boundary_mode="gemma4_native_image",
    )
    assert prepared.labels is not None
    label_positions = torch.nonzero(prepared.labels[0].ne(-100), as_tuple=False).flatten()
    causal_positions = (label_positions - 1).long()
    output = model(
        inputs_embeds=prepared.inputs_embeds,
        per_layer_inputs=prepared.per_layer_inputs,
        attention_mask=prepared.attention_mask,
        mm_token_type_ids=prepared.mm_token_type_ids,
        use_cache=False,
        labels=None,
        logits_to_keep=causal_positions,
        return_dict=True,
    )
    targets = prepared.labels[0, label_positions]
    nll = F.cross_entropy(output.logits[0].float(), targets, reduction="none").mean()
    return nll, output.logits.detach().clone()


def tiny_cpu_gradient_architecture_smoke() -> dict[str, Any]:
    """Prove exact no-op, answer-tail gradients, and layer-34 disjointness."""

    try:
        from transformers import Gemma4ForConditionalGeneration
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Tiny V6 smoke requires the isolated Gemma environment") from exc
    from semantic_3d_chat.evaluation.gemma4_tool_decoder_preregistration_v2 import (
        _tiny_gemma4_config_v2,
    )
    from semantic_3d_chat.language.gemma4_backend import Gemma4PrefixBackend

    started = time.perf_counter()
    torch.manual_seed(_SEED)
    model = Gemma4ForConditionalGeneration(_tiny_gemma4_config_v2()).cpu().eval()
    model.requires_grad_(False)
    backend = Gemma4PrefixBackend(model, model_revision="tiny-cpu-v6-draft")
    boi, eoi = backend.native_boundary_embeddings()
    scene_a = torch.cat((boi, torch.randn(1, 8, 32) * 0.08, eoi), dim=1)
    scene_b = torch.cat((boi, torch.randn(1, 8, 32) * 0.08, eoi), dim=1)
    prompt = torch.tensor([[2, 9, 13, 21]], dtype=torch.long)
    answer_a = torch.tensor([[17, 1]], dtype=torch.long)
    answer_b = torch.tensor([[19, 1]], dtype=torch.long)
    broad_answer = torch.tensor([[23, 1]], dtype=torch.long)

    with torch.inference_mode():
        frozen_nll, frozen_logits = _tiny_selected_nll(model, backend, scene_a, prompt, answer_a)
    installation = install_lora_adapters(model, decoder_reader_lora_settings_v6())
    if installation is None:
        raise RuntimeError("Tiny V6 adapter unexpectedly disabled")
    initialize_lora_adapter_state(installation, seed=_SEED)
    installation.assert_only_lora_trainable(model)
    with torch.inference_mode():
        zero_nll, zero_logits = _tiny_selected_nll(model, backend, scene_a, prompt, answer_a)
    if not torch.equal(frozen_logits, zero_logits) or not torch.equal(frozen_nll, zero_nll):
        raise RuntimeError("Tiny zero-output V6 adapter is not an exact no-op")

    correct_a, _ = _tiny_selected_nll(model, backend, scene_a, prompt, answer_a)
    wrong_a, _ = _tiny_selected_nll(model, backend, scene_b, prompt, answer_a)
    correct_b, _ = _tiny_selected_nll(model, backend, scene_b, prompt, answer_b)
    wrong_b, _ = _tiny_selected_nll(model, backend, scene_a, prompt, answer_b)
    broad, _ = _tiny_selected_nll(model, backend, scene_a, prompt, broad_answer)
    retention = torch.zeros((), dtype=torch.float32)
    loss, diagnostics = decoder_contrastive_objective(
        torch.stack((correct_a, correct_b)),
        torch.stack((wrong_a, wrong_b)),
        broad.reshape(1),
        retention,
    )
    loss.backward()
    gradients = installation.gradient_norms()
    b_gradients = [
        float(adapter.lora_b.grad.detach().float().norm()) for adapter in installation.adapters
    ]
    a_gradients = [
        float(adapter.lora_a.grad.detach().float().norm()) for adapter in installation.adapters
    ]
    if any(not math.isfinite(value) or value <= 0.0 for value in b_gradients):
        raise RuntimeError("Tiny V6 gradient did not reach both decoder adapters")
    if any(value != 0.0 for value in a_gradients):
        raise RuntimeError("Tiny exact-zero B initialization must give zero A gradient")
    layer34 = model.model.language_model.layers[34].mlp.down_proj
    if not isinstance(layer34, nn.Linear) or any(
        parameter.requires_grad for parameter in layer34.parameters()
    ):
        raise RuntimeError("Tiny V6 unexpectedly modified the reserved layer-34 target")
    with torch.no_grad():
        for adapter in installation.adapters:
            adapter.lora_b.fill_(0.01)
        _, influenced = _tiny_selected_nll(model, backend, scene_a, prompt, answer_a)
    maximum_change = float((influenced - zero_logits).abs().max())
    if maximum_change <= 0.0:
        raise RuntimeError("Tiny nonzero V6 adapter did not influence answer logits")
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return {
        "schema_version": 1,
        "artifact": f"{ARTIFACT}_tiny_cpu_gradient_architecture_smoke",
        "status": "passed_cpu_random_model_training_not_authorized",
        "passed": True,
        "device": "cpu",
        "tiny_model_random_initialization": True,
        "full_checkpoint_loaded": False,
        "mps_used": False,
        "optimizer_constructed": False,
        "training_executed": False,
        "tiny_target_modules": list(TARGET_MODULES),
        "tiny_target_shapes_out_in": [[32, 128], [32, 128]],
        "tiny_trainable_parameter_count": trainable,
        "base_or_reserved_layer34_trainable_parameter_count": 0,
        "zero_output_exact_noop": True,
        "answer_logit_positions_only": True,
        "loss": float(loss.detach()),
        "wrong_prefix_margins": diagnostics["wrong_prefix_margins"].detach().tolist(),
        "gradient_l2": gradients["total_l2"],
        "lora_b_gradient_l2_by_target": dict(zip(TARGET_MODULES, b_gradients, strict=True)),
        "lora_a_gradient_l2_expected_zero_by_target": dict(
            zip(TARGET_MODULES, a_gradients, strict=True)
        ),
        "nonzero_adapter_maximum_answer_logit_change": maximum_change,
        "elapsed_seconds": time.perf_counter() - started,
        "deferred_or_final_access": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("draft", "preflight", "tiny-smoke"))
    mode = parser.parse_args(argv).mode
    result = {
        "draft": build_preregistration_draft,
        "preflight": structural_preflight,
        "tiny-smoke": tiny_cpu_gradient_architecture_smoke,
    }[mode]()
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
