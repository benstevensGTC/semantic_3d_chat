"""Preregister and CPU-preflight the bounded V80 V75-atlas reader.

This module never instantiates Gemma, never selects an arm, and never writes a
checkpoint.  It authenticates the single YAML contract and its historical
V73/V75 inputs, verifies the physical Gemma topology from config/tensor
metadata, and exercises the exact adapter implementation on a shape-faithful
CPU surrogate.  A separate model-bearing gradient smoke remains mandatory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Final

import torch
import torch.nn.functional as F
import yaml
from safetensors import safe_open
from torch import nn

from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.language.v80_atlas_attention_reader import (
    ALPHA,
    EXPECTED_LAYER_TYPES,
    PARAMETER_COUNT,
    RANK,
    TARGET_MODULES,
    TARGET_SHAPES_OUT_IN,
    causal_prefix_visibility,
    install_v80,
)
from semantic_3d_chat.training.train_question_control_v73 import (
    EXPECTED_HELD_ROWS,
    EXPECTED_HELD_SCENES,
    EXPECTED_TRAIN_ROWS,
    EXPECTED_TRAIN_SCENES,
    ChangedUnitV73,
    RowV73,
    changed_units_v73,
    load_config_v73,
    load_training_rows_v73,
    split_rows_v73,
)

CONFIG: Final[str] = "configs/experiments/gemma4_v80_v75_atlas_attention_reader.yaml"
EXPECTED_CONFIG_SHA256: Final[str] = (
    "8e086d3d46b9723338cc249dcfcaeb663f19e4b4cb220a2255ec36644dbd7e6f"
)
EXPECTED_SCHEDULE_SHA256: Final[str] = (
    "be4375143a23cda3bd9ad765c253b6532ed920f0ce51f6434215c23eb383056a"
)
EXPECTED_HELD_SMOKE_SHA256: Final[str] = (
    "cd898bbf859fe1a5d5f798229813a2eec3fcc9f2d2de2eea1ed1ae81fc476056"
)
EXPECTED_BROAD_TRAIN_SHA256: Final[str] = (
    "d4adb3a3f781e45a5465172439146f1ab07a1de3eb28337d08cea453866b8f93"
)
EXPECTED_BROAD_HELD_SHA256: Final[str] = (
    "70580e40feba49c1896bddea3c3676433be6d492178d08793554916e49a8890a"
)
EXPECTED_CHANGED_FAMILIES: Final[dict[str, int]] = {
    "book_support": 8,
    "chair_orientation": 1,
    "color_swap": 4,
    "cube_support": 3,
    "mirror_lr": 8,
    "object_count": 1,
    "object_relocation": 4,
    "object_removal": 3,
    "picture_support": 8,
}
PRIOR_EVIDENCE: Final[dict[str, tuple[str, str]]] = {
    "v6": (
        "reports/gemma4/metrics/gemma4_v54_fixed_prefix_decoder_reader_v6_mps_smoke.json",
        "a78e38e9e5112f757927a9590cecb854c9c99f7881d929b531b83b9db305f2fa",
    ),
    "v6_1": (
        "reports/gemma4/metrics/gemma4_v54_fixed_prefix_decoder_reader_v6_1_mps_smoke.json",
        "099c1fa684439814b58c17223781b745e406d17cc20c65c402159bd0ede18add",
    ),
    "v6_2": (
        "data_gemma4/checkpoints/gemma4_v54_fixed_prefix_decoder_reader_v6_2/terminal_result.json",
        "e86b417d5edeaedc5f541171845c37d3e740b5b24468fb0b2b062a2b8ae85f12",
    ),
    "v6_3": (
        "reports/gemma4/metrics/gemma4_v54_fixed_prefix_attention_reader_v6_3_pilot.json",
        "43fbce25b0b1566ef73bc0ba8e0440f218f388497605359e26f97af306b3dc67",
    ),
    "v6_4": (
        "reports/gemma4/metrics/gemma4_v54_fixed_prefix_attention_reader_v6_4_pair_disjoint_screen.json",
        "a909c71e10c2cca5757556dd462132a499b09f05576bb11119bf1b7f424f0414",
    ),
    "v75_atlas": (
        "reports/gemma4/metrics/v75_fixed_atlas_historical_internal_score.json",
        "224886019172c5080f2bd976de74477d40e37db9a5635aae9c9b7697db53dfd2",
    ),
}


def _resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return (value if value.is_absolute() else PROJECT_ROOT / value).resolve()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with _resolve(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_create_json(path: str | Path, value: Mapping[str, Any]) -> tuple[Path, str]:
    destination = _resolve(path)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"V80 create-once output exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination, hashlib.sha256(payload).hexdigest()


def load_v80_config(path: str | Path = CONFIG) -> dict[str, Any]:
    source = _resolve(path)
    if source != _resolve(CONFIG):
        raise ValueError("V80 refuses a noncanonical experiment config path")
    if source.is_symlink() or not source.is_file():
        raise FileNotFoundError(f"V80 config is unavailable: {source}")
    if sha256_file(source) != EXPECTED_CONFIG_SHA256:
        raise ValueError("V80 preregistered config bytes changed")
    value = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping) or set(value) != {"v80"}:
        raise ValueError("V80 config must contain exactly one v80 mapping")
    config = value["v80"]
    if not isinstance(config, Mapping):
        raise TypeError("V80 config payload must be a mapping")
    required = {
        "schema_version",
        "artifact",
        "status",
        "seed",
        "wall_time_budget_seconds",
        "inputs",
        "atlas_contract",
        "split",
        "reader",
        "forward_contract",
        "optimization",
        "memory_safety",
        "mandatory_zero_step_gate",
        "held_smoke_gates",
        "publication",
        "scope",
        "outputs",
    }
    if set(config) != required:
        raise ValueError("V80 top-level config fields changed")
    exact = {
        "schema_version": 1,
        "artifact": "gemma4_v80_v75_atlas_attention_reader",
        "status": "preregistered_preflight_only_model_bearing_run_not_yet_authorized",
        "seed": 800080,
        "wall_time_budget_seconds": 1800,
    }
    if any(config.get(key) != expected for key, expected in exact.items()):
        raise ValueError("V80 exact experiment identity changed")
    reader = config["reader"]
    if (
        reader.get("rank") != RANK
        or float(reader.get("alpha", 0.0)) != ALPHA
        or tuple(reader.get("target_modules", ())) != TARGET_MODULES
        or reader.get("trainable_parameter_count") != PARAMETER_COUNT
        or {
            key: tuple(value)
            for key, value in reader.get("target_shapes_out_in", {}).items()
        }
        != TARGET_SHAPES_OUT_IN
    ):
        raise ValueError("V80 reader arm changed")
    optimization = config["optimization"]
    if (
        optimization.get("updates") != 16
        or optimization.get("changed_units_per_update") != [3] * 8 + [2] * 8
        or optimization.get("schedule_sha256") != EXPECTED_SCHEDULE_SHA256
        or optimization.get("intermediate_selection_or_checkpoint") is not False
    ):
        raise ValueError("V80 bounded optimization changed")
    memory = config["memory_safety"]
    expected_memory = {
        "host_unified_memory_bytes": 25_769_803_776,
        "minimum_host_available_bytes": 4_294_967_296,
        "maximum_mps_driver_allocated_bytes": 19_000_000_000,
        "maximum_process_rss_bytes": 9_663_676_416,
        "sequential_microbranches_only": True,
        "maximum_live_teacher_forced_batch_size": 1,
        "duplicate_model_instances": False,
        "persistent_atlas_device_cache": False,
        "persistent_teacher_logit_device_cache": False,
        "gradient_checkpointing": False,
        "use_cache": False,
        "checkpoint_writer_present": False,
        "memory_checked_before_and_after_every_microbranch": True,
        "hard_wall_timer": True,
    }
    if memory != expected_memory:
        raise ValueError("V80 host-memory safety contract changed")
    publication = config["publication"]
    if any(
        publication.get(field) is not False
        for field in (
            "held_smoke_sufficient_for_runtime_promotion",
            "checkpoint_publication_authorized",
            "runtime_publication_authorized",
            "official_validation_authorized",
            "official_test_authorized",
            "deferred_final_authorized",
            "oracle_authorized",
        )
    ):
        raise ValueError("V80 publication or protected-evaluation boundary changed")
    scope = config["scope"]
    if scope != {
        "historical_v73_training_pool_only": True,
        "official_validation_loaded": False,
        "official_test_loaded": False,
        "deferred_final_loaded": False,
        "oracle_loaded": False,
        "cloud_inference": False,
    }:
        raise ValueError("V80 scope changed")
    return dict(config)


def build_schedule_v80(units: Sequence[ChangedUnitV73]) -> tuple[tuple[ChangedUnitV73, ...], ...]:
    if len(units) != 40 or Counter(unit.change_type for unit in units) != Counter(
        EXPECTED_CHANGED_FAMILIES
    ):
        raise ValueError("V80 requires the exact 40 changed V73 train units")
    ordered = list(units)
    random.Random(800080).shuffle(ordered)
    sizes = [3] * 8 + [2] * 8
    result: list[tuple[ChangedUnitV73, ...]] = []
    offset = 0
    for size in sizes:
        result.append(tuple(ordered[offset : offset + size]))
        offset += size
    inventory = [
        [[unit.pair_id, unit.question_key] for unit in update] for update in result
    ]
    if offset != 40 or canonical_sha256(inventory) != EXPECTED_SCHEDULE_SHA256:
        raise RuntimeError("V80 exact-once schedule changed")
    return tuple(result)


def select_held_smoke_v80(held_rows: Sequence[RowV73]) -> tuple[ChangedUnitV73, ...]:
    first: dict[str, ChangedUnitV73] = {}
    for unit in changed_units_v73(held_rows):
        first.setdefault(unit.change_type, unit)
    result = tuple(first[family] for family in sorted(first))
    inventory = [
        [unit.change_type, unit.pair_id, unit.question_key] for unit in result
    ]
    if (
        len(result) != 8
        or len({row.scene_id for unit in result for row in (unit.left, unit.right)}) != 16
        or canonical_sha256(inventory) != EXPECTED_HELD_SMOKE_SHA256
    ):
        raise RuntimeError("V80 held smoke inventory changed")
    return result


def select_broad_train_v80(train_rows: Sequence[RowV73]) -> tuple[RowV73, ...]:
    by_type: defaultdict[str, list[RowV73]] = defaultdict(list)
    for row in train_rows:
        if not row.expected_change:
            by_type[row.answer_type].append(row)
    for values in by_type.values():
        values.sort(
            key=lambda row: hashlib.sha256(
                f"v80-broad-train|{row.scene_id}|{row.question_id}".encode()
            ).hexdigest()
        )
    types = sorted(by_type)
    positions: defaultdict[str, int] = defaultdict(int)
    result: list[RowV73] = []
    for index in range(16):
        answer_type = types[index % len(types)]
        result.append(by_type[answer_type][positions[answer_type]])
        positions[answer_type] += 1
    inventory = [[row.answer_type, row.scene_id, row.question_id] for row in result]
    if canonical_sha256(inventory) != EXPECTED_BROAD_TRAIN_SHA256:
        raise RuntimeError("V80 broad train inventory changed")
    return tuple(result)


def select_broad_held_v80(held_rows: Sequence[RowV73]) -> tuple[RowV73, ...]:
    result: list[RowV73] = []
    for scene_id in sorted({row.scene_id for row in held_rows}):
        candidates = [
            row for row in held_rows if row.scene_id == scene_id and not row.expected_change
        ]
        candidates.sort(
            key=lambda row: hashlib.sha256(
                f"v80-broad-held|{row.scene_id}|{row.question_id}".encode()
            ).hexdigest()
        )
        result.append(candidates[0])
    inventory = [[row.answer_type, row.scene_id, row.question_id] for row in result]
    if canonical_sha256(inventory) != EXPECTED_BROAD_HELD_SHA256:
        raise RuntimeError("V80 broad held inventory changed")
    return tuple(result)


def _historical_inventory(config: Mapping[str, Any]) -> dict[str, Any]:
    inputs = config["inputs"]
    v73 = load_config_v73(inputs["source_v73_config"])
    if _resolve(v73["training_qa"]) != _resolve(inputs["historical_training_qa"]):
        raise ValueError("V80 and V73 historical QA paths differ")
    rows = load_training_rows_v73(inputs["historical_training_qa"])
    train, held = split_rows_v73(rows)
    schedule = build_schedule_v80(changed_units_v73(train))
    held_smoke = select_held_smoke_v80(held)
    broad_train = select_broad_train_v80(train)
    broad_held = select_broad_held_v80(held)
    train_scenes = {row.scene_id for row in train}
    held_scenes = {row.scene_id for row in held}
    if train_scenes & held_scenes:
        raise RuntimeError("V80 train and held scenes overlap")
    return {
        "all_rows": len(rows),
        "train_rows": len(train),
        "train_scenes": len(train_scenes),
        "train_pairs": len({row.pair_id for row in train}),
        "train_changed_units": sum(len(update) for update in schedule),
        "held_rows": len(held),
        "held_scenes": len(held_scenes),
        "held_pairs": len({row.pair_id for row in held}),
        "held_smoke_units": len(held_smoke),
        "held_smoke_sides": 2 * len(held_smoke),
        "broad_train_rows": len(broad_train),
        "broad_held_rows": len(broad_held),
        "pair_disjoint": {row.pair_id for row in train}.isdisjoint(
            {row.pair_id for row in held}
        ),
        "scene_disjoint": train_scenes.isdisjoint(held_scenes),
        "schedule_sha256": EXPECTED_SCHEDULE_SHA256,
        "held_smoke_sha256": EXPECTED_HELD_SMOKE_SHA256,
        "broad_train_sha256": EXPECTED_BROAD_TRAIN_SHA256,
        "broad_held_sha256": EXPECTED_BROAD_HELD_SHA256,
    }


def _validate_inputs(config: Mapping[str, Any]) -> dict[str, Any]:
    inputs = config["inputs"]
    expected = {
        inputs["source_v73_config"]: inputs["source_v73_config_sha256"],
        inputs["historical_training_qa"]: inputs["historical_training_qa_sha256"],
        inputs["runtime_config"]: inputs["runtime_config_sha256"],
        Path(inputs["base_checkpoint"]) / "adapter.safetensors": inputs[
            "base_adapter_sha256"
        ],
        Path(inputs["atlas_controller"]) / "control.safetensors": inputs[
            "atlas_controller_weights_sha256"
        ],
        Path(inputs["numeric_probe_bank"]) / "probes.safetensors": inputs[
            "numeric_probe_file_sha256"
        ],
    }
    observed: dict[str, str] = {}
    for path, digest in expected.items():
        path_string = str(path)
        current = sha256_file(path_string)
        if current != digest:
            raise ValueError(f"V80 pinned input changed: {path_string}")
        observed[path_string] = current
    for label, (path, digest) in PRIOR_EVIDENCE.items():
        if sha256_file(path) != digest:
            raise ValueError(f"V80 prior evidence changed: {label}")
    prefix_manifest = _resolve(inputs["base_prefix_cache"]) / "manifest.json"
    manifest = json.loads(prefix_manifest.read_text(encoding="utf-8"))
    if (
        manifest.get("artifact") != "question_independent_scene_prefix_cache_v1"
        or manifest.get("scene_count") != 40
        or manifest.get("question_inputs_used") is not False
        or manifest.get("question_dependent_scene_retrieval") is not False
        or manifest.get("complete_scene_prefixes") is not True
        or manifest.get("environmental_text_inputs") != []
    ):
        raise ValueError("V80 base prefix cache contract changed")
    return {
        "file_sha256": observed,
        "prior_evidence_authenticated": sorted(PRIOR_EVIDENCE),
        "base_prefix_manifest_sha256": sha256_file(prefix_manifest),
        "base_prefix_scene_count": manifest["scene_count"],
    }


def _validate_model_metadata(config: Mapping[str, Any]) -> dict[str, Any]:
    inputs = config["inputs"]
    snapshot = (
        Path.home()
        / ".cache/huggingface/hub/models--google--gemma-4-E2B-it/snapshots"
        / inputs["model_revision"]
    )
    model_path = (snapshot / "model.safetensors").resolve(strict=True)
    if model_path.name != inputs["model_file_sha256"]:
        raise ValueError("V80 local model blob identity changed")
    model_config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    text = model_config.get("text_config")
    if (
        not isinstance(text, Mapping)
        or tuple(text.get("layer_types", ())) != EXPECTED_LAYER_TYPES
        or text.get("sliding_window") != 512
        or text.get("num_hidden_layers") != 35
        or text.get("num_kv_shared_layers") != 20
        or text.get("hidden_size") != 1536
    ):
        raise ValueError("V80 pinned Gemma topology changed")
    with safe_open(str(model_path), framework="pt", device="cpu") as archive:
        shapes = {
            name: tuple(archive.get_slice(f"{name}.weight").get_shape())
            for name in TARGET_MODULES
        }
        ignored_layer34_kv = {
            name: tuple(archive.get_slice(f"{name}.weight").get_shape())
            for name in (
                "model.language_model.layers.34.self_attn.k_proj",
                "model.language_model.layers.34.self_attn.v_proj",
            )
        }
    if shapes != TARGET_SHAPES_OUT_IN:
        raise ValueError("V80 checkpoint target shapes changed")
    return {
        "model_id": inputs["model_id"],
        "revision": inputs["model_revision"],
        "model_blob_sha256_identity": model_path.name,
        "target_shapes_out_in": {key: list(value) for key, value in shapes.items()},
        "ignored_physical_layer34_kv_shapes": {
            key: list(value) for key, value in ignored_layer34_kv.items()
        },
        "first_kv_shared_layer": 15,
        "layer_14_stores_full_length_full_attention_kv": True,
        "layer_34_reuses_full_attention_kv_from_layer_14": True,
        "full_model_loaded": False,
    }


class _StructuredProjection(nn.Module):
    """Low-memory frozen projection exposing the exact real module dimensions."""

    def __init__(self, in_features: int, out_features: int, scale: float) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = nn.Parameter(torch.tensor(float(scale)), requires_grad=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        repeats = (self.out_features + self.in_features - 1) // self.in_features
        return value.repeat_interleave(repeats, dim=-1)[..., : self.out_features] * self.scale


class _SyntheticAttention(nn.Module):
    def __init__(self, layer: int) -> None:
        super().__init__()
        self.layer_type = "full_attention"
        self.is_kv_shared_layer = layer == 34
        self.store_full_length_kv = layer == 14
        if layer == 14:
            self.k_proj = _StructuredProjection(1536, 512, 0.75)
            self.v_proj = _StructuredProjection(1536, 512, -0.5)
        else:
            self.q_proj = _StructuredProjection(1536, 4096, 0.6)
            self.o_proj = _StructuredProjection(4096, 1536, 0.8)


class _SyntheticV80Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = nn.Module()
        self.model.language_model.layers = nn.ModuleList(
            [nn.Identity() for _ in range(35)]
        )
        for layer in (14, 34):
            module = nn.Module()
            module.self_attn = _SyntheticAttention(layer)
            self.model.language_model.layers[layer] = module
        self.config = SimpleNamespace(
            text_config=SimpleNamespace(
                layer_types=EXPECTED_LAYER_TYPES,
                sliding_window=512,
                num_hidden_layers=35,
                num_kv_shared_layers=20,
            )
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        attention14 = self.model.language_model.layers[14].self_attn
        attention34 = self.model.language_model.layers[34].self_attn
        key = attention14.k_proj(value)
        val = attention14.v_proj(value)
        query = attention34.q_proj(value)
        repeats = query.shape[-1] // key.shape[-1]
        mixed = query + key.repeat_interleave(repeats, dim=-1) + val.repeat_interleave(
            repeats, dim=-1
        )
        return attention34.o_proj(torch.tanh(mixed))


def run_cpu_update_light_preflight() -> dict[str, Any]:
    """Exercise zero-init gradients and two tiny CPU updates; no Gemma weights."""

    torch.manual_seed(800080)
    model = _SyntheticV80Model().cpu()
    inputs = torch.randn(2, 7, 1536, dtype=torch.float32) * 0.1
    baseline = model(inputs).detach()
    installation = install_v80(model)
    initial = model(inputs)
    exact_delta = float((initial - baseline).detach().abs().max())
    labels = torch.tensor([3, 11], dtype=torch.long)
    logits = initial[:, -1, :23]
    loss = F.cross_entropy(logits, labels)
    loss.backward()
    gradients = installation.gradient_norms()
    zero_gate = {
        "exact_zero_output": exact_delta == 0.0,
        "all_a_gradients_exact_zero": all(
            value["residual_a"] == 0.0 for value in gradients.values()
        ),
        "all_b_gradients_finite_positive": all(
            value["residual_b"] is not None
            and math.isfinite(float(value["residual_b"]))
            and float(value["residual_b"]) > 0.0
            for value in gradients.values()
        ),
    }
    if not all(zero_gate.values()):
        raise RuntimeError(f"V80 CPU zero-step gate failed: {zero_gate}")

    optimizer = torch.optim.AdamW(
        installation.parameters(), lr=2.0e-5, weight_decay=0.0, foreach=False
    )
    traces: list[dict[str, Any]] = []
    # The first gradient is already populated; step it, then perform one more
    # update so both A and B receive nonzero gradients.
    for update in range(1, 3):
        if update > 1:
            optimizer.zero_grad(set_to_none=True)
            current = model(inputs)
            F.cross_entropy(current[:, -1, :23], labels).backward()
        pre = installation.gradient_norms()
        optimizer.step()
        traces.append({"update": update, "preupdate_gradient_l2": pre})
    changed = float((model(inputs).detach() - baseline).abs().max())
    a_changed = all(
        bool((adapter.residual_a.grad is not None) and (adapter.residual_a.grad.norm() > 0))
        for adapter in installation.adapters
    )
    return {
        "artifact": "gemma4_v80_v75_atlas_attention_reader_cpu_update_light_v1",
        "passed": changed > 0.0 and a_changed,
        "device": "cpu",
        "full_gemma_loaded": False,
        "optimizer_updates": 2,
        "parameter_count": installation.parameter_count,
        "zero_output_max_abs_delta": exact_delta,
        "zero_step_gradient_norms": gradients,
        "zero_step_checks": zero_gate,
        "postupdate_output_max_abs_delta": changed,
        "all_a_gradients_positive_on_second_update": a_changed,
        "trace": traces,
    }


def build_preregistration(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact": "gemma4_v80_v75_atlas_attention_reader_preregistration_v1",
        "status": "sealed_before_any_model_bearing_gradient_or_optimizer",
        "config_path": CONFIG,
        "config_sha256": EXPECTED_CONFIG_SHA256,
        "single_arm": True,
        "architecture": {
            "fixed_prefix_tokens": 738,
            "all_tokens_retained": True,
            "question_dependent_processing_or_selection": False,
            "target_modules": list(TARGET_MODULES),
            "rank": RANK,
            "alpha": ALPHA,
            "trainable_parameter_count": PARAMETER_COUNT,
            "physical_full_attention_kv_source_layer": 14,
            "final_full_attention_query_output_layer": 34,
            "full_huggingface_forward": True,
            "prepared_hidden_state_or_tail_forward_shortcut": False,
            "native_logits_to_keep_answer_tail_only": True,
        },
        "optimization": dict(config["optimization"]),
        "memory_safety": dict(config["memory_safety"]),
        "mandatory_zero_step_gate": dict(config["mandatory_zero_step_gate"]),
        "held_smoke_gates": dict(config["held_smoke_gates"]),
        "publication": dict(config["publication"]),
        "prior_failure_response": {
            "v6_v6_1": "removed_specialized_tail_forward_and_gradient_equivalence_surface",
            "v6_2": "reduced_96_updates_to_16_and_excluded_old_internal_validation_from_fit",
            "v6_3_v6_4": "added_final_global_QO_and_738_token_atlas_to_physical_layer14_KV",
            "v6_4_held_regression": "uses_complete_V73_pair_scene_disjoint_held_smoke_and_fail_closed_gates",
        },
        "protected_inputs_loaded": False,
        "model_loaded": False,
        "optimizer_constructed": False,
        "training_executed": False,
        "checkpoint_published": False,
        "runtime_promotion_authorized": False,
    }


def run_cpu_preflight(config_path: str | Path = CONFIG) -> dict[str, Any]:
    config = load_v80_config(config_path)
    inputs = _validate_inputs(config)
    historical = _historical_inventory(config)
    model = _validate_model_metadata(config)
    visibility = causal_prefix_visibility(
        prefix_tokens=738, prompt_tokens=64, answer_tokens=32
    )
    update_light = run_cpu_update_light_preflight()
    checks = {
        "single_locked_arm": True,
        "input_hashes_authenticated": True,
        "historical_split_exact": historical
        == {
            "all_rows": 960,
            "train_rows": EXPECTED_TRAIN_ROWS,
            "train_scenes": EXPECTED_TRAIN_SCENES,
            "train_pairs": 12,
            "train_changed_units": 40,
            "held_rows": EXPECTED_HELD_ROWS,
            "held_scenes": EXPECTED_HELD_SCENES,
            "held_pairs": 8,
            "held_smoke_units": 8,
            "held_smoke_sides": 16,
            "broad_train_rows": 16,
            "broad_held_rows": 16,
            "pair_disjoint": True,
            "scene_disjoint": True,
            "schedule_sha256": EXPECTED_SCHEDULE_SHA256,
            "held_smoke_sha256": EXPECTED_HELD_SMOKE_SHA256,
            "broad_train_sha256": EXPECTED_BROAD_TRAIN_SHA256,
            "broad_held_sha256": EXPECTED_BROAD_HELD_SHA256,
        },
        "all_738_prefix_tokens_visible": visibility["all_prefix_tokens_visible"] is True,
        "no_selection_or_top_k": visibility["selection_or_top_k"] is False,
        "shape_faithful_cpu_update_light": update_light["passed"] is True,
        "full_gemma_not_loaded": True,
        "optimizer_updates_on_real_model": True,
        "runtime_publication_forbidden": config["publication"][
            "runtime_publication_authorized"
        ]
        is False,
    }
    return {
        "schema_version": 1,
        "artifact": "gemma4_v80_v75_atlas_attention_reader_cpu_preflight_v1",
        "status": "cpu_preflight_pass_model_bearing_gradient_smoke_still_required",
        "passed": all(checks.values()),
        "config_sha256": EXPECTED_CONFIG_SHA256,
        "checks": checks,
        "inputs": inputs,
        "historical_inventory": historical,
        "model_metadata": model,
        "visibility": visibility,
        "cpu_update_light": update_light,
        "real_model": {
            "loaded": False,
            "gradient_smoke_run": False,
            "optimizer_constructed": False,
            "optimizer_updates": 0,
        },
        "official_validation_loaded": False,
        "official_test_loaded": False,
        "deferred_final_loaded": False,
        "oracle_loaded": False,
        "checkpoint_published": False,
        "runtime_promotion_authorized": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=CONFIG)
    parser.add_argument("--write-preregistration", action="store_true")
    parser.add_argument("--write-cpu-preflight", action="store_true")
    args = parser.parse_args(argv)
    config = load_v80_config(args.config)
    preregistration = build_preregistration(config)
    result: dict[str, Any] = {"preregistration": preregistration}
    if args.write_preregistration:
        path, digest = atomic_create_json(
            config["outputs"]["preregistration"], preregistration
        )
        result["preregistration_output"] = {"path": str(path), "sha256": digest}
    preflight = run_cpu_preflight(args.config)
    result["cpu_preflight"] = preflight
    if args.write_cpu_preflight:
        path, digest = atomic_create_json(config["outputs"]["cpu_preflight"], preflight)
        result["cpu_preflight_output"] = {"path": str(path), "sha256": digest}
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0 if preflight["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CONFIG",
    "EXPECTED_CONFIG_SHA256",
    "atomic_create_json",
    "build_preregistration",
    "build_schedule_v80",
    "canonical_sha256",
    "load_v80_config",
    "main",
    "run_cpu_preflight",
    "run_cpu_update_light_preflight",
    "select_broad_held_v80",
    "select_broad_train_v80",
    "select_held_smoke_v80",
]
