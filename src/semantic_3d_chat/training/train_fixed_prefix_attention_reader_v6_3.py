"""Add-only V6.3 train-only pilot for a causal Gemma attention reader.

V6.2 showed that a late MLP adapter can lower answer NLL without reliably
making answers depend on the supplied scene.  This bounded successor tests a
more causal surface: fresh rank-4 residuals around the four *already frozen*
V54 K/V projections at layers 13 and 14.  The existing modules are retained
by reference, never merged, and remain frozen.  Only the fresh FP32 A/B
factors are trainable (30,720 scalars total).

This file deliberately implements a train-only measurement.  It loads only
training scenes, training QA, fixed continuous prefixes, and a non-scene
retention corpus.  It cannot read internal validation, deferred/final, or
oracle data and has no checkpoint writer.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import random
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import torch
import torch.nn.functional as F
import yaml
from safetensors.torch import load_file
from torch import nn

from semantic_3d_chat.chat.file_audit import FileAccessAudit
from semantic_3d_chat.chat.runtime import StaticChatRuntime
from semantic_3d_chat.chat.runtime_config import (
    effective_runtime_config_sha256,
    load_runtime_config,
)
from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.evaluation.fixed_prefix_ple_v54_preregistration import (
    BASE_CHECKPOINT,
    PREFIX_CACHE,
    RUNTIME_CONFIG as BASE_RUNTIME_CONFIG,
    TRAIN_SCENES,
    VALIDATION_SCENES,
)
from semantic_3d_chat.evaluation.prediction_artifacts import checkpoint_fingerprint
from semantic_3d_chat.language.lora import LoRALinear, tensor_state_sha256
from semantic_3d_chat.training import train_fixed_prefix_decoder_reader_v6_1 as v61
from semantic_3d_chat.training import train_fixed_prefix_ple_v54 as v1

ARTIFACT: Final[str] = "gemma4_v54_fixed_prefix_attention_reader_v6_3"
CONFIG: Final[str] = (
    "configs/experiments/gemma4_v54_fixed_prefix_attention_reader_v6_3.yaml"
)
GRADIENT_REPORT: Final[str] = (
    "reports/gemma4/metrics/"
    "gemma4_v54_fixed_prefix_attention_reader_v6_3_gradient_screen.json"
)
PILOT_REPORT: Final[str] = (
    "reports/gemma4/metrics/"
    "gemma4_v54_fixed_prefix_attention_reader_v6_3_pilot.json"
)

TARGET_MODULES: Final[tuple[str, ...]] = (
    "model.language_model.layers.13.self_attn.k_proj",
    "model.language_model.layers.13.self_attn.v_proj",
    "model.language_model.layers.14.self_attn.k_proj",
    "model.language_model.layers.14.self_attn.v_proj",
)
TARGET_SHAPES: Final[dict[str, tuple[int, int]]] = {
    TARGET_MODULES[0]: (256, 1536),
    TARGET_MODULES[1]: (256, 1536),
    TARGET_MODULES[2]: (512, 1536),
    TARGET_MODULES[3]: (512, 1536),
}
RANK: Final[int] = 4
ALPHA: Final[float] = 8.0
INITIALIZATION_SEED: Final[int] = 720063
PARAMETER_COUNT: Final[int] = 30_720
BASE_CHECKPOINT_FINGERPRINT: Final[str] = (
    "3e128b40c1b73bb32750285679cda6b1bea364e67465e986a94a81dfc95e81e8"
)
BASE_RUNTIME_EFFECTIVE_SHA256: Final[str] = (
    "714c60ce9ccb1dff69c72f6618f8afb6f31bc60a830b5ee0fb794fedaa8a321e"
)
MARGIN_TARGET: Final[float] = 0.5
CORRECT_CE_WEIGHT: Final[float] = 0.1
RETENTION_WEIGHT: Final[float] = 0.25
PILOT_UPDATES: Final[int] = 8
UNITS_PER_UPDATE: Final[int] = 5
LEARNING_RATE: Final[float] = 2.0e-5
GRADIENT_CLIP: Final[float] = 0.5


def _resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return (value if value.is_absolute() else PROJECT_ROOT / value).resolve()


def _canonical_hash(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _write_report(path: str | Path, value: Mapping[str, Any]) -> Path:
    destination = _resolve(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    return destination


def _load_config() -> dict[str, Any]:
    value = yaml.safe_load(_resolve(CONFIG).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("V6.3 configuration must be a mapping")
    return value


class OuterAdditiveRankResidual(nn.Module):
    """Fresh FP32 low-rank residual around a frozen possibly-wrapped module."""

    def __init__(self, base: nn.Module, *, rank: int, alpha: float) -> None:
        super().__init__()
        if not isinstance(base, nn.Module):
            raise TypeError("V6.3 residual base must be a torch module")
        in_features = getattr(base, "in_features", None)
        out_features = getattr(base, "out_features", None)
        if type(in_features) is not int or type(out_features) is not int:
            raise TypeError("V6.3 residual base must expose integer in/out features")
        if type(rank) is not int or rank < 1:
            raise ValueError("V6.3 residual rank must be positive")
        if not math.isfinite(float(alpha)) or float(alpha) <= 0.0:
            raise ValueError("V6.3 residual alpha must be finite and positive")
        self.base = base
        self.base.requires_grad_(False)
        self.rank = rank
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        device = next(base.parameters()).device
        self.residual_a = nn.Parameter(
            torch.empty(rank, in_features, dtype=torch.float32, device=device)
        )
        self.residual_b = nn.Parameter(
            torch.zeros(out_features, rank, dtype=torch.float32, device=device)
        )

    @property
    def in_features(self) -> int:
        return int(getattr(self.base, "in_features"))

    @property
    def out_features(self) -> int:
        return int(getattr(self.base, "out_features"))

    @property
    def adapter_parameter_count(self) -> int:
        return self.rank * (self.in_features + self.out_features)

    def adapter_parameters(self) -> tuple[nn.Parameter, nn.Parameter]:
        return self.residual_a, self.residual_b

    def _apply(self, fn: Any, recurse: bool = True) -> OuterAdditiveRankResidual:
        super()._apply(fn, recurse=recurse)
        for parameter in (self.residual_a, self.residual_b):
            if parameter.dtype != torch.float32:
                parameter.data = parameter.data.float()
            if parameter.grad is not None and parameter.grad.dtype != torch.float32:
                parameter.grad.data = parameter.grad.data.float()
        return self

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        base_output = self.base(inputs)
        hidden = F.linear(inputs.float(), self.residual_a)
        update = F.linear(hidden, self.residual_b) * self.scaling
        return base_output + update.to(dtype=base_output.dtype)


@dataclass
class OuterResidualInstallation:
    target_names: tuple[str, ...]
    adapters: tuple[OuterAdditiveRankResidual, ...]

    def parameters(self) -> list[nn.Parameter]:
        return [
            parameter
            for adapter in self.adapters
            for parameter in adapter.adapter_parameters()
        ]

    @property
    def parameter_counts(self) -> dict[str, int]:
        return {
            name: adapter.adapter_parameter_count
            for name, adapter in zip(self.target_names, self.adapters, strict=True)
        }

    @property
    def parameter_count(self) -> int:
        return sum(self.parameter_counts.values())

    def state_dict(self) -> dict[str, torch.Tensor]:
        state: dict[str, torch.Tensor] = {}
        for index, adapter in enumerate(self.adapters):
            state[f"adapters.{index}.residual_a"] = adapter.residual_a
            state[f"adapters.{index}.residual_b"] = adapter.residual_b
        return state

    def state_sha256(self) -> str:
        return tensor_state_sha256(self.state_dict())

    def validate_state(self) -> None:
        if len(self.target_names) != len(self.adapters) or len(set(self.target_names)) != len(
            self.target_names
        ):
            raise ValueError("V6.3 target/adapter inventory is invalid")
        exact_surface = tuple(self.target_names) == TARGET_MODULES
        if exact_surface and self.parameter_count != PARAMETER_COUNT:
            raise ValueError("V6.3 trainable parameter count changed")
        for name, adapter in zip(self.target_names, self.adapters, strict=True):
            if exact_surface and (adapter.out_features, adapter.in_features) != TARGET_SHAPES[name]:
                raise ValueError(f"V6.3 target shape changed: {name}")
            for parameter in adapter.adapter_parameters():
                if parameter.dtype != torch.float32 or not torch.isfinite(parameter).all():
                    raise ValueError(f"V6.3 adapter state is invalid: {name}")

    def assert_only_outer_trainable(self, model: nn.Module) -> None:
        allowed = {id(parameter) for parameter in self.parameters()}
        missing = [
            name
            for name, parameter in model.named_parameters()
            if id(parameter) in allowed and not parameter.requires_grad
        ]
        unexpected = [
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and id(parameter) not in allowed
        ]
        if missing or unexpected:
            raise RuntimeError(
                "V6.3 trainable surface changed: "
                f"frozen_outer={missing}, unexpected={unexpected}"
            )

    def gradient_norms(self) -> dict[str, Any]:
        by_module: dict[str, Any] = {}
        total_squared = 0.0
        for name, adapter in zip(self.target_names, self.adapters, strict=True):
            values: dict[str, float | None] = {}
            module_squared = 0.0
            for short, parameter in (
                ("residual_a", adapter.residual_a),
                ("residual_b", adapter.residual_b),
            ):
                norm = (
                    None
                    if parameter.grad is None
                    else float(parameter.grad.detach().float().norm().cpu())
                )
                values[short] = norm
                if norm is not None:
                    module_squared += norm * norm
            values["total_l2"] = math.sqrt(module_squared)
            by_module[name] = values
            total_squared += module_squared
        return {"total_l2": math.sqrt(total_squared), "by_module": by_module}


def initialize_outer_residuals(
    installation: OuterResidualInstallation, *, seed: int = INITIALIZATION_SEED
) -> None:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for adapter in installation.adapters:
            source = torch.empty(adapter.residual_a.shape, dtype=torch.float32)
            nn.init.kaiming_uniform_(
                source, a=math.sqrt(5), generator=generator
            )
            adapter.residual_a.copy_(source.to(adapter.residual_a.device))
            adapter.residual_b.zero_()
    installation.validate_state()


def install_outer_residuals(model: nn.Module) -> OuterResidualInstallation:
    resolved: list[tuple[str, nn.Module, str, nn.Module]] = []
    for path in TARGET_MODULES:
        parent_path, _, attribute = path.rpartition(".")
        parent = model.get_submodule(parent_path)
        base = getattr(parent, attribute)
        if not isinstance(base, LoRALinear):
            raise TypeError(
                f"V6.3 must wrap the frozen inherited V54 LoRALinear: {path}; "
                f"observed={type(base).__name__}"
            )
        if (base.out_features, base.in_features) != TARGET_SHAPES[path]:
            raise ValueError(f"V6.3 inherited target shape changed: {path}")
        resolved.append((path, parent, attribute, base))
    adapters: list[OuterAdditiveRankResidual] = []
    for _path, parent, attribute, base in resolved:
        adapter = OuterAdditiveRankResidual(base, rank=RANK, alpha=ALPHA)
        setattr(parent, attribute, adapter)
        adapters.append(adapter)
    installation = OuterResidualInstallation(TARGET_MODULES, tuple(adapters))
    initialize_outer_residuals(installation)
    installation.assert_only_outer_trainable(model)
    return installation


@dataclass
class AttentionReaderBundle:
    runtime: StaticChatRuntime
    installation: OuterResidualInstallation | None
    prefixes: dict[str, torch.Tensor]
    config: dict[str, Any]

    @property
    def language(self) -> Any:
        return self.runtime.language


@dataclass(frozen=True)
class PairUnit:
    pair_id: str
    pair_question_key: str
    question: str
    first: v1.ReaderRecord
    second: v1.ReaderRecord

    @property
    def key(self) -> tuple[str, str]:
        return self.pair_id, self.pair_question_key


def build_pair_units(rows: Sequence[v1.ReaderRecord]) -> list[PairUnit]:
    grouped: defaultdict[tuple[str, str], list[v1.ReaderRecord]] = defaultdict(list)
    for row in rows:
        if row.changed:
            if row.pair_id is None or row.pair_question_key is None:
                raise ValueError("V6.3 changed training row lacks pair identity")
            grouped[(row.pair_id, row.pair_question_key)].append(row)
    result: list[PairUnit] = []
    for (pair_id, pair_key), members in sorted(grouped.items()):
        if len(members) != 2:
            raise ValueError("V6.3 pair unit must contain exactly two rows")
        members = sorted(
            members,
            key=lambda row: (row.role != "reference", row.scene_id, row.question_id),
        )
        first, second = members
        if (
            first.question != second.question
            or first.answer == second.answer
            or first.paired_scene_id != second.scene_id
            or second.paired_scene_id != first.scene_id
        ):
            raise ValueError("V6.3 pair unit is not a symmetric physical counterfactual")
        result.append(PairUnit(pair_id, pair_key, first.question, first, second))
    if len(result) != 40 or sum(len((unit.first, unit.second)) for unit in result) != 80:
        raise ValueError("V6.3 exact 40-unit training inventory changed")
    return result


def build_pilot_schedule(units: Sequence[PairUnit]) -> list[tuple[PairUnit, ...]]:
    ordered = list(units)
    random.Random(INITIALIZATION_SEED).shuffle(ordered)
    schedule = [
        tuple(ordered[index : index + UNITS_PER_UPDATE])
        for index in range(0, len(ordered), UNITS_PER_UPDATE)
    ]
    if (
        len(schedule) != PILOT_UPDATES
        or any(len(update) != UNITS_PER_UPDATE for update in schedule)
        or {unit.key for update in schedule for unit in update}
        != {unit.key for unit in units}
    ):
        raise ValueError("V6.3 balanced 8x5 pair schedule changed")
    return schedule


def training_forbidden_roots() -> list[Path]:
    roots = list(v61.training_forbidden_roots())
    roots.extend(
        _resolve(path)
        for path in (
            v1.VALIDATION_QUESTIONS,
            v1.VALIDATION_REFERENCES,
            v1.BASELINE_PREDICTIONS,
        )
    )
    prefix_root = _resolve(PREFIX_CACHE)
    roots.extend(prefix_root / f"{scene_id}.safetensors" for scene_id in VALIDATION_SCENES)
    for scene_id in VALIDATION_SCENES:
        roots.extend(
            _resolve(root) / scene_id
            for root in (
                "data/maps",
                "data/features",
                "data/rendered",
                "data_gemma4/maps",
                "data_gemma4/features",
                "data_gemma4/rendered",
                "data_gemma4/scene_tokens",
            )
        )
    return sorted(set(path.resolve() for path in roots))


def _load_train_prefixes(audit: FileAccessAudit) -> dict[str, torch.Tensor]:
    root = _resolve(PREFIX_CACHE)
    manifest_path = root / "manifest.json"
    audit.record(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    scenes = manifest.get("scenes") if isinstance(manifest, Mapping) else None
    if (
        manifest.get("artifact") != "question_independent_scene_prefix_cache_v1"
        or manifest.get("question_inputs_used") is not False
        or manifest.get("question_dependent_scene_retrieval") is not False
        or manifest.get("complete_scene_prefixes") is not True
        or manifest.get("environmental_text_inputs") != []
        or not isinstance(scenes, Mapping)
    ):
        raise ValueError("V6.3 fixed-prefix manifest contract changed")
    result: dict[str, torch.Tensor] = {}
    for scene_id in TRAIN_SCENES:
        entry = scenes.get(scene_id)
        if not isinstance(entry, Mapping):
            raise ValueError(f"V6.3 missing train prefix entry: {scene_id}")
        source = root / str(entry.get("filename"))
        audit.record(source)
        if (
            source.is_symlink()
            or not source.is_file()
            or source.stat().st_size != int(entry.get("file_size_bytes", -1))
            or v1.sha256_file(source) != entry.get("file_sha256")
        ):
            raise ValueError(f"V6.3 train prefix bytes changed: {scene_id}")
        state = load_file(str(source), device="cpu")
        if set(state) != {"scene_prefix"}:
            raise ValueError("V6.3 prefix file contains unexpected tensors")
        prefix = state["scene_prefix"].detach().contiguous()
        if (
            tuple(prefix.shape) != (1, 258, 1536)
            or prefix.dtype != torch.bfloat16
            or v1.prefix_sha256(prefix) != entry.get("prefix_sha256")
            or not torch.isfinite(prefix).all()
        ):
            raise ValueError(f"V6.3 train prefix tensor changed: {scene_id}")
        result[scene_id] = prefix
    if set(result) != set(TRAIN_SCENES):
        raise ValueError("V6.3 train-only prefix inventory changed")
    return result


def load_base_bundle(audit: FileAccessAudit) -> AttentionReaderBundle:
    config_path = _resolve(CONFIG)
    audit.record(config_path)
    experiment = _load_config()
    runtime_config = load_runtime_config(_resolve(BASE_RUNTIME_CONFIG), record_file=audit.record)
    if effective_runtime_config_sha256(runtime_config) != BASE_RUNTIME_EFFECTIVE_SHA256:
        raise ValueError("V6.3 effective runtime config changed")
    checkpoint = _resolve(BASE_CHECKPOINT)
    for source in sorted(path for path in checkpoint.rglob("*") if path.is_file()):
        audit.record(source)
    fingerprint, _ = checkpoint_fingerprint(checkpoint)
    if fingerprint != BASE_CHECKPOINT_FINGERPRINT:
        raise ValueError("V6.3 frozen V54 checkpoint changed")
    scene_id = TRAIN_SCENES[0]
    runtime = StaticChatRuntime.load(
        runtime_config,
        scene_id,
        checkpoint=checkpoint,
        audit=audit,
        local_files_only=True,
    )
    if runtime.language.device.type != "mps":
        raise RuntimeError("V6.3 bounded pilot requires local MPS")
    prefixes = _load_train_prefixes(audit)
    if runtime.scene_prefix_hash != v1.prefix_sha256(prefixes[scene_id]):
        raise ValueError("V6.3 cached fixed prefix differs from fresh frozen V54")
    runtime.language.model.requires_grad_(False)
    if any("down_proj" in name for name in TARGET_MODULES):
        raise AssertionError("V6.3 must not install the V6.2 down-projection reader")
    return AttentionReaderBundle(runtime, None, prefixes, experiment)


def _full_forward(
    bundle: AttentionReaderBundle,
    prefix: torch.Tensor,
    row: v1.ReaderRecord,
    *,
    return_answer_logits: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    prepared = v1._prepared_batch(bundle, prefix, row)
    if prepared.labels is None:
        raise RuntimeError("V6.3 full HF batch lost answer labels")
    output = bundle.language.model(
        inputs_embeds=prepared.inputs_embeds,
        attention_mask=prepared.attention_mask,
        labels=prepared.labels,
        per_layer_inputs=prepared.per_layer_inputs,
        mm_token_type_ids=prepared.mm_token_type_ids,
        use_cache=False,
        return_dict=True,
    )
    logits = output.logits
    nll = v1.token_normalized_nll(logits, prepared.labels)[0]
    if nll.ndim != 0 or not torch.isfinite(nll) or nll < 0:
        raise RuntimeError("V6.3 full HF answer NLL is invalid")
    answer_logits = None
    if return_answer_logits:
        answer_logits = logits[prepared.labels != -100].detach().cpu().contiguous()
    return nll, answer_logits


def softplus_margin_side(
    correct_nll: torch.Tensor,
    wrong_nll: torch.Tensor,
    *,
    margin_target: float = MARGIN_TARGET,
    correct_ce_weight: float = CORRECT_CE_WEIGHT,
) -> tuple[torch.Tensor, torch.Tensor]:
    if correct_nll.ndim != 0 or wrong_nll.ndim != 0:
        raise ValueError("V6.3 margin side requires scalar NLLs")
    margin = wrong_nll - correct_nll
    loss = F.softplus(float(margin_target) - margin) + float(correct_ce_weight) * correct_nll
    return loss, margin


def _answer_record(unit: PairUnit, side: int) -> v1.ReaderRecord:
    source = unit.first if side == 0 else unit.second
    return dataclasses.replace(
        unit.first,
        question=unit.question,
        answer=source.answer,
        question_id=f"{unit.pair_question_key}::answer_{side}",
    )


def _side_tensors(
    bundle: AttentionReaderBundle, unit: PairUnit, side: int
) -> tuple[torch.Tensor, torch.Tensor]:
    row = _answer_record(unit, side)
    correct_scene = unit.first.scene_id if side == 0 else unit.second.scene_id
    wrong_scene = unit.second.scene_id if side == 0 else unit.first.scene_id
    correct, _ = _full_forward(bundle, bundle.prefixes[correct_scene], row)
    wrong, _ = _full_forward(bundle, bundle.prefixes[wrong_scene], row)
    return correct, wrong


@torch.inference_mode()
def evaluate_pair_units(
    bundle: AttentionReaderBundle, units: Sequence[PairUnit]
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for unit in units:
        sides: list[dict[str, Any]] = []
        for side in (0, 1):
            correct, wrong = _side_tensors(bundle, unit, side)
            _loss, margin = softplus_margin_side(correct, wrong)
            sides.append(
                {
                    "answer": _answer_record(unit, side).answer,
                    "correct_nll": float(correct.cpu()),
                    "wrong_nll": float(wrong.cpu()),
                    "wrong_minus_correct_margin": float(margin.cpu()),
                }
            )
        records.append(
            {
                "pair_id": unit.pair_id,
                "pair_question_key": unit.pair_question_key,
                "answer_type": unit.first.answer_type,
                "scene_ids": [unit.first.scene_id, unit.second.scene_id],
                "sides": sides,
                "complete_unit": all(side["wrong_minus_correct_margin"] > 0 for side in sides),
            }
        )
    margins = [side["wrong_minus_correct_margin"] for row in records for side in row["sides"]]
    correct_nlls = [side["correct_nll"] for row in records for side in row["sides"]]
    softplus_values = [float(F.softplus(torch.tensor(MARGIN_TARGET - value))) for value in margins]
    return {
        "unit_count": len(records),
        "side_count": len(margins),
        "positive_margin_sides": sum(value > 0 for value in margins),
        "complete_units": sum(bool(row["complete_unit"]) for row in records),
        "mean_margin": sum(margins) / len(margins),
        "mean_margin_softplus": sum(softplus_values) / len(softplus_values),
        "mean_correct_nll": sum(correct_nlls) / len(correct_nlls),
        "records": records,
        "records_sha256": _canonical_hash(records),
    }


def _retention_teachers(
    bundle: AttentionReaderBundle, corpus: Sequence[Mapping[str, str]]
) -> list[torch.Tensor]:
    with torch.inference_mode():
        return [v1.retention_logits(bundle, row["prompt"]).detach().cpu() for row in corpus]


@torch.inference_mode()
def evaluate_retention(
    bundle: AttentionReaderBundle,
    corpus: Sequence[Mapping[str, str]],
    teachers: Sequence[torch.Tensor],
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for index, (row, teacher) in enumerate(zip(corpus, teachers, strict=True)):
        current = v1.retention_logits(bundle, row["prompt"]).detach().cpu().float()
        teacher = teacher.float()
        teacher_prob = torch.softmax(teacher.double(), dim=-1)
        kl = float(
            (
                teacher_prob
                * (torch.log_softmax(teacher.double(), dim=-1) - torch.log_softmax(current.double(), dim=-1))
            )
            .sum()
            .clamp_min(0.0)
        )
        records.append(
            {
                "index": index,
                "kl_nats": kl,
                "top1_agreement": int(teacher.argmax()) == int(current.argmax()),
            }
        )
    return {
        "example_count": len(records),
        "mean_kl_nats": sum(row["kl_nats"] for row in records) / len(records),
        "maximum_kl_nats": max(row["kl_nats"] for row in records),
        "top1_agreement": sum(row["top1_agreement"] for row in records) / len(records),
        "records": records,
    }


def _audit_summary(audit: FileAccessAudit) -> dict[str, Any]:
    forbidden = audit.forbidden_accesses()
    return {
        "loaded_file_count": len(audit.unique_paths),
        "loaded_files_sha256": _canonical_hash(audit.unique_paths),
        "forbidden_accesses": forbidden,
        "oracle_accessed": any(
            "oracle" in {part.casefold() for part in Path(path).parts}
            for path in forbidden
        ),
        "validation_deferred_or_final_accessed": bool(forbidden),
        "passed": not forbidden,
    }


def _screen_units(units: Sequence[PairUnit]) -> list[PairUnit]:
    by_type: dict[str, PairUnit] = {}
    for unit in units:
        by_type.setdefault(unit.first.answer_type, unit)
    chosen = [by_type["attribute"], by_type["spatial_relation"]]
    if len({unit.key for unit in chosen}) != 2:
        raise ValueError("V6.3 gradient-screen unit selection changed")
    return chosen


def run_gradient_screen() -> dict[str, Any]:
    started = time.perf_counter()
    audit = FileAccessAudit(
        training_forbidden_roots(), forbidden_component_names={"oracle"}, block_forbidden=True
    )
    with audit:
        torch.manual_seed(INITIALIZATION_SEED)
        bundle = load_base_bundle(audit)
        units = build_pair_units(v1.load_training_records())
        selected = _screen_units(units)

        identity_row = _answer_record(selected[0], 0)
        with torch.inference_mode():
            baseline_nll, baseline_logits = _full_forward(
                bundle,
                bundle.prefixes[selected[0].first.scene_id],
                identity_row,
                return_answer_logits=True,
            )
        installation = install_outer_residuals(bundle.language.model)
        bundle.installation = installation
        initial_hash = installation.state_sha256()
        with torch.inference_mode():
            wrapped_nll, wrapped_logits = _full_forward(
                bundle,
                bundle.prefixes[selected[0].first.scene_id],
                identity_row,
                return_answer_logits=True,
            )
        if baseline_logits is None or wrapped_logits is None:
            raise AssertionError("V6.3 identity screen did not retain answer logits")
        exact_logits = torch.equal(baseline_logits, wrapped_logits)
        exact_nll = torch.equal(baseline_nll.detach().cpu(), wrapped_nll.detach().cpu())

        for parameter in installation.parameters():
            parameter.grad = None
        side_records: list[dict[str, Any]] = []
        scale = 1.0 / (2.0 * len(selected))
        for unit in selected:
            unit_record = {"pair_id": unit.pair_id, "pair_question_key": unit.pair_question_key, "sides": []}
            for side in (0, 1):
                correct, wrong = _side_tensors(bundle, unit, side)
                loss, margin = softplus_margin_side(correct, wrong)
                (scale * loss).backward()
                unit_record["sides"].append(
                    {
                        "answer": _answer_record(unit, side).answer,
                        "correct_nll": float(correct.detach().cpu()),
                        "wrong_nll": float(wrong.detach().cpu()),
                        "margin": float(margin.detach().cpu()),
                        "unscaled_side_objective": float(loss.detach().cpu()),
                    }
                )
            side_records.append(unit_record)
        gradients = installation.gradient_norms()
        final_hash = installation.state_sha256()
        b_nonzero = all(
            float(values["residual_b"] or 0.0) > 0.0
            for values in gradients["by_module"].values()
        )
        a_zero = all(
            float(values["residual_a"] or 0.0) == 0.0
            for values in gradients["by_module"].values()
        )
        installation.assert_only_outer_trainable(bundle.language.model)
    audit_result = _audit_summary(audit)
    passed = bool(
        exact_logits
        and exact_nll
        and installation.parameter_count == PARAMETER_COUNT
        and gradients["total_l2"] > 0.0
        and b_nonzero
        and a_zero
        and initial_hash == final_hash
        and audit_result["passed"]
    )
    report = {
        "schema_version": 1,
        "artifact": f"{ARTIFACT}_gradient_screen",
        "status": "passed" if passed else "failed",
        "passed": passed,
        "full_huggingface_forward": True,
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "target_modules": list(TARGET_MODULES),
        "target_shapes_out_in": {key: list(value) for key, value in TARGET_SHAPES.items()},
        "trainable_parameter_count": installation.parameter_count,
        "exact_zero_initialization": {
            "answer_logits_bit_exact": exact_logits,
            "answer_nll_bit_exact": exact_nll,
            "baseline_nll": float(baseline_nll.cpu()),
            "wrapped_nll": float(wrapped_nll.cpu()),
            "state_sha256_before_backward": initial_hash,
            "state_sha256_after_backward": final_hash,
            "state_unchanged": initial_hash == final_hash,
        },
        "objective": {
            "kind": "symmetric_two_scene_by_two_answer_softplus_margin",
            "pair_units": side_records,
            "margin_target": MARGIN_TARGET,
            "correct_ce_weight": CORRECT_CE_WEIGHT,
            "pair_units_balanced_equally": True,
        },
        "gradients": gradients,
        "all_output_factor_gradients_nonzero": b_nonzero,
        "all_input_factor_gradients_exact_zero_at_zero_output_init": a_zero,
        "v6_2_down_projection_installed": False,
        "runtime_checkpoint_published": False,
        "audit": audit_result,
        "elapsed_seconds": time.perf_counter() - started,
    }
    _write_report(GRADIENT_REPORT, report)
    return report


def run_pilot() -> dict[str, Any]:
    started = time.perf_counter()
    screen = json.loads(_resolve(GRADIENT_REPORT).read_text(encoding="utf-8"))
    if screen.get("passed") is not True:
        raise RuntimeError("V6.3 pilot requires a passing no-step gradient screen")
    audit = FileAccessAudit(
        training_forbidden_roots(), forbidden_component_names={"oracle"}, block_forbidden=True
    )
    with audit:
        torch.manual_seed(INITIALIZATION_SEED)
        bundle = load_base_bundle(audit)
        units = build_pair_units(v1.load_training_records())
        schedule = build_pilot_schedule(units)
        installation = install_outer_residuals(bundle.language.model)
        bundle.installation = installation
        initial_state_hash = installation.state_sha256()
        retention = v1.load_retention_corpus()[:PILOT_UPDATES]
        teachers = _retention_teachers(bundle, retention)
        baseline_pairs = evaluate_pair_units(bundle, units)
        baseline_retention = evaluate_retention(bundle, retention, teachers)
        optimizer = torch.optim.AdamW(
            installation.parameters(),
            lr=LEARNING_RATE,
            weight_decay=0.0,
            betas=(0.9, 0.999),
            eps=1e-8,
            foreach=False,
            fused=False,
        )
        trace: list[dict[str, Any]] = []
        for update_index, update_units in enumerate(schedule, start=1):
            optimizer.zero_grad(set_to_none=True)
            unit_records: list[dict[str, Any]] = []
            scale = 1.0 / (2.0 * len(update_units))
            for unit in update_units:
                sides: list[dict[str, Any]] = []
                for side in (0, 1):
                    correct, wrong = _side_tensors(bundle, unit, side)
                    side_loss, margin = softplus_margin_side(correct, wrong)
                    (scale * side_loss).backward()
                    sides.append(
                        {
                            "answer": _answer_record(unit, side).answer,
                            "correct_nll": float(correct.detach().cpu()),
                            "wrong_nll": float(wrong.detach().cpu()),
                            "margin": float(margin.detach().cpu()),
                            "unscaled_objective": float(side_loss.detach().cpu()),
                        }
                    )
                unit_records.append(
                    {
                        "pair_id": unit.pair_id,
                        "pair_question_key": unit.pair_question_key,
                        "sides": sides,
                    }
                )
            retention_kl = v1.retention_kl_loss(
                bundle, retention[update_index - 1], teachers[update_index - 1]
            ).clamp_min(0.0)
            (RETENTION_WEIGHT * retention_kl).backward()
            preclip = float(
                torch.nn.utils.clip_grad_norm_(installation.parameters(), GRADIENT_CLIP)
                .detach()
                .cpu()
            )
            if not math.isfinite(preclip) or preclip <= 0.0:
                raise RuntimeError("V6.3 pilot gradient is invalid")
            optimizer.step()
            installation.validate_state()
            installation.assert_only_outer_trainable(bundle.language.model)
            item = {
                "update": update_index,
                "learning_rate": LEARNING_RATE,
                "pair_unit_count": len(unit_records),
                "pair_units": unit_records,
                "retention_index": update_index - 1,
                "retention_kl": float(retention_kl.detach().cpu()),
                "retention_weight": RETENTION_WEIGHT,
                "preclip_gradient_l2": preclip,
                "adapter_state_sha256": installation.state_sha256(),
            }
            trace.append(item)
            print(
                json.dumps(
                    {
                        "phase": "v6_3_attention_reader_pilot",
                        "update": update_index,
                        "updates": PILOT_UPDATES,
                        "mean_margin": sum(
                            side["margin"]
                            for unit in unit_records
                            for side in unit["sides"]
                        )
                        / (2 * len(unit_records)),
                        "preclip_gradient_l2": preclip,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        candidate_pairs = evaluate_pair_units(bundle, units)
        candidate_retention = evaluate_retention(bundle, retention, teachers)
        final_state_hash = installation.state_sha256()
    audit_result = _audit_summary(audit)
    checks = {
        "mean_margin_softplus_improved": candidate_pairs["mean_margin_softplus"]
        < baseline_pairs["mean_margin_softplus"],
        "positive_margin_sides_not_worse": candidate_pairs["positive_margin_sides"]
        >= baseline_pairs["positive_margin_sides"],
        "complete_units_not_worse": candidate_pairs["complete_units"]
        >= baseline_pairs["complete_units"],
        "retention_mean_kl_at_most_0_02": candidate_retention["mean_kl_nats"] <= 0.02,
        "retention_top1_exact": candidate_retention["top1_agreement"] == 1.0,
        "audit_clean": audit_result["passed"],
    }
    diagnostic_pass = all(checks.values())
    report = {
        "schema_version": 1,
        "artifact": f"{ARTIFACT}_pilot",
        "status": "diagnostic_pass" if diagnostic_pass else "diagnostic_fail",
        "diagnostic_pass": diagnostic_pass,
        "promotion_authorized": False,
        "runtime_checkpoint_published": False,
        "full_huggingface_forward": True,
        "target_modules": list(TARGET_MODULES),
        "trainable_parameter_count": installation.parameter_count,
        "optimization": {
            "optimizer": "AdamW",
            "learning_rate": LEARNING_RATE,
            "updates": PILOT_UPDATES,
            "pair_units_per_update": UNITS_PER_UPDATE,
            "total_pair_units": len(units),
            "all_pair_units_consumed_exactly_once": len(
                {unit.key for update in schedule for unit in update}
            )
            == len(units),
            "gradient_clip_l2": GRADIENT_CLIP,
            "initial_state_sha256": initial_state_hash,
            "final_state_sha256": final_state_hash,
        },
        "baseline_pair_metrics": baseline_pairs,
        "candidate_pair_metrics": candidate_pairs,
        "pair_metric_delta": {
            "positive_margin_sides": candidate_pairs["positive_margin_sides"]
            - baseline_pairs["positive_margin_sides"],
            "complete_units": candidate_pairs["complete_units"]
            - baseline_pairs["complete_units"],
            "mean_margin": candidate_pairs["mean_margin"] - baseline_pairs["mean_margin"],
            "mean_margin_softplus": candidate_pairs["mean_margin_softplus"]
            - baseline_pairs["mean_margin_softplus"],
            "mean_correct_nll": candidate_pairs["mean_correct_nll"]
            - baseline_pairs["mean_correct_nll"],
        },
        "baseline_retention": baseline_retention,
        "candidate_retention": candidate_retention,
        "checks": checks,
        "trace": trace,
        "trace_sha256": _canonical_hash(trace),
        "v6_2_down_projection_installed": False,
        "validation_inputs_loaded": False,
        "deferred_or_final_inputs_loaded": False,
        "oracle_inputs_loaded": False,
        "audit": audit_result,
        "elapsed_seconds": time.perf_counter() - started,
    }
    _write_report(PILOT_REPORT, report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("gradient-screen", "pilot"))
    args = parser.parse_args(argv)
    report = run_gradient_screen() if args.command == "gradient-screen" else run_pilot()
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report.get("passed", report.get("diagnostic_pass", False)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
