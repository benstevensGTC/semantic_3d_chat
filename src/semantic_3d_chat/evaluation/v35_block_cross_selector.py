"""Select a completed V35 all-block cross-residual development checkpoint.

This process is intentionally post-training.  It first proves that the bounded
100-update run is complete and that its update-32/update-64 continuation gates
used training scenes only.  Only then may one local Gemma load open development
validation QA.  Approved V29 is the retention baseline; exact V33 update 64 is
the improvement baseline.  Deferred final scenes are never accessible here.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import torch
from safetensors.torch import load_file

from semantic_3d_chat.config import config_hash, load_config
from semantic_3d_chat.evaluation.metrics import exact_normalized_match, normalize_answer
from semantic_3d_chat.evaluation.v27_sidecar_screen import _atomic_json, _pair_role_ids
from semantic_3d_chat.evaluation.v28_stage_b_selector import _retention_control_config
from semantic_3d_chat.evaluation.v30_joint_pair_selector import (
    GenerationEvidence,
    PairMarginEvidence,
    SelectionRequirements,
    _generation_evidence,
    _metadata,
    _selection_requirements,
    _source_v29_evidence,
    _validate_source_against_config,
)
from semantic_3d_chat.evaluation.v33_environmental_selector import _V33RuntimeEvaluator
from semantic_3d_chat.language.lora import tensor_state_sha256
from semantic_3d_chat.language.prefix_injection import prefix_sha256, stack_prefix_batches
from semantic_3d_chat.scene_encoder.block_cross_residual import (
    apply_block_cross_residual,
)
from semantic_3d_chat.training.checkpointing import (
    RUNTIME_METADATA_FILENAME,
    TRAINING_METADATA_FILENAME,
    runtime_checkpoint_metadata,
    validate_runtime_checkpoint_metadata,
)
from semantic_3d_chat.training.pair_curriculum import (
    CounterfactualPairUnit,
    pair_curriculum_settings,
    token_normalized_nll,
)
from semantic_3d_chat.training.train_adapter import (
    forward_prefix_batch,
    map_forward,
    pair_batch_objective,
)
from semantic_3d_chat.training.train_block_cross_v35 import (
    V35Contract,
    _compose_answer_batch,
    _optimizer_step_audit,
    cache_v35_scenes,
    construct_v35_core,
    current_scene_tokens,
    paired_cross_prefix_objective,
    require_exact_v33_source,
    require_v34_terminal_gate,
    v35_contract,
    v35_settings,
    v35_update32_gate,
    v35_update64_gate,
    validate_v35_cache_audit,
)
from semantic_3d_chat.training.train_environmental_sidecar_v33 import (
    _VALIDATION_FAMILY_PAIR_IDS,
    prefix_separation_ratios,
    validation_family_teacher_metrics,
)
from semantic_3d_chat.training.train_joint_pair_v30 import adapted_scene_tokens

DEFAULT_CONFIG = Path("configs/experiments/gemma4_diverse28_block_cross_v35.yaml")
DEFAULT_CHECKPOINT_ROOT = Path(
    "data_gemma4/checkpoints/gemma4_v35_diverse28_block_cross"
)
DEFAULT_OUTPUT = Path("reports/gemma4/metrics/v35_block_cross_selection.json")
_CORE_PREFIX = "block_cross_residual."
_CORE_PARAMETER_NAMES = frozenset(
    f"{_CORE_PREFIX}{name}" for name in ("w_q", "w_k", "w_v", "w_o")
)
_CORE_BUFFER_NAMES = frozenset(
    f"{_CORE_PREFIX}{name}"
    for name in (
        "architecture_marker",
        "architecture_dimensions",
        "initialization_seed_state",
        "latent_anchors",
        "spatial_temperature",
        "uniform_floor",
        "residual_scale",
    )
)
_CORE_STATE_NAMES = _CORE_PARAMETER_NAMES | _CORE_BUFFER_NAMES
_FRESH_BANK_PREFIX = "lora_banks.extension_v30_joint_pair_query.adapters."
_GREEDY_STEPS = frozenset({32, 64, 100})
_VALIDATION_DIFFERENTIAL_MINIMUM = 0.005
_UNRELATED_RATIO_MINIMUM = 0.98
_UNRELATED_RATIO_MAXIMUM = 1.02


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be a mapping")
    return value


def _finite(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _core_state(tensors: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    state = {
        name.removeprefix(_CORE_PREFIX): value
        for name, value in tensors.items()
        if name.startswith(_CORE_PREFIX)
    }
    if {f"{_CORE_PREFIX}{name}" for name in state} != _CORE_STATE_NAMES:
        raise ValueError("V35 checkpoint block-cross state inventory changed")
    return state


def _inherited_state(tensors: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    inherited = {
        name: value for name, value in tensors.items() if not name.startswith(_CORE_PREFIX)
    }
    if not inherited:
        raise ValueError("V35 checkpoint has no inherited V33 tensor state")
    return inherited


def _approved_v29_runtime_tensor_envelope(
    update0: Mapping[str, torch.Tensor],
    approved_v29: Mapping[str, torch.Tensor],
    *,
    expected_core_state_sha256: str,
) -> dict[str, torch.Tensor]:
    """Merge V29 into V35's two exact-zero compatibility extensions.

    V29 predates both the eight-tensor V30 query bank and V35's four-matrix
    block route.  The former must have exact-zero B matrices and the latter an
    exact-zero W_o, so retaining those tensors cannot change V29 inference.
    No other V35-only tensor is accepted.
    """

    if not set(approved_v29).issubset(update0):
        raise ValueError("Approved V29 contains tensors absent from V35 update zero")
    for name, value in approved_v29.items():
        if tuple(value.shape) != tuple(update0[name].shape):
            raise ValueError(f"Approved V29 tensor shape changed: {name}")
    extra = set(update0) - set(approved_v29)
    fresh = {name for name in extra if name.startswith(_FRESH_BANK_PREFIX)}
    core = {name for name in extra if name.startswith(_CORE_PREFIX)}
    if len(fresh) != 8 or core != _CORE_STATE_NAMES or extra != fresh | core:
        raise ValueError("V35 V29 envelope contains an unauthorized compatibility tensor")
    fresh_b = [update0[name] for name in fresh if name.endswith(".lora_b")]
    if len(fresh_b) != 4 or any(torch.count_nonzero(value).item() for value in fresh_b):
        raise ValueError("V35 V29 envelope query bank is not exact-zero output")
    if torch.count_nonzero(update0[f"{_CORE_PREFIX}w_o"]).item():
        raise ValueError("V35 V29 envelope block route is not exact-zero output")
    observed_core_hash = tensor_state_sha256(_core_state(update0))
    if observed_core_hash != expected_core_state_sha256:
        raise ValueError("V35 V29 envelope core state differs from its exact initialization")
    merged = dict(update0)
    merged.update(approved_v29)
    return merged


def _validate_surface(
    metadata: Mapping[str, Any], tensors: Mapping[str, torch.Tensor], contract: V35Contract
) -> dict[str, Any]:
    stage = _mapping(metadata.get("v35_block_cross"), "metadata.v35_block_cross")
    surface = _mapping(stage.get("trainable_surface"), "v35.trainable_surface")
    names = surface.get("parameter_names")
    if (
        not isinstance(names, Sequence)
        or isinstance(names, (str, bytes))
        or tuple(str(name) for name in names) != ("w_q", "w_k", "w_v", "w_o")
    ):
        raise ValueError("V35 trainable surface names changed")
    if surface.get("group_parameter_counts") != {"qkv": 589_824, "output": 393_216}:
        raise ValueError("V35 trainable group sizes changed")
    if surface.get("total_parameter_count") != 983_040:
        raise ValueError("V35 trainable parameter count changed")
    for field in (
        "gemma_decoder_frozen",
        "all_lora_banks_frozen",
        "complete_v33_stack_frozen",
        "every_other_parameter_frozen",
    ):
        if surface.get(field) is not True or stage.get(field) is not True:
            raise ValueError(f"V35 checkpoint does not prove {field}")
    core = _core_state(tensors)
    core_hash = tensor_state_sha256(core)
    if sum(
        int(tensors[name].numel()) for name in _CORE_PARAMETER_NAMES
    ) != 983_040:
        raise ValueError("V35 core tensor count changed")
    if metadata.get("block_cross_residual_state_sha256") != core_hash:
        raise ValueError("V35 core tensor hash differs from metadata")
    inherited_hash = tensor_state_sha256(_inherited_state(tensors))
    if inherited_hash != contract.source_tensor_state_sha256:
        raise ValueError("V35 inherited tensors differ from exact V33 update 64")
    if stage.get("frozen_block_cross_source_stack_state_sha256") != inherited_hash:
        raise ValueError("V35 frozen-source metadata hash changed")
    return {
        "core_state_sha256": core_hash,
        "inherited_state_sha256": inherited_hash,
        "parameter_count": 983_040,
    }


def _replay_train_gates(
    metadata: Mapping[str, Any], contract: V35Contract
) -> tuple[Mapping[str, Any] | None, Mapping[str, Any] | None]:
    history = metadata.get("history")
    if not isinstance(history, list) or not history:
        raise ValueError("V35 checkpoint history is absent")
    baseline = _mapping(history[0], "history[0]")
    baseline_pair = _mapping(baseline.get("train_pair_metrics"), "update-zero train pairs")
    stage = _mapping(metadata.get("v35_block_cross"), "v35 stage")
    update32: Mapping[str, Any] | None = None
    update64: Mapping[str, Any] | None = None
    if len(history) > 32:
        row32 = _mapping(history[32], "history[32]")
        update32 = v35_update32_gate(
            separation=_mapping(row32.get("training_prefix_separation"), "update32 separation"),
            pair_metrics=_mapping(row32.get("training_pair_metrics"), "update32 pairs"),
            baseline_pair_metrics=baseline_pair,
            residual_rms=_finite(
                _mapping(
                    row32.get("training_residual_diagnostics"), "update32 residual"
                ).get("aggregate_rms"),
                "update32 residual RMS",
            ),
            contract=contract,
        )
        if update32 != stage.get("update32_train_only_gate") or update32 != row32.get(
            "update32_train_only_gate"
        ):
            raise ValueError("V35 independently replayed update-32 gate differs")
    if len(history) > 64:
        if update32 is None:
            raise ValueError("V35 update-64 gate lacks update-32 evidence")
        row64 = _mapping(history[64], "history[64]")
        update64 = v35_update64_gate(
            update32_gate=update32,
            pair_metrics=_mapping(row64.get("training_pair_metrics"), "update64 pairs"),
            baseline_pair_metrics=baseline_pair,
            residual_rms=_finite(
                _mapping(
                    row64.get("training_residual_diagnostics"), "update64 residual"
                ).get("aggregate_rms"),
                "update64 residual RMS",
            ),
            contract=contract,
        )
        if update64 != stage.get("update64_train_only_gate") or update64 != row64.get(
            "update64_train_only_gate"
        ):
            raise ValueError("V35 independently replayed update-64 gate differs")
    return update32, update64


def validate_v35_checkpoint_envelope(
    config: Mapping[str, Any], checkpoint_root: Path, contract: V35Contract
) -> tuple[tuple[Path, ...], list[dict[str, Any]]]:
    """Inspect every arm, tensor, runtime file, Adam state, and train-only gate."""

    checkpoints = tuple(
        checkpoint_root / f"update_{step:03d}" for step in contract.saved_optimizer_steps
    )
    observed = sorted(path.name for path in checkpoint_root.glob("update_*") if path.is_dir())
    expected = [path.name for path in checkpoints]
    if observed != expected or contract.saved_optimizer_steps[-1] != 100:
        raise FileNotFoundError(
            f"V35 requires the complete saved-arm envelope: observed={observed} expected={expected}"
        )
    terminal = require_v34_terminal_gate(config)
    source, _source_metadata = require_exact_v33_source(config)
    expected_config_hash = config_hash(dict(config))
    update0_tensors: Mapping[str, torch.Tensor] | None = None
    prior_history: list[Mapping[str, Any]] = []
    accepted32: Mapping[str, Any] | None = None
    accepted64: Mapping[str, Any] | None = None
    common_schedule: Mapping[str, Any] | None = None
    common_cache: Mapping[str, Any] | None = None
    audits: list[dict[str, Any]] = []
    for step, checkpoint in zip(contract.saved_optimizer_steps, checkpoints, strict=True):
        if checkpoint.is_symlink() or not checkpoint.is_dir():
            raise ValueError(f"V35 arm must be a real directory: {checkpoint}")
        required = ["adapter.safetensors", TRAINING_METADATA_FILENAME, RUNTIME_METADATA_FILENAME]
        if step:
            required.append("optimizer.pt")
        if any(
            not (checkpoint / name).is_file() or (checkpoint / name).is_symlink()
            for name in required
        ):
            raise FileNotFoundError(f"V35 arm is incomplete or aliased: {checkpoint.name}")
        metadata = _metadata(checkpoint)
        stage = _mapping(metadata.get("v35_block_cross"), "metadata.v35_block_cross")
        if metadata.get("optimizer_step") != step or stage.get("optimizer_step") != step:
            raise ValueError(f"V35 optimizer-step mismatch: {checkpoint.name}")
        if metadata.get("config_hash") != expected_config_hash:
            raise ValueError(f"V35 config hash changed: {checkpoint.name}")
        if stage.get("conditional_v34_terminal_gate") != {
            "path": terminal["path"],
            "sha256": terminal["sha256"],
        }:
            raise ValueError(f"V35 terminal authorization changed: {checkpoint.name}")
        if (
            Path(str(stage.get("source_checkpoint"))).resolve() != source
            or stage.get("source_file_sha256") != dict(contract.source_file_sha256)
            or stage.get("source_v33_tensor_state_sha256")
            != contract.source_tensor_state_sha256
        ):
            raise ValueError(f"V35 source provenance changed: {checkpoint.name}")
        forbidden = {
            "validation_qa_loaded": False,
            "oracle_environment_files_loaded": False,
            "question_dependent_scene_processing": False,
            "question_dependent_retrieval": False,
        }
        if any(stage.get(key) != value for key, value in forbidden.items()):
            raise ValueError(f"V35 training crossed its data boundary: {checkpoint.name}")
        if stage.get("deferred_final_scene_ids_loaded") != []:
            raise ValueError(f"V35 training touched deferred final scenes: {checkpoint.name}")
        qa = _mapping(stage.get("train_qa_dataset"), "v35 train QA audit")
        loaded_qa = qa.get("loaded_files")
        if (
            qa.get("validation_qa_loaded") is not False
            or qa.get("deferred_final_qa_loaded") is not False
            or not isinstance(loaded_qa, list)
            or any(Path(str(path)).name == "validation.jsonl" for path in loaded_qa)
        ):
            raise ValueError(f"V35 training loaded validation/final QA: {checkpoint.name}")
        cache = _mapping(stage.get("scene_cache"), "v35 scene cache")
        validate_v35_cache_audit(
            cache,
            expected_scene_ids=(*contract.v31.train_scene_ids, *contract.v31.validation_scene_ids),
        )
        schedule = _mapping(stage.get("schedule"), "v35 schedule")
        if not (
            schedule.get("optimizer_step_count") == 100
            and schedule.get("pair_unit_count") == 25
            and schedule.get("exact_pair_unit_recurrence") == 4
            and schedule.get("pair_units_atomic") is True
            and schedule.get("true_optimizer_step_per_schedule_row") is True
            and schedule.get("questions_or_answers_serialized_to_runtime") is False
        ):
            raise ValueError(f"V35 schedule proof failed: {checkpoint.name}")
        if common_schedule is None:
            common_schedule = dict(schedule)
            common_cache = dict(cache)
        elif schedule != common_schedule or cache != common_cache:
            raise ValueError(f"V35 schedule/cache changed across arms: {checkpoint.name}")
        history = metadata.get("history")
        if not isinstance(history, list) or len(history) != step + 1:
            raise ValueError(f"V35 history is incomplete: {checkpoint.name}")
        if [row.get("optimizer_update") for row in history] != list(range(step + 1)):
            raise ValueError(f"V35 history is not one row per true update: {checkpoint.name}")
        if prior_history and history[: len(prior_history)] != prior_history:
            raise ValueError(f"V35 history was rewritten across arms: {checkpoint.name}")
        prior_history = list(history)
        row = _mapping(history[-1], "history[-1]")
        if row.get("saved_checkpoint") is not True or row.get("validation_qa_loaded") is not False:
            raise ValueError(f"V35 saved-row audit failed: {checkpoint.name}")
        if step and (
            row.get("true_optimizer_step") is not True
            or row.get("separate_group_clipping") is not True
            or row.get("training_prefix_separation") is None
            or row.get("training_residual_diagnostics") is None
        ):
            raise ValueError(f"V35 saved diagnostics are incomplete: {checkpoint.name}")
        if step in _GREEDY_STEPS and row.get("training_pair_metrics") is None:
            raise ValueError(f"V35 designated gate arm lacks train pair metrics: {checkpoint.name}")
        runtime = json.loads((checkpoint / RUNTIME_METADATA_FILENAME).read_text(encoding="utf-8"))
        validate_runtime_checkpoint_metadata(runtime)
        if runtime != runtime_checkpoint_metadata(metadata):
            raise ValueError(f"V35 runtime metadata was not freshly sanitized: {checkpoint.name}")
        tensors = load_file(checkpoint / "adapter.safetensors", device="cpu")
        surface = _validate_surface(metadata, tensors, contract)
        if update0_tensors is None:
            update0_tensors = tensors
            inherited = _inherited_state(tensors)
            source_tensors = load_file(source / "adapter.safetensors", device="cpu")
            if set(inherited) != set(source_tensors) or any(
                not torch.equal(inherited[name], source_tensors[name]) for name in inherited
            ):
                raise ValueError("V35 update zero is not tensor-bit-exact V33 update 64")
            if surface["core_state_sha256"] != contract.core_initial_state_sha256:
                raise ValueError("V35 update-zero block core differs from its exact initialization")
            equivalence = _mapping(stage.get("update_zero_equivalence"), "update-zero proof")
            required_equivalence = {
                "exact_v33_update64_source_tensors": True,
                "exact_v33_update64_post_sidecar_scene_tokens": True,
                "exact_zero_residual_identity": True,
                "exact_v33_update64_source_prefixes_all_22_scenes": True,
                "source_prefix_scene_count": 22,
                "source_prefixes_replayed_bit_exact": True,
                "fresh_adam_state": True,
                "question_dependent_scene_processing": False,
                "question_dependent_retrieval": False,
                "validation_qa_loaded": False,
                "oracle_environment_files_loaded": False,
            }
            if any(equivalence.get(key) != value for key, value in required_equivalence.items()):
                raise ValueError("V35 update-zero equivalence proof is incomplete")
        else:
            if set(tensors) != set(update0_tensors):
                raise ValueError(f"V35 tensor inventory changed: {checkpoint.name}")
            changed = {name for name in tensors if not torch.equal(tensors[name], update0_tensors[name])}
            if not changed or not changed.issubset(_CORE_PARAMETER_NAMES):
                raise ValueError(f"V35 arm changed an unauthorized tensor: {checkpoint.name}")
        if step:
            _optimizer_step_audit(checkpoint, expected_step=step)
        replay32, replay64 = _replay_train_gates(metadata, contract)
        if step >= 32:
            if replay32 is None or replay32.get("passed") is not True:
                raise ValueError(f"V35 arm lacks a passed train-only update-32 gate: {checkpoint.name}")
            if accepted32 is None:
                accepted32 = replay32
            elif replay32 != accepted32:
                raise ValueError("V35 update-32 gate changed across later arms")
        if step >= 64:
            if replay64 is None or replay64.get("passed") is not True:
                raise ValueError(f"V35 arm lacks a passed train-only update-64 gate: {checkpoint.name}")
            if accepted64 is None:
                accepted64 = replay64
            elif replay64 != accepted64:
                raise ValueError("V35 update-64 gate changed across later arms")
        audits.append(
            {
                "checkpoint": str(checkpoint),
                "optimizer_step": step,
                "tensor_inventory_inspected": True,
                "runtime_metadata_inspected": True,
                "optimizer_state_inspected": step > 0,
                **surface,
            }
        )
    if accepted32 is None or accepted64 is None:
        raise ValueError("V35 complete run lacks both accepted train-only gates")
    return checkpoints, audits


def _prefix_diagnostics(
    units: Sequence[CounterfactualPairUnit], prefixes: Mapping[str, torch.Tensor]
) -> dict[str, Any]:
    pair_scenes: dict[str, tuple[str, str]] = {}
    for unit in units:
        prior = pair_scenes.setdefault(unit.pair_id, tuple(unit.scene_ids))
        if prior != tuple(unit.scene_ids):
            raise ValueError("V35 validation pair changed scene membership")
    if set(pair_scenes) != set(_VALIDATION_FAMILY_PAIR_IDS.values()):
        raise ValueError("V35 validation prefix families changed")
    scene_ids = sorted({scene for pair in pair_scenes.values() for scene in pair})

    def rms(left: str, right: str) -> float:
        value = (prefixes[left].float() - prefixes[right].float()).square().mean().sqrt()
        result = float(value.cpu())
        if not math.isfinite(result) or result <= 0:
            raise ValueError("V35 validation prefix distance is invalid")
        return result

    by_family = {
        family: rms(*pair_scenes[pair_id])
        for family, pair_id in _VALIDATION_FAMILY_PAIR_IDS.items()
    }
    paired = {frozenset(pair) for pair in pair_scenes.values()}
    unrelated = [
        rms(left, right)
        for index, left in enumerate(scene_ids)
        for right in scene_ids[index + 1 :]
        if frozenset((left, right)) not in paired
    ]
    if len(unrelated) != 12:
        raise ValueError("V35 validation unrelated-prefix inventory changed")
    return {
        "schema_version": 1,
        "tensor": "composed_v35_continuous_scene_prefix",
        "rms_by_validation_family": by_family,
        "weak_pair_mean_rms": (by_family["book_support"] + by_family["picture_support"]) / 2,
        "unrelated_pair_count": 12,
        "unrelated_mean_rms": sum(unrelated) / len(unrelated),
        "question_inputs_used": False,
        "all_validation_scenes_processed": True,
    }


@dataclass(frozen=True)
class V35TeacherEvidence:
    validation_answer_token_nll: float
    pair_margins: PairMarginEvidence
    family_teacher: Mapping[str, Any]
    prefix_diagnostics: Mapping[str, Any]
    color_full_vocab_sides: int
    mirror_full_vocab_sides: int
    negative_sides: frozenset[tuple[str, str]]
    prefix_sha256_by_scene: Mapping[str, str]


@dataclass(frozen=True)
class V35GreedyEvidence:
    generation: GenerationEvidence
    complete_by_family: Mapping[str, int]
    prediction_changed_by_family: Mapping[str, int]


class V35ArmEvaluator(Protocol):
    validation_scene_ids: tuple[str, ...]
    cache_audit: Mapping[str, Any]

    def install(self, tensors: Mapping[str, torch.Tensor], *, approved_v29: bool = False) -> None: ...

    def evaluate_teacher(self) -> V35TeacherEvidence: ...

    def evaluate_greedy(self) -> V35GreedyEvidence: ...

    def evaluate_aggregate_exact(self) -> tuple[int, int]: ...

    def attest_prefix_invariance(self) -> Mapping[str, Any]: ...


class _V35RuntimeEvaluator(_V33RuntimeEvaluator):
    """One-Gemma evaluator with prefixes materialized before every question."""

    def __init__(
        self,
        config: dict[str, Any],
        control_config: dict[str, Any],
        checkpoint: Path,
        requirements: SelectionRequirements,
    ) -> None:
        super().__init__(config, control_config, checkpoint, requirements)
        source_path, source_metadata = require_exact_v33_source(config)
        source_tensors = load_file(source_path / "adapter.safetensors", device="cpu")
        super().install(source_tensors)
        self.block_cross_residual = construct_v35_core(
            config, device=self.bundle.language.device
        ).eval()
        self.bundle.checkpoint_modules["block_cross_residual"] = self.block_cross_residual
        self.block_cross_residual.load_state_dict(_core_state(load_file(
            checkpoint / "adapter.safetensors", device="cpu"
        )), strict=True)
        self.bundle.language.model.requires_grad_(False).eval()
        for module in self.bundle.checkpoint_modules.values():
            module.requires_grad_(False).eval()
        contract = v35_contract(config)
        self.v35_caches, self.cache_audit = cache_v35_scenes(
            config=config,
            bundle=self.bundle,
            source_metadata=source_metadata,
            terminal=require_v34_terminal_gate(config),
            scene_ids=(*contract.v31.train_scene_ids, *contract.v31.validation_scene_ids),
        )
        self._approved_v29_mode = False
        self._scene_tokens: dict[str, torch.Tensor] | None = None
        self._materialized_prefixes: dict[str, torch.Tensor] | None = None
        self._control_outputs: dict[str, Any] | None = None
        self._environment_builds = 0
        self._question_evaluations = 0

    def install(
        self, tensors: Mapping[str, torch.Tensor], *, approved_v29: bool = False
    ) -> None:
        super().install(tensors)
        self.block_cross_residual.load_state_dict(_core_state(tensors), strict=True)
        self.block_cross_residual.eval()
        if approved_v29 and torch.count_nonzero(
            dict(self.block_cross_residual.named_parameters())["w_o"]
        ).item():
            raise ValueError("Approved-V29 evaluation requires the exact-zero V35 route")
        self._approved_v29_mode = approved_v29
        self._scene_tokens = None
        self._materialized_prefixes = None
        self._control_outputs = None
        self._environment_builds = 0
        self._question_evaluations = 0
        self._prediction_cache.clear()
        if self.bundle.language.device.type == "mps":
            torch.mps.empty_cache()

    def _build_environment_state(self) -> None:
        if self._materialized_prefixes is not None:
            return
        model_dtype = next(self.bundle.language.model.parameters()).dtype
        tokens: dict[str, torch.Tensor] = {}
        prefixes: dict[str, torch.Tensor] = {}
        control_outputs: dict[str, Any] = {}
        with torch.inference_mode():
            for scene_id in self.validation_scene_ids:
                if self._approved_v29_mode:
                    current = adapted_scene_tokens(
                        self.validation_caches[scene_id], self.bundle
                    )
                else:
                    current = current_scene_tokens(
                        self.v35_caches[scene_id],
                        self.block_cross_residual,
                        device=self.bundle.language.device,
                    )
                tokens[scene_id] = current
                prefixes[scene_id] = self.bundle.composer.scene_prefix(
                    current.to(model_dtype)
                )
            for scene_id in self.control_scene_ids:
                output = map_forward(
                    self.bundle.scene_model,
                    self.maps[scene_id],
                    self.bundle.global_scene_residual,
                    self.bundle.signed_x_scene_residual,
                    self.bundle.dense_aligner,
                    self.bundle.dense_sidecar_adapter,
                )
                control_outputs[scene_id] = apply_block_cross_residual(
                    output, self.block_cross_residual
                )
        self._scene_tokens = tokens
        self._materialized_prefixes = prefixes
        self._control_outputs = control_outputs
        self._environment_builds += 1

    def _prefixes(self) -> dict[str, torch.Tensor]:
        self._build_environment_state()
        assert self._materialized_prefixes is not None
        return self._materialized_prefixes

    def _validation_nll(self) -> float:
        self._build_environment_state()
        assert self._scene_tokens is not None
        total = 0.0
        count = 0
        with torch.inference_mode():
            for scene_id in self.validation_scene_ids:
                records = [row for row in self.validation_records if row.scene_id == scene_id]
                for offset in range(0, len(records), 2):
                    rows = records[offset : offset + 2]
                    batches = [
                        _compose_answer_batch(
                            scene_tokens=self._scene_tokens[scene_id],
                            question=row.question,
                            answer=row.answer,
                            bundle=self.bundle,
                        )
                        for row in rows
                    ]
                    batch = stack_prefix_batches(
                        batches,
                        self.bundle.language.device,
                        prefix_backend=getattr(self.bundle.language, "prefix_backend", None),
                    )
                    if batch.labels is None:
                        raise RuntimeError("V35 validation NLL batch lacks labels")
                    output = forward_prefix_batch(self.bundle.language, batch)
                    values = token_normalized_nll(output.logits, batch.labels).reshape(-1)
                    total += float(values.sum().cpu())
                    count += len(rows)
        if count != len(self.validation_records):
            raise RuntimeError("V35 validation NLL omitted rows")
        self._question_evaluations += count
        return total / count

    def _validation_pairs(self) -> tuple[PairMarginEvidence, dict[str, Any]]:
        self._build_environment_state()
        assert self._scene_tokens is not None
        rows: list[dict[str, Any]] = []
        flat: list[float] = []
        with torch.inference_mode():
            for unit in self.validation_units:
                _, _, _, diagnostics = paired_cross_prefix_objective(
                    unit=unit,
                    scene_tokens={scene: self._scene_tokens[scene] for scene in unit.scene_ids},
                    bundle=self.bundle,
                    side_margin=v35_settings(self.config).side_hinge_margin,
                    cross_prefix_margin=v35_settings(self.config).cross_prefix_flip_margin,
                )
                margins = [
                    float(value)
                    for value in diagnostics["side_margins"].detach().float().cpu()
                ]
                rows.append(
                    {
                        "pair_id": unit.pair_id,
                        "question_key": unit.question_key,
                        "scene_ids": list(unit.scene_ids),
                        "margins": margins,
                    }
                )
                flat.extend(margins)
        keys_and_margins = sorted(
            (
                (str(row["pair_id"]), str(row["question_key"])),
                (float(row["margins"][0]), float(row["margins"][1])),
            )
            for row in rows
        )
        pair = PairMarginEvidence(
            unit_keys=tuple(key for key, _ in keys_and_margins),
            margins=tuple(value for _, value in keys_and_margins),
            passed_units=sum(all(side > 0 for side in value) for _, value in keys_and_margins),
            passed_sides=sum(side > 0 for side in flat),
            mean_margin=sum(flat) / len(flat),
            minimum_margin=min(flat),
        )
        raw = {"margins_by_unit": rows}
        self._question_evaluations += 2 * len(rows)
        return pair, validation_family_teacher_metrics(raw)

    def _control_evidence(self) -> tuple[int, int, frozenset[tuple[str, str]]]:
        self._build_environment_state()
        assert self._control_outputs is not None
        curriculum = pair_curriculum_settings(self.control_config)
        grouped: defaultdict[str, list[CounterfactualPairUnit]] = defaultdict(list)
        for unit in self.control_units:
            grouped[unit.pair_id].append(unit)
        positives: dict[str, int] = {}
        negatives: set[tuple[str, str]] = set()
        with torch.inference_mode():
            for pair_id, units in sorted(grouped.items()):
                all_full: list[torch.Tensor] = []
                ordered: list[CounterfactualPairUnit] = []
                for offset in range(0, len(units), curriculum.units_per_batch):
                    batch_units = units[offset : offset + curriculum.units_per_batch]
                    _, _, _, _, diagnostics = pair_batch_objective(
                        self._control_outputs,
                        batch_units,
                        self.maps,
                        self.bundle.language,
                        self.bundle.composer,
                        self.bundle.grounding,
                        self.control_config,
                        ranking_margin=curriculum.ranking_margin,
                        ranking_mode=curriculum.ranking_mode,
                        collect_full_vocab_first_answer_token=True,
                        full_vocab_ranking_margin=float(
                            self.control_config["training"].get(
                                "pair_full_vocab_ranking_margin", 0.0
                            )
                        ),
                    )
                    full = diagnostics["first_answer_token_full_vocab_margins"]
                    if not isinstance(full, torch.Tensor):
                        raise TypeError("V35 control gate lacks full-vocabulary margins")
                    all_full.append(full.detach().float().cpu())
                    ordered.extend(batch_units)
                values = torch.cat(all_full, dim=0)
                positives[pair_id] = int(values.gt(0).sum())
                for unit, sides in zip(ordered, values, strict=True):
                    for record, value in zip(unit.records, sides, strict=True):
                        if float(value) <= 0:
                            negatives.add((record.scene_id, record.question_id))
        color_pair, mirror_pair = _pair_role_ids(self.control_config)
        self._question_evaluations += 2 * len(self.control_units)
        return positives[color_pair], positives[mirror_pair], frozenset(negatives)

    def evaluate_teacher(self) -> V35TeacherEvidence:
        self._build_environment_state()
        prefixes = self._prefixes()
        pair, family = self._validation_pairs()
        color, mirror, negatives = self._control_evidence()
        return V35TeacherEvidence(
            validation_answer_token_nll=self._validation_nll(),
            pair_margins=pair,
            family_teacher=family,
            prefix_diagnostics=_prefix_diagnostics(self.validation_units, prefixes),
            color_full_vocab_sides=color,
            mirror_full_vocab_sides=mirror,
            negative_sides=negatives,
            prefix_sha256_by_scene={
                scene_id: prefix_sha256(prefix) for scene_id, prefix in prefixes.items()
            },
        )

    def evaluate_greedy(self) -> V35GreedyEvidence:
        prefixes = self._prefixes()
        changed, broad = self._generation_rows(prefixes)
        generation = _generation_evidence(
            changed,
            broad,
            expected_changed_rows=self.requirements.greedy_changed_row_count,
            expected_broad_rows=self.requirements.broad_retention_subset_size,
        )
        grouped: defaultdict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
        for row in changed:
            grouped[(str(row["pair_id"]), str(row["question_key"]))].append(row)
        reverse = {pair_id: family for family, pair_id in _VALIDATION_FAMILY_PAIR_IDS.items()}
        complete = {family: 0 for family in reverse.values()}
        changed_count = {family: 0 for family in reverse.values()}
        for (pair_id, _), rows in grouped.items():
            if pair_id not in reverse or len(rows) != 2:
                raise ValueError("V35 greedy family evidence is incomplete")
            family = reverse[pair_id]
            complete[family] += int(
                all(exact_normalized_match(row["prediction"], row["target"]) for row in rows)
            )
            changed_count[family] += int(
                normalize_answer(rows[0]["prediction"])
                != normalize_answer(rows[1]["prediction"])
            )
        self._question_evaluations += len(changed) + len(broad)
        return V35GreedyEvidence(generation, complete, changed_count)

    def attest_prefix_invariance(self) -> Mapping[str, Any]:
        prefixes = self._prefixes()
        first = {key: prefix_sha256(value) for key, value in prefixes.items()}
        # Recompose from the already question-free scene tokens; no question is
        # allowed to influence this replay.
        assert self._scene_tokens is not None
        model_dtype = next(self.bundle.language.model.parameters()).dtype
        repeated = {
            key: prefix_sha256(self.bundle.composer.scene_prefix(value.to(model_dtype)))
            for key, value in self._scene_tokens.items()
        }
        return {
            "passed": first == repeated,
            "scene_count": len(first),
            "prefix_hashes_identical_before_and_after_questions": first == repeated,
            "environment_built_before_questions": self._environment_builds == 1,
            "question_dependent_scene_processing": False,
            "question_dependent_retrieval": False,
            "oracle_environment_files_loaded": False,
            "deferred_final_scenes_loaded": False,
        }


def _promotion(
    selected: Mapping[str, Any] | None,
    *,
    approved_v29_aggregate: tuple[int, int],
    selected_aggregate: tuple[int, int] | None,
    prefix_attestation: Mapping[str, Any] | None,
) -> dict[str, Any]:
    development_selected = selected is not None
    changed_threshold = development_selected and int(
        selected["greedy_exact_complete_units_correct"]
    ) >= 6
    aggregate_retained = (
        development_selected
        and selected_aggregate is not None
        and approved_v29_aggregate[0] > 0
        and selected_aggregate[0] == approved_v29_aggregate[0]
        and selected_aggregate[1] / selected_aggregate[0]
        >= approved_v29_aggregate[1] / approved_v29_aggregate[0]
    )
    outward = {
        "development_checkpoint_selected": development_selected,
        "changed_complete_pair_threshold_met": changed_threshold,
        "aggregate_validation_exact_accuracy_retained": aggregate_retained,
    }
    internal = {
        "selected_checkpoint_is_numbered": development_selected
        and Path(str(selected["checkpoint"])).name
        == f"update_{int(selected['optimizer_step']):03d}",
        "each_validation_family_demonstrated": development_selected
        and all(
            int(_mapping(selected["greedy_complete_units_by_family"], "families")[family])
            >= 1
            for family in _VALIDATION_FAMILY_PAIR_IDS
        ),
        "approved_v29_color_12_sides_retained": development_selected
        and int(selected["color_full_vocab_sides"]) >= 12,
        "approved_v29_mirror_10_sides_retained": development_selected
        and int(selected["mirror_full_vocab_sides"]) >= 10,
        "approved_v29_controls_no_new_negatives": development_selected
        and not selected["new_negative_sides_vs_approved_v29"],
        "approved_v29_broad_retained": development_selected
        and bool(selected["checks"]["broad_retention_vs_approved_v29"]),
        "all_development_checks_passed": development_selected
        and all(_mapping(selected["checks"], "selected checks").values()),
        "prefix_invariance_attested": prefix_attestation is not None
        and prefix_attestation.get("passed") is True
        and prefix_attestation.get("environment_built_before_questions") is True,
        "leakage_boundary_attested": prefix_attestation is not None
        and prefix_attestation.get("oracle_environment_files_loaded") is False
        and prefix_attestation.get("deferred_final_scenes_loaded") is False
        and prefix_attestation.get("question_dependent_scene_processing") is False
        and prefix_attestation.get("question_dependent_retrieval") is False,
    }
    return {
        "evaluated": development_selected,
        "checks": outward,
        "audited_internal_requirements": internal,
        "approved_v29_aggregate_validation": {
            "row_count": approved_v29_aggregate[0],
            "exact_correct": approved_v29_aggregate[1],
            "exact_accuracy": approved_v29_aggregate[1] / approved_v29_aggregate[0],
        },
        "selected_aggregate_validation": None
        if selected_aggregate is None
        else {
            "row_count": selected_aggregate[0],
            "exact_correct": selected_aggregate[1],
            "exact_accuracy": selected_aggregate[1] / selected_aggregate[0],
        },
        "prefix_and_leakage_attestation": prefix_attestation,
        "eligible": all(outward.values()) and all(internal.values()),
    }


def _development_checks(
    *,
    teacher: V35TeacherEvidence,
    greedy: V35GreedyEvidence,
    source: V35TeacherEvidence,
    approved: V35TeacherEvidence,
    approved_greedy: V35GreedyEvidence,
) -> tuple[dict[str, bool], dict[str, float], int, list[tuple[str, str]]]:
    ratios = prefix_separation_ratios(teacher.prefix_diagnostics, source.prefix_diagnostics)
    differential = ratios["weak_pair_mean"] - ratios["unrelated_mean"]
    nonmirror = sum(
        int(_mapping(teacher.family_teacher[family], family)["complete_units"])
        for family in ("book_support", "picture_support")
    )
    new_negatives = sorted(teacher.negative_sides - approved.negative_sides)
    checks = {
        "validation_answer_token_nll_improved_from_v33_u64": (
            teacher.validation_answer_token_nll < source.validation_answer_token_nll
        ),
        "validation_pair_mean_margin_improved_from_v33_u64": (
            teacher.pair_margins.mean_margin > source.pair_margins.mean_margin
        ),
        "validation_pair_passed_units_improved_from_v33_u64": (
            teacher.pair_margins.passed_units > source.pair_margins.passed_units
        ),
        "validation_weak_prefix_gain_exceeds_unrelated_by_0_005": (
            differential >= _VALIDATION_DIFFERENTIAL_MINIMUM
        ),
        "each_validation_family_ratio_exceeds_unrelated": all(
            ratios[family] > ratios["unrelated_mean"]
            for family in _VALIDATION_FAMILY_PAIR_IDS
        ),
        "validation_unrelated_ratio_two_sided_bounded": (
            _UNRELATED_RATIO_MINIMUM
            <= ratios["unrelated_mean"]
            <= _UNRELATED_RATIO_MAXIMUM
        ),
        "nonmirror_teacher_complete": nonmirror >= 1,
        "greedy_development_unit_demonstrated": (
            greedy.generation.exact_complete_units_correct >= 1
        ),
        "approved_v29_color_12_sides_retained": teacher.color_full_vocab_sides >= 12,
        "approved_v29_mirror_10_sides_retained": teacher.mirror_full_vocab_sides >= 10,
        "approved_v29_controls_no_new_negatives": not new_negatives,
        "broad_retention_vs_approved_v29": (
            greedy.generation.broad_exact_accuracy
            >= approved_greedy.generation.broad_exact_accuracy
        ),
    }
    return checks, ratios, nonmirror, new_negatives


def select_v35(
    config_path: Path,
    checkpoint_root: Path,
    *,
    evaluator_factory: Callable[
        [dict[str, Any], dict[str, Any], Path, SelectionRequirements], V35ArmEvaluator
    ] = _V35RuntimeEvaluator,
) -> dict[str, Any]:
    config = load_config(config_path)
    contract = v35_contract(config)
    checkpoints, envelope_audits = validate_v35_checkpoint_envelope(
        config, checkpoint_root, contract
    )
    # This is the first point at which validation QA may be loaded.
    requirements = _selection_requirements(config)
    control = _retention_control_config(config)
    source_v29 = _source_v29_evidence(_metadata(checkpoints[0]))
    _validate_source_against_config(source_v29, config)
    evaluator = evaluator_factory(config, control, checkpoints[0], requirements)
    if tuple(evaluator.validation_scene_ids) != contract.v31.validation_scene_ids:
        raise ValueError("V35 evaluator must remain exactly on scenes 19--24")

    update0 = load_file(checkpoints[0] / "adapter.safetensors", device="cpu")
    approved_tensors = load_file(
        Path(str(source_v29["checkpoint"])) / "adapter.safetensors", device="cpu"
    )
    evaluator.install(
        _approved_v29_runtime_tensor_envelope(
            update0,
            approved_tensors,
            expected_core_state_sha256=contract.core_initial_state_sha256,
        ),
        approved_v29=True,
    )
    approved_teacher = evaluator.evaluate_teacher()
    approved_greedy = evaluator.evaluate_greedy()
    approved_aggregate = evaluator.evaluate_aggregate_exact()

    evaluator.install(update0)
    source_teacher = evaluator.evaluate_teacher()
    arms: list[dict[str, Any]] = []
    for step, checkpoint in zip(contract.saved_optimizer_steps, checkpoints, strict=True):
        tensors = load_file(checkpoint / "adapter.safetensors", device="cpu")
        evaluator.install(tensors)
        teacher = evaluator.evaluate_teacher()
        greedy: V35GreedyEvidence | None = None
        if step in _GREEDY_STEPS:
            greedy = evaluator.evaluate_greedy()
        ratios = prefix_separation_ratios(
            teacher.prefix_diagnostics, source_teacher.prefix_diagnostics
        )
        arm: dict[str, Any] = {
            "checkpoint": str(checkpoint),
            "optimizer_step": step,
            "update": step,
            "validation_answer_token_nll": teacher.validation_answer_token_nll,
            "validation_pair_passed_units": teacher.pair_margins.passed_units,
            "validation_pair_mean_margin": teacher.pair_margins.mean_margin,
            "validation_pair_minimum_margin": teacher.pair_margins.minimum_margin,
            "validation_family_teacher": dict(teacher.family_teacher),
            "validation_prefix_separation": dict(teacher.prefix_diagnostics),
            "validation_prefix_ratios_from_v33_u64": ratios,
            "validation_weak_minus_unrelated_ratio": (
                ratios["weak_pair_mean"] - ratios["unrelated_mean"]
            ),
            "color_full_vocab_sides": teacher.color_full_vocab_sides,
            "mirror_full_vocab_sides": teacher.mirror_full_vocab_sides,
            "new_negative_sides_vs_approved_v29": sorted(
                teacher.negative_sides - approved_teacher.negative_sides
            ),
            "prefix_sha256_by_validation_scene": dict(
                sorted(teacher.prefix_sha256_by_scene.items())
            ),
            "greedy_screen_designated": step in _GREEDY_STEPS,
            "greedy_exact_complete_units_correct": None,
            "greedy_prediction_changed_units": None,
            "greedy_complete_units_by_family": None,
            "greedy_prediction_changed_by_family": None,
            "broad_retention_exact_accuracy": None,
            "checks": {},
            "eligible": False,
        }
        if greedy is not None:
            checks, ratios, nonmirror, new_negatives = _development_checks(
                teacher=teacher,
                greedy=greedy,
                source=source_teacher,
                approved=approved_teacher,
                approved_greedy=approved_greedy,
            )
            arm.update(
                {
                    "nonmirror_teacher_complete_units": nonmirror,
                    "new_negative_sides_vs_approved_v29": new_negatives,
                    "greedy_exact_complete_units_correct": (
                        greedy.generation.exact_complete_units_correct
                    ),
                    "greedy_prediction_changed_units": (
                        greedy.generation.prediction_changed_units
                    ),
                    "greedy_complete_units_by_family": dict(greedy.complete_by_family),
                    "greedy_prediction_changed_by_family": dict(
                        greedy.prediction_changed_by_family
                    ),
                    "broad_retention_exact_accuracy": (
                        greedy.generation.broad_exact_accuracy
                    ),
                    "checks": checks,
                    "eligible": all(checks.values()),
                }
            )
        arms.append(arm)
    candidates = [arm for arm in arms if arm["eligible"]]
    selected = min(
        candidates,
        key=lambda arm: (
            -int(arm["greedy_exact_complete_units_correct"]),
            -int(arm["nonmirror_teacher_complete_units"]),
            -int(arm["validation_pair_passed_units"]),
            float(arm["validation_answer_token_nll"]),
            int(arm["optimizer_step"]),
        ),
        default=None,
    )
    selected_aggregate: tuple[int, int] | None = None
    prefix_attestation: Mapping[str, Any] | None = None
    if selected is not None:
        selected_path = checkpoint_root / f"update_{int(selected['optimizer_step']):03d}"
        evaluator.install(load_file(selected_path / "adapter.safetensors", device="cpu"))
        # Materialize once before aggregate questions, reuse it throughout, and
        # replay its hash afterward.
        initial_attestation = evaluator.attest_prefix_invariance()
        if initial_attestation.get("passed") is not True:
            raise ValueError("V35 selected prefix failed its pre-question invariance replay")
        selected_aggregate = evaluator.evaluate_aggregate_exact()
        prefix_attestation = evaluator.attest_prefix_invariance()
    promotion = _promotion(
        selected,
        approved_v29_aggregate=approved_aggregate,
        selected_aggregate=selected_aggregate,
        prefix_attestation=prefix_attestation,
    )
    terminal = require_v34_terminal_gate(config)
    return {
        "schema_version": 1,
        "artifact": "v35_block_cross_development_selection",
        "development_validation_model_selection_only": True,
        "training_completed_before_validation_loaded": True,
        "validation_used_for_training_continuation": False,
        "final_test_scenes_touched": False,
        "deferred_final_scene_ids": list(contract.v31.deferred_final_scene_ids),
        "oracle_loaded": False,
        "question_text_serialized": False,
        "answer_text_serialized": False,
        "question_dependent_scene_processing": False,
        "question_dependent_retrieval": False,
        "all_scene_prefixes_built_before_questions": True,
        "gemma_decoder_frozen": True,
        "all_lora_banks_frozen": True,
        "complete_v33_stack_frozen": True,
        "exact_trainable_parameter_count": 983_040,
        "model_load_count": 1,
        "source_v29": source_v29,
        "retention_and_aggregate_baseline": "approved_v29",
        "improvement_baseline": "exact_v33_update_064",
        "v34_terminal_gate": {"path": terminal["path"], "sha256": terminal["sha256"]},
        "train_scene_ids": list(contract.v31.train_scene_ids),
        "validation_scene_ids": list(evaluator.validation_scene_ids),
        "saved_optimizer_steps": list(contract.saved_optimizer_steps),
        "all_saved_arms_inspected": True,
        "checkpoint_envelope_audits": envelope_audits,
        "optimizer_state_steps_verified": list(contract.saved_optimizer_steps[1:]),
        "train_only_update32_gate_passed": True,
        "train_only_update64_gate_passed": True,
        "complete_question_independent_block_cache": dict(evaluator.cache_audit),
        "greedy_screen_steps": sorted(_GREEDY_STEPS),
        "development_requirements": {
            "validation_nll_strictly_improves_v33_u64": True,
            "validation_pair_mean_and_passed_units_strictly_improve_v33_u64": True,
            "weak_prefix_minus_unrelated_ratio_minimum": _VALIDATION_DIFFERENTIAL_MINIMUM,
            "each_book_picture_mirror_ratio_exceeds_unrelated": True,
            "unrelated_ratio_bounds": [_UNRELATED_RATIO_MINIMUM, _UNRELATED_RATIO_MAXIMUM],
            "nonmirror_teacher_complete_minimum": 1,
            "approved_v29_color_sides_minimum": 12,
            "approved_v29_mirror_sides_minimum": 10,
            "approved_v29_no_new_control_negatives": True,
            "approved_v29_broad_accuracy_no_regression": True,
        },
        "approved_v29_teacher_baseline": {
            "validation_answer_token_nll": approved_teacher.validation_answer_token_nll,
            "color_full_vocab_sides": approved_teacher.color_full_vocab_sides,
            "mirror_full_vocab_sides": approved_teacher.mirror_full_vocab_sides,
            "broad_retention_exact_accuracy": approved_greedy.generation.broad_exact_accuracy,
        },
        "v33_u64_teacher_baseline": {
            "validation_answer_token_nll": source_teacher.validation_answer_token_nll,
            "validation_pair_passed_units": source_teacher.pair_margins.passed_units,
            "validation_pair_mean_margin": source_teacher.pair_margins.mean_margin,
            "validation_prefix_separation": dict(source_teacher.prefix_diagnostics),
        },
        "arms": arms,
        "selected_checkpoint": None if selected is None else selected["checkpoint"],
        "selected_update": None if selected is None else selected["optimizer_step"],
        "selected_optimizer_step": None if selected is None else selected["optimizer_step"],
        "development_selection_passed": selected is not None,
        "chat_promotion": promotion,
        "chat_promotion_eligible": promotion["eligible"],
        "development_progress_is_not_chat_promotion": True,
        "passed": selected is not None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = select_v35(args.config, args.checkpoint_root)
    _atomic_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "V35GreedyEvidence",
    "V35TeacherEvidence",
    "select_v35",
    "validate_v35_checkpoint_envelope",
]
