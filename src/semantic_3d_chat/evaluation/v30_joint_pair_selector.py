"""Select a development-only V30 joint pair-training checkpoint.

V30 is allowed to change exactly two surfaces: the existing post-stack dense
sidecar output and one new Gemma query-LoRA bank.  This selector treats saved
training summaries as claims, recomputes their tensor and pair-margin evidence,
and then applies independent teacher-forced and greedy development gates.

This module is deliberately evaluation-only.  It never reads ``data/oracle``,
never evaluates deferred scenes 25--30, and builds every scene representation
before passing any question to Gemma.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import torch
from safetensors.torch import load_file

from semantic_3d_chat.config import PROJECT_ROOT, load_config, project_path
from semantic_3d_chat.data.dataset import QARecord, SceneQADataset
from semantic_3d_chat.evaluation.metrics import exact_normalized_match, normalize_answer
from semantic_3d_chat.evaluation.scene_signal_audit import _question_logits_and_answer
from semantic_3d_chat.evaluation.v27_sidecar_screen import (
    _atomic_json,
    _full_vocab_counts,
    _negative_sides,
    _pair_role_ids,
)
from semantic_3d_chat.evaluation.v28_stage_a_selector import _teacher_gate
from semantic_3d_chat.evaluation.v28_stage_b_selector import _retention_control_config
from semantic_3d_chat.language.lora import lora_banks_settings, tensor_state_sha256
from semantic_3d_chat.language.prefix_injection import prefix_sha256
from semantic_3d_chat.scene_encoder.map_io import MapTensorData, load_map_tensors
from semantic_3d_chat.training.checkpointing import (
    RUNTIME_METADATA_FILENAME,
    TRAINING_METADATA_FILENAME,
    runtime_checkpoint_metadata,
    validate_runtime_checkpoint_metadata,
)
from semantic_3d_chat.training.pair_curriculum import (
    build_exact_question_pair_units,
    cap_pair_units_per_pair,
    pair_curriculum_settings,
    select_pair_only_records,
)
from semantic_3d_chat.training.train_joint_pair_v30 import (
    adapted_scene_tokens,
    cache_pre_sidecar_scenes,
    load_v30_bundle,
    require_approved_v29_source,
    v30_settings,
    validation_pair_metrics,
)
from semantic_3d_chat.training.train_post_stack_sidecar import _file_sha256

FRESH_BANK_NAME = "extension_v30_joint_pair_query"
FRESH_BANK_PREFIX = f"lora_banks.{FRESH_BANK_NAME}."
FRESH_BANK_PARAMETER_COUNT = 131_072
SIDECAR_PARAMETER_COUNT = 198_144
TOTAL_TRAINABLE_PARAMETER_COUNT = 329_216
FRESH_BANK_INITIAL_STATE_SHA256 = "2b1d89fbb9189ac551bf12905cf94036ebaa84696449b31c2b37b69d478fb70d"
SIDECAR_PARAMETER_NAMES = frozenset(
    {
        "dense_sidecar_adapter.output_projection.weight",
        "dense_sidecar_adapter.channel_gain",
    }
)
FRESH_BANK_TARGET_MODULES = (
    "model.language_model.layers.18.self_attn.q_proj",
    "model.language_model.layers.19.self_attn.q_proj",
    "model.language_model.layers.20.self_attn.q_proj",
    "model.language_model.layers.21.self_attn.q_proj",
)
DEFERRED_FINAL_SCENE_IDS = frozenset(f"scene_{index:06d}" for index in range(25, 31))

_UPDATE_NAME = re.compile(r"update_([0-9]{3})")
_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class SelectionRequirements:
    final_update: int
    color_full_vocab_sides: int
    mirror_full_vocab_sides: int
    validation_pair_unit_count: int
    minimum_pair_margin: float
    minimum_mean_margin_improvement: float
    minimum_passed_unit_improvement: int
    greedy_changed_row_count: int
    minimum_greedy_complete_units_correct: int
    broad_retention_subset_size: int
    promotion_changed_complete_pairs_minimum: int
    promotion_label: str


@dataclass(frozen=True)
class PairMarginEvidence:
    unit_keys: tuple[tuple[str, str], ...]
    margins: tuple[tuple[float, float], ...]
    passed_units: int
    passed_sides: int
    mean_margin: float
    minimum_margin: float


@dataclass(frozen=True)
class GenerationEvidence:
    changed_row_count: int
    changed_unit_count: int
    exact_correct_sides: int
    exact_complete_units_correct: int
    prediction_changed_units: int
    broad_row_count: int
    broad_exact_correct: int

    @property
    def broad_exact_accuracy(self) -> float:
        return self.broad_exact_correct / self.broad_row_count


@dataclass(frozen=True)
class RuntimeArmEvidence:
    color_full_vocab_sides: int
    color_full_vocab_units: int
    mirror_full_vocab_sides: int
    mirror_full_vocab_units: int
    negative_sides: frozenset[tuple[str, str]]
    pair_margins: PairMarginEvidence
    generation: GenerationEvidence
    prefix_sha256_by_scene: Mapping[str, str]


class ArmEvaluator(Protocol):
    """Independent model scorer used after a checkpoint state is installed."""

    validation_scene_ids: tuple[str, ...]

    def install(self, tensors: Mapping[str, torch.Tensor]) -> None: ...

    def evaluate(self) -> RuntimeArmEvidence: ...

    def evaluate_aggregate_exact(self) -> tuple[int, int]: ...


def _resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (PROJECT_ROOT / value).resolve()


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be a mapping")
    return value


def _finite(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _integer(value: Any, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be an integer")
    if value < minimum:
        raise ValueError(f"{field} must be >= {minimum}")
    return value


def _sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


def _checkpoint_paths(root: Path, *, expected_final_update: int) -> list[Path]:
    paths = sorted(
        path
        for path in root.glob("update_*")
        if path.is_dir() and _UPDATE_NAME.fullmatch(path.name)
    )
    observed = [int(_UPDATE_NAME.fullmatch(path.name).group(1)) for path in paths]
    expected = list(range(expected_final_update + 1))
    if observed != expected:
        raise FileNotFoundError(
            f"V30 checkpoints must be complete and contiguous: observed={observed} "
            f"expected={expected}"
        )
    required = ("adapter.safetensors", TRAINING_METADATA_FILENAME, RUNTIME_METADATA_FILENAME)
    for path in paths:
        missing = [name for name in required if not (path / name).is_file()]
        if missing:
            raise FileNotFoundError(f"Incomplete V30 checkpoint {path.name}: {missing}")
    return paths


def _metadata(path: Path) -> dict[str, Any]:
    value = json.loads((path / TRAINING_METADATA_FILENAME).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Checkpoint metadata must be a JSON object: {path}")
    return value


def _validate_runtime_metadata(path: Path, metadata: Mapping[str, Any]) -> None:
    value = json.loads((path / RUNTIME_METADATA_FILENAME).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Runtime metadata must be a JSON object: {path}")
    validate_runtime_checkpoint_metadata(value)
    if value != runtime_checkpoint_metadata(metadata):
        raise ValueError(f"Runtime/training metadata mismatch in {path.name}")


def _fresh_bank_state(tensors: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    state = {
        name.removeprefix(FRESH_BANK_PREFIX): value
        for name, value in tensors.items()
        if name.startswith(FRESH_BANK_PREFIX)
    }
    if not state:
        raise ValueError(f"Checkpoint lacks fresh V30 bank {FRESH_BANK_NAME}")
    return state


def _sidecar_state(tensors: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    prefix = "dense_sidecar_adapter."
    state = {
        name.removeprefix(prefix): value
        for name, value in tensors.items()
        if name.startswith(prefix)
    }
    missing = SIDECAR_PARAMETER_NAMES - {f"{prefix}{name}" for name in state}
    if missing:
        raise ValueError(f"V30 checkpoint lacks authorized sidecar tensors: {sorted(missing)}")
    return state


def _frozen_tensor_sha256(tensors: Mapping[str, torch.Tensor]) -> str:
    frozen = {
        name: value
        for name, value in tensors.items()
        if not name.startswith(FRESH_BANK_PREFIX) and name not in SIDECAR_PARAMETER_NAMES
    }
    if not frozen:
        raise ValueError("V30 checkpoint has no inherited frozen tensors to audit")
    return tensor_state_sha256(frozen)


def _selection_requirements(config: Mapping[str, Any]) -> SelectionRequirements:
    contract = _mapping(config.get("v30_joint_pair"), "v30_joint_pair")
    if contract.get("schema_version") != 1:
        raise ValueError("v30_joint_pair.schema_version must be 1")
    if contract.get("fresh_bank") != FRESH_BANK_NAME:
        raise ValueError("V30 fresh-bank name mismatch")
    if contract.get("fresh_bank_initial_state_sha256") != FRESH_BANK_INITIAL_STATE_SHA256:
        raise ValueError("V30 fresh-bank initialization hash mismatch")
    if contract.get("fresh_bank_parameter_count") != FRESH_BANK_PARAMETER_COUNT:
        raise ValueError("V30 fresh-bank parameter count mismatch")
    configured_sidecar_names = contract.get("sidecar_trainable_parameter_names")
    if (
        not isinstance(configured_sidecar_names, Sequence)
        or isinstance(configured_sidecar_names, (str, bytes))
        or {f"dense_sidecar_adapter.{name}" for name in configured_sidecar_names}
        != SIDECAR_PARAMETER_NAMES
    ):
        raise ValueError("V30 configured sidecar surface mismatch")
    if contract.get("sidecar_trainable_parameter_count") != SIDECAR_PARAMETER_COUNT:
        raise ValueError("V30 configured sidecar parameter count mismatch")
    if contract.get("joint_trainable_parameter_count") != TOTAL_TRAINABLE_PARAMETER_COUNT:
        raise ValueError("V30 configured joint parameter count mismatch")
    requirements = _mapping(contract.get("selection_requires"), "selection_requires")
    promotion = _mapping(contract.get("promotion_requires"), "promotion_requires")
    true_fields = (
        "no_new_negative_sides",
        "source_v29_validation_nll_must_improve",
        "broad_exact_accuracy_no_regression",
    )
    for field in true_fields:
        if requirements.get(field) is not True:
            raise ValueError(f"v30_joint_pair.selection_requires.{field} must be true")
    if promotion.get("aggregate_validation_exact_accuracy_no_regression") is not True:
        raise ValueError("V30 chat promotion must require aggregate exact non-regression")
    promotion_label = promotion.get("label")
    if promotion_label != "chat_promotion_not_merely_development_progress":
        raise ValueError("V30 chat-promotion label mismatch")
    training = _mapping(config.get("training"), "training")
    settings = _mapping(training.get("v30_joint_pair"), "training.v30_joint_pair")
    return SelectionRequirements(
        final_update=_integer(
            settings.get("max_optimizer_steps"), "max_optimizer_steps", minimum=1
        ),
        color_full_vocab_sides=_integer(
            requirements.get("color_full_vocab_sides"), "color_full_vocab_sides", minimum=1
        ),
        mirror_full_vocab_sides=_integer(
            requirements.get("mirror_full_vocab_sides"), "mirror_full_vocab_sides", minimum=1
        ),
        validation_pair_unit_count=_integer(
            requirements.get("validation_pair_unit_count"),
            "validation_pair_unit_count",
            minimum=1,
        ),
        minimum_pair_margin=_finite(requirements.get("minimum_pair_margin"), "minimum_pair_margin"),
        minimum_mean_margin_improvement=_finite(
            requirements.get("minimum_mean_margin_improvement"),
            "minimum_mean_margin_improvement",
        ),
        minimum_passed_unit_improvement=_integer(
            requirements.get("minimum_passed_unit_improvement"),
            "minimum_passed_unit_improvement",
            minimum=1,
        ),
        greedy_changed_row_count=_integer(
            requirements.get("greedy_changed_row_count"),
            "greedy_changed_row_count",
            minimum=2,
        ),
        minimum_greedy_complete_units_correct=_integer(
            requirements.get("minimum_greedy_complete_units_correct"),
            "minimum_greedy_complete_units_correct",
            minimum=1,
        ),
        broad_retention_subset_size=_integer(
            requirements.get("broad_retention_subset_size"),
            "broad_retention_subset_size",
            minimum=1,
        ),
        promotion_changed_complete_pairs_minimum=_integer(
            promotion.get("validation_changed_complete_pairs_minimum"),
            "validation_changed_complete_pairs_minimum",
            minimum=1,
        ),
        promotion_label=promotion_label,
    )


def _validate_trainable_surface(
    metadata: Mapping[str, Any], tensors: Mapping[str, torch.Tensor]
) -> dict[str, Any]:
    contract = _mapping(metadata.get("v30_joint_pair"), "metadata.v30_joint_pair")
    surface = _mapping(contract.get("trainable_surface"), "trainable_surface")
    names = surface.get("sidecar_parameter_names")
    if not isinstance(names, Sequence) or isinstance(names, (str, bytes)):
        raise TypeError("sidecar_parameter_names must be a sequence")
    if frozenset(str(name) for name in names) != SIDECAR_PARAMETER_NAMES:
        raise ValueError("V30 authorized sidecar parameter names mismatch")
    if surface.get("sidecar_parameter_count") != SIDECAR_PARAMETER_COUNT:
        raise ValueError("V30 sidecar parameter count mismatch")
    if surface.get("fresh_bank") != FRESH_BANK_NAME:
        raise ValueError("V30 trainable surface fresh-bank name mismatch")
    if surface.get("fresh_bank_parameter_count") != FRESH_BANK_PARAMETER_COUNT:
        raise ValueError("V30 trainable surface fresh-bank count mismatch")
    targets = surface.get("fresh_bank_target_modules")
    if not isinstance(targets, Sequence) or isinstance(targets, (str, bytes)):
        raise TypeError("fresh-bank target_modules must be a sequence")
    if tuple(str(target) for target in targets) != FRESH_BANK_TARGET_MODULES:
        raise ValueError("V30 fresh-bank targets mismatch")
    if surface.get("total_parameter_count") != TOTAL_TRAINABLE_PARAMETER_COUNT:
        raise ValueError("V30 total trainable parameter count mismatch")
    if surface.get("every_other_parameter_frozen") is not True:
        raise ValueError("V30 metadata does not freeze every other parameter")

    fresh = _fresh_bank_state(tensors)
    parameter_names = surface.get("fresh_bank_parameter_names")
    if not isinstance(parameter_names, Sequence) or isinstance(parameter_names, (str, bytes)):
        raise TypeError("fresh_bank_parameter_names must be a sequence")
    expected_fresh_names = {f"{FRESH_BANK_PREFIX}{name}" for name in fresh}
    if {str(name) for name in parameter_names} != expected_fresh_names:
        raise ValueError("V30 fresh-bank parameter names mismatch checkpoint tensors")
    fresh_count = sum(int(value.numel()) for value in fresh.values())
    sidecar = _sidecar_state(tensors)
    sidecar_count = sum(int(tensors[name].numel()) for name in SIDECAR_PARAMETER_NAMES)
    if fresh_count != FRESH_BANK_PARAMETER_COUNT or sidecar_count != SIDECAR_PARAMETER_COUNT:
        raise ValueError(
            "V30 checkpoint tensor counts disagree with authorized surface: "
            f"fresh={fresh_count} sidecar={sidecar_count}"
        )
    fresh_hash = tensor_state_sha256(fresh)
    bank_hashes = _mapping(metadata.get("lora_bank_state_sha256"), "lora_bank_state_sha256")
    if bank_hashes.get(FRESH_BANK_NAME) != fresh_hash:
        raise ValueError("V30 fresh-bank tensor hash mismatch")
    return {
        "fresh_bank_state": fresh,
        "fresh_bank_state_sha256": fresh_hash,
        "sidecar_state": sidecar,
        "fresh_bank_parameter_count": fresh_count,
        "sidecar_parameter_count": sidecar_count,
        "total_parameter_count": fresh_count + sidecar_count,
    }


def _source_v29_evidence(metadata: Mapping[str, Any]) -> dict[str, Any]:
    contract = _mapping(metadata.get("v30_joint_pair"), "metadata.v30_joint_pair")
    report_path = _resolve(str(contract.get("source_v29_selection_report")))
    report_hash = _sha256(
        contract.get("source_v29_selection_report_sha256"),
        "source_v29_selection_report_sha256",
    )
    if not report_path.is_file() or _file_sha256(report_path) != report_hash:
        raise ValueError("Approved V29 selection report is missing or hash-mismatched")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    required_report = {
        "schema_version": 1,
        "artifact": "v28_post_stack_decoder_stage_b_selection",
        "training_evaluation_only": True,
        "question_text_serialized": False,
        "answer_text_serialized": False,
        "oracle_loaded": False,
        "question_dependent_scene_processing": False,
        "question_dependent_retrieval": False,
        "passed": True,
    }
    if not isinstance(report, Mapping) or any(
        report.get(key) != value for key, value in required_report.items()
    ):
        raise ValueError("V30 source V29 selection report did not pass")
    selected_update = _integer(
        contract.get("source_v29_selected_update"),
        "source_v29_selected_update",
        minimum=1,
    )
    if report.get("selected_update") != selected_update:
        raise ValueError("V30 metadata does not descend from selected V29 update")
    source_checkpoint = _resolve(str(contract.get("source_v29_checkpoint")))
    selected_checkpoint = _resolve(str(report.get("selected_checkpoint")))
    if source_checkpoint != selected_checkpoint:
        raise ValueError("V30 source checkpoint differs from approved V29 selection")
    adapter_hash = _sha256(contract.get("source_v29_adapter_sha256"), "source_v29_adapter_sha256")
    runtime_hash = _sha256(
        contract.get("source_v29_runtime_metadata_sha256"),
        "source_v29_runtime_metadata_sha256",
    )
    if _file_sha256(source_checkpoint / "adapter.safetensors") != adapter_hash:
        raise ValueError("V30 source V29 adapter hash mismatch")
    if _file_sha256(source_checkpoint / RUNTIME_METADATA_FILENAME) != runtime_hash:
        raise ValueError("V30 source V29 runtime metadata hash mismatch")
    selected_arms = [
        arm
        for arm in report.get("arms", [])
        if isinstance(arm, Mapping) and arm.get("update") == selected_update
    ]
    if len(selected_arms) != 1 or selected_arms[0].get("eligible") is not True:
        raise ValueError("Approved V29 selected arm is absent or ineligible")
    source_nll = _finite(
        selected_arms[0].get("validation_answer_token_nll"),
        "V29 selected validation_answer_token_nll",
    )
    source_training = _metadata(source_checkpoint)
    _validate_runtime_metadata(source_checkpoint, source_training)
    if not math.isclose(_validation_nll(source_training), source_nll, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("Approved V29 report and checkpoint validation NLL differ")
    return {
        "checkpoint": str(source_checkpoint),
        "selected_update": selected_update,
        "selection_report": str(report_path),
        "selection_report_sha256": report_hash,
        "adapter_sha256": adapter_hash,
        "runtime_metadata_sha256": runtime_hash,
        "validation_answer_token_nll": source_nll,
    }


def _validate_source_against_config(source: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    contract = _mapping(config.get("v30_joint_pair"), "v30_joint_pair")
    expected = {
        "selection_report": str(_resolve(str(contract.get("source_selection_report")))),
        "selection_report_sha256": contract.get("source_selection_report_sha256"),
        "selected_update": contract.get("source_selected_update"),
        "adapter_sha256": contract.get("source_adapter_sha256"),
        "runtime_metadata_sha256": contract.get("source_runtime_metadata_sha256"),
    }
    mismatches = {
        key: {"observed": source.get(key), "expected": value}
        for key, value in expected.items()
        if source.get(key) != value
    }
    root = _resolve(str(contract.get("source_checkpoint_root")))
    checkpoint = _resolve(str(source.get("checkpoint")))
    selected_update = source.get("selected_update")
    if (
        not isinstance(selected_update, int)
        or not checkpoint.is_relative_to(root)
        or checkpoint.name != f"update_{selected_update:03d}"
    ):
        mismatches["checkpoint"] = {
            "observed": str(checkpoint),
            "expected_root": str(root),
        }
    if mismatches:
        raise ValueError(f"V30 metadata source differs from pinned config: {mismatches}")


def _validate_no_leakage_or_final_scenes(metadata: Mapping[str, Any]) -> None:
    contract = _mapping(metadata.get("v30_joint_pair"), "metadata.v30_joint_pair")
    scene_cache = _mapping(contract.get("scene_cache"), "v30_joint_pair.scene_cache")
    required_cache = {
        "all_voxels_covered": True,
        "question_inputs_to_scene_cache": False,
        "question_dependent_scene_processing": False,
        "question_dependent_retrieval": False,
        "oracle_environment_files_loaded": False,
        "exact_source_scene_prefixes": True,
    }
    if any(scene_cache.get(key) != value for key, value in required_cache.items()):
        raise ValueError("V30 scene-cache leakage/completeness contract failed")
    loaded = scene_cache.get("loaded_environment_files", [])
    if not isinstance(loaded, Sequence) or isinstance(loaded, (str, bytes)):
        raise TypeError("loaded_environment_files must be a sequence")
    lowered = [str(path).casefold() for path in loaded]
    if any("oracle" in Path(path).parts for path in lowered):
        raise ValueError("V30 loaded an oracle environment file")
    if any(scene_id in path for path in lowered for scene_id in DEFERRED_FINAL_SCENE_IDS):
        raise ValueError("V30 loaded a deferred final-test scene")

    qa = _mapping(contract.get("qa_dataset"), "v30_joint_pair.qa_dataset")
    deferred_loaded = qa.get("deferred_test_scene_ids_loaded", [])
    if deferred_loaded != [] or contract.get("final_test_scene_ids_loaded") != []:
        raise ValueError("V30 loaded deferred final-test QA")
    top_level_required = {
        "oracle_environment_files_loaded": False,
        "question_dependent_scene_processing": False,
        "question_dependent_retrieval": False,
        "development_validation_model_selection_only": True,
    }
    if any(contract.get(key) != value for key, value in top_level_required.items()):
        raise ValueError("V30 top-level leakage/development contract failed")
    for field in ("train_scene_ids", "validation_scene_ids"):
        values = contract.get(field)
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise TypeError(f"v30_joint_pair.{field} must be a sequence")
        overlap = {str(value) for value in values} & DEFERRED_FINAL_SCENE_IDS
        if overlap:
            raise ValueError(f"V30 {field} contains deferred final scenes: {sorted(overlap)}")


def _validate_update_zero(
    metadata: Mapping[str, Any],
    audit: Mapping[str, Any],
    source: Mapping[str, Any],
    *,
    expected_nll_tolerance: float,
) -> None:
    if metadata.get("optimizer_step") != 0:
        raise ValueError("V30 update_000 must have optimizer_step 0")
    contract = _mapping(metadata.get("v30_joint_pair"), "metadata.v30_joint_pair")
    equivalence = _mapping(contract.get("update_zero_equivalence"), "update_zero_equivalence")
    required = {
        "approved_v29_source": True,
        "fresh_bank_exact_zero_output": True,
        "exact_source_scene_prefixes": True,
        "exact_source_validation_nll": True,
        "question_dependent_scene_processing": False,
        "question_dependent_retrieval": False,
        "oracle_environment_files_loaded": False,
    }
    if any(equivalence.get(key) != value for key, value in required.items()):
        raise ValueError("V30 update-zero equivalence contract is incomplete")
    exact_targets = equivalence.get("target_outputs_bit_exact")
    if (
        not isinstance(exact_targets, Mapping)
        or {str(name) for name in exact_targets} != set(FRESH_BANK_TARGET_MODULES)
        or any(value is not True for value in exact_targets.values())
    ):
        raise ValueError("V30 update-zero target outputs are not all bit-exact")
    if audit["fresh_bank_state_sha256"] != FRESH_BANK_INITIAL_STATE_SHA256:
        raise ValueError("V30 update-zero fresh-bank initial hash mismatch")
    fresh = _mapping(audit.get("fresh_bank_state"), "fresh_bank_state")
    b_tensors = [value for name, value in fresh.items() if name.endswith(".lora_b")]
    if not b_tensors or any(torch.count_nonzero(value).item() for value in b_tensors):
        raise ValueError("V30 update-zero fresh bank is not exact-zero output")
    if str(contract.get("source_v29_checkpoint")) != source["checkpoint"]:
        raise ValueError("V30 update-zero source provenance changed during validation")
    recorded_tolerance = _finite(
        equivalence.get("validation_nll_absolute_tolerance"),
        "update_zero_equivalence.validation_nll_absolute_tolerance",
    )
    if recorded_tolerance < 0 or recorded_tolerance != expected_nll_tolerance:
        raise ValueError("V30 update-zero validation-NLL tolerance mismatch")
    source_nll = _finite(
        equivalence.get("source_validation_answer_token_nll"),
        "update_zero_equivalence.source_validation_answer_token_nll",
    )
    observed_nll = _finite(
        equivalence.get("observed_validation_answer_token_nll"),
        "update_zero_equivalence.observed_validation_answer_token_nll",
    )
    independently_loaded_source_nll = _finite(
        source.get("validation_answer_token_nll"), "approved V29 validation NLL"
    )
    if (
        abs(source_nll - independently_loaded_source_nll) > recorded_tolerance
        or abs(observed_nll - independently_loaded_source_nll) > recorded_tolerance
    ):
        raise ValueError("V30 update-zero validation NLL is not equivalent to approved V29")


def _validation_nll(metadata: Mapping[str, Any]) -> float:
    history = metadata.get("history")
    if not isinstance(history, Sequence) or isinstance(history, (str, bytes)) or not history:
        raise ValueError("V30 checkpoint lacks training history")
    row = _mapping(history[-1], "history[-1]")
    return _finite(row.get("validation_answer_token_nll"), "validation_answer_token_nll")


def _pair_margin_evidence(
    metadata: Mapping[str, Any], *, expected_unit_count: int
) -> PairMarginEvidence:
    history = metadata.get("history")
    if not isinstance(history, Sequence) or isinstance(history, (str, bytes)) or not history:
        raise ValueError("V30 checkpoint lacks training history")
    metrics = _mapping(
        _mapping(history[-1], "history[-1]").get("validation_pair_metrics"),
        "validation_pair_metrics",
    )
    rows = metrics.get("margins_by_unit")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise TypeError("validation pair margins_by_unit must be a sequence")
    if len(rows) != expected_unit_count:
        raise ValueError(
            f"V30 validation pair-unit count mismatch: {len(rows)} != {expected_unit_count}"
        )
    parsed: list[tuple[tuple[str, str], tuple[float, float]]] = []
    seen: set[tuple[str, str]] = set()
    for index, raw in enumerate(rows):
        row = _mapping(raw, f"margins_by_unit[{index}]")
        pair_id = row.get("pair_id")
        question_key = row.get("question_key")
        if not isinstance(pair_id, str) or not pair_id:
            raise ValueError("Pair margin row lacks pair_id")
        if not isinstance(question_key, str) or not question_key:
            raise ValueError("Pair margin row lacks question_key")
        key = (pair_id, question_key)
        if key in seen:
            raise ValueError(f"Duplicate validation pair margin unit: {key}")
        seen.add(key)
        scene_ids = row.get("scene_ids")
        if (
            not isinstance(scene_ids, Sequence)
            or isinstance(scene_ids, (str, bytes))
            or len(scene_ids) != 2
            or len({str(value) for value in scene_ids}) != 2
        ):
            raise ValueError(f"Pair margin unit {key} must have two distinct scenes")
        if {str(value) for value in scene_ids} & DEFERRED_FINAL_SCENE_IDS:
            raise ValueError(f"Pair margin unit {key} contains a deferred final scene")
        raw_margins = row.get("margins")
        if (
            not isinstance(raw_margins, Sequence)
            or isinstance(raw_margins, (str, bytes))
            or len(raw_margins) != 2
        ):
            raise ValueError(f"Pair margin unit {key} must have two margins")
        margins = (
            _finite(raw_margins[0], f"{key}.margin[0]"),
            _finite(raw_margins[1], f"{key}.margin[1]"),
        )
        parsed.append((key, margins))
    parsed.sort(key=lambda item: item[0])
    flat = [value for _, margins in parsed for value in margins]
    passed_units = sum(all(value > 0.0 for value in margins) for _, margins in parsed)
    passed_sides = sum(value > 0.0 for value in flat)
    mean_margin = sum(flat) / len(flat)
    minimum_margin = min(flat)
    expected = {
        "unit_count": expected_unit_count,
        "side_count": 2 * expected_unit_count,
        "passed_units": passed_units,
    }
    for field, value in expected.items():
        if metrics.get(field) != value:
            raise ValueError(f"Recorded validation pair {field} disagrees with raw margins")
    numeric_expected = {
        "side_accuracy": passed_sides / len(flat),
        "unit_accuracy": passed_units / expected_unit_count,
        "mean_margin": mean_margin,
        "minimum_margin": minimum_margin,
    }
    for field, value in numeric_expected.items():
        if not math.isclose(
            _finite(metrics.get(field), f"validation_pair_metrics.{field}"),
            value,
            rel_tol=0.0,
            abs_tol=1e-5,
        ):
            raise ValueError(f"Recorded validation pair {field} disagrees with raw margins")
    return PairMarginEvidence(
        unit_keys=tuple(key for key, _ in parsed),
        margins=tuple(margins for _, margins in parsed),
        passed_units=passed_units,
        passed_sides=passed_sides,
        mean_margin=mean_margin,
        minimum_margin=minimum_margin,
    )


def _compare_pair_evidence(
    recorded: PairMarginEvidence, observed: PairMarginEvidence, *, tolerance: float = 1e-5
) -> None:
    if recorded.unit_keys != observed.unit_keys:
        raise ValueError("Runtime validation pair units differ from recorded training units")
    for recorded_pair, observed_pair in zip(recorded.margins, observed.margins, strict=True):
        for saved, rerun in zip(recorded_pair, observed_pair, strict=True):
            if not math.isclose(saved, rerun, rel_tol=0.0, abs_tol=tolerance):
                raise ValueError("Runtime validation pair margins differ from checkpoint evidence")


def _generation_evidence(
    changed_rows: Sequence[Mapping[str, Any]],
    broad_rows: Sequence[Mapping[str, Any]],
    *,
    expected_changed_rows: int,
    expected_broad_rows: int,
) -> GenerationEvidence:
    if len(changed_rows) != expected_changed_rows:
        raise ValueError(
            f"Greedy changed-row count mismatch: {len(changed_rows)} != {expected_changed_rows}"
        )
    grouped: defaultdict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in changed_rows:
        pair_id = row.get("pair_id")
        question_key = row.get("question_key")
        scene_id = row.get("scene_id")
        if not all(isinstance(value, str) and value for value in (pair_id, question_key, scene_id)):
            raise ValueError("Greedy changed rows require opaque pair/question/scene IDs")
        if scene_id in DEFERRED_FINAL_SCENE_IDS:
            raise ValueError("Greedy generation touched a deferred final scene")
        grouped[(str(pair_id), str(question_key))].append(row)
    if any(len(rows) != 2 for rows in grouped.values()):
        raise ValueError("Greedy generation must contain two rows for every atomic unit")
    if any(len({str(row["scene_id"]) for row in rows}) != 2 for rows in grouped.values()):
        raise ValueError("Greedy atomic unit must contain two distinct scenes")
    exact_sides = 0
    exact_units = 0
    changed_units = 0
    for rows in grouped.values():
        correctness = [
            exact_normalized_match(row.get("prediction"), row.get("target")) for row in rows
        ]
        exact_sides += sum(correctness)
        exact_units += int(all(correctness))
        changed_units += int(
            normalize_answer(rows[0].get("prediction"))
            != normalize_answer(rows[1].get("prediction"))
        )
    if len(broad_rows) != expected_broad_rows:
        raise ValueError(
            f"Broad-retention row count mismatch: {len(broad_rows)} != {expected_broad_rows}"
        )
    broad_keys: set[tuple[str, str]] = set()
    broad_correct = 0
    for row in broad_rows:
        scene_id = row.get("scene_id")
        question_id = row.get("question_id")
        if not isinstance(scene_id, str) or not isinstance(question_id, str):
            raise TypeError("Broad-retention rows require scene_id and question_id")
        if scene_id in DEFERRED_FINAL_SCENE_IDS:
            raise ValueError("Broad retention touched a deferred final scene")
        key = (scene_id, question_id)
        if key in broad_keys:
            raise ValueError(f"Duplicate broad-retention row: {key}")
        broad_keys.add(key)
        broad_correct += exact_normalized_match(row.get("prediction"), row.get("target"))
    return GenerationEvidence(
        changed_row_count=len(changed_rows),
        changed_unit_count=len(grouped),
        exact_correct_sides=exact_sides,
        exact_complete_units_correct=exact_units,
        prediction_changed_units=changed_units,
        broad_row_count=len(broad_rows),
        broad_exact_correct=broad_correct,
    )


def _select_eligible_arm(arms: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    return min(
        (arm for arm in arms if arm.get("eligible") is True),
        key=lambda arm: (
            -int(arm["greedy_exact_complete_units_correct"]),
            -int(arm["validation_pair_passed_units"]),
            float(arm["validation_answer_token_nll"]),
            int(arm["update"]),
        ),
        default=None,
    )


def _deterministic_broad_subset(
    records: Sequence[QARecord], *, size: int, seed: int
) -> list[QARecord]:
    if len(records) < size:
        raise ValueError(f"Broad-retention subset needs {size} rows; only {len(records)} exist")
    return sorted(
        records,
        key=lambda record: (
            hashlib.sha256(
                f"v30-broad:{seed}:{record.scene_id}:{record.question_id}".encode()
            ).digest(),
            record.scene_id,
            record.question_id,
        ),
    )[:size]


class _RuntimeEvaluator:
    """One-model-load evaluator for old controls and diverse validation."""

    def __init__(
        self,
        config: dict[str, Any],
        control_config: dict[str, Any],
        _checkpoint: Path,
        requirements: SelectionRequirements,
    ) -> None:
        validation_records = list(
            SceneQADataset(project_path(config, "qa", "validation.jsonl")).records
        )
        self.validation_records = validation_records
        self.validation_units = build_exact_question_pair_units(validation_records)
        if len(self.validation_units) != requirements.validation_pair_unit_count:
            raise ValueError(
                "V30 selector requires every expected-change validation unit: "
                f"{len(self.validation_units)} != {requirements.validation_pair_unit_count}"
            )
        self.validation_scene_ids = tuple(
            sorted({scene_id for unit in self.validation_units for scene_id in unit.scene_ids})
        )
        if set(self.validation_scene_ids) & DEFERRED_FINAL_SCENE_IDS:
            raise ValueError("V30 validation units include deferred final scenes")
        self.broad_records = _deterministic_broad_subset(
            validation_records,
            size=requirements.broad_retention_subset_size,
            seed=int(config["seed"]),
        )

        control_records = list(
            SceneQADataset(project_path(control_config, "qa", "train.jsonl")).records
        )
        control_curriculum = pair_curriculum_settings(control_config)

        control_records = select_pair_only_records(
            control_records, control_curriculum.pair_only_scene_ids
        )
        control_records = cap_pair_units_per_pair(
            control_records,
            control_curriculum.max_units_per_pair,
            seed=int(control_config["seed"]),
        )
        self.control_units = build_exact_question_pair_units(control_records)
        if len(self.control_units) != 12:
            raise ValueError(
                f"V30 old retention controls require 12 units; got {len(self.control_units)}"
            )
        self.control_scene_ids = tuple(
            sorted({scene_id for unit in self.control_units for scene_id in unit.scene_ids})
        )
        self.config = config
        self.control_config = control_config
        self.requirements = requirements
        source = require_approved_v29_source(config)
        self.bundle = load_v30_bundle(config, source)
        settings = lora_banks_settings(config).bank(FRESH_BANK_NAME).adapter
        if settings.rank != 8 or settings.alpha != 16.0:
            raise ValueError("V30 runtime fresh bank must be rank 8, alpha 16")
        if settings.target_modules != FRESH_BANK_TARGET_MODULES:
            raise ValueError("V30 runtime fresh bank targets mismatch")
        fresh_bank = self.bundle.lora_installation.bank(FRESH_BANK_NAME).installation
        self.bank_state = fresh_bank.state_module
        self._prediction_cache: dict[tuple[str, str], str] = {}
        self.validation_caches, cache_audit = cache_pre_sidecar_scenes(
            self.bundle, self.validation_scene_ids
        )
        if (
            cache_audit.get("all_voxels_covered") is not True
            or cache_audit.get("question_inputs_to_scene_cache") is not False
        ):
            raise ValueError("V30 selector validation cache is incomplete or question-dependent")
        self.maps: dict[str, MapTensorData] = {
            scene_id: load_map_tensors(
                project_path(config, "maps", scene_id, "voxel_map.npz"),
                config["scene"]["room_size_m"],
                self.bundle.language.device,
                input_voxel_size_m=config["scene_encoder"].get("input_voxel_size_m"),
            )
            for scene_id in self.control_scene_ids
        }

    def install(self, tensors: Mapping[str, torch.Tensor]) -> None:
        self.bank_state.load_state_dict(_fresh_bank_state(tensors), strict=True)
        self.bundle.dense_sidecar_adapter.load_state_dict(_sidecar_state(tensors), strict=True)
        self._prediction_cache.clear()

    def _prefixes(self) -> dict[str, torch.Tensor]:
        model_dtype = next(self.bundle.language.model.parameters()).dtype
        result: dict[str, torch.Tensor] = {}
        with torch.inference_mode():
            for scene_id in self.validation_scene_ids:
                tokens = adapted_scene_tokens(self.validation_caches[scene_id], self.bundle)
                result[scene_id] = self.bundle.composer.scene_prefix(tokens.to(model_dtype))
        return result

    def _generation_rows(
        self, prefixes: Mapping[str, torch.Tensor]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        changed: list[dict[str, Any]] = []
        for unit in self.validation_units:
            for record in unit.records:
                prediction = self._prediction(record, prefixes)
                changed.append(
                    {
                        "pair_id": unit.pair_id,
                        "question_key": unit.question_key,
                        "scene_id": record.scene_id,
                        "prediction": prediction,
                        "target": record.answer,
                    }
                )
        broad: list[dict[str, Any]] = []
        for record in self.broad_records:
            prediction = self._prediction(record, prefixes)
            broad.append(
                {
                    "scene_id": record.scene_id,
                    "question_id": record.question_id,
                    "prediction": prediction,
                    "target": record.answer,
                }
            )
        return changed, broad

    def _prediction(self, record: QARecord, prefixes: Mapping[str, torch.Tensor]) -> str:
        key = (record.scene_id, record.question_id)
        cached = self._prediction_cache.get(key)
        if cached is not None:
            return cached
        _, prediction, _ = _question_logits_and_answer(
            self.bundle.language,
            prefixes[record.scene_id],
            self.config,
            record.question,
        )
        self._prediction_cache[key] = prediction
        return prediction

    def evaluate_aggregate_exact(self) -> tuple[int, int]:
        """Greedily score every development-validation row, never final test."""

        prefixes = self._prefixes()
        correct = sum(
            exact_normalized_match(
                self._prediction(record, prefixes),
                record.answer,
            )
            for record in self.validation_records
        )
        return len(self.validation_records), correct

    def evaluate(self) -> RuntimeArmEvidence:
        control_gate = _teacher_gate(
            runtime=self.bundle,
            units=self.control_units,
            maps={scene_id: self.maps[scene_id] for scene_id in self.control_scene_ids},
            config=self.control_config,
        )
        color_id, mirror_id = _pair_role_ids(self.control_config)
        color_sides, color_units = _full_vocab_counts(control_gate["by_pair"][color_id])
        mirror_sides, mirror_units = _full_vocab_counts(control_gate["by_pair"][mirror_id])
        raw_pair_metrics = validation_pair_metrics(
            units=self.validation_units,
            caches=self.validation_caches,
            bundle=self.bundle,
            margin=v30_settings(self.config).pair_margin,
        )
        pair_margins = _pair_margin_evidence(
            {"history": [{"validation_pair_metrics": raw_pair_metrics}]},
            expected_unit_count=self.requirements.validation_pair_unit_count,
        )
        prefixes = self._prefixes()
        prefix_hashes = {scene_id: prefix_sha256(prefix) for scene_id, prefix in prefixes.items()}
        changed, broad = self._generation_rows(prefixes)
        generation = _generation_evidence(
            changed,
            broad,
            expected_changed_rows=self.requirements.greedy_changed_row_count,
            expected_broad_rows=self.requirements.broad_retention_subset_size,
        )
        return RuntimeArmEvidence(
            color_full_vocab_sides=color_sides,
            color_full_vocab_units=color_units,
            mirror_full_vocab_sides=mirror_sides,
            mirror_full_vocab_units=mirror_units,
            negative_sides=frozenset(_negative_sides(control_gate)),
            pair_margins=pair_margins,
            generation=generation,
            prefix_sha256_by_scene=prefix_hashes,
        )


def select_joint_pair(
    config_path: Path,
    checkpoint_root: Path,
    *,
    evaluator_factory: Callable[
        [dict[str, Any], dict[str, Any], Path, SelectionRequirements], ArmEvaluator
    ] = _RuntimeEvaluator,
) -> dict[str, Any]:
    config = load_config(config_path)
    requirements = _selection_requirements(config)
    checkpoints = _checkpoint_paths(
        checkpoint_root, expected_final_update=requirements.final_update
    )
    control_config = _retention_control_config(config)
    first_metadata = _metadata(checkpoints[0])
    source = _source_v29_evidence(first_metadata)
    _validate_source_against_config(source, config)
    evaluator = evaluator_factory(config, control_config, checkpoints[0], requirements)

    arms: list[dict[str, Any]] = []
    frozen_hash: str | None = None
    baseline_negatives: frozenset[tuple[str, str]] | None = None
    baseline_pair: PairMarginEvidence | None = None
    baseline_broad_accuracy: float | None = None
    exact_source_contract: Mapping[str, Any] | None = None
    for index, checkpoint in enumerate(checkpoints):
        metadata = _metadata(checkpoint)
        _validate_runtime_metadata(checkpoint, metadata)
        _validate_no_leakage_or_final_scenes(metadata)
        tensors = load_file(checkpoint / "adapter.safetensors", device="cpu")
        audit = _validate_trainable_surface(metadata, tensors)
        observed_frozen = _frozen_tensor_sha256(tensors)
        contract = _mapping(metadata.get("v30_joint_pair"), "metadata.v30_joint_pair")
        if contract.get("frozen_inherited_state_sha256") != observed_frozen:
            raise ValueError(f"Frozen inherited metadata hash mismatch in {checkpoint.name}")
        if frozen_hash is None:
            frozen_hash = observed_frozen
        elif frozen_hash != observed_frozen:
            raise RuntimeError(f"Inherited frozen tensors changed in {checkpoint.name}")
        if index == 0:
            _validate_update_zero(
                metadata,
                audit,
                source,
                expected_nll_tolerance=_finite(
                    _mapping(config.get("v30_joint_pair"), "v30_joint_pair").get(
                        "update_zero_validation_nll_absolute_tolerance"
                    ),
                    "update_zero_validation_nll_absolute_tolerance",
                ),
            )
            exact_source_contract = {
                key: contract.get(key)
                for key in (
                    "source_v29_checkpoint",
                    "source_v29_adapter_sha256",
                    "source_v29_runtime_metadata_sha256",
                    "source_v29_selection_report",
                    "source_v29_selection_report_sha256",
                    "source_v29_selected_update",
                )
            }
        else:
            assert exact_source_contract is not None
            if any(contract.get(key) != value for key, value in exact_source_contract.items()):
                raise ValueError(f"V29 source provenance changed in {checkpoint.name}")
        update = metadata.get("optimizer_step")
        if update != index:
            raise ValueError(f"Checkpoint/update mismatch in {checkpoint.name}")

        recorded_pair = _pair_margin_evidence(
            metadata, expected_unit_count=requirements.validation_pair_unit_count
        )
        evaluator.install(tensors)
        observed = evaluator.evaluate()
        _compare_pair_evidence(recorded_pair, observed.pair_margins)
        if baseline_negatives is None:
            baseline_negatives = observed.negative_sides
        if baseline_pair is None:
            baseline_pair = observed.pair_margins
        if baseline_broad_accuracy is None:
            baseline_broad_accuracy = observed.generation.broad_exact_accuracy
        new_negatives = sorted(observed.negative_sides - baseline_negatives)
        validation_nll = _validation_nll(metadata)
        mean_delta = observed.pair_margins.mean_margin - baseline_pair.mean_margin
        passed_delta = observed.pair_margins.passed_units - baseline_pair.passed_units
        checks = {
            "color_retained": (
                observed.color_full_vocab_sides >= requirements.color_full_vocab_sides
            ),
            "mirror_retained": (
                observed.mirror_full_vocab_sides >= requirements.mirror_full_vocab_sides
            ),
            "no_new_negative_sides": not new_negatives,
            "below_selected_v29_source_nll": (
                validation_nll < float(source["validation_answer_token_nll"])
            ),
            "pair_mean_margin_strictly_improved": (
                mean_delta > 0.0 and mean_delta >= requirements.minimum_mean_margin_improvement
            ),
            "pair_passed_units_improved": (
                passed_delta >= requirements.minimum_passed_unit_improvement
            ),
            "pair_minimum_margin_met": (
                observed.pair_margins.minimum_margin >= requirements.minimum_pair_margin
            ),
            "greedy_changed_units_demonstrated": (
                observed.generation.exact_complete_units_correct
                >= requirements.minimum_greedy_complete_units_correct
            ),
            "broad_exact_accuracy_retained": (
                observed.generation.broad_exact_accuracy >= baseline_broad_accuracy
            ),
        }
        arms.append(
            {
                "checkpoint": str(checkpoint),
                "update": index,
                "fresh_bank_state_sha256": audit["fresh_bank_state_sha256"],
                "frozen_inherited_state_sha256": observed_frozen,
                "validation_answer_token_nll": validation_nll,
                "validation_pair_passed_units": observed.pair_margins.passed_units,
                "validation_pair_passed_sides": observed.pair_margins.passed_sides,
                "validation_pair_mean_margin": observed.pair_margins.mean_margin,
                "validation_pair_minimum_margin": observed.pair_margins.minimum_margin,
                "validation_pair_mean_margin_delta_from_update0": mean_delta,
                "validation_pair_passed_unit_delta_from_update0": passed_delta,
                "color_full_vocab_sides": observed.color_full_vocab_sides,
                "color_full_vocab_units": observed.color_full_vocab_units,
                "mirror_full_vocab_sides": observed.mirror_full_vocab_sides,
                "mirror_full_vocab_units": observed.mirror_full_vocab_units,
                "new_negative_sides": new_negatives,
                "greedy_changed_row_count": observed.generation.changed_row_count,
                "greedy_changed_unit_count": observed.generation.changed_unit_count,
                "greedy_exact_correct_sides": observed.generation.exact_correct_sides,
                "greedy_exact_complete_units_correct": (
                    observed.generation.exact_complete_units_correct
                ),
                "greedy_prediction_changed_units": (observed.generation.prediction_changed_units),
                "broad_retention_exact_correct": observed.generation.broad_exact_correct,
                "broad_retention_row_count": observed.generation.broad_row_count,
                "broad_retention_exact_accuracy": (observed.generation.broad_exact_accuracy),
                "prefix_sha256_by_validation_scene": dict(
                    sorted(observed.prefix_sha256_by_scene.items())
                ),
                "checks": checks,
                "eligible": index > 0 and all(checks.values()),
            }
        )

    selected = _select_eligible_arm(arms)
    promotion_evaluator = getattr(evaluator, "evaluate_aggregate_exact", None)
    promotion: dict[str, Any] = {
        "label": requirements.promotion_label,
        "evaluated": False,
        "validation_changed_complete_pairs_minimum": (
            requirements.promotion_changed_complete_pairs_minimum
        ),
        "aggregate_validation_exact_accuracy_no_regression": True,
        "update0_aggregate_validation": None,
        "selected_aggregate_validation": None,
        "checks": {
            "development_checkpoint_selected": selected is not None,
            "changed_complete_pair_threshold_met": False,
            "aggregate_validation_exact_accuracy_retained": False,
        },
        "eligible": False,
    }
    if selected is not None and callable(promotion_evaluator):
        update0_tensors = load_file(checkpoints[0] / "adapter.safetensors", device="cpu")
        evaluator.install(update0_tensors)
        update0_count, update0_correct = promotion_evaluator()
        selected_index = int(selected["update"])
        selected_tensors = load_file(
            checkpoints[selected_index] / "adapter.safetensors", device="cpu"
        )
        evaluator.install(selected_tensors)
        selected_count, selected_correct = promotion_evaluator()
        if update0_count <= 0 or selected_count != update0_count:
            raise ValueError("Aggregate validation promotion audits are misaligned")
        update0_accuracy = update0_correct / update0_count
        selected_accuracy = selected_correct / selected_count
        promotion_checks = {
            "development_checkpoint_selected": True,
            "changed_complete_pair_threshold_met": (
                int(selected["greedy_exact_complete_units_correct"])
                >= requirements.promotion_changed_complete_pairs_minimum
            ),
            "aggregate_validation_exact_accuracy_retained": (selected_accuracy >= update0_accuracy),
        }
        promotion.update(
            {
                "evaluated": True,
                "update0_aggregate_validation": {
                    "row_count": update0_count,
                    "exact_correct": update0_correct,
                    "exact_accuracy": update0_accuracy,
                },
                "selected_aggregate_validation": {
                    "row_count": selected_count,
                    "exact_correct": selected_correct,
                    "exact_accuracy": selected_accuracy,
                },
                "checks": promotion_checks,
                "eligible": all(promotion_checks.values()),
            }
        )
    return {
        "schema_version": 1,
        "artifact": "v30_joint_pair_development_selection",
        "development_validation_model_selection_only": True,
        "final_test_scenes_touched": False,
        "deferred_final_scene_ids": sorted(DEFERRED_FINAL_SCENE_IDS),
        "training_evaluation_only": True,
        "question_text_serialized": False,
        "answer_text_serialized": False,
        "oracle_loaded": False,
        "question_dependent_scene_processing": False,
        "question_dependent_retrieval": False,
        "all_scene_prefixes_built_before_questions": True,
        "model_load_count": 1,
        "source_v29": source,
        "validation_scene_ids": list(evaluator.validation_scene_ids),
        "frozen_inherited_state_sha256": frozen_hash,
        "requirements": {
            "color_full_vocab_sides": requirements.color_full_vocab_sides,
            "mirror_full_vocab_sides": requirements.mirror_full_vocab_sides,
            "no_new_negative_sides": True,
            "selected_v29_source_nll_must_improve": True,
            "validation_pair_unit_count": requirements.validation_pair_unit_count,
            "minimum_pair_margin": requirements.minimum_pair_margin,
            "minimum_mean_margin_improvement": (requirements.minimum_mean_margin_improvement),
            "minimum_passed_unit_improvement": (requirements.minimum_passed_unit_improvement),
            "greedy_changed_row_count": requirements.greedy_changed_row_count,
            "minimum_greedy_complete_units_correct": (
                requirements.minimum_greedy_complete_units_correct
            ),
            "broad_retention_subset_size": requirements.broad_retention_subset_size,
            "broad_exact_accuracy_no_regression": True,
            "chat_promotion_changed_complete_pairs_minimum": (
                requirements.promotion_changed_complete_pairs_minimum
            ),
            "chat_promotion_aggregate_validation_exact_accuracy_no_regression": True,
        },
        "update0_pair_mean_margin": baseline_pair.mean_margin,
        "update0_pair_passed_units": baseline_pair.passed_units,
        "update0_broad_exact_accuracy": baseline_broad_accuracy,
        "arms": arms,
        "selected_checkpoint": None if selected is None else selected["checkpoint"],
        "selected_update": None if selected is None else selected["update"],
        "development_selection_passed": selected is not None,
        "chat_promotion": promotion,
        "chat_promotion_eligible": promotion["eligible"],
        "passed": selected is not None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/gemma4_diverse20_joint_pair_v30.yaml"),
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=Path("data_gemma4/checkpoints/gemma4_v30_diverse20_joint_pair"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/gemma4/metrics/v30_joint_pair_selection.json"),
    )
    args = parser.parse_args()
    report = select_joint_pair(args.config, args.checkpoint_root)
    _atomic_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
