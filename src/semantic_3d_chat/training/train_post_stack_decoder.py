"""Optional V28 Stage-B broad-QA adaptation of one fresh Gemma LoRA bank.

Stage B is a selector-gated transition, never a fallback.  It starts only from
the Stage-A checkpoint approved by ``v28_stage_a_selector``, freezes that
complete continuous-scene stack and every inherited decoder bank, then trains
one deterministic exact-zero rank-4 query bank at Gemma language layers 13
and 14.  Scene representations are built once before questions and reused for
all answer-token NLL batches.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from semantic_3d_chat.chat.runtime import (
    construct_scene_tokenizer,
    validate_checkpoint_contract,
)
from semantic_3d_chat.config import PROJECT_ROOT, artifact_root, config_hash, load_config
from semantic_3d_chat.data.dataset import QARecord
from semantic_3d_chat.data.splits import assert_scene_disjoint, split_fingerprint
from semantic_3d_chat.language.local_lm import load_local_language_model, prompt_token_ids
from semantic_3d_chat.language.lora import (
    LoRABankCollection,
    LoRALinear,
    install_lora_banks,
    lora_banks_checkpoint_contract,
    lora_banks_optimizer_settings,
    lora_banks_settings,
    tensor_state_sha256,
)
from semantic_3d_chat.language.prefix_injection import (
    ContinuousPrefixComposer,
    native_gemma4_image_contract_setting,
    prefix_sha256,
    scene_boundary_mode_setting,
    scene_prefix_after_bos_setting,
    stack_prefix_batches,
)
from semantic_3d_chat.scene_encoder.dense_alignment import (
    DenseAlignmentResidual,
    construct_dense_alignment,
    validate_dense_alignment_state,
)
from semantic_3d_chat.scene_encoder.dense_sidecar_adapter import (
    DenseSidecarAdapter,
    construct_dense_sidecar_adapter,
    validate_dense_sidecar_adapter_state,
)
from semantic_3d_chat.scene_encoder.global_residual import (
    GlobalSceneResidual,
    construct_global_scene_residual,
)
from semantic_3d_chat.scene_encoder.map_io import load_map_tensors
from semantic_3d_chat.scene_encoder.projector import SceneTokenizer
from semantic_3d_chat.scene_encoder.signed_x_dispatch import (
    SignedXSceneResidual,
    construct_signed_x_scene_residual,
)
from semantic_3d_chat.training.checkpointing import (
    RUNTIME_METADATA_FILENAME,
    TRAINING_METADATA_FILENAME,
    load_adapter_checkpoint,
    module_collection_state_sha256,
    save_adapter_checkpoint,
    save_optimizer_checkpoint,
    validate_runtime_checkpoint_metadata,
)
from semantic_3d_chat.training.losses import QuestionGroundingHead
from semantic_3d_chat.training.train_adapter import (
    forward_prefix_batch,
    load_qa_split_dataset,
    map_forward,
    named_lora_extension_checkpoint_modules,
    named_lora_extension_transition_mismatch,
    tokenize_answer,
    validate_named_lora_extension_transition_state,
)
from semantic_3d_chat.training.train_post_stack_sidecar import (
    _bounded_records,
    _epoch_batches,
    _file_sha256,
    _finite_number,
    _positive_int,
    _scalar_audit,
    records_by_scene,
)

DEFAULT_CONFIG = Path("configs/experiments/gemma4_color_mirror_post_stack_decoder_stage_b_v28.yaml")
DEFAULT_OUTPUT = Path("data_gemma4/checkpoints/gemma4_v28_post_stack_decoder_stage_b")
_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class StageBSettings:
    enabled: bool
    max_optimizer_steps: int
    evaluation_interval_steps: int
    batch_size: int
    gradient_accumulation: int
    learning_rate: float
    weight_decay: float
    gradient_clip_norm: float
    minimum_answer_types: int
    trainable_bank: str


@dataclass(frozen=True)
class V29DevelopmentContract:
    """Fail-closed contract for the scene-disjoint diverse20 development run."""

    qa_root: Path
    split_fingerprint: str
    train_scene_ids: tuple[str, ...]
    validation_scene_ids: tuple[str, ...]
    deferred_test_scene_ids: tuple[str, ...]
    train_question_count: int
    validation_question_count: int
    retention_control_config: Path
    update_zero_validation_reference: str


@dataclass(frozen=True)
class ApprovedStageASource:
    checkpoint: Path
    selection_report: Path
    selection_sha256: str
    selected_update: int
    selected_arm: dict[str, Any]


@dataclass(frozen=True)
class CachedFullScenePrefix:
    scene_id: str
    scene_tokens: torch.Tensor
    prefix_sha256: str
    voxel_count: int
    processed_voxels: int
    minimum_voxel_contribution: float


@dataclass
class StageBBundle:
    config: dict[str, Any]
    source_config: dict[str, Any]
    source_runtime_metadata: dict[str, Any]
    source_training_metadata: dict[str, Any]
    source: ApprovedStageASource
    language: Any
    scene_model: SceneTokenizer
    dense_aligner: DenseAlignmentResidual
    dense_sidecar_adapter: DenseSidecarAdapter
    global_scene_residual: GlobalSceneResidual
    signed_x_scene_residual: SignedXSceneResidual
    composer: ContinuousPrefixComposer
    grounding: QuestionGroundingHead
    lora_installation: LoRABankCollection
    checkpoint_modules: dict[str, torch.nn.Module]
    frozen_checkpoint_modules: dict[str, torch.nn.Module]
    trainable_bank_name: str


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (PROJECT_ROOT / value).resolve()


def v29_development_contract(
    config: Mapping[str, Any],
) -> V29DevelopmentContract | None:
    """Parse the optional V29 data lock without changing the V28 experiment."""

    raw = config.get("v29_development")
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise TypeError("v29_development must be a mapping")
    required = {
        "schema_version",
        "role",
        "qa_root",
        "split_fingerprint",
        "train_scene_ids",
        "validation_scene_ids",
        "deferred_test_scene_ids",
        "question_counts",
        "retention_control_config",
        "update_zero_validation_reference",
    }
    missing = sorted(required - set(raw))
    unknown = sorted(set(raw) - required)
    if missing or unknown:
        raise ValueError(f"Invalid v29_development fields: missing={missing} unknown={unknown}")
    if raw["schema_version"] != 1:
        raise ValueError("v29_development.schema_version must be 1")
    if raw["role"] not in {
        "scene_disjoint_diverse20_development",
        "scene_disjoint_diverse28_development",
    }:
        raise ValueError("v29_development.role does not authorize this run")

    qa_root = _resolve(str(raw["qa_root"]))
    configured_qa_root = artifact_root(dict(config), "qa").resolve()
    if qa_root != configured_qa_root:
        raise ValueError(
            "v29_development.qa_root must exactly match paths.qa_root: "
            f"contract={qa_root} configured={configured_qa_root}"
        )
    if "oracle" in {part.casefold() for part in qa_root.parts}:
        raise ValueError("V29 QA supervision must not use an oracle path")

    fingerprint = raw["split_fingerprint"]
    if not isinstance(fingerprint, str) or _SHA256.fullmatch(fingerprint) is None:
        raise ValueError("v29_development.split_fingerprint must be SHA-256")

    def scene_ids(field: str) -> tuple[str, ...]:
        values = raw[field]
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise TypeError(f"v29_development.{field} must be a sequence")
        parsed = tuple(str(value) for value in values)
        if not parsed or len(parsed) != len(set(parsed)):
            raise ValueError(f"v29_development.{field} must be nonempty and unique")
        if any(re.fullmatch(r"scene_[0-9]{6}", value) is None for value in parsed):
            raise ValueError(f"v29_development.{field} contains an invalid scene ID")
        return parsed

    train_ids = scene_ids("train_scene_ids")
    validation_ids = scene_ids("validation_scene_ids")
    deferred_ids = scene_ids("deferred_test_scene_ids")
    split_sets = {
        "train": list(train_ids),
        "validation": list(validation_ids),
        "test": list(deferred_ids),
    }
    assert_scene_disjoint(split_sets)

    counts = raw["question_counts"]
    if not isinstance(counts, Mapping) or set(counts) != {"train", "validation"}:
        raise ValueError(
            "v29_development.question_counts must contain exactly train and validation"
        )
    train_count = _positive_int("v29_development.question_counts.train", counts["train"])
    validation_count = _positive_int(
        "v29_development.question_counts.validation", counts["validation"]
    )

    retention_config = _resolve(str(raw["retention_control_config"]))
    if not retention_config.is_file():
        raise FileNotFoundError(f"V29 retention-control config is missing: {retention_config}")
    reference = raw["update_zero_validation_reference"]
    if reference != "current_dataset_exact_zero_fresh_bank":
        raise ValueError(
            "v29_development.update_zero_validation_reference must be "
            "current_dataset_exact_zero_fresh_bank"
        )
    return V29DevelopmentContract(
        qa_root=qa_root,
        split_fingerprint=fingerprint,
        train_scene_ids=train_ids,
        validation_scene_ids=validation_ids,
        deferred_test_scene_ids=deferred_ids,
        train_question_count=train_count,
        validation_question_count=validation_count,
        retention_control_config=retention_config,
        update_zero_validation_reference=reference,
    )


def load_stage_b_qa_records(
    config: Mapping[str, Any],
    *,
    max_train_questions: int | None,
    max_validation_questions: int | None,
) -> tuple[list[QARecord], list[QARecord], dict[str, Any]]:
    """Load Stage-B supervision and enforce V29's deferred-test boundary."""

    seed = int(config["seed"])
    qa_root = artifact_root(dict(config), "qa").resolve()
    train_dataset = load_qa_split_dataset(qa_root, "train")
    validation_dataset = load_qa_split_dataset(qa_root, "validation")
    contract = v29_development_contract(config)
    audit: dict[str, Any] = {
        "qa_root": str(qa_root),
        "split_guarded": (qa_root / "splits.json").is_file(),
        "v29_development_contract": contract is not None,
    }
    if contract is not None:
        manifest_path = qa_root / "splits.json"
        if not manifest_path.is_file():
            raise FileNotFoundError("V29 requires data_diverse20/qa/splits.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, Mapping) or not isinstance(manifest.get("splits"), Mapping):
            raise TypeError("V29 split manifest must contain a splits mapping")
        observed_splits = {
            name: sorted(str(value) for value in manifest["splits"].get(name, []))
            for name in ("train", "validation", "test")
        }
        assert_scene_disjoint(observed_splits)
        expected_splits = {
            "train": sorted(contract.train_scene_ids),
            "validation": sorted(contract.validation_scene_ids),
            # Deferred scenes are intentionally absent until the architecture is locked.
            "test": [],
        }
        if observed_splits != expected_splits:
            raise ValueError(
                "V29 persisted scene split differs from its development lock: "
                f"observed={observed_splits} expected={expected_splits}"
            )
        observed_fingerprint = split_fingerprint(observed_splits)
        if observed_fingerprint != contract.split_fingerprint:
            raise ValueError(
                "V29 persisted split fingerprint mismatch: "
                f"observed={observed_fingerprint} "
                f"expected={contract.split_fingerprint}"
            )
        train_records = list(train_dataset.records)
        validation_records = list(validation_dataset.records)
        observed_record_scenes = {
            "train": sorted({record.scene_id for record in train_records}),
            "validation": sorted({record.scene_id for record in validation_records}),
        }
        expected_record_scenes = {
            "train": sorted(contract.train_scene_ids),
            "validation": sorted(contract.validation_scene_ids),
        }
        if observed_record_scenes != expected_record_scenes:
            raise ValueError(
                "V29 QA records do not cover exactly the locked development scenes: "
                f"observed={observed_record_scenes} expected={expected_record_scenes}"
            )
        observed_counts = {
            "train": len(train_records),
            "validation": len(validation_records),
        }
        expected_counts = {
            "train": contract.train_question_count,
            "validation": contract.validation_question_count,
        }
        if observed_counts != expected_counts:
            raise ValueError(
                "V29 QA question counts differ from the locked balanced dataset: "
                f"observed={observed_counts} expected={expected_counts}"
            )
        deferred = set(contract.deferred_test_scene_ids)
        touched_deferred = sorted(
            deferred & {record.scene_id for record in (*train_records, *validation_records)}
        )
        if touched_deferred:
            raise ValueError(f"V29 development touched deferred test scenes: {touched_deferred}")
        test_path = qa_root / "test.jsonl"
        if not test_path.is_file() or test_path.read_text(encoding="utf-8").strip():
            raise ValueError("V29 requires an existing, empty deferred test.jsonl")
        audit.update(
            {
                "split_fingerprint": observed_fingerprint,
                "full_train_scene_ids": expected_record_scenes["train"],
                "full_validation_scene_ids": expected_record_scenes["validation"],
                "deferred_test_scene_ids_loaded": [],
                "full_question_counts": observed_counts,
                "retention_control_config": str(contract.retention_control_config),
            }
        )
    else:
        train_records = list(train_dataset.records)
        validation_records = list(validation_dataset.records)

    bounded_train = _bounded_records(
        train_records,
        max_train_questions,
        seed=seed,
    )
    bounded_validation = _bounded_records(
        validation_records,
        max_validation_questions,
        seed=seed + 1,
    )
    audit.update(
        {
            "selected_train_question_count": len(bounded_train),
            "selected_validation_question_count": len(bounded_validation),
            "selected_train_scene_ids": sorted({record.scene_id for record in bounded_train}),
            "selected_validation_scene_ids": sorted(
                {record.scene_id for record in bounded_validation}
            ),
        }
    )
    return bounded_train, bounded_validation, audit


def stage_b_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    raw = config.get("v28_stage_b")
    if not isinstance(raw, dict):
        raise TypeError("V28 Stage B requires a v28_stage_b mapping")
    required = {
        "schema_version",
        "role",
        "source_config",
        "stage_a_selection_report",
        "stage_a_checkpoint_root",
        "new_bank",
        "new_bank_parameter_count",
        "new_bank_initial_state_sha256",
        "update_zero_validation_nll_absolute_tolerance",
        "selection_requires",
    }
    missing = sorted(required - set(raw))
    unknown = sorted(set(raw) - required)
    if missing or unknown:
        raise ValueError(f"Invalid v28_stage_b fields: missing={missing} unknown={unknown}")
    if raw["schema_version"] != 1:
        raise ValueError("v28_stage_b.schema_version must be 1")
    if raw["role"] != "selector_gated_broad_qa_decoder_adaptation":
        raise ValueError("v28_stage_b.role does not authorize this transition")
    if not isinstance(raw["new_bank"], str) or not raw["new_bank"]:
        raise TypeError("v28_stage_b.new_bank must be a nonempty string")
    _positive_int("v28_stage_b.new_bank_parameter_count", raw["new_bank_parameter_count"])
    if (
        not isinstance(raw["new_bank_initial_state_sha256"], str)
        or _SHA256.fullmatch(raw["new_bank_initial_state_sha256"]) is None
    ):
        raise ValueError("v28_stage_b.new_bank_initial_state_sha256 must be SHA-256")
    _finite_number(
        "v28_stage_b.update_zero_validation_nll_absolute_tolerance",
        raw["update_zero_validation_nll_absolute_tolerance"],
        positive=False,
    )
    selection = raw["selection_requires"]
    if not isinstance(selection, Mapping):
        raise TypeError("v28_stage_b.selection_requires must be a mapping")
    selection_fields = {
        "color_full_vocab_sides",
        "mirror_full_vocab_sides",
        "no_new_negative_sides",
        "validation_nll_must_improve",
    }
    if set(selection) != selection_fields:
        raise ValueError(
            f"v28_stage_b.selection_requires fields must be exactly {sorted(selection_fields)}"
        )
    _positive_int(
        "v28_stage_b.selection_requires.color_full_vocab_sides",
        selection["color_full_vocab_sides"],
    )
    _positive_int(
        "v28_stage_b.selection_requires.mirror_full_vocab_sides",
        selection["mirror_full_vocab_sides"],
    )
    for field in ("no_new_negative_sides", "validation_nll_must_improve"):
        if selection[field] is not True:
            raise ValueError(f"v28_stage_b.selection_requires.{field} must be true")
    return dict(raw)


def stage_b_settings(config: Mapping[str, Any]) -> StageBSettings:
    training = config.get("training")
    if not isinstance(training, Mapping):
        raise TypeError("training config must be a mapping")
    raw = training.get("post_stack_decoder_stage_b")
    if not isinstance(raw, Mapping):
        raise TypeError("training.post_stack_decoder_stage_b must be a mapping")
    allowed = {
        "enabled",
        "max_optimizer_steps",
        "evaluation_interval_steps",
        "batch_size",
        "gradient_accumulation",
        "learning_rate",
        "weight_decay",
        "gradient_clip_norm",
        "minimum_answer_types",
        "trainable_bank",
    }
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"Unknown post_stack_decoder_stage_b settings: {unknown}")
    enabled = raw.get("enabled", False)
    if not isinstance(enabled, bool):
        raise TypeError("post_stack_decoder_stage_b.enabled must be boolean")
    bank = raw.get("trainable_bank")
    if not isinstance(bank, str) or not bank:
        raise TypeError("post_stack_decoder_stage_b.trainable_bank must be nonempty")
    settings = StageBSettings(
        enabled=enabled,
        max_optimizer_steps=_positive_int(
            "post_stack_decoder_stage_b.max_optimizer_steps",
            raw.get("max_optimizer_steps", 4),
        ),
        evaluation_interval_steps=_positive_int(
            "post_stack_decoder_stage_b.evaluation_interval_steps",
            raw.get("evaluation_interval_steps", 1),
        ),
        batch_size=_positive_int("post_stack_decoder_stage_b.batch_size", raw.get("batch_size", 1)),
        gradient_accumulation=_positive_int(
            "post_stack_decoder_stage_b.gradient_accumulation",
            raw.get("gradient_accumulation", 12),
        ),
        learning_rate=_finite_number(
            "post_stack_decoder_stage_b.learning_rate",
            raw.get("learning_rate", 1e-4),
            positive=True,
        ),
        weight_decay=_finite_number(
            "post_stack_decoder_stage_b.weight_decay",
            raw.get("weight_decay", 0.0),
            positive=False,
        ),
        gradient_clip_norm=_finite_number(
            "post_stack_decoder_stage_b.gradient_clip_norm",
            raw.get("gradient_clip_norm", 1.0),
            positive=True,
        ),
        minimum_answer_types=_positive_int(
            "post_stack_decoder_stage_b.minimum_answer_types",
            raw.get("minimum_answer_types", 4),
        ),
        trainable_bank=bank,
    )
    contract = stage_b_contract(config)
    if settings.trainable_bank != contract["new_bank"]:
        raise ValueError("Stage-B settings and transition contract name different banks")
    configured_lora_lr = training.get("lora_learning_rate")
    configured_lora_decay = training.get("lora_weight_decay")
    if settings.learning_rate != float(configured_lora_lr):
        raise ValueError("Stage-B learning rate must equal training.lora_learning_rate")
    if settings.weight_decay != float(configured_lora_decay):
        raise ValueError("Stage-B weight decay must equal training.lora_weight_decay")
    return settings


def require_approved_stage_a_source(
    config: Mapping[str, Any], report_override: Path | None = None
) -> ApprovedStageASource:
    """Resolve only a passed, eligible Stage-A selector decision."""

    contract = stage_b_contract(config)
    report_path = _resolve(
        report_override
        if report_override is not None
        else str(contract["stage_a_selection_report"])
    )
    if report_path.is_symlink() or not report_path.is_file():
        raise FileNotFoundError(f"Stage-A selector report is missing: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise TypeError("Stage-A selector report must be a JSON object")
    required_report = {
        "schema_version": 1,
        "artifact": "v28_post_stack_sidecar_stage_a_selection",
        "training_evaluation_only": True,
        "question_text_serialized": False,
        "answer_text_serialized": False,
        "oracle_loaded": False,
        "question_dependent_scene_processing": False,
        "passed": True,
    }
    mismatches = {
        key: {"observed": report.get(key), "required": value}
        for key, value in required_report.items()
        if report.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Stage-A selector did not approve Stage B: {mismatches}")
    selected_value = report.get("selected_checkpoint")
    selected_update = report.get("selected_update")
    if not isinstance(selected_value, str) or not selected_value:
        raise ValueError("Passed Stage-A selector lacks selected_checkpoint")
    if isinstance(selected_update, bool) or not isinstance(selected_update, int):
        raise TypeError("Passed Stage-A selector lacks integer selected_update")
    selected = _resolve(selected_value)
    root = _resolve(str(contract["stage_a_checkpoint_root"]))
    if not selected.is_relative_to(root):
        raise ValueError("Stage-A selector chose a checkpoint outside its contracted root")
    if selected.name != f"update_{selected_update:03d}":
        raise ValueError("Stage-A selected checkpoint name/update mismatch")
    for name in ("adapter.safetensors", TRAINING_METADATA_FILENAME, RUNTIME_METADATA_FILENAME):
        if not (selected / name).is_file():
            raise FileNotFoundError(f"Selected Stage-A checkpoint lacks {name}")
    arms = report.get("arms")
    if not isinstance(arms, list):
        raise TypeError("Stage-A selector report lacks arms")
    matching = [
        arm
        for arm in arms
        if isinstance(arm, dict)
        and _resolve(str(arm.get("checkpoint", ""))) == selected
        and arm.get("update") == selected_update
    ]
    if len(matching) != 1 or matching[0].get("eligible") is not True:
        raise ValueError("Stage-A selected checkpoint is not a unique eligible arm")
    if selected_update <= 0:
        raise ValueError("Stage B requires a nonzero selected Stage-A update")
    return ApprovedStageASource(
        checkpoint=selected,
        selection_report=report_path,
        selection_sha256=_file_sha256(report_path),
        selected_update=selected_update,
        selected_arm=dict(matching[0]),
    )


def _read_metadata(path: Path, filename: str) -> dict[str, Any]:
    value = json.loads((path / filename).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Checkpoint metadata must be a JSON object: {path / filename}")
    if filename == RUNTIME_METADATA_FILENAME:
        validate_runtime_checkpoint_metadata(value)
    return value


def _checkpoint_modules(
    *,
    scene_model: SceneTokenizer,
    composer: ContinuousPrefixComposer,
    grounding: QuestionGroundingHead,
    dense_aligner: DenseAlignmentResidual,
    sidecar: DenseSidecarAdapter,
    global_residual: GlobalSceneResidual,
    signed_x_residual: SignedXSceneResidual,
    lora: LoRABankCollection,
) -> dict[str, torch.nn.Module]:
    return {
        "scene_model": scene_model,
        "composer": composer,
        "grounding": grounding,
        "global_scene_residual": global_residual,
        "signed_x_scene_residual": signed_x_residual,
        "dense_aligner": dense_aligner,
        "dense_sidecar_adapter": sidecar,
        **lora.state_modules(),
    }


def freeze_for_stage_b(bundle: StageBBundle) -> list[torch.nn.Parameter]:
    """Freeze the selected scene stack and authorize only the fresh bank."""

    bundle.language.model.requires_grad_(False)
    for module in bundle.checkpoint_modules.values():
        module.requires_grad_(False)
        module.eval()
    bank = bundle.lora_installation.bank(bundle.trainable_bank_name)
    parameters = bank.installation.parameters()
    if not parameters:
        raise RuntimeError("Fresh Stage-B bank contains no parameters")
    for parameter in parameters:
        parameter.requires_grad_(True)
    bundle.lora_installation.train()
    bundle.lora_installation.assert_trainable_surface(bundle.language.model)
    return parameters


def assert_stage_b_trainable_surface(
    bundle: StageBBundle, optimizer: torch.optim.Optimizer | None = None
) -> dict[str, Any]:
    bank = bundle.lora_installation.bank(bundle.trainable_bank_name)
    authorized = {id(parameter) for parameter in bank.installation.parameters()}
    observed = {
        id(parameter) for parameter in bundle.language.model.parameters() if parameter.requires_grad
    }
    if observed != authorized:
        raise RuntimeError(
            "Stage-B trainable surface mismatch: "
            f"expected={len(authorized)} observed={len(observed)}"
        )
    for name, module in bundle.checkpoint_modules.items():
        if name == f"lora_banks.{bundle.trainable_bank_name}":
            continue
        if any(parameter.requires_grad for parameter in module.parameters()):
            raise RuntimeError(f"Frozen Stage-B checkpoint module became trainable: {name}")
    if optimizer is not None:
        optimizer_ids = {
            id(parameter) for group in optimizer.param_groups for parameter in group["params"]
        }
        if optimizer_ids != authorized:
            raise RuntimeError("Stage-B optimizer contains unauthorized parameters")
    return {
        "bank": bundle.trainable_bank_name,
        "parameter_count": sum(parameter.numel() for parameter in bank.installation.parameters()),
        "tensor_count": len(bank.installation.parameters()),
        "target_modules": list(bank.installation.target_names),
    }


def frozen_stage_b_state_sha256(bundle: StageBBundle) -> str:
    state = {
        f"{module_name}.{name}": value
        for module_name, module in bundle.frozen_checkpoint_modules.items()
        for name, value in module.state_dict().items()
    }
    return tensor_state_sha256(state)


def assert_frozen_stage_b_state(bundle: StageBBundle, expected: str) -> None:
    observed = frozen_stage_b_state_sha256(bundle)
    if observed != expected:
        raise RuntimeError(
            "Frozen selected Stage-A/inherited-LoRA state changed: "
            f"expected={expected} observed={observed}"
        )


def verify_fresh_bank_update_zero(
    bundle: StageBBundle, *, expected_hash: str, expected_parameter_count: int
) -> dict[str, Any]:
    """Prove the fresh query bank is deterministic and bit-exact zero-output."""

    bank = bundle.lora_installation.bank(bundle.trainable_bank_name)
    installation = bank.installation
    observed_hash = installation.state_sha256()
    if observed_hash != expected_hash:
        raise ValueError(
            f"Fresh Stage-B bank hash mismatch: expected={expected_hash} observed={observed_hash}"
        )
    if installation.parameter_count != expected_parameter_count:
        raise ValueError("Fresh Stage-B bank parameter count mismatch")
    targets: dict[str, bool] = {}
    for name, adapter in zip(installation.target_names, installation.adapters, strict=True):
        if not isinstance(adapter, LoRALinear):
            raise TypeError("Fresh Stage-B target is not a LoRALinear")
        if torch.count_nonzero(adapter.lora_b).item() != 0:
            raise ValueError(f"Fresh Stage-B target is not zero-output: {name}")
        values = (
            torch.linspace(
                -0.25,
                0.25,
                steps=2 * adapter.in_features,
                device="cpu",
                dtype=torch.float32,
            )
            .reshape(2, adapter.in_features)
            .to(
                device=adapter.base.weight.device,
                dtype=adapter.base.weight.dtype,
            )
        )
        was_training = adapter.training
        adapter.eval()
        with torch.inference_mode():
            base = adapter.base(values)
            wrapped = adapter(values)
        adapter.train(was_training)
        targets[name] = bool(torch.equal(base, wrapped))
        if not targets[name]:
            raise RuntimeError(f"Fresh Stage-B target changed update-0 output: {name}")
    return {
        "verified": True,
        "base": "selector_approved_stage_a_checkpoint",
        "bank": bundle.trainable_bank_name,
        "state_sha256": observed_hash,
        "parameter_count": installation.parameter_count,
        "target_outputs_bit_exact": targets,
        "all_scene_prefixes_cached_before_questions": True,
        "question_dependent_scene_processing": False,
    }


def load_stage_b_bundle(config: dict[str, Any], source: ApprovedStageASource) -> StageBBundle:
    contract = stage_b_contract(config)
    settings = stage_b_settings(config)
    source_config = load_config(str(contract["source_config"]))
    source_runtime = _read_metadata(source.checkpoint, RUNTIME_METADATA_FILENAME)
    source_training = _read_metadata(source.checkpoint, TRAINING_METADATA_FILENAME)
    semantic_dim = int(source_runtime["semantic_dim"])

    language = load_local_language_model(
        str(config["language"]["model_id"]),
        str(config["language"]["revision"]),
        str(config["language"]["dtype"]),
        freeze=True,
        local_files_only=True,
        backend=str(config["language"].get("backend", "auto")),
        decoder_gradient_checkpointing=bool(
            config["training"].get("language_decoder_gradient_checkpointing", True)
        ),
    )
    boundary_mode = scene_boundary_mode_setting(config)
    if language.scene_boundary_contract(boundary_mode) != (
        native_gemma4_image_contract_setting(config)
    ):
        raise ValueError("Loaded Gemma boundary contract does not match Stage-B config")
    lora = install_lora_banks(language.model, lora_banks_settings(config))
    if lora is None:
        raise ValueError("V28 Stage B requires named LoRA banks")
    fresh = lora.bank(settings.trainable_bank)
    if not fresh.settings.trainable:
        raise ValueError("V28 Stage-B fresh bank is not configured trainable")
    if fresh.settings.adapter.rank != 4 or float(fresh.settings.adapter.alpha) != 8.0:
        raise ValueError("V28 Stage-B fresh bank must be rank 4 / alpha 8")
    if tuple(fresh.settings.adapter.target_modules) != (
        "model.language_model.layers.13.self_attn.q_proj",
        "model.language_model.layers.14.self_attn.q_proj",
    ):
        raise ValueError("V28 Stage-B fresh bank target paths are not exact")

    scene_model = construct_scene_tokenizer(config, semantic_dim, language.hidden_size)
    dense_aligner = construct_dense_alignment(config, semantic_dim=semantic_dim)
    sidecar = construct_dense_sidecar_adapter(
        config,
        scene_dim=language.hidden_size,
        latent_count=int(config["scene_encoder"]["global_latents"]),
    )
    global_residual = construct_global_scene_residual(
        config,
        scene_dim=language.hidden_size,
        latent_count=int(config["scene_encoder"]["global_latents"]),
    )
    if dense_aligner is None or sidecar is None or global_residual is None:
        raise ValueError("V28 Stage B requires the complete selected Stage-A scene stack")
    signed_x = construct_signed_x_scene_residual(
        config,
        scene_dim=language.hidden_size,
        latent_count=int(config["scene_encoder"]["global_latents"]),
        content_dim=global_residual.width,
    )
    if signed_x is None:
        raise ValueError("V28 Stage B requires the frozen signed-X scene stack")
    composer = ContinuousPrefixComposer(
        language.hidden_size,
        scene_prefix_after_bos=scene_prefix_after_bos_setting(config),
        bos_token_id=language.bos_token_id,
        scene_boundary_mode=boundary_mode,
        native_boundary_embeddings=language.scene_boundary_embeddings(boundary_mode),
    )
    grounding = QuestionGroundingHead(
        int(config["scene_encoder"]["model_dim"]),
        language.hidden_size,
        int(config["scene_encoder"]["global_latents"]),
        int(config["scene_encoder"]["model_dim"]),
    )
    checkpoint_modules = _checkpoint_modules(
        scene_model=scene_model,
        composer=composer,
        grounding=grounding,
        dense_aligner=dense_aligner,
        sidecar=sidecar,
        global_residual=global_residual,
        signed_x_residual=signed_x,
        lora=lora,
    )
    source_modules = named_lora_extension_checkpoint_modules(checkpoint_modules, lora)
    transition_mismatch = named_lora_extension_transition_mismatch(source_runtime, lora)
    if transition_mismatch is not None:
        raise ValueError(f"Stage-A to Stage-B LoRA transition mismatch: {transition_mismatch}")
    loaded = load_adapter_checkpoint(
        source.checkpoint,
        source_modules,
        device="cpu",
        metadata_filename=RUNTIME_METADATA_FILENAME,
    )
    if loaded != source_runtime:
        raise RuntimeError("Stage-A source metadata changed during Stage-B load")
    validate_named_lora_extension_transition_state(source_runtime, lora)

    frozen_counts = {
        bank.settings.name: bank.installation.parameter_count
        for bank in lora.banks
        if not bank.settings.trainable
    }
    validate_checkpoint_contract(
        source_runtime,
        source_config,
        semantic_dim=semantic_dim,
        language_hidden_dim=language.hidden_size,
        lora_parameter_count=sum(frozen_counts.values()),
        lora_parameter_counts=frozen_counts,
        dense_alignment_parameter_count=dense_aligner.parameter_count,
        dense_sidecar_adapter_parameter_count=sidecar.parameter_count,
    )
    validate_dense_alignment_state(
        dense_aligner,
        expected_parameter_count=int(source_runtime["dense_alignment_parameter_count"]),
        context="V28 Stage-B source load",
    )
    validate_dense_sidecar_adapter_state(
        sidecar,
        expected_parameter_count=int(source_runtime["dense_sidecar_adapter_parameter_count"]),
        expected_state_sha256=str(source_runtime["dense_sidecar_adapter_state_sha256"]),
        context="V28 Stage-B source load",
    )
    if dense_aligner.state_sha256() != source_runtime["dense_alignment_state_sha256"]:
        raise ValueError("Stage-B frozen dense-aligner hash mismatch")
    if module_collection_state_sha256(
        {"global_scene_residual": global_residual}
    ) != source_runtime.get("global_scene_residual_state_sha256"):
        raise ValueError("Stage-B frozen global residual hash mismatch")
    if module_collection_state_sha256({"signed_x_scene_residual": signed_x}) != source_runtime.get(
        "signed_x_scene_residual_state_sha256"
    ):
        raise ValueError("Stage-B frozen signed-X residual hash mismatch")
    scene_modules = {
        "scene_model": scene_model,
        "composer": composer,
        "grounding": grounding,
    }
    if module_collection_state_sha256(scene_modules) != source_runtime.get(
        "frozen_scene_state_sha256"
    ):
        raise ValueError("Stage-B frozen scene/composer/grounding hash mismatch")

    device = language.device
    for module in checkpoint_modules.values():
        module.to(device)
    fresh_module_name = f"lora_banks.{settings.trainable_bank}"
    frozen_modules = {
        name: module for name, module in checkpoint_modules.items() if name != fresh_module_name
    }
    bundle = StageBBundle(
        config=config,
        source_config=source_config,
        source_runtime_metadata=source_runtime,
        source_training_metadata=source_training,
        source=source,
        language=language,
        scene_model=scene_model,
        dense_aligner=dense_aligner,
        dense_sidecar_adapter=sidecar,
        global_scene_residual=global_residual,
        signed_x_scene_residual=signed_x,
        composer=composer,
        grounding=grounding,
        lora_installation=lora,
        checkpoint_modules=checkpoint_modules,
        frozen_checkpoint_modules=frozen_modules,
        trainable_bank_name=settings.trainable_bank,
    )
    freeze_for_stage_b(bundle)
    assert_stage_b_trainable_surface(bundle)
    verify_fresh_bank_update_zero(
        bundle,
        expected_hash=str(contract["new_bank_initial_state_sha256"]),
        expected_parameter_count=int(contract["new_bank_parameter_count"]),
    )
    return bundle


def cache_full_scene_prefixes(
    bundle: StageBBundle, scene_ids: Sequence[str]
) -> tuple[dict[str, CachedFullScenePrefix], dict[str, Any]]:
    """Build each selected continuous scene prefix once, before questions."""

    caches: dict[str, CachedFullScenePrefix] = {}
    files: list[str] = []
    started = time.perf_counter()
    model_dtype = next(bundle.language.model.parameters()).dtype
    for scene_id in sorted(set(scene_ids)):
        map_path = (artifact_root(bundle.config, "maps") / scene_id / "voxel_map.npz").resolve()
        if "oracle" in {part.casefold() for part in map_path.parts}:
            raise RuntimeError("Refusing oracle environmental input during Stage B")
        data = load_map_tensors(
            map_path,
            bundle.config["scene"]["room_size_m"],
            bundle.language.device,
            input_voxel_size_m=bundle.config["scene_encoder"].get("input_voxel_size_m"),
        )
        with torch.no_grad():
            output = map_forward(
                bundle.scene_model,
                data,
                bundle.global_scene_residual,
                bundle.signed_x_scene_residual,
                bundle.dense_aligner,
                bundle.dense_sidecar_adapter,
            )
            processed = int(_scalar_audit(output.audit, "processed_voxels"))
            sidecar_processed = int(_scalar_audit(output.audit, "aligned_sidecar_processed_voxels"))
            minimum = _scalar_audit(output.audit, "aligned_sidecar_min_voxel_contribution")
            if processed != data.voxel_count or sidecar_processed != data.voxel_count:
                raise RuntimeError(f"Incomplete cached Stage-B scene: {scene_id}")
            if not math.isfinite(minimum) or minimum <= 0:
                raise RuntimeError(f"Stage-B sidecar omitted a voxel: {scene_id}")
            scene_tokens = output.scene_tokens.detach()
            prefix = bundle.composer.scene_prefix(scene_tokens.to(model_dtype))
            caches[scene_id] = CachedFullScenePrefix(
                scene_id=scene_id,
                scene_tokens=scene_tokens,
                prefix_sha256=prefix_sha256(prefix),
                voxel_count=data.voxel_count,
                processed_voxels=processed,
                minimum_voxel_contribution=minimum,
            )
        files.append(str(map_path))
        del output, data
        if bundle.language.device.type == "mps":
            torch.mps.empty_cache()
    return caches, {
        "schema_version": 1,
        "scene_count": len(caches),
        "cache_build_seconds": time.perf_counter() - started,
        "question_inputs_to_scene_cache": False,
        "question_dependent_retrieval": False,
        "all_voxels_covered": True,
        "oracle_environment_files_loaded": False,
        "loaded_environment_files": files,
        "prefix_sha256_by_scene": {
            scene_id: cache.prefix_sha256 for scene_id, cache in sorted(caches.items())
        },
    }


def cached_scene_answer_nll(
    *,
    cache: CachedFullScenePrefix,
    records: Sequence[QARecord],
    bundle: StageBBundle,
) -> torch.Tensor:
    """Run answer-token-only NLL using one immutable complete scene prefix."""

    if not records:
        raise ValueError("Stage-B answer NLL requires records")
    if any(record.scene_id != cache.scene_id for record in records):
        raise ValueError("Stage-B QA batch does not match its cached scene")
    scene_tokens = cache.scene_tokens.expand(len(records), -1, -1)
    model_dtype = next(bundle.language.model.parameters()).dtype
    embedding = bundle.language.model.get_input_embeddings()
    batches = []
    for index, record in enumerate(records):
        prompt_ids = prompt_token_ids(
            bundle.language.tokenizer,
            str(bundle.config["language"]["system_prompt"]),
            record.question,
            bundle.language.device,
        )
        answer_ids = tokenize_answer(
            bundle.language.tokenizer, record.answer, bundle.language.device
        )
        batches.append(
            bundle.composer.compose(
                scene_tokens[index : index + 1].to(model_dtype),
                prompt_ids,
                embedding,
                answer_ids,
                prefix_backend=getattr(bundle.language, "prefix_backend", None),
            )
        )
    batch = stack_prefix_batches(
        batches,
        bundle.language.device,
        prefix_backend=getattr(bundle.language, "prefix_backend", None),
    )
    if batch.labels is None or not torch.any(batch.labels != -100):
        raise RuntimeError("Stage-B NLL batch contains no answer labels")
    output = forward_prefix_batch(bundle.language, batch)
    loss = output.loss.float()
    if loss.ndim != 0 or not torch.isfinite(loss):
        raise RuntimeError("Stage-B answer-token NLL is invalid")
    return loss


def validation_answer_nll(
    *,
    records: Sequence[QARecord],
    caches: Mapping[str, CachedFullScenePrefix],
    bundle: StageBBundle,
    batch_size: int,
) -> dict[str, Any]:
    bundle.lora_installation.eval()
    weighted = 0.0
    questions = 0
    try:
        with torch.inference_mode():
            for scene_id, scene_records in records_by_scene(records).items():
                for offset in range(0, len(scene_records), batch_size):
                    batch = scene_records[offset : offset + batch_size]
                    loss = cached_scene_answer_nll(
                        cache=caches[scene_id], records=batch, bundle=bundle
                    )
                    weighted += float(loss.detach().cpu()) * len(batch)
                    questions += len(batch)
    finally:
        bundle.lora_installation.train()
    if questions == 0:
        raise ValueError("Stage-B validation records are empty")
    return {
        "answer_token_nll": weighted / questions,
        "question_count": questions,
        "cached_full_scene_prefixes_reused": True,
        "environmental_oracle_files_loaded": False,
        "question_dependent_scene_processing": False,
    }


def _source_validation_nll(metadata: Mapping[str, Any]) -> float:
    history = metadata.get("history")
    if not isinstance(history, list) or not history:
        raise ValueError("Selected Stage-A checkpoint lacks history")
    value = history[-1].get("validation_answer_token_nll")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("Selected Stage-A checkpoint lacks validation NLL")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("Selected Stage-A validation NLL is nonfinite")
    return parsed


def _optimizer(bundle: StageBBundle, settings: StageBSettings) -> torch.optim.AdamW:
    parameters = bundle.lora_installation.parameters()
    optimizer = torch.optim.AdamW(
        [
            {
                "name": settings.trainable_bank,
                "params": parameters,
                "lr": settings.learning_rate,
                "weight_decay": settings.weight_decay,
            }
        ]
    )
    assert_stage_b_trainable_surface(bundle, optimizer)
    return optimizer


def _metadata(
    *,
    bundle: StageBBundle,
    settings: StageBSettings,
    cache_audit: Mapping[str, Any],
    qa_audit: Mapping[str, Any],
    frozen_hash: str,
    update_zero: Mapping[str, Any],
    train_records: Sequence[QARecord],
    validation_records: Sequence[QARecord],
    history: Sequence[Mapping[str, Any]],
    optimizer_step: int,
    best_update: int,
    best_validation: float,
    trainable_surface: Mapping[str, Any],
) -> dict[str, Any]:
    result = dict(bundle.source_runtime_metadata)
    lora_settings = lora_banks_settings(bundle.config)
    lora_optimizer = lora_banks_optimizer_settings(bundle.config, lora_settings)
    result.update(
        {
            "schema_version": 3,
            "config_hash": config_hash(bundle.config),
            "epoch": optimizer_step,
            "optimizer_step": optimizer_step,
            "history": list(history),
            "best_epoch": best_update,
            "best_monitor_loss": best_validation,
            "monitor_name": "validation_answer_token_nll",
            "lora": lora_banks_checkpoint_contract(
                lora_settings,
                lora_optimizer,
                bundle.lora_installation.parameter_counts,
            ),
            **bundle.lora_installation.checkpoint_metadata(),
            "freeze_scene_adapter": True,
            "question_dependent_scene_processing": False,
            "v28_stage_b": {
                "schema_version": 1,
                "source_stage_a_checkpoint": str(bundle.source.checkpoint),
                "source_stage_a_adapter_sha256": _file_sha256(
                    bundle.source.checkpoint / "adapter.safetensors"
                ),
                "source_stage_a_runtime_metadata_sha256": _file_sha256(
                    bundle.source.checkpoint / RUNTIME_METADATA_FILENAME
                ),
                "stage_a_selection_report": str(bundle.source.selection_report),
                "stage_a_selection_report_sha256": bundle.source.selection_sha256,
                "stage_a_selected_update": bundle.source.selected_update,
                "stage_a_selected_arm": bundle.source.selected_arm,
                "objective": "broad_answer_token_nll",
                "settings": settings.__dict__,
                "trainable_surface": dict(trainable_surface),
                "update_zero_equivalence": dict(update_zero),
                "frozen_state_sha256": frozen_hash,
                "scene_cache": dict(cache_audit),
                "qa_dataset": dict(qa_audit),
                "train_scene_ids": sorted({item.scene_id for item in train_records}),
                "validation_scene_ids": sorted({item.scene_id for item in validation_records}),
                "train_question_count": len(train_records),
                "validation_question_count": len(validation_records),
                "train_answer_types": sorted({item.answer_type for item in train_records}),
                "qa_supervision_serialized_to_runtime": False,
                "oracle_environment_files_loaded": False,
                "selected_sidecar_frozen": True,
                "all_inherited_lora_banks_frozen": True,
                "composer_grounding_frozen": True,
                "causal_retention_required_before_promotion": True,
            },
        }
    )
    return result


def _save(
    path: Path,
    *,
    bundle: StageBBundle,
    metadata: dict[str, Any],
    optimizer: torch.optim.Optimizer | None,
) -> None:
    save_adapter_checkpoint(path, bundle.checkpoint_modules, metadata)
    if optimizer is not None:
        save_optimizer_checkpoint(path, optimizer)


def run_stage_b(
    *,
    config: dict[str, Any],
    output: Path,
    selection_report: Path | None = None,
    max_optimizer_steps_override: int | None = None,
    max_train_questions: int | None = None,
    max_validation_questions: int | None = None,
) -> dict[str, Any]:
    """Run the selector-gated bounded Stage-B experiment."""

    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty Stage-B output: {output}")
    settings = stage_b_settings(config)
    if not settings.enabled:
        raise ValueError("V28 Stage B is disabled")
    if max_optimizer_steps_override is not None:
        settings = StageBSettings(
            **{
                **settings.__dict__,
                "max_optimizer_steps": _positive_int(
                    "max_optimizer_steps", max_optimizer_steps_override
                ),
            }
        )
    source = require_approved_stage_a_source(config, selection_report)
    seed = int(config["seed"])
    torch.manual_seed(seed)
    random.seed(seed)
    train_records, validation_records, qa_audit = load_stage_b_qa_records(
        config,
        max_train_questions=max_train_questions,
        max_validation_questions=max_validation_questions,
    )
    train_scenes = {item.scene_id for item in train_records}
    validation_scenes = {item.scene_id for item in validation_records}
    overlap = sorted(train_scenes & validation_scenes)
    if overlap:
        raise ValueError(f"Stage-B train/validation scene overlap: {overlap}")
    answer_types = {item.answer_type for item in train_records}
    if len(answer_types) < settings.minimum_answer_types:
        raise ValueError("Stage-B selection is not broad across answer types")

    bundle = load_stage_b_bundle(config, source)
    caches, cache_audit = cache_full_scene_prefixes(
        bundle, sorted(train_scenes | validation_scenes)
    )
    frozen_hash = frozen_stage_b_state_sha256(bundle)
    trainable_surface = assert_stage_b_trainable_surface(bundle)
    contract = stage_b_contract(config)
    update_zero = verify_fresh_bank_update_zero(
        bundle,
        expected_hash=str(contract["new_bank_initial_state_sha256"]),
        expected_parameter_count=int(contract["new_bank_parameter_count"]),
    )
    baseline = validation_answer_nll(
        records=validation_records,
        caches=caches,
        bundle=bundle,
        batch_size=settings.batch_size,
    )
    tolerance = float(contract["update_zero_validation_nll_absolute_tolerance"])
    observed_nll = float(baseline["answer_token_nll"])
    development = v29_development_contract(config)
    if development is None:
        source_nll = _source_validation_nll(bundle.source_training_metadata)
        if abs(observed_nll - source_nll) > tolerance:
            raise RuntimeError(
                "Fresh Stage-B bank changed update-0 validation NLL: "
                f"source={source_nll} observed={observed_nll} tolerance={tolerance}"
            )
        update_zero = {
            **update_zero,
            "source_validation_answer_token_nll": source_nll,
            "source_validation_dataset_comparable": True,
            "observed_validation_answer_token_nll": observed_nll,
            "validation_nll_absolute_tolerance": tolerance,
            "validation_nll_equivalent": True,
            "validation_equivalence_basis": "same_validation_dataset_nll",
        }
    else:
        # Stage A was selected on the older control dataset, so its scalar NLL
        # cannot be compared to V29's new scene-disjoint validation records.
        # Exact-zero B tensors and per-target bit-exact forward checks above
        # establish that the fresh bank is a functional no-op; this observed
        # value is therefore the immutable update-0 reference for V29.
        if not all(update_zero["target_outputs_bit_exact"].values()):
            raise RuntimeError("V29 fresh Stage-B bank is not bit-exact at update zero")
        update_zero = {
            **update_zero,
            "source_validation_answer_token_nll": None,
            "source_validation_dataset_comparable": False,
            "observed_validation_answer_token_nll": observed_nll,
            "validation_nll_absolute_tolerance": tolerance,
            "validation_nll_equivalent": True,
            "validation_equivalence_basis": (development.update_zero_validation_reference),
        }

    optimizer = _optimizer(bundle, settings)
    history: list[dict[str, Any]] = [
        {
            "optimizer_update": 0,
            "validation_answer_token_nll": observed_nll,
            "update_0_equivalence_verified": True,
        }
    ]
    best_update = 0
    best_validation = observed_nll
    output.mkdir(parents=True, exist_ok=True)
    initial_metadata = _metadata(
        bundle=bundle,
        settings=settings,
        cache_audit=cache_audit,
        qa_audit=qa_audit,
        frozen_hash=frozen_hash,
        update_zero=update_zero,
        train_records=train_records,
        validation_records=validation_records,
        history=history,
        optimizer_step=0,
        best_update=0,
        best_validation=best_validation,
        trainable_surface=trainable_surface,
    )
    _save(output / "update_000", bundle=bundle, metadata=initial_metadata, optimizer=None)
    _save(output / "best", bundle=bundle, metadata=initial_metadata, optimizer=None)

    batch_queue: list[tuple[str, list[QARecord]]] = []
    curriculum_cycle = 0
    for update in range(1, settings.max_optimizer_steps + 1):
        bundle.lora_installation.train()
        window: list[tuple[str, list[QARecord]]] = []
        while len(window) < settings.gradient_accumulation:
            if not batch_queue:
                curriculum_cycle += 1
                batch_queue = _epoch_batches(
                    train_records,
                    batch_size=settings.batch_size,
                    seed=seed + curriculum_cycle,
                )
            window.append(batch_queue.pop())
        optimizer.zero_grad(set_to_none=True)
        weighted_loss = 0.0
        question_count = 0
        for scene_id, batch_records in window:
            loss = cached_scene_answer_nll(
                cache=caches[scene_id], records=batch_records, bundle=bundle
            )
            (loss / len(window)).backward()
            weighted_loss += float(loss.detach().cpu()) * len(batch_records)
            question_count += len(batch_records)
        parameters = bundle.lora_installation.parameters()
        assert_stage_b_trainable_surface(bundle, optimizer)
        if any(parameter.grad is None for parameter in parameters):
            raise RuntimeError("Stage-B fresh bank did not receive gradients")
        if any(not torch.isfinite(parameter.grad).all() for parameter in parameters):
            raise RuntimeError("Stage-B fresh-bank gradient is nonfinite")
        gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, settings.gradient_clip_norm)
        optimizer.step()
        assert_frozen_stage_b_state(bundle, frozen_hash)
        bundle.lora_installation.validate_state()
        validation = (
            validation_answer_nll(
                records=validation_records,
                caches=caches,
                bundle=bundle,
                batch_size=settings.batch_size,
            )
            if update % settings.evaluation_interval_steps == 0
            or update == settings.max_optimizer_steps
            else None
        )
        validation_value = None if validation is None else float(validation["answer_token_nll"])
        improved = validation_value is not None and validation_value < best_validation
        if improved:
            best_validation = float(validation_value)
            best_update = update
        bank_hash = bundle.lora_installation.bank(
            settings.trainable_bank
        ).installation.state_sha256()
        history.append(
            {
                "optimizer_update": update,
                "curriculum_cycle": curriculum_cycle,
                "train_window_answer_token_nll": weighted_loss / question_count,
                "train_question_count": question_count,
                "validation_answer_token_nll": validation_value,
                "preclip_gradient_norm": float(gradient_norm.detach().cpu()),
                "trainable_bank_state_sha256": bank_hash,
                "frozen_state_sha256": frozen_hash,
            }
        )
        metadata = _metadata(
            bundle=bundle,
            settings=settings,
            cache_audit=cache_audit,
            qa_audit=qa_audit,
            frozen_hash=frozen_hash,
            update_zero=update_zero,
            train_records=train_records,
            validation_records=validation_records,
            history=history,
            optimizer_step=update,
            best_update=best_update,
            best_validation=best_validation,
            trainable_surface=trainable_surface,
        )
        _save(
            output / f"update_{update:03d}", bundle=bundle, metadata=metadata, optimizer=optimizer
        )
        if improved:
            _save(output / "best", bundle=bundle, metadata=metadata, optimizer=None)
        print(
            json.dumps(
                {
                    "phase": "v28_stage_b_update",
                    "optimizer_update": update,
                    "train_window_answer_token_nll": history[-1]["train_window_answer_token_nll"],
                    "validation_answer_token_nll": validation_value,
                    "best_update": best_update,
                    "best_validation_answer_token_nll": best_validation,
                }
            ),
            flush=True,
        )
    assert_frozen_stage_b_state(bundle, frozen_hash)
    return {
        "schema_version": 1,
        "artifact": "v28_post_stack_decoder_stage_b",
        "output": str(output),
        "best_checkpoint": str(output / "best"),
        "best_update": best_update,
        "baseline_validation_answer_token_nll": observed_nll,
        "best_validation_answer_token_nll": best_validation,
        "optimizer_updates": settings.max_optimizer_steps,
        "trainable_surface": trainable_surface,
        "frozen_state_sha256": frozen_hash,
        "stage_a_selector_approval_sha256": source.selection_sha256,
        "causal_retention_required_before_promotion": True,
        "question_dependent_scene_processing": False,
        "oracle_environment_files_loaded": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stage-a-selection", type=Path)
    parser.add_argument("--max-optimizer-steps", type=int)
    parser.add_argument("--max-train-questions", type=int)
    parser.add_argument("--max-validation-questions", type=int)
    args = parser.parse_args()
    report = run_stage_b(
        config=load_config(args.config),
        output=_resolve(args.output),
        selection_report=args.stage_a_selection,
        max_optimizer_steps_override=args.max_optimizer_steps,
        max_train_questions=args.max_train_questions,
        max_validation_questions=args.max_validation_questions,
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
