from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import shutil
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from semantic_3d_chat.config import (
    PROJECT_ROOT,
    artifact_root,
    config_hash,
    load_config,
    project_path,
)
from semantic_3d_chat.data.dataset import QARecord, SceneQADataset
from semantic_3d_chat.evaluation.candidate_gate_detail import build_candidate_gate_detail
from semantic_3d_chat.language.local_lm import (
    load_local_language_model,
    prompt_token_ids,
    question_token_ids,
)
from semantic_3d_chat.language.lora import (
    InstalledLoRABank,
    LoRABankCollection,
    LoRAInstallation,
    LoRAOptimizerSettings,
    install_lora_banks,
    lora_banks_checkpoint_contract,
    lora_banks_optimizer_settings,
    lora_banks_settings,
    lora_checkpoint_contract_mismatch,
    validate_lora_banks_checkpoint_state,
    validate_lora_checkpoint_state,
)
from semantic_3d_chat.language.prefix_injection import (
    ContinuousPrefixComposer,
    PrefixBatch,
    native_gemma4_image_contract_setting,
    prefix_sha256,
    scene_boundary_contract_mismatch,
    scene_boundary_mode_setting,
    scene_prefix_after_bos_contract_mismatch,
    scene_prefix_after_bos_setting,
    stack_prefix_batches,
)
from semantic_3d_chat.scene_encoder.dense_alignment import (
    DenseAlignmentResidual,
    construct_dense_alignment,
    dense_alignment_settings,
    validate_dense_alignment_state,
)
from semantic_3d_chat.scene_encoder.dense_sidecar_adapter import (
    DenseSidecarAdapter,
    apply_dense_sidecar_adapter,
)
from semantic_3d_chat.scene_encoder.global_residual import (
    ZERO_SPATIAL_MEAN_CONTENT_GATE_V1,
    GlobalSceneResidual,
    apply_global_scene_residual,
    construct_global_scene_residual,
    global_scene_residual_settings,
)
from semantic_3d_chat.scene_encoder.map_io import MapTensorData, load_map_tensors
from semantic_3d_chat.scene_encoder.projector import SceneTokenizer
from semantic_3d_chat.scene_encoder.signed_x_dispatch import (
    SignedXSceneResidual,
    apply_signed_x_scene_residual,
    construct_signed_x_scene_residual,
    frozen_v18_centered_content_values,
    signed_x_scene_residual_settings,
)
from semantic_3d_chat.training.checkpointing import (
    load_adapter_checkpoint,
    load_optimizer_checkpoint,
    module_collection_state_sha256,
    save_adapter_checkpoint,
    save_optimizer_checkpoint,
)
from semantic_3d_chat.training.dense_alignment_calibration import (
    require_dense_alignment_calibration_authorized,
    run_dense_alignment_calibration_warmup,
)
from semantic_3d_chat.training.losses import (
    QuestionGroundingHead,
    latent_diversity_loss,
    nearest_spatial_anchor_indices,
    normalize_xyz,
    ordered_spatial_relation_contrastive_loss,
    paired_scene_separation_loss,
    spatial_scene_answer_contrastive_loss,
)
from semantic_3d_chat.training.pair_curriculum import (
    CounterfactualPairUnit,
    PairObjectivePolicy,
    build_epoch_curriculum,
    build_exact_question_pair_units,
    candidate_logit_margins,
    cap_pair_units_per_pair,
    differing_answer_token_masks,
    first_answer_token_full_vocab_margins,
    pair_curriculum_settings,
    pair_gate_metrics,
    pair_objective_policy_contract,
    pair_objective_policy_settings,
    pair_ranking_hinge,
    ranking_margin_hinge,
    restrict_labels_to_answer_mask,
    select_pair_only_records,
    single_differing_answer_token,
    token_normalized_nll,
    validate_pair_objective_policy_coverage,
)
from semantic_3d_chat.training.source_provenance import (
    capture_git_source_provenance,
    require_clean_committed_source,
    source_provenance_resume_contract_mismatch,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def file_sha256(path: Path) -> str:
    """Hash a local artifact without loading it into memory at once."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def optional_sha256_setting(settings: Mapping[str, object], key: str) -> str | None:
    """Parse an optional lowercase SHA-256 configuration assertion."""

    value = settings.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"training.{key} must be a lowercase SHA-256 hex digest")
    return value


def declared_global_scene_residual_parameter_count(config: Mapping[str, object]) -> int | None:
    """Return an optional experiment-level assertion for the residual surface."""

    experiment = config.get("experiment")
    if experiment is None:
        return None
    if not isinstance(experiment, Mapping):
        raise TypeError("experiment config must be a mapping")
    value = experiment.get("residual_parameter_count")
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("experiment.residual_parameter_count must be a positive integer")
    return value


def declared_signed_x_scene_residual_parameter_count(
    config: Mapping[str, object],
) -> int | None:
    """Return an optional experiment assertion for the signed-X surface."""

    experiment = config.get("experiment")
    if experiment is None:
        return None
    if not isinstance(experiment, Mapping):
        raise TypeError("experiment config must be a mapping")
    value = experiment.get("signed_x_residual_parameter_count")
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("experiment.signed_x_residual_parameter_count must be a positive integer")
    return value


def declared_dense_alignment_parameter_count(config: Mapping[str, object]) -> int | None:
    """Return an optional experiment assertion for the dense alignment surface."""

    experiment = config.get("experiment")
    if experiment is None:
        return None
    if not isinstance(experiment, Mapping):
        raise TypeError("experiment config must be a mapping")
    canonical_key = "dense_alignment_trainable_parameter_count"
    legacy_key = "dense_alignment_parameter_count"
    if canonical_key in experiment and legacy_key in experiment:
        raise ValueError(
            "experiment must not declare both dense-alignment parameter-count keys"
        )
    key = canonical_key if canonical_key in experiment else legacy_key
    value = experiment.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"experiment.{key} must be a positive integer")
    return value


def explicit_adamw_options(config: Mapping[str, object]) -> dict[str, object]:
    """Return optional, fully explicit AdamW implementation controls.

    Historical experiments omit ``training.optimizer`` and retain PyTorch's
    defaults. Architecture screens that predict an exact first optimizer state
    must declare every implementation-sensitive switch so the real optimizer
    cannot silently choose a foreach or fused path.
    """

    training = config.get("training")
    if not isinstance(training, Mapping):
        raise TypeError("training config must be a mapping")
    raw = training.get("optimizer")
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise TypeError("training.optimizer must be a mapping")
    expected = {
        "name",
        "learning_rate",
        "betas",
        "epsilon",
        "weight_decay",
        "foreach",
        "fused",
        "capturable",
        "maximize",
        "amsgrad",
        "gradient_clip_norm",
        "accumulation_divisor",
        "step_index",
    }
    unknown = sorted(set(raw) - expected)
    missing = sorted(expected - set(raw))
    if missing or unknown:
        raise ValueError(f"training.optimizer keys mismatch: missing={missing} unknown={unknown}")
    if raw["name"] != "AdamW":
        raise ValueError("training.optimizer.name must equal 'AdamW'")
    for optimizer_key, training_key in (
        ("learning_rate", "learning_rate"),
        ("weight_decay", "weight_decay"),
        ("gradient_clip_norm", "gradient_clip_norm"),
    ):
        value = float(raw[optimizer_key])
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"training.optimizer.{optimizer_key} must be finite and nonnegative")
        if value != float(training[training_key]):
            raise ValueError(
                f"training.optimizer.{optimizer_key} disagrees with training.{training_key}"
            )
    accumulation_divisor = raw["accumulation_divisor"]
    if (
        isinstance(accumulation_divisor, bool)
        or not isinstance(accumulation_divisor, int)
        or accumulation_divisor < 1
        or accumulation_divisor != int(training["gradient_accumulation"])
    ):
        raise ValueError(
            "training.optimizer.accumulation_divisor must equal positive "
            "training.gradient_accumulation"
        )
    if raw["step_index"] != 1:
        raise ValueError("training.optimizer.step_index must equal 1")
    betas = raw["betas"]
    if isinstance(betas, (str, bytes)) or not isinstance(betas, Sequence) or len(betas) != 2:
        raise TypeError("training.optimizer.betas must contain exactly two numbers")
    parsed_betas = tuple(float(value) for value in betas)
    if any(not math.isfinite(value) or not 0.0 <= value < 1.0 for value in parsed_betas):
        raise ValueError("training.optimizer.betas must be finite values in [0,1)")
    epsilon = float(raw["epsilon"])
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("training.optimizer.epsilon must be finite and positive")
    switches: dict[str, bool] = {}
    for name in ("foreach", "fused", "capturable", "maximize", "amsgrad"):
        value = raw[name]
        if not isinstance(value, bool):
            raise TypeError(f"training.optimizer.{name} must be a boolean")
        switches[name] = value
    if switches["foreach"] and switches["fused"]:
        raise ValueError("training.optimizer.foreach and fused cannot both be true")
    return {"betas": parsed_betas, "eps": epsilon, **switches}


def v18_stage_execution_metadata(config: Mapping[str, object]) -> dict[str, object] | None:
    """Return the exact staged-resume intent for the pinned V18 screen.

    Ordinary experiments have no ``v18_screen`` block and retain their legacy
    checkpoint schema.  A V18 checkpoint records the operational subset of the
    predeclared stage contract so a report-only verifier can prove that epoch 1
    stopped before an optimizer/history resume to the four-update target.
    """

    raw_screen = config.get("v18_screen")
    if raw_screen is None:
        return None
    if not isinstance(raw_screen, Mapping):
        raise TypeError("v18_screen must be a mapping")
    raw_stages = raw_screen.get("execution_stages")
    if not isinstance(raw_stages, Mapping):
        raise TypeError("v18_screen.execution_stages must be a mapping")
    expected_with_prediction = {
        "stage_1_exact_v14_restart_updates": 1,
        "stage_1_stop_required": True,
        "predicted_preflight_state_must_match_epoch_001": True,
        "stage_2_resume_from_epoch": 1,
        "stage_2_load_optimizer_state": True,
        "stage_2_load_history": True,
        "stage_2_target_total_optimizer_updates": 4,
    }
    observed = dict(raw_stages)
    if observed != expected_with_prediction:
        raise ValueError(
            "v18_screen.execution_stages differs from the pinned staged-resume contract: "
            f"expected={expected_with_prediction} observed={observed}"
        )
    return {
        key: value
        for key, value in expected_with_prediction.items()
        if key != "predicted_preflight_state_must_match_epoch_001"
    }


def validate_global_scene_residual_state(
    module: GlobalSceneResidual,
    *,
    expected_parameter_count: int | None,
    context: str,
) -> dict[str, object]:
    """Validate every persisted tensor and the optional declared parameter count."""

    audit = module.validate_structural_state()
    observed = module.parameter_count
    if expected_parameter_count is not None and observed != expected_parameter_count:
        raise ValueError(
            f"Global scene residual parameter-count mismatch during {context}: "
            f"expected={expected_parameter_count} observed={observed}"
        )
    if audit.get("parameter_count") != observed:
        raise RuntimeError("Global scene residual structural audit reported a stale count")
    return dict(audit)


def global_scene_residual_resume_metadata_mismatch(
    metadata: Mapping[str, object],
    module: GlobalSceneResidual,
    *,
    expected_initial_state_sha256: str,
) -> dict[str, object] | None:
    """Compare strict residual provenance fields before restoring resume tensors."""

    mismatches: dict[str, object] = {}
    saved_initial = metadata.get("global_scene_residual_initial_state_sha256")
    if saved_initial != expected_initial_state_sha256:
        mismatches["global_scene_residual_initial_state_sha256"] = {
            "checkpoint": saved_initial,
            "runtime": expected_initial_state_sha256,
        }
    saved_count = metadata.get("global_scene_residual_parameter_count")
    if saved_count != module.parameter_count:
        mismatches["global_scene_residual_parameter_count"] = {
            "checkpoint": saved_count,
            "runtime": module.parameter_count,
        }
    return mismatches or None


def validate_signed_x_scene_residual_state(
    module: SignedXSceneResidual,
    *,
    expected_parameter_count: int | None,
    context: str,
) -> dict[str, object]:
    """Validate the signed-X branch and its explicitly tiny parameter surface."""

    audit = module.validate_structural_state()
    observed = module.parameter_count
    if expected_parameter_count is not None and observed != expected_parameter_count:
        raise ValueError(
            f"Signed-X residual parameter-count mismatch during {context}: "
            f"expected={expected_parameter_count} observed={observed}"
        )
    if audit.get("parameter_count") != observed:
        raise RuntimeError("Signed-X structural audit reported a stale parameter count")
    return dict(audit)


def signed_x_scene_residual_resume_metadata_mismatch(
    metadata: Mapping[str, object],
    module: SignedXSceneResidual,
    *,
    expected_initial_state_sha256: str,
) -> dict[str, object] | None:
    """Compare strict signed-X provenance before restoring resume tensors."""

    mismatches: dict[str, object] = {}
    saved_initial = metadata.get("signed_x_scene_residual_initial_state_sha256")
    if saved_initial != expected_initial_state_sha256:
        mismatches["signed_x_scene_residual_initial_state_sha256"] = {
            "checkpoint": saved_initial,
            "runtime": expected_initial_state_sha256,
        }
    saved_count = metadata.get("signed_x_scene_residual_parameter_count")
    if saved_count != module.parameter_count:
        mismatches["signed_x_scene_residual_parameter_count"] = {
            "checkpoint": saved_count,
            "runtime": module.parameter_count,
        }
    return mismatches or None


def dense_alignment_resume_metadata_mismatch(
    metadata: Mapping[str, object],
    module: DenseAlignmentResidual,
    *,
    expected_initial_state_sha256: str,
) -> dict[str, object] | None:
    """Compare strict dense-alignment provenance before restoring resume tensors."""

    mismatches: dict[str, object] = {}
    saved_initial = metadata.get("dense_alignment_initial_state_sha256")
    if saved_initial != expected_initial_state_sha256:
        mismatches["dense_alignment_initial_state_sha256"] = {
            "checkpoint": saved_initial,
            "runtime": expected_initial_state_sha256,
        }
    saved_count = metadata.get("dense_alignment_parameter_count")
    if saved_count != module.parameter_count:
        mismatches["dense_alignment_parameter_count"] = {
            "checkpoint": saved_count,
            "runtime": module.parameter_count,
        }
    return mismatches or None


def validate_dense_alignment_calibration_audit(
    audit: Mapping[str, object],
    module: DenseAlignmentResidual,
    *,
    expected_initial_state_sha256: str,
    expected_calibration_final_state_sha256: str | None = None,
) -> None:
    """Validate immutable calibration provenance before fresh or resumed QA.

    Before QA update one, the module must still equal the calibration-final
    state.  On resume the module is expected to contain later QA updates, so
    the immutable calibration hash is instead bound to initialization
    provenance; the top-level checkpoint state hash independently binds the
    current module.
    """

    require_dense_alignment_calibration_authorized(audit)
    calibration_final = audit.get("final_state_sha256")
    required_calibration_final = (
        module.state_sha256()
        if expected_calibration_final_state_sha256 is None
        else expected_calibration_final_state_sha256
    )
    expected = {
        "initial_state_sha256": expected_initial_state_sha256,
        "final_state_sha256": required_calibration_final,
        "pair_optimizer_state_empty_before_warmup": True,
        "pair_optimizer_rebuilt_after_warmup": True,
        "pair_optimizer_state_empty_after_warmup": True,
        "pair_optimizer_steps_before_qa": 0,
        "held_out_scene_gradient_access": False,
        "category_text_prototypes_serialized": False,
        "oracle_payload_retained": False,
    }
    mismatches = {
        key: {"checkpoint": audit.get(key), "runtime": value}
        for key, value in expected.items()
        if audit.get(key) != value
    }
    training = audit.get("training")
    if not isinstance(training, Mapping) or training.get("final_state_sha256") != (
        calibration_final
    ):
        mismatches["training.final_state_sha256"] = {
            "checkpoint": (
                None if not isinstance(training, Mapping) else training.get("final_state_sha256")
            ),
            "runtime": calibration_final,
        }
    if mismatches:
        raise ValueError(f"Dense-alignment calibration audit mismatch: {mismatches}")


def verify_initialization_artifact_hashes(
    checkpoint: Path,
    *,
    expected_adapter_sha256: str | None,
    expected_metadata_sha256: str | None,
) -> dict[str, str]:
    """Fail before tensor load if a staged checkpoint differs from its pin."""

    observed = {
        "adapter_sha256": file_sha256(checkpoint / "adapter.safetensors"),
        "metadata_sha256": file_sha256(checkpoint / "metadata.json"),
    }
    expected = {
        "adapter_sha256": expected_adapter_sha256,
        "metadata_sha256": expected_metadata_sha256,
    }
    mismatches = {
        key: {"expected": value, "observed": observed[key]}
        for key, value in expected.items()
        if value is not None and observed[key] != value
    }
    if mismatches:
        raise ValueError(f"Initialization checkpoint content hash mismatch: {mismatches}")
    return observed


def legacy_lora_bank_source_mismatch(
    metadata: Mapping[str, object], bank: InstalledLoRABank
) -> dict[str, object] | None:
    """Compare a schema-1 checkpoint's architecture with one frozen named bank."""

    settings = bank.settings
    installation = bank.installation
    if settings.trainable:
        return {"target_bank_trainable": {"required": False, "runtime": True}}
    observed = metadata.get("lora")
    expected = settings.adapter.contract()
    if not isinstance(observed, Mapping):
        return {"lora": {"checkpoint": observed, "runtime": expected}}
    keys = ("schema_version", "enabled", "rank", "alpha", "dropout", "target_modules")
    mismatches = {
        key: {"checkpoint": observed.get(key), "runtime": expected.get(key)}
        for key in keys
        if observed.get(key) != expected.get(key)
    }
    source_count = observed.get("adapter_parameter_count")
    if source_count != installation.parameter_count:
        mismatches["adapter_parameter_count"] = {
            "checkpoint": source_count,
            "runtime": installation.parameter_count,
        }
    expected_hash = settings.expected_initial_state_sha256
    if expected_hash is not None and metadata.get("lora_state_sha256") != expected_hash:
        mismatches["lora_state_sha256"] = {
            "checkpoint": metadata.get("lora_state_sha256"),
            "runtime": expected_hash,
        }
    return mismatches or None


def assert_zero_output_lora_banks(
    collection: LoRABankCollection, *, exclude: Sequence[str] = ()
) -> None:
    """Require every selected bank to be an exact zero-residual initialization."""

    excluded = set(exclude)
    nonzero = [
        f"{bank.settings.name}:{name}"
        for bank in collection.banks
        if bank.settings.name not in excluded
        for name, adapter in zip(
            bank.installation.target_names, bank.installation.adapters, strict=True
        )
        if torch.count_nonzero(adapter.lora_b).item() != 0
    ]
    if nonzero:
        raise ValueError(f"New LoRA bank B tensors are not exact zero-output: {nonzero}")


def staged_legacy_lora_checkpoint_modules(
    scene_modules: Mapping[str, torch.nn.Module], bank: InstalledLoRABank
) -> dict[str, torch.nn.Module]:
    """Alias only a schema-1 ``lora`` payload into one named frozen bank."""

    if bank.settings.trainable:
        raise ValueError("A staged legacy LoRA source bank must be frozen")
    required = {"scene_model", "composer", "grounding"}
    if set(scene_modules) != required:
        raise ValueError(
            "Staged legacy initialization requires exactly the scene checkpoint modules: "
            f"expected={sorted(required)} observed={sorted(scene_modules)}"
        )
    return {**scene_modules, "lora": bank.installation.state_module}


def named_lora_freeze_transition_mismatch(
    metadata: Mapping[str, object], collection: LoRABankCollection | None
) -> dict[str, object] | None:
    """Validate a weights-only transition from trainable source banks to frozen banks.

    Only trainability and initialization provenance may change.  Bank names,
    ranks, shapes, targets, and the exact source state hashes must match.
    """

    if collection is None or collection.settings.legacy_single_bank:
        return {"collection": "named LoRA banks are required"}
    source_contract = metadata.get("lora")
    if not isinstance(source_contract, Mapping) or source_contract.get("schema_version") != 2:
        return {"checkpoint_lora": source_contract}
    source_records = source_contract.get("banks")
    if not isinstance(source_records, list):
        return {"checkpoint_banks": source_records}
    source_by_name = {
        str(record.get("name")): record
        for record in source_records
        if isinstance(record, Mapping) and isinstance(record.get("name"), str)
    }
    expected_names = {bank.settings.name for bank in collection.banks}
    mismatches: dict[str, object] = {}
    if set(source_by_name) != expected_names:
        mismatches["bank_names"] = {
            "checkpoint": sorted(source_by_name),
            "runtime": sorted(expected_names),
        }
    source_hashes = metadata.get("lora_bank_state_sha256")
    if not isinstance(source_hashes, Mapping):
        mismatches["lora_bank_state_sha256"] = source_hashes
        source_hashes = {}
    source_wrapped = metadata.get("lora_bank_wrapped_modules")
    source_counts = metadata.get("lora_bank_parameter_counts")
    for bank in collection.banks:
        name = bank.settings.name
        record = source_by_name.get(name)
        if record is None:
            continue
        if bank.settings.trainable:
            mismatches[f"{name}.trainable"] = {
                "checkpoint": record.get("trainable"),
                "runtime": True,
                "required": False,
            }
        expected_architecture = {
            "rank": bank.settings.adapter.rank,
            "alpha": bank.settings.adapter.alpha,
            "dropout": bank.settings.adapter.dropout,
            "target_modules": list(bank.settings.adapter.target_modules),
            "adapter_parameter_count": bank.installation.parameter_count,
        }
        observed_architecture = {key: record.get(key) for key in expected_architecture}
        if observed_architecture != expected_architecture:
            mismatches[f"{name}.architecture"] = {
                "checkpoint": observed_architecture,
                "runtime": expected_architecture,
            }
        expected_hash = bank.settings.expected_initial_state_sha256
        source_hash = source_hashes.get(name)
        if source_hash != expected_hash:
            mismatches[f"{name}.source_state"] = {
                "checkpoint": source_hash,
                "runtime_expected": expected_hash,
            }
        if isinstance(source_wrapped, Mapping) and source_wrapped.get(name) != list(
            bank.installation.target_names
        ):
            mismatches[f"{name}.wrapped_modules"] = source_wrapped.get(name)
        if isinstance(source_counts, Mapping) and source_counts.get(name) != (
            bank.installation.parameter_counts
        ):
            mismatches[f"{name}.parameter_counts"] = source_counts.get(name)
    return mismatches or None


def validate_named_lora_freeze_transition_state(
    metadata: Mapping[str, object], collection: LoRABankCollection
) -> None:
    expected_hashes = metadata.get("lora_bank_state_sha256")
    observed_hashes = collection.state_sha256()
    if expected_hashes != observed_hashes:
        raise ValueError(
            "Named LoRA freeze-transition state mismatch or tamper detected: "
            f"checkpoint={expected_hashes} runtime={observed_hashes}"
        )
    if any(parameter.requires_grad for parameter in collection.all_parameters()):
        raise RuntimeError("A source LoRA bank remained trainable after freeze transition")


def dense_alignment_source_checkpoint_modules(
    checkpoint_modules: Mapping[str, torch.nn.Module],
) -> dict[str, torch.nn.Module]:
    """Exclude only the fresh dense residual from a named-bank source load.

    The source must still populate the complete frozen scene/composer/grounding,
    global residual, signed-X residual, and named-LoRA stack.  This narrow helper
    prevents a future source migration from silently dropping another module.
    """

    expected_fresh_module = "dense_aligner"
    if expected_fresh_module not in checkpoint_modules:
        raise ValueError("Dense-alignment source transition requires a fresh dense module")
    modules = {
        name: module
        for name, module in checkpoint_modules.items()
        if name != expected_fresh_module
    }
    required = {
        "scene_model",
        "composer",
        "grounding",
        "global_scene_residual",
        "signed_x_scene_residual",
    }
    missing = sorted(required - set(modules))
    if missing:
        raise ValueError(
            "Dense-alignment source transition is missing frozen source modules: "
            f"{missing}"
        )
    return modules


def named_lora_extension_transition_mismatch(
    metadata: Mapping[str, object], collection: LoRABankCollection | None
) -> dict[str, object] | None:
    """Validate adding zero-output trainable banks to a frozen named-bank source.

    The source checkpoint must contain exactly the banks configured as frozen in
    the destination.  Destination-trainable banks must be new, exact-path LoRA
    banks and are deliberately absent from the source checkpoint.
    """

    if collection is None or collection.settings.legacy_single_bank:
        return {"collection": "named LoRA banks are required"}
    frozen_banks = tuple(bank for bank in collection.banks if not bank.settings.trainable)
    new_banks = tuple(bank for bank in collection.banks if bank.settings.trainable)
    if not frozen_banks or not new_banks:
        return {
            "bank_roles": {
                "frozen": [bank.settings.name for bank in frozen_banks],
                "new_trainable": [bank.settings.name for bank in new_banks],
            }
        }
    source_contract = metadata.get("lora")
    if not isinstance(source_contract, Mapping) or source_contract.get("schema_version") != 2:
        return {"checkpoint_lora": source_contract}
    source_records = source_contract.get("banks")
    if not isinstance(source_records, list):
        return {"checkpoint_banks": source_records}
    expected_source_names = {bank.settings.name for bank in frozen_banks}
    mismatches: dict[str, object] = {}
    source_by_name: dict[str, Mapping[str, object]] = {}
    malformed_records: list[dict[str, object]] = []
    duplicate_names: list[str] = []
    for index, record in enumerate(source_records):
        if not isinstance(record, Mapping):
            malformed_records.append({"index": index, "reason": "not_mapping"})
            continue
        name = record.get("name")
        if not isinstance(name, str) or not name:
            malformed_records.append({"index": index, "reason": "missing_name"})
            continue
        if name in source_by_name:
            duplicate_names.append(name)
            continue
        source_by_name[name] = record
    if len(source_records) != len(expected_source_names):
        mismatches["bank_record_count"] = {
            "checkpoint": len(source_records),
            "runtime_frozen_source": len(expected_source_names),
        }
    if malformed_records:
        mismatches["malformed_bank_records"] = malformed_records
    if duplicate_names:
        mismatches["duplicate_bank_names"] = sorted(set(duplicate_names))
    if set(source_by_name) != expected_source_names:
        mismatches["bank_names"] = {
            "checkpoint": sorted(source_by_name),
            "runtime_frozen_source": sorted(expected_source_names),
        }
    source_hashes = metadata.get("lora_bank_state_sha256")
    if not isinstance(source_hashes, Mapping):
        mismatches["lora_bank_state_sha256"] = source_hashes
        source_hashes = {}
    source_wrapped = metadata.get("lora_bank_wrapped_modules")
    source_counts = metadata.get("lora_bank_parameter_counts")
    for field, value in (
        ("lora_bank_state_sha256", source_hashes),
        ("lora_bank_wrapped_modules", source_wrapped),
        ("lora_bank_parameter_counts", source_counts),
    ):
        if not isinstance(value, Mapping):
            mismatches[field] = value
        elif set(value) != expected_source_names:
            mismatches[f"{field}.keys"] = {
                "checkpoint": sorted(value),
                "runtime": sorted(expected_source_names),
            }
    for bank in frozen_banks:
        name = bank.settings.name
        record = source_by_name.get(name)
        if record is None:
            continue
        if record.get("trainable") is not False:
            mismatches[f"{name}.source_trainable"] = record.get("trainable")
        expected_architecture = {
            "rank": bank.settings.adapter.rank,
            "alpha": bank.settings.adapter.alpha,
            "dropout": bank.settings.adapter.dropout,
            "target_modules": list(bank.settings.adapter.target_modules),
            "initialization_algorithm": bank.settings.initialization_algorithm,
            "initialization_seed": bank.settings.initialization_seed,
            "expected_initial_state_sha256": (bank.settings.expected_initial_state_sha256),
            "adapter_parameter_count": bank.installation.parameter_count,
        }
        expected_record_keys = {"name", "trainable", *expected_architecture}
        if set(record) != expected_record_keys:
            mismatches[f"{name}.record_keys"] = {
                "checkpoint": sorted(record),
                "runtime": sorted(expected_record_keys),
            }
        observed_architecture = {key: record.get(key) for key in expected_architecture}
        if observed_architecture != expected_architecture:
            mismatches[f"{name}.architecture"] = {
                "checkpoint": observed_architecture,
                "runtime": expected_architecture,
            }
        expected_hash = bank.settings.expected_initial_state_sha256
        source_hash = source_hashes.get(name)
        if expected_hash is None or source_hash != expected_hash:
            mismatches[f"{name}.source_state"] = {
                "checkpoint": source_hash,
                "runtime_expected": expected_hash,
            }
        if not isinstance(source_wrapped, Mapping) or source_wrapped.get(name) != list(
            bank.installation.target_names
        ):
            mismatches[f"{name}.wrapped_modules"] = (
                source_wrapped.get(name) if isinstance(source_wrapped, Mapping) else source_wrapped
            )
        if not isinstance(source_counts, Mapping) or source_counts.get(name) != (
            bank.installation.parameter_counts
        ):
            mismatches[f"{name}.parameter_counts"] = (
                source_counts.get(name) if isinstance(source_counts, Mapping) else source_counts
            )
    return mismatches or None


def named_lora_freeze_and_extend_transition_mismatch(
    metadata: Mapping[str, object], collection: LoRABankCollection | None
) -> dict[str, object] | None:
    """Validate freezing existing named banks while adding zero-output banks.

    This explicit transition is stricter than silently relaxing
    :func:`named_lora_extension_transition_mismatch`. The source checkpoint must
    contain exactly the destination's frozen-bank subset. Rank, alpha, dropout,
    target paths, parameter counts, wrapped modules, and current tensor hashes
    remain exact. Only source trainability and initialization provenance may be
    rewritten: each destination source bank must use ``checkpoint_overwrite``
    with its expected state hash pinned to the source's current state. At least
    one source bank must actually transition from trainable to frozen.
    """

    if collection is None or collection.settings.legacy_single_bank:
        return {"collection": "named LoRA banks are required"}
    frozen_banks = tuple(bank for bank in collection.banks if not bank.settings.trainable)
    new_banks = tuple(bank for bank in collection.banks if bank.settings.trainable)
    if not frozen_banks or not new_banks:
        return {
            "bank_roles": {
                "frozen": [bank.settings.name for bank in frozen_banks],
                "new_trainable": [bank.settings.name for bank in new_banks],
            }
        }
    source_contract = metadata.get("lora")
    if not isinstance(source_contract, Mapping) or source_contract.get("schema_version") != 2:
        return {"checkpoint_lora": source_contract}
    source_records = source_contract.get("banks")
    if not isinstance(source_records, list):
        return {"checkpoint_banks": source_records}

    expected_source_names = {bank.settings.name for bank in frozen_banks}
    mismatches: dict[str, object] = {}
    source_by_name: dict[str, Mapping[str, object]] = {}
    malformed_records: list[dict[str, object]] = []
    duplicate_names: list[str] = []
    for index, record in enumerate(source_records):
        if not isinstance(record, Mapping):
            malformed_records.append({"index": index, "reason": "not_mapping"})
            continue
        name = record.get("name")
        if not isinstance(name, str) or not name:
            malformed_records.append({"index": index, "reason": "missing_name"})
            continue
        if name in source_by_name:
            duplicate_names.append(name)
            continue
        source_by_name[name] = record
    if len(source_records) != len(expected_source_names):
        mismatches["bank_record_count"] = {
            "checkpoint": len(source_records),
            "runtime_frozen_source": len(expected_source_names),
        }
    if malformed_records:
        mismatches["malformed_bank_records"] = malformed_records
    if duplicate_names:
        mismatches["duplicate_bank_names"] = sorted(set(duplicate_names))
    if set(source_by_name) != expected_source_names:
        mismatches["bank_names"] = {
            "checkpoint": sorted(source_by_name),
            "runtime_frozen_source": sorted(expected_source_names),
        }

    source_hashes = metadata.get("lora_bank_state_sha256")
    source_wrapped = metadata.get("lora_bank_wrapped_modules")
    source_counts = metadata.get("lora_bank_parameter_counts")
    for field, value in (
        ("lora_bank_state_sha256", source_hashes),
        ("lora_bank_wrapped_modules", source_wrapped),
        ("lora_bank_parameter_counts", source_counts),
    ):
        if not isinstance(value, Mapping):
            mismatches[field] = value
        elif set(value) != expected_source_names:
            mismatches[f"{field}.keys"] = {
                "checkpoint": sorted(value),
                "runtime": sorted(expected_source_names),
            }
    source_hashes = source_hashes if isinstance(source_hashes, Mapping) else {}
    source_wrapped = source_wrapped if isinstance(source_wrapped, Mapping) else {}
    source_counts = source_counts if isinstance(source_counts, Mapping) else {}

    transitioned_trainable_banks: list[str] = []
    required_record_keys = {
        "name",
        "trainable",
        "rank",
        "alpha",
        "dropout",
        "target_modules",
        "initialization_algorithm",
        "initialization_seed",
        "expected_initial_state_sha256",
        "adapter_parameter_count",
    }
    allowed_record_keys = required_record_keys | {"learning_rate", "weight_decay"}
    for bank in frozen_banks:
        name = bank.settings.name
        record = source_by_name.get(name)
        if record is None:
            continue
        if not isinstance(record.get("trainable"), bool):
            mismatches[f"{name}.source_trainable"] = record.get("trainable")
        elif record.get("trainable") is True:
            transitioned_trainable_banks.append(name)
        missing_keys = required_record_keys - set(record)
        unknown_keys = set(record) - allowed_record_keys
        if missing_keys or unknown_keys:
            mismatches[f"{name}.record_keys"] = {
                "missing": sorted(missing_keys),
                "unknown": sorted(unknown_keys),
            }
        expected_architecture = {
            "rank": bank.settings.adapter.rank,
            "alpha": bank.settings.adapter.alpha,
            "dropout": bank.settings.adapter.dropout,
            "target_modules": list(bank.settings.adapter.target_modules),
            "adapter_parameter_count": bank.installation.parameter_count,
        }
        observed_architecture = {key: record.get(key) for key in expected_architecture}
        if observed_architecture != expected_architecture:
            mismatches[f"{name}.architecture"] = {
                "checkpoint": observed_architecture,
                "runtime": expected_architecture,
            }
        if (
            bank.settings.initialization_algorithm != "checkpoint_overwrite"
            or bank.settings.initialization_seed is not None
        ):
            mismatches[f"{name}.destination_provenance"] = {
                "initialization_algorithm": bank.settings.initialization_algorithm,
                "initialization_seed": bank.settings.initialization_seed,
            }
        source_hash = source_hashes.get(name)
        if (
            not isinstance(source_hash, str)
            or bank.settings.expected_initial_state_sha256 != source_hash
        ):
            mismatches[f"{name}.source_state"] = {
                "checkpoint": source_hash,
                "runtime_expected": bank.settings.expected_initial_state_sha256,
            }
        if source_wrapped.get(name) != list(bank.installation.target_names):
            mismatches[f"{name}.wrapped_modules"] = source_wrapped.get(name)
        if source_counts.get(name) != bank.installation.parameter_counts:
            mismatches[f"{name}.parameter_counts"] = source_counts.get(name)
    if not transitioned_trainable_banks:
        mismatches["source_trainable_bank_transition"] = {
            "required": "at least one existing source bank must become frozen",
            "observed": transitioned_trainable_banks,
        }
    return mismatches or None


def named_lora_extension_checkpoint_modules(
    checkpoint_modules: Mapping[str, torch.nn.Module], collection: LoRABankCollection
) -> dict[str, torch.nn.Module]:
    """Exclude new trainable banks while loading the exact frozen source stack."""

    modules = dict(checkpoint_modules)
    for bank in collection.banks:
        if bank.settings.trainable:
            key = f"lora_banks.{bank.settings.name}"
            if modules.pop(key, None) is None:
                raise KeyError(f"Missing new LoRA bank checkpoint module: {key}")
    return modules


def validate_named_lora_extension_transition_state(
    metadata: Mapping[str, object], collection: LoRABankCollection
) -> None:
    """Prove source banks loaded exactly and new banks remained zero-output."""

    source_hashes = metadata.get("lora_bank_state_sha256")
    if not isinstance(source_hashes, Mapping):
        raise TypeError("Named LoRA extension source is missing bank hashes")
    observed_hashes = collection.state_sha256()
    for bank in collection.banks:
        name = bank.settings.name
        observed = observed_hashes[name]
        expected = (
            bank.settings.expected_initial_state_sha256
            if bank.settings.trainable
            else source_hashes.get(name)
        )
        if expected is None or observed != expected:
            raise ValueError(
                "Named LoRA extension state mismatch or tamper detected: "
                f"bank={name} expected={expected} observed={observed}"
            )
        parameters = tuple(bank.installation.parameters())
        if bank.settings.trainable:
            if not parameters or not all(parameter.requires_grad for parameter in parameters):
                raise RuntimeError(f"New LoRA bank {name!r} is not wholly trainable")
        elif any(parameter.requires_grad for parameter in parameters):
            raise RuntimeError(f"Source LoRA bank {name!r} remained trainable")
    assert_zero_output_lora_banks(
        collection,
        exclude=tuple(
            bank.settings.name for bank in collection.banks if not bank.settings.trainable
        ),
    )


def resolve_checkpoint_sources(
    *,
    cli_resume: Path | None,
    cli_initialize_from: Path | None,
    training_config: Mapping[str, object],
) -> tuple[Path | str | None, Path | str | None]:
    """Resolve exact-resume and weights-only initialization inputs.

    An explicit resume continues an existing run and therefore intentionally
    overrides a config's staged ``initialize_from`` default. Supplying both
    command-line modes is still an error, as is an ambiguous config that sets
    both modes without that explicit resume override.
    """

    configured_resume = training_config.get("resume_from")
    configured_initialize = training_config.get("initialize_from")
    if cli_resume is not None:
        if cli_initialize_from is not None:
            raise ValueError("resume_from and initialize_from are mutually exclusive")
        return cli_resume, None
    initialize_value = (
        cli_initialize_from if cli_initialize_from is not None else configured_initialize
    )
    if configured_resume and initialize_value:
        raise ValueError("resume_from and initialize_from are mutually exclusive")
    return configured_resume, initialize_value


def combine_pair_training_losses(
    base_loss: torch.Tensor,
    pair_ranking_loss: torch.Tensor,
    full_vocab_ranking_loss: torch.Tensor,
    diversity_loss: torch.Tensor,
    scene_separation_loss: torch.Tensor,
    *,
    language_loss: torch.Tensor | None = None,
    language_nll_weight: float = 1.0,
    pair_ranking_weight: float,
    full_vocab_ranking_weight: float,
    diversity_weight: float,
    scene_separation_weight: float,
) -> torch.Tensor:
    """Compose the audited pair objective from raw differentiable terms."""

    if language_nll_weight != 1.0 and language_loss is None:
        raise ValueError("language_loss is required when language_nll_weight differs from one")
    weighted_base = base_loss
    if language_nll_weight != 1.0:
        assert language_loss is not None
        weighted_base = base_loss + (float(language_nll_weight) - 1.0) * language_loss

    return (
        weighted_base
        + float(pair_ranking_weight) * pair_ranking_loss
        + float(full_vocab_ranking_weight) * full_vocab_ranking_loss
        + float(diversity_weight) * diversity_loss
        + float(scene_separation_weight) * scene_separation_loss
    )


def validate_pair_objective_training_mode(
    *,
    configured: bool,
    curriculum_enabled: bool,
    pair_only: bool,
) -> None:
    """Keep schema-1 policies from being bypassed by legacy/standard batches."""

    if configured and not curriculum_enabled:
        raise ValueError("Explicit pair objective policies require an enabled pair curriculum")
    if configured and not pair_only:
        raise ValueError(
            "Schema-1 pair objective policies require pair_only training so standard "
            "batches cannot bypass a pair's configured language-NLL weight"
        )


def pair_gate_checkpoint_improved(
    *,
    monitor_value: float,
    best_monitor_value: float,
    min_delta: float,
    gate_passed: bool,
    best_gate_passed: bool,
) -> bool:
    """Prefer a configured gate pass before comparing its scalar monitor.

    This prevents fixed-length training from replacing a passing ``best``
    checkpoint with a later failed checkpoint that happens to have a smaller
    pairwise or full-vocabulary hinge.
    """

    if gate_passed != best_gate_passed:
        return gate_passed
    return monitor_value < best_monitor_value - min_delta


def pair_gate_monitor_value(pair_gate: dict[str, object], *, full_vocab_gate: bool) -> float:
    """Return a lower-is-better gate monitor that remains useful after passing.

    A hinge saturates at zero as soon as the gate passes. For a strict
    full-vocabulary gate, passing checkpoints are instead ranked by the
    negative minimum target-versus-best-other margin. This preserves the first
    genuine pass over every failure while allowing later, more stable passes to
    replace a barely positive checkpoint.
    """

    if not full_vocab_gate:
        return float(pair_gate["ranking_hinge_at_configured_margin"])
    if bool(pair_gate["passed"]):
        return -float(pair_gate["minimum_first_answer_token_target_vs_best_other_logit_margin"])
    return float(pair_gate["first_answer_token_target_vs_best_other_hinge"]) + float(
        pair_gate["ranking_hinge_at_configured_margin"]
    )


def should_stop_after_pair_gate(
    stop_when_gate_passes: bool,
    pair_gate: dict[str, object] | None,
) -> bool:
    """Apply the existing stop policy to the configured composite gate."""

    return bool(
        stop_when_gate_passes and pair_gate is not None and bool(pair_gate.get("passed", False))
    )


def best_pair_gate_passed_from_history(history: Sequence[dict], best_epoch: int | None) -> bool:
    """Recover gate-pass priority from checkpoints written before it was stored."""

    if best_epoch is None:
        return False
    for item in history:
        if int(item.get("epoch", -1)) != best_epoch:
            continue
        gate = item.get("pair_candidate_gate")
        return isinstance(gate, dict) and bool(gate.get("passed", False))
    return False


def construct_scene_tokenizer(
    config: dict, semantic_dim: int, language_hidden_dim: int
) -> SceneTokenizer:
    settings = config["scene_encoder"]
    return SceneTokenizer(
        semantic_dim=semantic_dim,
        model_dim=int(settings["model_dim"]),
        language_hidden_dim=language_hidden_dim,
        block_size_m=float(settings["block_size_m"]),
        tokens_per_block=int(settings["tokens_per_block"]),
        global_latents=int(settings["global_latents"]),
        heads=int(settings["heads"]),
        global_layers=int(settings["global_layers"]),
        fourier_bands=int(settings["fourier_bands"]),
        coverage_temperature=float(settings.get("coverage_temperature", 0.35)),
        coverage_scale=float(settings.get("coverage_scale", 1.0)),
        query_identity_scale=float(settings.get("query_identity_scale", 0.5)),
        projection_skip_scale=float(settings.get("projection_skip_scale", 1.0)),
        semantic_skip_scale=float(settings.get("semantic_skip_scale", 1.0)),
        geometry_skip_scale=float(settings.get("geometry_skip_scale", 0.5)),
        block_content_residual_scale=float(settings.get("block_content_residual_scale", 1.0)),
        language_aligned_tail_dim=int(settings.get("language_aligned_tail_dim", 0)),
        native_aligned_coverage_scale=float(settings.get("native_aligned_coverage_scale", 0.0)),
        learned_scene_token_scale=float(settings.get("learned_scene_token_scale", 1.0)),
        learned_scene_token_rms_target=settings.get("learned_scene_token_rms_target"),
        architecture_version=str(settings["architecture_version"]),
    )


def build_adapter_optimizer(
    config: dict,
    scene_parameters: Sequence[torch.nn.Parameter],
    lora_installation: LoRABankCollection | LoRAInstallation | None,
    configured_lora_optimizer: LoRAOptimizerSettings | None,
    *,
    dense_alignment_parameters: Sequence[torch.nn.Parameter] = (),
) -> tuple[torch.optim.AdamW, list[torch.nn.Parameter]]:
    """Build strict scene/LoRA/dense groups, omitting frozen surfaces."""

    scene_parameters = [parameter for parameter in scene_parameters if parameter.requires_grad]
    dense_alignment_parameters = [
        parameter for parameter in dense_alignment_parameters if parameter.requires_grad
    ]
    adamw_options = explicit_adamw_options(config)
    if dense_alignment_parameters:
        lora_parameters = [] if lora_installation is None else lora_installation.parameters()
        if scene_parameters or lora_parameters:
            raise ValueError(
                "Dense-alignment-only optimizer cannot include scene or language-LoRA parameters"
            )
        learning_rate = float(config["training"]["dense_alignment_learning_rate"])
        weight_decay = float(config["training"]["dense_alignment_weight_decay"])
        if not math.isfinite(learning_rate) or learning_rate <= 0.0:
            raise ValueError("training.dense_alignment_learning_rate must be finite and positive")
        if not math.isfinite(weight_decay) or weight_decay < 0.0:
            raise ValueError(
                "training.dense_alignment_weight_decay must be finite and nonnegative"
            )
        optimizer = torch.optim.AdamW(
            [
                {
                    "name": "dense_alignment",
                    "params": dense_alignment_parameters,
                    "lr": learning_rate,
                    "weight_decay": weight_decay,
                }
            ],
            **adamw_options,
        )
        return optimizer, dense_alignment_parameters
    if lora_installation is None:
        if not scene_parameters:
            raise ValueError("Adapter optimizer has no trainable parameters")
        optimizer = torch.optim.AdamW(
            scene_parameters,
            lr=float(config["training"]["learning_rate"]),
            weight_decay=float(config["training"]["weight_decay"]),
            **adamw_options,
        )
        return optimizer, scene_parameters
    lora_parameters = lora_installation.parameters()
    if lora_parameters and configured_lora_optimizer is None:
        raise ValueError("Trainable LoRA requires explicit optimizer settings")
    groups: list[dict] = []
    if scene_parameters:
        groups.append(
            {
                "name": (
                    "signed_x_output_projection"
                    if config["training"].get("train_signed_x_scene_residual_only", False)
                    else (
                        "global_scene_residual"
                        if config["training"].get("train_global_scene_residual_only", False)
                        else "scene_adapter"
                    )
                ),
                "params": scene_parameters,
                "lr": float(config["training"]["learning_rate"]),
                "weight_decay": float(config["training"]["weight_decay"]),
            }
        )
    if lora_parameters:
        assert configured_lora_optimizer is not None
        groups.append(
            {
                "name": "language_lora",
                "params": lora_parameters,
                "lr": configured_lora_optimizer.learning_rate,
                "weight_decay": configured_lora_optimizer.weight_decay,
            }
        )
    if not groups:
        raise ValueError("Adapter optimizer has no trainable parameters")
    optimizer = torch.optim.AdamW(groups, **adamw_options)
    return optimizer, scene_parameters + lora_parameters


def parameter_gradient_l2(parameters: Sequence[torch.nn.Parameter]) -> float:
    """Return the finite FP32 L2 norm over currently populated gradients."""

    squared = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        gradient = parameter.grad.detach().float()
        if not torch.isfinite(gradient).all():
            raise RuntimeError("Trainable adapter gradient contains NaN or infinity")
        squared += float(torch.sum(gradient.square()).cpu())
    return math.sqrt(squared)


def scene_token_mixing_settings(config: dict) -> dict[str, int | float | None]:
    settings = config["scene_encoder"]
    return {
        "language_aligned_tail_dim": int(settings.get("language_aligned_tail_dim", 0)),
        "native_aligned_coverage_scale": float(settings.get("native_aligned_coverage_scale", 0.0)),
        "learned_scene_token_scale": float(settings.get("learned_scene_token_scale", 1.0)),
        "learned_scene_token_rms_target": (
            None
            if settings.get("learned_scene_token_rms_target") is None
            else float(settings["learned_scene_token_rms_target"])
        ),
    }


def tokenize_answer(tokenizer, answer: str, device: torch.device) -> torch.Tensor:
    suffix = answer.strip() + (tokenizer.eos_token or "")
    return tokenizer(suffix, add_special_tokens=False, return_tensors="pt").input_ids.to(device)


def forward_prefix_batch(language, batch: PrefixBatch):
    """Forward one padded batch while retaining compatibility with test doubles."""

    method = getattr(language, "forward_prefix_batch", None)
    if method is not None:
        return method(batch, use_cache=False)
    return language.model(
        inputs_embeds=batch.inputs_embeds,
        attention_mask=batch.attention_mask,
        use_cache=False,
    )


def map_forward(
    model: SceneTokenizer,
    data: MapTensorData,
    global_scene_residual: GlobalSceneResidual | None = None,
    signed_x_scene_residual: SignedXSceneResidual | None = None,
    dense_aligner: torch.nn.Module | None = None,
    dense_sidecar_adapter: DenseSidecarAdapter | None = None,
):
    aligned_sidecar = None
    aligned_sidecar_scale = 0.0
    if dense_aligner is None:
        semantic = data.semantic
    elif isinstance(dense_aligner, DenseAlignmentResidual):
        semantic, aligned_sidecar, aligned_sidecar_scale = dense_aligner.scene_inputs(
            data.semantic
        )
    else:
        semantic = dense_aligner(data.semantic)
    if semantic.shape != data.semantic.shape:
        raise ValueError(
            "Dense aligner must preserve the complete semantic tensor shape: "
            f"input={tuple(data.semantic.shape)} output={tuple(semantic.shape)}"
        )
    if not torch.isfinite(semantic).all():
        raise ValueError("Dense aligner produced NaN or infinity")
    scene_arguments = (
        semantic,
        data.xyz,
        data.rgb,
        data.normal,
        data.confidence,
        data.observation_count,
        data.room_min,
        data.room_max,
    )
    output = (
        model(*scene_arguments)
        if aligned_sidecar is None
        else model(
            *scene_arguments,
            aligned_sidecar=aligned_sidecar,
            aligned_sidecar_scale=aligned_sidecar_scale,
        )
    )
    centered_content = None
    if signed_x_scene_residual is not None:
        if global_scene_residual is None:
            raise ValueError("Signed-X scene residual requires the frozen V18 residual base")
        centered_content = frozen_v18_centered_content_values(
            global_scene_residual,
            output.scene_tokens,
        )
    output = apply_global_scene_residual(output, global_scene_residual)
    if signed_x_scene_residual is not None:
        assert centered_content is not None
        output = apply_signed_x_scene_residual(
            output,
            signed_x_scene_residual,
            centered_content,
        )
    return apply_dense_sidecar_adapter(output, dense_sidecar_adapter)


def training_map_forward(
    model: SceneTokenizer,
    data: MapTensorData,
    *,
    freeze_scene_adapter: bool,
    global_scene_residual: GlobalSceneResidual | None = None,
    signed_x_scene_residual: SignedXSceneResidual | None = None,
    dense_aligner: torch.nn.Module | None = None,
    dense_sidecar_adapter: DenseSidecarAdapter | None = None,
):
    """Encode frozen scene tokens without creating inference tensors.

    ``torch.inference_mode`` is deliberately not used: the resulting tokens
    are constants, but the decoder must still save them for LoRA weight
    gradients.
    """

    dense_alignment_trainable = dense_aligner is not None and any(
        parameter.requires_grad for parameter in dense_aligner.parameters()
    )
    dense_sidecar_trainable = dense_sidecar_adapter is not None and any(
        parameter.requires_grad for parameter in dense_sidecar_adapter.parameters()
    )
    if dense_sidecar_adapter is not None and freeze_scene_adapter:
        # Build the complete immutable V24 stack and calibrated all-voxel field
        # without retaining its graph, then open gradients only through the
        # explicitly trainable post-stack adapter.
        with torch.no_grad():
            frozen_output = map_forward(
                model,
                data,
                global_scene_residual,
                signed_x_scene_residual,
                dense_aligner,
            )
        if dense_sidecar_trainable:
            return apply_dense_sidecar_adapter(frozen_output, dense_sidecar_adapter)
        with torch.no_grad():
            return apply_dense_sidecar_adapter(frozen_output, dense_sidecar_adapter)
    if not freeze_scene_adapter or dense_alignment_trainable:
        return map_forward(
            model,
            data,
            global_scene_residual,
            signed_x_scene_residual,
            dense_aligner,
            dense_sidecar_adapter,
        )
    with torch.no_grad():
        output = map_forward(model, data, dense_aligner=dense_aligner)
        if signed_x_scene_residual is not None:
            if global_scene_residual is None:
                raise ValueError("Signed-X scene residual requires the frozen V18 residual base")
            centered_content = frozen_v18_centered_content_values(
                global_scene_residual,
                output.scene_tokens,
            )
            output = apply_global_scene_residual(output, global_scene_residual)
        else:
            centered_content = None
    if signed_x_scene_residual is not None:
        assert centered_content is not None
        return apply_signed_x_scene_residual(
            output,
            signed_x_scene_residual,
            centered_content,
        )
    return apply_global_scene_residual(output, global_scene_residual)


def verify_zero_output_scene_residual_equivalence(
    scene_model: SceneTokenizer,
    global_scene_residual: GlobalSceneResidual,
    composer: ContinuousPrefixComposer,
    maps: Mapping[str, MapTensorData],
    *,
    model_dtype: torch.dtype,
) -> dict[str, object]:
    """Prove that a fresh residual preserves every selected scene prefix exactly."""

    previous_modes = (
        scene_model.training,
        global_scene_residual.training,
        composer.training,
    )
    scene_model.eval()
    global_scene_residual.eval()
    composer.eval()
    scene_hashes: dict[str, dict[str, str]] = {}
    try:
        with torch.inference_mode():
            for scene_id in sorted(maps):
                core = map_forward(scene_model, maps[scene_id])
                adapted = apply_global_scene_residual(core, global_scene_residual)
                if not torch.equal(core.scene_tokens, adapted.scene_tokens):
                    raise RuntimeError(
                        f"Fresh global scene residual changed update-0 tokens for {scene_id}"
                    )
                core_prefix = composer.scene_prefix(core.scene_tokens.to(dtype=model_dtype))
                adapted_prefix = composer.scene_prefix(adapted.scene_tokens.to(dtype=model_dtype))
                if not torch.equal(core_prefix, adapted_prefix):
                    raise RuntimeError(
                        f"Fresh global scene residual changed update-0 prefix for {scene_id}"
                    )
                core_hash = prefix_sha256(core_prefix)
                adapted_hash = prefix_sha256(adapted_prefix)
                if core_hash != adapted_hash:
                    raise RuntimeError(
                        f"Fresh global scene residual changed update-0 prefix hash for {scene_id}"
                    )
                scene_hashes[scene_id] = {
                    "core_prefix_sha256": core_hash,
                    "adapted_prefix_sha256": adapted_hash,
                }
    finally:
        scene_model.train(previous_modes[0])
        global_scene_residual.train(previous_modes[1])
        composer.train(previous_modes[2])
    return {
        "verified": True,
        "question_dependent_scene_processing": False,
        "scene_count": len(scene_hashes),
        "scene_prefixes": scene_hashes,
    }


def verify_zero_output_signed_x_residual_equivalence(
    scene_model: SceneTokenizer,
    global_scene_residual: GlobalSceneResidual,
    signed_x_scene_residual: SignedXSceneResidual,
    composer: ContinuousPrefixComposer,
    maps: Mapping[str, MapTensorData],
    *,
    model_dtype: torch.dtype,
) -> dict[str, object]:
    """Prove that a fresh signed-X branch exactly preserves the loaded V18 base."""

    modules: tuple[torch.nn.Module, ...] = (
        scene_model,
        global_scene_residual,
        signed_x_scene_residual,
        composer,
    )
    previous_modes = tuple(module.training for module in modules)
    for module in modules:
        module.eval()
    scene_hashes: dict[str, dict[str, str]] = {}
    try:
        with torch.inference_mode():
            for scene_id in sorted(maps):
                core = map_forward(scene_model, maps[scene_id])
                centered_content = frozen_v18_centered_content_values(
                    global_scene_residual,
                    core.scene_tokens,
                )
                base = apply_global_scene_residual(core, global_scene_residual)
                adapted = apply_signed_x_scene_residual(
                    base,
                    signed_x_scene_residual,
                    centered_content,
                )
                if not torch.equal(base.scene_tokens, adapted.scene_tokens):
                    raise RuntimeError(
                        f"Fresh signed-X residual changed update-0 tokens for {scene_id}"
                    )
                base_prefix = composer.scene_prefix(base.scene_tokens.to(dtype=model_dtype))
                adapted_prefix = composer.scene_prefix(adapted.scene_tokens.to(dtype=model_dtype))
                if not torch.equal(base_prefix, adapted_prefix):
                    raise RuntimeError(
                        f"Fresh signed-X residual changed update-0 prefix for {scene_id}"
                    )
                base_hash = prefix_sha256(base_prefix)
                adapted_hash = prefix_sha256(adapted_prefix)
                if base_hash != adapted_hash:
                    raise RuntimeError(
                        f"Fresh signed-X residual changed update-0 prefix hash for {scene_id}"
                    )
                scene_hashes[scene_id] = {
                    "v18_base_prefix_sha256": base_hash,
                    "signed_x_adapted_prefix_sha256": adapted_hash,
                }
    finally:
        for module, was_training in zip(modules, previous_modes, strict=True):
            module.train(was_training)
    return {
        "verified": True,
        "base": "loaded_frozen_global_scene_residual",
        "question_dependent_scene_processing": False,
        "all_scene_slots_accounted": True,
        "scene_count": len(scene_hashes),
        "scene_prefixes": scene_hashes,
    }


def verify_zero_output_dense_alignment_equivalence(
    scene_model: SceneTokenizer,
    global_scene_residual: GlobalSceneResidual,
    signed_x_scene_residual: SignedXSceneResidual,
    dense_aligner: DenseAlignmentResidual,
    composer: ContinuousPrefixComposer,
    maps: Mapping[str, MapTensorData],
    *,
    model_dtype: torch.dtype,
) -> dict[str, object]:
    """Prove a fresh dense residual preserves the complete loaded scene stack.

    This runs before the first question/update.  The source and adapted paths
    see every voxel and differ only by insertion of the exact-zero dense
    residual; neither path receives a question or oracle-derived selector.
    """

    modules: tuple[torch.nn.Module, ...] = (
        scene_model,
        global_scene_residual,
        signed_x_scene_residual,
        dense_aligner,
        composer,
    )
    previous_modes = tuple(module.training for module in modules)
    for module in modules:
        module.eval()
    scene_hashes: dict[str, dict[str, str]] = {}
    try:
        with torch.inference_mode():
            for scene_id in sorted(maps):
                data = maps[scene_id]
                semantic_version = data.semantic._version
                aligned_semantic = dense_aligner(data.semantic)
                if not torch.equal(data.semantic, aligned_semantic):
                    raise RuntimeError(
                        f"Fresh dense alignment changed update-0 voxel features for {scene_id}"
                    )
                if data.semantic._version != semantic_version:
                    raise RuntimeError(f"Dense alignment mutated the source map for {scene_id}")
                base = map_forward(
                    scene_model,
                    data,
                    global_scene_residual,
                    signed_x_scene_residual,
                )
                adapted = map_forward(
                    scene_model,
                    data,
                    global_scene_residual,
                    signed_x_scene_residual,
                    dense_aligner,
                )
                if data.semantic._version != semantic_version:
                    raise RuntimeError(f"Dense-aligned map forwarding mutated {scene_id}")
                if not torch.equal(base.scene_tokens, adapted.scene_tokens):
                    raise RuntimeError(
                        f"Fresh dense alignment changed update-0 scene tokens for {scene_id}"
                    )
                base_prefix = composer.scene_prefix(base.scene_tokens.to(dtype=model_dtype))
                adapted_prefix = composer.scene_prefix(adapted.scene_tokens.to(dtype=model_dtype))
                if not torch.equal(base_prefix, adapted_prefix):
                    raise RuntimeError(
                        f"Fresh dense alignment changed update-0 prefix for {scene_id}"
                    )
                base_hash = prefix_sha256(base_prefix)
                adapted_hash = prefix_sha256(adapted_prefix)
                if base_hash != adapted_hash:
                    raise RuntimeError(
                        f"Fresh dense alignment changed update-0 prefix hash for {scene_id}"
                    )
                scene_hashes[scene_id] = {
                    "frozen_source_prefix_sha256": base_hash,
                    "dense_aligned_prefix_sha256": adapted_hash,
                }
    finally:
        for module, was_training in zip(modules, previous_modes, strict=True):
            module.train(was_training)
    return {
        "verified": True,
        "base": "loaded_frozen_scene_global_signed_x_and_named_lora_stack",
        "question_dependent_scene_processing": False,
        "all_voxels_transformed": True,
        "source_map_mutated": False,
        "scene_count": len(scene_hashes),
        "scene_prefixes": scene_hashes,
    }


def select_training_records(
    records: Sequence[QARecord],
    *,
    max_questions: int | None = None,
    max_questions_per_scene: int | None = None,
) -> list[QARecord]:
    """Apply deterministic, answer-stratified, counterfactual-safe caps.

    The dataset is shuffled independently for each scene, so taking the first
    ``N`` records can omit a rare answer type and can select only one side of a
    counterfactual question.  Selection therefore treats every changed
    counterfactual key as an indivisible two-scene unit, then fills each scene
    by repeatedly choosing its least represented answer type.  No record from
    validation or test can enter here because the caller supplies only the
    persisted training split.
    """

    if max_questions is not None and max_questions < 1:
        raise ValueError("max_questions must be positive")
    if max_questions_per_scene is not None and max_questions_per_scene < 1:
        raise ValueError("max_questions_per_scene must be positive")
    indexed_by_scene: defaultdict[str, list[tuple[int, QARecord]]] = defaultdict(list)
    for index, record in enumerate(records):
        indexed_by_scene[record.scene_id].append((index, record))

    required_indices: set[int] = set()
    changed_units: defaultdict[tuple[str, str], list[tuple[int, QARecord]]] = defaultdict(list)
    for index, record in enumerate(records):
        if record.counterfactual_expected_change is not True:
            continue
        if not record.counterfactual_pair_id or not record.counterfactual_question_key:
            raise ValueError(
                f"Changed counterfactual record {record.question_id} lacks pair metadata"
            )
        changed_units[(record.counterfactual_pair_id, record.counterfactual_question_key)].append(
            (index, record)
        )
    for unit_key, members in changed_units.items():
        member_scenes = {record.scene_id for _, record in members}
        roles = {record.counterfactual_role for _, record in members}
        if (
            len(members) != 2
            or len(member_scenes) != 2
            or roles
            != {
                "reference",
                "counterfactual",
            }
        ):
            raise ValueError(
                "Changed counterfactual unit must contain one record from each paired "
                f"training scene: {unit_key}"
            )
        required_indices.update(index for index, _ in members)

    selected_indices: set[int] = set()
    for scene_id, indexed_records in indexed_by_scene.items():
        answer_types = sorted({record.answer_type for _, record in indexed_records})
        cap = (
            len(indexed_records)
            if max_questions_per_scene is None
            else min(max_questions_per_scene, len(indexed_records))
        )
        scene_required = [
            (index, record) for index, record in indexed_records if index in required_indices
        ]
        if len(scene_required) > cap:
            raise ValueError(
                f"Scene {scene_id} has {len(scene_required)} changed counterfactual "
                f"records but its cap is {cap}"
            )
        if len(answer_types) > cap:
            raise ValueError(
                f"Scene {scene_id} has {len(answer_types)} answer types but its cap is "
                f"only {cap}; answer-type coverage is impossible"
            )
        scene_selected = {index for index, _ in scene_required}
        by_type: dict[str, list[tuple[int, QARecord]]] = {
            answer_type: [
                (index, record)
                for index, record in indexed_records
                if record.answer_type == answer_type
            ]
            for answer_type in answer_types
        }

        # First guarantee one example of every answer type, preferring the
        # earliest record in the persisted deterministic dataset order.
        for answer_type in sorted(answer_types, key=lambda value: (len(by_type[value]), value)):
            if any(index in scene_selected for index, _ in by_type[answer_type]):
                continue
            scene_selected.add(by_type[answer_type][0][0])

        # Fill the remaining budget from the currently least represented type.
        # This preserves all scarce support/metric/count examples instead of
        # allowing the much larger spatial-relation category to dominate.
        while len(scene_selected) < cap:
            candidates: list[tuple[int, int, str, int]] = []
            for answer_type, type_records in by_type.items():
                unselected = [index for index, _ in type_records if index not in scene_selected]
                if not unselected:
                    continue
                selected_count = len(type_records) - len(unselected)
                candidates.append((selected_count, len(type_records), answer_type, unselected[0]))
            if not candidates:
                break
            _, _, _, next_index = min(candidates)
            scene_selected.add(next_index)
        selected_indices.update(scene_selected)

    selected = [record for index, record in enumerate(records) if index in selected_indices]
    if max_questions is not None:
        selected = selected[:max_questions]
        selected_ids = {id(record) for record in selected}
        for unit_key, members in changed_units.items():
            included = [id(record) in selected_ids for _, record in members]
            if any(included) and not all(included):
                raise ValueError(
                    "Global max_questions split changed counterfactual unit "
                    f"{unit_key}; use max_questions_per_scene or increase the global cap"
                )
    return selected


def training_selection_summary(available: Sequence[QARecord], selected: Sequence[QARecord]) -> dict:
    """Return aggregate selection evidence without questions or answer labels."""

    def aggregate(records: Sequence[QARecord]) -> dict[str, dict]:
        by_scene: defaultdict[str, list[QARecord]] = defaultdict(list)
        for record in records:
            by_scene[record.scene_id].append(record)
        result: dict[str, dict] = {}
        for scene_id, scene_records in sorted(by_scene.items()):
            answer_types: defaultdict[str, int] = defaultdict(int)
            changed_keys: set[str] = set()
            pair_ids: set[str] = set()
            for record in scene_records:
                answer_types[record.answer_type] += 1
                if record.counterfactual_pair_id:
                    pair_ids.add(record.counterfactual_pair_id)
                if (
                    record.counterfactual_expected_change is True
                    and record.counterfactual_question_key
                ):
                    changed_keys.add(record.counterfactual_question_key)
            result[scene_id] = {
                "count": len(scene_records),
                "answer_types": dict(sorted(answer_types.items())),
                "counterfactual_pair_ids": sorted(pair_ids),
                "expected_change_key_count": len(changed_keys),
            }
        return result

    selected_identifiers = "\n".join(
        f"{record.scene_id}:{record.question_id}" for record in selected
    )
    summary = {
        "schema_version": 1,
        "available_count": len(available),
        "selected_count": len(selected),
        "available_by_scene": aggregate(available),
        "selected_by_scene": aggregate(selected),
        "selected_ids_sha256": hashlib.sha256(selected_identifiers.encode("utf-8")).hexdigest(),
    }

    paired_selected: defaultdict[tuple[str, str], set[str]] = defaultdict(set)
    for record in selected:
        if (
            record.counterfactual_expected_change is True
            and record.counterfactual_pair_id
            and record.counterfactual_question_key
        ):
            paired_selected[
                (record.counterfactual_pair_id, record.counterfactual_question_key)
            ].add(record.scene_id)
    summary["expected_change_units_selected"] = len(paired_selected)
    summary["expected_change_units_complete"] = sum(
        len(scene_ids) == 2 for scene_ids in paired_selected.values()
    )
    summary["expected_change_units_incomplete"] = sum(
        len(scene_ids) != 2 for scene_ids in paired_selected.values()
    )
    return summary


def validate_output_namespace(value: str | None) -> str | None:
    """Validate the single path component used to isolate training artifacts."""

    if value is None:
        return None
    namespace = value.strip()
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", namespace):
        raise ValueError(
            "output_namespace must be 1-64 lowercase letters, digits, underscores, or hyphens"
        )
    return namespace


def training_artifact_paths(
    config: dict,
    output_namespace: str | None,
) -> tuple[Path, Path, Path]:
    """Resolve checkpoint, metrics, and loss-figure paths for one training run."""

    namespace = validate_output_namespace(output_namespace)
    reports_root = Path(str(config["paths"]["reports_root"]))
    if not reports_root.is_absolute():
        reports_root = PROJECT_ROOT / reports_root
    checkpoint_root = artifact_root(config, "checkpoints")
    if namespace is not None:
        checkpoint_root = checkpoint_root / namespace
    metrics_name = "training.json" if namespace is None else f"training_{namespace}.json"
    figure_name = "training_loss.png" if namespace is None else f"training_loss_{namespace}.png"
    return (
        checkpoint_root,
        reports_root / "metrics" / metrics_name,
        reports_root / "figures" / figure_name,
    )


def _persisted_split_scene_ids(qa_root: Path) -> dict[str, list[str]] | None:
    """Return the persisted split manifest, or ``None`` for legacy datasets."""

    manifest_path = qa_root / "splits.json"
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise TypeError("QA split manifest must be a JSON object")
    raw_splits = manifest.get("splits")
    if not isinstance(raw_splits, dict):
        raise TypeError("QA split manifest must contain a splits mapping")
    result = {
        name: sorted(str(scene_id) for scene_id in raw_splits.get(name, []))
        for name in ("train", "validation", "test")
    }
    all_scenes = [scene_id for values in result.values() for scene_id in values]
    if len(all_scenes) != len(set(all_scenes)):
        raise ValueError("QA split manifest contains scene leakage")
    return result


def validate_qa_split_membership(
    qa_root: Path,
    split_name: str,
    records: Sequence[QARecord],
) -> None:
    """Reject records whose scene is outside their persisted scene-level split."""

    if split_name not in {"train", "validation", "test"}:
        raise ValueError(f"Unsupported QA split name: {split_name!r}")
    splits = _persisted_split_scene_ids(qa_root)
    if splits is None:
        return
    allowed = set(splits[split_name])
    unexpected = sorted({record.scene_id for record in records} - allowed)
    if unexpected:
        raise ValueError(
            f"{split_name}.jsonl contains scenes outside splits.json {split_name} set: "
            f"{unexpected}"
        )


def load_qa_split_dataset(qa_root: Path, split_name: str) -> SceneQADataset:
    """Load one QA JSONL and validate all records before selection or filtering."""

    dataset = SceneQADataset(qa_root / f"{split_name}.jsonl")
    validate_qa_split_membership(qa_root, split_name, dataset.records)
    return dataset


def split_scene_ids(qa_root: Path, train_records: Sequence[QARecord]) -> dict[str, list[str]]:
    """Load the persisted scene-level split manifest for checkpoint provenance."""

    persisted = _persisted_split_scene_ids(qa_root)
    if persisted is not None:
        return persisted
    return {
        "train": sorted({record.scene_id for record in train_records}),
        "validation": [],
        "test": [],
    }


def anti_collapse_settings(config: dict) -> dict[str, float | int]:
    """Resolve and validate optional training-only regularizers.

    Defaults are deliberately supplied in code instead of ``default.yaml`` so
    existing zero-regularization checkpoints keep their original config hash.
    """

    training = config["training"]
    settings: dict[str, float | int] = {
        "latent_diversity_weight": float(training.get("latent_diversity_weight", 0.0)),
        "latent_diversity_cosine_margin": float(
            training.get("latent_diversity_cosine_margin", 0.2)
        ),
        "latent_diversity_max_latents": int(training.get("latent_diversity_max_latents", 128)),
        "paired_scene_separation_weight": float(
            training.get("paired_scene_separation_weight", 0.0)
        ),
        "paired_scene_cosine_distance_margin": float(
            training.get("paired_scene_cosine_distance_margin", 0.05)
        ),
    }
    if settings["latent_diversity_weight"] < 0:
        raise ValueError("latent_diversity_weight cannot be negative")
    if not -1.0 <= settings["latent_diversity_cosine_margin"] < 1.0:
        raise ValueError("latent_diversity_cosine_margin must be in [-1, 1)")
    if settings["latent_diversity_max_latents"] < 2:
        raise ValueError("latent_diversity_max_latents must be at least 2")
    if settings["paired_scene_separation_weight"] < 0:
        raise ValueError("paired_scene_separation_weight cannot be negative")
    if not 0.0 <= settings["paired_scene_cosine_distance_margin"] <= 2.0:
        raise ValueError("paired_scene_cosine_distance_margin must be in [0, 2]")
    return settings


def spatial_answer_contrastive_settings(config: dict) -> dict[str, float]:
    """Resolve the opt-in training-only target-localized answer objective.

    These defaults live in code so existing configurations and checkpoints
    remain byte-for-byte compatible.  Oracle target coordinates are consumed
    only by the supervised training process; chat and evaluation runtimes do
    not import or call this objective.
    """

    training = config["training"]
    settings = {
        "weight": float(training.get("spatial_answer_contrastive_weight", 0.0)),
        "margin": float(training.get("spatial_answer_contrastive_margin", 0.2)),
    }
    if settings["weight"] < 0:
        raise ValueError("spatial_answer_contrastive_weight cannot be negative")
    if not 0.0 <= settings["margin"] <= 2.0:
        raise ValueError("spatial_answer_contrastive_margin must be in [0, 2]")
    return settings


def spatial_answer_resume_contract_mismatch(
    checkpoint_metadata: dict,
    runtime_settings: dict[str, float],
) -> dict[str, object] | None:
    """Return a resume-contract mismatch while accepting legacy disabled runs."""

    saved = checkpoint_metadata.get("spatial_answer_contrastive")
    if saved is None and runtime_settings["weight"] == 0:
        return None
    if saved != runtime_settings:
        return {"checkpoint": saved, "runtime": runtime_settings}
    return None


def spatial_relation_contrastive_settings(config: dict) -> dict[str, float]:
    """Resolve the opt-in ordered target/reference relation objective.

    Oracle target and reference coordinates are available only to supervised
    training.  The resulting scene tokens remain global, question-independent,
    and metadata-free at inference.
    """

    training = config["training"]
    settings = {
        "weight": float(training.get("spatial_relation_contrastive_weight", 0.0)),
        "margin": float(training.get("spatial_relation_contrastive_margin", 0.1)),
        "temperature": float(training.get("spatial_relation_contrastive_temperature", 0.2)),
    }
    if settings["weight"] < 0:
        raise ValueError("spatial_relation_contrastive_weight cannot be negative")
    if not 0.0 <= settings["margin"] <= 1.0:
        raise ValueError("spatial_relation_contrastive_margin must be in [0, 1]")
    if settings["temperature"] <= 0:
        raise ValueError("spatial_relation_contrastive_temperature must be positive")
    return settings


def spatial_relation_resume_contract_mismatch(
    checkpoint_metadata: dict,
    runtime_settings: dict[str, float],
) -> dict[str, object] | None:
    """Protect exact resumes while accepting legacy disabled checkpoints."""

    saved = checkpoint_metadata.get("spatial_relation_contrastive")
    if saved is None and runtime_settings["weight"] == 0:
        return None
    if saved != runtime_settings:
        return {"checkpoint": saved, "runtime": runtime_settings}
    return None


def spatial_relation_warmup_settings(config: dict) -> dict[str, int | float]:
    """Resolve the scene-only ordered-relation warmup."""

    training = config["training"]
    settings: dict[str, int | float] = {
        "steps": int(training.get("spatial_relation_warmup_steps", 0)),
        "learning_rate": float(training.get("spatial_relation_warmup_learning_rate", 0.001)),
        "margin_target": float(training.get("spatial_relation_warmup_margin_target", 0.1)),
        "temperature": float(training.get("spatial_relation_warmup_temperature", 0.2)),
        "gradient_clip_norm": float(
            training.get("spatial_relation_warmup_gradient_clip_norm", 1.0)
        ),
    }
    if settings["steps"] < 0:
        raise ValueError("spatial_relation_warmup_steps cannot be negative")
    if settings["learning_rate"] <= 0:
        raise ValueError("spatial_relation_warmup_learning_rate must be positive")
    if not 0.0 <= settings["margin_target"] <= 1.0:
        raise ValueError("spatial_relation_warmup_margin_target must be in [0, 1]")
    if settings["temperature"] <= 0:
        raise ValueError("spatial_relation_warmup_temperature must be positive")
    if settings["gradient_clip_norm"] <= 0:
        raise ValueError("spatial_relation_warmup_gradient_clip_norm must be positive")
    return settings


def spatial_answer_warmup_settings(config: dict) -> dict[str, int | float]:
    """Resolve the opt-in scene-only spatial-answer warmup.

    The warmup is deliberately absent from the base YAML configuration.  A
    zero-step default therefore preserves both the behavior and config hashes
    of all earlier experiments.  Its optimizer is constructed only inside
    :func:`run_spatial_answer_warmup`, separately from the main adapter
    optimizer.
    """

    training = config["training"]
    settings: dict[str, int | float] = {
        "steps": int(training.get("spatial_answer_warmup_steps", 0)),
        "learning_rate": float(training.get("spatial_answer_warmup_learning_rate", 0.001)),
        "margin_target": float(training.get("spatial_answer_warmup_margin_target", 0.10)),
        "gradient_clip_norm": float(training.get("spatial_answer_warmup_gradient_clip_norm", 1.0)),
    }
    if settings["steps"] < 0:
        raise ValueError("spatial_answer_warmup_steps cannot be negative")
    if settings["learning_rate"] <= 0:
        raise ValueError("spatial_answer_warmup_learning_rate must be positive")
    if not 0.0 <= settings["margin_target"] <= 2.0:
        raise ValueError("spatial_answer_warmup_margin_target must be in [0, 2]")
    if settings["gradient_clip_norm"] <= 0:
        raise ValueError("spatial_answer_warmup_gradient_clip_norm must be positive")
    return settings


def spatial_answer_warmup_resume_contract_mismatch(
    checkpoint_metadata: dict,
    runtime_settings: dict[str, int | float],
    runtime_target_audit: dict[str, object],
) -> dict[str, object] | None:
    """Enforce exact warmup settings and target selection on resume.

    Legacy checkpoints remain resumable when warmup is disabled.  An enabled
    warmup cannot be silently rerun or skipped: its checkpoint must contain
    matching settings, the same deduplicated target fingerprint, and completed
    metrics from the original pre-LM stage.
    """

    saved_settings = checkpoint_metadata.get("spatial_answer_warmup")
    saved_audit = checkpoint_metadata.get("spatial_answer_warmup_target_audit")
    saved_metrics = checkpoint_metadata.get("spatial_answer_warmup_metrics")
    if saved_settings is None and int(runtime_settings["steps"]) == 0:
        return None
    mismatch: dict[str, object] = {}
    if saved_settings != runtime_settings:
        mismatch["settings"] = {
            "checkpoint": saved_settings,
            "runtime": runtime_settings,
        }
    if int(runtime_settings["steps"]) > 0:
        if saved_audit != runtime_target_audit:
            mismatch["target_audit"] = {
                "checkpoint": saved_audit,
                "runtime": runtime_target_audit,
            }
        if not isinstance(saved_metrics, dict) or not bool(saved_metrics.get("completed")):
            mismatch["metrics"] = {
                "checkpoint": saved_metrics,
                "runtime": "completed warmup metrics required",
            }
    return mismatch or None


def spatial_relation_warmup_resume_contract_mismatch(
    checkpoint_metadata: dict,
    runtime_settings: dict[str, int | float],
    runtime_target_audit: dict[str, object],
) -> dict[str, object] | None:
    """Enforce exact ordered-relation warmup settings and supervision."""

    saved_settings = checkpoint_metadata.get("spatial_relation_warmup")
    saved_audit = checkpoint_metadata.get("spatial_relation_warmup_target_audit")
    saved_metrics = checkpoint_metadata.get("spatial_relation_warmup_metrics")
    if saved_settings is None and int(runtime_settings["steps"]) == 0:
        return None
    mismatch: dict[str, object] = {}
    if saved_settings != runtime_settings:
        mismatch["settings"] = {
            "checkpoint": saved_settings,
            "runtime": runtime_settings,
        }
    if int(runtime_settings["steps"]) > 0:
        if saved_audit != runtime_target_audit:
            mismatch["target_audit"] = {
                "checkpoint": saved_audit,
                "runtime": runtime_target_audit,
            }
        if not isinstance(saved_metrics, dict) or not bool(saved_metrics.get("completed")):
            mismatch["metrics"] = {
                "checkpoint": saved_metrics,
                "runtime": "completed warmup metrics required",
            }
    return mismatch or None


def gradient_accumulation_resume_contract_mismatch(
    checkpoint_metadata: dict,
    runtime_accumulation: int,
) -> dict[str, object] | None:
    """Protect the optimizer-update schedule across checkpoint resumes.

    Checkpoints predating this explicit field used the historical default of
    one microbatch per update.  They remain resumable only with that default;
    a balanced multi-microbatch run cannot silently inherit their optimizer
    state under a different update schedule.
    """

    if runtime_accumulation < 1:
        raise ValueError("gradient_accumulation must be positive")
    saved = checkpoint_metadata.get("gradient_accumulation")
    if saved is None:
        if runtime_accumulation == 1:
            return None
        return {"checkpoint": "<legacy-default:1>", "runtime": runtime_accumulation}
    if isinstance(saved, bool) or not isinstance(saved, int) or saved != runtime_accumulation:
        return {"checkpoint": saved, "runtime": runtime_accumulation}
    return None


def spatial_answer_target_audit(
    units: Sequence[CounterfactualPairUnit],
) -> dict[str, int | float]:
    """Count grounded pair units and distinct physical target coordinates."""

    eligible_units = 0
    side_count = 0
    keys: set[tuple[str, float, float, float]] = set()
    for unit in units:
        if unit.reference.target_xyz is None or unit.counterfactual.target_xyz is None:
            continue
        eligible_units += 1
        for record in unit.records:
            side_count += 1
            keys.add(
                (
                    unit.pair_id,
                    *(round(float(value), 6) for value in record.target_xyz),
                )
            )
    return {
        "eligible_unit_count": eligible_units,
        "eligible_side_count": side_count,
        "unique_target_count": len(keys),
        "unit_to_unique_target_ratio": (float(eligible_units / len(keys)) if keys else 0.0),
    }


def spatial_relation_target_audit(
    units: Sequence[CounterfactualPairUnit],
) -> dict[str, int | float]:
    """Audit complete ordered target/reference supervision in pair units."""

    eligible_units = 0
    side_count = 0
    ordered_regions: set[
        tuple[str, str, tuple[float, float, float], tuple[float, float, float]]
    ] = set()
    for unit in units:
        if any(
            record.target_xyz is None or record.reference_xyz is None for record in unit.records
        ):
            continue
        eligible_units += 1
        for record in unit.records:
            side_count += 1
            ordered_regions.add(
                (
                    unit.pair_id,
                    record.scene_id,
                    tuple(round(float(value), 6) for value in record.target_xyz),
                    tuple(round(float(value), 6) for value in record.reference_xyz),
                )
            )
    return {
        "eligible_unit_count": eligible_units,
        "eligible_side_count": side_count,
        "unique_ordered_region_count": len(ordered_regions),
        "side_to_unique_ordered_region_ratio": (
            float(side_count / len(ordered_regions)) if ordered_regions else 0.0
        ),
    }


def deduplicate_spatial_relation_warmup_units(
    units: Sequence[CounterfactualPairUnit],
) -> list[CounterfactualPairUnit]:
    """Keep one deterministic unit per paired ordered physical relation."""

    selected: dict[tuple[object, ...], CounterfactualPairUnit] = {}
    for unit in units:
        if any(
            record.target_xyz is None or record.reference_xyz is None for record in unit.records
        ):
            continue
        key: tuple[object, ...] = (
            unit.pair_id,
            *(
                (
                    record.scene_id,
                    tuple(round(float(value), 6) for value in record.target_xyz),
                    tuple(round(float(value), 6) for value in record.reference_xyz),
                )
                for record in unit.records
            ),
        )
        previous = selected.get(key)
        answers = tuple(record.answer.strip() for record in unit.records)
        if previous is not None:
            if answers != tuple(record.answer.strip() for record in previous.records):
                raise ValueError(
                    "Relation warmup paraphrases for one ordered region disagree on answers"
                )
            continue
        selected[key] = unit
    return [selected[key] for key in sorted(selected)]


def spatial_relation_warmup_target_audit(
    units: Sequence[CounterfactualPairUnit],
) -> dict[str, object]:
    """Describe and fingerprint ordered relation warmup supervision."""

    deduplicated = deduplicate_spatial_relation_warmup_units(units)
    serialized = [
        {
            "pair_id": unit.pair_id,
            "sides": [
                {
                    "scene_id": record.scene_id,
                    "target_xyz": [round(float(value), 6) for value in record.target_xyz],
                    "reference_xyz": [round(float(value), 6) for value in record.reference_xyz],
                    "answer": record.answer.strip(),
                }
                for record in unit.records
            ],
        }
        for unit in deduplicated
    ]
    payload = json.dumps(serialized, sort_keys=True, separators=(",", ":"))
    return {
        **spatial_relation_target_audit(deduplicated),
        "deduplicated_unit_count": len(deduplicated),
        "target_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
    }


def deduplicate_spatial_answer_warmup_units(
    units: Sequence[CounterfactualPairUnit],
) -> list[CounterfactualPairUnit]:
    """Keep one deterministic supervision unit per physical paired target.

    Counterfactual QA generation intentionally emits multiple human-written
    paraphrases for one object.  Repeating them in this scene-only warmup would
    overweight that object even though no decoder or question representation is
    used.  The stable key therefore consists only of the opaque pair ID and the
    two metric target locations.  Duplicate targets must agree on their answer
    contrast; a disagreement is rejected instead of depending on input order.
    """

    selected: dict[
        tuple[str, tuple[float, float, float], tuple[float, float, float]],
        CounterfactualPairUnit,
    ] = {}
    for unit in units:
        if unit.reference.target_xyz is None or unit.counterfactual.target_xyz is None:
            continue
        key = (
            unit.pair_id,
            tuple(round(float(value), 6) for value in unit.reference.target_xyz),
            tuple(round(float(value), 6) for value in unit.counterfactual.target_xyz),
        )
        previous = selected.get(key)
        if previous is not None:
            previous_answers = (
                previous.reference.answer.strip(),
                previous.counterfactual.answer.strip(),
            )
            answers = (
                unit.reference.answer.strip(),
                unit.counterfactual.answer.strip(),
            )
            if previous_answers != answers:
                raise ValueError(
                    "Warmup paraphrases for one physical target disagree on paired answers"
                )
            continue
        selected[key] = unit
    return [selected[key] for key in sorted(selected)]


def spatial_answer_warmup_target_audit(
    units: Sequence[CounterfactualPairUnit],
) -> dict[str, object]:
    """Describe and fingerprint the exact deduplicated warmup supervision."""

    deduplicated = deduplicate_spatial_answer_warmup_units(units)
    serialized = [
        {
            "pair_id": unit.pair_id,
            "reference_scene_id": unit.reference.scene_id,
            "counterfactual_scene_id": unit.counterfactual.scene_id,
            "reference_target_xyz": [round(float(value), 6) for value in unit.reference.target_xyz],
            "counterfactual_target_xyz": [
                round(float(value), 6) for value in unit.counterfactual.target_xyz
            ],
            "reference_answer": unit.reference.answer.strip(),
            "counterfactual_answer": unit.counterfactual.answer.strip(),
        }
        for unit in deduplicated
    ]
    payload = json.dumps(serialized, sort_keys=True, separators=(",", ":"))
    audit = spatial_answer_target_audit(deduplicated)
    return {
        **audit,
        "deduplicated_unit_count": len(deduplicated),
        "target_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
    }


def frozen_answer_embedding(
    tokenizer,
    input_embeddings: torch.nn.Module,
    answer: str,
    device: torch.device,
) -> torch.Tensor:
    """Mean-pool an answer's frozen LM token embeddings without EOS tokens."""

    answer_ids = tokenizer(
        answer.strip(), add_special_tokens=False, return_tensors="pt"
    ).input_ids.to(device)
    if answer_ids.ndim != 2 or answer_ids.shape[0] != 1 or answer_ids.shape[1] < 1:
        raise ValueError("Answer must produce at least one non-special token")
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is not None and torch.any(answer_ids == int(eos_token_id)):
        # ``add_special_tokens=False`` should already guarantee this for the
        # supported local models. Reject a tokenizer that violates the contract
        # rather than silently allowing the shared EOS target to dilute pairs.
        raise ValueError("Answer tokenization unexpectedly included EOS")
    return input_embeddings(answer_ids).detach().float().mean(dim=1)


def training_counterfactual_scene_pairs(
    records: Sequence[QARecord],
) -> list[tuple[str, str, str]]:
    """Return complete counterfactual pairs found strictly in training records.

    No question, answer, target, change type, validation record, or test record
    participates. Incomplete pairs are skipped instead of reaching across a
    dataset split to find their missing scene.
    """

    members: defaultdict[str, set[str]] = defaultdict(set)
    for record in records:
        if record.counterfactual_pair_id:
            members[record.counterfactual_pair_id].add(record.scene_id)
    pairs: list[tuple[str, str, str]] = []
    for pair_id, scene_ids in sorted(members.items()):
        if len(scene_ids) > 2:
            raise ValueError(
                f"Counterfactual pair {pair_id} contains more than two training scenes"
            )
        if len(scene_ids) == 2:
            first, second = sorted(scene_ids)
            pairs.append((pair_id, first, second))
    return pairs


def batch_objective(
    output,
    batch_records: Sequence[QARecord],
    data: MapTensorData,
    language,
    composer: ContinuousPrefixComposer,
    grounding: QuestionGroundingHead,
    config: dict,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute teacher-forced language and auxiliary grounding loss."""

    scene_tokens = output.scene_tokens.expand(len(batch_records), -1, -1)
    model_dtype = next(language.model.parameters()).dtype
    scene_tokens = scene_tokens.to(model_dtype)
    prefix_batches = []
    question_embeddings = []
    for index, record in enumerate(batch_records):
        prompt_ids = prompt_token_ids(
            language.tokenizer,
            config["language"]["system_prompt"],
            record.question,
            language.device,
        )
        answer_ids = tokenize_answer(language.tokenizer, record.answer, language.device)
        prefix_batches.append(
            composer.compose(
                scene_tokens[index : index + 1],
                prompt_ids,
                language.model.get_input_embeddings(),
                answer_ids,
                prefix_backend=getattr(language, "prefix_backend", None),
            )
        )
        grounding_question_ids = question_token_ids(
            language.tokenizer, record.question, language.device
        )
        question_embeddings.append(
            language.model.get_input_embeddings()(grounding_question_ids).float().mean(dim=1)
        )
    prefix_batch = stack_prefix_batches(
        prefix_batches,
        language.device,
        prefix_backend=getattr(language, "prefix_backend", None),
    )
    lm_output = forward_prefix_batch(language, prefix_batch)
    language_loss = lm_output.loss.float()
    grounding_loss = torch.zeros((), device=language.device)
    grounded_indices = [
        index for index, record in enumerate(batch_records) if record.target_xyz is not None
    ]
    if grounded_indices:
        question_tensor = torch.cat(question_embeddings, dim=0)
        latent_batch = output.native_latents.expand(len(batch_records), -1, -1)
        predicted, anchor_logits, _ = grounding.forward_with_attention(
            latent_batch.float(), question_tensor.float()
        )
        targets = torch.tensor(
            [batch_records[index].target_xyz for index in grounded_indices],
            dtype=torch.float32,
            device=language.device,
        )
        normalized = normalize_xyz(targets, data.room_min, data.room_max)
        coordinate_loss = torch.nn.functional.smooth_l1_loss(
            predicted[grounded_indices], normalized
        )
        anchor_targets = grounding.nearest_anchor_targets(normalized)
        anchor_loss = torch.nn.functional.cross_entropy(
            anchor_logits[grounded_indices], anchor_targets
        )
        grounding_loss = (
            coordinate_loss
            + float(config["training"].get("grounding_anchor_weight", 1.0)) * anchor_loss
        )
    loss = language_loss + float(config["training"]["grounding_weight"]) * grounding_loss
    return loss, language_loss, grounding_loss


def pair_spatial_answer_contrastive_objective(
    outputs_by_scene: dict[str, object],
    units: Sequence[CounterfactualPairUnit],
    maps: dict[str, MapTensorData],
    language,
    *,
    latent_count: int,
    margin: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | int]]:
    """Compute target-localized answer alignment for complete grounded pairs.

    Oracle coordinates and answer strings enter only this supervised objective.
    The selected vectors are final continuous scene tokens, and this function is
    absent from chat/runtime imports.  A unit is eligible only when both paired
    records have targets, keeping the mean symmetric across its two sides.
    """

    complete_units = [
        unit
        for unit in units
        if unit.reference.target_xyz is not None and unit.counterfactual.target_xyz is not None
    ]
    if not complete_units:
        first_output = outputs_by_scene[next(iter(outputs_by_scene))]
        zero = first_output.scene_tokens.sum() * 0.0
        return zero, {
            "eligible_unit_count": 0,
            "eligible_side_count": 0,
            "unique_target_count": 0,
            "own_similarity": zero.detach().reshape(1)[:0],
            "alternate_similarity": zero.detach().reshape(1)[:0],
            "achieved_margin": zero.detach().reshape(1)[:0],
            "configured_margin": torch.tensor(
                float(margin), device=zero.device, dtype=torch.float32
            ),
            "anchor_indices": torch.empty(0, device=zero.device, dtype=torch.long),
        }

    scene_tokens: list[torch.Tensor] = []
    target_xyz: list[torch.Tensor] = []
    room_min: list[torch.Tensor] = []
    room_max: list[torch.Tensor] = []
    own_embeddings: list[torch.Tensor] = []
    alternate_embeddings: list[torch.Tensor] = []
    input_embeddings = language.model.get_input_embeddings()
    for unit in complete_units:
        reference, counterfactual = unit.records
        for record, alternate_record in (
            (reference, counterfactual),
            (counterfactual, reference),
        ):
            output = outputs_by_scene[record.scene_id]
            if output.scene_tokens.shape[0] != 1:
                raise ValueError("Each scene tokenizer output must have batch size one")
            scene_tokens.append(output.scene_tokens[0])
            target_xyz.append(
                torch.tensor(record.target_xyz, dtype=torch.float32, device=language.device)
            )
            data = maps[record.scene_id]
            room_min.append(data.room_min.float())
            room_max.append(data.room_max.float())
            own_embeddings.append(
                frozen_answer_embedding(
                    language.tokenizer,
                    input_embeddings,
                    record.answer,
                    language.device,
                )[0]
            )
            alternate_embeddings.append(
                frozen_answer_embedding(
                    language.tokenizer,
                    input_embeddings,
                    alternate_record.answer,
                    language.device,
                )[0]
            )

    scene_tensor = torch.stack(scene_tokens, dim=0)
    targets = torch.stack(target_xyz, dim=0)
    minimums = torch.stack(room_min, dim=0).to(device=language.device)
    maximums = torch.stack(room_max, dim=0).to(device=language.device)
    anchor_indices = nearest_spatial_anchor_indices(targets, minimums, maximums, latent_count)
    loss, diagnostics = spatial_scene_answer_contrastive_loss(
        scene_tensor,
        anchor_indices,
        torch.stack(own_embeddings, dim=0),
        torch.stack(alternate_embeddings, dim=0),
        margin=margin,
    )
    target_audit = spatial_answer_target_audit(complete_units)
    diagnostics["eligible_unit_count"] = len(complete_units)
    diagnostics["unique_target_count"] = int(target_audit["unique_target_count"])
    return loss, diagnostics


def _spatial_answer_warmup_step_metrics(
    step: int,
    loss: torch.Tensor,
    diagnostics: dict[str, torch.Tensor | int],
    margin_target: float,
) -> dict[str, int | float | bool]:
    margins = diagnostics["achieved_margin"]
    if not isinstance(margins, torch.Tensor) or margins.numel() == 0:
        raise ValueError("Spatial-answer warmup requires at least one eligible side")
    margins = margins.detach().float().cpu()
    successes = margins >= float(margin_target)
    return {
        "step": int(step),
        "loss": float(loss.detach().float().cpu()),
        "mean_achieved_margin": float(margins.mean()),
        "minimum_achieved_margin": float(margins.min()),
        "side_success_rate": float(successes.float().mean()),
        "successful_side_count": int(successes.sum()),
        "eligible_side_count": int(margins.numel()),
        "all_sides_passed": bool(successes.all()),
    }


def run_spatial_answer_warmup(
    scene_model: SceneTokenizer,
    maps: dict[str, MapTensorData],
    units: Sequence[CounterfactualPairUnit],
    language,
    *,
    latent_count: int,
    settings: dict[str, int | float],
) -> dict[str, object]:
    """Warm only the scene encoder against frozen answer embeddings.

    Each evaluation/update step forwards every participating scene exactly
    once, then applies the target-localized contrastive hinge over a
    deduplicated physical-target set.  The frozen decoder is never called, and
    neither the prefix composer nor grounding head is accepted by this API.
    A private AdamW instance prevents warmup moments from contaminating the
    subsequently constructed main optimization trajectory.
    """

    requested_steps = int(settings["steps"])
    target_audit = spatial_answer_warmup_target_audit(units)
    base_metrics: dict[str, object] = {
        "enabled": requested_steps > 0,
        "completed": True,
        "requested_steps": requested_steps,
        "forward_steps": 0,
        "optimizer_steps": 0,
        "stopped_early": False,
        "margin_target": float(settings["margin_target"]),
        "target_audit": target_audit,
        "history": [],
        "final": None,
    }
    if requested_steps == 0:
        return base_metrics

    deduplicated = deduplicate_spatial_answer_warmup_units(units)
    if not deduplicated:
        raise ValueError("Enabled spatial-answer warmup has no grounded pair targets")
    participating_scene_ids = sorted(
        {scene_id for unit in deduplicated for scene_id in unit.scene_ids}
    )
    missing = [scene_id for scene_id in participating_scene_ids if scene_id not in maps]
    if missing:
        raise ValueError(f"Spatial-answer warmup maps are missing scenes: {missing}")

    trainable_parameters = [
        parameter for parameter in scene_model.parameters() if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise ValueError("Spatial-answer warmup scene model has no trainable parameters")
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=float(settings["learning_rate"]),
        weight_decay=0.0,
    )
    was_training = scene_model.training
    history: list[dict[str, int | float | bool]] = []
    optimizer_steps = 0
    try:
        scene_model.train(True)
        for step in range(1, requested_steps + 1):
            optimizer.zero_grad(set_to_none=True)
            outputs_by_scene = {
                scene_id: map_forward(scene_model, maps[scene_id])
                for scene_id in participating_scene_ids
            }
            loss, diagnostics = pair_spatial_answer_contrastive_objective(
                outputs_by_scene,
                deduplicated,
                maps,
                language,
                latent_count=latent_count,
                margin=float(settings["margin_target"]),
            )
            step_metrics = _spatial_answer_warmup_step_metrics(
                step,
                loss,
                diagnostics,
                float(settings["margin_target"]),
            )
            history.append(step_metrics)
            print(json.dumps({"phase": "spatial_answer_warmup", **step_metrics}), flush=True)
            if bool(step_metrics["all_sides_passed"]):
                break
            if not loss.requires_grad:
                raise RuntimeError("Spatial-answer warmup loss is detached from the scene model")
            loss.backward()
            if not any(parameter.grad is not None for parameter in trainable_parameters):
                raise RuntimeError("Spatial-answer warmup produced no scene-model gradients")
            torch.nn.utils.clip_grad_norm_(
                trainable_parameters,
                float(settings["gradient_clip_norm"]),
            )
            optimizer.step()
            optimizer_steps += 1
    finally:
        optimizer.zero_grad(set_to_none=True)
        scene_model.train(was_training)

    final = history[-1]
    base_metrics.update(
        {
            "forward_steps": len(history),
            "optimizer_steps": optimizer_steps,
            "stopped_early": bool(final["all_sides_passed"]) and len(history) < requested_steps,
            "history": history,
            "final": final,
        }
    )
    return base_metrics


def pair_spatial_relation_contrastive_objective(
    outputs_by_scene: dict[str, object],
    units: Sequence[CounterfactualPairUnit],
    maps: dict[str, MapTensorData],
    language,
    *,
    temperature: float,
    margin: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | int]]:
    """Align ordered target/reference regions with paired answer directions.

    Coordinates and answers are training-only supervision.  Dense soft pooling
    touches all global scene latents and introduces no inference-time retrieval.
    """

    complete_units = [
        unit
        for unit in units
        if all(
            record.target_xyz is not None and record.reference_xyz is not None
            for record in unit.records
        )
    ]
    if not complete_units:
        first_output = outputs_by_scene[next(iter(outputs_by_scene))]
        zero = first_output.scene_tokens.sum() * 0.0
        empty = zero.detach().reshape(1)[:0]
        return zero, {
            "eligible_unit_count": 0,
            "eligible_side_count": 0,
            "unique_ordered_region_count": 0,
            "relation_answer_cosine": empty,
            "achieved_margin": empty,
            "relation_norm": empty,
            "configured_margin": torch.tensor(
                float(margin), device=zero.device, dtype=torch.float32
            ),
            "configured_temperature": torch.tensor(
                float(temperature), device=zero.device, dtype=torch.float32
            ),
        }

    scene_tokens: list[torch.Tensor] = []
    target_xyz: list[torch.Tensor] = []
    reference_xyz: list[torch.Tensor] = []
    room_min: list[torch.Tensor] = []
    room_max: list[torch.Tensor] = []
    own_embeddings: list[torch.Tensor] = []
    alternate_embeddings: list[torch.Tensor] = []
    input_embeddings = language.model.get_input_embeddings()
    for unit in complete_units:
        reference, counterfactual = unit.records
        for record, alternate_record in (
            (reference, counterfactual),
            (counterfactual, reference),
        ):
            output = outputs_by_scene[record.scene_id]
            if output.scene_tokens.shape[0] != 1:
                raise ValueError("Each scene tokenizer output must have batch size one")
            scene_tokens.append(output.scene_tokens[0])
            target_xyz.append(
                torch.tensor(record.target_xyz, dtype=torch.float32, device=language.device)
            )
            reference_xyz.append(
                torch.tensor(record.reference_xyz, dtype=torch.float32, device=language.device)
            )
            data = maps[record.scene_id]
            room_min.append(data.room_min.float())
            room_max.append(data.room_max.float())
            own_embeddings.append(
                frozen_answer_embedding(
                    language.tokenizer,
                    input_embeddings,
                    record.answer,
                    language.device,
                )[0]
            )
            alternate_embeddings.append(
                frozen_answer_embedding(
                    language.tokenizer,
                    input_embeddings,
                    alternate_record.answer,
                    language.device,
                )[0]
            )

    minimums = torch.stack(room_min, dim=0).to(device=language.device)
    maximums = torch.stack(room_max, dim=0).to(device=language.device)
    normalized_targets = normalize_xyz(torch.stack(target_xyz, dim=0), minimums, maximums)
    normalized_references = normalize_xyz(torch.stack(reference_xyz, dim=0), minimums, maximums)
    loss, diagnostics = ordered_spatial_relation_contrastive_loss(
        torch.stack(scene_tokens, dim=0),
        normalized_targets,
        normalized_references,
        torch.stack(own_embeddings, dim=0),
        torch.stack(alternate_embeddings, dim=0),
        temperature=temperature,
        margin=margin,
    )
    audit = spatial_relation_target_audit(complete_units)
    diagnostics["eligible_unit_count"] = len(complete_units)
    diagnostics["unique_ordered_region_count"] = int(audit["unique_ordered_region_count"])
    diagnostics["achieved_margin"] = diagnostics["relation_answer_cosine"]
    return loss, diagnostics


def run_spatial_relation_warmup(
    scene_model: SceneTokenizer,
    maps: dict[str, MapTensorData],
    units: Sequence[CounterfactualPairUnit],
    language,
    *,
    settings: dict[str, int | float],
) -> dict[str, object]:
    """Warm only the scene path on ordered spatial relation supervision."""

    requested_steps = int(settings["steps"])
    target_audit = spatial_relation_warmup_target_audit(units)
    base_metrics: dict[str, object] = {
        "enabled": requested_steps > 0,
        "completed": True,
        "requested_steps": requested_steps,
        "forward_steps": 0,
        "optimizer_steps": 0,
        "stopped_early": False,
        "margin_target": float(settings["margin_target"]),
        "temperature": float(settings["temperature"]),
        "target_audit": target_audit,
        "history": [],
        "final": None,
    }
    if requested_steps == 0:
        return base_metrics

    deduplicated = deduplicate_spatial_relation_warmup_units(units)
    if not deduplicated:
        raise ValueError("Enabled spatial-relation warmup has no ordered relation targets")
    participating_scene_ids = sorted(
        {scene_id for unit in deduplicated for scene_id in unit.scene_ids}
    )
    missing = [scene_id for scene_id in participating_scene_ids if scene_id not in maps]
    if missing:
        raise ValueError(f"Spatial-relation warmup maps are missing scenes: {missing}")

    trainable_parameters = [
        parameter for parameter in scene_model.parameters() if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise ValueError("Spatial-relation warmup scene model has no trainable parameters")
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=float(settings["learning_rate"]),
        weight_decay=0.0,
    )
    was_training = scene_model.training
    history: list[dict[str, int | float | bool]] = []
    optimizer_steps = 0
    try:
        scene_model.train(True)
        for step in range(1, requested_steps + 1):
            optimizer.zero_grad(set_to_none=True)
            outputs_by_scene = {
                scene_id: map_forward(scene_model, maps[scene_id])
                for scene_id in participating_scene_ids
            }
            loss, diagnostics = pair_spatial_relation_contrastive_objective(
                outputs_by_scene,
                deduplicated,
                maps,
                language,
                temperature=float(settings["temperature"]),
                margin=float(settings["margin_target"]),
            )
            step_metrics = _spatial_answer_warmup_step_metrics(
                step,
                loss,
                diagnostics,
                float(settings["margin_target"]),
            )
            history.append(step_metrics)
            print(json.dumps({"phase": "spatial_relation_warmup", **step_metrics}), flush=True)
            if bool(step_metrics["all_sides_passed"]):
                break
            if not loss.requires_grad:
                raise RuntimeError("Spatial-relation warmup loss is detached")
            loss.backward()
            if not any(parameter.grad is not None for parameter in trainable_parameters):
                raise RuntimeError("Spatial-relation warmup produced no scene-model gradients")
            torch.nn.utils.clip_grad_norm_(
                trainable_parameters,
                float(settings["gradient_clip_norm"]),
            )
            optimizer.step()
            optimizer_steps += 1
    finally:
        optimizer.zero_grad(set_to_none=True)
        scene_model.train(was_training)

    final = history[-1]
    base_metrics.update(
        {
            "forward_steps": len(history),
            "optimizer_steps": optimizer_steps,
            "stopped_early": bool(final["all_sides_passed"]) and len(history) < requested_steps,
            "history": history,
            "final": final,
        }
    )
    return base_metrics


def pair_batch_objective(
    outputs_by_scene: dict[str, object],
    units: Sequence[CounterfactualPairUnit],
    maps: dict[str, MapTensorData],
    language,
    composer: ContinuousPrefixComposer,
    grounding: QuestionGroundingHead,
    config: dict,
    *,
    ranking_margin: float,
    ranking_mode: str = "nll",
    collect_full_vocab_first_answer_token: bool = False,
    full_vocab_ranking_margin: float = 0.0,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    dict[str, torch.Tensor | str | None],
]:
    """Score paired prefixes with NLL swapping or direct candidate logits.

    ``nll`` preserves the original correct-versus-swapped teacher-forcing
    objective. ``candidate_logit`` runs only each correct sequence and compares
    the exact one-token alternatives in that sequence's same next-token
    distribution.
    """

    if not units:
        raise ValueError("A pair batch cannot be empty")
    if ranking_mode not in {"nll", "candidate_logit"}:
        raise ValueError("ranking_mode must be 'nll' or 'candidate_logit'")
    model_dtype = next(language.model.parameters()).dtype
    correct_prefix_batches = []
    swapped_prefix_batches = []
    correct_records: list[QARecord] = []
    correct_native_latents: list[torch.Tensor] = []
    correct_question_embeddings: list[torch.Tensor] = []
    correct_ranking_masks: list[torch.Tensor] = []
    swapped_ranking_masks: list[torch.Tensor] = []
    candidate_specs: list[tuple[int, int, int]] = []
    # Both sides of each unit remain adjacent in the correct microbatch. NLL
    # mode adds a sequential swapped microbatch; candidate-logit mode derives
    # both scores from the correct microbatch and never constructs that batch.
    for unit in units:
        reference, counterfactual = unit.records
        reference_answer_ids = tokenize_answer(
            language.tokenizer, reference.answer, language.device
        )
        counterfactual_answer_ids = tokenize_answer(
            language.tokenizer, counterfactual.answer, language.device
        )
        reference_answer_mask, counterfactual_answer_mask = differing_answer_token_masks(
            reference_answer_ids, counterfactual_answer_ids
        )
        correct_ranking_masks.extend((reference_answer_mask, counterfactual_answer_mask))
        swapped_ranking_masks.extend((counterfactual_answer_mask, reference_answer_mask))
        if ranking_mode == "candidate_logit":
            answer_offset, reference_token_id, counterfactual_token_id = (
                single_differing_answer_token(reference_answer_ids, counterfactual_answer_ids)
            )
            candidate_specs.extend(
                (
                    (answer_offset, reference_token_id, counterfactual_token_id),
                    (answer_offset, counterfactual_token_id, reference_token_id),
                )
            )
        correct_specs = (
            (reference, reference.answer),
            (counterfactual, counterfactual.answer),
        )
        swapped_specs = (
            (reference, counterfactual.answer),
            (counterfactual, reference.answer),
        )
        for record, answer in correct_specs:
            output = outputs_by_scene[record.scene_id]
            scene_tokens = output.scene_tokens.to(model_dtype)
            prompt_ids = prompt_token_ids(
                language.tokenizer,
                config["language"]["system_prompt"],
                record.question,
                language.device,
            )
            answer_ids = tokenize_answer(language.tokenizer, answer, language.device)
            correct_prefix_batches.append(
                composer.compose(
                    scene_tokens,
                    prompt_ids,
                    language.model.get_input_embeddings(),
                    answer_ids,
                    prefix_backend=getattr(language, "prefix_backend", None),
                )
            )
            correct_records.append(record)
            correct_native_latents.append(output.native_latents)
            grounding_ids = question_token_ids(language.tokenizer, record.question, language.device)
            correct_question_embeddings.append(
                language.model.get_input_embeddings()(grounding_ids).float().mean(dim=1)
            )
        if ranking_mode == "nll":
            for record, answer in swapped_specs:
                output = outputs_by_scene[record.scene_id]
                scene_tokens = output.scene_tokens.to(model_dtype)
                prompt_ids = prompt_token_ids(
                    language.tokenizer,
                    config["language"]["system_prompt"],
                    record.question,
                    language.device,
                )
                answer_ids = tokenize_answer(language.tokenizer, answer, language.device)
                swapped_prefix_batches.append(
                    composer.compose(
                        scene_tokens,
                        prompt_ids,
                        language.model.get_input_embeddings(),
                        answer_ids,
                        prefix_backend=getattr(language, "prefix_backend", None),
                    )
                )

    correct_batch = stack_prefix_batches(
        correct_prefix_batches,
        language.device,
        prefix_backend=getattr(language, "prefix_backend", None),
    )
    correct_output = forward_prefix_batch(language, correct_batch)
    correct_labels = correct_batch.labels
    correct_ranking_labels = correct_labels.clone()
    for row, answer_mask in enumerate(correct_ranking_masks):
        restrict_labels_to_answer_mask(correct_ranking_labels, row, answer_mask)
    correct_answer_nll = token_normalized_nll(correct_output.logits, correct_labels).reshape(
        len(units), 2
    )
    full_vocab_first_token_margins = (
        first_answer_token_full_vocab_margins(correct_output.logits, correct_labels).reshape(
            len(units), 2
        )
        if collect_full_vocab_first_answer_token
        else None
    )
    full_vocab_ranking_loss = torch.zeros((), device=language.device)
    if full_vocab_first_token_margins is not None:
        full_vocab_ranking_loss, _ = ranking_margin_hinge(
            full_vocab_first_token_margins,
            margin=full_vocab_ranking_margin,
        )
    correct_nll: torch.Tensor | None
    swapped_nll: torch.Tensor | None
    own_candidate_logits: torch.Tensor | None = None
    alternate_candidate_logits: torch.Tensor | None = None
    if ranking_mode == "candidate_logit":
        flat_margins, flat_own, flat_alternate = candidate_logit_margins(
            correct_output.logits,
            correct_labels,
            candidate_specs,
        )
        margins = flat_margins.reshape(len(units), 2)
        own_candidate_logits = flat_own.reshape(len(units), 2)
        alternate_candidate_logits = flat_alternate.reshape(len(units), 2)
        ranking_loss, margins = ranking_margin_hinge(margins, margin=ranking_margin)
        correct_nll = None
        swapped_nll = None
    else:
        correct_nll = token_normalized_nll(correct_output.logits, correct_ranking_labels).reshape(
            len(units), 2
        )
        del correct_output, correct_batch, correct_labels
        swapped_batch = stack_prefix_batches(
            swapped_prefix_batches,
            language.device,
            prefix_backend=getattr(language, "prefix_backend", None),
        )
        swapped_output = forward_prefix_batch(language, swapped_batch)
        swapped_labels = swapped_batch.labels
        swapped_ranking_labels = swapped_labels.clone()
        for row, answer_mask in enumerate(swapped_ranking_masks):
            restrict_labels_to_answer_mask(swapped_ranking_labels, row, answer_mask)
        swapped_nll = token_normalized_nll(swapped_output.logits, swapped_ranking_labels).reshape(
            len(units), 2
        )
        ranking_loss, margins = pair_ranking_hinge(
            correct_nll,
            swapped_nll,
            margin=ranking_margin,
        )
    if ranking_mode == "candidate_logit":
        del correct_output, correct_batch, correct_labels
    language_loss = correct_answer_nll.mean()

    grounding_loss = torch.zeros((), device=language.device)
    grounded_indices = [
        index for index, record in enumerate(correct_records) if record.target_xyz is not None
    ]
    if grounded_indices:
        latent_batch = torch.cat(correct_native_latents, dim=0)
        question_batch = torch.cat(correct_question_embeddings, dim=0)
        predicted, anchor_logits, _ = grounding.forward_with_attention(
            latent_batch.float(), question_batch.float()
        )
        normalized_targets = []
        for index in grounded_indices:
            record = correct_records[index]
            target = torch.tensor([record.target_xyz], dtype=torch.float32, device=language.device)
            data = maps[record.scene_id]
            normalized_targets.append(normalize_xyz(target, data.room_min, data.room_max))
        normalized_target_tensor = torch.cat(normalized_targets, dim=0)
        coordinate_loss = torch.nn.functional.smooth_l1_loss(
            predicted[grounded_indices], normalized_target_tensor
        )
        anchor_targets = grounding.nearest_anchor_targets(normalized_target_tensor)
        anchor_loss = torch.nn.functional.cross_entropy(
            anchor_logits[grounded_indices], anchor_targets
        )
        grounding_loss = (
            coordinate_loss
            + float(config["training"].get("grounding_anchor_weight", 1.0)) * anchor_loss
        )
    base_loss = language_loss + float(config["training"]["grounding_weight"]) * grounding_loss
    diagnostics = {
        "ranking_mode": ranking_mode,
        "evaluation_type": (
            "teacher_forced_same_distribution_candidate_logit_ranking"
            if ranking_mode == "candidate_logit"
            else "teacher_forced_discriminative_answer_candidate_ranking"
        ),
        "margin_name": (
            "own_vs_alternate_candidate_logit_margin"
            if ranking_mode == "candidate_logit"
            else "correct_vs_swapped_nll_margin"
        ),
        "correct_nll": correct_nll,
        "correct_answer_nll": correct_answer_nll,
        "swapped_nll": swapped_nll,
        "own_candidate_logits": own_candidate_logits,
        "alternate_candidate_logits": alternate_candidate_logits,
        "first_answer_token_full_vocab_margins": full_vocab_first_token_margins,
        "first_answer_token_full_vocab_ranking_loss": full_vocab_ranking_loss,
        "margins": margins,
        "ranking_tokens_per_side": torch.tensor(
            (
                [1] * len(correct_ranking_masks)
                if ranking_mode == "candidate_logit"
                else [int(mask.sum()) for mask in correct_ranking_masks]
            ),
            dtype=torch.int64,
            device=language.device,
        ).reshape(len(units), 2),
        "side_accuracy": margins.gt(0).float().mean(),
        "unit_accuracy": margins.gt(0).all(dim=1).float().mean(),
    }
    spatial_settings = spatial_answer_contrastive_settings(config)
    spatial_answer_loss = torch.zeros((), device=language.device)
    spatial_answer_diagnostics: dict[str, torch.Tensor | int] | None = None
    if spatial_settings["weight"] > 0:
        spatial_answer_loss, spatial_answer_diagnostics = pair_spatial_answer_contrastive_objective(
            outputs_by_scene,
            units,
            maps,
            language,
            latent_count=int(config["scene_encoder"]["global_latents"]),
            margin=spatial_settings["margin"],
        )
        base_loss = base_loss + spatial_settings["weight"] * spatial_answer_loss
    diagnostics["spatial_answer_contrastive_loss"] = spatial_answer_loss
    diagnostics["spatial_answer_contrastive"] = spatial_answer_diagnostics
    relation_settings = spatial_relation_contrastive_settings(config)
    spatial_relation_loss = torch.zeros((), device=language.device)
    spatial_relation_diagnostics: dict[str, torch.Tensor | int] | None = None
    if relation_settings["weight"] > 0:
        spatial_relation_loss, spatial_relation_diagnostics = (
            pair_spatial_relation_contrastive_objective(
                outputs_by_scene,
                units,
                maps,
                language,
                temperature=relation_settings["temperature"],
                margin=relation_settings["margin"],
            )
        )
        base_loss = base_loss + relation_settings["weight"] * spatial_relation_loss
        if int(spatial_relation_diagnostics["eligible_side_count"]) == 0:
            # A color/support pair has no ordered reference coordinate.  Keep
            # its differentiable zero loss but do not reduce empty diagnostics
            # into NaN logging or epoch aggregates.
            spatial_relation_diagnostics = None
    diagnostics["spatial_relation_contrastive_loss"] = spatial_relation_loss
    diagnostics["spatial_relation_contrastive"] = spatial_relation_diagnostics
    return base_loss, language_loss, grounding_loss, ranking_loss, diagnostics


def evaluate_pair_candidate_gate(
    units: Sequence[CounterfactualPairUnit],
    *,
    maps: dict[str, MapTensorData],
    config: dict,
    language,
    scene_model: SceneTokenizer,
    global_scene_residual: GlobalSceneResidual | None = None,
    signed_x_scene_residual: SignedXSceneResidual | None = None,
    composer: ContinuousPrefixComposer,
    grounding: QuestionGroundingHead,
    units_per_batch: int,
    ranking_margin: float,
    ranking_mode: str,
    changed_unit_accuracy_threshold: float,
    prediction_flip_threshold: float,
    wrong_prefix_flip_threshold: float,
    first_answer_token_top1_accuracy_threshold: float | None = None,
    lora_installation: LoRABankCollection | LoRAInstallation | None = None,
    dense_aligner: torch.nn.Module | None = None,
    dense_sidecar_adapter: DenseSidecarAdapter | None = None,
) -> dict[str, object]:
    """Evaluate all training pair units with the configured deterministic ranking."""

    if not units:
        raise ValueError("Pair gate evaluation requires pair units")
    modules = tuple(
        module
        for module in (
            scene_model,
            global_scene_residual,
            signed_x_scene_residual,
            dense_aligner,
            dense_sidecar_adapter,
            composer,
            grounding,
        )
        if module is not None
    )
    previous_modes = [module.training for module in modules]
    previous_lora_mode = None if lora_installation is None else lora_installation.training
    for module in modules:
        module.eval()
    if lora_installation is not None:
        lora_installation.eval()
    by_pair: defaultdict[str, list[CounterfactualPairUnit]] = defaultdict(list)
    for unit in units:
        by_pair[unit.pair_id].append(unit)
    all_margins: list[torch.Tensor] = []
    all_full_vocab_first_token_margins: list[torch.Tensor] = []
    ordered_units: list[CounterfactualPairUnit] = []
    gate_metrics_by_pair: dict[str, dict[str, object]] = {}
    try:
        with torch.inference_mode():
            for pair_id in sorted(by_pair):
                pair_units = by_pair[pair_id]
                ordered_units.extend(pair_units)
                scene_ids = pair_units[0].scene_ids
                pair_margins: list[torch.Tensor] = []
                pair_full_vocab_margins: list[torch.Tensor] = []
                outputs = {
                    scene_id: map_forward(
                        scene_model,
                        maps[scene_id],
                        global_scene_residual,
                        signed_x_scene_residual,
                        dense_aligner,
                        dense_sidecar_adapter,
                    )
                    for scene_id in scene_ids
                }
                for offset in range(0, len(pair_units), units_per_batch):
                    _, _, _, _, diagnostics = pair_batch_objective(
                        outputs,
                        pair_units[offset : offset + units_per_batch],
                        maps,
                        language,
                        composer,
                        grounding,
                        config,
                        ranking_margin=ranking_margin,
                        ranking_mode=ranking_mode,
                        collect_full_vocab_first_answer_token=True,
                        full_vocab_ranking_margin=float(
                            config["training"].get("pair_full_vocab_ranking_margin", 0.0)
                        ),
                    )
                    detached_margins = diagnostics["margins"].detach().float().cpu()
                    all_margins.append(detached_margins)
                    pair_margins.append(detached_margins)
                    full_vocab_margins = diagnostics["first_answer_token_full_vocab_margins"]
                    assert isinstance(full_vocab_margins, torch.Tensor)
                    detached_full_vocab_margins = full_vocab_margins.detach().float().cpu()
                    all_full_vocab_first_token_margins.append(detached_full_vocab_margins)
                    pair_full_vocab_margins.append(detached_full_vocab_margins)
                gate_metrics_by_pair[pair_id] = pair_gate_metrics(
                    torch.cat(pair_margins, dim=0),
                    changed_unit_accuracy_threshold=changed_unit_accuracy_threshold,
                    prediction_flip_threshold=prediction_flip_threshold,
                    wrong_prefix_flip_threshold=wrong_prefix_flip_threshold,
                    ranking_margin=ranking_margin,
                    ranking_mode=ranking_mode,
                    first_answer_token_full_vocab_margins=torch.cat(pair_full_vocab_margins, dim=0),
                    first_answer_token_top1_accuracy_threshold=(
                        first_answer_token_top1_accuracy_threshold
                    ),
                )
                del outputs
    finally:
        for module, was_training in zip(modules, previous_modes, strict=True):
            module.train(was_training)
        if lora_installation is not None:
            assert previous_lora_mode is not None
            lora_installation.train(previous_lora_mode)
    metrics = pair_gate_metrics(
        torch.cat(all_margins, dim=0),
        changed_unit_accuracy_threshold=changed_unit_accuracy_threshold,
        prediction_flip_threshold=prediction_flip_threshold,
        wrong_prefix_flip_threshold=wrong_prefix_flip_threshold,
        ranking_margin=ranking_margin,
        ranking_mode=ranking_mode,
        first_answer_token_full_vocab_margins=torch.cat(all_full_vocab_first_token_margins, dim=0),
        first_answer_token_top1_accuracy_threshold=(first_answer_token_top1_accuracy_threshold),
    )
    metrics["pair_count"] = len(by_pair)
    metrics["by_pair"] = gate_metrics_by_pair
    if ranking_mode == "candidate_logit":
        candidate_token_ids: list[list[list[int]]] = []
        for unit in ordered_units:
            reference, counterfactual = unit.records
            reference_answer_ids = tokenize_answer(
                language.tokenizer, reference.answer, language.device
            )
            counterfactual_answer_ids = tokenize_answer(
                language.tokenizer, counterfactual.answer, language.device
            )
            _, reference_token_id, counterfactual_token_id = single_differing_answer_token(
                reference_answer_ids, counterfactual_answer_ids
            )
            candidate_token_ids.append(
                [
                    [reference_token_id, counterfactual_token_id],
                    [counterfactual_token_id, reference_token_id],
                ]
            )
        metrics["detail"] = build_candidate_gate_detail(
            ordered_units,
            torch.cat(all_margins, dim=0),
            ranking_margin=ranking_margin,
            candidate_token_ids=candidate_token_ids,
            full_vocab_margins=torch.cat(all_full_vocab_first_token_margins, dim=0),
        )
    return metrics


def validation_loss(
    records_by_scene: dict[str, list[QARecord]],
    *,
    config: dict,
    language,
    scene_model: SceneTokenizer,
    global_scene_residual: GlobalSceneResidual | None = None,
    signed_x_scene_residual: SignedXSceneResidual | None = None,
    composer: ContinuousPrefixComposer,
    grounding: QuestionGroundingHead,
    semantic_dim: int,
    batch_size: int,
    lora_installation: LoRABankCollection | LoRAInstallation | None = None,
    dense_aligner: torch.nn.Module | None = None,
    dense_sidecar_adapter: DenseSidecarAdapter | None = None,
) -> dict[str, float] | None:
    """Evaluate held-out teacher-forced loss while loading one scene map at a time."""

    if not records_by_scene:
        return None
    modules = tuple(
        module
        for module in (
            scene_model,
            global_scene_residual,
            signed_x_scene_residual,
            dense_aligner,
            dense_sidecar_adapter,
            composer,
            grounding,
        )
        if module is not None
    )
    previous_modes = [module.training for module in modules]
    previous_lora_mode = None if lora_installation is None else lora_installation.training
    for module in modules:
        module.eval()
    if lora_installation is not None:
        lora_installation.eval()
    weighted_loss = 0.0
    weighted_language = 0.0
    weighted_grounding = 0.0
    question_count = 0
    try:
        with torch.inference_mode():
            for scene_id in sorted(records_by_scene):
                map_path = project_path(config, "maps", scene_id, "voxel_map.npz")
                data = load_map_tensors(
                    map_path,
                    config["scene"]["room_size_m"],
                    language.device,
                    input_voxel_size_m=config["scene_encoder"].get("input_voxel_size_m"),
                )
                if data.feature_dim != semantic_dim:
                    raise ValueError(
                        f"Validation scene {scene_id} feature dimension {data.feature_dim} "
                        f"does not match training dimension {semantic_dim}"
                    )
                output = map_forward(
                    scene_model,
                    data,
                    global_scene_residual,
                    signed_x_scene_residual,
                    dense_aligner,
                    dense_sidecar_adapter,
                )
                records = records_by_scene[scene_id]
                for offset in range(0, len(records), batch_size):
                    batch_records = records[offset : offset + batch_size]
                    loss, language_loss, grounding_loss = batch_objective(
                        output,
                        batch_records,
                        data,
                        language,
                        composer,
                        grounding,
                        config,
                    )
                    weight = len(batch_records)
                    weighted_loss += float(loss.cpu()) * weight
                    weighted_language += float(language_loss.cpu()) * weight
                    weighted_grounding += float(grounding_loss.cpu()) * weight
                    question_count += weight
                del output, data
                if language.device.type == "mps":
                    torch.mps.empty_cache()
    finally:
        for module, was_training in zip(modules, previous_modes, strict=True):
            module.train(was_training)
        if lora_installation is not None:
            assert previous_lora_mode is not None
            lora_installation.train(previous_lora_mode)
    if not question_count:
        return None
    return {
        "loss": weighted_loss / question_count,
        "language_loss": weighted_language / question_count,
        "grounding_loss": weighted_grounding / question_count,
        "question_count": float(question_count),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-questions", type=int)
    parser.add_argument("--max-questions-per-scene", type=int)
    parser.add_argument("--output-namespace")
    parser.add_argument(
        "--selection-only",
        action="store_true",
        help="Write the deterministic training-selection audit and exit before model load",
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--initialize-from",
        type=Path,
        help="Load compatible adapter weights for a new curriculum without optimizer/history",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    set_seed(int(config["seed"]))
    v18_stage_execution = v18_stage_execution_metadata(config)
    scene_prefix_after_bos = scene_prefix_after_bos_setting(config)
    scene_boundary_mode = scene_boundary_mode_setting(config)
    configured_native_boundary_contract = native_gemma4_image_contract_setting(config)
    configured_lora = lora_banks_settings(config)
    configured_lora_optimizer = lora_banks_optimizer_settings(config, configured_lora)
    residual_settings = global_scene_residual_settings(config)
    signed_x_residual_settings = signed_x_scene_residual_settings(config)
    dense_settings = dense_alignment_settings(config)
    declared_residual_parameter_count = declared_global_scene_residual_parameter_count(config)
    declared_signed_x_parameter_count = declared_signed_x_scene_residual_parameter_count(config)
    declared_dense_parameter_count = declared_dense_alignment_parameter_count(config)
    freeze_scene_adapter = config["training"].get("freeze_scene_adapter", False)
    if not isinstance(freeze_scene_adapter, bool):
        raise TypeError("training.freeze_scene_adapter must be a boolean")
    train_global_scene_residual_only = config["training"].get(
        "train_global_scene_residual_only", False
    )
    if not isinstance(train_global_scene_residual_only, bool):
        raise TypeError("training.train_global_scene_residual_only must be a boolean")
    train_signed_x_scene_residual_only = config["training"].get(
        "train_signed_x_scene_residual_only", False
    )
    if not isinstance(train_signed_x_scene_residual_only, bool):
        raise TypeError("training.train_signed_x_scene_residual_only must be a boolean")
    train_lora_with_frozen_scene_residual_stack = config["training"].get(
        "train_lora_with_frozen_scene_residual_stack", False
    )
    if not isinstance(train_lora_with_frozen_scene_residual_stack, bool):
        raise TypeError("training.train_lora_with_frozen_scene_residual_stack must be a boolean")
    train_dense_alignment_only = config["training"].get("train_dense_alignment_only", False)
    if not isinstance(train_dense_alignment_only, bool):
        raise TypeError("training.train_dense_alignment_only must be a boolean")
    exclusive_training_modes = sum(
        (
            train_global_scene_residual_only,
            train_signed_x_scene_residual_only,
            train_lora_with_frozen_scene_residual_stack,
            train_dense_alignment_only,
        )
    )
    if exclusive_training_modes > 1:
        raise ValueError(
            "Global-residual-only, signed-X-only, frozen-residual-stack LoRA, and dense-alignment "
            "training are mutually exclusive"
        )
    if residual_settings.enabled != (
        train_global_scene_residual_only
        or train_signed_x_scene_residual_only
        or train_lora_with_frozen_scene_residual_stack
        or train_dense_alignment_only
    ):
        raise ValueError(
            "Training residual-base enablement must be explicit and exclusive: "
            f"residual_enabled={residual_settings.enabled} "
            f"train_global_scene_residual_only={train_global_scene_residual_only} "
            f"train_signed_x_scene_residual_only={train_signed_x_scene_residual_only} "
            "train_lora_with_frozen_scene_residual_stack="
            f"{train_lora_with_frozen_scene_residual_stack} "
            f"train_dense_alignment_only={train_dense_alignment_only}"
        )
    if signed_x_residual_settings.enabled != (
        train_signed_x_scene_residual_only
        or train_lora_with_frozen_scene_residual_stack
        or train_dense_alignment_only
    ):
        raise ValueError(
            "Training signed-X enablement must be explicit and exclusive: "
            f"signed_x_enabled={signed_x_residual_settings.enabled} "
            f"train_signed_x_scene_residual_only={train_signed_x_scene_residual_only} "
            "train_lora_with_frozen_scene_residual_stack="
            f"{train_lora_with_frozen_scene_residual_stack} "
            f"train_dense_alignment_only={train_dense_alignment_only}"
        )
    if dense_settings.enabled != train_dense_alignment_only:
        raise ValueError(
            "Dense-alignment enablement must exactly match its exclusive training mode: "
            f"dense_alignment_enabled={dense_settings.enabled} "
            f"train_dense_alignment_only={train_dense_alignment_only}"
        )
    if train_signed_x_scene_residual_only and (
        residual_settings.architecture_version != ZERO_SPATIAL_MEAN_CONTENT_GATE_V1
    ):
        raise ValueError("Signed-X training requires the centered-content V18 residual base")
    if train_global_scene_residual_only and not freeze_scene_adapter:
        raise ValueError("Residual-only training requires the core scene adapter to be frozen")
    if train_global_scene_residual_only and configured_lora.trainable:
        raise ValueError("Residual-only training requires every language LoRA bank to be frozen")
    if train_signed_x_scene_residual_only and not freeze_scene_adapter:
        raise ValueError("Signed-X-only training requires the core scene adapter to be frozen")
    if train_signed_x_scene_residual_only and configured_lora.trainable:
        raise ValueError("Signed-X-only training requires every language LoRA bank to be frozen")
    if train_lora_with_frozen_scene_residual_stack:
        if not freeze_scene_adapter:
            raise ValueError("Frozen-residual-stack LoRA training requires a frozen scene adapter")
        if not configured_lora.trainable:
            raise ValueError(
                "Frozen-residual-stack LoRA training requires at least one trainable LoRA bank"
            )
    if train_dense_alignment_only:
        if not freeze_scene_adapter:
            raise ValueError("Dense-alignment-only training requires a frozen scene adapter")
        if configured_lora.trainable:
            raise ValueError("Dense-alignment-only training requires every LoRA bank to be frozen")
        if residual_settings.architecture_version != ZERO_SPATIAL_MEAN_CONTENT_GATE_V1:
            raise ValueError(
                "Dense-alignment-only training requires the centered-content V18 residual base"
            )
    initialize_legacy_lora_into_bank = config["training"].get("initialize_legacy_lora_into_bank")
    if initialize_legacy_lora_into_bank is not None and (
        not isinstance(initialize_legacy_lora_into_bank, str)
        or not initialize_legacy_lora_into_bank
    ):
        raise TypeError("training.initialize_legacy_lora_into_bank must be a non-empty string")
    initialize_named_lora_freeze_transition = config["training"].get(
        "initialize_named_lora_freeze_transition", False
    )
    if not isinstance(initialize_named_lora_freeze_transition, bool):
        raise TypeError("training.initialize_named_lora_freeze_transition must be a boolean")
    if initialize_named_lora_freeze_transition and initialize_legacy_lora_into_bank is not None:
        raise ValueError("Named-bank freeze transition and legacy-bank aliasing are exclusive")
    if initialize_named_lora_freeze_transition and not train_global_scene_residual_only:
        raise ValueError("Named-bank freeze transition is restricted to residual-only training")
    initialize_named_lora_freeze_and_extend_transition = config["training"].get(
        "initialize_named_lora_freeze_and_extend_transition", False
    )
    if not isinstance(initialize_named_lora_freeze_and_extend_transition, bool):
        raise TypeError(
            "training.initialize_named_lora_freeze_and_extend_transition must be a boolean"
        )
    if initialize_named_lora_freeze_and_extend_transition and not (
        train_lora_with_frozen_scene_residual_stack
    ):
        raise ValueError(
            "Named-bank freeze-and-extend transition is restricted to frozen-residual-stack "
            "LoRA training"
        )
    if initialize_named_lora_freeze_and_extend_transition and (
        initialize_named_lora_freeze_transition
        or initialize_legacy_lora_into_bank is not None
    ):
        raise ValueError(
            "Named-bank freeze-and-extend transition is exclusive with legacy aliasing and "
            "the residual-only named-bank freeze transition"
        )
    initialize_named_lora_freeze_for_dense_alignment_transition = config["training"].get(
        "initialize_named_lora_freeze_for_dense_alignment_transition", False
    )
    if not isinstance(initialize_named_lora_freeze_for_dense_alignment_transition, bool):
        raise TypeError(
            "training.initialize_named_lora_freeze_for_dense_alignment_transition must be "
            "a boolean"
        )
    if initialize_named_lora_freeze_for_dense_alignment_transition and not (
        train_dense_alignment_only
    ):
        raise ValueError(
            "Named-LoRA dense-alignment freeze transition is restricted to "
            "dense-alignment-only training"
        )
    if initialize_named_lora_freeze_for_dense_alignment_transition and (
        initialize_named_lora_freeze_transition
        or initialize_named_lora_freeze_and_extend_transition
        or initialize_legacy_lora_into_bank is not None
    ):
        raise ValueError(
            "Dense-alignment source transition is exclusive with every other named/legacy "
            "LoRA transition"
        )
    initialize_source_residual_into_frozen_base = config["training"].get(
        "initialize_source_residual_into_frozen_base", False
    )
    if not isinstance(initialize_source_residual_into_frozen_base, bool):
        raise TypeError("training.initialize_source_residual_into_frozen_base must be a boolean")
    if initialize_source_residual_into_frozen_base and not train_signed_x_scene_residual_only:
        raise ValueError(
            "Source-residual freeze transition is restricted to signed-X-only training"
        )
    if initialize_source_residual_into_frozen_base and initialize_named_lora_freeze_transition:
        raise ValueError("Residual-base and named-LoRA freeze transitions are mutually exclusive")
    if (
        initialize_source_residual_into_frozen_base
        and initialize_named_lora_freeze_and_extend_transition
    ):
        raise ValueError(
            "Residual-base and named-LoRA freeze-and-extend transitions are mutually exclusive"
        )
    if (
        initialize_source_residual_into_frozen_base
        and initialize_named_lora_freeze_for_dense_alignment_transition
    ):
        raise ValueError(
            "Residual-base and dense-alignment source transitions are mutually exclusive"
        )
    initialize_expected_adapter_sha256 = optional_sha256_setting(
        config["training"], "initialize_expected_adapter_sha256"
    )
    initialize_expected_metadata_sha256 = optional_sha256_setting(
        config["training"], "initialize_expected_metadata_sha256"
    )
    initialize_expected_scene_state_sha256 = optional_sha256_setting(
        config["training"], "initialize_expected_scene_state_sha256"
    )
    initialize_expected_global_scene_residual_state_sha256 = optional_sha256_setting(
        config["training"],
        "initialize_expected_global_scene_residual_state_sha256",
    )
    initialize_expected_signed_x_scene_residual_state_sha256 = optional_sha256_setting(
        config["training"],
        "initialize_expected_signed_x_scene_residual_state_sha256",
    )
    initialize_expected_dense_alignment_state_sha256 = optional_sha256_setting(
        config["training"],
        "initialize_expected_dense_alignment_state_sha256",
    )
    if (
        initialize_source_residual_into_frozen_base
        and initialize_expected_global_scene_residual_state_sha256 is None
    ):
        raise ValueError(
            "Source-residual freeze transition requires "
            "training.initialize_expected_global_scene_residual_state_sha256"
        )
    if (train_lora_with_frozen_scene_residual_stack or train_dense_alignment_only) and (
        initialize_expected_scene_state_sha256 is None
        or initialize_expected_global_scene_residual_state_sha256 is None
        or initialize_expected_signed_x_scene_residual_state_sha256 is None
    ):
        raise ValueError(
            "Frozen-stack LoRA/dense-alignment training requires exact source scene, global, "
            "and signed-X residual state hashes"
        )
    if train_dense_alignment_only and not (
        initialize_named_lora_freeze_for_dense_alignment_transition
    ):
        raise ValueError(
            "Dense-alignment-only training requires the explicit named-LoRA source freeze "
            "transition"
        )
    if train_dense_alignment_only:
        if initialize_expected_dense_alignment_state_sha256 is None:
            raise ValueError(
                "Dense-alignment-only training requires "
                "training.initialize_expected_dense_alignment_state_sha256"
            )
        if initialize_expected_dense_alignment_state_sha256 != (
            dense_settings.expected_initial_state_sha256
        ):
            raise ValueError(
                "Configured dense-alignment initialization pins disagree: "
                f"training={initialize_expected_dense_alignment_state_sha256} "
                f"scene_encoder={dense_settings.expected_initial_state_sha256}"
            )
    source_provenance = capture_git_source_provenance(PROJECT_ROOT)
    language_decoder_gradient_checkpointing = config["training"].get(
        "language_decoder_gradient_checkpointing", False
    )
    if not isinstance(language_decoder_gradient_checkpointing, bool):
        raise TypeError("language_decoder_gradient_checkpointing must be a boolean")
    gradient_accumulation = int(config["training"]["gradient_accumulation"])
    if gradient_accumulation < 1:
        raise ValueError("gradient_accumulation must be positive")
    pair_curriculum = pair_curriculum_settings(config)
    pair_objectives = pair_objective_policy_settings(config)
    pair_gate_monitor_name = (
        "pair_composite_full_vocab_gate_margin"
        if pair_curriculum.first_answer_token_top1_accuracy_threshold is not None
        else "pair_candidate_gate_hinge"
    )
    qa_root = artifact_root(config, "qa")
    dataset = load_qa_split_dataset(qa_root, "train")
    if not len(dataset):
        raise SystemExit("Training set is empty; run make generate-dataset")
    available_training_records = list(dataset.records)
    if pair_curriculum.pair_only:
        available_training_records = select_pair_only_records(
            available_training_records, pair_curriculum.pair_only_scene_ids
        )
        available_training_records = cap_pair_units_per_pair(
            available_training_records,
            pair_curriculum.max_units_per_pair,
            seed=int(config["seed"]),
        )
    configured_per_scene_cap = config["training"].get("max_questions_per_scene")
    per_scene_cap = (
        args.max_questions_per_scene
        if args.max_questions_per_scene is not None
        else configured_per_scene_cap
    )
    records = select_training_records(
        available_training_records,
        max_questions=args.max_questions,
        max_questions_per_scene=None if per_scene_cap is None else int(per_scene_cap),
    )
    if not records:
        raise SystemExit("Training question selection is empty")
    by_scene: dict[str, list[QARecord]] = defaultdict(list)
    for record in records:
        by_scene[record.scene_id].append(record)
    anti_collapse = anti_collapse_settings(config)
    spatial_answer_contrastive = spatial_answer_contrastive_settings(config)
    spatial_answer_warmup = spatial_answer_warmup_settings(config)
    spatial_relation_contrastive = spatial_relation_contrastive_settings(config)
    spatial_relation_warmup = spatial_relation_warmup_settings(config)
    if spatial_answer_contrastive["weight"] > 0 and not pair_curriculum.enabled:
        raise ValueError("spatial_answer_contrastive_weight requires an enabled pair curriculum")
    if spatial_answer_warmup["steps"] > 0 and not pair_curriculum.enabled:
        raise ValueError("spatial_answer_warmup_steps requires an enabled pair curriculum")
    if spatial_relation_contrastive["weight"] > 0 and not pair_curriculum.enabled:
        raise ValueError("spatial_relation_contrastive_weight requires an enabled pair curriculum")
    if spatial_relation_warmup["steps"] > 0 and not pair_curriculum.enabled:
        raise ValueError("spatial_relation_warmup_steps requires an enabled pair curriculum")
    if freeze_scene_adapter:
        frozen_scene_objectives = {
            "grounding_weight": float(config["training"].get("grounding_weight", 0.0)),
            "grounding_anchor_weight": float(
                config["training"].get("grounding_anchor_weight", 0.0)
            ),
            "latent_diversity_weight": float(anti_collapse["latent_diversity_weight"]),
            "paired_scene_separation_weight": float(
                anti_collapse["paired_scene_separation_weight"]
            ),
            "spatial_answer_contrastive_weight": float(spatial_answer_contrastive["weight"]),
            "spatial_answer_warmup_steps": int(spatial_answer_warmup["steps"]),
            "spatial_relation_contrastive_weight": float(spatial_relation_contrastive["weight"]),
            "spatial_relation_warmup_steps": int(spatial_relation_warmup["steps"]),
        }
        active = {key: value for key, value in frozen_scene_objectives.items() if value != 0}
        if active:
            raise ValueError(
                f"Frozen scene adapter requires all scene-only objectives to be disabled: {active}"
            )
        if (
            not configured_lora.trainable
            and not train_global_scene_residual_only
            and not train_signed_x_scene_residual_only
            and not train_dense_alignment_only
        ):
            raise ValueError(
                "Frozen scene adapter requires a trainable LoRA bank or the explicit "
                "global-scene-residual-only/signed-X-only/dense-alignment-only path"
            )
    token_mixing = scene_token_mixing_settings(config)
    pair_units = build_exact_question_pair_units(records)
    validate_pair_objective_training_mode(
        configured=pair_objectives.configured,
        curriculum_enabled=pair_curriculum.enabled,
        pair_only=pair_curriculum.pair_only,
    )
    if pair_curriculum.enabled and not pair_units:
        raise ValueError("The pair curriculum is enabled but selection contains no pair units")
    pair_objective_coverage = validate_pair_objective_policy_coverage(
        pair_objectives,
        sorted({unit.pair_id for unit in pair_units}),
    )
    resolved_pair_objective_contract = pair_objective_policy_contract(pair_objectives)
    spatial_answer_targets = spatial_answer_target_audit(pair_units)
    spatial_answer_warmup_targets = spatial_answer_warmup_target_audit(pair_units)
    spatial_relation_targets = spatial_relation_target_audit(pair_units)
    spatial_relation_warmup_targets = spatial_relation_warmup_target_audit(pair_units)
    if (
        spatial_answer_warmup["steps"] > 0
        and int(spatial_answer_warmup_targets["deduplicated_unit_count"]) == 0
    ):
        raise ValueError("Enabled spatial-answer warmup has no grounded pair targets")
    if (
        spatial_relation_warmup["steps"] > 0
        and int(spatial_relation_warmup_targets["deduplicated_unit_count"]) == 0
    ):
        raise ValueError("Enabled spatial-relation warmup has no ordered relation targets")
    training_pairs = training_counterfactual_scene_pairs(records)
    pair_for_first_scene = {
        first_scene: (pair_id, second_scene)
        for pair_id, first_scene, second_scene in training_pairs
    }
    pair_membership_text = "\n".join(
        f"{pair_id}:{first_scene}:{second_scene}"
        for pair_id, first_scene, second_scene in training_pairs
    )
    pair_membership_sha256 = hashlib.sha256(pair_membership_text.encode("utf-8")).hexdigest()
    pair_unit_selection = [
        {
            "pair_id": unit.pair_id,
            "question_key": unit.question_key,
            "scene_ids": list(unit.scene_ids),
            "question_ids": [record.question_id for record in unit.records],
        }
        for unit in sorted(
            pair_units,
            key=lambda item: (
                item.pair_id,
                item.question_key,
                item.reference.question_id,
                item.counterfactual.question_id,
            ),
        )
    ]
    pair_unit_selection_sha256 = hashlib.sha256(
        json.dumps(
            pair_unit_selection,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    split_ids = split_scene_ids(qa_root, records)
    validation_path = qa_root / "validation.jsonl"
    validation_records: list[QARecord] = []
    if validation_path.is_file() and not pair_curriculum.pair_only:
        validation_dataset = load_qa_split_dataset(qa_root, "validation")
        validation_cap = config["training"].get("max_validation_questions_per_scene", per_scene_cap)
        validation_records = select_training_records(
            validation_dataset.records,
            max_questions_per_scene=(None if validation_cap is None else int(validation_cap)),
        )
    validation_by_scene: dict[str, list[QARecord]] = defaultdict(list)
    for record in validation_records:
        validation_by_scene[record.scene_id].append(record)
    overlap = set(by_scene) & set(validation_by_scene)
    if overlap:
        raise ValueError(f"Training and validation scenes overlap: {sorted(overlap)}")

    configured_namespace = config["training"].get("output_namespace")
    output_namespace = validate_output_namespace(
        args.output_namespace if args.output_namespace is not None else configured_namespace
    )
    output_root, metrics_path, loss_figure_path = training_artifact_paths(config, output_namespace)
    selection_report = {
        "schema_version": 1,
        "strategy": "paired_expected_change_then_least_represented_answer_type_v1",
        "train": training_selection_summary(available_training_records, records),
        "validation": (
            training_selection_summary(validation_dataset.records, validation_records)
            if validation_path.is_file() and not pair_curriculum.pair_only
            else None
        ),
        "train_scene_ids": sorted(by_scene),
        "validation_scene_ids": sorted(validation_by_scene),
        "test_scene_ids": split_ids["test"],
        "training_counterfactual_pair_count": len(training_pairs),
        "training_counterfactual_pair_membership_sha256": pair_membership_sha256,
        "counterfactual_pair_unit_count": len(pair_units),
        "counterfactual_pair_unit_selection_sha256": pair_unit_selection_sha256,
        "scene_token_mixing": token_mixing,
        "spatial_answer_contrastive": spatial_answer_contrastive,
        "spatial_answer_target_audit": spatial_answer_targets,
        "spatial_answer_warmup": spatial_answer_warmup,
        "spatial_answer_warmup_target_audit": spatial_answer_warmup_targets,
        "spatial_relation_contrastive": spatial_relation_contrastive,
        "spatial_relation_target_audit": spatial_relation_targets,
        "spatial_relation_warmup": spatial_relation_warmup,
        "spatial_relation_warmup_target_audit": spatial_relation_warmup_targets,
        "language_decoder_gradient_checkpointing": (language_decoder_gradient_checkpointing),
        "freeze_scene_adapter": freeze_scene_adapter,
        "train_global_scene_residual_only": train_global_scene_residual_only,
        "train_signed_x_scene_residual_only": train_signed_x_scene_residual_only,
        "train_lora_with_frozen_scene_residual_stack": (
            train_lora_with_frozen_scene_residual_stack
        ),
        "train_dense_alignment_only": train_dense_alignment_only,
        "global_scene_residual": residual_settings.contract(),
        "signed_x_scene_residual": signed_x_residual_settings.contract(),
        "dense_alignment": dense_settings.contract(),
        "dense_alignment_optimizer": (
            {
                "name": "AdamW",
                "learning_rate": float(config["training"]["dense_alignment_learning_rate"]),
                "weight_decay": float(config["training"]["dense_alignment_weight_decay"]),
            }
            if train_dense_alignment_only
            else None
        ),
        "initialize_legacy_lora_into_bank": initialize_legacy_lora_into_bank,
        "initialize_named_lora_freeze_transition": initialize_named_lora_freeze_transition,
        "initialize_named_lora_freeze_and_extend_transition": (
            initialize_named_lora_freeze_and_extend_transition
        ),
        "initialize_named_lora_freeze_for_dense_alignment_transition": (
            initialize_named_lora_freeze_for_dense_alignment_transition
        ),
        "initialize_source_residual_into_frozen_base": (
            initialize_source_residual_into_frozen_base
        ),
        "initialize_expected_adapter_sha256": initialize_expected_adapter_sha256,
        "initialize_expected_metadata_sha256": initialize_expected_metadata_sha256,
        "initialize_expected_scene_state_sha256": initialize_expected_scene_state_sha256,
        "initialize_expected_global_scene_residual_state_sha256": (
            initialize_expected_global_scene_residual_state_sha256
        ),
        "initialize_expected_signed_x_scene_residual_state_sha256": (
            initialize_expected_signed_x_scene_residual_state_sha256
        ),
        "initialize_expected_dense_alignment_state_sha256": (
            initialize_expected_dense_alignment_state_sha256
        ),
        "scene_prefix_after_bos": scene_prefix_after_bos,
        "scene_boundary_mode": scene_boundary_mode,
        "gemma4_native_image_contract": configured_native_boundary_contract,
        "source_provenance": source_provenance,
        "gradient_accumulation": gradient_accumulation,
        "pair_curriculum": {
            "enabled": pair_curriculum.enabled,
            "pair_only": pair_curriculum.pair_only,
            "pair_only_scene_ids": list(pair_curriculum.pair_only_scene_ids),
            "max_units_per_pair": pair_curriculum.max_units_per_pair,
            "batch_fraction": pair_curriculum.batch_fraction,
            "units_per_batch": pair_curriculum.units_per_batch,
            "ranking_mode": pair_curriculum.ranking_mode,
            "ranking_margin": pair_curriculum.ranking_margin,
            "ranking_weight": pair_curriculum.ranking_weight,
            "full_vocab_ranking_margin": pair_curriculum.full_vocab_ranking_margin,
            "full_vocab_ranking_weight": pair_curriculum.full_vocab_ranking_weight,
            "gate_enabled": pair_curriculum.gate_enabled,
            "gate_every_epochs": pair_curriculum.gate_every_epochs,
            "gate_stop_when_passed": pair_curriculum.stop_when_gate_passes,
            "gate_first_answer_token_top1_accuracy": (
                pair_curriculum.first_answer_token_top1_accuracy_threshold
            ),
            "objective_policy": resolved_pair_objective_contract,
            "objective_policy_coverage": pair_objective_coverage,
        },
    }
    if configured_lora.enabled:
        selection_report.update(
            {
                "lora": configured_lora.contract(),
                "lora_optimizer": (
                    None
                    if configured_lora_optimizer is None
                    else configured_lora_optimizer.contract()
                ),
            }
        )
    selection_name = (
        "training_selection.json"
        if output_namespace is None
        else f"training_selection_{output_namespace}.json"
    )
    selection_path = metrics_path.parent / selection_name
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    selection_path.write_text(json.dumps(selection_report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "phase": "training_selection",
                "report": str(selection_path.relative_to(PROJECT_ROOT)),
                **selection_report["train"],
            }
        ),
        flush=True,
    )
    if args.selection_only:
        return
    if configured_lora.enabled:
        require_clean_committed_source(source_provenance)

    language = load_local_language_model(
        config["language"]["model_id"],
        config["language"]["revision"],
        config["language"]["dtype"],
        freeze=True,
        local_files_only=True,
        backend=str(config["language"].get("backend", "auto")),
        decoder_gradient_checkpointing=language_decoder_gradient_checkpointing,
    )
    loaded_native_boundary_contract = language.scene_boundary_contract(scene_boundary_mode)
    if loaded_native_boundary_contract != configured_native_boundary_contract:
        raise ValueError(
            "Loaded language model does not satisfy configured scene-boundary contract: "
            f"loaded={loaded_native_boundary_contract} "
            f"configured={configured_native_boundary_contract}"
        )
    lora_installation = install_lora_banks(language.model, configured_lora)
    configured_lora_checkpoint_contract = lora_banks_checkpoint_contract(
        configured_lora,
        configured_lora_optimizer,
        {} if lora_installation is None else lora_installation.parameter_counts,
    )
    if lora_installation is not None:
        print(
            json.dumps(
                {
                    "phase": "lora_installed",
                    "contract": configured_lora_checkpoint_contract,
                    "wrapped_modules": lora_installation.wrapped_modules,
                    "parameter_counts": lora_installation.parameter_counts,
                    "trainable_parameter_count": lora_installation.trainable_parameter_count,
                    "optimizer": (
                        None
                        if configured_lora_optimizer is None
                        else configured_lora_optimizer.contract()
                    ),
                }
            ),
            flush=True,
        )
    loaded_native_boundary_embeddings = language.scene_boundary_embeddings(scene_boundary_mode)
    language.model.config.use_cache = False
    maps: dict[str, MapTensorData] = {}
    for scene_id in by_scene:
        path = project_path(config, "maps", scene_id, "voxel_map.npz")
        maps[scene_id] = load_map_tensors(
            path,
            config["scene"]["room_size_m"],
            language.device,
            input_voxel_size_m=config["scene_encoder"].get("input_voxel_size_m"),
        )
    feature_dims = {data.feature_dim for data in maps.values()}
    if len(feature_dims) != 1:
        raise ValueError(f"Scenes use inconsistent feature dimensions: {feature_dims}")
    semantic_dim = feature_dims.pop()
    scene_model = construct_scene_tokenizer(config, semantic_dim, language.hidden_size).to(
        language.device
    )
    dense_aligner = construct_dense_alignment(config, semantic_dim=semantic_dim)
    if dense_aligner is not None:
        dense_aligner = dense_aligner.to(language.device)
        dense_audit = validate_dense_alignment_state(
            dense_aligner,
            expected_parameter_count=declared_dense_parameter_count,
            context="deterministic construction",
        )
        observed_initial_dense_alignment_sha256 = str(dense_audit["state_sha256"])
        if observed_initial_dense_alignment_sha256 != (
            dense_settings.expected_initial_state_sha256
        ):
            raise ValueError(
                "Dense-alignment deterministic initial-state hash mismatch: "
                f"expected={dense_settings.expected_initial_state_sha256} "
                f"observed={observed_initial_dense_alignment_sha256}"
            )
        if not bool(dense_audit["b_exact_zero"]):
            raise ValueError("Dense alignment is not exact zero-output at initialization")
    else:
        if declared_dense_parameter_count is not None:
            raise ValueError(
                "experiment dense-alignment parameter count requires enabled dense alignment"
            )
        observed_initial_dense_alignment_sha256 = None
    global_scene_residual = construct_global_scene_residual(
        config,
        scene_dim=language.hidden_size,
        latent_count=int(config["scene_encoder"]["global_latents"]),
    )
    if global_scene_residual is not None:
        global_scene_residual = global_scene_residual.to(language.device)
        validate_global_scene_residual_state(
            global_scene_residual,
            expected_parameter_count=declared_residual_parameter_count,
            context="deterministic construction",
        )
        observed_initial_residual_sha256 = module_collection_state_sha256(
            {"global_scene_residual": global_scene_residual}
        )
        if observed_initial_residual_sha256 != residual_settings.expected_initial_state_sha256:
            raise ValueError(
                "Global scene residual deterministic initial-state hash mismatch: "
                f"expected={residual_settings.expected_initial_state_sha256} "
                f"observed={observed_initial_residual_sha256}"
            )
        if torch.count_nonzero(global_scene_residual.output_projection.weight).item() != 0:
            raise ValueError("Global scene residual is not exact zero-output at initialization")
    else:
        if declared_residual_parameter_count is not None:
            raise ValueError(
                "experiment.residual_parameter_count requires an enabled global scene residual"
            )
        observed_initial_residual_sha256 = None
    signed_x_scene_residual = construct_signed_x_scene_residual(
        config,
        scene_dim=language.hidden_size,
        latent_count=int(config["scene_encoder"]["global_latents"]),
        content_dim=residual_settings.width,
    )
    if signed_x_scene_residual is not None:
        signed_x_scene_residual = signed_x_scene_residual.to(language.device)
        validate_signed_x_scene_residual_state(
            signed_x_scene_residual,
            expected_parameter_count=declared_signed_x_parameter_count,
            context="deterministic construction",
        )
        observed_initial_signed_x_sha256 = module_collection_state_sha256(
            {"signed_x_scene_residual": signed_x_scene_residual}
        )
        if (
            observed_initial_signed_x_sha256
            != signed_x_residual_settings.expected_initial_state_sha256
        ):
            raise ValueError(
                "Signed-X residual deterministic initial-state hash mismatch: "
                f"expected={signed_x_residual_settings.expected_initial_state_sha256} "
                f"observed={observed_initial_signed_x_sha256}"
            )
        if torch.count_nonzero(signed_x_scene_residual.output_projection.weight).item() != 0:
            raise ValueError("Signed-X scene residual is not exact zero-output at initialization")
    else:
        if declared_signed_x_parameter_count is not None:
            raise ValueError(
                "experiment.signed_x_residual_parameter_count requires an enabled "
                "signed-X scene residual"
            )
        observed_initial_signed_x_sha256 = None
    composer = ContinuousPrefixComposer(
        language.hidden_size,
        scene_prefix_after_bos=scene_prefix_after_bos,
        bos_token_id=language.bos_token_id,
        scene_boundary_mode=scene_boundary_mode,
        native_boundary_embeddings=loaded_native_boundary_embeddings,
    ).to(language.device)
    grounding = QuestionGroundingHead(
        int(config["scene_encoder"]["model_dim"]),
        language.hidden_size,
        int(config["scene_encoder"]["global_latents"]),
        int(config["scene_encoder"]["model_dim"]),
    ).to(language.device)
    if freeze_scene_adapter:
        scene_model.requires_grad_(False).eval()
        composer.requires_grad_(False).eval()
        grounding.requires_grad_(False).eval()
    if train_signed_x_scene_residual_only:
        if global_scene_residual is None or signed_x_scene_residual is None:
            raise RuntimeError("Signed-X-only training lost one of its residual modules")
        global_scene_residual.requires_grad_(False).eval()
    if train_lora_with_frozen_scene_residual_stack:
        if global_scene_residual is None or signed_x_scene_residual is None:
            raise RuntimeError("Frozen-residual-stack LoRA training lost a residual module")
        global_scene_residual.requires_grad_(False).eval()
        signed_x_scene_residual.requires_grad_(False).eval()
    if train_dense_alignment_only:
        if (
            dense_aligner is None
            or global_scene_residual is None
            or signed_x_scene_residual is None
        ):
            raise RuntimeError("Dense-alignment-only training lost part of its frozen source stack")
        scene_model.requires_grad_(False).eval()
        composer.requires_grad_(False).eval()
        grounding.requires_grad_(False).eval()
        global_scene_residual.requires_grad_(False).eval()
        signed_x_scene_residual.requires_grad_(False).eval()
        dense_aligner.requires_grad_(True).train()
    scene_parameters = (
        list(scene_model.parameters()) + list(composer.parameters()) + list(grounding.parameters())
    )
    if global_scene_residual is not None:
        scene_parameters += list(global_scene_residual.parameters())
    if signed_x_scene_residual is not None:
        scene_parameters += list(signed_x_scene_residual.parameters())
    optimizer, parameters = build_adapter_optimizer(
        config,
        scene_parameters,
        lora_installation,
        configured_lora_optimizer,
        dense_alignment_parameters=(
            () if dense_aligner is None else tuple(dense_aligner.parameters())
        ),
    )
    if train_global_scene_residual_only:
        assert global_scene_residual is not None
        expected_trainable_ids = {id(parameter) for parameter in global_scene_residual.parameters()}
        observed_trainable_ids = {id(parameter) for parameter in parameters}
        if observed_trainable_ids != expected_trainable_ids:
            raise RuntimeError(
                "Residual-only optimizer surface contains missing or unexpected parameters"
            )
        if [group.get("name") for group in optimizer.param_groups] != ["global_scene_residual"]:
            raise RuntimeError("Residual-only optimizer must contain exactly one named group")
    if train_signed_x_scene_residual_only:
        assert signed_x_scene_residual is not None
        expected_trainable_ids = {id(signed_x_scene_residual.output_projection.weight)}
        observed_trainable_ids = {id(parameter) for parameter in parameters}
        if observed_trainable_ids != expected_trainable_ids:
            raise RuntimeError(
                "Signed-X-only optimizer surface contains missing or unexpected parameters"
            )
        if [group.get("name") for group in optimizer.param_groups] != [
            "signed_x_output_projection"
        ]:
            raise RuntimeError(
                "Signed-X-only optimizer must contain exactly one named output group"
            )
    if train_lora_with_frozen_scene_residual_stack:
        assert lora_installation is not None
        expected_trainable_ids = [id(parameter) for parameter in lora_installation.parameters()]
        observed_trainable_ids = [id(parameter) for parameter in parameters]
        if not expected_trainable_ids or observed_trainable_ids != expected_trainable_ids:
            raise RuntimeError(
                "Frozen-residual-stack optimizer surface contains missing, reordered, or "
                "unexpected parameters"
            )
        if [group.get("name") for group in optimizer.param_groups] != ["language_lora"]:
            raise RuntimeError(
                "Frozen-residual-stack optimizer must contain exactly one language-LoRA group"
            )
    if train_dense_alignment_only:
        assert dense_aligner is not None
        expected_trainable_ids = [id(parameter) for parameter in dense_aligner.parameters()]
        observed_trainable_ids = [id(parameter) for parameter in parameters]
        if not expected_trainable_ids or observed_trainable_ids != expected_trainable_ids:
            raise RuntimeError(
                "Dense-alignment-only optimizer contains a missing, reordered, or unexpected "
                "parameter"
            )
        if [group.get("name") for group in optimizer.param_groups] != ["dense_alignment"]:
            raise RuntimeError("Dense-alignment-only optimizer must contain exactly one group")
    scene_checkpoint_modules = {
        "scene_model": scene_model,
        "composer": composer,
        "grounding": grounding,
    }
    checkpoint_modules = dict(scene_checkpoint_modules)
    if global_scene_residual is not None:
        checkpoint_modules["global_scene_residual"] = global_scene_residual
    if signed_x_scene_residual is not None:
        checkpoint_modules["signed_x_scene_residual"] = signed_x_scene_residual
    if dense_aligner is not None:
        checkpoint_modules["dense_aligner"] = dense_aligner
    if lora_installation is not None:
        checkpoint_modules.update(lora_installation.state_modules())
    batch_size = int(config["training"]["batch_size"])
    accumulation = gradient_accumulation
    epochs = args.epochs or int(config["training"]["epochs"])
    history: list[dict] = []
    best_monitor_loss = math.inf
    best_epoch: int | None = None
    best_pair_gate_passed = False
    epochs_without_improvement = 0
    started = time.perf_counter()
    global_step = 0
    optimizer_step = 0
    start_epoch = 1
    resume_value, initialize_value = resolve_checkpoint_sources(
        cli_resume=args.resume,
        cli_initialize_from=args.initialize_from,
        training_config=config["training"],
    )
    if initialize_legacy_lora_into_bank is not None and not (initialize_value or resume_value):
        raise ValueError("initialize_legacy_lora_into_bank requires initialize_from for a new run")
    if initialize_named_lora_freeze_transition and not (initialize_value or resume_value):
        raise ValueError(
            "initialize_named_lora_freeze_transition requires initialize_from for a new run"
        )
    if initialize_named_lora_freeze_and_extend_transition and not (
        initialize_value or resume_value
    ):
        raise ValueError(
            "initialize_named_lora_freeze_and_extend_transition requires initialize_from "
            "for a new run"
        )
    if initialize_named_lora_freeze_for_dense_alignment_transition and not (
        initialize_value or resume_value
    ):
        raise ValueError(
            "initialize_named_lora_freeze_for_dense_alignment_transition requires "
            "initialize_from for a new run"
        )
    if initialize_source_residual_into_frozen_base and not (initialize_value or resume_value):
        raise ValueError(
            "initialize_source_residual_into_frozen_base requires initialize_from for a new run"
        )
    if train_lora_with_frozen_scene_residual_stack and not (initialize_value or resume_value):
        raise ValueError(
            "Frozen-residual-stack LoRA training requires initialize_from for a new run"
        )
    if train_dense_alignment_only and not (initialize_value or resume_value):
        raise ValueError("Dense-alignment-only training requires initialize_from for a new run")
    initialization_provenance: dict | None = None
    if initialize_value:
        initialize_path = Path(initialize_value).expanduser()
        if not initialize_path.is_absolute():
            initialize_path = PROJECT_ROOT / initialize_path
        initialize_path = initialize_path.resolve()
        checkpoint_root = artifact_root(config, "checkpoints").resolve()
        if not initialize_path.is_relative_to(checkpoint_root):
            raise ValueError(
                f"initialize_from must be inside the configured checkpoint root: {checkpoint_root}"
            )
        initialization_artifact_hashes = verify_initialization_artifact_hashes(
            initialize_path,
            expected_adapter_sha256=initialize_expected_adapter_sha256,
            expected_metadata_sha256=initialize_expected_metadata_sha256,
        )
        initialize_preflight = json.loads(
            (initialize_path / "metadata.json").read_text(encoding="utf-8")
        )
        initialization_mismatches: dict[str, object] = {}
        expected_initialization = {
            "semantic_dim": semantic_dim,
            "language_hidden_dim": language.hidden_size,
            "language_model_id": config["language"]["model_id"],
            "language_revision": config["language"]["revision"],
            "scene_latents": int(config["scene_encoder"]["global_latents"]),
            "scene_model_dim": int(config["scene_encoder"]["model_dim"]),
            "scene_encoder_architecture_version": config["scene_encoder"].get(
                "architecture_version"
            ),
            "input_voxel_size_m": config["scene_encoder"].get("input_voxel_size_m"),
            "scene_prefix_after_bos": scene_prefix_after_bos,
            **token_mixing,
        }
        if "language_backend" in initialize_preflight:
            expected_initialization["language_backend"] = language.backend_name
        for key, runtime_value in expected_initialization.items():
            if initialize_preflight.get(key) != runtime_value:
                initialization_mismatches[key] = {
                    "checkpoint": initialize_preflight.get(key),
                    "runtime": runtime_value,
                }
        boundary_mismatch = scene_boundary_contract_mismatch(
            initialize_preflight,
            scene_boundary_mode,
            loaded_native_boundary_contract,
        )
        if boundary_mismatch is not None:
            initialization_mismatches["scene_boundary_mode"] = boundary_mismatch
        legacy_initialization_bank: InstalledLoRABank | None = None
        if initialize_legacy_lora_into_bank is not None:
            if lora_installation is None or lora_installation.settings.legacy_single_bank:
                initialization_mismatches["initialize_legacy_lora_into_bank"] = {
                    "checkpoint": initialize_preflight.get("lora"),
                    "runtime": initialize_legacy_lora_into_bank,
                }
            else:
                try:
                    legacy_initialization_bank = lora_installation.bank(
                        initialize_legacy_lora_into_bank
                    )
                except KeyError:
                    initialization_mismatches["initialize_legacy_lora_into_bank"] = {
                        "checkpoint": initialize_preflight.get("lora"),
                        "runtime": initialize_legacy_lora_into_bank,
                    }
                else:
                    legacy_mismatch = legacy_lora_bank_source_mismatch(
                        initialize_preflight, legacy_initialization_bank
                    )
                    if legacy_mismatch is not None:
                        initialization_mismatches["legacy_lora_bank"] = legacy_mismatch
        elif initialize_named_lora_freeze_transition:
            freeze_transition_mismatch = named_lora_freeze_transition_mismatch(
                initialize_preflight,
                lora_installation if isinstance(lora_installation, LoRABankCollection) else None,
            )
            if freeze_transition_mismatch is not None:
                initialization_mismatches["named_lora_freeze_transition"] = (
                    freeze_transition_mismatch
                )
        elif initialize_named_lora_freeze_for_dense_alignment_transition:
            dense_freeze_mismatch = named_lora_freeze_transition_mismatch(
                initialize_preflight,
                lora_installation if isinstance(lora_installation, LoRABankCollection) else None,
            )
            if dense_freeze_mismatch is not None:
                initialization_mismatches[
                    "named_lora_freeze_for_dense_alignment_transition"
                ] = dense_freeze_mismatch
            source_scene_hash = initialize_preflight.get("frozen_scene_state_sha256")
            if source_scene_hash != initialize_expected_scene_state_sha256:
                initialization_mismatches["frozen_scene_state_sha256"] = {
                    "checkpoint": source_scene_hash,
                    "runtime": initialize_expected_scene_state_sha256,
                }
            if initialize_preflight.get("global_scene_residual") != residual_settings.contract():
                initialization_mismatches["global_scene_residual"] = {
                    "checkpoint": initialize_preflight.get("global_scene_residual"),
                    "runtime": residual_settings.contract(),
                }
            if initialize_preflight.get("signed_x_scene_residual") != (
                signed_x_residual_settings.contract()
            ):
                initialization_mismatches["signed_x_scene_residual"] = {
                    "checkpoint": initialize_preflight.get("signed_x_scene_residual"),
                    "runtime": signed_x_residual_settings.contract(),
                }
            source_global_hash = initialize_preflight.get("global_scene_residual_state_sha256")
            if source_global_hash != initialize_expected_global_scene_residual_state_sha256:
                initialization_mismatches["global_scene_residual_state_sha256"] = {
                    "checkpoint": source_global_hash,
                    "runtime": initialize_expected_global_scene_residual_state_sha256,
                }
            source_signed_x_hash = initialize_preflight.get("signed_x_scene_residual_state_sha256")
            if source_signed_x_hash != initialize_expected_signed_x_scene_residual_state_sha256:
                initialization_mismatches["signed_x_scene_residual_state_sha256"] = {
                    "checkpoint": source_signed_x_hash,
                    "runtime": initialize_expected_signed_x_scene_residual_state_sha256,
                }
            source_dense_contract = initialize_preflight.get(
                "dense_alignment", {"schema_version": 1, "enabled": False}
            )
            if source_dense_contract != {"schema_version": 1, "enabled": False}:
                initialization_mismatches["source_dense_alignment"] = {
                    "checkpoint": source_dense_contract,
                    "runtime_required": {"schema_version": 1, "enabled": False},
                }
        elif train_lora_with_frozen_scene_residual_stack:
            transition_collection = (
                lora_installation if isinstance(lora_installation, LoRABankCollection) else None
            )
            extension_transition_mismatch = (
                named_lora_freeze_and_extend_transition_mismatch(
                    initialize_preflight,
                    transition_collection,
                )
                if initialize_named_lora_freeze_and_extend_transition
                else named_lora_extension_transition_mismatch(
                    initialize_preflight,
                    transition_collection,
                )
            )
            if extension_transition_mismatch is not None:
                transition_name = (
                    "named_lora_freeze_and_extend_transition"
                    if initialize_named_lora_freeze_and_extend_transition
                    else "named_lora_extension_transition"
                )
                initialization_mismatches[transition_name] = extension_transition_mismatch
            source_scene_hash = initialize_preflight.get("frozen_scene_state_sha256")
            if source_scene_hash != initialize_expected_scene_state_sha256:
                initialization_mismatches["frozen_scene_state_sha256"] = {
                    "checkpoint": source_scene_hash,
                    "runtime": initialize_expected_scene_state_sha256,
                }
            if initialize_preflight.get("global_scene_residual") != residual_settings.contract():
                initialization_mismatches["global_scene_residual"] = {
                    "checkpoint": initialize_preflight.get("global_scene_residual"),
                    "runtime": residual_settings.contract(),
                }
            if initialize_preflight.get("signed_x_scene_residual") != (
                signed_x_residual_settings.contract()
            ):
                initialization_mismatches["signed_x_scene_residual"] = {
                    "checkpoint": initialize_preflight.get("signed_x_scene_residual"),
                    "runtime": signed_x_residual_settings.contract(),
                }
            source_global_hash = initialize_preflight.get("global_scene_residual_state_sha256")
            if source_global_hash != initialize_expected_global_scene_residual_state_sha256:
                initialization_mismatches["global_scene_residual_state_sha256"] = {
                    "checkpoint": source_global_hash,
                    "runtime": initialize_expected_global_scene_residual_state_sha256,
                }
            source_signed_x_hash = initialize_preflight.get("signed_x_scene_residual_state_sha256")
            if source_signed_x_hash != initialize_expected_signed_x_scene_residual_state_sha256:
                initialization_mismatches["signed_x_scene_residual_state_sha256"] = {
                    "checkpoint": source_signed_x_hash,
                    "runtime": initialize_expected_signed_x_scene_residual_state_sha256,
                }
        elif initialize_source_residual_into_frozen_base:
            lora_mismatch = lora_checkpoint_contract_mismatch(
                initialize_preflight, configured_lora_checkpoint_contract
            )
            if lora_mismatch is not None:
                initialization_mismatches["lora"] = lora_mismatch
            if initialize_preflight.get("global_scene_residual") != residual_settings.contract():
                initialization_mismatches["global_scene_residual"] = {
                    "checkpoint": initialize_preflight.get("global_scene_residual"),
                    "runtime": residual_settings.contract(),
                }
            source_residual_hash = initialize_preflight.get("global_scene_residual_state_sha256")
            if source_residual_hash != initialize_expected_global_scene_residual_state_sha256:
                initialization_mismatches["global_scene_residual_state_sha256"] = {
                    "checkpoint": source_residual_hash,
                    "runtime": initialize_expected_global_scene_residual_state_sha256,
                }
        else:
            lora_mismatch = lora_checkpoint_contract_mismatch(
                initialize_preflight, configured_lora_checkpoint_contract
            )
            if lora_mismatch is not None:
                initialization_mismatches["lora"] = lora_mismatch
        if initialization_mismatches:
            raise ValueError(
                f"Initialization checkpoint architecture mismatch: {initialization_mismatches}"
            )
        initialization_modules = checkpoint_modules
        if legacy_initialization_bank is not None:
            initialization_modules = staged_legacy_lora_checkpoint_modules(
                scene_checkpoint_modules, legacy_initialization_bank
            )
        elif initialize_named_lora_freeze_transition:
            initialization_modules = {
                name: module
                for name, module in checkpoint_modules.items()
                if name != "global_scene_residual"
            }
        elif initialize_named_lora_freeze_for_dense_alignment_transition:
            initialization_modules = dense_alignment_source_checkpoint_modules(
                checkpoint_modules
            )
        elif train_lora_with_frozen_scene_residual_stack:
            if not isinstance(lora_installation, LoRABankCollection):
                raise TypeError("Named LoRA extension transition requires named banks")
            initialization_modules = named_lora_extension_checkpoint_modules(
                checkpoint_modules,
                lora_installation,
            )
        elif initialize_source_residual_into_frozen_base:
            initialization_modules = {
                name: module
                for name, module in checkpoint_modules.items()
                if name != "signed_x_scene_residual"
            }
        loaded_initialization = load_adapter_checkpoint(
            initialize_path,
            initialization_modules,
            device=str(language.device),
        )
        if loaded_initialization != initialize_preflight:
            raise RuntimeError("Initialization checkpoint metadata changed while loading")
        if loaded_native_boundary_embeddings is not None:
            composer.validate_native_boundary_embeddings(loaded_native_boundary_embeddings)
        if legacy_initialization_bank is not None:
            validate_lora_checkpoint_state(
                loaded_initialization, legacy_initialization_bank.installation
            )
            assert lora_installation is not None
            assert_zero_output_lora_banks(
                lora_installation, exclude=(legacy_initialization_bank.settings.name,)
            )
        elif initialize_named_lora_freeze_transition:
            if not isinstance(lora_installation, LoRABankCollection):
                raise TypeError("Named LoRA freeze transition did not install named banks")
            validate_named_lora_freeze_transition_state(loaded_initialization, lora_installation)
            if global_scene_residual is None:
                raise RuntimeError("Named LoRA freeze transition lost the residual module")
            validate_global_scene_residual_state(
                global_scene_residual,
                expected_parameter_count=declared_residual_parameter_count,
                context="source checkpoint initialization",
            )
            residual_sha_after_source_load = module_collection_state_sha256(
                {"global_scene_residual": global_scene_residual}
            )
            if residual_sha_after_source_load != observed_initial_residual_sha256:
                raise RuntimeError("Source checkpoint mutated the fresh residual state")
        elif initialize_named_lora_freeze_for_dense_alignment_transition:
            if not isinstance(lora_installation, LoRABankCollection):
                raise TypeError("Dense-alignment source transition requires named LoRA banks")
            if (
                dense_aligner is None
                or global_scene_residual is None
                or signed_x_scene_residual is None
            ):
                raise RuntimeError("Dense-alignment source transition lost a required module")
            validate_named_lora_freeze_transition_state(
                loaded_initialization,
                lora_installation,
            )
            loaded_scene_hash = module_collection_state_sha256(scene_checkpoint_modules)
            if loaded_scene_hash != initialize_expected_scene_state_sha256:
                raise ValueError(
                    "Loaded dense-alignment source scene stack differs from its exact pin: "
                    f"expected={initialize_expected_scene_state_sha256} "
                    f"observed={loaded_scene_hash}"
                )
            validate_global_scene_residual_state(
                global_scene_residual,
                expected_parameter_count=declared_residual_parameter_count,
                context="dense-alignment source checkpoint initialization",
            )
            loaded_global_hash = module_collection_state_sha256(
                {"global_scene_residual": global_scene_residual}
            )
            if loaded_global_hash != initialize_expected_global_scene_residual_state_sha256:
                raise ValueError(
                    "Loaded dense-alignment source global residual differs from its exact pin: "
                    f"expected={initialize_expected_global_scene_residual_state_sha256} "
                    f"observed={loaded_global_hash}"
                )
            validate_signed_x_scene_residual_state(
                signed_x_scene_residual,
                expected_parameter_count=declared_signed_x_parameter_count,
                context="dense-alignment source checkpoint initialization",
            )
            loaded_signed_x_hash = module_collection_state_sha256(
                {"signed_x_scene_residual": signed_x_scene_residual}
            )
            if loaded_signed_x_hash != initialize_expected_signed_x_scene_residual_state_sha256:
                raise ValueError(
                    "Loaded dense-alignment source signed-X residual differs from its exact pin: "
                    f"expected={initialize_expected_signed_x_scene_residual_state_sha256} "
                    f"observed={loaded_signed_x_hash}"
                )
            validate_dense_alignment_state(
                dense_aligner,
                expected_parameter_count=declared_dense_parameter_count,
                context="fresh dense alignment after source checkpoint initialization",
            )
            dense_hash_after_source_load = dense_aligner.state_sha256()
            if dense_hash_after_source_load != observed_initial_dense_alignment_sha256:
                raise RuntimeError("Source checkpoint mutated the fresh dense-alignment state")
            if not all(parameter.requires_grad for parameter in dense_aligner.parameters()):
                raise RuntimeError("Fresh dense-alignment parameters are unexpectedly frozen")
        elif train_lora_with_frozen_scene_residual_stack:
            if not isinstance(lora_installation, LoRABankCollection):
                raise TypeError("Named LoRA extension transition requires named banks")
            if global_scene_residual is None or signed_x_scene_residual is None:
                raise RuntimeError("Named LoRA extension transition lost a residual module")
            validate_named_lora_extension_transition_state(
                loaded_initialization,
                lora_installation,
            )
            loaded_scene_hash = module_collection_state_sha256(scene_checkpoint_modules)
            if loaded_scene_hash != initialize_expected_scene_state_sha256:
                raise ValueError(
                    "Loaded source scene stack differs from its exact pin: "
                    f"expected={initialize_expected_scene_state_sha256} "
                    f"observed={loaded_scene_hash}"
                )
            validate_global_scene_residual_state(
                global_scene_residual,
                expected_parameter_count=declared_residual_parameter_count,
                context="frozen-stack source checkpoint initialization",
            )
            loaded_global_hash = module_collection_state_sha256(
                {"global_scene_residual": global_scene_residual}
            )
            if loaded_global_hash != initialize_expected_global_scene_residual_state_sha256:
                raise ValueError(
                    "Loaded source global residual differs from its exact pin: "
                    f"expected={initialize_expected_global_scene_residual_state_sha256} "
                    f"observed={loaded_global_hash}"
                )
            validate_signed_x_scene_residual_state(
                signed_x_scene_residual,
                expected_parameter_count=declared_signed_x_parameter_count,
                context="frozen-stack source checkpoint initialization",
            )
            loaded_signed_x_hash = module_collection_state_sha256(
                {"signed_x_scene_residual": signed_x_scene_residual}
            )
            if loaded_signed_x_hash != initialize_expected_signed_x_scene_residual_state_sha256:
                raise ValueError(
                    "Loaded source signed-X residual differs from its exact pin: "
                    f"expected={initialize_expected_signed_x_scene_residual_state_sha256} "
                    f"observed={loaded_signed_x_hash}"
                )
        elif initialize_source_residual_into_frozen_base:
            if global_scene_residual is None or signed_x_scene_residual is None:
                raise RuntimeError("Source-residual transition lost a residual module")
            if lora_installation is not None:
                validate_lora_banks_checkpoint_state(
                    loaded_initialization,
                    lora_installation,
                )
            validate_global_scene_residual_state(
                global_scene_residual,
                expected_parameter_count=declared_residual_parameter_count,
                context="frozen source-residual checkpoint initialization",
            )
            loaded_source_residual_hash = module_collection_state_sha256(
                {"global_scene_residual": global_scene_residual}
            )
            if (
                loaded_source_residual_hash
                != initialize_expected_global_scene_residual_state_sha256
            ):
                raise ValueError(
                    "Loaded source global residual state differs from its exact pin: "
                    f"expected={initialize_expected_global_scene_residual_state_sha256} "
                    f"observed={loaded_source_residual_hash}"
                )
            validate_signed_x_scene_residual_state(
                signed_x_scene_residual,
                expected_parameter_count=declared_signed_x_parameter_count,
                context="fresh signed-X source transition",
            )
            signed_x_hash_after_source_load = module_collection_state_sha256(
                {"signed_x_scene_residual": signed_x_scene_residual}
            )
            if signed_x_hash_after_source_load != observed_initial_signed_x_sha256:
                raise RuntimeError("Source checkpoint mutated the fresh signed-X state")
        elif lora_installation is not None:
            validate_lora_banks_checkpoint_state(loaded_initialization, lora_installation)
        initialization_provenance = {
            "schema_version": (
                7
                if initialize_named_lora_freeze_for_dense_alignment_transition
                else 6
                if initialize_named_lora_freeze_and_extend_transition
                else 5
                if train_lora_with_frozen_scene_residual_stack
                else 4
                if initialize_source_residual_into_frozen_base
                else 3
                if initialize_named_lora_freeze_transition
                else (2 if legacy_initialization_bank is not None else 1)
            ),
            "mode": (
                "frozen_named_lora_scene_stack_plus_zero_output_dense_alignment"
                if initialize_named_lora_freeze_for_dense_alignment_transition
                else "existing_named_lora_banks_frozen_plus_zero_output_named_lora_extension"
                if initialize_named_lora_freeze_and_extend_transition
                else "frozen_scene_residual_stack_plus_zero_output_named_lora_extension"
                if train_lora_with_frozen_scene_residual_stack
                else "frozen_v18_residual_base_plus_zero_output_signed_x_residual"
                if initialize_source_residual_into_frozen_base
                else "named_lora_banks_frozen_plus_zero_output_scene_residual"
                if initialize_named_lora_freeze_transition
                else (
                    "legacy_lora_into_frozen_named_bank"
                    if legacy_initialization_bank is not None
                    else "weights_only_new_curriculum"
                )
            ),
            "checkpoint": str(initialize_path.relative_to(PROJECT_ROOT)),
            **initialization_artifact_hashes,
            "expected_adapter_sha256": initialize_expected_adapter_sha256,
            "expected_metadata_sha256": initialize_expected_metadata_sha256,
            "checkpoint_epoch": initialize_preflight.get("epoch"),
            "checkpoint_output_namespace": initialize_preflight.get("output_namespace"),
            "checkpoint_config_hash": initialize_preflight.get("config_hash"),
            "checkpoint_source_provenance": initialize_preflight.get("source_provenance"),
            "initialize_named_lora_freeze_for_dense_alignment_transition": (
                initialize_named_lora_freeze_for_dense_alignment_transition
            ),
            "optimizer_state_loaded": False,
            "history_loaded": False,
        }
        if legacy_initialization_bank is not None:
            initialization_provenance.update(
                {
                    "legacy_source_module": "lora",
                    "target_bank": legacy_initialization_bank.settings.name,
                    "target_bank_state_sha256": (
                        legacy_initialization_bank.installation.state_sha256()
                    ),
                    "new_trainable_banks_zero_output": True,
                }
            )
        if initialize_named_lora_freeze_transition:
            assert isinstance(lora_installation, LoRABankCollection)
            initialization_provenance.update(
                {
                    "source_lora_bank_state_sha256": lora_installation.state_sha256(),
                    "all_source_lora_banks_frozen": True,
                    "global_scene_residual_initial_state_sha256": (
                        observed_initial_residual_sha256
                    ),
                    "global_scene_residual_zero_output": True,
                }
            )
        if train_lora_with_frozen_scene_residual_stack:
            assert isinstance(lora_installation, LoRABankCollection)
            assert global_scene_residual is not None
            assert signed_x_scene_residual is not None
            initialization_provenance.update(
                {
                    "source_lora_bank_state_sha256": {
                        bank.settings.name: bank.installation.state_sha256()
                        for bank in lora_installation.banks
                        if not bank.settings.trainable
                    },
                    "source_scene_state_sha256": module_collection_state_sha256(
                        scene_checkpoint_modules
                    ),
                    "expected_source_scene_state_sha256": (initialize_expected_scene_state_sha256),
                    "source_global_scene_residual_state_sha256": (
                        module_collection_state_sha256(
                            {"global_scene_residual": global_scene_residual}
                        )
                    ),
                    "expected_source_global_scene_residual_state_sha256": (
                        initialize_expected_global_scene_residual_state_sha256
                    ),
                    "source_signed_x_scene_residual_state_sha256": (
                        module_collection_state_sha256(
                            {"signed_x_scene_residual": signed_x_scene_residual}
                        )
                    ),
                    "expected_source_signed_x_scene_residual_state_sha256": (
                        initialize_expected_signed_x_scene_residual_state_sha256
                    ),
                    "all_source_scene_residuals_frozen": True,
                    "new_trainable_lora_banks_zero_output": True,
                }
            )
        if initialize_named_lora_freeze_for_dense_alignment_transition:
            assert isinstance(lora_installation, LoRABankCollection)
            assert global_scene_residual is not None
            assert signed_x_scene_residual is not None
            assert dense_aligner is not None
            initialization_provenance.update(
                {
                    "source_lora_bank_state_sha256": lora_installation.state_sha256(),
                    "source_scene_state_sha256": module_collection_state_sha256(
                        scene_checkpoint_modules
                    ),
                    "expected_source_scene_state_sha256": (
                        initialize_expected_scene_state_sha256
                    ),
                    "source_global_scene_residual_state_sha256": (
                        module_collection_state_sha256(
                            {"global_scene_residual": global_scene_residual}
                        )
                    ),
                    "expected_source_global_scene_residual_state_sha256": (
                        initialize_expected_global_scene_residual_state_sha256
                    ),
                    "source_signed_x_scene_residual_state_sha256": (
                        module_collection_state_sha256(
                            {"signed_x_scene_residual": signed_x_scene_residual}
                        )
                    ),
                    "expected_source_signed_x_scene_residual_state_sha256": (
                        initialize_expected_signed_x_scene_residual_state_sha256
                    ),
                    "all_source_modules_frozen": True,
                    "dense_alignment_initial_state_sha256": (
                        observed_initial_dense_alignment_sha256
                    ),
                    "expected_dense_alignment_initial_state_sha256": (
                        initialize_expected_dense_alignment_state_sha256
                    ),
                    "dense_alignment_zero_output": True,
                    "source_checkpoint_loaded_dense_alignment": False,
                }
            )
        if initialize_source_residual_into_frozen_base:
            assert global_scene_residual is not None
            assert signed_x_scene_residual is not None
            initialization_provenance.update(
                {
                    "source_global_scene_residual_state_sha256": (
                        module_collection_state_sha256(
                            {"global_scene_residual": global_scene_residual}
                        )
                    ),
                    "expected_source_global_scene_residual_state_sha256": (
                        initialize_expected_global_scene_residual_state_sha256
                    ),
                    "global_scene_residual_frozen": True,
                    "signed_x_scene_residual_initial_state_sha256": (
                        observed_initial_signed_x_sha256
                    ),
                    "signed_x_scene_residual_zero_output": True,
                }
            )
        print(
            json.dumps({"phase": "adapter_initialized", **initialization_provenance}),
            flush=True,
        )
    resume_metadata: dict | None = None
    if resume_value:
        resume_path = Path(resume_value)
        if not resume_path.is_absolute():
            resume_path = PROJECT_ROOT / resume_path
        resume_preflight = json.loads((resume_path / "metadata.json").read_text(encoding="utf-8"))
        if resume_preflight.get("v18_stage_execution") != v18_stage_execution:
            raise ValueError(
                "Resume checkpoint V18 stage-execution contract mismatch: "
                f"checkpoint={resume_preflight.get('v18_stage_execution')} "
                f"runtime={v18_stage_execution}"
            )
        boundary_preflight_mismatch = scene_boundary_contract_mismatch(
            resume_preflight,
            scene_boundary_mode,
            loaded_native_boundary_contract,
        )
        if boundary_preflight_mismatch is not None:
            raise ValueError(
                "Resume checkpoint architecture mismatch: "
                f"{{'scene_boundary_mode': {boundary_preflight_mismatch}}}"
            )
        lora_preflight_mismatch = lora_checkpoint_contract_mismatch(
            resume_preflight, configured_lora_checkpoint_contract
        )
        if lora_preflight_mismatch is not None:
            raise ValueError(
                f"Resume checkpoint architecture mismatch: {{'lora': {lora_preflight_mismatch}}}"
            )
        if resume_preflight.get("global_scene_residual") != residual_settings.contract():
            raise ValueError(
                "Resume checkpoint global-scene-residual contract mismatch: "
                f"checkpoint={resume_preflight.get('global_scene_residual')} "
                f"runtime={residual_settings.contract()}"
            )
        if global_scene_residual is not None:
            expected_initial_hash = residual_settings.expected_initial_state_sha256
            if expected_initial_hash is None:  # guarded by enabled settings validation
                raise RuntimeError("Enabled residual lost its expected initial-state hash")
            residual_metadata_mismatch = global_scene_residual_resume_metadata_mismatch(
                resume_preflight,
                global_scene_residual,
                expected_initial_state_sha256=expected_initial_hash,
            )
            if residual_metadata_mismatch is not None:
                raise ValueError(
                    "Resume checkpoint global-scene-residual provenance mismatch: "
                    f"{residual_metadata_mismatch}"
                )
        if resume_preflight.get("signed_x_scene_residual") != (
            signed_x_residual_settings.contract()
        ):
            raise ValueError(
                "Resume checkpoint signed-X residual contract mismatch: "
                f"checkpoint={resume_preflight.get('signed_x_scene_residual')} "
                f"runtime={signed_x_residual_settings.contract()}"
            )
        if signed_x_scene_residual is not None:
            expected_initial_signed_x_hash = (
                signed_x_residual_settings.expected_initial_state_sha256
            )
            if expected_initial_signed_x_hash is None:
                raise RuntimeError("Enabled signed-X residual lost its initial-state hash")
            signed_x_metadata_mismatch = signed_x_scene_residual_resume_metadata_mismatch(
                resume_preflight,
                signed_x_scene_residual,
                expected_initial_state_sha256=expected_initial_signed_x_hash,
            )
            if signed_x_metadata_mismatch is not None:
                raise ValueError(
                    "Resume checkpoint signed-X residual provenance mismatch: "
                    f"{signed_x_metadata_mismatch}"
                )
        if resume_preflight.get("dense_alignment") != dense_settings.contract():
            raise ValueError(
                "Resume checkpoint dense-alignment contract mismatch: "
                f"checkpoint={resume_preflight.get('dense_alignment')} "
                f"runtime={dense_settings.contract()}"
            )
        if dense_aligner is not None:
            expected_initial_dense_hash = dense_settings.expected_initial_state_sha256
            if expected_initial_dense_hash is None:
                raise RuntimeError("Enabled dense alignment lost its initial-state hash")
            dense_metadata_mismatch = dense_alignment_resume_metadata_mismatch(
                resume_preflight,
                dense_aligner,
                expected_initial_state_sha256=expected_initial_dense_hash,
            )
            if dense_metadata_mismatch is not None:
                raise ValueError(
                    "Resume checkpoint dense-alignment provenance mismatch: "
                    f"{dense_metadata_mismatch}"
                )
        if configured_lora.enabled:
            provenance_mismatch = source_provenance_resume_contract_mismatch(
                resume_preflight, source_provenance
            )
            if provenance_mismatch is not None:
                raise ValueError(
                    f"Resume checkpoint source provenance mismatch: {provenance_mismatch}"
                )
        resume_metadata = load_adapter_checkpoint(
            resume_path,
            checkpoint_modules,
            device=str(language.device),
        )
        if resume_metadata != resume_preflight:
            raise RuntimeError("Resume checkpoint metadata changed while loading")
        saved_initialization_provenance = resume_metadata.get("initialization_provenance")
        initialization_provenance = (
            dict(saved_initialization_provenance)
            if isinstance(saved_initialization_provenance, Mapping)
            else None
        )
        expected_initialization_hashes = {
            "adapter_sha256": initialize_expected_adapter_sha256,
            "metadata_sha256": initialize_expected_metadata_sha256,
        }
        initialization_hash_mismatches = {
            key: {
                "checkpoint": (
                    None
                    if initialization_provenance is None
                    else initialization_provenance.get(key)
                ),
                "runtime": value,
            }
            for key, value in expected_initialization_hashes.items()
            if value is not None
            and (initialization_provenance is None or initialization_provenance.get(key) != value)
        }
        if initialize_expected_scene_state_sha256 is not None and (
            initialization_provenance is None
            or initialization_provenance.get("expected_source_scene_state_sha256")
            != initialize_expected_scene_state_sha256
        ):
            initialization_hash_mismatches["expected_source_scene_state_sha256"] = {
                "checkpoint": (
                    None
                    if initialization_provenance is None
                    else initialization_provenance.get("expected_source_scene_state_sha256")
                ),
                "runtime": initialize_expected_scene_state_sha256,
            }
        if initialize_expected_global_scene_residual_state_sha256 is not None and (
            initialization_provenance is None
            or initialization_provenance.get("expected_source_global_scene_residual_state_sha256")
            != initialize_expected_global_scene_residual_state_sha256
        ):
            initialization_hash_mismatches["expected_source_global_scene_residual_state_sha256"] = {
                "checkpoint": (
                    None
                    if initialization_provenance is None
                    else initialization_provenance.get(
                        "expected_source_global_scene_residual_state_sha256"
                    )
                ),
                "runtime": initialize_expected_global_scene_residual_state_sha256,
            }
        if initialize_expected_signed_x_scene_residual_state_sha256 is not None and (
            initialization_provenance is None
            or initialization_provenance.get("expected_source_signed_x_scene_residual_state_sha256")
            != initialize_expected_signed_x_scene_residual_state_sha256
        ):
            initialization_hash_mismatches[
                "expected_source_signed_x_scene_residual_state_sha256"
            ] = {
                "checkpoint": (
                    None
                    if initialization_provenance is None
                    else initialization_provenance.get(
                        "expected_source_signed_x_scene_residual_state_sha256"
                    )
                ),
                "runtime": initialize_expected_signed_x_scene_residual_state_sha256,
            }
        if initialize_expected_dense_alignment_state_sha256 is not None and (
            initialization_provenance is None
            or initialization_provenance.get(
                "expected_dense_alignment_initial_state_sha256"
            )
            != initialize_expected_dense_alignment_state_sha256
        ):
            initialization_hash_mismatches[
                "expected_dense_alignment_initial_state_sha256"
            ] = {
                "checkpoint": (
                    None
                    if initialization_provenance is None
                    else initialization_provenance.get(
                        "expected_dense_alignment_initial_state_sha256"
                    )
                ),
                "runtime": initialize_expected_dense_alignment_state_sha256,
            }
        if initialization_hash_mismatches:
            raise ValueError(
                "Resume checkpoint initialization artifact mismatch: "
                f"{initialization_hash_mismatches}"
            )
        if loaded_native_boundary_embeddings is not None:
            # The checkpoint load overwrites persistent BOI/EOI buffers. Verify
            # them immediately against the already loaded, pinned Gemma model
            # before optimizer restore, scene warmup, or any composed forward.
            composer.validate_native_boundary_embeddings(loaded_native_boundary_embeddings)
        if lora_installation is not None:
            validate_lora_banks_checkpoint_state(resume_metadata, lora_installation)
        if global_scene_residual is not None:
            validate_global_scene_residual_state(
                global_scene_residual,
                expected_parameter_count=declared_residual_parameter_count,
                context="resume checkpoint load",
            )
            observed_resumed_residual_hash = module_collection_state_sha256(
                {"global_scene_residual": global_scene_residual}
            )
            if observed_resumed_residual_hash != resume_metadata.get(
                "global_scene_residual_state_sha256"
            ):
                raise ValueError(
                    "Resumed global scene residual state mismatch or tamper detected: "
                    f"checkpoint={resume_metadata.get('global_scene_residual_state_sha256')} "
                    f"runtime={observed_resumed_residual_hash}"
                )
        if signed_x_scene_residual is not None:
            validate_signed_x_scene_residual_state(
                signed_x_scene_residual,
                expected_parameter_count=declared_signed_x_parameter_count,
                context="resume checkpoint load",
            )
            observed_resumed_signed_x_hash = module_collection_state_sha256(
                {"signed_x_scene_residual": signed_x_scene_residual}
            )
            if observed_resumed_signed_x_hash != resume_metadata.get(
                "signed_x_scene_residual_state_sha256"
            ):
                raise ValueError(
                    "Resumed signed-X residual state mismatch or tamper detected: "
                    f"checkpoint={resume_metadata.get('signed_x_scene_residual_state_sha256')} "
                    f"runtime={observed_resumed_signed_x_hash}"
                )
        if dense_aligner is not None:
            validate_dense_alignment_state(
                dense_aligner,
                expected_parameter_count=declared_dense_parameter_count,
                context="resume checkpoint load",
            )
            observed_resumed_dense_hash = dense_aligner.state_sha256()
            if observed_resumed_dense_hash != resume_metadata.get(
                "dense_alignment_state_sha256"
            ):
                raise ValueError(
                    "Resumed dense-alignment state mismatch or tamper detected: "
                    f"checkpoint={resume_metadata.get('dense_alignment_state_sha256')} "
                    f"runtime={observed_resumed_dense_hash}"
                )
        saved_freeze_scene_adapter = resume_metadata.get("freeze_scene_adapter", False)
        if saved_freeze_scene_adapter != freeze_scene_adapter:
            raise ValueError(
                "Resume checkpoint freeze_scene_adapter mismatch: "
                f"checkpoint={saved_freeze_scene_adapter} runtime={freeze_scene_adapter}"
            )
        if resume_metadata.get("train_global_scene_residual_only", False) != (
            train_global_scene_residual_only
        ):
            raise ValueError("Resume checkpoint residual-only training mode mismatch")
        if resume_metadata.get("train_signed_x_scene_residual_only", False) != (
            train_signed_x_scene_residual_only
        ):
            raise ValueError("Resume checkpoint signed-X-only training mode mismatch")
        if resume_metadata.get("train_lora_with_frozen_scene_residual_stack", False) != (
            train_lora_with_frozen_scene_residual_stack
        ):
            raise ValueError("Resume checkpoint frozen-residual-stack LoRA mode mismatch")
        if resume_metadata.get("train_dense_alignment_only", False) != (
            train_dense_alignment_only
        ):
            raise ValueError("Resume checkpoint dense-alignment-only training mode mismatch")
        expected = {
            "semantic_dim": semantic_dim,
            "language_hidden_dim": language.hidden_size,
            "language_model_id": config["language"]["model_id"],
            "language_revision": config["language"]["revision"],
            "scene_latents": int(config["scene_encoder"]["global_latents"]),
            "scene_model_dim": int(config["scene_encoder"]["model_dim"]),
        }
        resume_selection_contract = {
            "train_scene_ids": sorted(by_scene),
            "counterfactual_pair_unit_count": len(pair_units),
            "training_counterfactual_pair_membership_sha256": pair_membership_sha256,
            "counterfactual_pair_unit_selection_sha256": pair_unit_selection_sha256,
        }
        for key, value in resume_selection_contract.items():
            if pair_objectives.configured or key in resume_metadata:
                expected[key] = value
        if "language_backend" in resume_metadata:
            expected["language_backend"] = language.backend_name
        saved_language_checkpointing = resume_metadata.get(
            "language_decoder_gradient_checkpointing", False
        )
        architecture_version = config["scene_encoder"].get("architecture_version")
        if architecture_version is not None:
            expected["scene_encoder_architecture_version"] = str(architecture_version)
        mixing_defaults = {
            "language_aligned_tail_dim": 0,
            "native_aligned_coverage_scale": 0.0,
            "learned_scene_token_scale": 1.0,
            "learned_scene_token_rms_target": None,
        }
        metadata_has_mixing = any(key in resume_metadata for key in token_mixing)
        mixing_contract_active = metadata_has_mixing or token_mixing != mixing_defaults
        if mixing_contract_active:
            expected.update(token_mixing)
        mismatches = {
            key: {"checkpoint": resume_metadata.get(key), "runtime": value}
            for key, value in expected.items()
            if resume_metadata.get(key) != value
        }
        if (
            not isinstance(saved_language_checkpointing, bool)
            or saved_language_checkpointing != language_decoder_gradient_checkpointing
        ):
            mismatches["language_decoder_gradient_checkpointing"] = {
                "checkpoint": saved_language_checkpointing,
                "runtime": language_decoder_gradient_checkpointing,
            }
        accumulation_mismatch = gradient_accumulation_resume_contract_mismatch(
            resume_metadata,
            gradient_accumulation,
        )
        if accumulation_mismatch is not None:
            mismatches["gradient_accumulation"] = accumulation_mismatch
        prefix_layout_mismatch = scene_prefix_after_bos_contract_mismatch(
            resume_metadata,
            scene_prefix_after_bos,
        )
        if prefix_layout_mismatch is not None:
            mismatches["scene_prefix_after_bos"] = prefix_layout_mismatch
        boundary_mismatch = scene_boundary_contract_mismatch(
            resume_metadata,
            scene_boundary_mode,
            loaded_native_boundary_contract,
        )
        if boundary_mismatch is not None:
            mismatches["scene_boundary_mode"] = boundary_mismatch
        if mixing_contract_active:
            for key, value in token_mixing.items():
                if key not in resume_metadata:
                    mismatches[key] = {
                        "checkpoint": "<missing>",
                        "runtime": value,
                    }
        saved_anti_collapse = resume_metadata.get("anti_collapse_objectives")
        if saved_anti_collapse is not None and saved_anti_collapse != anti_collapse:
            mismatches["anti_collapse_objectives"] = {
                "checkpoint": saved_anti_collapse,
                "runtime": anti_collapse,
            }
        if saved_anti_collapse is None and (
            anti_collapse["latent_diversity_weight"] > 0
            or anti_collapse["paired_scene_separation_weight"] > 0
        ):
            mismatches["anti_collapse_objectives"] = {
                "checkpoint": None,
                "runtime": anti_collapse,
            }
        spatial_answer_mismatch = spatial_answer_resume_contract_mismatch(
            resume_metadata, spatial_answer_contrastive
        )
        if spatial_answer_mismatch is not None:
            mismatches["spatial_answer_contrastive"] = spatial_answer_mismatch
        spatial_warmup_mismatch = spatial_answer_warmup_resume_contract_mismatch(
            resume_metadata,
            spatial_answer_warmup,
            spatial_answer_warmup_targets,
        )
        if spatial_warmup_mismatch is not None:
            mismatches["spatial_answer_warmup"] = spatial_warmup_mismatch
        spatial_relation_mismatch = spatial_relation_resume_contract_mismatch(
            resume_metadata, spatial_relation_contrastive
        )
        if spatial_relation_mismatch is not None:
            mismatches["spatial_relation_contrastive"] = spatial_relation_mismatch
        spatial_relation_warmup_mismatch = spatial_relation_warmup_resume_contract_mismatch(
            resume_metadata,
            spatial_relation_warmup,
            spatial_relation_warmup_targets,
        )
        if spatial_relation_warmup_mismatch is not None:
            mismatches["spatial_relation_warmup"] = spatial_relation_warmup_mismatch
        expected_pair_curriculum = {
            "enabled": pair_curriculum.enabled,
            "pair_only": pair_curriculum.pair_only,
            "pair_only_scene_ids": list(pair_curriculum.pair_only_scene_ids),
            "max_units_per_pair": pair_curriculum.max_units_per_pair,
            "ranking_weight": pair_curriculum.ranking_weight,
            "ranking_margin": pair_curriculum.ranking_margin,
            "ranking_mode": pair_curriculum.ranking_mode,
            "full_vocab_ranking_weight": pair_curriculum.full_vocab_ranking_weight,
            "full_vocab_ranking_margin": pair_curriculum.full_vocab_ranking_margin,
            "batch_fraction": pair_curriculum.batch_fraction,
            "units_per_batch": pair_curriculum.units_per_batch,
            "steps_per_epoch": pair_curriculum.steps_per_epoch,
            "gate_enabled": pair_curriculum.gate_enabled,
            "objective_policy": resolved_pair_objective_contract,
            "objective_policy_coverage": pair_objective_coverage,
        }
        saved_pair_curriculum = resume_metadata.get("pair_curriculum")
        if isinstance(saved_pair_curriculum, dict) and "ranking_mode" not in saved_pair_curriculum:
            # Checkpoints written before ranking modes existed used the NLL
            # objective exclusively. Treat the missing field as that legacy
            # default without allowing it to resume a candidate-logit run.
            saved_pair_curriculum = {**saved_pair_curriculum, "ranking_mode": "nll"}
        if (
            isinstance(saved_pair_curriculum, dict)
            and "max_units_per_pair" not in saved_pair_curriculum
        ):
            saved_pair_curriculum = {**saved_pair_curriculum, "max_units_per_pair": None}
        if isinstance(saved_pair_curriculum, dict):
            saved_pair_curriculum = {
                "full_vocab_ranking_weight": 0.0,
                "full_vocab_ranking_margin": 0.0,
                **saved_pair_curriculum,
            }
            if "objective_policy" not in saved_pair_curriculum and not pair_objectives.configured:
                saved_pair_curriculum = {
                    **saved_pair_curriculum,
                    "objective_policy": resolved_pair_objective_contract,
                    "objective_policy_coverage": pair_objective_coverage,
                }
        if saved_pair_curriculum is not None and (
            saved_pair_curriculum != expected_pair_curriculum
        ):
            mismatches["pair_curriculum"] = {
                "checkpoint": saved_pair_curriculum,
                "runtime": expected_pair_curriculum,
            }
        if saved_pair_curriculum is None and pair_curriculum.enabled:
            mismatches["pair_curriculum"] = {
                "checkpoint": None,
                "runtime": expected_pair_curriculum,
            }
        if mismatches:
            raise ValueError(f"Resume checkpoint architecture mismatch: {mismatches}")
        expected_optimizer_groups = [
            {key: value for key, value in group.items() if key != "params"}
            for group in optimizer.param_groups
        ]
        expected_optimizer_parameter_ids = [
            id(parameter) for group in optimizer.param_groups for parameter in group["params"]
        ]
        if train_lora_with_frozen_scene_residual_stack or train_dense_alignment_only:
            raw_optimizer_state = torch.load(
                resume_path / "optimizer.pt",
                map_location="cpu",
                weights_only=True,
            )
            if not isinstance(raw_optimizer_state, Mapping) or set(raw_optimizer_state) != {
                "state",
                "param_groups",
            }:
                raise ValueError("Resume optimizer root violates the exact AdamW contract")
            raw_groups = raw_optimizer_state["param_groups"]
            raw_states = raw_optimizer_state["state"]
            if not isinstance(raw_groups, list) or len(raw_groups) != len(
                expected_optimizer_groups
            ):
                raise ValueError("Resume optimizer group count violates the exact contract")
            flat_raw_parameter_ids: list[int] = []
            for index, (raw_group, expected_group) in enumerate(
                zip(raw_groups, expected_optimizer_groups, strict=True)
            ):
                if not isinstance(raw_group, Mapping):
                    raise TypeError(f"Resume optimizer group {index} is not a mapping")
                raw_options = {key: value for key, value in raw_group.items() if key != "params"}
                if raw_options != expected_group:
                    raise ValueError(
                        f"Resume optimizer group {index} hyperparameters violate the exact contract"
                    )
                raw_ids = raw_group.get("params")
                if not isinstance(raw_ids, list) or any(
                    isinstance(item, bool) or not isinstance(item, int) for item in raw_ids
                ):
                    raise ValueError(f"Resume optimizer group {index} parameter IDs are invalid")
                flat_raw_parameter_ids.extend(raw_ids)
            expected_serialized_ids = list(range(len(parameters)))
            if flat_raw_parameter_ids != expected_serialized_ids:
                raise ValueError(
                    "Resume optimizer serialized parameter order violates the exact live order"
                )
            if not isinstance(raw_states, Mapping) or set(raw_states) != set(
                expected_serialized_ids
            ):
                raise ValueError("Resume optimizer state IDs violate the exact live surface")
            for index, parameter in enumerate(parameters):
                raw_state = raw_states[index]
                if not isinstance(raw_state, Mapping) or set(raw_state) != {
                    "step",
                    "exp_avg",
                    "exp_avg_sq",
                }:
                    raise ValueError(f"Resume optimizer raw state {index} has unexpected keys")
                raw_exp_avg = raw_state["exp_avg"]
                raw_exp_avg_sq = raw_state["exp_avg_sq"]
                if (
                    not isinstance(raw_exp_avg, torch.Tensor)
                    or not isinstance(raw_exp_avg_sq, torch.Tensor)
                    or tuple(raw_exp_avg.shape) != tuple(parameter.shape)
                    or tuple(raw_exp_avg_sq.shape) != tuple(parameter.shape)
                ):
                    raise ValueError(
                        f"Resume optimizer raw state {index} is attached to the wrong parameter"
                    )
        load_optimizer_checkpoint(resume_path, optimizer, language.device)
        if train_lora_with_frozen_scene_residual_stack:
            resumed_optimizer_groups = [
                {key: value for key, value in group.items() if key != "params"}
                for group in optimizer.param_groups
            ]
            resumed_optimizer_parameter_ids = [
                id(parameter) for group in optimizer.param_groups for parameter in group["params"]
            ]
            if resumed_optimizer_groups != expected_optimizer_groups:
                raise ValueError(
                    "Exclusive adapter optimizer hyperparameters changed during resume: "
                    f"checkpoint={resumed_optimizer_groups} runtime={expected_optimizer_groups}"
                )
            if resumed_optimizer_parameter_ids != expected_optimizer_parameter_ids:
                raise ValueError(
                    "Exclusive adapter optimizer parameter order changed during resume"
                )
            expected_step = int(resume_metadata.get("optimizer_step", -1))
            if expected_step < 1 or set(optimizer.state) != set(parameters):
                raise ValueError(
                    "Exclusive adapter optimizer state does not cover the exact live surface"
                )
            for index, parameter in enumerate(parameters):
                state = optimizer.state[parameter]
                if set(state) != {"step", "exp_avg", "exp_avg_sq"}:
                    raise ValueError(f"Resume optimizer state {index} has unexpected keys")
                step = state["step"]
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                if (
                    not isinstance(step, torch.Tensor)
                    or not isinstance(exp_avg, torch.Tensor)
                    or not isinstance(exp_avg_sq, torch.Tensor)
                    or step.dtype != torch.float32
                    or exp_avg.dtype != torch.float32
                    or exp_avg_sq.dtype != torch.float32
                    or tuple(step.shape) != ()
                    or tuple(exp_avg.shape) != tuple(parameter.shape)
                    or tuple(exp_avg_sq.shape) != tuple(parameter.shape)
                    or float(step.detach().cpu()) != float(expected_step)
                    or not bool(torch.isfinite(exp_avg).all())
                    or not bool(torch.isfinite(exp_avg_sq).all())
                    or bool(torch.lt(exp_avg_sq, 0).any())
                ):
                    raise ValueError(
                        f"Resume optimizer state {index} violates the exact AdamW tensor contract"
                    )
        start_epoch = int(resume_metadata["epoch"]) + 1
        global_step = int(resume_metadata.get("global_step", 0))
        optimizer_step = int(resume_metadata.get("optimizer_step", 0))
        history = list(resume_metadata.get("history", []))
        best_monitor_loss = float(resume_metadata.get("best_monitor_loss", math.inf))
        best_epoch_value = resume_metadata.get("best_epoch")
        best_epoch = None if best_epoch_value is None else int(best_epoch_value)
        saved_best_pair_gate_passed = resume_metadata.get("best_pair_gate_passed")
        best_pair_gate_passed = (
            bool(saved_best_pair_gate_passed)
            if isinstance(saved_best_pair_gate_passed, bool)
            else best_pair_gate_passed_from_history(history, best_epoch)
        )
        epochs_without_improvement = int(resume_metadata.get("epochs_without_improvement", 0))
        if (
            pair_curriculum.pair_only
            and pair_curriculum.gate_enabled
            and resume_metadata.get("monitor_name") != pair_gate_monitor_name
        ):
            # Eval-only gate policy may change on an otherwise compatible
            # resume. Its scalar monitor is not comparable with the old one.
            best_monitor_loss = math.inf
            best_epoch = None
            best_pair_gate_passed = False
            epochs_without_improvement = 0
        if start_epoch > epochs:
            raise SystemExit(
                f"Resume checkpoint already completed epoch {start_epoch - 1}; "
                f"target epochs is {epochs}"
            )
    scene_state_modules = scene_checkpoint_modules
    current_scene_state_sha256 = module_collection_state_sha256(scene_state_modules)
    current_global_scene_residual_sha256 = (
        None
        if global_scene_residual is None
        else module_collection_state_sha256({"global_scene_residual": global_scene_residual})
    )
    current_signed_x_scene_residual_sha256 = (
        None
        if signed_x_scene_residual is None
        else module_collection_state_sha256({"signed_x_scene_residual": signed_x_scene_residual})
    )
    current_dense_alignment_sha256 = (
        None if dense_aligner is None else dense_aligner.state_sha256()
    )
    current_frozen_lora_hashes = (
        {}
        if lora_installation is None
        else {
            bank.settings.name: bank.installation.state_sha256()
            for bank in lora_installation.banks
            if not bank.settings.trainable
        }
    )
    if (
        train_dense_alignment_only
        and resume_metadata is None
        and current_dense_alignment_sha256 != observed_initial_dense_alignment_sha256
    ):
        raise RuntimeError(
            "Fresh dense-alignment state changed before update-0 equivalence: "
            f"expected={observed_initial_dense_alignment_sha256} "
            f"observed={current_dense_alignment_sha256}"
        )
    if train_lora_with_frozen_scene_residual_stack:
        assert isinstance(lora_installation, LoRABankCollection)
        exact_frozen_lora_hashes = {
            bank.settings.name: bank.settings.expected_initial_state_sha256
            for bank in lora_installation.banks
            if not bank.settings.trainable
        }
        direct_pin_mismatches = {
            field: {"expected": expected, "observed": observed}
            for field, expected, observed in (
                (
                    "scene_state_sha256",
                    initialize_expected_scene_state_sha256,
                    current_scene_state_sha256,
                ),
                (
                    "global_scene_residual_state_sha256",
                    initialize_expected_global_scene_residual_state_sha256,
                    current_global_scene_residual_sha256,
                ),
                (
                    "signed_x_scene_residual_state_sha256",
                    initialize_expected_signed_x_scene_residual_state_sha256,
                    current_signed_x_scene_residual_sha256,
                ),
                (
                    "frozen_lora_bank_state_sha256",
                    exact_frozen_lora_hashes,
                    current_frozen_lora_hashes,
                ),
            )
            if expected != observed
        }
        if direct_pin_mismatches:
            raise ValueError(
                "Frozen source stack differs from direct source pins: "
                f"{direct_pin_mismatches}"
            )
    if resume_metadata is None:
        frozen_scene_state_sha256 = current_scene_state_sha256 if freeze_scene_adapter else None
        frozen_global_scene_residual_state_sha256 = (
            current_global_scene_residual_sha256
            if (
                train_signed_x_scene_residual_only
                or train_lora_with_frozen_scene_residual_stack
                or train_dense_alignment_only
            )
            else None
        )
        frozen_signed_x_scene_residual_state_sha256 = (
            current_signed_x_scene_residual_sha256
            if train_lora_with_frozen_scene_residual_stack or train_dense_alignment_only
            else None
        )
        frozen_lora_bank_state_sha256 = current_frozen_lora_hashes
        if lora_installation is not None:
            initial_hash_mismatches = {
                bank.settings.name: {
                    "expected": bank.settings.expected_initial_state_sha256,
                    "observed": bank.installation.state_sha256(),
                }
                for bank in lora_installation.banks
                if bank.settings.expected_initial_state_sha256 is not None
                and bank.settings.expected_initial_state_sha256 != bank.installation.state_sha256()
            }
            if initial_hash_mismatches:
                raise ValueError(
                    "LoRA bank initial-state hash mismatch before training: "
                    f"{initial_hash_mismatches}"
                )
    else:
        frozen_scene_state_sha256 = resume_metadata.get("frozen_scene_state_sha256")
        frozen_global_scene_residual_state_sha256 = resume_metadata.get(
            "frozen_global_scene_residual_state_sha256"
        )
        frozen_signed_x_scene_residual_state_sha256 = resume_metadata.get(
            "frozen_signed_x_scene_residual_state_sha256"
        )
        frozen_lora_bank_state_sha256 = resume_metadata.get("frozen_lora_bank_state_sha256", {})
        if freeze_scene_adapter and frozen_scene_state_sha256 != current_scene_state_sha256:
            raise ValueError(
                "Frozen scene adapter changed across resume: "
                f"checkpoint={frozen_scene_state_sha256} runtime={current_scene_state_sha256}"
            )
        if (
            train_signed_x_scene_residual_only
            or train_lora_with_frozen_scene_residual_stack
            or train_dense_alignment_only
        ) and frozen_global_scene_residual_state_sha256 != current_global_scene_residual_sha256:
            raise ValueError(
                "Frozen global scene residual changed across signed-X resume: "
                f"checkpoint={frozen_global_scene_residual_state_sha256} "
                f"runtime={current_global_scene_residual_sha256}"
            )
        if (
            (train_lora_with_frozen_scene_residual_stack or train_dense_alignment_only)
            and frozen_signed_x_scene_residual_state_sha256
            != current_signed_x_scene_residual_sha256
        ):
            raise ValueError(
                "Frozen signed-X residual changed across LoRA resume: "
                f"checkpoint={frozen_signed_x_scene_residual_state_sha256} "
                f"runtime={current_signed_x_scene_residual_sha256}"
            )
        if frozen_lora_bank_state_sha256 != current_frozen_lora_hashes:
            raise ValueError(
                "Frozen LoRA bank changed across resume: "
                f"checkpoint={frozen_lora_bank_state_sha256} "
                f"runtime={current_frozen_lora_hashes}"
            )
    if signed_x_scene_residual is not None:
        assert global_scene_residual is not None
        zero_output_equivalence = None
        if resume_metadata is None and (
            train_lora_with_frozen_scene_residual_stack or train_dense_alignment_only
        ):
            source_equivalence = loaded_initialization.get(
                "signed_x_scene_residual_zero_output_equivalence"
            )
            if (
                not isinstance(source_equivalence, dict)
                or source_equivalence.get("verified") is not True
            ):
                raise ValueError(
                    "Frozen signed-X source lacks its verified update-0 base equivalence"
                )
            signed_x_zero_output_equivalence = source_equivalence
        elif resume_metadata is None:
            signed_x_zero_output_equivalence = verify_zero_output_signed_x_residual_equivalence(
                scene_model,
                global_scene_residual,
                signed_x_scene_residual,
                composer,
                maps,
                model_dtype=next(language.model.parameters()).dtype,
            )
        else:
            saved_signed_x_equivalence = resume_metadata.get(
                "signed_x_scene_residual_zero_output_equivalence"
            )
            if (
                not isinstance(saved_signed_x_equivalence, dict)
                or saved_signed_x_equivalence.get("verified") is not True
            ):
                raise ValueError(
                    "Signed-X resume checkpoint lacks verified update-0 base equivalence"
                )
            signed_x_zero_output_equivalence = saved_signed_x_equivalence
    elif global_scene_residual is not None:
        signed_x_zero_output_equivalence = None
        if resume_metadata is None:
            zero_output_equivalence = verify_zero_output_scene_residual_equivalence(
                scene_model,
                global_scene_residual,
                composer,
                maps,
                model_dtype=next(language.model.parameters()).dtype,
            )
        else:
            saved_equivalence = resume_metadata.get("global_scene_residual_zero_output_equivalence")
            if (
                not isinstance(saved_equivalence, dict)
                or saved_equivalence.get("verified") is not True
            ):
                raise ValueError(
                    "Residual resume checkpoint lacks verified update-0 prefix equivalence"
                )
            zero_output_equivalence = saved_equivalence
    else:
        zero_output_equivalence = None
        signed_x_zero_output_equivalence = None
    if dense_aligner is not None:
        assert global_scene_residual is not None
        assert signed_x_scene_residual is not None
        if resume_metadata is None:
            dense_alignment_zero_output_equivalence = (
                verify_zero_output_dense_alignment_equivalence(
                    scene_model,
                    global_scene_residual,
                    signed_x_scene_residual,
                    dense_aligner,
                    composer,
                    maps,
                    model_dtype=next(language.model.parameters()).dtype,
                )
            )
            if initialization_provenance is not None:
                initialization_provenance["dense_alignment_zero_output_equivalence"] = (
                    dense_alignment_zero_output_equivalence
                )
        else:
            saved_dense_equivalence = resume_metadata.get(
                "dense_alignment_zero_output_equivalence"
            )
            if (
                not isinstance(saved_dense_equivalence, dict)
                or saved_dense_equivalence.get("verified") is not True
            ):
                raise ValueError(
                    "Dense-alignment resume checkpoint lacks verified update-0 equivalence"
                )
            dense_alignment_zero_output_equivalence = saved_dense_equivalence
    else:
        dense_alignment_zero_output_equivalence = None
    if dense_aligner is not None:
        if observed_initial_dense_alignment_sha256 is None:
            raise RuntimeError("Enabled dense alignment lost its deterministic initial hash")
        if resume_metadata is None:
            pair_optimizer_empty_before_warmup = not optimizer.state
            if (
                not pair_optimizer_empty_before_warmup
                or optimizer_step != 0
                or global_step != 0
            ):
                raise RuntimeError(
                    "Dense calibration must precede every paired-QA optimizer update/state"
                )
            dense_training_device = dense_aligner.alignment_a.device
            dense_aligner.to("cpu")
            try:
                dense_alignment_calibration = run_dense_alignment_calibration_warmup(
                    config,
                    dense_aligner,
                )
                require_dense_alignment_calibration_authorized(
                    dense_alignment_calibration
                )
            finally:
                dense_aligner.to(dense_training_device)

            # The calibration optimizer is deliberately local to the isolated
            # runner. Rebuild the paired-QA optimizer against the post-warmup
            # device tensors so QA update 1 starts from empty AdamW state.
            optimizer, parameters = build_adapter_optimizer(
                config,
                scene_parameters,
                lora_installation,
                configured_lora_optimizer,
                dense_alignment_parameters=tuple(dense_aligner.parameters()),
            )
            if optimizer.state:
                raise RuntimeError("Rebuilt paired-QA optimizer unexpectedly retained state")
            expected_dense_ids = [id(parameter) for parameter in dense_aligner.parameters()]
            observed_dense_ids = [id(parameter) for parameter in parameters]
            if observed_dense_ids != expected_dense_ids:
                raise RuntimeError(
                    "Rebuilt paired-QA optimizer is detached from the calibrated bridge"
                )
            if [group.get("name") for group in optimizer.param_groups] != [
                "dense_alignment"
            ]:
                raise RuntimeError("Rebuilt paired-QA optimizer has an invalid group surface")
            dense_alignment_calibration.update(
                {
                    "pair_optimizer_state_empty_before_warmup": (
                        pair_optimizer_empty_before_warmup
                    ),
                    "pair_optimizer_rebuilt_after_warmup": True,
                    "pair_optimizer_state_empty_after_warmup": not optimizer.state,
                    "pair_optimizer_steps_before_qa": optimizer_step,
                    "held_out_scene_gradient_access": False,
                    "category_text_prototypes_serialized": False,
                    "oracle_payload_retained": False,
                }
            )
            validate_dense_alignment_calibration_audit(
                dense_alignment_calibration,
                dense_aligner,
                expected_initial_state_sha256=observed_initial_dense_alignment_sha256,
            )
            if initialization_provenance is not None:
                initialization_provenance.update(
                    {
                        "dense_alignment_calibration_authorized": True,
                        "dense_alignment_calibration_final_state_sha256": (
                            dense_aligner.state_sha256()
                        ),
                        "pair_optimizer_rebuilt_after_dense_alignment_calibration": True,
                    }
                )
        else:
            saved_calibration = resume_metadata.get("dense_alignment_calibration")
            if not isinstance(saved_calibration, Mapping):
                raise ValueError(
                    "Dense-alignment resume checkpoint lacks its calibration audit"
                )
            dense_alignment_calibration = dict(saved_calibration)
            resume_initialization = resume_metadata.get("initialization_provenance")
            if not isinstance(resume_initialization, Mapping):
                raise ValueError(
                    "Dense-alignment resume checkpoint lacks initialization provenance"
                )
            calibration_final_hash = resume_initialization.get(
                "dense_alignment_calibration_final_state_sha256"
            )
            if (
                not isinstance(calibration_final_hash, str)
                or re.fullmatch(r"[0-9a-f]{64}", calibration_final_hash) is None
            ):
                raise ValueError(
                    "Dense-alignment resume checkpoint lacks a valid calibration-final hash"
                )
            validate_dense_alignment_calibration_audit(
                dense_alignment_calibration,
                dense_aligner,
                expected_initial_state_sha256=observed_initial_dense_alignment_sha256,
                expected_calibration_final_state_sha256=calibration_final_hash,
            )
    else:
        dense_alignment_calibration = None
    if resume_metadata is None:
        spatial_answer_warmup_metrics = run_spatial_answer_warmup(
            scene_model,
            maps,
            pair_units,
            language,
            latent_count=int(config["scene_encoder"]["global_latents"]),
            settings=spatial_answer_warmup,
        )
        spatial_relation_warmup_metrics = run_spatial_relation_warmup(
            scene_model,
            maps,
            pair_units,
            language,
            settings=spatial_relation_warmup,
        )
    else:
        saved_warmup_metrics = resume_metadata.get("spatial_answer_warmup_metrics")
        spatial_answer_warmup_metrics = (
            saved_warmup_metrics
            if isinstance(saved_warmup_metrics, dict)
            else run_spatial_answer_warmup(
                scene_model,
                maps,
                pair_units,
                language,
                latent_count=int(config["scene_encoder"]["global_latents"]),
                settings=spatial_answer_warmup,
            )
        )
        saved_relation_warmup_metrics = resume_metadata.get("spatial_relation_warmup_metrics")
        spatial_relation_warmup_metrics = (
            saved_relation_warmup_metrics
            if isinstance(saved_relation_warmup_metrics, dict)
            else run_spatial_relation_warmup(
                scene_model,
                maps,
                pair_units,
                language,
                settings=spatial_relation_warmup,
            )
        )
    optimizer.zero_grad(set_to_none=True)

    validation_interval = int(config["training"].get("validation_every_epochs", 1))
    if validation_interval < 1:
        raise ValueError("validation_every_epochs must be positive")
    early_stopping_patience = int(config["training"].get("early_stopping_patience", 0))
    min_delta = float(config["training"].get("early_stopping_min_delta", 0.0))
    stopped_early = False
    accumulated_batches = 0

    for epoch in range(start_epoch, epochs + 1):
        epoch_losses: list[float] = []
        epoch_language_losses: list[float] = []
        epoch_weighted_language_losses: list[float] = []
        epoch_grounding_losses: list[float] = []
        epoch_ranking_losses: list[float] = []
        epoch_weighted_ranking_losses: list[float] = []
        epoch_ranking_margins: list[float] = []
        epoch_ranking_min_margins: list[float] = []
        epoch_ranking_side_accuracies: list[float] = []
        epoch_ranking_unit_accuracies: list[float] = []
        epoch_full_vocab_ranking_losses: list[float] = []
        epoch_weighted_full_vocab_ranking_losses: list[float] = []
        epoch_full_vocab_ranking_margins: list[float] = []
        epoch_full_vocab_ranking_min_margins: list[float] = []
        epoch_full_vocab_side_accuracies: list[float] = []
        epoch_full_vocab_unit_accuracies: list[float] = []
        epoch_pair_scene_token_gradient_norms: list[float] = []
        epoch_lora_gradient_norms: list[float] = []
        epoch_dense_alignment_gradient_norms: list[float] = []
        epoch_spatial_answer_losses: list[float] = []
        epoch_spatial_answer_own_similarities: list[float] = []
        epoch_spatial_answer_alternate_similarities: list[float] = []
        epoch_spatial_answer_margins: list[float] = []
        epoch_spatial_answer_eligible_units: list[int] = []
        epoch_spatial_relation_losses: list[float] = []
        epoch_spatial_relation_margins: list[float] = []
        epoch_spatial_relation_minimum_margins: list[float] = []
        epoch_spatial_relation_eligible_units: list[int] = []
        epoch_diversity_losses: list[float] = []
        epoch_diversity_cosines: list[float] = []
        epoch_diversity_max_cosines: list[float] = []
        epoch_pair_losses: list[float] = []
        epoch_pair_cosines: list[float] = []
        epoch_pair_distances: list[float] = []
        curriculum = build_epoch_curriculum(
            by_scene,
            pair_units,
            standard_batch_size=batch_size,
            pair_units_per_batch=pair_curriculum.units_per_batch,
            pair_batch_fraction=pair_curriculum.batch_fraction,
            pair_only=pair_curriculum.pair_only,
            seed=int(config["seed"]) + epoch,
            steps_per_epoch=pair_curriculum.steps_per_epoch,
        )
        pair_batch_count = sum(batch.kind == "pair" for batch in curriculum)
        actual_pair_batch_fraction = pair_batch_count / len(curriculum)
        separated_pair_ids: set[str] = set()
        scene_model.train(not freeze_scene_adapter)
        composer.train(not freeze_scene_adapter)
        grounding.train(not freeze_scene_adapter)
        if global_scene_residual is not None:
            global_scene_residual.train(
                not (
                    train_signed_x_scene_residual_only
                    or train_lora_with_frozen_scene_residual_stack
                    or train_dense_alignment_only
                )
            )
        if signed_x_scene_residual is not None:
            signed_x_scene_residual.train(
                not (
                    train_lora_with_frozen_scene_residual_stack
                    or train_dense_alignment_only
                )
            )
        if dense_aligner is not None:
            dense_aligner.train(train_dense_alignment_only)
        if lora_installation is not None:
            lora_installation.train()
        for curriculum_batch in curriculum:
            effective_pair_objective: PairObjectivePolicy = pair_objectives.legacy_default
            pair_ranking_loss = torch.zeros((), device=language.device)
            full_vocab_ranking_loss = torch.zeros((), device=language.device)
            pair_ranking_diagnostics: dict[str, torch.Tensor | str | None] | None = None
            spatial_answer_loss = torch.zeros((), device=language.device)
            spatial_answer_diagnostics: dict[str, torch.Tensor | int] | None = None
            spatial_relation_loss = torch.zeros((), device=language.device)
            spatial_relation_diagnostics: dict[str, torch.Tensor | int] | None = None
            partner_output = None
            if curriculum_batch.kind == "standard":
                scene_id = str(curriculum_batch.scene_id)
                batch_records = curriculum_batch.records
                data = maps[scene_id]
                output = training_map_forward(
                    scene_model,
                    data,
                    freeze_scene_adapter=freeze_scene_adapter,
                    global_scene_residual=global_scene_residual,
                    signed_x_scene_residual=signed_x_scene_residual,
                    dense_aligner=dense_aligner,
                )
                outputs_by_scene = {scene_id: output}
                base_loss, language_loss, grounding_loss = batch_objective(
                    output,
                    batch_records,
                    data,
                    language,
                    composer,
                    grounding,
                    config,
                )
                pair_spec = pair_for_first_scene.get(scene_id)
                if (
                    anti_collapse["paired_scene_separation_weight"] > 0
                    and pair_spec is not None
                    and pair_spec[0] not in separated_pair_ids
                ):
                    pair_id, partner_scene_id = pair_spec
                    partner_output = training_map_forward(
                        scene_model,
                        maps[partner_scene_id],
                        freeze_scene_adapter=freeze_scene_adapter,
                        global_scene_residual=global_scene_residual,
                        signed_x_scene_residual=signed_x_scene_residual,
                        dense_aligner=dense_aligner,
                    )
                    separated_pair_ids.add(pair_id)
                log_scene_ids = [scene_id]
                pair_unit_count = 0
            else:
                units = curriculum_batch.pair_units
                if curriculum_batch.pair_id is None:
                    raise RuntimeError("Pair curriculum batch has no opaque pair ID")
                effective_pair_objective = pair_objectives.policy_for(curriculum_batch.pair_id)
                scene_ids = units[0].scene_ids
                if any(unit.scene_ids != scene_ids for unit in units):
                    raise ValueError("A pair batch contains inconsistent scene ordering")
                outputs_by_scene = {
                    scene_id: training_map_forward(
                        scene_model,
                        maps[scene_id],
                        freeze_scene_adapter=freeze_scene_adapter,
                        global_scene_residual=global_scene_residual,
                        signed_x_scene_residual=signed_x_scene_residual,
                        dense_aligner=dense_aligner,
                    )
                    for scene_id in scene_ids
                }
                for scene_output in outputs_by_scene.values():
                    if scene_output.scene_tokens.requires_grad:
                        scene_output.scene_tokens.retain_grad()
                (
                    base_loss,
                    language_loss,
                    grounding_loss,
                    pair_ranking_loss,
                    pair_ranking_diagnostics,
                ) = pair_batch_objective(
                    outputs_by_scene,
                    units,
                    maps,
                    language,
                    composer,
                    grounding,
                    config,
                    ranking_margin=pair_curriculum.ranking_margin,
                    ranking_mode=pair_curriculum.ranking_mode,
                    collect_full_vocab_first_answer_token=(
                        effective_pair_objective.full_vocab_hinge_weight > 0
                    ),
                    full_vocab_ranking_margin=(effective_pair_objective.full_vocab_margin),
                )
                margins = pair_ranking_diagnostics["margins"]
                if not isinstance(margins, torch.Tensor):
                    raise RuntimeError("Pair objective did not return differentiable margins")
                pair_ranking_loss, _ = ranking_margin_hinge(
                    margins,
                    margin=effective_pair_objective.candidate_margin,
                )
                spatial_answer_loss = pair_ranking_diagnostics["spatial_answer_contrastive_loss"]
                spatial_answer_diagnostics = pair_ranking_diagnostics["spatial_answer_contrastive"]
                spatial_relation_loss = pair_ranking_diagnostics[
                    "spatial_relation_contrastive_loss"
                ]
                spatial_relation_diagnostics = pair_ranking_diagnostics[
                    "spatial_relation_contrastive"
                ]
                full_vocab_ranking_loss = pair_ranking_diagnostics[
                    "first_answer_token_full_vocab_ranking_loss"
                ]
                if anti_collapse["paired_scene_separation_weight"] > 0:
                    # Pair batches already paid for both scene forwards. Apply
                    # separation on every paired update rather than only once
                    # per epoch; the once-per-epoch policy was too weak to
                    # escape the balanced answer-ranking saddle.
                    partner_output = outputs_by_scene[scene_ids[1]]
                log_scene_ids = list(scene_ids)
                pair_unit_count = len(units)

            diversity_losses = []
            diversity_cosines = []
            diversity_max_cosines = []
            if anti_collapse["latent_diversity_weight"] > 0:
                for scene_output in outputs_by_scene.values():
                    one_diversity_loss, one_diversity_diagnostics = latent_diversity_loss(
                        scene_output.native_latents,
                        cosine_margin=float(anti_collapse["latent_diversity_cosine_margin"]),
                        max_latents=int(anti_collapse["latent_diversity_max_latents"]),
                    )
                    diversity_losses.append(one_diversity_loss)
                    diversity_cosines.append(one_diversity_diagnostics["mean_off_diagonal_cosine"])
                    diversity_max_cosines.append(
                        one_diversity_diagnostics["max_off_diagonal_cosine"]
                    )
            diversity_loss = (
                torch.stack(diversity_losses).mean()
                if diversity_losses
                else torch.zeros((), device=language.device)
            )
            diversity_mean_cosine = (
                torch.stack(diversity_cosines).mean()
                if diversity_cosines
                else torch.zeros((), device=language.device)
            )
            diversity_max_cosine = (
                torch.stack(diversity_max_cosines).mean()
                if diversity_max_cosines
                else torch.zeros((), device=language.device)
            )

            pair_loss = torch.zeros((), device=language.device)
            pair_diagnostics: dict[str, torch.Tensor] | None = None
            if partner_output is not None:
                first_output = outputs_by_scene[log_scene_ids[0]]
                pair_loss, pair_diagnostics = paired_scene_separation_loss(
                    first_output.native_latents,
                    partner_output.native_latents,
                    cosine_distance_margin=float(
                        anti_collapse["paired_scene_cosine_distance_margin"]
                    ),
                )
                del first_output

            loss = combine_pair_training_losses(
                base_loss,
                pair_ranking_loss,
                full_vocab_ranking_loss,
                diversity_loss,
                pair_loss,
                language_loss=language_loss,
                language_nll_weight=effective_pair_objective.language_nll_weight,
                pair_ranking_weight=effective_pair_objective.candidate_hinge_weight,
                full_vocab_ranking_weight=effective_pair_objective.full_vocab_hinge_weight,
                diversity_weight=float(anti_collapse["latent_diversity_weight"]),
                scene_separation_weight=float(anti_collapse["paired_scene_separation_weight"]),
            )
            (loss / accumulation).backward()
            pair_scene_token_gradient_norms = None
            if curriculum_batch.kind == "pair" and (
                not freeze_scene_adapter or train_dense_alignment_only
            ):
                pair_scene_token_gradient_norms = {
                    scene_id: (
                        None
                        if scene_output.scene_tokens.grad is None
                        else float(scene_output.scene_tokens.grad.detach().float().norm().cpu())
                    )
                    for scene_id, scene_output in outputs_by_scene.items()
                }
                if any(value is None for value in pair_scene_token_gradient_norms.values()):
                    raise RuntimeError("Pair objective is detached from a projected scene prefix")
                epoch_pair_scene_token_gradient_norms.extend(
                    value for value in pair_scene_token_gradient_norms.values() if value is not None
                )
            global_step += 1
            accumulated_batches += 1
            lora_gradient_norms = (
                None if lora_installation is None else lora_installation.gradient_norms()
            )
            dense_alignment_gradient_norm = None
            if accumulated_batches == accumulation:
                if lora_gradient_norms is not None:
                    epoch_lora_gradient_norms.append(float(lora_gradient_norms["total_l2"]))
                    if (
                        freeze_scene_adapter
                        and configured_lora.trainable
                        and float(lora_gradient_norms["total_l2"]) <= 0.0
                    ):
                        raise RuntimeError("Frozen-scene training produced no LoRA-bank gradient")
                if train_dense_alignment_only:
                    assert dense_aligner is not None
                    dense_alignment_gradient_norm = parameter_gradient_l2(
                        tuple(dense_aligner.parameters())
                    )
                    epoch_dense_alignment_gradient_norms.append(
                        dense_alignment_gradient_norm
                    )
                    if dense_alignment_gradient_norm <= 0.0:
                        raise RuntimeError(
                            "Dense-alignment-only training produced no dense gradient"
                        )
                torch.nn.utils.clip_grad_norm_(
                    parameters, float(config["training"]["gradient_clip_norm"])
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1
                accumulated_batches = 0
            scalar = float(loss.detach().cpu())
            epoch_losses.append(scalar)
            epoch_language_losses.append(float(language_loss.detach().cpu()))
            epoch_weighted_language_losses.append(
                float(effective_pair_objective.language_nll_weight * language_loss.detach().cpu())
            )
            epoch_grounding_losses.append(float(grounding_loss.detach().cpu()))
            epoch_diversity_losses.append(float(diversity_loss.detach().cpu()))
            epoch_diversity_cosines.append(float(diversity_mean_cosine.detach().cpu()))
            epoch_diversity_max_cosines.append(float(diversity_max_cosine.detach().cpu()))
            if pair_ranking_diagnostics is not None:
                margins = pair_ranking_diagnostics["margins"]
                epoch_ranking_losses.append(float(pair_ranking_loss.detach().cpu()))
                epoch_weighted_ranking_losses.append(
                    float(
                        effective_pair_objective.candidate_hinge_weight
                        * pair_ranking_loss.detach().cpu()
                    )
                )
                epoch_ranking_margins.append(float(margins.detach().mean().cpu()))
                epoch_ranking_min_margins.append(float(margins.detach().min().cpu()))
                epoch_ranking_side_accuracies.append(
                    float(pair_ranking_diagnostics["side_accuracy"].detach().cpu())
                )
                epoch_ranking_unit_accuracies.append(
                    float(pair_ranking_diagnostics["unit_accuracy"].detach().cpu())
                )
                full_vocab_margins = pair_ranking_diagnostics[
                    "first_answer_token_full_vocab_margins"
                ]
                epoch_weighted_full_vocab_ranking_losses.append(
                    float(
                        effective_pair_objective.full_vocab_hinge_weight
                        * full_vocab_ranking_loss.detach().cpu()
                    )
                )
                if full_vocab_margins is not None:
                    epoch_full_vocab_ranking_losses.append(
                        float(full_vocab_ranking_loss.detach().cpu())
                    )
                    epoch_full_vocab_ranking_margins.append(
                        float(full_vocab_margins.detach().mean().cpu())
                    )
                    epoch_full_vocab_ranking_min_margins.append(
                        float(full_vocab_margins.detach().min().cpu())
                    )
                    epoch_full_vocab_side_accuracies.append(
                        float(full_vocab_margins.detach().gt(0).float().mean().cpu())
                    )
                    epoch_full_vocab_unit_accuracies.append(
                        float(full_vocab_margins.detach().gt(0).all(dim=1).float().mean().cpu())
                    )
            if spatial_answer_diagnostics is not None:
                epoch_spatial_answer_losses.append(float(spatial_answer_loss.detach().cpu()))
                epoch_spatial_answer_own_similarities.append(
                    float(spatial_answer_diagnostics["own_similarity"].mean().cpu())
                )
                epoch_spatial_answer_alternate_similarities.append(
                    float(spatial_answer_diagnostics["alternate_similarity"].mean().cpu())
                )
                epoch_spatial_answer_margins.append(
                    float(spatial_answer_diagnostics["achieved_margin"].mean().cpu())
                )
                epoch_spatial_answer_eligible_units.append(
                    int(spatial_answer_diagnostics["eligible_unit_count"])
                )
            if spatial_relation_diagnostics is not None:
                relation_margins = spatial_relation_diagnostics["achieved_margin"]
                if not isinstance(relation_margins, torch.Tensor):
                    raise TypeError("Spatial-relation achieved margins must be a tensor")
                epoch_spatial_relation_losses.append(float(spatial_relation_loss.detach().cpu()))
                epoch_spatial_relation_margins.append(float(relation_margins.detach().mean().cpu()))
                epoch_spatial_relation_minimum_margins.append(
                    float(relation_margins.detach().min().cpu())
                )
                epoch_spatial_relation_eligible_units.append(
                    int(spatial_relation_diagnostics["eligible_unit_count"])
                )
            if pair_diagnostics is not None:
                epoch_pair_losses.append(float(pair_loss.detach().cpu()))
                epoch_pair_cosines.append(float(pair_diagnostics["mean_aligned_cosine"].cpu()))
                epoch_pair_distances.append(float(pair_diagnostics["cosine_distance"].cpu()))
            print(
                json.dumps(
                    {
                        "phase": "training",
                        "epoch": epoch,
                        "step": global_step,
                        "batch_kind": curriculum_batch.kind,
                        "scenes": log_scene_ids,
                        "counterfactual_pair_id": curriculum_batch.pair_id,
                        "pair_unit_count": pair_unit_count,
                        "pair_objective_policy": (
                            None
                            if curriculum_batch.kind != "pair"
                            else effective_pair_objective.contract()
                        ),
                        "loss": scalar,
                        "language_loss": float(language_loss.detach().cpu()),
                        "grounding_loss": float(grounding_loss.detach().cpu()),
                        "pair_ranking_loss": (
                            None
                            if pair_ranking_diagnostics is None
                            else float(pair_ranking_loss.detach().cpu())
                        ),
                        "pair_ranking_mode": pair_curriculum.ranking_mode,
                        "pair_ranking_margin_name": (
                            None
                            if pair_ranking_diagnostics is None
                            else pair_ranking_diagnostics["margin_name"]
                        ),
                        "pair_ranking_margin": (
                            None
                            if pair_ranking_diagnostics is None
                            else float(pair_ranking_diagnostics["margins"].detach().mean().cpu())
                        ),
                        "pair_full_vocab_ranking_loss": (
                            None
                            if pair_ranking_diagnostics is None
                            or pair_ranking_diagnostics["first_answer_token_full_vocab_margins"]
                            is None
                            else float(full_vocab_ranking_loss.detach().cpu())
                        ),
                        "pair_full_vocab_weighted_ranking_loss": (
                            None
                            if pair_ranking_diagnostics is None
                            or pair_ranking_diagnostics["first_answer_token_full_vocab_margins"]
                            is None
                            else float(
                                effective_pair_objective.full_vocab_hinge_weight
                                * full_vocab_ranking_loss.detach().cpu()
                            )
                        ),
                        "pair_full_vocab_mean_margin": (
                            None
                            if pair_ranking_diagnostics is None
                            or pair_ranking_diagnostics["first_answer_token_full_vocab_margins"]
                            is None
                            else float(
                                pair_ranking_diagnostics["first_answer_token_full_vocab_margins"]
                                .detach()
                                .mean()
                                .cpu()
                            )
                        ),
                        "pair_full_vocab_minimum_margin": (
                            None
                            if pair_ranking_diagnostics is None
                            or pair_ranking_diagnostics["first_answer_token_full_vocab_margins"]
                            is None
                            else float(
                                pair_ranking_diagnostics["first_answer_token_full_vocab_margins"]
                                .detach()
                                .min()
                                .cpu()
                            )
                        ),
                        "pair_full_vocab_top1_side_accuracy": (
                            None
                            if pair_ranking_diagnostics is None
                            or pair_ranking_diagnostics["first_answer_token_full_vocab_margins"]
                            is None
                            else float(
                                pair_ranking_diagnostics["first_answer_token_full_vocab_margins"]
                                .detach()
                                .gt(0)
                                .float()
                                .mean()
                                .cpu()
                            )
                        ),
                        "pair_full_vocab_top1_unit_accuracy": (
                            None
                            if pair_ranking_diagnostics is None
                            or pair_ranking_diagnostics["first_answer_token_full_vocab_margins"]
                            is None
                            else float(
                                pair_ranking_diagnostics["first_answer_token_full_vocab_margins"]
                                .detach()
                                .gt(0)
                                .all(dim=1)
                                .float()
                                .mean()
                                .cpu()
                            )
                        ),
                        "correct_vs_swapped_nll_margin": (
                            None
                            if pair_ranking_diagnostics is None
                            or pair_curriculum.ranking_mode != "nll"
                            else float(pair_ranking_diagnostics["margins"].detach().mean().cpu())
                        ),
                        "minimum_correct_vs_swapped_nll_margin": (
                            None
                            if pair_ranking_diagnostics is None
                            or pair_curriculum.ranking_mode != "nll"
                            else float(pair_ranking_diagnostics["margins"].detach().min().cpu())
                        ),
                        "own_vs_alternate_candidate_logit_margin": (
                            None
                            if pair_ranking_diagnostics is None
                            or pair_curriculum.ranking_mode != "candidate_logit"
                            else float(pair_ranking_diagnostics["margins"].detach().mean().cpu())
                        ),
                        "minimum_own_vs_alternate_candidate_logit_margin": (
                            None
                            if pair_ranking_diagnostics is None
                            or pair_curriculum.ranking_mode != "candidate_logit"
                            else float(pair_ranking_diagnostics["margins"].detach().min().cpu())
                        ),
                        "pair_side_accuracy": (
                            None
                            if pair_ranking_diagnostics is None
                            else float(pair_ranking_diagnostics["side_accuracy"].detach().cpu())
                        ),
                        "pair_scene_token_gradient_norms": (pair_scene_token_gradient_norms),
                        "lora_gradient_norms": lora_gradient_norms,
                        "dense_alignment_gradient_norm": (
                            dense_alignment_gradient_norm
                        ),
                        "spatial_answer_contrastive_loss": (
                            None
                            if spatial_answer_diagnostics is None
                            else float(spatial_answer_loss.detach().cpu())
                        ),
                        "spatial_answer_own_similarity": (
                            None
                            if spatial_answer_diagnostics is None
                            else float(spatial_answer_diagnostics["own_similarity"].mean().cpu())
                        ),
                        "spatial_answer_alternate_similarity": (
                            None
                            if spatial_answer_diagnostics is None
                            else float(
                                spatial_answer_diagnostics["alternate_similarity"].mean().cpu()
                            )
                        ),
                        "spatial_answer_achieved_margin": (
                            None
                            if spatial_answer_diagnostics is None
                            else float(spatial_answer_diagnostics["achieved_margin"].mean().cpu())
                        ),
                        "spatial_answer_configured_margin": (
                            None
                            if spatial_answer_diagnostics is None
                            else spatial_answer_contrastive["margin"]
                        ),
                        "spatial_answer_eligible_units": (
                            None
                            if spatial_answer_diagnostics is None
                            else int(spatial_answer_diagnostics["eligible_unit_count"])
                        ),
                        "spatial_answer_unique_targets": (
                            None
                            if spatial_answer_diagnostics is None
                            else int(spatial_answer_diagnostics["unique_target_count"])
                        ),
                        "spatial_relation_contrastive_loss": (
                            None
                            if spatial_relation_diagnostics is None
                            else float(spatial_relation_loss.detach().cpu())
                        ),
                        "spatial_relation_mean_achieved_margin": (
                            None
                            if spatial_relation_diagnostics is None
                            else float(
                                spatial_relation_diagnostics["achieved_margin"]
                                .detach()
                                .mean()
                                .cpu()
                            )
                        ),
                        "spatial_relation_minimum_achieved_margin": (
                            None
                            if spatial_relation_diagnostics is None
                            else float(
                                spatial_relation_diagnostics["achieved_margin"].detach().min().cpu()
                            )
                        ),
                        "spatial_relation_configured_margin": (
                            None
                            if spatial_relation_diagnostics is None
                            else spatial_relation_contrastive["margin"]
                        ),
                        "spatial_relation_temperature": (
                            None
                            if spatial_relation_diagnostics is None
                            else spatial_relation_contrastive["temperature"]
                        ),
                        "spatial_relation_eligible_units": (
                            None
                            if spatial_relation_diagnostics is None
                            else int(spatial_relation_diagnostics["eligible_unit_count"])
                        ),
                        "spatial_relation_unique_ordered_regions": (
                            None
                            if spatial_relation_diagnostics is None
                            else int(spatial_relation_diagnostics["unique_ordered_region_count"])
                        ),
                        "latent_diversity_loss": float(diversity_loss.detach().cpu()),
                        "latent_mean_off_diagonal_cosine": float(
                            diversity_mean_cosine.detach().cpu()
                        ),
                        "paired_scene_separation_loss": (
                            None if pair_diagnostics is None else float(pair_loss.detach().cpu())
                        ),
                        "paired_scene_cosine_distance": (
                            None
                            if pair_diagnostics is None
                            else float(pair_diagnostics["cosine_distance"].cpu())
                        ),
                        "voxels": sum(maps[scene_id].voxel_count for scene_id in log_scene_ids),
                        "raw_voxels": sum(
                            maps[scene_id].source_voxel_count for scene_id in log_scene_ids
                        ),
                        "blocks": sum(
                            int(scene_output.audit["voxel_counts"].numel())
                            for scene_output in outputs_by_scene.values()
                        ),
                    }
                ),
                flush=True,
            )
            del outputs_by_scene, partner_output
            if curriculum_batch.kind == "standard":
                del output
        if accumulated_batches:
            if lora_installation is not None:
                lora_gradient_norms = lora_installation.gradient_norms()
                epoch_lora_gradient_norms.append(float(lora_gradient_norms["total_l2"]))
                if (
                    freeze_scene_adapter
                    and configured_lora.trainable
                    and float(lora_gradient_norms["total_l2"]) <= 0.0
                ):
                    raise RuntimeError("Frozen-scene training produced no LoRA-bank gradient")
                print(
                    json.dumps(
                        {
                            "phase": "lora_gradient",
                            "epoch": epoch,
                            "optimizer_step": optimizer_step + 1,
                            "gradient_norms": lora_gradient_norms,
                        }
                    ),
                    flush=True,
                )
            if train_dense_alignment_only:
                assert dense_aligner is not None
                dense_alignment_gradient_norm = parameter_gradient_l2(
                    tuple(dense_aligner.parameters())
                )
                epoch_dense_alignment_gradient_norms.append(
                    dense_alignment_gradient_norm
                )
                if dense_alignment_gradient_norm <= 0.0:
                    raise RuntimeError(
                        "Dense-alignment-only training produced no dense gradient"
                    )
            torch.nn.utils.clip_grad_norm_(
                parameters, float(config["training"]["gradient_clip_norm"])
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_step += 1
            accumulated_batches = 0
        mean_loss = float(np.mean(epoch_losses))
        mean_language_loss = float(np.mean(epoch_language_losses))
        mean_weighted_language_loss = float(np.mean(epoch_weighted_language_losses))
        mean_grounding_loss = float(np.mean(epoch_grounding_losses))
        mean_diversity_loss = float(np.mean(epoch_diversity_losses))
        mean_diversity_cosine = float(np.mean(epoch_diversity_cosines))
        mean_diversity_max_cosine = float(np.mean(epoch_diversity_max_cosines))
        mean_pair_loss = float(np.mean(epoch_pair_losses)) if epoch_pair_losses else 0.0
        mean_pair_cosine = float(np.mean(epoch_pair_cosines)) if epoch_pair_cosines else None
        mean_pair_distance = float(np.mean(epoch_pair_distances)) if epoch_pair_distances else None
        mean_ranking_loss = float(np.mean(epoch_ranking_losses)) if epoch_ranking_losses else None
        mean_weighted_ranking_loss = (
            float(np.mean(epoch_weighted_ranking_losses)) if epoch_weighted_ranking_losses else None
        )
        mean_ranking_margin = (
            float(np.mean(epoch_ranking_margins)) if epoch_ranking_margins else None
        )
        minimum_ranking_margin = (
            float(np.min(epoch_ranking_min_margins)) if epoch_ranking_min_margins else None
        )
        mean_ranking_side_accuracy = (
            float(np.mean(epoch_ranking_side_accuracies)) if epoch_ranking_side_accuracies else None
        )
        mean_ranking_unit_accuracy = (
            float(np.mean(epoch_ranking_unit_accuracies)) if epoch_ranking_unit_accuracies else None
        )
        mean_full_vocab_ranking_loss = (
            float(np.mean(epoch_full_vocab_ranking_losses))
            if epoch_full_vocab_ranking_losses
            else None
        )
        mean_weighted_full_vocab_ranking_loss = (
            float(np.mean(epoch_weighted_full_vocab_ranking_losses))
            if epoch_weighted_full_vocab_ranking_losses
            else None
        )
        mean_full_vocab_ranking_margin = (
            float(np.mean(epoch_full_vocab_ranking_margins))
            if epoch_full_vocab_ranking_margins
            else None
        )
        minimum_full_vocab_ranking_margin = (
            float(np.min(epoch_full_vocab_ranking_min_margins))
            if epoch_full_vocab_ranking_min_margins
            else None
        )
        mean_full_vocab_side_accuracy = (
            float(np.mean(epoch_full_vocab_side_accuracies))
            if epoch_full_vocab_side_accuracies
            else None
        )
        mean_full_vocab_unit_accuracy = (
            float(np.mean(epoch_full_vocab_unit_accuracies))
            if epoch_full_vocab_unit_accuracies
            else None
        )
        mean_pair_scene_token_gradient_norm = (
            float(np.mean(epoch_pair_scene_token_gradient_norms))
            if epoch_pair_scene_token_gradient_norms
            else None
        )
        mean_spatial_answer_loss = (
            float(np.mean(epoch_spatial_answer_losses)) if epoch_spatial_answer_losses else None
        )
        mean_spatial_answer_own_similarity = (
            float(np.mean(epoch_spatial_answer_own_similarities))
            if epoch_spatial_answer_own_similarities
            else None
        )
        mean_spatial_answer_alternate_similarity = (
            float(np.mean(epoch_spatial_answer_alternate_similarities))
            if epoch_spatial_answer_alternate_similarities
            else None
        )
        mean_spatial_answer_margin = (
            float(np.mean(epoch_spatial_answer_margins)) if epoch_spatial_answer_margins else None
        )
        mean_spatial_relation_loss = (
            float(np.mean(epoch_spatial_relation_losses)) if epoch_spatial_relation_losses else None
        )
        mean_spatial_relation_margin = (
            float(np.mean(epoch_spatial_relation_margins))
            if epoch_spatial_relation_margins
            else None
        )
        minimum_spatial_relation_margin = (
            float(np.min(epoch_spatial_relation_minimum_margins))
            if epoch_spatial_relation_minimum_margins
            else None
        )
        should_evaluate_pair_gate = pair_curriculum.gate_enabled and (
            epoch % pair_curriculum.gate_every_epochs == 0 or epoch == epochs
        )
        pair_gate = (
            evaluate_pair_candidate_gate(
                pair_units,
                maps=maps,
                config=config,
                language=language,
                scene_model=scene_model,
                global_scene_residual=global_scene_residual,
                signed_x_scene_residual=signed_x_scene_residual,
                dense_aligner=dense_aligner,
                composer=composer,
                grounding=grounding,
                units_per_batch=pair_curriculum.units_per_batch,
                ranking_margin=pair_curriculum.ranking_margin,
                ranking_mode=pair_curriculum.ranking_mode,
                changed_unit_accuracy_threshold=(pair_curriculum.changed_unit_accuracy_threshold),
                prediction_flip_threshold=pair_curriculum.prediction_flip_threshold,
                wrong_prefix_flip_threshold=pair_curriculum.wrong_prefix_flip_threshold,
                first_answer_token_top1_accuracy_threshold=(
                    pair_curriculum.first_answer_token_top1_accuracy_threshold
                ),
                lora_installation=lora_installation,
            )
            if should_evaluate_pair_gate
            else None
        )
        should_validate = bool(validation_by_scene) and (
            epoch % validation_interval == 0 or epoch == epochs
        )
        validation_metrics = (
            validation_loss(
                validation_by_scene,
                config=config,
                language=language,
                scene_model=scene_model,
                global_scene_residual=global_scene_residual,
                signed_x_scene_residual=signed_x_scene_residual,
                dense_aligner=dense_aligner,
                composer=composer,
                grounding=grounding,
                semantic_dim=semantic_dim,
                batch_size=batch_size,
                lora_installation=lora_installation,
            )
            if should_validate
            else None
        )
        validation_value = None if validation_metrics is None else validation_metrics["loss"]
        if pair_curriculum.pair_only and pair_gate is not None:
            monitor_name = pair_gate_monitor_name
            monitor_value = pair_gate_monitor_value(
                pair_gate,
                full_vocab_gate=(
                    pair_curriculum.first_answer_token_top1_accuracy_threshold is not None
                ),
            )
        else:
            monitor_name = "validation_loss" if validation_by_scene else "train_loss"
            monitor_value = validation_value if validation_by_scene else mean_loss
        improved = False
        if monitor_value is not None:
            if pair_curriculum.pair_only and pair_gate is not None:
                improved = pair_gate_checkpoint_improved(
                    monitor_value=monitor_value,
                    best_monitor_value=best_monitor_loss,
                    min_delta=min_delta,
                    gate_passed=bool(pair_gate["passed"]),
                    best_gate_passed=best_pair_gate_passed,
                )
            else:
                improved = monitor_value < best_monitor_loss - min_delta
            if improved:
                best_monitor_loss = monitor_value
                best_epoch = epoch
                if pair_curriculum.pair_only and pair_gate is not None:
                    best_pair_gate_passed = bool(pair_gate["passed"])
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
        history.append(
            {
                "epoch": epoch,
                "train_loss": mean_loss,
                "train_language_loss": mean_language_loss,
                "train_weighted_language_loss": mean_weighted_language_loss,
                "train_grounding_loss": mean_grounding_loss,
                "pair_ranking_loss": mean_ranking_loss,
                "pair_weighted_ranking_loss": mean_weighted_ranking_loss,
                "pair_ranking_mode": pair_curriculum.ranking_mode,
                "pair_mean_ranking_margin": mean_ranking_margin,
                "pair_minimum_ranking_margin": minimum_ranking_margin,
                "pair_mean_correct_vs_swapped_nll_margin": (
                    mean_ranking_margin if pair_curriculum.ranking_mode == "nll" else None
                ),
                "pair_minimum_correct_vs_swapped_nll_margin": (
                    minimum_ranking_margin if pair_curriculum.ranking_mode == "nll" else None
                ),
                "pair_mean_own_vs_alternate_candidate_logit_margin": (
                    mean_ranking_margin
                    if pair_curriculum.ranking_mode == "candidate_logit"
                    else None
                ),
                "pair_minimum_own_vs_alternate_candidate_logit_margin": (
                    minimum_ranking_margin
                    if pair_curriculum.ranking_mode == "candidate_logit"
                    else None
                ),
                "pair_side_accuracy": mean_ranking_side_accuracy,
                "pair_changed_unit_accuracy": mean_ranking_unit_accuracy,
                "pair_full_vocab_ranking_loss": mean_full_vocab_ranking_loss,
                "pair_full_vocab_weighted_ranking_loss": (mean_weighted_full_vocab_ranking_loss),
                "pair_full_vocab_mean_margin": mean_full_vocab_ranking_margin,
                "pair_full_vocab_minimum_margin": minimum_full_vocab_ranking_margin,
                "pair_full_vocab_top1_side_accuracy": mean_full_vocab_side_accuracy,
                "pair_full_vocab_top1_unit_accuracy": mean_full_vocab_unit_accuracy,
                "pair_mean_scene_token_gradient_norm": (mean_pair_scene_token_gradient_norm),
                "lora_mean_optimizer_step_gradient_norm": (
                    float(np.mean(epoch_lora_gradient_norms)) if epoch_lora_gradient_norms else None
                ),
                "lora_max_optimizer_step_gradient_norm": (
                    max(epoch_lora_gradient_norms) if epoch_lora_gradient_norms else None
                ),
                "dense_alignment_mean_optimizer_step_gradient_norm": (
                    float(np.mean(epoch_dense_alignment_gradient_norms))
                    if epoch_dense_alignment_gradient_norms
                    else None
                ),
                "dense_alignment_max_optimizer_step_gradient_norm": (
                    max(epoch_dense_alignment_gradient_norms)
                    if epoch_dense_alignment_gradient_norms
                    else None
                ),
                "spatial_answer_contrastive_loss": mean_spatial_answer_loss,
                "spatial_answer_mean_own_similarity": (mean_spatial_answer_own_similarity),
                "spatial_answer_mean_alternate_similarity": (
                    mean_spatial_answer_alternate_similarity
                ),
                "spatial_answer_mean_achieved_margin": mean_spatial_answer_margin,
                "spatial_answer_configured_margin": (
                    spatial_answer_contrastive["margin"]
                    if spatial_answer_contrastive["weight"] > 0
                    else None
                ),
                "spatial_answer_eligible_unit_observations": int(
                    sum(epoch_spatial_answer_eligible_units)
                ),
                "spatial_answer_selected_unique_targets": int(
                    spatial_answer_targets["unique_target_count"]
                ),
                "spatial_relation_contrastive_loss": mean_spatial_relation_loss,
                "spatial_relation_mean_achieved_margin": mean_spatial_relation_margin,
                "spatial_relation_minimum_achieved_margin": (minimum_spatial_relation_margin),
                "spatial_relation_configured_margin": (
                    spatial_relation_contrastive["margin"]
                    if spatial_relation_contrastive["weight"] > 0
                    else None
                ),
                "spatial_relation_temperature": (
                    spatial_relation_contrastive["temperature"]
                    if spatial_relation_contrastive["weight"] > 0
                    else None
                ),
                "spatial_relation_eligible_unit_observations": int(
                    sum(epoch_spatial_relation_eligible_units)
                ),
                "spatial_relation_selected_unique_ordered_regions": int(
                    spatial_relation_targets["unique_ordered_region_count"]
                ),
                "pair_batch_count": pair_batch_count,
                "pair_batch_fraction": actual_pair_batch_fraction,
                "pair_candidate_gate": pair_gate,
                "latent_diversity_loss": mean_diversity_loss,
                "latent_mean_off_diagonal_cosine": mean_diversity_cosine,
                "latent_mean_max_off_diagonal_cosine": mean_diversity_max_cosine,
                "paired_scene_separation_loss": mean_pair_loss,
                "paired_scene_mean_aligned_cosine": mean_pair_cosine,
                "paired_scene_mean_cosine_distance": mean_pair_distance,
                "paired_scene_pairs_evaluated": len(epoch_pair_losses),
                "validation_loss": validation_value,
                "validation_language_loss": (
                    None if validation_metrics is None else validation_metrics["language_loss"]
                ),
                "validation_grounding_loss": (
                    None if validation_metrics is None else validation_metrics["grounding_loss"]
                ),
            }
        )
        metadata = {
            "schema_version": 3,
            "epoch": epoch,
            "train_loss": mean_loss,
            "validation_loss": validation_value,
            "monitor_name": monitor_name,
            "best_monitor_loss": best_monitor_loss,
            "best_epoch": best_epoch,
            "best_pair_gate_passed": best_pair_gate_passed,
            "epochs_without_improvement": epochs_without_improvement,
            "global_step": global_step,
            "optimizer_step": optimizer_step,
            "history": history,
            "semantic_dim": semantic_dim,
            "language_hidden_dim": language.hidden_size,
            "language_model_id": config["language"]["model_id"],
            "language_revision": config["language"]["revision"],
            "language_backend": language.backend_name,
            "language_decoder_gradient_checkpointing": (language_decoder_gradient_checkpointing),
            "scene_prefix_after_bos": scene_prefix_after_bos,
            "scene_boundary_mode": scene_boundary_mode,
            "gemma4_native_image_contract": loaded_native_boundary_contract,
            "source_provenance": source_provenance,
            "initialization_provenance": initialization_provenance,
            "gradient_accumulation": gradient_accumulation,
            "pair_gate_policy": {
                "stop_when_passed": pair_curriculum.stop_when_gate_passes,
                "first_answer_token_top1_accuracy_threshold": (
                    pair_curriculum.first_answer_token_top1_accuracy_threshold
                ),
            },
            "config_hash": config_hash(config),
            "scene_ids": sorted(by_scene),
            "scene_latents": int(config["scene_encoder"]["global_latents"]),
            "scene_model_dim": int(config["scene_encoder"]["model_dim"]),
            "scene_encoder_architecture_version": config["scene_encoder"].get(
                "architecture_version", "legacy_perceiver_v1"
            ),
            "global_scene_residual": residual_settings.contract(),
            "global_scene_residual_parameter_count": (
                0
                if global_scene_residual is None
                else sum(parameter.numel() for parameter in global_scene_residual.parameters())
            ),
            "global_scene_residual_initial_state_sha256": (observed_initial_residual_sha256),
            "global_scene_residual_state_sha256": (
                None
                if global_scene_residual is None
                else module_collection_state_sha256(
                    {"global_scene_residual": global_scene_residual}
                )
            ),
            "global_scene_residual_zero_output_equivalence": zero_output_equivalence,
            "signed_x_scene_residual": signed_x_residual_settings.contract(),
            "signed_x_scene_residual_parameter_count": (
                0 if signed_x_scene_residual is None else signed_x_scene_residual.parameter_count
            ),
            "signed_x_scene_residual_initial_state_sha256": (observed_initial_signed_x_sha256),
            "signed_x_scene_residual_state_sha256": (
                None
                if signed_x_scene_residual is None
                else module_collection_state_sha256(
                    {"signed_x_scene_residual": signed_x_scene_residual}
                )
            ),
            "signed_x_scene_residual_zero_output_equivalence": (signed_x_zero_output_equivalence),
            "dense_alignment": dense_settings.contract(),
            "dense_alignment_parameter_count": (
                0 if dense_aligner is None else dense_aligner.parameter_count
            ),
            "dense_alignment_initial_state_sha256": (
                observed_initial_dense_alignment_sha256
            ),
            "dense_alignment_state_sha256": (
                None if dense_aligner is None else dense_aligner.state_sha256()
            ),
            "dense_alignment_zero_output_equivalence": (
                dense_alignment_zero_output_equivalence
            ),
            "dense_alignment_calibration": dense_alignment_calibration,
            "dense_alignment_optimizer": (
                {
                    "name": "AdamW",
                    "learning_rate": float(
                        config["training"]["dense_alignment_learning_rate"]
                    ),
                    "weight_decay": float(
                        config["training"]["dense_alignment_weight_decay"]
                    ),
                }
                if train_dense_alignment_only
                else None
            ),
            "question_dependent_scene_processing": False,
            "all_voxels_transformed": dense_aligner is not None,
            "input_voxel_size_m": config["scene_encoder"].get("input_voxel_size_m"),
            **token_mixing,
            "output_namespace": output_namespace,
            "max_questions_per_scene": per_scene_cap,
            "train_scene_ids": sorted(by_scene),
            "validation_scene_ids": sorted(validation_by_scene),
            "test_scene_ids": split_ids["test"],
            "anti_collapse_objectives": anti_collapse,
            "spatial_answer_contrastive": spatial_answer_contrastive,
            "spatial_answer_target_audit": spatial_answer_targets,
            "spatial_answer_warmup": spatial_answer_warmup,
            "spatial_answer_warmup_target_audit": spatial_answer_warmup_targets,
            "spatial_answer_warmup_metrics": spatial_answer_warmup_metrics,
            "spatial_relation_contrastive": spatial_relation_contrastive,
            "spatial_relation_target_audit": spatial_relation_targets,
            "spatial_relation_warmup": spatial_relation_warmup,
            "spatial_relation_warmup_target_audit": spatial_relation_warmup_targets,
            "spatial_relation_warmup_metrics": spatial_relation_warmup_metrics,
            "pair_curriculum": {
                "enabled": pair_curriculum.enabled,
                "pair_only": pair_curriculum.pair_only,
                "pair_only_scene_ids": list(pair_curriculum.pair_only_scene_ids),
                "max_units_per_pair": pair_curriculum.max_units_per_pair,
                "ranking_weight": pair_curriculum.ranking_weight,
                "ranking_margin": pair_curriculum.ranking_margin,
                "ranking_mode": pair_curriculum.ranking_mode,
                "full_vocab_ranking_weight": pair_curriculum.full_vocab_ranking_weight,
                "full_vocab_ranking_margin": pair_curriculum.full_vocab_ranking_margin,
                "batch_fraction": pair_curriculum.batch_fraction,
                "units_per_batch": pair_curriculum.units_per_batch,
                "steps_per_epoch": pair_curriculum.steps_per_epoch,
                "gate_enabled": pair_curriculum.gate_enabled,
                "objective_policy": resolved_pair_objective_contract,
                "objective_policy_coverage": pair_objective_coverage,
            },
            "pair_candidate_gate": pair_gate,
            "counterfactual_pair_unit_count": len(pair_units),
            "counterfactual_pair_unit_selection_sha256": pair_unit_selection_sha256,
            "training_counterfactual_pair_count": len(training_pairs),
            "training_counterfactual_pair_membership_sha256": (pair_membership_sha256),
        }
        if v18_stage_execution is not None:
            metadata["v18_stage_execution"] = v18_stage_execution
        current_scene_hash = module_collection_state_sha256(scene_state_modules)
        current_residual_hash = (
            None
            if global_scene_residual is None
            else module_collection_state_sha256({"global_scene_residual": global_scene_residual})
        )
        if current_residual_hash != metadata["global_scene_residual_state_sha256"]:
            raise RuntimeError(
                "Global scene residual state changed during checkpoint metadata save"
            )
        current_signed_x_hash = (
            None
            if signed_x_scene_residual is None
            else module_collection_state_sha256(
                {"signed_x_scene_residual": signed_x_scene_residual}
            )
        )
        if current_signed_x_hash != metadata["signed_x_scene_residual_state_sha256"]:
            raise RuntimeError(
                "Signed-X scene residual state changed during checkpoint metadata save"
            )
        current_dense_hash = None if dense_aligner is None else dense_aligner.state_sha256()
        if current_dense_hash != metadata["dense_alignment_state_sha256"]:
            raise RuntimeError(
                "Dense-alignment state changed during checkpoint metadata save"
            )
        if global_scene_residual is not None:
            validate_global_scene_residual_state(
                global_scene_residual,
                expected_parameter_count=declared_residual_parameter_count,
                context="checkpoint save",
            )
        if signed_x_scene_residual is not None:
            validate_signed_x_scene_residual_state(
                signed_x_scene_residual,
                expected_parameter_count=declared_signed_x_parameter_count,
                context="checkpoint save",
            )
        if dense_aligner is not None:
            validate_dense_alignment_state(
                dense_aligner,
                expected_parameter_count=declared_dense_parameter_count,
                context="checkpoint save",
            )
        current_frozen_bank_hashes = (
            {}
            if lora_installation is None
            else {
                bank.settings.name: bank.installation.state_sha256()
                for bank in lora_installation.banks
                if not bank.settings.trainable
            }
        )
        if freeze_scene_adapter and current_scene_hash != frozen_scene_state_sha256:
            raise RuntimeError(
                "Frozen scene adapter changed before checkpoint save: "
                f"expected={frozen_scene_state_sha256} observed={current_scene_hash}"
            )
        if (
            train_signed_x_scene_residual_only
            or train_lora_with_frozen_scene_residual_stack
            or train_dense_alignment_only
        ) and current_residual_hash != frozen_global_scene_residual_state_sha256:
            raise RuntimeError(
                "Frozen global scene residual changed during signed-X training: "
                f"expected={frozen_global_scene_residual_state_sha256} "
                f"observed={current_residual_hash}"
            )
        if (
            (train_lora_with_frozen_scene_residual_stack or train_dense_alignment_only)
            and current_signed_x_hash != frozen_signed_x_scene_residual_state_sha256
        ):
            raise RuntimeError(
                "Frozen signed-X residual changed during language-LoRA training: "
                f"expected={frozen_signed_x_scene_residual_state_sha256} "
                f"observed={current_signed_x_hash}"
            )
        if current_frozen_bank_hashes != frozen_lora_bank_state_sha256:
            raise RuntimeError(
                "Frozen LoRA bank changed before checkpoint save: "
                f"expected={frozen_lora_bank_state_sha256} "
                f"observed={current_frozen_bank_hashes}"
            )
        if freeze_scene_adapter or not configured_lora.legacy_single_bank:
            metadata.update(
                {
                    "freeze_scene_adapter": freeze_scene_adapter,
                    "train_global_scene_residual_only": train_global_scene_residual_only,
                    "train_signed_x_scene_residual_only": (train_signed_x_scene_residual_only),
                    "train_lora_with_frozen_scene_residual_stack": (
                        train_lora_with_frozen_scene_residual_stack
                    ),
                    "train_dense_alignment_only": train_dense_alignment_only,
                    "frozen_scene_state_sha256": frozen_scene_state_sha256,
                    "frozen_global_scene_residual_state_sha256": (
                        frozen_global_scene_residual_state_sha256
                    ),
                    "frozen_signed_x_scene_residual_state_sha256": (
                        frozen_signed_x_scene_residual_state_sha256
                    ),
                    "frozen_lora_bank_state_sha256": frozen_lora_bank_state_sha256,
                    "initialize_named_lora_freeze_for_dense_alignment_transition": (
                        initialize_named_lora_freeze_for_dense_alignment_transition
                    ),
                }
            )
        if (
            initialize_expected_adapter_sha256 is not None
            or initialize_expected_metadata_sha256 is not None
            or initialize_expected_scene_state_sha256 is not None
            or initialize_expected_global_scene_residual_state_sha256 is not None
            or initialize_expected_signed_x_scene_residual_state_sha256 is not None
            or initialize_expected_dense_alignment_state_sha256 is not None
        ):
            metadata.update(
                {
                    "initialize_expected_adapter_sha256": (initialize_expected_adapter_sha256),
                    "initialize_expected_metadata_sha256": (initialize_expected_metadata_sha256),
                    "initialize_expected_scene_state_sha256": (
                        initialize_expected_scene_state_sha256
                    ),
                    "initialize_expected_global_scene_residual_state_sha256": (
                        initialize_expected_global_scene_residual_state_sha256
                    ),
                    "initialize_expected_signed_x_scene_residual_state_sha256": (
                        initialize_expected_signed_x_scene_residual_state_sha256
                    ),
                    "initialize_expected_dense_alignment_state_sha256": (
                        initialize_expected_dense_alignment_state_sha256
                    ),
                    "initialize_source_residual_into_frozen_base": (
                        initialize_source_residual_into_frozen_base
                    ),
                    "initialize_named_lora_freeze_for_dense_alignment_transition": (
                        initialize_named_lora_freeze_for_dense_alignment_transition
                    ),
                }
            )
        if lora_installation is not None:
            lora_installation.validate_state()
            metadata.update({"lora": configured_lora_checkpoint_contract})
            metadata.update(lora_installation.checkpoint_metadata())
        checkpoint = save_adapter_checkpoint(
            output_root / f"epoch_{epoch:03d}",
            checkpoint_modules,
            metadata,
        )
        save_optimizer_checkpoint(checkpoint, optimizer)
        if improved:
            best = output_root / "best"
            if best.exists():
                shutil.rmtree(best)
            shutil.copytree(checkpoint, best)
        print(
            json.dumps(
                {
                    "phase": "validation" if validation_metrics is not None else "epoch_complete",
                    "epoch": epoch,
                    "train_loss": mean_loss,
                    "validation": validation_metrics,
                    "latent_diversity_loss": mean_diversity_loss,
                    "latent_mean_off_diagonal_cosine": mean_diversity_cosine,
                    "paired_scene_separation_loss": mean_pair_loss,
                    "paired_scene_mean_cosine_distance": mean_pair_distance,
                    "paired_scene_pairs_evaluated": len(epoch_pair_losses),
                    "pair_ranking_loss": mean_ranking_loss,
                    "pair_weighted_ranking_loss": mean_weighted_ranking_loss,
                    "pair_ranking_mode": pair_curriculum.ranking_mode,
                    "pair_mean_ranking_margin": mean_ranking_margin,
                    "pair_minimum_ranking_margin": minimum_ranking_margin,
                    "pair_mean_correct_vs_swapped_nll_margin": (
                        mean_ranking_margin if pair_curriculum.ranking_mode == "nll" else None
                    ),
                    "pair_minimum_correct_vs_swapped_nll_margin": (
                        minimum_ranking_margin if pair_curriculum.ranking_mode == "nll" else None
                    ),
                    "pair_mean_own_vs_alternate_candidate_logit_margin": (
                        mean_ranking_margin
                        if pair_curriculum.ranking_mode == "candidate_logit"
                        else None
                    ),
                    "pair_minimum_own_vs_alternate_candidate_logit_margin": (
                        minimum_ranking_margin
                        if pair_curriculum.ranking_mode == "candidate_logit"
                        else None
                    ),
                    "pair_side_accuracy": mean_ranking_side_accuracy,
                    "pair_changed_unit_accuracy": mean_ranking_unit_accuracy,
                    "pair_full_vocab_ranking_loss": mean_full_vocab_ranking_loss,
                    "pair_full_vocab_weighted_ranking_loss": (
                        mean_weighted_full_vocab_ranking_loss
                    ),
                    "pair_full_vocab_mean_margin": mean_full_vocab_ranking_margin,
                    "pair_full_vocab_minimum_margin": minimum_full_vocab_ranking_margin,
                    "pair_full_vocab_top1_side_accuracy": mean_full_vocab_side_accuracy,
                    "pair_full_vocab_top1_unit_accuracy": mean_full_vocab_unit_accuracy,
                    "pair_mean_scene_token_gradient_norm": (mean_pair_scene_token_gradient_norm),
                    "spatial_answer_contrastive_loss": mean_spatial_answer_loss,
                    "spatial_answer_mean_own_similarity": (mean_spatial_answer_own_similarity),
                    "spatial_answer_mean_alternate_similarity": (
                        mean_spatial_answer_alternate_similarity
                    ),
                    "spatial_answer_mean_achieved_margin": mean_spatial_answer_margin,
                    "spatial_answer_configured_margin": (
                        spatial_answer_contrastive["margin"]
                        if spatial_answer_contrastive["weight"] > 0
                        else None
                    ),
                    "spatial_answer_selected_unique_targets": int(
                        spatial_answer_targets["unique_target_count"]
                    ),
                    "spatial_relation_contrastive_loss": mean_spatial_relation_loss,
                    "spatial_relation_mean_achieved_margin": (mean_spatial_relation_margin),
                    "spatial_relation_minimum_achieved_margin": (minimum_spatial_relation_margin),
                    "spatial_relation_configured_margin": (
                        spatial_relation_contrastive["margin"]
                        if spatial_relation_contrastive["weight"] > 0
                        else None
                    ),
                    "spatial_relation_temperature": (
                        spatial_relation_contrastive["temperature"]
                        if spatial_relation_contrastive["weight"] > 0
                        else None
                    ),
                    "spatial_relation_selected_unique_ordered_regions": int(
                        spatial_relation_targets["unique_ordered_region_count"]
                    ),
                    "pair_batch_count": pair_batch_count,
                    "pair_batch_fraction": actual_pair_batch_fraction,
                    "pair_candidate_gate": pair_gate,
                    "monitor": monitor_name,
                    "best_epoch": best_epoch,
                    "best_monitor_loss": best_monitor_loss,
                    "epochs_without_improvement": epochs_without_improvement,
                }
            ),
            flush=True,
        )
        if should_stop_after_pair_gate(pair_curriculum.stop_when_gate_passes, pair_gate):
            stopped_early = True
            break
        if (
            monitor_value is not None
            and early_stopping_patience > 0
            and epochs_without_improvement >= early_stopping_patience
        ):
            stopped_early = True
            break

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    completed_epochs = int(history[-1]["epoch"]) if history else start_epoch - 1
    summary = {
        "target_epochs": epochs,
        "epochs": completed_epochs,
        "steps": global_step,
        "optimizer_steps": optimizer_step,
        "best_train_loss": min(item["train_loss"] for item in history),
        "best_validation_loss": (
            min(
                item["validation_loss"]
                for item in history
                if item.get("validation_loss") is not None
            )
            if any(item.get("validation_loss") is not None for item in history)
            else None
        ),
        "monitor_name": (
            pair_gate_monitor_name
            if pair_curriculum.pair_only and pair_curriculum.gate_enabled
            else ("validation_loss" if validation_by_scene else "train_loss")
        ),
        "best_monitor_loss": best_monitor_loss,
        "best_epoch": best_epoch,
        "best_pair_gate_passed": best_pair_gate_passed,
        "stopped_early": stopped_early,
        "elapsed_seconds": time.perf_counter() - started,
        "history": history,
        "scene_count": len(by_scene),
        "question_count": len(records),
        "validation_scene_count": len(validation_by_scene),
        "validation_question_count": len(validation_records),
        "train_scene_ids": sorted(by_scene),
        "validation_scene_ids": sorted(validation_by_scene),
        "test_scene_ids": split_ids["test"],
        "device": str(language.device),
        "language_decoder_gradient_checkpointing": (language_decoder_gradient_checkpointing),
        "scene_prefix_after_bos": scene_prefix_after_bos,
        "scene_boundary_mode": scene_boundary_mode,
        "gemma4_native_image_contract": loaded_native_boundary_contract,
        "source_provenance": source_provenance,
        "initialization_provenance": initialization_provenance,
        "initialize_expected_adapter_sha256": initialize_expected_adapter_sha256,
        "initialize_expected_metadata_sha256": initialize_expected_metadata_sha256,
        "initialize_expected_scene_state_sha256": initialize_expected_scene_state_sha256,
        "initialize_expected_global_scene_residual_state_sha256": (
            initialize_expected_global_scene_residual_state_sha256
        ),
        "initialize_expected_signed_x_scene_residual_state_sha256": (
            initialize_expected_signed_x_scene_residual_state_sha256
        ),
        "initialize_source_residual_into_frozen_base": (
            initialize_source_residual_into_frozen_base
        ),
        "freeze_scene_adapter": freeze_scene_adapter,
        "train_global_scene_residual_only": train_global_scene_residual_only,
        "train_signed_x_scene_residual_only": train_signed_x_scene_residual_only,
        "train_lora_with_frozen_scene_residual_stack": (
            train_lora_with_frozen_scene_residual_stack
        ),
        "global_scene_residual": residual_settings.contract(),
        "global_scene_residual_parameter_count": (
            0
            if global_scene_residual is None
            else sum(parameter.numel() for parameter in global_scene_residual.parameters())
        ),
        "global_scene_residual_initial_state_sha256": observed_initial_residual_sha256,
        "global_scene_residual_state_sha256": (
            None
            if global_scene_residual is None
            else module_collection_state_sha256({"global_scene_residual": global_scene_residual})
        ),
        "global_scene_residual_zero_output_equivalence": zero_output_equivalence,
        "signed_x_scene_residual": signed_x_residual_settings.contract(),
        "signed_x_scene_residual_parameter_count": (
            0 if signed_x_scene_residual is None else signed_x_scene_residual.parameter_count
        ),
        "signed_x_scene_residual_initial_state_sha256": observed_initial_signed_x_sha256,
        "signed_x_scene_residual_state_sha256": (
            None
            if signed_x_scene_residual is None
            else module_collection_state_sha256(
                {"signed_x_scene_residual": signed_x_scene_residual}
            )
        ),
        "signed_x_scene_residual_zero_output_equivalence": (signed_x_zero_output_equivalence),
        "question_dependent_scene_processing": False,
        "frozen_scene_state_sha256": frozen_scene_state_sha256,
        "frozen_global_scene_residual_state_sha256": (frozen_global_scene_residual_state_sha256),
        "frozen_signed_x_scene_residual_state_sha256": (
            frozen_signed_x_scene_residual_state_sha256
        ),
        "frozen_lora_bank_state_sha256": frozen_lora_bank_state_sha256,
        "gradient_accumulation": gradient_accumulation,
        "semantic_dim": semantic_dim,
        "raw_voxel_count": max(data.source_voxel_count for data in maps.values()),
        "tokenizer_input_voxel_count": max(data.voxel_count for data in maps.values()),
        "scene_latents": int(config["scene_encoder"]["global_latents"]),
        "output_namespace": output_namespace,
        "max_questions_per_scene": per_scene_cap,
        "selection_report": str(selection_path.relative_to(PROJECT_ROOT)),
        "selection": selection_report,
        "anti_collapse_objectives": anti_collapse,
        "spatial_answer_contrastive": spatial_answer_contrastive,
        "spatial_answer_target_audit": spatial_answer_targets,
        "spatial_answer_warmup": spatial_answer_warmup,
        "spatial_answer_warmup_target_audit": spatial_answer_warmup_targets,
        "spatial_answer_warmup_metrics": spatial_answer_warmup_metrics,
        "spatial_relation_contrastive": spatial_relation_contrastive,
        "spatial_relation_target_audit": spatial_relation_targets,
        "spatial_relation_warmup": spatial_relation_warmup,
        "spatial_relation_warmup_target_audit": spatial_relation_warmup_targets,
        "spatial_relation_warmup_metrics": spatial_relation_warmup_metrics,
        "pair_curriculum": {
            "enabled": pair_curriculum.enabled,
            "pair_only": pair_curriculum.pair_only,
            "pair_only_scene_ids": list(pair_curriculum.pair_only_scene_ids),
            "max_units_per_pair": pair_curriculum.max_units_per_pair,
            "ranking_weight": pair_curriculum.ranking_weight,
            "ranking_margin": pair_curriculum.ranking_margin,
            "ranking_mode": pair_curriculum.ranking_mode,
            "full_vocab_ranking_weight": pair_curriculum.full_vocab_ranking_weight,
            "full_vocab_ranking_margin": pair_curriculum.full_vocab_ranking_margin,
            "batch_fraction": pair_curriculum.batch_fraction,
            "units_per_batch": pair_curriculum.units_per_batch,
            "steps_per_epoch": pair_curriculum.steps_per_epoch,
            "gate_enabled": pair_curriculum.gate_enabled,
            "gate_every_epochs": pair_curriculum.gate_every_epochs,
            "gate_stop_when_passed": pair_curriculum.stop_when_gate_passes,
            "objective_policy": resolved_pair_objective_contract,
            "objective_policy_coverage": pair_objective_coverage,
            "gate_thresholds": {
                "changed_unit_accuracy": (pair_curriculum.changed_unit_accuracy_threshold),
                "prediction_flip_rate": pair_curriculum.prediction_flip_threshold,
                "wrong_prefix_flip_rate": pair_curriculum.wrong_prefix_flip_threshold,
                "first_answer_token_top1_accuracy": (
                    pair_curriculum.first_answer_token_top1_accuracy_threshold
                ),
            },
        },
        "counterfactual_pair_unit_count": len(pair_units),
        "counterfactual_pair_unit_selection_sha256": pair_unit_selection_sha256,
        "pair_candidate_gate": (history[-1].get("pair_candidate_gate") if history else None),
        "training_counterfactual_pair_count": len(training_pairs),
        "training_counterfactual_pair_membership_sha256": (pair_membership_sha256),
    }
    if v18_stage_execution is not None:
        summary["v18_stage_execution"] = v18_stage_execution
    if lora_installation is not None:
        summary.update(
            {
                "lora": configured_lora_checkpoint_contract,
                "lora_optimizer": (
                    None
                    if configured_lora_optimizer is None
                    else configured_lora_optimizer.contract()
                ),
                **lora_installation.checkpoint_metadata(),
            }
        )
    metrics_path.write_text(json.dumps(summary, indent=2) + "\n")
    loss_figure_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.plot(
        [item["epoch"] for item in history],
        [item["train_loss"] for item in history],
        marker="o",
        label="train",
    )
    validation_history = [item for item in history if item.get("validation_loss") is not None]
    if validation_history:
        plt.plot(
            [item["epoch"] for item in validation_history],
            [item["validation_loss"] for item in validation_history],
            marker="o",
            label="validation",
        )
        plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Teacher-forced loss")
    plt.title("Continuous scene adapter training")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(loss_figure_path, dpi=160)
    plt.close()
    print(json.dumps({"phase": "training_complete", **summary}, default=str))


if __name__ == "__main__":
    main()
