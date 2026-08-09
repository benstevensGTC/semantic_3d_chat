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
from collections.abc import Sequence
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
from semantic_3d_chat.language.local_lm import (
    load_local_language_model,
    prompt_token_ids,
    question_token_ids,
)
from semantic_3d_chat.language.lora import (
    LoRAInstallation,
    LoRAOptimizerSettings,
    install_lora_adapters,
    lora_checkpoint_contract,
    lora_checkpoint_contract_mismatch,
    lora_optimizer_settings,
    lora_settings,
    validate_lora_checkpoint_state,
)
from semantic_3d_chat.language.prefix_injection import (
    ContinuousPrefixComposer,
    PrefixBatch,
    native_gemma4_image_contract_setting,
    scene_boundary_contract_mismatch,
    scene_boundary_mode_setting,
    scene_prefix_after_bos_contract_mismatch,
    scene_prefix_after_bos_setting,
    stack_prefix_batches,
)
from semantic_3d_chat.scene_encoder.map_io import MapTensorData, load_map_tensors
from semantic_3d_chat.scene_encoder.projector import SceneTokenizer
from semantic_3d_chat.training.checkpointing import (
    load_adapter_checkpoint,
    load_optimizer_checkpoint,
    save_adapter_checkpoint,
    save_optimizer_checkpoint,
)
from semantic_3d_chat.training.losses import (
    QuestionGroundingHead,
    latent_diversity_loss,
    nearest_spatial_anchor_indices,
    normalize_xyz,
    paired_scene_separation_loss,
    spatial_scene_answer_contrastive_loss,
)
from semantic_3d_chat.training.pair_curriculum import (
    CounterfactualPairUnit,
    build_epoch_curriculum,
    build_exact_question_pair_units,
    candidate_logit_margins,
    cap_pair_units_per_pair,
    differing_answer_token_masks,
    first_answer_token_full_vocab_margins,
    pair_curriculum_settings,
    pair_gate_metrics,
    pair_ranking_hinge,
    ranking_margin_hinge,
    restrict_labels_to_answer_mask,
    select_pair_only_records,
    single_differing_answer_token,
    token_normalized_nll,
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
    lora_installation: LoRAInstallation | None,
    configured_lora_optimizer: LoRAOptimizerSettings | None,
) -> tuple[torch.optim.AdamW, list[torch.nn.Parameter]]:
    """Build the legacy single group or strict v8 scene/LoRA groups."""

    scene_parameters = list(scene_parameters)
    if lora_installation is None:
        optimizer = torch.optim.AdamW(
            scene_parameters,
            lr=float(config["training"]["learning_rate"]),
            weight_decay=float(config["training"]["weight_decay"]),
        )
        return optimizer, scene_parameters
    if configured_lora_optimizer is None:
        raise ValueError("Enabled LoRA requires explicit optimizer settings")
    lora_parameters = lora_installation.parameters()
    optimizer = torch.optim.AdamW(
        [
            {
                "name": "scene_adapter",
                "params": scene_parameters,
                "lr": float(config["training"]["learning_rate"]),
                "weight_decay": float(config["training"]["weight_decay"]),
            },
            {
                "name": "language_lora",
                "params": lora_parameters,
                "lr": configured_lora_optimizer.learning_rate,
                "weight_decay": configured_lora_optimizer.weight_decay,
            },
        ]
    )
    return optimizer, scene_parameters + lora_parameters


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


def map_forward(model: SceneTokenizer, data: MapTensorData):
    return model(
        data.semantic,
        data.xyz,
        data.rgb,
        data.normal,
        data.confidence,
        data.observation_count,
        data.room_min,
        data.room_max,
    )


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


def split_scene_ids(qa_root: Path, train_records: Sequence[QARecord]) -> dict[str, list[str]]:
    """Load the persisted scene-level split manifest for checkpoint provenance."""

    manifest_path = qa_root / "splits.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        raw_splits = manifest.get("splits", {})
        if isinstance(raw_splits, dict):
            result = {
                name: sorted(str(scene_id) for scene_id in raw_splits.get(name, []))
                for name in ("train", "validation", "test")
            }
            all_scenes = [scene_id for values in result.values() for scene_id in values]
            if len(all_scenes) != len(set(all_scenes)):
                raise ValueError("QA split manifest contains scene leakage")
            return result
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
    return base_loss, language_loss, grounding_loss, ranking_loss, diagnostics


def evaluate_pair_candidate_gate(
    units: Sequence[CounterfactualPairUnit],
    *,
    maps: dict[str, MapTensorData],
    config: dict,
    language,
    scene_model: SceneTokenizer,
    composer: ContinuousPrefixComposer,
    grounding: QuestionGroundingHead,
    units_per_batch: int,
    ranking_margin: float,
    ranking_mode: str,
    changed_unit_accuracy_threshold: float,
    prediction_flip_threshold: float,
    wrong_prefix_flip_threshold: float,
    first_answer_token_top1_accuracy_threshold: float | None = None,
    lora_installation: LoRAInstallation | None = None,
) -> dict[str, float | int | bool | str | None]:
    """Evaluate all training pair units with the configured deterministic ranking."""

    if not units:
        raise ValueError("Pair gate evaluation requires pair units")
    modules = (scene_model, composer, grounding)
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
    try:
        with torch.inference_mode():
            for pair_id in sorted(by_pair):
                pair_units = by_pair[pair_id]
                scene_ids = pair_units[0].scene_ids
                outputs = {
                    scene_id: map_forward(scene_model, maps[scene_id]) for scene_id in scene_ids
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
                    )
                    all_margins.append(diagnostics["margins"].detach().float().cpu())
                    full_vocab_margins = diagnostics["first_answer_token_full_vocab_margins"]
                    assert isinstance(full_vocab_margins, torch.Tensor)
                    all_full_vocab_first_token_margins.append(
                        full_vocab_margins.detach().float().cpu()
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
    return metrics


def validation_loss(
    records_by_scene: dict[str, list[QARecord]],
    *,
    config: dict,
    language,
    scene_model: SceneTokenizer,
    composer: ContinuousPrefixComposer,
    grounding: QuestionGroundingHead,
    semantic_dim: int,
    batch_size: int,
    lora_installation: LoRAInstallation | None = None,
) -> dict[str, float] | None:
    """Evaluate held-out teacher-forced loss while loading one scene map at a time."""

    if not records_by_scene:
        return None
    modules = (scene_model, composer, grounding)
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
                output = map_forward(scene_model, data)
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
    scene_prefix_after_bos = scene_prefix_after_bos_setting(config)
    scene_boundary_mode = scene_boundary_mode_setting(config)
    configured_native_boundary_contract = native_gemma4_image_contract_setting(config)
    configured_lora = lora_settings(config)
    configured_lora_optimizer = lora_optimizer_settings(config, configured_lora)
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
    pair_gate_monitor_name = (
        "pair_composite_full_vocab_gate_margin"
        if pair_curriculum.first_answer_token_top1_accuracy_threshold is not None
        else "pair_candidate_gate_hinge"
    )
    qa_root = artifact_root(config, "qa")
    qa_path = qa_root / "train.jsonl"
    dataset = SceneQADataset(qa_path)
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
    if spatial_answer_contrastive["weight"] > 0 and not pair_curriculum.enabled:
        raise ValueError("spatial_answer_contrastive_weight requires an enabled pair curriculum")
    if spatial_answer_warmup["steps"] > 0 and not pair_curriculum.enabled:
        raise ValueError("spatial_answer_warmup_steps requires an enabled pair curriculum")
    token_mixing = scene_token_mixing_settings(config)
    pair_units = build_exact_question_pair_units(records)
    if pair_curriculum.enabled and not pair_units:
        raise ValueError("The pair curriculum is enabled but selection contains no pair units")
    spatial_answer_targets = spatial_answer_target_audit(pair_units)
    spatial_answer_warmup_targets = spatial_answer_warmup_target_audit(pair_units)
    if (
        spatial_answer_warmup["steps"] > 0
        and int(spatial_answer_warmup_targets["deduplicated_unit_count"]) == 0
    ):
        raise ValueError("Enabled spatial-answer warmup has no grounded pair targets")
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
    split_ids = split_scene_ids(qa_root, records)
    validation_path = qa_root / "validation.jsonl"
    validation_records: list[QARecord] = []
    if validation_path.is_file() and not pair_curriculum.pair_only:
        validation_dataset = SceneQADataset(validation_path)
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
        "scene_token_mixing": token_mixing,
        "spatial_answer_contrastive": spatial_answer_contrastive,
        "spatial_answer_target_audit": spatial_answer_targets,
        "spatial_answer_warmup": spatial_answer_warmup,
        "spatial_answer_warmup_target_audit": spatial_answer_warmup_targets,
        "language_decoder_gradient_checkpointing": (language_decoder_gradient_checkpointing),
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
            "gate_enabled": pair_curriculum.gate_enabled,
            "gate_every_epochs": pair_curriculum.gate_every_epochs,
            "gate_stop_when_passed": pair_curriculum.stop_when_gate_passes,
            "gate_first_answer_token_top1_accuracy": (
                pair_curriculum.first_answer_token_top1_accuracy_threshold
            ),
        },
    }
    if configured_lora.enabled:
        assert configured_lora_optimizer is not None
        selection_report.update(
            {
                "lora": configured_lora.contract(),
                "lora_optimizer": configured_lora_optimizer.contract(),
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
    lora_installation = install_lora_adapters(language.model, configured_lora)
    configured_lora_checkpoint_contract = lora_checkpoint_contract(
        configured_lora,
        configured_lora_optimizer,
        0 if lora_installation is None else lora_installation.parameter_count,
    )
    if lora_installation is not None:
        print(
            json.dumps(
                {
                    "phase": "lora_installed",
                    "contract": configured_lora_checkpoint_contract,
                    "wrapped_modules": list(lora_installation.target_names),
                    "trainable_parameter_counts": lora_installation.parameter_counts,
                    "trainable_parameter_count": lora_installation.parameter_count,
                    "optimizer": configured_lora_optimizer.contract(),
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
    scene_parameters = (
        list(scene_model.parameters()) + list(composer.parameters()) + list(grounding.parameters())
    )
    optimizer, parameters = build_adapter_optimizer(
        config,
        scene_parameters,
        lora_installation,
        configured_lora_optimizer,
    )
    checkpoint_modules = {
        "scene_model": scene_model,
        "composer": composer,
        "grounding": grounding,
    }
    if lora_installation is not None:
        checkpoint_modules["lora"] = lora_installation.state_module
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
    resume_value = args.resume or config["training"].get("resume_from")
    initialize_value = args.initialize_from or config["training"].get("initialize_from")
    if resume_value and initialize_value:
        raise ValueError("resume_from and initialize_from are mutually exclusive")
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
        lora_mismatch = lora_checkpoint_contract_mismatch(
            initialize_preflight, configured_lora_checkpoint_contract
        )
        if lora_mismatch is not None:
            initialization_mismatches["lora"] = lora_mismatch
        if initialization_mismatches:
            raise ValueError(
                f"Initialization checkpoint architecture mismatch: {initialization_mismatches}"
            )
        loaded_initialization = load_adapter_checkpoint(
            initialize_path,
            checkpoint_modules,
            device=str(language.device),
        )
        if loaded_initialization != initialize_preflight:
            raise RuntimeError("Initialization checkpoint metadata changed while loading")
        if loaded_native_boundary_embeddings is not None:
            composer.validate_native_boundary_embeddings(loaded_native_boundary_embeddings)
        if lora_installation is not None:
            validate_lora_checkpoint_state(loaded_initialization, lora_installation)
        initialization_provenance = {
            "schema_version": 1,
            "mode": "weights_only_new_curriculum",
            "checkpoint": str(initialize_path.relative_to(PROJECT_ROOT)),
            "adapter_sha256": file_sha256(initialize_path / "adapter.safetensors"),
            "metadata_sha256": file_sha256(initialize_path / "metadata.json"),
            "checkpoint_epoch": initialize_preflight.get("epoch"),
            "checkpoint_output_namespace": initialize_preflight.get("output_namespace"),
            "checkpoint_config_hash": initialize_preflight.get("config_hash"),
            "checkpoint_source_provenance": initialize_preflight.get("source_provenance"),
            "optimizer_state_loaded": False,
            "history_loaded": False,
        }
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
        if loaded_native_boundary_embeddings is not None:
            # The checkpoint load overwrites persistent BOI/EOI buffers. Verify
            # them immediately against the already loaded, pinned Gemma model
            # before optimizer restore, scene warmup, or any composed forward.
            composer.validate_native_boundary_embeddings(loaded_native_boundary_embeddings)
        if lora_installation is not None:
            validate_lora_checkpoint_state(resume_metadata, lora_installation)
        expected = {
            "semantic_dim": semantic_dim,
            "language_hidden_dim": language.hidden_size,
            "language_model_id": config["language"]["model_id"],
            "language_revision": config["language"]["revision"],
            "scene_latents": int(config["scene_encoder"]["global_latents"]),
            "scene_model_dim": int(config["scene_encoder"]["model_dim"]),
        }
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
        expected_pair_curriculum = {
            "enabled": pair_curriculum.enabled,
            "pair_only": pair_curriculum.pair_only,
            "pair_only_scene_ids": list(pair_curriculum.pair_only_scene_ids),
            "max_units_per_pair": pair_curriculum.max_units_per_pair,
            "ranking_weight": pair_curriculum.ranking_weight,
            "ranking_margin": pair_curriculum.ranking_margin,
            "ranking_mode": pair_curriculum.ranking_mode,
            "batch_fraction": pair_curriculum.batch_fraction,
            "units_per_batch": pair_curriculum.units_per_batch,
            "steps_per_epoch": pair_curriculum.steps_per_epoch,
            "gate_enabled": pair_curriculum.gate_enabled,
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
        load_optimizer_checkpoint(resume_path, optimizer, language.device)
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
    if resume_metadata is None:
        spatial_answer_warmup_metrics = run_spatial_answer_warmup(
            scene_model,
            maps,
            pair_units,
            language,
            latent_count=int(config["scene_encoder"]["global_latents"]),
            settings=spatial_answer_warmup,
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
        epoch_grounding_losses: list[float] = []
        epoch_ranking_losses: list[float] = []
        epoch_ranking_margins: list[float] = []
        epoch_ranking_min_margins: list[float] = []
        epoch_ranking_side_accuracies: list[float] = []
        epoch_ranking_unit_accuracies: list[float] = []
        epoch_pair_scene_token_gradient_norms: list[float] = []
        epoch_lora_gradient_norms: list[float] = []
        epoch_spatial_answer_losses: list[float] = []
        epoch_spatial_answer_own_similarities: list[float] = []
        epoch_spatial_answer_alternate_similarities: list[float] = []
        epoch_spatial_answer_margins: list[float] = []
        epoch_spatial_answer_eligible_units: list[int] = []
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
        scene_model.train()
        composer.train()
        grounding.train()
        if lora_installation is not None:
            lora_installation.train()
        for curriculum_batch in curriculum:
            pair_ranking_loss = torch.zeros((), device=language.device)
            pair_ranking_diagnostics: dict[str, torch.Tensor | str | None] | None = None
            spatial_answer_loss = torch.zeros((), device=language.device)
            spatial_answer_diagnostics: dict[str, torch.Tensor | int] | None = None
            partner_output = None
            if curriculum_batch.kind == "standard":
                scene_id = str(curriculum_batch.scene_id)
                batch_records = curriculum_batch.records
                data = maps[scene_id]
                output = map_forward(scene_model, data)
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
                    partner_output = map_forward(scene_model, maps[partner_scene_id])
                    separated_pair_ids.add(pair_id)
                log_scene_ids = [scene_id]
                pair_unit_count = 0
            else:
                units = curriculum_batch.pair_units
                scene_ids = units[0].scene_ids
                if any(unit.scene_ids != scene_ids for unit in units):
                    raise ValueError("A pair batch contains inconsistent scene ordering")
                outputs_by_scene = {
                    scene_id: map_forward(scene_model, maps[scene_id]) for scene_id in scene_ids
                }
                for scene_output in outputs_by_scene.values():
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
                )
                spatial_answer_loss = pair_ranking_diagnostics["spatial_answer_contrastive_loss"]
                spatial_answer_diagnostics = pair_ranking_diagnostics["spatial_answer_contrastive"]
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

            loss = (
                base_loss
                + pair_curriculum.ranking_weight * pair_ranking_loss
                + float(anti_collapse["latent_diversity_weight"]) * diversity_loss
                + float(anti_collapse["paired_scene_separation_weight"]) * pair_loss
            )
            (loss / accumulation).backward()
            pair_scene_token_gradient_norms = None
            if curriculum_batch.kind == "pair":
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
            if accumulated_batches == accumulation:
                if lora_gradient_norms is not None:
                    epoch_lora_gradient_norms.append(float(lora_gradient_norms["total_l2"]))
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
            epoch_grounding_losses.append(float(grounding_loss.detach().cpu()))
            epoch_diversity_losses.append(float(diversity_loss.detach().cpu()))
            epoch_diversity_cosines.append(float(diversity_mean_cosine.detach().cpu()))
            epoch_diversity_max_cosines.append(float(diversity_max_cosine.detach().cpu()))
            if pair_ranking_diagnostics is not None:
                margins = pair_ranking_diagnostics["margins"]
                epoch_ranking_losses.append(float(pair_ranking_loss.detach().cpu()))
                epoch_ranking_margins.append(float(margins.detach().mean().cpu()))
                epoch_ranking_min_margins.append(float(margins.detach().min().cpu()))
                epoch_ranking_side_accuracies.append(
                    float(pair_ranking_diagnostics["side_accuracy"].detach().cpu())
                )
                epoch_ranking_unit_accuracies.append(
                    float(pair_ranking_diagnostics["unit_accuracy"].detach().cpu())
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
            torch.nn.utils.clip_grad_norm_(
                parameters, float(config["training"]["gradient_clip_norm"])
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_step += 1
            accumulated_batches = 0
        mean_loss = float(np.mean(epoch_losses))
        mean_language_loss = float(np.mean(epoch_language_losses))
        mean_grounding_loss = float(np.mean(epoch_grounding_losses))
        mean_diversity_loss = float(np.mean(epoch_diversity_losses))
        mean_diversity_cosine = float(np.mean(epoch_diversity_cosines))
        mean_diversity_max_cosine = float(np.mean(epoch_diversity_max_cosines))
        mean_pair_loss = float(np.mean(epoch_pair_losses)) if epoch_pair_losses else 0.0
        mean_pair_cosine = float(np.mean(epoch_pair_cosines)) if epoch_pair_cosines else None
        mean_pair_distance = float(np.mean(epoch_pair_distances)) if epoch_pair_distances else None
        mean_ranking_loss = float(np.mean(epoch_ranking_losses)) if epoch_ranking_losses else None
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
                "train_grounding_loss": mean_grounding_loss,
                "pair_ranking_loss": mean_ranking_loss,
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
                "pair_mean_scene_token_gradient_norm": (mean_pair_scene_token_gradient_norm),
                "lora_mean_optimizer_step_gradient_norm": (
                    float(np.mean(epoch_lora_gradient_norms)) if epoch_lora_gradient_norms else None
                ),
                "lora_max_optimizer_step_gradient_norm": (
                    max(epoch_lora_gradient_norms) if epoch_lora_gradient_norms else None
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
            "pair_curriculum": {
                "enabled": pair_curriculum.enabled,
                "pair_only": pair_curriculum.pair_only,
                "pair_only_scene_ids": list(pair_curriculum.pair_only_scene_ids),
                "max_units_per_pair": pair_curriculum.max_units_per_pair,
                "ranking_weight": pair_curriculum.ranking_weight,
                "ranking_margin": pair_curriculum.ranking_margin,
                "ranking_mode": pair_curriculum.ranking_mode,
                "batch_fraction": pair_curriculum.batch_fraction,
                "units_per_batch": pair_curriculum.units_per_batch,
                "steps_per_epoch": pair_curriculum.steps_per_epoch,
                "gate_enabled": pair_curriculum.gate_enabled,
            },
            "pair_candidate_gate": pair_gate,
            "counterfactual_pair_unit_count": len(pair_units),
            "training_counterfactual_pair_count": len(training_pairs),
            "training_counterfactual_pair_membership_sha256": (pair_membership_sha256),
        }
        if lora_installation is not None:
            assert configured_lora_optimizer is not None
            lora_installation.validate_state()
            metadata.update(
                {
                    "lora": configured_lora_checkpoint_contract,
                    "lora_wrapped_modules": list(lora_installation.target_names),
                    "lora_trainable_parameter_counts": lora_installation.parameter_counts,
                    "lora_trainable_parameter_count": lora_installation.parameter_count,
                    "lora_state_sha256": lora_installation.state_sha256(),
                }
            )
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
        "pair_curriculum": {
            "enabled": pair_curriculum.enabled,
            "pair_only": pair_curriculum.pair_only,
            "pair_only_scene_ids": list(pair_curriculum.pair_only_scene_ids),
            "max_units_per_pair": pair_curriculum.max_units_per_pair,
            "ranking_weight": pair_curriculum.ranking_weight,
            "ranking_margin": pair_curriculum.ranking_margin,
            "ranking_mode": pair_curriculum.ranking_mode,
            "batch_fraction": pair_curriculum.batch_fraction,
            "units_per_batch": pair_curriculum.units_per_batch,
            "steps_per_epoch": pair_curriculum.steps_per_epoch,
            "gate_enabled": pair_curriculum.gate_enabled,
            "gate_every_epochs": pair_curriculum.gate_every_epochs,
            "gate_stop_when_passed": pair_curriculum.stop_when_gate_passes,
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
        "pair_candidate_gate": (history[-1].get("pair_candidate_gate") if history else None),
        "training_counterfactual_pair_count": len(training_pairs),
        "training_counterfactual_pair_membership_sha256": (pair_membership_sha256),
    }
    if lora_installation is not None:
        assert configured_lora_optimizer is not None
        summary.update(
            {
                "lora": configured_lora_checkpoint_contract,
                "lora_optimizer": configured_lora_optimizer.contract(),
                "lora_wrapped_modules": list(lora_installation.target_names),
                "lora_trainable_parameter_counts": lora_installation.parameter_counts,
                "lora_trainable_parameter_count": lora_installation.parameter_count,
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
