"""Audit whether V29 counterfactual changes survive the continuous scene path.

The representation pass is deliberately question blind: it receives only opaque
scene IDs, numeric voxel maps, learned parameters, and geometry.  QA records are
read separately as evaluation supervision and are never passed to scene
tokenization.  An optional teacher-forced diagnostic then asks whether the
selected decoder changes its preference between the two canonical answers.

This module is read-only with respect to models, maps, QA, and checkpoints.  Its
only write is the requested machine-readable evaluation report.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from semantic_3d_chat.chat.file_audit import FileAccessAudit
from semantic_3d_chat.chat.runtime import StaticChatRuntime
from semantic_3d_chat.config import (
    artifact_root,
    load_config,
    project_path,
)
from semantic_3d_chat.data.dataset import QARecord, SceneQADataset
from semantic_3d_chat.language.local_lm import prompt_token_ids
from semantic_3d_chat.language.prefix_injection import prefix_sha256, stack_prefix_batches
from semantic_3d_chat.scene_encoder.map_io import load_map_tensors
from semantic_3d_chat.training.pair_curriculum import token_normalized_nll
from semantic_3d_chat.training.train_adapter import (
    forward_prefix_batch,
    map_forward,
    tokenize_answer,
)
from semantic_3d_chat.training.train_post_stack_decoder import v29_development_contract

DEFAULT_CONFIG = Path(
    "configs/experiments/gemma4_diverse20_post_stack_decoder_stage_b_v29.yaml"
)
DEFAULT_CHECKPOINT = Path(
    "data_gemma4/checkpoints/"
    "gemma4_v29_diverse20_post_stack_decoder_stage_b/best"
)
DEFAULT_OUTPUT = Path(
    "reports/gemma4/metrics/v29_counterfactual_representation_audit.json"
)
_OPAQUE_SCENE_ID = re.compile(r"scene_[0-9]{6}")
_FORBIDDEN_RUNTIME_PARTS = frozenset({"oracle", "qa", "rendered", "features"})


@dataclass(frozen=True)
class ExpectedChangeUnit:
    """One same-question, two-scene canonical answer intervention."""

    pair_id: str
    question_key: str
    change_type: str
    question: str
    reference_scene_id: str
    counterfactual_scene_id: str
    reference_answer: str
    counterfactual_answer: str


@dataclass(frozen=True)
class CounterfactualScenePair:
    """Configured physical pair plus its expected-change evaluation units."""

    pair_id: str
    change_type: str
    reference_scene_id: str
    counterfactual_scene_id: str
    expected_change_units: tuple[ExpectedChangeUnit, ...]

    @property
    def scene_ids(self) -> tuple[str, str]:
        return self.reference_scene_id, self.counterfactual_scene_id


@dataclass(frozen=True)
class SceneRepresentation:
    """CPU copy of one complete, question-independent continuous scene memory."""

    scene_id: str
    scene_tokens: torch.Tensor
    prefix: torch.Tensor
    scene_token_sha256: str
    prefix_sha256: str
    source_voxel_count: int
    input_voxel_count: int
    processed_voxel_count: int
    all_voxels_covered: bool


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _opaque_scene_id(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _OPAQUE_SCENE_ID.fullmatch(value) is None:
        raise ValueError(f"{field} must be an opaque scene_[0-9]{{6}} ID")
    return value


def _pair_roles(records: Sequence[QARecord], pair_id: str) -> tuple[str, str, str]:
    scene_by_role: dict[str, set[str]] = defaultdict(set)
    change_types: set[str] = set()
    for record in records:
        if record.counterfactual_role not in {"reference", "counterfactual"}:
            raise ValueError(f"{pair_id} contains an invalid counterfactual role")
        scene_by_role[record.counterfactual_role].add(record.scene_id)
        if not record.counterfactual_change_type:
            raise ValueError(f"{pair_id} lacks counterfactual change type")
        change_types.add(record.counterfactual_change_type)
    if set(scene_by_role) != {"reference", "counterfactual"}:
        raise ValueError(f"{pair_id} does not contain both counterfactual roles")
    if any(len(scene_ids) != 1 for scene_ids in scene_by_role.values()):
        raise ValueError(f"{pair_id} maps one role to multiple scenes")
    if len(change_types) != 1:
        raise ValueError(f"{pair_id} contains inconsistent change types")
    reference = next(iter(scene_by_role["reference"]))
    counterfactual = next(iter(scene_by_role["counterfactual"]))
    if reference == counterfactual:
        raise ValueError(f"{pair_id} uses the same scene on both sides")
    return reference, counterfactual, next(iter(change_types))


def counterfactual_scene_pairs(
    records: Sequence[QARecord],
    *,
    expected_scene_ids: Sequence[str] | None = None,
) -> tuple[CounterfactualScenePair, ...]:
    """Build strict pair specifications from evaluation-only QA annotations."""

    by_pair: dict[str, list[QARecord]] = defaultdict(list)
    for record in records:
        if record.counterfactual_pair_id is None:
            raise ValueError(f"QA record {record.question_id} lacks a counterfactual pair ID")
        by_pair[record.counterfactual_pair_id].append(record)
    if not by_pair:
        raise ValueError("Counterfactual representation audit requires paired QA")

    pairs: list[CounterfactualScenePair] = []
    seen_scene_ids: set[str] = set()
    for pair_id, pair_records in sorted(by_pair.items()):
        reference_scene, counterfactual_scene, change_type = _pair_roles(
            pair_records, pair_id
        )
        _opaque_scene_id(reference_scene, field=f"{pair_id}.reference_scene_id")
        _opaque_scene_id(
            counterfactual_scene, field=f"{pair_id}.counterfactual_scene_id"
        )
        overlap = seen_scene_ids & {reference_scene, counterfactual_scene}
        if overlap:
            raise ValueError(f"Scenes occur in more than one configured pair: {sorted(overlap)}")
        seen_scene_ids.update((reference_scene, counterfactual_scene))

        changed_by_key: dict[str, list[QARecord]] = defaultdict(list)
        for record in pair_records:
            if record.counterfactual_expected_change is True:
                if not record.counterfactual_question_key:
                    raise ValueError(
                        f"Expected-change record {record.question_id} lacks a question key"
                    )
                changed_by_key[record.counterfactual_question_key].append(record)

        units: list[ExpectedChangeUnit] = []
        for question_key, unit_records in sorted(changed_by_key.items()):
            if len(unit_records) != 2:
                raise ValueError(
                    f"Expected-change unit {pair_id}/{question_key} must have exactly two sides"
                )
            by_role = {record.counterfactual_role: record for record in unit_records}
            if set(by_role) != {"reference", "counterfactual"}:
                raise ValueError(
                    f"Expected-change unit {pair_id}/{question_key} lacks both roles"
                )
            reference = by_role["reference"]
            counterfactual = by_role["counterfactual"]
            if (
                reference.scene_id != reference_scene
                or counterfactual.scene_id != counterfactual_scene
            ):
                raise ValueError(
                    f"Expected-change unit {pair_id}/{question_key} swaps configured roles"
                )
            if reference.question != counterfactual.question:
                raise ValueError(
                    f"Expected-change unit {pair_id}/{question_key} changes the question"
                )
            if reference.answer == counterfactual.answer:
                raise ValueError(
                    f"Expected-change unit {pair_id}/{question_key} has identical answers"
                )
            units.append(
                ExpectedChangeUnit(
                    pair_id=pair_id,
                    question_key=question_key,
                    change_type=change_type,
                    question=reference.question,
                    reference_scene_id=reference_scene,
                    counterfactual_scene_id=counterfactual_scene,
                    reference_answer=reference.answer,
                    counterfactual_answer=counterfactual.answer,
                )
            )
        pairs.append(
            CounterfactualScenePair(
                pair_id=pair_id,
                change_type=change_type,
                reference_scene_id=reference_scene,
                counterfactual_scene_id=counterfactual_scene,
                expected_change_units=tuple(units),
            )
        )

    if expected_scene_ids is not None:
        expected = {
            _opaque_scene_id(scene_id, field="expected_scene_ids")
            for scene_id in expected_scene_ids
        }
        if seen_scene_ids != expected:
            raise ValueError(
                "Counterfactual QA scene inventory differs from the configured split: "
                f"qa={sorted(seen_scene_ids)} configured={sorted(expected)}"
            )
    return tuple(pairs)


def load_counterfactual_scene_pairs(
    qa_path: Path,
    *,
    expected_scene_ids: Sequence[str] | None = None,
) -> tuple[CounterfactualScenePair, ...]:
    """Read QA only in the evaluation layer, before the runtime audit begins."""

    resolved = qa_path.resolve()
    if "oracle" in {part.casefold() for part in resolved.parts}:
        raise ValueError("Counterfactual evaluation must use QA, not oracle metadata")
    return counterfactual_scene_pairs(
        SceneQADataset(resolved).records,
        expected_scene_ids=expected_scene_ids,
    )


def _token_matrix(value: torch.Tensor, *, name: str) -> torch.Tensor:
    tensor = value.detach().cpu().to(torch.float64)
    if tensor.ndim == 3 and tensor.shape[0] == 1:
        tensor = tensor[0]
    if tensor.ndim != 2 or tensor.shape[0] == 0 or tensor.shape[1] == 0:
        raise ValueError(f"{name} must have shape [T,D] or [1,T,D]")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} contains NaN or infinity")
    return tensor


def tensor_delta_metrics(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    include_per_token: bool = True,
) -> dict[str, Any]:
    """Measure complete and tokenwise change between two scene memories."""

    first = _token_matrix(left, name="left tensor")
    second = _token_matrix(right, name="right tensor")
    if first.shape != second.shape:
        raise ValueError(f"Representation shapes differ: {first.shape} != {second.shape}")
    delta = second - first
    delta_rms = torch.sqrt(delta.square().mean())
    reference_rms = torch.sqrt((first.square().mean() + second.square().mean()) / 2.0)
    relative_rms = delta_rms / reference_rms.clamp_min(torch.finfo(torch.float64).eps)
    flat_norm_product = torch.linalg.vector_norm(first) * torch.linalg.vector_norm(second)
    cosine = (
        torch.tensor(1.0, dtype=torch.float64)
        if float(flat_norm_product) == 0.0 and torch.equal(first, second)
        else (first * second).sum()
        / flat_norm_product.clamp_min(torch.finfo(torch.float64).eps)
    )

    per_token_rms = torch.sqrt(delta.square().mean(dim=-1))
    per_token_scale = torch.sqrt(
        (first.square().mean(dim=-1) + second.square().mean(dim=-1)) / 2.0
    )
    per_token_relative = per_token_rms / per_token_scale.clamp_min(
        torch.finfo(torch.float64).eps
    )
    per_token_norm_product = torch.linalg.vector_norm(first, dim=-1) * torch.linalg.vector_norm(
        second, dim=-1
    )
    per_token_cosine = (first * second).sum(dim=-1) / per_token_norm_product.clamp_min(
        torch.finfo(torch.float64).eps
    )
    both_zero = per_token_norm_product.eq(0) & first.eq(second).all(dim=-1)
    per_token_cosine = torch.where(both_zero, torch.ones_like(per_token_cosine), per_token_cosine)

    metrics: dict[str, Any] = {
        "shape": [1, int(first.shape[0]), int(first.shape[1])],
        "delta_rms": float(delta_rms),
        "reference_rms": float(reference_rms),
        "relative_rms": float(relative_rms),
        "cosine": float(cosine),
        "maximum_absolute_delta": float(delta.abs().max()),
        "changed_token_count_exact": int(per_token_rms.gt(0).sum()),
        "changed_token_fraction_exact": float(per_token_rms.gt(0).float().mean()),
        "changed_token_count_relative_gt_1e_4": int(per_token_relative.gt(1e-4).sum()),
        "changed_token_count_relative_gt_1e_3": int(per_token_relative.gt(1e-3).sum()),
        "per_token_rms_summary": {
            "minimum": float(per_token_rms.min()),
            "median": float(per_token_rms.median()),
            "mean": float(per_token_rms.mean()),
            "p90": float(torch.quantile(per_token_rms, 0.9)),
            "maximum": float(per_token_rms.max()),
        },
        "per_token_relative_rms_summary": {
            "minimum": float(per_token_relative.min()),
            "median": float(per_token_relative.median()),
            "mean": float(per_token_relative.mean()),
            "p90": float(torch.quantile(per_token_relative, 0.9)),
            "maximum": float(per_token_relative.max()),
        },
    }
    if include_per_token:
        metrics.update(
            {
                "per_token_rms": [float(value) for value in per_token_rms],
                "per_token_relative_rms": [float(value) for value in per_token_relative],
                "per_token_cosine": [float(value) for value in per_token_cosine],
            }
        )
    return metrics


def unrelated_scene_pairs(
    scene_ids: Sequence[str], configured_pairs: Sequence[CounterfactualScenePair]
) -> tuple[tuple[str, str], ...]:
    """Return every deterministic cross-pair scene comparison."""

    unique = sorted({_opaque_scene_id(value, field="scene_ids") for value in scene_ids})
    excluded = {frozenset(pair.scene_ids) for pair in configured_pairs}
    return tuple(
        (left, right)
        for left, right in itertools.combinations(unique, 2)
        if frozenset((left, right)) not in excluded
    )


def _distribution(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("A delta distribution cannot be empty")
    tensor = torch.tensor(values, dtype=torch.float64)
    if not torch.isfinite(tensor).all():
        raise ValueError("Delta distribution contains NaN or infinity")
    return {
        "count": len(values),
        "minimum": float(tensor.min()),
        "median": float(tensor.median()),
        "mean": float(tensor.mean()),
        "p90": float(torch.quantile(tensor, 0.9)),
        "maximum": float(tensor.max()),
    }


def compare_paired_to_unrelated(
    paired_rows: Sequence[Mapping[str, Any]],
    unrelated_rows: Sequence[Mapping[str, Any]],
    *,
    representation_key: str,
) -> dict[str, Any]:
    """Contrast configured physical interventions with ordinary scene variation."""

    paired_rms = [float(row[representation_key]["delta_rms"]) for row in paired_rows]
    unrelated_rms = [float(row[representation_key]["delta_rms"]) for row in unrelated_rows]
    paired_cosine = [float(row[representation_key]["cosine"]) for row in paired_rows]
    unrelated_cosine = [float(row[representation_key]["cosine"]) for row in unrelated_rows]
    paired_distribution = _distribution(paired_rms)
    unrelated_distribution = _distribution(unrelated_rms)
    unrelated_mean = float(unrelated_distribution["mean"])
    unrelated_median = float(unrelated_distribution["median"])
    per_pair_percentiles = []
    for row, value in zip(paired_rows, paired_rms, strict=True):
        percentile = sum(candidate <= value for candidate in unrelated_rms) / len(unrelated_rms)
        per_pair_percentiles.append(
            {
                "pair_id": row["pair_id"],
                "delta_rms": value,
                "percentile_among_unrelated": percentile,
            }
        )
    return {
        "paired_delta_rms": paired_distribution,
        "unrelated_delta_rms": unrelated_distribution,
        "paired_cosine": _distribution(paired_cosine),
        "unrelated_cosine": _distribution(unrelated_cosine),
        "paired_mean_rms_to_unrelated_mean_ratio": (
            float(paired_distribution["mean"])
            / max(unrelated_mean, torch.finfo(torch.float64).eps)
        ),
        "paired_fraction_below_unrelated_median": (
            sum(value < unrelated_median for value in paired_rms) / len(paired_rms)
        ),
        "per_pair_unrelated_percentile": per_pair_percentiles,
    }


def _guard_runtime_path(path: Path, *, purpose: str) -> Path:
    resolved = path.expanduser().resolve()
    forbidden = _FORBIDDEN_RUNTIME_PARTS & {part.casefold() for part in resolved.parts}
    if forbidden:
        raise ValueError(f"Refusing {purpose} from forbidden runtime path: {resolved}")
    return resolved


def _representation(
    scene_id: str,
    scene_tokens: torch.Tensor,
    prefix: torch.Tensor,
    *,
    source_voxels: int,
    input_voxels: int,
    processed_voxels: int,
) -> SceneRepresentation:
    tokens = scene_tokens.detach().cpu().contiguous()
    prefix_cpu = prefix.detach().cpu().contiguous()
    return SceneRepresentation(
        scene_id=scene_id,
        scene_tokens=tokens,
        prefix=prefix_cpu,
        scene_token_sha256=prefix_sha256(tokens),
        prefix_sha256=prefix_sha256(prefix_cpu),
        source_voxel_count=int(source_voxels),
        input_voxel_count=int(input_voxels),
        processed_voxel_count=int(processed_voxels),
        all_voxels_covered=int(processed_voxels) == int(input_voxels) > 0,
    )


def encode_scene_representations(
    config: dict[str, Any],
    checkpoint: Path,
    scene_ids: Sequence[str],
    *,
    audit: FileAccessAudit | None = None,
) -> tuple[dict[str, SceneRepresentation], StaticChatRuntime, dict[str, Any]]:
    """Encode opaque numeric maps once each without receiving QA text or labels."""

    ordered = sorted({_opaque_scene_id(value, field="scene_ids") for value in scene_ids})
    if not ordered:
        raise ValueError("Representation audit requires at least one scene")
    checkpoint = _guard_runtime_path(checkpoint, purpose="checkpoint")
    runtime = StaticChatRuntime.load(
        config,
        ordered[0],
        checkpoint=checkpoint,
        audit=audit,
        local_files_only=True,
    )
    representations: dict[str, SceneRepresentation] = {}
    first_processed = int(runtime.scene_output.audit["processed_voxels"].detach().cpu())
    if first_processed != runtime.map_data.voxel_count:
        raise RuntimeError("Initial runtime omitted occupied voxels")
    representations[ordered[0]] = _representation(
        ordered[0],
        runtime.scene_output.scene_tokens,
        runtime.scene_prefix,
        source_voxels=runtime.map_data.source_voxel_count,
        input_voxels=runtime.map_data.voxel_count,
        processed_voxels=first_processed,
    )
    loaded_map_paths = [
        str(
            _guard_runtime_path(
                project_path(config, "maps", ordered[0], "voxel_map.npz"),
                purpose="numeric voxel map",
            )
        )
    ]
    model_dtype = next(runtime.language.model.parameters()).dtype
    with torch.inference_mode():
        for scene_id in ordered[1:]:
            map_path = _guard_runtime_path(
                project_path(config, "maps", scene_id, "voxel_map.npz"),
                purpose="numeric voxel map",
            )
            if audit is not None:
                audit.record(map_path)
            data = load_map_tensors(
                map_path,
                config["scene"]["room_size_m"],
                runtime.language.device,
                input_voxel_size_m=config["scene_encoder"].get("input_voxel_size_m"),
            )
            output = map_forward(
                runtime.scene_model,
                data,
                runtime.global_scene_residual,
                runtime.signed_x_scene_residual,
                runtime.dense_aligner,
                runtime.dense_sidecar_adapter,
            )
            processed = int(output.audit["processed_voxels"].detach().cpu())
            if processed != data.voxel_count:
                raise RuntimeError(
                    f"Incomplete full-scene representation for {scene_id}: "
                    f"{processed}/{data.voxel_count}"
                )
            prefix = runtime.composer.scene_prefix(output.scene_tokens.to(model_dtype))
            representations[scene_id] = _representation(
                scene_id,
                output.scene_tokens,
                prefix,
                source_voxels=data.source_voxel_count,
                input_voxels=data.voxel_count,
                processed_voxels=processed,
            )
            loaded_map_paths.append(str(map_path))
            del data, output, prefix
            if runtime.language.device.type == "mps":
                torch.mps.empty_cache()

    # The same path used above is exact for the scene StaticChatRuntime encoded.
    # Make the coverage invariant explicit instead of inferring it from hashes.
    for scene_id, representation in representations.items():
        if representation.processed_voxel_count <= 0:
            raise RuntimeError(f"Scene {scene_id} has no processed voxels")
    return representations, runtime, {
        "schema_version": 1,
        "scene_encoder_question_text_received": False,
        "scene_encoder_answer_text_received": False,
        "question_dependent_retrieval": False,
        "oracle_environment_files_loaded": False,
        "model_load_count": 1,
        "scene_count": len(representations),
        "loaded_environment_files": loaded_map_paths,
        "checkpoint": str(checkpoint),
        "device": str(runtime.language.device),
        "all_scene_tokens_computed_before_teacher_forcing": True,
        "all_voxels_covered": all(
            representation.all_voxels_covered for representation in representations.values()
        ),
    }


def representation_delta_report(
    representations: Mapping[str, SceneRepresentation],
    pairs: Sequence[CounterfactualScenePair],
) -> dict[str, Any]:
    """Measure configured pairs and all unrelated-scene controls."""

    paired_rows: list[dict[str, Any]] = []
    for pair in pairs:
        reference = representations[pair.reference_scene_id]
        counterfactual = representations[pair.counterfactual_scene_id]
        paired_rows.append(
            {
                "pair_id": pair.pair_id,
                "change_type": pair.change_type,
                "reference_scene_id": pair.reference_scene_id,
                "counterfactual_scene_id": pair.counterfactual_scene_id,
                "expected_change_unit_count": len(pair.expected_change_units),
                "scene_tokens": tensor_delta_metrics(
                    reference.scene_tokens,
                    counterfactual.scene_tokens,
                    include_per_token=True,
                ),
                "prefix": tensor_delta_metrics(
                    reference.prefix,
                    counterfactual.prefix,
                    include_per_token=True,
                ),
            }
        )

    unrelated_rows: list[dict[str, Any]] = []
    for left_id, right_id in unrelated_scene_pairs(list(representations), pairs):
        left = representations[left_id]
        right = representations[right_id]
        unrelated_rows.append(
            {
                "left_scene_id": left_id,
                "right_scene_id": right_id,
                "scene_tokens": tensor_delta_metrics(
                    left.scene_tokens, right.scene_tokens, include_per_token=False
                ),
                "prefix": tensor_delta_metrics(
                    left.prefix, right.prefix, include_per_token=False
                ),
            }
        )
    if not unrelated_rows:
        raise ValueError("At least two configured pairs are required for unrelated controls")
    return {
        "configured_pair_count": len(paired_rows),
        "unrelated_pair_count": len(unrelated_rows),
        "paired": paired_rows,
        "unrelated": unrelated_rows,
        "contrast": {
            key: compare_paired_to_unrelated(
                paired_rows, unrelated_rows, representation_key=key
            )
            for key in ("scene_tokens", "prefix")
        },
        "all_configured_scene_tokens_distinct": all(
            row["scene_tokens"]["delta_rms"] > 0.0 for row in paired_rows
        ),
        "all_configured_prefixes_distinct": all(
            row["prefix"]["delta_rms"] > 0.0 for row in paired_rows
        ),
    }


def _first_differing_offset(
    first_ids: torch.Tensor, second_ids: torch.Tensor
) -> tuple[int, int, int] | None:
    first = first_ids.reshape(-1)
    second = second_ids.reshape(-1)
    common = min(first.numel(), second.numel())
    differing = first[:common].ne(second[:common]).nonzero(as_tuple=False).flatten()
    if differing.numel() == 0:
        return None
    offset = int(differing[0])
    return offset, int(first[offset]), int(second[offset])


def _score_two_answers(
    runtime: StaticChatRuntime,
    representation: SceneRepresentation,
    *,
    question: str,
    first_answer: str,
    second_answer: str,
) -> dict[str, Any]:
    device = runtime.language.device
    model_dtype = next(runtime.language.model.parameters()).dtype
    scene_tokens = representation.scene_tokens.to(device=device, dtype=model_dtype)
    prompt_ids = prompt_token_ids(
        runtime.language.tokenizer,
        str(runtime.config["language"]["system_prompt"]),
        question,
        device,
    )
    answer_ids = [
        tokenize_answer(runtime.language.tokenizer, answer, device)
        for answer in (first_answer, second_answer)
    ]
    batches = [
        runtime.composer.compose(
            scene_tokens,
            prompt_ids,
            runtime.language.model.get_input_embeddings(),
            candidate_ids,
            prefix_backend=getattr(runtime.language, "prefix_backend", None),
        )
        for candidate_ids in answer_ids
    ]
    batch = stack_prefix_batches(
        batches,
        device,
        prefix_backend=getattr(runtime.language, "prefix_backend", None),
    )
    output = forward_prefix_batch(runtime.language, batch)
    if batch.labels is None:
        raise RuntimeError("Teacher-forced candidate batch lacks labels")
    nll = token_normalized_nll(output.logits, batch.labels)
    result: dict[str, Any] = {
        "first_answer_mean_log_probability": -float(nll[0].detach().cpu()),
        "second_answer_mean_log_probability": -float(nll[1].detach().cpu()),
        "first_minus_second_mean_log_probability_margin": float(
            (nll[1] - nll[0]).detach().cpu()
        ),
    }
    differing = _first_differing_offset(answer_ids[0], answer_ids[1])
    if differing is None:
        result["first_differing_token"] = None
    else:
        offset, first_token_id, second_token_id = differing
        supervised = batch.labels[0].ne(-100).nonzero(as_tuple=False).flatten()
        if offset >= supervised.numel():
            raise RuntimeError("Differing answer offset exceeds supervised labels")
        label_position = int(supervised[offset])
        if label_position == 0:
            raise RuntimeError("Differing answer token lacks a causal predecessor")
        logits = output.logits[0, label_position - 1].float()
        margin = logits[first_token_id] - logits[second_token_id]
        result["first_differing_token"] = {
            "answer_offset": offset,
            "first_token_id": first_token_id,
            "second_token_id": second_token_id,
            "first_minus_second_logit_margin": float(margin.detach().cpu()),
        }
    del output, batch, batches
    return result


def summarize_teacher_forced_margins(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize whether changing only the scene reverses answer preference."""

    if not rows:
        raise ValueError("Teacher-forced summary requires at least one unit")
    both_correct = 0
    side_correct = 0
    flips = 0
    sequence_shifts: list[float] = []
    sequence_biases: list[float] = []
    token_shift_values: list[float] = []
    token_both_correct = 0
    token_units = 0
    for row in rows:
        reference_margin = float(
            row["reference_scene"]["first_minus_second_mean_log_probability_margin"]
        )
        counterfactual_margin = float(
            row["counterfactual_scene"][
                "first_minus_second_mean_log_probability_margin"
            ]
        )
        reference_correct = reference_margin > 0.0
        counterfactual_correct = counterfactual_margin < 0.0
        both_correct += int(reference_correct and counterfactual_correct)
        side_correct += int(reference_correct) + int(counterfactual_correct)
        flips += int((reference_margin > 0.0) != (counterfactual_margin > 0.0))
        sequence_shifts.append(counterfactual_margin - reference_margin)
        sequence_biases.append((counterfactual_margin + reference_margin) / 2.0)

        reference_token = row["reference_scene"].get("first_differing_token")
        counterfactual_token = row["counterfactual_scene"].get("first_differing_token")
        if isinstance(reference_token, Mapping) and isinstance(
            counterfactual_token, Mapping
        ):
            reference_token_margin = float(
                reference_token["first_minus_second_logit_margin"]
            )
            counterfactual_token_margin = float(
                counterfactual_token["first_minus_second_logit_margin"]
            )
            token_units += 1
            token_shift_values.append(counterfactual_token_margin - reference_token_margin)
            token_both_correct += int(
                reference_token_margin > 0.0 and counterfactual_token_margin < 0.0
            )

    mean_abs_shift = sum(abs(value) for value in sequence_shifts) / len(sequence_shifts)
    mean_abs_bias = sum(abs(value) for value in sequence_biases) / len(sequence_biases)
    return {
        "unit_count": len(rows),
        "side_count": 2 * len(rows),
        "two_sided_sequence_preference_accuracy": both_correct / len(rows),
        "per_side_sequence_preference_accuracy": side_correct / (2 * len(rows)),
        "same_question_preference_flip_rate": flips / len(rows),
        "mean_absolute_scene_induced_sequence_log_odds_shift": mean_abs_shift,
        "mean_absolute_question_answer_bias_midpoint": mean_abs_bias,
        "scene_shift_to_bias_ratio": mean_abs_shift
        / max(mean_abs_bias, torch.finfo(torch.float64).eps),
        "first_differing_token_unit_count": token_units,
        "two_sided_first_differing_token_accuracy": (
            None if token_units == 0 else token_both_correct / token_units
        ),
        "mean_absolute_scene_induced_first_token_logit_shift": (
            None
            if not token_shift_values
            else sum(abs(value) for value in token_shift_values) / len(token_shift_values)
        ),
    }


def teacher_forced_answer_margins(
    runtime: StaticChatRuntime,
    representations: Mapping[str, SceneRepresentation],
    pairs: Sequence[CounterfactualScenePair],
) -> dict[str, Any]:
    """Score both canonical answers under both complete scene prefixes."""

    rows: list[dict[str, Any]] = []
    runtime.language.model.eval()
    with torch.inference_mode():
        for pair in pairs:
            for unit in pair.expected_change_units:
                reference = _score_two_answers(
                    runtime,
                    representations[unit.reference_scene_id],
                    question=unit.question,
                    first_answer=unit.reference_answer,
                    second_answer=unit.counterfactual_answer,
                )
                counterfactual = _score_two_answers(
                    runtime,
                    representations[unit.counterfactual_scene_id],
                    question=unit.question,
                    first_answer=unit.reference_answer,
                    second_answer=unit.counterfactual_answer,
                )
                rows.append(
                    {
                        "pair_id": unit.pair_id,
                        "question_key": unit.question_key,
                        "change_type": unit.change_type,
                        "question": unit.question,
                        "reference_scene_id": unit.reference_scene_id,
                        "counterfactual_scene_id": unit.counterfactual_scene_id,
                        "first_answer": unit.reference_answer,
                        "second_answer": unit.counterfactual_answer,
                        "reference_scene": reference,
                        "counterfactual_scene": counterfactual,
                    }
                )
                if runtime.language.device.type == "mps":
                    torch.mps.empty_cache()
    return {
        "schema_version": 1,
        "evaluation_only_qa_used": True,
        "qa_serialized_to_checkpoint": False,
        "scene_representations_recomputed_from_question": False,
        "rows": rows,
        "summary": summarize_teacher_forced_margins(rows),
    }


def _diagnosis(
    representation: Mapping[str, Any],
    teacher: Mapping[str, Any] | None,
) -> dict[str, Any]:
    prefixes_distinct = representation["all_configured_prefixes_distinct"] is True
    paired_ratio = float(
        representation["contrast"]["prefix"][
            "paired_mean_rms_to_unrelated_mean_ratio"
        ]
    )
    pair_percentiles = representation["contrast"]["prefix"][
        "per_pair_unrelated_percentile"
    ]
    below_unrelated_minimum = [
        str(row["pair_id"])
        for row in pair_percentiles
        if float(row["percentile_among_unrelated"]) == 0.0
    ]
    result: dict[str, Any] = {
        "configured_prefixes_are_not_byte_identical": prefixes_distinct,
        "counterfactual_prefix_delta_to_unrelated_scene_delta_ratio": paired_ratio,
        "counterfactual_change_is_diluted_relative_to_scene_identity": paired_ratio < 0.5,
        "pair_ids_below_every_unrelated_prefix_delta": below_unrelated_minimum,
    }
    if teacher is None:
        result.update(
            {
                "decoder_preference_flip_rate": None,
                "failure_localization": (
                    "representation metrics ready; run --teacher-forced to distinguish "
                    "decoder insensitivity from free-generation decoding effects"
                ),
            }
        )
        return result
    summary = teacher["summary"]
    flip_rate = float(summary["same_question_preference_flip_rate"])
    result["decoder_preference_flip_rate"] = flip_rate
    if not prefixes_distinct:
        localization = "scene tokenizer collapsed at least one physical pair"
    elif flip_rate == 0.0:
        localization = (
            "nonzero continuous pair deltas reach the prefix, but the selected decoder "
            "does not reverse canonical-answer preference; failure is downstream of "
            "prefix construction"
        )
    elif float(summary["two_sided_sequence_preference_accuracy"]) == 0.0:
        localization = (
            "the decoder reacts to scene changes, but its canonical answer binding is wrong"
        )
    else:
        localization = (
            "teacher forcing is scene-sensitive; remaining invariance is at least partly "
            "a free-generation or answer-normalization failure"
        )
    result["failure_localization"] = localization
    return result


def run_audit(
    config_path: Path,
    checkpoint: Path,
    *,
    qa_path: Path | None = None,
    include_teacher_forced: bool = False,
) -> dict[str, Any]:
    config = load_config(config_path)
    development = v29_development_contract(config)
    if development is None:
        raise ValueError("The V29 audit requires a locked v29_development contract")
    expected_scene_ids = development.validation_scene_ids
    qa_path = (
        artifact_root(config, "qa") / "validation.jsonl"
        if qa_path is None
        else qa_path
    ).resolve()

    # Evaluation metadata is completely materialized before the runtime audit.
    pairs = load_counterfactual_scene_pairs(
        qa_path,
        expected_scene_ids=expected_scene_ids,
    )
    expected_change_count = sum(len(pair.expected_change_units) for pair in pairs)
    if expected_change_count == 0:
        raise ValueError("Validation split has no expected-change counterfactual units")

    oracle_root = (project_path(config) / "oracle").resolve()
    runtime_audit = FileAccessAudit(forbidden_roots=[oracle_root, qa_path.parent])
    with runtime_audit:
        representations, runtime, runtime_contract = encode_scene_representations(
            config,
            checkpoint,
            expected_scene_ids,
            audit=runtime_audit,
        )
        representation_report = representation_delta_report(representations, pairs)
        teacher_report = (
            teacher_forced_answer_margins(runtime, representations, pairs)
            if include_teacher_forced
            else None
        )
    runtime_audit.assert_clean()

    scene_inventory = {
        scene_id: {
            "scene_token_sha256": value.scene_token_sha256,
            "prefix_sha256": value.prefix_sha256,
            "scene_token_shape": list(value.scene_tokens.shape),
            "prefix_shape": list(value.prefix.shape),
            "source_voxel_count": value.source_voxel_count,
            "input_voxel_count": value.input_voxel_count,
            "processed_voxel_count": value.processed_voxel_count,
            "all_voxels_covered": value.all_voxels_covered,
        }
        for scene_id, value in sorted(representations.items())
    }
    return {
        "schema_version": 1,
        "artifact": "v29_counterfactual_representation_audit",
        "read_only_evaluation": True,
        "checkpoint": str(checkpoint.resolve()),
        "config": str(config_path.resolve()),
        "evaluation_qa": {
            "path": str(qa_path),
            "loaded_before_runtime_audit": True,
            "configured_pair_count": len(pairs),
            "expected_change_unit_count": expected_change_count,
            "environment_text_passed_to_scene_encoder": False,
        },
        "runtime_isolation": {
            **runtime_contract,
            "forbidden_roots": [str(oracle_root), str(qa_path.parent)],
            "forbidden_accesses": runtime_audit.forbidden_accesses(),
            "loaded_file_count": len(runtime_audit.unique_paths),
            "loaded_files": runtime_audit.unique_paths,
            "passed": not runtime_audit.forbidden_accesses(),
        },
        "scene_inventory": scene_inventory,
        "representation_deltas": representation_report,
        "teacher_forced_answer_margins": teacher_report,
        "diagnosis": _diagnosis(representation_report, teacher_report),
        "passed": (
            not runtime_audit.forbidden_accesses()
            and runtime_contract["all_voxels_covered"] is True
            and representation_report["all_configured_prefixes_distinct"] is True
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--qa", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--teacher-forced",
        action="store_true",
        help="Also score both canonical answers under both selected scene prefixes.",
    )
    args = parser.parse_args()
    report = run_audit(
        args.config,
        args.checkpoint,
        qa_path=args.qa,
        include_teacher_forced=args.teacher_forced,
    )
    _atomic_json(args.output, report)
    print(
        json.dumps(
            {
                "artifact": report["artifact"],
                "output": str(args.output.resolve()),
                "passed": report["passed"],
                "diagnosis": report["diagnosis"],
                "teacher_forced_summary": (
                    None
                    if report["teacher_forced_answer_margins"] is None
                    else report["teacher_forced_answer_margins"]["summary"]
                ),
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
