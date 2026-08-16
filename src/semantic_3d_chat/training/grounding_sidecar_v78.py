"""Train a numeric grounding repair from historical training scenes only.

The V78 sidecar consumes the immutable, question-independent 256-token scene
prefix and a frozen Gemma input-embedding summary of the user's question.  It
never consumes an answer, category codebook, object ID, oracle file, or scene
description at inference.  Oracle-derived ``target_xyz`` values are used only
while fitting and scoring this diagnostic in the training/evaluation process.

This module intentionally does not modify the V54/V75 chat runtime.  A V78
candidate remains a separately sealed diagnostic until a later promotion step
adds a strictly audited runtime loader.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from semantic_3d_chat.chat.runtime_config import (
    effective_runtime_config_sha256,
    load_runtime_config,
)
from semantic_3d_chat.evaluation.gemma4_semantic_sanity import (
    GEMMA4_PROJECTED_DIM,
    GEMMA4_TOKEN_EMBEDDING_KEY,
    load_category_embeddings_selective,
    resolve_local_snapshot,
)
from semantic_3d_chat.evaluation.prediction_artifacts import checkpoint_fingerprint
from semantic_3d_chat.scene_encoder.grounding_sidecar_v78 import (
    ARCHITECTURE,
    ARTIFACT,
    EXPECTED_CHECKPOINT_FILES,
    METADATA_FILENAME,
    WEIGHTS_FILENAME,
    GroundingSidecarV78,
    denormalize_xyz,
    normalize_xyz,
)

EXPECTED_CANDIDATE_FILES = EXPECTED_CHECKPOINT_FILES
DEFAULT_MODEL_ID = "google/gemma-4-E2B-it"
DEFAULT_MODEL_REVISION = "3e22461f65e89153144f8adb70e3b8c2cc9845a7"
DEFAULT_ROOM_MIN = (-3.0, -2.5, 0.0)
DEFAULT_ROOM_MAX = (3.0, 2.5, 3.0)
DEFAULT_BASE_CHECKPOINT = "data_gemma4/runtime/checkpoints/gemma4_v54_release_v1"
DEFAULT_RUNTIME_CONFIG = "configs/runtime/gemma4_v56_question_control.yaml"

_BLOCKED_DATA_WORDS = frozenset(
    {"oracle", "official", "validation", "test", "deferred", "final_once"}
)


@dataclass(frozen=True)
class GroundingRecord:
    """Minimal in-memory training record; answer and instance fields are dropped."""

    question_id: str
    scene_id: str
    pair_id: str
    paired_scene_id: str
    question_key: str
    question: str
    target_xyz: tuple[float, float, float]


@dataclass(frozen=True)
class GroundingPrediction:
    record: GroundingRecord
    xyz: tuple[float, float, float]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _validate_historical_training_path(path: str | Path, *, label: str) -> Path:
    source = Path(path).expanduser().resolve()
    lowered_parts = [part.casefold() for part in source.parts]
    lowered = set(lowered_parts)
    training_indices = [
        index for index, part in enumerate(lowered_parts) if part == "training"
    ]
    scoped_parts = (
        lowered_parts[training_indices[-1] + 1 :] if training_indices else [source.name.casefold()]
    )
    searchable = "/".join(scoped_parts)
    tokens = set(re.split(r"[^a-z0-9]+", searchable))
    blocked = sorted((lowered & _BLOCKED_DATA_WORDS) | (tokens & _BLOCKED_DATA_WORDS))
    if "final_once" in searchable:
        blocked.append("final_once")
    if blocked:
        raise ValueError(f"{label} must not use held/official/oracle data: {blocked}")
    if not source.is_file():
        raise FileNotFoundError(f"{label} not found: {source}")
    filename_tokens = source.stem.casefold().replace("-", "_").split("_")
    if "training" not in lowered and "training" not in filename_tokens:
        raise ValueError(f"{label} must be explicitly identified as historical training data")
    return source


def load_historical_grounding_records(path: str | Path) -> list[GroundingRecord]:
    """Load grounded historical rows and immediately discard semantic metadata."""

    source = _validate_historical_training_path(path, label="Grounding QA source")
    records: list[GroundingRecord] = []
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        raw = json.loads(line)
        target = raw.get("target_xyz")
        if target is None:
            continue
        if not isinstance(target, list) or len(target) != 3:
            raise ValueError(f"Invalid target_xyz on line {line_number}")
        xyz = tuple(float(value) for value in target)
        if not all(math.isfinite(value) for value in xyz):
            raise ValueError(f"Non-finite target_xyz on line {line_number}")
        required = {
            "question_id",
            "scene_id",
            "counterfactual_pair_id",
            "counterfactual_paired_scene_id",
            "counterfactual_question_key",
            "question",
        }
        missing = sorted(key for key in required if not isinstance(raw.get(key), str))
        if missing:
            raise ValueError(f"Missing string fields on line {line_number}: {missing}")
        question = str(raw["question"]).strip()
        if not question:
            raise ValueError(f"Empty question on line {line_number}")
        records.append(
            GroundingRecord(
                question_id=str(raw["question_id"]),
                scene_id=str(raw["scene_id"]),
                pair_id=str(raw["counterfactual_pair_id"]),
                paired_scene_id=str(raw["counterfactual_paired_scene_id"]),
                question_key=str(raw["counterfactual_question_key"]),
                question=question,
                target_xyz=xyz,
            )
        )
    if not records:
        raise ValueError("Historical training source has no grounded rows")
    keys = [(record.scene_id, record.question_id) for record in records]
    if len(keys) != len(set(keys)):
        raise ValueError("Historical grounding rows contain duplicate scene/question IDs")
    return records


def pair_disjoint_internal_split(
    records: Sequence[GroundingRecord],
    *,
    modulo: int = 4,
    held_remainder: int = 3,
) -> tuple[list[GroundingRecord], list[GroundingRecord], dict[str, Any]]:
    """Create a deterministic pair- and scene-disjoint historical holdout."""

    if modulo < 2 or not 0 <= held_remainder < modulo:
        raise ValueError("Invalid deterministic split parameters")
    pairs = sorted({record.pair_id for record in records})
    held_pairs = {pair for index, pair in enumerate(pairs) if index % modulo == held_remainder}
    train = [record for record in records if record.pair_id not in held_pairs]
    held = [record for record in records if record.pair_id in held_pairs]
    train_scenes = {record.scene_id for record in train}
    held_scenes = {record.scene_id for record in held}
    if not train or not held:
        raise ValueError("Deterministic split produced an empty partition")
    if train_scenes & held_scenes:
        raise RuntimeError("Historical grounding split is not scene-disjoint")
    for pair in pairs:
        destinations = {
            "held" if record in held else "train"
            for record in records
            if record.pair_id == pair
        }
        if len(destinations) != 1:
            raise RuntimeError(f"Counterfactual pair {pair} crossed the internal split")
    audit = {
        "algorithm": "lexicographic_pair_index_modulo",
        "modulo": modulo,
        "held_remainder": held_remainder,
        "pair_disjoint": True,
        "scene_disjoint": True,
        "train_pairs": sorted({record.pair_id for record in train}),
        "held_pairs": sorted(held_pairs),
        "train_scenes": sorted(train_scenes),
        "held_scenes": sorted(held_scenes),
        "train_grounded_rows": len(train),
        "held_grounded_rows": len(held),
    }
    return train, held, audit


class QuestionIndependentPrefixStore:
    """Verified reader for sanitized question-independent continuous prefixes."""

    def __init__(self, directory: str | Path, *, latent_count: int, scene_dim: int) -> None:
        self.directory = Path(directory).expanduser().resolve()
        manifest_path = self.directory / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Prefix-cache manifest not found: {manifest_path}")
        self.manifest_path = manifest_path
        self.manifest_sha256 = sha256_file(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("artifact") != "question_independent_scene_prefix_cache_v1":
            raise ValueError("Unsupported scene-prefix cache artifact")
        if manifest.get("complete_scene_prefixes") is not True:
            raise ValueError("Scene-prefix cache is not complete")
        if manifest.get("question_inputs_used") is not False:
            raise ValueError("Scene-prefix cache was influenced by questions")
        if manifest.get("question_dependent_scene_retrieval") is not False:
            raise ValueError("Scene-prefix cache used question-dependent retrieval")
        if manifest.get("environmental_text_inputs") != []:
            raise ValueError("Scene-prefix cache declares environmental text inputs")
        scenes = manifest.get("scenes")
        if not isinstance(scenes, dict) or not scenes:
            raise ValueError("Scene-prefix cache has no scene records")
        self._scenes = scenes
        self.source_base_checkpoint_sha256 = str(manifest.get("base_checkpoint_sha256", ""))
        self.source_runtime_config_sha256 = str(
            manifest.get("base_runtime_config_sha256", "")
        )
        for value, label in (
            (self.source_base_checkpoint_sha256, "base checkpoint"),
            (self.source_runtime_config_sha256, "runtime config"),
        ):
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise ValueError(f"Prefix-cache {label} identity is invalid")
        self.latent_count = int(latent_count)
        self.scene_dim = int(scene_dim)
        self._cache: dict[str, torch.Tensor] = {}

    def load(self, scene_id: str) -> torch.Tensor:
        cached = self._cache.get(scene_id)
        if cached is not None:
            return cached
        record = self._scenes.get(scene_id)
        if not isinstance(record, dict):
            raise KeyError(f"Scene is absent from the prefix cache: {scene_id}")
        filename = record.get("filename")
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise ValueError(f"Unsafe prefix-cache filename for {scene_id}")
        source = self.directory / filename
        expected_sha = record.get("file_sha256")
        if not isinstance(expected_sha, str) or sha256_file(source) != expected_sha:
            raise ValueError(f"Prefix-cache file hash mismatch for {scene_id}")
        tensors = load_file(str(source), device="cpu")
        if set(tensors) != {"scene_prefix"}:
            raise ValueError(f"Unexpected tensors in prefix cache for {scene_id}")
        prefix = tensors["scene_prefix"]
        expected = (1, self.latent_count + 2, self.scene_dim)
        if tuple(prefix.shape) != expected:
            raise ValueError(
                f"Prefix shape mismatch for {scene_id}: {tuple(prefix.shape)} != {expected}"
            )
        if not torch.isfinite(prefix.float()).all():
            raise ValueError(f"Prefix contains NaN or infinity for {scene_id}")
        # The two boundary embeddings contain no scene slot.  Every one of the
        # 256 intervening scene tokens remains present and is scored below.
        scene_tokens = prefix[0, 1:-1].float().contiguous()
        self._cache[scene_id] = scene_tokens
        return scene_tokens


def _group_indices(records: Sequence[GroundingRecord]) -> dict[str, list[int]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        grouped[record.scene_id].append(index)
    return dict(sorted(grouped.items()))


def _prediction_metrics(
    records: Sequence[GroundingRecord], predictions: torch.Tensor
) -> dict[str, Any]:
    if tuple(predictions.shape) != (len(records), 3):
        raise ValueError("Grounding predictions have the wrong shape")
    target = torch.tensor([record.target_xyz for record in records], dtype=torch.float32)
    errors = torch.linalg.vector_norm(predictions.float().cpu() - target, dim=-1)
    return {
        "count": len(records),
        "mean_coordinate_error_m": float(errors.mean()),
        "median_coordinate_error_m": float(errors.median()),
        "rmse_m": float(errors.square().mean().sqrt()),
        "within_0_50m_accuracy": float((errors <= 0.5).float().mean()),
        "within_1m_accuracy": float((errors <= 1.0).float().mean()),
        "maximum_coordinate_error_m": float(errors.max()),
    }


def _predict(
    model: GroundingSidecarV78,
    records: Sequence[GroundingRecord],
    question_embeddings: torch.Tensor,
    prefix_store: QuestionIndependentPrefixStore,
    room_min: torch.Tensor,
    room_max: torch.Tensor,
    *,
    scene_override: Sequence[str] | None = None,
    token_permutation: torch.Tensor | None = None,
    zero_scene: bool = False,
) -> tuple[torch.Tensor, dict[str, float]]:
    if question_embeddings.shape != (len(records), model.scene_dim):
        raise ValueError("Question-embedding count or dimension mismatch")
    if scene_override is not None and len(scene_override) != len(records):
        raise ValueError("Scene override must have one entry per record")
    outputs = torch.empty((len(records), 3), dtype=torch.float32)
    minimum_weight = math.inf
    maximum_weight = 0.0
    model.eval()
    with torch.inference_mode():
        for index, record in enumerate(records):
            selected_scene = record.scene_id if scene_override is None else scene_override[index]
            tokens = prefix_store.load(selected_scene)
            if token_permutation is not None:
                tokens = tokens[token_permutation]
            if zero_scene:
                tokens = torch.zeros_like(tokens)
            normalized, _, weights = model(
                question_embeddings[index : index + 1], tokens.unsqueeze(0)
            )
            outputs[index] = denormalize_xyz(normalized, room_min, room_max)[0].cpu()
            minimum_weight = min(minimum_weight, float(weights.min()))
            maximum_weight = max(maximum_weight, float(weights.max()))
    return outputs, {
        "minimum_attention_weight": minimum_weight,
        "maximum_attention_weight": maximum_weight,
        "every_scene_token_positive_weight": minimum_weight > 0.0,
    }


def _load_v54_training_baseline(
    path: str | Path, records: Sequence[GroundingRecord]
) -> tuple[torch.Tensor, str]:
    source = _validate_historical_training_path(path, label="V54 grounding baseline")
    indexed: dict[tuple[str, str], tuple[float, float, float]] = {}
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        xyz = raw.get("grounding_xyz")
        key = (raw.get("scene_id"), raw.get("question_id"))
        if (
            not all(isinstance(item, str) for item in key)
            or not isinstance(xyz, list)
            or len(xyz) != 3
        ):
            raise ValueError("Malformed V54 historical grounding baseline")
        indexed[(str(key[0]), str(key[1]))] = tuple(float(value) for value in xyz)
    missing = [
        (record.scene_id, record.question_id)
        for record in records
        if (record.scene_id, record.question_id) not in indexed
    ]
    if missing:
        raise ValueError(f"V54 baseline is missing {len(missing)} historical rows")
    tensor = torch.tensor(
        [indexed[(record.scene_id, record.question_id)] for record in records],
        dtype=torch.float32,
    )
    return tensor, sha256_file(source)


def _paired_scene_causal_metrics(
    records: Sequence[GroundingRecord],
    correct_predictions: torch.Tensor,
    paired_scene_predictions: torch.Tensor,
) -> dict[str, Any]:
    counterpart = {
        (record.pair_id, record.question_key, record.scene_id): record for record in records
    }
    rows: list[tuple[int, GroundingRecord]] = []
    for index, record in enumerate(records):
        other = counterpart.get((record.pair_id, record.question_key, record.paired_scene_id))
        if other is not None and math.dist(record.target_xyz, other.target_xyz) > 1e-5:
            rows.append((index, other))
    if not rows:
        return {"changed_target_sides": 0}
    correct_original: list[float] = []
    paired_original: list[float] = []
    paired_target: list[float] = []
    for index, other in rows:
        original = torch.tensor(records[index].target_xyz)
        changed = torch.tensor(other.target_xyz)
        correct_original.append(float(torch.linalg.vector_norm(correct_predictions[index] - original)))
        paired_original.append(
            float(torch.linalg.vector_norm(paired_scene_predictions[index] - original))
        )
        paired_target.append(float(torch.linalg.vector_norm(paired_scene_predictions[index] - changed)))
    return {
        "changed_target_sides": len(rows),
        "correct_scene_mean_error_to_original_target_m": float(np.mean(correct_original)),
        "paired_scene_mean_error_to_original_target_m": float(np.mean(paired_original)),
        "paired_scene_mean_error_to_paired_target_m": float(np.mean(paired_target)),
        "correct_scene_closer_to_original_fraction": float(
            np.mean(np.asarray(correct_original) < np.asarray(paired_original))
        ),
        "paired_scene_follows_paired_target_fraction": float(
            np.mean(np.asarray(paired_target) < np.asarray(paired_original))
        ),
    }


def _save_candidate(
    directory: Path,
    model: GroundingSidecarV78,
    *,
    prefix_manifest_sha256: str,
    model_id: str,
    model_revision: str,
    room_min: Sequence[float],
    room_max: Sequence[float],
    seed: int,
) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=True)
    unexpected = {path.name for path in directory.iterdir()} - EXPECTED_CANDIDATE_FILES
    if unexpected:
        raise ValueError(f"Candidate directory contains unexpected files: {sorted(unexpected)}")
    weights = directory / WEIGHTS_FILENAME
    temporary = weights.with_name(f".{weights.name}.tmp-{os.getpid()}")
    state = {
        key: value.detach().float().cpu().contiguous()
        for key, value in sorted(model.state_dict().items())
    }
    # Keep the safetensors payload purely numeric.  PySafeTensors serializes
    # its optional string metadata through a hash map whose byte order is not
    # guaranteed across processes; omitting it makes repeated seeded runs
    # byte-identical while the external sealed JSON carries the contract.
    save_file(state, temporary)
    temporary.replace(weights)
    metadata = {
        "schema_version": 1,
        "artifact": ARTIFACT,
        "architecture": ARCHITECTURE,
        "weights_sha256": sha256_file(weights),
        "scene_dim": model.scene_dim,
        "scene_latent_count": model.latent_count,
        "question_adapter_rank": model.rank,
        "coordinate_hidden_dim": model.hidden_dim,
        "maximum_residual": model.maximum_residual,
        "room_min_m": [float(value) for value in room_min],
        "room_max_m": [float(value) for value in room_max],
        "model_id": model_id,
        "model_revision": model_revision,
        "embedding_tensor_key": GEMMA4_TOKEN_EMBEDDING_KEY,
        "source_prefix_manifest_sha256": prefix_manifest_sha256,
        "initialization_seed": int(seed),
        "all_scene_tokens_scored": True,
        "positive_softmax_attention": True,
        "question_dependent_scene_retrieval": False,
        "question_only_coordinate_path_exists": False,
        "zero_scene_produces_exact_room_center": True,
        "answer_text_serialized": False,
        "question_text_serialized": False,
        "target_coordinates_serialized": False,
        "object_ids_serialized": False,
        "environmental_text_inputs": [],
        "oracle_runtime_loaded": False,
        "training_metadata_runtime_loaded": False,
        "official_validation_loaded": False,
        "official_test_loaded": False,
        "runtime_promotion_authorized": False,
    }
    _atomic_json(directory / METADATA_FILENAME, metadata)
    return validate_candidate(directory)


def validate_candidate(directory: str | Path) -> dict[str, Any]:
    root = Path(directory).expanduser().resolve()
    files = {path.name for path in root.iterdir() if path.is_file()}
    if files != EXPECTED_CANDIDATE_FILES:
        raise ValueError(f"V78 candidate must contain exactly {sorted(EXPECTED_CANDIDATE_FILES)}")
    metadata = json.loads((root / METADATA_FILENAME).read_text(encoding="utf-8"))
    if metadata.get("artifact") != ARTIFACT or metadata.get("architecture") != ARCHITECTURE:
        raise ValueError("V78 candidate metadata contract mismatch")
    if metadata.get("weights_sha256") != sha256_file(root / WEIGHTS_FILENAME):
        raise ValueError("V78 candidate weights hash mismatch")
    prohibited_true = {
        "answer_text_serialized",
        "question_text_serialized",
        "target_coordinates_serialized",
        "object_ids_serialized",
        "oracle_runtime_loaded",
        "training_metadata_runtime_loaded",
        "official_validation_loaded",
        "official_test_loaded",
        "runtime_promotion_authorized",
    }
    if any(metadata.get(key) is not False for key in prohibited_true):
        raise ValueError("V78 diagnostic metadata permits a prohibited runtime input or promotion")
    if metadata.get("environmental_text_inputs") != []:
        raise ValueError("V78 candidate contains environmental text inputs")
    for field in (
        "weights_sha256",
        "source_prefix_manifest_sha256",
    ):
        value = metadata.get(field)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"V78 candidate {field} identity is invalid")
    model = GroundingSidecarV78(
        scene_dim=int(metadata["scene_dim"]),
        latent_count=int(metadata["scene_latent_count"]),
        rank=int(metadata["question_adapter_rank"]),
        hidden_dim=int(metadata["coordinate_hidden_dim"]),
        maximum_residual=float(metadata["maximum_residual"]),
    )
    tensors = load_file(str(root / WEIGHTS_FILENAME), device="cpu")
    if set(tensors) != set(model.state_dict()):
        raise ValueError("V78 candidate tensor schema mismatch")
    model.load_state_dict(tensors, strict=True)
    if not all(torch.isfinite(value).all() for value in tensors.values()):
        raise ValueError("V78 candidate contains NaN or infinity")
    return {
        "directory": str(root),
        "files": sorted(files),
        "weights_sha256": metadata["weights_sha256"],
        "metadata_sha256": sha256_file(root / METADATA_FILENAME),
        "metadata": metadata,
    }


def run_training(
    *,
    qa_path: str | Path,
    prefix_directory: str | Path,
    baseline_path: str | Path,
    candidate_directory: str | Path,
    report_path: str | Path,
    base_checkpoint: str | Path = DEFAULT_BASE_CHECKPOINT,
    runtime_config: str | Path = DEFAULT_RUNTIME_CONFIG,
    model_snapshot: str | Path | None = None,
    model_id: str = DEFAULT_MODEL_ID,
    model_revision: str = DEFAULT_MODEL_REVISION,
    epochs: int = 120,
    learning_rate: float = 2e-3,
    weight_decay: float = 1e-4,
    anchor_weight: float = 0.15,
    rank: int = 64,
    hidden_dim: int = 256,
    maximum_residual: float = 0.5,
    seed: int = 78078,
) -> dict[str, Any]:
    if epochs < 1 or learning_rate <= 0 or weight_decay < 0 or anchor_weight < 0:
        raise ValueError("Invalid V78 optimization settings")
    started = time.perf_counter()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    records = load_historical_grounding_records(qa_path)
    train, held, split_audit = pair_disjoint_internal_split(records)
    prefix_store = QuestionIndependentPrefixStore(
        prefix_directory, latent_count=256, scene_dim=GEMMA4_PROJECTED_DIM
    )
    for scene_id in sorted({record.scene_id for record in records}):
        prefix_store.load(scene_id)
    base_checkpoint_sha256, _ = checkpoint_fingerprint(base_checkpoint)
    runtime_config_value = load_runtime_config(runtime_config)
    base_runtime_config_sha256 = effective_runtime_config_sha256(runtime_config_value)
    if prefix_store.source_runtime_config_sha256 != base_runtime_config_sha256:
        raise ValueError(
            "Historical prefix cache and selected runtime config have different identities"
        )

    snapshot = resolve_local_snapshot(model_id, model_revision, model_snapshot)
    questions = [record.question for record in records]
    embeddings_np, embedding_audit = load_category_embeddings_selective(snapshot, questions)
    all_embeddings = torch.from_numpy(embeddings_np).float()
    embedding_by_key = {
        (record.scene_id, record.question_id): all_embeddings[index]
        for index, record in enumerate(records)
    }
    train_embeddings = torch.stack(
        [embedding_by_key[(record.scene_id, record.question_id)] for record in train]
    )
    held_embeddings = torch.stack(
        [embedding_by_key[(record.scene_id, record.question_id)] for record in held]
    )
    room_min = torch.tensor(DEFAULT_ROOM_MIN, dtype=torch.float32)
    room_max = torch.tensor(DEFAULT_ROOM_MAX, dtype=torch.float32)
    train_targets_metric = torch.tensor(
        [record.target_xyz for record in train], dtype=torch.float32
    )
    train_targets = normalize_xyz(train_targets_metric, room_min, room_max)
    model = GroundingSidecarV78(
        rank=rank, hidden_dim=hidden_dim, maximum_residual=maximum_residual
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    grouped = _group_indices(train)
    epoch_losses: list[float] = []
    for _epoch in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_loss = 0.0
        for scene_id, indices in grouped.items():
            question = train_embeddings[indices]
            scene = prefix_store.load(scene_id).unsqueeze(0).expand(len(indices), -1, -1)
            predicted, logits, _ = model(question, scene)
            target = train_targets[indices]
            anchor_target = model.nearest_anchor_targets(target)
            loss = F.smooth_l1_loss(predicted, target) + anchor_weight * F.cross_entropy(
                logits, anchor_target
            )
            (loss / len(grouped)).backward()
            epoch_loss += float(loss.detach())
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        epoch_losses.append(epoch_loss / len(grouped))

    train_prediction, train_attention = _predict(
        model, train, train_embeddings, prefix_store, room_min, room_max
    )
    held_prediction, held_attention = _predict(
        model, held, held_embeddings, prefix_store, room_min, room_max
    )
    baseline_prediction, baseline_sha256 = _load_v54_training_baseline(baseline_path, held)
    paired_prediction, _ = _predict(
        model,
        held,
        held_embeddings,
        prefix_store,
        room_min,
        room_max,
        scene_override=[record.paired_scene_id for record in held],
    )
    generator = torch.Generator(device="cpu").manual_seed(seed + 1)
    permutation = torch.randperm(model.latent_count, generator=generator)
    shuffled_prediction, _ = _predict(
        model,
        held,
        held_embeddings,
        prefix_store,
        room_min,
        room_max,
        token_permutation=permutation,
    )
    rolled_embeddings = held_embeddings.roll(shifts=1, dims=0)
    question_shuffle_prediction, _ = _predict(
        model, held, rolled_embeddings, prefix_store, room_min, room_max
    )
    zero_prediction, zero_attention = _predict(
        model,
        held,
        held_embeddings,
        prefix_store,
        room_min,
        room_max,
        zero_scene=True,
    )
    room_center = ((room_min + room_max) * 0.5).expand_as(zero_prediction)
    zero_exact = bool(torch.equal(zero_prediction, room_center))

    candidate_audit = _save_candidate(
        Path(candidate_directory).expanduser().resolve(),
        model,
        prefix_manifest_sha256=prefix_store.manifest_sha256,
        model_id=model_id,
        model_revision=model_revision,
        room_min=DEFAULT_ROOM_MIN,
        room_max=DEFAULT_ROOM_MAX,
        seed=seed,
    )
    baseline_metrics = _prediction_metrics(held, baseline_prediction)
    held_metrics = _prediction_metrics(held, held_prediction)
    report: dict[str, Any] = {
        "schema_version": 1,
        "artifact": "v78_historical_training_grounding_repair_report_v1",
        "status": "internal_historical_diagnostic_only",
        "runtime_promotion_authorized": False,
        "official_validation_loaded": False,
        "official_test_loaded": False,
        "deferred_final_loaded": False,
        "oracle_files_loaded": False,
        "environmental_text_inputs": [],
        "training_source": {
            "sha256": sha256_file(qa_path),
            "grounded_rows": len(records),
            "answers_used": False,
            "target_instance_ids_used": False,
            "target_xyz_training_only": True,
        },
        "split": split_audit,
        "continuous_scene_source": {
            "manifest_sha256": prefix_store.manifest_sha256,
            "question_independent": True,
            "complete_scene_prefixes": True,
            "scene_latent_count": model.latent_count,
            "scene_dim": model.scene_dim,
            "every_scene_token_scored": True,
            "source_base_checkpoint_sha256": (
                prefix_store.source_base_checkpoint_sha256
            ),
            "selected_runtime_base_checkpoint_sha256": base_checkpoint_sha256,
            "base_runtime_config_sha256": base_runtime_config_sha256,
        },
        "question_embedding_source": {
            "model_id": model_id,
            "model_revision": model_revision,
            "tensor_key": GEMMA4_TOKEN_EMBEDDING_KEY,
            "full_language_model_loaded": False,
            "selective_embedding_rows_only": True,
            "unique_token_rows_read": int(embedding_audit["unique_token_rows_read"]),
            "question_text_serialized_in_candidate": False,
        },
        "optimization": {
            "seed": seed,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "anchor_weight": anchor_weight,
            "gradient_clip_norm": 1.0,
            "device": "cpu",
            "train_loss_initial": epoch_losses[0],
            "train_loss_final": epoch_losses[-1],
            "train_loss_minimum": min(epoch_losses),
            "held_partition_used_for_optimization": False,
        },
        "candidate": candidate_audit,
        "metrics": {
            "historical_internal_train": _prediction_metrics(train, train_prediction),
            "historical_internal_held": held_metrics,
            "v54_same_historical_internal_held": baseline_metrics,
            "zero_scene_same_historical_internal_held": _prediction_metrics(
                held, zero_prediction
            ),
            "scene_token_position_shuffle": _prediction_metrics(
                held, shuffled_prediction
            ),
            "question_embedding_shuffle": _prediction_metrics(
                held, question_shuffle_prediction
            ),
            "paired_wrong_scene": _prediction_metrics(held, paired_prediction),
            "paired_scene_causal": _paired_scene_causal_metrics(
                held, held_prediction, paired_prediction
            ),
        },
        "attention_audit": {
            "train": train_attention,
            "held": held_attention,
            "zero_scene": zero_attention,
            "zero_scene_exact_room_center": zero_exact,
            "question_only_coordinate_path_exists": False,
        },
        "baseline_predictions_sha256": baseline_sha256,
        "material_internal_improvement": bool(
            held_metrics["mean_coordinate_error_m"]
            <= baseline_metrics["mean_coordinate_error_m"] * 0.75
            and held_metrics["within_1m_accuracy"]
            >= baseline_metrics["within_1m_accuracy"] + 0.25
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "limitations": [
            "This is a historical train-pool, scene-disjoint internal evaluation, not official validation.",
            "The candidate is not wired into chat and is not authorized for runtime promotion.",
            "Grounding uses the full cached scene prefix rather than raw voxel-resolution features.",
        ],
    }
    _atomic_json(Path(report_path).expanduser().resolve(), report)
    return report


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--qa",
        default="data_gemma4/training/v62_pair_disjoint/train.jsonl",
    )
    parser.add_argument(
        "--prefix-cache",
        default="data_gemma4/scene_tokens/v62_pair_disjoint_train_prefixes",
    )
    parser.add_argument(
        "--baseline",
        default="reports/gemma4/predictions/v65_v54_training_natural.jsonl",
    )
    parser.add_argument(
        "--candidate",
        default="reports/gemma4/artifacts/v78_grounding_sidecar_diagnostic",
    )
    parser.add_argument(
        "--report",
        default="reports/gemma4/metrics/v78_grounding_sidecar_internal_held.json",
    )
    parser.add_argument("--base-checkpoint", default=DEFAULT_BASE_CHECKPOINT)
    parser.add_argument("--runtime-config", default=DEFAULT_RUNTIME_CONFIG)
    parser.add_argument("--model-snapshot")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--anchor-weight", type=float, default=0.15)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--maximum-residual", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=78078)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = run_training(
        qa_path=args.qa,
        prefix_directory=args.prefix_cache,
        baseline_path=args.baseline,
        candidate_directory=args.candidate,
        report_path=args.report,
        base_checkpoint=args.base_checkpoint,
        runtime_config=args.runtime_config,
        model_snapshot=args.model_snapshot,
        model_id=args.model_id,
        model_revision=args.model_revision,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        anchor_weight=args.anchor_weight,
        rank=args.rank,
        hidden_dim=args.hidden_dim,
        maximum_residual=args.maximum_residual,
        seed=args.seed,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


__all__ = [
    "ARCHITECTURE",
    "ARTIFACT",
    "GroundingRecord",
    "GroundingSidecarV78",
    "QuestionIndependentPrefixStore",
    "load_historical_grounding_records",
    "main",
    "pair_disjoint_internal_split",
    "run_training",
    "validate_candidate",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
