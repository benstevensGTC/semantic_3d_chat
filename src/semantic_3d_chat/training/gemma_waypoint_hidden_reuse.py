"""Exact-input reuse for frozen-Gemma waypoint hidden-state caches."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import torch

from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.robot.waypoint_history import (
    HISTORY_FEATURE_DIM_V1,
    HISTORY_FEATURE_DIM_V2,
    HISTORY_PARAMETERIZATION_V1,
    HISTORY_PARAMETERIZATION_V2,
)
from semantic_3d_chat.training.gemma_waypoint_policy import (
    ActualGemmaWaypointForward,
    WaypointTraceDataset,
    WaypointTraceSample,
    cache_actual_gemma_decision_hidden,
    load_waypoint_trace_jsonl,
)

_LEGACY_MANIFEST_SCHEMA = "semantic_3d_chat.gemma_waypoint_trace_dataset.v1"
_SUPPORTED_HISTORY_CONTRACTS = {
    HISTORY_PARAMETERIZATION_V1: HISTORY_FEATURE_DIM_V1,
    HISTORY_PARAMETERIZATION_V2: HISTORY_FEATURE_DIM_V2,
}


def validate_forward_revalidation_destination(
    reuse_cache: str | Path, output: str | Path
) -> Path:
    """Require a fresh migration destination distinct from its source cache."""

    source_candidate = Path(reuse_cache).expanduser()
    source = Path(
        os.path.abspath(
            source_candidate
            if source_candidate.is_absolute()
            else PROJECT_ROOT / source_candidate
        )
    )
    output_candidate = Path(output).expanduser()
    destination = Path(
        os.path.abspath(
            output_candidate
            if output_candidate.is_absolute()
            else PROJECT_ROOT / output_candidate
        )
    )
    if destination == source:
        raise ValueError("forward revalidation output must differ from reuse cache")
    if destination.exists():
        raise ValueError("forward revalidation output must not already exist")
    return destination


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _unique_json(path: Path) -> dict[str, object]:
    def unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate legacy waypoint manifest field: {key}")
            result[key] = value
        return result

    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique)
    if not isinstance(value, dict):
        raise TypeError("Legacy waypoint reuse manifest must be a JSON object")
    return value


def load_legacy_waypoint_dataset_for_hidden_reuse(
    source: str | Path,
    *,
    state_dim: int,
    history_dim: int,
    history_parameterization: str = HISTORY_PARAMETERIZATION_V1,
    max_history_tokens: int,
    max_waypoint_step_m: float,
) -> WaypointTraceDataset:
    """Authenticate a pre-canonical dataset solely for exact-input cache reuse.

    Normal training loading requires the canonical history parameterization.
    This deliberately narrower path accepts only historical manifests where
    that field is absent, authenticates the complete legacy manifest and trace
    bytes, and then exposes samples only to the exact frozen-input hash gate.
    It must never be used as the active training dataset loader.
    """

    if (
        history_parameterization != HISTORY_PARAMETERIZATION_V1
        or history_dim != HISTORY_FEATURE_DIM_V1
    ):
        raise ValueError(
            "Legacy waypoint reuse history parameterization differs from target"
        )
    candidate = Path(source).expanduser()
    rooted = candidate if candidate.is_absolute() else PROJECT_ROOT / candidate
    root = Path(os.path.abspath(rooted))
    if "training" not in {part.casefold() for part in root.parts}:
        raise ValueError("Legacy waypoint reuse source must remain in a training tree")
    current = Path(root.anchor)
    for component in root.parts[1:]:
        current /= component
        if current.is_symlink():
            raise ValueError("Legacy waypoint reuse path cannot contain symlinks")
    manifest_path = root / "manifest.json"
    traces_path = root / "traces.jsonl"
    if (
        not root.is_dir()
        or {entry.name for entry in root.iterdir()} != {"manifest.json", "traces.jsonl"}
        or any(path.is_symlink() or not path.is_file() for path in (manifest_path, traces_path))
    ):
        raise ValueError("Legacy waypoint reuse dataset must contain two regular files")
    manifest = _unique_json(manifest_path)
    body = {key: value for key, value in manifest.items() if key != "dataset_sha256"}
    train_scenes = manifest.get("train_scene_ids")
    validation_scenes = manifest.get("validation_scene_ids")
    if (
        manifest.get("schema") != _LEGACY_MANIFEST_SCHEMA
        or "history_parameterization" in manifest
        or manifest.get("dataset_sha256") != _canonical_sha256(body)
        or manifest.get("traces_sha256") != _sha256_file(traces_path)
        or manifest.get("scene_splits_disjoint") is not True
        or manifest.get("runtime_compatible") is not False
        or manifest.get("runtime_must_block_parent_tree") is not True
        or manifest.get("environmental_text_training_only") is not True
        or manifest.get("expert_planners_available_at_runtime") is not False
        or manifest.get("oracle_inputs_at_runtime") is not False
        or manifest.get("runtime_preprogrammed_lap_function") is not False
        or manifest.get("action_names") != ["MOVE_TO", "FACE", "STOP"]
        or manifest.get("state_feature_dim") != state_dim
        or manifest.get("history_feature_dim") != history_dim
        or manifest.get("history_length") != max_history_tokens
        or manifest.get("max_waypoint_step_m") != max_waypoint_step_m
        or not isinstance(train_scenes, list)
        or not train_scenes
        or not isinstance(validation_scenes, list)
        or not validation_scenes
        or set(train_scenes) & set(validation_scenes)
    ):
        raise ValueError("Legacy waypoint reuse manifest authentication failed")
    loaded = load_waypoint_trace_jsonl(
        traces_path,
        state_dim=state_dim,
        history_dim=history_dim,
        history_parameterization=history_parameterization,
        max_history_tokens=max_history_tokens,
        max_waypoint_step_m=max_waypoint_step_m,
    )
    if (
        manifest.get("sample_count") != len(loaded.samples)
        or loaded.traces_sha256 != manifest.get("traces_sha256")
        or list(loaded.scene_splits.get("train", ())) != train_scenes
        or list(loaded.scene_splits.get("validation", ())) != validation_scenes
    ):
        raise ValueError("Legacy waypoint reuse trace rows differ from their manifest")
    return WaypointTraceDataset(
        samples=loaded.samples,
        sha256=str(manifest["dataset_sha256"]),
        traces_sha256=loaded.traces_sha256,
        state_dim=loaded.state_dim,
        history_dim=loaded.history_dim,
        history_parameterization=history_parameterization,
    )


def load_waypoint_dataset_for_hidden_reuse(
    source: str | Path,
    *,
    state_dim: int,
    history_dim: int,
    history_parameterization: str = HISTORY_PARAMETERIZATION_V1,
    max_history_tokens: int,
    max_waypoint_step_m: float,
) -> WaypointTraceDataset:
    """Authenticate either a canonical or pre-canonical exact-reuse source.

    Canonical datasets go through the active strict generator loader before
    their numeric training view is created. Historical datasets are accepted
    only by the deliberately narrow missing-parameterization loader above.
    In both cases the downstream reuse key remains the exact frozen-Gemma
    scene/instruction/state/history fingerprint.
    """

    if _SUPPORTED_HISTORY_CONTRACTS.get(history_parameterization) != history_dim:
        raise ValueError("Waypoint reuse history dimension/parameterization pair differs")
    candidate = Path(source).expanduser()
    rooted = candidate if candidate.is_absolute() else PROJECT_ROOT / candidate
    root = Path(os.path.abspath(rooted))
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ValueError("Waypoint reuse source has no regular manifest")
    manifest = _unique_json(manifest_path)
    parameterization = manifest.get("history_parameterization")
    if parameterization is None:
        if history_parameterization != HISTORY_PARAMETERIZATION_V1:
            raise ValueError("Waypoint reuse history parameterization differs")
        return load_legacy_waypoint_dataset_for_hidden_reuse(
            root,
            state_dim=state_dim,
            history_dim=history_dim,
            history_parameterization=history_parameterization,
            max_history_tokens=max_history_tokens,
            max_waypoint_step_m=max_waypoint_step_m,
        )
    if parameterization != history_parameterization:
        raise ValueError("Waypoint reuse history parameterization differs")

    # Local import avoids making the ordinary hidden-cache module part of the
    # trace generator's dependency surface.
    from semantic_3d_chat.training.gemma_waypoint_trace_generator import (
        load_gemma_waypoint_trace_dataset,
    )

    strict_manifest, strict_rows = load_gemma_waypoint_trace_dataset(root)
    loaded = load_waypoint_trace_jsonl(
        root,
        state_dim=state_dim,
        history_dim=history_dim,
        history_parameterization=history_parameterization,
        max_history_tokens=max_history_tokens,
        max_waypoint_step_m=max_waypoint_step_m,
    )
    if (
        strict_manifest.get("sample_count") != len(strict_rows)
        or len(loaded.samples) != len(strict_rows)
        or loaded.sha256 != strict_manifest.get("dataset_sha256")
        or loaded.traces_sha256 != strict_manifest.get("traces_sha256")
        or strict_manifest.get("history_feature_dim") != history_dim
        or strict_manifest.get("history_parameterization")
        != history_parameterization
    ):
        raise ValueError("Canonical waypoint reuse views differ after authentication")
    return loaded


def frozen_gemma_input_sha256(sample: WaypointTraceSample) -> str:
    """Hash exactly the sample fields consumed by the frozen Gemma forward."""

    digest = hashlib.sha256()
    for value in (sample.scene_id, sample.instruction):
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    for tensor in (sample.state, sample.history):
        value = tensor.detach().float().cpu().contiguous()
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode())
        digest.update(b"\0")
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def reusable_hidden_rows(
    samples: Sequence[WaypointTraceSample], hidden: torch.Tensor
) -> Mapping[str, torch.Tensor]:
    """Index authenticated hidden rows by exact frozen-Gemma input."""

    if hidden.ndim != 2 or hidden.shape[0] != len(samples) or hidden.shape[1] < 1:
        raise ValueError("Reusable Gemma hidden rows differ from their sample order")
    result: dict[str, torch.Tensor] = {}
    for sample, row in zip(samples, hidden, strict=True):
        key = frozen_gemma_input_sha256(sample)
        previous = result.get(key)
        if previous is not None and not torch.equal(previous, row):
            raise RuntimeError("Identical Gemma inputs have different cached hidden states")
        result[key] = row.detach().float().cpu().clone()
    return result


def assemble_hidden_with_reuse(
    runner: ActualGemmaWaypointForward,
    cache: object,
    samples: Sequence[WaypointTraceSample],
    reusable: Mapping[str, torch.Tensor],
    *,
    forward_chunk_size: int = 64,
    gemma_batch_size: int = 1,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[torch.Tensor, int, int]:
    """Assemble target order, forwarding only exact-input cache misses."""

    if (
        isinstance(forward_chunk_size, bool)
        or not isinstance(forward_chunk_size, int)
        or not 1 <= forward_chunk_size <= 1024
    ):
        raise ValueError("forward_chunk_size must be an integer in [1,1024]")
    if (
        isinstance(gemma_batch_size, bool)
        or not isinstance(gemma_batch_size, int)
        or not 1 <= gemma_batch_size <= 16
    ):
        raise ValueError("gemma_batch_size must be an integer in [1,16]")

    rows: list[torch.Tensor | None] = []
    missing: list[WaypointTraceSample] = []
    missing_positions: list[int] = []
    reused = 0
    for position, sample in enumerate(samples):
        previous = reusable.get(frozen_gemma_input_sha256(sample))
        if previous is None:
            rows.append(None)
            missing.append(sample)
            missing_positions.append(position)
        else:
            rows.append(previous.detach().float().cpu().clone())
            reused += 1
    for start in range(0, len(missing), forward_chunk_size):
        stop = min(start + forward_chunk_size, len(missing))
        if gemma_batch_size == 1:
            computed = cache_actual_gemma_decision_hidden(
                runner,
                cache,
                missing[start:stop],
            )
        else:
            computed = cache_actual_gemma_decision_hidden(
                runner,
                cache,
                missing[start:stop],
                forward_batch_size=gemma_batch_size,
            )
        for position, row in zip(missing_positions[start:stop], computed, strict=True):
            rows[position] = row.detach().float().cpu()
        if progress is not None:
            progress(stop, len(missing))
    if any(row is None for row in rows):
        raise RuntimeError("Gemma hidden cache assembly left an empty row")
    combined = torch.stack([row for row in rows if row is not None])
    return combined, reused, len(missing)


def _stratified_revalidation_indices(
    samples: Sequence[WaypointTraceSample], sample_count: int
) -> tuple[int, ...]:
    if (
        isinstance(sample_count, bool)
        or not isinstance(sample_count, int)
        or sample_count < 1
    ):
        raise ValueError("forward revalidation sample_count must be positive")
    if len(samples) <= sample_count:
        return tuple(range(len(samples)))
    keyed = [
        (frozen_gemma_input_sha256(sample), sample.sample_id, index)
        for index, sample in enumerate(samples)
    ]
    mandatory: set[int] = set()
    for grouping in (
        lambda sample: ("action", sample.action_index),
        lambda sample: ("history_length", int(sample.history.shape[0])),
        lambda sample: (
            "instruction",
            hashlib.sha256(sample.instruction.encode("utf-8")).hexdigest(),
        ),
        lambda sample: ("scene", sample.scene_id),
    ):
        buckets: dict[object, list[tuple[str, str, int]]] = {}
        for descriptor, sample in zip(keyed, samples, strict=True):
            buckets.setdefault(grouping(sample), []).append(descriptor)
        mandatory.update(min(values)[2] for values in buckets.values())
    if len(mandatory) > sample_count:
        raise ValueError(
            "forward revalidation sample_count cannot cover "
            "action/history/instruction/scene strata"
        )
    selected = set(mandatory)
    for _digest, _sample_id, index in sorted(keyed):
        if len(selected) >= sample_count:
            break
        selected.add(index)
    return tuple(sorted(selected))


def _source_order_revalidation_batches(
    samples: Sequence[WaypointTraceSample],
    target_indices: Sequence[int],
    *,
    gemma_batch_size: int,
) -> tuple[tuple[int, ...], ...]:
    """Recover every full source-order batch containing an audit target.

    MPS bfloat16 kernels can be bit-stable for a fixed batch shape while
    producing a different last-bit result when an otherwise independent row is
    forwarded alone.  A cache contract therefore has to be checked in the same
    deterministic batching context that produced it.  Companion rows are part
    of that context and are deliberately returned for strict comparison too.
    """

    selected = set(target_indices)
    if not selected or any(index < 0 or index >= len(samples) for index in selected):
        raise ValueError("forward revalidation target indices are out of bounds")
    groups: dict[tuple[str, int], list[int]] = {}
    for index, sample in enumerate(samples):
        groups.setdefault(
            (sample.instruction, int(sample.history.shape[0])), []
        ).append(index)
    batches: list[tuple[int, ...]] = []
    for instruction_and_length in sorted(groups):
        group = groups[instruction_and_length]
        for offset in range(0, len(group), gemma_batch_size):
            batch = tuple(group[offset : offset + gemma_batch_size])
            if selected.intersection(batch):
                batches.append(batch)
    covered = {index for batch in batches for index in batch}
    if not selected.issubset(covered):
        raise RuntimeError("forward revalidation did not recover every target batch")
    return tuple(batches)


def revalidate_cached_hidden_forward_contract(
    runner: ActualGemmaWaypointForward,
    cache: object,
    train_samples: Sequence[WaypointTraceSample],
    validation_samples: Sequence[WaypointTraceSample],
    train_hidden: torch.Tensor,
    validation_hidden: torch.Tensor,
    *,
    sample_count_per_split: int = 64,
    gemma_batch_size: int = 1,
) -> dict[str, object]:
    """Prove old hidden tensors remain bit-exact under a new source binding.

    Only a caller that already authenticated every cache field except the
    forward-source hash may invoke this migration check. No tolerance is used:
    any changed float rejects reuse of the complete old tensor.
    """

    if (
        train_hidden.ndim != 2
        or validation_hidden.ndim != 2
        or train_hidden.shape[0] != len(train_samples)
        or validation_hidden.shape[0] != len(validation_samples)
        or train_hidden.shape[1] != validation_hidden.shape[1]
        or not torch.isfinite(train_hidden).all()
        or not torch.isfinite(validation_hidden).all()
    ):
        raise ValueError("forward revalidation tensors differ from sample order")
    if (
        isinstance(gemma_batch_size, bool)
        or not isinstance(gemma_batch_size, int)
        or not 1 <= gemma_batch_size <= 16
    ):
        raise ValueError("forward revalidation gemma_batch_size must be in [1,16]")

    reports: dict[str, object] = {}
    for split, samples, hidden in (
        ("train", train_samples, train_hidden),
        ("validation", validation_samples, validation_hidden),
    ):
        indices = _stratified_revalidation_indices(samples, sample_count_per_split)
        batches = _source_order_revalidation_batches(
            samples,
            indices,
            gemma_batch_size=gemma_batch_size,
        )
        context_indices = tuple(index for batch in batches for index in batch)
        observed_rows: list[torch.Tensor] = []
        for batch in batches:
            recomputed = cache_actual_gemma_decision_hidden(
                runner,
                cache,
                tuple(samples[index] for index in batch),
                forward_batch_size=gemma_batch_size,
            )
            if tuple(recomputed.shape) != (len(batch), hidden.shape[1]):
                raise RuntimeError(
                    "Gemma forward-contract revalidation returned the wrong shape"
                )
            observed_rows.extend(recomputed.detach().float().cpu().unbind(dim=0))
        observed = torch.stack(observed_rows)
        expected = hidden[list(context_indices)].detach().float().cpu()
        if not torch.equal(observed, expected):
            mismatch_mask = (observed != expected).any(dim=-1)
            mismatches = int(mismatch_mask.sum())
            target_set = set(indices)
            target_mismatches = sum(
                bool(mismatch_mask[position])
                for position, index in enumerate(context_indices)
                if index in target_set
            )
            maximum = float((observed - expected).abs().max())
            raise RuntimeError(
                "Gemma forward-contract revalidation changed cached hidden rows: "
                f"split={split} target_mismatches={target_mismatches} "
                f"context_mismatches={mismatches} max_abs={maximum}"
            )
        reports[f"{split}_rows_recomputed"] = len(indices)
        reports[f"{split}_target_rows_recomputed"] = len(indices)
        reports[f"{split}_context_rows_recomputed"] = len(context_indices)
        reports[f"{split}_companion_rows_recomputed"] = len(context_indices) - len(
            set(indices).intersection(context_indices)
        )
        reports[f"{split}_source_order_batches_recomputed"] = len(batches)
        reports[f"{split}_action_strata"] = len(
            {samples[index].action_index for index in indices}
        )
        reports[f"{split}_history_length_strata"] = len(
            {int(samples[index].history.shape[0]) for index in indices}
        )
        reports[f"{split}_instruction_strata"] = len(
            {samples[index].instruction for index in indices}
        )
        reports[f"{split}_scene_strata"] = len(
            {samples[index].scene_id for index in indices}
        )
    return {
        "forward_contract_revalidated": True,
        "bit_exact_hidden_equality_required": True,
        "all_context_rows_bit_exact_required": True,
        "source_order_batch_context_reconstructed": True,
        "full_tensor_reuse_allowed": True,
        "sample_count_per_split": sample_count_per_split,
        "gemma_batch_size": gemma_batch_size,
        **reports,
    }


__all__ = [
    "assemble_hidden_with_reuse",
    "frozen_gemma_input_sha256",
    "load_legacy_waypoint_dataset_for_hidden_reuse",
    "load_waypoint_dataset_for_hidden_reuse",
    "reusable_hidden_rows",
    "revalidate_cached_hidden_forward_contract",
    "validate_forward_revalidation_destination",
]
