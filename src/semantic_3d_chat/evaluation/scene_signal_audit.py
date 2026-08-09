"""Trace counterfactual scene signal through the continuous-prefix pipeline.

This is an evaluation-only diagnostic. It reads numeric maps, adapter weights,
and QA targets, but does not alter training artifacts or the primary runtime.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from semantic_3d_chat.chat.runtime import construct_scene_tokenizer, validate_checkpoint_contract
from semantic_3d_chat.config import (
    PROJECT_ROOT,
    artifact_root,
    default_checkpoint_path,
    load_config,
    project_path,
    reports_root,
)
from semantic_3d_chat.data.dataset import SceneQADataset
from semantic_3d_chat.device import safe_dtype, select_device
from semantic_3d_chat.evaluation.metrics import canonical_relation, normalize_answer
from semantic_3d_chat.language.generation import generate_from_embeddings
from semantic_3d_chat.language.local_lm import load_local_language_model, prompt_token_ids
from semantic_3d_chat.language.lora import (
    LoRAInstallation,
    install_lora_adapters,
    lora_optimizer_settings,
    lora_settings,
    validate_lora_checkpoint_state,
)
from semantic_3d_chat.language.prefix_injection import (
    SCENE_BOUNDARY_MODE_GEMMA4_NATIVE_IMAGE,
    SCENE_BOUNDARY_MODE_LEARNED,
    ContinuousPrefixComposer,
    native_gemma4_image_contract_setting,
    prefix_sha256,
    scene_boundary_mode_setting,
    scene_prefix_after_bos_setting,
)
from semantic_3d_chat.scene_encoder.map_io import MapTensorData, load_map_tensors
from semantic_3d_chat.training.checkpointing import load_adapter_checkpoint
from semantic_3d_chat.training.pair_curriculum import (
    cap_pair_units_per_pair,
    pair_curriculum_settings,
    select_pair_only_records,
)

PAIR_SPECS = (
    {
        "pair_id": "pair_000001",
        "change_type": "color_swap",
        "split": "train",
        "scene_a": "scene_000003",
        "scene_b": "scene_000004",
    },
    {
        "pair_id": "pair_000003",
        "change_type": "mirror_lr",
        "split": "train",
        "scene_a": "scene_000007",
        "scene_b": "scene_000008",
    },
    {
        "pair_id": "pair_000002",
        "change_type": "cube_support",
        "split": "test",
        "scene_a": "scene_000005",
        "scene_b": "scene_000006",
    },
)


def _dtype_name(dtype: torch.dtype) -> str:
    """Return the stable report spelling used for the effective model dtype."""

    return str(dtype).removeprefix("torch.")


def _configured_runtime_dtype(config: dict[str, Any], device: torch.device) -> torch.dtype:
    """Mirror the language loader's device-aware dtype selection without loading weights."""

    return safe_dtype(device, str(config["language"].get("dtype", "float16")))


def _unvalidated_runtime_prefix_status(
    config: dict[str, Any], runtime_dtype: torch.dtype
) -> dict[str, Any]:
    """Describe a checkpoint projection produced without loading the base language model."""

    native_required = scene_boundary_mode_setting(config) == SCENE_BOUNDARY_MODE_GEMMA4_NATIVE_IMAGE
    return {
        "status": "checkpoint_projected_not_model_validated",
        "base_model_loaded": False,
        "configured_runtime_dtype": _dtype_name(runtime_dtype),
        "runtime_dtype_validated_against_loaded_model": False,
        "native_boundary_validation_required": native_required,
        "native_boundary_embeddings_validated": False if native_required else None,
        "runtime_prefix_parity_validated": False,
        "reason": (
            "--skip-generation intentionally avoids loading the base language model; "
            "the prefix uses the device/config-selected runtime dtype, but model-derived "
            "dtype and native boundary embeddings were not validated"
        ),
    }


def _validate_runtime_prefix_against_loaded_model(
    config: dict[str, Any],
    language: Any,
    composer: ContinuousPrefixComposer,
    expected_runtime_dtype: torch.dtype,
) -> dict[str, Any]:
    """Validate dtype, protocol identities, and persisted delimiters against the model."""

    boundary_mode = scene_boundary_mode_setting(config)
    configured_contract = native_gemma4_image_contract_setting(config)
    loaded_contract = language.scene_boundary_contract(boundary_mode)
    if loaded_contract != configured_contract:
        raise ValueError(
            "Loaded language model does not satisfy configured scene-boundary contract: "
            f"loaded={loaded_contract} configured={configured_contract}"
        )
    actual_runtime_dtype = next(language.model.parameters()).dtype
    if actual_runtime_dtype != expected_runtime_dtype:
        raise ValueError(
            "Loaded language-model dtype does not match the configured runtime dtype: "
            f"{_dtype_name(actual_runtime_dtype)} != {_dtype_name(expected_runtime_dtype)}"
        )
    native_required = boundary_mode == SCENE_BOUNDARY_MODE_GEMMA4_NATIVE_IMAGE
    native_embeddings = language.scene_boundary_embeddings(boundary_mode)
    if native_embeddings is not None:
        composer.validate_native_boundary_embeddings(native_embeddings)
    return {
        "status": "model_validated_runtime_prefix",
        "base_model_loaded": True,
        "configured_runtime_dtype": _dtype_name(expected_runtime_dtype),
        "loaded_model_runtime_dtype": _dtype_name(actual_runtime_dtype),
        "runtime_dtype_validated_against_loaded_model": True,
        "native_boundary_validation_required": native_required,
        "native_boundary_embeddings_validated": True if native_required else None,
        "runtime_prefix_parity_validated": True,
        "reason": None,
    }


def _install_checkpoint_lora(
    config: dict[str, Any],
    language: Any,
    checkpoint: Path,
    metadata: dict[str, Any],
) -> LoRAInstallation | None:
    """Install exact LoRA targets and restore/check A/B before any audit forward."""

    configured_lora = lora_settings(config)
    lora_optimizer_settings(config, configured_lora)
    installation = install_lora_adapters(language.model, configured_lora)
    if installation is None:
        return None
    loaded_metadata = load_adapter_checkpoint(
        checkpoint,
        {"lora": installation.state_module},
        device="cpu",
    )
    if loaded_metadata != metadata:
        raise RuntimeError("Checkpoint metadata changed while loading LoRA for audit")
    validate_lora_checkpoint_state(metadata, installation)
    language.model.requires_grad_(False)
    language.model.eval()
    return installation


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_epoch_fields(metadata: dict[str, Any]) -> dict[str, Any]:
    """Keep the selected checkpoint epoch distinct from its training run's best epoch."""

    return {
        "checkpoint_epoch": metadata.get("epoch"),
        "checkpoint_best_epoch": metadata.get("best_epoch"),
    }


def _structured_keys(values: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(values)
    if contiguous.ndim != 2 or contiguous.shape[1] != 3:
        raise ValueError("Coordinate keys must have shape [N,3]")
    names = ("x", "y", "z")
    dtype = np.dtype([(name, contiguous.dtype) for name in names])
    return contiguous.view(dtype).reshape(-1)


def _align_indices(keys_a: np.ndarray, keys_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    _, indices_a, indices_b = np.intersect1d(
        _structured_keys(keys_a),
        _structured_keys(keys_b),
        assume_unique=True,
        return_indices=True,
    )
    return indices_a, indices_b


def _topology_metrics(
    keys_a: np.ndarray,
    keys_b: np.ndarray,
    indices_a: np.ndarray,
    indices_b: np.ndarray,
) -> dict[str, Any]:
    common = int(indices_a.size)
    count_a = int(keys_a.shape[0])
    count_b = int(keys_b.shape[0])
    union = count_a + count_b - common
    return {
        "count_a": count_a,
        "count_b": count_b,
        "common_count": common,
        "only_a_count": count_a - common,
        "only_b_count": count_b - common,
        "union_count": union,
        "jaccard": common / union if union else 1.0,
        "common_fraction_of_smaller": common / min(count_a, count_b),
    }


def _difference_concentration(row_squared_l2: np.ndarray) -> dict[str, float]:
    values = np.sort(np.asarray(row_squared_l2, dtype=np.float64))[::-1]
    total = float(values.sum())
    if values.size == 0 or total <= 0:
        return {
            "top_1_fraction": 0.0,
            "top_10_fraction": 0.0,
            "top_1_percent_fraction": 0.0,
            "effective_changed_rows": 0.0,
        }
    top_percent = max(1, math.ceil(values.size * 0.01))
    return {
        "top_1_fraction": float(values[:1].sum() / total),
        "top_10_fraction": float(values[:10].sum() / total),
        "top_1_percent_fraction": float(values[:top_percent].sum() / total),
        "effective_changed_rows": float(total * total / np.square(values).sum()),
    }


def _aligned_array_metrics(
    values_a: np.ndarray,
    values_b: np.ndarray,
    indices_a: np.ndarray,
    indices_b: np.ndarray,
    *,
    coordinates: np.ndarray | None = None,
    chunk_size: int = 2048,
    top_n: int = 10,
) -> dict[str, Any]:
    if indices_a.size != indices_b.size or indices_a.size == 0:
        raise ValueError("Aligned arrays must have a nonempty one-to-one correspondence")
    if values_a.ndim == 1:
        values_a = values_a[:, None]
    if values_b.ndim == 1:
        values_b = values_b[:, None]
    width = int(values_a.shape[1])
    rows = int(indices_a.size)
    dot = 0.0
    norm_a_squared = 0.0
    norm_b_squared = 0.0
    diff_squared = 0.0
    absolute_difference = 0.0
    changed_elements = 0
    row_squared_l2 = np.empty(rows, dtype=np.float64)
    row_cosine = np.empty(rows, dtype=np.float64)
    for start in range(0, rows, chunk_size):
        stop = min(start + chunk_size, rows)
        a = np.asarray(values_a[indices_a[start:stop]], dtype=np.float32)
        b = np.asarray(values_b[indices_b[start:stop]], dtype=np.float32)
        difference = a - b
        row_diff = np.einsum("ij,ij->i", difference, difference, dtype=np.float64)
        row_a = np.einsum("ij,ij->i", a, a, dtype=np.float64)
        row_b = np.einsum("ij,ij->i", b, b, dtype=np.float64)
        row_dot = np.einsum("ij,ij->i", a, b, dtype=np.float64)
        denominator = np.sqrt(row_a * row_b)
        row_squared_l2[start:stop] = row_diff
        row_cosine[start:stop] = np.divide(
            row_dot,
            denominator,
            out=np.ones_like(row_dot),
            where=denominator > np.finfo(np.float64).eps,
        )
        dot += float(row_dot.sum())
        norm_a_squared += float(row_a.sum())
        norm_b_squared += float(row_b.sum())
        diff_squared += float(row_diff.sum())
        absolute_difference += float(np.abs(difference).sum(dtype=np.float64))
        changed_elements += int(np.count_nonzero(np.abs(difference) > 1e-6))
    norm_a = math.sqrt(norm_a_squared)
    norm_b = math.sqrt(norm_b_squared)
    symmetric_norm = max(0.5 * (norm_a + norm_b), np.finfo(np.float64).eps)
    top_indices = np.argsort(row_squared_l2)[-top_n:][::-1]
    top_rows: list[dict[str, Any]] = []
    for rank, aligned_index in enumerate(top_indices, start=1):
        item: dict[str, Any] = {
            "rank": rank,
            "aligned_row": int(aligned_index),
            "l2": float(math.sqrt(row_squared_l2[aligned_index])),
            "cosine": float(row_cosine[aligned_index]),
        }
        if coordinates is not None:
            item["coordinate"] = [
                float(value) for value in coordinates[indices_a[aligned_index]].tolist()
            ]
        top_rows.append(item)
    return {
        "shape_common": [rows, width],
        "cosine_flat": dot / max(norm_a * norm_b, np.finfo(np.float64).eps),
        "relative_l2": math.sqrt(diff_squared) / symmetric_norm,
        "rms_difference": math.sqrt(diff_squared / (rows * width)),
        "mean_absolute_difference": absolute_difference / (rows * width),
        "changed_element_fraction_at_1e-6": changed_elements / (rows * width),
        "mean_row_l2": float(np.sqrt(row_squared_l2).mean()),
        "max_row_l2": float(np.sqrt(row_squared_l2).max()),
        "mean_row_cosine": float(row_cosine.mean()),
        "minimum_row_cosine": float(row_cosine.min()),
        "row_difference_concentration": _difference_concentration(row_squared_l2),
        "top_changed_rows": top_rows,
    }


def _tensor_metrics(a: torch.Tensor, b: torch.Tensor, top_n: int = 10) -> dict[str, Any]:
    a_np = a.detach().float().cpu().reshape(-1, a.shape[-1]).numpy()
    b_np = b.detach().float().cpu().reshape(-1, b.shape[-1]).numpy()
    if a_np.shape != b_np.shape:
        raise ValueError(f"Tensor shape mismatch: {a_np.shape} != {b_np.shape}")
    indices = np.arange(a_np.shape[0], dtype=np.int64)
    result = _aligned_array_metrics(a_np, b_np, indices, indices, top_n=top_n)
    result["shape"] = list(a.shape)
    for item in result["top_changed_rows"]:
        item["token_index"] = item.pop("aligned_row")
    return result


def _latent_diversity(values: torch.Tensor) -> dict[str, float]:
    matrix = values.detach().float().cpu().reshape(-1, values.shape[-1])
    normalized = torch.nn.functional.normalize(matrix, dim=-1)
    similarities = normalized @ normalized.transpose(0, 1)
    mask = ~torch.eye(similarities.shape[0], dtype=torch.bool)
    off_diagonal = similarities[mask]
    return {
        "token_count": int(matrix.shape[0]),
        "mean_off_diagonal_cosine": float(off_diagonal.mean()),
        "std_off_diagonal_cosine": float(off_diagonal.std()),
        "minimum_off_diagonal_cosine": float(off_diagonal.min()),
        "mean_feature_std_across_tokens": float(matrix.std(dim=0).mean()),
    }


def _raw_pair_metrics(map_a: Path, map_b: Path) -> dict[str, Any]:
    with (
        np.load(map_a, allow_pickle=False) as archive_a,
        np.load(map_b, allow_pickle=False) as archive_b,
    ):
        keys_a = archive_a["voxel_coordinates"]
        keys_b = archive_b["voxel_coordinates"]
        indices_a, indices_b = _align_indices(keys_a, keys_b)
        centers_a = archive_a["centers_world"]
        result = {
            "voxel_topology": _topology_metrics(keys_a, keys_b, indices_a, indices_b),
            "semantic": _aligned_array_metrics(
                archive_a["semantic_features"],
                archive_b["semantic_features"],
                indices_a,
                indices_b,
                coordinates=centers_a,
            ),
            "rgb": _aligned_array_metrics(
                archive_a["mean_rgb"],
                archive_b["mean_rgb"],
                indices_a,
                indices_b,
                coordinates=centers_a,
            ),
        }
    return result


def _map_keys(data: MapTensorData, voxel_size_m: float) -> np.ndarray:
    xyz = data.xyz.detach().cpu().numpy()
    return np.floor(xyz / voxel_size_m).astype(np.int32)


def _aggregated_pair_metrics(
    data_a: MapTensorData,
    data_b: MapTensorData,
    voxel_size_m: float,
) -> dict[str, Any]:
    keys_a = _map_keys(data_a, voxel_size_m)
    keys_b = _map_keys(data_b, voxel_size_m)
    indices_a, indices_b = _align_indices(keys_a, keys_b)
    xyz_a = data_a.xyz.detach().cpu().numpy()
    return {
        "voxel_topology": _topology_metrics(keys_a, keys_b, indices_a, indices_b),
        "semantic": _aligned_array_metrics(
            data_a.semantic.detach().cpu().numpy(),
            data_b.semantic.detach().cpu().numpy(),
            indices_a,
            indices_b,
            coordinates=xyz_a,
        ),
        "rgb": _aligned_array_metrics(
            data_a.rgb.detach().cpu().numpy(),
            data_b.rgb.detach().cpu().numpy(),
            indices_a,
            indices_b,
            coordinates=xyz_a,
        ),
    }


def _block_pair_metrics(
    representation_a: dict[str, torch.Tensor],
    representation_b: dict[str, torch.Tensor],
    block_size_m: float,
    room_min: np.ndarray,
    tokens_per_block: int,
) -> dict[str, Any]:
    keys_a = representation_a["block_indices"].numpy().astype(np.int32)
    keys_b = representation_b["block_indices"].numpy().astype(np.int32)
    indices_a, indices_b = _align_indices(keys_a, keys_b)
    token_offsets = np.arange(tokens_per_block, dtype=np.int64)
    token_indices_a = (indices_a[:, None] * tokens_per_block + token_offsets).reshape(-1)
    token_indices_b = (indices_b[:, None] * tokens_per_block + token_offsets).reshape(-1)
    coordinates = room_min[None, :] + (keys_a.astype(np.float32) + 0.5) * block_size_m
    token_coordinates = np.repeat(coordinates, tokens_per_block, axis=0)
    metrics = _aligned_array_metrics(
        representation_a["block_tokens"].numpy(),
        representation_b["block_tokens"].numpy(),
        token_indices_a,
        token_indices_b,
        coordinates=token_coordinates,
    )
    for item in metrics["top_changed_rows"]:
        aligned_token = int(item.pop("aligned_row"))
        common_block_offset = aligned_token // tokens_per_block
        item["block_index"] = [int(value) for value in keys_a[indices_a[common_block_offset]]]
        item["token_within_block"] = aligned_token % tokens_per_block
    return {
        "block_topology": _topology_metrics(keys_a, keys_b, indices_a, indices_b),
        "tokens_per_block": tokens_per_block,
        "common_block_tokens": metrics,
    }


def _encode_scene(
    config: dict[str, Any],
    scene_id: str,
    scene_model: torch.nn.Module,
    composer: ContinuousPrefixComposer,
    device: torch.device,
    runtime_dtype: torch.dtype,
) -> tuple[MapTensorData, dict[str, torch.Tensor]]:
    map_path = project_path(config, "maps", scene_id, "voxel_map.npz")
    data = load_map_tensors(
        map_path,
        config["scene"]["room_size_m"],
        device="cpu",
        input_voxel_size_m=float(config["scene_encoder"]["input_voxel_size_m"]),
    )
    device_data = data.to(device)
    with torch.inference_mode():
        output = scene_model(
            device_data.semantic,
            device_data.xyz,
            device_data.rgb,
            device_data.normal,
            device_data.confidence,
            device_data.observation_count,
            device_data.room_min,
            device_data.room_max,
        )
        projected_runtime = output.scene_tokens.to(runtime_dtype)
        final_prefix = composer.scene_prefix(projected_runtime)
    representation = {
        "block_tokens": output.block_tokens.detach().float().cpu(),
        "block_indices": output.audit["block_indices"].detach().cpu(),
        "native_latents": output.native_latents.detach().float().cpu(),
        "projected_scene_tokens_float32": output.scene_tokens.detach().float().cpu(),
        "projected_scene_tokens_runtime_dtype": projected_runtime.detach().cpu(),
        "final_prefix_runtime_dtype": final_prefix.detach().cpu(),
    }
    del device_data, output, projected_runtime, final_prefix
    if device.type == "mps":
        torch.mps.empty_cache()
    return data, representation


def _changed_question_pairs(qa_path: Path, pair_id: str) -> list[tuple[dict, dict]]:
    records = [json.loads(line) for line in qa_path.read_text(encoding="utf-8").splitlines()]
    grouped: defaultdict[str, list[dict]] = defaultdict(list)
    for record in records:
        if (
            record.get("counterfactual_pair_id") == pair_id
            and record.get("counterfactual_expected_change") is True
        ):
            grouped[str(record["counterfactual_question_key"])].append(record)
    pairs: list[tuple[dict, dict]] = []
    for key, members in sorted(grouped.items()):
        if len(members) != 2:
            raise ValueError(f"Counterfactual question {key} has {len(members)} members")
        by_role = {str(member["counterfactual_role"]): member for member in members}
        pairs.append((by_role["reference"], by_role["counterfactual"]))
    return pairs


def _training_selected_pair_keys(config: dict[str, Any]) -> set[tuple[str, str]]:
    """Reconstruct the exact deterministic pair-only training selection.

    This evaluation-only helper deliberately reads the supervised train split;
    neither it nor its training-data imports are reachable from chat runtime.
    A key is ``(opaque pair ID, opaque question key)`` so similarly named keys
    from different counterfactual pairs cannot collide.
    """

    raw_training = config.get("training")
    if not isinstance(raw_training, dict) or not bool(raw_training.get("pair_only_mode", False)):
        return set()
    settings = pair_curriculum_settings(config)
    train_path = artifact_root(config, "qa") / "train.jsonl"
    records = SceneQADataset(train_path).records
    selected = select_pair_only_records(records, settings.pair_only_scene_ids)
    selected = cap_pair_units_per_pair(
        selected,
        settings.max_units_per_pair,
        seed=int(config["seed"]),
    )
    return {
        (str(record.counterfactual_pair_id), str(record.counterfactual_question_key))
        for record in selected
        if record.counterfactual_expected_change is True
        and record.counterfactual_pair_id
        and record.counterfactual_question_key
    }


def _eos_ids(language) -> int | list[int] | None:
    values: list[int] = []
    for candidate in (
        language.tokenizer.eos_token_id,
        getattr(language.model.generation_config, "eos_token_id", None),
    ):
        if candidate is None:
            continue
        if isinstance(candidate, (list, tuple)):
            values.extend(int(value) for value in candidate)
        else:
            values.append(int(candidate))
    unique = sorted(set(values))
    if not unique:
        return None
    return unique[0] if len(unique) == 1 else unique


def _generation_answer_and_provenance(
    tokenizer: Any,
    generated: torch.Tensor,
    *,
    eos_token_ids: int | list[int] | None,
    max_new_tokens: int,
) -> tuple[str, dict[str, Any]]:
    """Decode a generation while retaining evidence hidden by text fallback.

    Greedy generation always emits at least one token, but an immediate EOS can
    decode to an empty string when special tokens are skipped.  The interactive
    path historically represented that case as ``"unknown"``.  Keep that stable
    answer behavior while recording the original token IDs, decoded text, and
    termination reason so an EOS-only generation cannot be mistaken for a
    literal generated ``"unknown"`` answer in audit artifacts.
    """

    if generated.ndim != 2 or generated.shape[0] != 1 or generated.shape[1] < 1:
        raise ValueError("Generated token IDs must have shape [1, T] with T >= 1")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    token_ids = [int(value) for value in generated[0].detach().cpu().tolist()]
    stop_ids = (
        set()
        if eos_token_ids is None
        else (
            {int(eos_token_ids)}
            if isinstance(eos_token_ids, int)
            else {int(value) for value in eos_token_ids}
        )
    )
    decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
    stripped = decoded.strip()
    terminated_by_eos = token_ids[-1] in stop_ids
    token_budget_exhausted = not terminated_by_eos and len(token_ids) >= max_new_tokens
    termination_reason = (
        "eos_token"
        if terminated_by_eos
        else "max_new_tokens"
        if token_budget_exhausted
        else "decoder_returned_early_without_eos"
    )
    fallback_used = not bool(stripped)
    return stripped or "unknown", {
        "generated_token_ids": token_ids,
        "generated_token_count": len(token_ids),
        "max_new_tokens": int(max_new_tokens),
        "eos_token_ids": sorted(stop_ids),
        "termination_reason": termination_reason,
        "terminated_by_eos": terminated_by_eos,
        "termination_token_id": token_ids[-1] if terminated_by_eos else None,
        "token_budget_exhausted": token_budget_exhausted,
        "decoded_text_skip_special_tokens": decoded,
        "decoded_text_after_stripping": stripped,
        "decoded_text_was_empty": fallback_used,
        "fallback_answer": "unknown" if fallback_used else None,
        "fallback_answer_used": fallback_used,
    }


@torch.inference_mode()
def _question_logits_and_answer(language, prefix: torch.Tensor, config: dict, question: str):
    prompt_ids = prompt_token_ids(
        language.tokenizer,
        str(config["language"]["system_prompt"]),
        question,
        language.device,
    )
    runtime_prefix = prefix.to(language.device, dtype=next(language.model.parameters()).dtype)
    scene_prefix_after_bos = scene_prefix_after_bos_setting(config)
    scene_boundary_mode = scene_boundary_mode_setting(config)
    max_new_tokens = int(config["language"]["max_answer_tokens"])
    eos_token_ids = _eos_ids(language)
    if language.prefix_backend is not None:
        prepared = language.prefix_backend.prepare(
            runtime_prefix,
            prompt_ids,
            scene_prefix_after_bos=scene_prefix_after_bos,
            scene_boundary_mode=scene_boundary_mode,
        )
        output = language.prefix_backend.prefill(prepared, use_cache=False)
        logits = output.logits[:, -1].float().cpu()[0]
        generated = language.prefix_backend.generate(
            prepared,
            max_new_tokens=max_new_tokens,
            eos_token_ids=eos_token_ids,
        )
    else:
        if scene_boundary_mode != SCENE_BOUNDARY_MODE_LEARNED:
            raise ValueError("gemma4_native_image boundary mode requires the Gemma4 prefix backend")
        prompt_embeddings = language.model.get_input_embeddings()(prompt_ids)
        if scene_prefix_after_bos:
            bos_token_id = language.bos_token_id
            if bos_token_id is None or not torch.all(prompt_ids[:, 0] == bos_token_id):
                raise ValueError("BOS-first scene-prefix layout received a prompt without BOS")
            inputs = torch.cat(
                (prompt_embeddings[:, :1], runtime_prefix, prompt_embeddings[:, 1:]), dim=1
            )
        else:
            inputs = torch.cat((runtime_prefix, prompt_embeddings), dim=1)
        attention = torch.ones(inputs.shape[:2], dtype=torch.long, device=language.device)
        output = language.model(inputs_embeds=inputs, attention_mask=attention, use_cache=False)
        logits = output.logits[:, -1].float().cpu()[0]
        generated = generate_from_embeddings(
            language.model,
            inputs,
            attention,
            max_new_tokens,
            eos_token_ids,
        )
    answer, provenance = _generation_answer_and_provenance(
        language.tokenizer,
        generated,
        eos_token_ids=eos_token_ids,
        max_new_tokens=max_new_tokens,
    )
    return logits, answer, provenance


def _first_answer_token_id(tokenizer, answer: str) -> int:
    encoded = tokenizer(answer.strip(), add_special_tokens=False, return_tensors="pt").input_ids
    if encoded.numel() == 0:
        raise ValueError(f"Answer has no tokens: {answer!r}")
    return int(encoded[0, 0])


def _logit_pair_metrics(logits_a: torch.Tensor, logits_b: torch.Tensor) -> dict[str, float]:
    difference = logits_a - logits_b
    log_probability_a = torch.log_softmax(logits_a, dim=-1)
    log_probability_b = torch.log_softmax(logits_b, dim=-1)
    probability_a = log_probability_a.exp()
    probability_b = log_probability_b.exp()
    kl_a_b = torch.sum(probability_a * (log_probability_a - log_probability_b))
    kl_b_a = torch.sum(probability_b * (log_probability_b - log_probability_a))
    cosine = torch.nn.functional.cosine_similarity(logits_a, logits_b, dim=0)
    return {
        "cosine": float(cosine),
        "rms_delta": float(torch.sqrt(torch.mean(difference.square()))),
        "max_absolute_delta": float(difference.abs().max()),
        "symmetric_kl_nats": float(0.5 * (kl_a_b + kl_b_a)),
    }


def _normalized_generation_result(
    prediction_a: str,
    prediction_b: str,
    expected_a: Any,
    expected_b: Any,
) -> dict[str, Any]:
    """Score both generated sides with the project's canonical exact normalization."""

    normalized_prediction_a = normalize_answer(prediction_a)
    normalized_prediction_b = normalize_answer(prediction_b)
    normalized_expected_a = normalize_answer(expected_a)
    normalized_expected_b = normalize_answer(expected_b)
    side_a_correct = normalized_prediction_a == normalized_expected_a
    side_b_correct = normalized_prediction_b == normalized_expected_b
    canonical_prediction_a = canonical_relation(prediction_a)
    canonical_prediction_b = canonical_relation(prediction_b)
    canonical_expected_a = canonical_relation(expected_a)
    canonical_expected_b = canonical_relation(expected_b)
    canonical_eligible_a = canonical_expected_a is not None
    canonical_eligible_b = canonical_expected_b is not None
    canonical_correct_a = canonical_eligible_a and canonical_prediction_a == canonical_expected_a
    canonical_correct_b = canonical_eligible_b and canonical_prediction_b == canonical_expected_b
    return {
        "normalized_prediction_a": normalized_prediction_a,
        "normalized_prediction_b": normalized_prediction_b,
        "normalized_expected_a": normalized_expected_a,
        "normalized_expected_b": normalized_expected_b,
        "normalized_exact_correct_a": side_a_correct,
        "normalized_exact_correct_b": side_b_correct,
        "normalized_exact_complete_unit_correct": side_a_correct and side_b_correct,
        # Secondary diagnostic only.  These fields must never replace the
        # normalized-exact scores used for promotion decisions.
        "canonical_relation_prediction_a": canonical_prediction_a,
        "canonical_relation_prediction_b": canonical_prediction_b,
        "canonical_relation_expected_a": canonical_expected_a,
        "canonical_relation_expected_b": canonical_expected_b,
        "canonical_relation_secondary_eligible_a": canonical_eligible_a,
        "canonical_relation_secondary_eligible_b": canonical_eligible_b,
        "canonical_relation_secondary_correct_a": canonical_correct_a,
        "canonical_relation_secondary_correct_b": canonical_correct_b,
        "canonical_relation_secondary_complete_unit_eligible": (
            canonical_eligible_a and canonical_eligible_b
        ),
        "canonical_relation_secondary_complete_unit_correct": (
            canonical_eligible_a
            and canonical_eligible_b
            and canonical_correct_a
            and canonical_correct_b
        ),
    }


def _normalized_generation_summary(examples: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate exact generation correctness without conflating it with change rate."""

    side_count = 2 * len(examples)
    correct_side_count = sum(
        int(example[side])
        for example in examples
        for side in ("normalized_exact_correct_a", "normalized_exact_correct_b")
    )
    complete_unit_correct_count = sum(
        int(example["normalized_exact_complete_unit_correct"]) for example in examples
    )
    return {
        "normalized_exact_side_count": side_count,
        "normalized_exact_correct_side_count": correct_side_count,
        "normalized_exact_side_accuracy": (correct_side_count / side_count if side_count else None),
        "normalized_exact_complete_unit_count": len(examples),
        "normalized_exact_complete_unit_correct_count": complete_unit_correct_count,
        "normalized_exact_complete_unit_accuracy": (
            complete_unit_correct_count / len(examples) if examples else None
        ),
    }


def _canonical_relation_generation_summary(examples: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate canonical-relation extraction as a secondary diagnostic only."""

    eligible_sides = [
        (example, side)
        for example in examples
        for side in ("a", "b")
        if example[f"canonical_relation_secondary_eligible_{side}"]
    ]
    correct_side_count = sum(
        int(example[f"canonical_relation_secondary_correct_{side}"])
        for example, side in eligible_sides
    )
    eligible_units = [
        example
        for example in examples
        if example["canonical_relation_secondary_complete_unit_eligible"]
    ]
    correct_unit_count = sum(
        int(example["canonical_relation_secondary_complete_unit_correct"])
        for example in eligible_units
    )
    return {
        "canonical_relation_secondary_only": True,
        "canonical_relation_secondary_side_count": len(eligible_sides),
        "canonical_relation_secondary_correct_side_count": correct_side_count,
        "canonical_relation_secondary_side_accuracy": (
            correct_side_count / len(eligible_sides) if eligible_sides else None
        ),
        "canonical_relation_secondary_complete_unit_count": len(eligible_units),
        "canonical_relation_secondary_complete_unit_correct_count": correct_unit_count,
        "canonical_relation_secondary_complete_unit_accuracy": (
            correct_unit_count / len(eligible_units) if eligible_units else None
        ),
    }


def _left_right_reference_orientation_summary(
    examples: list[dict[str, Any]],
) -> dict[str, Any]:
    """Macro-average exact behavior across reference-left and reference-right units.

    Counterfactual mirror sets can be imbalanced by orientation. Reporting only
    micro accuracy would therefore reward a fixed per-scene direction prior.
    The reference-side macro score gives left and right equal weight, while the
    complete-unit companion additionally requires the mirrored side to be right.
    """

    opposite = {"left": "right", "right": "left"}
    by_orientation: dict[str, dict[str, int | float | None]] = {}
    reference_accuracies: list[float] = []
    complete_unit_accuracies: list[float] = []
    eligible_count = 0
    for orientation in ("left", "right"):
        members = [
            example
            for example in examples
            if example.get("normalized_expected_a") == orientation
            and example.get("normalized_expected_b") == opposite[orientation]
        ]
        eligible_count += len(members)
        reference_correct = sum(int(example["normalized_exact_correct_a"]) for example in members)
        complete_correct = sum(
            int(example["normalized_exact_complete_unit_correct"]) for example in members
        )
        reference_accuracy = reference_correct / len(members) if members else None
        complete_accuracy = complete_correct / len(members) if members else None
        if reference_accuracy is not None:
            reference_accuracies.append(reference_accuracy)
        if complete_accuracy is not None:
            complete_unit_accuracies.append(complete_accuracy)
        by_orientation[orientation] = {
            "complete_unit_count": len(members),
            "normalized_exact_reference_side_correct_count": reference_correct,
            "normalized_exact_reference_side_accuracy": reference_accuracy,
            "normalized_exact_complete_unit_correct_count": complete_correct,
            "normalized_exact_complete_unit_accuracy": complete_accuracy,
        }
    return {
        "eligible_complete_unit_count": eligible_count,
        "both_reference_orientations_present": len(reference_accuracies) == 2,
        "normalized_exact_reference_orientation_macro_accuracy": (
            float(np.mean(reference_accuracies)) if reference_accuracies else None
        ),
        "normalized_exact_complete_unit_reference_orientation_macro_accuracy": (
            float(np.mean(complete_unit_accuracies)) if complete_unit_accuracies else None
        ),
        "by_reference_expected_orientation": by_orientation,
    }


def _generation_subset_summary(examples: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate exact, change-rate, and orientation-balanced subset behavior."""

    changed_count = sum(int(example["prediction_changed"]) for example in examples)
    return {
        "changed_fact_question_count": len(examples),
        "prediction_changed_count": changed_count,
        "prediction_changed_rate": changed_count / len(examples) if examples else None,
        **_normalized_generation_summary(examples),
        **_canonical_relation_generation_summary(examples),
        "left_right_reference_orientation": _left_right_reference_orientation_summary(examples),
    }


def _generation_audit(
    config: dict[str, Any],
    representations: dict[str, dict[str, torch.Tensor]],
    pair_specs: tuple[dict[str, str], ...],
    composer: ContinuousPrefixComposer,
    runtime_dtype: torch.dtype,
    checkpoint: Path,
    metadata: dict[str, Any],
    language: Any | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if language is None:
        language = load_local_language_model(
            str(config["language"]["model_id"]),
            str(config["language"]["revision"]),
            str(config["language"]["dtype"]),
            freeze=True,
            local_files_only=True,
            backend=str(config["language"].get("backend", "auto")),
        )
        _install_checkpoint_lora(config, language, checkpoint, metadata)
    runtime_validation = _validate_runtime_prefix_against_loaded_model(
        config, language, composer, runtime_dtype
    )
    training_selected_pair_keys = _training_selected_pair_keys(config)
    pair_results = []
    for spec in pair_specs:
        qa_path = artifact_root(config, "qa") / f"{spec['split']}.jsonl"
        changed_pairs = _changed_question_pairs(qa_path, spec["pair_id"])
        examples = []
        for record_a, record_b in changed_pairs:
            if record_a["question"] != record_b["question"]:
                raise ValueError("Counterfactual pair questions are not identical")
            prefix_a = representations[spec["scene_a"]]["final_prefix_runtime_dtype"]
            prefix_b = representations[spec["scene_b"]]["final_prefix_runtime_dtype"]
            logits_a, prediction_a, generation_a = _question_logits_and_answer(
                language, prefix_a, config, record_a["question"]
            )
            logits_b, prediction_b, generation_b = _question_logits_and_answer(
                language, prefix_b, config, record_b["question"]
            )
            token_a = _first_answer_token_id(language.tokenizer, record_a["answer"])
            token_b = _first_answer_token_id(language.tokenizer, record_b["answer"])
            normalized_result = _normalized_generation_result(
                prediction_a,
                prediction_b,
                record_a["answer"],
                record_b["answer"],
            )
            examples.append(
                {
                    "question_key": record_a["counterfactual_question_key"],
                    "training_selected": (
                        str(spec["pair_id"]),
                        str(record_a["counterfactual_question_key"]),
                    )
                    in training_selected_pair_keys,
                    "question": record_a["question"],
                    "expected_a": record_a["answer"],
                    "expected_b": record_b["answer"],
                    "prediction_a": prediction_a,
                    "prediction_b": prediction_b,
                    "generation_a": generation_a,
                    "generation_b": generation_b,
                    "prediction_changed": prediction_a != prediction_b,
                    **normalized_result,
                    "logits": _logit_pair_metrics(logits_a, logits_b),
                    "expected_first_token": {
                        "token_a": token_a,
                        "token_b": token_b,
                        "decoded_a": language.tokenizer.decode([token_a]),
                        "decoded_b": language.tokenizer.decode([token_b]),
                        "scene_a_preference_margin": float(logits_a[token_a] - logits_a[token_b]),
                        "scene_b_preference_margin": float(logits_b[token_b] - logits_b[token_a]),
                        "desired_margin_shift": float(
                            (logits_b[token_b] - logits_b[token_a])
                            - (logits_a[token_b] - logits_a[token_a])
                        ),
                    },
                }
            )
        changed_count = sum(int(example["prediction_changed"]) for example in examples)
        normalized_summary = _normalized_generation_summary(examples)
        canonical_relation_summary = _canonical_relation_generation_summary(examples)
        selected_examples = [example for example in examples if example["training_selected"]]
        unselected_examples = [example for example in examples if not example["training_selected"]]
        pair_results.append(
            {
                **spec,
                "changed_fact_question_count": len(examples),
                "prediction_changed_count": changed_count,
                "prediction_changed_rate": changed_count / len(examples) if examples else None,
                **normalized_summary,
                **canonical_relation_summary,
                "training_selection_breakdown": {
                    "selected": _generation_subset_summary(selected_examples),
                    "unselected": _generation_subset_summary(unselected_examples),
                },
                "left_right_reference_orientation": (
                    _left_right_reference_orientation_summary(examples)
                ),
                "mean_logit_rms_delta": float(
                    np.mean([example["logits"]["rms_delta"] for example in examples])
                )
                if examples
                else None,
                "mean_symmetric_kl_nats": float(
                    np.mean([example["logits"]["symmetric_kl_nats"] for example in examples])
                )
                if examples
                else None,
                "examples": examples,
            }
        )
    del language
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    serialized_selection = "\n".join(
        f"{pair_id}:{question_key}" for pair_id, question_key in sorted(training_selected_pair_keys)
    )
    return {
        "scoring_policy": {
            "primary_promotion_metric": "normalized_exact_complete_unit_accuracy",
            "primary_side_metric": "normalized_exact_side_accuracy",
            "canonical_relation_metric_role": "secondary_diagnostic_only",
            "canonical_relation_does_not_satisfy_promotion": True,
        },
        "training_pair_selection": {
            "selected_complete_unit_count": len(training_selected_pair_keys),
            "selected_pair_question_keys_sha256": hashlib.sha256(
                serialized_selection.encode("utf-8")
            ).hexdigest(),
        },
        "pairs": pair_results,
    }, runtime_validation


def _make_figure(pair_results: list[dict[str, Any]], output: Path, runtime_dtype_name: str) -> None:
    stages = (
        ("raw semantic", "raw_map", "semantic"),
        ("15 cm semantic", "aggregated_map", "semantic"),
        ("block tokens", "block_tokens", "common_block_tokens"),
        ("native latents", "native_latents", None),
        ("projected tokens", "projected_scene_tokens_float32", None),
        (
            f"final prefix {runtime_dtype_name}",
            "final_prefix_runtime_dtype",
            None,
        ),
    )
    figure, (axis_l2, axis_cosine) = plt.subplots(1, 2, figsize=(13, 4.8))
    x = np.arange(len(stages))
    width = 0.24
    for pair_index, pair in enumerate(pair_results):
        l2_values = []
        cosine_values = []
        for _, outer, inner in stages:
            metrics = pair[outer]
            if inner is not None:
                metrics = metrics[inner]
            l2_values.append(max(float(metrics["relative_l2"]), 1e-9))
            cosine_values.append(max(1e-12, 1.0 - float(metrics["cosine_flat"])))
        label = f"{pair['change_type']} ({pair['split']})"
        offset = (pair_index - 1) * width
        axis_l2.bar(x + offset, l2_values, width=width, label=label)
        axis_cosine.bar(x + offset, cosine_values, width=width, label=label)
    labels = [stage[0] for stage in stages]
    for axis in (axis_l2, axis_cosine):
        axis.set_xticks(x, labels, rotation=35, ha="right")
        axis.grid(axis="y", alpha=0.25)
        axis.set_yscale("log")
    axis_l2.set_ylabel("symmetric relative L2 difference")
    axis_cosine.set_ylabel("1 - flattened cosine similarity")
    axis_l2.set_title("Counterfactual signal magnitude")
    axis_cosine.set_title("Counterfactual angular separation")
    axis_l2.legend(fontsize=8)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _signal_retention(pair: dict[str, Any]) -> dict[str, float]:
    raw = float(pair["raw_map"]["semantic"]["relative_l2"])
    aggregated = float(pair["aggregated_map"]["semantic"]["relative_l2"])
    blocks = float(pair["block_tokens"]["common_block_tokens"]["relative_l2"])
    native = float(pair["native_latents"]["relative_l2"])
    projected = float(pair["projected_scene_tokens_float32"]["relative_l2"])
    return {
        "aggregation_over_raw": aggregated / raw,
        "block_over_aggregated": blocks / aggregated,
        "native_latents_over_blocks": native / blocks,
        "projected_over_native_latents": projected / native,
        "projected_over_raw": projected / raw,
        "raw_to_projected_attenuation_factor": raw / projected,
    }


def _summary_findings(
    pair_results: list[dict[str, Any]], runtime_dtype_name: str
) -> dict[str, Any]:
    raw = [pair["raw_map"]["semantic"]["relative_l2"] for pair in pair_results]
    blocks = [pair["block_tokens"]["common_block_tokens"]["relative_l2"] for pair in pair_results]
    native = [pair["native_latents"]["relative_l2"] for pair in pair_results]
    projected = [pair["projected_scene_tokens_float32"]["relative_l2"] for pair in pair_results]
    runtime_changed = [
        pair["final_prefix_runtime_dtype"]["changed_element_fraction_at_1e-6"]
        for pair in pair_results
    ]
    latent_cosines = [
        pair["latent_diversity"][scene]["mean_off_diagonal_cosine"]
        for pair in pair_results
        for scene in ("scene_a_native", "scene_b_native")
    ]
    native_over_blocks = [
        pair["signal_retention"]["native_latents_over_blocks"] for pair in pair_results
    ]
    projected_over_native = [
        pair["signal_retention"]["projected_over_native_latents"] for pair in pair_results
    ]
    raw_signal_present = all(value > 0.0 for value in raw)
    severe_native_attenuation = all(value <= 0.1 for value in native_over_blocks)
    near_duplicate_native_latents = all(value >= 0.99 for value in latent_cosines)
    runtime_erased_any_pair = any(value == 0.0 for value in runtime_changed)

    if severe_native_attenuation and near_duplicate_native_latents:
        diagnosis = (
            "Current-checkpoint metrics support severe global-resampler attenuation: every "
            "pair retains at most 10% of block-token relative L2 in native latents, and all "
            "measured native-latent mean off-diagonal cosines are at least 0.99. This is a "
            "structural warning, not by itself proof of behavioral failure."
        )
        recommended_fix = (
            "Test a spatially anchored resampler or an explicit latent-diversity/separation "
            "objective, then require normalized exact generation and held-out controls to "
            "show that the intervention improves behavior."
        )
    elif severe_native_attenuation:
        diagnosis = (
            "Current-checkpoint metrics show severe attenuation between block tokens and "
            "native latents for every audited pair, but latent-diversity measurements do not "
            "meet the audit's near-duplicate threshold. The metrics therefore do not support "
            "the stronger claim that duplicate Perceiver latents caused the attenuation."
        )
        recommended_fix = (
            "Trace the resampler's scale and attention before changing its architecture, and "
            "judge any intervention with normalized exact generation and held-out controls."
        )
    elif near_duplicate_native_latents:
        diagnosis = (
            "Native latents meet the audit's near-duplicate cosine threshold, but audited "
            "scene-pair relative L2 is not uniformly attenuated by 10x at the resampler. "
            "Duplicate-latent collapse is therefore not established by the current metrics."
        )
        recommended_fix = (
            "Inspect token diversity alongside exact generation behavior before prescribing a "
            "resampler change; pairwise signal magnitude alone is not a success metric."
        )
    else:
        diagnosis = (
            "The current-checkpoint measurements do not support the historical global "
            "Perceiver-collapse diagnosis: native-latent/block-token relative-L2 retention "
            f"ranges from {min(native_over_blocks):.6g} to {max(native_over_blocks):.6g}, and "
            "native-latent mean off-diagonal cosine ranges from "
            f"{min(latent_cosines):.6g} to {max(latent_cosines):.6g}. Behavioral usefulness "
            "must be established by normalized exact generation and held-out controls, not by "
            "prefix hashes or nonzero tensor differences."
        )
        recommended_fix = (
            "Do not infer a resampler defect from this structural audit alone. Use the exact "
            "generation scores and held-out counterfactual controls to localize the next "
            "failure before changing the encoder."
        )

    if runtime_erased_any_pair:
        diagnosis += (
            f" Casting the final prefix to {runtime_dtype_name} erases all changes above the "
            "audit threshold for at least one pair, so runtime quantization also requires "
            "investigation."
        )
    return {
        "raw_semantic_relative_l2_range": [min(raw), max(raw)],
        "block_token_relative_l2_range": [min(blocks), max(blocks)],
        "native_latent_relative_l2_range": [min(native), max(native)],
        "projected_token_relative_l2_range": [min(projected), max(projected)],
        "runtime_prefix_changed_element_fraction_range": [
            min(runtime_changed),
            max(runtime_changed),
        ],
        "native_latent_mean_off_diagonal_cosine_range": [
            min(latent_cosines),
            max(latent_cosines),
        ],
        "native_latents_over_block_tokens_range": [
            min(native_over_blocks),
            max(native_over_blocks),
        ],
        "projected_tokens_over_native_latents_range": [
            min(projected_over_native),
            max(projected_over_native),
        ],
        "evidence_flags": {
            "raw_signal_present_for_all_pairs": raw_signal_present,
            "severe_native_attenuation_for_all_pairs_at_0_1": severe_native_attenuation,
            "near_duplicate_native_latents_for_all_scenes_at_0_99": (near_duplicate_native_latents),
            "runtime_cast_erased_any_pair_at_1e_6": runtime_erased_any_pair,
        },
        "diagnosis_basis": (
            "Computed from this checkpoint's reported pair metrics. Thresholds are <=0.1 "
            "native-latent/block-token relative-L2 retention for severe attenuation and "
            ">=0.99 mean off-diagonal cosine for near-duplicate native latents."
        ),
        "diagnosis": diagnosis,
        "recommended_fix": recommended_fix,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/multiscene.yaml")
    parser.add_argument("--checkpoint")
    parser.add_argument("--output")
    parser.add_argument("--figure")
    parser.add_argument("--skip-generation", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    checkpoint = (
        default_checkpoint_path(config)
        if args.checkpoint is None
        else Path(args.checkpoint).expanduser()
    )
    if not checkpoint.is_absolute():
        checkpoint = PROJECT_ROOT / checkpoint
    metadata = json.loads((checkpoint / "metadata.json").read_text(encoding="utf-8"))
    device = select_device()
    runtime_dtype = _configured_runtime_dtype(config, device)
    runtime_dtype_name = _dtype_name(runtime_dtype)
    scene_model = construct_scene_tokenizer(
        config,
        int(metadata["semantic_dim"]),
        int(metadata["language_hidden_dim"]),
    ).to(device)
    composer = ContinuousPrefixComposer(
        int(metadata["language_hidden_dim"]),
        scene_prefix_after_bos=scene_prefix_after_bos_setting(config),
        bos_token_id=(
            None
            if native_gemma4_image_contract_setting(config) is None
            else int(native_gemma4_image_contract_setting(config)["bos_token_id"])
        ),
        scene_boundary_mode=scene_boundary_mode_setting(config),
    ).to(device)
    configured_lora = lora_settings(config)
    configured_lora_optimizer = lora_optimizer_settings(config, configured_lora)
    audit_language = None
    lora_installation = None
    if configured_lora.enabled:
        audit_language = load_local_language_model(
            str(config["language"]["model_id"]),
            str(config["language"]["revision"]),
            str(config["language"]["dtype"]),
            freeze=True,
            local_files_only=True,
            backend=str(config["language"].get("backend", "auto")),
        )
        lora_installation = _install_checkpoint_lora(config, audit_language, checkpoint, metadata)
        assert lora_installation is not None
    contract_warnings = validate_checkpoint_contract(
        metadata,
        config,
        semantic_dim=int(metadata["semantic_dim"]),
        language_hidden_dim=int(metadata["language_hidden_dim"]),
        lora_parameter_count=(
            0 if lora_installation is None else lora_installation.parameter_count
        ),
    )
    checkpoint_modules = {"scene_model": scene_model, "composer": composer}
    load_adapter_checkpoint(
        checkpoint,
        checkpoint_modules,
        device="cpu",
    )
    runtime_validation = _unvalidated_runtime_prefix_status(config, runtime_dtype)
    if lora_installation is not None:
        assert audit_language is not None and configured_lora_optimizer is not None
        runtime_validation = _validate_runtime_prefix_against_loaded_model(
            config, audit_language, composer, runtime_dtype
        )
    scene_model.eval()
    composer.eval()
    learned_query_diversity = {
        "global_resampler_queries": _latent_diversity(scene_model.resampler.learned_queries),
        "spatial_block_queries": _latent_diversity(scene_model.block_encoder.queries),
    }

    representations: dict[str, dict[str, torch.Tensor]] = {}
    aggregated_data: dict[str, MapTensorData] = {}
    raw_pair_results: dict[str, dict[str, Any]] = {}
    for spec in PAIR_SPECS:
        map_a = project_path(config, "maps", spec["scene_a"], "voxel_map.npz")
        map_b = project_path(config, "maps", spec["scene_b"], "voxel_map.npz")
        raw_pair_results[spec["pair_id"]] = _raw_pair_metrics(map_a, map_b)
        gc.collect()

    for scene_id in sorted(
        {scene for spec in PAIR_SPECS for scene in (spec["scene_a"], spec["scene_b"])}
    ):
        data, representation = _encode_scene(
            config,
            scene_id,
            scene_model,
            composer,
            device,
            runtime_dtype,
        )
        aggregated_data[scene_id] = data
        representations[scene_id] = representation

    pair_results = []
    input_voxel_size = float(config["scene_encoder"]["input_voxel_size_m"])
    block_size = float(config["scene_encoder"]["block_size_m"])
    tokens_per_block = int(config["scene_encoder"]["tokens_per_block"])
    room_size = np.asarray(config["scene"]["room_size_m"], dtype=np.float32)
    room_min = np.asarray([-room_size[0] / 2, -room_size[1] / 2, 0.0], dtype=np.float32)
    for spec in PAIR_SPECS:
        data_a = aggregated_data[spec["scene_a"]]
        data_b = aggregated_data[spec["scene_b"]]
        representation_a = representations[spec["scene_a"]]
        representation_b = representations[spec["scene_b"]]
        pair_result: dict[str, Any] = {
            **spec,
            "raw_map": raw_pair_results[spec["pair_id"]],
            "aggregated_map": _aggregated_pair_metrics(data_a, data_b, input_voxel_size),
            "block_tokens": _block_pair_metrics(
                representation_a,
                representation_b,
                block_size,
                room_min,
                tokens_per_block,
            ),
        }
        for name in (
            "native_latents",
            "projected_scene_tokens_float32",
            "projected_scene_tokens_runtime_dtype",
            "final_prefix_runtime_dtype",
        ):
            pair_result[name] = _tensor_metrics(representation_a[name], representation_b[name])
        pair_result["prefix_hash_a"] = prefix_sha256(representation_a["final_prefix_runtime_dtype"])
        pair_result["prefix_hash_b"] = prefix_sha256(representation_b["final_prefix_runtime_dtype"])
        pair_result["prefix_exactly_equal"] = bool(
            torch.equal(
                representation_a["final_prefix_runtime_dtype"],
                representation_b["final_prefix_runtime_dtype"],
            )
        )
        pair_result["latent_diversity"] = {
            "scene_a_native": _latent_diversity(representation_a["native_latents"]),
            "scene_b_native": _latent_diversity(representation_b["native_latents"]),
            "scene_a_projected": _latent_diversity(
                representation_a["projected_scene_tokens_float32"]
            ),
            "scene_b_projected": _latent_diversity(
                representation_b["projected_scene_tokens_float32"]
            ),
        }
        pair_result["signal_retention"] = _signal_retention(pair_result)
        pair_results.append(pair_result)

    generation = None
    del scene_model, aggregated_data
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    if not args.skip_generation:
        generation, runtime_validation = _generation_audit(
            config,
            representations,
            PAIR_SPECS,
            composer,
            runtime_dtype,
            checkpoint,
            metadata,
            language=audit_language,
        )
    if audit_language is not None:
        del audit_language
    del composer

    configured_reports = reports_root(config)
    output = (
        configured_reports / "metrics" / "scene_signal_audit.json"
        if args.output is None
        else Path(args.output).expanduser()
    )
    figure = (
        configured_reports / "figures" / "scene_signal_audit.png"
        if args.figure is None
        else Path(args.figure).expanduser()
    )
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    if not figure.is_absolute():
        figure = PROJECT_ROOT / figure
    payload = {
        "schema_version": 2,
        "purpose": (
            "Counterfactual signal audit from numeric maps through frozen-LM behavior"
            if generation is not None
            else (
                "Counterfactual checkpoint projection audit with base-model and LoRA "
                "checkpoint validation; generation intentionally skipped"
                if runtime_validation["base_model_loaded"]
                else "Counterfactual checkpoint projection audit without base-model validation"
            )
        ),
        "config": str(Path(args.config)),
        "checkpoint": str(checkpoint.relative_to(PROJECT_ROOT)),
        "checkpoint_sha256": _sha256(checkpoint / "adapter.safetensors"),
        **_checkpoint_epoch_fields(metadata),
        "scene_prefix_after_bos": scene_prefix_after_bos_setting(config),
        "scene_boundary_mode": scene_boundary_mode_setting(config),
        "gemma4_native_image_contract": native_gemma4_image_contract_setting(config),
        "lora": metadata.get("lora", {"schema_version": 1, "enabled": False}),
        "checkpoint_contract_warnings": contract_warnings,
        "device": str(device),
        "runtime_dtype": runtime_dtype_name,
        "runtime_validation": runtime_validation,
        "prefix_hash_semantics": (
            "SHA-256 over the final checkpoint-projected scene prefix after casting "
            f"scene tokens and persisted boundaries to {runtime_dtype_name}; consult "
            "runtime_validation.runtime_prefix_parity_validated before treating it as "
            "a model-validated chat-runtime prefix"
        ),
        "learned_query_diversity": learned_query_diversity,
        "comparison_note": (
            "Map semantic/RGB metrics compare only shared spatial cells; topology metrics "
            "separately measure cells added or removed. Downstream token metrics compare "
            "corresponding block queries or fixed global latent indices."
        ),
        "summary_findings": _summary_findings(pair_results, runtime_dtype_name),
        "pairs": pair_results,
        "generation": generation,
        "figure": str(figure.relative_to(PROJECT_ROOT)),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _make_figure(pair_results, figure, runtime_dtype_name)
    print(
        json.dumps(
            {
                "output": str(output),
                "figure": str(figure),
                "pair_count": len(pair_results),
                "generation_included": generation is not None,
                "runtime_dtype": runtime_dtype_name,
                "runtime_prefix_parity_validated": runtime_validation[
                    "runtime_prefix_parity_validated"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
