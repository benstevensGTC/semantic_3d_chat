"""Evaluation-only Gemma 4 token/voxel semantic localization.

This command scores the final 1536D native projected visual stream in the
3072D Gemma map against mean language-token embeddings from Gemma's tied input
embedding table. It never constructs the language model and never reads any
other model parameter. Semantic category names and oracle boxes enter only in
this evaluation process; they are not runtime inputs and are written only to
the evaluation report tree.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch

from semantic_3d_chat.config import load_config, project_path, reports_root
from semantic_3d_chat.evaluation.semantic_sanity import (
    SemanticFeatureSlice,
    _atomic_json,
    _default_feature_location,
    _default_map_path,
    _load_oracle_for_evaluation,
    _reject_oracle_runtime_input,
    _validate_scene_id,
    compute_multiview_consistency,
    normalize_embedding_matrix,
    oracle_targets,
    queries_from_targets,
    score_semantic_queries,
    write_consistency_histogram,
    write_query_heatmaps,
)
from semantic_3d_chat.mapping.voxel_map import SparseVoxelMap
from semantic_3d_chat.vision.model_registry import get_model_spec

LOGGER = logging.getLogger(__name__)

GEMMA4_NATIVE_DIM = 768
GEMMA4_PROJECTED_DIM = 1536
GEMMA4_PROJECTED_START = GEMMA4_NATIVE_DIM * 2
GEMMA4_TOTAL_SEMANTIC_DIM = GEMMA4_PROJECTED_START + GEMMA4_PROJECTED_DIM
GEMMA4_TOKEN_EMBEDDING_KEY = "model.language_model.embed_tokens.weight"
GEMMA4_PROJECTED_SLICE = SemanticFeatureSlice(
    total_dim=GEMMA4_TOTAL_SEMANTIC_DIM,
    start=GEMMA4_PROJECTED_START,
    dimension=GEMMA4_PROJECTED_DIM,
    name="Gemma 4 native projected visual slice",
)


class CategoryTokenizer(Protocol):
    def __call__(self, text: str, *, add_special_tokens: bool) -> dict[str, Any]: ...


def category_token_ids(
    categories: Sequence[str],
    tokenizer: CategoryTokenizer,
) -> list[list[int]]:
    """Tokenize bare category names without BOS, prompts, or chat templates."""

    if not categories:
        raise ValueError("At least one category is required")
    sequences: list[list[int]] = []
    for category in categories:
        if not isinstance(category, str) or not category.strip():
            raise ValueError("Category names must be non-empty strings")
        encoded = tokenizer(category.strip(), add_special_tokens=False)
        raw_ids = encoded.get("input_ids")
        if isinstance(raw_ids, torch.Tensor):
            raw_ids = raw_ids.detach().cpu().reshape(-1).tolist()
        if not isinstance(raw_ids, list) or not raw_ids:
            raise ValueError(f"Tokenizer returned no token IDs for category {category!r}")
        if raw_ids and isinstance(raw_ids[0], list):
            if len(raw_ids) != 1:
                raise ValueError("Tokenizer unexpectedly returned a batched category result")
            raw_ids = raw_ids[0]
        token_ids = [int(token_id) for token_id in raw_ids]
        if any(token_id < 0 for token_id in token_ids):
            raise ValueError("Tokenizer returned a negative token ID")
        sequences.append(token_ids)
    return sequences


def mean_token_embeddings(
    token_id_sequences: Sequence[Sequence[int]],
    embedding_weight: np.ndarray | torch.Tensor,
) -> np.ndarray:
    """Mean token rows per query and return unit-normalized float32 vectors."""

    weight = (
        embedding_weight.detach().float().cpu().numpy()
        if isinstance(embedding_weight, torch.Tensor)
        else np.asarray(embedding_weight, dtype=np.float32)
    )
    if weight.ndim != 2 or weight.shape[0] < 1 or weight.shape[1] < 1:
        raise ValueError(f"Embedding weight must have shape [vocab, dim], got {weight.shape}")
    if not np.isfinite(weight).all():
        raise ValueError("Embedding weight contains NaN or infinite values")
    vectors: list[np.ndarray] = []
    for token_ids in token_id_sequences:
        ids = np.asarray(list(token_ids), dtype=np.int64)
        if ids.ndim != 1 or not ids.size:
            raise ValueError("Every query must contain at least one token ID")
        if np.any(ids < 0) or np.any(ids >= weight.shape[0]):
            raise ValueError("Query token ID is outside the embedding vocabulary")
        vectors.append(weight[ids].mean(axis=0, dtype=np.float32))
    return normalize_embedding_matrix(
        np.stack(vectors),
        expected_dim=weight.shape[1],
        label="Mean category token embeddings",
    )


def _checkpoint_for_key(snapshot: Path, tensor_key: str) -> Path:
    direct = snapshot / "model.safetensors"
    if direct.is_file():
        return direct
    index_path = snapshot / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"No safetensors checkpoint found in {snapshot}")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    shard_name = index.get("weight_map", {}).get(tensor_key)
    if not isinstance(shard_name, str):
        raise KeyError(f"Safetensors index does not contain {tensor_key!r}")
    shard = snapshot / shard_name
    if not shard.is_file():
        raise FileNotFoundError(f"Indexed safetensors shard is absent: {shard}")
    return shard


def resolve_local_snapshot(
    model_id: str,
    revision: str,
    explicit_snapshot: str | Path | None = None,
) -> Path:
    """Resolve only an already-cached pinned snapshot; network is never used."""

    if explicit_snapshot is not None:
        snapshot = Path(explicit_snapshot).expanduser().resolve()
        if not snapshot.is_dir():
            raise FileNotFoundError(f"Model snapshot directory not found: {snapshot}")
        return _reject_oracle_runtime_input(snapshot, "Model snapshot")
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:  # pragma: no cover - setup error
        raise RuntimeError("huggingface-hub is required to resolve the local snapshot") from error
    try:
        resolved = snapshot_download(
            repo_id=model_id,
            revision=revision,
            local_files_only=True,
        )
    except Exception as error:
        raise FileNotFoundError(
            f"Pinned local snapshot is unavailable for {model_id}@{revision}; "
            "run `make download-gemma4-weights` first"
        ) from error
    return _reject_oracle_runtime_input(Path(resolved), "Model snapshot")


def load_category_embeddings_selective(
    snapshot: str | Path,
    categories: Sequence[str],
    *,
    tokenizer: CategoryTokenizer | None = None,
    tensor_key: str = GEMMA4_TOKEN_EMBEDDING_KEY,
    expected_dim: int = GEMMA4_PROJECTED_DIM,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Read only the requested token rows from exactly one safetensors key."""

    source = _reject_oracle_runtime_input(Path(snapshot), "Model snapshot")
    if tokenizer is None:
        try:
            from transformers import AutoTokenizer
        except ImportError as error:  # pragma: no cover - setup error
            raise RuntimeError("Transformers is required to load the local tokenizer") from error
        tokenizer = AutoTokenizer.from_pretrained(
            source,
            local_files_only=True,
            trust_remote_code=False,
        )
    sequences = category_token_ids(categories, tokenizer)
    unique_ids = sorted({token_id for sequence in sequences for token_id in sequence})
    checkpoint = _checkpoint_for_key(source, tensor_key)
    try:
        from safetensors import safe_open
    except ImportError as error:  # pragma: no cover - setup error
        raise RuntimeError("safetensors is required for selective Gemma loading") from error

    with safe_open(checkpoint, framework="pt", device="cpu") as tensors:
        available_keys = set(tensors.keys())
        if tensor_key not in available_keys:
            raise KeyError(f"Safetensors checkpoint does not contain {tensor_key!r}")
        tensor_slice = tensors.get_slice(tensor_key)
        shape = tuple(int(value) for value in tensor_slice.get_shape())
        if len(shape) != 2 or shape[1] != expected_dim:
            raise ValueError(
                f"Expected {tensor_key} [vocab, {expected_dim}], found {shape}"
            )
        if unique_ids[-1] >= shape[0]:
            raise ValueError("Tokenizer emitted an ID outside the embedding vocabulary")
        # PySafeSlice performs row-selective I/O. No full model or full table is
        # constructed, and no parameter besides tensor_key is accessed.
        selected_raw_rows = tensor_slice[unique_ids].cpu()
        source_dtype = str(selected_raw_rows.dtype).removeprefix("torch.")
        selected_rows = selected_raw_rows.float()

    row_for_token = {token_id: index for index, token_id in enumerate(unique_ids)}
    remapped_sequences = [
        [row_for_token[token_id] for token_id in sequence] for sequence in sequences
    ]
    embeddings = mean_token_embeddings(remapped_sequences, selected_rows)
    metadata = {
        "checkpoint_path": str(checkpoint),
        "loaded_parameter_keys": [tensor_key],
        "weight_shape": list(shape),
        "weight_dtype": source_dtype,
        "mean_compute_dtype": str(selected_rows.dtype).removeprefix("torch."),
        "selective_row_read": True,
        "unique_token_rows_read": len(unique_ids),
        "query_token_ids": {
            category: token_ids for category, token_ids in zip(categories, sequences, strict=True)
        },
    }
    return embeddings, metadata


def run_gemma4_semantic_sanity(
    config: dict[str, Any],
    scene_id: str,
    *,
    map_path: str | Path | None = None,
    oracle_path: str | Path | None = None,
    render_manifest_path: str | Path | None = None,
    feature_location: str | Path | None = None,
    model_snapshot: str | Path | None = None,
    output_path: str | Path | None = None,
    figures_directory: str | Path | None = None,
    top_k: int | None = None,
    skip_consistency: bool = False,
    write_figures: bool = True,
    tokenizer: CategoryTokenizer | None = None,
) -> dict[str, Any]:
    """Run the CPU-only, evaluation-only Gemma semantic-map sanity check."""

    scene_id = _validate_scene_id(scene_id)
    vision_config = config["vision"]
    if str(vision_config.get("backend")) != "gemma4":
        raise ValueError("Gemma semantic sanity requires vision.backend=gemma4")
    model_id = str(vision_config["model_id"])
    revision = str(vision_config["revision"])
    spec = get_model_spec(model_id)
    if spec.native_dim != GEMMA4_NATIVE_DIM or spec.aligned_dim != GEMMA4_PROJECTED_DIM:
        raise ValueError(f"Unsupported Gemma semantic feature dimensions: {spec}")

    selected_map_path = _reject_oracle_runtime_input(
        map_path or _default_map_path(config, scene_id), "Fused map"
    )
    voxel_map = SparseVoxelMap.load(selected_map_path)
    arrays = voxel_map.to_arrays(encode_semantics=False)
    if arrays["semantic_features"].shape[1] != GEMMA4_TOTAL_SEMANTIC_DIM:
        raise ValueError(
            f"Map feature dimension is {arrays['semantic_features'].shape[1]}, expected "
            f"{GEMMA4_TOTAL_SEMANTIC_DIM} = middle768 + late768 + native_projected1536"
        )

    sanity_config = config.get("evaluation", {}).get("semantic_sanity", {})
    selected_top_k = int(top_k if top_k is not None else sanity_config.get("top_k", 100))
    bbox_padding_voxels = float(sanity_config.get("bbox_padding_voxels", 0.75))
    if selected_top_k < 1:
        raise ValueError("semantic_sanity.top_k must be positive")
    if not np.isfinite(bbox_padding_voxels) or bbox_padding_voxels < 0:
        raise ValueError("semantic_sanity.bbox_padding_voxels must be non-negative")

    # Semantic names enter only from this isolated evaluation oracle. The map
    # above is already loaded and remains the sole environmental representation.
    selected_oracle_path = (
        Path(oracle_path)
        if oracle_path is not None
        else project_path(config, "oracle", scene_id, "oracle.json")
    )
    oracle = _load_oracle_for_evaluation(selected_oracle_path, scene_id)
    targets = oracle_targets(oracle)
    queries = queries_from_targets(targets, "{}")
    categories = [query.category for query in queries]

    snapshot = resolve_local_snapshot(model_id, revision, model_snapshot)
    text_embeddings, selective_load = load_category_embeddings_selective(
        snapshot,
        categories,
        tokenizer=tokenizer,
    )
    padding_m = max(voxel_map.voxel_size_m * bbox_padding_voxels, 1e-4)
    localization, similarities = score_semantic_queries(
        arrays["centers_world"],
        arrays["semantic_features"],
        queries,
        text_embeddings,
        targets,
        top_k=selected_top_k,
        bbox_padding_m=padding_m,
        feature_slice=GEMMA4_PROJECTED_SLICE,
    )

    selected_output_path = (
        Path(output_path)
        if output_path
        else reports_root(config)
        / "metrics"
        / f"gemma4_semantic_sanity_{scene_id}.json"
    )
    selected_figures_directory = (
        Path(figures_directory)
        if figures_directory
        else reports_root(config)
        / "figures"
        / "gemma4_semantic_sanity"
        / scene_id
    )
    heatmaps: list[dict[str, str]] = []
    if write_figures:
        heatmaps = write_query_heatmaps(
            arrays["centers_world"],
            similarities,
            queries,
            targets,
            selected_figures_directory,
            max_points=int(sanity_config.get("heatmap_max_points", 100_000)),
            similarity_label="Gemma 4 token/visual cosine similarity",
        )

    consistency_metrics: dict[str, Any]
    consistency_figure: str | None = None
    if skip_consistency:
        consistency_metrics = {"available": False, "reason": "disabled_by_cli"}
    else:
        selected_render_manifest = (
            Path(render_manifest_path)
            if render_manifest_path
            else project_path(config, "rendered", scene_id, "manifest.json")
        )
        selected_feature_location = (
            Path(feature_location)
            if feature_location
            else _default_feature_location(config, scene_id)
        )
        try:
            consistency_metrics, same_values, different_values = compute_multiview_consistency(
                selected_render_manifest,
                selected_feature_location,
                voxel_size_m=voxel_map.voxel_size_m,
                depth_min_m=float(config["mapping"].get("depth_min_m", 0.1)),
                depth_max_m=float(config["mapping"].get("depth_max_m", 10.0)),
                pixel_stride=int(config["mapping"].get("pixel_stride", 1)),
                feature_slice=GEMMA4_PROJECTED_SLICE,
            )
            if write_figures and consistency_metrics["available"]:
                consistency_path = write_consistency_histogram(
                    same_values,
                    different_values,
                    selected_figures_directory / "view_consistency.png",
                    feature_label="Gemma 4 native projected cosine similarity",
                )
                consistency_figure = str(consistency_path)
        except FileNotFoundError as error:
            consistency_metrics = {"available": False, "reason": str(error)}

    metrics = {
        "schema_version": 1,
        "phase": "gemma4_semantic_sanity",
        "evaluation_only": True,
        "scene_id": scene_id,
        "map_path": str(selected_map_path),
        "map_content_hash": voxel_map.content_hash(),
        "voxel_count": len(voxel_map),
        "voxel_size_m": voxel_map.voxel_size_m,
        "feature_layout": {
            "total_dim": GEMMA4_TOTAL_SEMANTIC_DIM,
            "middle": [0, GEMMA4_NATIVE_DIM],
            "late": [GEMMA4_NATIVE_DIM, GEMMA4_PROJECTED_START],
            "gemma_native_projected": [
                GEMMA4_PROJECTED_START,
                GEMMA4_TOTAL_SEMANTIC_DIM,
            ],
            "scored_slice": "gemma_native_projected",
            "aligned_method": str(vision_config.get("aligned_method")),
        },
        "vision_model": model_id,
        "vision_revision": revision,
        "language_embedding_model": model_id,
        "language_embedding_revision": revision,
        "text_embedding_method": "mean_bare_category_input_token_embeddings",
        "local_files_only": True,
        "cpu_only": True,
        "selective_model_load": selective_load,
        "query_count": len(queries),
        "top_k": selected_top_k,
        "bbox_padding_voxels": bbox_padding_voxels,
        "bbox_padding_m": padding_m,
        **localization,
        "same_voxel_consistency": consistency_metrics,
        "artifacts": {
            "heatmaps": heatmaps,
            "view_consistency_histogram": consistency_figure,
        },
    }
    _atomic_json(selected_output_path, metrics)
    LOGGER.info(
        "phase=gemma4_semantic_sanity scene=%s voxels=%d queries=%d "
        "top1_accuracy=%s top_k_accuracy=%s",
        scene_id,
        len(voxel_map),
        len(queries),
        metrics["aggregate"]["top1_localization_accuracy"],
        metrics["aggregate"]["top_k_localization_accuracy"],
    )
    return {**metrics, "metrics_path": str(selected_output_path)}


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/gemma4_e2b.yaml")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--map", type=Path)
    parser.add_argument("--oracle", type=Path)
    parser.add_argument("--render-manifest", type=Path)
    parser.add_argument("--features", type=Path)
    parser.add_argument("--model-snapshot", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--figures", type=Path)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--offline", action="store_true", help="Accepted for explicitness; always on")
    parser.add_argument("--skip-consistency", action="store_true")
    parser.add_argument("--no-figures", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    metrics = run_gemma4_semantic_sanity(
        load_config(args.config),
        args.scene,
        map_path=args.map,
        oracle_path=args.oracle,
        render_manifest_path=args.render_manifest,
        feature_location=args.features,
        model_snapshot=args.model_snapshot,
        output_path=args.output,
        figures_directory=args.figures,
        top_k=args.top_k,
        skip_consistency=args.skip_consistency,
        write_figures=not args.no_figures,
    )
    print(
        json.dumps(
            {
                "scene_id": metrics["scene_id"],
                "metrics_path": metrics["metrics_path"],
                "voxel_count": metrics["voxel_count"],
                "aggregate": metrics["aggregate"],
                "same_voxel_consistency": metrics["same_voxel_consistency"],
                "loaded_parameter_keys": metrics["selective_model_load"][
                    "loaded_parameter_keys"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
