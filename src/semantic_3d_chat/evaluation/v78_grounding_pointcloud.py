"""Reproduce and visualize V78's historical-held numeric grounding.

This command is deliberately evaluation-only.  It reconstructs the already
sealed 94-row historical holdout, verifies that its aggregate metrics exactly
match the sealed V78 report, and overlays predicted and oracle coordinates on
sanitized RGB point clouds.  Oracle-derived target coordinates and question
text are written only below ``reports/``; no runtime artifact is created or
modified.

The full Gemma model is not loaded.  The evaluator reads only the requested
rows of Gemma's frozen input-embedding table, the complete question-independent
scene prefixes, the sealed V78 sidecar, and numeric/RGB map arrays.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import textwrap
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from safetensors.torch import load_file

from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.evaluation.gemma4_semantic_sanity import (
    load_category_embeddings_selective,
    resolve_local_snapshot,
)
from semantic_3d_chat.scene_encoder.grounding_sidecar_v78 import GroundingSidecarV78
from semantic_3d_chat.training.grounding_sidecar_v78 import (
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    DEFAULT_ROOM_MAX,
    DEFAULT_ROOM_MIN,
    GroundingRecord,
    QuestionIndependentPrefixStore,
    _predict,
    _prediction_metrics,
    load_historical_grounding_records,
    pair_disjoint_internal_split,
    validate_candidate,
)

SEALED_REPORT_SHA256: Final[str] = (
    "557cc497dd12bd74f45cecd3624e18649ddc548af091ed719c22c7998942b84b"
)
DEFAULT_QA: Final[Path] = Path("data_gemma4/training/v62_pair_disjoint/train.jsonl")
DEFAULT_PREFIX_CACHE: Final[Path] = Path(
    "data_gemma4/scene_tokens/v62_pair_disjoint_train_prefixes"
)
DEFAULT_CANDIDATE: Final[Path] = Path(
    "reports/gemma4/artifacts/v78_grounding_sidecar_diagnostic"
)
DEFAULT_SEALED_REPORT: Final[Path] = Path(
    "reports/gemma4/metrics/v78_grounding_sidecar_internal_held.json"
)
DEFAULT_MAPS_ROOT: Final[Path] = Path("data_gemma4/maps")
DEFAULT_METRICS: Final[Path] = Path(
    "reports/gemma4/metrics/v78_grounding_held_pointcloud_evaluation.json"
)
DEFAULT_FIGURE: Final[Path] = Path(
    "reports/gemma4/figures/v78_grounding_held_pointcloud_examples.png"
)
PNG_METADATA: Final[dict[str, str]] = {
    "Software": "semantic_3d_chat deterministic V78 grounding evaluator"
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_array(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return (value if value.is_absolute() else PROJECT_ROOT / value).resolve()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _assert_evaluation_output(path: Path) -> None:
    runtime_root = _resolve("data_gemma4/runtime")
    try:
        path.relative_to(runtime_root)
    except ValueError:
        return
    raise ValueError("V78 oracle-bearing evaluation output must not be written to runtime data")


def load_sealed_report(path: str | Path) -> dict[str, Any]:
    """Load the one authenticated historical-held V78 result."""

    source = _resolve(path)
    if source.is_symlink() or not source.is_file():
        raise FileNotFoundError(f"Sealed V78 report is unavailable: {source}")
    observed = _sha256_file(source)
    if observed != SEALED_REPORT_SHA256:
        raise ValueError(
            "V78 sealed report digest differs: "
            f"expected={SEALED_REPORT_SHA256} observed={observed}"
        )
    payload = json.loads(source.read_text(encoding="utf-8"))
    split = payload.get("split")
    metrics = payload.get("metrics")
    if not (
        payload.get("artifact") == "v78_historical_training_grounding_repair_report_v1"
        and payload.get("status") == "internal_historical_diagnostic_only"
        and payload.get("runtime_promotion_authorized") is False
        and payload.get("official_validation_loaded") is False
        and payload.get("official_test_loaded") is False
        and payload.get("deferred_final_loaded") is False
        and payload.get("oracle_files_loaded") is False
        and isinstance(split, dict)
        and split.get("held_grounded_rows") == 94
        and split.get("pair_disjoint") is True
        and split.get("scene_disjoint") is True
        and isinstance(metrics, dict)
    ):
        raise ValueError("V78 sealed report identity, isolation, or split changed")
    return payload


def select_visualization_indices(records: Sequence[GroundingRecord]) -> list[int]:
    """Select one example per held scene without looking at predictions or errors."""

    grouped: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        grouped[record.scene_id].append(index)
    if not grouped:
        raise ValueError("No held records are available for visualization")
    return [
        min(grouped[scene_id], key=lambda index: records[index].question_id)
        for scene_id in sorted(grouped)
    ]


def _load_map_arrays(
    maps_root: Path, scene_id: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    if Path(scene_id).name != scene_id or not scene_id.startswith("scene_"):
        raise ValueError(f"Unsafe scene ID: {scene_id!r}")
    source = maps_root / scene_id / "voxel_map.npz"
    if source.is_symlink() or not source.is_file():
        raise FileNotFoundError(f"Sanitized voxel map is unavailable: {source}")
    with np.load(source, allow_pickle=False) as archive:
        required = {"centers_world", "mean_rgb", "confidence"}
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError(f"Voxel map is missing numeric fields: {missing}")
        xyz = np.asarray(archive["centers_world"], dtype=np.float32)
        rgb = np.asarray(archive["mean_rgb"], dtype=np.float32)
        confidence = np.asarray(archive["confidence"], dtype=np.float32)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or xyz.shape[0] < 1:
        raise ValueError(f"Invalid voxel coordinates for {scene_id}: {xyz.shape}")
    if rgb.shape != xyz.shape or confidence.shape != (xyz.shape[0],):
        raise ValueError(f"Invalid RGB/confidence map shapes for {scene_id}")
    if not all(np.isfinite(value).all() for value in (xyz, rgb, confidence)):
        raise ValueError(f"Voxel map contains NaN or infinity for {scene_id}")
    identity = {
        "path": source.relative_to(PROJECT_ROOT).as_posix(),
        "voxel_count": int(xyz.shape[0]),
        "xyz_sha256": _sha256_array(xyz),
        "rgb_sha256": _sha256_array(rgb),
        "confidence_sha256": _sha256_array(confidence),
        "semantic_features_loaded": False,
        "metadata_json_loaded": False,
    }
    return xyz, np.clip(rgb / 255.0, 0.0, 1.0), confidence, identity


def _nearest_map_support(
    prediction: np.ndarray, xyz: np.ndarray, confidence: np.ndarray
) -> dict[str, Any]:
    distances = np.linalg.norm(xyz.astype(np.float64) - prediction.astype(np.float64), axis=1)
    nearest = int(np.argmin(distances))
    return {
        "distance_m": float(distances[nearest]),
        "nearest_voxel_xyz_m": [float(value) for value in xyz[nearest]],
        "nearest_voxel_confidence": float(np.clip(confidence[nearest], 0.0, 1.0)),
    }


def _assert_aggregate_reproduction(
    observed: Mapping[str, Any], expected: Mapping[str, Any]
) -> dict[str, float]:
    if set(observed) != set(expected):
        raise ValueError("Reproduced V78 metric fields differ from the sealed report")
    deltas: dict[str, float] = {}
    for key, expected_value in expected.items():
        observed_value = observed[key]
        if isinstance(expected_value, bool) or not isinstance(expected_value, (int, float)):
            if observed_value != expected_value:
                raise ValueError(f"Reproduced V78 metric differs: {key}")
            continue
        delta = abs(float(observed_value) - float(expected_value))
        deltas[key] = delta
        if delta > 1e-7:
            raise ValueError(
                f"Reproduced V78 metric differs: {key} delta={delta:.9g}"
            )
    return deltas


def plot_pointcloud_examples(
    examples: Sequence[Mapping[str, Any]],
    map_arrays: Mapping[str, tuple[np.ndarray, np.ndarray]],
    path: Path,
    *,
    maximum_points_per_scene: int = 18_000,
) -> None:
    """Render deterministic top-down RGB point clouds with grounding overlays."""

    if not examples:
        raise ValueError("At least one grounding example is required")
    if maximum_points_per_scene < 100:
        raise ValueError("maximum_points_per_scene must be at least 100")
    columns = min(3, len(examples))
    rows = math.ceil(len(examples) / columns)
    figure = Figure(figsize=(6.1 * columns, 5.75 * rows + 1.35), dpi=150, facecolor="white")
    FigureCanvasAgg(figure)
    for panel, example in enumerate(examples, start=1):
        axis = figure.add_subplot(rows, columns, panel)
        scene_id = str(example["scene_id"])
        xyz, rgb = map_arrays[scene_id]
        # Suppress the dense floor/ceiling sheets so furniture and wall
        # structure remain legible in a top-down projection.  This is a visual
        # filter only; predictions and coordinate metrics still use every map
        # voxel and every scene token.
        visible = (xyz[:, 2] > 0.04) & (xyz[:, 2] < 2.90)
        visible_xyz = xyz[visible]
        visible_rgb = rgb[visible]
        stride = max(1, math.ceil(visible_xyz.shape[0] / maximum_points_per_scene))
        sampled_xyz = visible_xyz[::stride]
        sampled_rgb = np.clip(visible_rgb[::stride] * 0.68, 0.0, 1.0)
        axis.scatter(
            sampled_xyz[:, 0],
            sampled_xyz[:, 1],
            c=sampled_rgb,
            s=0.75,
            alpha=0.62,
            linewidths=0,
            rasterized=True,
        )
        predicted = np.asarray(example["predicted_xyz_m"], dtype=np.float64)
        target = np.asarray(example["target_xyz_m"], dtype=np.float64)
        axis.plot(
            [predicted[0], target[0]],
            [predicted[1], target[1]],
            color="#111827",
            linestyle="--",
            linewidth=1.2,
            zorder=4,
        )
        axis.scatter(
            [target[0]],
            [target[1]],
            marker="*",
            s=180,
            c="#22c55e",
            edgecolors="#052e16",
            linewidths=0.9,
            zorder=6,
        )
        axis.scatter(
            [predicted[0]],
            [predicted[1]],
            marker="X",
            s=115,
            c="#e11d48",
            edgecolors="white",
            linewidths=0.8,
            zorder=7,
        )
        question = textwrap.fill(str(example["question"]), width=43)
        axis.set_title(
            f"{scene_id} · {example['question_id']} · error {example['coordinate_error_m']:.3f} m\n"
            f"{question}",
            fontsize=8.9,
            pad=8,
        )
        axis.set_xlim(-3.15, 3.15)
        axis.set_ylim(-2.65, 2.65)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("world X (m)")
        axis.set_ylabel("world Y (m)")
        axis.grid(alpha=0.16, linewidth=0.55)
    for panel in range(len(examples) + 1, rows * columns + 1):
        axis = figure.add_subplot(rows, columns, panel)
        axis.axis("off")
    legend = [
        Line2D(
            [0],
            [0],
            marker="X",
            color="none",
            markerfacecolor="#e11d48",
            markeredgecolor="white",
            markersize=10,
            label="V78 continuous prediction",
        ),
        Line2D(
            [0],
            [0],
            marker="*",
            color="none",
            markerfacecolor="#22c55e",
            markeredgecolor="#052e16",
            markersize=13,
            label="oracle target (evaluation only)",
        ),
    ]
    figure.legend(handles=legend, loc="lower center", ncol=2, frameon=False, fontsize=10)
    figure.suptitle(
        "V78 grounding — deterministic held-scene RGB point-cloud overlays · "
        "markers show XY, errors are 3D · historical internal evaluation only",
        fontsize=14,
        y=0.975,
    )
    figure.subplots_adjust(
        left=0.055,
        right=0.985,
        top=0.88,
        bottom=0.085,
        hspace=0.48,
        wspace=0.18,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    figure.savefig(
        temporary,
        format="png",
        dpi=150,
        facecolor="white",
        metadata=PNG_METADATA,
    )
    os.replace(temporary, path)
    figure.clear()


def evaluate_and_plot(
    *,
    qa_path: str | Path = DEFAULT_QA,
    prefix_cache: str | Path = DEFAULT_PREFIX_CACHE,
    candidate_directory: str | Path = DEFAULT_CANDIDATE,
    sealed_report_path: str | Path = DEFAULT_SEALED_REPORT,
    maps_root: str | Path = DEFAULT_MAPS_ROOT,
    metrics_path: str | Path = DEFAULT_METRICS,
    figure_path: str | Path = DEFAULT_FIGURE,
    model_snapshot: str | Path | None = None,
) -> dict[str, Any]:
    """Reproduce the sealed held score and create its point-cloud evidence."""

    resolved_metrics = _resolve(metrics_path)
    resolved_figure = _resolve(figure_path)
    _assert_evaluation_output(resolved_metrics)
    _assert_evaluation_output(resolved_figure)
    sealed = load_sealed_report(sealed_report_path)
    records = load_historical_grounding_records(_resolve(qa_path))
    _train, held, split = pair_disjoint_internal_split(records)
    if split != sealed["split"]:
        raise ValueError("Reconstructed V78 historical split differs from the sealed report")
    qa_sha256 = _sha256_file(_resolve(qa_path))
    if qa_sha256 != sealed["training_source"]["sha256"]:
        raise ValueError("V78 historical QA source changed")

    candidate = validate_candidate(_resolve(candidate_directory))
    sealed_candidate = sealed["candidate"]
    if (
        candidate["weights_sha256"] != sealed_candidate["weights_sha256"]
        or candidate["metadata_sha256"] != sealed_candidate["metadata_sha256"]
    ):
        raise ValueError("V78 candidate differs from the sealed historical report")
    metadata = candidate["metadata"]
    model = GroundingSidecarV78(
        scene_dim=int(metadata["scene_dim"]),
        latent_count=int(metadata["scene_latent_count"]),
        rank=int(metadata["question_adapter_rank"]),
        hidden_dim=int(metadata["coordinate_hidden_dim"]),
        maximum_residual=float(metadata["maximum_residual"]),
    )
    state = load_file(str(_resolve(candidate_directory) / "grounding.safetensors"), device="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()

    prefix_store = QuestionIndependentPrefixStore(
        _resolve(prefix_cache),
        latent_count=model.latent_count,
        scene_dim=model.scene_dim,
    )
    if prefix_store.manifest_sha256 != sealed["continuous_scene_source"]["manifest_sha256"]:
        raise ValueError("V78 full-scene prefix manifest differs from the sealed report")
    snapshot = resolve_local_snapshot(DEFAULT_MODEL_ID, DEFAULT_MODEL_REVISION, model_snapshot)
    embeddings, embedding_audit = load_category_embeddings_selective(
        snapshot, [record.question for record in held]
    )
    room_min = torch.tensor(DEFAULT_ROOM_MIN, dtype=torch.float32)
    room_max = torch.tensor(DEFAULT_ROOM_MAX, dtype=torch.float32)
    predictions, attention = _predict(
        model,
        held,
        torch.from_numpy(embeddings),
        prefix_store,
        room_min,
        room_max,
    )
    aggregate = _prediction_metrics(held, predictions)
    metric_deltas = _assert_aggregate_reproduction(
        aggregate, sealed["metrics"]["historical_internal_held"]
    )

    resolved_maps = _resolve(maps_root)
    map_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    map_identities: dict[str, Any] = {}
    for scene_id in sorted({record.scene_id for record in held}):
        xyz, rgb, confidence, identity = _load_map_arrays(resolved_maps, scene_id)
        map_cache[scene_id] = (xyz, rgb, confidence)
        map_identities[scene_id] = identity

    targets = torch.tensor([record.target_xyz for record in held], dtype=torch.float32)
    errors = torch.linalg.vector_norm(predictions - targets, dim=-1)
    per_example: list[dict[str, Any]] = []
    for index, record in enumerate(held):
        prediction = predictions[index].detach().float().cpu().numpy()
        xyz, _rgb, confidence = map_cache[record.scene_id]
        support = _nearest_map_support(prediction, xyz, confidence)
        per_example.append(
            {
                "scene_id": record.scene_id,
                "question_id": record.question_id,
                "question": record.question,
                "predicted_xyz_m": [float(value) for value in prediction],
                "target_xyz_m": [float(value) for value in record.target_xyz],
                "coordinate_error_m": float(errors[index]),
                "map_support": support,
            }
        )
    selected_indices = select_visualization_indices(held)
    selected = [per_example[index] for index in selected_indices]
    plot_pointcloud_examples(
        selected,
        {
            scene_id: (arrays[0], arrays[1])
            for scene_id, arrays in map_cache.items()
        },
        resolved_figure,
    )
    support_distances = np.asarray(
        [row["map_support"]["distance_m"] for row in per_example], dtype=np.float64
    )
    report = {
        "artifact": "v78_historical_held_pointcloud_evaluation_v1",
        "schema_version": 1,
        "status": "internal_historical_evaluation_only",
        "scope": {
            "historical_internal_held_only": True,
            "official_validation_loaded": False,
            "official_test_loaded": False,
            "deferred_final_loaded": False,
            "runtime_promotion_authorized": False,
            "runtime_path_executed": False,
            "runtime_artifacts_written": [],
            "oracle_target_coordinates_loaded_by_evaluator": True,
            "oracle_or_qa_loaded_by_primary_runtime": False,
            "environmental_text_inputs_to_primary_runtime": [],
            "full_gemma_model_loaded": False,
            "selective_frozen_input_embedding_rows_only": True,
            "post_seal_optimization_or_tuning": False,
        },
        "architecture_audit": {
            "complete_question_independent_scene_prefix": True,
            "scene_latent_count": model.latent_count,
            "scene_dimension": model.scene_dim,
            "every_scene_token_scored": attention["every_scene_token_positive_weight"],
            "minimum_attention_weight": attention["minimum_attention_weight"],
            "maximum_attention_weight": attention["maximum_attention_weight"],
            "question_dependent_scene_retrieval": False,
            "top_k_scene_selection": False,
            "grounding_readout_is_question_conditioned": True,
            "strict_identical_total_environment_conditioned_input": False,
        },
        "sources": {
            "sealed_report": {
                "path": _resolve(sealed_report_path).relative_to(PROJECT_ROOT).as_posix(),
                "sha256": SEALED_REPORT_SHA256,
            },
            "historical_training_qa_evaluation_only": {
                "path": _resolve(qa_path).relative_to(PROJECT_ROOT).as_posix(),
                "sha256": qa_sha256,
            },
            "candidate_weights_sha256": candidate["weights_sha256"],
            "candidate_metadata_sha256": candidate["metadata_sha256"],
            "prefix_manifest_sha256": prefix_store.manifest_sha256,
            "numeric_rgb_maps": map_identities,
        },
        "split": split,
        "embedding_read": {
            "model_id": DEFAULT_MODEL_ID,
            "model_revision": DEFAULT_MODEL_REVISION,
            "loaded_parameter_keys": embedding_audit["loaded_parameter_keys"],
            "unique_token_rows_read": embedding_audit["unique_token_rows_read"],
            "full_language_model_loaded": False,
        },
        "reproduced_metrics": aggregate,
        "sealed_metric_absolute_deltas": metric_deltas,
        "sealed_metrics_reproduced_within_1e_7": True,
        "numeric_map_support": {
            "count": len(per_example),
            "mean_nearest_voxel_distance_m": float(support_distances.mean()),
            "maximum_nearest_voxel_distance_m": float(support_distances.max()),
            "within_0_25m_fraction": float(np.mean(support_distances <= 0.25)),
        },
        "visualization": {
            "path": resolved_figure.relative_to(PROJECT_ROOT).as_posix(),
            "sha256": _sha256_file(resolved_figure),
            "selection_rule": (
                "lexicographically smallest question_id in each held scene; "
                "selected before inspecting predictions or coordinate errors"
            ),
            "selected_count": len(selected),
            "selected_scene_question_ids": [
                [row["scene_id"], row["question_id"]] for row in selected
            ],
            "oracle_markers_evaluation_only": True,
            "sanitized_pointcloud_fields": ["centers_world", "mean_rgb", "confidence"],
        },
        "per_example": per_example,
        "limitations": [
            "This is a historical training-pool holdout, not official validation or test.",
            "The sidecar's readout is question-conditioned even though its complete scene prefix is fixed.",
            "The point-cloud overlay visualizes predicted coordinates; it is not evidence that Gemma generated a correct language answer.",
        ],
    }
    _atomic_json(resolved_metrics, report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa", default=str(DEFAULT_QA))
    parser.add_argument("--prefix-cache", default=str(DEFAULT_PREFIX_CACHE))
    parser.add_argument("--candidate", default=str(DEFAULT_CANDIDATE))
    parser.add_argument("--sealed-report", default=str(DEFAULT_SEALED_REPORT))
    parser.add_argument("--maps-root", default=str(DEFAULT_MAPS_ROOT))
    parser.add_argument("--metrics", default=str(DEFAULT_METRICS))
    parser.add_argument("--figure", default=str(DEFAULT_FIGURE))
    parser.add_argument("--model-snapshot")
    args = parser.parse_args(argv)
    report = evaluate_and_plot(
        qa_path=args.qa,
        prefix_cache=args.prefix_cache,
        candidate_directory=args.candidate,
        sealed_report_path=args.sealed_report,
        maps_root=args.maps_root,
        metrics_path=args.metrics,
        figure_path=args.figure,
        model_snapshot=args.model_snapshot,
    )
    summary = {
        "artifact": report["artifact"],
        "held_count": report["reproduced_metrics"]["count"],
        "mean_coordinate_error_m": report["reproduced_metrics"][
            "mean_coordinate_error_m"
        ],
        "within_1m_accuracy": report["reproduced_metrics"]["within_1m_accuracy"],
        "selected_pointcloud_examples": report["visualization"]["selected_count"],
        "metrics": _resolve(args.metrics).relative_to(PROJECT_ROOT).as_posix(),
        "figure": report["visualization"]["path"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_FIGURE",
    "DEFAULT_METRICS",
    "SEALED_REPORT_SHA256",
    "evaluate_and_plot",
    "load_sealed_report",
    "main",
    "plot_pointcloud_examples",
    "select_visualization_indices",
]
